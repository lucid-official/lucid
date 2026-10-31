"""(IA)^3: the closed forms, on CPU and Metal.

An output-side layer computes ``l * (W x + b)``, a feed-forward one
``W (l * x) + b``; several active adapters multiply; merging scales the
weight's rows (and the bias) or columns.  The expected values are written out
from those formulas with the vector held beside them — never read back
through the layer's own helpers — and the convolutions are checked against a
real convolution of the base's geometry.
"""

import types
import warnings

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.peft import IA3Config, IA3Model, PeftModel, get_peft_model
from lucid.peft.tuners import ia3
from lucid.peft.tuners._base import _tuner_layers
from lucid.peft.tuners.ia3._defaults import (
    IA3_DEFAULT_FEEDFORWARD_MODULES,
    IA3_DEFAULT_TARGET_MODULES,
)


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


def _config(**kwargs: object) -> IA3Config:
    fields: dict[str, object] = {"target_modules": ["x"], "feedforward_modules": []} | kwargs
    return IA3Config(**fields)  # type: ignore[arg-type]


def _layer(base: nn.Module, is_feedforward: bool = False, **kwargs: object) -> ia3.IA3Layer:
    layer_type = ia3.IA3Layer._layer_types[type(base)]
    layer = layer_type(base, "default", _config(init_ia3_weights=False, **kwargs), is_feedforward)
    assert isinstance(layer, ia3.IA3Layer)
    return layer


def _vector(layer: ia3.IA3Layer, adapter: str = "default") -> lucid.Tensor:
    vector = layer.ia3_l[adapter]
    assert vector is not None
    return vector


class TestLinear:
    def test_the_vector_has_the_reference_shape(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        assert tuple(_vector(_layer(base)).shape) == (5, 1)
        assert tuple(_vector(_layer(base, is_feedforward=True)).shape) == (1, 8)
        assert _vector(_layer(base)).device == base.weight.device

    def test_the_default_start_is_ones_and_the_base_bit_for_bit(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = ia3.Linear(base, "default", _config())
        assert _same(_vector(layer), lucid.ones(5, 1, device=device))
        x = lucid.randn(3, 8, device=device)
        assert _same(layer(x), base(x))

    def test_an_output_side_layer_scales_the_output_bias_included(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = _layer(base)
        x = lucid.randn(3, 8, device=device)
        v = _vector(layer).flatten()
        expected = (x @ base.weight.mT + base.bias) * v
        assert _close(layer(x), expected)

    def test_a_feedforward_layer_scales_the_input(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = _layer(base, is_feedforward=True)
        x = lucid.randn(3, 8, device=device)
        v = _vector(layer).flatten()
        expected = (x * v) @ base.weight.mT + base.bias
        assert _close(layer(x), expected)

    def test_several_active_adapters_multiply(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = _layer(base)
        layer.update_layer("other", _config(init_ia3_weights=False))
        layer.set_adapter(["default", "other"])
        x = lucid.randn(3, 8, device=device)
        v = _vector(layer).flatten() * _vector(layer, "other").flatten()
        assert _close(layer(x), (x @ base.weight.mT + base.bias) * v)

    def test_an_adapter_the_layer_lacks_is_skipped(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = _layer(base)
        layer.set_adapter(["absent"])
        x = lucid.randn(3, 8, device=device)
        assert _same(layer(x), base(x))

    def test_merging_scales_rows_and_the_bias(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = _layer(base)
        weight, bias = base.weight.clone(), base.bias.clone()
        v = _vector(layer).clone()
        x = lucid.randn(3, 8, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert _close(base.weight, weight * v) and _close(base.bias, bias * v.flatten())
            assert _close(layer(x), adapted)
            with pytest.warns(UserWarning, match="inaccurate"):
                layer.unmerge()
            assert _close(base.weight, weight) and _close(base.bias, bias)

    def test_merging_a_feedforward_layer_scales_columns_and_not_the_bias(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = _layer(base, is_feedforward=True)
        weight, bias = base.weight.clone(), base.bias.clone()
        v = _vector(layer).clone()
        x = lucid.randn(3, 8, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert _close(base.weight, weight * v) and _same(base.bias, bias)
            assert _close(layer(x), adapted)

    def test_two_merged_adapters_multiply_and_unmerge_in_either_order(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = _layer(base)
        layer.update_layer("other", _config(init_ia3_weights=False))
        weight = base.weight.clone()
        with lucid.no_grad():
            layer.merge(adapter_names=["default", "other"])
            product = _vector(layer) * _vector(layer, "other")
            assert _close(base.weight, weight * product, atol=1e-5)
            with pytest.warns(UserWarning):
                layer.unmerge()
            assert _close(base.weight, weight, atol=1e-5)

    def test_a_bias_free_layer_merges(self, device: str) -> None:
        base = nn.Linear(8, 5, bias=False).to(device)
        layer = _layer(base)
        weight = base.weight.clone()
        with lucid.no_grad():
            layer.merge()
        assert _close(base.weight, weight * _vector(layer))
        assert list(layer._merge_targets()) == ["base_layer.weight"]

    def test_the_delta_weight_is_what_merging_adds(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = _layer(base)
        weight = base.weight.clone()
        delta = layer.get_delta_weight("default")
        assert _close(delta, weight * _vector(layer) - weight)

    def test_the_vector_gets_a_gradient_through_a_frozen_base(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = _layer(base)
        x = lucid.randn(3, 8, device=device)
        layer(x).sum().backward()
        v = _vector(layer)
        assert v.grad is not None
        # d/dv_o sum_n (x W^T + b)_{n,o} * v_o = sum_n (x W^T + b)_{n,o}
        expected = (x @ base.weight.mT + base.bias).sum(dim=0).reshape(5, 1)
        assert _close(v.grad, expected, atol=1e-4)

    def test_it_refuses_what_it_cannot_do(self) -> None:
        with pytest.raises(ValueError, match="fan_in_fan_out"):
            ia3.Linear(nn.Linear(4, 4), "default", _config(fan_in_fan_out=True))
        with pytest.raises(TypeError, match="adapts nn.Linear"):
            ia3.Linear(nn.Conv2d(2, 2, 1), "default", _config())  # type: ignore[arg-type]
        layer = _layer(nn.Linear(4, 4))
        with pytest.raises(ValueError, match="already exists"):
            layer.update_layer("default", _config())
        layer.merge()
        with pytest.warns(UserWarning, match="Already merged"):
            layer.merge()

    def test_the_repr_and_features(self) -> None:
        layer = _layer(nn.Linear(8, 5), is_feedforward=True)
        assert repr(layer).startswith("ia3.Linear(")
        assert "is_feedforward=True" in repr(layer)
        assert (layer.in_features, layer.out_features) == (8, 5)


# ── convolutions ─────────────────────────────────────────────────────────────


def _conv2d(device: str, groups: int = 1, bias: bool = True) -> nn.Conv2d:
    lucid.manual_seed(0)
    return nn.Conv2d(4, 6, 3, padding=1, groups=groups, bias=bias).to(device)


class TestConvolutions:
    def test_the_vector_is_over_the_channels(self, device: str) -> None:
        base = _conv2d(device)
        assert tuple(_vector(_layer(base)).shape) == (1, 6, 1, 1)
        assert tuple(_vector(_layer(base, is_feedforward=True)).shape) == (1, 4, 1, 1)
        volume = nn.Conv3d(2, 3, 2).to(device)
        assert tuple(_vector(_layer(volume)).shape) == (1, 3, 1, 1, 1)

    def test_an_output_side_convolution_scales_its_output_channels(self, device: str) -> None:
        base = _conv2d(device)
        layer = _layer(base)
        x = lucid.randn(2, 4, 5, 5, device=device)
        assert _close(layer(x), base(x) * _vector(layer), atol=1e-5)

    @pytest.mark.parametrize("groups", [1, 2])
    def test_a_feedforward_convolution_scales_its_input_channels(
        self, device: str, groups: int
    ) -> None:
        base = _conv2d(device, groups)
        layer = _layer(base, is_feedforward=True)
        x = lucid.randn(2, 4, 5, 5, device=device)
        assert _close(layer(x), base(x * _vector(layer)), atol=1e-5)

    @pytest.mark.parametrize("feedforward", [False, True])
    @pytest.mark.parametrize("groups", [1, 2])
    def test_merging_a_convolution_gives_the_same_output(
        self, device: str, feedforward: bool, groups: int
    ) -> None:
        base = _conv2d(device, groups)
        layer = _layer(base, is_feedforward=feedforward)
        weight = base.weight.clone()
        x = lucid.randn(2, 4, 5, 5, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert layer.merged and not _close(base.weight, weight, atol=1e-3)
            assert _close(layer(x), adapted, atol=1e-5)
            # The merged conv alone equals the adapted layer: the base run on its own weights.
            assert _close(F.conv2d(x, base.weight, base.bias, 1, 1, 1, groups), adapted, atol=1e-5)
            with pytest.warns(UserWarning):
                layer.unmerge()
            assert _close(base.weight, weight, atol=1e-5)

    def test_a_bias_free_convolution_and_conv3d(self, device: str) -> None:
        base = _conv2d(device, bias=False)
        layer = _layer(base)
        x = lucid.randn(2, 4, 5, 5, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert _close(layer(x), adapted, atol=1e-5)
        volume = nn.Conv3d(2, 3, 2).to(device)
        layer3 = _layer(volume)
        y = lucid.randn(1, 2, 3, 3, 3, device=device)
        assert _close(layer3(y), volume(y) * _vector(layer3), atol=1e-5)
        assert isinstance(layer3, ia3.Conv3d) and repr(layer3).startswith("ia3.Conv3d(")


# ── the config, the defaults and the tuner ───────────────────────────────────


class TestConfig:
    def test_feedforward_modules_must_be_a_subset(self) -> None:
        with pytest.raises(ValueError, match="subset"):
            IA3Config(target_modules=["k"], feedforward_modules=["wo"])
        # A regex on either side cannot be compared, so it is accepted.
        IA3Config(target_modules="k.*", feedforward_modules=["wo"])
        with pytest.raises(ValueError, match="regular expression"):
            IA3Config(target_modules=["k"], feedforward_modules="(")

    def test_the_fields_are_checked(self) -> None:
        with pytest.raises(TypeError, match="fan_in_fan_out"):
            IA3Config(fan_in_fan_out="yes")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="init_ia3_weights"):
            IA3Config(init_ia3_weights=1)  # type: ignore[arg-type]

    def test_lists_become_sets_and_the_file_round_trips(self) -> None:
        config = IA3Config(target_modules=["v", "k"], feedforward_modules=["v"])
        assert config.target_modules == {"k", "v"} and config.feedforward_modules == {"v"}
        data = config.to_dict()
        assert data["target_modules"] == ["k", "v"] and data["peft_type"] == "IA3"
        assert IA3Config.from_dict(data) == config

    def test_the_reference_s_file_fields_are_read(self) -> None:
        config = IA3Config.from_dict(
            {
                "peft_type": "IA3",
                "target_modules": ["k", "v", "wo"],
                "feedforward_modules": ["wo"],
                "fan_in_fan_out": False,
                "init_ia3_weights": True,
                "modules_to_save": None,
                "exclude_modules": None,
                "peft_version": "0.21.2",
            }
        )
        assert config.feedforward_modules == {"wo"} and config.init_ia3_weights is True


class _Model(nn.Module):
    """A BERT-shaped toy with a ``model_type``: attention ``key`` / ``value`` and ``output.dense``."""

    def __init__(self, model_type: object = "bert") -> None:
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.attention = nn.ModuleDict({"key": nn.Linear(8, 8), "value": nn.Linear(8, 8)})
        self.output = nn.ModuleDict({"dense": nn.Linear(8, 8)})

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.output["dense"](self.attention["key"](x) + self.attention["value"](x))


class TestDefaults:
    def test_the_tables_cover_the_zoo_types_the_reference_has(self) -> None:
        assert set(IA3_DEFAULT_TARGET_MODULES) == set(IA3_DEFAULT_FEEDFORWARD_MODULES)
        assert set(IA3_DEFAULT_TARGET_MODULES) == {"gpt2", "bert", "t5"}
        for model_type, feedforward in IA3_DEFAULT_FEEDFORWARD_MODULES.items():
            assert feedforward <= IA3_DEFAULT_TARGET_MODULES[model_type]

    def test_no_names_takes_the_defaults_for_the_model_type(self) -> None:
        net = _Model()
        tuner = IA3Model(net, IA3Config())
        config = tuner.peft_config["default"]
        assert isinstance(config, IA3Config)
        assert config.target_modules == {"key", "value", "output.dense"}
        assert config.feedforward_modules == {"output.dense"}
        assert tuner.targeted_module_names == ["attention.key", "attention.value", "output.dense"]
        layers = dict(_tuner_layers(net))
        assert not layers["attention.key"].is_feedforward  # type: ignore[attr-defined]
        assert layers["output.dense"].is_feedforward  # type: ignore[attr-defined]

    def test_given_names_are_kept_and_feedforward_still_comes_from_the_table(self) -> None:
        tuner = IA3Model(_Model(), IA3Config(target_modules=["key", "output.dense"]))
        config = tuner.peft_config["default"]
        assert isinstance(config, IA3Config) and config.feedforward_modules == {"output.dense"}

    def test_a_model_type_with_no_row_is_told_what_to_give(self) -> None:
        with pytest.raises(ValueError, match="Please specify `target_modules`"):
            IA3Model(_Model("custom"), IA3Config())
        with pytest.raises(ValueError, match="Please specify `feedforward_modules`"):
            IA3Model(_Model("custom"), IA3Config(target_modules=["key"]))
        IA3Model(_Model("custom"), IA3Config(target_modules=["key"], feedforward_modules=[]))

    def test_the_config_passed_in_is_not_changed(self) -> None:
        config = IA3Config()
        IA3Model(_Model(), config)
        assert config.target_modules is None and config.feedforward_modules is None

    def test_feedforward_is_matched_as_a_regex_over_the_whole_name(self) -> None:
        net = _Model()
        config = IA3Config(target_modules=["key", "dense"], feedforward_modules="output\\..*")
        IA3Model(net, config)
        layers = dict(_tuner_layers(net))
        assert layers["output.dense"].is_feedforward  # type: ignore[attr-defined]
        assert not layers["attention.key"].is_feedforward  # type: ignore[attr-defined]
        net2 = _Model()
        # A regex must match the whole name, not a tail of it.
        IA3Model(net2, IA3Config(target_modules=["key"], feedforward_modules="key"))
        assert not dict(_tuner_layers(net2))["attention.key"].is_feedforward  # type: ignore[attr-defined]

    def test_fan_in_fan_out_is_reset_with_a_warning(self) -> None:
        with pytest.warns(UserWarning, match="fan_in_fan_out is set to True"):
            tuner = IA3Model(_Model(), IA3Config(fan_in_fan_out=True))
        config = tuner.peft_config["default"]
        assert isinstance(config, IA3Config) and config.fan_in_fan_out is False

    def test_an_unsupported_module_is_named(self) -> None:
        net = nn.Sequential(nn.Embedding(5, 4))
        with pytest.raises(ValueError, match=r"Target module 0 \(Embedding\) is not supported"):
            IA3Model(net, IA3Config(target_modules=["0"], feedforward_modules=[]))


class TestTuner:
    def _tuner(self) -> tuple[IA3Model, _Model]:
        net = _Model()
        return IA3Model(net, IA3Config(init_ia3_weights=False), adapter_name="a"), net

    def test_an_adapter_cannot_flip_a_layer_between_input_and_output(self) -> None:
        tuner, net = self._tuner()
        before = sorted(net.state_dict())
        with pytest.raises(ValueError, match="scales either its input or its output"):
            tuner.add_adapter("b", IA3Config(target_modules=["dense"], feedforward_modules=[]))
        assert sorted(net.state_dict()) == before and sorted(tuner.peft_config) == ["a"]

    def test_the_weighted_adapter_is_the_weighted_sum_of_the_vectors(self) -> None:
        tuner, net = self._tuner()
        tuner.add_adapter("b", IA3Config(target_modules=["key"], feedforward_modules=[]))
        with lucid.no_grad():
            layer_b = dict(_tuner_layers(net))["attention.key"]
            layer_b.ia3_l["b"].copy_(lucid.full((8, 1), 3.0))  # type: ignore[attr-defined]
        tuner.add_weighted_adapter(["a", "b"], [0.25, 2.0], "mix")
        layers = dict(_tuner_layers(net))
        key, value, dense = (
            layers[n] for n in ("attention.key", "attention.value", "output.dense")
        )
        a_key = key.ia3_l["a"]  # type: ignore[attr-defined]
        assert _close(key.ia3_l["mix"], a_key * 0.25 + 3.0 * 2.0)  # type: ignore[attr-defined]
        # A layer only "a" reaches sums the sources it has.
        assert _close(value.ia3_l["mix"], value.ia3_l["a"] * 0.25)  # type: ignore[attr-defined]
        assert tuple(dense.ia3_l["mix"].shape) == (1, 8)  # type: ignore[attr-defined]
        assert tuner.active_adapters == ["a"]
        assert not any(p.requires_grad for p in key._adapter_parameters("mix"))  # type: ignore[attr-defined]
        tuner.set_adapter("mix")
        assert all(p.requires_grad for p in key._adapter_parameters("mix"))  # type: ignore[attr-defined]

    def test_the_weighted_adapter_through_the_wrapper_forwards(self) -> None:
        peft = get_peft_model(_Model(), IA3Config(init_ia3_weights=False), adapter_name="a")
        peft.add_adapter("b", IA3Config(init_ia3_weights=False))
        peft.add_weighted_adapter(["a", "b"], [0.5, 0.5], "mix")
        assert sorted(peft.peft_config) == ["a", "b", "mix"]
        assert isinstance(peft, PeftModel)

    def test_the_weighted_adapter_refuses_bad_arguments_and_leaves_no_trace(self) -> None:
        tuner, net = self._tuner()
        before = sorted(net.state_dict())
        with pytest.raises(ValueError, match="already exists"):
            tuner.add_weighted_adapter(["a"], [1.0], "a")
        with pytest.raises(ValueError, match="does not exist"):
            tuner.add_weighted_adapter(["a", "ghost"], [1.0, 1.0], "mix")
        with pytest.raises(ValueError, match="one weight per adapter"):
            tuner.add_weighted_adapter(["a"], [1.0, 2.0], "mix")
        with pytest.raises(ValueError, match="no adapter matching"):
            tuner.add_weighted_adapter([], [], "mix")
        assert sorted(net.state_dict()) == before and sorted(tuner.peft_config) == ["a"]

    def test_two_sources_training_one_module_cannot_be_combined(self) -> None:
        net = _Model()
        tuner = IA3Model(
            net, IA3Config(init_ia3_weights=False, modules_to_save=["output.dense"]), "a"
        )
        tuner.add_adapter("b", IA3Config(modules_to_save=["output.dense"]))
        with pytest.raises(ValueError, match="modules_to_save"):
            tuner.add_weighted_adapter(["a", "b"], [1.0, 1.0], "mix")

    def test_a_half_filled_weighted_adapter_is_withdrawn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tuner, net = self._tuner()
        tuner.add_adapter("b", IA3Config(init_ia3_weights=False))

        def broken(*args: object, **kwargs: object) -> object:
            raise KeyboardInterrupt

        monkeypatch.setattr(lucid, "zeros_like", broken)
        with pytest.raises(KeyboardInterrupt):
            tuner.add_weighted_adapter(["a", "b"], [1.0, 1.0], "mix")
        assert "mix" not in tuner.peft_config
        assert not any("mix" in k for k in net.state_dict())

    def test_unloading_gives_back_the_modules(self) -> None:
        tuner, net = self._tuner()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tuner.merge_adapter()
        assert tuner.unload() is net
        assert type(net.attention["key"]) is nn.Linear
