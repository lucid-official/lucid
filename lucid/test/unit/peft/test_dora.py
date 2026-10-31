"""DoRA: the weight-decomposed update, in closed form, on CPU and Metal.

``y = f(x) + (g - 1) (f(x) - b) + g s B A x`` with ``g = m / ||W + s B A||``
per output unit, the norm a constant for the gradient: the layer computes
``g (W + s B A) x + b``.  The tests write that formula out and compare the
layer with it, then check what follows from it: merging is exact, grads reach
the magnitude and the factors only, the norm carries none, and several
adapters (DoRA or not) add up the way the merged weight says.
"""

from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.peft import LoraConfig, PeftModel, get_peft_model
from lucid.peft.tuners.lora import (
    Conv1d,
    Conv2d,
    Embedding,
    Linear,
    LoraLayer,
    MultiheadAttention,
    ParamWrapper,
)
from lucid.test._fixtures.devices import metal_available


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol, rtol=1e-4))


def _linear(device: str, **kwargs: object) -> tuple[nn.Linear, Linear]:
    """A DoRA layer whose factors and magnitude are all non-trivial."""
    lucid.manual_seed(0)
    base = nn.Linear(6, 5).to(device)
    options: dict[str, object] = {
        "r": 3,
        "lora_alpha": 6,
        "use_dora": True,
        "init_lora_weights": False,
    } | kwargs
    layer = Linear(base, "default", **options)  # type: ignore[arg-type]
    _perturb(layer, "default")
    return base, layer


def _perturb(layer: LoraLayer, adapter: str) -> None:
    """Move the magnitude off its initial value so ``g != 1``."""
    with lucid.no_grad():
        magnitude = layer.lora_magnitude_vector[adapter].weight  # type: ignore[attr-defined]
        magnitude.copy_(
            magnitude * (1.0 + 0.5 * lucid.randn(*magnitude.shape, device=magnitude.device))
        )


def _rows_norm(w: lucid.Tensor) -> lucid.Tensor:
    flat = w.reshape(w.shape[0], -1)
    return (flat * flat).sum(dim=1).sqrt()


class TestStructure:
    def test_the_state_dict_keys_are_the_reference_s(self, device: str) -> None:
        _, layer = _linear(device)
        assert sorted(layer.state_dict()) == [
            "base_layer.bias",
            "base_layer.weight",
            "lora_A.default.weight",
            "lora_B.default.weight",
            "lora_magnitude_vector.default.weight",
        ]
        assert tuple(layer.lora_magnitude_vector["default"].weight.shape) == (5,)

    def test_the_magnitude_starts_at_the_row_norms_of_the_weight(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(6, 5).to(device)
        layer = Linear(base, "default", r=3, lora_alpha=6, use_dora=True)
        assert _close(layer.lora_magnitude_vector["default"].weight, _rows_norm(base.weight))

    def test_a_fresh_dora_layer_is_the_base(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(6, 5).to(device)
        layer = Linear(base, "default", r=3, lora_alpha=6, use_dora=True)
        x = lucid.randn(4, 6, device=device)
        assert _close(layer(x), base(x))

    def test_the_flags_are_read_back_per_adapter(self, device: str) -> None:
        _, layer = _linear(device)
        assert layer.use_dora == {"default": True} and layer.lora_bias == {"default": False}
        assert layer._specs["default"].variant == "dora"

    def test_the_magnitude_is_a_trainable_parameter_of_the_adapter(self, device: str) -> None:
        _, layer = _linear(device)
        magnitude = layer.lora_magnitude_vector["default"].weight
        assert isinstance(magnitude, nn.Parameter) and magnitude.requires_grad
        assert magnitude.device == layer.weight.device

    def test_plain_layers_keep_an_empty_table(self, device: str) -> None:
        lucid.manual_seed(0)
        layer = Linear(nn.Linear(6, 5).to(device), "default", r=2)
        assert len(layer.lora_magnitude_vector) == 0
        assert "lora_magnitude_vector" not in " ".join(layer.state_dict())

    def test_deleting_the_adapter_deletes_its_magnitude(self, device: str) -> None:
        _, layer = _linear(device)
        layer.update_layer("other", 2, 4.0, 0.0, True, False, True)
        layer.delete_adapter("default")
        assert list(layer.lora_magnitude_vector) == ["other"]
        assert layer.use_dora == {"other": True}


class TestForward:
    def test_linear_is_the_decomposed_weight(self, device: str) -> None:
        base, layer = _linear(device)
        x = lucid.randn(4, 6, device=device)
        w = base.weight + layer.get_delta_weight("default")
        g = layer.lora_magnitude_vector["default"].weight / _rows_norm(w)
        expected = F.linear(x, g.reshape(-1, 1) * w, base.bias)
        assert _close(layer(x), expected)

    def test_the_formula_in_its_additive_form(self, device: str) -> None:
        base, layer = _linear(device)
        x = lucid.randn(4, 6, device=device)
        w = base.weight + layer.get_delta_weight("default")
        g = layer.lora_magnitude_vector["default"].weight / _rows_norm(w)
        no_bias = F.linear(x, base.weight)
        expected = base(x) + (g - 1) * no_bias + g * F.linear(x, layer.get_delta_weight("default"))
        assert _close(layer(x), expected)

    def test_three_dimensional_input(self, device: str) -> None:
        base, layer = _linear(device)
        x = lucid.randn(2, 3, 6, device=device)
        w = base.weight + layer.get_delta_weight("default")
        g = layer.lora_magnitude_vector["default"].weight / _rows_norm(w)
        assert _close(layer(x), F.linear(x, g.reshape(-1, 1) * w, base.bias))

    def test_a_base_without_bias(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(6, 5, bias=False).to(device)
        layer = Linear(base, "default", r=3, lora_alpha=6, use_dora=True, init_lora_weights=False)
        _perturb(layer, "default")
        x = lucid.randn(4, 6, device=device)
        w = base.weight + layer.get_delta_weight("default")
        g = layer.lora_magnitude_vector["default"].weight / _rows_norm(w)
        assert _close(layer(x), F.linear(x, g.reshape(-1, 1) * w))

    def test_dropout_reaches_the_base_term_in_training(self, device: str) -> None:
        base, layer = _linear(device, lora_dropout=0.5)
        layer.train()
        x = lucid.randn(8, 6, device=device)
        w = base.weight + layer.get_delta_weight("default")
        g = layer.lora_magnitude_vector["default"].weight / _rows_norm(w)
        lucid.manual_seed(7)
        got = layer(x)
        lucid.manual_seed(7)
        dropped = layer.lora_dropout["default"](x)
        expected = (
            base(x)
            + (g - 1) * F.linear(dropped, base.weight)
            + g * F.linear(dropped, layer.get_delta_weight("default"))
        )
        assert _close(got, expected)

    def test_dropout_is_off_in_eval(self, device: str) -> None:
        base, layer = _linear(device, lora_dropout=0.5)
        layer.eval()
        x = lucid.randn(4, 6, device=device)
        w = base.weight + layer.get_delta_weight("default")
        g = layer.lora_magnitude_vector["default"].weight / _rows_norm(w)
        assert _close(layer(x), F.linear(x, g.reshape(-1, 1) * w, base.bias))

    def test_disabled_runs_the_base(self, device: str) -> None:
        base, layer = _linear(device)
        layer.enable_adapters(False)
        x = lucid.randn(4, 6, device=device)
        assert _same(layer(x), base(x))

    def test_a_half_precision_base_gets_a_float32_magnitude(self) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(6, 5).bfloat16()
        layer = Linear(base, "default", r=2, lora_alpha=4, use_dora=True)
        assert layer.lora_magnitude_vector["default"].weight.dtype == lucid.float32
        x = lucid.randn(3, 6).bfloat16()
        assert layer(x).dtype == lucid.bfloat16


class TestGradients:
    def test_grads_reach_the_magnitude_and_the_factors_only(self, device: str) -> None:
        base, layer = _linear(device)
        base.weight.requires_grad_(False)
        base.bias.requires_grad_(False)
        x = lucid.randn(4, 6, device=device)
        layer(x).sum().backward()
        assert layer.lora_magnitude_vector["default"].weight.grad is not None
        assert layer.lora_A["default"].weight.grad is not None
        assert layer.lora_B["default"].weight.grad is not None
        assert base.weight.grad is None and base.bias.grad is None

    def test_the_norm_is_a_constant_for_the_gradient(self, device: str) -> None:
        base, layer = _linear(device)
        x = lucid.randn(4, 6, device=device)
        layer(x).sum().backward()
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        m = layer.lora_magnitude_vector["default"].weight
        got = (a.grad.clone(), b.grad.clone(), m.grad.clone())  # type: ignore[union-attr]
        for p in (a, b, m):
            p.grad = None
        # The same output with the norm computed on detached copies of the factors.
        scaling = layer.scaling["default"]
        with lucid.no_grad():
            norm = _rows_norm(base.weight + (b @ a) * scaling)
        w = base.weight + (b @ a) * scaling
        expected = F.linear(x, (m / norm).reshape(-1, 1) * w, base.bias)
        expected.sum().backward()
        for have, want in zip(got, (a.grad, b.grad, m.grad), strict=True):
            assert _close(have, want, atol=1e-4)  # type: ignore[arg-type]


class TestMerge:
    def test_merging_changes_nothing_in_the_output(self, device: str) -> None:
        base, layer = _linear(device)
        x = lucid.randn(4, 6, device=device)
        before = layer(x)
        layer.merge()
        assert layer.merged and _close(layer(x), before)

    def test_the_merged_weight_is_the_decomposed_one(self, device: str) -> None:
        base, layer = _linear(device)
        w = base.weight.detach().clone() + layer.get_delta_weight("default").detach()
        g = layer.lora_magnitude_vector["default"].weight.detach() / _rows_norm(w)
        layer.merge()
        assert _close(base.weight, g.reshape(-1, 1) * w)

    def test_unmerge_restores_the_weight(self, device: str) -> None:
        base, layer = _linear(device)
        original = base.weight.detach().clone()
        layer.merge()
        assert not _close(base.weight, original)
        layer.unmerge()
        assert not layer.merged and _close(base.weight, original, atol=1e-5)
        assert layer.lora_magnitude_vector["default"].merged_gain is None

    def test_safe_merge_matches(self, device: str) -> None:
        base, layer = _linear(device)
        _, other = _linear(device)
        layer.merge(safe_merge=True)
        other.merge()
        assert _close(base.weight, other.base_layer.weight)

    def test_the_weight_keeps_its_identity(self, device: str) -> None:
        base, layer = _linear(device)
        weight = base.weight
        layer.merge()
        assert base.weight is weight and layer.weight is weight

    def test_a_non_finite_magnitude_fails_the_safe_merge_and_changes_nothing(
        self, device: str
    ) -> None:
        base, layer = _linear(device)
        original = base.weight.detach().clone()
        with lucid.no_grad():
            layer.lora_magnitude_vector["default"].weight.copy_(
                layer.lora_magnitude_vector["default"].weight * float("inf")
            )
        with pytest.raises(ValueError, match="NaNs detected"):
            layer.merge(safe_merge=True)
        assert _same(base.weight, original) and not layer.merged

    def test_two_dora_adapters_merge_into_what_the_forward_adds(self, device: str) -> None:
        base, layer = _linear(device)
        layer.update_layer("second", 2, 4.0, 0.0, False, False, True)
        _perturb(layer, "second")
        layer.set_adapter(["default", "second"])
        x = lucid.randn(4, 6, device=device)
        before = layer(x)
        original = base.weight.detach().clone()
        layer.merge()
        assert layer.merged_adapters == ["default", "second"]
        assert _close(layer(x), before, atol=1e-4)
        layer.unmerge()
        assert _close(base.weight, original, atol=1e-5)
        assert _close(layer(x), before, atol=1e-4)

    def test_a_plain_and_a_dora_adapter_together(self, device: str) -> None:
        base, layer = _linear(device)
        layer.update_layer("plain", 2, 4.0, 0.0, False, False, False)
        layer.set_adapter(["plain", "default"])
        x = lucid.randn(4, 6, device=device)
        before = layer(x)
        original = base.weight.detach().clone()
        layer.merge()
        assert _close(layer(x), before, atol=1e-4)
        layer.unmerge()
        assert _close(base.weight, original, atol=1e-5)

    def test_merging_one_after_the_other_equals_merging_together(self, device: str) -> None:
        base, layer = _linear(device)
        layer.update_layer("second", 2, 4.0, 0.0, False, False, True)
        _perturb(layer, "second")
        layer.set_adapter(["default", "second"])
        x = lucid.randn(4, 6, device=device)
        before = layer(x)
        layer.merge(adapter_names=["default"])
        layer.merge(adapter_names=["second"])
        assert _close(layer(x), before, atol=1e-4)
        assert _close(F.linear(x, base.weight, base.bias), before, atol=1e-4)


class TestEmbedding:
    def _layer(self, device: str) -> tuple[nn.Embedding, Embedding]:
        lucid.manual_seed(0)
        base = nn.Embedding(9, 4).to(device)
        layer = Embedding(
            base, "default", r=2, lora_alpha=4, use_dora=True, init_lora_weights=False
        )
        _perturb(layer, "default")
        return base, layer

    def test_the_magnitude_is_per_embedding_dimension(self, device: str) -> None:
        base, layer = self._layer(device)
        assert tuple(layer.lora_magnitude_vector["default"].weight.shape) == (4,)
        assert sorted(layer.state_dict()) == [
            "base_layer.weight",
            "lora_embedding_A.default",
            "lora_embedding_B.default",
            "lora_magnitude_vector.default.weight",
        ]

    def test_a_fresh_layer_is_the_base(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Embedding(9, 4).to(device)
        layer = Embedding(base, "default", r=2, lora_alpha=4, use_dora=True)
        ids = lucid.tensor([[1, 3], [7, 0]], device=device)
        assert _close(layer(ids), base(ids))

    def test_the_forward_scales_each_dimension_of_the_adapted_table(self, device: str) -> None:
        base, layer = self._layer(device)
        ids = lucid.tensor([[1, 3], [7, 0]], device=device)
        table = base.weight + layer.get_delta_weight("default")  # (vocab, dim)
        g = layer.lora_magnitude_vector["default"].weight / _rows_norm(table.mT)
        assert _close(layer(ids), g * F.embedding(ids, table))

    def test_merge_and_unmerge(self, device: str) -> None:
        base, layer = self._layer(device)
        ids = lucid.tensor([[1, 3], [7, 0]], device=device)
        before = layer(ids)
        original = base.weight.detach().clone()
        layer.merge()
        assert _close(layer(ids), before)
        layer.unmerge()
        assert _close(base.weight, original, atol=1e-5)

    def test_grads_reach_the_magnitude(self, device: str) -> None:
        _, layer = self._layer(device)
        ids = lucid.tensor([[1, 3], [7, 0]], device=device)
        layer(ids).sum().backward()
        assert layer.lora_magnitude_vector["default"].weight.grad is not None


class TestConv:
    def _layer(self, device: str, cls: type[LoraLayer], base: nn.Module) -> LoraLayer:
        layer = cls(base, "default", r=2, lora_alpha=4, use_dora=True, init_lora_weights=False)
        _perturb(layer, "default")
        return layer

    def test_conv2d_is_the_decomposed_kernel(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Conv2d(3, 5, kernel_size=3, padding=1).to(device)
        layer = self._layer(device, Conv2d, base)
        x = lucid.randn(2, 3, 6, 6, device=device)
        w = base.weight + layer.get_delta_weight("default")
        g = layer.lora_magnitude_vector["default"].weight / _rows_norm(w)
        expected = F.conv2d(x, g.reshape(-1, 1, 1, 1) * w, base.bias, padding=1)
        assert _close(layer(x), expected, atol=1e-4)

    def test_conv1d_merge_round_trip(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Conv1d(3, 4, kernel_size=3).to(device)
        layer = self._layer(device, Conv1d, base)
        x = lucid.randn(2, 3, 8, device=device)
        before = layer(x)
        original = base.weight.detach().clone()
        layer.merge()
        assert _close(layer(x), before, atol=1e-4)
        layer.unmerge()
        assert _close(base.weight, original, atol=1e-5)

    def test_a_base_without_bias(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False).to(device)
        layer = self._layer(device, Conv2d, base)
        x = lucid.randn(2, 3, 6, 6, device=device)
        w = base.weight + layer.get_delta_weight("default")
        g = layer.lora_magnitude_vector["default"].weight / _rows_norm(w)
        assert _close(layer(x), F.conv2d(x, g.reshape(-1, 1, 1, 1) * w, padding=1), atol=1e-4)

    def test_dropout_in_training(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Conv2d(3, 5, kernel_size=3, padding=1).to(device)
        layer = Conv2d(
            base, "default", r=2, lora_alpha=4, lora_dropout=0.5, use_dora=True,
            init_lora_weights=False,
        )  # fmt: skip
        _perturb(layer, "default")
        layer.train()
        x = lucid.randn(2, 3, 6, 6, device=device)
        assert tuple(layer(x).shape) == (2, 5, 6, 6)

    def test_a_grouped_convolution_is_refused_before_anything_is_built(self, device: str) -> None:
        base = nn.Conv2d(4, 4, kernel_size=3, groups=2).to(device)
        with pytest.warns(UserWarning, match="groups > 1"):
            plain = Conv2d(base, "default", r=2, lora_alpha=4)
        with pytest.raises(NotImplementedError, match="DoRA"):
            plain.update_layer("dora", 2, 4.0, 0.0, True, False, True)
        assert list(plain.lora_A) == ["default"] and list(plain.lora_magnitude_vector) == []


class TestRefusals:
    def test_attention_and_raw_parameters_do_not_take_dora(self, device: str) -> None:
        attention = nn.MultiheadAttention(8, 2, batch_first=True).to(device)
        with pytest.raises(ValueError, match="MultiheadAttention does not support DoRA"):
            MultiheadAttention(attention, "default", r=2, lora_alpha=4, use_dora=True)

        class Gate(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(lucid.randn(4, 6))

        with pytest.raises(ValueError, match="ParamWrapper does not support DoRA"):
            ParamWrapper(Gate().to(device), "default", "w", r=2, lora_alpha=4, use_dora=True)

    def test_a_refused_adapter_leaves_the_layer_as_it_was(self, device: str) -> None:
        _, layer = _linear(device)
        with pytest.raises(ValueError, match="already exists"):
            layer.update_layer("default", 2, 4.0, 0.0, True, False, True)
        assert list(layer.lora_magnitude_vector) == ["default"]


def _mlp(device: str) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3)).to(device)


def _config(**kwargs: object) -> LoraConfig:
    fields: dict[str, object] = {
        "r": 3,
        "lora_alpha": 6,
        "target_modules": ["0", "2"],
        "init_lora_weights": False,
    } | kwargs
    return LoraConfig(**fields)  # type: ignore[arg-type]


class TestModel:
    def test_the_config_builds_dora_layers(self, device: str) -> None:
        peft = get_peft_model(_mlp(device), _config(use_dora=True))
        layers = [m for m in peft.modules() if isinstance(m, Linear)]
        assert len(layers) == 2 and all(layer.use_dora == {"default": True} for layer in layers)
        names = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert sum("lora_magnitude_vector" in n for n in names) == 2
        assert not any(p.requires_grad for n, p in peft.named_parameters() if "base_layer" in n)

    def test_mixed_adapters_in_one_model(self, device: str) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, _config(), adapter_name="plain")
        peft.add_adapter("dora", _config(use_dora=True, r=2))
        assert list(peft.base_model.model[0].lora_magnitude_vector) == ["dora"]
        x = lucid.randn(4, 6, device=device)
        peft.set_adapter("plain")
        plain = peft(x)
        peft.set_adapter("dora")
        assert not _close(peft(x), plain)
        peft.set_adapter(["plain", "dora"])
        both = peft(x)
        peft.base_model.merge_adapter()
        assert _close(peft(x), both, atol=1e-4)
        peft.base_model.unmerge_adapter()
        assert _close(peft(x), both, atol=1e-4)

    def test_merge_and_unload_gives_the_decomposed_model(self, device: str) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, _config(use_dora=True))
        for layer in peft.modules():
            if isinstance(layer, Linear):
                _perturb(layer, "default")
        x = lucid.randn(4, 6, device=device)
        expected = peft(x)
        merged = peft.merge_and_unload()
        assert type(merged[0]) is nn.Linear and _close(merged(x), expected, atol=1e-4)

    def test_the_adapter_file_uses_the_reference_keys(self, device: str, tmp_path: Path) -> None:
        peft = get_peft_model(_mlp(device), _config(use_dora=True, target_modules=["0"]))
        peft.save_pretrained(tmp_path)
        from lucid.peft import get_peft_model_state_dict

        state = get_peft_model_state_dict(peft)
        assert sorted(state) == [
            "base_model.model.0.lora_A.weight",
            "base_model.model.0.lora_B.weight",
            "base_model.model.0.lora_magnitude_vector.weight",
        ]
        assert (tmp_path / "adapter_model.safetensors").exists()

    def test_save_then_load_reproduces_the_model(self, device: str, tmp_path: Path) -> None:
        trained = get_peft_model(_mlp(device), _config(use_dora=True))
        for layer in trained.modules():
            if isinstance(layer, Linear):
                _perturb(layer, "default")
        x = lucid.randn(4, 6, device=device)
        expected = trained(x)
        trained.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_mlp(device), tmp_path)
        assert _close(loaded(x), expected, atol=1e-5)
        assert loaded.peft_config["default"].use_dora is True

    def test_weighted_adapters_refuse_a_dora_source(self, device: str) -> None:
        peft = get_peft_model(_mlp(device), _config(use_dora=True), adapter_name="a")
        peft.add_adapter("b", _config())
        with pytest.raises(NotImplementedError, match="DoRA"):
            peft.base_model.add_weighted_adapter(["a", "b"], [0.5, 0.5], "mix")
        assert sorted(peft.peft_config) == ["a", "b"]

    def test_deleting_a_dora_adapter(self, device: str) -> None:
        peft = get_peft_model(_mlp(device), _config(use_dora=True), adapter_name="a")
        peft.add_adapter("b", _config())
        with pytest.warns(UserWarning, match="now deleted"):
            peft.delete_adapter("a")
        for layer in peft.modules():
            if isinstance(layer, Linear):
                assert len(layer.lora_magnitude_vector) == 0


class TestConfig:
    def test_use_dora_is_a_field(self) -> None:
        config = LoraConfig(use_dora=True)
        assert config.use_dora is True and LoraConfig.from_dict(config.to_dict()) == config

    def test_use_dora_must_be_a_bool(self) -> None:
        with pytest.raises(TypeError, match="use_dora"):
            LoraConfig(use_dora=1)  # type: ignore[arg-type]


@pytest.mark.skipif(not metal_available(), reason="compiled executables run on Metal")
class TestCompile:
    def test_a_compiled_model_follows_dora_switches(self) -> None:
        net = _mlp("metal")
        peft = get_peft_model(net, _config(target_modules=["0"]), adapter_name="plain")
        peft.add_adapter("dora", _config(use_dora=True, target_modules=["0"]))
        for layer in peft.modules():
            if isinstance(layer, Linear):
                _perturb(layer, "dora")
        compiled = lucid.compile(peft)
        x = lucid.randn(4, 6).to("metal")
        with lucid.no_grad():
            for active in ("plain", "dora", ["plain", "dora"], "plain"):
                peft.set_adapter(active)
                assert _close(compiled(x), peft(x), atol=1e-4), active
            peft.set_adapter("dora")
            eager = peft(x)
            peft.base_model.merge_adapter()
            assert _close(compiled(x), eager, atol=1e-4)
            peft.base_model.unmerge_adapter()
            assert _close(compiled(x), eager, atol=1e-4)
