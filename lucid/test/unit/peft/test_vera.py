"""VeRA: its config, the shared random projections, the layers and the adapter file (CPU and Metal).

The projections are the method's one unusual state, so most of the tests are about them: how they
are drawn (a keyed generator, independent of the global one and of the device), what size they
take, who shares them, and what a file does with them (``save_projection`` on and off).  The VeRA
entries of the generic tuner contract (merging, multi-adapter, files, compile) are in
``test_tuner_contract.py``.
"""

import json
import math
import warnings
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    PeftConfig,
    PeftModel,
    PeftType,
    get_peft_model,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from lucid.peft.tuners._base import _tuner_layers
from lucid.peft.tuners.vera import Linear, VeraConfig, VeraLayer, VeraModel
from lucid.peft.tuners.vera._projection import _BufferDict, _Projection
from lucid.serialization import load_safetensors, save_safetensors
from lucid.test._fixtures.devices import metal_available


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


def _mlp(device: str = "cpu") -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)


def _config(**kw: object) -> VeraConfig:
    fields: dict[str, object] = {"r": 3, "target_modules": ["0", "2"], **kw}
    return VeraConfig(**fields)  # type: ignore[arg-type]


def _tuner(device: str = "cpu", **kw: object) -> VeraModel:
    return VeraModel(_mlp(device), _config(**kw))


def _vera(model: nn.Module, name: str) -> VeraLayer:
    found = dict(_tuner_layers(model))[name]
    assert isinstance(found, VeraLayer)
    return found


# ── the config ───────────────────────────────────────────────────────────────


class TestConfig:
    def test_the_reference_defaults(self) -> None:
        config = VeraConfig()
        assert (config.r, config.projection_prng_key, config.save_projection) == (256, 0, True)
        assert (config.vera_dropout, config.d_initial, config.init_weights) == (0.0, 0.1, True)
        assert config.peft_type is PeftType.VERA and config.bias == "none"

    @pytest.mark.parametrize(
        ("kw", "error", "match"),
        [
            ({"r": 0}, ValueError, "positive"),
            ({"vera_dropout": 1.0}, ValueError, "lie in"),
            ({"projection_prng_key": 1.5}, TypeError, "projection_prng_key"),
            ({"projection_prng_key": True}, TypeError, "projection_prng_key"),
            ({"d_initial": "x"}, TypeError, "d_initial"),
            ({"save_projection": 1}, TypeError, "save_projection"),
            ({"bias": "all"}, NotImplementedError, "bias='all'"),
            ({"bias": "vera_only"}, NotImplementedError, "bias='vera_only'"),
            ({"bias": "x"}, ValueError, "bias"),
        ],
    )
    def test_bad_fields_are_refused(
        self, kw: dict[str, object], error: type[Exception], match: str
    ) -> None:
        with pytest.raises(error, match=match):
            VeraConfig(**kw)  # type: ignore[arg-type]

    def test_not_saving_the_projections_warns(self) -> None:
        with pytest.warns(UserWarning, match="restored using the PRNG key"):
            VeraConfig(save_projection=False)

    def test_the_dict_round_trip(self) -> None:
        config = _config(projection_prng_key=7, d_initial=0.05, vera_dropout=0.1)
        assert VeraConfig.from_dict(config.to_dict()) == config
        assert PeftConfig.from_dict(config.to_dict()) == config

    def test_a_reference_file_reads_without_a_warning(self) -> None:
        reference = {
            "auto_mapping": None,
            "base_model_name_or_path": "tiny",
            "bias": "none",
            "d_initial": 0.1,
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "modules_to_save": None,
            "peft_type": "VERA",
            "peft_version": "0.21.2",
            "projection_prng_key": 0,
            "r": 256,
            "revision": None,
            "save_projection": True,
            "target_modules": ["q", "v"],
            "task_type": None,
            "vera_dropout": 0.0,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config = VeraConfig.from_dict(reference)
        assert config.r == 256 and config.target_modules == {"q", "v"}


# ── the projections ──────────────────────────────────────────────────────────


class TestProjection:
    def test_the_same_key_draws_the_same_pair_whatever_the_global_generator_does(self) -> None:
        lucid.manual_seed(1)
        first = _tuner(projection_prng_key=5)
        lucid.manual_seed(999)
        lucid.randn(100)
        second = _tuner(projection_prng_key=5)
        assert _same(first.vera_A["default"], second.vera_A["default"])
        assert _same(first.vera_B["default"], second.vera_B["default"])

    def test_another_key_draws_another_pair(self) -> None:
        first, second = _tuner(projection_prng_key=0), _tuner(projection_prng_key=1)
        assert not _same(first.vera_A["default"], second.vera_A["default"])
        assert not _same(first.vera_B["default"], second.vera_B["default"])

    def test_a_pair_is_not_the_same_as_its_transpose_twin(self) -> None:
        tuner = _tuner()
        # A and B come from one generator in turn: B does not repeat A's numbers.
        a, b = tuner.vera_A["default"], tuner.vera_B["default"]
        assert not _same(a.flatten()[:12], b.flatten()[:12])

    def test_the_pair_is_sized_for_the_largest_layer(self) -> None:
        tuner = _tuner(r=5)
        # Layers 0 (in 6, out 8) and 2 (in 8, out 4): the largest in is 8 and the largest out is 8.
        assert tuple(tuner.vera_A["default"].shape) == (5, 8)
        assert tuple(tuner.vera_B["default"].shape) == (8, 5)

    def test_the_values_are_kaiming_uniform_with_the_reference_s_bound(self) -> None:
        a, b = _tuner(r=4).vera_A["default"], _tuner(r=4).vera_B["default"]
        bound_a = math.sqrt(3.0) * math.sqrt(2.0) / math.sqrt(a.shape[1])
        bound_b = math.sqrt(3.0) * math.sqrt(2.0) / math.sqrt(b.shape[1])
        assert float(a.abs().max().item()) <= bound_a and float(b.abs().max().item()) <= bound_b
        assert float(a.abs().max().item()) > 0.5 * bound_a  # and it fills the range
        assert abs(float(a.mean().item())) < 0.15

    @pytest.mark.skipif(not metal_available(), reason="needs Metal")
    def test_the_draw_does_not_depend_on_the_device(self) -> None:
        cpu, metal = _tuner("cpu", projection_prng_key=3), _tuner("metal", projection_prng_key=3)
        assert metal.vera_A["default"].device.type == "metal"
        assert _same(cpu.vera_A["default"], metal.vera_A["default"].to("cpu"))
        assert _same(cpu.vera_B["default"], metal.vera_B["default"].to("cpu"))

    def test_draw_is_a_pure_function_of_its_arguments(self) -> None:
        one = _Projection.draw(9, 3, 8, 6, "cpu", lucid.float32)
        two = _Projection.draw(9, 3, 8, 6, "cpu", lucid.float32)
        assert _same(one.vera_a, two.vera_a) and _same(one.vera_b, two.vera_b)
        assert tuple(one.vera_a.shape) == (3, 6) and tuple(one.vera_b.shape) == (8, 3)

    def test_the_projections_are_buffers_and_never_train(self) -> None:
        peft = get_peft_model(_mlp(), _config(init_weights=False))
        names = {n for n, _ in peft.named_parameters()}
        assert not any("vera_A" in n or "vera_B" in n for n in names)
        # The one pair is listed once, under whichever holder comes first.
        assert sum(n.endswith("vera_A.default") for n, _ in peft.named_buffers()) == 1
        peft(lucid.randn(4, 6)).sum().backward()
        trained = {n for n, p in peft.named_parameters() if p.grad is not None}
        assert trained and all("vera_lambda" in n for n in trained)

    def test_every_layer_reads_the_one_pair_the_tuner_holds(self) -> None:
        tuner = _tuner()
        for _, layer in _tuner_layers(tuner.model):
            assert isinstance(layer, VeraLayer)
            assert layer.vera_A is tuner.vera_A and layer.vera_B is tuner.vera_B

    def test_a_layer_alone_draws_a_projection_of_its_own_size_from_its_key(self) -> None:
        base = nn.Linear(6, 4)
        layer = Linear(base, "a", _config(projection_prng_key=2), r=3)
        assert layer.vera_A is None
        x = lucid.randn(3, 6)
        layer(x)
        drawn = _Projection.draw(2, 3, 4, 6, "cpu", lucid.float32)
        assert _same(layer.vera_A["a"], drawn.vera_a) and _same(layer.vera_B["a"], drawn.vera_b)  # type: ignore[index]


class TestBufferDict:
    def test_it_holds_buffers_by_name(self) -> None:
        held = _BufferDict(persistent=True)
        held["a"] = lucid.ones(2)
        assert "a" in held and len(held) == 1 and list(held) == ["a"]
        assert list(held.state_dict()) == ["a"]
        del held["a"]
        assert "a" not in held and len(held) == 0
        with pytest.raises(KeyError):
            held["a"]
        with pytest.raises(KeyError):
            del held["a"]

    def test_a_non_persistent_one_is_not_in_the_state_dict(self) -> None:
        held = _BufferDict(persistent=False)
        held["a"] = lucid.ones(2)
        assert not held.persistent and list(held.state_dict()) == [] and "a" in held


# ── the layer ────────────────────────────────────────────────────────────────


class TestLayer:
    def test_the_vectors_and_their_shapes(self) -> None:
        tuner = _tuner(r=3)
        layer = _vera(tuner.model, "0")
        assert tuple(layer.vera_lambda_d["default"].shape) == (3,)
        assert tuple(layer.vera_lambda_b["default"].shape) == (8,)
        assert dict(layer.r) == {"default": 3}
        assert layer.adapter_layer_names == ("vera_lambda_b", "vera_lambda_d")
        assert (layer.in_features, layer.out_features) == (6, 8)

    def test_the_default_start_is_d_initial_and_zero_and_the_base_bit_for_bit(
        self, device: str
    ) -> None:
        net = _mlp(device)
        x = lucid.randn(5, 6, device=device)
        base = net(x)
        tuner = VeraModel(net, _config(d_initial=0.25))
        layer = _vera(net, "0")
        assert _same(layer.vera_lambda_d["default"], lucid.full((3,), 0.25, device=device))
        assert _same(layer.vera_lambda_b["default"], lucid.zeros(8, device=device))
        assert _same(tuner(x), base)

    def test_a_random_start_is_ones_and_noise(self) -> None:
        layer = _vera(_tuner(init_weights=False).model, "0")
        assert _same(layer.vera_lambda_b["default"], lucid.ones(8))
        assert not _same(layer.vera_lambda_d["default"], lucid.full((3,), 0.1))

    def test_the_update_is_lambda_b_times_b_lambda_d_a_x(self, device: str) -> None:
        net = _mlp(device)
        tuner = VeraModel(net, _config(init_weights=False))
        layer = _vera(net, "0")
        x = lucid.randn(5, 6, device=device)
        a, b = tuner.vera_A["default"], tuner.vera_B["default"]
        ld, lb = layer.vera_lambda_d["default"], layer.vera_lambda_b["default"]
        base = layer.base_layer(x)
        # layer 0 is (in 6, out 8): A is sliced to its 6 columns, B to its 8 rows.
        expected = base + lb * ((ld * (x @ a[:, :6].mT)) @ b[:8, :].mT)
        assert _close(layer(x), expected)

    def test_a_smaller_layer_reads_a_slice_of_the_pair(self, device: str) -> None:
        net = _mlp(device)
        tuner = VeraModel(net, _config(init_weights=False))
        layer = _vera(net, "2")  # in 8, out 4: B is sliced to 4 rows
        x = lucid.randn(5, 8, device=device)
        a, b = tuner.vera_A["default"], tuner.vera_B["default"]
        ld, lb = layer.vera_lambda_d["default"], layer.vera_lambda_b["default"]
        expected = layer.base_layer(x) + lb * ((ld * (x @ a.mT)) @ b[:4, :].mT)
        assert _close(layer(x), expected)

    def test_the_delta_weight_is_the_update_as_a_matrix(self, device: str) -> None:
        net = _mlp(device)
        VeraModel(net, _config(init_weights=False))
        layer = _vera(net, "0")
        x = lucid.randn(5, 6, device=device)
        delta = layer.get_delta_weight("default")
        assert tuple(delta.shape) == (8, 6)
        assert _close(layer(x), layer.base_layer(x) + x @ delta.mT)

    def test_the_merge_is_exact_and_undone_by_unmerge(self, device: str) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, _config(init_weights=False))
        layer = _vera(net, "0")
        weight = layer.weight.clone()
        x = lucid.randn(5, 6, device=device)
        with lucid.no_grad():
            adapted = peft(x)
            peft.base_model.merge_adapter()
            assert not _same(layer.weight, weight)
            assert _close(peft(x), adapted, atol=1e-5)
            peft.base_model.unmerge_adapter()
            assert _close(layer.weight, weight, atol=1e-6)

    def test_dropout_acts_on_the_adapter_input_only(self) -> None:
        net = _mlp()
        VeraModel(net, _config(init_weights=False, vera_dropout=0.5))
        layer = _vera(net, "0")
        x = lucid.randn(8, 6)
        layer.eval()
        assert _same(layer(x), layer(x))
        layer.train()
        lucid.manual_seed(1)
        first = layer(x)
        lucid.manual_seed(2)
        assert not _same(first, layer(x))

    def test_a_non_positive_rank_and_a_taken_name_are_refused(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            Linear(nn.Linear(4, 3), "a", _config(), r=0)
        layer = Linear(nn.Linear(4, 3), "a", _config(), r=2)
        with pytest.raises(ValueError, match="already exists"):
            layer.update_layer("a", 2, _config())

    def test_fan_in_fan_out_is_refused_by_the_layer_and_reset_by_the_tuner(self) -> None:
        with pytest.raises(ValueError, match="fan_in_fan_out"):
            Linear(nn.Linear(4, 3), "a", _config(fan_in_fan_out=True), r=2)
        with pytest.warns(UserWarning, match="fan_in_fan_out"):
            tuner = _tuner(fan_in_fan_out=True)
        config = tuner.peft_config["default"]
        assert isinstance(config, VeraConfig) and not config.fan_in_fan_out

    def test_a_half_base_gets_float32_vectors_and_keeps_its_dtype(self) -> None:
        net = nn.Sequential(nn.Linear(6, 4)).to(lucid.float16)
        tuner = VeraModel(net, _config(target_modules=["0"], init_weights=False))
        layer = _vera(net, "0")
        assert layer.vera_lambda_d["default"].dtype == lucid.float32
        assert tuner.vera_A["default"].dtype == lucid.float32
        assert tuner(lucid.randn(2, 6).to(lucid.float16)).dtype == lucid.float16

    def test_only_nn_linear_is_adapted(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4), nn.Conv2d(2, 2, 1))
        with pytest.raises(ValueError, match="not supported"):
            VeraModel(net, _config(target_modules=["0", "1"]))
        assert type(net[0]) is nn.Linear

    def test_the_delete_removes_the_dropout_and_the_spec(self) -> None:
        net = _mlp()
        tuner = VeraModel(net, _config(vera_dropout=0.1))
        tuner.add_adapter("b", _config(vera_dropout=0.1))
        layer = _vera(net, "0")
        tuner.delete_adapter("b")
        assert layer._adapter_names() == ["default"] and list(layer.vera_dropout) == ["default"]


# ── several adapters share the pair ──────────────────────────────────────────


class TestSharing:
    def test_a_later_adapter_reads_the_first_one_s_tensors(self) -> None:
        tuner = _tuner()
        tuner.add_adapter("b", _config(target_modules=["0"]))
        assert tuner.vera_A["b"] is tuner.vera_A["default"]
        assert tuner.vera_B["b"] is tuner.vera_B["default"]

    def test_a_lower_rank_adapter_reads_the_leading_rows(self, device: str) -> None:
        net = _mlp(device)
        tuner = VeraModel(net, _config(r=4, init_weights=False), adapter_name="big")
        tuner.add_adapter("small", _config(r=2, init_weights=False))
        tuner.set_adapter("small")
        layer = _vera(net, "0")
        x = lucid.randn(3, 6, device=device)
        a, b = tuner.vera_A["small"], tuner.vera_B["small"]
        ld, lb = layer.vera_lambda_d["small"], layer.vera_lambda_b["small"]
        assert tuple(ld.shape) == (2,)
        expected = layer.base_layer(x) + lb * ((ld * (x @ a[:2, :6].mT)) @ b[:8, :2].mT)
        assert _close(tuner(x), tuner.model[2](nn.functional.relu(expected)))

    def test_a_larger_layer_than_the_pair_covers_is_refused_and_nothing_changes(self) -> None:
        tuner = _tuner(target_modules=["2"])  # the pair is (3, 8) and (4, 3)
        before = sorted(tuner.state_dict())
        with pytest.raises(ValueError, match="vera_B has a size of 4 but 8 or greater"):
            tuner.add_adapter("b", _config(target_modules=["0"]))
        assert sorted(tuner.state_dict()) == before and "b" not in tuner.peft_config
        assert "b" not in tuner.vera_A and "b" not in tuner.vera_B

    def test_a_higher_rank_than_the_pair_is_refused(self) -> None:
        tuner = _tuner(r=2)
        with pytest.raises(ValueError, match="higher rank"):
            tuner.add_adapter("b", _config(r=4))

    def test_adapters_must_share_the_key_and_the_decision_to_save(self) -> None:
        tuner = _tuner(projection_prng_key=1)
        with pytest.raises(ValueError, match="PRNG initialisation key must be the same"):
            tuner.add_adapter("b", _config(projection_prng_key=2))
        with pytest.raises(ValueError, match="saved for all adapters or none"):
            with pytest.warns(UserWarning):
                tuner.add_adapter("c", _config(projection_prng_key=1, save_projection=False))

    def test_deleting_an_adapter_drops_its_entry_and_the_last_one_lets_a_new_pair_be_drawn(
        self,
    ) -> None:
        tuner = _tuner()
        tuner.add_adapter("b", _config())
        tuner.delete_adapter("b")
        assert list(tuner.vera_A) == ["default"] and list(tuner.vera_B) == ["default"]

    def test_a_failed_first_adapter_leaves_the_model_alone(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4), nn.Embedding(5, 4))
        with pytest.raises(ValueError, match="not supported"):
            VeraModel(net, _config(target_modules=["0", "1"]))
        assert type(net[0]) is nn.Linear


# ── the adapter file ─────────────────────────────────────────────────────────


def _trained(**kw: object) -> PeftModel:
    peft = get_peft_model(_mlp(), _config(init_weights=False, **kw))
    return peft


class TestFile:
    def test_the_file_holds_the_vectors_and_one_projection_pair(self, tmp_path: Path) -> None:
        _trained().save_pretrained(tmp_path)
        saved = load_safetensors(str(tmp_path / "adapter_model.safetensors"))
        shapes = {k: tuple(v.shape) for k, v in saved.items() if isinstance(v, lucid.Tensor)}
        assert shapes == {
            "base_model.model.0.vera_lambda_b": (8,),
            "base_model.model.0.vera_lambda_d": (3,),
            "base_model.model.2.vera_lambda_b": (4,),
            "base_model.model.2.vera_lambda_d": (3,),
            "base_model.vera_A": (3, 8),
            "base_model.vera_B": (8, 3),
        }
        config = json.loads((tmp_path / "adapter_config.json").read_text())
        assert config["peft_type"] == "VERA" and config["save_projection"] is True

    def test_a_bare_model_reads_its_projection_from_any_layer(self) -> None:
        net = _mlp()
        inject_adapter_in_model(_config(init_weights=False), net)
        state = get_peft_model_state_dict(net)
        assert tuple(state["base_model.vera_A"].shape) == (3, 8) and len(state) == 6
        other = _mlp()
        inject_adapter_in_model(_config(init_weights=False, projection_prng_key=9), other)
        set_peft_model_state_dict(other, state)
        assert _same(_vera(other, "0").vera_A["default"], state["base_model.vera_A"])  # type: ignore[index]

    def test_a_saved_projection_wins_over_the_key(self, tmp_path: Path) -> None:
        trained = _trained(projection_prng_key=4)
        with lucid.no_grad():  # a projection no key would draw
            trained.base_model.vera_A["default"].copy_(lucid.ones(3, 8))  # type: ignore[union-attr]
        x = lucid.randn(4, 6)
        trained.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_mlp(), tmp_path)
        assert _same(loaded.base_model.vera_A["default"], lucid.ones(3, 8))  # type: ignore[union-attr]
        with lucid.no_grad():
            assert _same(loaded(x), trained(x))

    def test_without_the_projections_the_file_is_small_and_the_key_restores_them(
        self, tmp_path: Path
    ) -> None:
        with pytest.warns(UserWarning, match="restored using the PRNG key"):
            trained = _trained(save_projection=False, projection_prng_key=6)
        assert not any("vera_A" in k for k in trained.state_dict())
        x = lucid.randn(4, 6)
        trained.save_pretrained(tmp_path)
        saved = load_safetensors(str(tmp_path / "adapter_model.safetensors"))
        assert sorted(saved) == [
            "base_model.model.0.vera_lambda_b",
            "base_model.model.0.vera_lambda_d",
            "base_model.model.2.vera_lambda_b",
            "base_model.model.2.vera_lambda_d",
        ]
        with pytest.warns(UserWarning, match="relying on PRNG initialisation"):
            loaded = PeftModel.from_pretrained(_mlp(), tmp_path)
        with lucid.no_grad():
            assert _same(loaded(x), trained(x))

    def test_a_file_that_carries_projections_the_config_does_not_want_ignores_them(
        self, tmp_path: Path
    ) -> None:
        with_pair = _trained()
        with_pair.save_pretrained(tmp_path)
        config = json.loads((tmp_path / "adapter_config.json").read_text())
        config["save_projection"] = False
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))
        with pytest.warns(UserWarning, match="however they are present"):
            PeftModel.from_pretrained(_mlp(), tmp_path)

    def test_a_config_that_wants_projections_the_file_lacks_is_refused(
        self, tmp_path: Path
    ) -> None:
        _trained().save_pretrained(tmp_path)
        saved = dict(load_safetensors(str(tmp_path / "adapter_model.safetensors")))
        del saved["base_model.vera_A"], saved["base_model.vera_B"]
        save_safetensors(
            saved, str(tmp_path / "adapter_model.safetensors"), metadata={"format": "pt"}
        )  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="were not present"):
            PeftModel.from_pretrained(_mlp(), tmp_path)

    def test_projections_drawn_for_a_bigger_model_are_taken_when_they_cover_the_layers(
        self, tmp_path: Path
    ) -> None:
        # An adapter on both layers saved a (3, 8) pair; one that reaches layer 0 alone drew (3, 6).
        peft = get_peft_model(_mlp(), _config(init_weights=False), adapter_name="a")
        peft.add_adapter("b", _config(init_weights=False, target_modules=["0"]))
        peft.save_pretrained(tmp_path)
        x = lucid.randn(4, 6)
        loaded = PeftModel.from_pretrained(_mlp(), tmp_path / "b", adapter_name="b")
        assert tuple(loaded.base_model.vera_A["b"].shape) == (3, 8)  # type: ignore[union-attr]
        peft.set_adapter("b")
        with lucid.no_grad():
            assert _same(loaded(x), peft(x))

    def test_projections_that_cannot_cover_the_layers_are_refused_with_the_model_untouched(
        self,
    ) -> None:
        peft = get_peft_model(_mlp(), _config(init_weights=False))
        state = get_peft_model_state_dict(peft)
        state["base_model.vera_A"] = lucid.zeros(3, 4)  # layer 0 reads 6 columns
        with pytest.raises(ValueError, match="do not fit"):
            set_peft_model_state_dict(peft, state)
        assert tuple(peft.base_model.vera_A["default"].shape) == (3, 8)  # type: ignore[union-attr]

    def test_a_projection_of_another_rank_is_refused(self) -> None:
        peft = get_peft_model(_mlp(), _config(init_weights=False))
        state = get_peft_model_state_dict(peft)
        state["base_model.vera_A"] = lucid.zeros(5, 8)
        with pytest.raises(ValueError, match="do not fit"):
            set_peft_model_state_dict(peft, state)

    def test_a_reference_written_adapter_loads_with_its_own_projection(
        self, tmp_path: Path
    ) -> None:
        lucid.manual_seed(3)
        tensors = {
            "base_model.model.0.vera_lambda_b": lucid.randn(8),
            "base_model.model.0.vera_lambda_d": lucid.randn(3),
            "base_model.model.2.vera_lambda_b": lucid.randn(4),
            "base_model.model.2.vera_lambda_d": lucid.randn(3),
            "base_model.vera_A": lucid.randn(3, 8),
            "base_model.vera_B": lucid.randn(8, 3),
        }
        save_safetensors(
            tensors, str(tmp_path / "adapter_model.safetensors"), metadata={"format": "pt"}
        )
        _config(inference_mode=True).save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_mlp(), tmp_path)
        net = _mlp()
        a, b = tensors["base_model.vera_A"], tensors["base_model.vera_B"]
        x = lucid.randn(5, 6)

        def updated(index: int, h: lucid.Tensor) -> lucid.Tensor:
            p = f"base_model.model.{index}."
            out = net[index](h)
            cols = net[index].weight.shape[1]
            rows = net[index].weight.shape[0]
            update = (h @ a[:, :cols].mT * tensors[p + "vera_lambda_d"]) @ b[:rows, :].mT
            return out + tensors[p + "vera_lambda_b"] * update

        expected = updated(2, nn.functional.relu(updated(0, x)))
        with lucid.no_grad():
            assert _close(loaded(x), expected, atol=1e-4)


# ── the tuner ────────────────────────────────────────────────────────────────


class TestTuner:
    def test_the_attributes_are_the_reference_s(self) -> None:
        tuner = _tuner()
        assert isinstance(tuner.vera_A, _BufferDict) and isinstance(tuner.vera_B, _BufferDict)
        assert sorted(k for k in tuner.state_dict() if "vera_" in k and "lambda" not in k)[:2] == [
            "model.0.vera_A.default",
            "model.0.vera_B.default",
        ]
        assert "vera_A.default" in tuner.state_dict()

    def test_unload_gives_the_base_back_and_merge_and_unload_matches(self, device: str) -> None:
        net = _mlp(device)
        before = {k: v.clone() for k, v in net.state_dict().items()}
        peft = get_peft_model(net, _config(init_weights=False))
        x = lucid.randn(4, 6, device=device)
        with lucid.no_grad():
            adapted = peft(x)
            merged = peft.merge_and_unload()
            assert merged is net and _close(net(x), adapted, atol=1e-5)
        peft2 = get_peft_model(_mlp(device), _config())
        net2 = peft2.unload()
        assert all(_same(v, before[k]) for k, v in net2.state_dict().items())

    def test_modules_to_save_and_the_task_head(self) -> None:
        peft = get_peft_model(_mlp(), _config(target_modules=["0"], modules_to_save=["2"]))
        assert "2" in {
            n.split(".")[2] for n, _ in peft.named_parameters() if "modules_to_save" in n
        }

    def test_no_default_for_a_model_type_without_a_row(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4))
        net.config = type("C", (), {"model_type": "resnet"})()  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Please specify `target_modules`"):
            VeraModel(net, VeraConfig())

    def test_the_default_targets_are_the_reference_s_for_the_zoo_types(self) -> None:
        from lucid.peft.tuners.vera._defaults import VERA_DEFAULT_TARGET_MODULES

        assert dict(VERA_DEFAULT_TARGET_MODULES) == {
            "gpt2": frozenset({"c_attn"}),
            "bert": frozenset({"query", "value"}),
            "t5": frozenset({"q", "v"}),
        }
