"""DeLoRA: the update in closed form, its bound, the start, merging, files and the reference adapter library's numbers.

DeLoRA (Bini et al., 2025) normalises each rank-one component of the LoRA update and scales the sum by
a learned boundary and the pretrained weight's column norms:
``dW = (lambda / r) sum_i b_i a_i^T / (|b_i| |a_i|) * diag(w_norm)``.  The tests expect that matrix
from plain loops (``_delta_forms``), the bound ``|dW diag(1/w_norm)|_F <= lambda`` the paper proves,
and the invariances it is built for, rather than through the layer's own code; the ``parity`` tests
build the same layer in the reference library and compare.
"""

from pathlib import Path
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import DeloraConfig, DeloraModel, PeftModel, get_peft_model
from lucid.peft.tuners import delora
from lucid.test.unit.peft import _delta_forms as forms
from lucid.test.unit.peft._parity import adapter_library, gap


def _layer(
    in_features: int = 6, out_features: int = 4, seed: int = 1, **kw: object
) -> tuple[delora.Linear, nn.Linear]:
    lucid.manual_seed(seed)
    base = nn.Linear(in_features, out_features)
    fields: dict[str, object] = {"r": 3, "target_modules": ["x"]} | kw
    return delora.Linear(base, "default", DeloraConfig.from_dict(fields)), base


def _randomise(layer: delora.Linear, seed: int = 2) -> None:
    """Both factors non-zero, so the update is."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        layer.delora_A["default"].copy_(lucid.randn(*layer.delora_A["default"].shape))
        layer.delora_B["default"].copy_(lucid.randn(*layer.delora_B["default"].shape))


def _span(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = DeloraConfig(target_modules=["a"])
        assert (config.r, config.delora_lambda, config.module_dropout) == (8, 15, 0.0)
        assert (config.bias, config.init_weights) == ("none", True)
        assert config.rank_pattern == {} and config.lambda_pattern == {}

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DeloraConfig(r=0, target_modules=["a"])
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            DeloraConfig(module_dropout=1.0, target_modules=["a"])
        with pytest.raises(TypeError, match="init_weights"):
            DeloraConfig.from_dict({"init_weights": "yes", "target_modules": ["a"]})
        with pytest.raises(ValueError, match="positive"):
            DeloraConfig(rank_pattern={"q": 0}, target_modules=["a"])

    def test_trained_biases_are_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="delora_only"):
            DeloraConfig(bias="delora_only", target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = DeloraConfig(
            r=4,
            delora_lambda=3.5,
            rank_pattern={"q": 2},
            lambda_pattern={"v": 1.0},
            target_modules=["q", "v"],
        )
        assert DeloraConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "DELORA"


class TestStart:
    def test_the_update_starts_at_zero_and_the_output_is_the_base_bit_for_bit(
        self, device: str
    ) -> None:
        layer, base = _layer()
        layer.to(device)
        x = lucid.randn(5, 6, device=device)
        assert float(layer.get_delta_weight("default").abs().max().item()) == 0.0
        assert bool((layer(x) == base(x)).all().item())

    def test_a_random_start_is_not_the_base(self) -> None:
        layer, base = _layer(init_weights=False)
        x = lucid.randn(5, 6)
        assert _span(layer(x), base(x)) > 1e-3

    def test_the_first_step_reaches_b_and_nothing_is_nan(self) -> None:
        # B starts at zero and its column norm is floored, so the update is exactly zero and
        # the gradient finite: ``B`` learns first, ``A`` and ``lambda`` have nothing to say yet.
        layer, base = _layer()
        layer(lucid.randn(5, 6)).sum().backward()
        grads = {
            name: getattr(layer, name)["default"].grad
            for name in ("delora_A", "delora_B", "delora_lambda")
        }
        assert all(g is not None and bool(lucid.isfinite(g).all().item()) for g in grads.values())
        assert float(grads["delora_B"].abs().max().item()) > 0
        assert float(grads["delora_A"].abs().max().item()) == 0.0
        assert float(grads["delora_lambda"].abs().max().item()) == 0.0

    def test_the_boundary_starts_at_the_configured_value(self) -> None:
        layer, _ = _layer(delora_lambda=7.5)
        assert layer.delora_lambda["default"].tolist() == [7.5]


class TestFormula:
    def test_the_update_is_the_papers(self, device: str) -> None:
        layer, base = _layer(delora_lambda=4.0)
        layer.to(device)
        _randomise(layer)
        a = layer.delora_A["default"].tolist()
        b = layer.delora_B["default"].tolist()
        norms = layer.delora_w_norm["default"].tolist()
        expected = forms.delora_delta(a, b, 4.0, norms)
        got = layer.get_delta_weight("default")
        assert bool(lucid.allclose(got, lucid.tensor(expected, device=device), atol=1e-5))

    def test_the_norms_are_the_weights_column_norms(self) -> None:
        layer, base = _layer()
        columns = forms.transpose(base.weight.tolist())
        expected = [forms.frobenius([[v] for v in column]) for column in columns]
        got = layer.delora_w_norm["default"].tolist()
        assert all(abs(g - e) <= 1e-6 * e for g, e in zip(got, expected, strict=True))

    def test_the_forward_pass_is_the_update_applied_to_the_input(self, device: str) -> None:
        # A dense probe: the identity through the layer, minus the identity through the base,
        # is the transpose of the update.
        layer, base = _layer()
        layer.to(device)
        _randomise(layer)
        probe = lucid.eye(6, device=device)
        applied = layer(probe) - base(probe)
        assert bool(lucid.allclose(applied, layer.get_delta_weight("default").mT, atol=1e-5))

    def test_the_update_stays_inside_the_boundary(self) -> None:
        # |dW diag(1 / w_norm)|_F <= lambda, whatever the factors are (Eq. 6-7).
        layer, _ = _layer(delora_lambda=3.0)
        for seed in range(5):
            _randomise(layer, seed=seed)
            delta = layer.get_delta_weight("default") / layer.delora_w_norm["default"]
            assert float(forms.frobenius(delta.tolist())) <= 3.0 * (1 + 1e-5)

    def test_strength_is_apart_from_direction(self) -> None:
        # Scaling a factor changes nothing; the boundary scales the update linearly.
        layer, _ = _layer(delora_lambda=2.0)
        _randomise(layer)
        before = layer.get_delta_weight("default")
        with lucid.no_grad():
            layer.delora_A["default"].copy_(layer.delora_A["default"] * 5.0)
            layer.delora_B["default"].copy_(layer.delora_B["default"] * 0.25)
        assert bool(lucid.allclose(layer.get_delta_weight("default"), before, atol=1e-5))
        with lucid.no_grad():
            layer.delora_lambda["default"].copy_(layer.delora_lambda["default"] * 3.0)
        assert bool(lucid.allclose(layer.get_delta_weight("default"), before * 3.0, atol=1e-5))

    def test_the_weight_norms_are_fixed_when_the_adapter_is_made(self) -> None:
        layer, base = _layer()
        _randomise(layer)
        before = layer.get_delta_weight("default")
        with lucid.no_grad():
            base.weight.copy_(base.weight * 2.0)
        assert bool(lucid.equal(layer.get_delta_weight("default"), before))


class TestDropout:
    def test_dropout_acts_on_the_input_in_training_only(self) -> None:
        layer, base = _layer(module_dropout=0.5)
        _randomise(layer)
        x = lucid.randn(8, 6)
        layer.eval()
        steady = layer(x)
        assert bool((layer(x) == steady).all().item())
        layer.train()
        lucid.manual_seed(0)
        assert _span(layer(x), steady) > 1e-3


class TestMerging:
    def test_merging_adds_the_update_to_the_weight(self) -> None:
        layer, base = _layer()
        _randomise(layer)
        weight = base.weight.tolist()
        delta = layer.get_delta_weight("default").tolist()
        layer.merge()
        merged = base.weight
        assert bool(lucid.allclose(merged, lucid.tensor(forms.add(weight, delta)), atol=1e-6))
        assert layer.merged

    def test_the_merged_layer_is_the_adapted_one(self, device: str) -> None:
        layer, base = _layer()
        layer.to(device)
        _randomise(layer)
        x = lucid.randn(5, 6, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert bool(lucid.allclose(layer(x), adapted, atol=1e-5))

    def test_unmerging_gives_the_weights_back_to_rounding(self) -> None:
        # Adding and subtracting one update is exact to a unit in the last place of the larger
        # of the two, and no better: the base is not kept (it would double the merged weights).
        layer, base = _layer()
        _randomise(layer)
        before = base.weight.clone()
        layer.merge()
        layer.unmerge()
        spacing = 2.0**-23 * max(
            float(before.abs().max().item()),
            float(layer.get_delta_weight("default").abs().max().item()),
        )
        assert _span(base.weight, before) <= 2 * spacing

    def test_two_active_adapters_equal_their_merge_in_either_order(self) -> None:
        lucid.manual_seed(3)
        net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
        peft = get_peft_model(
            net, DeloraConfig(r=2, target_modules=["0", "2"], init_weights=False), adapter_name="a"
        )
        peft.add_adapter("b", DeloraConfig(r=3, target_modules=["0", "2"], init_weights=False))
        x = lucid.randn(5, 6)
        peft.set_adapter(["a", "b"])
        with lucid.no_grad():
            both = peft(x)
            weights = []
            for order in (["a", "b"], ["b", "a"]):
                peft.base_model.merge_adapter(order)
                weights.append(net[0].base_layer.weight.clone())
                assert bool(lucid.allclose(peft(x), both, atol=1e-4))
                peft.base_model.unmerge_adapter()
        assert _span(weights[0], weights[1]) <= 1e-6 * float(weights[0].abs().max().item())

    def test_patterns_set_a_modules_own_rank_and_boundary(self) -> None:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
        config = DeloraConfig(
            r=2,
            delora_lambda=9.0,
            rank_pattern={"2": 3},
            lambda_pattern={"0": 1.5},
            target_modules=["0", "2"],
        )
        peft = get_peft_model(net, config)
        first, last = peft.base_model.model[0], peft.base_model.model[2]
        assert first.r == {"default": 2} and last.r == {"default": 3}
        assert first.delora_lambda["default"].tolist() == [1.5]
        assert last.delora_lambda["default"].tolist() == [9.0]
        assert peft.peft_config["default"].rank_pattern == {"2": 3}


class TestFiles:
    def test_the_weight_norms_are_saved_and_read_back(self, tmp_path: Path) -> None:
        lucid.manual_seed(4)
        make = lambda: nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))  # noqa: E731
        trained = get_peft_model(make(), DeloraConfig(r=2, target_modules=["0", "2"]))
        _randomise(trained.base_model.model[0])
        trained.save_pretrained(tmp_path)
        lucid.manual_seed(9)  # another base: its own column norms are not the file's
        loaded = PeftModel.from_pretrained(make(), tmp_path)
        norms = loaded.base_model.model[0].delora_w_norm["default"]
        assert bool(lucid.equal(norms, trained.base_model.model[0].delora_w_norm["default"]))

    def test_a_deleted_adapter_leaves_no_norms(self) -> None:
        layer, _ = _layer()
        layer.update_layer("other", DeloraConfig(r=2, target_modules=["x"]))
        layer.delete_adapter("other")
        assert list(layer.delora_w_norm) == ["default"]
        assert "other" not in layer.delora_dropout


class TestHalfPrecision:
    @pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["float16", "bfloat16"])
    def test_the_adapter_is_float32_beside_a_half_base(self, dtype: lucid.dtype) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(6, 4).to(dtype)
        layer = delora.Linear(
            base, "default", DeloraConfig(r=2, target_modules=["x"], init_weights=False)
        )
        assert {p.dtype for p in layer._adapter_parameters("default")} == {lucid.float32}
        assert layer.delora_w_norm["default"].dtype == lucid.float32
        out = layer(lucid.randn(3, 6).to(dtype))
        assert out.dtype == dtype and bool(lucid.isfinite(out).all().item())


class TestErrors:
    def test_a_module_that_is_not_linear_is_refused(self) -> None:
        net = nn.Sequential(nn.Conv2d(2, 2, 1))
        with pytest.raises(ValueError, match="not supported"):
            DeloraModel(net, DeloraConfig(target_modules=["0"]), "default")


# ── the reference adapter library ────────────────────────────────────────────


@pytest.mark.parity
def test_a_layer_matches_the_reference(ref: ModuleType) -> None:
    library = adapter_library()
    lucid.manual_seed(5)
    net = nn.Sequential(nn.Linear(8, 6))
    twin = ref.nn.Sequential(ref.nn.Linear(8, 6))
    x = lucid.randn(4, 8)
    options = {"r": 3, "delora_lambda": 4.0, "init_weights": False}
    peft = get_peft_model(net, DeloraConfig.from_dict({"target_modules": ["0"], **options}))
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    theirs = library.get_peft_model(twin, library.DeloraConfig(target_modules=["0"], **options))
    other = theirs.base_model.model[0]
    with ref.no_grad():
        other.delora_A["default"].copy_(ref.tensor(mine.delora_A["default"].tolist()))
        other.delora_B["default"].copy_(ref.tensor(mine.delora_B["default"].tolist()))
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    assert gap(ref, mine.get_delta_weight("default"), other.get_delta_weight("default")) < 1e-5
    assert gap(ref, mine.delora_w_norm["default"], other.delora_w_norm["default"]) < 1e-6
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = DeloraConfig(r=4, target_modules=["a"]).to_dict()
    theirs = library.DeloraConfig(r=4, target_modules=["a"]).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
