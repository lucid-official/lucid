"""GraLoRA: the block grid in closed form, its rank, its locality, the start, merging and the reference adapter library's numbers.

GraLoRA (Jung et al., 2025) cuts the weight into a ``k x k`` grid of blocks, the block that maps input
block ``n`` to output block ``j`` being ``B_{j,n} A_{j,n}^T`` of rank ``r / k`` (Eq. 5), so that the
whole has a rank-``r`` LoRA's parameters and up to ``k r`` rank (section 3.2), and a gradient that
reaches only the factors of the input blocks that were active (section 3.3).  The tests expect the grid
from plain loops (``_delta_forms``) and those two properties, rather than through the layer's own code;
the ``parity`` tests build the same layer in the reference library and compare.
"""

import warnings
from types import ModuleType

import pytest

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid.peft import GraloraConfig, GraloraModel, get_peft_model
from lucid.peft.tuners import gralora
from lucid.test.unit.peft import _delta_forms as forms
from lucid.test.unit.peft._parity import adapter_library, gap


def _layer(
    in_features: int = 8, out_features: int = 12, seed: int = 1, **kw: object
) -> tuple[gralora.Linear, nn.Linear]:
    lucid.manual_seed(seed)
    base = nn.Linear(in_features, out_features)
    fields: dict[str, object] = {"r": 4, "gralora_k": 2, "target_modules": ["x"]} | kw
    return gralora.Linear(base, "default", GraloraConfig.from_dict(fields)), base


def _randomise(layer: gralora.Linear, seed: int = 2) -> None:
    """Every factor non-zero, so the update is."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for parameter in layer._adapter_parameters("default"):
            parameter.copy_(lucid.randn(*parameter.shape))


def _span(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


def _general(layer: gralora.Linear) -> tuple[forms.Matrix, forms.Matrix] | None:
    spec = layer.hybrid_r["default"]
    if not spec:
        return None
    return (
        layer.gralora_A_general["default"].weight.tolist(),
        layer.gralora_B_general["default"].weight.tolist(),
    )


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = GraloraConfig(target_modules=["a"])
        assert (config.r, config.hybrid_r, config.alpha, config.gralora_k) == (32, 0, 64, 2)
        assert (config.gralora_dropout, config.bias, config.init_weights) == (0.0, "none", True)

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            GraloraConfig(r=8, gralora_k=3, target_modules=["a"])
        with pytest.raises(ValueError, match="positive"):
            GraloraConfig(gralora_k=0, target_modules=["a"])
        with pytest.raises(ValueError, match="non-negative"):
            GraloraConfig(hybrid_r=-1, target_modules=["a"])
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            GraloraConfig(gralora_dropout=1.0, target_modules=["a"])

    def test_trained_biases_are_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="gralora_only"):
            GraloraConfig(bias="gralora_only", target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = GraloraConfig(r=8, gralora_k=4, hybrid_r=2, alpha=16, target_modules=["q", "v"])
        assert GraloraConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "GRALORA"


class TestStart:
    def test_the_update_starts_at_zero_and_the_output_is_the_base_bit_for_bit(
        self, device: str
    ) -> None:
        for hybrid in (0, 2):
            layer, base = _layer(hybrid_r=hybrid)
            layer.to(device)
            x = lucid.randn(5, 8, device=device)
            assert float(layer.get_delta_weight("default").abs().max().item()) == 0.0
            assert bool((layer(x) == base(x)).all().item())

    def test_a_random_start_is_not_the_base(self) -> None:
        layer, base = _layer(init_weights=False)
        x = lucid.randn(5, 8)
        assert _span(layer(x), base(x)) > 1e-3

    def test_the_shapes_and_the_parameter_count_are_a_loras(self) -> None:
        layer, _ = _layer(r=4, gralora_k=2)
        a, b = layer.gralora_A["default"], layer.gralora_B["default"]
        assert tuple(a.shape) == (2, 4, 4) and tuple(b.shape) == (2, 4, 6)
        assert a.numel() + b.numel() == 4 * (8 + 12)  # r (in + out)


class TestFormula:
    @pytest.mark.parametrize(
        ("k", "hybrid"), [(2, 0), (2, 2), (4, 0)], ids=["k2", "k2-hybrid", "k4"]
    )
    def test_the_update_is_the_papers_grid(self, k: int, hybrid: int, device: str) -> None:
        layer, _ = _layer(r=4, gralora_k=k, hybrid_r=hybrid, alpha=6.0)
        layer.to(device)
        _randomise(layer)
        expected = forms.gralora_delta(
            layer.gralora_A["default"].tolist(),
            layer.gralora_B["default"].tolist(),
            6.0 / (4 + hybrid),
            _general(layer),
        )
        got = layer.get_delta_weight("default")
        assert bool(lucid.allclose(got, lucid.tensor(expected, device=device), atol=1e-5))

    def test_the_forward_pass_is_the_update_applied_to_the_input(self, device: str) -> None:
        # The dense probe: the identity through the layer minus the identity through the base.
        layer, base = _layer(hybrid_r=2)
        layer.to(device)
        _randomise(layer)
        probe = lucid.eye(8, device=device)
        applied = layer(probe) - base(probe)
        assert bool(lucid.allclose(applied, layer.get_delta_weight("default").mT, atol=1e-5))

    def test_any_number_of_leading_axes_gives_the_same_rows(self) -> None:
        layer, base = _layer(hybrid_r=1)
        _randomise(layer)
        x = lucid.randn(2, 3, 8)
        batched = layer(x)
        assert tuple(batched.shape) == (2, 3, 12)
        assert bool(lucid.allclose(batched.reshape(6, 12), layer(x.reshape(6, 8)), atol=1e-6))
        assert tuple(layer(x[0, 0]).shape) == (12,)

    def test_the_update_has_k_times_the_rank(self) -> None:
        # Section 3.2: rank k r, against r for a LoRA with the same parameters.
        layer, _ = _layer(in_features=8, out_features=8, r=2, gralora_k=2, init_weights=False)
        _randomise(layer)
        assert int(linalg.matrix_rank(layer.get_delta_weight("default")).item()) == 4
        plain, _ = _layer(in_features=8, out_features=8, r=2, gralora_k=1, init_weights=False)
        _randomise(plain)
        assert int(linalg.matrix_rank(plain.get_delta_weight("default")).item()) == 2

    def test_an_input_block_reaches_only_its_own_factors(self) -> None:
        # Section 3.3: an input that is non-zero in block n alone has gradient on A_n alone.
        layer, _ = _layer(r=4, gralora_k=2)
        _randomise(layer)
        x = lucid.zeros(3, 8)
        x[:, :4] = lucid.randn(3, 4)  # input block 0
        layer(x).sum().backward()
        grad = layer.gralora_A["default"].grad
        assert grad is not None
        assert float(grad[0].abs().max().item()) > 0
        assert float(grad[1].abs().max().item()) == 0.0

    def test_dropout_acts_on_the_input_in_training_only(self) -> None:
        layer, _ = _layer(gralora_dropout=0.5)
        _randomise(layer)
        x = lucid.randn(8, 8)
        layer.eval()
        steady = layer(x)
        assert bool((layer(x) == steady).all().item())
        layer.train()
        lucid.manual_seed(0)
        assert _span(layer(x), steady) > 1e-3


class TestMerging:
    def test_merging_adds_the_update_to_the_weight(self) -> None:
        layer, base = _layer(hybrid_r=2)
        _randomise(layer)
        weight = base.weight.tolist()
        delta = layer.get_delta_weight("default").tolist()
        layer.merge()
        assert bool(lucid.allclose(base.weight, lucid.tensor(forms.add(weight, delta)), atol=1e-6))

    def test_the_merged_layer_is_the_adapted_one(self, device: str) -> None:
        layer, _ = _layer(hybrid_r=2)
        layer.to(device)
        _randomise(layer)
        x = lucid.randn(5, 8, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert bool(lucid.allclose(layer(x), adapted, atol=1e-5))

    def test_unmerging_gives_the_weights_back_to_rounding(self) -> None:
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
        net = nn.Sequential(nn.Linear(8, 12), nn.ReLU(), nn.Linear(12, 4))
        first = GraloraConfig(r=4, gralora_k=2, target_modules=["0", "2"], init_weights=False)
        peft = get_peft_model(net, first, adapter_name="a")
        peft.add_adapter(
            "b",
            GraloraConfig(r=2, gralora_k=2, hybrid_r=1, target_modules=["0"], init_weights=False),
        )
        x = lucid.randn(5, 8)
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


class TestFiles:
    def test_a_deleted_adapter_leaves_nothing_behind(self) -> None:
        layer, _ = _layer(hybrid_r=2)
        layer.update_layer("other", GraloraConfig(r=2, gralora_k=2, target_modules=["x"]))
        layer.delete_adapter("other")
        for name in layer.adapter_layer_names:
            assert list(getattr(layer, name)) == ["default"]
        assert "other" not in layer.gralora_dropout


class TestErrors:
    def test_sizes_the_grid_does_not_divide_are_refused_and_the_model_is_left_alone(self) -> None:
        net = nn.Sequential(nn.Linear(6, 8))
        before = [type(m) for m in net.modules()]
        with pytest.raises(ValueError, match="divisible by gralora_k"):
            GraloraModel(net, GraloraConfig(r=6, gralora_k=3, target_modules=["0"]), "default")
        assert [type(m) for m in net.modules()] == before

    def test_a_module_that_is_not_linear_is_refused(self) -> None:
        net = nn.Sequential(nn.Conv2d(2, 2, 1))
        with pytest.raises(ValueError, match="not supported"):
            GraloraModel(net, GraloraConfig(r=2, gralora_k=2, target_modules=["0"]), "default")

    def test_fan_in_fan_out_warns_and_is_cleared(self) -> None:
        net = nn.Sequential(nn.Linear(8, 8))
        config = GraloraConfig(r=2, gralora_k=2, fan_in_fan_out=True, target_modules=["0"])
        with pytest.warns(UserWarning, match="fan_in_fan_out"):
            peft = get_peft_model(net, config)
        stored = peft.peft_config["default"]
        assert isinstance(stored, GraloraConfig) and stored.fan_in_fan_out is False
        with pytest.raises(ValueError, match="fan_in_fan_out"):
            gralora.Linear(nn.Linear(8, 8), "x", config)


class TestHalfPrecision:
    @pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["float16", "bfloat16"])
    def test_the_adapter_is_float32_beside_a_half_base(self, dtype: lucid.dtype) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(8, 12).to(dtype)
        config = GraloraConfig(
            r=4, gralora_k=2, hybrid_r=2, target_modules=["x"], init_weights=False
        )
        layer = gralora.Linear(base, "default", config)
        assert {p.dtype for p in layer._adapter_parameters("default")} == {lucid.float32}
        out = layer(lucid.randn(3, 8).to(dtype))
        assert out.dtype == dtype and bool(lucid.isfinite(out).all().item())


# ── the reference adapter library ────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("hybrid", [0, 2], ids=["grid", "hybrid"])
def test_a_layer_matches_the_reference(ref: ModuleType, hybrid: int) -> None:
    library = adapter_library()
    lucid.manual_seed(5)
    net = nn.Sequential(nn.Linear(8, 12))
    twin = ref.nn.Sequential(ref.nn.Linear(8, 12))
    x = lucid.randn(4, 8)
    options = {"r": 4, "gralora_k": 2, "hybrid_r": hybrid, "alpha": 6, "init_weights": False}
    peft = get_peft_model(net, GraloraConfig.from_dict({"target_modules": ["0"], **options}))
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = library.get_peft_model(
            twin, library.GraloraConfig(target_modules=["0"], **options)
        )
    other = theirs.base_model.model[0]
    with ref.no_grad():
        other.gralora_A["default"].copy_(ref.tensor(mine.gralora_A["default"].tolist()))
        other.gralora_B["default"].copy_(ref.tensor(mine.gralora_B["default"].tolist()))
        if hybrid:
            for name in ("gralora_A_general", "gralora_B_general"):
                other_w = getattr(other, name)["default"].weight
                other_w.copy_(ref.tensor(getattr(mine, name)["default"].weight.tolist()))
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    assert gap(ref, mine.get_delta_weight("default"), other.get_delta_weight("default")) < 1e-5
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = GraloraConfig(r=4, gralora_k=2, target_modules=["a"]).to_dict()
    theirs = library.GraloraConfig(r=4, gralora_k=2, target_modules=["a"]).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
