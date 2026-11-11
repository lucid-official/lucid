"""DEFT: the residual projection in closed form, its identities, the stash that makes a merge exact and the reference adapter library's numbers.

DEFT (Kumar et al., 2025) adapts a weight as ``W' = (I - P_proj) W + Q_P R`` (Eq. 1, 2): it removes the
subspace ``P_proj`` projects onto and writes the injection ``R`` into the span of ``Q_P``.  The tests
expect that matrix from plain loops (``_delta_forms``), the identities the equation gives for an
orthonormal ``Q_P`` (the complement of the subspace is untouched, the subspace holds exactly ``R``),
and the merge the update's dependence on ``W`` asks for: unmerging gives the weight back bit for bit.
The ``parity`` tests build the same layers in the reference library and compare.
"""

from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import DeftConfig, DeftModel, get_peft_model, get_peft_model_state_dict
from lucid.peft.tuners import deft
from lucid.test.unit.peft import _delta_forms as forms
from lucid.test.unit.peft._parity import adapter_library, gap

_METHODS = ["relu", "qr"]


def _config(**kw: object) -> DeftConfig:
    fields: dict[str, object] = {"r": 3, "target_modules": ["x"]} | kw
    return DeftConfig.from_dict(fields)


def _layer(
    out: int = 6, inn: int = 5, seed: int = 1, **kw: object
) -> tuple[deft.Linear, nn.Linear]:
    lucid.manual_seed(seed)
    base = nn.Linear(inn, out)
    return deft.Linear(base, "default", _config(**kw)), base


def _randomise(layer: deft.DeftLayer, seed: int = 2, scale: float = 1.0) -> None:
    """P and R at a scale that makes the update large."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for parameter in layer._adapter_parameters("default"):
            parameter.copy_(scale * lucid.randn(*parameter.shape))


def _span(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _expected(layer: deft.DeftLayer, base: nn.Linear, **options: object) -> forms.Matrix:
    spec = layer._spec("default")
    injection = layer.deft_R["default"].tolist() if "default" in layer.deft_R else None
    return forms.deft_delta(
        base.weight.tolist(),
        layer.deft_P["default"].tolist(),
        injection,
        method=spec.deft_decomposition,
        scaling=spec.deft_scaling,
    )


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = DeftConfig(target_modules=["a"])
        assert (config.r, config.decomposition_method, config.init_scale) == (8, "relu", 1.0)
        assert (config.alpha, config.para, config.deft_dropout) == (None, False, 0.0)
        assert (config.init_weights, config.bias, config.fan_in_fan_out) == (True, "none", False)

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DeftConfig(r=0, target_modules=["a"])
        with pytest.raises(ValueError, match="svd"):
            DeftConfig.from_dict({"decomposition_method": "svd", "target_modules": ["a"]})
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            DeftConfig(deft_dropout=1.0, target_modules=["a"])

    def test_trained_biases_are_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="deft_only"):
            DeftConfig(bias="deft_only", target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = DeftConfig(
            r=4, decomposition_method="qr", alpha=8.0, para=True, target_modules=["q"]
        )
        assert DeftConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "DEFT"


class TestStart:
    @pytest.mark.parametrize("method", _METHODS)
    @pytest.mark.parametrize("alpha", [None, 6.0], ids=["unscaled", "alpha"])
    def test_the_default_start_is_the_identity_to_rounding(
        self, method: str, alpha: float | None, device: str
    ) -> None:
        layer, base = _layer(decomposition_method=method, alpha=alpha)
        layer.to(device)
        x = lucid.randn(5, 5, device=device)
        scale = float(base(x).abs().max().item())
        assert _span(layer(x), base(x)) <= 1e-5 * scale
        delta = layer.get_delta_weight("default")
        assert float(delta.abs().max().item()) <= 1e-5 * float(base.weight.abs().max().item())

    def test_pure_removal_cannot_start_as_the_identity(self) -> None:
        layer, base = _layer(para=True, decomposition_method="qr")
        assert "default" not in layer.deft_R
        x = lucid.randn(5, 5)
        assert _span(layer(x), base(x)) > 1e-3

    def test_a_random_start_draws_r_with_the_scaled_deviation(self) -> None:
        lucid.manual_seed(0)
        layer = deft.Linear(
            nn.Linear(400, 8), "default", _config(r=8, init_weights=False, init_scale=5.0)
        )
        assert abs(float(layer.deft_R["default"].std().item()) - 0.1) < 0.01  # 0.02 * 5
        assert abs(float(layer.deft_P["default"].std().item()) - 0.02) < 0.01

    def test_the_rank_is_capped_at_the_output_size(self) -> None:
        layer, _ = _layer(out=2, inn=5, r=8)
        assert layer.deft_r == {"default": 2}
        assert tuple(layer.deft_P["default"].shape) == (2, 2)
        assert tuple(layer.deft_R["default"].shape) == (2, 5)

    def test_the_scale_is_alpha_over_the_rank(self) -> None:
        assert _layer(alpha=6.0)[0].deft_scaling == {"default": 2.0}
        assert _layer()[0].deft_scaling == {"default": 1.0}
        assert _layer(out=2, r=8, alpha=4.0)[0].deft_scaling == {"default": 2.0}  # the capped rank


class TestFormula:
    @pytest.mark.parametrize("method", _METHODS)
    @pytest.mark.parametrize("para", [False, True], ids=["inject", "para"])
    def test_the_update_is_the_papers_projection(
        self, method: str, para: bool, device: str
    ) -> None:
        layer, base = _layer(decomposition_method=method, para=para, alpha=6.0)
        layer.to(device)
        base.to(device)
        _randomise(layer)
        expected = _expected(layer, base)
        got = layer.get_delta_weight("default")
        assert bool(lucid.allclose(got, lucid.tensor(expected, device=device), atol=1e-4))

    @pytest.mark.parametrize("method", _METHODS)
    def test_the_forward_pass_is_the_adapted_weight_applied_to_the_input(
        self, method: str, device: str
    ) -> None:
        # The dense probe: the identity through the layer, minus the identity through the base.
        layer, base = _layer(decomposition_method=method, alpha=6.0)
        layer.to(device)
        _randomise(layer)
        probe = lucid.eye(5, device=device)
        applied = layer(probe) - base(probe)
        assert bool(lucid.allclose(applied, layer.get_delta_weight("default").mT, atol=1e-4))

    def test_the_relu_projector_is_p_relu_p_transposed(self) -> None:
        # Eq. 2 written with the out x out projector: W' = W - P relu(P)^T W + P (s R).
        layer, base = _layer(alpha=6.0)
        _randomise(layer)
        p = layer.deft_P["default"].tolist()
        right = [[max(v, 0.0) for v in row] for row in p]
        projector = forms.matmul(p, forms.transpose(right))
        weight = base.weight.tolist()
        adapted = forms.add(
            forms.sub(weight, forms.matmul(projector, weight)),
            forms.matmul(p, forms.scaled(layer.deft_R["default"].tolist(), 2.0)),
        )
        merged = lucid.tensor(weight) + layer.get_delta_weight("default")
        assert bool(lucid.allclose(merged, lucid.tensor(adapted), atol=1e-4))

    def test_the_qr_projector_keeps_the_complement_and_holds_exactly_r_in_the_subspace(
        self,
    ) -> None:
        layer, base = _layer(decomposition_method="qr", alpha=6.0)
        _randomise(layer)
        q, _ = lucid.linalg.qr(layer.deft_P["default"])
        adapted = base.weight + layer.get_delta_weight("default")
        # Q^T W' = s R: the subspace holds the injection ...
        assert bool(lucid.allclose(q.mT @ adapted, layer.deft_R["default"] * 2.0, atol=1e-4))
        # ... and (I - Q Q^T) W' = (I - Q Q^T) W: the complement is untouched.
        eye = lucid.eye(6)
        complement = eye - q @ q.mT
        assert bool(lucid.allclose(complement @ adapted, complement @ base.weight, atol=1e-4))

    def test_pure_removal_leaves_nothing_in_the_subspace(self) -> None:
        layer, base = _layer(decomposition_method="qr", para=True)
        _randomise(layer)
        q, _ = lucid.linalg.qr(layer.deft_P["default"])
        adapted = base.weight + layer.get_delta_weight("default")
        assert float((q.mT @ adapted).abs().max().item()) <= 1e-5

    def test_a_gradient_reaches_the_direction_matrix_through_the_decomposition(
        self, device: str
    ) -> None:
        layer, _ = _layer(decomposition_method="qr", init_weights=False)
        layer.to(device)
        layer(lucid.randn(5, 5, device=device)).sum().backward()
        grad = layer.deft_P["default"].grad
        assert grad is not None and bool(lucid.isfinite(grad).all().item())
        assert float(grad.abs().max().item()) > 0

    def test_dropout_acts_on_the_injection_only_and_in_training_only(self) -> None:
        layer, _ = _layer(deft_dropout=0.5)
        _randomise(layer)
        x = lucid.randn(8, 5)
        layer.eval()
        steady = layer(x)
        assert _same(layer(x), steady)
        layer.train()
        lucid.manual_seed(0)
        assert _span(layer(x), steady) > 1e-3
        removal_only, _ = _layer(para=True, deft_dropout=0.5)
        _randomise(removal_only)
        removal_only.eval()
        quiet = removal_only(x)
        removal_only.train()
        assert _same(removal_only(x), quiet)  # no injection, no dropout to apply


class TestMerging:
    @pytest.mark.parametrize("method", _METHODS)
    def test_the_merged_weight_is_the_adapted_weight(self, method: str) -> None:
        layer, base = _layer(decomposition_method=method)
        _randomise(layer)
        weight = base.weight.tolist()
        delta = _expected(layer, base)
        layer.merge()
        assert bool(lucid.allclose(base.weight, lucid.tensor(forms.add(weight, delta)), atol=1e-4))

    @pytest.mark.parametrize("method", _METHODS)
    def test_unmerging_gives_the_weights_back_bit_for_bit(self, method: str, device: str) -> None:
        layer, base = _layer(decomposition_method=method)
        layer.to(device)
        _randomise(layer)
        before = base.weight.clone()
        layer.merge()
        assert _span(base.weight, before) > 1e-3
        layer.unmerge()
        assert _same(base.weight, before)

    def test_the_merged_layer_is_the_adapted_one(self, device: str) -> None:
        layer, _ = _layer(decomposition_method="qr", alpha=6.0)
        layer.to(device)
        _randomise(layer)
        x = lucid.randn(5, 5, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert bool(lucid.allclose(layer(x), adapted, atol=1e-4))

    def test_two_adapters_are_each_read_from_the_original_weight(self) -> None:
        # The projector of the second adapter must not read the first's merged weight.
        layer, base = _layer()
        layer.update_layer("b", _config(r=2, decomposition_method="qr", para=True))
        for name, seed in (("default", 7), ("b", 8)):
            _randomise_named(layer, name, seed)
        layer.set_adapter(["default", "b"])
        weight = base.weight.clone()
        x = lucid.randn(5, 5)
        with lucid.no_grad():
            both = layer(x)
            for order in (["default", "b"], ["b", "default"]):
                layer.merge(adapter_names=order)
                assert bool(lucid.allclose(layer(x), both, atol=1e-4))
                merged = base.weight.clone()
                layer.unmerge()
                assert _same(base.weight, weight)
            expected = weight + layer.get_delta_weight("default") + layer.get_delta_weight("b")
            assert bool(lucid.allclose(merged, expected, atol=1e-5))

    def test_merge_and_unload_gives_the_adapted_model(self, device: str) -> None:
        lucid.manual_seed(1)
        net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)
        config = DeftConfig(
            r=2, decomposition_method="qr", init_weights=False, target_modules=["0", "2"]
        )
        peft = get_peft_model(net, config)
        x = lucid.randn(5, 6, device=device)
        with lucid.no_grad():
            adapted = peft(x)
            merged = peft.merge_and_unload()
            assert merged is net and type(net[0]) is nn.Linear
            assert bool(lucid.allclose(net(x), adapted, atol=1e-4))


def _randomise_named(layer: deft.DeftLayer, name: str, seed: int) -> None:
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for parameter in layer._adapter_parameters(name):
            parameter.copy_(lucid.randn(*parameter.shape))


class TestFiles:
    def test_pure_removal_saves_no_injection(self) -> None:
        net = nn.Sequential(nn.Linear(6, 4))
        peft = get_peft_model(net, DeftConfig(r=2, para=True, target_modules=["0"]))
        assert sorted(get_peft_model_state_dict(peft)) == ["base_model.model.0.deft_P"]

    def test_the_stash_is_not_saved(self) -> None:
        layer, _ = _layer()
        _randomise(layer)
        keys = sorted(layer.state_dict())
        layer.merge()
        assert sorted(layer.state_dict()) == keys

    def test_a_deleted_adapter_leaves_nothing_behind(self) -> None:
        layer, _ = _layer()
        layer.update_layer("other", _config(para=True))
        layer.delete_adapter("other")
        assert list(layer.deft_P) == ["default"] and list(layer.deft_R) == ["default"]
        assert list(layer.deft_dropout) == ["default"]
        assert "other" not in layer._specs


class TestErrors:
    def test_a_module_that_is_not_linear_is_refused(self) -> None:
        net = nn.Sequential(nn.Conv2d(2, 2, 1))
        with pytest.raises(ValueError, match="not supported"):
            DeftModel(net, DeftConfig(target_modules=["0"]), "default")

    def test_fan_in_fan_out_warns_and_is_cleared(self) -> None:
        net = nn.Sequential(nn.Linear(6, 4))
        with pytest.warns(UserWarning, match="fan_in_fan_out"):
            peft = get_peft_model(net, DeftConfig(r=2, fan_in_fan_out=True, target_modules=["0"]))
        stored = peft.peft_config["default"]
        assert isinstance(stored, DeftConfig) and stored.fan_in_fan_out is False
        with pytest.raises(ValueError, match="fan_in_fan_out"):
            deft.Linear(nn.Linear(6, 4), "x", _config(fan_in_fan_out=True))


class TestHalfPrecision:
    @pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["float16", "bfloat16"])
    @pytest.mark.parametrize("method", _METHODS)
    def test_the_adapter_is_float32_beside_a_half_base(
        self, dtype: lucid.dtype, method: str
    ) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(6, 4).to(dtype)
        layer = deft.Linear(
            base, "default", _config(decomposition_method=method, init_weights=False)
        )
        assert {p.dtype for p in layer._adapter_parameters("default")} == {lucid.float32}
        out = layer(lucid.randn(3, 6).to(dtype))
        assert out.dtype == dtype and bool(lucid.isfinite(out).all().item())
        before = base.weight.clone()
        layer.merge()
        layer.unmerge()
        assert _same(base.weight, before) and base.weight.dtype == dtype


# ── the reference adapter library ────────────────────────────────────────────

_OPTIONS = [
    {},
    {"decomposition_method": "qr"},
    {"decomposition_method": "qr", "alpha": 6.0, "init_weights": False},
    {"para": True},
    {"para": True, "decomposition_method": "qr"},
    {"init_weights": False, "init_scale": 5.0},
]


@pytest.mark.parity
@pytest.mark.parametrize(
    "options", _OPTIONS, ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()) or "default"
)
def test_a_layer_matches_the_reference(ref: ModuleType, options: dict[str, object]) -> None:
    library = adapter_library()
    lucid.manual_seed(5)
    net = nn.Sequential(nn.Linear(8, 6))
    twin = ref.nn.Sequential(ref.nn.Linear(8, 6))
    x = lucid.randn(4, 8)
    fields = {"r": 3, **options}
    peft = get_peft_model(net, DeftConfig.from_dict({"target_modules": ["0"], **fields}))
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    theirs = library.get_peft_model(twin, library.DeftConfig(target_modules=["0"], **fields))
    ours = get_peft_model_state_dict(peft)
    library.set_peft_model_state_dict(theirs, {k: ref.tensor(v.tolist()) for k, v in ours.items()})
    other = theirs.base_model.model[0]
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    assert gap(ref, mine.get_delta_weight("default"), other.get_delta_weight("default")) < 1e-5
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5


@pytest.mark.parity
def test_the_qr_basis_has_the_references_signs(ref: ModuleType) -> None:
    # The injection Q_P R depends on the sign of every column of Q_P, so a reference-trained R only
    # means the same here if the decomposition signs its columns the same way.
    ref.manual_seed(3)
    p = ref.randn(7, 3)
    expected = lucid.tensor(ref.linalg.qr(p)[0].tolist())
    assert bool(lucid.allclose(lucid.tensor(forms.householder_q(p.tolist())), expected, atol=1e-5))
    assert bool(lucid.allclose(lucid.linalg.qr(lucid.tensor(p.tolist()))[0], expected, atol=1e-5))


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = DeftConfig(r=4, target_modules=["a"]).to_dict()
    theirs = library.DeftConfig(r=4, target_modules=["a"]).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
