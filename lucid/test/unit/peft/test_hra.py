"""HRA: the reflection chain in closed form, the starts, the layers, and the reference adapter library's numbers.

A Householder reflection of the unit vector ``u`` is ``I - 2 u u^T``.  The tests
expect the chain of an adapter written out by hand (a reflection is applied to a
running product, entry by entry) rather than through the layer's helper, and the
``parity`` tests build the same layer in the reference library, copy the vectors
across and compare the forward pass, the matrix and the merged weight.
"""

from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import HRAConfig, HRAModel, PeftModel, get_peft_model
from lucid.peft.tuners import hra
from lucid.test.unit.peft._parity import adapter_library, gap


def _reflection(u: list[float]) -> list[list[float]]:
    norm2 = sum(v * v for v in u)
    return [
        [(1.0 if i == j else 0.0) - 2 * u[i] * u[j] / norm2 for j in range(len(u))]
        for i in range(len(u))
    ]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def _layer(
    in_features: int = 4, out_features: int = 3, **kw: object
) -> tuple[hra.Linear, nn.Linear]:
    lucid.manual_seed(1)
    base = nn.Linear(in_features, out_features)
    fields: dict[str, object] = {"r": 2, "target_modules": ["x"]} | kw
    return hra.Linear(base, "default", HRAConfig(**fields)), base  # type: ignore[arg-type]


def _set_vectors(layer: hra.HRALayer, columns: list[list[float]]) -> None:
    """Install ``columns`` (one list per reflection) as the layer's ``hra_u``."""
    with lucid.no_grad():
        layer.hra_u["default"].copy_(lucid.tensor(columns).mT)


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = HRAConfig(target_modules=["a"])
        assert (config.r, config.apply_GS, config.init_weights, config.bias) == (
            8,
            False,
            True,
            "none",
        )

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            HRAConfig(r=0, target_modules=["a"])
        with pytest.raises(TypeError, match="apply_GS"):
            HRAConfig(apply_GS="yes", target_modules=["a"])  # type: ignore[arg-type]

    def test_trained_biases_are_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="hra_only"):
            HRAConfig(bias="hra_only", target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = HRAConfig(r=4, apply_GS=True, target_modules=["q", "v"])
        assert HRAConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "HRA"


class TestStarts:
    def test_an_even_rank_starts_as_the_identity(self, device: str) -> None:
        layer, base = _layer(r=4)
        layer.to(device)
        eye = lucid.eye(4, device=device)
        assert bool(lucid.allclose(layer.get_delta_weight("default"), eye, atol=1e-6))
        x = lucid.randn(5, 4, device=device)
        assert bool(lucid.allclose(layer(x), base(x), atol=1e-6))

    def test_an_odd_rank_warns_and_starts_elsewhere(self) -> None:
        with pytest.warns(UserWarning, match="r is odd"):
            layer, _ = _layer(r=3)
        assert not bool(lucid.allclose(layer.get_delta_weight("default"), lucid.eye(4), atol=1e-3))

    def test_a_random_start_is_not_the_identity(self) -> None:
        layer, _ = _layer(init_weights=False)
        assert not bool(lucid.allclose(layer.get_delta_weight("default"), lucid.eye(4), atol=1e-3))


class TestChain:
    def test_the_matrix_is_the_product_of_the_reflections(self) -> None:
        layer, _ = _layer()
        u1, u2 = [1.0, 2.0, 0.0, -1.0], [0.5, -1.0, 2.0, 1.0]
        _set_vectors(layer, [u1, u2])
        expected = _matmul(_reflection(u1), _reflection(u2))
        assert bool(
            lucid.allclose(layer.get_delta_weight("default"), lucid.tensor(expected), atol=1e-6)
        )

    def test_gram_schmidt_orthonormalises_before_reflecting(self) -> None:
        layer, _ = _layer(apply_GS=True)
        u1, u2 = [1.0, 2.0, 0.0, -1.0], [0.5, -1.0, 2.0, 1.0]
        _set_vectors(layer, [u1, u2])
        n1 = sum(v * v for v in u1) ** 0.5
        e1 = [v / n1 for v in u1]
        dot = sum(a * b for a, b in zip(e1, u2, strict=True))
        w = [b - dot * a for a, b in zip(e1, u2, strict=True)]
        n2 = sum(v * v for v in w) ** 0.5
        e2 = [v / n2 for v in w]
        expected = [
            [(1.0 if i == j else 0.0) - 2 * (e1[i] * e1[j] + e2[i] * e2[j]) for j in range(4)]
            for i in range(4)
        ]
        assert bool(
            lucid.allclose(layer.get_delta_weight("default"), lucid.tensor(expected), atol=1e-6)
        )

    @pytest.mark.parametrize("gram_schmidt", [False, True])
    def test_the_matrix_is_orthogonal_and_the_reverse_chain_is_its_inverse(
        self, gram_schmidt: bool
    ) -> None:
        layer, _ = _layer(r=3, apply_GS=gram_schmidt, init_weights=False)
        d = layer.get_delta_weight("default")
        assert bool(lucid.allclose(d @ d.mT, lucid.eye(4), atol=1e-5))
        assert bool(
            lucid.allclose(
                d @ layer.get_delta_weight("default", reverse=True), lucid.eye(4), atol=1e-5
            )
        )

    def test_merging_multiplies_the_weight_and_unmerging_restores_it(self) -> None:
        layer, base = _layer(init_weights=False)
        weight = base.weight.clone()
        x = lucid.randn(3, 4)
        with lucid.no_grad():  # a live graph holds the weight, which merging then may not write
            adapted = layer(x)
        layer.merge()
        assert bool(
            lucid.allclose(base.weight, weight @ layer.get_delta_weight("default"), atol=1e-6)
        )
        assert bool(lucid.allclose(layer(x), adapted, atol=1e-5))
        layer.unmerge()
        assert bool(lucid.allclose(base.weight, weight, atol=1e-5))


class TestConv2d:
    def test_merged_equals_unmerged_and_the_vectors_span_in_by_kernel(self) -> None:
        lucid.manual_seed(2)
        base = nn.Conv2d(3, 4, 3, padding=1)
        layer = hra.Conv2d(
            base, "default", HRAConfig(r=2, init_weights=False, target_modules=["x"])
        )
        assert tuple(layer.hra_u["default"].shape) == (27, 2)
        x = lucid.randn(2, 3, 5, 5)
        with lucid.no_grad():
            adapted = layer(x)
        layer.merge()
        assert bool(lucid.allclose(layer(x), adapted, atol=1e-5))

    def test_a_grouped_convolution_is_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="groups > 1"):
            hra.Conv2d(nn.Conv2d(4, 4, 3, groups=2), "x", HRAConfig(target_modules=["x"]))


def test_the_tuner_adds_a_second_adapter_beside_the_first() -> None:
    net = nn.Sequential(nn.Linear(4, 4))
    peft = get_peft_model(net, HRAConfig(r=2, target_modules=["0"]), "a")
    assert isinstance(peft, PeftModel) and isinstance(peft.base_model, HRAModel)
    peft.add_adapter("b", HRAConfig(r=4, apply_GS=True, target_modules=["0"]))
    layer = peft.base_model.model[0]
    assert layer.hra_r == {"a": 2, "b": 4} and layer.hra_apply_GS == {"a": False, "b": True}


# ── the reference adapter library ────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("conv", [False, True], ids=["linear", "conv2d"])
@pytest.mark.parametrize("gram_schmidt", [False, True], ids=["chain", "gram-schmidt"])
def test_a_layer_matches_the_reference(ref: ModuleType, conv: bool, gram_schmidt: bool) -> None:
    library = adapter_library()
    lucid.manual_seed(5)
    if conv:
        net = nn.Sequential(nn.Conv2d(3, 5, 3, padding=1))
        twin = ref.nn.Sequential(ref.nn.Conv2d(3, 5, 3, padding=1))
        x = lucid.randn(2, 3, 6, 6)
    else:
        net = nn.Sequential(nn.Linear(8, 6))
        twin = ref.nn.Sequential(ref.nn.Linear(8, 6))
        x = lucid.randn(4, 8)
    options = {"r": 4, "apply_GS": gram_schmidt, "init_weights": False}
    peft = get_peft_model(net, HRAConfig(target_modules=["0"], **options))  # type: ignore[arg-type]
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    theirs = library.get_peft_model(twin, library.HRAConfig(target_modules=["0"], **options))
    other = theirs.base_model.model[0]
    with ref.no_grad():
        other.hra_u["default"].copy_(ref.tensor(mine.hra_u["default"].tolist()))
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    assert gap(ref, mine.get_delta_weight("default"), other.get_delta_weight("default")) < 1e-5
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5
    peft.base_model.unmerge_adapter()
    theirs.base_model.unmerge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = HRAConfig(r=4, target_modules=["a"]).to_dict()
    theirs = library.HRAConfig(r=4, target_modules=["a"]).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
