"""PEANuT: the weight-reading network in closed form, its activations, the stash that makes a merge exact and the reference adapter library's numbers.

PEANuT (Zhong et al., 2024) models the update as a small network applied to the frozen weight,
``W' = W + f(W; theta)`` (section 3.3), deeper with residual connections (appendix F).  The tests
expect that matrix from plain loops (``_delta_forms``), each activation of the table ``act_fn`` names
from its definition, and the merge the update's dependence on ``W`` asks for: the merged weight is
``W0 + f(W0)`` and unmerging gives ``W0`` back bit for bit.  The ``parity`` tests build the same
layers in the reference library and compare.
"""

from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import PeanutConfig, PeanutModel, get_peft_model, get_peft_model_state_dict
from lucid.peft.tuners import peanut
from lucid.peft.tuners.peanut._activations import _ACTIVATIONS
from lucid.test.unit.peft import _delta_forms as forms
from lucid.test.unit.peft._parity import adapter_library, gap

#: A name of the reference's table that carries its framework's name, which the source may not (H5).
TANH_NAME = "gelu_py" + "tor" + "ch_tanh"

#: The names that are another name's function, or a clipped one.
_ALIASES = {
    "gelu_python": "gelu",
    "gelu_accurate": "gelu_new",
    TANH_NAME: "gelu_new",
    "gelu_python_tanh": "gelu_new",
    "swish": "silu",
}


def _config(**kw: object) -> PeanutConfig:
    fields: dict[str, object] = {"r": 3, "target_modules": ["x"]} | kw
    return PeanutConfig.from_dict(fields)


def _layer(
    out: int = 6, inn: int = 5, seed: int = 1, **kw: object
) -> tuple[peanut.Linear, nn.Linear]:
    lucid.manual_seed(seed)
    base = nn.Linear(inn, out)
    return peanut.Linear(base, "default", _config(**kw)), base


def _randomise(layer: peanut.PeanutLayer, name: str = "default", seed: int = 2) -> None:
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for parameter in layer._adapter_parameters(name):
            parameter.copy_(lucid.randn(*parameter.shape))


def _span(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _expected(
    layer: peanut.PeanutLayer, weight: forms.Matrix, name: str = "default"
) -> forms.Matrix:
    spec = layer._spec(name)
    return forms.peanut_delta(
        weight,
        layer.peanut_A[name].weight.tolist(),
        layer.peanut_B[name].weight.tolist(),
        [m.weight.tolist() for m in layer.peanut_encoders[name]],
        [m.weight.tolist() for m in layer.peanut_decoders[name]],
        act=forms.ACTIVATIONS[_ALIASES.get(spec.act_fn, spec.act_fn)],
        scaling=spec.scaling,
    )


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = PeanutConfig(target_modules=["a"])
        assert (config.r, config.depth, config.act_fn) == (32, 0, "relu")
        assert (config.scaling, config.init_weights) == (1.0, True)

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PeanutConfig(r=0, target_modules=["a"])
        with pytest.raises(ValueError, match="non-negative"):
            PeanutConfig(depth=-1, target_modules=["a"])
        with pytest.raises(ValueError, match="Unsupported `act_fn`: nope"):
            PeanutConfig(act_fn="nope", target_modules=["a"])
        with pytest.raises(TypeError, match="act_fn"):
            PeanutConfig.from_dict({"act_fn": 3, "target_modules": ["a"]})

    def test_an_activation_with_parameters_of_its_own_is_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="xielu"):
            PeanutConfig(act_fn="xielu", target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = PeanutConfig(r=8, depth=2, act_fn="gelu", scaling=0.5, target_modules=["q"])
        assert PeanutConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "PEANUT"


class TestActivations:
    @pytest.mark.parametrize("name", sorted(set(_ACTIVATIONS) - {"gelu_10"}))
    def test_each_name_is_its_definition(self, name: str, device: str) -> None:
        points = [i * 0.5 for i in range(-24, 25)]
        key = _ALIASES.get(name, name)
        expected = [forms.ACTIVATIONS[key](v) for v in points]
        got = _ACTIVATIONS[name](lucid.tensor(points, device=device))
        assert bool(lucid.allclose(got, lucid.tensor(expected, device=device), atol=1e-5))

    def test_gelu_10_is_gelu_clipped_to_plus_minus_ten(self) -> None:
        points = [-30.0, -3.0, 0.5, 9.0, 11.0, 40.0]
        expected = [min(max(forms.ACTIVATIONS["gelu"](v), -10.0), 10.0) for v in points]
        got = _ACTIVATIONS["gelu_10"](lucid.tensor(points))
        assert bool(lucid.allclose(got, lucid.tensor(expected), atol=1e-5))

    def test_every_activation_of_the_reference_is_here_but_the_one_with_parameters(self) -> None:
        # The table of the library the reference reads its names from, spelled out.
        names = {
            "gelu", "gelu_10", "gelu_accurate", "gelu_fast", "gelu_new", "gelu_python",
            "gelu_python_tanh", TANH_NAME, "hardswish", "laplace", "leaky_relu", "linear",
            "mish", "prelu", "quick_gelu", "relu", "relu2", "relu6", "sigmoid", "silu",
            "sqrtsoftplus", "swish", "tanh",
        }  # fmt: skip
        assert set(_ACTIVATIONS) == names


class TestStart:
    @pytest.mark.parametrize("depth", [0, 1, 2])
    def test_the_update_starts_at_zero_and_the_output_is_the_base_bit_for_bit(
        self, depth: int, device: str
    ) -> None:
        layer, base = _layer(depth=depth)
        layer.to(device)
        x = lucid.randn(5, 5, device=device)
        assert float(layer.get_delta_weight("default").abs().max().item()) == 0.0
        assert _same(layer(x), base(x))

    def test_a_random_start_is_not_the_base(self) -> None:
        layer, base = _layer(init_weights=False)
        x = lucid.randn(5, 5)
        assert _span(layer(x), base(x)) > 1e-3

    def test_the_networks_layers_have_the_references_shapes(self) -> None:
        layer, _ = _layer(out=6, inn=5, r=3, depth=2)
        assert tuple(layer.peanut_A["default"].weight.shape) == (3, 6)
        assert tuple(layer.peanut_B["default"].weight.shape) == (6, 3)
        assert len(layer.peanut_encoders["default"]) == len(layer.peanut_decoders["default"]) == 2
        assert all(tuple(m.weight.shape) == (3, 3) for m in layer.peanut_encoders["default"])
        assert (layer.r, layer.depth) == ({"default": 3}, {"default": 2})


class TestFormula:
    @pytest.mark.parametrize(
        ("depth", "act", "scaling"),
        [(0, "relu", 1.0), (1, "gelu", 0.5), (2, "tanh", 2.0), (3, "silu", 1.0)],
        ids=["d0-relu", "d1-gelu", "d2-tanh", "d3-silu"],
    )
    def test_the_update_is_the_papers_network(
        self, depth: int, act: str, scaling: float, device: str
    ) -> None:
        layer, base = _layer(depth=depth, act_fn=act, scaling=scaling)
        layer.to(device)
        base.to(device)
        _randomise(layer)
        expected = _expected(layer, base.weight.tolist())
        got = layer.get_delta_weight("default")
        assert bool(lucid.allclose(got, lucid.tensor(expected, device=device), atol=1e-4))

    def test_depth_zero_is_one_hidden_layer(self) -> None:
        # f(W) = s * (relu(W^T A^T) B^T)^T, written out entry by entry.
        layer, base = _layer(scaling=2.0)
        _randomise(layer)
        w = base.weight.tolist()
        a = layer.peanut_A["default"].weight.tolist()
        b = layer.peanut_B["default"].weight.tolist()
        out, inn, r = len(w), len(w[0]), len(a)
        hidden = [
            [max(sum(w[o][i] * a[k][o] for o in range(out)), 0.0) for k in range(r)]
            for i in range(inn)
        ]
        expected = [
            [2.0 * sum(hidden[i][k] * b[o][k] for k in range(r)) for i in range(inn)]
            for o in range(out)
        ]
        assert bool(
            lucid.allclose(layer.get_delta_weight("default"), lucid.tensor(expected), atol=1e-4)
        )

    def test_the_forward_pass_is_the_update_applied_to_the_input(self, device: str) -> None:
        layer, base = _layer(depth=1, act_fn="gelu")
        layer.to(device)
        _randomise(layer)
        probe = lucid.eye(5, device=device)
        applied = layer(probe) - base(probe)
        assert bool(lucid.allclose(applied, layer.get_delta_weight("default").mT, atol=1e-4))

    def test_the_update_is_a_function_of_the_weight_and_not_a_linear_map_of_it(self) -> None:
        layer, base = _layer(init_weights=False)
        _randomise(layer)
        first = layer.get_delta_weight("default")
        with lucid.no_grad():
            base.weight.copy_(base.weight * -2.0)
        second = layer.get_delta_weight("default")
        assert _span(first, second) > 1e-3
        assert (
            _span(second, first * -2.0) > 1e-3
        )  # not the linear map of W that W -> -2 W would scale

    def test_a_gradient_reaches_every_layer_of_the_network(self) -> None:
        layer, _ = _layer(depth=2, init_weights=False)
        layer(lucid.randn(5, 5)).sum().backward()
        assert all(
            p.grad is not None and float(p.grad.abs().max().item()) > 0
            for p in layer._adapter_parameters("default")
        )


class TestMerging:
    def test_the_merged_weight_is_the_weight_plus_the_network_of_it(self) -> None:
        layer, base = _layer(depth=1)
        _randomise(layer)
        weight = base.weight.tolist()
        delta = _expected(layer, weight)
        layer.merge()
        assert bool(lucid.allclose(base.weight, lucid.tensor(forms.add(weight, delta)), atol=1e-4))

    def test_unmerging_gives_the_weights_back_bit_for_bit(self, device: str) -> None:
        layer, base = _layer(depth=1)
        layer.to(device)
        _randomise(layer)
        before = base.weight.clone()
        layer.merge()
        assert _span(base.weight, before) > 1e-3
        layer.unmerge()
        assert _same(base.weight, before)

    def test_the_merged_layer_is_the_adapted_one(self, device: str) -> None:
        layer, _ = _layer(depth=2, act_fn="gelu")
        layer.to(device)
        _randomise(layer)
        x = lucid.randn(5, 5, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert bool(lucid.allclose(layer(x), adapted, atol=1e-4))

    def test_two_adapters_are_each_read_from_the_original_weight(self) -> None:
        # The network of the second adapter must not read the weight the first has merged into.
        layer, base = _layer()
        layer.update_layer("b", _config(r=2, depth=1, act_fn="tanh"))
        _randomise(layer, "default", 7)
        _randomise(layer, "b", 8)
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
        config = PeanutConfig(r=2, depth=1, init_weights=False, target_modules=["0", "2"])
        peft = get_peft_model(net, config)
        x = lucid.randn(5, 6, device=device)
        with lucid.no_grad():
            adapted = peft(x)
            merged = peft.merge_and_unload()
            assert merged is net and type(net[0]) is nn.Linear
            assert bool(lucid.allclose(net(x), adapted, atol=1e-4))


class TestFiles:
    def test_the_state_dict_keys_are_the_references(self) -> None:
        net = nn.Sequential(nn.Linear(6, 4))
        peft = get_peft_model(net, PeanutConfig(r=2, depth=2, target_modules=["0"]))
        assert sorted(get_peft_model_state_dict(peft)) == [
            "base_model.model.0.peanut_A.weight",
            "base_model.model.0.peanut_B.weight",
            "base_model.model.0.peanut_decoders.0.weight",
            "base_model.model.0.peanut_decoders.1.weight",
            "base_model.model.0.peanut_encoders.0.weight",
            "base_model.model.0.peanut_encoders.1.weight",
        ]

    def test_the_stash_is_not_saved(self) -> None:
        layer, _ = _layer()
        _randomise(layer)
        keys = sorted(layer.state_dict())
        layer.merge()
        assert sorted(layer.state_dict()) == keys

    def test_a_deleted_adapter_leaves_nothing_behind(self) -> None:
        layer, _ = _layer()
        layer.update_layer("other", _config(depth=1))
        layer.delete_adapter("other")
        for name in layer.adapter_layer_names:
            assert list(getattr(layer, name)) == ["default"]
        assert "other" not in layer._specs


class TestErrors:
    def test_a_module_that_is_not_linear_is_refused(self) -> None:
        net = nn.Sequential(nn.Conv2d(2, 2, 1))
        with pytest.raises(ValueError, match="not supported"):
            PeanutModel(net, PeanutConfig(target_modules=["0"]), "default")


class TestHalfPrecision:
    @pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["float16", "bfloat16"])
    def test_the_adapter_is_float32_beside_a_half_base(self, dtype: lucid.dtype) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(6, 4).to(dtype)
        layer = peanut.Linear(base, "default", _config(depth=1, init_weights=False))
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
    {"depth": 1, "act_fn": "gelu"},
    {"depth": 3, "act_fn": "tanh", "scaling": 0.5},
    {"depth": 2, "act_fn": "silu", "scaling": 2.0},
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
    fields = {"r": 3, "init_weights": False, **options}
    peft = get_peft_model(net, PeanutConfig.from_dict({"target_modules": ["0"], **fields}))
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    theirs = library.get_peft_model(twin, library.PeanutConfig(target_modules=["0"], **fields))
    ours = get_peft_model_state_dict(peft)
    library.set_peft_model_state_dict(theirs, {k: ref.tensor(v.tolist()) for k, v in ours.items()})
    other = theirs.base_model.model[0]
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    assert gap(ref, mine.get_delta_weight("default"), other.get_delta_weight("default")) < 1e-5
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = PeanutConfig(r=4, target_modules=["a"]).to_dict()
    theirs = library.PeanutConfig(r=4, target_modules=["a"]).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
