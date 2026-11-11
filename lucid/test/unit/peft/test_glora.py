"""GLoRA: the five support tensors in closed form, the merge that keeps the base, the starts and the reference adapter library's numbers.

GLoRA (Chavan et al., 2023) corrects a linear layer as ``W' = W + W * A + B`` and
``b' = b + b * D + E + W C`` (Eq. 13, 14), each of ``A .. E`` being a low-rank product, a vector, a
scalar or absent.  The tests expect those corrections from plain loops (``_delta_forms``) for each
form, expect the merged weight and bias to be the sums of Eq. 13 and 14 read from the stashed
originals and unmerging to give both back bit for bit; the ``parity`` tests build the same layer in
the reference library and compare.
"""

from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import GloraConfig, GloraModel, get_peft_model, get_peft_model_state_dict
from lucid.peft.tuners import glora
from lucid.test.unit.peft import _delta_forms as forms
from lucid.test.unit.peft._parity import adapter_library, gap

_FORMS = {
    "default": {},
    "vector": {"config_A_B": "vector", "config_C": "vector", "config_D_E": "vector"},
    "scalar-and-none": {"config_A_B": "constant", "config_C": "none", "config_D_E": "none"},
    "none-and-lora": {"config_A_B": "none", "config_C": "lora", "config_D_E": "constant"},
}


def _config(**kw: object) -> GloraConfig:
    fields: dict[str, object] = {"r": 3, "target_modules": ["x"]} | kw
    return GloraConfig.from_dict(fields)


def _layer(bias: bool = True, seed: int = 1, **kw: object) -> tuple[glora.Linear, nn.Linear]:
    lucid.manual_seed(seed)
    base = nn.Linear(6, 4, bias=bias)
    return glora.Linear(base, "default", _config(**kw)), base


def _randomise(layer: glora.GloraLayer, seed: int = 2) -> None:
    """Every parameter and buffer of the supports non-zero (an absent support's buffer too, as a file may fill it)."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for name in layer.adapter_layer_names:
            for tensor in (
                *getattr(layer, name)["default"].parameters(),
                *getattr(layer, name)["default"].buffers(),
            ):
                tensor.copy_(lucid.randn(*tensor.shape))


def _support(layer: glora.GloraLayer, name: str) -> forms.Support:
    path = getattr(layer, name)["default"]
    state = {key: value.tolist() for key, value in path.state_dict().items()}
    if "Xd" in state:
        return "lora", [state["Xd"], state["Xu"]]
    if "X" in state:
        return ("constant" if tuple(path.X.shape) == (1,) else "vector"), [state["X"]]
    return "none", [state["zeros"]]


def _expected(layer: glora.GloraLayer, base: nn.Linear) -> tuple[forms.Matrix, list[float] | None]:
    bias = None if base.bias is None else base.bias.tolist()
    return forms.glora_changes(
        base.weight.tolist(), bias, *(_support(layer, f"glora_{name}") for name in "ABCDE")
    )


def _span(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = GloraConfig(target_modules=["a"])
        assert (config.r, config.config_A_B, config.config_C, config.config_D_E) == (
            8,
            "lora",
            "lora",
            "constant",
        )
        assert (config.bias, config.init_weights) == ("none", True)

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            GloraConfig(r=0, target_modules=["a"])
        with pytest.raises(ValueError, match="config_C"):
            GloraConfig(config_C="constant", target_modules=["a"])
        with pytest.raises(ValueError, match="config_D_E"):
            GloraConfig(config_D_E="lora", target_modules=["a"])

    def test_trained_biases_are_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="glora_only"):
            GloraConfig(bias="glora_only", target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = GloraConfig(r=4, config_A_B="vector", config_C="none", target_modules=["q"])
        assert GloraConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "GLORA"


class TestSupports:
    def test_each_form_has_the_references_tensors_and_shapes(self) -> None:
        layer, _ = _layer(r=3)
        shapes = {
            name: {
                k: tuple(v.shape) for k, v in getattr(layer, name)["default"].state_dict().items()
            }
            for name in layer.adapter_layer_names
        }
        assert shapes == {
            "glora_A": {"Xd": (4, 3), "Xu": (3, 6)},
            "glora_B": {"Xd": (4, 3), "Xu": (3, 6)},
            "glora_C": {"Xd": (6, 3), "Xu": (3, 1)},
            "glora_D": {"X": (1,)},
            "glora_E": {"X": (1,)},
        }
        layer, _ = _layer(**_FORMS["vector"])
        assert {
            name: {
                k: tuple(v.shape) for k, v in getattr(layer, name)["default"].state_dict().items()
            }
            for name in layer.adapter_layer_names
        } == {
            "glora_A": {"X": (4, 1)},
            "glora_B": {"X": (4, 1)},
            "glora_C": {"X": (6, 1)},
            "glora_D": {"X": (4,)},
            "glora_E": {"X": (4,)},
        }
        layer, _ = _layer(**_FORMS["scalar-and-none"])
        assert {k: tuple(v.shape) for k, v in layer.glora_A["default"].state_dict().items()} == {
            "X": (1,)
        }
        assert {k: tuple(v.shape) for k, v in layer.glora_C["default"].state_dict().items()} == {
            "zeros": (6, 1)
        }
        assert {k: tuple(v.shape) for k, v in layer.glora_D["default"].state_dict().items()} == {
            "zeros": (4,)
        }

    def test_a_layer_without_a_bias_has_no_bias_terms_whatever_the_config_asks(self) -> None:
        layer, _ = _layer(bias=False)
        for name in ("glora_C", "glora_D", "glora_E"):
            assert list(getattr(layer, name)["default"].state_dict()) == ["zeros"]
            assert not list(getattr(layer, name)["default"].parameters())
        assert [p.shape for p in layer._adapter_parameters("default")] == [(4, 3), (3, 6)] * 2

    def test_the_trainable_parameters_are_the_supports_that_are_present(self) -> None:
        layer, _ = _layer(**_FORMS["scalar-and-none"])
        assert sorted(tuple(p.shape) for p in layer._adapter_parameters("default")) == [(1,), (1,)]


class TestStart:
    @pytest.mark.parametrize("name", list(_FORMS))
    @pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
    def test_the_adapter_starts_as_the_identity_bit_for_bit(
        self, name: str, bias: bool, device: str
    ) -> None:
        layer, base = _layer(bias=bias, **_FORMS[name])
        layer.to(device)
        x = lucid.randn(5, 6, device=device)
        assert _same(layer(x), base(x))
        assert float(layer.get_delta_weight("default").abs().max().item()) == 0.0

    def test_a_random_start_is_not_the_base(self) -> None:
        layer, base = _layer(init_weights=False)
        x = lucid.randn(5, 6)
        assert _span(layer(x), base(x)) > 1e-3


class TestFormula:
    @pytest.mark.parametrize("name", list(_FORMS))
    @pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
    def test_the_corrections_are_the_papers(self, name: str, bias: bool, device: str) -> None:
        layer, base = _layer(bias=bias, **_FORMS[name])
        layer.to(device)
        base.to(device)
        _randomise(layer)
        delta_w, delta_b = _expected(layer, base)
        assert bool(
            lucid.allclose(
                layer.get_delta_weight("default"), lucid.tensor(delta_w, device=device), atol=1e-5
            )
        )
        # The forward pass: the zero input shows the bias correction, the identity the weight's.
        zero = lucid.zeros(1, 6, device=device)
        eye = lucid.eye(6, device=device)
        shift = layer(zero) - base(zero)
        applied = layer(eye) - base(eye)
        expected_shift = lucid.tensor([delta_b if delta_b else [0.0] * 4], device=device)
        assert bool(lucid.allclose(shift, expected_shift, atol=1e-5))
        assert bool(
            lucid.allclose(applied - shift, lucid.tensor(delta_w, device=device).mT, atol=1e-5)
        )

    def test_the_bias_is_coupled_to_the_weight_through_c(self) -> None:
        # With every support but C absent, the bias moves by W C and nothing else.
        layer, base = _layer(config_A_B="none", config_D_E="none")
        lucid.manual_seed(2)
        with lucid.no_grad():
            for tensor in layer.glora_C["default"].parameters():
                tensor.copy_(lucid.randn(*tensor.shape))
        column = lucid.matmul(layer.glora_C["default"].Xd, layer.glora_C["default"].Xu)
        shift = layer(lucid.zeros(1, 6)) - base(lucid.zeros(1, 6))
        assert bool(lucid.allclose(shift[0], (base.weight @ column).squeeze(-1), atol=1e-5))

    def test_a_scalar_scales_the_whole_weight(self) -> None:
        # A = a constant: W' = (1 + a) W.
        layer, base = _layer(config_A_B="constant", config_C="none", config_D_E="none")
        with lucid.no_grad():
            layer.glora_A["default"].X.copy_(lucid.tensor([0.25]))
        x = lucid.randn(5, 6)
        out = layer(x)
        expected = (1.25 * x) @ base.weight.mT + base.bias
        assert bool(lucid.allclose(out, expected, atol=1e-5))


class TestMerging:
    def test_the_merged_layer_is_eq_13_and_14(self) -> None:
        layer, base = _layer()
        _randomise(layer)
        delta_w, delta_b = _expected(layer, base)
        weight, bias = base.weight.tolist(), base.bias.tolist()
        layer.merge()
        assert bool(
            lucid.allclose(base.weight, lucid.tensor(forms.add(weight, delta_w)), atol=1e-5)
        )
        assert delta_b is not None
        assert bool(
            lucid.allclose(
                base.bias,
                lucid.tensor([b + d for b, d in zip(bias, delta_b, strict=True)]),
                atol=1e-5,
            )
        )

    def test_the_merged_layer_is_the_adapted_one(self, device: str) -> None:
        layer, _ = _layer()
        layer.to(device)
        _randomise(layer)
        x = lucid.randn(5, 6, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert bool(lucid.allclose(layer(x), adapted, atol=1e-5))

    @pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
    def test_unmerging_gives_the_weight_and_the_bias_back_bit_for_bit(
        self, bias: bool, device: str
    ) -> None:
        layer, base = _layer(bias=bias)
        layer.to(device)
        _randomise(layer)
        weight = base.weight.clone()
        held = None if base.bias is None else base.bias.clone()
        layer.merge()
        assert _span(base.weight, weight) > 1e-3
        layer.unmerge()
        assert _same(base.weight, weight)
        if held is not None:
            assert base.bias is not None and _same(base.bias, held)

    def test_two_adapters_are_each_read_from_the_original(self) -> None:
        layer, base = _layer()
        layer.update_layer("b", _config(r=2, config_A_B="vector"))
        for name, seed in (("default", 7), ("b", 8)):
            lucid.manual_seed(seed)
            with lucid.no_grad():
                for tensor in (
                    *layer.glora_A[name].parameters(),
                    *layer.glora_B[name].parameters(),
                    *layer.glora_C[name].parameters(),
                    *layer.glora_D[name].parameters(),
                    *layer.glora_E[name].parameters(),
                ):
                    tensor.copy_(lucid.randn(*tensor.shape))
        layer.set_adapter(["default", "b"])
        weight, bias = base.weight.clone(), base.bias.clone()
        x = lucid.randn(5, 6)
        with lucid.no_grad():
            both = layer(x)
            for order in (["default", "b"], ["b", "default"]):
                layer.merge(adapter_names=order)
                assert bool(lucid.allclose(layer(x), both, atol=1e-4))
                merged_weight = base.weight.clone()
                layer.unmerge()
                assert _same(base.weight, weight) and _same(base.bias, bias)
            expected = weight + layer.get_delta_weight("default") + layer.get_delta_weight("b")
            assert bool(lucid.allclose(merged_weight, expected, atol=1e-5))

    def test_merge_and_unload_gives_the_corrected_model(self, device: str) -> None:
        lucid.manual_seed(1)
        net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)
        peft = get_peft_model(net, GloraConfig(r=2, init_weights=False, target_modules=["0", "2"]))
        x = lucid.randn(5, 6, device=device)
        with lucid.no_grad():
            adapted = peft(x)
            merged = peft.merge_and_unload()
            assert merged is net and type(net[0]) is nn.Linear
            assert bool(lucid.allclose(net(x), adapted, atol=1e-4))


class TestFiles:
    def test_the_empty_supports_are_saved_as_the_reference_saves_them(self) -> None:
        net = nn.Sequential(nn.Linear(6, 4, bias=False))
        peft = get_peft_model(net, GloraConfig(r=2, target_modules=["0"]))
        keys = sorted(get_peft_model_state_dict(peft))
        assert "base_model.model.0.glora_C.zeros" in keys
        assert "base_model.model.0.glora_D.zeros" in keys
        assert "base_model.model.0.glora_A.Xd" in keys

    def test_a_deleted_adapter_leaves_nothing_behind(self) -> None:
        layer, _ = _layer()
        layer.update_layer("other", _config())
        layer.delete_adapter("other")
        for name in layer.adapter_layer_names:
            assert list(getattr(layer, name)) == ["default"]


class TestErrors:
    def test_a_module_that_is_not_linear_is_refused(self) -> None:
        net = nn.Sequential(nn.Conv2d(2, 2, 1))
        with pytest.raises(ValueError, match="not supported"):
            GloraModel(net, GloraConfig(target_modules=["0"]), "default")


class TestHalfPrecision:
    @pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["float16", "bfloat16"])
    def test_the_adapter_is_float32_beside_a_half_base(self, dtype: lucid.dtype) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(6, 4).to(dtype)
        layer = glora.Linear(base, "default", _config(init_weights=False))
        assert {p.dtype for p in layer._adapter_parameters("default")} == {lucid.float32}
        out = layer(lucid.randn(3, 6).to(dtype))
        assert out.dtype == dtype and bool(lucid.isfinite(out).all().item())
        weight, bias = base.weight.clone(), base.bias.clone()
        layer.merge()
        layer.unmerge()
        assert _same(base.weight, weight) and _same(base.bias, bias)


# ── the reference adapter library ────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_FORMS))
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
def test_a_layer_matches_the_reference(ref: ModuleType, name: str, bias: bool) -> None:
    library = adapter_library()
    lucid.manual_seed(5)
    net = nn.Sequential(nn.Linear(8, 6, bias=bias))
    twin = ref.nn.Sequential(ref.nn.Linear(8, 6, bias=bias))
    x = lucid.randn(4, 8)
    options = {"r": 3, "init_weights": False, **_FORMS[name]}
    peft = get_peft_model(net, GloraConfig.from_dict({"target_modules": ["0"], **options}))
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        if bias:
            twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    theirs = library.get_peft_model(twin, library.GloraConfig(target_modules=["0"], **options))
    ours = get_peft_model_state_dict(peft)
    library.set_peft_model_state_dict(theirs, {k: ref.tensor(v.tolist()) for k, v in ours.items()})
    other = theirs.base_model.model[0]
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5
    if bias:
        assert gap(ref, mine.base_layer.bias, other.base_layer.bias) < 1e-5


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = GloraConfig(r=4, target_modules=["a"]).to_dict()
    theirs = library.GloraConfig(r=4, target_modules=["a"]).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
