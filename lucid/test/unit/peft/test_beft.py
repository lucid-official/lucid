"""BEFT: the bias in closed form, the starts, merging into the bias, a base without one, and the reference library's numbers.

The paper's layer is ``y = W x + b + beta`` with one vector ``beta`` of the output width.  The tests
state that sum with plain tensors, never through the adapter's own code, and expect a merged layer to
hold ``b + beta`` and an unmerged one to hold ``b`` again, bit for bit (the bias is stashed).  The
``parity`` tests build the same layer in the reference library, copy ``beta`` across and compare the
forward pass, the merged bias and the shape of the stored tensor.
"""

import types
import warnings
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import BeftConfig, BeftModel, get_peft_model
from lucid.peft.tuners import beft
from lucid.test.unit.peft._parity import adapter_library, gap
from lucid.test.unit.peft._tuner_recipes import magnitude, same, span

#: Relative to the largest value compared: one float32 sum.
_TOLERANCE = 1e-6


def _layer(bias: bool = True, **kw: object) -> tuple[beft.Linear, nn.Linear]:
    lucid.manual_seed(1)
    base = nn.Linear(4, 6, bias=bias)
    fields: dict[str, object] = {"target_modules": ["x"]} | kw
    return beft.Linear(base, "default", BeftConfig(**fields)), base  # type: ignore[arg-type]


def _set(layer: beft.BeftLayer, adapter: str, value: lucid.Tensor) -> None:
    with lucid.no_grad():
        layer.beft_bias[adapter].copy_(value.reshape(1, -1))


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = BeftConfig(target_modules=["a"])
        assert config.init_weights is True and config.modules_to_save is None

    def test_a_field_of_the_wrong_type_is_refused(self) -> None:
        with pytest.raises(TypeError, match="init_weights"):
            BeftConfig(init_weights="yes", target_modules=["a"])  # type: ignore[arg-type]

    def test_a_file_round_trips(self) -> None:
        config = BeftConfig(target_modules=["q", "v"], init_weights=False)
        assert BeftConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "BEFT"


class TestStarts:
    def test_the_default_start_is_the_base_bit_for_bit(self, device: str) -> None:
        layer, base = _layer()
        layer.to(device)
        x = lucid.randn(5, 4, device=device)
        assert same(layer(x), base(x))

    def test_the_bias_is_one_row_of_the_output_width(self) -> None:
        layer, _ = _layer()
        beta = layer.beft_bias["default"]
        assert tuple(beta.shape) == (1, 6) and bool((beta == 0).all().item())

    def test_a_random_start_is_a_standard_normal_draw(self) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(4, 2000)
        layer = beft.Linear(base, "default", BeftConfig(init_weights=False, target_modules=["x"]))
        beta = layer.beft_bias["default"].detach()
        assert abs(float(beta.mean().item())) < 0.1
        assert abs(float(beta.std().item()) - 1.0) < 0.1


class TestBias:
    def test_the_layer_adds_the_bias_to_the_base_output(self, device: str) -> None:
        layer, base = _layer()
        layer.to(device)
        lucid.manual_seed(2)
        beta = lucid.randn(6, device=device)
        _set(layer, "default", beta)
        x = lucid.randn(5, 4, device=device)
        with lucid.no_grad():
            expected = x @ base.weight.mT + base.bias + beta
            assert span(layer(x), expected) <= _TOLERANCE * magnitude(expected)

    def test_the_bias_reaches_every_row_of_a_batch(self) -> None:
        layer, base = _layer()
        beta = lucid.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
        _set(layer, "default", beta)
        x = lucid.randn(2, 3, 4)  # a sequence batch: the bias broadcasts over both leading axes
        with lucid.no_grad():
            expected = base(x) + beta
            assert span(layer(x), expected) <= _TOLERANCE * magnitude(expected)

    def test_active_adapters_add_their_biases(self) -> None:
        net = nn.Sequential(nn.Linear(4, 6))
        config = BeftConfig(target_modules=["0"], init_weights=False)
        peft = get_peft_model(net, config)
        peft.add_adapter("b", config)
        peft.set_adapter(["default", "b"])
        layer = peft.base_model.model[0]
        x = lucid.randn(5, 4)
        with lucid.no_grad():
            both = (
                layer.get_base_layer()(x)
                + layer.get_delta_bias("default")
                + layer.get_delta_bias("b")
            )
            assert span(peft(x), both) <= _TOLERANCE * magnitude(both)

    def test_the_change_of_the_weight_is_zero(self) -> None:
        layer, base = _layer()
        _set(layer, "default", lucid.ones(6))
        delta = layer.get_delta_weight("default")
        assert tuple(delta.shape) == tuple(base.weight.shape) and not bool(delta.any().item())

    def test_only_the_bias_gets_a_gradient(self) -> None:
        net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
        peft = get_peft_model(net, BeftConfig(target_modules=["0", "2"]))
        peft(lucid.randn(3, 4)).sum().backward()
        for name, p in peft.named_parameters():
            if p.requires_grad:
                assert "beft_bias" in name and p.grad is not None, name
            else:
                assert p.grad is None, name

    def test_a_bias_gradient_is_the_sum_of_the_output_gradients(self) -> None:
        layer, _ = _layer()
        x = lucid.randn(7, 4)
        layer(x).sum().backward()
        grad = layer.beft_bias["default"].grad
        assert grad is not None
        assert span(grad.reshape(-1), lucid.full((6,), 7.0)) <= 1e-6


class TestMerging:
    def test_merging_adds_the_bias_to_the_base_bias(self) -> None:
        layer, base = _layer()
        beta = lucid.tensor([0.5, -1.0, 2.0, 0.25, -0.75, 1.5])
        _set(layer, "default", beta)
        old = base.bias.clone()
        weight = base.weight.clone()
        layer.merge()
        assert span(base.bias, old + beta) <= _TOLERANCE * magnitude(old + beta)
        assert same(base.weight, weight)

    def test_unmerging_gives_the_bias_back_bit_for_bit(self) -> None:
        layer, base = _layer()
        lucid.manual_seed(3)
        _set(layer, "default", 1e3 * lucid.randn(6))  # large enough that a subtraction would round
        old = base.bias.clone()
        layer.merge()
        layer.unmerge()
        assert same(base.bias, old)

    def test_two_adapters_merge_to_the_sum_in_either_order(self) -> None:
        net = nn.Sequential(nn.Linear(4, 6))
        config = BeftConfig(target_modules=["0"], init_weights=False)
        peft = get_peft_model(net, config)
        peft.add_adapter("b", config)
        layer = peft.base_model.model[0]
        old = layer.get_base_layer().bias.clone()
        total = old + layer.get_delta_bias("default") + layer.get_delta_bias("b")
        peft.set_adapter(["b", "default"])
        peft.base_model.merge_adapter()
        assert span(layer.get_base_layer().bias, total) <= _TOLERANCE * magnitude(total)
        peft.base_model.unmerge_adapter()
        assert same(layer.get_base_layer().bias, old)

    def test_a_half_precision_base_keeps_its_dtype_and_is_restored_exactly(self) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(4, 6).to(lucid.float16)
        layer = beft.Linear(base, "default", BeftConfig(init_weights=False, target_modules=["x"]))
        assert layer.beft_bias["default"].dtype == lucid.float32
        old = base.bias.clone()
        layer.merge()
        assert base.bias.dtype == lucid.float16
        layer.unmerge()
        assert same(base.bias, old)


class TestWithoutABias:
    def test_the_adapter_still_runs_and_the_layer_warns_that_it_cannot_merge(self) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(4, 6, bias=False)
        with pytest.warns(UserWarning, match="no bias term"):
            layer = beft.Linear(base, "default", BeftConfig(target_modules=["x"]))
        beta = lucid.randn(6)
        _set(layer, "default", beta)
        x = lucid.randn(5, 4)
        with lucid.no_grad():
            expected = x @ base.weight.mT + beta
            assert span(layer(x), expected) <= _TOLERANCE * magnitude(expected)

    def test_merging_is_refused_and_changes_nothing(self) -> None:
        base = nn.Linear(4, 6, bias=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            layer = beft.Linear(base, "default", BeftConfig(target_modules=["x"]))
        with pytest.raises(RuntimeError, match="no bias"):
            layer.merge()
        assert not layer.merged and layer.merged_adapters == []


class TestTargets:
    def test_the_default_is_the_value_projection_of_the_types_the_table_covers(self) -> None:
        for model_type, name in (("bert", "value"), ("t5", "v")):
            net = nn.Sequential()
            net.add_module("query", nn.Linear(4, 4))
            net.add_module(name, nn.Linear(4, 4))
            net.config = types.SimpleNamespace(model_type=model_type)  # type: ignore[assignment]
            peft = BeftModel(net, BeftConfig(), "default")
            assert peft.targeted_module_names == [name]

    def test_a_model_type_the_table_lacks_must_name_its_modules(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4))
        net.config = types.SimpleNamespace(model_type="resnet")  # type: ignore[assignment]
        with pytest.raises(ValueError, match="target_modules"):
            BeftModel(net, BeftConfig(), "default")

    def test_only_a_linear_is_adapted(self) -> None:
        net = nn.Sequential(nn.Conv2d(2, 4, 3))
        with pytest.raises(ValueError, match="not supported"):
            get_peft_model(net, BeftConfig(target_modules=["0"]))


def test_the_tuner_wraps_the_linear_layers_and_unloads_them() -> None:
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    peft = get_peft_model(net, BeftConfig(target_modules=["0", "2"]))
    assert isinstance(peft.base_model, BeftModel)
    assert type(net[0]) is beft.Linear and type(net[2]) is beft.Linear
    assert peft.unload() is net and type(net[0]) is nn.Linear


# ── the reference adapter library ────────────────────────────────────────────


@pytest.mark.parity
def test_a_layer_matches_the_reference(ref: ModuleType) -> None:
    library = adapter_library()
    lucid.manual_seed(5)
    net = nn.Sequential(nn.Linear(6, 10))
    twin = ref.nn.Sequential(ref.nn.Linear(6, 10))
    x = lucid.randn(4, 6)
    peft = get_peft_model(net, BeftConfig(target_modules=["0"], init_weights=False))
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    theirs = library.get_peft_model(twin, library.BeftConfig(target_modules=["0"]))
    other = theirs.base_model.model[0]
    assert tuple(other.beft_bias["default"].shape) == tuple(mine.beft_bias["default"].shape)
    with ref.no_grad():
        other.beft_bias["default"].copy_(ref.tensor(mine.beft_bias["default"].tolist()))
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-6
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.bias, other.base_layer.bias) < 1e-6
    peft.base_model.unmerge_adapter()
    theirs.base_model.unmerge_adapter()
    assert gap(ref, mine.base_layer.bias, other.base_layer.bias) < 1e-6


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = BeftConfig(target_modules=["a"], init_weights=False).to_dict()
    theirs = library.BeftConfig(target_modules=["a"], init_weights=False).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
