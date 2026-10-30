"""The LoRA layer's closed-form contract, on CPU and Metal.

``y = W x + b + s B A x``: zero ``B`` gives the base output bit for bit,
the scale is exactly ``alpha / r`` (``alpha / sqrt(r)`` with rsLoRA),
merging folds the same update into ``W`` in place, and the base keeps its
``Parameter`` objects throughout.
"""

import math

import pytest

import lucid
import lucid.nn as nn
from lucid.peft.tuners.lora import Linear, LoraLayer


def _layer(device: str, **kwargs: object) -> tuple[nn.Linear, Linear]:
    lucid.manual_seed(0)
    base = nn.Linear(6, 5).to(device)
    options: dict[str, object] = {"r": 3, "lora_alpha": 6} | kwargs
    return base, Linear(base, "default", **options)  # type: ignore[arg-type]


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


class TestStructure:
    def test_wraps_the_base_by_identity(self, device: str) -> None:
        base, layer = _layer(device)
        assert layer.base_layer is base and layer.get_base_layer() is base
        assert layer.weight is base.weight and layer.bias is base.bias
        assert isinstance(layer, LoraLayer)
        assert (layer.in_features, layer.out_features) == (6, 5)
        assert layer.r == {"default": 3} and layer.lora_alpha == {"default": 6}
        assert tuple(layer.lora_A["default"].weight.shape) == (3, 6)
        assert tuple(layer.lora_B["default"].weight.shape) == (5, 3)
        assert layer.lora_A["default"].bias is None and layer.lora_B["default"].bias is None
        assert layer.lora_A["default"].weight.device == base.weight.device

    def test_state_dict_keys_are_the_reference_s(self, device: str) -> None:
        _, layer = _layer(device)
        assert sorted(layer.state_dict()) == [
            "base_layer.bias",
            "base_layer.weight",
            "lora_A.default.weight",
            "lora_B.default.weight",
        ]

    def test_repr_marks_the_layer(self, device: str) -> None:
        assert repr(_layer(device)[1]).startswith("lora.Linear(")

    def test_dropout_is_identity_at_zero(self, device: str) -> None:
        assert isinstance(_layer(device)[1].lora_dropout["default"], nn.Identity)
        assert isinstance(_layer(device, lora_dropout=0.1)[1].lora_dropout["default"], nn.Dropout)


class TestForward:
    def test_zero_b_is_the_base_bit_for_bit(self, device: str) -> None:
        base, layer = _layer(device)
        x = lucid.randn(4, 6, device=device)
        expected = base(x)
        assert _same(layer(x), expected)
        assert _same(layer.lora_B["default"].weight, lucid.zeros(5, 3, device=device))

    @pytest.mark.parametrize(("use_rslora", "scale"), [(False, 6 / 3), (True, 6 / math.sqrt(3))])
    def test_the_update_is_scale_times_b_a_x(
        self, device: str, use_rslora: bool, scale: float
    ) -> None:
        base, layer = _layer(device, init_lora_weights=False, use_rslora=use_rslora)
        assert layer.scaling["default"] == scale
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        x = lucid.randn(4, 6, device=device)
        expected = base(x) + (x @ a.mT @ b.mT) * scale
        assert bool(lucid.allclose(layer(x), expected, atol=1e-5))

    def test_disabled_runs_the_base_alone(self, device: str) -> None:
        base, layer = _layer(device, init_lora_weights=False)
        x = lucid.randn(4, 6, device=device)
        layer.enable_adapters(False)
        assert layer.disable_adapters and _same(layer(x), base(x))
        layer.enable_adapters(True)
        assert not layer.disable_adapters and not _same(layer(x), base(x))

    def test_dropout_acts_only_in_training(self, device: str) -> None:
        _, layer = _layer(device, init_lora_weights=False, lora_dropout=0.5)
        x = lucid.randn(64, 6, device=device)
        layer.eval()
        assert _same(layer(x), layer(x))
        layer.train()
        assert not _same(layer(x), layer(x))

    def test_bfloat16_base_gets_a_float32_adapter(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(6, 5).to(device).bfloat16()
        layer = Linear(base, "default", r=2, lora_alpha=2, init_lora_weights=False)
        assert layer.lora_A["default"].weight.dtype == lucid.float32
        x = lucid.randn(4, 6, device=device).bfloat16()
        out = layer(x)
        assert out.dtype == lucid.bfloat16
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        update = x.float() @ a.mT @ b.mT
        expected = (base(x).float() + update).bfloat16()
        assert _same(out, expected)

    def test_a_float32_base_keeps_a_float32_adapter(self, device: str) -> None:
        _, layer = _layer(device)
        assert layer.lora_A["default"].weight.dtype == lucid.float32


class TestInit:
    def test_default_init_is_kaiming_a_and_zero_b(self, device: str) -> None:
        _, layer = _layer(device)
        a = layer.lora_A["default"].weight
        bound = 1 / math.sqrt(6)  # kaiming_uniform(a=sqrt 5) on fan_in 6
        assert float(a.abs().max().item()) <= bound + 1e-6
        assert float(a.abs().max().item()) > 0

    def test_gaussian_init_has_std_one_over_r(self) -> None:
        lucid.manual_seed(0)
        layer = Linear(nn.Linear(4000, 2), "default", r=4, init_lora_weights="gaussian")
        a = layer.lora_A["default"].weight
        assert abs(float(a.std().item()) - 0.25) < 0.01
        assert _same(layer.lora_B["default"].weight, lucid.zeros(2, 4))

    def test_false_leaves_b_non_zero(self, device: str) -> None:
        _, layer = _layer(device, init_lora_weights=False)
        assert float(layer.lora_B["default"].weight.abs().sum().item()) > 0


class TestMerge:
    def test_merged_equals_unmerged_and_keeps_identity(self, device: str) -> None:
        base, layer = _layer(device, init_lora_weights=False)
        weight = base.weight
        x = lucid.randn(4, 6, device=device)
        unmerged = layer(x)
        expected_weight = weight + layer.get_delta_weight("default")
        layer.merge()
        assert layer.merged and layer.merged_adapters == ["default"]
        assert base.weight is weight
        assert bool(lucid.allclose(base.weight, expected_weight, atol=1e-6))
        assert bool(lucid.allclose(layer(x), unmerged, atol=1e-5))

    def test_unmerge_restores_the_weight(self, device: str) -> None:
        base, layer = _layer(device, init_lora_weights=False)
        before = base.weight.clone()
        layer.merge(safe_merge=True)
        layer.unmerge()
        assert not layer.merged
        assert bool(lucid.allclose(base.weight, before, atol=1e-6))

    def test_merging_twice_warns_and_changes_nothing(self, device: str) -> None:
        base, layer = _layer(device, init_lora_weights=False)
        layer.merge()
        once = base.weight.clone()
        with pytest.warns(UserWarning, match="Already merged"):
            layer.merge(adapter_names=["default"])
        assert _same(base.weight, once)

    def test_unmerging_an_unmerged_layer_warns(self, device: str) -> None:
        with pytest.warns(UserWarning, match="Already unmerged"):
            _layer(device)[1].unmerge()

    def test_disabling_a_merged_layer_unmerges_it(self, device: str) -> None:
        base, layer = _layer(device, init_lora_weights=False)
        x = lucid.randn(4, 6, device=device)
        reference = base(x)
        layer.merge()
        layer.enable_adapters(False)
        assert bool(lucid.allclose(layer(x), reference, atol=1e-5)) and not layer.merged

    def test_a_merged_layer_runs_the_base_alone(self, device: str) -> None:
        base, layer = _layer(device, init_lora_weights=False)
        layer.merge()
        x = lucid.randn(4, 6, device=device)
        assert _same(layer(x), base(x))

    def test_safe_merge_refuses_a_non_finite_update(self, device: str) -> None:
        base, layer = _layer(device, init_lora_weights=False)
        before = base.weight.clone()
        with lucid.no_grad():
            layer.lora_B["default"].weight.fill_(math.nan)
        with pytest.raises(ValueError, match="NaNs detected"):
            layer.merge(safe_merge=True)
        assert _same(base.weight, before) and not layer.merged

    def test_an_unknown_adapter_is_refused(self, device: str) -> None:
        with pytest.raises(ValueError, match="unknown adapter"):
            _layer(device)[1].merge(adapter_names=["other"])

    def test_bfloat16_merge_rounds_once(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(6, 5).to(device).bfloat16()
        layer = Linear(base, "default", r=2, init_lora_weights=False)
        expected = (base.weight.float() + layer.get_delta_weight("default")).bfloat16()
        layer.merge()
        assert base.weight.dtype == lucid.bfloat16 and _same(base.weight, expected)


BAD_LAYERS = [
    ({"r": 0}, ValueError, "positive integer"),
    ({"r": 2, "lora_dropout": 1.5}, ValueError, "lora_dropout"),
    ({"r": 2, "fan_in_fan_out": True}, ValueError, "fan_in_fan_out"),
    ({"r": 2, "is_target_conv_1d_layer": True}, ValueError, "Conv1D"),
    ({"r": 2, "init_lora_weights": "olora"}, NotImplementedError, "olora"),
    ({"r": 2, "use_dora": True}, NotImplementedError, "DoRA"),
    ({"r": 2, "lora_alpha": "x"}, TypeError, "lora_alpha"),
]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"), BAD_LAYERS, ids=[str(b[0]) for b in BAD_LAYERS]
)
def test_a_bad_layer_is_refused(
    kwargs: dict[str, object], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        Linear(nn.Linear(3, 3), "default", **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(("name", "error"), [("", ValueError), ("a.b", ValueError), (3, TypeError)])
def test_a_bad_adapter_name_is_refused(name: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="adapter_name"):
        Linear(nn.Linear(3, 3), name, r=1)  # type: ignore[arg-type]


def test_only_a_linear_base_is_accepted() -> None:
    with pytest.raises(TypeError, match="nn.Linear"):
        Linear(nn.Conv1d(2, 2, 1), "default", r=1)  # type: ignore[arg-type]


def test_a_second_adapter_on_one_layer_is_refused() -> None:
    layer = Linear(nn.Linear(3, 3), "default", r=1)
    with pytest.raises(NotImplementedError, match="one adapter per layer"):
        layer.update_layer("second", 1, 1, 0.0, True, False)
    assert layer.active_adapter == "default"


def test_positional_arguments_line_up_with_the_reference() -> None:
    # base_layer, adapter_name, r, lora_alpha, lora_dropout, fan_in_fan_out,
    # is_target_conv_1d_layer, init_lora_weights, use_rslora
    layer = Linear(nn.Linear(4, 3), "default", 2, 8, 0.0, False, False, "gaussian", True)
    assert layer.r == {"default": 2} and layer.scaling["default"] == 8 / 2**0.5
    assert layer.lora_A["default"].weight.abs().max().item() > 1 / 2**0.5 / 2


def test_reset_with_false_leaves_the_weights() -> None:
    layer = Linear(nn.Linear(3, 3), "default", r=1, init_lora_weights=False)
    before = layer.lora_B["default"].weight.clone()
    layer.reset_lora_parameters("default", False)
    assert _same(layer.lora_B["default"].weight, before)
