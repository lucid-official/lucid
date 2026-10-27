"""Multi-operand mismatch errors name the operand that differs (LCD-562, LCD-585).

Testing three or more operands in a single disjunction and reporting only the
first pair raises "expected cpu, got cpu" or "expected float32, got float32"
when a later operand is the one on another device or dtype.  Every multi-tensor
boundary check names the specific operand pair that disagreed.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available


def _qkv(odd: str, **change: object) -> tuple[lucid.Tensor, ...]:
    lucid.manual_seed(0)
    q, k, v = (lucid.randn(1, 2, 4, 8) for _ in range(3))
    named = {"q": q, "k": k, "v": v}
    named[odd] = named[odd].to(**change)
    return named["q"], named["k"], named["v"]


@pytest.mark.parametrize(("odd", "pair"), [("k", "Q/K"), ("v", "Q/V")])
def test_dtype_mismatch_names_the_odd_operand(odd: str, pair: str) -> None:
    q, k, v = _qkv(odd, dtype=lucid.float64)
    with pytest.raises(TypeError) as err:
        F.scaled_dot_product_attention(q, k, v)
    message = str(err.value)
    assert pair in message, message
    assert "expected float32, got float64" in message, message


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize(("odd", "pair"), [("k", "Q/K"), ("v", "Q/V")])
def test_device_mismatch_names_the_odd_operand(odd: str, pair: str) -> None:
    q, k, v = _qkv(odd, device="metal")
    with pytest.raises(ValueError) as err:
        F.scaled_dot_product_attention(q, k, v)
    message = str(err.value)
    assert pair in message, message
    assert "expected cpu, got metal" in message, message


def _device_cases() -> list[tuple[str, str, Callable[[str], object]]]:
    return [
        (
            "layer_norm",
            "x/weight",
            lambda dev: F.layer_norm(
                lucid.randn(2, 4),
                (4,),
                weight=lucid.ones(4, device=dev),
                bias=lucid.zeros(4),
            ),
        ),
        (
            "layer_norm",
            "x/bias",
            lambda dev: F.layer_norm(
                lucid.randn(2, 4),
                (4,),
                weight=lucid.ones(4),
                bias=lucid.zeros(4, device=dev),
            ),
        ),
        (
            "batch_norm",
            "x/weight",
            lambda dev: F.batch_norm(
                lucid.randn(2, 3, 4, 4),
                None,
                None,
                weight=lucid.ones(3, device=dev),
                bias=lucid.zeros(3),
                training=True,
            ),
        ),
        (
            "batch_norm",
            "x/bias",
            lambda dev: F.batch_norm(
                lucid.randn(2, 3, 4, 4),
                None,
                None,
                weight=lucid.ones(3),
                bias=lucid.zeros(3, device=dev),
                training=True,
            ),
        ),
        (
            "batch_norm_eval",
            "x/running_mean",
            lambda dev: F.batch_norm(
                lucid.randn(2, 3, 4, 4),
                lucid.zeros(3, device=dev),
                lucid.ones(3),
                weight=lucid.ones(3),
                bias=lucid.zeros(3),
                training=False,
            ),
        ),
        (
            "batch_norm_eval",
            "x/running_var",
            lambda dev: F.batch_norm(
                lucid.randn(2, 3, 4, 4),
                lucid.zeros(3),
                lucid.ones(3, device=dev),
                weight=lucid.ones(3),
                bias=lucid.zeros(3),
                training=False,
            ),
        ),
        (
            "batch_norm_eval",
            "x/weight",
            lambda dev: F.batch_norm(
                lucid.randn(2, 3, 4, 4),
                lucid.zeros(3),
                lucid.ones(3),
                weight=lucid.ones(3, device=dev),
                bias=lucid.zeros(3),
                training=False,
            ),
        ),
        (
            "batch_norm_eval",
            "x/bias",
            lambda dev: F.batch_norm(
                lucid.randn(2, 3, 4, 4),
                lucid.zeros(3),
                lucid.ones(3),
                weight=lucid.ones(3),
                bias=lucid.zeros(3, device=dev),
                training=False,
            ),
        ),
        (
            "group_norm",
            "x/weight",
            lambda dev: F.group_norm(
                lucid.randn(2, 4, 3, 3),
                2,
                weight=lucid.ones(4, device=dev),
                bias=lucid.zeros(4),
            ),
        ),
        (
            "group_norm",
            "x/bias",
            lambda dev: F.group_norm(
                lucid.randn(2, 4, 3, 3),
                2,
                weight=lucid.ones(4),
                bias=lucid.zeros(4, device=dev),
            ),
        ),
        (
            "rms_norm",
            "x/weight",
            lambda dev: F.rms_norm(lucid.randn(2, 4), (4,), weight=lucid.ones(4, device=dev)),
        ),
        (
            "instance_norm",
            "x/running_mean",
            lambda dev: F.instance_norm(
                lucid.randn(2, 3, 4, 4),
                running_mean=lucid.zeros(3, device=dev),
                running_var=lucid.ones(3),
                use_input_stats=False,
            ),
        ),
        (
            "instance_norm",
            "x/running_var",
            lambda dev: F.instance_norm(
                lucid.randn(2, 3, 4, 4),
                running_mean=lucid.zeros(3),
                running_var=lucid.ones(3, device=dev),
                use_input_stats=False,
            ),
        ),
        (
            "instance_norm",
            "x/weight",
            lambda dev: F.instance_norm(
                lucid.randn(2, 3, 4, 4),
                weight=lucid.ones(3, device=dev),
                bias=lucid.zeros(3),
            ),
        ),
        (
            "instance_norm",
            "x/bias",
            lambda dev: F.instance_norm(
                lucid.randn(2, 3, 4, 4),
                weight=lucid.ones(3),
                bias=lucid.zeros(3, device=dev),
            ),
        ),
        (
            "conv2d",
            "x/weight",
            lambda dev: F.conv2d(
                lucid.randn(1, 2, 4, 4),
                lucid.randn(3, 2, 3, 3, device=dev),
                lucid.zeros(3),
            ),
        ),
        (
            "conv2d",
            "x/bias",
            lambda dev: F.conv2d(
                lucid.randn(1, 2, 4, 4),
                lucid.randn(3, 2, 3, 3),
                lucid.zeros(3, device=dev),
            ),
        ),
        (
            "conv_transpose2d",
            "x/weight",
            lambda dev: F.conv_transpose2d(
                lucid.randn(1, 2, 4, 4),
                lucid.randn(2, 3, 3, 3, device=dev),
                lucid.zeros(3),
            ),
        ),
        (
            "conv_transpose2d",
            "x/bias",
            lambda dev: F.conv_transpose2d(
                lucid.randn(1, 2, 4, 4),
                lucid.randn(2, 3, 3, 3),
                lucid.zeros(3, device=dev),
            ),
        ),
        (
            "linear",
            "x/weight",
            lambda dev: F.linear(lucid.randn(2, 4), lucid.randn(3, 4, device=dev), lucid.zeros(3)),
        ),
        (
            "linear",
            "x/bias",
            lambda dev: F.linear(lucid.randn(2, 4), lucid.randn(3, 4), lucid.zeros(3, device=dev)),
        ),
        (
            "bilinear_layer",
            "x1/x2",
            lambda dev: F.bilinear(
                lucid.randn(2, 3),
                lucid.randn(2, 4, device=dev),
                lucid.randn(5, 3, 4),
                lucid.zeros(5),
            ),
        ),
        (
            "bilinear_layer",
            "x1/weight",
            lambda dev: F.bilinear(
                lucid.randn(2, 3),
                lucid.randn(2, 4),
                lucid.randn(5, 3, 4, device=dev),
                lucid.zeros(5),
            ),
        ),
        (
            "bilinear_layer",
            "x1/bias",
            lambda dev: F.bilinear(
                lucid.randn(2, 3),
                lucid.randn(2, 4),
                lucid.randn(5, 3, 4),
                lucid.zeros(5, device=dev),
            ),
        ),
        (
            "where",
            "x/condition",
            lambda dev: lucid.where(
                lucid.tensor([True, False], device=dev),
                lucid.ones(2),
                lucid.zeros(2),
            ),
        ),
        (
            "where",
            "x/y",
            lambda dev: lucid.where(
                lucid.tensor([True, False]),
                lucid.ones(2),
                lucid.zeros(2, device=dev),
            ),
        ),
        (
            "lerp",
            "input/end",
            lambda dev: lucid.ones(2).lerp(lucid.zeros(2, device=dev), lucid.full((2,), 0.5)),
        ),
        (
            "lerp",
            "input/weight",
            lambda dev: lucid.ones(2).lerp(lucid.zeros(2), lucid.full((2,), 0.5, device=dev)),
        ),
        (
            "addcmul",
            "input/tensor1",
            lambda dev: lucid.addcmul(lucid.ones(2), lucid.ones(2, device=dev), lucid.ones(2)),
        ),
        (
            "addcmul",
            "input/tensor2",
            lambda dev: lucid.addcmul(lucid.ones(2), lucid.ones(2), lucid.ones(2, device=dev)),
        ),
        (
            "addcdiv",
            "input/tensor1",
            lambda dev: lucid.addcdiv(lucid.ones(2), lucid.ones(2, device=dev), lucid.ones(2)),
        ),
        (
            "addcdiv",
            "input/tensor2",
            lambda dev: lucid.addcdiv(lucid.ones(2), lucid.ones(2), lucid.ones(2, device=dev)),
        ),
    ]


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize(
    ("op", "pair", "call"),
    _device_cases(),
    ids=[f"{op}-{pair.replace('/', '_')}" for op, pair, _ in _device_cases()],
)
def test_multi_operand_device_mismatch_names_offending_pair(
    op: str, pair: str, call: Callable[[str], object]
) -> None:
    with pytest.raises(ValueError) as err:
        call("metal")
    message = str(err.value)
    assert op in message, message
    assert pair in message, message
    assert "expected cpu, got metal" in message, message


@pytest.mark.parametrize(
    ("op", "pair", "call"),
    [
        (
            "bilinear_layer",
            "x1/x2",
            lambda dt: F.bilinear(
                lucid.randn(2, 3),
                lucid.randn(2, 4, dtype=dt),
                lucid.randn(5, 3, 4),
                lucid.zeros(5),
            ),
        ),
        (
            "bilinear_layer",
            "x1/weight",
            lambda dt: F.bilinear(
                lucid.randn(2, 3),
                lucid.randn(2, 4),
                lucid.randn(5, 3, 4, dtype=dt),
                lucid.zeros(5),
            ),
        ),
        (
            "bilinear_layer",
            "x1/bias",
            lambda dt: F.bilinear(
                lucid.randn(2, 3),
                lucid.randn(2, 4),
                lucid.randn(5, 3, 4),
                lucid.zeros(5, dtype=dt),
            ),
        ),
        (
            "lerp",
            "input/end",
            lambda dt: lucid.ones(2).lerp(lucid.zeros(2, dtype=dt), lucid.full((2,), 0.5)),
        ),
        (
            "lerp",
            "input/weight",
            lambda dt: lucid.ones(2).lerp(lucid.zeros(2), lucid.full((2,), 0.5, dtype=dt)),
        ),
    ],
)
def test_multi_operand_dtype_mismatch_names_offending_pair(
    op: str, pair: str, call: Callable[[lucid.dtype], object]
) -> None:
    with pytest.raises(TypeError) as err:
        call(lucid.float64)
    message = str(err.value)
    assert op in message, message
    assert pair in message, message
    assert "expected float32, got float64" in message, message
