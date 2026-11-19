"""Cancellation in the incomplete-gamma prefactor and Temme expansion."""

import decimal

import pytest

import lucid
from lucid._ops.composite._gamma import _deviance
from lucid.test._fixtures.devices import metal_available


@pytest.mark.parametrize(
    ("device", "dtype", "epsilon"),
    [
        ("cpu", lucid.float32, 2.0**-23),
        ("cpu", lucid.float64, 2.0**-52),
        pytest.param(
            "metal",
            lucid.float32,
            2.0**-23,
            marks=pytest.mark.skipif(not metal_available(), reason="requires Metal"),
        ),
    ],
)
def test_small_gamma_deviances_survive_rounding(
    device: str, dtype: lucid.dtype, epsilon: float
) -> None:
    magnitudes = [0.0, 2.0**-40, 1e-8, 4.5e-4, 0.049, 0.25]
    x = lucid.tensor([sign * a for a in magnitudes for sign in (-1, 1)], device=device, dtype=dtype)
    values = _deviance(x).tolist()
    with decimal.localcontext(prec=90):
        for operand, got in zip(x.tolist(), values, strict=True):
            d = decimal.Decimal.from_float(operand)
            want = float(2 * (d - (1 + d).ln()))
            budget = (4 if abs(operand) < 0.05 else 20) * epsilon * abs(want)
            assert abs(got - want) <= budget, (operand, got, want)
