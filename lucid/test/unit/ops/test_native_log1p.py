"""The CPU and Metal log1p kernels keep a few ulp across their float inputs."""

import math
import struct

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available


def _bits(value: float, dtype: lucid.dtype) -> int:
    return int.from_bytes(struct.pack(">d" if dtype == lucid.float64 else ">f", value), "big")


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", lucid.float32),
        ("cpu", lucid.float64),
        pytest.param(
            "metal", lucid.float32, marks=pytest.mark.skipif(not metal_available(), reason="Metal")
        ),
    ],
)
def test_log1p_over_the_exponent_range(device: str, dtype: lucid.dtype) -> None:
    tiny = 2.0 ** (-1074 if dtype == lucid.float64 else -149)
    values = [0.0, -0.0, tiny, -tiny, 1e-20, -1e-20]
    values.extend(10.0 ** (k / 20) for k in range(-600, 601))
    values.extend(-(10.0 ** (k / 20)) for k in range(-600, 0))
    values.extend(-1.0 + 2.0**-k for k in range(1, 54))
    x = lucid.tensor(values, dtype=dtype, device=device)
    got = lucid.log1p(x).tolist()
    for operand, value in zip(x.tolist(), got, strict=True):
        if operand == -1.0:
            assert value == -math.inf
        else:
            want = math.log1p(operand)
            assert abs(_bits(value, dtype) - _bits(want, dtype)) <= 4, (operand, value, want)
