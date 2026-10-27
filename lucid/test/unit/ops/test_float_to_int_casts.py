"""A float cast to an integer reads the same on every device.

NaN becomes 0 and anything out of range saturates at the type's limits, as
the CPU conversion does.  Metal's own float-to-int64 conversion gave 0 for
±inf and wrapped past the range, so one tensor read different numbers on the
two devices.
"""

import pytest

import lucid

_DEVICES = [
    "cpu",
    pytest.param(
        "metal", marks=pytest.mark.skipif(not lucid.metal.is_available(), reason="no Metal device")
    ),
]
_VALUES = [float("inf"), float("-inf"), float("nan"), 1.5, -2.5, 3e19, -3e19]


@pytest.mark.parametrize("device", _DEVICES)
def test_float_to_int64_saturates(device: str) -> None:
    out = lucid.tensor(_VALUES, device=device).long()
    assert out.tolist() == [2**63 - 1, -(2**63), 0, 1, -2, 2**63 - 1, -(2**63)]


@pytest.mark.parametrize("device", _DEVICES)
def test_float_to_int32_saturates(device: str) -> None:
    out = lucid.tensor(_VALUES, device=device).int()
    assert out.tolist() == [2**31 - 1, -(2**31), 0, 1, -2, 2**31 - 1, -(2**31)]
