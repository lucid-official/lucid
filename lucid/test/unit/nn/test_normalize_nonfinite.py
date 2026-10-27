"""``F.normalize`` on a row that contains a non-finite value.

The norm of a vector with a NaN in it is NaN, so every output is NaN.
The CPU kernel said otherwise:

    normalize([nan, inf, -inf, -1])
      cpu       [nan, inf, -inf, -1e12]
      metal     [nan, nan,  nan,   nan]
      reference [nan, nan,  nan,   nan]

The denominator was ``max(‖x‖, eps)`` written as a comparison —
``nm > eps ? nm : eps`` — and every comparison against NaN is false, so
the ternary quietly took the other branch and divided by ``1e-12``.  The
``-1e12`` is that division, and it is the kind of number that propagates
into a loss and never explains itself.

``std::fmax`` would not have helped: it is specified to return the
non-NaN operand, which is the same swallowing by another route.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available


def _out(arr: np.ndarray, device: str = "cpu") -> np.ndarray:
    return np.asarray(F.normalize(lucid.tensor(arr, device=device), dim=-1).numpy()).ravel()


@pytest.mark.parametrize(
    "row", [[np.nan, np.inf, -np.inf, -1.0], [np.nan, 1.0, 2.0, 3.0], [1.0, 2.0, np.nan, 4.0]]
)
def test_a_nan_anywhere_makes_every_output_nan(row) -> None:
    got = _out(np.array([row], dtype=np.float32))
    assert np.isnan(got).all(), got


def test_the_denominator_is_not_silently_eps() -> None:
    """The finite entries are the tell: dividing by ``1e-12`` turns a
    ``-1.0`` into ``-1e12`` rather than into NaN."""
    got = _out(np.array([[np.nan, 0.0, -1.0, 2.0]], dtype=np.float32))
    assert not np.isfinite(got).any()
    assert np.abs(np.nan_to_num(got, nan=0.0)).max() == 0.0


def test_an_infinity_without_a_nan() -> None:
    """``‖x‖`` is inf, so the finite entries go to zero and inf/inf is
    NaN — no division by eps anywhere."""
    got = _out(np.array([[np.inf, 1.0, 2.0, 3.0]], dtype=np.float32))
    assert np.isnan(got[0])
    assert np.allclose(got[1:], 0.0)


def test_an_all_zero_row_still_uses_eps() -> None:
    """The clamp exists for this case and must keep working: ``0/eps`` is
    0, not a division by zero."""
    got = _out(np.zeros((1, 4), dtype=np.float32))
    assert np.allclose(got, 0.0)


def test_ordinary_rows_are_untouched() -> None:
    got = _out(np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32))
    assert np.allclose(got, [0.6, 0.8, 0.0, 0.0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_both_widths(dtype) -> None:
    got = _out(np.array([[np.nan, 1.0, 2.0, 3.0]], dtype=dtype))
    assert np.isnan(got).all()


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize(
    "row",
    [
        [np.nan, np.inf, -np.inf, -1.0],
        [np.inf, 1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0, 0.0],
    ],
)
def test_the_two_devices_agree(row) -> None:
    arr = np.array([row], dtype=np.float32)
    cpu, metal = _out(arr, "cpu"), _out(arr, "metal")
    assert np.array_equal(np.isnan(cpu), np.isnan(metal)), (cpu, metal)
    assert np.allclose(cpu, metal, equal_nan=True), (cpu, metal)
