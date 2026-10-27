"""Deterministic tensor factories.

Every numerical test that needs random data should pull through one of
these helpers — they give a reproducible NumPy seed (so parity tests
get bit-equal lucid + reference-framework inputs) and standardize how
shapes/dtypes/devices flow through the suite.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import lucid


def make_array(
    shape: Sequence[int],
    *,
    dtype: lucid.dtype | None = None,
    seed: int = 0,
    low: float = -1.0,
    high: float = 1.0,
) -> np.ndarray:
    """Build a deterministic NumPy array of the requested shape.

    For float dtypes the values are drawn from a uniform ``[low, high]``;
    for int dtypes they're drawn from ``randint(low, high)`` clipped to
    the dtype range.  ``seed`` is hashed with the shape so different
    callers get independent streams without colliding.
    """
    rng = np.random.default_rng(seed=hash((seed, tuple(shape))) & 0xFFFF_FFFF)
    if dtype is None or dtype in (lucid.float32, lucid.float64, lucid.float16, lucid.bfloat16):
        arr = rng.uniform(low=low, high=high, size=shape)
        # bfloat16 / float16 not directly representable in numpy across
        # versions — keep f32 here and let the lucid factory cast.
        return arr.astype(np.float32)
    if dtype in (lucid.int8, lucid.int16, lucid.int32, lucid.int64):
        # Clamp the requested range to a safe span; the caller can
        # always pass a tighter ``low``/``high`` when needed.
        i_lo = int(max(low, -128))
        i_hi = int(min(high, 127))
        if i_hi <= i_lo:
            i_hi = i_lo + 1
        arr = rng.integers(low=i_lo, high=i_hi + 1, size=shape)
        return arr.astype(np.int64)
    if dtype == lucid.bool_:
        return rng.integers(low=0, high=2, size=shape).astype(bool)
    return rng.standard_normal(size=shape).astype(np.float32)


def make_tensor(
    shape: Sequence[int],
    *,
    dtype: lucid.dtype | None = None,
    device: str = "cpu",
    seed: int = 0,
    low: float = -1.0,
    high: float = 1.0,
    requires_grad: bool = False,
) -> lucid.Tensor:
    """Build a deterministic Lucid tensor on the requested device."""
    arr = make_array(shape, dtype=dtype, seed=seed, low=low, high=high)
    return lucid.tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)


def make_pair(
    shape: Sequence[int],
    *,
    dtype: lucid.dtype | None = None,
    seed: int = 0,
    low: float = -1.0,
    high: float = 1.0,
) -> tuple[lucid.Tensor, Any]:
    """Build a matched ``(lucid_tensor, ref_tensor)`` pair from the
    same NumPy seed.  Calling this without the reference framework
    installed raises — parity tests that need it should already be
    behind the ``parity/`` collect-time skip."""
    from lucid.test._fixtures.ref_framework import require_ref

    arr = make_array(shape, dtype=dtype, seed=seed, low=low, high=high)
    ref = require_ref()
    l = lucid.tensor(arr.copy(), dtype=dtype)
    r = ref.tensor(arr.copy())
    return l, r


@pytest.fixture
def tensor_factory(device: str) -> Any:
    """Convenience: return a ``make_tensor``-like callable bound to the
    current ``device`` parametrize value.  Saves passing ``device=``
    on every call inside a test body."""

    def _make(shape: Sequence[int], **kwargs: Any) -> lucid.Tensor:
        kwargs.setdefault("device", device)
        return make_tensor(shape, **kwargs)

    return _make
