"""Local response norm divides alpha by the window size (LCD-440).

The scale is ``(k + alpha * mean(x**2 over the window)) ** beta``: the window
statistic is a mean, and padded channels stay in its denominator.  The window
reaches ``size // 2`` channels below a channel and ``(size - 1) // 2`` above
it.  Lucid summed the window instead of averaging it, and padded the two
sides the other way round, so every LRN output was wrong — by about 1e-3 in
float64 at the default alpha, and its second derivative by up to 170%.
``nn.LocalResponseNorm`` carried its own copy of the rule, with a third
padding, and now calls the functional.

The definition is checked against a direct numpy transcription (no reference
needed), and forward, gradient and the create_graph second derivative against
the reference, for every window size from 1 to past the channel count.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._fixtures.devices import device_dtype_params

_SHAPES = {
    "3d": (2, 5, 6),
    "4d": (2, 5, 3, 4),
    "5d": (1, 5, 2, 3, 2),
}
# Window sizes 1-5, and 7 — wider than the 5 channels.
_SIZES = (1, 2, 3, 4, 5, 7)
_PARAMS = {
    "default": {},
    "custom": {"alpha": 0.5, "beta": 0.6, "k": 1.5},
}
_DEVICE_DTYPES = device_dtype_params((lucid.float32, lucid.float64))
_TOL = {lucid.float32: 2e-5, lucid.float64: 1e-12}


def _values(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape) * 2.0


def _definition(x: np.ndarray, size: int, alpha: float, beta: float, k: float) -> np.ndarray:
    """The rule written out channel by channel, in float64."""
    channels = x.shape[1]
    out = np.empty_like(x)
    for c in range(channels):
        lo, hi = c - size // 2, c + (size - 1) // 2
        window = x[:, max(lo, 0) : min(hi, channels - 1) + 1] ** 2
        mean = window.sum(axis=1) / size
        out[:, c] = x[:, c] / (k + alpha * mean) ** beta
    return out


def _kwargs(params: str) -> dict[str, float]:
    return {"alpha": 1e-4, "beta": 0.75, "k": 1.0, **_PARAMS[params]}


@pytest.mark.parametrize("params", list(_PARAMS))
@pytest.mark.parametrize("size", _SIZES)
@pytest.mark.parametrize("rank", list(_SHAPES))
def test_matches_the_definition(rank: str, size: int, params: str) -> None:
    kw = _kwargs(params)
    a = _values(_SHAPES[rank], seed=size)
    want = _definition(a, size, **kw)
    got_f = F.local_response_norm(lucid.tensor(a), size, **kw).numpy()
    got_m = nn.LocalResponseNorm(size, **kw)(lucid.tensor(a)).numpy()
    np.testing.assert_allclose(got_f, want, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(got_m, got_f)


def _derivatives(lib: Any, fn: Any, a: np.ndarray, v: np.ndarray, w: np.ndarray, **tkw: Any):
    """Forward, ``d<y, v>/dx`` and ``d<d<y, v>/dx, w>/dx``."""
    x = lib.tensor(a, **tkw).requires_grad_(True)
    y = fn(x)
    (g1,) = lib.autograd.grad((y * lib.tensor(v, **tkw)).sum(), x, create_graph=True)
    (g2,) = lib.autograd.grad((g1 * lib.tensor(w, **tkw)).sum(), x)
    return [t.detach().cpu().numpy() if lib is not lucid else t.numpy() for t in (y, g1, g2)]


@pytest.mark.parity
@pytest.mark.parametrize("params", list(_PARAMS))
@pytest.mark.parametrize("size", _SIZES)
@pytest.mark.parametrize("rank", list(_SHAPES))
@pytest.mark.parametrize(("device", "dtype"), _DEVICE_DTYPES)
def test_forward_and_derivatives_match_reference(
    ref: Any, device: str, dtype: lucid.dtype, rank: str, size: int, params: str
) -> None:
    kw = _kwargs(params)
    shape = _SHAPES[rank]
    np_dtype = np.float32 if dtype == lucid.float32 else np.float64
    a = _values(shape, seed=size).astype(np_dtype)
    v = _values(shape, seed=100 + size).astype(np_dtype)
    w = _values(shape, seed=200 + size).astype(np_dtype)

    got = _derivatives(
        lucid,
        lambda x: F.local_response_norm(x, size, **kw),
        a,
        v,
        w,
        device=device,
        dtype=dtype,
    )
    want = _derivatives(
        ref, lambda x: ref.nn.functional.local_response_norm(x, size, **kw), a, v, w
    )
    tol = _TOL[dtype]
    for label, g, r in zip(("forward", "gradient", "second derivative"), got, want, strict=True):
        np.testing.assert_allclose(g, r, rtol=tol, atol=tol, err_msg=label)
