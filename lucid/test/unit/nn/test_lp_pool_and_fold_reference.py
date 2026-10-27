"""``lp_pool`` on signed input, and ``unfold`` / ``fold`` unbatched (LCD-415).

``lp_pool{1,2,3}d`` pooled ``|x| ** p``; the reference pools ``x ** p``, so
an odd or fractional ``p`` on a signed input gave a different, finite answer
(``lp_pool1d([-1, 2, -3, 4], 1, 2)`` was ``[3, 7]`` against ``[1, 1]``).
``p = inf`` was refused.  ``unfold`` and ``fold`` refused an input without
its batch axis.  The cases hold every rank, exponent and device to the
reference, values and gradients, ``nan`` included.
"""

from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

pytestmark = pytest.mark.parity

_EXPONENTS = [1.0, 2.0, 3.0, 0.5, float("inf")]


def _signed(*shape: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _close(got: lucid.Tensor, want: object) -> None:
    want_np = want.detach().numpy()
    assert tuple(got.shape) == tuple(want_np.shape)
    np.testing.assert_allclose(got.numpy(), want_np, rtol=1e-4, atol=1e-5, equal_nan=True)


def _run_both(
    ref: ModuleType, device: str, x: np.ndarray, lucid_fn: object, ref_fn: object
) -> None:
    """Values and the gradient of a weighted sum, Lucid against the reference."""
    xs = lucid.tensor(x, device=device, requires_grad=True)
    xr = ref.tensor(x, requires_grad=True)
    got = lucid_fn(xs)
    want = ref_fn(xr)
    _close(got, want)
    w = _signed(*tuple(got.shape), seed=1)
    (got * lucid.tensor(w, device=device)).sum().backward()
    (want * ref.tensor(w)).sum().backward()
    _close(xs.grad, xr.grad)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("p", _EXPONENTS, ids=lambda p: f"p{p}")
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("ceil_mode", [False, True], ids=["floor", "ceil"])
def test_lp_pool_signed_input_matches_the_reference(
    ref: ModuleType, device: str, rank: int, p: float, batched: bool, ceil_mode: bool
) -> None:
    x = _signed(*((2, 3) if batched else (3,)), *(5,) * rank)
    name = f"lp_pool{rank}d"
    _run_both(
        ref,
        device,
        x,
        lambda t: getattr(F, name)(t, p, 2, ceil_mode=ceil_mode),
        lambda t: getattr(ref.nn.functional, name)(t, p, 2, ceil_mode=ceil_mode),
    )
    module = getattr(nn, f"LPPool{rank}d")(p, 2, ceil_mode=ceil_mode)
    want = getattr(ref.nn.functional, name)(ref.tensor(x), p, 2, ceil_mode=ceil_mode)
    _close(module(lucid.tensor(x, device=device)), want)


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0], ids=lambda p: f"p{p}")
def test_lp_pool_zero_sum_window_has_zero_gradient(ref: ModuleType, device: str, p: float) -> None:
    # Windows ``(1, -1)`` (odd p sums to zero) and ``(0, 0)`` (every p does):
    # the root's derivative is infinite there, the reference's gradient 0.
    x = np.array([[[1.0, -1.0, 0.0, 0.0, 2.0, -3.0]]], dtype=np.float32)
    _run_both(
        ref,
        device,
        x,
        lambda t: F.lp_pool1d(t, p, 2),
        lambda t: ref.nn.functional.lp_pool1d(t, p, 2),
    )


@pytest.mark.parametrize("p", [0.0, -1.0, float("-inf"), float("nan")])
def test_lp_pool_refuses_a_non_positive_exponent(p: float) -> None:
    with pytest.raises(ValueError, match="norm_type must be > 0"):
        F.lp_pool1d(lucid.ones(1, 1, 4), p, 2)


# (kernel, dilation, padding, stride)
_WINDOWS = [(2, 1, 0, 1), (3, 1, 1, 2), ((2, 3), (2, 1), (1, 0), (1, 2))]


@pytest.mark.parametrize("window", _WINDOWS, ids=["k2", "k3p1s2", "asym"])
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
def test_unfold_and_fold_match_the_reference(
    ref: ModuleType, device: str, window: tuple[object, ...], batched: bool
) -> None:
    k, d, pad, s = window
    x = _signed(*((2,) if batched else ()), 3, 8, 7)
    _run_both(
        ref,
        device,
        x,
        lambda t: F.unfold(t, k, d, pad, s),
        lambda t: ref.nn.functional.unfold(t, k, d, pad, s),
    )
    blocks = ref.nn.functional.unfold(ref.tensor(x), k, d, pad, s).numpy()
    _run_both(
        ref,
        device,
        blocks,
        lambda t: F.fold(t, (8, 7), k, d, pad, s),
        lambda t: ref.nn.functional.fold(t, (8, 7), k, d, pad, s),
    )
    module = nn.Fold((8, 7), k, d, pad, s)
    want = ref.nn.functional.fold(ref.tensor(blocks), (8, 7), k, d, pad, s)
    _close(module(lucid.tensor(blocks, device=device)), want)
    _close(
        nn.Unfold(k, d, pad, s)(lucid.tensor(x, device=device)),
        ref.nn.functional.unfold(ref.tensor(x), k, d, pad, s),
    )
