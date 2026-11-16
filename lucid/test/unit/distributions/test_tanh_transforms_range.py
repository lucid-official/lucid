"""LCD-1035: the ``tanh`` transforms keep their inverse and log-determinant across the range.

``TanhTransform`` and ``CorrCholeskyTransform`` spelled the inverse by hand as
``0.5 * log((1 + y) / (1 - y))``: ``1 + y`` rounds to 1 below ``|y|`` ~ 1.8e-5 in float32, so the
inverse of ``tanh(1e-8)`` was 0, and next to ``+-1`` the quotient lost its digits.  ``lucid.atanh``
(range-safe since LCD-1028) is the one owner of that function.

The log-determinant ``log(1 - tanh(x)^2)`` is ``2 (log 2 - |x| - log1p(exp(-2|x|)))``:
``TanhTransform`` wrote ``log1p(exp(-2x))`` out, which overflows float32 below ``x = -44`` (a ``-inf``
for an answer of about ``-2|x|``), and ``CorrCholeskyTransform`` took ``log(1 - tanh(x)^2)`` itself,
clipped at 1e-12, so a large ``|x|`` read as a constant ``-27.6``.

The oracles are plain float64 closed forms (``math``) at the values read back from the tensors.
"""

import math

import pytest

import lucid
import lucid.distributions.transforms as T
from lucid.test._fixtures.devices import device_dtype_params

_FLOATS = device_dtype_params([lucid.float32, lucid.float64])
_EPS = {lucid.float32: 1.2e-7, lucid.float64: 2.3e-16}


def _log_one_minus_tanh_sq(x: float) -> float:
    """``log(1 - tanh(x)^2)`` without the cancellation: even in ``x``, linear for a large ``|x|``."""
    a = abs(x)
    return 2.0 * (math.log(2.0) - a - math.log1p(math.exp(-2.0 * a)))


def _logspace(lo: float, hi: float, n: int) -> list[float]:
    step = (math.log(hi) - math.log(lo)) / (n - 1)
    return [math.exp(math.log(lo) + step * i) for i in range(n)]


# ── TanhTransform ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_tanh_inverse_is_atanh_of_the_value_it_is_given(device: str, dtype: lucid.dtype) -> None:
    """``|y|`` from 1e-8 to just below 1: the closed form of the float64 value of ``y``."""
    mags = _logspace(1e-8, 0.5, 40) + [0.9, 0.99, 0.999, 0.9999] + [1.0 - 2.0**-20]
    ys = [s * m for m in mags for s in (1.0, -1.0)]
    y = lucid.tensor(ys, dtype=dtype, device=device)
    got = T.TanhTransform().inv(y).tolist()
    for yv, g in zip(y.tolist(), got, strict=True):
        assert math.isclose(g, math.atanh(yv), rel_tol=8 * _EPS[dtype]), (yv, g, math.atanh(yv))


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_tanh_round_trips_from_1e_minus_8_to_5(device: str, dtype: lucid.dtype) -> None:
    """``inv(fwd(x)) == x``, to the conditioning of ``tanh`` at ``x`` (what ``y`` rounds to)."""
    xs = _logspace(1e-8, 5.0, 60)
    x = lucid.tensor(xs, dtype=dtype, device=device)
    transform = T.TanhTransform()
    y = transform(x)
    back = transform.inv(y).tolist()
    for xv, yv, b in zip(x.tolist(), y.tolist(), back, strict=True):
        # ``y`` carries a relative error of eps, which ``arctanh`` scales by ``y / ((1 - y^2) x)``
        cond = abs(yv) / ((1.0 - yv * yv) * xv)
        assert abs(b - xv) <= 4 * _EPS[dtype] * (cond + 1.0) * xv, (xv, b)
    # the point of the card: a small ``x`` comes back as itself, not as 0
    assert back[0] > 0.0 and math.isclose(back[0], xs[0], rel_tol=1e-6)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_tanh_log_det_is_finite_and_exact_over_the_range(device: str, dtype: lucid.dtype) -> None:
    """Float32 from -100 to 100 (``exp(-2x)`` overflowed below -44), float64 to 400 (below -355)."""
    top = 100.0 if dtype == lucid.float32 else 400.0
    mags = [0.0, 1e-8, 1e-3, 0.5, 2.0, 9.0, 20.0, 44.0, 45.0, 60.0, top]
    xs = [s * m for m in mags for s in (1.0, -1.0)]
    x = lucid.tensor(xs, dtype=dtype, device=device, requires_grad=True)
    transform = T.TanhTransform()
    # ``y`` is ignored by the log-determinant; ``tanh`` saturates at 1 from |x| ~ 9 here
    ld = transform.log_abs_det_jacobian(x, transform(x))
    atol = 3e-6 if dtype == lucid.float32 else 1e-15
    for xv, g in zip(x.tolist(), ld.tolist(), strict=True):
        want = _log_one_minus_tanh_sq(xv)
        assert math.isfinite(g), (xv, g)
        assert abs(g - want) <= 4 * _EPS[dtype] * abs(want) + atol, (xv, g, want)
    # d/dx log(1 - tanh^2 x) = -2 tanh x: -2 sign(x) once saturated, finite all through
    (grad,) = lucid.autograd.grad(ld.sum(), x)
    for xv, gv in zip(x.tolist(), grad.tolist(), strict=True):
        assert math.isclose(
            gv, -2.0 * math.tanh(xv), abs_tol=2e-6 if dtype == lucid.float32 else 1e-12
        ), (xv, gv)


# ── CorrCholeskyTransform ───────────────────────────────────────────────────

_TINY_FREE = [1e-8, -1e-7, 3e-6, 1e-5, 0.5, -1.5]  # dim 4: 6 free entries, tiny to moderate


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_corr_cholesky_inverse_keeps_tiny_off_diagonals(device: str, dtype: lucid.dtype) -> None:
    """The factor's off-diagonal ``1e-8`` is a free entry of ``1e-8``, not 0 (float32) or ``1 +- 1e-8``."""
    transform = T.CorrCholeskyTransform(4)
    free = lucid.tensor(_TINY_FREE, dtype=dtype, device=device)
    back = transform.inv(transform(free)).tolist()
    # the stick-breaking scale ``sqrt(1 - cum_sq)`` and the ``tanh`` it came through cost a few eps
    rel = 5e-6 if dtype == lucid.float32 else 1e-12
    for want, got in zip(_TINY_FREE, back, strict=True):
        assert math.isclose(got, want, rel_tol=rel), (want, got)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_corr_cholesky_inverse_of_a_tiny_two_by_two(device: str, dtype: lucid.dtype) -> None:
    """The smallest case, one free entry: a factor with ``L[1, 0] = 1e-8``."""
    transform = T.CorrCholeskyTransform(2)
    factor = lucid.tensor([[1.0, 0.0], [1e-8, 1.0]], dtype=dtype, device=device)
    (got,) = transform.inv(factor).tolist()
    assert math.isclose(got, 1e-8, rel_tol=1e-6), got


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_corr_cholesky_log_det_reads_a_large_x_off_the_formula(
    device: str, dtype: lucid.dtype
) -> None:
    """``log(1 - tanh(x)^2)`` for ``x = 20`` is ``-38.6``; the clipped textbook form said ``-27.6``.

    Dim 3, free entries ``[x0, x1, x2]`` fill ``L[1, 0]``, ``L[2, 0]`` and ``L[2, 1]``: the stick-breaking
    term is half the log-determinant of the ``tanh`` at ``x1`` (the only entry two columns below
    the diagonal), so the whole is ``ld(x0) + 1.5 ld(x1) + ld(x2)``.
    """
    free = [20.0, 0.5, -0.3] if dtype == lucid.float64 else [9.0, 0.5, -0.3]
    transform = T.CorrCholeskyTransform(3)
    x = lucid.tensor(free, dtype=dtype, device=device)
    got = transform.log_abs_det_jacobian(x, transform(x)).item()
    want = (
        _log_one_minus_tanh_sq(free[0])
        + 1.5 * _log_one_minus_tanh_sq(free[1])
        + _log_one_minus_tanh_sq(free[2])
    )
    assert math.isclose(got, want, rel_tol=2e-6 if dtype == lucid.float32 else 1e-12), (got, want)
