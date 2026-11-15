"""LCD-1029: the composites ``log1p`` / ``logaddexp`` / ``atan2`` / ``hypot`` across the float range.

Defect class: an intermediate leaves float32 (or cancels) while the result still fits, so
a value or a derivative of some order comes back wrong where the closed form is an
ordinary float.  Each op is fixed in its own formula (``ops/composite/Math.cpp``):

* ``log1p``: the lanes where ``1 + x`` rounds to 1 returned ``x`` itself, whose graph has
  the constant slope 1, so every derivative past the first was 0 (``d2 log1p(0)`` is -1).
  Every lane keeps ``log(1 + x)``'s graph now; ``log1p(-0)`` is still -0.
* ``logaddexp``: ``m + log(exp(a - m) + exp(b - m))`` rounds the partner's weight away under
  6e-8 (``logaddexp(0, -17)`` was 0 against 4.1e-8).  It is ``m + log1p(exp(min - max))``.
* ``atan2``: the slope went through ``1 / (1 + q^2)`` with ``q = y / x``, which is 0 once the
  operands are 1e20 apart.  The arctangent is taken on the ratio that is at most 1.
* ``hypot``: ``maximum`` / ``minimum`` split a tied cotangent, which left the ratio's slope 0
  and the second derivative at ``|a| = |b|`` 0 at every scale.

Closed forms are plain float64 on the float32 values read back from the tensors, written in
ratios (``x / h``) so they stay floats where ``x^2`` would not.
"""

import math
from collections.abc import Callable

import pytest

import lucid
import lucid.special
from lucid.autograd import grad

INF = math.inf
NAN = math.nan

#: Decades the operand grids span (float32 holds 1e-38 .. 3e38).
EXPONENTS = list(range(-30, 31, 3))
#: A relative tolerance float32 meets on a derivative of order 2 (a few rounded divides).
TOL = 3e-5


def _leaf(values: list[float], device: str, dtype: lucid.dtype = lucid.float32) -> lucid.Tensor:
    return lucid.tensor(values, dtype=dtype, device=device, requires_grad=True)


def _finite(x: float) -> bool:
    return x - x == 0.0


def _close(got: list[float], want: list[float], scale: list[float], tol: float) -> None:
    """Every entry finite and within ``tol * scale`` of ``want`` (scale never 0)."""
    for i, (x, y, s) in enumerate(zip(got, want, scale, strict=True)):
        assert _finite(x), f"entry {i} is {x}, want {y}"
        assert abs(x - y) <= tol * s, f"entry {i}: {x} vs {y} (scale {s})"


def _close_each(got: list[float], want: list[float], tol: float, floor: float = 1e-36) -> None:
    """Every entry finite and within ``tol`` of ITS OWN magnitude; below ``floor`` only ~0 is asked.

    Stricter than ``_close``'s shared scale: a component that is 1e-20 of its partner's must
    still be right (``atan2``'s slope at ``y = 1e-10, x = 1e-30`` was 0 where it is 1e-10).
    """
    for i, (x, y) in enumerate(zip(got, want, strict=True)):
        assert _finite(x), f"entry {i} is {x}, want {y}"
        if abs(y) < floor:
            assert abs(x) <= floor, f"entry {i}: {x} vs {y}"
        else:
            assert abs(x - y) <= tol * abs(y), f"entry {i}: {x} vs {y}"


def _same_value(got: float, want: float, tol: float = 2e-6) -> bool:
    """NaN matches NaN, a zero matches in sign, anything else to a relative ``tol``."""
    if math.isnan(want):
        return math.isnan(got)
    if want == 0.0 or math.isinf(want):
        return got == want and math.copysign(1.0, got) == math.copysign(1.0, want)
    return abs(got - want) <= tol * abs(want)


def _pairs(exps_a: list[int], exps_b: list[int]) -> tuple[list[float], list[float]]:
    """Every (a, b) of the two decade lists, with signs and a 1.7 mantissa on one side."""
    rows = [
        (sa * 1.7 * 10.0**ea, sb * 10.0**eb)
        for ea in exps_a
        for eb in exps_b
        for sa, sb in ((1, 1), (-1, 1), (1, -1), (-1, -1))
    ]
    return [r[0] for r in rows], [r[1] for r in rows]


# ── log1p ────────────────────────────────────────────────────────────────────────────


def _log1p_orders(x: list[float]) -> tuple[list[float], list[float], list[float]]:
    return (
        [1 / (1 + v) for v in x],
        [-1 / (1 + v) ** 2 for v in x],
        [2 / (1 + v) ** 3 for v in x],
    )


def _three_derivatives(
    fn: Callable[[lucid.Tensor], lucid.Tensor], x: lucid.Tensor
) -> tuple[list[float], list[float], list[float]]:
    y = fn(x)
    (d1,) = grad(y.sum(), [x], create_graph=True)
    (d2,) = grad(d1.sum(), [x], create_graph=True)
    (d3,) = grad(d2.sum(), [x])
    return d1.tolist(), d2.tolist(), d3.tolist()


@pytest.mark.parametrize(
    "xs",
    [
        [0.0, -0.0, 1e-8, -1e-8, 1e-12, 1e-30, -1e-30, 1e-37],
        [3e-8, 5.9e-8, 6.1e-8, 1e-7, 1e-5, 0.01, 0.5, 1.0, 100.0, 1e10, 1e30],
        [-0.5, -0.9, -0.99, -1e-3, -1e-7],
    ],
    ids=["vanished", "rounding", "negative"],
)
def test_log1p_every_derivative_order_is_exact(device: str, xs: list[float]) -> None:
    """The lanes where ``1 + x == 1`` used to hand back ``x``: slope 1, then 0, 0 ..."""
    x = _leaf(xs, device)
    got = _three_derivatives(lucid.log1p, x)
    want = _log1p_orders(x.tolist())
    for order in range(3):
        _close(got[order], want[order], [max(abs(w), 1e-30) for w in want[order]], TOL)


def test_log1p_second_derivative_at_zero_is_minus_one_not_minus_zero(device: str) -> None:
    x = _leaf([0.0, 1e-8], device)
    _, d2, d3 = _three_derivatives(lucid.log1p, x)
    assert d2 == [-1.0, -1.0]
    assert d3 == [2.0, 2.0]


def test_xlog1py_second_derivative_through_a_vanishing_argument(device: str) -> None:
    y = _leaf([1e-9, 0.0, 1e-30], device)
    out = lucid.special.xlog1py(lucid.tensor([2.0, 2.0, 2.0], device=device), y)
    (d1,) = grad(out.sum(), [y], create_graph=True)
    (d2,) = grad(d1.sum(), [y])
    assert d1.tolist() == [2.0, 2.0, 2.0]
    _close(d2.tolist(), [-2.0] * 3, [2.0] * 3, TOL)


def test_log1p_values_are_unchanged(device: str) -> None:
    """Every value is ``log1p`` of the float32 operand, and the zeros keep their sign."""
    xs = [1e-30, -1e-30, 1e-12, 3e-8, 6e-8, 1e-5, 0.5, -0.5, 10.0, 1e30, 3e38]
    got = lucid.log1p(lucid.tensor(xs, device=device)).tolist()
    for g, v in zip(got, xs, strict=True):
        assert _same_value(g, math.log1p(v), 1e-6), (v, g)
    for zero in (0.0, -0.0):
        (out,) = lucid.log1p(lucid.tensor([zero], device=device)).tolist()
        assert out == 0.0 and math.copysign(1.0, out) == math.copysign(1.0, zero)
        # the same through a recorded graph
        z = _leaf([zero], device)
        (out,) = lucid.log1p(z).tolist()
        assert math.copysign(1.0, out) == math.copysign(1.0, zero)


@pytest.mark.parametrize("tracked", [False, True])
@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16, lucid.float32])
def test_log1p_keeps_the_sign_of_zero_in_every_dtype(
    device: str, dtype: lucid.dtype, tracked: bool
) -> None:
    """``log(1 + x)`` is +0 where ``x`` vanishes, so the sum has to be written to keep -0."""
    x = lucid.tensor([-0.0, 0.0], dtype=dtype, device=device, requires_grad=tracked)
    signs = [math.copysign(1.0, v) for v in lucid.log1p(x).tolist()]
    assert signs == [-1.0, 1.0]


def test_log1p_special_values(device: str) -> None:
    xs = [-INF, -2.0, -1.0, INF, NAN]
    got = lucid.log1p(lucid.tensor(xs, device=device)).tolist()
    want = [NAN, NAN, -INF, INF, NAN]
    assert all(_same_value(g, w) for g, w in zip(got, want, strict=True)), got
    x = _leaf([-1.0, -2.0, 0.0], device)
    (d1,) = grad(lucid.log1p(x).sum(), [x])
    assert d1.tolist()[0] == INF and d1.tolist()[1:] == [-1.0, 1.0]


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
def test_log1p_half_precision(device: str, dtype: lucid.dtype) -> None:
    xs = [1e-4, 1e-3, 0.25]
    x = lucid.tensor(xs, dtype=dtype, device=device, requires_grad=True)
    out = lucid.log1p(x)
    assert out.dtype == dtype
    for g, v in zip(out.tolist(), xs, strict=True):
        assert abs(g - math.log1p(v)) <= 1e-2 * math.log1p(v), (v, g)
    (d1,) = grad(out.sum(), [x], create_graph=True)
    (d2,) = grad(d1.sum(), [x])
    assert all(abs(g + 1.0) < 0.05 for g in d2.tolist()[:2])


def test_log1p_float64_on_the_cpu() -> None:
    xs = [0.0, 1e-300, 1e-17, 1e-9, 0.3, 1e100]
    x = _leaf(xs, "cpu", lucid.float64)
    out = lucid.log1p(x).tolist()
    for g, v in zip(out, xs, strict=True):
        assert _same_value(g, math.log1p(v), 1e-14), (v, g)
    d1, d2, d3 = _three_derivatives(lucid.log1p, x)
    want = _log1p_orders(xs)
    for got, w in zip((d1, d2, d3), want, strict=True):
        _close(got, w, [abs(v) for v in w], 1e-12)


# ── logaddexp ────────────────────────────────────────────────────────────────────────


def _logaddexp(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return NAN
    if a == b and math.isinf(a):
        return a
    hi, lo = max(a, b), min(a, b)
    if math.isinf(hi):
        return hi
    return hi + math.log1p(math.exp(lo - hi))


LOGADDEXP_BASES = [-40.0, -17.5, -3.0, -1e-3, 0.0, 1e-3, 2.0, 25.0]
LOGADDEXP_GAPS = [0.0, 1e-6, 1e-3, 0.5, 5.0, 10.0, 17.0, 20.0, 30.0, 60.0]


def _logaddexp_operands() -> tuple[list[float], list[float]]:
    a, b = [], []
    for base in LOGADDEXP_BASES:
        for gap in LOGADDEXP_GAPS:
            a += [base, base - gap]
            b += [base - gap, base]
    return a, b


def test_logaddexp_the_partner_below_six_e_minus_eight_is_kept(device: str) -> None:
    got = lucid.logaddexp(
        lucid.tensor([0.0, 0.0, 0.0], device=device),
        lucid.tensor([-17.0, -20.0, -30.0], device=device),
    ).tolist()
    want = [math.log1p(math.exp(-t)) for t in (17.0, 20.0, 30.0)]
    assert all(_same_value(g, w, 1e-5) for g, w in zip(got, want, strict=True)), got


def test_logaddexp_values_over_the_grid(device: str) -> None:
    a, b = _logaddexp_operands()
    ta, tb = lucid.tensor(a, device=device), lucid.tensor(b, device=device)
    got = lucid.logaddexp(ta, tb).tolist()
    for g, x, y in zip(got, ta.tolist(), tb.tolist(), strict=True):
        want = _logaddexp(x, y)
        assert _same_value(g, want, 1e-5) or abs(g - want) < 1e-6, (x, y, g, want)


def test_logaddexp_first_and_second_derivatives_are_the_softmax_weights(device: str) -> None:
    a0, b0 = _logaddexp_operands()
    a, b = _leaf(a0, device), _leaf(b0, device)
    dy_a, dy_b = grad(lucid.logaddexp(a, b).sum(), [a, b], create_graph=True)
    weights_a, weights_b, curvature = [], [], []
    for x, y in zip(a.tolist(), b.tolist(), strict=True):
        hi, lo = max(x, y), min(x, y)
        tail = math.exp(lo - hi)
        w_hi, w_lo = 1 / (1 + tail), tail / (1 + tail)
        weights_a.append(w_hi if x >= y else w_lo)
        weights_b.append(w_lo if x >= y else w_hi)
        curvature.append(tail / (1 + tail) ** 2)
    _close(dy_a.tolist(), weights_a, [max(w, 1e-30) for w in weights_a], TOL)
    _close(dy_b.tolist(), weights_b, [max(w, 1e-30) for w in weights_b], TOL)
    # d(dy_a - dy_b)/d(a, b): the Hessian is c * [[1, -1], [-1, 1]]
    h_a, h_b = grad((dy_a - dy_b).sum(), [a, b])
    _close(h_a.tolist(), [2 * c for c in curvature], [max(2 * c, 1e-30) for c in curvature], 1e-4)
    _close(h_b.tolist(), [-2 * c for c in curvature], [max(2 * c, 1e-30) for c in curvature], 1e-4)


def test_logaddexp_a_tie_keeps_a_half_and_a_quarter(device: str) -> None:
    a, b = _leaf([1.5, -30.0], device), _leaf([1.5, -30.0], device)
    dy_a, dy_b = grad(lucid.logaddexp(a, b).sum(), [a, b], create_graph=True)
    assert dy_a.tolist() == [0.5, 0.5] and dy_b.tolist() == [0.5, 0.5]
    h_a, h_b = grad((dy_a - dy_b).sum(), [a, b])
    assert h_a.tolist() == [0.5, 0.5] and h_b.tolist() == [-0.5, -0.5]


def test_logaddexp_slopes_of_the_infinite_lanes_are_the_weights(device: str) -> None:
    """``(x, +inf)``: the infinity has weight 1, the finite partner 0 (it was NaN / 0.047)."""
    a = _leaf([-3.0, INF, -INF, 0.0, INF], device)
    b = _leaf([INF, -3.0, 2.0, -INF, -INF], device)
    da, db = grad(lucid.logaddexp(a, b).sum(), [a, b])
    assert da.tolist() == [0.0, 1.0, 0.0, 1.0, 1.0]
    assert db.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    nan = _leaf([NAN, 1.0], device)
    other = _leaf([1.0, NAN], device)
    for g in grad(lucid.logaddexp(nan, other).sum(), [nan, other]):
        assert all(math.isnan(v) for v in g.tolist())


def test_logaddexp_special_values(device: str) -> None:
    pool = [-INF, -3.0, 0.0, 2.0, INF, NAN]
    a = [x for x in pool for _ in pool]
    b = [y for _ in pool for y in pool]
    got = lucid.logaddexp(lucid.tensor(a, device=device), lucid.tensor(b, device=device)).tolist()
    for g, x, y in zip(got, a, b, strict=True):
        assert _same_value(g, _logaddexp(x, y), 1e-6), (x, y, g)


def test_logaddexp2_inherits_the_partner(device: str) -> None:
    a = lucid.tensor([0.0, 0.0, 3.0], device=device)
    b = lucid.tensor([-25.0, -40.0, 3.0 - 28.0], device=device)
    got = lucid.logaddexp2(a, b).tolist()
    for g, x, y in zip(got, a.tolist(), b.tolist(), strict=True):
        hi, lo = max(x, y), min(x, y)
        want = hi + math.log1p(2.0 ** (lo - hi)) / math.log(2.0)
        assert _same_value(g, want, 1e-5), (x, y, g, want)


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
def test_logaddexp_half_precision(device: str, dtype: lucid.dtype) -> None:
    a = lucid.tensor([0.0, 0.0], dtype=dtype, device=device)
    b = lucid.tensor([-12.0, -8.0], dtype=dtype, device=device)
    out = lucid.logaddexp(a, b)
    assert out.dtype == dtype
    for g, t in zip(out.tolist(), (12.0, 8.0), strict=True):
        assert abs(g - math.log1p(math.exp(-t))) <= 2e-2 * math.exp(-t), (t, g)


def test_logaddexp_float64_on_the_cpu() -> None:
    a = lucid.tensor([0.0, 0.0, -500.0], dtype=lucid.float64)
    b = lucid.tensor([-40.0, -700.0, -540.0], dtype=lucid.float64)
    got = lucid.logaddexp(a, b).tolist()
    for g, x, y in zip(got, a.tolist(), b.tolist(), strict=True):
        assert _same_value(g, _logaddexp(x, y), 1e-13), (x, y, g)
    p, q = _leaf([0.0], "cpu", lucid.float64), _leaf([-40.0], "cpu", lucid.float64)
    dp, dq = grad(lucid.logaddexp(p, q).sum(), [p, q], create_graph=True)
    h_p, _ = grad((dp - dq).sum(), [p, q])
    tail = math.exp(-40.0)
    assert abs(dq.tolist()[0] - tail / (1 + tail)) <= 1e-12 * tail
    assert abs(h_p.tolist()[0] - 2 * tail / (1 + tail) ** 2) <= 1e-12 * tail


# ── atan2 ────────────────────────────────────────────────────────────────────────────


def _atan2_first(y: float, x: float) -> tuple[float, float]:
    h = math.hypot(x, y)
    return (x / h) / h, -(y / h) / h  # d/dy, d/dx


def _atan2_second(y: float, x: float) -> tuple[float, float, float]:
    h = math.hypot(x, y)
    xr, yr = x / h, y / h
    return (-2 * xr * yr / h / h, (yr * yr - xr * xr) / h / h, 2 * xr * yr / h / h)  # yy, xy, xx


def test_atan2_the_slope_at_extreme_ratios(device: str) -> None:
    y, x = _leaf([1e-10, 1e30, 1e-30], device), _leaf([1e-30, 1e-10, 1e30], device)
    dy, dx = grad(lucid.atan2(y, x).sum(), [y, x])
    for got_y, got_x, yy, xx in zip(dy.tolist(), dx.tolist(), y.tolist(), x.tolist(), strict=True):
        want_y, want_x = _atan2_first(yy, xx)
        scale = 1 / math.hypot(xx, yy)
        assert abs(got_y - want_y) <= TOL * scale and abs(got_x - want_x) <= TOL * scale
    assert dy.tolist()[0] == pytest.approx(1e-10, rel=1e-5, abs=0.0)
    assert dx.tolist()[0] == pytest.approx(-1e10, rel=1e-5, abs=0.0)


def test_atan2_first_derivative_over_the_grid(device: str) -> None:
    ys, xs = _pairs(EXPONENTS, EXPONENTS)
    y, x = _leaf(ys, device), _leaf(xs, device)
    dy, dx = grad(lucid.atan2(y, x).sum(), [y, x])
    wants = [_atan2_first(a, b) for a, b in zip(y.tolist(), x.tolist(), strict=True)]
    _close_each(dy.tolist(), [w[0] for w in wants], TOL)
    _close_each(dx.tolist(), [w[1] for w in wants], TOL)


def test_atan2_second_derivative_over_the_grid(device: str) -> None:
    ys, xs = _pairs(EXPONENTS, EXPONENTS)
    rows = [(a, b) for a, b in zip(ys, xs, strict=True) if 1e-14 < max(abs(a), abs(b)) < 1e14]
    y, x = _leaf([r[0] for r in rows], device), _leaf([r[1] for r in rows], device)
    n = len(rows)
    uw = [[0.7, -1.1, 0.9, 1.3, -0.5][i % 5] for i in range(n)]
    vw = [[1.3, 0.6, -0.8, 1.1, 0.9][i % 5] for i in range(n)]
    dy, dx = grad(lucid.atan2(y, x).sum(), [y, x], create_graph=True)
    h_y, h_x = grad(
        dy * lucid.tensor(uw, device=device) + dx * lucid.tensor(vw, device=device),
        [y, x],
        grad_outputs=lucid.ones(n, device=device),
    )
    seconds = [_atan2_second(a, b) for a, b in zip(y.tolist(), x.tolist(), strict=True)]
    scale = [1 / math.hypot(a, b) ** 2 for a, b in zip(y.tolist(), x.tolist(), strict=True)]
    want_y = [u_ * s[0] + v_ * s[1] for u_, v_, s in zip(uw, vw, seconds, strict=True)]
    want_x = [u_ * s[1] + v_ * s[2] for u_, v_, s in zip(uw, vw, seconds, strict=True)]
    _close(h_y.tolist(), want_y, scale, 1e-4)
    _close(h_x.tolist(), want_x, scale, 1e-4)


@pytest.mark.parametrize("sy", [1.0, -1.0])
@pytest.mark.parametrize("sx", [1.0, -1.0])
def test_atan2_a_tie_of_the_operands_keeps_both_derivatives(
    device: str, sx: float, sy: float
) -> None:
    """``|y| = |x|`` is where the swap switches: both forms differentiate the same."""
    y, x = _leaf([sy * 2.0, sy * 3e-10], device), _leaf([sx * 2.0, sx * 3e-10], device)
    dy, dx = grad(lucid.atan2(y, x).sum(), [y, x], create_graph=True)
    want = [_atan2_first(a, b) for a, b in zip(y.tolist(), x.tolist(), strict=True)]
    scale = [1 / math.hypot(a, b) for a, b in zip(y.tolist(), x.tolist(), strict=True)]
    _close(dy.tolist(), [w[0] for w in want], scale, TOL)
    _close(dx.tolist(), [w[1] for w in want], scale, TOL)
    h_y, h_x = grad((dy + dx).sum(), [y, x])
    second = [_atan2_second(a, b) for a, b in zip(y.tolist(), x.tolist(), strict=True)]
    sc2 = [s * s for s in scale]
    _close(h_y.tolist(), [s[0] + s[1] for s in second], sc2, 1e-4)
    _close(h_x.tolist(), [s[1] + s[2] for s in second], sc2, 1e-4)


ATAN2_POOL = [-INF, -3.0, -0.0, 0.0, 3.0, INF, NAN]


def test_atan2_special_value_table_is_unchanged(device: str) -> None:
    ys = [v for v in ATAN2_POOL for _ in ATAN2_POOL]
    xs = [v for _ in ATAN2_POOL for v in ATAN2_POOL]
    got = lucid.atan2(lucid.tensor(ys, device=device), lucid.tensor(xs, device=device)).tolist()
    for g, y, x in zip(got, ys, xs, strict=True):
        assert _same_value(g, math.atan2(y, x), 1e-6), (y, x, g, math.atan2(y, x))


def test_atan2_values_over_the_grid(device: str) -> None:
    ys, xs = _pairs(EXPONENTS, EXPONENTS)
    got = lucid.atan2(lucid.tensor(ys, device=device), lucid.tensor(xs, device=device)).tolist()
    for g, y, x in zip(got, ys, xs, strict=True):
        # float32 holds the operands exactly as read back; Metal flushes a result below 1e-37
        a, b = lucid.tensor([y]).tolist()[0], lucid.tensor([x]).tolist()[0]
        want = math.atan2(a, b)
        assert abs(g - want) <= 2e-6 * abs(want) + 2e-38, (y, x, g, want)


def test_atan2_float64_on_the_cpu_keeps_its_range() -> None:
    ys = [1e-100, 1e200, 1e-200, 3.0, -1e-300]
    xs = [1e-200, 1e-100, 1e100, 3.0, -1e300]
    y, x = _leaf(ys, "cpu", lucid.float64), _leaf(xs, "cpu", lucid.float64)
    out = lucid.atan2(y, x)
    for g, a, b in zip(out.tolist(), ys, xs, strict=True):
        assert _same_value(g, math.atan2(a, b), 1e-14), (a, b, g)
    dy, dx = grad(out.sum(), [y, x])
    for gy, gx, a, b in zip(dy.tolist(), dx.tolist(), ys, xs, strict=True):
        want_y, want_x = _atan2_first(a, b)
        scale = 1 / math.hypot(a, b)
        assert abs(gy - want_y) <= 1e-12 * scale and abs(gx - want_x) <= 1e-12 * scale


# ── hypot ────────────────────────────────────────────────────────────────────────────


def _hypot_second(a: float, b: float) -> tuple[float, float, float]:
    h = math.hypot(a, b)
    ar, br = a / h, b / h
    return br * br / h, -ar * br / h, ar * ar / h  # aa, ab, bb


@pytest.mark.parametrize("sign_b", [1.0, -1.0])
@pytest.mark.parametrize("sign_a", [1.0, -1.0])
def test_hypot_second_derivative_at_a_tie_is_not_zero(
    device: str, sign_a: float, sign_b: float
) -> None:
    """``|a| = |b|``: ``maximum`` / ``minimum`` split the cotangent and the second derivative was 0.

    ``hypot`` is homogeneous of degree 1: at ``|a| = |b| = m`` the Hessian is
    ``1 / (2 sqrt(2) m) * [[1, -s], [-s, 1]]`` with ``s = sign(a) sign(b)``.
    """
    magnitudes = [1e-30, 1e-10, 1.0, 1e10, 1e30]
    a = _leaf([sign_a * m for m in magnitudes], device)
    b = _leaf([sign_b * m for m in magnitudes], device)
    da, db = grad(lucid.hypot(a, b).sum(), [a, b], create_graph=True)
    h_aa, h_ab = grad(da.sum(), [a, b], retain_graph=True)
    (h_bb,) = grad(db.sum(), [b])
    curvature = [1.0 / (2.0 * math.sqrt(2.0) * m) for m in magnitudes]
    sign = sign_a * sign_b
    _close(h_aa.tolist(), curvature, curvature, TOL)
    _close(h_ab.tolist(), [-sign * c for c in curvature], curvature, TOL)
    _close(h_bb.tolist(), curvature, curvature, TOL)


def test_hypot_first_derivative_over_the_grid(device: str) -> None:
    a0, b0 = _pairs(EXPONENTS, EXPONENTS)
    a, b = _leaf(a0, device), _leaf(b0, device)
    da, db = grad(lucid.hypot(a, b).sum(), [a, b])
    rows = list(zip(a.tolist(), b.tolist(), strict=True))
    hs = [math.hypot(x, y) for x, y in rows]
    _close_each(da.tolist(), [x / h for (x, _), h in zip(rows, hs, strict=True)], TOL)
    _close_each(db.tolist(), [y / h for (_, y), h in zip(rows, hs, strict=True)], TOL)


def test_hypot_second_derivative_over_the_grid(device: str) -> None:
    a0, b0 = _pairs(EXPONENTS, EXPONENTS)
    rows = [(x, y) for x, y in zip(a0, b0, strict=True) if 1e-30 < math.hypot(x, y) < 1e30]
    n = len(rows)
    a, b = _leaf([r[0] for r in rows], device), _leaf([r[1] for r in rows], device)
    uw = [[0.7, -1.1, 0.9, 1.3, -0.5][i % 5] for i in range(n)]
    vw = [[1.3, 0.6, -0.8, 1.1, 0.9][i % 5] for i in range(n)]
    da, db = grad(lucid.hypot(a, b).sum(), [a, b], create_graph=True)
    h_a, h_b = grad(
        da * lucid.tensor(uw, device=device) + db * lucid.tensor(vw, device=device),
        [a, b],
        grad_outputs=lucid.ones(n, device=device),
    )
    seconds = [_hypot_second(x, y) for x, y in zip(a.tolist(), b.tolist(), strict=True)]
    scale = [1 / math.hypot(x, y) for x, y in zip(a.tolist(), b.tolist(), strict=True)]
    want_a = [u_ * s[0] + v_ * s[1] for u_, v_, s in zip(uw, vw, seconds, strict=True)]
    want_b = [u_ * s[1] + v_ * s[2] for u_, v_, s in zip(uw, vw, seconds, strict=True)]
    _close(h_a.tolist(), want_a, scale, 1e-4)
    _close(h_b.tolist(), want_b, scale, 1e-4)


def test_hypot_values_and_special_values(device: str) -> None:
    a0, b0 = _pairs(EXPONENTS, EXPONENTS)
    got = lucid.hypot(lucid.tensor(a0, device=device), lucid.tensor(b0, device=device)).tolist()
    for g, x, y in zip(got, a0, b0, strict=True):
        assert _same_value(g, math.hypot(x, y), 2e-6), (x, y, g)
    pool = [-INF, -3.0, 0.0, 4.0, INF, NAN]
    xs = [v for v in pool for _ in pool]
    ys = [v for _ in pool for v in pool]
    got = lucid.hypot(lucid.tensor(xs, device=device), lucid.tensor(ys, device=device)).tolist()
    for g, x, y in zip(got, xs, ys, strict=True):
        assert _same_value(g, math.hypot(x, y), 1e-6), (x, y, g)


def test_hypot_float64_on_the_cpu_keeps_its_range() -> None:
    a0, b0 = [1e-200, 1e200, 3e150, 1e-300], [1e-200, 2e-100, 4e150, 1e-300]
    a, b = _leaf(a0, "cpu", lucid.float64), _leaf(b0, "cpu", lucid.float64)
    out = lucid.hypot(a, b)
    for g, x, y in zip(out.tolist(), a0, b0, strict=True):
        assert _same_value(g, math.hypot(x, y), 1e-14), (x, y, g)
    da, db = grad(out.sum(), [a, b], create_graph=True)
    h_a, _ = grad((da + db).sum(), [a, b])
    for ga, gh, x, y in zip(da.tolist(), h_a.tolist(), a0, b0, strict=True):
        h = math.hypot(x, y)
        assert abs(ga - x / h) <= 1e-12
        want = _hypot_second(x, y)
        assert abs(gh - (want[0] + want[1])) <= 1e-10 / h


def test_complex_abs_is_unchanged_by_the_hypot_rule(device: str) -> None:
    """``abs`` of a complex tensor reads ``hypot_lanes``: forward only, derivative stated."""
    re = [3e-30, 1e30, 0.0, -0.0, 3.0, INF]
    im = [4e-30, 2e30, 0.0, -0.0, -4.0, NAN]
    z = lucid.complex(lucid.tensor(re, device=device), lucid.tensor(im, device=device))
    got = lucid.abs(z).tolist()
    for g, x, y in zip(got, re, im, strict=True):
        assert _same_value(g, math.hypot(x, y), 2e-6), (x, y, g)


@pytest.mark.parametrize("name", ["log1p", "logaddexp", "atan2", "hypot"])
def test_the_recorded_first_derivative_equals_the_eager_one(device: str, name: str) -> None:
    xs, ys = [0.3, -2.0, 1e-8, 4.0, 0.0], [1.5, 0.7, -3.0, 1e-8, 2.0]
    a, b = _leaf(xs, device), _leaf(ys, device)
    if name == "log1p":
        out_fn, leaves = (lambda: lucid.log1p(a)), [a]
    else:
        binary = getattr(lucid, name)
        out_fn, leaves = (lambda: binary(a, b)), [a, b]
    eager = grad(out_fn().sum(), leaves)
    recorded = grad(out_fn().sum(), leaves, create_graph=True)
    for want, got in zip(eager, recorded, strict=True):
        _close(got.tolist(), want.tolist(), [max(abs(w), 1e-30) for w in want.tolist()], 1e-6)
