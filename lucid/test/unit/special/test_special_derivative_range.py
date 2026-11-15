"""LCD-1030: ``spherical_bessel_j0`` and ``erfcx`` differentiate across their range.

Both lost their derivatives where an intermediate cancelled while the result
was an ordinary number:

* ``spherical_bessel_j0`` differentiated ``sin(x) / x`` through the quotient.  Its
  slope ``cos(x) / x - sin(x) / x²`` is a difference of two ``1 / x``'s, and each
  further order has a larger power of ``x`` under it: the slope was 0 at ``1e-4``
  (it is ``-3.3e-5``), the curvature 0, 1, ``-0.125`` or 2 where it is ``-1/3``
  (``0`` exactly at the origin too, a constant ``where`` branch).  It is now a
  series in ``x²`` on ``|x| < 1``, exact to every order through the origin.
* ``erfcx`` took the reference's analytic slope ``2 x erfcx(x) - 2/sqrt(pi)``, whose
  two terms both tend to ``2/sqrt(pi)``: from ``x = 2`` their difference lost its
  digits (1.7% off at 370, 0 from 1e4 in float32), and the second derivative,
  built from it, 12% off at 37, five-fold at 100 and twenty orders at 1e10.  From
  ``x = 1.5`` the slope is ``-2/sqrt(pi) f / (x + f)`` for the continued
  fraction's tail ``f``, a quotient of positive numbers.

The oracles are plain float64 / ``decimal`` closed forms evaluated in this
file, at the float32 values read back from the tensors.
"""

import math
from collections.abc import Callable
from decimal import Decimal, localcontext
from functools import cache

import pytest

import lucid
import lucid.special as S
from lucid.test._fixtures.devices import device_dtype_params, devices_supporting

_FLOATS = device_dtype_params([lucid.float32, lucid.float64])
_HALVES = device_dtype_params([lucid.float16, lucid.bfloat16])

_PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")
_F32_NORMAL_MIN = 1.2e-38


def _derivatives(
    fn: Callable[[lucid.Tensor], lucid.Tensor],
    xs: list[float],
    device: str,
    dtype: lucid.dtype,
    orders: int,
) -> list[list[float]]:
    """``[value, d1, .., d<orders>]`` of ``fn`` at ``xs``, each by ``grad`` of the one before."""
    x = lucid.tensor(xs, dtype=dtype, device=device, requires_grad=True)
    out = [fn(x)]
    for order in range(orders):
        (g,) = lucid.autograd.grad(out[-1].sum(), x, create_graph=order < orders - 1)
        out.append(g)
    return [t.tolist() for t in out]


def _rounded(xs: list[float], dtype: lucid.dtype) -> list[float]:
    """``xs`` as the tensors of ``dtype`` hold them: the oracles see these values."""
    return lucid.tensor(xs, dtype=dtype).tolist()


def _check(
    name: str, x: float, order: int, got: float, want: float, rtol: float, dtype: lucid.dtype
) -> None:
    """``got`` within ``rtol`` of ``want`` in relative terms, a zero exactly zero.

    float32 results below its normal range are skipped: the CPU keeps denormals
    and Metal flushes them, and neither answer is wrong.
    """
    if dtype == lucid.float32 and 0.0 < abs(want) < _F32_NORMAL_MIN:
        return
    where = f"{name} d{order} at {x!r} ({dtype}): got {got!r}, want {want!r}"
    if want == 0.0:
        assert got == 0.0, where
    else:
        assert abs(got - want) <= rtol * abs(want), where


def _check_all(
    name: str,
    xs: list[float],
    got: list[list[float]],
    oracle: Callable[[float, int], float],
    rtols: tuple[float, ...],
    dtype: lucid.dtype,
    orders: range = range(4),
) -> None:
    for order in orders:
        for x, g in zip(xs, got[order], strict=True):
            _check(name, x, order, g, oracle(x, order), rtols[order], dtype)


# ── spherical_bessel_j0 ─────────────────────────────────────────────────────


def _j0_derivative(x: float, n: int) -> float:
    """``d^n/dx^n sin(x) / x`` in float64: its Maclaurin series up to 6, closed forms beyond."""
    if abs(x) <= 6.0:
        return sum(
            (-1.0) ** k / ((2 * k + 1) * math.factorial(2 * k - n)) * x ** (2 * k - n)
            for k in range((n + 1) // 2, 40)
        )
    s, c = math.sin(x), math.cos(x)
    return (
        s / x,
        c / x - s / x**2,
        -s / x - 2 * c / x**2 + 2 * s / x**3,
        -c / x + 3 * s / x**2 + 6 * c / x**3 - 6 * s / x**4,
    )[n]


#: Where the quotient's graph was wrong (the origin, its neighbourhood, and the
#: float32 range around it), then the series' edge at 1.
_J0_NEAR = [
    0.0, 1e-30, 1e-8, 1e-4, 3.7e-4, 1e-3, 3.7e-3, 0.05, 0.3, 0.9, -1e-4, -0.5, -3.7e-4,
]  # fmt: skip
#: Beyond the series, clear of the zeros of every order (where a relative
#: tolerance means nothing).
_J0_FAR = [
    1.0, 1.5, 2.5, 3.5, 5.0, 6.5, 8.0, 12.0, 25.5, 100.5, 1000.5, 10000.5, -1.2, -7.0, -40.5,
]  # fmt: skip
#: Per order: float32 then float64.  The third derivative of a quotient loses a
#: digit or two near 1 (its terms are 35 times its value), so it is looser.
#: float32 order 2 measured 0.92e-6 at x = -1.2 (just past the series) on an M1 Pro; the bound
#: leaves room for the chip's float32 sin/cos to differ (LCD-1015 saw that on an M4 Max).
_J0_RTOL = {lucid.float32: (4e-7, 1e-6, 2e-6, 1e-5), lucid.float64: (1e-14, 1e-13, 1e-13, 1e-12)}


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("xs", [_J0_NEAR, _J0_FAR], ids=["near-origin", "beyond-series"])
def test_spherical_bessel_j0_derivatives_match_the_closed_form(
    device: str, dtype: lucid.dtype, xs: list[float]
) -> None:
    xs = _rounded(xs, dtype)
    got = _derivatives(S.spherical_bessel_j0, xs, device, dtype, 3)
    _check_all("spherical_bessel_j0", xs, got, _j0_derivative, _J0_RTOL[dtype], dtype)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_spherical_bessel_j0_curvature_is_minus_a_third_through_the_origin(
    device: str, dtype: lucid.dtype
) -> None:
    # The card's case: cpu answered [0, 0, 1, -0.125, -0.32], metal [0, 0, 2, -0.5, -0.328].
    xs = _rounded([0.0, 1e-4, 3.7e-4, 1e-3, 3.7e-3], dtype)
    _, d1, d2, d3 = _derivatives(S.spherical_bessel_j0, xs, device, dtype, 3)
    for x, g in zip(xs, d2, strict=True):
        assert math.isclose(g, -1 / 3, rel_tol=1e-5), (x, g)
    # the slope was 0 at 1e-4 (-3.33e-5), -2.4e-4 at 3.7e-4 (-1.23e-4)
    for x, g in zip(xs[1:], d1[1:], strict=True):
        assert math.isclose(g, -x / 3, rel_tol=2e-5), (x, g)
    assert d1[0] == 0.0 and d3[0] == 0.0


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_spherical_bessel_j0_sweep_across_the_series_seam(device: str, dtype: lucid.dtype) -> None:
    """Every order is smooth across ``|x| = 1``, where the series hands over to the quotient."""
    xs = _rounded([0.9 + 0.2 * i / 40 for i in range(41)], dtype)
    xs += [-x for x in xs]
    got = _derivatives(S.spherical_bessel_j0, xs, device, dtype, 3)
    _check_all("spherical_bessel_j0", xs, got, _j0_derivative, _J0_RTOL[dtype], dtype)


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_spherical_bessel_j0_special_values_and_their_slopes(device: str) -> None:
    xs = [math.inf, -math.inf, math.nan, 0.0, -0.0]
    value, d1, d2, d3 = _derivatives(S.spherical_bessel_j0, xs, device, lucid.float32, 3)
    # ``sin(inf) / inf`` is NaN, the envelope ``1 / |x|`` carries the limit to 0.
    assert value[:2] == [0.0, 0.0] and d1[:2] == [0.0, 0.0] and d2[:2] == [0.0, 0.0]
    assert d3[:2] == [0.0, 0.0]
    assert all(math.isnan(v[2]) for v in (value, d1, d2, d3))
    assert value[3:] == [1.0, 1.0]
    assert d1[3:] == [0.0, 0.0] and d3[3:] == [0.0, 0.0]
    assert all(math.isclose(g, -1 / 3, rel_tol=1e-6) for g in d2[3:])


@pytest.mark.parametrize("dtype", [lucid.int32, lucid.int64, lucid.bool_])
def test_spherical_bessel_j0_integer_operand_is_the_default_float(dtype: lucid.dtype) -> None:
    out = S.spherical_bessel_j0(lucid.tensor([0, 1, 1], dtype=dtype))
    assert out.dtype == lucid.get_default_dtype()
    assert out.tolist() == pytest.approx([1.0, math.sin(1.0), math.sin(1.0)], rel=1e-6)


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_sinc_keeps_its_curvature_at_zero(device: str) -> None:
    # sinc(x) = j0(pi x), so sinc^(n)(x) = pi^n j0^(n)(pi x): curvature -pi² / 3 at 0.
    xs = _rounded([0.0, 1e-4, 1e-2, 0.2], lucid.float32)
    _, d1, d2 = _derivatives(lucid.sinc, xs, device, lucid.float32, 2)
    for x, g1, g2 in zip(xs, d1, d2, strict=True):
        u = math.pi * x
        _check("sinc", x, 1, g1, math.pi * _j0_derivative(u, 1), 1e-5, lucid.float32)
        _check("sinc", x, 2, g2, math.pi**2 * _j0_derivative(u, 2), 1e-5, lucid.float32)


# ── bessel_j0, float64: the same defect through |x| ─────────────────────────


@pytest.mark.parametrize("device", devices_supporting(lucid.float64))
def test_bessel_j0_float64_keeps_its_curvature_at_the_origin(device: str) -> None:
    """``J0`` is a function of ``x²``; read through ``|x|`` its curvature at 0 was 0."""
    xs = [0.0, -0.0, 1e-30, 1e-9, -2.0, 4.9999, 5.0, 5.0001]
    _, d1, d2 = _derivatives(S.bessel_j0, xs, device, lucid.float64, 2)
    assert d1[:2] == [0.0, 0.0]
    for g in d2[:4]:
        assert math.isclose(g, -0.5, rel_tol=1e-12), g
    # J0'' = -J0 + J1 / x, either side of the rational form's own edge at 5
    for x, g in zip(xs[4:], d2[4:], strict=True):
        want = -S.bessel_j0(lucid.tensor(x, dtype=lucid.float64)).item() + (
            S.bessel_j1(lucid.tensor(x, dtype=lucid.float64)).item() / x
        )
        assert math.isclose(g, want, rel_tol=1e-9), (x, g, want)


# ── erfcx ───────────────────────────────────────────────────────────────────


@cache
def _erfcx_exact(x: float) -> tuple[float, float, float, float]:
    """``erfcx(x)`` and its first three derivatives, at 90 digits.

    ``erfcx'(x) = 2 x erfcx(x) - 2/sqrt(pi)``, so ``erfcx'' = 2 erfcx + 2 x erfcx'`` and
    ``erfcx''' = 4 erfcx' + 2 x erfcx''``.  The value is the continued fraction from
    ``x = 2`` and ``exp(x²) (1 - erf(x))`` below it, the error function by its
    Maclaurin series: at 90 digits the subtractions lose nothing a float64 can see up to
    1e6; beyond 1e8 the leading term ``(-1)^n n! / (sqrt(pi) x^(n+1))`` is exact to
    float64 (its correction is ``O(1 / x²)``), and the subtractions would not be.
    """
    if x >= 1e8:
        lead = [
            (-1.0) ** n * math.factorial(n) / (math.sqrt(math.pi) * x ** (n + 1)) for n in range(4)
        ]
        return lead[0], lead[1], lead[2], lead[3]
    with localcontext() as ctx:
        ctx.prec = 90
        xd = Decimal(x)
        sqrt_pi = _PI.sqrt()
        c = 2 / sqrt_pi
        if x >= 2.0:
            tail = Decimal(0)
            for k in range(400, 0, -1):
                tail = Decimal(k) / 2 / (xd + tail)
            y = 1 / (sqrt_pi * (xd + tail))
        else:
            term, total = xd, Decimal(0)
            for n in range(400):
                total += term / (2 * n + 1)
                term = -term * xd * xd / (n + 1)
            y = (xd * xd).exp() * (1 - c * total)
        d1 = 2 * xd * y - c
        d2 = 2 * y + 2 * xd * d1
        d3 = 4 * d1 + 2 * xd * d2
        return float(y), float(d1), float(d2), float(d3)


#: From the value's cutoff (2) to the end of float32's range and past it.
_ERFCX_TAIL = [
    2.0, 2.0001, 2.5, 3.0, 4.0, 6.0, 10.0, 37.0, 100.0, 370.0, 1e3, 1e4, 1e5, 1e6, 1e10, 1e30,
]  # fmt: skip
#: The slope's cutoff (1.5) to the value's: the slope is the tail form here, the value
#: the direct one (whose float64 ``erfc`` is itself good to 1e-14 only).
_ERFCX_BRIDGE = [1.5, 1.6, 1.75, 1.9, 1.99, 1.9999]
#: Below: the direct form, whose conditioning is the limit (``2 x y`` is up to 5 times the slope).
_ERFCX_BODY = [-6.0, -3.0, -1.0, -0.1, 0.0, 1e-3, 0.5, 1.0, 1.25, 1.4]
_ERFCX_RTOL = {
    lucid.float32: {"tail": (4e-7, 5e-7, 1e-6, 1e-6), "body": (4e-6, 5e-6, 2e-5, 5e-5)},
    lucid.float64: {"tail": (5e-15, 5e-15, 1e-14, 2e-14), "body": (2e-14, 1e-13, 3e-13, 1e-12)},
}


def _erfcx_derivative(x: float, n: int) -> float:
    return _erfcx_exact(x)[n]


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_erfcx_derivatives_in_the_tail_match_the_exact_form(
    device: str, dtype: lucid.dtype
) -> None:
    xs = _rounded(_ERFCX_TAIL, dtype)
    got = _derivatives(S.erfcx, xs, device, dtype, 3)
    _check_all("erfcx", xs, got, _erfcx_derivative, _ERFCX_RTOL[dtype]["tail"], dtype)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_erfcx_slope_between_its_cutoff_and_the_values(device: str, dtype: lucid.dtype) -> None:
    xs = _rounded(_ERFCX_BRIDGE, dtype)
    got = _derivatives(S.erfcx, xs, device, dtype, 3)
    rtols = _ERFCX_RTOL[dtype]["tail"]
    _check_all("erfcx", xs, got, _erfcx_derivative, rtols, dtype, orders=range(1, 4))


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_erfcx_derivatives_below_the_cutoff_are_unchanged(device: str, dtype: lucid.dtype) -> None:
    xs = _rounded(_ERFCX_BODY, dtype)
    got = _derivatives(S.erfcx, xs, device, dtype, 3)
    _check_all("erfcx", xs, got, _erfcx_derivative, _ERFCX_RTOL[dtype]["body"], dtype)


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_erfcx_slope_at_the_cards_points(device: str) -> None:
    # float32: d1 at [100, 1e4] was [-5.64e-5, 0] for [-5.641e-5, -5.642e-9]; 1.7% off at 370.
    points = [100.0, 370.0, 1e4]
    d1 = _derivatives(S.erfcx, points, device, lucid.float32, 1)[1]
    assert d1[0] == pytest.approx(-5.641e-5, rel=1e-3)
    assert d1[1] == pytest.approx(-4.121e-6, rel=1e-3)
    assert d1[2] == pytest.approx(-5.642e-9, rel=1e-3)
    # d2 was 12% off at 37, 5x at 100, 20 orders at 1e10.
    points = [37.0, 100.0, 1e10]
    d2 = _derivatives(S.erfcx, points, device, lucid.float32, 2)[2]
    for x, g in zip(points, d2, strict=True):
        assert math.isclose(g, _erfcx_exact(x)[2], rel_tol=1e-6), (x, g)


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_erfcx_first_derivative_without_a_graph_is_the_recorded_one(device: str) -> None:
    xs = [-2.0, 0.0, 1.7, 3.0, 50.0, 1e4]
    x = lucid.tensor(xs, device=device, requires_grad=True)
    S.erfcx(x).sum().backward()
    assert x.grad is not None
    recorded = _derivatives(S.erfcx, xs, device, lucid.float32, 1)[1]
    assert x.grad.tolist() == recorded


@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float64])
def test_erfcx_special_values_are_unchanged(dtype: lucid.dtype) -> None:
    """The reference's: -inf is -inf, +inf is ``inf * 0``, zero is ``-2 / sqrt(pi)`` and 2."""
    xs = [-math.inf, math.inf, math.nan, 0.0, -0.0]
    value, d1, d2, _ = _derivatives(S.erfcx, xs, "cpu", dtype, 3)
    assert value[0] == math.inf and value[1] == 0.0 and math.isnan(value[2])
    assert d1[0] == -math.inf and d2[0] == math.inf
    assert math.isnan(d1[1]) and math.isnan(d2[1])
    assert math.isnan(d1[2]) and math.isnan(d2[2])
    for g in d1[3:]:
        assert math.isclose(g, -2 / math.sqrt(math.pi), rel_tol=1e-6)
    assert d2[3:] == [2.0, 2.0]


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_erfcx_a_masked_branch_that_overflows_does_not_reach_a_neighbour(device: str) -> None:
    """``2 x`` overflows float32 at 1.7e38 while ``erfcx`` is a number: ``inf * 0`` stays masked."""
    xs = [5.0, 3.0e38, math.inf, math.nan, 1e4, -math.inf, 2.0]
    whole = _derivatives(S.erfcx, xs, device, lucid.float32, 3)
    for i, x in enumerate(xs):
        alone = _derivatives(S.erfcx, [x], device, lucid.float32, 3)
        for order in range(4):
            a, b = whole[order][i], alone[order][0]
            assert (math.isnan(a) and math.isnan(b)) or a == b, (x, order, a, b)
    assert whole[1][1] == 0.0 and whole[2][1] == 0.0


@pytest.mark.parametrize(("device", "dtype"), _HALVES)
def test_erfcx_half_precision_slope_in_the_tail(device: str, dtype: lucid.dtype) -> None:
    xs = _rounded([3.0, 10.0, 40.0], dtype)
    got = _derivatives(S.erfcx, xs, device, dtype, 2)
    for order in (1, 2):
        for x, g in zip(xs, got[order], strict=True):
            assert math.isclose(g, _erfcx_exact(x)[order], rel_tol=3e-2), (dtype, x, order, g)
