"""Special functions over their whole real domain (LCD-98, LCD-99, LCD-246).

The defect class: a special-function implementation covered the
convenient half of its domain and answered ordinary-looking numbers
everywhere else.  ``digamma`` and ``polygamma`` had no reflection for
``x < 0`` (``digamma(-7.9)`` was ``-3897.67`` against ``-7.54``, and
NaN below ``-8``), and put ``-inf`` at poles where the reference has
NaN.  ``zeta`` carried one factor too many in every Euler-Maclaurin
term (``zeta(2, 1) = 1.6450854`` against ``pi^2/6``).  ``log1p(+inf)``
was NaN, and its gradient was NaN at ``y <= -1``.  ``lgamma(+inf)`` was
NaN, and ``expm1`` and CPU ``erfinv`` lost the sign of ``-0``.


Every function here is checked on one grid that holds the cases a
one-sided implementation gets wrong: negative non-integers, poles,
``±0``, ``±inf``, NaN, and large and small magnitudes.  The grid runs
in float32 and float64 on CPU, and in float32 on Metal, against the
reference: values and first derivatives.  A few cells differ on purpose
(``_DEVIATIONS``).  There the reference's own value is pinned too, so
that the table goes stale loudly rather than silently.  The closed forms
at the end need no reference.
"""

import math
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.special
from lucid.test._fixtures.devices import metal_available

_INF = math.inf
_NAN = math.nan

_NEG = [-1000.5, -10.5, -7.9, -7.5, -6.5, -3.3, -0.5, -1e-6]
_POLES = [-3.0, -1.0, -0.0, 0.0]
_EDGES = [-_INF, _INF, _NAN]
_POS = [1e-6, 0.3, 1.0, 2.5, 30.0, 1e6]
_GAMMA_GRID = _NEG + _POLES + _EDGES + _POS

_LOG1P_GRID = [-2.0, -1.0, -0.5, -1e-30, -0.0, 0.0, 1e-30, 1e-8, 0.5, 1e30]
_LOG1P_GRID += _EDGES
_EXPM1_GRID = [-100.0, -1.0, -1e-30, -0.0, 0.0, 1e-30, 1e-8, 1.0, 80.0] + _EDGES
_ERFINV_GRID = [-1.5, -1.0, -0.999999, -0.5, -0.0, 0.0, 1e-30, 0.5, 0.999999]
_ERFINV_GRID += [1.0, 1.5, _NAN]

UnaryFn = Callable[[lucid.Tensor], lucid.Tensor]


def _polygamma(n: int) -> UnaryFn:
    return lambda t: lucid.special.polygamma(n, t)


#: name -> (Lucid function, reference attribute path or polygamma order, grid)
_UNARY: dict[str, tuple[UnaryFn, str | int, list[float]]] = {
    "digamma": (lucid.digamma, "digamma", _GAMMA_GRID),
    "psi": (lucid.special.psi, "digamma", _GAMMA_GRID),
    "polygamma0": (_polygamma(0), 0, _GAMMA_GRID),
    "polygamma1": (_polygamma(1), 1, _GAMMA_GRID),
    "polygamma2": (_polygamma(2), 2, _GAMMA_GRID),
    "polygamma3": (_polygamma(3), 3, _GAMMA_GRID),
    "lgamma": (lucid.lgamma, "lgamma", _GAMMA_GRID),
    "log1p": (lucid.log1p, "log1p", _LOG1P_GRID),
    "expm1": (lucid.expm1, "expm1", _EXPM1_GRID),
    "erfinv": (lucid.erfinv, "erfinv", _ERFINV_GRID),
}

_POLE_REASON = (
    "the reference's trigamma takes pi^2 / sin^2(pi x) with sin(pi x) "
    "rounding to ~1e-16 at an integer, so its pole is a large finite number"
)

#: (name, x) -> (the reference's value there, Lucid's value, why they differ).
#: A reference value of ``None`` means "finite and above 1e15".
_DEVIATIONS: dict[tuple[str, float], tuple[float | None, float, str]] = {
    **{("polygamma1", p): (None, _INF, _POLE_REASON) for p in (-3.0, -1.0)},
    ("polygamma1", -_INF): (
        _NAN,
        _INF,
        "-inf is the limit of the poles; polygamma(n >= 2, -inf) is the pole "
        "value in the reference too, and trigamma's NaN is the odd one out",
    ),
    ("polygamma2", _INF): (
        _NAN,
        0.0,
        "psi^(n)(x) -> 0 as x -> inf; the reference's Hurwitz zeta evaluates the tail as inf * 0",
    ),
    ("polygamma3", _INF): (
        _NAN,
        0.0,
        "psi^(n)(x) -> 0 as x -> inf; the reference's Hurwitz zeta evaluates the tail as inf * 0",
    ),
}

#: The gradient is not compared at points where no derivative exists.
_NO_GRADIENT = set(_POLES) | {-_INF}

#: (name, x, dtype) -> why the reference's derivative is the one that is off.
_GRADIENT_DEVIATIONS: dict[tuple[str, float, str], str] = {
    ("erfinv", sign * 0.999999, "float32"): (
        "the reference's float32 erfinv(±0.999999) is ±3.4570882 against "
        "±3.4570747 (Lucid's), and exp(y^2) magnifies that to 1e-4: the "
        "exact derivative is 137390.62"
    )
    for sign in (-1.0, 1.0)
}


def _ref_fn(R: ModuleType, path: str | int) -> Callable[[object], object]:
    if isinstance(path, int):
        return lambda t: R.special.polygamma(path, t)
    return lambda t: getattr(R, path)(t)


def _tol(dtype: str) -> tuple[float, float]:
    return (1e-9, 1e-12) if dtype == "float64" else (2e-5, 1e-6)


def _same(got: float, want: float, dtype: str) -> bool:
    if math.isnan(want):
        return math.isnan(got)
    if math.isinf(want):
        return got == want
    rtol, atol = _tol(dtype)
    return math.isclose(got, want, rel_tol=rtol, abs_tol=atol)


def _same_signed(got: float, want: float, dtype: str) -> bool:
    if want == 0.0 and got == 0.0:
        return math.copysign(1.0, got) == math.copysign(1.0, want)
    return _same(got, want, dtype)


def _placements() -> list[tuple[str, str, list[pytest.MarkDecorator]]]:
    """(dtype, device, marks): float32 and float64 on CPU, float32 on Metal."""
    metal = [pytest.mark.skipif(not metal_available(), reason="needs Metal")]
    return [("float32", "cpu", []), ("float64", "cpu", []), ("float32", "metal", metal)]


def _cases() -> list[object]:
    return [
        pytest.param(name, dtype, dev, marks=marks, id=f"{name}-{dtype}-{dev}")
        for dtype, dev, marks in _placements()
        for name in _UNARY
    ]


def _run(name: str, dtype: str, dev: str) -> tuple[list[float], list[float]]:
    fn, _, grid = _UNARY[name]
    x = lucid.tensor(grid, dtype=getattr(lucid, dtype), device=dev, requires_grad=True)
    out = fn(x)
    out.backward(lucid.ones_like(out))
    assert x.grad is not None
    return out.to("cpu").tolist(), x.grad.to("cpu").tolist()


def _run_ref(R: ModuleType, name: str, dtype: str) -> tuple[list[float], list[float]]:
    _, path, grid = _UNARY[name]
    x = R.tensor(grid, dtype=getattr(R, dtype), requires_grad=True)
    out = _ref_fn(R, path)(x)
    out.backward(R.ones_like(out))
    return out.tolist(), x.grad.tolist()


@pytest.mark.parity
@pytest.mark.parametrize(("name", "dtype", "dev"), _cases())
def test_values_match_the_reference(name: str, dtype: str, dev: str, ref: ModuleType) -> None:
    got, _ = _run(name, dtype, dev)
    want, _ = _run_ref(ref, name, dtype)
    for x, g, w in zip(_UNARY[name][2], got, want, strict=True):
        deviation = _DEVIATIONS.get((name, x))
        if deviation is not None:
            ref_value, ours, why = deviation
            if ref_value is None:
                assert math.isfinite(w) and abs(w) > 1e15, f"reference changed at {x}"
            else:
                assert _same(w, ref_value, dtype), f"reference changed at {x}: {w}"
            assert _same(g, ours, dtype), f"{name}({x}): {g} — {why}"
            continue
        assert _same_signed(g, w, dtype), f"{name}({x}) {dtype} {dev}: {g} != {w}"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cpu_erfinv_keeps_the_sign_of_minus_zero(dtype: str) -> None:
    v = lucid.erfinv(lucid.tensor([-0.0], dtype=getattr(lucid, dtype))).item()
    assert v == 0.0 and math.copysign(1.0, v) == -1.0


@pytest.mark.parity
@pytest.mark.parametrize(("name", "dtype", "dev"), _cases())
def test_gradients_match_the_reference(name: str, dtype: str, dev: str, ref: ModuleType) -> None:
    _, got = _run(name, dtype, dev)
    _, want = _run_ref(ref, name, dtype)
    for x, g, w in zip(_UNARY[name][2], got, want, strict=True):
        if x in _NO_GRADIENT or (name, x) in _DEVIATIONS or math.isnan(x):
            continue
        if (name, x, dtype) in _GRADIENT_DEVIATIONS:
            continue
        if name.startswith(("digamma", "psi", "polygamma")) and x == _INF:
            # d/dx psi^(n) at +inf is psi^(n+1)(inf): 0 here, and the
            # reference's NaN for n + 1 >= 2 is the deviation above.
            assert g == 0.0, f"{name}'({x}): {g}"
            continue
        assert _same(g, w, dtype), f"{name}'({x}) {dtype} {dev}: {g} != {w}"


# ── zeta ────────────────────────────────────────────────────────────────────

_ZETA_XS = [1.0, 1.5, 2.0, 3.0, 4.0, 10.0, 0.5, _INF, _NAN]
_ZETA_QS = [1.0, 0.5, 2.0, 1e-3, 1e6, -1.5, -20.5, -0.25, -2.0, 0.0, _NAN]
_ZETA_GRID = [(x, q) for x in _ZETA_XS for q in _ZETA_QS]


#: (x, q, dtype) -> why the reference's d/dq is the one that is off.
_ZETA_GRADIENT_DEVIATIONS: dict[tuple[float, float, str], str] = {
    (10.0, -1.5, "float64"): (
        "-10 zeta(11, -1.5) = -4.30526886555e-4 sums terms of size 2^11 that "
        "cancel; the reference's direct sum is 5e-9 off, Lucid's is exact"
    ),
}


def _zeta_cases() -> list[object]:
    return [
        pytest.param(dtype, dev, marks=marks, id=f"{dtype}-{dev}")
        for dtype, dev, marks in _placements()
    ]


@pytest.mark.parity
@pytest.mark.parametrize(("dtype", "dev"), _zeta_cases())
def test_zeta_matches_the_reference(dtype: str, dev: str, ref: ModuleType) -> None:
    xs = [p[0] for p in _ZETA_GRID]
    qs = [p[1] for p in _ZETA_GRID]
    q = lucid.tensor(qs, dtype=getattr(lucid, dtype), device=dev, requires_grad=True)
    out = lucid.special.zeta(lucid.tensor(xs, dtype=getattr(lucid, dtype), device=dev), q)
    out.backward(lucid.ones_like(out))
    assert q.grad is not None
    got, got_dq = out.to("cpu").tolist(), q.grad.to("cpu").tolist()

    rq = ref.tensor(qs, dtype=getattr(ref, dtype), requires_grad=True)
    rout = ref.special.zeta(ref.tensor(xs, dtype=getattr(ref, dtype)), rq)
    rout.backward(ref.ones_like(rout))
    want, want_dq = rout.tolist(), rq.grad.tolist()

    for (x, qv), g, w, gd, wd in zip(_ZETA_GRID, got, want, got_dq, want_dq, strict=True):
        assert _same(g, w, dtype), f"zeta({x}, {qv}) {dtype} {dev}: {g} != {w}"
        # d/dq zeta(x, q) = -x zeta(x + 1, q) where the value is finite.
        if (x, qv, dtype) in _ZETA_GRADIENT_DEVIATIONS:
            continue
        if math.isfinite(w) and math.isfinite(wd):
            assert _same(gd, wd, dtype), f"d/dq zeta({x}, {qv}): {gd} != {wd}"


def test_zeta_at_plus_inf_q_is_the_limit() -> None:
    """``zeta(x, q) -> 0`` as ``q -> inf``; the reference answers NaN."""
    out = lucid.special.zeta(lucid.tensor([2.0, 4.0]), lucid.tensor([_INF, _INF]))
    assert out.tolist() == [0.0, 0.0]


# ── Closed forms, no reference needed ───────────────────────────────────────


def test_riemann_zeta_closed_forms_in_float64() -> None:
    x = lucid.tensor([2.0, 4.0, 6.0], dtype=lucid.float64)
    got = lucid.special.zeta(x, lucid.ones_like(x)).tolist()
    want = [math.pi**2 / 6.0, math.pi**4 / 90.0, math.pi**6 / 945.0]
    for g, w in zip(got, want, strict=True):
        assert math.isclose(g, w, rel_tol=1e-15), (g, w)


def test_zeta_negative_q_reduces_to_the_direct_sum() -> None:
    """zeta(x, q) = sum_{k<m} (q + k)^-x + zeta(x, q + m), for integer x.

    The terms at q + k = -0.5 and 0.5 are -8 and 8 and cancel against a
    result of 1e-3, so either side carries about 1e-15 absolute.
    """
    q = -20.5
    head = math.fsum((q + k) ** -3 for k in range(30))
    tail = lucid.special.zeta(
        lucid.tensor([3.0], dtype=lucid.float64),
        lucid.tensor([q + 30], dtype=lucid.float64),
    ).item()
    got = lucid.special.zeta(
        lucid.tensor([3.0], dtype=lucid.float64), lucid.tensor([q], dtype=lucid.float64)
    ).item()
    assert math.isclose(got, head + tail, rel_tol=1e-11), (got, head + tail)


_EULER_GAMMA = 0.57721566490153286061


@pytest.mark.parametrize("dev", ["cpu", "metal"])
def test_digamma_and_its_derivatives_at_minus_one_half(dev: str) -> None:
    """psi(-1/2) = 2 - gamma - 2 log 2, psi'(-1/2) = pi^2/2 + 4 and
    psi''(-1/2) = 16 - 14 zeta(3), the last two through autograd."""
    if dev == "metal" and not metal_available():
        pytest.skip("needs Metal")
    dtype = lucid.float64 if dev == "cpu" else lucid.float32
    # The value is psi(3/2) = 0.0365, a difference of series terms of size
    # 2, so it carries the dtype's epsilon times 2 in absolute terms.
    atol = 1e-14 if dev == "cpu" else 1e-6
    rtol = 1e-14 if dev == "cpu" else 1e-6
    x = lucid.tensor([-0.5], dtype=dtype, device=dev, requires_grad=True)
    y = lucid.digamma(x)
    (g,) = lucid.autograd.grad(y.sum(), x, create_graph=True)
    (h,) = lucid.autograd.grad(g.sum(), x)
    want = 2.0 - _EULER_GAMMA - 2.0 * math.log(2.0)
    assert math.isclose(y.item(), want, abs_tol=atol)
    assert math.isclose(g.item(), math.pi**2 / 2.0 + 4.0, rel_tol=rtol)
    assert math.isclose(h.item(), 16.0 - 14.0 * 1.2020569031595942, rel_tol=rtol * 10)


def test_digamma_poles_and_signed_zero() -> None:
    out = lucid.digamma(lucid.tensor([-3.0, -1.0, -_INF, 0.0, -0.0])).tolist()
    assert all(math.isnan(v) for v in out[:3]), out
    assert out[3:] == [-_INF, _INF], out


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_negative_half_obeys_the_recurrence(n: int) -> None:
    """psi^(n)(x + 1) - psi^(n)(x) = (-1)^n n! / x^(n+1), at negative x.

    Independent of the reference, and it only holds if the reflected left
    half and the shifted right half agree — the pair -0.5 / 0.5 straddles
    the two.  Near the poles both operands are large and their difference
    is small, so the tolerance scales with the operands.
    """
    x = lucid.tensor([-7.9, -3.3, -0.5, -0.1], dtype=lucid.float64)
    hi = lucid.special.polygamma(n, x + 1.0).tolist()
    lo = lucid.special.polygamma(n, x).tolist()
    for xv, a, b in zip(x.tolist(), hi, lo, strict=True):
        want = (-1.0) ** n * math.factorial(n) / xv ** (n + 1)
        assert abs((a - b) - want) <= 1e-14 * max(abs(a), abs(b)), (n, xv, a - b, want)


def test_trigamma_reflection_identity() -> None:
    """psi'(1 - x) + psi'(x) = pi^2 / sin^2(pi x)."""
    x = lucid.tensor([-7.9, -3.3, -0.5, -0.1], dtype=lucid.float64)
    s = lucid.special.polygamma(1, 1.0 - x) + lucid.special.polygamma(1, x)
    for xv, got in zip(x.tolist(), s.tolist(), strict=True):
        want = math.pi**2 / math.sin(math.pi * xv) ** 2
        assert math.isclose(got, want, rel_tol=1e-12), (xv, got, want)


@pytest.mark.parametrize("dev", ["cpu", "metal"])
def test_log1p_edges_value_and_gradient(dev: str) -> None:
    if dev == "metal" and not metal_available():
        pytest.skip("needs Metal")
    ys = [_INF, -1.0, -2.0, -0.0, 1e-30, 1e38]
    y = lucid.tensor(ys, device=dev, requires_grad=True)
    out = lucid.log1p(y)
    out.backward(lucid.ones_like(out))
    assert y.grad is not None
    v, g = out.to("cpu").tolist(), y.grad.to("cpu").tolist()
    assert v[0] == _INF and v[1] == -_INF and math.isnan(v[2])
    assert v[3] == 0.0 and math.copysign(1.0, v[3]) == -1.0
    assert v[4] == pytest.approx(1e-30, rel=1e-6)
    assert v[5] == pytest.approx(math.log1p(1e38), rel=1e-6)
    assert g[0] == 0.0 and g[1] == _INF and g[2] == -1.0
    assert g[3] == 1.0 and g[4] == 1.0
