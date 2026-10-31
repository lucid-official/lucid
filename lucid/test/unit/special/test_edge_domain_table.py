"""One edge-domain table for ``lucid.special`` and the top-level specials (LCD-772).

The defect class: a special function was checked on the points its author
thought of, and answered something ordinary-looking at the edges nobody
tried.  ``expm1`` overflowed to ``inf`` ten orders of magnitude before the
answer does; ``lgamma``, ``entr``, ``logit`` and the half-line Bessel
functions had a gradient of ``0`` (or a finite number) where the value is
``inf`` or NaN; ``polygamma(1.5, x)`` silently ran order 1;
``spherical_bessel_j0(inf)`` was NaN where the limit is 0; the orthogonal
polynomials overflowed to NaN through an ``inf - inf`` in the recurrence;
``expit`` refused a complex tensor.

Each row below is one function.  Every row runs on one grid per dtype that
holds what a one-sided implementation gets wrong: ``±inf``, NaN, ``±0``,
tiny and huge magnitudes, both signs, and the dtype's overflow thresholds
(``exp`` overflows at 88.7 in float32 and 709.8 in float64).  Values and
first derivatives are compared with the reference, float32 and float64 on CPU
and float32 on Metal.  Where the reference itself has no derivative (the
Bessel and polynomial families) the gradient is pinned to the rule the module
documents instead.  Every cell that differs on purpose is listed with its
reason in ``_VALUE_DEVIATIONS`` / ``_GRAD_DEVIATIONS``; the reference's own
answer is pinned there too, so that the table goes stale loudly.

Binary specials (``xlogy``, ``xlog1py``, ``zeta``, ``gammainc``,
``gammaincc``) have their own tables beside this one.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest

import lucid
import lucid.special as S
from lucid.test._fixtures.devices import metal_available

_INF = math.inf
_NAN = math.nan

UnaryFn = Callable[[lucid.Tensor], lucid.Tensor]
RefFn = Callable[[ModuleType, Any], Any]


@dataclass(frozen=True)
class _Row:
    """One function: Lucid's, the reference's, and where the reference falls short."""

    name: str
    fn: UnaryFn
    ref: RefFn
    #: The polynomial recurrences: the reference overflows to NaN or to a
    #: wrong-signed ``inf`` there, so the expected value is Lucid's own float64
    #: evaluation, cast.  They also leave out the 1e30 probe: past ~2**63 a
    #: single step's multiplier ``2x`` overflows by itself, which the rescaling
    #: of the recurrence does not reach.
    recurrence: bool = False

    def grid(self, dtype: str) -> list[float]:
        """Signs, zeros, scales, infinities, NaN and the dtype's overflow thresholds."""
        thresholds = [85.0, 88.0, 88.7, 89.0] if dtype == "float32" else [705.0, 709.0, 709.8]
        huge = [] if self.recurrence else [1e30 if dtype == "float32" else 1e300]
        return [
            -_INF, -1e4, -30.0, -2.5, -1.0, -0.5, -1e-30, -0.0,
            0.0, 1e-30, 1e-6, 0.3, 1.0, 2.5, 8.0, 30.0, 100.0, 1e4, _INF, _NAN,
            *thresholds, *huge,
        ]  # fmt: skip


def _top(attr: str) -> RefFn:
    return lambda R, t: getattr(R, attr)(t)


def _spec(attr: str) -> RefFn:
    return lambda R, t: getattr(R.special, attr)(t)


def _poly(name: str, n: int) -> _Row:
    return _Row(
        f"{name}[{n}]",
        lambda t: getattr(S, name)(t, n),
        lambda R, t: getattr(R.special, name)(t, n),
        recurrence=True,
    )


def _polygamma_row(n: int) -> _Row:
    return _Row(
        f"polygamma[{n}]", lambda t: S.polygamma(n, t), lambda R, t: R.special.polygamma(n, t)
    )


def _multigammaln_row(p: int) -> _Row:
    return _Row(
        f"multigammaln[{p}]",
        lambda t: S.multigammaln(t, p),
        lambda R, t: R.special.multigammaln(t, p),
    )


_FAMILIES = (
    "chebyshev_polynomial_t",
    "chebyshev_polynomial_u",
    "chebyshev_polynomial_v",
    "chebyshev_polynomial_w",
    "shifted_chebyshev_polynomial_t",
    "hermite_polynomial_h",
    "hermite_polynomial_he",
    "legendre_polynomial_p",
    "laguerre_polynomial_l",
)

_ROWS: list[_Row] = [
    _Row("expm1", lucid.expm1, _top("expm1")),
    _Row("log1p", lucid.log1p, _top("log1p")),
    _Row("logit", lucid.logit, _top("logit")),
    _Row("lgamma", lucid.lgamma, _top("lgamma")),
    _Row("digamma", lucid.digamma, _top("digamma")),
    _Row("erf", lucid.erf, _top("erf")),
    _Row("erfc", lucid.erfc, _top("erfc")),
    _Row("erfinv", lucid.erfinv, _top("erfinv")),
    _Row("i0", lucid.i0, _top("i0")),
    _Row("sinc", lucid.sinc, _top("sinc")),
    _Row("erfcx", S.erfcx, _spec("erfcx")),
    _Row("i0e", S.i0e, _spec("i0e")),
    _Row("i1", S.i1, _spec("i1")),
    _Row("i1e", S.i1e, _spec("i1e")),
    _Row("ndtr", S.ndtr, _spec("ndtr")),
    _Row("ndtri", S.ndtri, _spec("ndtri")),
    _Row("log_ndtr", S.log_ndtr, _spec("log_ndtr")),
    _Row("entr", S.entr, _spec("entr")),
    _Row("spherical_bessel_j0", S.spherical_bessel_j0, _spec("spherical_bessel_j0")),
    _Row("bessel_j0", S.bessel_j0, _spec("bessel_j0")),
    _Row("bessel_j1", S.bessel_j1, _spec("bessel_j1")),
    _Row("bessel_y0", S.bessel_y0, _spec("bessel_y0")),
    _Row("bessel_y1", S.bessel_y1, _spec("bessel_y1")),
    _Row("modified_bessel_k0", S.modified_bessel_k0, _spec("modified_bessel_k0")),
    _Row("modified_bessel_k1", S.modified_bessel_k1, _spec("modified_bessel_k1")),
    _Row("scaled_k0", S.scaled_modified_bessel_k0, _spec("scaled_modified_bessel_k0")),
    _Row("scaled_k1", S.scaled_modified_bessel_k1, _spec("scaled_modified_bessel_k1")),
    _Row("modified_bessel_i0", S.modified_bessel_i0, _spec("modified_bessel_i0")),
    _Row("modified_bessel_i1", S.modified_bessel_i1, _spec("modified_bessel_i1")),
    _Row("gammaln", S.gammaln, _spec("gammaln")),
    _Row("psi", S.psi, _spec("psi")),
    _Row("expit", S.expit, _spec("expit")),
    *[_polygamma_row(n) for n in range(4)],
    *[_multigammaln_row(p) for p in (1, 2)],
    *[_poly(name, n) for name in _FAMILIES for n in (1, 5, 50)],
]
_BY_NAME = {row.name: row for row in _ROWS}

#: The reference cannot differentiate these: their gradient is pinned by the
#: rule tests below, not compared.
_NO_REFERENCE_GRADIENT = {
    "bessel_j0", "bessel_j1", "bessel_y0", "bessel_y1", "modified_bessel_k0",
    "modified_bessel_k1", "scaled_k0", "scaled_k1",
}  # fmt: skip


def _cases() -> list[object]:
    """float32 and float64 on CPU, float32 on Metal."""
    metal = [pytest.mark.skipif(not metal_available(), reason="needs Metal")]
    placements = [("float32", "cpu", []), ("float64", "cpu", []), ("float32", "metal", metal)]
    return [
        pytest.param(row.name, dtype, dev, marks=marks, id=f"{row.name}-{dtype}-{dev}")
        for dtype, dev, marks in placements
        for row in _ROWS
    ]


def _same(got: float, want: float, dtype: str, rtol: float | None = None) -> bool:
    if math.isnan(want):
        return math.isnan(got)
    if math.isinf(want):
        return got == want
    base, atol = (1e-9, 1e-12) if dtype == "float64" else (2e-5, 1e-6)
    return math.isclose(got, want, rel_tol=rtol or base, abs_tol=atol)


def _dtype_max(dtype: str) -> float:
    return 3.4028234663852886e38 if dtype == "float32" else 1.7976931348623157e308


def _run(row: _Row, dtype: str, dev: str) -> tuple[list[float], list[float] | None]:
    x = lucid.tensor(row.grid(dtype), dtype=getattr(lucid, dtype), device=dev, requires_grad=True)
    out = row.fn(x)
    if not out.requires_grad:  # a polynomial of degree 0 is a constant
        return out.to("cpu").tolist(), None
    out.backward(lucid.ones_like(out))
    assert x.grad is not None
    return out.to("cpu").tolist(), x.grad.to("cpu").tolist()


def _run_ref(R: ModuleType, row: _Row, dtype: str) -> tuple[list[float], list[float] | None]:
    x = R.tensor(row.grid(dtype), dtype=getattr(R, dtype), requires_grad=True)
    out = row.ref(R, x)
    value = out.tolist()
    try:
        out.backward(R.ones_like(out))
    except RuntimeError:  # the reference has no derivative for this function
        return value, None
    return value, x.grad.tolist()


def _truth(row: _Row, dtype: str, x: float) -> float:
    """Lucid's float64 value at ``x`` as ``dtype`` holds it: past its range, ``inf``."""
    value = row.fn(lucid.tensor([x], dtype=lucid.float64)).item()
    if math.isfinite(value) and abs(value) > _dtype_max(dtype) * (1.0 + 1e-7):
        return math.copysign(_INF, value)
    return value


_POLE_WHY = (
    "the reference's trigamma pole is pi^2 / sin^2(pi x) with sin(pi x) rounding "
    "to ~1e-16: a large finite number"
)
_TAIL_WHY = "psi^(n)(x) -> 0 as x -> inf; the reference's Hurwitz tail is inf * 0"

#: (name, x) -> (the reference's value there or None for "not pinned", Lucid's,
#: why they differ).
_VALUE_DEVIATIONS: dict[tuple[str, float], tuple[float | None, float, str]] = {
    **{("polygamma[1]", p): (None, _INF, _POLE_WHY) for p in (-1e4, -30.0, -1.0)},
    ("polygamma[1]", -_INF): (_NAN, _INF, "-inf is the limit of the poles, trigamma's +inf"),
    ("polygamma[2]", _INF): (_NAN, 0.0, _TAIL_WHY),
    ("polygamma[3]", _INF): (_NAN, 0.0, _TAIL_WHY),
}

#: (name, dtype, x) -> a looser relative tolerance, where the reference is the
#: inexact side: J0(8) = 0.17165080713755390 and Lucid's float64 has it.
_VALUE_LOOSE: dict[tuple[str, str, float], float] = {
    (name, "float64", 8.0): 5e-8 for name in ("bessel_j0", "bessel_j1", "bessel_y0", "bessel_y1")
}


@pytest.mark.parity
@pytest.mark.parametrize(("name", "dtype", "dev"), _cases())
def test_values_match_the_reference(name: str, dtype: str, dev: str, ref: ModuleType) -> None:
    row = _BY_NAME[name]
    got, _ = _run(row, dtype, dev)
    want, _ = _run_ref(ref, row, dtype)
    for x, g, w in zip(row.grid(dtype), got, want, strict=True):
        if (deviation := _VALUE_DEVIATIONS.get((name, x))) is not None:
            ref_value, ours, why = deviation
            if ref_value is not None:
                assert _same(w, ref_value, dtype), f"reference changed at {x}: {w}"
            assert _same(g, ours, dtype), f"{name}({x}) {dtype} {dev}: {g} — {why}"
        elif row.recurrence and math.isnan(x):
            assert math.isnan(g), f"{name}(nan) = {g}"  # the reference answers 1.0 for some
        elif row.recurrence and (not math.isfinite(w) or abs(w) > 1e30):
            # The reference overflows through an inf - inf, or with the wrong
            # sign: the answer is the float64 value, cast.
            assert _same(g, _truth(row, dtype, x), dtype, rtol=1e-3), f"{name}({x}) {dtype}: {g}"
        elif row.recurrence:
            # A float32 recurrence accumulates ~n^2 eps against the reference's
            # trigonometric form for |x| < 1.
            assert math.isclose(g, w, rel_tol=1e-3, abs_tol=1e-4), (
                f"{name}({x}) {dtype}: {g} != {w}"
            )
        else:
            loose = _VALUE_LOOSE.get((name, dtype, x))
            assert _same(g, w, dtype, rtol=loose), f"{name}({x}) {dtype} {dev}: {g} != {w}"


_POLYGAMMA_NAMES = (
    "digamma",
    "psi",
    "polygamma[0]",
    "polygamma[1]",
    "polygamma[2]",
    "polygamma[3]",
)

#: (name, x, dtype or None) -> why the derivative differs from the reference's.
_GRAD_DEVIATIONS: dict[tuple[str, float, str | None], str] = {
    **{
        (name, x, None): (
            "d/dx psi^(n) at a pole is psi^(n+1)'s pole value; the reference's is a "
            "rounding artefact there (a large finite number, NaN at -inf)"
        )
        for name in _POLYGAMMA_NAMES
        for x in (-_INF, -1e4, -30.0, -1.0)
    },
    **{
        (name, _INF, None): "psi^(n+1)(inf) = 0; the reference's n + 1 >= 2 tail is NaN"
        for name in ("polygamma[1]", "polygamma[2]", "polygamma[3]")
    },
    **{
        ("i0", sign * _INF, None): "I1(±inf) = ±inf is the limit; the reference forms inf * 0"
        for sign in (-1.0, 1.0)
    },
    **{
        ("sinc", x, None): (
            "sin(u)/u's slope cancels to nothing next to 0: the reference's is 0 in "
            "float32 and 7e-6 relative off in float64; Lucid's is the series (closed form below)"
        )
        for x in (-1e-30, 1e-30, 1e-6)
    },
    ("log_ndtr", -1e4, "float32"): "the reference's float32 tail slope is 64929.77, not 10000.0001",
    ("log_ndtr", -1e4, "float64"): "the reference is 2.3e-8 relative off the exact 10000.0001",
}


@pytest.mark.parity
@pytest.mark.parametrize(("name", "dtype", "dev"), _cases())
def test_gradients_match_the_reference(name: str, dtype: str, dev: str, ref: ModuleType) -> None:
    row = _BY_NAME[name]
    if name in _NO_REFERENCE_GRADIENT or row.recurrence:
        pytest.skip("the reference has no derivative for this function")  # saves the run
    _, got = _run(row, dtype, dev)
    _, want = _run_ref(ref, row, dtype)
    if want is None:
        pytest.skip("the reference has no derivative for this function")
    assert got is not None
    for x, g, w in zip(row.grid(dtype), got, want, strict=True):
        if math.isnan(x):
            continue
        if (name, x, None) in _GRAD_DEVIATIONS or (name, x, dtype) in _GRAD_DEVIATIONS:
            continue
        assert _same(g, w, dtype), f"{name}'({x}) {dtype} {dev}: {g} != {w}"


# ── Gradients the reference does not have: the documented rule ──────────────


@pytest.mark.parametrize("name", ["bessel_y0", "bessel_y1"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_second_kind_slope_follows_the_value_to_the_edge(name: str, dtype: str) -> None:
    """The half-line functions have the slope of their divergence at 0 and NaN below."""
    row = _BY_NAME[name]
    _, grad = _run(row, dtype, "cpu")
    assert grad is not None
    by_x = dict(zip(row.grid(dtype), grad, strict=True))
    assert by_x[0.0] == _INF and by_x[-0.0] == _INF
    assert all(math.isnan(by_x[x]) for x in (-_INF, -1e4, -30.0, -2.5, -1.0, -0.5, -1e-30))
    # ... and none of it reaches the open half-line.
    assert all(math.isfinite(by_x[x]) for x in (1e-6, 0.3, 1.0, 2.5, 8.0, 30.0))


@pytest.mark.parametrize(
    "name", ["modified_bessel_k0", "modified_bessel_k1", "scaled_k0", "scaled_k1"]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_modified_second_kind_slope_follows_the_value_to_the_edge(name: str, dtype: str) -> None:
    row = _BY_NAME[name]
    _, grad = _run(row, dtype, "cpu")
    assert grad is not None
    by_x = dict(zip(row.grid(dtype), grad, strict=True))
    assert by_x[0.0] == -_INF and by_x[-0.0] == -_INF
    assert all(math.isnan(by_x[x]) for x in (-_INF, -1e4, -30.0, -2.5, -1.0, -0.5, -1e-30))
    assert all(math.isfinite(by_x[x]) for x in (1e-6, 0.3, 1.0, 2.5, 8.0, 30.0))


@pytest.mark.parametrize("n", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_polygamma_slope_is_the_next_order_everywhere(n: int, dtype: str) -> None:
    """d/dx psi^(n) = psi^(n+1), poles and infinities included."""
    row = _polygamma_row(n)
    _, grad = _run(row, dtype, "cpu")
    assert grad is not None
    nxt = S.polygamma(n + 1, lucid.tensor(row.grid(dtype), dtype=getattr(lucid, dtype))).tolist()
    for x, g, v in zip(row.grid(dtype), grad, nxt, strict=True):
        assert _same(g, v, dtype), f"d/dx psi^({n})({x}): {g} != psi^({n + 1})({x}) = {v}"


# ── Value cells with a single owner ─────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("dev", ["cpu", "metal"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_expm1_does_not_overflow_before_the_answer_does(
    dev: str, dtype: str, ref: ModuleType
) -> None:
    """``(u - 1) * x`` left the dtype's range while ``expm1(x)`` was finite (LCD-772)."""
    if dev == "metal" and (not metal_available() or dtype == "float64"):
        pytest.skip("Metal needs a device and has no float64")
    probes = {
        "float16": [10.0, 11.0, 11.1],
        "bfloat16": [80.0, 88.0, 88.7],
        "float32": [85.0, 88.0, 88.7],
        "float64": [705.0, 709.0, 709.78],
    }[dtype]
    xs = lucid.tensor(probes + [-p for p in probes], dtype=getattr(lucid, dtype), device=dev)
    xs.requires_grad_(True)
    out = lucid.expm1(xs)
    out.backward(lucid.ones_like(out))
    rx = ref.tensor(xs.detach().to("cpu").to(lucid.float64).tolist(), dtype=getattr(ref, dtype))
    rx.requires_grad_(True)
    rout = ref.expm1(rx)
    rout.backward(ref.ones_like(rout))
    rtol = {"float16": 2e-3, "bfloat16": 1.6e-2, "float32": 3e-7, "float64": 1e-15}[dtype]
    for g, w in zip(out.to("cpu").to(lucid.float64).tolist(), rout.double().tolist(), strict=True):
        assert g == w or math.isclose(g, w, rel_tol=rtol), (g, w)
    grads = zip(
        xs.grad.to("cpu").to(lucid.float64).tolist(), rx.grad.double().tolist(), strict=True
    )
    for g, w in grads:
        # The reference's slope is ``expm1(x) + 1``: 0 once that rounds to -1,
        # where exp(x) is a tiny positive number.
        assert g == w or math.isclose(g, w, rel_tol=4 * rtol, abs_tol=max(1e-5, rtol)), (g, w)


@pytest.mark.parametrize("bad", [1.5, 2.0, True, None, "1"])
def test_integer_orders_are_validated(bad: object) -> None:
    x = lucid.tensor([3.0, 2.0])
    with pytest.raises(TypeError):
        S.polygamma(bad, x)  # type: ignore[arg-type]  # the point of the test
    with pytest.raises(TypeError):
        S.multigammaln(x, bad)  # type: ignore[arg-type]  # the point of the test


@pytest.mark.parity
@pytest.mark.parametrize("bad", [1.5, True])
def test_integer_orders_are_refused_as_the_reference_does(bad: object, ref: ModuleType) -> None:
    rx = ref.tensor([3.0, 2.0])
    with pytest.raises(TypeError):
        ref.special.polygamma(bad, rx)
    with pytest.raises(TypeError):
        ref.special.multigammaln(rx, bad)


def test_integer_orders_accept_index_like_values() -> None:
    x = lucid.tensor([3.0, 2.0])
    assert S.polygamma(lucid.tensor(1), x).tolist() == S.polygamma(1, x).tolist()
    assert S.multigammaln(x, lucid.tensor(2)).tolist() == S.multigammaln(x, 2).tolist()


@pytest.mark.parity
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_expit_accepts_complex(dtype: str, ref: ModuleType) -> None:
    values = [1 + 2j, 0j, 1e3 + 0j, -1e3 + 0j, complex(_INF, 0), complex(-_INF, 0)]
    values += [complex(_NAN, 0), complex(0, 3.14159265), -2 + 0.5j, 30 - 30j, -30 + 30j]
    got = S.expit(lucid.tensor(values, dtype=getattr(lucid, dtype))).tolist()
    want = getattr(ref.special, "expit")(ref.tensor(values, dtype=getattr(ref, dtype))).tolist()
    for v, g, w in zip(values, got, want, strict=True):
        if abs(v.imag - 3.14159265) < 1e-12:
            # Next to the pole 1/(1 + e^{-z}) the reference loses the real part (1/2).
            assert math.isclose(g.imag, w.imag, rel_tol=1e-4) and g.real == pytest.approx(0.5)
        elif math.isnan(w.real):
            assert math.isnan(g.real) and math.isnan(g.imag), (v, g, w)
        else:
            assert abs(g - w) <= 2e-6 * max(abs(w), 1e-30) + 1e-30, (v, g, w)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sinc_slope_next_to_zero_is_the_series(dtype: str) -> None:
    """sinc'(x) = -pi^2 x / 3 + pi^4 x^3 / 30 - ...: no cancellation, no inf."""
    xs = [-1e-30, 1e-30, 1e-8, 1e-6, 1e-3]
    x = lucid.tensor(xs, dtype=getattr(lucid, dtype), requires_grad=True)
    lucid.sinc(x).sum().backward()
    assert x.grad is not None
    rtol = 1e-5 if dtype == "float32" else 1e-12
    for v, g in zip(xs, x.grad.tolist(), strict=True):
        want = -(math.pi**2) * v / 3.0 + math.pi**4 * v**3 / 30.0
        assert math.isclose(g, want, rel_tol=rtol, abs_tol=1e-37), (v, g, want)
