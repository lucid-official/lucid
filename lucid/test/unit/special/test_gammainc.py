"""``lucid.special.gammainc`` / ``gammaincc`` — the regularized incomplete gamma.

``Gamma.cdf``, ``Chi2.cdf`` and ``InverseGamma.cdf`` are this function, so
it is held to the reference on a grid that crosses every branch the scalar
algorithm picks between: the power series (x below a), the continued
fraction (x above a), the small-x series of Q, and Temme's expansion
(a large, x near a) — with a from 1e-30 to 1e3 and x from 0 to inf, in
float32 and float64 on CPU and float32 on Metal, and in float16 and
bfloat16 (evaluated in float32, as the reference's opmath does) on both.
The edges of the domain
(a or x zero, infinite, negative or NaN) are spelled out, so they need no
reference.

``_DEVIATIONS`` lists the cells where the reference is the one that is off,
with the exact value Lucid matches; the reference's own value is required
to still differ, so the table goes stale loudly.
"""

import math
from typing import Any

import numpy as np
import pytest

import lucid
from lucid.special import gammainc, gammaincc
from lucid.test._fixtures.devices import metal_available

_INF = math.inf
_NAN = math.nan

_A = [1e-30, 1e-3, 0.5, 1.0, 10.0, 1e3]
_X = [0.0, 1e-6, 0.3, 0.5, 1.0, 1.05, 9.0, 10.0, 11.0, 30.0, 990.0, 1000.0]
_X += [1010.0, 1e4, _INF]
_GRID = [(a, x) for a in _A for x in _X]

#: (a, x) -> (the exact Q(a, x), the dtypes in which the reference is off).
#: For a tiny shape the reference evaluates lgamma(1 + a) after rounding
#: 1 + a to 1, and loses the Euler-Mascheroni term of Q ~ a (E1(x) + ...).
_DEVIATIONS: dict[tuple[float, float], tuple[float, tuple[str, ...]]] = {
    (1e-30, 1e-6): (1.3238295893062492e-29, ("float32", "float64")),
    (1e-30, 0.3): (9.0567665167584682e-31, ("float32", "float64")),
    (1e-30, 0.5): (5.5977359477616086e-31, ("float32", "float64")),
    (1e-30, 1.0): (2.1938393439552029e-31, ("float32", "float64")),
    (1e-30, 1.05): (2.018728132201966e-31, ("float32", "float64")),
    (1e-3, 0.3): (0.00090583808830664699, ("float32",)),
    (1e-3, 0.5): (0.00056006665647074989, ("float32",)),
    (1e-3, 1.0): (0.0002196083575855564, ("float32",)),
    (1e-3, 1.05): (0.00020208671632562252, ("float32",)),
}

#: (a, x) -> (P, Q) on the edges of the domain, as the reference has them.
_EDGES: dict[tuple[float, float], tuple[float, float]] = {
    (-1.0, 1.0): (_NAN, _NAN),
    (1.0, -1.0): (_NAN, _NAN),
    (0.0, 0.0): (_NAN, _NAN),
    (0.0, -1.0): (_NAN, _NAN),
    (0.0, 1.0): (1.0, 0.0),
    (2.0, 0.0): (0.0, 1.0),
    (_INF, 1.0): (0.0, 1.0),
    (_INF, _INF): (_NAN, _NAN),
    (2.0, _INF): (1.0, 0.0),
    (_NAN, 1.0): (_NAN, _NAN),
    (1.0, _NAN): (_NAN, _NAN),
}


_HALF = ("float16", "bfloat16")


def _placements(half: bool = True) -> list[Any]:
    """float32 and float64 on CPU, float32 on Metal — and the two 16-bit
    floats on both, which are evaluated in float32 and rounded back."""
    metal = [pytest.mark.skipif(not metal_available(), reason="needs Metal")]
    places = [
        pytest.param("float32", "cpu", id="cpu-f32"),
        pytest.param("float64", "cpu", id="cpu-f64"),
        pytest.param("float32", "metal", marks=metal, id="metal-f32"),
    ]
    if half:
        for dtype, tag in (("float16", "f16"), ("bfloat16", "bf16")):
            places.append(pytest.param(dtype, "cpu", id=f"cpu-{tag}"))
            places.append(pytest.param(dtype, "metal", marks=metal, id=f"metal-{tag}"))
    return places


def _lucid(values: list[float], dtype: str, device: str) -> lucid.Tensor:
    wide = lucid.tensor(np.asarray(values, dtype=np.float64))
    return wide.to(getattr(lucid, dtype)).to(device)


def _ref(R: Any, values: list[float], dtype: str) -> Any:
    return R.tensor(np.asarray(values, dtype=np.float64)).to(getattr(R, dtype))


def _floats(t: lucid.Tensor) -> list[float]:
    """The values as Python floats; a 16-bit tensor has no numpy dtype."""
    return [float(v) for v in t.to("cpu").to(lucid.float64).numpy().reshape(-1)]


def _tols(dtype: str) -> tuple[float, float]:
    """(rtol, atol).  float64 is 1e-8 rather than its rounding: in Temme's
    expansion the reference takes pi in single precision, 1e-9 off the
    exact value Lucid agrees with.  A 16-bit answer may round to the next
    representable value, and float16's subnormals hold few digits."""
    return {
        "float64": (1e-8, 0.0),
        "float32": (2e-5, 0.0),
        "float16": (2e-3, 1e-6),
        "bfloat16": (1.6e-2, 1e-30),
    }[dtype]


def _same(got: float, want: float, dtype: str) -> bool:
    if math.isnan(want):
        return math.isnan(got)
    rtol, atol = _tols(dtype)
    return math.isclose(got, want, rel_tol=rtol, abs_tol=atol)


@pytest.mark.parity
@pytest.mark.parametrize("lower", [True, False], ids=["gammainc", "gammaincc"])
@pytest.mark.parametrize("dtype,device", _placements())
def test_matches_the_reference_on_the_grid(dtype: str, device: str, lower: bool, ref: Any) -> None:
    a = [c[0] for c in _GRID]
    x = [c[1] for c in _GRID]
    ours_fn, ref_fn = (
        (gammainc, ref.special.gammainc) if lower else (gammaincc, ref.special.gammaincc)
    )
    ours = _floats(ours_fn(_lucid(a, dtype, device), _lucid(x, dtype, device)))
    theirs = ref_fn(_ref(ref, a, dtype), _ref(ref, x, dtype)).double().tolist()
    wrong = []
    for (ai, xi), o, t in zip(_GRID, ours, theirs, strict=True):
        deviation = None if lower else _DEVIATIONS.get((ai, xi))
        if deviation is not None and dtype in deviation[1]:
            exact = deviation[0]
            if not _same(o, exact, dtype) or _same(t, exact, dtype):
                wrong.append((ai, xi, o, t, "deviation"))
        elif not _same(o, t, dtype):
            wrong.append((ai, xi, o, t))
    assert not wrong, wrong


@pytest.mark.parametrize("dtype,device", _placements())
def test_edges_of_the_domain(dtype: str, device: str) -> None:
    a = [c[0] for c in _EDGES]
    x = [c[1] for c in _EDGES]
    p = _floats(gammainc(_lucid(a, dtype, device), _lucid(x, dtype, device)))
    q = _floats(gammaincc(_lucid(a, dtype, device), _lucid(x, dtype, device)))
    for (cell, (want_p, want_q)), got_p, got_q in zip(_EDGES.items(), p, q, strict=True):
        assert _same(got_p, want_p, dtype), (cell, got_p, want_p)
        assert _same(got_q, want_q, dtype), (cell, got_q, want_q)


@pytest.mark.parametrize("dtype,device", _placements())
def test_closed_forms_and_complement(dtype: str, device: str) -> None:
    x = _lucid([1e-3, 0.3, 1.0, 4.0, 25.0], dtype, device)
    xs = _floats(x)  # as rounded to ``dtype``
    one = _lucid([1.0] * len(xs), dtype, device)
    half = _lucid([0.5] * len(xs), dtype, device)
    rtol, atol = (1e-12, 0.0) if dtype == "float64" else (2e-6, 0.0)
    if dtype in _HALF:
        rtol, atol = _tols(dtype)
    # P(1, x) = 1 - e^-x and Q(1/2, x) = erfc(sqrt(x)).
    for got, want in zip(_floats(gammainc(one, x)), [-math.expm1(-v) for v in xs], strict=True):
        assert math.isclose(got, want, rel_tol=rtol, abs_tol=atol)
    for got, want in zip(
        _floats(gammaincc(half, x)), [math.erfc(math.sqrt(v)) for v in xs], strict=True
    ):
        assert math.isclose(got, want, rel_tol=10 * rtol, abs_tol=atol)
    a = _lucid([0.5, 3.0, 30.0, 300.0, 1e3], dtype, device)
    total = _floats(gammainc(a, x * 40.0) + gammaincc(a, x * 40.0))
    assert np.allclose(total, 1.0, rtol=0.0, atol=10 * max(rtol, atol))


@pytest.mark.parametrize("lower", [True, False], ids=["gammainc", "gammaincc"])
@pytest.mark.parametrize("dtype,device", _placements())
def test_gradient_in_x_is_the_density(dtype: str, device: str, lower: bool) -> None:
    a_vals, x_vals = [0.5, 2.0, 30.0, 300.0], [0.7, 1.5, 28.0, 310.0]
    a = _lucid(a_vals, dtype, device)
    x = _lucid(x_vals, dtype, device)
    x.requires_grad_(True)
    (gammainc(a, x) if lower else gammaincc(a, x)).sum().backward()
    assert x.grad is not None
    assert x.grad.dtype == x.dtype
    want = [
        (1.0 if lower else -1.0) * math.exp((ai - 1.0) * math.log(xi) - xi - math.lgamma(ai))
        for ai, xi in zip(_floats(a), _floats(x), strict=True)
    ]
    tol = {"float64": 1e-10, "float32": 1e-4}.get(dtype, _tols(dtype)[0])
    assert np.allclose(_floats(x.grad), want, rtol=tol, atol=0.0)


@pytest.mark.parametrize("dtype,device", _placements(half=False))
def test_lanczos_band_keeps_tiny_values(dtype: str, device: str) -> None:
    # |x - a| <= 0.4 a below 200 takes x^a e^-x / Gamma(a) as one exponent:
    # as e^(a - x) times (x / fac)^a the product passed through a subnormal
    # Metal flushes, and P(174.49, 105.92) came out 0.  Exact values: the
    # reference in float64 at these float32 points.
    a = [174.49, 150.0, 199.0, 120.0, 60.0]
    x = [105.92, 91.0, 121.0, 73.0, 36.5]
    want = [6.72544889e-10, 9.40197250e-09, 5.33831938e-11, 2.90967830e-07]
    want += [2.23076156e-04]
    if dtype == "float32":
        got = _floats(gammainc(_lucid(a, dtype, device), _lucid(x, dtype, device)))
    else:  # the float32 points, so the exact values above still apply
        a32 = lucid.tensor(np.asarray(a, dtype=np.float32)).to(lucid.float64)
        x32 = lucid.tensor(np.asarray(x, dtype=np.float32)).to(lucid.float64)
        got = _floats(gammainc(a32.to(device), x32.to(device)))
    assert np.allclose(got, want, rtol=2e-5, atol=0.0), got


@pytest.mark.parametrize("fn", [gammainc, gammaincc], ids=["gammainc", "gammaincc"])
def test_no_gradient_in_the_shape(fn: Any) -> None:
    # The reference has no derivative in a either, and says so on backward.
    a = lucid.tensor(2.0, requires_grad=True)
    x = lucid.tensor(1.5, requires_grad=True)
    out = fn(a, x)
    with pytest.raises(NotImplementedError, match="with respect to `input`"):
        out.backward()


def test_broadcasts_and_promotes() -> None:
    a = lucid.tensor(np.asarray([[1.0], [2.0]], dtype=np.float32))
    x = lucid.tensor(np.asarray([0.5, 1.0, 2.0], dtype=np.float64))
    out = gammainc(a, x)
    assert out.shape == (2, 3)
    assert out.dtype == lucid.float64
    # An integer argument is promoted to a float, as the other special
    # functions promote it.
    ints = gammainc(lucid.tensor([1, 2]), lucid.tensor([1, 1]))
    assert ints.dtype == lucid.float32
    assert np.allclose(ints.numpy(), [1.0 - math.exp(-1.0), 1.0 - 2.0 * math.exp(-1.0)])
