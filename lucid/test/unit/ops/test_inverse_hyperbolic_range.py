"""LCD-1028: ``asinh`` / ``acosh`` / ``atanh`` keep their value and their derivatives across the range.

The composites were the textbook closed forms ``log(x + sqrt(x^2 + 1))``, ``log(x + sqrt(x^2 - 1))``
and ``log((1 + x) / (1 - x)) / 2``, each in the middle of the line and wrong at both ends, in float32
and in float64: ``asinh(-1e5)`` was ``-inf`` (the sum cancels), ``asinh(1e-8)`` was 0 (it rounds to
1), ``asinh(1e20)`` and ``acosh(3e19)`` were ``inf`` (``x^2`` overflows), ``atanh(1e-8)`` was 0, and
the derivatives, taken through the same formulas, were NaN or off by the same ranges (``asinh``'s d1
NaN from -1e5, ``atanh``'s d2 wrong below 1e-6).  Each is now one ``log1p`` of an argument that
neither rounds toward 1 nor overflows, with the closed-form slope as its backward.

The closed forms here are plain Python float64 on the float32 values read back from the leaf: the
answer, its three derivatives (a graph built with ``create_graph``, so the slope's own graph is what
is differentiated), over 1e-30 .. 1e30 (1e-300 .. 1e300 in float64) and across the point
(``2^20`` / ``2^100``) where the value clamps its argument.  An answer below float32's smallest
normal is not asked for: Metal flushes it.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import pytest

import lucid
from lucid.autograd import grad
from lucid.test._fixtures.devices import metal_available

NAN = float("nan")
INF = float("inf")

#: ``(value, d1, d2, d3)`` relative tolerances: a few ulp, a little more for each order.
TOLERANCE = {
    lucid.float32: (5e-7, 5e-7, 1e-6, 1e-5),
    lucid.float64: (1.5e-15, 1.5e-15, 3e-15, 3e-14),
}
FLOOR = {lucid.float32: 1e-36, lucid.float64: 1e-290}
CEILING = {lucid.float32: 3e38, lucid.float64: 1e300}
SPLIT = {lucid.float32: 2.0**20, lucid.float64: 2.0**100}


# ── closed forms (plain float64; products rather than ``**`` so an overflow is ``inf``) ──────────


def _asinh_d1(x: float) -> float:
    return 1.0 / math.hypot(1.0, x)


def _asinh_d2(x: float) -> float:
    h = math.hypot(1.0, x)
    return -x / (h * h * h)


def _asinh_d3(x: float) -> float:
    h = math.hypot(1.0, x)
    return (2.0 * x * x - 1.0) / (h * h * h * h * h)


def _asinh_scale3(x: float) -> float:
    """The magnitude the two terms of ``d3`` are of: it changes sign at ``x = 1 / sqrt(2)``."""
    h = math.hypot(1.0, x)
    return (2.0 * x * x + 1.0) / (h * h * h * h * h)


def _acosh_p(x: float) -> float:
    return (x - 1.0) * (x + 1.0)


def _acosh_d1(x: float) -> float:
    return 1.0 / math.sqrt(_acosh_p(x))


def _acosh_d2(x: float) -> float:
    p = _acosh_p(x)
    return -x / (p * math.sqrt(p))


def _acosh_d3(x: float) -> float:
    p = _acosh_p(x)
    return (2.0 * x * x + 1.0) / (p * p * math.sqrt(p))


def _atanh_p(x: float) -> float:
    return (1.0 - x) * (1.0 + x)


def _atanh_d1(x: float) -> float:
    return 1.0 / _atanh_p(x)


def _atanh_d2(x: float) -> float:
    p = _atanh_p(x)
    return 2.0 * x / (p * p)


def _atanh_d3(x: float) -> float:
    p = _atanh_p(x)
    return 2.0 * (1.0 + 3.0 * x * x) / (p * p * p)


@dataclass(frozen=True)
class Curve:
    name: str
    fn: Callable[[lucid.Tensor], lucid.Tensor]
    value: Callable[[float], float]
    d1: Callable[[float], float]
    d2: Callable[[float], float]
    d3: Callable[[float], float]
    scale3: Callable[[float], float] | None = None

    def derivatives(self) -> tuple[Callable[[float], float], ...]:
        return (self.value, self.d1, self.d2, self.d3)


ASINH = Curve("asinh", lucid.asinh, math.asinh, _asinh_d1, _asinh_d2, _asinh_d3, _asinh_scale3)
ACOSH = Curve("acosh", lucid.acosh, math.acosh, _acosh_d1, _acosh_d2, _acosh_d3)
ATANH = Curve("atanh", lucid.atanh, math.atanh, _atanh_d1, _atanh_d2, _atanh_d3)
CURVES = [ASINH, ACOSH, ATANH]


# ── the points ────────────────────────────────────────────────────────────────────────────────────


def _decades(low: int, high: int) -> list[float]:
    return [m * 10.0**e for e in range(low, high + 1) for m in (1.0, 2.5, 7.1)]


def _both(values: list[float]) -> list[float]:
    return values + [-v for v in values]


def _around(split: float) -> list[float]:
    return [split * f for f in (0.5, 0.999, 1.0, 1.001, 2.0)]


def _points(name: str, dtype: lucid.dtype) -> list[float]:
    double = dtype == lucid.float64
    reach = 300 if double else 30
    bits = 53 if double else 24
    split = SPLIT[dtype]
    if name == "asinh":
        return _both(_decades(-reach, reach) + _around(split))
    if name == "acosh":
        return (
            [1.0 + 2.0**-k for k in range(1, bits)]
            + [1.5, 2.0, 3.0]
            + _decades(0, reach)
            + _around(split)
        )
    return _both(
        _decades(-reach, -1) + [1.0 - 2.0**-k for k in range(1, bits + 1)] + [0.5, 0.9, 0.99]
    )


# ── the check ─────────────────────────────────────────────────────────────────────────────────────


def _evaluate(
    fn: Callable[[lucid.Tensor], lucid.Tensor], xs: list[float], dtype: lucid.dtype, device: str
) -> tuple[list[float], list[list[float]]]:
    """The leaf as read back, and the answer and its first three derivatives."""
    x = lucid.tensor(xs, dtype=dtype, device=device, requires_grad=True)
    out = fn(x)
    rows = [out.tolist()]
    for _ in range(3):
        (out,) = grad(out.sum(), [x], create_graph=True)
        rows.append(out.tolist())
    return x.tolist(), rows


def _misses(curve: Curve, dtype: lucid.dtype, device: str) -> list[str]:
    xs, rows = _evaluate(curve.fn, _points(curve.name, dtype), dtype, device)
    floor, ceiling = FLOOR[dtype], CEILING[dtype]
    misses: list[str] = []
    for order, (got_row, closed, tol) in enumerate(
        zip(rows, curve.derivatives(), TOLERANCE[dtype], strict=True)
    ):
        for x, got in zip(xs, got_row, strict=True):
            try:
                want = closed(x)
                scale = curve.scale3(x) if order == 3 and curve.scale3 else abs(want)
            except (ZeroDivisionError, ValueError, OverflowError):
                continue  # a closed form at its own pole
            if want - want != 0.0 or not floor <= scale <= ceiling:
                continue
            if got - got != 0.0 or abs(got - want) > tol * scale:
                misses.append(f"{curve.name} d{order}({x!r}) = {got!r}, want {want!r}")
    return misses


@pytest.mark.parametrize("curve", CURVES, ids=lambda c: c.name)
def test_float32_matches_the_closed_forms_to_a_few_ulp(curve: Curve, device: str) -> None:
    misses = _misses(curve, lucid.float32, device)
    assert not misses, f"{len(misses)} off:\n" + "\n".join(misses[:6])


@pytest.mark.parametrize("curve", CURVES, ids=lambda c: c.name)
def test_float64_matches_the_closed_forms_to_a_few_ulp(curve: Curve) -> None:
    misses = _misses(curve, lucid.float64, "cpu")
    assert not misses, f"{len(misses)} off:\n" + "\n".join(misses[:6])


# ── the reported cases ────────────────────────────────────────────────────────────────────────────


def test_the_reported_values(device: str) -> None:
    """The card: float32, both streams -- each of these was ``inf`` or 0."""
    cases = {
        "asinh": [-1e5, 1e-8, 1e20, -3e38],
        "acosh": [3e19, 1e30, 3e38],
        "atanh": [1e-8, -1e-8, 1e-30],
    }
    for name, xs in cases.items():
        got = getattr(lucid, name)(lucid.tensor(xs, device=device)).tolist()
        for x, g in zip(xs, got, strict=True):
            want = getattr(math, name)(lucid.tensor(x).item())
            assert abs(g - want) <= 5e-7 * abs(want), f"{name}({x}) = {g}, want {want}"


def test_the_reported_derivatives(device: str) -> None:
    """``asinh``'s d1 was NaN from -1e5 and wrong at -3000 (4.88e-4 for 3.33e-4); ``atanh``'s d2
    was wrong below 1e-6."""
    x = lucid.tensor([-1e5, -3000.0, -1e19, 1e20], device=device, requires_grad=True)
    (d1,) = grad(lucid.asinh(x).sum(), [x])
    for xi, g in zip(x.tolist(), d1.tolist(), strict=True):
        assert abs(g - _asinh_d1(xi)) <= 5e-7 * _asinh_d1(xi), (xi, g)
    t = lucid.tensor([1e-6, 1e-8, -1e-7], device=device, requires_grad=True)
    (g1,) = grad(lucid.atanh(t).sum(), [t], create_graph=True)
    (g2,) = grad(g1.sum(), [t])
    for xi, g in zip(t.tolist(), g2.tolist(), strict=True):
        assert abs(g - 2.0 * xi) <= 1e-6 * abs(2.0 * xi), (xi, g)


def test_acosh_is_accurate_next_to_one(device: str) -> None:
    """``x - 1`` is exact, so the answer is ``sqrt(2 (x - 1))`` to the last digit."""
    x = lucid.tensor([1.0 + 2.0**-k for k in (23, 20, 15, 10, 5)], device=device)
    for xi, g in zip(x.tolist(), lucid.acosh(x).tolist(), strict=True):
        assert abs(g - math.acosh(xi)) <= 5e-7 * math.acosh(xi)


# ── the special values ────────────────────────────────────────────────────────────────────────────


def _is_negative_zero(v: float) -> bool:
    return v == 0.0 and math.copysign(1.0, v) < 0.0


SPECIAL = {
    "asinh": [(NAN, NAN), (INF, INF), (-INF, -INF), (0.0, 0.0), (-0.0, -0.0)],
    "acosh": [
        (NAN, NAN),
        (INF, INF),
        (-INF, NAN),
        (1.0, 0.0),
        (0.5, NAN),
        (0.0, NAN),
        (-1.0, NAN),
        (-1e30, NAN),
        (-9999.0, NAN),
    ],
    "atanh": [
        (NAN, NAN),
        (INF, NAN),
        (-INF, NAN),
        (1.0, INF),
        (-1.0, -INF),
        (1.5, NAN),
        (-2.0, NAN),
        (1e30, NAN),
        (0.0, 0.0),
        (-0.0, -0.0),
    ],
}


@pytest.mark.parametrize("name", ["asinh", "acosh", "atanh"])
def test_the_special_values(name: str, device: str) -> None:
    xs = [x for x, _ in SPECIAL[name]]
    got = getattr(lucid, name)(lucid.tensor(xs, device=device)).tolist()
    for (x, want), g in zip(SPECIAL[name], got, strict=True):
        if want != want:
            assert g != g, f"{name}({x}) = {g}, want nan"
        else:
            assert g == want, f"{name}({x}) = {g}, want {want}"
            assert math.copysign(1.0, g) == math.copysign(1.0, want), f"{name}({x}): sign of zero"


@pytest.mark.parametrize("name", ["asinh", "atanh"])
def test_a_negative_zero_stays_one_and_has_unit_slope(name: str, device: str) -> None:
    """Both are odd with slope 1 at the origin, third derivative -1 (``asinh``) and 2 (``atanh``)."""
    x = lucid.tensor([0.0, -0.0], device=device, requires_grad=True)
    out = getattr(lucid, name)(x)
    assert _is_negative_zero(out.tolist()[1])
    orders = []
    for _ in range(3):
        (out,) = grad(out.sum(), [x], create_graph=True)
        orders.append(out.tolist())
    third = -1.0 if name == "asinh" else 2.0
    assert orders == [[1.0, 1.0], [0.0, 0.0], [third, third]]


def test_the_slopes_at_the_poles_and_at_infinity(device: str) -> None:
    def slope(fn: Callable[[lucid.Tensor], lucid.Tensor], xs: list[float]) -> list[float]:
        x = lucid.tensor(xs, device=device, requires_grad=True)
        (g,) = grad(fn(x).sum(), [x])
        return g.tolist()

    assert slope(lucid.asinh, [INF, -INF]) == [0.0, 0.0]
    assert slope(lucid.acosh, [INF]) == [0.0]
    assert slope(lucid.acosh, [1.0]) == [INF]
    assert slope(lucid.atanh, [1.0, -1.0]) == [INF, INF]
    assert slope(lucid.atanh, [INF, -INF]) == [0.0, 0.0]
    for fn in (lucid.asinh, lucid.acosh, lucid.atanh):
        assert all(g != g for g in slope(fn, [NAN]))
    assert all(g != g for g in slope(lucid.acosh, [0.5, -3.0]))


# ── every entry point is the same function ───────────────────────────────────────────────────────

ENTRY_POINTS = [
    ("asinh", "arcsinh", [-1e5, 1e-8, 1e20, -0.0]),
    ("acosh", "arccosh", [1.0 + 2.0**-20, 3e19, 1e30]),
    ("atanh", "arctanh", [1e-8, -1e-30, 0.999]),
]


@pytest.mark.parametrize("short,long,xs", ENTRY_POINTS, ids=[e[0] for e in ENTRY_POINTS])
def test_the_aliases_methods_and_in_place_twins_agree(
    short: str, long: str, xs: list[float], device: str
) -> None:
    x = lucid.tensor(xs, device=device)
    want = getattr(lucid, short)(x).tolist()
    assert all(math.isfinite(v) for v in want)
    assert getattr(lucid, long)(x).tolist() == want
    for entry in (getattr(x, short)(), getattr(x, long)()):
        assert entry.tolist() == want
    for twin in (short + "_", long + "_"):
        assert getattr(x.clone(), twin)().tolist() == want


# ── half precision, complex, compile ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["float16", "bfloat16"])
def test_half_precision_answers_in_float32_and_rounds(dtype: lucid.dtype, device: str) -> None:
    """``x * x`` left float16 at 256 (``asinh(-60000)`` was ``-inf``)."""
    cases = {
        "asinh": [-60000.0, 300.0, -3000.0, 1e-4],
        "acosh": [1.0 + 2.0**-7, 60000.0, 300.0],
        "atanh": [1e-4, 0.5, -0.5, 0.75],
    }
    for name, xs in cases.items():
        x = lucid.tensor(xs, dtype=dtype, device=device)
        out = getattr(lucid, name)(x)
        assert out.dtype == dtype
        for xi, g in zip(x.tolist(), out.tolist(), strict=True):
            want = getattr(math, name)(xi)
            assert abs(g - want) <= 8e-3 * abs(want), f"{name}({xi}) = {g}, want {want}"


@pytest.mark.skipif(not metal_available(), reason="complex64 inverse hyperbolics are Metal's")
def test_a_complex_input_keeps_the_textbook_forms() -> None:
    """The stable forms are real-valued; a complex input is answered as it always was (its branch
    is the textbook formula's, not ``cmath``'s: ``acosh(-0.5 + 0.3j)`` is on the other sheet)."""
    t = lucid.tensor([1 + 2j, -0.5 + 0.3j, 0.7 - 1.1j], dtype=lucid.complex64, device="metal")
    textbook = {
        "asinh": lucid.log(t + lucid.sqrt(t * t + 1.0)),
        "acosh": lucid.log(t + lucid.sqrt(t * t - 1.0)),
        "atanh": lucid.log((1.0 + t) / (1.0 - t)) * 0.5,
    }
    for name, want in textbook.items():
        assert getattr(lucid, name)(t).tolist() == want.tolist(), name


@pytest.mark.skipif(not metal_available(), reason="lucid.compile is Metal's")
@pytest.mark.parametrize(
    "name,xs",
    [
        ("asinh", [-3.0, -0.5, 0.0, 1e-8, 0.7, 2.0, 40.0, 1e6]),
        ("acosh", [1.5, 2.0, 3.0, 10.0, 1e3, 1e6, 1.0001, 7.0]),
        ("atanh", [-0.9, -0.5, 0.0, 1e-8, 0.3, 0.7, 0.99, 0.5]),
    ],
)
def test_a_compiled_step_takes_the_same_gradient(name: str, xs: list[float]) -> None:
    """The recording takes the plain ops (``Function._trace_form``) and differentiates *them*: a
    small value (1e-8) lost its slope through a ``copysign``, and the origin has none through an
    ``abs``."""
    import lucid.nn as nn

    fn = getattr(lucid, name)

    class Probe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(lucid.ones(len(xs), device="metal"))
            self.mix = lucid.linspace(0.5, 1.5, len(xs), device="metal")

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return (fn(x * self.w) * self.mix).sum()

    first = lucid.tensor(xs, device="metal")
    second = lucid.tensor(xs[::-1], device="metal")
    model = Probe()
    model(second).backward()
    want = model.w.grad.tolist()
    step = lucid.compile.make_step(model, lambda out: out)
    for batch in (first, second):
        model.w.grad = None
        step(batch).backward()
    got = model.w.grad.tolist()
    for g, w in zip(got, want, strict=True):
        assert abs(g - w) <= 1e-5 * abs(w) + 1e-30, (got, want)


# ── the reference ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("name", ["asinh", "acosh", "atanh"])
def test_the_value_agrees_with_the_reference(name: str, device: str, ref: ModuleType) -> None:
    """The reference's value is right over the whole range, which is what this is held to."""
    xs = [
        v
        for v in _points(name, lucid.float32)
        if (name != "atanh" or abs(v) < 1.0) and (name != "acosh" or v > 1.0 + 2.0**-22)
    ]
    mine = getattr(lucid, name)(lucid.tensor(xs, device=device)).tolist()
    theirs = getattr(ref, name)(ref.tensor(xs, dtype=ref.float32)).tolist()
    for x, a, b in zip(xs, mine, theirs, strict=True):
        assert abs(a - b) <= 1e-6 * abs(b), f"{name}({x}): {a} vs {b}"


@pytest.mark.parity
@pytest.mark.parametrize("name", ["asinh", "acosh", "atanh"])
def test_the_derivatives_agree_with_the_reference_where_it_is_right(
    name: str, device: str, ref: ModuleType
) -> None:
    """Its first two derivatives hold to 1e-5 up to |x| = 1e10 (``asinh`` / ``acosh``) and, for
    ``atanh``, away from the poles; past 1e19 its ``x * x`` overflows and it answers 0."""
    pool = {
        "asinh": _both(_decades(-30, 10)),
        "acosh": [1.5, 2.0, 3.0] + _decades(0, 10)[1:],
        "atanh": _both(_decades(-30, -1) + [0.5, 0.9]),
    }
    xs = pool[name]
    x = lucid.tensor(xs, device=device, requires_grad=True)
    (m1,) = grad(getattr(lucid, name)(x).sum(), [x], create_graph=True)
    (m2,) = grad(m1.sum(), [x])
    r = ref.tensor(xs, dtype=ref.float32, requires_grad=True)
    (r1,) = ref.autograd.grad(getattr(ref, name)(r).sum(), r, create_graph=True)
    (r2,) = ref.autograd.grad(r1.sum(), r)
    floor = FLOOR[lucid.float32]
    for order, mine, theirs in ((1, m1.tolist(), r1.tolist()), (2, m2.tolist(), r2.tolist())):
        for xi, a, b in zip(xs, mine, theirs, strict=True):
            if abs(b) < floor:
                continue
            assert abs(a - b) <= 1e-5 * abs(b), f"{name} d{order}({xi}): {a} vs {b}"
