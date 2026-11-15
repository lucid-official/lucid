"""LCD-1018: the recorded second derivative of ``vector_norm`` on a slice that spans decades.

Below ``p = 2`` the power ``|u|^(p-1)`` of the ratio ``u = |x| / N`` has the slope
``(p-1) |u|^(p-2)``, huge for an entry far below the norm.  The second derivative walks it, and
the cotangent ``w`` met ``|u|^(p-2)`` *before* the division by ``N`` that brings the product
back: ``x = [1e20, 1e-10, 2e-10]``, ``p = 1.1``, ``w = 1e20`` formed ``1e20 * 1e27`` (past
float32's 3.4e38) on the way to the answer 1e27 (``[-inf, nan, nan]`` on both streams).  The
same class cut the quasi-norms ``0 < p < 1`` (``p = 0.9``: a slice of 1e5 spanning 35 decades).
The recorded graph now takes the power in the slice's own unit (its smallest non-zero entry), so
the cotangent never grows on its way through the power.

The closed forms are plain Python float64 from the float32 values read back from the leaf; a
case whose closed form does not fit float32 is not asked for an answer.
"""

import math

import pytest

import lucid
import lucid.linalg
from lucid.autograd import grad

F32_MAX = 3.0e38
#: A slice is ``magnitude * 10^(-span * T[i])`` with these signs: its smallest entry is ``span``
#: decades below its largest.
FRACTIONS = [0.0, 0.5, 1.0, 0.25, 0.75]
SIGNS = [1.0, -1.0, 1.0, -1.0, 1.0]
DIRECTION = [0.3, -1.0, 0.5, 2.0, -0.7]
SECOND = [-0.4, 0.9, 1.3, -0.2, 0.6]
BASE = [3.0, -4.0, 1.0, 2.0, 0.5]

BETWEEN_ONE_AND_TWO = [1.05, 1.1, 1.25, 1.5, 1.75, 1.9]
QUASI = [0.25, 0.5, 0.75, 0.9]
ABOVE_TWO = [2.0, 2.5, 3.0, 4.0]
MAGNITUDES = [10.0**e for e in range(-30, 31, 5)]
SPANS = [0, 10, 20, 30, 35]


def _slice(magnitude: float, span: int) -> list[float]:
    return [s * magnitude * 10.0 ** (-span * f) for s, f in zip(SIGNS, FRACTIONS, strict=True)]


def _hvp(x: list[float], w: list[float], p: float) -> tuple[list[float], list[float]]:
    """``H w = (p-1)/N (|r|^(p-2) w - s <s, w>)`` for ``r = |x| / N`` and ``s`` the gradient,
    and per entry the sum of the two terms' magnitudes (what its error is measured against)."""
    top = max(abs(v) for v in x)
    n = top * sum((abs(v) / top) ** p for v in x) ** (1.0 / p)
    ratio = [abs(v) / n for v in x]
    slope = [math.copysign(r ** (p - 1.0), v) for r, v in zip(ratio, x, strict=True)]
    dot = sum(s * v for s, v in zip(slope, w, strict=True))
    first = [(p - 1.0) / n * r ** (p - 2.0) * v for r, v in zip(ratio, w, strict=True)]
    second = [(p - 1.0) / n * s * dot for s in slope]
    return (
        [a - b for a, b in zip(first, second, strict=True)],
        [abs(a) + abs(b) for a, b in zip(first, second, strict=True)],
    )


def _asked_for(x: list[float], scale: list[float]) -> bool:
    """Whether the closed form fits float32 *and* so does the chain's largest intermediate.

    The recorded gradient is each entry's power law about its own value, so the cotangent
    reaches ``(p-1) / |x|`` last and what it holds before that is the answer times ``|x|``
    (``w * G``: at most ``w`` for ``p > 1``).  Only an entry above 1 of a quasi-norm whose answer
    is within a factor ``|x|`` of float32's range has a partial product that overflows while the
    answer fits: the formulation's limit, not asked for here.
    """
    biggest = max(scale)
    partial = max(s * abs(v) for s, v in zip(scale, x, strict=True))
    return 1e-30 <= biggest <= F32_MAX and partial <= F32_MAX


def _second_derivative(device: str, p: float, x: list[float], w: list[float]) -> lucid.Tensor:
    leaf = lucid.tensor(x, device=device, requires_grad=True)
    (g,) = grad(lucid.linalg.vector_norm(leaf, ord=p), [leaf], create_graph=True)
    (hw,) = grad(g, [leaf], grad_outputs=lucid.tensor(w, device=device))
    return hw


def _relative_errors(got: list[float], want: list[float], scale: list[float]) -> tuple[float, bool]:
    """The largest error against each entry's own terms, and whether every entry is finite."""
    worst = 0.0
    for g, w, s in zip(got, want, scale, strict=True):
        if g - g != 0.0:
            return math.inf, False
        if s >= 1e-30:
            worst = max(worst, abs(g - w) / s)
    return worst, True


def _sweep(device: str, p: float, tol: float, smallest: float = 1e-30) -> int:
    """Every (magnitude, span, cotangent scale) whose closed form fits float32: finite and within
    ``tol``.  Returns how many cases were checked."""
    checked = 0
    failures: list[str] = []
    for magnitude in MAGNITUDES:
        if magnitude < smallest:
            continue
        for span in SPANS:
            if magnitude * 10.0 ** (-span) < 3e-38:
                continue
            for cotangent in (1.0, 1e-3):
                x = lucid.tensor(_slice(magnitude, span)).tolist()
                w = lucid.tensor([d * magnitude * cotangent for d in DIRECTION]).tolist()
                want, scale = _hvp(x, w, p)
                if not _asked_for(x, scale):
                    continue
                got = _second_derivative(device, p, x, w).tolist()
                worst, finite = _relative_errors(got, want, scale)
                checked += 1
                if not finite or worst > tol:
                    failures.append(f"m={magnitude:g} span={span} w*{cotangent:g}: {got} vs {want}")
    assert not failures, f"p={p}: {len(failures)} of {checked} cases off\n" + "\n".join(
        failures[:4]
    )
    return checked


def test_the_reported_slice_has_a_finite_second_derivative(device: str) -> None:
    """The card: ``x = [1e20, 1e-10, 2e-10]``, ``p = 1.1``, ``w = 1e20`` gave ``[-inf, nan, nan]``."""
    x = [1e20, 1e-10, 2e-10]
    w = [1e20] * 3
    got = _second_derivative(device, 1.1, x, w).tolist()
    want, scale = _hvp(lucid.tensor(x).tolist(), w, 1.1)
    worst, finite = _relative_errors(got, want, scale)
    assert finite, got
    assert worst < 2e-5, (got, want)
    assert abs(got[1]) > 1e25


@pytest.mark.parametrize("p", BETWEEN_ONE_AND_TWO)
def test_between_one_and_two_the_second_derivative_matches_float64(device: str, p: float) -> None:
    assert _sweep(device, p, 3e-5) >= 80


@pytest.mark.parametrize("p", QUASI)
def test_below_one_the_second_derivative_matches_float64(device: str, p: float) -> None:
    assert _sweep(device, p, 3e-5) >= 40


@pytest.mark.parametrize("p", ABOVE_TWO)
def test_from_two_up_the_second_derivative_matches_float64(device: str, p: float) -> None:
    """These orders' recorded graph was already ratios in [0, 1]: kept as it was.  Below 1e-5
    the cotangent times ``|u|^(p-2)`` underflows (:func:`test_the_tiny_scale_underflow...`)."""
    assert _sweep(device, p, 3e-5, smallest=1e-5) >= 60


@pytest.mark.xfail(
    strict=True,
    reason="LCD-1018 FOUND: from p = 3 up, a slice below ~1e-15 spanning 10+ decades loses the "
    "small entries: w * |u|^(p-2) is below float32's smallest normal before the division by N",
)
def test_the_tiny_scale_underflow_from_three_up_is_still_open(device: str) -> None:
    x = _slice(1e-25, 10)
    w = [d * 1e-25 for d in DIRECTION]
    want, scale = _hvp(lucid.tensor(x).tolist(), w, 4.0)
    got = _second_derivative(device, 4.0, x, w).tolist()
    worst, finite = _relative_errors(got, want, scale)
    assert finite and worst < 3e-5


@pytest.mark.parametrize("p", [1.1, 1.5, 1.9, 0.5, 0.9])
def test_a_slice_with_zero_entries_leaves_the_others_exact_on_a_wide_spread(
    device: str, p: float
) -> None:
    """A zero entry's own second derivative is 0 (the true Hessian diagonal is infinite there);
    the unit of the slice is its smallest *non-zero* entry, so a zero does not make it 0."""
    x = [1e10, 0.0, -3e-10, 2e-5, 0.0]
    w = [1e10 * d for d in DIRECTION]
    got = _second_derivative(device, p, x, w).tolist()
    live = [i for i, v in enumerate(x) if v != 0.0]
    want, scale = _hvp([x[i] for i in live], [w[i] for i in live], p)
    assert all(v - v == 0.0 for v in got), got
    worst, _ = _relative_errors([got[i] for i in live], want, scale)
    assert worst < 5e-5, (got, want)


@pytest.mark.parametrize("p", [1.1, 1.5, 1.9, 0.9])
def test_each_slice_of_a_batch_has_its_own_scale_and_spread(device: str, p: float) -> None:
    """Rows 25 decades apart in scale and 30 in spread: nothing is shared across rows."""
    rows = [_slice(1e20, 30), _slice(1e-5, 20), _slice(1.0, 0)]
    cot = [[d * 1e20 for d in DIRECTION], [d * 1e-5 for d in DIRECTION], list(DIRECTION)]
    flat = [v for row in rows for v in row]
    leaf = lucid.tensor(flat, device=device, requires_grad=True).reshape(3, 5)
    (g,) = grad(lucid.linalg.vector_norm(leaf, ord=p, dim=1).sum(), [leaf], create_graph=True)
    w = lucid.tensor([v for row in cot for v in row], device=device).reshape(3, 5)
    (hw,) = grad(g, [leaf], grad_outputs=w)
    got = hw.reshape(15).tolist()
    xs = lucid.tensor(flat).tolist()
    for r in range(3):
        row_x, row_w = xs[5 * r : 5 * r + 5], cot[r]
        want, scale = _hvp(row_x, row_w, p)
        assert _asked_for(row_x, scale), (r, want)
        worst, finite = _relative_errors(got[5 * r : 5 * r + 5], want, scale)
        assert finite and worst < 5e-5, (r, got[5 * r : 5 * r + 5], want)


@pytest.mark.parametrize("magnitude", [1e-30, 1e-20, 1e-10, 1e-5, 1e5, 1e10, 1e20, 1e30])
@pytest.mark.parametrize("p", [1.05, 1.1, 1.25, 1.5, 1.75, 1.9])
def test_the_third_derivative_between_one_and_two_does_not_depend_on_the_scale(
    device: str, p: float, magnitude: float
) -> None:
    """``d/dx (u . H(x) w)`` is homogeneous of degree -2 in ``x``: scaling ``x``, ``w`` and ``u``
    by ``m`` gives the same number.  The graph that fixed the second derivative for a wide spread
    must keep that: its unit is a constant of the slice, not of the absolute scale."""

    def third(m: float) -> list[float]:
        x = lucid.tensor([v * m for v in BASE], device=device, requires_grad=True)
        (g,) = grad(lucid.linalg.vector_norm(x, ord=p), [x], create_graph=True)
        w = lucid.tensor([v * m for v in DIRECTION], device=device)
        (hw,) = grad(g, [x], grad_outputs=w, create_graph=True)
        u = lucid.tensor([v * m for v in SECOND], device=device)
        (out,) = grad(hw, [x], grad_outputs=u)
        return out.tolist()

    want, got = third(1.0), third(magnitude)
    scale = max(abs(v) for v in want)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert a - a == 0.0, f"entry {i} is {a}, want {b}"
        assert abs(a - b) <= 2e-3 * scale, f"entry {i}: {a} vs {b}"


@pytest.mark.parametrize("p", [1.1, 1.5, 1.9, 0.5, 0.9])
def test_the_recorded_first_derivative_still_equals_the_eager_one_on_a_wide_spread(
    device: str, p: float
) -> None:
    x = lucid.tensor(_slice(1e5, 30), device=device, requires_grad=True)
    n = lucid.linalg.vector_norm(x, ord=p)
    (eager,) = grad(n, [x], retain_graph=True)
    (recorded,) = grad(n, [x], create_graph=True)
    for i, (a, b) in enumerate(zip(recorded.tolist(), eager.tolist(), strict=True)):
        assert abs(a - b) <= 2e-6 * max(abs(b), 1e-30), f"entry {i}: {a} vs {b}"
