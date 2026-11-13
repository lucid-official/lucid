"""LCD-998: the backward of a division across the float range.

``DivBackward`` formed ``db = -(g * a) / (b * b)``.  ``b * b`` leaves float32 for a
``|b|`` outside ~1e-19 .. 1e19, so a gradient that is an ordinary float came back 0
(``b`` near 1e25), NaN (``b`` near 1e-25, where ``0 / 0`` hid a zero numerator too) or
lost its second derivative under ``create_graph=True``: the quotient rule of every
division in the recorded graph forms the same square.  It is now
``-(g * (a / b)) / b`` in both formula paths, so every intermediate keeps the scale of
the result and so does the graph a second or third derivative walks.

Each case scales the operands the way a normalised quantity scales: ``b = beta * m``,
``a = alpha * b`` and ``g = gamma * b`` for a magnitude ``m``, so the gradients are of
order 1 (``da``, ``db``) or ``1 / m`` (their derivatives), inside float32 whatever ``m``
is, and the closed forms are evaluated in plain float64 from the float32 values read
back from the tensors.
"""

import math

import pytest

import lucid
import lucid.linalg
from lucid.autograd import grad

#: ``b``'s magnitude, log-spaced across (nearly) the whole float32 range.
MAGNITUDES = [10.0**e for e in range(-30, 31, 5)]
#: ``a / b``, with a zero numerator among them: ``0 / b^2`` is 0, ``0 / 0`` after underflow.
ALPHA = [0.5, -2.0, 3.0, 0.0, 1.5]
#: ``g / b`` and ``b / m``.
GAMMA = [1.0, 0.7, -1.3, 2.0, 0.4]
BETA = [1.0, -1.5, 0.8, 2.5, -0.6]
#: Cotangents of ``da`` and ``db`` for the second derivative (of order 1).
U = [0.7, -1.1, 0.9, 1.3, -0.5]
V = [1.3, 0.6, -0.8, 1.1, 0.9]
TOLERANCE = 2e-5


def _operands(magnitude: float) -> tuple[list[float], list[float], list[float]]:
    b = [beta * magnitude for beta in BETA]
    return (
        [alpha * x for alpha, x in zip(ALPHA, b, strict=True)],
        b,
        [gamma * x for gamma, x in zip(GAMMA, b, strict=True)],
    )


def _leaf(values: list[float], device: str, dtype: lucid.dtype = lucid.float32) -> lucid.Tensor:
    return lucid.tensor(values, dtype=dtype, device=device, requires_grad=True)


def _close(got: lucid.Tensor, want: list[float], scale: list[float], tol: float) -> None:
    """Every entry finite and within ``tol`` of ``want``, relative to ``scale`` (not 0)."""
    for i, (x, y, s) in enumerate(zip(got.tolist(), want, scale, strict=True)):
        assert x - x == 0.0, f"entry {i} is {x}, want {y}"
        assert abs(x - y) <= tol * s, f"entry {i}: {x} vs {y} (scale {s})"


def _first(a: list[float], b: list[float], g: list[float]) -> tuple[list[float], list[float]]:
    return (
        [gi / bi for gi, bi in zip(g, b, strict=True)],
        [-(gi / bi) * (ai / bi) for gi, ai, bi in zip(g, a, b, strict=True)],
    )


def _hessian_b(a: float, b: float, g: float, u: float, v: float) -> tuple[float, float]:
    """The two terms of ``d(u da + v db) / db``, by ratios so float64 at 1e200 stays a float."""
    return -u * (g / b) / b, 2 * v * (g / b) * (a / b) / b


def _scale(*terms: list[float]) -> list[float]:
    """The largest term of each entry, so a cancellation does not tighten the bound."""
    return [max(abs(t) for t in entry) or 1.0 for entry in zip(*terms, strict=True)]


@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_the_first_derivative_is_finite_and_exact(device: str, magnitude: float) -> None:
    a0, b0, g0 = _operands(magnitude)
    a, b, g = _leaf(a0, device), _leaf(b0, device), lucid.tensor(g0, device=device)
    da, db = grad(a / b, [a, b], grad_outputs=g)
    want_a, want_b = _first(a.tolist(), b.tolist(), g.tolist())
    _close(da, want_a, _scale(want_a), TOLERANCE)
    _close(db, want_b, _scale(want_b), TOLERANCE)


@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_the_recorded_first_derivative_equals_the_eager_one(device: str, magnitude: float) -> None:
    a0, b0, g0 = _operands(magnitude)
    a, b, g = _leaf(a0, device), _leaf(b0, device), lucid.tensor(g0, device=device)
    eager = grad(a / b, [a, b], grad_outputs=g)
    recorded = grad(a / b, [a, b], grad_outputs=g, create_graph=True)
    for want, got in zip(eager, recorded, strict=True):
        _close(got, want.tolist(), _scale(want.tolist()), 1e-6)


@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_the_second_derivatives_match_the_closed_form(device: str, magnitude: float) -> None:
    """``s = u . da + v . db``: ``ds/da``, ``ds/db`` and ``ds/dg`` of the recorded backward."""
    a0, b0, g0 = _operands(magnitude)
    a, b, g = _leaf(a0, device), _leaf(b0, device), _leaf(g0, device)
    u, v = lucid.tensor(U, device=device), lucid.tensor(V, device=device)
    da, db = grad(a / b, [a, b], grad_outputs=g, create_graph=True)
    h_a, h_b, h_g = grad(da * u + db * v, [a, b, g], grad_outputs=lucid.ones(5, device=device))
    av, bv, gv = a.tolist(), b.tolist(), g.tolist()
    rows = list(zip(av, bv, gv, U, V, strict=True))
    want_a = [-v_ * (g_ / b_) / b_ for a_, b_, g_, u_, v_ in rows]
    want_b = [_hessian_b(a_, b_, g_, u_, v_) for a_, b_, g_, u_, v_ in rows]
    want_g = [(u_ / b_, -v_ * (a_ / b_) / b_) for a_, b_, g_, u_, v_ in rows]
    _close(h_a, want_a, _scale(want_a), TOLERANCE)
    _close(h_b, [x + y for x, y in want_b], _scale(*zip(*want_b, strict=True)), TOLERANCE)
    _close(h_g, [x + y for x, y in want_g], _scale(*zip(*want_g, strict=True)), TOLERANCE)


@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_a_third_derivative_through_the_divisor(device: str, magnitude: float) -> None:
    """The recorded second derivative is itself recorded and keeps its range.

    The weights scale with ``m`` so that the third derivative (``~ 1 / m``) is a float.
    """
    a0, b0, g0 = _operands(magnitude)
    a, b, g = _leaf(a0, device), _leaf(b0, device), lucid.tensor(g0, device=device)
    u = lucid.tensor([x * magnitude for x in U], device=device)
    v = lucid.tensor([x * magnitude for x in V], device=device)
    da, db = grad(a / b, [a, b], grad_outputs=g, create_graph=True)
    ones = lucid.ones(5, device=device)
    (h_b,) = grad(da * u + db * v, [b], grad_outputs=ones, create_graph=True)
    (third,) = grad(h_b, [b], grad_outputs=ones)
    av, bv, gv = a.tolist(), b.tolist(), g.tolist()
    uv, vv = [x * magnitude for x in U], [x * magnitude for x in V]
    terms = [
        [2 * u_ * (g_ / b_) / b_ / b_, -6 * v_ * (g_ / b_) * (a_ / b_) / b_ / b_]
        for a_, b_, g_, u_, v_ in zip(av, bv, gv, uv, vv, strict=True)
    ]
    _close(third, [x + y for x, y in terms], _scale(*zip(*terms, strict=True)), 1e-4)


@pytest.mark.parametrize("magnitude", [1e-30, 1e-20, 1e-5, 1.0, 1e5, 1e20, 1e30])
def test_a_broadcast_divisor_second_derivative(device: str, magnitude: float) -> None:
    """``a`` of shape (3, 1) over ``b`` of shape (4,): both gradients reduce over the broadcast."""
    rows, cols = 3, 4
    b0 = [BETA[j] * magnitude for j in range(cols)]
    a0 = [[ALPHA[i] * b0[0] + 0.25 * i * b0[1]] for i in range(rows)]
    g0 = [[GAMMA[(i + j) % 5] * b0[j] for j in range(cols)] for i in range(rows)]
    a, b, g = _leaf(a0, device), _leaf(b0, device), _leaf(g0, device)
    u, v = lucid.tensor(U[:rows], device=device), lucid.tensor(V[:cols], device=device)
    da, db = grad(a / b, [a, b], grad_outputs=g, create_graph=True)
    assert da.shape == a.shape and db.shape == b.shape
    h_a, h_b, h_g = grad((da.reshape(rows) * u).sum() + (db * v).sum(), [a, b, g])
    av = [x[0] for x in a.tolist()]
    bv, gv = b.tolist(), g.tolist()
    want_da = [sum(gv[i][j] / bv[j] for j in range(cols)) for i in range(rows)]
    want_db = [-sum(gv[i][j] / bv[j] * (av[i] / bv[j]) for i in range(rows)) for j in range(cols)]
    _close(da.reshape(rows), want_da, _scale(want_da), TOLERANCE)
    _close(db, want_db, _scale(want_db), TOLERANCE)
    uv, vv = U[:rows], V[:cols]
    want_ha = [-sum(vv[j] * (gv[i][j] / bv[j]) / bv[j] for j in range(cols)) for i in range(rows)]
    want_hb = [
        sum(sum(_hessian_b(av[i], bv[j], gv[i][j], uv[i], vv[j])) for i in range(rows))
        for j in range(cols)
    ]
    want_hg = [
        uv[i] / bv[j] - vv[j] * (av[i] / bv[j]) / bv[j] for i in range(rows) for j in range(cols)
    ]
    _close(h_a.reshape(rows), want_ha, [abs(x) or 1.0 for x in want_ha], TOLERANCE)
    _close(h_b, want_hb, [abs(x) or 1.0 for x in want_hb], 1e-4)
    _close(h_g.reshape(rows * cols), want_hg, [abs(x) or 1.0 for x in want_hg], TOLERANCE)


@pytest.mark.parametrize("b_value", [1e-30, -1e-25, 1e-19, 1e-10, 1e10, -1e25, 1e30])
@pytest.mark.parametrize("create_graph", [False, True])
def test_a_zero_numerator_has_a_zero_divisor_gradient(
    device: str, b_value: float, create_graph: bool
) -> None:
    """``0 / b^2`` underflowed to ``0 / 0`` once ``b^2`` did; a large ``g`` must not matter either."""
    a = lucid.tensor([0.0, 0.0], device=device)
    b = _leaf([b_value, 2 * b_value], device)
    g = lucid.tensor([1e10, -3e8], device=device)
    (db,) = grad(a / b, [b], grad_outputs=g, create_graph=create_graph)
    assert all(x == 0.0 for x in db.tolist())


@pytest.mark.parametrize("magnitude", [1e-200, 1e-100, 1e100, 1e200])
def test_float64_on_the_cpu_keeps_its_range(magnitude: float) -> None:
    """``b^2`` leaves float64 outside ~1e-154 .. 1e154."""
    a0, b0, g0 = _operands(magnitude)
    a = _leaf(a0, "cpu", lucid.float64)
    b = _leaf(b0, "cpu", lucid.float64)
    g = _leaf(g0, "cpu", lucid.float64)
    u, v = lucid.tensor(U, dtype=lucid.float64), lucid.tensor(V, dtype=lucid.float64)
    da, db = grad(a / b, [a, b], grad_outputs=g, create_graph=True)
    want_a, want_b = _first(a.tolist(), b.tolist(), g.tolist())
    _close(da, want_a, _scale(want_a), 1e-12)
    _close(db, want_b, _scale(want_b), 1e-12)
    h_b = grad(da * u + db * v, [b], grad_outputs=lucid.ones(5, dtype=lucid.float64))[0]
    av, bv, gv = a.tolist(), b.tolist(), g.tolist()
    rows = list(zip(av, bv, gv, U, V, strict=True))
    terms = [list(_hessian_b(a_, b_, g_, u_, v_)) for a_, b_, g_, u_, v_ in rows]
    _close(h_b, [x + y for x, y in terms], _scale(*zip(*terms, strict=True)), 1e-12)


@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_reciprocal_keeps_its_range_through_two_derivatives(device: str, magnitude: float) -> None:
    """``reciprocal``'s ``-g / x^2`` is the same square: ``x = beta m``, ``g = gamma m``.

    The weights scale with ``m`` so the second derivative (``~ 1 / m``) is a float.
    """
    x0 = [beta * magnitude for beta in BETA]
    g0 = [gamma * magnitude for gamma in GAMMA]
    x, g = _leaf(x0, device), _leaf(g0, device)
    v = lucid.tensor([w * magnitude for w in V], device=device)
    (dx,) = grad(lucid.reciprocal(x), [x], grad_outputs=g, create_graph=True)
    h_x, h_g = grad(dx * v, [x, g], grad_outputs=lucid.ones(5, device=device))
    xv, gv, vv = x.tolist(), g.tolist(), [w * magnitude for w in V]
    want_dx = [-(g_ / x_) / x_ for x_, g_ in zip(xv, gv, strict=True)]
    want_hx = [2 * v_ * (g_ / x_) / x_ / x_ for x_, g_, v_ in zip(xv, gv, vv, strict=True)]
    want_hg = [-(v_ / x_) / x_ for x_, v_ in zip(xv, vv, strict=True)]
    _close(dx, want_dx, _scale(want_dx), TOLERANCE)
    _close(h_x, want_hx, _scale(want_hx), TOLERANCE)
    _close(h_g, want_hg, _scale(want_hg), TOLERANCE)
    (eager,) = grad(lucid.reciprocal(x), [x], grad_outputs=g)
    _close(eager, want_dx, _scale(want_dx), TOLERANCE)


#: The norm's coordinates and the probe direction of ``H w`` (one slice, largest entry 4).
NORM_BASE = [3.0, -4.0, 1.0, 2.0, 0.5]
NORM_DIRECTION = [0.3, -1.0, 0.5, 2.0, -0.7]


def _norm_hvp(x: list[float], w: list[float], p: float) -> list[float]:
    """``H w`` of ``||x||_p`` in float64: ``(p-1)/n (|r|^(p-2) w - s <s, w>)``, ``r = x/n``."""
    top = max(abs(v) for v in x)
    n = top * sum((abs(v) / top) ** p for v in x) ** (1 / p)
    r = [v / n for v in x]
    s = [math.copysign(abs(ri) ** (p - 1), ri) for ri in r]
    sw = sum(si * wi for si, wi in zip(s, w, strict=True))
    return [(p - 1) / n * (abs(ri) ** (p - 2) * wi - si * sw) for ri, wi, si in zip(r, w, s, strict=True)]


def _check_norm_hvp(device: str, ord_: float, magnitude: float) -> None:
    w = [v * magnitude for v in NORM_DIRECTION]
    x = _leaf([v * magnitude for v in NORM_BASE], device)
    (g,) = grad(lucid.linalg.vector_norm(x, ord=ord_), [x], create_graph=True)
    (hw,) = grad(g, [x], grad_outputs=lucid.tensor(w, device=device))
    want = _norm_hvp(x.tolist(), w, ord_)
    _close(hw, want, [max(abs(v) for v in want)] * 5, 1e-5)


@pytest.mark.parametrize("magnitude", MAGNITUDES)
@pytest.mark.parametrize("ord_", [2.0, 3.0])
def test_a_norm_second_derivative_divides_by_the_tracked_norm(
    device: str, ord_: float, magnitude: float
) -> None:
    """``NormBackward`` / ``PNormBackward`` record ``A / N`` for the tracked norm ``N``.

    That is a ``DivBackward`` whose divisor is of the slice's magnitude: its quotient
    rule returned NaN for a slice near 1e-25 or 1e25 on Metal.  ``H w`` of the norm is
    the closed form ``(p-1)/n (|r|^(p-2) w - s <s, w>)``.
    """
    _check_norm_hvp(device, ord_, magnitude)


#: bfloat16 keeps 8 bits of significand: its gradients are checked to 1e-2, the float32 0-d's to 1e-5.
HALF_TOLERANCE = 1e-2
SCALAR_MAGNITUDES = [1e-30, 1e-25, 1e-10, 1.0, 1e10, 1e25, 1e30]


def _bf16(values: list[float], device: str, requires_grad: bool = False) -> lucid.Tensor:
    return lucid.tensor(values, dtype=lucid.bfloat16, device=device, requires_grad=requires_grad)


@pytest.mark.parametrize("create_graph", [False, True])
@pytest.mark.parametrize("magnitude", SCALAR_MAGNITUDES)
def test_a_half_tensor_over_a_zero_dim_divisor(
    device: str, magnitude: float, create_graph: bool
) -> None:
    """``t / s`` for a bfloat16 ``t`` and a 0-d float32 ``s`` (``_Opmath.h``): ``s^2`` was formed.

    ``ds = -sum(g t) / s^2``, with ``t = beta s`` and ``g = gamma s`` so that it is of order 1.
    """
    s = lucid.tensor(magnitude, device=device, requires_grad=True)
    t = _bf16([beta * magnitude for beta in BETA], device, True)
    g = _bf16([gamma * magnitude for gamma in GAMMA], device)
    ds, dt = grad(t / s, [s, t], grad_outputs=g, create_graph=create_graph)
    sv, tv, gv = s.item(), t.tolist(), g.tolist()
    terms = [-(g_ / sv) * (t_ / sv) for t_, g_ in zip(tv, gv, strict=True)]
    assert abs(ds.item() - sum(terms)) <= 1e-5 * sum(abs(x) for x in terms)
    want_dt = [g_ / sv for g_ in gv]
    _close(dt, want_dt, [abs(x) for x in want_dt], HALF_TOLERANCE)


@pytest.mark.parametrize("create_graph", [False, True])
@pytest.mark.parametrize("magnitude", SCALAR_MAGNITUDES)
def test_a_zero_dim_numerator_over_a_half_tensor(
    device: str, magnitude: float, create_graph: bool
) -> None:
    """``s / t``: ``dt = -(g s) / t^2`` formed ``t^2``, which leaves float32 for a bfloat16 ``t``."""
    s = lucid.tensor(magnitude, device=device, requires_grad=True)
    t = _bf16([beta * magnitude for beta in BETA], device, True)
    g = _bf16([gamma * magnitude for gamma in GAMMA], device)
    ds, dt = grad(s / t, [s, t], grad_outputs=g, create_graph=create_graph)
    sv, tv, gv = s.item(), t.tolist(), g.tolist()
    want_dt = [-(g_ / t_) * (sv / t_) for t_, g_ in zip(tv, gv, strict=True)]
    _close(dt, want_dt, [abs(x) for x in want_dt], HALF_TOLERANCE)
    want_ds = [g_ / t_ for t_, g_ in zip(tv, gv, strict=True)]
    assert abs(ds.item() - sum(want_ds)) <= 1e-5 * sum(abs(x) for x in want_ds)
    if create_graph:
        (h_s,) = grad(dt, [s], grad_outputs=_bf16([1.0] * 5, device))
        want_h = -sum(g_ / t_ / t_ for t_, g_ in zip(tv, gv, strict=True))
        assert abs(h_s.item() - want_h) <= 2e-2 * abs(want_h)
