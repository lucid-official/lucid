"""LCD-1003: the derivatives of ``vector_norm`` across the float range, on both streams.

The CPU reduction was the plain graph ``(sum |x|^p)^(1/p)`` behind a window on the
*forward* value: outside ``2^+-(100/p)`` the call was redone on a rescaled copy, inside it
the plain graph stood.  The plain graph's second derivative forms ``s^(1/p - 2)`` for
``s = sum |x|^p`` (the root's own second derivative, reached through the recorded first
backward), which is ``N^(1 - 2p)``: float32 loses it for ``N`` near 1e-9 or 1e8 at ``p = 3``
(``inf`` / a subnormal), and the window narrows as ``p`` grows (1e-7 and 1e6 at ``p = 4``,
1e-5 and 1e3 at ``p = 6``).  Metal records ``PNormBackward``, whose backward only forms
ratios ``|x| / N`` in [0, 1]; the CPU stream now runs the same op, so every derivative order
is as scale-free as the first.

The closed forms are plain Python float64 from the float32 values read back from the leaf.
"""

import itertools
import math

import pytest

import lucid
import lucid.linalg
from lucid.autograd import grad

BASE = [3.0, -4.0, 1.0, 2.0, 0.5]
DIRECTION = [0.3, -1.0, 0.5, 2.0, -0.7]
SECOND = [-0.4, 0.9, 1.3, -0.2, 0.6]
ORDERS = [1.5, 2.0, 3.0, 4.0, 6.0]
#: Decades across (nearly) the float32 range, plus the edges of both bands the plain graph lost.
MAGNITUDES = sorted({10.0**e for e in range(-30, 31, 5)} | {1e-10, 1e-9, 1e-8, 1e-7, 1e6, 1e8, 1e9})


def _leaf(values: list[float], device: str) -> lucid.Tensor:
    return lucid.tensor(values, device=device, requires_grad=True)


def _norm_and_slope(x: list[float], p: float) -> tuple[float, list[float], list[float]]:
    """``N``, the ratios ``|x| / N`` and the gradient ``sign(x) (|x| / N)^(p-1)``."""
    top = max(abs(v) for v in x)
    n = top * sum((abs(v) / top) ** p for v in x) ** (1.0 / p)
    ratio = [abs(v) / n for v in x]
    return n, ratio, [math.copysign(r ** (p - 1.0), v) for r, v in zip(ratio, x, strict=True)]


def _hvp(x: list[float], w: list[float], p: float) -> list[float]:
    """``H w = (p-1)/N (|r|^(p-2) w - s <s, w>)`` for ``r = |x| / N`` and ``s`` the gradient."""
    n, ratio, slope = _norm_and_slope(x, p)
    dot = sum(s * v for s, v in zip(slope, w, strict=True))
    return [
        (p - 1.0) / n * (r ** (p - 2.0) * v - s * dot)
        for r, v, s in zip(ratio, w, slope, strict=True)
    ]


def _close(got: lucid.Tensor, want: list[float], tol: float) -> None:
    scale = max(abs(v) for v in want)
    for i, (x, y) in enumerate(zip(got.tolist(), want, strict=True)):
        assert x - x == 0.0, f"entry {i} is {x}, want {y}"
        assert abs(x - y) <= tol * scale, f"entry {i}: {x} vs {y} (scale {scale})"


@pytest.mark.parametrize("magnitude", MAGNITUDES)
@pytest.mark.parametrize("p", ORDERS)
def test_the_second_derivative_matches_the_closed_form(
    device: str, p: float, magnitude: float
) -> None:
    w = [v * magnitude for v in DIRECTION]
    x = _leaf([v * magnitude for v in BASE], device)
    (g,) = grad(lucid.linalg.vector_norm(x, ord=p), [x], create_graph=True)
    (hw,) = grad(g, [x], grad_outputs=lucid.tensor(w, device=device))
    _close(hw, _hvp(x.tolist(), w, p), 2e-5)


@pytest.mark.parametrize("magnitude", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("p", ORDERS)
def test_the_first_derivative_is_the_closed_form_at_moderate_magnitudes(
    device: str, p: float, magnitude: float
) -> None:
    x = _leaf([v * magnitude for v in BASE], device)
    (g,) = grad(lucid.linalg.vector_norm(x, ord=p), [x])
    _close(g, _norm_and_slope(x.tolist(), p)[2], 5e-7)


@pytest.mark.parametrize("p", ORDERS)
def test_the_recorded_first_derivative_equals_the_eager_one(device: str, p: float) -> None:
    x = _leaf(BASE, device)
    n = lucid.linalg.vector_norm(x, ord=p)
    (eager,) = grad(n, [x], retain_graph=True)
    (recorded,) = grad(n, [x], create_graph=True)
    _close(recorded, eager.tolist(), 1e-6)


@pytest.mark.parametrize("magnitude", [1e-30, 1e-20, 1e-10, 1e-5, 1e5, 1e10, 1e20, 1e30])
@pytest.mark.parametrize("p", [2.0, 3.0, 4.0])
def test_the_third_derivative_does_not_depend_on_the_scale(
    device: str, p: float, magnitude: float
) -> None:
    """``d/dx (u . H(x) w)`` is homogeneous of degree -2 in ``x``: with ``u`` and ``w`` scaled
    by ``m`` too, it is the same number at every magnitude ``m``."""

    def third(m: float) -> list[float]:
        x = _leaf([v * m for v in BASE], device)
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


@pytest.mark.parametrize("p", [2.0, 3.0, 4.0, 6.0])
def test_a_zero_slice_has_a_zero_gradient_and_a_zero_second_derivative(
    device: str, p: float
) -> None:
    """Below 2 a zero entry's second derivative is infinite, as the true Hessian's is."""
    x = _leaf([0.0, 0.0, 0.0], device)
    (g,) = grad(lucid.linalg.vector_norm(x, ord=p), [x], create_graph=True)
    assert g.tolist() == [0.0, 0.0, 0.0]
    (hw,) = grad(g, [x], grad_outputs=lucid.tensor([1.0, 2.0, 3.0], device=device))
    assert hw.tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("p", [1.5, 2.0, 3.0])
def test_a_zero_slice_beside_a_live_one_keeps_both_gradients(device: str, p: float) -> None:
    x = _leaf([0.0, 0.0, 0.0, 3.0, -4.0, 1.0], device).reshape(2, 3)
    (g,) = grad(lucid.linalg.vector_norm(x, ord=p, dim=1).sum(), [x])
    got = g.reshape(6).tolist()
    assert got[:3] == [0.0, 0.0, 0.0]
    _close(lucid.tensor(got[3:]), _norm_and_slope([3.0, -4.0, 1.0], p)[2], 5e-7)


def _slices(shape: tuple[int, ...], axes: set[int]) -> list[list[int]]:
    """The flat indices of each slice of a row-major tensor reduced over ``axes``."""
    strides = [math.prod(shape[d + 1 :]) for d in range(len(shape))]
    groups: dict[tuple[int, ...], list[int]] = {}
    for index in itertools.product(*(range(extent) for extent in shape)):
        kept = tuple(i for d, i in enumerate(index) if d not in axes)
        groups.setdefault(kept, []).append(sum(i * s for i, s in zip(index, strides, strict=True)))
    return list(groups.values())


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dim", [None, 0, 1, -1, (0, 2), (0, 1)])
@pytest.mark.parametrize("magnitude", [1e-9, 1.0, 1e8])
def test_a_batched_second_derivative_reduces_over_its_axes(
    device: str, dim: int | tuple[int, ...] | None, keepdim: bool, magnitude: float
) -> None:
    """Every slice has its own scale and its own closed form: ``d/dx sum_i (dN/dx)_i = H 1``."""
    shape = (2, 3, 4)
    flat = [(BASE[i % 5] + 0.37 * (i // 5)) * magnitude * (1.0 + 7.0 * (i % 3)) for i in range(24)]
    x = _leaf(flat, device).reshape(*shape)
    n = lucid.linalg.vector_norm(x, ord=3.0, dim=dim, keepdim=keepdim)
    (g,) = grad(n.sum(), [x], create_graph=True)
    (h,) = grad(g.sum(), [x])
    axes = (
        {0, 1, 2} if dim is None else {d % 3 for d in (dim if isinstance(dim, tuple) else (dim,))}
    )
    xs, hs = x.reshape(24).tolist(), h.reshape(24).tolist()
    for members in _slices(shape, axes):
        want = _hvp([xs[i] for i in members], [1.0] * len(members), 3.0)
        scale = max(abs(v) for v in want)
        for i, w in zip(members, want, strict=True):
            assert hs[i] - hs[i] == 0.0, f"entry {i} is {hs[i]}, want {w}"
            assert abs(hs[i] - w) <= 5e-5 * scale, f"entry {i}: {hs[i]} vs {w} (scale {scale})"
