"""LCD-996: ``F.normalize`` under ``create_graph=True`` across the float range.

The graph-mode backward recomputed the norm as an unscaled ``sum(|x|^p)`` and divided
by ``d * d``: a slice near 1e-25 came back NaN and one near 1e25 all zeros, though the
gradient is an ordinary float.  It is now the eager formula

    dx = (g - [||x|| >= eps] sgn(u) |u|^(p-1) <g, u>) / d,    u = x / d

built from tracked ops at the scale of the forward's denominator, so the tracked
arithmetic is order 1 whatever the slice's magnitude and the quotient rule's ``b^2``
(``DivBackward``) never leaves float32.  The second derivative is checked against a
central difference of the float64 closed form: ``dx`` is the gradient of
``<g, y(x)>``, so the Jacobian of ``dx`` is that function's symmetric Hessian and
``autograd.grad(dx, x, w)`` is the directional derivative of ``dx`` along ``w``.
The cotangent ``w`` is of the slice's magnitude: the Hessian is ~1/magnitude^2, which
leaves float32 itself at 1e-25 where the product ``H w`` (~1/magnitude) does not.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test.unit.nn.test_normalize_range import BASE, _close, _run, _truth

MAGNITUDES = [1e-30, 1e-25, 1e-10, 1.0, 1e10, 1e25, 1e30]
ORDERS = [1.0, 1.5, 2.0, 3.0]
#: The probe direction ``w`` and the incoming gradient ``g``, one row each of ``BASE``.
DIRECTION = np.array([[0.3, -1.0, 0.5, 2.0, -0.7], [1.0, 0.4, -0.2, 0.6, -1.5]], dtype=np.float32)
INCOMING = np.tile(np.arange(5.0, dtype=np.float32), (2, 1))
#: The finite-difference step, relative to the entries (float64 error ~1e-10 at this size).
STEP = 1e-5


def _derivative_along(
    arr: np.ndarray, g: np.ndarray, w: np.ndarray, p: float, eps: float, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(d y / d x . w, d dx / d x . w)`` by central differences of the float64 closed form."""
    x, gd, wd = (a.astype(np.float64) for a in (arr, g, w))
    y_plus, dx_plus = _truth(x + STEP * wd, gd, p, eps, axis)
    y_minus, dx_minus = _truth(x - STEP * wd, gd, p, eps, axis)
    return (y_plus - y_minus) / (2 * STEP), (dx_plus - dx_minus) / (2 * STEP)


def _twice(
    arr: np.ndarray, g: np.ndarray, w: np.ndarray, p: float, eps: float, dim: int, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(dx, H w, d(dx)/dg . w)`` from one ``create_graph=True`` backward."""
    x = lucid.tensor(arr, device=device, requires_grad=True)
    gt = lucid.tensor(g, device=device, requires_grad=True)
    y = F.normalize(x, p, dim, eps)
    (dx,) = lucid.autograd.grad(y, [x], grad_outputs=gt, create_graph=True)
    hw, jw = lucid.autograd.grad(dx, [x, gt], grad_outputs=lucid.tensor(w, device=device))
    return dx.numpy(), hw.numpy(), jw.numpy()


@pytest.mark.parametrize("p", ORDERS)
@pytest.mark.parametrize("magnitude", MAGNITUDES)
@pytest.mark.parametrize("eps", [1e-12, 1e-30])
def test_first_derivative_is_the_eager_one_and_the_closed_form(
    device: str, magnitude: float, eps: float, p: float
) -> None:
    arr = BASE * np.float32(magnitude)
    dx, _, _ = _twice(arr, INCOMING, DIRECTION * np.float32(magnitude), p, eps, -1, device)
    assert np.all(np.isfinite(dx))
    _close(dx, _truth(arr, INCOMING, p, eps, -1)[1], -1)
    eager = _run(arr, INCOMING, p, eps, -1, device)[1]
    scale = np.abs(eager).max(axis=-1, keepdims=True)
    np.testing.assert_allclose(dx / scale, eager / scale, rtol=0, atol=2e-6)


@pytest.mark.parametrize("p", ORDERS)
@pytest.mark.parametrize("magnitude", MAGNITUDES)
@pytest.mark.parametrize("eps", [1e-12, 1e-30])
def test_second_derivatives_match_the_float64_closed_form(
    device: str, magnitude: float, eps: float, p: float
) -> None:
    """``H w`` (through x) and ``J w`` (through g), where the slice is above eps and below it."""
    arr = BASE * np.float32(magnitude)
    w = DIRECTION * np.float32(magnitude)
    _, hw, jw = _twice(arr, INCOMING, w, p, eps, -1, device)
    want_j, want_h = _derivative_along(arr, INCOMING, w, p, eps, -1)
    assert np.all(np.isfinite(hw)) and np.all(np.isfinite(jw))
    _close(hw, want_h, -1)
    _close(jw, want_j, -1)


@pytest.mark.parametrize("p", [1.5, 2.0, 3.0])
@pytest.mark.parametrize("magnitude", [1e-25, 1.0, 1e25])
@pytest.mark.parametrize("dim", [0, 1, 2, -2])
def test_every_axis_of_a_strided_input(device: str, dim: int, magnitude: float, p: float) -> None:
    """``outer`` and ``inner`` both above 1: the saved norm's layout is the CPU kernel's."""
    rng = np.random.default_rng(11)
    arr = (rng.standard_normal((3, 4, 5)) * magnitude).astype(np.float32)
    g = rng.standard_normal((3, 4, 5)).astype(np.float32)
    w = (rng.standard_normal((3, 4, 5)) * magnitude).astype(np.float32)
    dx, hw, jw = _twice(arr, g, w, p, 1e-30, dim, device)
    want_j, want_h = _derivative_along(arr, g, w, p, 1e-30, dim)
    _close(dx, _truth(arr, g, p, 1e-30, dim)[1], dim)
    _close(hw, want_h, dim)
    _close(jw, want_j, dim)


@pytest.mark.parametrize("p", ORDERS)
@pytest.mark.parametrize("eps", [1e-12, 1e-30])
def test_a_zero_slice_has_gradient_g_over_eps_and_no_second_derivative(
    device: str, eps: float, p: float
) -> None:
    """``0 / eps`` is 0 with the constant denominator ``eps``: finite, and flat in ``x``."""
    arr = np.zeros((2, 4), dtype=np.float32)
    g = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 0.5, 2.0]], dtype=np.float32)
    w = np.ones((2, 4), dtype=np.float32)
    dx, hw, _ = _twice(arr, g, w, p, eps, -1, device)
    np.testing.assert_allclose(dx, g / np.float32(eps), rtol=1e-6, atol=0)
    assert np.all(hw == 0.0)


@pytest.mark.parametrize("p", ORDERS)
def test_a_floored_slice_with_a_zero_entry(device: str, p: float) -> None:
    """Under eps the denominator is constant and so is dx; a zero entry changes nothing.

    The slope of ``|u|^(p-1)`` is infinite at 0 below p = 2 and the gradient that reaches it
    from a masked slice is 0: ``inf * 0`` made the second derivative NaN where it is 0.
    """
    arr = np.array([[3e-4, 0.0, -4e-4]], dtype=np.float32)
    g = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    dx, hw, jw = _twice(arr, g, np.ones_like(arr), p, 1e-3, -1, device)
    np.testing.assert_allclose(dx, g / np.float32(1e-3), rtol=1e-6, atol=0)
    assert np.all(hw == 0.0)
    np.testing.assert_allclose(jw, 1.0 / np.float32(1e-3), rtol=1e-6, atol=0)


@pytest.mark.parametrize("p", [1.5, 2.0, 3.0])
def test_rows_on_both_sides_of_the_floor(device: str, p: float) -> None:
    """Norms ~5e-6 and 5e-5 are under eps 1e-4, ~5e-4 and 5 over it: each row takes its branch."""
    scales = np.array([1e-6, 1e-5, 1e-4, 1.0], dtype=np.float32)[:, None]
    arr = BASE[:1] * scales
    g = np.tile(np.arange(5.0, dtype=np.float32), (4, 1))
    w = np.tile(DIRECTION[:1], (4, 1)) * scales
    dx, hw, jw = _twice(arr, g, w, p, 1e-4, -1, device)
    want_j, want_h = _derivative_along(arr, g, w, p, 1e-4, -1)
    _close(dx, _truth(arr, g, p, 1e-4, -1)[1], -1)
    _close(hw, want_h, -1)
    _close(jw, want_j, -1)
    assert np.all(hw[:2] == 0.0)


@pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0])
def test_a_zero_entry_in_a_live_slice(device: str, p: float) -> None:
    """A zero entry's own component may be infinite for ``1 < p < 2``; no other one is.

    The slope of ``|u|^(p-1)`` is infinite at 0 below p = 2, so the Hessian's diagonal there is
    (the reference has the same infinity).  It must not travel: through ``u = x / d`` it
    reached the gradient of ``d`` and turned every entry of the slice into NaN.
    """
    arr = np.array([[3.0, 0.0, -4.0, 1.0]], dtype=np.float32)
    g = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    w = np.array([[0.5, 0.0, -1.0, 2.0]], dtype=np.float32)
    dx, hw, _ = _twice(arr, g, w, p, 1e-12, -1, device)
    want_h = _derivative_along(arr, g, w, p, 1e-12, -1)[1]
    assert np.all(np.isfinite(dx))
    live = [0, 2, 3]
    assert np.all(np.isfinite(hw[:, live]))
    _close(hw[:, live], want_h[:, live], -1)
    if p >= 2.0:
        assert np.isfinite(hw[0, 1])
        _close(hw, want_h, -1)


def test_a_non_finite_slice_poisons_only_itself(device: str) -> None:
    nan, inf = np.nan, np.inf
    arr = np.array(
        [[3.0, 4.0, 0.0], [nan, 1.0, 2.0], [0.0, 0.0, 0.0], [inf, 1.0, 2.0], [3e25, 4e25, 0.0]],
        dtype=np.float32,
    )
    g = np.ones_like(arr)
    w = np.ones_like(arr)
    for p in (2.0, 3.0):
        dx, hw, _ = _twice(arr, g, w, p, 1e-30, -1, device)
        assert np.all(np.isnan(dx[1])) and np.all(np.isnan(hw[1]))
        assert np.all(np.isfinite(dx[[0, 2, 4]])) and np.all(np.isfinite(hw[[0, 2, 4]]))
    # The slice at 1e25 has the ordinary gradient ~1/d; its H w ~ 1e-52 is below float32.
    assert np.all(dx[4] != 0.0)


@pytest.mark.parametrize("magnitude", [1e-200, 1e-100, 1.0, 1e100, 1e200])
@pytest.mark.parametrize("p", [1.5, 2.0, 3.0])
def test_float64_on_the_cpu(device_cpu_only: str, magnitude: float, p: float) -> None:
    """Float64 has its own range: ``d^2`` of 1e200 is inf there."""
    arr = BASE.astype(np.float64) * magnitude
    g = INCOMING.astype(np.float64)
    w = DIRECTION.astype(np.float64) * magnitude
    dx, hw, jw = _twice(arr, g, w, p, 1e-300, -1, device_cpu_only)
    want_j, want_h = _derivative_along(arr, g, w, p, 1e-300, -1)
    assert np.all(np.isfinite(hw)) and np.all(np.isfinite(jw))
    _close(dx, _truth(arr, g, p, 1e-300, -1)[1], -1)
    _close(hw, want_h, -1)
    _close(jw, want_j, -1)


def test_half_precision_differentiates_in_float32(device: str) -> None:
    """A half slice over 256 has a squared sum past float16's range; it is differentiated wide."""
    arr = (BASE[:1] * np.float32(300.0)).astype(np.float16)
    g = INCOMING[:1].astype(np.float16)
    w = (DIRECTION[:1] * np.float32(300.0)).astype(np.float16)
    x = lucid.tensor(arr, device=device, requires_grad=True)
    gt = lucid.tensor(g, device=device, requires_grad=True)
    y = F.normalize(x, 2.0, -1, 1e-12)
    (dx,) = lucid.autograd.grad(y, [x], grad_outputs=gt, create_graph=True)
    (hw,) = lucid.autograd.grad(dx, [x], grad_outputs=lucid.tensor(w, device=device))
    assert dx.dtype == hw.dtype == lucid.float16
    a32, g32, w32 = (a.astype(np.float32) for a in (arr, g, w))
    want_dx = _truth(a32, g32, 2.0, 1e-12, -1)[1]
    want_h = _derivative_along(a32, g32, w32, 2.0, 1e-12, -1)[1]
    # float16 carries ~1e-3 of the entry's own size.
    for got, want in ((dx, want_dx), (hw, want_h)):
        scale = np.abs(want).max()
        np.testing.assert_allclose(
            got.numpy().astype(np.float64) / scale, want / scale, rtol=0, atol=1e-2
        )


@pytest.mark.parametrize("magnitude", [1e-25, 1.0, 1e25])
def test_a_third_derivative_has_no_nan(device: str, magnitude: float) -> None:
    """The recorded second-order graph is itself differentiable at the extremes.

    The third derivative is ~1/magnitude^2: ``inf`` at 1e-25 and 0 at 1e25 are its
    float32 values there, a NaN is not.
    """
    arr = BASE[:1] * np.float32(magnitude)
    x = lucid.tensor(arr, device=device, requires_grad=True)
    y = F.normalize(x, 2.0, -1, 1e-30)
    (dx,) = lucid.autograd.grad(
        y, [x], grad_outputs=lucid.tensor(INCOMING[:1], device=device), create_graph=True
    )
    (hw,) = lucid.autograd.grad(
        dx,
        [x],
        grad_outputs=lucid.tensor(DIRECTION[:1] * np.float32(magnitude), device=device),
        create_graph=True,
    )
    (third,) = lucid.autograd.grad(hw.sum(), [x])
    assert not np.any(np.isnan(third.numpy()))
