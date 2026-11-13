"""LCD-994: ``F.normalize`` forward and backward across the float range.

The eager backward formed ``|x|^(p-1) <g, x> / d^(p+1)``, a power of the norm ``d``
that leaves the float32 range for a norm outside ~1e-13 .. 1e13 (``d^3`` at
``p = 2``): the gradient of a row near 1e25 or 1e-25 came back NaN where it is an
ordinary float.  The CPU forward accumulated ``|x|^p`` in float32, so its norm was
inf (the row came back all zeros) where Metal's range-safe kernel was right.

The backward is now ``(g - sgn(u) |u|^(p-1) <g, u>) / d`` with ``u = x / d``, the
forward output, and the CPU norm is taken relative to the largest magnitude.  Every
case compares both devices with a float64 closed form of the same float32 inputs.

Below ``eps`` the denominator is the constant ``eps``, so the gradient is ``g / eps``:
the dependence of the norm on ``x`` is clamped away, as the reference's ``clamp_min``
does and graph mode (``create_graph=True``) already did.  Eager kept the second term,
and a norm exactly ``eps`` keeps it (the clamp passes its gradient at the boundary).
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

#: Five entries, the largest 4, so a slice at magnitude 1e36 stays inside float32.
BASE = np.array([[3.0, -4.0, 1.0, 2.0, 0.5], [0.25, 0.5, -2.0, 7.0, 1.0]], dtype=np.float32)
#: Float32's smallest normal is 1.18e-38 and Metal flushes below it.
MAGNITUDES = [1e-30, 1e-25, 1e-10, 1.0, 1e10, 1e25, 1e30]
ORDERS = [0.5, 1.0, 1.5, 2.0, 3.0]
#: The bound on an entry's error, relative to the largest entry of its slice.
SCALE = 2e-5


def _truth(
    arr: np.ndarray, g: np.ndarray, p: float, eps: float, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(y, dx)`` in float64 for ``y = x / max(||x||_p, eps)``, relative to the largest entry.

    A norm below ``eps`` is a constant denominator: ``dx = g / eps``.
    """
    x = arr.astype(np.float64)
    top = np.abs(x).max(axis=axis, keepdims=True)
    safe = np.where(top > 0, top, 1.0)
    n = np.where(
        top > 0, safe * ((np.abs(x) / safe) ** p).sum(axis=axis, keepdims=True) ** (1 / p), 0.0
    )
    d = np.maximum(n, eps)
    u = x / d
    along = np.where(n < eps, 0.0, (g.astype(np.float64) * u).sum(axis=axis, keepdims=True))
    return u, (g - np.sign(u) * np.abs(u) ** (p - 1) * along) / d


def _close(got: np.ndarray, want: np.ndarray, axis: int) -> None:
    """Each slice within ``SCALE`` of its largest expected entry."""
    scale = np.abs(want).max(axis=axis, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float64) / scale, want / scale, rtol=0, atol=SCALE
    )


def _run(
    arr: np.ndarray, g: np.ndarray, p: float, eps: float, dim: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    x = lucid.tensor(arr, device=device, requires_grad=True)
    y = F.normalize(x, p, dim, eps)
    y.backward(lucid.tensor(g, device=device))
    return y.numpy(), x.grad.numpy()


@pytest.mark.parametrize("p", ORDERS)
@pytest.mark.parametrize("magnitude", MAGNITUDES)
@pytest.mark.parametrize("eps", [1e-12, 1e-30])
def test_forward_and_backward_match_the_float64_closed_form(
    device: str, magnitude: float, eps: float, p: float
) -> None:
    arr = BASE * np.float32(magnitude)
    # The incoming gradient is the same at every magnitude, so dx is ~1/magnitude.
    g = np.tile(np.arange(5.0, dtype=np.float32), (2, 1))
    y, dx = _run(arr, g, p, eps, -1, device)
    want_y, want_dx = _truth(arr, g, p, eps, -1)
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(dx))
    _close(y, want_y, -1)
    _close(dx, want_dx, -1)


@pytest.mark.parametrize("p", [1.5, 2.0, 3.0])
@pytest.mark.parametrize("magnitude", [1e-25, 1.0, 1e25])
@pytest.mark.parametrize("dim", [0, 1, 2, -2])
def test_every_axis_of_a_strided_input(device: str, dim: int, magnitude: float, p: float) -> None:
    """``outer`` and ``inner`` both above 1 exercise the CPU kernel's stride."""
    rng = np.random.default_rng(7)
    arr = (rng.standard_normal((3, 4, 5)) * magnitude).astype(np.float32)
    g = rng.standard_normal((3, 4, 5)).astype(np.float32)
    y, dx = _run(arr, g, p, 1e-30, dim, device)
    want_y, want_dx = _truth(arr, g, p, 1e-30, dim)
    _close(y, want_y, dim)
    _close(dx, want_dx, dim)


@pytest.mark.parametrize("eps", [1e-12, 1e-30])
def test_a_zero_slice_takes_the_eps_path(device: str, eps: float) -> None:
    """``0 / eps`` is 0 and its gradient ``g / eps`` — finite, not the NaN of ``0 / eps^3``."""
    arr = np.zeros((2, 4), dtype=np.float32)
    g = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 0.5, 2.0]], dtype=np.float32)
    for p in (1.0, 2.0, 3.0):
        y, dx = _run(arr, g, p, eps, -1, device)
        assert np.all(y == 0.0)
        np.testing.assert_allclose(dx, g / np.float32(eps), rtol=1e-6, atol=0)


def test_slices_are_independent(device: str) -> None:
    """A NaN or an infinity poisons its own slice and no other."""
    nan, inf = np.nan, np.inf
    arr = np.array(
        [[3.0, 4.0, 0.0], [nan, 1.0, 2.0], [0.0, 0.0, 0.0], [inf, 1.0, 2.0], [3e25, 4e25, 0.0]],
        dtype=np.float32,
    )
    g = np.ones_like(arr)
    for p in (1.5, 2.0, 3.0):
        y, dx = _run(arr, g, p, 1e-30, -1, device)
        assert np.all(np.isnan(y[1])) and np.all(np.isnan(dx[1]))
        assert np.all(np.isnan(dx[3]))
        assert np.all(np.isfinite(y[[0, 2, 4]])) and np.all(np.isfinite(dx[[0, 2, 4]]))
        _close(y[[0, 4]], _truth(arr[[0, 4]], g[[0, 4]], p, 1e-30, -1)[0], -1)
        _close(dx[[0, 4]], _truth(arr[[0, 4]], g[[0, 4]], p, 1e-30, -1)[1], -1)


@pytest.mark.parametrize("magnitude", [1e-200, 1e-100, 1.0, 1e100, 1e200])
@pytest.mark.parametrize("p", [1.5, 2.0, 3.0])
def test_float64_on_the_cpu(device_cpu_only: str, magnitude: float, p: float) -> None:
    """Float64 has its own range: ``x^2`` of 1e200 is inf there."""
    arr = BASE.astype(np.float64) * magnitude
    g = np.tile(np.arange(5.0), (2, 1))
    y, dx = _run(arr, g, p, 1e-300, -1, device_cpu_only)
    want_y, want_dx = _truth(arr, g, p, 1e-300, -1)
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(dx))
    _close(y, want_y, -1)
    _close(dx, want_dx, -1)


@pytest.mark.parametrize("dtype", [np.float16])
def test_half_precision_reduces_in_float32(device: str, dtype: type) -> None:
    arr = BASE.astype(dtype)
    g = np.tile(np.arange(5.0), (2, 1)).astype(dtype)
    x = lucid.tensor(arr, device=device, requires_grad=True)
    y = F.normalize(x, 2.0, -1, 1e-12)
    y.backward(lucid.tensor(g, device=device))
    assert y.dtype == x.grad.dtype == lucid.float16
    want_y, want_dx = _truth(arr.astype(np.float32), g.astype(np.float32), 2.0, 1e-12, -1)
    np.testing.assert_allclose(y.numpy().astype(np.float64), want_y, rtol=0, atol=2e-3)
    np.testing.assert_allclose(x.grad.numpy().astype(np.float64), want_dx, rtol=0, atol=4e-3)


def test_a_transposed_view(device: str) -> None:
    arr = np.ascontiguousarray(BASE.T)
    x = lucid.tensor(arr, device=device, requires_grad=True)
    view = x.T
    y = F.normalize(view, 2.0, -1, 1e-12)
    y.sum().backward()
    want_y, want_dx = _truth(BASE, np.ones_like(BASE), 2.0, 1e-12, -1)
    np.testing.assert_allclose(y.numpy(), want_y, rtol=2e-6, atol=0)
    np.testing.assert_allclose(x.grad.numpy(), want_dx.T, rtol=0, atol=2e-6)


@pytest.mark.parametrize("shape", [(0, 5), (3, 0)])
def test_an_empty_input(device: str, shape: tuple[int, int]) -> None:
    x = lucid.tensor(np.zeros(shape, dtype=np.float32), device=device, requires_grad=True)
    y = F.normalize(x, 2.0, -1, 1e-12)
    y.sum().backward()
    assert tuple(y.shape) == shape and tuple(x.grad.shape) == shape


def test_a_zero_dimensional_input(device: str) -> None:
    x = lucid.tensor(np.float32(-3.0 * 1e25), device=device, requires_grad=True)
    y = F.normalize(x, 2.0, 0, 1e-12)
    y.backward()
    assert float(y.item()) == pytest.approx(-1.0, rel=1e-6)
    assert math.isfinite(float(x.grad.item()))
    assert float(x.grad.item()) == pytest.approx(0.0, abs=1e-30)


# ── the eps floor ────────────────────────────────────────────────────────────


def _graph_mode(arr: np.ndarray, g: np.ndarray, p: float, eps: float, device: str) -> np.ndarray:
    x = lucid.tensor(arr, device=device, requires_grad=True)
    y = F.normalize(x, p, -1, eps)
    (dx,) = lucid.autograd.grad(
        y, [x], grad_outputs=lucid.tensor(g, device=device), create_graph=True
    )
    return dx.numpy()


@pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0])
def test_below_eps_the_gradient_is_g_over_eps(device: str, p: float) -> None:
    """The norm 5e-4 is under eps 1e-3: the unfloored form gave ``[1060, 1920, 3020]`` at p = 2."""
    arr = np.array([[3e-4, -4e-4, 1e-4]], dtype=np.float32)
    g = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    y, dx = _run(arr, g, p, 1e-3, -1, device)
    np.testing.assert_allclose(y, arr / np.float32(1e-3), rtol=1e-6, atol=0)
    np.testing.assert_allclose(dx, g / np.float32(1e-3), rtol=1e-6, atol=0)
    np.testing.assert_allclose(_graph_mode(arr, g, p, 1e-3, device), dx, rtol=1e-5, atol=0)


@pytest.mark.parametrize("p", [1.0, 2.0])
def test_a_norm_exactly_eps_keeps_its_derivative(device: str, p: float) -> None:
    """``max`` is the reference's ``clamp_min``: at the boundary the gradient passes."""
    arr = np.array([[3.0, 4.0]] if p == 2.0 else [[1.0, 4.0]], dtype=np.float32)
    eps = 5.0
    g = np.array([[1.0, 2.0]], dtype=np.float32)
    y, dx = _run(arr, g, p, eps, -1, device)
    _, want = _truth(arr, g, p, eps, -1)
    np.testing.assert_allclose(y, arr / eps, rtol=1e-6, atol=0)
    np.testing.assert_allclose(dx, want, rtol=2e-6, atol=1e-7)
    assert not np.allclose(dx, g / np.float32(eps), rtol=1e-3)
    np.testing.assert_allclose(_graph_mode(arr, g, p, eps, device), dx, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("p", [1.5, 2.0, 3.0])
def test_rows_on_both_sides_of_the_floor(device: str, p: float) -> None:
    """Norms ~5e-6 and 5e-5 are under eps 1e-4, ~5e-4 and 5 over it: each row takes its own branch."""
    arr = BASE[:1] * np.array([1e-6, 1e-5, 1e-4, 1.0], dtype=np.float32)[:, None]
    g = np.tile(np.arange(5.0, dtype=np.float32), (4, 1))
    _, dx = _run(arr, g, p, 1e-4, -1, device)
    _close(dx, _truth(arr, g, p, 1e-4, -1)[1], -1)
    _close(dx, _graph_mode(arr, g, p, 1e-4, device), -1)
