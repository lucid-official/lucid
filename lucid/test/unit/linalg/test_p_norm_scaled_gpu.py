"""LCD-944: the GPU stream's range-safe entrywise p-norm.

``vector_norm`` / ``matrix_norm('fro')`` / ``pdist`` / ``cdist`` / ``F.normalize``
answered 0 on Metal for a float32 slice near 1e-20 (``sum(x * x)`` flushes the
squares) and ``inf`` for one near 1e25.  ``backend/gpu/PNorm.h`` accumulates
relative to the running largest magnitude in one Metal kernel.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.linalg as LA
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.test._fixtures.tolerance_guard import noise_when_below

#: Metal flushes float32 subnormal operands, so 1.18e-38 is the floor.
TINY = [1e-20, 1e-30, 1e-37]
HUGE = [1e25, 1e30, 1e36]
ORDERS = [1.5, 2.0, 3.5]


def _t(a: np.ndarray, device: str, **kw: object) -> lucid.Tensor:
    return lucid.tensor(a.astype(np.float32), device=device, **kw)


def _truth(a: np.ndarray, p: float, axis: object = None, keepdims: bool = False) -> np.ndarray:
    """The float64 norm of the float32 values, computed relative to the largest."""
    x = np.abs(a.astype(np.float32).astype(np.float64))
    top = x.max(axis=axis, keepdims=True)
    safe = np.where(top > 0, top, 1.0)
    s = ((x / safe) ** p).sum(axis=axis, keepdims=True)
    out = safe * s ** (1.0 / p)
    if not keepdims:
        out = out.reshape(np.sum(a, axis=axis).shape) if axis is not None else out.reshape(())
    return out


def _ulps(got: np.ndarray, want: np.ndarray) -> float:
    rel = np.abs(np.asarray(got, dtype=np.float64) - want) / np.where(want == 0, 1.0, want)
    return float(rel.max()) / 1.1920929e-07


@pytest.mark.parametrize("p", ORDERS)
@pytest.mark.parametrize("magnitude", TINY + HUGE)
def test_extreme_magnitudes_keep_their_norm(
    device_gpu_only: str, magnitude: float, p: float
) -> None:
    arr = np.full((2, 3), magnitude, dtype=np.float32) * np.array([1, -2, 3], dtype=np.float32)
    got = LA.vector_norm(_t(arr, device_gpu_only), p).item()
    assert got == pytest.approx(float(_truth(arr, p)), rel=1e-5, abs=0)


@pytest.mark.parametrize("kind", ["vector_norm", "matrix_norm_fro"])
def test_metal_entrywise_norm_at_tiny_magnitude(device_gpu_only: str, kind: str) -> None:
    """The LCD-940 xfail: Metal answered 0 for a float32 slice at 1e-20."""
    x = _t(np.full((2, 3), 1e-20), device_gpu_only)
    got = LA.vector_norm(x) if kind == "vector_norm" else LA.matrix_norm(x, "fro")
    assert got.item() == pytest.approx(float(np.sqrt(6.0)) * 1e-20, rel=1e-5, abs=0)


@pytest.mark.parametrize("p", ORDERS)
@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        ((7,), None),
        ((5, 9), 1),
        ((5, 9), 0),
        ((4, 6, 5), (0, 2)),
        ((4, 6, 5), (1,)),
        ((3, 4, 5, 6), (1, 2)),
        ((70, 3), -1),
        ((2, 2049), 1),
        ((3, 40000), 1),
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
def test_matches_float64_at_every_layout(
    device_gpu_only: str, shape: tuple[int, ...], dim: object, p: float, keepdim: bool
) -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(shape).astype(np.float32)
    got = LA.vector_norm(_t(arr, device_gpu_only), p, dim=dim, keepdim=keepdim).numpy()
    axis = dim if dim is None or isinstance(dim, int) else tuple(dim)
    want = _truth(arr, p, axis, keepdim)
    assert got.shape == want.shape
    assert _ulps(got, want) < 12


@pytest.mark.parametrize("p", ORDERS)
def test_every_row_is_scaled_on_its_own(device_gpu_only: str, p: float) -> None:
    """A tiny row beside a normal one and a huge one keeps each row's own scale."""
    base = np.array([3.0, -4.0, 1.0, 2.0] * 40, dtype=np.float32)
    arr = np.stack([base * 1e-25, base, base * 1e25])
    got = LA.vector_norm(_t(arr, device_gpu_only), p, dim=1).numpy()
    assert _ulps(got, _truth(arr, p, 1)) < 8


@pytest.mark.parametrize("width", [3, 64, 65, 700, 2049, 5000, 100000])
def test_a_spike_anywhere_in_a_row_is_the_scale(device_gpu_only: str, width: int) -> None:
    """The largest entry may sit in any lane or chunk: first, middle, last."""
    arr = np.full((3, width), 1e-22, dtype=np.float32)
    for row, col in enumerate((0, width // 2, width - 1)):
        arr[row, col] = 3.0
    got = LA.vector_norm(_t(arr, device_gpu_only), 2.0, dim=1).numpy()
    assert _ulps(got, _truth(arr, 2.0, 1)) < 4
    assert np.all(got >= 3.0)


@pytest.mark.parametrize("p", ORDERS)
def test_special_values(device_gpu_only: str, p: float) -> None:
    nan, inf = float("nan"), float("inf")
    arr = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, inf, 2.0, 0.0],
            [inf, inf, 1.0, 1.0],
            [1.0, nan, 2.0, 3.0],
            [inf, nan, 0.0, 0.0],
            [-inf, 1.0, 1.0, 1.0],
            [3.0, -4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    got = LA.vector_norm(_t(arr, device_gpu_only), p, dim=1).numpy()
    five = (3.0**p + 4.0**p) ** (1.0 / p)
    assert got[0] == 0.0
    assert got[1] == math.inf and got[2] == math.inf and got[5] == math.inf
    assert math.isnan(got[3]) and math.isnan(got[4])
    assert got[6] == pytest.approx(five, rel=1e-6)


@pytest.mark.parametrize("width", [4, 5000, 100000])
def test_nan_and_inf_survive_the_chunk_combine(device_gpu_only: str, width: int) -> None:
    arr = np.zeros((3, width), dtype=np.float32)
    arr[0, width - 1] = np.nan
    arr[1, width // 2] = np.inf
    arr[2, 0] = np.inf
    arr[2, width - 1] = np.nan
    got = LA.vector_norm(_t(arr, device_gpu_only), 2.0, dim=1).numpy()
    assert math.isnan(got[0]) and got[1] == math.inf and math.isnan(got[2])


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
def test_half_inputs_accumulate_in_float32(device_gpu_only: str, dtype: lucid.dtype) -> None:
    ones = lucid.ones(65536, dtype=dtype, device=device_gpu_only)
    got = LA.vector_norm(ones)
    assert got.dtype == dtype and got.item() == 256.0
    small = lucid.full((6,), 1e-3, dtype=dtype, device=device_gpu_only)
    assert LA.vector_norm(small, 3.5).item() == pytest.approx(
        float((6 * float(small[0].item()) ** 3.5) ** (1 / 3.5)), rel=1e-2
    )


@pytest.mark.parametrize("p", ORDERS)
@pytest.mark.parametrize("magnitude", [1.0, 1e-20, 1e25])
def test_gradient_is_the_analytic_slope_at_any_magnitude(
    device_gpu_only: str, p: float, magnitude: float
) -> None:
    base = np.array([[3.0, -4.0, 1.0, 0.0], [0.5, 0.5, -2.0, 7.0]], dtype=np.float32)
    arr = base * np.float32(magnitude)
    x = _t(arr, device_gpu_only, requires_grad=True)
    LA.vector_norm(x, p, dim=1).sum().backward()
    xd = arr.astype(np.float64)
    top = np.abs(xd).max(axis=1, keepdims=True)
    norm = top * (np.abs(xd / top) ** p).sum(axis=1, keepdims=True) ** (1 / p)
    want = np.sign(xd) * (np.abs(xd) / norm) ** (p - 1)
    got = x.grad.numpy()
    assert np.all(np.isfinite(got))
    assert _ulps(got, want) < 40


def test_gradient_of_a_zero_slice_is_zero(device_gpu_only: str) -> None:
    x = _t(np.zeros((2, 3)), device_gpu_only, requires_grad=True)
    LA.vector_norm(x, 2.0, dim=1).sum().backward()
    assert x.grad.numpy().tolist() == [[0.0] * 3] * 2


def test_gradient_keeps_a_nan_slice_nan(device_gpu_only: str) -> None:
    x = _t(np.array([[1.0, np.nan, 2.0]]), device_gpu_only, requires_grad=True)
    LA.vector_norm(x, 2.0, dim=1).sum().backward()
    assert np.all(np.isnan(x.grad.numpy()))


@pytest.mark.parametrize("p", [2.0, 3.5])
def test_second_derivative_agrees_with_the_cpu(device_gpu_only: str, p: float) -> None:
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((3, 5)).astype(np.float32) + 2.0
    out = []
    for dev in ("cpu", device_gpu_only):
        x = _t(arr, dev, requires_grad=True)
        (g,) = lucid.autograd.grad(LA.vector_norm(x, p, dim=1).sum(), x, create_graph=True)
        (h,) = lucid.autograd.grad((g * g).sum(), x)
        out.append(h.numpy())
    # For p=2 the gradient is a unit vector, so (g * g).sum() is constant: both h are roundoff.
    with noise_when_below(out[0], 1e-6, "the second derivative of a constant is 0"):
        np.testing.assert_allclose(out[1], out[0], rtol=2e-4, atol=1e-6)


def test_pdist_and_cdist_keep_tiny_distances(device_gpu_only: str) -> None:
    """The entrywise path (any p but the default-mode 2, which is a matmul)."""
    pts = np.array([[0.0, 0.0], [2e-20, 0.0], [0.0, 4e-20]], dtype=np.float32)
    x = _t(pts, device_gpu_only)
    want = np.array([2e-20, 4e-20, (2e-20**3.5 + 4e-20**3.5) ** (1 / 3.5)])
    assert _ulps(F.pdist(x, p=3.5).numpy(), want) < 8
    exact = lucid.cdist(x[:1], x, p=2.0, compute_mode="donot_use_mm_for_euclid_dist").numpy()
    assert exact[0, 1] == pytest.approx(2e-20, rel=1e-5, abs=0)
    assert exact[0, 2] == pytest.approx(4e-20, rel=1e-5, abs=0)
    d = lucid.cdist(x[:1], x, p=3.5).numpy()
    assert d[0, 1] == pytest.approx(2e-20, rel=1e-5, abs=0)
    assert d[0, 2] == pytest.approx(4e-20, rel=1e-5, abs=0)


def test_normalize_with_a_small_eps_keeps_a_tiny_row(device_gpu_only: str) -> None:
    arr = np.array([[3e-20, 4e-20], [3.0, 4.0]], dtype=np.float32)
    got = F.normalize(_t(arr, device_gpu_only), p=2.0, dim=1, eps=1e-30).numpy()
    np.testing.assert_allclose(got, [[0.6, 0.8], [0.6, 0.8]], rtol=1e-5)
    both = F.normalize(_t(arr, device_gpu_only), p=2.0, dim=None, eps=1e-30).numpy()
    assert np.all(np.isfinite(both))


@pytest.mark.parametrize("p", [1.0, 0.5, math.inf, -math.inf, 0.0])
def test_orders_that_power_nothing_are_untouched(device_gpu_only: str, p: float) -> None:
    arr = np.array([3.0, -4.0, 1.0, 2.0], dtype=np.float32)
    got = LA.vector_norm(_t(arr, device_gpu_only), p).item()
    cpu = LA.vector_norm(_t(arr, "cpu"), p).item()
    assert got == pytest.approx(cpu, rel=1e-6)


@pytest.mark.parametrize("p", [1.0, 0.5, 0.0, -2.0, math.inf, math.nan])
def test_the_engine_op_refuses_orders_that_are_not_finite_and_above_one(
    device_gpu_only: str, p: float
) -> None:
    x = _t(np.ones(4), device_gpu_only)
    with pytest.raises(ValueError):
        _C_engine.linalg.p_norm(x._impl, p, [0], False)


def test_the_engine_op_is_a_gpu_stream_kernel(device_cpu_only: str) -> None:
    x = _t(np.ones(4), device_cpu_only)
    with pytest.raises(NotImplementedError):
        _C_engine.linalg.p_norm(x._impl, 2.0, [0], False)


def test_an_empty_reduction_is_zero(device_gpu_only: str) -> None:
    assert LA.vector_norm(lucid.zeros((0, 3), device=device_gpu_only)).item() == 0.0
    assert LA.vector_norm(lucid.zeros((4, 0), device=device_gpu_only), dim=1).tolist() == [0.0] * 4
