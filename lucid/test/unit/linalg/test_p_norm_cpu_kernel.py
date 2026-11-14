"""LCD-1003: the CPU stream's range-safe entrywise p-norm (``cpu::PNorm``).

``vector_norm`` of a float CPU tensor at ``p > 2`` was the plain graph
``(sum |x|^p)^(1/p)`` behind a window on the forward value; its second derivative left
float32 for a slice near 1e-9 or 1e8 at ``p = 3``.  The CPU stream now runs the same
``p_norm`` op as Metal — ``CpuBackend::p_norm_forward`` for the value, ``PNormBackward``
(ratios ``|x| / N`` in [0, 1]) for every derivative — except for ``p = 2``, whose plain graph
has no such window.  The derivatives' own tests are in
``unit/autograd/test_p_norm_derivative_range.py``; this file covers the kernel: its value over
every layout, order and dtype, the special values, and which calls it serves.

The reference is plain Python float64, relative to each slice's largest entry.
"""

import itertools
import math
import random

import pytest

import lucid
import lucid.linalg
from lucid._C import engine as _C_engine
from lucid.autograd import grad
from lucid._dispatch import _unwrap, _wrap
from lucid._ops import _reduce
from lucid.compile import _tracing
from lucid.linalg import _norm_scale

#: Orders taking each kernel path: 3/2, 2, 3, 4 (multiplications), 6 (the float limit of the
#: plain double sum), 2.7 (vForce blocks), 7.5 (rescaled, a half-integer order past the limit).
ORDERS = [1.5, 2.0, 2.7, 3.0, 4.0, 6.0, 7.5]


def _slices(shape: tuple[int, ...], axes: list[int]) -> list[list[int]]:
    """The flat indices of each slice of a row-major tensor reduced over ``axes``."""
    strides = [math.prod(shape[d + 1 :]) for d in range(len(shape))]
    groups: dict[tuple[int, ...], list[int]] = {}
    for index in itertools.product(*(range(extent) for extent in shape)):
        kept = tuple(i for d, i in enumerate(index) if d not in axes)
        groups.setdefault(kept, []).append(sum(i * s for i, s in zip(index, strides, strict=True)))
    return list(groups.values())


def _norm(entries: list[float], p: float) -> float:
    top = max(abs(v) for v in entries)
    if top == 0.0 or top != top or math.isinf(top):
        return top
    return top * sum((abs(v) / top) ** p for v in entries) ** (1.0 / p)


def _reference(flat: list[float], shape: tuple[int, ...], axes: list[int], p: float) -> list[float]:
    return [_norm([flat[i] for i in members], p) for members in _slices(shape, axes)]


def _values(count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    out = [round(rng.uniform(-3.0, 3.0), 3) for _ in range(count)]
    out[count // 2] = 0.0
    return out


def _kernel(x: lucid.Tensor, p: float, axes: list[int], keepdim: bool) -> lucid.Tensor:
    return _wrap(_C_engine.linalg.p_norm(_unwrap(x), p, axes, keepdim))


LAYOUTS = [
    ((7,), [0]),
    ((5, 9), [1]),
    ((5, 9), [0]),
    ((5, 9), [0, 1]),
    ((4, 6, 5), [0, 2]),
    ((4, 6, 5), [1]),
    ((4, 6, 5), [0, 1]),
    ((4, 6, 5), [1, 2]),
    ((3, 4, 5, 6), [0, 2]),
    ((3, 4, 5, 6), [1, 3]),
    ((3, 4, 5, 6), [0, 3]),
    ((3, 4, 5, 6), [0, 1, 2, 3]),
    ((2, 1, 3), [1]),
    ((1, 5), [0]),
    ((600, 3), [0]),
    ((3, 1300), [1]),
    # a wide inner extent is swept in tiles of 256 columns: one tile, several, a remainder
    ((9, 40), [0]),
    ((37, 700), [0]),
    ((3, 50, 20), [1]),
    ((2, 5, 9, 2), [1, 2]),
]


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(("shape", "axes"), LAYOUTS)
@pytest.mark.parametrize("p", ORDERS)
@pytest.mark.parametrize(("dtype", "tol"), [(lucid.float32, 3e-6), (lucid.float64, 1e-12)])
def test_the_kernel_matches_the_float64_norm_over_every_layout(
    device_cpu_only: str,
    dtype: lucid.dtype,
    tol: float,
    p: float,
    shape: tuple[int, ...],
    axes: list[int],
    keepdim: bool,
) -> None:
    x = lucid.tensor(_values(math.prod(shape), len(shape)), dtype=dtype).reshape(*shape)
    got = _kernel(x, p, axes, keepdim)
    want_shape = tuple(
        1 if d in axes else s for d, s in enumerate(shape) if keepdim or d not in axes
    )
    assert tuple(got.shape) == want_shape
    want = _reference(x.reshape(-1).tolist(), shape, axes, p)
    for g, w in zip(got.reshape(-1).tolist(), want, strict=True):
        assert abs(g - w) <= tol * w


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("length", [16384, 20000, 32768, 40000, 70001, 150001])
@pytest.mark.parametrize("p", ORDERS)
def test_a_long_slice_is_summed_in_blocks_with_the_same_value(
    device_cpu_only: str, p: float, length: int, rows: int
) -> None:
    """A slice is summed over blocks of 16384 entries: one block, two (serial) and past two
    blocks (32768), where a few slices are split across the cores, are the same value."""
    flat = _values(rows * length, length)
    x = lucid.tensor(flat).reshape(rows, length)
    got = _kernel(x, p, [1], False).tolist()
    for row, g in enumerate(got):
        want = _norm(flat[row * length : (row + 1) * length], p)
        assert g == pytest.approx(want, rel=3e-6, abs=0)


@pytest.mark.parametrize("p", [3.0, 2.7, 7.5])
def test_a_long_slice_with_a_nan_or_an_infinity_answers_as_itself(
    device_cpu_only: str, p: float
) -> None:
    for value, check in ((math.nan, math.isnan), (math.inf, math.isinf)):
        flat = _values(100000, 4)
        flat[99999] = value
        got = _kernel(lucid.tensor(flat).reshape(1, 100000), p, [1], False).item()
        assert check(got)


@pytest.mark.parametrize("p", ORDERS)
def test_an_empty_axes_list_reduces_every_axis(device_cpu_only: str, p: float) -> None:
    flat = _values(24, 3)
    x = lucid.tensor(flat).reshape(2, 3, 4)
    got = _kernel(x, p, [], False)
    assert got.shape == ()
    assert got.item() == pytest.approx(_norm(x.reshape(-1).tolist(), p), rel=3e-6, abs=0)


@pytest.mark.parametrize("p", ORDERS)
def test_a_zero_dim_input_is_its_magnitude(device_cpu_only: str, p: float) -> None:
    got = _kernel(lucid.tensor(-2.5), p, [], False)
    assert got.shape == ()
    assert got.item() == pytest.approx(2.5, rel=1e-6, abs=0)


@pytest.mark.parametrize(
    ("shape", "axes", "out_shape"),
    [
        ((0, 3), [0], (3,)),
        ((3, 0), [1], (3,)),
        ((0, 3), [1], (0,)),
        ((2, 0, 3), [1], (2, 3)),
        # a wide inner extent (swept in column tiles) with nothing outer, nothing to reduce, nothing inner
        ((0, 4, 16), [1], (0, 16)),
        ((0, 4, 16), [0], (4, 16)),
        ((3, 0, 16), [1], (3, 16)),
        ((3, 4, 0), [1], (3, 0)),
    ],
)
def test_an_empty_reduction_is_zero(
    device_cpu_only: str, shape: tuple[int, ...], axes: list[int], out_shape: tuple[int, ...]
) -> None:
    got = _kernel(lucid.zeros(*shape), 3.0, axes, False)
    assert tuple(got.shape) == out_shape
    assert all(v == 0.0 for v in got.reshape(-1).tolist())


#: Float32 deep in the subnormals (the result still has 5 digits) and near its largest finite,
#: float64 likewise.
EXTREME_F32 = [1e-40, 1e-30, 1e-20, 1e20, 1e30, 1e37]
EXTREME_F64 = [1e-300, 1e-200, 1e200, 1e300]


@pytest.mark.parametrize("p", [1.5, 3.0, 4.0, 2.7, 6.0, 7.5])
@pytest.mark.parametrize("magnitude", EXTREME_F32)
def test_a_float32_slice_at_the_edge_of_the_range_keeps_its_norm(
    device_cpu_only: str, magnitude: float, p: float
) -> None:
    x = lucid.tensor([m * magnitude for m in (1.0, -2.0, 3.0, 0.5)])
    got = lucid.linalg.vector_norm(x, p).item()
    assert got == pytest.approx(_norm(x.tolist(), p), rel=1e-5, abs=0)


@pytest.mark.parametrize("p", [1.5, 3.0, 2.7, 7.5])
@pytest.mark.parametrize("magnitude", EXTREME_F64)
def test_a_float64_slice_at_the_edge_of_the_range_keeps_its_norm(
    device_cpu_only: str, magnitude: float, p: float
) -> None:
    x = lucid.tensor([m * magnitude for m in (1.0, -2.0, 3.0, 0.5)], dtype=lucid.float64)
    assert lucid.linalg.vector_norm(x, p).item() == pytest.approx(
        _norm(x.tolist(), p), rel=1e-12, abs=0
    )


@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float64])
@pytest.mark.parametrize("p", [1.5, 3.0, 2.7, 7.5])
def test_a_nan_an_infinity_and_a_zero_slice_answer_as_themselves(
    device_cpu_only: str, dtype: lucid.dtype, p: float
) -> None:
    nan, inf = math.nan, math.inf
    rows = [[0.0, 0.0, 0.0], [1.0, nan, 2.0], [1.0, inf, 2.0], [nan, inf, 1.0], [-inf, 1.0, 1.0]]
    got = lucid.linalg.vector_norm(lucid.tensor(rows, dtype=dtype), p, dim=1).tolist()
    assert got[0] == 0.0
    assert math.isnan(got[1]) and math.isnan(got[3])
    assert got[2] == inf and got[4] == inf


@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float64])
@pytest.mark.parametrize("p", [1.5, 3.0, 2.7, 7.5])
def test_the_columns_of_a_matrix_answer_as_themselves(
    device_cpu_only: str, dtype: lucid.dtype, p: float
) -> None:
    """A leading-axis reduction sweeps rows: a zero, NaN, infinite and ordinary column side by side."""
    nan, inf = math.nan, math.inf
    cols = [[0.0] * 4, [1.0, nan, 2.0, 3.0], [1.0, -inf, 2.0, 3.0], [nan, inf, 1.0, 1.0]]
    cols += [[0.5, -1.5, 2.5, 0.25]] * 6 + [[1e-3, 2e-3, -3e-3, 4e-3]] * 6
    x = lucid.tensor([list(row) for row in zip(*cols, strict=True)], dtype=dtype)
    got = lucid.linalg.vector_norm(x, p, dim=0).tolist()
    assert got[0] == 0.0 and math.isnan(got[1]) and got[2] == inf and math.isnan(got[3])
    for column, g in zip(cols[4:], got[4:], strict=True):
        assert g == pytest.approx(_norm(column, p), rel=3e-6, abs=0)


def test_a_float64_column_with_a_subnormal_top_divides(device_cpu_only: str) -> None:
    """Its reciprocal overflows, so the tile hands that column to the slice path."""
    tiny = 4e-320
    x = lucid.tensor(
        [[tiny, 1.0] * 6, [-2 * tiny, 2.0] * 6, [3 * tiny, 0.5] * 6], dtype=lucid.float64
    )
    got = lucid.linalg.vector_norm(x, 7.5, dim=0).tolist()
    assert got[0] == pytest.approx(_norm(x[:, 0].tolist(), 7.5), rel=1e-6, abs=0)
    assert got[1] == pytest.approx(_norm(x[:, 1].tolist(), 7.5), rel=1e-12, abs=0)


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
@pytest.mark.parametrize("p", [1.5, 3.0, 2.7])
def test_a_half_input_reduces_in_float32_and_answers_in_its_own_dtype(
    device_cpu_only: str, dtype: lucid.dtype, p: float
) -> None:
    x = lucid.tensor(_values(60, 5), dtype=dtype).reshape(6, 10)
    got = lucid.linalg.vector_norm(x, p, dim=1)
    assert got.dtype == dtype
    want = lucid.linalg.vector_norm(x.to(lucid.float32), p, dim=1).to(dtype)
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
def test_the_engine_kernel_takes_float_and_double_only(
    device_cpu_only: str, dtype: lucid.dtype
) -> None:
    """A half input reaches it through ``_scaled_p_norm``, which reduces in float32."""
    with pytest.raises(NotImplementedError):
        _kernel(lucid.ones(4, dtype=dtype), 3.0, [0], False)


# ── which calls the kernel serves ──────────────────────────────────────────────


def _impl(dtype: lucid.dtype = lucid.float32, device: str = "cpu", n: int = 6) -> object:
    return _unwrap(lucid.ones(n, dtype=dtype, device=device))


@pytest.mark.parametrize(
    ("p", "dtype", "serves"),
    [
        (3.0, lucid.float32, True),
        (3.0, lucid.float64, True),
        (1.5, lucid.float32, True),
        (2.0, lucid.float32, False),  # the plain graph's root has no window problem
        (1.0, lucid.float32, False),
        (0.5, lucid.float32, False),
        (math.inf, lucid.float32, False),
        (3.0, lucid.float16, False),  # reduces in float32 through ``_scaled_p_norm``
        (3.0, lucid.bfloat16, False),
    ],
)
def test_the_cpu_stream_serves_the_orders_that_power_the_entries(
    device_cpu_only: str, p: float, dtype: lucid.dtype, serves: bool
) -> None:
    assert _reduce._takes_scaled_kernel(_impl(dtype), p) is serves


def test_a_scalar_takes_the_kernel_on_the_cpu(device_cpu_only: str) -> None:
    """It is its own magnitude; the plain graph's second derivative of it was ``-inf`` at 1e-10."""
    assert _reduce._takes_scaled_kernel(_unwrap(lucid.tensor(1.5)), 3.0) is True


def test_an_empty_tensor_takes_the_plain_graph(device_cpu_only: str) -> None:
    assert _reduce._takes_scaled_kernel(_unwrap(lucid.zeros(0, 3)), 3.0) is False


def test_metal_keeps_its_own_dtypes_and_serves_p_two(device_gpu_only: str) -> None:
    assert _reduce._takes_scaled_kernel(_impl(device=device_gpu_only), 2.0) is True
    assert _reduce._takes_scaled_kernel(_impl(lucid.float16, device_gpu_only), 3.0) is False


def test_a_cpu_compile_trace_records_the_plain_graph(device_cpu_only: str) -> None:
    """Core ML has no emitter for the ``p_norm`` node, and a trace of a CPU tensor is exportable."""
    x = lucid.tensor(_values(12, 1)).reshape(3, 4)
    with _tracing(watch=((x,), {})) as tracer:
        assert _reduce._takes_scaled_kernel(_unwrap(x), 3.0) is False
        lucid.linalg.vector_norm(x, 3.0, dim=1)
    names = [op.name for op in tracer.graph.ops]
    assert "p_norm" not in names
    assert "pow_scalar" in names


def test_a_metal_compile_trace_keeps_a_scalar_on_the_plain_graph(device_gpu_only: str) -> None:
    """The compiled graph's ``p_norm`` emitter has no 0-d variant: a scalar in a trace would
    fall back to eager with a warning, so it records the plain ops."""
    scalar = lucid.tensor(1.5, device=device_gpu_only)
    with _tracing(watch=((scalar,), {})) as tracer:
        lucid.linalg.vector_norm(scalar, 3.0)
    assert "p_norm" not in [op.name for op in tracer.graph.ops]
    matrix = lucid.tensor([[3.0, -4.0], [1.0, 2.0]], device=device_gpu_only)
    for dim in (None, (), []):
        with _tracing(watch=((matrix,), {})) as tracer:
            lucid.linalg.vector_norm(matrix, 3.0, dim=dim)
        assert "p_norm" in [op.name for op in tracer.graph.ops], dim


# ── the kernel's value against the plain graph it replaced ─────────────────────


@pytest.mark.parametrize("p", [1.5, 2.7, 3.0, 4.0, 6.0])
def test_the_kernel_agrees_with_the_plain_graph_at_moderate_magnitudes(
    device_cpu_only: str, p: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    flat = _values(40, 11)

    def value_and_gradient() -> tuple[list[float], list[float]]:
        x = lucid.tensor(flat, requires_grad=True).reshape(4, 10)
        n = lucid.linalg.vector_norm(x, p, dim=1)
        (g,) = grad(n.sum(), [x])
        return n.tolist(), g.reshape(-1).tolist()

    kernel_n, kernel_g = value_and_gradient()
    never = lambda r, p: False  # noqa: E731 - the plain graph, as before the kernel
    monkeypatch.setattr(_reduce, "_takes_scaled_kernel", never)
    monkeypatch.setattr(_norm_scale, "_takes_scaled_kernel", never)
    plain_n, plain_g = value_and_gradient()
    for a, b in zip(kernel_n, plain_n, strict=True):
        assert a == pytest.approx(b, rel=3e-6, abs=0)
    scale = max(abs(v) for v in plain_g)
    for a, b in zip(kernel_g, plain_g, strict=True):
        assert abs(a - b) <= 2e-6 * scale


@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float64])
@pytest.mark.parametrize("p", [1.5, 2.7, 3.0, 7.5])
@pytest.mark.parametrize(
    ("shape", "axes"),
    [
        ((5, 9), [1]),
        ((4, 6, 5), [0, 2]),  # gathered to put the reduced axes last
        ((37, 700), [0]),  # tiles of columns
        ((3, 50, 20), [1]),
        ((3, 40000), [1]),  # past two blocks of 16384: split across the cores
        ((150001,), [0]),
    ],
)
def test_the_kernel_returns_the_same_bits_every_time(
    device_cpu_only: str,
    dtype: lucid.dtype,
    p: float,
    shape: tuple[int, ...],
    axes: list[int],
) -> None:
    """Blocks are added in order and a slice or a tile is one chunk's, so the core a chunk
    lands on does not change the total."""
    x = lucid.tensor(_values(math.prod(shape), 17), dtype=dtype).reshape(*shape)
    first = _kernel(x, p, axes, False).reshape(-1).tolist()
    for _ in range(3):
        assert _kernel(x, p, axes, False).reshape(-1).tolist() == first
