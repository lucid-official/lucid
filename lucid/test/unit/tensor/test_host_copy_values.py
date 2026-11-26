"""Paths that read a tensor out of, or into, a host buffer return its values bit for bit (LCD-1012).

A Metal tensor now owns a buffer MLX allocates: ``upload_cpu_to_gpu`` copies into it and keeps
nothing of the host block, so ``to("metal")``, NumPy ingestion and ``from_bytes`` read their
source where it lies instead of staging a host copy.  ``to_bytes``, ``numpy()`` of a bfloat16
view, ``repr`` and the writes through a view dropped their full-size snapshots the same way.
Each path is held to NumPy, which reads the same strides independently through ``.numpy()``,
across the layouts a path has to read through (dense, transposed, sliced at an offset,
expanded, empty, 0-d) and every dtype the device holds.
"""

import gc
import pickle
from collections.abc import Callable

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import metal_available

_CPU_DTYPES = [
    np.float32,
    np.float64,
    np.float16,
    np.int64,
    np.int32,
    np.int16,
    np.int8,
    np.bool_,
    np.complex64,
    np.complex128,
]
# Metal holds neither float64 nor complex128.
_METAL_DTYPES = [d for d in _CPU_DTYPES if d not in (np.float64, np.complex128)]

_VIEWS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "dense": lambda x: x,
    "transposed": lambda x: x.transpose(0, x.ndim - 1),
    "permuted": lambda x: x.permute(*reversed(range(x.ndim))),
    "inner slice": lambda x: x[..., 1:-1],
    "steps": lambda x: x[::2, ..., ::3],
    "expanded": lambda x: x[:1].expand(4, *x.shape[1:]),
    "row run at an offset": lambda x: x[1:3],
    "unfolded": lambda x: x.reshape(-1).unfold(0, 4, 3),
}
_SHAPES = [(7, 9), (5, 6, 7)]

needs_metal = pytest.mark.skipif(not metal_available(), reason="Metal not available")


def _array(dtype: type, shape: tuple[int, ...]) -> np.ndarray:
    base = np.random.default_rng(0).standard_normal(shape) * 10
    if np.issubdtype(dtype, np.complexfloating):
        base = base + 1j * base[::-1]
    return base.astype(dtype)


def _c_bytes(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a).tobytes()


# ── to_bytes: one pass from the view's strides into the bytes object ──


@pytest.mark.parametrize("shape", _SHAPES, ids=["2-D", "3-D"])
@pytest.mark.parametrize("view", list(_VIEWS))
@pytest.mark.parametrize("dtype", _CPU_DTYPES, ids=lambda d: np.dtype(d).name)
def test_to_bytes_of_a_view_is_numpys_c_order_bytes(
    dtype: type, view: str, shape: tuple[int, ...]
) -> None:
    v = _VIEWS[view](lucid.tensor(_array(dtype, shape), device="cpu"))
    assert bytes(v._impl.to_bytes()) == _c_bytes(v.numpy())


@pytest.mark.parametrize("dtype", _CPU_DTYPES, ids=lambda d: np.dtype(d).name)
def test_to_bytes_of_an_empty_and_a_zero_d_tensor(dtype: type) -> None:
    empty = lucid.tensor(np.zeros((0, 3), dtype=dtype), device="cpu")
    assert bytes(empty._impl.to_bytes()) == b""
    assert bytes(empty.t()._impl.to_bytes()) == b""
    scalar = lucid.tensor(np.array(3, dtype=dtype), device="cpu")
    assert bytes(scalar._impl.to_bytes()) == np.array(3, dtype=dtype).tobytes()
    assert scalar.tolist() == np.array(3, dtype=dtype).tolist()


@pytest.mark.parametrize("dtype", _CPU_DTYPES, ids=lambda d: np.dtype(d).name)
def test_a_pickled_view_unpickles_to_its_own_values(dtype: type) -> None:
    v = lucid.tensor(_array(dtype, (6, 5)), device="cpu").t()[1:4]
    back = pickle.loads(pickle.dumps(v))
    assert back.shape == v.shape
    np.testing.assert_array_equal(back.numpy(), v.numpy())


# ── bfloat16 .numpy() widens straight from the strides ──


def _bf16_exact(shape: tuple[int, ...]) -> np.ndarray:
    """float32 values a bfloat16 holds exactly, so the widened result is the source."""
    return (np.random.default_rng(1).integers(-128, 128, size=shape) / 4).astype(np.float32)


@pytest.mark.parametrize("shape", _SHAPES, ids=["2-D", "3-D"])
@pytest.mark.parametrize("view", list(_VIEWS))
def test_bfloat16_numpy_of_a_view_matches_the_float32_view(
    view: str, shape: tuple[int, ...]
) -> None:
    src = _bf16_exact(shape)
    got = _VIEWS[view](lucid.tensor(src, device="cpu").to(lucid.bfloat16)).numpy()
    want = _VIEWS[view](lucid.tensor(src, device="cpu")).numpy()
    assert got.dtype == np.float32 and got.shape == want.shape
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("rows", [3, 300_000], ids=["fits one chunk", "many chunks"])
def test_bfloat16_numpy_across_the_chunk_boundary(rows: int) -> None:
    """The conversion packs 2 MiB at a time: a tensor spanning many chunks, and one row alone past it."""
    src = _bf16_exact((rows, 5))
    t = lucid.tensor(src, device="cpu").to(lucid.bfloat16)
    np.testing.assert_array_equal(t.numpy(), src)
    np.testing.assert_array_equal(t.t().numpy(), src.T)
    wide = _bf16_exact((2, 1_100_000))  # one row is more than a chunk
    np.testing.assert_array_equal(lucid.tensor(wide, device="cpu").to(lucid.bfloat16).numpy(), wide)


def test_bfloat16_numpy_of_an_empty_and_a_zero_d_tensor() -> None:
    empty = lucid.tensor(np.zeros((0, 4), dtype=np.float32), device="cpu").to(lucid.bfloat16)
    assert empty.numpy().shape == (0, 4)
    assert empty.t().numpy().shape == (4, 0)
    scalar = lucid.tensor(np.float32(2.5), device="cpu").to(lucid.bfloat16)
    got = scalar.numpy()
    assert got.shape == () and float(got) == 2.5


# ── moving to Metal reads the source where it lies ──


@needs_metal
@pytest.mark.parametrize("shape", _SHAPES, ids=["2-D", "3-D"])
@pytest.mark.parametrize("view", list(_VIEWS))
@pytest.mark.parametrize("dtype", _METAL_DTYPES, ids=lambda d: np.dtype(d).name)
def test_a_cpu_view_moved_to_metal_keeps_every_byte(
    dtype: type, view: str, shape: tuple[int, ...]
) -> None:
    src = _VIEWS[view](lucid.tensor(_array(dtype, shape), device="cpu"))
    want = _c_bytes(src.numpy())
    moved = src.to("metal")
    assert moved.device.type == "metal" and moved.shape == src.shape
    assert bytes(moved._impl.to_bytes()) == want
    assert bytes(moved.to("cpu")._impl.to_bytes()) == want


@needs_metal
@pytest.mark.parametrize("dtype", _METAL_DTYPES, ids=lambda d: np.dtype(d).name)
def test_an_empty_and_a_zero_d_tensor_move_to_metal(dtype: type) -> None:
    empty = lucid.tensor(np.zeros((0, 3), dtype=dtype), device="cpu").to("metal")
    assert empty.shape == (0, 3) and empty.device.type == "metal"
    assert bytes(empty.to("cpu")._impl.to_bytes()) == b""
    scalar = lucid.tensor(np.array(3, dtype=dtype), device="cpu").to("metal")
    assert scalar.shape == () and scalar.tolist() == np.array(3, dtype=dtype).tolist()


@needs_metal
def test_the_moved_tensor_owns_its_bytes() -> None:
    """The upload copies: a write to the source, or its death, must not reach the Metal tensor."""
    src = lucid.arange(1 << 16, dtype=lucid.float32, device="cpu")
    moved = src.to("metal")
    want = np.arange(1 << 16, dtype=np.float32)
    src.add_(1.0)
    src_window = src[10:20]
    del src
    gc.collect()
    np.testing.assert_array_equal(moved.numpy(), want)
    assert src_window.tolist() == [float(i + 1) for i in range(10, 20)]
    moved.add_(1.0)
    np.testing.assert_array_equal(moved.to("cpu").numpy(), want + 1)


@needs_metal
@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16, lucid.float32])
def test_a_cast_while_moving_keeps_the_values(dtype: lucid.dtype) -> None:
    """Values a half format holds exactly, through a strided source."""
    src = lucid.tensor(_bf16_exact((6, 5)), device="cpu").t()
    moved = src.to("metal", dtype)
    assert moved.dtype == dtype
    np.testing.assert_array_equal(moved.to("cpu", lucid.float32).numpy(), src.numpy())


# ── NumPy and bytes ingestion onto Metal ──


def _ingestion_arrays() -> dict[str, np.ndarray]:
    base = np.arange(2 * 6 * 7, dtype=np.float32).reshape(2, 6, 7) / 3
    readonly = base.copy()
    readonly.flags.writeable = False
    return {
        "c order": base,
        "fortran order": np.asfortranarray(base),
        "strided slice": base[:, ::2, 1:],
        "broadcast": np.broadcast_to(base[:1], (4, 6, 7)),
        "read only": readonly,
        "empty": np.zeros((0, 5), dtype=np.float32),
        "zero-d": np.array(2.5, dtype=np.float32),
    }


@needs_metal
@pytest.mark.parametrize("name", list(_ingestion_arrays()))
def test_numpy_ingestion_onto_metal_keeps_the_values(name: str) -> None:
    arr = _ingestion_arrays()[name]
    t = lucid.tensor(arr, device="metal")
    assert t.device.type == "metal" and t.shape == arr.shape
    assert bytes(t._impl.to_bytes()) == _c_bytes(arr)


@needs_metal
@pytest.mark.parametrize("dtype", _METAL_DTYPES, ids=lambda d: np.dtype(d).name)
def test_numpy_ingestion_onto_metal_by_dtype(dtype: type) -> None:
    arr = _array(dtype, (5, 4)).T
    t = lucid.tensor(arr, device="metal")
    assert bytes(t._impl.to_bytes()) == _c_bytes(arr)


@needs_metal
def test_the_ingested_array_is_neither_changed_nor_aliased() -> None:
    arr = np.arange(4096, dtype=np.float32)
    keep = arr.copy()
    t = lucid.tensor(arr, device="metal")
    arr += 1  # the Metal tensor owns its bytes: this must not reach it
    np.testing.assert_array_equal(t.numpy(), keep)
    t.add_(1.0)
    np.testing.assert_array_equal(arr, keep + 1)  # nor the other way
    del arr
    gc.collect()
    np.testing.assert_array_equal(t.to("cpu").numpy(), keep + 1)


@needs_metal
@pytest.mark.parametrize("dtype", _METAL_DTYPES, ids=lambda d: np.dtype(d).name)
def test_from_bytes_onto_metal_keeps_the_values_and_the_bytes_object(dtype: type) -> None:
    arr = _array(dtype, (6, 5))
    raw = arr.tobytes()
    engine_dtype = lucid.tensor(arr, device="cpu").dtype._engine
    impl = _C_engine.TensorImpl.from_bytes(
        raw, list(arr.shape), engine_dtype, _C_engine.Device.GPU, False
    )
    assert impl.device == _C_engine.Device.GPU
    assert bytes(impl.to_bytes()) == raw
    assert raw == arr.tobytes()


@needs_metal
@pytest.mark.parametrize("dtype", _METAL_DTYPES, ids=lambda d: np.dtype(d).name)
def test_a_metal_tensor_survives_a_pickle_round_trip(dtype: type) -> None:
    t = lucid.tensor(_array(dtype, (5, 6)), device="metal").t()
    back = pickle.loads(pickle.dumps(t))
    assert back.device.type == "metal"
    assert bytes(back._impl.to_bytes()) == bytes(t._impl.to_bytes())


@needs_metal
def test_a_zero_byte_bytes_upload_makes_an_empty_metal_tensor() -> None:
    impl = _C_engine.TensorImpl.from_bytes(
        b"", [0, 4], lucid.float32._engine, _C_engine.Device.GPU, False
    )
    assert list(impl.shape) == [0, 4] and impl.device == _C_engine.Device.GPU


# ── repr reads a row-major tensor where it lies and packs a strided one ──


def _same_text(view: lucid.Tensor) -> None:
    assert repr(view) == repr(view.contiguous())
    assert str(view) == str(view.contiguous())


@pytest.mark.parametrize("view", list(_VIEWS))
@pytest.mark.parametrize("dtype", _CPU_DTYPES, ids=lambda d: np.dtype(d).name)
def test_repr_of_a_view_is_repr_of_its_contiguous_copy(dtype: type, view: str) -> None:
    _same_text(_VIEWS[view](lucid.tensor(_array(dtype, (5, 6, 7)), device="cpu")))


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
def test_repr_of_a_half_tensor_and_its_view(dtype: lucid.dtype) -> None:
    x = lucid.tensor(_bf16_exact((9, 11)), device="cpu").to(dtype)
    _same_text(x)
    _same_text(x.t())
    _same_text(x[2:7, 1:9])


def test_repr_of_a_large_tensor_truncates_without_reading_past_it() -> None:
    x = lucid.arange(5000, dtype=lucid.float32, device="cpu").reshape(50, 100)
    text = repr(x)
    assert "..." in text and "4999." in text
    assert repr(x[:, 10:90]) == repr(x[:, 10:90].contiguous())


def test_repr_of_an_empty_and_a_zero_d_tensor() -> None:
    assert "[]" in repr(lucid.zeros(0, device="cpu"))
    _same_text(lucid.zeros(0, 3, device="cpu").t())
    assert repr(lucid.tensor(2.5, device="cpu")) == "tensor(2.5)"


@needs_metal
@pytest.mark.parametrize("dtype", _METAL_DTYPES, ids=lambda d: np.dtype(d).name)
def test_repr_of_a_metal_tensor_is_the_cpu_tensors_up_to_the_device(dtype: type) -> None:
    arr = _array(dtype, (6, 7))
    on_cpu = repr(lucid.tensor(arr, device="cpu").t())
    on_metal = repr(lucid.tensor(arr, device="metal").t())
    assert on_metal.replace(", device='metal'", "") == on_cpu


# ── writes through a view pack straight into a dense destination ──


def test_assigning_a_view_into_a_tensor_with_live_views_lands_on_every_view() -> None:
    src = np.arange(20, dtype=np.float32).reshape(4, 5)
    x = lucid.zeros(5, 4, device="cpu")
    window = x[1:3]
    x[:] = lucid.tensor(src, device="cpu").t()
    np.testing.assert_array_equal(x.numpy(), src.T)
    np.testing.assert_array_equal(window.numpy(), src.T[1:3])


def test_assigning_a_view_without_live_views_lands_on_the_tensor() -> None:
    src = np.arange(20, dtype=np.float32).reshape(4, 5)
    x = lucid.zeros(5, 4, device="cpu")
    x[:] = lucid.tensor(src, device="cpu").t()
    np.testing.assert_array_equal(x.numpy(), src.T)
    x.copy_(lucid.tensor(src + 1, device="cpu").t())
    np.testing.assert_array_equal(x.numpy(), (src + 1).T)


@pytest.mark.parametrize("live_view", [False, True], ids=["no live view", "live view"])
def test_a_tensor_assigned_its_own_transpose_reads_the_old_values(live_view: bool) -> None:
    """The source is a view of the destination: packing straight into it would read what it wrote."""
    ref = np.arange(25, dtype=np.float32).reshape(5, 5)
    x = lucid.tensor(ref.copy(), device="cpu")
    window = x[1:3] if live_view else None
    x[:] = x.t()
    np.testing.assert_array_equal(x.numpy(), ref.T)
    if window is not None:
        np.testing.assert_array_equal(window.numpy(), ref.T[1:3])
    y = lucid.tensor(ref.copy(), device="cpu")
    y.copy_(y.t())
    np.testing.assert_array_equal(y.numpy(), ref.T)


@pytest.mark.parametrize("dtype", [np.float32, np.int16, np.bool_, np.complex64, np.float64])
def test_assigning_a_view_keeps_every_dtype(dtype: type) -> None:
    src = _array(dtype, (4, 6))
    x = lucid.zeros(6, 4, device="cpu", dtype=lucid.tensor(src, device="cpu").dtype)
    x.copy_(lucid.tensor(src, device="cpu").t())
    np.testing.assert_array_equal(x.numpy(), src.T)


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_copy_handles_overlap_broadcast_strided_dst_and_source_isolation(device: str) -> None:
    if device == "metal" and not lucid.metal.is_available():
        pytest.skip("Metal not available")
    sq = np.arange(25, dtype=np.float32).reshape(5, 5)
    # Self-overlap with live view: engine buffers_overlap snapshots safely without Python contiguous.
    x = lucid.tensor(sq.copy(), device=device)
    view = x[1:3]
    x.copy_(x.t())
    np.testing.assert_array_equal(x.numpy(), sq.T)
    np.testing.assert_array_equal(view.numpy(), sq.T[1:3])

    # Strided destination <- strided source.
    rect = np.arange(20, dtype=np.float32).reshape(4, 5)
    dst = lucid.zeros(4, 5, device=device)
    dst.t().copy_(lucid.tensor(rect, device=device).t())
    np.testing.assert_array_equal(dst.numpy(), rect)

    # Broadcast source (broadcast_to allocates once; copy_ adds no second copy).
    row = lucid.arange(5, dtype=lucid.float32, device=device)
    b = lucid.zeros(4, 5, device=device)
    b.copy_(row)
    np.testing.assert_array_equal(
        b.numpy(), np.broadcast_to(np.arange(5, dtype=np.float32), (4, 5))
    )

    # Overlapping strided dst <- dense src (must snapshot before scattering).
    sq3 = np.arange(9, dtype=np.float32).reshape(3, 3)
    m = lucid.tensor(sq3.copy(), device=device)
    m.t().copy_(m)
    np.testing.assert_array_equal(m.numpy(), sq3.T)

    vec = np.arange(10, dtype=np.float32)
    v1 = lucid.tensor(vec.copy(), device=device)
    v1[::2].copy_(v1[:5])
    expected_v1 = vec.copy()
    expected_v1[::2] = vec[:5]
    np.testing.assert_array_equal(v1.numpy(), expected_v1)

    v2 = lucid.tensor(vec.copy(), device=device)
    v2.reshape(2, 5).t().copy_(v2.reshape(5, 2))
    expected_v2 = vec.copy().reshape(2, 5)
    expected_v2.T[:] = vec.reshape(5, 2)
    np.testing.assert_array_equal(v2.numpy(), expected_v2.reshape(-1))

    # Overlapping dense block views & identical self-copy (memmove, no memcpy-param-overlap UB).
    fwd = lucid.tensor(vec.copy(), device=device)
    fwd[1:].copy_(fwd[:-1])
    expected_fwd = vec.copy()
    expected_fwd[1:] = vec[:-1]
    np.testing.assert_array_equal(fwd.numpy(), expected_fwd)

    bwd = lucid.tensor(vec.copy(), device=device)
    bwd[:-1].copy_(bwd[1:])
    expected_bwd = vec.copy()
    expected_bwd[:-1] = vec[1:]
    np.testing.assert_array_equal(bwd.numpy(), expected_bwd)

    same = lucid.tensor(vec.copy(), device=device)
    same.copy_(same)
    np.testing.assert_array_equal(same.numpy(), vec)

    # Mutating source after copy_ must not affect destination.
    src = lucid.tensor(rect.copy(), device=device)
    out = lucid.zeros(5, 4, device=device)
    out.copy_(src.t())
    src.fill_(99.0)
    np.testing.assert_array_equal(out.numpy(), rect.T)


@needs_metal
def test_copy_from_metal_shared_tensor_isolates_aliased_destination_and_self_transpose() -> None:
    arr = np.arange(20, dtype=np.float32).reshape(4, 5)
    s_cpu = lucid.metal.shared_tensor((4, 5), dtype=lucid.float32)
    s_cpu.copy_(lucid.tensor(arr, device="cpu"))
    s_gpu = s_cpu.to("metal")

    for src in (s_gpu, s_gpu[:]):
        dst = lucid.zeros(4, 5, device="metal")
        view = dst[0:2]
        dst.copy_(src)
        s_cpu.fill_(99.0)
        np.testing.assert_array_equal(dst.numpy(), arr)
        np.testing.assert_array_equal(view.numpy(), arr[0:2])
        s_cpu.copy_(lucid.tensor(arr, device="cpu"))

    # Self-transpose of a shared CPU tensor must snapshot before scattering.
    sq = np.arange(16, dtype=np.float32).reshape(4, 4)
    sh_sq = lucid.metal.shared_tensor((4, 4), dtype=lucid.float32)
    sh_sq.copy_(lucid.tensor(sq, device="cpu"))
    sh_sq.t().copy_(sh_sq)
    np.testing.assert_array_equal(sh_sq.numpy(), sq.T)


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_copy_preserves_autograd_for_leaf_and_strided_sources(device: str) -> None:
    if device == "metal" and not lucid.metal.is_available():
        pytest.skip("Metal not available")
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)

    leaf = lucid.tensor(arr, device=device, requires_grad=True)
    dst = lucid.zeros(3, 4, device=device)
    dst.copy_(leaf)
    assert dst.requires_grad
    dst.sum().backward()
    assert leaf.grad is not None
    np.testing.assert_array_equal(leaf.grad.numpy(), np.ones((3, 4), dtype=np.float32))

    leaf2 = lucid.tensor(arr, device=device, requires_grad=True)
    dst2 = lucid.zeros(4, 3, device=device)
    dst2.copy_(leaf2.t())
    assert dst2.requires_grad
    dst2.sum().backward()
    assert leaf2.grad is not None
    np.testing.assert_array_equal(leaf2.grad.numpy(), np.ones((3, 4), dtype=np.float32))

    # Non-leaf source isolation: hooks, retain_grad, and autograd.grad must not alias y's node.
    x = lucid.tensor(arr, device=device, requires_grad=True)
    y = x * 2.0
    other = lucid.zeros(3, 4, device=device) * 1.0
    other.copy_(y)
    other.register_hook(lambda g: g * 10.0)
    y.sum().backward()
    assert x.grad is not None
    np.testing.assert_array_equal(x.grad.numpy(), np.full((3, 4), 2.0, dtype=np.float32))

    x2 = lucid.tensor(arr, device=device, requires_grad=True)
    y2 = x2 * 2.0
    y2.retain_grad()
    other2 = lucid.zeros(3, 4, device=device) * 1.0
    other2.copy_(y2)
    other2.retain_grad()
    assert lucid.autograd.grad(y2.sum(), other2, allow_unused=True, retain_graph=True) == (None,)
    other2.sum().backward()
    assert y2.grad is not None and other2.grad is not None
    np.testing.assert_array_equal(y2.grad.numpy(), np.ones((3, 4), dtype=np.float32))
    np.testing.assert_array_equal(other2.grad.numpy(), np.ones((3, 4), dtype=np.float32))

    # Copying a grad-requiring source under no_grad detaches without tracking grad.
    ng_dst = lucid.zeros(3, 4, device=device)
    with lucid.no_grad():
        ng_dst.copy_(y2)
    assert not ng_dst.requires_grad
    np.testing.assert_array_equal(ng_dst.numpy(), arr * 2.0)
