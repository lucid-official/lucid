"""Zero-extent and 0-d layout ops are free of undefined behaviour (LCD-1119).

``CpuBackend::pad`` computed ``rows = in_numel / shape.back()`` unconditionally,
which divides by zero whenever the trailing extent is zero (``F.pad(zeros(2, 0),
(1, 1))``).  On arm64 an integer divide by zero returns 0, so release builds
silently masked it while the nightly UBSan native job halted.  A 0-d input also
indexed ``shape.back()`` and allocated ``ndim - 1`` coordinates, segfaulting on
CPU and failing in Metal's copy kernel.

Sibling CPU layout kernels in ``backend/cpu/methods/Layout.cpp`` and
``DtypeDoors.h::fill_pad_buffer`` shared the same defect class: calling
``std::memset`` / ``std::memcpy`` on an empty tensor's null buffer or dividing
by an empty extent.

This file sits in ``lucid/test/unit/ops/`` so the nightly ASan + UBSan native
gate runs every layout op across shapes that place a zero extent in every axis
position, plus 0-d ``F.pad``, on CPU and Metal.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.test._default_device import metal_available

Shape = tuple[int, ...]

#: Shapes placing a zero extent at every axis position across ranks 1, 2 and 3.
_EMPTY_SHAPES: tuple[Shape, ...] = ((0,), (0, 3), (3, 0), (0, 0), (0, 2, 3), (2, 0, 3), (2, 3, 0))

_DEVICES: tuple[str, ...] = ("cpu", "metal") if metal_available() else ("cpu",)

_PAD_DTYPES: tuple[lucid.dtype, ...] = (
    lucid.float32,
    lucid.float64,
    lucid.float16,
    lucid.bfloat16,
    lucid.int32,
    lucid.int64,
    lucid.bool,
)


def _empty(shape: Shape, dtype: lucid.dtype, device: str, *, grad: bool = False) -> Tensor:
    t = lucid.zeros(shape, dtype=dtype, device=device)
    if grad and dtype in (lucid.float32, lucid.float64, lucid.float16, lucid.bfloat16):
        t = t.requires_grad_(True)
    return t


def test_layout_sweep_covers_every_empty_position() -> None:
    """Guard on the guard: every axis position in ranks 1-3 has a zero-extent shape."""
    by_rank: dict[int, set[int]] = {1: set(), 2: set(), 3: set()}
    for sh in _EMPTY_SHAPES:
        for axis, extent in enumerate(sh):
            if extent == 0:
                by_rank[len(sh)].add(axis)
    assert by_rank == {1: {0}, 2: {0, 1}, 3: {0, 1, 2}}


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _PAD_DTYPES, ids=str)
@pytest.mark.parametrize("value", [0.0, 3.0], ids=["zero", "nonzero"])
@pytest.mark.parametrize("shape", _EMPTY_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_pad_empty_input_fills_every_position(
    device: str, dtype: lucid.dtype, value: float, shape: Shape
) -> None:
    """Constant pad of an empty tensor fills non-empty outputs and leaves empty ones empty."""
    if device == "metal" and dtype == lucid.float64:
        pytest.skip("Metal has no float64")
    x = _empty(shape, dtype, device, grad=True)
    # Pad the last axis by (1, 1); when the last axis was 0, output becomes non-empty
    # unless an earlier axis is also 0.
    y = F.pad(x, (1, 1), mode="constant", value=value)
    expected_shape = (*shape[:-1], shape[-1] + 2)
    assert tuple(y.shape) == expected_shape
    assert y.dtype == dtype
    if y.numel() > 0:
        expected_fill = (value != 0.0) if dtype == lucid.bool else value
        assert y.reshape(-1).tolist() == [expected_fill] * y.numel()
    if x.requires_grad:
        y.sum().backward()
        assert x.grad is not None
        assert tuple(x.grad.shape) == shape


@pytest.mark.parametrize("device", _DEVICES)
def test_pad_empty_leading_axis_becomes_filled(device: str) -> None:
    """Padding a leading zero axis produces a constant-filled tensor and clean backward."""
    x = _empty((0, 3), lucid.float32, device, grad=True)
    y = F.pad(x, (0, 0, 1, 2), mode="constant", value=4.5)
    assert tuple(y.shape) == (3, 3)
    assert y.reshape(-1).tolist() == [4.5] * 9
    y.sum().backward()
    assert x.grad is not None and tuple(x.grad.shape) == (0, 3)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [lucid.float32, lucid.int32, lucid.bool], ids=str)
def test_pad_scalar_empty_spec_returns_copy(device: str, dtype: lucid.dtype) -> None:
    """0-d tensor with empty pad spec returns an equal scalar without crashing."""
    val = True if dtype == lucid.bool else (5 if dtype == lucid.int32 else 5.0)
    x = lucid.tensor(val, dtype=dtype, device=device)
    if dtype == lucid.float32:
        x = x.requires_grad_(True)
    y = F.pad(x, (), mode="constant", value=3.0)
    assert tuple(y.shape) == ()
    assert y.dtype == dtype
    assert y.item() == val
    if x.requires_grad:
        y.backward()
        assert x.grad is not None and x.grad.item() == 1.0


#: Sibling layout ops that must handle zero extents at every axis without UB.
_LAYOUT_OPS: dict[str, Callable[[Tensor], Tensor | tuple[Tensor, ...]]] = {
    "cat_empty": lambda x: lucid.cat([x, x], dim=0),
    "cat_last": lambda x: lucid.cat([x, x], dim=-1),
    "stack_0": lambda x: lucid.stack([x, x], dim=0),
    "stack_last": lambda x: lucid.stack([x, x], dim=-1),
    "flip_0": lambda x: lucid.flip(x, (0,)),
    "flip_last": lambda x: lucid.flip(x, (-1,)),
    "roll_0": lambda x: lucid.roll(x, 1, 0),
    "roll_last": lambda x: lucid.roll(x, 1, -1),
    "repeat_interleave_0": lambda x: lucid.repeat_interleave(x, 2, dim=0),
    "repeat_interleave_last": lambda x: lucid.repeat_interleave(x, 2, dim=-1),
    "tile": lambda x: lucid.tile(x, tuple(2 for _ in x.shape)),
    "repeat": lambda x: x.repeat(*((2,) * x.ndim)),
    "permute": lambda x: x.permute(*reversed(range(x.ndim))),
    "reshape": lambda x: x.reshape(-1),
    "broadcast_to": lambda x: lucid.broadcast_to(x, (2, *x.shape)),
    "narrow": lambda x: x.narrow(0, 0, 0),
    "split": lambda x: lucid.split(x, 1, dim=0),
    "chunk": lambda x: lucid.chunk(x, 2, dim=-1),
}


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("op_name", list(_LAYOUT_OPS))
@pytest.mark.parametrize("shape", _EMPTY_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_sibling_layout_ops_handle_every_empty_position(
    device: str, op_name: str, shape: Shape
) -> None:
    """Every layout op forward and backward survives a zero extent at any axis."""
    x = _empty(shape, lucid.float32, device, grad=True)
    out = _LAYOUT_OPS[op_name](x)
    tensors = out if isinstance(out, tuple) else (out,)
    for t in tensors:
        assert t.numel() == 0
    if tensors:
        sum((t.sum() for t in tensors[1:]), tensors[0].sum()).backward()
        assert x.grad is not None and tuple(x.grad.shape) == shape


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("axis", [0, 1])
def test_cat_mixing_empty_and_non_empty_inputs(device: str, axis: int) -> None:
    """Concatenating an empty tensor beside a non-empty one never memcpy's a null source."""
    empty_shape = (0, 3) if axis == 0 else (2, 0)
    full_shape = (2, 3)
    e = _empty(empty_shape, lucid.float32, device, grad=True)
    f = lucid.ones(full_shape, dtype=lucid.float32, device=device, requires_grad=True)
    y = lucid.cat([e, f, e], dim=axis)
    assert tuple(y.shape) == full_shape
    assert y.reshape(-1).tolist() == [1.0] * 6
    y.sum().backward()
    assert e.grad is not None and tuple(e.grad.shape) == empty_shape
    assert f.grad is not None and f.grad.reshape(-1).tolist() == [1.0] * 6


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0), (0, 2, 3), (2, 0, 3), (2, 3, 0)])
def test_diagonal_and_unfold_empty_extents(device: str, shape: Shape) -> None:
    """Diagonal and unfold (and their backwards) handle empty extents at every axis."""
    x = _empty(shape, lucid.float32, device, grad=True)
    d = lucid.diagonal(x, dim1=-2, dim2=-1)
    assert d.numel() == 0
    d.sum().backward()
    assert x.grad is not None and tuple(x.grad.shape) == shape

    x2 = _empty(shape, lucid.float32, device, grad=True)
    u = x2.unfold(0, 0, 1)
    assert u.numel() == 0


@pytest.mark.parametrize("device", _DEVICES)
def test_dropout_empty_batch_no_divide_by_zero(device: str) -> None:
    """Empty-batch dropout / channel-mask multiply avoids B_sz == 0 division."""
    x = _empty((0, 4), lucid.float32, device, grad=True)
    y = F.dropout(x, p=0.5, training=True)
    assert tuple(y.shape) == (0, 4)
    y.sum().backward()
    assert x.grad is not None and tuple(x.grad.shape) == (0, 4)
