"""
The argument contract every spatial ``nn.functional`` op shares.

Convolution, transposed convolution and pooling take the same kinds of
arguments: a size given once or per spatial axis, a stride that may be
left out, padding as a number or as ``"same"`` / ``"valid"``, and an input
with or without its batch axis.  The reference reads them the same way for
every op.  Each op used to read its own, and they drifted:

- ``conv2d`` split ``padding="same"`` into its four characters;
- ``max_pool2d`` handed an explicit ``stride=0`` to an engine that reads 0
  as "use the kernel size", so it pooled where the reference refuses;
- no op took an unbatched input.

Every spatial entry point now reads its arguments here, so a rule changes
in one place.  The engine entry points stay strict about rank and geometry;
this layer turns the reference's spellings into the one form the engine
takes, and refuses a value whose meaning the engine would silently change.
"""

from collections.abc import Sequence
from typing import NamedTuple

from lucid._tensor.tensor import Tensor


def _ntuple(value: int | Sequence[int], n: int, op: str, arg: str) -> tuple[int, ...]:
    """``value`` as one integer per spatial axis.

    A single integer, or a one-element sequence, applies to every axis, as
    the reference reads it.  Any other length is refused: it used to reach
    an unpacking assignment and fail with "too many values to unpack".
    """
    if isinstance(value, int):
        return (value,) * n
    items = tuple(value)
    if len(items) == 1:
        items = items * n
    if len(items) != n:
        raise ValueError(
            f"{op}: expected {arg} to be a single integer or {n} integers, " f"got {value!r}"
        )
    return tuple(int(v) for v in items)


def _require_positive(values: tuple[int, ...], op: str, arg: str) -> None:
    """Refuse a stride, kernel or dilation that is zero or negative."""
    if any(v <= 0 for v in values):
        raise ValueError(f"{op}: {arg} must be greater than zero, got {values}")


def _require_non_negative(values: tuple[int, ...], op: str, arg: str) -> None:
    """Refuse a negative padding."""
    if any(v < 0 for v in values):
        raise ValueError(f"{op}: {arg} must be non-negative, got {values}")


def _spatial_stride(
    stride: int | Sequence[int] | None, default: tuple[int, ...], n: int, op: str
) -> tuple[int, ...]:
    """The stride per axis; left out (``None`` or empty) means ``default``.

    A stride that is given has to be positive.  The engine's pooling doors
    read a stride of 0 as "the kernel size", so passing the caller's 0
    through pooled with the kernel's stride where the reference refuses.
    """
    if stride is None or (not isinstance(stride, int) and len(stride) == 0):
        return default
    s = _ntuple(stride, n, op, "stride")
    _require_positive(s, op, "stride")
    return s


def _add_batch_axis(x: Tensor, n: int) -> tuple[Tensor, bool]:
    """An unbatched ``(C, *spatial)`` input as a batch of one.

    Returns the input to run and whether the batch axis was added, which
    :func:`_drop_batch_axis` takes to put the result back.  Any other rank
    goes through unchanged, for the engine's own rank check to refuse.
    """
    if x.ndim == n + 1:
        return x.unsqueeze(0), True
    return x, False


def _drop_batch_axis(out: Tensor, unbatched: bool) -> Tensor:
    """Undo :func:`_add_batch_axis` on a result."""
    return out.squeeze(0) if unbatched else out


def _spatial_extent(x: Tensor, n: int, op: str) -> tuple[int, ...]:
    """The last ``n`` sizes of an ``(N, C, *spatial)`` or ``(C, *spatial)`` input.

    Any other rank is refused, as the reference refuses it.
    """
    if x.ndim not in (n + 1, n + 2):
        raise ValueError(
            f"{op}: expected a {n + 1}-D or {n + 2}-D input, "
            f"got shape {tuple(int(d) for d in x.shape)}"
        )
    return tuple(int(v) for v in x.shape[-n:])


def _check_string_padding(padding: str, stride: tuple[int, ...], op: str) -> None:
    """Refuse a padding string the reference refuses.

    ``"same"`` keeps the output the size of the input, which only a
    stride of 1 can do.
    """
    if padding not in ("same", "valid"):
        raise ValueError(f"{op}: invalid padding string {padding!r}, expected 'same' or 'valid'")
    if padding == "same" and any(s != 1 for s in stride):
        raise ValueError(
            f"{op}: padding='same' is not supported for strided convolutions " f"(stride={stride})"
        )


def _conv_padding(
    padding: str | int | Sequence[int],
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
    op: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """A convolution's padding as ``(before, after)`` per spatial axis.

    ``"same"`` pads ``dilation * (kernel - 1)`` in all; when that is odd,
    the extra element goes after, as the reference places it.  A number
    pads both sides alike.
    """
    n = len(kernel)
    if isinstance(padding, str):
        _check_string_padding(padding, stride, op)
        if padding == "valid":
            return (0,) * n, (0,) * n
        total = tuple(d * (k - 1) for k, d in zip(kernel, dilation, strict=True))
        before = tuple(t // 2 for t in total)
        return before, tuple(t - b for t, b in zip(total, before, strict=True))
    p = _ntuple(padding, n, op, "padding")
    _require_non_negative(p, op, "padding")
    return p, p


class _PoolGeometry(NamedTuple):
    """A pooling window, one entry per spatial axis."""

    kernel: tuple[int, ...]
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]


def _pool_geometry(
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None,
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    n: int,
    op: str,
) -> _PoolGeometry:
    """A pooling op's window, checked the way the reference checks it.

    Padding may be at most half the kernel size, ``kernel_size // 2``.
    Dilation does not widen the limit; the reference refuses
    ``kernel_size=3, padding=2, dilation=2`` too, though that window would
    reach the input.
    """
    kernel = _ntuple(kernel_size, n, op, "kernel_size")
    _require_positive(kernel, op, "kernel_size")
    strides = _spatial_stride(stride, kernel, n, op)
    dilations = _ntuple(dilation, n, op, "dilation")
    _require_positive(dilations, op, "dilation")
    pads = _ntuple(padding, n, op, "padding")
    _require_non_negative(pads, op, "padding")
    if any(p > k // 2 for k, p in zip(kernel, pads, strict=True)):
        raise ValueError(
            f"{op}: padding should be at most half of the kernel size, got "
            f"padding={pads} and kernel_size={kernel}"
        )
    return _PoolGeometry(kernel, strides, pads, dilations)
