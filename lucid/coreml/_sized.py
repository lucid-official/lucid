"""Emitter steps whose sizes a flexible export leaves to run time (LCD-365).

Two operations record a size the input decides as an attribute of the
trace: a slice cut at the input's length (``pos_table[:T]``, a
``split_at``) and a resample (its output height and width). Under a fixed
export the emitters in :mod:`lucid.coreml._emit` write those sizes down;
under a flexible one they call here, where each size is read off the
input when the program runs, or the export is refused by name.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from lucid.coreml import _flex, _spec

if TYPE_CHECKING:
    from lucid.coreml._build import Builder
    from lucid.coreml._emit import Bindings, EmitResult, TracedOp

__all__ = ["declared", "resample", "split_at_run_time"]

_MIL_INT32 = _spec.INT32


def declared(shape: Sequence[int]) -> list[int]:
    """A traced shape as numbers, ``-1`` on each axis a flexible export opens.

    Parameters
    ----------
    shape : sequence of int
        A result shape as an emitter sees it: :class:`~lucid.coreml._flex.Dims`
        under a flexible export, a plain sequence otherwise.

    Returns
    -------
    list of int
        The sizes, open axes included as ``-1`` rather than refused.
    """
    if isinstance(shape, _flex.Dims):
        return shape.declared()
    return [int(d) for d in shape]


def split_at_run_time(b: Builder, op: TracedOp, x: str, traced_axis: int, axis: int) -> list[str]:
    """A split whose sections follow the input: one ``slice_by_size`` each.

    ``pos_table[:, :T]`` for a sequence of ``T`` is a split of the table
    at ``T``, and MIL's ``split`` takes its sizes as a constant.  Each
    section starts where the ones before it end and is as long as the input
    axis its length equals at every traced shape; the last runs to the end
    of the operand, whatever that is.  A section of any other length is
    refused, since there is nothing to read it from at run time.

    Parameters
    ----------
    b : Builder
        Builder appending the slices.
    op : TracedOp
        The traced ``split_at``.
    x : str
        The operand.
    traced_axis : int
        The split axis as the trace records it.
    axis : int
        The same axis in the program, one lower when the operand lost a
        leading axis of one.

    Returns
    -------
    list of str
        One value per section, in order.

    Raises
    ------
    ShapeNotFlexible
        A section other than the last changes length with no input axis to
        read the length from.
    """
    rank = len(b.shape_of(x))
    fixed, read = 0, None  # where the next section starts: fixed + read
    sections: list[str] = []
    for index, out in enumerate(op.outputs):
        size = declared(out.shape)[traced_axis]
        last = index == len(op.outputs) - 1
        if last:
            length = b.const_ints([-1])  # to the end of the operand
        elif size >= 0:
            length = b.const_ints([size])
        else:
            source = b.flex.source(int(out.id), traced_axis)
            if source is None or b.flex_input is None:
                raise _flex.ShapeNotFlexible(
                    op.name,
                    f"section {index} changes length with the input, and no input "
                    "axis has its length at every traced shape",
                )
            length = b.run_time_dim(b.flex_input, source)
        start = b.const_ints([fixed])
        if read is not None:
            start = read if fixed == 0 else _plus(b, read, start)
        sections.append(
            b.emit(
                "slice_by_size",
                [
                    ("x", x),
                    ("begin", _on_axis(b, rank, axis, start, 0)),
                    # -1 on every other axis: all of it.
                    ("size", _on_axis(b, rank, axis, length, -1)),
                ],
                b.result_dims(op, index),
            )
        )
        if size >= 0:
            fixed += size
        elif not last:
            read = length if read is None else _plus(b, read, length)
    return sections


def _plus(b: Builder, left: str, right: str) -> str:
    """The sum of two int32 ``[1]`` values, computed at run time."""
    return b.emit("add", [("x", left), ("y", right)], [1], dtype=_MIL_INT32)


def _on_axis(b: Builder, rank: int, axis: int, value: str, other: int) -> str:
    """An int32 vector of ``rank``: ``value`` (an int32 ``[1]``) at ``axis``,
    ``other`` everywhere else."""
    pieces = []
    if axis > 0:
        pieces.append(b.const_ints([other] * axis))
    pieces.append(value)
    if axis < rank - 1:
        pieces.append(b.const_ints([other] * (rank - axis - 1)))
    if len(pieces) == 1:
        return value
    return b.emit(
        "concat",
        [("values", pieces), ("axis", b.const_int(0)), ("interleave", b.const_bool(False))],
        [rank],
        dtype=_MIL_INT32,
    )


def _scales(op: TracedOp, b: Builder, value: str) -> tuple[float, float] | None:
    """Height and width scale factors, or None for a fixed output size.

    A flexible export opens the input's height and width. ``Upsample(size=)``
    keeps its output size whatever the input, so its scale follows the input
    and no scale factor is right: None says to resize to that size instead.
    ``Upsample(scale_factor=)`` opens the output too; its scale is the one
    ratio of output to input at every traced shape, and is refused when the
    probes disagree (a size floored differently at one of them).
    """
    src = b.shape_of(value).declared()
    out = b.result_dims(op).declared()
    plane = (len(out) - 2, len(out) - 1)
    if all(out[axis] >= 0 for axis in plane) and any(src[axis] < 0 for axis in plane):
        return None
    operand = b.traced_id(value)
    scales = []
    for axis in plane:
        if src[axis] >= 0 and out[axis] >= 0:
            scales.append(out[axis] / src[axis])
            continue
        ratio = None
        if operand is not None:
            ratio = b.flex.ratio(int(op.outputs[0].id), operand, axis)
        if ratio is None:
            raise _flex.ShapeNotFlexible(
                op.name,
                f"its output's axis {axis} is not one multiple of the input's at "
                "every traced shape, so no scale factor reproduces it",
            )
        scales.append(ratio)
    return scales[0], scales[1]


def _nearest_to_size(b: Builder, op: TracedOp, x: str) -> EmitResult:
    """A nearest resample to a fixed size from an input of any size.

    Eager reads source row ``floor(i * in / out)`` for output row ``i``.
    MIL's ``resize_nearest_neighbor`` centres each pixel first — row
    ``floor((i + 0.5) * in / out)`` — and answers differently whenever the
    sizes do not divide.  So each axis is a ``gather`` at those rows, the
    product ``i * in`` taken at run time and divided by ``out`` in integers,
    which is eager's double-precision quotient exactly.
    """
    out = b.result_dims(op).declared()
    src = b.shape_of(x).declared()
    height, width = len(out) - 2, len(out) - 1

    def rows(value: str, axis: int) -> Bindings:
        size = out[axis]
        extent = b.run_time_dim(x, axis) if src[axis] < 0 else b.const_ints([src[axis]])
        scaled = b.emit(
            "mul", [("x", b.const_ints(list(range(size)))), ("y", extent)], [size], dtype=_MIL_INT32
        )
        picked = b.emit(
            "floor_div", [("x", scaled), ("y", b.const_ints([size]))], [size], dtype=_MIL_INT32
        )
        return [
            ("x", value),
            ("indices", picked),
            ("axis", b.const_int(axis)),
            ("validate_indices", b.const_bool(False)),
        ]

    tall = b.emit("gather", rows(x, height), [*src[:height], out[height], src[width]])
    return "gather", rows(tall, width)


def resample(b: Builder, op: TracedOp, x: str) -> EmitResult:
    """A 2-D resample: an upsample by its scale, or a resize to its size.

    ``interpolate_nearest_2d`` and ``interpolate_bilinear``. A bilinear
    resize samples at half-pixel centres under MIL's ``UNALIGN_CORNERS`` and
    pins the corners under ``STRICT_ALIGN_CORNERS``, eager's two
    ``align_corners`` settings.

    Parameters
    ----------
    b : Builder
        Builder minting the operands.
    op : TracedOp
        The traced resample.
    x : str
        The image.

    Returns
    -------
    tuple[str, list]
        The last operation, for the driver to append.
    """
    nearest = op.name == "interpolate_nearest_2d"
    align = bool(op.attrs.get("align_corners", False))
    scales = _scales(op, b, x)
    if scales is None and nearest:
        return _nearest_to_size(b, op, x)
    if scales is None:
        size = b.result_dims(op)
        mode = "STRICT_ALIGN_CORNERS" if align else "UNALIGN_CORNERS"
        return "resize_bilinear", [
            ("x", x),
            ("target_size_height", b.const_int(size[-2])),
            ("target_size_width", b.const_int(size[-1])),
            ("sampling_mode", b.const_str(mode)),
        ]
    bindings: Bindings = [
        ("x", x),
        ("scale_factor_height", b.const_float32(scales[0])),
        ("scale_factor_width", b.const_float32(scales[1])),
    ]
    if nearest:
        return "upsample_nearest_neighbor", bindings
    return "upsample_bilinear", [*bindings, ("align_corners", b.const_bool(align))]
