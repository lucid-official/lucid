"""Rank-6 shapes that Core ML's rank cap would refuse, staged under it.

Core ML's program dialect stops at rank five. Two patterns reach rank six
only on the way somewhere: a window partition (``reshape → permute →
reshape``) and a zero-stuffing upsample (``reshape → pad → reshape``).
Both are matched here on their shapes and rewritten as a sequence of
rank-four steps, so the tall tensor never exists in the package.
"""

import dataclasses
import math
from typing import Any, NamedTuple

__all__ = [
    "MAX_RANK",
    "StuffRewrite",
    "WindowRewrite",
    "stage_permutation",
    "stage_zero_stuff",
    "stuff_rewrites",
    "window_rewrites",
]

# Core ML's ML Program dialect refuses tensors above this rank.
MAX_RANK = 5


class WindowRewrite(NamedTuple):
    """A ``reshape → permute → reshape`` whose middle exceeds the rank cap.

    Keyed and cross-referenced by value id rather than by op identity:
    ``graph.ops`` hands out fresh Python wrappers on each access, so
    ``id(op)`` is stable only by accident and silently stops matching on
    a graph large enough to matter.

    ``absorbs`` are the two follow-on ops the rewrite consumes, so the
    emit loop knows to skip them; ``out_id`` is the value the rewritten
    chain has to bind, which is the final reshape's.
    """

    mid_shape: list[int]
    perm: list[int]
    out_shape: list[int]
    out_id: int
    absorbs: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class StuffRewrite:
    """A ``reshape`` / ``pad`` / ``reshape`` that stuffs zeros between elements.

    Upsampling before a convolution is often written by splitting each
    axis into ``(d, 1)``, padding the new axis out to ``(d, k)`` with
    zeros, and merging back to ``d * k`` — every element followed by
    ``k - 1`` zeros. Doing two axes at once needs rank 6, which Core ML
    does not have.
    """

    source_shape: list[int]
    factors: list[int]
    fill: float
    out_shape: list[int]
    out_id: int
    absorbs: tuple[int, ...]


def _chains(graph: Any, middle: str) -> list[tuple[Any, Any, Any]]:
    """Each ``reshape → middle → reshape`` whose first result is too tall.

    Every link has to be the only reader of the one before — a second
    reader would need the tall tensor to exist for real — and the last
    reshape has to come back under the cap.
    """
    readers: dict[int, list[Any]] = {}
    for op in graph.ops:
        for iid in op.inputs:
            readers.setdefault(iid, []).append(op)

    def only_reader(op: Any, name: str) -> Any:
        found = readers.get(op.outputs[0].id, [])
        if len(found) != 1 or found[0].name != name or len(found[0].outputs) != 1:
            return None
        return found[0]

    chains: list[tuple[Any, Any, Any]] = []
    for op in graph.ops:
        if op.name != "reshape" or len(op.outputs) != 1:
            continue
        if len(op.outputs[0].shape) <= MAX_RANK:
            continue
        step = only_reader(op, middle)
        down = only_reader(step, "reshape") if step is not None else None
        if down is not None and len(down.outputs[0].shape) <= MAX_RANK:
            chains.append((op.outputs[0], step, down.outputs[0]))
    return chains


def _interleave(tall: list[int], pads: list[int]) -> tuple[list[int], list[int]] | None:
    """The source shape and per-axis factors a pad stuffs zeros by, or ``None``.

    Only singleton axes may grow, and only at their far end: that is what
    makes the pad an interleave rather than a border. The source is the
    tall shape without those singletons — read off the shapes rather than
    from the producing operation, which does not exist when the reshape
    reads a model input.
    """
    if len(pads) != 2 * len(tall):
        return None
    widths: dict[int, int] = {}
    for axis, extent in enumerate(tall):
        before, after = pads[2 * axis], pads[2 * axis + 1]
        if before != 0 or (after != 0 and extent != 1):
            return None
        if after:
            widths[axis] = after + 1
    if not widths or 0 in widths:
        return None
    source = [d for i, d in enumerate(tall) if i not in widths]
    factors = [1] * len(source)
    walk = -1
    for axis in range(len(tall)):
        if axis in widths:
            factors[walk] = widths[axis]
        else:
            walk += 1
    return source, factors


def stuff_rewrites(graph: Any) -> dict[int, StuffRewrite]:
    """Find the zero-stuffing upsamples that can be staged under the cap.

    Matched on shape rather than on the model: the tall tensor has to be
    the source with singleton axes interleaved, the pad has to add only
    at the end of exactly those axes, and the result has to merge each
    pair back. Anything else keeps its rank and is refused as before.
    """
    found: dict[int, StuffRewrite] = {}
    for tall, pad_op, out in _chains(graph, "pad"):
        pads = [int(v) for v in (pad_op.attrs.get("pads") or [])]
        plan = _interleave([int(d) for d in tall.shape], pads)
        if plan is None:
            continue
        source_shape, factors = plan
        out_shape = [int(d) for d in out.shape]
        if [d * f for d, f in zip(source_shape, factors, strict=True)] != out_shape:
            continue
        found[tall.id] = StuffRewrite(
            source_shape=source_shape,
            factors=factors,
            fill=float(pad_op.attrs.get("constant", 0.0)),
            out_shape=out_shape,
            out_id=out.id,
            absorbs=(pad_op.outputs[0].id, out.id),
        )
    return found


def stage_zero_stuff(builder: Any, source: str, plan: StuffRewrite) -> str:
    """Stuff one axis at a time, so no stage exceeds the rank cap.

    Splitting every axis at once is what needed rank 6. Splitting one
    axis, padding it and merging it back leaves the rank where it
    started, so a rank-4 tensor never passes rank 5 however many axes
    are upsampled.

    Parameters
    ----------
    builder : Builder
        Builder appending the stages.
    source : str
        Value the rewritten chain starts from.
    plan : StuffRewrite
        What the chain does, with open axes as ``-1``.

    Returns
    -------
    str
        The value the chain's last reshape produced.
    """
    current = list(plan.source_shape)
    value = source
    for axis, factor in enumerate(plan.factors):
        if factor <= 1:
            continue
        split = current[:axis] + [current[axis], 1] + current[axis + 1 :]
        value = builder.emit(
            "reshape", [("x", value), ("shape", builder.shape_operand(split))], split
        )
        pads = [0] * (2 * len(split))
        pads[2 * (axis + 1) + 1] = factor - 1
        widened = list(split)
        widened[axis + 1] = factor
        value = builder.emit(
            "pad",
            [
                ("x", value),
                ("pad", builder.const_ints(pads)),
                ("mode", builder.const_str("constant")),
                ("constant_val", builder.const_float(plan.fill)),
            ],
            widened,
        )
        current[axis] = current[axis] * factor if current[axis] >= 0 else -1
        value = builder.emit(
            "reshape", [("x", value), ("shape", builder.shape_operand(current))], list(current)
        )
    return value


def window_rewrites(graph: Any) -> dict[int, WindowRewrite]:
    """Find the high-rank triples that can be staged under the cap.

    A reshape that lifts a tensor past rank 5 is representable only if
    the rank comes straight back down, which in practice means a permute
    and a reshape follow it and nothing else reads the tall value. That
    is the window partition (and its inverse, the merge); the pattern is
    matched rather than the model, so a grid partition or any other
    factorised permutation is the same code path.
    """
    found: dict[int, WindowRewrite] = {}
    for tall, perm_op, out in _chains(graph, "permute"):
        raw = perm_op.attrs["permutation"]
        perm = [int(a) for a in (raw if isinstance(raw, (list, tuple)) else [raw])]
        if sorted(perm) != list(range(len(tall.shape))):
            continue
        found[tall.id] = WindowRewrite(
            mid_shape=[int(d) for d in tall.shape],
            perm=perm,
            out_shape=[int(d) for d in out.shape],
            out_id=out.id,
            absorbs=(perm_op.outputs[0].id, out.id),
        )
    return found


def stage_permutation(builder: Any, source: str, plan: WindowRewrite) -> str:
    """Realise ``plan`` without ever materialising the tall tensor.

    A reshape is a reinterpretation of a contiguous buffer, so any run of
    adjacent axes may be grouped into one. Each stage therefore views the
    value as ``(left, a, b, right)`` — rank 4, whatever the logical rank
    is — swaps the middle pair, and leaves the result contiguous in the
    new order for the next stage to regroup. Sorting the axis order into
    the wanted one by adjacent swaps then costs one stage per inversion,
    and the tall shape stays a bookkeeping detail.

    Parameters
    ----------
    builder : Builder
        Builder appending the stages.
    source : str
        Value the rewritten chain starts from.
    plan : WindowRewrite
        What the chain does, with open axes as ``-1``.

    Returns
    -------
    str
        The value the chain's last reshape produced.
    """
    shape = list(plan.mid_shape)
    order = list(range(len(shape)))
    name = source

    for slot, axis in enumerate(plan.perm):
        at = order.index(axis)
        while at > slot:
            i = at - 1
            left = _volume(shape[:i])
            right = _volume(shape[i + 2 :])
            grouped = [left, shape[i], shape[i + 1], right]
            viewed = builder.emit(
                "reshape", [("x", name), ("shape", builder.shape_operand(grouped))], grouped
            )
            swapped = [left, shape[i + 1], shape[i], right]
            name = builder.emit(
                "transpose", [("x", viewed), ("perm", builder.const_ints([0, 2, 1, 3]))], swapped
            )
            shape[i], shape[i + 1] = shape[i + 1], shape[i]
            order[i], order[i + 1] = order[i + 1], order[i]
            at = i

    staged: str = builder.emit(
        "reshape",
        [("x", name), ("shape", builder.shape_operand(plan.out_shape, plan.out_id))],
        plan.out_shape,
    )
    return staged


def _volume(dims: list[int]) -> int:
    """How many elements ``dims`` hold, or ``-1`` if an open axis is among them."""
    return -1 if any(d < 0 for d in dims) else math.prod(dims)
