"""Which traced values a package carries as constants instead of computing.

Split from :mod:`lucid.coreml._build`, the driver, which asks once per
export (:func:`foldable`) and writes the answer down as constants.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol

__all__ = ["TRUE_FACTORIES", "f32_opmath_scalars", "foldable"]


class _Value(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def dtype(self) -> object: ...

    @property
    def shape(self) -> Sequence[int]: ...


class _Feed(Protocol):
    @property
    def dtype(self) -> object: ...

    @property
    def shape(self) -> Sequence[int]: ...


class _Op(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def inputs(self) -> Sequence[int]: ...

    @property
    def outputs(self) -> Sequence[_Value]: ...


class _Graph(Protocol):
    @property
    def ops(self) -> Sequence[_Op]: ...


#: Operations that make a tensor out of nothing. A traced node with no
#: recorded inputs is one of these, or it is an operation whose inputs
#: the tracer failed to record — and treating the second kind as a
#: constant is how a package comes to answer every input with the first
#: one's result. Only names that genuinely take no tensor are here:
#: ``tril`` and ``meshgrid`` do take one, whatever a node missing its
#: edges suggests.
TRUE_FACTORIES = frozenset(
    {"arange", "empty", "eye", "full", "linspace", "logspace", "ones", "zeros"}
)


def foldable(
    graph: _Graph,
    feeds: Mapping[int, object],
    input_ids: set[int],
    traced_values: Mapping[int, object],
    varying: set[int],
    output_ids: set[int],
) -> tuple[set[int], set[int]]:
    """Values a package can carry instead of computing, and who makes them.

    A windowed transformer spends a third of its graph on arithmetic that
    has nothing to do with its input: Swin builds the index into its
    relative-position table out of ``arange``, and rebuilds it on every
    prediction. 503 of its 1473 operations are that, and they run on the
    CPU, between operations that run on the Neural Engine.

    Constant means constant *here*: reachable only from the model's
    weights and from operations that make a tensor out of nothing.
    Anything one step from the input is not, and an operation whose
    inputs the tracer never recorded is treated as unknown rather than
    as a factory — that mistake is what freezes a model's answer.

    Parameters
    ----------
    graph : TraceGraph
        The traced graph.
    feeds : mapping of int to TensorImpl
        Its external feeds: the model's weights and the inputs.
    input_ids : set of int
        The feeds that are inputs, and so not constant.
    traced_values : mapping of int to TensorImpl
        Every value the trace kept, which a constant is written from.
    varying : set of int
        Values whose shape follows a flexible input.
    output_ids : set of int
        The model's outputs, which the caller reads.

    Returns
    -------
    tuple[set[int], set[int]]
        The values to emit as constants — only those something outside
        the constant region reads, since the rest have no consumer left
        — and the ids of every operation head inside the region, which
        the driver skips.
    """
    constant: set[int] = {tid for tid in feeds if tid not in input_ids}
    inside: set[int] = set()
    for op in graph.ops:
        if _constant_op(op, constant, traced_values, varying):
            inside.add(int(op.outputs[0].id))
            constant.update(int(o.id) for o in op.outputs)

    if not inside:
        return set(), set()

    # Only the values something outside the region reads have to be
    # written; the rest were steps on the way to them.  A model output is
    # read too, by the caller: ``return x * 2, lucid.zeros(2, 3)`` left
    # the zeros with no name, and the export failed saying nothing in the
    # graph produced them.
    made = constant.difference(feeds)
    wanted = made.intersection(output_ids)
    for op in graph.ops:
        if not op.outputs or int(op.outputs[0].id) not in inside:
            wanted.update(made.intersection(op.inputs))
    return wanted, inside


def _constant_op(
    op: _Op,
    constant: set[int],
    traced_values: Mapping[int, object],
    varying: set[int],
) -> bool:
    """Whether ``op``'s results are constant, given what already is."""
    if not op.outputs:
        return False
    if op.inputs:
        if not all(i in constant for i in op.inputs):
            return False
    elif op.name not in TRUE_FACTORIES:
        return False
    if not all(int(o.id) in traced_values for o in op.outputs):
        # Nothing to write in its place.
        return False
    # ``zeros(x.shape[0], ...)``: made out of nothing, and still the
    # input's size. The traced value is one size of many.
    return not any(int(o.id) in varying for o in op.outputs)


def _opmath_scalar_input(op: _Op, meta: Mapping[int, tuple[str, tuple[int, ...]]]) -> int | None:
    if op.name not in ("mul", "div", "floordiv") or len(op.inputs) != 2 or not op.outputs:
        return None
    if str(op.outputs[0].dtype).split(".")[-1] != "F16":
        return None
    a_id, b_id = int(op.inputs[0]), int(op.inputs[1])
    a_m = meta.get(a_id, ("", ()))
    b_m = meta.get(b_id, ("", ()))
    if a_m[0] == "F16" and b_m == ("F32", ()):
        return b_id
    if b_m[0] == "F16" and a_m == ("F32", ()):
        return a_id
    return None


def f32_opmath_scalars(graph: _Graph, feeds: Mapping[int, _Feed]) -> set[int]:
    """0-d float32 values that pair with an F16 tensor in ``mul``/``div``/``floordiv``."""
    meta: dict[int, tuple[str, tuple[int, ...]]] = {
        tid: (str(impl.dtype).split(".")[-1], tuple(int(d) for d in impl.shape))
        for tid, impl in feeds.items()
    }
    for op in graph.ops:
        for o in op.outputs:
            meta[int(o.id)] = (
                str(o.dtype).split(".")[-1],
                tuple(int(d) for d in o.shape),
            )
    scalars: set[int] = set()
    for op in graph.ops:
        sid = _opmath_scalar_input(op, meta)
        if sid is not None:
            scalars.add(sid)
    return scalars
