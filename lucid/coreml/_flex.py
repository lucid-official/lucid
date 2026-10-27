"""What a flexible export knows about a shape, and what it must not assume.

A package exported with ``shapes=`` or ``shape_range=`` accepts sizes the
trace never saw. Which axes those are is found by tracing at several
shapes and comparing (:func:`compare`), and every emitter then works from
that answer. The failure this module exists to prevent is quiet: an
emitter that reads the traced size of an axis the export leaves open and
writes it into a constant — a ``tile`` count, a ``reshape`` target, a
``zeros`` — builds a package that loads, accepts every allowed shape, and
answers at the traced one. :class:`Dims` makes such a read raise
:class:`~lucid.coreml.ShapeNotFlexible` naming the operation, so an
emitter either handles the open axis or is refused by name.
"""

import dataclasses
from collections.abc import Iterator, Sequence
from typing import NamedTuple, Protocol, overload, override

__all__ = ["Dims", "Flexibility", "GuardedOp", "ShapeNotFlexible", "compare", "flex", "plan_probes"]


class ShapeNotFlexible(NotImplementedError):
    """An operation's configuration was derived from the input's size.

    Named rather than generic because the alternative is a package that
    accepts several shapes and is only correct at one of them. An adaptive
    pool is the usual cause: the tracer records it as an average pool
    whose kernel came from the input, so the same model traced at two
    resolutions produces two different kernels.

    Examples
    --------
    A range of resolutions, since that is what an adaptive pool's kernel
    is taken from — a range of batch sizes leaves it alone:

    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.coreml as cml
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten()
    ... ).eval()
    >>> room = tempfile.mkdtemp()
    >>> try:
    ...     cml.export(model, lucid.randn(1, 3, 16, 16), f"{room}/m.mlpackage",
    ...                shape_range={2: (16, 32), 3: (16, 32)})
    ... except cml.ShapeNotFlexible as refusal:
    ...     print(refusal)
    lucid.coreml: operation 'avg_pool2d' cannot take a flexible shape ...
    >>> shutil.rmtree(room)
    """

    def __init__(self, op_name: str, detail: str) -> None:
        super().__init__(
            f"lucid.coreml: operation {op_name!r} cannot take a flexible shape — "
            f"{detail}. Trace it at one shape, or replace the layer with one whose "
            "configuration does not depend on the input size."
        )
        self.op_name = op_name


def flex(shape: Sequence[int], axes: set[int] | None) -> list[int]:
    """A shape with its varying axes written ``-1``, which Core ML fills in.

    Parameters
    ----------
    shape : sequence of int
        Traced shape.
    axes : set of int or None
        Axes the export leaves open.
    """
    if not axes:
        return [int(d) for d in shape]
    return [-1 if axis in axes else int(dim) for axis, dim in enumerate(shape)]


class Dims(Sequence[int]):
    """A recorded shape whose flexible axes cannot be read as numbers.

    Taking its length, slicing it and reading a fixed axis behave as on a
    list. Reading an axis the export leaves open raises — iterating does
    too, once it reaches one — because the only number available is the
    traced size, and a translation that used it would be right at that
    size alone. :meth:`declared` is the form to declare a result type
    with: the open axes there are ``-1``, which is what Core ML reads as
    "decided at run time".
    """

    __slots__ = ("_dims", "_op")

    def __init__(self, dims: Sequence[int], op_name: str) -> None:
        self._dims = [int(d) for d in dims]
        self._op = op_name

    @override
    def __len__(self) -> int:
        return len(self._dims)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> Dims: ...

    @override
    def __getitem__(self, index: int | slice) -> int | Dims:
        if isinstance(index, slice):
            return Dims(self._dims[index], self._op)
        size = self._dims[index]
        if size < 0:
            raise ShapeNotFlexible(
                self._op,
                f"its translation reads the size of axis {index % len(self._dims)}, "
                "which this export leaves free to change, so the package would "
                "keep the traced size whatever the input",
            )
        return size

    @override
    def __iter__(self) -> Iterator[int]:
        for axis in range(len(self._dims)):
            yield self[axis]

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dims):
            return self._dims == other._dims
        if isinstance(other, (list, tuple)):
            return self._dims == [int(d) for d in other]
        return NotImplemented

    @override
    def __hash__(self) -> int:
        return hash(tuple(self._dims))

    @override
    def __repr__(self) -> str:
        return f"Dims({self._dims})"

    def declared(self) -> list[int]:
        """The shape to declare a value with, ``-1`` on every open axis."""
        return list(self._dims)

    @property
    def open(self) -> bool:
        """Whether any axis follows the input."""
        return any(d < 0 for d in self._dims)


@dataclasses.dataclass
class Flexibility:
    """Which axes of which traced values follow the input, and how.

    Everything is keyed by the value identifiers of the probe traces,
    which the graph that is emitted must share — :attr:`signature` is
    what that is checked against.
    """

    #: Value id to the axes whose size changed between probes.
    varying: dict[int, set[int]] = dataclasses.field(default_factory=dict)
    #: Value id to its shape at each probe, for every value that varies.
    probes: dict[int, list[tuple[int, ...]]] = dataclasses.field(default_factory=dict)
    #: The flexible input's identifier, ``-1`` for a fixed-shape export.
    input_id: int = -1
    #: Operation names and result ids of the first probe, in order.
    signature: list[tuple[str, tuple[int, ...]]] = dataclasses.field(default_factory=list)

    def source(self, value_id: int, axis: int) -> int | None:
        """The input axis this value's axis equals at every probe.

        Only a varying input axis qualifies, and only a unique one, so a
        guess is refused rather than made. The probes move one input axis
        at a time, so two input axes never agree at all of them.

        Parameters
        ----------
        value_id : int
            Traced value.
        axis : int
            One of its open axes.
        """
        seen = self.probes.get(value_id)
        given = self.probes.get(self.input_id)
        if not seen or not given or len(seen) != len(given):
            return None
        sizes = [shape[axis] for shape in seen]
        matches = [
            j
            for j in sorted(self.varying.get(self.input_id, ()))
            if [shape[j] for shape in given] == sizes
        ]
        return matches[0] if len(matches) == 1 else None

    def ratio(self, value_id: int, operand_id: int, axis: int) -> float | None:
        """The one ratio of a value's axis to its operand's at every probe.

        Parameters
        ----------
        value_id, operand_id : int
            Traced values; ``axis`` indexes both.
        axis : int
            The axis compared.

        Returns
        -------
        float or None
            The ratio, or None when it is not the same at every probe or a
            probe of either value is missing.
        """
        seen = self.probes.get(value_id)
        given = self.probes.get(operand_id)
        if not seen or not given or len(seen) != len(given):
            return None
        found = {mine[axis] / theirs[axis] for mine, theirs in zip(seen, given, strict=True)}
        return found.pop() if len(found) == 1 else None

    def carried(
        self, value: tuple[int, list[int]], operand: tuple[int, list[int]]
    ) -> dict[int, int]:
        """Which open axes of a reshaped value are axes of its operand.

        Walked from both ends: an axis is kept while the two agree on it —
        the same fixed size, or the same size at every probe — and an axis
        of one inserted or dropped on either side is stepped over. What a
        reshape merges or splits stays unmatched. The kept ones are the
        operand's own axes, so their size at run time is the operand's,
        exactly, whatever any input axis happened to equal.

        Parameters
        ----------
        value, operand : tuple of int and list of int
            Each one's traced id and declared shape, ``-1`` on open axes.

        Returns
        -------
        dict of int to int
            Open axis of the value to the axis of the operand it is.
        """
        ours, theirs = self._keys(*value), self._keys(*operand)
        found = _walk(ours, theirs, range(len(ours)), range(len(theirs)))
        back = _walk(ours, theirs, range(len(ours) - 1, -1, -1), range(len(theirs) - 1, -1, -1))
        for axis, other in back.items():
            if axis not in found and other not in found.values():
                found[axis] = other
        return {axis: other for axis, other in found.items() if ours[axis][0]}

    def _keys(self, value_id: int, dims: list[int]) -> list[tuple[bool, object]]:
        """Per axis: open or not, and the size — at every probe when open."""
        seen = self.probes.get(value_id) or []
        return [
            ((True, tuple(shape[axis] for shape in seen) or None) if size < 0 else (False, size))
            for axis, size in enumerate(dims)
        ]


def signature(graph: _Graph) -> list[tuple[str, tuple[int, ...]]]:
    """A traced graph's operation names and result ids, in order.

    Two traces of the same model at the same shape that disagree here are
    two different graphs, whatever their outputs look like.
    """
    return [(op.name, tuple(int(o.id) for o in op.outputs)) for op in graph.ops]


def plan_probes(
    default: tuple[int, ...],
    inputs: int,
    shapes: list[tuple[int, ...]] | None,
    shape_range: dict[int, tuple[int, int]] | None,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[tuple[int, int]]]:
    """The shapes to trace at, after checking the request makes sense.

    Parameters
    ----------
    default : tuple of int
        The example's shape.
    inputs : int
        How many inputs the model takes; flexibility needs one.
    shapes : list of tuple of int or None
        The enumerated shapes ``export`` was given.
    shape_range : dict of int to tuple of int or None
        The per-axis range ``export`` was given.

    Returns
    -------
    tuple
        ``(probes, ordered, bounds)``: the shapes to trace, the default
        first; the enumerated shapes in declaration order; the per-axis
        range. Empty lists for a fixed-shape export.
    """
    if shapes is not None and shape_range is not None:
        raise ValueError(
            "lucid.coreml: shapes and shape_range say different things about the "
            "same input — a list of allowed shapes, or a range that admits "
            "everything between. Give one"
        )
    if shapes is None and shape_range is None:
        return [], [], []
    if inputs != 1:
        kind = "a shape range needs" if shape_range else "flexible shapes need"
        raise ValueError(f"lucid.coreml: {kind} a single-input model, and this one takes {inputs}")
    if shapes is not None:
        ordered = _enumerated(default, shapes)
        return ordered, ordered, []
    assert shape_range is not None
    bounds = _bounds(default, shape_range)
    # Each ranged axis to its two ends, the others held at the example's
    # size: a dependence on the size shows up at the ends, and moving one
    # axis at a time is what tells apart two values that moving them
    # together would make look alike — the batch and a stride-32 height
    # are both 1 then 4. One trace per end, so ``2k + 1`` for ``k`` axes.
    probes = [default]
    for axis, ends in enumerate(bounds):
        for end in ends:
            moved = default[:axis] + (end,) + default[axis + 1 :]
            if moved not in probes:
                probes.append(moved)
    if len(probes) < 2:
        raise ValueError(f"lucid.coreml: the range admits only the example's own shape {default}")
    return probes, [], bounds


def _enumerated(default: tuple[int, ...], shapes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """The listed shapes, the example's first and nowhere else."""
    ordered = [default] + [tuple(s) for s in shapes if tuple(s) != default]
    if len(ordered) < 2:
        raise ValueError(
            "lucid.coreml: shapes must name at least one shape besides the "
            f"example's own {default}"
        )
    return ordered


def _bounds(
    default: tuple[int, ...], shape_range: dict[int, tuple[int, int]]
) -> list[tuple[int, int]]:
    """Each axis's ``(low, high)``, after checking the range admits the example."""
    for axis, (low, high) in sorted(shape_range.items()):
        if not 0 <= axis < len(default):
            raise ValueError(
                f"lucid.coreml: axis {axis} is outside the input's rank {len(default)}"
            )
        if low > high:
            raise ValueError(f"lucid.coreml: axis {axis} has a range of ({low}, {high})")
        if not low <= default[axis] <= high:
            raise ValueError(
                f"lucid.coreml: the example's axis {axis} is {default[axis]}, "
                f"outside the range ({low}, {high}) it is meant to sit in"
            )
    return [shape_range.get(axis, (size, size)) for axis, size in enumerate(default)]


def _walk(
    ours: list[tuple[bool, object]], theirs: list[tuple[bool, object]], mine: range, other: range
) -> dict[int, int]:
    """Pair axes along one direction until the two shapes part ways."""
    pairs: dict[int, int] = {}
    left, right = iter(mine), iter(other)
    i, j = next(left, None), next(right, None)
    unit = (False, 1)
    while i is not None and j is not None:
        if ours[i] == theirs[j] and ours[i][1] is not None:
            pairs[i] = j
            i, j = next(left, None), next(right, None)
        elif ours[i] == unit:
            i = next(left, None)
        elif theirs[j] == unit:
            j = next(right, None)
        else:
            break
    return pairs


class Record(NamedTuple):
    """One probe trace, reduced to what :func:`compare` reads.

    Kept apart from the graph so the comparison is plain data, testable
    without tracing anything.
    """

    ops: list[tuple[str, dict[str, object], list[tuple[int, ...]]]]
    ids: list[int]
    input_id: int
    signature: list[tuple[str, tuple[int, ...]]]
    #: Every external feed but the input, to its shape.
    feeds: dict[int, tuple[int, ...]]
    #: The graph changed between two calls at this one shape.
    unsettled: bool = False


def compare(
    records: list[Record], probes: list[tuple[int, ...]], skip_attrs: frozenset[str]
) -> Flexibility:
    """Which axes of which values changed between the probe traces.

    Parameters
    ----------
    records : list of Record
        One per probe, the default shape first.
    probes : list of tuple of int
        The input shape each record was traced at.
    skip_attrs : frozenset of str
        Operations whose attributes differ between traces for a reason
        that is not the size — a draw records its seed.

    Raises
    ------
    ShapeNotFlexible
        The graph or an operation's attributes changed with the shape.
    """
    base = records[0]
    for other in records[1:]:
        _agree(base, other, skip_attrs)

    found = Flexibility(input_id=base.input_id, signature=base.signature)
    # The input is an external feed rather than an operation's result, so
    # the walk below never reaches it; its open axes are the ones the
    # probes disagree on.
    found.varying[base.input_id] = {
        axis for shape in probes[1:] for axis in range(len(shape)) if shape[axis] != probes[0][axis]
    }
    found.probes[base.input_id] = list(probes)
    position = 0
    for index, (name, _attrs, base_shapes) in enumerate(base.ops):
        for output, base_shape in enumerate(base_shapes):
            value_id = base.ids[position]
            position += 1
            seen = [record.ops[index][2][output] for record in records]
            if any(len(shape) != len(base_shape) for shape in seen):
                raise ShapeNotFlexible(name, "its result changes rank with the input size")
            axes = {
                axis
                for shape in seen[1:]
                for axis, (a, b) in enumerate(zip(base_shape, shape, strict=True))
                if a != b
            }
            if axes:
                found.varying[value_id] = axes
                found.probes[value_id] = seen
    return found


#: Attributes that restate the size of an operation's result: the cut
#: points of a slice (``pos_table[:, :T]``), the output size of a resample.
#: They change with the input exactly when the result's shape does, and the
#: emitter reads that shape instead — through :class:`Dims`, which refuses
#: an open axis the emitter does not handle — so a change in one of them is
#: not by itself a refusal (LCD-365).
SIZE_ATTRS: dict[str, frozenset[str]] = {
    "split_at": frozenset({"indices"}),
    "interpolate_nearest_2d": frozenset({"H_out", "W_out"}),
    "interpolate_bilinear": frozenset({"H_out", "W_out"}),
}


def _agree(base: Record, other: Record, skip_attrs: frozenset[str]) -> None:
    """Refuse unless two probes traced the same graph with the same settings."""
    if [n for n, _a, _s in other.ops] != [n for n, _a, _s in base.ops]:
        cache = (
            " — and at one shape the graph changed between its first call and "
            "the next, which is a cache the model fills on first use"
            if base.unsettled or other.unsettled
            else ""
        )
        raise ShapeNotFlexible(
            "<graph>", f"the traced operations themselves differ between shapes{cache}"
        )
    if other.ids != base.ids:
        raise ShapeNotFlexible("<graph>", "the traced values differ between shapes")
    if other.feeds != base.feeds:
        # A tensor the model made outside what the trace records — an
        # ``arange(x.shape[0])`` read back as a buffer — goes into the
        # package as a constant, at one of its sizes.
        raise ShapeNotFlexible(
            "<constant>",
            "a tensor that enters the graph as a constant has the input's "
            "size, so the package would hold it at the traced size",
        )
    for (name, attrs, _shape), (_n, other_attrs, _s) in zip(base.ops, other.ops, strict=True):
        if name in skip_attrs or attrs == other_attrs:
            continue
        changed = sorted(
            key
            for key in set(attrs) | set(other_attrs)
            if attrs.get(key) != other_attrs.get(key) and key not in SIZE_ATTRS.get(name, ())
        )
        if changed:
            raise ShapeNotFlexible(name, f"its {', '.join(changed)} changed with the input size")


class _Value(NamedTuple):
    id: int
    shape: Dims
    dtype: object


class GuardedOp:
    """A traced operation as an emitter sees it under a flexible export.

    Its results' shapes are :class:`Dims`, so an emitter that reads a
    result's shape directly — rather than through the builder — cannot
    take the traced size of an open axis either.
    """

    __slots__ = ("attrs", "name", "outputs")

    def __init__(self, op: _Traced, varying: dict[int, set[int]]) -> None:
        self.name = str(op.name)
        self.attrs = op.attrs
        self.outputs = [
            _Value(
                int(out.id), Dims(flex(out.shape, varying.get(int(out.id))), self.name), out.dtype
            )
            for out in op.outputs
        ]


class _TracedValue(Protocol):
    id: int
    shape: Sequence[int]
    dtype: object


class _Traced(Protocol):
    name: str
    attrs: dict[str, object]

    @property
    def outputs(self) -> Sequence[_TracedValue]: ...


class _Graph(Protocol):
    @property
    def ops(self) -> Sequence[_Traced]: ...


def same_graph(found: Flexibility, graph: _Graph, input_id: int) -> bool:
    """Whether ``graph`` is the one the probes traced, value for value.

    Parameters
    ----------
    found : Flexibility
        What the probes recorded.
    graph : TraceGraph
        The graph about to be emitted.
    input_id : int
        Its input's identifier.
    """
    return found.signature == signature(graph) and found.input_id == int(input_id)
