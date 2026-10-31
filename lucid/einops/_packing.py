"""``pack`` and ``unpack`` — several tensors in one, and back.

``pack`` joins tensors of different rank along one axis the pattern marks
with ``*``: the axes before and after the star must agree, the axes the
star covers are flattened and the results are concatenated.  What the star
covered is returned beside the packed tensor, and ``unpack`` takes it back
to split the packed axis and give each part its shape again.  Nothing is
copied but the concatenation.

The pattern is the reference einops grammar for these two calls: a
space-separated list of axis names with exactly one ``*``.  The names say
how many axes come before and after the star, nothing more, so they are
validated like any axis name but never matched against a shape.
"""

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import lucid
from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid.einops._recipe import axis_name_problem

_STAR = "*"


@dataclass(frozen=True, slots=True)
class _StarPattern:
    """Where the packed axis sits: ``before`` axes ahead of it, ``after`` behind."""

    before: int
    after: int

    @property
    def fixed(self) -> int:
        """How many axes a tensor must have besides the star's."""
        return self.before + self.after


@functools.lru_cache(maxsize=128)
def _star_pattern(pattern: str, operation: str) -> _StarPattern:
    where = f"einops.{operation}(..., {pattern!r})"
    names = pattern.split()
    if len(set(names)) != len(names):
        raise _C_engine.InvalidArgument(f"{where}: an axis name appears more than once")
    if _STAR not in names:
        raise _C_engine.InvalidArgument(f"{where}: the pattern has no '{_STAR}' axis")
    for name in names:
        problem = None if name == _STAR else axis_name_problem(name)
        if problem is not None:
            raise _C_engine.InvalidArgument(
                f"{where}: {name!r} is not a valid axis name: {problem}"
            )
    star = names.index(_STAR)
    return _StarPattern(before=star, after=len(names) - star - 1)


def pack(tensors: Sequence[Tensor], pattern: str) -> tuple[Tensor, list[tuple[int, ...]]]:
    """Join tensors of different rank into one along the axis a ``*`` marks.

    Parameters
    ----------
    tensors : sequence of Tensor
        The tensors to pack.  They agree on every axis the pattern names
        besides ``*`` and may differ in what ``*`` covers.
    pattern : str
        Space-separated axis names with one ``*``, for example
        ``"b seq * c"``.

    Returns
    -------
    tuple of (Tensor, list of tuple of int)
        The packed tensor and, for each input, the shape of the axes the
        star covered there (empty if it covered none).

    Raises
    ------
    lucid._C.engine.InvalidArgument
        The pattern is malformed or no tensor is given.
    lucid._C.engine.ShapeMismatch
        A tensor has fewer axes than the pattern names, or the axes around
        the star differ between tensors.
    TypeError
        An element of ``tensors`` is not a tensor.

    Examples
    --------
    >>> import lucid
    >>> parts = [lucid.zeros(2, 5), lucid.zeros(2, 3, 5), lucid.zeros(2, 3, 4, 5)]
    >>> packed, shapes = lucid.einops.pack(parts, "n * c")
    >>> packed.shape
    (2, 16, 5)
    >>> shapes
    [(), (3,), (3, 4)]
    """
    star = _star_pattern(pattern, "pack")
    if len(tensors) == 0:
        raise _C_engine.InvalidArgument(f"einops.pack(..., {pattern!r}): no tensors to pack")
    flattened: list[Tensor] = []
    covered: list[tuple[int, ...]] = []
    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"einops.pack: tensors[{i}] is a {type(tensor).__name__}, not a Tensor")
        shape = tensor.shape
        if len(shape) < star.fixed:
            raise _C_engine.ShapeMismatch(
                f"einops.pack(..., {pattern!r}): tensors[{i}] has shape {shape}, "
                f"while the pattern needs at least {star.fixed} axes"
            )
        stop = len(shape) - star.after
        covered.append(tuple(shape[star.before : stop]))
        flattened.append(
            tensor.reshape(
                (*shape[: star.before], math.prod(shape[star.before : stop]), *shape[stop:])
            )
        )
    return lucid.cat(flattened, dim=star.before), covered


def _lengths(packed_shapes: Sequence[Sequence[int]], pattern: str) -> list[int]:
    """Each part's length along the packed axis; ``-1`` for the one to infer."""
    lengths: list[int] = []
    for i, part in enumerate(packed_shapes):
        if any(n < -1 for n in part) or list(part).count(-1) > 1:
            raise _C_engine.InvalidArgument(
                f"einops.unpack(..., {pattern!r}): packed_shapes[{i}] = {tuple(part)} is not a "
                "shape (at most one -1, no other negative length)"
            )
        lengths.append(-1 if -1 in part else math.prod(part))
    if lengths.count(-1) > 1:
        raise _C_engine.InvalidArgument(
            f"einops.unpack(..., {pattern!r}): more than one -1 in {list(packed_shapes)}, "
            "so the lengths cannot be inferred"
        )
    return lengths


def _fill_unknown(
    lengths: list[int], packed_shapes: Sequence[Sequence[int]], total: int, pattern: str
) -> list[tuple[int, ...]]:
    """Settle the part with a ``-1`` and return every part's final shape."""
    known = sum(n for n in lengths if n != -1)
    unknown = lengths.index(-1) if -1 in lengths else None
    if unknown is None and known != total:
        raise _C_engine.ShapeMismatch(
            f"einops.unpack(..., {pattern!r}): the parts hold {known} along the packed axis, "
            f"the tensor has {total}"
        )
    shapes = [tuple(part) for part in packed_shapes]
    if unknown is not None:
        rest = math.prod(n for n in packed_shapes[unknown] if n != -1)
        left = total - known
        if left < 0 or rest == 0 or left % rest != 0:
            raise _C_engine.ShapeMismatch(
                f"einops.unpack(..., {pattern!r}): cannot split a packed axis of {total} into "
                f"{list(packed_shapes)}"
            )
        shapes[unknown] = tuple(left // rest if n == -1 else n for n in packed_shapes[unknown])
    return shapes


def unpack(tensor: Tensor, packed_shapes: Sequence[Sequence[int]], pattern: str) -> list[Tensor]:
    """Split a tensor ``pack`` made back into the tensors it joined.

    Parameters
    ----------
    tensor : Tensor
        The packed tensor.
    packed_shapes : sequence of sequence of int
        What the star covered in each output, as ``pack`` returned it.  One
        shape may hold a single ``-1``, which is inferred.
    pattern : str
        The pattern ``pack`` was called with.

    Returns
    -------
    list of Tensor
        One tensor per entry of ``packed_shapes``.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        The pattern is malformed or ``packed_shapes`` is not a list of shapes
        with at most one ``-1`` among them.
    lucid._C.engine.ShapeMismatch
        ``tensor`` does not have the rank the pattern names, or the parts do
        not add up to its packed axis.

    Examples
    --------
    >>> import lucid
    >>> packed = lucid.zeros(2, 16, 5)
    >>> [t.shape for t in lucid.einops.unpack(packed, [(), (3,), (3, -1)], "n * c")]
    [(2, 5), (2, 3, 5), (2, 3, 4, 5)]
    """
    star = _star_pattern(pattern, "unpack")
    shape = tensor.shape
    if len(shape) != star.fixed + 1:
        raise _C_engine.ShapeMismatch(
            f"einops.unpack(..., {pattern!r}): the input has shape {shape}, "
            f"while the pattern names {star.fixed + 1} axes"
        )
    parts = _fill_unknown(
        _lengths(packed_shapes, pattern), packed_shapes, shape[star.before], pattern
    )
    head, tail = shape[: star.before], shape[star.before + 1 :]
    leading = (slice(None),) * star.before
    out: list[Tensor] = []
    start = 0
    for part in parts:
        stop = start + math.prod(part)
        out.append(tensor[(*leading, slice(start, stop))].reshape((*head, *part, *tail)))
        start = stop
    return out


__all__ = ["pack", "unpack"]
