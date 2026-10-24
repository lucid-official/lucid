"""The dim rule on the Python side: the owner of the engine's ``wrap_dim``.

Every dim argument the engine takes goes through its ``wrap_dim``: a tensor
of rank ``n`` answers to ``[-max(n, 1), max(n, 1))``, so a 0-d tensor has
one notional axis, named by both 0 and -1.  A Python wrapper that reads
``shape[dim]``, builds a permutation, or checks a dim list before the
engine sees it has to resolve the dim by the same rule.  Copies of it that
used the raw rank indexed past a 0-d shape (``list index out of range``) or
let an axis past the last one through.  This module owns the rule on the
Python side: :func:`_to_axes` reads a dim spelling, :func:`_wrap_dim` and
:func:`_wrap_dims` resolve it, :func:`_dim_size` reads the axis's extent.
A new Python axis path takes its dims here.  :func:`_kept_sizes` reads the
``-1`` of a target size the same way, aligned from the right.
"""

import operator
from collections.abc import Sequence
from typing import SupportsIndex

from lucid._C import engine as _C_engine


def _to_axes(dim: object, name: str, what: str = "dim") -> list[int]:
    """``dim`` — ``None``, an index or a sequence of them — as engine axes.

    ``what`` names the argument in the refusal (``roll`` reads its
    ``shifts`` here too).  A bool is refused, as the reference refuses it:
    ``dim=True`` used to pass as axis 1.
    """
    if dim is None:
        return []
    if type(dim) is int:
        return [dim]
    # Plain ints in a tuple or list skip the protocol checks below, which
    # cost more than the reduction they guard.
    if type(dim) is tuple or type(dim) is list:
        axes = list(dim)
        for axis in axes:
            if type(axis) is not int:
                break
        else:
            return axes
    try:
        if isinstance(dim, SupportsIndex) and not isinstance(dim, bool):
            return [operator.index(dim)]
        if (
            isinstance(dim, Sequence)
            and not isinstance(dim, str)
            and not any(isinstance(d, bool) for d in dim)
        ):
            return [operator.index(d) for d in dim]
    except TypeError:
        pass  # an element that is not an index; refused below with the whole dim
    raise TypeError(
        f"{name}(): {what} must be an int, a sequence of ints or None, got {dim!r}"
    )


def _wrap_dim(dim: int, ndim: int, op: str) -> int:
    """``dim`` counted from the front of a rank-``ndim`` tensor.

    A 0-d tensor's notional axis wraps to 0, which names no entry of its
    shape: read its extent with :func:`_dim_size`.

    Raises
    ------
    IndexError
        ``dim`` outside ``[-max(ndim, 1), max(ndim, 1))``.
    """
    extent = max(ndim, 1)
    wrapped = dim + extent if dim < 0 else dim
    if not 0 <= wrapped < extent:
        raise IndexError(
            f"{op}: axis out of range: got {dim}, expected "
            f"[{-extent}, {extent - 1}] for a tensor of rank {ndim}"
        )
    return wrapped


def _wrap_dims(dims: Sequence[int], ndim: int, op: str) -> list[int]:
    """Each of ``dims`` through :func:`_wrap_dim`, in order, none named twice.

    Raises
    ------
    IndexError
        Some dim is out of range.
    lucid._C.engine.InvalidArgument
        Two dims name the same axis (``0`` and ``-1`` of a 0-d tensor do).
    """
    wrapped = [_wrap_dim(d, ndim, op) for d in dims]
    if len(set(wrapped)) != len(wrapped):
        repeated = next(d for d in wrapped if wrapped.count(d) > 1)
        raise _C_engine.InvalidArgument(
            f"{op}(): dim {repeated} appears multiple times in the list of "
            f"dims {tuple(dims)}"
        )
    return wrapped


def _dim_size(shape: Sequence[int], wrapped: int) -> int:
    """The extent of the wrapped axis: 1 for a 0-d tensor's notional one."""
    return int(shape[wrapped]) if shape else 1


def _kept_sizes(shape: Sequence[int], size: Sequence[int], op: str) -> list[int]:
    """``size`` for ``expand`` / ``broadcast_to``, each ``-1`` the size it keeps.

    ``size`` aligns with ``shape`` from the right, so ``-1`` stands for the
    size of the dim it lines up with.  In a leading dim ``shape`` does not
    have it keeps nothing and is refused, as the reference refuses it —
    ``expand`` read it as a 1, and ``broadcast_to`` refused every ``-1``.
    """
    lead = len(size) - len(shape)
    for i, d in enumerate(size[: max(lead, 0)]):
        if d == -1:
            raise _C_engine.InvalidArgument(
                f"{op}: the size -1 isn't allowed in a leading, non-existing "
                f"dimension {i} (target size {list(size)}, input shape {list(shape)})"
            )
    return [int(shape[i - lead]) if d == -1 else d for i, d in enumerate(size)]


def _require_real_axis(ndim: int, message: str) -> None:
    """Refuse a 0-d tensor for an op that cuts its axis into pieces.

    A 0-d tensor's notional axis (:func:`_wrap_dim`) holds its one element,
    but there is nothing to cut it into: the reference refuses ``split``,
    ``chunk`` and ``tensor_split`` of one with a ``RuntimeError``.  It is
    asked after the dim is wrapped, so a dim past the notional axis stays
    an ``IndexError``, as every other op answers it.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        ``ndim`` is 0.
    """
    if not ndim:
        raise _C_engine.InvalidArgument(message)
