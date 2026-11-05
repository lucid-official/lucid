"""In-place writes to tensors from outside a traced call.

A traced forward that updates a tensor it did not create — one of its model's
buffers (an exponential moving average, a step counter, spectral norm's power
iterate), or any other tensor it holds — records the write like any other op,
but nothing asked for the value it produced: the executable pruned it, and
every compiled call left the tensor where the trace-time call had put it.
The update happened once.

Batch norm was the one case handled, by a kernel that leaves its buffers
alone under a tracer and hands the new statistics to the executable as extra
outputs (``bn_runstats``).  An in-place op cannot do that — by the time the
tracer sees it the write has happened — so this path works after the fact.
The tracer follows each write (``Tracer::on_inplace_write``) and keeps a copy
of every outside tensor from before its first one.  What counts as outside is
the bytes a write lands in: a view the call takes of a buffer it did not create
is written like the buffer, and carrying the value back through that view
reaches everything reading those bytes.  After the trace:

* every written tensor is put back to its value from before the trace, so
  whatever runs next — the executable, or an eager fallback — applies the
  write once, as an eager call does;
* its final id is asked of the executable as an extra output;
* after every run that output is copied into the tensor.

A write that lands in one of the call's own arguments, directly or through a
view of it, is not written back — the next call's arguments are other tensors
— and makes the call run eager, as does any entry point that does not write
back (:func:`refuse`).
"""

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._C.engine.compile import Tracer

_OUTSIDE = (
    "the traced call wrote in place to a tensor it did not create; "
    "compiled, the write would happen once, at trace time"
)
_ARGUMENT = (
    "the traced call wrote in place to one of its own arguments; "
    "a compiled call cannot carry the write back to the next call's"
)


def _written(tracer: Tracer) -> list[tuple[int, _C_engine.TensorImpl]]:
    """``(final id, impl)`` for each outside tensor the trace wrote in place."""
    return [(int(tid), impl) for tid, impl in tracer.outside_writes()]


def _lands_in_argument(
    impl: _C_engine.TensorImpl, arguments: Sequence[_C_engine.TensorImpl]
) -> bool:
    """Whether a write to ``impl`` lands in a tensor the call was given: a view
    of an argument is the argument's bytes, whichever of them the call wrote."""
    family = impl.family_id()
    return any(a is impl or (family != 0 and a.family_id() == family) for a in arguments)


class _HeldPutBack(threading.local):
    """What the last refused trace put back of the tensors it made that Python
    still holds: ``(tensor, its value after the trace, its version once put
    back)``, for the eager call that follows (:func:`eager_rerun`)."""

    def __init__(self) -> None:
        self.pending: list[tuple[_C_engine.TensorImpl, _C_engine.TensorImpl, int]] = []


_HELD = _HeldPutBack()


def _put_back(tracer: Tracer) -> None:
    """Undo the trace-time call's writes that whatever runs next applies itself."""
    _HELD.pending = [
        (impl, after, int(impl.version)) for impl, after in tracer.refuse_kept_writes()
    ]
    tracer.restore_outside_writes()


@contextmanager
def eager_rerun() -> Iterator[None]:
    """Run the eager call that follows a refused trace.

    A tensor the trace made, wrote from its inputs, and left held by Python was
    put back so that the eager call writes it once.  An eager call that finds it
    already built — a cache filled from the model's state on the first call —
    writes nothing to it, and would leave it empty for good.  Each such tensor
    the call did not write gets the trace's values back.

    Yields
    ------
    None
        While the eager call runs.
    """
    pending, _HELD.pending = _HELD.pending, []
    try:
        yield
    finally:
        for impl, after, version in pending:
            if int(impl.version) == version:
                impl.copy_from(after)


def require_outside_writes(
    tracer: Tracer, arguments: Sequence[_C_engine.TensorImpl], caller: str
) -> tuple[list[int], list[_C_engine.TensorImpl]]:
    """Put the written tensors back; return their ids and impls or raise ``RuntimeError``."""
    _put_back(tracer)
    ids: list[int] = []
    targets: list[_C_engine.TensorImpl] = []
    for latest, impl in _written(tracer):
        if _lands_in_argument(impl, arguments):
            raise RuntimeError(f"{caller}: {_ARGUMENT}")
        ids.append(latest)
        targets.append(impl)
    return ids, targets


def outside_writes(
    tracer: Tracer, arguments: Sequence[_C_engine.TensorImpl]
) -> tuple[list[int], list[object]] | str:
    """Put the written tensors back; return what the executable must carry.

    Call with the traced call's result dropped: a tensor the call made and wrote
    that Python still holds marks the trace unsupported, and a result it is about
    to hand back on every run is not holding it over.

    Parameters
    ----------
    tracer : Tracer
        The finished trace.
    arguments : sequence of TensorImpl
        Every tensor impl the traced call was given.

    Returns
    -------
    tuple of (list of int, list of TensorImpl), or str
        The trace id holding each written tensor's final value, and the
        tensors in the same order — every one of them already put back to
        its value from before the trace.  A string instead when a write
        landed in one of its own arguments: the reason it has to run eager.
    """
    _put_back(tracer)
    ids: list[int] = []
    targets: list[object] = []
    for latest, impl in _written(tracer):
        if _lands_in_argument(impl, arguments):
            return _ARGUMENT
        ids.append(latest)
        targets.append(impl)
    return ids, targets


def write_back(targets: list[object], values: list[object]) -> None:
    """Copy each executable output into the tensor it is the new value of.

    Parameters
    ----------
    targets : list of TensorImpl
        What :func:`outside_writes` returned, in order.
    values : list of TensorImpl
        The matching executable outputs.
    """
    for target, value in zip(targets, values, strict=True):
        dst = cast(_C_engine.TensorImpl, target)
        src = cast(_C_engine.TensorImpl, value)
        if src.dtype != dst.dtype:
            src = _C_engine.astype(src, dst.dtype)
        dst.copy_from(src)


def refuse(tracer: Tracer) -> None:
    """Put the written tensors back and mark the trace unsupported if any were.

    For the entry points that do not write back: running eager is right, and
    compiled the write would be lost.

    Parameters
    ----------
    tracer : Tracer
        The finished trace.
    """
    _put_back(tracer)
    if _written(tracer):
        tracer.mark_unsupported(_OUTSIDE)
