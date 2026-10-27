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
of every outside tensor from before its first one.  After the trace:

* every written tensor is put back to its value from before the trace, so
  whatever runs next — the executable, or an eager fallback — applies the
  write once, as an eager call does;
* its final id is asked of the executable as an extra output;
* after every run that output is copied into the tensor.

A write to one of the call's own arguments is not written back — the next
call's arguments are other tensors — and makes the call run eager, as does
any entry point that does not write back (:func:`refuse`).
"""

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


def _written(tracer: Tracer) -> list[tuple[int, object]]:
    """``(final id, impl)`` for each outside tensor the trace wrote in place."""
    return [(int(tid), impl) for tid, impl in tracer.outside_writes()]


def outside_writes(tracer: Tracer, arguments: set[int]) -> tuple[list[int], list[object]] | str:
    """Put the written tensors back; return what the executable must carry.

    Parameters
    ----------
    tracer : Tracer
        The finished trace.
    arguments : set of int
        ``id`` of every tensor impl the traced call was given.

    Returns
    -------
    tuple of (list of int, list of TensorImpl), or str
        The trace id holding each written tensor's final value, and the
        tensors in the same order — every one of them already put back to
        its value from before the trace.  A string instead when the call
        wrote to one of its own arguments: the reason it has to run eager.
    """
    tracer.restore_outside_writes()
    ids: list[int] = []
    targets: list[object] = []
    for latest, impl in _written(tracer):
        if id(impl) in arguments:
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
    tracer.restore_outside_writes()
    if _written(tracer):
        tracer.mark_unsupported(_OUTSIDE)
