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

What a compiled run carries back is one plan (:class:`CarriedWrites`): those
writes and the statistics a batch norm handed out.

A write that lands in one of the call's own arguments, directly or through a
view of it, is not written back — the next call's arguments are other tensors
— and makes the call run eager, as does any entry point that does not write
back (:func:`refuse`).  So does a batch norm's statistic with no tensor to go
to: one kept in a view the call took, or in a tensor it made.
"""

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast, final

from lucid._C import engine as _C_engine
from lucid.compile._core.bn_runstats import BnWriteback, bn_statistics_lost, bn_writeback_targets

if TYPE_CHECKING:
    from lucid._C.engine.compile import Tracer
    from lucid.nn.module import Module

_OUTSIDE = (
    "the traced call wrote in place to a tensor it did not create; "
    "compiled, the write would happen once, at trace time"
)
_ARGUMENT = (
    "the traced call wrote to one of its own arguments; "
    "a compiled call cannot carry the write back to the next call's"
)
_LOST_STATISTICS = (
    "the traced call gave a batch norm running statistics it made or took a view of; "
    "a compiled call has no tensor to write the new values to"
)


class UncarriedWrite(RuntimeError):
    """A traced call's write has nowhere to go in a compiled run (:func:`_why_uncarried`)."""


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


def _why_uncarried(tracer: Tracer, arguments: Sequence[_C_engine.TensorImpl]) -> str | None:
    """Why a compiled run cannot carry this trace's writes, or ``None`` if it can.

    A statistic counts as a write: a batch norm given its statistics as arguments
    updates that call's, and a later run would write the first call's.
    """
    graph, ext = tracer.graph, tracer.external_feeds
    stats = bn_writeback_targets(graph, ext)
    written = _written(tracer)
    landing = [*(impl for _latest, impl in written), *(impl for _feed, _out, impl in stats)]
    if any(_lands_in_argument(impl, arguments) for impl in landing):
        return _ARGUMENT
    carried = {*(latest for latest, _impl in written), *(out for _feed, out, _impl in stats)}
    return _LOST_STATISTICS if bn_statistics_lost(graph, carried) else None


@final
@dataclass(frozen=True, slots=True)
class CarriedWrites:
    """What a compiled run writes back after it, in the order of :attr:`out_ids`.

    The statistics a traced batch norm handed out instead of writing them, a
    functional one's too, with the counters of the layers it ran; then each
    tensor from outside the trace that the call wrote in place.  An entry point
    asks its executable for :attr:`out_ids` as extra outputs and hands them to
    :meth:`apply` after every run.  Built in one place (:func:`carried_writes`):
    ``lucid.compile`` carried the in-place writes, and under ``no_grad`` a
    functional batch norm's statistics stayed where they were (LCD-875).

    ``lucid.compile`` and the monolithic ``make_step`` carry this plan, and only
    they.  A segmented step and ``compiled_step`` carry a :class:`BnWriteback` and
    :func:`refuse` any in-place write; ``fused_step``'s executable writes the
    statistics itself and carries the rest (:func:`require_outside_writes`).
    ``compiled_decode_step`` has no part in it.

    Attributes
    ----------
    bn : BnWriteback
        The batch norm statistics and counters.
    targets : tuple of TensorImpl
        The tensors written in place, each already put back to its value from
        before the trace.
    ids : tuple of int
        The trace id of each target's final value.
    """

    bn: BnWriteback = field(default_factory=BnWriteback)
    targets: tuple[_C_engine.TensorImpl, ...] = ()
    ids: tuple[int, ...] = ()

    @property
    def out_ids(self) -> list[int]:
        """The trace ids to ask the executable for, in :meth:`apply`'s order."""
        return [*self.bn.out_ids, *self.ids]

    def __len__(self) -> int:
        """How many extra outputs the executable returns for these."""
        return len(self.bn.out_ids) + len(self.ids)

    def apply(self, values: Sequence[_C_engine.TensorImpl]) -> None:
        """Write a run's values for :attr:`out_ids` back, and advance the counters.

        Parameters
        ----------
        values : sequence of TensorImpl
            The executable's outputs for :attr:`out_ids`, in order.
        """
        stats = len(self.bn.out_ids)
        self.bn.apply(values[:stats])
        write_back(list(self.targets), list(values[stats:]))


type _PutBack = list[tuple[_C_engine.TensorImpl, _C_engine.TensorImpl, int]]


class _HeldPutBack(threading.local):
    """What a refused trace put back of the tensors it made that Python still
    holds: ``(tensor, its value after the trace, its version once put back)``.

    ``pending`` while the compile attempt that put them back runs, ``ready``
    once it has ended normally, for the eager call that follows it
    (:func:`eager_rerun`).  An attempt that raises drops them: no other call
    may take them for its own.
    """

    def __init__(self) -> None:
        self.pending: _PutBack = []
        self.ready: _PutBack = []


_HELD = _HeldPutBack()


def begin_attempt() -> _PutBack:
    """Open a compile attempt's record of what it puts back.

    Returns
    -------
    list
        The enclosing attempt's record, for :func:`end_attempt` to give back:
        attempts nest.
    """
    outer, _HELD.pending, _HELD.ready = _HELD.pending, [], []
    return outer


def end_attempt(outer: _PutBack, *, finished: bool) -> None:
    """Close a compile attempt's record.

    Parameters
    ----------
    outer : list
        What :func:`begin_attempt` answered.
    finished : bool, keyword-only
        Whether the attempt ended normally: only then does the eager call that
        follows it take what it put back.
    """
    _HELD.ready = _HELD.pending if finished else []
    _HELD.pending = outer


#: ``(tensor, its value before the trace's write, its value after)``.
type _HeldWrite = tuple[_C_engine.TensorImpl, _C_engine.TensorImpl, _C_engine.TensorImpl]
type HeldWrites = list[_HeldWrite]


@dataclass(frozen=True)
class TraceAnswer[T]:
    """The trace-time call, standing as the call it traced.

    For a trace refused because the call wrote a tensor it made and Python keeps
    (:func:`held_writes`): run again eagerly, the call would read that tensor
    put back, and one that finds it already built (a cache filled from the
    model's state) answers from the empty one.  The trace ran the call once, in
    the caller's grad mode, so its result and its writes are the call's.
    """

    value: T


def stood_as_the_call[T](tracer: Tracer, model: Module, answer: T) -> TraceAnswer[T]:
    """The trace-time call's result as the call's, saying why it was not compiled.

    Parameters
    ----------
    tracer : Tracer
        The finished trace, refused by :func:`held_writes`.
    model : Module
        The traced model, whose batch norms :func:`complete_the_call` steps.
    answer : object
        What the trace-time call returned.

    Returns
    -------
    TraceAnswer
        ``answer``, for the entry point to hand back.
    """
    from lucid.compile._core.fallback import _fallback_notice

    complete_the_call(tracer, model)
    _fallback_notice("held-write", str(tracer.graph.unsupported))
    return TraceAnswer(answer)


def complete_the_call(tracer: Tracer, model: Module) -> None:
    """Do what the trace-time call left undone, once it stands as the call.

    A traced batch norm leaves its running statistics alone and hands the new
    values out (a functional one too), and its counter advances only after a
    compiled run.  Without both, the statistics miss this call for good.

    Parameters
    ----------
    tracer : Tracer
        The finished trace.
    model : Module
        The traced model, which owns the counters.
    """
    from lucid.compile._core.bn_runstats import advance_bn_counters, bn_counter_targets

    tracer.apply_buffer_updates()
    advance_bn_counters(bn_counter_targets(model, tracer))


def held_writes(tracer: Tracer, *, stands: bool) -> HeldWrites:
    """The tensors the call made, wrote from its inputs and Python still holds.

    Any of them marks the trace unsupported.  Their values are not changed, but
    without ``stands`` they leave the trace-time graph here: whatever runs next
    must not chain onto that call.  Call once, with
    the call's result dropped: a result is handed back on every run.

    Parameters
    ----------
    tracer : Tracer
        The finished trace.
    stands : bool, keyword-only
        Whether the caller lets the trace-time call stand as the call when there
        are any (:func:`stood_as_the_call`): then every such write counts.
        Without that only one that reads the tensor's own earlier value does,
        to be put back for an eager call.  An overwrite stays compiled, which
        is wrong for one that runs only on the first call, a buffer filled from
        the first batch (LCD-876): so every entry point that has the call's
        result stands, a segmented step's segment too.

    Returns
    -------
    HeldWrites
        ``(tensor, value before the trace's write, value after)`` for each.
    """
    kept = tracer.refuse_kept_writes(stands=stands)
    return [(impl, before, after) for impl, before, after in kept]


def _put_back(tracer: Tracer, held: Sequence[_HeldWrite]) -> None:
    """Undo the trace-time call's writes that whatever runs next applies itself."""
    for impl, before, _after in held:
        impl.copy_from(before)
    _HELD.pending = [(impl, after, int(impl.version)) for impl, _before, after in held]
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
    ready, _HELD.ready = _HELD.ready, []
    try:
        yield
    finally:
        for impl, after, version in ready:
            if int(impl.version) == version:
                impl.copy_from(after)


def require_outside_writes(
    tracer: Tracer, arguments: Sequence[_C_engine.TensorImpl], caller: str, held: HeldWrites
) -> tuple[list[int], list[_C_engine.TensorImpl]]:
    """Put the written tensors back; return their ids and impls or raise.

    For ``fused_step``, whose executable writes batch norm's statistics in
    place itself.

    Parameters
    ----------
    tracer : Tracer
        The finished trace.
    arguments : sequence of TensorImpl
        Every tensor impl the traced call was given.
    caller : str
        The entry point named in the error.
    held : HeldWrites
        What :func:`held_writes` answered: put back with the rest.

    Returns
    -------
    tuple of (list of int, list of TensorImpl)
        The trace id of each written tensor's final value, and the tensors in
        the same order, every one already put back.

    Raises
    ------
    UncarriedWrite
        A write landed in one of the call's own arguments, or a batch norm's
        statistic has no tensor to go to.
    """
    _put_back(tracer, held)
    why = _why_uncarried(tracer, arguments)
    if why is not None:
        raise UncarriedWrite(f"{caller}: {why}")
    written = _written(tracer)
    return [latest for latest, _impl in written], [impl for _latest, impl in written]


def carried_writes(
    tracer: Tracer, model: Module, arguments: Sequence[_C_engine.TensorImpl], held: HeldWrites
) -> CarriedWrites | str:
    """Put the written tensors back; return everything the executable must carry.

    Call with the traced call's result dropped: a tensor the call made and wrote
    that Python still holds marks the trace unsupported, and a result it is about
    to hand back on every run is not holding it over.

    Parameters
    ----------
    tracer : Tracer
        The finished trace.
    model : Module
        The traced model, which owns the batch norm counters.
    arguments : sequence of TensorImpl
        Every tensor impl the traced call was given.
    held : HeldWrites
        What :func:`held_writes` answered: put back with the rest.

    Returns
    -------
    CarriedWrites or str
        The batch norm statistics the trace handed out and the tensors it
        wrote in place, every one of them already put back to its value from
        before the trace.  A string instead when a write landed in one of its
        own arguments or a statistic has no tensor to go to: the reason it has
        to run eager.
    """
    _put_back(tracer, held)
    why = _why_uncarried(tracer, arguments)
    if why is not None:
        return why
    written = _written(tracer)
    return CarriedWrites(
        BnWriteback.of(model, tracer),
        tuple(impl for _latest, impl in written),
        tuple(latest for latest, _impl in written),
    )


def write_back(targets: list[object], values: list[object]) -> None:
    """Copy each executable output into the tensor it is the new value of.

    Parameters
    ----------
    targets : list of TensorImpl
        The tensors to write, in order (:attr:`CarriedWrites.targets`).
    values : list of TensorImpl
        The matching executable outputs.
    """
    for target, value in zip(targets, values, strict=True):
        dst = cast(_C_engine.TensorImpl, target)
        src = cast(_C_engine.TensorImpl, value)
        if src.dtype != dst.dtype:
            src = _C_engine.astype(src, dst.dtype)
        dst.copy_from(src)


def refuse(tracer: Tracer, held: Sequence[_HeldWrite] = ()) -> None:
    """Put the written tensors back and mark the trace unsupported if any were.

    For the entry points that do not write back: running eager is right, and
    compiled the write would be lost.  A batch norm's statistic with no tensor
    to go to marks it too: these carry the statistics of a feed, and only those.

    Parameters
    ----------
    tracer : Tracer
        The finished trace.
    held : sequence of (TensorImpl, TensorImpl, TensorImpl), optional
        What :func:`held_writes` answered, put back for the eager call that
        follows, when the trace did not stand as the call.  Empty without one
        (``compiled_step``, which traces every call afresh): the trace's
        writes are the call's, a tensor the call made and Python holds keeps
        them, whatever they read, and the next call's trace finds it from
        outside.
    """
    _put_back(tracer, held)
    if _written(tracer):
        tracer.mark_unsupported(_OUTSIDE)
    elif _why_uncarried(tracer, ()) is not None:
        tracer.mark_unsupported(_LOST_STATISTICS)
