"""
lucid.compile._core.call_values — one owner for "which segment made this value".

A segmented step traces each segment on its own and runs its executables
afterwards.  A value a segment reads from outside its arguments and its model's
state is a constant of that executable: its identity is fed on every run, and
whatever put it there — a ``box["v"] = x`` in an earlier segment's forward, a
module attribute, a closure cell, a list — runs only while that segment is
traced, never in a compiled call.  Read from a later segment, such a value is
the first call's, for good, and the gradient that should flow back through it
to the segment that made it is cut (LCD-902).

A tracer cannot tell it apart from a constant the caller built once: it sees a
pointer it has no op for, and the value's node (if it has one) is older than
the tracer.  The step can: it traced the segment that made the value.
:class:`CallValues` holds, weakly, every tensor the call's traces made and the
inputs its segments were handed, and names the first one that a segment other
than its maker reads.  Weakly, because a segment's trace owns every
intermediate of the segment and holding them past the trace would keep a whole
forward alive; only a tensor something else still holds can be read again, and
that one is alive as long as the holder is.
"""

import weakref
from collections.abc import Iterable
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from lucid._C import engine as _C_engine
    from lucid._C.engine.compile import Tracer

__all__ = ["CallValues"]


@final
class CallValues:
    """The tensors one segmented call made, by the segment that made each.

    Built per call and filled only while a segment compiles: a call that runs
    cached executables never traces, so it costs nothing.  The key is the
    identity of the Python object wrapping a tensor, which a tensor held in
    Python keeps for as long as it is held (the same object comes back from
    the engine on every read), and an entry whose tensor died never matches.
    """

    __slots__ = ("_by_id",)

    def __init__(self) -> None:
        # id of the wrapper -> (weak reference to it, the segment that made it)
        self._by_id: dict[int, tuple[weakref.ReferenceType[_C_engine.TensorImpl], int]] = {}

    def hand(self, segment: int, impls: Iterable[_C_engine.TensorImpl]) -> None:
        """Record ``impls`` as the input ``segment`` was handed.

        Parameters
        ----------
        segment : int
            The segment whose argument they are.
        impls : iterable of TensorImpl
            The tensors handed to it.
        """
        for impl in impls:
            self._by_id.setdefault(id(impl), (weakref.ref(impl), segment))

    def made_by(self, segment: int, tracer: Tracer) -> None:
        """Record what ``segment``'s finished trace made: every value it holds
        but was not fed.

        Only those something else still holds outlive this call
        (:meth:`foreign_read`).

        Parameters
        ----------
        segment : int
            The segment that was traced.
        tracer : Tracer
            Its finished trace.
        """
        feeds = tracer.external_feeds
        fed = {id(impl) for impl in feeds.values()}
        made = [impl for impl in tracer.retained_values.values() if id(impl) not in fed]
        self.hand(segment, made)
        del made, feeds
        self._by_id = {key: entry for key, entry in self._by_id.items() if entry[0]() is not None}

    def foreign_read(self, segment: int, tracer: Tracer) -> int | None:
        """The segment that made a value ``segment``'s finished trace was fed from outside.

        Parameters
        ----------
        segment : int
            The segment that was traced.
        tracer : Tracer
            Its finished trace.

        Returns
        -------
        int or None
            The maker of the first feed that another segment made or was
            handed, or ``None`` when every feed is the segment's own or came
            from before the call.
        """
        for impl in tracer.external_feeds.values():
            entry = self._by_id.get(id(impl))
            if entry is not None and entry[0]() is impl and entry[1] != segment:
                return entry[1]
        return None
