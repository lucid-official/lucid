"""
lucid.compile._core.rng_replay — one owner for "a recompute draws what its first run drew".

A compiled run draws its random values anew from the default stream
(``lucid/_C/compile/RngFeeds.h``).  An entry point that runs the same
forward twice — a segmented step's backward executable recomputes its
segment's forward — would draw twice, and the second draw would differ:
a dropout's backward differentiating another mask than the one its forward
applied, and the stream a step further along than eager's.  The recompute
instead runs :func:`replayed` from the position its forward started at: it
draws the forward's values, and the stream is put back where it stood.
This is :func:`lucid.autograd.checkpoint`'s ``preserve_rng_state``, for the
streams a compile restores (:func:`~lucid.compile._core.trace_context.compile_attempt`).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from lucid._factories._generator import _restore_stream_states, _stream_states

if TYPE_CHECKING:
    from lucid._C import engine as _C_engine
    from lucid._factories._generator import _StreamState

__all__ = ["advance_since", "replayed"]


def advance_since(before: tuple[_StreamState, ...]) -> dict[_C_engine.Generator, int]:
    """Blocks each stream drew since ``before`` was read, the ones that drew.

    ``before`` is what :func:`~lucid._factories._generator._stream_states`
    returned.  Empty when the code in between drew nothing, whatever it was.
    """
    start = {stream: counter for stream, _, counter in before}
    drew: dict[_C_engine.Generator, int] = {}
    for stream, _, counter in _stream_states():
        moved = counter - start.get(stream, counter)
        if moved:
            drew[stream] = moved
    return drew


@contextmanager
def replayed(states: tuple[_StreamState, ...] | None) -> Iterator[None]:
    """Run the body with every stream where ``states`` read it, and put them back.

    ``None`` runs the body as it is: a recompute of a forward that drew
    nothing has nothing to replay.  The streams end where they stood on
    entry, also when the body raises — the replay is not a draw of the caller's.
    """
    if states is None:
        yield
        return
    stood = _stream_states()
    _restore_stream_states(states)
    try:
        yield
    finally:
        _restore_stream_states(stood)
