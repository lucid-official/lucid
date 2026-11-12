"""Scoped thread-local engine state helpers (tracer suspension and refusal)."""

from collections.abc import Iterator
from contextlib import contextmanager

from lucid._C import engine as _C_engine


def refuse_trace(why: str) -> None:
    """Mark an active ``lucid.compile`` trace as one no executable can stand for.

    The one owner of that verdict: a call the trace cannot stand for (a backward
    pass, a read of the grad-mode flag, a version counter, a router's prototypes)
    says why here, every compile entry point then refuses the trace, and the call
    runs eagerly.  Outside a trace this does nothing.
    """
    tracer = _C_engine.compile.current_tracer()
    if tracer is not None:
        tracer.mark_unsupported(why)


@contextmanager
def untraced() -> Iterator[None]:
    """Temporarily detach the active ``lucid.compile`` tracer on this thread.

    Operations executed inside this ``with`` block run eagerly without being
    recorded into the active ``Tracer`` DAG, and the previous tracer is
    restored on exit even if an exception is raised.
    """
    saved = _C_engine.compile.current_tracer()
    if saved is None:
        yield
        return
    _C_engine.compile.set_current_tracer(None)
    try:
        yield
    finally:
        _C_engine.compile.set_current_tracer(saved)
