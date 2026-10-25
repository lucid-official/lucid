"""
lucid.compile._core.trace_context — whether a trace recording is active.

This is one primitive of the bigger ``trace_context`` module the
trace-autograd ADR (``arch-compile-trace-autograd``, cards E1/E3) plans:
a single owner for the entry points' ``no_grad()`` + :func:`_tracing`
scaffolding.  That consolidation is its own cards; this module holds
only what LCD-626 needs today, under the name the ADR already commits
to so E1/E3 can grow it in place rather than duplicate it.
"""

from lucid._C import engine as _C_engine

__all__ = ["recording_active"]


def recording_active() -> bool:
    """Whether a :class:`Tracer` is installed on the calling thread right now.

    True for the whole body of any entry point built on
    :func:`lucid.compile._tracing` — a :class:`CompiledModule`'s forward
    trace, :func:`make_step`, :func:`fused_step`, :func:`compiled_step`,
    the segmented/decode step tracers, ``compile_optimizer``'s probe
    trace, ``diagnose``.

    A compiled callable invoked while this is ``True`` is nested inside
    someone else's recording.  It must not run its own cached executable
    (the result would enter the outer trace as a tensor no recorded op
    produced — an opaque external feed the outer compile pins at its
    trace-time value, so every later call answers with the first one's
    result, silently — LCD-626) and it must not install a tracer of its
    own either (:func:`lucid.compile._tracing` detaches unconditionally
    on exit, which would drop the outer recording).  See
    :func:`lucid.compile._entry.module.CompiledModule._dispatch_call`
    for the inline-eager fix this predicate drives.
    """
    return _C_engine.compile.current_tracer() is not None
