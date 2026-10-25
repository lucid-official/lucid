"""
lucid.compile._core.trace_context — single owner for compile recordings and attempts.

See ADR ``arch-compile-trace-autograd`` (compile C'):

* :func:`recording` installs a :class:`Tracer` in the caller's grad mode (or an
  explicit ``grad_mode`` when requested) and, when the block exits normally,
  scans ``tracer.retained_values`` for tensors carrying gradient hooks
  (:meth:`~lucid.Tensor.register_hook`) or :meth:`~lucid.Tensor.retain_grad`
  and marks the trace unsupported so differentiating or forward entry points
  fall back to eager instead of silently skipping the hook or retained
  gradient.
* :func:`compile_attempt` snapshots the default RNG stream and every live
  :class:`~lucid.Generator` (:func:`~lucid._factories._generator._stream_states`)
  on entry and restores them on exit (LIFO across nested attempts), so a
  compile-on-miss — whether it succeeds and then runs the executable for the
  first time, or fails and runs eagerly — advances the user's RNG streams
  exactly once.
* :func:`is_tensor_hook_error` and :func:`notice_if_tensor_hook_error` provide
  one shared predicate and fallback notice for the ``RuntimeError`` raised when
  ``register_hook`` is called on a tensor that does not require grad during a
  trace.
* :func:`recording_active` reports whether a :class:`Tracer` is currently
  installed on the calling thread (LCD-626).
"""

from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._factories._generator import _restore_stream_states, _stream_states
from lucid.autograd._grad_mode import enable_grad, no_grad
from lucid.compile._core.fallback import _fallback_notice

if TYPE_CHECKING:
    from lucid._C.engine.compile import Tracer
    from lucid.nn.module import Module

__all__ = [
    "compile_attempt",
    "is_tensor_hook_error",
    "notice_if_tensor_hook_error",
    "recording",
    "recording_active",
]

_TENSOR_HOOK_REASON = (
    "the traced call registered a tensor gradient hook (register_hook), "
    "which a compiled graph cannot run"
)
_RETAIN_GRAD_REASON = (
    "the traced call called retain_grad() on an intermediate tensor, "
    "whose gradient a compiled graph does not materialise"
)


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


@contextmanager
def compile_attempt() -> Iterator[None]:
    """Snapshot and restore every RNG stream across a compile-on-miss attempt.

    A compile-on-miss traces the callable (which advances any RNG streams it
    draws from), optionally runs capability probes, and lowers the graph; the
    caller then executes the compiled graph (whose ``RngFeeds`` redraws from
    the default stream) or falls back to an eager call.  Restoring the stream
    states in ``finally`` ensures the first real execution sees the exact
    pre-call stream positions — LIFO when attempts nest.
    """
    saved = _stream_states()
    try:
        yield
    finally:
        _restore_stream_states(saved)


def _scan_retained_values(tracer: Tracer) -> None:
    """Mark ``tracer`` unsupported if any traced tensor has hooks or ``retains_grad``.

    Checked against ``tracer.retained_values`` (every op output recorded during
    the trace) as well as ``tracer.external_feeds`` (inputs and parameters
    watched or fed into the trace):

    * Any tensor with an active gradient hook (``_has_tensor_hooks``) cannot
      have its hook run inside a compiled executable.
    * Any non-leaf tensor created during the trace with ``retains_grad`` would
      have its ``.grad`` discarded by a compiled executable.
    """
    from lucid.autograd._hooks import _HookRunner

    feeds = tracer.external_feeds
    feed_ids = {id(impl) for impl in feeds.values()}
    retained = tracer.retained_values
    for impl in (*feeds.values(), *retained.values()):
        if _C_engine._has_tensor_hooks(impl):
            runner = _C_engine._tensor_hook_runner(impl, _HookRunner)
            if getattr(runner, "hooks", True):
                tracer.mark_unsupported(_TENSOR_HOOK_REASON)
                return
    for impl in retained.values():
        if impl.retains_grad and not impl.is_leaf and id(impl) not in feed_ids:
            tracer.mark_unsupported(_RETAIN_GRAD_REASON)
            return


@contextmanager
def recording(
    watch: Iterable[object] = (),
    module: Module | None = None,
    *,
    fold_constants: bool = False,
    grad_mode: bool | None = None,
) -> Iterator[Tracer]:
    """Record a trace in the caller's grad mode (or ``grad_mode`` if specified).

    On normal exit of the ``with`` block, scans ``tracer.retained_values`` for
    tensor gradient hooks and ``retain_grad()`` and marks the trace
    unsupported when either is present.

    Parameters
    ----------
    watch : iterable of object, optional
        Tensors (or nested containers of tensors) to watch for host reads.
    module : Module or None, optional
        Module whose parameters and buffers are watched for host reads.
    fold_constants : bool, optional
        Forwarded to :func:`lucid.compile._tracing`.
    grad_mode : bool or None, optional
        ``None`` (default) keeps the caller's current grad mode; ``True``
        enters :func:`enable_grad`; ``False`` enters :func:`no_grad`.

    Yields
    ------
    Tracer
        The active tracer for this recording.
    """
    from lucid.compile import _tracing

    mode_ctx: AbstractContextManager[object]
    if grad_mode is None:
        mode_ctx = nullcontext()
    elif grad_mode:
        mode_ctx = enable_grad()
    else:
        mode_ctx = no_grad()

    with (
        mode_ctx,
        _tracing(watch=watch, module=module, fold_constants=fold_constants) as tracer,
    ):
        yield tracer
        _scan_retained_values(tracer)


def is_tensor_hook_error(exc: BaseException) -> bool:
    """Whether ``exc`` is the ``RuntimeError`` from ``register_hook`` on an ungrad tensor."""
    return isinstance(exc, RuntimeError) and "register_hook" in str(exc)


def notice_if_tensor_hook_error(exc: BaseException) -> bool:
    """If ``exc`` is a ``register_hook`` error, emit the ``tensor-hook`` notice and return ``True``."""
    if not is_tensor_hook_error(exc):
        return False
    _fallback_notice("tensor-hook", str(exc))
    return True
