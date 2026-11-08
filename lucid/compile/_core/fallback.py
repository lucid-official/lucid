"""
lucid.compile._core.fallback — Phase 1.4 eager-escape helpers.

When :class:`CompiledModule` cannot compile a given signature (a
forward that touches an op with no emitter, or that breaks an
invariant of the MPSGraph builder), it must fall back to the regular
eager forward.  This module factors the bookkeeping out of
:class:`CompiledModule` so the wrapper stays focused on the cache /
dispatch logic.

Three responsibilities:

1. Provide a :func:`run_eager` helper that calls ``model(*args,
   **kwargs)`` exactly the way the user would (no special arguments,
   no monkey-patching).
2. Maintain a per-:class:`CompiledModule` *blacklist* of signatures
   that already failed to compile, so we don't re-trace + re-attempt
   on every call.
3. Say that a fallback happened (:func:`_fallback_notice`), once per
   kind of reason, so a caller who asked for a compiled model and got
   an eager one finds out without having to know about
   ``LUCID_COMPILE_VERBOSE``.
"""

import os
import sys
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, cast

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor

if TYPE_CHECKING:
    from lucid.compile._core.signature import CacheKey
    from lucid.nn.module import Module

__all__ = [
    "run_eager",
    "EagerFallbackSet",
    "is_first_run_failure",
    "run_or_answer_eagerly",
    "off_gpu_detail",
    "refuse_off_gpu",
]


class CompileFallbackWarning(UserWarning):
    """A compiled call ran eagerly instead.

    Its own category so a caller can filter it — or turn it into an error
    in a test that must stay compiled — without touching every other
    ``UserWarning``.  Not exported from :mod:`lucid.compile`, which
    exports no warning classes; ``warnings.filterwarnings`` with a
    message pattern reaches it all the same.
    """


#: Reason categories already announced in this process.  A fallback is
#: correct, and a model that falls back falls back on every call, so a
#: warning per call would be noise; one per kind of reason is the news.
_NOTICED: set[str] = set()


def _fallback_notice(reason_category: str, detail: str) -> None:
    """Report that a compiled call is running eagerly, and why.

    Under ``LUCID_COMPILE_VERBOSE=1`` every fallback is printed, as the
    build is narrated there already.  Otherwise the first fallback of each
    category warns, and later ones — the same model on the next call, or
    another model hitting the same gap — stay quiet.

    Parameters
    ----------
    reason_category : str
        Short key for the kind of fallback (``"lowering"``,
        ``"first-run"``, ...); the once-per-process bookkeeping is keyed
        on it.
    detail : str
        The specific reason, as the engine or the check stated it.
    """
    if os.environ.get("LUCID_COMPILE_VERBOSE") == "1":
        print(f"[compile] eager fallback: {detail}", file=sys.stderr)
        return
    if reason_category in _NOTICED:
        return
    _NOTICED.add(reason_category)
    # Attributed to the first frame outside ``lucid.compile`` — the
    # caller's own call of the compiled model — however deep inside the
    # package the fallback was decided: a forward's first run, a lowering
    # and a training step reach this from different depths, so no one
    # fixed level names the caller's line for all of them.
    warnings.warn(
        f"lucid.compile: {detail}; running eagerly. The result is correct; set "
        "LUCID_COMPILE_VERBOSE=1 to see every fallback as it happens",
        CompileFallbackWarning,
        stacklevel=2,
        skip_file_prefixes=(_PACKAGE_ROOT,),
    )


def is_first_run_failure(why: BaseException) -> bool:
    """Whether ``why`` is the engine failing an executable's first run.

    MPSGraph specializes an executable inside its first run, and has gaps
    there that compiling does not reveal.  A failure of that kind is the same
    verdict as a compile that declines: the run wrote nothing yet, so the call
    is answered eagerly and its signature remembered.  Every entry point that
    can answer eagerly asks this one question of the error it caught, rather
    than matching the engine's message itself.

    Parameters
    ----------
    why : BaseException
        What the first ``run_executable*`` of an entry raised.

    Returns
    -------
    bool
        ``True`` when the engine reported the run itself failing.
    """
    return "MPSGraph failed" in str(why)


def run_or_answer_eagerly[T](
    run: Callable[[], T], *, warm: bool, drop: Callable[[], None], eagerly: Callable[[], T]
) -> T:
    """Run a compiled entry, answering eagerly if its first run is the engine's to fail.

    A first run writes nothing (the carried writes are applied only after a run
    that succeeded), so an eager rerun is the answer a compile that declined
    would have given: say so once, forget the executable, and let ``eagerly``
    answer.  Anything else the run raises, and any failure of a ``warm`` entry
    (one that has run before), is not a verdict about the signature and
    propagates.

    Parameters
    ----------
    run : callable
        Executes the compiled entry; returns the call's result.
    warm : bool
        Whether the entry has run to the end before.
    drop : callable
        Forgets the entry and marks its signature eager-only.
    eagerly : callable
        The call, run eagerly.

    Returns
    -------
    T
        What ``run`` returned, or what ``eagerly`` did after a refused first run.
    """
    try:
        return run()
    except RuntimeError as why:
        if warm or not is_first_run_failure(why):
            raise
        _fallback_notice("first-run", str(why))
    drop()
    return eagerly()


def off_gpu_detail(tensors: Iterable[Tensor]) -> str | None:
    """Why ``tensors`` leave nothing to compile, or ``None`` when all are on Metal.

    The executables run on the GPU only, so a model whose parameters live
    on the CPU has no compiled path — every entry point that trains such a
    model owns the decision to run eagerly, or to refuse, from this one
    answer rather than from the engine's ``only Device::GPU`` error.

    Parameters
    ----------
    tensors : Iterable[Tensor]
        The parameters the step trains.

    Returns
    -------
    str or None
        The detail for :func:`_fallback_notice` (or an error message).
    """
    off = sum(1 for t in tensors if t._impl.device != _C_engine.Device.GPU)
    if off == 0:
        return None
    return f"{off} parameter(s) live on the cpu, and compiled execution needs Metal"


def refuse_off_gpu(tensors: Iterable[Tensor], owner: str) -> None:
    """Raise the one error for an entry point that has no eager step to fall back to.

    ``compile_optimizer``'s wrapper owns the optimizer's state, so a CPU step
    would split it between two owners; it refuses, before any state moves.

    Parameters
    ----------
    tensors : Iterable[Tensor]
        The parameters the step would update.
    owner : str
        The entry point named in the message.

    Raises
    ------
    RuntimeError
        Some ``tensors`` live on the CPU; the message names Metal.
    """
    if (off_gpu := off_gpu_detail(tensors)) is not None:
        raise RuntimeError(
            f"{owner}: {off_gpu}. Move the model to Metal (`model.to('metal')`) "
            "before building the optimizer, or keep the eager optimizer for CPU training."
        )


def _narrate(message: str) -> None:
    """Print a step of the build under ``LUCID_COMPILE_VERBOSE=1`` only.

    For a failure that is not yet a fallback — a symbolic lowering that
    the per-shape one is about to retry — which a warning would misreport
    as the call running eagerly.

    Parameters
    ----------
    message : str
        What happened, without the ``[compile]`` prefix.
    """
    if os.environ.get("LUCID_COMPILE_VERBOSE") == "1":
        print(f"[compile] {message}", file=sys.stderr)


#: Directory of :mod:`lucid.compile`, whose frames a fallback warning
#: skips on its way to the caller's.
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def run_eager(model: Module, args: tuple[object, ...], kwargs: dict[str, object]) -> object:
    """Invoke ``model(*args, **kwargs)`` on the eager path.

    Thin wrapper to keep the call-site self-documenting and easy to
    instrument later (timing, counters).

    Parameters
    ----------
    model : Module
        The callable to invoke.  Almost always a ``Module`` subclass,
        but the parameter is typed loosely so call sites can hand in
        any plain callable.
    args : tuple
        Positional arguments to forward.
    kwargs : dict
        Keyword arguments to forward.

    Returns
    -------
    object
        Whatever ``model`` returns — including non-tensor structured
        outputs (lists, dicts, dataclasses).
    """

    from lucid.compile._core.buffer_writes import eager_rerun

    # Module.__call__ wants Tensor positionals; the docstring promises
    # this entry accepts whatever the user passed (incl. non-Tensor).
    with eager_rerun():
        return model(*(cast(Tensor, a) for a in args), **kwargs)


class EagerFallbackSet:
    """Track signatures that already failed to compile.

    A signature lands in here when :func:`MpsBuilder.compile_trace…`
    returns nullptr (eager-only) or when tracing itself raises.  All
    future calls with the same key skip the compile attempt entirely
    and route straight to :func:`run_eager`.

    Cleared by :meth:`CompiledModule.clear_cache` so the user can opt
    back into a compile retry after a deliberate model change.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.compile._core.fallback import EagerFallbackSet, run_eager
    >>> from lucid.compile._core.signature import signature_of
    >>> model = nn.Linear(8, 4)
    >>> x = lucid.randn(2, 8)
    >>> key = signature_of(model, (x,), {}, dynamic=False)
    >>> blacklist = EagerFallbackSet()
    >>> blacklist.add(key)
    >>> if key in blacklist:                 # fast-path skip
    ...     out = run_eager(model, (x,), {})
    >>> out.shape
    (2, 4)

    See Also
    --------
    run_eager : the actual eager-dispatch helper.
    lucid.compile.CompiledModule : keeps one ``EagerFallbackSet`` per
        compiled module instance.
    """

    def __init__(self) -> None:
        """Construct an empty blacklist; one instance per :class:`CompiledModule`."""
        self._sigs: set[CacheKey] = set()

    def add(self, key: CacheKey) -> None:
        """Mark ``key`` as eager-only — future calls skip the compile attempt."""
        self._sigs.add(key)

    def __contains__(self, key: CacheKey) -> bool:
        """Membership test — ``key in blacklist`` returns ``True`` iff added."""
        return key in self._sigs

    def __len__(self) -> int:
        """Number of blacklisted signatures (for telemetry / tests)."""
        return len(self._sigs)

    def clear(self) -> None:
        """Drop every blacklisted signature; call after deliberate model edits."""
        self._sigs.clear()

    def snapshot(self) -> tuple[CacheKey, ...]:
        """Return blacklisted signatures in deterministic ``hash``-sorted order.

        Used by :meth:`CompiledModule.cache_info` so test assertions
        and telemetry get stable orderings across calls within a
        session.
        """
        # Deterministic ordering for cache_info / debugging.  ``hash``
        # is content-based on the frozen dataclass so the sort is
        # stable across calls within a session.
        return tuple(sorted(self._sigs, key=hash))
