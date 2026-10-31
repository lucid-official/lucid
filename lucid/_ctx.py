"""Thread-local ``lucid.func`` execution context.

Tracks the active ``randomness`` mode so that random op entry points can
raise immediately when ``randomness='error'`` is set by an enclosing
``vmap`` call, and how many differentiating transforms are evaluating a
user function right now, so a transform entered inside one knows its
result has to stay differentiable.  The module has **zero** dependencies
on the rest of Lucid so it is safe to import from anywhere without
circular-import risk.
"""

import threading
from types import TracebackType
from typing import Final, final

_local: threading.local = threading.local()

#: The values ``vmap(randomness=)`` and ``jacfwd(randomness=)`` accept.
RANDOMNESS_MODES: Final = ("error", "different", "same")


def check_randomness(mode: str) -> str:
    """``mode`` itself, if it is one of ``RANDOMNESS_MODES``.

    Parameters
    ----------
    mode : str
        The ``randomness`` argument of a transform.

    Returns
    -------
    str
        ``mode``, unchanged.

    Raises
    ------
    RuntimeError
        ``mode`` is none of ``"error"``, ``"different"`` and ``"same"``.
    """
    if mode not in RANDOMNESS_MODES:
        raise RuntimeError(
            f"Only allowed values for randomness are 'error', 'different', or 'same'. Got {mode}"
        )
    return mode


def get_randomness() -> str:
    """Return the active ``vmap`` randomness mode for this thread.

    Read by random ops inside a vectorised function to decide whether
    to draw independent samples, share a single sample across the
    batch, or raise.

    Returns
    -------
    str
        One of ``"different"`` / ``"same"`` / ``"error"``.  Defaults
        to ``"different"`` when no enclosing :func:`vmap` is active —
        random ops outside vmap behave as they always have.
    """
    return str(getattr(_local, "randomness", "different"))


def set_randomness(mode: str) -> None:
    """Set the ``vmap`` randomness mode on the current thread.

    Normally driven by the :class:`_RandomnessGuard` inside
    :func:`vmap` — direct user calls are uncommon but allowed for
    advanced control (e.g. integrating Lucid with an external batched
    sampler that already enforces its own randomness contract).

    Parameters
    ----------
    mode : str
        One of ``"different"`` (per-batch-element independent draws),
        ``"same"`` (single sample broadcast across the batch), or
        ``"error"`` (random ops inside the vmap raise).
    """
    _local.randomness = mode


@final
class _RandomnessGuard:
    """RAII guard: push *mode* as the vmap randomness context, pop on exit.

    Nested vmap calls each push their own mode so the innermost vmap's
    ``randomness`` setting takes effect for random ops inside it.
    """

    def __init__(self, mode: str) -> None:
        self._mode = check_randomness(mode)
        self._prev: str = "different"

    def __enter__(self) -> None:
        self._prev = get_randomness()
        set_randomness(self._mode)

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        set_randomness(self._prev)


def check_random_allowed() -> None:
    """Raise ``RuntimeError`` if the current vmap context forbids random ops.

    Call this at the top of every Python-level random factory
    (``randn``, ``rand``, ``randint``, …) so that ``vmap(...,
    randomness='error')`` produces a clear error instead of silently
    sampling inside the vectorised function.
    """
    if get_randomness() == "error":
        raise RuntimeError(
            "vmap: called a random op inside a vmapped function with "
            "randomness='error'. Pass randomness='different' to allow "
            "independent random draws per batch element, or "
            "randomness='same' for a shared stream across elements."
        )


def transform_depth() -> int:
    """How many differentiating transforms are evaluating a function now.

    ``grad`` / ``vjp`` / ``jvp`` / ``jacrev`` / ``jacfwd`` / ``hessian``
    each count themselves for as long as they run the user's function.  A
    transform entered while the count is above zero sits inside another
    one's function, so what it returns is an intermediate of the outer
    derivative and must stay attached to the graph the outer transform
    differentiates.  ``0`` outside every transform.
    """
    return int(getattr(_local, "depth", 0))


@final
class _TransformLevel:
    """RAII guard: count one more differentiating transform, uncount on exit.

    Held only while the user's function runs.  The backward passes a
    transform makes after it contain no user code and so need no level.
    """

    def __enter__(self) -> None:
        _local.depth = transform_depth() + 1

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        _local.depth = transform_depth() - 1
