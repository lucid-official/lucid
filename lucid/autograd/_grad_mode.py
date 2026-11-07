"""
Gradient mode context managers and decorators.
"""

import functools
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Callable, Self, overload, override

from lucid._C import engine as _C_engine


def _scoped[F: Callable[..., object]](
    fn: F, scope: Callable[[], AbstractContextManager[object]]
) -> F:
    """Wrap ``fn`` so each call runs inside a fresh ``scope()`` context."""

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        """Decorator-generated wrapper that applies the surrounding behaviour to the wrapped callable."""
        with scope():
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]  # wraps() result is not typed as F


class _GradScope:
    """Single owner of the save / switch / restore of the grad-mode flag.

    ``no_grad``, ``enable_grad`` and ``set_grad_enabled`` differ only in the
    mode they switch to.  The flag it replaces is kept on a stack so one
    instance can be entered again — nested or after it has exited — and each
    exit restores what its own entry saw.
    """

    mode: bool
    _prev: list[bool]

    def __init__(self, mode: bool) -> None:
        """Record the mode this scope switches to."""
        self.mode = mode
        self._prev = []

    def __enter__(self) -> Self:
        """Switch to ``mode``, remembering the previous one.  Returns self for ``with ... as``."""
        self._prev.append(_C_engine.raw_grad_enabled())
        _C_engine.set_grad_enabled(self.mode)
        return self

    def __exit__(self, *args: object) -> None:
        """Restore the mode saved by the matching ``__enter__``, also when the body raised."""
        _C_engine.set_grad_enabled(self._prev.pop())

    def __call__[F: Callable[..., object]](self, fn: F) -> F:
        """Wrap ``fn`` so every call runs inside a fresh scope of this mode."""
        mode = self.mode
        return _scoped(fn, lambda: _GradScope(mode))


class no_grad(_GradScope):
    r"""Context manager / decorator that disables gradient tracking.

    Inside a ``no_grad`` scope every op skips the graph-building
    bookkeeping that autograd normally performs: no autograd node
    is registered, no activations are saved, and the produced
    tensors carry ``requires_grad=False`` regardless of their
    inputs. The flag is process-global, so this affects all code
    running inside the scope.

    Use ``no_grad`` for inference, for validation loops, and for
    parameter updates that should not themselves be differentiable
    (e.g. EMA accumulation, optimizer steps if performed outside
    an Optimizer object). The savings — fewer Python objects
    allocated, no saved activations holding memory, no overhead
    on each op — are typically large.

    The previous gradient mode is restored on exit (RAII), so
    ``no_grad`` may be safely nested inside :class:`enable_grad`
    and vice-versa.

    Parameters
    ----------
    None
        ``no_grad`` takes no arguments; configure with
        :func:`set_grad_enabled` if a dynamic flag is needed.

    Attributes
    ----------
    mode : bool
        The grad mode the scope switches to.

    Notes
    -----
    Gradient mode controls whether autograd records the chain rule

    .. math::

        \frac{\partial \mathcal{L}}{\partial x}
        = \sum_i \frac{\partial \mathcal{L}}{\partial y_i}
          \cdot \frac{\partial y_i}{\partial x}.

    With ``no_grad`` the graph that this sum is computed over is
    never built, so calling :func:`backward` afterwards is a
    no-op on tensors produced inside the scope.

    Examples
    --------
    As a context manager:

    >>> import lucid
    >>> from lucid.autograd import no_grad
    >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
    >>> with no_grad():
    ...     y = x * 2
    >>> y.requires_grad
    False

    As a decorator:

    >>> @no_grad()
    ... def eval_step(x):
    ...     return (x * x).sum()
    """

    def __init__(self) -> None:
        """Build a scope that disables gradient tracking."""
        super().__init__(False)


class enable_grad(_GradScope):
    r"""Context manager / decorator that (re-)enables gradient tracking.

    Counterpart of :class:`no_grad`: turns gradient recording back
    on inside the wrapped scope, even when an outer scope has
    disabled it. Useful when a small differentiable subroutine
    runs inside a larger ``no_grad`` block — for example, an
    inner training step embedded in an outer evaluation loop, or
    explicit gradient computation for diagnostic plots inside a
    decorated inference function.

    The previous gradient mode is restored on exit (RAII), so the
    surrounding ``no_grad`` continues to apply once the inner
    scope finishes.

    Parameters
    ----------
    None

    Attributes
    ----------
    mode : bool
        The grad mode the scope switches to.

    Notes
    -----
    Setting the flag back to ``True`` re-arms autograd to record
    the chain rule

    .. math::

        \frac{\partial \mathcal{L}}{\partial x}
        = \sum_i \frac{\partial \mathcal{L}}{\partial y_i}
          \cdot \frac{\partial y_i}{\partial x}

    for every op executed inside the scope.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import no_grad, enable_grad
    >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
    >>> with no_grad():
    ...     with enable_grad():
    ...         y = (x * x).sum()
    >>> y.requires_grad
    True
    """

    def __init__(self) -> None:
        """Build a scope that enables gradient tracking."""
        super().__init__(True)


class set_grad_enabled(_GradScope):
    r"""Set the autograd gradient-tracking flag, as a function or a scope.

    Constructing it switches the flag at once, so a bare
    ``set_grad_enabled(False)`` call works as an imperative setter.  Used in
    a ``with`` block the previous mode is restored on exit, also when the
    body raises; used as a decorator every call of the wrapped function runs
    in a scope of ``mode`` and the decoration itself changes nothing.  The
    flag is per thread, like :class:`no_grad` and :class:`enable_grad`.

    Parameters
    ----------
    mode : bool
        ``True`` to enable gradient tracking, ``False`` to disable it.
        Anything that is not a ``bool`` raises :class:`TypeError`.

    Attributes
    ----------
    mode : bool
        The grad mode the scope switches to.

    Notes
    -----
    Called as a function the switch is permanent until another call changes
    it; nothing restores it.  Pair it with :func:`is_grad_enabled` to save
    and restore by hand, or use the ``with`` form.

    Examples
    --------
    As a function:

    >>> import lucid
    >>> from lucid.autograd import set_grad_enabled, is_grad_enabled
    >>> _ = set_grad_enabled(False)
    >>> is_grad_enabled()
    False
    >>> _ = set_grad_enabled(True)

    As a context manager:

    >>> is_train = False
    >>> with set_grad_enabled(is_train):
    ...     is_grad_enabled()
    False
    >>> is_grad_enabled()
    True

    As a decorator:

    >>> @set_grad_enabled(False)
    ... def predict(x):
    ...     return x * 2
    """

    _pending: bool

    def __init__(self, mode: bool) -> None:
        """Switch the flag to ``mode`` now, keeping the mode it replaces."""
        if not isinstance(mode, bool):
            raise TypeError(f"set_grad_enabled expects a bool, got {type(mode).__name__}")
        super().__init__(mode)
        self._prev.append(_C_engine.raw_grad_enabled())
        self._pending = True
        _C_engine.set_grad_enabled(mode)

    @override
    def __enter__(self) -> Self:
        """Enter the scope; the first entry reuses the mode the constructor saved."""
        if self._pending:
            self._pending = False
            _C_engine.set_grad_enabled(self.mode)
            return self
        return super().__enter__()

    @override
    def __call__[F: Callable[..., object]](self, fn: F) -> F:
        """Wrap ``fn``; the constructor's switch is undone so decorating changes nothing."""
        if self._pending:
            self._pending = False
            _C_engine.set_grad_enabled(self._prev.pop())
        return super().__call__(fn)


class _ReadsRefused(threading.local):
    """The reasons, innermost last, a read of the grad mode is refused for on this thread."""

    def __init__(self) -> None:
        self.whys: list[str] = []


_READS_REFUSED = _ReadsRefused()


@contextmanager
def _grad_mode_reads_refused(why: str) -> Iterator[None]:
    """Make ``is_grad_enabled()`` refuse an active compile recording while the body runs.

    A recording runs some functions once where eager runs them twice under
    different modes (a checkpointed segment: forward under ``no_grad``, the
    recompute under ``enable_grad``).  What such a function reads from the mode
    cannot be one recorded answer, so a read marks the recording unsupported and
    the call runs eagerly.  Outside a recording the read is unchanged.
    """
    _READS_REFUSED.whys.append(why)
    try:
        yield
    finally:
        _READS_REFUSED.whys.pop()


def is_grad_enabled() -> bool:
    r"""Return whether autograd gradient tracking is currently enabled.

    Reports the current state of the process-wide flag that
    :class:`no_grad`, :class:`enable_grad`, and
    :func:`set_grad_enabled` mutate. The result reflects whether
    subsequent ops will register autograd nodes.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        ``True`` if gradient tracking is enabled (the default
        outside any ``no_grad`` scope); ``False`` otherwise.

    Notes
    -----
    The flag is the same Boolean queried by the C++ engine at
    op construction time — the result is therefore guaranteed
    consistent with the actual behaviour of the next op.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import no_grad, is_grad_enabled
    >>> is_grad_enabled()
    True
    >>> with no_grad():
    ...     is_grad_enabled()
    False
    """
    if _READS_REFUSED.whys:
        tracer = _C_engine.compile.current_tracer()
        if tracer is not None:
            tracer.mark_unsupported(_READS_REFUSED.whys[-1])
    return _C_engine.raw_grad_enabled()


def is_inference_mode_enabled() -> bool:
    r"""Return whether inference mode is active on the calling thread.

    Reports the flag that :class:`inference_mode` toggles.  It is
    independent of :func:`is_grad_enabled`: ``inference_mode(False)``
    clears this flag *and* turns gradient tracking back on, while
    :func:`set_grad_enabled` never touches it.

    Returns
    -------
    bool
        ``True`` inside an ``inference_mode(True)`` scope, ``False``
        otherwise.

    Examples
    --------
    >>> import lucid
    >>> lucid.is_inference_mode_enabled()
    False
    >>> with lucid.inference_mode():
    ...     lucid.is_inference_mode_enabled()
    True
    """
    return _C_engine.inference_enabled()


class _InferenceModeMeta(type):
    """Lets a bare ``@inference_mode`` wrap the function instead of building a context."""

    @overload
    def __call__(cls, mode: bool = True) -> inference_mode: ...
    @overload
    def __call__[F: Callable[..., object]](cls, mode: F) -> F: ...
    @override
    def __call__(cls, mode: object = True) -> object:
        """Build the context, or wrap ``mode`` when it is a bare-decorator callable."""
        if isinstance(mode, bool):
            return super().__call__(mode)
        if callable(mode):
            return super().__call__(True)(mode)
        raise TypeError(f"inference_mode expects a bool or a callable, got {type(mode).__name__}")


class inference_mode(metaclass=_InferenceModeMeta):
    r"""Context manager / decorator for inference-time autograd suppression.

    ``inference_mode(True)`` (the default) disables gradient tracking
    and marks the thread as being in inference mode; ``inference_mode(False)``
    clears the mark *and* re-enables gradient tracking, so it can carve a
    differentiable region out of an enclosing inference or ``no_grad``
    scope.  The previous gradient mode and inference flag are restored
    on exit, so scopes nest freely.

    Parameters
    ----------
    mode : bool, optional
        ``True`` to enter inference mode, ``False`` to leave it (and
        re-enable gradients) for the scope.  Default ``True``.  When a
        callable is passed instead (bare ``@inference_mode``) it is
        wrapped and returned.

    Notes
    -----
    On exit the state captured at ``__enter__`` is restored:

    .. math::

        (\text{grad}, \text{inference})
        \leftarrow (\neg m, m) \quad\text{on entry},
        \qquad
        (\text{grad}, \text{inference})_{\text{prev}}
        \quad\text{on exit},

    where :math:`m` is ``mode``.  Tensors created inside an
    ``inference_mode(True)`` scope are marked as inference tensors
    (``Tensor.is_inference()``): saving them for backward or
    mutating them in place outside ``inference_mode`` raises a
    :class:`RuntimeError`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
    >>> with lucid.inference_mode():
    ...     y = x * 3
    >>> y.requires_grad
    False
    >>> @lucid.inference_mode()
    ... def predict(x):
    ...     return x * 3
    >>> predict(x).requires_grad
    False
    """

    mode: bool
    _prev: list[tuple[bool, bool]]

    def __init__(self, mode: bool = True) -> None:
        """Record the requested mode."""
        self.mode = mode
        self._prev = []

    def __enter__(self) -> None:
        """Enter the scope: grad tracking follows ``not mode``, the inference flag follows ``mode``."""
        self._prev.append((_C_engine.raw_grad_enabled(), _C_engine.inference_enabled()))
        _C_engine.set_grad_enabled(not self.mode)
        _C_engine.set_inference_enabled(self.mode)

    def __exit__(self, *args: object) -> None:
        """Restore the grad mode and inference flag captured on entry."""
        grad, inference = self._prev.pop()
        _C_engine.set_grad_enabled(grad)
        _C_engine.set_inference_enabled(inference)

    def __call__[F: Callable[..., object]](self, fn: F) -> F:
        """Wrap ``fn`` so every call runs inside a fresh scope of this mode."""
        mode = self.mode
        return _scoped(fn, lambda: inference_mode(mode))


# Mutable flag consulted by detect_anomaly; kept here to avoid circular imports.
_ANOMALY_ENABLED: list[bool] = [False]
