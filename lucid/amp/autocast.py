"""
autocast context manager for automatic mixed precision.
"""

import functools
import threading
from typing import Callable
from lucid._C import engine as _C_engine
from lucid._dtype import bfloat16, dtype, float16, to_engine_dtype

#: The device types an autocast scope can name — the same two ``lucid.device``
#: accepts.
_DEVICE_TYPES = frozenset({"metal", "cpu"})


def _to_engine_device(device_type: str) -> _C_engine.Device:
    """Validate ``device_type`` and map it to :class:`_C_engine.Device`."""
    if device_type not in _DEVICE_TYPES:
        raise ValueError(f"autocast: unknown device_type {device_type!r}. Use 'metal' or 'cpu'.")
    return _C_engine.Device.GPU if device_type == "metal" else _C_engine.Device.CPU


class autocast:
    """Enable automatic mixed-precision computation inside a context.

    Operations that opt into AMP (matmul, conv, attention, …) cast their
    inputs to the target lower-precision dtype on entry, reducing memory
    pressure and — on Metal — improving throughput by activating the
    M-series GPU's half-precision tensor pipelines.  Cast points are
    inserted *only* at op boundaries; activations themselves are stored
    in the lower-precision dtype but accumulator and reduction state stay
    in float32 to preserve numerical fidelity (the standard "mixed" recipe).

    Leaving the scope restores exactly the state it was entered from —
    autocast off, or the enclosing scope's dtype — whether it is left
    normally or by an exception, and however long the ``autocast`` object
    itself lives on afterwards.  One object can be entered again after it
    exits, entered while it is already active (each entry is undone by its
    own exit), and used as a decorator.  ``enabled=False`` disables
    autocast on ``device_type`` for the duration of the scope while leaving
    the other device's autocast state untouched.

    Autocast state is tracked independently per device (``"metal"`` vs
    ``"cpu"``) on each thread.

    Parameters
    ----------
    device_type : str, optional
        Device the autocast scope applies to.  ``"metal"`` (GPU stream,
        default) or ``"cpu"`` (Accelerate stream); anything else raises
        :class:`ValueError`.
    dtype : lucid.dtype, optional
        Lower-precision dtype that supported ops cast inputs to inside
        the scope.  Default :data:`lucid.float16` on ``"metal"`` and
        :data:`lucid.bfloat16` on ``"cpu"`` (matching the reference
        framework).
    enabled : bool, optional
        When ``False`` autocast is disabled on ``device_type`` inside the
        scope — useful for ablation or forcing full precision in a nested
        subregion.  Default ``True``.

    Examples
    --------
    Context-manager form (typical training loop):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2).to("metal")
    >>> x = lucid.randn(8, 4, device="metal")
    >>> with lucid.amp.autocast():
    ...     output = model(x)            # ops cast to float16 inside
    >>> output.dtype
    lucid.float16

    Decorator form (every call wraps itself in a fresh scope):

    >>> @lucid.amp.autocast()
    ... def predict(x):
    ...     return model(x)
    >>> predict(x).dtype
    lucid.float16

    Nested scopes restore the outer dtype on exit:

    >>> a = lucid.randn(2, 2, device="metal")
    >>> with lucid.amp.autocast(dtype=lucid.float16):
    ...     print((a @ a).dtype)  # fp16 here
    ...     with lucid.amp.autocast(dtype=lucid.bfloat16):
    ...         print((a @ a).dtype)  # bf16 here
    ...     print((a @ a).dtype)  # back to fp16 — prior guard reinstalled
    lucid.float16
    lucid.bfloat16
    lucid.float16

    See Also
    --------
    lucid.amp.GradScaler : loss-scaling counterpart for fp16 training.
    """

    def __init__(
        self,
        device_type: str = "metal",
        dtype: dtype | None = None,
        enabled: bool = True,
        cache_enabled: bool | None = None,
    ) -> None:
        """Configure the autocast context.

        Parameters
        ----------
        device_type : str, default='metal'
            Device the autocast scope applies to. Apple Silicon supports
            ``'metal'`` (GPU stream) and ``'cpu'`` (Accelerate stream).
        dtype : lucid.dtype or None, default=None
            Lower-precision dtype that supported ops cast inputs to inside
            the scope (defaults to :data:`lucid.float16` on ``'metal'`` and
            :data:`lucid.bfloat16` on ``'cpu'`` when ``None``).
        enabled : bool, default=True
            When ``False`` autocast is disabled on ``device_type`` inside
            the scope (useful for ablation or nested full-precision blocks).
        cache_enabled : bool or None, default=None
            Whether weight-cast caching is enabled inside the autocast scope.
            Defaults to ``True`` when ``None``.

        Raises
        ------
        ValueError
            If ``device_type`` is neither ``'metal'`` nor ``'cpu'``.
        """
        self._engine_device = _to_engine_device(device_type)
        self._dtype = (bfloat16 if device_type == "cpu" else float16) if dtype is None else dtype
        self._enabled = enabled
        self._cache_enabled = True if cache_enabled is None else bool(cache_enabled)
        # One stack of live engine guards per thread: the AMP state the
        # guards change is thread-local, so a guard must be released on the
        # thread that made it.  A stack (not a single slot) lets the same
        # object be entered again while it is already active.
        self._local = threading.local()

    @property
    def cache_enabled(self) -> bool:
        """Whether weight-cast caching is enabled inside this autocast scope."""
        return self._cache_enabled

    def _guards(self) -> list[_C_engine.AutocastGuard]:
        """Return this thread's stack of guards opened by this object."""
        stack: list[_C_engine.AutocastGuard] | None = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def __enter__(self) -> autocast:
        """Activate (or disable) the autocast scope and return ``self``.

        Constructing the engine ``AutocastGuard`` sets the AMP state for
        ``self._device_type`` to ``(enabled, dtype)`` and records the state
        it replaced; the guard is kept until :meth:`__exit__`.
        """
        self._guards().append(
            _C_engine.AutocastGuard(
                to_engine_dtype(self._dtype), self._engine_device, self._enabled
            )
        )
        return self

    def __exit__(self, *args: object) -> None:
        """Restore the AMP state that was active when the scope was entered.

        Releases the guard :meth:`__enter__` made.  Its destructor runs at
        once and puts back the state it recorded, so the scope ends here
        rather than whenever the ``autocast`` object is collected.
        Exceptions propagate.
        """
        guard = self._guards().pop()
        del guard  # the last reference — ~AutocastGuard restores the prior state

    def __call__[F: Callable[..., object]](self, fn: F) -> F:
        """Use as a function decorator.

        Every call of the decorated function runs inside this scope — the
        same ``device_type``, ``dtype`` and ``enabled`` — and leaves it on
        return or on an exception.

        Parameters
        ----------
        fn : callable
            The function to wrap.

        Returns
        -------
        callable
            ``fn`` wrapped so that each call enters and exits the scope.
        """

        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            """Invoke ``fn`` inside this autocast scope."""
            with self:
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    @staticmethod
    def is_autocast_enabled(device_type: str | None = None) -> bool:
        """Return True if autocast is currently active (for ``device_type`` if given)."""
        if device_type is None:
            return _C_engine.amp_is_active()
        return _C_engine.amp_is_active(_to_engine_device(device_type))

    @staticmethod
    def get_autocast_dtype(device_type: str | None = None) -> dtype | None:
        """Return the currently active autocast dtype (for ``device_type`` if given), or None."""
        from lucid._dtype import _ENGINE_TO_DTYPE

        eng = (
            _C_engine.amp_active_dtype()
            if device_type is None
            else _C_engine.amp_active_dtype(_to_engine_device(device_type))
        )
        if eng is None:
            return None
        return _ENGINE_TO_DTYPE.get(eng)
