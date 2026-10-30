"""``custom_fwd`` / ``custom_bwd`` — make a custom :class:`~lucid.autograd.Function` autocast-aware.

A custom ``Function.forward`` runs under whatever autocast scope its caller
is in and its ``backward`` runs later, usually outside that scope.  A
``forward`` doing ``a @ b`` therefore returns a float16 / bfloat16 result,
while the ``backward``'s ``grad @ b.T`` pairs that half-precision gradient
with the float32 ``b`` it saved and fails with a dtype mismatch.

The two decorators close the gap.  :func:`custom_fwd` records the autocast
state of its device on the context — and may first cast the floating inputs
and run ``forward`` with autocast off; :func:`custom_bwd` runs ``backward``
in the state that was recorded.  Each is a small decorator class holding what
it was configured with, built (and validated) once at the decoration site.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Final, final, overload

from lucid._dtype import dtype, float64
from lucid.amp._state import AutocastState
from lucid.amp.autocast import _to_engine_device, autocast

if TYPE_CHECKING:
    from lucid.autograd.function import FunctionCtx

#: Where ``custom_fwd`` leaves the forward's state on the context.  Private
#: (underscore) so ``FunctionCtx`` keeps it on the instance, not in the bag
#: of user attributes a compiled backward walks.
_STATE_ATTR: Final = "_amp_forward_state"


def _cast(value: object, device_type: str, target: dtype) -> object:
    """``value`` with its floating tensors on ``device_type`` cast to ``target``.

    Reaches into plain ``list`` / ``tuple`` / ``dict`` arguments, as the
    reference framework does; every other container passes through whole, so
    a subclass (a named tuple) is never rebuilt as something it is not.
    float64 stays: autocast never narrows it.
    """
    from lucid._tensor.tensor import Tensor

    if isinstance(value, Tensor):
        castable = (
            value.is_floating_point()
            and value.device.type == device_type
            and value.dtype != float64
        )
        return value.to(target) if castable else value
    if type(value) is list:
        return [_cast(v, device_type, target) for v in value]
    if type(value) is tuple:
        return tuple(_cast(v, device_type, target) for v in value)
    if type(value) is dict:
        return {k: _cast(v, device_type, target) for k, v in value.items()}
    return value


@final
class _CustomForward:
    """The decorator :func:`custom_fwd` builds: records autocast state, optionally casts."""

    __slots__ = ("_device_type", "_cast_inputs")

    def __init__(self, device_type: str, cast_inputs: dtype | None) -> None:
        _to_engine_device(device_type, "custom_fwd")
        self._device_type = device_type
        self._cast_inputs = cast_inputs

    def __call__[R](self, fwd: Callable[..., R]) -> Callable[..., R]:
        device_type, cast_inputs = self._device_type, self._cast_inputs

        @functools.wraps(fwd)
        def decorated(ctx: FunctionCtx, *args: object, **kwargs: object) -> R:
            seen = AutocastState.capture(device_type)
            if cast_inputs is None:
                setattr(ctx, _STATE_ATTR, seen)
                return fwd(ctx, *args, **kwargs)
            # Inputs cast, body in full precision: backward must not re-enter
            # autocast either, so the state recorded is "off" whatever the
            # caller's scope was.
            setattr(ctx, _STATE_ATTR, dataclasses.replace(seen, enabled=False))
            if not seen.enabled:
                return fwd(ctx, *args, **kwargs)
            with autocast(device_type, enabled=False):
                return fwd(
                    ctx,
                    *(_cast(a, device_type, cast_inputs) for a in args),
                    **{k: _cast(v, device_type, cast_inputs) for k, v in kwargs.items()},
                )

        return decorated


@final
class _CustomBackward:
    """The decorator :func:`custom_bwd` builds: runs ``backward`` in the forward's state."""

    __slots__ = ("_device_type",)

    def __init__(self, device_type: str) -> None:
        _to_engine_device(device_type, "custom_bwd")
        self._device_type = device_type

    def __call__[R](self, bwd: Callable[..., R]) -> Callable[..., R]:
        device_type = self._device_type

        @functools.wraps(bwd)
        def decorated(ctx: FunctionCtx, *args: object, **kwargs: object) -> R:
            match getattr(ctx, _STATE_ATTR, None):
                case AutocastState(device_type=recorded) as state if recorded == device_type:
                    with state.scope():
                        return bwd(ctx, *args, **kwargs)
                case AutocastState(device_type=recorded):
                    raise RuntimeError(
                        f"custom_bwd(device_type={device_type!r}): the forward recorded its "
                        f"autocast state for {recorded!r} — decorate forward and "
                        "backward with the same device_type."
                    )
                case _:
                    raise RuntimeError(
                        "custom_bwd: this context holds no autocast state — decorate the "
                        "Function's forward with custom_fwd, with the same device_type."
                    )

        return decorated


@overload
def custom_fwd[R](
    fwd: Callable[..., R], *, device_type: str, cast_inputs: dtype | None = None
) -> Callable[..., R]: ...
@overload
def custom_fwd(
    fwd: None = None, *, device_type: str, cast_inputs: dtype | None = None
) -> _CustomForward: ...
def custom_fwd[R](
    fwd: Callable[..., R] | None = None, *, device_type: str, cast_inputs: dtype | None = None
) -> Callable[..., R] | _CustomForward:
    r"""Decorate a custom ``Function.forward`` so ``backward`` can follow its autocast state.

    Under :class:`~lucid.amp.autocast` an op inside ``forward`` computes in
    half precision, so its result and the gradient that comes back are half
    precision while the tensors ``forward`` saved are still float32 — and the
    ``backward``'s arithmetic on the pair fails.  The decorated ``forward``
    stores the autocast state of ``device_type`` on ``ctx``; decorate
    ``backward`` with :func:`custom_bwd` to run it in that state.

    Used bare as ``custom_fwd(fwd, device_type=...)`` or, as a decorator
    factory, ``@custom_fwd(device_type=...)`` placed under ``@staticmethod``.

    Parameters
    ----------
    fwd : callable, optional
        The ``forward`` to decorate; leave it out to get the decorator.
    device_type : str
        ``"metal"`` or ``"cpu"``; anything else raises :class:`ValueError`.
        Keyword-only and required.
    cast_inputs : lucid.dtype, optional
        When given and autocast is enabled on ``device_type`` at the call,
        every floating-point tensor argument on that device (float64
        excepted) is cast to this dtype — nested lists, tuples and dicts
        included — and ``forward`` runs with autocast **off**; ``backward``
        then runs with it off too.  ``None`` (default) leaves the inputs
        alone and runs ``forward`` in the caller's state.  Outside an
        autocast scope ``cast_inputs`` does nothing.

    Returns
    -------
    callable
        The decorated ``forward`` (``fwd`` given), or the decorator.

    Raises
    ------
    ValueError
        If ``device_type`` is neither ``"metal"`` nor ``"cpu"``.

    See Also
    --------
    custom_bwd : the matching ``backward`` decorator.
    lucid.amp.autocast : the scope whose state is recorded.

    Examples
    --------
    >>> import lucid
    >>> from lucid.amp import autocast, custom_bwd, custom_fwd
    >>> from lucid.autograd import Function
    >>> class MatMul(Function):
    ...     @staticmethod
    ...     @custom_fwd(device_type="cpu")
    ...     def forward(ctx, a, b):
    ...         ctx.save_for_backward(a, b)
    ...         return a @ b
    ...     @staticmethod
    ...     @custom_bwd(device_type="cpu")
    ...     def backward(ctx, grad):
    ...         a, b = ctx.saved_tensors
    ...         return grad @ b.mT, a.mT @ grad
    >>> a = lucid.randn(3, 4, requires_grad=True)
    >>> b = lucid.randn(4, 2, requires_grad=True)
    >>> with autocast("cpu", dtype=lucid.bfloat16):
    ...     y = MatMul.apply(a, b)
    >>> y.dtype
    lucid.bfloat16
    >>> y.float().sum().backward()
    >>> a.grad.dtype
    lucid.float32
    """
    decorator = _CustomForward(device_type, cast_inputs)
    return decorator if fwd is None else decorator(fwd)


@overload
def custom_bwd[R](bwd: Callable[..., R], *, device_type: str) -> Callable[..., R]: ...
@overload
def custom_bwd(bwd: None = None, *, device_type: str) -> _CustomBackward: ...
def custom_bwd[R](
    bwd: Callable[..., R] | None = None, *, device_type: str
) -> Callable[..., R] | _CustomBackward:
    r"""Decorate a custom ``Function.backward`` to run in its ``forward``'s autocast state.

    The decorated ``backward`` runs inside an autocast scope matching the one
    :func:`custom_fwd` recorded — on at the same dtype, or off — whatever
    scope ``.backward()`` was called from, so the arithmetic on the gradient
    and the tensors ``forward`` saved sees the dtypes it saw going forward.

    Parameters
    ----------
    bwd : callable, optional
        The ``backward`` to decorate; leave it out to get the decorator.
    device_type : str
        ``"metal"`` or ``"cpu"``, the same as ``forward``'s.  Keyword-only
        and required.

    Returns
    -------
    callable
        The decorated ``backward`` (``bwd`` given), or the decorator.

    Raises
    ------
    ValueError
        If ``device_type`` is neither ``"metal"`` nor ``"cpu"``.
    RuntimeError
        When the decorated ``backward`` runs: if ``forward`` was not
        decorated with :func:`custom_fwd`, or was with another
        ``device_type``.

    See Also
    --------
    custom_fwd : the matching ``forward`` decorator, with a worked example.
    """
    decorator = _CustomBackward(device_type)
    return decorator if bwd is None else decorator(bwd)
