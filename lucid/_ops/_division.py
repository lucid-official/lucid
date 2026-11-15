"""``div`` and the rounding its ``rounding_mode`` asks for.

``rounding_mode=None`` is true division, the ``/`` operator's path.
``"floor"`` is ``floor_divide``'s path — its promotion and its exact kernel
— and ``"trunc"`` rounds toward zero.  A rounded quotient keeps an integer
dtype, and its gradient is zero for both operands, as the reference's is.
``lucid.div``, ``Tensor.div`` and ``Tensor.div_`` are all made here, so the
three cannot disagree about a mode.

Integer division by zero (``//``, ``div`` with a rounding mode, ``%``,
``fmod``, ``remainder``) raises a ``RuntimeError`` naming ``ZeroDivisionError``
on the CPU, as the reference does.  On Metal it does not raise: refusing would mean reading every
divisor back to the host.  The remainder ops return the dividend, and the
quotient is unspecified.  The reference's GPU backend behaves the same way,
and its integer quotient differs between int32 and int64.
"""

import inspect
from collections.abc import Callable

from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _REAL_FLOATS, _binary_operands
from lucid._dtype import _ENGINE_TO_DTYPE
from lucid._ops._adapters import _BINARY_SIGNATURE

_Impl = _C_engine.TensorImpl

#: ``div``'s ``(input, other, *, rounding_mode=None)``, the reference's.
_DIV_SIGNATURE = _BINARY_SIGNATURE.replace(
    parameters=[
        *_BINARY_SIGNATURE.parameters.values(),
        inspect.Parameter(
            "rounding_mode", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=str | None
        ),
    ]
)

#: ``round``'s ``(input, *, decimals=0)``, the reference's.
_ROUND_SIGNATURE = inspect.Signature(
    [
        inspect.Parameter("input", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("decimals", inspect.Parameter.KEYWORD_ONLY, default=0, annotation=int),
    ]
)

#: The dtypes ``div`` rounds no quotient of — the reference refuses both.
_UNROUNDED_DTYPES = (_C_engine.Bool, _C_engine.C64, _C_engine.C128)


def _trunc_quotient(a: _Impl, b: _Impl) -> _Impl:
    """``trunc(a / b)`` for two real floating operands.

    Not an exact rounding of the real quotient: ``1.0 / 0.1`` rounds up to
    ``10``, where ``1.0 // 0.1`` is ``9`` — the reference's results.  A half
    tensor beside its op-math partner (a 0-d float32 from
    ``_binary_operands``) truncates in float32 and rounds once, as the
    reference does; two half tensors truncate their half quotient.
    """
    if a.dtype == b.dtype:
        return _C_engine.trunc(_C_engine.div(a, b))
    half = a.dtype if _promotion.is_half(a.dtype) else b.dtype
    wide = _C_engine.F32
    q = _C_engine.div(_C_engine.astype(a, wide), _C_engine.astype(b, wide))
    return _C_engine.astype(_C_engine.trunc(q), half)


def _trunc_div(a: _Impl, b: _Impl) -> _Impl:
    """``a / b`` rounded toward zero, for two operands from ``_binary_operands``.

    An integer quotient is exact: ``fmod`` keeps the dividend's sign, so
    ``a - fmod(a, b)`` is a multiple of ``b`` no wider than ``a``, and its
    floor quotient is the truncated one.  A zero divisor is refused on the
    CPU and gives -1 on Metal, as for ``floor_divide``.  A float quotient
    that autograd records goes through a node that saves nothing
    (:class:`lucid._ops.composite._rounding._TruncQuotient`).

    Raises
    ------
    lucid._C.engine.NotImplementedError
        A bool or complex operand.
    """
    if a.dtype in _UNROUNDED_DTYPES or b.dtype in _UNROUNDED_DTYPES:
        raise _C_engine.NotImplementedError(
            "div: rounding_mode='trunc' takes integer or real floating-point "
            f"operands, got {_ENGINE_TO_DTYPE[a.dtype]} and {_ENGINE_TO_DTYPE[b.dtype]}"
        )
    if _promotion.is_integral(a.dtype):
        return _C_engine.floordiv(_C_engine.sub(a, _C_engine.fmod(a, b)), b)
    if _C_engine.grad_enabled() and (a.requires_grad or b.requires_grad):
        from lucid._ops.composite._rounding import _recorded_trunc_quotient

        return _recorded_trunc_quotient(a, b)
    return _trunc_quotient(a, b)


def _make_div_adapter(*, inplace: bool = False) -> Callable[..., _Impl]:
    """``div(input, other, *, rounding_mode=None)``, or ``div_`` with ``inplace``.

    In place, a rounded quotient is written back into ``input`` by the
    method; its dtype must be castable to ``input``'s, so an integer tensor
    divided by a float is refused, as it is for ``rounding_mode=None``.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        A ``rounding_mode`` other than ``None``, ``"trunc"`` or ``"floor"``.
    """

    def _adapter(input: _Impl, other: _Impl, /, *, rounding_mode: str | None = None) -> _Impl:
        if rounding_mode is None:
            a, b = _binary_operands(input, other, true_div=True, opmath=True, inplace=inplace)
            return _C_engine.div_(a, b) if inplace else _C_engine.div(a, b)
        if rounding_mode not in ("trunc", "floor"):
            raise _C_engine.InvalidArgument(
                "div expected rounding_mode to be one of None, 'trunc', or "
                f"'floor' but found {rounding_mode!r}"
            )
        a, b = _binary_operands(input, other, opmath=True, inplace=inplace)
        if rounding_mode == "floor":
            return _C_engine.floordiv(a, b)
        return _trunc_div(a, b)

    _adapter.__doc__ = getattr(_C_engine.div, "__doc__", None)
    _adapter.__name__ = "div_" if inplace else "div"
    _adapter.__dict__["__signature__"] = _DIV_SIGNATURE
    return _adapter


def _round_decimals(a: _Impl, dec: int) -> _Impl:
    """Round a real floating-point ``a`` to ``dec`` decimal places."""
    work = _C_engine.astype(a, _C_engine.F32) if a.dtype == _C_engine.BF16 else a
    scale = _C_engine.full([], 10.0 ** abs(dec), work.dtype, work.device)
    if dec > 0:
        out = _C_engine.div(_C_engine.round(_C_engine.mul(work, scale)), scale)
    else:
        out = _C_engine.mul(_C_engine.round(_C_engine.div(work, scale)), scale)
    return _C_engine.astype(out, a.dtype) if work.dtype != a.dtype else out


def _make_round_adapter(*, inplace: bool = False) -> Callable[..., _Impl]:
    """``round(input, *, decimals=0)``, or ``round_`` with ``inplace``."""

    def _adapter(input: _Impl, /, *, decimals: int = 0) -> _Impl:
        dec = int(decimals)
        if dec == 0 or input.dtype not in _REAL_FLOATS:
            return _C_engine.round_(input) if inplace else _C_engine.round(input)
        if _C_engine.grad_enabled() and input.requires_grad:
            from lucid._ops.composite._rounding import _recorded_round_decimals

            return _recorded_round_decimals(input, dec)
        return _round_decimals(input, dec)

    _adapter.__doc__ = getattr(_C_engine.round, "__doc__", None)
    _adapter.__name__ = "round_" if inplace else "round"
    _adapter.__dict__["__signature__"] = _ROUND_SIGNATURE
    return _adapter
