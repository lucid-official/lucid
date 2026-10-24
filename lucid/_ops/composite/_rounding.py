"""The graph node of a quotient rounded toward zero.

A rounded quotient is piecewise constant, so its derivative is zero for both
operands — the reference's, for ``div(..., rounding_mode="trunc")``.  The
node is a Python :class:`~lucid.autograd.function.Function` because it
saves nothing: the differentiable ``div`` under ``trunc`` saves its
operands, so a later in-place write to either was refused
(``y.mul_(3)`` after ``div(y, 2, rounding_mode="trunc")``, or
``y.div_(2, rounding_mode="trunc")`` with a view of ``y`` alive) for a
gradient that does not read them.

A compile recording takes the plain op for either node
(:attr:`~lucid.autograd.function.Function._trace_form`): it runs no eager
backward, so the saved operands are never checked, and the plain op's own
derivative is the same zero.
"""

from typing import final, override

from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._factories.creation import zeros_like
from lucid._ops._division import _round_decimals, _trunc_quotient
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx


def _plain_trunc_quotient(a: Tensor, b: Tensor) -> Tensor:
    """``trunc(a / b)`` from the plain ops, for a compile recording."""
    return _wrap(_trunc_quotient(a._impl, b._impl))


def _plain_round_decimals(a: Tensor, dec: int) -> Tensor:
    """``round(a, decimals=dec)`` from the plain ops, for a compile recording."""
    return _wrap(_round_decimals(a._impl, dec))


@final
class _TruncQuotient(Function):
    """``trunc(a / b)``, with a zero derivative with respect to both."""

    _trace_form = staticmethod(_plain_trunc_quotient)

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, a: Tensor, b: Tensor) -> Tensor:
        return _wrap(_trunc_quotient(a._impl.data_alias(), b._impl.data_alias()))

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor | None, Tensor | None]:
        # The node reduces each gradient to its operand's shape and dtype.
        zero = zeros_like(grad)
        needs_a, needs_b = ctx.needs_input_grad
        return (zero if needs_a else None), (zero if needs_b else None)


@final
class _RoundDecimals(Function):
    """``round(a, decimals=dec)``, with a zero derivative."""

    _trace_form = staticmethod(_plain_round_decimals)

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, a: Tensor, dec: int) -> Tensor:
        return _wrap(_round_decimals(a._impl.data_alias(), dec))

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor | None, None]:
        return (zeros_like(grad) if ctx.needs_input_grad[0] else None), None


def _recorded_trunc_quotient(
    a: _C_engine.TensorImpl, b: _C_engine.TensorImpl
) -> _C_engine.TensorImpl:
    """``trunc(a / b)`` in the graph, through :class:`_TruncQuotient`."""
    out = _TruncQuotient.apply(_wrap(a), _wrap(b))
    assert isinstance(out, Tensor)
    return out._impl


def _recorded_round_decimals(a: _C_engine.TensorImpl, dec: int) -> _C_engine.TensorImpl:
    """``round(a, decimals=dec)`` in the graph, through :class:`_RoundDecimals`."""
    out = _RoundDecimals.apply(_wrap(a), dec)
    assert isinstance(out, Tensor)
    return out._impl
