"""``special.entr``: the entropy kernel ``-x log x`` with the reference's
gradient at the origin (LCD-772).  Kept out of ``lucid/special/__init__.py``,
which is over its size cap."""

from typing import final, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid._ops.composite._opmath import _real


def _plain_entr(x: Tensor) -> Tensor:
    """``-x log x`` with the ``0 log 0 = 0`` limit and ``-inf`` below zero."""
    x = _real(x)
    # ``0.0 - …`` rather than a unary minus: the convention's 0 at x == 0
    # is +0, and negating it would print as -0.
    val = 0.0 - lucid.xlogy(x, x)
    # Negative x is -inf, not NaN.  The entropy kernel is an extended-real
    # function — ``-x log x`` on the positive half, ``0`` at zero and ``-inf``
    # below — and a NaN there loses the ordering that makes it usable as a
    # penalty.  A NaN input needs no branch: ``nan < 0`` is false.
    return lucid.where(x < 0.0, float("-inf"), val)


@final
class _Entr(lucid.autograd.Function):
    """``entr`` with the analytic derivative ``-(1 + log x)``.

    The value is :func:`_plain_entr`, whose ``xlogy`` is a ``0`` at the
    origin with a ``0`` gradient.  The reference does not carry that
    convention into the derivative: it differentiates the formula, so the
    slope is the one-sided ``+inf`` at ``0`` (the function really is
    vertical there) and NaN below it.  An optimiser that starts a
    probability at ``0`` is told to leave it, rather than that it is already
    at a stationary point.  ``log`` is differentiable, so higher orders
    follow.
    """

    _trace_form = staticmethod(_plain_entr)

    @override
    @staticmethod
    def forward(ctx: lucid.autograd.FunctionCtx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return _plain_entr(x.detach())

    @override
    @staticmethod
    def backward(ctx: lucid.autograd.FunctionCtx, grad: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return grad * (-1.0 - lucid.log(x))
