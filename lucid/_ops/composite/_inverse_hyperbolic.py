"""``asinh`` / ``acosh`` / ``atanh``: values and slopes that keep the dtype's range (LCD-1028).

The textbook forms ``log(x + sqrt(x^2 + 1))``, ``log(x + sqrt(x^2 - 1))`` and
``log((1 + x) / (1 - x)) / 2`` are right in the middle of the line and wrong at both
ends.  ``x^2`` leaves float32 at 1.8e19 (``asinh(1e20)`` was ``inf``, answer 46.7); the sum
``x + sqrt(x^2 + 1)`` rounds to 1 below 3e-5, so ``asinh(1e-8)`` was 0; for a negative
``x`` it cancels from -560 down (``asinh(-1e5)`` was ``-inf``); ``(1 + x) / (1 - x)``
rounds to 1 below 1.8e-5 (``atanh(1e-8)`` was 0).  The derivatives, taken through the
same formulas, were NaN or wrong over the same ranges.

The value of each is one ``log1p`` of an argument that never rounds toward 1 and never
overflows:

* ``asinh a = log1p(a + a^2 / (1 + sqrt(1 + a^2)))`` and
  ``acosh x = log1p(t + sqrt(t) sqrt(t + 2))`` with ``t = x - 1``, which is exact up to 2;
* ``atanh a = log1p(2 a / (1 - a)) / 2``, on ``a = |x|`` since ``1 + x`` is the quantity
  that cancels for a negative ``x``.

Past ``2^20`` (float32) or ``2^100`` (float64) the argument is clamped and the missing
``log(x / 2^k)`` is added back: the terms ``asinh`` and ``acosh`` drop there, ``1 / (4 x^2)``
and its sign flip, are far below the dtype's precision.  A power of two keeps the division
exact.

The derivative is the closed form (``1 / sqrt(1 + x^2)``, ``1 / sqrt(x^2 - 1)``,
``1 / (1 - x^2)``), written so that no intermediate leaves the range and so that
differentiating it again stays exact, which a ``Function`` carries as its ``backward``:
the slope is a graph of ordinary ops, so every higher order follows.
"""

from collections.abc import Callable
from typing import final, override

import lucid
from lucid._dispatch import _float_input
from lucid._ops.composite._opmath import _in_float32
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx


def _split(x: Tensor) -> float:
    """Where the stable argument is clamped: a power of two whose square is in range.

    Low enough that the third derivative of the slope, whose terms go as ``x^-5``, is
    still above the dtype's smallest normal at the split, high enough that the dropped
    ``1 / (4 x^2)`` is far below its precision.
    """
    return 2.0**100 if x.dtype == lucid.float64 else 2.0**20


def _beyond(a: Tensor, split: float) -> Tensor:
    """``a / split`` past the split, 1 up to it.

    Where ``clamp(a, max=split)`` carries the slope up to the split (and at it), this is
    the part that carries it beyond.  A ``clamp(min=1)`` would be on at ``a == split`` too
    and count the slope twice at that one point, which is 1048576 in float32.
    """
    return lucid.where(a > split, a * (1.0 / split), 1.0)


def _asinh_magnitude(a: Tensor) -> Tensor:
    """``asinh`` of ``a >= 0``, to a few ulp over the whole half line."""
    split = _split(a)
    near = lucid.clamp(a, max=split)
    q = near / (1.0 + lucid.sqrt(near * near + 1.0))
    # the tail is zero below the split, so it adds no rounding to a small answer
    return lucid.log1p(near * (1.0 + q)) + lucid.log(_beyond(a, split))


def _atanh_magnitude(a: Tensor) -> Tensor:
    """``atanh`` of ``a >= 0``: ``inf`` at 1, NaN beyond."""
    return 0.5 * lucid.log1p((a + a) / (1.0 - a))


def _acosh_value(x: Tensor) -> Tensor:
    """``acosh`` of a real float tensor; NaN below 1, to a few ulp above."""
    split = _split(x)
    t = lucid.clamp(x, max=split) - 1.0
    # the two roots rather than one root of the product: below 1 the first is NaN,
    # where ``t * (t + 2)`` is positive again once ``t < -2``
    near = lucid.log1p(t + lucid.sqrt(t) * lucid.sqrt(t + 2.0))
    return near + lucid.log(_beyond(x, split))


def _odd_value(magnitude: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """The odd function whose value on ``|x|`` is ``magnitude``; a signed zero stays."""
    return lucid.copysign(magnitude(lucid.abs(x)), x)


def _odd_traced(magnitude: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """:func:`_odd_value` from ops a compile can differentiate, unit slope at 0.

    ``copysign`` has no gradient a compiled step takes (it lost the slope of a small
    value), and ``|x|`` has none at the origin, where an odd function of unit slope is
    1.  The sign is a ``where`` on ``x < 0`` and the origin takes ``x`` itself, which
    keeps a signed zero as well.
    """
    v = magnitude(lucid.abs(x))
    return lucid.where(x == 0.0, x, lucid.where(x < 0.0, -v, v))


def _asinh_slope(x: Tensor) -> Tensor:
    """``1 / sqrt(1 + x^2)``: ``1 / |x|`` past the split, where ``x^2`` would overflow."""
    split = _split(x)
    near = lucid.clamp(x, min=-split, max=split)
    return lucid.rsqrt(near * near + 1.0) / _beyond(lucid.abs(x), split)


def _acosh_slope(x: Tensor) -> Tensor:
    """``1 / sqrt(x^2 - 1)``, as the product of two roots so that ``x^2`` is never formed."""
    return 1.0 / (lucid.sqrt(x - 1.0) * lucid.sqrt(x + 1.0))


def _atanh_slope(x: Tensor) -> Tensor:
    """``1 / (1 - x^2)``, exact next to ±1 and with a derivative that is exact next to 0.

    ``(1 - x)(1 + x)`` is exact next to 1 where ``1 - x^2`` is not, but its derivative,
    ``-(1 + x) + (1 - x)``, rounds to 0 below 6e-8 where the true one is ``-2x``.  The
    denominator takes its value from the product and its derivative from ``1 - x^2``:
    ``flat - flat.detach()`` is a zero that carries ``-2x``, with ``x`` clamped inside
    ``flat`` so that the zero is finite for an infinite ``x``.
    """
    reach = 2.0**500 if x.dtype == lucid.float64 else 2.0**60
    held = x.detach()
    exact = (1.0 - held) * (1.0 + held)
    xk = lucid.clamp(x, min=-reach, max=reach)
    flat = 1.0 - xk * xk
    return 1.0 / (exact + (flat - flat.detach()))


def _asinh_value(x: Tensor) -> Tensor:
    return _odd_value(_asinh_magnitude, x)


def _atanh_value(x: Tensor) -> Tensor:
    return _odd_value(_atanh_magnitude, x)


def _asinh_traced(x: Tensor) -> Tensor:
    return _odd_traced(_asinh_magnitude, x)


def _atanh_traced(x: Tensor) -> Tensor:
    return _odd_traced(_atanh_magnitude, x)


@final
class _Asinh(Function):
    """``asinh`` with its exact slope ``1 / sqrt(1 + x^2)``, a graph of ordinary ops."""

    _trace_form = staticmethod(_asinh_traced)

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return _asinh_value(x.detach())

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return grad * _asinh_slope(x)


@final
class _Acosh(Function):
    """``acosh`` with its exact slope ``1 / sqrt(x^2 - 1)``, a graph of ordinary ops."""

    _trace_form = staticmethod(_acosh_value)

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return _acosh_value(x.detach())

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return grad * _acosh_slope(x)


@final
class _Atanh(Function):
    """``atanh`` with its exact slope ``1 / (1 - x^2)``, a graph of ordinary ops."""

    _trace_form = staticmethod(_atanh_traced)

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return _atanh_value(x.detach())

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return grad * _atanh_slope(x)


def _apply_asinh(x: Tensor) -> Tensor:
    out = _Asinh.apply(x)
    assert isinstance(out, Tensor)
    return out


def _apply_acosh(x: Tensor) -> Tensor:
    out = _Acosh.apply(x)
    assert isinstance(out, Tensor)
    return out


def _apply_atanh(x: Tensor) -> Tensor:
    out = _Atanh.apply(x)
    assert isinstance(out, Tensor)
    return out


def _asinh(input: Tensor) -> Tensor:
    """``lucid.asinh``: a 16-bit input is evaluated in float32, a complex one by the textbook form."""
    x = _float_input(input)
    if x.is_complex():
        return lucid.log(x + lucid.sqrt(x * x + 1.0))
    return _in_float32(_apply_asinh, x)


def _acosh(input: Tensor) -> Tensor:
    """``lucid.acosh``: a 16-bit input is evaluated in float32, a complex one by the textbook form."""
    x = _float_input(input)
    if x.is_complex():
        return lucid.log(x + lucid.sqrt(x * x - 1.0))
    return _in_float32(_apply_acosh, x)


def _atanh(input: Tensor) -> Tensor:
    """``lucid.atanh``: a 16-bit input is evaluated in float32, a complex one by the textbook form."""
    x = _float_input(input)
    if x.is_complex():
        return lucid.log((1.0 + x) / (1.0 - x)) * 0.5
    return _in_float32(_apply_atanh, x)
