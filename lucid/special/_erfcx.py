"""``special.erfcx``: the stable scaled complementary error function and its
analytic derivative (LCD-656, LCD-1030).  Kept out of ``lucid/special/__init__.py``,
which is over its size cap."""

import math
from typing import TYPE_CHECKING, final, override

import lucid
from lucid._ops.composite._opmath import _in_float32, _real

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _plain_erfcx(x: Tensor) -> Tensor:
    return _in_float32(_erfcx_stable, _real(x))


_CUTOFF = 2.0
# The slope's own cutoff is lower: ``2 x y - 2/sqrt(pi)`` already cancels a digit
# before the value's cutoff, and the fraction converges there with the terms
# ``_erfcx_tail`` gives it.
_SLOPE_CUTOFF = 1.5
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


@final
class _Erfcx(lucid.autograd.Function):
    """``erfcx`` with the reference's analytic derivative (LCD-656).

    The value comes from :func:`_erfcx_stable`. The derivative is
    ``erfcx'(x) = 2 x erfcx(x) - 2/sqrt(pi)`` (:func:`_erfcx_slope`), built from
    the saved output, so the second derivative ``2 erfcx(x) + 2 x erfcx'(x)``
    follows. That gives the reference's d1 = -inf and d2 = +inf at -inf, and NaN
    (inf * 0) at +inf, where differentiating the continued fraction produced NaN
    and 0 instead.
    """

    _trace_form = staticmethod(_plain_erfcx)

    @override
    @staticmethod
    def forward(ctx: lucid.autograd.FunctionCtx, x: Tensor) -> Tensor:
        y = _plain_erfcx(x.detach())
        ctx.save_for_backward(x, y)
        return y

    @override
    @staticmethod
    def backward(ctx: lucid.autograd.FunctionCtx, grad: Tensor) -> Tensor:
        x, y = ctx.saved_tensors
        return grad * _in_float32(_erfcx_slope, x, y)


def _erfcx_slope(x: Tensor, y: Tensor) -> Tensor:
    """``erfcx'(x)`` from ``x`` and the saved ``y = erfcx(x)``.

    Below ``_SLOPE_CUTOFF`` it is ``2 x y - 2/sqrt(pi)``.  Beyond it both terms
    tend to ``2/sqrt(pi)`` and their difference to ``1 / (sqrt(pi) x²)``, so the
    subtraction lost a digit at 2, was 1.7% off at 370 and 0 from 1e4 in
    float32, and its derivative was worse (12% off at 37, five-fold at 100,
    twenty orders at 1e10).  With ``y = 1 / (sqrt(pi) (x + f))``, ``f`` the
    continued fraction's tail (:func:`_erfcx_tail`), the same slope is

        erfcx'(x) = -2/sqrt(pi) · f / (x + f)

    a quotient of positive numbers that never forms the difference, and whose
    own derivative (the chain rule through the fraction) has no subtraction
    either.  It is evaluated from ``x`` alone, so every order of derivative
    reads the tail form.  A non-finite ``x`` keeps the direct form: ``-inf``
    and NaN are its answers, and ``+inf`` is the reference's ``inf * 0``.

    Each form runs on ``x`` replaced by the cutoff where the other one is
    selected, as in :func:`_erfcx_stable`: ``2 x`` overflows float32 at 1.7e38
    while ``y`` is a number, and the masked branch's ``inf * 0`` would be a NaN
    in the gradient.
    """
    use_tail = (x >= _SLOPE_CUTOFF) & lucid.isfinite(x)
    cutoff = lucid.full_like(x, _SLOPE_CUTOFF)
    direct = 2.0 * lucid.where(use_tail, cutoff, x) * y - _TWO_OVER_SQRT_PI
    big = lucid.where(use_tail, x, cutoff)
    tail = _erfcx_tail(big, deep=True)
    return lucid.where(use_tail, -(_TWO_OVER_SQRT_PI * tail) / (big + tail), direct)


def _erfcx_tail(big: Tensor, *, deep: bool = False) -> Tensor:
    """The continued fraction's tail ``f`` of ``erfcx(x) = 1/sqrt(pi) / (x + f)``.

    ``f = ½/(x + 1/(x + 3/2/(x + 2/(x + …))))``.  It is the small part of the
    denominator (``~ 1 / (2x)``), which is why the slope is read from it and not
    from ``y``.  The value takes 24 terms in float32 and 60 in float64, good
    from ``x = 2``.  The slope takes 48 and 120 (``deep``): from 1.5, and with
    its derivatives, whose truncation error falls an order of magnitude slower
    per order than the value's (at 2, 60 terms leave the second derivative
    2e-14 off and the third 3e-13; at 1.5 the value alone is 1e-15 off).
    """
    f64 = big.dtype == lucid.float64
    terms = (120 if f64 else 48) if deep else (60 if f64 else 24)
    frac = lucid.zeros_like(big)
    for k in range(terms, 0, -1):
        frac = (0.5 * k) / (big + frac)
    return frac


def _erfcx_stable(x: Tensor) -> Tensor:
    """``exp(x²) erfc(x)``, without ever forming ``exp(x²)``.

    From x = 2 on the product is evaluated through the continued fraction

        erfcx(x) = 1/√π · 1/(x + ½/(x + 1/(x + 3/2/(x + 2/(x + …)))))

    which never leaves the interval the answer lives in.  At 24 terms it is
    good to about 1e-10 at x = 2 and 1e-12 at x = 2.5, which is float32's
    business; float64 takes 60, good to its last digit from x = 2 on (the
    log-normal tails and ``ndtri`` read it there).  Below that the direct
    form is used; ``exp(4)`` is an ordinary number.

    The switch used to sit at x = 4, which left ``[2, 4)`` to the direct
    form — and there ``erfc`` has lost its relative precision: its answer
    is a small multiple of float32's spacing below 1 (``erfc(3.9)`` comes
    back as 6.0e-8, where the true value is 3.5e-8), so ``erfcx(3.9)``
    was 0.240 instead of 0.140.  At x = 2 the direct form still holds to
    about 5e-6.

    Both branches are evaluated — ``where`` selects, it does not
    short-circuit — so each is clamped onto its own domain first.  An inf
    or a NaN in the branch that loses is still a NaN in the *gradient*,
    which is how a where-guarded formula usually goes wrong.
    """
    xs = lucid.clamp(x, max=_CUTOFF)
    small = lucid.exp(xs * xs) * lucid.erfc(xs)
    big = lucid.clamp(x, min=_CUTOFF)
    large = (1.0 / math.sqrt(math.pi)) / (big + _erfcx_tail(big))
    return lucid.where(x < _CUTOFF, small, large)


# ── Modified Bessel I₀ / I₁ family ─────────────────────────────────────────
