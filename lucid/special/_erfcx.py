"""``special.erfcx``: the stable scaled complementary error function and its
analytic derivative (LCD-656).  Kept out of ``lucid/special/__init__.py``,
which is over its size cap."""

import math
from typing import TYPE_CHECKING, final, override

import lucid
from lucid._ops.composite._opmath import _in_float32, _real

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _plain_erfcx(x: Tensor) -> Tensor:
    return _in_float32(_erfcx_stable, _real(x))


@final
class _Erfcx(lucid.autograd.Function):
    """``erfcx`` with the reference's analytic derivative (LCD-656).

    The value comes from :func:`_erfcx_stable`. The derivative is
    ``2 x erfcx(x) - 2/sqrt(pi)``, built from the saved output, so the second
    derivative ``2 erfcx(x) + 2 x erfcx'(x)`` follows. That gives the
    reference's d1 = -inf and d2 = +inf at -inf, and NaN (inf * 0) at +inf,
    where differentiating the continued fraction produced NaN and 0 instead.
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
        return grad * (2.0 * x * y - 2.0 / math.sqrt(math.pi))


def _erfcx_stable(x: Tensor) -> Tensor:
    """``exp(x²) erfc(x)``, without ever forming ``exp(x²)``.

    From x = 2 on the product is evaluated through the continued fraction

        erfcx(x) = 1/√π · 1/(x + ½/(x + 1/(x + 3/2/(x + 2/(x + …)))))

    which at 24 terms is already good to about 1e-10 at x = 2, and never
    leaves the interval the answer lives in.  Below that the direct form
    is used; ``exp(4)`` is an ordinary number.

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
    cutoff = 2.0
    xs = lucid.clamp(x, max=cutoff)
    small = lucid.exp(xs * xs) * lucid.erfc(xs)
    big = lucid.clamp(x, min=cutoff)
    frac = lucid.zeros_like(big)
    for k in range(24, 0, -1):
        frac = (0.5 * k) / (big + frac)
    large = (1.0 / math.sqrt(math.pi)) / (big + frac)
    return lucid.where(x < cutoff, small, large)


# ── Modified Bessel I₀ / I₁ family ─────────────────────────────────────────
