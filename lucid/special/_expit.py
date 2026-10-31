"""``special.expit`` on a complex tensor: the logistic function, composed from
real and imaginary parts (LCD-772).  The engine's ``sigmoid`` is real-only, and
so is its ``exp``, so the complex form cannot be written as ``1 / (1 + exp(-z))``.
Kept out of ``lucid/special/__init__.py``, which is over its size cap."""

import lucid
from lucid._tensor.tensor import Tensor


def _complex_expit(z: Tensor) -> Tensor:
    r""":math:`1 / (1 + e^{-z})` for complex ``z = a + ib``, in real arithmetic.

    With :math:`s = e^{-|a|} \in [0, 1]` the denominator of the rationalised
    form is :math:`D = |1 + e^{-z}|^2 = 1 + 2 s \cos b + s^2` for either sign
    of :math:`a`, and

    .. math::

        \sigma(z) = \frac{(1 + s\cos b) + i\,s\sin b}{D} \;(a \ge 0),
        \qquad
        \sigma(z) = \frac{(s\cos b + s^2) + i\,s\sin b}{D} \;(a < 0),

    the second being the first with :math:`z \to -z` taken through
    :math:`\sigma(z) = e^{z} / (1 + e^{z})`.  Neither branch forms an
    :math:`e^{x}` with :math:`x > 0`, so ``a = ±inf`` gives ``1`` / ``0``
    rather than ``inf / inf``.  :math:`D` is written
    :math:`(1 - s)^2 + 4 s \cos^2(b / 2)`, which keeps its digits next to the
    poles :math:`z = i\pi (2k + 1)` where the plain sum cancels.
    """
    a, b = lucid.real(z), lucid.imag(z)
    s = lucid.exp(-lucid.abs(a))
    half_cos = lucid.cos(0.5 * b)
    sq = 2.0 * half_cos * half_cos  # 1 + cos b, without the cancellation
    denom = (1.0 - s) * (1.0 - s) + 2.0 * s * sq
    # 1 + s cos b = (1 - s) + s (1 + cos b);  s cos b + s^2 = s (s - 1 + 1 + cos b)
    numer = lucid.where(a >= 0.0, (1.0 - s) + s * sq, s * ((s - 1.0) + sq))
    return lucid.complex(numer / denom, s * lucid.sin(b) / denom)
