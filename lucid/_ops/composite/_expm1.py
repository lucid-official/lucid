"""``expm1``'s value: Kahan's ratio, kept apart from its gradient (LCD-772)."""

import math

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _REAL_FLOATS, _binary_operands, _wrap
from lucid._tensor.tensor import Tensor
from lucid._types import TensorOrScalar


def _expm1(input: Tensor) -> Tensor:
    """:math:`e^x - 1` to full relative precision at every magnitude.

    The value is Kahan's form; the gradient is ``exp(x)``'s.
    """
    # Kahan's form: with u = exp(x), (u - 1) * x / log(u) cancels the rounding
    # of u against the same rounding in log(u), so it keeps full relative
    # precision near 0.  The ratio x / log(u) is grouped first: it is ~1 at
    # every magnitude, whereas (u - 1) * x leaves the dtype's range while the
    # answer is still finite (float16 at x = 10, float32 at 85, float64 at 705
    # all came back inf).  The guards take u == 1 (x underflowed), u == 0 and
    # u == inf, where the ratio is 0/0 or inf/inf.
    with lucid.no_grad():
        u = lucid.exp(input)
        accurate = lucid.where(u == 1.0, input, (u - 1.0) * (input / lucid.log(u)))
        # Kahan's second guard: once u - 1 rounds to -1, -1 is the answer.  The
        # ratio would take log(u) of a subnormal u, whose rounding is no longer
        # relative — expm1(-100) in float32 came out -1.000169.
        accurate = lucid.where(u - 1.0 == -1.0, u - 1.0, accurate)
        accurate = lucid.where((u == 0.0) | ~lucid.isfinite(u), u - 1.0, accurate)
    # The value is the accurate one; the gradient is exp(x)'s, carried by the
    # naive form under it, so the guards' 0/0 branches never reach backward.
    naive = lucid.exp(input) - 1.0
    with lucid.no_grad():
        correction = lucid.where(
            lucid.isfinite(naive), accurate - naive, lucid.zeros_like(accurate)
        )
    # Where exp(x) rounds to 1, x + 0.5 * c * (c - c.detach()) with clamped c
    # equals x bit-for-bit (preserving expm1(-0) = -0 and keeping ±inf finite
    # in MulBackward) while carrying both the unit slope and unit curvature of
    # exp(x) at the origin for higher-order autograd.
    c = lucid.where(u == 1.0, input, lucid.zeros_like(input))
    small_branch = input + 0.5 * c * (c - c.detach())
    return lucid.where(u == 1.0, small_branch, naive + correction)


def _erfc_single(input: Tensor) -> Tensor:
    """erfc by the Numerical Recipes Chebyshev fit, good to 1.2e-7."""
    z = lucid.where(input >= 0.0, input, -input)
    t = 1.0 / (1.0 + 0.5 * z)
    poly = lucid.full_like(t, 0.17087277)
    for c in (
        -0.82215223,
        1.48851587,
        -1.13520398,
        0.27886807,
        -0.18628806,
        0.09678418,
        0.37409196,
        1.00002368,
        -1.26551223,
    ):
        poly = c + t * poly
    tail = t * lucid.exp(poly - z * z)
    return lucid.where(input >= 0.0, tail, 2.0 - tail)


def _erfc_double(x: Tensor) -> Tensor:
    """erfc in float64 (1 - erf below 3, continued fraction above)."""
    z = lucid.where(x >= 0.0, x, -x)
    head = 1.0 - lucid.erf(lucid.clamp(z, max=3.0))
    far = lucid.clamp(z, min=3.0)
    frac = lucid.zeros_like(far)
    for k in range(60, 0, -1):
        frac = (0.5 * k) / (far + frac)
    tail = lucid.exp(-far * far) / (math.sqrt(math.pi) * (far + frac))
    value = lucid.where(z < 3.0, head, tail)
    return lucid.where(x >= 0.0, value, 2.0 - value)


def _sgn(input: Tensor) -> Tensor:
    """Element-wise sign (z / |z| for complex, sign(x) for real/int)."""
    if not input.is_complex():
        return lucid.sign(input)
    mag = lucid.abs(input)
    is_zero = mag == 0.0
    safe_mag = lucid.where(is_zero, lucid.ones_like(mag), mag)
    zero_mask = lucid.where(is_zero, lucid.zeros_like(mag), lucid.ones_like(mag))
    return (input * zero_mask) / safe_mag


def _logaddexp2(input: TensorOrScalar, other: TensorOrScalar) -> Tensor:
    """Base-2 logarithm of the sum of exponentiations of the inputs."""
    a_impl, b_impl = _binary_operands(input, other)
    if a_impl.dtype not in _REAL_FLOATS:
        raise _C_engine.NotImplementedError(
            f"logaddexp2: not implemented for {_wrap(a_impl).dtype}; cast to a floating dtype first"
        )
    a, b = _wrap(a_impl), _wrap(b_impl)
    ln2 = math.log(2.0)
    hi = lucid.maximum(a, b)
    m = lucid.where(lucid.isfinite(hi), hi, lucid.zeros_like(hi)).detach()
    return m + lucid.logaddexp((a - m) * ln2, (b - m) * ln2) * (1.0 / ln2)
