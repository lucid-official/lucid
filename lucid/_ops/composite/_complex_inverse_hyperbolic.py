"""Principal inverse hyperbolic functions for Metal complex tensors."""

import math

import lucid
from lucid._tensor.tensor import Tensor


def _root_components(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Cartesian components of the principal square root, including the lips of its cut."""
    # On the imaginary axis take the derivative of the selected square-root
    # formula. abs(x) at x=0 would incorrectly give zero in the second VJP.
    positive = x >= 0
    magnitude = lucid.where(positive, x, -x)
    hi = lucid.sqrt(0.5 * lucid.hypot(x, y) + 0.5 * magnitude)
    small = y / lucid.where(hi == 0, 1.0, 2.0 * hi)
    sign = lucid.copysign(lucid.ones_like(y), y)
    return (lucid.where(positive, hi, sign * small), lucid.where(positive, small, sign * hi))


def _ellipse(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Return sinh(u) and cosh(u)*sin(v), where cosh(u+i*v)=x+i*y in quadrant I."""
    pr, pi = _root_components(x + 1.0, y)
    mr, mi = _root_components(x - 1.0, y)
    # sqrt(z+1) = sqrt(2)*cosh(w/2), sqrt(z-1) = sqrt(2)*sinh(w/2).
    # The real part of conj(p)*m and imaginary part of p*m are sums of like signs.
    return pr * mr + pi * mi, pr * mi + pi * mr


def _tail_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Divide by a large positive number, rounding even a subnormal quotient on Metal."""
    scaled = numerator / (denominator * 2.0**-64)
    return lucid.ldexp(scaled, lucid.full_like(scaled, -64))


def _tail_angle(y: Tensor, x: Tensor) -> Tensor:
    small = (x > 0.0) & y.isfinite() & (y <= x * 2.0**-16)
    # Here atan(y/x) and y/x round alike, including below the smallest normal.
    ratio = _tail_ratio(lucid.where(small, y, 0.0), lucid.where(small, x, 1.0))
    return lucid.where(small, ratio, lucid.atan2(y, x))


def _inverse_cosh_coordinates(x: Tensor, y: Tensor, rotated: bool) -> tuple[Tensor, Tensor]:
    """Read the angle or its complement directly, including when it is smaller than one ulp of pi/2."""
    scale = lucid.maximum(x.abs(), y.abs())
    # Beyond this radius the correction to log(2*z) is O(1/|z|^2).
    far = scale >= 2.0**20
    xx = lucid.where(far, 1.0, x.abs())
    yy = lucid.where(far, 1.0, y.abs())
    sinh_u, perpendicular = _ellipse(xx, yy)
    safe = lucid.where(far, scale, 1.0)
    ratio = lucid.minimum(x.abs(), y.abs()) / safe
    tail = math.log(2.0) + lucid.log(safe) + 0.5 * lucid.log1p(ratio * ratio)
    u = lucid.where(far, tail, lucid.asinh(sinh_u))
    if rotated:
        angle = lucid.where(far, _tail_angle(x.abs(), y.abs()), lucid.atan2(xx, perpendicular))
    else:
        cosine = lucid.copysign(xx, x)
        angle = lucid.where(far, _tail_angle(y.abs(), x), lucid.atan2(perpendicular, cosine))
    return lucid.where(x.isinf() | y.isinf(), math.inf, u), angle


def asinh_value(z: Tensor) -> Tensor:
    """Rotate the coordinates into the principal acosh half-plane. Recover the signs of both components from the original input, including signed zero on an axis."""
    u, angle = _inverse_cosh_coordinates(z.imag, z.real, True)
    angle = lucid.where(z.imag == 0, 0.0, angle)
    return lucid.complex(lucid.copysign(u, z.real), lucid.copysign(angle, z.imag))


def acosh_value(z: Tensor) -> Tensor:
    """Return the nonnegative real part and the principal angle in [-pi, pi]. The sign of the imaginary component selects the lip of the negative real cut."""
    u, angle = _inverse_cosh_coordinates(z.real, z.imag, False)
    angle = lucid.where((z.real == 0) & z.imag.isnan(), math.pi / 2, angle)
    return lucid.complex(u, lucid.copysign(angle, z.imag))


def atanh_value(z: Tensor) -> Tensor:
    """Evaluate the real logarithmic distance ratio and the signed half-angle separately. This preserves the small real component near both zero and infinity."""
    x, y = z.real.abs(), z.imag.abs()
    # Re atanh(z) = log1p(4*x/|1-z|^2)/4. Work with the square root of
    # the ratio until its size is known, so neither the pole nor the tail overflows.
    distance = lucid.hypot(1.0 - x, y)
    numerator = 2.0 * lucid.sqrt(x)
    large = distance < numerator * 2.0**-30
    scale = lucid.maximum(lucid.maximum(x, y), lucid.ones_like(x))
    xs, ys, one = x / scale, y / scale, 1.0 / scale
    dx = (1.0 - x) / scale
    norm_sq = lucid.where(large, 1.0, dx * dx + ys * ys)
    ratio_sq = 4.0 * (xs / scale) / norm_sq
    safe_n = lucid.where(large, numerator, 1.0)
    safe_d = lucid.where(large, distance, 1.0)
    real = lucid.where(
        large, 0.5 * (lucid.log(safe_n) - lucid.log(safe_d)), 0.25 * lucid.log1p(ratio_sq)
    )
    # Beyond this scale Re atanh(z) rounds to Re(1/z). Divide once
    # more using a scaled result so x/|z|^2 can be subnormal.
    far = scale >= 2.0**20
    norm = lucid.where(far, xs * xs + ys * ys, 1.0)
    tail = _tail_ratio(xs / norm, scale)
    real = lucid.where(far, tail, real)
    angle = 0.5 * lucid.atan2((2.0 * ys) * one, dx * (one + xs) - ys * ys)
    infinite = x.isinf() | y.isinf()
    real = lucid.where(infinite | (x == 0), 0.0, real)
    angle = lucid.where(infinite & ~y.isnan(), math.pi / 2, angle)
    return lucid.complex(lucid.copysign(real, z.real), lucid.copysign(angle, z.imag))


def _reciprocal(x: Tensor, y: Tensor) -> Tensor:
    """Reciprocal without squaring a large or small component."""
    scale = lucid.maximum(x.abs(), y.abs())
    xs, ys = x / scale, y / scale
    norm = xs * xs + ys * ys
    return lucid.complex((xs / norm) / scale, (-ys / norm) / scale)


def _root(x: Tensor, y: Tensor) -> Tensor:
    return lucid.complex(*_root_components(x, y))


def asinh_slope(z: Tensor) -> Tensor:
    """Read the derivative from the two principal square roots of 1+i*z and 1-i*z. Their product fixes the sign on either side of the imaginary cuts."""
    # sqrt(1+i*z)*sqrt(1-i*z) selects the analytic root on both cuts.
    x, y = z.real, z.imag
    product = _root(1.0 - y, x) * _root(1.0 + y, -x)
    return _reciprocal(product.real, product.imag)


def acosh_slope(z: Tensor) -> Tensor:
    """Take the reciprocal of the product of the principal roots of z-1 and z+1. The separate roots retain the derivative's sheet to the left of the real cut."""
    x, y = z.real, z.imag
    product = _root(x - 1.0, y) * _root(x + 1.0, y)
    return _reciprocal(product.real, product.imag)


def atanh_slope(z: Tensor) -> Tensor:
    """Multiply the reciprocals of 1-z and 1+z. Forming their product first can overflow even when its reciprocal fits in the output."""
    x, y = z.real, z.imag
    # 1 / ((1-z)*(1+z)); a reciprocal of each factor keeps a huge z in range.
    return _reciprocal(1.0 - x, -y) * _reciprocal(1.0 + x, y)
