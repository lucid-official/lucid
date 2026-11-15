"""``sin(u) / u`` across its removable singularity, for
``lucid.special.spherical_bessel_j0`` (LCD-1030).  ``lucid.sinc`` is the same
function at ``u = pi x`` and still carries its own copy of this series in
``elementwise.py``."""

import math
from typing import TYPE_CHECKING

import lucid

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# ``sin(u) / u = sum (-1)^k u^(2k) / (2k + 1)!``, highest power first.  Twelve
# terms hold ``|u| < 1`` to float64's last digit (the next is 1e-28).
_SIN_OVER_SERIES: tuple[float, ...] = tuple(
    (-1.0) ** k / math.factorial(2 * k + 1) for k in range(12, -1, -1)
)


def _sin_over(u: Tensor) -> Tensor:
    """``sin(u) / u``, 1 at 0, every derivative exact through the origin.

    For ``|u| < 1`` the value is the Maclaurin series in ``u²``: ``sin(u) / u``
    and its derivatives (``cos(u) / u - sin(u) / u²`` ...) cancel to nothing
    there, and each order has a larger power of ``u`` in its denominator, so the
    quotient form loses range and digits long before it loses the origin
    (float32 at ``u = 1e-4`` had a slope of 0 where it is ``-3.3e-5``, and a
    curvature of 0 or 2 where it is ``-1/3``).  A polynomial has neither the
    removable singularity nor the cancellation, to any order.

    Each branch runs on its own interval, so neither forms an inf or a NaN the
    ``where`` would mask in the value and hand to the gradient: ``sin(v) / v``
    at the ``v = 1`` that stands in for a small ``u``, the series on a ``u``
    that is 0 outside ``(-1, 1)``.  A NaN stays NaN.
    """
    small = lucid.abs(u) < 1.0
    safe = lucid.where(small, lucid.full_like(u, 1.0), u)
    val = lucid.sin(safe) / safe
    u2 = lucid.where(small, u, lucid.zeros_like(u)) ** 2
    series = lucid.full_like(u, _SIN_OVER_SERIES[0])
    for c in _SIN_OVER_SERIES[1:]:
        series = series * u2 + c
    return lucid.where(small, series, val)
