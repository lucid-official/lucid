"""``special.ndtri``: the inverse normal CDF (Beasley-Springer-Moro) and its
float64 refinement.  Kept out of ``lucid/special/__init__.py``, which is over
its size cap."""

import math
from typing import TYPE_CHECKING

import lucid
from lucid.special._erfcx import _erfcx_stable

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_RSQRT2 = 1.0 / math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

#: Where the approximation hands the central rational over to the tail one.
_P_LOW = 0.02425

# Beasley-Springer-Moro coefficients for the inverse normal CDF.  Hosted as
# module-level constants so ``ndtri`` stays a tight composite.
_NDTRI_A = (
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.383577518672690e2,
    -3.066479806614716e1,
    2.506628277459239,
)
_NDTRI_B = (
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1,
)
_NDTRI_C = (
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
)
_NDTRI_D = (7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416)


def _ndtri(p: Tensor) -> Tensor:
    """The Beasley-Springer-Moro approximation, good to about 1.15e-9."""
    plow = _P_LOW
    phigh = 1.0 - plow
    in_open = (p > 0.0) & (p < 1.0)
    p_safe = lucid.where((p <= 0.0) | (p >= 1.0), lucid.full_like(p, 0.5), p)

    q = p_safe - 0.5
    r_central = q * q
    num_c = (
        (
            ((_NDTRI_A[0] * r_central + _NDTRI_A[1]) * r_central + _NDTRI_A[2]) * r_central
            + _NDTRI_A[3]
        )
        * r_central
        + _NDTRI_A[4]
    ) * r_central + _NDTRI_A[5]
    den_c = (
        (
            ((_NDTRI_B[0] * r_central + _NDTRI_B[1]) * r_central + _NDTRI_B[2]) * r_central
            + _NDTRI_B[3]
        )
        * r_central
        + _NDTRI_B[4]
    ) * r_central + 1.0
    central = q * num_c / den_c

    p_tail_in = lucid.where(p_safe < 0.5, p_safe, 1.0 - p_safe)
    r_tail = lucid.sqrt(-2.0 * lucid.log(p_tail_in))
    num_t = (
        (((_NDTRI_C[0] * r_tail + _NDTRI_C[1]) * r_tail + _NDTRI_C[2]) * r_tail + _NDTRI_C[3])
        * r_tail
        + _NDTRI_C[4]
    ) * r_tail + _NDTRI_C[5]
    den_t = (
        ((_NDTRI_D[0] * r_tail + _NDTRI_D[1]) * r_tail + _NDTRI_D[2]) * r_tail + _NDTRI_D[3]
    ) * r_tail + 1.0
    tail_low = num_t / den_t
    tail = lucid.where(p_safe < 0.5, tail_low, -tail_low)

    # Endpoints map to ±inf; anything outside [0, 1] or NaN maps to NaN.
    # Both inner branches ran on p_safe in (0, 1), so neither leaks NaN to grad.
    in_central = (p_safe > plow) & (p_safe < phigh)
    inside = lucid.where(in_central, central, tail)
    boundary = lucid.where(
        p == 0.0,
        lucid.full_like(p, -math.inf),
        lucid.where(p == 1.0, lucid.full_like(p, math.inf), lucid.full_like(p, math.nan)),
    )
    return lucid.where(in_open, inside, boundary)


def _ndtri_double(p: Tensor) -> Tensor:
    """:func:`_ndtri` for float64, with one Newton step toward the exact root.

    The approximation is good to 1.15e-9 relative, which is float32's
    business and nowhere near float64's.  Its square, ``~1e-18``, is what one
    Newton step leaves, so one step is all float64 needs.  The step is
    taken where each stretch of the line keeps full relative precision:

    * **Centre** (``p`` in ``[p_lo, 1 - p_lo]``): on ``Phi(x) - p``, written
      ``erf(x/sqrt 2)/2 - (p - 1/2)``.  ``p - 1/2`` is exact, so the
      cancellation costs nothing as ``x`` goes to 0.
    * **Tails**: on ``log Phi(x) - log p`` with
      ``Phi(x) = erfcx(t) exp(-t^2) / 2``, ``t = -x/sqrt 2``.  ``Phi - p``
      itself would underflow with ``p``, and ``exp(x^2/2)`` in the step
      overflows from ``|x|`` = 37.7; the log form has neither, and its
      step ``Phi/phi = sqrt(pi/2) erfcx(t)`` stays near ``1/|x|``.  The
      upper tail is the lower one reflected: ``1 - p`` is exact for
      ``p >= 1/2``.

    Both stretches are evaluated everywhere and ``where`` picks; each is fed
    its own clamped copy of the approximation so the one that loses stays
    finite (an inf there is a NaN in the gradient).  The endpoints and NaN
    keep the approximation's ±inf and NaN.
    """
    approx = _ndtri(p)
    in_open = (p > 0.0) & (p < 1.0)
    p_safe = lucid.where(in_open, p, lucid.full_like(p, 0.5))
    x0 = lucid.where(in_open, approx, lucid.zeros_like(approx))
    low = p_safe < 0.5

    xc = lucid.clamp(x0, -2.0, 2.0)
    centre = xc - (0.5 * lucid.erf(xc * _RSQRT2) - (p_safe - 0.5)) * (
        _SQRT_2PI * lucid.exp(0.5 * xc * xc)
    )

    t = lucid.clamp(lucid.abs(x0), min=1.9) * _RSQRT2
    mills = _erfcx_stable(t)
    p_low_side = lucid.where(low, p_safe, 1.0 - p_safe)
    log_cdf = lucid.log(0.5 * mills) - t * t
    lower = -t * math.sqrt(2.0) - (log_cdf - lucid.log(p_low_side)) * (0.5 * _SQRT_2PI * mills)

    in_centre = (p_safe > _P_LOW) & (p_safe < 1.0 - _P_LOW)
    refined = lucid.where(in_centre, centre, lucid.where(low, lower, -lower))
    return lucid.where(in_open, refined, approx)
