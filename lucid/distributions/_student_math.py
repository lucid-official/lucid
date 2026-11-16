"""The range-safe pieces of Student's t: a gamma ratio and a log-kernel that do not cancel.

Student's t log-density is

    log p = [lgamma(a + 1/2) - lgamma(a)] - 1/2 log(pi nu) - log sigma
            - (nu + 1)/2 log(1 + z^2 / nu),      a = nu / 2

and written that way each term fails for a large ``nu`` while the answer is
an ordinary number (about ``-1.42`` at ``z = 1``).  ``lgamma`` of a large
argument is large, so their difference is a rounding artefact: at
``nu = 1e8`` it is ``-9.78`` for ``-1.42`` in float32, and the ``1e-7`` it is
off in float64 is the same cancellation.  The two helpers below own the two
cancelling groups; the density is ``E(a) - 1/2 log(2 pi) - log sigma - T``.

* :func:`_gamma_half_excess` is ``E(a) = lgamma(a + 1/2) - lgamma(a) -
  1/2 log(a)``.  The ``1/2 log(a)`` is the large part of the difference, and
  taking it out *before* the subtraction is what makes ``1/2 log(a)`` and
  ``1/2 log(pi nu)`` cancel exactly instead of in rounded floats.  ``E`` is
  small (``-1/(8a)``) and has an asymptotic series in ``1/a``.
* :func:`_student_exponent` is ``T = (nu + 1)/2 log(1 + z^2/nu)``.  Its value
  is fine; its derivative in ``nu`` is a difference of two terms that both
  tend to ``z^2 / (2 nu)`` and leave ``O(z^4 / nu^2)``.

``_digamma_half_gap`` is ``E``'s derivative companion for the entropy.
None of them calls ``lgamma`` or ``digamma``: those cancel the same way in
their own derivative, and an engine ``lgamma`` is not the thing to measure a
``1e-9`` residue with.
"""

import lucid
from lucid._ops.composite._gamma import _horner_float
from lucid._ops.composite._opmath import _in_float32
from lucid._tensor.tensor import Tensor

# lgamma(a + 1/2) - lgamma(a) - log(a)/2 = sum_k  _E_COEFFS[k] / a**(2k + 1)
# (Stirling's series at the shift 1/2: (2^(1-2k) - 2) B_2k / (2k (2k - 1))).
_E_COEFFS: tuple[float, ...] = (
    -1 / 8,
    1 / 192,
    -1 / 640,
    17 / 14336,
    -31 / 18432,
    691 / 180224,
    -5461 / 425984,
    929569 / 15728640,
    -3202291 / 8912896,
    221930581 / 79691776,
)
# digamma(a + 1/2) - digamma(a) = 1/(2a) + sum_k _F_COEFFS[k] / a**(2k + 2),
# the derivative of the series above: -(2k + 1) * _E_COEFFS[k].
_F_COEFFS: tuple[float, ...] = tuple(-(2 * k + 1) * c for k, c in enumerate(_E_COEFFS))

# The series is asymptotic: its best accuracy at ``a`` is about ``exp(-2 pi a)``.
# ``E`` is moved up to ``a + shift`` by the exact recurrence below, and the series
# is read there, with just enough terms for the dtype.  Absolute error of ``E``
# at the shifted argument: float32 6e-9 (6 terms, a >= 3), float64 3e-18 (10
# terms, a >= 8).
_SHIFT_32, _TERMS_32 = 3, 6
_SHIFT_64, _TERMS_64 = 8, 10

# ``u = z^2 / nu`` below this takes the series form of ``T``; at and above it the
# direct form, where ``log1p(u) - u / (1 + u)`` keeps most of its digits.
_U_SERIES = 1.0
# atanh(s) / s = sum_k s^(2k) / (2k + 1), at s = u / (2 + u) <= 1/3.
_ATANH_COEFFS: tuple[float, ...] = tuple(1.0 / (2 * k + 1) for k in range(17))
_ATANH_TERMS_32, _ATANH_TERMS_64 = 8, 17


def _is_double(x: Tensor) -> bool:
    return x.dtype == lucid.float64


def _gamma_half_excess_impl(a: Tensor) -> Tensor:
    shift, terms = (_SHIFT_64, _TERMS_64) if _is_double(a) else (_SHIFT_32, _TERMS_32)
    # E(a) = E(a + 1) - 1/2 log1p(1 / (4 a (a + 1))): the step is exact
    # ((a + 1/2)^2 = a (a + 1) + 1/4) and every term is negative, like E.  The
    # product is two reciprocals so that a huge ``a`` underflows it to 0 rather
    # than overflowing ``a (a + 1)``.
    steps = lucid.zeros_like(a)
    for k in range(shift):
        x = a + float(k)
        steps = steps + lucid.log1p(0.25 * (1.0 / x) * (1.0 / (x + 1.0)))
    r = 1.0 / (a + float(shift))
    return r * _horner_float(_E_COEFFS[:terms], r * r) - 0.5 * steps


def _gamma_half_excess(a: Tensor) -> Tensor:
    r"""``lgamma(a + 1/2) - lgamma(a) - log(a) / 2``, with no ``lgamma``.

    :math:`E(a) \to -1/(8a)` as :math:`a \to \infty`, and :math:`E(1/2) =
    -0.2258`.  From ``a = 3`` (float32) or ``8`` (float64) an asymptotic
    series in ``1/a`` gives it; below, the exact recurrence
    ``E(a) = E(a + 1) - log1p(1 / (4 a (a + 1))) / 2`` raises ``a`` to that
    bound first.  Every term of the recurrence has the sign of ``E`` and so
    does every term of its derivative: a gradient through this reads no
    difference of ``digamma``.  An argument of 0 or below is out of its domain
    and the result is not meaningful (the density masks it).
    """
    return _in_float32(_gamma_half_excess_impl, a)


def _digamma_half_gap_impl(a: Tensor) -> Tensor:
    shift, terms = (_SHIFT_64, _TERMS_64) if _is_double(a) else (_SHIFT_32, _TERMS_32)
    # F(a) = F(a + 1) + 1 / (2 a (a + 1/2)); same shape as the excess above.
    steps = lucid.zeros_like(a)
    for k in range(shift):
        x = a + float(k)
        steps = steps + 0.5 * (1.0 / x) * (1.0 / (x + 0.5))
    r = 1.0 / (a + float(shift))
    return steps + 0.5 * r + r * r * _horner_float(_F_COEFFS[:terms], r * r)


def _digamma_half_gap(a: Tensor) -> Tensor:
    r"""``digamma(a + 1/2) - digamma(a)``, with no ``digamma``.

    :math:`\to 1/(2a)` for large :math:`a`; the plain difference of two
    digammas is ``0`` once ``a`` passes about ``1e6`` in float32.
    """
    return _in_float32(_digamma_half_gap_impl, a)


def _student_exponent_impl(df: Tensor, z: Tensor) -> Tensor:
    double = _is_double(z)
    terms = _ATANH_TERMS_64 if double else _ATANH_TERMS_32
    q = z * z
    u = q / df
    small = u < _U_SERIES
    zero = lucid.zeros_like(u)
    # Series form: T = q (1 + 1/nu) / 2 * psi(u), psi(u) = log1p(u) / u
    #            = 2 / (2 + u) * atanh(s) / s,  s = u / (2 + u).
    # Differentiating T in nu now reads ``q / nu^2`` times terms of one sign,
    # where (nu + 1)/2 * log1p(q / nu) subtracts two ``q / (2 nu)``.
    u_s = lucid.where(small, u, zero)
    s = u_s / (2.0 + u_s)
    psi = 2.0 / (2.0 + u_s) * _horner_float(_ATANH_COEFFS[:terms], s * s)
    q_s = lucid.where(small, q, zero)
    near = 0.5 * q_s * (1.0 + 1.0 / df) * psi
    # Direct form, on ``u`` held at the switch where the series is the answer
    # (its slope in ``u`` is infinite at ``nu = inf``, and 0 * inf is a NaN).
    u_d = lucid.where(small, lucid.full_like(u, _U_SERIES), u)
    far = 0.5 * (df + 1.0) * lucid.log1p(u_d)
    return lucid.where(small, near, far)


def _student_exponent(df: Tensor, z: Tensor) -> Tensor:
    r"""``(df + 1) / 2 * log(1 + z^2 / df)``, differentiable in ``df`` without cancelling.

    Its value is the textbook one.  The point is the slope in ``df``:
    :math:`\tfrac12\log(1+u) - \tfrac{(\nu+1)u}{2\nu(1+u)}` with
    :math:`u = z^2/\nu` is a difference of two :math:`u/2`, so at
    :math:`\nu = 10^6` float32 left 12% of it and at :math:`10^8` only noise.
    For :math:`u < 1/2` the function is evaluated as
    :math:`\tfrac{z^2}{2}(1 + 1/\nu)\,\log(1+u)/u` through the series of
    :math:`\operatorname{atanh}`, whose every derivative adds terms of one
    sign.  :math:`z = \pm\infty` is :math:`+\infty`, as before.
    """
    return _in_float32(_student_exponent_impl, df, z)
