"""The Hurwitz zeta function, ``lucid.special.zeta``.

Evaluated in float32 for a 16-bit argument (:func:`_in_float32`): the
Euler-Maclaurin tail raises ``q + 12`` to powers outside float16's range,
and ``zeta(2, 1)`` answered NaN there.
"""

import math

import lucid
from lucid._ops.composite._opmath import _in_float32, _real_pair
from lucid._tensor.tensor import Tensor
from lucid._types import TensorOrScalar


def zeta(input: TensorOrScalar, other: TensorOrScalar) -> Tensor:
    r"""Hurwitz zeta function :math:`\zeta(x, q)`.

    Computes the Hurwitz zeta function, a two-argument generalisation
    of the Riemann zeta that arises in moment-generating functions of
    discrete distributions, in expansions of polygamma functions, and
    as a regulariser in analytic number theory.

    Parameters
    ----------
    input : Tensor | Scalar
        Exponent; must satisfy :math:`\Re(x) > 1` for the series to
        converge.  Any real dtype.
    other : Tensor | Scalar
        Shift parameter; broadcast-compatible with ``input``.  The
        non-positive integers are poles.  Either operand may be a Python
        number, which follows the other's device and dtype; not both.

    Returns
    -------
    Tensor
        :math:`\zeta(x, q)` element-wise, broadcast to the common shape
        of ``input`` and ``other``.

    Notes
    -----
    Series definition:

    .. math::

        \zeta(x, q) = \sum_{k=0}^\infty \frac{1}{(k + q)^x}.

    The implementation accumulates an explicit prefix of 12 terms and
    closes the tail with the Euler–Maclaurin correction

    .. math::

        \sum_{k=N}^\infty (k + q)^{-x}
            \approx \frac{a^{1-x}}{x - 1}
                  + \frac{a^{-x}}{2}
                  + \sum_j \frac{B_{2j}}{(2j)!}
                    \prod_{i=0}^{2j-2}(x + i)\, a^{-x-2j+1},

    where :math:`a = q + N` and the :math:`B_{2j}` are Bernoulli
    numbers; that is accurate to float64 rounding for :math:`x > 1`
    and :math:`q > 0`.  A negative non-integer ``other`` is real only for
    an integer ``input`` and is reduced to positive arguments; otherwise it
    is NaN.  The non-positive integers ``other`` are poles (``+inf``),
    ``input = 1`` is ``+inf`` and ``input < 1`` is NaN.  The Riemann zeta is the
    special case :math:`\zeta(x) = \zeta(x, 1)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import zeta
    >>> x = lucid.tensor([2.0, 3.0])
    >>> q = lucid.tensor([1.0, 1.0])
    >>> zeta(x, q)
    tensor([1.645, 1.202])
    >>> zeta(x, 1.0)
    tensor([1.645, 1.202])
    """
    return _in_float32(_zeta, *_real_pair("zeta", input, other))


def _zeta(x: Tensor, q: Tensor) -> Tensor:
    """:func:`zeta` on floating operands of one precision."""
    # Broadcast up front: every guard below is a per-element mask.
    x, q = x + lucid.zeros_like(q), q + lucid.zeros_like(x)
    nan = lucid.full_like(x, math.nan)
    inf = lucid.full_like(x, math.inf)
    one = lucid.ones_like(x)

    # Each series evaluation below is handed an argument inside its domain,
    # so the losing side of a ``where`` never meets 0/0 and no NaN reaches
    # the gradient through the mask.
    x_safe = lucid.where(x <= 1.0, lucid.full_like(x, 2.0), x)
    pole = (q <= 0.0) & (q == lucid.floor(q))
    negative = (q < 0.0) & ~pole
    f = q - lucid.floor(q)
    # zeta(x, q) itself, or for a negative q the zeta(x, f) term below.
    q_safe = lucid.where(negative, f, lucid.where(pole, one, q))

    # A negative non-integer q has terms (q + k)^-x of negative bases, real
    # only for an integer x.  Writing q = f - m with f in (0, 1), the m
    # negative terms are (-1)^x sum_{j=1..m} (j - f)^-x, which is a
    # difference of two Hurwitz zetas at positive arguments:
    #
    #     zeta(x, q) = zeta(x, f) + (-1)^x [zeta(x, 1 - f) - zeta(x, 1 - q)].
    #
    # The direct sum it replaces needed one term per unit of |q| before the
    # Euler-Maclaurin tail was valid; with a fixed prefix it was wrong for
    # q below about -3 and NaN once q + 12 went negative.
    #
    # The three evaluations go through one stacked series instead of three
    # calls: on Metal the traced graph of three calls reconverging on
    # ``x_safe`` compiled in 7.6 s, the stacked one in under a second, and
    # the values are element-for-element the same.
    out, mirror, far = lucid.unbind(
        _hurwitz_zeta_positive(
            lucid.stack([x_safe, x_safe, x_safe]),
            lucid.stack(
                [q_safe, lucid.where(negative, 1.0 - f, one), lucid.where(negative, 1.0 - q, one)]
            ),
        )
    )
    xd = x_safe.detach()
    parity = 1.0 - 2.0 * (xd - 2.0 * lucid.floor(xd / 2.0))
    out = lucid.where(negative, out + parity * (mirror - far), out)

    # x = +inf leaves only the terms with |q + k| <= 1: the series is +inf
    # if one is below 1 and 1 if q = 1.  The Euler-Maclaurin form evaluates
    # both as inf * 0.  With every term above 1 the limit is 0, but the
    # reference answers NaN there (its tail is inf * 0 too) and so does this.
    at_inf = lucid.where(q < 1.0, inf, lucid.where(q == 1.0, one, nan))
    out = lucid.where(x == math.inf, at_inf, out)

    # The remaining guards follow the reference's order of precedence,
    # lowest first.  A negative q with a non-integer x has no real value.
    out = lucid.where(negative & (x != lucid.floor(x)), nan, out)
    # The non-positive integers make one term 1/0.
    out = lucid.where(pole, inf, out)
    # The Hurwitz zeta converges only for x > 1.  At or below it the
    # series diverges, and the Euler-Maclaurin form evaluated anyway
    # returns an ordinary-looking number: -7.15e-06 at x = -1 on the CPU
    # and exactly 0 on Metal, neither of which is a value zeta takes
    # there.  The reference returns NaN across the whole divergent half
    # and so does this.
    out = lucid.where(x < 1.0, nan, out)
    # x = 1 is the harmonic series, which diverges *upward* rather than
    # merely failing to converge — so it is +inf, not NaN.
    return lucid.where(x == 1.0, inf, out)


# B_2, B_4, ..., B_10 over (2j)!, for the Euler-Maclaurin tail of ``zeta``.
_ZETA_TAIL_COEFFS: tuple[float, ...] = (
    (1.0 / 6.0) / 2.0,
    (-1.0 / 30.0) / 24.0,
    (1.0 / 42.0) / 720.0,
    (-1.0 / 30.0) / 40320.0,
    (5.0 / 66.0) / 3628800.0,
)


def _hurwitz_zeta_positive(x: Tensor, q: Tensor) -> Tensor:
    r""":math:`\zeta(x, q)` for ``x > 1`` and ``q > 0``.

    An explicit prefix of N = 12 terms, then the Euler-Maclaurin tail at
    a = q + N:

        sum_{k>=N} (k + q)^-x ~ a^(1-x)/(x-1) + a^-x/2
                                + sum_j B_2j/(2j)! x(x+1)...(x+2j-2) a^(-x-2j+1).

    The rising product has 2j - 1 factors — the (2j-1)-th derivative of
    t^-x.  It used to start at x(x+1) and so carried one factor too many in
    every term, which put zeta(2) at 1.6450854 against pi^2/6 = 1.6449341.
    """
    n_prefix = 12
    neg_x = -x
    s = q**neg_x
    for k in range(1, n_prefix):
        s = s + (q + float(k)) ** neg_x
    a = q + float(n_prefix)
    a_neg_x = a**neg_x
    inv_a = 1.0 / a
    inv_a2 = inv_a * inv_a
    head = a ** (1.0 - x) / (x - 1.0)
    half = 0.5 * a_neg_x
    a_pow = a_neg_x * inv_a

    rising = x
    tail = _ZETA_TAIL_COEFFS[0] * rising * a_pow
    for j, c in enumerate(_ZETA_TAIL_COEFFS[1:], start=1):
        rising = rising * (x + float(2 * j - 1)) * (x + float(2 * j))
        a_pow = a_pow * inv_a2
        tail = tail + c * rising * a_pow
    return s + head + half + tail


# Documented and listed under ``lucid.special``, the one path it is reached
# by (H8), like the functions defined there.
zeta.__module__ = "lucid.special"
