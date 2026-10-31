"""``lucid.special`` — special mathematical functions.

This sub-package adds functions that complement the elementary math
already available at the top level (``erf`` / ``sinc`` / ``lgamma`` /
``digamma`` / ``i0`` / ``log1p`` / ``logit`` / ``xlogy`` / ``erfc`` /
``erfinv`` / ``expm1`` …).  Per **H8** there is exactly one canonical
path per op — those stay accessible only at the top level, and the
extra entries below stay accessible only via ``lucid.special.<name>``.

Implemented (all pure-Python composites over engine ops — no C++ work):

* ``erfcx``                 — scaled complementary error: ``exp(x²) · erfc(x)``
* ``i0e``                   — exponentially-scaled Bessel I₀: ``exp(-|x|) · i0(x)``
* ``i1``                    — modified Bessel of order 1
* ``i1e``                   — exponentially-scaled ``i1``
* ``ndtr``                  — normal CDF ``Φ(x) = ½·(1 + erf(x/√2))``
* ``ndtri``                 — inverse normal CDF (Beasley-Springer-Moro)
* ``log_ndtr``              — numerically-stable ``log(Φ(x))``
* ``xlog1py``               — ``x · log1p(y)``, taken as 0 wherever ``x = 0``
* ``entr``                  — entropy element ``-x · log(x)``, 0 at x=0
* ``multigammaln``          — log of the multivariate gamma function
* ``polygamma``             — derivatives of ``digamma``, any order ``n``, on the
                              whole real line (reflection for ``x < 0``)
* ``spherical_bessel_j0``   — ``sin(x)/x`` with continuous extension at 0

Also: the order-0/1 Bessel family (``bessel_j0``/``j1``/``y0``/``y1``,
``modified_bessel_k0``/``k1`` and their scaled forms — float32 by
Abramowitz & Stegun polynomials, float64 by the Cephes double-precision
series), the Hurwitz ``zeta``, the regularized incomplete gamma
``gammainc`` / ``gammaincc`` and the orthogonal polynomials.  Deferred
(would need arbitrary-order Bessel machinery): ``iv``, ``ive``, Jacobi.
"""

import math
from typing import TYPE_CHECKING, Callable

import lucid
from lucid._ops.composite import _bessel
from lucid._ops.composite._bessel import _half_line_split
from lucid._ops.composite._gamma import _integer_order, _polygamma, gammainc, gammaincc
from lucid._ops.composite._opmath import _in_float32, _real
from lucid._ops.composite._zeta import zeta
from lucid._ops.composite.elementwise import _x_times_log
from lucid._types import TensorOrScalar
from lucid.special._entr import _Entr
from lucid.special._expit import _complex_expit
from lucid.special._erfcx import _Erfcx, _erfcx_stable
from lucid.special._ndtri import _Ndtri
from lucid.special._polynomials import (
    chebyshev_polynomial_t,
    chebyshev_polynomial_u,
    chebyshev_polynomial_v,
    chebyshev_polynomial_w,
    shifted_chebyshev_polynomial_t,
    shifted_chebyshev_polynomial_u,
    shifted_chebyshev_polynomial_v,
    shifted_chebyshev_polynomial_w,
    hermite_polynomial_h,
    hermite_polynomial_he,
    legendre_polynomial_p,
    laguerre_polynomial_l,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Error-function family ──────────────────────────────────────────────────


def erfcx(input: Tensor) -> Tensor:
    r"""Scaled complementary error function.

    Computes :math:`\mathrm{erfcx}(x) = e^{x^2}\, \mathrm{erfc}(x)`, the
    Mills-ratio-friendly scaling of the complementary error function.
    Whereas :math:`\mathrm{erfc}(x)` underflows to zero for moderately
    large :math:`x`, ``erfcx`` decays only as :math:`1/(x\sqrt\pi)` and
    remains finite, which is essential for tail-probability evaluations
    of Gaussian-related distributions.

    Parameters
    ----------
    input : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`e^{x^2}\,\mathrm{erfc}(x)` element-wise, same shape and
        dtype as ``input``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \mathrm{erfcx}(x) = e^{x^2}\,\mathrm{erfc}(x)
                         = e^{x^2} \cdot \frac{2}{\sqrt{\pi}}
                           \int_x^\infty e^{-t^2}\, dt.

    The implementation forms the product of ``exp(x*x)`` and ``erfc(x)``
    directly, which is accurate for :math:`|x| \lesssim 5`.  For very
    large positive ``x`` the two factors hit opposite floating-point
    extremes and accuracy degrades — use a dedicated continued-fraction
    expansion if extreme-value precision is required.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import erfcx
    >>> erfcx(lucid.tensor([0.0, 1.0, 5.0]))
    tensor([1., 0.4276, 0.1107])
    """
    out = _Erfcx.apply(input)
    assert isinstance(out, lucid.Tensor)
    return out


# Every scaled special function below had the same defect: it was written
# as its own definition — the unscaled function times an exponential —
# which is the one arrangement the scaling exists to avoid.
#
#     erfcx(100)  =  exp(10000) * erfc(100)  =  inf * 0    =  nan
#     i0e(1e6)    =  exp(-1e6)  * i0(1e6)    =  0   * inf  =  nan
#     k0e(1e6)    =  k0(1e6)    * exp(1e6)   =  0   * inf  =  nan
#
# Each answers a perfectly ordinary number — 0.00564, 0.000399, 0.00125 —
# and each is *bounded*, which is the entire reason the scaled variant is
# the one you reach for in a log-density.  The fix in every case is to
# push the exponential inside, where it cancels against the branch's own
# ``exp`` instead of meeting it as inf times zero.


def i0e(input: Tensor) -> Tensor:
    r"""Exponentially scaled modified Bessel function of order 0.

    Computes :math:`I_0(x)\, e^{-|x|}`, the standard exponentially-scaled
    variant of the modified Bessel function of the first kind.  ``I_0``
    itself grows as :math:`e^{|x|}/\sqrt{|x|}` and therefore overflows
    floating-point representation for moderately large arguments; the
    scaled form is bounded by 1 for :math:`x \ge 0` and is the natural
    quantity to manipulate in log-density computations (von Mises /
    Bessel distributions, Rice distribution, etc.).

    Parameters
    ----------
    input : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`I_0(x)\, e^{-|x|}` element-wise, same shape and dtype as
        ``input``.

    Notes
    -----
    Mathematical definition:

    .. math::

        I_0(x) e^{-|x|}, \quad
        I_0(x) = \sum_{k=0}^\infty \frac{(x/2)^{2k}}{(k!)^2}.

    The function is even and equals 1 at ``x = 0``.  For
    :math:`x \to \infty`,
    :math:`I_0(x) e^{-x} \sim 1/\sqrt{2\pi x}`.  Related identities:
    ``i0e(0) = 1`` and ``i0e(x) <= 1`` for all real ``x``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import i0e
    >>> i0e(lucid.tensor([0.0, 1.0, 5.0, 20.0]))
    tensor([1., 0.4658, 0.1835, 0.08978])
    """
    x = _real(input)
    if x.dtype == lucid.float64:
        return _bessel._i0e(x)
    return _in_float32(lambda v: _bessel._vanish_at_inf(v, _i0e_as_fit), x)


def _i0e_as_fit(x: Tensor) -> Tensor:
    ax = lucid.abs(x)
    xs = lucid.clamp(ax, max=3.75)
    small = lucid.exp(-xs) * lucid.i0(lucid.clip(x, -3.75, 3.75))
    big = lucid.clamp(ax, min=3.75)
    y = 3.75 / big
    poly = lucid.full_like(x, _bessel._AS_I0_LARGE[0])
    y_pow = lucid.ones_like(x)
    for c in _bessel._AS_I0_LARGE[1:]:
        y_pow = y_pow * y
        poly = poly + c * y_pow
    return lucid.where(ax <= 3.75, small, poly / lucid.sqrt(big))


# Abramowitz & Stegun §9.8 — same polynomial form as ``lucid.i0`` but for I₁.
_I1_SMALL_COEFFS: list[float] = [
    0.5,
    0.87890594,
    0.51498869,
    0.15084934,
    0.02658733,
    0.00301532,
    0.00032411,
]
_I1_LARGE_COEFFS: list[float] = [
    0.39894228,
    -0.03988024,
    -0.00362018,
    0.00163801,
    -0.01031555,
    0.02282967,
    -0.02895312,
    0.01787654,
    -0.00420059,
]


def i1(input: Tensor) -> Tensor:
    r"""Modified Bessel function of the first kind, order 1.

    Computes :math:`I_1(x)`, the order-one solution of the modified
    Bessel equation :math:`x^2 y'' + x y' - (x^2 + 1) y = 0` that is
    regular at the origin.  Used in Rician statistics, Bessel /
    von Mises – Fisher distributions, and rotational-symmetric kernels.

    Parameters
    ----------
    input : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`I_1(x)` element-wise, same shape and dtype as ``input``.

    Notes
    -----
    Series definition:

    .. math::

        I_1(x) = \sum_{k=0}^\infty
                 \frac{1}{k!\,(k+1)!}\left(\frac{x}{2}\right)^{2k+1}.

    The implementation uses the two-branch Abramowitz & Stegun §9.8
    polynomial approximation: a power series in :math:`(x/3.75)^2` for
    :math:`|x| \le 3.75`, and an asymptotic expansion in :math:`3.75/|x|`
    scaled by :math:`e^{|x|}/\sqrt{|x|}` for :math:`|x| > 3.75`.  Accuracy
    is roughly :math:`10^{-7}` over the real line; a ``float64`` input takes
    the Cephes double-precision series instead.  ``I_1`` is odd,
    ``I_1(0) = 0``, and :math:`I_1(x) \sim e^{x}/\sqrt{2\pi x}` as
    :math:`x \to \infty`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import i1
    >>> i1(lucid.tensor([0.0, 1.0, 5.0]))
    tensor([0., 0.5652, 24.34])
    """
    x = _real(input)
    if x.dtype == lucid.float64:
        return _bessel._i1(x)
    return _in_float32(_i1_as_fit, x)


def _i1_as_fit(x: Tensor) -> Tensor:
    ax = lucid.abs(x)
    ax_safe = lucid.clamp(ax, min=3.75)
    xs = lucid.clip(x, -3.75, 3.75)
    t_small = (xs * (1.0 / 3.75)) ** 2
    val_small = lucid.full_like(x, _I1_SMALL_COEFFS[0])
    t_pow = lucid.ones_like(x)
    for c in _I1_SMALL_COEFFS[1:]:
        t_pow = t_pow * t_small
        val_small = val_small + c * t_pow
    val_small = val_small * xs
    y_large = 3.75 / ax_safe
    val_large = lucid.full_like(x, _I1_LARGE_COEFFS[0])
    y_pow = lucid.ones_like(x)
    for c in _I1_LARGE_COEFFS[1:]:
        y_pow = y_pow * y_large
        val_large = val_large + c * y_pow
    val_large = val_large * lucid.exp(ax_safe) / lucid.sqrt(ax_safe)
    return lucid.where(ax <= 3.75, val_small, val_large * lucid.sign(x))


def i1e(input: Tensor) -> Tensor:
    r"""Exponentially scaled modified Bessel function of order 1.

    Computes :math:`I_1(x)\, e^{-|x|}`, the bounded counterpart of
    :func:`i1`.  Because :math:`I_1` itself grows exponentially with
    :math:`|x|`, this scaled form is the safer building block for
    log-likelihood expressions (Rice / Skellam distributions and
    similar) where evaluating ``I_1`` directly would overflow.

    Parameters
    ----------
    input : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`I_1(x)\, e^{-|x|}` element-wise, same shape and dtype as
        ``input``.

    Notes
    -----
    Definition:

    .. math::

        I_1(x)\, e^{-|x|}.

    The function is odd in ``x``, equals 0 at ``x = 0``, and approaches
    :math:`1/\sqrt{2\pi x}` as :math:`x \to \infty`.  Together with
    :func:`i0e` this forms the standard scaled-Bessel pair for stable
    evaluation of rotational-symmetric likelihoods.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import i1e
    >>> i1e(lucid.tensor([0.0, 1.0, 5.0]))
    tensor([0., 0.2079, 0.164])
    """
    x = _real(input)
    if x.dtype == lucid.float64:
        return _bessel._i1e(x)
    return _in_float32(lambda v: _bessel._vanish_at_inf(v, _i1e_as_fit), x)


def _i1e_as_fit(x: Tensor) -> Tensor:
    ax = lucid.abs(x)
    small = lucid.exp(-lucid.clamp(ax, max=3.75)) * _i1_as_fit(lucid.clip(x, -3.75, 3.75))
    big = lucid.clamp(ax, min=3.75)
    y = 3.75 / big
    poly = lucid.full_like(x, _I1_LARGE_COEFFS[0])
    y_pow = lucid.ones_like(x)
    for c in _I1_LARGE_COEFFS[1:]:
        y_pow = y_pow * y
        poly = poly + c * y_pow
    return lucid.where(ax <= 3.75, small, poly / lucid.sqrt(big) * lucid.sign(x))


# ── Normal-distribution helpers ────────────────────────────────────────────

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def ndtr(input: Tensor) -> Tensor:
    r"""Standard normal cumulative distribution function.

    Evaluates :math:`\Phi(x) = P(Z \le x)` where :math:`Z` is a standard
    normal random variable.  Implemented as a thin wrapper around the
    error function via the textbook identity
    :math:`\Phi(x) = \tfrac{1}{2}\,(1 + \mathrm{erf}(x/\sqrt{2}))`.

    Parameters
    ----------
    input : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`\Phi(x)` element-wise, same shape and dtype as ``input``;
        values lie in ``(0, 1)``.

    Notes
    -----
    Definition:

    .. math::

        \Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-t^2/2}\, dt
                = \frac{1}{2}\left(1 + \mathrm{erf}\!\left(
                  \frac{x}{\sqrt{2}}\right)\right).

    For ``x`` deep in the left tail, ``ndtr`` underflows to ``0`` and
    losses precision; prefer :func:`log_ndtr` for log-domain work.
    Special values: :math:`\Phi(-\infty) = 0`, :math:`\Phi(0) = 0.5`,
    :math:`\Phi(+\infty) = 1`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import ndtr
    >>> ndtr(lucid.tensor([-3.0, -1.0, 0.0, 1.0, 3.0]))
    tensor([0.00135, 0.1587, 0.5, 0.8413, 0.9987])
    """
    return 0.5 * (1.0 + lucid.erf(input * _INV_SQRT2))


# Where ``log_ndtr`` hands the upper tail to ``log1p`` of the complement.
# It is the point at which ``_erfcx_stable`` switches to its continued
# fraction: below it the complement would come from ``erfc`` again, which
# holds only about 1e-7 in float64, where the direct ``log(ndtr(x))`` is
# still good to 1e-14.
_LOG_NDTR_UPPER = 2.0 * math.sqrt(2.0)


def log_ndtr(input: Tensor) -> Tensor:
    r"""Numerically stable logarithm of the standard normal CDF.

    Computes :math:`\log \Phi(x)` without losing precision in the deep
    left tail, where the direct composition ``log(ndtr(x))`` underflows.
    Essential for log-likelihood evaluation of probit / Tobit / censored
    models that need to handle :math:`x \ll 0` cleanly.

    Parameters
    ----------
    input : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`\log \Phi(x)` element-wise, same shape and dtype as
        ``input``; values lie in :math:`(-\infty, 0]`.

    Notes
    -----
    The implementation has three branches; with :math:`t = -x/\sqrt{2}`
    in the lower tail and :math:`u = x/\sqrt{2}` in the upper one,

    .. math::

        \log \Phi(x) = \begin{cases}
            \log\!\left(\tfrac{1}{2}\,\mathrm{erfcx}(t)\right) - t^2,
                & x < -1 \\[2pt]
            \log \Phi(x), & -1 \le x \le 2\sqrt{2} \\[2pt]
            \operatorname{log1p}\!\left(
                -\tfrac{1}{2}\,\mathrm{erfcx}(u)\, e^{-u^2}\right),
                & x > 2\sqrt{2}.
        \end{cases}

    Between the cuts :math:`0.16 \lesssim \Phi(x) \lesssim 0.998`, so the
    direct ``log(ndtr(x))`` is safe.  Below :math:`x = -1` the identity
    :math:`\Phi(x) = \tfrac{1}{2}\,\mathrm{erfc}(t)
    = \tfrac{1}{2}\,\mathrm{erfcx}(t)\, e^{-t^2}` is used: ``erfc(t)``
    itself underflows to zero near :math:`x = -13` in float32 (and loses
    its relative precision well before that), while ``erfcx`` stays
    bounded and the Gaussian factor comes out exactly as :math:`-t^2`.

    The upper tail fails the other way round.  There
    :math:`\log \Phi(x) \approx -Q` with :math:`Q = 1 - \Phi(x)` small, and
    forming :math:`\Phi(x)` first leaves :math:`Q` to rounding: float32
    answered ``-2.98e-7`` at :math:`x = 5` for ``-2.867e-7``, and ``0``
    from :math:`x \approx 5.4` on.  Computing
    :math:`Q = \tfrac{1}{2}\,\mathrm{erfcx}(u)\, e^{-u^2}` directly and
    taking ``log1p`` of its negative keeps the relative precision until
    :math:`Q` itself underflows.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import log_ndtr
    >>> log_ndtr(lucid.tensor([-10.0, -1.0, 0.0, 1.0]))
    tensor([-53.23, -1.841, -0.6931, -0.1728])
    """
    x = _real(input)
    lo = lucid.full_like(x, -1.0)
    hi = lucid.full_like(x, _LOG_NDTR_UPPER)
    # Each branch sees only its own stretch of the line: ``where``
    # evaluates all three, and a -inf in a branch that loses is a NaN in
    # the gradient.
    direct = lucid.log(ndtr(lucid.clip(x, -1.0, _LOG_NDTR_UPPER)))
    t = -lucid.clamp(x, max=-1.0) * _INV_SQRT2
    tail = lucid.log(0.5 * _erfcx_stable(t)) - t * t
    # Capped as well: at +inf the exp(-x^2/2) factor meets its own slope as
    # 0 * inf, a NaN gradient where the reference has 0; 1e4 is already 0 in
    # either dtype and the clamp passes no gradient beyond it.
    xu = lucid.clamp(x, min=_LOG_NDTR_UPPER, max=1e4)
    # exp(-x²/2) is as ill-conditioned as its argument is large: a rounded
    # u = x/√2 squared carries ~4 times the error of x*x, ~4e-13 at x = 37.
    upper = lucid.log1p(-0.5 * _erfcx_stable(xu * _INV_SQRT2) * lucid.exp(-0.5 * xu * xu))
    return lucid.where(x < lo, tail, lucid.where(x > hi, upper, direct))


def ndtri(input: Tensor) -> Tensor:
    r"""Inverse standard normal CDF (probit / quantile function).

    Computes :math:`\Phi^{-1}(p)`, the quantile function of the standard
    normal distribution.  This is the workhorse for sampling Gaussian
    variates from uniforms via inverse-CDF, for probit regression, and
    for converting tail probabilities back to standard scores.

    Parameters
    ----------
    input : Tensor
        Input tensor of probabilities in :math:`(0, 1)`; any
        floating-point dtype.

    Returns
    -------
    Tensor
        :math:`\Phi^{-1}(p)` element-wise, same shape and dtype as
        ``input``; values are unbounded reals.

    Notes
    -----
    The implementation uses the Beasley-Springer-Moro rational
    approximation.  The unit interval is split into a central region
    :math:`[p_{lo}, p_{hi}]` with :math:`p_{lo} = 0.02425`,
    :math:`p_{hi} = 1 - p_{lo}`, in which a rational polynomial in
    :math:`q = p - 0.5` is evaluated, and two symmetric tail regions in
    which a rational polynomial in
    :math:`r = \sqrt{-2 \log\min(p, 1-p)}` is used:

    .. math::

        \Phi^{-1}(p) \approx \begin{cases}
            q \cdot \frac{N_c(q^2)}{D_c(q^2)},
                & p \in [p_{lo}, p_{hi}] \\[4pt]
            \pm \frac{N_t(r)}{D_t(r)},
                & \text{otherwise}.
        \end{cases}

    The approximation is accurate to roughly :math:`1.15 \times 10^{-9}`
    over its supported domain, which is as much as ``float32`` holds.  A
    ``float64`` input gets one Newton step on top of it, taken where each
    stretch keeps its relative precision (on :math:`\Phi(x) - p` in the
    centre, on :math:`\log\Phi(x) - \log p` in the tails), and is then
    accurate to the last few digits across :math:`10^{-300}` to
    :math:`1 - 10^{-16}`.  Boundary behaviour:
    :math:`\Phi^{-1}(0.5) = 0`, and the function blows up to
    :math:`\mp \infty` at the endpoints.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import ndtri
    >>> ndtri(lucid.tensor([0.025, 0.5, 0.975]))
    tensor([-1.96, 0., 1.96])
    """
    out = _Ndtri.apply(_real(input))
    assert isinstance(out, lucid.Tensor)
    return out


# ── x-times-log-of-something helpers ───────────────────────────────────────


def xlog1py(input: TensorOrScalar, other: TensorOrScalar) -> Tensor:
    r"""Safe product :math:`x \log(1 + y)` with limit-convention zero handling.

    Computes :math:`x \log(1 + y)` element-wise but takes the product as
    ``0`` wherever :math:`x = 0`, including where :math:`\log(1 + y)` is
    infinite (``y = -1``).  Mirrors :func:`lucid.xlogy` but uses
    :math:`\log(1 + y)` instead of :math:`\log y`, which is the right
    primitive for log-densities of distributions expressed in terms of
    small offsets (Negative Binomial log-likelihood, Beta survival
    functions, etc.).

    Parameters
    ----------
    input : Tensor | Scalar
        Multiplier.  Either operand may be a Python number.
    other : Tensor | Scalar
        Argument of ``log1p``; broadcast-compatible with ``input``.

    Returns
    -------
    Tensor
        :math:`x \log(1 + y)` element-wise, broadcast to the common
        shape of ``input`` and ``other``.

    Notes
    -----
    Mathematical definition with the limit convention:

    .. math::

        \text{xlog1py}(x, y) = \begin{cases}
            \mathrm{NaN}, & y = \mathrm{NaN} \\[2pt]
            0, & x = 0 \\[2pt]
            x \log(1 + y), & \text{otherwise}.
        \end{cases}

    The convention covers only ``x == 0``: with ``x != 0`` the plain
    definition applies, so ``xlog1py(2, -1)`` is ``-inf``.  Using
    :math:`\log(1 + y)` avoids precision loss for small ``y`` where
    ``1 + y`` would round to ``1.0``.

    Gradients are :math:`\partial/\partial x = \log(1 + y)` and
    :math:`\partial/\partial y = x / (1 + y)`, except where ``x == 0`` and
    :math:`\log(1 + y)` is not finite (``y <= -1`` or ``y`` infinite): there
    both are ``0``, so the backward never forms ``0 \cdot \infty``.  This is
    the convention of :func:`lucid.xlogy`, and the two functions share one
    implementation.  Two cells differ from the reference framework on purpose:
    at ``(0, -1)`` its ``y`` gradient is ``0 / 0 = NaN`` and Lucid's is ``0``,
    and at ``(0, +\infty)`` its ``x`` gradient is :math:`+\infty` and Lucid's
    is ``0``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import xlog1py
    >>> x = lucid.tensor([0.0, 1.0, 2.0])
    >>> y = lucid.tensor([0.0, 1.0, 3.0])
    >>> xlog1py(x, y)
    tensor([0., 0.6931, 2.773])
    >>> xlog1py(lucid.tensor([0.0, 2.0]), lucid.tensor([-1.0, -1.0]))
    tensor([0., -inf])
    """
    return _x_times_log(input, other, lucid.log1p, pole=-1.0, root=0.0)


def entr(input: Tensor) -> Tensor:
    r"""Element-wise entropy kernel :math:`-x \log x`.

    Returns the per-element contribution to Shannon entropy.  This is
    the standard building block for evaluating :math:`H(p) =
    \sum_i \mathrm{entr}(p_i)` for a discrete distribution, with the
    limit convention :math:`0 \log 0 = 0` applied at the boundary.

    Parameters
    ----------
    input : Tensor
        Input tensor.  Integer and bool inputs are taken as ``float32``.
        Values below zero give ``-inf``.

    Returns
    -------
    Tensor
        Element-wise :math:`-x \log x` with the limit convention at
        ``x = 0``; same shape as ``input``.

    Notes
    -----
    Definition:

    .. math::

        \mathrm{entr}(x) = \begin{cases}
            -x \log x, & x > 0 \\[2pt]
            0,         & x = 0 \\[2pt]
            -\infty,   & x < 0.
        \end{cases}

    The function is concave with maximum :math:`1/e \approx 0.3679` at
    :math:`x = 1/e`.  It vanishes at both ``x = 0`` (limit) and
    ``x = 1``.  Related quantities include `rel_entr` (point-wise
    relative entropy) and :func:`kl_div` (KL kernel).

    The value is :math:`-\operatorname{xlogy}(x, x)` and takes the
    ``0 \log 0`` convention from :func:`lucid.xlogy`.  The gradient is
    :math:`-(1 + \log x)` as the reference framework has it: the one-sided
    :math:`+\infty` at ``x = 0`` and NaN below zero.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import entr
    >>> entr(lucid.tensor([0.0, 0.5, 1.0, 2.0]))
    tensor([0., 0.3466, 0., -1.386])
    """
    out = _Entr.apply(input)
    assert isinstance(out, lucid.Tensor)
    return out


# ── Gamma family ───────────────────────────────────────────────────────────


def multigammaln(input: Tensor, p: int) -> Tensor:
    r"""Logarithm of the multivariate gamma function :math:`\Gamma_p(a)`.

    Computes :math:`\log \Gamma_p(a)`, the natural generalisation of
    ``lgamma`` that appears in the normalising constants of the Wishart
    and inverse-Wishart distributions over positive-definite matrices,
    and in matrix-Bayes log-evidence formulas.

    Parameters
    ----------
    input : Tensor
        Real argument; any floating-point dtype.  Each entry must
        satisfy :math:`a > (p - 1)/2` for the function to be finite
        (otherwise the underlying ``lgamma`` returns ``inf``).
    p : int
        Dimensionality of the matrix argument; must be a positive
        integer.

    Returns
    -------
    Tensor
        :math:`\log \Gamma_p(a)` element-wise over ``input``, same shape
        and dtype as ``input``.

    Notes
    -----
    Definition:

    .. math::

        \log \Gamma_p(a)
            = \frac{p(p-1)}{4}\,\log \pi
              + \sum_{i=1}^{p} \log \Gamma\!\left(a + \frac{1 - i}{2}
              \right).

    For ``p = 1`` this reduces to :math:`\log \Gamma(a)`.  Raises
    ``ValueError`` if ``p < 1`` and ``TypeError`` if ``p`` is not an integer
    (a float or a ``bool``), as the reference does.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import multigammaln
    >>> multigammaln(lucid.tensor([3.0, 5.0]), p=2)
    tensor([1.55, 6.204])
    """
    p = _integer_order(p, "p")
    if p < 1:
        raise ValueError(f"multigammaln requires p >= 1, got {p}")
    accum = lucid.zeros_like(input)
    for i in range(1, p + 1):
        accum = accum + lucid.lgamma(input + (1.0 - float(i)) / 2.0)
    return float(p * (p - 1)) / 4.0 * math.log(math.pi) + accum


def polygamma(n: int, input: Tensor) -> Tensor:
    r"""Polygamma function :math:`\psi^{(n)}(x)`.

    The ``n``-th derivative of the digamma function.  Polygamma
    functions appear in the derivatives of log-partition functions of
    the Gamma / Beta / Dirichlet families and are needed for evaluating
    Fisher information matrices for these distributions analytically.

    Parameters
    ----------
    n : int
        Non-negative integer order.  ``n = 0`` recovers
        :func:`lucid.digamma`, ``n = 1`` is the trigamma function.  Any
        order is accepted, though :math:`\psi^{(n)}` grows like
        :math:`n!` and overflows float32 for large ``n`` — see Notes.
    input : Tensor
        Real argument; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`\psi^{(n)}(x)` element-wise, same shape and dtype as
        ``input``.

    Notes
    -----
    Definition (for ``n >= 1``):

    .. math::

        \psi^{(n)}(x) = \frac{d^{n+1}}{dx^{n+1}} \log \Gamma(x)
                     = (-1)^{n+1}\, n!\, \sum_{k=0}^\infty
                       \frac{1}{(x + k)^{n+1}}.

    Implementation: for ``x >= 0`` shift ``x`` upward by :math:`K = 8`
    using the recurrence

    .. math::

        \psi^{(n)}(x) = \psi^{(n)}(x + 1)
                      + \frac{(-1)^{n+1} n!}{x^{n+1}},

    accumulate the per-step corrections, then evaluate the Bernoulli
    asymptotic series at the shifted argument where it is
    well-conditioned.  For ``x < 0`` the ``n``-th derivative of the
    reflection formula,

    .. math::

        \psi^{(n)}(x) = (-1)^n \psi^{(n)}(1 - x)
                      - \pi \frac{d^n}{dx^n} \cot(\pi x),

    maps the argument onto the right half.  The non-positive integers
    (and ``-inf``) are poles: ``n = 0`` gives NaN there, except
    ``psi(+0) = -inf`` and ``psi(-0) = +inf``; ``n >= 1`` gives
    :math:`(-1)^{n+1} \infty`.  The gradient is :math:`\psi^{(n+1)}`
    everywhere, a pole's slope being the next order's pole value.

    Raises ``ValueError`` for ``n < 0`` and ``TypeError`` if ``n`` is not an
    integer (a float or a ``bool``), as the reference does.

    Accuracy was checked against an independent direct summation
    :math:`\psi^{(n)}(x) = (-1)^{n+1} n! \sum_{k \ge 0} (x+k)^{-(n+1)}`
    for :math:`n = 1 \dots 12` and :math:`x \in \{0.3, 1, 2.5, 7\}`: the
    worst relative disagreement was 7e-11, which is the reference's own
    truncation floor rather than this routine's error, and is far below
    what float32 can represent anyway.

    Note that :math:`\psi^{(n)}(1) = (-1)^{n+1} n!\,\zeta(n+1)`, so the
    result itself exceeds the float32 range somewhere around
    :math:`n \approx 34`; that is the dtype's limit, not the method's.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import polygamma
    >>> polygamma(1, lucid.tensor([1.0, 2.0, 5.0]))
    tensor([1.645, 0.6449, 0.2213])
    """
    n = _integer_order(n, "n")
    if n < 0:
        raise ValueError(f"polygamma: n must be ≥ 0, got {n}")
    return _polygamma(n, input)


# ── Spherical Bessel ───────────────────────────────────────────────────────


def spherical_bessel_j0(x: Tensor) -> Tensor:
    r"""Spherical Bessel function of the first kind, order 0.

    Computes :math:`j_0(x) = \sin x / x` with the continuous extension
    :math:`j_0(0) = 1`.  Arises in 3D wave / scattering problems and is
    the radial component of plane waves expanded in spherical harmonics.

    Parameters
    ----------
    x : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`j_0(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Mathematical definition:

    .. math::

        j_0(x) = \frac{\sin x}{x}, \qquad j_0(0) = 1, \qquad j_0(\pm\infty) = 0.

    The implementation guards the removable singularity at the origin
    by substituting ``x = 1`` into the division and patching the result
    with ``where(x == 0, 1, ...)``, so the function is well-defined and
    differentiable across the origin.  Related to the normalised sinc:
    :math:`j_0(x) = \mathrm{sinc}(x / \pi)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import spherical_bessel_j0
    >>> spherical_bessel_j0(lucid.tensor([0.0, 1.0, 2.0]))
    tensor([1., 0.8415, 0.4546])
    """
    return _bessel._spherical_j0(x)


# ── Bessel functions ───────────────────────────────────────────────────────
#
# float32: polynomial / asymptotic approximations from Abramowitz & Stegun
# §9 (J0, J1, Y0, Y1) and §9.8 (K0, K1), each split into a small-|x|
# polynomial branch and a large-|x| asymptotic-with-trig branch.  Their
# eight-digit coefficients are accurate to ≈ 1e-7 — float32's own rounding
# level, and no further.  float64 therefore takes the Cephes
# double-precision series in ``lucid._ops.composite._bessel``.  Y and K
# share ``_half_line_split``'s domain rule (±inf at 0, NaN below) in either dtype.


# A&S 9.8.6 — ``e^x √x K(x)`` for x > 2: ``_modified_bessel_k{0,1}_large``
# without the ``exp(-x)/√x`` tail.
def _K0E_LARGE(z: Tensor) -> Tensor:
    return (
        1.25331414
        - 0.07832358 * z
        + 0.02189568 * z**2
        - 0.01062446 * z**3
        + 0.00587872 * z**4
        - 0.00251540 * z**5
        + 0.00053208 * z**6
    )


def _K1E_LARGE(z: Tensor) -> Tensor:
    return (
        1.25331414
        + 0.23498619 * z
        - 0.03655620 * z**2
        + 0.01504268 * z**3
        - 0.00780353 * z**4
        + 0.00325614 * z**5
        - 0.00068245 * z**6
    )


def _split(
    x: Tensor,
    threshold: float,
    small_fn: Callable[[Tensor], Tensor],
    large_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    def split(v: Tensor) -> Tensor:
        small = small_fn(lucid.clip(v, -threshold, threshold))
        return lucid.where(lucid.abs(v) <= threshold, small, large_fn(v))

    return _in_float32(split, _real(x))


def _bessel_j0_small(x: Tensor) -> Tensor:
    # Abramowitz & Stegun 9.4.1 — polynomial fit on |x| <= 3.
    y = (x / 3.0) ** 2
    return (
        1.0
        - 2.2499997 * y
        + 1.2656208 * y**2
        - 0.3163866 * y**3
        + 0.0444479 * y**4
        - 0.0039444 * y**5
        + 0.00021 * y**6
    )


def _bessel_j0_large(x: Tensor) -> Tensor:
    # A&S 9.4.3 on |x| > 3; clamped, as ``_split`` runs it at 0 too (3/0: NaN grad).
    ax = lucid.clamp(lucid.abs(x), min=3.0)
    z = 3.0 / ax
    f0 = (
        0.79788456
        - 0.00000077 * z
        - 0.00552740 * z**2
        - 0.00009512 * z**3
        + 0.00137237 * z**4
        - 0.00072805 * z**5
        + 0.00014476 * z**6
    )
    theta = (
        ax
        - 0.78539816
        - 0.04166397 * z
        - 0.00003954 * z**2
        + 0.00262573 * z**3
        - 0.00054125 * z**4
        - 0.00029333 * z**5
        + 0.00013558 * z**6
    )
    return f0 * lucid.cos(theta) / lucid.sqrt(ax)


def bessel_j0(input: Tensor) -> Tensor:
    r"""Bessel function of the first kind, order 0.

    Computes :math:`J_0(x)`, the regular-at-origin solution of Bessel's
    equation :math:`x^2 y'' + x y' + x^2 y = 0` for order 0.  Appears
    throughout wave propagation, optical diffraction (the Airy disk
    profile), and vibration of circular membranes.

    Parameters
    ----------
    input : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`J_0(x)` element-wise, same shape and dtype as ``input``.

    Notes
    -----
    Series representation:

    .. math::

        J_0(x) = \sum_{k=0}^\infty \frac{(-1)^k}{(k!)^2}
                 \left(\frac{x}{2}\right)^{2k}.

    The implementation uses the Abramowitz & Stegun §9.4 two-branch
    polynomial fit: a power series in :math:`(x/3)^2` for :math:`|x| \le 3`
    and an asymptotic
    :math:`\sqrt{2/(\pi x)} \cos(x - \pi/4) \cdot f(3/|x|)` form for
    :math:`|x| > 3`.  Accuracy is :math:`\approx 10^{-7}`
    on the real line; a ``float64`` input takes the Cephes double-precision
    series instead, with its first two zeros exact.  ``J_0`` is even,
    :math:`J_0(0) = 1`, and the function decays as :math:`\sqrt{2/(\pi x)}`
    for large ``x``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import bessel_j0
    >>> bessel_j0(lucid.tensor([0.0, 1.0, 5.0]))
    tensor([1., 0.7652, -0.1776])
    >>> bool(bessel_j0(lucid.tensor(2.4048)).abs() < 1e-4)  # the first zero
    True
    """
    if input.dtype == lucid.float64:
        return _bessel._j0(input)
    return _split(input, 3.0, _bessel_j0_small, _bessel_j0_large)


def _bessel_j1_small(x: Tensor) -> Tensor:
    y = (x / 3.0) ** 2
    poly = (
        0.5
        - 0.56249985 * y
        + 0.21093573 * y**2
        - 0.03954289 * y**3
        + 0.00443319 * y**4
        - 0.00031761 * y**5
        + 0.00001109 * y**6
    )
    return x * poly


def _bessel_j1_large(x: Tensor) -> Tensor:
    ax = lucid.clamp(lucid.abs(x), min=3.0)  # see _bessel_j0_large
    z = 3.0 / ax
    f1 = (
        0.79788456
        + 0.00000156 * z
        + 0.01659667 * z**2
        + 0.00017105 * z**3
        - 0.00249511 * z**4
        + 0.00113653 * z**5
        - 0.00020033 * z**6
    )
    theta = (
        ax
        - 2.35619449
        + 0.12499612 * z
        + 0.00005650 * z**2
        - 0.00637879 * z**3
        + 0.00074348 * z**4
        + 0.00079824 * z**5
        - 0.00029166 * z**6
    )
    val = f1 * lucid.cos(theta) / lucid.sqrt(ax)
    # j1 is odd in x; flip sign for negative inputs.
    return lucid.where(x < 0, -val, val)


def bessel_j1(input: Tensor) -> Tensor:
    r"""Bessel function of the first kind, order 1.

    Computes :math:`J_1(x)`, the regular order-one solution of Bessel's
    equation.  Closely related to :math:`J_0` by
    :math:`J_0'(x) = -J_1(x)`; appears as the radial derivative of
    cylindrical waves and in the antenna-pattern of a uniformly
    illuminated circular aperture.

    Parameters
    ----------
    input : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`J_1(x)` element-wise, same shape and dtype as ``input``.

    Notes
    -----
    Series representation:

    .. math::

        J_1(x) = \sum_{k=0}^\infty \frac{(-1)^k}{k!\, (k+1)!}
                 \left(\frac{x}{2}\right)^{2k+1}.

    Implementation: Abramowitz & Stegun §9.4 two-branch polynomial, accurate
    to :math:`\approx 10^{-7}`; a ``float64`` input takes the Cephes
    double-precision series instead, with its first two zeros exact.
    ``J_1`` is odd, :math:`J_1(0) = 0`, with maximum :math:`\approx 0.5819`
    at :math:`x \approx 1.8412`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import bessel_j1
    >>> bessel_j1(lucid.tensor([0.0, 1.0, 5.0]))
    tensor([0., 0.4401, -0.3276])
    >>> bool(bessel_j1(lucid.tensor(3.8317)).abs() < 1e-4)  # the first zero
    True
    """
    if input.dtype == lucid.float64:
        return _bessel._j1(input)
    return _split(input, 3.0, _bessel_j1_small, _bessel_j1_large)


_TWO_OVER_PI = 2.0 / math.pi


# Y0 / Y1 use the rational approximations from Numerical Recipes (Press,
# Teukolsky, Vetterling, Flannery — 2nd ed., §6.5).  These are carefully
# fitted on (0, 8] with the J0 / J1·log(x) Wronskian term added on top,
# and on (8, ∞) via the standard A&S 9.4.5 trigonometric asymptotic.


def _bessel_y0_small(x: Tensor) -> Tensor:
    y = x * x
    num = -2957821389.0 + y * (
        7062834065.0 + y * (-512359803.6 + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733)))
    )
    den = 40076544269.0 + y * (
        745249964.8 + y * (7189466.438 + y * (47447.26470 + y * (226.1030244 + y * 1.0)))
    )
    return num / den + _TWO_OVER_PI * lucid.log(x) * bessel_j0(x)


def _bessel_y0_large(x: Tensor) -> Tensor:
    z = 8.0 / x
    y = z * z
    p = 1.0 + y * (
        -0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6))
    )
    q = -0.1562499995e-1 + y * (
        0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * (-0.934935152e-7)))
    )
    xx = x - 0.785398164
    return lucid.sqrt(0.636619772 / x) * (lucid.sin(xx) * p + z * lucid.cos(xx) * q)


def bessel_y0(input: Tensor) -> Tensor:
    r"""Bessel function of the second kind, order 0.

    Computes :math:`Y_0(x)`, the singular-at-origin partner of
    :math:`J_0`.  Together :math:`J_0` and :math:`Y_0` span the
    solution space of Bessel's equation at order 0; :math:`Y_0` is the
    one that diverges (logarithmically) at the origin.

    Parameters
    ----------
    input : Tensor
        Input tensor; only :math:`x > 0` is in the domain.  Any
        floating-point dtype.

    Returns
    -------
    Tensor
        :math:`Y_0(x)` element-wise, same shape and dtype as ``input``.

    Notes
    -----
    Asymptotic behaviour:

    .. math::

        Y_0(x) \sim \frac{2}{\pi}\left(\log\frac{x}{2}
                                       + \gamma\right) \;
                                 \text{as } x \to 0^+,
        \qquad
        Y_0(x) \sim \sqrt{\frac{2}{\pi x}}
                   \sin\!\left(x - \frac{\pi}{4}\right)
                                 \;\text{as } x \to \infty.

    Implementation: Numerical Recipes §6.5 rational fit on :math:`(0, 8]`
    with the :math:`J_0 \log` Wronskian term, and the standard A&S 9.4.5
    trigonometric asymptotic on :math:`(8, \infty)`. Accuracy is
    :math:`\approx 10^{-7}`; a ``float64`` input takes the Cephes
    double-precision series instead. ``Y_0`` diverges to :math:`-\infty` as
    :math:`x \to 0^+`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import bessel_y0
    >>> bessel_y0(lucid.tensor([1.0, 5.0, 10.0]))
    tensor([0.08826, -0.3085, 0.05567])
    """
    if input.dtype == lucid.float64:
        return _bessel._y0(input)
    return _half_line_split(_real(input), 8.0, _bessel_y0_small, _bessel_y0_large, -math.inf)


def _bessel_y1_small(x: Tensor) -> Tensor:
    y = x * x
    num = x * (
        -0.4900604943e13
        + y
        * (
            0.1275274390e13
            + y
            * (-0.5153438139e11 + y * (0.7349264551e9 + y * (-0.4237922726e7 + y * 0.8511937935e4)))
        )
    )
    den = 0.2499580570e14 + y * (
        0.4244419664e12
        + y
        * (
            0.3733650367e10
            + y * (0.2245904002e8 + y * (0.1020426050e6 + y * (0.3549632885e3 + y * 1.0)))
        )
    )
    return num / den + _TWO_OVER_PI * (lucid.log(x) * bessel_j1(x) - 1.0 / x)


def _bessel_y1_large(x: Tensor) -> Tensor:
    z = 8.0 / x
    y = z * z
    p = 1.0 + y * (
        0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6)))
    )
    q = 0.04687499995 + y * (
        -0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6))
    )
    xx = x - 2.356194491
    return lucid.sqrt(0.636619772 / x) * (lucid.sin(xx) * p + z * lucid.cos(xx) * q)


def bessel_y1(input: Tensor) -> Tensor:
    r"""Bessel function of the second kind, order 1.

    Computes :math:`Y_1(x)`, the singular-at-origin order-one solution
    of Bessel's equation.  Diverges as :math:`-2/(\pi x)` at the origin
    and decays as :math:`\sqrt{2/(\pi x)}` for large ``x``.

    Parameters
    ----------
    input : Tensor
        Input tensor; only :math:`x > 0` is in the domain.  Any
        floating-point dtype.

    Returns
    -------
    Tensor
        :math:`Y_1(x)` element-wise, same shape and dtype as ``input``.

    Notes
    -----
    Asymptotic forms:

    .. math::

        Y_1(x) \sim -\frac{2}{\pi x} \;\text{as } x \to 0^+,
        \qquad
        Y_1(x) \sim \sqrt{\frac{2}{\pi x}}
                    \sin\!\left(x - \frac{3\pi}{4}\right)
                    \;\text{as } x \to \infty.

    Implementation: Numerical Recipes §6.5 rational fit on :math:`(0, 8]`
    (Wronskian-corrected with the :math:`J_1 \log` term), trigonometric
    asymptotic on :math:`(8, \infty)`.  Accuracy is :math:`\approx 10^{-7}`;
    a ``float64`` input takes the Cephes double-precision series instead.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import bessel_y1
    >>> bessel_y1(lucid.tensor([1.0, 5.0, 10.0]))
    tensor([-0.7812, 0.1479, 0.249])
    """
    if input.dtype == lucid.float64:
        return _bessel._y1(input)
    return _half_line_split(_real(input), 8.0, _bessel_y1_small, _bessel_y1_large, -math.inf)


def _modified_bessel_k0_small(x: Tensor) -> Tensor:
    # A&S 9.8.5 — polynomial fit for 0 < x <= 2.
    y = (x / 2.0) ** 2
    poly = (
        -0.57721566
        + 0.42278420 * y
        + 0.23069756 * y**2
        + 0.03488590 * y**3
        + 0.00262698 * y**4
        + 0.00010750 * y**5
        + 0.00000740 * y**6
    )
    return -lucid.log(x / 2.0) * lucid.i0(x) + poly


def _modified_bessel_k0_large(x: Tensor) -> Tensor:
    # A&S 9.8.6 — asymptotic for x > 2.
    return _K0E_LARGE(2.0 / x) * lucid.exp(-x) / lucid.sqrt(x)


def modified_bessel_k0(input: Tensor) -> Tensor:
    r"""Modified Bessel function of the second kind, order 0.

    Computes :math:`K_0(x)`, the exponentially decaying solution of the
    modified Bessel equation at order 0.  Appears as the Green's
    function of the 2D Helmholtz operator and as the log-density kernel
    of certain heavy-tailed distributions.

    Parameters
    ----------
    input : Tensor
        Input tensor; only :math:`x > 0` is in the domain.  Any
        floating-point dtype.

    Returns
    -------
    Tensor
        :math:`K_0(x)` element-wise, same shape and dtype as ``input``.

    Notes
    -----
    Asymptotic forms:

    .. math::

        K_0(x) \sim -\log(x/2) - \gamma \;\text{as } x \to 0^+,
        \qquad
        K_0(x) \sim \sqrt{\frac{\pi}{2x}}\, e^{-x}
                  \;\text{as } x \to \infty.

    Implementation: Abramowitz & Stegun §9.8 polynomial branches at the
    cutoff :math:`x = 2`, accurate to :math:`\approx 10^{-7}`; a ``float64``
    input takes the Cephes double-precision series instead.  For numerically
    stable evaluation at large arguments use
    :func:`scaled_modified_bessel_k0`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import modified_bessel_k0
    >>> modified_bessel_k0(lucid.tensor([0.5, 1.0, 5.0]))
    tensor([0.9244, 0.421, 0.003691])
    """
    if input.dtype == lucid.float64:
        return _bessel._k0(input)
    return _half_line_split(
        _real(input), 2.0, _modified_bessel_k0_small, _modified_bessel_k0_large, math.inf
    )


def scaled_modified_bessel_k0(x: Tensor) -> Tensor:
    r"""Exponentially scaled modified Bessel :math:`K_0`: :math:`e^x K_0(x)`.

    Computes :math:`e^{x} K_0(x)`, the bounded counterpart of
    :func:`modified_bessel_k0`.  Because :math:`K_0` decays
    exponentially as :math:`e^{-x}/\sqrt{x}`, the scaled form is the
    natural quantity to manipulate in log-density expressions where
    direct evaluation of :math:`K_0` would underflow.

    Parameters
    ----------
    x : Tensor
        Input tensor; only :math:`x > 0` is in the domain.

    Returns
    -------
    Tensor
        :math:`e^x K_0(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Asymptotic: :math:`e^x K_0(x) \sim \sqrt{\pi/(2x)}` as
    :math:`x \to \infty`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import scaled_modified_bessel_k0
    >>> scaled_modified_bessel_k0(lucid.tensor([1.0, 5.0, 20.0]))
    tensor([1.144, 0.5478, 0.2785])
    """
    if x.dtype == lucid.float64:
        return _bessel._k0e(x)
    return _half_line_split(
        _real(x),
        2.0,
        lambda s: _modified_bessel_k0_small(s) * lucid.exp(s),
        lambda big: _K0E_LARGE(2.0 / big) / lucid.sqrt(big),
        math.inf,
    )


def _modified_bessel_k1_small(x: Tensor) -> Tensor:
    y = (x / 2.0) ** 2
    poly = (
        1.0
        + 0.15443144 * y
        - 0.67278579 * y**2
        - 0.18156897 * y**3
        - 0.01919402 * y**4
        - 0.00110404 * y**5
        - 0.00004686 * y**6
    )
    return lucid.log(x / 2.0) * i1(x) + poly / x


def _modified_bessel_k1_large(x: Tensor) -> Tensor:
    return _K1E_LARGE(2.0 / x) * lucid.exp(-x) / lucid.sqrt(x)


def modified_bessel_k1(input: Tensor) -> Tensor:
    r"""Modified Bessel function of the second kind, order 1.

    Computes :math:`K_1(x)`, the order-one analogue of :math:`K_0`.
    Diverges as :math:`1/x` near the origin and decays exponentially
    for large argument.  Appears in Generalised-Inverse-Gaussian and
    Normal-Inverse-Gaussian log-densities.

    Parameters
    ----------
    input : Tensor
        Input tensor; only :math:`x > 0` is in the domain.

    Returns
    -------
    Tensor
        :math:`K_1(x)` element-wise, same shape and dtype as ``input``.

    Notes
    -----
    Asymptotic forms:

    .. math::

        K_1(x) \sim \frac{1}{x} \;\text{as } x \to 0^+,
        \qquad
        K_1(x) \sim \sqrt{\frac{\pi}{2x}}\, e^{-x}
                  \;\text{as } x \to \infty.

    Implementation: Abramowitz & Stegun §9.8 two-branch polynomial, accurate
    to :math:`\approx 10^{-7}`; a ``float64`` input takes the Cephes
    double-precision series instead.  Use :func:`scaled_modified_bessel_k1`
    for numerically stable evaluation at large ``x``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import modified_bessel_k1
    >>> modified_bessel_k1(lucid.tensor([0.5, 1.0, 5.0]))
    tensor([1.656, 0.6019, 0.004045])
    """
    if input.dtype == lucid.float64:
        return _bessel._k1(input)
    return _half_line_split(
        _real(input), 2.0, _modified_bessel_k1_small, _modified_bessel_k1_large, math.inf
    )


def scaled_modified_bessel_k1(x: Tensor) -> Tensor:
    r"""Exponentially scaled modified Bessel :math:`K_1`: :math:`e^x K_1(x)`.

    Computes :math:`e^x K_1(x)`, the bounded scaled variant of
    :func:`modified_bessel_k1` suitable for stable evaluation of
    log-densities involving :math:`K_1` at large argument.

    Parameters
    ----------
    x : Tensor
        Input tensor; only :math:`x > 0` is in the domain.

    Returns
    -------
    Tensor
        :math:`e^x K_1(x)` element-wise, same shape and dtype as ``x``.

    Notes
    -----
    Asymptotic: :math:`e^x K_1(x) \sim \sqrt{\pi/(2x)}` as
    :math:`x \to \infty`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import scaled_modified_bessel_k1
    >>> scaled_modified_bessel_k1(lucid.tensor([1.0, 5.0, 20.0]))
    tensor([1.636, 0.6003, 0.2854])
    """
    if x.dtype == lucid.float64:
        return _bessel._k1e(x)
    return _half_line_split(
        _real(x),
        2.0,
        lambda s: _modified_bessel_k1_small(s) * lucid.exp(s),
        lambda big: _K1E_LARGE(2.0 / big) / lucid.sqrt(big),
        math.inf,
    )


# ── Name aliases matching reference framework's lucid.special surface ─────────


def gammaln(input: Tensor) -> Tensor:
    r"""Natural logarithm of the absolute value of the gamma function.

    Convenience alias for :func:`lucid.lgamma`.  Computes
    :math:`\log|\Gamma(x)|` with numerical stability for large
    arguments where :math:`\Gamma(x)` would itself overflow.

    Parameters
    ----------
    input : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`\log|\Gamma(x)|` element-wise.

    Notes
    -----
    Definition:

    .. math::

        \log|\Gamma(x)| = \log\!\left|
            \int_0^\infty t^{x-1} e^{-t}\, dt \right|.

    For positive integer ``n``, :math:`\Gamma(n) = (n - 1)!`, so
    ``gammaln(n) = log((n-1)!)``.  ``gammaln`` returns ``+inf`` at
    non-positive integers (poles of the gamma function).

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import gammaln
    >>> gammaln(lucid.tensor([1.0, 5.0, 100.0]))
    tensor([0., 3.178, 359.1])
    """
    return lucid.lgamma(input)


def psi(input: Tensor) -> Tensor:
    r"""Digamma function :math:`\psi(x) = \frac{d}{dx} \log \Gamma(x)`.

    Convenience alias for :func:`lucid.digamma`.  The digamma function
    is the logarithmic derivative of the gamma function and the
    fundamental building block of the polygamma family.

    Parameters
    ----------
    input : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`\psi(x)` element-wise.

    Notes
    -----
    Definition:

    .. math::

        \psi(x) = \frac{d}{dx} \log \Gamma(x)
                = \frac{\Gamma'(x)}{\Gamma(x)}
                = -\gamma + \sum_{k=0}^\infty
                  \left(\frac{1}{k + 1} - \frac{1}{k + x}\right),

    where :math:`\gamma \approx 0.5772` is the Euler–Mascheroni
    constant.  Used in the derivatives of log-likelihoods of the
    Gamma / Beta / Dirichlet distributions.  Has simple poles at the
    non-positive integers.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import psi
    >>> psi(lucid.tensor([1.0, 2.0, 10.0]))
    tensor([-0.5772, 0.4228, 2.252])
    """
    return lucid.digamma(input)


def expit(input: Tensor) -> Tensor:
    r"""Logistic sigmoid (expit) function :math:`\sigma(x)`.

    Convenience alias for :func:`lucid.sigmoid`.  Maps the real line
    onto the open interval :math:`(0, 1)` and is the inverse of
    :func:`lucid.logit`.  Unlike :func:`lucid.sigmoid` it accepts a complex
    input, evaluated as :math:`1 / (1 + e^{-z})` in real arithmetic.

    Parameters
    ----------
    input : Tensor
        Input tensor; any floating-point or complex dtype.

    Returns
    -------
    Tensor
        :math:`\sigma(x) = 1 / (1 + e^{-x})` element-wise, same shape
        and dtype as ``input``.

    Notes
    -----
    Definition:

    .. math::

        \sigma(x) = \frac{1}{1 + e^{-x}}
                  = \frac{e^x}{1 + e^x}.

    Symmetry: :math:`\sigma(-x) = 1 - \sigma(x)`.  Derivative:
    :math:`\sigma'(x) = \sigma(x)\,(1 - \sigma(x))`.  Frequently used
    as the probabilistic-output activation of binary classifiers.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import expit
    >>> expit(lucid.tensor([-2.0, 0.0, 2.0]))
    tensor([0.1192, 0.5, 0.8808])
    """
    if input.is_complex():
        return _complex_expit(input)
    return lucid.sigmoid(input)


def modified_bessel_i0(input: Tensor) -> Tensor:
    r"""Modified Bessel function of the first kind, order 0.

    Convenience alias for :func:`lucid.i0`.  Computes :math:`I_0(x)`,
    the order-zero regular solution of the modified Bessel equation,
    used in the von Mises density and rotational kernel families.

    Parameters
    ----------
    input : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`I_0(x)` element-wise, same shape and dtype as ``input``.

    Notes
    -----
    Series form:

    .. math::

        I_0(x) = \sum_{k=0}^\infty \frac{1}{(k!)^2}
                 \left(\frac{x}{2}\right)^{2k}.

    Even function; :math:`I_0(0) = 1`; grows as
    :math:`e^{|x|}/\sqrt{2\pi |x|}` for large argument — use
    :func:`i0e` for numerical stability at large ``|x|``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import modified_bessel_i0
    >>> modified_bessel_i0(lucid.tensor([0.0, 1.0, 3.0]))
    tensor([1., 1.266, 4.881])
    """
    return lucid.i0(input)


def modified_bessel_i1(input: Tensor) -> Tensor:
    r"""Modified Bessel function of the first kind, order 1.

    Convenience wrapper around :func:`i1`.  Computes :math:`I_1(x)`,
    the order-one regular solution of the modified Bessel equation,
    forming a natural pair with :math:`I_0` for log-density evaluation
    of Rician / Bessel-family distributions.

    Parameters
    ----------
    input : Tensor
        Input tensor; any floating-point dtype.

    Returns
    -------
    Tensor
        :math:`I_1(x)` element-wise.

    Notes
    -----
    Series form:

    .. math::

        I_1(x) = \sum_{k=0}^\infty \frac{1}{k!\,(k+1)!}
                 \left(\frac{x}{2}\right)^{2k+1}.

    Odd function; :math:`I_1(0) = 0`; grows like :math:`I_0` at
    infinity.  Use :func:`i1e` for stable evaluation at large ``|x|``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import modified_bessel_i1
    >>> modified_bessel_i1(lucid.tensor([0.0, 1.0, 3.0]))
    tensor([0., 0.5652, 3.953])
    """
    return i1(input)


__all__ = [
    "erfcx",
    "i0e",
    "i1",
    "i1e",
    "ndtr",
    "ndtri",
    "log_ndtr",
    "xlog1py",
    "entr",
    "multigammaln",
    "polygamma",
    "spherical_bessel_j0",
    # Orthogonal polynomials (12)
    "chebyshev_polynomial_t",
    "chebyshev_polynomial_u",
    "chebyshev_polynomial_v",
    "chebyshev_polynomial_w",
    "shifted_chebyshev_polynomial_t",
    "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v",
    "shifted_chebyshev_polynomial_w",
    "hermite_polynomial_h",
    "hermite_polynomial_he",
    "legendre_polynomial_p",
    "laguerre_polynomial_l",
    # Bessel + modified Bessel K (8)
    "bessel_j0",
    "bessel_j1",
    "bessel_y0",
    "bessel_y1",
    "modified_bessel_k0",
    "modified_bessel_k1",
    "scaled_modified_bessel_k0",
    "scaled_modified_bessel_k1",
    # Hurwitz zeta, regularized incomplete gamma (3)
    "zeta",
    "gammainc",
    "gammaincc",
    # Name aliases
    "gammaln",
    "psi",
    "expit",
    "modified_bessel_i0",
    "modified_bessel_i1",
]
