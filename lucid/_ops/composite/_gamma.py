"""The gamma family: :math:`\\ln\\Gamma` and the polygammas :math:`\\psi^{(n)}`.

``lucid.lgamma`` is :func:`_lgamma`: a Lanczos approximation (error 2e-10 in
float32, 1e-15 in float64; :class:`_Lanczos`) on the right half, Euler's
reflection on the left.  The approximation is evaluated as a *difference* from
its own value at ``x = 1`` (:meth:`_Lanczos.difference`), which makes ``lgamma``
exactly 0 at its two zeros ``x = 1, 2`` and accurate near them, and runs once
per element whichever half the element is on.

Two entries share each evaluation.  The public ones (``lucid.lgamma``,
``lucid.digamma``, ``lucid.special.polygamma``) cover the whole real line.
:func:`_lgamma_positive`, :func:`_digamma_positive` and
:func:`_polygamma_positive` are for a caller that guarantees ``x > 0`` (a
concentration, a count plus one, a degrees-of-freedom): they skip the
reflection, the pole tests and the infinity guard: a third of ``lgamma``'s
engine events and over half of ``digamma``'s.  Their answer on a non-positive
``x`` is not specified.

``lucid.digamma`` is ``n = 0`` and ``lucid.special.polygamma`` is every
``n``; both call :func:`_polygamma`.  They used to be two separate series
that shared one defect: neither had a left half.  The shift-by-8
recurrence only moves an argument above 8 if it started above 0, so for
``x < -8`` the asymptotic series was evaluated at a negative point and
answered ``nan``; between ``-8`` and ``0`` it answered finite numbers
that were simply wrong (``digamma(-7.9)`` was ``-3897.67`` against
``-7.54``), and the poles came back ``-inf`` where the reference has
``nan``.

The left half comes from differentiating Euler's reflection formula
:math:`\\psi(1 - x) - \\psi(x) = \\pi \\cot(\\pi x)` ``n`` times:

.. math::

    \\psi^{(n)}(x) = (-1)^n \\psi^{(n)}(1 - x) - \\pi^{n+1} P_n(\\cot \\pi x),

where :math:`\\frac{d^n}{dx^n} \\cot(\\pi x) = \\pi^n P_n(\\cot \\pi x)` and
:math:`P_n` is a polynomial with integer coefficients
(:func:`_cot_derivative_poly`).  ``1 - x`` is positive whenever ``x`` is
negative, so the right-half series covers it.

``lucid.special.gammainc`` / ``gammaincc`` are :func:`_regularized_gamma`,
the regularized incomplete gamma :math:`P(a, x)` / :math:`Q(a, x)` — the
one owner the Gamma-family CDFs in ``lucid.distributions`` go through.
"""

import functools
import math
import operator
from collections.abc import Callable
from typing import final, override

import lucid
from lucid._ops._bool_refusal import refuse_common
from lucid._ops.composite._opmath import _in_float32, _real
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx

# A Lanczos approximation is  lgamma(z + 1) = F(z) = (c - z) + (z + 1/2) log t + log s(z),
# t = z + g + 1/2,  s(z) = P_0 + sum_{k=1..n} P_k / (z + k),  to the approximation's own error.
# Two are used: Numerical Recipes (2nd ed.) g = 5, n = 6, error 2e-10, for float32 (its epsilon is
# 1.2e-7, and six terms are two passes fewer than the eight of the older g = 7, n = 9 set), and
# Godfrey's g = 607/128, n = 14 (Numerical Recipes, 3rd ed.), error ~1e-15, for float64.
_LANCZOS_F32_G = 5.0
_LANCZOS_F32_P: tuple[float, ...] = (
    1.000000000190015,
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5,
)
_LANCZOS_F64_G = 607.0 / 128.0
_LANCZOS_F64_P: tuple[float, ...] = (
    0.99999999999999709182,
    57.156235665862923517,
    -59.597960355475491248,
    14.136097974741747174,
    -0.49191381609762019978,
    0.33994649984811888699e-4,
    0.46523628927048575665e-4,
    -0.98374475304879564677e-4,
    0.15808870322491248884e-3,
    -0.21026444172410488319e-3,
    0.21743961811521264320e-3,
    -0.16431810653676389022e-3,
    0.84418223983852743293e-4,
    -0.26190838401581408670e-4,
    0.36899182659531622704e-5,
)


@final
class _Lanczos:
    """The constants of one Lanczos approximation's *difference form*.

    Summed as written, F is three terms of size ~7 (float32: ulp 4.8e-7) that cancel to 0 at
    ``z = 0`` and ``z = 1`` (``x = 1, 2``): ``lgamma(2)`` was 4.77e-7 in float32 and ``lgamma(1)``
    -8.9e-16 in float64 (the approximation's own error at the point), and near a zero the error
    stays that size against a true value of 0.577 |x - 1|.

    The same F, grouped as a *difference* from its value at ``z = 0``,

        lgamma(1 + z) = F(z) - F(0)
                      = z (log t0 - 1) + (z + 1/2) log1p(z / t0) + log1p(z u(z)),
        u(z)          = -sum_k [P_k / (k s0)] / (z + k),         (s0 = s(0), t0 = g + 1/2)

    has no term that does not vanish with ``z``: it is exactly 0 at ``z = 0`` and carries the
    digits of a small ``z``.  ``s(z) / s0 - 1 = z u(z)`` is the same sum with each term cancelled
    by its value at 0 (``P_k/(z+k) - P_k/k = -z P_k / (k (z+k))``).  F(0) is replaced by the true
    value 0, which moves the approximation by its own error at 1 and nowhere else.

    ``log1p(z u)`` loses ``1/(1 + z u) = s0/s(z)`` of the absolute error of ``z u`` (up to 42 for
    g = 5, 263 for the older g = 7), and the terms of ``u`` alternate in sign and are an order
    larger than their sum.  Adjacent terms are therefore added as one fraction,

        c_a / (z + a) + c_b / (z + b) = ((c_a + c_b) z + (c_a b + c_b a)) / ((z + a)(z + b)),

    whose parts no longer cancel against each other.  Held to a 50-digit oracle (a thousand points
    per band), the worst error over 1e-3 .. 1e6 is 3.9 (float32) / 5.6 (float64) epsilons where
    the direct form's was 18 / 15, and the direct form's mid-band noise is kept: at 10-20 float64
    is 2.6 against 2.9 (the g = 7 set without the pairing: 8).  Near ``x = 1, 2`` the error is at
    most 5 / 8 epsilons *of the tiny value* (float32 / float64) where it was 10^7.  The large
    bands carry the rounding of three terms of size ``z log z`` where the direct form carried two:
    up to 0.6 epsilon more there (at most 2.1 in all).

    The constants are formed in double precision.
    """

    def __init__(self, g: float, coeffs: tuple[float, ...]) -> None:
        n = len(coeffs) - 1
        if n < 2 or n % 2:
            raise ValueError(
                f"a Lanczos set adds its partial fractions in pairs: need an even count >= 2 of "
                f"terms after P_0, got {n}"
            )
        t0 = g + 0.5
        s0 = coeffs[0] + math.fsum(coeffs[k] / k for k in range(1, n + 1))
        diff = [-coeffs[k] / (k * s0) for k in range(1, n + 1)]
        self.c1 = math.log(t0) - 1.0
        self.inv_t0 = 1.0 / t0
        # (numerator slope, numerator intercept) of the fraction for the terms k = i + 1, i + 2
        self.pairs: tuple[tuple[float, float], ...] = tuple(
            (diff[i] + diff[i + 1], diff[i] * (i + 2) + diff[i + 1] * (i + 1))
            for i in range(0, n, 2)
        )

    def difference(self, z: Tensor, one: Tensor) -> Tensor:
        """:math:`\\ln\\Gamma(1 + z)` for ``z >= -1/2``, exactly 0 at ``z = 0``.

        ``one`` is a 0-d one of ``z``'s dtype and device: the shifts ``z + k`` are built by adding
        it to the last, one op each, not a constant tensor each.
        """
        shifted = z
        fractions: list[Tensor] = []
        for slope, intercept in self.pairs:
            shifted = shifted + one
            upper = shifted + one
            fractions.append((z * slope + intercept) / (shifted * upper))
            shifted = upper
        total = fractions[0]
        for fraction in fractions[1:]:
            total = total + fraction
        return z * self.c1 + (z + 0.5) * lucid.log1p(z * self.inv_t0) + lucid.log1p(z * total)


_LANCZOS_F32 = _Lanczos(_LANCZOS_F32_G, _LANCZOS_F32_P)
_LANCZOS_F64 = _Lanczos(_LANCZOS_F64_G, _LANCZOS_F64_P)


def _lgamma_core(a: Tensor, one: Tensor) -> Tensor:
    """:math:`\\ln\\Gamma(a)` for ``a >= 1/2``, exactly 0 at ``a = 1`` and ``a = 2``.

    Below 3/2 the difference form is read at ``z = a - 1``.  From 3/2 on it is read one step
    down, ``lgamma(a) = lgamma(a - 1) + log(a - 1)``, so that the second zero ``a = 2`` is a zero
    of the difference form as well; ``log(a - 1)`` is taken of the exact ``a - 1`` there
    (subtracting 1 from ``a`` in [1, 2^24] is exact) and of 1 elsewhere, so it is 0.
    """
    am1 = a - one
    hi = am1 >= 0.5
    z = lucid.where(hi, am1 - one, am1)
    lanczos = _LANCZOS_F64 if a.dtype == lucid.float64 else _LANCZOS_F32
    return lanczos.difference(z, one) + lucid.log(lucid.where(hi, am1, one))


def _lgamma(x: Tensor, positive: bool = False) -> Tensor:
    """:math:`\\ln|\\Gamma(x)|` on the whole real line; float32 for 16 bits.

    ``positive`` is the caller's promise that ``x > 0`` (:func:`_lgamma_positive`).

    An integer ``x`` is the default float first (:func:`_real`): the integer
    went into the Lanczos series as itself and failed to hold its first
    ``inf``.
    """
    return _in_float32(_lgamma_positive_wide if positive else _lgamma_wide, _real(x))


def _lgamma_positive_wide(x: Tensor) -> Tensor:
    """:func:`_lgamma` for ``x > 0`` in ``x``'s own dtype, float32 or float64."""
    # Below 1/2 the difference form is read one step up,
    # lgamma(x) = lgamma(x + 1) - log(x): the two terms are > 0.57 apart from 0 and do not
    # cancel, and log(x) is the whole answer for a tiny x.  ``log`` meets 1 off that side.
    one = lucid.ones((), dtype=x.dtype, device=x.device)
    low = x < 0.5
    return _lgamma_core(lucid.where(low, x + one, x), one) - lucid.log(lucid.where(low, x, one))


def _lgamma_wide(x: Tensor) -> Tensor:
    """:func:`_lgamma` in ``x``'s own dtype, float32 or float64."""
    # The Lanczos series is only valid on the right half of the line, and
    # nothing here said so: ``lgamma(-0.5)`` came back NaN against the
    # reference's 1.2655, and ``lgamma(1e-30)`` came back inf against
    # 69.0776.  Γ has poles at the non-positive integers and is perfectly
    # finite everywhere between them, so the left half is not a domain
    # error — it is the half this implementation never covered.
    #
    # Euler's reflection formula supplies it:
    #
    #     lgamma(x) = log(π) − log|sin(πx)| − lgamma(1 − x)
    #
    # The series is needed at x or at 1 − x, never both, so it runs once at
    # whichever the element is on; ``where`` selects the argument, not the
    # result, and each side is given one inside its own half.  Without that
    # the losing branch divides by zero at ``z + k`` and the NaN reaches the
    # *gradient* through the mask, which is the usual way a where-guarded
    # formula goes wrong.
    one = lucid.ones((), dtype=x.dtype, device=x.device)
    reflect = x < 0.5
    arg = lucid.where(reflect, one - x, x)
    # ±inf has no finite half to read: 1 − (−inf) and +inf are both inf.
    core = _lgamma_core(lucid.where(lucid.isinf(arg), one, arg), one)
    # ``sin(πx)`` is zero at every integer, where Γ has its poles and the
    # answer is +inf.  Only the reflected side may see it: elsewhere it is
    # replaced by 1, so ``log`` never meets a zero it was not asked about.
    # |sin(pi x)| has period 1 and ``x - round(x)`` is exact, so the sine is
    # taken in [-pi/2, pi/2]: at pi * x the product's rounding grows with
    # |x|, and in float32 the gradient at -1000.5 came out 6.90836 against
    # digamma's 6.90875.  Only the reflected side takes it, so that +inf on
    # the other never forms inf - round(inf) = NaN under the mask.
    x_left = lucid.where(reflect, x, 0.5)
    sin_term = lucid.abs(lucid.sin(math.pi * (x_left - lucid.round(x_left))))
    sin_safe = lucid.where(reflect, sin_term, one)
    out = lucid.where(reflect, math.log(math.pi) - lucid.log(sin_safe) - core, core)

    # The poles are named rather than computed.  ``sin(π · -1.0)`` came out
    # 1.2e-16 before the reduction above, and lgamma(-1) answered 37.78
    # instead of +inf; the reduced form is exact at integers, but -inf
    # reduces to NaN, and a pole should not hinge on rounding either way.
    at_pole = (x <= 0.0) & (x == lucid.round(x))
    out = lucid.where(at_pole, math.inf, out)
    # The Lanczos form at +inf is inf - inf.  +inf is handed to x * x
    # instead, which has the same limit and, like digamma, an infinite
    # derivative there (x log x would form inf / inf in its backward).
    big = x == math.inf
    x_big = lucid.where(big, x, one)
    return lucid.where(big, x_big * x_big, out)


def _lgamma_positive(x: Tensor) -> Tensor:
    """:math:`\\ln\\Gamma(x)` for a caller that guarantees ``x > 0`` (:class:`_LogGamma`).

    One evaluation of the difference form, with no reflection, pole test or infinity guard.
    ``0`` gives ``+inf`` as :func:`_lgamma` does; ``nan`` stays ``nan``; a negative ``x`` or
    ``+inf`` is outside the promise and answers ``nan``.
    """
    out = _LogGamma.apply(x, True)
    assert isinstance(out, Tensor)
    return out


@final
class _LogGamma(Function):
    """``lgamma`` with its exact derivative, ``digamma`` (LCD-772).

    The value is :func:`_lgamma`.  Differentiating that composite instead
    meets ``log(sin(pi x)) = log 0`` at the non-positive integers: the pole
    is named in the value but not in the gradient, which came out NaN at
    ``0`` where the reference has ``digamma(0) = -inf`` (``+inf`` at ``-0``).
    ``digamma`` also carries the other poles (NaN at the negative integers),
    ``+inf`` at ``+inf``, and is itself differentiable, so higher orders
    follow.

    ``positive`` selects the entry of a caller that guarantees ``x > 0``: the value and the slope
    both skip the left half.
    """

    _trace_form = staticmethod(_lgamma)

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, positive: bool) -> Tensor:
        ctx.save_for_backward(x)
        ctx.positive = positive
        return _lgamma(x.detach(), positive)

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:
        (x,) = ctx.saved_tensors
        return grad * _polygamma(0, x, bool(ctx.positive)), None


def _integer_order(value: int, name: str) -> int:
    """``value`` as an ``int`` order, or ``TypeError`` — as the reference.

    ``polygamma`` and ``multigammaln`` take an integer order.  ``int(value)``
    truncated ``1.5`` to ``1`` and answered the wrong function without a
    word; an index-like value (a Python or 0-d integer, a NumPy integer)
    passes, and a float, a ``bool`` or ``None`` is refused.
    """
    if isinstance(value, bool):
        raise TypeError(f"`{name}` must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"`{name}` must be an integer, not {type(value).__name__}") from None


# B_2, B_4, ..., B_12.  Every odd Bernoulli number past B_1 is zero, which
# is why the asymptotic series skips odd powers.
_BERNOULLI_2K: tuple[float, ...] = (
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
)

# The recurrence moves the argument up by this much before the asymptotic
# series is evaluated.  At x + 8 >= 8, six Bernoulli terms leave the n = 0
# series a truncation error near 1e-14 relative, below float64's own noise
# for the values involved.
_SHIFT = 8

# ``digamma``'s Bernoulli terms and shift for float32.  Float64 keeps the plan above.  Float32's
# epsilon is 1e-7, so at x + 4 >= 4 four terms leave a truncation of 7e-9 (worst at x -> 0, where
# psi is large), and the shorter shift also sums fewer rounded reciprocals: against a 50-digit
# oracle the worst error over (0, 1e30] is 3.4 epsilons, from 5.3 at the shift of 8, for half the
# passes.
_SHIFT_F32 = 4
_TERMS_F32 = 4


@functools.cache
def _cot_derivative_poly(n: int) -> tuple[int, ...]:
    """Coefficients (lowest power first) of :math:`P_n`.

    :math:`P_0(c) = c`, and since :math:`\\frac{d}{dx} \\cot(\\pi x) =
    -\\pi (1 + \\cot^2 \\pi x)`, :math:`P_{k+1}(c) = -(1 + c^2) P_k'(c)`.
    """
    coeffs = [0, 1]
    for _ in range(n):
        deriv = [i * coeffs[i] for i in range(1, len(coeffs))]
        nxt = [0] * (len(deriv) + 2)
        for i, d in enumerate(deriv):
            nxt[i] -= d
            nxt[i + 2] -= d
        coeffs = nxt
    return tuple(coeffs)


def _psi_series(x: Tensor, one: Tensor) -> Tensor:
    """:math:`\\psi(x)` for ``x > 0``: recurrence shift, then the series."""
    # psi(x) = psi(x + K) - sum_{k<K} 1/(x + k), and
    # psi(z) ~ log z - 1/(2z) - sum_k B_2k / (2k z^2k).  ``one`` is a 0-d one of x's dtype and
    # device: x + k is built by adding it to the last, one op per term.
    shift, terms = (
        (_SHIFT, len(_BERNOULLI_2K)) if x.dtype == lucid.float64 else (_SHIFT_F32, _TERMS_F32)
    )
    xk = x
    total = lucid.reciprocal(xk)
    for _ in range(shift - 1):
        xk = xk + one
        total = total + lucid.reciprocal(xk)
    xr = xk + one
    r = lucid.reciprocal(xr)
    r2 = r * r
    # Horner, opened with the top term so that no full-size constant is filled first.
    poly = r2 * (_BERNOULLI_2K[terms - 1] / (2.0 * terms)) + (
        _BERNOULLI_2K[terms - 2] / (2.0 * (terms - 1))
    )
    for k in range(terms - 2, 0, -1):
        poly = poly * r2 + (_BERNOULLI_2K[k - 1] / (2.0 * k))
    return lucid.log(xr) - 0.5 * r - r2 * poly - total


def _positive(n: int, x: Tensor) -> Tensor:
    """:math:`\\psi^{(n)}(x)` for ``x > 0``: recurrence shift, then the series."""
    one = lucid.ones((), dtype=x.dtype, device=x.device)
    if n == 0:
        return _psi_series(x, one)

    # Every power is taken of a reciprocal, never divided by: at x = +inf
    # the backward of 1 / z^m is (-m z^(m-1)) / z^(2m) = inf / inf, while
    # that of (1/z)^m is m (1/z)^(m-1) * (-1/z^2) = 0 * 0.  It also keeps a
    # float32 x = 1e-6 from passing through x^(n+1) = 1e-24 on the way to
    # 1/x^(n+1).
    #
    # psi^(n)(x) = psi^(n)(x + K) + (-1)^(n+1) n! sum_{k<K} 1/(x + k)^(n+1), and
    # psi^(n)(z) ~ (-1)^(n+1) [ (n-1)!/z^n + n!/(2 z^(n+1))
    #                          + sum_k B_2k (2k+n-1)!/(2k)! / z^(2k+n) ].
    # The series is written with (n-1)!/z^n factored out: evaluated term by
    # term, B_2k (2k+n-1)!/(2k)! alone passes float32's range from about
    # n = 31 although the terms are tiny.  The one quantity that genuinely
    # grows like a factorial is evaluated once, in log space.
    xk = x
    powers = lucid.reciprocal(xk) ** (n + 1)
    for _ in range(1, _SHIFT):
        xk = xk + one
        powers = powers + lucid.reciprocal(xk) ** (n + 1)
    xr = xk + one
    r = lucid.reciprocal(xr)
    r2 = r * r
    sign = 1.0 if n % 2 == 1 else -1.0
    shift = (sign * float(math.factorial(n))) * powers
    lead = lucid.exp(math.lgamma(n) - n * lucid.log(xr))
    coeffs = [
        b * float(math.prod(range(n, 2 * k + n))) / float(math.factorial(2 * k))
        for k, b in enumerate(_BERNOULLI_2K, start=1)
    ]
    poly = r2 * coeffs[-1] + coeffs[-2]
    for c in reversed(coeffs[:-2]):
        poly = poly * r2 + c
    bracket = 1.0 + (n / 2.0) * r + r2 * poly
    return sign * lead * bracket + shift


def _horner(coeffs: tuple[int, ...], c: Tensor) -> Tensor:
    acc = lucid.full_like(c, float(coeffs[-1]))
    for a in reversed(coeffs[:-1]):
        acc = acc * c + float(a)
    return acc


def _polygamma_value(n: int, x: Tensor, positive: bool = False) -> Tensor:
    """:math:`\\psi^{(n)}(x)` on the whole real line, ``n >= 0``, as plain ops.

    A 16-bit ``x`` is evaluated in float32: the recurrence's shift and the
    series cancel, and ``digamma(1.5)`` in float16 was 0.0391 for 0.0365.
    An integer ``x`` is the default float first (:func:`_real`).  ``positive`` is the caller's
    promise that ``x > 0`` (:func:`_polygamma_positive`): the series alone.
    """
    return _in_float32(lambda v: _positive(n, v) if positive else _polygamma_wide(n, v), _real(x))


@final
class _Polygamma(Function):
    """:math:`\\psi^{(n)}` with its exact derivative :math:`\\psi^{(n+1)}` (LCD-772).

    The value is :func:`_polygamma_value`, whose poles are named constants and
    so carry a gradient of 0 — ``digamma`` at ``-3`` had a slope of 0 next to
    a trigamma of ``+inf``.  The derivative is the next order of the same
    function, so a pole's slope is that order's pole value, and every order
    above is differentiable in turn.  ``positive`` selects the entry of a caller that guarantees
    ``x > 0``; the slope is then the next order of that entry.
    """

    _trace_form = staticmethod(_polygamma_value)

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, n: int, x: Tensor, positive: bool) -> Tensor:
        ctx.save_for_backward(x)
        ctx.order = n
        ctx.positive = positive
        return _polygamma_value(n, x.detach(), positive)

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[None, Tensor, None]:
        (x,) = ctx.saved_tensors
        return None, grad * _polygamma(ctx.order + 1, x, bool(ctx.positive)), None


def _polygamma(n: int, x: Tensor, positive: bool = False) -> Tensor:
    """:math:`\\psi^{(n)}(x)` on the whole real line, ``n >= 0`` (:class:`_Polygamma`)."""
    out = _Polygamma.apply(n, x, positive)
    assert isinstance(out, Tensor)
    return out


def _polygamma_positive(n: int, x: Tensor) -> Tensor:
    """:math:`\\psi^{(n)}(x)` for a caller that guarantees ``x > 0``: the series, no left half.

    ``0`` gives the pole (``-inf`` for ``n = 0``) and ``nan`` stays ``nan``; a negative ``x`` is
    outside the promise.
    """
    return _polygamma(n, x, True)


def _digamma_positive(x: Tensor) -> Tensor:
    """:math:`\\psi(x)` for a caller that guarantees ``x > 0`` (:func:`_polygamma_positive`)."""
    return _polygamma(0, x, True)


def _polygamma_wide(n: int, x: Tensor) -> Tensor:
    """:func:`_polygamma` in ``x``'s own dtype, float32 or float64."""
    one = lucid.ones((), dtype=x.dtype, device=x.device)
    # The poles are the non-positive integers, -inf among them.
    pole = (x <= 0.0) & (x == lucid.floor(x))
    reflect = (x < 0.0) & ~pole
    # Each element needs the right-half series at one point only — x, or
    # 1 - x where it is reflected — so the series runs once, at that point.
    # A pole gets 1 instead: ``where`` selects rather than short-circuits,
    # and a 1/0 under the mask would reach the gradient as 0 * inf.
    series = _positive(n, lucid.where(reflect, 1.0 - x, lucid.where(pole, one, x)))

    # cot(pi x) has period 1, and ``x - round(x)`` is exact, so the cotangent
    # is taken at a point r in [-1/2, 1/2] instead of at pi * x, whose
    # rounding grows with |x|.  cos(pi r) is taken as sin(pi (1/2 - |r|)),
    # whose argument is exact near |r| = 1/2: cos(pi/2) itself rounds to
    # 6e-17, and at the half-integers, where P_n(0) = 0 for even n, that
    # residue was the whole error — 4e-9 relative in polygamma(2, -1000.5).
    frac = lucid.where(reflect, x - lucid.round(x), 0.25)
    cot = lucid.sin(math.pi * (0.5 - lucid.abs(frac))) / lucid.sin(math.pi * frac)
    cot_term = math.pi ** (n + 1) * _horner(_cot_derivative_poly(n), cot)
    left = (series if n % 2 == 0 else -series) - cot_term
    out = lucid.where(reflect, left, series)

    if n == 0:
        # psi(x) ~ -1/x at the origin, so psi(+0) = -inf and psi(-0) = +inf;
        # the other poles have no sign and are NaN.
        at_zero = x == 0.0
        out = lucid.where(pole, math.nan, out)
        return lucid.where(at_zero, -1.0 / lucid.where(at_zero, x, one), out)
    # (-1)^(n+1) n! / (x + m)^(n+1): with n odd the power is even and the pole
    # is +inf from both sides; with n even the reference takes the sign
    # (-1)^(n+1) as well, and so does this.
    pole_value = math.inf if n % 2 == 1 else -math.inf
    return lucid.where(pole, pole_value, out)


# ── Regularized incomplete gamma ──────────────────────────────────────────────
#
# P(a, x) = γ(a, x) / Γ(a) and Q(a, x) = 1 - P(a, x), evaluated as SciPy's
# ``igam`` / ``igamc`` evaluate them (Cephes, with Boost's Lanczos sum and
# Temme's uniform expansion — see THIRD_PARTY_NOTICES.md).  The reference
# framework uses the same algorithm, so the branch taken for each (a, x) is
# the reference's own.
#
# The scalar algorithm loops until a term is negligible.  Here every element
# loops together, and the loop ends once every element that takes the branch
# has converged — checked every ``_IGAM_CHUNK`` steps, since each check reads
# a flag back to the host.  Elements on other branches are handed a harmless
# argument, so they neither overflow nor keep the loop running.

_IGAM_MAX_ITER = 2000
_IGAM_CHUNK = 16
_IGAM_SMALL = 20.0
_IGAM_LARGE = 200.0
_IGAM_SMALL_RATIO = 0.3
_IGAM_LARGE_RATIO = 4.5

# Boost's ``lanczos13m53`` sum ``lanczos_sum_expg_scaled``: numerator and
# denominator, highest power first.  (C) Copyright John Maddock 2006;
# distributed under the Boost Software License, Version 1.0 — see
# THIRD_PARTY_NOTICES.md.
_LANCZOS13_G = 6.024680040776729583740234375
_LANCZOS13_NUM: tuple[float, ...] = (
    0.006061842346248906525783753964555936883222,
    0.5098416655656676188125178644804694509993,
    19.51992788247617482847860966235652136208,
    449.9445569063168119446858607650988409623,
    6955.999602515376140356310115515198987526,
    75999.29304014542649875303443598909137092,
    601859.6171681098786670226533699352302507,
    3481712.15498064590882071018964774556468,
    14605578.08768506808414169982791359218571,
    43338889.32467613834773723740590533316085,
    86363131.28813859145546927288977868422342,
    103794043.1163445451906271053616070238554,
    56906521.91347156388090791033559122686859,
)
_LANCZOS13_DEN: tuple[float, ...] = (
    1.0,
    66.0,
    1925.0,
    32670.0,
    357423.0,
    2637558.0,
    13339535.0,
    45995730.0,
    105258076.0,
    150917976.0,
    120543840.0,
    39916800.0,
    0.0,
)


class _IgamConstants:
    """The per-precision constants of the scalar algorithm."""

    def __init__(self, dtype: lucid.dtype) -> None:
        double = dtype == lucid.float64
        self.machep = 1.11022302462515654042e-16 if double else 5.9604644775390625e-8
        self.big = 4.503599627370496e15 if double else 16777216.0
        self.biginv = 2.22044604925031308085e-16 if double else 5.9604644775390625e-8
        self.maxlog = 7.09782712893383996843e2 if double else 88.72283905206835


def _horner_float(coeffs: tuple[float, ...], y: Tensor) -> Tensor:
    """``sum_i coeffs[i] * y**i`` — coefficients lowest power first."""
    acc = lucid.full_like(y, coeffs[-1])
    for c in reversed(coeffs[1:-1]):
        acc = acc * y + c
    acc = acc * y
    return acc + coeffs[0] if coeffs[0] != 0.0 else acc


def _lanczos_sum_expg_scaled(a: Tensor) -> Tensor:
    """Boost's rational Lanczos sum; above 1 it is evaluated in ``1/a``."""
    # Both polynomials have degree 12, so in ``1/a`` the ratio needs no
    # ``a**(N - M)`` factor: the coefficients are just read in reverse.
    big = lucid.abs(a) > 1.0
    y_small = lucid.where(big, lucid.zeros_like(a), a)
    y_big = 1.0 / lucid.where(big, a, lucid.ones_like(a))
    num_low = tuple(reversed(_LANCZOS13_NUM))
    den_low = tuple(reversed(_LANCZOS13_DEN))
    small = _horner_float(num_low, y_small) / _horner_float(den_low, y_small)
    large = _horner_float(_LANCZOS13_NUM, y_big) / _horner_float(_LANCZOS13_DEN, y_big)
    return lucid.where(big, large, small)


def _igam_fac(a: Tensor, x: Tensor, k: _IgamConstants) -> Tensor:
    """:math:`x^a e^{-x} / \\Gamma(a)`, by the Lanczos form near ``x = a``."""
    far = lucid.abs(a - x) > 0.4 * lucid.abs(a)
    ax = a * lucid.log(x) - x - lucid.lgamma(a)
    far_value = lucid.where(ax < -k.maxlog, lucid.zeros_like(ax), lucid.exp(ax))

    fac = a + (_LANCZOS13_G - 0.5)
    res = lucid.sqrt(fac / math.e) / _lanczos_sum_expg_scaled(a)
    moderate = (a < 200.0) & (x < 200.0)
    # One exponent, not exp(a - x) * (x / fac)**a: the two factors can each
    # leave float32's normal range while their product does not, and Metal
    # flushes the subnormal — P(174.49, 105.92) came out 0 for 6.7e-10.
    direct = lucid.exp(a - x + a * lucid.log(x / fac))
    numfac = (x - a - _LANCZOS13_G + 0.5) / fac
    scaled = lucid.exp((-0.5 * a) * _deviance(numfac) + x * (0.5 - _LANCZOS13_G) / fac)
    near_value = res * lucid.where(moderate, direct, scaled)
    return lucid.where(far, far_value, near_value)


def _zeta_int(s: int) -> float:
    """Riemann :math:`\\zeta(s)` for an integer ``s >= 2``, to double precision.

    A prefix of 16 terms and the Euler-Maclaurin tail; the first omitted
    correction is below 1e-17 for every ``s >= 2``.
    """
    n = 16
    head = math.fsum(float(j) ** -s for j in range(1, n))
    a = float(n)
    tail = a ** (1 - s) / (s - 1) + 0.5 * a**-s
    rising = float(s)
    for j, b in enumerate(_BERNOULLI_2K, start=1):
        if j > 1:
            rising *= (s + 2 * j - 3) * (s + 2 * j - 2)
        tail += b / math.factorial(2 * j) * rising * a ** (-s - 2 * j + 1)
    return head + tail


# lgamma(1 + a) = -euler_gamma a + sum_{k>=2} (-1)^k zeta(k) a^k / k, for
# |a| < 1.  Below ``_LGAMMA1P_SERIES`` the series is used: ``lgamma`` at
# ``1 + a`` rounds the argument first and, being an approximation, is not
# exactly 0 at 1, so for a tiny ``a`` its absolute error swamped the
# result — ``gammaincc(1e-30, x)`` came out -8.9e-16 for a value near 1e-30.
_LGAMMA1P_SERIES = 0.2
_LGAMMA1P_COEFFS: tuple[float, ...] = (0.0, -0.5772156649015329) + tuple(
    (-1.0) ** k * _zeta_int(k) / k for k in range(2, 26)
)


def _lgamma1p(a: Tensor) -> Tensor:
    """:math:`\\ln\\Gamma(1 + a)`, accurate for ``a`` near 0."""
    near = lucid.abs(a) < _LGAMMA1P_SERIES
    series = _horner_float(_LGAMMA1P_COEFFS, lucid.where(near, a, lucid.zeros_like(a)))
    return lucid.where(near, series, lucid.lgamma(1.0 + a))


def _all_true(mask: Tensor) -> bool:
    return bool(lucid.all(mask).item())


def _any_true(mask: Tensor) -> bool:
    return bool(lucid.any(mask).item())


def _igam_series(a: Tensor, x: Tensor, k: _IgamConstants) -> Tensor:
    """P by its power series (DLMF 8.11.4) — for ``x`` up to about ``a``."""
    ax = _igam_fac(a, x, k)
    done = ax == 0.0
    r = a
    c = lucid.ones_like(a)
    ans = lucid.ones_like(a)
    for i in range(1, _IGAM_MAX_ITER + 1):
        r = r + 1.0
        c = c * x / r
        ans = ans + c
        if i % _IGAM_CHUNK == 0:
            done = done | (c <= k.machep * ans)
            if _all_true(done):
                break
    return lucid.where(ax == 0.0, lucid.zeros_like(a), ans * ax / a)


def _igamc_series(a: Tensor, x: Tensor, k: _IgamConstants) -> Tensor:
    """Q by DLMF 8.7.3, free of the cancellation in ``1 - P`` at small x."""
    fac = lucid.ones_like(a)
    total = lucid.zeros_like(a)
    done = lucid.zeros_like(a) != 0.0
    for n in range(1, _IGAM_MAX_ITER):
        fac = fac * (-x / float(n))
        term = fac / (a + float(n))
        total = total + term
        if n % _IGAM_CHUNK == 0:
            done = done | (lucid.abs(term) <= k.machep * lucid.abs(total))
            if _all_true(done):
                break
    logx = lucid.log(x)
    lg1p = _lgamma1p(a)
    head = -lucid.expm1(a * logx - lg1p)
    return head - a * lucid.exp(a * logx - lg1p) * total


def _igamc_continued_fraction(a: Tensor, x: Tensor, k: _IgamConstants) -> Tensor:
    """Q by its continued fraction (DLMF 8.9.2) — for ``x`` above ``a``.

    ``a`` and ``x`` are 1-D.  Every ``_IGAM_CHUNK`` steps the lanes that
    have converged are written out and dropped, so the step's ~15 ops run
    only on the lanes still converging — near ``x = a`` a few lanes take
    hundreds of steps that the rest would otherwise be carried through.
    """
    ax = _igam_fac(a, x, k)
    result = lucid.zeros_like(a)
    live = ax != 0.0
    lanes = lucid.arange(0, a.shape[0], 1, dtype=lucid.int64, device=a.device)[live]
    a, x, ax = a[live], x[live], ax[live]
    y = 1.0 - a
    z = x + y + 1.0
    pkm2, qkm2 = lucid.ones_like(a), x
    pkm1, qkm1 = x + 1.0, z * x
    ans = pkm1 / qkm1
    one = lucid.ones_like(ans)
    biginv = lucid.full_like(ans, k.biginv)
    done = lucid.zeros_like(a) != 0.0
    for i in range(1, _IGAM_MAX_ITER + 1):
        if lanes.shape[0] == 0:
            break
        y = y + 1.0
        z = z + 2.0
        yc = y * float(i)
        pk = pkm1 * z - pkm2 * yc
        qk = qkm1 * z - qkm2 * yc
        nonzero = qk != 0.0
        r = pk / lucid.where(nonzero, qk, one)
        t = lucid.where(nonzero, lucid.abs((ans - r) / r), one)
        ans = lucid.where(nonzero, r, ans)
        # The four are rescaled together, so every ratio read is unchanged.
        s = lucid.where(lucid.abs(pk) > k.big, biginv, one)
        pkm2, pkm1, qkm2, qkm1 = pkm1 * s, pk * s, qkm1 * s, qk * s
        done = done | (t <= k.machep)
        if i % _IGAM_CHUNK == 0 and _any_true(done):
            result[lanes[done]] = ans[done] * ax[done]
            keep = ~done
            lanes, ax, ans, y, z = (v[keep] for v in (lanes, ax, ans, y, z))
            pkm2, pkm1, qkm2, qkm1 = (v[keep] for v in (pkm2, pkm1, qkm2, qkm1))
            one, biginv, done = one[keep], biginv[keep], done[keep]
    if lanes.shape[0] > 0:
        result[lanes] = ans * ax
    return result


# -2 * (log1p(s) - s) / s^2 = sum_{k>=2} (-1)^k (2/k) s^{k-2} for |s| < 0.05,
# avoiding cancellation in log1p(sigma) - sigma when sigma = (x - a) / a -> 0.
_IGAM_ETA_SERIES = 0.05
_IGAM_SERIES_MIN_REL = 0.05
_ETA_SQ_COEFFS: tuple[float, ...] = tuple((-1.0) ** k * (2.0 / k) for k in range(2, 20))


def _deviance(s: Tensor) -> Tensor:
    """Twice the difference between s and log1p(s), including when they round alike."""
    near = s.abs() < _IGAM_ETA_SERIES
    small = lucid.where(near, s, lucid.zeros_like(s))
    series = small * small * _horner_float(_ETA_SQ_COEFFS, small)
    return lucid.where(near, series, 2.0 * (s - lucid.log1p(s)))


def _igam_asymptotic(a: Tensor, x: Tensor, *, lower: bool) -> Tensor:
    """P or Q by Temme's uniform expansion (DLMF 8.12.3/4) — for x near a."""
    from lucid._ops.composite._gamma_temme import _TEMME_D

    sgn = -1.0 if lower else 1.0
    sigma = (x - a) / a
    zero = lucid.zeros_like(a)
    eta_sq = _deviance(sigma)
    eta_abs = lucid.sqrt(eta_sq)
    eta = lucid.where(x > a, eta_abs, lucid.where(x < a, -eta_abs, zero))
    res = 0.5 * lucid.erfc(sgn * eta * lucid.sqrt(a / 2.0))

    # C_k(eta) for every k at once: the powers of eta times the table.
    n_terms = len(_TEMME_D)
    powers = lucid.arange(0, n_terms, 1, dtype=a.dtype, device=a.device)
    table = lucid.tensor(_TEMME_D, dtype=a.dtype, device=a.device)
    ck = lucid.matmul(eta.unsqueeze(-1) ** powers, table.mT)
    term = ck * a.unsqueeze(-1) ** (-powers)
    # An asymptotic series is summed until its terms start to grow: the
    # first term larger than the one before it, and every term after, are
    # dropped.
    mag = lucid.abs(term)
    grows = (mag[..., 1:] > mag[..., :-1]).to(a.dtype)
    grown = lucid.cumsum(lucid.cat([lucid.zeros_like(mag[..., :1]), grows], -1), -1)
    total = lucid.where(grown == 0.0, term, lucid.zeros_like(term)).sum(-1)
    scale = lucid.exp(-0.5 * a * eta_sq) / lucid.sqrt(2.0 * math.pi * a)
    return res + sgn * scale * total


def _igam_interior(a: Tensor, x: Tensor, *, lower: bool) -> Tensor:
    """P or Q off the edges of the domain, by the branch ``igam`` picks."""
    k = _IgamConstants(a.dtype)
    edge = (
        lucid.isnan(a) | lucid.isnan(x) | lucid.isinf(a) | lucid.isinf(x) | (a <= 0.0) | (x <= 0.0)
    )
    rel = lucid.abs(x - a) / a
    asym_large = (rel < _IGAM_LARGE_RATIO / lucid.sqrt(a)) | (
        (x < a) & (rel < _IGAM_SERIES_MIN_REL)
    )
    asym = ~edge & (
        ((a > _IGAM_SMALL) & (a < _IGAM_LARGE) & (rel < _IGAM_SMALL_RATIO))
        | ((a > _IGAM_LARGE) & asym_large)
    )
    rest = ~edge & ~asym
    # Q's own choice between its branches.  P takes ``1 - Q`` only for
    # x > max(1, a), where its series would converge slowly.
    q_by_p_series = (
        ((x > 1.1) & (x < a))
        | ((x <= 0.5) & (-0.4 / lucid.log(x) < a))
        | ((x > 0.5) & (x <= 1.1) & (x * 1.1 < a))
    )
    via_q = ((x > 1.0) & (x > a)) if lower else rest
    use_p_series = rest & (~via_q | q_by_p_series)
    use_cf = rest & via_q & ~q_by_p_series & (x > 1.1)
    use_q_series = rest & via_q & ~q_by_p_series & (x <= 1.1)

    # Each branch runs on its own elements only, gathered: evaluated
    # everywhere under a mask, the continued fraction alone cost every
    # element ~15 ops per step for as long as the slowest lane took.
    flat_a, flat_x = a.reshape(-1), x.reshape(-1)
    out = lucid.zeros_like(flat_a)
    branches: list[tuple[Tensor, Callable[[Tensor, Tensor], Tensor], bool]] = [
        (use_p_series, lambda u, v: _igam_series(u, v, k), lower),
        (use_cf, lambda u, v: _igamc_continued_fraction(u, v, k), not lower),
        (use_q_series, lambda u, v: _igamc_series(u, v, k), not lower),
        (asym, lambda u, v: _igam_asymptotic(u, v, lower=lower), True),
    ]
    for mask, branch, as_is in branches:
        flat_mask = mask.reshape(-1)
        if not _any_true(flat_mask):
            continue
        value = branch(flat_a[flat_mask], flat_x[flat_mask])
        out[flat_mask] = value if as_is else 1.0 - value
    return out.reshape(a.shape)


def _igam_edges(a: Tensor, x: Tensor, out: Tensor, *, lower: bool) -> Tensor:
    """The edges of the domain, applied lowest precedence first."""
    nan = lucid.full_like(out, math.nan)
    at_x_inf, at_a_inf = (1.0, 0.0) if lower else (0.0, 1.0)
    out = lucid.where(lucid.isinf(x), lucid.full_like(out, at_x_inf), out)
    a_inf = lucid.where(lucid.isinf(x), nan, lucid.full_like(out, at_a_inf))
    out = lucid.where(lucid.isinf(a), a_inf, out)
    out = lucid.where(x == 0.0, lucid.full_like(out, 0.0 if lower else 1.0), out)
    a_zero = lucid.where(x > 0.0, lucid.full_like(out, 1.0 if lower else 0.0), nan)
    out = lucid.where(a == 0.0, a_zero, out)
    outside = (x < 0.0) | (a < 0.0) | lucid.isnan(a) | lucid.isnan(x)
    return lucid.where(outside, nan, out)


@final
class _RegularizedGamma(Function):
    """P(a, x) or Q(a, x); differentiable in ``x`` only, as the reference is."""

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, a: Tensor, x: Tensor, lower: bool) -> Tensor:
        ctx.save_for_backward(a, x)
        ctx.lower = lower
        a_d, x_d = a.detach(), x.detach()
        inner = _igam_interior(a_d, x_d, lower=lower)
        return _igam_edges(a_d, x_d, inner, lower=lower)

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[None, Tensor, None]:
        name = "gammainc" if ctx.lower else "gammaincc"
        if ctx.needs_input_grad[0]:
            raise NotImplementedError(
                f"lucid.special.{name}: the derivative with respect to `input` "
                "(the shape a) is not implemented; only `other` (x) is "
                "differentiable"
            )
        a, x = ctx.saved_tensors
        density = lucid.exp((a - 1.0) * lucid.log(x) - x - lucid.lgamma(a))
        return None, grad * (density if ctx.lower else -density), None


def _regularized_gamma(a: Tensor, x: Tensor, *, lower: bool) -> Tensor:
    """P(a, x) (``lower``) or Q(a, x), broadcast and promoted together.

    An integer argument is promoted to the default float (:func:`_real`), and
    a 16-bit pair is evaluated in float32 (:func:`_in_float32`): the scalar
    algorithm's constants — ``2**24`` to rescale the continued fraction,
    ``exp`` of ``a`` and ``x`` — are outside float16's range, and
    ``gammainc`` answered NaN there.
    """

    def evaluate(a: Tensor, x: Tensor) -> Tensor:
        a, x = a + lucid.zeros_like(x), x + lucid.zeros_like(a)
        out = _RegularizedGamma.apply(a, x, lower)
        assert isinstance(out, Tensor)
        return out

    return _in_float32(evaluate, _real(a), _real(x))


def gammainc(input: Tensor, other: Tensor) -> Tensor:
    r"""Regularized lower incomplete gamma function :math:`P(a, x)`.

    .. math::

        P(a, x) = \frac{1}{\Gamma(a)} \int_0^x t^{a-1} e^{-t}\, dt

    The CDF of a ``Gamma(a, 1)`` variable at ``x`` — what
    :meth:`lucid.distributions.Gamma.cdf` evaluates.

    Parameters
    ----------
    input : Tensor
        Shape :math:`a \geq 0`.
    other : Tensor
        Upper limit :math:`x \geq 0`; broadcast against ``input``.

    Returns
    -------
    Tensor
        :math:`P(a, x) \in [0, 1]`, in the promoted dtype of the two.

    Raises
    ------
    NotImplementedError
        From ``backward``, when ``input`` requires a gradient: only
        ``other`` is differentiable, with
        :math:`\partial P / \partial x = x^{a-1} e^{-x} / \Gamma(a)`.

    Notes
    -----
    Evaluated by the algorithm of Cephes' ``igam`` as SciPy refines it: the
    power series below :math:`x \approx a`, ``1 - Q`` by the continued
    fraction above, and Temme's uniform asymptotic expansion where ``a`` is
    large and ``x`` near it (:math:`20 < a < 200` with
    :math:`|x - a|/a < 0.3`, or :math:`a > 200` with
    :math:`|x - a|/a < 4.5/\sqrt{a}`).  The edges follow the reference
    framework: :math:`P(a, 0) = 0`, :math:`P(a, \infty) = 1`,
    :math:`P(0, x) = 1` for :math:`x > 0`, :math:`P(\infty, x) = 0` for a
    finite ``x``; a negative argument, :math:`P(0, 0)`,
    :math:`P(\infty, \infty)` and NaN answer NaN.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import gammainc
    >>> gammainc(lucid.tensor([1.0, 2.0, 3.0]), lucid.tensor([1.0, 2.0, 3.0]))
    tensor([0.6321, 0.594, 0.5768])
    """
    refuse_common("gammainc", input, other)
    return _regularized_gamma(input, other, lower=True)


def gammaincc(input: Tensor, other: Tensor) -> Tensor:
    r"""Regularized upper incomplete gamma function :math:`Q(a, x)`.

    .. math::

        Q(a, x) = \frac{1}{\Gamma(a)} \int_x^\infty t^{a-1} e^{-t}\, dt
                = 1 - P(a, x)

    Evaluated directly rather than as ``1 - gammainc``, so a small upper
    tail keeps its relative precision.

    Parameters
    ----------
    input : Tensor
        Shape :math:`a \geq 0`.
    other : Tensor
        Lower limit :math:`x \geq 0`; broadcast against ``input``.

    Returns
    -------
    Tensor
        :math:`Q(a, x) \in [0, 1]`, in the promoted dtype of the two.

    Raises
    ------
    NotImplementedError
        From ``backward``, when ``input`` requires a gradient: only
        ``other`` is differentiable, with
        :math:`\partial Q / \partial x = -x^{a-1} e^{-x} / \Gamma(a)`.

    Notes
    -----
    The branches and edge values mirror :func:`gammainc`'s:
    :math:`Q(a, 0) = 1`, :math:`Q(a, \infty) = 0`, :math:`Q(0, x) = 0` for
    :math:`x > 0`, :math:`Q(\infty, x) = 1` for a finite ``x``.  For a tiny
    shape the reference framework evaluates :math:`\ln\Gamma(1 + a)` after
    rounding :math:`1 + a` to 1 and loses the Euler-Mascheroni term: at
    :math:`a = 10^{-30}` its value is 4.4% above the one mpmath computes at
    :math:`x = 10^{-6}`, 103% above at :math:`x = 0.5` and 286% above at
    :math:`x = 1.05`.  Here that term is kept, and the value agrees with
    mpmath's to float64 rounding.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import gammaincc
    >>> gammaincc(lucid.tensor([1.0, 2.0]), lucid.tensor([1.0, 10.0]))
    tensor([0.3679, 0.0004994])
    """
    refuse_common("gammaincc", input, other)
    return _regularized_gamma(input, other, lower=False)


# Documented and listed under ``lucid.special``, the one path they are
# reached by (H8), like the functions defined there.
gammainc.__module__ = "lucid.special"
gammaincc.__module__ = "lucid.special"
