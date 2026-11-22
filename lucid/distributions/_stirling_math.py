"""The range-safe pieces of the Gamma / Beta / Dirichlet family: Stirling remainders and ``bd0``.

``lgamma`` of a large argument is large (``1.3e7`` at ``1e6``), so any formula that
subtracts one from another large term, or from the main terms of a density, answers the
rounding of its inputs rather than the quantity::

    Beta(1e6, 1e6).log_prob(0.5)  = (a-1) log x + (b-1) log(1-x) - [lg(a) + lg(b) - lg(a+b)]
                                  = -1.4e6 - (-1.4e6) = 7.03          (float32: 9.0)
    Gamma(1e6, 1).entropy()       = a + lg(a) + (1-a) psi(a) = 8.33   (float32: 7.0)

The pieces below take the large parts out *before* any subtraction.  With
``L(z) = (z - 1/2) log z - z + log(2 pi)/2`` the Stirling remainder is

    S(z) = lgamma(z) - L(z)  =  1/(12 z) - 1/(360 z^3) + ...           (:func:`_stirling_remainder`)
    D(z) = log z - 1/(2 z) - digamma(z)  =  1/(12 z^2) - ...           (:func:`_digamma_remainder`)

and every identity below is exact for *all* ``z > 0`` (a large argument reads the
asymptotic series, a small one ``lgamma``/``digamma`` where nothing is large).  They give

* ``log B(alpha) = sum a_i log(a_i / s) + R(alpha)`` with the centred remainder
  ``R = -1/2 sum log(a_i / s) + (K-1)/2 log(2 pi / s) + sum S(a_i) - S(s)``, ``s = sum a_i``;
* the density of a Beta / Dirichlet / F as ``-sum bd0(a_i, s x_i) - sum log x_i - R`` where
  ``bd0(x, m) = x log(x / m) + m - x`` (Loader 2000) is the deviance kernel: it is the whole
  of the ``a log(x/p) + b log((1-x)/q)`` that cancels the ``a log p + b log q`` inside
  ``log B``, so a value at the mode reads ``R`` and nothing else;
* the entropy of a Dirichlet, ``(K-1)/2 log(2 pi e / s) + 1/2 sum log(a_i / s) + sum G(a_i) -
  G(s) + (K-1)(1/(2 s) + D(s))`` with the Gamma entropy's tail ``G(z) = -1/(2z) + S(z) +
  (z-1) D(z)`` (``H_Gamma(a) = 1/2 log(2 pi e a) + G(a)``).

Below ``_CENTRED_FROM`` the plain ``lgamma`` / ``digamma`` form is kept, bit for bit (every term
is small there).  The threshold is where the two errors cross, measured on ``log B`` against a
90-digit oracle over ``a, b`` in ``[0.05, 20]``: the plain form's worst error in epsilons is 9 at
``s = a + b`` in ``[10, 15)``, 18-35 in ``[15, 20)`` and 47-55 in ``[20, 40)``, the centred form's
4-6 from ``s = 10`` on (it is a little worse below, 10-26).  The result is a ``where`` of the two
on a per-lane mask; each branch is evaluated on a domain held inside its own, so a lane the other
branch owns cannot put a NaN into a gradient.

The entropies have a second threshold, lower than ``_CENTRED_FROM``: ``G(z)`` and ``D(z)`` are read
through the exact recurrences (:func:`_gamma_tail_recurred`, :func:`_digamma_recurred`) from
``z = _SHIFT_FROM``, and a Dirichlet / Beta entropy takes the centred form from ``sum(alpha) >=
_ENTROPY_FROM``.  The plain ``lgamma`` / ``digamma`` entropy and its slope ``1 + (1 - z) psi_1(z)``
subtract numbers of size 1 for an answer of ``1/(2z)``, which on ``1 <= z < 10`` is 22 (value) and
55 (slope) float32 epsilons off, and on one Metal GPU 36 at ``z = 9`` (LCD-1053).

Cost (LCD-1050).  Both branches run on every lane (a data-dependent branch would sync a device and
break tracing), so what is evaluated twice is the price of a lane the other branch owns.  Four rules
keep it near the plain form's:

* the ``K`` components of a vector lead, ``(K, *batch)``, and every per-category op runs once over
  them (:func:`_components`): the op count does not depend on ``K``;
* one ``lgamma`` pass over the components and the total (``_Stirling._small_lgamma``) serves the
  small-lane remainder of a centred lane and the plain log-beta of a plain lane; the entropy shares a
  ``lgamma`` and a ``digamma`` pass the same way;
* the series is read alone where the argument is known to be at its start (a total of a centred
  lane, ``Gamma`` / ``Poisson`` at ``alpha, k >= 10``), and ``lgamma`` is never formed to be
  discarded;
* ``bd0`` and the Stirling series are ``Function`` s with their exact slope (``log1p((x - m)/m)``,
  ``-D(z)``) instead of a taped graph of both lanes.

Entry points (one per quantity, each also used by every law that needs it):
:func:`_log_beta`, :func:`_log_multi_beta`, :func:`_log_comb`, :func:`_log_multinomial_coeff`;
:func:`_beta_log_prob`, :func:`_dirichlet_log_prob`, :func:`_f_log_prob`, :func:`_gamma_log_prob`,
:func:`_inverse_gamma_log_prob`, :func:`_poisson_log_prob`, :func:`_binomial_pmf_centred`,
:func:`_negative_binomial_pmf_centred`; :func:`_beta_entropy`, :func:`_dirichlet_entropy`,
:func:`_gamma_entropy`, :func:`_inverse_gamma_entropy`; :func:`_gamma_kl`.
"""

import math
from typing import final, override

import lucid
from lucid._ops.composite._gamma import _digamma_positive, _horner_float, _lgamma_positive
from lucid._ops.composite._opmath import _in_float32
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx
from lucid.distributions._util import _align_device
from lucid.special import xlog1py as _xlog1py

_FLOATS = (lucid.float16, lucid.bfloat16, lucid.float32, lucid.float64)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_LOG_2PI = math.log(2.0 * math.pi)

# Bernoulli numbers B_2k, k = 1..10.
_B2K: tuple[float, ...] = (
    1 / 6,
    -1 / 30,
    1 / 42,
    -1 / 30,
    5 / 66,
    -691 / 2730,
    7 / 6,
    -3617 / 510,
    43867 / 798,
    -174611 / 330,
)
# S(z) = sum_k _S_COEFFS[k-1] / z^(2k-1), D(z) = sum_k _D_COEFFS[k-1] / z^(2k): Stirling's series,
# and its derivative's (S' = -D).
_S_COEFFS: tuple[float, ...] = tuple(b / (2 * k * (2 * k - 1)) for k, b in enumerate(_B2K, 1))
_D_COEFFS: tuple[float, ...] = tuple(b / (2 * k) for k, b in enumerate(_B2K, 1))

# The series is asymptotic; from ``z = 10`` its truncation error relative to its value is
# below 1e-10 (4 terms) in float32 and 3e-17 (10 terms) in float64.  Below that ``lgamma`` and
# ``digamma`` are read: ``S`` and ``D`` are O(1) there, so the subtraction costs nothing that
# the caller's own terms do not already cost.
_SERIES_FROM = 10.0
# The centred forms replace the plain ``lgamma`` ones for ``sum(alpha) >= _CENTRED_FROM``.
_CENTRED_FROM = 10.0
_TERMS_32, _TERMS_64 = 4, 10

# The entropy's pieces ``G(z)`` (the Gamma entropy's tail) and ``D(z)`` are read from the series at
# ``z + k`` for ``z >= _SHIFT_FROM``, ``k`` the whole steps up to ``_RECUR_TO_*`` (see
# :func:`_gamma_tail_recurred`): float32's four terms are good to 6e-9 relative from 6, float64
# keeps its ten from 10.  Below ``_SHIFT_FROM`` (and for a non-finite or non-positive argument, with
# its inf / NaN as before) the plain ``lgamma`` / ``digamma`` form is kept.  On ``1 <= z < 10``
# the plain Gamma entropy is off by up to 22 / 27 epsilons in value and 55 / 35 in slope against a
# 60-digit oracle (float32 / float64): its slope ``1 + (1 - z) psi_1(z)`` subtracts two numbers
# near 1 for an answer of ``1/(2z)``, and one Metal GPU's ``psi_1`` is 2.4 ulp off at 9, which is 36
# epsilons there.
_SHIFT_FROM = 1.0
_RECUR_TO_32, _RECUR_TO_64 = 6.0, 10.0
# The Dirichlet / Beta entropy reads the centred form (and so ``G``, ``D``) from ``sum(alpha) >=
# _ENTROPY_FROM``: from 4 its worst error is 9 epsilons in value and 26 in slope near a root, the
# plain form's 55 and 114 for ``sum(alpha)`` in [7, 10]; below 4 they cross.
_ENTROPY_FROM = 4.0

# ``bd0``: the series in ``v = (x - m) / (x + m)`` below this ``|v|`` (the plain form loses
# ``1/|v|`` of its digits, the series needs ``|v|^(2 terms)`` below the epsilon).
_BD0_V = 0.3
_BD0_COEFFS: tuple[float, ...] = tuple(1.0 / (2 * j + 3) for j in range(24))  # v^(2j+3)/(2j+3)
_BD0_TERMS_32, _BD0_TERMS_64 = 7, 18

# The centred density of a Beta / Dirichlet / F is used where every ``a_i`` is within
# ``|a_i - s x_i| < _NEAR_V (a_i + s x_i)`` of its mean count: there the plain form's large terms
# cancel to a small answer.  Farther out the answer is as large as they are (``s KL``), the plain
# form is accurate to a few ulp, and its slope in ``x`` is a single term: the centred slope adds
# ``a/x - 1/x``, which for ``a = 1`` and a tiny ``x`` loses the ``O(s)`` remainder to rounding.
_NEAR_V = 0.5


def _is_double(x: Tensor) -> bool:
    return x.dtype == lucid.float64


def _stirling_series_value(z: Tensor) -> Tensor:
    r = 1.0 / z
    return r * _horner_float(_S_COEFFS[: _TERMS_64 if _is_double(z) else _TERMS_32], r * r)


def _digamma_series(z: Tensor) -> Tensor:
    """``D(z)`` from its asymptotic series, for ``z`` at or above the dtype's series start."""
    r = 1.0 / z
    return r * r * _horner_float(_D_COEFFS[: _TERMS_64 if _is_double(z) else _TERMS_32], r * r)


@final
class _StirlingSeries(Function):
    """``S(z)`` from its series with the slope ``-D(z)``: the derivative of the *same* truncation.

    ``S_T' = -D_T`` term by term (``(2k-1) c_k = B_2k / 2k``), so the slope is one Horner pass of
    ``D``'s series; differentiating the value's graph instead tapes every step of its own.
    """

    _trace_form = staticmethod(_stirling_series_value)

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, z: Tensor) -> Tensor:
        ctx.save_for_backward(z)
        return _stirling_series_value(z.detach())

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> Tensor:
        (z,) = ctx.saved_tensors
        return -(grad * _digamma_series(z))


def _stirling_series(z: Tensor) -> Tensor:
    """``S(z)`` from its asymptotic series, for ``z`` at or above the dtype's series start."""
    out = _StirlingSeries.apply(z)
    assert isinstance(out, Tensor)
    return out


def _stirling_plain(z: Tensor) -> Tensor:
    """``S(z)`` as ``lgamma`` minus the closed part: both are O(10) for ``z < _SERIES_FROM``."""
    return _lgamma_positive(z) - ((z - 0.5) * lucid.log(z) - z + _HALF_LOG_2PI)


def _digamma_plain(z: Tensor) -> Tensor:
    """``D(z)`` as the closed part minus ``digamma``: both are O(1) for ``z < _SERIES_FROM``."""
    return lucid.log(z) - 0.5 / z - _digamma_positive(z)


def _stirling_remainder_impl(z: Tensor) -> Tensor:
    far = z >= _SERIES_FROM
    # Each branch runs on its own domain: a lane of the other is held at the switch.
    series = _stirling_series(lucid.where(far, z, lucid.full_like(z, _SERIES_FROM)))
    plain = _stirling_plain(lucid.where(far, lucid.full_like(z, _SERIES_FROM), z))
    return lucid.where(far, series, plain)


def _stirling_remainder(z: Tensor) -> Tensor:
    r"""``lgamma(z) - [(z - 1/2) log z - z + log(2 pi) / 2]``, small for a large ``z``.

    :math:`S(z) \to 1/(12 z)`.  The ``lgamma`` of a large argument is never formed: from
    ``z = 10`` the asymptotic series in ``1/z`` is read, below it the difference of ``lgamma``
    and the closed part (both O(10) there).  An argument of 0 or below is out of its domain.
    """
    return _in_float32(_stirling_remainder_impl, z)


def _climb(z: Tensor) -> tuple[Tensor, Tensor, int]:
    """``(k, b, steps)``: ``z`` raised by ``k`` whole steps to ``b``, where the series is good.

    ``steps`` is the most a lane in ``[_SHIFT_FROM, ...)`` can need; a lane already at the series'
    start has ``k = 0`` and ``b = z``.  ``k`` is a constant of the graph (no gradient).
    """
    start = _RECUR_TO_64 if _is_double(z) else _RECUR_TO_32
    steps = math.ceil(start - _SHIFT_FROM)
    k = lucid.clamp(lucid.ceil(start - z), min=0.0, max=float(steps)).detach()
    return k, z + k, steps


def _digamma_recurred(z: Tensor) -> Tensor:
    r"""``D(z)`` for ``z >= _SHIFT_FROM`` through the exact recurrence, with no ``digamma``.

    With ``b = z + k`` (``k`` whole steps, ``b`` in the series' range) ``D(z) = D(b) - k / (2 b z) +
    sum_{j<k} [1/(z + j) - log(1 + 1/(z + j))]``: the plain ``log z - 1/(2z) - digamma(z)`` is a
    difference of numbers near 2 for a result near ``1/(12 z^2)``, and its slope ``1/z + 1/(2 z^2) -
    psi_1(z)`` one of numbers near 0.1 for ``1/(6 z^3)``.  A step's ``u - log(1 + u)``, ``u =
    1/(z + j) <= 1``, keeps most of its digits and so does its slope ``-u^2 / (1 + u)``; the
    ``k``-multiplied terms vanish with ``k``, so a ``z`` at the series' start reads it unchanged.
    """
    k, b, steps = _climb(z)
    gap = lucid.zeros_like(z)
    for j in range(steps):
        u = 1.0 / (z + float(j))
        gap = gap + lucid.where(k > float(j), u - lucid.log1p(u), 0.0)
    return _digamma_series(b) - 0.5 * k / (b * z) + gap


def _digamma_remainder_impl(z: Tensor) -> Tensor:
    stable = z >= _SHIFT_FROM
    # Each branch runs on its own domain: a lane of the other is held inside it.
    recurred = _digamma_recurred(lucid.where(stable, z, lucid.full_like(z, _SERIES_FROM)))
    plain = _digamma_plain(lucid.where(stable, lucid.ones_like(z), z))
    return lucid.where(stable, recurred, plain)


def _digamma_remainder(z: Tensor) -> Tensor:
    r"""``log(z) - 1/(2z) - digamma(z)``, small for a large ``z``: :math:`D(z) \to 1/(12 z^2)`.

    The derivative companion of :func:`_stirling_remainder` (``S' = -D``).
    """
    return _in_float32(_digamma_remainder_impl, z)


def _gamma_tail_recurred(z: Tensor) -> Tensor:
    r"""``G(z)`` for ``z >= _SHIFT_FROM`` through the exact recurrences, with no ``lgamma`` / ``digamma``.

    With ``b = z + k`` (``k`` whole steps, ``b`` in the series' range), ``lgamma(z) = lgamma(b) -
    sum_j log(z + j)`` and ``digamma(z) = digamma(b) - sum_j 1/(z + j)`` turn the Gamma entropy's
    ``H(z) = z + lgamma(z) + (1 - z) digamma(z)`` into

    .. math::
        G(z) = G(b) + \tfrac12\log\bigl(1 + \tfrac{k}{z}\bigr) - k\bigl(\tfrac{1}{2b} + D(b)\bigr)
               + \sum_{j<k}\Bigl[\log\bigl(1 + \tfrac{k-j}{z+j}\bigr) - \tfrac{j+1}{z+j}\Bigr],

    where ``G(b)`` and ``D(b)`` are read from the series and no logarithm of a large number is
    formed.  The slope ``H'(z) = 1 - (z - 1) psi_1(z)`` of the closed form subtracts two numbers
    near 1 for an answer of ``1/(2z)`` (0.06 at ``z = 9``), and the gradient through ``lgamma`` and
    ``digamma`` adds ``1 + psi - psi`` besides.  Here no ``digamma`` is called and each term of the
    slope is a fraction of ``1/z``.  A ``z`` at the series' start has ``k = 0``: every step is dead
    and ``G(z)`` is read from the series unchanged.
    """
    k, b, steps = _climb(z)
    gap = lucid.zeros_like(z)
    for j in range(steps):
        # A dead step is the constant 0: it adds nothing to the value or to the gradient.
        r = 1.0 / (z + float(j))
        step = lucid.log1p((k - float(j)) * r) - (j + 1.0) * r
        gap = gap + lucid.where(k > float(j), step, 0.0)
    # ``G(b) - k (1/(2b) + D(b))`` with ``G(b) = -1/(2b) + S(b) + (b - 1) D(b)`` and ``b - 1 - k = z - 1``.
    return (
        -0.5 * (1.0 + k) / b
        + _stirling_series(b)
        + (z - 1.0) * _digamma_series(b)
        + 0.5 * lucid.log1p(k / z)
        + gap
    )


def _gamma_tail_small(lg: Tensor, psi: Tensor, z: Tensor) -> Tensor:
    """``G(z)`` from the ``lgamma`` and ``digamma`` of ``z`` the caller holds: O(1/z) terms for ``z < 1``."""
    return -0.5 / z + _stirling_small(lg, z) + (z - 1.0) * (lucid.log(z) - 0.5 / z - psi)


def _bd0_value(x: Tensor, m: Tensor) -> Tensor:
    """``bd0`` for positive finite ``x`` and ``m``: both lanes are finite there, so no lane needs holding."""
    terms = _BD0_TERMS_64 if _is_double(x) else _BD0_TERMS_32
    gap = x - m
    v = gap / (x + m)
    v2 = v * v
    # Series lane: (x - m) v + 2 x sum_{j>=1} v^(2j+1)/(2j+1), every term of one sign; for |v| < 1 the
    # polynomial is finite.  Direct lane: finite wherever x, m > 0 (a lane of the other costs no NaN).
    series = gap * v + 2.0 * x * v * v2 * _horner_float(_BD0_COEFFS[:terms], v2)
    direct = x * lucid.log(x / m) + m - x
    return lucid.where(v.abs() < _BD0_V, series, direct)


@final
class _Bd0(Function):
    """``bd0`` with its exact slopes ``log(x/m)`` and ``1 - x/m``, each read as ``t = (x - m)/m``.

    ``1 - x/m = -t`` keeps every digit near ``x = m`` (``x - m`` is exact there), and so does
    ``log(x/m) = log1p(t)`` on the lanes of the series; farther out ``t`` rounds to -1 as ``x/m``
    underflows and the plain ``log(x/m)`` is the accurate one.  A handful of ops where the graph of
    the two lanes tapes some sixty.
    """

    _trace_form = staticmethod(_bd0_value)

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, m: Tensor) -> Tensor:
        ctx.save_for_backward(x, m)
        return _bd0_value(x.detach(), m.detach())

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor | None, Tensor | None]:
        x, m = ctx.saved_tensors
        gap = x - m
        t = gap / m
        need_x, need_m = ctx.needs_input_grad
        slope = None
        if need_x:
            # Each lane reads the form of its own domain: the other is held where it stays finite.
            near = (gap / (x + m)).abs() < _BD0_V
            slope = lucid.where(
                near,
                lucid.log1p(lucid.where(near, t, 0.0)),
                lucid.log(lucid.where(near, 1.0, x / m)),
            )
        return (None if slope is None else grad * slope, -(grad * t) if need_m else None)


def _bd0_impl(x: Tensor, m: Tensor) -> Tensor:
    out = _Bd0.apply(x, m)
    assert isinstance(out, Tensor)
    return out


def _bd0(x: Tensor, m: Tensor) -> Tensor:
    r"""``x log(x / m) + m - x`` (Loader's deviance kernel), without its cancellation near ``x = m``.

    Non-negative, 0 only at ``x = m``, and its slope in ``m`` is ``1 - x/m``.  Near ``x = m`` the
    plain form subtracts terms of size ``x`` to leave one of size ``(x - m)^2 / (x + m)``;
    with :math:`v = (x - m)/(x + m)` it is :math:`(x - m) v + 2 x \sum_{j \ge 1}
    v^{2j+1}/(2j+1)`, all terms positive.
    """
    return _in_float32(_bd0_impl, x, m)


def _common(*tensors: Tensor) -> tuple[Tensor, ...]:
    """The operands on one device, broadcast to one shape and promoted to one floating dtype.

    A 0-dim host tensor follows the operand it meets, as it does in every other op.
    """
    ref = next((t for t in tensors if t.ndim > 0 or t.device.type != "cpu"), tensors[0])
    views = lucid.broadcast_tensors(*(_align_device(t, ref) for t in tensors))
    dtype = views[0].dtype
    if len(views) > 1:
        dtype = lucid.result_type(views[0], views[1])
    for t in views[2:]:
        dtype = lucid.result_type(views[0].to(dtype), t)
    if dtype not in _FLOATS:
        dtype = lucid.get_default_dtype()
    return tuple(t if t.dtype == dtype else t.to(dtype) for t in views)


def _components(alpha: Tensor) -> Tensor:
    """The ``K`` components of a vector along its last axis, moved to the front: shape ``(K, *batch)``.

    Every quantity of the frame is elementwise in the components, so each runs once over the whole
    ``(K, *batch)`` tensor: the op count does not grow with ``K`` (on a device that pays per op, a
    second and a third category cost nothing but data).  The components lead because a ``stack`` or a
    reduction along a tiny *last* axis costs as much as an ``lgamma``.
    """
    return alpha.movedim(-1, 0)


def _pair(a: Tensor, b: Tensor) -> Tensor:
    """``(a, b)`` as ``(2, *batch)`` components."""
    return lucid.stack([a, b], dim=0)


def _log_shares(a: Tensor, s: Tensor) -> Tensor:
    """``log(a_i / s)`` for components ``(K, *batch)``; the largest (the first of equals) is ``log1p(-rest / s)``.

    ``a_i / s`` rounds to a relative ``eps``, which is an absolute ``eps`` in its logarithm; times
    ``a_i`` it is ``eps * s`` for the one share near 1.  ``rest``, the sum of the others, is
    exact where ``s - a_i`` is not.
    """
    k = a.shape[0]
    # The first of the largest: weights ``K - i`` make the earliest index the unique maximum of the hits.
    weight = (float(k) - lucid.arange(k, dtype=a.dtype, device=a.device)).reshape(
        k, *([1] * (a.ndim - 1))
    )
    hit = lucid.where(a.detach() == a.detach().amax(dim=0), weight, 0.0)
    dominant = hit == hit.amax(dim=0)
    rest = lucid.where(dominant, 0.0, a).sum(dim=0)
    return lucid.where(dominant, lucid.log1p(-rest / s), lucid.log(a / s))


def _stirling_small(lg: Tensor, z: Tensor) -> Tensor:
    """``S(z)`` from the ``lgamma`` of ``z`` the caller already holds: both terms are O(10) for ``z < 10``."""
    return lg - ((z - 0.5) * lucid.log(z) - z + _HALF_LOG_2PI)


class _Stirling:
    """The Stirling frame of one concentration vector given as components ``(K, *batch)``.

    ``large`` is the lane mask of the centred forms (a valid vector with ``sum >= centred_from``);
    ``a`` are the components held inside that domain (a dead lane holds ``_CENTRED_FROM / K`` each,
    so the sum is the series' start), so the centred pieces are finite on every lane.  The plain
    forms read the raw ``comps`` and ``total``.  Pieces that only some callers need are built on
    first use: a quantity nobody asks for is not formed (each costs a pass over the batch).
    """

    def __init__(self, comps: Tensor, centred_from: float = _CENTRED_FROM) -> None:
        self.comps: Tensor = comps
        self.k: int = comps.shape[0]
        self.total: Tensor = comps.sum(dim=0)
        valid = (comps.amin(dim=0) > 0.0) & lucid.isfinite(self.total)
        self.large: Tensor = valid & (self.total >= centred_from)
        self.a: Tensor = lucid.where(self.large, comps, _CENTRED_FROM / self.k)
        self.s: Tensor = self.a.sum(dim=0)
        self.shares: Tensor = _log_shares(self.a, self.s)
        self._lgamma: tuple[Tensor, Tensor] | None = None
        self._tail: Tensor | None = None

    def _small_lgamma(self) -> tuple[Tensor, Tensor]:
        """``(big, lgamma)``: ``lgamma`` of the ``K`` components and of the total, in one pass.

        A component of a centred lane that is ``>= _SERIES_FROM`` reads its remainder from the series
        and holds 1 here; every other component is its own raw value: a small one of a centred lane
        (whose remainder is ``lgamma - closed part``) and every one of a plain lane (whose log-beta
        is the plain ``lgamma`` sum) share the one pass.  The total of a plain lane is the last row
        (a centred lane holds 1: its remainder is the series).
        """
        if self._lgamma is None:
            big = self.large & (self.comps >= _SERIES_FROM)
            held = lucid.cat(
                [
                    lucid.where(big, 1.0, self.comps),
                    lucid.where(self.large, 1.0, self.total).unsqueeze(0),
                ],
                dim=0,
            )
            self._lgamma = (big, _lgamma_positive(held))
        return self._lgamma

    @property
    def tail(self) -> Tensor:
        """``R = log B(a) - sum a_i log(a_i / s)`` on the centred domain (batch shape)."""
        if self._tail is None:
            big, lg = self._small_lgamma()
            # The small lane's closed part is read at a held operand: 1 on a lane it does not own.
            held = lucid.where(self.large & ~big, self.comps, 1.0)
            remainders = lucid.where(
                big,
                _stirling_series(lucid.where(big, self.comps, _SERIES_FROM)),
                _stirling_small(lg[: self.k], held),
            )
            self._tail = (
                -0.5 * self.shares.sum(dim=0)
                + 0.5 * (self.k - 1) * (_LOG_2PI - lucid.log(self.s))
                + remainders.sum(dim=0)
                - _stirling_series(self.s)
            )
        return self._tail

    def log_beta(self) -> Tensor:
        """``sum lgamma(a_i) - lgamma(s)``: centred on the large lanes, plain elsewhere."""
        _, lg = self._small_lgamma()
        centred = (self.a * self.shares).sum(dim=0) + self.tail
        plain = lg[: self.k].sum(dim=0) - lg[self.k]
        return lucid.where(self.large, centred, plain)

    def deviance(self, x: Tensor) -> Tensor:
        """``sum_i bd0(a_i, s x_i)``: ``-`` this is ``sum a_i log(x_i / p_i)`` when ``sum x = 1``."""
        return _bd0_impl(self.a, self.s * x).sum(dim=0)

    def near(self, x: Tensor) -> Tensor:
        """Per lane: every ``x_i`` within ``_NEAR_V`` of its mean count ``a_i / s`` (see ``_NEAR_V``)."""
        m = self.s * x
        return ((self.a - m).abs() - _NEAR_V * (self.a + m)).amax(dim=0) <= 0.0


def _log_multi_beta_impl(alpha: Tensor) -> Tensor:
    return _Stirling(_components(alpha)).log_beta()


def _log_multi_beta(alpha: Tensor) -> Tensor:
    r"""``sum_i lgamma(alpha_i) - lgamma(sum_i alpha_i)`` along the last axis, without the cancellation.

    For ``sum alpha >= 10`` it is :math:`\sum_i \alpha_i \log(\alpha_i/s) + R`, the entropy-like
    main part (large, but one rounding from its own value) and the centred remainder ``R``
    (``O(log s)``); below it the plain form, where every term is small.  A vector outside
    ``alpha > 0`` and finite takes the plain form (and its NaN / inf), as before.
    """
    return _in_float32(_log_multi_beta_impl, alpha)


def _log_beta_impl(a: Tensor, b: Tensor) -> Tensor:
    return _Stirling(_pair(a, b)).log_beta()


def _log_beta(a: Tensor, b: Tensor) -> Tensor:
    """``lgamma(a) + lgamma(b) - lgamma(a + b)``: :func:`_log_multi_beta` of the pair."""
    a, b = _common(a, b)
    return _in_float32(_log_beta_impl, a, b)


def _log_comb(n: Tensor, k: Tensor) -> Tensor:
    r"""``lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)``, from ``n >= 8`` through the log-beta.

    :math:`\log\binom{n}{k} = -\log(n + 1) - \log B(k + 1, n - k + 1)`.  The ends ``k = 0`` and
    ``k = n`` are exactly 0.  Outside ``0 <= k <= n`` (or a non-finite ``n``) the plain
    ``lgamma`` form is kept.
    """
    n, k = _common(n, k)
    m = n - k
    valid = (k >= 0.0) & (m >= 0.0) & lucid.isfinite(n)
    big = valid & (n + 2.0 >= _CENTRED_FROM)
    ends = (k == 0.0) | (m == 0.0)
    zero = lucid.zeros_like(n)
    nc, kc, mc = (lucid.where(big, t, zero) for t in (n, k, m))
    centred = -lucid.log1p(nc) - _log_beta(kc + 1.0, mc + 1.0)
    one = lucid.ones_like(n)
    # One pass over the three counts.  A count + 1 below 0 is outside the support: the clamp keeps
    # its ``lgamma`` at the pole (+inf, so a log-mass of -inf) as the plain ``lgamma`` had it.
    held = lucid.stack([lucid.where(big, one, t) + 1.0 for t in (n, k, m)], dim=0)
    lg = _lgamma_positive(lucid.clamp(held, min=0.0))
    plain = lg[0] - lg[1] - lg[2]
    # The ends are exactly 0 in value (as ``lgamma(n + 1) - lgamma(1) - lgamma(n + 1)`` is) and keep the
    # slope of the function: ``centred - centred.detach()`` is 0.0 with the gradient of ``centred``.
    return lucid.where(big, lucid.where(ends, centred - centred.detach(), centred), plain)


def _entropy_of(frame: _Stirling) -> Tensor:
    """The Dirichlet entropy ``log B + (s - K) psi(s) - sum (a_i - 1) psi(a_i)`` of a frame's vector.

    A component of a centred lane that is ``>= _SHIFT_FROM`` reads ``G`` through the recurrence; every
    other one (a smaller component of a centred lane, every component of a plain lane) shares one
    ``lgamma`` and one ``digamma`` pass between ``G``'s small form and the plain entropy.  ``G(s)`` and
    ``D(s)`` of a centred lane are always ``>= _ENTROPY_FROM``: the recurrence alone, and ``G(s)``
    rides the components' recurrence as one more row.
    """
    k, k1, comps = frame.k, frame.k - 1.0, frame.comps
    rec = frame.large & (comps >= _SHIFT_FROM)
    ps = lucid.where(frame.large, 1.0, frame.total)
    held = lucid.cat([lucid.where(rec, 1.0, comps), ps.unsqueeze(0)], dim=0)
    lg, psi = _lgamma_positive(held), _digamma_positive(held)
    z = lucid.where(frame.large & ~rec, comps, 1.0)
    recurred = _gamma_tail_recurred(
        lucid.cat([lucid.where(rec, comps, _SERIES_FROM), frame.s.unsqueeze(0)], dim=0)
    )
    tails = lucid.where(rec, recurred[:k], _gamma_tail_small(lg[:k], psi[:k], z))
    centred = (
        0.5 * frame.shares.sum(dim=0)
        + 0.5 * k1 * (_LOG_2PI + 1.0 - lucid.log(frame.s))
        + tails.sum(dim=0)
        - recurred[k]
        + k1 * (0.5 / frame.s + _digamma_recurred(frame.s))
    )
    plain = lg[:k].sum(dim=0) - lg[k] + (ps - k) * psi[k] - ((comps - 1.0) * psi[:k]).sum(dim=0)
    return lucid.where(frame.large, centred, plain)


def _dirichlet_entropy_impl(alpha: Tensor) -> Tensor:
    return _entropy_of(_Stirling(_components(alpha), _ENTROPY_FROM))


def _dirichlet_entropy(alpha: Tensor) -> Tensor:
    r"""Entropy of ``Dirichlet(alpha)`` along the last axis, without the digamma cancellation.

    :math:`\log B(\alpha) + (s - K)\psi(s) - \sum_i (\alpha_i - 1)\psi(\alpha_i)` is a sum of
    :math:`\alpha\log\alpha`-sized terms; from ``s >= 4`` it is read from the centred
    form instead (see the module docstring) and the plain one is kept below.
    """
    return _in_float32(_dirichlet_entropy_impl, alpha)


def _beta_entropy_impl(a: Tensor, b: Tensor) -> Tensor:
    return _entropy_of(_Stirling(_pair(a, b), _ENTROPY_FROM))


def _beta_entropy(a: Tensor, b: Tensor) -> Tensor:
    """Entropy of ``Beta(a, b)``: :func:`_dirichlet_entropy` of the pair."""
    a, b = _common(a, b)
    return _in_float32(_beta_entropy_impl, a, b)


def _gamma_entropy_recurred(alpha: Tensor) -> Tensor:
    """``alpha + lgamma(alpha) + (1 - alpha) digamma(alpha)`` for ``alpha >= _SHIFT_FROM``."""
    return 0.5 * (_LOG_2PI + 1.0 + lucid.log(alpha)) + _gamma_tail_recurred(alpha)


def _gamma_entropy_impl(alpha: Tensor) -> Tensor:
    centred_lane = (alpha >= _SHIFT_FROM) & lucid.isfinite(alpha)
    big = lucid.where(centred_lane, alpha, lucid.full_like(alpha, _SERIES_FROM))
    small = lucid.where(centred_lane, lucid.ones_like(alpha), alpha)
    plain = small + _lgamma_positive(small) + (1.0 - small) * _digamma_positive(small)
    return lucid.where(centred_lane, _gamma_entropy_recurred(big), plain)


def _gamma_entropy(alpha: Tensor) -> Tensor:
    r"""``alpha + lgamma(alpha) + (1 - alpha) digamma(alpha)``: the entropy of ``Gamma(alpha, 1)``.

    For ``alpha >= 1`` it is :math:`\tfrac12\log(2\pi e\,\alpha) + G(\alpha)` with the tail
    ``G`` of :func:`_gamma_tail_recurred`; below, the plain form.  (``Gamma(1e6, 1)``: float32
    gave 7.0 for 8.3267.)  Its slope in ``alpha`` is ``1/(2 alpha) + G'`` and not
    ``1 - (alpha - 1) psi_1(alpha)``, a difference of two numbers near 1 (and, through ``lgamma`` and
    ``digamma``, ``1 + psi - psi`` besides): that cost up to 55 float32 epsilons from ``alpha = 5`` to
    10, and 36 at 9 on a Metal GPU whose ``psi_1`` is 2.4 ulp off.
    """
    return _in_float32(_gamma_entropy_impl, alpha)


def _beta_log_prob_impl(a: Tensor, b: Tensor, x: Tensor) -> Tensor:
    frame = _Stirling(_pair(a, b))
    y = 1.0 - x
    xy = _pair(x, y)
    # The centred form takes the interior of the support; a boundary point is a different
    # (xlogy) formula, which sees the stable log B.
    inside = frame.large & (x > 0.0) & (y > 0.0) & frame.near(xy)
    xyc = lucid.where(inside, xy, 0.5)
    centred = -frame.deviance(xyc) - lucid.log(xyc).sum(dim=0) - frame.tail
    plain = lucid.xlogy(a - 1.0, x) + _xlog1py(b - 1.0, -x) - frame.log_beta()
    return lucid.where(inside, centred, plain)


def _beta_log_prob(a: Tensor, b: Tensor, x: Tensor) -> Tensor:
    r"""``(a - 1) log x + (b - 1) log(1 - x) - log B(a, b)`` with the large parts taken out.

    For ``a + b >= 10`` and ``0 < x < 1``: :math:`-\mathrm{bd0}(a, s x) - \mathrm{bd0}(b, s(1-x))
    - \log x - \log(1 - x) - R(a, b)`` (``s = a + b``); the deviance terms are 0 at the mean
    ``x = a/s`` and the rest is ``O(log s)``, so ``Beta(1e6, 1e6)`` at 0.5 reads 7.0285 in
    float32 where the plain form read 9.  The pair ``x``, ``1 - x`` is taken as complementary.
    """
    a, b, x = _common(a, b, x)
    return _in_float32(_beta_log_prob_impl, a, b, x)


def _dirichlet_log_prob_impl(alpha: Tensor, x: Tensor) -> Tensor:
    frame = _Stirling(_components(alpha))
    xs = _components(x)
    inside = frame.large & (xs.amin(dim=0) > 0.0) & lucid.isfinite(xs.sum(dim=0)) & frame.near(xs)
    xc = lucid.where(inside, xs, 1.0 / frame.k)
    # sum a_i log(x_i / p_i) = -sum bd0(a_i, s x_i) + s (sum x_i - 1); the last term is 0 on the
    # simplex, and keeps the slope off it the plain formula's.
    centred = (
        -frame.deviance(xc)
        + frame.s * (xc.sum(dim=0) - 1.0)
        - lucid.log(xc).sum(dim=0)
        - frame.tail
    )
    plain = lucid.xlogy(frame.comps - 1.0, xs).sum(dim=0) - frame.log_beta()
    return lucid.where(inside, centred, plain)


def _dirichlet_log_prob(alpha: Tensor, x: Tensor) -> Tensor:
    r"""``sum_i (alpha_i - 1) log x_i - log B(alpha)``, the same way as :func:`_beta_log_prob`."""
    alpha, x = _common(alpha, x)
    return _in_float32(_dirichlet_log_prob_impl, alpha, x)


def _f_log_prob_impl(d1: Tensor, d2: Tensor, x: Tensor) -> Tensor:
    half1, half2 = d1 * 0.5, d2 * 0.5
    frame = _Stirling(_pair(half1, half2))
    lane = frame.large & (x > 0.0) & lucid.isfinite(x)
    # The centred pieces are formed from sanitised operands: an infinite ``x`` or ``d`` would meet
    # the zero cotangent of the ``where`` below as ``inf * 0``.
    xs = lucid.where(lane, x, 1.0)
    c1, c2 = 2.0 * frame.a[0], 2.0 * frame.a[1]
    w = c1 * xs + c2
    u, y = c1 * xs / w, c2 / w
    uy = _pair(u, y)
    inside = lane & (u > 0.0) & (y > 0.0) & frame.near(uy)
    xc = lucid.where(inside, xs, 0.5)
    uyc = lucid.where(inside, uy, 0.5)
    # log p(x) = log Beta(u; a, b) + log(d1 d2 / w^2), and -log u - log y + log(d1 d2 / w^2) is
    # -log x: the large logs cancel before they are formed.
    centred = -frame.deviance(uyc) - lucid.log(xc) - frame.tail
    ratio = d1 / d2
    plain = (
        half1 * lucid.log(ratio)
        + lucid.xlogy(half1 - 1.0, x)
        - (half1 + half2) * lucid.log1p(ratio * x)
        - frame.log_beta()
    )
    return lucid.where(inside, centred, plain)


def _f_log_prob(d1: Tensor, d2: Tensor, x: Tensor) -> Tensor:
    r"""The log-density of ``F(d1, d2)`` at ``x``: the Beta density at ``d1 x / (d1 x + d2)``.

    With :math:`u = d_1 x / (d_1 x + d_2)` the density is Beta(:math:`d_1/2, d_2/2`) at ``u``
    times :math:`du/dx`; the Jacobian's logarithms cancel against :math:`-\log u - \log(1 - u)`
    to :math:`-\log x`, so the centred form is :math:`-\mathrm{bd0}(a, s u) - \mathrm{bd0}(b, s
    (1 - u)) - \log x - R(a, b)`.
    """
    d1, d2, x = _common(d1, d2, x)
    return _in_float32(_f_log_prob_impl, d1, d2, x)


def _log_multinomial_coeff(counts: Tensor) -> Tensor:
    r"""``lgamma(n + 1) - sum_i lgamma(k_i + 1)`` along the last axis, ``n = sum k_i``.

    :math:`-\log B(k + 1) - \sum_{j=1}^{K-1}\log(n + j)` once ``n + K >= 10``; below, the plain
    form (bit for bit as before).  The last sum is ``lgamma(n + K) - lgamma(n + 1)``, formed as the
    sum of its ``K - 1`` logarithms.
    """
    k = counts.shape[-1]
    total = counts.sum(dim=-1)
    valid = (counts.amin(dim=-1) >= 0.0) & lucid.isfinite(total)
    big = valid & (total + k >= _CENTRED_FROM)
    cc = lucid.where(big.unsqueeze(-1), counts, lucid.zeros_like(counts))
    nc = cc.sum(dim=-1)
    centred = -_log_multi_beta(cc + 1.0)
    if k > 1:
        steps = lucid.arange(1, k, dtype=counts.dtype, device=counts.device)
        centred = centred - lucid.log(nc.unsqueeze(-1) + steps).sum(dim=-1)
    plain_counts = lucid.where(big.unsqueeze(-1), lucid.zeros_like(counts), counts)
    # A count + 1 below 0 stays at the pole (clamp), as in :func:`_log_comb`.
    whole = _lgamma_positive(lucid.clamp(plain_counts.sum(dim=-1) + 1.0, min=0.0))
    parts = _lgamma_positive(lucid.clamp(plain_counts + 1.0, min=0.0)).sum(dim=-1)
    plain = whole - parts
    return lucid.where(big, centred, plain)


def _saddle_lane(n: Tensor, k: Tensor, p_k: Tensor, p_rest: Tensor) -> Tensor:
    """The lanes of the saddle-point pmf: ``0 < k < n``, a finite ``n >= 10``, both probabilities > 0."""
    return (
        (k > 0.0)
        & (n - k > 0.0)
        & (n >= _CENTRED_FROM)
        & lucid.isfinite(n)
        & (p_k > 0.0)
        & (p_rest > 0.0)
    )


def _saddle_pmf_impl(n: Tensor, k: Tensor, p_k: Tensor, p_rest: Tensor) -> Tensor:
    inside = _saddle_lane(n, k, p_k, p_rest)
    base = lucid.full_like(n, 2.0 * _CENTRED_FROM)
    half = lucid.full_like(n, 0.5)
    nc, kc = lucid.where(inside, n, base), lucid.where(inside, k, 0.5 * base)
    pk, pr = lucid.where(inside, p_k, half), lucid.where(inside, p_rest, half)
    rest = nc - kc
    # Loader (2000): C(n, k) p^k q^(n-k) = exp(S(n) - S(k) - S(n-k) - bd0(k, n p) - bd0(n-k, n q))
    # / sqrt(2 pi k (n-k) / n), with S the Stirling remainder (``nc >= 10`` on every lane: the series).
    return (
        _stirling_series(nc)
        - _stirling_remainder_impl(kc)
        - _stirling_remainder_impl(rest)
        - _bd0_impl(kc, nc * pk)
        - _bd0_impl(rest, nc * pr)
        - 0.5 * (_LOG_2PI + lucid.log(kc) + lucid.log(rest) - lucid.log(nc))
    )


def _binomial_pmf_centred_impl(n: Tensor, k: Tensor, logits: Tensor) -> Tensor:
    return _saddle_pmf_impl(n, k, lucid.sigmoid(logits), lucid.sigmoid(-logits))


def _binomial_pmf_centred(n: Tensor, k: Tensor, logits: Tensor) -> tuple[Tensor, Tensor]:
    r"""``log P(X = k)`` of ``Binomial(n, logits)`` in the saddle-point form, and the lanes it covers.

    On ``0 < k < n`` with ``n >= 10`` and a finite logit, the log-coefficient and ``k log p + (n -
    k) log(1 - p)`` are not formed apart: :math:`\log\binom{n}{k} p^k q^{n-k} = S(n) - S(k) -
    S(n - k) - \mathrm{bd0}(k, n p) - \mathrm{bd0}(n - k, n q) - \tfrac12\log(2\pi k (n-k)/n)`.
    Elsewhere the mask is false and the value is not meaningful.
    """
    n, k, logits = _common(n, k, logits)
    lane = _saddle_lane(n, k, lucid.sigmoid(logits), lucid.sigmoid(-logits))
    return _in_float32(_binomial_pmf_centred_impl, n, k, logits), lane


def _negative_binomial_pmf_centred_impl(r: Tensor, k: Tensor, logits: Tensor) -> Tensor:
    p_k, p_rest = lucid.sigmoid(-logits), lucid.sigmoid(logits)
    inside = _saddle_lane(k + r, r, p_k, p_rest)
    safe_r = lucid.where(inside, r, lucid.ones_like(r))
    safe_n = lucid.where(inside, k + r, 2.0 * lucid.ones_like(r))
    value = _saddle_pmf_impl(k + r, r, p_k, p_rest)
    return value + lucid.log(safe_r) - lucid.log(safe_n)


def _negative_binomial_pmf_centred(r: Tensor, k: Tensor, logits: Tensor) -> tuple[Tensor, Tensor]:
    r"""``log P(X = k)`` of ``NegativeBinomial(r, logits)``, in the same form, and the lanes it covers.

    :math:`P(k) = \tfrac{r}{k + r}\binom{k + r}{r} q^r p^k` is a Binomial mass at ``r`` of ``k +
    r`` trials (``p = sigmoid(logits)`` the failure probability); on ``k > 0``, ``r > 0`` with
    ``k + r >= 10``.
    """
    r, k, logits = _common(r, k, logits)
    lane = _saddle_lane(k + r, r, lucid.sigmoid(-logits), lucid.sigmoid(logits))
    return _in_float32(_negative_binomial_pmf_centred_impl, r, k, logits), lane


def _gamma_family_log_prob(
    alpha: Tensor, rate: Tensor, x: Tensor, plain: Tensor, *, inverse: bool
) -> Tensor:
    """``-bd0(alpha, m) - log x + log(alpha / (2 pi)) / 2 - S(alpha)`` on its lanes, else ``plain``.

    ``m = rate * x`` (``rate / x`` for the inverse law).  Formed from operands held inside the
    lane, so an infinite ``x`` cannot meet the ``where``'s zero cotangent as ``inf * 0``.
    """
    inside = (
        (alpha >= _CENTRED_FROM)
        & lucid.isfinite(alpha)
        & (rate > 0.0)
        & lucid.isfinite(rate)
        & (x > 0.0)
        & lucid.isfinite(x)
    )
    ac = lucid.where(inside, alpha, lucid.full_like(alpha, _CENTRED_FROM))
    rc = lucid.where(inside, rate, lucid.ones_like(rate))
    xc = lucid.where(inside, x, lucid.ones_like(x))
    m = rc / xc if inverse else rc * xc
    centred = (
        -_bd0_impl(ac, m) - lucid.log(xc) + 0.5 * (lucid.log(ac) - _LOG_2PI) - _stirling_series(ac)
    )
    return lucid.where(inside, centred, plain)


def _gamma_log_prob_impl(alpha: Tensor, rate: Tensor, x: Tensor) -> Tensor:
    plain = (
        alpha * lucid.log(rate) + lucid.xlogy(alpha - 1.0, x) - rate * x - _lgamma_positive(alpha)
    )
    return _gamma_family_log_prob(alpha, rate, x, plain, inverse=False)


def _gamma_log_prob(alpha: Tensor, rate: Tensor, x: Tensor) -> Tensor:
    r"""``alpha log(rate) + (alpha - 1) log x - rate x - lgamma(alpha)``, without the cancellation.

    For ``alpha >= 10`` and ``x > 0`` the density is :math:`-\mathrm{bd0}(\alpha, \beta x) -
    \log x + \tfrac12\log(\alpha / 2\pi) - S(\alpha)`: the deviance is 0 at ``x = alpha / rate`` and the
    rest is ``O(log alpha)``, where the plain form is the difference of three numbers of size
    ``alpha log alpha`` (float32 ``Gamma(1e6, 1)`` at ``x = 1e6``: ``-6`` for ``-7.83``).
    """
    alpha, rate, x = _common(alpha, rate, x)
    return _in_float32(_gamma_log_prob_impl, alpha, rate, x)


def _inverse_gamma_log_prob_impl(alpha: Tensor, rate: Tensor, x: Tensor) -> Tensor:
    plain = (
        alpha * lucid.log(rate) - _lgamma_positive(alpha) - (alpha + 1.0) * lucid.log(x) - rate / x
    )
    return _gamma_family_log_prob(alpha, rate, x, plain, inverse=True)


def _inverse_gamma_log_prob(alpha: Tensor, rate: Tensor, x: Tensor) -> Tensor:
    r"""The log-density of ``InverseGamma(alpha, rate)``: :func:`_gamma_log_prob`'s form at ``rate / x``."""
    alpha, rate, x = _common(alpha, rate, x)
    return _in_float32(_inverse_gamma_log_prob_impl, alpha, rate, x)


def _poisson_log_prob_impl(rate: Tensor, k: Tensor) -> Tensor:
    plain = lucid.xlogy(k, rate) - rate - _lgamma_positive(lucid.clamp(k + 1.0, min=0.0))
    inside = (k >= _CENTRED_FROM) & lucid.isfinite(k) & (rate > 0.0) & lucid.isfinite(rate)
    base = lucid.full_like(k, _CENTRED_FROM)
    kc = lucid.where(inside, k, base)
    rc = lucid.where(inside, rate, base)
    centred = -_bd0_impl(kc, rc) - 0.5 * (_LOG_2PI + lucid.log(kc)) - _stirling_series(kc)
    return lucid.where(inside, centred, plain)


def _poisson_log_prob(rate: Tensor, k: Tensor) -> Tensor:
    r"""``k log(rate) - rate - lgamma(k + 1)``: for ``k >= 10`` it is :math:`-\mathrm{bd0}(k, \lambda)
    - \tfrac12\log(2\pi k) - S(k)`, with no term of size ``k log k``."""
    rate, k = _common(rate, k)
    return _in_float32(_poisson_log_prob_impl, rate, k)


def _inverse_gamma_entropy_impl(alpha: Tensor, rate: Tensor) -> Tensor:
    centred_lane = (alpha >= _SHIFT_FROM) & lucid.isfinite(alpha)
    big = lucid.where(centred_lane, alpha, lucid.full_like(alpha, _SERIES_FROM))
    small = lucid.where(centred_lane, lucid.ones_like(alpha), alpha)
    log_rate = lucid.log(rate)
    centred = log_rate + _gamma_entropy_recurred(big) - 2.0 * _digamma_positive(big)
    plain = small + log_rate + _lgamma_positive(small) - (small + 1.0) * _digamma_positive(small)
    return lucid.where(centred_lane, centred, plain)


def _inverse_gamma_entropy(alpha: Tensor, rate: Tensor) -> Tensor:
    r"""``alpha + log(rate) + lgamma(alpha) - (1 + alpha) digamma(alpha)``: the Gamma entropy's tail
    with ``-2 digamma(alpha)`` (the one term that does not cancel)."""
    alpha, rate = _common(alpha, rate)
    return _in_float32(_inverse_gamma_entropy_impl, alpha, rate)


def _gamma_kl_impl(a_p: Tensor, b_p: Tensor, a_q: Tensor, b_q: Tensor) -> Tensor:
    valid = (
        (a_p > 0.0)
        & (a_q > 0.0)
        & (b_p > 0.0)
        & (b_q > 0.0)
        & lucid.isfinite(a_p)
        & lucid.isfinite(a_q)
        & lucid.isfinite(b_p)
        & lucid.isfinite(b_q)
    )
    one = lucid.ones_like(a_p)
    ap, aq = lucid.where(valid, a_p, one), lucid.where(valid, a_q, one)
    bp, bq = lucid.where(valid, b_p, one), lucid.where(valid, b_q, one)
    rate_ratio = bq / bp
    # With S, D the Stirling remainders, the digamma and lgamma of the closed form regroup into
    # terms that are each 0 at ``p = q`` and positive-sized next to it (``a_q bd0(1, r)`` is
    # ``a_q (r - 1 - log r)``; ``bd0(a_q, a_p)`` the shape part of the same).
    exact = (
        aq * _bd0_impl(one, rate_ratio)
        + (ap - aq) * (rate_ratio - 1.0)
        + _bd0_impl(aq, ap)
        + _bd0_impl(ap, aq) / (2.0 * ap)
        + _stirling_remainder_impl(aq)
        - _stirling_remainder_impl(ap)
        - (ap - aq) * _digamma_remainder_impl(ap)
    )
    plain = (
        a_q * lucid.log(b_p / b_q)
        + _lgamma_positive(a_q)
        - _lgamma_positive(a_p)
        + (a_p - a_q) * _digamma_positive(a_p)
        + (b_q - b_p) * (a_p / b_p)
    )
    return lucid.where(valid, exact, plain)


def _gamma_kl(a_p: Tensor, b_p: Tensor, a_q: Tensor, b_q: Tensor) -> Tensor:
    r"""``KL(Gamma(a_p, b_p) || Gamma(a_q, b_q))`` without the cancellation of the closed form.

    The closed form ``a_q log(b_p/b_q) + lgamma(a_q) - lgamma(a_p) + (a_p - a_q) psi(a_p) + (b_q -
    b_p) a_p / b_p`` is a sum of terms of size ``a log a`` for an answer that is ``O((a_p - a_q)^2 /
    a)`` between neighbours.  It equals ``a_q bd0(1, r) + (a_p - a_q)(r - 1) + bd0(a_q, a_p) +
    bd0(a_p, a_q) / (2 a_p) + S(a_q) - S(a_p) - (a_p - a_q) D(a_p)`` with ``r = b_q / b_p``;
    parameters outside ``(0, inf)`` keep the closed form (and its NaN / inf).
    """
    a_p, b_p, a_q, b_q = _common(a_p, b_p, a_q, b_q)
    return _in_float32(_gamma_kl_impl, a_p, b_p, a_q, b_q)
