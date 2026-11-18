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

Entry points (one per quantity, each also used by every law that needs it):
:func:`_log_beta`, :func:`_log_multi_beta`, :func:`_log_comb`, :func:`_log_multinomial_coeff`;
:func:`_beta_log_prob`, :func:`_dirichlet_log_prob`, :func:`_f_log_prob`, :func:`_gamma_log_prob`,
:func:`_inverse_gamma_log_prob`, :func:`_poisson_log_prob`, :func:`_binomial_pmf_centred`,
:func:`_negative_binomial_pmf_centred`; :func:`_beta_entropy`, :func:`_dirichlet_entropy`,
:func:`_gamma_entropy`, :func:`_inverse_gamma_entropy`; :func:`_gamma_kl`.
"""

import math

import lucid
from lucid._ops.composite._gamma import _horner_float
from lucid._ops.composite._opmath import _in_float32
from lucid._tensor.tensor import Tensor
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


def _stirling_remainder_impl(z: Tensor) -> Tensor:
    terms = _TERMS_64 if _is_double(z) else _TERMS_32
    far = z >= _SERIES_FROM
    # Each branch runs on its own domain: a lane of the other is held at the switch.
    r = 1.0 / lucid.where(far, z, lucid.full_like(z, _SERIES_FROM))
    series = r * _horner_float(_S_COEFFS[:terms], r * r)
    near = lucid.where(far, lucid.full_like(z, _SERIES_FROM), z)
    plain = lucid.lgamma(near) - ((near - 0.5) * lucid.log(near) - near + _HALF_LOG_2PI)
    return lucid.where(far, series, plain)


def _stirling_remainder(z: Tensor) -> Tensor:
    r"""``lgamma(z) - [(z - 1/2) log z - z + log(2 pi) / 2]``, small for a large ``z``.

    :math:`S(z) \to 1/(12 z)`.  The ``lgamma`` of a large argument is never formed: from
    ``z = 10`` the asymptotic series in ``1/z`` is read, below it the difference of ``lgamma``
    and the closed part (both O(10) there).  An argument of 0 or below is out of its domain.
    """
    return _in_float32(_stirling_remainder_impl, z)


def _digamma_remainder_impl(z: Tensor) -> Tensor:
    terms = _TERMS_64 if _is_double(z) else _TERMS_32
    far = z >= _SERIES_FROM
    r = 1.0 / lucid.where(far, z, lucid.full_like(z, _SERIES_FROM))
    series = r * r * _horner_float(_D_COEFFS[:terms], r * r)
    near = lucid.where(far, lucid.full_like(z, _SERIES_FROM), z)
    plain = lucid.log(near) - 0.5 / near - lucid.digamma(near)
    return lucid.where(far, series, plain)


def _digamma_remainder(z: Tensor) -> Tensor:
    r"""``log(z) - 1/(2z) - digamma(z)``, small for a large ``z``: :math:`D(z) \to 1/(12 z^2)`.

    The derivative companion of :func:`_stirling_remainder` (``S' = -D``).
    """
    return _in_float32(_digamma_remainder_impl, z)


def _gamma_tail_impl(z: Tensor) -> Tensor:
    return -0.5 / z + _stirling_remainder_impl(z) + (z - 1.0) * _digamma_remainder_impl(z)


def _gamma_tail(z: Tensor) -> Tensor:
    r"""``G(z) = -1/(2z) + S(z) + (z - 1) D(z)``, so that ``z + lgamma(z) + (1 - z) digamma(z)
    = log(2 pi e z) / 2 + G(z)``.

    :math:`G \to -1/(3z)`.  The Gamma entropy ``alpha + lgamma(alpha) + (1 - alpha)
    digamma(alpha)`` is a sum of ``alpha log alpha``-sized terms that cancel to
    ``1/2 log(2 pi e alpha)``; written through ``G`` none of them is formed and the slope in
    ``alpha`` is a sum of ``1/alpha^2`` terms, not ``1 - alpha psi_1(alpha)``.
    """
    return _in_float32(_gamma_tail_impl, z)


def _bd0_impl(x: Tensor, m: Tensor) -> Tensor:
    terms = _BD0_TERMS_64 if _is_double(x) else _BD0_TERMS_32
    v = (x - m) / (x + m)
    near = v.abs() < _BD0_V
    one = lucid.ones_like(x)
    # Series lane: (x - m) v + 2 x sum_{j>=1} v^(2j+1)/(2j+1), every term of one sign.  The
    # far lane is held at x = m = 1 there (v = 0), and the series lane at x = 1, m = 2 below.
    xs = lucid.where(near, x, one)
    ms = lucid.where(near, m, one)
    vs = (xs - ms) / (xs + ms)
    v2 = vs * vs
    series = (xs - ms) * vs + 2.0 * xs * vs * v2 * _horner_float(_BD0_COEFFS[:terms], v2)
    xd = lucid.where(near, one, x)
    md = lucid.where(near, 2.0 * one, m)
    direct = xd * lucid.log(xd / md) + md - xd
    return lucid.where(near, series, direct)


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


def _log_shares(a: Tensor, s: Tensor) -> Tensor:
    """``log(a_i / s)`` along the last axis; the largest share is read as ``log1p(-rest / s)``.

    ``a_i / s`` rounds to a relative ``eps``, which is an absolute ``eps`` in its logarithm; times
    ``a_i`` it is ``eps * s`` for the one share near 1.  ``rest``, the sum of the others, is
    exact where ``s - a_i`` is not.
    """
    k = a.shape[-1]
    dominant = lucid.arange(k, device=a.device) == a.argmax(dim=-1, keepdim=True)
    rest = lucid.where(dominant, lucid.zeros_like(a), a).sum(dim=-1, keepdim=True)
    return lucid.where(dominant, lucid.log1p(-rest / s), lucid.log(a / s))


class _Stirling:
    """The Stirling frame of one concentration vector (last axis ``K``): shares, remainder, lanes.

    ``large`` is the per-lane mask of the centred forms (a valid vector with ``s >= _CENTRED_FROM``);
    ``a`` is the vector held inside that domain, so the centred pieces are finite on every
    lane.  ``plain`` holds the complementary domain for the ``lgamma`` / ``digamma`` forms.
    """

    def __init__(self, alpha: Tensor) -> None:
        total = alpha.sum(dim=-1, keepdim=True)
        valid = (alpha.amin(dim=-1, keepdim=True) > 0.0) & lucid.isfinite(total)
        self.k: int = alpha.shape[-1]
        self.large: Tensor = valid & (total >= _CENTRED_FROM)
        self.a: Tensor = lucid.where(self.large, alpha, lucid.full_like(alpha, 2.0))
        self.plain: Tensor = lucid.where(self.large, lucid.ones_like(alpha), alpha)
        self.s: Tensor = self.a.sum(dim=-1, keepdim=True)
        self.shares: Tensor = _log_shares(self.a, self.s)
        self._tail: Tensor | None = None

    @property
    def tail(self) -> Tensor:
        """``R = log B(a) - sum a_i log(a_i / s)`` on the centred domain (shape without the last axis)."""
        if self._tail is None:
            s = self.s.squeeze(-1)
            self._tail = (
                -0.5 * self.shares.sum(dim=-1)
                + 0.5 * (self.k - 1) * (_LOG_2PI - lucid.log(s))
                + _stirling_remainder(self.a).sum(dim=-1)
                - _stirling_remainder(s)
            )
        return self._tail

    def _log_beta(self) -> Tensor:
        """``sum lgamma(a_i) - lgamma(s)``: centred on the large lanes, plain elsewhere."""
        centred = (self.a * self.shares).sum(dim=-1) + self.tail
        plain = lucid.lgamma(self.plain).sum(dim=-1) - lucid.lgamma(self.plain.sum(dim=-1))
        return lucid.where(self.large.squeeze(-1), centred, plain)

    def deviance(self, x: Tensor) -> Tensor:
        """``sum_i bd0(a_i, s x_i)``: ``-`` this is ``sum a_i log(x_i / p_i)`` when ``sum x = 1``."""
        return _bd0(self.a, self.s * x).sum(dim=-1)

    def near(self, x: Tensor) -> Tensor:
        """Per lane: every ``x_i`` within ``_NEAR_V`` of its mean count ``a_i / s`` (see ``_NEAR_V``)."""
        m = self.s * x
        return ((self.a - m).abs() <= _NEAR_V * (self.a + m)).all(dim=-1)

    def entropy(self) -> Tensor:
        """The Dirichlet entropy ``log B + (s - K) psi(s) - sum (a_i - 1) psi(a_i)``."""
        s = self.s.squeeze(-1)
        k1 = self.k - 1.0
        centred = (
            0.5 * self.shares.sum(dim=-1)
            + 0.5 * k1 * (_LOG_2PI + 1.0 - lucid.log(s))
            + _gamma_tail(self.a).sum(dim=-1)
            - _gamma_tail(s)
            + k1 * (0.5 / s + _digamma_remainder(s))
        )
        ps = self.plain.sum(dim=-1)
        plain = (
            self._log_beta()
            + (ps - self.k) * lucid.digamma(ps)
            - ((self.plain - 1.0) * lucid.digamma(self.plain)).sum(dim=-1)
        )
        return lucid.where(self.large.squeeze(-1), centred, plain)


def _log_multi_beta_impl(alpha: Tensor) -> Tensor:
    return _Stirling(alpha)._log_beta()


def _log_multi_beta(alpha: Tensor) -> Tensor:
    r"""``sum_i lgamma(alpha_i) - lgamma(sum_i alpha_i)`` along the last axis, without the cancellation.

    For ``sum alpha >= 10`` it is :math:`\sum_i \alpha_i \log(\alpha_i/s) + R`, the entropy-like
    main part (large, but one rounding from its own value) and the centred remainder ``R``
    (``O(log s)``); below it the plain form, where every term is small.  A vector outside
    ``alpha > 0`` and finite takes the plain form (and its NaN / inf), as before.
    """
    return _in_float32(_log_multi_beta_impl, alpha)


def _log_beta(a: Tensor, b: Tensor) -> Tensor:
    """``lgamma(a) + lgamma(b) - lgamma(a + b)``: :func:`_log_multi_beta` of the pair."""
    a, b = _common(a, b)
    return _log_multi_beta(lucid.stack([a, b], dim=-1))


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
    plain = (
        lucid.lgamma(lucid.where(big, one, n) + 1.0)
        - lucid.lgamma(lucid.where(big, one, k) + 1.0)
        - lucid.lgamma(lucid.where(big, one, m) + 1.0)
    )
    # The ends are exactly 0 in value (as ``lgamma(n + 1) - lgamma(1) - lgamma(n + 1)`` is) and keep the
    # slope of the function: ``centred - centred.detach()`` is 0.0 with the gradient of ``centred``.
    return lucid.where(big, lucid.where(ends, centred - centred.detach(), centred), plain)


def _dirichlet_entropy_impl(alpha: Tensor) -> Tensor:
    return _Stirling(alpha).entropy()


def _dirichlet_entropy(alpha: Tensor) -> Tensor:
    r"""Entropy of ``Dirichlet(alpha)`` along the last axis, without the digamma cancellation.

    :math:`\log B(\alpha) + (s - K)\psi(s) - \sum_i (\alpha_i - 1)\psi(\alpha_i)` is a sum of
    :math:`\alpha\log\alpha`-sized terms; from ``s >= 10`` it is read from the centred
    form instead (see the module docstring) and the plain one is kept below.
    """
    return _in_float32(_dirichlet_entropy_impl, alpha)


def _beta_entropy(a: Tensor, b: Tensor) -> Tensor:
    """Entropy of ``Beta(a, b)``: :func:`_dirichlet_entropy` of the pair."""
    a, b = _common(a, b)
    return _dirichlet_entropy(lucid.stack([a, b], dim=-1))


def _gamma_entropy_impl(alpha: Tensor) -> Tensor:
    centred_lane = (alpha >= _CENTRED_FROM) & lucid.isfinite(alpha)
    big = lucid.where(centred_lane, alpha, lucid.full_like(alpha, _CENTRED_FROM))
    small = lucid.where(centred_lane, lucid.ones_like(alpha), alpha)
    centred = 0.5 * (_LOG_2PI + 1.0 + lucid.log(big)) + _gamma_tail_impl(big)
    plain = small + lucid.lgamma(small) + (1.0 - small) * lucid.digamma(small)
    return lucid.where(centred_lane, centred, plain)


def _gamma_entropy(alpha: Tensor) -> Tensor:
    r"""``alpha + lgamma(alpha) + (1 - alpha) digamma(alpha)``: the entropy of ``Gamma(alpha, 1)``.

    For ``alpha >= 10`` it is :math:`\tfrac12\log(2\pi e\,\alpha) + G(\alpha)` with the tail
    ``G`` of :func:`_gamma_tail`; below, the plain form.  (``Gamma(1e6, 1)``: float32 gave 7.0
    for 8.3267.)  Its slope in ``alpha`` is ``1/(2 alpha) + G'`` and not
    ``1 - (alpha - 1) psi_1(alpha)``, a difference of two numbers near 1.
    """
    return _in_float32(_gamma_entropy_impl, alpha)


def _beta_log_prob_impl(a: Tensor, b: Tensor, x: Tensor) -> Tensor:
    frame = _Stirling(lucid.stack([a, b], dim=-1))
    y = 1.0 - x
    # The centred form takes the interior of the support; a boundary point is a different
    # (xlogy) formula, which sees the stable log B.
    inside = frame.large.squeeze(-1) & (x > 0.0) & (y > 0.0)
    inside = inside & frame.near(lucid.stack([x, y], dim=-1))
    half = lucid.full_like(x, 0.5)
    xc, yc = lucid.where(inside, x, half), lucid.where(inside, y, half)
    centred = (
        -frame.deviance(lucid.stack([xc, yc], dim=-1)) - lucid.log(xc) - lucid.log(yc) - frame.tail
    )
    plain = lucid.xlogy(a - 1.0, x) + _xlog1py(b - 1.0, -x) - frame._log_beta()
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
    frame = _Stirling(alpha)
    inside = (
        frame.large.squeeze(-1)
        & (x.amin(dim=-1) > 0.0)
        & lucid.isfinite(x.amax(dim=-1))
        & frame.near(x)
    )
    xc = lucid.where(inside.unsqueeze(-1), x, lucid.full_like(x, 1.0 / frame.k))
    # sum a_i log(x_i / p_i) = -sum bd0(a_i, s x_i) + s (sum x_i - 1); the last term is 0 on the
    # simplex, and keeps the slope off it the plain formula's.
    centred = (
        -frame.deviance(xc)
        + frame.s.squeeze(-1) * (xc.sum(dim=-1) - 1.0)
        - lucid.log(xc).sum(dim=-1)
        - frame.tail
    )
    plain = lucid.xlogy(alpha - 1.0, x).sum(dim=-1) - frame._log_beta()
    return lucid.where(inside, centred, plain)


def _dirichlet_log_prob(alpha: Tensor, x: Tensor) -> Tensor:
    r"""``sum_i (alpha_i - 1) log x_i - log B(alpha)``, the same way as :func:`_beta_log_prob`."""
    alpha, x = _common(alpha, x)
    return _in_float32(_dirichlet_log_prob_impl, alpha, x)


def _f_log_prob_impl(d1: Tensor, d2: Tensor, x: Tensor) -> Tensor:
    half1, half2 = d1 * 0.5, d2 * 0.5
    frame = _Stirling(lucid.stack([half1, half2], dim=-1))
    lane = frame.large.squeeze(-1) & (x > 0.0) & lucid.isfinite(x)
    # The centred pieces are formed from sanitised operands: an infinite ``x`` or ``d`` would meet
    # the zero cotangent of the ``where`` below as ``inf * 0``.
    xs = lucid.where(lane, x, lucid.ones_like(x))
    c1, c2 = 2.0 * frame.a[..., 0], 2.0 * frame.a[..., 1]
    w = c1 * xs + c2
    u, y = c1 * xs / w, c2 / w
    inside = lane & (u > 0.0) & (y > 0.0) & frame.near(lucid.stack([u, y], dim=-1))
    half = lucid.full_like(x, 0.5)
    xc, uc, yc = (lucid.where(inside, t, half) for t in (xs, u, y))
    # log p(x) = log Beta(u; a, b) + log(d1 d2 / w^2), and -log u - log y + log(d1 d2 / w^2) is
    # -log x: the large logs cancel before they are formed.
    centred = -frame.deviance(lucid.stack([uc, yc], dim=-1)) - lucid.log(xc) - frame.tail
    ratio = d1 / d2
    plain = (
        half1 * lucid.log(ratio)
        + lucid.xlogy(half1 - 1.0, x)
        - (half1 + half2) * lucid.log1p(ratio * x)
        - frame._log_beta()
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
    plain = lucid.lgamma(plain_counts.sum(dim=-1) + 1.0) - lucid.lgamma(plain_counts + 1.0).sum(
        dim=-1
    )
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
    # / sqrt(2 pi k (n-k) / n), with S the Stirling remainder.
    return (
        _stirling_remainder_impl(nc)
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
        -_bd0_impl(ac, m)
        - lucid.log(xc)
        + 0.5 * (lucid.log(ac) - _LOG_2PI)
        - _stirling_remainder_impl(ac)
    )
    return lucid.where(inside, centred, plain)


def _gamma_log_prob_impl(alpha: Tensor, rate: Tensor, x: Tensor) -> Tensor:
    plain = alpha * lucid.log(rate) + lucid.xlogy(alpha - 1.0, x) - rate * x - lucid.lgamma(alpha)
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
    plain = alpha * lucid.log(rate) - lucid.lgamma(alpha) - (alpha + 1.0) * lucid.log(x) - rate / x
    return _gamma_family_log_prob(alpha, rate, x, plain, inverse=True)


def _inverse_gamma_log_prob(alpha: Tensor, rate: Tensor, x: Tensor) -> Tensor:
    r"""The log-density of ``InverseGamma(alpha, rate)``: :func:`_gamma_log_prob`'s form at ``rate / x``."""
    alpha, rate, x = _common(alpha, rate, x)
    return _in_float32(_inverse_gamma_log_prob_impl, alpha, rate, x)


def _poisson_log_prob_impl(rate: Tensor, k: Tensor) -> Tensor:
    plain = lucid.xlogy(k, rate) - rate - lucid.lgamma(k + 1.0)
    inside = (k >= _CENTRED_FROM) & lucid.isfinite(k) & (rate > 0.0) & lucid.isfinite(rate)
    base = lucid.full_like(k, _CENTRED_FROM)
    kc = lucid.where(inside, k, base)
    rc = lucid.where(inside, rate, base)
    centred = -_bd0_impl(kc, rc) - 0.5 * (_LOG_2PI + lucid.log(kc)) - _stirling_remainder_impl(kc)
    return lucid.where(inside, centred, plain)


def _poisson_log_prob(rate: Tensor, k: Tensor) -> Tensor:
    r"""``k log(rate) - rate - lgamma(k + 1)``: for ``k >= 10`` it is :math:`-\mathrm{bd0}(k, \lambda)
    - \tfrac12\log(2\pi k) - S(k)`, with no term of size ``k log k``."""
    rate, k = _common(rate, k)
    return _in_float32(_poisson_log_prob_impl, rate, k)


def _inverse_gamma_entropy_impl(alpha: Tensor, rate: Tensor) -> Tensor:
    centred_lane = (alpha >= _CENTRED_FROM) & lucid.isfinite(alpha)
    big = lucid.where(centred_lane, alpha, lucid.full_like(alpha, _CENTRED_FROM))
    small = lucid.where(centred_lane, lucid.ones_like(alpha), alpha)
    log_rate = lucid.log(rate)
    centred = log_rate + _gamma_entropy_impl(big) - 2.0 * lucid.digamma(big)
    plain = small + log_rate + lucid.lgamma(small) - (small + 1.0) * lucid.digamma(small)
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
        + lucid.lgamma(a_q)
        - lucid.lgamma(a_p)
        + (a_p - a_q) * lucid.digamma(a_p)
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
