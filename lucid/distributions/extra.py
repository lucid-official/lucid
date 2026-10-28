"""Additional distributions: ``Gumbel``, ``InverseGamma``, ``Kumaraswamy``,
``Multinomial``, ``ContinuousBernoulli``.

All are pure-Lucid composites — no engine work required.  Each distribution
is numerically stable in its main operating region, using Taylor
approximations or clamping where analytic expressions degenerate.
"""

import math
from collections.abc import Sequence
from typing import Self, override

import lucid
from lucid._dtype import finfo
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import _align_device, _as_tensor
from lucid.distributions._util import _broadcast_pair
from lucid.distributions._util import _clamp_probs, _lazy_param, _set_probs_or_logits
from lucid.distributions.bernoulli import _logits_to_probs, _probs_to_logits
from lucid.distributions.constraints import (
    Constraint,
    multinomial,
    nonnegative_integer,
    positive,
    real,
    real_vector,
    simplex,
    unit_interval,
)
from lucid.distributions.distribution import Distribution
from lucid.special import gammaincc, xlog1py

# ── Euler–Mascheroni constant ─────────────────────────────────────────────────
_EULER_GAMMA: float = 0.5772156649015329

# Half-width of the window around ``λ = ½`` where ``ContinuousBernoulli``'s
# mean and variance switch from their closed forms to Taylor series.  At
# ``|λ − ½| = 0.1`` the closed forms lose under 2e-6 to cancellation in
# float32 and the truncated series are within 2e-7 of the exact value;
# nearer ½ the closed forms degrade fast (a float32 variance off by more
# than 1 at ``λ = 0.499``), farther out the series do.
_CB_TAYLOR_HALF_WIDTH: float = 0.1


# ── Gumbel ────────────────────────────────────────────────────────────────────


class Gumbel(Distribution):
    r"""Gumbel (Type-I extreme value) distribution on :math:`\mathbb{R}`.

    Continuous distribution that models the maximum (or minimum) of a
    sample drawn from a number of distributions with exponential-like
    tails.  It is the limiting distribution of the maximum of IID
    exponential-tailed variables (Fisher–Tippett–Gnedenko theorem) and is
    the basis of the celebrated **Gumbel-max trick** for differentiable
    sampling from categorical distributions.

    Parameters
    ----------
    loc : Tensor or float
        Location parameter :math:`\mu \in \mathbb{R}` (mode of the
        density).
    scale : Tensor or float
        Scale parameter :math:`s > 0`.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density (with :math:`z = (x - \mu)/s`):

    .. math::

        p(x; \mu, s) = \frac{1}{s} \exp\!\bigl(-(z + e^{-z})\bigr)

    Cumulative distribution:

    .. math::

        F(x; \mu, s) = \exp\!\bigl(-e^{-z}\bigr)

    Moments:

    .. math::

        \mathbb{E}[X] = \mu + s\,\gamma_{\mathrm{E}}, \qquad
        \mathrm{Var}[X] = \frac{\pi^2 s^2}{6}

    where :math:`\gamma_{\mathrm{E}} \approx 0.5772` is the
    Euler–Mascheroni constant.  The mode is :math:`\mu` and the entropy
    is :math:`\log s + \gamma_{\mathrm{E}} + 1`.

    **Gumbel-softmax trick**: if :math:`g_i \sim \mathrm{Gumbel}(0, 1)`
    IID and :math:`\pi_i` are category probabilities, then
    :math:`\arg\max_i (\log \pi_i + g_i)` is a sample from
    :math:`\mathrm{Categorical}(\pi)`.  Replacing the argmax with a
    softmax yields a continuous relaxation that supports gradient flow
    through discrete choices (Jang et al., 2017; Maddison et al., 2017).

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Gumbel
    >>> d = Gumbel(loc=0.0, scale=1.0)
    >>> d.rsample((4,)).shape
    (4,)
    >>> d.log_prob(lucid.tensor(0.0))  # -(0 + e^0)
    tensor(-1.)
    """

    arg_constraints = {"loc": real, "scale": positive}
    support: Constraint | None = real
    has_rsample: bool = True

    def __init__(
        self, loc: Tensor | float, scale: Tensor | float, validate_args: bool | None = None
    ) -> None:
        r"""Initialise a Gumbel distribution.

        Parameters
        ----------
        loc : Tensor | float
            Location parameter :math:`\mu \in \mathbb{R}`.  Shifts the
            distribution along the real line.
        scale : Tensor | float
            Scale parameter :math:`s > 0`.  Stretches the distribution.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
        self.loc, self.scale = _broadcast_pair(_as_tensor(loc), _as_tensor(scale))
        super().__init__(
            batch_shape=tuple(self.loc.shape), event_shape=(), validate_args=validate_args
        )

    @property
    @override
    def mean(self) -> Tensor:
        """``μ + s · γ`` where ``γ`` is the Euler–Mascheroni constant."""
        return self.loc + self.scale * _EULER_GAMMA

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Gumbel` with batch dimensions expanded to ``batch_shape``."""
        return self._expand_attributes(Gumbel, batch_shape, _instance, "loc", "scale")

    @property
    @override
    def variance(self) -> Tensor:
        """``(π · s)² / 6``."""
        return (math.pi * self.scale) ** 2 / 6.0

    @property
    @override
    def mode(self) -> Tensor:
        """Mode of the Gumbel distribution.

        Returns
        -------
        Tensor
            The location parameter :math:`\\mu`, which coincides with the mode
            of the Gumbel PDF.
        """
        return self.loc

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Reparameterised sample via the Gumbel icdf: ``μ − s · log(−log U)``."""
        shape = self._extended_shape(sample_shape)
        u: Tensor = lucid.rand(*shape, dtype=self.loc.dtype, device=self.loc.device)
        # Clamp away from 0 and 1 to avoid log(0) = -inf.
        u = lucid.clamp(u, min=1e-6, max=1.0 - 1e-6)
        return self.loc - self.scale * (-(u.log())).log()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the Gumbel distribution.

        Parameters
        ----------
        value : Tensor
            Point(s) :math:`x \in \mathbb{R}` at which to evaluate the density.

        Returns
        -------
        Tensor
            Log-density :math:`\log p(x) = -(z + e^{-z}) - \log s`
            where :math:`z = (x - \mu) / s`.
        """
        z: Tensor = (value - self.loc) / self.scale
        return -(z + (-z).exp()) - self.scale.log()

    @override
    def entropy(self) -> Tensor:
        """``log(s) + γ + 1``."""
        return self.scale.log() + _EULER_GAMMA + 1.0

    @override
    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the Gumbel distribution.

        .. math::

            F(x) = \exp\!\left(-e^{-(x - \mu)/s}\right)

        Returns probabilities in :math:`[0, 1]`.

        Examples
        --------
        >>> Gumbel(loc=0.0, scale=1.0).cdf(lucid.tensor(0.0))  # 1/e
        tensor(0.3679)
        """
        return (-(-(value - self.loc) / self.scale).exp()).exp()

    @override
    def icdf(self, value: Tensor) -> Tensor:
        r"""Quantile function of the Gumbel distribution.

        .. math::

            Q(p) = \mu - s \ln(-\ln u), \qquad
            u = \epsilon_{\text{tiny}} + p\,(1 - \epsilon - \epsilon_{\text{tiny}})

        The reference framework builds the Gumbel from a uniform on
        :math:`[\epsilon_{\text{tiny}}, 1 - \epsilon]` (the dtype's
        smallest normal number and machine epsilon) so that no draw is
        infinite, and its quantile function maps :math:`[0, 1]` onto that
        interval first.  This does the same: the quantiles at 0 and 1 are
        finite — :math:`\mu - 4.47 s` and :math:`\mu + 15.9 s` in float32
        — and every other one differs from the exact
        :math:`\mu - s\ln(-\ln p)` by a relative :math:`\epsilon`.

        Returns quantiles.

        Examples
        --------
        >>> Gumbel(loc=0.0, scale=1.0).icdf(lucid.tensor(0.5))  # -ln(ln 2)
        tensor(0.3665)
        """
        info = finfo(self.loc.dtype)
        tiny, eps = float(info.tiny), float(info.eps)
        u = tiny + value * (1.0 - eps - tiny)
        return self.loc - self.scale * (-u.log()).log()


# ── InverseGamma ──────────────────────────────────────────────────────────────


class InverseGamma(Distribution):
    r"""Inverse-Gamma distribution on :math:`(0, \infty)`.

    Distribution of the reciprocal of a Gamma random variable: if
    :math:`Y \sim \mathrm{Gamma}(\alpha, \beta)` then
    :math:`X = 1/Y \sim \mathrm{InverseGamma}(\alpha, \beta)`.  Widely used
    as the **conjugate prior** for the variance parameter of a Normal
    distribution with known mean, and as a prior on positive scale
    parameters more generally.

    Parameters
    ----------
    concentration : Tensor or float
        Shape parameter :math:`\alpha > 0`.
    rate : Tensor or float
        Rate (inverse-scale) parameter :math:`\beta > 0`.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density on :math:`x > 0`:

    .. math::

        p(x; \alpha, \beta) =
            \frac{\beta^\alpha}{\Gamma(\alpha)}
            x^{-\alpha - 1} \exp\!\left(-\frac{\beta}{x}\right)

    Moments (when defined):

    .. math::

        \mathbb{E}[X] = \frac{\beta}{\alpha - 1}\;\;(\alpha > 1), \qquad
        \mathrm{Var}[X] = \frac{\beta^2}{(\alpha - 1)^2 (\alpha - 2)}
        \;\;(\alpha > 2)

    Mode: :math:`\beta / (\alpha + 1)`.

    **Conjugacy**: with a :math:`\mathrm{Normal}(\mu, \sigma^2)`
    likelihood and known mean :math:`\mu`, the posterior over
    :math:`\sigma^2` is again Inverse-Gamma with updated parameters
    :math:`\alpha' = \alpha + n/2`,
    :math:`\beta' = \beta + \tfrac{1}{2}\sum_i (x_i - \mu)^2`.

    Because Lucid's :class:`~lucid.distributions.Gamma` sampler is
    rejection-based, ``has_rsample`` is ``False`` here and
    :meth:`sample` returns detached samples obtained by reciprocating
    a Gamma draw.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import InverseGamma
    >>> d = InverseGamma(concentration=3.0, rate=2.0)
    >>> d.mean  # β / (α - 1) = 1.0
    tensor(1.)
    >>> d.sample((4,)).shape
    (4,)
    """

    arg_constraints = {"concentration": positive, "rate": positive}
    support: Constraint | None = positive
    has_rsample: bool = True

    def __init__(
        self, concentration: Tensor | float, rate: Tensor | float, validate_args: bool | None = None
    ) -> None:
        r"""Initialise an Inverse-Gamma distribution.

        Parameters
        ----------
        concentration : Tensor | float
            Shape parameter :math:`\alpha > 0`.
        rate : Tensor | float
            Rate (inverse-scale) parameter :math:`\beta > 0`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
        self.concentration = _as_tensor(concentration)
        self.rate = _as_tensor(rate)
        self.concentration, self.rate = _broadcast_pair(self.concentration, self.rate)
        # Lazily import Gamma to avoid a circular dependency at import time.
        from lucid.distributions.gamma import Gamma as _Gamma

        self._gamma = _Gamma(self.concentration, self.rate, validate_args=False)
        super().__init__(
            batch_shape=tuple(self.concentration.shape), event_shape=(), validate_args=validate_args
        )

    @property
    @override
    def mean(self) -> Tensor:
        """Defined for ``concentration > 1``: ``β / (α − 1)``."""
        # Only valid for α > 1; the mean diverges at or below it.
        closed = self.rate / (self.concentration - 1.0)
        return lucid.where(self.concentration > 1.0, closed, lucid.full_like(closed, float("inf")))

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`InverseGamma` with batch dimensions expanded to ``batch_shape``."""
        return self._expand_attributes(
            InverseGamma, batch_shape, _instance, "concentration", "rate", "_gamma"
        )

    @property
    @override
    def mode(self) -> Tensor:
        r"""Mode of the inverse-Gamma distribution: :math:`\beta / (\alpha + 1)`."""
        return self.rate / (self.concentration + 1.0)

    @property
    @override
    def variance(self) -> Tensor:
        """Defined for ``concentration > 2``:
        ``β² / ((α − 1)² (α − 2))``."""
        c, r = self.concentration, self.rate
        # Only valid for α > 2 — see Pareto for the same defect and the
        # same reason: below the threshold the formula returns a negative
        # variance rather than the divergence it stands for.
        closed = r * r / ((c - 1.0) ** 2 * (c - 2.0))
        return lucid.where(c > 2.0, closed, lucid.full_like(closed, float("inf")))

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw reparameterised samples by taking the reciprocal of a ``Gamma`` sample."""
        return 1.0 / self._gamma.rsample(sample_shape)

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Sample by taking the reciprocal of a ``Gamma`` sample."""
        y: Tensor = self._gamma.sample(sample_shape)
        return (1.0 / y).detach()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the Inverse-Gamma distribution.

        Parameters
        ----------
        value : Tensor
            Point(s) :math:`x > 0` at which to evaluate the density.

        Returns
        -------
        Tensor
            Log-density :math:`\alpha\log\beta - \log\Gamma(\alpha)
            - (\alpha+1)\log x - \beta/x`.
        """
        c, r = self.concentration, self.rate
        return c * r.log() - lucid.lgamma(c) - (c + 1.0) * value.log() - r / value

    @override
    def entropy(self) -> Tensor:
        """``α + log β + log Γ(α) − (1 + α) · ψ(α)``."""
        c, r = self.concentration, self.rate
        return c + r.log() + lucid.lgamma(c) - (c + 1.0) * lucid.digamma(c)

    @override
    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the inverse-Gamma distribution.

        .. math::

            F(x; \alpha, \beta) = Q(\alpha, \beta / x), \quad x > 0

        the regularized upper incomplete gamma
        (:func:`lucid.special.gammaincc`) — :math:`1/X` is
        ``Gamma(alpha, beta)``, and :math:`X \leq x` is
        :math:`1/X \geq 1/x`.  Evaluated as :math:`Q` itself, not as
        :math:`1 - P`, so the small left tail keeps its relative precision.

        Returns probabilities in :math:`[0, 1]`; 0 at and below 0.

        Raises
        ------
        NotImplementedError
            From ``backward``, when ``concentration`` requires a gradient,
            as in the reference framework.

        Examples
        --------
        >>> InverseGamma(concentration=1.0, rate=1.0).cdf(lucid.tensor(1.0))  # 1/e
        tensor(0.3679)
        """
        # beta / 0 = inf, where Q is 0: the clamp makes a negative value
        # score 0 as well, rather than NaN.
        return gammaincc(self.concentration, self.rate / value.clamp(min=0.0))


# ── Kumaraswamy ───────────────────────────────────────────────────────────────


class Kumaraswamy(Distribution):
    r"""Kumaraswamy distribution on :math:`[0, 1]`.

    Two-parameter continuous distribution on the unit interval that
    mimics the shapes of the :class:`~lucid.distributions.Beta`
    distribution but has a **closed-form CDF and inverse CDF**.  This
    makes it a popular choice in variational autoencoders and normalising
    flows where cheap, exact reparameterised sampling is needed and no
    special functions are required.

    Parameters
    ----------
    concentration1 : Tensor or float
        First shape parameter :math:`a > 0`.
    concentration0 : Tensor or float
        Second shape parameter :math:`b > 0`.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density on :math:`x \in [0, 1]`:

    .. math::

        p(x; a, b) = a\, b\, x^{a - 1} (1 - x^a)^{b - 1}

    Cumulative distribution (closed form):

    .. math::

        F(x; a, b) = 1 - (1 - x^a)^b

    Inverse CDF (used by :meth:`rsample`):

    .. math::

        F^{-1}(u; a, b) = (1 - (1 - u)^{1/b})^{1/a}

    Moments are expressible in closed form via the Beta function:

    .. math::

        \mathbb{E}[X^n] = b\, B(1 + n/a,\, b)

    Special cases:

    * :math:`a = 1` → :math:`\mathrm{Beta}(1, b)`
    * :math:`b = 1` → :math:`\mathrm{Beta}(a, 1)`

    Unlike Beta, Kumaraswamy is *not* a member of the exponential family
    and lacks an analytic normalising integral for sums.  Its main
    advantage is the lack of any reliance on the gamma function in
    forward or backward passes.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Kumaraswamy
    >>> d = Kumaraswamy(concentration1=2.0, concentration0=5.0)
    >>> d.rsample((4,)).shape
    (4,)
    >>> d.log_prob(lucid.tensor(0.3))  # log(2 * 5 * 0.3 * (1 - 0.3**2)**4)
    tensor(0.7214)
    """

    arg_constraints = {"concentration1": positive, "concentration0": positive}
    # ``[0, 1]``, as in the reference framework.  At an end the density is
    # finite, infinite or 0 depending on the shape parameter, and
    # :meth:`log_prob` scores each.
    support: Constraint | None = unit_interval
    has_rsample: bool = True

    def __init__(
        self,
        concentration1: Tensor | float,
        concentration0: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        """Initialise a Kumaraswamy distribution.

        Parameters
        ----------
        concentration1 : Tensor | float
            First shape parameter :math:`a > 0` (controls the left tail).
        concentration0 : Tensor | float
            Second shape parameter :math:`b > 0` (controls the right tail).
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
        self.concentration1 = _as_tensor(concentration1)  # a
        self.concentration0 = _as_tensor(concentration0)  # b
        self.concentration1, self.concentration0 = _broadcast_pair(
            self.concentration1, self.concentration0
        )
        super().__init__(
            batch_shape=tuple(self.concentration1.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    @override
    def mean(self) -> Tensor:
        """``E[X] = b · B(1 + 1/a, b)`` expressed via ``lgamma``."""
        a, b = self.concentration1, self.concentration0
        log_mean = (
            b.log()
            + lucid.lgamma(1.0 + 1.0 / a)
            + lucid.lgamma(b)
            - lucid.lgamma(1.0 + 1.0 / a + b)
        )
        return log_mean.exp()

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Kumaraswamy` with batch dimensions expanded to ``batch_shape``."""
        return self._expand_attributes(
            Kumaraswamy, batch_shape, _instance, "concentration1", "concentration0"
        )

    @property
    @override
    def mode(self) -> Tensor:
        r"""Mode of the Kumaraswamy distribution.

        .. math::

            \left(\frac{a - 1}{a b - 1}\right)^{1/a} \quad (a, b \geq 1)

        with :math:`a` = ``concentration1`` and :math:`b` = ``concentration0``;
        ``nan`` when either is below 1, as the reference framework answers.
        The reference evaluates a different expression and answers ``nan``
        for every :math:`b > 1` as well; this is the closed form.
        """
        a = self.concentration1
        b = self.concentration0
        interior = ((a - 1.0) / (a * b - 1.0)) ** a.reciprocal()
        return lucid.where((a < 1.0) | (b < 1.0), float("nan"), interior)

    @property
    @override
    def variance(self) -> Tensor:
        """``Var[X] = E[X²] − E[X]²``  where ``E[X²] = b · B(1 + 2/a, b)``."""
        a, b = self.concentration1, self.concentration0
        log_sq_mean = (
            b.log()
            + lucid.lgamma(1.0 + 2.0 / a)
            + lucid.lgamma(b)
            - lucid.lgamma(1.0 + 2.0 / a + b)
        )
        return log_sq_mean.exp() - self.mean**2

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Reparameterised sample via the icdf:
        ``x = (1 − (1 − U)^(1/b))^(1/a)``."""
        shape = self._extended_shape(sample_shape)
        u: Tensor = lucid.rand(
            *shape, dtype=self.concentration1.dtype, device=self.concentration1.device
        )
        u = lucid.clamp(u, min=1e-6, max=1.0 - 1e-6)
        a, b = self.concentration1, self.concentration0
        return (1.0 - (1.0 - u) ** (1.0 / b)) ** (1.0 / a)

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the Kumaraswamy distribution.

        Parameters
        ----------
        value : Tensor
            Point(s) :math:`x \in [0, 1]` at which to evaluate the density.

        Returns
        -------
        Tensor
            Log-density :math:`\log a + \log b + (a-1)\log x + (b-1)\log(1-x^a)`.

        Notes
        -----
        The two weighted logarithms are :func:`lucid.xlogy` and
        :func:`lucid.special.xlog1py`, so the ends of the support score the
        density's limit there: at :math:`x = 0` it is ``+inf`` for
        :math:`a < 1`, ``-inf`` for :math:`a > 1` and :math:`\log b` for
        :math:`a = 1`; :math:`x = 1` is the same in :math:`b`.  The
        reference framework answers NaN at both ends.
        """
        a, b = self.concentration1, self.concentration0
        return a.log() + b.log() + lucid.xlogy(a - 1.0, value) + xlog1py(b - 1.0, -(value**a))

    @override
    def entropy(self) -> Tensor:
        """Entropy via the Beta-distributed auxiliary variable Y = X^a.

        If X ~ Kumaraswamy(a, b) then Y = X^a ~ Beta(1, b), giving:
          E[log X]       = (1/a)(ψ(1) − ψ(b+1)) = −(1/a)(γ + ψ(b+1))
          E[log(1−X^a)] = ψ(b) − ψ(b+1)         = −1/b

        H = −log a − log b − (a−1)·E[log X] − (b−1)·E[log(1−X^a)]
          = −log a − log b + (1 − 1/a)(γ + ψ(b+1)) + (1 − 1/b)
        """
        a, b = self.concentration1, self.concentration0
        # ψ(b+1) = ψ(concentration0 + 1)
        harmonic: Tensor = _EULER_GAMMA + lucid.digamma(b + 1.0)
        return (1.0 - 1.0 / a) * harmonic + (1.0 - 1.0 / b) - a.log() - b.log()

    @override
    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the Kumaraswamy distribution.

        .. math::

            F(x; a, b) = 1 - (1 - x^a)^b, \quad x \in [0, 1]

        Evaluated as :math:`-\operatorname{expm1}(b \log(1 - x^a))`, which
        keeps the left tail's relative precision.

        Returns probabilities in :math:`[0, 1]`; 0 below the support and 1 above
        it.

        Examples
        --------
        >>> Kumaraswamy(1.0, 1.0).cdf(lucid.tensor(0.25))  # Uniform(0, 1)
        tensor(0.25)
        """
        x = value.clamp(min=0.0, max=1.0)
        return -lucid.expm1(self.concentration0 * (-(x**self.concentration1)).log1p())

    @override
    def icdf(self, value: Tensor) -> Tensor:
        r"""Quantile function of the Kumaraswamy distribution.

        .. math::

            Q(p; a, b) = \left(1 - (1 - p)^{1/b}\right)^{1/a}

        Returns quantiles in :math:`[0, 1]`.

        Examples
        --------
        >>> Kumaraswamy(1.0, 1.0).icdf(lucid.tensor(0.25))
        tensor(0.25)
        """
        a, b = self.concentration1, self.concentration0
        return (-lucid.expm1((-value).log1p() / b)) ** (1.0 / a)


# ── Multinomial ───────────────────────────────────────────────────────────────


class Multinomial(Distribution):
    r"""Multinomial distribution over :math:`K` categories.

    Multivariate generalisation of the :class:`~lucid.distributions.Bernoulli`
    /Binomial: the joint distribution of the counts
    :math:`(k_1, \ldots, k_K)` obtained when :math:`n = \text{total\_count}`
    independent draws are made from a Categorical with probabilities
    :math:`(p_1, \ldots, p_K)`.  The :math:`k_i` are non-negative integers
    summing to :math:`n`.

    The event dimension is the last axis of ``probs`` / ``logits``; all
    preceding axes form the batch shape.

    Parameters
    ----------
    total_count : int or Tensor, optional
        Total number of trials :math:`n \geq 0`.  Default ``1`` (a
        one-hot Categorical sample).
    probs : Tensor, optional
        Unnormalised category probabilities of shape ``(..., K)`` —
        normalised internally to sum to 1 along the last axis.  Mutually
        exclusive with ``logits``.
    logits : Tensor, optional
        Unnormalised log-probabilities of shape ``(..., K)``, converted
        via softmax.  Mutually exclusive with ``probs``.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability mass function (joint over the count vector
    :math:`\mathbf{k}` with :math:`\sum_i k_i = n`):

    .. math::

        P(\mathbf{K} = \mathbf{k} \mid n, \mathbf{p}) =
            \binom{n}{k_1, \ldots, k_K} \prod_{i=1}^{K} p_i^{k_i},
        \qquad
        \binom{n}{k_1, \ldots, k_K} = \frac{n!}{\prod_i k_i!}

    Moments (per category):

    .. math::

        \mathbb{E}[K_i] = n p_i, \qquad
        \mathrm{Var}[K_i] = n p_i (1 - p_i), \qquad
        \mathrm{Cov}[K_i, K_j] = -n p_i p_j \;\; (i \neq j)

    Special cases:

    * :math:`K = 2` → :math:`\mathrm{Binomial}(n, p_1)`
    * :math:`n = 1` → one-hot :class:`~lucid.distributions.Categorical`

    **Conjugate prior**: :class:`~lucid.distributions.Dirichlet` —
    observing counts :math:`\mathbf{k}` updates
    ``Dirichlet(α) → Dirichlet(α + k)``.

    Sampling is non-reparameterised (``has_rsample = False``) and
    implemented by summing :math:`n` one-hot Categorical draws.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Multinomial
    >>> d = Multinomial(total_count=10, probs=lucid.tensor([0.2, 0.3, 0.5]))
    >>> d.mean  # n * p
    tensor([2., 3., 5.])
    >>> x = d.sample((4,))
    >>> x.shape
    (4, 3)
    >>> bool((x.sum(dim=-1) == 10).all())  # every draw places all n trials
    True
    """

    # ``total_count`` is in the table although the reference's is not: the
    # reference takes only a Python ``int`` and refuses a negative one through
    # the ``Binomial`` it builds, while this one also takes a tensor.
    arg_constraints = {"total_count": nonnegative_integer, "probs": simplex, "logits": real_vector}
    has_rsample: bool = False

    def __init__(
        self,
        total_count: Tensor | int = 1,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a Multinomial distribution.

        Parameters
        ----------
        total_count : Tensor | int, optional
            Total number of draws :math:`n \geq 0`.  Must be a scalar.
            Default is ``1`` (reduces to a one-hot Categorical).
        probs, logits : Tensor | None, optional
            The category weights, as probabilities or as log-probabilities,
            of shape ``(..., K)`` — exactly one of the two.  Either is
            normalised over the last axis before it is stored.
        validate_args : bool | None, optional
            Whether to check the parameters here and the counts later scored.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` and ``logits`` are given, or —
            under validation — a normalised parameter is not a probability
            vector or ``total_count`` is not a non-negative whole number.
        """
        from lucid.distributions.categorical import Categorical

        if (probs is None) == (logits is None):
            raise ValueError("Multinomial: pass exactly one of `probs` or `logits`.")
        # The categorical normalises whichever was given, and the normalised
        # value is what is stored here, so it is what validation checks — as
        # for ``Categorical`` itself.
        self._cat = Categorical(probs=probs, logits=logits, validate_args=False)
        self._is_logits = logits is not None
        if self._is_logits:
            self.logits = self._cat.logits
        else:
            self.probs = self._cat.probs
        # An ``int`` count becomes a 0-dim host tensor; it joins the
        # probabilities on their device, or ``n · p`` cannot run on Metal.
        self._total_count: Tensor = _align_device(_as_tensor(total_count), self._cat._probs)
        super().__init__(
            batch_shape=tuple(self._cat._batch_shape),
            event_shape=(self._cat._num_events,),
            validate_args=validate_args,
        )

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Multinomial` with batch dimensions expanded to ``batch_shape``."""
        new = self._get_checked_instance(Multinomial, _instance)
        batch_shape_t = tuple(batch_shape)
        new._cat = self._cat.expand(batch_shape_t)
        new._is_logits = self._is_logits
        new._total_count = self._total_count
        key = "logits" if new._is_logits else "probs"
        new.__dict__[key] = getattr(new._cat, key)
        return self._finish_expand(new, batch_shape_t, self._event_shape)

    @property
    @override
    def support(self) -> Constraint:
        """Count vectors of ``total_count`` draws.

        Returns
        -------
        Constraint
            ``multinomial(total_count)`` — non-negative entries summing to at
            most ``total_count``, the reference framework's support.
        """
        return multinomial(self.total_count)

    @property
    def _probs(self) -> Tensor:
        """Category probabilities, summing to 1 along the last axis."""
        return self._cat._probs

    @_lazy_param
    def probs(self) -> Tensor:
        """Category probabilities, normalised along the last axis."""
        return self._cat.probs

    @_lazy_param
    def logits(self) -> Tensor:
        """Log-probabilities — the ``logits`` argument shifted by its
        ``logsumexp``, or derived from ``probs`` on access.

        Derived as ``log(probs)`` with ``probs`` clamped one epsilon inside
        ``[0, 1]``, as the reference framework derives it.
        """
        return self._cat.logits

    @property
    def total_count(self) -> Tensor:
        """Total number of trials :math:`n` per Multinomial draw.

        Returns
        -------
        Tensor
            The integer-valued ``n`` parameter.  Each independent
            Multinomial sums to this many trials across the ``K`` categories.
        """
        return self._total_count

    @property
    @override
    def mean(self) -> Tensor:
        """``n · p`` (element-wise)."""
        return self.total_count * self._probs

    @property
    @override
    def variance(self) -> Tensor:
        """``n · p · (1 − p)`` (element-wise)."""
        p: Tensor = self._probs
        return self.total_count * p * (1.0 - p)

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Multinomial distribution (in nats).

        .. math::

            H = n H(\mathbf p) - \ln n!
                + \sum_{i=1}^{K} \sum_{j=1}^{n}
                  \binom{n}{j} p_i^j (1 - p_i)^{n-j} \ln j!

        where :math:`H(\mathbf p)` is the entropy of one categorical draw.
        From :math:`-\mathbb E[\ln P(\mathbf K)]` with
        :math:`\ln P = \ln n! - \sum_i \ln K_i! + \sum_i K_i \ln p_i`:
        each count :math:`K_i` is ``Binomial(n, p_i)``, so the expectation of
        :math:`\ln K_i!` is a sum over its :math:`n + 1` values — the
        reference framework's formula.

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions import Multinomial
        >>> Multinomial(2, probs=lucid.tensor([0.5, 0.5])).entropy()  # ln 4 - ln 2/2
        tensor(1.04)
        """
        from lucid.distributions.discrete import Binomial

        n = int(self.total_count.item())
        p: Tensor = self._probs
        term1 = n * self._cat.entropy() - math.lgamma(n + 1.0)
        if n == 0:
            return term1
        # Counts 1..n down a leading axis, against every batch member and
        # category; a count of 0 has ln 0! = 0 and is left out.
        counts = lucid.arange(1, n + 1, 1, dtype=p.dtype, device=p.device)
        counts = counts.reshape([n] + [1] * p.ndim)
        marginal = Binomial(self.total_count, probs=p, validate_args=False)
        weights = marginal.log_prob(counts).exp() * lucid.lgamma(counts + 1.0)
        return term1 + weights.sum(dim=0).sum(dim=-1)

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""``log C(n; k₁,…,kK) + Σ kᵢ log pᵢ`` with :math:`n = Σ kᵢ`.

        The coefficient counts the arrangements of the counts given, as the
        reference framework's does, so a vector summing to less than
        ``total_count`` is scored as a multinomial of its own total.  A
        category with :math:`k_i = 0` contributes ``0`` even where
        :math:`p_i = 0` (a zero in ``probs`` or a logit of ``-inf``), rather
        than ``0 · (-inf) = NaN``.
        """
        log_coeff: Tensor = lucid.lgamma(value.sum(dim=-1) + 1.0) - lucid.lgamma(value + 1.0).sum(
            dim=-1
        )
        if self._is_logits:
            log_p: Tensor = self.logits
            # ``log_softmax`` differentiates without dividing by p, so the
            # masked infinity can be swapped out after it is formed.
            log_p = lucid.where((value == 0) & lucid.isinf(log_p), 0.0, log_p)
            log_p_term: Tensor = (value * log_p).sum(dim=-1)
        else:
            log_p_term = lucid.xlogy(value, self._probs).sum(dim=-1)
        return log_coeff + log_p_term

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw samples by summing one-hot Categorical draws."""
        n: int = int(self.total_count.item())
        p: Tensor = self._probs  # (..., K)
        K: int = int(p.shape[-1])
        # Draw n categorical samples: (n, *batch_shape)
        draws: list[Tensor] = [self._cat.sample(sample_shape) for _ in range(n)]
        # Count each category k in {0, ..., K-1}.
        counts: list[Tensor] = [
            lucid.stack([(d == k).to(p.dtype) for d in draws]).sum(dim=0) for k in range(K)
        ]
        # Stack along last dim: (*sample_shape, *batch_shape, K)
        return lucid.stack(counts, dim=-1).to(lucid.int64).detach()


# ── ContinuousBernoulli ───────────────────────────────────────────────────────


class ContinuousBernoulli(Distribution):
    r"""Continuous Bernoulli distribution on :math:`[0, 1]`.

    A continuous relaxation of the :class:`~lucid.distributions.Bernoulli`
    distribution introduced by Loaiza-Ganem & Cunningham (2019) to fix the
    well-known bias incurred by VAE decoders that use a Bernoulli
    likelihood on real-valued image pixels in :math:`[0, 1]`.  Unlike the
    naive "Bernoulli on [0,1]" likelihood, the Continuous Bernoulli has a
    proper normalising constant and yields unbiased ELBO gradients.

    Parameters
    ----------
    probs : Tensor or float, optional
        Parameter :math:`\lambda \in [0, 1]`.  Note that
        :math:`\mathbb{E}[X] \neq \lambda` in general — see Notes for
        the closed-form mean.  Mutually exclusive with ``logits``.
    logits : Tensor or float, optional
        Log-odds :math:`\ell = \log(\lambda / (1 - \lambda))`.  Mutually
        exclusive with ``probs``.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density on :math:`x \in [0, 1]`:

    .. math::

        p(x; \lambda) = C(\lambda)\, \lambda^x (1 - \lambda)^{1 - x}

    where the normalising constant is

    .. math::

        C(\lambda) =
        \begin{cases}
            \dfrac{2 \tanh^{-1}(1 - 2\lambda)}{1 - 2\lambda}
                & \lambda \neq \tfrac{1}{2} \\[4pt]
            2 & \lambda = \tfrac{1}{2}
        \end{cases}

    Mean (closed form):

    .. math::

        \mathbb{E}[X] =
        \begin{cases}
            \dfrac{\lambda}{2\lambda - 1} +
            \dfrac{1}{2 \tanh^{-1}(1 - 2\lambda)}
                & \lambda \neq \tfrac{1}{2} \\[4pt]
            \tfrac{1}{2} & \lambda = \tfrac{1}{2}
        \end{cases}

    Variance (closed form, :math:`\ell = \operatorname{logit}\lambda`):

    .. math::

        \mathrm{Var}[X] = \frac{\lambda(\lambda - 1)}{(1 - 2\lambda)^2}
                        + \frac{1}{\ell^2}
        \quad (\lambda \neq \tfrac{1}{2}), \qquad
        \mathrm{Var}[X] = \tfrac{1}{12} \quad (\lambda = \tfrac{1}{2})

    Both closed forms are differences of terms of order
    :math:`1/(1 - 2\lambda)`, which cancel catastrophically as
    :math:`\lambda \to \tfrac{1}{2}`; within :math:`|\lambda - \tfrac{1}{2}|
    < 0.1` the mean and variance are evaluated from their Taylor series in
    :math:`\lambda - \tfrac{1}{2}` instead.

    Important properties:

    * Support: :math:`[0, 1]` (continuous, not just :math:`\{0, 1\}`).
    * Reduces to :math:`\mathrm{Uniform}(0, 1)` at :math:`\lambda = 1/2`.
    * At :math:`\lambda \to 0` or :math:`\lambda \to 1` the density
      concentrates near 0 or 1 respectively.
    * Reparameterisable via inverse CDF.

    The Continuous Bernoulli is the **correct** likelihood for decoders
    that output a value in :math:`[0, 1]` (e.g. MNIST pixel intensities)
    when the model is to remain a valid probabilistic model.  Using a
    plain Bernoulli on continuous data produces an *improper* density and
    a systematically biased ELBO.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import ContinuousBernoulli
    >>> d = ContinuousBernoulli(probs=0.7)
    >>> d.rsample((4,)).shape
    (4,)
    >>> d.log_prob(lucid.tensor(0.5))
    tensor(-0.02974)
    """

    arg_constraints = {"probs": unit_interval, "logits": real}
    support: Constraint | None = unit_interval
    has_rsample: bool = True
    _is_logits: bool

    def __init__(
        self,
        probs: Tensor | float | None = None,
        logits: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a Continuous Bernoulli distribution.

        Parameters
        ----------
        probs : Tensor | float | None, optional
            Parameter :math:`p \in [0, 1]`.  Mutually exclusive with
            ``logits``.
        logits : Tensor | float | None, optional
            Log-odds :math:`l = \log(p/(1-p)) \in \mathbb{R}`.  Mutually
            exclusive with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
        # Stored through the lazy parameters, as ``Bernoulli`` does, so the
        # one given is in ``arg_constraints``' reach: held in a private
        # ``_param`` instead, both ``probs`` and ``logits`` looked derived
        # and validation checked neither — ``probs=1.5`` and NaN constructed.
        shape = _set_probs_or_logits(self, "ContinuousBernoulli", probs, logits)
        super().__init__(batch_shape=shape, event_shape=(), validate_args=validate_args)

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`ContinuousBernoulli` with batch dimensions expanded to ``batch_shape``."""
        return self._expand_probs_or_logits(ContinuousBernoulli, batch_shape, _instance)

    @property
    def _param(self) -> Tensor:
        """The parameter as given — ``logits`` or ``probs``."""
        return self.logits if self._is_logits else self.probs

    @property
    def _probs(self) -> Tensor:
        """Probability parameter :math:`p \\in [0, 1]` — :attr:`probs`.

        As given, or the sigmoid of the given logits.
        """
        return self.probs

    @property
    def _logits(self) -> Tensor:
        """Lazily resolved logit parameter :math:`\\ell = \\log(p/(1-p))`.

        Returns the stored parameter when constructed from ``logits``;
        otherwise the log-odds of ``probs`` held one epsilon inside
        :math:`[0, 1]` (:attr:`logits`), so a degenerate ``probs`` of 0 or 1
        has a large finite logit, as in the reference framework.
        """
        return self.logits

    @_lazy_param
    def probs(self) -> Tensor:
        r"""Parameter :math:`\lambda` — as given, or ``sigmoid(logits)``."""
        return _logits_to_probs(self.logits)

    @_lazy_param
    def logits(self) -> Tensor:
        r"""Log-odds of :math:`\lambda` — as given, or derived on access.

        Derived from ``probs`` clamped one epsilon inside :math:`[0, 1]`, as
        the reference framework derives it.
        """
        return _probs_to_logits(_clamp_probs(self.probs))

    # -- helpers ---------------------------------------------------------------

    def _log_normalizer(self) -> Tensor:
        r"""Log normaliser :math:`\log C = \log|\ell| - \log|\tanh(\ell/2)|`.

        :math:`C(\lambda) = \ell / (2\lambda - 1)` and
        :math:`2\lambda - 1 = \tanh(\ell/2)`, so the normaliser is read off
        the logit without going through :math:`\lambda` — which rounds to
        exactly 0 or 1 once :math:`|\ell| \gtrsim 17` in float32 and took
        ``log_prob`` to ``inf`` with it.  Near :math:`\ell = 0` both logs
        diverge, and the series :math:`\log 2 + \ell^2/12 - 7\ell^4/1440`
        is used.
        """
        l: Tensor = self._logits
        near: Tensor = l.abs() < 0.05
        l_far: Tensor = lucid.where(near, 1.0, l)
        closed: Tensor = l_far.abs().log() - (0.5 * l_far).tanh().abs().log()
        x: Tensor = l * l
        series: Tensor = math.log(2.0) + x * (1.0 / 12.0 - x * (7.0 / 1440.0))
        return lucid.where(near, series, closed)

    # -- distribution interface ------------------------------------------------

    def _moment_operands(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""What :attr:`mean` and :attr:`variance` are evaluated from.

        Returns ``(t, near, p_far, l_far)``: :math:`t = \lambda - \tfrac12`,
        the mask of the Taylor window, and :math:`\lambda` and its logit
        with the window's entries replaced by a harmless point, so the
        closed form — evaluated everywhere and then discarded inside the
        window — never divides by zero.  An infinity in the discarded
        branch would still reach the gradient through the mask.
        """
        p: Tensor = self._probs
        t: Tensor = p - 0.5
        near: Tensor = t.abs() < _CB_TAYLOR_HALF_WIDTH
        p_far: Tensor = lucid.where(near, 0.25, p)
        l_far: Tensor = lucid.where(near, 1.0, self._logits)
        return t, near, p_far, l_far

    @property
    @override
    def mean(self) -> Tensor:
        r"""Mean :math:`\lambda/(2\lambda - 1) + 1/(2\tanh^{-1}(1 - 2\lambda))`.

        The second term is :math:`-1/\operatorname{logit}\lambda`.  Near
        :math:`\lambda = \tfrac12` the two terms cancel, so there the
        Taylor series
        :math:`\tfrac12 + \tfrac{t}{3} + \tfrac{16 t^3}{45}
        + \tfrac{704 t^5}{945} + \tfrac{27392 t^7}{14175}` in
        :math:`t = \lambda - \tfrac12` is used.

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions import ContinuousBernoulli
        >>> ContinuousBernoulli(probs=lucid.tensor([0.2, 0.5, 0.7])).mean
        tensor([0.388, 0.5, 0.5698])
        """
        t, near, p_far, l_far = self._moment_operands()
        closed: Tensor = p_far / (2.0 * p_far - 1.0) - 1.0 / l_far
        x: Tensor = t * t
        series: Tensor = 0.5 + t * (
            1.0 / 3.0 + x * (16.0 / 45.0 + x * (704.0 / 945.0 + x * 27392.0 / 14175.0))
        )
        return lucid.where(near, series, closed)

    @property
    @override
    def variance(self) -> Tensor:
        r"""Variance :math:`\lambda(\lambda-1)/(1-2\lambda)^2 + 1/\ell^2`.

        :math:`\ell = \operatorname{logit}\lambda`.  The two terms are each
        of order :math:`1/(1-2\lambda)^2` and cancel near
        :math:`\lambda = \tfrac12`, so there the Taylor series
        :math:`\tfrac1{12} - \tfrac{x}{15} - \tfrac{128 x^2}{945}
        - \tfrac{4864 x^3}{14175}` in :math:`x = (\lambda - \tfrac12)^2`
        is used.  Nothing is clamped: the variance is positive for every
        :math:`\lambda`, and a negative value would be a defect to see,
        not to hide.

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions import ContinuousBernoulli
        >>> ContinuousBernoulli(probs=lucid.tensor([0.2, 0.5, 0.7])).variance
        tensor([0.0759, 0.08333, 0.08043])
        """
        t, near, p_far, l_far = self._moment_operands()
        closed: Tensor = p_far * (p_far - 1.0) / (1.0 - 2.0 * p_far) ** 2 + 1.0 / (l_far * l_far)
        x: Tensor = t * t
        series: Tensor = 1.0 / 12.0 - x * (1.0 / 15.0 + x * (128.0 / 945.0 + x * 4864.0 / 14175.0))
        return lucid.where(near, series, closed)

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Reparameterised sample via the closed-form icdf."""
        shape = self._extended_shape(sample_shape)
        u: Tensor = lucid.rand(*shape, dtype=self._param.dtype, device=self._param.device)
        u = lucid.clamp(u, min=1e-6, max=1.0 - 1e-6)
        # Broadcast distribution parameters to the full sample shape so that
        # lucid.where (which requires same-shape operands) always succeeds.
        zeros_b: Tensor = lucid.zeros(*shape, dtype=self._param.dtype, device=self._param.device)
        p: Tensor = self._probs + zeros_b
        l: Tensor = self._logits + zeros_b
        d: Tensor = 2.0 * p - 1.0  # ∈ (−1, 1) \ {0}
        abs_d: Tensor = d.abs()
        eps: float = 1e-4
        safe_d: Tensor = lucid.where(abs_d < eps, lucid.full_like(d, eps), d)
        safe_l: Tensor = lucid.where(abs_d < eps, lucid.full_like(l, eps), l)
        # icdf: x = log(1 + u · (2p−1) / (1−p)) / logit(p)
        stable_x: Tensor = (1.0 + u * safe_d / (1.0 - p)).log() / safe_l
        near_x: Tensor = u  # uniform limit at p = ½
        return lucid.clamp(lucid.where(abs_d < eps, near_x, stable_x), min=0.0, max=1.0)

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        """``x · l − softplus(l) + log C(p)``.

        ``softplus`` rather than ``log(1 + exp(l))``, which reached ``inf``
        at ``l ≈ 89`` in float32; and the logit of a ``probs`` of exactly
        0 or 1 is taken one epsilon inside ``[0, 1]`` (:attr:`logits`), so it
        is never multiplied by a zero ``x`` while infinite.
        """
        l: Tensor = self._logits
        return value * l - l.softplus() + self._log_normalizer()

    def _flipped(self) -> tuple[Tensor, Tensor, Tensor]:
        r"""``(m, flip, zero)``: :math:`m = -|\ell|`, where :math:`\ell > 0`,
        and where :math:`\ell = 0` — with :math:`m` there replaced by -1.

        :math:`X \mapsto 1 - X` takes :math:`\lambda` to :math:`1 - \lambda`
        and :math:`\ell` to :math:`-\ell`, so :meth:`cdf` and :meth:`icdf`
        evaluate only a non-positive logit, where :math:`e^{\ell x}` cannot
        overflow, and reflect the answer for a positive one.  At
        :math:`\ell = 0` the distribution is uniform and the closed forms
        are 0/0; the stand-in keeps that NaN out of the gradient.
        """
        l: Tensor = self._logits
        zero = l == 0.0
        m = lucid.where(zero, -1.0, -l.abs())
        return m, l > 0.0, zero

    @override
    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the Continuous Bernoulli.

        .. math::

            F(x) = \frac{\lambda^x (1 - \lambda)^{1 - x} + \lambda - 1}
                        {2\lambda - 1}
                 = \frac{\operatorname{expm1}(\ell x)}{\operatorname{expm1}(\ell)}

        with :math:`\ell = \operatorname{logit}\lambda`, and :math:`x` at
        :math:`\lambda = \tfrac12`.  The second form has no cancellation as
        :math:`\lambda \to \tfrac12`, where the reference framework
        replaces the first with :math:`x` inside
        :math:`|\lambda - \tfrac12| < 10^{-3}`; there the two differ by up
        to :math:`|\ell|/8`, and this is the exact value.

        Returns probabilities in :math:`[0, 1]`; 0 at and below 0, 1 at and
        above 1.

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions import ContinuousBernoulli
        >>> ContinuousBernoulli(probs=0.7).cdf(lucid.tensor(0.5))
        tensor(0.3956)
        """
        m, flip, zero = self._flipped()
        x = lucid.where(flip, 1.0 - value, value)
        r = lucid.expm1(x * m) / lucid.expm1(m)
        inner = lucid.where(zero, value, lucid.where(flip, 1.0 - r, r))
        inner = lucid.where(value <= 0.0, 0.0, inner)
        return lucid.where(value >= 1.0, 1.0, inner)

    @override
    def icdf(self, value: Tensor) -> Tensor:
        r"""Quantile function of the Continuous Bernoulli.

        .. math::

            Q(p) = \frac{\operatorname{log1p}\!\left(p \operatorname{expm1}(\ell)\right)}{\ell}

        for :math:`\ell < 0`, reflected through :math:`Q_\ell(p) =
        1 - Q_{-\ell}(1 - p)` for :math:`\ell > 0`, and :math:`p` at
        :math:`\ell = 0`.

        Returns quantiles in :math:`[0, 1]`.

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions import ContinuousBernoulli
        >>> ContinuousBernoulli(probs=0.7).icdf(lucid.tensor(0.3956))
        tensor(0.5)
        """
        m, flip, zero = self._flipped()
        p = lucid.where(flip, 1.0 - value, value)
        r = (p * lucid.expm1(m)).log1p() / m
        inner = lucid.where(zero, value, lucid.where(flip, 1.0 - r, r))
        # The ends are named and the rest held in [0, 1]: the closed form
        # rounds there — icdf(0) was -7.4e-11 at probs = 1 - 1e-7 and
        # 1.1e-16 at probs = 0.501.
        inner = lucid.where(value <= 0.0, 0.0, inner.clamp(0.0, 1.0))
        return lucid.where(value >= 1.0, 1.0, inner)

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Continuous Bernoulli (in nats).

        .. math::

            H = \operatorname{softplus}(\ell) - \ell\,\mathbb E[X] - \ln C(\lambda)

        :math:`-\mathbb E[\log p(X)]` with
        :math:`\log p(x) = x\ell - \operatorname{softplus}(\ell) + \ln C`,
        so it reuses :attr:`mean` and the normaliser, each already evaluated
        by its series near :math:`\lambda = \tfrac12`.

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``; 0 at
            :math:`\lambda = \tfrac12`, where the distribution is uniform.

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions import ContinuousBernoulli
        >>> ContinuousBernoulli(probs=0.5).entropy()
        tensor(0.)
        """
        l: Tensor = self._logits
        return l.softplus() - l * self.mean - self._log_normalizer()
