"""Univariate ``Normal`` and ``LogNormal``."""

import math
from collections.abc import Sequence
from typing import Self, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import (
    Constraint,
    positive,
    real,
)
from lucid.distributions.distribution import Distribution, ExponentialFamily

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


from lucid.distributions._util import (
    _as_tensor,
    _broadcast_pair,
    _image_cdf,
    _image_icdf,
)


class Normal(ExponentialFamily):
    r"""Univariate Gaussian (Normal) distribution :math:`\mathcal{N}(\mu, \sigma^2)`.

    The Normal distribution is arguably the most important distribution in
    probability and statistics.  Its probability density function is the
    iconic bell curve:

    .. math::

        p(x \mid \mu, \sigma) =
            \frac{1}{\sqrt{2\pi\sigma^2}}
            \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),
        \quad x \in \mathbb{R}

    where :math:`\mu` is the *location* (mean) and :math:`\sigma > 0`
    is the *scale* (standard deviation).

    Parameters
    ----------
    loc : Tensor or float
        Mean :math:`\mu` of the distribution.  May be any real value.
        Can be a scalar, a 1-D tensor of means (producing a batch), or
        any higher-dimensional tensor.
    scale : Tensor or float
        Standard deviation :math:`\sigma > 0`.  Must be strictly positive.
        Broadcast-compatible with ``loc``.
    validate_args : bool or None, optional
        Enable parameter and support validation.  Default ``None``.

    Attributes
    ----------
    loc : Tensor
        Broadcast-expanded mean tensor.
    scale : Tensor
        Broadcast-expanded standard-deviation tensor.

    Notes
    -----
    **Location-scale family**

    The Normal distribution is closed under linear transformations: if
    :math:`X \sim \mathcal{N}(\mu, \sigma^2)` then
    :math:`aX + b \sim \mathcal{N}(a\mu + b, a^2\sigma^2)`.
    In particular the *standard Normal* :math:`Z = (X - \mu)/\sigma`
    satisfies :math:`Z \sim \mathcal{N}(0, 1)`.

    **Central Limit Theorem**

    By the CLT, the (rescaled) sum of :math:`n` i.i.d. random variables
    with finite mean and variance converges in distribution to a Normal.
    This explains why the Normal appears naturally across virtually all
    domains of science and engineering.

    **Reparameterisation**

    ``has_rsample = True`` — samples are drawn via the standard-Normal
    reparameterisation trick:

    .. math::

        X = \mu + \sigma \varepsilon, \quad
        \varepsilon \sim \mathcal{N}(0, 1)

    so gradients flow through :math:`\mu` and :math:`\sigma`.

    **Exponential family**

    The Normal is an exponential-family distribution with natural
    parameters :math:`\eta_1 = \mu/\sigma^2`,
    :math:`\eta_2 = -1/(2\sigma^2)` and sufficient statistics
    :math:`T(x) = (x,\, x^2)`.

    Examples
    --------
    >>> import lucid.distributions as dist
    >>> d = dist.Normal(loc=0.0, scale=1.0)
    >>> d.mean
    tensor(0.)
    >>> d.variance
    tensor(1.)
    >>> x = d.rsample((5,))   # 5 standard-Normal draws
    >>> lp = d.log_prob(x)    # evaluate log-density at those points
    """

    arg_constraints = {"loc": real, "scale": positive}
    support: Constraint | None = real
    has_rsample = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Construct a Normal distribution.

        Parameters
        ----------
        loc : Tensor or float
            Mean :math:`\mu`.
        scale : Tensor or float
            Standard deviation :math:`\sigma > 0`.
        validate_args : bool or None, optional
            Validate parameter constraints on construction.
        """
        self.loc = _as_tensor(loc)
        self.scale = _as_tensor(scale)
        self.loc, self.scale = _broadcast_pair(self.loc, self.scale)
        super().__init__(
            batch_shape=tuple(self.loc.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Normal` with batch dimensions expanded to ``batch_shape``."""
        new = self._get_checked_instance(Normal, _instance)
        batch_shape_t = tuple(batch_shape)
        new.loc = self.loc.expand(batch_shape_t)
        new.scale = self.scale.expand(batch_shape_t)
        super(Normal, new).__init__(batch_shape_t, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @override
    @property
    def mean(self) -> Tensor:
        r"""Mean of the Normal distribution.

        For :math:`X \sim \mathcal{N}(\mu, \sigma^2)`,

        .. math::

            \mathbb{E}[X] = \mu

        Returns
        -------
        Tensor
            The ``loc`` parameter tensor (shape ``batch_shape``).
        """
        return self.loc

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the Normal distribution.

        The Normal density is symmetric and unimodal; its unique maximum
        is at :math:`x = \mu`.

        .. math::

            \text{mode}(X) = \mu

        Returns
        -------
        Tensor
            The ``loc`` parameter tensor (shape ``batch_shape``).
        """
        return self.loc

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the Normal distribution.

        .. math::

            \text{Var}[X] = \sigma^2

        Returns
        -------
        Tensor
            Element-wise square of ``scale`` (shape ``batch_shape``).
        """
        return self.scale * self.scale

    @override
    @property
    def stddev(self) -> Tensor:
        r"""Standard deviation of the Normal distribution.

        .. math::

            \text{std}(X) = \sigma

        Overrides the base-class default (which would compute
        ``variance.sqrt()``) for a cheaper, exact result.

        Returns
        -------
        Tensor
            The ``scale`` parameter tensor (shape ``batch_shape``).
        """
        return self.scale

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw reparameterised samples via the location-scale trick.

        Samples are generated as

        .. math::

            X = \mu + \sigma \varepsilon, \quad
            \varepsilon \sim \mathcal{N}(0, 1)

        Because the stochasticity is isolated in :math:`\varepsilon`
        (which does not depend on the parameters), gradients flow back
        through both ``loc`` (:math:`\mu`) and ``scale``
        (:math:`\sigma`).

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Shape prefix for the batch of samples.

        Returns
        -------
        Tensor
            Samples of shape ``sample_shape + batch_shape``, attached
            to the autograd graph.
        """
        shape = self._extended_shape(sample_shape)
        eps = lucid.randn(*shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + self.scale * eps

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the Normal distribution.

        .. math::

            \log p(x \mid \mu, \sigma)
            = -\frac{(x - \mu)^2}{2\sigma^2}
              - \log \sigma
              - \frac{1}{2}\log(2\pi)

        This is computed in a numerically stable form using
        ``log_scale`` rather than squaring and dividing.

        Parameters
        ----------
        value : Tensor
            Observation(s) :math:`x \in \mathbb{R}`.

        Returns
        -------
        Tensor
            Log-density at ``value``, shape
            ``broadcast(value, batch_shape)``.
        """
        var = self.variance
        log_scale = self.scale.log()
        return -((value - self.loc) ** 2) / (2.0 * var) - log_scale - _LOG_SQRT_2PI

    @override
    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the Normal distribution.

        Expressed in terms of the Gauss error function:

        .. math::

            F(x) = \frac{1}{2}\left[1 + \operatorname{erf}
                   \!\left(\frac{x - \mu}{\sigma\sqrt{2}}\right)\right]

        Parameters
        ----------
        value : Tensor
            Evaluation point(s) :math:`x`.

        Returns
        -------
        Tensor
            Probabilities in :math:`(0, 1)`.
        """
        return 0.5 * (1.0 + lucid.erf((value - self.loc) / (self.scale * math.sqrt(2.0))))

    @override
    def icdf(self, value: Tensor) -> Tensor:
        r"""Inverse CDF (quantile function / probit function) of the Normal.

        .. math::

            Q(p) = \mu + \sigma\,\sqrt{2}\,\operatorname{erfinv}(2p - 1)

        Parameters
        ----------
        value : Tensor
            Probability values :math:`p \in (0, 1)`.

        Returns
        -------
        Tensor
            Corresponding quantiles :math:`x = Q(p)`.
        """
        return self.loc + self.scale * lucid.erfinv(2.0 * value - 1.0) * math.sqrt(2.0)

    @override
    def entropy(self) -> Tensor:
        r"""Shannon differential entropy of the Normal distribution.

        The Normal maximises entropy among all distributions with fixed
        mean and variance.  The closed-form entropy is

        .. math::

            H[X] = \frac{1}{2}\ln(2\pi e\sigma^2)
                 = \frac{1}{2} + \frac{1}{2}\ln(2\pi) + \ln\sigma

        Measured in nats.

        Returns
        -------
        Tensor
            Entropy values of shape ``batch_shape``.
        """
        return 0.5 + 0.5 * math.log(2.0 * math.pi) + self.scale.log()

    @override
    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        var = self.variance
        return (self.loc / var, -0.5 / var)

    @override
    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        x, y = natural_params
        return -0.25 * (x * x) / y + 0.5 * (-math.pi / y).log()


class LogNormal(Distribution):
    r"""Log-Normal distribution: :math:`X = \exp(Y)` where :math:`Y \sim \mathcal{N}(\mu, \sigma^2)`.

    If a random variable :math:`X` has a Log-Normal distribution then its
    natural logarithm :math:`\ln X` is Normally distributed.  The
    probability density function is

    .. math::

        p(x \mid \mu, \sigma) =
            \frac{1}{x\,\sigma\sqrt{2\pi}}
            \exp\!\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right),
        \quad x > 0

    Parameters
    ----------
    loc : Tensor or float
        Mean :math:`\mu` of the underlying Normal :math:`\ln X`.
    scale : Tensor or float
        Standard deviation :math:`\sigma > 0` of the underlying Normal.
    validate_args : bool or None, optional
        Enable constraint validation.  Default ``None``.

    Attributes
    ----------
    loc : Tensor
        Mean of the latent Normal.
    scale : Tensor
        Standard deviation of the latent Normal.

    Notes
    -----
    The Log-Normal arises naturally whenever a quantity is the product of
    many independent positive factors (multiplicative growth), just as the
    Normal arises from additive contributions.  Applications include
    particle-size distributions, financial asset prices, and reaction
    times.

    **Mean and variance** of :math:`X = e^Y`:

    .. math::

        \mathbb{E}[X] = e^{\mu + \sigma^2/2}, \qquad
        \text{Var}[X] = (e^{\sigma^2} - 1)\,e^{2\mu + \sigma^2}

    Note that both grow super-exponentially in :math:`\sigma`.

    **Mode**:

    .. math::

        \text{mode}(X) = e^{\mu - \sigma^2}

    The mode is always less than the mean, reflecting right-skewness.

    Examples
    --------
    >>> import lucid.distributions as dist
    >>> d = dist.LogNormal(loc=0.0, scale=1.0)
    >>> x = d.rsample((100,))
    >>> bool((x > 0).all())  # support is strictly positive
    True
    """

    arg_constraints = {"loc": real, "scale": positive}
    support: Constraint | None = positive
    has_rsample = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Construct a Log-Normal distribution.

        Parameters
        ----------
        loc : Tensor or float
            Mean of the latent Normal :math:`\ln X`.
        scale : Tensor or float
            Standard deviation of the latent Normal, :math:`\sigma > 0`.
        validate_args : bool or None, optional
            Validate parameter constraints on construction.
        """
        self._base = Normal(loc, scale, validate_args=False)
        self.loc = self._base.loc
        self.scale = self._base.scale
        super().__init__(
            batch_shape=tuple(self.loc.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`LogNormal` with batch dimensions expanded to ``batch_shape``."""
        new = self._get_checked_instance(LogNormal, _instance)
        batch_shape_t = tuple(batch_shape)
        new._base = self._base.expand(batch_shape_t)
        new.loc = new._base.loc
        new.scale = new._base.scale
        super(LogNormal, new).__init__(batch_shape_t, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @override
    @property
    def mean(self) -> Tensor:
        r"""Mean of the Log-Normal distribution.

        .. math::

            \mathbb{E}[X] = e^{\mu + \sigma^2/2}

        Returns
        -------
        Tensor
            Shape ``batch_shape``.
        """
        return (self.loc + 0.5 * self.scale * self.scale).exp()

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the Log-Normal distribution.

        .. math::

            \text{mode}(X) = e^{\mu - \sigma^2}

        The mode is strictly less than the mean, reflecting the
        right-skewed nature of the distribution.

        Returns
        -------
        Tensor
            Shape ``batch_shape``.
        """
        return (self.loc - self.scale * self.scale).exp()

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the Log-Normal distribution.

        .. math::

            \text{Var}[X] = (e^{\sigma^2} - 1)\,e^{2\mu + \sigma^2}

        Returns
        -------
        Tensor
            Shape ``batch_shape``.
        """
        s2 = self.scale * self.scale
        return (s2.exp() - 1.0) * (2.0 * self.loc + s2).exp()

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw reparameterised samples.

        Samples are obtained by exponentiating Normal samples from the
        underlying :class:`Normal` base distribution:

        .. math::

            X = \exp(\mu + \sigma \varepsilon), \quad
            \varepsilon \sim \mathcal{N}(0,1)

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Shape prefix for the batch of samples.

        Returns
        -------
        Tensor
            Strictly positive samples of shape
            ``sample_shape + batch_shape``.
        """
        return self._base.rsample(sample_shape).exp()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the Log-Normal distribution.

        By the change-of-variables formula:

        .. math::

            \log p(x) = \log p_{\mathcal{N}}(\ln x) - \ln x

        where :math:`p_{\mathcal{N}}` is the density of the underlying
        Normal distribution.

        Parameters
        ----------
        value : Tensor
            Strictly positive observation(s) :math:`x > 0`.

        Returns
        -------
        Tensor
            Log-density at ``value``.
        """
        log_v = value.log()
        return self._base.log_prob(log_v) - log_v

    @override
    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the Log-Normal distribution.

        .. math::

            F(x) = \Phi\!\left(\frac{\ln x - \mu}{\sigma}\right),
            \quad x > 0

        The latent Normal's CDF at :math:`\ln x`; 0 at and below 0.

        Returns probabilities in :math:`[0, 1]`.

        Examples
        --------
        >>> LogNormal(0.0, 1.0).cdf(lucid.tensor(1.0))  # the median is e^mu
        tensor(0.5)
        """
        # ``log`` of a clamped value: a negative one scores ``log 0 = -inf``
        # and so 0, rather than NaN, when validation is off.
        return _image_cdf(self._base, value.clamp(min=0.0).log())

    @override
    def icdf(self, value: Tensor) -> Tensor:
        r"""Quantile function of the Log-Normal distribution.

        .. math::

            Q(p) = \exp\!\left(\mu + \sigma \Phi^{-1}(p)\right)

        Returns quantiles in :math:`[0, \infty]`.

        Examples
        --------
        >>> LogNormal(0.0, 1.0).icdf(lucid.tensor(0.5))
        tensor(1.)
        """
        return _image_icdf(self._base, value).exp()

    @override
    def entropy(self) -> Tensor:
        r"""Shannon differential entropy of the Log-Normal distribution.

        .. math::

            H[X] = \mu + H[\mathcal{N}(0,\sigma^2)]
                 = \mu + \frac{1}{2} + \frac{1}{2}\ln(2\pi) + \ln\sigma

        where the extra :math:`\mu` term accounts for the Jacobian of
        the exponential transformation.

        Returns
        -------
        Tensor
            Entropy values of shape ``batch_shape`` (in nats).
        """
        return self._base.entropy() + self.loc
