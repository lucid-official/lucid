"""``Exponential``, ``Laplace``, and ``Cauchy`` — icdf-sampled families."""

import math
from collections.abc import Sequence
from typing import Self, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import (
    Constraint,
    nonnegative,
    positive,
    real,
)
from lucid.distributions.distribution import Distribution, ExponentialFamily


from lucid.distributions._util import (
    _as_tensor,
    _broadcast_pair,
)


class Exponential(ExponentialFamily):
    r"""Exponential distribution on :math:`[0, \infty)`.

    Continuous distribution describing the waiting time between events of
    a Poisson process with rate :math:`\lambda`.  It is the continuous
    analogue of the :class:`~lucid.distributions.Geometric` distribution
    and the unique continuous distribution with the *memoryless* property.

    Parameters
    ----------
    rate : Tensor or float
        Rate parameter :math:`\lambda > 0`.  The mean of the distribution
        is :math:`1/\lambda`.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density on :math:`x \geq 0`:

    .. math::

        p(x; \lambda) = \lambda e^{-\lambda x}

    Moments:

    .. math::

        \mathbb{E}[X] = \frac{1}{\lambda}, \qquad
        \mathrm{Var}[X] = \frac{1}{\lambda^2}, \qquad
        H[X] = 1 - \log \lambda

    **Memoryless property:**

    .. math::

        P(X > s + t \mid X > s) = P(X > t)

    Special cases / relations:

    * :math:`\mathrm{Exponential}(\lambda) = \mathrm{Gamma}(1, \lambda)`
    * The sum of :math:`k` IID :math:`\mathrm{Exponential}(\lambda)`
      variables is :math:`\mathrm{Gamma}(k, \lambda)` (Erlang).
    * :math:`-\log U / \lambda` with :math:`U \sim \mathrm{Uniform}(0, 1)`
      is the inverse-CDF sampler used by :meth:`rsample`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Exponential
    >>> d = Exponential(rate=2.0)
    >>> d.mean  # 1/rate
    tensor(0.5)
    >>> d.rsample((4,)).shape
    (4,)
    >>> d.log_prob(lucid.tensor(1.0))
    tensor(-1.307)
    """

    arg_constraints = {"rate": positive}
    # ``[0, ∞)``: the density is ``λ`` at 0.  ``positive`` made
    # ``log_prob(0)`` — validated below — raise.
    support: Constraint | None = nonnegative
    has_rsample = True

    def __init__(
        self,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Construct an Exponential distribution.

        Parameters
        ----------
        rate : Tensor | float
            Rate parameter :math:`\lambda > 0`.  The mean of the
            distribution is :math:`1/\lambda`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        The Exponential distribution with rate :math:`\lambda` has PDF:

        .. math::

            p(x; \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0

        It is the continuous analogue of the Geometric distribution and
        describes the waiting time between events in a Poisson process.

        Examples
        --------
        >>> from lucid.distributions import Exponential
        >>> d = Exponential(rate=2.0)
        >>> d.mean  # 1/rate = 0.5
        tensor(0.5)
        """
        self.rate = _as_tensor(rate)
        super().__init__(
            batch_shape=tuple(self.rate.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Exponential` with batch dimensions expanded to ``batch_shape``."""
        new = self._get_checked_instance(Exponential, _instance)
        batch_shape_t = tuple(batch_shape)
        new.rate = self.rate.expand(batch_shape_t)
        super(Exponential, new).__init__(batch_shape_t, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @override
    @property
    def mean(self) -> Tensor:
        r"""Expected value of the Exponential distribution.

        .. math::

            E[X] = \frac{1}{\lambda}

        Returns
        -------
        Tensor
            Mean :math:`1/\lambda`, shape ``batch_shape``.

        Examples
        --------
        >>> Exponential(rate=4.0).mean
        tensor(0.25)
        """
        return 1.0 / self.rate

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the Exponential distribution.

        The Exponential distribution is monotonically decreasing on
        :math:`[0, \infty)`, so the mode is always zero regardless of the
        rate parameter.

        .. math::

            \text{mode} = 0

        Returns
        -------
        Tensor
            Zero tensor of shape ``batch_shape``.

        Examples
        --------
        >>> Exponential(rate=5.0).mode
        tensor(0.)
        """
        return lucid.zeros_like(self.rate)

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the Exponential distribution.

        .. math::

            \operatorname{Var}[X] = \frac{1}{\lambda^2}

        Returns
        -------
        Tensor
            Variance :math:`1/\lambda^2`, shape ``batch_shape``.

        Examples
        --------
        >>> Exponential(rate=2.0).variance
        tensor(0.25)
        """
        return 1.0 / (self.rate * self.rate)

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Reparameterised sample via the inverse-CDF method.

        Uses the transform :math:`X = -\log(1 - U) / \lambda` where
        :math:`U \sim \text{Uniform}(0, 1)`.  This is an exact
        reparameterisation — gradients flow through :math:`\lambda`.

        .. math::

            X = -\frac{\log(1 - U)}{\lambda}

        Subtracting from 1 (rather than directly using :math:`U`) ensures
        the argument to :math:`\log` is bounded away from zero.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  Default is ``()``.

        Returns
        -------
        Tensor
            Reparameterised samples of shape ``sample_shape + batch_shape``.

        Examples
        --------
        >>> lucid.manual_seed(0)
        >>> d = Exponential(rate=1.0)
        >>> x = d.rsample((500,))
        >>> x.shape
        (500,)
        >>> bool((x.mean() - 1.0).abs() < 0.15)  # the sample mean approaches 1/rate
        True
        """
        # Inverse CDF sampling via U ~ Uniform(0, 1):  x = -log(1 - U) / rate.
        # Subtract from 1 to keep U away from 0 (which would give -inf).
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.rate.dtype, device=self.rate.device)
        return -(1.0 - u).log() / self.rate

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Exponential distribution.

        .. math::

            \log p(x; \lambda) = \log \lambda - \lambda x

        Parameters
        ----------
        value : Tensor
            Non-negative real values :math:`x \geq 0`.

        Returns
        -------
        Tensor
            Element-wise log-densities, shape ``batch_shape``.

        Examples
        --------
        >>> d = Exponential(rate=1.0)
        >>> d.log_prob(lucid.tensor(1.0))  # -1.0
        tensor(-1.)
        """
        return self.rate.log() - self.rate * value

    @override
    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the Exponential distribution.

        .. math::

            F(x; \lambda) = 1 - e^{-\lambda x}, \quad x \geq 0

        Parameters
        ----------
        value : Tensor
            Non-negative real values at which to evaluate the CDF.

        Returns
        -------
        Tensor
            CDF values in :math:`[0, 1]`, shape ``batch_shape``.

        Examples
        --------
        >>> Exponential(rate=1.0).cdf(lucid.tensor(1.0))  # 1 - e^-1 ≈ 0.632
        tensor(0.6321)
        >>> # below the support: refused under validation, 0 without it
        >>> Exponential(rate=1.0, validate_args=False).cdf(lucid.tensor(-1.0))
        tensor(0.)

        Notes
        -----
        Clamped below the support.  ``1 - exp(-λx)`` is the formula *on*
        :math:`[0, \infty)`; evaluated at a negative argument it grows
        without bound and returned -1.718 for ``x = -1``, while the line
        above this one promises a value in :math:`[0, 1]`.  A CDF is
        defined on the whole line and is zero to the left of its support;
        with validation on (the default) such a value raises ``ValueError``
        instead.
        """
        below = value < 0.0
        return lucid.where(below, lucid.zeros_like(value), 1.0 - (-self.rate * value).exp())

    @override
    def icdf(self, value: Tensor) -> Tensor:
        r"""Inverse CDF (quantile function) of the Exponential distribution.

        .. math::

            F^{-1}(u; \lambda) = -\frac{\log(1 - u)}{\lambda}

        Parameters
        ----------
        value : Tensor
            Probability values :math:`u \in [0, 1)`.

        Returns
        -------
        Tensor
            Quantiles in :math:`[0, \infty)`, shape ``batch_shape``.

        Examples
        --------
        >>> Exponential(rate=1.0).icdf(lucid.tensor(0.5))  # log(2) ≈ 0.693
        tensor(0.6931)

        Notes
        -----
        ``-log1p(-u)`` rather than ``-log(1 - u)``: for a small ``u`` the
        subtraction rounds ``1 - u`` to 1 and the quantile to 0 — every
        ``u`` below 6e-8 in float32 — where the reference answers ``u``.
        """
        return -(-value).log1p() / self.rate

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Exponential distribution (in nats).

        .. math::

            H(X) = 1 - \log \lambda

        The entropy decreases as the rate increases (concentrated
        distributions have lower uncertainty).

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Examples
        --------
        >>> Exponential(rate=1.0).entropy()  # 1 - log(1) = 1.0
        tensor(1.)
        """
        return 1.0 - self.rate.log()

    @override
    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (-self.rate,)

    @override
    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        (x,) = natural_params
        return -(-x).log()


class Laplace(Distribution):
    r"""Laplace (double-exponential) distribution on :math:`\mathbb{R}`.

    Symmetric continuous distribution composed of two mirrored Exponential
    densities placed back-to-back at the location :math:`\mu`.  It is the
    maximum-entropy distribution with a given mean and a given mean
    absolute deviation, and is the prior whose negative log-density induces
    the L1 (Lasso) penalty.

    Parameters
    ----------
    loc : Tensor or float
        Location parameter :math:`\mu \in \mathbb{R}` (mean = median = mode).
    scale : Tensor or float
        Scale / diversity parameter :math:`b > 0`.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density:

    .. math::

        p(x; \mu, b) = \frac{1}{2b} \exp\!\left(-\frac{|x - \mu|}{b}\right)

    Moments:

    .. math::

        \mathbb{E}[X] = \mu, \qquad
        \mathrm{Var}[X] = 2 b^2, \qquad
        H[X] = 1 + \log(2b)

    The Laplace has heavier tails than the Normal: its kurtosis is 6 vs. 3
    for the Normal, and its tails decay exponentially rather than
    quadratically in log-space.

    Sampling uses the closed-form inverse CDF:

    .. math::

        X = \mu - b \operatorname{sgn}(U - \tfrac{1}{2})
            \log(1 - 2|U - \tfrac{1}{2}|),
        \quad U \sim \mathrm{Uniform}(0, 1)

    so :meth:`rsample` is exact and gradient-friendly through both
    :math:`\mu` and :math:`b`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Laplace
    >>> d = Laplace(loc=0.0, scale=1.0)
    >>> d.mean
    tensor(0.)
    >>> d.rsample((4,)).shape
    (4,)
    >>> d.log_prob(lucid.tensor(0.0))
    tensor(-0.6931)
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
        r"""Construct a Laplace distribution.

        Parameters
        ----------
        loc : Tensor | float
            Location parameter :math:`\mu \in \mathbb{R}` (the mean and
            median of the distribution).
        scale : Tensor | float
            Scale (diversity) parameter :math:`b > 0`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        The Laplace distribution has PDF:

        .. math::

            p(x; \mu, b) = \frac{1}{2b} \exp\!\left(-\frac{|x - \mu|}{b}\right)

        It is also called the double-exponential distribution because it
        resembles two mirrored Exponential densities placed back-to-back at
        the location :math:`\mu`.  It is frequently used in robust
        statistics and as the prior inducing L1 (Lasso) regularisation.

        Examples
        --------
        >>> from lucid.distributions import Laplace
        >>> d = Laplace(loc=0.0, scale=1.0)
        >>> d.mean
        tensor(0.)
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
        """Return a new :class:`Laplace` with batch dimensions expanded to ``batch_shape``."""
        new = self._get_checked_instance(Laplace, _instance)
        batch_shape_t = tuple(batch_shape)
        new.loc = self.loc.expand(batch_shape_t)
        new.scale = self.scale.expand(batch_shape_t)
        super(Laplace, new).__init__(batch_shape_t, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @override
    @property
    def mean(self) -> Tensor:
        r"""Expected value of the Laplace distribution.

        .. math::

            E[X] = \mu

        Returns
        -------
        Tensor
            Location parameter :math:`\mu`, shape ``batch_shape``.

        Examples
        --------
        >>> Laplace(loc=3.0, scale=1.0).mean
        tensor(3.)
        """
        return self.loc

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the Laplace distribution: the location :math:`\mu`."""
        return self.loc

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the Laplace distribution.

        .. math::

            \operatorname{Var}[X] = 2b^2

        Returns
        -------
        Tensor
            Variance :math:`2b^2`, shape ``batch_shape``.

        Examples
        --------
        >>> Laplace(loc=0.0, scale=1.0).variance
        tensor(2.)
        """
        return 2.0 * self.scale * self.scale

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Reparameterised sample via the inverse-CDF method.

        Uses the closed-form quantile function:

        .. math::

            X = \mu - b \operatorname{sgn}(u) \log(1 - 2|u|)

        where :math:`u = U - 0.5` and :math:`U \sim \text{Uniform}(0, 1)`.
        Gradients flow through both :math:`\mu` and :math:`b`.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  Default is ``()``.

        Returns
        -------
        Tensor
            Reparameterised samples of shape ``sample_shape + batch_shape``.

        Examples
        --------
        >>> d = Laplace(loc=0.0, scale=1.0)
        >>> x = d.rsample((200,))
        """
        # icdf: loc - scale · sign(u-0.5) · log(1 - 2|u-0.5|).
        # Reparameterise as 2U-1 then split into sign+magnitude.
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.loc.dtype, device=self.loc.device) - 0.5
        sign_u = u.sign()
        return self.loc - self.scale * sign_u * (1.0 - 2.0 * u.abs()).log()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Laplace distribution.

        .. math::

            \log p(x; \mu, b) = -\frac{|x - \mu|}{b} - \log(2b)

        Parameters
        ----------
        value : Tensor
            Real-valued observations.

        Returns
        -------
        Tensor
            Element-wise log-densities, shape ``batch_shape``.

        Examples
        --------
        >>> Laplace(loc=0.0, scale=1.0).log_prob(lucid.tensor(0.0))
        tensor(-0.6931)
        """
        return -((value - self.loc).abs()) / self.scale - (2.0 * self.scale).log()

    @override
    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the Laplace distribution.

        .. math::

            F(x; \mu, b) = \frac{1}{2} + \frac{1}{2}
            \operatorname{sgn}(x - \mu)
            \left(1 - e^{-|x-\mu|/b}\right)

        Parameters
        ----------
        value : Tensor
            Real values at which to evaluate the CDF.

        Returns
        -------
        Tensor
            CDF values in :math:`[0, 1]`, shape ``batch_shape``.

        Notes
        -----
        The sign in front of the second term is ``+``.  It was ``-`` in
        both the formula above and the line below it, which makes the
        function *decreasing*: ``cdf(-3)`` answered 0.975 and ``cdf(3)``
        answered 0.025, exactly reversed.  It stayed in :math:`[0, 1]` and
        was symmetric about ``0.5`` at the median, so nothing looked odd
        until the audit checked that a CDF increases.
        """
        z = (value - self.loc) / self.scale
        return 0.5 + 0.5 * z.sign() * (1.0 - (-z.abs()).exp())

    @override
    def icdf(self, value: Tensor) -> Tensor:
        r"""Quantile function of the Laplace distribution.

        .. math::

            Q(p) = \mu - b \operatorname{sgn}(p - \tfrac12)
                   \ln\!\left(1 - 2\left|p - \tfrac12\right|\right)

        Returns quantiles; :math:`\mp\infty` at :math:`p = 0` and :math:`1`.

        Examples
        --------
        >>> Laplace(loc=0.0, scale=1.0).icdf(lucid.tensor(0.75))  # ln 2
        tensor(0.6931)
        """
        term = value - 0.5
        return self.loc - self.scale * term.sign() * (-2.0 * term.abs()).log1p()

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Laplace distribution (in nats).

        .. math::

            H(X) = 1 + \log(2b)

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Examples
        --------
        >>> Laplace(loc=0.0, scale=1.0).entropy()  # 1 + log(2) ≈ 1.693
        tensor(1.693)
        """
        return 1.0 + (2.0 * self.scale).log()


class Cauchy(Distribution):
    r"""Cauchy distribution on :math:`\mathbb{R}` — heavy-tailed, all moments undefined.

    The canonical example of a *pathological* heavy-tailed distribution:
    its tails decay so slowly that no integer moment (mean, variance,
    skew, kurtosis) is defined.  The Cauchy is the Student's-t
    distribution with one degree of freedom and arises naturally as the
    ratio of two independent standard Normals.

    Parameters
    ----------
    loc : Tensor or float
        Location parameter :math:`x_0 \in \mathbb{R}` (median and mode).
        The "mean" does *not* exist.
    scale : Tensor or float
        Scale parameter :math:`\gamma > 0` (half-width at half-maximum).
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density:

    .. math::

        p(x; x_0, \gamma) =
            \frac{1}{\pi \gamma}
            \left[1 + \left(\frac{x - x_0}{\gamma}\right)^2\right]^{-1}

    Cumulative distribution:

    .. math::

        F(x; x_0, \gamma) =
            \frac{1}{\pi} \arctan\!\left(\frac{x - x_0}{\gamma}\right) + \frac{1}{2}

    Properties (no moments — *all* are undefined or infinite):

    * **Median / mode**: :math:`x_0`
    * **Mean / variance**: undefined (integrals diverge)
    * **Entropy**: :math:`H[X] = \log(4 \pi \gamma)`

    **Sample mean does not converge** — the strong law of large numbers
    fails for IID Cauchy variates.  Any finite-sample average of Cauchy
    draws is itself Cauchy-distributed with the same parameters.

    **Stability**: a sum of independent Cauchy variables remains Cauchy
    (it is an :math:`\alpha`-stable distribution with stability index 1).

    Reparameterised sampling uses the inverse-CDF
    :math:`X = x_0 + \gamma \tan(\pi(U - \tfrac{1}{2}))`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Cauchy
    >>> d = Cauchy(loc=0.0, scale=1.0)
    >>> d.rsample((4,)).shape
    (4,)
    >>> d.log_prob(lucid.tensor(0.0))
    tensor(-1.145)
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
        r"""Construct a Cauchy distribution.

        Parameters
        ----------
        loc : Tensor | float
            Location parameter (median and mode) :math:`x_0 \in \mathbb{R}`.
        scale : Tensor | float
            Scale (half-width at half-maximum) parameter :math:`\gamma > 0`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        The Cauchy distribution has PDF:

        .. math::

            p(x; x_0, \gamma) = \frac{1}{\pi \gamma}
            \left[1 + \left(\frac{x - x_0}{\gamma}\right)^2\right]^{-1}

        The Cauchy is the Student's t-distribution with one degree of
        freedom.  Because all moments of order :math:`\geq 1` are undefined,
        it is a canonical example of a heavy-tailed distribution where the
        sample mean does *not* converge to any value.

        Examples
        --------
        >>> from lucid.distributions import Cauchy
        >>> d = Cauchy(loc=0.0, scale=1.0)
        >>> x = d.rsample((1000,))
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
        """Return a new :class:`Cauchy` with batch dimensions expanded to ``batch_shape``."""
        new = self._get_checked_instance(Cauchy, _instance)
        batch_shape_t = tuple(batch_shape)
        new.loc = self.loc.expand(batch_shape_t)
        new.scale = self.scale.expand(batch_shape_t)
        super(Cauchy, new).__init__(batch_shape_t, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Reparameterised sample via the inverse-CDF method.

        Uses the quantile function of the Cauchy distribution:

        .. math::

            X = x_0 + \gamma \tan\!\bigl(\pi (U - \tfrac{1}{2})\bigr)

        where :math:`U \sim \text{Uniform}(0, 1)`.  Gradients flow
        through both :math:`x_0` and :math:`\gamma`.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  Default is ``()``.

        Returns
        -------
        Tensor
            Reparameterised samples of shape ``sample_shape + batch_shape``.

        Examples
        --------
        >>> d = Cauchy(loc=0.0, scale=1.0)
        >>> x = d.rsample((500,))
        """
        # icdf: loc + scale · tan(π·(U − 0.5)).
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + self.scale * (math.pi * (u - 0.5)).tan()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Cauchy distribution.

        .. math::

            \log p(x; x_0, \gamma) = -\log \pi - \log \gamma
            - \log\!\left(1 + \left(\frac{x - x_0}{\gamma}\right)^2\right)

        Parameters
        ----------
        value : Tensor
            Real-valued observations.

        Returns
        -------
        Tensor
            Element-wise log-densities, shape ``batch_shape``.

        Examples
        --------
        >>> Cauchy(loc=0.0, scale=1.0).log_prob(lucid.tensor(0.0))
        tensor(-1.145)
        """
        z = (value - self.loc) / self.scale
        return -math.log(math.pi) - self.scale.log() - (1.0 + z * z).log()

    @override
    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the Cauchy distribution.

        .. math::

            F(x; x_0, \gamma) = \frac{1}{\pi}
            \arctan\!\left(\frac{x - x_0}{\gamma}\right) + \frac{1}{2}

        Parameters
        ----------
        value : Tensor
            Real values at which to evaluate the CDF.

        Returns
        -------
        Tensor
            CDF values in :math:`(0, 1)`, shape ``batch_shape``.
        """
        z = (value - self.loc) / self.scale
        return 0.5 + z.arctan() / math.pi

    @override
    def icdf(self, value: Tensor) -> Tensor:
        r"""Quantile function of the Cauchy distribution.

        .. math::

            Q(p) = x_0 + \gamma \tan\!\left(\pi\left(p - \tfrac12\right)\right)

        Returns quantiles.  At :math:`p = 0` and :math:`1` the tangent of
        :math:`\mp\pi/2` is evaluated, which is a large finite number rather
        than an infinity, as in the reference framework.

        Examples
        --------
        >>> Cauchy(loc=0.0, scale=1.0).icdf(lucid.tensor(0.75))
        tensor(1.)
        """
        return (math.pi * (value - 0.5)).tan() * self.scale + self.loc

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Cauchy distribution (in nats).

        .. math::

            H(X) = \log(4 \pi \gamma)

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Examples
        --------
        >>> Cauchy(loc=0.0, scale=1.0).entropy()  # log(4π) ≈ 2.531
        tensor(2.531)
        """
        return math.log(4.0 * math.pi) + self.scale.log()

    # A moment that does not exist is still an answer, and the shape of the
    # answer matters: the reference returns NaN or infinity rather than
    # raising, so a caller can carry it through an expression and test for
    # it at the end.  Raising ``NotImplementedError`` reads as "Lucid has
    # not got round to this" when the truth is that the integral diverges.

    @override
    @property
    def mean(self) -> Tensor:
        r"""Mean of the Cauchy distribution — undefined, returns NaN.

        The defining integral :math:`\int x\, p(x)\, dx` does not converge
        for the Cauchy: the tails decay as :math:`x^{-2}`, so the two halves
        are each infinite and their difference has no value.  This is not a
        limitation of the parameters — no Cauchy has a mean.

        Returns
        -------
        Tensor
            Tensor of ``float('nan')`` with shape ``batch_shape``.
        """
        return lucid.full_like(self.scale, float("nan"))

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the Cauchy distribution: the location :math:`x_0`."""
        return self.loc

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the Cauchy distribution — infinite.

        Unlike the mean, the second moment integral diverges in one
        direction only, so the answer is :math:`+\infty` rather than NaN.

        Returns
        -------
        Tensor
            Tensor of ``inf`` with shape ``batch_shape``.
        """
        return lucid.full_like(self.scale, float("inf"))

    @override
    @property
    def stddev(self) -> Tensor:
        r"""Standard deviation — infinite, as the square root of :attr:`variance`."""
        return lucid.full_like(self.scale, float("inf"))
