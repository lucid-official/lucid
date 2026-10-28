"""Discrete ``Bernoulli`` and ``Geometric``."""

from collections.abc import Sequence
from typing import Self, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import (
    Constraint,
    _positive_unit_interval,
    boolean,
    nonnegative_integer,
    real,
    unit_interval,
)

# No class here uses it any more, but the last release's surface lists it
# under this module (``lucid/test/audit/released_surface.json``), and a name
# leaves only through a deprecation cycle.
from lucid.distributions.constraints import open_unit_interval  # noqa: F401
from lucid.distributions.distribution import Distribution, ExponentialFamily


from lucid.distributions._util import _clamp_probs, _lazy_param, _set_probs_or_logits
from lucid.special import xlog1py


def _probs_to_logits(probs: Tensor) -> Tensor:
    r"""Convert a probability tensor to logits (log-odds).

    Computes :math:`\ell = \log(p) - \log(1 - p)` element-wise.

    Parameters
    ----------
    probs : Tensor
        Probabilities in :math:`(0, 1)`.

    Returns
    -------
    Tensor
        Log-odds with the same shape as ``probs``.
    """
    return probs.log() - (1.0 - probs).log()


def _logits_to_probs(logits: Tensor) -> Tensor:
    r"""Convert logits (log-odds) to probabilities via the sigmoid transform.

    Computes :math:`p = \sigma(\ell) = 1 / (1 + e^{-\ell})`.

    Parameters
    ----------
    logits : Tensor
        Real-valued log-odds.

    Returns
    -------
    Tensor
        Probabilities in :math:`(0, 1)`, same shape as ``logits``.
    """
    return logits.sigmoid()


class Bernoulli(ExponentialFamily):
    r"""Bernoulli distribution over :math:`\{0, 1\}`.

    The Bernoulli is the simplest discrete distribution: a single coin
    flip with success probability :math:`p \in [0, 1]`.  It models a binary
    outcome and is the building block of the Binomial (sum of ``n`` IID
    Bernoullis), the Categorical (its multinomial generalisation), and most
    classification likelihoods in supervised learning.

    Specify *exactly one* of ``probs`` or ``logits``; the other is derived
    lazily via the sigmoid / logit transform so there is no redundant
    storage and the parameterisation chosen at construction remains the
    canonical one for autograd.

    Parameters
    ----------
    probs : Tensor or float, optional
        Success probability :math:`p \in [0, 1]`.  Mutually exclusive
        with ``logits``.
    logits : Tensor or float, optional
        Log-odds :math:`\ell = \log(p / (1 - p)) \in \mathbb{R}`.
        Mutually exclusive with ``probs``.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability mass function on :math:`x \in \{0, 1\}`:

    .. math::

        P(X = k \mid p) = p^k (1 - p)^{1-k}

    Moments:

    .. math::

        \mathbb{E}[X] = p, \qquad
        \mathrm{Var}[X] = p(1 - p), \qquad
        H[X] = -p\log p - (1-p)\log(1-p)

    The variance is maximised at :math:`p = 0.5` (maximum uncertainty) and
    vanishes at the degenerate endpoints :math:`p \in \{0, 1\}`.

    **Relation to other distributions:**

    * :math:`\mathrm{Binomial}(n, p)` is the sum of :math:`n` independent
      :math:`\mathrm{Bernoulli}(p)` draws.
    * :math:`\mathrm{Geometric}(p)` counts Bernoulli failures before the
      first success.
    * :math:`\mathrm{Categorical}` generalises Bernoulli to :math:`K > 2`
      categories.

    Conjugate prior: :class:`~lucid.distributions.Beta` — observing
    :math:`k` successes out of :math:`n` updates
    ``Beta(α, β) → Beta(α + k, β + n - k)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Bernoulli
    >>> d = Bernoulli(probs=0.7)
    >>> d.mean
    tensor(0.7)
    >>> d.sample((4,)).shape
    (4,)
    >>> d.log_prob(lucid.tensor(1.0))
    tensor(-0.3567)
    >>> Bernoulli(logits=0.0).probs  # the other parameterisation, on demand
    tensor(0.5)
    """

    arg_constraints = {"probs": unit_interval, "logits": real}
    support: Constraint | None = boolean
    has_enumerate_support = True
    _is_logits: bool

    def __init__(
        self,
        probs: Tensor | float | None = None,
        logits: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        r"""Construct a Bernoulli distribution.

        Exactly one of ``probs`` or ``logits`` must be supplied.  The other
        is derived on demand via the sigmoid / logit transform, so there is no
        redundant storage.

        Parameters
        ----------
        probs : Tensor | float | None, optional
            Success probability :math:`p \in [0, 1]`.  Mutually exclusive
            with ``logits``.
        logits : Tensor | float | None, optional
            Log-odds :math:`\ell = \log(p / (1-p)) \in \mathbb{R}`.
            Mutually exclusive with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` / ``logits`` are provided.

        Examples
        --------
        >>> from lucid.distributions import Bernoulli
        >>> d = Bernoulli(probs=0.7)
        >>> d.mean
        tensor(0.7)
        >>> d2 = Bernoulli(logits=0.0)
        >>> d2.mean  # p = sigmoid(0), derived from the logits on demand
        tensor(0.5)
        """
        shape = _set_probs_or_logits(self, "Bernoulli", probs, logits)
        super().__init__(batch_shape=shape, event_shape=(), validate_args=validate_args)

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Bernoulli` with batch dimensions expanded to ``batch_shape``."""
        return self._expand_probs_or_logits(Bernoulli, batch_shape, _instance)

    @_lazy_param
    def probs(self) -> Tensor:
        """Success probability — as given, or ``sigmoid(logits)`` on access."""
        return _logits_to_probs(self.logits)

    @_lazy_param
    def logits(self) -> Tensor:
        r"""Log-odds :math:`\log(p/(1-p))` — as given, or derived on access.

        Derived from ``probs`` clamped one epsilon inside :math:`[0, 1]`, as
        the reference framework derives it, so a degenerate probability
        gives a large finite logit rather than an infinite one.
        """
        return _probs_to_logits(_clamp_probs(self.probs))

    @property
    def param(self) -> Tensor:
        """The canonical parameter used at construction time.

        Returns the logits tensor when the distribution was constructed with
        ``logits``, or the probs tensor when constructed with ``probs``.
        This is used by ``ExponentialFamily`` machinery to access the
        sufficient statistic parameter without forcing a conversion.

        Returns
        -------
        Tensor
            Either ``self.logits`` or ``self.probs``, depending on which was
            provided at construction.

        Examples
        --------
        >>> d = Bernoulli(probs=0.3)
        >>> d.param  # returns self.probs
        tensor(0.3)
        """
        return self.logits if self._is_logits else self.probs

    @property
    def _probs(self) -> Tensor:
        """Lazily resolved success probability tensor.

        Returns ``self.probs`` directly when the distribution was constructed
        with ``probs``; otherwise computes ``sigmoid(self.logits)`` on demand.
        """
        return self.probs if not self._is_logits else _logits_to_probs(self.logits)

    @property
    def _logits(self) -> Tensor:
        """Lazily resolved log-odds tensor.

        Returns ``self.logits`` directly when the distribution was constructed
        with ``logits``; otherwise the log-odds of ``probs`` held one epsilon
        inside :math:`[0, 1]`, as :attr:`logits` derives them.  Unclamped, a
        probability of exactly 0 or 1 gave an infinite logit, and
        ``log_prob`` then multiplied it by a zero count.
        """
        return self.logits

    @override
    @property
    def mean(self) -> Tensor:
        r"""Expected value of the distribution.

        For a Bernoulli random variable :math:`X \in \{0, 1\}` with
        success probability :math:`p`:

        .. math::

            E[X] = p

        Returns
        -------
        Tensor
            Success probability :math:`p`, shape ``batch_shape``.

        Examples
        --------
        >>> Bernoulli(probs=0.3).mean
        tensor(0.3)
        """
        return self._probs

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the Bernoulli distribution: ``1`` if :math:`p > 0.5`, ``0``
        if :math:`p < 0.5`, and ``nan`` at :math:`p = 0.5`, where both
        outcomes are equally likely.
        """
        p = self._probs
        mode = (p >= 0.5).to(p.dtype)
        return lucid.where(p == 0.5, float("nan"), mode)

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the distribution.

        For a Bernoulli random variable :math:`X` with success probability
        :math:`p`:

        .. math::

            \operatorname{Var}[X] = p(1 - p)

        The variance is maximised at :math:`p = 0.5` and equals zero at the
        degenerate extremes :math:`p \in \{0, 1\}`.

        Returns
        -------
        Tensor
            Variance :math:`p(1-p)`, shape ``batch_shape``.

        Examples
        --------
        >>> Bernoulli(probs=0.5).variance
        tensor(0.25)
        """
        p = self._probs
        return p * (1.0 - p)

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the Bernoulli distribution.

        Samples by comparing uniform noise :math:`U \sim \text{Uniform}(0,1)`
        against the success probability :math:`p`:

        .. math::

            X = \mathbf{1}[U < p]

        The returned tensor has integer-valued entries in :math:`\{0, 1\}`
        stored as floating-point and is **detached** from the autograd graph.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  The full output
            shape is ``sample_shape + batch_shape``.  Default is ``()``.

        Returns
        -------
        Tensor
            Binary samples of shape ``sample_shape + batch_shape``.

        Examples
        --------
        >>> lucid.manual_seed(0)
        >>> d = Bernoulli(probs=0.6)
        >>> x = d.sample((1000,))
        >>> x.shape
        (1000,)
        >>> bool((x.mean() - 0.6).abs() < 0.05)  # the sample mean approaches p
        True
        """
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self._probs.dtype, device=self._probs.device)
        p = self._probs + lucid.zeros(shape, dtype=self._probs.dtype, device=self._probs.device)
        out = (u < p).to(self._probs.dtype)
        return out.detach()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability of ``value`` under the Bernoulli distribution.

        Uses the logits form, with the softplus evaluated so it cannot
        overflow:

        .. math::

            \log p(x \mid \ell) = x \cdot \ell - \operatorname{softplus}(\ell),
            \qquad \operatorname{softplus}(\ell) = \log(1 + e^\ell)

        where :math:`\ell = \log(p / (1-p))` is the log-odds.  This is
        equivalent to the cross-entropy form
        :math:`x \log p + (1-x) \log(1-p)`; written with ``log(1 + exp(ℓ))``
        it reached ``inf`` at :math:`\ell \approx 89` in float32 and scored
        a certain outcome as ``-inf``.  A probability of exactly 0 or 1 is
        held one epsilon inside :math:`[0, 1]` first, as the reference
        framework does, so its logit is large but finite.

        Parameters
        ----------
        value : Tensor
            Observed values.  Should be in :math:`\{0, 1\}`.

        Returns
        -------
        Tensor
            Element-wise log-probabilities, shape ``batch_shape``.

        Examples
        --------
        >>> d = Bernoulli(probs=0.7)
        >>> d.log_prob(lucid.tensor(1.0))  # log(0.7)
        tensor(-0.3567)
        """
        l = self._logits
        return value * l - l.softplus()

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Bernoulli distribution (in nats).

        .. math::

            H(X) = -p \log p - (1-p) \log(1-p)

        Computed in the softplus form, which stays finite at any logit:

        .. math::

            H = \operatorname{softplus}(\ell) - p \cdot \ell

        where :math:`\ell` is the log-odds.  The entropy is maximised at
        :math:`p = 0.5` (maximum uncertainty) and is zero at the degenerate
        cases :math:`p \in \{0, 1\}`.

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Examples
        --------
        >>> Bernoulli(probs=0.5).entropy()  # log(2) ≈ 0.693
        tensor(0.6931)
        """
        l = self._logits
        return l.softplus() - self._probs * l

    @override
    def enumerate_support(self, expand: bool = True) -> Tensor:
        r"""The two outcomes :math:`\{0, 1\}` along axis 0, in the parameter's
        floating dtype and device."""
        param = self.param
        values = lucid.arange(0, 2, 1, dtype=param.dtype, device=param.device)
        return self._expand_support(values, expand)

    @override
    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (self.logits,)

    @override
    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        (x,) = natural_params
        return x.softplus()


class Geometric(Distribution):
    r"""Geometric distribution: number of failures before the first success.

    Models the number of failed Bernoulli trials :math:`X \in \{0, 1, 2, \ldots\}`
    preceding the first success in an unbounded sequence of independent
    trials with success probability :math:`p`.  This is the *shifted*
    convention (counting failures); the alternative convention counts trials
    including the success and starts at :math:`1`.

    Parameters
    ----------
    probs : Tensor or float, optional
        Success probability :math:`p \in (0, 1]` per trial.  Mutually
        exclusive with ``logits``.
    logits : Tensor or float, optional
        Log-odds :math:`\ell = \log(p/(1-p))`; ``probs`` is then derived
        from it on access.  Mutually exclusive with ``probs``.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability mass function:

    .. math::

        P(X = k \mid p) = p (1 - p)^k, \quad k = 0, 1, 2, \ldots

    Moments:

    .. math::

        \mathbb{E}[X] = \frac{1 - p}{p}, \qquad
        \mathrm{Var}[X] = \frac{1 - p}{p^2}

    **Memoryless property** — the Geometric is the *only* discrete
    distribution with the memoryless property:

    .. math::

        P(X \geq m + n \mid X \geq m) = P(X \geq n)

    Continuous analogue: :class:`~lucid.distributions.Exponential`
    (the only continuous memoryless distribution).

    The expected number of *trials* (including the success) is
    :math:`1/p`; entropy grows without bound as :math:`p \to 0` and is
    zero at :math:`p = 1`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Geometric
    >>> d = Geometric(probs=0.25)
    >>> d.mean  # (1 - p)/p = 3.0
    tensor(3.)
    >>> d.sample((4,)).shape
    (4,)
    >>> d.log_prob(lucid.tensor(2.0))  # 2 log(0.75) + log(0.25)
    tensor(-1.962)
    """

    # ``(0, 1]``: ``p = 1`` is a certain first success (all mass at 0), and
    # ``p = 0`` — no trial ever succeeds — is not a distribution.
    arg_constraints = {"probs": _positive_unit_interval, "logits": real}
    support: Constraint | None = nonnegative_integer

    def __init__(
        self,
        probs: Tensor | float | None = None,
        logits: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        r"""Construct a Geometric distribution.

        Exactly one of ``probs`` or ``logits`` must be provided.  The one
        given is stored; the other is derived from it on access.

        Parameters
        ----------
        probs : Tensor | float | None, optional
            Success probability :math:`p \in (0, 1]` of a single Bernoulli
            trial.  Mutually exclusive with ``logits``.
        logits : Tensor | float | None, optional
            Log-odds :math:`\ell = \log(p / (1-p))`.  Mutually exclusive
            with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` / ``logits`` are provided.

        Notes
        -----
        The support is :math:`\{0, 1, 2, \ldots\}` representing the number
        of *failures* before the first success.  The PMF is:

        .. math::

            P(X = k) = (1 - p)^k \, p, \quad k = 0, 1, 2, \ldots

        Examples
        --------
        >>> from lucid.distributions import Geometric
        >>> d = Geometric(probs=0.25)
        >>> d.mean  # E[X] = (1-p)/p = 3
        tensor(3.)
        """
        # Whichever was given is stored and the other derived on access.
        # Converting ``logits`` up front also validated the converted value,
        # so ``logits=20.0`` — a probability that rounds to 1 in float32 —
        # was refused as outside the open interval.
        shape = _set_probs_or_logits(self, "Geometric", probs, logits)
        super().__init__(batch_shape=shape, event_shape=(), validate_args=validate_args)

    @_lazy_param
    def probs(self) -> Tensor:
        """Success probability — as given, or ``sigmoid(logits)`` on access."""
        return _logits_to_probs(self.logits)

    @_lazy_param
    def logits(self) -> Tensor:
        r"""Log-odds :math:`\log(p/(1-p))` — as given, or derived on access.

        Derived from ``probs`` clamped one epsilon inside :math:`[0, 1]`, as
        the reference framework derives it.
        """
        return _probs_to_logits(_clamp_probs(self.probs))

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Geometric` with batch dimensions expanded to ``batch_shape``."""
        return self._expand_probs_or_logits(Geometric, batch_shape, _instance)

    @override
    @property
    def mean(self) -> Tensor:
        r"""Expected number of failures before the first success.

        .. math::

            E[X] = \frac{1 - p}{p}

        Returns
        -------
        Tensor
            Mean :math:`(1-p)/p`, shape ``batch_shape``.

        Examples
        --------
        >>> Geometric(probs=0.5).mean
        tensor(1.)
        """
        return (1.0 - self.probs) / self.probs

    @override
    @property
    def mode(self) -> Tensor:
        """Mode of the Geometric distribution: ``0`` failures, for every ``p``.

        :math:`P(X = k) = (1 - p)^k p` decreases in :math:`k`.
        """
        return lucid.zeros_like(self.probs)

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the number of failures before the first success.

        .. math::

            \operatorname{Var}[X] = \frac{1 - p}{p^2}

        Returns
        -------
        Tensor
            Variance :math:`(1-p)/p^2`, shape ``batch_shape``.

        Examples
        --------
        >>> Geometric(probs=0.5).variance
        tensor(2.)
        """
        return (1.0 - self.probs) / (self.probs * self.probs)

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the Geometric distribution.

        Uses the inverse-CDF trick: if :math:`U \sim \text{Uniform}(0, 1)`,

        .. math::

            X = \left\lfloor \frac{\log U}{\log(1 - p)} \right\rfloor

        A small epsilon clamp guards against :math:`U = 0` which would
        produce :math:`-\infty`.  The returned tensor is **detached** since
        the Geometric is discrete.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  The full output
            shape is ``sample_shape + batch_shape``.  Default is ``()``.

        Returns
        -------
        Tensor
            Non-negative integer samples of shape ``sample_shape + batch_shape``.

        Examples
        --------
        >>> lucid.manual_seed(0)
        >>> d = Geometric(probs=0.5)
        >>> x = d.sample((1000,))
        >>> x.shape
        (1000,)
        >>> bool((x.mean() - 1.0).abs() < 0.2)  # the sample mean approaches (1-p)/p
        True
        """
        # icdf-trick: floor(log(U) / log(1 - p)).  Detached — discrete.
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.probs.dtype, device=self.probs.device)
        # Avoid u=0 which would give -inf.  Lucid's rand draws from
        # [0, 1) so u=0 is admissible — clamp upward by epsilon.
        u_safe = u.clip(1e-7, 1.0 - 1e-7)
        return (u_safe.log() / (1.0 - self.probs).log()).floor().detach()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability of ``value`` under the Geometric distribution.

        .. math::

            \log P(X = k) = k \log(1-p) + \log p

        with :math:`k \log(1-p)` taken as ``0`` at :math:`k = 0`, so
        :math:`p = 1` scores its only outcome as :math:`\log 1 = 0` rather
        than ``0 · (-inf) = NaN``.

        Parameters
        ----------
        value : Tensor
            Non-negative integer outcomes :math:`k \in \{0, 1, 2, \ldots\}`.

        Returns
        -------
        Tensor
            Element-wise log-probabilities, shape ``batch_shape``.

        Examples
        --------
        >>> d = Geometric(probs=0.5)
        >>> d.log_prob(lucid.tensor(0.0))  # log(0.5) ≈ -0.693
        tensor(-0.6931)
        """
        p = self.probs
        return xlog1py(value, -p) + p.log()

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Geometric distribution (in nats).

        .. math::

            H(X) = \frac{-(1-p)\log(1-p) - p \log p}{p}
                 = \frac{\operatorname{softplus}(\ell) - p\,\ell}{p}

        The numerator is the Bernoulli entropy, evaluated in its logits form
        so that :math:`p = 1` gives (almost exactly) ``0`` rather than
        ``0 · log 0 = NaN``.

        The entropy grows without bound as :math:`p \to 0` (more uncertainty
        over many possible outcomes) and is zero at :math:`p = 1` (certain
        success on first trial).

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Examples
        --------
        >>> Geometric(probs=0.5).entropy()  # 2 log(2)
        tensor(1.386)
        """
        p = self.probs
        l = self.logits
        return (l.softplus() - p * l) / p
