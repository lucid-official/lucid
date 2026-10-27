"""Discrete count distributions: ``Poisson``, ``Binomial``,
``NegativeBinomial``.

All three are pure-Lucid composites — sampling reuses
``lucid.poisson`` and the per-distribution helpers, log-prob is
closed-form via lgamma.
"""

from collections.abc import Sequence
from typing import Self, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import _as_tensor
from lucid.distributions._util import _broadcast_pair
from lucid.distributions._util import _clamp_probs, _lazy_param
from lucid.distributions.bernoulli import _logits_to_probs, _probs_to_logits
from lucid.distributions.constraints import (
    Constraint,
    half_open_interval,
    integer_interval,
    nonnegative,
    nonnegative_integer,
    real,
    unit_interval,
)
from lucid.distributions.distribution import Distribution, ExponentialFamily


class Poisson(ExponentialFamily):
    r"""Poisson distribution over the non-negative integers.

    Models the number of events occurring in a fixed interval when events
    happen independently at a constant mean rate :math:`\lambda`.  The
    Poisson distribution is the limiting case of a Binomial as
    :math:`n \to \infty` and :math:`p \to 0` with :math:`np = \lambda`
    fixed.

    Parameters
    ----------
    rate : Tensor | float
        Mean rate :math:`\lambda \geq 0`.  Determines both the mean and the
        variance of the distribution.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    rate : Tensor
        The mean rate parameter :math:`\lambda`.

    Notes
    -----
    **PMF** (probability mass function):

    .. math::

        P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!},
        \quad k \in \{0, 1, 2, \ldots\}

    **Moments**:

    - Mean: :math:`E[X] = \lambda`
    - Variance: :math:`\operatorname{Var}[X] = \lambda`
    - Mode: :math:`\lfloor \lambda \rfloor`

    The log-probability is computed as
    :math:`k \log \lambda - \lambda - \log\Gamma(k+1)`, which is
    numerically stable via the ``lgamma`` function.

    The Poisson has no closed-form entropy (it is an infinite sum over all
    non-negative integers), so :meth:`entropy` raises ``NotImplementedError``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Poisson
    >>> dist = Poisson(rate=3.5)
    >>> samples = dist.sample((100,))
    >>> samples.shape
    (100,)
    >>> # log-probability at k=4
    >>> dist.log_prob(lucid.tensor(4.0))
    tensor(-1.667)
    """

    # ``λ = 0`` is the point mass at 0, and every closed form below holds
    # there (``xlogy`` scores it).
    arg_constraints = {"rate": nonnegative}
    support: Constraint | None = nonnegative_integer

    def __init__(self, rate: Tensor | float, validate_args: bool | None = None) -> None:
        r"""Initialise a Poisson distribution.

        Parameters
        ----------
        rate : Tensor | float
            Mean rate parameter :math:`\lambda \geq 0`.  Determines both the
            mean and the variance.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
        self.rate = _as_tensor(rate)
        super().__init__(
            batch_shape=tuple(self.rate.shape), event_shape=(), validate_args=validate_args
        )

    @override
    @property
    def mean(self) -> Tensor:
        r"""Mean of the Poisson distribution: :math:`E[X] = \lambda`.

        Returns
        -------
        Tensor
            Mean values equal to ``rate``, shape ``batch_shape``.
        """
        return self.rate

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the Poisson distribution: :math:`\lfloor \lambda \rfloor`.

        When :math:`\lambda` is an integer, both :math:`\lambda` and
        :math:`\lambda - 1` are modes; this property returns
        :math:`\lfloor \lambda \rfloor`.

        Returns
        -------
        Tensor
            Mode values of shape ``batch_shape``.
        """
        # ``floor(rate)`` — when rate is integer, both floor and floor−1
        # are modes; we follow the reference framework and take floor.
        return self.rate.floor()

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the Poisson distribution: :math:`\operatorname{Var}[X] = \lambda`.

        Returns
        -------
        Tensor
            Variance values equal to ``rate``, shape ``batch_shape``.
        """
        return self.rate

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the Poisson distribution.

        Delegates to :func:`lucid.poisson`, which uses the Knuth exact
        algorithm for small :math:`\lambda` and a Normal approximation
        with rounding for large :math:`\lambda`.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Non-negative integer samples of shape
            ``(*sample_shape, *batch_shape)``.
        """
        # Top-level ``lucid.poisson`` does the Knuth + Normal-approx work
        # using the Lucid Philox PRNG, so ``manual_seed`` controls the
        # stream end-to-end.
        rate_b: Tensor = self.rate + lucid.zeros(
            self._extended_shape(sample_shape), dtype=self.rate.dtype, device=self.rate.device
        )
        return lucid.poisson(rate_b)

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability of the given counts under the Poisson distribution.

        .. math::

            \log P(X = k) = k \log \lambda - \lambda - \log\Gamma(k + 1)

        :math:`k \log \lambda` is taken as ``0`` at :math:`k = 0`, so a
        zero rate scores a zero count as :math:`\log 1 = 0` rather than
        ``0 · (-inf) = NaN``.

        Parameters
        ----------
        value : Tensor
            Non-negative integer counts :math:`k \geq 0`.

        Returns
        -------
        Tensor
            Log-probability values of the same shape as ``value``.
        """
        return lucid.xlogy(value, self.rate) - self.rate - lucid.lgamma(value + 1.0)

    @override
    def entropy(self) -> Tensor:
        """Entropy of the Poisson distribution (not available in closed form).

        The Poisson entropy is an infinite sum over all non-negative integers
        and has no simple closed form.

        Raises
        ------
        NotImplementedError
            Always — use numerical approximation if needed.
        """
        # Closed-form entropy is an infinite sum; the reference framework
        # returns NotImplemented.  We follow.
        raise NotImplementedError("Poisson.entropy: no closed form available")

    @override
    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (self.rate.log(),)

    @override
    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        (x,) = natural_params
        return x.exp()

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Poisson` instance with parameters expanded to ``batch_shape``."""
        new = self._get_checked_instance(Poisson, _instance)
        batch_shape = tuple(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        return self._finish_expand(new, batch_shape)


class Binomial(Distribution):
    r"""Binomial distribution over the number of successes in n independent trials.

    ``Binomial(total_count=n, probs=p)`` models the count of successes when
    each of :math:`n` i.i.d. Bernoulli trials has success probability
    :math:`p`.  Parameterisation is via either ``probs`` (in :math:`[0, 1]`)
    or ``logits`` (the log-odds :math:`\log(p/(1-p)) \in \mathbb{R}`);
    exactly one must be supplied.

    Parameters
    ----------
    total_count : Tensor | int, optional
        Number of trials :math:`n \geq 0`.  Default is ``1`` (reduces to
        Bernoulli).
    probs : Tensor | float | None, optional
        Success probability :math:`p \in [0, 1]`.  Mutually exclusive with
        ``logits``.
    logits : Tensor | float | None, optional
        Log-odds :math:`l = \log(p / (1-p)) \in \mathbb{R}`.  Mutually
        exclusive with ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    total_count : Tensor
        Number of trials :math:`n`, broadcast against the probability.
    probs : Tensor
        Success probability — stored when given, otherwise derived from
        ``logits`` on access.
    logits : Tensor
        Log-odds — stored when given, otherwise derived from ``probs`` on
        access.

    Notes
    -----
    **PMF**:

    .. math::

        P(X = k) = \binom{n}{k} p^k (1-p)^{n-k},
        \quad k \in \{0, 1, \ldots, n\}

    **Log-PMF** via logits (numerically stable form):

    .. math::

        \log P(X = k) = \log\binom{n}{k} + k \, l - n \log(1 + e^l)

    where :math:`l = \operatorname{logit}(p)` and the binomial coefficient is
    evaluated via :math:`\log \Gamma(n+1) - \log\Gamma(k+1) - \log\Gamma(n-k+1)`.

    **Moments**:

    - Mean: :math:`E[X] = np`
    - Variance: :math:`\operatorname{Var}[X] = np(1-p)`

    **Sampling strategy**: for :math:`n \leq 25` Bernoulli draws are summed
    exactly along a dedicated axis.  For larger :math:`n` a Normal
    approximation is used with the result rounded and clamped to
    :math:`[0, n]`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Binomial
    >>> dist = Binomial(total_count=10, probs=0.3)
    >>> samples = dist.sample((50,))
    >>> samples.shape
    (50,)
    >>> # PMF at k=3
    >>> dist.log_prob(lucid.tensor(3.0)).exp()
    tensor(0.2668)
    """

    arg_constraints = {"total_count": nonnegative_integer, "probs": unit_interval, "logits": real}
    has_enumerate_support = True

    _SMALL_N_CUTOFF: int = 25

    def __init__(
        self,
        total_count: Tensor | int = 1,
        probs: Tensor | float | None = None,
        logits: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a Binomial distribution.

        Parameters
        ----------
        total_count : Tensor | int, optional
            Number of trials :math:`n \geq 0`.  Default is ``1`` (reduces to
            Bernoulli).
        probs : Tensor | float | None, optional
            Success probability :math:`p \in [0, 1]`.  Mutually exclusive with
            ``logits``.
        logits : Tensor | float | None, optional
            Log-odds :math:`l = \log(p / (1-p)) \in \mathbb{R}`.  Mutually
            exclusive with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` and ``logits`` are provided.
        """
        if (probs is None) == (logits is None):
            raise ValueError("Binomial: pass exactly one of `probs` or `logits`.")
        self._is_logits = probs is None
        if probs is not None:
            param = _as_tensor(probs)
        else:
            param = _as_tensor(logits)  # type: ignore[arg-type]
        # Both sides broadcast, not only the count.  The batch used to be
        # sized from the probability alone, so ``Binomial([2, 5, 9], 0.4)``
        # reported ``batch_shape == ()`` and ``sample`` raised a shape
        # mismatch for a perfectly good batch of three.
        self.total_count, param = _broadcast_pair(_as_tensor(total_count), param)
        if self._is_logits:
            self.logits = param
        else:
            self.probs = param
        super().__init__(
            batch_shape=tuple(param.shape), event_shape=(), validate_args=validate_args
        )

    @_lazy_param
    def probs(self) -> Tensor:
        """Success probability — as given, or ``sigmoid(logits)`` on access."""
        return _logits_to_probs(self.logits)

    @_lazy_param
    def logits(self) -> Tensor:
        r"""Log-odds :math:`\log(p/(1-p))` — as given, or derived on access.

        Derived from ``probs`` clamped one epsilon inside :math:`[0, 1]`, as
        the reference framework derives it.  :meth:`log_prob` does not use
        this attribute: it keeps the exact, possibly infinite, log-odds that
        its degenerate-probability guards are written around.
        """
        return _probs_to_logits(_clamp_probs(self.probs))

    @override
    @property
    def support(self) -> Constraint:
        r"""Support of the Binomial distribution: :math:`\{0, 1, \ldots, n\}`.

        Returns
        -------
        Constraint
            ``integer_interval(0, total_count)`` — the bound is the
            ``total_count`` tensor, so each batch member is held to its own
            :math:`n`.
        """
        return integer_interval(0, self.total_count)

    @property
    def _probs(self) -> Tensor:
        """Success probability :math:`p`, computed from logits if necessary.

        Returns
        -------
        Tensor
            Probability values in :math:`[0, 1]`, shape ``batch_shape``.
        """
        return self.probs if not self._is_logits else _logits_to_probs(self.logits)

    @property
    def _logits(self) -> Tensor:
        r"""Log-odds :math:`\log(p/(1-p))`, computed from probs if necessary.

        Returns
        -------
        Tensor
            Log-odds values in :math:`\mathbb{R}`, shape ``batch_shape``.
        """
        return self.logits if self._is_logits else _probs_to_logits(self.probs)

    @override
    @property
    def mean(self) -> Tensor:
        """Mean of the Binomial distribution: :math:`E[X] = np`.

        Returns
        -------
        Tensor
            Mean values of shape ``batch_shape``.
        """
        return self.total_count * self._probs

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the Binomial distribution: :math:`\min(\lfloor (n+1) p \rfloor, n)`."""
        n = self.total_count
        return lucid.minimum(((n + 1.0) * self._probs).floor(), n)

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the Binomial distribution: :math:`\operatorname{Var}[X] = np(1-p)`.

        Returns
        -------
        Tensor
            Variance values of shape ``batch_shape``.
        """
        p: Tensor = self._probs
        return self.total_count * p * (1.0 - p)

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the Binomial distribution.

        Uses an exact strategy for small :math:`n` (sum of Bernoulli draws)
        and a Normal approximation with rounding for large :math:`n`.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Non-negative integer samples in :math:`\{0, 1, \ldots, n\}`,
            shape ``(*sample_shape, *batch_shape)``.
        """
        # Strategy:
        #   * If max(total_count) ≤ cutoff: sum of Bernoullis along an
        #     extra axis — exact.
        #   * Otherwise: Normal approximation rounded to the integer
        #     range [0, total_count].
        max_n: int = int(self.total_count.max().item())
        out_shape: tuple[int, ...] = self._extended_shape(sample_shape)
        device = self._probs.device
        dtype = self._probs.dtype

        if max_n <= self._SMALL_N_CUTOFF:
            # Build (max_n, *out_shape) uniform draws, mask by trial-wise
            # `i < total_count`, count where U < p.
            full_shape: tuple[int, ...] = (max_n,) + out_shape
            u: Tensor = lucid.rand(max_n, *out_shape, dtype=dtype, device=device)
            p_b: Tensor = self._probs + lucid.zeros(full_shape, dtype=dtype, device=device)
            tc_b_full: Tensor = self.total_count + lucid.zeros(
                full_shape, dtype=dtype, device=device
            )
            # Per-trial active mask: trial index i is active if i < tc.
            i_flat: Tensor = lucid.arange(0, max_n, 1, dtype=dtype, device=device)
            i_idx: Tensor = i_flat.reshape([max_n] + [1] * len(out_shape)) + lucid.zeros(
                full_shape, dtype=dtype, device=device
            )
            active: Tensor = (i_idx < tc_b_full).to(dtype)
            success: Tensor = (u < p_b).to(dtype) * active
            return success.sum(dim=0).to(lucid.int64).detach()

        # Normal approximation.
        mu: Tensor = self.mean + lucid.zeros(out_shape, dtype=dtype, device=device)
        sigma: Tensor = self.variance.sqrt() + lucid.zeros(out_shape, dtype=dtype, device=device)
        z: Tensor = lucid.randn(*out_shape, dtype=dtype, device=device)
        sample: Tensor = (mu + sigma * z + 0.5).floor()
        # Clamp to [0, total_count].
        zero: Tensor = lucid.zeros_like(sample)
        tc_b = self.total_count + zero
        sample = sample.maximum(zero).minimum(tc_b)
        return sample.to(lucid.int64).detach()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability of the given counts under the Binomial distribution.

        Computed via logits for numerical stability:

        .. math::

            \log P(X = k) = \log\binom{n}{k} + k \, l - n \log(1 + e^l)

        where :math:`l = \operatorname{logit}(p)` and the binomial coefficient
        is evaluated via :math:`\log\Gamma`.

        Parameters
        ----------
        value : Tensor
            Count values :math:`k \in \{0, 1, \ldots, n\}`.

        Returns
        -------
        Tensor
            Log-probability values of the same shape as ``value``.
        """
        # log p(k) = lgamma(n+1) − lgamma(k+1) − lgamma(n−k+1)
        #           + k·log(p) + (n−k)·log(1−p).
        n: Tensor = self.total_count
        # ``sample`` answers in the integer dtype a count belongs in, and
        # everything below is float arithmetic — so the counts join the
        # parameters rather than the other way round.
        k: Tensor = value.to(self._logits.dtype)
        log_comb: Tensor = lucid.lgamma(n + 1.0) - lucid.lgamma(k + 1.0) - lucid.lgamma(n - k + 1.0)
        # Stable form via logits: k·l − n·softplus(l).
        l: Tensor = self._logits

        # Written as ``−k·softplus(−l) − (n−k)·softplus(l)`` rather than
        # ``k·l − n·softplus(l)``.  The two are algebraically the same —
        # ``softplus(−l) = softplus(l) − l`` — and they differ at the
        # degenerate probabilities, where the logit is infinite.
        #
        # ``Binomial(n=5, p=1)``: the old form is ``5·inf − 5·inf``, which
        # is NaN, for an outcome that is certain and whose log-probability
        # is 0.  In this arrangement each infinity sits in the term whose
        # count is zero, so the guards below reach it before it is
        # multiplied.  ``Binomial(n=0, p=1)`` was the same defect one step
        # earlier: the distribution could not score the only sample it can
        # draw.
        #
        # Guarded on the *operand*, not on the product.  Replacing a NaN
        # afterwards leaves it in the gradient, where the mask multiplies
        # it straight back in.  And only where the operand is infinite: a
        # finite one stays, so ``(n - k) * softplus(l)`` keeps its
        # derivative ``softplus(l)`` in ``n`` at ``k = n`` — zeroed there,
        # it put the entropy's gradient in ``total_count`` 4% off.
        #
        # ``softplus`` rather than ``log(1 + exp(l))``, which reaches inf by
        # l ≈ 89 in float32 and takes the answer with it.  The engine's own,
        # whose derivative at 0 is 1/2: the ``max(x, 0) + log1p(exp(-|x|))``
        # spelled out here before took the derivative of ``max`` at its tie,
        # 1, and doubled the gradient at ``p = 1/2``.
        sp_neg, sp_pos = (-l).softplus(), l.softplus()
        lower = lucid.where((k == 0.0) & lucid.isinf(sp_neg), 0.0, sp_neg)
        upper = lucid.where((n - k == 0.0) & lucid.isinf(sp_pos), 0.0, sp_pos)
        return log_comb - k * lower - (n - k) * upper

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Binomial distribution (in nats).

        .. math::

            H = -\sum_{k=0}^{n} P(X = k) \log P(X = k)

        summed over the whole support, as the reference framework sums it.

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Raises
        ------
        NotImplementedError
            If ``total_count`` differs across the batch: the support is
            enumerated once, for one :math:`n`, as in the reference.

        Examples
        --------
        >>> Binomial(total_count=2, probs=0.5).entropy()  # 1.5 ln 2
        tensor(1.04)
        """
        log_p = self.log_prob(self.enumerate_support(expand=False))
        # p log p is 0 where p is: an outcome a degenerate probability rules
        # out scores -inf, and exp(-inf) * -inf is NaN.
        finite = lucid.isfinite(log_p)
        safe = lucid.where(finite, log_p, 0.0)
        return -lucid.where(finite, safe.exp() * safe, 0.0).sum(dim=0)

    @override
    def enumerate_support(self, expand: bool = True) -> Tensor:
        r"""The counts :math:`0, \ldots, n` along axis 0, when ``total_count``
        is homogeneous across the batch."""
        total_count = int(self.total_count.max().item())
        if float(self.total_count.min().item()) != total_count:
            raise NotImplementedError(
                "Inhomogeneous total count not supported by `enumerate_support`."
            )
        param = self.logits if self._is_logits else self.probs
        values = lucid.arange(0, 1 + total_count, 1, dtype=param.dtype, device=param.device)
        return self._expand_support(values, expand)

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Binomial` instance with parameters expanded to ``batch_shape``."""
        new = self._get_checked_instance(Binomial, _instance)
        batch_shape = tuple(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new._is_logits = self._is_logits
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
        return self._finish_expand(new, batch_shape)


class NegativeBinomial(Distribution):
    r"""Negative Binomial distribution over the number of failures before r successes.

    ``NegativeBinomial(total_count=r, probs=p)`` models the count of failures
    :math:`k` before achieving :math:`r` successes in a sequence of independent
    Bernoulli trials each with success probability :math:`1 - p`.

    Equivalently, this is a **Gamma-Poisson compound**: draw
    :math:`\lambda \sim \operatorname{Gamma}(r, (1-p)/p)`, then
    :math:`X \sim \operatorname{Poisson}(\lambda)`.  This interpretation
    extends the distribution to real-valued ``total_count`` (the
    *generalised* Negative Binomial) and forms the basis of the sampler.

    Parameters
    ----------
    total_count : Tensor | float
        Dispersion / number of successes :math:`r \geq 0`.  May be non-integer
        for the generalised version.
    probs : Tensor | float | None, optional
        Probability of failure :math:`p \in [0, 1)`.  Mutually exclusive
        with ``logits``.
    logits : Tensor | float | None, optional
        Log-odds of failure :math:`l = \log(p / (1-p)) \in \mathbb{R}`.
        Mutually exclusive with ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    total_count : Tensor
        Dispersion parameter :math:`r`, broadcast against the probability.
    probs : Tensor
        Failure probability — stored when given, otherwise derived from
        ``logits`` on access.
    logits : Tensor
        Log-odds of failure — stored when given, otherwise derived from
        ``probs`` on access.

    Notes
    -----
    **PMF**:

    .. math::

        P(X = k) = \binom{k + r - 1}{k} (1-p)^r p^k,
        \quad k \in \{0, 1, 2, \ldots\}

    **Log-PMF** (via ``lgamma`` for numerical stability):

    .. math::

        \log P(X = k) = \log\Gamma(k+r) - \log\Gamma(r) - \log\Gamma(k+1)
                       + r \log(1-p) + k \log p

    **Moments**:

    - Mean: :math:`E[X] = r p / (1-p)`
    - Variance: :math:`\operatorname{Var}[X] = r p / (1-p)^2`

    **Sampler**: uses the Gamma-Poisson representation.  A standard-Gamma
    variate with concentration :math:`r` is drawn via rejection sampling,
    scaled by :math:`p/(1-p)`, and passed to the Poisson sampler.  This
    produces exact (non-approximate) samples for all real :math:`r > 0`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import NegativeBinomial
    >>> dist = NegativeBinomial(total_count=5.0, probs=0.4)
    >>> samples = dist.sample((100,))
    >>> samples.shape
    (100,)
    >>> dist.mean  # r p / (1 - p) = 5 * 0.4 / 0.6
    tensor(3.333)
    >>> samples.dtype  # floating counts, like the parameters
    lucid.float32
    >>> NegativeBinomial(lucid.tensor([2.0, 5.0, 9.0]), probs=0.4).batch_shape
    (3,)
    """

    arg_constraints = {
        # ``r = 0`` or ``p = 0`` is the point mass at 0; ``p = 1`` (every
        # trial a failure) never reaches the ``r``-th success.
        "total_count": nonnegative,
        "probs": half_open_interval(0.0, 1.0),
        "logits": real,
    }
    support: Constraint | None = nonnegative_integer

    def __init__(
        self,
        total_count: Tensor | float,
        probs: Tensor | float | None = None,
        logits: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a Negative Binomial distribution.

        Parameters
        ----------
        total_count : Tensor | float
            Dispersion / number of successes :math:`r \geq 0`.  May be
            non-integer for the generalised version.
        probs : Tensor | float | None, optional
            Probability of failure :math:`p \in [0, 1)`.  Mutually exclusive
            with ``logits``.
        logits : Tensor | float | None, optional
            Log-odds of failure :math:`l = \log(p / (1-p))`.  Mutually
            exclusive with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` and ``logits`` are provided.
        """
        if (probs is None) == (logits is None):
            raise ValueError("NegativeBinomial: pass exactly one of `probs` or `logits`.")
        self._is_logits = probs is None
        if probs is not None:
            param = _as_tensor(probs)
        else:
            param = _as_tensor(logits)  # type: ignore[arg-type]
        # Both sides broadcast, as in :class:`Binomial`: sizing the batch
        # from the probability alone gave ``NegativeBinomial([2, 5, 9],
        # probs=0.4)`` a ``batch_shape`` of ``()`` beside a three-element
        # ``total_count``.
        self.total_count, param = _broadcast_pair(_as_tensor(total_count), param)
        if self._is_logits:
            self.logits = param
        else:
            self.probs = param
        super().__init__(
            batch_shape=tuple(param.shape), event_shape=(), validate_args=validate_args
        )

    @_lazy_param
    def probs(self) -> Tensor:
        """Failure probability — as given, or ``sigmoid(logits)`` on access."""
        return _logits_to_probs(self.logits)

    @_lazy_param
    def logits(self) -> Tensor:
        r"""Log-odds of failure — as given, or derived on access.

        Derived from ``probs`` clamped one epsilon inside :math:`[0, 1]`, as
        the reference framework derives it.
        """
        return _probs_to_logits(_clamp_probs(self.probs))

    @property
    def _probs(self) -> Tensor:
        """Failure probability :math:`p`, computed from logits if necessary.

        Returns
        -------
        Tensor
            Failure probability values in :math:`[0, 1)`, shape ``batch_shape``.
        """
        return self.probs if not self._is_logits else _logits_to_probs(self.logits)

    @property
    def _logits(self) -> Tensor:
        r"""Log-odds of failure :math:`\log(p/(1-p))`, computed from probs if necessary.

        Returns
        -------
        Tensor
            Log-odds values in :math:`\mathbb{R}`, shape ``batch_shape``.
        """
        return self.logits if self._is_logits else _probs_to_logits(self.probs)

    @override
    @property
    def mean(self) -> Tensor:
        """Mean of the Negative Binomial: :math:`E[X] = r p / (1-p)`.

        Returns
        -------
        Tensor
            Mean values of shape ``batch_shape``.
        """
        # E[X] = r · p / (1 − p).
        p: Tensor = self._probs
        return self.total_count * p / (1.0 - p)

    @override
    @property
    def mode(self) -> Tensor:
        r"""Mode of the Negative Binomial distribution.

        .. math::

            \max\left(\left\lfloor \frac{(r - 1)\, p}{1 - p} \right\rfloor, 0\right)
        """
        ratio = self._logits.exp()
        return ((self.total_count - 1.0) * ratio).floor().clip(0.0, None)

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the Negative Binomial: :math:`\operatorname{Var}[X] = r p / (1-p)^2`.

        Returns
        -------
        Tensor
            Variance values of shape ``batch_shape``.
        """
        # Var = r · p / (1 − p)².
        p: Tensor = self._probs
        return self.total_count * p / ((1.0 - p) ** 2)

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples via the Gamma-Poisson compound representation.

        Draws :math:`\lambda \sim \operatorname{Gamma}(r, (1-p)/p)` then
        :math:`X \sim \operatorname{Poisson}(\lambda)`.  This produces exact
        (non-approximate) samples for all real :math:`r > 0`.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Non-negative integer-valued samples of shape
            ``(*sample_shape, *batch_shape)``, in the parameters' floating
            dtype.
        """
        # Gamma-Poisson compound: X = Poisson(λ),  λ = Gamma(r, (1−p)/p).
        from lucid.distributions.gamma import _sample_standard_gamma

        out_shape: tuple[int, ...] = self._extended_shape(sample_shape)
        p: Tensor = self._probs + lucid.zeros(
            out_shape, dtype=self._probs.dtype, device=self._probs.device
        )
        r: Tensor = self.total_count + lucid.zeros(
            out_shape, dtype=self._probs.dtype, device=self._probs.device
        )
        # Gamma(r, rate=(1−p)/p) — sample standard Gamma(r) then divide
        # by the rate.
        rate: Tensor = (1.0 - p) / p
        # ``r`` already carries ``sample_shape`` in front; passing it again
        # would prepend it twice and hand back ``sample_shape * 2``.
        std_gamma: Tensor = _sample_standard_gamma(r, ())
        lam: Tensor = std_gamma / rate
        # In the parameters' floating dtype, as the reference framework
        # returns its counts — ``poisson`` itself answers in int64.
        return lucid.poisson(lam).to(lam.dtype).detach()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability of counts under the Negative Binomial distribution.

        .. math::

            \log P(X = k) = \log\Gamma(k+r) - \log\Gamma(r) - \log\Gamma(k+1)
                           + r \log(1-p) + k \log p

        The two logarithms are evaluated from the logit :math:`\ell` as
        :math:`\log p = -\operatorname{softplus}(-\ell)` and
        :math:`\log(1-p) = -\operatorname{softplus}(\ell)`.  Through
        ``probs`` they gave ``-inf`` once :math:`|\ell|` was large enough for
        :math:`p` to round to 0 or 1 (:math:`|\ell| \gtrsim 17` in
        float32), where the answer is a finite :math:`-|\ell|`-sized value;
        and ``0 · log 0 = NaN`` at :math:`p = 0`.  A ``probs`` of exactly 0 or
        1 is held one epsilon inside :math:`[0, 1]` first, as the reference
        framework does.

        At :math:`r = 0` the distribution is the point mass at 0: the
        coefficient :math:`\log\Gamma(k + r) - \log\Gamma(r)` is
        :math:`\infty - \infty` at :math:`k = 0` and is taken as its limit
        0, so :math:`\log P(X = 0) = 0` and every :math:`k > 0` scores
        ``-inf``, as in the reference framework.

        Parameters
        ----------
        value : Tensor
            Non-negative integer counts :math:`k \geq 0`.

        Returns
        -------
        Tensor
            Log-probability values of the same shape as ``value``.
        """
        k: Tensor = value
        r: Tensor = self.total_count
        l: Tensor = self.logits
        # ``k · softplus(−ℓ)`` is ``0 · inf`` for a zero count at ``ℓ = −inf``
        # (a logit given as ``-inf``), and ``r · softplus(ℓ)`` the same for
        # ``r = 0`` at ``ℓ = inf``; guarded on the operand, as in
        # :class:`Binomial`, so the gradient never sees the infinity.
        neg_log_p: Tensor = lucid.where(k == 0, 0.0, (-l).softplus())
        neg_log_q: Tensor = lucid.where(r == 0, 0.0, l.softplus())
        # ``lgamma(0) − lgamma(0)`` at r = k = 0 — both arguments swapped
        # for 1 there, where the coefficient's limit is log 1 = 0.
        empty: Tensor = (k + r) == 0
        k_plus_r: Tensor = lucid.where(empty, 1.0, k + r)
        r_safe: Tensor = lucid.where(empty, 1.0, r)
        return (
            lucid.lgamma(k_plus_r)
            - lucid.lgamma(r_safe)
            - lucid.lgamma(k + 1.0)
            - r * neg_log_q
            - k * neg_log_p
        )

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`NegativeBinomial` instance with parameters expanded to ``batch_shape``."""
        new = self._get_checked_instance(NegativeBinomial, _instance)
        batch_shape = tuple(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new._is_logits = self._is_logits
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
        return self._finish_expand(new, batch_shape)
