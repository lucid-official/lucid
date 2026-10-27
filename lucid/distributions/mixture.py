"""``MixtureSameFamily`` — finite mixture of identically-typed components."""

from collections.abc import Sequence
from typing import Self, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.categorical import Categorical
from lucid.distributions.constraints import Constraint
from lucid.distributions.distribution import Distribution


class MixtureSameFamily(Distribution):
    r"""Finite mixture model where all components share the same distribution family.

    ``MixtureSameFamily`` combines:

    - a **mixture distribution** — a :class:`Categorical` over :math:`K`
      components that assigns mixing weight :math:`\pi_k` to component :math:`k`,
    - a **component distribution** — a single :class:`Distribution` whose
      rightmost batch dimension has size :math:`K` (one set of parameters
      per component).

    This encodes the generative process:

    .. math::

        k \sim \operatorname{Categorical}(\pi_1, \ldots, \pi_K), \quad
        X \mid k \sim p_k(\cdot)

    **Sampling** is non-reparameterised because drawing the discrete index
    :math:`k` creates a discontinuous path through the mixture weights.
    For differentiable training through mixture models consider the
    **ELBO** lower bound or use relaxed Categorical samples.

    Parameters
    ----------
    mixture_distribution : Categorical
        A :class:`Categorical` distribution over :math:`K` components.
        Its ``batch_shape`` must be compatible with the leading batch
        dimensions of ``component_distribution``.
    component_distribution : Distribution
        Any distribution whose rightmost batch dimension equals :math:`K`
        (the number of components).  For example, a
        ``Normal(loc=..., scale=...)`` with ``loc.shape[-1] == K``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    mixture_distribution : Categorical
        The :math:`K`-way categorical mixing weights distribution.
    component_distribution : Distribution
        The per-component distributions batched over :math:`K`.

    Notes
    -----
    **Log-probability** is computed via the **log-sum-exp trick** to avoid
    underflow when summing exponentially small terms:

    .. math::

        \log p(x) = \log \sum_{k=1}^{K} \pi_k p_k(x)
                   = \operatorname{logsumexp}_k
                     \bigl[\log \pi_k + \log p_k(x)\bigr]

    **Mean** (law of total expectation):

    .. math::

        E[X] = \sum_{k=1}^{K} \pi_k \mu_k

    **Variance** (law of total variance):

    .. math::

        \operatorname{Var}[X] = \sum_k \pi_k \sigma_k^2
                                + \sum_k \pi_k (\mu_k - E[X])^2

    This decomposes into *within-component* variance (first term) and
    *between-component* variance (second term).

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import MixtureSameFamily, Categorical, Normal
    >>> # 2-component Gaussian mixture
    >>> mix = Categorical(probs=lucid.tensor([0.3, 0.7]))
    >>> comp = Normal(
    ...     loc=lucid.tensor([-2.0, 2.0]),
    ...     scale=lucid.tensor([0.5, 1.0]),
    ... )
    >>> dist = MixtureSameFamily(mix, comp)
    >>> samples = dist.sample((200,))
    >>> samples.shape
    (200,)
    """

    def __init__(
        self,
        mixture_distribution: Categorical,
        component_distribution: Distribution,
        validate_args: bool | None = None,
    ) -> None:
        """Construct a mixture-of-experts distribution.

        Parameters
        ----------
        mixture_distribution : Categorical
            Discrete distribution over the ``K`` mixture components. Its
            ``batch_shape`` must match the resulting mixture batch shape, and
            ``event_shape`` must be empty.
        component_distribution : Distribution
            Family of component distributions. The rightmost batch dimension
            indexes the ``K`` components, i.e. ``component_distribution.batch_shape``
            must end with ``K`` matching ``mixture_distribution._num_events``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If ``mixture_distribution`` is not ``Categorical`` or if the
            rightmost batch dim of ``component_distribution`` does not equal
            ``K``.
        """
        if not isinstance(mixture_distribution, Categorical):
            raise TypeError("MixtureSameFamily: mixture_distribution must be Categorical")
        # Component dist's rightmost batch dim is the number of components.
        comp_batch = tuple(component_distribution.batch_shape)
        if len(comp_batch) == 0 or comp_batch[-1] != mixture_distribution._num_events:
            raise ValueError(
                f"MixtureSameFamily: component_distribution.batch_shape "
                f"{comp_batch} must end with K={mixture_distribution._num_events}"
            )
        self.mixture_distribution = mixture_distribution
        self.component_distribution = component_distribution
        # Drop the K dim from the batch shape.
        new_batch = comp_batch[:-1]
        super().__init__(
            batch_shape=new_batch,
            event_shape=tuple(component_distribution.event_shape),
            validate_args=validate_args,
        )

    @property
    @override
    def support(self) -> Constraint | None:
        """Support of the distribution — that of the components.

        Returns
        -------
        Constraint or None
            ``component_distribution.support``: every component shares a
            family, so a value the mixture can produce is one a component
            can.
        """
        return self.component_distribution.support

    @property
    @override
    def mean(self) -> Tensor:
        r"""Expected value via the law of total expectation: :math:`E[X] = \sum_k \pi_k \mu_k`.

        Returns
        -------
        Tensor
            Mean of the mixture, shape ``(*batch_shape, *event_shape)``.
        """
        # E[X] = Σ_k π_k · μ_k along the K dim.
        probs: Tensor = self.mixture_distribution._probs  # (..., K)
        comp_mean: Tensor = self.component_distribution.mean  # (..., K, *event)
        # Multiply along the K axis.  ``comp_mean`` may have trailing
        # event dims; broadcast ``probs`` accordingly.
        weight = probs.reshape(list(probs.shape) + [1] * len(self._event_shape))
        return (weight * comp_mean).sum(dim=-1 - len(self._event_shape))

    @property
    @override
    def variance(self) -> Tensor:
        r"""Variance via the law of total variance.

        Decomposes as within-component variance plus between-component variance:

        .. math::

            \operatorname{Var}[X] = \underbrace{\sum_k \pi_k \sigma_k^2}_{\text{within}}
                                   + \underbrace{\sum_k \pi_k (\mu_k - E[X])^2}_{\text{between}}

        Returns
        -------
        Tensor
            Variance of the mixture, shape ``(*batch_shape, *event_shape)``.
        """
        # Var = E[Var(X|k)] + Var(E[X|k]) — law of total variance.
        probs: Tensor = self.mixture_distribution._probs
        comp_mean: Tensor = self.component_distribution.mean
        comp_var: Tensor = self.component_distribution.variance
        weight = probs.reshape(list(probs.shape) + [1] * len(self._event_shape))
        ax = -1 - len(self._event_shape)
        mean_of_mean = (weight * comp_mean).sum(dim=ax, keepdim=True)
        within: Tensor = (weight * comp_var).sum(dim=ax)
        between: Tensor = (weight * (comp_mean - mean_of_mean) ** 2).sum(dim=ax)
        return within + between

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the mixture by ancestral sampling.

        The procedure is:

        1. Draw component indices :math:`k \sim \operatorname{Categorical}(\pi)`.
        2. Draw one sample per component for the full output shape.
        3. Gather the sample corresponding to the drawn component index.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Samples of shape ``(*sample_shape, *batch_shape, *event_shape)``.
        """
        # 1. Draw component indices from the mixture categorical.
        comp_idx: Tensor = self.mixture_distribution.sample(sample_shape)
        # 2. Draw one sample per component for the same shape, then gather
        #    along the K axis.  Cheap when K is modest; exact otherwise.
        comp_samples: Tensor = self.component_distribution.sample(sample_shape)
        # comp_samples shape: (*sample, *batch, K, *event); pick along K.
        ax: int = comp_samples.ndim - 1 - len(self._event_shape)
        idx_unsq: Tensor = comp_idx.to(lucid.int64)
        # Expand idx to match comp_samples shape (insert event dims as 1
        # then broadcast).
        idx_shape: list[int] = list(idx_unsq.shape) + [1] + [1] * len(self._event_shape)
        idx_b: Tensor = (
            idx_unsq.reshape(idx_shape)
            .broadcast_to(list(comp_samples.shape[:ax]) + [1] + list(comp_samples.shape[ax + 1 :]))
            .contiguous()
        )
        gathered: Tensor = lucid.gather(comp_samples, ax, idx_b)
        return gathered.squeeze(ax)

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability of the mixture evaluated at ``value``.

        Uses the numerically stable log-sum-exp identity:

        .. math::

            \log p(x) = \operatorname{logsumexp}_k
            \bigl[\log \pi_k + \log p_k(x)\bigr]

        Parameters
        ----------
        value : Tensor
            Observation of shape ``(*batch_shape, *event_shape)``.

        Returns
        -------
        Tensor
            Log-density values of shape ``(*batch_shape,)``.
        """
        # log p(x) = logsumexp_k [ log π_k + log p_k(x) ].
        log_pi: Tensor = self.mixture_distribution._log_probs  # (..., K)
        # Insert K-axis into value so component_distribution sees it as a
        # broadcast input over K.  Value shape: (..., *event); we want
        # (..., 1, *event) so the component log_prob produces (..., K, *event_collapsed).
        v_unsq: Tensor = value.reshape(
            list(value.shape[: value.ndim - len(self._event_shape)])
            + [1]
            + list(value.shape[value.ndim - len(self._event_shape) :])
        )
        log_pk: Tensor = self.component_distribution.log_prob(v_unsq)
        return lucid.logsumexp(log_pi + log_pk, dim=-1)

    @override
    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the mixture.

        .. math::

            F(x) = \sum_k \pi_k F_k(x)

        Parameters
        ----------
        value : Tensor
            Point(s) of shape ``(*batch_shape,)`` — the components must be
            univariate, as for the reference framework's.

        Returns
        -------
        Tensor
            Probabilities in :math:`[0, 1]`, shape ``(*batch_shape,)``.

        Raises
        ------
        NotImplementedError
            If the component family has no ``cdf``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions import Categorical, MixtureSameFamily, Normal
        >>> mix = MixtureSameFamily(
        ...     Categorical(probs=lucid.tensor([0.5, 0.5])),
        ...     Normal(lucid.tensor([-1.0, 1.0]), lucid.tensor([1.0, 1.0])),
        ... )
        >>> mix.cdf(lucid.tensor(0.0))  # symmetric about 0
        tensor(0.5)
        """
        cdf_k: Tensor = self.component_distribution.cdf(value.unsqueeze(-1))
        return (cdf_k * self.mixture_distribution._probs).sum(dim=-1)

    @override
    def expand(
        self,
        batch_shape: Sequence[int],
        _instance: Self | None = None,
    ) -> Self:
        """Return a new :class:`MixtureSameFamily` instance with parameters expanded to ``batch_shape``."""
        new = self._get_checked_instance(MixtureSameFamily, _instance)
        batch_shape = tuple(batch_shape)
        comp_batch = batch_shape + (self.mixture_distribution._num_events,)
        new.mixture_distribution = self.mixture_distribution.expand(batch_shape)
        new.component_distribution = self.component_distribution.expand(comp_batch)
        super(MixtureSameFamily, new).__init__(
            batch_shape=batch_shape,
            event_shape=self.event_shape,
            validate_args=False,
        )
        new._validate_args = self._validate_args
        return new
