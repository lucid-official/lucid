"""``Categorical`` (discrete with K outcomes) and ``OneHotCategorical``."""

from collections.abc import Sequence
from typing import Self, override

import lucid
from lucid._dtype import finfo
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import (
    Constraint,
    integer_interval,
    one_hot,
    real_vector,
    simplex,
)
from lucid.distributions.distribution import Distribution


from lucid.distributions._util import (
    _broadcast_shapes,
    _clamp_probs,
    _lazy_param,
    _require_probs_or_logits,
)


def _normalize_probs(probs: Tensor) -> Tensor:
    """Project a non-negative tensor onto the K-simplex along the last
    dim (matches the reference framework's preprocessing)."""
    return probs / probs.sum(dim=-1, keepdim=True)


def _normalize_logits(logits: Tensor) -> Tensor:
    """Shift logits along the last dim so they are log-probabilities.

    The counterpart of :func:`_normalize_probs`, and the reference
    framework's preprocessing: the stored ``logits`` are
    ``logits - logsumexp(logits)``, so ``logits.exp()`` sums to 1.  What the
    constraint then validates is the normalised value, which is what makes
    the domain the reference's — a row that is all ``-inf`` (no category
    possible) or holds a ``+inf`` has no normalisation and comes out NaN,
    which ``real`` refuses, while a masked ``-inf`` among finite logits
    passes.
    """
    return logits - lucid.logsumexp(logits, dim=-1, keepdim=True)


class Categorical(Distribution):
    r"""Categorical distribution — a discrete distribution over K labelled outcomes.

    ``Categorical(probs=p)`` or ``Categorical(logits=l)`` defines a distribution
    over the integer set :math:`\{0, 1, \ldots, K-1\}` where :math:`K` is the
    number of categories.  Exactly one of ``probs`` or ``logits`` must be given.

    Parameters
    ----------
    probs : Tensor | None, optional
        Non-negative probability vector (or batch of vectors) of shape
        ``(..., K)``.  Rows are automatically normalised to sum to 1.
        Mutually exclusive with ``logits``.
    logits : Tensor | None, optional
        Unnormalised log-probabilities of shape ``(..., K)``.
        The distribution uses :math:`\text{softmax}` internally to convert to
        normalised probabilities.  Mutually exclusive with ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    probs : Tensor
        Normalised probability vector of shape ``(..., K)`` — stored when
        given, otherwise ``softmax(logits)``, derived on access.
    logits : Tensor
        Log-probability vector of shape ``(..., K)`` — as given
        (unnormalised), otherwise ``log(probs)``, derived on access.

    Notes
    -----
    **PMF**:

    .. math::

        P(X = k) = p_k, \quad k \in \{0, 1, \ldots, K-1\},
        \quad \sum_{k} p_k = 1

    **Parameterisations** are related by:

    .. math::

        p_k = \frac{e^{l_k}}{\sum_j e^{l_j}}, \qquad
        l_k = \log p_k \;(\text{up to additive constant})

    **Entropy**:

    .. math::

        H[X] = -\sum_{k=0}^{K-1} p_k \log p_k

    **Mean** is not well-defined for a general Categorical (the labels have no
    canonical metric), so ``mean`` returns a ``NaN`` tensor of the batch shape.

    **Sampling** uses the **Gumbel-max trick**: add i.i.d.
    :math:`\operatorname{Gumbel}(0, 1)` noise to the log-probabilities and
    take the argmax.  This is equivalent to ancestral sampling and avoids
    cumulative-sum + binary-search.

    The batch dimensions of the input correspond to independent distributions.
    For example, ``probs`` of shape ``(B, K)`` yields a batch of :math:`B`
    Categorical distributions.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Categorical
    >>> # Uniform over 4 categories
    >>> dist = Categorical(probs=lucid.tensor([0.25, 0.25, 0.25, 0.25]))
    >>> samples = dist.sample((10,))
    >>> # Batch of 2 distributions
    >>> dist_b = Categorical(logits=lucid.zeros(2, 5))
    >>> dist_b.batch_shape, dist_b.event_shape
    ((2,), ())
    """

    arg_constraints = {"probs": simplex, "logits": real_vector}
    has_enumerate_support = True

    def __init__(
        self,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        """Initialise a Categorical distribution.

        Parameters
        ----------
        probs : Tensor | None, optional
            Non-negative probability vector of shape ``(..., K)``.  Rows are
            automatically normalised to sum to 1.  Mutually exclusive with
            ``logits``.
        logits : Tensor | None, optional
            Unnormalised log-probabilities of shape ``(..., K)``.  Stored
            shifted by their ``logsumexp`` (so :attr:`logits` holds
            log-probabilities), as in the reference framework; a row with
            no such shift — all ``-inf``, or holding ``+inf`` — is refused
            under validation.  Mutually exclusive with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` and ``logits`` are provided.
        """
        raw, self._is_logits = _require_probs_or_logits("Categorical", probs, logits)
        if self._is_logits:
            self.logits = _normalize_logits(raw)
            shape = tuple(self.logits.shape)
        else:
            self.probs = _normalize_probs(raw)
            shape = tuple(self.probs.shape)
        self._num_events = shape[-1]
        super().__init__(batch_shape=shape[:-1], event_shape=(), validate_args=validate_args)

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`Categorical` with batch dimensions expanded to ``batch_shape``."""
        new = self._get_checked_instance(Categorical, _instance)
        batch_shape_t = tuple(batch_shape)
        param_shape = batch_shape_t + (self._num_events,)
        if not self._is_logits:
            new.probs = self.probs.expand(param_shape)
            new._is_logits = False
        else:
            new.logits = self.logits.expand(param_shape)
            new._is_logits = True
        new._num_events = self._num_events
        super(Categorical, new).__init__(batch_shape_t, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @_lazy_param
    def probs(self) -> Tensor:
        """Normalised probabilities — as given, or ``softmax(logits)``."""
        from lucid.nn.functional.activations import softmax

        return softmax(self.logits, dim=-1)

    @_lazy_param
    def logits(self) -> Tensor:
        """Log-probabilities — as given (shifted by their ``logsumexp``), or
        derived from ``probs`` on access.

        Derived as ``log(probs)`` with ``probs`` clamped one epsilon inside
        ``[0, 1]``, as the reference framework derives it, so a category of
        probability zero gets a large negative logit rather than ``-inf``.
        """
        return _clamp_probs(self.probs).log()

    @override
    @property
    def support(self) -> Constraint:
        r"""Support of the distribution: integer interval :math:`\{0, \ldots, K-1\}`.

        Returns
        -------
        Constraint
            An ``integer_interval`` constraint from ``0`` to ``K - 1``.
        """
        return integer_interval(0, self._num_events - 1)

    @property
    def _log_probs(self) -> Tensor:
        """Lazily resolved log-probability tensor over the ``K`` categories.

        Applies ``log_softmax`` to ``self.logits`` when the distribution was
        constructed from logits; otherwise returns ``self.probs.log()``.
        """
        if self._is_logits:
            from lucid.nn.functional.activations import log_softmax

            return log_softmax(self.logits, dim=-1)
        return self.probs.log()

    @property
    def _probs(self) -> Tensor:
        """Lazily resolved probability tensor (summing to 1 along the last axis).

        Applies ``softmax`` to ``self.logits`` when the distribution was
        constructed from logits; otherwise returns ``self.probs`` directly.
        """
        if self._is_logits:
            from lucid.nn.functional.activations import softmax

            return softmax(self.logits, dim=-1)
        return self.probs

    @override
    @property
    def mean(self) -> Tensor:
        """Mean of the Categorical distribution (undefined — returns NaN).

        The Categorical distribution assigns labels with no inherent ordering
        or metric, so the mean is not well-defined.  This property returns a
        ``NaN`` tensor of the batch shape to match expected behaviour.

        Returns
        -------
        Tensor
            Tensor of ``float('nan')`` values with shape ``batch_shape``.
        """
        # mean of Categorical isn't well-defined (no metric on labels)
        # but reference framework returns NaN with the right shape.
        return lucid.full(
            self._batch_shape, float("nan"), device=self._probs.device, dtype=self._probs.dtype
        )

    @override
    @property
    def mode(self) -> Tensor:
        """Mode of the Categorical distribution: the most probable category.

        Returns
        -------
        Tensor
            Integer category indices of shape ``batch_shape``; the first one
            on a tie.
        """
        return self._probs.argmax(dim=-1)

    @override
    @property
    def variance(self) -> Tensor:
        """Variance of the Categorical distribution (undefined — NaN).

        A variance is a squared distance, and the labels this distribution
        assigns have no metric to measure one with — the same reason
        :attr:`mean` is NaN.  Returned rather than raised so the moment
        composes with the rest of the API the way the reference framework's
        does; a caller who wants an error can test for NaN.

        Returns
        -------
        Tensor
            Tensor of ``float('nan')`` with shape ``batch_shape``.
        """
        return lucid.full(
            self._batch_shape, float("nan"), device=self._probs.device, dtype=self._probs.dtype
        )

    @override
    @property
    def stddev(self) -> Tensor:
        """Standard deviation of the Categorical distribution (NaN).

        The square root of :attr:`variance`, which is NaN for the reason
        given there.

        Returns
        -------
        Tensor
            Tensor of ``float('nan')`` with shape ``batch_shape``.
        """
        return self.variance

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the Categorical distribution.

        Uses the **Gumbel-max trick**: add i.i.d.
        :math:`\operatorname{Gumbel}(0, 1)` noise to the log-probabilities and
        take the argmax, which is equivalent to ancestral sampling but avoids
        cumulative-sum and binary search.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Integer tensor of shape ``(*sample_shape, *batch_shape)`` with
            values in :math:`\{0, 1, \ldots, K-1\}`.  The result is detached
            (no gradients flow through discrete samples).
        """
        # Gumbel-max trick, no replacement: draw G ~ −log(−log U), pick argmax.
        shape = tuple(sample_shape) + tuple(self._batch_shape) + (self._num_events,)
        u = lucid.rand(*shape, dtype=self._probs.dtype, device=self._probs.device)
        u = u.clip(1e-7, 1.0 - 1e-7)
        gumbel = -(-(u.log())).log()
        scores = self._log_probs + gumbel
        return scores.argmax(dim=-1).detach()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability of the given category indices.

        For a one-hot index :math:`k`, the log-probability is:

        .. math::

            \log P(X = k) = \log p_k

        Parameters
        ----------
        value : Tensor
            Integer tensor of category indices with shape compatible with
            ``batch_shape``.  Values must be in :math:`\{0, \ldots, K-1\}`.

        Returns
        -------
        Tensor
            Log-probabilities of shape ``batch_shape``.
        """
        # Both sides broadcast, neither one onto the other.  Each
        # one-directional version of this breaks the case the other one
        # exists for: forcing ``value`` into ``batch_shape`` cannot score
        # a draw that carries a ``sample_shape`` in front of it, and
        # forcing ``log_p`` onto ``value`` cannot score a lower-rank
        # ``value`` — ``broadcast_to`` grows a shape, it never shrinks
        # one, so a scalar index against a batched distribution raised.
        # Scoring its own draws and scoring one index are both the round
        # trip this method exists for.
        log_p = self._log_probs
        num_categories = int(log_p.shape[-1])
        common = _broadcast_shapes(tuple(value.shape), tuple(log_p.shape[:-1]))
        log_p_broadcast = log_p.broadcast_to(common + (num_categories,))
        index = value.to(lucid.int64).broadcast_to(common).unsqueeze(-1)
        return lucid.gather(log_p_broadcast, -1, index).squeeze(-1)

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Categorical distribution.

        .. math::

            H[X] = -\sum_{k=0}^{K-1} p_k \log p_k

        with :math:`0 \log 0 = 0`: a category of probability zero — a zero
        in ``probs``, or a logit of ``-inf`` masking an action out —
        contributes nothing rather than ``0 · (-inf) = NaN``.

        Returns
        -------
        Tensor
            Entropy values of shape ``batch_shape`` (nats).

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions import Categorical
        >>> Categorical(probs=lucid.tensor([0.0, 0.5, 0.5])).entropy()
        tensor(0.6931)
        >>> Categorical(logits=lucid.tensor([float("-inf"), 0.0, 0.0])).entropy()
        tensor(0.6931)
        """
        if self._is_logits:
            # ``-inf`` raised to the most negative finite value: its weight is
            # an exact 0, so the product is 0 and so is its gradient.
            log_p = self._log_probs.clip(float(finfo(self.logits.dtype).min), None)
        else:
            # The log of ``probs`` held one epsilon inside [0, 1] — finite,
            # with a finite derivative, where ``log 0`` has neither.
            log_p = self.logits
        return -(self._probs * log_p).sum(dim=-1)

    @override
    def enumerate_support(self, expand: bool = True) -> Tensor:
        r"""Category indices :math:`0, \ldots, K - 1` along axis 0, in ``int64``."""
        param = self.logits if self._is_logits else self.probs
        values = lucid.arange(0, self._num_events, 1, dtype=lucid.int64, device=param.device)
        return self._expand_support(values, expand)


class OneHotCategorical(Distribution):
    r"""Categorical distribution with one-hot encoded samples.

    ``OneHotCategorical`` wraps a :class:`Categorical` and returns samples
    as one-hot vectors of shape ``(..., K)`` instead of integer indices.
    It is particularly useful for:

    - **REINFORCE-style gradient estimators** where you need a discrete
      sample but want to use it in differentiable downstream computation.
    - **Relaxations** — replacing with :class:`RelaxedOneHotCategorical`
      gives a differentiable approximation that converges to one-hot as
      temperature :math:`\to 0`.

    Parameters
    ----------
    probs : Tensor | None, optional
        Non-negative probability vector ``(..., K)``.  Normalised internally.
        Mutually exclusive with ``logits``.
    logits : Tensor | None, optional
        Unnormalised log-probabilities ``(..., K)``.  Mutually exclusive with
        ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    probs : Tensor
        Normalised probability vector — stored when given, otherwise derived
        from ``logits`` on access.
    logits : Tensor
        Log-probability vector — as given, otherwise derived from ``probs``
        on access.

    Notes
    -----
    **Samples** are integer-valued one-hot vectors in :math:`\{0, 1\}^K`
    with exactly one 1 at the sampled category index.

    **Log-probability** for a one-hot vector :math:`e_k`:

    .. math::

        \log P(X = e_k) = \sum_{j} e_{kj} \log p_j = \log p_k

    which is simply the log-probability of the selected category.

    **Entropy** equals that of the underlying :class:`Categorical`:

    .. math::

        H[X] = -\sum_{k} p_k \log p_k

    The ``event_shape`` is ``(K,)`` whereas :class:`Categorical` has
    ``event_shape = ()``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import OneHotCategorical
    >>> dist = OneHotCategorical(probs=lucid.tensor([0.1, 0.5, 0.4]))
    >>> sample = dist.sample()
    >>> sample.shape  # (3,) — one-hot
    (3,)
    >>> sample.sum()  # always 1
    tensor(1.)
    """

    has_enumerate_support = True
    arg_constraints = {"probs": simplex, "logits": real_vector}

    def __init__(
        self,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        """Initialise a OneHotCategorical distribution.

        Parameters
        ----------
        probs : Tensor | None, optional
            Non-negative probability vector of shape ``(..., K)``.  Normalised
            internally.  Mutually exclusive with ``logits``.
        logits : Tensor | None, optional
            Unnormalised log-probabilities of shape ``(..., K)``.  Mutually
            exclusive with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` and ``logits`` are provided.
        """
        cat = Categorical(probs=probs, logits=logits, validate_args=False)
        self._cat = cat
        key = "logits" if cat._is_logits else "probs"
        self.__dict__[key] = getattr(cat, key)
        super().__init__(tuple(cat._batch_shape), (cat._num_events,), validate_args=validate_args)

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`OneHotCategorical` with batch dimensions expanded to ``batch_shape``."""
        new = self._get_checked_instance(OneHotCategorical, _instance)
        batch_shape_t = tuple(batch_shape)
        new._cat = self._cat.expand(batch_shape_t)
        key = "logits" if new._cat._is_logits else "probs"
        new.__dict__[key] = getattr(new._cat, key)
        return self._finish_expand(new, batch_shape_t, self._event_shape)

    @_lazy_param
    def probs(self) -> Tensor:
        """Normalised probabilities, from the underlying :class:`Categorical`."""
        return self._cat.probs

    @_lazy_param
    def logits(self) -> Tensor:
        """Log-probabilities, from the underlying :class:`Categorical`."""
        return self._cat.logits

    @override
    @property
    def support(self) -> Constraint:
        """Support of the distribution: one-hot vectors.

        Returns
        -------
        Constraint
            The ``one_hot`` constraint — every entry 0 or 1, summing to 1.
            ``simplex`` admitted interior points such as ``[0.2, 0.3, 0.5]``,
            which no draw can be, and was not discrete.
        """
        return one_hot

    @override
    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw one-hot encoded samples.

        Internally samples category indices from the underlying
        :class:`Categorical` distribution and converts them to one-hot
        vectors via ``one_hot``.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Float tensor of shape ``(*sample_shape, *batch_shape, K)``
            containing one-hot vectors (exactly one 1 per row).
        """
        idx = self._cat.sample(sample_shape)
        from lucid.nn.functional.sparse import one_hot

        return one_hot(idx, num_classes=self._cat._num_events).to(self._cat._probs.dtype)

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability of a one-hot encoded sample.

        For a one-hot vector :math:`e_k` (all zeros except a 1 at position
        :math:`k`):

        .. math::

            \log P(X = e_k) = \sum_j e_{kj} \log p_j = \log p_k

        Parameters
        ----------
        value : Tensor
            One-hot tensor of shape ``(..., K)`` with ``float`` dtype.

        Returns
        -------
        Tensor
            Log-probabilities of shape ``batch_shape``.
        """
        # Read the category off the one-hot vector and score it.  The sum
        # ``Σ value · log p`` meets every *other* category's ``log p`` with a
        # weight of 0, and a zero-probability category made that 0 · (-inf).
        return self._cat.log_prob(value.argmax(dim=-1))

    @override
    def entropy(self) -> Tensor:
        r"""Shannon entropy of the OneHotCategorical distribution.

        Equal to the entropy of the underlying :class:`Categorical`:

        .. math::

            H[X] = -\sum_{k=0}^{K-1} p_k \log p_k

        Returns
        -------
        Tensor
            Entropy values of shape ``batch_shape`` (nats).
        """
        return self._cat.entropy()

    @override
    def enumerate_support(self, expand: bool = True) -> Tensor:
        r"""The one-hot basis vectors in :math:`\{0, 1\}^K` along axis 0, in
        the parameter's floating dtype and device."""
        n = self._event_shape[0]
        param = self._cat.logits if self._cat._is_logits else self._cat.probs
        values = lucid.eye(n, dtype=param.dtype, device=param.device)
        return self._expand_support(values, expand)

    @override
    @property
    def mean(self) -> Tensor:
        r"""Mean of the OneHotCategorical distribution.

        A one-hot sample is a vector of indicators, and the expectation of
        an indicator is the probability of the event it indicates — so the
        mean is the probability vector itself.

        .. math::

            \mathbb{E}[X_k] = p_k

        Unlike the underlying :class:`Categorical`, whose labels carry no
        metric, this one is perfectly well defined: the one-hot encoding
        supplies the vector space the expectation needs.

        Returns
        -------
        Tensor
            Probabilities of shape ``batch_shape + (K,)``.
        """
        return self._cat._probs

    @override
    @property
    def mode(self) -> Tensor:
        """Mode of the OneHotCategorical distribution: the most probable
        category, one-hot encoded.

        Returns
        -------
        Tensor
            One-hot vectors of shape ``batch_shape + (K,)``.
        """
        from lucid.nn.functional.sparse import one_hot

        probs = self._cat._probs
        return one_hot(self._cat.mode, num_classes=self._cat._num_events).to(probs.dtype)

    @override
    @property
    def variance(self) -> Tensor:
        r"""Variance of the OneHotCategorical distribution.

        Each coordinate is a Bernoulli indicator, so

        .. math::

            \operatorname{Var}[X_k] = p_k (1 - p_k)

        This is the diagonal of the covariance; the off-diagonal terms
        :math:`-p_j p_k` are not returned.

        Returns
        -------
        Tensor
            Per-coordinate variances of shape ``batch_shape + (K,)``.
        """
        probs = self._cat._probs
        return probs * (1.0 - probs)

    @override
    @property
    def stddev(self) -> Tensor:
        r"""Per-coordinate standard deviation, :math:`\sqrt{p_k (1 - p_k)}`."""
        return self.variance.sqrt()


class OneHotCategoricalStraightThrough(OneHotCategorical):
    r"""Reparameterisable :class:`OneHotCategorical` based on the straight-through gradient estimator.

    Uses the straight-through estimator (Bengio et al., 2013): the forward pass
    returns a discrete one-hot sample while the backward pass propagates
    gradients through :attr:`probs`.
    """

    has_rsample = True

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`OneHotCategoricalStraightThrough` with batch dimensions expanded."""
        new = self._get_checked_instance(OneHotCategoricalStraightThrough, _instance)
        return super().expand(batch_shape, _instance=new)

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw one-hot samples with straight-through gradients through ``probs``."""
        samples = self.sample(sample_shape)
        probs = self._cat.probs
        return samples + (probs - probs.detach())
