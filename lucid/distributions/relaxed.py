"""Concrete (Gumbel-softmax) relaxed distributions.

* :class:`RelaxedBernoulli` — relaxation of ``Bernoulli`` whose samples
  live in ``[0, 1]`` and pass gradients through.
* :class:`RelaxedOneHotCategorical` — Concrete distribution over the
  open simplex; relaxation of :class:`OneHotCategorical`.

Both forward into :func:`lucid.nn.functional.gumbel_softmax` for the
sampling math, so the same Lucid Philox stream applies.
"""

from collections.abc import Sequence
from typing import Self, override

import lucid
from lucid._dtype import finfo
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import (
    _align_device,
    _as_tensor,
    _broadcast_shapes,
    _clamp_probs,
    _lazy_param,
    _require_probs_or_logits,
)
from lucid.distributions.bernoulli import _logits_to_probs, _probs_to_logits
from lucid.distributions.constraints import (
    Constraint,
    positive,
    real,
    real_vector,
    simplex,
    unit_interval,
)
from lucid.distributions.distribution import Distribution


class RelaxedBernoulli(Distribution):
    r"""Concrete (Gumbel-sigmoid) relaxation of the Bernoulli distribution.

    ``RelaxedBernoulli(temperature=τ, probs=p)`` defines a continuous
    distribution over :math:`(0, 1)` whose samples are differentiable
    surrogates for Bernoulli samples.  As :math:`\tau \to 0` the distribution
    concentrates on :math:`\{0, 1\}`, recovering the discrete Bernoulli.
    As :math:`\tau \to \infty` the distribution approaches
    :math:`\operatorname{Uniform}(0, 1)`.

    The **Concrete distribution** (Maddison et al. 2017) / **Gumbel-softmax**
    (Jang et al. 2017) trick enables gradient-based optimisation through
    discrete latent variables in variational autoencoders and related models.

    Parameters
    ----------
    temperature : Tensor | float
        Temperature parameter :math:`\tau > 0`.  Controls the sharpness
        of the relaxation.  Small values give near-discrete samples;
        large values give near-uniform samples.
    probs : Tensor | float | None, optional
        Bernoulli success probability :math:`p \in [0, 1]`.  Mutually
        exclusive with ``logits``.
    logits : Tensor | float | None, optional
        Log-odds :math:`l = \log(p/(1-p)) \in \mathbb{R}`.  Mutually
        exclusive with ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    temperature : Tensor
        Temperature :math:`\tau`.
    probs : Tensor
        Success probability — stored when given, otherwise derived from
        ``logits`` on access.
    logits : Tensor
        Log-odds — stored when given, otherwise derived from ``probs`` on
        access.

    Notes
    -----
    **Reparameterised sampling** (Gumbel-sigmoid trick):

    .. math::

        y = \sigma\!\left(\frac{l + g_1 - g_2}{\tau}\right),
        \quad g_1, g_2 \overset{\text{iid}}{\sim} \operatorname{Gumbel}(0, 1)

    where :math:`\sigma(\cdot)` is the sigmoid function.  Gradients
    propagate through both :math:`l` and :math:`\tau`.

    **Log-PDF** (Maddison et al. 2017, the binary Concrete density):

    .. math::

        \log p(y; l, \tau) = \log\tau + d - 2\log\!\left(1 + e^{d}\right)
                             - \log y - \log(1 - y),
        \qquad d = l - \tau\,\text{logit}(y)

    where :math:`\text{logit}(y) = \log(y/(1-y))`.  The first three terms
    are the logistic density of :math:`\text{logit}(Y)`; the last two are
    the Jacobian of the sigmoid that maps it into :math:`(0, 1)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import RelaxedBernoulli
    >>> _ = lucid.manual_seed(0)
    >>> dist = RelaxedBernoulli(temperature=0.5, probs=0.7)
    >>> samples = dist.rsample((100,))
    >>> # Samples are in (0, 1)
    >>> bool(((samples > 0) & (samples < 1)).all())
    True
    """

    # The relaxation temperature divides the logits, so zero is not a
    # degenerate-but-valid setting — it is a division by zero, and the
    # sample comes back non-finite with nothing raised.  Declared
    # ``real`` here, it was never checked; the reference does not check it
    # either, and produces the same non-finite draw.  Stricter than the
    # reference on purpose — listed in the constraint-parity test's
    # exceptions.  ``probs`` and the support are the closed interval, as
    # there: a probability of 0 or 1 is held one epsilon inside it
    # (``logits``), and a value of 0 or 1 is scored one step inside it
    # (:meth:`log_prob`).
    arg_constraints = {"temperature": positive, "probs": unit_interval, "logits": real}
    support: Constraint | None = unit_interval
    has_rsample = True

    def __init__(
        self,
        temperature: Tensor | float,
        probs: Tensor | float | None = None,
        logits: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        """Construct a RelaxedBernoulli (Binary Concrete) distribution.

        Parameters
        ----------
        temperature : Tensor | float
            Temperature :math:`\\lambda > 0` controlling the relaxation tightness.
            As :math:`\\lambda \\to 0`, samples concentrate on :math:`\\{0, 1\\}`
            recovering a hard Bernoulli; as :math:`\\lambda \\to \\infty`, samples
            concentrate around :math:`1/2`.
        probs : Tensor | float | None, optional
            Probability :math:`p \\in [0, 1]` parameter. Mutually exclusive
            with ``logits``.
        logits : Tensor | float | None, optional
            Log-odds :math:`\\log\\frac{p}{1-p}` parameter. Mutually exclusive
            with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` and ``logits`` are provided.
        """
        param, self._is_logits = _require_probs_or_logits("RelaxedBernoulli", probs, logits)
        self.temperature = _as_tensor(temperature)
        # A temperature or a probability given as a number is a 0-dim host
        # tensor, and either may be the one that is not: whichever is the
        # host scalar follows the other to its device, so the two can meet
        # on Metal — the rule ``_broadcast_pair`` applies.
        if param.ndim == 0 and param.device.type == "cpu":
            param = _align_device(param, self.temperature)
        self.temperature = _align_device(self.temperature, param)
        # Both set the batch: a temperature per row over one probability is a
        # batch, and its draws divide by that temperature.
        shape = _broadcast_shapes(tuple(self.temperature.shape), tuple(param.shape))
        if tuple(self.temperature.shape) != shape:
            self.temperature = self.temperature.expand(shape)
        if tuple(param.shape) != shape:
            param = param.expand(shape)
        if self._is_logits:
            self.logits = param
        else:
            self.probs = param
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

    @property
    def _logits(self) -> Tensor:
        """Lazily resolved log-odds tensor.

        Returns :attr:`logits`: as given, or the log-odds of ``probs`` held
        one epsilon inside :math:`[0, 1]`.  Unclamped, a probability of 0 or
        1 gave an infinite logit, and :meth:`log_prob` answered NaN.
        """
        return self.logits

    @property
    def _probs(self) -> Tensor:
        """Lazily resolved probability tensor in :math:`(0, 1)`.

        Returns ``self.probs`` when the distribution was constructed from
        probs; otherwise applies ``sigmoid`` to the stored logits.
        """
        return self.probs if not self._is_logits else _logits_to_probs(self.logits)

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw a reparameterised sample via the Gumbel-sigmoid trick.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Samples :math:`y \in (0, 1)` of shape
            ``(*sample_shape, *batch_shape)``, with gradients flowing through
            both ``logits`` and ``temperature``.
        """
        shape = self._extended_shape(sample_shape)
        l: Tensor = self._logits + lucid.zeros(
            shape, dtype=self._logits.dtype, device=self._logits.device
        )
        u1: Tensor = lucid.rand(*shape, dtype=l.dtype, device=l.device).clip(1e-7, 1.0 - 1e-7)
        u2: Tensor = lucid.rand(*shape, dtype=l.dtype, device=l.device).clip(1e-7, 1.0 - 1e-7)
        # Gumbel(0,1) = −log(−log U).
        g1: Tensor = -(-(u1.log())).log()
        g2: Tensor = -(-(u2.log())).log()
        return ((l + g1 - g2) / self.temperature).sigmoid()

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the Concrete/RelaxedBernoulli distribution.

        .. math::

            \log p(y) = \log\tau + d - 2\,\mathrm{softplus}(d)
                        - \log y - \log(1 - y),
            \qquad d = l - \tau \log\frac{y}{1 - y}

        Parameters
        ----------
        value : Tensor
            Point(s) :math:`y \in [0, 1]` at which to evaluate the density.

        Returns
        -------
        Tensor
            Log-density values of the same shape as ``value``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions import RelaxedBernoulli
        >>> d = RelaxedBernoulli(temperature=0.7, probs=0.3)
        >>> d.log_prob(lucid.tensor([0.1, 0.5, 0.9]))
        tensor([0.548, -0.531, -0.5102])
        """
        # ``logit(Y)`` is logistic with location ``l`` and scale ``1/τ``;
        # ``Y = sigmoid(logit(Y))`` adds the Jacobian ``−log y − log(1−y)``.
        # The support is the closed interval but ``logit`` is infinite at
        # its ends, so ``y`` is held in ``[tiny, 1 − eps]`` first — the
        # reference framework's inverse sigmoid does the same, and scores
        # an endpoint by the density one step inside it.
        info = finfo(value.dtype)
        y: Tensor = value.clip(float(info.tiny), 1.0 - float(info.eps))
        log_y: Tensor = y.log()
        log_1my: Tensor = (-y).log1p()
        l: Tensor = self._logits
        tau: Tensor = self.temperature
        diff: Tensor = l - tau * (log_y - log_1my)
        return tau.log() + diff - 2.0 * diff.softplus() - log_y - log_1my

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`RelaxedBernoulli` instance with parameters expanded to ``batch_shape``."""
        return self._expand_probs_or_logits(RelaxedBernoulli, batch_shape, _instance, "temperature")


class RelaxedOneHotCategorical(Distribution):
    r"""Concrete distribution — a continuous relaxation of OneHotCategorical.

    ``RelaxedOneHotCategorical(temperature=τ, probs=p)`` defines a
    distribution over the open :math:`K`-simplex whose samples are
    differentiable surrogates for one-hot categorical samples.  As
    :math:`\tau \to 0` the distribution concentrates on the :math:`K`
    vertices of the simplex (recovering one-hot samples); as
    :math:`\tau \to \infty` samples approach the uniform distribution over
    the simplex.

    This is the **Gumbel-softmax** distribution of Jang et al. (2017) and the
    **Concrete distribution** of Maddison et al. (2017).  The
    straight-through estimator (``hard=True`` in the sampler) rounds to
    one-hot in the forward pass while using the soft sample for gradients.

    Parameters
    ----------
    temperature : Tensor | float
        Temperature parameter :math:`\tau > 0`.  Smaller values give
        sharper (more discrete) samples.
    probs : Tensor | None, optional
        Probability vector(s) of shape ``(..., K)``.  Normalised internally.
        Mutually exclusive with ``logits``.
    logits : Tensor | None, optional
        Unnormalised log-probability vector(s) of shape ``(..., K)``.
        Mutually exclusive with ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    temperature : Tensor
        Temperature :math:`\tau`.
    probs : Tensor
        Normalised probability vector — stored when given, otherwise derived
        from ``logits`` on access.
    logits : Tensor
        Log-probability vector — as given, otherwise derived from ``probs``
        on access.

    Notes
    -----
    **Reparameterised sampling** (Gumbel-softmax trick):

    .. math::

        y_k = \frac{\exp\!\bigl((l_k + g_k)/\tau\bigr)}
                    {\sum_j \exp\!\bigl((l_j + g_j)/\tau\bigr)},
        \quad g_k \overset{\text{iid}}{\sim} \operatorname{Gumbel}(0, 1)

    The result :math:`y \in \Delta^{K-1}` (open simplex) is differentiable
    w.r.t. :math:`l` and :math:`\tau`.

    **Log-PDF** (Maddison et al. 2017, Eq. 1):

    .. math::

        \log p(y; l, \tau) = \log\Gamma(K)
                             + (K-1)\log\tau
                             + \sum_k (l_k - (\tau+1)\log y_k)
                             - K \operatorname{logsumexp}_k(l_k - \tau\log y_k)

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import RelaxedOneHotCategorical
    >>> dist = RelaxedOneHotCategorical(
    ...     temperature=0.5,
    ...     probs=lucid.tensor([0.1, 0.4, 0.5]),
    ... )
    >>> samples = dist.rsample((50,))
    >>> samples.shape  # (50, 3) — lies on the open simplex
    (50, 3)
    >>> bool(((samples.sum(dim=-1) - 1.0).abs() < 1e-5).all())  # each row sums to ~1
    True
    """

    # The relaxation temperature divides the logits, so zero is not a
    # degenerate-but-valid setting — it is a division by zero, and the
    # sample comes back non-finite with nothing raised.  Declared
    # ``real`` here, it was never checked; the reference does not check it
    # either, and produces the same non-finite draw.  Stricter than the
    # reference on purpose — listed in the constraint-parity test's
    # exceptions.
    arg_constraints = {"temperature": positive, "probs": simplex, "logits": real_vector}
    support: Constraint | None = simplex
    has_rsample = True

    def __init__(
        self,
        temperature: Tensor | float,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        """Construct a RelaxedOneHotCategorical (Concrete) distribution.

        Parameters
        ----------
        temperature : Tensor | float
            Temperature :math:`\\lambda > 0` controlling relaxation tightness.
            As :math:`\\lambda \\to 0`, samples concentrate on the vertices of
            the simplex recovering a hard one-hot Categorical; as
            :math:`\\lambda \\to \\infty`, samples concentrate near the simplex
            centroid :math:`(1/K, \\ldots, 1/K)`.
        probs : Tensor | None, optional
            Probability vector of shape ``(..., K)`` on the K-simplex.
            Rows are automatically normalised to sum to 1.  Mutually exclusive
            with ``logits``.
        logits : Tensor | None, optional
            Unnormalised log-probabilities of shape ``(..., K)``. Mutually
            exclusive with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` and ``logits`` are provided.
        """
        from lucid.distributions.categorical import _normalize_logits, _normalize_probs

        raw, self._is_logits = _require_probs_or_logits("RelaxedOneHotCategorical", probs, logits)
        self.temperature = _as_tensor(temperature)
        if self._is_logits:
            self.logits = _normalize_logits(raw)
            shape = tuple(self.logits.shape)
        else:
            self.probs = _normalize_probs(raw)
            shape = tuple(self.probs.shape)
        # A temperature given as a number is a 0-dim host tensor; it follows
        # the parameters to their device so the two can meet on Metal.
        self.temperature = _align_device(
            self.temperature, self.logits if self._is_logits else self.probs
        )
        self._num_events = shape[-1]
        super().__init__(
            batch_shape=shape[:-1], event_shape=shape[-1:], validate_args=validate_args
        )

    @_lazy_param
    def probs(self) -> Tensor:
        """Normalised probabilities — as given, or ``softmax(logits)``."""
        from lucid.nn.functional.activations import softmax

        return softmax(self.logits, dim=-1)

    @_lazy_param
    def logits(self) -> Tensor:
        """Log-probabilities — as given, or derived from ``probs`` on access.

        Derived as ``log(probs)`` with ``probs`` clamped one epsilon inside
        ``[0, 1]``, as the reference framework derives it.
        """
        return _clamp_probs(self.probs).log()

    @property
    def _logits(self) -> Tensor:
        """Lazily resolved per-category logits.

        Returns ``self.logits`` when the distribution was constructed from
        logits; otherwise returns ``self.probs.log()``.
        """
        if self._is_logits:
            return self.logits
        return self.probs.log()

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw a reparameterised sample via the Gumbel-softmax trick.

        Computes the Concrete relaxation directly on ``logits`` and
        ``temperature``, ensuring the output lies strictly inside the open
        :math:`K`-simplex and gradients propagate through both ``logits`` and
        ``temperature``.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Samples on the open simplex of shape
            ``(*sample_shape, *batch_shape, K)``.
        """
        from lucid.nn.functional.activations import softmax

        out_shape = tuple(sample_shape) + tuple(self._batch_shape) + tuple(self._event_shape)
        l = self._logits + lucid.zeros(
            out_shape, dtype=self._logits.dtype, device=self._logits.device
        )
        u = lucid.rand(*out_shape, dtype=l.dtype, device=l.device).clip(1e-7, 1.0 - 1e-7)
        g = -(-(u.log())).log()
        return softmax((l + g) / self.temperature.unsqueeze(-1), dim=-1)

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        """Log-probability density of the Concrete distribution over the simplex.

        Parameters
        ----------
        value : Tensor
            Point(s) :math:`y` on the open :math:`K`-simplex, shape
            ``(..., K)``.

        Returns
        -------
        Tensor
            Log-density values, shape ``(...,)`` (batch dimensions only).
        """
        # Concrete density:  log p(y) = log Γ(K) + (K−1)·log τ
        #                   + Σ (logits − (τ+1)·log y) − K·logsumexp(logits − τ·log y).
        K: int = self._num_events
        l: Tensor = self._logits
        tau: Tensor = self.temperature.unsqueeze(-1)
        log_y: Tensor = value.log()
        log_z: Tensor = lucid.logsumexp(l - tau * log_y, dim=-1)
        return (
            float(_lgamma_int(K))
            + (K - 1.0) * self.temperature.log()
            + (l - (tau + 1.0) * log_y).sum(dim=-1)
            - K * log_z
        )

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`RelaxedOneHotCategorical` expanded to ``batch_shape``."""
        new = self._get_checked_instance(RelaxedOneHotCategorical, _instance)
        batch_shape = tuple(batch_shape)
        param_shape = batch_shape + (self._num_events,)
        new.temperature = self.temperature.expand(batch_shape)
        new._num_events = self._num_events
        new._is_logits = self._is_logits
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(param_shape)
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(param_shape)
        super(RelaxedOneHotCategorical, new).__init__(
            batch_shape=batch_shape, event_shape=self._event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new


def _lgamma_int(n: int) -> float:
    """``log(Γ(n)) = log((n-1)!)`` for small positive integers."""
    import math

    return math.lgamma(n)
