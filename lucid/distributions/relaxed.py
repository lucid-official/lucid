"""Concrete (Gumbel-softmax) relaxed distributions.

* :class:`RelaxedBernoulli` — relaxation of ``Bernoulli`` whose samples
  live in ``[0, 1]`` and pass gradients through.
* :class:`RelaxedOneHotCategorical` — Concrete distribution over the
  open simplex; relaxation of :class:`OneHotCategorical`.

As in the reference framework each is a
:class:`~lucid.distributions.TransformedDistribution`: the sigmoid of a
:class:`LogitRelaxedBernoulli`, and the exponential of an
:class:`ExpRelaxedCategorical`.  The two base distributions hold the
parameters and the sampling math (Gumbel noise from the Lucid Philox
stream); the density on the interval or the simplex is the change of
variables through the transform.
"""

import math
from collections.abc import Sequence
from typing import Self, override

import lucid
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
from lucid.distributions.transformed_distribution import TransformedDistribution
from lucid.distributions.transforms import ExpTransform, SigmoidTransform


class LogitRelaxedBernoulli(Distribution):
    r"""Logit of a :class:`RelaxedBernoulli` — the Gumbel-sigmoid noise before the sigmoid.

    A sample is the pre-activation :math:`z = (l + g_1 - g_2)/\tau` of the
    Concrete relaxation; :class:`RelaxedBernoulli` is its sigmoid.  Its
    support is the whole real line.

    Parameters
    ----------
    temperature : Tensor | float
        Temperature :math:`\tau > 0` dividing the logits.
    probs : Tensor | float | None, optional
        Bernoulli success probability :math:`p \in [0, 1]`.  Mutually
        exclusive with ``logits``.
    logits : Tensor | float | None, optional
        Log-odds :math:`l`.  Mutually exclusive with ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    The attributes ``temperature``, ``probs`` and ``logits`` are those of
    :class:`RelaxedBernoulli`.  Log-density of :math:`z` (the logistic
    density of scale :math:`1/\tau`):

    .. math::

        \log p(z; l, \tau) = \log\tau + d - 2\,\operatorname{softplus}(d),
        \qquad d = l - \tau z

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.relaxed import LogitRelaxedBernoulli
    >>> d = LogitRelaxedBernoulli(temperature=0.5, probs=0.7)
    >>> d.rsample((3,)).shape
    (3,)
    """

    # The relaxation temperature divides the logits, so zero is not a
    # degenerate-but-valid setting — it is a division by zero, and the
    # sample comes back non-finite with nothing raised.  The reference does
    # not check it either, and produces the same non-finite draw.  Stricter
    # than the reference on purpose — listed in the constraint-parity test's
    # exceptions.  A ``probs`` of 0 or 1 is held one epsilon inside the
    # interval when it is turned into ``logits``.
    arg_constraints = {"temperature": positive, "probs": unit_interval, "logits": real}
    support: Constraint | None = real
    has_rsample = True

    def __init__(
        self,
        temperature: Tensor | float,
        probs: Tensor | float | None = None,
        logits: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        """Construct the distribution; raises ``ValueError`` unless exactly one of ``probs``/``logits`` is given."""
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

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw a reparameterised logit sample via the Gumbel-sigmoid trick.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Samples :math:`z = (l + g_1 - g_2)/\tau` of shape
            ``(*sample_shape, *batch_shape)``, with gradients flowing through
            both ``logits`` and ``temperature``.
        """
        shape = self._extended_shape(sample_shape)
        logits: Tensor = self.logits
        l: Tensor = logits + lucid.zeros(shape, dtype=logits.dtype, device=logits.device)
        u1: Tensor = lucid.rand(*shape, dtype=l.dtype, device=l.device).clip(1e-7, 1.0 - 1e-7)
        u2: Tensor = lucid.rand(*shape, dtype=l.dtype, device=l.device).clip(1e-7, 1.0 - 1e-7)
        # Gumbel(0,1) = −log(−log U).
        g1: Tensor = -(-(u1.log())).log()
        g2: Tensor = -(-(u2.log())).log()
        return (l + g1 - g2) / self.temperature

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of the logit-space Concrete distribution.

        .. math::

            \log p(z) = \log\tau + d - 2\,\operatorname{softplus}(d),
            \qquad d = l - \tau z

        Parameters
        ----------
        value : Tensor
            Point(s) :math:`z \in \mathbb{R}` at which to evaluate the density.

        Returns
        -------
        Tensor
            Log-density values of the same shape as ``value``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions.relaxed import LogitRelaxedBernoulli
        >>> d = LogitRelaxedBernoulli(temperature=0.5, probs=0.5)
        >>> d.log_prob(lucid.tensor(0.0))
        tensor(-2.079)
        """
        tau: Tensor = self.temperature
        diff: Tensor = self.logits - tau * value
        return tau.log() + diff - 2.0 * diff.softplus()

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`LogitRelaxedBernoulli` instance with parameters expanded to ``batch_shape``."""
        return self._expand_probs_or_logits(
            LogitRelaxedBernoulli, batch_shape, _instance, "temperature"
        )


class _RelaxedFamily(TransformedDistribution):
    """A relaxed family: a transformed base distribution that owns the parameters.

    The parameters live on ``base_dist`` and the family reads them through
    properties, as the reference does.  The base validated them when it was
    built, so the family does not check them a second time, and asks the
    base which of ``probs`` / ``logits`` was only derived.
    """

    @override
    def _validate_params(self) -> None:
        """No-op: ``base_dist`` validated the parameters it owns at construction."""

    @override
    def _is_pending(self, name: str) -> bool:
        """Whether ``name`` is a dual ``base_dist`` derived rather than was given."""
        return self.base_dist._is_pending(name)


class RelaxedBernoulli(_RelaxedFamily):
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

    It is a :class:`~lucid.distributions.TransformedDistribution`: a
    :class:`LogitRelaxedBernoulli` (``base_dist``) pushed through a
    :class:`~lucid.distributions.transforms.SigmoidTransform` (``transforms``).

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
    base_dist : LogitRelaxedBernoulli
        The distribution of the pre-sigmoid logit.
    transforms : list[Transform]
        ``[SigmoidTransform()]``.

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
    the Jacobian of the sigmoid that maps it into :math:`(0, 1)`.  A value
    of 0 or 1 is scored one step inside the interval, as the reference
    framework's inverse sigmoid does.

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

    arg_constraints = {"temperature": positive, "probs": unit_interval, "logits": real}
    support: Constraint | None = unit_interval
    has_rsample = True
    base_dist: LogitRelaxedBernoulli

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
        base_dist = LogitRelaxedBernoulli(temperature, probs, logits, validate_args=validate_args)
        super().__init__(base_dist, SigmoidTransform(), validate_args=validate_args)

    @property
    def temperature(self) -> Tensor:
        """Temperature :math:`\\tau` of the relaxation."""
        return self.base_dist.temperature

    @property
    def probs(self) -> Tensor:
        """Success probability — as given, or ``sigmoid(logits)``."""
        return self.base_dist.probs

    @property
    def logits(self) -> Tensor:
        """Log-odds — as given, or derived from ``probs``."""
        return self.base_dist.logits

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`RelaxedBernoulli` instance with parameters expanded to ``batch_shape``."""
        new = self._get_checked_instance(RelaxedBernoulli, _instance)
        return super().expand(batch_shape, _instance=new)


class ExpRelaxedCategorical(Distribution):
    r"""Logarithm of a :class:`RelaxedOneHotCategorical` — Gumbel-softmax before the exponential.

    A sample is the log-simplex point :math:`z_k = s_k - \operatorname{logsumexp}_j s_j`
    with :math:`s = (l + g)/\tau`; :class:`RelaxedOneHotCategorical` is its
    exponential.  Its support is :math:`\mathbb{R}^K`.

    Parameters
    ----------
    temperature : Tensor | float
        Temperature :math:`\tau > 0`.
    probs : Tensor | None, optional
        Probability vector(s) of shape ``(..., K)``.  Normalised internally.
        Mutually exclusive with ``logits``.
    logits : Tensor | None, optional
        Unnormalised log-probability vector(s) of shape ``(..., K)``.
        Mutually exclusive with ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    The attributes ``temperature``, ``probs`` and ``logits`` are those of
    :class:`RelaxedOneHotCategorical`.  Log-density of :math:`z`:

    .. math::

        \log p(z; l, \tau) = \log\Gamma(K) + (K-1)\log\tau
            + \sum_k \bigl(s_k - \operatorname{logsumexp}_j s_j\bigr),
        \qquad s = l - \tau z

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions.relaxed import ExpRelaxedCategorical
    >>> d = ExpRelaxedCategorical(0.5, probs=lucid.tensor([0.2, 0.3, 0.5]))
    >>> d.rsample((4,)).shape
    (4, 3)
    """

    # See ``LogitRelaxedBernoulli``: the temperature check is stricter than
    # the reference on purpose.
    arg_constraints = {"temperature": positive, "probs": simplex, "logits": real_vector}
    support: Constraint | None = real_vector
    has_rsample = True

    def __init__(
        self,
        temperature: Tensor | float,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        """Construct the distribution; raises ``ValueError`` unless exactly one of ``probs``/``logits`` is given."""
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

    @override
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw a reparameterised log-simplex sample via the Gumbel-softmax trick.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Number of draws, prepended to the batch shape.

        Returns
        -------
        Tensor
            Draws of shape ``(*sample_shape, *batch_shape, K)`` whose
            exponential sums to 1 along the last axis, with gradients
            flowing through both ``logits`` and ``temperature``.
        """
        out_shape = self._extended_shape(sample_shape)
        logits: Tensor = self.logits
        l = logits + lucid.zeros(out_shape, dtype=logits.dtype, device=logits.device)
        u = lucid.rand(*out_shape, dtype=l.dtype, device=l.device).clip(1e-7, 1.0 - 1e-7)
        g = -(-(u.log())).log()
        scores = (l + g) / self.temperature.unsqueeze(-1)
        return scores - lucid.logsumexp(scores, dim=-1, keepdim=True)

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of the log-space Concrete distribution.

        .. math::

            \log p(z) = \log\Gamma(K) + (K-1)\log\tau
                + \sum_k \bigl(s_k - \operatorname{logsumexp}_j s_j\bigr),
            \qquad s = l - \tau z

        Parameters
        ----------
        value : Tensor
            Point(s) :math:`z`, shape ``(..., K)``.

        Returns
        -------
        Tensor
            Log-density values, shape ``(...,)`` (batch dimensions only).
        """
        K: int = self._num_events
        tau: Tensor = self.temperature
        score: Tensor = self.logits - tau.unsqueeze(-1) * value
        score = (score - lucid.logsumexp(score, dim=-1, keepdim=True)).sum(dim=-1)
        return score + (K - 1.0) * tau.log() + math.lgamma(K)

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`ExpRelaxedCategorical` expanded to ``batch_shape``."""
        new = self._get_checked_instance(ExpRelaxedCategorical, _instance)
        batch_shape = tuple(batch_shape)
        param_shape = batch_shape + (self._num_events,)
        new.temperature = self.temperature.expand(batch_shape)
        new._num_events = self._num_events
        new._is_logits = self._is_logits
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(param_shape)
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(param_shape)
        super(ExpRelaxedCategorical, new).__init__(
            batch_shape=batch_shape, event_shape=self._event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new


class RelaxedOneHotCategorical(_RelaxedFamily):
    r"""Concrete distribution — a continuous relaxation of OneHotCategorical.

    ``RelaxedOneHotCategorical(temperature=τ, probs=p)`` defines a
    distribution over the open :math:`K`-simplex whose samples are
    differentiable surrogates for one-hot categorical samples.  As
    :math:`\tau \to 0` the distribution concentrates on the :math:`K`
    vertices of the simplex (recovering one-hot samples); as
    :math:`\tau \to \infty` samples approach the uniform distribution over
    the simplex.

    This is the **Gumbel-softmax** distribution of Jang et al. (2017) and the
    **Concrete distribution** of Maddison et al. (2017).

    It is a :class:`~lucid.distributions.TransformedDistribution`: an
    :class:`ExpRelaxedCategorical` (``base_dist``) pushed through an
    :class:`~lucid.distributions.transforms.ExpTransform` (``transforms``).

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
    base_dist : ExpRelaxedCategorical
        The distribution of the log-simplex point.
    transforms : list[Transform]
        ``[ExpTransform()]``.

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

    arg_constraints = {"temperature": positive, "probs": simplex, "logits": real_vector}
    support: Constraint | None = simplex
    has_rsample = True
    base_dist: ExpRelaxedCategorical

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
        base_dist = ExpRelaxedCategorical(temperature, probs, logits, validate_args=validate_args)
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    @property
    def temperature(self) -> Tensor:
        """Temperature :math:`\\tau` of the relaxation."""
        return self.base_dist.temperature

    @property
    def probs(self) -> Tensor:
        """Normalised probabilities — as given, or ``softmax(logits)``."""
        return self.base_dist.probs

    @property
    def logits(self) -> Tensor:
        """Log-probabilities — as given, or derived from ``probs``."""
        return self.base_dist.logits

    @override
    def expand(self, batch_shape: Sequence[int], _instance: Self | None = None) -> Self:
        """Return a new :class:`RelaxedOneHotCategorical` expanded to ``batch_shape``."""
        new = self._get_checked_instance(RelaxedOneHotCategorical, _instance)
        return super().expand(batch_shape, _instance=new)
