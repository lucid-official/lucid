"""A truncated normal, for the actor that has to stay inside its box.

Dreamer squashed its Gaussian with ``tanh``; DreamerV2 truncates one
instead.  The difference matters for the entropy bonus this family adds:
a squashed normal has no closed-form entropy — the released implementation
estimates it by sampling — whereas a truncated one does, so the bonus is
exact and costs nothing.

Shared across the world models rather than owned by one: DreamerV2 needed
these first and DreamerV3 needs the same two, so they sit beside the RSSM
and the pixel networks.

``lucid.distributions`` is where a truncated normal ought to eventually
live, and it is missing one, but that module has a contract of its own —
constraints, registered divergences, transform composition — and adding a
half-conforming member to it would be worse than this.
"""

import math

import lucid
import lucid.nn.functional as F
from lucid.models.generative._common._rssm import _gumbel_argmax
from lucid._tensor.tensor import Tensor

__all__ = ["TruncatedNormal", "OneHotCategorical"]

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def _standard_pdf(x: Tensor) -> Tensor:
    """Standard normal density."""
    return lucid.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _standard_cdf(x: Tensor) -> Tensor:
    """Standard normal CDF, via ``erf``."""
    return 0.5 * (1.0 + lucid.erf(x / _SQRT_2))


class TruncatedNormal:
    r"""A normal restricted to ``[low, high]`` and renormalised.

    Parameters
    ----------
    loc, scale : Tensor
        The underlying normal's parameters, broadcast together.
    low, high : float, default=-1.0, 1.0
        The interval kept.
    epsilon : float, default=1e-6
        Floor on the surviving probability mass, and the margin samples
        are held away from the endpoints.  Without it a ``scale`` driven
        small next to a bound leaves ``Z`` at zero and every quantity
        divides by it.

    Notes
    -----
    Writing :math:`\alpha = (a - \mu)/\sigma`, :math:`\beta = (b - \mu)/\sigma`
    and :math:`Z = \Phi(\beta) - \Phi(\alpha)`:

    .. math::

        \log p(x) = -\tfrac{1}{2}\Big(\tfrac{x - \mu}{\sigma}\Big)^2
                    - \log \sigma - \tfrac{1}{2}\log 2\pi - \log Z,

    .. math::

        H = \log\!\big(\sqrt{2\pi e}\, \sigma Z\big)
            + \frac{\alpha\,\phi(\alpha) - \beta\,\phi(\beta)}{2 Z}.

    Sampling inverts the CDF, so it is reparameterised: a uniform draw is
    mapped through :math:`\Phi^{-1}`, and the gradient reaches ``loc`` and
    ``scale`` through the affine step that follows.  That is what lets the
    actor's gradient come through the dynamics.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._dists import TruncatedNormal
    >>> dist = TruncatedNormal(lucid.zeros((2, 3)), lucid.ones((2, 3)))
    >>> sample = dist.rsample()
    >>> bool((sample.abs() <= 1.0).all().item())
    True
    >>> dist.entropy().shape
    (2, 3)
    """

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        low: float = -1.0,
        high: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        """Initialise the distribution. See the class docstring."""
        if high <= low:
            raise ValueError(f"high must exceed low, got low={low}, high={high}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.epsilon = epsilon

        self._alpha = (low - loc) / scale
        self._beta = (high - loc) / scale
        self._cdf_alpha = _standard_cdf(self._alpha)
        self._z = (_standard_cdf(self._beta) - self._cdf_alpha).clip(epsilon, None)

    @property
    def mean(self) -> Tensor:
        r"""The truncated mean, :math:`\mu + \sigma (\phi(\alpha) - \phi(\beta)) / Z`."""
        shift = (_standard_pdf(self._alpha) - _standard_pdf(self._beta)) / self._z
        return self.loc + self.scale * shift

    @property
    def mode(self) -> Tensor:
        """The density's peak — ``loc``, clipped into the interval."""
        return self.loc.clip(self.low + self.epsilon, self.high - self.epsilon)

    def rsample(self) -> Tensor:
        """Draw reparameterised, so gradients reach ``loc`` and ``scale``.

        Returns
        -------
        Tensor
            A sample strictly inside ``(low, high)``.

        Notes
        -----
        Inverse-CDF sampling: a uniform draw is placed inside the surviving
        mass and pushed back through :math:`\\Phi^{-1}`.  The result is
        clipped a hair away from the endpoints, because a sample landing
        exactly on one makes ``log_prob`` there finite but its gradient
        enormous.
        """
        shape = tuple(int(s) for s in self.loc.shape)
        uniform = lucid.rand(shape, device=self.loc.device, dtype=self.loc.dtype)
        probability = (self._cdf_alpha + uniform * self._z).clip(self.epsilon, 1.0 - self.epsilon)
        standard = _SQRT_2 * lucid.erfinv(2.0 * probability - 1.0)
        drawn = self.loc + self.scale * standard
        return drawn.clip(self.low + self.epsilon, self.high - self.epsilon)

    def log_prob(self, value: Tensor) -> Tensor:
        """Log density at ``value``.

        Parameters
        ----------
        value : Tensor
            Broadcastable against ``loc``.  Values outside the interval are
            not special-cased; they are the caller's error, and clipping
            them would hide it behind a plausible number.

        Returns
        -------
        Tensor
            Elementwise log density.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative._common._dists import TruncatedNormal
        >>> dist = TruncatedNormal(lucid.tensor([0.5]), lucid.tensor([0.3]))
        >>> grid = lucid.linspace(-1.0, 1.0, 2001)
        >>> density = dist.log_prob(grid).exp()
        >>> density.shape
        (2001,)
        >>> round(float(density.sum().item()) * 0.001, 3)  # mass on [-1, 1]
        1.0
        """
        standard = (value - self.loc) / self.scale
        return -0.5 * standard * standard - self.scale.log() - _LOG_SQRT_2PI - self._z.log()

    def entropy(self) -> Tensor:
        r"""Differential entropy, in closed form.

        Returns
        -------
        Tensor
            Elementwise entropy.

        Notes
        -----
        This is why the family truncates rather than squashing: a
        ``tanh``-squashed normal has no closed-form entropy, so the
        released Dreamer estimates it from samples, and an entropy bonus
        built on an estimate is noisier than the thing it regularises.
        """
        correction = (
            self._alpha * _standard_pdf(self._alpha) - self._beta * _standard_pdf(self._beta)
        ) / (2.0 * self._z)
        return (
            0.5 * math.log(2.0 * math.pi * math.e) + self.scale.log() + self._z.log() + correction
        )


class OneHotCategorical:
    r"""A categorical over actions, sampled as a one-hot with a gradient.

    What the actor becomes when the action space is discrete.  Atari has
    eighteen buttons, not a box, and there is no reparameterisation of a
    button press — which is why the paper switches to REINFORCE there and
    why this carries ``log_prob``.

    Parameters
    ----------
    logits : Tensor
        Unnormalised scores, ``(..., action_dim)``.
    unimix : float, default=0.0
        Uniform mass mixed into the probabilities before anything else
        reads them.  DreamerV3 sets 1%, so no action's probability can
        reach zero — and a probability at zero is a log-probability at
        :math:`-\infty` and a gradient that never comes back.

    Raises
    ------
    ValueError
        If ``unimix`` is outside ``[0, 1)``.

    Notes
    -----
    The draw is straight-through, exactly as the latent's is: the one-hot
    goes forward and the probabilities' gradient comes back.  That leaves
    a usable gradient even under ``actor_grad="dynamics"``, though it is
    biased, which is the reason the reference prefers the score-function
    estimator for discrete actions rather than the analytic path.

    ``self.logits`` is the *mixed* score, not the argument: everything
    downstream — the sample, the log-probability, the entropy — has to
    describe one distribution, and it is the mixed one.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._dists import OneHotCategorical
    >>> dist = OneHotCategorical(lucid.zeros((2, 3, 4)))
    >>> sample = dist.rsample()
    >>> bool((sample.sum(dim=-1) - 1.0).abs().max().item() < 1e-6)
    True
    >>> dist.entropy().shape
    (2, 3)
    """

    def __init__(self, logits: Tensor, unimix: float = 0.0) -> None:
        """Initialise the distribution. See the class docstring."""
        if not 0.0 <= unimix < 1.0:
            raise ValueError(f"unimix must be in [0, 1), got {unimix}")
        if unimix:
            classes = int(logits.shape[-1])
            probs = F.softmax(logits, dim=-1)
            logits = ((1.0 - unimix) * probs + unimix / classes).log()
        self.logits = logits
        self.log_probs = F.log_softmax(logits, dim=-1)
        self.probs = self.log_probs.exp()

    @property
    def mode(self) -> Tensor:
        """The most likely action, one-hot."""
        classes = int(self.logits.shape[-1])
        index = self.probs.argmax(dim=-1)
        hard = F.one_hot(index, num_classes=classes)
        return hard.to(self.probs.device).to(self.probs.dtype)

    def rsample(self) -> Tensor:
        """Draw a one-hot, with a straight-through gradient.

        Returns
        -------
        Tensor
            ``(..., action_dim)``, one entry set.

        Notes
        -----
        Drawn with the Gumbel-max trick, which is exact and stays on
        the accelerator.  ``multinomial`` is the obvious call and the
        wrong one: it is data-dependent, so on Metal it round-trips to
        the CPU once per draw.
        """
        classes = int(self.logits.shape[-1])
        hard = F.one_hot(_gumbel_argmax(self.logits), num_classes=classes)
        hard = hard.to(self.probs.device).to(self.probs.dtype)
        return F.straight_through(hard, self.probs)

    def log_prob(self, value: Tensor) -> Tensor:
        """Log probability of a one-hot action.

        Parameters
        ----------
        value : Tensor
            One-hot, ``(..., action_dim)``.

        Returns
        -------
        Tensor
            ``(...)`` — the event axis is summed away, because a one-hot
            action is a single event rather than a vector of independent
            ones.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative._common._dists import OneHotCategorical
        >>> dist = OneHotCategorical(lucid.zeros((2, 4)))
        >>> action = lucid.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        >>> dist.log_prob(action).shape
        (2,)
        >>> round(float(dist.log_prob(action)[0].item()), 4)  # log(1 / 4)
        -1.3863
        >>> mixed = OneHotCategorical(lucid.tensor([[2.0, 0.0, -1.0]]), unimix=0.01)
        >>> every_action = lucid.eye(3)  # one row per one-hot choice
        >>> round(float(mixed.log_prob(every_action).exp().sum().item()), 5)
        1.0
        """
        return (value * self.log_probs).sum(dim=-1)

    def entropy(self) -> Tensor:
        r"""Shannon entropy, ``(...)``.

        Returns
        -------
        Tensor
            Non-negative, at most ``log(action_dim)``.

        Notes
        -----
        Computed as :math:`\mathrm{logsumexp}(\ell) - \sum_k p_k \ell_k`
        rather than the textbook :math:`-\sum_k p_k \log p_k`.  The two
        are the same number, but the textbook form multiplies a
        probability that has underflowed to zero by a log-probability that
        has gone to :math:`-\infty`, and ``0 * -inf`` is ``NaN``.  A
        policy that has become confident — which is what training is for —
        walks straight into that.  This form only ever multiplies finite
        logits by finite probabilities.

        The normaliser is computed directly, shifting by the maximum,
        rather than recovered as ``logits - log_probs``.  The shorter
        route reads the same on paper and inherits whatever the
        log-probabilities carry; this one depends on nothing but the
        logits.
        """
        maximum = self.logits.max(dim=-1, keepdim=True).values
        shifted = (self.logits - maximum).exp().sum(dim=-1, keepdim=True)
        normaliser = (maximum + shifted.log()).squeeze(dim=-1)
        return normaliser - (self.probs * self.logits).sum(dim=-1)
