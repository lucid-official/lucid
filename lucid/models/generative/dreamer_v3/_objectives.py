r"""The three quantities DreamerV3 changes about how a world model is scored.

Each one exists to remove a number the practitioner would otherwise have
had to supply per domain, and each is written here rather than inline
because each has a failure mode worth naming in one place.
"""

from typing import cast, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.models.generative._common._rssm import RSSMState, categorical_kl

__all__ = ["free_bits_kl", "percentile", "ReturnNormaliser"]


def free_bits_kl(
    posterior: RSSMState,
    prior: RSSMState,
    *,
    dyn_scale: float = 1.0,
    rep_scale: float = 0.1,
    free_nats: float = 1.0,
) -> tuple[Tensor, Tensor]:
    r"""The divergence, split by direction and floored per step.

    Parameters
    ----------
    posterior, prior : RSSMState
        Both categorical, from the same step.
    dyn_scale, rep_scale : float, default=1.0, 0.1
        :math:`\beta_{\mathrm{dyn}}` and :math:`\beta_{\mathrm{rep}}`.
    free_nats : float, default=1.0
        The floor.  Below it a term contributes a constant, so no gradient
        is spent closing a gap that is already small.

    Returns
    -------
    dynamics, representation : Tensor
        Scalars, already scaled.  Returned apart because they are the two
        halves the paper reports separately and the pair is what a reader
        checks when a latent collapses.

    Raises
    ------
    ValueError
        If either state is Gaussian.

    Notes
    -----
    .. math::

        \mathcal{L}_{\mathrm{dyn}} = \max\big(\text{free},\;
            \mathrm{KL}[\mathrm{sg}(q) \,\|\, p]\big),
        \qquad
        \mathcal{L}_{\mathrm{rep}} = \max\big(\text{free},\;
            \mathrm{KL}[q \,\|\, \mathrm{sg}(p)]\big).

    The clip is applied **per step, before averaging**, which is the whole
    point and easy to get backwards.  Clipping the mean lets a batch where
    most steps are already below the floor be dragged by a few that are
    not; clipping each step first means those steps simply stop
    contributing.  DreamerV2's ``free_nats`` clipped the mean, and that
    difference is part of why this one does not need a tuned KL scale.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._rssm import RSSMState
    >>> state = RSSMState(
    ...     deter=lucid.zeros(2, 8),
    ...     stoch=lucid.zeros(2, 4, 8),
    ...     mean=None,
    ...     std=None,
    ...     logits=lucid.zeros(2, 4, 8),
    ... )
    >>> from lucid.models.generative.dreamer_v3._objectives import (
    ...     free_bits_kl,
    ... )
    >>> dynamics, representation = free_bits_kl(
    ...     state, state, dyn_scale=1.0, rep_scale=1.0, free_nats=1.0
    ... )
    >>> float(dynamics.item()), float(representation.item())
    (1.0, 1.0)

    Two terms rather than v2's single balanced one, each clamped at the
    free-nats floor on its own. Returning them apart is what lets the
    two be scaled differently — v3 weights the dynamics term far above
    the representation one.
    """
    if not (posterior.is_discrete and prior.is_discrete):
        raise ValueError("free bits here expects two categorical states")
    if free_nats < 0.0:
        raise ValueError(f"free_nats must be non-negative, got {free_nats}")

    posterior_logits = cast(Tensor, posterior.logits)
    prior_logits = cast(Tensor, prior.logits)

    toward_prior = categorical_kl(posterior_logits.detach(), prior_logits)
    toward_posterior = categorical_kl(posterior_logits, prior_logits.detach())

    def floored(value: Tensor) -> Tensor:
        return value.clip(free_nats, None).mean()

    return dyn_scale * floored(toward_prior), rep_scale * floored(toward_posterior)


def percentile(values: Tensor, fraction: float) -> Tensor:
    r"""The ``fraction``-th percentile of a flattened tensor.

    Parameters
    ----------
    values : Tensor
        Any shape; flattened first.
    fraction : float
        In ``[0, 100]``.

    Returns
    -------
    Tensor
        A scalar, by nearest-rank on the sorted values.

    Notes
    -----
    Nearest-rank rather than interpolated.  The consumer is a scale
    estimate smoothed by a long moving average, so the difference between
    the two conventions is far below the noise the average is there to
    remove, and nearest-rank needs no gather.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dreamer_v3._objectives import percentile
    >>> values = lucid.tensor([float(i) for i in range(1, 101)])
    >>> float(percentile(values, 95).item())
    95.0

    ``fraction`` is in [0, 100], not [0, 1] — 0.5 asks for the half-th
    percentile and returns the smallest value, not the median.

    >>> float(percentile(values, 0.5).item())
    1.0
    """
    if not 0.0 <= fraction <= 100.0:
        raise ValueError(f"fraction must be in [0, 100], got {fraction}")
    flat = values.reshape(-1)
    count = int(flat.shape[0])
    ordered = lucid.sort(flat).values
    index = min(count - 1, max(0, int(round(fraction / 100.0 * (count - 1)))))
    return ordered[index]


class ReturnNormaliser(nn.Module):
    r"""A slow estimate of how wide the returns currently are.

    Parameters
    ----------
    decay : float, default=0.99
        Weight kept by the running estimate at each update.
    low, high : float, default=5.0, 95.0
        Percentiles whose difference is the spread.
    floor : float, default=1.0
        The divisor is ``max(floor, spread)``.

    Attributes
    ----------
    spread : Tensor
        The running estimate, a 0-d buffer.

    Notes
    -----
    DreamerV3 divides the actor's objective by

    .. math::

        S = \mathrm{EMA}\big(\mathrm{Per}(R^\lambda, 95)
                           - \mathrm{Per}(R^\lambda, 5),\; 0.99\big),

    which is what lets a single entropy coefficient mean the same thing
    in a domain where returns are 0.01 and one where they are 10000 —
    the bonus is traded against a quantity that has been made
    dimensionless.

    Two details carry the design.  The **percentile range** rather than
    the standard deviation, because a handful of enormous returns should
    not shrink every other gradient.  And the **floor at one**, because
    dividing by a genuinely small spread would amplify what is mostly
    noise: below that width the returns are left in their own units.

    The estimate is a tensor updated in place, not a Python float read
    back from the device each step: that read stalled every step on a
    device sync and made a compiled training step impossible, since a
    value read on the host is frozen into the compiled graph.  It is a
    *non-persistent* buffer — it follows the model across devices, but it
    is a statistic of the current returns, not a weight, and stays out of
    ``state_dict``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dreamer_v3._objectives import ReturnNormaliser
    >>> norm = ReturnNormaliser()
    >>> scaled = norm(lucid.randn((4, 8)))
    >>> scaled.shape
    (4, 8)
    """

    spread: Tensor

    def __init__(
        self,
        decay: float = 0.99,
        low: float = 5.0,
        high: float = 95.0,
        floor: float = 1.0,
    ) -> None:
        """Initialise the estimate. See the class docstring for parameters."""
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if not 0.0 <= low < high <= 100.0:
            raise ValueError(f"percentiles must satisfy 0 <= low < high <= 100, got {low}, {high}")
        if floor <= 0.0:
            raise ValueError(f"floor must be positive, got {floor}")
        self.decay = decay
        self.low = low
        self.high = high
        self.floor = floor
        self.register_buffer("spread", lucid.zeros(()), persistent=False)

    @property
    def scale(self) -> Tensor:
        """The current divisor, without folding anything in.

        Returns
        -------
        Tensor
            ``max(floor, spread)``, 0-d — what :meth:`update` would return
            if this batch happened to match the running estimate exactly.

        Notes
        -----
        Exists so that a model in evaluation mode can read the estimate
        without moving it.  Scoring a policy is not evidence about the
        return distribution the policy is being trained on, and letting
        an evaluation pass update the divisor makes a run's numbers
        depend on how often it was measured.
        """
        return self.spread.clip(self.floor, None)

    @lucid.no_grad()
    def update(self, returns: Tensor) -> Tensor:
        """Fold this batch's spread into the estimate.

        Parameters
        ----------
        returns : Tensor
            Lambda-returns, any shape, on the estimate's device.

        Returns
        -------
        Tensor
            The divisor after the update — ``max(floor, spread)``, 0-d.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.dreamer_v3._objectives import (
        ...     ReturnNormaliser)
        >>> norm = ReturnNormaliser()
        >>> returns = lucid.tensor([float(i) for i in range(101)])
        >>> float(norm.update(returns))
        1.0

        This batch's 5th-to-95th percentile spread is 90, but one update
        moves the estimate only ``1 - decay`` of the way there, and the
        floor holds the divisor at 1 until the estimate passes it:

        >>> round(float(norm.spread), 4)
        0.9
        >>> for _ in range(99):
        ...     scale = norm.update(returns)
        >>> round(float(scale), 2)
        57.06
        """
        observed = percentile(returns, self.high) - percentile(returns, self.low)
        self.spread.mul_(self.decay).add_(observed * (1.0 - self.decay))
        return self.scale

    @override
    def forward(self, returns: Tensor) -> Tensor:
        """Update the estimate and return the scaled returns.

        Parameters
        ----------
        returns : Tensor
            Lambda-returns.

        Returns
        -------
        Tensor
            ``returns / max(floor, spread)`` — the divisor is computed under
            ``no_grad``, so nothing differentiates through the normalisation.
        """
        return cast(Tensor, returns / self.update(returns))
