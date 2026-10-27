r"""DreamerV2 — Hafner et al., 2021.

Dreamer's machine with a discrete latent, a balanced divergence, a target
critic and an entropy bonus.  The encoder, decoder, dense heads, recurrent
state-space model and lambda-returns are all the shared ones; what is
written here is the four changes and the objectives that use them.

Sequences are ``(B, T, ...)``, and ``actions[:, t]`` is the action taken
*into* step ``t``, as in the rest of the family.

Three losses, three optimisers — the same contract Dreamer has, for the
same reason, with the same :meth:`DreamerV2ForWorldModeling.backward` to
spend them correctly.
"""

from dataclasses import dataclass
from typing import ClassVar, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._utils._common import update_averages
from lucid.models._tasks import WorldModelingModel
from lucid.models._output import ModelOutput
from lucid.models.generative._common._pixel_nets import (
    DenseHead,
    PixelDecoder,
    PixelEncoder,
)
from lucid.models.generative._common._returns import lambda_return
from lucid.models.generative._common._rssm import RSSM, RSSMState, categorical_kl
from lucid.models.generative.dreamer_v2._config import DreamerV2Config
from lucid.models.generative._common._actor import Actor

__all__ = [
    "DreamerV2Model",
    "DreamerV2ForWorldModeling",
    "DreamerV2Output",
    "DreamerV2BehaviorOutput",
]


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DreamerV2BehaviorOutput(ModelOutput):
    r"""What the imagination pass produces.

    Attributes
    ----------
    actor_loss, value_loss : Tensor
        Scalars, each for its own optimiser.
    lambda_return : Tensor
        The targets, ``(N, H)``, computed from the *target* critic.
    entropy : Tensor
        Mean policy entropy over the scored states — the quantity the
        bonus is paid on, exposed because it is the first thing to look
        at when a policy stops exploring.
    imagined_reward, imagined_value : Tensor
        Along the imagined trajectory, ``(N, H + 1)``.
    imagined_action : Tensor
        What the actor proposed, ``(N, H, action_dim)``.
    imagined_discount : Tensor or None
        Predicted continuation probability, ``(N, H + 1)``; ``None`` when
        the discount is held constant.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import dreamer_v2_world_model
    >>> model = dreamer_v2_world_model(action_dim=2, cnn_depth=2, stoch_size=3,
    ...     discrete=4, deter_size=8, hidden_size=8, actor_hidden=8,
    ...     value_hidden=8, reward_hidden=8, horizon=3, pcont=False)
    >>> out = model(lucid.randn((1, 4, 3, 64, 64)), lucid.randn((1, 4, 2)),
    ...             lucid.randn((1, 4)))
    >>> out.behavior.lambda_return.shape
    (4, 3)
    """

    actor_loss: Tensor
    value_loss: Tensor
    lambda_return: Tensor
    entropy: Tensor
    imagined_reward: Tensor
    imagined_value: Tensor
    imagined_action: Tensor
    imagined_discount: Tensor | None = None


@dataclass(slots=True)
class DreamerV2Output(ModelOutput):
    r"""What :class:`DreamerV2Model` returns after filtering a trajectory.

    Attributes
    ----------
    observation : Tensor
        Reconstruction, ``(B, T, C, 64, 64)``.
    reward, value : Tensor
        Predictions at each posterior state, ``(B, T)``.
    posterior_stoch : Tensor
        The filtered latent, flattened, ``(B, T, stoch_size * discrete)``.
    posterior_logits, prior_logits : Tensor
        Class scores before and after seeing the frame,
        ``(B, T, stoch_size, discrete)``.
    deter : Tensor
        The deterministic path, ``(B, T, D)``.
    loss, recon_loss, reward_loss, kl_loss, pcont_loss : Tensor or None
        World-model terms, set only by
        :class:`DreamerV2ForWorldModeling`.
    behavior : DreamerV2BehaviorOutput or None
        Actor and critic terms, set only by that wrapper.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import dreamer_v2
    >>> model = dreamer_v2(action_dim=2, cnn_depth=2, stoch_size=3, discrete=4,
    ...     deter_size=8, hidden_size=8, actor_hidden=8, value_hidden=8,
    ...     reward_hidden=8)
    >>> out = model(lucid.randn((1, 2, 3, 64, 64)), lucid.randn((1, 2, 2)))
    >>> out.posterior_stoch.shape, out.posterior_logits.shape
    ((1, 2, 12), (1, 2, 3, 4))
    """

    observation: Tensor
    reward: Tensor
    value: Tensor
    posterior_stoch: Tensor
    posterior_logits: Tensor
    prior_logits: Tensor
    deter: Tensor

    loss: Tensor | None = None
    recon_loss: Tensor | None = None
    reward_loss: Tensor | None = None
    kl_loss: Tensor | None = None
    pcont_loss: Tensor | None = None
    behavior: DreamerV2BehaviorOutput | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal building blocks
# ─────────────────────────────────────────────────────────────────────────────


def balanced_kl(
    posterior: RSSMState, prior: RSSMState, *, balance: float, free_nats: float = 0.0
) -> Tensor:
    r"""KL balancing — the two directions weighted apart.

    Parameters
    ----------
    posterior, prior : RSSMState
        Both categorical, from the same step.
    balance : float
        :math:`\alpha`.  The share that trains the *prior* toward the
        posterior; ``1 - alpha`` trains the posterior toward the prior.
    free_nats : float, default=0.0
        Clamp applied to each side before mixing.

    Returns
    -------
    Tensor
        A scalar.

    Raises
    ------
    ValueError
        If either state is Gaussian — balancing needs both sides
        differentiable *and* detachable, and this family is categorical.

    Notes
    -----
    A single KL term moves prior and posterior toward each other at the
    same rate, and the paper reports the posterior giving way: it
    collapses onto a prior that has not learned anything yet, and the
    latent stops carrying the observation.  Splitting the term fixes the
    ratio instead of leaving it to whichever side has the steeper
    gradient:

    .. math::

        \alpha\, \mathrm{KL}\big(\mathrm{sg}(q) \,\|\, p\big)
        + (1 - \alpha)\, \mathrm{KL}\big(q \,\|\, \mathrm{sg}(p)\big).

    Both halves are the same number when nothing is detached — the split
    is entirely about where the gradient goes, which is why a test that
    only checks the value cannot tell this apart from a plain KL.

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
    >>> from lucid.models.generative.dreamer_v2._model import balanced_kl
    >>> balanced_kl(state, state, balance=0.8, free_nats=1.0).shape
    ()

    A scalar, and against itself it is the free-nats floor rather than
    zero — the clamp is what stops the KL collapsing the stochastic
    state to the prior early in training.

    ``balance`` splits the term in two: the posterior is pulled toward a
    stopped-gradient prior and the prior toward a stopped-gradient
    posterior, with 0.8 of the weight on the second. Without that split
    the representation gives way and the dynamics never have to learn.
    """
    if not (posterior.is_discrete and prior.is_discrete):
        raise ValueError("KL balancing here expects two categorical states")
    if not 0.0 <= balance <= 1.0:
        raise ValueError(f"balance must be in [0, 1], got {balance}")

    posterior_logits = cast(Tensor, posterior.logits)
    prior_logits = cast(Tensor, prior.logits)

    def clamp(value: Tensor) -> Tensor:
        mean = value.mean()
        return (mean - free_nats).clip(0.0, None) if free_nats > 0.0 else mean

    toward_prior = clamp(categorical_kl(posterior_logits.detach(), prior_logits))
    toward_posterior = clamp(categorical_kl(posterior_logits, prior_logits.detach()))
    return balance * toward_prior + (1.0 - balance) * toward_posterior


# ─────────────────────────────────────────────────────────────────────────────
# Direct model
# ─────────────────────────────────────────────────────────────────────────────


class DreamerV2Model(PretrainedModel):
    r"""Dreamer's architecture with a categorical latent and a target critic.

    Parameters
    ----------
    config : DreamerV2Config
        Frozen configuration.

    Notes
    -----
    Reference: Hafner, Lillicrap, Norouzi, and Ba, *"Mastering Atari with
    Discrete World Models"*, ICLR, 2021 (arXiv:2010.02193).

    The critic is duplicated.  ``value_head`` is what learns; the frozen
    ``target_value_head`` is what the returns are computed from, and it is
    refreshed on a schedule.  Without it the critic regresses onto targets
    built from itself, and a small error compounds every step it is fed
    back through.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import dreamer_v2
    >>> model = dreamer_v2(action_dim=2, cnn_depth=2, stoch_size=3, discrete=4,
    ...     deter_size=8, hidden_size=8, actor_hidden=8, value_hidden=8,
    ...     reward_hidden=8)
    >>> _, posteriors = model.observe(lucid.randn((1, 3, 3, 64, 64)),
    ...                               lucid.randn((1, 3, 2)))
    >>> posteriors.is_discrete, model.act(posteriors, sample=False).shape
    (True, (1, 3, 2))
    """

    config_class: ClassVar[type[DreamerV2Config]] = DreamerV2Config
    base_model_prefix: ClassVar[str] = "dreamer_v2"

    def __init__(self, config: DreamerV2Config) -> None:
        super().__init__(config)
        self.encoder = PixelEncoder(config.in_channels, config.cnn_depth, config.act_fn)
        self.rssm = RSSM(
            stoch_size=config.stoch_size,
            deter_size=config.deter_size,
            hidden_size=config.hidden_size,
            action_dim=config.action_dim,
            embed_size=config.embed_size,
            act_fn=config.act_fn,
            min_std=config.min_std,
            discrete=config.discrete,
        )
        self.decoder = PixelDecoder(
            config.latent_size, config.out_channels, config.cnn_depth, config.act_fn
        )
        self.reward_head = DenseHead(
            config.latent_size,
            config.reward_hidden,
            config.reward_layers,
            act_fn=config.act_fn,
            squeeze=True,
        )
        self.value_head = DenseHead(
            config.latent_size,
            config.value_hidden,
            config.value_layers,
            act_fn=config.act_fn,
            squeeze=True,
        )
        self.target_value_head = DenseHead(
            config.latent_size,
            config.value_hidden,
            config.value_layers,
            act_fn=config.act_fn,
            squeeze=True,
        )
        self.pcont_head = (
            DenseHead(
                config.latent_size,
                config.value_hidden,
                config.pcont_layers,
                act_fn=config.act_fn,
                squeeze=True,
            )
            if config.pcont
            else None
        )
        self.actor = Actor(
            config.latent_size,
            config.actor_hidden,
            config.actor_layers,
            config.action_dim,
            config.act_fn,
            config.actor_min_std,
            discrete=config.action_space == "discrete",
        )
        self._sample = not config.mean_only

    def encode(self, observations: Tensor) -> Tensor:
        """Embed a frame sequence — ``(B, T, C, 64, 64) -> (B, T, embed_size)``."""
        return self.encoder(observations)

    def observe(
        self,
        observations: Tensor,
        actions: Tensor,
        state: RSSMState | None = None,
        *,
        sample: bool | None = None,
    ) -> tuple[RSSMState, RSSMState]:
        """Filter a trajectory into posterior states.

        Parameters
        ----------
        observations : Tensor
            Frames, ``(B, T, C, 64, 64)``.
        actions : Tensor
            Actions taken *into* each step, ``(B, T, action_dim)``.
        state : RSSMState or None, optional
            Starting belief; ``None`` starts from a uniform prior.
        sample : bool or None, optional, keyword-only
            Draw the latent, take its mode, or follow ``mean_only``.

        Returns
        -------
        priors, posteriors : RSSMState
            ``(B, T, ·)`` each, carrying categorical logits.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import dreamer_v2
        >>> model = dreamer_v2(action_dim=2, cnn_depth=2, stoch_size=3, discrete=4,
        ...     deter_size=8, hidden_size=8, actor_hidden=8, value_hidden=8,
        ...     reward_hidden=8)
        >>> obs, act = lucid.randn((1, 3, 3, 64, 64)), lucid.randn((1, 3, 2))
        >>> priors, posteriors = model.observe(obs, act)
        >>> posteriors.stoch.shape, posteriors.logits.shape, posteriors.mean
        ((1, 3, 12), (1, 3, 3, 4), None)

        The latent is ``stoch_size`` one-hot draws over ``discrete`` classes,
        flattened, so each group of four holds a single 1:

        >>> posteriors.stoch.reshape(1, 3, 3, 4).sum(dim=-1).tolist()
        [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
        """
        draw = self._sample if sample is None else sample
        return self.rssm.observe(self.encode(observations), actions, state, sample=draw)

    def decode(self, state: RSSMState) -> Tensor:
        """Reconstruct frames from a state — ``(B, T, C, 64, 64)``."""
        return self.decoder(state.feature)

    def predict_reward(self, state: RSSMState) -> Tensor:
        """Predict reward from a state — ``(B, T)``."""
        return self.reward_head(state.feature)

    def predict_value(self, state: RSSMState, *, target: bool = False) -> Tensor:
        """Estimate a state's value — ``(B, T)``.

        Parameters
        ----------
        state : RSSMState
            The states to score.
        target : bool, default=False, keyword-only
            Read the frozen target copy instead of the learning critic.
            The returns are built from the target; the critic's own loss
            is measured against the learner.

        Returns
        -------
        Tensor
            Value estimates.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import dreamer_v2_world_model
        >>> wrapper = dreamer_v2_world_model(action_dim=2, cnn_depth=2,
        ...     stoch_size=3, discrete=4, deter_size=8, hidden_size=8,
        ...     actor_hidden=8, value_hidden=8, reward_hidden=8)
        >>> model = wrapper.dreamer_v2
        >>> _, posteriors = model.observe(lucid.randn((1, 3, 3, 64, 64)),
        ...                               lucid.randn((1, 3, 2)))
        >>> value = model.predict_value(posteriors)
        >>> value.shape
        (1, 3)

        The target copy starts from its own initialisation, and only
        :meth:`~DreamerV2ForWorldModeling.update_slow_target` moves it —
        the first call copies the learner outright:

        >>> target = model.predict_value(posteriors, target=True)
        >>> bool((value == target).all())
        False
        >>> wrapper.update_slow_target()
        >>> bool((value == model.predict_value(posteriors, target=True)).all())
        True
        """
        head = self.target_value_head if target else self.value_head
        return head(state.feature)

    def predict_pcont(self, state: RSSMState) -> Tensor:
        """Predict the discount at a state — logits, ``(B, T)``.

        Parameters
        ----------
        state : RSSMState
            The states to score.

        Returns
        -------
        Tensor
            Bernoulli logits; apply ``sigmoid`` for the probability.

        Raises
        ------
        ValueError
            If the model was configured without ``pcont``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import dreamer_v2
        >>> model = dreamer_v2(action_dim=2, cnn_depth=2, stoch_size=3, discrete=4,
        ...     deter_size=8, hidden_size=8, actor_hidden=8, value_hidden=8,
        ...     reward_hidden=8)
        >>> _, posteriors = model.observe(lucid.randn((1, 3, 3, 64, 64)),
        ...                               lucid.randn((1, 3, 2)))
        >>> logits = model.predict_pcont(posteriors)
        >>> logits.shape
        (1, 3)

        The head is on by default in this family; ``sigmoid`` gives the
        probability of continuing that imagination discounts by:

        >>> model.config.pcont
        True
        >>> keep = lucid.sigmoid(logits)
        >>> bool(((keep > 0) & (keep < 1)).all())
        True
        """
        if self.pcont_head is None:
            raise ValueError(
                "this model has no discount head; construct it with " "DreamerV2Config(pcont=True)"
            )
        return self.pcont_head(state.feature)

    def act(self, state: RSSMState, *, sample: bool = True) -> Tensor:
        """Propose actions — ``(B, T, action_dim)`` or ``(B, action_dim)``.

        Parameters
        ----------
        state : RSSMState
            A sequence or a single step; the rank is preserved.
        sample : bool, default=True, keyword-only
            Draw, or take the policy's mode.

        Returns
        -------
        Tensor
            Actions inside ``(-1, 1)``, or one-hot when discrete.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import dreamer_v2
        >>> kwargs = dict(action_dim=2, cnn_depth=2, stoch_size=3, discrete=4,
        ...               deter_size=8, hidden_size=8, actor_hidden=8,
        ...               value_hidden=8, reward_hidden=8)
        >>> model = dreamer_v2(**kwargs)
        >>> obs, act = lucid.randn((1, 3, 3, 64, 64)), lucid.randn((1, 3, 2))
        >>> _, posteriors = model.observe(obs, act)
        >>> last = posteriors.map(lambda t: t[:, -1])
        >>> model.act(posteriors).shape, model.act(last).shape
        ((1, 3, 2), (1, 2))
        >>> bool((model.act(last).abs() < 1).all())
        True

        A discrete action space picks one alternative instead, as a one-hot:

        >>> game = dreamer_v2(action_space="discrete", **kwargs)
        >>> _, beliefs = game.observe(obs, act)
        >>> choice = game.act(beliefs.map(lambda t: t[:, -1]))
        >>> choice.shape, choice.sum(dim=-1).tolist()
        ((1, 2), [1.0])
        """
        return self.actor(state.feature, sample=sample)

    def imagine(
        self, state: RSSMState, horizon: int, *, sample: bool | None = None
    ) -> tuple[RSSMState, Tensor]:
        """Roll the dynamics forward under the actor's own policy.

        Parameters
        ----------
        state : RSSMState
            Flat starting beliefs, ``(N, ·)``.
        horizon : int
            Steps to imagine.
        sample : bool or None, optional, keyword-only
            Draw both latent and action, take their modes, or follow
            ``mean_only``.

        Returns
        -------
        states : RSSMState
            The imagined states including the start, ``(N, horizon + 1, ·)``.
        actions : Tensor
            What the actor proposed, ``(N, horizon, action_dim)``.

        Notes
        -----
        The state the actor reads is detached, as the released
        implementation does — the gradient still reaches the policy
        through each action, and still travels back through the dynamics
        from the return.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import dreamer_v2
        >>> model = dreamer_v2(action_dim=2, cnn_depth=2, stoch_size=3, discrete=4,
        ...     deter_size=8, hidden_size=8, actor_hidden=8, value_hidden=8,
        ...     reward_hidden=8)
        >>> _, posteriors = model.observe(lucid.randn((1, 3, 3, 64, 64)),
        ...                               lucid.randn((1, 3, 2)))

        Every filtered step becomes an imagination start:

        >>> start = posteriors.map(lambda t: t.reshape(3, *t.shape[2:]))
        >>> states, actions = model.imagine(start, horizon=5)
        >>> states.stoch.shape, states.logits.shape, actions.shape
        ((3, 6, 12), (3, 6, 3, 4), (3, 5, 2))

        The start is kept as the first state, which is why there is one more
        state than action:

        >>> bool((states.stoch[:, 0] == start.stoch).all())
        True
        """
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")

        draw = self._sample if sample is None else sample
        current = state
        deters, stochs, logits = [state.deter], [state.stoch], []
        actions: list[Tensor] = []
        for _ in range(horizon):
            feature = current.feature.reshape(int(current.deter.shape[0]), 1, -1)
            action = self.actor(feature.detach(), sample=draw)[:, 0]
            current = self.rssm.prior_step(current, action, sample=draw)
            actions.append(action)
            deters.append(current.deter)
            stochs.append(current.stoch)
            logits.append(cast(Tensor, current.logits))

        # The start came from a posterior and has no prior logits of its
        # own; the first imagined step's stand in so every field is the
        # same length.  Nothing reads index 0 — the divergence is a
        # world-model term and does not run inside imagination.
        rolled = RSSMState(
            deter=lucid.stack(deters, dim=1),
            stoch=lucid.stack(stochs, dim=1),
            logits=lucid.stack([logits[0]] + logits, dim=1),
        )
        return rolled, lucid.stack(actions, dim=1)

    @override
    def forward(self, observations: Tensor, actions: Tensor) -> DreamerV2Output:
        priors, posteriors = self.observe(observations, actions)
        return DreamerV2Output(
            observation=self.decode(posteriors),
            reward=self.predict_reward(posteriors),
            value=self.predict_value(posteriors),
            posterior_stoch=posteriors.stoch,
            posterior_logits=cast(Tensor, posteriors.logits),
            prior_logits=cast(Tensor, priors.logits),
            deter=posteriors.deter,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper — the three objectives
# ─────────────────────────────────────────────────────────────────────────────


class DreamerV2ForWorldModeling(WorldModelingModel):
    r"""DreamerV2 with its world-model, actor and critic objectives.

    Parameters
    ----------
    config : DreamerV2Config
        Frozen configuration.

    Notes
    -----
    Reference: Hafner, Lillicrap, Norouzi, and Ba, *"Mastering Atari with
    Discrete World Models"*, ICLR, 2021 (arXiv:2010.02193).

    Same three-optimiser contract as Dreamer, and the same reason for it.
    Use :meth:`backward` — spending the losses by hand either contaminates
    the world model or raises, depending on the order.

    ``update_slow_target`` must be called once per gradient step.  It is
    not folded into :meth:`backward` because it counts *optimiser* steps,
    not backward passes, and only the caller knows when one has happened.

    The indexing follows the released implementation, which is worth
    stating because it is not obvious.  Imagination produces ``H + 1``
    states.  Two are lost at the end — one is the bootstrap, and one has
    an action that leads nowhere — and one target is lost at the start,
    because the first state came from the replay buffer rather than from
    the policy.  So the actor is scored on states ``0 .. H-2`` against
    targets ``1 .. H-1``, and the critic on states ``0 .. H-1``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import dreamer_v2_world_model
    >>> model = dreamer_v2_world_model(action_dim=2, cnn_depth=2, stoch_size=3,
    ...     discrete=4, deter_size=8, hidden_size=8, actor_hidden=8,
    ...     value_hidden=8, reward_hidden=8, horizon=4, pcont=False)
    >>> out = model(lucid.randn((1, 3, 3, 64, 64)), lucid.randn((1, 3, 2)),
    ...             lucid.randn((1, 3)))
    >>> bool(out.loss.ndim == 0), bool(out.behavior.actor_loss.ndim == 0)
    (True, True)
    """

    config_class: ClassVar[type[DreamerV2Config]] = DreamerV2Config
    base_model_prefix: ClassVar[str] = "dreamer_v2"

    def __init__(self, config: DreamerV2Config) -> None:
        super().__init__(config)
        self.dreamer_v2 = DreamerV2Model(config)
        self._kl_weight = config.kl_weight
        self._kl_balance = config.kl_balance
        self._free_nats = config.free_nats
        self._horizon = config.horizon
        self._discount = config.discount
        self._lambda = config.lambda_
        self._actor_grad = config.resolved_actor_grad
        self._actor_grad_mix = config.actor_grad_mix
        self._actor_entropy = config.actor_entropy
        self._pcont_scale = config.pcont_scale
        self._slow_update = config.slow_target_update
        self._slow_fraction = config.slow_target_fraction
        self._updates = 0

    # ── parameter groups ─────────────────────────────────────────────────

    def world_parameters(self) -> list[nn.Parameter]:
        """Everything the world-model loss trains.

        Returns
        -------
        list of Parameter
            Encoder, RSSM, decoder, reward head, and the discount head
            when there is one.
        """
        model = self.dreamer_v2
        modules: list[nn.Module] = [
            model.encoder,
            model.rssm,
            model.decoder,
            model.reward_head,
        ]
        if model.pcont_head is not None:
            modules.append(model.pcont_head)
        return [p for module in modules for p in module.parameters()]

    def actor_parameters(self) -> list[nn.Parameter]:
        """The actor's parameters.

        Returns
        -------
        list of Parameter
            Trained by ``actor_loss`` alone.
        """
        return list(self.dreamer_v2.actor.parameters())

    def value_parameters(self) -> list[nn.Parameter]:
        """The learning critic's parameters — *not* the target copy.

        Returns
        -------
        list of Parameter
            Trained by ``value_loss``.  The target copy is written by
            :meth:`update_slow_target` and never by an optimiser, which is
            what makes it a fixed point to regress onto.
        """
        return list(self.dreamer_v2.value_head.parameters())

    def update_slow_target(self) -> None:
        """Refresh the target critic, on the paper's schedule.

        Notes
        -----
        Called once per gradient step; it moves the target only every
        ``slow_target_update`` of them.  The very first call copies
        outright whatever ``slow_target_fraction`` says, since a target
        left at its initialisation is not a useful thing to regress onto.
        """
        if self._updates % self._slow_update == 0:
            mix = 1.0 if self._updates == 0 else self._slow_fraction
            update_averages(
                self.dreamer_v2.value_head,
                [(self.dreamer_v2.target_value_head, 1.0 - mix)],
            )
        self._updates += 1

    # ── objectives ───────────────────────────────────────────────────────

    def _behavior(self, posteriors: RSSMState) -> DreamerV2BehaviorOutput:
        """Imagine under the policy and score it.

        Parameters
        ----------
        posteriors : RSSMState
            Filtered beliefs, flattened and detached into imagination
            starts.

        Returns
        -------
        DreamerV2BehaviorOutput
            Actor and critic terms.
        """
        model = self.dreamer_v2
        if model.pcont_head is not None:
            keep = int(posteriors.deter.shape[1]) - 1
            if keep < 1:
                raise ValueError(
                    "pcont drops the last filtered step, so it needs a "
                    f"sequence of at least 2, got {int(posteriors.deter.shape[1])}"
                )
            posteriors = posteriors.map(lambda x: x[:, :keep])

        b, t = int(posteriors.deter.shape[0]), int(posteriors.deter.shape[1])
        start = posteriors.map(lambda x: x.reshape(b * t, *(int(v) for v in x.shape[2:])).detach())

        states, actions = model.imagine(start, self._horizon)
        reward = model.predict_reward(states)
        value = model.predict_value(states, target=True)

        pcont: Tensor | None = None
        if model.pcont_head is not None:
            pcont = F.sigmoid(model.predict_pcont(states))
            discount: float | Tensor = pcont
        else:
            discount = self._discount

        target = lambda_return(reward, value, discount, self._lambda)

        # Cumulative discount, detached: it weights the objectives, it is
        # not part of them.
        if pcont is None:
            weight = lucid.tensor(
                [[self._discount**i for i in range(self._horizon)]],
                device=target.device,
                dtype=target.dtype,
            )
        else:
            ones = lucid.ones((int(pcont.shape[0]), 1), device=pcont.device, dtype=pcont.dtype)
            running = lucid.cat([ones, pcont[:, : self._horizon - 1]], dim=1)
            weight = lucid.cumprod(running, dim=1).detach()

        actor_loss, entropy = self._actor_objective(states, actions, target, weight)
        value_loss = self._critic_objective(states, target, weight)

        return DreamerV2BehaviorOutput(
            actor_loss=actor_loss,
            value_loss=value_loss,
            lambda_return=target,
            entropy=entropy,
            imagined_reward=reward,
            imagined_value=value,
            imagined_action=actions,
            imagined_discount=pcont,
        )

    def _actor_objective(
        self, states: RSSMState, actions: Tensor, target: Tensor, weight: Tensor
    ) -> tuple[Tensor, Tensor]:
        """The policy's objective, by whichever gradient estimator is set.

        Parameters
        ----------
        states : RSSMState
            Imagined states, ``(N, H + 1, ·)``.
        actions : Tensor
            What the policy proposed, ``(N, H, action_dim)``.
        target : Tensor
            Lambda-returns, ``(N, H)``.
        weight : Tensor
            Cumulative discount, ``(N, H)`` or broadcastable.

        Returns
        -------
        actor_loss, entropy : Tensor
            The loss to minimise, and the mean entropy it was paid on.
        """
        model = self.dreamer_v2
        horizon = self._horizon
        scored = states.map(lambda x: x[:, : horizon - 1])
        feature = scored.feature.detach()
        objective = target[:, 1:]

        if self._actor_grad != "dynamics":
            baseline = model.predict_value(scored, target=True)
            advantage = (objective - baseline).detach()
            # `actions[t]` is the action taken *from* `states[t]`, so the
            # states scored above pair with the actions at the same index.
            # The released implementation writes `action[1:-1]` because its
            # action array carries a leading placeholder for the step that
            # led into the start state; copying that index without the
            # placeholder credits the policy with the next action's
            # log-probability, which silently disables REINFORCE.
            chosen = actions[:, : horizon - 1].detach()
            log_prob = model.actor.log_prob(feature, chosen)
            score = log_prob * advantage
            objective = (
                score
                if self._actor_grad == "reinforce"
                else self._actor_grad_mix * objective + (1.0 - self._actor_grad_mix) * score
            )

        entropy = model.actor.entropy(feature)
        objective = objective + self._actor_entropy * entropy
        actor_loss = -(weight[:, : horizon - 1] * objective).mean()
        return actor_loss, entropy.mean()

    def _critic_objective(self, states: RSSMState, target: Tensor, weight: Tensor) -> Tensor:
        """Half the weighted squared error against a fixed target.

        Parameters
        ----------
        states : RSSMState
            Imagined states, ``(N, H + 1, ·)``.
        target : Tensor
            Lambda-returns, ``(N, H)``.
        weight : Tensor
            Cumulative discount.

        Returns
        -------
        Tensor
            A scalar.

        Notes
        -----
        Reads detached states, so the critic's gradient reaches nothing
        but the critic.  The released implementation gets the same
        separation from its tape; doing it with a detach means it survives
        a caller who reaches for one optimiser anyway.
        """
        detached = states.map(lambda x: x[:, : self._horizon].detach())
        predicted = self.dreamer_v2.predict_value(detached)
        return (0.5 * weight * (predicted - target.detach()) ** 2).mean()

    def backward(self, output: DreamerV2Output) -> None:
        """Give every parameter group the gradient of its own loss.

        Parameters
        ----------
        output : DreamerV2Output
            The result of :meth:`forward`, with ``behavior`` populated.

        Raises
        ------
        ValueError
            If ``output`` carries no losses.

        Notes
        -----
        Identical in shape to Dreamer's, and identical in motivation: the
        three losses share one graph, so backpropagating them by hand
        either lets the actor's gradient descend the world model or raises
        on a parameter the imagination's graph still needed.

        Examples
        --------
        >>> import lucid
        >>> import lucid.optim as optim
        >>> from lucid.models import dreamer_v2_world_model
        >>> model = dreamer_v2_world_model(action_dim=2, cnn_depth=2,
        ...     stoch_size=3, discrete=4, deter_size=8, hidden_size=8,
        ...     actor_hidden=8, value_hidden=8, reward_hidden=8, horizon=3,
        ...     pcont=False)
        >>> groups = (model.world_parameters(), model.actor_parameters(),
        ...           model.value_parameters())
        >>> opts = [optim.Adam(g, lr=1e-4) for g in groups]
        >>> out = model(lucid.randn((1, 3, 3, 64, 64)),
        ...             lucid.randn((1, 3, 2)), lucid.randn((1, 3)))
        >>> model.backward(out)
        >>> [all(p.grad is not None for p in g) for g in groups]
        [True, True, True]

        The target critic is no optimiser's to move — it holds no gradient,
        and follows the learner through :meth:`update_slow_target`, called
        once per step:

        >>> target = model.dreamer_v2.target_value_head.parameters()
        >>> any(p.grad is not None for p in target)
        False
        >>> for opt in opts:
        ...     opt.step()
        >>> model.update_slow_target()
        """
        behavior = output.behavior
        if output.loss is None or behavior is None:
            raise ValueError(
                "backward() needs the losses; this output came from "
                "DreamerV2Model rather than DreamerV2ForWorldModeling."
            )

        def take(params: list[nn.Parameter]) -> list[Tensor | None]:
            return [None if p.grad is None else p.grad.clone() for p in params]

        self.zero_grad()
        behavior.actor_loss.backward(retain_graph=True)
        actor_grads = take(self.actor_parameters())

        self.zero_grad()
        behavior.value_loss.backward(retain_graph=True)
        value_grads = take(self.value_parameters())

        self.zero_grad()
        output.loss.backward()

        for param, grad in zip(self.actor_parameters(), actor_grads, strict=True):
            param.grad = grad
        for param, grad in zip(self.value_parameters(), value_grads, strict=True):
            param.grad = grad

    @override
    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
        rewards: Tensor,
        discounts: Tensor | None = None,
    ) -> DreamerV2Output:
        """Train the world model and the behaviour on one batch.

        Parameters
        ----------
        observations : Tensor
            Frames, ``(B, T, C, 64, 64)``.
        actions : Tensor
            Actions taken *into* each step, ``(B, T, action_dim)``.
        rewards : Tensor
            Observed reward, ``(B, T)``.
        discounts : Tensor or None, optional
            ``(B, T)``, ``0`` where the episode terminated.  Required when
            the config asks for a discount head.

        Returns
        -------
        DreamerV2Output
            ``loss`` is the world-model loss; the behaviour losses are on
            ``.behavior`` and take their own optimisers.

        Raises
        ------
        ValueError
            If ``pcont`` is configured and ``discounts`` is omitted.
        """
        model = self.dreamer_v2
        priors, posteriors = model.observe(observations, actions)
        reconstruction = model.decode(posteriors)
        predicted_reward = model.predict_reward(posteriors)

        b = int(observations.shape[0])
        t = int(observations.shape[1])
        diff = (reconstruction - observations) ** 2
        recon_loss = 0.5 * diff.reshape(b, t, -1).sum(dim=-1).mean()
        reward_loss = 0.5 * ((predicted_reward - rewards) ** 2).mean()
        kl_loss = balanced_kl(
            posteriors, priors, balance=self._kl_balance, free_nats=self._free_nats
        )
        loss = recon_loss + reward_loss + self._kl_weight * kl_loss

        pcont_loss: Tensor | None = None
        if model.pcont_head is not None:
            if discounts is None:
                raise ValueError(
                    "pcont=True needs `discounts` — 1 where the episode "
                    "continued past a step and 0 where it ended."
                )
            target = self._discount * discounts
            pcont_loss = self._pcont_scale * F.binary_cross_entropy_with_logits(
                model.predict_pcont(posteriors), target
            )
            loss = loss + pcont_loss

        return DreamerV2Output(
            observation=reconstruction,
            reward=predicted_reward,
            value=model.predict_value(posteriors),
            posterior_stoch=posteriors.stoch,
            posterior_logits=cast(Tensor, posteriors.logits),
            prior_logits=cast(Tensor, priors.logits),
            deter=posteriors.deter,
            loss=loss,
            recon_loss=recon_loss,
            reward_loss=reward_loss,
            kl_loss=kl_loss,
            pcont_loss=pcont_loss,
            behavior=self._behavior(posteriors),
        )
