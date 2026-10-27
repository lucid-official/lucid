r"""Dreamer — Hafner et al., 2020.

PlaNet's world model with its planner replaced by two learned networks.
The world model is trained exactly as before; what is new is that an actor
and a critic are fitted on trajectories the model *imagines*, so the
gradient of a predicted return reaches the policy through the learned
dynamics rather than through sampled experience.

Layout
------
``DreamerModel`` is the world model — encoder, RSSM, decoder, reward head —
plus the actor and critic.  ``DreamerForWorldModeling`` adds the three
objectives.

Sequences are ``(B, T, ...)`` throughout, matching the rest of the zoo.
``actions[:, t]`` is the action taken *into* step ``t``.

Three losses, three optimisers
------------------------------
The world model, the actor and the critic are trained by three separate
optimisers over three disjoint parameter groups.  This is not a stylistic
choice: the actor *maximises* what the critic predicts, so descending their
sum would have the actor's gradient also drag the world model, and the
critic would chase a target it is simultaneously moving.  Use
:meth:`DreamerForWorldModeling.world_parameters`,
:meth:`~DreamerForWorldModeling.actor_parameters` and
:meth:`~DreamerForWorldModeling.value_parameters`.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import WorldModelingModel
from lucid.models._output import ModelOutput
from lucid.models.generative._common._dists import OneHotCategorical
from lucid.models.generative._common._pixel_nets import DenseHead, PixelDecoder, PixelEncoder
from lucid.models.generative._common._returns import lambda_return
from lucid.models.generative._common._rssm import RSSM, RSSMState, rssm_kl
from lucid.models.generative.dreamer._config import DreamerConfig

__all__ = ["DreamerModel", "DreamerForWorldModeling", "DreamerOutput", "DreamerBehaviorOutput"]


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DreamerBehaviorOutput(ModelOutput):
    r"""What the imagination pass produces — the actor's and critic's terms.

    Attributes
    ----------
    actor_loss : Tensor
        ``-mean(w * V_lambda)``, minimised by the actor's optimiser only.
    value_loss : Tensor
        Half the discounted squared error between the critic and the
        (detached) :math:`V_\lambda` target.
    lambda_return : Tensor
        The targets themselves, ``(N, H)`` for ``N`` imagined trajectories
        over a horizon of ``H``.
    imagined_reward : Tensor
        Reward the model predicted along the imagined trajectories,
        ``(N, H + 1)``.
    imagined_value : Tensor
        The critic's estimate along the same states, ``(N, H + 1)``.
    imagined_action : Tensor
        Actions the actor proposed, ``(N, H, action_dim)``.
    imagined_pcont : Tensor or None
        Predicted continuation probability at each imagined state,
        ``(N, H + 1)``; ``None`` when the discount is held constant.

    Notes
    -----
    ``actor_loss`` and ``value_loss`` must be given **separate optimisers
    over disjoint parameter groups** — see the module docstring.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dreamer import (
    ...     DreamerConfig, DreamerForWorldModeling)
    >>> cfg = DreamerConfig(action_dim=2, cnn_depth=2, stoch_size=4,
    ...                     deter_size=8, hidden_size=8, actor_hidden=8,
    ...                     value_hidden=8, reward_hidden=8, horizon=3)
    >>> model = DreamerForWorldModeling(cfg)
    >>> out = model(lucid.randn((1, 3, 3, 64, 64)), lucid.randn((1, 3, 2)),
    ...             lucid.randn((1, 3)))
    >>> out.behavior.lambda_return.shape
    (3, 3)
    """

    actor_loss: Tensor
    value_loss: Tensor
    lambda_return: Tensor
    imagined_reward: Tensor
    imagined_value: Tensor
    imagined_action: Tensor
    imagined_pcont: Tensor | None = None


@dataclass(slots=True)
class DreamerOutput(ModelOutput):
    r"""What :class:`DreamerModel` returns after filtering a trajectory.

    Attributes
    ----------
    observation : Tensor
        Reconstruction from the posterior states, ``(B, T, C, 64, 64)``.
    reward : Tensor
        Predicted reward at each posterior state, ``(B, T)``.
    value : Tensor
        The critic's estimate at each posterior state, ``(B, T)``.
    posterior_stoch, posterior_mean, posterior_std : Tensor
        The filtered latent and the Gaussian it came from, ``(B, T, S)``.
    prior_mean, prior_std : Tensor
        The Gaussian the dynamics predicted before seeing the frame,
        ``(B, T, S)``.
    deter : Tensor
        The deterministic path, ``(B, T, D)``.
    loss, recon_loss, reward_loss, kl_loss, pcont_loss : Tensor or None
        World-model terms, set only by
        :class:`DreamerForWorldModeling`.  ``loss`` is the **world-model**
        loss alone — the behaviour losses live on
        :class:`DreamerBehaviorOutput`, because they belong to different
        optimisers.  ``pcont_loss`` is ``None`` unless the config asks for
        a discount head.
    behavior : DreamerBehaviorOutput or None
        Actor and critic terms, set only by
        :class:`DreamerForWorldModeling`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dreamer import DreamerConfig, DreamerModel
    >>> cfg = DreamerConfig(action_dim=2, cnn_depth=2, stoch_size=4,
    ...                     deter_size=8, hidden_size=8, actor_hidden=8,
    ...                     value_hidden=8, reward_hidden=8)
    >>> model = DreamerModel(cfg)
    >>> out = model(lucid.randn((1, 3, 3, 64, 64)), lucid.randn((1, 3, 2)))
    >>> out.observation.shape, out.value.shape
    ((1, 3, 3, 64, 64), (1, 3))
    """

    observation: Tensor
    reward: Tensor
    value: Tensor
    posterior_stoch: Tensor
    posterior_mean: Tensor
    posterior_std: Tensor
    prior_mean: Tensor
    prior_std: Tensor
    deter: Tensor

    loss: Tensor | None = None
    recon_loss: Tensor | None = None
    reward_loss: Tensor | None = None
    kl_loss: Tensor | None = None
    pcont_loss: Tensor | None = None
    behavior: DreamerBehaviorOutput | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal building blocks
# ─────────────────────────────────────────────────────────────────────────────


class _Actor(nn.Module):
    r"""A ``tanh``-squashed diagonal Gaussian over actions.

    Reads a latent state and emits an action in ``(-1, 1)``.  The squash is
    what bounds the action to the Control Suite's range, and doing it with
    ``tanh`` — rather than clipping — keeps the sample differentiable, which
    is the whole point: Dreamer's actor gradient arrives *through* the
    action.

    Parameters
    ----------
    latent_size : int
        Width of the state read.
    hidden, layers : int
        Shape of the dense trunk.
    action_dim : int
        Width of the action emitted.
    act_fn : str
        Activation in the trunk.
    min_std, init_std, mean_scale : float
        See :class:`DreamerConfig`.  Ignored when ``discrete``.
    discrete : bool, default=False
        Emit a one-hot over ``action_dim`` alternatives instead of a
        bounded vector.

    Notes
    -----
    ``init_std`` is applied as an offset *inside* ``softplus``, chosen so
    that an untrained head — whose raw output is near zero — produces
    exactly ``init_std``.  Solving ``softplus(c) = init_std`` gives
    ``c = log(exp(init_std) - 1)``, which is what is added.

    The discrete branch is the paper's *Discrete control* paragraph: "the
    action model predicts the logits of a categorical distribution.  We
    use straight-through gradients for the sampling step during latent
    imagination."  Both halves matter.  The one-hot is what an Atari
    button is; the straight-through draw is what keeps this actor
    trainable the same way the continuous one is — Dreamer's gradient
    arrives *through* the action, and a hard sample would sever it.

    That is where this parts company with DreamerV2, which scores a
    discrete policy with the score function instead.  The estimator here
    is biased and the reference says so; it is also what this paper ran.
    """

    def __init__(
        self,
        latent_size: int,
        hidden: int,
        layers: int,
        action_dim: int,
        act_fn: str,
        min_std: float,
        init_std: float,
        mean_scale: float,
        discrete: bool = False,
    ) -> None:
        """Initialise the actor. See the class docstring for parameters."""
        super().__init__()
        self.action_dim = action_dim
        self.min_std = min_std
        self.mean_scale = mean_scale
        self.discrete = discrete
        # softplus(c) = init_std  =>  c = log(exp(init_std) - 1)
        self._raw_init_std = math.log(math.expm1(init_std))
        # A one-hot needs one score per alternative; a squashed Gaussian
        # needs a location and a scale per dimension.
        width = action_dim if discrete else 2 * action_dim
        self.head = DenseHead(latent_size, hidden, layers, out_features=width, act_fn=act_fn)

    def distribution(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        """Return the pre-squash ``(mean, std)`` for a state.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, latent_size)``.

        Returns
        -------
        mean : Tensor
            Pre-squash mean, ``(B, T, action_dim)``, softly bounded to
            ``(-mean_scale, mean_scale)``.
        std : Tensor
            Scale, ``(B, T, action_dim)``, floored at ``min_std``.
        """
        if self.discrete:
            raise ValueError(
                "a discrete actor has no (mean, std) — it emits categorical "
                "logits; use `logits()` or call the actor"
            )
        out = self.head(feature)
        raw_mean = out[..., : self.action_dim]
        raw_std = out[..., self.action_dim :]
        mean = self.mean_scale * lucid.tanh(raw_mean / self.mean_scale)
        std = F.softplus(raw_std + self._raw_init_std) + self.min_std
        return mean, std

    @override
    def forward(self, feature: Tensor, *, sample: bool = True) -> Tensor:
        """Propose an action for a state — ``(B, T, action_dim)`` in ``(-1, 1)``.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, latent_size)``.
        sample : bool, default=True, keyword-only
            Draw reparameterised (``True``) or take the squashed mean
            (``False``, which is how a trained policy should act).

        Returns
        -------
        Tensor
            Actions bounded to ``(-1, 1)``, or one-hot rows when discrete.
        """
        if self.discrete:
            policy = OneHotCategorical(self.logits(feature))
            return policy.rsample() if sample else policy.mode
        mean, std = self.distribution(feature)
        if not sample:
            return lucid.tanh(mean)
        noise = lucid.randn(tuple(int(s) for s in mean.shape), device=mean.device, dtype=mean.dtype)
        return lucid.tanh(mean + std * noise)

    def logits(self, feature: Tensor) -> Tensor:
        """Categorical scores over the alternatives — ``(B, T, action_dim)``.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, latent_size)``.

        Returns
        -------
        Tensor
            Unnormalised scores.

        Raises
        ------
        ValueError
            If the actor is continuous, which has no logits to give.
        """
        if not self.discrete:
            raise ValueError(
                "a continuous actor has no logits — it emits a mean and a "
                "scale; use `distribution()`"
            )
        return self.head(feature)


# ─────────────────────────────────────────────────────────────────────────────
# Direct model
# ─────────────────────────────────────────────────────────────────────────────


class DreamerModel(PretrainedModel):
    r"""PlaNet's world model plus a learned actor and critic.

    The world model is unchanged from PlaNet — the same encoder, the same
    RSSM, the same decoder and reward head.  Two heads are added: an actor
    that proposes actions from a latent state and a critic that scores one.

    Parameters
    ----------
    config : DreamerConfig
        Frozen configuration.

    Notes
    -----
    Reference: Hafner, Lillicrap, Ba, and Norouzi, *"Dream to Control:
    Learning Behaviors by Latent Imagination"*, ICLR, 2020.

    ``actions[:, t]`` is the action taken *into* step ``t``, matching
    :class:`~lucid.models.generative.planet.PlaNetModel`.

    This class holds no environment and never steps one.  Everything it
    does — filtering, imagining, acting — happens in latent space, which is
    what makes the whole family trainable from a replay buffer alone.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dreamer import DreamerConfig, DreamerModel
    >>> cfg = DreamerConfig(action_dim=2, cnn_depth=2, stoch_size=4,
    ...                     deter_size=8, hidden_size=8, actor_hidden=8,
    ...                     value_hidden=8, reward_hidden=8)
    >>> model = DreamerModel(cfg)
    >>> _, posteriors = model.observe(lucid.randn((1, 4, 3, 64, 64)),
    ...                               lucid.randn((1, 4, 2)))
    >>> model.act(posteriors, sample=False).shape
    (1, 4, 2)
    """

    config_class: ClassVar[type[DreamerConfig]] = DreamerConfig
    base_model_prefix: ClassVar[str] = "dreamer"

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__(config)
        self.encoder = PixelEncoder(config.in_channels, config.cnn_depth, config.cnn_act)
        self.rssm = RSSM(
            stoch_size=config.stoch_size,
            deter_size=config.deter_size,
            hidden_size=config.hidden_size,
            action_dim=config.action_dim,
            embed_size=config.embed_size,
            act_fn=config.act_fn,
            min_std=config.min_std,
        )
        self.decoder = PixelDecoder(
            config.latent_size, config.out_channels, config.cnn_depth, config.cnn_act
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
        self.actor = _Actor(
            config.latent_size,
            config.actor_hidden,
            config.actor_layers,
            config.action_dim,
            config.act_fn,
            config.actor_min_std,
            config.actor_init_std,
            config.actor_mean_scale,
            discrete=config.action_space == "discrete",
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
        self._sample = not config.mean_only
        self._detach_actor_input = config.detach_actor_input

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
            Starting belief; ``None`` starts from zeros.
        sample : bool or None, optional, keyword-only
            Draw the latent (``True``) or take its mean (``False``).
            ``None`` follows the config's ``mean_only`` setting.

        Returns
        -------
        priors : RSSMState
            What the dynamics predicted, ``(B, T, ·)``.
        posteriors : RSSMState
            What they believed after seeing each frame, ``(B, T, ·)``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.dreamer import DreamerConfig, DreamerModel
        >>> cfg = DreamerConfig(action_dim=2, cnn_depth=2, stoch_size=4,
        ...                     deter_size=8, hidden_size=8, actor_hidden=8,
        ...                     value_hidden=8, reward_hidden=8)
        >>> model = DreamerModel(cfg)
        >>> obs, act = lucid.randn((1, 3, 3, 64, 64)), lucid.randn((1, 3, 2))
        >>> priors, posteriors = model.observe(obs, act)
        >>> posteriors.deter.shape, posteriors.stoch.shape, posteriors.std.shape
        ((1, 3, 8), (1, 3, 4), (1, 3, 4))

        Prior and posterior share one deterministic path — the frame refines
        the belief about the latent, not the history that led to it:

        >>> bool((priors.deter == posteriors.deter).all())
        True

        ``sample=False`` takes each latent's mean, so filtering repeats
        exactly:

        >>> first = model.observe(obs, act, sample=False)[1].stoch
        >>> bool((first == model.observe(obs, act, sample=False)[1].stoch).all())
        True
        """
        draw = self._sample if sample is None else sample
        return self.rssm.observe(self.encode(observations), actions, state, sample=draw)

    def decode(self, state: RSSMState) -> Tensor:
        """Reconstruct frames from a state — ``(B, T, C, 64, 64)``."""
        return self.decoder(state.feature)

    def predict_reward(self, state: RSSMState) -> Tensor:
        """Predict reward from a state — ``(B, T)``."""
        return self.reward_head(state.feature)

    def predict_value(self, state: RSSMState) -> Tensor:
        """Estimate the value of a state — ``(B, T)``."""
        return self.value_head(state.feature)

    def predict_pcont(self, state: RSSMState) -> Tensor:
        r"""Predict the discount at a state — logits, ``(B, T)``.

        The head is Bernoulli: its probability is how likely the episode is
        to continue past this state, and the discount used downstream is
        that probability rather than a constant.  A state the agent will
        not survive therefore discounts everything after it to nothing,
        which is the point — a constant :math:`\gamma` would have the
        planner keep collecting rewards past the end of the episode.

        Parameters
        ----------
        state : RSSMState
            The states to score.

        Returns
        -------
        Tensor
            Logits, ``(B, T)``.  Apply ``sigmoid`` for the probability;
            the loss consumes the logits directly.

        Raises
        ------
        ValueError
            If the model was configured without ``pcont``.

        Examples
        --------
        >>> import dataclasses
        >>> import lucid
        >>> from lucid.models.generative.dreamer import DreamerConfig, DreamerModel
        >>> cfg = DreamerConfig(action_dim=2, cnn_depth=2, stoch_size=4,
        ...                     deter_size=8, hidden_size=8, actor_hidden=8,
        ...                     value_hidden=8, reward_hidden=8, pcont=True)
        >>> model = DreamerModel(cfg)
        >>> _, posteriors = model.observe(lucid.randn((1, 3, 3, 64, 64)),
        ...                               lucid.randn((1, 3, 2)))
        >>> logits = model.predict_pcont(posteriors)
        >>> logits.shape
        (1, 3)

        ``sigmoid`` turns the logits into the probability of continuing,
        which is what imagination then discounts by:

        >>> keep = lucid.sigmoid(logits)
        >>> bool(((keep > 0) & (keep < 1)).all())
        True

        The head exists only when the configuration asks for it:

        >>> plain = DreamerModel(dataclasses.replace(cfg, pcont=False))
        >>> plain.predict_pcont(posteriors)
        Traceback (most recent call last):
            ...
        ValueError: this model has no discount head...
        """
        if self.pcont_head is None:
            raise ValueError(
                "this model has no discount head; construct it with " "DreamerConfig(pcont=True)"
            )
        return self.pcont_head(state.feature)

    def act(self, state: RSSMState, *, sample: bool = True) -> Tensor:
        """Propose actions for a state — in ``(-1, 1)``, or one-hot if discrete.

        Parameters
        ----------
        state : RSSMState
            The belief to act from, either a sequence ``(B, T, ·)`` or a
            single step ``(B, ·)``.
        sample : bool, default=True, keyword-only
            Draw from the policy (``True``) or take its squashed mean
            (``False``).

        Returns
        -------
        Tensor
            ``(B, T, action_dim)`` for a sequence, ``(B, action_dim)`` for
            a single step — the rank that went in.  A discrete action
            space gives one-hot rows rather than values in ``(-1, 1)``.

        Notes
        -----
        Both ranks are accepted because acting is inherently a single-step
        operation: an agent choosing its next move holds one belief, not a
        sequence of them.  Demanding a length-1 time axis at the call site
        would be an artifact of how the heads are batched, and every
        caller would strip it again immediately.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.dreamer import DreamerConfig, DreamerModel
        >>> cfg = DreamerConfig(action_dim=2, cnn_depth=2, stoch_size=4,
        ...                     deter_size=8, hidden_size=8, actor_hidden=8,
        ...                     value_hidden=8, reward_hidden=8)
        >>> model = DreamerModel(cfg)
        >>> _, posteriors = model.observe(lucid.randn((1, 3, 3, 64, 64)),
        ...                               lucid.randn((1, 3, 2)))
        >>> model.act(posteriors).shape
        (1, 3, 2)

        One belief per batch element gives one action each, squashed inside
        ``(-1, 1)``:

        >>> last = posteriors.map(lambda t: t[:, -1])
        >>> action = model.act(last)
        >>> action.shape, bool((action.abs() < 1).all())
        ((1, 2), True)

        ``sample=False`` is the squashed mean — the same action every time:

        >>> mode = model.act(last, sample=False)
        >>> bool((mode == model.act(last, sample=False)).all())
        True
        """
        feature = state.feature
        stepwise = feature.ndim == 2
        if stepwise:
            feature = feature.reshape(int(feature.shape[0]), 1, -1)
        action = self.actor(feature, sample=sample)
        return action[:, 0] if stepwise else action

    def imagine(
        self, state: RSSMState, horizon: int, *, sample: bool | None = None
    ) -> tuple[RSSMState, Tensor]:
        r"""Roll the dynamics forward under the actor's own policy.

        This is the loop the whole method rests on.  Unlike PlaNet's
        :meth:`~lucid.models.generative.planet.PlaNetModel.imagine`, which
        is handed a fixed action sequence to evaluate, here the action at
        each step is *produced* by the actor from the state the model just
        imagined — so the trajectory and the policy are coupled, and a
        gradient taken at the end reaches the policy at every step along
        the way.

        Parameters
        ----------
        state : RSSMState
            Flat starting beliefs, ``(N, ·)``.
        horizon : int
            Number of steps to imagine.
        sample : bool or None, optional, keyword-only
            Draw both the latent and the action (``True``) or take their
            means (``False``).  ``None`` follows the config's ``mean_only``
            setting, matching :meth:`observe` — a model configured
            deterministic must imagine deterministically too.

        Returns
        -------
        states : RSSMState
            The imagined states including the start, ``(N, horizon + 1, ·)``.
        actions : Tensor
            What the actor proposed, ``(N, horizon, action_dim)``.

        Notes
        -----
        The start state is *not* detached here — that is the caller's
        decision, and :class:`DreamerForWorldModeling` does detach it so
        the actor's gradient cannot reach the world model.

        The state the *actor reads* is detached when the config says so,
        which is the released implementation's behaviour and the default.
        It does not stop the actor learning: the gradient still arrives
        through each action it produced.  What it drops are the terms in
        which a return depends on the policy through the state it read.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.dreamer import DreamerConfig, DreamerModel
        >>> cfg = DreamerConfig(action_dim=2, cnn_depth=2, stoch_size=4,
        ...                     deter_size=8, hidden_size=8, actor_hidden=8,
        ...                     value_hidden=8, reward_hidden=8)
        >>> model = DreamerModel(cfg)
        >>> _, posteriors = model.observe(lucid.randn((1, 3, 3, 64, 64)),
        ...                               lucid.randn((1, 3, 2)))

        Every filtered step becomes an independent start, as the training
        objective uses them:

        >>> start = posteriors.map(lambda t: t.reshape(3, -1))
        >>> states, actions = model.imagine(start, horizon=5)
        >>> states.deter.shape, actions.shape
        ((3, 6, 8), (3, 5, 2))

        The start is kept as the first imagined state — one more state than
        action — and the heads score every one of them:

        >>> bool((states.deter[:, 0] == start.deter).all())
        True
        >>> model.predict_reward(states).shape
        (3, 6)
        """
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")

        draw = self._sample if sample is None else sample
        current = state
        deters, stochs, means, stds = [state.deter], [state.stoch], [], []
        actions: list[Tensor] = []
        for _ in range(horizon):
            feature = current.feature.reshape(int(current.deter.shape[0]), 1, -1)
            if self._detach_actor_input:
                feature = feature.detach()
            action = self.actor(feature, sample=draw)[:, 0]
            current = self.rssm.prior_step(current, action, sample=draw)
            actions.append(action)
            deters.append(current.deter)
            stochs.append(current.stoch)
            step_mean, step_std = current.gaussian()
            means.append(step_mean)
            stds.append(step_std)

        # The start state has no prior Gaussian of its own — it came from a
        # posterior — so its mean/std slots are filled with the first
        # imagined step's, keeping every field the same length.  Nothing
        # reads index 0 of mean/std; the KL is a world-model term and does
        # not run inside imagination.
        rolled = RSSMState(
            deter=lucid.stack(deters, dim=1),
            stoch=lucid.stack(stochs, dim=1),
            mean=lucid.stack([means[0]] + means, dim=1),
            std=lucid.stack([stds[0]] + stds, dim=1),
        )
        return rolled, lucid.stack(actions, dim=1)

    @override
    def forward(self, observations: Tensor, actions: Tensor) -> DreamerOutput:
        priors, posteriors = self.observe(observations, actions)
        posterior_mean, posterior_std = posteriors.gaussian()
        prior_mean, prior_std = priors.gaussian()
        return DreamerOutput(
            observation=self.decode(posteriors),
            reward=self.predict_reward(posteriors),
            value=self.predict_value(posteriors),
            posterior_stoch=posteriors.stoch,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            prior_mean=prior_mean,
            prior_std=prior_std,
            deter=posteriors.deter,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper — the three objectives
# ─────────────────────────────────────────────────────────────────────────────


class DreamerForWorldModeling(WorldModelingModel):
    r"""Dreamer with its world-model, actor and critic objectives.

    Three losses come out of one forward pass:

    - the **world-model** loss, identical in form to PlaNet's — pixel and
      reward reconstruction against a free-nats-clamped KL;
    - the **actor** loss, :math:`-\mathbb{E}[V_\lambda]` over trajectories
      imagined under the policy itself;
    - the **critic** loss, regressing onto those same targets held fixed.

    Parameters
    ----------
    config : DreamerConfig
        Frozen configuration.

    Notes
    -----
    Reference: Hafner, Lillicrap, Ba, and Norouzi, *"Dream to Control:
    Learning Behaviors by Latent Imagination"*, ICLR, 2020.

    **These three must be optimised separately.**  Summing them and taking
    one step is a different algorithm: the actor's gradient would descend
    the world model too, and the critic would chase a moving target.  The
    parameter groups are exposed as :meth:`world_parameters`,
    :meth:`actor_parameters` and :meth:`value_parameters`, which partition
    the model exactly.

    Use :meth:`backward` to fill those groups' gradients.  The losses share
    a graph, so backpropagating them by hand either contaminates the world
    model or raises, depending on the order chosen — :meth:`backward` is
    the whole training step bar the ``step`` calls.

    Imagination starts from **detached** posterior states, so no behaviour
    gradient reaches the encoder or the filtering that produced them.
    Within the horizon, though, the actor's gradient deliberately flows
    *through the learned dynamics* — through the RSSM transition, the
    reward head and the critic — because that analytic path is the paper's
    entire contribution.  Both the latent draw and the action draw are
    reparameterised, which is what lets a return computed 15 steps out
    reach the policy that chose step 1.  Only the actor's parameters are
    updated by it; that is what the parameter grouping is for.

    The critic is the exception: it reads detached states, so ``value_loss``
    reaches nothing but the critic itself.

    Reconstruction is a Gaussian log-likelihood with unit variance, so it
    reduces to a squared error summed over pixels and averaged over the
    batch — the same convention as PlaNet.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dreamer import (
    ...     DreamerConfig, DreamerForWorldModeling)
    >>> cfg = DreamerConfig(action_dim=2, cnn_depth=2, stoch_size=4,
    ...                     deter_size=8, hidden_size=8, actor_hidden=8,
    ...                     value_hidden=8, reward_hidden=8, horizon=3)
    >>> model = DreamerForWorldModeling(cfg)
    >>> out = model(lucid.randn((1, 3, 3, 64, 64)), lucid.randn((1, 3, 2)),
    ...             lucid.randn((1, 3)))
    >>> bool(out.loss.ndim == 0), bool(out.behavior.actor_loss.ndim == 0)
    (True, True)
    """

    config_class: ClassVar[type[DreamerConfig]] = DreamerConfig
    base_model_prefix: ClassVar[str] = "dreamer"

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__(config)
        self.dreamer = DreamerModel(config)
        self._free_nats = config.free_nats
        self._kl_weight = config.kl_weight
        self._horizon = config.horizon
        self._discount = config.discount
        self._lambda = config.lambda_
        self._pcont_scale = config.pcont_scale

    def world_parameters(self) -> list[nn.Parameter]:
        """Everything the world-model loss trains — encoder, RSSM, decoder, reward.

        Returns
        -------
        list of Parameter
            The world model's parameters, disjoint from
            :meth:`actor_parameters` and :meth:`value_parameters`.
        """
        model = self.dreamer
        modules: list[nn.Module] = [model.encoder, model.rssm, model.decoder, model.reward_head]
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
        return list(self.dreamer.actor.parameters())

    def value_parameters(self) -> list[nn.Parameter]:
        """The critic's parameters.

        Returns
        -------
        list of Parameter
            Trained by ``value_loss`` alone.
        """
        return list(self.dreamer.value_head.parameters())

    def backward(self, output: DreamerOutput) -> None:
        r"""Give every parameter group the gradient of *its own* loss.

        The three losses share one graph, and that makes the obvious ways
        of using them both wrong:

        - Backward all three and then step — every world-model parameter
          ends up carrying the actor's gradient as well as its own, so the
          world model quietly descends the policy's objective.  Nothing
          errors; the algorithm is simply no longer Dreamer's.
        - Backward-then-step each in turn — the first ``step`` mutates
          parameters that the imagination's deeper graph still needs, and
          the next ``backward`` raises.  It happens to work if the actor
          goes first, which is not a property anyone should have to know.

        So this does it once, correctly: each loss is backpropagated in
        isolation, its group's gradients are kept, and the rest are
        discarded.  Afterwards every group holds exactly its own gradient
        and the three optimisers may ``step`` in any order.

        Parameters
        ----------
        output : DreamerOutput
            The result of :meth:`forward`, with ``behavior`` populated.

        Raises
        ------
        ValueError
            If ``output`` carries no losses — it came from
            :class:`DreamerModel` rather than from this wrapper.

        Examples
        --------
        >>> import lucid
        >>> import lucid.optim as optim
        >>> from lucid.models import dreamer_world_model
        >>> model = dreamer_world_model(action_dim=2, cnn_depth=2,
        ...     stoch_size=4, deter_size=8, hidden_size=8, actor_hidden=8,
        ...     value_hidden=8, reward_hidden=8, horizon=3)
        >>> opts = [optim.Adam(g, lr=1e-4) for g in (model.world_parameters(),
        ...     model.actor_parameters(), model.value_parameters())]
        >>> out = model(lucid.randn((1, 3, 3, 64, 64)),
        ...             lucid.randn((1, 3, 2)), lucid.randn((1, 3)))
        >>> model.backward(out)
        >>> for opt in opts:
        ...     opt.step()
        """
        behavior = output.behavior
        if output.loss is None or behavior is None:
            raise ValueError(
                "backward() needs the losses; this output came from "
                "DreamerModel rather than DreamerForWorldModeling."
            )

        def take(params: list[nn.Parameter]) -> list[Tensor | None]:
            return [None if p.grad is None else p.grad.clone() for p in params]

        # Deepest graph first, and retain it — the actor's reaches back
        # through every imagined step.
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

    def _behavior(self, posteriors: RSSMState) -> DreamerBehaviorOutput:
        """Imagine under the policy and score it.

        Parameters
        ----------
        posteriors : RSSMState
            Filtered beliefs, ``(B, T, ·)``.  Flattened to ``(B * T, ·)`` and
            detached, so every filtered step becomes an independent
            imagination start.  With ``pcont`` the final step is dropped
            first — it may be the terminal one, and imagining onward from
            a state the episode already ended in trains the policy on
            something that cannot happen.

        Returns
        -------
        DreamerBehaviorOutput
            The actor and critic terms.
        """
        if self.dreamer.pcont_head is not None:
            keep = int(posteriors.deter.shape[1]) - 1
            if keep < 1:
                raise ValueError(
                    "pcont drops the last filtered step, so it needs a "
                    f"sequence of at least 2, got {int(posteriors.deter.shape[1])}"
                )
            posteriors = posteriors.map(lambda x: x[:, :keep])

        b, t = int(posteriors.deter.shape[0]), int(posteriors.deter.shape[1])

        def flat(x: Tensor) -> Tensor:
            return x.reshape(b * t, int(x.shape[2])).detach()

        start_mean, start_std = posteriors.gaussian()
        start = RSSMState(
            deter=flat(posteriors.deter),
            stoch=flat(posteriors.stoch),
            mean=flat(start_mean),
            std=flat(start_std),
        )

        states, actions = self.dreamer.imagine(start, self._horizon)
        reward = self.dreamer.predict_reward(states)
        value = self.dreamer.predict_value(states)

        # Constant gamma, or the discount the model predicts for each
        # imagined state when the episode can end.
        pcont: Tensor | None = None
        discount: float | Tensor = self._discount
        if self.dreamer.pcont_head is not None:
            pcont = lucid.sigmoid(self.dreamer.predict_pcont(states))
            discount = pcont

        returns = lambda_return(reward, value, discount, self._lambda)

        # The released implementation weights both behaviour losses by the
        # cumulative discount, so a step the agent is unlikely to still be
        # around for counts for less.  It is a weight, never a thing to
        # differentiate, so it is detached either way.
        if pcont is None:
            weight = lucid.tensor(
                [[self._discount**i for i in range(self._horizon)]],
                device=returns.device,
                dtype=returns.dtype,
            )
        else:
            ones = lucid.ones((int(pcont.shape[0]), 1), device=pcont.device, dtype=pcont.dtype)
            running = lucid.cat([ones, pcont[:, : self._horizon - 1]], dim=1)
            weight = lucid.cumprod(running, dim=1).detach()

        actor_loss = -(weight * returns).mean()

        # The critic reads *detached* states.  Its job is to regress onto
        # the target, not to shape the dynamics that produced it — and the
        # released implementation gets the same effect by giving the critic
        # its own tape over its own variables.  Doing it with a detach
        # instead means the separation survives a caller who reaches for
        # one optimiser anyway.
        detached = states.map(lambda t: t.detach())
        target = returns.detach()
        predicted = self.dreamer.predict_value(detached)[:, : self._horizon]
        value_loss = (0.5 * weight * (predicted - target) ** 2).mean()

        return DreamerBehaviorOutput(
            actor_loss=actor_loss,
            value_loss=value_loss,
            lambda_return=returns,
            imagined_reward=reward,
            imagined_value=value,
            imagined_action=actions,
            imagined_pcont=pcont,
        )

    @override
    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
        rewards: Tensor,
        discounts: Tensor | None = None,
    ) -> DreamerOutput:
        """Train the world model and the behaviour on one batch of trajectories.

        Parameters
        ----------
        observations : Tensor
            Frames, ``(B, T, C, 64, 64)``.
        actions : Tensor
            Actions taken *into* each step, ``(B, T, action_dim)``.
        rewards : Tensor
            Observed reward at each step, ``(B, T)``.
        discounts : Tensor or None, optional
            ``(B, T)``, ``1`` where the episode continued past that step
            and ``0`` where it ended.  Required when the config asks for a
            discount head, ignored otherwise.

        Returns
        -------
        DreamerOutput
            ``loss`` is the world-model loss; the behaviour losses are on
            ``.behavior`` and take their own optimisers.

        Raises
        ------
        ValueError
            If ``pcont`` is configured and ``discounts`` is omitted — the
            head has nothing to learn from without it, and defaulting to
            "never terminates" would silently train it to a constant.
        """
        model = self.dreamer
        priors, posteriors = model.observe(observations, actions)
        reconstruction = model.decode(posteriors)
        predicted_reward = model.predict_reward(posteriors)

        b = int(observations.shape[0])
        t = int(observations.shape[1])
        # Unit-variance Gaussian log-densities, exactly as PlaNet scores
        # them: one-half squared error, summed over pixels and averaged
        # over (B, T).
        diff = (reconstruction - observations) ** 2
        recon_loss = 0.5 * diff.reshape(b, t, -1).sum(dim=-1).mean()
        reward_loss = 0.5 * ((predicted_reward - rewards) ** 2).mean()
        kl_loss = rssm_kl(posteriors, priors, free_nats=self._free_nats)
        loss = recon_loss + reward_loss + self._kl_weight * kl_loss
        posterior_mean, posterior_std = posteriors.gaussian()
        prior_mean, prior_std = priors.gaussian()

        pcont_loss: Tensor | None = None
        if model.pcont_head is not None:
            if discounts is None:
                raise ValueError(
                    "pcont=True needs `discounts` — 1 where the episode "
                    "continued past a step and 0 where it ended."
                )
            # Bernoulli likelihood against a soft target: the released
            # implementation regresses onto `gamma * d`, so a surviving
            # step is taught `gamma` rather than 1 and the head's output
            # is directly usable as the discount.
            target = self._discount * discounts
            pcont_loss = self._pcont_scale * F.binary_cross_entropy_with_logits(
                model.predict_pcont(posteriors), target
            )
            loss = loss + pcont_loss

        return DreamerOutput(
            observation=reconstruction,
            reward=predicted_reward,
            value=model.predict_value(posteriors),
            posterior_stoch=posteriors.stoch,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            prior_mean=prior_mean,
            prior_std=prior_std,
            deter=posteriors.deter,
            loss=loss,
            recon_loss=recon_loss,
            reward_loss=reward_loss,
            kl_loss=kl_loss,
            pcont_loss=pcont_loss,
            behavior=self._behavior(posteriors),
        )
