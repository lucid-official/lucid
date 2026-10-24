r"""DreamerV3 — Hafner et al., Nature 640 (2025).

The same machine as DreamerV2, with every place a domain-specific
constant used to be replaced by something scale-free.  The encoder,
decoder, RSSM, actor and lambda-returns are all the shared ones; what is
written here is how the four changes fit together and what each objective
now looks like.

Sequences are ``(B, T, ...)``, and ``actions[:, t]`` is the action taken
*into* step ``t``, as in the rest of the family.

Three losses, three optimisers — the same contract the earlier families
have, with the same :meth:`DreamerV3ForWorldModeling.backward` to spend
them correctly.

Where this departs from the released implementation, and why
---------------------------------------------------------

The paper's own architecture list — block GRU, RMS normalisation, SiLU —
is implemented: the first two live in :class:`~.RSSM` behind ``blocks``,
the third is the family's default activation.  What is left out is what
the released code has added *since*: extra dynamics layers, an
``absolute`` posterior variant, adaptive gradient clipping and LaProp.
The last two are optimiser behaviour and belong to :mod:`lucid.optim` if
anywhere.

Two further departures are deliberate and worth stating because they are
not visible from the configuration:

* The **replay critic reads detached features.**  The reference lets that
  loss reach the world model (``repval_grad: True``) because it has a
  single optimiser and a single loss.  This family has three parameter
  groups by construction, and :meth:`~DreamerV3ForWorldModeling.backward`
  gives the world model only its own gradient, so a gradient sent along
  that path would be discarded anyway.  Detaching makes that explicit
  rather than accidental.
* **Value normalisation and advantage normalisation are absent.**  The
  reference carries both, disabled (``impl: none``) in every published
  configuration including the ones for the paper's results.  A knob no
  one has ever turned is not worth a field.
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
from lucid.models.generative._common._actor import Actor
from lucid.models.generative._common._pixel_nets import (
    DenseHead,
    PixelDecoder,
    PixelEncoder,
)
from lucid.models.generative._common._returns import lambda_return
from lucid.models.generative._common._rssm import RSSM, RSSMState
from lucid.models.generative.dreamer_v3._config import DreamerV3Config
from lucid.models.generative.dreamer_v3._heads import TwoHotHead
from lucid.models.generative.dreamer_v3._objectives import (
    ReturnNormaliser,
    free_bits_kl,
)

__all__ = [
    "DreamerV3Model",
    "DreamerV3ForWorldModeling",
    "DreamerV3Output",
    "DreamerV3BehaviorOutput",
]


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DreamerV3BehaviorOutput(ModelOutput):
    r"""What the imagination pass produces.

    Attributes
    ----------
    actor_loss, value_loss : Tensor
        Scalars, each for its own optimiser.  ``value_loss`` already
        carries the replayed-trajectory term when one is configured.
    lambda_return : Tensor
        The targets, ``(N, H)``, bootstrapped from the *live* critic.
    return_scale : Tensor
        What the advantage was divided by — ``max(1, S)``, 0-d.  Exposed
        because it is the number that decides whether the entropy bonus
        is currently doing anything.
    entropy : Tensor
        Mean policy entropy over the scored states.
    imagined_reward, imagined_value : Tensor
        Along the imagined trajectory, ``(N, H + 1)``.
    imagined_action : Tensor
        What the actor proposed, ``(N, H, action_dim)``.
    imagined_discount : Tensor or None
        Predicted continuation probability, ``(N, H + 1)``; ``None`` when
        the discount is held constant.
    replay_value_loss : Tensor or None
        The critic's term over the replayed trajectory, before its scale
        is applied; ``None`` when that term is switched off.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import dreamer_v3_12m_world_model
    >>> model = dreamer_v3_12m_world_model(action_dim=2, cnn_depth=2, stoch_size=3,
    ...     discrete=4, deter_size=8, hidden_size=8, actor_hidden=8,
    ...     value_hidden=8, reward_hidden=8, num_bins=5, horizon=3, pcont=False)
    >>> out = model(lucid.randn((1, 4, 3, 64, 64)), lucid.randn((1, 4, 2)),
    ...             lucid.randn((1, 4)))
    >>> out.behavior.lambda_return.shape
    (4, 3)
    """

    actor_loss: Tensor
    value_loss: Tensor
    lambda_return: Tensor
    return_scale: Tensor
    entropy: Tensor
    imagined_reward: Tensor
    imagined_value: Tensor
    imagined_action: Tensor
    imagined_discount: Tensor | None = None
    replay_value_loss: Tensor | None = None


@dataclass(slots=True)
class DreamerV3Output(ModelOutput):
    r"""What :class:`DreamerV3Model` returns after filtering a trajectory.

    Attributes
    ----------
    observation : Tensor
        Reconstruction, ``(B, T, C, 64, 64)``.
    reward, value : Tensor
        Predictions at each posterior state, ``(B, T)``, decoded from
        their bin distributions and back in the target's own units.
    posterior_stoch : Tensor
        The filtered latent, flattened, ``(B, T, stoch_size * discrete)``.
    posterior_logits, prior_logits : Tensor
        Class scores before and after seeing the frame,
        ``(B, T, stoch_size, discrete)``.
    deter : Tensor
        The deterministic path, ``(B, T, D)``.
    loss, recon_loss, reward_loss, pcont_loss : Tensor or None
        World-model terms, set only by
        :class:`DreamerV3ForWorldModeling`.
    dynamics_loss, representation_loss : Tensor or None
        The two halves of the divergence, already scaled and floored.
        Kept apart because their ratio is what a reader checks when a
        latent collapses.
    kl_loss : Tensor or None
        Their sum, for symmetry with the earlier families.
    behavior : DreamerV3BehaviorOutput or None
        Actor and critic terms, set only by that wrapper.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import dreamer_v3_12m
    >>> model = dreamer_v3_12m(action_dim=2, cnn_depth=2, stoch_size=3, discrete=4,
    ...     deter_size=8, hidden_size=8, actor_hidden=8, value_hidden=8,
    ...     reward_hidden=8, num_bins=5)
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
    dynamics_loss: Tensor | None = None
    representation_loss: Tensor | None = None
    kl_loss: Tensor | None = None
    pcont_loss: Tensor | None = None
    behavior: DreamerV3BehaviorOutput | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Direct model
# ─────────────────────────────────────────────────────────────────────────────


class DreamerV3Model(PretrainedModel):
    r"""DreamerV2's architecture with distributional heads and a slow critic.

    Parameters
    ----------
    config : DreamerV3Config
        Frozen configuration.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104).

    Two architectural differences from DreamerV2 are visible here.

    The reward and value heads are :class:`~.TwoHotHead`\ s — they predict
    a distribution over exponentially spaced bins and are read back
    through ``symexp``, which is what decouples the gradients from the
    reward's magnitude.

    The critic is duplicated, but the copy plays the opposite role to
    DreamerV2's.  There the frozen copy produced the *targets*; here the
    returns bootstrap from the live critic and the slow copy appears only
    as a regulariser pulling the live one toward its own moving average.
    The copy is therefore updated every step by a small fraction rather
    than replaced wholesale on a schedule.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import dreamer_v3_12m
    >>> model = dreamer_v3_12m(action_dim=2, cnn_depth=2, stoch_size=3, discrete=4,
    ...     deter_size=8, hidden_size=8, actor_hidden=8, value_hidden=8,
    ...     reward_hidden=8, num_bins=5)
    >>> _, posteriors = model.observe(lucid.randn((1, 3, 3, 64, 64)),
    ...                               lucid.randn((1, 3, 2)))
    >>> posteriors.is_discrete, model.act(posteriors, sample=False).shape
    (True, (1, 3, 2))
    """

    config_class: ClassVar[type[DreamerV3Config]] = DreamerV3Config
    base_model_prefix: ClassVar[str] = "dreamer_v3"

    def __init__(self, config: DreamerV3Config) -> None:
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
            unimix=config.unimix,
            blocks=config.blocks,
        )
        self.decoder = PixelDecoder(
            config.latent_size, config.out_channels, config.cnn_depth, config.act_fn
        )
        self.reward_head = TwoHotHead(
            config.latent_size,
            config.reward_hidden,
            config.reward_layers,
            num_bins=config.num_bins,
            bin_range=config.bin_range,
            act_fn=config.act_fn,
        )
        # Both critics start predicting exactly zero.  A critic that opens
        # by asserting large returns sends the actor chasing them, and the
        # slow copy would then spend its whole warm-up defending that.
        self.value_head = TwoHotHead(
            config.latent_size,
            config.value_hidden,
            config.value_layers,
            num_bins=config.num_bins,
            bin_range=config.bin_range,
            act_fn=config.act_fn,
            zero_init=True,
        )
        self.slow_value_head = TwoHotHead(
            config.latent_size,
            config.value_hidden,
            config.value_layers,
            num_bins=config.num_bins,
            bin_range=config.bin_range,
            act_fn=config.act_fn,
            zero_init=True,
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
            unimix=config.unimix,
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
            ``(B, T, ·)`` each, carrying categorical logits already mixed
            with ``unimix``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import dreamer_v3_12m
        >>> model = dreamer_v3_12m(action_dim=2, cnn_depth=2, stoch_size=3,
        ...     discrete=4, deter_size=8, hidden_size=8, actor_hidden=8,
        ...     value_hidden=8, reward_hidden=8, num_bins=5)
        >>> obs, act = lucid.randn((1, 3, 3, 64, 64)), lucid.randn((1, 3, 2))
        >>> priors, posteriors = model.observe(obs, act)
        >>> posteriors.stoch.shape, posteriors.logits.shape
        ((1, 3, 12), (1, 3, 3, 4))

        ``unimix`` is already folded in: every class keeps at least
        ``unimix / discrete`` of the mass, so no outcome is ever ruled out:

        >>> floor = model.config.unimix / model.config.discrete
        >>> bool((lucid.softmax(posteriors.logits, dim=-1) >= floor).all())
        True
        """
        draw = self._sample if sample is None else sample
        return self.rssm.observe(self.encode(observations), actions, state, sample=draw)

    def decode(self, state: RSSMState) -> Tensor:
        """Reconstruct frames from a state — ``(B, T, C, 64, 64)``.

        Notes
        -----
        Frames are *not* passed through ``symlog``.  The transform is for
        quantities whose range differs between domains; a pixel is always
        in the same interval, and the reference applies it to vector
        observations and to reward and value only.
        """
        return self.decoder(state.feature)

    def predict_reward(self, state: RSSMState) -> Tensor:
        """Predict reward from a state — ``(B, T)``, in reward units."""
        return self.reward_head.predict(state.feature)

    def predict_value(self, state: RSSMState, *, slow: bool = False) -> Tensor:
        """Estimate a state's value — ``(B, T)``.

        Parameters
        ----------
        state : RSSMState
            The states to score.
        slow : bool, default=False, keyword-only
            Read the slow copy instead of the learning critic.  Note this
            is *not* where the returns come from — they bootstrap from
            the learner; the slow copy only regularises it.

        Returns
        -------
        Tensor
            Value estimates, in return units.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import dreamer_v3_12m
        >>> model = dreamer_v3_12m(action_dim=2, cnn_depth=2, stoch_size=3,
        ...     discrete=4, deter_size=8, hidden_size=8, actor_hidden=8,
        ...     value_hidden=8, reward_hidden=8, num_bins=5)
        >>> _, posteriors = model.observe(lucid.randn((1, 3, 3, 64, 64)),
        ...                               lucid.randn((1, 3, 2)))
        >>> value = model.predict_value(posteriors)
        >>> value.shape
        (1, 3)

        Both critics are zero-initialised, so an untrained critic asserts no
        return for the actor to chase, and the slow copy agrees with it:

        >>> slow = model.predict_value(posteriors, slow=True)
        >>> float(value.abs().max().item()), float(slow.abs().max().item())
        (0.0, 0.0)
        """
        head = self.slow_value_head if slow else self.value_head
        return head.predict(state.feature)

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
        >>> from lucid.models import dreamer_v3_12m
        >>> model = dreamer_v3_12m(action_dim=2, cnn_depth=2, stoch_size=3,
        ...     discrete=4, deter_size=8, hidden_size=8, actor_hidden=8,
        ...     value_hidden=8, reward_hidden=8, num_bins=5)
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
                "this model has no discount head; construct it with "
                "DreamerV3Config(pcont=True)"
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
        >>> from lucid.models import dreamer_v3_12m
        >>> kwargs = dict(action_dim=2, cnn_depth=2, stoch_size=3, discrete=4,
        ...               deter_size=8, hidden_size=8, actor_hidden=8,
        ...               value_hidden=8, reward_hidden=8, num_bins=5)
        >>> model = dreamer_v3_12m(**kwargs)
        >>> obs, act = lucid.randn((1, 3, 3, 64, 64)), lucid.randn((1, 3, 2))
        >>> _, posteriors = model.observe(obs, act)
        >>> last = posteriors.map(lambda t: t[:, -1])
        >>> model.act(posteriors).shape, model.act(last).shape
        ((1, 3, 2), (1, 2))
        >>> bool((model.act(last).abs() < 1).all())
        True

        A discrete action space picks one alternative instead, as a one-hot:

        >>> game = dreamer_v3_12m(action_space="discrete", **kwargs)
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
        The state the actor reads is detached.  Unlike the earlier
        families this costs nothing: DreamerV3's actor is trained purely
        by the score function, so no gradient was ever going to travel
        back through the dynamics from the return.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import dreamer_v3_12m
        >>> model = dreamer_v3_12m(action_dim=2, cnn_depth=2, stoch_size=3,
        ...     discrete=4, deter_size=8, hidden_size=8, actor_hidden=8,
        ...     value_hidden=8, reward_hidden=8, num_bins=5)
        >>> _, posteriors = model.observe(lucid.randn((1, 3, 3, 64, 64)),
        ...                               lucid.randn((1, 3, 2)))

        Every filtered step becomes an imagination start:

        >>> start = posteriors.map(lambda t: t.reshape(3, *t.shape[2:]))
        >>> states, actions = model.imagine(start, horizon=5)
        >>> states.stoch.shape, states.logits.shape, actions.shape
        ((3, 6, 12), (3, 6, 3, 4), (3, 5, 2))

        Reward along the trajectory is decoded from its bins in reward
        units — one per state, the start included:

        >>> model.predict_reward(states).shape
        (3, 6)
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
    def forward(self, observations: Tensor, actions: Tensor) -> DreamerV3Output:
        priors, posteriors = self.observe(observations, actions)
        return DreamerV3Output(
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


class DreamerV3ForWorldModeling(WorldModelingModel):
    r"""DreamerV3 with its world-model, actor and critic objectives.

    Parameters
    ----------
    config : DreamerV3Config
        Frozen configuration.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104).

    Same three-optimiser contract as the earlier families, and the same
    reason for it.  Use :meth:`backward` — spending the losses by hand
    either contaminates the world model or raises, depending on the order.

    :meth:`update_slow_critic` must be called once per gradient step.  It
    is not folded into :meth:`backward` because it counts *optimiser*
    steps, not backward passes, and only the caller knows when one has
    happened.

    The indexing is simpler than DreamerV2's, and the simplification is
    the point.  Imagination produces ``H + 1`` states and ``H`` actions;
    ``action[t]`` is taken from ``state[t]``, the return ``R[t]`` starts
    at ``state[t]``, and the critic's estimate ``v[t]`` scores the same
    state.  So the advantage is ``R[t] - v(s_t)`` paired with
    ``log pi(a_t | s_t)`` at every ``t`` in ``0 .. H-1`` — the textbook
    alignment, available here because the actor is score-function only
    and does not need the one-step shift dynamics backpropagation forced
    on DreamerV2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import dreamer_v3_12m_world_model
    >>> model = dreamer_v3_12m_world_model(action_dim=2, cnn_depth=2, stoch_size=3,
    ...     discrete=4, deter_size=8, hidden_size=8, actor_hidden=8,
    ...     value_hidden=8, reward_hidden=8, num_bins=5, horizon=4, pcont=False)
    >>> out = model(lucid.randn((1, 3, 3, 64, 64)), lucid.randn((1, 3, 2)),
    ...             lucid.randn((1, 3)))
    >>> bool(out.loss.ndim == 0), bool(out.behavior.actor_loss.ndim == 0)
    (True, True)
    """

    config_class: ClassVar[type[DreamerV3Config]] = DreamerV3Config
    base_model_prefix: ClassVar[str] = "dreamer_v3"

    def __init__(self, config: DreamerV3Config) -> None:
        super().__init__(config)
        self.dreamer_v3 = DreamerV3Model(config)
        self._kl_weight = config.kl_weight
        self._dyn_scale = config.dyn_scale
        self._rep_scale = config.rep_scale
        self._pred_scale = config.pred_scale
        self._free_nats = config.free_nats
        self._horizon = config.horizon
        self._discount = config.discount
        self._lambda = config.lambda_
        self._actor_entropy = config.actor_entropy
        self._critic_ema = config.critic_ema
        self._critic_slowreg = config.critic_slowreg
        self._replay_value_scale = config.replay_value_scale
        self._pcont_scale = config.pcont_scale
        self._updates = 0
        self.returns = ReturnNormaliser(
            decay=config.return_ema_decay,
            low=config.return_low,
            high=config.return_high,
        )

    # ── parameter groups ─────────────────────────────────────────────────

    def world_parameters(self) -> list[nn.Parameter]:
        """Everything the world-model loss trains.

        Returns
        -------
        list of Parameter
            Encoder, RSSM, decoder, reward head, and the discount head
            when there is one.
        """
        model = self.dreamer_v3
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
        return list(self.dreamer_v3.actor.parameters())

    def value_parameters(self) -> list[nn.Parameter]:
        """The learning critic's parameters — *not* the slow copy.

        Returns
        -------
        list of Parameter
            Trained by ``value_loss``.  The slow copy is written by
            :meth:`update_slow_critic` and never by an optimiser, which is
            what makes it a fixed point to regress toward.
        """
        return list(self.dreamer_v3.value_head.parameters())

    def update_slow_critic(self) -> None:
        """Move the slow critic a little toward the live one.

        Notes
        -----
        Every gradient step, by ``critic_ema`` — 2% at the paper's value.
        DreamerV2 copied its target outright every hundred steps; a
        continuous average has no schedule to tune and no discontinuity
        for the critic's loss to step over.  The first call copies
        outright, since a slow critic left at its initialisation is not
        something worth regressing toward.
        """
        mix = 1.0 if self._updates == 0 else self._critic_ema
        update_averages(
            self.dreamer_v3.value_head, [(self.dreamer_v3.slow_value_head, 1.0 - mix)]
        )
        self._updates += 1

    # ── objectives ───────────────────────────────────────────────────────

    def _behavior(
        self, posteriors: RSSMState, rewards: Tensor, continues: Tensor | None
    ) -> DreamerV3BehaviorOutput:
        """Imagine under the policy and score it.

        Parameters
        ----------
        posteriors : RSSMState
            Filtered beliefs, flattened and detached into imagination
            starts.
        rewards : Tensor
            Observed reward over the replayed batch, ``(B, T)`` — needed
            by the critic's replay term.
        continues : Tensor or None
            ``(B, T)``, ``0`` where the episode terminated.

        Returns
        -------
        DreamerV3BehaviorOutput
            Actor and critic terms.
        """
        model = self.dreamer_v3
        batch = int(posteriors.deter.shape[0])
        kept = posteriors
        if model.pcont_head is not None:
            keep = int(posteriors.deter.shape[1]) - 1
            if keep < 1:
                raise ValueError(
                    "pcont drops the last filtered step, so it needs a "
                    f"sequence of at least 2, got {int(posteriors.deter.shape[1])}"
                )
            kept = posteriors.map(lambda x: x[:, :keep])

        steps = int(kept.deter.shape[1])
        start = kept.map(
            lambda x: x.reshape(batch * steps, *(int(v) for v in x.shape[2:])).detach()
        )

        states, actions = model.imagine(start, self._horizon)
        reward = model.predict_reward(states)
        value = model.predict_value(states)

        pcont: Tensor | None = None
        if model.pcont_head is not None:
            pcont = F.sigmoid(model.predict_pcont(states))
            discount: float | Tensor = pcont
        else:
            discount = self._discount

        target = lambda_return(reward, value, discount, self._lambda)

        # How much of the objective each imagined step is worth: the
        # probability the episode is still running *when it arrives*, so
        # the product runs over the steps before it and the first one is
        # worth 1.  With a learned discount that probability already
        # carries gamma, which is why the two branches look different and
        # mean the same thing — and they only do while both start at 1.
        # Including ``pcont[0]`` instead scales every objective by the
        # head's opinion of a state the trajectory has not left yet.
        if pcont is None:
            weight = lucid.tensor(
                [[self._discount**i for i in range(self._horizon)]],
                device=target.device,
                dtype=target.dtype,
            )
        else:
            ones = lucid.ones(
                (int(pcont.shape[0]), 1), device=pcont.device, dtype=pcont.dtype
            )
            running = lucid.cat([ones, pcont[:, : self._horizon - 1]], dim=1)
            weight = lucid.cumprod(running, dim=1).detach()

        scale = self.returns.update(target) if self.training else self.returns.scale
        actor_loss, entropy = self._actor_objective(
            states, actions, target, weight, scale
        )
        value_loss = self._critic_objective(states, target, weight)

        replay_loss: Tensor | None = None
        if self._replay_value_scale > 0.0:
            bootstrap = target[:, 0].reshape(batch, steps).detach()
            replay_loss = self._replay_critic_objective(
                kept,
                rewards[:, :steps],
                None if continues is None else continues[:, :steps],
                bootstrap,
            )
            value_loss = value_loss + self._replay_value_scale * replay_loss

        return DreamerV3BehaviorOutput(
            actor_loss=actor_loss,
            value_loss=value_loss,
            lambda_return=target,
            return_scale=scale,
            entropy=entropy,
            imagined_reward=reward,
            imagined_value=value,
            imagined_action=actions,
            imagined_discount=pcont,
            replay_value_loss=replay_loss,
        )

    def _actor_objective(
        self,
        states: RSSMState,
        actions: Tensor,
        target: Tensor,
        weight: Tensor,
        scale: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""The policy's objective — score function, on a scale-free advantage.

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
        scale : Tensor
            ``max(1, S)``, 0-d — the divisor that makes the advantage
            dimensionless.

        Returns
        -------
        actor_loss, entropy : Tensor
            The loss to minimise, and the mean entropy it was paid on.

        Notes
        -----
        .. math::

            \mathcal{L}(\theta) = -\,\mathbb{E}\Big[
                w_t \big(\mathrm{sg}\big[(R^\lambda_t - v(s_t)) / S\big]
                \log \pi_\theta(a_t \mid s_t)
                + \eta\, \mathrm{H}[\pi_\theta(\cdot \mid s_t)]\big)\Big].

        Only one estimator, for both action spaces.  Dreamer used dynamics
        backpropagation for continuous actions and the score function for
        discrete ones, and DreamerV2 kept the split; the normalisation
        above is what makes the score function's variance tolerable
        everywhere, so the split stops earning its keep.

        The division is by the *scale* alone and not a standardisation:
        subtracting the returns' mean as well would remove the sign of the
        advantage, which is the only thing the score function reads.
        """
        model = self.dreamer_v3
        horizon = self._horizon
        scored = states.map(lambda x: x[:, :horizon])
        feature = scored.feature.detach()

        baseline = model.predict_value(scored)
        advantage = ((target - baseline) / scale).detach()
        chosen = actions.detach()
        log_prob = model.actor.log_prob(feature, chosen)

        entropy = model.actor.entropy(feature)
        objective = log_prob * advantage + self._actor_entropy * entropy
        actor_loss = -(weight * objective).mean()
        return actor_loss, entropy.mean()

    def _critic_objective(
        self, states: RSSMState, target: Tensor, weight: Tensor
    ) -> Tensor:
        """Two-hot cross-entropy against the returns, plus the slow-copy pull.

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
        Two terms.  The first fits the returns; the second fits the slow
        copy's own prediction at the same states, weighted by
        ``critic_slowreg``.  The second is what a target critic used to
        do, arranged so that the targets themselves stay fresh: the
        returns bootstrap from the live critic and only the *pull* is
        slow, where DreamerV2 froze the targets and left the critic
        chasing a stale estimate between refreshes.

        Reads detached states, so the critic's gradient reaches nothing
        but the critic.
        """
        model = self.dreamer_v3
        detached = states.map(lambda x: x[:, : self._horizon].detach())
        feature = detached.feature
        fitted = model.value_head.cross_entropy(feature, target.detach())
        toward_slow = model.value_head.cross_entropy(
            feature, model.slow_value_head.predict(feature).detach()
        )
        return (weight * (fitted + self._critic_slowreg * toward_slow)).mean()

    def _replay_critic_objective(
        self,
        posteriors: RSSMState,
        rewards: Tensor,
        continues: Tensor | None,
        bootstrap: Tensor,
    ) -> Tensor:
        """The same critic loss, over what actually happened.

        Parameters
        ----------
        posteriors : RSSMState
            Filtered beliefs, ``(B, T, ·)``.
        rewards : Tensor
            Observed reward, ``(B, T)``.
        continues : Tensor or None
            ``(B, T)``, ``0`` where the episode terminated.
        bootstrap : Tensor
            The imagined return at each posterior state, ``(B, T)``.  Only
            its last column is used — it closes the replayed sequence with
            an estimate that looks past the end of the chunk, which the
            critic's own value at that step cannot do.

        Returns
        -------
        Tensor
            A scalar.

        Raises
        ------
        ValueError
            If the sequence is a single step, which leaves no return to
            compute.

        Notes
        -----
        The critic sees imagined states almost exclusively, and imagined
        states are wherever the world model happens to have wandered.
        Training it on replayed states as well anchors it to the
        distribution the agent is actually in.  This is the term the
        reference weights at ``0.3``.

        Features are detached — see the module docstring for why that
        differs from the reference and why the difference is not
        observable under this family's three-optimiser contract.
        """
        if int(rewards.shape[1]) < 2:
            raise ValueError(
                "the replayed critic term needs at least two steps, got "
                f"{int(rewards.shape[1])}"
            )
        model = self.dreamer_v3
        feature = posteriors.feature.detach()
        estimate = model.value_head.predict(feature).detach()
        # Everything but the last step is the critic's own estimate; the
        # last is the imagined return, which is the only thing here that
        # knows what comes after the chunk.
        value = lucid.cat([estimate[:, :-1], bootstrap[:, -1:]], dim=1)

        discount: float | Tensor = (
            self._discount if continues is None else self._discount * continues
        )
        target = lambda_return(rewards, value, discount, self._lambda)

        scored = feature[:, : int(target.shape[1])]
        fitted = model.value_head.cross_entropy(scored, target.detach())
        toward_slow = model.value_head.cross_entropy(
            scored, model.slow_value_head.predict(scored).detach()
        )
        return (fitted + self._critic_slowreg * toward_slow).mean()

    def backward(self, output: DreamerV3Output) -> None:
        """Give every parameter group the gradient of its own loss.

        Parameters
        ----------
        output : DreamerV3Output
            The result of :meth:`forward`, with ``behavior`` populated.

        Raises
        ------
        ValueError
            If ``output`` carries no losses.

        Notes
        -----
        Identical in shape to the earlier families', and identical in
        motivation: the three losses share one graph, so backpropagating
        them by hand either lets the actor's gradient descend the world
        model or raises on a parameter the imagination's graph still
        needed.

        Examples
        --------
        >>> import lucid
        >>> import lucid.optim as optim
        >>> from lucid.models import dreamer_v3_12m_world_model
        >>> model = dreamer_v3_12m_world_model(action_dim=2, cnn_depth=2,
        ...     stoch_size=3, discrete=4, deter_size=8, hidden_size=8,
        ...     actor_hidden=8, value_hidden=8, reward_hidden=8, num_bins=5,
        ...     horizon=3, pcont=False)
        >>> groups = (model.world_parameters(), model.actor_parameters(),
        ...           model.value_parameters())
        >>> opts = [optim.Adam(g, lr=1e-4) for g in groups]
        >>> out = model(lucid.randn((1, 3, 3, 64, 64)),
        ...             lucid.randn((1, 3, 2)), lucid.randn((1, 3)))
        >>> model.backward(out)
        >>> [all(p.grad is not None for p in g) for g in groups]
        [True, True, True]

        The slow critic is no optimiser's to move — it holds no gradient,
        and is pulled toward the learner by :meth:`update_slow_critic`,
        called once per step:

        >>> slow = model.dreamer_v3.slow_value_head.parameters()
        >>> any(p.grad is not None for p in slow)
        False
        >>> for opt in opts:
        ...     opt.step()
        >>> model.update_slow_critic()
        """
        behavior = output.behavior
        if output.loss is None or behavior is None:
            raise ValueError(
                "backward() needs the losses; this output came from "
                "DreamerV3Model rather than DreamerV3ForWorldModeling."
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
    ) -> DreamerV3Output:
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
        DreamerV3Output
            ``loss`` is the world-model loss; the behaviour losses are on
            ``.behavior`` and take their own optimisers.

        Raises
        ------
        ValueError
            If ``pcont`` is configured and ``discounts`` is omitted.

        Notes
        -----
        The reward term is a two-hot cross-entropy rather than a squared
        error, so a domain whose rewards are in the thousands produces
        gradients of the same size as one whose rewards are in
        hundredths.  The reconstruction stays a squared error: pixels are
        already in a fixed range and have nothing to normalise away.
        """
        model = self.dreamer_v3
        priors, posteriors = model.observe(observations, actions)
        reconstruction = model.decode(posteriors)

        b = int(observations.shape[0])
        t = int(observations.shape[1])
        diff = (reconstruction - observations) ** 2
        recon_loss = 0.5 * diff.reshape(b, t, -1).sum(dim=-1).mean()
        reward_loss = model.reward_head.loss(posteriors.feature, rewards)
        dynamics_loss, representation_loss = free_bits_kl(
            posteriors,
            priors,
            dyn_scale=self._dyn_scale,
            rep_scale=self._rep_scale,
            free_nats=self._free_nats,
        )
        kl_loss = dynamics_loss + representation_loss
        loss = self._pred_scale * (recon_loss + reward_loss) + self._kl_weight * kl_loss

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

        return DreamerV3Output(
            observation=reconstruction,
            reward=model.predict_reward(posteriors),
            value=model.predict_value(posteriors),
            posterior_stoch=posteriors.stoch,
            posterior_logits=cast(Tensor, posteriors.logits),
            prior_logits=cast(Tensor, priors.logits),
            deter=posteriors.deter,
            loss=loss,
            recon_loss=recon_loss,
            reward_loss=reward_loss,
            dynamics_loss=dynamics_loss,
            representation_loss=representation_loss,
            kl_loss=kl_loss,
            pcont_loss=pcont_loss,
            behavior=self._behavior(posteriors, rewards, discounts),
        )
