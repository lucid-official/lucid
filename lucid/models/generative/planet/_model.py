"""PlaNet model + task wrapper + output dataclass.

Shapes, following Hafner et al., 2019 and the released implementation's
``conv_ha.py`` (which the paper cites as "the convolutional and
deconvolutional networks from Ha & Schmidhuber, 2018")::

    encoder:  (B, T, 3, 64, 64)
              -> [Conv2d 4x4 s2 -> act] x 4   channels d, 2d, 4d, 8d
              -> flatten                       (B, T, 32d)      = 1024 at d=32

    dynamics: RSSM over (embed, action)        -> prior, posterior

    decoder:  feature = [h; s]
              -> Linear -> reshape (N, 32d, 1, 1)
              -> ConvT 5x5 s2 -> act           4d
              -> ConvT 5x5 s2 -> act           2d
              -> ConvT 6x6 s2 -> act           d
              -> ConvT 6x6 s2                  out_channels

    reward:   feature -> [Linear -> act] x L -> Linear -> (B, T)

The spatial chains are ``64 -> 31 -> 14 -> 6 -> 2`` down and
``1 -> 5 -> 13 -> 30 -> 64`` up, with no padding anywhere.  Both land on
their targets exactly, which is why the kernel schedule is irregular and
why :class:`PlaNetConfig` refuses any resolution other than 64.

The encoder, decoder and heads themselves live in ``_pixel_nets.py``:
Dreamer uses the same networks, so they are shared rather than copied.
"""

from dataclasses import dataclass
from functools import partial
from typing import ClassVar, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import WorldModelingModel
from lucid.models._output import ModelOutput
from lucid.models.generative._common._pixel_nets import (
    DenseHead,
    PixelDecoder,
    PixelEncoder,
)
from lucid.models.generative._common._rssm import RSSM, RSSMState, rssm_kl
from lucid.models.generative.planet._config import PlaNetConfig

# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PlaNetOutput(ModelOutput):
    r"""Forward output of the latent dynamics model.

    Attributes
    ----------
    observation : Tensor
        Reconstruction :math:`\hat{o}` shaped ``(B, T, C, 64, 64)``, decoded
        from the posterior.
    reward : Tensor
        Predicted reward shaped ``(B, T)``.
    posterior_stoch : Tensor
        The sampled latent :math:`s`, ``(B, T, stoch_size)``.
    posterior_mean, posterior_std : Tensor
        Parameters of :math:`q(s_t \mid h_t, o_t)`.
    prior_mean, prior_std : Tensor
        Parameters of :math:`p(s_t \mid h_t)` — what the dynamics predicted
        before seeing the frame.
    deter : Tensor
        The deterministic path :math:`h`, ``(B, T, deter_size)``.  Carried
        once, not twice: prior and posterior share it by construction,
        because the observation refines the belief about :math:`s_t`
        without changing the path that led there.
    loss : Tensor or None, optional
        ``recon_loss + reward_loss + kl_weight * kl_loss``.  ``None`` on
        :class:`PlaNetModel`, which builds no objective.
    recon_loss, reward_loss, kl_loss : Tensor or None, optional
        The one-step terms separately.  ``kl_loss`` is already clamped
        at ``free_nats``.
    overshoot_kl_loss, overshoot_reward_loss : Tensor or None, optional
        The latent-overshooting terms, averaged over distances.  ``None``
        when overshooting is off or the sequence is too short.  Reported
        separately because they enter ``loss`` with their own weights
        and are otherwise invisible.

    Notes
    -----
    Returned by both :meth:`PlaNetModel.forward` (losses ``None``) and
    :meth:`PlaNetForWorldModeling.forward` (losses populated).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.planet import PlaNetConfig, PlaNetModel
    >>> cfg = PlaNetConfig(action_dim=2, stoch_size=4, deter_size=8,
    ...                    hidden_size=8, cnn_depth=4, reward_hidden=8)
    >>> model = PlaNetModel(cfg).eval()
    >>> out = model(lucid.randn((1, 3, 3, 64, 64)), lucid.zeros(1, 3, 2))
    >>> out.observation.shape, out.reward.shape
    ((1, 3, 3, 64, 64), (1, 3))
    """

    observation: Tensor
    reward: Tensor
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
    overshoot_kl_loss: Tensor | None = None
    overshoot_reward_loss: Tensor | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Private building blocks
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Direct model — encoder, dynamics, decoder, reward head; no objective
# ─────────────────────────────────────────────────────────────────────────────


class PlaNetModel(PretrainedModel):
    r"""PlaNet's latent dynamics model (Hafner et al., 2019).

    A convolutional encoder embeds each frame, a recurrent state-space
    model carries the belief forward, and a decoder plus a reward head read
    predictions back off that belief.  The point of the architecture is
    that everything downstream of the encoder happens in a compact latent —
    which is what makes searching over thousands of imagined action
    sequences affordable.

    This class computes no loss.  Use :class:`PlaNetForWorldModeling` for
    the training objective, or read the distributions off the returned
    :class:`PlaNetOutput` and build your own.

    Parameters
    ----------
    config : PlaNetConfig
        Hyperparameters.  See :class:`PlaNetConfig`.

    Attributes
    ----------
    encoder : nn.Module
        Per-frame convolutional embedding.
    rssm : lucid.models.generative.RSSM
        The dynamics.  Shared with the Dreamer families by design.
    decoder : nn.Module
        Reconstructs a frame from ``[h; s]``.
    reward_head : nn.Module
        Predicts reward from ``[h; s]``.

    Notes
    -----
    Reference: Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, and Davidson,
    *"Learning Latent Dynamics for Planning from Pixels"*, ICML, 2019
    (arXiv:1811.04551).

    **Action alignment.**  ``actions[:, t]`` is the action taken *into*
    step ``t`` — the one that produced ``observations[:, t]``.  A caller
    holding :math:`(o_0, a_0, o_1, a_1, \dots)` shifts by one and passes a
    zero action at index 0.

    Planning is **not** implemented.  The paper pairs this model with a
    cross-entropy-method planner, which is a control algorithm rather than
    a network; :meth:`imagine` is the piece of it that belongs to the
    model, and the search on top is left to the caller.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.planet import PlaNetConfig, PlaNetModel
    >>> cfg = PlaNetConfig(action_dim=2, stoch_size=4, deter_size=8,
    ...                    hidden_size=8, cnn_depth=4, reward_hidden=8)
    >>> model = PlaNetModel(cfg).eval()
    >>> obs, act = lucid.randn((1, 3, 3, 64, 64)), lucid.zeros(1, 3, 2)
    >>> priors, posteriors = model.observe(obs, act)
    >>> posteriors.stoch.shape
    (1, 3, 4)
    >>> model.imagine(model.rssm.initial(1), act).stoch.shape
    (1, 3, 4)
    """

    config_class: ClassVar[type[PlaNetConfig]] = PlaNetConfig
    base_model_prefix: ClassVar[str] = "planet"

    def __init__(self, config: PlaNetConfig) -> None:
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
            Actions taken *into* each step, ``(B, T, action_dim)`` — see the
            class docstring on alignment.
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
        >>> from lucid.models.generative.planet import PlaNetConfig, PlaNetModel
        >>> cfg = PlaNetConfig(action_dim=2, stoch_size=4, deter_size=8,
        ...                    hidden_size=8, cnn_depth=4, reward_hidden=8)
        >>> model = PlaNetModel(cfg).eval()
        >>> obs, act = lucid.randn((1, 3, 3, 64, 64)), lucid.zeros(1, 3, 2)
        >>> priors, posteriors = model.observe(obs, act)
        >>> posteriors.mean.shape, posteriors.std.shape
        ((1, 3, 4), (1, 3, 4))

        The frame refines the latent, not the path that led to it, so prior
        and posterior share one deterministic state — and ``min_std``
        floors every scale:

        >>> bool((priors.deter == posteriors.deter).all())
        True
        >>> bool((posteriors.std >= cfg.min_std).all())
        True

        A long episode is filtered in chunks by carrying the last belief in:

        >>> last = posteriors.map(lambda t: t[:, -1])
        >>> model.observe(obs, act, last)[1].deter.shape
        (1, 3, 8)
        """
        draw = self._sample if sample is None else sample
        return self.rssm.observe(self.encode(observations), actions, state, sample=draw)

    def imagine(
        self, state: RSSMState, actions: Tensor, *, sample: bool | None = None
    ) -> RSSMState:
        """Roll the dynamics forward with no observations at all.

        Parameters
        ----------
        state : RSSMState
            The belief to roll from, ``(B, ·)``.
        actions : Tensor
            Actions to imagine taking, ``(B, T, action_dim)``.
        sample : bool or None, optional, keyword-only
            Draw the latent (``True``) or take its mean (``False``).
            ``None`` follows the config's ``mean_only`` setting.

        Returns
        -------
        RSSMState
            The imagined prior states, ``(B, T, ·)``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.planet import PlaNetConfig, PlaNetModel
        >>> cfg = PlaNetConfig(action_dim=2, stoch_size=4, deter_size=8,
        ...                    hidden_size=8, cnn_depth=4, reward_hidden=8)
        >>> model = PlaNetModel(cfg).eval()
        >>> obs, act = lucid.randn((1, 3, 3, 64, 64)), lucid.zeros(1, 3, 2)
        >>> _, posteriors = model.observe(obs, act)
        >>> belief = posteriors.map(lambda t: t[:, -1])

        Score four five-step plans from that one belief — the part of a
        planner that belongs to the model:

        >>> plans = lucid.randn((4, 5, 2)).clip(-1, 1)
        >>> starts = belief.map(lambda t: lucid.cat([t] * 4))
        >>> imagined = model.imagine(starts, plans, sample=False)
        >>> imagined.stoch.shape
        (4, 5, 4)
        >>> model.predict_reward(imagined).sum(dim=1).shape
        (4,)

        Planning takes the mean: a search that re-sampled each candidate
        would rank its own noise instead of the actions.

        >>> again = model.imagine(starts, plans, sample=False)
        >>> bool((imagined.stoch == again.stoch).all())
        True
        """
        draw = self._sample if sample is None else sample
        return self.rssm.imagine(state, actions, sample=draw)

    def decode(self, state: RSSMState) -> Tensor:
        """Reconstruct frames from a state — ``(B, T, C, 64, 64)``."""
        return self.decoder(state.feature)

    def predict_reward(self, state: RSSMState) -> Tensor:
        """Predict reward from a state — ``(B, T)``."""
        return self.reward_head(state.feature)

    @override
    def forward(self, observations: Tensor, actions: Tensor) -> PlaNetOutput:
        priors, posteriors = self.observe(observations, actions)
        posterior_mean, posterior_std = posteriors.gaussian()
        prior_mean, prior_std = priors.gaussian()
        return PlaNetOutput(
            observation=self.decode(posteriors),
            reward=self.predict_reward(posteriors),
            posterior_stoch=posteriors.stoch,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            prior_mean=prior_mean,
            prior_std=prior_std,
            deter=posteriors.deter,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Overshooting window helpers — ``(B, T, ...)`` state fields
# ─────────────────────────────────────────────────────────────────────────────


def _fold_window(x: Tensor, span: int, batch: int) -> Tensor:
    """The first ``span`` steps, with time folded into the batch axis."""
    return x[:, :span].reshape(batch * span, -1)


def _unfold_window(x: Tensor, span: int, batch: int) -> Tensor:
    """Inverse of :func:`_fold_window`: ``(batch * span, D) -> (batch, span, D)``."""
    return x.reshape(batch, span, -1)


def _detached_window(x: Tensor, start: int, span: int) -> Tensor:
    """Steps ``start .. start + span``, cut from the graph."""
    return x[:, start : start + span].detach()


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper — the variational objective
# ─────────────────────────────────────────────────────────────────────────────


class PlaNetForWorldModeling(WorldModelingModel):
    r"""PlaNet with the variational training objective.

    Wraps :class:`PlaNetModel` with the bound of Hafner et al., 2019:

    .. math::

        \mathcal{L} =
            \underbrace{\mathbb{E}_q\big[\log p(o_t \mid h_t, s_t)
                              + \log p(r_t \mid h_t, s_t)\big]}
                       _{\text{reconstruction}}
            \;-\;
            \beta\,\underbrace{\mathrm{KL}\big(q(s_t \mid h_t, o_t)
                              \,\|\, p(s_t \mid h_t)\big)}_{\text{consistency}} .

    Both likelihoods are unit-variance Gaussians, so their log-densities
    reduce to one-half squared error up to constants that do not affect the
    gradient — the observation term summed over pixels, the reward term a
    scalar.  The KL carries a free-nats floor; see :func:`rssm_kl`.

    Parameters
    ----------
    config : PlaNetConfig
        Hyperparameters.  ``free_nats`` and ``kl_weight`` shape the KL term.

    Attributes
    ----------
    planet : PlaNetModel
        The underlying trunk.

    Notes
    -----
    Reference: Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, and Davidson,
    *"Learning Latent Dynamics for Planning from Pixels"*, ICML, 2019
    (arXiv:1811.04551).

    **The reconstruction term never reaches the prior head.**  Nothing
    reconstructed is computed from the prior — the decoder and the reward
    head both read the posterior — so the KL is the prior's only teacher.
    Train without it and the dynamics never learn to predict, while every
    shape and every loss value still looks reasonable.

    The released implementation scales the reward term by 10 and uses
    3-layer, 300-unit heads.  Neither appears in the paper, which specifies
    "two fully connected layers of size 200"; the paper's values are used
    here, and the discrepancy is recorded rather than silently split.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.planet import (
    ...     PlaNetConfig, PlaNetForWorldModeling,
    ... )
    >>> cfg = PlaNetConfig(action_dim=2, stoch_size=4, deter_size=8,
    ...                    hidden_size=8, cnn_depth=4, reward_hidden=8)
    >>> model = PlaNetForWorldModeling(cfg).eval()
    >>> obs, act = lucid.randn((1, 3, 3, 64, 64)), lucid.zeros(1, 3, 2)
    >>> out = model(obs, act, rewards=lucid.zeros(1, 3))
    >>> out.loss.shape, out.observation.shape
    ((), (1, 3, 3, 64, 64))
    """

    config_class: ClassVar[type[PlaNetConfig]] = PlaNetConfig
    base_model_prefix: ClassVar[str] = "planet"

    def __init__(self, config: PlaNetConfig) -> None:
        super().__init__(config)
        self.planet = PlaNetModel(config)
        self._free_nats = config.free_nats
        self._kl_weight = config.kl_weight
        self._overshoot_distance = config.overshoot_distance
        self._overshoot_weight = config.overshoot_weight
        self._overshoot_reward_weight = config.overshoot_reward_weight
        self._reward_scale = config.reward_loss_scale

    def _overshooting(
        self, posteriors: RSSMState, actions: Tensor, rewards: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None]:
        r"""Latent overshooting — train the dynamics beyond one step ahead.

        The ordinary bound only ever asks the prior to predict the *next*
        state, so nothing stops multi-step predictions drifting: the model
        is never scored on where it thinks it will be in ten steps, which
        is exactly what planning needs.  Overshooting closes that by
        rolling the prior :math:`d` steps forward from each posterior and
        scoring it against the posterior that actually arrived:

        .. math::

            \frac{1}{D-1} \sum_{d=2}^{D} \sum_t
                \mathrm{KL}\big(
                    \mathrm{sg}\big[q(s_{t+d} \mid o_{\le t+d})\big]
                    \,\big\|\,
                    p(s_{t+d} \mid s_t, a_{t+1:t+d})
                \big).

        The posterior is stop-gradiented, so this term teaches the
        transition and never pulls the encoder toward being easier to
        predict.  ``d = 1`` is excluded because it *is* the ordinary KL,
        which the caller already charges.

        The reward head is supervised on the overshot states too.  It is
        otherwise trained only on posteriors while the planner evaluates it
        only on priors, and that mismatch is exactly what a planner scoring
        imagined trajectories would trip over.

        Returns ``(kl, reward)``, either of which may be ``None`` when the
        sequence is too short, overshooting is off, or no rewards were
        supplied.
        """
        t = int(actions.shape[1])
        limit = t - 1 if self._overshoot_distance is None else self._overshoot_distance
        limit = min(limit, t - 1)
        if limit < 2:
            return None, None

        b = int(actions.shape[0])
        # ``cur`` holds one rollout per start position; after step d, the
        # rollout that began at t is standing at t + d.  Advancing them all
        # at once keeps this to one recurrence call per distance rather
        # than one per (start, distance) pair.
        # Detached: the rollout *starts* from the posterior, so without
        # this the term would also push the encoder toward states that
        # happen to be easy to roll forward.  Both ends are held fixed
        # so only the transition learns.
        cur = posteriors.map(lambda x: x.detach())
        total: Tensor | None = None
        reward_total: Tensor | None = None
        counted = 0
        for d in range(1, limit + 1):
            span = t - d
            state = cur.map(partial(_fold_window, span=span, batch=b))
            step = self.planet.rssm.prior_step(
                state, actions[:, d : d + span].reshape(b * span, -1)
            )
            cur = step.map(partial(_unfold_window, span=span, batch=b))
            if d < 2:
                continue
            target = posteriors.map(partial(_detached_window, start=d, span=span))
            term = rssm_kl(target, cur, free_nats=0.0)
            total = term if total is None else total + term
            if rewards is not None and self._overshoot_reward_weight > 0.0:
                predicted = self.planet.predict_reward(cur)
                actual = rewards[:, d : d + span].detach()
                rterm = 0.5 * ((predicted - actual) ** 2).mean()
                reward_total = rterm if reward_total is None else reward_total + rterm
            counted += 1
        n = float(counted)
        return (
            None if total is None else total / n,
            None if reward_total is None else reward_total / n,
        )

    @override
    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
        rewards: Tensor | None = None,
    ) -> PlaNetOutput:
        priors, posteriors = self.planet.observe(observations, actions)
        reconstruction = self.planet.decode(posteriors)
        predicted = self.planet.predict_reward(posteriors)

        b = int(observations.shape[0])
        t = int(observations.shape[1])
        # Unit-variance Gaussian log-densities: one-half squared error,
        # summed over pixels for the frame and averaged over (B, T).
        diff = (reconstruction - observations) ** 2
        recon_loss = 0.5 * diff.reshape(b, t, -1).sum(dim=-1).mean()
        kl_loss = rssm_kl(posteriors, priors, free_nats=self._free_nats)

        reward_loss = None
        total = recon_loss + self._kl_weight * kl_loss
        if rewards is not None:
            reward_loss = self._reward_scale * 0.5 * ((predicted - rewards) ** 2).mean()
            total = total + reward_loss

        overshoot, overshoot_reward = self._overshooting(posteriors, actions, rewards)
        if overshoot is not None:
            total = total + self._overshoot_weight * overshoot
        if overshoot_reward is not None:
            total = total + self._overshoot_reward_weight * overshoot_reward

        posterior_mean, posterior_std = posteriors.gaussian()
        prior_mean, prior_std = priors.gaussian()
        return PlaNetOutput(
            observation=reconstruction,
            reward=predicted,
            posterior_stoch=posteriors.stoch,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            prior_mean=prior_mean,
            prior_std=prior_std,
            deter=posteriors.deter,
            loss=total,
            recon_loss=recon_loss,
            reward_loss=reward_loss,
            kl_loss=kl_loss,
            overshoot_kl_loss=overshoot,
            overshoot_reward_loss=overshoot_reward,
        )

    @lucid.no_grad()
    def plan(
        self,
        state: RSSMState,
        *,
        horizon: int = 12,
        iterations: int = 10,
        candidates: int = 1000,
        elites: int = 100,
    ) -> Tensor:
        r"""Choose an action by searching imagined trajectories (CEM).

        The paper's planner.  It never touches an environment: candidate
        action sequences are rolled forward through the learned dynamics
        and scored by the learned reward head, so the entire search is a
        forward pass over this model's own parameters.

        The cross-entropy method fits a diagonal Gaussian over action
        sequences to its own best samples, repeatedly:

        .. math::

            \mu, \sigma \;\leftarrow\;
                \mathrm{mean}, \mathrm{std}\Big(
                    \operatorname*{top-K}_{a^{(j)} \sim
                        \mathcal{N}(\mu, \sigma^2)}
                    \textstyle\sum_{\tau} r\big(s_\tau^{(j)}\big)
                \Big),

        and the first action of the final mean is returned.  Searching in
        a compact latent rather than in pixels is what makes a thousand
        candidates per step affordable.

        Parameters
        ----------
        state : RSSMState
            The belief to plan from, ``(B, ·)`` — typically one step of a
            posterior produced by :meth:`PlaNetModel.observe`.
        horizon : int, default=12
            How many steps ahead to imagine.
        iterations : int, default=10
            Refits of the action distribution.
        candidates : int, default=1000
            Sequences sampled per refit.
        elites : int, default=100
            Best sequences kept to refit from.

        Returns
        -------
        Tensor
            The first action of the planned sequence, ``(B, action_dim)``.

        Notes
        -----
        Defaults are the paper's: :math:`H = 12`, :math:`I = 10`,
        :math:`J = 1000`, :math:`K = 100`.

        The search is unbounded — the paper clips actions to the
        environment's range, which this model cannot know.  Clip the
        result yourself if your action space is bounded.

        Exploration noise is also the caller's: the paper adds Gaussian
        noise to the planned action when collecting episodes, which is a
        property of the data-collection loop rather than of the planner.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.planet import (
        ...     PlaNetConfig, PlaNetForWorldModeling,
        ... )
        >>> cfg = PlaNetConfig(action_dim=2, stoch_size=4, deter_size=8,
        ...                    hidden_size=8, cnn_depth=4, reward_hidden=8)
        >>> model = PlaNetForWorldModeling(cfg).eval()
        >>> start = model.planet.rssm.initial(1)
        >>> model.plan(start, horizon=3, iterations=2, candidates=8,
        ...            elites=2).shape
        (1, 2)
        """
        if elites > candidates:
            raise ValueError(
                f"elites ({elites}) cannot exceed candidates ({candidates})"
            )
        for name, value in (
            ("horizon", horizon),
            ("iterations", iterations),
            ("candidates", candidates),
            ("elites", elites),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")

        b = int(state.deter.shape[0])
        action_dim = self.planet.rssm.action_dim
        device, dt = state.deter.device.type, state.deter.dtype

        mean = lucid.zeros(b, horizon, action_dim, dtype=dt, device=device)
        std = lucid.ones(b, horizon, action_dim, dtype=dt, device=device)
        # One rollout per (batch item, candidate); the dynamics only know
        # how to advance a flat batch, so the candidate axis is folded in.
        tiled = state.map(lambda x: lucid.cat([x] * candidates, dim=0))

        for _ in range(iterations):
            noise = lucid.randn(
                (candidates, b, horizon, action_dim), device=device, dtype=mean.dtype
            )
            samples = mean + std * noise
            flat = samples.reshape(candidates * b, horizon, action_dim)

            rollout = self.planet.rssm.imagine(tiled, flat, sample=False)
            returns = self.planet.predict_reward(rollout).sum(dim=-1)
            returns = returns.reshape(candidates, b)

            order = lucid.argsort(returns, dim=0)
            keep = order[candidates - elites :].long()  # (elites, b)
            # Each batch item's elites, gathered on device.
            items = lucid.arange(b, device=device).long().reshape(1, b)
            chosen = samples[keep, items]  # (elites, b, horizon, action_dim)
            mean = chosen.mean(dim=0)
            std = ((chosen - mean) ** 2).mean(dim=0) ** 0.5 + 1e-6

        return mean[:, 0]
