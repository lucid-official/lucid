"""PlaNet configuration — Hafner et al., 2019.

The Deep Planning Network: a latent dynamics model trained from pixels, whose
contribution is the recurrent state-space model (RSSM) that keeps a
deterministic path and a stochastic path side by side.

Defaults reproduce the paper's DeepMind Control Suite setup — 64x64 frames, a
200-unit deterministic state, a 30-unit stochastic state, a 1024-wide
embedding, and 3 free nats on the KL.
"""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import GenerativeActivation, WorldModelConfig


@model_family_meta(
    canonical_name="PlaNet",
    citation=(
        "Hafner, Danijar, et al. "
        '"Learning Latent Dynamics for Planning from Pixels." '
        "Proceedings of the 36th International Conference on Machine "
        "Learning, 2019, pp. 2555-2565."
    ),
    theory=r"""
    PlaNet learns the dynamics of an environment *in latent space* and plans
    there, never reconstructing a frame in order to choose an action.  The
    difficulty it addresses is that pixels are a partial observation: the
    agent must both remember what it has seen and stay uncertain about what
    it has not.

    Its answer is the **recurrent state-space model**, which refuses to pick
    between a deterministic and a stochastic latent and carries both.  A
    deterministic path :math:`h_t` is updated by a gated recurrence, and a
    stochastic latent :math:`s_t` is drawn conditioned on it:

    .. math::

        h_t = f\big(h_{t-1},\, s_{t-1},\, a_{t-1}\big),
        \qquad
        s_t \sim p\big(s_t \mid h_t\big).

    The motivation is stated as an ablation in the paper: a purely
    stochastic model cannot retain information over many steps, because each
    step's sampling noise degrades it; a purely deterministic model cannot
    represent the several futures a partially-observed environment permits,
    so it averages them and blurs.  Only the model carrying both matches the
    reported performance.

    Training maximises a variational bound.  An encoder supplies an
    approximate posterior :math:`q(s_t \mid h_t, o_t)` that has seen the
    frame; the objective rewards reconstructing the observation and the
    reward from that posterior while pulling the dynamics' own prior toward
    it:

    .. math::

        \mathcal{L} =
            \underbrace{\mathbb{E}_q\big[\log p(o_t \mid h_t, s_t)
                                    + \log p(r_t \mid h_t, s_t)\big]}
                       _{\text{reconstruction}}
            \;-\;
            \beta \underbrace{\mathrm{KL}\big(q(s_t \mid h_t, o_t)
                              \,\big\|\, p(s_t \mid h_t)\big)}
                       _{\text{consistency}} .

    The KL is clamped below at a **free-nats** threshold, so no gradient is
    spent driving an already-small divergence lower — without it the
    posterior collapses onto the prior and the latent stops carrying the
    observation.  The asymmetry is worth noting: the reconstruction term
    never reaches the prior head at all, since nothing reconstructed is
    computed from the prior.  The KL is the prior's only teacher.

    Once trained, planning runs the prior forward alone — the model
    imagines action sequences and scores them without touching the
    environment.  That the rollout happens entirely in a compact latent
    makes searching over thousands of candidate trajectories affordable,
    which is what lets PlaNet match model-free agents at a fraction of the
    episodes.
    """,
)
@dataclass(frozen=True)
class PlaNetConfig(WorldModelConfig):
    r"""Frozen configuration for the PlaNet family.

    Defaults reproduce Hafner et al., 2019 on the DeepMind Control Suite.

    Parameters
    ----------
    act_fn : {"silu", "swish", "relu", "gelu", "elu"}, default="relu"
        Activation throughout.  The paper uses ReLU, which is why this
        family overrides the generative-domain default of ``"silu"``.
        Dreamer uses ELU, which is why neither value sits on
        :class:`WorldModelConfig`.
    overshoot_distance : int or None, default=None
        How many steps ahead **latent overshooting** trains the dynamics.
        ``None`` overshoots as far as each batch allows (the paper's full
        setting); ``1`` disables it, leaving the ordinary one-step bound.
        Costs one extra recurrence sweep per distance — see Notes.
    overshoot_weight : float, default=1.0
        Multiplier on the averaged multi-step divergence.
    overshoot_reward_weight : float, default=1.0
        Multiplier on reward prediction at the overshot states.  Without
        it the reward head only ever sees posterior states while the
        planner only ever evaluates prior ones — see Notes.
    reward_hidden : int, default=200
        Width of each reward-head hidden layer.  Dreamer's is 300, so this
        stays here rather than on the shared base.
    reward_layers : int, default=2
        Number of reward-head hidden layers.
    reward_loss_scale : float, default=1.0
        Multiplier on the reward likelihood.  The paper states none, so
        the default is 1; the released implementation uses 10 — see
        Notes.

    Notes
    -----
    The state geometry, the frame size, the action width and the two KL
    knobs are inherited from :class:`WorldModelConfig`, which is where
    Dreamer reads the same values from.

    Reference: Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, and Davidson,
    *"Learning Latent Dynamics for Planning from Pixels"*, ICML, 2019
    (arXiv:1811.04551).

    ``sample_size`` is pinned to 64 rather than made general.  The encoder
    is four stride-2 convolutions with 4x4 kernels and the decoder four
    transposed convolutions with kernels ``5, 5, 6, 6`` — an irregular
    schedule that lands on ``64`` exactly (``1 -> 5 -> 13 -> 30 -> 64``) and
    on nothing else.  Generalising it would mean inventing a schedule the
    paper does not give, so an unsupported size raises here instead of
    silently reconstructing to the wrong shape.  Scale the model with
    ``cnn_depth`` / ``deter_size`` / ``stoch_size``.

    Examples
    --------
    >>> from lucid.models.generative.planet import PlaNetConfig
    >>> cfg = PlaNetConfig(action_dim=6)
    >>> cfg.stoch_size, cfg.deter_size, cfg.free_nats
    (30, 200, 3.0)
    >>> cfg.latent_size
    230
    """

    model_type: ClassVar[str] = "planet"

    act_fn: GenerativeActivation = "relu"

    overshoot_distance: int | None = None
    overshoot_weight: float = 1.0
    overshoot_reward_weight: float = 1.0

    reward_hidden: int = 200
    reward_layers: int = 2
    reward_loss_scale: float = 1.0

    @override
    def __post_init__(self) -> None:
        super().__post_init__()

        if self.reward_hidden <= 0:
            raise ValueError(f"reward_hidden must be positive, got {self.reward_hidden}")
        if self.reward_layers <= 0:
            raise ValueError(f"reward_layers must be positive, got {self.reward_layers}")
        if self.overshoot_distance is not None and self.overshoot_distance < 1:
            raise ValueError(
                "overshoot_distance must be at least 1 (1 disables overshooting) "
                f"or None for the full sequence, got {self.overshoot_distance}"
            )
        if self.overshoot_weight < 0.0:
            raise ValueError(f"overshoot_weight must be non-negative, got {self.overshoot_weight}")
        if self.reward_loss_scale < 0.0:
            raise ValueError(
                f"reward_loss_scale must be non-negative, got {self.reward_loss_scale}"
            )
        if self.overshoot_reward_weight < 0.0:
            raise ValueError(
                "overshoot_reward_weight must be non-negative, got "
                f"{self.overshoot_reward_weight}"
            )
