"""DreamerV3 configuration — Hafner et al., 2023.

The paper's claim is not a new mechanism but the absence of one: the same
hyperparameters solve Atari, DeepMind Control, Minecraft and the rest,
with no per-domain tuning.  Everything here exists to make that true —
transforms that flatten the scale of rewards, a divergence that cannot
collapse, and an actor objective normalised by its own return spread.

Defaults are the paper's, at the size whose categorical grid matches the
32x32 it describes.  The scaling ladder is a table below rather than a
knob, because the released implementation moves five quantities together.
"""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    GenerativeActivation,
    WorldModelConfig,
)

#: The released scaling ladder, ``name -> (deter, hidden, classes, depth, units)``.
#: The paper reports six sizes from 12M to 400M parameters; the released
#: configuration also carries a 1M entry, which is not in the paper and is
#: therefore not shipped as a factory.
DREAMER_V3_SIZES: dict[str, tuple[int, int, int, int, int]] = {
    "12m": (2048, 256, 16, 16, 256),
    "25m": (3072, 384, 24, 24, 384),
    "50m": (4096, 512, 32, 32, 512),
    "100m": (6144, 768, 48, 48, 768),
    "200m": (8192, 1024, 64, 64, 1024),
    "400m": (12288, 1536, 96, 96, 1536),
}


@model_family_meta(
    canonical_name="Dreamer-v3",
    citation=(
        "Hafner, Danijar, et al. "
        '"Mastering Diverse Domains through World Models." '
        "Nature, vol. 640, 2025, pp. 647-653."
    ),
    theory=r"""
    DreamerV3's contribution is that its hyperparameters stop mattering.
    Its predecessors needed a different divergence scale for Atari than
    for continuous control, a different entropy bonus, a different
    discount; this one setting spans them, and the paper's headline
    result — collecting diamonds in Minecraft from scratch — is run with
    the same numbers as everything else.

    Four changes buy that, and each removes a scale the practitioner was
    otherwise forced to supply.

    **Symlog.** Rewards differ by orders of magnitude between domains, and
    a network fitted to one range is wrong on another.  Compressing them
    through :math:`\mathrm{sign}(x)\ln(|x| + 1)` leaves small values
    alone and folds large ones in, so the same head covers both.

    **Two-hot regression.** Reward and value heads predict a
    *distribution* over exponentially spaced bins rather than a number.
    A squared error's gradient scales with the error, which is precisely
    the coupling to reward magnitude the paper is trying to break; a
    cross-entropy's does not.  It also lets a prediction be bimodal,
    which the mean of a squared-error head cannot represent.

    **Free bits.** The divergence is clipped below at one nat per
    variable:

    .. math::

        \mathcal{L}_{\mathrm{dyn}} = \max\big(1,\;
            \mathrm{KL}[\mathrm{sg}(q) \,\|\, p]\big),
        \qquad
        \mathcal{L}_{\mathrm{rep}} = \max\big(1,\;
            \mathrm{KL}[q \,\|\, \mathrm{sg}(p)\big]),

    weighted :math:`\beta_{\mathrm{dyn}} = 1` and
    :math:`\beta_{\mathrm{rep}} = 0.1`.  Below one nat the term is flat,
    so no gradient is spent closing a gap that is already small — which
    is what stopped the posterior collapsing and made the KL scale a
    tuning knob in the first place.  The categoricals are also mixed with
    1% uniform, so no class can reach probability zero and strand its
    gradient.

    **Normalised returns.** The actor divides its objective by the spread
    of its own recent returns,

    .. math::

        S = \mathrm{EMA}\big(\mathrm{Per}(R^\lambda, 95)
                           - \mathrm{Per}(R^\lambda, 5),\; 0.99\big),

    dividing by :math:`\max(1, S)` so that small returns are left alone
    rather than amplified into noise.  A fixed entropy bonus then means
    the same thing in every domain, because the thing it is traded
    against has been made scale-free.
    """,
)
@dataclass(frozen=True)
class DreamerV3Config(WorldModelConfig):
    r"""Frozen configuration for the DreamerV3 family.

    Defaults are the paper's values at the ``50m`` rung of the scaling
    ladder, whose 32-class categorical grid is the one the paper
    describes.  Fields shared with the earlier world models are inherited
    from :class:`WorldModelConfig`.

    Parameters
    ----------
    act_fn : {"silu", "swish", "relu", "gelu", "elu"}, default="silu"
        Activation throughout.  DreamerV2 used ELU; this uses SiLU.
    stoch_size : int, default=32
        Number of categorical variables.
    discrete : int, default=32
        Classes per variable.
    deter_size, hidden_size : int, default=4096, 512
        Deterministic path and the RSSM heads' hidden width.
    cnn_depth : int, default=32
        First encoder convolution's width.
    unimix : float, default=0.01
        Uniform mass mixed into every categorical.  Keeps a class from
        reaching probability zero, where its gradient would vanish.
    blocks : int, default=8
        Groups in the block-diagonal sequence model.  The paper's
        recurrence is eight times the hidden width, split into eight
        independent blocks of that width — which is the only reason a
        4096-unit recurrence fits inside a 50M-parameter model.  Must
        divide ``deter_size``.
    free_nats : float, default=1.0
        Free bits — the divergence is flat below this, per step.  One nat
        is the paper's value, and the reason it no longer needs tuning.
    kl_weight : float, default=1.0
        Overall multiplier on the divergence, on top of the split below.
    dyn_scale, rep_scale : float, default=1.0, 0.1
        :math:`\beta_{\mathrm{dyn}}` and :math:`\beta_{\mathrm{rep}}` —
        how hard the prior is pulled toward the posterior against the
        reverse.  The asymmetry is DreamerV2's KL balancing, restated.
    pred_scale : float, default=1.0
        :math:`\beta_{\mathrm{pred}}` on the reconstruction, reward and
        continuation likelihoods.
    num_bins : int, default=255
        Bins in the reward and value heads' two-hot output.
    bin_range : float, default=20.0
        The grid spans ``[-bin_range, +bin_range]`` in symlog space, so in
        reward units it reaches ``symexp(20)`` — roughly 5e8.
    reward_hidden, reward_layers : int, default=512, 1
        Reward head.  One hidden layer, as the reference's ``rewhead``
        has — the three-layer stack is the actor's and the critic's.
    actor_hidden, actor_layers : int, default=512, 3
        Actor.
    value_hidden, value_layers : int, default=512, 3
        Critic.
    horizon : int, default=15
        Imagination horizon.  ``H`` in the paper's Table 4.
    discount : float, default=0.997
        Reward discount.  Stated as a horizon of 333 steps in the
        released configuration, which is the same number.
    lambda_ : float, default=0.95
        TD(:math:`\lambda`) weighting.
    actor_entropy : float, default=3e-4
        Entropy bonus.  Fixed across domains, which only works because
        the returns it competes with are normalised.
    actor_min_std : float, default=0.1
        Floor on the continuous policy's standard deviation, added
        after the squashing: ``std = 2 * sigmoid(raw / 2) + min_std``.
        Without a floor the actor can drive the deviation to zero,
        which ends exploration and leaves the entropy bonus fighting a
        term that has already collapsed.  Ignored when
        ``action_space="discrete"``.
    return_ema_decay : float, default=0.99
        Decay of the moving return-spread estimate.
    return_low, return_high : float, default=5.0, 95.0
        Percentiles whose difference is that spread.
    critic_ema : float, default=0.02
        Rate at which the critic's slow copy follows it.  DreamerV2 used
        a hard copy every hundred steps; this is a continuous average.
    critic_slowreg : float, default=1.0
        Weight on the term pulling the critic toward its own slow copy.
        The slow copy is a regulariser here, not the source of the
        targets — the returns are bootstrapped from the *live* critic,
        which is the opposite of DreamerV2's arrangement.
    replay_value_scale : float, default=0.3
        Weight on the critic's loss over *replayed* trajectories, in
        addition to imagined ones.
    pcont : bool, default=True
        Build the continuation head, which predicts whether the episode
        is still running one step on.  With it, the discount applied to
        an imagined step is that predicted probability rather than the
        constant ``discount``, so a trajectory stops being credited
        once the model expects it to have ended — the difference
        between a world model that knows about termination and one that
        assumes the episode runs forever.  Costs one filtered step per
        sequence, because the last one has no successor to label it, so
        training needs sequences of at least two.
    pcont_scale : float, default=1.0
        Weight on the continuation head's loss within the world-model
        objective.
    pcont_layers : int, default=1
        Hidden layers in that head.  One is enough because it predicts
        a single Bernoulli from a state the RSSM has already built.
    action_space : {"continuous", "discrete"}, default="continuous"
        What an action is, as in DreamerV2.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104).

    The scaling ladder moves five quantities together, so it is a table
    rather than a knob — see :data:`DREAMER_V3_SIZES` and the factories
    built from it:

    ========  =======  ========  =======  =====  =====
    size      deter    hidden    classes  depth  units
    ========  =======  ========  =======  =====  =====
    ``12m``   2048     256       16       16     256
    ``25m``   3072     384       24       24     384
    ``50m``   4096     512       32       32     512
    ``100m``  6144     768       48       48     768
    ``200m``  8192     1024      64       64     1024
    ``400m``  12288    1536      96       96     1536
    ========  =======  ========  =======  =====  =====

    Two details of that table are worth writing down.  Its ``12m`` column
    prints 1024 recurrent units where the stated rule — eight times the
    hidden size — gives 2048; the released implementation uses 2048, and
    so does this.  And the paper's own default is the ``200m`` rung for
    every benchmark except the two control suites, which use ``12m``; the
    defaults here are ``50m``, the middle of the ladder, because a 200M
    default would make merely constructing the model an expensive
    operation.  Use :func:`dreamer_v3_200m` for the paper's setting.

    The released implementation carries a few things the paper does not
    report — extra dynamics layers, an ``absolute`` posterior variant, and
    a value- and advantage-normalisation pair disabled in every published
    configuration.  Where the two disagree the paper is followed, as it is
    for the earlier families.

    Examples
    --------
    >>> from lucid.models.generative.dreamer_v3 import DreamerV3Config
    >>> cfg = DreamerV3Config(action_dim=6)
    >>> cfg.stoch_size, cfg.discrete, cfg.free_nats
    (32, 32, 1.0)
    >>> cfg.discount, cfg.horizon, cfg.num_bins
    (0.997, 15, 255)
    """

    model_type: ClassVar[str] = "dreamer_v3"

    act_fn: GenerativeActivation = "silu"

    stoch_size: int = 32
    discrete: int = 32
    deter_size: int = 4096
    hidden_size: int = 512
    cnn_depth: int = 32
    unimix: float = 0.01
    blocks: int = 8

    free_nats: float = 1.0
    kl_weight: float = 1.0
    dyn_scale: float = 1.0
    rep_scale: float = 0.1
    pred_scale: float = 1.0

    num_bins: int = 255
    bin_range: float = 20.0

    reward_hidden: int = 512
    reward_layers: int = 1
    actor_hidden: int = 512
    actor_layers: int = 3
    value_hidden: int = 512
    value_layers: int = 3

    horizon: int = 15
    discount: float = 0.997
    lambda_: float = 0.95

    actor_entropy: float = 3e-4
    actor_min_std: float = 0.1
    return_ema_decay: float = 0.99
    return_low: float = 5.0
    return_high: float = 95.0

    critic_ema: float = 0.02
    critic_slowreg: float = 1.0
    replay_value_scale: float = 0.3

    pcont: bool = True
    pcont_scale: float = 1.0
    pcont_layers: int = 1

    action_space: str = "continuous"

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        for name, value in (
            ("discrete", self.discrete),
            ("horizon", self.horizon),
            ("num_bins", self.num_bins),
            ("reward_hidden", self.reward_hidden),
            ("reward_layers", self.reward_layers),
            ("actor_hidden", self.actor_hidden),
            ("actor_layers", self.actor_layers),
            ("value_hidden", self.value_hidden),
            ("value_layers", self.value_layers),
            ("pcont_layers", self.pcont_layers),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if self.discrete < 2:
            raise ValueError(f"discrete must be at least 2, got {self.discrete}")
        if self.num_bins < 2:
            raise ValueError(
                f"num_bins must be at least 2 — a two-hot needs two bins to "
                f"interpolate between; got {self.num_bins}"
            )
        if self.blocks < 1:
            raise ValueError(f"blocks must be at least 1, got {self.blocks}")
        if self.deter_size % self.blocks:
            raise ValueError(
                f"blocks must divide deter_size, got {self.deter_size} and " f"{self.blocks}"
            )
        if not 0.0 <= self.unimix < 1.0:
            raise ValueError(f"unimix must be in [0, 1), got {self.unimix}")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {self.discount}")
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {self.lambda_}")
        if self.bin_range <= 0.0:
            raise ValueError(f"bin_range must be positive, got {self.bin_range}")
        if not 0.0 <= self.return_low < self.return_high <= 100.0:
            raise ValueError(
                f"percentiles must satisfy 0 <= low < high <= 100, got "
                f"{self.return_low} and {self.return_high}"
            )
        if not 0.0 < self.critic_ema <= 1.0:
            raise ValueError(f"critic_ema must be in (0, 1], got {self.critic_ema}")
        if not 0.0 <= self.return_ema_decay < 1.0:
            raise ValueError(f"return_ema_decay must be in [0, 1), got {self.return_ema_decay}")
        for label, scale in (
            ("dyn_scale", self.dyn_scale),
            ("rep_scale", self.rep_scale),
            ("pred_scale", self.pred_scale),
            ("actor_entropy", self.actor_entropy),
            ("critic_slowreg", self.critic_slowreg),
            ("replay_value_scale", self.replay_value_scale),
            ("pcont_scale", self.pcont_scale),
        ):
            if scale < 0.0:
                raise ValueError(f"{label} must be non-negative, got {scale}")
        if self.action_space not in ("continuous", "discrete"):
            raise ValueError(
                f"action_space must be 'continuous' or 'discrete', got " f"{self.action_space!r}"
            )

    @property
    def stoch_width(self) -> int:
        """Flattened width of the stochastic latent — ``stoch_size * discrete``."""
        return self.stoch_size * self.discrete

    @property
    @override
    def latent_size(self) -> int:
        """Width of ``[h; s]``, with ``s`` the flattened categorical grid."""
        return self.deter_size + self.stoch_width
