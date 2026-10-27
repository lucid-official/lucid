"""DreamerV2 configuration — Hafner et al., 2021.

The change that names the paper is the latent: a grid of categorical
variables in place of Dreamer's diagonal Gaussian.  Everything else —
the actor, the critic, the imagined lambda-returns — is inherited, then
adjusted for the fact that the latent is now discrete.

Defaults are the released implementation's ``defaults`` block, which is
the configuration it ships rather than either of the two the paper tunes.
The Atari and DeepMind Control settings are tabulated in the class Notes
so nothing is lost by not picking one.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    GenerativeActivation,
    WorldModelConfig,
)

#: How the actor's gradient is estimated.  ``"dynamics"`` backpropagates
#: through the learned transition, as Dreamer does; ``"reinforce"`` uses the
#: score-function estimator; ``"both"`` interpolates between them.
#: ``"auto"`` resolves the way the released implementation does — the
#: analytic path for a continuous action, the score function for a discrete
#: one, where there is no path to differentiate.
ActorGradient = Literal["auto", "dynamics", "reinforce", "both"]

#: Whether actions are a bounded vector or a choice among alternatives.
ActionSpace = Literal["continuous", "discrete"]


@model_family_meta(
    canonical_name="Dreamer-v2",
    citation=(
        "Hafner, Danijar, et al. "
        '"Mastering Atari with Discrete World Models." '
        "International Conference on Learning Representations, 2021."
    ),
    theory=r"""
    DreamerV2 keeps its predecessor's shape — a recurrent state-space
    model, an actor and a critic trained on imagined trajectories — and
    changes what the stochastic latent *is*.  Where Dreamer draws
    :math:`s_t` from a diagonal Gaussian, this draws a grid of
    **categorical** variables: 32 of them, 32 classes each, sampled as
    one-hots.  The state is then a sparse binary vector of length 1024
    with exactly 32 bits set.

    A one-hot sample has no gradient, so the paper takes the
    **straight-through** estimator — the same device that makes vector
    quantisation trainable:

    .. math::

        \tilde{s} = \mathrm{sample} + p - \mathrm{sg}(p),

    which passes the sample forward and the class probabilities' gradient
    backward.

    The motivation the paper gives is representational rather than
    computational.  A Gaussian latent must describe a multi-modal future —
    a ball that will bounce left *or* right — with a single mean, and so
    describes neither; a categorical one can put mass on both outcomes and
    keep them apart.  Atari is full of such futures, which is where the
    Gaussian model had been losing.

    The second change is to the divergence.  A single KL term makes the
    prior chase the posterior and the posterior chase the prior at the
    same rate, and the paper observes the posterior giving way — it
    collapses toward a prior that has not learned anything yet.  **KL
    balancing** splits the term and weights the two directions
    separately:

    .. math::

        \mathcal{L}_{\mathrm{KL}} =
            \alpha\, \mathrm{KL}\big(\mathrm{sg}(q) \,\|\, p\big)
            + (1 - \alpha)\, \mathrm{KL}\big(q \,\|\, \mathrm{sg}(p)\big),

    with :math:`\alpha = 0.8`.  Most of the pressure is on the prior to
    explain what the posterior saw, and only a fifth on the posterior to
    stay predictable — which is the asymmetry that keeps the latent
    carrying the observation.

    Two smaller changes complete it.  The critic regresses onto a **target
    network** refreshed every hundred steps, so its own moving estimate
    does not chase itself.  And the actor's gradient can come either
    through the dynamics, as before, or from a score-function estimator;
    the paper uses the analytic path for continuous control and REINFORCE
    for Atari, where the actions are discrete and there is no path to
    differentiate.
    """,
)
@dataclass(frozen=True)
class DreamerV2Config(WorldModelConfig):
    r"""Frozen configuration for the DreamerV2 family.

    Defaults follow the released implementation's ``defaults`` block.
    Fields shared with PlaNet and Dreamer are inherited from
    :class:`WorldModelConfig`, and the ones this paper changes are
    re-declared here.

    Parameters
    ----------
    act_fn : {"silu", "swish", "relu", "gelu", "elu"}, default="elu"
        Activation throughout, as in Dreamer.
    stoch_size : int, default=32
        Number of categorical variables.  Note the change of meaning: in
        the Gaussian families this is the *width* of the latent vector,
        here it is how many independent categoricals make it up.
    discrete : int, default=32
        Classes per categorical variable.  The stochastic state is
        therefore ``stoch_size * discrete`` wide — 1024 by default, of
        which exactly ``stoch_size`` entries are 1.
    deter_size, hidden_size : int, default=1024, 1024
        Deterministic path and the RSSM heads' hidden width.
    cnn_depth : int, default=48
        First encoder convolution's width; the schedule is Dreamer's.
    free_nats : float, default=0.0
        Off by default — KL balancing does the job free nats was doing,
        and the released implementation sets it to zero.
    kl_weight : float, default=1.0
        Multiplier on the balanced divergence.
    kl_balance : float, default=0.8
        :math:`\alpha` — the share of the divergence that pulls the prior
        toward the posterior rather than the reverse.
    reward_hidden, reward_layers : int, default=400, 4
        Reward head.
    actor_hidden, actor_layers : int, default=400, 4
        Actor.
    value_hidden, value_layers : int, default=400, 4
        Critic.
    horizon : int, default=15
        Imagination horizon.
    discount : float, default=0.99
        Reward discount :math:`\gamma`.
    lambda_ : float, default=0.95
        TD(:math:`\lambda`) weighting.
    action_space : {"continuous", "discrete"}, default="continuous"
        What an action *is*.  ``"continuous"`` gives a truncated Gaussian
        over ``[-1, 1]^action_dim``, which is the Control Suite;
        ``"discrete"`` gives a one-hot over ``action_dim`` alternatives,
        which is Atari.  It also decides how the actor's gradient is
        estimated when ``actor_grad`` is left on ``"auto"``.
    actor_grad : {"auto", "dynamics", "reinforce", "both"}, default="auto"
        How the actor's gradient is estimated.  ``"auto"`` resolves as the
        released implementation does: the analytic path for a continuous
        action, the score function for a discrete one, where a one-hot
        offers nothing to differentiate through except a biased
        straight-through estimate.
    actor_grad_mix : float, default=0.1
        Weight on the dynamics term when ``actor_grad="both"``.
    actor_entropy : float, default=2e-3
        Coefficient on the actor's entropy bonus.
    actor_min_std : float, default=0.1
        Floor on the action distribution's scale.
    slow_target_update : int, default=100
        Gradient steps between refreshes of the critic's target copy.
    slow_target_fraction : float, default=1.0
        Fraction of the way the target moves at each refresh; ``1`` is a
        hard copy.
    pcont : bool, default=True
        Predict the discount.  Unlike Dreamer this is **on** by default —
        the paper's headline domain is Atari, where episodes end.
    pcont_scale : float, default=1.0
        Multiplier on the discount head's likelihood.
    pcont_layers : int, default=4
        Depth of the discount head; width follows ``value_hidden``.

    Notes
    -----
    Reference: Hafner, Lillicrap, Norouzi, and Ba, *"Mastering Atari with
    Discrete World Models"*, ICLR, 2021 (arXiv:2010.02193).

    The paper tunes two configurations, and the released implementation
    keeps both as overrides on the defaults used here:

    ==================  ==========  ==========  ============
    field               default     Atari       DMC (vision)
    ==================  ==========  ==========  ============
    ``deter_size``      1024        600         200
    ``hidden_size``     1024        600         200
    ``kl_weight``       1.0         0.1         1.0
    ``discount``        0.99        0.999       0.99
    ``actor_entropy``   2e-3        1e-3        1e-4
    ``pcont``           True        True        False
    ``pcont_scale``     1.0         5.0         1.0
    ==================  ==========  ==========  ============

    Learning rates are not config fields — no model config in this zoo
    carries them — but for the record the paper's are ``2e-4`` (world),
    ``4e-5`` (actor) and ``1e-4`` (critic) for Atari, against the released
    defaults of ``1e-4`` / ``8e-5`` / ``2e-4``.

    ``stoch_size`` changing meaning between this family and the Gaussian
    ones is deliberate: it is the same *slot* in the shared config — how
    much stochastic state there is — and re-naming it would break the
    inheritance for the sake of a word.  What it costs is that
    :attr:`latent_size` is no longer ``deter_size + stoch_size``, which is
    why this class overrides it.

    Examples
    --------
    >>> from lucid.models.generative.dreamer_v2 import DreamerV2Config
    >>> cfg = DreamerV2Config(action_dim=6)
    >>> cfg.stoch_size, cfg.discrete, cfg.stoch_width
    (32, 32, 1024)
    >>> cfg.latent_size
    2048
    >>> cfg.kl_balance
    0.8
    """

    model_type: ClassVar[str] = "dreamer_v2"

    act_fn: GenerativeActivation = "elu"

    stoch_size: int = 32
    discrete: int = 32
    deter_size: int = 1024
    hidden_size: int = 1024
    cnn_depth: int = 48

    free_nats: float = 0.0
    kl_weight: float = 1.0
    kl_balance: float = 0.8

    reward_hidden: int = 400
    reward_layers: int = 4
    actor_hidden: int = 400
    actor_layers: int = 4
    value_hidden: int = 400
    value_layers: int = 4

    horizon: int = 15
    discount: float = 0.99
    lambda_: float = 0.95

    action_space: ActionSpace = "continuous"
    actor_grad: ActorGradient = "auto"
    actor_grad_mix: float = 0.1
    actor_entropy: float = 2e-3
    actor_min_std: float = 0.1

    slow_target_update: int = 100
    slow_target_fraction: float = 1.0

    pcont: bool = True
    pcont_scale: float = 1.0
    pcont_layers: int = 4

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        for name, value in (
            ("discrete", self.discrete),
            ("horizon", self.horizon),
            ("reward_hidden", self.reward_hidden),
            ("reward_layers", self.reward_layers),
            ("actor_hidden", self.actor_hidden),
            ("actor_layers", self.actor_layers),
            ("value_hidden", self.value_hidden),
            ("value_layers", self.value_layers),
            ("pcont_layers", self.pcont_layers),
            ("slow_target_update", self.slow_target_update),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if self.discrete < 2:
            raise ValueError(
                f"discrete must be at least 2 — a one-class categorical carries "
                f"no information; got {self.discrete}"
            )
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {self.discount}")
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {self.lambda_}")
        if not 0.0 <= self.kl_balance <= 1.0:
            raise ValueError(f"kl_balance must be in [0, 1], got {self.kl_balance}")
        if not 0.0 <= self.actor_grad_mix <= 1.0:
            raise ValueError(f"actor_grad_mix must be in [0, 1], got {self.actor_grad_mix}")
        if not 0.0 < self.slow_target_fraction <= 1.0:
            raise ValueError(
                f"slow_target_fraction must be in (0, 1], got " f"{self.slow_target_fraction}"
            )
        if self.actor_entropy < 0.0:
            raise ValueError(f"actor_entropy must be non-negative, got {self.actor_entropy}")
        if self.pcont_scale < 0.0:
            raise ValueError(f"pcont_scale must be non-negative, got {self.pcont_scale}")

    @property
    def resolved_actor_grad(self) -> ActorGradient:
        """``actor_grad`` with ``"auto"`` settled against the action space.

        Returns
        -------
        {"dynamics", "reinforce", "both"}
            ``"reinforce"`` for a discrete action space, ``"dynamics"``
            for a continuous one, when ``actor_grad`` is ``"auto"``;
            otherwise whatever was asked for.
        """
        if self.actor_grad != "auto":
            return self.actor_grad
        return "reinforce" if self.action_space == "discrete" else "dynamics"

    @property
    def stoch_width(self) -> int:
        """Flattened width of the stochastic latent — ``stoch_size * discrete``."""
        return self.stoch_size * self.discrete

    @property
    @override
    def latent_size(self) -> int:
        """Width of ``[h; s]``, with ``s`` the flattened categorical grid."""
        return self.deter_size + self.stoch_width
