"""Dreamer configuration — Hafner et al., 2020.

Dreamer keeps PlaNet's world model and replaces its planner with two learned
networks: an actor that proposes actions and a critic that scores them.
Both are trained on trajectories the model imagines, so the gradient of the
return flows *through* the learned dynamics rather than through sampled
environment interaction.

Defaults reproduce the paper's DeepMind Control Suite setup — 64x64 frames,
a 200-unit deterministic state, a 30-unit stochastic state, 3-layer
300-unit ELU heads, a 15-step imagination horizon, and TD(0.95) returns.
"""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    GenerativeActivation,
    WorldModelConfig,
)


@model_family_meta(
    canonical_name="Dreamer",
    citation=(
        "Hafner, Danijar, et al. "
        '"Dream to Control: Learning Behaviors by Latent Imagination." '
        "International Conference on Learning Representations, 2020."
    ),
    theory=r"""
    Dreamer inherits PlaNet's recurrent state-space model and throws away
    its planner.  Where PlaNet searches over action sequences at every step
    — thousands of imagined trajectories to choose one action — Dreamer
    *learns* a policy, so acting costs a single forward pass.

    The trade is that a learned policy needs a learning signal, and the
    paper's contribution is where that signal comes from.  An **actor**
    :math:`a_\tau \sim q_\phi(a_\tau \mid s_\tau)` and a **critic**
    :math:`v_\psi(s_\tau)` are trained entirely on trajectories the world
    model imagines, never on environment interaction.  Because the imagined
    states are produced by a differentiable transition and the actions are
    reparameterised samples, the gradient of a predicted return flows
    *backwards through the dynamics* into the actor:

    .. math::

        \max_\phi \; \mathbb{E}_{q_\theta, q_\phi}
            \Big[\textstyle\sum_{\tau} V_\lambda(s_\tau)\Big].

    This is what separates it from a model-free actor-critic, which can
    only estimate that gradient from scalar rewards it has actually
    observed.  Here the model supplies an analytic path.

    The target :math:`V_\lambda` is an exponentially-weighted average of
    :math:`k`-step returns — TD(:math:`\lambda`) computed inside the
    imagination:

    .. math::

        V_\lambda(s_\tau) = (1 - \lambda)
            \sum_{n=1}^{H-1} \lambda^{n-1} V_N^n(s_\tau)
            + \lambda^{H-1} V_N^H(s_\tau),

    with each :math:`V_N^k` summing discounted imagined rewards and
    bootstrapping from the critic.  The weighting is the usual bias-variance
    dial: short horizons trust the critic, long ones trust the model, and
    :math:`\lambda` interpolates.  The critic then regresses onto that same
    quantity, which is why the two networks are trained together but with
    the target held fixed.

    Imagination is short — 15 steps in the paper — because the model's
    error compounds, and the critic's bootstrap is what lets a short
    horizon still represent long-term value.
    """,
)
@dataclass(frozen=True)
class DreamerConfig(WorldModelConfig):
    r"""Frozen configuration for the Dreamer family.

    Defaults reproduce Hafner et al., 2020 on the DeepMind Control Suite.
    Fields shared with PlaNet are inherited from :class:`WorldModelConfig`.

    Parameters
    ----------
    act_fn : {"silu", "swish", "relu", "gelu", "elu"}, default="elu"
        Activation throughout.  The paper specifies ELU, where PlaNet
        specifies ReLU — the two families genuinely differ here.
    horizon : int, default=15
        How many steps each imagined trajectory runs for.  Short on
        purpose: model error compounds, and the critic's bootstrap is what
        carries value beyond the horizon.
    discount : float, default=0.99
        Reward discount :math:`\gamma`.
    lambda_ : float, default=0.95
        TD(:math:`\lambda`) weighting between short- and long-horizon
        returns.  Trailing underscore because ``lambda`` is a keyword.
    cnn_act : {"silu", "swish", "relu", "gelu", "elu"}, default="relu"
        Activation for the convolutional encoder and decoder only. The
        released implementation carries two — ``dense_act`` is ELU and
        ``cnn_act`` is ReLU — and the paper states the first while
        saying nothing about the second, so this follows the rule the
        rest of this class uses when the paper is silent.
    actor_hidden, actor_layers : int, default=300, 3
        Width and depth of the actor.
    value_hidden, value_layers : int, default=300, 3
        Width and depth of the critic.
    actor_min_std : float, default=1e-4
        Floor on the action distribution's scale.
    actor_init_std : float, default=5.0
        Offset added before ``softplus`` so the policy starts wide enough
        to explore.
    actor_mean_scale : float, default=5.0
        The mean is passed through ``s * tanh(x / s)`` before squashing,
        which keeps it from saturating the outer ``tanh``.
    reward_hidden, reward_layers : int, default=300, 3
        Reward head.  Appendix A gives one shape for everything it does
        not name separately — "all other functions as three dense layers
        of size 300" — so the reward, action and value models share it.
        The released implementation disagrees on both counts (400 units,
        with two layers for reward and four for the actor); the paper is
        followed here, as everywhere else in this family.
        Width and depth of the reward head.  The paper names only the
        action and value models, so the depth here is the released
        implementation's — which also matches PlaNet's reward head.
    action_space : {"continuous", "discrete"}, default="continuous"
        What an action is.  ``"continuous"`` is the tanh-squashed Gaussian
        the Control Suite needs.  ``"discrete"`` makes the action model
        "predict the logits of a categorical distribution", which is what
        the paper's *Discrete control* paragraph specifies for Atari and
        DeepMind Lab, sampled with straight-through gradients so the
        actor's gradient still arrives through the action.
    detach_actor_input : bool, default=True
        Hide the state's gradient from the actor during imagination.  The
        released implementation does this; the paper does not discuss it.
        Setting ``False`` gives the exact gradient of the stated objective
        — see Notes.
    pcont : bool, default=False
        Predict the discount factor from each latent state instead of
        holding it constant.  The paper introduces this for "tasks with
        early termination"; with no termination there is nothing to learn,
        which is why it is off by default.
    pcont_scale : float, default=10.0
        Multiplier on the discount head's likelihood.
    pcont_layers : int, default=3
        Depth of the discount head.  Width follows ``value_hidden``.

    Notes
    -----
    Reference: Hafner, Lillicrap, Ba, and Norouzi, *"Dream to Control:
    Learning Behaviors by Latent Imagination"*, ICLR, 2020
    (arXiv:1912.01603).

    The actor emits a ``tanh``-squashed diagonal Gaussian, so actions are
    bounded to ``(-1, 1)`` — the Control Suite's range.  ``actor_min_std``,
    ``actor_init_std`` and ``actor_mean_scale`` are **not stated in the
    paper**; they are the released implementation's, and are exposed here
    rather than buried so a reader can see which numbers have a citation
    and which have a source.

    Where the paper and the released implementation disagree, the paper
    wins and the difference is recorded here rather than hidden:

    ==============  ====================  ==========================
    quantity        paper                 released implementation
    ==============  ====================  ==========================
    head width      300                   400 throughout
    actor depth     3                     4
    value depth     3                     3
    reward depth    not stated            2  (taken, as the default)
    CNN activation  not stated            ReLU (taken; dense stays ELU)
    ==============  ====================  ==========================

    ``detach_actor_input`` is the one behavioural fork.  The released
    implementation feeds the actor a ``stop_gradient``-ed state during
    imagination, which drops the terms in which a return depends on the
    policy *through the state it read*.  Keeping them (``False``) is the
    exact gradient of :math:`\max_\phi \mathbb{E}[\sum_\tau V_\lambda]`;
    dropping them is cheaper and lower-variance.  The default follows the
    released implementation because the paper is silent.

    The three losses need **three optimisers over three parameter groups**
    — see :class:`DreamerForWorldModeling`.  Summing them and taking one
    step is not this algorithm: the actor's gradient would then also
    descend the world model, and the critic would chase a target it is
    simultaneously moving.

    Examples
    --------
    >>> from lucid.models.generative.dreamer import DreamerConfig
    >>> cfg = DreamerConfig(action_dim=6)
    >>> cfg.horizon, cfg.lambda_, cfg.discount
    (15, 0.95, 0.99)
    >>> cfg.act_fn
    'elu'
    """

    model_type: ClassVar[str] = "dreamer"

    act_fn: GenerativeActivation = "elu"
    cnn_act: GenerativeActivation = "relu"

    horizon: int = 15
    discount: float = 0.99
    lambda_: float = 0.95

    actor_hidden: int = 300
    actor_layers: int = 3
    value_hidden: int = 300
    value_layers: int = 3

    actor_min_std: float = 1e-4
    actor_init_std: float = 5.0
    actor_mean_scale: float = 5.0

    reward_hidden: int = 300
    reward_layers: int = 3

    detach_actor_input: bool = True

    pcont: bool = False
    pcont_scale: float = 10.0
    pcont_layers: int = 3

    action_space: str = "continuous"

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_space not in ("continuous", "discrete"):
            raise ValueError(
                f"action_space must be 'continuous' or 'discrete', got " f"{self.action_space!r}"
            )
        for name, value in (
            ("horizon", self.horizon),
            ("actor_hidden", self.actor_hidden),
            ("actor_layers", self.actor_layers),
            ("value_hidden", self.value_hidden),
            ("value_layers", self.value_layers),
            ("reward_hidden", self.reward_hidden),
            ("reward_layers", self.reward_layers),
            ("pcont_layers", self.pcont_layers),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {self.discount}")
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {self.lambda_}")
        if self.actor_min_std <= 0.0:
            raise ValueError(f"actor_min_std must be positive, got {self.actor_min_std}")
        if self.pcont_scale < 0.0:
            raise ValueError(f"pcont_scale must be non-negative, got {self.pcont_scale}")
        if self.actor_mean_scale <= 0.0:
            raise ValueError(f"actor_mean_scale must be positive, got {self.actor_mean_scale}")
