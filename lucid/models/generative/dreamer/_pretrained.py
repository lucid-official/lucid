"""Registry factories for Dreamer.

Hafner et al., 2020 report one architecture at two settings.  Appendix A
gives the Control Suite setup in its main paragraph and then a second,
headed *Discrete control*, for Atari and DeepMind Lab.  The tasks inside
each setting differ only in action dimensionality and the action-repeat
constant — neither is a variant of the network — so under the project's
paper-cited-variants-only rule this family gets nominal names and no size
suffix:

    * ``dreamer``          — the Control Suite setting: a tanh-squashed
      Gaussian policy, 15-step horizon, unscaled divergence, no discount
      head.
    * ``dreamer_discrete`` — the *Discrete control* setting: a categorical
      policy over buttons, 10-step horizon, divergence scaled to 0.1, and a
      discount head.

Each has a ``_world_model`` counterpart carrying the objectives.  Set
``action_dim`` for your environment at ``create_model`` time — for Atari
that is how many buttons the game has.

Two things the *Discrete control* paragraph specifies are **not** model
fields and are absent here on purpose.  Its exploration is "epsilon greedy
where epsilon is linearly scheduled from 0.4 to 0.1 over the first 200,000
gradient steps", which belongs to whatever drives
:func:`lucid.utils.rollout.rollout`; and it bounds "rewards using tanh",
which is a property of the environment's reward, applied before the batch
reaches the model.  DreamerV2 draws the same line for the same two
quantities.

No parameter count is registered — the paper states no trainable-parameter
total, and the docs site introspects the real figure anyway.
"""

from dataclasses import replace

from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.generative.dreamer._config import DreamerConfig
from lucid.models.generative.dreamer._model import (
    DreamerForWorldModeling,
    DreamerModel,
)

# The paper's DeepMind Control Suite setup: PlaNet's world model unchanged,
# with 300-unit ELU heads, a 15-step imagination horizon and TD(0.95)
# returns.  Depths differ per head and are cited individually — the paper
# states three layers for the action and value models and says nothing
# about the reward model, whose two layers come from the released
# implementation (and match PlaNet's).
_CFG_DREAMER = DreamerConfig(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    action_dim=1,
    stoch_size=30,
    deter_size=200,
    hidden_size=200,
    cnn_depth=32,
    min_std=0.1,
    free_nats=3.0,
    horizon=15,
    discount=0.99,
    lambda_=0.95,
    actor_hidden=300,
    actor_layers=3,
    value_hidden=300,
    value_layers=3,
    reward_hidden=300,
    reward_layers=2,
)


# ── Trunk ────────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="dreamer",
    model_type="dreamer",
    model_class=DreamerModel,
    default_config=_CFG_DREAMER,
)
def dreamer(pretrained: bool = False, **overrides: object) -> DreamerModel:
    r"""Construct Dreamer's world model, actor and critic — no objectives.

    PlaNet's architecture unchanged — a four-layer stride-2 convolutional
    encoder, a recurrent state-space model with a 200-unit deterministic
    path and a 30-unit stochastic state, a mirrored decoder — plus the two
    heads that replace PlaNet's planner: a ``tanh``-squashed Gaussian actor
    and a scalar critic, each three 300-unit ELU layers.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises
        rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DreamerConfig` field overrides.  ``action_dim`` is
        the one you almost always need — it is set by the environment, not
        by the paper.

    Returns
    -------
    DreamerModel
        The trunk, configured with the paper defaults and any overrides.

    Notes
    -----
    Reference: Hafner, Lillicrap, Ba, and Norouzi, *"Dream to Control:
    Learning Behaviors by Latent Imagination"*, ICLR, 2020
    (arXiv:1912.01603).

    Examples
    --------
    >>> from lucid.models.generative.dreamer import dreamer
    >>> model = dreamer(action_dim=6).eval()
    >>> model.config.horizon, model.config.lambda_
    (15, 0.95)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer")
    return DreamerModel(apply_overrides(_CFG_DREAMER, overrides))


# ── World-modeling head ──────────────────────────────────────────────────────


@register_model(
    task="world-modeling",
    family="dreamer",
    model_type="dreamer",
    model_class=DreamerForWorldModeling,
    default_config=_CFG_DREAMER,
)
def dreamer_world_model(pretrained: bool = False, **overrides: object) -> DreamerForWorldModeling:
    r"""Construct Dreamer with all three of its objectives.

    Same trunk as :func:`dreamer`, wrapped with the world-model bound of
    Hafner et al., 2019 and the two behaviour losses of Hafner et al.,
    2020 — an actor maximising :math:`V_\lambda` over imagined
    trajectories and a critic regressing onto it.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerConfig` field overrides.  ``horizon``,
        ``lambda_`` and ``discount`` shape the imagined return;
        ``action_dim`` is set by the environment.

    Returns
    -------
    DreamerForWorldModeling
        The trunk plus the objectives.

    Notes
    -----
    Reference: Hafner, Lillicrap, Ba, and Norouzi, *"Dream to Control:
    Learning Behaviors by Latent Imagination"*, ICLR, 2020
    (arXiv:1912.01603).

    The three losses take **three separate optimisers** over the parameter
    groups the model exposes — see :class:`DreamerForWorldModeling`.

    Examples
    --------
    >>> from lucid.models.generative.dreamer import dreamer_world_model
    >>> model = dreamer_world_model(action_dim=6).eval()
    >>> len(model.actor_parameters()) > 0
    True
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_world_model")
    return DreamerForWorldModeling(apply_overrides(_CFG_DREAMER, overrides))


# The paper's *Discrete control* paragraph, which changes four things about
# the model and nothing else.  The categorical policy is what makes this a
# different network rather than a resize: an Atari action is a button, not
# a box, and the sample is drawn straight-through so the actor's gradient
# still arrives through it.
_CFG_DISCRETE = replace(
    _CFG_DREAMER,
    action_space="discrete",
    horizon=10,
    kl_weight=0.1,
    pcont=True,
)


@register_model(
    task="base",
    family="dreamer",
    model_type="dreamer",
    model_class=DreamerModel,
    default_config=_CFG_DISCRETE,
)
def dreamer_discrete(pretrained: bool = False, **overrides: object) -> DreamerModel:
    r"""Construct Dreamer as the paper configures it for Atari and DMLab.

    A categorical policy over the game's buttons, a 10-step imagination
    horizon, the divergence scaled to 0.1, and a discount head — the four
    changes the paper's *Discrete control* paragraph makes "to account for
    the higher complexity of these tasks".

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerConfig` field overrides.  ``action_dim`` is
        the number of actions the game exposes.

    Returns
    -------
    DreamerModel
        The trunk, configured for discrete control.

    Notes
    -----
    Reference: Hafner, Lillicrap, Ba, and Norouzi, *"Dream to Control:
    Learning Behaviors by Latent Imagination"*, ICLR, 2020
    (arXiv:1912.01603), Appendix A.

    The action is sampled with straight-through gradients, as the paper
    specifies.  That is a biased estimator, and it is the one this paper
    used; DreamerV2 later replaced it with the score function for discrete
    actions.

    Epsilon-greedy exploration and tanh reward bounding belong to the
    rollout and the environment respectively — see this module's docstring.

    Examples
    --------
    >>> from lucid.models.generative.dreamer import dreamer_discrete
    >>> model = dreamer_discrete(action_dim=18).eval()
    >>> model.config.action_space, model.config.horizon, model.config.kl_weight
    ('discrete', 10, 0.1)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_discrete")
    return DreamerModel(apply_overrides(_CFG_DISCRETE, overrides))


@register_model(
    task="world-modeling",
    family="dreamer",
    model_type="dreamer",
    model_class=DreamerForWorldModeling,
    default_config=_CFG_DISCRETE,
)
def dreamer_discrete_world_model(
    pretrained: bool = False, **overrides: object
) -> DreamerForWorldModeling:
    r"""Construct the discrete-control configuration with all three objectives.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerConfig` field overrides.

    Returns
    -------
    DreamerForWorldModeling
        The discrete-control trunk plus the objectives.

    Notes
    -----
    Reference: Hafner, Lillicrap, Ba, and Norouzi, *"Dream to Control:
    Learning Behaviors by Latent Imagination"*, ICLR, 2020
    (arXiv:1912.01603), Appendix A.

    ``pcont=True`` here, so ``forward`` needs its ``discounts`` argument —
    the discount head is "trained towards the soft labels of 0 and gamma",
    which is what this family already does with that tensor.

    Examples
    --------
    >>> from lucid.models.generative.dreamer import dreamer_discrete_world_model
    >>> model = dreamer_discrete_world_model(action_dim=18).eval()
    >>> model.config.pcont, model.config.pcont_scale
    (True, 10.0)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_discrete_world_model")
    return DreamerForWorldModeling(apply_overrides(_CFG_DISCRETE, overrides))
