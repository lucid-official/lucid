"""Registry factories for DreamerV2.

The paper tunes two settings, and both ship — the same shape as
``ddpm_cifar`` / ``ddpm_lsun`` / ``ddpm_imagenet64``, which are dataset
configurations rather than size variants:

    * ``dreamer_v2``       — the released ``defaults`` block.
    * ``dreamer_v2_atari`` — discrete actions, a 600-unit state, and the
      paper's Atari divergence and discount.
    * ``dreamer_v2_dmc``   — continuous actions, a 200-unit state, and no
      discount head, because the Control Suite does not terminate.

Each has a ``_world_model`` counterpart carrying the objectives.  Set
``action_dim`` for your environment at ``create_model`` time — for Atari
that is how many buttons the game has.

The released ``crafter`` configuration is deliberately absent: it is a
later addition to the repository rather than something the paper reports,
and this zoo ships paper-cited configurations only.

What is *not* here is the parts of those configurations that are not the
model — action repeat, replay prefill, reward clipping and the learning
rates.  No config in this zoo carries them; ``action_repeat`` belongs to
:mod:`lucid.utils.rollout`, and the rates are quoted in
:class:`DreamerV2Config`'s Notes.

No parameter count is registered — the paper reports 20M for the world
model without breaking it down, and the docs site introspects the real
figure anyway.
"""

from dataclasses import replace

from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.generative.dreamer_v2._config import DreamerV2Config
from lucid.models.generative.dreamer_v2._model import DreamerV2ForWorldModeling, DreamerV2Model

_CFG_DREAMER_V2 = DreamerV2Config()


@register_model(
    task="base",
    family="dreamer_v2",
    model_type="dreamer_v2",
    model_class=DreamerV2Model,
    default_config=_CFG_DREAMER_V2,
)
def dreamer_v2(pretrained: bool = False, **overrides: object) -> DreamerV2Model:
    r"""Construct DreamerV2's world model, actor and critic — no objectives.

    Dreamer's architecture with a categorical latent: 32 variables of 32
    classes, sampled as one-hots and carried as a 1024-wide sparse binary
    vector.  Four-layer 400-unit ELU heads for reward, value and policy,
    plus a frozen copy of the critic to compute returns against.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV2Config` field overrides.  ``action_dim``
        is set by the environment, not by the paper.

    Returns
    -------
    DreamerV2Model
        The trunk, configured with the released defaults.

    Notes
    -----
    Reference: Hafner, Lillicrap, Norouzi, and Ba, *"Mastering Atari with
    Discrete World Models"*, ICLR, 2021 (arXiv:2010.02193).

    Examples
    --------
    >>> from lucid.models import dreamer_v2
    >>> model = dreamer_v2(action_dim=6).eval()
    >>> model.config.stoch_size, model.config.discrete
    (32, 32)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v2")
    return DreamerV2Model(apply_overrides(_CFG_DREAMER_V2, overrides))


@register_model(
    task="world-modeling",
    family="dreamer_v2",
    model_type="dreamer_v2",
    model_class=DreamerV2ForWorldModeling,
    default_config=_CFG_DREAMER_V2,
)
def dreamer_v2_world_model(
    pretrained: bool = False, **overrides: object
) -> DreamerV2ForWorldModeling:
    r"""Construct DreamerV2 with all three of its objectives.

    Same trunk as :func:`dreamer_v2`, wrapped with the world-model bound
    under a **balanced** divergence, an actor maximising imagined
    :math:`V_\lambda` with an entropy bonus, and a critic regressing onto
    a target copy of itself.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV2Config` field overrides.  ``kl_balance``
        and ``actor_grad`` are the two that change the algorithm rather
        than its size.

    Returns
    -------
    DreamerV2ForWorldModeling
        The trunk plus the objectives.

    Notes
    -----
    Reference: Hafner, Lillicrap, Norouzi, and Ba, *"Mastering Atari with
    Discrete World Models"*, ICLR, 2021 (arXiv:2010.02193).

    The three losses take **three separate optimisers**; use
    :meth:`~lucid.models.DreamerV2ForWorldModeling.backward` to fill their
    gradients, and call ``update_slow_target`` once per step.

    Examples
    --------
    >>> from lucid.models import dreamer_v2_world_model
    >>> model = dreamer_v2_world_model(action_dim=6).eval()
    >>> model.config.kl_balance, model.config.resolved_actor_grad
    (0.8, 'dynamics')
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v2_world_model")
    return DreamerV2ForWorldModeling(apply_overrides(_CFG_DREAMER_V2, overrides))


# The paper's Atari column, as the released implementation writes it.
# Discrete actions are what make this a different model rather than a
# resize: the policy becomes a one-hot and its gradient comes from the
# score function, since there is nothing to differentiate through.
_CFG_ATARI = replace(
    _CFG_DREAMER_V2,
    action_space="discrete",
    deter_size=600,
    hidden_size=600,
    discount=0.999,
    kl_weight=0.1,
    pcont=True,
    pcont_scale=5.0,
    actor_entropy=1e-3,
)

# The DeepMind Control column. `dmc_vision` and `dmc_proprio` differ only
# in what the encoder reads, and this family reads pixels, so they are one
# configuration here.
_CFG_DMC = replace(
    _CFG_DREAMER_V2,
    action_space="continuous",
    deter_size=200,
    hidden_size=200,
    discount=0.99,
    kl_weight=1.0,
    free_nats=1.0,
    pcont=False,
    actor_entropy=1e-4,
)


@register_model(
    task="base",
    family="dreamer_v2",
    model_type="dreamer_v2",
    model_class=DreamerV2Model,
    default_config=_CFG_ATARI,
)
def dreamer_v2_atari(pretrained: bool = False, **overrides: object) -> DreamerV2Model:
    r"""Construct DreamerV2 as the paper configures it for Atari.

    A 600-unit deterministic state, a divergence scaled to 0.1, a discount
    of 0.999 weighted five times over, and a **discrete** policy — a
    one-hot over the game's buttons rather than a box.

    The discount is the one number here the paper and the released
    configuration disagree on: Table D.1 prints 0.995, the released
    ``atari`` config uses 0.999, and the table's own note recommends
    searching :math:`\gamma \in \{0.99, 0.999\}`.  The released value
    is followed, because it is the one that produced the reported Atari
    results.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV2Config` field overrides.  ``action_dim``
        is the number of actions the game exposes.

    Returns
    -------
    DreamerV2Model
        The trunk, configured for Atari.

    Notes
    -----
    Reference: Hafner, Lillicrap, Norouzi, and Ba, *"Mastering Atari with
    Discrete World Models"*, ICLR, 2021 (arXiv:2010.02193).

    Action repeat of 4 is part of this configuration and is not a model
    field — pass it to :func:`lucid.utils.rollout.rollout`.

    Examples
    --------
    >>> from lucid.models import dreamer_v2_atari
    >>> model = dreamer_v2_atari(action_dim=18).eval()
    >>> model.config.action_space, model.config.resolved_actor_grad
    ('discrete', 'reinforce')
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v2_atari")
    return DreamerV2Model(apply_overrides(_CFG_ATARI, overrides))


@register_model(
    task="world-modeling",
    family="dreamer_v2",
    model_type="dreamer_v2",
    model_class=DreamerV2ForWorldModeling,
    default_config=_CFG_ATARI,
)
def dreamer_v2_atari_world_model(
    pretrained: bool = False, **overrides: object
) -> DreamerV2ForWorldModeling:
    r"""Construct the Atari configuration with all three objectives.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV2Config` field overrides.

    Returns
    -------
    DreamerV2ForWorldModeling
        The Atari trunk plus the objectives.

    Notes
    -----
    Reference: Hafner, Lillicrap, Norouzi, and Ba, *"Mastering Atari with
    Discrete World Models"*, ICLR, 2021 (arXiv:2010.02193).

    The actor's gradient is the score function here, resolved from the
    discrete action space — a one-hot offers nothing to differentiate
    through but a biased straight-through estimate.

    Examples
    --------
    >>> from lucid.models import dreamer_v2_atari_world_model
    >>> model = dreamer_v2_atari_world_model(action_dim=18).eval()
    >>> model.config.kl_weight, model.config.pcont_scale
    (0.1, 5.0)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v2_atari_world_model")
    return DreamerV2ForWorldModeling(apply_overrides(_CFG_ATARI, overrides))


@register_model(
    task="base",
    family="dreamer_v2",
    model_type="dreamer_v2",
    model_class=DreamerV2Model,
    default_config=_CFG_DMC,
)
def dreamer_v2_dmc(pretrained: bool = False, **overrides: object) -> DreamerV2Model:
    r"""Construct DreamerV2 as the paper configures it for the Control Suite.

    A 200-unit deterministic state, the full divergence floored at one
    free nat, a continuous policy, and **no discount head** — these
    episodes end on a clock rather than on a failure, and there is
    nothing for one to predict.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV2Config` field overrides.  ``action_dim``
        is set by the task.

    Returns
    -------
    DreamerV2Model
        The trunk, configured for the Control Suite.

    Notes
    -----
    Reference: Hafner, Lillicrap, Norouzi, and Ba, *"Mastering Atari with
    Discrete World Models"*, ICLR, 2021 (arXiv:2010.02193).

    The released implementation splits this into ``dmc_vision`` and
    ``dmc_proprio``, which differ only in whether the encoder reads pixels
    or joint angles.  This family reads pixels, so they are one
    configuration here.

    Examples
    --------
    >>> from lucid.models import dreamer_v2_dmc
    >>> model = dreamer_v2_dmc(action_dim=6).eval()
    >>> model.config.deter_size, model.config.pcont
    (200, False)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v2_dmc")
    return DreamerV2Model(apply_overrides(_CFG_DMC, overrides))


@register_model(
    task="world-modeling",
    family="dreamer_v2",
    model_type="dreamer_v2",
    model_class=DreamerV2ForWorldModeling,
    default_config=_CFG_DMC,
)
def dreamer_v2_dmc_world_model(
    pretrained: bool = False, **overrides: object
) -> DreamerV2ForWorldModeling:
    r"""Construct the Control Suite configuration with all three objectives.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV2Config` field overrides.

    Returns
    -------
    DreamerV2ForWorldModeling
        The Control Suite trunk plus the objectives.

    Notes
    -----
    Reference: Hafner, Lillicrap, Norouzi, and Ba, *"Mastering Atari with
    Discrete World Models"*, ICLR, 2021 (arXiv:2010.02193).

    Action repeat of 2 belongs to this configuration and is not a model
    field — pass it to :func:`lucid.utils.rollout.rollout`.

    Examples
    --------
    >>> from lucid.models import dreamer_v2_dmc_world_model
    >>> model = dreamer_v2_dmc_world_model(action_dim=6).eval()
    >>> model.config.actor_entropy
    0.0001
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v2_dmc_world_model")
    return DreamerV2ForWorldModeling(apply_overrides(_CFG_DMC, overrides))
