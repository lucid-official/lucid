"""Registry factories for DreamerV3 — the paper's scaling ladder.

DreamerV3 is the family with the fewest configurations to ship and the
most published ones, which sounds contradictory until you look at what
the published configurations change.  The reference repository carries ten
domain sections — Minecraft, DMLab, Atari, ProcGen, Atari100k, Crafter,
two Control Suites, BSuite and LocoNav — and between them they set
``task``, ``run.steps``, ``run.envs`` and ``run.train_ratio``, and nothing
else.  They are *training schedules*, not model configurations, and the
paper's whole claim is that they can be: one setting of the model spans
every domain.

So there is no ``dreamer_v3_atari`` here, because there is no such model.
What there is instead is Table 3's ladder, six rungs from 12M to 400M
parameters, all six of which the paper trains (Figure 6c).  Each rung is
defined by one number — the MLP hidden width :math:`d` — from which the
recurrence (:math:`8d`, in eight blocks), the first convolution
(:math:`d/16`) and the classes per latent (:math:`d/16`) all follow.  The
number of layers and the number of latents are constant, as are every
hyperparameter of the objective.

Which rung the paper uses where, from Table 2: ``200m`` for Minecraft,
DMLab, ProcGen, Atari, Atari100K and BSuite; ``12m`` for both Control
Suites, where it reports the small model matching the large one at a
fraction of the cost.

The registered ``params`` are the paper's labels.  They are met because
the recurrence is block-diagonal — measured at 12.03M, 26.90M, 47.67M,
106.93M, 189.81M and 426.41M against labels of 12M, 25M, 50M, 100M, 200M
and 400M.  A dense GRU at the same widths costs 26.8M where the label
says 12M, which is why the block structure is not optional.

Set ``action_dim`` for your environment at ``create_model`` time, and
``action_space="discrete"`` when the actions are buttons rather than a
box.  Neither is a property of the rung.
"""

from dataclasses import replace

from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.generative.dreamer_v3._config import DREAMER_V3_SIZES, DreamerV3Config
from lucid.models.generative.dreamer_v3._model import DreamerV3ForWorldModeling, DreamerV3Model


def _rung(name: str) -> DreamerV3Config:
    """Build one rung of Table 3 from the model dimension it is defined by."""
    deter, hidden, classes, depth, units = DREAMER_V3_SIZES[name]
    return replace(
        DreamerV3Config(),
        deter_size=deter,
        hidden_size=hidden,
        discrete=classes,
        cnn_depth=depth,
        reward_hidden=units,
        actor_hidden=units,
        value_hidden=units,
    )


_CFG_12M = _rung("12m")
_CFG_25M = _rung("25m")
_CFG_50M = _rung("50m")
_CFG_100M = _rung("100m")
_CFG_200M = _rung("200m")
_CFG_400M = _rung("400m")


@register_model(
    task="base",
    family="dreamer_v3",
    model_type="dreamer_v3",
    model_class=DreamerV3Model,
    default_config=_CFG_12M,
    params=12_000_000,
    summary="auto",
)
def dreamer_v3_12m(pretrained: bool = False, **overrides: object) -> DreamerV3Model:
    r"""Construct the smallest rung — no objectives, just the model.

    A model dimension of 256: a 2048-unit recurrence in eight blocks, 16
    classes per latent, and a first convolution of 16 channels.  This is
    the rung the paper uses for both Control Suites, where it reports the
    same performance as the 200M model while being substantially faster.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV3Config` field overrides.  ``action_dim``
        and ``action_space`` are set by the environment, not by the rung.

    Returns
    -------
    DreamerV3Model
        Encoder, RSSM, decoder, reward head, both critics and the actor.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104), Table 3.

    Examples
    --------
    >>> from lucid.models import dreamer_v3_12m
    >>> model = dreamer_v3_12m(action_dim=6).eval()
    >>> model.config.deter_size, model.config.discrete, model.config.blocks
    (2048, 16, 8)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v3_12m")
    return DreamerV3Model(apply_overrides(_CFG_12M, overrides))


@register_model(
    task="world-modeling",
    family="dreamer_v3",
    model_type="dreamer_v3",
    model_class=DreamerV3ForWorldModeling,
    default_config=_CFG_12M,
    params=12_000_000,
    summary="auto",
)
def dreamer_v3_12m_world_model(
    pretrained: bool = False, **overrides: object
) -> DreamerV3ForWorldModeling:
    r"""The smallest rung with all three objectives.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV3Config` field overrides.

    Returns
    -------
    DreamerV3ForWorldModeling
        The rung plus the world-model, actor and critic objectives.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104).

    The three losses take **three separate optimisers**; use
    :meth:`~lucid.models.DreamerV3ForWorldModeling.backward` to fill their
    gradients, and call ``update_slow_critic`` once per step.

    Examples
    --------
    >>> from lucid.models import dreamer_v3_12m_world_model
    >>> model = dreamer_v3_12m_world_model(action_dim=6).eval()
    >>> model.config.free_nats, model.config.rep_scale
    (1.0, 0.1)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v3_12m_world_model")
    return DreamerV3ForWorldModeling(apply_overrides(_CFG_12M, overrides))


@register_model(
    task="base",
    family="dreamer_v3",
    model_type="dreamer_v3",
    model_class=DreamerV3Model,
    default_config=_CFG_25M,
    params=25_000_000,
    summary="auto",
)
def dreamer_v3_25m(pretrained: bool = False, **overrides: object) -> DreamerV3Model:
    r"""Construct the second rung — no objectives, just the model.

    A model dimension of 384.  The ladder steps by roughly 1.5 each rung,
    alternating powers of two with powers of two times 1.5, which keeps
    every tensor width a multiple of eight.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV3Config` field overrides.

    Returns
    -------
    DreamerV3Model
        The trunk at 25M parameters.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104), Table 3.

    Examples
    --------
    >>> from lucid.models import dreamer_v3_25m
    >>> model = dreamer_v3_25m(action_dim=6).eval()
    >>> model.config.hidden_size, model.config.cnn_depth
    (384, 24)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v3_25m")
    return DreamerV3Model(apply_overrides(_CFG_25M, overrides))


@register_model(
    task="world-modeling",
    family="dreamer_v3",
    model_type="dreamer_v3",
    model_class=DreamerV3ForWorldModeling,
    default_config=_CFG_25M,
    params=25_000_000,
    summary="auto",
)
def dreamer_v3_25m_world_model(
    pretrained: bool = False, **overrides: object
) -> DreamerV3ForWorldModeling:
    r"""The second rung with all three objectives.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV3Config` field overrides.

    Returns
    -------
    DreamerV3ForWorldModeling
        The rung plus the objectives.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104).

    Examples
    --------
    >>> from lucid.models import dreamer_v3_25m_world_model
    >>> model = dreamer_v3_25m_world_model(action_dim=6).eval()
    >>> model.config.horizon, model.config.lambda_
    (15, 0.95)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v3_25m_world_model")
    return DreamerV3ForWorldModeling(apply_overrides(_CFG_25M, overrides))


@register_model(
    task="base",
    family="dreamer_v3",
    model_type="dreamer_v3",
    model_class=DreamerV3Model,
    default_config=_CFG_50M,
    params=50_000_000,
    summary="auto",
)
def dreamer_v3_50m(pretrained: bool = False, **overrides: object) -> DreamerV3Model:
    r"""Construct the middle rung — no objectives, just the model.

    A model dimension of 512: a 4096-unit recurrence, 32 classes per
    latent, 32 channels at the first convolution.  These are the values
    :class:`DreamerV3Config` defaults to, so this rung and a bare
    configuration are the same model.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV3Config` field overrides.

    Returns
    -------
    DreamerV3Model
        The trunk at 50M parameters.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104), Table 3.

    Examples
    --------
    >>> from lucid.models import dreamer_v3_50m
    >>> model = dreamer_v3_50m(action_dim=6).eval()
    >>> model.config.deter_size, model.config.discrete
    (4096, 32)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v3_50m")
    return DreamerV3Model(apply_overrides(_CFG_50M, overrides))


@register_model(
    task="world-modeling",
    family="dreamer_v3",
    model_type="dreamer_v3",
    model_class=DreamerV3ForWorldModeling,
    default_config=_CFG_50M,
    params=50_000_000,
    summary="auto",
)
def dreamer_v3_50m_world_model(
    pretrained: bool = False, **overrides: object
) -> DreamerV3ForWorldModeling:
    r"""The middle rung with all three objectives.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV3Config` field overrides.

    Returns
    -------
    DreamerV3ForWorldModeling
        The rung plus the objectives.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104).

    Examples
    --------
    >>> from lucid.models import dreamer_v3_50m_world_model
    >>> model = dreamer_v3_50m_world_model(action_dim=6).eval()
    >>> model.config.critic_ema, model.config.replay_value_scale
    (0.02, 0.3)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v3_50m_world_model")
    return DreamerV3ForWorldModeling(apply_overrides(_CFG_50M, overrides))


@register_model(
    task="base",
    family="dreamer_v3",
    model_type="dreamer_v3",
    model_class=DreamerV3Model,
    default_config=_CFG_100M,
    params=100_000_000,
    summary="auto",
)
def dreamer_v3_100m(pretrained: bool = False, **overrides: object) -> DreamerV3Model:
    r"""Construct the fourth rung — no objectives, just the model.

    A model dimension of 768.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV3Config` field overrides.

    Returns
    -------
    DreamerV3Model
        The trunk at 100M parameters.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104), Table 3.

    Examples
    --------
    >>> from lucid.models import dreamer_v3_100m
    >>> model = dreamer_v3_100m(action_dim=6).eval()
    >>> model.config.deter_size, model.config.hidden_size
    (6144, 768)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v3_100m")
    return DreamerV3Model(apply_overrides(_CFG_100M, overrides))


@register_model(
    task="world-modeling",
    family="dreamer_v3",
    model_type="dreamer_v3",
    model_class=DreamerV3ForWorldModeling,
    default_config=_CFG_100M,
    params=100_000_000,
    summary="auto",
)
def dreamer_v3_100m_world_model(
    pretrained: bool = False, **overrides: object
) -> DreamerV3ForWorldModeling:
    r"""The fourth rung with all three objectives.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV3Config` field overrides.

    Returns
    -------
    DreamerV3ForWorldModeling
        The rung plus the objectives.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104).

    Examples
    --------
    >>> from lucid.models import dreamer_v3_100m_world_model
    >>> model = dreamer_v3_100m_world_model(action_dim=6).eval()
    >>> model.config.num_bins, model.config.bin_range
    (255, 20.0)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v3_100m_world_model")
    return DreamerV3ForWorldModeling(apply_overrides(_CFG_100M, overrides))


@register_model(
    task="base",
    family="dreamer_v3",
    model_type="dreamer_v3",
    model_class=DreamerV3Model,
    default_config=_CFG_200M,
    params=200_000_000,
    summary="auto",
)
def dreamer_v3_200m(pretrained: bool = False, **overrides: object) -> DreamerV3Model:
    r"""Construct the paper's default rung — no objectives, just the model.

    A model dimension of 1024: an 8192-unit recurrence in eight blocks, 64
    classes per latent, 64 channels at the first convolution.  This is the
    size behind every headline result in the paper except the two Control
    Suites — Minecraft, DMLab, ProcGen, Atari, Atari100K and BSuite all
    run here, with identical hyperparameters.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV3Config` field overrides.

    Returns
    -------
    DreamerV3Model
        The trunk at 200M parameters.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104), Tables 2 and 3.

    Examples
    --------
    >>> from lucid.models import dreamer_v3_200m
    >>> model = dreamer_v3_200m(action_dim=18, action_space="discrete").eval()
    >>> model.config.deter_size, model.config.discrete
    (8192, 64)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v3_200m")
    return DreamerV3Model(apply_overrides(_CFG_200M, overrides))


@register_model(
    task="world-modeling",
    family="dreamer_v3",
    model_type="dreamer_v3",
    model_class=DreamerV3ForWorldModeling,
    default_config=_CFG_200M,
    params=200_000_000,
    summary="auto",
)
def dreamer_v3_200m_world_model(
    pretrained: bool = False, **overrides: object
) -> DreamerV3ForWorldModeling:
    r"""The paper's default rung with all three objectives.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV3Config` field overrides.

    Returns
    -------
    DreamerV3ForWorldModeling
        The rung plus the objectives.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104).

    This is the configuration that collected diamonds in Minecraft from
    scratch — with, the paper is careful to note, the same hyperparameters
    as everything else.  What differs per benchmark is the *schedule*:
    action repeat, environment count and replay ratio, none of which are
    model fields.  Pass them to :func:`lucid.utils.rollout.rollout` and to
    your training loop.

    Examples
    --------
    >>> from lucid.models import dreamer_v3_200m_world_model
    >>> model = dreamer_v3_200m_world_model(action_dim=18).eval()
    >>> model.config.discount, model.config.actor_entropy
    (0.997, 0.0003)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v3_200m_world_model")
    return DreamerV3ForWorldModeling(apply_overrides(_CFG_200M, overrides))


@register_model(
    task="base",
    family="dreamer_v3",
    model_type="dreamer_v3",
    model_class=DreamerV3Model,
    default_config=_CFG_400M,
    params=400_000_000,
    summary="auto",
)
def dreamer_v3_400m(pretrained: bool = False, **overrides: object) -> DreamerV3Model:
    r"""Construct the largest rung — no objectives, just the model.

    A model dimension of 1536.  The top of the scaling study, where the
    paper reports that larger models not only score higher but reach a
    given score in *fewer* environment steps — the data efficiency
    improves with capacity rather than trading against it.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV3Config` field overrides.

    Returns
    -------
    DreamerV3Model
        The trunk at 400M parameters.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104), Figure 6c and Table 3.

    Examples
    --------
    >>> from lucid.models import dreamer_v3_400m
    >>> model = dreamer_v3_400m(action_dim=6).eval()
    >>> model.config.deter_size, model.config.hidden_size
    (12288, 1536)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v3_400m")
    return DreamerV3Model(apply_overrides(_CFG_400M, overrides))


@register_model(
    task="world-modeling",
    family="dreamer_v3",
    model_type="dreamer_v3",
    model_class=DreamerV3ForWorldModeling,
    default_config=_CFG_400M,
    params=400_000_000,
    summary="auto",
)
def dreamer_v3_400m_world_model(
    pretrained: bool = False, **overrides: object
) -> DreamerV3ForWorldModeling:
    r"""The largest rung with all three objectives.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`DreamerV3Config` field overrides.

    Returns
    -------
    DreamerV3ForWorldModeling
        The rung plus the objectives.

    Notes
    -----
    Reference: Hafner, Pasukonis, Ba, and Norouzi, *"Mastering Diverse
    Domains through World Models"*, Nature 640 (2025), 647-653
    (arXiv:2301.04104).

    Examples
    --------
    >>> from lucid.models import dreamer_v3_400m_world_model
    >>> model = dreamer_v3_400m_world_model(action_dim=6).eval()
    >>> model.config.blocks, model.config.deter_size // model.config.blocks
    (8, 1536)
    """
    if pretrained:
        reject_unavailable_pretrained("dreamer_v3_400m_world_model")
    return DreamerV3ForWorldModeling(apply_overrides(_CFG_400M, overrides))
