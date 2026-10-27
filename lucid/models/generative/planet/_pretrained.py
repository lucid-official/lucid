"""Registry factories for PlaNet.

Hafner et al., 2019 report **one** architecture, evaluated across six
DeepMind Control Suite tasks.  The tasks differ in two things, neither of
which is a variant of the network: the environment's action
dimensionality, which is a property of the data in the same way
``num_classes`` is, and the action-repeat constant, which belongs to a
rollout loop this package does not ship.  The encoder, decoder, RSSM and
reward head are identical across all six.

So under the project's paper-cited-variants-only rule this family gets
nominal names and no size suffix:

    * ``planet``             — encoder / dynamics / decoder / reward head.
    * ``planet_world_model`` — the same trunk with the variational objective.

Set ``action_dim`` for your environment at ``create_model`` time; the
Control Suite tasks in the paper range from 1 to 6.

No parameter count is registered — the paper states no trainable-parameter
total, and the docs site introspects the real figure anyway.
"""

from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.generative.planet._config import PlaNetConfig
from lucid.models.generative.planet._model import PlaNetForWorldModeling, PlaNetModel

# The paper's DeepMind Control Suite setup: 64x64 frames, a 200-unit
# deterministic path, a 30-unit stochastic state, two 200-unit layers for
# every other function, and 3 free nats on the divergence.
_CFG_PLANET = PlaNetConfig(
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
    reward_hidden=200,
    reward_layers=2,
)


# ── Trunk ────────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="planet",
    model_type="planet",
    model_class=PlaNetModel,
    default_config=_CFG_PLANET,
)
def planet(pretrained: bool = False, **overrides: object) -> PlaNetModel:
    r"""Construct PlaNet's latent dynamics model — no training objective.

    The architecture of Hafner et al., 2019: a four-layer stride-2
    convolutional encoder, a recurrent state-space model with a 200-unit
    deterministic path and a 30-unit stochastic state, a mirrored
    transposed-convolutional decoder, and a two-layer reward head.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises
        rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`PlaNetConfig` field overrides.  ``action_dim`` is
        the one you almost always need — it is set by the environment, not
        by the paper.

    Returns
    -------
    PlaNetModel
        The trunk, configured with the paper defaults and any overrides.

    Notes
    -----
    Reference: Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, and Davidson,
    *"Learning Latent Dynamics for Planning from Pixels"*, ICML, 2019
    (arXiv:1811.04551).

    The recurrence carries both a deterministic and a stochastic path:

    .. math::

        h_t = f\big(h_{t-1},\, s_{t-1},\, a_{t-1}\big),
        \qquad
        s_t \sim p\big(s_t \mid h_t\big).

    Examples
    --------
    >>> from lucid.models.generative.planet import planet
    >>> model = planet(action_dim=6).eval()
    >>> model.config.stoch_size, model.config.deter_size
    (30, 200)
    """
    if pretrained:
        reject_unavailable_pretrained("planet")
    return PlaNetModel(apply_overrides(_CFG_PLANET, overrides))


# ── World-modeling head ──────────────────────────────────────────────────────


@register_model(
    task="world-modeling",
    family="planet",
    model_type="planet",
    model_class=PlaNetForWorldModeling,
    default_config=_CFG_PLANET,
)
def planet_world_model(pretrained: bool = False, **overrides: object) -> PlaNetForWorldModeling:
    r"""Construct PlaNet with its variational objective.

    Same trunk as :func:`planet`, wrapped with the bound of Hafner et al.,
    2019 — reconstruction and reward likelihoods against a KL between the
    posterior and the dynamics' own prior, floored at ``free_nats``.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`PlaNetConfig` field overrides.  ``free_nats`` and
        ``kl_weight`` shape the divergence term; ``action_dim`` is set by
        the environment.

    Returns
    -------
    PlaNetForWorldModeling
        The trunk plus the objective.

    Notes
    -----
    Reference: Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, and Davidson,
    *"Learning Latent Dynamics for Planning from Pixels"*, ICML, 2019
    (arXiv:1811.04551).

    Training objective:

    .. math::

        \mathcal{L} = \mathbb{E}_q\big[\log p(o_t \mid h_t, s_t)
                          + \log p(r_t \mid h_t, s_t)\big]
                      - \beta\,\mathrm{KL}\big(q(s_t \mid h_t, o_t)
                          \,\|\, p(s_t \mid h_t)\big).

    Examples
    --------
    >>> from lucid.models.generative.planet import planet_world_model
    >>> model = planet_world_model(action_dim=6).eval()
    >>> model.config.free_nats
    3.0
    """
    if pretrained:
        reject_unavailable_pretrained("planet_world_model")
    return PlaNetForWorldModeling(apply_overrides(_CFG_PLANET, overrides))
