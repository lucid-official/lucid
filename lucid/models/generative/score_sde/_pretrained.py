"""Registry factories for Score-SDE.

The paper's variants are the three stochastic differential equations, not
three networks — Table 3 reports NCSN++ under VE and DDPM++ under VP and
sub-VP, and the difference that makes a row is which process was trained
under.  So this family ships one factory per SDE.

No parameter count is registered: the size here is the backbone's, and
the backbone is DDPM's rather than the paper's NCSN++/DDPM++, so quoting
the paper's figures would attribute someone else's architecture to it.
"""

from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.generative.score_sde._config import ScoreSDEConfig
from lucid.models.generative.score_sde._model import (
    ScoreSDEForImageGeneration,
    ScoreSDEModel,
)

_CFG_VP = ScoreSDEConfig(sde_type="vp")
_CFG_VE = ScoreSDEConfig(sde_type="ve")
_CFG_SUBVP = ScoreSDEConfig(sde_type="subvp")


@register_model(
    task="base",
    family="score_sde",
    model_type="score_sde",
    model_class=ScoreSDEModel,
    default_config=_CFG_VP,
    summary="auto",
)
def score_sde_vp(pretrained: bool = False, **overrides: object) -> ScoreSDEModel:
    r"""Construct the Variance Preserving model — DDPM's continuous limit.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`ScoreSDEConfig` field overrides.

    Returns
    -------
    ScoreSDEModel
        Score network over the VP SDE.

    Notes
    -----
    Reference: Song et al., ICLR 2021 (arXiv:2011.13456), equation 32.

    Examples
    --------
    >>> from lucid.models import score_sde_vp
    >>> model = score_sde_vp().eval()
    >>> model.config.sde_type, model.config.beta_max
    ('vp', 20.0)
    """
    if pretrained:
        reject_unavailable_pretrained("score_sde_vp")
    return ScoreSDEModel(apply_overrides(_CFG_VP, overrides))


@register_model(
    task="image-generation",
    family="score_sde",
    model_type="score_sde",
    model_class=ScoreSDEForImageGeneration,
    default_config=_CFG_VP,
    summary="auto",
)
def score_sde_vp_gen(pretrained: bool = False, **overrides: object) -> ScoreSDEForImageGeneration:
    r"""The VP model with its objective and the three samplers.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`ScoreSDEConfig` field overrides.

    Returns
    -------
    ScoreSDEForImageGeneration
        Trainable, and able to sample by ``"euler"``, ``"pc"`` or
        ``"ode"``.

    Notes
    -----
    Reference: Song et al., ICLR 2021 (arXiv:2011.13456).

    Examples
    --------
    >>> from lucid.models import score_sde_vp_gen
    >>> model = score_sde_vp_gen().eval()
    >>> model.config.num_scales
    1000
    """
    if pretrained:
        reject_unavailable_pretrained("score_sde_vp_gen")
    return ScoreSDEForImageGeneration(apply_overrides(_CFG_VP, overrides))


@register_model(
    task="base",
    family="score_sde",
    model_type="score_sde",
    model_class=ScoreSDEModel,
    default_config=_CFG_VE,
    summary="auto",
)
def score_sde_ve(pretrained: bool = False, **overrides: object) -> ScoreSDEModel:
    r"""Construct the Variance Exploding model — NCSN's continuous limit.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`ScoreSDEConfig` field overrides.  ``sigma_max``
        is chosen per dataset; 50 is the paper's CIFAR-10 value.

    Returns
    -------
    ScoreSDEModel
        Score network over the VE SDE.

    Notes
    -----
    Reference: Song et al., ICLR 2021 (arXiv:2011.13456), equation 30.

    Examples
    --------
    >>> from lucid.models import score_sde_ve
    >>> model = score_sde_ve().eval()
    >>> model.config.sigma_min, model.config.sigma_max
    (0.01, 50.0)
    """
    if pretrained:
        reject_unavailable_pretrained("score_sde_ve")
    return ScoreSDEModel(apply_overrides(_CFG_VE, overrides))


@register_model(
    task="image-generation",
    family="score_sde",
    model_type="score_sde",
    model_class=ScoreSDEForImageGeneration,
    default_config=_CFG_VE,
    summary="auto",
)
def score_sde_ve_gen(pretrained: bool = False, **overrides: object) -> ScoreSDEForImageGeneration:
    r"""The VE model with its objective and the three samplers.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`ScoreSDEConfig` field overrides.

    Returns
    -------
    ScoreSDEForImageGeneration
        Trainable and samplable.

    Notes
    -----
    Reference: Song et al., ICLR 2021 (arXiv:2011.13456).

    The VE prior is wide — :math:`\mathcal{N}(0, \sigma_{\max}^2 I)` —
    so a sampler that starts from a standard normal will not work here,
    which is why the prior belongs to the SDE rather than the sampler.

    Examples
    --------
    >>> from lucid.models import score_sde_ve_gen
    >>> model = score_sde_ve_gen().eval()
    >>> model.sde.__class__.__name__
    'VESDE'
    """
    if pretrained:
        reject_unavailable_pretrained("score_sde_ve_gen")
    return ScoreSDEForImageGeneration(apply_overrides(_CFG_VE, overrides))


@register_model(
    task="base",
    family="score_sde",
    model_type="score_sde",
    model_class=ScoreSDEModel,
    default_config=_CFG_SUBVP,
    summary="auto",
)
def score_sde_subvp(pretrained: bool = False, **overrides: object) -> ScoreSDEModel:
    r"""Construct the sub-VP model — the SDE this paper introduces.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`ScoreSDEConfig` field overrides.

    Returns
    -------
    ScoreSDEModel
        Score network over the sub-VP SDE.

    Notes
    -----
    Reference: Song et al., ICLR 2021 (arXiv:2011.13456), equation 12.

    Its variance is bounded by the VP process at every intermediate step,
    and the paper's best likelihoods — 2.99 bits per dimension on
    CIFAR-10 — are reported here.

    Examples
    --------
    >>> from lucid.models import score_sde_subvp
    >>> model = score_sde_subvp().eval()
    >>> model.config.sde_type
    'subvp'
    """
    if pretrained:
        reject_unavailable_pretrained("score_sde_subvp")
    return ScoreSDEModel(apply_overrides(_CFG_SUBVP, overrides))


@register_model(
    task="image-generation",
    family="score_sde",
    model_type="score_sde",
    model_class=ScoreSDEForImageGeneration,
    default_config=_CFG_SUBVP,
    summary="auto",
)
def score_sde_subvp_gen(
    pretrained: bool = False, **overrides: object
) -> ScoreSDEForImageGeneration:
    r"""The sub-VP model with its objective and the three samplers.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`ScoreSDEConfig` field overrides.

    Returns
    -------
    ScoreSDEForImageGeneration
        Trainable and samplable.

    Notes
    -----
    Reference: Song et al., ICLR 2021 (arXiv:2011.13456).

    Examples
    --------
    >>> from lucid.models import score_sde_subvp_gen
    >>> model = score_sde_subvp_gen().eval()
    >>> model.config.sde_type
    'subvp'
    """
    if pretrained:
        reject_unavailable_pretrained("score_sde_subvp_gen")
    return ScoreSDEForImageGeneration(apply_overrides(_CFG_SUBVP, overrides))
