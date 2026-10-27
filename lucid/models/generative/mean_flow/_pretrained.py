"""Registry factories for MeanFlow.

The four sizes the paper reports on ImageNet 256x256 (Table 2), plus the
ablation backbone it runs its own study on (Table 4's ``B/4``).  The
suffix is the patch side, so ``base_2`` is the paper's ``B/2``; H10
forbids the paper's own single-letter shorthand, which is why the size
is spelled out.

No CIFAR-10 factory ships.  The paper's unconditional CIFAR-10 result
uses a U-Net taken from another paper rather than this DiT backbone, and
gives its size only as "~55M" — a family whose parameter count cannot be
cited exactly is what H11 exists to keep out.  The objective and the
sampler here are the same ones; only the network differs.

The parameter counts registered below are the paper's, from Table 2.
Table 4 disagrees for ``M/2``, listing 497.8M against Table 2's 308M;
308M is the consistent one — the same backbone width as ``L/2`` at two
thirds of its depth should land near two thirds of its 459M, and it
does.
"""

from lucid.models._registry import register_model
from lucid.models.generative.mean_flow._config import MeanFlowConfig
from lucid.models.generative.mean_flow._model import MeanFlowForImageGeneration, MeanFlowModel
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

__all__ = [
    "mean_flow_base_4",
    "mean_flow_base_2",
    "mean_flow_medium_2",
    "mean_flow_large_2",
    "mean_flow_xlarge_2",
    "mean_flow_base_4_gen",
    "mean_flow_base_2_gen",
    "mean_flow_medium_2_gen",
    "mean_flow_large_2_gen",
    "mean_flow_xlarge_2_gen",
]

# Table 4.  Guidance differs per size: the two larger backbones use a
# stronger effective scale over a narrower band of ``t``, which is what
# the paper tunes rather than the architecture.
_CFG_BASE_4 = MeanFlowConfig(patch_size=4, hidden_size=768, depth=12, num_heads=12)
_CFG_BASE_2 = MeanFlowConfig(patch_size=2, hidden_size=768, depth=12, num_heads=12)
_CFG_MEDIUM_2 = MeanFlowConfig(patch_size=2, hidden_size=1024, depth=16, num_heads=16)
_CFG_LARGE_2 = MeanFlowConfig(
    patch_size=2,
    hidden_size=1024,
    depth=24,
    num_heads=16,
    guidance_scale=0.2,
    guidance_interval=(0.0, 0.8),
)
_CFG_XLARGE_2 = MeanFlowConfig(
    patch_size=2,
    hidden_size=1152,
    depth=28,
    num_heads=16,
    guidance_scale=0.2,
    guidance_interval=(0.0, 0.75),
)


@register_model(
    task="base",
    family="mean_flow",
    model_type="mean_flow",
    model_class=MeanFlowModel,
    default_config=_CFG_BASE_4,
    params=131_000_000,
    summary="auto",
)
def mean_flow_base_4(pretrained: bool = False, **overrides: object) -> MeanFlowModel:
    """The ablation backbone — Base width, patch 4.

    Parameters
    ----------
    pretrained : bool, default=False
        No MeanFlow checkpoint ships — the paper releases none — so
        ``True`` raises rather than returning a randomly initialised
        model.
    **overrides : object
        Optional :class:`MeanFlowConfig` field overrides.

    Returns
    -------
    MeanFlowModel
        The average-velocity network.

    Notes
    -----
    Reference: Geng, Deng, Bai, Kolter, and He, *"Mean Flows for One-step
    Generative Modeling"*, arXiv:2505.13447, 2025, Table 4.

    This is the configuration every ablation in Table 1 runs on, at 80
    epochs rather than the 240 the reported models get — its FID of 61.06
    is a comparison point between design choices, not a headline number.
    Patch 4 over a 32-wide latent leaves 64 tokens against ``B/2``'s 256,
    which is what makes a six-way study affordable.

    Examples
    --------
    >>> from lucid.models import mean_flow_base_4
    >>> config = mean_flow_base_4().config
    >>> config.patch_size, config.num_patches
    (4, 64)
    """
    if pretrained:
        reject_unavailable_pretrained("mean_flow_base_4")
    return MeanFlowModel(apply_overrides(_CFG_BASE_4, overrides))


@register_model(
    task="base",
    family="mean_flow",
    model_type="mean_flow",
    model_class=MeanFlowModel,
    default_config=_CFG_BASE_2,
    params=131_000_000,
    summary="auto",
)
def mean_flow_base_2(pretrained: bool = False, **overrides: object) -> MeanFlowModel:
    """Base width, patch 2 — the smallest reported model.

    Parameters
    ----------
    pretrained : bool, default=False
        No MeanFlow checkpoint ships — the paper releases none — so
        ``True`` raises rather than returning a randomly initialised
        model.
    **overrides : object
        Optional :class:`MeanFlowConfig` field overrides.

    Returns
    -------
    MeanFlowModel
        The average-velocity network.

    Notes
    -----
    Reference: Geng et al., arXiv:2505.13447, 2025, Tables 2 and 4.
    Reports 6.17 FID at one function evaluation on ImageNet 256x256.

    Examples
    --------
    >>> from lucid.models import mean_flow_base_2
    >>> config = mean_flow_base_2().config
    >>> config.hidden_size, config.depth, config.num_patches
    (768, 12, 256)
    """
    if pretrained:
        reject_unavailable_pretrained("mean_flow_base_2")
    return MeanFlowModel(apply_overrides(_CFG_BASE_2, overrides))


@register_model(
    task="base",
    family="mean_flow",
    model_type="mean_flow",
    model_class=MeanFlowModel,
    default_config=_CFG_MEDIUM_2,
    params=308_000_000,
    summary="auto",
)
def mean_flow_medium_2(pretrained: bool = False, **overrides: object) -> MeanFlowModel:
    """Medium width, patch 2.

    Parameters
    ----------
    pretrained : bool, default=False
        No MeanFlow checkpoint ships — the paper releases none — so
        ``True`` raises rather than returning a randomly initialised
        model.
    **overrides : object
        Optional :class:`MeanFlowConfig` field overrides.

    Returns
    -------
    MeanFlowModel
        The average-velocity network.

    Notes
    -----
    Reference: Geng et al., arXiv:2505.13447, 2025, Tables 2 and 4.
    Reports 5.01 FID at one function evaluation.

    A size DiT does not define — the paper introduces it to fill the gap
    between Base and Large, at Large's width and two thirds of its depth.

    Examples
    --------
    >>> from lucid.models import mean_flow_medium_2
    >>> config = mean_flow_medium_2().config
    >>> config.hidden_size, config.depth
    (1024, 16)
    """
    if pretrained:
        reject_unavailable_pretrained("mean_flow_medium_2")
    return MeanFlowModel(apply_overrides(_CFG_MEDIUM_2, overrides))


@register_model(
    task="base",
    family="mean_flow",
    model_type="mean_flow",
    model_class=MeanFlowModel,
    default_config=_CFG_LARGE_2,
    params=459_000_000,
    summary="auto",
)
def mean_flow_large_2(pretrained: bool = False, **overrides: object) -> MeanFlowModel:
    """Large width, patch 2.

    Parameters
    ----------
    pretrained : bool, default=False
        No MeanFlow checkpoint ships — the paper releases none — so
        ``True`` raises rather than returning a randomly initialised
        model.
    **overrides : object
        Optional :class:`MeanFlowConfig` field overrides.

    Returns
    -------
    MeanFlowModel
        The average-velocity network.

    Notes
    -----
    Reference: Geng et al., arXiv:2505.13447, 2025, Tables 2 and 4.
    Reports 3.84 FID at one function evaluation.

    The guidance defaults change here: the paper drops the mixing weight
    to 0.2 and stops applying guidance above ``t = 0.8``, which is a
    tuning of the target rather than of the network.

    Examples
    --------
    >>> from lucid.models import mean_flow_large_2
    >>> config = mean_flow_large_2().config
    >>> config.depth, config.guidance_interval
    (24, (0.0, 0.8))
    """
    if pretrained:
        reject_unavailable_pretrained("mean_flow_large_2")
    return MeanFlowModel(apply_overrides(_CFG_LARGE_2, overrides))


@register_model(
    task="base",
    family="mean_flow",
    model_type="mean_flow",
    model_class=MeanFlowModel,
    default_config=_CFG_XLARGE_2,
    params=676_000_000,
    summary="auto",
)
def mean_flow_xlarge_2(pretrained: bool = False, **overrides: object) -> MeanFlowModel:
    """Extra-large width, patch 2 — the paper's headline model.

    Parameters
    ----------
    pretrained : bool, default=False
        No MeanFlow checkpoint ships — the paper releases none — so
        ``True`` raises rather than returning a randomly initialised
        model.
    **overrides : object
        Optional :class:`MeanFlowConfig` field overrides.

    Returns
    -------
    MeanFlowModel
        The average-velocity network.

    Notes
    -----
    Reference: Geng et al., arXiv:2505.13447, 2025, Tables 2 and 4.
    Reports 3.43 FID at one function evaluation on ImageNet 256x256 and
    2.93 at two — the first figure being the one that closes most of the
    gap to 250-step diffusion models of the same backbone (DiT-XL/2 at
    2.27, SiT-XL/2 at 2.06).

    Examples
    --------
    >>> from lucid.models import mean_flow_xlarge_2
    >>> config = mean_flow_xlarge_2().config
    >>> config.hidden_size, config.depth, config.num_heads
    (1152, 28, 16)
    """
    if pretrained:
        reject_unavailable_pretrained("mean_flow_xlarge_2")
    return MeanFlowModel(apply_overrides(_CFG_XLARGE_2, overrides))


def _gen(cfg: MeanFlowConfig, overrides: dict[str, object]) -> MeanFlowForImageGeneration:
    return MeanFlowForImageGeneration(apply_overrides(cfg, overrides))


@register_model(
    task="image-generation",
    family="mean_flow",
    model_type="mean_flow",
    model_class=MeanFlowForImageGeneration,
    default_config=_CFG_BASE_4,
    params=131_000_000,
    summary="auto",
)
def mean_flow_base_4_gen(
    pretrained: bool = False, **overrides: object
) -> MeanFlowForImageGeneration:
    """:func:`mean_flow_base_4` posed as a sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No MeanFlow checkpoint ships — the paper releases none — so
        ``True`` raises rather than returning a randomly initialised
        model.
    **overrides : object
        Optional :class:`MeanFlowConfig` field overrides.

    Returns
    -------
    MeanFlowForImageGeneration
        Exposes the training objective and :meth:`generate`.

    Notes
    -----
    Reference: Geng et al., arXiv:2505.13447, 2025.

    Examples
    --------
    >>> from lucid.models import mean_flow_base_4_gen
    >>> model = mean_flow_base_4_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("mean_flow_base_4_gen")
    return _gen(_CFG_BASE_4, overrides)


@register_model(
    task="image-generation",
    family="mean_flow",
    model_type="mean_flow",
    model_class=MeanFlowForImageGeneration,
    default_config=_CFG_BASE_2,
    params=131_000_000,
    summary="auto",
)
def mean_flow_base_2_gen(
    pretrained: bool = False, **overrides: object
) -> MeanFlowForImageGeneration:
    """:func:`mean_flow_base_2` posed as a sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No MeanFlow checkpoint ships — the paper releases none — so
        ``True`` raises rather than returning a randomly initialised
        model.
    **overrides : object
        Optional :class:`MeanFlowConfig` field overrides.

    Returns
    -------
    MeanFlowForImageGeneration
        Exposes the training objective and :meth:`generate`.

    Notes
    -----
    Reference: Geng et al., arXiv:2505.13447, 2025.

    Examples
    --------
    >>> from lucid.models import mean_flow_base_2_gen
    >>> model = mean_flow_base_2_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("mean_flow_base_2_gen")
    return _gen(_CFG_BASE_2, overrides)


@register_model(
    task="image-generation",
    family="mean_flow",
    model_type="mean_flow",
    model_class=MeanFlowForImageGeneration,
    default_config=_CFG_MEDIUM_2,
    params=308_000_000,
    summary="auto",
)
def mean_flow_medium_2_gen(
    pretrained: bool = False, **overrides: object
) -> MeanFlowForImageGeneration:
    """:func:`mean_flow_medium_2` posed as a sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No MeanFlow checkpoint ships — the paper releases none — so
        ``True`` raises rather than returning a randomly initialised
        model.
    **overrides : object
        Optional :class:`MeanFlowConfig` field overrides.

    Returns
    -------
    MeanFlowForImageGeneration
        Exposes the training objective and :meth:`generate`.

    Notes
    -----
    Reference: Geng et al., arXiv:2505.13447, 2025.

    Examples
    --------
    >>> from lucid.models import mean_flow_medium_2_gen
    >>> model = mean_flow_medium_2_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("mean_flow_medium_2_gen")
    return _gen(_CFG_MEDIUM_2, overrides)


@register_model(
    task="image-generation",
    family="mean_flow",
    model_type="mean_flow",
    model_class=MeanFlowForImageGeneration,
    default_config=_CFG_LARGE_2,
    params=459_000_000,
    summary="auto",
)
def mean_flow_large_2_gen(
    pretrained: bool = False, **overrides: object
) -> MeanFlowForImageGeneration:
    """:func:`mean_flow_large_2` posed as a sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No MeanFlow checkpoint ships — the paper releases none — so
        ``True`` raises rather than returning a randomly initialised
        model.
    **overrides : object
        Optional :class:`MeanFlowConfig` field overrides.

    Returns
    -------
    MeanFlowForImageGeneration
        Exposes the training objective and :meth:`generate`.

    Notes
    -----
    Reference: Geng et al., arXiv:2505.13447, 2025.

    Examples
    --------
    >>> from lucid.models import mean_flow_large_2_gen
    >>> model = mean_flow_large_2_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("mean_flow_large_2_gen")
    return _gen(_CFG_LARGE_2, overrides)


@register_model(
    task="image-generation",
    family="mean_flow",
    model_type="mean_flow",
    model_class=MeanFlowForImageGeneration,
    default_config=_CFG_XLARGE_2,
    params=676_000_000,
    summary="auto",
)
def mean_flow_xlarge_2_gen(
    pretrained: bool = False, **overrides: object
) -> MeanFlowForImageGeneration:
    """:func:`mean_flow_xlarge_2` posed as a sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No MeanFlow checkpoint ships — the paper releases none — so
        ``True`` raises rather than returning a randomly initialised
        model.
    **overrides : object
        Optional :class:`MeanFlowConfig` field overrides.

    Returns
    -------
    MeanFlowForImageGeneration
        Exposes the training objective and :meth:`generate`.

    Notes
    -----
    Reference: Geng et al., arXiv:2505.13447, 2025.  This is the
    configuration behind the 3.43 one-step FID.

    Examples
    --------
    >>> from lucid.models import mean_flow_xlarge_2_gen
    >>> model = mean_flow_xlarge_2_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("mean_flow_xlarge_2_gen")
    return _gen(_CFG_XLARGE_2, overrides)
