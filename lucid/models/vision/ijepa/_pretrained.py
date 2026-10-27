"""Registry factories for I-JEPA (the paper's Table 1 sizes).

Four encoders, each with a predictor whose width is 384 regardless —
Appendix A.1 keeps the predictor narrow at every scale, and Table 14
measures that choice against a 1024-wide one and prefers it.

The paper also reports a ViT-H/16 at 224 pixels (Table 11) and a ViT-g/16
trained on ImageNet-22k (Table 5).  Neither is in the main results table
and both would need a second set of claims to describe, so they are
overrides of the sizes below rather than factories.

No weights ship here.  Meta released four checkpoints — ViT-H/14 and
ViT-g/16 on ImageNet-22k, ViT-H/14 and ViT-H/16₄₄₈ on ImageNet-1k — but
they are ``.pth.tar`` archives under a repository whose only licence is
CC BY-NC 4.0, with no separate statement covering the weights, so nothing
is redistributed from here.  ViT-B/16 and ViT-L/16 were never released at
all, which is an open issue on that repository.
"""

from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.vision.ijepa._config import IJEPAConfig
from lucid.models.vision.ijepa._model import IJEPAForImageClassification, IJEPAModel

__all__ = [
    "ijepa_base_16",
    "ijepa_base_16_cls",
    "ijepa_large_16",
    "ijepa_large_16_cls",
    "ijepa_huge_14",
    "ijepa_huge_14_cls",
    "ijepa_huge_16_448",
    "ijepa_huge_16_448_cls",
]


# Table 1's four rows.  Depths, widths and head counts are ViT's own; the
# predictor depth is Appendix A.1 — 6 for ViT-B/16, 12 for everything
# larger.
_CFG_BASE_16 = IJEPAConfig(
    image_size=224, patch_size=16, dim=768, depth=12, num_heads=12, predictor_depth=6
)
_CFG_LARGE_16 = IJEPAConfig(
    image_size=224, patch_size=16, dim=1024, depth=24, num_heads=16, predictor_depth=12
)
_CFG_HUGE_14 = IJEPAConfig(
    image_size=224, patch_size=14, dim=1280, depth=32, num_heads=16, predictor_depth=12
)
_CFG_HUGE_16_448 = IJEPAConfig(
    image_size=448, patch_size=16, dim=1280, depth=32, num_heads=16, predictor_depth=12
)


@register_model(
    task="base",
    family="ijepa",
    model_type="ijepa",
    model_class=IJEPAModel,
    default_config=_CFG_BASE_16,
    summary="auto",
)
def ijepa_base_16(pretrained: bool = False, **overrides: object) -> IJEPAModel:
    r"""I-JEPA with a ViT-B/16 encoder.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this size; ``True`` raises.
    **overrides : object
        Optional :class:`IJEPAConfig` field overrides.

    Returns
    -------
    IJEPAModel
        Context encoder, target encoder and predictor, untrained.

    Notes
    -----
    Reference: Assran, Mahmoud, et al., *"Self-Supervised Learning from
    Images with a Joint-Embedding Predictive Architecture"*, CVPR 2023
    (arXiv:2301.08243), Table 1 — 72.9% ImageNet-1k linear probe after 600
    epochs.  Its predictor is 6 blocks deep, the shallowest of the four.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("ijepa_base_16")
    >>> config.dim, config.depth, config.num_heads, config.predictor_depth
    (768, 12, 12, 6)
    >>> config.num_patches
    196
    """
    if pretrained:
        reject_unavailable_pretrained("ijepa_base_16")
    return IJEPAModel(apply_overrides(_CFG_BASE_16, overrides))


@register_model(
    task="image-classification",
    family="ijepa",
    model_type="ijepa",
    model_class=IJEPAForImageClassification,
    default_config=_CFG_BASE_16,
    summary="auto",
)
def ijepa_base_16_cls(pretrained: bool = False, **overrides: object) -> IJEPAForImageClassification:
    r"""A linear probe on I-JEPA ViT-B/16.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this size; ``True`` raises.
    **overrides : object
        Optional :class:`IJEPAConfig` field overrides.

    Returns
    -------
    IJEPAForImageClassification
        The pretraining networks plus a linear head over the frozen
        target encoder.

    Notes
    -----
    Reference: Assran et al., arXiv:2301.08243, Appendix A.2 — the paper's
    evaluation protocol is a linear probe on the average-pooled target
    encoder.

    Examples
    --------
    >>> from lucid.models import ijepa_base_16_cls
    >>> model = ijepa_base_16_cls(
    ...     image_size=32, patch_size=8, dim=16, depth=1, num_heads=2,
    ...     predictor_dim=8, predictor_depth=1, num_classes=10, min_keep=1)
    >>> model.config.num_classes
    10
    """
    if pretrained:
        reject_unavailable_pretrained("ijepa_base_16_cls")
    return IJEPAForImageClassification(apply_overrides(_CFG_BASE_16, overrides))


@register_model(
    task="base",
    family="ijepa",
    model_type="ijepa",
    model_class=IJEPAModel,
    default_config=_CFG_LARGE_16,
    summary="auto",
)
def ijepa_large_16(pretrained: bool = False, **overrides: object) -> IJEPAModel:
    r"""I-JEPA with a ViT-L/16 encoder.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this size; ``True`` raises.
    **overrides : object
        Optional :class:`IJEPAConfig` field overrides.

    Returns
    -------
    IJEPAModel
        Context encoder, target encoder and predictor, untrained.

    Notes
    -----
    Reference: Assran et al., arXiv:2301.08243, Table 1 — 77.5% ImageNet-1k
    linear probe after 600 epochs, and 69.4% on 1% of ImageNet (Table 2).
    Table 13 reports 77.8% for the same model under the weight-decay
    schedule it prefers, which is the paper disagreeing with itself by 0.3.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("ijepa_large_16")
    >>> config.dim, config.depth, config.num_heads
    (1024, 24, 16)
    """
    if pretrained:
        reject_unavailable_pretrained("ijepa_large_16")
    return IJEPAModel(apply_overrides(_CFG_LARGE_16, overrides))


@register_model(
    task="image-classification",
    family="ijepa",
    model_type="ijepa",
    model_class=IJEPAForImageClassification,
    default_config=_CFG_LARGE_16,
    summary="auto",
)
def ijepa_large_16_cls(
    pretrained: bool = False, **overrides: object
) -> IJEPAForImageClassification:
    r"""A linear probe on I-JEPA ViT-L/16.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this size; ``True`` raises.
    **overrides : object
        Optional :class:`IJEPAConfig` field overrides.

    Returns
    -------
    IJEPAForImageClassification
        The pretraining networks plus a linear head.

    Notes
    -----
    Reference: Assran et al., arXiv:2301.08243, Table 1.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> AutoConfig.from_pretrained("ijepa_large_16_cls").predictor_depth
    12
    """
    if pretrained:
        reject_unavailable_pretrained("ijepa_large_16_cls")
    return IJEPAForImageClassification(apply_overrides(_CFG_LARGE_16, overrides))


@register_model(
    task="base",
    family="ijepa",
    model_type="ijepa",
    model_class=IJEPAModel,
    default_config=_CFG_HUGE_14,
    summary="auto",
)
def ijepa_huge_14(pretrained: bool = False, **overrides: object) -> IJEPAModel:
    r"""I-JEPA with a ViT-H/14 encoder.

    Parameters
    ----------
    pretrained : bool, default=False
        The released ``.pth.tar`` checkpoints are not redistributed here;
        ``True`` raises.
    **overrides : object
        Optional :class:`IJEPAConfig` field overrides.

    Returns
    -------
    IJEPAModel
        Context encoder, target encoder and predictor, untrained.

    Notes
    -----
    Reference: Assran et al., arXiv:2301.08243, Table 1 — 79.3% ImageNet-1k
    linear probe after 300 epochs, 73.3% on 1% of ImageNet (Table 2), and
    the transfer results of Tables 3 and 4 (CIFAR-100 87.5, Places205
    58.4, Clevr/Count 86.7).

    A 14-pixel patch over 224 pixels is a 16×16 grid, the same token count
    as the /16 models at that resolution.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("ijepa_huge_14")
    >>> config.patch_size, config.grid_size, config.num_patches
    (14, 16, 256)
    """
    if pretrained:
        reject_unavailable_pretrained("ijepa_huge_14")
    return IJEPAModel(apply_overrides(_CFG_HUGE_14, overrides))


@register_model(
    task="image-classification",
    family="ijepa",
    model_type="ijepa",
    model_class=IJEPAForImageClassification,
    default_config=_CFG_HUGE_14,
    summary="auto",
)
def ijepa_huge_14_cls(pretrained: bool = False, **overrides: object) -> IJEPAForImageClassification:
    r"""A linear probe on I-JEPA ViT-H/14.

    Parameters
    ----------
    pretrained : bool, default=False
        Not redistributed; ``True`` raises.
    **overrides : object
        Optional :class:`IJEPAConfig` field overrides.

    Returns
    -------
    IJEPAForImageClassification
        The pretraining networks plus a linear head.

    Notes
    -----
    Reference: Assran et al., arXiv:2301.08243, Table 1.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> AutoConfig.from_pretrained("ijepa_huge_14_cls").dim
    1280
    """
    if pretrained:
        reject_unavailable_pretrained("ijepa_huge_14_cls")
    return IJEPAForImageClassification(apply_overrides(_CFG_HUGE_14, overrides))


@register_model(
    task="base",
    family="ijepa",
    model_type="ijepa",
    model_class=IJEPAModel,
    default_config=_CFG_HUGE_16_448,
    summary="auto",
)
def ijepa_huge_16_448(pretrained: bool = False, **overrides: object) -> IJEPAModel:
    r"""I-JEPA with a ViT-H/16 encoder at 448 pixels.

    Parameters
    ----------
    pretrained : bool, default=False
        The released ``.pth.tar`` checkpoint is not redistributed here;
        ``True`` raises.
    **overrides : object
        Optional :class:`IJEPAConfig` field overrides.

    Returns
    -------
    IJEPAModel
        Context encoder, target encoder and predictor, untrained.

    Notes
    -----
    Reference: Assran et al., arXiv:2301.08243, Table 1 — 81.1% ImageNet-1k
    linear probe, the paper's best, and 77.3% on 1% of ImageNet (Table 2);
    87.1% when fine-tuned on all of it (Table 15).

    Four times the pixels is four times the tokens: 784 patches against
    196, which is where the accuracy comes from and what it costs.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("ijepa_huge_16_448")
    >>> config.image_size, config.num_patches
    (448, 784)
    """
    if pretrained:
        reject_unavailable_pretrained("ijepa_huge_16_448")
    return IJEPAModel(apply_overrides(_CFG_HUGE_16_448, overrides))


@register_model(
    task="image-classification",
    family="ijepa",
    model_type="ijepa",
    model_class=IJEPAForImageClassification,
    default_config=_CFG_HUGE_16_448,
    summary="auto",
)
def ijepa_huge_16_448_cls(
    pretrained: bool = False, **overrides: object
) -> IJEPAForImageClassification:
    r"""A linear probe on I-JEPA ViT-H/16 at 448 pixels.

    Parameters
    ----------
    pretrained : bool, default=False
        Not redistributed; ``True`` raises.
    **overrides : object
        Optional :class:`IJEPAConfig` field overrides.

    Returns
    -------
    IJEPAForImageClassification
        The pretraining networks plus a linear head.

    Notes
    -----
    Reference: Assran et al., arXiv:2301.08243, Table 1 and Appendix A.2.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> AutoConfig.from_pretrained("ijepa_huge_16_448_cls").image_size
    448
    """
    if pretrained:
        reject_unavailable_pretrained("ijepa_huge_16_448_cls")
    return IJEPAForImageClassification(apply_overrides(_CFG_HUGE_16_448, overrides))
