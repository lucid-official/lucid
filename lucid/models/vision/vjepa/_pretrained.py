"""Registry factories for V-JEPA (the paper's three models).

Table 6 names exactly three, and the released configurations are the same
three: ViT-L/16 and ViT-H/16 at 224 pixels, and ViT-H/16 at 384.  Each
carries the same predictor — twelve blocks, 384 wide — because the
predictor's job does not grow with the encoder's.

The repository also defines ``vit_giant`` and ``vit_gigantic`` factories,
but no configuration uses them and no such V-JEPA model is reported; the
ViT-g rows in the paper's tables are other people's models.  They are not
registered here.

No weights ship from here.  Three checkpoints are published — 4.8 GB for
ViT-L/16 and 9.7 GB for each ViT-H — but the only licence in that
repository is CC BY-NC 4.0 and nothing states a separate one for the
weights, so nothing is redistributed.  ``params`` is left unregistered
for the same reason as I-JEPA: the paper's 200M and 630M describe one
encoder, while a factory here builds two of them and a predictor.
"""

from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.vision.vjepa._config import VJEPAConfig
from lucid.models.vision.vjepa._model import VJEPAForVideoClassification, VJEPAModel

__all__ = [
    "vjepa_large_16",
    "vjepa_large_16_cls",
    "vjepa_huge_16",
    "vjepa_huge_16_cls",
    "vjepa_huge_16_384",
    "vjepa_huge_16_384_cls",
]


_CFG_LARGE_16 = VJEPAConfig(image_size=224, dim=1024, depth=24, num_heads=16)
_CFG_HUGE_16 = VJEPAConfig(image_size=224, dim=1280, depth=32, num_heads=16)
_CFG_HUGE_16_384 = VJEPAConfig(image_size=384, dim=1280, depth=32, num_heads=16)


@register_model(
    task="base",
    family="vjepa",
    model_type="vjepa",
    model_class=VJEPAModel,
    default_config=_CFG_LARGE_16,
    summary="auto",
)
def vjepa_large_16(pretrained: bool = False, **overrides: object) -> VJEPAModel:
    r"""V-JEPA with a ViT-L/16 encoder at 224 pixels.

    Parameters
    ----------
    pretrained : bool, default=False
        The released checkpoints are not redistributed here; ``True``
        raises.
    **overrides : object
        Optional :class:`VJEPAConfig` field overrides.

    Returns
    -------
    VJEPAModel
        Context encoder, target encoder and predictor, untrained.

    Notes
    -----
    Reference: Bardes, Adrien, et al., *"Revisiting Feature Prediction for
    Learning Visual Representations from Video"*, arXiv:2404.08471, 2024,
    Table 6 — frozen with an attentive probe: 80.8% on Kinetics-400,
    69.5% on Something-Something-v2, 74.8% on ImageNet-1k.  The paper
    reports 200M parameters for this encoder.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("vjepa_large_16")
    >>> config.dim, config.depth, config.num_heads
    (1024, 24, 16)
    >>> config.token_grid, config.num_tokens
    ((8, 14, 14), 1568)
    """
    if pretrained:
        reject_unavailable_pretrained("vjepa_large_16")
    return VJEPAModel(apply_overrides(_CFG_LARGE_16, overrides))


@register_model(
    task="image-classification",
    family="vjepa",
    model_type="vjepa",
    model_class=VJEPAForVideoClassification,
    default_config=_CFG_LARGE_16,
    summary="auto",
)
def vjepa_large_16_cls(
    pretrained: bool = False, **overrides: object
) -> VJEPAForVideoClassification:
    r"""V-JEPA ViT-L/16 under the paper's attentive probe.

    Parameters
    ----------
    pretrained : bool, default=False
        Not redistributed; ``True`` raises.
    **overrides : object
        Optional :class:`VJEPAConfig` field overrides.

    Returns
    -------
    VJEPAForVideoClassification
        The pretraining networks, an attentive pooler and a classifier.

    Notes
    -----
    Reference: Bardes et al., arXiv:2404.08471, Section 4.3.  ``num_classes``
    defaults to 400, Kinetics-400's count.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> AutoConfig.from_pretrained("vjepa_large_16_cls").num_classes
    400
    """
    if pretrained:
        reject_unavailable_pretrained("vjepa_large_16_cls")
    return VJEPAForVideoClassification(apply_overrides(_CFG_LARGE_16, overrides))


@register_model(
    task="base",
    family="vjepa",
    model_type="vjepa",
    model_class=VJEPAModel,
    default_config=_CFG_HUGE_16,
    summary="auto",
)
def vjepa_huge_16(pretrained: bool = False, **overrides: object) -> VJEPAModel:
    r"""V-JEPA with a ViT-H/16 encoder at 224 pixels.

    Parameters
    ----------
    pretrained : bool, default=False
        The released checkpoints are not redistributed here; ``True``
        raises.
    **overrides : object
        Optional :class:`VJEPAConfig` field overrides.

    Returns
    -------
    VJEPAModel
        Context encoder, target encoder and predictor, untrained.

    Notes
    -----
    Reference: Bardes et al., arXiv:2404.08471, Table 6 — 82.0% on
    Kinetics-400 and 71.4% on Something-Something-v2, the paper's best at
    224 pixels.  630M parameters in the encoder.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("vjepa_huge_16")
    >>> config.dim, config.depth
    (1280, 32)
    >>> config.predictor_dim, config.predictor_depth
    (384, 12)
    """
    if pretrained:
        reject_unavailable_pretrained("vjepa_huge_16")
    return VJEPAModel(apply_overrides(_CFG_HUGE_16, overrides))


@register_model(
    task="image-classification",
    family="vjepa",
    model_type="vjepa",
    model_class=VJEPAForVideoClassification,
    default_config=_CFG_HUGE_16,
    summary="auto",
)
def vjepa_huge_16_cls(pretrained: bool = False, **overrides: object) -> VJEPAForVideoClassification:
    r"""V-JEPA ViT-H/16 under the paper's attentive probe.

    Parameters
    ----------
    pretrained : bool, default=False
        Not redistributed; ``True`` raises.
    **overrides : object
        Optional :class:`VJEPAConfig` field overrides.

    Returns
    -------
    VJEPAForVideoClassification
        The pretraining networks, an attentive pooler and a classifier.

    Notes
    -----
    Reference: Bardes et al., arXiv:2404.08471, Table 6.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> AutoConfig.from_pretrained("vjepa_huge_16_cls").dim
    1280
    """
    if pretrained:
        reject_unavailable_pretrained("vjepa_huge_16_cls")
    return VJEPAForVideoClassification(apply_overrides(_CFG_HUGE_16, overrides))


@register_model(
    task="base",
    family="vjepa",
    model_type="vjepa",
    model_class=VJEPAModel,
    default_config=_CFG_HUGE_16_384,
    summary="auto",
)
def vjepa_huge_16_384(pretrained: bool = False, **overrides: object) -> VJEPAModel:
    r"""V-JEPA with a ViT-H/16 encoder at 384 pixels.

    Parameters
    ----------
    pretrained : bool, default=False
        The released checkpoints are not redistributed here; ``True``
        raises.
    **overrides : object
        Optional :class:`VJEPAConfig` field overrides.

    Returns
    -------
    VJEPAModel
        Context encoder, target encoder and predictor, untrained.

    Notes
    -----
    Reference: Bardes et al., arXiv:2404.08471, Table 6 — 72.2% on
    Something-Something-v2 and 77.4% on ImageNet-1k, the paper's best on
    both.  Three times the tokens of the 224-pixel model: 4608 against
    1568.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("vjepa_huge_16_384")
    >>> config.image_size, config.num_tokens
    (384, 4608)
    """
    if pretrained:
        reject_unavailable_pretrained("vjepa_huge_16_384")
    return VJEPAModel(apply_overrides(_CFG_HUGE_16_384, overrides))


@register_model(
    task="image-classification",
    family="vjepa",
    model_type="vjepa",
    model_class=VJEPAForVideoClassification,
    default_config=_CFG_HUGE_16_384,
    summary="auto",
)
def vjepa_huge_16_384_cls(
    pretrained: bool = False, **overrides: object
) -> VJEPAForVideoClassification:
    r"""V-JEPA ViT-H/16 at 384 pixels under the paper's attentive probe.

    Parameters
    ----------
    pretrained : bool, default=False
        Not redistributed; ``True`` raises.
    **overrides : object
        Optional :class:`VJEPAConfig` field overrides.

    Returns
    -------
    VJEPAForVideoClassification
        The pretraining networks, an attentive pooler and a classifier.

    Notes
    -----
    Reference: Bardes et al., arXiv:2404.08471, Table 6.  The paper's
    77.9% on ImageNet-1k uses a two-layer probe; the 77.4% this one
    matches is the single-layer setting.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> AutoConfig.from_pretrained("vjepa_huge_16_384_cls").image_size
    384
    """
    if pretrained:
        reject_unavailable_pretrained("vjepa_huge_16_384_cls")
    return VJEPAForVideoClassification(apply_overrides(_CFG_HUGE_16_384, overrides))
