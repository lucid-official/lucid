"""Registry factories for CLIP.

The paper trains eight models: five with modified-ResNet image towers
(RN50, RN101, RN50x4, RN50x16, RN50x64) and three with Vision
Transformers (ViT-B/32, ViT-B/16, ViT-L/14), plus a fourth ViT obtained
by fine-tuning ViT-L/14 at 336 pixels for one epoch.

**The ResNet towers are not implemented here**, and the omission is
deliberate rather than pending: they are not the ResNet this zoo already
has.  The paper replaces the stem with three convolutions, swaps every
stride-2 convolution for an anti-aliased blur-pooled one, and replaces
global average pooling with a single transformer-style attention pooling
layer whose query is conditioned on the average-pooled vector.  That is
a third ResNet variant, not a configuration of an existing one, and
shipping it half-done would be worse than saying so.  The four ViT
factories below are complete.

All four ship pretrained weights, verified numerically against the
reference before publication rather than merely loaded — see
``_weights.py``.

No parameter counts are registered.  The paper reports compute (in
GPU-days and GFLOPs) rather than trainable-parameter totals for its
variants — the only count it states is the text encoder's 63M — so any
per-variant figure here would be an introspection result dressed up as a
citation.  ``summary="auto"`` reports the real number instead.
"""

from dataclasses import replace

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.multimodal.clip._config import CLIPConfig
from lucid.models.multimodal.clip._weights import (
    CLIPViTBase16Weights,
    CLIPViTBase32Weights,
    CLIPViTLarge14_336Weights,
    CLIPViTLarge14Weights,
)
from lucid.models.multimodal.clip._model import CLIPModel, CLIPForZeroShotImageClassification
from lucid.models._utils._common import apply_overrides

__all__ = [
    "clip_vit_base_32",
    "clip_vit_base_16",
    "clip_vit_large_14",
    "clip_vit_large_14_336",
    "clip_vit_base_32_zero_shot",
    "clip_vit_base_16_zero_shot",
    "clip_vit_large_14_zero_shot",
    "clip_vit_large_14_336_zero_shot",
]

# Widths and depths as released.  The text tower widens with the image
# tower for ViT-L/14 (768/12 rather than 512/8) and stays 12 deep
# everywhere, which is the paper's stated scaling rule.
_CFG_BASE_32 = CLIPConfig(
    embed_dim=512,
    image_size=224,
    patch_size=32,
    vision_layers=12,
    vision_width=768,
    vision_heads=12,
    text_width=512,
    text_heads=8,
    text_layers=12,
)
_CFG_BASE_16 = replace(_CFG_BASE_32, patch_size=16)
_CFG_LARGE_14 = CLIPConfig(
    embed_dim=768,
    image_size=224,
    patch_size=14,
    vision_layers=24,
    vision_width=1024,
    vision_heads=16,
    text_width=768,
    text_heads=12,
    text_layers=12,
)
_CFG_LARGE_14_336 = replace(_CFG_LARGE_14, image_size=336)


@register_model(
    task="base",
    family="clip",
    model_type="clip",
    model_class=CLIPModel,
    default_config=_CFG_BASE_32,
    summary="auto",
)
def clip_vit_base_32(pretrained: bool = False, **overrides: object) -> CLIPModel:
    """Construct CLIP with a ViT-B/32 image tower.

    Parameters
    ----------
    pretrained : bool, default=False
        Load the OpenAI WIT-400M checkpoint from the Lucid hub.
    **overrides : object
        Optional :class:`CLIPConfig` field overrides.

    Returns
    -------
    CLIPModel
        Both towers and the learned temperature.

    Notes
    -----
    Reference: Radford et al., ICML 2021 (arXiv:2103.00020), §2.4.

    The cheapest of the ViT variants: 32-pixel patches leave 49 patch
    tokens at 224 pixels, against ViT-B/16's 196, so it costs roughly a
    quarter of the attention for the same depth and width.

    Examples
    --------
    >>> from lucid.models import clip_vit_base_32
    >>> model = clip_vit_base_32()
    >>> model.config.patch_size, model.config.embed_dim
    (32, 512)
    """
    model = CLIPModel(apply_overrides(_CFG_BASE_32, overrides))
    entry = weights_mod.resolve_weights(CLIPViTBase32Weights, pretrained, None)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="clip_vit_base_32")
    return model


@register_model(
    task="base",
    family="clip",
    model_type="clip",
    model_class=CLIPModel,
    default_config=_CFG_BASE_16,
    summary="auto",
)
def clip_vit_base_16(pretrained: bool = False, **overrides: object) -> CLIPModel:
    """Construct CLIP with a ViT-B/16 image tower.

    Parameters
    ----------
    pretrained : bool, default=False
        Load the OpenAI WIT-400M checkpoint from the Lucid hub.
    **overrides : object
        Optional :class:`CLIPConfig` field overrides.

    Returns
    -------
    CLIPModel
        Both towers and the learned temperature.

    Notes
    -----
    Reference: Radford et al., ICML 2021 (arXiv:2103.00020), §2.4.

    Identical to ViT-B/32 apart from the patch size, which alone
    quadruples the token count — the paper's evidence that resolution of
    the tokenisation, not width, is what the image tower was starved of.

    Examples
    --------
    >>> from lucid.models import clip_vit_base_16
    >>> model = clip_vit_base_16()
    >>> model.config.patch_size, model.config.vision_width
    (16, 768)
    """
    model = CLIPModel(apply_overrides(_CFG_BASE_16, overrides))
    entry = weights_mod.resolve_weights(CLIPViTBase16Weights, pretrained, None)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="clip_vit_base_16")
    return model


@register_model(
    task="base",
    family="clip",
    model_type="clip",
    model_class=CLIPModel,
    default_config=_CFG_LARGE_14,
    summary="auto",
)
def clip_vit_large_14(pretrained: bool = False, **overrides: object) -> CLIPModel:
    """Construct CLIP with a ViT-L/14 image tower.

    Parameters
    ----------
    pretrained : bool, default=False
        Load the OpenAI WIT-400M checkpoint from the Lucid hub.
    **overrides : object
        Optional :class:`CLIPConfig` field overrides.

    Returns
    -------
    CLIPModel
        Both towers and the learned temperature.

    Notes
    -----
    Reference: Radford et al., ICML 2021 (arXiv:2103.00020), §2.4.

    The best-performing variant in the paper, and the one whose text
    tower is widened along with the image tower — 768 wide with 12 heads
    against the base models' 512 and 8, at the same 12 layers.

    Examples
    --------
    >>> from lucid.models import clip_vit_large_14
    >>> model = clip_vit_large_14()
    >>> model.config.vision_layers, model.config.text_width
    (24, 768)
    """
    model = CLIPModel(apply_overrides(_CFG_LARGE_14, overrides))
    entry = weights_mod.resolve_weights(CLIPViTLarge14Weights, pretrained, None)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="clip_vit_large_14")
    return model


@register_model(
    task="base",
    family="clip",
    model_type="clip",
    model_class=CLIPModel,
    default_config=_CFG_LARGE_14_336,
    summary="auto",
)
def clip_vit_large_14_336(pretrained: bool = False, **overrides: object) -> CLIPModel:
    """Construct CLIP with a ViT-L/14 image tower at 336 pixels.

    Parameters
    ----------
    pretrained : bool, default=False
        Load the OpenAI WIT-400M checkpoint from the Lucid hub.
    **overrides : object
        Optional :class:`CLIPConfig` field overrides.

    Returns
    -------
    CLIPModel
        Both towers and the learned temperature.

    Notes
    -----
    Reference: Radford et al., ICML 2021 (arXiv:2103.00020), §2.5 — "for
    the ViT-L/14 we also pre-train at a higher 336 pixel resolution for
    one additional epoch", denoted ViT-L/14@336px.

    Architecturally this differs from :func:`clip_vit_large_14` in one
    number, but that number changes the positional table from 257 rows to
    577, so the two are not weight-compatible.

    Examples
    --------
    >>> from lucid.models import clip_vit_large_14_336
    >>> model = clip_vit_large_14_336()
    >>> model.config.image_size
    336
    """
    model = CLIPModel(apply_overrides(_CFG_LARGE_14_336, overrides))
    entry = weights_mod.resolve_weights(CLIPViTLarge14_336Weights, pretrained, None)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="clip_vit_large_14_336")
    return model


@register_model(
    task="image-classification",
    family="clip",
    model_type="clip",
    model_class=CLIPForZeroShotImageClassification,
    default_config=_CFG_BASE_32,
    summary="auto",
)
def clip_vit_base_32_zero_shot(
    pretrained: bool = False, **overrides: object
) -> CLIPForZeroShotImageClassification:
    """ViT-B/32 CLIP posed as an open-vocabulary classifier.

    Parameters
    ----------
    pretrained : bool, default=False
        Load the OpenAI WIT-400M checkpoint from the Lucid hub.
    **overrides : object
        Optional :class:`CLIPConfig` field overrides.

    Returns
    -------
    CLIPForZeroShotImageClassification
        Scores images against tokenised prompts.

    Notes
    -----
    Reference: Radford et al., ICML 2021 (arXiv:2103.00020), §3.1.

    Examples
    --------
    >>> from lucid.models import clip_vit_base_32_zero_shot
    >>> model = clip_vit_base_32_zero_shot()
    >>> model.config.patch_size
    32
    """
    model = CLIPForZeroShotImageClassification(apply_overrides(_CFG_BASE_32, overrides))
    entry = weights_mod.resolve_weights(CLIPViTBase32Weights, pretrained, None)
    if entry is not None:
        weights_mod.load_weight_entry(model.clip, entry, name="clip_vit_base_32_zero_shot")
    return model


@register_model(
    task="image-classification",
    family="clip",
    model_type="clip",
    model_class=CLIPForZeroShotImageClassification,
    default_config=_CFG_BASE_16,
    summary="auto",
)
def clip_vit_base_16_zero_shot(
    pretrained: bool = False, **overrides: object
) -> CLIPForZeroShotImageClassification:
    """ViT-B/16 CLIP posed as an open-vocabulary classifier.

    Parameters
    ----------
    pretrained : bool, default=False
        Load the OpenAI WIT-400M checkpoint from the Lucid hub.
    **overrides : object
        Optional :class:`CLIPConfig` field overrides.

    Returns
    -------
    CLIPForZeroShotImageClassification
        Scores images against tokenised prompts.

    Notes
    -----
    Reference: Radford et al., ICML 2021 (arXiv:2103.00020), §3.1.

    Examples
    --------
    >>> from lucid.models import clip_vit_base_16_zero_shot
    >>> model = clip_vit_base_16_zero_shot()
    >>> model.config.patch_size
    16
    """
    model = CLIPForZeroShotImageClassification(apply_overrides(_CFG_BASE_16, overrides))
    entry = weights_mod.resolve_weights(CLIPViTBase16Weights, pretrained, None)
    if entry is not None:
        weights_mod.load_weight_entry(model.clip, entry, name="clip_vit_base_16_zero_shot")
    return model


@register_model(
    task="image-classification",
    family="clip",
    model_type="clip",
    model_class=CLIPForZeroShotImageClassification,
    default_config=_CFG_LARGE_14,
    summary="auto",
)
def clip_vit_large_14_zero_shot(
    pretrained: bool = False, **overrides: object
) -> CLIPForZeroShotImageClassification:
    """ViT-L/14 CLIP posed as an open-vocabulary classifier.

    Parameters
    ----------
    pretrained : bool, default=False
        Load the OpenAI WIT-400M checkpoint from the Lucid hub.
    **overrides : object
        Optional :class:`CLIPConfig` field overrides.

    Returns
    -------
    CLIPForZeroShotImageClassification
        Scores images against tokenised prompts.

    Notes
    -----
    Reference: Radford et al., ICML 2021 (arXiv:2103.00020), Table 11 —
    the variant the paper's headline zero-shot ImageNet number comes
    from.

    Examples
    --------
    >>> from lucid.models import clip_vit_large_14_zero_shot
    >>> model = clip_vit_large_14_zero_shot()
    >>> model.config.embed_dim
    768
    """
    model = CLIPForZeroShotImageClassification(apply_overrides(_CFG_LARGE_14, overrides))
    entry = weights_mod.resolve_weights(CLIPViTLarge14Weights, pretrained, None)
    if entry is not None:
        weights_mod.load_weight_entry(model.clip, entry, name="clip_vit_large_14_zero_shot")
    return model


@register_model(
    task="image-classification",
    family="clip",
    model_type="clip",
    model_class=CLIPForZeroShotImageClassification,
    default_config=_CFG_LARGE_14_336,
    summary="auto",
)
def clip_vit_large_14_336_zero_shot(
    pretrained: bool = False, **overrides: object
) -> CLIPForZeroShotImageClassification:
    """ViT-L/14@336px CLIP posed as an open-vocabulary classifier.

    Parameters
    ----------
    pretrained : bool, default=False
        Load the OpenAI WIT-400M checkpoint from the Lucid hub.
    **overrides : object
        Optional :class:`CLIPConfig` field overrides.

    Returns
    -------
    CLIPForZeroShotImageClassification
        Scores images against tokenised prompts.

    Notes
    -----
    Reference: Radford et al., ICML 2021 (arXiv:2103.00020), §2.5.

    Examples
    --------
    >>> from lucid.models import clip_vit_large_14_336_zero_shot
    >>> model = clip_vit_large_14_336_zero_shot()
    >>> model.config.image_size
    336
    """
    model = CLIPForZeroShotImageClassification(apply_overrides(_CFG_LARGE_14_336, overrides))
    entry = weights_mod.resolve_weights(CLIPViTLarge14_336Weights, pretrained, None)
    if entry is not None:
        weights_mod.load_weight_entry(model.clip, entry, name="clip_vit_large_14_336_zero_shot")
    return model
