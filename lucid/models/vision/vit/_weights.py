"""Pretrained-weight declarations for the Vision Transformer family.

Four paper-cited variants (Dosovitskiy et al., ICLR 2021) ship the
official ImageNet-1k checkpoints converted from reference_vision's
``ViT_*_Weights.IMAGENET1K_V1`` tag: :class:`ViTBase16Weights`,
:class:`ViTBase32Weights`, :class:`ViTLarge16Weights`, and
:class:`ViTLarge32Weights`.

ViT-Huge/14 is intentionally not shipped: reference_vision only distributes
it as a 518x518 SWAG checkpoint whose ``pos_embedding`` token count does
not match Lucid's default ``image_size=224`` config, so there is no clean
224x224 1k-class checkpoint to host.

The B/16, B/32, and L/32 checkpoints all use the same eval pipeline (224
crop / 256 resize / bilinear / ImageNet stats); L/16 differs only in its
242-pixel resize (``crop_pct = 0.9667``).
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# Shared eval preset for B/16, B/32, L/32 (L/16 uses resize_size=242).
_PRESET = ImageClassification(crop_size=224, resize_size=256, interpolation="bilinear")


@register_weights("vit_base_16_cls")
class ViTBase16Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.vit_base_16_cls`.

    Single ImageNet-1k checkpoint converted from reference_vision's
    ``ViT_B_16_Weights.IMAGENET1K_V1`` (Dosovitskiy et al., 2021;
    ~86.6M params, 81.072% top-1).

    Examples
    --------
    >>> from lucid.models.vision.vit._weights import ViTBase16Weights
    >>> list(ViTBase16Weights.__members__)
    ['IMAGENET1K_V1', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> ViTBase16Weights.DEFAULT is ViTBase16Weights.IMAGENET1K_V1
    True
    >>> ViTBase16Weights.IMAGENET1K_V1.num_classes
    1000
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/vit-base-16/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="9d8c5987d9201082e19af85ab1f08f5cb2796aabda1d10dfed69647c3d07f840",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/ViT_B_16_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 86_567_656,
            "metrics": {"ImageNet-1k": {"acc@1": 81.072}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("vit_base_32_cls")
class ViTBase32Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.vit_base_32_cls`.

    Single ImageNet-1k checkpoint converted from reference_vision's
    ``ViT_B_32_Weights.IMAGENET1K_V1`` (Dosovitskiy et al., 2021;
    ~88.2M params, 75.912% top-1).

    Examples
    --------
    >>> from lucid.models.vision.vit._weights import ViTBase32Weights
    >>> list(ViTBase32Weights.__members__)
    ['IMAGENET1K_V1', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> ViTBase32Weights.DEFAULT is ViTBase32Weights.IMAGENET1K_V1
    True
    >>> ViTBase32Weights.IMAGENET1K_V1.num_classes
    1000
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/vit-base-32/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="999bfaddabc4ce62cb7adc100fcffec7a9d9e5f1abb04daa8355973c60cfbee8",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/ViT_B_32_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 88_224_232,
            "metrics": {"ImageNet-1k": {"acc@1": 75.912}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("vit_large_16_cls")
class ViTLarge16Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.vit_large_16_cls`.

    Single ImageNet-1k checkpoint converted from reference_vision's
    ``ViT_L_16_Weights.IMAGENET1K_V1`` (Dosovitskiy et al., 2021;
    ~304.3M params, 79.662% top-1).  Uses a 242-pixel resize before the
    224 center crop (vs. 256 for the other variants).

    Examples
    --------
    >>> from lucid.models.vision.vit._weights import ViTLarge16Weights
    >>> list(ViTLarge16Weights.__members__)
    ['IMAGENET1K_V1', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> ViTLarge16Weights.DEFAULT is ViTLarge16Weights.IMAGENET1K_V1
    True
    >>> ViTLarge16Weights.IMAGENET1K_V1.num_classes
    1000
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/vit-large-16/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="0c925d3c73c3cbe7129a71ddc8ff7ff97fc34a9d1be78f833cf8004f88859ec3",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=242, interpolation="bilinear"),
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/ViT_L_16_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 304_326_632,
            "metrics": {"ImageNet-1k": {"acc@1": 79.662}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("vit_large_32_cls")
class ViTLarge32Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.vit_large_32_cls`.

    Single ImageNet-1k checkpoint converted from reference_vision's
    ``ViT_L_32_Weights.IMAGENET1K_V1`` (Dosovitskiy et al., 2021;
    ~306.5M params, 76.972% top-1).

    Examples
    --------
    >>> from lucid.models.vision.vit._weights import ViTLarge32Weights
    >>> list(ViTLarge32Weights.__members__)
    ['IMAGENET1K_V1', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> ViTLarge32Weights.DEFAULT is ViTLarge32Weights.IMAGENET1K_V1
    True
    >>> ViTLarge32Weights.IMAGENET1K_V1.num_classes
    1000
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/vit-large-32/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="7566ba2f981294ede52fba94bfedf0d13dc00025f0e80c6bd63d1f13af642248",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "reference_vision/ViT_L_32_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 306_535_400,
            "metrics": {"ImageNet-1k": {"acc@1": 76.972}},
        },
    )
    DEFAULT = IMAGENET1K_V1
