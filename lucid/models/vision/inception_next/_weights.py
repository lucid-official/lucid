"""Pretrained-weight declarations for the InceptionNeXt family.

Per-model weight enums live *with the model* (not in
:mod:`lucid.weights`).  :mod:`lucid.weights` is purely the porting
infrastructure — base classes, transforms, hub download, loading, and
the discovery registry; it knows nothing about specific architectures.
Each model family declares its checkpoints here by importing those
primitives.  Importing this module (which happens when
:mod:`lucid.models.vision.inception_next` loads) registers the enums
with the discovery registry (:func:`lucid.weights.list_pretrained` /
:func:`lucid.weights.get_weight`).

Three paper-cited variants ship ImageNet-1k checkpoints converted from
timm's ``sail_in1k`` weights (Yu et al., "InceptionNeXt: When Inception
Meets ConvNeXt", CVPR 2024 — :class:`InceptionNeXtTinyWeights`,
:class:`InceptionNeXtSmallWeights`, :class:`InceptionNeXtBaseWeights`).
Tiny / Small use the standard 224-crop / 256-resize bicubic pipeline
(``crop_pct=0.875``); Base uses a tighter 224-crop / 235-resize bicubic
pipeline (``crop_pct=0.95``), all with ImageNet stats.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# Tiny / Small: crop_pct=0.875 → resize 256, crop 224.
_PRESET_875 = ImageClassification(crop_size=224, resize_size=256, interpolation="bicubic")
# Base: crop_pct=0.95 → resize 235, crop 224.  timm truncates 224 / 0.95 =
# 235.8, and rounding it to 236 moved the crop window by a pixel.
_PRESET_95 = ImageClassification(crop_size=224, resize_size=235, interpolation="bicubic")


@register_weights("inception_next_tiny_cls")
class InceptionNeXtTinyWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.inception_next_tiny_cls`.

    Ships a single ImageNet-1k checkpoint (:attr:`SAIL_IN1K`) converted
    from timm's ``inception_next_tiny.sail_in1k`` weights and re-hosted
    under ``huggingface.co/lucid-dl/inception-next-tiny``.

    Examples
    --------
    >>> from lucid.models.vision.inception_next._weights import InceptionNeXtTinyWeights
    >>> list(InceptionNeXtTinyWeights.__members__)
    ['SAIL_IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> InceptionNeXtTinyWeights.DEFAULT is InceptionNeXtTinyWeights.SAIL_IN1K
    True
    >>> InceptionNeXtTinyWeights.SAIL_IN1K.num_classes
    1000
    """

    SAIL_IN1K = WeightEntry(
        url=f"{HUB_BASE}/inception-next-tiny/resolve/main/SAIL_IN1K/model.safetensors",
        sha256="16cd250ce8c28a748f5e399d0cb300a4eb417065f379d70fbdd5c54cabee1052",
        num_classes=1000,
        transforms=_PRESET_875,
        meta={
            "tag": "SAIL_IN1K",
            "source": "timm/inception_next_tiny.sail_in1k",
            "license": "apache-2.0",
            "num_params": 28_055_680,
            "metrics": {"ImageNet-1k": {"acc@1": 82.3, "acc@5": 96.1}},
        },
    )
    DEFAULT = SAIL_IN1K


@register_weights("inception_next_small_cls")
class InceptionNeXtSmallWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.inception_next_small_cls`.

    Ships a single ImageNet-1k checkpoint (:attr:`SAIL_IN1K`) converted
    from timm's ``inception_next_small.sail_in1k`` weights and re-hosted
    under ``huggingface.co/lucid-dl/inception-next-small``.

    Examples
    --------
    >>> from lucid.models.vision.inception_next._weights import InceptionNeXtSmallWeights
    >>> list(InceptionNeXtSmallWeights.__members__)
    ['SAIL_IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> InceptionNeXtSmallWeights.DEFAULT is InceptionNeXtSmallWeights.SAIL_IN1K
    True
    >>> InceptionNeXtSmallWeights.SAIL_IN1K.num_classes
    1000
    """

    SAIL_IN1K = WeightEntry(
        url=f"{HUB_BASE}/inception-next-small/resolve/main/SAIL_IN1K/model.safetensors",
        sha256="229eb279ab96ab602ba129535711c4843013adfe6da22eac28e7b00766eb6b70",
        num_classes=1000,
        transforms=_PRESET_875,
        meta={
            "tag": "SAIL_IN1K",
            "source": "timm/inception_next_small.sail_in1k",
            "license": "apache-2.0",
            "num_params": 49_374_016,
            "metrics": {"ImageNet-1k": {"acc@1": 83.5, "acc@5": 96.6}},
        },
    )
    DEFAULT = SAIL_IN1K


@register_weights("inception_next_base_cls")
class InceptionNeXtBaseWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.inception_next_base_cls`.

    Ships a single ImageNet-1k checkpoint (:attr:`SAIL_IN1K`) converted
    from timm's ``inception_next_base.sail_in1k`` weights and re-hosted
    under ``huggingface.co/lucid-dl/inception-next-base``.  Uses a
    tighter ``crop_pct=0.95`` eval pipeline (235 resize / 224 crop).

    Examples
    --------
    >>> from lucid.models.vision.inception_next._weights import InceptionNeXtBaseWeights
    >>> list(InceptionNeXtBaseWeights.__members__)
    ['SAIL_IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> InceptionNeXtBaseWeights.DEFAULT is InceptionNeXtBaseWeights.SAIL_IN1K
    True
    >>> InceptionNeXtBaseWeights.SAIL_IN1K.num_classes
    1000
    """

    SAIL_IN1K = WeightEntry(
        url=f"{HUB_BASE}/inception-next-base/resolve/main/SAIL_IN1K/model.safetensors",
        sha256="3928b2fbd8f25bf575c7506586b17340abc6ae9227b9b44ada4cab0d9c61d6b5",
        num_classes=1000,
        transforms=_PRESET_95,
        meta={
            "tag": "SAIL_IN1K",
            "source": "timm/inception_next_base.sail_in1k",
            "license": "apache-2.0",
            "num_params": 86_672_136,
            "metrics": {"ImageNet-1k": {"acc@1": 84.0, "acc@5": 96.9}},
        },
    )
    DEFAULT = SAIL_IN1K
