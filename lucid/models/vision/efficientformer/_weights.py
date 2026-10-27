"""Pretrained-weight declarations for the EfficientFormer family.

All three paper-cited variants (Li et al., NeurIPS 2022) ship ImageNet-1k
checkpoints converted from timm's snapshot-distilled model zoo:
:class:`EfficientFormerL1Weights`, :class:`EfficientFormerL3Weights`,
:class:`EfficientFormerL7Weights`.

Every checkpoint uses the timm EfficientFormer eval pipeline: 224 crop /
235 resize (``crop_pct = 0.95``, and timm truncates 224 / 0.95 = 235.8) /
bicubic interpolation / ImageNet stats.
The published checkpoints are *distilled* (DeiT-style hard distillation),
so the Lucid classifier keeps both ``head`` and ``head_dist`` and averages
their logits at inference to reproduce the reported top-1 accuracy.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = ImageClassification(crop_size=224, resize_size=235, interpolation="bicubic")


@register_weights("efficientformer_l1_cls")
class EfficientFormerL1Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.efficientformer_l1_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``efficientformer_l1.snap_dist_in1k`` (Li et al., 2022; ~12.3M params,
    79.2% top-1).

    Examples
    --------
    >>> from lucid.models.vision.efficientformer._weights import EfficientFormerL1Weights
    >>> list(EfficientFormerL1Weights.__members__)
    ['SNAP_DIST_IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> EfficientFormerL1Weights.DEFAULT is EfficientFormerL1Weights.SNAP_DIST_IN1K
    True
    >>> EfficientFormerL1Weights.SNAP_DIST_IN1K.num_classes
    1000
    """

    SNAP_DIST_IN1K = WeightEntry(
        url=(f"{HUB_BASE}/efficientformer-l1/resolve/main/" "SNAP_DIST_IN1K/model.safetensors"),
        sha256="cb1320ef1279f7e3d257785660e64c44a43bb6e50e75fbb326d0b171b4ef731c",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "SNAP_DIST_IN1K",
            "source": "timm/efficientformer_l1.snap_dist_in1k",
            "license": "apache-2.0",
            "num_params": 12_289_928,
            "metrics": {"ImageNet-1k": {"acc@1": 79.2}},
        },
    )
    DEFAULT = SNAP_DIST_IN1K


@register_weights("efficientformer_l3_cls")
class EfficientFormerL3Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.efficientformer_l3_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``efficientformer_l3.snap_dist_in1k`` (Li et al., 2022; ~31.4M params,
    82.4% top-1).

    Examples
    --------
    >>> from lucid.models.vision.efficientformer._weights import EfficientFormerL3Weights
    >>> list(EfficientFormerL3Weights.__members__)
    ['SNAP_DIST_IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> EfficientFormerL3Weights.DEFAULT is EfficientFormerL3Weights.SNAP_DIST_IN1K
    True
    >>> EfficientFormerL3Weights.SNAP_DIST_IN1K.num_classes
    1000
    """

    SNAP_DIST_IN1K = WeightEntry(
        url=(f"{HUB_BASE}/efficientformer-l3/resolve/main/" "SNAP_DIST_IN1K/model.safetensors"),
        sha256="3de48cd9b1d94c90441d230034e629e7eb927d39b3ea01766c25606ec2ac5b5a",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "SNAP_DIST_IN1K",
            "source": "timm/efficientformer_l3.snap_dist_in1k",
            "license": "apache-2.0",
            "num_params": 31_406_000,
            "metrics": {"ImageNet-1k": {"acc@1": 82.4}},
        },
    )
    DEFAULT = SNAP_DIST_IN1K


@register_weights("efficientformer_l7_cls")
class EfficientFormerL7Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.efficientformer_l7_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``efficientformer_l7.snap_dist_in1k`` (Li et al., 2022; ~82.2M params,
    83.3% top-1).

    Examples
    --------
    >>> from lucid.models.vision.efficientformer._weights import EfficientFormerL7Weights
    >>> list(EfficientFormerL7Weights.__members__)
    ['SNAP_DIST_IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> EfficientFormerL7Weights.DEFAULT is EfficientFormerL7Weights.SNAP_DIST_IN1K
    True
    >>> EfficientFormerL7Weights.SNAP_DIST_IN1K.num_classes
    1000
    """

    SNAP_DIST_IN1K = WeightEntry(
        url=(f"{HUB_BASE}/efficientformer-l7/resolve/main/" "SNAP_DIST_IN1K/model.safetensors"),
        sha256="66b10e5952a853b758bc8647110e0526a8a8c79ef45bd201c379bd24b1cd445c",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "SNAP_DIST_IN1K",
            "source": "timm/efficientformer_l7.snap_dist_in1k",
            "license": "apache-2.0",
            "num_params": 82_229_328,
            "metrics": {"ImageNet-1k": {"acc@1": 83.3}},
        },
    )
    DEFAULT = SNAP_DIST_IN1K
