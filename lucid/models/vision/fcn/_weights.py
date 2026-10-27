"""Pretrained-weight declarations for the FCN family.

Both paper-cited variants (Long et al., CVPR 2015) ship COCO-with-VOC
semantic-segmentation checkpoints converted from reference_vision:
:class:`FCNResNet50Weights` and :class:`FCNResNet101Weights`.

Each checkpoint is the 21-class (20 Pascal-VOC categories + background)
segmenter trained on the COCO-val2017 images filtered to the VOC label
set, using the reference_vision segmentation eval pipeline: shorter-side 520
resize / bilinear interpolation / ImageNet normalisation, with masks
riding along the geometric stages (the
:class:`~lucid.utils.transforms.Segmentation` preset).
"""

from lucid.utils.transforms import Segmentation
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# Reference ``FCN_ResNet{50,101}_Weights.COCO_WITH_VOC_LABELS_V1`` uses
# ``SemanticSegmentation(resize_size=520)`` — a resize, no centre crop.
_PRESET = Segmentation(crop_size=None, resize_size=520)


@register_weights("fcn_resnet50")
class FCNResNet50Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.fcn_resnet50`.

    Single COCO-with-VOC-labels checkpoint converted from reference_vision's
    ``FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1`` (Long et al., 2015;
    ~35.3M params, 60.5 mIoU / 91.4 pixel acc).

    Examples
    --------
    >>> from lucid.models.vision.fcn import FCNResNet50Weights
    >>> list(FCNResNet50Weights.__members__)
    ['COCO_WITH_VOC_LABELS_V1', 'DEFAULT']

    ``DEFAULT`` is an alias for the tag a bare
    ``pretrained=True`` resolves to, so the two ways of asking for
    weights cannot drift apart.
    """

    COCO_WITH_VOC_LABELS_V1 = WeightEntry(
        url=f"{HUB_BASE}/fcn-resnet-50/resolve/main/" "COCO_WITH_VOC_LABELS_V1/model.safetensors",
        sha256="4e87ab18d1a18fac1d40f15984b70ae4a13cee336d9c0c646cd474a937600234",
        num_classes=21,
        transforms=_PRESET,
        meta={
            "tag": "COCO_WITH_VOC_LABELS_V1",
            "source": "reference_vision/FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1",
            "license": "bsd-3-clause",
            "num_params": 35_322_218,
            "metrics": {"COCO-val2017-VOC-labels": {"mIoU": 60.5, "pixel acc": 91.4}},
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


@register_weights("fcn_resnet101")
class FCNResNet101Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.fcn_resnet101`.

    Single COCO-with-VOC-labels checkpoint converted from reference_vision's
    ``FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1`` (Long et al., 2015;
    ~54.3M params, 63.7 mIoU / 91.9 pixel acc).

    Examples
    --------
    >>> from lucid.models.vision.fcn import FCNResNet101Weights
    >>> list(FCNResNet101Weights.__members__)
    ['COCO_WITH_VOC_LABELS_V1', 'DEFAULT']

    ``DEFAULT`` is an alias for the tag a bare
    ``pretrained=True`` resolves to, so the two ways of asking for
    weights cannot drift apart.
    """

    COCO_WITH_VOC_LABELS_V1 = WeightEntry(
        url=f"{HUB_BASE}/fcn-resnet-101/resolve/main/" "COCO_WITH_VOC_LABELS_V1/model.safetensors",
        sha256="f46d07ed89706a0a51d327e10c748f109580862d96e6ea8afd0cfaee6fb1d133",
        num_classes=21,
        transforms=_PRESET,
        meta={
            "tag": "COCO_WITH_VOC_LABELS_V1",
            "source": "reference_vision/FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1",
            "license": "bsd-3-clause",
            "num_params": 54_314_346,
            "metrics": {"COCO-val2017-VOC-labels": {"mIoU": 63.7, "pixel acc": 91.9}},
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1
