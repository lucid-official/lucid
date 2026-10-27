"""Pretrained-weight declarations for the DETR family.

Both paper-cited variants (Carion et al., ECCV 2020) ship COCO 2017
detection checkpoints converted from the original Facebook DETR
reference checkpoints: :class:`DETRResNet50Weights` and
:class:`DETRResNet101Weights`.

Each checkpoint is the COCO detector at ``num_classes = 91`` (91
foreground + 1 no-object) and uses the reference DETR eval pipeline:
longest-side 1333 resize + square pad on the bottom and right /
bilinear interpolation / ImageNet normalisation, with bounding boxes
riding along the geometric stages (the
:class:`~lucid.utils.transforms.Detection` preset).
"""

from lucid.utils.transforms import Detection
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# DETR's reference pads each image on the bottom and right, leaving it in the
# top-left corner -- the frame ``postprocess`` scales normalised boxes into --
# and does not round its canvas to a stride multiple, so DETR keeps the 1333
# square it was converted and checked on rather than the R-CNNs' 1344.
_PRESET = Detection(max_size=1333, size_divisible=1, pad_position="top_left", pad_value=0.0)


@register_weights("detr_resnet50")
class DETRResNet50Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.detr_resnet50`.

    Single COCO 2017 checkpoint converted from Facebook DETR's
    ``detr_resnet50`` reference checkpoint (Carion et al., 2020; ~41.5M
    params, 42.0 box mAP).

    Examples
    --------
    >>> from lucid.models.vision.detr import DETRResNet50Weights
    >>> list(DETRResNet50Weights.__members__)
    ['COCO_2017', 'DEFAULT']

    ``DEFAULT`` is an alias for the tag a bare
    ``pretrained=True`` resolves to, so the two ways of asking for
    weights cannot drift apart.
    """

    COCO_2017 = WeightEntry(
        url=f"{HUB_BASE}/detr-resnet-50/resolve/main/COCO_2017/model.safetensors",
        sha256="b4f246aeef2f7ef67e42d59e3ff6eda151351404bafd0c6e70b8d52b7147ec95",
        num_classes=91,
        transforms=_PRESET,
        meta={
            "tag": "COCO_2017",
            "source": "facebookresearch/detr/detr_resnet50",
            "license": "apache-2.0",
            "num_params": 41_524_768,
            "metrics": {"COCO": {"box mAP": 42.0}},
        },
    )
    DEFAULT = COCO_2017


@register_weights("detr_resnet101")
class DETRResNet101Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.detr_resnet101`.

    Single COCO 2017 checkpoint converted from Facebook DETR's
    ``detr_resnet101`` reference checkpoint (Carion et al., 2020; ~60.5M
    params, 43.5 box mAP).

    Examples
    --------
    >>> from lucid.models.vision.detr import DETRResNet101Weights
    >>> list(DETRResNet101Weights.__members__)
    ['COCO_2017', 'DEFAULT']

    ``DEFAULT`` is an alias for the tag a bare
    ``pretrained=True`` resolves to, so the two ways of asking for
    weights cannot drift apart.
    """

    COCO_2017 = WeightEntry(
        url=f"{HUB_BASE}/detr-resnet-101/resolve/main/COCO_2017/model.safetensors",
        sha256="799df46dcfbcb0492c790fc8263766457fefbb419107483c8d6030a76a241510",
        num_classes=91,
        transforms=_PRESET,
        meta={
            "tag": "COCO_2017",
            "source": "facebookresearch/detr/detr_resnet101",
            "license": "apache-2.0",
            "num_params": 60_464_672,
            "metrics": {"COCO": {"box mAP": 43.5}},
        },
    )
    DEFAULT = COCO_2017
