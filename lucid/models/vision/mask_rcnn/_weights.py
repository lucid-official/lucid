"""Pretrained-weight declarations for the Mask R-CNN family.

The single paper-cited reference variant (He et al., ICCV 2017; Faster
R-CNN from Ren et al., NeurIPS 2015; FPN from Lin et al., CVPR 2017)
ships a COCO instance-segmentation checkpoint converted from the
reference ``MaskRCNN_ResNet50_FPN_Weights.COCO_V1``:
:class:`MaskRCNNResNet50FPNWeights`.

The checkpoint is the COCO detector at ``num_classes = 91`` (90
categories + background slot 0) and uses the reference detection eval
pipeline: longest-side 1333 resize, the image copied into the top-left
corner of a 1344 square (1333 rounded up to a multiple of 32, as the
reference pads its batches) / bilinear interpolation / ImageNet
normalisation (the :class:`~lucid.utils.transforms.Detection` preset).
``weights.transforms().image_size(h, w)`` is the ``image_sizes`` entry
:meth:`~lucid.models.vision.mask_rcnn.MaskRCNNForObjectDetection.postprocess`
clips boxes and pastes masks to.
"""

from lucid.utils.transforms import Detection
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# The reference transform copies each image into the top-left of a canvas
# rounded up to a multiple of 32, which is where ``postprocess`` measures
# ``image_sizes`` from.
_PRESET = Detection(max_size=1333, size_divisible=32, pad_position="top_left", pad_value=0.0)


@register_weights("mask_rcnn")
@register_weights("mask_rcnn_resnet50_fpn")
class MaskRCNNResNet50FPNWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mask_rcnn_resnet50_fpn`.

    Single COCO checkpoint converted from the reference
    ``MaskRCNN_ResNet50_FPN_Weights.COCO_V1`` (91 classes, box AP 37.9 /
    mask AP 34.6).

    Examples
    --------
    >>> from lucid.models.vision.mask_rcnn import MaskRCNNResNet50FPNWeights
    >>> list(MaskRCNNResNet50FPNWeights.__members__)
    ['COCO_V1', 'DEFAULT']

    ``DEFAULT`` is an alias for the tag a bare
    ``pretrained=True`` resolves to, so the two ways of asking for
    weights cannot drift apart.
    """

    COCO_V1 = WeightEntry(
        url=(f"{HUB_BASE}/mask-rcnn-resnet-50-fpn/resolve/main/" "COCO_V1/model.safetensors"),
        sha256="120a1648dda49f1799d8cc581bef8ec83ba679313602a80d5278409ab4e0ddcd",
        num_classes=91,
        transforms=_PRESET,
        meta={
            "tag": "COCO_V1",
            "source": "reference_vision/MaskRCNN_ResNet50_FPN_Weights.COCO_V1",
            "license": "bsd-3-clause",
            "num_params": 44_401_393,
            "metrics": {"COCO": {"box mAP": 37.9, "mask mAP": 34.6}},
        },
    )
    DEFAULT = COCO_V1
