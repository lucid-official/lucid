"""Pretrained-weight declarations for the Faster R-CNN family.

The single paper-cited reference variant (Ren et al., NeurIPS 2015; FPN
from Lin et al., CVPR 2017) ships a COCO detection checkpoint converted
from the reference ``FasterRCNN_ResNet50_FPN_Weights.COCO_V1``:
:class:`FasterRCNNResNet50FPNWeights`.

The checkpoint is the COCO detector at ``num_classes = 91`` (90
categories + background slot 0) and uses the reference detection eval
pipeline: shortest side to 800 with the longest capped at 1333, the image
copied into the top-left corner of a 1344 square (1333 rounded up to a
multiple of 32, as the reference pads its batches) / bilinear
interpolation / ImageNet normalisation (the
:class:`~lucid.utils.transforms.Detection` preset).
``weights.transforms().image_size(h, w)`` is the ``image_sizes`` entry
:meth:`~lucid.models.vision.faster_rcnn.FasterRCNNForObjectDetection.postprocess`
clips to.
"""

from lucid.utils.transforms import Detection
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# Reference ``FasterRCNN`` defaults: min_size=800, max_size=1333.  Its
# transform copies each image into the top-left of a canvas rounded up to a
# multiple of 32, which is where ``postprocess`` measures ``image_sizes`` from.
_PRESET = Detection(
    min_size=800, max_size=1333, size_divisible=32, pad_position="top_left", pad_value=0.0
)


@register_weights("faster_rcnn")
@register_weights("faster_rcnn_resnet50_fpn")
class FasterRCNNResNet50FPNWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.faster_rcnn_resnet50_fpn`.

    Single COCO checkpoint converted from the reference
    ``FasterRCNN_ResNet50_FPN_Weights.COCO_V1`` (91 classes, box AP 37.0).

    Examples
    --------
    >>> from lucid.models.vision.faster_rcnn import FasterRCNNResNet50FPNWeights
    >>> list(FasterRCNNResNet50FPNWeights.__members__)
    ['COCO_V1', 'DEFAULT']

    ``DEFAULT`` is an alias for the tag a bare
    ``pretrained=True`` resolves to, so the two ways of asking for
    weights cannot drift apart.
    """

    COCO_V1 = WeightEntry(
        url=(f"{HUB_BASE}/faster-rcnn-resnet-50-fpn/resolve/main/" "COCO_V1/model.safetensors"),
        sha256="cc8d2c79125d73432bee0eadd049f2871368a90a7bb0b0c5cba63af3c29f3205",
        num_classes=91,
        transforms=_PRESET,
        meta={
            "tag": "COCO_V1",
            "source": "reference_vision/FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
            "license": "bsd-3-clause",
            "num_params": 41_755_286,
            "metrics": {"COCO": {"box mAP": 37.0}},
        },
    )
    DEFAULT = COCO_V1
