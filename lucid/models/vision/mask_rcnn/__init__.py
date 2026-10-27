"""Mask R-CNN — instance segmentation (He et al., ICCV 2017).

Paper: "Mask R-CNN"

ResNet-50-FPN reference configuration (Faster R-CNN backbone + parallel
mask branch) with a COCO-pretrained checkpoint.
"""

from lucid.models.vision.mask_rcnn._config import MaskRCNNConfig
from lucid.models.vision.mask_rcnn._model import MaskRCNNForObjectDetection
from lucid.models.vision.mask_rcnn._pretrained import mask_rcnn, mask_rcnn_resnet50_fpn
from lucid.models.vision.mask_rcnn._weights import MaskRCNNResNet50FPNWeights

__all__ = [
    "MaskRCNNConfig",
    "MaskRCNNForObjectDetection",
    "mask_rcnn",
    "mask_rcnn_resnet50_fpn",
    "MaskRCNNResNet50FPNWeights",
]
