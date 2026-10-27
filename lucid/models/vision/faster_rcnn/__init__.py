"""Faster R-CNN — object detection (Ren et al., NeurIPS 2015).

Paper: "Faster R-CNN: Towards Real-Time Object Detection with
        Region Proposal Networks"

ResNet-50-FPN reference configuration (Lin et al., CVPR 2017 FPN) with a
COCO-pretrained checkpoint.
"""

from lucid.models.vision.faster_rcnn._config import FasterRCNNConfig
from lucid.models.vision.faster_rcnn._model import FasterRCNNForObjectDetection
from lucid.models.vision.faster_rcnn._pretrained import faster_rcnn, faster_rcnn_resnet50_fpn
from lucid.models.vision.faster_rcnn._weights import FasterRCNNResNet50FPNWeights

__all__ = [
    "FasterRCNNConfig",
    "FasterRCNNForObjectDetection",
    "faster_rcnn",
    "faster_rcnn_resnet50_fpn",
    "FasterRCNNResNet50FPNWeights",
]
