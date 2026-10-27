"""YOLO family — You Only Look Once object detectors.

References
----------
- YOLOv1: Redmon et al., CVPR 2016
- YOLOv2: Redmon & Farhadi, CVPR 2017 (YOLO9000)
- YOLOv3: Redmon & Farhadi, arXiv 2018
- YOLOv4: Bochkovskiy et al., arXiv 2020
"""

from lucid.models.vision.yolo._v1 import YOLOConfig, YOLOForObjectDetection, yolo, yolo_tiny
from lucid.models._utils._detection import MultiScaleResolution
from lucid.models.vision.yolo._v2 import YOLOV2Config, YOLOV2ForObjectDetection, yolo_v2
from lucid.models.vision.yolo._v3 import (
    YOLOV3Config,
    YOLOV3ForObjectDetection,
    YOLOV3TinyForObjectDetection,
    yolo_v3,
    yolo_v3_tiny,
)
from lucid.models.vision.yolo._v4 import YOLOV4Config, YOLOV4ForObjectDetection, yolo_v4
from lucid.models.vision.yolo._weights import (
    YOLOV2Weights,
    YOLOV3TinyWeights,
    YOLOV3Weights,
    YOLOV4Weights,
)

__all__ = [
    # YOLOv1
    "YOLOConfig",
    "YOLOForObjectDetection",
    "yolo",
    "yolo_tiny",
    # YOLOv2
    "MultiScaleResolution",
    "YOLOV2Config",
    "YOLOV2ForObjectDetection",
    "yolo_v2",
    "YOLOV2Weights",
    # YOLOv3
    "YOLOV3Config",
    "YOLOV3ForObjectDetection",
    "YOLOV3TinyForObjectDetection",
    "yolo_v3",
    "yolo_v3_tiny",
    "YOLOV3Weights",
    "YOLOV3TinyWeights",
    # YOLOv4
    "YOLOV4Config",
    "YOLOV4ForObjectDetection",
    "yolo_v4",
    "YOLOV4Weights",
]
