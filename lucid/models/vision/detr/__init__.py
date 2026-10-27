"""DETR — DEtection TRansformer (Carion et al., ECCV 2020).

Paper: "End-to-End Object Detection with Transformers"
"""

from lucid.models.vision.detr._config import DETRConfig
from lucid.models.vision.detr._model import DETRForObjectDetection
from lucid.models.vision.detr._pretrained import detr_resnet50, detr_resnet101
from lucid.models.vision.detr._weights import DETRResNet50Weights, DETRResNet101Weights

__all__ = [
    "DETRConfig",
    "DETRForObjectDetection",
    "detr_resnet50",
    "detr_resnet101",
    "DETRResNet50Weights",
    "DETRResNet101Weights",
]
