"""FCN — Fully Convolutional Network (Long et al., CVPR 2015).

Paper: "Fully Convolutional Networks for Semantic Segmentation"
"""

from lucid.models.vision.fcn._config import FCNConfig
from lucid.models.vision.fcn._model import FCNForSemanticSegmentation
from lucid.models.vision.fcn._pretrained import fcn_resnet50, fcn_resnet101
from lucid.models.vision.fcn._weights import FCNResNet50Weights, FCNResNet101Weights

__all__ = [
    "FCNConfig",
    "FCNForSemanticSegmentation",
    "fcn_resnet50",
    "fcn_resnet101",
    "FCNResNet50Weights",
    "FCNResNet101Weights",
]
