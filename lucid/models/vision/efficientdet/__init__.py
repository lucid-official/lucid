"""EfficientDet — scalable compound-scaled detector (Tan et al., CVPR 2020).

Paper: "EfficientDet: Scalable and Efficient Object Detection"
"""

from lucid.models.vision.efficientdet._config import EfficientDetConfig
from lucid.models.vision.efficientdet._model import EfficientDetForObjectDetection
from lucid.models.vision.efficientdet._pretrained import (
    efficientdet_d0,
    efficientdet_d1,
    efficientdet_d2,
    efficientdet_d3,
    efficientdet_d4,
    efficientdet_d5,
    efficientdet_d6,
    efficientdet_d7,
)

__all__ = [
    "EfficientDetConfig",
    "EfficientDetForObjectDetection",
    "efficientdet_d0",
    "efficientdet_d1",
    "efficientdet_d2",
    "efficientdet_d3",
    "efficientdet_d4",
    "efficientdet_d5",
    "efficientdet_d6",
    "efficientdet_d7",
]
