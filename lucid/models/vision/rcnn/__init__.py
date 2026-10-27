"""R-CNN — object detection (Girshick et al., CVPR 2014).

Paper: "Rich feature hierarchies for accurate object detection
        and semantic segmentation"
"""

from lucid.models.vision.rcnn._config import RCNNConfig
from lucid.models.vision.rcnn._model import RCNNForObjectDetection
from lucid.models.vision.rcnn._pretrained import rcnn

__all__ = ["RCNNConfig", "RCNNForObjectDetection", "rcnn"]
