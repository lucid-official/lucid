"""Fast R-CNN — object detection (Girshick, ICCV 2015).

Paper: "Fast R-CNN"
"""

from lucid.models.vision.fast_rcnn._config import FastRCNNConfig
from lucid.models.vision.fast_rcnn._model import FastRCNNForObjectDetection
from lucid.models.vision.fast_rcnn._pretrained import fast_rcnn

__all__ = ["FastRCNNConfig", "FastRCNNForObjectDetection", "fast_rcnn"]
