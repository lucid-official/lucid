"""MaskFormer (Cheng et al., NeurIPS 2021).

Paper: "Per-Pixel Classification is Not All You Need for Semantic Segmentation"
"""

from lucid.models.vision.maskformer._config import MaskFormerConfig
from lucid.models.vision.maskformer._model import (
    MaskFormerForSemanticSegmentation,
    MaskFormerOutput,
)
from lucid.models.vision.maskformer._pretrained import maskformer_resnet50, maskformer_resnet101
from lucid.models.vision.maskformer._weights import (
    MaskFormerResNet50Weights,
    MaskFormerResNet101Weights,
)

__all__ = [
    "MaskFormerConfig",
    "MaskFormerForSemanticSegmentation",
    "MaskFormerOutput",
    "maskformer_resnet50",
    "maskformer_resnet101",
    "MaskFormerResNet50Weights",
    "MaskFormerResNet101Weights",
]
