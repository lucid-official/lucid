"""V-JEPA — feature prediction for video.

Bardes et al., 2024 (arXiv:2404.08471).  A clip is cut into tubelets, two
collections of blocks hide most of them through the whole depth of the
clip, and a predictor says what an averaged copy of the encoder would
report about what is hidden.  The image version is
:mod:`lucid.models.vision.ijepa`; the transformer they share is in
``vision/_common``.
"""

from lucid.models.vision.vjepa._config import VJEPAConfig, VJEPAObjective
from lucid.models.vision.vjepa._model import VJEPAForVideoClassification, VJEPAModel, VJEPAOutput
from lucid.models.vision.vjepa._pretrained import (
    vjepa_huge_16,
    vjepa_huge_16_384,
    vjepa_huge_16_384_cls,
    vjepa_huge_16_cls,
    vjepa_large_16,
    vjepa_large_16_cls,
)

__all__ = [
    "VJEPAConfig",
    "VJEPAObjective",
    "VJEPAModel",
    "VJEPAForVideoClassification",
    "VJEPAOutput",
    "vjepa_large_16",
    "vjepa_large_16_cls",
    "vjepa_huge_16",
    "vjepa_huge_16_cls",
    "vjepa_huge_16_384",
    "vjepa_huge_16_384_cls",
]
