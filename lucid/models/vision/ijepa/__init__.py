"""I-JEPA — self-supervised pretraining that predicts representations.

Assran et al., CVPR 2023 (arXiv:2301.08243).  A context block is encoded,
and a narrow predictor must say what an exponential moving average of that
encoder would report about four blocks it was never shown.  Nothing is
reconstructed in pixel space, which is the point.
"""

from lucid.models.vision.ijepa._config import IJEPAConfig, IJEPAObjective
from lucid.models.vision.ijepa._model import IJEPAForImageClassification, IJEPAModel, IJEPAOutput
from lucid.models.vision.ijepa._pretrained import (
    ijepa_base_16,
    ijepa_base_16_cls,
    ijepa_huge_14,
    ijepa_huge_14_cls,
    ijepa_huge_16_448,
    ijepa_huge_16_448_cls,
    ijepa_large_16,
    ijepa_large_16_cls,
)

__all__ = [
    "IJEPAConfig",
    "IJEPAObjective",
    "IJEPAModel",
    "IJEPAForImageClassification",
    "IJEPAOutput",
    "ijepa_base_16",
    "ijepa_base_16_cls",
    "ijepa_large_16",
    "ijepa_large_16_cls",
    "ijepa_huge_14",
    "ijepa_huge_14_cls",
    "ijepa_huge_16_448",
    "ijepa_huge_16_448_cls",
]
