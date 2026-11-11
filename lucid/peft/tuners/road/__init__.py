r"""``lucid.peft.tuners.road`` — RoAd: a rotation and scaling of pairs of output features.

RoAd (Liao and Monz, 2024, arXiv:2409.00119 — 2D Rotary Adaptation) pairs the output features of a
layer into 2-D vectors, within groups of ``group_size`` (element :math:`i` with element
:math:`i + \text{group}/2`), and rotates and scales each pair by a learned angle and scale.  The
rotation folds into the weight and the bias at no inference cost.  Pick it for an adapter with very
few parameters (about half a number per output feature) that merges into the weight.

Where the reference library keeps them, so ``isinstance(m, RoadLayer)`` and ``from
lucid.peft.tuners.road import RoadConfig, Linear`` read as they do there.  One layer type for the one
module the method adapts, ``nn.Linear``.  ``RoadConfig`` and ``RoadModel`` are here and exported from
``lucid.peft``.  Importing the package registers the method (its config under ``PeftType.ROAD``, its
tuner and layer), which is what makes ``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.road._config import RoadConfig
from lucid.peft.tuners.road._layer import Linear, RoadLayer
from lucid.peft.tuners.road._model import RoadModel

__all__ = ["Linear", "RoadConfig", "RoadLayer", "RoadModel"]
