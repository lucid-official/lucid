"""``lucid.peft.tuners.road`` — RoAd: a rotation and scaling of pairs of output features.

RoAd (Liao and Monz, 2024, arXiv:2409.00119) pairs the output features of a layer into 2-D vectors
and rotates and scales each pair, which folds into the weight and the bias at no inference cost.
Pick it for an adapter that merges into the weight and also batches requests of different adapters
cheaply.

Not implemented yet.  ``RoadConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``RoadModel`` is the tuner's place in the registry:
building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class itself, raises
``NotImplementedError`` naming LCD-825, which tracks it, before any model is touched.  Importing the
package registers the config under ``PeftType.ROAD`` and the placeholder tuner; both are exported
from ``lucid.peft``.
"""

from lucid.peft.tuners.road._config import RoadConfig
from lucid.peft.tuners.road._model import RoadModel

__all__ = ["RoadConfig", "RoadModel"]
