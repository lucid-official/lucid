"""``lucid.peft.tuners.frod`` — FRoD: a full-rank update from a sparse trainable rotation.

FRoD (Wan et al., 2025, arXiv:2512.23485) builds a basis from the base weights, shared by layers of
one kind, and trains a sparse rotation matrix over it, for a full-rank update from few parameters.
Pick it for converging faster than a low-rank adapter at a similar number of trainable parameters.

Not implemented yet.  ``FrodConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``FrodModel`` is the tuner's place in the registry:
building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class itself, raises
``NotImplementedError`` naming LCD-825, which tracks it, before any model is touched.  Importing the
package registers the config under ``PeftType.FROD`` and the placeholder tuner; both are exported
from ``lucid.peft``.
"""

from lucid.peft.tuners.frod._config import FrodConfig
from lucid.peft.tuners.frod._model import FrodModel

__all__ = ["FrodConfig", "FrodModel"]
