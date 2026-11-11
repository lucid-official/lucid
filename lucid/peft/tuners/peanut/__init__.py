"""``lucid.peft.tuners.peanut`` — PEANuT: a small non-linear network that reads the frozen weight and tweaks it.

PEANuT (Zhong et al., 2024, arXiv:2410.01870) produces the weight update from the frozen weight
itself through a small trained non-linear network, not a fixed low-rank product.  Pick it for a non-
linear update that can follow the structure of the weight it adapts.

Not implemented yet.  ``PeanutConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``PeanutModel`` is the tuner's place in the
registry: building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class
itself, raises ``NotImplementedError`` naming LCD-823, which tracks it, before any model is touched.
Importing the package registers the config under ``PeftType.PEANUT`` and the placeholder tuner; both
are exported from ``lucid.peft``.
"""

from lucid.peft.tuners.peanut._config import PeanutConfig
from lucid.peft.tuners.peanut._model import PeanutModel

__all__ = ["PeanutConfig", "PeanutModel"]
