"""``lucid.peft.tuners.cartridge`` — Cartridge: a trained key-value cache that stands in for a long document.

Cartridge (Eyuboglu et al., 2025, arXiv:2506.06266) is a trained key-value cache: learned key and
value pairs put in front of the model's cache in place of a long document it would otherwise read at
every query.  Pick it for serving many queries over one large, unchanging document.

Not implemented yet.  ``CartridgeConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``CartridgeModel`` is the tuner's place in the
registry: building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class
itself, raises ``NotImplementedError`` naming LCD-826, which tracks it, before any model is touched.
Importing the package registers the config under ``PeftType.CARTRIDGE`` and the placeholder tuner;
both are exported from ``lucid.peft``.
"""

from lucid.peft.tuners.cartridge._config import CartridgeConfig
from lucid.peft.tuners.cartridge._model import CartridgeModel

__all__ = ["CartridgeConfig", "CartridgeModel"]
