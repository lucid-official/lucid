"""``lucid.peft.tuners.gralora`` — GraLoRA: block-wise low-rank updates that exchange information between blocks.

GraLoRA (Jung et al., 2025, arXiv:2505.20355) cuts a weight into a grid of sub-blocks and gives each
its own low-rank pair, with information exchanged between blocks, so the update has the parameter
count of a LoRA and a higher effective rank.  Pick it for a LoRA-sized adapter on layers where plain
LoRA underfits, such as at higher ranks.

Not implemented yet.  ``GraloraConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``GraloraModel`` is the tuner's place in the
registry: building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class
itself, raises ``NotImplementedError`` naming LCD-823, which tracks it, before any model is touched.
Importing the package registers the config under ``PeftType.GRALORA`` and the placeholder tuner;
both are exported from ``lucid.peft``.
"""

from lucid.peft.tuners.gralora._config import GraloraConfig
from lucid.peft.tuners.gralora._model import GraloraModel

__all__ = ["GraloraConfig", "GraloraModel"]
