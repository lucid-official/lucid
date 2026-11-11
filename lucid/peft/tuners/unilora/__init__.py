"""``lucid.peft.tuners.unilora`` — Uni-LoRA: every LoRA factor of the model read out of one trainable vector.

Uni-LoRA (Li et al., 2025, arXiv:2506.00799) reads every LoRA factor of every adapted layer out of
one trainable vector through a fixed projection, so the adapter of the whole model is that vector.
Pick it for the smallest adapter file for a given rank, shared across all layers.

Not implemented yet.  ``UniLoraConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``UniLoraModel`` is the tuner's place in the
registry: building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class
itself, raises ``NotImplementedError`` naming LCD-824, which tracks it, before any model is touched.
Importing the package registers the config under ``PeftType.UNILORA`` and the placeholder tuner;
both are exported from ``lucid.peft``.
"""

from lucid.peft.tuners.unilora._config import UniLoraConfig
from lucid.peft.tuners.unilora._model import UniLoraModel

__all__ = ["UniLoraConfig", "UniLoraModel"]
