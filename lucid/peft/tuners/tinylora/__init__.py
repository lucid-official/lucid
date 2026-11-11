"""``lucid.peft.tuners.tinylora`` — TinyLoRA: a trainable vector of a few numbers projected into the SVD basis.

TinyLoRA (Morris et al., 2026, arXiv:2602.04118) freezes the SVD of each adapted weight and trains a
vector of a few numbers, projected by fixed random tensors into the singular basis, which can be
shared across the whole model.  Pick it for an adapter of tens of parameters, shared across the
model.

Not implemented yet.  ``TinyLoraConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``TinyLoraModel`` is the tuner's place in the
registry: building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class
itself, raises ``NotImplementedError`` naming LCD-824, which tracks it, before any model is touched.
Importing the package registers the config under ``PeftType.TINYLORA`` and the placeholder tuner;
both are exported from ``lucid.peft``.
"""

from lucid.peft.tuners.tinylora._config import TinyLoraConfig
from lucid.peft.tuners.tinylora._model import TinyLoraModel

__all__ = ["TinyLoraConfig", "TinyLoraModel"]
