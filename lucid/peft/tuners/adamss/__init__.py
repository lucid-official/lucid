"""``lucid.peft.tuners.adamss`` — AdaMSS: independent low-rank updates inside clustered subspaces of the SVD.

AdaMSS (Zheng et al., 2025, NeurIPS) takes the SVD of each weight, clusters its top singular
directions into subspaces and learns an independent low-rank update inside each, optionally pruning
the less important subspaces while training.  Pick it for a low-rank adapter whose capacity is spent
where the weight's spectrum needs it.

Not implemented yet.  ``AdamssConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``AdamssModel`` is the tuner's place in the
registry: building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class
itself, raises ``NotImplementedError`` naming LCD-828, which tracks it, before any model is touched.
Importing the package registers the config under ``PeftType.ADAMSS`` and the placeholder tuner; both
are exported from ``lucid.peft``.
"""

from lucid.peft.tuners.adamss._config import AdamssConfig
from lucid.peft.tuners.adamss._model import AdamssModel

__all__ = ["AdamssConfig", "AdamssModel"]
