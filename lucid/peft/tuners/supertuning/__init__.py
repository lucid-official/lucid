"""``lucid.peft.tuners.supertuning`` — Super-Tuning: a sparse subset of the weight's entries, chosen by magnitude.

Super-Tuning (Ilin et al., 2026, arXiv:2607.09287) freezes the base weight and trains a sparse
support of its entries, the ones of largest magnitude, as an additive update that starts at zero,
with an optional LoRA beside it.  Pick it for a data-free sparse fine-tune whose trainable entries
follow the weight itself.

Not implemented yet.  ``SupertuningConfig`` is complete: it has every field of the reference
library's config, in its order and with its defaults, and reads and writes ``adapter_config.json``,
so an adapter file of this method loads as a config.  ``SupertuningModel`` is the tuner's place in
the registry: building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class
itself, raises ``NotImplementedError`` naming LCD-826, which tracks it, before any model is touched.
Importing the package registers the config under ``PeftType.SUPERTUNING`` and the placeholder tuner;
both are exported from ``lucid.peft``.
"""

from lucid.peft.tuners.supertuning._config import SupertuningConfig
from lucid.peft.tuners.supertuning._model import SupertuningModel

__all__ = ["SupertuningConfig", "SupertuningModel"]
