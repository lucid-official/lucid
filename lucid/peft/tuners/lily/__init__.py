"""``lucid.peft.tuners.lily`` — Lily: low-rank adapters shared across layers and mixed by a router.

Lily (Zhong et al., 2024, arXiv:2407.09946) shares one low-rank down-projection among several
consecutive layers and a pool of up-projections among all layers, mixed in each layer by a learned
router.  Pick it for fewer, larger adapters than one LoRA per layer, which then can carry a higher
rank.

Not implemented yet.  ``LilyConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``LilyModel`` is the tuner's place in the registry:
building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class itself, raises
``NotImplementedError`` naming LCD-824, which tracks it, before any model is touched.  Importing the
package registers the config under ``PeftType.LILY`` and the placeholder tuner; both are exported
from ``lucid.peft``.
"""

from lucid.peft.tuners.lily._config import LilyConfig
from lucid.peft.tuners.lily._model import LilyModel

__all__ = ["LilyConfig", "LilyModel"]
