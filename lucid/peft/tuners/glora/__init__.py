"""``lucid.peft.tuners.glora`` — GLoRA: learned corrections to both the weight and the bias of a layer.

GLoRA (Chavan et al., 2023, arXiv:2306.07967) corrects both the weight and the bias of a layer with
learned terms, each a low-rank product, a vector, a scalar or off, so one family covers LoRA-like
and scale-and-shift adapters.  Pick it for a search space of adapter shapes, or an adapter that also
moves the bias.

Not implemented yet.  ``GloraConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``GloraModel`` is the tuner's place in the registry:
building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class itself, raises
``NotImplementedError`` naming LCD-823, which tracks it, before any model is touched.  Importing the
package registers the config under ``PeftType.GLORA`` and the placeholder tuner; both are exported
from ``lucid.peft``.
"""

from lucid.peft.tuners.glora._config import GloraConfig
from lucid.peft.tuners.glora._model import GloraModel

__all__ = ["GloraConfig", "GloraModel"]
