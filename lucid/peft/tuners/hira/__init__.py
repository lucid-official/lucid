"""``lucid.peft.tuners.hira`` — HiRA: a low-rank matrix that modulates the weight elementwise.

HiRA (Huang et al., 2025, ICLR) multiplies the frozen weight elementwise by a low-rank matrix, so
the update can have a rank far above the one it is built from.  Pick it for more expressive updates
than LoRA at the same parameter count.

Not implemented yet.  ``HiraConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``HiraModel`` is the tuner's place in the registry:
building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class itself, raises
``NotImplementedError`` naming LCD-823, which tracks it, before any model is touched.  Importing the
package registers the config under ``PeftType.HIRA`` and the placeholder tuner; both are exported
from ``lucid.peft``.
"""

from lucid.peft.tuners.hira._config import HiraConfig
from lucid.peft.tuners.hira._model import HiraModel

__all__ = ["HiraConfig", "HiraModel"]
