"""``lucid.peft.tuners.osf`` — OSF: constrained full fine-tuning in the weight's low singular directions.

OSF (Nayak et al., 2025, arXiv:2504.07097) freezes the top singular directions of each weight and
trains the rest with gradients projected orthogonally to the frozen ones, so tasks learned in
sequence do not overwrite each other; no parameter is added.  Pick it for learning several tasks in
sequence without keeping an adapter per task.

Not implemented yet.  ``OSFConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``OSFModel`` is the tuner's place in the registry:
building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class itself, raises
``NotImplementedError`` naming LCD-825, which tracks it, before any model is touched.  Importing the
package registers the config under ``PeftType.OSF`` and the placeholder tuner; both are exported
from ``lucid.peft``.
"""

from lucid.peft.tuners.osf._config import OSFConfig
from lucid.peft.tuners.osf._model import OSFModel

__all__ = ["OSFConfig", "OSFModel"]
