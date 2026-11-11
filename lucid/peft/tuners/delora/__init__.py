"""``lucid.peft.tuners.delora`` — DeLoRA: a low-rank update whose strength is bounded apart from its direction.

DeLoRA (Bini et al., 2025, arXiv:2503.18225) normalises each rank-one component of the LoRA update
and scales the sum by a learned boundary, so the direction of the update is learned apart from its
strength and its norm cannot exceed that boundary.  Pick it for a fine-tune that must stay close to
the base model, with the distance a hyper-parameter.

Not implemented yet.  ``DeloraConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``DeloraModel`` is the tuner's place in the
registry: building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class
itself, raises ``NotImplementedError`` naming LCD-823, which tracks it, before any model is touched.
Importing the package registers the config under ``PeftType.DELORA`` and the placeholder tuner; both
are exported from ``lucid.peft``.
"""

from lucid.peft.tuners.delora._config import DeloraConfig
from lucid.peft.tuners.delora._model import DeloraModel

__all__ = ["DeloraConfig", "DeloraModel"]
