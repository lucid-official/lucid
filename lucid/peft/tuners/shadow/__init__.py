"""``lucid.peft.tuners.shadow`` — ShadowPEFT: a parallel shadow network whose state rides the decoder loop.

ShadowPEFT (Li et al., 2026, arXiv:2604.19254) augments a frozen decoder with a small parallel
network whose state rides the base model's decoder loop, injecting the difference from the base
hidden states into every wrapped block.  Use it for reading a config file of this method; no adapter
of it is built.

Refused.  ``ShadowConfig`` is complete: it has every field of the reference library's config, in its
order and with its defaults, and reads and writes ``adapter_config.json``, so a file of this method
loads as a config.  ``ShadowModel`` is the tuner's place in the registry: building one, through
``get_peft_model``, ``PeftModel.from_pretrained`` or the class itself, raises
``NotImplementedError`` saying why, before any model is touched.  Importing the package registers
the config under ``PeftType.SHADOW`` and the placeholder tuner; both are exported from
``lucid.peft``.
"""

from lucid.peft.tuners.shadow._config import ShadowConfig
from lucid.peft.tuners.shadow._model import ShadowModel

__all__ = ["ShadowConfig", "ShadowModel"]
