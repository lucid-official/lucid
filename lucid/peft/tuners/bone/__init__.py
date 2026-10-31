"""``lucid.peft.tuners.bone`` — Bone, the method MiSS replaced: its config, tuner and layer types.

Kept because adapter files written under the name exist; new work uses
``lucid.peft.tuners.miss``.  Importing the package registers the method (its config
under ``PeftType.BONE``, its tuner and layer), which is what lets ``PeftModel.from_pretrained``
read such a file.
"""

from lucid.peft.tuners.bone._config import BoneConfig
from lucid.peft.tuners.bone._layer import BoneLayer, Linear
from lucid.peft.tuners.bone._model import BoneModel

__all__ = ["BoneConfig", "BoneLayer", "BoneModel", "Linear"]
