"""``lucid.peft.tuners.hra`` — HRA: its config, tuner and layer types.

Where the reference library keeps them, so ``isinstance(m, HRALayer)`` and
``from lucid.peft.tuners.hra import HRAConfig, Linear`` read as they do there.
One layer type per module the method adapts: ``Linear`` and ``Conv2d``.
Importing the package registers the method (its config under ``PeftType.HRA``,
its tuner and layers), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.hra._config import HRAConfig
from lucid.peft.tuners.hra._layer import Conv2d, HRALayer, Linear
from lucid.peft.tuners.hra._model import HRAModel

__all__ = ["Conv2d", "HRAConfig", "HRALayer", "HRAModel", "Linear"]
