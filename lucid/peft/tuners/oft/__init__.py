"""``lucid.peft.tuners.oft`` — OFT: its config, tuner and layer types.

Where the reference library keeps them, so ``isinstance(m, OFTLayer)`` and
``from lucid.peft.tuners.oft import OFTConfig, Linear`` read as they do there.
One layer type per module the method adapts: ``Linear`` and ``Conv2d``.
Importing the package registers the method (its config under ``PeftType.OFT``,
its tuner and layers), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.oft._config import OFTConfig
from lucid.peft.tuners.oft._layer import Conv2d, Linear, OFTLayer
from lucid.peft.tuners.oft._model import OFTModel

__all__ = ["Conv2d", "Linear", "OFTConfig", "OFTLayer", "OFTModel"]
