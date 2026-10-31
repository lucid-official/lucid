"""``lucid.peft.tuners.boft`` — BOFT: its config, tuner and layer types.

Where the reference library keeps them, so ``isinstance(m, BOFTLayer)`` and
``from lucid.peft.tuners.boft import BOFTConfig, Linear`` read as they do there.
One layer type per module the method adapts: ``Linear`` and ``Conv2d``.
Importing the package registers the method (its config under ``PeftType.BOFT``,
its tuner and layers), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.boft._config import BOFTConfig
from lucid.peft.tuners.boft._layer import BOFTLayer, Conv2d, Linear
from lucid.peft.tuners.boft._model import BOFTModel

__all__ = ["BOFTConfig", "BOFTLayer", "BOFTModel", "Conv2d", "Linear"]
