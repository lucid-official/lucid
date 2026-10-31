"""``lucid.peft.tuners.shira`` — SHiRA: its config, tuner and layer types.

Where the reference library keeps them, so ``isinstance(m, ShiraLayer)`` and
``from lucid.peft.tuners.shira import ShiraConfig, Linear`` read as they do
there.  One layer type for the one module the method adapts, ``nn.Linear``.
Importing the package registers the method (its config under ``PeftType.SHIRA``,
its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.shira._config import ShiraConfig
from lucid.peft.tuners.shira._layer import Linear, ShiraLayer
from lucid.peft.tuners.shira._mask import random_mask
from lucid.peft.tuners.shira._model import ShiraModel

__all__ = ["Linear", "ShiraConfig", "ShiraLayer", "ShiraModel", "random_mask"]
