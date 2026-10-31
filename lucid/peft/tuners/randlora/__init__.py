"""``lucid.peft.tuners.randlora`` — RandLoRA: its config, tuner and layer types.

Where the reference library keeps them, so ``isinstance(m, RandLoraLayer)`` and
``from lucid.peft.tuners.randlora import RandLoraConfig, Linear`` read as they do
there.  One layer type for the one module the method adapts, ``nn.Linear``.
Importing the package registers the method (its config under ``PeftType.RANDLORA``,
its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.randlora._config import RandLoraConfig
from lucid.peft.tuners.randlora._layer import Linear, RandLoraLayer
from lucid.peft.tuners.randlora._model import RandLoraModel

__all__ = ["Linear", "RandLoraConfig", "RandLoraLayer", "RandLoraModel"]
