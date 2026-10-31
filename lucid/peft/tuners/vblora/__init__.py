"""``lucid.peft.tuners.vblora`` — VB-LoRA: its config, tuner and layer types.

Where the reference library keeps them, so ``isinstance(m, VBLoRALayer)`` and
``from lucid.peft.tuners.vblora import VBLoRAConfig, Linear`` read as they do
there.  One layer type for the one module the method adapts, ``nn.Linear``.
Importing the package registers the method (its config under ``PeftType.VBLORA``,
its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.vblora._config import VBLoRAConfig
from lucid.peft.tuners.vblora._layer import Linear, VBLoRALayer
from lucid.peft.tuners.vblora._model import VBLoRAModel

__all__ = ["Linear", "VBLoRAConfig", "VBLoRALayer", "VBLoRAModel"]
