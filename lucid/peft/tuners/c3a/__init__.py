"""``lucid.peft.tuners.c3a`` — C3A: its config, tuner and layer types.

Where the reference library keeps them, so ``isinstance(m, C3ALayer)`` and
``from lucid.peft.tuners.c3a import C3AConfig, Linear`` read as they do
there.  One layer type for the one module the method adapts, ``nn.Linear``.
Importing the package registers the method (its config under ``PeftType.C3A``,
its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.c3a._config import C3AConfig
from lucid.peft.tuners.c3a._layer import C3ALayer, Linear
from lucid.peft.tuners.c3a._model import C3AModel

__all__ = ["C3AConfig", "C3ALayer", "C3AModel", "Linear"]
