"""``lucid.peft.tuners.vera`` — VeRA: its config, tuner and layer types.

Where the reference library keeps them, so ``isinstance(m, VeraLayer)`` and
``from lucid.peft.tuners.vera import VeraConfig, Linear`` read as they do
there.  One layer type for the one module the method adapts, ``nn.Linear``.
Importing the package registers the method (its config under ``PeftType.VERA``,
its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.vera._config import VeraConfig
from lucid.peft.tuners.vera._layer import Linear, VeraLayer
from lucid.peft.tuners.vera._model import VeraModel

__all__ = ["Linear", "VeraConfig", "VeraLayer", "VeraModel"]
