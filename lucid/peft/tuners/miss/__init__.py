"""``lucid.peft.tuners.miss`` — MiSS: its config, tuner and layer types.

Where the reference library keeps them, so ``isinstance(m, MissLayer)`` and
``from lucid.peft.tuners.miss import MissConfig, Linear`` read as they do
there.  One layer type for the one module the method adapts, ``nn.Linear``.
Importing the package registers the method (its config under ``PeftType.MISS``,
its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.miss._config import MissConfig
from lucid.peft.tuners.miss._layer import Linear, MissLayer
from lucid.peft.tuners.miss._model import MissModel

__all__ = ["Linear", "MissConfig", "MissLayer", "MissModel"]
