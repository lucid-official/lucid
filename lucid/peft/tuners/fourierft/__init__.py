"""``lucid.peft.tuners.fourierft`` — FourierFT: its config, tuner and layer types.

Where the reference library keeps them, so ``isinstance(m, FourierFTLayer)`` and
``from lucid.peft.tuners.fourierft import FourierFTConfig, Linear`` read as they
do there.  One layer type for the one module the method adapts, ``nn.Linear``.
Importing the package registers the method (its config under
``PeftType.FOURIERFT``, its tuner and layer), which is what makes
``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.fourierft._config import FourierFTConfig
from lucid.peft.tuners.fourierft._layer import FourierFTLayer, Linear
from lucid.peft.tuners.fourierft._model import FourierFTModel

__all__ = ["FourierFTConfig", "FourierFTLayer", "FourierFTModel", "Linear"]
