r"""``lucid.peft.tuners.fourierft`` — FourierFT: the weight update as a few frequency coefficients.

FourierFT (Gao et al., 2024, arXiv:2405.03003) learns ``n_frequency`` coefficients of a
:math:`d_{out}\times d_{in}` spectrum at fixed pseudo-random locations; the inverse 2-D Fourier
transform of that sparse spectrum, times ``scaling``, is the update.  The locations come from
``random_loc_seed`` alone, so an adapter holds just the coefficients.  Pick it for an adapter
smaller than LoRA's whose update has no low-rank structure.

Where the reference library keeps them, so ``isinstance(m, FourierFTLayer)`` and ``from
lucid.peft.tuners.fourierft import FourierFTConfig, Linear`` read as they do there.  One layer type
for the one module the method adapts, ``nn.Linear``.  ``FourierFTConfig`` and ``FourierFTModel`` are
here and exported from ``lucid.peft``.  Importing the package registers the method (its config under
``PeftType.FOURIERFT``, its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.fourierft._config import FourierFTConfig
from lucid.peft.tuners.fourierft._layer import FourierFTLayer, Linear
from lucid.peft.tuners.fourierft._model import FourierFTModel

__all__ = ["FourierFTConfig", "FourierFTLayer", "FourierFTModel", "Linear"]
