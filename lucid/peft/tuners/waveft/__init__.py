r"""``lucid.peft.tuners.waveft`` — WaveFT: a few coefficients of the update in a wavelet basis.

WaveFT (Bilican et al., 2025, arXiv:2505.12532) learns a few coefficients of the weight update in a
wavelet basis, at random positions, and rebuilds the update by an inverse 2-D wavelet transform
(Daubechies, symlet or coiflet filters, one level).  The parameter count is set directly, by
``n_frequency`` coefficients a layer, rather than by a rank, and a wavelet basis keeps the update
local in space as well as in frequency.  Pick it for a sparse update that is not low rank, when the
count of trained numbers has to be exactly what the budget allows.

Where the reference library keeps them, so ``isinstance(m, WaveFTLayer)`` and ``from
lucid.peft.tuners.waveft import WaveFTConfig, Linear`` read as they do there.  One layer type for the
one module the method adapts, ``nn.Linear``.  ``WaveFTConfig`` and ``WaveFTModel`` are here and
exported from ``lucid.peft``.  Importing the package registers the method (its config under
``PeftType.WAVEFT``, its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.waveft._config import WaveFTConfig
from lucid.peft.tuners.waveft._layer import Linear, WaveFTLayer
from lucid.peft.tuners.waveft._model import WaveFTModel

__all__ = ["Linear", "WaveFTConfig", "WaveFTLayer", "WaveFTModel"]
