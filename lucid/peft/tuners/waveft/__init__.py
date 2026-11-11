"""``lucid.peft.tuners.waveft`` — WaveFT: a few coefficients of the update in a wavelet basis.

WaveFT (Bilican et al., 2025, arXiv:2505.12532) learns a few coefficients of the weight update in a
wavelet basis, at random positions, and rebuilds the update by an inverse wavelet transform.  Pick
it for a sparse update in a frequency basis, with the parameter count set directly.

Not implemented yet.  ``WaveFTConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``WaveFTModel`` is the tuner's place in the
registry: building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class
itself, raises ``NotImplementedError`` naming LCD-826, which tracks it, before any model is touched.
Importing the package registers the config under ``PeftType.WAVEFT`` and the placeholder tuner; both
are exported from ``lucid.peft``.
"""

from lucid.peft.tuners.waveft._config import WaveFTConfig
from lucid.peft.tuners.waveft._model import WaveFTModel

__all__ = ["WaveFTConfig", "WaveFTModel"]
