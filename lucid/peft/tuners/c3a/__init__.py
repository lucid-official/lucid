"""``lucid.peft.tuners.c3a`` — C3A: a block-circulant weight update applied by FFT.

C3A (Chen et al., 2024, arXiv:2407.19342 — Circular Convolution Adaptation) cuts the update into
``block_size``-square blocks and makes each block a circulant, fixed by one kernel of length
``block_size``.  The layer trains those kernels, which give a high-rank update, and applies them by
FFT without building the matrix.  Pick it for a high-rank update with a parameter count set by
``block_size`` instead of a rank.

Where the reference library keeps them, so ``isinstance(m, C3ALayer)`` and ``from
lucid.peft.tuners.c3a import C3AConfig, Linear`` read as they do there.  One layer type for the one
module the method adapts, ``nn.Linear``.  ``C3AConfig`` and ``C3AModel`` are here and exported from
``lucid.peft``.  Importing the package registers the method (its config under ``PeftType.C3A``, its
tuner and layer), which is what makes ``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.c3a._config import C3AConfig
from lucid.peft.tuners.c3a._layer import C3ALayer, Linear
from lucid.peft.tuners.c3a._model import C3AModel

__all__ = ["C3AConfig", "C3ALayer", "C3AModel", "Linear"]
