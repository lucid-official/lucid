r"""``lucid.peft.tuners.psoft`` — PSOFT: orthogonal fine-tuning inside the weight's principal subspace.

PSOFT (Wu et al., 2025, arXiv:2505.11235 — Efficient Orthogonal Fine-Tuning with Principal Subspace
Adaptation) rotates a weight inside its principal subspace only: the top-:math:`r` singular
directions are frozen as two factors, and an :math:`r \times r` matrix between them, kept orthogonal
by a Cayley parameterisation and relaxed by two diagonal scalings, is all that trains (about
:math:`r^2/2 + 2r` numbers per layer, whatever the layer's width).  The update starts at zero and
folds into the weight.  Pick it for orthogonal fine-tuning that costs one small matrix per layer
instead of a block-diagonal one.

Where the reference library keeps them, so ``isinstance(m, PsoftLayer)`` and ``from
lucid.peft.tuners.psoft import PsoftConfig, Linear`` read as they do there.  One layer type for the one
module the method adapts, ``nn.Linear``; ``OrthLayer`` is the trained matrix.  ``PsoftConfig`` and
``PsoftModel`` are here and exported from ``lucid.peft``.  Importing the package registers the method
(its config under ``PeftType.PSOFT``, its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.psoft._config import PsoftConfig
from lucid.peft.tuners.psoft._layer import Linear, PsoftLayer
from lucid.peft.tuners.psoft._model import PsoftModel
from lucid.peft.tuners.psoft._orth import OrthLayer

__all__ = ["Linear", "OrthLayer", "PsoftConfig", "PsoftLayer", "PsoftModel"]
