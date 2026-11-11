"""``lucid.peft.tuners.psoft`` — PSOFT: orthogonal fine-tuning inside the weight's principal subspace.

PSOFT (Wu et al., 2025, arXiv:2505.11235) rotates a weight inside its principal subspace only: the
top singular directions are frozen and an r by r matrix between them, kept orthogonal, is trained.
Pick it for orthogonal fine-tuning that costs an r by r matrix per layer instead of a block-diagonal
one.

Not implemented yet.  ``PsoftConfig`` is complete: it has every field of the reference library's
config, in its order and with its defaults, and reads and writes ``adapter_config.json``, so an
adapter file of this method loads as a config.  ``PsoftModel`` is the tuner's place in the registry:
building one, through ``get_peft_model``, ``PeftModel.from_pretrained`` or the class itself, raises
``NotImplementedError`` naming LCD-825, which tracks it, before any model is touched.  Importing the
package registers the config under ``PeftType.PSOFT`` and the placeholder tuner; both are exported
from ``lucid.peft``.
"""

from lucid.peft.tuners.psoft._config import PsoftConfig
from lucid.peft.tuners.psoft._model import PsoftModel

__all__ = ["PsoftConfig", "PsoftModel"]
