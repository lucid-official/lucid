"""``lucid.peft.tuners.ln_tuning`` — LN tuning: train a copy of each chosen normalisation layer.

LN tuning (Zhao et al., 2023, arXiv:2312.11420 — Tuning LayerNorm in Attention) trains the
normalisation layers of a model and nothing else.  Each chosen layer gets a copy of itself that
trains while the original stays frozen; the method adds no tensor of its own, so an adapter is a few
:math:`d` parameters per layer, and merging makes the trained copy the layer.  Pick it for a tiny
adapter that needs no new tensors.  It applies one adapter at a time.

Where the reference library keeps them, so ``isinstance(m, LNTuningLayer)`` and ``from
lucid.peft.tuners.ln_tuning import LNTuningConfig, LNTuningLayer`` read as they do there.  One layer
type, which wraps whichever module ``target_modules`` names.  ``LNTuningConfig`` and
``LNTuningModel`` are here and exported from ``lucid.peft``.  Importing the package registers the
method (its config under ``PeftType.LN_TUNING``, its tuner and layer), which is what makes
``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.ln_tuning._config import LNTuningConfig
from lucid.peft.tuners.ln_tuning._layer import LNTuningLayer
from lucid.peft.tuners.ln_tuning._model import LNTuningModel

__all__ = ["LNTuningConfig", "LNTuningLayer", "LNTuningModel"]
