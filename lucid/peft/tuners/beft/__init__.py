r"""``lucid.peft.tuners.beft`` — BEFT: a trained bias added to chosen projections, and nothing else.

BEFT (Huang et al., 2025, arXiv:2509.15974) trains nothing but a bias added to chosen linear layers,
the value projections by default, starting at zero.  The bias folds into the layer's own bias when
it has one, so a merged model costs what the base does.  Pick it for a very small adapter when there
is little training data: one vector of the output width per layer, a few thousand numbers for a
language model.

Where the reference library keeps them, so ``isinstance(m, BeftLayer)`` and ``from
lucid.peft.tuners.beft import BeftConfig, Linear`` read as they do there.  One layer type for the one
module the method adapts, ``nn.Linear``.  ``BeftConfig`` and ``BeftModel`` are here and exported from
``lucid.peft``.  Importing the package registers the method (its config under ``PeftType.BEFT``, its
tuner and layer), which is what makes ``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.beft._config import BeftConfig
from lucid.peft.tuners.beft._layer import BeftLayer, Linear
from lucid.peft.tuners.beft._model import BeftModel

__all__ = ["BeftConfig", "BeftLayer", "BeftModel", "Linear"]
