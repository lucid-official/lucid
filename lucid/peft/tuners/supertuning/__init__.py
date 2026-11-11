r"""``lucid.peft.tuners.supertuning`` — Super-Tuning: a sparse subset of the weight's entries, chosen by magnitude.

Super-Tuning (Ilin et al., 2026, arXiv:2607.09287) freezes the base weight and trains a sparse
support of its entries, the ones of largest magnitude, as an additive update that starts at zero,
with an optional LoRA beside it (the paper's Supra hybrid).  Nothing is measured to choose the
support: it is a function of the weight alone, so the choice is repeatable and needs no data.  Pick
it for a sparse fine-tune whose trainable entries follow the weight itself, at the parameter count
``sparsity`` sets.

Where the reference library keeps them, so ``isinstance(m, SupertuningLayer)`` and ``from
lucid.peft.tuners.supertuning import SupertuningConfig, Linear`` read as they do there.  One layer
type for the one module the method adapts, ``nn.Linear``.  ``SupertuningConfig`` and
``SupertuningModel`` are here and exported from ``lucid.peft``.  Importing the package registers the
method (its config under ``PeftType.SUPERTUNING``, its tuner and layer), which is what makes
``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.supertuning._config import SupertuningConfig
from lucid.peft.tuners.supertuning._layer import Linear, SupertuningLayer
from lucid.peft.tuners.supertuning._model import SupertuningModel

__all__ = ["Linear", "SupertuningConfig", "SupertuningLayer", "SupertuningModel"]
