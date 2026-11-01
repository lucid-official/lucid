"""``lucid.peft.tuners.vblora`` — VB-LoRA: low-rank factors mixed from one shared vector bank.

VB-LoRA (Li et al., 2024, arXiv:2405.15179 — Vector Bank LoRA) cuts the LoRA factors of every layer
into sub-vectors and builds each as a softmax-weighted mix of the top-:math:`k` vectors of a single
bank shared by the whole model; the learned logits that pick and weight them, and the bank, are the
trained parameters.  The adapter stores logits and a bank, not per-layer factors, so it holds far
fewer numbers than LoRA's.  Unlike LoRA it does not start as the base.  Pick it when storage per
adapter is the constraint.

Where the reference library keeps them, so ``isinstance(m, VBLoRALayer)`` and ``from
lucid.peft.tuners.vblora import VBLoRAConfig, Linear`` read as they do there.  One layer type for
the one module the method adapts, ``nn.Linear``.  ``VBLoRAConfig`` and ``VBLoRAModel`` (which holds
the bank) are here and exported from ``lucid.peft``.  Importing the package registers the method
(its config under ``PeftType.VBLORA``, its tuner and layer), which is what makes ``get_peft_model``
and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.vblora._config import VBLoRAConfig
from lucid.peft.tuners.vblora._layer import Linear, VBLoRALayer
from lucid.peft.tuners.vblora._model import VBLoRAModel

__all__ = ["Linear", "VBLoRAConfig", "VBLoRALayer", "VBLoRAModel"]
