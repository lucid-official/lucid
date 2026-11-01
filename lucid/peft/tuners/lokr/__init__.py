r"""``lucid.peft.tuners.lokr`` — LoKr: a weight update that is a Kronecker product.

LoKr (Yeh et al., 2023, arXiv:2309.14859 — LyCORIS) learns
:math:`\Delta W = \frac{\alpha}{r}\,W_1 \otimes W_2`, with each dimension of the weight factored
into two near-equal parts, one per Kronecker factor; a factor is a full matrix or, past a size, a
low-rank product.  The Kronecker structure gives a high-rank update from very few parameters, and
the first factor starts at zero so the adapted model starts as the base.  Pick it when LoRA's rank
is the limit and parameters are scarce.

Where the reference library keeps them, so ``isinstance(m, LoKrLayer)`` and ``from
lucid.peft.tuners.lokr import Linear`` read as they do there.  One layer type per module the method
adapts: ``Linear``, ``Conv1d`` and ``Conv2d``; ``factorization`` is the split of a dimension into
the two Kronecker factors.  ``LoKrConfig`` and ``LoKrModel`` are here and exported from
``lucid.peft``.  Importing the package registers the method (its config under ``PeftType.LOKR``, its
tuner and layers), which is what makes ``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.lokr._layer import Conv1d, Conv2d, Linear, LoKrLayer, factorization

__all__ = ["LoKrLayer", "Linear", "Conv1d", "Conv2d", "factorization"]
