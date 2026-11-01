r"""``lucid.peft.tuners.ia3`` — (IA)^3: one learned rescaling vector per layer.

(IA)^3 (Liu et al., 2022, arXiv:2205.05638 — Infused Adapter by Inhibiting and Amplifying Inner
Activations) learns a vector :math:`\ell` per adapted layer that rescales its output channels,
:math:`y = \ell\odot(Wx + b)`, or, for a block's feed-forward layers, its input,
:math:`y = W(\ell\odot x) + b`.  It starts at ones, so the adapted model starts as the base, and it
folds into the weight.  Pick it for an adapter of one vector per layer, when rescaling channels is
enough; that carries less than a low-rank update.

Where the reference library keeps them, so ``isinstance(m, IA3Layer)`` and ``from
lucid.peft.tuners.ia3 import Linear`` read as they do there.  One layer type per module the method
adapts: ``Linear`` and the convolutions ``Conv2d`` / ``Conv3d``.  ``IA3Config`` and ``IA3Model`` are
here and exported from ``lucid.peft``.  Importing the package registers the method (its config under
``PeftType.IA3``, its tuner and layers), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.ia3._layer import Conv2d, Conv3d, IA3Layer, Linear

__all__ = ["IA3Layer", "Linear", "Conv2d", "Conv3d"]
