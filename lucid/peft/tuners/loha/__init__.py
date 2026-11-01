r"""``lucid.peft.tuners.loha`` — LoHa: a Hadamard product of low-rank products as the update.

LoHa (Hyeon-Woo et al., 2021, arXiv:2108.06098 — the FedPara parameterisation, in the LyCORIS form)
learns :math:`\Delta W = \frac{\alpha}{r}(W_{1a}W_{1b})\odot(W_{2a}W_{2b})`.  Two rank-:math:`r`
factors multiplied elementwise reach rank :math:`r^2` for twice LoRA's parameters, and one of them
starts at zero so the adapted model starts as the base.  Pick it over LoRA when a higher-rank update
is worth the extra parameters (the LyCORIS family is mostly used for diffusion models).

Where the reference library keeps them, so ``isinstance(m, LoHaLayer)`` and ``from
lucid.peft.tuners.loha import Linear`` read as they do there.  One layer type per module the method
adapts: ``Linear``, ``Conv1d`` and ``Conv2d``.  ``LoHaConfig`` and ``LoHaModel`` are here and
exported from ``lucid.peft``.  Importing the package registers the method (its config under
``PeftType.LOHA``, its tuner and layers), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.loha._layer import Conv1d, Conv2d, Linear, LoHaLayer

__all__ = ["LoHaLayer", "Linear", "Conv1d", "Conv2d"]
