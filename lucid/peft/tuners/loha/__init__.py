"""``lucid.peft.tuners.loha`` — the LoHa layer types.

Where the reference library keeps them, so ``isinstance(m, LoHaLayer)`` and
``from lucid.peft.tuners.loha import Linear`` read as they do there.  One
layer type per module the method adapts: ``Linear``, ``Conv1d`` and
``Conv2d``.  The config and the model-level entry points are one level up, in
``lucid.peft``.
"""

from lucid.peft.tuners.loha._layer import Conv1d, Conv2d, Linear, LoHaLayer

__all__ = ["LoHaLayer", "Linear", "Conv1d", "Conv2d"]
