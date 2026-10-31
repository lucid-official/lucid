"""``lucid.peft.tuners.ia3`` — the (IA)^3 layer types.

Where the reference library keeps them, so ``isinstance(m, IA3Layer)`` and
``from lucid.peft.tuners.ia3 import Linear`` read as they do there.  One layer
type per module the method adapts: ``Linear`` and the convolutions
``Conv2d`` / ``Conv3d``.  The config and the model-level entry points are one
level up, in ``lucid.peft``.
"""

from lucid.peft.tuners.ia3._layer import Conv2d, Conv3d, IA3Layer, Linear

__all__ = ["IA3Layer", "Linear", "Conv2d", "Conv3d"]
