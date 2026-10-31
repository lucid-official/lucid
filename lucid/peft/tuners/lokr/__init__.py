"""``lucid.peft.tuners.lokr`` — the LoKr layer types.

Where the reference library keeps them, so ``isinstance(m, LoKrLayer)`` and
``from lucid.peft.tuners.lokr import Linear`` read as they do there.  One
layer type per module the method adapts: ``Linear``, ``Conv1d`` and
``Conv2d``; ``factorization`` is the split of a dimension into the two
Kronecker factors.  The config and the model-level entry points are one level
up, in ``lucid.peft``.
"""

from lucid.peft.tuners.lokr._layer import Conv1d, Conv2d, Linear, LoKrLayer, factorization

__all__ = ["LoKrLayer", "Linear", "Conv1d", "Conv2d", "factorization"]
