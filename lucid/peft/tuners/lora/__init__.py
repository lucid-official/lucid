"""``lucid.peft.tuners.lora`` — the LoRA layer types.

Where the reference library keeps them, so ``isinstance(m, LoraLayer)``
and ``from lucid.peft.tuners.lora import Linear`` read as they do there.
The config and the model-level entry points are one level up, in
``lucid.peft``.
"""

from lucid.peft.tuners.lora._layer import Linear, LoraLayer

__all__ = ["LoraLayer", "Linear"]
