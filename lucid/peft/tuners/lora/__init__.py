"""``lucid.peft.tuners.lora`` — the LoRA layer types.

Where the reference library keeps them, so ``isinstance(m, LoraLayer)``
and ``from lucid.peft.tuners.lora import Linear`` read as they do there.
One layer type per module the method adapts: ``Linear``, ``Embedding``,
the convolutions ``Conv1d`` / ``Conv2d`` / ``Conv3d`` and their transposes
``ConvTranspose1d`` / ``ConvTranspose2d`` / ``ConvTranspose3d``, ``MultiheadAttention``,
and ``ParamWrapper`` for a raw parameter.  The config and the model-level
entry points are one level up, in ``lucid.peft``.
"""

from lucid.peft.tuners.lora._attention import MultiheadAttention
from lucid.peft.tuners.lora._conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from lucid.peft.tuners.lora._embedding import Embedding
from lucid.peft.tuners.lora._layer import Linear, LoraLayer
from lucid.peft.tuners.lora._parameter import ParamWrapper

__all__ = [
    "LoraLayer",
    "Linear",
    "Embedding",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "MultiheadAttention",
    "ParamWrapper",
]
