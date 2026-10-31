"""``lucid.peft.tuners.adalora`` — AdaLoRA: its config, tuner, layer types and rank allocator.

Where the reference library keeps them, so ``isinstance(m, AdaLoraLayer)`` and
``from lucid.peft.tuners.adalora import AdaLoraConfig, SVDLinear`` read as they
do there.  One layer type per module the method adapts: ``SVDLinear`` and
``SVDConv2d``.  Importing the package registers the method (its config under
``PeftType.ADALORA``, its tuner and layers), which is what makes
``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.adalora._allocator import RankAllocator
from lucid.peft.tuners.adalora._config import AdaLoraConfig
from lucid.peft.tuners.adalora._layer import AdaLoraLayer, SVDConv2d, SVDLinear
from lucid.peft.tuners.adalora._model import AdaLoraModel

__all__ = [
    "AdaLoraConfig",
    "AdaLoraLayer",
    "AdaLoraModel",
    "RankAllocator",
    "SVDConv2d",
    "SVDLinear",
]
