r"""``lucid.peft.tuners.adalora`` — AdaLoRA: LoRA whose rank per layer is allocated while it trains.

AdaLoRA (Zhang et al., 2023, arXiv:2303.10512) writes the update of a layer as a truncated SVD,
:math:`\Delta W = \frac{\alpha}{\mathrm{ranknum}}\,P\,\mathrm{diag}(\lambda)\,Q`, with :math:`P` and
:math:`Q` kept near-orthogonal by a penalty on the loss.  It tracks how much each singular triplet
matters during training and masks the least important ones on a schedule, so a fixed parameter
budget goes to the layers that need it.  Pick it over LoRA when you know the budget but not which
layers deserve the rank; call ``AdaLoraModel.update_and_allocate`` after each optimizer step.

Where the reference library keeps them, so ``isinstance(m, AdaLoraLayer)`` and ``from
lucid.peft.tuners.adalora import AdaLoraConfig, SVDLinear`` read as they do there.  One layer type
per module the method adapts: ``SVDLinear`` and ``SVDConv2d``; ``RankAllocator`` is the budget
schedule.  ``AdaLoraConfig`` and ``AdaLoraModel`` are here and exported from ``lucid.peft``.
Importing the package registers the method (its config under ``PeftType.ADALORA``, its tuner and
layers), which is what makes ``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
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
