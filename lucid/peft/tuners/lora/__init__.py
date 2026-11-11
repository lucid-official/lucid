r"""``lucid.peft.tuners.lora`` — LoRA: a low-rank update learned beside each frozen weight.

LoRA (Hu et al., 2021, arXiv:2106.09685) freezes a pretrained weight :math:`W` and learns a
rank-:math:`r` update next to it, :math:`y = W x + s\,B A x` with :math:`s = \alpha / r`.  :math:`B`
starts at zero, so the adapted model starts as the base, and the update folds into :math:`W` on
merge.  It is the method to start with: it has a layer for every common module type and the
widest set of options — DoRA (``use_dora``), rsLoRA, PiSSA and OLoRA starts, per-module ranks and
alphas, and combining adapters (``add_weighted_adapter``).

Where the reference library keeps them, so ``isinstance(m, LoraLayer)`` and ``from
lucid.peft.tuners.lora import Linear`` read as they do there.  One layer type per module the method
adapts: ``Linear``, ``Embedding``, the convolutions ``Conv1d`` / ``Conv2d`` / ``Conv3d`` and their
transposes ``ConvTranspose1d`` / ``ConvTranspose2d`` / ``ConvTranspose3d``, ``MultiheadAttention``,
and ``ParamWrapper`` for a raw parameter. ``LoraRuntimeConfig`` is the type of ``LoraConfig.runtime_config``
(settings valid at runtime only), and ``ArrowConfig``, ``BdLoraConfig``, ``KasaConfig``,
``MontecloraConfig`` and ``VeloraConfig`` hold the options of the extensions ``LoraConfig`` has a field for
and refuses to apply.  ``LoraConfig`` and ``LoraModel`` are defined here too and
exported from ``lucid.peft``, which is where ``get_peft_model`` is.  Importing the package registers
the method (its config under ``PeftType.LORA``, its tuner and layers), which is what makes
``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.lora._attention import MultiheadAttention
from lucid.peft.tuners.lora._config import LoraRuntimeConfig
from lucid.peft.tuners.lora._conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from lucid.peft.tuners.lora._embedding import Embedding
from lucid.peft.tuners.lora._layer import Linear, LoraLayer, QuantizedLinear
from lucid.peft.tuners.lora._parameter import ParamWrapper

__all__ = [
    "LoraLayer",
    "Linear",
    "QuantizedLinear",
    "Embedding",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "MultiheadAttention",
    "ParamWrapper",
    "LoraRuntimeConfig",
]
