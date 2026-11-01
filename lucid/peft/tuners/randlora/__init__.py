r"""``lucid.peft.tuners.randlora`` — RandLoRA: a full-rank update from frozen random bases.

RandLoRA (Albert et al., 2025, arXiv:2502.00987) sums :math:`n = \lceil\min(d_{out},
d_{in})/r\rceil` frozen random rank-:math:`r` pairs :math:`(B_i, A_i)`, shared by every adapted
layer, and trains only a diagonal scale on each side of each pair.  The update is full rank, yet the
trained parameters are scales; lowering ``r`` raises their number. Pick it when LoRA's rank is the
limit and the adapter must stay small.

Where the reference library keeps them, so ``isinstance(m, RandLoraLayer)`` and ``from
lucid.peft.tuners.randlora import RandLoraConfig, Linear`` read as they do there.  One layer type
for the one module the method adapts, ``nn.Linear``.  ``RandLoraConfig`` and ``RandLoraModel``
(which holds the shared bases) are here and exported from ``lucid.peft``.  Importing the package
registers the method (its config under ``PeftType.RANDLORA``, its tuner and layer), which is what
makes ``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.randlora._config import RandLoraConfig
from lucid.peft.tuners.randlora._layer import Linear, RandLoraLayer
from lucid.peft.tuners.randlora._model import RandLoraModel

__all__ = ["Linear", "RandLoraConfig", "RandLoraLayer", "RandLoraModel"]
