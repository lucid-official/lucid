r"""``lucid.peft.tuners.glora`` — GLoRA: learned corrections to both the weight and the bias of a layer.

GLoRA (Chavan et al., 2023, arXiv:2306.07967 — Generalized LoRA) corrects the weight and the bias
of a layer with five learned support tensors, :math:`W' = W + W A + B` and
:math:`b' = b + b D + E + W C`, each a low-rank product, a vector, a scalar or off, so one family
covers LoRA-like, scale-and-shift and prompt-like adapters.  The update reads the weight and the
bias, so a merge keeps the values it was folded into and an unmerge gives them back exactly.  Pick it
for a search space of adapter shapes, or an adapter that also moves the bias.

Where the reference library keeps them, so ``isinstance(m, GloraLayer)`` and ``from
lucid.peft.tuners.glora import GloraConfig, Linear`` read as they do there.  One layer type for the
one module the method adapts, ``nn.Linear``.  ``GloraConfig`` and ``GloraModel`` are here and
exported from ``lucid.peft``.  Importing the package registers the method (its config under
``PeftType.GLORA``, its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.glora._config import GloraConfig
from lucid.peft.tuners.glora._layer import GloraLayer, Linear
from lucid.peft.tuners.glora._model import GloraModel

__all__ = ["GloraConfig", "GloraLayer", "GloraModel", "Linear"]
