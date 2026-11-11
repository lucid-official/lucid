r"""``lucid.peft.tuners.delora`` — DeLoRA: a low-rank update whose strength is bounded apart from its direction.

DeLoRA (Bini et al., 2025, arXiv:2503.18225 — Decoupled Low-rank Adaptation) normalises each
rank-one component :math:`b_i a_i^\top` of the LoRA update and scales the sum by a learned boundary
:math:`\lambda / r` and by the pretrained weight's column norms, so the direction of the update is
learned apart from its strength and its norm cannot exceed :math:`\lambda`.  Pick it for a
fine-tune that must stay close to the base model, with the distance a hyper-parameter, or for one
that has to survive a high learning rate or a long run.

Where the reference library keeps them, so ``isinstance(m, DeloraLayer)`` and ``from
lucid.peft.tuners.delora import DeloraConfig, Linear`` read as they do there.  One layer type for
the one module the method adapts, ``nn.Linear``.  ``DeloraConfig`` and ``DeloraModel`` are here and
exported from ``lucid.peft``.  Importing the package registers the method (its config under
``PeftType.DELORA``, its tuner and layer), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.delora._config import DeloraConfig
from lucid.peft.tuners.delora._layer import DeloraLayer, Linear
from lucid.peft.tuners.delora._model import DeloraModel

__all__ = ["DeloraConfig", "DeloraLayer", "DeloraModel", "Linear"]
