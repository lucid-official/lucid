r"""``lucid.peft.tuners.osf`` — OSF: constrained full fine-tuning in the weight's low singular directions.

OSF (Nayak et al., 2025, arXiv:2504.07097 — Sculpting Subspaces) takes the SVD of each weight,
freezes its top singular directions, which carry what the model already knows, and trains the rest, with
the gradients projected onto the complement of the frozen directions so that a new task does not overwrite
an old one.  Nothing is added to the model: the trained tensors are the weight's own low part.  Pick it for
learning several tasks in sequence without keeping an adapter per task.

Where the reference library keeps them, so ``isinstance(m, OSFLayer)`` and ``from
lucid.peft.tuners.osf import OSFConfig, Linear`` read as they do there.  One layer type for the one
module the method adapts, ``nn.Linear``.  ``OSFConfig`` and ``OSFModel`` are here and exported from
``lucid.peft``.  Importing the package registers the method (its config under ``PeftType.OSF``, its
tuner and layer), which is what makes ``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.osf._config import OSFConfig
from lucid.peft.tuners.osf._layer import Linear, OSFLayer
from lucid.peft.tuners.osf._model import OSFModel

__all__ = ["Linear", "OSFConfig", "OSFLayer", "OSFModel"]
