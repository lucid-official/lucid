r"""``lucid.peft.tuners.unilora`` — Uni-LoRA: every LoRA factor of the model read out of one trainable vector.

Uni-LoRA (Li et al., 2025, arXiv:2506.00799) writes the LoRA parameters of every adapted layer as a
read of one vector :math:`\theta_d` through a fixed projection, :math:`\theta_D = P\,\theta_d`, whose
rows are one-hot at indices drawn uniformly and whose columns are normalised so that :math:`P` is an
isometry.  The adapter of the whole model is that vector: ``theta_d_length`` numbers however many
layers it adapts.  Pick it for the smallest adapter file for a given rank, shared across all layers.

Where the reference library keeps them, so ``isinstance(m, UniLoraLayer)`` and ``from
lucid.peft.tuners.unilora import UniLoraConfig, Linear`` read as they do there.  One layer type for
the one module the method adapts, ``nn.Linear``.  ``UniLoraConfig`` and ``UniLoraModel`` (which
holds the shared vector) are here and exported from ``lucid.peft``.  Importing the package registers
the method (its config under ``PeftType.UNILORA``, its tuner and layer), which is what makes
``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.unilora._config import UniLoraConfig
from lucid.peft.tuners.unilora._layer import Linear, UniLoraLayer
from lucid.peft.tuners.unilora._model import UniLoraModel

__all__ = ["Linear", "UniLoraConfig", "UniLoraLayer", "UniLoraModel"]
