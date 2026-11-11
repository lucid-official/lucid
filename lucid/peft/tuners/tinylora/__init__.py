r"""``lucid.peft.tuners.tinylora`` — TinyLoRA: a trainable vector of a few numbers projected into the SVD basis.

TinyLoRA (Morris et al., 2026, arXiv:2602.04118) freezes the truncated SVD of each adapted weight and
trains a vector of ``u`` numbers, which fixed random matrices turn into the small matrix between the
frozen factors; modules can share the vector, down to one vector for the whole model.  Pick it for an
adapter of tens of parameters, or fewer.

Where the reference library keeps them, so ``isinstance(m, TinyLoraLayer)`` and ``from
lucid.peft.tuners.tinylora import TinyLoraConfig, Linear`` read as they do there.  One layer type for
each module the method adapts, ``nn.Linear`` and ``nn.Embedding``.  ``TinyLoraConfig`` and
``TinyLoraModel`` (which holds the trained vectors) are here and exported from ``lucid.peft``.
Importing the package registers the method (its config under ``PeftType.TINYLORA``, its tuner and
layers), which is what makes ``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.tinylora._config import TinyLoraConfig
from lucid.peft.tuners.tinylora._layer import Embedding, Linear, TinyLoraLayer
from lucid.peft.tuners.tinylora._model import TinyLoraModel

__all__ = ["Embedding", "Linear", "TinyLoraConfig", "TinyLoraLayer", "TinyLoraModel"]
