"""``lucid.peft.tuners.trainable_tokens`` — trainable tokens: its config, tuner and layer types.

Where the reference library keeps them, so ``isinstance(m, TrainableTokensLayer)`` and
``from lucid.peft.tuners.trainable_tokens import TrainableTokensConfig, Embedding`` read as
they do there.  Two layer types: ``Embedding`` for the table, ``Linear`` for an output head
that is the table.  Importing the package registers the method (its config under
``PeftType.TRAINABLE_TOKENS``, its tuner and layers), which is what makes ``get_peft_model``
and ``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.trainable_tokens._config import TrainableTokensConfig
from lucid.peft.tuners.trainable_tokens._layer import Embedding, Linear, TrainableTokensLayer
from lucid.peft.tuners.trainable_tokens._model import TrainableTokensModel

__all__ = [
    "Embedding",
    "Linear",
    "TrainableTokensConfig",
    "TrainableTokensLayer",
    "TrainableTokensModel",
]
