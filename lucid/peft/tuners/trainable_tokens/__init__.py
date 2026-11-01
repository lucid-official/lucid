"""``lucid.peft.tuners.trainable_tokens`` — trainable tokens: a few rows of an embedding table.

Trainable tokens re-train existing token embeddings, or the fresh ones a resized vocabulary added,
without training the whole table: the listed rows are copied into a small parameter, the layer reads
those rows from it instead of from the frozen table, and only the copy is trained and saved.  A
file holds ``len(token_indices) x embedding_dim`` numbers whatever the vocabulary's size.  Pick it
to teach a model new special tokens, or to move a few existing ones, with an adapter in place of a
full embedding table.

Where the reference library keeps them, so ``isinstance(m, TrainableTokensLayer)`` and ``from
lucid.peft.tuners.trainable_tokens import TrainableTokensConfig, Embedding`` read as they do there.
Two layer types: ``Embedding`` for the table, ``Linear`` for an output head that is the table.
``TrainableTokensConfig`` and ``TrainableTokensModel`` are here and exported from ``lucid.peft``.
Importing the package registers the method (its config under ``PeftType.TRAINABLE_TOKENS``, its
tuner and layers), which is what makes ``get_peft_model`` and ``PeftModel.from_pretrained`` know it.
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
