"""``lucid.peft.tuners.p_tuning`` — p-tuning: virtual tokens produced by an MLP or LSTM.

P-tuning (Liu et al., 2021, arXiv:2103.10385 — "GPT Understands, Too") puts ``num_virtual_tokens``
learned embeddings in front of the input like prompt tuning, but produces them through an encoder,
an MLP or a bidirectional LSTM followed by one, so the tokens depend on each other.  The encoder is
only used while training: the file stores the prompt it produced.  Pick it over prompt tuning when
independent token embeddings train poorly.

The prompt is applied by hooks on the model's own call.  Here: ``PromptEncoder`` (what is learned)
and ``PromptEncoderModel`` (the tuner).  ``PromptEncoderConfig`` and
``PromptEncoderReparameterizationType`` are here too and exported from ``lucid.peft``.
"""

from lucid.peft.tuners.p_tuning._config import (
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
)
from lucid.peft.tuners.p_tuning._model import PromptEncoder, PromptEncoderModel

__all__ = [
    "PromptEncoder",
    "PromptEncoderConfig",
    "PromptEncoderModel",
    "PromptEncoderReparameterizationType",
]
