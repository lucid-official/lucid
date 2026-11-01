r"""``lucid.peft.tuners.xlora`` — X-LoRA: a mixture of LoRA experts weighed per token by a learned classifier.

X-LoRA (Buehler & Buehler, 2024, arXiv:2402.07148) starts from several LoRA adapters trained on
different tasks and learns only how to combine them.  A scalings pass runs the model with the
experts held at a fixed weight; a small classifier on its final hidden states then predicts, for
every token (and, if asked, every adapted layer), one weight per expert; the real pass adds each
expert's low-rank update on the input scaled by its weight, optionally keeping only the top-k.
Pick it to specialise a model on a mix of skills without choosing one expert up front, and to
train a few thousand classifier parameters rather than the experts.

The experts are loaded by name through the multi-adapter API (``XLoraModel.lora_model``), the
model is called as usual, and the adapter file holds the classifier alone
(``internal_xlora_classifier.*``) — the experts stay where they were saved, named by
``XLoraConfig.adapters``.  The weights come from the whole sequence, so generate with
``use_cache=False``; nothing merges.  ``XLoraConfig`` and ``XLoraModel`` are here and exported from
``lucid.peft``; ``XLoraClassifier`` is the classifier.  Importing the package registers the method
under ``PeftType.XLORA``.
"""

from lucid.peft.tuners.xlora._classifier import XLoraClassifier
from lucid.peft.tuners.xlora._config import XLoraConfig
from lucid.peft.tuners.xlora._model import XLoraModel

__all__ = ["XLoraClassifier", "XLoraConfig", "XLoraModel"]
