r"""``lucid.peft.tuners.cpt`` — context-aware prompt tuning: a prompt of in-context examples with a bounded learned offset.

CPT (Liu et al., 2024, arXiv:2410.17222) tunes a prompt that is itself a few in-context examples.
Its tokens start as the embeddings of ``cpt_token_ids`` and each gets a trainable delta, held near
zero by a projection after every step (templates and inputs may move, outputs may not), so the
examples keep their meaning while the prompt adapts.  The loss counts the output tokens of the
prompt as well as of the batch, optionally weighting earlier examples less.  Pick it for few-shot
fine-tuning of a causal language model when the examples should stay in context; for a free
prompt that starts anywhere, use prompt tuning.

Like the other prompt methods it adds no layer: the prompt is put in front of the input by hooks
on the model's call, so ``model(...)`` and ``generate`` both carry it, and the adapter file holds
the finished prompt as ``prompt_embeddings``.  ``CPTConfig`` and ``CPTModel`` are here and exported
from ``lucid.peft``; ``CPTEmbedding`` is the learned part.  Importing the package registers the
method under ``PeftType.CPT``.
"""

from lucid.peft.tuners.cpt._config import CPTConfig
from lucid.peft.tuners.cpt._embedding import CPTEmbedding
from lucid.peft.tuners.cpt._model import CPTModel

__all__ = ["CPTConfig", "CPTEmbedding", "CPTModel"]
