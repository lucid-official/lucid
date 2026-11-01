"""``lucid.peft.tuners.prompt_tuning`` — prompt tuning: learned embeddings in front of the input.

Prompt tuning (Lester et al., 2021, arXiv:2104.08691) freezes the model and learns
``num_virtual_tokens`` embedding vectors that are put in front of the input embeddings.  The only
trained parameters are ``num_virtual_tokens * token_dim`` numbers, so a task is a file of that size.
The prompt starts from text, from sampled vocabulary rows or at random (``PromptTuningInit``).
Pick it for the smallest per-task files on a large model; it adds no layer, and trains one prompt at
a time.

The prompt is applied by hooks on the model's own call.  Here: ``PromptEmbedding`` (what is
learned) and ``PromptTuningModel`` (the tuner).  ``PromptTuningConfig`` and ``PromptTuningInit`` are
here too and exported from ``lucid.peft``.
"""

from lucid.peft.tuners.prompt_tuning._config import PromptTuningConfig, PromptTuningInit
from lucid.peft.tuners.prompt_tuning._model import PromptEmbedding, PromptTuningModel

__all__ = ["PromptEmbedding", "PromptTuningConfig", "PromptTuningInit", "PromptTuningModel"]
