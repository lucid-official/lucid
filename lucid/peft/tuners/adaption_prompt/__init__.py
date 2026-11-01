r"""``lucid.peft.tuners.adaption_prompt`` — adaption prompts (LLaMA-Adapter): learned prompts the top attention layers attend to.

LLaMA-Adapter (Zhang et al., 2023, arXiv:2303.16199) freezes a decoder language model and gives
each of its top :math:`L` attention layers ``adapter_len`` learned prompt vectors.  The layer's
queries attend to keys and values that the layer's own projections make of the prompt, and the
result is added to the layer's output through a gate that starts at zero:
:math:`y = f(x) + g\cdot\mathrm{softmax}(q K_p^\top/\sqrt{d})\,V_p`.  The adapted model therefore
starts as the base, and the first gradients move the gate before the prompt matters.  Pick it to
steer a decoder with a few thousand parameters when the weights should stay untouched; unlike
prompt tuning it acts inside the layers, and unlike LoRA it cannot be folded into the weights.

It needs the attention's keys and values and its queries, which a module does not return, so each
supported family is one strategy: the zoo's ``gpt`` and ``gpt2`` decoders (fused ``c_attn``
projection).  Any other model type raises ``ValueError`` naming the supported ones.  ``AdaptionPromptConfig``
and ``AdaptionPromptModel`` are here and exported from ``lucid.peft``; ``AdaptedAttention`` is the
layer.  Importing the package registers the method under ``PeftType.ADAPTION_PROMPT``.
"""

from lucid.peft.tuners.adaption_prompt._config import AdaptionPromptConfig
from lucid.peft.tuners.adaption_prompt._layer import AdaptedAttention
from lucid.peft.tuners.adaption_prompt._model import AdaptionPromptModel

__all__ = ["AdaptedAttention", "AdaptionPromptConfig", "AdaptionPromptModel"]
