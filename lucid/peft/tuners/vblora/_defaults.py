"""The modules a VB-LoRA adapter takes when its config does not say: a default per model type.

The reference keeps a table of its own for VB-LoRA — wider than LoRA's for
the sequence-to-sequence and OPT families (every attention projection and the
feed-forward matrices).  These are the rows of that table for the model types
Lucid's zoo has (``gpt2``, ``bert``, ``t5``), spelled as the zoo's classes name
the modules — each checked against the zoo by
``lucid/test/unit/peft/test_lora_family_zoo.py``.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

#: ``model_type`` → the module names a VB-LoRA adapter takes by default.  Read-only.
VBLORA_DEFAULT_TARGET_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "gpt2": frozenset({"c_attn"}),
        "bert": frozenset({"query", "value"}),
        "t5": frozenset({"q", "k", "v", "o", "wi", "wo"}),
    }
)
