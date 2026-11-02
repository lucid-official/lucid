"""The modules an AdaLoRA adapter takes when its config does not say: a default per model type.

The reference keeps a table of its own for AdaLoRA: LoRA's for most model types, but wider
for the sequence-to-sequence family (every attention projection and the feed-forward
matrices, so the rank allocator has more triplets to trade between).  This is that table for
the model types Lucid's zoo registers: LoRA's rows, with ``t5`` widened and spelled as the
zoo's classes name the modules — checked against the zoo by
``lucid/test/unit/peft/test_reference_targets.py``.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from lucid.peft._target_defaults import DEFAULT_LORA_TARGET_MODULES

#: ``model_type`` → the module names an AdaLoRA adapter takes by default.  Read-only.
ADALORA_DEFAULT_TARGET_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {**DEFAULT_LORA_TARGET_MODULES, "t5": frozenset({"q", "k", "v", "o", "wi", "wo"})}
)
