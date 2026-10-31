"""The modules a C3A adapter takes when its config does not say: a default per model type.

The reference keeps one table for C3A — its LoRA table, with ``gpt2`` narrowed to
the feed-forward output projection (a block size that divides the wider attention
sizes fits fewer GPT-2 layers).  This is that table for the model types Lucid's zoo
registers: LoRA's rows, spelled as the zoo names the modules, with ``gpt2`` replaced.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from lucid.peft._target_defaults import DEFAULT_LORA_TARGET_MODULES

#: ``model_type`` → the module names a C3A adapter takes by default.  Read-only.
C3A_DEFAULT_TARGET_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {**DEFAULT_LORA_TARGET_MODULES, "gpt2": frozenset({"mlp.c_proj"})}
)
