"""The modules a FourierFT adapter takes when its config does not say: a default per model type.

The reference keeps one table for FourierFT — its LoRA table, with ``gpt2``
(and ``gpt_bigcode``) narrowed to the feed-forward output projection.  This is
that table for the model types Lucid's zoo registers: LoRA's rows, spelled as
the zoo names the modules, with ``gpt2`` replaced.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from lucid.peft._target_defaults import DEFAULT_LORA_TARGET_MODULES

#: ``model_type`` → the module names a FourierFT adapter takes by default.  Read-only.
FOURIERFT_DEFAULT_TARGET_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {**DEFAULT_LORA_TARGET_MODULES, "gpt2": frozenset({"mlp.c_proj"})}
)
