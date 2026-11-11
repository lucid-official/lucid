"""The modules a WaveFT adapter takes when its config does not say: LoRA's attention projections, with ``gpt2`` narrowed.

The reference keeps a table of its own for WaveFT.  For the model types Lucid's zoo registers it is
LoRA's, except that ``gpt2`` (like ``gpt_bigcode``) takes the feed-forward output projection
``mlp.c_proj`` instead of the fused attention projection.  The zoo's names for those modules are the
reference's, so the rows are LoRA's rows and the one replacement.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from lucid.peft._target_defaults import DEFAULT_LORA_TARGET_MODULES

#: ``model_type`` → the module names a WaveFT adapter takes by default.  Read-only.
WAVEFT_DEFAULT_TARGET_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {**DEFAULT_LORA_TARGET_MODULES, "gpt2": frozenset({"mlp.c_proj"})}
)
