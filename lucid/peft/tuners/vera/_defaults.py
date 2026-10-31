"""The modules a VeRA adapter takes when its config does not say: a default per model type.

The reference keeps one table for VeRA — its LoRA table, with the projections of
``phi`` narrowed — keyed by the base model's ``model_type``.  These are the rows
of that table for the model types Lucid's zoo has (``gpt2``, ``bert``, ``t5``),
spelled as the zoo's classes name the modules — each checked against the zoo by
``lucid/test/unit/peft/test_vera.py``.  Unlike LoRA's, the table has no row for
a convolutional model: VeRA adapts ``nn.Linear`` only.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

#: ``model_type`` → the module names a VeRA adapter takes by default.  Read-only.
VERA_DEFAULT_TARGET_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "gpt2": frozenset({"c_attn"}),
        "bert": frozenset({"query", "value"}),
        "t5": frozenset({"q", "v"}),
    }
)
