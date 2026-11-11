"""The modules an OSF adapter takes when its config does not say: a default per model type.

The reference keeps a table of the projections to split, keyed by the base model's ``model_type``.
These are the rows for the model types Lucid's zoo shares with it (``gpt2``), spelled as the zoo's
classes name the modules.  A model type with no row is not an error, as it is for most methods:
OSF adapts every linear layer then (``"all-linear"``), as the reference does.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

#: ``model_type`` → the module names an OSF adapter takes by default.  Read-only.
OSF_DEFAULT_TARGET_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {"gpt2": frozenset({"c_attn", "c_proj"})}
)
