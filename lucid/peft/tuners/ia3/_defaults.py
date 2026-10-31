"""The modules an (IA)^3 adapter takes when its config does not say: a default per model type.

(IA)^3 scales the keys and values of attention and the output of the
feed-forward block, and the feed-forward modules scale their *input*
instead (``feedforward_modules``).  The reference keeps one table of the
modules to adapt and one of the feed-forward subset, both by the base
model's ``model_type``; these are those rows for the model types Lucid's zoo
has (``gpt2``, ``bert``, ``t5``), spelled as the zoo's classes name the
modules — each checked against the zoo by ``lucid/test/unit/peft/test_new_tuners_zoo.py``.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import lucid.nn as nn
from lucid.peft._target_defaults import _model_type

#: ``model_type`` → the module names an (IA)^3 adapter takes by default.  Read-only.
IA3_DEFAULT_TARGET_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "gpt2": frozenset({"c_attn", "mlp.c_proj"}),
        "bert": frozenset({"key", "value", "output.dense"}),
        "t5": frozenset({"k", "v", "wo"}),
    }
)

#: ``model_type`` → which of those are feed-forward modules (scaled on the input).  Read-only.
IA3_DEFAULT_FEEDFORWARD_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "gpt2": frozenset({"mlp.c_proj"}),
        "bert": frozenset({"output.dense"}),
        "t5": frozenset({"wo"}),
    }
)


def _default_feedforward_modules(model: nn.Module) -> frozenset[str]:
    """The feed-forward names for ``model``'s type; ``ValueError`` if the table has no row."""
    model_type = _model_type(model)
    names = IA3_DEFAULT_FEEDFORWARD_MODULES.get(model_type)
    if names is None:
        raise ValueError(
            "Please specify `feedforward_modules` in `peft_config`: a list of module names (the "
            "subset of `target_modules` that scale their input; [] for none) or a regex.  There "
            f"is no default for model type {model_type!r} (the defaults cover "
            f"{', '.join(sorted(IA3_DEFAULT_FEEDFORWARD_MODULES))})."
        )
    return names
