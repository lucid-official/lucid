"""The modules an LN-tuning adapter takes when its config does not say: a default per model type.

The reference keeps one table of the normalisation layers to tune, keyed by the
base model's ``model_type``.  These are the rows for the model types Lucid's
zoo shares with it (``gpt2``, ``t5``), spelled as the zoo's classes name the
modules — each checked against the zoo by ``lucid/test/unit/peft/test_ln_tuning.py``.
A model type with no row raises: the reference names no norms for ``bert`` or
``vit`` either, so the config must.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

#: ``model_type`` → the module names an LN-tuning adapter takes by default.  Read-only.
LN_TUNING_DEFAULT_TARGET_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "gpt2": frozenset({"ln_1", "ln_2", "ln_f"}),
        "t5": frozenset(
            {
                "self_attention_norm",
                "cross_attention_norm",
                "feed_forward_norm",
                "final_layer_norm",
            }
        ),
    }
)
