"""The modules an adapter takes when its config does not say: a default per model type.

``LoraConfig()`` names no ``target_modules``.  The reference library
answers from a table keyed by the base model's ``config.model_type`` —
attention's query and value projections, by the names that architecture
gives them — and refuses a model type it has no row for.  This is that
table for the model types Lucid's zoo has.  The three the reference lists
(``t5``, ``gpt2``, ``bert``) keep its choice; the others take the same
projections under the zoo's own names, each checked against the zoo class
by ``lucid/test/unit/models/test_peft_zoo.py``.

``nn.MultiheadAttention`` holds its projections as raw parameters, which
LoRA on ``nn.Linear`` cannot reach, so the families built on it
(``transformer``, ``clip``, ``dit``, ``detr``) have no row.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import lucid.nn as nn

#: ``model_type`` → the module names a LoRA adapter takes by default.  Read-only.
DEFAULT_LORA_TARGET_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "gpt": frozenset({"c_attn"}),
        "gpt2": frozenset({"c_attn"}),
        "bert": frozenset({"query", "value"}),
        "roformer": frozenset({"query", "value"}),
        "t5": frozenset({"q", "v"}),
        "vit": frozenset({"qkv"}),
        "swin": frozenset({"qkv"}),
        "pvt": frozenset({"q", "kv"}),
    }
)

#: The model type of a module that carries none, as the reference spells it.
_UNKNOWN_MODEL_TYPE: Final = "custom"


def _model_type(model: nn.Module) -> str:
    """``model.config.model_type``, or ``"custom"`` for a module without one."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return model_type if isinstance(model_type, str) else _UNKNOWN_MODEL_TYPE


def _default_target_modules(
    model: nn.Module, table: Mapping[str, frozenset[str]]
) -> frozenset[str]:
    """The names ``table`` gives ``model``'s type; ``ValueError`` if it has no row."""
    model_type = _model_type(model)
    names = table.get(model_type)
    if names is None:
        raise ValueError(
            "Please specify `target_modules` in `peft_config`: a list of module names, a "
            f"regex, or 'all-linear'.  There is no default for model type {model_type!r} "
            f"(the defaults cover {', '.join(sorted(table))})."
        )
    return names
