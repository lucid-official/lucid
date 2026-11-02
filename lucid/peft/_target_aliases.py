"""The names the reference library's models give a module, for the zoo's classes that name it otherwise.

A script written for the reference library says ``target_modules=["query", "value"]``
for a ViT, ``["q_proj", "v_proj"]`` for a CLIP, ``["to_q", "to_v"]`` for a diffusion
U-Net: the names *its* classes use.  Most of the zoo's text models (``gpt``, ``gpt2``,
``bert``, ``roformer``, ``t5``) are named alike, so those scripts already select the
same modules.  Where the zoo's class lays a module out otherwise (a ViT's one fused
``qkv`` for the reference's three projections, a CLIP's single
:class:`~lucid.nn.MultiheadAttention`), a name the model lacks would match nothing and the
adapter would be refused.  This table says, per ``model_type``, which zoo module a
reference name stands for.

An alias only fills a name that selects **no** module of the model: a name the model has is
taken as it is, so a zoo name always wins.  An alias can be wider than the name it stands for
(``query`` → the fused ``qkv`` adapts the key too), which is why using one warns, and the
config stores the zoo's names, so a saved adapter reads the same everywhere.

Only names the reference's classes really use are listed; a model type with no counterpart
there (``transformer``, the convolutional families) has no row.
"""

import warnings
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final

import lucid.nn as nn
from lucid.peft._target_defaults import _model_type
from lucid.peft._targeting import _module_names_matching

#: ``model_type`` → reference module name → the zoo module names that play its part.  Read-only.
REFERENCE_TARGET_ALIASES: Final[Mapping[str, Mapping[str, tuple[str, ...]]]] = MappingProxyType(
    {
        "t5": MappingProxyType(
            {"layer_norm": ("self_attention_norm", "cross_attention_norm", "feed_forward_norm")}
        ),
        "vit": MappingProxyType(
            {
                "query": ("qkv",),
                "key": ("qkv",),
                "value": ("qkv",),
                "attention.output.dense": ("attn.proj",),
                "intermediate.dense": ("mlp.fc1",),
                "output.dense": ("attn.proj", "mlp.fc2"),
                "dense": ("attn.proj", "mlp.fc1", "mlp.fc2"),
            }
        ),
        "swin": MappingProxyType(
            {
                "query": ("qkv",),
                "key": ("qkv",),
                "value": ("qkv",),
                "attention.output.dense": ("attn.proj",),
                "intermediate.dense": ("mlp.0",),
                "output.dense": ("attn.proj", "mlp.3"),
                "dense": ("attn.proj", "mlp.0", "mlp.3"),
            }
        ),
        "clip": MappingProxyType(
            {
                "q_proj": ("attn",),
                "k_proj": ("attn",),
                "v_proj": ("attn",),
                "out_proj": ("attn",),
                "fc1": ("mlp.0",),
                "fc2": ("mlp.2",),
            }
        ),
        "detr": MappingProxyType(
            {
                "q_proj": ("self_attn", "multihead_attn"),
                "k_proj": ("self_attn", "multihead_attn"),
                "v_proj": ("self_attn", "multihead_attn"),
                "out_proj": ("self_attn", "multihead_attn"),
                "encoder_attn": ("multihead_attn",),
                "fc1": ("linear1",),
                "fc2": ("linear2",),
            }
        ),
        "maskformer": MappingProxyType({"out_proj": ("o_proj",)}),
        "stable_diffusion": MappingProxyType(
            {
                "to_q": ("attn1", "attn2"),
                "to_k": ("attn1", "attn2"),
                "to_v": ("attn1", "attn2"),
                "to_out.0": ("attn1", "attn2"),
            }
        ),
        "dit": MappingProxyType(
            {"to_q": ("attn",), "to_k": ("attn",), "to_v": ("attn",), "to_out.0": ("attn",)}
        ),
    }
)


def _selects_a_module(model: nn.Module, name: str) -> bool:
    return bool(_module_names_matching(model, [name]))


def _zoo_target_names(model: nn.Module, names: Iterable[str]) -> set[str]:
    """``names`` with each one that selects no module of ``model`` replaced by the zoo's names for it.

    A name that selects a module stays; one that selects none and has no alias for
    ``model``'s type, or whose aliases select nothing either, stays too, so the caller's
    "target modules not found" still names what the user wrote.  Using an alias warns.
    """
    table = REFERENCE_TARGET_ALIASES.get(_model_type(model), MappingProxyType({}))
    chosen: set[str] = set()
    used: dict[str, tuple[str, ...]] = {}
    for name in names:
        found = tuple(a for a in table.get(name, ()) if _selects_a_module(model, a))
        if found and not _selects_a_module(model, name):
            used[name] = found
            chosen.update(found)
        else:
            chosen.add(name)
    if used:
        shown = ", ".join(f"{name!r} -> {list(found)}" for name, found in sorted(used.items()))
        warnings.warn(
            f"target_modules names no module of this {_model_type(model)!r} model under the "
            f"reference library's names; using the zoo's modules for them: {shown}.  A zoo "
            "module can cover more than the name it stands for (a fused qkv, a whole attention "
            "block); name the modules directly to choose.",
            stacklevel=2,
        )
    return chosen
