"""CLIP: separate ``q_proj`` / ``k_proj`` / ``v_proj`` / ``out_proj`` → ``nn.MultiheadAttention``.

The reference library's ``CLIPModel`` / ``CLIPVisionModel`` / ``CLIPTextModel``
organises each encoder block under ``{vision,text}_model.encoder.layers.N`` with
separate ``self_attn.{q_proj,k_proj,v_proj,out_proj}`` (or ``o_proj`` in release 5)
and ``mlp.{fc1,fc2}``.  The zoo's ``CLIP`` places its towers at ``visual`` and
``textual``, each holding ``transformer.resblocks.N`` with an ``nn.MultiheadAttention``
at ``attn`` and a sequential MLP at ``mlp.{0,2}``.
"""

import re
from typing import Final, override

from lucid.peft._convert._fused import _FusedMhaConverter

_TOWER_MAP: Final = {"vision": "visual", "text": "textual"}
_VIS_BLOCK: Final = r"(?:clip\.)?vision_model\.encoder\.layers\.(?P<n>\d+)"
_TXT_BLOCK: Final = r"(?:clip\.)?text_model\.encoder\.layers\.(?P<n>\d+)"
_RENAMES: Final = (
    (rf"{_VIS_BLOCK}\.mlp\.fc1", r"visual.transformer.resblocks.\g<n>.mlp.0"),
    (rf"{_VIS_BLOCK}\.mlp\.fc2", r"visual.transformer.resblocks.\g<n>.mlp.2"),
    (rf"{_VIS_BLOCK}\.layer_norm1", r"visual.transformer.resblocks.\g<n>.ln_1"),
    (rf"{_VIS_BLOCK}\.layer_norm2", r"visual.transformer.resblocks.\g<n>.ln_2"),
    (rf"{_TXT_BLOCK}\.mlp\.fc1", r"textual.transformer.resblocks.\g<n>.mlp.0"),
    (rf"{_TXT_BLOCK}\.mlp\.fc2", r"textual.transformer.resblocks.\g<n>.mlp.2"),
    (rf"{_TXT_BLOCK}\.layer_norm1", r"textual.transformer.resblocks.\g<n>.ln_1"),
    (rf"{_TXT_BLOCK}\.layer_norm2", r"textual.transformer.resblocks.\g<n>.ln_2"),
    (r"(?:clip\.)?vision_model\.embeddings\.patch_embedding", "visual.conv1"),
    (r"(?:clip\.)?vision_model\.pre_laye?rnorm", "visual.ln_pre"),
    (r"(?:clip\.)?vision_model\.post_layernorm", "visual.ln_post"),
    (r"(?:clip\.)?text_model\.embeddings\.token_embedding", "textual.token_embedding"),
    (r"(?:clip\.)?text_model\.final_layer_norm", "textual.ln_final"),
)


class CLIPConverter(_FusedMhaConverter, model_type="clip"):
    """CLIP adapters: ``self_attn.{q,k,v,out}_proj`` fuse into ``{visual,textual}.transformer.resblocks.N.attn``."""

    reference_classes = frozenset(
        {
            "CLIPModel",
            "CLIPVisionModel",
            "CLIPVisionModelWithProjection",
            "CLIPTextModel",
            "CLIPTextModelWithProjection",
            "CLIPForImageClassification",
        }
    )
    _FUSED = re.compile(
        r"(?:clip\.)?(?P<tower>vision|text)_model\.encoder\.layers\.(?P<n>\d+)"
        r"\.self_attn\.(?P<role>q_proj|k_proj|v_proj|out_proj|o_proj)"
    )
    _COMPILED = tuple((re.compile(pattern), target) for pattern, target in _RENAMES)

    @override
    def _fused_name(self, match: re.Match[str]) -> str:
        tower = _TOWER_MAP[match["tower"]]
        return f"{tower}.transformer.resblocks.{match['n']}.attn"
