r"""DETR: separate ``q_proj`` / ``k_proj`` / ``v_proj`` / ``out_proj`` → ``nn.MultiheadAttention``.

The reference library's ``DetrModel`` / ``DetrForObjectDetection`` places its
transformer stacks at ``(?:model\.)?{encoder,decoder}.layers.N`` with separate
``self_attn.{q_proj,k_proj,v_proj,out_proj}`` (or ``o_proj`` in release 5),
``encoder_attn.{q_proj,k_proj,v_proj,out_proj}`` on decoder blocks, and feed-forward
layers ``fc1`` / ``fc2`` (release 4) or ``mlp.fc1`` / ``mlp.fc2`` (release 5).
In the zoo's ``DETR``, those live under ``transformer.{encoder,decoder}.layers.N``
as ``self_attn``, ``multihead_attn``, ``linear1`` and ``linear2``.
"""

import re
from typing import Final, override

from lucid.peft._convert._fused import _FusedMhaConverter

_ENC_BLOCK: Final = r"(?:model\.)?encoder\.layers\.(?P<n>\d+)"
_DEC_BLOCK: Final = r"(?:model\.)?decoder\.layers\.(?P<n>\d+)"
_ATTN_MAP: Final = {"self_attn": "self_attn", "encoder_attn": "multihead_attn"}
_BBOX_MAP: Final = {"0": "0", "1": "2", "2": "4"}
_RENAMES: Final = (
    (rf"{_ENC_BLOCK}\.(?:mlp\.)?fc1", r"transformer.encoder.layers.\g<n>.linear1"),
    (rf"{_ENC_BLOCK}\.(?:mlp\.)?fc2", r"transformer.encoder.layers.\g<n>.linear2"),
    (rf"{_ENC_BLOCK}\.self_attn_layer_norm", r"transformer.encoder.layers.\g<n>.norm1"),
    (rf"{_ENC_BLOCK}\.final_layer_norm", r"transformer.encoder.layers.\g<n>.norm2"),
    (rf"{_DEC_BLOCK}\.(?:mlp\.)?fc1", r"transformer.decoder.layers.\g<n>.linear1"),
    (rf"{_DEC_BLOCK}\.(?:mlp\.)?fc2", r"transformer.decoder.layers.\g<n>.linear2"),
    (rf"{_DEC_BLOCK}\.self_attn_layer_norm", r"transformer.decoder.layers.\g<n>.norm1"),
    (rf"{_DEC_BLOCK}\.encoder_attn_layer_norm", r"transformer.decoder.layers.\g<n>.norm2"),
    (rf"{_DEC_BLOCK}\.final_layer_norm", r"transformer.decoder.layers.\g<n>.norm3"),
    (r"(?:model\.)?decoder\.layernorm", "transformer.decoder.norm"),
    (r"(?:model\.)?input_projection", "input_proj"),
    (r"(?:model\.)?query_position_embeddings", "query_embed"),
    (r"class_labels_classifier", "class_embed"),
    (r"bbox_predictor\.layers\.0", "bbox_embed.net.0"),
    (r"bbox_predictor\.layers\.1", "bbox_embed.net.2"),
    (r"bbox_predictor\.layers\.2", "bbox_embed.net.4"),
)


class DETRConverter(_FusedMhaConverter, model_type="detr"):
    """DETR adapters: ``{self,encoder}_attn.{q,k,v,out}_proj`` fuse into ``transformer.{encoder,decoder}.layers.N.{self_attn,multihead_attn}``."""

    reference_classes = frozenset({"DetrModel", "DetrForObjectDetection", "DetrForSegmentation"})
    _FUSED = re.compile(
        r"(?:model\.)?(?:(?P<enc>encoder)\.layers\.(?P<en>\d+)\.(?P<eattn>self_attn)"
        r"|(?P<dec>decoder)\.layers\.(?P<dn>\d+)\.(?P<dattn>self_attn|encoder_attn))"
        r"\.(?P<role>q_proj|k_proj|v_proj|out_proj|o_proj)"
    )
    _COMPILED = tuple((re.compile(pattern), target) for pattern, target in _RENAMES)

    @override
    def _fused_name(self, match: re.Match[str]) -> str:
        if match["enc"] is not None:
            return f"transformer.encoder.layers.{match['en']}.self_attn"
        attn = _ATTN_MAP[match["dattn"]]
        return f"transformer.decoder.layers.{match['dn']}.{attn}"
