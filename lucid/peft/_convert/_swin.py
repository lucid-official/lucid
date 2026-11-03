"""Swin: separate ``query`` / ``key`` / ``value`` adapters → the zoo's fused ``stages.S.blocks.B.attn.qkv``.

Both generations of the reference library's Swin are read: through release 4 a
block's projections are ``attention.self.{query,key,value}``,
``attention.output.dense``, ``intermediate.dense`` and ``output.dense``; since
release 5 they are ``attention.{q_proj,k_proj,v_proj,o_proj}`` and ``mlp.{fc1,fc2}``.
In the zoo, ``SwinTransformerBlock`` holds ``attn.qkv``, ``attn.proj`` and
``mlp = nn.Sequential(Linear, GELU, Dropout, Linear, Dropout)`` (so ``fc1`` is
``mlp.0`` and ``fc2`` is ``mlp.3``), and ``classifier`` keeps its name.
"""

import re
from typing import Final, override

from lucid.peft._convert._fused import _FusedQKVConverter

_STAGE: Final = r"(?:swin\.)?encoder\.layers\.(?P<s>\d+)"
_BLOCK: Final = rf"{_STAGE}\.blocks\.(?P<b>\d+)"
_RENAMES: Final = (
    (rf"{_BLOCK}\.attention\.(?:output\.dense|o_proj)", r"stages.\g<s>.blocks.\g<b>.attn.proj"),
    (rf"{_BLOCK}\.(?:intermediate\.dense|mlp\.fc1)", r"stages.\g<s>.blocks.\g<b>.mlp.0"),
    (rf"{_BLOCK}\.(?:output\.dense|mlp\.fc2)", r"stages.\g<s>.blocks.\g<b>.mlp.3"),
    (rf"{_BLOCK}\.layernorm_before", r"stages.\g<s>.blocks.\g<b>.norm1"),
    (rf"{_BLOCK}\.layernorm_after", r"stages.\g<s>.blocks.\g<b>.norm2"),
    (rf"{_STAGE}\.downsample\.reduction", r"stages.\g<s>.downsample.proj"),
    (rf"{_STAGE}\.downsample\.norm", r"stages.\g<s>.downsample.norm"),
    (r"(?:swin\.)?embeddings\.patch_embeddings\.projection", "patch_embed.proj"),
    (r"(?:swin\.)?embeddings\.norm", "patch_embed.norm"),
    (r"(?:swin\.)?layernorm", "norm"),
    (r"classifier", "classifier"),
)


class SwinConverter(_FusedQKVConverter, model_type="swin"):
    """Swin adapters: reference ``query`` / ``key`` / ``value`` projections fuse into ``stages.S.blocks.B.attn.qkv``."""

    reference_classes = frozenset(
        {
            "SwinModel",
            "SwinForImageClassification",
            "SwinForMaskedImageModeling",
            "SwinBackbone",
        }
    )
    _FUSED = re.compile(
        rf"{_BLOCK}\.attention\.(?:self\.)?(?P<role>query|key|value|q_proj|k_proj|v_proj)"
    )
    _COMPILED = tuple((re.compile(pattern), target) for pattern, target in _RENAMES)

    @override
    def _fused_name(self, match: re.Match[str]) -> str:
        return f"stages.{match['s']}.blocks.{match['b']}.attn.qkv"
