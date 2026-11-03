r"""ViT: separate ``query`` / ``key`` / ``value`` adapters → the zoo's one fused ``qkv``.

The reference library's ViT has three projections per attention block and an
adapter on each; the zoo's has one ``qkv`` ``nn.Linear`` whose output is the
three stacked.  A LoRA update :math:`s_i B_i A_i x` on each projection is the
update of one fused layer exactly, with :math:`r_g = \max_i r_i`:

.. math::

    A_f = \begin{bmatrix}A_q\\A_k\\A_v\end{bmatrix},\qquad
    B_f = \mathrm{blockdiag}(B_q, B_k, B_v)\ (\text{each }s_i/s_f\text{-scaled}),\qquad
    B_f A_f x = \begin{bmatrix}B_q A_q x\\B_k A_k x\\B_v A_v x\end{bmatrix}

a rank-:math:`3 r_g` layer.  A projection the adapter did not target gets zero
blocks; a rank smaller than :math:`r_g` is zero-padded.  The fused layer's
scaling :math:`s_f` is that of the first projection present, so in the usual
case (one :math:`\alpha`, one :math:`r`) no tensor is rescaled at all; where the
projections' scalings differ, the ratio :math:`s_i / s_f` is folded into
:math:`B_i`.  The layer's rank and :math:`\alpha` go into ``rank_pattern`` /
``alpha_pattern`` under its own name, so the other modules keep the config's.

The rest of the layout is renamed: ``encoder.layer.N.attention.output.dense`` →
``blocks.N.attn.proj``, ``intermediate.dense`` / ``output.dense`` →
``mlp.fc1`` / ``mlp.fc2``, ``layernorm_before`` / ``layernorm_after`` →
``norm1`` / ``norm2``, the head ``classifier`` → ``head``.

Both generations of the reference's ViT are read.  Through release 4 a block is
``encoder.layer.N`` with ``attention.attention.{query,key,value}``,
``attention.output.dense``, ``intermediate.dense`` and ``output.dense``; since release 5 it is
``layers.N`` with ``attention.{q_proj,k_proj,v_proj,o_proj}`` and ``mlp.{fc1,fc2}``.

Beside the factors, a ``lora_bias`` bias is stacked (zeros for an absent
projection) and a DoRA magnitude is stacked — both are per output row, so
stacking is exact.  An absent projection's DoRA magnitude is its base
weight's row norm (the unadapted layer), so that case needs the model.  Only
LoRA is fused: another method refuses, naming itself.
"""

import re
from typing import Final, override

from lucid.peft._convert._fused import _FusedQKVConverter

#: A block of the reference's ViT: ``encoder.layer.N`` through release 4, ``layers.N`` since 5.
_BLOCK: Final = r"(?:vit\.)?(?:encoder\.layer|layers)\.(?P<n>\d+)"
_RENAMES: Final = (
    (rf"{_BLOCK}\.attention\.(?:output\.dense|o_proj)", r"blocks.\g<n>.attn.proj"),
    (rf"{_BLOCK}\.(?:intermediate\.dense|mlp\.fc1)", r"blocks.\g<n>.mlp.fc1"),
    (rf"{_BLOCK}\.(?:output\.dense|mlp\.fc2)", r"blocks.\g<n>.mlp.fc2"),
    (rf"{_BLOCK}\.layernorm_before", r"blocks.\g<n>.norm1"),
    (rf"{_BLOCK}\.layernorm_after", r"blocks.\g<n>.norm2"),
    (r"(?:vit\.)?layernorm", "norm"),
    (r"(?:vit\.)?embeddings\.patch_embeddings\.projection", "patch_embed.proj"),
    (r"classifier", "head"),
)


class ViTConverter(_FusedQKVConverter, model_type="vit"):
    """ViT adapters: reference ``query`` / ``key`` / ``value`` projections fuse into ``blocks.N.attn.qkv``."""

    reference_classes = frozenset({"ViTModel", "ViTForImageClassification"})
    _FUSED = re.compile(
        rf"{_BLOCK}\.attention\.(?:attention\.)?(?P<role>query|key|value|q_proj|k_proj|v_proj)"
    )
    _COMPILED = tuple((re.compile(pattern), target) for pattern, target in _RENAMES)

    @override
    def _fused_name(self, match: re.Match[str]) -> str:
        return f"blocks.{match['n']}.attn.qkv"
