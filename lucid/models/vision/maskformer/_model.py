"""MaskFormer (Cheng et al., NeurIPS 2021).

Paper: "Per-Pixel Classification is Not All You Need for Semantic Segmentation"

Key innovation
--------------
MaskFormer reformulates semantic segmentation as **mask classification**:

1. A CNN backbone + FPN pixel decoder extracts per-pixel embeddings.
2. N learnable queries attend to the backbone's last feature map via a
   DETR-style Transformer decoder.
3. Two prediction heads per query:
   - Class head: Linear(d_model, K+1) — class distribution (K+1 for no-object).
   - Mask head: a 3-layer MLP producing a per-query mask embedding, then a
     dot product with the per-pixel embedding → binary mask logits
     (B, N, H/4, W/4).
4. Training: Hungarian matching (mask IoU + cross-entropy) + BCE mask loss +
   class CE loss on matched pairs.
5. Inference (semantic): drop the no-object slot, then
   ``softmax(class)[..., :-1] ⊗ sigmoid(mask)`` summed over queries → per-pixel
   class scores, upsampled to (H, W).

Reference-faithful layout
-------------------------
The submodule names mirror the reference framework's MaskFormer verbatim so
the pretrained-weight converter is a near-identity key map:

  * ``pixel_level_module.encoder`` — a HF-style ResNet
    (``embedder.embedder`` stem + ``encoder.stages.{s}.layers.{l}``
    bottlenecks; ``shortcut`` / ``layer.{0,1,2}`` each a
    ``convolution`` + regular ``normalization`` BatchNorm).
  * ``pixel_level_module.decoder`` — the FPN pixel decoder
    (``fpn.stem`` + ``fpn.layers.{i}`` with ``proj`` / ``block`` conv-GN
    pairs) + a final ``mask_projection`` 3x3 conv.
  * ``transformer_module`` — ``queries_embedder`` (nn.Embedding),
    ``input_projection`` (1x1 conv, C5 → d_model) and a 6-layer
    ``decoder`` with separate ``q/k/v/o_proj`` attentions
    (``self_attn`` + ``encoder_attn``), ``mlp.fc1`` / ``mlp.fc2`` and a
    trailing ``decoder.layernorm``.
  * ``class_predictor`` (Linear → K+1) and ``mask_embedder.{0,1,2}.0``
    (3-layer MLP mask head).
"""

from typing import ClassVar, cast, final, override

from dataclasses import dataclass

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._tasks import SemanticSegmentationModel
from lucid.models._output import SemanticSegmentationOutput
from lucid.models.vision._common._position import SinePositionEmbedding
from lucid.models.vision.maskformer._config import MaskFormerConfig

# Relative weight of the "no object" class in the classification loss —
# paper §4.1 (following DETR), and ``NO_OBJECT_WEIGHT: 0.1`` in the official
# configs.
_NO_OBJECT_WEIGHT = 0.1

# Section 4.1 / the official weight_dict: the mask term is 20x the class term
# and dice is 1x.  An equal 1:1:1 balance under-weights the mask objective by
# a factor of 20.
_MASK_WEIGHT: float = 20.0
_DICE_WEIGHT: float = 1.0

# Section 4.1 / the released criterion: the mask term is sigmoid *focal*
# loss, not plain BCE.  Its defaults come from RetinaNet.
_FOCAL_ALPHA: float = 0.25
_FOCAL_GAMMA: float = 2.0

# ---------------------------------------------------------------------------
# ResNet backbone (HF-style key layout, regular BatchNorm)
# ---------------------------------------------------------------------------


@final
class _ConvNorm(nn.Module):
    """``convolution`` + ``normalization`` (BatchNorm) with optional ReLU.

    Mirrors the reference ResNet ``ResNetConvLayer`` key layout so the
    converter is a pure prefix strip.  ``normalization`` is a regular
    :class:`nn.BatchNorm2d` (trainable, eval at inference) — *not* a
    frozen variant.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(out_ch)
        self._act = activation

    @override
    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.normalization(self.convolution(x))
        if self._act:
            out = F.relu(out)
        return out


@final
class _ResNetEmbedder(nn.Module):
    """Stem: a 7x7 stride-2 conv-BN-ReLU followed by a 3x3 stride-2 maxpool.

    Key layout ``embedder.{convolution,normalization}`` mirrors the
    reference ``ResNetEmbeddings.embedder``.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.embedder = _ConvNorm(in_channels, 64, 7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.embedder(x)
        return self.pool(x)


class _ResNetBottleneck(nn.Module):
    """Bottleneck block with stride on the 3x3 conv (v1.5 / reference layout).

    Holds an optional ``shortcut`` (1x1 conv-BN downsample) and a 3-conv
    ``layer`` stack (``layer.0`` 1x1, ``layer.1`` 3x3 stride, ``layer.2``
    1x1).  The final ReLU is applied after the residual add.
    """

    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        stride: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.shortcut: _ConvNorm | None
        if downsample:
            self.shortcut = _ConvNorm(in_ch, out_ch, 1, stride=stride, activation=False)
        else:
            self.shortcut = None
        self.layer = nn.Sequential(
            _ConvNorm(in_ch, mid_ch, 1),
            _ConvNorm(mid_ch, mid_ch, 3, stride=stride, padding=1),
            _ConvNorm(mid_ch, out_ch, 1, activation=False),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        out: Tensor = self.layer(x)
        return F.relu(out + identity)


@final
class _ResNetStage(nn.Module):
    """A stack of bottleneck ``layers``; the first carries the stride/downsample."""

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        num_blocks: int,
        stride: int,
    ) -> None:
        super().__init__()
        out_ch = mid_ch * _ResNetBottleneck.expansion
        blocks: list[nn.Module] = [_ResNetBottleneck(in_ch, mid_ch, stride=stride, downsample=True)]
        for _ in range(1, num_blocks):
            blocks.append(_ResNetBottleneck(out_ch, mid_ch))
        self.layers = nn.Sequential(*blocks)
        self.out_channels: int = out_ch

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


@final
class _ResNetEncoderStages(nn.Module):
    """The four bottleneck ``stages`` of a ResNet trunk."""

    def __init__(self, layers: tuple[int, int, int, int]) -> None:
        super().__init__()
        # Reference ResNet: stage 0 keeps stride 1 (downsample for channel
        # change only), stages 1-3 each halve resolution.
        s0 = _ResNetStage(64, 64, layers[0], stride=1)
        s1 = _ResNetStage(s0.out_channels, 128, layers[1], stride=2)
        s2 = _ResNetStage(s1.out_channels, 256, layers[2], stride=2)
        s3 = _ResNetStage(s2.out_channels, 512, layers[3], stride=2)
        self.stages: nn.ModuleList[_ResNetStage] = nn.ModuleList([s0, s1, s2, s3])
        self.out_channels: list[int] = [
            s0.out_channels,
            s1.out_channels,
            s2.out_channels,
            s3.out_channels,
        ]

    @override
    def forward(self, x: Tensor) -> list[Tensor]:
        feats: list[Tensor] = []
        out = x
        for stage in self.stages:
            out = stage(out)
            feats.append(out)
        return feats


@final
class _ResNetBackbone(nn.Module):
    """ResNet backbone returning ``[C2, C3, C4, C5]`` feature maps.

    Key layout (``embedder.embedder`` + ``encoder.stages.{s}.layers.{l}``)
    mirrors the reference ``ResNetBackbone`` so the converter is a pure
    prefix strip.
    """

    def __init__(
        self,
        in_channels: int,
        layers: tuple[int, int, int, int],
    ) -> None:
        super().__init__()
        self.embedder = _ResNetEmbedder(in_channels)
        self.encoder = _ResNetEncoderStages(layers)
        self.out_channels: list[int] = self.encoder.out_channels

    @override
    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.embedder(x)
        return self.encoder(x)


# ---------------------------------------------------------------------------
# FPN Pixel Decoder (reference key layout: conv + GroupNorm)
# ---------------------------------------------------------------------------


@final
class _FPNConvLayer(nn.Sequential):
    """3x3 conv → GroupNorm(32) → ReLU; reference ``MaskFormerFPNConvLayer``.

    Subclasses :class:`nn.Sequential` so the persistent keys are ``0``
    (conv) / ``1`` (GroupNorm) — matching the reference
    ``add_module(str(i), ...)`` naming (ReLU has no parameters).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )


@final
class _FPNLayer(nn.Module):
    """One FPN merge step: lateral ``proj`` + upsample-add + ``block``.

    ``proj`` is a (1x1 conv, GroupNorm) Sequential matching the reference
    ``proj.0`` / ``proj.1`` keys; ``block`` is a :class:`_FPNConvLayer`.
    """

    def __init__(self, in_features: int, lateral_features: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(lateral_features, in_features, 1, padding=0, bias=False),
            nn.GroupNorm(32, in_features),
        )
        self.block = _FPNConvLayer(in_features, in_features)

    @override
    def forward(self, down: Tensor, left: Tensor) -> Tensor:
        left = self.proj(left)
        h = int(left.shape[2])
        w = int(left.shape[3])
        down = F.interpolate(down, size=(h, w), mode="nearest")
        down = down + left
        return self.block(down)


@final
class _FPNModel(nn.Module):
    """Feature Pyramid Network: a ``stem`` + top-down ``layers``.

    Returns the list of fused feature maps (low → high resolution); the
    pixel decoder uses the last (highest-resolution) one.
    """

    def __init__(
        self,
        in_features: int,
        lateral_widths: list[int],
        feature_size: int,
    ) -> None:
        super().__init__()
        self.stem = _FPNConvLayer(in_features, feature_size)
        fpn_layers = [_FPNLayer(feature_size, lw) for lw in lateral_widths[::-1]]
        self.layers = nn.Sequential(*fpn_layers)
        self._fpn_layers: list[_FPNLayer] = fpn_layers

    @override
    def forward(self, features: list[Tensor]) -> list[Tensor]:
        fpn_features: list[Tensor] = []
        last_feature = features[-1]
        other_features = features[:-1]
        output: Tensor = self.stem(last_feature)
        rev = other_features[::-1]
        for i, layer in enumerate(self._fpn_layers):
            output = layer(output, rev[i])
            fpn_features.append(output)
        return fpn_features


@final
class _FPNPixelDecoder(nn.Module):
    """Reference ``MaskFormerPixelDecoder``: FPN + a 3x3 ``mask_projection``.

    Produces the per-pixel mask-feature map at 1/4 resolution and
    ``mask_feature_size`` channels.
    """

    def __init__(
        self,
        in_features: int,  # C5 channels
        lateral_widths: list[int],  # [c2, c3, c4]
        feature_size: int,
        mask_feature_size: int,
    ) -> None:
        super().__init__()
        self.fpn = _FPNModel(in_features, lateral_widths, feature_size)
        self.mask_projection = nn.Conv2d(feature_size, mask_feature_size, 3, padding=1)

    @override
    def forward(self, features: list[Tensor]) -> Tensor:
        fpn_features = self.fpn(features)
        return self.mask_projection(fpn_features[-1])


# ---------------------------------------------------------------------------
# DETR-style decoder with separate q/k/v/o projections (reference-exact)
# ---------------------------------------------------------------------------


def _with_pos(tensor: Tensor, pos: Tensor | None) -> Tensor:
    """Add positional embedding to a tensor (identity when ``pos`` is None)."""
    return tensor if pos is None else tensor + pos


class _Attention(nn.Module):
    """Multi-head attention with separate ``q/k/v/o_proj`` linears.

    Operates on ``(B, L, d)`` sequences.  Position embeddings are added to
    the *query* and *key* inputs (never the value) before projection, per
    the reference MaskFormer DETR attention.
    """

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def _split(self, x: Tensor, B: int, L: int) -> Tensor:
        # (B, L, d) → (B, n_head, L, head_dim)
        return x.reshape(B, L, self.n_head, self.head_dim).permute(0, 2, 1, 3)

    @override
    def forward(
        self,
        query_input: Tensor,  # (B, Lq, d) — query stream (+ query pos)
        key_input: Tensor,  # (B, Lk, d) — key stream (+ key pos)
        value_input: Tensor,  # (B, Lk, d) — value stream (no pos)
    ) -> Tensor:
        B = int(query_input.shape[0])
        Lq = int(query_input.shape[1])
        Lk = int(key_input.shape[1])

        q = self._split(self.q_proj(query_input), B, Lq)
        k = self._split(self.k_proj(key_input), B, Lk)
        v = self._split(self.v_proj(value_input), B, Lk)

        # (B, h, Lq, hd) @ (B, h, hd, Lk) → (B, h, Lq, Lk)
        attn = lucid.matmul(q * self.scaling, k.permute(0, 1, 3, 2))
        attn = F.softmax(attn, dim=-1)
        out = lucid.matmul(attn, v)  # (B, h, Lq, hd)
        out = out.permute(0, 2, 1, 3).reshape(B, Lq, self.n_head * self.head_dim)
        return self.o_proj(out)


@final
class _DecoderMLP(nn.Module):
    """Feed-forward block (``fc1`` → ReLU → ``fc2``); reference ``MaskFormerDetrMLP``."""

    def __init__(self, d_model: int, dim_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(x)))


@final
class _DecoderLayer(nn.Module):
    """One post-norm decoder layer (reference ``MaskFormerDetrDecoderLayer``).

    Submodule names match the reference: ``self_attn`` + ``self_attn_layer_norm``,
    ``encoder_attn`` + ``encoder_attn_layer_norm``, ``mlp`` + ``final_layer_norm``.
    Self-attention adds the query positions to Q/K; cross-attention adds the
    query positions to Q and the spatial positions to K.
    """

    def __init__(self, d_model: int, n_head: int, dim_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = _Attention(d_model, n_head)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = _Attention(d_model, n_head)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.mlp = _DecoderMLP(d_model, dim_ff)
        self.final_layer_norm = nn.LayerNorm(d_model)
        # ``config.dropout`` was declared, documented, and set to 0.1 by both
        # registered factories — and no Dropout was ever constructed.  The
        # reference applies residual dropout on each of the three sub-blocks.
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    @override
    def forward(
        self,
        hidden: Tensor,  # (B, N, d) query stream
        memory: Tensor,  # (B, S, d) encoder features
        spatial_pos: Tensor,  # (B, S, d) spatial position embeddings
        query_pos: Tensor,  # (B, N, d) query position embeddings
    ) -> Tensor:
        # Self-attention (query pos added to Q + K).
        residual = hidden
        q_in = _with_pos(hidden, query_pos)
        hidden = self.self_attn(q_in, q_in, hidden)
        hidden = residual + self.dropout1(hidden)
        hidden = self.self_attn_layer_norm(hidden)

        # Cross-attention (query pos to Q, spatial pos to K, plain memory V).
        residual = hidden
        hidden = self.encoder_attn(
            _with_pos(hidden, query_pos),
            _with_pos(memory, spatial_pos),
            memory,
        )
        hidden = residual + self.dropout2(hidden)
        hidden = self.encoder_attn_layer_norm(hidden)

        # Feed-forward.
        residual = hidden
        hidden = self.mlp(hidden)
        hidden = residual + self.dropout3(hidden)
        hidden = self.final_layer_norm(hidden)
        return hidden


@final
class _TransformerDecoder(nn.Module):
    """Stack of decoder layers + a trailing ``layernorm`` (reference layout)."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_ff: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers: nn.ModuleList[_DecoderLayer] = nn.ModuleList(
            [_DecoderLayer(d_model, n_head, dim_ff, dropout) for _ in range(depth)]
        )
        self.layernorm = nn.LayerNorm(d_model)

    @override
    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
        spatial_pos: Tensor,
        query_pos: Tensor,
    ) -> list[Tensor]:
        """Return the normalised output of *every* layer, deepest last.

        Section 4.1 applies the same loss after each decoder layer, so the
        intermediates are part of the training signal, not debug output.
        Callers that only want the prediction take ``[-1]``.
        """
        out = hidden
        stages: list[Tensor] = []
        for layer in self.layers:
            out = layer(out, memory, spatial_pos, query_pos)
            stages.append(self.layernorm(out))
        return stages


class _TransformerModule(nn.Module):
    """Reference ``MaskFormerTransformerModule``.

    Holds ``queries_embedder`` (nn.Embedding), ``input_projection``
    (1x1 conv C5 → d_model) and the DETR-style ``decoder``.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_head: int,
        dim_ff: int,
        depth: int,
        num_queries: int,
        dropout: float = 0.0,
        encoder_depth: int = 0,
    ) -> None:
        super().__init__()
        self._pos_embed = SinePositionEmbedding(num_position_features=d_model // 2)
        self.queries_embedder = nn.Embedding(num_queries, d_model)
        self.input_projection = nn.Conv2d(in_features, d_model, 1)
        # ``encoder_depth = 0`` (the reference semantic setting) leaves this
        # empty and the C5 memory reaches the decoder untouched.
        self.encoder: nn.ModuleList[nn.TransformerEncoderLayer] = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    n_head,
                    dim_feedforward=dim_ff,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.decoder = _TransformerDecoder(d_model, n_head, dim_ff, depth, dropout)

    @override
    def forward(self, image_features: Tensor) -> list[Tensor]:
        feat: Tensor = self.input_projection(image_features)
        B = int(feat.shape[0])
        c = int(feat.shape[1])
        h = int(feat.shape[2])
        w = int(feat.shape[3])

        # Query positions: (N, d) → (B, N, d)
        q_weight: Tensor = self.queries_embedder.weight
        query_pos = q_weight.unsqueeze(0).repeat(B, 1, 1)
        # The reference's ``zeros_like(query_embeddings)``: the decoder's
        # first input takes the queries' dtype as well as their shape.
        hidden = lucid.zeros_like(query_pos)

        # Spatial positions + flattened memory: (B, S, d)
        pos = self._pos_embed(feat)  # (B, d, h, w)
        spatial_pos = pos.reshape(B, c, h * w).permute(0, 2, 1)
        memory = feat.reshape(B, c, h * w).permute(0, 2, 1)
        for enc in self.encoder:
            memory = enc(memory + spatial_pos)

        return self.decoder(hidden, memory, spatial_pos, query_pos)


class _PixelLevelModule(nn.Module):
    """Reference ``MaskFormerPixelLevelModule``: ``encoder`` + FPN ``decoder``."""

    def __init__(
        self,
        in_channels: int,
        layers: tuple[int, int, int, int],
        feature_size: int,
        mask_feature_size: int,
    ) -> None:
        super().__init__()
        self.encoder = _ResNetBackbone(in_channels, layers)
        ch = self.encoder.out_channels  # [c2, c3, c4, c5]
        self.decoder = _FPNPixelDecoder(
            in_features=ch[-1],
            lateral_widths=ch[:-1],
            feature_size=feature_size,
            mask_feature_size=mask_feature_size,
        )

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder(x)
        pixel_embeddings = self.decoder(features)
        # (C5 raw backbone feature, per-pixel mask features)
        return features[-1], pixel_embeddings


@final
class _MaskEmbedder(nn.Sequential):
    """3-layer MLP mask head (reference ``MaskformerMLPPredictionHead``).

    Subclasses :class:`nn.Sequential` so the per-layer keys are
    ``{0,1,2}`` and each layer is itself a ``nn.Sequential`` of
    ``Linear`` (``.0``) + activation (``.1``, no parameters), giving the
    reference ``mask_embedder.{0,1,2}.0.{weight,bias}`` key layout.
    Layers 0-1 use ReLU; the final layer is identity.
    """

    def __init__(self, d_model: int, mask_feature_size: int, num_layers: int = 3) -> None:
        in_dims = [d_model] + [d_model] * (num_layers - 1)
        out_dims = [d_model] * (num_layers - 1) + [mask_feature_size]
        blocks: list[nn.Module] = []
        for i in range(num_layers):
            act: nn.Module = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            blocks.append(nn.Sequential(nn.Linear(in_dims[i], out_dims[i]), act))
        super().__init__(*blocks)


# ---------------------------------------------------------------------------
# Hungarian matching for semantic segmentation masks
# ---------------------------------------------------------------------------


def _focal_dice_terms(
    mask_logits: Tensor,  # (N, P) flattened predictions
    gt_masks: Tensor,  # (M, P) flattened binary targets
) -> tuple[Tensor, Tensor]:
    r"""Pairwise focal and dice terms between every query and every GT mask.

    Both are computed in closed matrix form — an ``(N, M)`` pair of tensors
    from two matmuls — rather than looping over the :math:`N \times M` pairs.

    The focal term factorises because binary focal loss is a sum of a
    positive-class part weighted by :math:`t` and a negative-class part
    weighted by :math:`1 - t`; contracting each part against the target
    matrix gives every pairing at once.

    Args:
        mask_logits: ``(N, P)`` raw (pre-sigmoid) query mask logits.
        gt_masks:    ``(M, P)`` binary ground-truth masks.

    Returns:
        ``(focal, dice)``, each ``(N, M)``.
    """
    P = int(mask_logits.shape[1])
    prob: Tensor = F.sigmoid(mask_logits)  # (N, P)

    # -log sigmoid(x) and -log(1 - sigmoid(x)), computed through logsigmoid so
    # saturated logits do not produce inf.
    neg_log_p: Tensor = -F.logsigmoid(mask_logits)
    neg_log_1mp: Tensor = -F.logsigmoid(-mask_logits)

    pos: Tensor = _FOCAL_ALPHA * ((1.0 - prob) ** _FOCAL_GAMMA) * neg_log_p
    neg: Tensor = (1.0 - _FOCAL_ALPHA) * (prob**_FOCAL_GAMMA) * neg_log_1mp

    gt_t: Tensor = gt_masks.permute(1, 0)  # (P, M)
    ones_t: Tensor = 1.0 - gt_t
    focal: Tensor = (pos @ gt_t + neg @ ones_t) / P  # (N, M)

    inter: Tensor = prob @ gt_t  # (N, M)
    denom: Tensor = prob.sum(dim=-1).reshape(-1, 1) + gt_masks.sum(dim=-1).reshape(1, -1)
    # The reference smooths with +1 on *both* numerator and denominator; a bare
    # epsilon guard would score an empty mask 1.0 instead of the reference's 0.
    dice: Tensor = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return focal, dice


def _hungarian_match_masks(
    class_logits: Tensor,  # (N, K+1)
    mask_logits: Tensor,  # (N, H, W)
    gt_labels: Tensor,  # (M,) integer class ids
    gt_masks: Tensor,  # (M, H, W) binary
) -> tuple[list[int], list[int]]:
    r"""Hungarian matching between N queries and M GT segments.

    Cost, following the paper's Eq. 3 and the released ``HungarianMatcher``:

    .. math::

        -p_i(c_j)
        + \lambda_{\mathrm{mask}}\, \mathcal{L}_{\mathrm{focal}}
        + \lambda_{\mathrm{dice}}\, \mathcal{L}_{\mathrm{dice}}.

    The earlier cost used hard-thresholded mask IoU, which is piecewise
    constant: it is exactly zero-gradient in the matcher's own right and,
    worse, ranks a query that is confidently wrong the same as one that is
    marginally wrong, so early in training the assignment is close to
    arbitrary.

    Returns:
        (pred_indices, gt_indices) matched pairs.
    """
    N = int(class_logits.shape[0])
    M = int(gt_labels.shape[0])
    if M == 0 or N == 0:
        return [], []

    probs = F.softmax(class_logits, dim=-1)  # (N, K+1)
    cost_class: Tensor = -probs.index_select(1, gt_labels.long())  # (N, M)

    flat_pred = mask_logits.reshape(N, -1)
    flat_gt = gt_masks.reshape(M, -1)
    focal, dice = _focal_dice_terms(flat_pred, flat_gt)

    cost: Tensor = cost_class + _MASK_WEIGHT * focal + _DICE_WEIGHT * dice

    from lucid.models._utils._detection import solve_assignment

    rows = cast(list[list[float]], cost.tolist())
    pred_idx, gt_idx = solve_assignment(rows)
    return pred_idx, gt_idx


# ---------------------------------------------------------------------------
# MaskFormer model
# ---------------------------------------------------------------------------


@dataclass
class MaskFormerOutput(SemanticSegmentationOutput):
    r"""Segmentation output that also carries the raw per-query predictions.

    ``logits`` is the semantic map the paper's *semantic* inference
    produces by marginalising the queries.  That marginalisation is
    lossy: it collapses N queries into K class channels, so the general
    mask-classification inference of §3.4 — which needs each query's
    class distribution and its own mask — cannot be recovered from it.
    Both are returned here so panoptic and instance inference are
    expressible without re-running the model.

    Attributes
    ----------
    class_queries_logits : Tensor
        Per-query class logits, shape ``(B, N, K + 1)``; the last column
        is the "no object" class.
    masks_queries_logits : Tensor
        Per-query mask logits at the mask-feature resolution, shape
        ``(B, N, fH, fW)`` — *not* upsampled, matching the reference.

    Examples
    --------
    Mask classification returns three tensors, not one. ``logits`` is the
    per-pixel semantic map the two query tensors were resolved into, and
    is what you would compare against a segmentation label.

    >>> import lucid
    >>> from lucid.models.vision.maskformer import maskformer_resnet50
    >>> model = maskformer_resnet50().eval()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape                      # (B, K, H, W)
    (1, 150, 224, 224)

    The queries are the model's own representation: 100 of them, each
    carrying a class distribution and a mask. The class axis is 151 and
    not 150 because the last slot is "no object" — a query that matched
    nothing, which is how a fixed number of queries covers a variable
    number of regions.

    >>> out.class_queries_logits.shape        # (B, Q, K + 1)
    (1, 100, 151)
    >>> out.masks_queries_logits.shape        # (B, Q, H/4, W/4)
    (1, 100, 56, 56)
    """

    class_queries_logits: Tensor | None = None
    masks_queries_logits: Tensor | None = None


class MaskFormerForSemanticSegmentation(SemanticSegmentationModel):
    r"""MaskFormer semantic-segmentation model (Cheng et al., NeurIPS 2021).

    Recasts semantic segmentation as **mask classification**: instead of
    per-pixel cross-entropy, the model predicts a fixed set of
    :math:`N` (mask, class-probability) pairs, computes the per-pixel
    class probability as

    .. math::

        p_\mathrm{px}(c) = \sum_{i=1}^{N} p_i(c)\, m_i(\mathrm{px}),

    and supervises the set with bipartite (Hungarian) matching to ground-
    truth masks.  This unification lets a single architecture handle
    semantic, instance, and panoptic segmentation with the same training
    objective — the breakthrough that the follow-up Mask2Former extends
    with masked attention.

    Architecturally: a ResNet backbone -> FPN-style pixel decoder
    produces a per-pixel embedding of dimension ``mask_feature_size``, and
    a DETR-style transformer decoder operates on :math:`N` query
    embeddings, cross-attending to the projected backbone ``C5`` feature
    map; sibling heads then produce :math:`(N, K+1)` class logits and a
    3-layer-MLP mask embedding.  Each query's binary mask is recovered by
    a dot product with the pixel embedding.

    Parameters
    ----------
    config : MaskFormerConfig
        Frozen architecture spec.  Use :func:`maskformer_resnet50` /
        :func:`maskformer_resnet101` for the paper-cited variants
        (ADE20K, 150 classes).

    Attributes
    ----------
    config : MaskFormerConfig
        Stored copy of the config that built this model.
    pixel_level_module : _PixelLevelModule
        ResNet ``encoder`` + FPN pixel ``decoder`` yielding the raw ``C5``
        feature and a per-pixel embedding :math:`(B, d, H/4, W/4)`.
    transformer_module : _TransformerModule
        Learned queries + ``input_projection`` + a 6-layer DETR-style
        decoder.
    class_predictor : nn.Linear
        Per-query :math:`(K + 1)` classification head (``+1`` for the
        "no object" class).
    mask_embedder : _MaskEmbedder
        Per-query 3-layer MLP producing the dot-product mask weights.

    Notes
    -----
    See Cheng et al., "Per-Pixel Classification is Not All You Need for
    Semantic Segmentation", NeurIPS 2021 (arXiv:2107.06278).  The total
    loss combines a CE term over query class predictions and a per-mask
    binary CE + dice term, summed only over the Hungarian-matched query
    permutation :math:`\hat{\sigma}`.

    For semantic-segmentation inference the model drops the "no-object"
    slot and collapses queries by softmax-weighted summation into a
    standard :math:`(B, K, H, W)` logit map for direct argmax evaluation.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.maskformer import maskformer_resnet50
    >>> model = maskformer_resnet50()
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.logits.shape   # (B, K, H, W)
    (1, 150, 512, 512)
    """

    config_class: ClassVar[type[MaskFormerConfig]] = MaskFormerConfig
    base_model_prefix: ClassVar[str] = "maskformer"

    def __init__(self, config: MaskFormerConfig) -> None:
        super().__init__(config)
        self._cfg = config
        d = config.d_model
        K = config.num_classes

        # 1. Pixel-level module (ResNet encoder + FPN pixel decoder)
        self.pixel_level_module = _PixelLevelModule(
            in_channels=config.in_channels,
            layers=config.backbone_layers,
            feature_size=config.fpn_out_channels,
            mask_feature_size=d,
        )
        c5_channels = self.pixel_level_module.encoder.out_channels[-1]

        # 2. Transformer module (queries + input projection + decoder)
        self.transformer_module = _TransformerModule(
            in_features=c5_channels,
            d_model=d,
            n_head=config.n_head,
            dim_ff=config.dim_feedforward,
            depth=config.num_decoder_layers,
            num_queries=config.num_queries,
            dropout=config.dropout,
            encoder_depth=config.num_encoder_layers,
        )

        # 3. Prediction heads
        self.class_predictor = nn.Linear(d, K + 1)
        self.mask_embedder = _MaskEmbedder(d, d)

    @override
    def forward(
        self,
        x: Tensor,
        targets: dict[str, Tensor] | None = None,
    ) -> SemanticSegmentationOutput:
        """Run MaskFormer.

        Args:
            x:       (B, C, H, W) image batch.
            targets: Optional dict with ``"masks"`` (B, H, W) integer masks.

        Returns:
            ``SemanticSegmentationOutput``:
              ``logits``: (B, K, H, W) per-pixel class logits (no-object
                          slot dropped — matches the reference semantic
                          post-processing).
              ``loss``:   training loss when targets provided.
        """
        B = int(x.shape[0])
        iH = int(x.shape[2])
        iW = int(x.shape[3])

        # 1. Pixel-level module → (C5 feature, per-pixel mask embedding)
        image_features, pixel_embeddings = self.pixel_level_module(x)
        fH = int(pixel_embeddings.shape[2])
        fW = int(pixel_embeddings.shape[3])
        d = int(pixel_embeddings.shape[1])

        # 2. Transformer module → one (B, N, d) query tensor per decoder layer
        hs_stages: list[Tensor] = self.transformer_module(image_features)
        hs: Tensor = hs_stages[-1]
        N = int(hs.shape[1])

        pixel_flat: Tensor = pixel_embeddings.reshape(B, d, fH * fW)  # (B, d, S)

        def _predict(stage: Tensor) -> tuple[Tensor, Tensor]:
            # 3. Class predictions: (B, N, K+1)
            cls_l: Tensor = self.class_predictor(stage)
            # 4. Mask predictions: einsum("bqc,bchw->bqhw") via matmul
            m_embed: Tensor = self.mask_embedder(stage)  # (B, N, d)
            m_flat: Tensor = lucid.matmul(m_embed, pixel_flat)  # (B, N, S)
            return cls_l, m_flat.reshape(B, N, fH, fW)

        class_logits, mask_logits = _predict(hs)

        # 5. Semantic post-processing (reference-exact):
        #    masks_classes = softmax(class)[..., :-1]   (drop no-object)
        #    masks_probs   — sigmoid(mask)
        #    seg = einsum("bqc,bqhw->bchw")             (at feature resolution)
        K_plus_1 = self._cfg.num_classes + 1
        class_probs: Tensor = F.softmax(class_logits, dim=-1)  # (B, N, K+1)
        masks_classes: Tensor = class_probs[:, :, : K_plus_1 - 1]  # (B, N, K)
        masks_probs: Tensor = F.sigmoid(mask_logits)  # (B, N, fH, fW)

        # seg[b,k,h,w] = sum_n masks_classes[b,n,k] * masks_probs[b,n,h,w]
        masks_classes_t: Tensor = masks_classes.permute(0, 2, 1)  # (B, K, N)
        masks_flat: Tensor = masks_probs.reshape(B, N, fH * fW)  # (B, N, S)
        seg_flat: Tensor = lucid.matmul(masks_classes_t, masks_flat)  # (B, K, S)
        seg_logits_small: Tensor = seg_flat.reshape(B, self._cfg.num_classes, fH, fW)

        # 6. Upsample to input resolution (per the reference image processor)
        seg_logits: Tensor = F.interpolate(
            seg_logits_small, size=(iH, iW), mode="bilinear", align_corners=False
        )

        # 7. Loss.  Deep supervision (§4.1, "we apply the same loss after each
        #    decoder"): every intermediate decoder output goes through the same
        #    prediction heads and the same criterion, and the terms are summed.
        loss: Tensor | None = None
        if targets is not None:
            loss = self._compute_loss(class_logits, mask_logits, targets, (fH, fW))
            for stage in hs_stages[:-1]:
                aux_cls, aux_mask = _predict(stage)
                loss = loss + self._compute_loss(aux_cls, aux_mask, targets, (fH, fW))

        return MaskFormerOutput(
            logits=seg_logits,
            loss=loss,
            class_queries_logits=class_logits,
            masks_queries_logits=mask_logits,
        )

    def _compute_loss(
        self,
        class_logits: Tensor,  # (B, N, K+1)
        mask_logits: Tensor,  # (B, N, H/4, W/4)
        targets: dict[str, Tensor],
        feat_size: tuple[int, int],
    ) -> Tensor:
        """Compute Hungarian-matched mask + class loss across batch.

        Every tensor the loss builds itself — class targets, the no-object
        class weight, match indices, the zero fallback — is created on the
        predictions' device and, if floating, in their dtype: a CPU one
        raised ``DeviceMismatch`` against Metal logits, a float32 one widened
        a half loss.  ``targets["masks"]`` must already be on that device, as
        the predictions are.
        """
        B = int(class_logits.shape[0])
        N = int(class_logits.shape[1])
        K = self._cfg.num_classes
        fH, fW = feat_size
        dev = class_logits.device.type

        gt_masks_full: Tensor = targets["masks"]  # (B, H, W) integer

        cls_losses: list[Tensor] = []
        mask_losses: list[Tensor] = []

        for b in range(B):
            cl_b: Tensor = class_logits[b]  # (N, K+1)
            ml_b: Tensor = mask_logits[b]  # (N, H/4, W/4)
            seg_b: Tensor = gt_masks_full[b]  # (H, W) integer
            seg_H = int(seg_b.shape[0])
            seg_W = int(seg_b.shape[1])

            # One-hot the label map against the K valid class ids in a single
            # broadcast compare.  Enumerating the distinct values with a
            # per-pixel .item() scan cost H*W device syncs — 260k for a
            # 512x512 crop, which made the training path unusable — and it
            # also admitted labels that are not classes at all: the standard
            # 255 "ignore" value became a GT segment and then indexed the
            # (K+1)-wide class head out of bounds.  Comparing against
            # arange(K) drops ignore and any other out-of-range label for
            # free.
            class_ids: Tensor = lucid.arange(K, device=seg_b.device.type).long()
            onehot: Tensor = (seg_b.reshape(1, seg_H, seg_W) == class_ids.reshape(K, 1, 1)).to(
                ml_b.dtype
            )
            counts: list[float] = cast(list[float], onehot.sum(dim=(1, 2)).tolist())
            unique_classes: list[int] = [c for c in range(K) if counts[c] > 0.0]

            M = len(unique_classes)

            if M == 0:
                # Nothing but background / ignore.
                bg_tgt_data: list[int] = [K] * N  # "no-object" class index = K
                bg_tgt: Tensor = lucid.tensor(bg_tgt_data, device=dev)
                cls_losses.append(F.cross_entropy(cl_b, bg_tgt, reduction="mean"))
                continue

            sel: Tensor = lucid.tensor(unique_classes, device=seg_b.device.type).long()
            gt_labels_t: Tensor = sel  # (M,)
            gt_masks_t: Tensor = onehot.index_select(0, sel)  # (M, H, W)

            # Downsample GT masks to feature resolution for matching
            gt_masks_small: Tensor = F.interpolate(
                gt_masks_t.reshape(M, 1, seg_H, seg_W),
                size=(fH, fW),
                mode="nearest",
            ).reshape(M, fH, fW)

            # Hungarian match
            pred_idx, gt_idx = _hungarian_match_masks(
                cl_b,
                ml_b.detach() if hasattr(ml_b, "detach") else ml_b,
                gt_labels_t,
                gt_masks_small,
            )

            # Class loss: matched → gt class, unmatched → K (no-object).
            # Paper §4.1, following DETR: "the weight for the 'no object'
            # (empty set) in the classification loss is set to 0.1".  Without
            # it the N−M unmatched queries (N=100 against a handful of
            # segments) dominate the term and push every query to no-object.
            cls_tgt_data: list[int] = [K] * N
            for pi, gi in zip(pred_idx, gt_idx, strict=True):
                cls_tgt_data[pi] = unique_classes[gi]
            cls_tgt: Tensor = lucid.tensor(cls_tgt_data, device=dev)
            cls_weight: Tensor = lucid.tensor(
                [1.0] * K + [_NO_OBJECT_WEIGHT], dtype=cl_b.dtype, device=cl_b.device
            )
            cls_losses.append(F.cross_entropy(cl_b, cls_tgt, weight=cls_weight, reduction="mean"))

            # Mask loss on matched pairs.  Section 4.1 weights the terms
            # lambda_mask=20 and lambda_dice=1 against a class term of 1, so
            # an equal 1:1:1 balance under-weights the mask objective 20x.
            # The mask term is sigmoid focal loss: plain BCE lets the vast
            # background of every binary mask swamp the thin foreground the
            # query is supposed to learn.
            #
            # The loss compares at *ground-truth* resolution: predictions are
            # bilinearly upsampled, GT is left alone.  Downsampling the GT
            # instead (which is what the MATCHER does, deliberately, to save
            # memory) throws away every structure finer than the 1/4 stride
            # before the gradient ever sees it, so thin classes can never be
            # learned.
            if pred_idx:
                pi_t: Tensor = lucid.tensor(pred_idx, device=ml_b.device.type).long()
                gi_t: Tensor = lucid.tensor(gt_idx, device=ml_b.device.type).long()
                n_sel = len(pred_idx)
                up_pred: Tensor = F.interpolate(
                    ml_b.index_select(0, pi_t).reshape(n_sel, 1, fH, fW),
                    size=(seg_H, seg_W),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(n_sel, -1)
                sel_pred: Tensor = up_pred
                sel_gt: Tensor = gt_masks_t.reshape(M, -1).index_select(0, gi_t)
                focal_m, dice_m = _focal_dice_terms(sel_pred, sel_gt)
                n_match = len(pred_idx)
                diag: Tensor = lucid.tensor(
                    [i * n_match + i for i in range(n_match)],
                    device=ml_b.device.type,
                ).long()
                focal_d: Tensor = focal_m.reshape(-1).index_select(0, diag)
                dice_d: Tensor = dice_m.reshape(-1).index_select(0, diag)
                mask_losses.append((_MASK_WEIGHT * focal_d + _DICE_WEIGHT * dice_d).mean())

        # Scalar fallbacks: a (1,) zero made the loss shape (1,) whenever no
        # image in the batch held a valid class, and a scalar everywhere else.
        cls_loss: Tensor = (
            lucid.cat([l.reshape(1) for l in cls_losses]).mean()
            if cls_losses
            else class_logits.new_zeros()
        )
        mask_loss: Tensor = (
            lucid.cat([l.reshape(1) for l in mask_losses]).mean()
            if mask_losses
            else class_logits.new_zeros()
        )

        return cls_loss + mask_loss
