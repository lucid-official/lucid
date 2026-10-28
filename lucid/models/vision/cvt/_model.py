"""CvT backbone and classification head (Wu et al., 2021).

Paper: "CvT: Introducing Convolutions to Vision Transformers"
        https://arxiv.org/abs/2103.15808

Key idea: replace standard linear Q/K/V projections in MHA with
depthwise-separable convolutional projections (DWConv3×3 + BN → flatten →
Linear). This gives translation equivariance + local context to the attention
queries/keys/values, capturing both local and global relationships.

Unlike plain ViT there are NO positional embeddings — position information
is implicitly encoded by the strided convolutional token embeddings.

Architecture (CvT-13, image=224):
  Stage 1: ConvEmbed(s=4) → (56×56, 64),   1 × CvTBlock(heads=1)
  Stage 2: ConvEmbed(s=2) → (28×28, 192),  2 × CvTBlock(heads=3)
  Stage 3: ConvEmbed(s=2) → (14×14, 384), 10 × CvTBlock(heads=6)
  Head   : LN → mean-pool → FC(384, num_classes)
"""

from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import init_transformer_trunc_normal
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._utils._classification import DropPath
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.cvt._config import CvTConfig

# ---------------------------------------------------------------------------
# Convolutional token embedding (overlapping)
# ---------------------------------------------------------------------------


@final
class _ConvEmbed(nn.Module):
    """Overlapping convolutional patch embedding.

    Conv2d (kernel > stride, so overlapping) → LN → return spatial map.
    No explicit positional embeddings needed since conv is position-aware.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, padding: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(out_ch)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        # (B, C_in, H, W) → (B, C_out, H', W')
        x = self.proj(x)
        B, C, H, W = x.shape
        # (B, H'*W', C) for LayerNorm, then back
        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1)
        x_flat = self.norm(x_flat)
        # Return spatial layout for downstream conv projections
        x_out = x_flat.permute(0, 2, 1).reshape(B, C, H, W)
        return x_out, H, W


# ---------------------------------------------------------------------------
# Convolutional Q/K/V projection (the CvT innovation)
# ---------------------------------------------------------------------------


@final
class _ConvProj(nn.Module):
    """Depthwise conv (3×3, stride) + BN + flatten → linear projection.

    For K and V: stride_kv may downsample the spatial map.
    For Q: stride=1 (no downsampling, full resolution).
    """

    def __init__(self, dim: int, kernel: int = 3, stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        # Depthwise conv (groups=dim) + BN.  The reference CvT
        # checkpoints disable the conv bias (the following BN already
        # absorbs any per-channel additive term), so we match that
        # exactly to keep the state-dict 1:1.
        self.dw = nn.Conv2d(
            dim, dim, kernel, stride=stride, padding=padding, groups=dim, bias=False
        )
        self.bn = nn.BatchNorm2d(dim)
        # Linear projection — biased to match the Microsoft / HF
        # reference CvT checkpoints (``projection_query/key/value`` carry
        # biases in ``transformers.CvtForImageClassification``).
        self.proj = nn.Linear(dim, dim, bias=True)

    @override
    def forward(self, x: Tensor, H: int, W: int, cls: Tensor | None = None) -> Tensor:
        # x: (B, N, C) — the *spatial* token sequence (cls already split off
        # by the caller).  Reshape to a feature map for the depthwise conv.
        B, N, C = x.shape
        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_2d = self.bn(self.dw(x_2d))  # (B, C, H', W')
        _B, _C, H2, W2 = x_2d.shape
        x_flat = x_2d.reshape(B, C, H2 * W2).permute(0, 2, 1)  # (B, H'*W', C)
        # Re-attach the CLS token (which bypasses the conv) before the
        # shared linear projection, matching the reference CvT exactly.
        if cls is not None:
            x_flat = lucid.cat([cls, x_flat], dim=1)
        return self.proj(x_flat)


# ---------------------------------------------------------------------------
# CvT Attention: convolutional projections for Q, K, V
# ---------------------------------------------------------------------------


@final
class _CvTAttention(nn.Module):
    """Multi-head self-attention with convolutional Q/K/V projections.

    Q is projected with stride=1 (full resolution).
    K and V are projected with stride=stride_kv (can downsample spatially).
    This reduces the O(N²) cost at high resolutions (like SRA in PVT).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        stride_kv: int = 1,
        with_cls_token: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.with_cls_token = with_cls_token
        # ``config.dropout`` reached only the MLP: the reference also drops on
        # the attention probabilities and on the output projection.
        self.attn_drop_p = attn_drop
        self.proj_drop = nn.Dropout(p=proj_drop)
        # CvT scales attention logits by the *full* embedding dim, not the
        # per-head dim (an unusual but deliberate choice — see the
        # reference ``CvtSelfAttention.scale = embed_dim ** -0.5``).  This
        # only matters when ``num_heads > 1``; stage 0 has a single head
        # so ``head_dim == dim`` and the distinction is invisible there.
        self.scale = dim**-0.5

        # Q: stride=1 (full resolution)
        self.proj_q = _ConvProj(dim, kernel=3, stride=1, padding=1)
        # K, V: strided (may reduce spatial resolution)
        self.proj_k = _ConvProj(dim, kernel=3, stride=stride_kv, padding=1)
        self.proj_v = _ConvProj(dim, kernel=3, stride=stride_kv, padding=1)

        self.out_proj = nn.Linear(dim, dim)

    @override
    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, N, C = x.shape
        head_dim = C // self.num_heads

        # When the stage carries a CLS token it occupies slot 0 of the
        # sequence.  The conv projections operate on the spatial tokens
        # only; the CLS token bypasses the conv and is re-attached inside
        # ``_ConvProj`` before the shared linear projection.
        cls: Tensor | None = None
        spatial = x
        if self.with_cls_token:
            cls = x[:, 0:1, :]
            spatial = x[:, 1:, :]

        q = self.proj_q(spatial, H, W, cls)
        k = self.proj_k(spatial, H, W, cls)
        v = self.proj_v(spatial, H, W, cls)

        Nq = q.shape[1]
        Nkv = k.shape[1]

        q = q.reshape(B, Nq, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, Nkv, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, Nkv, self.num_heads, head_dim).permute(0, 2, 1, 3)

        # ``scale=self.scale`` is CvT's non-standard ``dim**-0.5`` (not the
        # per-head default), so it must be passed explicitly.  The fused
        # kernel runs unless attention dropout is on.
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0, scale=self.scale
        )
        out = out.permute(0, 2, 1, 3).reshape(B, Nq, C)
        return self.proj_drop(self.out_proj(out))


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(p=dropout)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(F.gelu(self.fc1(x)))
        return self.drop(self.fc2(x))


# ---------------------------------------------------------------------------
# CvT transformer block
# ---------------------------------------------------------------------------


@final
class _CvTBlock(nn.Module):
    """Pre-norm CvT block: LN → ConvAttn → residual → LN → MLP → residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        stride_kv: int,
        mlp_ratio: float,
        dropout: float,
        with_cls_token: bool = False,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _CvTAttention(
            dim,
            num_heads,
            stride_kv=stride_kv,
            with_cls_token=with_cls_token,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio, dropout)
        self.drop_path = DropPath(drop_path)

    @override
    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        # Input x — (B, N, C)
        n = self.norm1(x)
        attended = self.attn(n, H, W)
        x = x + self.drop_path(attended)
        return x + self.drop_path(self.mlp(self.norm2(x)))


# ---------------------------------------------------------------------------
# CvT stage
# ---------------------------------------------------------------------------


@final
class _CvTStage(nn.Module):
    """One CvT stage = ConvEmbed + N × CvTBlock.

    Input: (B, C_in, H_in, W_in)  [except stage 0 which also accepts this]
    Output: (B, C_out, H_out, W_out) spatial map
    """

    def __init__(
        self,
        in_ch: int,
        dim: int,
        depth: int,
        num_heads: int,
        embed_stride: int,
        mlp_ratio: float,
        dropout: float,
        with_cls_token: bool = False,
        drop_path_rates: tuple[float, ...] = (),
    ) -> None:
        super().__init__()
        # kernel=7 for stage 0 (large receptive field), kernel=3 for subsequent.
        # Padding follows the reference CvT exactly: the 7×7 stride-4 stem
        # uses padding=2 (NOT kernel//2=3), the 3×3 stride-2 embeds use
        # padding=1.  Getting the stem padding wrong shifts the entire
        # feature map and breaks pretrained-weight parity.
        if embed_stride == 4:
            kernel, padding = 7, 2
        else:
            kernel, padding = 3, 1
        self.embed = _ConvEmbed(in_ch, dim, kernel=kernel, stride=embed_stride, padding=padding)
        # Paper §3.2 / Table 1: K/V conv-projection uses stride=2 in *all*
        # three stages (Q is always stride=1).  This is the key trick that
        # makes CvT attention cheaper than ViT's O(N²).
        stride_kv = 2
        self.with_cls_token = with_cls_token
        # Dropout on the embedded token sequence, as the reference applies.
        self.pos_drop = nn.Dropout(p=dropout)
        # Learnable CLS token, only on the stage(s) flagged in the config
        # (the last stage for paper-cited CvT).  Prepended to the token
        # sequence so it gathers global context through attention.
        self.cls_token: nn.Parameter | None = (
            nn.Parameter(lucid.zeros(1, 1, dim)) if with_cls_token else None
        )
        self.blocks = nn.ModuleList(
            [
                _CvTBlock(
                    dim,
                    num_heads,
                    stride_kv,
                    mlp_ratio,
                    dropout,
                    with_cls_token=with_cls_token,
                    drop_path=(drop_path_rates[i] if i < len(drop_path_rates) else 0.0),
                )
                for i in range(depth)
            ]
        )
        # No stage-level LayerNorm.  The reference HF / Microsoft CvT
        # checkpoint applies only a *single* final ``layernorm`` after
        # the entire trunk; there is no per-stage normalisation between
        # blocks.

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None, int, int]:
        # Input x — (B, C_in, H_in, W_in)
        x_spatial, H, W = self.embed(x)
        # Flatten to sequence: (B, H*W, C)
        B, C, _H, _W = x_spatial.shape
        tokens = x_spatial.reshape(B, C, H * W).permute(0, 2, 1)
        tokens = self.pos_drop(tokens)
        # Prepend the CLS token (if this stage owns one).
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, 1, C)
            tokens = lucid.cat([cls, tokens], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        # Split the CLS token back off (if present); the spatial tokens
        # are reshaped to a feature map for the next stage / pooling.
        cls_out: Tensor | None = None
        if self.cls_token is not None:
            cls_out = tokens[:, 0:1, :]
            tokens = tokens[:, 1:, :]
        x_out = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return x_out, cls_out, H, W


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def _build_stages(config: CvTConfig) -> tuple[list[_CvTStage], list[FeatureInfo]]:
    stages: list[_CvTStage] = []
    in_ch = config.in_channels
    cum_stride = 1
    fi: list[FeatureInfo] = []
    # The reference schedules stochastic depth *per stage*: each stage runs its
    # own linspace(0, drop_path_rate, depth), rather than one schedule spanning
    # the whole trunk.
    for i, (dim, depth, heads, stride) in enumerate(
        zip(config.dims, config.depths, config.num_heads, config.embed_strides, strict=True)
    ):
        with_cls = bool(config.cls_token[i]) if i < len(config.cls_token) else False
        if depth > 1:
            rates = tuple(config.drop_path_rate * j / (depth - 1) for j in range(depth))
        else:
            rates = (0.0,) * depth
        stages.append(
            _CvTStage(
                in_ch,
                dim,
                depth,
                heads,
                stride,
                config.mlp_ratio,
                config.dropout,
                with_cls_token=with_cls,
                drop_path_rates=rates,
            )
        )
        in_ch = dim
        cum_stride *= stride
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=cum_stride))
    return stages, fi


# ---------------------------------------------------------------------------
# CvT backbone
# ---------------------------------------------------------------------------


class CvT(PretrainedModel, BackboneMixin):
    r"""CvT backbone (Wu et al., 2021).

    The Convolutional vision Transformer (CvT) reinjects the locality
    inductive bias of CNNs into ViT through two changes: (i) every
    stage starts with an *overlapping convolutional token embedding*
    (kernel > stride) that produces a new lower-resolution token grid,
    and (ii) the Q / K / V projections inside every self-attention are
    *depthwise-separable convolutions* rather than linear maps:

    .. math::

        Q = \mathrm{Flatten}(\mathrm{DWConv}_q(x_{2D})),\quad
        K = \mathrm{Flatten}(\mathrm{DWConv}_k(x_{2D})),\quad
        V = \mathrm{Flatten}(\mathrm{DWConv}_v(x_{2D})),

    where :math:`x_{2D}` is the spatially reshaped token map.  When the
    K / V depthwise conv uses stride 2 the attention becomes *strided*,
    reducing :math:`N` by 4x inside attention without losing the
    full-resolution output for the next block.  CvT drops positional
    embeddings entirely — locality is supplied implicitly by the
    convolutional projections.

    :meth:`forward_features` returns the mean-pooled token feature
    :math:`(B, \texttt{dims[-1]})` over the final stage's tokens.

    Parameters
    ----------
    config : CvTConfig
        Frozen dataclass specifying ``dims``, ``depths``, ``num_heads``,
        ``embed_strides``, ``mlp_ratio``, ``dropout``, and
        ``in_channels``.  See :class:`CvTConfig`.

    Attributes
    ----------
    stages : nn.ModuleList
        Three :class:`_CvTStage` modules, each composed of a
        convolutional token embedding plus a stack of CvT blocks.
    feature_info : list[FeatureInfo]
        Three-stage feature description with cumulative reductions
        given by the cumulative product of ``config.embed_strides``.

    Notes
    -----
    Reference: Haiping Wu *et al.*, *"CvT: Introducing Convolutions to
    Vision Transformers"*, ICCV 2021,
    `arXiv:2103.15808 <https://arxiv.org/abs/2103.15808>`_.

    Examples
    --------
    Build a CvT-13 backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.cvt import CvT, CvTConfig
    >>> model = CvT(CvTConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape                       # (B, dims[-1])
    (1, 384)
    """

    config_class: ClassVar[type[CvTConfig]] = CvTConfig
    base_model_prefix: ClassVar[str] = "cvt"

    def __init__(self, config: CvTConfig) -> None:
        super().__init__(config)
        stage_list, fi = _build_stages(config)
        self.stages = nn.ModuleList(stage_list)
        # Trunk LayerNorm -- the reference's ``forward_features`` ends with
        # it, so the backbone's advertised feature is the normalised one.
        self.norm = nn.LayerNorm(config.dims[-1])
        self._feature_info = fi

        # Reference initialisation: trunc_normal_(0.02) on Linear / patch-embed
        # convs with zero bias, unit LayerNorms, and the same draw for the
        # class token and positional table (bare Parameters the framework
        # default never touches — they were sitting at exact zeros).
        init_transformer_trunc_normal(self)

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        cls: Tensor | None = None
        for stage in self.stages:
            out = stage(x)
            x, cls = out[0], out[1]
        # Prefer the CLS token from the last stage that produced one;
        # fall back to spatial mean-pool for cls-token-free configs.
        #
        # The reference's ``forward_features`` ends with the trunk LayerNorm,
        # so the backbone's own output is normalised.  Omitting it here handed
        # downstream consumers of ``task="base"`` a differently-scaled feature
        # than the classifier head sees.
        if cls is not None:
            return self.norm(cls[:, 0])
        tokens = x.flatten(2).permute(0, 2, 1)  # (B, N, C)
        return self.norm(tokens).mean(dim=1)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# CvT for image classification
# ---------------------------------------------------------------------------


class CvTForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""CvT with a linear classification head (Wu et al., 2021).

    Wraps the same three-stage trunk as :class:`CvT` and adds a global
    mean pool over tokens, a final LayerNorm, and a single
    :class:`nn.Linear` classification head:

    .. math::

        \text{logits} = W_{\text{head}}\,
            \mathrm{LN}\!\bigl(\mathrm{Mean}(z^{L})\bigr)
            + b_{\text{head}}.

    Pass ``labels`` to :meth:`forward` to compute the cross-entropy
    loss in the same pass.

    Parameters
    ----------
    config : CvTConfig
        Architecture specification.  Must set ``num_classes`` to the
        desired number of output categories.  See :class:`CvTConfig`.

    Attributes
    ----------
    stages : nn.ModuleList
        Three CvT stages.
    head_norm : nn.LayerNorm
        Final LayerNorm applied to the mean-pooled feature.
    classifier : nn.Linear
        Final linear projection of width ``(num_classes, dims[-1])``.

    Notes
    -----
    Reference: Wu *et al.*, *"CvT: Introducing Convolutions to Vision
    Transformers"*, ICCV 2021.  CvT-13 / CvT-21 reach **81.6% / 82.5%
    top-1 on ImageNet-1k** at 224x224 (Table 1 of the paper).

    Examples
    --------
    End-to-end inference with the default CvT-13 classifier:

    >>> import lucid
    >>> from lucid.models.vision.cvt import (
    ...     CvTConfig, CvTForImageClassification,
    ... )
    >>> model = CvTForImageClassification(CvTConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """

    config_class: ClassVar[type[CvTConfig]] = CvTConfig
    base_model_prefix: ClassVar[str] = "cvt"

    def __init__(self, config: CvTConfig) -> None:
        super().__init__(config)
        stage_list, _ = _build_stages(config)
        self.stages = nn.ModuleList(stage_list)
        self.head_norm = nn.LayerNorm(config.dims[-1])
        self._build_classifier(config.dims[-1], config.num_classes, dropout=config.dropout)
        init_transformer_trunc_normal(self)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        cls: Tensor | None = None
        for stage in self.stages:
            out = stage(x)
            x, cls = out[0], out[1]
        # Classify from the last-stage CLS token when present (reference
        # CvT: ``layernorm(cls).mean(1)``); otherwise from the spatial
        # mean-pool.  ``cls`` is ``(B, 1, C)`` so the mean over dim 1 is
        # a no-op that just squeezes the token axis.
        if cls is not None:
            feat = self.head_norm(cls).mean(dim=1)
        else:
            # Normalise *then* pool, matching the CLS branch directly above
            # and the reference.  Pooling first hands LayerNorm a single
            # already-averaged vector, so it normalises across channels of a
            # mean instead of normalising each token — a different statistic,
            # and the two branches disagreed with each other.
            tokens = x.flatten(2).permute(0, 2, 1)  # (B, N, C)
            feat = self.head_norm(tokens).mean(dim=1)
        return self._classify(feat, labels)
