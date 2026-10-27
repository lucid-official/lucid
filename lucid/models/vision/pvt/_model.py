"""PVT v2 (Pyramid Vision Transformer v2) backbone and classifier.

Paper: "PVT v2: Improved Baselines with Pyramid Vision Transformer"
        Wang et al., 2022 — https://arxiv.org/abs/2106.13797

Key differences from PVT v1:
  1. Overlapping patch embedding (kernel=7, stride=4, padding=3 for stage 0;
     kernel=3, stride=2, padding=1 for subsequent stages) — no positional
     embeddings needed because the overlapping conv implicitly encodes position.
  2. MLP contains a depthwise Conv2d (3×3, same padding) between fc1 and fc2,
     providing spatial awareness within the MLP.
  3. Spatial Reduction Attention (SRA) unchanged from v1: K and V are reduced
     via a stride-sr_ratio Conv2d before MHA, keeping cost manageable at high
     resolutions.

Architecture (PVT v2-B1 / our 'pvt_tiny' default, 224 input):
  patch_embed: OverlapPatchEmbed(k=7,s=4) — top-level, used by stage 0
  Stage 0: blocks (heads=1, sr=8) + norm   — no downsample; uses top-level patch_embed
  Stage 1: downsample(k=3,s=2) + blocks (heads=2, sr=4) + norm
  Stage 2: downsample(k=3,s=2) + blocks (heads=5, sr=2) + norm
  Stage 3: downsample(k=3,s=2) + blocks (heads=8, sr=1) + norm
  Head   : global avg-pool → FC(512, num_classes)

timm key layout (pvt_v2_b1):
  patch_embed.proj.*  patch_embed.norm.*          — top-level
  stages.0.blocks.N.*  stages.0.norm.*            — stage 0 (no downsample)
  stages.1.downsample.proj/norm  stages.1.blocks.N.*  stages.1.norm.*
  ...
  head.weight  head.bias
"""

from typing import ClassVar, final, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import init_transformer_trunc_normal
from lucid.models._utils._classification import DropPath
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.pvt._config import PVTConfig

# ---------------------------------------------------------------------------
# Overlapping patch embedding (also used as "downsample" for stages 1-3)
# ---------------------------------------------------------------------------


# Both references run PVT v2's block and stage LayerNorms at 1e-6, not
# the framework default of 1e-5.
_LN_EPS: float = 1e-6


@final
class _OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding via strided Conv2d + LayerNorm.

    Returns spatial feature map (B, C, H', W') where H'=H/stride, W'=W/stride.
    The overlapping kernel (size > stride) means adjacent windows share context,
    implicitly encoding position so explicit positional embeddings are unnecessary.
    """

    def __init__(self, in_ch: int, embed_dim: int, patch_size: int, stride: int) -> None:
        super().__init__()
        padding = patch_size // 2
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(x)  # (B, C, H', W')
        B, C, H, W = x.shape
        # permute to (B, H'*W', C), apply LN, return
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
# MLP with depthwise conv (PVT v2 key change)
# ---------------------------------------------------------------------------


@final
class _DWConvMLP(nn.Module):
    """Two-layer MLP with a DWConv after fc1 for spatial mixing.

    fc1 → reshape→ DWConv3×3 → reshape → GELU → fc2
    """

    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)

    @override
    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, _N, _C = x.shape
        x = self.fc1(x)
        # After fc1: (B, N, hidden) — reshape to spatial for DWConv
        hidden = x.shape[-1]
        x_2d = x.permute(0, 2, 1).reshape(B, hidden, H, W)
        x_2d = self.dwconv(x_2d)
        x = x_2d.reshape(B, hidden, H * W).permute(0, 2, 1)
        x = F.gelu(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Spatial Reduction Attention (SRA)
# ---------------------------------------------------------------------------


@final
class _SRAttention(nn.Module):
    """MHA with optional spatial reduction on K and V.

    When sr_ratio > 1, K and V are computed from a spatially reduced feature
    map (via a stride-sr_ratio Conv2d), reducing the O(N²) cost for large N.
    """

    def __init__(
        self, dim: int, num_heads: int, sr_ratio: int, linear: bool = False, pool_size: int = 7
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.linear = linear
        if linear:
            # Linear SRA (§3.3/§3.5): pool to a fixed P x P grid, then a 1x1
            # projection + norm + GELU.  Independent of ``sr_ratio``, which
            # is why B2-Li uses P_i = 7 at every stage instead of R_i.
            self.pool: nn.AdaptiveAvgPool2d | None = nn.AdaptiveAvgPool2d((pool_size, pool_size))
            self.sr = nn.Conv2d(dim, dim, 1, stride=1)
            self.norm = nn.LayerNorm(dim)
        elif sr_ratio > 1:
            self.pool = None
            self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.pool = None
            self.sr = None  # type: ignore[assignment]
            self.norm = None  # type: ignore[assignment]

    @override
    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, N, C = x.shape
        head_dim = C // self.num_heads

        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
            if self.pool is not None:
                x_2d = self.pool(x_2d)  # (B, C, P, P)
            x_2d = self.sr(x_2d)  # (B, C, H', W')
            x_2d = x_2d.flatten(2).permute(0, 2, 1)  # (B, H'*W', C)
            x_2d = self.norm(x_2d)
            if self.linear:
                x_2d = F.gelu(x_2d)
            kv_src = x_2d
        else:
            kv_src = x

        kv = self.kv(kv_src)
        N2 = kv_src.shape[1]
        kv = kv.reshape(B, N2, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Fused SDPA — never forms the (B,H,N,N2) score matrix; ``scale``
        # reproduces the manual ``* self.scale`` (no attention dropout here).
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)


# ---------------------------------------------------------------------------
# PVT v2 transformer block
# ---------------------------------------------------------------------------


@final
class _PVTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: float,
        drop_path_rate: float = 0.0,
        linear: bool = False,
        pool_size: int = 7,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=_LN_EPS)
        self.attn = _SRAttention(dim, num_heads, sr_ratio, linear, pool_size)
        self.norm2 = nn.LayerNorm(dim, eps=_LN_EPS)
        self.mlp = _DWConvMLP(dim, mlp_ratio)
        # Stochastic depth on both residual branches, as the reference does.
        self.drop_path = DropPath(drop_path_rate)

    @override
    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# ---------------------------------------------------------------------------
# One PVT stage (blocks + norm only; downsample handled outside)
# ---------------------------------------------------------------------------


@final
class _PVTStage(nn.Module):
    """One PVT v2 stage = optional downsample + N × Block + LayerNorm.

    For stage 0, there is no ``downsample`` sub-module — the patch embedding
    lives at top-level on the model (``self.patch_embed``).
    For stages 1-3, the patch embedding is stored as ``self.downsample`` to
    match the timm key layout (``stages.N.downsample.proj/norm``).
    """

    def __init__(
        self,
        in_ch: int,
        embed_dim: int,
        patch_size: int,
        stride: int,
        depth: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: float,
        is_first: bool,
        dpr: list[float] | None = None,
        linear: bool = False,
        pool_size: int = 7,
    ) -> None:
        super().__init__()
        dpr = dpr if dpr is not None else [0.0] * depth
        # Stage 0: patch_embed lives on the model, not inside the stage.
        # Stages 1-3: patch_embed stored as downsample (timm key layout).
        if not is_first:
            self.downsample = _OverlapPatchEmbed(in_ch, embed_dim, patch_size, stride)
        self.blocks: nn.ModuleList[_PVTBlock] = nn.ModuleList(
            [
                _PVTBlock(
                    embed_dim,
                    num_heads,
                    sr_ratio,
                    mlp_ratio,
                    drop_path_rate=dpr[j],
                    linear=linear,
                    pool_size=pool_size,
                )
                for j in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=_LN_EPS)

    def forward_tokens(self, tokens: Tensor, H: int, W: int) -> tuple[Tensor, int, int]:
        """Run blocks + norm on pre-embedded tokens (B, N, C)."""
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        tokens = self.norm(tokens)
        B, _, C = tokens.shape
        x_out = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return x_out, H, W

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        # x is a spatial map (B, C_in, H_in, W_in).
        # Stage 0's patch embedding lives on the model, not here, so this
        # path has nothing to downsample with.  Say so instead of raising an
        # AttributeError from the middle of the call.
        if not hasattr(self, "downsample"):
            raise RuntimeError(
                "stage 0 has no downsample — its patch embedding lives on the "
                "model as ``patch_embed``.  Call ``forward_tokens(tokens, H, W)`` "
                "with the already-embedded tokens instead."
            )
        tokens, H, W = self.downsample(x)
        return self.forward_tokens(tokens, H, W)


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_pvt(
    cfg: PVTConfig,
) -> tuple[
    _OverlapPatchEmbed,  # top-level patch_embed (stage 0)
    nn.ModuleList[_PVTStage],  # stages (0-indexed)
    list[FeatureInfo],
    int,  # final embed dim
]:
    """Build PVT v2 stages matching timm pvt_v2_bN key layout.

    timm layout:
      patch_embed  — top-level; handles stage-0 spatial downsampling
      stages.0     — blocks + norm only (no downsample submodule)
      stages.1-3   — downsample + blocks + norm
    """
    stages: list[_PVTStage] = []

    # Linear stochastic-depth schedule over the global block index.
    total = sum(cfg.depths)
    dpr_all = [cfg.drop_path_rate * k / max(total - 1, 1) for k in range(total)]
    cursor = 0
    fi: list[FeatureInfo] = []

    in_ch = cfg.in_channels
    reduction = 1

    # Build top-level patch_embed for stage 0
    patch_embed = _OverlapPatchEmbed(in_ch, cfg.embed_dims[0], patch_size=7, stride=4)
    reduction *= 4

    for i, (dim, depth, heads, sr, mlp_r) in enumerate(
        zip(cfg.embed_dims, cfg.depths, cfg.num_heads, cfg.sr_ratios, cfg.mlp_ratios, strict=True)
    ):
        is_first = i == 0
        # patch_size/stride used by downsample in stages 1-3
        patch_size = 3
        stride = 2
        if not is_first:
            reduction *= stride
        stages.append(
            _PVTStage(
                in_ch=in_ch,
                embed_dim=dim,
                patch_size=patch_size,
                stride=stride,
                depth=depth,
                num_heads=heads,
                sr_ratio=sr,
                mlp_ratio=mlp_r,
                is_first=is_first,
                dpr=dpr_all[cursor : cursor + depth],
                linear=cfg.linear_sra,
                pool_size=cfg.linear_pool_size,
            )
        )
        cursor += depth
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))
        in_ch = dim

    return patch_embed, nn.ModuleList(stages), fi, cfg.embed_dims[-1]


# ---------------------------------------------------------------------------
# PVT backbone
# ---------------------------------------------------------------------------


class PVT(PretrainedModel, BackboneMixin):
    r"""PVT v2 backbone (Wang et al., 2022).

    The Pyramid Vision Transformer v2 is a pure-transformer backbone
    that produces a *hierarchical* feature pyramid — making it a
    drop-in replacement for ResNet-style backbones in dense prediction
    tasks (detection, segmentation).  The trunk has four stages, each
    beginning with an *overlapping* convolutional patch embedding
    (:math:`7 \times 7` stride 4 for stage 0; :math:`3 \times 3`
    stride 2 for subsequent stages).  Inside each stage every
    transformer block applies *Spatial-Reduction Attention* (SRA):

    .. math::

        K, V \leftarrow \mathrm{LN}\!\bigl(
            W_R \cdot \mathrm{Reshape}_{R_i \times R_i}(x)\bigr),
        \qquad
        \mathrm{SRA}(x) = \mathrm{softmax}\!\left(
            \frac{Q K^\top}{\sqrt{d}}\right) V,

    where :math:`R_i` decreases with depth (e.g. :math:`8, 4, 2, 1`).
    This cuts the per-stage attention cost from
    :math:`\mathcal{O}((HW)^2)` to :math:`\mathcal{O}((HW)^2 / R_i^2)`.
    PVT v2 additionally inserts a depthwise :math:`3 \times 3`
    convolution between the two MLP linears, providing spatial mixing
    inside the feed-forward sublayer and removing the need for
    explicit positional embeddings.

    :meth:`forward_features` returns the mean-pooled final-stage
    feature :math:`(B, \texttt{embed\_dims[-1]})`.

    Parameters
    ----------
    config : PVTConfig
        Frozen dataclass specifying ``embed_dims``, ``depths``,
        ``num_heads``, ``sr_ratios``, ``mlp_ratios``, ``in_channels``,
        and ``num_classes``.  See :class:`PVTConfig`.

    Attributes
    ----------
    patch_embed : _OverlapPatchEmbed
        Top-level :math:`7 \times 7` stride-4 patch embedding feeding
        stage 0.
    stages : nn.ModuleList
        Four :class:`_PVTStage` modules; stages 1-3 each contain their
        own :math:`3 \times 3` stride-2 patch-embedding downsampler.
    feature_info : list[FeatureInfo]
        Four-stage feature description with cumulative reductions
        :math:`(4, 8, 16, 32)`.

    Notes
    -----
    Reference (v1): Wenhai Wang *et al.*, *"Pyramid Vision Transformer:
    A Versatile Backbone for Dense Prediction without Convolutions"*,
    ICCV 2021, `arXiv:2102.12122 <https://arxiv.org/abs/2102.12122>`_.
    Reference (v2): Wenhai Wang *et al.*, *"PVT v2: Improved Baselines
    with Pyramid Vision Transformer"*, CVMJ 2022.

    Examples
    --------
    Build a PVT v2-B1 backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.pvt import PVT, PVTConfig
    >>> model = PVT(PVTConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape                       # (B, embed_dims[-1])
    (1, 512)
    """

    config_class: ClassVar[type[PVTConfig]] = PVTConfig
    base_model_prefix: ClassVar[str] = "pvt"

    def __init__(self, config: PVTConfig) -> None:
        super().__init__(config)
        self.patch_embed, self.stages, fi, _out_dim = _build_pvt(config)
        self._feature_info = fi

        # Reference initialisation: trunc_normal_(0.02) on Linear / patch-embed
        # convs with zero bias, unit LayerNorms, and the same draw for the
        # class token and positional table (bare Parameters the framework
        # default never touches — they were sitting at exact zeros).
        init_transformer_trunc_normal(self)

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        # Stage 0: use top-level patch_embed then stage blocks
        tokens, H, W = self.patch_embed(x)
        x_spatial, H, W = self.stages[0].forward_tokens(tokens, H, W)
        # Stages 1-3: each stage calls its own downsample internally
        for stage in list(self.stages)[1:]:
            x_spatial, H, W = stage(x_spatial)
        # x_spatial: (B, C, H, W) — global average pool to (B, C)
        return x_spatial.flatten(2).mean(dim=2)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# PVT for image classification
# ---------------------------------------------------------------------------


class PVTForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""PVT v2 with a linear classification head (Wang et al., 2022).

    Wraps the same hierarchical four-stage trunk as :class:`PVT`
    (overlapping patch embeddings + Spatial-Reduction Attention) and
    adds a global mean pool + single :class:`nn.Linear` classification
    head:

    .. math::

        \text{logits} = W_{\text{head}}\,
            \mathrm{Mean}(z^{L}) + b_{\text{head}},
        \qquad W_{\text{head}} \in
            \mathbb{R}^{C_{\text{out}} \times d_{L}}.

    Pass ``labels`` to :meth:`forward` to compute the cross-entropy
    loss in the same pass.

    Parameters
    ----------
    config : PVTConfig
        Architecture specification.  Must set ``num_classes`` to the
        desired number of output categories.  See :class:`PVTConfig`.

    Attributes
    ----------
    patch_embed : _OverlapPatchEmbed
        Top-level :math:`7 \times 7` stride-4 patch embedding.
    stages : nn.ModuleList
        Four PVT v2 stages.
    classifier : nn.Linear
        Final linear projection of width
        ``(num_classes, embed_dims[-1])``.

    Notes
    -----
    Reference: Wang *et al.*, *"PVT v2: Improved Baselines with
    Pyramid Vision Transformer"*, CVMJ 2022.  PVT v2-B0 / B1 / B2 / B3
    / B4 / B5 reach **70.5 / 78.7 / 82.0 / 83.1 / 83.6 / 83.8 %
    top-1 on ImageNet-1k** at 224x224 (Wang et al., 2022, Table 1).

    Examples
    --------
    End-to-end inference with the default PVT v2-B1 classifier:

    >>> import lucid
    >>> from lucid.models.vision.pvt import (
    ...     PVTConfig, PVTForImageClassification,
    ... )
    >>> model = PVTForImageClassification(PVTConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """

    config_class: ClassVar[type[PVTConfig]] = PVTConfig
    base_model_prefix: ClassVar[str] = "pvt"

    def __init__(self, config: PVTConfig) -> None:
        super().__init__(config)
        self.patch_embed, self.stages, _, out_dim = _build_pvt(config)
        self._build_classifier(out_dim, config.num_classes)
        init_transformer_trunc_normal(self)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        # Stage 0: use top-level patch_embed then stage blocks
        tokens, H, W = self.patch_embed(x)
        x_spatial, H, W = self.stages[0].forward_tokens(tokens, H, W)
        # Stages 1-3
        for stage in list(self.stages)[1:]:
            x_spatial, H, W = stage(x_spatial)
        # x_spatial: (B, C, H, W) — mean pool to (B, C)
        x_vec = x_spatial.flatten(2).mean(dim=2)
        logits = self.classifier(x_vec)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
