"""EfficientFormer backbone and classifier (Li et al., 2022).

Paper: "EfficientFormer: Vision Transformers at MobileNet Speed"

Key ideas:
  1. MetaFormer insight: most of the capacity comes from the MLP, not the
     token mixer. Replace global attention with cheap local pooling in early
     stages.
  2. All-4D processing in early stages (B,C,H,W) — no reshape overhead.
  3. Stage 4 (last stage only): switch to standard MHA for global context,
     using a learned attention-bias table indexed by a relative-position
     grid (LeViT-style).
  4. Two stride-2 conv stem for efficient spatial downsampling.

The module tree mirrors the reference EfficientFormer implementation
verbatim so a near-identity weight remap loads the published checkpoints:

  stem.{conv1,norm1,conv2,norm2}
  stages.0.blocks.{0..}            (no downsample on stage 0)
  stages.{1,2,3}.downsample.{conv,norm}
  stages.{1,2,3}.blocks.{0..}
  norm                             (channel-last LayerNorm)
  head, head_dist                  (averaged at inference)

Architecture (EfficientFormer-L1, image=224):
  Stem   : Conv3x3(s=2) -> Conv3x3(s=2) -> (56x56, 48)
  Stage 1: 3 x MetaBlock2d(48)
  Stage 2: downsample -> 2 x MetaBlock2d(96)   -> (28x28, 96)
  Stage 3: downsample -> 6 x MetaBlock2d(224)  -> (14x14, 224)
  Stage 4: downsample -> 3 x MetaBlock2d(448) + 1 x MetaBlock1d(448) -> (7x7)
  Head   : Flat -> LN -> mean pool tokens -> (head + head_dist) / 2
"""

from dataclasses import dataclass
from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import init_transformer_trunc_normal
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._classification import DropPath, LayerScale
from lucid.models.vision.efficientformer._config import EfficientFormerConfig

# ---------------------------------------------------------------------------
# Stem: two stride-2 3x3 convolutions with BatchNorm + ReLU
# ---------------------------------------------------------------------------


class _Stem(nn.Module):
    """Stem reducing the input by 4x: Conv-BN-ReLU x2."""

    def __init__(self, in_chs: int, out_chs: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_chs, out_chs // 2, 3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(out_chs // 2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_chs // 2, out_chs, 3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(out_chs)
        self.act2 = nn.ReLU()

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        return self.act2(self.norm2(self.conv2(x)))


# ---------------------------------------------------------------------------
# Downsample between stages: strided conv + BatchNorm
# ---------------------------------------------------------------------------


class _Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.conv(x))


# ---------------------------------------------------------------------------
# 1x1-conv MLP with BatchNorm (used by the 4-D pooling blocks)
# ---------------------------------------------------------------------------


@final
class _ConvMlpWithNorm(nn.Module):
    """MLP via 1x1 convolutions over (B, C, H, W) with BatchNorm + GELU."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.norm1 = nn.BatchNorm2d(hidden)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.norm2 = nn.BatchNorm2d(dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(self.fc1(x))
        x = F.gelu(x)
        return self.norm2(self.fc2(x))


# ---------------------------------------------------------------------------
# Plain MLP (Linear-GELU-Linear) used by the 3-D attention blocks
# ---------------------------------------------------------------------------


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Pooling token mixer (parameter-free): AvgPool3x3 - identity
# ---------------------------------------------------------------------------


@final
class _Pooling(nn.Module):
    """AvgPool(pool_size, count_include_pad=False) - identity.

    The token mixer averages over *valid* (non-padded) neighbours only.
    The reference pooling sets ``count_include_pad=False`` so border
    positions divide by the actual window overlap rather than the full
    kernel area.  We realise this exactly by dividing the
    pad-included average by the pad-included average of an all-ones map
    (which equals ``valid_count / kernel_area`` at every position), then
    subtracting the identity branch.
    """

    def __init__(self, pool_size: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=pool_size, stride=1, padding=pool_size // 2, count_include_pad=True
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        # ``Pool(x) - x``, not Eqn 4's ``Pool(x) + x``.  Both references do
        # this: the mixer returns the pooling *residual*, and the enclosing
        # block adds it back as ``x + LayerScale(mixer(x))`` — so the sum
        # Eqn 4 writes is formed one level up, and subtracting here is what
        # stops the identity being counted twice.  Inherited from
        # PoolFormer, which EfficientFormer's MB4D block is built on.
        num = self.pool(x)
        den = self.pool(lucid.ones_like(x))
        return num / den - x


# ---------------------------------------------------------------------------
# Attention token mixer with learned relative-position bias (LeViT-style)
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    """Multi-head attention with a learned attention-bias table.

    Mirrors the reference EfficientFormer attention: a single ``qkv``
    projection whose key/query dimension is ``key_dim`` and whose value
    dimension is ``attn_ratio * key_dim`` per head, plus a ``proj`` back
    to ``dim``.  A per-head bias table of ``resolution ** 2`` entries is
    gathered through a fixed relative-position index grid and added to the
    pre-softmax scores.
    """

    attention_bias_idxs: Tensor

    def __init__(
        self, dim: int, key_dim: int, num_heads: int, attn_ratio: float, resolution: int
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        # Both references scale by the per-head key dim (32 -> 0.17678).  A
        # literal reading of Eqn 6 scales by the stage *width* C_j
        # (448/512/768), giving ~0.047 for L1 — the shipped checkpoints were
        # trained under the reference convention, so that is what runs here.
        self.scale = key_dim**-0.5
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * num_heads
        key_attn_dim = key_dim * num_heads
        self.resolution = resolution

        self.qkv = nn.Linear(dim, key_attn_dim * 2 + self.val_attn_dim)
        self.proj = nn.Linear(self.val_attn_dim, dim)

        n = resolution * resolution
        self.attention_biases = nn.Parameter(lucid.zeros(num_heads, n))
        self._init_bias_idxs(resolution)

    def _init_bias_idxs(self, resolution: int) -> None:
        # Row-major (resolution x resolution) position grid; the relative
        # index for a pair is |dy| * resolution + |dx|.  ``abs`` is computed
        # through a float round-trip — integer ``abs`` is not available on
        # the integer tensor path, and the indices are tiny so the cast is
        # exact.
        r = lucid.arange(resolution).to(lucid.int64)
        n = resolution * resolution
        ys = r.reshape(resolution, 1).expand(resolution, resolution).reshape(-1)
        xs = r.reshape(1, resolution).expand(resolution, resolution).reshape(-1)
        rel_y = (ys.reshape(n, 1) - ys.reshape(1, n)).to(lucid.float32).abs()
        rel_x = (xs.reshape(n, 1) - xs.reshape(1, n)).to(lucid.float32).abs()
        idx = (rel_y * resolution + rel_x).to(lucid.int64)  # (n, n)
        self.register_buffer("attention_bias_idxs", idx, persistent=False)

    def _bias(self) -> Tensor:
        idx = self.attention_bias_idxs
        n = self.resolution * self.resolution
        gathered = self.attention_biases[:, idx.reshape(-1)]  # (heads, n*n)
        return gathered.reshape(self.num_heads, n, n)

    @override
    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        # The attention-bias table is built for a fixed ``resolution**2``
        # token grid, so a differently-sized input cannot be served.  Say so
        # here rather than letting the bias reshape below fail with a bare
        # ShapeMismatch that names neither the input size nor the knob.
        expected = self.resolution * self.resolution
        if N != expected:
            side = int(round(N**0.5))
            raise ValueError(
                f"attention got {N} tokens but its bias table is built for "
                f"{expected} ({self.resolution}x{self.resolution}).  The token "
                f"grid is {side}x{side}, i.e. an input of {side * 32}px against "
                f"the {self.resolution * 32}px this model was configured for. "
                f"Rebuild with resolution={side} (and the matching image size) "
                f"to run at that resolution."
            )
        qkv = self.qkv(x)  # (B, N, key*2 + val) * heads
        qkv = qkv.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        q = qkv[:, :, :, : self.key_dim]
        k = qkv[:, :, :, self.key_dim : 2 * self.key_dim]
        v = qkv[:, :, :, 2 * self.key_dim :]

        # Fused SDPA with the learned attention-bias table as an additive mask.
        # ``scale=key_dim**-0.5`` matches q's per-head dim; v carries a distinct
        # ``val_dim`` head width, which SDPA supports.
        bias = self._bias().reshape(1, self.num_heads, N, N)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.val_attn_dim)
        return self.proj(out)


# ---------------------------------------------------------------------------
# MetaBlock2d: 4-D pooling block (stages 1-3 + early blocks of last stage)
# ---------------------------------------------------------------------------


@final
class _MetaBlock2d(nn.Module):
    """Pooling MetaFormer block operating in (B, C, H, W) layout."""

    def __init__(
        self,
        dim: int,
        pool_size: int,
        mlp_ratio: float,
        drop_path_rate: float,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.token_mixer = _Pooling(pool_size)
        self.ls1 = LayerScale(dim, layer_scale_init)
        self.drop_path1 = DropPath(drop_path_rate)
        self.mlp = _ConvMlpWithNorm(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, layer_scale_init)
        self.drop_path2 = DropPath(drop_path_rate)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.token_mixer(x)))
        return x + self.drop_path2(self.ls2(self.mlp(x)))


# ---------------------------------------------------------------------------
# MetaBlock1d: 3-D attention block (last stage only)
# ---------------------------------------------------------------------------


@final
class _MetaBlock1d(nn.Module):
    """Attention MetaFormer block operating in (B, N, C) layout."""

    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int,
        attn_ratio: float,
        resolution: int,
        mlp_ratio: float,
        drop_path_rate: float,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = _Attention(dim, key_dim, num_heads, attn_ratio, resolution)
        self.ls1 = LayerScale(dim, layer_scale_init)
        self.drop_path1 = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, layer_scale_init)
        self.drop_path2 = DropPath(drop_path_rate)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.token_mixer(self.norm1(x))))
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


# ---------------------------------------------------------------------------
# Flat: (B, C, H, W) -> (B, N, C) reshape (parameter-free placeholder block)
# ---------------------------------------------------------------------------


@final
class _Flat(nn.Module):
    """Flatten spatial dims and move channels last: (B,C,H,W) -> (B,H*W,C)."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(2).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Stage: optional downsample + a Sequence of (2d / Flat / 1d) blocks
# ---------------------------------------------------------------------------


class _Stage(nn.Module):
    """One EfficientFormer stage: ``downsample`` (or identity) + ``blocks``.

    The ``blocks`` index layout matches the reference implementation
    exactly — including the parameter-free :class:`_Flat` placeholder that
    transitions from 4-D pooling blocks to 3-D attention blocks — so the
    published checkpoint loads with an identity key map.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        downsample: bool,
        num_vit: int,
        cfg: EfficientFormerConfig,
        dp_rates: list[float],
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        if downsample:
            self.downsample: _Downsample | nn.Identity = _Downsample(in_dim, out_dim)
            dim = out_dim
        else:
            self.downsample = nn.Identity()
            dim = in_dim

        blocks: list[_Flat | _MetaBlock1d | _MetaBlock2d] = []
        if num_vit and num_vit >= depth:
            blocks.append(_Flat())

        for block_idx in range(depth):
            remain_idx = depth - block_idx - 1
            if num_vit and num_vit > remain_idx:
                blocks.append(
                    _MetaBlock1d(
                        dim,
                        cfg.key_dim,
                        cfg.num_heads,
                        cfg.attn_ratio,
                        cfg.resolution,
                        mlp_ratio,
                        dp_rates[block_idx],
                        cfg.layer_scale_init,
                    )
                )
            else:
                blocks.append(
                    _MetaBlock2d(
                        dim, cfg.pool_size, mlp_ratio, dp_rates[block_idx], cfg.layer_scale_init
                    )
                )
                if num_vit and num_vit == remain_idx:
                    blocks.append(_Flat())

        self.blocks: nn.ModuleList[_Flat | _MetaBlock1d | _MetaBlock2d] = nn.ModuleList(blocks)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_efficientformer(
    cfg: EfficientFormerConfig,
) -> tuple[_Stem, nn.ModuleList[_Stage], nn.LayerNorm, list[FeatureInfo], int]:
    stem = _Stem(cfg.in_channels, cfg.embed_dims[0])

    num_stages = len(cfg.depths)
    last_stage = num_stages - 1

    # Stage-wise linear stochastic-depth schedule (paper §4.1).
    total = sum(cfg.depths)
    if total > 1 and cfg.drop_path_rate > 0.0:
        flat = [cfg.drop_path_rate * i / (total - 1) for i in range(total)]
    else:
        flat = [cfg.drop_path_rate] * total

    stages: list[_Stage] = []
    fi: list[FeatureInfo] = []
    cursor = 0
    prev_dim = cfg.embed_dims[0]
    for i, (depth, dim) in enumerate(zip(cfg.depths, cfg.embed_dims, strict=True)):
        dp_rates = flat[cursor : cursor + depth]
        cursor += depth
        stages.append(
            _Stage(
                prev_dim,
                dim,
                depth,
                downsample=(i > 0),
                num_vit=cfg.num_vit if i == last_stage else 0,
                cfg=cfg,
                dp_rates=dp_rates,
                # ``mlp_ratios`` is declared and documented as *per-stage*;
                # every stage was reading the last entry, so the stage-4 ratio
                # governed all four and a non-uniform tuple did nothing.
                mlp_ratio=cfg.mlp_ratios[i],
            )
        )
        prev_dim = dim
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=2 ** (i + 2)))

    head_norm = nn.LayerNorm(cfg.embed_dims[-1])
    return stem, nn.ModuleList(stages), head_norm, fi, cfg.embed_dims[-1]


def _forward_trunk(
    stem: _Stem, stages: nn.ModuleList[_Stage], head_norm: nn.LayerNorm, x: Tensor
) -> Tensor:
    """Run stem + stages, ending in channel-last (B, N, C), then LayerNorm."""
    x = stem(x)
    last = len(stages) - 1
    for i, stage in enumerate(stages):
        x = stage(x)
        if i == last and x.ndim == 4:
            # No attention block flattened the tensor (num_vit == 0): flatten now.
            x = x.flatten(2).permute(0, 2, 1)
    return head_norm(x)  # (B, N, C)


# ---------------------------------------------------------------------------
# EfficientFormer backbone
# ---------------------------------------------------------------------------


class EfficientFormer(PretrainedModel, BackboneMixin):
    r"""EfficientFormer backbone (Li et al., 2022).

    EfficientFormer is a *mobile-grade* vision backbone designed so
    that its on-device latency, rather than its FLOP count, matches
    MobileNet on the same hardware while retaining transformer-level
    accuracy.  The trunk has four stages preceded by a two stride-2
    convolutional stem.  Stages 1-3 are MetaFormer-style *pooling*
    blocks operating in :math:`(B, C, H, W)` layout — no reshape, no
    self-attention:

    .. math::

        \begin{aligned}
        x &\leftarrow x + \gamma_1 \odot \bigl(
            \mathrm{AvgPool}_{3 \times 3}(x) - x\bigr), \\
        x &\leftarrow x + \gamma_2 \odot \mathrm{ConvMLP}(x),
        \end{aligned}

    with :math:`\gamma_1, \gamma_2` initialised to :math:`10^{-5}`
    (CaiT-style layer scale).  The last stage switches its trailing
    :math:`\texttt{num\_vit}` blocks to standard multi-head
    self-attention with a learned relative-position bias on the
    now-tiny token grid.

    :meth:`forward_features` returns the mean-pooled
    :math:`(B, \texttt{embed\_dims[-1]})` feature.

    Parameters
    ----------
    config : EfficientFormerConfig
        Frozen dataclass specifying ``depths``, ``embed_dims``,
        ``mlp_ratios``, ``num_vit``, ``drop_path_rate``,
        ``layer_scale_init``, ``in_channels``, and ``num_classes``.
        See :class:`EfficientFormerConfig`.

    Attributes
    ----------
    stem : _Stem
        Two stride-2 :math:`3 \times 3` convolutions reducing the
        input by 4x.
    stages : nn.ModuleList
        Four stages — three pooling stages and one mixed
        pooling/attention stage; stages 1-3 carry a downsample.
    norm : nn.LayerNorm
        Final LayerNorm applied to the channel-last token sequence.
    feature_info : list[FeatureInfo]
        Four-stage feature description with reductions
        :math:`(4, 8, 16, 32)`.

    Notes
    -----
    Reference: Yanyu Li *et al.*, *"EfficientFormer: Vision
    Transformers at MobileNet Speed"*, NeurIPS 2022,
    `arXiv:2206.01191 <https://arxiv.org/abs/2206.01191>`_.

    Examples
    --------
    Build an EfficientFormer-L1 backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.efficientformer import (
    ...     EfficientFormer, EfficientFormerConfig,
    ... )
    >>> model = EfficientFormer(EfficientFormerConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape                       # (B, embed_dims[-1])
    (1, 448)
    """

    config_class: ClassVar[type[EfficientFormerConfig]] = EfficientFormerConfig
    base_model_prefix: ClassVar[str] = "efficientformer"

    def __init__(self, config: EfficientFormerConfig) -> None:
        super().__init__(config)
        stem, stages, hn, fi, _out_dim = _build_efficientformer(config)
        self.stem = stem
        self.stages = stages
        self.norm = hn
        self._feature_info = fi

        # Reference initialisation: trunc_normal_(0.02) on Linear / patch-embed
        # convs with zero bias, unit LayerNorms, and the same draw for the
        # class token and positional table (bare Parameters the framework
        # default never touches — they were sitting at exact zeros).
        init_transformer_trunc_normal(self)

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        seq = _forward_trunk(self.stem, self.stages, self.norm, x)  # (B, N, C)
        return seq.mean(dim=1)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# EfficientFormer for image classification (distilled dual head)
# ---------------------------------------------------------------------------


@dataclass
class EfficientFormerOutput(ImageClassificationOutput):
    r"""Classification output carrying both distillation heads.

    The reference keeps ``head`` and ``head_dist`` separate in training —
    that separation is the whole point of a distilled dual head, since the
    two are supervised by different targets (ground truth and the teacher).
    Averaging before the loss trains a single averaged predictor instead,
    and left the individual logits unreachable to any caller wanting to
    apply a distillation objective.

    ``logits`` stays the averaged value the reference uses at inference, so
    existing consumers are unaffected.

    Attributes
    ----------
    head_logits : Tensor or None, optional
        Logits from the ground-truth classification head.
    head_dist_logits : Tensor or None, optional
        Logits from the distillation head.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.efficientformer._model import (
    ...     EfficientFormerOutput,
    ... )
    >>> out = EfficientFormerOutput(logits=lucid.zeros(1, 1000))
    >>> out.logits.shape
    (1, 1000)

    ``head_logits`` and ``head_dist_logits`` hold the two heads apart
    when distillation is on; ``logits`` is their average, which is what
    inference should read.

    >>> out.head_logits is None, out.head_dist_logits is None
    (True, True)
    """

    head_logits: Tensor | None = None
    head_dist_logits: Tensor | None = None


class EfficientFormerForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""EfficientFormer with a distilled dual classification head (Li et al., 2022).

    Wraps the same trunk as :class:`EfficientFormer` and adds a final
    LayerNorm, a mean pool over tokens, and *two* linear heads — the
    classification head and the distillation head — whose logits are
    averaged at inference:

    .. math::

        \text{logits} = \tfrac{1}{2}\bigl(
            W_{\text{head}}\,z + b_{\text{head}}
            + W_{\text{dist}}\,z + b_{\text{dist}}\bigr),
        \qquad
        z = \mathrm{Mean}(\mathrm{LN}(z^{L})).

    The published EfficientFormer checkpoints were trained with hard
    knowledge distillation (DeiT-style), so reproducing the reported
    top-1 accuracy requires averaging both heads.

    Pass ``labels`` to :meth:`forward` to compute the cross-entropy
    loss in the same pass.

    Parameters
    ----------
    config : EfficientFormerConfig
        Architecture specification.  Must set ``num_classes`` to the
        desired number of output categories.  See
        :class:`EfficientFormerConfig`.

    Attributes
    ----------
    stem : _Stem
        Two stride-2 convolutions.
    stages : nn.ModuleList
        Three pooling stages + one mixed pooling/attention stage.
    norm : nn.LayerNorm
        Final LayerNorm.
    head : nn.Linear
        Classification head ``(num_classes, embed_dims[-1])``.
    head_dist : nn.Linear
        Distillation head ``(num_classes, embed_dims[-1])``; averaged
        with ``head`` at inference.

    Notes
    -----
    Reference: Li *et al.*, *"EfficientFormer: Vision Transformers at
    MobileNet Speed"*, NeurIPS 2022.  EfficientFormer-L1 / L3 / L7
    reach **79.2 / 82.4 / 83.3 % top-1 on ImageNet-1k** at MobileNetV2-
    class on-device latency (Li et al., 2022, Table 4).

    Examples
    --------
    End-to-end inference with the default EfficientFormer-L1
    classifier:

    >>> import lucid
    >>> from lucid.models.vision.efficientformer import (
    ...     EfficientFormerConfig, EfficientFormerForImageClassification,
    ... )
    >>> model = EfficientFormerForImageClassification(EfficientFormerConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """

    config_class: ClassVar[type[EfficientFormerConfig]] = EfficientFormerConfig
    base_model_prefix: ClassVar[str] = "efficientformer"

    def __init__(self, config: EfficientFormerConfig) -> None:
        super().__init__(config)
        stem, stages, hn, _, out_dim = _build_efficientformer(config)
        self.stem = stem
        self.stages = stages
        self.norm = hn
        self.head = nn.Linear(out_dim, config.num_classes)
        self.head_dist = nn.Linear(out_dim, config.num_classes)
        init_transformer_trunc_normal(self)

    @override
    def reset_classifier(self, num_classes: int) -> None:
        in_features = int(self.head.in_features)
        self.head = nn.Linear(in_features, num_classes)
        self.head_dist = nn.Linear(in_features, num_classes)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> EfficientFormerOutput:
        seq = _forward_trunk(self.stem, self.stages, self.norm, x)  # (B, N, C)
        feat = seq.mean(dim=1)
        head_logits = self.head(feat)
        head_dist_logits = self.head_dist(feat)
        logits = (head_logits + head_dist_logits) / 2.0

        loss: Tensor | None = None
        if labels is not None:
            # Supervise each head, rather than their average.  With no teacher
            # supplied both see the ground truth; a caller doing real
            # distillation reads ``head_dist_logits`` and applies its own
            # objective.
            loss = (
                F.cross_entropy(head_logits, labels) + F.cross_entropy(head_dist_logits, labels)
            ) / 2.0

        return EfficientFormerOutput(
            logits=logits, loss=loss, head_logits=head_logits, head_dist_logits=head_dist_logits
        )
