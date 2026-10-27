"""Swin Transformer backbone and classifier (Liu et al., 2021).

Paper: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"

Key ideas vs plain ViT:
  1. Hierarchical feature maps (4 stages, 2× spatial downsampling between stages).
  2. Local window attention — each token only attends within its W×W window.
  3. Shifted window partition (alternating between two offset grids) enables
     cross-window information flow without global attention.
  4. Relative position bias added to attention logits.

Architecture (Swin-T, image=224, patch=4, window=7):
  PatchEmbed : Conv2d(4×4, stride=4) → (56×56, 96)
  Stage 1    : 2 × SwinBlock(window) → PatchMerge → (28×28, 192)
  Stage 2    : 2 × SwinBlock(window) → PatchMerge → (14×14, 384)
  Stage 3    : 6 × SwinBlock(window) → PatchMerge → (7×7,  768)
  Stage 4    : 2 × SwinBlock(window)              → (7×7,  768)
  Head       : LayerNorm → AdaptiveAvgPool(1×1) → FC

Each SwinBlock:
  LayerNorm → WindowAttention (w/ rel-pos bias) → residual
  LayerNorm → MLP(GELU) → residual
  Alternating blocks use cyclic shift + mask to implement shifted windows.
"""

from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._engine_scope import untraced
from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import init_transformer_trunc_normal
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._classification import DropPath
from lucid.models.vision.swin._config import SwinConfig

# ---------------------------------------------------------------------------
# Patch embedding (non-overlapping, stride = patch_size)
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, patch_size: int, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        # The reference holds ``pos_drop = nn.Dropout(drop_rate)`` and applies
        # it to the patch-embedding output before stage 1; nothing sat on that
        # path here, so ``config.dropout`` never reached it.
        self.pos_drop = nn.Dropout(p=dropout)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        return self.pos_drop(x)


# ---------------------------------------------------------------------------
# Patch merging (spatial 2× downsampling + channel 2× expansion)
# ---------------------------------------------------------------------------


@final
class _PatchMerge(nn.Module):
    """Concat 2x2 neighbours, normalise, project 4C -> 2C.

    Table 7 writes the downsampler as "concat 2x2, 192-d, LN", which read
    literally is concat -> Linear -> LayerNorm.  Both reference
    implementations do concat -> LayerNorm -> Linear, and the shipped
    checkpoints were trained that way, so that order is what runs here.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.proj = nn.Linear(4 * dim, 2 * dim, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Input x — (B, H, W, C)
        B, H, W, C = x.shape
        # The reference pads before merging when a side is odd; the strided
        # slices below otherwise drop the last row/column silently (or
        # mismatch, since 0::2 and 1::2 differ in length at odd sizes).
        if H % 2 or W % 2:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            H, W = H + H % 2, W + W % 2
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = lucid.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = self.norm(x)
        return self.proj(x)  # (B, H/2, W/2, 2C)


# ---------------------------------------------------------------------------
# Window partition / reverse helpers
# ---------------------------------------------------------------------------


def _window_partition(x: Tensor, ws: int) -> tuple[Tensor, int, int]:
    """Split (B, H, W, C) into (num_windows*B, ws, ws, C)."""
    B, H, W, C = x.shape
    x = x.reshape(B, H // ws, ws, W // ws, ws, C)
    # (B, nH, ws, nW, ws, C) → (B*nH*nW, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)
    return x, H // ws, W // ws


def _window_reverse(windows: Tensor, ws: int, nH: int, nW: int) -> Tensor:
    """Reverse of _window_partition: (B*nH*nW, ws, ws, C) → (B, H, W, C)."""
    B_total = windows.shape[0]
    B = B_total // (nH * nW)
    C = windows.shape[-1]
    x = windows.reshape(B, nH, nW, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, nH * ws, nW * ws, C)


# ---------------------------------------------------------------------------
# Window Multi-Head Self-Attention with relative position bias
# ---------------------------------------------------------------------------


@final
class _WindowAttention(nn.Module):
    """Local window attention with learnable relative position bias."""

    rel_pos_idx: Tensor

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(p=attn_drop)
        # The reference wires ``proj_drop`` to the same rate as the MLP
        # dropout and applies it right after the output projection; without
        # it ``config.dropout`` regularised the MLP branch only.
        self.proj_drop = nn.Dropout(p=proj_drop)

        # Relative position bias table: (2W-1)^2 × num_heads
        n = (2 * window_size - 1) ** 2
        self.rel_pos_bias = nn.Parameter(lucid.zeros(n, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        # Pre-compute relative position index as int64 (no in-place ops).
        # Registered as a non-persistent buffer so that .to(device=...) moves
        # it together with the model (and to keep it out of state_dicts).
        coords_1d = lucid.arange(window_size).to(lucid.int64)
        gy, gx = lucid.meshgrid(coords_1d, coords_1d, indexing="ij")  # (ws, ws)
        flat_y, flat_x = gy.flatten(), gx.flatten()  # (ws^2,)
        rel_y = flat_y.unsqueeze(1) - flat_y.unsqueeze(0) + (window_size - 1)  # (ws^2, ws^2)
        rel_x = flat_x.unsqueeze(1) - flat_x.unsqueeze(0) + (window_size - 1)
        rel_idx = rel_y * (2 * window_size - 1) + rel_x  # (ws^2, ws^2)
        self.register_buffer("rel_pos_idx", rel_idx, persistent=False)

    @override
    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        B_, N, C = x.shape  # B_ = num_windows*B
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.permute(0, 1, 3, 2)) * self.scale

        # Relative position bias.  When the feature map is smaller than the
        # configured window the block clamps down to it, so derive the index
        # for the *effective* window — its offsets are a sub-range of the
        # table's, so the same learned parameters still apply.
        eff = int(round(float(N) ** 0.5))
        if eff == self.ws:
            rel_pos_idx = self.rel_pos_idx
        else:
            c1d = lucid.arange(eff).to(lucid.int64)
            gy_e, gx_e = lucid.meshgrid(c1d, c1d, indexing="ij")
            fy, fx = gy_e.flatten(), gx_e.flatten()
            ry = fy.unsqueeze(1) - fy.unsqueeze(0) + (self.ws - 1)
            rx = fx.unsqueeze(1) - fx.unsqueeze(0) + (self.ws - 1)
            rel_pos_idx = ry * (2 * self.ws - 1) + rx
        idx = rel_pos_idx.reshape(-1)
        bias = self.rel_pos_bias[idx].reshape(N, N, self.num_heads).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias

        if mask is not None:
            # mask: (num_windows, N, N) with -100 for masked positions
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 1, 3).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# Swin Transformer block
# ---------------------------------------------------------------------------


@final
class _SwinBlock(nn.Module):
    """One Swin Transformer block (regular or shifted window)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift: bool,
        mlp_ratio: float,
        dropout: float,
        attn_drop: float,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.ws = window_size
        self.shift = shift
        self.shift_size = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, window_size, num_heads, attn_drop, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(p=dropout),
        )
        self.drop_path = DropPath(drop_path_rate)
        # The shifted-window mask depends only on the padded resolution, the
        # shift and the window — never on the input values — so it is built
        # once per distinct geometry instead of on every forward.  Derived
        # state, so deliberately not a registered buffer: it must not enter
        # ``state_dict`` and it must not be restored from a checkpoint.
        self._mask_cache: dict[tuple[int, int, int, int, str, str], Tensor] = {}

    def _attn_mask(
        self, H: int, W: int, shift_size: int, like: Tensor, ws: int | None = None
    ) -> Tensor | None:
        """The shifted-window mask, on ``like``'s device and in its dtype.

        Additive to the attention scores, so it takes their dtype; its
        values (region ids, ``-100``) are exact in half precision.

        Built outside any active trace, so a recording always holds the mask
        as a constant feed and never the ops that make it.  Built inside, a
        block's first call put them in the graph and every later call found
        the cached mask, so an export depended on what the block had run
        before (LCD-627).
        """
        if shift_size == 0:
            return None
        ws = self.ws if ws is None else ws
        ss = shift_size
        device, dtype = like.device.type, like.dtype
        with untraced():
            img_mask = lucid.zeros(1, H, W, 1, dtype=dtype, device=device)
            slices_h = [slice(0, -ws), slice(-ws, -ss), slice(-ss, None)]
            slices_w = [slice(0, -ws), slice(-ws, -ss), slice(-ss, None)]
            cnt = 0
            for sh in slices_h:
                for sw in slices_w:
                    img_mask[0, sh, sw, 0] = cnt
                    cnt += 1
            mask_windows, nH, nW = _window_partition(img_mask, ws)  # (nW, ws, ws, 1)
            mask_windows = mask_windows.reshape(-1, ws * ws)  # (nW, ws^2)
            # (nW, ws^2, ws^2): non-zero where two tokens come from different
            # regions, which is where the mask blocks attention with -100.
            mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            return lucid.where(
                mask != 0,
                lucid.full(mask.shape, -100.0, dtype=dtype, device=device),
                lucid.zeros(mask.shape, dtype=dtype, device=device),
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # When the whole feature map fits inside a single window, the
        # reference disables the cyclic shift *and* clamps the window down to
        # the map — the second half was missing, so ``_window_partition`` was
        # still handed the configured window and produced a zero-sized grid.
        eff_ws = min(self.ws, H, W)
        eff_ss = 0 if eff_ws >= min(H, W) else self.shift_size

        # Section 3.2 footnote 4: "bottom-right padding is employed on the
        # feature map if needed" so the window size divides it.  Without this
        # any resolution that is not a multiple of the window raised a
        # ShapeMismatch part-way through the forward.
        pad_h = (eff_ws - H % eff_ws) % eff_ws
        pad_w = (eff_ws - W % eff_ws) % eff_ws
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp = H + pad_h
        Wp = W + pad_w

        if eff_ss > 0:
            x = lucid.roll(x, [-eff_ss, -eff_ss], dims=[1, 2])

        dev = x.device.type
        cache_key = (Hp, Wp, eff_ss, eff_ws, dev, str(x.dtype))
        mask = self._mask_cache.get(cache_key)
        if mask is None and eff_ss > 0:
            mask = self._attn_mask(Hp, Wp, eff_ss, x, ws=eff_ws)
            if mask is not None:
                self._mask_cache[cache_key] = mask
        windows, nH, nW = _window_partition(x, eff_ws)
        windows = windows.reshape(-1, eff_ws * eff_ws, C)

        attn_out = self.attn(windows, mask=mask)
        attn_out = attn_out.reshape(-1, eff_ws, eff_ws, C)
        x = _window_reverse(attn_out, eff_ws, nH, nW)

        if eff_ss > 0:
            x = lucid.roll(x, [eff_ss, eff_ss], dims=[1, 2])

        # Crop the padding back off before the residual add.
        if pad_h or pad_w:
            x = x[:, :H, :W, :]

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Swin stage (sequence of blocks + optional patch merge)
# ---------------------------------------------------------------------------


class _SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        attn_drop: float,
        downsample: bool,
        drop_path_rates: list[float] | None = None,
    ) -> None:
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        if len(drop_path_rates) != depth:
            raise ValueError(
                f"drop_path_rates length must equal depth ({len(drop_path_rates)} vs {depth})"
            )
        self.blocks: nn.ModuleList[_SwinBlock] = nn.ModuleList(
            [
                _SwinBlock(
                    dim,
                    num_heads,
                    window_size,
                    shift=(i % 2 == 1),
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    drop_path_rate=drop_path_rates[i],
                )
                for i in range(depth)
            ]
        )
        self.downsample: _PatchMerge | None = _PatchMerge(dim) if downsample else None

    @override
    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_swin(
    cfg: SwinConfig,
) -> tuple[_PatchEmbed, nn.ModuleList[_SwinStage], nn.LayerNorm, list[FeatureInfo], int]:
    patch_embed = _PatchEmbed(cfg.in_channels, cfg.patch_size, cfg.embed_dim, cfg.dropout)

    stages: list[_SwinStage] = []
    dim = cfg.embed_dim
    fi: list[FeatureInfo] = []
    reduction = cfg.patch_size

    # Linear stochastic-depth schedule across the trunk (Liu 2021 §A).
    # ``linspace(0, drop_path_rate, total_blocks)``: the first block never
    # drops and the last drops at the configured rate.  A one-block trunk is
    # the degenerate case of that line, so it gets 0.0 — not the max rate,
    # which would drop the only block in the network.
    total_blocks = sum(cfg.depths)
    dp_all = [
        cfg.drop_path_rate * i / (total_blocks - 1) if total_blocks > 1 else 0.0
        for i in range(total_blocks)
    ]
    block_cursor = 0

    for i, (depth, heads) in enumerate(zip(cfg.depths, cfg.num_heads, strict=True)):
        downsample = i < len(cfg.depths) - 1
        stage_dp = dp_all[block_cursor : block_cursor + depth]
        block_cursor += depth
        stages.append(
            _SwinStage(
                dim,
                depth,
                heads,
                cfg.window_size,
                cfg.mlp_ratio,
                cfg.dropout,
                cfg.attention_dropout,
                downsample,
                drop_path_rates=list(stage_dp),
            )
        )
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))
        if downsample:
            reduction *= 2
            dim *= 2

    norm = nn.LayerNorm(dim)
    return patch_embed, nn.ModuleList(stages), norm, fi, dim


# ---------------------------------------------------------------------------
# Swin Transformer backbone  (task="base")
# ---------------------------------------------------------------------------


class SwinTransformer(PretrainedModel, BackboneMixin):
    r"""Swin Transformer backbone (Liu et al., 2021).

    A Swin Transformer first splits the input image into non-overlapping
    :math:`4 \times 4` patches via a strided convolution and then
    progresses through four hierarchical stages.  Each stage stacks
    pairs of Swin blocks alternating between *window* attention (W-MSA)
    and *shifted-window* attention (SW-MSA), with patch-merging
    downsamplers between stages that halve spatial resolution and
    double channel width:

    .. math::

        \begin{aligned}
        \hat{z}^{l} &= \mathrm{W\text{-}MSA}(\mathrm{LN}(z^{l-1})) + z^{l-1}, \\
        z^{l} &= \mathrm{MLP}(\mathrm{LN}(\hat{z}^{l})) + \hat{z}^{l}, \\
        \hat{z}^{l+1} &= \mathrm{SW\text{-}MSA}(\mathrm{LN}(z^{l})) + z^{l}, \\
        z^{l+1} &= \mathrm{MLP}(\mathrm{LN}(\hat{z}^{l+1})) + \hat{z}^{l+1}.
        \end{aligned}

    Window attention partitions the feature map into non-overlapping
    :math:`M \times M` windows and runs self-attention inside each,
    making the per-image cost linear in :math:`HW`.  The shifted variant
    cycles each window by :math:`\lfloor M/2 \rfloor` pixels so that
    information flows between neighbouring windows across pairs of
    blocks.  A learnable *relative position bias* is added inside the
    softmax of every window attention.

    :meth:`forward_features` returns a global-average-pooled
    :math:`(B, 8C)` feature where :math:`C = \texttt{embed\_dim}`.  Use
    this backbone when you need features rather than logits — e.g.
    transfer learning, dense prediction, or contrastive pretraining.
    For end-to-end image classification, use
    :class:`SwinTransformerForImageClassification` instead.

    Parameters
    ----------
    config : SwinConfig
        Frozen dataclass specifying ``image_size``, ``patch_size``,
        ``embed_dim``, ``depths``, ``num_heads``, ``window_size``,
        ``mlp_ratio``, ``dropout``, ``attention_dropout``,
        ``drop_path_rate``, and ``in_channels``.  See :class:`SwinConfig`.

    Attributes
    ----------
    patch_embed : _PatchEmbed
        Strided patch-extraction convolution + LayerNorm.
    stages : nn.ModuleList
        Four hierarchical :class:`_SwinStage` modules, each containing a
        stack of Swin blocks and (for the first three) a patch-merging
        downsampler.
    norm : nn.LayerNorm
        Final LayerNorm applied to the channel-last feature map before
        global pooling.
    avgpool : nn.AdaptiveAvgPool2d
        :math:`1 \times 1` adaptive average pool used to produce the
        final :math:`(B, 8C)` feature.
    feature_info : list[FeatureInfo]
        Four-stage feature description with per-stage channel counts
        :math:`(C, 2C, 4C, 8C)` and spatial reductions
        :math:`(p, 2p, 4p, 8p)` relative to the input.

    Notes
    -----
    Reference: Ze Liu *et al.*, *"Swin Transformer: Hierarchical Vision
    Transformer using Shifted Windows"*, ICCV 2021,
    `arXiv:2103.14030 <https://arxiv.org/abs/2103.14030>`_.

    Examples
    --------
    Build a Swin-Tiny backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.swin import SwinTransformer, SwinConfig
    >>> model = SwinTransformer(SwinConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape                       # (B, 8 * embed_dim)
    (1, 768)
    >>> out = model(x)
    >>> out.last_hidden_state.shape      # (B, 1, 8 * embed_dim)
    (1, 1, 768)
    """

    config_class: ClassVar[type[SwinConfig]] = SwinConfig
    base_model_prefix: ClassVar[str] = "swin"

    def __init__(self, config: SwinConfig) -> None:
        super().__init__(config)
        pe, stages, norm, fi, _out_dim = _build_swin(config)
        self.patch_embed = pe
        self.stages = stages
        self.norm = norm
        self._feature_info = fi
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Reference initialisation: trunc_normal_(0.02) on Linear/patch-embed
        # convs with zero bias, LayerNorms at unit weight.  The framework
        # default (kaiming_uniform, a=sqrt(5)) is far wider and the paper's
        # LR schedule is not tuned for it.
        init_transformer_trunc_normal(self)

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)  # (B, H/p, W/p, C)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)  # (B, H', W', C)
        # Global average pool: permute to (B, C, H', W') → avgpool → flatten
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, C, H', W')
        x = self.avgpool(x).flatten(1)  # (B, C)
        return x

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# Swin Transformer for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class SwinTransformerForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""Swin Transformer with a linear classification head (Liu et al., 2021).

    Wraps the same hierarchical trunk as :class:`SwinTransformer` (patch
    embedding → four stages of window / shifted-window attention →
    LayerNorm) and adds a global average pool plus a single
    :class:`nn.Linear` classification head that maps the
    :math:`(B, 8C)` feature to ``config.num_classes`` logits:

    .. math::

        \text{logits} = W_{\text{head}}\,
            \mathrm{GAP}(\mathrm{LN}(z^{L})) + b_{\text{head}},
        \qquad W_{\text{head}} \in \mathbb{R}^{C_{\text{out}} \times 8C}.

    Pass ``labels`` to :meth:`forward` to compute the cross-entropy
    loss in the same pass.

    Parameters
    ----------
    config : SwinConfig
        Architecture specification.  Must set ``num_classes`` to the
        desired number of output categories (default 1000 for ImageNet).
        See :class:`SwinConfig`.

    Attributes
    ----------
    patch_embed : _PatchEmbed
        Strided patch-extraction convolution + LayerNorm.
    stages : nn.ModuleList
        Four hierarchical Swin stages with patch-merging downsamplers.
    norm : nn.LayerNorm
        Final LayerNorm applied to the channel-last feature map.
    avgpool : nn.AdaptiveAvgPool2d
        :math:`1 \times 1` adaptive average pool over spatial dims.
    classifier : nn.Linear
        Final linear projection of width ``(num_classes, 8 * embed_dim)``
        built via :meth:`ClassificationHeadMixin._build_classifier`.

    Notes
    -----
    Reference: Ze Liu *et al.*, *"Swin Transformer: Hierarchical Vision
    Transformer using Shifted Windows"*, ICCV 2021,
    `arXiv:2103.14030 <https://arxiv.org/abs/2103.14030>`_.  Swin-T /
    S / B / L reach **81.3 / 83.0 / 83.5 / 86.4 % top-1 on ImageNet-1k**
    (Table 1 of the paper, 224x224 input).

    Examples
    --------
    End-to-end inference with the default Swin-T classifier:

    >>> import lucid
    >>> from lucid.models.vision.swin import (
    ...     SwinConfig, SwinTransformerForImageClassification,
    ... )
    >>> model = SwinTransformerForImageClassification(SwinConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """

    config_class: ClassVar[type[SwinConfig]] = SwinConfig
    base_model_prefix: ClassVar[str] = "swin"

    def __init__(self, config: SwinConfig) -> None:
        super().__init__(config)
        pe, stages, norm, _, out_dim = _build_swin(config)
        self.patch_embed = pe
        self.stages = stages
        self.norm = norm
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._build_classifier(out_dim, config.num_classes, dropout=config.dropout)
        init_transformer_trunc_normal(self)

    @override
    def forward(
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x).flatten(1)
        logits = self.classifier(x)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
