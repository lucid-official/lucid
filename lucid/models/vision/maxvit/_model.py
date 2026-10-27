"""MaxViT backbone and classifier (Tu et al., 2022).

Paper: "MaxViT: Multi-Axis Vision Transformer"
        https://arxiv.org/abs/2204.01697

Key ideas:
  1. Multi-Axis attention: combine local window attention (block-attention) and
     global grid attention in every block — O(n) overall complexity.
  2. MBConv (mobile inverted bottleneck) in each block for local features.
  3. Window (block) attention: partition spatial into non-overlapping ws×ws
     windows, run MHA within each window.
  4. Grid attention: dilated / strided partition (every ws-th pixel in each
     direction forms a virtual "grid"), run MHA within each grid.

Architecture (MaxViT-T, image=224, ws=7):
  Stem   : Conv3×3(s=2, 3→64, bias) → BN → GELU → Conv3×3(s=1, 64→64, bias)
  Stage 0: 2 × MaxViTBlock(64, stride1=2, stride_rest=1) → (56×56)
  Stage 1: 2 × MaxViTBlock(128, stride1=2, stride_rest=1) → (28×28)
  Stage 2: 5 × MaxViTBlock(256, stride1=2, stride_rest=1) → (14×14)
  Stage 3: 2 × MaxViTBlock(512, stride1=2, stride_rest=1) → (7×7)
  Head   : LayerNorm → Pre-logits Linear → GELU (or Tanh) → FC

Key attribute naming mirrors timm's maxvit_tiny_tf_224 exactly so that
state-dict transfer works without any key remapping.

MBConv per block:
  conv.pre_norm (BN) → conv.conv1_1x1 → conv.norm1 (BN+GELU) →
  conv.conv2_kxk (DW, stride) → conv.norm2 (BN+GELU) →
  conv.se (SE: fc1/fc2 as Conv2d) →
  conv.conv3_1x1 (projection, no BN/act, with bias)

  Shortcut: when in_dim != out_dim, conv.shortcut.expand (1×1 Conv2d).

Attention blocks:
  attn_block.norm1 / .attn.qkv / .attn.rel_pos / .attn.proj / .norm2 / .mlp
  attn_grid.norm1  / .attn.qkv / .attn.rel_pos / .attn.proj / .norm2 / .mlp

  rel_pos holds a learnable table (num_heads, 2*ws-1, 2*ws-1) — RelPosBias.
  head_dim = 32 (fixed); num_heads = dim // 32.

Head:
  head.norm (LayerNorm) → head.pre_logits.fc (Linear) → Tanh →
  head.fc (Linear, num_classes)
"""

import math
from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._utils._common import init_transformer_trunc_normal
from lucid.models._utils._classification import DropPath
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.maxvit._config import MaxViTConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEAD_DIM: int = 32  # fixed head dimension (same as timm default)
_SE_RATIO: int = 16  # SE squeeze ratio relative to expanded dim

# ---------------------------------------------------------------------------
# Window partition / reverse
# ---------------------------------------------------------------------------


# The official MaxViT sets ``bn_momentum = 0.99`` in the TF sense — the
# fraction of the *old* running estimate kept — which is 0.01 in this
# convention.  The eps was already carried over as 1e-3; the momentum was
# left at the framework default of 0.1, ten times too fast.
_BN_MOMENTUM: float = 0.01


def _window_partition(x: Tensor, ws: int) -> tuple[Tensor, int, int]:
    """(B, H, W, C) → (B*nH*nW, ws, ws, C).

    Divides the spatial map into non-overlapping ws×ws windows.
    Requires H and W to be divisible by ws (caller must pad if needed).
    """
    B, H, W, C = x.shape
    nH, nW = H // ws, W // ws
    x = x.reshape(B, nH, ws, nW, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)
    return x, nH, nW


def _window_reverse(windows: Tensor, ws: int, nH: int, nW: int) -> Tensor:
    """(B*nH*nW, ws, ws, C) → (B, H, W, C)."""
    B_total = windows.shape[0]
    B = B_total // (nH * nW)
    C = windows.shape[-1]
    x = windows.reshape(B, nH, nW, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, nH * ws, nW * ws, C)


# ---------------------------------------------------------------------------
# Grid partition / reverse
# ---------------------------------------------------------------------------


def _grid_partition(x: Tensor, ws: int) -> tuple[Tensor, int, int]:
    """(B, H, W, C) → (B*nH*nW, ws, ws, C).

    Matches timm's grid_partition(x, grid_size=(ws, ws)).

    Each of the B*nH*nW groups contains the ws×ws pixels sampled at the
    same grid-stride offset.  Reshape:
      x.view(B, ws, nH, ws, nW, C) → permute(0,2,4,1,3,5)
      → view(-1, ws, ws, C)

    Requires H, W divisible by ws (caller must pad if needed).
    """
    B, H, W, C = x.shape
    nH, nW = H // ws, W // ws
    x = x.reshape(B, ws, nH, ws, nW, C)
    x = x.permute(0, 2, 4, 1, 3, 5)  # (B, nH, nW, ws, ws, C)
    x = x.reshape(B * nH * nW, ws, ws, C)
    return x, nH, nW


def _grid_reverse(x: Tensor, ws: int, nH: int, nW: int, B: int) -> Tensor:
    """(B*nH*nW, ws, ws, C) → (B, H, W, C)."""
    C = x.shape[-1]
    x = x.reshape(B, nH, nW, ws, ws, C)
    x = x.permute(0, 3, 1, 4, 2, 5)  # (B, ws, nH, ws, nW, C)
    return x.reshape(B, ws * nH, ws * nW, C)


# ---------------------------------------------------------------------------
# Helpers: pad spatial to multiple of ws, then crop back
# ---------------------------------------------------------------------------


def _tf_same_pad2d(x: Tensor, kernel_size: int, stride: int) -> Tensor:
    """Apply TF-style 'same' padding to a NCHW tensor.

    timm uses ``Conv2dSame`` for stride-2 convolutions in MaxViT.  TF pads
    right/bottom first (ceil division), giving asymmetric padding whenever
    the total pad is odd.  Standard ``padding=1`` in most frameworks pads
    symmetrically, which produces different border values for real inputs.
    """
    H, W = x.shape[2], x.shape[3]
    pad_h = max(int(math.ceil(H / stride) - 1) * stride + kernel_size - H, 0)
    pad_w = max(int(math.ceil(W / stride) - 1) * stride + kernel_size - W, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


def _pad_to_multiple(x_cl: Tensor, ws: int) -> tuple[Tensor, int, int]:
    """Pad (B, H, W, C) tensor so H and W are multiples of ws.

    Returns padded tensor and original (H, W) for later cropping.
    """
    B, H, W, C = x_cl.shape
    # The pad has to be born on the same device as the tensor it is
    # concatenated to.  Allocating on the default device meant any input whose
    # spatial size is not a multiple of the window (so any non-224 resolution,
    # which the config explicitly documents as supported) raised
    # ``DeviceMismatch (concatenate)`` on Metal.
    dev = x_cl.device.type
    pH = (ws - H % ws) % ws
    pW = (ws - W % ws) % ws
    if pH > 0:
        pad_h = lucid.zeros(B, pH, W, C, device=dev)
        x_cl = lucid.cat([x_cl, pad_h], dim=1)
    if pW > 0:
        Hp = x_cl.shape[1]
        pad_w = lucid.zeros(B, Hp, pW, C, device=dev)
        x_cl = lucid.cat([x_cl, pad_w], dim=2)
    return x_cl, H, W


# ---------------------------------------------------------------------------
# Relative-position bias (matches timm's RelPosBiasTf)
# ---------------------------------------------------------------------------


@final
class _RelPosBias(nn.Module):
    """Learnable relative position bias table: (num_heads, 2*ws-1, 2*ws-1).

    Matches timm's ``RelPosBiasTf`` — the table is indexed by (row_offset,
    col_offset) in [-ws+1, ws-1] range.  We store as ``relative_position_bias_table``
    so the key name matches timm exactly.

    At forward the bias is one gather with a pre-built flat index tensor.
    """

    def __init__(self, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.ws = window_size
        self.num_heads = num_heads
        table_size = 2 * window_size - 1
        self.relative_position_bias_table = nn.Parameter(
            lucid.zeros(num_heads, table_size, table_size)
        )
        # Both references draw this table from N(0, 0.02^2).  Left at exactly
        # zero every head's relative-position bias starts identical and the
        # term contributes nothing to break the symmetry between positions.
        nn.init.normal_(self.relative_position_bias_table, mean=0.0, std=0.02)
        # Build flat Python index lists for the gather at forward time.
        # coords: list of (row, col) for each of the ws² positions
        coords = [(i, j) for i in range(window_size) for j in range(window_size)]
        n = len(coords)
        # row_flat[k] and col_flat[k] are the table row/col index for
        # the k-th entry in the flattened (n*n,) gather.
        row_flat: list[int] = []
        col_flat: list[int] = []
        # Reference (timm RelPosBiasTf) indexes key-minus-query, so the
        # relative offset is cj - ci (not ci - cj).
        for ci in coords:
            for cj in coords:
                row_flat.append(cj[0] - ci[0] + (window_size - 1))
                col_flat.append(cj[1] - ci[1] + (window_size - 1))
        self._n = n
        # One flat index into the (T, T) table, so the gather is a single op
        # instead of ws^4 scalar reads followed by a ws^4-way stack.  The
        # index is constant for the module's lifetime; registering it
        # non-persistently keeps it off the state dict and lets ``.to()``
        # move it with the module.
        self.register_buffer(
            "_flat_index",
            lucid.tensor(
                [r * table_size + c for r, c in zip(row_flat, col_flat, strict=True)]
            ).long(),
            persistent=False,
        )

    @override
    def forward(self) -> Tensor:
        """Return bias tensor of shape (1, num_heads, ws², ws²)."""
        table = self.relative_position_bias_table  # (H, T, T)
        n = self._n
        H = self.num_heads
        T = int(table.shape[-1])
        # (H, T*T) → gather n*n columns → (H, ws², ws²)
        bias = table.reshape(H, T * T)[:, self._flat_index]
        return bias.reshape(H, n, n).unsqueeze(0)


# ---------------------------------------------------------------------------
# Scaled dot-product attention with relative position bias
# (channel-last, used by both block and grid attention)
# ---------------------------------------------------------------------------


@final
class _AttnCl(nn.Module):
    """Attention with relative position bias.

    Matches timm's ``AttentionCl`` structure:
      qkv: Linear(dim, 3*dim)
      rel_pos: RelPosBiasTf  (→ relative_position_bias_table)
      proj: Linear(dim, dim)

    Input/output: (*, N, C) channel-last format.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim) ** -1
        self.qkv = nn.Linear(dim, dim * 3)
        self.rel_pos = _RelPosBias(num_heads, window_size)
        self.proj = nn.Linear(dim, dim)

    @override
    def forward(self, x: Tensor, key_valid: Tensor | None = None) -> Tensor:
        # x: (B, N, C)  where B may be a compound batch
        B, N, C = x.shape
        H = self.num_heads
        hd = self.head_dim

        qkv = self.qkv(x)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, N, hd)

        # Fused SDPA with the relative-position bias as an additive mask.
        bias = self.rel_pos()  # (1, H, N, N)
        if key_valid is not None:
            # Non-divisible feature maps are zero-padded before partitioning,
            # and a zero row is not "absent" — it is a key with a real
            # embedding that every query then attends to.  Both references
            # refuse the non-divisible case outright; since padding is used
            # here as an extension, the pad columns must at least be driven
            # out of the softmax, which is what the official Attention's
            # ``attn_mask`` exists for.
            pad_bias = (1.0 - key_valid.reshape(-1, 1, 1, N)) * -1e4
            bias = bias + pad_bias
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# MLP block (matches timm's Mlp with fc1/fc2 named children)
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Two-layer MLP with GELU: fc1 → GELU → fc2.

    Matches timm Mlp attribute names so keys transfer directly.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Partition attention block (Pre-LN, matches timm's PartitionAttentionCl)
# ---------------------------------------------------------------------------


@final
class _PartitionAttn(nn.Module):
    """Pre-LN attention block matching timm's PartitionAttentionCl.

    Attribute layout:
      norm1 / attn / norm2 / mlp
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        window_size: int,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _AttnCl(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = _MLP(dim, hidden)
        # Stochastic depth on each residual branch separately, matching the
        # reference's drop_path1 / drop_path2.  Parameter-free, so the
        # state-dict layout this family mirrors is unaffected.
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)

    @override
    def forward(self, x: Tensor, key_valid: Tensor | None = None) -> Tensor:
        attended = self.attn(self.norm1(x), key_valid)
        x = x + self.drop_path1(attended)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# SE (Squeeze-and-Excitation) sub-module using Conv2d fc1/fc2
# Matches timm's SEModule with fc1/fc2 as Conv2d
# ---------------------------------------------------------------------------


class _SE(nn.Module):
    """Squeeze-and-Excitation with Conv2d fc1 and fc2 (matching timm SEModule).

    fc1: Conv2d(expanded_dim, se_mid, 1)
    fc2: Conv2d(se_mid, expanded_dim, 1)
    """

    def __init__(self, expanded_dim: int) -> None:
        super().__init__()
        se_mid = max(1, expanded_dim // _SE_RATIO)
        self.fc1 = nn.Conv2d(expanded_dim, se_mid, 1)
        self.fc2 = nn.Conv2d(se_mid, expanded_dim, 1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Input x — (B, C, H, W)
        s = F.adaptive_avg_pool2d(x, (1, 1))
        s = F.silu(self.fc1(s))
        s = F.sigmoid(self.fc2(s))
        return x * s


# ---------------------------------------------------------------------------
# Shortcut module (matches timm's Downsample2d for cross-stage projection)
# ---------------------------------------------------------------------------


@final
class _Shortcut(nn.Module):
    """Cross-stage shortcut: AvgPool (stride) + expand Conv2d.

    Matches timm's Downsample2d with pool + expand:
      - stride > 1: pool = AvgPool2d(stride)
      - in_dim != out_dim: expand = Conv2d(in_dim, out_dim, 1, bias=True)
      - in_dim == out_dim: expand = Identity (no state-dict keys)

    Stage 0 block 0 has stride=2, same dim → pool only (expand is Identity).
    Cross-stage blocks have stride=2, different dims → pool + Conv2d expand.
    """

    def __init__(self, in_dim: int, out_dim: int, stride: int) -> None:
        super().__init__()
        self._stride = stride
        if stride > 1:
            self.pool: nn.AvgPool2d | nn.Identity = nn.AvgPool2d(stride, stride=stride)
        else:
            self.pool = nn.Identity()
        if in_dim != out_dim:
            self.expand: nn.Conv2d | nn.Identity = nn.Conv2d(in_dim, out_dim, 1, bias=True)
        else:
            self.expand = nn.Identity()

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self._stride > 1:
            # timm's Downsample2d pools with TF-SAME geometry, matching the
            # main branch's Conv2dSame.  A plain floor-mode AvgPool loses the
            # trailing row/column on odd sizes, so the residual add hit
            # 12x12 + 13x13 for any input that is not a multiple of the window
            # grid — every resolution except the 224 the tests happen to use.
            x = _tf_same_pad2d(x, kernel_size=self._stride, stride=self._stride)
        x = self.pool(x)
        return self.expand(x)


# ---------------------------------------------------------------------------
# MBConv block (matches timm's MbConvBlock)
# ---------------------------------------------------------------------------


class _MBConv(nn.Module):
    """MBConv: BN pre-norm → expand Conv1×1 → BN+GELU →
    DWConv3×3(stride) → BN+GELU → SE → project Conv1×1 (bias).

    Shortcut: identity or Downsample2d (when dims differ / stride > 1).

    Attribute names mirror timm's MbConvBlock exactly:
      shortcut (optional, only when in_dim != out_dim),
      pre_norm, conv1_1x1, norm1, conv2_kxk, norm2, se, conv3_1x1
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int = 1,
        expand_ratio: int = 4,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        mid = out_dim * expand_ratio
        # Stochastic depth on the residual branch (reference: MbConvBlock ends
        # with ``x = self.drop_path(x) + shortcut``).
        self.drop_path = DropPath(drop_path_rate)

        # Shortcut: always create when in_dim != out_dim or stride > 1
        if in_dim != out_dim or stride > 1:
            self.shortcut = _Shortcut(in_dim, out_dim, stride)
        # If same dim and stride=1, no shortcut attribute (identity skip)

        self.pre_norm = nn.BatchNorm2d(in_dim, eps=1e-3, momentum=_BN_MOMENTUM)
        self.conv1_1x1 = nn.Conv2d(in_dim, mid, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid, eps=1e-3, momentum=_BN_MOMENTUM)
        # DW conv with stride; when stride=2 timm uses Conv2dSame (TF-same padding),
        # so we store padding=0 and apply _tf_same_pad2d manually in forward.
        self._conv2_stride = stride
        self.conv2_kxk = nn.Conv2d(
            mid,
            mid,
            3,
            stride=stride,
            padding=0 if stride > 1 else 1,
            groups=mid,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(mid, eps=1e-3, momentum=_BN_MOMENTUM)
        self.se = _SE(mid)
        self.conv3_1x1 = nn.Conv2d(mid, out_dim, 1, bias=True)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Shortcut path
        if hasattr(self, "shortcut"):
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        # Main path
        out = self.pre_norm(x)
        out = self.conv1_1x1(out)
        out = F.gelu(self.norm1(out), approximate="tanh")
        if self._conv2_stride > 1:
            out = _tf_same_pad2d(out, kernel_size=3, stride=self._conv2_stride)
        out = self.conv2_kxk(out)
        out = F.gelu(self.norm2(out), approximate="tanh")
        out = self.se(out)
        out = self.conv3_1x1(out)
        return shortcut + self.drop_path(out)


# ---------------------------------------------------------------------------
# MaxViT block: conv (MBConv) → attn_block (window) → attn_grid (grid)
# ---------------------------------------------------------------------------


@final
class _MaxViTBlock(nn.Module):
    """Single MaxViT block = MBConv + window attention + grid attention.

    Attribute naming matches timm's MaxxVitBlock:
      conv, attn_block, attn_grid

    The stride/in_dim/out_dim for conv come from the stage configuration.
    Window and grid attention convert to channel-last internally.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        stride: int = 1,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.ws = window_size
        # One block-level rate shared by the MBConv and both attention
        # branches, as the reference does.
        self.conv = _MBConv(in_dim, out_dim, stride=stride, drop_path_rate=drop_path_rate)
        self.attn_block = _PartitionAttn(
            out_dim, num_heads, mlp_ratio, window_size, drop_path_rate=drop_path_rate
        )
        self.attn_grid = _PartitionAttn(
            out_dim, num_heads, mlp_ratio, window_size, drop_path_rate=drop_path_rate
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Input x — (B, C_in, H, W)
        ws = self.ws

        # 1. MBConv (NCHW)
        x = self.conv(x)

        B, C, H, W = x.shape
        x_cl = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Pad to multiple of ws
        pre_H, pre_W = int(x_cl.shape[1]), int(x_cl.shape[2])
        x_cl, orig_H, orig_W = _pad_to_multiple(x_cl, ws)
        _Hp, _Wp = x_cl.shape[1], x_cl.shape[2]

        # A validity plane marking the real positions.  It rides through the
        # exact same partitioning as the features, so each attention window
        # learns which of its tokens are padding.  One plane per sample: the
        # partitions put the batch outermost, so a single plane gave one mask
        # row per window of the *first* image and failed to broadcast against
        # the B·windows attention batch for any B > 1.
        valid_cl: Tensor | None = None
        if int(_Hp) != pre_H or int(_Wp) != pre_W:
            ones = lucid.ones(B, pre_H, pre_W, 1, device=x_cl.device.type)
            valid_cl, _, _ = _pad_to_multiple(ones, ws)

        # 2. Block (window) attention — local within ws×ws windows
        wins, nH, nW = _window_partition(x_cl, ws)  # (B*nH*nW, ws, ws, C)
        wins_seq = wins.reshape(-1, ws * ws, C)
        win_valid: Tensor | None = None
        if valid_cl is not None:
            vw, _, _ = _window_partition(valid_cl, ws)
            win_valid = vw.reshape(-1, ws * ws)
        wins_seq = self.attn_block(wins_seq, win_valid)
        wins = wins_seq.reshape(-1, ws, ws, C)
        x_cl = _window_reverse(wins, ws, nH, nW)  # (B, Hp, Wp, C)

        # 3. Grid attention — (B*nH*nW, ws², C) like window attention
        grids, gH, gW = _grid_partition(x_cl, ws)  # (B*nH*nW, ws, ws, C)
        grids_seq = grids.reshape(-1, ws * ws, C)  # (B*nH*nW, ws², C)
        grid_valid: Tensor | None = None
        if valid_cl is not None:
            vg, _, _ = _grid_partition(valid_cl, ws)
            grid_valid = vg.reshape(-1, ws * ws)
        grids_seq = self.attn_grid(grids_seq, grid_valid)
        grids = grids_seq.reshape(-1, ws, ws, C)
        x_cl = _grid_reverse(grids, ws, gH, gW, B)  # (B, Hp, Wp, C)

        # Crop back to (post-stride) spatial size
        x_cl = x_cl[:, :orig_H, :orig_W, :]

        return x_cl.permute(0, 3, 1, 2)  # (B, C, H, W)


# ---------------------------------------------------------------------------
# Stage: nn.Sequential of blocks inside a 'blocks' attribute
# (matches timm's MaxxVitStage.blocks)
# ---------------------------------------------------------------------------


@final
class _MaxViTStage(nn.Module):
    """One MaxViT stage: a 'blocks' Sequential of _MaxViTBlock modules.

    Matches timm's MaxxVitStage which exposes a 'blocks' Sequential child.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dpr: list[float] | None = None,
    ) -> None:
        super().__init__()
        rates = dpr if dpr is not None else [0.0] * depth
        block_list: list[nn.Module] = []
        for i in range(depth):
            # First block does spatial downsampling (stride=2) and dim expansion
            blk_in = in_dim if i == 0 else out_dim
            blk_stride = 2 if i == 0 else 1
            block_list.append(
                _MaxViTBlock(
                    in_dim=blk_in,
                    out_dim=out_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    stride=blk_stride,
                    drop_path_rate=rates[i],
                )
            )
        self.blocks = nn.Sequential(*block_list)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Head sub-modules matching timm's NormMlpClassifierHead structure
# ---------------------------------------------------------------------------


@final
class _PreLogits(nn.Module):
    """Pre-logits: Linear → Tanh  (matches timm head.pre_logits.fc)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(self.fc(x))


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


class _Stem(nn.Module):
    """Two-conv stem matching timm's stem.conv1 / stem.norm1 / stem.conv2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # padding=0: TF-same is applied manually in forward before conv1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=0, bias=True)
        self.norm1 = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=_BN_MOMENTUM)
        # conv2 is stride=1 so symmetric padding=1 matches TF-same exactly
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=True)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = _tf_same_pad2d(x, kernel_size=3, stride=2)
        x = self.conv1(x)
        x = F.gelu(self.norm1(x), approximate="tanh")
        return self.conv2(x)


def _build_maxvit(
    cfg: MaxViTConfig,
) -> tuple[
    _Stem,
    nn.ModuleList[_MaxViTStage],
    list[FeatureInfo],
    int,
]:
    """Build shared MaxViT body: stem + stages.

    Stem: Conv3×3(s=2, bias) → BN → GELU → Conv3×3(s=1, bias)
    (attribute names: stem.conv1, stem.norm1, stem.conv2)

    Stages: stages.N = _MaxViTStage with internal 'blocks' Sequential
    """
    stem_out = cfg.stem_width if cfg.stem_width is not None else cfg.dims[0]

    stem = _Stem(cfg.in_channels, stem_out)

    stages_list: list[_MaxViTStage] = []
    fi: list[FeatureInfo] = []
    reduction = 4  # stem stride=2, then each stage block0 stride=2 → 4 total after stage 0

    prev_dim = stem_out
    # Stochastic depth ramps linearly with the *global* block index and is then
    # split per stage, so deeper blocks are dropped more often; a flat rate is
    # a different regulariser.
    total_blocks = sum(cfg.depths)
    dpr_all = [cfg.drop_path_rate * k / max(total_blocks - 1, 1) for k in range(total_blocks)]
    cursor = 0
    for i, (depth, dim) in enumerate(zip(cfg.depths, cfg.dims, strict=True)):
        num_heads = dim // cfg.head_dim
        stages_list.append(
            _MaxViTStage(
                in_dim=prev_dim,
                out_dim=dim,
                depth=depth,
                num_heads=num_heads,
                window_size=cfg.window_size,
                mlp_ratio=cfg.mlp_ratio,
                dpr=dpr_all[cursor : cursor + depth],
            )
        )
        cursor += depth
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))
        prev_dim = dim
        reduction *= 2

    return stem, nn.ModuleList(stages_list), fi, cfg.dims[-1]


# ---------------------------------------------------------------------------
# MaxViT backbone
# ---------------------------------------------------------------------------


class MaxViT(PretrainedModel, BackboneMixin):
    r"""MaxViT backbone (Tu et al., 2022).

    MaxViT combines a convolutional MBConv stage with two complementary
    sparse self-attention mechanisms — *block attention* and *grid
    attention* — to obtain dense local plus dense global receptive
    fields at linear complexity in the number of tokens.  The backbone
    is a four-stage pyramid preceded by a two-conv stem, and every
    MaxViT block applies (i) MobileNetV3-style MBConv, (ii) block
    attention, and (iii) grid attention in sequence.

    Given a feature map of shape :math:`(H, W, C)`, *block attention*
    partitions it into non-overlapping :math:`P \times P` windows
    (default :math:`P = 7`) and runs full self-attention inside each
    window — identical to Swin's W-MSA.  *Grid attention* takes the
    same feature map and gathers tokens that share the same intra-window
    position across all windows.  Formally, with :math:`H = h \cdot P`,
    :math:`W = w \cdot P`:

    .. math::

        \mathrm{Block}: (H, W) \to (h \cdot w,\ P \cdot P),\quad
        \mathrm{Grid} : (H, W) \to (P \cdot P,\ h \cdot w).

    :meth:`forward_features` returns the global-average-pooled
    :math:`(B, \texttt{dims[-1]})` feature.

    Parameters
    ----------
    config : MaxViTConfig
        Frozen dataclass specifying ``depths``, ``dims``,
        ``window_size``, ``num_heads``, ``mlp_ratio``, ``in_channels``,
        and ``num_classes``.  See :class:`MaxViTConfig`.

    Attributes
    ----------
    stem : nn.Module
        Two-layer convolutional stem with stride-2 first conv.
    stages : nn.ModuleList
        Four MaxViT stages, each containing a stack of MaxViT blocks.
    head : _HeadNorm
        Pre-classification head: LayerNorm → ``Linear + Tanh`` → final
        Linear.  Built only on the classifier — the backbone carries no head.
    feature_info : list[FeatureInfo]
        Four-stage feature description with reductions
        :math:`(4, 8, 16, 32)`.

    Notes
    -----
    Reference: Zhengzhong Tu *et al.*, *"MaxViT: Multi-Axis Vision
    Transformer"*, ECCV 2022,
    `arXiv:2204.01697 <https://arxiv.org/abs/2204.01697>`_.

    Examples
    --------
    Build a MaxViT-Tiny backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.maxvit import MaxViT, MaxViTConfig
    >>> model = MaxViT(MaxViTConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape                       # (B, dims[-1])
    (1, 512)
    """

    config_class: ClassVar[type[MaxViTConfig]] = MaxViTConfig
    base_model_prefix: ClassVar[str] = "maxvit"

    def __init__(self, config: MaxViTConfig) -> None:
        super().__init__(config)
        stem, stages, fi, _out_dim = _build_maxvit(config)
        self.stem = stem
        self.stages = stages
        self._feature_info = fi

        # Reference initialisation: trunc_normal_(0.02) on Linear / patch-embed
        # convs with zero bias, unit LayerNorms, and the same draw for the
        # class token and positional table (bare Parameters the framework
        # default never touches — they were sitting at exact zeros).
        init_transformer_trunc_normal(self)
        # No classifier here: this is the ``BackboneMixin`` half, and neither
        # ``forward_features`` nor ``forward`` ever called the head.  The
        # docstring justified it as "parameter sharing", but the classifier
        # class builds its own independent trunk *and* head — nothing was
        # shared, so a 1000-class projection was just dead weight on every
        # backbone instance.

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        # Global average pool → flatten
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# Head module (matches timm NormMlpClassifierHead)
# ---------------------------------------------------------------------------


@final
class _HeadNorm(nn.Module):
    """Classifier head: LayerNorm2d → flatten → pre_logits.fc → Tanh → fc.

    Attribute names match timm's NormMlpClassifierHead:
      head.norm, head.pre_logits.fc, head.fc
    """

    def __init__(self, in_dim: int, pre_logits_dim: int, num_classes: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.pre_logits = _PreLogits(in_dim, pre_logits_dim)
        self.fc = nn.Linear(pre_logits_dim, num_classes)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C) after global pool + flatten
        x = self.norm(x)
        x = self.pre_logits(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# MaxViT for image classification
# ---------------------------------------------------------------------------


class MaxViTForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""MaxViT with a linear classification head (Tu et al., 2022).

    Wraps the same multi-axis attention trunk as :class:`MaxViT`
    (stem + four MaxViT stages) and adds the reference-recipe head:
    global average pool → LayerNorm → ``Linear(dim) + Tanh`` → final
    linear classifier:

    .. math::

        \text{logits} = W_{\text{cls}}\,
            \mathrm{Tanh}\!\bigl(W_{\text{pre}}\,
            \mathrm{LN}(\mathrm{GAP}(z^{L}))\bigr) + b_{\text{cls}}.

    Pass ``labels`` to :meth:`forward` to compute the cross-entropy
    loss in the same pass.

    Parameters
    ----------
    config : MaxViTConfig
        Architecture specification.  Must set ``num_classes`` to the
        desired number of output categories.  See :class:`MaxViTConfig`.

    Attributes
    ----------
    stem : nn.Module
        Two-layer stride-2 convolutional stem.
    stages : nn.ModuleList
        Four MaxViT stages.
    head : _HeadNorm
        LayerNorm → pre_logits.fc (Linear) → Tanh → fc (Linear).

    Notes
    -----
    Reference: Zhengzhong Tu *et al.*, *"MaxViT: Multi-Axis Vision
    Transformer"*, ECCV 2022.  MaxViT-Tiny / Small / Base / Large /
    XLarge reach **83.6 / 84.5 / 84.9 / 85.2 / 85.5 % top-1 on
    ImageNet-1k** (Tu et al., 2022, Table 6).

    Examples
    --------
    End-to-end inference with the default MaxViT-Tiny classifier:

    >>> import lucid
    >>> from lucid.models.vision.maxvit import (
    ...     MaxViTConfig, MaxViTForImageClassification,
    ... )
    >>> model = MaxViTForImageClassification(MaxViTConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """

    config_class: ClassVar[type[MaxViTConfig]] = MaxViTConfig
    base_model_prefix: ClassVar[str] = "maxvit"

    def __init__(self, config: MaxViTConfig) -> None:
        super().__init__(config)
        stem, stages, _, out_dim = _build_maxvit(config)
        self.stem = stem
        self.stages = stages
        self.head = _HeadNorm(out_dim, out_dim, config.num_classes)
        init_transformer_trunc_normal(self)

    @override
    def reset_classifier(self, num_classes: int) -> None:
        """Replace the final projection of the norm-MLP head.

        The head is a ``_HeadNorm`` (matching timm's ``NormMlpClassifierHead``
        layout), so only its inner ``fc`` is class-dependent; the norm and
        pre-logits stages carry over.  The mixin's default looks for
        ``self.classifier`` and raised ``AttributeError`` here.
        """
        self.head.fc = nn.Linear(self.head.fc.in_features, num_classes)

    @override
    def forward(
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        # Global average pool → flatten → LayerNorm expects (B, C)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        logits = self.head(x)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
