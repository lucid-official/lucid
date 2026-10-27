r"""Mask2Former (Cheng et al., CVPR 2022).

Paper: "Masked-attention Mask Transformer for Universal Image Segmentation"
(arXiv:2112.01527).

Reference-faithful layout
-------------------------
The submodule names mirror the reference framework's Mask2Former verbatim so
the pretrained-weight converter is a near-identity key map after a ``model.``
prefix strip:

  * ``pixel_level_module.encoder`` — a Swin backbone (``swin.embeddings`` +
    ``swin.encoder.layers.{0..3}.blocks.{i}`` window-attention blocks +
    ``downsample`` patch merging) plus per-stage ``hidden_states_norms``.
  * ``pixel_level_module.decoder`` — the MSDeformAttn pixel decoder
    (``input_projections`` 1x1-conv+GroupNorm per level, ``level_embed``,
    a 6-layer deformable ``encoder``, ``mask_projection``, and the extra
    FPN ``adapter_{i}`` / ``layer_{i}`` lateral/output convs).
  * ``transformer_module`` — ``queries_embedder`` (query positional
    embedding), ``queries_features`` (learnable query init), ``level_embed``
    and a 9-layer masked-attention ``decoder`` (self-attn ``q/k/v/out_proj``,
    cross-attn packed ``nn.MultiheadAttention``, ``fc1`` / ``fc2``) holding
    the ``mask_predictor.mask_embedder`` MLP and a trailing ``layernorm``.
  * top-level ``class_predictor`` (Linear → K+1).

The masked-attention decoder cross-attends to one of the three multi-scale
memory levels (cycling ``level = layer_idx % 3``), restricting attention to
the previous layer's predicted foreground (``attn_mask = sigmoid(mask) <
0.5``, with the all-masked-row fallback).
"""

import math
from typing import ClassVar, cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._tasks import SemanticSegmentationModel
from lucid.models._output import SemanticSegmentationOutput
from lucid.models._utils._detection import (
    multi_scale_deformable_attention,
    solve_assignment,
)
from lucid.models._utils._detection_results import check_per_image
from lucid.models._utils._segmentation import (
    dice_loss,
    pairwise_dice,
    pairwise_sigmoid_ce,
    point_sample,
    sample_point_coords,
    sigmoid_ce_loss,
    uncertain_point_coords,
)
from lucid.models.vision._common._position import SinePositionEmbedding
from lucid.models.vision.mask2former._config import Mask2FormerConfig

# ---------------------------------------------------------------------------
# Swin backbone (reference SwinModel key layout)
# ---------------------------------------------------------------------------


@final
class _SwinPatchEmbeddings(nn.Module):
    """Patch embedding: a ``projection`` conv stride/kernel = patch_size."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        # Pad to a multiple of patch_size, as the reference does.  A
        # stride-``patch_size`` convolution silently *drops* the remainder
        # otherwise, so an image whose side is not a multiple loses its last
        # rows/columns before the backbone ever sees them.
        ph = (self.patch_size - int(x.shape[2]) % self.patch_size) % self.patch_size
        pw = (self.patch_size - int(x.shape[3]) % self.patch_size) % self.patch_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        emb: Tensor = self.projection(x)  # (B, C, H', W')
        h = int(emb.shape[2])
        w = int(emb.shape[3])
        # (B, C, H', W') -> (B, H'*W', C)
        emb = emb.reshape(int(emb.shape[0]), int(emb.shape[1]), h * w).permute(0, 2, 1)
        return emb, (h, w)


@final
class _SwinEmbeddings(nn.Module):
    """Patch embeddings + LayerNorm (``patch_embeddings`` + ``norm``)."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_embeddings = _SwinPatchEmbeddings(in_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        emb, dims = self.patch_embeddings(x)
        emb = self.norm(emb)
        return emb, dims


@final
class _SwinPatchMerging(nn.Module):
    """Patch merging downsample (``reduction`` + ``norm``)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    @override
    def forward(self, x: Tensor, dims: tuple[int, int]) -> Tensor:
        h, w = dims
        b = int(x.shape[0])
        c = int(x.shape[2])
        x = x.reshape(b, h, w, c)
        # The reference pads before merging when a side is odd; the strided
        # slices below otherwise disagree in length (0::2 and 1::2 differ by
        # one at odd sizes) and the concat raises ShapeMismatch.  A 384x384
        # input reaches an odd feature map by stage 3.
        if h % 2 or w % 2:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            h, w = h + h % 2, w + w % 2
        # Interleave 2x2 neighbourhoods: order is (row, col) for col in 0,1 / row in 0,1
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        merged: Tensor = lucid.cat([x0, x1, x2, x3], dim=-1)
        merged = merged.reshape(b, -1, 4 * c)
        merged = self.norm(merged)
        merged = self.reduction(merged)
        return merged


@final
class _SwinRelativePositionBias(nn.Module):
    """Window relative-position bias (``relative_position_bias_table``).

    ``relative_position_index`` is fully determined by ``window_size`` and is
    recreated here (a non-persistent buffer in the reference — never loaded).
    """

    relative_position_index: Tensor

    def __init__(self, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size * window_size
        self.num_heads = num_heads
        self.relative_position_bias_table = nn.Parameter(
            lucid.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        index = _build_relative_position_index(window_size)
        self.register_buffer(
            "relative_position_index",
            lucid.tensor(index, dtype=lucid.int64),
            persistent=False,
        )

    @override
    def forward(self) -> Tensor:
        idx: Tensor = self.relative_position_index
        table: Tensor = self.relative_position_bias_table
        bias: Tensor = table[idx]  # (window_area*window_area, num_heads)
        bias = bias.reshape(self.window_area, self.window_area, self.num_heads)
        # -> (1, num_heads, window_area, window_area)
        return bias.permute(2, 0, 1).unsqueeze(0)


def _build_relative_position_index(window_size: int) -> list[list[int]]:
    """Build the (window_area, window_area) relative-position lookup index."""
    coords: list[tuple[int, int]] = []
    for i in range(window_size):
        for j in range(window_size):
            coords.append((i, j))
    area = window_size * window_size
    index: list[list[int]] = []
    for a in range(area):
        row: list[int] = []
        ia, ja = coords[a]
        for b in range(area):
            ib, jb = coords[b]
            dh = ia - ib + window_size - 1
            dw = ja - jb + window_size - 1
            row.append(dh * (2 * window_size - 1) + dw)
        index.append(row)
    return index


@final
class _SwinAttention(nn.Module):
    """Window multi-head self-attention with separate ``q/k/v/o_proj``."""

    def __init__(self, dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.relative_position_bias = _SwinRelativePositionBias(num_heads, window_size)

    @override
    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        # Input x — (num_windows*B, ws*ws, C)
        nw = int(x.shape[0])
        seq = int(x.shape[1])
        q: Tensor = (
            self.q_proj(x).reshape(nw, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        )
        k: Tensor = (
            self.k_proj(x).reshape(nw, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        )
        v: Tensor = (
            self.v_proj(x).reshape(nw, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        )

        attn: Tensor = lucid.matmul(q * self.scaling, k.permute(0, 1, 3, 2))
        bias: Tensor = self.relative_position_bias()  # (1, H, seq, seq)
        attn = attn + bias
        if attn_mask is not None:
            # attn_mask: (num_windows, seq, seq) -> broadcast over batch & heads
            num_windows = int(attn_mask.shape[0])
            b = nw // num_windows
            am: Tensor = attn_mask.unsqueeze(1).unsqueeze(0)
            am = am.reshape(1, num_windows, 1, seq, seq)
            attn = attn.reshape(b, num_windows, self.num_heads, seq, seq) + am
            attn = attn.reshape(nw, self.num_heads, seq, seq)

        attn = F.softmax(attn, dim=-1)
        out: Tensor = lucid.matmul(attn, v)  # (nw, H, seq, hd)
        out = out.permute(0, 2, 1, 3).reshape(nw, seq, self.num_heads * self.head_dim)
        return self.o_proj(out)


@final
class _SwinMLP(nn.Module):
    """Swin block MLP (``fc1`` -> GELU -> ``fc2``)."""

    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def _window_partition(x: Tensor, window_size: int) -> Tensor:
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)."""
    b = int(x.shape[0])
    h = int(x.shape[1])
    w = int(x.shape[2])
    c = int(x.shape[3])
    x = x.reshape(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, c)
    return windows


def _window_reverse(windows: Tensor, window_size: int, h: int, w: int) -> Tensor:
    """(num_windows*B, window_size, window_size, C) -> (B, H, W, C)."""
    c = int(windows.shape[-1])
    x = windows.reshape(-1, h // window_size, w // window_size, window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, h, w, c)
    return x


def _build_swin_attn_mask(
    height: int, width: int, window_size: int, shift_size: int, like: Tensor
) -> Tensor:
    """Cyclic-shift attention mask for shifted-window MSA."""
    img_mask: list[list[float]] = [[0.0] * width for _ in range(height)]
    h_slices = [
        (0, height - window_size),
        (height - window_size, height - shift_size),
        (height - shift_size, height),
    ]
    w_slices = [
        (0, width - window_size),
        (width - window_size, width - shift_size),
        (width - shift_size, width),
    ]
    count = 0
    for hs, he in h_slices:
        for ws, we in w_slices:
            for i in range(hs, he):
                for j in range(ws, we):
                    img_mask[i][j] = float(count)
            count += 1
    mask_t: Tensor = lucid.tensor(img_mask, dtype=like.dtype, device=like.device).reshape(
        1, height, width, 1
    )
    mask_windows = _window_partition(mask_t, window_size).reshape(-1, window_size * window_size)
    # (nW, 1, seq) - (nW, seq, 1) -> (nW, seq, seq)
    diff = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    nonzero = (diff != 0.0).to(like.dtype)
    attn_mask: Tensor = nonzero * (-100.0)
    return attn_mask


@final
class _SwinLayer(nn.Module):
    """One Swin block: (S)W-MSA + MLP with pre-LayerNorms."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.attention = _SwinAttention(dim, num_heads, window_size)
        self.layernorm_before = nn.LayerNorm(dim)
        self.layernorm_after = nn.LayerNorm(dim)
        self.mlp = _SwinMLP(dim, mlp_ratio)

    @override
    def forward(self, x: Tensor, dims: tuple[int, int]) -> Tensor:
        h, w = dims
        b = int(x.shape[0])
        c = int(x.shape[2])
        shortcut = x

        # Mask2Former drives the Swin as a *backbone*, which runs every stage
        # with ``always_partition=True``.  That flag makes the reference skip
        # ``set_shift_and_window_size`` entirely, so the window size and the
        # cyclic shift are *never* clamped to the feature resolution — the
        # shifted-window attention is preserved even when the feature map
        # equals (or is smaller than) the window (e.g. base/large with
        # window 12 at the 12x12 last stage).  Use the configured ws/ss as-is.
        ws = self.window_size
        ss = self.shift_size

        x = self.layernorm_before(x)
        x = x.reshape(b, h, w, c)

        # Pad to window-size multiples
        pad_r = (ws - w % ws) % ws
        pad_b = (ws - h % ws) % ws
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        hp = h + pad_b
        wp = w + pad_r

        # Cyclic shift
        if ss > 0:
            x = lucid.roll(x, [-ss, -ss], dims=[1, 2])

        x_windows = _window_partition(x, ws).reshape(-1, ws * ws, c)
        attn_mask: Tensor | None = None
        if ss > 0:
            attn_mask = _build_swin_attn_mask(hp, wp, ws, ss, x)
        attn_out = self.attention(x_windows, attn_mask)

        attn_windows = attn_out.reshape(-1, ws, ws, c)
        x = _window_reverse(attn_windows, ws, hp, wp)

        # Reverse cyclic shift
        if ss > 0:
            x = lucid.roll(x, [ss, ss], dims=[1, 2])

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :]
        x = x.reshape(b, h * w, c)
        x = shortcut + x

        residual = x
        x = self.layernorm_after(x)
        x = self.mlp(x)
        x = x + residual
        return x


class _SwinStage(nn.Module):
    """A Swin stage: ``blocks`` + optional ``downsample``."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        downsample: bool,
    ) -> None:
        super().__init__()
        blocks: list[_SwinLayer] = []
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            blocks.append(_SwinLayer(dim, num_heads, window_size, shift, mlp_ratio))
        self.blocks: nn.ModuleList[_SwinLayer] = nn.ModuleList(blocks)
        self.downsample: _SwinPatchMerging | None = _SwinPatchMerging(dim) if downsample else None

    @override
    def forward(self, x: Tensor, dims: tuple[int, int]) -> tuple[Tensor, Tensor, tuple[int, int]]:
        h, w = dims
        for block in self.blocks:
            blk = block
            x = blk(x, dims)
        before = x
        if self.downsample is not None:
            ds = self.downsample
            x = ds(before, dims)
            new_dims = ((h + 1) // 2, (w + 1) // 2)
        else:
            new_dims = dims
        return x, before, new_dims


@final
class _SwinEncoder(nn.Module):
    """Swin encoder: 4 ``layers`` (stages)."""

    def __init__(
        self,
        embed_dim: int,
        depths: tuple[int, int, int, int],
        num_heads: tuple[int, int, int, int],
        window_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        stages: list[_SwinStage] = []
        dim = embed_dim
        num_layers = len(depths)
        for i in range(num_layers):
            stages.append(
                _SwinStage(
                    dim=dim,
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    downsample=(i < num_layers - 1),
                )
            )
            if i < num_layers - 1:
                dim *= 2
        self.layers: nn.ModuleList[_SwinStage] = nn.ModuleList(stages)

    @override
    def forward(self, x: Tensor, dims: tuple[int, int]) -> list[tuple[Tensor, tuple[int, int]]]:
        """Return per-stage (before-downsample hidden state, spatial dims)."""
        out: list[tuple[Tensor, tuple[int, int]]] = []
        for stage in self.layers:
            stg = stage
            x, before, new_dims = stg(x, dims)
            out.append((before, dims))
            dims = new_dims
        return out


@final
class _SwinModel(nn.Module):
    """Reference ``SwinModel`` (``embeddings`` + ``encoder``); no pooler."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depths: tuple[int, int, int, int],
        num_heads: tuple[int, int, int, int],
        window_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.embeddings = _SwinEmbeddings(in_channels, embed_dim)
        self.encoder = _SwinEncoder(embed_dim, depths, num_heads, window_size, mlp_ratio)

    @override
    def forward(self, x: Tensor) -> list[tuple[Tensor, tuple[int, int]]]:
        emb, dims = self.embeddings(x)
        return self.encoder(emb, dims)


@final
class _SwinBackbone(nn.Module):
    """Reference ``SwinBackbone``: ``swin`` model + per-stage ``hidden_states_norms``.

    Emits ``[stage1..stage4]`` feature maps in ``(B, C, H, W)`` layout after
    a stage-specific LayerNorm applied to the pre-downsampling hidden states.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depths: tuple[int, int, int, int],
        num_heads: tuple[int, int, int, int],
        window_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.swin = _SwinModel(in_channels, embed_dim, depths, num_heads, window_size, mlp_ratio)
        channels = [int(embed_dim * 2**i) for i in range(len(depths))]
        self.out_channels: list[int] = channels
        norms: dict[str, nn.LayerNorm] = {}
        for i, ch in enumerate(channels):
            norms[f"stage{i + 1}"] = nn.LayerNorm(ch)
        self.hidden_states_norms: nn.ModuleDict[nn.LayerNorm] = nn.ModuleDict(norms)

    @override
    def forward(self, x: Tensor) -> list[Tensor]:
        stage_outputs = self.swin(x)
        feature_maps: list[Tensor] = []
        for i, (hidden, dims) in enumerate(stage_outputs):
            h, w = dims
            b = int(hidden.shape[0])
            c = int(hidden.shape[2])
            norm = self.hidden_states_norms[f"stage{i + 1}"]
            normed: Tensor = norm(hidden)  # (B, H*W, C)
            spatial: Tensor = normed.reshape(b, h, w, c).permute(0, 3, 1, 2)
            feature_maps.append(spatial)
        return feature_maps


# ---------------------------------------------------------------------------
# MSDeformAttn pixel decoder (reference Mask2FormerPixelDecoder)
# ---------------------------------------------------------------------------


@final
class _DeformableAttention(nn.Module):
    """Multi-scale deformable attention (reference layer)."""

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        r"""The MSDeformAttn-specific reset both references apply.

        The sampling offsets start at zero *weight* with a **bias laid out
        as a ring**: head :math:`h` points at angle :math:`2\pi h / H`, and
        the ``p``-th sampling point along each ray sits at radius
        :math:`p + 1`.  That gives every head a distinct initial direction
        and every point a distinct distance, so the deformable sampler
        starts as a spread-out grid rather than collapsing all heads onto
        the same location.  Attention weights start at exactly zero (a
        uniform softmax), and the two projections are Xavier-uniform with
        zero bias.
        """
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = [2.0 * math.pi * h / self.num_heads for h in range(self.num_heads)]
        grid = [(math.cos(t), math.sin(t)) for t in thetas]
        scale = max(max(abs(x), abs(y)) for x, y in grid)
        bias: list[float] = []
        for gx, gy in grid:
            for _lvl in range(self.n_levels):
                for pt in range(self.n_points):
                    bias.append(gx / scale * (pt + 1))
                    bias.append(gy / scale * (pt + 1))
        if self.sampling_offsets.bias is not None:
            with lucid.no_grad():
                self.sampling_offsets.bias[:] = lucid.tensor(
                    bias, device=self.sampling_offsets.bias.device.type
                )

        nn.init.zeros_(self.attention_weights.weight)
        if self.attention_weights.bias is not None:
            nn.init.zeros_(self.attention_weights.bias)

        for proj in (self.value_proj, self.output_proj):
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    @override
    def forward(
        self,
        hidden_states: Tensor,  # (B, S, C) + pos
        encoder_hidden_states: Tensor,  # (B, S, C)
        position_embeddings: Tensor,  # (B, S, C)
        reference_points: Tensor,  # (B, S, n_levels, 2)
        spatial_shapes: list[tuple[int, int]],
    ) -> Tensor:
        hs: Tensor = hidden_states + position_embeddings
        b = int(hs.shape[0])
        nq = int(hs.shape[1])
        s = int(encoder_hidden_states.shape[1])
        head_dim = self.embed_dim // self.num_heads

        value: Tensor = self.value_proj(encoder_hidden_states)
        value = value.reshape(b, s, self.num_heads, head_dim)

        offsets: Tensor = self.sampling_offsets(hs).reshape(
            b, nq, self.num_heads, self.n_levels, self.n_points, 2
        )
        weights: Tensor = self.attention_weights(hs).reshape(
            b, nq, self.num_heads, self.n_levels * self.n_points
        )
        weights = F.softmax(weights, dim=-1).reshape(
            b, nq, self.num_heads, self.n_levels, self.n_points
        )

        # offset_normalizer: [[W, H], ...] per level
        normalizer: list[list[float]] = [[float(w), float(h)] for (h, w) in spatial_shapes]
        norm_t: Tensor = lucid.tensor(normalizer, dtype=hs.dtype, device=hs.device)  # (n_levels, 2)
        # sampling_locations — ref[:, :, None, :, None, :] + offsets / norm[None,None,None,:,None,:]
        ref_exp: Tensor = reference_points[:, :, None, :, None, :]
        norm_exp: Tensor = norm_t[None, None, None, :, None, :]
        sampling_locations: Tensor = ref_exp + offsets / norm_exp

        out: Tensor = multi_scale_deformable_attention(
            value, spatial_shapes, sampling_locations, weights
        )
        return self.output_proj(out)


@final
class _PixelDecoderEncoderLayer(nn.Module):
    """One deformable-encoder layer (self-attn + FFN, post-norm)."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.self_attn = _DeformableAttention(embed_dim, num_heads, 3, 4)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    @override
    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tensor,
        reference_points: Tensor,
        spatial_shapes: list[tuple[int, int]],
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states,
            hidden_states,
            position_embeddings,
            reference_points,
            spatial_shapes,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = F.relu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


@final
class _PixelDecoderEncoder(nn.Module):
    """Stack of deformable-encoder layers (``layers``)."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, depth: int) -> None:
        super().__init__()
        self.layers: nn.ModuleList[_PixelDecoderEncoderLayer] = nn.ModuleList(
            [_PixelDecoderEncoderLayer(embed_dim, num_heads, ffn_dim) for _ in range(depth)]
        )

    @staticmethod
    def get_reference_points(spatial_shapes: list[tuple[int, int]], like: Tensor) -> Tensor:
        """Per-level reference points (valid_ratios are all 1 at inference)."""
        ref_list: list[Tensor] = []
        n_levels = len(spatial_shapes)
        for h, w in spatial_shapes:
            ref_y, ref_x = lucid.meshgrid(
                lucid.linspace(0.5, h - 0.5, h, dtype=like.dtype, device=like.device),
                lucid.linspace(0.5, w - 0.5, w, dtype=like.dtype, device=like.device),
                indexing="ij",
            )
            ry: Tensor = ref_y.reshape(-1)[None] / float(h)
            rx: Tensor = ref_x.reshape(-1)[None] / float(w)
            ref: Tensor = lucid.stack([rx, ry], dim=-1)  # (1, h*w, 2)
            ref_list.append(ref)
        reference_points: Tensor = lucid.cat(ref_list, dim=1)  # (1, S, 2)
        # (1, S, 1, 2) -> (1, S, n_levels, 2)
        reference_points = reference_points[:, :, None].repeat(1, 1, n_levels, 1)
        return reference_points

    @override
    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tensor,
        spatial_shapes: list[tuple[int, int]],
    ) -> Tensor:
        reference_points = self.get_reference_points(spatial_shapes, hidden_states)
        b = int(hidden_states.shape[0])
        if b > 1:
            reference_points = reference_points.repeat(b, 1, 1, 1)
        for layer in self.layers:
            enc_layer = layer
            hidden_states = enc_layer(
                hidden_states,
                position_embeddings,
                reference_points,
                spatial_shapes,
            )
        return hidden_states


@final
class _PixelDecoder(nn.Module):
    """Reference ``Mask2FormerPixelDecoder`` (MSDeformAttn pixel decoder)."""

    def __init__(
        self,
        feature_channels: list[int],  # [stage1..stage4]
        feature_dim: int,
        mask_dim: int,
        num_heads: int,
        ffn_dim: int,
        encoder_depth: int,
        feature_strides: tuple[int, int, int, int],
        common_stride: int,
    ) -> None:
        super().__init__()
        self.num_feature_levels = 3
        self.position_embedding = SinePositionEmbedding(num_position_features=feature_dim // 2)
        self.level_embed = nn.Parameter(lucid.zeros(self.num_feature_levels, feature_dim))

        # Input projections: highest-stride first (reversed last 3 channels)
        transformer_in_channels = feature_channels[-self.num_feature_levels :]
        projections: list[nn.Sequential] = []
        for in_ch in transformer_in_channels[::-1]:
            projections.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, feature_dim, kernel_size=1),
                    nn.GroupNorm(32, feature_dim),
                )
            )
        self.input_projections = nn.ModuleList(projections)

        self.encoder = _PixelDecoderEncoder(feature_dim, num_heads, ffn_dim, encoder_depth)
        self.mask_projection = nn.Conv2d(feature_dim, mask_dim, kernel_size=1)

        # Extra FPN levels
        transformer_strides = feature_strides[-self.num_feature_levels :]
        stride = min(transformer_strides)
        self.num_fpn_levels = int(math.log2(stride) - math.log2(common_stride))
        self.feature_channels = feature_channels

        lateral_convs: list[nn.Sequential] = []
        output_convs: list[nn.Sequential] = []
        for idx in range(self.num_fpn_levels):
            in_ch = feature_channels[idx]
            lateral = nn.Sequential(
                nn.Conv2d(in_ch, feature_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, feature_dim),
            )
            output = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, feature_dim),
                nn.ReLU(),
            )
            self.add_module(f"adapter_{idx + 1}", lateral)
            self.add_module(f"layer_{idx + 1}", output)
            lateral_convs.append(lateral)
            output_convs.append(output)
        # Ordered low → high resolution
        self._lateral_convolutions = lateral_convs[::-1]
        self._output_convolutions = output_convs[::-1]

    @override
    def forward(self, features: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
        b = int(features[0].shape[0])

        # Reverse last 3 levels (highest stride first), project + pos embed
        inputs_embeds: list[Tensor] = []
        position_embeddings: list[Tensor] = []
        selected = features[::-1][: self.num_feature_levels]
        for level, x in enumerate(selected):
            proj = self.input_projections[level]
            inputs_embeds.append(proj(x))
            h = int(x.shape[2])
            w = int(x.shape[3])
            position_embeddings.append(self.position_embedding(x))

        spatial_shapes: list[tuple[int, int]] = [
            (int(e.shape[2]), int(e.shape[3])) for e in inputs_embeds
        ]

        # Flatten: (B, C, H, W) -> (B, H*W, C)
        flat_embeds: list[Tensor] = []
        flat_pos: list[Tensor] = []
        level_embed_w = self.level_embed
        for i, e in enumerate(inputs_embeds):
            c = int(e.shape[1])
            hw = int(e.shape[2]) * int(e.shape[3])
            flat_embeds.append(e.reshape(b, c, hw).permute(0, 2, 1))
            p = position_embeddings[i]
            p_flat = p.reshape(b, c, hw).permute(0, 2, 1)
            # add level embed
            lvl = level_embed_w[i].reshape(1, 1, c)
            flat_pos.append(p_flat + lvl)

        input_embeds_flat: Tensor = lucid.cat(flat_embeds, dim=1)
        level_pos_flat: Tensor = lucid.cat(flat_pos, dim=1)

        encoded: Tensor = self.encoder(input_embeds_flat, level_pos_flat, spatial_shapes)

        # Split back to per-level spatial maps
        outputs: list[Tensor] = []
        offset = 0
        for h, w in spatial_shapes:
            chunk = encoded[:, offset : offset + h * w, :]
            offset += h * w
            c = int(chunk.shape[2])
            outputs.append(chunk.permute(0, 2, 1).reshape(b, c, h, w))

        # Extra FPN levels (low → high resolution)
        fpn_features = features[: self.num_fpn_levels][::-1]
        for idx, feature in enumerate(fpn_features):
            lateral = self._lateral_convolutions[idx]
            output_conv = self._output_convolutions[idx]
            current_fpn: Tensor = lateral(feature)
            up = F.interpolate(
                outputs[-1],
                size=(int(current_fpn.shape[2]), int(current_fpn.shape[3])),
                mode="bilinear",
                align_corners=False,
            )
            out = current_fpn + up
            outputs.append(output_conv(out))

        multi_scale_features = outputs[: self.num_feature_levels]
        mask_features: Tensor = self.mask_projection(outputs[-1])
        return mask_features, multi_scale_features


class _PixelLevelModule(nn.Module):
    """Reference ``Mask2FormerPixelLevelModule``: Swin ``encoder`` + ``decoder``."""

    def __init__(self, config: Mask2FormerConfig) -> None:
        super().__init__()
        self.encoder = _SwinBackbone(
            in_channels=config.in_channels,
            embed_dim=config.swin_embed_dim,
            depths=config.swin_depths,
            num_heads=config.swin_num_heads,
            window_size=config.swin_window_size,
            mlp_ratio=config.swin_mlp_ratio,
        )
        self.decoder = _PixelDecoder(
            feature_channels=self.encoder.out_channels,
            feature_dim=config.d_model,
            mask_dim=config.mask_feature_size,
            num_heads=config.n_head,
            ffn_dim=config.encoder_feedforward_dim,
            encoder_depth=config.num_encoder_layers,
            feature_strides=config.feature_strides,
            common_stride=config.common_stride,
        )

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        features = self.encoder(x)
        return self.decoder(features)


# ---------------------------------------------------------------------------
# Masked-attention transformer decoder (reference Mask2FormerMaskedAttentionDecoder)
# ---------------------------------------------------------------------------


@final
class _DecoderSelfAttention(nn.Module):
    """Decoder self-attention with separate ``q/k/v/out_proj`` (DETR-style)."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _shape(self, x: Tensor, seq: int, b: int) -> Tensor:
        return x.reshape(b, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    @override
    def forward(self, hidden_states: Tensor, position_embeddings: Tensor) -> Tensor:
        # hidden_states: (N, B, C) -> operate as (B, N, C)
        hs: Tensor = hidden_states.permute(1, 0, 2)
        pos: Tensor = position_embeddings.permute(1, 0, 2)
        b = int(hs.shape[0])
        n = int(hs.shape[1])

        hs_pos: Tensor = hs + pos
        q: Tensor = self.q_proj(hs_pos) * self.scaling
        k: Tensor = self.k_proj(hs_pos)
        v: Tensor = self.v_proj(hs)

        qh = self._shape(q, n, b).reshape(b * self.num_heads, n, self.head_dim)
        kh = self._shape(k, n, b).reshape(b * self.num_heads, n, self.head_dim)
        vh = self._shape(v, n, b).reshape(b * self.num_heads, n, self.head_dim)

        attn = lucid.matmul(qh, kh.permute(0, 2, 1))
        attn = F.softmax(attn, dim=-1)
        out = lucid.matmul(attn, vh)  # (b*h, n, hd)
        out = out.reshape(b, self.num_heads, n, self.head_dim).permute(0, 2, 1, 3)
        out = out.reshape(b, n, self.embed_dim)
        out = self.out_proj(out)
        return out.permute(1, 0, 2)  # back to (N, B, C)


@final
class _MaskedAttentionDecoderLayer(nn.Module):
    """Reference ``Mask2FormerMaskedAttentionDecoderLayer`` (post-norm)."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.self_attn = _DecoderSelfAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    @override
    def forward(
        self,
        hidden_states: Tensor,  # (N, B, C)
        level_memory: Tensor,  # (S, B, C)
        level_pos: Tensor,  # (S, B, C)
        query_pos: Tensor,  # (N, B, C)
        attn_mask: Tensor | None,  # (B*H, N, S) bool
    ) -> Tensor:
        # Masked cross-attention
        residual = hidden_states
        query = hidden_states + query_pos
        key = level_memory + level_pos
        cross_out, _ = self.cross_attn(
            query, key, level_memory, attn_mask=attn_mask, need_weights=False
        )
        hidden_states = residual + cross_out
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # Self-attention
        residual = hidden_states
        sa = self.self_attn(hidden_states, query_pos)
        hidden_states = residual + sa
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # FFN
        residual = hidden_states
        ff = F.relu(self.fc1(hidden_states))
        ff = self.fc2(ff)
        hidden_states = residual + ff
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


@final
class _MLPPredictionHead(nn.Sequential):
    """Reference ``Mask2FormerMLPPredictionHead`` (3-layer MLP).

    Subclasses :class:`nn.Sequential` so the per-layer keys are ``{0,1,2}``
    and each layer is itself a ``nn.Sequential`` of ``Linear`` (``.0``) +
    activation (``.1``, no parameters), giving the reference
    ``mask_embedder.{0,1,2}.0.{weight,bias}`` key layout.  Layers 0-1 use
    ReLU; the final layer is identity.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3
    ) -> None:
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        blocks: list[nn.Module] = []
        for i in range(num_layers):
            act: nn.Module = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            blocks.append(nn.Sequential(nn.Linear(in_dims[i], out_dims[i]), act))
        super().__init__(*blocks)


@final
class _MaskPredictor(nn.Module):
    """Reference ``Mask2FormerMaskPredictor``: ``mask_embedder`` MLP.

    Produces a per-query mask and the binarised attention mask for the next
    decoder layer's masked cross-attention.
    """

    def __init__(self, hidden_size: int, num_heads: int, mask_feature_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.mask_embedder = _MLPPredictionHead(hidden_size, hidden_size, mask_feature_size)

    @override
    def forward(
        self,
        outputs: Tensor,  # (N, B, C)
        pixel_embeddings: Tensor,  # (B, C, H, W)
        target_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        # mask_embeddings — (B, N, C)
        mask_embeddings: Tensor = self.mask_embedder(outputs.permute(1, 0, 2))
        b = int(pixel_embeddings.shape[0])
        c = int(pixel_embeddings.shape[1])
        ph = int(pixel_embeddings.shape[2])
        pw = int(pixel_embeddings.shape[3])
        n = int(mask_embeddings.shape[1])
        pix_flat = pixel_embeddings.reshape(b, c, ph * pw)
        outputs_mask: Tensor = lucid.matmul(mask_embeddings, pix_flat).reshape(b, n, ph, pw)

        # attention mask at target size
        th, tw = target_size
        attn: Tensor = F.interpolate(
            outputs_mask, size=(th, tw), mode="bilinear", align_corners=False
        )  # (B, N, th, tw)
        attn_sig: Tensor = F.sigmoid(attn).reshape(b, n, th * tw)
        # (B, N, S) -> (B, 1, N, S) -> repeat heads -> (B*H, N, S)
        attn_rep: Tensor = attn_sig.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attn_mask: Tensor = (attn_rep < 0.5).reshape(b * self.num_heads, n, th * tw)
        return outputs_mask, attn_mask


@final
class _MaskedAttentionDecoder(nn.Module):
    """Reference ``Mask2FormerMaskedAttentionDecoder``.

    Holds ``num_decoder_layers - 1`` masked layers, a trailing ``layernorm``
    and a ``mask_predictor``.  Each layer cross-attends to one cycling memory
    level; mask predictions feed the next layer's attention mask.
    """

    def __init__(self, config: Mask2FormerConfig) -> None:
        super().__init__()
        self.num_feature_levels = config.num_feature_levels
        self.decoder_layers = config.num_decoder_layers - 1
        self.layers = nn.ModuleList(
            [
                _MaskedAttentionDecoderLayer(config.d_model, config.n_head, config.dim_feedforward)
                for _ in range(self.decoder_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(config.d_model)
        self.mask_predictor = _MaskPredictor(
            config.d_model, config.n_head, config.mask_feature_size
        )

    @override
    def forward(
        self,
        inputs_embeds: Tensor,  # (N, B, C) query features
        multi_stage_positional_embeddings: list[Tensor],  # per-level (S, B, C)
        pixel_embeddings: Tensor,  # (B, C, H, W) mask features
        encoder_hidden_states: list[Tensor],  # per-level (S, B, C)
        query_position_embeddings: Tensor,  # (N, B, C)
        feature_size_list: list[tuple[int, int]],
    ) -> tuple[list[Tensor], list[Tensor]]:
        hidden_states = inputs_embeds
        intermediate: list[Tensor] = []
        intermediate_mask_predictions: list[Tensor] = []

        intermediate_hidden_states: Tensor = self.layernorm(inputs_embeds)
        intermediate.append(intermediate_hidden_states)

        predicted_mask, attention_mask = self.mask_predictor(
            intermediate_hidden_states, pixel_embeddings, feature_size_list[0]
        )
        intermediate_mask_predictions.append(predicted_mask)

        for idx in range(self.decoder_layers):
            level_index = idx % self.num_feature_levels

            # all-masked-row fallback: rows attending to nothing → attend all
            seq = int(attention_mask.shape[-1])
            row_sum = attention_mask.float().sum(dim=-1)
            where = (row_sum != float(seq)).float()
            attention_mask = attention_mask.float() * where.unsqueeze(-1)
            attn_mask_bool = attention_mask != 0.0

            layer = self.layers[idx]
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states[level_index],
                multi_stage_positional_embeddings[level_index],
                query_position_embeddings,
                attn_mask_bool,
            )

            intermediate_hidden_states = self.layernorm(hidden_states)
            predicted_mask, attention_mask = self.mask_predictor(
                intermediate_hidden_states,
                pixel_embeddings,
                feature_size_list[(idx + 1) % self.num_feature_levels],
            )
            intermediate_mask_predictions.append(predicted_mask)
            intermediate.append(intermediate_hidden_states)

        return intermediate, intermediate_mask_predictions


class _TransformerModule(nn.Module):
    """Reference ``Mask2FormerTransformerModule``.

    ``queries_embedder`` (query positional embed), ``queries_features``
    (learnable query init), ``level_embed`` and the masked-attention
    ``decoder``.  ``input_projections`` are identity here (in == hidden).
    """

    def __init__(self, config: Mask2FormerConfig) -> None:
        super().__init__()
        d = config.d_model
        self.num_feature_levels = config.num_feature_levels
        self.position_embedder = SinePositionEmbedding(num_position_features=d // 2)
        self.queries_embedder = nn.Embedding(config.num_queries, d)
        self.queries_features = nn.Embedding(config.num_queries, d)
        self.decoder = _MaskedAttentionDecoder(config)
        self.level_embed = nn.Embedding(self.num_feature_levels, d)

    @override
    def forward(
        self,
        multi_scale_features: list[Tensor],  # per-level (B, C, H, W)
        mask_features: Tensor,  # (B, C, H, W)
    ) -> tuple[list[Tensor], list[Tensor]]:
        b = int(multi_scale_features[0].shape[0])

        multi_stage_features: list[Tensor] = []
        multi_stage_pos: list[Tensor] = []
        size_list: list[tuple[int, int]] = []

        level_w = self.level_embed.weight
        for i in range(self.num_feature_levels):
            feat = multi_scale_features[i]
            c = int(feat.shape[1])
            h = int(feat.shape[2])
            w = int(feat.shape[3])
            size_list.append((h, w))
            pos = self.position_embedder(feat)  # (B, C, H, W)
            pos_flat = pos.reshape(b, c, h * w)  # (B, C, S)
            feat_flat = feat.reshape(b, c, h * w)  # (B, C, S)
            lvl = level_w[i].reshape(1, c, 1)
            feat_lvl = feat_flat + lvl
            # Permute (B, C, S) -> (S, B, C)
            multi_stage_pos.append(pos_flat.permute(2, 0, 1))
            multi_stage_features.append(feat_lvl.permute(2, 0, 1))

        # Query embeddings (N, B, C)
        q_pos: Tensor = self.queries_embedder.weight.unsqueeze(1)
        q_pos = q_pos.repeat(1, b, 1)
        q_feat: Tensor = self.queries_features.weight.unsqueeze(1)
        q_feat = q_feat.repeat(1, b, 1)

        return self.decoder(
            q_feat,
            multi_stage_pos,
            mask_features,
            multi_stage_features,
            q_pos,
            size_list,
        )


# ---------------------------------------------------------------------------
# Mask2Former model
# ---------------------------------------------------------------------------


def _m2f_match(
    class_logits: Tensor,
    mask_logits: Tensor,
    gt_labels: Tensor,
    gt_masks: Tensor,
    coords: Tensor,
    cfg: Mask2FormerConfig,
) -> tuple[list[int], list[int]]:
    r"""Hungarian match between queries and ground-truth segments.

    The cost is the same three terms the loss uses, which is the point of
    the construction: matching on one objective and training on another
    would hand each query a target it was never scored against.

    .. math::

        \lambda_{\mathrm{cls}}\,(-p_i(c_j))
        + \lambda_{\mathrm{mask}}\, \mathcal{L}_{\mathrm{ce}}
        + \lambda_{\mathrm{dice}}\, \mathcal{L}_{\mathrm{dice}}

    All mask terms are evaluated on ``coords`` rather than densely, exactly
    as in the loss.

    Returns:
        ``(pred_indices, gt_indices)``.
    """
    n = int(class_logits.shape[0])
    m = int(gt_labels.shape[0])
    if n == 0 or m == 0:
        return [], []

    probs = F.softmax(class_logits, dim=-1)
    cost_class = -probs.index_select(1, gt_labels.long())  # (N, M)

    pred_pts = point_sample(mask_logits, coords)  # (N, K)
    gt_pts = point_sample(gt_masks.float(), coords)  # (M, K)

    cost = (
        cfg.class_weight * cost_class
        + cfg.mask_weight * pairwise_sigmoid_ce(pred_pts, gt_pts)
        + cfg.dice_weight * pairwise_dice(pred_pts, gt_pts)
    )
    pred_idx, gt_idx = solve_assignment(cast(list[list[float]], cost.tolist()))
    return pred_idx, gt_idx


def _m2f_stage_loss(
    class_logits: Tensor,
    mask_logits: Tensor,
    targets: list[dict[str, Tensor]],
    cfg: Mask2FormerConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Criterion for one decoder layer's predictions.

    Args:
        class_logits: ``(B, N, K + 1)`` per-query class logits.
        mask_logits:  ``(B, N, h, w)`` per-query mask logits.
        targets:      Per-image ``{"labels": (M,), "masks": (M, H, W)}``.
        cfg:          Supplies the weights and the point budget.

    Returns:
        ``(class_loss, mask_ce_loss, dice_loss)``, each already summed over
        the batch and divided by the number of images.
    """
    b = int(class_logits.shape[0])
    dev = class_logits.device.type
    bg_index = int(class_logits.shape[-1]) - 1

    cls_parts: list[Tensor] = []
    ce_parts: list[Tensor] = []
    dice_parts: list[Tensor] = []

    for i in range(b):
        cls_i = class_logits[i]  # (N, K+1)
        msk_i = mask_logits[i]  # (N, h, w)
        gt_labels = targets[i]["labels"]
        gt_masks = targets[i]["masks"]
        n = int(cls_i.shape[0])

        # Every query defaults to "no object"; matching overwrites the
        # matched ones.  Weighting that class down by no_object_weight is
        # what keeps the N - M unmatched queries from dominating.
        weights = [1.0] * (bg_index + 1)
        weights[bg_index] = cfg.no_object_weight
        weight_t = lucid.tensor(weights, dtype=cls_i.dtype, device=cls_i.device)

        if int(gt_labels.shape[0]) == 0:
            targets_i = lucid.full((n,), float(bg_index), device=dev).long()
            cls_parts.append(F.cross_entropy(cls_i, targets_i, weight=weight_t))
            continue

        # Matching uses uniform points; the loss re-samples with the
        # importance sampler, as the reference does.
        match_coords = sample_point_coords(cfg.train_num_points, device=dev)
        pred_idx, gt_idx = _m2f_match(cls_i, msk_i, gt_labels, gt_masks, match_coords, cfg)

        pred_t = lucid.tensor(pred_idx, device=dev).long()
        gt_t = lucid.tensor(gt_idx, device=dev).long()
        # Matched queries take their target's class, on device.
        targets_i = lucid.full((n,), float(bg_index), device=dev).long()
        if pred_idx:
            targets_i[pred_t] = gt_labels.long()[gt_t]
        cls_parts.append(F.cross_entropy(cls_i, targets_i, weight=weight_t))

        if not pred_idx:
            continue
        src = msk_i[pred_t]  # (P, h, w)
        tgt = gt_masks[gt_t].to(msk_i.dtype)  # (P, H, W)

        coords = uncertain_point_coords(
            src, cfg.train_num_points, cfg.oversample_ratio, cfg.importance_sample_ratio
        )
        src_pts = point_sample(src, coords)
        tgt_pts = point_sample(tgt, coords)
        ce_parts.append(sigmoid_ce_loss(src_pts, tgt_pts))
        dice_parts.append(dice_loss(src_pts, tgt_pts))

    def _mean(parts: list[Tensor]) -> Tensor:
        if not parts:
            return class_logits.new_zeros()
        return lucid.cat([t.reshape(1) for t in parts]).sum() / float(b)

    return _mean(cls_parts), _mean(ce_parts), _mean(dice_parts)


class Mask2FormerForSemanticSegmentation(SemanticSegmentationModel):
    r"""Mask2Former universal segmentation model (Cheng et al., CVPR 2022).

    A unified mask-classification architecture that achieves state-of-the-art
    results across semantic, instance, and panoptic segmentation with the same
    code path.  Its key advance over MaskFormer is **masked attention**: each
    transformer decoder layer restricts cross-attention to the foreground
    region of the *previous* layer's predicted mask, which sharply accelerates
    convergence and improves small-object segmentation.

    The pipeline is: a Swin backbone -> a multi-scale deformable-attention
    pixel decoder producing per-pixel mask features at stride 4 plus three
    coarser memory levels -> :math:`L` masked-attention decoder layers cycling
    through the three memory levels with learnable query embeddings ->
    sibling class and mask-embedding heads.

    Parameters
    ----------
    config : Mask2FormerConfig
        Frozen architecture spec.  Use the Swin-backbone factories
        (:func:`mask2former_swin_tiny` / ``small`` / ``base`` / ``large``)
        for the paper-cited ADE20K variants.

    Attributes
    ----------
    config : Mask2FormerConfig
        Stored copy of the config that built this model.
    pixel_level_module : _PixelLevelModule
        Swin ``encoder`` + MSDeformAttn pixel ``decoder``.
    transformer_module : _TransformerModule
        Learnable queries + a 9-layer masked-attention decoder.
    class_predictor : nn.Linear
        Per-query :math:`(K + 1)` class head (``+1`` for "no object").

    Notes
    -----
    See Cheng et al., "Masked-attention Mask Transformer for Universal Image
    Segmentation", CVPR 2022 (arXiv:2112.01527).  For semantic inference the
    model drops the no-object slot and collapses queries via the
    ``softmax(class)[..., :-1] ⊗ sigmoid(mask)`` einsum into a
    :math:`(B, K, H, W)` logit map, upsampled to the input resolution.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask2former import mask2former_swin_tiny
    >>> model = mask2former_swin_tiny()
    >>> x = lucid.randn(1, 3, 384, 384)
    >>> out = model(x)
    >>> out.logits.shape   # (B, K, H, W)
    (1, 150, 384, 384)
    """

    config_class: ClassVar[type[Mask2FormerConfig]] = Mask2FormerConfig
    base_model_prefix: ClassVar[str] = "mask2former"

    def __init__(self, config: Mask2FormerConfig) -> None:
        super().__init__(config)
        self._cfg = config
        d = config.d_model
        K = config.num_classes

        self.pixel_level_module = _PixelLevelModule(config)
        self.transformer_module = _TransformerModule(config)
        self.class_predictor = nn.Linear(d, K + 1)

    @override
    def forward(
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> SemanticSegmentationOutput:
        """Run Mask2Former for semantic segmentation.

        Args:
            x:       (B, C, H, W) image batch.
            targets: Optional per-image ``{"labels": (M,) class ids,
                     "masks": (M, H, W) binary}``.  When given, §3.2.2's
                     objective is computed — Hungarian matching plus the
                     class / mask-BCE / dice terms, all mask terms on
                     importance-sampled points, repeated on every decoder
                     layer under ``deep_supervision``.

        Returns:
            ``SemanticSegmentationOutput`` with ``logits`` of shape
            ``(B, K, H, W)`` (no-object slot dropped — matching the reference
            semantic post-processing), upsampled to the input resolution.
        """
        b, _, iH, iW = (int(d) for d in x.shape)
        check_per_image(type(self).__name__, b, targets=targets)

        # 1. Pixel-level module → (mask features, 3 multi-scale memory levels)
        mask_features, multi_scale_features = self.pixel_level_module(x)

        # 2. Transformer module → per-stage hidden states + mask predictions
        intermediate, mask_predictions = self.transformer_module(
            multi_scale_features, mask_features
        )

        # 3. Class + mask logits from the LAST decoder stage
        last_hidden: Tensor = intermediate[-1]  # (N, B, C)
        class_logits: Tensor = self.class_predictor(last_hidden.permute(1, 0, 2))  # (B, N, K+1)
        mask_logits: Tensor = mask_predictions[-1]  # (B, N, H/4, W/4)

        # 4. Semantic post-processing:
        #    masks_queries_logits are bilinearly upsampled FIRST, then
        #      masks_classes = softmax(class)[..., :-1]   (drop no-object)
        #      masks_probs   — sigmoid(mask)
        #      seg           — einsum("bqc,bqhw->bchw")
        #    The ordering (upsample → sigmoid → combine) is the reference's.
        #    Divergence, deliberate: the reference upsamples to a hard-coded
        #    384x384 — the size its own preprocessor resizes every image to —
        #    and only then resizes to the requested target.  Lucid has no
        #    such fixed preprocessing step, so it upsamples straight to the
        #    input resolution.  The two agree exactly for a 384x384 input;
        #    at any other size the reference interpolates twice and this
        #    interpolates once, which is the more faithful result for the
        #    resolution the caller actually passed.
        n = int(class_logits.shape[1])
        K_plus_1 = self._cfg.num_classes + 1

        mask_logits_up: Tensor = F.interpolate(
            mask_logits, size=(iH, iW), mode="bilinear", align_corners=False
        )  # (B, N, iH, iW)

        class_probs: Tensor = F.softmax(class_logits, dim=-1)  # (B, N, K+1)
        masks_classes: Tensor = class_probs[:, :, : K_plus_1 - 1]  # (B, N, K)
        masks_probs: Tensor = F.sigmoid(mask_logits_up)  # (B, N, iH, iW)

        masks_classes_t: Tensor = masks_classes.permute(0, 2, 1)  # (B, K, N)
        masks_flat: Tensor = masks_probs.reshape(b, n, iH * iW)  # (B, N, S)
        seg_flat: Tensor = lucid.matmul(masks_classes_t, masks_flat)  # (B, K, S)
        seg_logits: Tensor = seg_flat.reshape(b, self._cfg.num_classes, iH, iW)
        loss: Tensor | None = None
        if targets is not None:
            cfg = self._cfg
            # Deep supervision: 3.2.2 applies the criterion after every
            # decoder layer, not only the last.  ``intermediate`` already
            # carries them, so the auxiliary terms cost no extra forward.
            stages = range(len(intermediate)) if cfg.deep_supervision else [len(intermediate) - 1]
            parts: list[Tensor] = []
            for si in stages:
                stage_cls = self.class_predictor(intermediate[si].permute(1, 0, 2))
                cls_l, ce_l, dice_l = _m2f_stage_loss(stage_cls, mask_predictions[si], targets, cfg)
                parts.append(
                    cfg.class_weight * cls_l + cfg.mask_weight * ce_l + cfg.dice_weight * dice_l
                )
            loss = lucid.cat([t.reshape(1) for t in parts]).sum()

        return SemanticSegmentationOutput(logits=seg_logits, loss=loss)
