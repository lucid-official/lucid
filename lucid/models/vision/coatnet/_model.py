"""CoAtNet backbone and classification head (Dai et al., 2021).

Combines depthwise-separable MBConv stages with multi-head relative
self-attention Transformer stages following the CoAtNet-0 specification.

Architecture (CoAtNet-0, 4 stages after the stem):
  Stem   : 3×3 Conv(3→64) s=2 → BN → GELU → 3×3 Conv(64→64) → BN → GELU
  Stage 1: 2 × MBConv(64→96,  s=2)   [C-stage]
  Stage 2: 3 × MBConv(96→192, s=2)   [C-stage]
  Stage 3: 5 × RelAttnTransformer(192→384, pool-s=2)  [T-stage]
  Stage 4: 2 × RelAttnTransformer(384→768, pool-s=2)  [T-stage]
  Head   : AdaptiveAvgPool → LayerNorm → Linear(768→1000)

Reference param count: ~25.6 M (timm coatnet_0).
"""

from collections import OrderedDict
from typing import ClassVar, final, override

import threading

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._engine_scope import untraced
from lucid._tensor.tensor import Tensor
from lucid.nn._shadow import is_active as _shadow_active
from lucid.models._utils._classification import DropPath
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.coatnet._config import CoAtNetConfig

# ---------------------------------------------------------------------------
# Squeeze-and-Excitation channel attention
# ---------------------------------------------------------------------------


class _SE(nn.Module):
    """Squeeze-and-Excitation (SE) channel attention block."""

    def __init__(self, in_ch: int, se_ch: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_ch, se_ch)
        self.fc2 = nn.Linear(se_ch, in_ch)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)  →  squeeze to (B, C)  →  excite  →  (B, C, 1, 1)
        s = x.mean(dim=(2, 3))  # global average pool
        s = F.silu(self.fc1(s))
        s = F.sigmoid(self.fc2(s))
        return x * s.reshape(s.shape[0], s.shape[1], 1, 1)


# ---------------------------------------------------------------------------
# MBConv block (Mobile Inverted Bottleneck with SE)
# ---------------------------------------------------------------------------


class _MBConv(nn.Module):
    """Mobile Inverted Bottleneck: BN-pre → expand → DWConv → SE → project.

    Expansion uses ``out_ch * expand`` as mid-channels (expand_output style).
    Squeeze-and-Excitation is applied between DWConv and projection, with
    se_ch = max(1, round(out_ch * se_ratio)).

    Downsampling: stride=2 on the expansion conv (Eqn 5); shortcut pools then
    projects.  An odd side downsamples to its floor on both branches.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand: int = 4,
        stride: int = 1,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        # Table 10 lists stochastic depth for every CoAtNet variant; the
        # reference wires it through both block types.
        self.drop_path = DropPath(drop_path_rate)
        mid_ch = out_ch * expand
        se_ch = max(1, round(out_ch * se_ratio))
        self.stride = stride

        self.bn_pre = nn.BatchNorm2d(in_ch)
        # Eqn (5) puts the stride on the expansion conv:
        # ``Conv(DepthConv(Conv(Norm(x), stride=2)))``.  Striding the
        # depthwise instead is the paper's "Strided DConv" *alternative*
        # (Table 9), not the configuration its reported numbers use.
        self.expand = nn.Conv2d(in_ch, mid_ch, 1, stride=stride, bias=False)
        self.bn_exp = nn.BatchNorm2d(mid_ch)
        self.dw = nn.Conv2d(mid_ch, mid_ch, 3, stride=1, padding=1, groups=mid_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid_ch)
        self.se = _SE(mid_ch, se_ch)
        self.project = nn.Conv2d(mid_ch, out_ch, 1, bias=True)

        self.shortcut: nn.Sequential
        if stride != 1 or in_ch != out_ch:
            sc_layers: list[nn.Module] = []
            if stride != 1:
                sc_layers.append(nn.AvgPool2d(stride, stride=stride))
            sc_layers.append(nn.Conv2d(in_ch, out_ch, 1, bias=True))
            self.shortcut = nn.Sequential(*sc_layers)
        else:
            self.shortcut = nn.Sequential()  # identity

    @override
    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x)

        # Pre-activation form is ``x + Module(Norm(x))`` — Norm only; the
        # reference's pre_norm is built with apply_act=False.
        out = self.expand(self.bn_pre(x))
        if self.stride != 1:
            # A 1x1 stride-2 conv keeps ceil(H/2) rows, the shortcut's
            # AvgPool2d floor(H/2), so an odd side failed the residual add.
            # Odd grids pool with floor — what timm's down-sample and the
            # transformer stages do — so the conv drops its last sample too.
            # Even sides already agree and take no slice.
            rows, cols = int(x.shape[2]) // self.stride, int(x.shape[3]) // self.stride
            if (int(out.shape[2]), int(out.shape[3])) != (rows, cols):
                out = out[:, :, :rows, :cols]
        out = F.gelu(self.bn_exp(out))
        out = F.gelu(self.bn_dw(self.dw(out)))
        out = self.se(out)
        out = self.project(out)
        return self.drop_path(out) + shortcut


# ---------------------------------------------------------------------------
# Relative-position self-attention block (used in Transformer stages)
# ---------------------------------------------------------------------------


# The reference runs CoAtNet's transformer LayerNorms at 1e-6, not the
# framework default of 1e-5.  The gap only matters where the normalised
# variance is small, which is exactly where a deep transformer stack
# spends its time.
_LN_EPS: float = 1e-6


#: Bytes of index tables kept alive across all models and devices.  Each table
#: is ``(N, N)`` int64 for an ``N``-token grid: 8 MB for the 32×32 grid S3
#: attends on at 512 px, 134 MB for the 64×64 one at 1024 px.  The cache is
#: module-global, so what it holds outlives the model that asked for it; the
#: budget caps that.  It fits both grids of a 1024-px model, or every grid of
#: a multi-scale loop over a dozen sizes up to 512 px.  Past that, each
#: forward rebuilds the tables it misses, and a single table larger than the
#: budget (S3 above about 1220 px) is rebuilt on every call and never kept.
_INDEX_CACHE_BYTES: int = 256 * 2**20


@final
class _IndexCache:
    """Least-recently-used ``(grid, device)`` → index tables, bounded in bytes.

    One table serves every block of every model that attends on that grid —
    the blocks of a stage share their grid — where each block used to keep
    its own copy.
    """

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.nbytes = 0
        self._tables: OrderedDict[tuple[int, int, str], Tensor] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._tables)

    def get(self, grid_h: int, grid_w: int, device: str) -> Tensor:
        key = (grid_h, grid_w, device)
        with self._lock:
            held = self._tables.get(key)
            if held is not None:
                self._tables.move_to_end(key)
                return held
        # Built outside the lock: a second thread missing the same key builds
        # an equal table, and the first one stored is the one both return.
        table = _build_relative_index(grid_h, grid_w, device)
        size = int(table.nbytes)
        with self._lock:
            held = self._tables.get(key)
            if held is not None:
                return held
            if size <= self.budget:
                self._tables[key] = table
                self.nbytes += size
                while self.nbytes > self.budget:
                    _, dropped = self._tables.popitem(last=False)
                    self.nbytes -= int(dropped.nbytes)
        return table

    def clear(self) -> None:
        with self._lock:
            self._tables.clear()
            self.nbytes = 0


_INDEX_CACHE = _IndexCache(_INDEX_CACHE_BYTES)


def _relative_index(grid_h: int, grid_w: int, device: str) -> Tensor:
    """The shared ``(N, N)`` relative-offset index for ``grid`` on ``device``.

    A shadow build gets a table of its own: a phantom one in the shared cache
    would reach real models, which cannot read it.
    """
    if _shadow_active():
        return _build_relative_index(grid_h, grid_w, device)
    return _INDEX_CACHE.get(grid_h, grid_w, device)


def _build_relative_index(grid_h: int, grid_w: int, device: str) -> Tensor:
    """Flat ``(N, N)`` index of each position pair into a ``(2H-1)×(2W-1)`` table.

    Built outside any active trace, so a recording always holds the table as
    a constant feed — as it held the per-block buffer this replaced — and
    never the ops that make it.  Built inside, a cold cache put them in the
    graph and a warm one did not, so an export depended on what the process
    had run before.
    """
    with untraced():
        coords_h = lucid.arange(grid_h, device=device).to(lucid.int64)
        coords_w = lucid.arange(grid_w, device=device).to(lucid.int64)
        ys = coords_h.reshape(grid_h, 1).expand(grid_h, grid_w).reshape(-1)
        xs = coords_w.reshape(1, grid_w).expand(grid_h, grid_w).reshape(-1)
        n = grid_h * grid_w
        # Offsets shifted into [0, 2H-2] and [0, 2W-2], then flattened row-major.
        rel_h = ys.reshape(n, 1) - ys.reshape(1, n) + (grid_h - 1)
        rel_w = xs.reshape(n, 1) - xs.reshape(1, n) + (grid_w - 1)
        return rel_h * (2 * grid_w - 1) + rel_w


def _resize_rel_table(
    table: Tensor, num_heads: int, src: tuple[int, int], dst: tuple[int, int]
) -> Tensor:
    r"""Bilinearly resize a per-head offset table from grid ``src`` to ``dst``.

    Section A.1: "we simply use bi-linear interpolation to increase the size
    :math:`(2H-1) \times (2W-1)` to the desired size
    :math:`(2H'-1) \times (2W'-1)`."
    """
    src_h, src_w = 2 * src[0] - 1, 2 * src[1] - 1
    dst_h, dst_w = 2 * dst[0] - 1, 2 * dst[1] - 1
    resized = F.interpolate(
        table.reshape(1, num_heads, src_h, src_w),
        size=(dst_h, dst_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(num_heads, dst_h * dst_w)


@final
class _RelAttnBlock(nn.Module):
    """Pre-norm Transformer block with relative position bias.

    ``rel_bias`` holds one entry per relative offset of the ``(grid_h,
    grid_w)`` attention grid the block was built for.  A forward on another
    grid resizes that table bilinearly to the new grid's offsets (§A.1) *as
    a function of the parameter*, which is what the official implementation
    does: its ``Attention`` keeps ``relative_bias`` at the pre-training
    shape and resizes it with ``tf.image.resize`` inside the graph
    (``scale_ratio``).  So a forward never replaces or rewrites the
    parameter — gradients reach the original table and an optimizer built
    beforehand still owns it.  :meth:`resample_rel_pos` is the explicit way
    to rebase the table itself onto a new grid.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_h: int,
        grid_w: int,
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.0,
        out_dim: int | None = None,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)
        self.dim = dim
        self.out_dim = out_dim
        self.downsample = downsample
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim**-0.5
        # The grid the bias table is built for — the *post*-pool grid, which
        # is what attention runs on.
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Relative position bias: table is (2H-1) × (2W-1) per head
        self.rel_bias = nn.Parameter(lucid.zeros(num_heads, (2 * grid_h - 1) * (2 * grid_w - 1)))

        # Eqn (4): ``x <- Proj(Pool(x)) + Attention(Pool(Norm(x)))``.  The
        # identity branch projects the pooled input to the new width, and the
        # residual branch normalises at the *input* width before pooling — so
        # the widening happens inside attention, not in a stage-level
        # projection applied before any block.
        self.norm1 = nn.LayerNorm(dim, eps=_LN_EPS)
        self.qkv = nn.Linear(dim, out_dim * 3)
        self.proj = nn.Linear(out_dim, out_dim)
        self.shortcut: nn.Linear | nn.Identity = (
            nn.Linear(dim, out_dim) if dim != out_dim else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(out_dim, eps=_LN_EPS)
        mlp_dim = out_dim * mlp_ratio
        self.fc1 = nn.Linear(out_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)

    def resample_rel_pos(self, grid_h: int, grid_w: int) -> None:
        r"""Rebase the relative-position table onto a new attention grid.

        Section A.1: "When finetuned on a larger resolution, we simply use
        bi-linear interpolation to increase the size
        :math:`(2H-1) \times (2W-1)` to the desired size
        :math:`(2H'-1) \times (2W'-1)`."  CoAtNet-3/-4 pre-train at 224 and
        fine-tune at 384/512, so a checkpoint must not be welded to the
        resolution it was built at.

        A forward on another grid already resizes the table on the fly and
        leaves this parameter alone, training the pre-training-size table
        through the resize.  Call this instead when the fine-tune should
        train the full-size table, as a checkpoint filter that resizes at
        load time does.  It *replaces* ``rel_bias``, so build the optimizer
        after calling it.

        Args:
            grid_h: New attention-grid height.
            grid_w: New attention-grid width.

        Raises:
            ValueError: If either side of the grid is not positive.
        """
        if grid_h < 1 or grid_w < 1:
            raise ValueError(
                f"resample_rel_pos needs a positive attention grid, got " f"({grid_h}, {grid_w})"
            )
        if (grid_h, grid_w) == (self.grid_h, self.grid_w):
            return
        with lucid.no_grad():
            resized = _resize_rel_table(
                self.rel_bias, self.num_heads, (self.grid_h, self.grid_w), (grid_h, grid_w)
            )
        self.rel_bias = nn.Parameter(resized)
        self.grid_h, self.grid_w = grid_h, grid_w

    def _rel_pos_bias(self, grid: tuple[int, int]) -> Tensor:
        """The ``(num_heads, N, N)`` bias for attention on ``grid``."""
        table: Tensor = self.rel_bias
        if grid != (self.grid_h, self.grid_w):
            table = _resize_rel_table(table, self.num_heads, (self.grid_h, self.grid_w), grid)
        # Looked up by the table's device on every call, so a model moved to
        # another device reads an index that lives there.
        idx = _relative_index(grid[0], grid[1], self.rel_bias.device.type)
        n = grid[0] * grid[1]
        return table[:, idx.reshape(-1)].reshape(self.num_heads, n, n)

    def _attn(self, x: Tensor, grid: tuple[int, int]) -> Tensor:
        B, N, _C = x.shape
        qkv = self.qkv(x)  # (B, N, 3 * out_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        # Fused SDPA with the relative-position bias as an additive mask —
        # softmax((q·kᵀ)·scale + bias)·v, never forming the (B,H,N,N) scores.
        bias = self._rel_pos_bias(grid).reshape(1, self.num_heads, N, N)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.out_dim)
        return self.proj(out)

    @override
    def forward(self, x: Tensor, grid: tuple[int, int]) -> Tensor:
        # x is (B, N, C), flattened row-major from the 2-D ``grid``.  An odd
        # side pools to its floor, as the constructor's ``input_grid // 2``
        # and timm's down-sample do; the TF original pads "same" (ceil).
        # Every paper resolution gives even grids, so only odd overrides of
        # ``image_size`` see the difference.
        attn_grid = (grid[0] // 2, grid[1] // 2) if self.downsample else grid
        identity = self._pool(x, grid)
        shortcut = self.shortcut(identity)
        h = self._pool(self.norm1(x), grid)
        x = shortcut + self.drop_path1(self._attn(h, attn_grid))
        x = x + self.drop_path2(self.fc2(F.gelu(self.fc1(self.norm2(x)))))
        return x

    def _pool(self, seq: Tensor, grid: tuple[int, int]) -> Tensor:
        """Max-pool a ``(B, N, C)`` sequence on its 2-D grid, per Eqn (4)."""
        if not self.downsample:
            return seq
        b = int(seq.shape[0])
        c = int(seq.shape[2])
        planes = seq.permute(0, 2, 1).reshape(b, c, grid[0], grid[1])
        planes = F.max_pool2d(planes, 2, stride=2)
        pooled = (grid[0] // 2) * (grid[1] // 2)
        return planes.reshape(b, c, pooled).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Transformer stage (handles pool → flatten → blocks → unflatten)
# ---------------------------------------------------------------------------


@final
class _TransformerStage(nn.Module):
    """Transformer stage: flatten → N×RelAttnBlock → LayerNorm → unflatten.

    When ``downsample=True`` (default — what S3 and S4 use in CoAtNet-0..5)
    the first block max-pools both of its branches by 2 and widens
    ``in_ch → out_ch`` (Eqn 4).  When ``downsample=False`` the stage keeps
    the spatial dims — the mode CoAtNet-6 / CoAtNet-7 use for the
    transformer part of the mixed S3 (spatial halving already happened
    inside the preceding MBConv sub-stage).

    ``input_grid`` only sizes the blocks' bias tables.  The grids a forward
    runs on come from the input, so another resolution is a pure function
    of the same parameters (see :class:`_RelAttnBlock`).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        num_heads: int,
        input_grid: tuple[int, int],
        downsample: bool = True,
        dpr: list[float] | None = None,
    ) -> None:
        super().__init__()
        # Grid after optional 2× pooling.
        if downsample:
            grid_h = input_grid[0] // 2
            grid_w = input_grid[1] // 2
        else:
            grid_h = input_grid[0]
            grid_w = input_grid[1]

        self.downsample = downsample
        rates = dpr if dpr is not None else [0.0] * num_blocks
        # Eqn (4) makes the down-sample part of the *first block's* residual
        # structure, not a stage-level step: pooling once up front and then
        # running plain blocks gives a different function, because the
        # identity branch never sees the projection paired with its own pool.
        self.blocks: nn.ModuleList[_RelAttnBlock] = nn.ModuleList(
            [
                _RelAttnBlock(
                    in_ch if i == 0 else out_ch,
                    num_heads,
                    grid_h,
                    grid_w,
                    drop_path_rate=rates[i],
                    out_dim=out_ch,
                    downsample=(downsample and i == 0),
                )
                for i in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(out_ch)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) — flattened once; the first block pools if needed.
        B, C, H, W = x.shape
        out_grid = (H // 2, W // 2) if self.downsample else (H, W)
        seq = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        grid = (H, W)
        for blk in self.blocks:
            seq = blk(seq, grid)
            grid = out_grid
        seq = self.norm(seq)
        D = int(seq.shape[2])
        return seq.permute(0, 2, 1).reshape(B, D, out_grid[0], out_grid[1])


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def _build_body(
    config: CoAtNetConfig,
) -> tuple[
    nn.Sequential,  # stem
    nn.Sequential,  # s1 (MBConv)
    nn.Sequential,  # s2 (MBConv)
    _TransformerStage
    | nn.Sequential,  # s3 (Transformer for variants 0..5; Sequential for mixed-S3 variants 6/7)
    _TransformerStage,  # s4 (Transformer)
    list[FeatureInfo],
]:
    d = config.dims  # (96, 192, 384, 768)
    n = config.blocks_per_stage  # (2, 3, 5, 2)
    exp = config.mbconv_expand
    heads = config.attn_heads
    img_size = config.image_size

    # ------------------------------------------------------------------ stem
    # Two conv layers, total stride 2:  3→stem_ch→stem_ch (s=2, then s=1)
    stem_ch = config.stem_width
    # No norm/activation after the second conv: the first MBConv's pre-norm
    # supplies it immediately, and §A.1 is explicit that the architecture is
    # pre-activation throughout ("x <- x + Module(Norm(x))").  Normalising
    # here as well ran BN -> GELU -> BN -> GELU back to back, which is not a
    # no-op — the second BN renormalises an already-rectified distribution —
    # and left a dead stem_ch-sized affine pair.  timm's Stem is
    # conv1 -> norm_act -> conv2 for the same reason.
    stem = nn.Sequential(
        nn.Conv2d(config.in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch),
        nn.GELU(),
        nn.Conv2d(stem_ch, stem_ch, 3, stride=1, padding=1, bias=False),
    )
    # Stochastic depth ramps linearly with the global block index (Table 10
    # lists a per-variant rate); a dispenser keeps the cursor correct across
    # the branching S3 layouts below.
    _total_blocks = sum(n) + (
        config.mixed_s3[0] + config.mixed_s3[1] if config.mixed_s3 is not None else 0
    )
    _dpr = [config.drop_path_rate * k / max(_total_blocks - 1, 1) for k in range(_total_blocks)]
    _cursor = [0]

    def _next_dp() -> float:
        rate = _dpr[_cursor[0]] if _cursor[0] < len(_dpr) else 0.0
        _cursor[0] += 1
        return rate

    # After stem: H/2 × W/2

    # ------------------------------------------------------------------ S1
    s1_layers: list[nn.Module] = []
    s1_layers.append(_MBConv(stem_ch, d[0], expand=exp, stride=2, drop_path_rate=_next_dp()))
    for _ in range(1, n[0]):
        s1_layers.append(_MBConv(d[0], d[0], expand=exp, stride=1, drop_path_rate=_next_dp()))
    s1 = nn.Sequential(*s1_layers)
    # After S1: H/4 × W/4

    # ------------------------------------------------------------------ S2
    s2_layers: list[nn.Module] = []
    s2_layers.append(_MBConv(d[0], d[1], expand=exp, stride=2, drop_path_rate=_next_dp()))
    for _ in range(1, n[1]):
        s2_layers.append(_MBConv(d[1], d[1], expand=exp, stride=1, drop_path_rate=_next_dp()))
    s2 = nn.Sequential(*s2_layers)
    # After S2: H/8 × W/8

    # ------------------------------------------------------------------ S3
    # Two code paths:
    #   * Uniform (variants 0..5): single transformer stage as before.
    #   * Mixed (variants 6/7): MBConv sub-stage (does the spatial halving)
    #     → 1×1 channel-expand → transformer sub-stage at the wider width.
    s3_grid = (img_size // 8, img_size // 8)  # input to S3 (H/8 × W/8)
    s3: _TransformerStage | nn.Sequential
    s3_out_ch: int
    if config.mixed_s3 is None:
        s3 = _TransformerStage(
            d[1], d[2], n[2], heads[0], input_grid=s3_grid, dpr=[_next_dp() for _ in range(n[2])]
        )
        s3_out_ch = d[2]
    else:
        L_mb, L_attn, D_attn = config.mixed_s3
        # MBConv sub-stage: d[1] → d[2] with stride-2 in the first block,
        # then L_mb − 1 more isotropic blocks at d[2].  Mirrors S1/S2 layout.
        s3_mb_layers: list[nn.Module] = [
            _MBConv(d[1], d[2], expand=exp, stride=2, drop_path_rate=_next_dp())
        ]
        for _ in range(1, L_mb):
            s3_mb_layers.append(
                _MBConv(d[2], d[2], expand=exp, stride=1, drop_path_rate=_next_dp())
            )
        # 1×1 channel-expand transition: d[2] → D_attn (no spatial change).
        # Paper §A.2 only says "double the hidden dimension"; we realise it
        # as the standard 1×1 conv + BN + GELU used elsewhere in CoAtNet.
        s3_expand = nn.Sequential(
            nn.Conv2d(d[2], D_attn, 1, bias=False), nn.BatchNorm2d(D_attn), nn.GELU()
        )
        # Transformer sub-stage at width D_attn, grid 1/16 (no further pool).
        # Head count comes from config.attn_heads[0] — paper convention is
        # head_dim=32 so this stays D_attn/32 for paper-faithful configs but
        # the user can override on the config without forking the model.
        s3_attn_grid = (img_size // 16, img_size // 16)
        s3_attn = _TransformerStage(
            D_attn,
            D_attn,
            L_attn,
            heads[0],
            input_grid=s3_attn_grid,
            downsample=False,
            dpr=[_next_dp() for _ in range(L_attn)],
        )
        s3 = nn.Sequential(*s3_mb_layers, s3_expand, s3_attn)
        s3_out_ch = D_attn
    # After S3: H/16 × W/16

    # ------------------------------------------------------------------ S4
    s4_grid = (img_size // 16, img_size // 16)  # input to S4
    s4 = _TransformerStage(
        s3_out_ch, d[3], n[3], heads[1], input_grid=s4_grid, dpr=[_next_dp() for _ in range(n[3])]
    )
    # After S4: H/32 × W/32

    feature_info = [
        FeatureInfo(stage=1, num_channels=d[0], reduction=4),
        FeatureInfo(stage=2, num_channels=d[1], reduction=8),
        FeatureInfo(stage=3, num_channels=s3_out_ch, reduction=16),
        FeatureInfo(stage=4, num_channels=d[3], reduction=32),
    ]
    return stem, s1, s2, s3, s4, feature_info


# ---------------------------------------------------------------------------
# CoAtNet backbone (task="base")
# ---------------------------------------------------------------------------


class CoAtNet(PretrainedModel, BackboneMixin):
    r"""CoAtNet backbone (Dai et al., 2021).

    CoAtNet is a *hybrid* backbone that interleaves depthwise
    convolutions and relative self-attention in a single four-stage
    pyramid (preceded by a two-layer convolutional stem).  The first
    two stages (:math:`S_1, S_2`) use *MBConv* blocks (squeeze-and-
    excitation, expansion ratio 4) and the last two stages
    (:math:`S_3, S_4`) use *relative-attention* transformer blocks
    operating on flattened token sequences:

    .. math::

        \mathrm{Attn}(Q, K, V)_{ij} = \mathrm{softmax}\!\left(
            \frac{Q_i K_j^\top}{\sqrt{d}} + r_{i - j}
        \right) V_j,

    where :math:`r_{i-j}` is a learned bias indexed by the *relative*
    spatial offset between tokens.  This recovers the translation
    equivariance that convolutions provide while still permitting
    global, data-dependent mixing.  Each stage downsamples
    :math:`2\times`, so the final feature map is
    :math:`(B, d_{S_4}, H/32, W/32)`.

    :meth:`forward_features` returns the raw spatial feature map from
    the last attention stage.  Use this backbone when you need
    multi-scale or spatial features for detection / segmentation; for
    end-to-end classification use
    :class:`CoAtNetForImageClassification`.

    Parameters
    ----------
    config : CoAtNetConfig
        Frozen dataclass specifying ``blocks_per_stage``, ``dims``,
        ``stem_width``, ``attn_heads``, ``mbconv_expand``,
        ``image_size``, and ``in_channels``.  See :class:`CoAtNetConfig`.

    Attributes
    ----------
    stem : nn.Sequential
        Two-layer stride-2 convolutional stem
        :math:`(3\times3 \,\mathrm{Conv}, \mathrm{BN}, \mathrm{GELU})^2`.
    s1 : nn.Sequential
        First MBConv stage with downsampling.
    s2 : nn.Sequential
        Second MBConv stage with downsampling.
    s3 : _TransformerStage
        First relative-attention transformer stage.
    s4 : _TransformerStage
        Second relative-attention transformer stage.
    feature_info : list[FeatureInfo]
        Four-stage feature description with reductions
        :math:`(4, 8, 16, 32)`.

    Notes
    -----
    Reference: Zihang Dai *et al.*, *"CoAtNet: Marrying Convolution
    and Attention for All Data Sizes"*, NeurIPS 2021,
    `arXiv:2106.04803 <https://arxiv.org/abs/2106.04803>`_.

    Examples
    --------
    Build a CoAtNet-0 backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.coatnet import CoAtNet, CoAtNetConfig
    >>> model = CoAtNet(CoAtNetConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape                       # (B, dims[-1], H/32, W/32)
    (1, 768, 7, 7)
    """

    config_class: ClassVar[type[CoAtNetConfig]] = CoAtNetConfig
    base_model_prefix: ClassVar[str] = "coatnet"

    def __init__(self, config: CoAtNetConfig) -> None:
        super().__init__(config)
        stem, s1, s2, s3, s4, fi = _build_body(config)
        self.stem = stem
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self._feature_info = fi

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        return x

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# CoAtNet for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class CoAtNetForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""CoAtNet with a linear classification head (Dai et al., 2021).

    Wraps the same conv + attention trunk as :class:`CoAtNet` (stem
    + two MBConv stages + two relative-attention transformer stages)
    and adds the standard reference recipe head: global average pool
    → LayerNorm → optional pre-logits Linear + Tanh → linear
    classifier.

    .. math::

        \text{logits} = W_{\text{cls}}\,
            \mathrm{Tanh}\!\bigl(W_{\text{pre}}\,
            \mathrm{LN}(\mathrm{GAP}(z^{S_4}))\bigr) + b_{\text{cls}}.

    Pass ``labels`` to :meth:`forward` to compute the cross-entropy
    loss in the same pass.

    Parameters
    ----------
    config : CoAtNetConfig
        Architecture specification.  Must set ``num_classes`` to the
        desired number of output categories.  Set
        ``head_hidden_size=None`` to drop the pre-logits projection.
        See :class:`CoAtNetConfig`.

    Attributes
    ----------
    stem : nn.Sequential
        Two-layer stride-2 convolutional stem.
    s1, s2 : nn.Sequential
        Two MBConv stages.
    s3, s4 : _TransformerStage
        Two relative-attention transformer stages.
    avgpool : nn.AdaptiveAvgPool2d
        :math:`1 \times 1` adaptive average pool over spatial dims.
    norm : nn.LayerNorm
        LayerNorm applied to the pooled feature.
    pre_logits : nn.Module
        Either ``Linear + Tanh`` (when ``config.head_hidden_size`` is
        set) or an identity ``nn.Sequential``.
    classifier : nn.Linear
        Final linear projection of width ``(num_classes, head_in)``
        where ``head_in`` is either ``config.head_hidden_size`` or
        ``config.dims[-1]``.

    Notes
    -----
    Reference: Zihang Dai *et al.*, *"CoAtNet: Marrying Convolution
    and Attention for All Data Sizes"*, NeurIPS 2021.  CoAtNet-0
    reaches **81.6% top-1 on ImageNet-1k** at 224x224 (Table 5).

    Examples
    --------
    End-to-end inference with the default CoAtNet-0 classifier:

    >>> import lucid
    >>> from lucid.models.vision.coatnet import (
    ...     CoAtNetConfig, CoAtNetForImageClassification,
    ... )
    >>> model = CoAtNetForImageClassification(CoAtNetConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """

    config_class: ClassVar[type[CoAtNetConfig]] = CoAtNetConfig
    base_model_prefix: ClassVar[str] = "coatnet"

    def __init__(self, config: CoAtNetConfig) -> None:
        super().__init__(config)
        stem, s1, s2, s3, s4, _ = _build_body(config)
        self.stem = stem
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = config.dims[-1]
        # Three pieces here are reference-faithful but *not* described in the
        # paper: this head LayerNorm, the pre-logits Linear+Tanh below, and
        # the per-stage LayerNorm in ``_Stage``.  They come from timm's
        # ``NormMlpClassifierHead`` (pool -> norm -> fc1 -> tanh -> fc2) and
        # from the ``head_hidden_size=768`` its paper-matching ``coatnet_0``
        # entry carries.  The cost is measurable — 24.79M for the backbone
        # against 26.15M with the head, so ~1.36M of it, versus the paper's
        # 25M total in Table 4.  Kept because the converted checkpoints have
        # these tensors; removing them would strand every shipped weight.
        self.norm = nn.LayerNorm(feat_dim)
        # Optional hidden layer (timm head_hidden_size=768 for coatnet_0)
        if config.head_hidden_size is not None:
            self.pre_logits: nn.Sequential = nn.Sequential(
                nn.Linear(feat_dim, config.head_hidden_size), nn.Tanh()
            )
            head_in = config.head_hidden_size
        else:
            self.pre_logits = nn.Sequential()
            head_in = feat_dim
        self._build_classifier(head_in, config.num_classes, dropout=config.dropout)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.norm(x)
        x = self.pre_logits(x)
        logits = self.classifier(x)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
