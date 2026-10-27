"""Attention U-Net (Oktay et al., MIDL 2018).

Paper: "Attention U-Net: Learning Where to Look for the Pancreas"

Key innovation
--------------
Each skip connection in the U-Net decoder is gated by a soft spatial
attention map.  The attention gate (AG) takes two inputs:

* **x**: (B, F, H, W) — the skip feature from the encoder.
* **g**: (B, G, H', W') — the gating signal from the decoder (coarser).

Steps inside the AG:
  1. Wx = Conv1x1(F → inter) applied to x,   then upsample to (H, W).
  2. Wg = Conv1x1(G → inter) applied to g.
  3. psi = sigmoid(Conv1x1(inter → 1)(ReLU(Wx + Wg))) — spatial map.
  4. Output = x * psi   (element-wise, broadcast over channels).

Architecture
------------
  Encoder: depth stages of DoubleConv + MaxPool2d (halves spatial dims).
  Bottleneck: DoubleConv at the deepest level.
  Decoder: depth stages of Upsample/ConvTranspose2d
             → AttentionGate on skip
             → Cat(gated_skip, upsampled_decoder)
             → DoubleConv.
  Head: Conv2d(base_channels, num_classes, 1).

Losses (training)
-----------------
  Requires integer ``targets`` of shape (B, H, W).
  Uses cross-entropy loss: F.cross_entropy(logits, targets).

Rank
----
  ``spatial_dims`` selects the convolution rank: 2 for images (the
  default, and what most callers reach for) or 3 for volumes.  The
  paper's Implementation Details specify the 3-D form — "we propose a
  3D-model to capture sufficient semantic context" — and every released
  network is Conv3d / BatchNorm3d with trilinear resampling, so
  :func:`attention_unet_3d` is the paper-faithful factory.  The gating
  mechanism, which is the paper's actual contribution, is identical in
  both; only the rank differs.
"""

from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._tasks import SemanticSegmentationModel
from lucid.models._output import SemanticSegmentationOutput
from lucid.models.vision.attention_unet._config import AttentionUNetConfig

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _conv_nd(dims: int) -> type[nn.Conv2d] | type[nn.Conv3d]:
    """``nn.Conv2d`` or ``nn.Conv3d`` for the requested rank."""
    return nn.Conv2d if dims == 2 else nn.Conv3d


def _norm_nd(dims: int) -> type[nn.BatchNorm2d] | type[nn.BatchNorm3d]:
    """``nn.BatchNorm2d`` or ``nn.BatchNorm3d`` for the requested rank."""
    return nn.BatchNorm2d if dims == 2 else nn.BatchNorm3d


def _pool_nd(dims: int) -> type[nn.MaxPool2d] | type[nn.MaxPool3d]:
    """``nn.MaxPool2d`` or ``nn.MaxPool3d`` for the requested rank."""
    return nn.MaxPool2d if dims == 2 else nn.MaxPool3d


def _deconv_nd(dims: int) -> type[nn.ConvTranspose2d] | type[nn.ConvTranspose3d]:
    """``nn.ConvTranspose2d`` or ``nn.ConvTranspose3d`` for the rank."""
    return nn.ConvTranspose2d if dims == 2 else nn.ConvTranspose3d


def _interp_mode(dims: int) -> str:
    """The interpolation mode matching the rank: bilinear or trilinear."""
    return "bilinear" if dims == 2 else "trilinear"


class _DoubleConv(nn.Module):
    """Two sequential Conv3x3-BN-ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int, dims: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _conv_nd(dims)(in_ch, out_ch, 3, padding=1, bias=False),
            _norm_nd(dims)(out_ch),
            nn.ReLU(inplace=True),
            _conv_nd(dims)(out_ch, out_ch, 3, padding=1, bias=False),
            _norm_nd(dims)(out_ch),
            nn.ReLU(inplace=True),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@final
class _AttentionGate(nn.Module):
    """Soft attention gate for U-Net skip connections.

    Args:
        x_channels: Channels in the skip feature x (F).
        g_channels: Channels in the gating signal g (G).
        inter_channels: Intermediate projection size.
    """

    def __init__(
        self, x_channels: int, g_channels: int, inter_channels: int, dims: int = 2
    ) -> None:
        super().__init__()
        # Section 3.2: "input feature-maps are downsampled to the resolution
        # of gating signal".  Wx therefore carries stride 2, so the whole
        # ReLU / psi / sigmoid stack runs on the coarse grid -- a quarter of
        # the activations, and the attention coefficients are decided at the
        # scale the gate actually carries semantics at.
        self._dims = dims
        self.Wx = _conv_nd(dims)(x_channels, inter_channels, 1, stride=2, bias=True)
        self.Wg = _conv_nd(dims)(g_channels, inter_channels, 1, bias=True)
        self.psi = _conv_nd(dims)(inter_channels, 1, 1, bias=True)

    @override
    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        """Apply attention gate.

        Args:
            x: ``(B, F, *S)`` skip features (finer scale).
            g: ``(B, G, *S/2)`` gating signal (coarser).  ``S`` is two axes
                for images and three for volumes.

        Returns:
            ``(B, F, *S)`` gated skip features.  The coefficients themselves
            are computed at the gating resolution and resampled back up.
        """
        # 1. Project *and* downsample the skip: (B, inter, H/2, W/2)
        wx: Tensor = self.Wx(x)
        # 2. Project gate and resample to wx's *actual* size.  A fixed
        # scale_factor only lands on the right shape when every spatial
        # dimension is even all the way down: MaxPool2d floors, so a 25x25
        # skip pools to 12x12 and the two grids disagree.  The reference
        # resamples to ``theta_x_size[2:]`` for exactly this reason.
        wg_raw: Tensor = self.Wg(g)
        wg: Tensor = F.interpolate(
            wg_raw,
            size=tuple(int(v) for v in wx.shape[2:]),
            mode=_interp_mode(self._dims),
            align_corners=True,
        )
        # 3. Combine and compute the attention map on the gating grid
        combined: Tensor = F.relu(wx + wg)
        att: Tensor = F.sigmoid(self.psi(combined))
        # 4. "Grid resampling of attention coefficients" back to x's size
        att = F.interpolate(
            att,
            size=tuple(int(v) for v in x.shape[2:]),
            mode=_interp_mode(self._dims),
            align_corners=True,
        )
        return x * att


class _EncoderBlock(nn.Module):
    """Encoder stage: DoubleConv → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, dims: int = 2) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_ch, out_ch, dims)
        self.pool = _pool_nd(dims)(2, stride=2)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (pooled, skip)."""
        skip: Tensor = self.conv(x)
        pooled: Tensor = self.pool(skip)
        return pooled, skip


class _DecoderBlock(nn.Module):
    """Decoder stage: Upsample → AttentionGate → Cat → DoubleConv.

    ``gated=False`` (the finest level) skips the gate entirely.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        bilinear: bool = False,
        gated: bool = True,
        dims: int = 2,
    ) -> None:
        super().__init__()
        inter_ch = out_ch // 2 if out_ch // 2 > 0 else 1
        # Section 3.2: "low-level feature-maps, i.e. the first skip
        # connections, are not used in the gating function since they do not
        # represent the input data in a high dimensional space."  The
        # reference builds 3 gates for a 4-level U-Net, passing conv1
        # straight through.
        self.gate: _AttentionGate | None = (
            _AttentionGate(skip_ch, in_ch, inter_ch, dims) if gated else None
        )
        if bilinear:
            self.up: nn.Upsample | nn.ConvTranspose2d | nn.ConvTranspose3d = nn.Upsample(
                scale_factor=2, mode=_interp_mode(dims), align_corners=True
            )
            self.conv = _DoubleConv(in_ch + skip_ch, out_ch, dims)
        else:
            self.up = _deconv_nd(dims)(in_ch, in_ch // 2, 2, stride=2)
            self.conv = _DoubleConv(in_ch // 2 + skip_ch, out_ch, dims)

    @override
    def forward(self, x: Tensor, skip: Tensor, gate: Tensor | None = None) -> Tensor:
        """Decode one level.

        Args:
            x:    (B, in_ch, H', W') — decoder tensor (coarser scale).
            skip: (B, skip_ch, 2H', 2W') — encoder skip tensor.

        Returns:
            (B, out_ch, 2H', 2W')
        """
        x_up: Tensor = self.up(x)
        # The deepest stage is handed the projected gating signal; shallower
        # stages gate on the previous decoder output, as the reference does.
        g_src = x if gate is None else gate
        gated_skip: Tensor = skip if self.gate is None else self.gate(skip, g_src)
        # The decoder's own upsample is also a fixed factor of 2, so pad it
        # up to the skip whenever the encoder's floor-division lost a row or
        # column.  Same guard the sibling U-Net applies.
        sH, sW = int(gated_skip.shape[2]), int(gated_skip.shape[3])
        uH, uW = int(x_up.shape[2]), int(x_up.shape[3])
        if sH != uH or sW != uW:
            x_up = F.pad(x_up, (0, sW - uW, 0, sH - uH))
        combined: Tensor = lucid.cat([x_up, gated_skip], dim=1)
        return self.conv(combined)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AttentionUNetForSemanticSegmentation(SemanticSegmentationModel):
    r"""Attention U-Net for medical image segmentation (Oktay et al., MIDL 2018).

    A U-Net variant that inserts a soft **attention gate** on every skip
    connection.  Each gate fuses the encoder feature :math:`x^\ell` (the
    "input" branch) with the up-sampled decoder feature :math:`g^\ell`
    (the "gating signal") via

    .. math::

        \alpha^\ell = \sigma\!\bigl(\psi^\top \mathrm{ReLU}(W_x x^\ell + W_g g^\ell)\bigr),
        \qquad
        \hat{x}^\ell = \alpha^\ell \odot x^\ell,

    suppressing skip-feature activations outside the regions of interest
    highlighted by the decoder.  This focuses the decoder's attention on
    relevant anatomy and removes the "noise" carried over from encoder
    layers — a consistent +1-3 Dice gain on medical-imaging benchmarks
    in the original paper.

    Parameters
    ----------
    config : AttentionUNetConfig
        Frozen architecture spec.  Use :func:`attention_unet` for the
        standard 4-level configuration.

    Attributes
    ----------
    config : AttentionUNetConfig
        Stored copy of the config that built this model.
    encoders : list[_EncoderBlock]
        ``config.depth`` 2-D encoder stages (DoubleConv + MaxPool).
    bottleneck : _DoubleConv
        DoubleConv at the bottom of the U.
    decoders : list[_DecoderBlock]
        ``config.depth`` decoder stages, each applying an attention gate
        to the corresponding skip feature *before* concatenating it into
        the decoder DoubleConv.
    head : nn.Conv2d
        1x1 convolution producing ``num_classes`` channels.

    Notes
    -----
    See Oktay et al., "Attention U-Net: Learning Where to Look for the
    Pancreas", MIDL 2018 (arXiv:1804.03999).  The additive attention
    gate is identical to that of Bahdanau et al. (2015) generalised to
    feature maps:

    .. math::

        q^\ell = \psi^\top \sigma_1\!\bigl(W_x x^\ell + W_g g^\ell + b_g\bigr),
        \quad
        \alpha^\ell = \sigma_2(q^\ell + b_\psi),

    with :math:`\sigma_1 = \mathrm{ReLU}`, :math:`\sigma_2 = \mathrm{sigmoid}`.
    The intermediate channel count is typically half the input channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.attention_unet import attention_unet
    >>> model = attention_unet()
    >>> x = lucid.randn(1, 1, 256, 256)
    >>> out = model(x)
    >>> out.logits.shape   # (B, num_classes, H, W)
    (1, 2, 256, 256)
    """

    config_class: ClassVar[type[AttentionUNetConfig]] = AttentionUNetConfig
    base_model_prefix: ClassVar[str] = "attention_unet"

    def __init__(self, config: AttentionUNetConfig) -> None:
        super().__init__(config)
        self._cfg = config

        ch = config.base_channels
        depth = config.depth

        # Encoder stages
        self.encoders: list[_EncoderBlock] = []
        in_ch = config.in_channels
        enc_channels: list[int] = []  # output channels at each stage
        for i in range(depth):
            out_ch = ch * (2**i)
            block = _EncoderBlock(in_ch, out_ch, config.spatial_dims)
            self.add_module(f"encoder_{i}", block)
            self.encoders.append(block)
            enc_channels.append(out_ch)
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = ch * (2**depth)
        dims = config.spatial_dims
        self.bottleneck = _DoubleConv(in_ch, bottleneck_ch, dims)
        # The reference does not hand the raw bottleneck to the gates: a
        # ``UnetGridGatingSignal`` (1x1 conv + BN + ReLU) projects it first,
        # so the gating signal is a learned summary rather than whatever the
        # deepest convolution happened to leave behind.
        self.gating = nn.Sequential(
            _conv_nd(dims)(bottleneck_ch, bottleneck_ch, 1, bias=False),
            _norm_nd(dims)(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        # Decoder stages (reverse order)
        self.decoders: list[_DecoderBlock] = []
        dec_in = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            skip_ch = enc_channels[i]
            dec_out = enc_channels[i]
            dec_block = _DecoderBlock(
                dec_in, skip_ch, dec_out, bilinear=config.bilinear, gated=i > 0, dims=dims
            )
            self.add_module(f"decoder_{i}", dec_block)
            self.decoders.append(dec_block)
            dec_in = dec_out

        # Segmentation head — also the level-0 classifier when deep
        # supervision is on, since that level needs no resampling and the
        # op is identical.
        self.head = _conv_nd(dims)(enc_channels[0], config.num_classes, 1)

        # Deep supervision (``unet_CT_multi_att_dsv``).  Every *other* decoder
        # level gets its own 1x1 classifier; the results are resampled to the
        # input resolution, concatenated, and fused by a final 1x1 conv.
        #
        # The name is misleading: this is not an auxiliary loss bolted onto
        # training.  The reference returns the fused map as *the* prediction,
        # so the deeper levels reach the output directly and the mechanism is
        # part of inference.
        self.dsv: list[nn.Conv2d | nn.Conv3d] = []
        if config.deep_supervision:
            for level in range(1, depth):
                proj = _conv_nd(dims)(enc_channels[level], config.num_classes, 1)
                self.add_module(f"dsv_{level}", proj)
                self.dsv.append(proj)
            self.fuse = _conv_nd(dims)(config.num_classes * depth, config.num_classes, 1)

    @override
    def forward(self, x: Tensor, targets: Tensor | None = None) -> SemanticSegmentationOutput:
        """Run Attention U-Net.

        Args:
            x:       ``(B, C, H, W)`` images, or ``(B, C, D, H, W)`` volumes
                when the model was built with ``spatial_dims=3``.
            targets: Optional integer ground-truth masks with the same
                spatial extent as ``x``.

        Returns:
            ``SemanticSegmentationOutput`` with logits shaped like ``x`` in
            its spatial axes and ``num_classes`` channels, plus an optional
            cross-entropy loss.
        """
        # Rank-generic: the trailing axes are the spatial ones whatever the
        # rank, so the same body serves images and volumes.
        spatial = tuple(int(v) for v in x.shape[2:])

        # Encoder path
        skips: list[Tensor] = []
        feat = x
        for enc in self.encoders:
            feat, skip = enc(feat)
            skips.append(skip)

        # Bottleneck
        feat = self.bottleneck(feat)
        gating: Tensor = self.gating(feat)

        # Decoder path.  Deep supervision classifies every stage, so keep
        # the outputs when it is on — but only then: holding them under
        # ``no_grad`` would pin intermediates the default path is done with.
        deep = self._cfg.deep_supervision
        dec_outputs: list[Tensor] = []
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            feat = dec(feat, skip, gating if i == 0 else None)
            if deep:
                dec_outputs.append(feat)

        # Segmentation head
        logits: Tensor = self.head(feat)  # (B, num_classes, H, W)

        def to_input_size(t: Tensor) -> Tensor:
            if tuple(int(v) for v in t.shape[2:]) == spatial:
                return t
            return F.interpolate(
                t, size=spatial, mode=_interp_mode(len(spatial)), align_corners=False
            )

        if deep:
            # ``decoders`` is ordered deepest-first, so ``dec_outputs[-1]`` is
            # the full-resolution level already classified by ``head``; walk
            # backwards from there to reach the coarser ones.
            maps = [to_input_size(logits)]
            for level, proj in enumerate(self.dsv, start=1):
                coarse = proj(dec_outputs[-(level + 1)])
                maps.append(to_input_size(coarse))
            logits = self.fuse(lucid.cat(maps, dim=1))

        logits = to_input_size(logits)

        loss: Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return SemanticSegmentationOutput(logits=logits, loss=loss)
