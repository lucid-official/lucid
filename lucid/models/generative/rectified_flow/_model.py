"""Rectified Flow model — straight paths and reflow (Liu, Gong & Liu, 2023).

Training, per batch:

    x1 ~ data,  x0 ~ pi_0,  t ~ U[eps, 1]
    x_t    = t * x1 + (1 - t) * x0        (the straight line between them)
    target = x1 - x0                      (its constant velocity)
    loss   = || v(x_t, t) - target ||^2

Nothing is solved.  This is Flow Matching's optimal-transport objective at
``sigma_min = 0``, and on its own it buys nothing new — the paper's
contribution is what happens next.

**Reflow.**  Solve the trained ODE from noise to data and keep the
endpoint pairs.  Because the solve preserves marginals (Theorem 3.3) those
pairs are a valid — but no longer independent — coupling of the same two
distributions, so the same objective can be run again on them.  ``x0`` is
then the noise that *produced* ``x1`` rather than an unrelated draw, and
the field has a far easier function to fit: the paths straighten.

**Distillation.**  The same loss with ``t`` pinned instead of drawn.  At
``t = 0`` it regresses ``v(x0, 0)`` onto ``x1 - x0``, which makes
``x0 + v(x0, 0)`` a one-step generator.

All three are one method — :meth:`RectifiedFlowModel.rectified_flow_loss`
— separated only by where ``t`` comes from and whether the pair was drawn
or generated.
"""

import math
from collections.abc import Callable
from typing import ClassVar, TypeVar, final, override

import lucid
import lucid.diffeq
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._dtype import _float32_or_wider
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageGenerationModel
from lucid.models._output import DiffusionModelOutput, GenerationOutput
from lucid.models._utils._generative import (
    LikelihoodDynamics,
    check_image_shape,
    cnf_gaussian_log_prob,
    generative_activation,
    odeint_endpoint,
    resolve_generation_device,
    start_noise,
)
from lucid.models.generative.rectified_flow._config import RectifiedFlowConfig, TimeSchedule

# What ``lucid.diffeq`` calls a state: one tensor, or a tuple of them when
# more than one quantity is integrated at once.
_StateT = TypeVar("_StateT", Tensor, tuple[Tensor, Tensor])

# The time embedding was designed for the integer step index of a
# discrete-time diffusion model.  The reference implementation feeds it
# ``t * 999`` — the last index of a 1000-step grid, not 1000 — so a
# continuous ``t`` in ``[0, 1]`` covers exactly the range the embedding
# was built for.
_TIME_SCALE = 999.0

# ``variance_scaling(scale=0)`` in the reference substitutes this for the
# requested scale, so a nominally zero-initialised branch starts at
# effectively — but not exactly — zero.
_ZERO_INIT_SCALE = 1e-10


def _gain(scale: float) -> float:
    """Xavier gain matching the reference's ``variance_scaling(scale)``.

    The reference initialises every weight uniformly with variance
    ``scale / mean(fan_in, fan_out)``, giving a bound of
    ``sqrt(6 * scale / (fan_in + fan_out))``.  Xavier-uniform with gain
    ``g`` gives ``g * sqrt(6 / (fan_in + fan_out))``, so the two agree
    exactly at ``g = sqrt(scale)``.
    """
    return math.sqrt(_ZERO_INIT_SCALE if scale == 0.0 else scale)


def _init_conv(conv: nn.Conv2d, scale: float) -> nn.Conv2d:
    """Initialise a convolution the way the reference does."""
    nn.init.xavier_uniform_(conv.weight, gain=_gain(scale))
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def _conv3x3(in_ch: int, out_ch: int, *, scale: float = 1.0) -> nn.Conv2d:
    return _init_conv(nn.Conv2d(in_ch, out_ch, 3, padding=1), scale)


def _conv1x1(in_ch: int, out_ch: int, *, scale: float = 1.0) -> nn.Conv2d:
    return _init_conv(nn.Conv2d(in_ch, out_ch, 1), scale)


def _groups(channels: int, limit: int) -> int:
    """GroupNorm group count — ``min(channels // 4, limit)``, as in the reference."""
    return max(1, min(channels // 4, limit))


# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────


def _fir_kernel_2d(taps: tuple[int, ...], gain: float) -> Tensor:
    """Separable FIR filter as a normalised 2-D kernel, ready to correlate.

    The outer product of the taps with themselves, scaled to unit sum and
    then by ``gain``.  It is reversed on both axes because Lucid's
    ``conv2d`` cross-correlates: flipping the kernel first turns that into
    a true convolution, which is what the reference performs.  For the
    symmetric default ``(1, 3, 3, 1)`` the flip is a no-op, but a
    user-supplied asymmetric filter would otherwise be applied mirrored.
    """
    reversed_taps = list(taps)[::-1]
    total = float(sum(taps)) ** 2
    rows = [[a * b * gain / total for b in reversed_taps] for a in reversed_taps]
    return lucid.tensor(rows)


def _zero_insert(x: Tensor, factor: int) -> Tensor:
    """Upsample by writing each value into the corner of a ``factor`` block."""
    batch, channels, height, width = (int(s) for s in x.shape)
    grid = x.reshape(batch, channels, height, 1, width, 1)
    padded = F.pad(grid, (0, factor - 1, 0, 0, 0, factor - 1))
    return padded.reshape(batch, channels, height * factor, width * factor)


@final
class _FIRResample(nn.Module):
    """Filtered 2x up- or downsampling — the reference's ``upfirdn2d``.

    Upsampling inserts zeros, pads asymmetrically and convolves with the
    filter at gain ``factor**2`` so the mean is preserved; downsampling
    pads and convolves with stride 2.  Both are parameter-free: the kernel
    is a buffer, so it moves with the module but is never trained.

    The strided convolution is exactly the reference's convolve-then-slice
    — stride 2 keeps output columns ``0, 2, 4, …``, which is the same set
    ``[::2]`` selects — at half the arithmetic.
    """

    kernel: Tensor

    def __init__(self, channels: int, taps: tuple[int, ...], *, up: bool) -> None:
        super().__init__()
        self._up = up
        self._channels = channels
        size = len(taps)
        # ``p`` is how much longer the filter is than the resampling
        # factor; splitting it unevenly is what keeps the output aligned
        # with the input grid rather than shifted by half a pixel.
        overhang = size - 2
        if up:
            self._pad = ((overhang + 1) // 2 + 1, overhang // 2)
            gain = 4.0
        else:
            self._pad = ((overhang + 1) // 2, overhang // 2)
            gain = 1.0
        kernel = _fir_kernel_2d(taps, gain).reshape(1, 1, size, size)
        self.register_buffer("kernel", kernel * lucid.ones((channels, 1, 1, 1)))

    @override
    def forward(self, x: Tensor) -> Tensor:
        left, right = self._pad
        if self._up:
            x = _zero_insert(x, 2)
            stride = 1
        else:
            stride = 2
        x = F.pad(x, (left, right, left, right))
        return F.conv2d(x, self.kernel, stride=stride, groups=self._channels)


def _naive_upsample(x: Tensor) -> Tensor:
    """Nearest-neighbour 2x — each pixel repeated into a 2x2 block."""
    return F.interpolate(x, scale_factor=2.0, mode="nearest")


def _naive_downsample(x: Tensor) -> Tensor:
    """Mean over each 2x2 block."""
    return F.avg_pool2d(x, 2)


@final
class _Resample(nn.Module):
    """2x resampling, filtered or not, in one direction.

    Used for the resolution pyramid, which carries the *image* rather than
    features and therefore has no convolution of its own.
    """

    def __init__(self, channels: int, taps: tuple[int, ...], *, up: bool, fir: bool) -> None:
        super().__init__()
        self._up = up
        self.filter = _FIRResample(channels, taps, up=up) if fir else None

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.filter is not None:
            return self.filter(x)
        return _naive_upsample(x) if self._up else _naive_downsample(x)


# ─────────────────────────────────────────────────────────────────────────────
# Time conditioning
# ─────────────────────────────────────────────────────────────────────────────


@final
class _FourierProjection(nn.Module):
    """Random Fourier features of the (log) time.

    The frequencies are drawn once and frozen — a buffer, not a parameter
    — so the embedding is fixed but still travels with the checkpoint and
    with ``.to(device=...)``.
    """

    freqs: Tensor

    def __init__(self, size: int, scale: float) -> None:
        super().__init__()
        self.register_buffer("freqs", lucid.randn((size,)) * scale)

    @override
    def forward(self, x: Tensor) -> Tensor:
        projected = x.reshape(-1, 1) * self.freqs.reshape(1, -1) * (2.0 * math.pi)
        return lucid.cat([lucid.sin(projected), lucid.cos(projected)], dim=-1)


def _sinusoidal(t: Tensor, dim: int, base: float = 10_000.0) -> Tensor:
    """Sinusoidal ladder, in the reference's ``[sin, cos]`` order."""
    half = dim // 2
    denom = max(half - 1, 1)
    step = math.log(base) / denom
    freqs = lucid.tensor(
        [math.exp(-step * index) for index in range(half)], dtype=t.dtype, device=t.device
    )
    angles = t.reshape(-1, 1) * freqs.reshape(1, -1)
    emb = lucid.cat([lucid.sin(angles), lucid.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Trunk blocks
# ─────────────────────────────────────────────────────────────────────────────


@final
class _ResBlock(nn.Module):
    """Residual block that can resample inside itself.

    Two normalise-activate-convolve halves with an additive per-channel
    time projection between them.  Where a plain U-Net puts resampling in
    a separate layer, this block applies it to *both* branches after the
    first normalisation, so the skip and the residual are resampled by the
    same operator — the arrangement large-scale GAN generators use and
    that the architecture here inherits.

    ``skip_rescale`` divides the sum by :math:`\\sqrt{2}`.  With the last
    convolution initialised near zero, each block starts as
    :math:`x/\\sqrt{2}`, so a deep stack neither explodes nor decays.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        *,
        act_fn: str,
        dropout: float,
        group_limit: int,
        skip_rescale: bool,
        init_scale: float,
        taps: tuple[int, ...],
        fir: bool,
        up: bool = False,
        down: bool = False,
    ) -> None:
        super().__init__()
        self._act_name = act_fn
        self._skip_rescale = skip_rescale
        self._resample: _Resample | None = None
        if up or down:
            self._resample = _Resample(in_channels, taps, up=up, fir=fir)

        self.norm1 = nn.GroupNorm(_groups(in_channels, group_limit), in_channels, eps=1e-6)
        self.conv1 = _conv3x3(in_channels, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        nn.init.xavier_uniform_(self.time_proj.weight, gain=_gain(1.0))
        if self.time_proj.bias is not None:
            nn.init.zeros_(self.time_proj.bias)
        self.norm2 = nn.GroupNorm(_groups(out_channels, group_limit), out_channels, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = _conv3x3(out_channels, out_channels, scale=init_scale)

        self.skip: nn.Conv2d | None = None
        if in_channels != out_channels or up or down:
            self.skip = _conv1x1(in_channels, out_channels)

    @override
    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        act = self._act_name
        h = generative_activation(act, self.norm1(x))
        if self._resample is not None:
            # Both branches, so the residual sum stays well defined.
            h = self._resample(h)
            x = self._resample(x)
        h = self.conv1(h)
        shift = self.time_proj(generative_activation(act, t_emb))
        h = h + shift.unsqueeze(-1).unsqueeze(-1)
        h = generative_activation(act, self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        if self.skip is not None:
            x = self.skip(x)
        out = x + h
        return out / math.sqrt(2.0) if self._skip_rescale else out


@final
class _AttnBlock(nn.Module):
    """Single-head self-attention over a feature map.

    One head, scaled by :math:`C^{-1/2}` over the full channel width —
    the original formulation, not the multi-head arrangement later U-Nets
    adopted.  Kept faithful because it is the architecture the reported
    numbers were produced with.
    """

    def __init__(
        self, channels: int, *, group_limit: int, skip_rescale: bool, init_scale: float
    ) -> None:
        super().__init__()
        self._skip_rescale = skip_rescale
        self.norm = nn.GroupNorm(_groups(channels, group_limit), channels, eps=1e-6)
        # The reference's ``NIN`` is a 1x1 projection at gain sqrt(0.1).
        self.query = _conv1x1(channels, channels, scale=0.1)
        self.key = _conv1x1(channels, channels, scale=0.1)
        self.value = _conv1x1(channels, channels, scale=0.1)
        self.proj = _conv1x1(channels, channels, scale=init_scale)
        self.scale = 1.0 / math.sqrt(channels)

    @override
    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = (int(s) for s in x.shape)
        n = height * width
        h = self.norm(x)
        q = self.query(h).reshape(batch, channels, n)
        k = self.key(h).reshape(batch, channels, n)
        v = self.value(h).reshape(batch, channels, n)
        scores = q.mT @ k * self.scale
        out: Tensor = v @ F.softmax(scores, dim=-1).mT
        out = self.proj(out.reshape(batch, channels, height, width))
        result = x + out
        return result / math.sqrt(2.0) if self._skip_rescale else result


# ─────────────────────────────────────────────────────────────────────────────
# Velocity field
# ─────────────────────────────────────────────────────────────────────────────


class _VelocityField(nn.Module):
    r"""U-Net velocity field :math:`v(x, t; \theta)`.

    The architecture the paper trains, which differs from the U-Net Flow
    Matching borrows in four ways that all matter to reproduction:

    * **Residual blocks resample internally**, applying the same operator
      to the skip and the residual, rather than through separate layers.
    * **Attention is single-head** and applied at one resolution only.
    * **Every residual and attention sum is divided by**
      :math:`\sqrt{2}`, and each block's last convolution starts at
      effectively zero, so a deep stack begins near the identity.
    * **A resolution pyramid** may run alongside the trunk at high
      resolution, feeding the input down (``progressive_input``) and
      accumulating the output up (``progressive``) through filtered
      resampling.

    Parameters
    ----------
    config : RectifiedFlowConfig
        Width, depth, attention resolutions, resampling and pyramid
        settings.

    Attributes
    ----------
    time_mlp : nn.Module
        Projects the time embedding to the conditioning width.
    conv_in : nn.Conv2d
        Stem convolution.
    down_res, down_attn, down_sample : nn.ModuleList
        Encoder blocks; ``down_sample`` entries are residual blocks with
        internal downsampling.
    mid_block1, mid_attn, mid_block2 : nn.Module
        Bottleneck stack.
    up_res, up_attn, up_sample : nn.ModuleList
        Decoder blocks.  Attention here is applied once per resolution,
        after the block group, not once per block.

    Notes
    -----
    Reference: Liu, Gong, and Liu, *"Flow Straight and Fast"*, ICLR, 2023
    (arXiv:2209.03003).  The network is the one of Song et al., *"Score-Based
    Generative Modeling through Stochastic Differential Equations"*, ICLR,
    2021 (arXiv:2011.13456); the released configurations for this paper are
    what the defaults reproduce.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.rectified_flow import RectifiedFlowConfig
    >>> from lucid.models.generative.rectified_flow._model import _VelocityField
    >>> cfg = RectifiedFlowConfig(sample_size=8, base_channels=16,
    ...                           channel_mult=(1, 2), num_res_blocks=1,
    ...                           attention_resolutions=(4,))
    >>> field = _VelocityField(cfg).eval()
    >>> field(lucid.randn((2, 3, 8, 8)), lucid.tensor([0.3, 0.7])).shape
    (2, 3, 8, 8)
    """

    def __init__(self, config: RectifiedFlowConfig) -> None:
        super().__init__()
        self._config = config
        levels = len(config.channel_mult)
        base = config.base_channels
        time_dim = base * 4
        self._act_name = config.act_fn
        self._data_centered = config.data_centered
        self._levels = levels
        self._blocks_per_down = config.num_res_blocks
        self._blocks_per_up = config.num_res_blocks + 1
        self._scale_by_sigma = config.scale_by_sigma
        self._progressive = config.progressive
        self._progressive_input = config.progressive_input
        self._time_floor = config.time_eps * _TIME_SCALE

        taps = config.fir_kernel

        def _res(in_ch: int, out_ch: int, *, up: bool = False, down: bool = False) -> _ResBlock:
            return _ResBlock(
                in_ch,
                out_ch,
                time_dim,
                act_fn=config.act_fn,
                dropout=config.dropout,
                group_limit=config.resnet_groups,
                skip_rescale=config.skip_rescale,
                init_scale=config.init_scale,
                taps=taps,
                fir=config.fir,
                up=up,
                down=down,
            )

        def _attn(c: int) -> _AttnBlock:
            return _AttnBlock(
                c,
                group_limit=config.resnet_groups,
                skip_rescale=config.skip_rescale,
                init_scale=config.init_scale,
            )

        if config.embedding_type == "fourier":
            self.fourier: _FourierProjection | None = _FourierProjection(base, config.fourier_scale)
            embed_dim = 2 * base
        else:
            self.fourier, embed_dim = None, base
        # ``config.act_fn`` drives every other activation in the net; the
        # time MLP was the one place that hardcoded SiLU, so asking for a
        # different activation silently left this branch alone.
        self.time_mlp_fc1 = nn.Linear(embed_dim, time_dim)
        self.time_mlp_fc2 = nn.Linear(time_dim, time_dim)
        # The reference initialises both of these with its default
        # variance_scaling(1.0, 'fan_avg', 'uniform') — Xavier-uniform gain 1
        # — and zeroes the biases, exactly as ``_init_conv`` and
        # ``_ResBlock.time_proj`` already do here.  Left on Lucid's Linear
        # default the second layer came out ~1.7x too narrow and the biases
        # were non-zero.
        for _linear in (self.time_mlp_fc1, self.time_mlp_fc2):
            nn.init.xavier_uniform_(_linear.weight, gain=1.0)
            if _linear.bias is not None:
                nn.init.zeros_(_linear.bias)

        self.conv_in = _conv3x3(config.in_channels, base)

        size = config.sample_size
        # The reference keys attention on the *width*; on a
        # non-square input the two disagree and attention lands at
        # the wrong stage entirely.
        spatial = size[1] if isinstance(size, tuple) else int(size)

        # Pyramid resamplers carry the image itself, so they keep the
        # sample's channel count and hold no convolution.
        self.pyramid_down: _Resample | None = (
            _Resample(config.in_channels, taps, up=False, fir=config.fir)
            if config.progressive_input == "input_skip"
            else None
        )
        self.pyramid_up: _Resample | None = (
            _Resample(config.in_channels, taps, up=True, fir=config.fir)
            if config.progressive == "output_skip"
            else None
        )

        self.down_res: nn.ModuleList[_ResBlock] = nn.ModuleList()
        self.down_attn: nn.ModuleList[_AttnBlock | nn.Identity] = nn.ModuleList()
        self.down_sample: nn.ModuleList[_ResBlock] = nn.ModuleList()
        self.down_combine: nn.ModuleList[nn.Conv2d | nn.Identity] = nn.ModuleList()
        skip_channels: list[int] = [base]
        ch = base
        for level, mult in enumerate(config.channel_mult):
            out_ch = base * mult
            for _ in range(config.num_res_blocks):
                self.down_res.append(_res(ch, out_ch))
                ch = out_ch
                self.down_attn.append(
                    _attn(ch) if spatial in config.attention_resolutions else nn.Identity()
                )
                skip_channels.append(ch)
            if level != levels - 1:
                self.down_sample.append(_res(ch, ch, down=True))
                self.down_combine.append(
                    _conv1x1(config.in_channels, ch)
                    if config.progressive_input == "input_skip"
                    else nn.Identity()
                )
                skip_channels.append(ch)
                spatial //= 2
            else:
                self.down_combine.append(nn.Identity())

        self.mid_block1 = _res(ch, ch)
        self.mid_attn = _attn(ch)
        self.mid_block2 = _res(ch, ch)

        self.up_res: nn.ModuleList[_ResBlock] = nn.ModuleList()
        self.up_attn: nn.ModuleList[_AttnBlock | nn.Identity] = nn.ModuleList()
        self.up_sample: nn.ModuleList[_ResBlock] = nn.ModuleList()
        self.pyramid_norm: nn.ModuleList[nn.GroupNorm | nn.Identity] = nn.ModuleList()
        self.pyramid_conv: nn.ModuleList[nn.Conv2d | nn.Identity] = nn.ModuleList()
        for level, mult in list(enumerate(config.channel_mult))[::-1]:
            out_ch = base * mult
            for _ in range(self._blocks_per_up):
                skip_ch = skip_channels.pop()
                self.up_res.append(_res(ch + skip_ch, out_ch))
                ch = out_ch
            # One attention per resolution here, unlike the encoder.
            self.up_attn.append(
                _attn(ch) if spatial in config.attention_resolutions else nn.Identity()
            )
            if config.progressive == "output_skip":
                self.pyramid_norm.append(
                    nn.GroupNorm(_groups(ch, config.resnet_groups), ch, eps=1e-6)
                )
                self.pyramid_conv.append(_conv3x3(ch, config.in_channels, scale=config.init_scale))
            else:
                self.pyramid_norm.append(nn.Identity())
                self.pyramid_conv.append(nn.Identity())
            if level != 0:
                self.up_sample.append(_res(ch, ch, up=True))
                spatial *= 2

        if config.progressive == "output_skip":
            # The pyramid already carries the output; a trunk head would
            # be dead weight.
            self.norm_out: nn.GroupNorm | nn.Identity = nn.Identity()
            self.conv_out: nn.Conv2d | nn.Identity = nn.Identity()
        else:
            self.norm_out = nn.GroupNorm(_groups(ch, config.resnet_groups), ch, eps=1e-6)
            self.conv_out = _conv3x3(ch, config.in_channels, scale=config.init_scale)

    def _time_embedding(self, t: Tensor) -> Tensor:
        dtype = self.time_mlp_fc1.weight.dtype  # angles reach 999 rad: float32+
        scaled = t.to(_float32_or_wider(dtype)) * _TIME_SCALE
        if self.fourier is not None:
            # The reference embeds the log of the time, a habit from conditioning
            # on a noise level: monotone in ``t``, but singular at ``t = 0``.  The
            # floor is the smallest time training or sampling produces, so the
            # clamp changes nothing in range and removes the singularity.
            floor = lucid.full_like(scaled, max(self._time_floor, 1e-12))
            return self.fourier(lucid.log(lucid.maximum(scaled, floor))).to(dtype)
        return _sinusoidal(scaled, self._config.base_channels).to(dtype)

    @override
    def forward(self, sample: Tensor, t: Tensor) -> Tensor:
        """Evaluate the velocity field.

        Parameters
        ----------
        sample : Tensor
            ``(B, C, H, W)`` point to evaluate at.
        t : Tensor
            ``(B,)`` or ``()`` time in ``[0, 1]``.  A scalar is broadcast
            across the batch, which is what the ODE solver passes.

        Returns
        -------
        Tensor
            ``(B, C, H, W)`` velocity.
        """
        if t.ndim == 0:
            t = t.reshape((1,)).expand((int(sample.shape[0]),))
        t_hidden = self.time_mlp_fc1(self._time_embedding(t))
        t_emb = self.time_mlp_fc2(generative_activation(self._act_name, t_hidden))

        pyramid_in = sample if self._progressive_input == "input_skip" else None
        if not self._data_centered:
            # Reference ncsnpp.forward: ``if not config.data.centered: x = 2*x - 1``.
            sample = sample * 2.0 - 1.0
        h = self.conv_in(sample)
        skips: list[Tensor] = [h]

        idx = 0
        for level in range(self._levels):
            for _ in range(self._blocks_per_down):
                h = self.down_res[idx](h, t_emb)
                h = self.down_attn[idx](h)
                skips.append(h)
                idx += 1
            if level != self._levels - 1:
                h = self.down_sample[level](h, t_emb)
                if pyramid_in is not None:
                    assert self.pyramid_down is not None
                    pyramid_in = self.pyramid_down(pyramid_in)
                    h = h + self.down_combine[level](pyramid_in)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        pyramid_out: Tensor | None = None
        idx = 0
        for order, level in enumerate(range(self._levels - 1, -1, -1)):
            for _ in range(self._blocks_per_up):
                h = lucid.cat([h, skips.pop()], dim=1)
                h = self.up_res[idx](h, t_emb)
                idx += 1
            h = self.up_attn[order](h)
            if self._progressive == "output_skip":
                contribution = self.pyramid_conv[order](
                    generative_activation(self._act_name, self.pyramid_norm[order](h))
                )
                if pyramid_out is None:
                    pyramid_out = contribution
                else:
                    assert self.pyramid_up is not None
                    pyramid_out = self.pyramid_up(pyramid_out) + contribution
            if level != 0:
                h = self.up_sample[order](h, t_emb)

        if pyramid_out is not None:
            out = pyramid_out
        else:
            h = generative_activation(self._act_name, self.norm_out(h))
            out = self.conv_out(h)

        if self._scale_by_sigma:
            # NCSN++ output scaling, which every released 256x256 config turns
            # on: ``h = h / used_sigmas``.  Under the Fourier embedding the
            # reference's ``used_sigmas`` *is* the conditioning value it was
            # handed, and its callers always pass ``t * 999`` — so the trained
            # high-resolution velocity is ``unet(x, t) / (999 t)``.  The
            # CIFAR-10 config sets this False, which is why it stays off by
            # default here.
            scaled_t = t * _TIME_SCALE
            # Same floor the Fourier embedding uses: the divisor is singular
            # at t = 0, and the floor is the smallest value training or
            # sampling ever produces, so clamping changes nothing in range.
            floor = lucid.full_like(scaled_t, max(self._time_floor, 1e-12))
            denom = lucid.maximum(scaled_t, floor).reshape(-1, *([1] * (len(sample.shape) - 1)))
            out = out / denom
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Likelihood dynamics — only used at inference
# ─────────────────────────────────────────────────────────────────────────────


_LikelihoodDynamics = LikelihoodDynamics


# ─────────────────────────────────────────────────────────────────────────────
# Direct model
# ─────────────────────────────────────────────────────────────────────────────


class RectifiedFlowModel(PretrainedModel):
    r"""Velocity field trained to transport along straight paths.

    ``forward(sample, t)`` evaluates :math:`v(x, t; \theta)`.  Everything
    that makes it the paper's method is separate, and each piece maps onto
    one part of it:

    * :meth:`path_sample` — the straight line :math:`t x_1 + (1-t) x_0`.
    * :meth:`conditional_target` — its velocity, :math:`x_1 - x_0`.
    * :meth:`rectified_flow_loss` — the objective, eq. (1).  With a
      ``noise`` argument it is the reflow objective; with a pinned
      ``t_schedule`` it is distillation.
    * :meth:`sample` — integrate :math:`t: \varepsilon \to 1`.
    * :meth:`reflow_pairs` — generate the ``(z0, z1)`` couplings the next
      round trains on.
    * :meth:`straightness` — the measure :math:`S(\mathbf{Z})` that reflow
      is supposed to reduce.
    * :meth:`one_step` — :math:`z_0 + v(z_0, 0)`, exact once the flow is
      straight.
    * :meth:`log_prob` / :meth:`bits_per_dim` — integrate back, tracking
      the divergence.

    Parameters
    ----------
    config : RectifiedFlowConfig
        Architecture, schedule and solver settings.

    Attributes
    ----------
    field : nn.Module
        The velocity field itself.

    Notes
    -----
    Reference: Liu, Gong, and Liu, *"Flow Straight and Fast: Learning to
    Generate and Transfer Data with Rectified Flow"*, ICLR, 2023
    (arXiv:2209.03003).  Eq. (1) is the objective, eq. (3) the straightness
    measure, and Theorem 3.3 the marginal preservation that makes reflow
    legitimate.

    **On the coupling.**  ``rectified_flow_loss(x1)`` draws ``x0``
    independently, which is the 1-rectified flow of the paper.  Passing
    ``noise=`` supplies a *paired* ``x0`` instead and is what turns the
    same call into reflow — there is no separate objective.

    **On transfer.**  Nothing here requires the source to be Gaussian.
    The paper's other half — transporting between two image domains
    rather than from noise — is the same objective with ``x0`` drawn from
    the first domain instead of a prior, so it goes through
    :meth:`rectified_flow_loss`, :meth:`sample` and :meth:`reflow_pairs`
    unchanged by passing ``noise=``.  Only :meth:`log_prob` assumes a
    standard normal at ``t = 0``, and a likelihood is not defined for a
    domain-to-domain transport anyway.

    **Where to run it.**  Training is a plain convolutional forward and
    backward with no solve, so it belongs on ``"metal"``.  Generating
    reflow pairs is a solve per batch and is the expensive part of the
    method; the paper generates on the order of a million.

    **Not implemented, and why.**  Two settings the reference exposes are
    absent rather than overlooked:

    * **A perceptual reflow loss.**  The reference offers ``lpips`` and
      ``lpips+l2`` alongside ``l2`` for the distillation stage.  Both
      score images through a pre-trained external network, which Lucid's
      compute paths may not depend on, so only the squared error is
      available here.  It is the reference's own default.
    * **The stochastic sampler.**  The reference can add noise of
      magnitude ``(1 - t) * sigma_variance`` to turn the probability-flow
      ODE into an SDE.  Its default is ``0``, and the paper is stated and
      evaluated as an ODE method, so sampling here is deterministic given
      the starting point.

    Weight averaging (the reference's ``ema_rate``) is a property of a
    training loop rather than of a model, and is not part of any family
    in this zoo.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.rectified_flow import (
    ...     RectifiedFlowConfig, RectifiedFlowModel,
    ... )
    >>> cfg = RectifiedFlowConfig(sample_size=8, base_channels=16,
    ...                           channel_mult=(1, 2), num_res_blocks=1,
    ...                           attention_resolutions=())
    >>> model = RectifiedFlowModel(cfg).eval()
    >>> loss, _, _ = model.rectified_flow_loss(lucid.randn((2, 3, 8, 8)))
    >>> loss.shape
    ()
    >>> model.sample(n_samples=2, steps=4).shape
    (2, 3, 8, 8)
    """

    config_class: ClassVar[type[RectifiedFlowConfig]] = RectifiedFlowConfig
    base_model_prefix: ClassVar[str] = "rectified_flow"

    def __init__(self, config: RectifiedFlowConfig) -> None:
        super().__init__(config)
        self.field = _VelocityField(config)
        self.dynamics = _LikelihoodDynamics(self.field, config)

        size = config.sample_size
        height, width = size if isinstance(size, tuple) else (size, size)
        self._image_shape = (config.in_channels, height, width)
        self._input_dim = config.data_dim

        self._t_schedule = config.t_schedule
        self._time_eps = config.time_eps
        self._solver = config.solver
        self._rtol = config.rtol
        self._atol = config.atol
        self._trace_method = config.resolved_trace_method

    @property
    def input_dim(self) -> int:
        """int: Flattened width ``D`` of one sample."""
        return self._input_dim

    @property
    def t_schedule(self) -> TimeSchedule | int:
        """str or int: Which times the objective draws."""
        return self._t_schedule

    @property
    def trace_method(self) -> str:
        """str: How the divergence is obtained when scoring likelihood."""
        return self._trace_method

    @property
    def nfe(self) -> int:
        """int: Field evaluations spent by the most recent solve.

        A training step solves nothing and leaves the count where the last
        solve put it.  The number the paper is about: a straight flow needs
        one.
        """
        return self.dynamics.nfe

    def _check_image(self, x: Tensor, what: str = "Rectified Flow") -> None:
        check_image_shape(x, self._image_shape, what)

    def sample_times(self, batch: int, device: str | None = None) -> Tensor:
        r"""Draw the times the objective is evaluated at.

        Four behaviours, one per ``t_schedule``:

        * ``"uniform"`` — :math:`t \sim U[\varepsilon, 1]`, the objective
          itself and what reflow retrains under.
        * ``"t0"`` — every ``t`` at :math:`\varepsilon`.  Regressing there
          makes one Euler step a generator: one-step distillation.
        * ``"t1"`` — every ``t`` at 1, distilling the reverse map.
        * ``k`` — the ``k``-point grid a ``k``-step Euler sampler visits,
          for ``k``-step distillation.

        Parameters
        ----------
        batch : int
            How many times to draw.
        device : str, optional
            Where to allocate them.  Defaults to the device the model's
            parameters are on; an explicit value wins.

        Returns
        -------
        Tensor
            ``(batch,)`` times in ``[time_eps, 1]``.

        Examples
        --------
        >>> import dataclasses
        >>> from lucid.models.generative.rectified_flow import (
        ...     RectifiedFlowConfig, RectifiedFlowModel,
        ... )
        >>> cfg = RectifiedFlowConfig(sample_size=8, base_channels=16,
        ...                           channel_mult=(1, 2), num_res_blocks=1,
        ...                           attention_resolutions=())
        >>> model = RectifiedFlowModel(cfg).eval()
        >>> t = model.sample_times(6)
        >>> t.shape
        (6,)
        >>> bool(((t >= cfg.time_eps) & (t <= 1.0)).all().item())
        True
        >>> # k = 2: only the two times a two-step Euler sampler visits.
        >>> two_step = RectifiedFlowModel(dataclasses.replace(cfg, t_schedule=2))
        >>> sorted({round(v, 4) for v in two_step.sample_times(64).tolist()})
        [0.001, 0.5005]
        """
        device = resolve_generation_device(self, device)
        eps = self._time_eps
        span = 1.0 - eps
        schedule = self._t_schedule
        if schedule == "uniform":
            return lucid.rand((batch,), device=device) * span + eps
        if schedule == "t0":
            return lucid.full((batch,), eps, device=device)
        if schedule == "t1":
            return lucid.full((batch,), 1.0, device=device)
        # Every string case is handled above, so what is left is the k of
        # k-step distillation.
        steps = schedule
        index = (lucid.rand((batch,), device=device) * steps).floor()
        # ``floor`` of a uniform draw can land on ``steps`` itself when the
        # draw rounds up to 1.0; clamping keeps every time on the grid.
        index = lucid.minimum(index, lucid.full_like(index, steps - 1))
        return index * (span / steps) + eps

    def path_sample(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        r"""The point :math:`x_t = t x_1 + (1 - t) x_0` on the straight line.

        Parameters
        ----------
        x1 : Tensor
            ``(B, C, H, W)`` data samples.
        x0 : Tensor
            ``(B, C, H, W)`` samples from the source distribution.
        t : Tensor
            ``(B,)`` times in ``[0, 1]``.

        Returns
        -------
        Tensor
            ``(B, C, H, W)`` interpolation.  This is Flow Matching's
            optimal-transport path at :math:`\sigma_{\min} = 0`.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.rectified_flow import (
        ...     RectifiedFlowConfig, RectifiedFlowModel,
        ... )
        >>> cfg = RectifiedFlowConfig(sample_size=8, base_channels=16,
        ...                           channel_mult=(1, 2), num_res_blocks=1,
        ...                           attention_resolutions=())
        >>> model = RectifiedFlowModel(cfg).eval()
        >>> x1, x0 = lucid.randn((2, 3, 8, 8)), lucid.randn((2, 3, 8, 8))
        >>> model.path_sample(x1, x0, lucid.tensor([0.25, 0.75])).shape
        (2, 3, 8, 8)
        >>> bool(lucid.allclose(model.path_sample(x1, x0, lucid.zeros(2)), x0))
        True
        >>> bool(lucid.allclose(model.path_sample(x1, x0, lucid.ones(2)), x1))
        True
        >>> half = model.path_sample(x1, x0, lucid.full((2,), 0.5))
        >>> bool(lucid.allclose(half, (x0 + x1) / 2))  # the chord's midpoint
        True
        """
        t_ = t.reshape(-1, 1, 1, 1)
        return t_ * x1 + (1.0 - t_) * x0

    def conditional_target(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        r"""The velocity of that line, :math:`x_1 - x_0`.

        Independent of ``t`` — a straight line traversed at constant speed
        has one velocity — which is why ``t`` is accepted but unused.  It
        stays in the signature because the quantity *is* a function of
        time in general, and a caller comparing paths should not have to
        special-case this one.

        Parameters
        ----------
        x1, x0 : Tensor
            ``(B, C, H, W)`` endpoint samples.
        t : Tensor
            ``(B,)`` times, accepted for interface symmetry.

        Returns
        -------
        Tensor
            ``(B, C, H, W)`` regression target.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.rectified_flow import (
        ...     RectifiedFlowConfig, RectifiedFlowModel,
        ... )
        >>> cfg = RectifiedFlowConfig(sample_size=8, base_channels=16,
        ...                           channel_mult=(1, 2), num_res_blocks=1,
        ...                           attention_resolutions=())
        >>> model = RectifiedFlowModel(cfg).eval()
        >>> x1, x0 = lucid.randn((2, 3, 8, 8)), lucid.randn((2, 3, 8, 8))
        >>> target = model.conditional_target(x1, x0, lucid.tensor([0.2, 0.7]))
        >>> target.shape
        (2, 3, 8, 8)
        >>> chord = (model.path_sample(x1, x0, lucid.ones(2))
        ...          - model.path_sample(x1, x0, lucid.zeros(2)))
        >>> bool(lucid.allclose(target, chord))  # one velocity for the whole line
        True
        """
        del t
        return x1 - x0

    def rectified_flow_loss(
        self, x1: Tensor, noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""One optimisation step of eq. (1) — no ODE is solved.

        Parameters
        ----------
        x1 : Tensor
            ``(B, C, H, W)`` targets.  Data for the first flow; the
            *previous flow's samples* during reflow.
        noise : Tensor, optional
            ``(B, C, H, W)`` sources.  Omitted, they are drawn
            independently and this is the 1-rectified flow.  Supplied as
            the noise that *produced* ``x1``, the very same call becomes
            the reflow objective — the pairing is the whole difference.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(loss, prediction, target)``.

        Raises
        ------
        ValueError
            If the model is configured to distil but no pairing is given.
            Distillation regresses at a fixed time, which is only
            meaningful against the coupling a previous flow produced;
            independent draws there would train the field to predict the
            mean of the data, and would do so silently.

        Examples
        --------
        >>> import dataclasses
        >>> import lucid
        >>> from lucid.models.generative.rectified_flow import (
        ...     RectifiedFlowConfig, RectifiedFlowModel,
        ... )
        >>> cfg = RectifiedFlowConfig(sample_size=8, base_channels=16,
        ...                           channel_mult=(1, 2), num_res_blocks=1,
        ...                           attention_resolutions=())
        >>> model = RectifiedFlowModel(cfg).eval()
        >>> x1, z0 = lucid.randn((2, 3, 8, 8)), lucid.randn((2, 3, 8, 8))
        >>> loss, prediction, target = model.rectified_flow_loss(x1, noise=z0)
        >>> loss.shape, prediction.shape
        ((), (2, 3, 8, 8))
        >>> bool(lucid.allclose(target, x1 - z0))  # the pairing sets the target
        True
        >>> bool(lucid.allclose(loss, ((prediction - target) ** 2).mean()))
        True
        >>> distil = RectifiedFlowModel(dataclasses.replace(cfg, t_schedule="t0"))
        >>> distil.rectified_flow_loss(x1)  # a pinned time needs paired noise
        Traceback (most recent call last):
            ...
        ValueError: t_schedule='t0' pins the time, ...
        """
        self._check_image(x1)
        if noise is None:
            if self._config_distils():
                raise ValueError(
                    f"t_schedule={self._t_schedule!r} pins the time, which is "
                    f"a distillation objective and needs the (noise, sample) "
                    f"pairs of a trained flow; pass noise=, e.g. from "
                    f"reflow_pairs()"
                )
            noise = lucid.randn(x1.shape, dtype=x1.dtype, device=x1.device)
        else:
            self._check_image(noise, "the noise passed to Rectified Flow")
            if noise.shape != x1.shape:
                raise ValueError(
                    f"noise must be paired with x1 one-for-one, got shapes "
                    f"{tuple(noise.shape)} and {tuple(x1.shape)}"
                )

        batch = int(x1.shape[0])
        t = self.sample_times(batch, device=str(x1.device.type)).to(x1.dtype)
        x_t = self.path_sample(x1, noise, t)
        target = self.conditional_target(x1, noise, t)
        prediction = self.field(x_t, t)
        loss = ((prediction - target) ** 2).mean()
        return loss, prediction, target

    def _config_distils(self) -> bool:
        return self._t_schedule != "uniform"

    def _integrate(
        self, func: Callable[..., object], state: _StateT, t0: float, t1: float
    ) -> _StateT:
        """One adaptive solve, endpoint only."""
        return odeint_endpoint(
            func, state, [t0, t1], rtol=self._rtol, atol=self._atol, method=self._solver
        )

    @lucid.no_grad()
    def sample(
        self,
        n_samples: int = 1,
        *,
        steps: int | None = None,
        device: str | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        r"""Generate by integrating the field from noise to data.

        Parameters
        ----------
        n_samples : int, default=1
            How many samples to draw.  Ignored when ``noise`` is given.
        steps : int, optional
            Force a fixed-budget Euler solve with this many uniform steps.
            **This is the setting the method exists for**: a straight flow
            is integrated exactly by Euler, so quality at ``steps=1``
            measures straightness directly.  Left ``None``, an adaptive
            solve runs instead and spends whatever the tolerance demands.
        device : str, optional
            Where to allocate the prior draw.  Defaults to the device the
            model's parameters are on; an explicit value wins.
        noise : Tensor, optional
            ``(N, C, H, W)`` starting point in place of a fresh draw.
            Keeping it is what makes a reflow pair.

        Returns
        -------
        Tensor
            ``(N, C, H, W)`` samples at ``t = 1``.

        Notes
        -----
        Euler is used for the fixed-budget path rather than a higher-order
        method on purpose: the paper's claim is about the error a *single
        first-order step* makes, and a Runge–Kutta step would hide it by
        spending four evaluations per step.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.rectified_flow import (
        ...     RectifiedFlowConfig, RectifiedFlowModel,
        ... )
        >>> cfg = RectifiedFlowConfig(sample_size=8, base_channels=16,
        ...                           channel_mult=(1, 2), num_res_blocks=1,
        ...                           attention_resolutions=())
        >>> model = RectifiedFlowModel(cfg).eval()
        >>> model.sample(n_samples=2, steps=1).shape
        (2, 3, 8, 8)
        >>> model.nfe  # steps=1: a single field evaluation
        1
        >>> noise = lucid.randn((3, 3, 8, 8))
        >>> model.sample(noise=noise, steps=4).shape  # the batch follows noise
        (3, 3, 8, 8)
        >>> model.nfe  # Euler, so the budget is exactly the step count
        4
        """
        if noise is None:
            if n_samples <= 0:
                raise ValueError(f"n_samples must be positive, got {n_samples}")
            noise = start_noise(self, (n_samples, *self._image_shape), device)
        else:
            self._check_image(noise)

        batch = int(noise.shape[0])
        self.dynamics.reset()
        flat = noise.reshape(batch, -1)

        if steps is None:
            out = self._integrate(self.dynamics.velocity, flat, self._time_eps, 1.0)
        else:
            if steps <= 0:
                raise ValueError(f"steps must be positive, got {steps}")
            out = self._euler(flat, steps)
        return out.reshape(batch, *self._image_shape)

    def _euler(self, flat: Tensor, steps: int) -> Tensor:
        """Explicit Euler over ``steps`` uniform intervals.

        Written out rather than handed to the solver because the step size
        the paper uses is ``1/steps`` while the grid starts at
        ``time_eps`` — the last step therefore slightly overshoots ``t =
        1``, which is what the reference does and what the reported
        one-step numbers were produced with.
        """
        dt = 1.0 / steps
        span = 1.0 - self._time_eps
        x = flat
        for index in range(steps):
            t = lucid.tensor(
                index / steps * span + self._time_eps, dtype=flat.dtype, device=flat.device
            )
            x = x + self.dynamics.velocity(t, x) * dt
        return x

    @lucid.no_grad()
    def reflow_pairs(
        self,
        n_samples: int = 1,
        *,
        steps: int | None = None,
        device: str | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Couplings :math:`(Z_0, Z_1)` for the next rectified flow.

        Draws from the source, solves the current flow forward, and
        returns both ends.  Theorem 3.3 is what licenses feeding these
        back in: the solve preserves the marginals, so ``z1`` is still
        distributed as the data while the pair is no longer independent —
        and it is that dependence the next round exploits.

        Parameters
        ----------
        n_samples : int, default=1
            How many pairs to generate.  Ignored when ``noise`` is given.
        steps : int, optional
            Fixed Euler budget instead of an adaptive solve.
        device : str, optional
            Where to allocate the prior draw.  Defaults to the device the
            model's parameters are on; an explicit value wins.
        noise : Tensor, optional
            Sources to use in place of a fresh draw.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(z0, z1)``, both ``(N, C, H, W)`` — pass them straight to
            ``rectified_flow_loss(z1, noise=z0)``.

        Notes
        -----
        This is the expensive half of the method.  The paper generates on
        the order of a million pairs before each reflow round, and it is
        the only place a solve appears in training at all.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.rectified_flow import (
        ...     RectifiedFlowConfig, RectifiedFlowModel,
        ... )
        >>> cfg = RectifiedFlowConfig(sample_size=8, base_channels=16,
        ...                           channel_mult=(1, 2), num_res_blocks=1,
        ...                           attention_resolutions=())
        >>> model = RectifiedFlowModel(cfg).eval()
        >>> noise = lucid.randn((2, 3, 8, 8))
        >>> z0, z1 = model.reflow_pairs(noise=noise, steps=2)
        >>> z0 is noise, z1.shape
        (True, (2, 3, 8, 8))
        >>> bool(lucid.allclose(z1, model.sample(noise=z0, steps=2)))  # z0's endpoint
        True
        >>> _, _, target = model.rectified_flow_loss(z1, noise=z0)
        >>> bool(lucid.allclose(target, z1 - z0))  # the next round's target
        True
        """
        if noise is None:
            if n_samples <= 0:
                raise ValueError(f"n_samples must be positive, got {n_samples}")
            noise = start_noise(self, (n_samples, *self._image_shape), device)
        else:
            self._check_image(noise)
        return noise, self.sample(steps=steps, noise=noise)

    @lucid.no_grad()
    def one_step(self, noise: Tensor) -> Tensor:
        r"""The one-step map :math:`z_0 + v(z_0, \varepsilon)`.

        A single explicit Euler step across the whole interval.  Exact if
        the flow is straight, which is what reflow and then ``t0``
        distillation are for; on a 1-rectified flow it is a crude
        approximation and is expected to look it.

        Parameters
        ----------
        noise : Tensor
            ``(N, C, H, W)`` source samples.

        Returns
        -------
        Tensor
            ``(N, C, H, W)`` one-step generations.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> from lucid.models.generative.rectified_flow import (
        ...     RectifiedFlowConfig, RectifiedFlowModel,
        ... )
        >>> cfg = RectifiedFlowConfig(sample_size=8, base_channels=16,
        ...                           channel_mult=(1, 2), num_res_blocks=1,
        ...                           attention_resolutions=())
        >>> model = RectifiedFlowModel(cfg).eval()
        >>> # The output layer starts near zero; give it weights so the check bites.
        >>> _ = nn.init.normal_(model.field.conv_out.weight, std=0.05)
        >>> noise = lucid.randn((2, 3, 8, 8))
        >>> out = model.one_step(noise)
        >>> out.shape, model.nfe
        ((2, 3, 8, 8), 1)
        >>> bool(lucid.allclose(out, model.sample(noise=noise, steps=1)))
        True
        """
        self._check_image(noise)
        self.dynamics.reset()
        batch = int(noise.shape[0])
        flat = noise.reshape(batch, -1)
        t = lucid.tensor(self._time_eps, dtype=flat.dtype, device=flat.device)
        out = flat + self.dynamics.velocity(t, flat)
        return out.reshape(batch, *self._image_shape)

    @lucid.no_grad()
    def straightness(
        self,
        noise: Tensor | None = None,
        *,
        n_samples: int = 1,
        steps: int = 32,
        device: str | None = None,
    ) -> Tensor:
        r"""The measure :math:`S(\mathbf{Z})` of paper eq. (3).

        .. math::

            S(\mathbf{Z}) = \int_0^1 \mathbb{E}\Bigl[
                \bigl\| (Z_1 - Z_0) - \dot{Z}_t \bigr\|^2
            \Bigr]\,\mathrm{d}t

        Estimated by stepping the flow on a uniform grid, recording the
        field along the way, and comparing each velocity against the chord
        the trajectory actually spans.  Zero exactly when every path is a
        straight line at constant speed — and therefore exactly when one
        Euler step is enough.

        Parameters
        ----------
        noise : Tensor, optional
            ``(N, C, H, W)`` starting points.  Fixing them is the only way
            to compare two models fairly, since the measure is an
            expectation over the source.
        n_samples : int, default=1
            How many paths to average over when ``noise`` is not given.
        steps : int, default=32
            Grid resolution for both the solve and the quadrature.
        device : str, optional
            Where to draw the paths when ``noise`` is not given.  Defaults
            to the device the model's parameters are on; an explicit value
            wins.

        Returns
        -------
        Tensor
            Scalar estimate, in the sample's own squared units.

        Notes
        -----
        Reported per-dimension (a mean, not a sum), so the number is
        comparable across resolutions.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> from lucid.models.generative.rectified_flow import (
        ...     RectifiedFlowConfig, RectifiedFlowModel,
        ... )
        >>> cfg = RectifiedFlowConfig(sample_size=8, base_channels=16,
        ...                           channel_mult=(1, 2), num_res_blocks=1,
        ...                           attention_resolutions=())
        >>> model = RectifiedFlowModel(cfg).eval()
        >>> noise = lucid.randn((2, 3, 8, 8))
        >>> s = model.straightness(noise, steps=4)
        >>> s.shape, model.nfe
        ((), 4)
        >>> bool(s.item() < 1e-8)  # a fresh field barely moves, so nothing bends
        True
        >>> _ = nn.init.normal_(model.field.conv_out.weight, std=0.05)
        >>> bool(model.straightness(noise, steps=4).item() > 0.0)
        True
        """
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if noise is None:
            if n_samples <= 0:
                raise ValueError(f"n_samples must be positive, got {n_samples}")
            noise = start_noise(self, (n_samples, *self._image_shape), device)
        else:
            self._check_image(noise)

        self.dynamics.reset()
        batch = int(noise.shape[0])
        z0 = noise.reshape(batch, -1)
        dt = 1.0 / steps
        span = 1.0 - self._time_eps

        x = z0
        velocities: list[Tensor] = []
        for index in range(steps):
            t = lucid.tensor(
                index / steps * span + self._time_eps, dtype=z0.dtype, device=z0.device
            )
            v = self.dynamics.velocity(t, x)
            velocities.append(v)
            x = x + v * dt

        chord = x - z0
        # The trapezoid rule would be gratuitous here: the integrand is
        # sampled at the same grid the solve used, so a left Riemann sum
        # keeps the two consistent.
        total = velocities[0] * 0.0
        for v in velocities:
            total = total + (chord - v) ** 2
        return (total / steps).mean()

    def log_prob(self, x: Tensor) -> Tensor:
        r"""Per-sample :math:`\log p_1(x)` in nats.

        Integrates the field from ``t = 1`` back to ``t = time_eps``
        alongside the accumulated divergence, then scores the arrival
        against the standard normal the path starts from.

        Parameters
        ----------
        x : Tensor
            ``(B, C, H, W)`` data samples.

        Returns
        -------
        Tensor
            ``(B,)`` log-likelihood.

        Notes
        -----
        With ``trace_method`` resolving to ``"hutchinson"`` — which it does
        at any image size — this is an unbiased *estimate*, and two calls
        on the same input will not agree.

        The flow is only a normalizing flow while ``t_schedule`` is
        ``"uniform"``.  A distilled model is trained to be stepped a fixed
        number of times, not integrated, so its density is not the one it
        generates from; this method does not refuse it, but the number
        means less than it appears to.

        Examples
        --------
        >>> import math
        >>> import lucid
        >>> from lucid.models.generative.rectified_flow import (
        ...     RectifiedFlowConfig, RectifiedFlowModel,
        ... )
        >>> cfg = RectifiedFlowConfig(sample_size=8, base_channels=16,
        ...                           channel_mult=(1, 2), num_res_blocks=1,
        ...                           attention_resolutions=())
        >>> model = RectifiedFlowModel(cfg).eval()
        >>> x = lucid.randn((2, 3, 8, 8))
        >>> log_p = model.log_prob(x)
        >>> log_p.shape
        (2,)
        >>> # A fresh field is effectively zero, so x is scored as N(0, I) itself.
        >>> const = 0.5 * model.input_dim * math.log(2 * math.pi)
        >>> gauss = -0.5 * (x.reshape(2, -1) ** 2).sum(dim=-1) - const
        >>> bool(lucid.allclose(log_p, gauss, atol=1e-2))
        True
        """
        self._check_image(x)
        return cnf_gaussian_log_prob(
            self.dynamics,
            x,
            integrate=self._integrate,
            trace_method=self._trace_method,
            input_dim=self._input_dim,
            t0=self._time_eps,
        )

    def bits_per_dim(self, x: Tensor) -> Tensor:
        """Per-sample negative log-likelihood in bits per dimension."""
        return -self.log_prob(x) / (self._input_dim * math.log(2.0))

    @override
    def forward(self, sample: Tensor, t: Tensor) -> DiffusionModelOutput:
        """Evaluate the velocity field at ``(x, t)``."""
        return DiffusionModelOutput(sample=self.field(sample, t))


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper
# ─────────────────────────────────────────────────────────────────────────────


class RectifiedFlowForImageGeneration(ImageGenerationModel):
    r"""Rectified Flow with the training loss, reflow and ``.generate()``.

    ``forward(x)`` returns a :class:`DiffusionModelOutput` whose ``loss``
    is eq. (1).  ``forward(x, noise=...)`` is the reflow objective over a
    previous flow's couplings, which :meth:`reflow_pairs` produces.

    Parameters
    ----------
    config : RectifiedFlowConfig
        Architecture, schedule and solver settings.

    Attributes
    ----------
    rectified_flow : RectifiedFlowModel
        Underlying velocity field, exposing straightness and likelihood.

    Notes
    -----
    Reference: Liu, Gong, and Liu, *"Flow Straight and Fast"*, ICLR, 2023
    (arXiv:2209.03003).  Reported: a one-step FID of **4.85** on CIFAR-10,
    state of the art among one-step diffusion and flow models at
    publication.

    A full run is three stages, all through this one class:

    1. train with ``forward(x)`` — the 1-rectified flow;
    2. ``reflow_pairs()``, then train with ``forward(z1, noise=z0)``;
    3. optionally rebuild with ``t_schedule="t0"`` and train on the same
       pairs — one-step distillation.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.rectified_flow import (
    ...     RectifiedFlowConfig, RectifiedFlowForImageGeneration,
    ... )
    >>> cfg = RectifiedFlowConfig(sample_size=8, base_channels=16,
    ...                           channel_mult=(1, 2), num_res_blocks=1,
    ...                           attention_resolutions=())
    >>> model = RectifiedFlowForImageGeneration(cfg).eval()
    >>> model(lucid.randn((2, 3, 8, 8))).loss.shape
    ()
    >>> model.generate(n_samples=2, steps=1).samples.shape
    (2, 3, 8, 8)
    """

    config_class: ClassVar[type[RectifiedFlowConfig]] = RectifiedFlowConfig
    base_model_prefix: ClassVar[str] = "rectified_flow"

    def __init__(self, config: RectifiedFlowConfig) -> None:
        super().__init__(config)
        self.rectified_flow = RectifiedFlowModel(config)

    @property
    def nfe(self) -> int:
        """int: Field evaluations spent by the most recent solve."""
        return self.rectified_flow.nfe

    @override
    def forward(self, x: Tensor, noise: Tensor | None = None) -> DiffusionModelOutput:
        loss, prediction, _ = self.rectified_flow.rectified_flow_loss(x, noise)
        return DiffusionModelOutput(sample=prediction, loss=loss)

    @lucid.no_grad()
    def reflow_pairs(
        self,
        n_samples: int = 1,
        *,
        steps: int | None = None,
        device: str | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Couplings for the next round, from the wrapped flow.

        See :meth:`RectifiedFlowModel.reflow_pairs` for why these pairs are
        what the next round trains on.

        Parameters
        ----------
        n_samples : int, default=1
            How many pairs to generate.  Ignored when ``noise`` is given.
        steps : int, optional
            Fixed Euler budget instead of an adaptive solve.
        device : str, optional
            Where to allocate the prior draw.  Defaults to the device the
            model's parameters are on; an explicit value wins.
        noise : Tensor, optional
            Sources to use in place of a fresh draw.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(z0, z1)``, both ``(N, C, H, W)``.
        """
        device = resolve_generation_device(self, device)
        z0, z1 = self.rectified_flow.reflow_pairs(
            n_samples, steps=steps, device=device, noise=noise
        )
        assert isinstance(z0, Tensor) and isinstance(z1, Tensor)
        return z0, z1

    @lucid.no_grad()
    def generate(
        self,
        n_samples: int = 1,
        *,
        steps: int | None = None,
        device: str | None = None,
        noise: Tensor | None = None,
    ) -> GenerationOutput:
        """Sample by integrating the field from ``t = 0`` to ``t = 1``.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to draw.
        steps : int, optional
            Fixed Euler budget instead of an adaptive solve.  ``steps=1``
            is the regime the method exists for.
        device : str, optional
            Where to allocate the prior draw.  Defaults to the device the
            model's parameters are on; an explicit value wins.
        noise : Tensor, optional
            Starting point in place of a fresh draw.

        Returns
        -------
        GenerationOutput
            ``samples`` of shape ``(n_samples, C, H, W)``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.rectified_flow import (
        ...     RectifiedFlowConfig, RectifiedFlowForImageGeneration,
        ... )
        >>> cfg = RectifiedFlowConfig(sample_size=8, base_channels=16,
        ...                           channel_mult=(1, 2), num_res_blocks=1,
        ...                           attention_resolutions=())
        >>> model = RectifiedFlowForImageGeneration(cfg).eval()
        >>> model.generate(n_samples=3, steps=1).samples.shape
        (3, 3, 8, 8)
        >>> model.nfe  # one step, one field evaluation: the regime it is for
        1
        >>> noise = lucid.randn((1, 3, 8, 8))
        >>> model.generate(noise=noise, steps=2).samples.shape  # batch follows noise
        (1, 3, 8, 8)
        """
        device = resolve_generation_device(self, device)
        return GenerationOutput(
            samples=self.rectified_flow.sample(n_samples, steps=steps, device=device, noise=noise)
        )


__all__ = ["RectifiedFlowModel", "RectifiedFlowForImageGeneration"]
