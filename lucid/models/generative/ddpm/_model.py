"""DDPM model (Ho et al., 2020) — U-Net denoiser with timestep conditioning.

Architecture summary
--------------------
Time embedding:
    t → SinusoidalEmbedding(base_channels) → Linear(4·base_channels) → SiLU → Linear

Encoder (per stage l = 0 … L-1):
    L_l ResBlocks (in_ch → base_channels · channel_mult[l])
    optional AttentionBlock at the spatial resolutions in ``attention_resolutions``
    Downsample (stride-2 Conv) — skipped for the last stage

Middle:
    ResBlock → AttentionBlock → ResBlock

Decoder (per stage l = L-1 … 0):
    L_l + 1 ResBlocks consuming concatenated skip features
    optional AttentionBlock
    Upsample (nearest + Conv) — skipped for stage 0

Out:
    GroupNorm → SiLU → Conv 3×3 → ``out_channels_effective``

ResBlock conditioning recipe (Ho 2020 §3.2 + Appendix B):
    h = Conv₂(SiLU(Norm₂(SiLU(t·Linear)+ Conv₁(SiLU(Norm₁(x))))))
    out = h + Skip(x)
"""

import math
from dataclasses import dataclass
from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageGenerationModel
from lucid.models._mixins import DiffusionMixin
from lucid.models._output import DiffusionModelOutput
from lucid.models._utils._generative import (
    DiffusionPosterior,
    diffusion_posterior,
    diffusion_posterior_constants,
    diffusion_vlb_term,
    extract_into_tensor,
    make_beta_schedule,
    resolve_generation_device,
)
from lucid.models.generative._common._schedulers import DDPMScheduler
from lucid.models.generative.ddpm._config import DDPMConfig
from lucid.models._utils._common import host_floats

# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────


@final
class _ResBlock(nn.Module):
    """Residual block with timestep conditioning.

    Two GroupNorm + SiLU + Conv 3×3 sub-layers, with a per-channel additive
    timestep embedding injected between them.  Identity (or 1×1 Conv) skip.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        *,
        dropout: float = 0.0,
        groups: int = 32,
        scale_shift: bool = False,
    ) -> None:
        super().__init__()
        # GroupNorm requires channels % groups == 0; clamp ``groups`` down
        # when the user-supplied count exceeds the available channels.
        # ``min(groups, in_channels)`` is not necessarily a *divisor* of
        # in_channels — and the decoder's ResBlock input is ch + skip_ch,
        # a width the config never validates.  Walk down to the largest
        # divisor instead so GroupNorm can always be built.
        g_in = groups
        if in_channels % g_in != 0:
            g_in = next(
                (g for g in range(min(groups, in_channels), 0, -1) if in_channels % g == 0), 1
            )
        g_out = min(groups, out_channels) if out_channels % groups != 0 else groups

        # eps=1e-6 is the canonical DDPM GroupNorm epsilon (Ho 2020 TF code +
        # diffusers ``norm_eps``); required for checkpoint parity.
        self.norm1 = nn.GroupNorm(g_in, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self._scale_shift = scale_shift
        # FiLM needs two vectors per channel; additive needs one.
        self.time_proj = nn.Linear(time_emb_dim, out_channels * 2 if scale_shift else out_channels)
        self.norm2 = nn.GroupNorm(g_out, out_channels, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # The reference zero-initialises this projection so the block is
        # exactly the identity at step 0 — the standard diffusion U-Net
        # recipe, and what keeps a deep stack stable from the first step.
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        if in_channels != out_channels:
            self.skip: nn.Conv2d | nn.Identity = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    @override
    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # (B, out_ch) → (B, out_ch, 1, 1) so it broadcasts across spatial.
        t = self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        if self._scale_shift:
            # Improved-DDPM: h = norm(h) * (1 + scale) + shift.
            half = int(t.shape[1]) // 2
            scale, shift = t[:, :half], t[:, half:]
            h = self.norm2(h) * (1.0 + scale) + shift
        else:
            h = h + t
            h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


@final
class _AttentionBlock(nn.Module):
    """Self-attention on a 2-D feature map (Ho 2020 §3.2).

    GroupNorm → 1×1 Conv → split QKV → scaled-dot-product → 1×1 Conv proj.
    Output is added to the input (residual).
    """

    def __init__(self, channels: int, *, num_heads: int = 1, groups: int = 32) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        g = min(groups, channels) if channels % groups != 0 else groups
        self.norm = nn.GroupNorm(g, channels, eps=1e-6)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        # The reference zero-initialises this projection so the block is
        # exactly the identity at step 0 — the standard diffusion U-Net
        # recipe, and what keeps a deep stack stable from the first step.
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        N = int(H) * int(W)
        h = self.norm(x)
        qkv = self.qkv(h)  # (B, 3C, H, W)
        # Reshape to (B, 3, num_heads, head_dim, N) then permute heads to first.
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, H, N, D)
        q: Tensor = qkv[0]  # (B, H, N, D)
        k: Tensor = qkv[1]
        v: Tensor = qkv[2]
        scores = q @ k.permute(0, 1, 3, 2) * self.scale  # (B, H, N, N)
        attn = F.softmax(scores, dim=-1)
        out: Tensor = attn @ v  # (B, H, N, D)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class _Downsample(nn.Module):
    """Stride-2 3×3 conv with asymmetric (0,1,0,1) padding (Ho 2020 / diffusers).

    The canonical DDPM downsample pads only the right/bottom edges before a
    ``padding=0`` stride-2 conv (equivalent to TF ``SAME`` on even inputs),
    rather than a symmetric ``padding=1`` — required for checkpoint parity.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # F.pad order is (W_left, W_right, H_top, H_bottom).
        x = F.pad(x, (0, 1, 0, 1))
        return self.op(x)


@final
class _Upsample(nn.Module):
    """Nearest-neighbour ×2 + 3×3 conv (Ho 2020 §3.2)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_up = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x_up)


# ─────────────────────────────────────────────────────────────────────────────
# DDPM U-Net trunk
# ─────────────────────────────────────────────────────────────────────────────


class _DDPMUNet(nn.Module):
    r"""U-Net denoiser shared by every DDPM variant.

    Implements the architecture introduced by Ho, Jain, and Abbeel, 2020:
    a symmetric U-Net augmented with sinusoidal-timestep conditioning,
    GroupNorm-equipped residual blocks, and self-attention at low-resolution
    feature maps.  The encoder reduces spatial resolution by a factor of
    :math:`2^{L-1}` over :math:`L = \mathrm{len}(\text{channel\_mult})`
    stages; the middle block applies ResBlock → Attn → ResBlock at the
    bottleneck; the decoder mirrors the encoder with concatenated skip
    connections.  The output convolution emits ``out_channels_effective``
    channels — equal to ``in_channels`` (or ``2 * in_channels`` when
    ``learn_sigma`` is enabled for variance prediction).

    Spatial layout per stage :math:`l \in [0, L)`:
        * ``num_res_blocks`` ResBlocks (with optional attention at
          ``attention_resolutions``).
        * Downsample → next stage (skipped for ``l == L - 1``).

    Decoder mirrors the encoder with concatenated skip features; the extra
    ResBlock per stage (``num_res_blocks + 1`` total on the decoder side)
    consumes the post-downsample skip.

    Parameters
    ----------
    config : DDPMConfig
        Hyperparameters controlling ``base_channels``, ``channel_mult``,
        ``num_res_blocks``, ``attention_resolutions``, ``num_heads``,
        ``dropout``, ``resnet_groups``, and ``learn_sigma``.

    Attributes
    ----------
    time_mlp : nn.TimestepEmbedding
        Sinusoidal timestep embedding + 2-layer MLP producing the per-
        timestep conditioning vector of width ``4 * base_channels``.
    conv_in : nn.Conv2d
        Stem convolution lifting input channels to ``base_channels``.
    down_res, down_attn, down_sample : nn.ModuleList
        Encoder ResBlocks, optional AttentionBlocks, and Downsamples.
    mid_block1, mid_attn, mid_block2 : nn.Module
        Bottleneck stack at the lowest spatial resolution.
    up_res, up_attn, up_sample : nn.ModuleList
        Decoder ResBlocks, optional AttentionBlocks, and Upsamples.
    norm_out : nn.GroupNorm
        Output GroupNorm before the final convolution.
    conv_out : nn.Conv2d
        Final 3x3 convolution projecting to ``out_channels_effective``.

    Notes
    -----
    Reference: Ho, Jain, and Abbeel, *"Denoising Diffusion Probabilistic
    Models"*, NeurIPS, 2020 (arXiv:2006.11239); architecture details in
    Appendix B.  Improved-DDPM additions follow Nichol and Dhariwal, *
    "Improved Denoising Diffusion Probabilistic Models"*, ICML, 2021
    (arXiv:2102.09672).

    ResBlock conditioning recipe:

    .. math::

        h = \mathrm{Conv}_2\!\big(
            \mathrm{SiLU}(
                \mathrm{Norm}_2(
                    \mathrm{SiLU}(t \cdot W_t) + \mathrm{Conv}_1(
                        \mathrm{SiLU}(\mathrm{Norm}_1(x))
                    )
                )
            )
        \big)
        + \mathrm{Skip}(x).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ddpm import DDPMConfig
    >>> from lucid.models.generative.ddpm._model import _DDPMUNet
    >>> cfg = DDPMConfig(sample_size=32, base_channels=32,
    ...                  channel_mult=(1, 2), num_res_blocks=1,
    ...                  attention_resolutions=(16,), resnet_groups=16)
    >>> unet = _DDPMUNet(cfg).eval()
    >>> x = lucid.randn((1, 3, 32, 32))
    >>> t = lucid.tensor([42]).long()
    >>> out = unet(x, t)
    >>> out.shape   # (1, 3, 32, 32)
    (1, 3, 32, 32)
    """

    def __init__(self, config: DDPMConfig) -> None:
        super().__init__()
        self._config = config
        L = len(config.channel_mult)
        base = config.base_channels
        time_dim = base * 4

        # Ho 2020 / diffusers DDPM use the [sin, cos] ordering (flip_sin_to_cos
        # = False); required for checkpoint parity.
        self.time_mlp = nn.TimestepEmbedding(in_dim=base, out_dim=time_dim, flip_sin_to_cos=False)

        self.conv_in = nn.Conv2d(config.in_channels, base, 3, padding=1)

        # ── Encoder ─────────────────────────────────────────────────────────
        if isinstance(config.sample_size, tuple):
            h0 = config.sample_size[0]
        else:
            h0 = int(config.sample_size)
        spatial = h0  # use H

        self.down_res: nn.ModuleList[_ResBlock] = nn.ModuleList()
        self.down_attn: nn.ModuleList[_AttentionBlock | nn.Identity] = nn.ModuleList()
        self.down_sample: nn.ModuleList[_Downsample | nn.Identity] = nn.ModuleList()
        skip_channels: list[int] = [base]  # initial conv output
        ch = base
        for level, mult in enumerate(config.channel_mult):
            out_ch = base * mult
            for _ in range(config.num_res_blocks):
                self.down_res.append(
                    _ResBlock(
                        ch,
                        out_ch,
                        time_dim,
                        dropout=config.dropout,
                        groups=config.resnet_groups,
                        scale_shift=config.use_scale_shift_norm,
                    )
                )
                ch = out_ch
                if spatial in config.attention_resolutions:
                    self.down_attn.append(
                        _AttentionBlock(ch, num_heads=config.num_heads, groups=config.resnet_groups)
                    )
                else:
                    self.down_attn.append(nn.Identity())
                skip_channels.append(ch)
            if level != L - 1:
                self.down_sample.append(_Downsample(ch))
                skip_channels.append(ch)
                spatial //= 2
            else:
                self.down_sample.append(nn.Identity())

        # Cache the per-block stage so forward can decide when to downsample.
        # Encoder loop visits N = sum_l num_res_blocks ResBlocks; we apply
        # downsample once after each stage except the last.
        self._encoder_stages = L

        # ── Middle ──────────────────────────────────────────────────────────
        self.mid_block1 = _ResBlock(
            ch,
            ch,
            time_dim,
            dropout=config.dropout,
            groups=config.resnet_groups,
            scale_shift=config.use_scale_shift_norm,
        )
        self.mid_attn = _AttentionBlock(ch, num_heads=config.num_heads, groups=config.resnet_groups)
        self.mid_block2 = _ResBlock(
            ch,
            ch,
            time_dim,
            dropout=config.dropout,
            groups=config.resnet_groups,
            scale_shift=config.use_scale_shift_norm,
        )

        # ── Decoder ─────────────────────────────────────────────────────────
        self.up_res: nn.ModuleList[_ResBlock] = nn.ModuleList()
        self.up_attn: nn.ModuleList[_AttentionBlock | nn.Identity] = nn.ModuleList()
        self.up_sample: nn.ModuleList[_Upsample | nn.Identity] = nn.ModuleList()
        for level, mult in list(enumerate(config.channel_mult))[::-1]:
            out_ch = base * mult
            for _ in range(config.num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                self.up_res.append(
                    _ResBlock(
                        ch + skip_ch,
                        out_ch,
                        time_dim,
                        dropout=config.dropout,
                        groups=config.resnet_groups,
                        scale_shift=config.use_scale_shift_norm,
                    )
                )
                ch = out_ch
                if spatial in config.attention_resolutions:
                    self.up_attn.append(
                        _AttentionBlock(ch, num_heads=config.num_heads, groups=config.resnet_groups)
                    )
                else:
                    self.up_attn.append(nn.Identity())
            if level != 0:
                self.up_sample.append(_Upsample(ch))
                spatial *= 2
            else:
                self.up_sample.append(nn.Identity())

        # ── Out ─────────────────────────────────────────────────────────────
        g_out = (
            min(config.resnet_groups, ch)
            if ch % config.resnet_groups != 0
            else config.resnet_groups
        )
        self.norm_out = nn.GroupNorm(g_out, ch, eps=1e-6)
        self.conv_out = nn.Conv2d(ch, config.out_channels_effective, 3, padding=1)
        # The reference zero-initialises this projection so the block is
        # exactly the identity at step 0 — the standard diffusion U-Net
        # recipe, and what keeps a deep stack stable from the first step.
        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

        # Pre-computed "blocks per stage" for forward dispatch.
        self._blocks_per_down_stage = config.num_res_blocks
        # 1.0 disables the rescale entirely, so the common T=1000 case pays
        # nothing for the flag existing.
        self._timestep_scale = (
            1000.0 / float(config.num_train_timesteps) if config.rescale_timesteps else 1.0
        )
        self._blocks_per_up_stage = config.num_res_blocks + 1

    @override
    def forward(self, sample: Tensor, timestep: Tensor) -> Tensor:
        """Predict noise (or ``[noise, σ²]``) at the given timestep.

        Args:
            sample:   ``(B, in_channels, H, W)`` noisy image ``x_t``.
            timestep: ``(B,)`` or ``()`` int timestep tensor.  Scalar tensors
                are broadcast.

        Returns:
            ``(B, out_channels_effective, H, W)`` raw network output.
        """
        if timestep.ndim == 0:
            timestep = timestep.reshape((1,)).expand((int(sample.shape[0]),))
        if self._timestep_scale != 1.0:
            # Improved-DDPM's ``rescale_timesteps``: a T != 1000 model is fed
            # ``t * (1000 / T)`` so the sinusoidal embedding still sees the
            # magnitude range it was designed for.
            timestep = timestep * self._timestep_scale  # int → float32
        t_emb = self.time_mlp(timestep)  # (B, time_dim)

        h = self.conv_in(sample)
        skips: list[Tensor] = [h]

        # Encoder.
        L = self._encoder_stages
        blocks_per = self._blocks_per_down_stage
        idx = 0
        for level in range(L):
            for _ in range(blocks_per):
                h = self.down_res[idx](h, t_emb)
                h = self.down_attn[idx](h)
                skips.append(h)
                idx += 1
            if level != L - 1:
                h = self.down_sample[level](h)
                skips.append(h)
            # last stage: down_sample slot holds an Identity; do nothing.

        # Middle.
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder.
        blocks_up = self._blocks_per_up_stage
        idx = 0
        for level in range(L - 1, -1, -1):
            for _ in range(blocks_up):
                skip = skips.pop()
                h = lucid.cat([h, skip], dim=1)
                h = self.up_res[idx](h, t_emb)
                h = self.up_attn[idx](h)
                idx += 1
            if level != 0:
                # up_sample list has one entry per stage; index ``L - 1 - level``
                # is the stage's upsample (skip for level 0).
                h = self.up_sample[L - 1 - level](h)

        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out(h)


# ─────────────────────────────────────────────────────────────────────────────
# Bare DDPM model — wraps the U-Net, returns DiffusionModelOutput
# ─────────────────────────────────────────────────────────────────────────────


class DDPMModel(PretrainedModel, DiffusionMixin):
    r"""Bare DDPM U-Net trunk with the sampling-capable :class:`DiffusionMixin`.

    Returns a :class:`DiffusionModelOutput` whose ``sample`` field is the
    raw network output — interpreted as predicted noise :math:`\epsilon`,
    clean signal :math:`x_0`, or velocity :math:`v` depending on
    ``config.prediction_type``.  When ``config.learn_sigma`` is ``True``
    the output channels are ``2 * in_channels``: the first half is the
    mean prediction, the second half the (raw) variance prediction;
    downstream code splits the two if needed.

    Parameters
    ----------
    config : DDPMConfig
        Hyperparameters governing the U-Net topology and the diffusion
        noise schedule (``num_train_timesteps``, ``beta_start``,
        ``beta_end``, ``beta_schedule``, ``prediction_type``).

    Attributes
    ----------
    unet : _DDPMUNet
        Underlying U-Net denoiser (see :class:`_DDPMUNet`).
    config_class : type[DDPMConfig]
        Registry pointer for matching-config instantiation.
    base_model_prefix : str
        State-dict prefix (``"ddpm"``) under which the trunk is nested in
        task-head variants.

    Notes
    -----
    Reference: Ho, Jain, and Abbeel, *"Denoising Diffusion Probabilistic
    Models"*, NeurIPS, 2020 (arXiv:2006.11239).

    The forward Markov chain has the closed-form marginal

    .. math::

        q(x_t \mid x_0)
            = \mathcal{N}\!\big(
                x_t;\;
                \sqrt{\bar{\alpha}_t}\, x_0,\;
                (1 - \bar{\alpha}_t)\, \mathbf{I}
              \big),

    with :math:`\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)`.  The
    U-Net is trained to predict the injected noise so reverse sampling
    iterates :math:`t = T, T-1, \ldots, 1` via the standard ancestral
    Gaussian update.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ddpm import DDPMConfig, DDPMModel
    >>> cfg = DDPMConfig(sample_size=32, base_channels=32,
    ...                  channel_mult=(1, 2), num_res_blocks=1,
    ...                  resnet_groups=16, num_train_timesteps=100)
    >>> model = DDPMModel(cfg).eval()
    >>> x_t = lucid.randn((1, 3, 32, 32))
    >>> t = lucid.tensor([42]).long()
    >>> out = model(x_t, t)
    >>> out.sample.shape   # (1, 3, 32, 32) — predicted noise
    (1, 3, 32, 32)
    """

    config_class: ClassVar[type[DDPMConfig]] = DDPMConfig
    base_model_prefix: ClassVar[str] = "ddpm"

    def __init__(self, config: DDPMConfig) -> None:
        super().__init__(config)
        self.unet = _DDPMUNet(config)

    @override
    def forward(self, sample: Tensor, timestep: Tensor) -> DiffusionModelOutput:
        out = self.unet(sample, timestep)
        return DiffusionModelOutput(sample=out)


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper — training loss
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DDPMOutput(DiffusionModelOutput):
    r"""Training output of :class:`DDPMForImageGeneration`.

    Adds the two halves of Improved-DDPM's hybrid objective alongside the
    inherited total, so a training loop can watch them separately —
    :math:`L_{\text{vlb}}` is far noisier than :math:`L_{\text{simple}}` and
    a single blended number hides which one is moving.

    Attributes
    ----------
    loss_simple : Tensor or None, optional
        The Ho 2020 Eq. 14 MSE term on its own.
    loss_vlb : Tensor or None, optional
        The variational term :math:`L_{t-1}` at the sampled timesteps, in
        bits, *before* the :math:`\lambda T` scaling folded into ``loss``.
        ``None`` unless ``learn_sigma=True``.

    Examples
    --------
    >>> out = model(x_t, t, target=noise)      # doctest: +SKIP
    >>> float(out.loss_simple.item()) > 0      # doctest: +SKIP
    True
    """

    loss_simple: Tensor | None = None
    loss_vlb: Tensor | None = None


def _x0_recovery_constants(betas: Tensor) -> tuple[Tensor, Tensor]:
    r"""Constants for inverting :math:`x_t = \sqrt{\bar\alpha_t}x_0 +
    \sqrt{1 - \bar\alpha_t}\varepsilon`.

    Returns ``(sqrt(1/ᾱ_t), sqrt(1/ᾱ_t - 1))``, so that
    :math:`x_0 = \sqrt{1/\bar\alpha_t}\,x_t -
    \sqrt{1/\bar\alpha_t - 1}\,\varepsilon`.  Accumulated in Python floats
    for the same reason as :func:`diffusion_posterior_constants` — ``ᾱ_t``
    reaches ~4e-5 at ``t = T``, and its reciprocal square root magnifies
    whatever error it carries.
    """
    recip: list[float] = []
    recipm1: list[float] = []
    acc = 1.0
    for beta in host_floats(betas):
        acc *= 1.0 - beta
        recip.append(math.sqrt(1.0 / acc))
        recipm1.append(math.sqrt(1.0 / acc - 1.0))
    dev = betas.device.type
    return lucid.tensor(recip, device=dev), lucid.tensor(recipm1, device=dev)


class DDPMForImageGeneration(ImageGenerationModel, DiffusionMixin):
    r"""DDPM with the noise-prediction training loss and Markov-chain sampler.

    Wraps a :class:`_DDPMUNet` denoiser, supplies the Ho 2020 §3.2
    "simplified" training objective when ``target`` is provided, and inherits
    :meth:`DiffusionMixin.generate` for ancestral sampling over the configured
    noise schedule.  This is the standard entry point for training and
    inference on the DDPM family.

    Training
        * ``forward(sample, timestep)`` returns the raw network output.
        * ``forward(sample, timestep, target=...)`` (``target`` = ground-truth
          noise for ``epsilon`` parameterisation) additionally fills the
          ``loss`` field with the MSE loss of the paper's simple objective
          (Ho 2020 Eq. 14).
        * With ``config.learn_sigma=True`` the loss becomes Improved-DDPM's
          hybrid :math:`L_{\text{simple}} + \lambda L_{\text{vlb}}`, and
          ``loss_simple`` / ``loss_vlb`` report the two terms separately.
          Pass ``x_start=`` when you have the clean image — see
          :meth:`forward`.

    Sampling
        * Use :meth:`generate` from :class:`DiffusionMixin` — pass any
          ``Scheduler`` (``DDPMScheduler`` for ancestral sampling).

    Parameters
    ----------
    config : DDPMConfig
        Hyperparameters.  ``config.learn_sigma=True`` switches on the learned
        variance head and the hybrid objective; ``config.vlb_weight`` is the
        :math:`\lambda` that weights its variational term (default ``1e-3``,
        the value in Nichol & Dhariwal §4).

    Attributes
    ----------
    unet : _DDPMUNet
        Underlying U-Net denoiser (see :class:`_DDPMUNet`).

    Notes
    -----
    Reference: Ho, Jain, and Abbeel, *"Denoising Diffusion Probabilistic
    Models"*, NeurIPS, 2020 (arXiv:2006.11239); Improved-DDPM additions
    from Nichol and Dhariwal, ICML 2021 (arXiv:2102.09672).

    Training loss (when ``target`` is supplied):

    .. math::

        \mathcal{L}_{\text{simple}}(\theta)
            = \mathbb{E}_{t, x_0, \epsilon}\!\left[
                \big\lVert \epsilon - \epsilon_\theta(x_t, t)
                \big\rVert^2
              \right],

    where :math:`x_t = \sqrt{\bar{\alpha}_t} x_0 +
    \sqrt{1 - \bar{\alpha}_t}\, \epsilon`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ddpm import (
    ...     DDPMConfig, DDPMForImageGeneration,
    ... )
    >>> cfg = DDPMConfig(sample_size=32, base_channels=32,
    ...                  channel_mult=(1, 2), num_res_blocks=1,
    ...                  resnet_groups=16, num_train_timesteps=100)
    >>> model = DDPMForImageGeneration(cfg).eval()
    >>> x_t = lucid.randn((1, 3, 32, 32))
    >>> t = lucid.tensor([42]).long()
    >>> out = model(x_t, t)
    >>> out.sample.shape   # (1, 3, 32, 32)
    (1, 3, 32, 32)
    """

    config_class: ClassVar[type[DDPMConfig]] = DDPMConfig
    base_model_prefix: ClassVar[str] = "ddpm"

    # Forward-process constants of the hybrid objective, present only with
    # ``learn_sigma``.  Non-persistent buffers: they follow the model's
    # device, so the per-sample gather runs where the batch is, and stay
    # out of the checkpoint, being functions of the config alone.
    _log_betas: Tensor
    _sqrt_recip_ab: Tensor
    _sqrt_recipm1_ab: Tensor
    _posterior_mean_coef_start: Tensor
    _posterior_mean_coef_t: Tensor
    _posterior_log_variance: Tensor

    def __init__(self, config: DDPMConfig) -> None:
        super().__init__(config)
        self.unet = _DDPMUNet(config)
        # ``self.config`` is typed as the base ``ModelConfig``; keep the
        # narrow one so the schedule fields can be read without a cast.
        self._cfg = config
        self._in_channels = config.in_channels
        self._learn_sigma = config.learn_sigma
        self._prediction_type = config.prediction_type
        self._vlb_weight = config.vlb_weight
        self._num_train_timesteps = config.num_train_timesteps

        if config.learn_sigma:
            # The bound needs the forward-process schedule, which the U-Net
            # itself never sees.  Build it once — the constants depend only
            # on ``betas``, not on any parameter.
            betas = make_beta_schedule(
                config.num_train_timesteps,
                config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
            )
            posterior = diffusion_posterior_constants(betas)
            sqrt_recip_ab, sqrt_recipm1_ab = _x0_recovery_constants(betas)
            constants = {
                "_log_betas": lucid.log(betas),
                "_sqrt_recip_ab": sqrt_recip_ab,
                "_sqrt_recipm1_ab": sqrt_recipm1_ab,
                "_posterior_mean_coef_start": posterior.mean_coef_start,
                "_posterior_mean_coef_t": posterior.mean_coef_t,
                "_posterior_log_variance": posterior.log_variance,
            }
            for name, value in constants.items():
                self.register_buffer(name, value, persistent=False)

    @property
    def _posterior(self) -> DiffusionPosterior:
        """The posterior constants, on whatever device the model is."""
        return DiffusionPosterior(
            mean_coef_start=self._posterior_mean_coef_start,
            mean_coef_t=self._posterior_mean_coef_t,
            log_variance=self._posterior_log_variance,
        )

    def make_scheduler(self) -> DDPMScheduler:
        r"""The ancestral sampler this model's own config describes.

        :meth:`DiffusionMixin.generate` takes any scheduler, which leaves
        the caller to retype the schedule::

            scheduler = DDPMScheduler(
                config.num_train_timesteps,
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                prediction_type=config.prediction_type,
            )

        Every one of those is already on the config, and a scheduler
        built with a different schedule from the one the weights were
        trained under produces plausible-looking noise rather than an
        error.  ``config.clip_denoised`` was not reachable at all: the
        field existed, the sampler honoured its own copy, and nothing
        carried one to the other.

        Returns
        -------
        DDPMScheduler
            Configured from ``self.config``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.ddpm import (
        ...     DDPMConfig, DDPMForImageGeneration,
        ... )
        >>> cfg = DDPMConfig(sample_size=32, base_channels=32,
        ...                  channel_mult=(1, 2), num_res_blocks=1,
        ...                  resnet_groups=16, num_train_timesteps=100)
        >>> model = DDPMForImageGeneration(cfg).eval()
        >>> scheduler = model.make_scheduler()
        >>> scheduler.num_train_timesteps, scheduler.clip_denoised
        (100, True)
        """
        config = self._cfg
        return DDPMScheduler(
            config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_denoised=config.clip_denoised,
            # Built where the model is, so sampling a Metal model does not
            # start by moving the schedule across.
            device=resolve_generation_device(self, None),
        )

    def _split_output(self, raw: Tensor) -> tuple[Tensor, Tensor | None]:
        """When ``learn_sigma=True`` the network emits ``2·in_channels`` —
        split into ``(mean_pred, log_variance_pred)``."""
        if not self._learn_sigma:
            return raw, None
        C = int(raw.shape[1])
        half = C // 2
        return raw[:, :half], raw[:, half:]

    def _to_x_start(self, prediction: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Invert the parameterisation to recover :math:`x_0`.

        The bound is written in terms of the clean signal whichever way the
        network is parameterised, so ``epsilon`` and ``v`` predictions get
        converted; a ``sample`` prediction already *is* :math:`x_0`.
        """
        if self._prediction_type == "sample":
            return prediction

        shape = tuple(int(s) for s in x_t.shape)
        sqrt_recip = extract_into_tensor(self._sqrt_recip_ab, t, shape)
        sqrt_recipm1 = extract_into_tensor(self._sqrt_recipm1_ab, t, shape)
        if self._prediction_type == "epsilon":
            return sqrt_recip * x_t - sqrt_recipm1 * prediction

        # ``v = sqrt(ᾱ)·ε - sqrt(1-ᾱ)·x_0``  ⟹  ``x_0 = sqrt(ᾱ)·x_t - sqrt(1-ᾱ)·v``
        sqrt_ab = 1.0 / sqrt_recip
        sqrt_one_minus_ab = sqrt_recipm1 * sqrt_ab
        return sqrt_ab * x_t - sqrt_one_minus_ab * prediction

    def _learned_log_variance(self, raw_logvar: Tensor, t: Tensor) -> Tensor:
        r"""Map the raw variance channels onto :math:`[\log\tilde\beta_t,
        \log\beta_t]`.

        Improved-DDPM §3.1 does not have the network emit a log-variance
        directly — the useful range spans several orders of magnitude and is
        strongly ``t``-dependent.  Instead the output ``v`` interpolates
        between the two natural bounds,
        :math:`\Sigma_\theta = \exp(v\log\beta_t + (1-v)\log\tilde\beta_t)`,
        which keeps the prediction inside a range that is always sane.
        """
        shape = tuple(int(s) for s in raw_logvar.shape)
        min_log = extract_into_tensor(self._posterior.log_variance, t, shape)
        max_log = extract_into_tensor(self._log_betas, t, shape)
        # ``v`` arrives in [-1, 1] from the network; rescale to [0, 1].
        frac = (raw_logvar + 1.0) / 2.0
        return frac * max_log + (1.0 - frac) * min_log

    @override
    def forward(
        self,
        sample: Tensor,
        timestep: Tensor,
        target: Tensor | None = None,
        x_start: Tensor | None = None,
    ) -> DDPMOutput:
        raw = self.unet(sample, timestep)
        mean_pred, logvar_pred = self._split_output(raw)

        if target is None:
            return DDPMOutput(sample=mean_pred)

        # Ho 2020 §3.2 "simple" objective — MSE on ε (or x_0 / v, depending
        # on ``prediction_type``; the caller decides what's in ``target``).
        loss_simple = ((mean_pred - target) ** 2).mean()
        if logvar_pred is None:
            return DDPMOutput(sample=mean_pred, loss=loss_simple)

        if x_start is None:
            # ``target`` pins down x_0 exactly given x_t and t, so the caller
            # need not pass it — but recovering it divides by sqrt(ᾱ_t),
            # which is ~6e-3 at t=T, amplifying rounding error 160×.  Pass
            # ``x_start`` explicitly when you have it.
            x_start = self._to_x_start(target, sample, timestep).detach()

        model_log_variance = self._learned_log_variance(logvar_pred, timestep)
        # Freeze the mean inside the bound.  L_vlb is a far noisier signal
        # than L_simple, and Nichol & Dhariwal §4 found that letting it reach
        # the mean destabilises training — the variational term is there to
        # train the *variance* head and nothing else.
        model_mean, _ = diffusion_posterior(
            x_start=self._to_x_start(mean_pred.detach(), sample, timestep),
            x_t=sample,
            t=timestep,
            posterior=self._posterior,
        )
        vlb = diffusion_vlb_term(
            x_start=x_start,
            x_t=sample,
            t=timestep,
            model_mean=model_mean,
            model_log_variance=model_log_variance,
            posterior=self._posterior,
        ).mean()

        # ``vlb`` is one term L_t at a uniformly drawn t, so its expectation
        # is L_vlb/T; scaling by λ·T recovers the paper's λ·L_vlb.
        loss = loss_simple + (self._vlb_weight * self._num_train_timesteps) * vlb
        return DDPMOutput(sample=mean_pred, loss=loss, loss_simple=loss_simple, loss_vlb=vlb)
