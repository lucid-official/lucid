r"""DIAMOND — a world model that predicts frames instead of latents.

Three networks, trained separately, sharing nothing.

The **denoiser** answers the only question a dynamics model has to
answer: given the last :math:`L` frames and the actions taken, what does
the next frame look like?  It is a plain U-Net 2D wrapped in EDM's
preconditioners, so the thing that varies with the noise level is the
*target*, not the architecture.

The **reward/termination model** answers what an image cannot: scalar
reward and whether the episode ended.  Both are sequence predictions
under partial observability, so it is convolutional blocks feeding an
LSTM rather than a second diffusion model.

The **actor-critic** is trained entirely on frames the denoiser
imagined, never on real experience.

Keeping them apart is the paper's design, not an implementation
convenience: the diffusion loss is a reconstruction in pixel space, the
reward loss is cross-entropy on three classes, and the policy loss is
REINFORCE.  Nothing would be shared by fusing them but a checkpoint.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, Literal, overload, override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
import lucid.nn.functional as F
from lucid._dtype import _float32_or_wider
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import ModelOutput
from lucid.models._tasks import WorldModelingModel
from lucid.models._utils._generative import model_dtype
from lucid.models.generative._common._returns import lambda_return
from lucid.models.generative.diamond._config import DIAMONDConfig

__all__ = ["DIAMONDModel", "DIAMONDForWorldModeling", "DIAMONDOutput", "DIAMONDBehaviorOutput"]


def _groups(channels: int) -> int:
    """Group count for a normalisation over ``channels``.

    ``max(1, channels // 32)`` — 64 channels normalise in two groups, 32
    in one.  ⚠️ The paper says only "a group normalization layer"; this
    rule is read off the released implementation, and it is the sort of
    choice that changes activations without changing a single shape.
    """
    return max(1, channels // 32)


class _GroupNorm(nn.Module):
    """Plain group norm, wrapped so its parameter path matches the norms
    that *are* conditioned.

    Notes
    -----
    Both kinds appear in the released checkpoint and they are not
    interchangeable: the residual blocks' norms take their scale and
    shift from the conditioning vector, while the output norm and the
    attention's norm carry their own affine.
    """

    def __init__(self, channels: int) -> None:
        """Initialise the layer.

        Parameters
        ----------
        channels : int
            Channels being normalised.
        """
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Normalise ``(B, C, H, W)``."""
        return self.norm(x)


class _AdaGroupNorm(nn.Module):
    """Group norm whose scale and shift are regressed from a vector.

    Parameters
    ----------
    channels : int
        Channels being normalised.
    cond_dim : int
        Width of the conditioning vector.

    Notes
    -----
    Reference: Zheng et al., *"Learning Semantic-Aware Normalization for
    Generative Adversarial Networks"*, NeurIPS, 2020 — the mechanism
    DIAMOND admits both the action history and the diffusion time
    through.

    The normalisation itself is affine-free: an affine that the
    conditioning immediately overrides would be two parameters per
    channel doing nothing.
    """

    def __init__(self, channels: int, cond_dim: int) -> None:
        """Initialise the layer. See the class docstring for parameters."""
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels, affine=False)
        self.linear = nn.Linear(cond_dim, 2 * channels)

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Normalise ``(B, C, H, W)`` and modulate it by ``(B, cond_dim)``."""
        params = self.linear(cond)
        channels = int(x.shape[1])
        scale = params[:, :channels].reshape(-1, channels, 1, 1)
        shift = params[:, channels:].reshape(-1, channels, 1, 1)
        return self.norm(x) * (1.0 + scale) + shift


class _SelfAttention2d(nn.Module):
    """Multi-head self-attention over a feature map.

    Parameters
    ----------
    channels : int
        Feature width.
    head_dim : int, default=8
        Channels per head.

    Notes
    -----
    ⚠️ The paper's configuration table lists no attention, and the
    released Atari config sets ``attn_depths: [0,0,0,0]`` — yet the
    checkpoint carries attention weights in the U-Net's middle blocks
    and in the reward encoder's last stage.  Those depths govern the
    *resolution* stages; the middle is separate and always attends.
    """

    def __init__(self, channels: int, head_dim: int = 8) -> None:
        """Initialise the layer. See the class docstring for parameters."""
        super().__init__()
        self.heads = max(1, channels // head_dim)
        self.norm = _GroupNorm(channels)
        self.qkv_proj = nn.Conv2d(channels, channels * 3, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        # Zero-init, so an attention block starts as exactly its group
        # norm and contributes nothing until it has learnt to.  It is
        # also what makes the residual's source observable: at init the
        # block must return norm(x), never x.
        init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            init.zeros_(self.out_proj.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Attend over the spatial positions of ``(B, C, H, W)``."""
        batch, channels = int(x.shape[0]), int(x.shape[1])
        height, width = int(x.shape[2]), int(x.shape[3])
        # The residual leaves from the *normalised* stream, not the raw
        # input — the released block reassigns ``x = norm(x)`` before
        # using it.  ``out_proj`` is zero-init, so at initialisation an
        # attention block is exactly its group norm, and the trained
        # weights are fitted to a stream that gets renormalised here.
        # Adding to raw ``x`` instead leaves the denoiser blind to its
        # image input: F_theta pins at a constant for every sigma.
        normed = self.norm(x)
        qkv = self.qkv_proj(normed)
        qkv = qkv.reshape(batch, 3, self.heads, channels // self.heads, height * width)
        query = qkv[:, 0].permute(0, 1, 3, 2)
        key = qkv[:, 1].permute(0, 1, 3, 2)
        value = qkv[:, 2].permute(0, 1, 3, 2)
        scale = float(channels // self.heads) ** -0.5
        scores = lucid.softmax(query @ key.permute(0, 1, 3, 2) * scale, dim=-1)
        out = (scores @ value).permute(0, 1, 3, 2)
        out = out.reshape(batch, channels, height, width)
        return normed + self.out_proj(out)


def _quantize(x: Tensor) -> Tensor:
    """Clamp to ``[-1, 1]`` and round-trip through 8 bits.

    Every frame the released sampler feeds back into itself has been
    through this.  A rollout is autoregressive, so a denoised frame
    becomes conditioning for the next one — and the network was only
    ever conditioned on frames that came out of a byte.  Letting the
    estimate keep its full float range hands it inputs it never saw.

    Truncation rather than rounding, which is what casting to a byte
    does in the released code.

    Parameters
    ----------
    x : Tensor
        The denoised estimate.

    Returns
    -------
    Tensor
        The same shape, on the 256-value grid of ``[-1, 1]``.
    """
    clamped = lucid.clip(x, -1.0, 1.0)
    return lucid.floor((clamped + 1.0) / 2.0 * 255.0) / 255.0 * 2.0 - 1.0


class _ResBlock(nn.Module):
    """Two conditioned norm-activation-convolution stacks, plus attention.

    Parameters
    ----------
    in_channels, out_channels : int
        Channel widths.  When they differ a 1x1 ``proj`` brings the skip
        along — which is every decoder block, since each concatenates an
        encoder skip onto its input.
    cond_dim : int
        Width of the conditioning vector.
    attention : bool, default=False
        Whether the block attends after its convolutions.
    head_dim : int, default=8
        Channels per attention head.

    Notes
    -----
    Reference: Alonso et al., arXiv:2405.12399, Appendix D — "a group
    normalization layer, a SiLU activation, and a 3x3 convolution with
    stride 1 and padding 1".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        attention: bool = False,
        head_dim: int = 8,
    ) -> None:
        """Initialise the block. See the class docstring for parameters."""
        super().__init__()
        self.proj: nn.Conv2d | None = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.norm1 = _AdaGroupNorm(in_channels, cond_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = _AdaGroupNorm(out_channels, cond_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        # The second convolution of a residual block starts at zero too,
        # so the block is the identity on its skip at initialisation.
        init.zeros_(self.conv2.weight)
        self.attn: _SelfAttention2d | None = (
            _SelfAttention2d(out_channels, head_dim) if attention else None
        )
        self.act = nn.SiLU()

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply the block to ``(B, C, H, W)`` under ``cond``."""
        residual = x if self.proj is None else self.proj(x)
        h = self.norm1(x, cond)
        h = self.conv1(self.act(h))
        h = self.norm2(h, cond)
        h = self.conv2(self.act(h))
        out = residual + h
        return out if self.attn is None else self.attn(out)


class _ResBlocks(nn.Module):
    """A run of residual blocks that all see the same conditioning."""

    def __init__(self, blocks: list[_ResBlock]) -> None:
        """Initialise the run.

        Parameters
        ----------
        blocks : list of _ResBlock
            The blocks, in order.
        """
        super().__init__()
        self.resblocks = nn.ModuleList(list(blocks))

    def each(self) -> list[_ResBlock]:
        """The blocks, typed, for a caller that manages skips itself."""
        return [b for b in self.resblocks]

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Run every block in order."""
        for block in self.each():
            x = block(x, cond)
        return x


class _Downsample(nn.Module):
    """Halve the resolution with a stride-2 convolution."""

    def __init__(self, channels: int) -> None:
        """Initialise the layer.

        Parameters
        ----------
        channels : int
            Width, unchanged by the operation.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        init.orthogonal_(self.conv.weight)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Halve ``(B, C, H, W)``."""
        return self.conv(x)


class _Upsample(nn.Module):
    """Double the resolution, then convolve.

    Notes
    -----
    Nearest-neighbour interpolation followed by a 3x3 convolution, which
    is the usual way to avoid the checkerboard a transposed convolution
    leaves.

    The target size is passed in rather than assumed.  A stride-2
    convolution rounds up, so 15 becomes 8 going down and doubling would
    give 16 coming back — one row too many.  Interpolating to the skip's
    own size is exact at any resolution, and is what lets the CS:GO model
    work at 30x56.
    """

    def __init__(self, channels: int) -> None:
        """Initialise the layer.

        Parameters
        ----------
        channels : int
            Width, unchanged by the operation.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Double ``(B, C, H, W)`` by nearest sampling, then convolve.

        A plain factor of two, not a resize onto the skip's size: the
        U-Net pads its input up to a multiple of the total stride, so
        every resolution halves and doubles exactly and the two always
        meet.  Resizing to the skip instead lands on pixel alignments
        the network never saw in training.
        """
        up = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(up)


class _FourierFeatures(nn.Module):
    r"""Random-frequency embedding of a scalar.

    Parameters
    ----------
    width : int
        Output width; half of it is cosines and half sines, so the
        learned frequency vector is ``width // 2`` wide.

    Notes
    -----
    ⚠️ The paper does not say how the diffusion time is embedded.  The
    checkpoint's ``noise_emb.weight`` is ``(1, cond/2)``, which is
    Fourier features rather than the fixed sinusoidal ladder a
    transformer would use — the frequencies are **learned**.
    """

    def __init__(self, width: int) -> None:
        """Initialise the layer. See the class docstring for parameters."""
        super().__init__()
        self.weight = nn.Parameter(lucid.randn((1, width // 2)))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Embed ``(B,)`` into ``(B, width)``."""
        args = x.reshape(-1, 1) * self.weight * (2.0 * math.pi)
        return lucid.cat([lucid.cos(args), lucid.sin(args)], dim=-1)


class _UNet(nn.Module):
    """The U-Net the denoiser's vector field runs on.

    Parameters
    ----------
    cond_dim : int
        Conditioning width.
    channels, layers : tuple of int
        Per-resolution widths and block counts.
    head_dim : int
        Channels per attention head in the middle blocks.

    Notes
    -----
    The decoder has one more block per resolution than the encoder,
    because the encoder pushes a skip for its input as well as for each
    of its blocks.  Getting that wrong builds a working network with
    visibly fewer parameters.
    """

    def __init__(
        self,
        cond_dim: int,
        channels: tuple[int, ...],
        layers: tuple[int, ...],
        head_dim: int,
        attn: tuple[int, ...] | None = None,
    ) -> None:
        """Initialise the network. See the class docstring for parameters."""
        super().__init__()
        attn = (0,) * len(channels) if attn is None else attn
        self.d_blocks: nn.ModuleList[_ResBlocks] = nn.ModuleList()
        self.downsamples: nn.ModuleList[_Downsample | nn.Identity] = nn.ModuleList()
        width = channels[0]
        for stage, (out_ch, count) in enumerate(zip(channels, layers, strict=True)):
            self.downsamples.append(nn.Identity() if stage == 0 else _Downsample(width))
            self.d_blocks.append(
                _ResBlocks(
                    [
                        _ResBlock(
                            width if i == 0 else out_ch,
                            out_ch,
                            cond_dim,
                            bool(attn[stage]),
                            head_dim,
                        )
                        for i in range(count)
                    ]
                )
            )
            width = out_ch

        self.mid_blocks = _ResBlocks(
            [_ResBlock(width, width, cond_dim, True, head_dim) for _ in range(2)]
        )

        # The decoder concatenates whatever the encoder pushed, and at a
        # resolution boundary that is the *previous* stage's width rather
        # than this one's.  Recording the pushes is the only way to size
        # the blocks when the widths differ per stage, as CS:GO's do.
        pushes: list[int] = [channels[0]]
        width = channels[0]
        for stage, (out_ch, count) in enumerate(zip(channels, layers, strict=True)):
            if stage > 0:
                pushes.append(width)
            pushes.extend([out_ch] * count)
            width = out_ch

        self.u_blocks: nn.ModuleList[_ResBlocks] = nn.ModuleList()
        self.upsamples: nn.ModuleList[_Upsample | nn.Identity] = nn.ModuleList()
        cursor = len(pushes)
        for stage in reversed(range(len(channels))):
            out_ch = channels[stage]
            self.upsamples.append(nn.Identity() if stage == len(channels) - 1 else _Upsample(width))
            # The last block of each decoder resolution steps down to the
            # next one's width, so the upsample that follows already works
            # at the shallower size.  Holding the width for the whole
            # stage instead builds a wider network — 43M wider, at CS:GO's
            # channel counts — that still runs and still trains.
            shallower = channels[max(stage - 1, 0)]
            blocks = []
            count = layers[stage] + 1
            for index in range(count):
                cursor -= 1
                target = shallower if index == count - 1 else out_ch
                blocks.append(
                    _ResBlock(width + pushes[cursor], target, cond_dim, bool(attn[stage]), head_dim)
                )
                width = target
            self.u_blocks.append(_ResBlocks(blocks))

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Map ``(B, C, H, W)`` through the U and back.

        The skips are one flat stack, not one list per resolution: the
        encoder pushes its input, then every block's output and every
        downsample's, and the decoder pops exactly one per block.  The
        counts have to agree — twelve either way at the paper's depths —
        and they only do because each decoder resolution has one block
        more than its encoder counterpart.
        """
        # Pad up to a multiple of the total stride so every resolution
        # halves and doubles exactly, then crop back.  30 becomes 32
        # here; without it a decoder stage has to resize onto its skip,
        # which puts the features on a grid training never produced.
        height, width = int(x.shape[2]), int(x.shape[3])
        stride = 2 ** (len(self.d_blocks) - 1)
        pad_h = -height % stride
        pad_w = -width % stride
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        skips: list[Tensor] = [x]
        for stage, blocks in enumerate(self.d_blocks):
            if stage > 0:
                x = self.downsamples[stage](x)
                skips.append(x)
            for block in blocks.each():
                x = block(x, cond)
                skips.append(x)

        x = self.mid_blocks(x, cond)

        for index, blocks in enumerate(self.u_blocks):
            if index > 0:
                x = self.upsamples[index](x)
            for block in blocks.each():
                x = lucid.cat([x, skips.pop()], dim=1)
                x = block(x, cond)
        if pad_h or pad_w:
            x = x[..., :height, :width]
        return x


class _Denoiser(nn.Module):
    r"""The network EDM's preconditioners wrap.

    Parameters
    ----------
    config : DIAMONDConfig
        The variant to build.

    Notes
    -----
    Reference: Alonso et al., arXiv:2405.12399, Section 3.1 and Table 2,
    with the conditioning path taken from the released checkpoint — the
    paper says only that actions and diffusion time enter through
    adaptive group normalisation, not how they are embedded first.

    The past frames arrive already concatenated to the noised frame, so
    the image path is an ordinary U-Net that has no idea it is
    conditioned on anything.
    """

    def __init__(self, config: DIAMONDConfig) -> None:
        """Initialise the network. See the class docstring for parameters."""
        super().__init__()
        self.config = config
        cond_dim = config.cond_dim
        self.noise_emb = _FourierFeatures(cond_dim)
        # When the history is noised too, its level is a second thing the
        # network has to be told — the same frame at two different
        # degradations is two different conditionings.  Absent from the
        # Atari checkpoint, present in CS:GO's, exactly tracking
        # ``noise_previous_obs``.
        self.noise_cond_emb: _FourierFeatures | None = (
            _FourierFeatures(cond_dim) if config.noise_previous_obs else None
        )
        # One embedding per conditioning step, flattened back to
        # ``cond_dim`` — so the width per action falls as the history
        # grows, rather than the conditioning vector growing with it.
        self.action_embed = nn.Embedding(config.num_actions, cond_dim // config.conditioning_frames)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.conv_in = nn.Conv2d(config.denoiser_in_channels, config.unet_channels[0], 3, padding=1)
        self.unet = _UNet(
            cond_dim,
            config.unet_channels,
            config.unet_layers,
            config.attention_head_dim,
            config.attn_depths,
        )
        self.norm_out = _GroupNorm(config.unet_channels[0])
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(config.unet_channels[0], config.out_channels, 3, padding=1)
        # The output convolution starts at zero, so an untrained model
        # predicts the skip exactly rather than noise.
        init.zeros_(self.conv_out.weight)

    def conditioning(
        self, c_noise: Tensor, actions: Tensor, cond_noise: Tensor | None = None
    ) -> Tensor:
        """Fuse the noise level and the action history into one vector.

        Parameters
        ----------
        c_noise : Tensor
            ``(B,)`` preconditioned noise level of the frame being denoised.
        actions : Tensor
            Either ``(B, L)`` action indices, or ``(B, L, num_actions)``
            multi-hot rows for an action set whose components combine.
        cond_noise : Tensor or None, optional
            ``(B,)`` level the *history* was noised at, when it was.
            Ignored unless the configuration asks for it.

        Returns
        -------
        Tensor
            ``(B, cond_dim)``.
        """
        time = self.noise_emb(c_noise)
        # An index selects one row of the embedding; a multi-hot row sums
        # the rows it names.  Selecting row i *is* the one-hot case of that
        # sum, so both go through the same weights — which is what lets one
        # checkpoint serve a categorical action set and a compositional one.
        if actions.ndim == 3:
            embedded = actions.to(self.action_embed.weight.dtype) @ (self.action_embed.weight)
        else:
            embedded = self.action_embed(actions)
        flat = embedded.reshape(int(actions.shape[0]), -1)
        total = time + flat
        if self.noise_cond_emb is not None:
            level = lucid.zeros_like(c_noise) if cond_noise is None else cond_noise
            total = total + self.noise_cond_emb(level)
        return self.cond_proj(total)

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Map ``(B, (L+1)*C, H, W)`` to a single ``(B, C, H, W)`` frame."""
        h = self.conv_in(x)
        h = self.unet(h, cond)
        h = self.norm_out(h)
        return self.conv_out(self.act(h))


class _Upsampler(nn.Module):
    r"""A second diffusion model that magnifies the denoiser's output.

    Parameters
    ----------
    config : DIAMONDConfig
        The variant to build; its ``upsampler_*`` fields describe this
        network and its ``upsampling_factor`` how far it magnifies.

    Notes
    -----
    ⚠️ Nowhere in the paper.  It exists only in the released CS:GO
    configuration, and it is why that experiment can afford a 3D scene:
    the world model diffuses at 30x56 and this brings the frame to
    150x280, rather than paying full-resolution diffusion for detail a
    cheaper network can add.

    It conditions on three images — the noised full-resolution frame, the
    low-resolution one scaled up to meet it, and the previous
    full-resolution frame — and on nothing else.  There is no action
    embedding: what the agent did is already in the frame it is being
    asked to sharpen.
    """

    def __init__(self, config: DIAMONDConfig) -> None:
        """Initialise the network. See the class docstring for parameters."""
        super().__init__()
        if config.upsampler_channels is None or config.upsampler_layers is None:
            raise ValueError("this configuration describes no upsampler")
        self.config = config
        cond_dim = config.cond_dim
        self.noise_emb = _FourierFeatures(cond_dim)
        self.noise_cond_emb = _FourierFeatures(cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.conv_in = nn.Conv2d(config.in_channels * 3, config.upsampler_channels[0], 3, padding=1)
        self.unet = _UNet(
            cond_dim,
            config.upsampler_channels,
            config.upsampler_layers,
            config.attention_head_dim,
            config.upsampler_attn_depths,
        )
        self.norm_out = _GroupNorm(config.upsampler_channels[0])
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(config.upsampler_channels[0], config.out_channels, 3, padding=1)

    def conditioning(self, c_noise: Tensor, cond_noise: Tensor | None = None) -> Tensor:
        """The two noise levels, embedded and summed.

        Parameters
        ----------
        c_noise : Tensor
            ``(B,)`` level of the frame being sharpened.
        cond_noise : Tensor or None, optional
            ``(B,)`` level its conditioning was noised at.

        Returns
        -------
        Tensor
            ``(B, cond_dim)``.
        """
        total = self.noise_emb(c_noise)
        level = lucid.zeros_like(c_noise) if cond_noise is None else cond_noise
        total = total + self.noise_cond_emb(level)
        return self.cond_proj(total)

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Map ``(B, 3*C, H, W)`` to one sharpened ``(B, C, H, W)`` frame."""
        h = self.conv_in(x)
        h = self.unet(h, cond)
        h = self.norm_out(h)
        return self.conv_out(self.act(h))


class _Encoder(nn.Module):
    """Downsampling residual stack, with attention at the bottom.

    Parameters
    ----------
    in_channels : int
        Channels entering the stack.
    channels, layers : tuple of int
        Per-resolution widths and block counts.
    cond_dim : int
        Conditioning width.
    head_dim : int
        Channels per attention head in the final stage.

    Notes
    -----
    One stage more than ``channels`` names: the extra one sits at the
    bottom resolution and attends, mirroring the U-Net's middle blocks.
    Without it the reward model's LSTM would see a feature map from a
    stack that never looked across the frame.
    """

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...],
        layers: tuple[int, ...],
        cond_dim: int,
        head_dim: int,
    ) -> None:
        """Initialise the stack. See the class docstring for parameters."""
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.blocks: nn.ModuleList[_ResBlocks] = nn.ModuleList()
        self.downsamples: nn.ModuleList[_Downsample | nn.Identity] = nn.ModuleList()
        width = channels[0]
        for stage, (out_ch, count) in enumerate(zip(channels, layers, strict=True)):
            self.downsamples.append(nn.Identity() if stage == 0 else _Downsample(width))
            self.blocks.append(
                _ResBlocks(
                    [_ResBlock(width if i == 0 else out_ch, out_ch, cond_dim) for i in range(count)]
                )
            )
            width = out_ch
        self.blocks.append(
            _ResBlocks(
                [_ResBlock(width, width, cond_dim, True, head_dim) for _ in range(layers[-1])]
            )
        )
        self.out_channels = width

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Reduce ``(B, C, H, W)`` by a factor of two per downsample."""
        h = self.conv_in(x)
        for stage, blocks in enumerate(self.blocks):
            if 0 < stage < len(self.downsamples):
                h = self.downsamples[stage](h)
            h = blocks(h, cond)
        return h


class _RewardEndModel(nn.Module):
    r"""Reward and episode termination, over a sequence.

    Parameters
    ----------
    config : DIAMONDConfig
        The variant to build.

    Notes
    -----
    Reference: Alonso et al., arXiv:2405.12399, Appendix D and
    Algorithm 1, with three details from the released checkpoint that
    the paper does not state.

    It reads **two** frames, not one — the observation and the one the
    action led to.  A reward is a property of the transition, and a
    single frame cannot show one.

    Its head emits **five** numbers: three reward classes and two
    termination classes.  Algorithm 1's ``CE(r_hat, sign(r))`` explains
    the first three, since the environment clips reward to
    :math:`\{-1, 0, 1\}`; the checkpoint explains the rest.
    """

    def __init__(self, config: DIAMONDConfig) -> None:
        """Initialise the model. See the class docstring for parameters."""
        super().__init__()
        self.config = config
        cond_dim = config.reward_cond_dim
        self.action_embed = nn.Embedding(config.num_actions, cond_dim)
        self.encoder = _Encoder(
            config.in_channels * config.reward_frames,
            config.reward_channels,
            config.reward_layers,
            cond_dim,
            config.attention_head_dim,
        )
        side = _side(config) // (2 ** (len(config.reward_channels) - 1))
        flat = self.encoder.out_channels * side * side
        self.cell = nn.LSTMCell(flat, config.reward_lstm_dim)
        self.head = nn.Sequential(
            nn.Linear(config.reward_lstm_dim, config.reward_lstm_dim),
            nn.SiLU(),
            # No bias on the last layer — the released checkpoint has
            # none, and a bias here would be five parameters the port
            # could not fill.
            nn.Linear(config.reward_lstm_dim, 5, bias=False),
        )

    @override
    def forward(
        self, frames: Tensor, action: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Advance one step.

        Parameters
        ----------
        frames : Tensor
            ``(B, reward_frames * C, H, W)`` — the transition, stacked.
        action : Tensor
            ``(B,)`` action indices taken from the first of them.
        state : tuple of Tensor or None, optional
            Carried LSTM ``(hidden, cell)``; ``None`` starts at zero,
            which is what the burn-in then overwrites.

        Returns
        -------
        (Tensor, Tensor, tuple of Tensor)
            Reward class logits ``(B, 3)``, termination class logits
            ``(B, 2)``, and the LSTM state to carry forward.
        """
        batch = int(frames.shape[0])
        cond = self.action_embed(action)
        feature = self.encoder(frames, cond)
        flat = feature.reshape(batch, -1)
        if state is None:
            shape = (batch, self.config.reward_lstm_dim)  # start state in our dtype
            zeros = lucid.zeros(shape, dtype=model_dtype(self), device=frames.device)
            state = (zeros, zeros)
        hidden, cell = self.cell(flat, state)
        out = self.head(hidden)
        return out[:, :3], out[:, 3:], (hidden, cell)


class _SimpleResBlock(nn.Module):
    """Norm, activation, one convolution, plus a projected skip.

    Notes
    -----
    The actor-critic's block is *not* the denoiser's: one convolution
    rather than two, an ordinary affine group norm rather than a
    conditioned one, and a 1x1 ``skip_projection`` when the width
    changes.  It has nothing to condition on — the policy sees a frame
    and nothing else.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialise the block.

        Parameters
        ----------
        in_channels, out_channels : int
            Channel widths.
        """
        super().__init__()
        self.f = nn.Sequential(
            _GroupNorm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        )
        self.skip_projection: nn.Conv2d | None = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply the block to ``(B, C, H, W)``."""
        residual = x if self.skip_projection is None else self.skip_projection(x)
        return residual + self.f(x)


class _ActorCritic(nn.Module):
    r"""Shared trunk, a policy head and a state-value head.

    Parameters
    ----------
    config : DIAMONDConfig
        The variant to build.

    Notes
    -----
    Reference: Alonso et al., arXiv:2405.12399, Appendix D — the weights
    "are shared except for the last layer".  :math:`V` is a state-value
    network rather than a critic proper, which is why the policy loss is
    REINFORCE with a baseline and not an advantage actor-critic.
    """

    def __init__(self, config: DIAMONDConfig) -> None:
        """Initialise the network. See the class docstring for parameters."""
        super().__init__()
        self.config = config
        layers: list[nn.Module] = [
            nn.Conv2d(config.in_channels, config.actor_channels[0], 3, padding=1)
        ]
        width = config.actor_channels[0]
        for out_ch, count in zip(config.actor_channels, config.actor_layers, strict=True):
            for _ in range(count):
                layers.append(_SimpleResBlock(width, out_ch))
                width = out_ch
            layers.append(nn.MaxPool2d(2, stride=2))
        self.encoder = nn.Sequential(*layers)
        side = _side(config) // (2 ** len(config.actor_channels))
        self.cell = nn.LSTMCell(width * side * side, config.actor_lstm_dim)
        self.actor_linear = nn.Linear(config.actor_lstm_dim, config.num_actions)
        self.critic_linear = nn.Linear(config.actor_lstm_dim, 1)

    @override
    def forward(
        self, frame: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Advance one step.

        Parameters
        ----------
        frame : Tensor
            ``(B, C, H, W)``.
        state : tuple of Tensor or None, optional
            Carried LSTM ``(hidden, cell)``.

        Returns
        -------
        (Tensor, Tensor, tuple of Tensor)
            Action logits ``(B, num_actions)``, state value ``(B,)``, and
            the LSTM state.
        """
        batch = int(frame.shape[0])
        feature = self.encoder(frame)
        flat = feature.reshape(batch, -1)
        if state is None:
            shape = (batch, self.config.actor_lstm_dim)  # start state in our dtype
            zeros = lucid.zeros(shape, dtype=model_dtype(self), device=frame.device)
            state = (zeros, zeros)
        hidden, cell = self.cell(flat, state)
        logits = self.actor_linear(hidden)
        value = self.critic_linear(hidden).reshape(-1)
        return logits, value, (hidden, cell)


# The decoder's upsample takes a target size rather than a second tensor,
# for the same reason: a stride-2 encoder does not halve cleanly.
def _side(config: DIAMONDConfig) -> int:
    """Frame side length, whichever way ``sample_size`` was written."""
    return config.sample_size if isinstance(config.sample_size, int) else config.sample_size[0]


@dataclass(slots=True)
class DIAMONDOutput(ModelOutput):
    """What the world model returns after denoising one step.

    Attributes
    ----------
    loss : Tensor
        The reconstruction loss of Algorithm 1, ``||D(x_noised) - x||^2``.
        Scalar.
    prediction : Tensor
        The denoised next frame, ``(B, C, H, W)``.
    sigma : Tensor
        The noise level each element was trained at, ``(B,)``.  Carried
        because the loss is an average over a *distribution* of noise
        levels, and a run that only ever drew easy ones would report a
        falling loss while learning nothing about the hard regime.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.diamond._model import DIAMONDOutput
    >>> out = DIAMONDOutput(
    ...     loss=lucid.zeros(()),
    ...     prediction=lucid.zeros(1, 3, 64, 64),
    ...     sigma=lucid.zeros(1),
    ... )
    >>> out.prediction.shape
    (1, 3, 64, 64)

    ``sigma`` is carried out with the prediction rather than left behind:
    the noise level a denoiser was asked to work at is what the loss has
    to weight by, so losing it makes the number meaningless.
    """

    loss: Tensor
    prediction: Tensor
    sigma: Tensor


@dataclass(slots=True)
class DIAMONDBehaviorOutput(ModelOutput):
    """What a pass of imagination returns.

    Attributes
    ----------
    policy_loss, value_loss : Tensor
        REINFORCE with a baseline, and the squared error against
        :math:`\\lambda`-returns.  Scalars.
    entropy : Tensor
        Mean policy entropy over the imagined trajectory, before the
        weight :math:`\\eta` is applied.  Scalar.
    returns : Tensor
        The :math:`\\lambda`-returns the value head regressed to,
        ``(B, H)``.
    frames : Tensor
        Every frame the denoiser imagined, ``(B, H, C, H_img, W_img)``.
    history : Tensor
        The conditioning the rollout ended on, ``(B, L, C, H_img, W_img)``
        — what a continuation would start from.  Once the horizon
        reaches :math:`L` this is entirely imagined, which is the
        clearest statement of what "training in imagination" means.
    history_actions : Tensor
        The actions beside it, ``(B, L)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.diamond._model import (
    ...     DIAMONDBehaviorOutput,
    ... )
    >>> zero = lucid.zeros(())
    >>> out = DIAMONDBehaviorOutput(
    ...     policy_loss=zero,
    ...     value_loss=zero,
    ...     entropy=zero,
    ...     returns=lucid.zeros(1, 4),
    ...     frames=lucid.zeros(1, 4, 3, 64, 64),
    ...     history=lucid.zeros(1, 4, 3, 64, 64),
    ...     history_actions=lucid.zeros(1, 4),
    ... )
    >>> out.frames.shape
    (1, 4, 3, 64, 64)

    The imagined rollout comes back with the losses, not instead of them:
    the frames the policy was trained against are the only way to see
    what the world model actually showed it.
    """

    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    returns: Tensor
    frames: Tensor
    history: Tensor
    history_actions: Tensor


class DIAMONDModel(PretrainedModel):
    r"""The three networks that make up a DIAMOND agent.

    Parameters
    ----------
    config : DIAMONDConfig
        Frozen configuration.

    Attributes
    ----------
    denoiser : _Denoiser
        The U-Net whose preconditioned wrapper predicts the next frame.
    reward_end : _RewardEndModel
        Reward class and termination, over a sequence.
    actor_critic : _ActorCritic
        The policy and its value baseline.

    Notes
    -----
    Reference: Alonso, Eloi, et al., *"Diffusion for World Modeling:
    Visual Details Matter in Atari"*, NeurIPS, 2024 (arXiv:2405.12399).

    The EDM preconditioners live on this class rather than inside the
    U-Net, because they are a property of the *diffusion*, not of the
    network: the same U-Net under DDPM's parameterisation is the
    comparison the paper's Section 5.1 runs, and it drifts.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.diamond import DIAMONDConfig, DIAMONDModel
    >>> config = DIAMONDConfig(
    ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
    ...     reward_channels=(8, 8), reward_layers=(1, 1),
    ...     actor_channels=(8, 8), actor_layers=(1, 1),
    ...     cond_dim=16, reward_cond_dim=8,
    ...     reward_lstm_dim=16, actor_lstm_dim=16, num_actions=4)
    >>> model = DIAMONDModel(config).eval()
    >>> frames = lucid.randn((2, 4, 3, 16, 16))
    >>> actions = lucid.tensor([[0, 1, 2, 3], [1, 1, 0, 2]], dtype=lucid.int64)
    >>> with lucid.no_grad():
    ...     nxt = model.imagine_frame(frames, actions)
    >>> nxt.shape
    (2, 3, 16, 16)
    """

    config_class: ClassVar[type[DIAMONDConfig]] = DIAMONDConfig

    def __init__(self, config: DIAMONDConfig) -> None:
        """Initialise the agent. See the class docstring for parameters."""
        super().__init__(config)
        self.config: DIAMONDConfig = config
        self.denoiser = _Denoiser(config)
        # The CS:GO configuration sets both of these to null: that
        # experiment trains a world model on static data, with no
        # reinforcement learning to give an agent anything to learn from.
        self.reward_end = _RewardEndModel(config) if config.with_agent else None
        self.actor_critic = _ActorCritic(config) if config.with_agent else None
        self.upsampler = _Upsampler(config) if config.upsampler_channels is not None else None

    def step_reward_end(
        self,
        frame: Tensor,
        next_frame: Tensor,
        action: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """One step of the reward/termination model, typed.

        Parameters
        ----------
        frame, next_frame : Tensor
            ``(B, C, H, W)`` each — the transition.  Both are needed
            because a reward is a property of the transition, and the
            released encoder reads them stacked.
        action : Tensor
            ``(B,)`` action indices taken from ``frame``.
        state : tuple of Tensor or None, optional
            Carried LSTM state.

        Returns
        -------
        (Tensor, Tensor, tuple of Tensor)
            Reward class logits ``(B, 3)``, termination class logits
            ``(B, 2)``, and the new state.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.diamond import DIAMONDConfig, DIAMONDModel
        >>> config = DIAMONDConfig(
        ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
        ...     reward_channels=(8, 8), reward_layers=(1, 1),
        ...     actor_channels=(8, 8), actor_layers=(1, 1),
        ...     cond_dim=16, reward_cond_dim=8,
        ...     reward_lstm_dim=16, actor_lstm_dim=16, num_actions=4)
        >>> model = DIAMONDModel(config).eval()
        >>> frame, after = lucid.randn((2, 3, 16, 16)), lucid.randn((2, 3, 16, 16))
        >>> action = lucid.tensor([0, 3], dtype=lucid.int64)
        >>> with lucid.no_grad():
        ...     reward, end, state = model.step_reward_end(frame, after, action)
        >>> reward.shape, end.shape, [s.shape for s in state]
        ((2, 3), (2, 2), [(2, 16), (2, 16)])

        The three reward classes are the clipped signs ``-1, 0, +1``, in
        that order, so their probabilities give an expected reward in
        ``[-1, 1]``:

        >>> probs = lucid.softmax(reward, dim=-1)
        >>> expected = probs[:, 2] - probs[:, 0]
        >>> bool((expected.abs() <= 1).all())
        True
        """
        if self.reward_end is None:
            raise ValueError(
                "this configuration has no reward/termination model " "(config.with_agent=False)"
            )
        pair = lucid.cat([frame, next_frame], dim=1)
        return self.reward_end(pair, action, state)

    def step_actor_critic(
        self, frame: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """One step of the actor-critic, typed.

        Parameters
        ----------
        frame : Tensor
            ``(B, C, H, W)``.
        state : tuple of Tensor or None, optional
            Carried LSTM state.

        Returns
        -------
        (Tensor, Tensor, tuple of Tensor)
            Action logits, state value, and the new state.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.diamond import DIAMONDConfig, DIAMONDModel
        >>> config = DIAMONDConfig(
        ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
        ...     reward_channels=(8, 8), reward_layers=(1, 1),
        ...     actor_channels=(8, 8), actor_layers=(1, 1),
        ...     cond_dim=16, reward_cond_dim=8,
        ...     reward_lstm_dim=16, actor_lstm_dim=16, num_actions=4)
        >>> model = DIAMONDModel(config).eval()
        >>> frame = lucid.randn((2, 3, 16, 16))
        >>> with lucid.no_grad():
        ...     logits, value, state = model.step_actor_critic(frame)
        ...     carried, _, _ = model.step_actor_critic(frame, state)
        >>> logits.shape, value.shape, state[0].shape
        ((2, 4), (2,), (2, 16))

        The policy has a memory: the same frame read with the carried state
        gives different logits than it did from a blank one.

        >>> bool((carried == logits).all())
        False
        """
        if self.actor_critic is None:
            raise ValueError("this configuration has no actor-critic (config.with_agent=False)")
        return self.actor_critic(frame, state)

    def preconditioners(self, sigma: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""EDM's four scalings at a noise level.

        Parameters
        ----------
        sigma : Tensor
            ``(B,)`` noise levels.

        Returns
        -------
        (Tensor, Tensor, Tensor, Tensor)
            ``c_in``, ``c_out``, ``c_skip`` — each ``(B, 1, 1, 1)`` so
            they broadcast over a frame — and ``c_noise`` as ``(B,)``.

        Notes
        -----
        Reference: Alonso et al., arXiv:2405.12399, Appendix C,
        equations 9-12, with :math:`\sigma_{\text{data}} = 0.5`.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.diamond import DIAMONDConfig, DIAMONDModel
        >>> config = DIAMONDConfig(
        ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
        ...     reward_channels=(8, 8), reward_layers=(1, 1),
        ...     actor_channels=(8, 8), actor_layers=(1, 1),
        ...     cond_dim=16, reward_cond_dim=8,
        ...     reward_lstm_dim=16, actor_lstm_dim=16, num_actions=4)
        >>> model = DIAMONDModel(config)
        >>> c_in, c_out, c_skip, c_noise = model.preconditioners(
        ...     lucid.tensor([0.002, 0.5, 5.0]))
        >>> c_in.shape, c_noise.shape
        ((3, 1, 1, 1), (3,))

        The skip hands over to the network as the noise grows, until the
        network is asked for the clean frame rather than for the noise:

        >>> [round(v, 3) for v in c_skip.reshape(-1).tolist()]
        [0.735, 0.424, 0.01]

        The low end stops short of 1 because the offset noise is folded
        into every level first, so even a clean level is scaled as if it
        carried ``sigma_offset_noise``:

        >>> data, offset = config.sigma_data, config.sigma_offset_noise
        >>> round(data**2 / (data**2 + offset**2), 3)
        0.735
        """
        # The offset noise folds into sigma *before* anything else, so a
        # step at sigma=2e-3 is really operating at 0.1 for CS:GO.  Every
        # preconditioner, c_noise included, sees the folded level.
        sigma = (sigma**2 + self.config.sigma_offset_noise**2).sqrt()
        data = self.config.sigma_data
        total = (sigma**2 + data**2).sqrt()
        c_in = (1.0 / total).reshape(-1, 1, 1, 1)
        c_out = (sigma * data / total).reshape(-1, 1, 1, 1)
        c_skip = (data**2 / (data**2 + sigma**2)).reshape(-1, 1, 1, 1)
        c_noise = lucid.log(sigma) * 0.25
        return c_in, c_out, c_skip, c_noise

    def denoise(
        self,
        noised: Tensor,
        sigma: Tensor,
        frames: Tensor,
        actions: Tensor,
        cond_sigma: Tensor | None = None,
        *,
        quantize: bool = True,
    ) -> Tensor:
        r"""Apply :math:`D_\theta` — the preconditioned denoiser.

        Parameters
        ----------
        noised : Tensor
            ``(B, C, H, W)``, the next frame with noise added.
        sigma : Tensor
            ``(B,)`` noise levels.
        frames : Tensor
            ``(B, L, C, H, W)`` clean history.
        actions : Tensor
            ``(B, L)`` indices, or ``(B, L, num_actions)`` multi-hot.
        cond_sigma : Tensor or None, optional
            ``(B,)`` level to noise the *history* at.  Only meaningful
            when the configuration asks for it; ``None`` leaves the
            history clean, which is what Atari does.
        quantize : bool, default=True, keyword-only
            Put the estimate back on the 8-bit grid, as the released
            sampler does.  Training passes ``False``: the quantiser's
            gradient is zero, and the loss belongs on the estimate anyway.

        Returns
        -------
        Tensor
            ``(B, C, H, W)`` estimate of the clean next frame.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.diamond import DIAMONDConfig, DIAMONDModel
        >>> config = DIAMONDConfig(
        ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
        ...     reward_channels=(8, 8), reward_layers=(1, 1),
        ...     actor_channels=(8, 8), actor_layers=(1, 1),
        ...     cond_dim=16, reward_cond_dim=8,
        ...     reward_lstm_dim=16, actor_lstm_dim=16, num_actions=4)
        >>> model = DIAMONDModel(config).eval()
        >>> frames = lucid.randn((2, 4, 3, 16, 16))
        >>> actions = lucid.tensor([[0, 1, 2, 3], [1, 1, 0, 2]], dtype=lucid.int64)
        >>> noised, sigma = lucid.randn((2, 3, 16, 16)), lucid.tensor([0.5, 2.0])
        >>> clean = model.denoise(noised, sigma, frames, actions)
        >>> clean.shape
        (2, 3, 16, 16)

        By default the estimate is put back on the 8-bit grid of ``[-1, 1]``,
        whose gradient is zero everywhere; training asks for the raw estimate
        instead:

        >>> bool(clean.min() >= -1.0), bool(clean.max() <= 1.0)
        (True, True)
        >>> model.denoise(noised, sigma, frames, actions, quantize=False).requires_grad
        True
        """
        c_in, c_out, c_skip, c_noise = self.preconditioners(sigma)
        if self.config.noise_previous_obs and cond_sigma is not None:
            frames = frames + lucid.randn_like(frames) * cond_sigma.reshape(-1, 1, 1, 1, 1)
        history = frames.reshape(int(frames.shape[0]), -1, *frames.shape[3:])
        # ``conv_in`` reads the history first, oldest to newest, and the
        # noised frame last.  The checkpoint says so: its per-input-channel
        # weight norms are 0.436, 0.437, 0.520, 0.676 across the four
        # history frames and 2.365 for the noised one.  The history is
        # rescaled by sigma_data, the noised frame by c_in.
        stacked = lucid.cat([history / self.config.sigma_data, noised * c_in], dim=1)
        level = None if cond_sigma is None else lucid.log(cond_sigma) * 0.25
        cond = self.denoiser.conditioning(c_noise, actions, level)
        out = c_skip * noised + c_out * self.denoiser(stacked, cond)
        return _quantize(out) if quantize else out

    def sigma_schedule(self, steps: int, device: str) -> Tensor:
        r"""The noise levels an Euler sampler walks down.

        Parameters
        ----------
        steps : int
            Number of denoising steps.
        device : str
            Where to build the schedule.

        Returns
        -------
        Tensor
            ``(steps + 1,)`` descending to exactly zero.

        Notes
        -----
        Karras et al.'s :math:`\rho`-schedule, which Algorithm 1 refers
        to as "the default identity schedule from EDM".  ⚠️ DIAMOND's
        tables give the sampler and the step count but **not**
        :math:`\sigma_{\min}`, :math:`\sigma_{\max}` or :math:`\rho`, so
        those come from EDM itself and are the one place here where a
        number is inherited rather than cited.

        Examples
        --------
        >>> from lucid.models.generative.diamond import DIAMONDConfig, DIAMONDModel
        >>> config = DIAMONDConfig(
        ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
        ...     reward_channels=(8, 8), reward_layers=(1, 1),
        ...     actor_channels=(8, 8), actor_layers=(1, 1),
        ...     cond_dim=16, reward_cond_dim=8,
        ...     reward_lstm_dim=16, actor_lstm_dim=16, num_actions=4)
        >>> model = DIAMONDModel(config)
        >>> schedule = model.sigma_schedule(3, "cpu")
        >>> schedule.shape
        (4,)

        EDM's ``sigma_max`` down to its ``sigma_min``, then an exact zero, so
        the last Euler step lands on the denoiser's own estimate:

        >>> [round(v, 4) for v in schedule.tolist()]
        [5.0, 0.2831, 0.002, 0.0]
        """
        rho, sigma_min, sigma_max = 7.0, 2e-3, 5.0
        work = _float32_or_wider(model_dtype(self))  # float32+, cast per step
        ramp = lucid.arange(steps, dtype=work, device=device) / max(steps - 1, 1)
        inv = sigma_max ** (1.0 / rho) + ramp * (
            sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho)
        )
        return lucid.cat([inv**rho, lucid.zeros((1,), dtype=work, device=device)])

    def upsample(
        self,
        noised: Tensor,
        sigma: Tensor,
        low_res: Tensor,
        previous: Tensor,
        *,
        quantize: bool = True,
    ) -> Tensor:
        r"""Apply the upsampler's :math:`D_\theta`.

        Parameters
        ----------
        noised : Tensor
            ``(B, C, H*f, W*f)`` the full-resolution frame with noise.
        sigma : Tensor
            ``(B,)`` noise levels.
        low_res : Tensor
            ``(B, C, H, W)`` the frame the world model generated.
        previous : Tensor
            ``(B, C, H*f, W*f)`` the previous full-resolution frame, so
            the sharpening is temporally consistent rather than
            independent per frame.
        quantize : bool, default=True, keyword-only
            Put the estimate back on the 8-bit grid, as the released
            sampler does.  Pass ``False`` to keep a gradient — the
            quantiser has none.

        Returns
        -------
        Tensor
            ``(B, C, H*f, W*f)``.

        Raises
        ------
        ValueError
            If this configuration has no upsampler.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import diamond_csgo
        >>> model = diamond_csgo(
        ...     unet_channels=(8, 8), unet_layers=(1, 1), cond_dim=16,
        ...     attn_depths=(0, 0), upsampler_channels=(4, 4),
        ...     upsampler_layers=(1, 1), upsampler_attn_depths=(0, 0),
        ...     num_actions=4).eval()
        >>> low = lucid.randn((1, 3, 30, 56))
        >>> noised = lucid.randn((1, 3, 150, 280))
        >>> previous = lucid.randn((1, 3, 150, 280))
        >>> with lucid.no_grad():
        ...     full = model.upsample(noised, lucid.tensor([1.0]), low, previous)
        >>> full.shape
        (1, 3, 150, 280)

        One denoiser evaluation at one noise level — :meth:`upsample_frame`
        walks it down a schedule — quantised back onto ``[-1, 1]``:

        >>> bool(full.abs().max() <= 1.0)
        True
        """
        if self.upsampler is None:
            raise ValueError(
                "this configuration has no upsampler — only the CS:GO one "
                "does, and it generates at 30x56 to magnify five times"
            )
        c_in, c_out, c_skip, c_noise = self.preconditioners(sigma)
        size = (int(noised.shape[2]), int(noised.shape[3]))
        # Bicubic, because that is how the training pairs were built:
        # the released loss upscales the low-resolution frame with
        # ``mode="bicubic"`` before handing it to the upsampler.  Nearest
        # gives the network a blockier conditioning image than it ever saw.
        scaled = F.interpolate(low_res, size=size, mode="bicubic")
        # Same order as the denoiser: conditioning images first — the
        # previous full-resolution frame, then the low-resolution one
        # scaled up — each divided by sigma_data, and the noised frame
        # last, scaled by c_in.
        data = self.config.sigma_data
        stacked = lucid.cat([previous / data, scaled / data, noised * c_in], dim=1)
        cond = self.upsampler.conditioning(c_noise)
        out = c_skip * noised + c_out * self.upsampler(stacked, cond)
        return _quantize(out) if quantize else out

    def upsample_frame(
        self,
        low_res: Tensor,
        previous: Tensor | None = None,
        *,
        steps: int | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        r"""Sample a full-resolution frame from a generated low-resolution one.

        Parameters
        ----------
        low_res : Tensor
            ``(B, C, H, W)`` — what :meth:`imagine_frame` produced.
        previous : Tensor or None, optional
            ``(B, C, H*f, W*f)`` previous full-resolution frame.  Absent
            on the first frame of a rollout, where the low-resolution one
            scaled up stands in for it.
        steps : int or None, optional
            Denoising steps; defaults to the configured 3.
        noise : Tensor or None, optional
            Starting sample, drawn when absent.

        Returns
        -------
        Tensor
            ``(B, C, H*f, W*f)``.

        Raises
        ------
        ValueError
            If this configuration has no upsampler, or ``steps`` is not
            positive.

        Notes
        -----
        ⚠️ The upsampler appears in the released CS:GO configuration and
        nowhere in the paper.  It is why that experiment is affordable:
        the world model diffuses at 30x56 and this brings the frame to
        150x280, rather than paying full-resolution diffusion for detail
        a cheaper network can add.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import diamond_csgo
        >>> model = diamond_csgo(
        ...     unet_channels=(8, 8), unet_layers=(1, 1), cond_dim=16,
        ...     attn_depths=(0, 0), upsampler_channels=(4, 4),
        ...     upsampler_layers=(1, 1), upsampler_attn_depths=(0, 0),
        ...     num_actions=4).eval()
        >>> low = lucid.randn((1, 3, 30, 56))
        >>> with lucid.no_grad():
        ...     full = model.upsample_frame(low, steps=1)
        >>> full.shape
        (1, 3, 150, 280)
        """
        if self.upsampler is None:
            raise ValueError(
                "this configuration has no upsampler — only the CS:GO one "
                "does, and it generates at 30x56 to magnify five times"
            )
        steps = self.config.denoise_steps if steps is None else steps
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}")

        factor = self.config.upsampling_factor
        batch, dt = int(low_res.shape[0]), model_dtype(self)
        device = low_res.device.type
        size = (int(low_res.shape[2]) * factor, int(low_res.shape[3]) * factor)
        if previous is None:
            # The first frame has no predecessor; its own upscaling is the
            # closest thing to one, and is what the conditioning expects
            # rather than a zero field the model never saw in training.
            previous = F.interpolate(low_res, size=size, mode="bicubic")

        schedule = self.sigma_schedule(steps, device)
        shape = (batch, self.config.in_channels, *size)
        x = lucid.randn(shape, dtype=dt, device=device) if noise is None else noise
        for index in range(steps):
            sigma, step = schedule[index], schedule[index + 1] - schedule[index]
            level = sigma.to(dt) + lucid.zeros((batch,), dtype=dt, device=device)
            denoised = self.upsample(x, level, low_res, previous)
            x = x + (x - denoised) / sigma.to(dt) * step.to(dt)
        return x

    def imagine_frame(
        self,
        frames: Tensor,
        actions: Tensor,
        *,
        steps: int | None = None,
        noise: Tensor | None = None,
        cond_sigma: float | None = None,
    ) -> Tensor:
        r"""Sample the next frame with Euler's method.

        Parameters
        ----------
        frames : Tensor
            ``(B, L, C, H, W)`` clean history.
        actions : Tensor
            ``(B, L)`` indices, or ``(B, L, num_actions)`` multi-hot.
        steps : int or None, optional
            Denoising steps; defaults to the configured 3.
        cond_sigma : float or None, optional, keyword-only
            Noise the history at this level, and tell the network so.
            The released CS:GO sampler uses ``0.005``; ``None`` leaves
            the history clean, which is what Atari does.
        noise : Tensor or None, optional
            Starting sample, drawn when absent.

        Returns
        -------
        Tensor
            ``(B, C, H, W)``.

        Notes
        -----
        Each step moves along :math:`\mathrm{d}x/\mathrm{d}\sigma =
        (x - D_\theta(x, \sigma)) / \sigma`, which is the probability-flow
        ODE written in EDM's variables.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.diamond import DIAMONDConfig, DIAMONDModel
        >>> config = DIAMONDConfig(
        ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
        ...     reward_channels=(8, 8), reward_layers=(1, 1),
        ...     actor_channels=(8, 8), actor_layers=(1, 1),
        ...     cond_dim=16, reward_cond_dim=8,
        ...     reward_lstm_dim=16, actor_lstm_dim=16, num_actions=4)
        >>> model = DIAMONDModel(config).eval()
        >>> frames = lucid.randn((2, 4, 3, 16, 16))
        >>> actions = lucid.tensor([[0, 1, 2, 3], [1, 1, 0, 2]], dtype=lucid.int64)
        >>> noise = lucid.randn((2, 3, 16, 16))
        >>> with lucid.no_grad():
        ...     first = model.imagine_frame(frames, actions, noise=noise)
        ...     again = model.imagine_frame(frames, actions, noise=noise)
        ...     single = model.imagine_frame(frames, actions, noise=noise, steps=1)
        >>> first.shape
        (2, 3, 16, 16)

        The sampler solves an ODE, so the starting noise fixes the frame and
        a rollout can be reproduced; the number of steps changes it:

        >>> bool((first == again).all()), bool((first == single).all())
        (True, False)
        """
        steps = self.config.denoise_steps if steps is None else steps
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}")
        batch, dt = int(frames.shape[0]), model_dtype(self)
        device = frames.device.type
        schedule = self.sigma_schedule(steps, device)

        # Unit variance, not sigma_max: the released sampler starts the
        # walk at randn() and lets the first Euler step do the scaling.
        # ``noise`` is therefore used as given, which is what makes it
        # usable for reproducing a rollout.
        shape = (batch, self.config.in_channels, *frames.shape[3:])
        x = lucid.randn(shape, dtype=dt, device=device) if noise is None else noise
        level_cond = None
        if cond_sigma is not None:
            level_cond = cond_sigma + lucid.zeros((batch,), dtype=dt, device=device)
        for index in range(steps):
            sigma, step = schedule[index], schedule[index + 1] - schedule[index]
            level = sigma.to(dt) + lucid.zeros((batch,), dtype=dt, device=device)
            denoised = self.denoise(x, level, frames, actions, level_cond)
            derivative = (x - denoised) / sigma.to(dt)
            x = x + derivative * step.to(dt)
        return x

    @override
    def forward(
        self, frames: Tensor, actions: Tensor, next_frame: Tensor, *, sigma: Tensor | None = None
    ) -> DIAMONDOutput:
        r"""Train the denoiser on one transition.

        Parameters
        ----------
        frames : Tensor
            ``(B, L, C, H, W)`` clean history.
        actions : Tensor
            ``(B, L)`` indices, or ``(B, L, num_actions)`` multi-hot.
        next_frame : Tensor
            ``(B, C, H, W)`` the frame to predict.
        sigma : Tensor or None, optional, keyword-only
            ``(B,)`` noise levels.  Drawn from the paper's log-normal
            when absent; supplied by tests that need a fixed level.

        Returns
        -------
        DIAMONDOutput
            Loss, denoised frame, and the noise levels used.

        Notes
        -----
        Reference: Alonso et al., arXiv:2405.12399, Algorithm 1 — sample
        :math:`\log \sigma \sim \mathcal{N}(P_{\text{mean}},
        P_{\text{std}}^2)`, noise the target, and take the squared error
        in *pixel* space.  The EDM weighting is already inside
        :math:`D_\theta`, so no extra loss weight appears here.
        """
        batch = int(next_frame.shape[0])
        device, dt = next_frame.device.type, next_frame.dtype  # draws in the frames'
        if sigma is None:
            log_sigma = (
                lucid.randn((batch,), dtype=dt, device=device) * self.config.p_std
                + self.config.p_mean
            )
            sigma = lucid.exp(log_sigma)

        # Offset noise: one scalar per sample and channel, on top of the
        # usual isotropic draw.  It lets the target's overall level shift,
        # which isotropic noise at a single sigma cannot express — and it
        # exists only in the released configuration, not the paper.
        shape = (batch, int(next_frame.shape[1]), 1, 1)
        offset_noise = lucid.randn(shape, dtype=dt, device=device)
        offset = offset_noise * self.config.sigma_offset_noise
        noised = next_frame + offset + lucid.randn_like(next_frame) * sigma.reshape(-1, 1, 1, 1)
        # The CS:GO configuration noises the history too — a form of
        # augmentation against the error a long autoregressive rollout
        # compounds.  Atari leaves it clean.
        cond_sigma = None
        if self.config.noise_previous_obs:
            cond_sigma = lucid.exp(
                lucid.randn((batch,), dtype=dt, device=device) * self.config.p_std
                + self.config.p_mean
            )
        # No quantiser on the training path: it has no gradient, and the
        # released code computes its loss on the unwrapped estimate too.
        prediction = self.denoise(noised, sigma, frames, actions, cond_sigma, quantize=False)
        loss = ((prediction - next_frame) ** 2).mean()
        return DIAMONDOutput(loss=loss, prediction=prediction, sigma=sigma)


class DIAMONDForWorldModeling(WorldModelingModel):
    r"""DIAMOND posed as a world model: imagination and its objectives.

    Parameters
    ----------
    config : DIAMONDConfig
        Frozen configuration.

    Attributes
    ----------
    diamond : DIAMONDModel
        The three networks.

    Notes
    -----
    Reference: Alonso et al., arXiv:2405.12399, Appendix F and
    Algorithm 1.

    The imagined trajectory is autoregressive in the strong sense: the
    frame the denoiser produced and the action the policy chose from it
    both become conditioning for the next step, so an error in either
    compounds.  That is the whole reason the paper cares which diffusion
    parameterisation it uses.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.diamond import (
    ...     DIAMONDConfig, DIAMONDForWorldModeling)
    >>> config = DIAMONDConfig(
    ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
    ...     reward_channels=(8, 8), reward_layers=(1, 1),
    ...     actor_channels=(8, 8), actor_layers=(1, 1),
    ...     cond_dim=16, reward_cond_dim=8, reward_lstm_dim=16,
    ...     actor_lstm_dim=16, num_actions=4, horizon=3)
    >>> model = DIAMONDForWorldModeling(config).eval()
    >>> frames = lucid.randn((2, 4, 3, 16, 16))
    >>> actions = lucid.tensor([[0, 1, 2, 3], [1, 1, 0, 2]], dtype=lucid.int64)
    >>> with lucid.no_grad():
    ...     out = model(frames, actions)
    >>> out.frames.shape
    (2, 3, 3, 16, 16)
    """

    config_class: ClassVar[type[DIAMONDConfig]] = DIAMONDConfig

    def __init__(self, config: DIAMONDConfig) -> None:
        """Initialise the wrapper. See the class docstring for parameters."""
        super().__init__(config)
        self.config: DIAMONDConfig = config
        self.diamond = DIAMONDModel(config)

    @overload
    def act(
        self,
        frame: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
        *,
        return_state: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def act(
        self,
        frame: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
        *,
        return_state: Literal[True],
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]: ...

    @overload
    def act(
        self,
        frame: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
        *,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]: ...

    def act(
        self,
        frame: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
        *,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """Sample an action from the policy.

        Parameters
        ----------
        frame : Tensor
            ``(B, C, H, W)``.
        state : tuple of Tensor or None, optional
            Carried actor-critic LSTM ``(hidden, cell)``.  ``None`` reads
            ``frame`` from a blank memory, as at the start of an episode.
        return_state : bool, default=False, keyword-only
            Also return the LSTM state after reading ``frame``.  The policy
            is recurrent, so an agent loop must pass this back as the next
            call's ``state``; without it every frame is read from a blank
            memory and the policy remembers nothing.

        Returns
        -------
        Tensor or (Tensor, tuple of Tensor)
            ``(B,)`` action indices.  With ``return_state=True``, the pair
            ``(action, state)``, where ``state`` is the one
            :meth:`DIAMONDModel.step_actor_critic` returns for the same
            frame and carried state.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.diamond import (
        ...     DIAMONDConfig, DIAMONDForWorldModeling)
        >>> config = DIAMONDConfig(
        ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
        ...     reward_channels=(8, 8), reward_layers=(1, 1),
        ...     actor_channels=(8, 8), actor_layers=(1, 1),
        ...     cond_dim=16, reward_cond_dim=8, reward_lstm_dim=16,
        ...     actor_lstm_dim=16, num_actions=4)
        >>> model = DIAMONDForWorldModeling(config).eval()
        >>> with lucid.no_grad():
        ...     action = model.act(lucid.randn((2, 3, 16, 16)))
        >>> action.shape, action.dtype
        ((2,), lucid.int64)

        A draw from the policy rather than its argmax, but always a valid
        index into the action set:

        >>> bool(action.min() >= 0), bool(action.max() < config.num_actions)
        (True, True)

        The policy is recurrent, so an agent loop asks for the state back
        and hands it to the next call — that is what gives it a memory:

        >>> frames = lucid.randn((2, 5, 3, 16, 16))
        >>> state = None
        >>> with lucid.no_grad():
        ...     for t in range(5):
        ...         action, state = model.act(frames[:, t], state, return_state=True)
        ...     _, blank = model.act(frames[:, -1], return_state=True)
        >>> action.shape, [s.shape for s in state]
        ((2,), [(2, 16), (2, 16)])

        The last frame, read after the four before it, leaves a different
        memory than the same frame read cold:

        >>> bool((state[0] == blank[0]).all())
        False
        """
        logits, _value, new_state = self.diamond.step_actor_critic(frame, state)
        probs = lucid.softmax(logits, dim=-1)
        action = lucid.multinomial(probs, num_samples=1).reshape(-1)
        if return_state:
            return action, new_state
        return action

    @override
    def forward(
        self, frames: Tensor, actions: Tensor, *, horizon: int | None = None
    ) -> DIAMONDBehaviorOutput:
        r"""Imagine a trajectory and score the policy on it.

        Parameters
        ----------
        frames : Tensor
            ``(B, L, C, H, W)`` real frames the rollout starts from.
        actions : Tensor
            ``(B, L)`` the actions that produced them.
        horizon : int or None, optional, keyword-only
            Steps to imagine; defaults to the configured 15.

        Returns
        -------
        DIAMONDBehaviorOutput
            The two losses, the policy's entropy, the
            :math:`\lambda`-returns and every imagined frame.

        Notes
        -----
        Reference: Alonso et al., arXiv:2405.12399, equations 14-16.
        The value target stops gradients, and the policy is REINFORCE
        against :math:`\Lambda_t - V(x_t)` — an advantage estimate, but
        one built from a *state-value* baseline rather than a critic.
        """
        horizon = self.config.horizon if horizon is None else horizon
        if horizon < 2:
            raise ValueError(f"lambda-returns need at least two imagined states, got {horizon}")
        config = self.config
        history = frames
        history_actions = actions

        reward_state: tuple[Tensor, Tensor] | None = None
        actor_state: tuple[Tensor, Tensor] | None = None
        # The reward model reads a transition, so the burn-in can only
        # feed it pairs — one fewer step than the actor-critic gets.
        for step in range(int(frames.shape[1])):
            if step + 1 < int(frames.shape[1]):
                _r, _d, reward_state = self.diamond.step_reward_end(
                    frames[:, step], frames[:, step + 1], actions[:, step], reward_state
                )
            _l, _v, actor_state = self.diamond.step_actor_critic(frames[:, step], actor_state)

        imagined: list[Tensor] = []
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        values: list[Tensor] = []
        rewards: list[Tensor] = []
        continues: list[Tensor] = []

        for _ in range(horizon):
            logits, value, actor_state = self.diamond.step_actor_critic(history[:, -1], actor_state)
            probs = lucid.softmax(logits, dim=-1)
            action = lucid.multinomial(probs, num_samples=1).reshape(-1)
            log_prob = lucid.log(probs + 1e-8)
            # Picked out with a one-hot rather than advanced indexing: the
            # gradient path is the same and it does not depend on how the
            # backend handles a pair of index tensors.
            onehot = F.one_hot(action, num_classes=config.num_actions).to(log_prob.dtype)
            chosen = (log_prob * onehot).sum(dim=-1)
            entropy = -(probs * log_prob).sum(dim=-1)

            frame = self.diamond.imagine_frame(history, history_actions)
            reward_logits, end_logits, reward_state = self.diamond.step_reward_end(
                history[:, -1], frame, action, reward_state
            )
            # Three classes, in the order the clipping produces them.
            reward_probs = lucid.softmax(reward_logits, dim=-1)
            reward = reward_probs[:, 2] - reward_probs[:, 0]
            keep = lucid.softmax(end_logits, dim=-1)[:, 0]

            imagined.append(frame)
            log_probs.append(chosen)
            entropies.append(entropy)
            values.append(value)
            rewards.append(reward)
            continues.append(keep)

            history = lucid.cat(
                [history[:, 1:], frame.reshape(int(frame.shape[0]), 1, *frame.shape[1:])], dim=1
            )
            history_actions = lucid.cat([history_actions[:, 1:], action.reshape(-1, 1)], dim=1)

        _logits, bootstrap, _state = self.diamond.step_actor_critic(history[:, -1], actor_state)
        value_seq = lucid.stack([*values, bootstrap], dim=1)
        reward_seq = lucid.stack([*rewards, lucid.zeros_like(rewards[0])], dim=1)
        discount = lucid.stack(continues, dim=1) * config.gamma
        discount = lucid.cat([discount, lucid.zeros_like(discount[:, :1])], dim=1)

        returns = lambda_return(reward_seq, value_seq, discount, config.lambda_)
        advantage = (returns - value_seq[:, :horizon]).detach()

        log_prob_seq = lucid.stack(log_probs, dim=1)
        entropy_seq = lucid.stack(entropies, dim=1)
        policy_loss = -(log_prob_seq * advantage + config.entropy_weight * entropy_seq).mean()
        value_loss = ((value_seq[:, :horizon] - returns.detach()) ** 2).mean()

        return DIAMONDBehaviorOutput(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy_seq.mean(),
            returns=returns,
            frames=lucid.stack(imagined, dim=1),
            history=history,
            history_actions=history_actions,
        )

    def world_model_loss(
        self, frames: Tensor, actions: Tensor, next_frame: Tensor
    ) -> DIAMONDOutput:
        """Train the denoiser — :meth:`DIAMONDModel.forward` by another name.

        Parameters
        ----------
        frames : Tensor
            ``(B, L, C, H, W)`` clean history.
        actions : Tensor
            ``(B, L)`` action indices.
        next_frame : Tensor
            ``(B, C, H, W)`` target.

        Returns
        -------
        DIAMONDOutput
            Loss, denoised frame, and the noise levels used.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.diamond import (
        ...     DIAMONDConfig, DIAMONDForWorldModeling)
        >>> config = DIAMONDConfig(
        ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
        ...     reward_channels=(8, 8), reward_layers=(1, 1),
        ...     actor_channels=(8, 8), actor_layers=(1, 1),
        ...     cond_dim=16, reward_cond_dim=8, reward_lstm_dim=16,
        ...     actor_lstm_dim=16, num_actions=4)
        >>> model = DIAMONDForWorldModeling(config)
        >>> frames = lucid.randn((2, 4, 3, 16, 16))
        >>> actions = lucid.tensor([[0, 1, 2, 3], [1, 1, 0, 2]], dtype=lucid.int64)
        >>> target = lucid.randn((2, 3, 16, 16))
        >>> out = model.world_model_loss(frames, actions, target)
        >>> out.loss.shape, out.prediction.shape, out.sigma.shape
        ((), (2, 3, 16, 16), (2,))

        Only the denoiser learns from it — the reward model and the
        actor-critic are trained by losses of their own:

        >>> out.loss.backward()
        >>> any(p.grad is not None for p in model.diamond.denoiser.parameters())
        True
        >>> any(p.grad is not None for p in model.diamond.reward_end.parameters())
        False
        """
        return self.diamond(frames, actions, next_frame)

    def reward_end_loss(
        self, frames: Tensor, actions: Tensor, rewards: Tensor, ends: Tensor
    ) -> Tensor:
        r"""Cross-entropy on the reward's *sign* and on termination.

        Parameters
        ----------
        frames : Tensor
            ``(B, T, C, H, W)``, with ``T >= 2``.
        actions : Tensor
            ``(B, T)``; ``actions[:, t]`` is the action taken at frame ``t``.
        rewards : Tensor
            ``(B, T)`` real rewards; ``rewards[:, t]`` is the reward for the
            step from frame ``t`` to frame ``t + 1``.  Only their sign is
            predicted — Algorithm 1 writes ``CE(r_hat, sign(r))``, which is
            all the environment's clipping to :math:`\{-1, 0, 1\}` leaves.
        ends : Tensor
            ``(B, T)`` termination flags in ``{0, 1}``, aligned the same
            way: ``ends[:, t]`` marks the step out of frame ``t``.

        Returns
        -------
        Tensor
            Scalar, the two cross-entropies summed, averaged over the
            ``T - 1`` transitions.

        Notes
        -----
        The reward model reads *transitions*, so ``T`` frames give
        ``T - 1`` of them: step ``t`` pairs ``frames[:, t]`` with
        ``frames[:, t + 1]`` and ``actions[:, t]``, and scores
        ``rewards[:, t]`` and ``ends[:, t]``.  The last column of
        ``actions``, ``rewards`` and ``ends`` has no next frame to pair
        with and is ignored — so each column must hold what the step *out
        of* its frame earned, not what was observed on arriving there.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.diamond import (
        ...     DIAMONDConfig, DIAMONDForWorldModeling)
        >>> config = DIAMONDConfig(
        ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
        ...     reward_channels=(8, 8), reward_layers=(1, 1),
        ...     actor_channels=(8, 8), actor_layers=(1, 1),
        ...     cond_dim=16, reward_cond_dim=8, reward_lstm_dim=16,
        ...     actor_lstm_dim=16, num_actions=4)
        >>> model = DIAMONDForWorldModeling(config)
        >>> frames = lucid.randn((2, 4, 3, 16, 16))
        >>> actions = lucid.tensor([[0, 1, 2, 3], [1, 1, 0, 2]], dtype=lucid.int64)
        >>> rewards = lucid.tensor([[0.0, 1.0, -2.0, 0.0], [5.0, 0.0, 0.0, 0.0]])
        >>> ends = lucid.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        >>> loss = model.reward_end_loss(frames, actions, rewards, ends)
        >>> loss.shape
        ()

        Only the sign of a reward is a target, so rescaling the rewards
        leaves the loss exactly where it was:

        >>> scaled = model.reward_end_loss(frames, actions, rewards * 10.0, ends)
        >>> scaled.item() == loss.item()
        True

        The last column pairs with no next frame, so it is never scored:

        >>> moved = lucid.cat([rewards[:, :-1], lucid.tensor([[-1.0], [1.0]])], dim=1)
        >>> model.reward_end_loss(frames, actions, moved, ends).item() == loss.item()
        True
        """
        state: tuple[Tensor, Tensor] | None = None
        # The accumulators live with the frames: a CPU zero refuses to add the
        # first loss of a model that has been moved to another device.
        device = frames.device.type
        reward_loss = lucid.zeros((), dtype=frames.dtype, device=device)
        end_loss = lucid.zeros((), dtype=frames.dtype, device=device)
        steps = int(frames.shape[1]) - 1
        if steps < 1:
            raise ValueError(
                f"the reward model reads transitions, so it needs at least "
                f"two frames, got {int(frames.shape[1])}"
            )
        for step in range(steps):
            logits, end_logits, state = self.diamond.step_reward_end(
                frames[:, step], frames[:, step + 1], actions[:, step], state
            )
            target = (rewards[:, step].sign() + 1.0).to(lucid.int64)
            reward_loss = reward_loss + F.cross_entropy(logits, target)
            end_loss = end_loss + F.cross_entropy(end_logits, ends[:, step].to(lucid.int64))
        return (reward_loss + end_loss) / float(steps)
