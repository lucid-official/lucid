r"""The second stage — a time-conditional U-Net with cross-attention.

The U-Net itself is DDPM's, at a different resolution.  What makes it a
*latent diffusion* U-Net is one block inserted after every self-attention:

.. math::

    Q = W_Q^{(i)} \varphi_i(z_t), \quad
    K = W_K^{(i)} \tau_\theta(y), \quad
    V = W_V^{(i)} \tau_\theta(y).

The latent asks and the conditioning answers, which is why the modality
of :math:`y` never reaches the U-Net.  A text encoder, a class embedding
and a segmentation map all arrive as ``(B, L, cross_attention_dim)``, and
nothing downstream can tell which it was reading.

**The asymmetry is the whole point and is easy to invert.**  Swapping Q
with K/V produces a model of identical shape that attends from the
prompt to the image — it trains, its loss falls, and it ignores what you
asked for.  The test suite pins the direction by changing the
conditioning and requiring the output to move; a reversed model fails
that while passing every shape check.
"""

from typing import final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models.generative._common._transformers import timestep_embedding
from lucid.models.generative.stable_diffusion._config import StableDiffusionConfig

__all__ = ["UNet2DConditionModel"]

# The reference gives the transformer's own normalisation a tighter
# epsilon than the residual blocks around it — 1e-6 against the U-Net
# config's ``norm_eps`` of 1e-5, which is the framework default the
# residual blocks pick up.  The two look interchangeable and are not:
# a checkpoint converted under the wrong one denoises to a *slightly*
# different image at every one of the sixteen transformer blocks, and
# nothing about a shape, a count or a loss reports it.
_ATTENTION_NORM_EPS = 1e-6


@final
class _ResBlock(nn.Module):
    """Residual block carrying the timestep embedding.

    Parameters
    ----------
    in_channels, out_channels, time_dim, groups : int
        Widths.

    Notes
    -----
    The timestep enters as a per-channel *shift* added between the two
    convolutions — after the first norm, before the second. Adding it to
    the input instead would let the first normalisation remove it, which
    is a silent way to build a diffusion model that cannot tell one step
    from another.
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int) -> None:
        """Initialise the block. See the class docstring for parameters."""
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.shortcut: nn.Conv2d | None = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    @override
    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Apply the block.

        Parameters
        ----------
        x : Tensor
            ``(B, in_channels, H, W)``.
        emb : Tensor
            ``(B, time_dim)``.

        Returns
        -------
        Tensor
            ``(B, out_channels, H, W)``.
        """
        h = self.conv1(self.act(self.norm1(x)))
        shift = self.time_proj(self.act(emb))
        h = h + shift.reshape(int(shift.shape[0]), int(shift.shape[1]), 1, 1)
        h = self.conv2(self.act(self.norm2(h)))
        skip = x if self.shortcut is None else self.shortcut(x)
        return skip + h


@final
class _GEGLU(nn.Module):
    r"""Gated GELU — ``(xW + b) \odot \mathrm{gelu}(xV + c)``.

    Parameters
    ----------
    in_features, out_features : int
        Widths.  The projection is twice ``out_features`` wide and its
        halves are the value and the gate.

    Notes
    -----
    Not interchangeable with ``Linear -> GELU``.  A GEGLU feed-forward
    carries :math:`3 d^2` parameters against a plain one's :math:`2 d^2`
    at the same inner width, and the released U-Net is built from the
    former — the difference is 49.5M parameters across its sixteen
    transformer blocks, which is how its absence was found.

    Reference: Shazeer, *"GLU Variants Improve Transformer"*, 2020
    (`arXiv:2002.05202 <https://arxiv.org/abs/2002.05202>`_).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialise the layer. See the class docstring for parameters."""
        super().__init__()
        self.proj = nn.Linear(in_features, out_features * 2)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Project, split, and gate.

        Parameters
        ----------
        x : Tensor
            ``(..., in_features)``.

        Returns
        -------
        Tensor
            ``(..., out_features)``.
        """
        projected = self.proj(x)
        half = int(projected.shape[-1]) // 2
        return projected[..., :half] * F.gelu(projected[..., half:])


@final
class _TransformerBlock(nn.Module):
    """Self-attention, then cross-attention, then a feed-forward.

    Parameters
    ----------
    channels : int
        Feature width; also the query width.
    heads : int
        Attention heads.
    context_dim : int
        Width of the conditioning sequence — the key/value width.

    Notes
    -----
    Cross-attention takes its keys and values from the conditioning and
    its queries from the image. :class:`lucid.nn.MultiheadAttention`
    accepts differing key/value widths through ``kdim`` / ``vdim``, so no
    projection of the conditioning to the image width is needed and the
    conditioning arrives at its own dimension.

    The released blocks carry **no bias on q/k/v and a bias on the output
    projection** — the checkpoint has ``to_q.weight`` with no
    ``to_q.bias`` beside it, and ``to_out.0.bias`` present.
    ``MultiheadAttention`` controls both with one flag, so the attention
    is built bias-free and the output bias is a separate parameter added
    afterwards. Folding it into the flag instead costs 24,960 parameters
    across the U-Net, which is exactly how the discrepancy was located.
    """

    def __init__(self, channels: int, heads: int, context_dim: int) -> None:
        """Initialise the block. See the class docstring for parameters."""
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        # SD's attention projections carry no bias — the released
        # checkpoint has ``to_q.weight`` and ``to_k.weight`` with no
        # matching ``.bias`` entries.
        self.attn1 = nn.MultiheadAttention(channels, heads, bias=False, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.attn2 = nn.MultiheadAttention(
            channels,
            heads,
            kdim=context_dim,
            vdim=context_dim,
            bias=False,
            batch_first=True,
        )
        # Restores the output bias ``bias=False`` removed above.
        self.attn1_out_bias = nn.Parameter(lucid.zeros((channels,)))
        self.attn2_out_bias = nn.Parameter(lucid.zeros((channels,)))
        self.norm3 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            _GEGLU(channels, channels * 4),
            nn.Linear(channels * 4, channels),
        )

    @override
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """Run the three sub-layers.

        Parameters
        ----------
        x : Tensor
            ``(B, T, channels)`` — the flattened feature map.
        context : Tensor
            ``(B, L, context_dim)`` — the conditioning sequence.

        Returns
        -------
        Tensor
            ``(B, T, channels)``.
        """
        normed = self.norm1(x)
        attended, _ = self.attn1(normed, normed, normed, need_weights=False)
        x = x + attended + self.attn1_out_bias

        # The load-bearing line: query from the image, key/value from the
        # conditioning.  Reversed, this still runs and still trains.
        query = self.norm2(x)
        crossed, _ = self.attn2(query, context, context, need_weights=False)
        x = x + crossed + self.attn2_out_bias

        return x + self.ff(self.norm3(x))


@final
class _SpatialTransformer(nn.Module):
    """Flatten a feature map, run transformer blocks, restore the shape.

    Parameters
    ----------
    channels, heads, context_dim, groups : int
        See :class:`_TransformerBlock`.
    depth : int, default=1
        Transformer blocks at this resolution.

    Notes
    -----
    The projections in and out are ``1 x 1`` convolutions rather than
    linear layers, which is the same map written so the framework can
    keep the tensor in ``NCHW`` until the flatten.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        context_dim: int,
        groups: int,
        depth: int = 1,
    ) -> None:
        """Initialise the module. See the class docstring for parameters."""
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, eps=_ATTENTION_NORM_EPS)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(channels, heads, context_dim) for _ in range(depth)]
        )
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    @override
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """Apply the transformer over spatial positions.

        Parameters
        ----------
        x : Tensor
            ``(B, C, H, W)``.
        context : Tensor
            ``(B, L, context_dim)``.

        Returns
        -------
        Tensor
            ``(B, C, H, W)``.
        """
        residual = x
        b, c, h, w = (int(s) for s in x.shape)
        y = self.proj_in(self.norm(x))
        y = y.reshape(b, c, h * w).swapaxes(1, 2)
        for block in self.blocks:
            y = block(y, context)
        y = y.swapaxes(1, 2).reshape(b, c, h, w)
        return residual + self.proj_out(y)


class UNet2DConditionModel(nn.Module):
    r"""The denoiser :math:`\epsilon_\theta(z_t, t, \tau_\theta(y))`.

    Parameters
    ----------
    config : StableDiffusionConfig
        Read for the U-Net fields.

    Notes
    -----
    Reference: Rombach et al., CVPR 2022 (arXiv:2112.10752), §3.3.

    Cross-attention runs at every resolution except the deepest
    down-sampling stage, which is the released arrangement
    (``CrossAttnDownBlock2D`` three times, then ``DownBlock2D``).

    ``attention_head_dim`` is read as a head **count**, not a dimension,
    despite its name. The released configuration says ``8`` and the
    reference builds eight heads of forty channels at the 320-wide
    stage — not forty heads of eight. Reading it the other way costs
    nothing in parameters, so the counts still match tensor for tensor;
    only the activations disagree, which is how it was found.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.stable_diffusion import (
    ...     StableDiffusionConfig, UNet2DConditionModel)
    >>> config = StableDiffusionConfig(sample_size=32, downsample_factor=4,
    ...                                vae_block_out_channels=(32, 64, 64),
    ...                                unet_block_out_channels=(32, 64),
    ...                                attention_head_dim=32,
    ...                                cross_attention_dim=16, context_length=4)
    >>> unet = UNet2DConditionModel(config).eval()
    >>> noise = unet(lucid.randn((1, 4, 8, 8)), lucid.tensor([10.0]),
    ...              lucid.randn((1, 4, 16)))
    >>> noise.shape
    (1, 4, 8, 8)
    """

    def __init__(self, config: StableDiffusionConfig) -> None:
        """Initialise the U-Net. See the class docstring for parameters."""
        super().__init__()
        self.config = config
        widths = config.unet_block_out_channels
        groups = config.norm_num_groups
        heads = config.attention_head_dim
        context_dim = config.cross_attention_dim
        time_dim = widths[0] * 4
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(widths[0], time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.conv_in = nn.Conv2d(config.latent_channels, widths[0], kernel_size=3, padding=1)

        # Down path.  ``_attn`` mirrors ``down`` and holds either a
        # transformer or None, so the forward can stay a plain zip.
        self.down_blocks: nn.ModuleList[_ResBlock] = nn.ModuleList()
        self.down_attns: nn.ModuleList[_SpatialTransformer | nn.Identity] = nn.ModuleList()
        self.downsamplers: nn.ModuleList[nn.Conv2d | nn.Identity] = nn.ModuleList()
        self.skip_channels: list[int] = [widths[0]]
        current = widths[0]
        for level, width in enumerate(widths):
            last = level == len(widths) - 1
            for _ in range(config.unet_layers_per_block):
                self.down_blocks.append(_ResBlock(current, width, time_dim, groups))
                current = width
                self.down_attns.append(
                    nn.Identity()
                    if last
                    else _SpatialTransformer(width, heads, context_dim, groups)
                )
                self.skip_channels.append(current)
            if not last:
                self.downsamplers.append(
                    nn.Conv2d(current, current, kernel_size=3, stride=2, padding=1)
                )
                self.skip_channels.append(current)
            else:
                self.downsamplers.append(nn.Identity())

        self.mid_block_1 = _ResBlock(current, current, time_dim, groups)
        self.mid_attn = _SpatialTransformer(current, heads, context_dim, groups)
        self.mid_block_2 = _ResBlock(current, current, time_dim, groups)

        # Up path — one extra block per level for the extra skip.
        self.up_blocks: nn.ModuleList[_ResBlock] = nn.ModuleList()
        self.up_attns: nn.ModuleList[_SpatialTransformer | nn.Identity] = nn.ModuleList()
        self.upsamplers: nn.ModuleList[nn.Conv2d | nn.Identity] = nn.ModuleList()
        skips = list(self.skip_channels)
        for level, width in enumerate(reversed(widths)):
            first = level == 0
            for _ in range(config.unet_layers_per_block + 1):
                self.up_blocks.append(_ResBlock(current + skips.pop(), width, time_dim, groups))
                current = width
                self.up_attns.append(
                    nn.Identity()
                    if first
                    else _SpatialTransformer(width, heads, context_dim, groups)
                )
            self.upsamplers.append(
                nn.Identity()
                if level == len(widths) - 1
                else nn.Conv2d(current, current, kernel_size=3, padding=1)
            )

        self.norm_out = nn.GroupNorm(groups, current)
        self.conv_out = nn.Conv2d(current, config.latent_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")

    @override
    def forward(self, latent: Tensor, timestep: Tensor, context: Tensor) -> Tensor:
        """Predict the noise in ``latent`` at ``timestep``.

        Parameters
        ----------
        latent : Tensor
            ``(B, latent_channels, h, w)``.
        timestep : Tensor
            ``(B,)`` or a scalar; broadcast to the batch.
        context : Tensor
            ``(B, L, cross_attention_dim)`` — :math:`\\tau_\\theta(y)`.

        Returns
        -------
        Tensor
            ``(B, latent_channels, h, w)``.
        """
        batch = int(latent.shape[0])
        steps = timestep.reshape(-1)
        if int(steps.shape[0]) == 1 and batch > 1:
            steps = steps.repeat(batch)
        # The released ``[cos, sin]`` order (``flip_sin_to_cos``), computed in
        # float32 at least and handed to the time MLP in the U-Net's dtype.
        emb = self.time_mlp(
            timestep_embedding(
                steps,
                self.config.unet_block_out_channels[0],
                dtype=self.conv_in.weight.dtype,
            )
        )
        if int(context.shape[-1]) != self.config.cross_attention_dim:
            raise ValueError(
                f"context width {int(context.shape[-1])} does not match "
                f"cross_attention_dim {self.config.cross_attention_dim}"
            )

        h = self.conv_in(latent)
        skips: list[Tensor] = [h]
        per_level = self.config.unet_layers_per_block
        index = 0
        for level in range(len(self.config.unet_block_out_channels)):
            for _ in range(per_level):
                h = self.down_blocks[index](h, emb)
                attn = self.down_attns[index]
                if not isinstance(attn, nn.Identity):
                    h = attn(h, context)
                skips.append(h)
                index += 1
            sampler = self.downsamplers[level]
            if not isinstance(sampler, nn.Identity):
                h = sampler(h)
                skips.append(h)

        h = self.mid_block_1(h, emb)
        h = self.mid_attn(h, context)
        h = self.mid_block_2(h, emb)

        index = 0
        for level in range(len(self.config.unet_block_out_channels)):
            for _ in range(per_level + 1):
                h = lucid.cat([h, skips.pop()], dim=1)
                h = self.up_blocks[index](h, emb)
                attn = self.up_attns[index]
                if not isinstance(attn, nn.Identity):
                    h = attn(h, context)
                index += 1
            sampler = self.upsamplers[level]
            if not isinstance(sampler, nn.Identity):
                h = sampler(self.upsample(h))

        return self.conv_out(self.act(self.norm_out(h)))
