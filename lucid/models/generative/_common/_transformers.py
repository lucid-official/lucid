r"""The DiT block, shared by the families whose backbone is one.

Peebles and Xie's Diffusion Transformer replaced the U-Net of a latent
diffusion model with a plain ViT stack, and the part that made it work is
not the stack but how the conditioning enters it: **adaLN-Zero**.  Each
block regresses a shift, a scale and a *gate* per sub-layer from the
conditioning vector, and the projection producing them starts at zero, so
every block begins as the identity on the residual stream.  The paper
compares four conditioning designs and this one wins at every training
budget — at 400K steps its FID is close to half the in-context variant's.

It lives at the domain layer because two families use it and neither owns
it: :mod:`lucid.models.generative.dit` is where it comes from, and
:mod:`lucid.models.generative.mean_flow` says outright that its network
"is DiT's, unchanged".  What differs between them is only what produces
the conditioning vector — a timestep and a class for DiT, two times and a
class for MeanFlow — and that stays in each family.

The same placement rule the world-model families follow: one consumer
keeps a module inside its family, two or more move it here.
"""

import math
from typing import cast, override

import lucid
import lucid.nn as nn
from lucid._dtype import _float32_or_wider
from lucid._tensor.tensor import Tensor

__all__ = [
    "timestep_embedding",
    "sincos_position_embedding",
    "unpatchify",
    "DiTBlock",
    "DiTFinalLayer",
]


def timestep_embedding(
    t: Tensor,
    dim: int,
    max_period: float = 10000.0,
    dtype: lucid.dtype | None = None,
) -> Tensor:
    r"""Sinusoidal embedding of a scalar, as in the transformer.

    Parameters
    ----------
    t : Tensor
        ``(B,)`` of values.  Diffusion timesteps are integral and flow
        times are continuous; nothing here cares which.
    dim : int
        Width of the embedding; the halves hold cosine and sine.
    max_period : float, default=10000.0
        Longest wavelength in the ladder.
    dtype : lucid.dtype or None, optional
        Dtype of the embedding — the consuming MLP's.  The angles are
        computed in float32 at least either way (a diffusion timestep
        reaches a thousand radians, which float16 cannot carry); ``None``
        returns them in that working dtype.

    Returns
    -------
    Tensor
        ``(B, dim)``.

    Notes
    -----
    The ladder divides by ``half``, which is ADM's convention and the one
    DiT inherits.  It is worth stating because the other convention is
    equally common: DDPM divides by ``half - 1``, and so does the
    ``diffusers`` export of DiT's own published checkpoints — measured,
    not assumed, and it moves the embedding by as much as 0.8.

    The released weights want this one.  Asked to denoise a latent whose
    answer is known, ``DiT-XL/2`` predicts the noise more accurately with
    ``half`` at every timestep tried — mean squared error 0.0336 against
    0.0373 at :math:`t=200`, and 0.00009 against 0.00018 at
    :math:`t=800`.  So the export feeds the paper's network a ladder it
    was not trained on, and matching the paper is both the more faithful
    choice and the better-calibrated one.  Everything else in the
    conversion agrees with that export to float32 noise, which is what
    isolates this as the only difference.  Changing the line silently
    changes what a ported checkpoint computes, so a test pins it.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._transformers import timestep_embedding
    >>> timestep_embedding(lucid.tensor([0.0, 1.0]), 8).shape
    (2, 8)
    """
    half = dim // 2
    work = _float32_or_wider(t.dtype if dtype is None else dtype)
    freqs = lucid.exp(
        -math.log(max_period)
        * lucid.arange(half, dtype=work, device=t.device.type)
        / float(half)
    )
    args = t.to(work).reshape(-1, 1) * freqs.reshape(1, -1)
    emb = lucid.cat([lucid.cos(args), lucid.sin(args)], dim=-1)
    if dim % 2:
        emb = lucid.cat([emb, lucid.zeros_like(emb[:, :1])], dim=-1)
    return emb if dtype is None else emb.to(dtype)


def sincos_position_embedding(dim: int, side: int) -> Tensor:
    """Fixed two-dimensional sine-cosine table over a ``side x side`` grid.

    DiT freezes its positional embedding rather than learning it — the
    paper calls them "standard ViT frequency-based positional embeddings".
    Half the width encodes one axis and half the other, so ``dim`` must be
    divisible by four.

    **The column half comes first.**  The reference implementation builds
    its two halves from ``numpy.meshgrid(w, h)``, whose first output is
    the *column* coordinate — so despite the variable being named for the
    row, the leading half of every embedding is the one that varies
    fastest down the flattened sequence.  Reproducing that order is not
    cosmetic: the halves are interchangeable only if nothing was ever
    trained against them, and swapping them silently corrupts every
    ported checkpoint.

    Parameters
    ----------
    dim : int
        Embedding width.  Must be divisible by 4.
    side : int
        Tokens along one axis of the square grid.

    Returns
    -------
    Tensor
        ``(1, side * side, dim)``, ready to add to a token stream.

    Raises
    ------
    ValueError
        If ``dim`` is not divisible by 4.

    Examples
    --------
    >>> from lucid.models.generative._common._transformers import (
    ...     sincos_position_embedding)
    >>> sincos_position_embedding(16, 4).shape
    (1, 16, 16)
    """
    if dim % 4 != 0:
        raise ValueError(
            f"dim must be divisible by 4 — the table splits it into sine and "
            f"cosine over each of two axes, got {dim}"
        )
    quarter = dim // 4
    omega = 1.0 / (10000.0 ** (lucid.arange(quarter, dtype=lucid.float32) / quarter))
    pos = lucid.arange(side, dtype=lucid.float32)
    out = pos.reshape(-1, 1) * omega.reshape(1, -1)
    axis = lucid.cat([lucid.sin(out), lucid.cos(out)], dim=1)

    rows = axis.reshape(side, 1, -1) + lucid.zeros((side, side, 1))
    cols = axis.reshape(1, side, -1) + lucid.zeros((side, side, 1))
    # Columns lead — see the note above.  Row-major flattening makes this
    # the half that changes between adjacent tokens.
    return lucid.cat([cols, rows], dim=-1).reshape(1, side * side, -1)


def unpatchify(x: Tensor, grid: int, patch: int, channels: int) -> Tensor:
    """Put ``(B, N, patch^2 * channels)`` tokens back as ``(B, channels, H, W)``.

    The inverse of a stride-``patch`` patch embedding over a ``grid x grid``
    token grid, row-major — DiT's ``unpatchify``.

    Parameters
    ----------
    x : Tensor
        ``(B, grid * grid, patch * patch * channels)`` decoder output.
    grid : int
        Tokens per side.
    patch : int
        Side of a patch.
    channels : int
        Channels per pixel.

    Returns
    -------
    Tensor
        ``(B, channels, grid * patch, grid * patch)``.
    """
    x = x.reshape(-1, grid, grid, patch, patch, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(-1, channels, grid * patch, grid * patch)


class DiTBlock(nn.Module):
    r"""A transformer block whose norms and residuals the conditioning drives.

    Parameters
    ----------
    hidden_size : int
        Residual stream width.
    num_heads : int
        Attention heads.
    mlp_ratio : float, default=4.0
        Feed-forward expansion.
    gated : bool, default=True
        Whether to regress the residual gate.  ``True`` is adaLN-Zero's
        block; ``False`` is plain adaLN, which the paper describes as
        regressing the shift and scale only.  The distinction is not
        cosmetic — the gate is the entire difference between the two
        designs the paper compares, and a build that gave both the gate
        could not reproduce that comparison.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748), Section 3.2.

    Six vectors come out of the conditioning per gated block — a shift, a
    scale and a gate for each of the two sub-layers; four when ungated.
    The gate is what "Zero" refers to: the paper regresses
    "dimension-wise scaling parameters :math:`\alpha` that are applied
    immediately prior to any residual connections" and initialises the
    projection so they start at zero.  Every gated block is then the
    identity at initialisation, which is what lets depth grow without
    re-tuning the schedule.

    Zeroing is not done here — a block does not know whether it is being
    built fresh or loaded from a checkpoint.  The owning model calls
    :meth:`zero_conditioning` after construction.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._transformers import DiTBlock
    >>> block = DiTBlock(32, 4).eval()
    >>> tokens = lucid.randn((2, 9, 32))
    >>> block(tokens, lucid.randn((2, 32))).shape
    (2, 9, 32)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        gated: bool = True,
    ) -> None:
        """Initialise the block. See the class docstring for parameters."""
        super().__init__()
        self.gated = gated
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        inner = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner),
            # DiT's reference implementation uses the tanh approximation
            # — `nn.GELU(approximate="tanh")` in the released code, and
            # `"gelu-approximate"` in the published checkpoint's config.
            # Exact GELU here would be a quiet parity gap the moment
            # those weights are loaded.
            nn.GELU(approximate="tanh"),
            nn.Linear(inner, hidden_size),
        )
        fan = 6 if gated else 4
        self.ada_ln = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, fan * hidden_size)
        )

    def zero_conditioning(self) -> None:
        """Zero the modulation projection — the "Zero" of adaLN-Zero."""
        with lucid.no_grad():
            linear = cast(nn.Linear, self.ada_ln[1])
            linear.weight.zero_()
            if linear.bias is not None:
                linear.bias.zero_()

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Attention and MLP, each modulated and gated by ``cond``.

        Parameters
        ----------
        x : Tensor
            ``(B, N, D)`` token stream.
        cond : Tensor
            ``(B, D)`` conditioning vector.

        Returns
        -------
        Tensor
            ``(B, N, D)``.
        """
        params = self.ada_ln(cond)
        width = int(x.shape[2])
        fan = 6 if self.gated else 4
        chunks = [
            params[:, i * width : (i + 1) * width].reshape(-1, 1, width)
            for i in range(fan)
        ]
        if self.gated:
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = chunks
        else:
            shift_a, scale_a, shift_m, scale_m = chunks
            gate_a = gate_m = lucid.ones((1, 1, 1), device=x.device.type)

        h = self.norm1(x) * (1.0 + scale_a) + shift_a
        attended, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a * attended

        h = self.norm2(x) * (1.0 + scale_m) + shift_m
        return x + gate_m * self.mlp(h)


class DiTFinalLayer(nn.Module):
    r"""Modulated norm and the projection back to patch space.

    Parameters
    ----------
    hidden_size : int
        Residual stream width.
    patch_size : int
        Side of a patch.
    out_channels : int
        Channels per pixel the decoder emits.  DiT passes ``2 * C``
        because it predicts a noise *and* a diagonal covariance; a family
        that predicts one field passes ``C``.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023, Section 3.2 —
    "we apply the final layer norm (adaptive if using adaLN) and linearly
    decode each token into a ``p x p x 2C`` tensor".

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._transformers import DiTFinalLayer
    >>> head = DiTFinalLayer(32, 2, 4).eval()
    >>> head(lucid.randn((1, 9, 32)), lucid.randn((1, 32))).shape
    (1, 9, 16)
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        """Initialise the head. See the class docstring for parameters."""
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def zero_conditioning(self) -> None:
        """Zero the modulation projection and the decoder.

        The paper initialises the final linear layer with zeros too, so an
        untrained model predicts a zero field rather than noise.
        """
        with lucid.no_grad():
            linear = cast(nn.Linear, self.ada_ln[1])
            linear.weight.zero_()
            if linear.bias is not None:
                linear.bias.zero_()
            self.proj.weight.zero_()
            if self.proj.bias is not None:
                self.proj.bias.zero_()

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Project ``(B, N, D)`` tokens to ``(B, N, patchˆ2 * out_channels)``."""
        params = self.ada_ln(cond)
        width = int(x.shape[2])
        shift = params[:, :width].reshape(-1, 1, width)
        scale = params[:, width:].reshape(-1, 1, width)
        h = self.norm(x) * (1.0 + scale) + shift
        return self.proj(h)
