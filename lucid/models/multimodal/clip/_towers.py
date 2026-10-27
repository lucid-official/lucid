"""Transformer building blocks shared by multimodal towers.

A two-tower model runs the same block over two modalities, and the only
thing that differs between the towers is whether the sequence has an
order to respect — text does, patches do not.  So the block is written
once here and parameterised by that, rather than twice in a family and
kept in step by hand.

These live in the family rather than one level up because CLIP is their
only consumer.  The domain layer holds what several families share; a
block with one user belongs to that user, and moves up when a second
appears — SigLIP and ALIGN would both want exactly this block.

They are not in :mod:`lucid.nn` for a different reason: they encode a
pre-norm-with-``QuickGELU`` shape because that is what the released
contrastive checkpoints were trained in.
:class:`lucid.nn.TransformerEncoderLayer` is the same idea with the
framework's own defaults, and it will not load these weights.
"""

from typing import final, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor

__all__ = ["QuickGELU", "ResidualAttentionBlock", "TransformerTower"]


@final
class QuickGELU(nn.Module):
    r"""``x * sigmoid(1.702 x)`` — the activation the released weights use.

    Notes
    -----
    Not the same function as :func:`lucid.nn.functional.gelu`.  It is a
    sigmoid approximation of it from a time when the exact form was
    expensive, and it survives because the published checkpoints were
    trained with it: the upstream configurations name it outright
    (``hidden_act: "quick_gelu"``).  Swapping in exact GELU shifts every
    activation slightly, which no shape test can see and which shows up
    as a quietly degraded score.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.multimodal.clip._towers import QuickGELU
    >>> float(QuickGELU()(lucid.tensor([0.0])).item())
    0.0
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply the activation.

        Parameters
        ----------
        x : Tensor
            Any shape.

        Returns
        -------
        Tensor
            The same shape.
        """
        return x * F.sigmoid(1.702 * x)


@final
class ResidualAttentionBlock(nn.Module):
    """Pre-norm Transformer block, used by both towers of a two-tower model.

    Parameters
    ----------
    width : int
        Residual stream width.
    heads : int
        Attention heads.  ``width`` must divide by it.
    causal : bool, default=False
        Mask future positions.  True for a text tower, false for an
        image one — patches have no order to respect, and masking them
        would hide half of every image from the class token.

    Attributes
    ----------
    ln_1, ln_2 : lucid.nn.LayerNorm
        Applied *before* attention and the MLP respectively.
    attn : lucid.nn.MultiheadAttention
        Self-attention with fused Q/K/V.
    mlp : lucid.nn.Sequential
        ``Linear → QuickGELU → Linear`` at four times the width.

    Notes
    -----
    Pre-norm rather than post-norm: the residual path stays unnormalised
    end to end, which is what lets these depths train without a warmup
    schedule tuned per variant.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.multimodal.clip._towers import ResidualAttentionBlock
    >>> block = ResidualAttentionBlock(32, 4).eval()
    >>> block(lucid.randn((2, 5, 32))).shape
    (2, 5, 32)
    """

    def __init__(self, width: int, heads: int, causal: bool = False) -> None:
        """Initialise the block. See the class docstring for parameters."""
        if heads < 1:
            raise ValueError(f"heads must be positive, got {heads}")
        if width % heads != 0:
            raise ValueError(f"width must be divisible by heads, got {width} and {heads}")
        super().__init__()
        self.causal = causal
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            QuickGELU(),
            nn.Linear(width * 4, width),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Attention and MLP, each on the residual stream.

        Parameters
        ----------
        x : Tensor
            ``(B, T, width)``.

        Returns
        -------
        Tensor
            ``(B, T, width)``.
        """
        normed = self.ln_1(x)
        attended, _ = self.attn(normed, normed, normed, need_weights=False, is_causal=self.causal)
        x = x + attended
        return x + self.mlp(self.ln_2(x))


@final
class TransformerTower(nn.Module):
    """A stack of :class:`ResidualAttentionBlock`.

    Parameters
    ----------
    width : int
        Residual stream width.
    layers : int
        Number of blocks.
    heads : int
        Attention heads per block.
    causal : bool, default=False
        Passed to every block.

    Attributes
    ----------
    resblocks : lucid.nn.ModuleList
        The blocks, in order.  Named for the released checkpoints, whose
        keys this reproduces.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.multimodal.clip._towers import TransformerTower
    >>> tower = TransformerTower(32, 2, 4, causal=True).eval()
    >>> tower(lucid.randn((1, 6, 32))).shape
    (1, 6, 32)
    """

    def __init__(self, width: int, layers: int, heads: int, causal: bool = False) -> None:
        """Initialise the stack. See the class docstring for parameters."""
        if layers < 1:
            raise ValueError(f"layers must be positive, got {layers}")
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks: nn.ModuleList[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(width, heads, causal) for _ in range(layers)]
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Run every block in order.

        Parameters
        ----------
        x : Tensor
            ``(B, T, width)``.

        Returns
        -------
        Tensor
            ``(B, T, width)``.
        """
        for block in self.resblocks:
            x = block(x)
        return x
