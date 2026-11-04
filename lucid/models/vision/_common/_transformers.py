"""The transformer the JEPA families share.

I-JEPA reads patches of an image and V-JEPA reads tubelets of a clip, but
between the tokeniser and the loss they are the same network: a pre-norm
block with a fused QKV projection, and the sine/cosine arithmetic their
position tables are built from.  What differs — how pixels become tokens,
how many axes a position has, what a mask covers — stays in the family
that means it.

The names follow the released implementations of both papers
(``blocks.N.attn.qkv``, ``mlp.fc1``), so a published checkpoint maps onto
either model by prefix alone.
"""

import math
from typing import override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor

__all__ = [
    "Attention",
    "MLP",
    "Block",
    "sincos_embedding",
    "gather_tokens",
    "init_transformer_weights",
    "scale_residual_projection",
]


def sincos_embedding(positions: Tensor, dim: int) -> Tensor:
    """Sine/cosine embedding of one coordinate, ``(N,) -> (N, dim)``.

    Built at double precision and cast down at the end.  The frequencies
    span four decades, so evaluating ``10000 ** k`` in float32 moves the
    angles by about 1e-4 — negligible on its own, but this table is the
    input to every block, and at float32 it sat a visible distance from
    the released checkpoint's own table.  Double lives on the host, so the
    arithmetic runs there whatever device ``positions`` is on.

    Parameters
    ----------
    positions : Tensor
        Coordinates along one axis, ``(N,)``.
    dim : int
        Width of the embedding; half of it is sines and half cosines.

    Returns
    -------
    Tensor
        ``(N, dim)``, in float32, on the device of ``positions``.
    """
    exponents = lucid.arange(dim // 2, dtype=lucid.float64, device="cpu")
    omega = 1.0 / (10000.0 ** (exponents * (2.0 / dim)))
    angles = positions.to("cpu").to(lucid.float64).reshape(-1, 1) * omega.reshape(1, -1)
    table = lucid.cat([lucid.sin(angles), lucid.cos(angles)], dim=1)
    return table.to(lucid.float32).to(positions.device)


def gather_tokens(tokens: Tensor, indices: Tensor) -> Tensor:
    """Pick tokens per batch element.

    Takes ``(B, N, D)`` tokens and ``(B, K)`` indices and returns the
    ``(B, K, D)`` tokens named, each row of the batch choosing its own.
    Masking in both JEPA families is exactly this: the tokens not chosen
    never reach the encoder at all.

    Parameters
    ----------
    tokens : Tensor
        The full sequence, ``(B, N, D)``.
    indices : Tensor
        Which tokens to keep, ``(B, K)``, one row per batch element.

    Returns
    -------
    Tensor
        ``(B, K, D)``.
    """
    width = int(tokens.shape[2])
    # The broadcast helper must sit with the tokens it gathers from: left
    # on the CPU it made every masked forward raise on a Metal model.
    spread = indices.unsqueeze(dim=-1) + lucid.zeros(
        int(indices.shape[0]),
        int(indices.shape[1]),
        width,
        dtype=indices.dtype,
        device=tokens.device,
    )
    return lucid.gather(tokens, 1, spread)


class Attention(nn.Module):
    """Multi-head self-attention with a fused QKV projection.

    One linear layer produces queries, keys and values together, which is
    how both released JEPA implementations store them — ``attn.qkv`` and
    ``attn.proj`` — so a published checkpoint loads without renaming.

    Parameters
    ----------
    dim : int
        Width of the input and output, divisible by ``num_heads``.
    num_heads : int
        Attention heads; each gets ``dim // num_heads`` channels.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    @override
    def forward(self, x: Tensor) -> Tensor:
        b, n, c = (int(s) for s in x.shape)
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        return self.proj(out)


class MLP(nn.Module):
    """The position-wise feed-forward of a transformer block.

    Two linear layers around a GELU, named ``fc1`` and ``fc2`` after the
    convention both released implementations use.

    Parameters
    ----------
    dim : int
        Width of the input and output.
    hidden : int
        Width between the two layers, usually four times ``dim``.
    """

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """A pre-norm transformer block: attention, then feed-forward.

    Layer norms come before each sub-layer and the residual carries the
    input around both, which is what lets a stack this deep train.  It is
    the whole of what I-JEPA and V-JEPA share between tokenising their
    input and comparing representations.

    Parameters
    ----------
    dim : int
        Width of the residual stream.
    num_heads : int
        Heads in the attention sub-layer.
    hidden : int
        Width inside the feed-forward sub-layer.
    eps : float
        Epsilon of both layer norms.
    """

    def __init__(self, dim: int, num_heads: int, hidden: int, eps: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = MLP(dim, hidden)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def init_transformer_weights(module: nn.Module, std: float = 0.02) -> None:
    """Draw a JEPA tower's weights the way the released code draws them.

    Both implementations initialise *every* linear and convolutional
    weight from the same narrow truncated normal, zero every bias, and
    leave layer norms at unit weight and zero bias.  Left at fan-in
    defaults the tower starts several times wider than that, which only a
    from-scratch run notices — and a run like this one is peculiarly
    exposed to it, because the targets come from an averaged copy of the
    encoder itself.  A tower that starts too wide is chasing a target it
    is producing.

    Parameters
    ----------
    module : Module
        Root of the subtree to initialise; every descendant is visited.
    std : float, optional
        Width of the draw, 0.02 in both papers' code.
    """

    def draw(layer: nn.Module) -> None:
        if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(layer.weight, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            if layer.weight is not None:
                nn.init.ones_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    module.apply(draw)


def scale_residual_projection(
    attention_out: nn.Linear, mlp_out: nn.Linear, depth_index: int, std: float = 0.02
) -> None:
    """Narrow the two projections a block writes into the residual with.

    Every block adds to the residual stream twice, so the stream widens
    as the stack deepens unless the things being added are narrowed to
    match.  Both released implementations divide block ``i``'s two output
    projections by ``sqrt(2 * i)``, counting from one — so the deepest
    block of a 24-layer tower contributes about a seventh of what the
    first does.  Drawing at the reduced width is that same distribution,
    and is how this zoo already writes it elsewhere (GPT-2 §2.3).

    Parameters
    ----------
    attention_out : Linear
        The block's ``attn.proj``.
    mlp_out : Linear
        The block's ``mlp.fc2``.
    depth_index : int
        Where the block sits in the stack, counting from one.
    std : float, optional
        Width before the depth factor is applied.
    """
    if depth_index < 1:
        raise ValueError(f"depth_index counts from one, got {depth_index}")
    factor = 1.0 / math.sqrt(2.0 * float(depth_index))
    nn.init.trunc_normal_(attention_out.weight, std=std * factor)
    nn.init.trunc_normal_(mlp_out.weight, std=std * factor)
