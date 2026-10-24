"""Positional encoding modules — Lucid-specific extensions to ``nn``.

Stateful (buffer-caching) wrappers around the pure functions in
:mod:`lucid.nn.functional.positional`.  Each module precomputes its table
once in ``__init__`` and exposes it via ``forward()`` or a registered
buffer so that ``.to(device=...)`` and serialisation Just Work.

Three variants are shipped:

    * :class:`SinusoidalEmbedding`   — Vaswani et al., 2017
    * :class:`SinusoidalEmbedding2D` — DETR §A.4
    * :class:`RotaryEmbedding`                 — Su et al., 2021

The reference framework has none of these in its ``nn`` namespace; every
model file reimplements them.  Lucid centralises them so families share
a single canonical implementation (and weight checkpoints don't drift
between siblings).
"""

from typing import override

import lucid
from lucid._dtype import _float32_or_wider
from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid.nn.functional.positional import (
    sinusoidal_embedding,
    sinusoidal_embedding_2d,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal positional embedding — 1D
# ─────────────────────────────────────────────────────────────────────────────


class SinusoidalEmbedding(Module):
    """Fixed 1-D sinusoidal positional encoding (Vaswani et al., 2017).

    Carries no learnable parameters — the encoding table is precomputed once
    in ``__init__`` and stored as a non-persistent buffer.  ``forward()``
    returns the full ``(num_positions, embedding_dim)`` table; callers
    typically slice it with ``pe()[:seq_len]`` and broadcast against their
    batched hidden states.

    Parameters
    ----------
    num_positions : int
        Maximum sequence length supported.
    embedding_dim : int
        Per-position embedding size; must be even.
    base : float, optional
        Frequency base :math:`\\theta_0` in the sin/cos formula.  ``10000``
        per Vaswani 2017; some long-context models use larger.  Default
        ``10000.0``.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> pe = nn.SinusoidalEmbedding(num_positions=512, embedding_dim=64)
    >>> table = pe()
    >>> table.shape
    (512, 64)

    Notes
    -----
    The encoding table is materialised once in ``__init__`` and registered
    as a non-persistent buffer; it is therefore not learnable and is not
    saved in ``state_dict`` (but moves with ``.to(device=...)``).  Prefer
    the module form when the same maximum sequence length is reused across
    many forward passes; use the functional
    :func:`lucid.nn.functional.sinusoidal_embedding` form for one-shot or
    variable-length encodings where caching is wasteful.  See that function
    for the underlying sin/cos formula.
    """

    pe: Tensor

    def __init__(
        self,
        num_positions: int,
        embedding_dim: int,
        *,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(
                f"SinusoidalEmbedding requires an even embedding_dim, "
                f"got {embedding_dim}"
            )
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim
        self.base = base
        table = sinusoidal_embedding(num_positions, embedding_dim, base=base)
        self.register_buffer("pe", table, persistent=False)

    @override
    def forward(self) -> Tensor:
        """Return the precomputed ``(num_positions, embedding_dim)`` table."""
        return self.pe


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal positional embedding — 2D (DETR-style)
# ─────────────────────────────────────────────────────────────────────────────


class SinusoidalEmbedding2D(Module):
    """Fixed 2-D sinusoidal positional encoding (DETR §A.4 / Carion 2020).

    Encodes spatial ``(row, column)`` coordinates: first half of the
    embedding dim encodes the column index, second half the row index.  Used
    in DETR and reusable for any image transformer that wants to inject
    absolute spatial position without learnable parameters.  ``forward()``
    returns the ``(H * W, embedding_dim)`` table in row-major order.

    Parameters
    ----------
    height : int
        Feature-map height :math:`H`.
    width : int
        Feature-map width :math:`W`.
    embedding_dim : int
        Per-position embedding size; must be divisible by 4 (half encodes
        column, half encodes row, each split into sin/cos).
    base : float, optional
        Frequency base.  DETR uses ``10000``.  Default ``10000.0``.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> pe = nn.SinusoidalEmbedding2D(height=16, width=24, embedding_dim=128)
    >>> pe().shape
    (384, 128)

    Notes
    -----
    The table is built once in ``__init__`` and registered as a
    non-persistent buffer, so it travels with ``.to(device=...)`` but is
    omitted from ``state_dict``.  Use the module form when ``(H, W)`` is
    fixed across forward passes (typical for DETR-style decoders fed by a
    fixed-size CNN feature map); fall back to the functional
    :func:`lucid.nn.functional.sinusoidal_embedding_2d` when the spatial
    grid varies per batch.  See that function for the row / column
    encoding split.
    """

    pe: Tensor

    def __init__(
        self,
        height: int,
        width: int,
        embedding_dim: int,
        *,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if embedding_dim % 4 != 0:
            raise ValueError(
                f"SinusoidalEmbedding2D requires embedding_dim divisible "
                f"by 4, got {embedding_dim}"
            )
        self.height = height
        self.width = width
        self.embedding_dim = embedding_dim
        self.base = base
        table = sinusoidal_embedding_2d(height, width, embedding_dim, base=base)
        self.register_buffer("pe", table, persistent=False)

    @override
    def forward(self) -> Tensor:
        """Return the precomputed ``(H * W, embedding_dim)`` table."""
        return self.pe


# ─────────────────────────────────────────────────────────────────────────────
# Rotary position embedding (RoPE)
# ─────────────────────────────────────────────────────────────────────────────


class RotaryEmbedding(Module):
    """Precomputed cos / sin tables for rotary positional embedding.

    Owns no learnable parameters — a thin wrapper around two registered
    buffers so the tables move with ``.to(device=...)`` and serialise with
    the rest of the model state.  ``forward()`` returns the precomputed
    ``(cos, sin)`` pair; callers pass them into
    :func:`lucid.nn.functional.apply_rotary_emb` along with the query /
    key tensors.

    Parameters
    ----------
    head_dim : int
        Per-head feature dim :math:`d_\\text{head}` (must be even).
    max_position_embeddings : int
        Largest sequence length the model will see.
    base : float, optional
        Frequency base :math:`\\theta_0` in the formula
        :math:`\\theta_i = \\text{base}^{-2 i / d_\\text{head}}`.
        Default ``10000.0`` per the RoFormer / LLaMA / GPT-NeoX
        convention; some long-context models (e.g. CodeLlama) use
        ``1_000_000``.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> rope = nn.RotaryEmbedding(head_dim=64, max_position_embeddings=2048)
    >>> cos, sin = rope()
    >>> cos.shape, sin.shape
    ((2048, 64), (2048, 64))

    Notes
    -----
    The ``cos_cached`` / ``sin_cached`` tables are built once at
    construction and registered as non-persistent buffers.  They follow
    ``.to(device=...)`` automatically but are not saved in ``state_dict``
    (RoPE has no learnable state — regenerate at load time).  The module
    form is the right choice for any transformer that reuses the same
    ``max_position_embeddings`` across calls; the functional
    :func:`lucid.nn.functional.apply_rotary_emb` consumes the cached
    pair and applies the rotation in place to ``q`` and ``k``.  See
    that function for the rotation math.
    """

    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"RotaryEmbedding requires an even head_dim, got {head_dim}"
            )
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # inv_freq[i] = base ** (-2 i / d_head) for i = 0 .. d_head/2 - 1
        half = head_dim // 2
        inv_freq_vals: list[float] = [
            1.0 / (base ** (2.0 * i / head_dim)) for i in range(half)
        ]
        inv_freq = lucid.tensor(inv_freq_vals)

        # Outer product positions × inv_freq → (max_pos, head_dim / 2).
        pos = lucid.arange(max_position_embeddings).float()
        freqs = pos.unsqueeze(1) * inv_freq.unsqueeze(0)

        # Duplicate to head_dim so cos / sin broadcast against (..., head_dim).
        # HF convention: concatenate along the last dim, not interleave.
        emb = lucid.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", lucid.cos(emb), persistent=False)
        self.register_buffer("sin_cached", lucid.sin(emb), persistent=False)

    @override
    def forward(self) -> tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` lookup tables.

        Shapes: each ``(max_position_embeddings, head_dim)``.  Callers index
        into them with the position ids of the current minibatch.
        """
        return self.cos_cached, self.sin_cached


# ─────────────────────────────────────────────────────────────────────────────
# Timestep embedding — diffusion-model standard (Ho et al., 2020)
# ─────────────────────────────────────────────────────────────────────────────


class TimestepEmbedding(Module):
    """Sinusoidal-frequency embedding of integer timesteps + 2-layer MLP.

    Diffusion U-Nets condition every residual block on the current timestep
    ``t``.  The canonical recipe (Ho et al., 2020 §3.2):

        emb(t) = MLP(sinusoidal_embedding(t, dim))

    where the sinusoidal part uses the same half-sin / half-cos formula as
    :class:`SinusoidalEmbedding` but is *queried per scalar* ``t``, not
    looked up by position index.  Every diffusion model reimplements this
    — Lucid centralises it so VAE / DDPM / NCSN share one canonical layer.

    Parameters
    ----------
    in_dim : int
        Dimension of the raw sinusoidal embedding.  Must be even.
    out_dim : int
        Dimension of the projected output — the conditioning vector
        consumed by U-Net residual blocks.  Often ``4 * in_dim``.
    base : float, optional
        Frequency base for the sinusoidal table.  Default ``10_000.0``
        per the original Transformer convention.

    Notes
    -----
    The output is **not** a learnable position table — only the two
    linear layers of the MLP are trainable.  Different ``timesteps``
    values produce different conditioning vectors via the deterministic
    sinusoidal lookup followed by the learned projection.  For DDPM-style
    training where each step samples a random ``t``, the layer adds
    ``2 * in_dim * out_dim`` parameters total.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn import TimestepEmbedding
    >>> emb = TimestepEmbedding(in_dim=128, out_dim=512)
    >>> t = lucid.tensor([0, 250, 500, 999])         # batch of 4 timesteps
    >>> cond = emb(t)
    >>> cond.shape
    (4, 512)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        base: float = 10_000.0,
        flip_sin_to_cos: bool = True,
    ) -> None:
        super().__init__()
        if in_dim % 2 != 0:
            raise ValueError(f"TimestepEmbedding requires an even in_dim, got {in_dim}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base = base
        # ``True`` → ``[cos, sin]`` (Lucid default); ``False`` → ``[sin, cos]``
        # (original DDPM / Ho 2020 + diffusers ``flip_sin_to_cos=False``).
        self.flip_sin_to_cos = flip_sin_to_cos

        # 2-layer MLP — Linear → SiLU → Linear (Ho et al. uses Swish).
        # SiLU lives in lucid.nn.modules.activation; import lazily to dodge
        # circular imports with this module being part of nn.modules itself.
        from lucid.nn.modules.activation import SiLU
        from lucid.nn.modules.linear import Linear

        self.linear_1 = Linear(in_dim, out_dim)
        self.act = SiLU()
        self.linear_2 = Linear(out_dim, out_dim)

    def _sinusoidal(self, timesteps: Tensor) -> Tensor:
        """Compute ``(*timesteps.shape, in_dim)`` sinusoidal embedding.

        Equivalent to Vaswani's 1-D PE but queried per *scalar* ``t``
        (typically with ``t`` continuously varying in [0, num_train_timesteps),
        not bounded by ``num_positions`` like SinusoidalEmbedding's table).

        The angles reach ``num_train_timesteps`` radians, which float16
        cannot hold to a useful fraction, so they are computed in float32
        (float64 for a float64 layer) and the embedding is handed to the MLP
        in its weights' dtype — the cast the reference U-Nets make before
        their time MLP.  Without it a float16 model ran its conditioning
        path widened to float32.
        """
        half = self.in_dim // 2
        # Frequencies: ω_i = base ** (-i / (half - 1))    for i = 0 … half-1
        denom = float(half - 1) if half > 1 else 1.0
        inv_freq_vals: list[float] = [
            1.0 / (self.base ** (i / denom)) for i in range(half)
        ]
        out_dtype = self.linear_1.weight.dtype
        work = _float32_or_wider(out_dtype)
        inv_freq = lucid.tensor(inv_freq_vals, dtype=work, device=timesteps.device.type)
        # outer product (* …, half)
        t = timesteps.to(work).unsqueeze(-1)  # (*shape, 1)
        # broadcast (*shape, 1) × (half,) → (*shape, half)
        while inv_freq.ndim < t.ndim:
            inv_freq = inv_freq.unsqueeze(0)
        ang = t * inv_freq
        if self.flip_sin_to_cos:
            emb = lucid.cat([lucid.cos(ang), lucid.sin(ang)], dim=-1)
        else:
            emb = lucid.cat([lucid.sin(ang), lucid.cos(ang)], dim=-1)
        return emb.to(out_dtype)

    @override
    def forward(self, timesteps: Tensor) -> Tensor:
        """Project ``timesteps`` into an ``(*timesteps.shape, out_dim)``
        conditioning vector."""
        emb = self._sinusoidal(timesteps)
        emb = self.linear_1(emb)
        emb = self.act(emb)
        return self.linear_2(emb)
