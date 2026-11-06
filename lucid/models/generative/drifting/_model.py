r"""Drifting Models — a one-step generator whose iteration happens in training.

Four pieces, each a one-to-one port of the official JAX release
(``lambertae/drifting``):

**The generator** (:class:`DriftingModel`) is the release's ``DitGen`` around
a ``LightningDiT``: a DiT whose blocks use RMSNorm, QK-norm, SwiGLU and 1-D
RoPE over the whole sequence, with 16 in-context tokens prepended.  Its input
is noise in two parts — a Gaussian field and 32 random *style* indices into a
64-entry codebook whose embeddings are summed into the conditioning vector —
and its conditioning is the class plus the guidance scale :math:`\alpha`,
embedded like a timestep.  One evaluation produces a sample.

**The drifting loss** (:func:`_drift_loss`) is the release's ``drift_loss``:
pairwise distances, a global distance normalisation, a doubly-normalised
softmax kernel at several temperatures, the attraction-minus-repulsion
field of Alg. 2, a per-temperature drift normalisation, and an MSE onto the
stop-gradiented drifted target.  Everything up to the target runs without
gradient; the loss's gradient reaches the generator only through its own
sample.

**The feature space** (:class:`DriftingFeatureEncoder`) is the frozen encoder
the loss is computed in: the paper's ResNet-style MAE encoder (latent- or
pixel-MAE), and for pixel models ConvNeXt-V2-Base as well, each turned into
dozens of per-scale, per-location feature sets (A.5).  It is a training-time
network and is built on request, never with the generator.

**The sample queue** (:class:`DriftingSampleQueue`) is A.8's MoCo-style
per-class ring buffer of real samples, from which positives and
unconditional negatives are drawn.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._dtype import _float32_or_wider
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import DiffusionModelOutput, GenerationOutput, ModelOutput
from lucid.models._tasks import ImageGenerationModel
from lucid.models._utils._generative import resolve_generation_device, start_noise
from lucid.models.generative._common._transformers import (
    check_latent_size,
    sincos_position_embedding,
    timestep_embedding,
    unpatchify,
)
from lucid.models.generative.drifting._config import DriftingConfig
from lucid.models.generative.drifting._weights import DriftingConvNeXtV2Weights, DriftingMAEWeights
from lucid.utils.checkpoint import checkpoint
from lucid.weights import load_weight_entry

__all__ = [
    "DriftingModel",
    "DriftingForImageGeneration",
    "DriftingFeatureEncoder",
    "DriftingSampleQueue",
    "DriftingOutput",
]

# ``cond = class + styles + 0.02 * RMSNorm(embed(alpha))`` — a constant in
# the release's ``c_cfg_noise_to_cond``, not a config field there either.
_CFG_EMBED_SCALE = 0.02
# Added to a generated sample's distance to itself so it is never its own
# neighbour (``mask_val`` in the release).
_SELF_MASK = 100.0
# ImageNet statistics the release normalises by before ConvNeXt-V2.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# ConvNeXt-V2 reads 224-pixel images; the pixel generators emit 256.
_CONVNEXT_SIZE = 224


# ─────────────────────────────────────────────────────────────────────────────
# Generator building blocks (LightningDiT)
# ─────────────────────────────────────────────────────────────────────────────


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN modulation ``x * (1 + scale) + shift``, broadcast over tokens."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _norm(width: int, rms: bool, affine: bool) -> nn.Module:
    """The release's norm choice: RMSNorm (always affine) or LayerNorm, eps 1e-6.

    The block norms are ``LayerNorm(use_scale=False, use_bias=False)`` when
    RMSNorm is off; the QK norms are an affine LayerNorm.  RMSNorm carries a
    weight in both places.
    """
    if rms:
        return nn.RMSNorm(width, eps=1e-6)
    return nn.LayerNorm(width, eps=1e-6, elementwise_affine=affine)


class _CFGEmbedder(nn.Module):
    """The guidance scale's embedding: a timestep sinusoid, then a two-layer MLP.

    The release reuses DiT's ``TimestepEmbedder`` for :math:`\\alpha`
    (cosine half first, ladder divided by ``half``) — exactly the shared
    :func:`timestep_embedding`.
    """

    def __init__(self, cond_dim: int, frequency_embedding_size: int) -> None:
        """Initialise the embedder. See the class docstring."""
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.fc1 = nn.Linear(frequency_embedding_size, cond_dim)
        self.fc2 = nn.Linear(cond_dim, cond_dim)

    @override
    def forward(self, alpha: Tensor) -> Tensor:
        """Embed ``(B,)`` guidance scales into ``(B, cond_dim)``."""
        emb = timestep_embedding(alpha, self.frequency_embedding_size, dtype=self.fc1.weight.dtype)
        return self.fc2(F.silu(self.fc1(emb)))


class _Attention(nn.Module):
    """Multi-head self-attention with optional QK-norm and 1-D RoPE.

    Order matches the release: project, normalise ``q`` and ``k`` per head,
    rotate them, then scale and softmax — in float32 when ``attn_fp32``.
    """

    def __init__(
        self, dim: int, num_heads: int, qk_norm: bool, use_rmsnorm: bool, attn_fp32: bool
    ) -> None:
        """Initialise the attention. See the class docstring."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.attn_fp32 = attn_fp32
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        if qk_norm:
            self.q_norm = _norm(self.head_dim, use_rmsnorm, affine=True)
            self.k_norm = _norm(self.head_dim, use_rmsnorm, affine=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    @override
    def forward(self, x: Tensor, cos: Tensor | None = None, sin: Tensor | None = None) -> Tensor:
        """Attend over ``(B, N, C)``; ``cos``/``sin`` are the RoPE tables or ``None``."""
        b, n, c = (int(s) for s in x.shape)
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # ``(B, N, H, D)`` -> ``(B, H, N, D)``: the layout RoPE and SDPA expect.
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
        work = _float32_or_wider(x.dtype) if self.attn_fp32 else x.dtype
        q, k, v = q.to(work), k.to(work), v.to(work)
        if cos is not None and sin is not None:
            q, k = F.apply_rotary_emb(q, k, cos.to(work), sin.to(work))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c).to(x.dtype)
        return self.proj(out)


class _SwiGLU(nn.Module):
    """SwiGLU feed-forward, ``w2(silu(w1 x) * w3 x)``, all three with bias."""

    def __init__(self, dim: int, hidden: int) -> None:
        """Initialise the FFN. See the class docstring."""
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated FFN."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _MLP(nn.Module):
    """The non-SwiGLU path: Linear, exact GELU, Linear."""

    def __init__(self, dim: int, hidden: int) -> None:
        """Initialise the FFN. See the class docstring."""
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply the FFN."""
        return self.fc2(F.gelu(self.fc1(x)))


class _LightningDiTBlock(nn.Module):
    """adaLN-Zero transformer block — shift, scale and gate per sub-layer."""

    def __init__(self, config: DriftingConfig) -> None:
        """Initialise the block from the family config."""
        super().__init__()
        h = config.hidden_size
        self.norm1 = _norm(h, config.use_rmsnorm, affine=False)
        self.norm2 = _norm(h, config.use_rmsnorm, affine=False)
        self.attn = _Attention(
            h, config.num_heads, config.use_qknorm, config.use_rmsnorm, config.attn_fp32
        )
        self.mlp: _SwiGLU | _MLP = (
            _SwiGLU(h, config.swiglu_hidden_size)
            if config.use_swiglu
            else _MLP(h, int(h * config.mlp_ratio))
        )
        self.ada_ln = nn.Linear(config.cond_dim, 6 * h)

    @override
    def forward(
        self, x: Tensor, cond: Tensor, cos: Tensor | None = None, sin: Tensor | None = None
    ) -> Tensor:
        """Advance ``(B, N, H)`` tokens under ``(B, cond_dim)`` conditioning."""
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada_ln(F.silu(cond)).chunk(
            6, dim=1
        )
        x = x + gate_a.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_a, scale_a), cos, sin
        )
        return x + gate_m.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))


class _FinalLayer(nn.Module):
    """Modulated norm and the projection back to patch space."""

    def __init__(self, config: DriftingConfig) -> None:
        """Initialise the head from the family config."""
        super().__init__()
        h = config.hidden_size
        self.norm = _norm(h, config.use_rmsnorm, affine=False)
        self.ada_ln = nn.Linear(config.cond_dim, 2 * h)
        self.proj = nn.Linear(h, config.patch_size * config.patch_size * config.out_channels)

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Project ``(B, N, H)`` tokens to ``(B, N, p*p*C)``."""
        shift, scale = self.ada_ln(F.silu(cond)).chunk(2, dim=1)
        return self.proj(_modulate(self.norm(x), shift, scale))


class DriftingModel(PretrainedModel):
    r"""The one-step generator :math:`f_\theta(\epsilon, c, \alpha)`.

    Parameters
    ----------
    config : DriftingConfig
        The variant to build.

    Attributes
    ----------
    class_embed : lucid.nn.Embedding
        Class table, ``(num_classes, cond_dim)``.
    noise_embeds : lucid.nn.ModuleList
        ``noise_coords`` style codebooks, each ``(noise_classes, cond_dim)``.
    cfg_embedder : _CFGEmbedder
        Embedding of the guidance scale :math:`\alpha`.
    patch_embed : lucid.nn.Linear
        Patch projection; its input is a ``(p, p, C)``-ordered patch.
    pos_embed : lucid.nn.Parameter
        ``(1, num_patches, hidden_size)`` — *trainable*, initialised to the
        2-D sine-cosine table.
    cls_embed : lucid.nn.Parameter
        ``(1, n_cls_tokens, hidden_size)`` learnable in-context offsets.
    blocks : lucid.nn.ModuleList
        The transformer blocks.
    final : _FinalLayer
        Modulated norm and projection back to patches.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, *"Generative Modeling via
    Drifting"*, arXiv:2602.04770, 2026, Appendix A.2 and Table 8.

    The conditioning vector is the release's ``c_cfg_noise_to_cond``:

    .. math::

        c = E_\mathrm{cls}[y] + \sum_{i=1}^{32} E_i[s_i]
            + 0.02\,\mathrm{RMSNorm}\bigl(\mathrm{MLP}(\mathrm{sin}(\alpha))\bigr),

    and it drives both adaLN and the 16 in-context tokens, which are
    ``Linear(c) + cls_embed`` prepended before the first block.  RoPE runs
    over the flattened sequence, in-context tokens included, as in the
    release.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.drifting import DriftingConfig, DriftingModel
    >>> config = DriftingConfig(sample_size=8, hidden_size=32, cond_dim=32, depth=1,
    ...                         num_heads=4, num_classes=10, noise_classes=4,
    ...                         noise_coords=2, n_cls_tokens=2)
    >>> model = DriftingModel(config).eval()
    >>> model(lucid.randn(2, 4, 8, 8), lucid.tensor([0, 1])).sample.shape
    (2, 4, 8, 8)
    """

    config_class: ClassVar[type[DriftingConfig]] = DriftingConfig
    base_model_prefix = "drifting"

    def __init__(self, config: DriftingConfig) -> None:
        """Initialise the generator. See the class docstring for parameters."""
        super().__init__(config)
        self.config: DriftingConfig = config
        h, cond = config.hidden_size, config.cond_dim
        self.class_embed = nn.Embedding(config.num_classes, cond)
        self.noise_embeds = nn.ModuleList(
            [nn.Embedding(config.noise_classes, cond) for _ in range(config.noise_coords)]
            if config.noise_classes > 0
            else []
        )
        self.cfg_embedder = _CFGEmbedder(cond, config.frequency_embedding_size)
        self.cfg_norm = nn.RMSNorm(cond, eps=1e-6)

        patch_dim = config.patch_size * config.patch_size * config.in_channels
        self.patch_embed = nn.Linear(patch_dim, h)
        self.pos_embed = nn.Parameter(lucid.zeros(1, config.num_patches, h))
        if config.n_cls_tokens > 0:
            self.cls_proj = nn.Linear(cond, h)
            self.cls_embed = nn.Parameter(lucid.zeros(1, config.n_cls_tokens, h))
        if config.use_rope:
            self.rope = nn.RotaryEmbedding(
                config.head_dim, config.n_cls_tokens + config.num_patches
            )

        self.blocks = nn.ModuleList([_LightningDiTBlock(config) for _ in range(config.depth)])
        self.final = _FinalLayer(config)
        self._init_weights()

    def _init_weights(self) -> None:
        """The release's initialisers, one for one.

        The release's linear layer is Xavier-uniform with zero bias; the guidance MLP is
        ``normal(0.02)``; every adaLN projection and the output projection
        start at zero (adaLN-Zero); embedding tables and in-context offsets
        are ``normal(0.02)``; the position table starts as the sine-cosine
        grid, then trains.
        """
        with lucid.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, std=0.02)
            for linear in (self.cfg_embedder.fc1, self.cfg_embedder.fc2):
                nn.init.normal_(linear.weight, std=0.02)
            zeroed: list[nn.Linear] = [block.ada_ln for block in self.blocks]
            zeroed += [self.final.ada_ln, self.final.proj]
            for linear in zeroed:
                nn.init.zeros_(linear.weight)
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)
            self.pos_embed.copy_(
                sincos_position_embedding(self.config.hidden_size, self.config.grid_size)
            )
            if self.config.n_cls_tokens > 0:
                nn.init.normal_(self.cls_embed, std=0.02)

    def draw_style_indices(self, batch: int, device: str) -> Tensor:
        """The discrete half of the noise: ``(batch, noise_coords)`` codebook indices.

        Drawn uniformly, as the release does (``max(1, ·)`` keeps the shape
        valid when styles are disabled; the indices are then unused).

        Parameters
        ----------
        batch : int
            Number of samples to draw indices for.
        device : str
            Device the indices are created on.

        Returns
        -------
        Tensor
            ``int64`` indices of shape ``(batch, max(1, noise_coords))``, each
            in ``[0, max(1, noise_classes))``.
        """
        config = self.config
        return lucid.randint(
            0,
            max(1, config.noise_classes),
            (batch, max(1, config.noise_coords)),
            dtype=lucid.int64,
            device=device,
        )

    def conditioning(
        self, labels: Tensor, cfg_scale: Tensor, style_indices: Tensor, dtype: lucid.dtype
    ) -> Tensor:
        r"""Class, styles and guidance scale summed into ``(B, cond_dim)``."""
        cond = self.class_embed(labels)
        for i, embed in enumerate(self.noise_embeds):
            cond = cond + embed(style_indices[:, i])
        alpha = self.cfg_norm(self.cfg_embedder(cfg_scale))
        return (cond + alpha * _CFG_EMBED_SCALE).to(dtype)

    @override
    def forward(
        self,
        x: Tensor,
        labels: Tensor,
        cfg_scale: float | Tensor = 1.0,
        style_indices: Tensor | None = None,
    ) -> DiffusionModelOutput:
        r"""Map noise to a sample in one evaluation.

        Parameters
        ----------
        x : Tensor
            ``(B, in_channels, H, W)`` Gaussian noise (already scaled by the
            sampling temperature, if any).
        labels : Tensor
            ``(B,)`` class indices.
        cfg_scale : float or Tensor, default=1.0
            Guidance scale :math:`\alpha`, a scalar or ``(B,)``.  ``1`` is
            no guidance.
        style_indices : Tensor or None, optional
            ``(B, noise_coords)`` style indices.  Drawn uniformly when
            ``None``, as the release always does.

        Returns
        -------
        DiffusionModelOutput
            ``sample`` of shape ``(B, out_channels, H, W)``.
        """
        config = self.config
        b, c, height, width = (int(s) for s in x.shape)
        grid, p = config.grid_size, config.patch_size
        check_latent_size(height, width, grid, p, "Drifting")
        device = x.device.type

        if isinstance(cfg_scale, Tensor):
            alpha = cfg_scale.to(x.dtype).reshape(-1) + lucid.zeros(
                (b,), dtype=x.dtype, device=device
            )
        else:
            alpha = lucid.full((b,), float(cfg_scale), dtype=x.dtype, device=device)
        if style_indices is None:
            style_indices = self.draw_style_indices(b, device)
        cond = self.conditioning(labels, alpha, style_indices, x.dtype)

        # Patchify to the release's ``(p_h, p_w, C)`` inner order, which the
        # ported patch projection expects.
        tokens = x.reshape(b, c, grid, p, grid, p).permute(0, 2, 4, 3, 5, 1)
        tokens = self.patch_embed(tokens.reshape(b, grid * grid, p * p * c)) + self.pos_embed
        if config.n_cls_tokens > 0:
            in_context = self.cls_proj(cond).unsqueeze(1) + self.cls_embed
            tokens = lucid.cat([in_context, tokens], dim=1)

        rope: tuple[Tensor, ...] = (
            (self.rope.cos_cached, self.rope.sin_cached) if config.use_rope else ()
        )
        for block in self.blocks:
            if config.use_remat and self.training:
                recomputed = checkpoint(block, tokens, cond, *rope)
                # A block returns one tensor; checkpoint's return type also
                # admits the tuple a multi-output segment would give.
                assert isinstance(recomputed, Tensor)
                tokens = recomputed
            else:
                tokens = block(tokens, cond, *rope)

        out = self.final(tokens, cond)[:, config.n_cls_tokens :]
        image = unpatchify(out, grid, p, config.out_channels)
        return DiffusionModelOutput(sample=image)


# ─────────────────────────────────────────────────────────────────────────────
# The drifting field and its loss (``drift_loss.py``)
# ─────────────────────────────────────────────────────────────────────────────


def _pairwise_distance(x: Tensor, y: Tensor, same: Tensor, eps: float = 1e-8) -> Tensor:
    """``(B, N, D) x (B, M, D) -> (B, N, M)`` L2 distances, the release's ``cdist``.

    Expanded-square form clipped at ``eps`` before the root, so a sample's
    distance to itself is ``1e-4`` rather than zero — reproduced exactly
    because the global normalisation averages over it.  ``same`` is one on
    the ``(N, M)`` pairs that are one point twice, and their square is set
    to its exact zero.  Cancellation would leave whatever ulp the matmul and
    the sum round apart by, and the root magnifies it: on the CPU in float32
    a residue of ``4.8e-7`` became a self distance of ``6.9e-4``, which moved
    the normalisation by ``1e-5``.
    """
    xy = lucid.matmul(x, y.permute(0, 2, 1))
    xx = (x * x).sum(dim=-1)
    yy = (y * y).sum(dim=-1)
    square = (xx.unsqueeze(2) + yy.unsqueeze(1) - 2.0 * xy) * (1.0 - same)
    return lucid.sqrt(lucid.clamp(square, min=eps))


def _drift_loss(
    gen: Tensor,
    pos: Tensor,
    neg: Tensor | None,
    weight_neg: Tensor | None,
    temperatures: tuple[float, ...],
) -> tuple[Tensor, dict[str, Tensor]]:
    r"""The drifting loss of one feature, batched over independent groups.

    Parameters
    ----------
    gen : Tensor
        ``(B, C_g, S)`` generated features — the only input with gradient.
        They are also the generated negatives.
    pos : Tensor
        ``(B, C_p, S)`` positive (same-class real) features.
    neg : Tensor or None
        ``(B, C_n, S)`` unconditional real features — the extra negatives
        guidance adds — or ``None``.
    weight_neg : Tensor or None
        ``(B, C_n)`` kernel weight :math:`w` of those negatives.
    temperatures : tuple of float
        :math:`\tau` values; one normalised drift each, summed.

    Returns
    -------
    (Tensor, dict)
        ``(B,)`` per-group loss, and the scalar diagnostics the release
        logs: the distance ``scale`` and the raw drift energy per
        temperature.

    Notes
    -----
    Line for line the release's ``drift_loss``.  Targets are ordered
    ``[generated, unconditional, positive]``; the softmax kernel is taken
    over targets and over generated samples and combined by a geometric
    mean; the negatives' coefficients are weighted by the positives' total
    affinity and vice versa (Alg. 2), which keeps the field
    anti-symmetric.  The distance normalisation (A.6) is one scalar over
    every group and location, as is each temperature's drift
    normalisation.
    """
    gen = gen.to(_float32_or_wider(gen.dtype))
    s = int(gen.shape[-1])
    info: dict[str, Tensor] = {}

    with lucid.no_grad():
        old = gen.detach()
        targets, target_w, c_n = _drift_targets(old, pos, neg, weight_neg)
        self_mask = _self_mask(old, targets)

        dist = _pairwise_distance(old, targets, self_mask)
        scale = (dist * target_w.unsqueeze(1)).mean() / target_w.mean()
        info["scale"] = scale
        # Coordinates normalised so the mean distance is about sqrt(S).
        scale_inputs = lucid.clamp(scale / math.sqrt(s), min=1e-3)
        old_scaled = old / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / lucid.clamp(scale, min=1e-3)

        force, energies = _drift_force(
            dist_normed, self_mask, target_w, targets_scaled, old_scaled, c_n, temperatures
        )
        info.update(energies)
        goal = old_scaled + force

    diff = gen / scale_inputs - goal
    return (diff * diff).mean(dim=(1, 2)), info


def _drift_targets(
    old: Tensor, pos: Tensor, neg: Tensor | None, weight_neg: Tensor | None
) -> tuple[Tensor, Tensor, int]:
    """Everything the field compares a sample with, ordered ``[generated, unconditional, positive]``.

    Returns the ``(B, C_g + C_n + C_p, S)`` targets in ``old``'s dtype, their
    ``(B, C_g + C_n + C_p)`` kernel weights — one, except :math:`w` on the
    unconditional negatives — and ``C_n``, zero when there are none.
    """
    b, c_g, _ = (int(d) for d in old.shape)
    work, device = old.dtype, old.device.type
    parts = [old]
    weights = [lucid.ones((b, c_g), dtype=work, device=device)]
    c_n = 0
    if neg is not None and int(neg.shape[1]) > 0:
        c_n = int(neg.shape[1])
        parts.append(neg.to(work))
        weights.append(
            weight_neg.to(work)
            if weight_neg is not None
            else lucid.ones((b, c_n), dtype=work, device=device)
        )
    c_p = int(pos.shape[1])
    parts.append(pos.to(work))
    weights.append(lucid.ones((b, c_p), dtype=work, device=device))
    return lucid.cat(parts, dim=1), lucid.cat(weights, dim=1), c_n


def _self_mask(old: Tensor, targets: Tensor) -> Tensor:
    """``(C_g, C_g + C_n + C_p)``: one where a generated sample meets itself among the targets."""
    c_g, n_targets = int(old.shape[1]), int(targets.shape[1])
    work, device = old.dtype, old.device.type
    return lucid.cat(
        [
            lucid.eye(c_g, dtype=work, device=device),
            lucid.zeros((c_g, n_targets - c_g), dtype=work, device=device),
        ],
        dim=1,
    )


def _drift_force(
    dist_normed: Tensor,
    self_mask: Tensor,
    target_w: Tensor,
    targets_scaled: Tensor,
    old_scaled: Tensor,
    c_n: int,
    temperatures: tuple[float, ...],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Alg. 2's field at every generated sample, normalised per temperature and summed.

    The first ``C_g`` targets are the samples themselves, masked out of
    their own kernel row by ``self_mask``; those and the next ``c_n``
    repel, the rest attract.  Returns the force and each temperature's raw
    drift energy, keyed ``loss_{tau}``.
    """
    c_g = int(old_scaled.shape[1])
    dist_normed = dist_normed + self_mask.unsqueeze(0) * _SELF_MASK

    split = c_g + c_n
    energies: dict[str, Tensor] = {}
    force = lucid.zeros_like(old_scaled)
    for tau in temperatures:
        logits = -dist_normed / tau
        affinity = F.softmax(logits, dim=-1) * F.softmax(logits, dim=-2)
        affinity = lucid.sqrt(lucid.clamp(affinity, min=1e-6)) * target_w.unsqueeze(1)
        aff_neg = affinity[:, :, :split]
        aff_pos = affinity[:, :, split:]
        coeff_neg = -aff_neg * aff_pos.sum(dim=-1, keepdim=True)
        coeff_pos = aff_pos * aff_neg.sum(dim=-1, keepdim=True)
        coeff = lucid.cat([coeff_neg, coeff_pos], dim=2)
        # sum_y coeff * (y - x): the coefficients total zero only when
        # nothing repels, so the ``- x`` term is kept explicit.
        drift = lucid.matmul(coeff, targets_scaled) - coeff.sum(dim=-1).unsqueeze(-1) * old_scaled
        energy = (drift * drift).mean()
        energies[f"loss_{tau}"] = energy
        force = force + drift / lucid.sqrt(lucid.clamp(energy, min=1e-8))
    return force, energies


# ─────────────────────────────────────────────────────────────────────────────
# Feature space: the ResNet-style MAE encoder and ConvNeXt-V2 (A.3–A.5)
# ─────────────────────────────────────────────────────────────────────────────


def _gn_groups(channels: int, max_groups: int = 32) -> int:
    """The largest group count ``<= max_groups`` that divides ``channels``."""
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with the release's group rule and Flax's ``epsilon=1e-6``."""
    return nn.GroupNorm(_gn_groups(channels), channels, eps=1e-6)


def _safe_std(x: Tensor, dims: tuple[int, ...], eps: float = 1e-6) -> Tensor:
    """Population std over ``dims`` in float32-or-wider, ``sqrt(max(var, 0) + eps)``."""
    x = x.to(_float32_or_wider(x.dtype))
    mean = x.mean(dim=dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dims)
    return lucid.sqrt(lucid.clamp(var, min=0.0) + eps)


def _tokens(feat: Tensor) -> Tensor:
    """``(B, C, H, W) -> (B, H*W, C)``, row-major locations."""
    b, c, h, w = (int(s) for s in feat.shape)
    return feat.reshape(b, c, h * w).permute(0, 2, 1)


def _space_to_depth(x: Tensor, patch: int) -> Tensor:
    """``b c (h p) (w p) -> b (p p c) h w`` with the release's ``(h2, w2, c)`` order."""
    if patch == 1:
        return x
    b, c, h, w = (int(s) for s in x.shape)
    x = x.reshape(b, c, h // patch, patch, w // patch, patch).permute(0, 3, 5, 1, 2, 4)
    return x.reshape(b, patch * patch * c, h // patch, w // patch)


def _add_feature_sets(
    out: dict[str, Tensor], name: str, feat: Tensor, config: DriftingConfig
) -> None:
    """Add one ``(B, C, H, W)`` map's feature sets to ``out``, in the release's order.

    Its locations as ``name``; its global ``_mean`` and ``_std``; then the
    means and stds pooled over each ``patch_mean_size`` / ``patch_std_size``
    square that tiles the map.
    """
    b, c, h, w = (int(s) for s in feat.shape)
    out[name] = _tokens(feat)
    if config.use_mean:
        out[f"{name}_mean"] = feat.mean(dim=(2, 3)).unsqueeze(1)
    if config.use_std:
        out[f"{name}_std"] = _safe_std(feat, (2, 3)).unsqueeze(1)
    for size in config.patch_mean_size:
        if h % size == 0 and w % size == 0:
            grouped = feat.reshape(b, c, h // size, size, w // size, size)
            out[f"{name}_mean_{size}"] = _tokens(grouped.mean(dim=(3, 5)))
    for size in config.patch_std_size:
        if h % size == 0 and w % size == 0:
            grouped = feat.reshape(b, c, h // size, size, w // size, size)
            out[f"{name}_std_{size}"] = _tokens(_safe_std(grouped, (3, 5)))


class _MAEBasicBlock(nn.Module):
    """ResNet basic block with GroupNorm: two 3x3 convs, projected skip on change."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """Initialise the block. See the class docstring."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.gn1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.gn2 = _group_norm(out_channels)
        # The release builds these in every block and Flax only materialises
        # them where the shape changes; building them only there is the same
        # parameter set.
        self.has_proj = stride != 1 or in_channels != out_channels
        if self.has_proj:
            self.proj_conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.proj_gn = _group_norm(out_channels)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """``relu(skip(x) + gn2(conv2(relu(gn1(conv1(x))))))``."""
        y = self.gn2(self.conv2(F.relu(self.gn1(self.conv1(x)))))
        skip = self.proj_gn(self.proj_conv(x)) if self.has_proj else x
        return F.relu(skip + y)


class _MAEResNetEncoder(nn.Module):
    """The latent-/pixel-MAE's encoder: a GroupNorm ResNet-34 at width ``C``."""

    def __init__(self, config: DriftingConfig) -> None:
        """Build the encoder from the family config's ``mae_*`` fields."""
        super().__init__()
        self.input_patch_size = config.mae_input_patch_size
        base = config.mae_base_channels
        in_ch = config.in_channels * self.input_patch_size**2
        self.conv1 = nn.Conv2d(in_ch, base, 3, stride=1, padding=1, bias=False)
        self.gn1 = _group_norm(base)
        stages: list[nn.ModuleList[_MAEBasicBlock]] = []
        norms: list[nn.GroupNorm] = []
        ch = base
        for stage, num_blocks in enumerate(config.mae_layers):
            out_ch = base * (2**stage)
            blocks = [_MAEBasicBlock(ch, out_ch, 2 if stage > 0 else 1)]
            blocks += [_MAEBasicBlock(out_ch, out_ch, 1) for _ in range(1, num_blocks)]
            stages.append(nn.ModuleList(blocks))
            norms.append(_group_norm(out_ch))
            ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.stage_norms = nn.ModuleList(norms)

    def activations(self, x: Tensor, config: DriftingConfig) -> dict[str, Tensor]:
        """The release's ``MAEResNetJAX.get_activations``: every feature set, ``(B, T, D)``.

        Keys: ``norm_x``; ``conv1`` and ``layer{1-4}`` (stage outputs after
        their GroupNorm); ``layer{i}_blk{j}`` every ``every_k_block`` blocks
        (pre-norm); and for each, ``_mean``/``_std`` and the pooled
        ``_mean_{s}``/``_std_{s}`` sets.
        """
        x = _space_to_depth(x, self.input_patch_size)
        out: dict[str, Tensor] = {}
        work = _float32_or_wider(x.dtype)
        out["norm_x"] = lucid.sqrt((x.to(work) ** 2).mean(dim=(2, 3)) + 1e-6).unsqueeze(1)

        x = F.relu(self.gn1(self.conv1(x)))
        stage_feats: dict[str, Tensor] = {"conv1": x}
        block_feats: list[tuple[str, Tensor]] = []
        k = config.every_k_block
        for i, (blocks, norm) in enumerate(zip(self.stages, self.stage_norms, strict=True)):
            for j, block in enumerate(blocks, start=1):
                x = block(x)
                if k is not None and j % k == 0:
                    block_feats.append((f"layer{i + 1}_blk{j}", x))
            x = norm(x)
            stage_feats[f"layer{i + 1}"] = x
        for name, feat in [*stage_feats.items(), *block_feats]:
            _add_feature_sets(out, name, feat, config)
        return out


class _ChannelsLastLayerNorm(nn.Module):
    """ConvNeXt's LayerNorm over the channel axis of an ``NCHW`` map, eps 1e-6."""

    def __init__(self, dim: int) -> None:
        """Initialise the norm. See the class docstring."""
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(lucid.ones(dim))
        self.bias = nn.Parameter(lucid.zeros(dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Normalise each location's channel vector."""
        y = F.layer_norm(x.permute(0, 2, 3, 1), (self.dim,), self.weight, self.bias, eps=1e-6)
        return y.permute(0, 3, 1, 2)


class _GRN(nn.Module):
    """ConvNeXt-V2 Global Response Normalisation on a channels-last map.

    The release's form: ``gx = sqrt(sum_hw x^2 + eps)``,
    ``nx = gx / (mean_c gx + eps)``, ``gamma * (x * nx) + beta + x``.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialise the GRN. See the class docstring."""
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(lucid.zeros(dim))
        self.beta = nn.Parameter(lucid.zeros(dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply GRN to ``(B, H, W, C)``."""
        gx = lucid.sqrt((x * x).sum(dim=(1, 2), keepdim=True) + self.eps)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class _ConvNeXtV2Block(nn.Module):
    """ConvNeXt-V2 block: depthwise 7x7, LN, 4x MLP with GRN, residual."""

    def __init__(self, dim: int) -> None:
        """Initialise the block. See the class docstring."""
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.grn = _GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Advance an ``NCHW`` map."""
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.pwconv2(self.grn(F.gelu(self.pwconv1(self.norm(y)))))
        return x + y.permute(0, 3, 1, 2)


class _ConvNeXtV2(nn.Module):
    """ConvNeXt-V2-Base (``depths=(3, 3, 27, 3)``, ``dims=(128, 256, 512, 1024)``)."""

    def __init__(
        self, depths: tuple[int, ...] = (3, 3, 27, 3), dims: tuple[int, ...] = (128, 256, 512, 1024)
    ) -> None:
        """Initialise the backbone. See the class docstring."""
        super().__init__()
        layers: list[nn.Module] = [
            nn.Sequential(nn.Conv2d(3, dims[0], 4, stride=4), _ChannelsLastLayerNorm(dims[0]))
        ]
        layers.extend(
            nn.Sequential(
                _ChannelsLastLayerNorm(dims[i]), nn.Conv2d(dims[i], dims[i + 1], 2, stride=2)
            )
            for i in range(3)
        )
        self.downsample_layers = nn.ModuleList(layers)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(*[_ConvNeXtV2Block(dims[i]) for _ in range(depths[i])])
                for i in range(4)
            ]
        )
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

    def activations(self, x: Tensor) -> dict[str, Tensor]:
        """The release's ``ConvNextV2.get_activations`` on ImageNet-normalised ``x``.

        The keys keep the release's spelling (``convenxt_*``).
        """
        x = F.interpolate(
            x,
            size=(_CONVNEXT_SIZE, _CONVNEXT_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        work = _float32_or_wider(x.dtype)

        def normalize(y: Tensor) -> Tensor:
            y32 = y.to(work)
            mean = y32.mean(dim=1, keepdim=True)
            std = lucid.sqrt(((y32 - mean) ** 2).mean(dim=1, keepdim=True))
            return ((y32 - mean) / (std + 1e-3)).to(y.dtype)

        out: dict[str, Tensor] = {}
        for i in range(4):
            x = self.stages[i](self.downsample_layers[i](x))
            tokens = _tokens(normalize(x))
            if i > 0:
                out[f"convenxt_stage_{i}"] = tokens
            out[f"convenxt_stage_{i}_mean"] = tokens.mean(dim=1).unsqueeze(1)
            out[f"convenxt_stage_{i}_std"] = _safe_std(tokens, (1,)).unsqueeze(1)
        out["global_mean"] = self.norm(x.mean(dim=(2, 3))).unsqueeze(1)
        out["global_std"] = _safe_std(_tokens(normalize(x)), (1,)).unsqueeze(1)
        return out


class DriftingFeatureEncoder(nn.Module):
    r"""The frozen feature map :math:`\phi` the drifting loss is computed in.

    Parameters
    ----------
    config : DriftingConfig
        Reads ``use_mae``, ``use_convnext``, the ``mae_*`` architecture and
        the A.5 feature options.
    build : bool, default=True
        Construct the encoders now.  ``False`` leaves an empty shell that
        :meth:`DriftingForImageGeneration.load_feature_encoder` fills —
        the MAE-640 encoder is far larger than the generator, and sampling
        never needs it.

    Attributes
    ----------
    mae : _MAEResNetEncoder
        Present when ``use_mae`` and built.
    convnext : _ConvNeXtV2
        Present when ``use_convnext`` and built.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026,
    Appendices A.3–A.5.  The output is the release's ``activation_fn``: a
    ``"global"`` set holding the raw sample (the vanilla drifting loss
    without :math:`\phi`), then the MAE's sets, then ConvNeXt-V2's, which
    read ``(x + 1) / 2`` normalised by ImageNet statistics.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.drifting import (
    ...     DriftingConfig, DriftingFeatureEncoder)
    >>> config = DriftingConfig(sample_size=8, hidden_size=32, num_heads=4,
    ...                         mae_base_channels=16, mae_layers=(1, 1, 1, 1))
    >>> feats = DriftingFeatureEncoder(config)(lucid.randn(2, 4, 8, 8))
    >>> feats["global"].shape, feats["layer1"].shape
    ((2, 1, 256), (2, 64, 16))
    """

    def __init__(self, config: DriftingConfig, *, build: bool = True) -> None:
        """Initialise the encoder. See the class docstring for parameters."""
        super().__init__()
        self.config = config
        self.built = False
        if build:
            self.build()

    @property
    def needed(self) -> bool:
        """Whether the config asks for any encoder beyond the raw sample."""
        return self.config.use_mae or self.config.use_convnext

    def build(self) -> None:
        """Construct the configured encoders (idempotent)."""
        if self.built:
            return
        if self.config.use_mae:
            self.mae = _MAEResNetEncoder(self.config)
        if self.config.use_convnext:
            self.convnext = _ConvNeXtV2()
        self.built = True

    @override
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Every feature set of ``(B, C, H, W)`` samples, each ``(B, T, D)``."""
        if self.needed and not self.built:
            raise RuntimeError(
                "the drifting loss is configured in a learned feature space "
                "(use_mae / use_convnext) but no encoder has been built; call "
                "DriftingForImageGeneration.load_feature_encoder() first, or set "
                "use_mae=False and use_convnext=False to drift in the raw space"
            )
        config = self.config
        b = int(x.shape[0])
        feats: dict[str, Tensor] = {"global": x.reshape(b, 1, -1)}
        if config.use_mae:
            feats.update(self.mae.activations(x, config))
        if config.use_convnext:
            mean = lucid.tensor(list(_IMAGENET_MEAN), dtype=x.dtype, device=x.device.type)
            std = lucid.tensor(list(_IMAGENET_STD), dtype=x.dtype, device=x.device.type)
            rgb = ((x + 1.0) / 2.0 - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
            feats.update(self.convnext.activations(rgb))
        return feats


# ─────────────────────────────────────────────────────────────────────────────
# The sample queue (A.8, ``memory_bank.py``)
# ─────────────────────────────────────────────────────────────────────────────


def _int_list(indices: Tensor) -> list[int]:
    """A 1-D integer tensor's elements as Python ints, read back in one transfer.

    Raises
    ------
    TypeError
        If ``indices`` is not 1-D or holds anything but integers — the error
        indexing a queue with such a value would raise.
    """
    values = indices.tolist()
    if not isinstance(values, list):
        raise TypeError(f"expected a 1-D tensor of indices, got shape {tuple(indices.shape)}")
    ints: list[int] = []
    for value in values:
        if not isinstance(value, int):
            raise TypeError(f"expected integer indices, got {type(value).__name__}")
        ints.append(value)
    return ints


class DriftingSampleQueue:
    """Per-class ring buffers of real samples — the release's ``ArrayMemoryBank``.

    Parameters
    ----------
    num_classes : int
        Number of queues.  The unconditional queue is a single class.
    max_size : int
        Length of each queue; the oldest entry is overwritten.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, A.8.
    Sampling is without replacement while a queue holds enough entries and
    with replacement otherwise; an empty queue yields zeros, as the
    release's zero-initialised bank does.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.drifting import DriftingSampleQueue
    >>> queue = DriftingSampleQueue(num_classes=2, max_size=4)
    >>> queue.add(lucid.randn(3, 4, 8, 8), lucid.tensor([0, 1, 0]))
    >>> queue.sample(lucid.tensor([0, 1]), 2).shape
    (2, 2, 4, 8, 8)
    """

    def __init__(self, num_classes: int, max_size: int) -> None:
        """Initialise empty queues. See the class docstring."""
        self.num_classes = int(num_classes)
        self.max_size = int(max_size)
        # A queue grows to ``max_size`` and is then overwritten in place, so
        # its length is how many entries it holds.
        self._items: list[list[Tensor]] = [[] for _ in range(self.num_classes)]
        self._ptr = [0] * self.num_classes
        self._like: Tensor | None = None

    def add(self, samples: Tensor, labels: Tensor) -> None:
        """Push ``(N, ...)`` samples into the queues named by ``(N,)`` labels."""
        for i, lbl in enumerate(_int_list(labels)):
            sample = samples[i].detach()
            queue = self._items[lbl]
            if len(queue) < self.max_size:
                queue.append(sample)
            else:
                queue[self._ptr[lbl]] = sample
            self._ptr[lbl] = (self._ptr[lbl] + 1) % self.max_size
            self._like = sample

    def sample(self, labels: Tensor, n_samples: int) -> Tensor:
        """Draw ``n_samples`` per label: ``(B, n_samples, ...)``."""
        if self._like is None:
            raise RuntimeError("the sample queue is empty; call add() before sample()")
        like = self._like
        rows: list[Tensor] = []
        for lbl in _int_list(labels):
            queue = self._items[lbl]
            valid = len(queue)
            if valid == 0:
                rows.append(
                    lucid.zeros((n_samples, *like.shape), dtype=like.dtype, device=like.device.type)
                )
                continue
            if valid < n_samples:
                picks = _int_list(lucid.randint(0, valid, (n_samples,)))
            else:
                picks = _int_list(lucid.randperm(valid)[:n_samples])
            rows.append(lucid.stack([queue[j] for j in picks]))
        return lucid.stack(rows)


# ─────────────────────────────────────────────────────────────────────────────
# The task model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DriftingOutput(ModelOutput):
    r"""What a drifting training step reports.

    Attributes
    ----------
    loss : Tensor
        Sum over feature sets of the mean drifting loss — a scalar whose
        value is the normalised drift energy :math:`\lVert\tilde V\rVert^2`.
    samples : Tensor
        ``(N_c * gen_per_label, C, H, W)`` — the generated batch, grouped
        label-major.
    cfg_scale : Tensor
        ``(N_c,)`` — the guidance scale :math:`\alpha` each label trained at.
    feature_losses : dict of str to Tensor
        The per-feature-set loss terms, keyed as the encoder names them.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.drifting import DriftingOutput
    >>> out = DriftingOutput(loss=lucid.zeros(()), samples=lucid.zeros(4, 4, 8, 8),
    ...                      cfg_scale=lucid.ones(1), feature_losses={})
    >>> out.samples.shape
    (4, 4, 8, 8)
    """

    loss: Tensor
    samples: Tensor
    cfg_scale: Tensor
    feature_losses: dict[str, Tensor]


class DriftingForImageGeneration(ImageGenerationModel):
    r"""The drifting objective and its one-step sampler.

    Parameters
    ----------
    config : DriftingConfig
        The variant to build.

    Attributes
    ----------
    drifting : DriftingModel
        The generator — the only trainable network.
    feature_encoder : DriftingFeatureEncoder
        The frozen :math:`\phi`.  Empty until
        :meth:`load_feature_encoder` builds it, unless the config drifts
        in the raw space (``use_mae=False, use_convnext=False``).

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, *"Generative Modeling via
    Drifting"*, arXiv:2602.04770, 2026.  Training is Alg. 1 with the
    feature-space loss of Eq. (14) and the guidance of A.7; sampling is
    one evaluation of :math:`f_\theta`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.drifting import (
    ...     DriftingConfig, DriftingForImageGeneration)
    >>> config = DriftingConfig(sample_size=8, hidden_size=32, cond_dim=32, depth=1,
    ...                         num_heads=4, num_classes=10, noise_classes=4,
    ...                         noise_coords=2, n_cls_tokens=2, gen_per_label=4,
    ...                         use_mae=False)
    >>> model = DriftingForImageGeneration(config)
    >>> model(lucid.randn(2, 3, 4, 8, 8), lucid.tensor([0, 1])).loss.ndim
    0
    >>> model.generate(2, labels=lucid.tensor([3, 7])).samples.shape
    (2, 4, 8, 8)
    """

    config_class: ClassVar[type[DriftingConfig]] = DriftingConfig
    base_model_prefix = "drifting"

    def __init__(self, config: DriftingConfig) -> None:
        """Initialise the model. See the class docstring for parameters."""
        super().__init__(config)
        self.config: DriftingConfig = config
        self.drifting = DriftingModel(config)
        # Raw-space drifting needs no network, so the shell is complete.
        encoder = DriftingFeatureEncoder(config, build=False)
        if not encoder.needed:
            encoder.build()
        self.feature_encoder = encoder

    # ── the feature encoder ──────────────────────────────────────────────

    def load_feature_encoder(self, pretrained: bool | str = False) -> DriftingFeatureEncoder:
        """Build the frozen feature encoder, optionally with the released weights.

        Parameters
        ----------
        pretrained : bool or str, default=False
            ``True`` loads the MAE encoder matching the config
            (``LATENT_640``, ``LATENT_256`` or ``PIXEL_640``) and, for pixel
            models, ConvNeXt-V2-Base.  A string names the MAE tag.

        Returns
        -------
        DriftingFeatureEncoder
            The encoder, also stored as ``self.feature_encoder``, with every
            parameter frozen.

        Raises
        ------
        ValueError
            If no released MAE matches the config, or the tag is unknown.
        """
        encoder = DriftingFeatureEncoder(self.config, build=True)
        if pretrained is not False and self.config.use_mae:
            tag = pretrained if isinstance(pretrained, str) else self._mae_tag()
            if tag not in DriftingMAEWeights.__members__:
                raise ValueError(
                    f"no released MAE encoder is tagged {tag!r}; known: "
                    f"{list(DriftingMAEWeights.__members__)}"
                )
            load_weight_entry(encoder.mae, DriftingMAEWeights[tag], name=f"drifting_mae/{tag}")
        if pretrained is not False and self.config.use_convnext:
            load_weight_entry(
                encoder.convnext,
                DriftingConvNeXtV2Weights.DEFAULT,
                name="drifting_convnext_v2/IMAGENET22K_224",
            )
        encoder.requires_grad_(False)
        encoder.eval()
        encoder.to(resolve_generation_device(self.drifting, None))
        self.feature_encoder = encoder
        return encoder

    def _mae_tag(self) -> str:
        """The released MAE encoder this config was trained against."""
        config = self.config
        for name, member in DriftingMAEWeights.__members__.items():
            wanted = member.value.meta["config"]
            if isinstance(wanted, dict) and all(getattr(config, k) == v for k, v in wanted.items()):
                return name
        raise ValueError(
            f"no released MAE encoder has in_channels={config.in_channels}, "
            f"mae_base_channels={config.mae_base_channels}, "
            f"mae_input_patch_size={config.mae_input_patch_size}"
        )

    # ── the sample queues (A.8) ──────────────────────────────────────────

    def make_sample_queues(self) -> tuple[DriftingSampleQueue, DriftingSampleQueue]:
        """The per-class positive queue and the global unconditional queue."""
        config = self.config
        return (
            DriftingSampleQueue(config.num_classes, config.positive_bank_size),
            DriftingSampleQueue(1, config.negative_bank_size),
        )

    def draw_training_batch(
        self,
        positive_queue: DriftingSampleQueue,
        unconditional_queue: DriftingSampleQueue,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Positives ``(N_c, pos_per_sample, ...)`` and negatives ``(N_c, neg_per_sample, ...)``."""
        config = self.config
        positives = positive_queue.sample(labels, config.pos_per_sample)
        negatives = unconditional_queue.sample(lucid.zeros_like(labels), config.neg_per_sample)
        return positives, negatives

    # ── the objective ────────────────────────────────────────────────────

    def _sample_cfg(self, batch: int, device: str, dtype: lucid.dtype) -> Tensor:
        r"""Draw :math:`\alpha \in [\alpha_{\min}, \alpha_{\max}]` from :math:`p(\alpha) \propto \alpha^{-k}`.

        Inverse-CDF sampling, log-uniform at :math:`k = 1`; then a share
        ``no_cfg_frac`` is reset to ``1``.  The release's ``train_step``.
        """
        config = self.config
        lo, hi = config.cfg_min, config.cfg_max
        frac = lucid.rand((batch,), device=device)
        pw = 1.0 - config.neg_cfg_pw
        if abs(pw) < 1e-6:
            cfg = lucid.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
        else:
            cfg = (lo**pw + frac * (hi**pw - lo**pw)) ** (1.0 / pw)
        keep = (lucid.rand((batch,), device=device) >= config.no_cfg_frac).to(cfg.dtype)
        return (cfg * keep + (1.0 - keep)).to(dtype)

    @override
    def forward(
        self,
        images: Tensor,
        labels: Tensor,
        negatives: Tensor | None = None,
        cfg_scale: Tensor | None = None,
    ) -> DriftingOutput:
        r"""One drifting training step.

        Parameters
        ----------
        images : Tensor
            Positives: ``(N_c, N_pos, C, H, W)`` real samples of each label,
            or ``(N_c, C, H, W)`` for one each.
        labels : Tensor
            ``(N_c,)`` class labels.
        negatives : Tensor or None, optional
            ``(N_c, N_unc, C, H, W)`` unconditional real samples (random
            classes) — the extra negatives guidance trains with.  ``None``
            trains without guidance.
        cfg_scale : Tensor or None, optional
            ``(N_c,)`` guidance scales.  Drawn from the configured power law
            when ``None`` and negatives are given; ``1`` when neither is.

        Returns
        -------
        DriftingOutput
            The loss, the generated batch, the scales and the per-feature
            terms.

        Notes
        -----
        ``gen_per_label`` samples are generated per label.  Each feature set
        gets its own drifting loss, computed independently per label and
        location (rows ``(label, location)``), against that label's
        positives, its own generated batch, and — weighted by
        :math:`w = (\alpha - 1)(N_\mathrm{neg} - 1) / N_\mathrm{unc}` — the
        unconditional samples; the terms are summed (Eq. 14).
        """
        config = self.config
        positives = images if images.ndim == 5 else images.unsqueeze(1)
        n_c, n_pos = int(positives.shape[0]), int(positives.shape[1])
        sample_shape = tuple(int(s) for s in positives.shape[2:])
        device = positives.device.type
        dtype = positives.dtype
        g = config.gen_per_label
        n_unc = 0 if negatives is None else int(negatives.shape[1])

        if cfg_scale is None:
            cfg = (
                self._sample_cfg(n_c, device, dtype)
                if n_unc > 0
                else lucid.ones((n_c,), dtype=dtype, device=device)
            )
        else:
            cfg = cfg_scale.to(dtype).reshape(-1)
        uncond_w = (cfg - 1.0) * (g - 1) / max(1, n_unc)

        noise = lucid.randn(n_c * g, *sample_shape, dtype=dtype, device=device)
        gen = self.drifting(
            noise, labels.repeat_interleave(g, dim=0), cfg.repeat_interleave(g, dim=0)
        ).sample

        real = (
            positives if negatives is None else lucid.cat([positives, negatives.to(dtype)], dim=1)
        )
        n_real = n_pos + n_unc
        with lucid.no_grad():
            real_feats = self.feature_encoder(real.reshape(n_c * n_real, *sample_shape))
        gen_feats = self.feature_encoder(gen)

        total: Tensor | None = None
        feature_losses: dict[str, Tensor] = {}
        for key, gen_feat in gen_feats.items():
            f, d = int(gen_feat.shape[1]), int(gen_feat.shape[2])
            rows = gen_feat.reshape(n_c, g, f, d).permute(0, 2, 1, 3).reshape(n_c * f, g, d)
            real_feat = real_feats[key].reshape(n_c, n_real, f, d).permute(0, 2, 1, 3)
            pos = real_feat[:, :, :n_pos].reshape(n_c * f, n_pos, d)
            neg: Tensor | None = None
            weight: Tensor | None = None
            if n_unc > 0:
                neg = real_feat[:, :, n_pos:].reshape(n_c * f, n_unc, d)
                weight = uncond_w.reshape(n_c, 1, 1).expand(n_c, f, n_unc).reshape(n_c * f, n_unc)
            loss, _ = _drift_loss(rows, pos, neg, weight, config.temperatures)
            term = loss.mean()
            feature_losses[key] = term.to(dtype)
            total = term if total is None else total + term

        assert total is not None
        return DriftingOutput(
            loss=total.to(dtype), samples=gen, cfg_scale=cfg, feature_losses=feature_losses
        )

    # ── sampling ─────────────────────────────────────────────────────────

    def generate(
        self,
        n_samples: int = 1,
        *,
        labels: Tensor | None = None,
        cfg_scale: float | Tensor = 1.0,
        temperature: float = 1.0,
        noise: Tensor | None = None,
        style_indices: Tensor | None = None,
        device: str | None = None,
    ) -> GenerationOutput:
        r"""Sample in one network evaluation.

        Parameters
        ----------
        n_samples : int, default=1
            How many to draw.
        labels : Tensor or None, optional
            ``(n_samples,)`` class indices.  Drawn uniformly when ``None`` —
            the model is class-conditional and has no null class.
        cfg_scale : float or Tensor, default=1.0
            Guidance scale :math:`\alpha`; any value in the trained range
            works without retraining.  The released best: ``1.0`` for
            ``L/2``, ``B/16``, ``L/16``; ``1.1`` for ``B/2``.
        temperature : float, default=1.0
            Multiplier on the Gaussian noise (the release's ``temp``).
        noise : Tensor or None, optional
            ``(n_samples, C, H, W)`` noise; drawn when absent (and then
            scaled by ``temperature``).
        style_indices : Tensor or None, optional
            ``(n_samples, noise_coords)`` style indices; drawn when absent.
        device : str or None, optional
            Where to draw.  Defaults to the model's device.

        Returns
        -------
        GenerationOutput
            ``samples`` of shape ``(n_samples, out_channels, H, W)`` — an
            SD-VAE latent for the latent models, ``[-1, 1]`` RGB for the
            pixel models.

        Examples
        --------
        A released configuration, shrunk by overrides so it builds quickly:

        >>> from lucid.models import create_model
        >>> model = create_model("drifting_base_2_gen", sample_size=8, hidden_size=32,
        ...                      cond_dim=32, depth=1, num_heads=4).eval()
        >>> model.generate(3, cfg_scale=1.5).samples.shape
        (3, 4, 8, 8)
        """
        config = self.config
        side = config.grid_size * config.patch_size
        device = resolve_generation_device(self, device)
        if noise is None:
            noise = (
                start_noise(self, (n_samples, config.in_channels, side, side), device) * temperature
            )
        if labels is None:
            labels = lucid.randint(
                0, config.num_classes, (n_samples,), dtype=lucid.int64, device=device
            )
        with lucid.no_grad():
            sample = self.drifting(noise, labels, cfg_scale, style_indices).sample
        return GenerationOutput(samples=sample)
