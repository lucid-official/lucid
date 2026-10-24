"""Flow Matching model — simulation-free CNF training (Lipman et al., 2023).

Training, per batch:

    x1 ~ data,  x0 ~ N(0, I),  t ~ U[0, 1]
    x_t    = mu_t * x1 + sigma_t * x0                (the conditional path)
    target = mu_t' * x1 + sigma_t' * x0              (Theorem 3)
    loss   = || v(t, x_t) - target ||^2

No ODE is solved.  The solver only appears at inference, where sampling
integrates ``v`` from ``t = 0`` to ``t = 1`` and likelihood integrates it
back the other way while accumulating the divergence.

The path enters only through two scalar schedules and their derivatives,
which is exactly how the paper states Theorem 3 — so a new path is four
lines in :func:`_path_coefficients`, not a new model.
"""

import math
from collections.abc import Callable
from typing import ClassVar, cast, final, override

import lucid
import lucid.autograd
import lucid.diffeq
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageGenerationModel
from lucid.models._output import DiffusionModelOutput, GenerationOutput
from lucid.models._utils._generative import (
    resolve_generation_device,
    exact_divergence,
    generative_activation,
    hutchinson_divergence,
    start_noise,
    trace_probe,
)
from lucid.models.generative.flow_matching._config import FlowMatchingConfig

# What ``lucid.diffeq`` calls a state: one tensor, or a tuple of them when
# more than one quantity is integrated at once.
_State = Tensor | tuple[Tensor, ...]

# The sinusoidal embedding was designed for the 0…1000 integer grid that
# discrete-time diffusion models step over.  Flow Matching's time is
# continuous on [0, 1], so it is scaled onto the same range rather than
# occupying a single bucket at the bottom of the frequency ladder.
_TIME_SCALE = 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Conditional probability paths — Theorem 3
# ─────────────────────────────────────────────────────────────────────────────


def _path_coefficients(
    path: str, t: Tensor, *, sigma_min: float, beta_min: float, beta_max: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Schedules and their derivatives for one Gaussian conditional path.

    A Gaussian conditional path is fixed by
    :math:`p_t(x \mid x_1) = \mathcal{N}(x \mid a_t x_1, \sigma_t^2 I)`,
    and Theorem 3 turns those two scalars into the unique field that
    generates it.  Everything the paths differ by lives here.

    Parameters
    ----------
    path : {"ot", "diffusion"}
        Which path to build.
    t : Tensor
        Times, broadcastable against the sample — ``(B, 1, 1, 1)`` during
        training, ``()`` during a solve.
    sigma_min : float
        Terminal width of the optimal-transport path.
    beta_min, beta_max : float
        Endpoints of the linear :math:`\beta` used by the diffusion path.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(a_t, sigma_t, a_t', sigma_t')``, where the mean is
        :math:`a_t x_1`.

    Raises
    ------
    ValueError
        If ``path`` names neither schedule.
    """
    if path == "ot":
        # Paper eq. (20): straight schedules, hence straight conditional
        # flows.  a_t goes 0 → 1 while sigma_t goes 1 → sigma_min.
        slope = 1.0 - sigma_min
        a = t
        sigma = 1.0 - slope * t
        a_dot = lucid.ones_like(t)
        sigma_dot = lucid.full_like(t, -slope)
        return a, sigma, a_dot, sigma_dot

    if path == "diffusion":
        # The variance-preserving process of score-based diffusion, read in
        # this paper's time direction (t = 0 noise, t = 1 data), so the
        # usual schedule is evaluated at 1 - t.
        span = beta_max - beta_min
        s = 1.0 - t
        # T(s) = ∫_0^s beta(r) dr with beta linear.
        total = beta_min * s + 0.5 * span * s * s
        a = lucid.exp(-0.5 * total)
        var = 1.0 - a * a
        # Clamped only to keep the square root real against round-off; the
        # value it protects is 0 at t = 1 by construction.
        sigma = lucid.sqrt(lucid.maximum(var, lucid.full_like(var, 1e-12)))
        beta_s = beta_min + span * s
        # d/dt a(1 - t) = +½ beta(1 - t) a, since d/ds a = -½ beta(s) a.
        a_dot = 0.5 * beta_s * a
        sigma_dot = -0.5 * beta_s * a * a / sigma
        return a, sigma, a_dot, sigma_dot

    raise ValueError(f"path must be 'ot' or 'diffusion', got {path!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Velocity-field building blocks
# ─────────────────────────────────────────────────────────────────────────────


@final
class _ResBlock(nn.Module):
    """Residual block with continuous-time conditioning.

    Two GroupNorm + activation + Conv 3x3 sub-layers with a per-channel
    additive time embedding injected between them.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        *,
        act_fn: str,
        dropout: float,
        groups: int,
    ) -> None:
        super().__init__()
        g_in = min(groups, in_channels) if in_channels % groups != 0 else groups
        g_out = min(groups, out_channels) if out_channels % groups != 0 else groups

        self._act_name = act_fn
        self.norm1 = nn.GroupNorm(g_in, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
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
        act = self._act_name
        h = self.conv1(generative_activation(act, self.norm1(x)))
        shift = self.time_proj(generative_activation(act, t_emb))
        h = h + shift.unsqueeze(-1).unsqueeze(-1)
        h = generative_activation(act, self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


@final
class _AttentionBlock(nn.Module):
    """Self-attention on a feature map, with a width-derived head count.

    The head *count* follows from ``num_head_channels`` rather than being
    fixed, so a wider stage gets proportionally more heads — the
    multi-resolution attention the borrowed architecture specifies.
    """

    def __init__(self, channels: int, *, head_channels: int, groups: int) -> None:
        super().__init__()
        heads = max(1, channels // head_channels)
        while channels % heads != 0:
            heads -= 1
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
        self.num_heads = heads
        self.head_dim = channels // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        b, c, h_dim, w_dim = x.shape
        n = int(h_dim) * int(w_dim)
        h = self.norm(x)
        qkv = self.qkv(h).reshape(b, 3, self.num_heads, self.head_dim, n)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = q @ k.permute(0, 1, 3, 2) * self.scale
        out: Tensor = F.softmax(scores, dim=-1) @ v
        out = out.permute(0, 1, 3, 2).reshape(b, c, h_dim, w_dim)
        return x + self.proj(out)


@final
class _Downsample(nn.Module):
    """Stride-2 3x3 conv with right/bottom padding only."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.op(F.pad(x, (0, 1, 0, 1)))


@final
class _Upsample(nn.Module):
    """Nearest-neighbour x2 followed by a 3x3 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.op(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ─────────────────────────────────────────────────────────────────────────────
# Velocity field
# ─────────────────────────────────────────────────────────────────────────────


class _VelocityField(nn.Module):
    r"""U-Net velocity field :math:`v_t(x; \theta)`.

    Structurally the architecture Flow Matching borrows: a symmetric
    encoder/decoder with GroupNorm residual blocks, self-attention at the
    listed resolutions, and a conditioning vector added per channel inside
    every block.  Two things distinguish it from the denoiser of a
    discrete diffusion model:

    * **Time is continuous.**  There is no step index — ``t`` is a real
      number in ``[0, 1]``, scaled onto the embedding's usual range.
    * **The output is a velocity**, in the same space and with the same
      channel count as the input.  There is no variance head: a flow's
      density comes from the divergence of this field, not from a
      predicted variance.

    Parameters
    ----------
    config : FlowMatchingConfig
        Width, depth, attention resolutions, head channels and dropout.

    Attributes
    ----------
    time_mlp : nn.TimestepEmbedding
        Sinusoidal embedding of the scaled time plus a 2-layer MLP,
        producing a conditioning vector of width ``4 * base_channels``.
    conv_in : nn.Conv2d
        Stem convolution.
    down_res, down_attn, down_sample : nn.ModuleList
        Encoder blocks.
    mid_block1, mid_attn, mid_block2 : nn.Module
        Bottleneck stack.
    up_res, up_attn, up_sample : nn.ModuleList
        Decoder blocks, consuming concatenated skips.
    norm_out : nn.GroupNorm
        Output normalisation.
    conv_out : nn.Conv2d
        Final projection back to ``in_channels``.

    Notes
    -----
    Reference: Lipman et al., *"Flow Matching for Generative Modeling"*,
    ICLR, 2023 (arXiv:2210.02747), which states it uses the U-Net of
    Dhariwal and Nichol, *"Diffusion Models Beat GANs on Image
    Synthesis"*, NeurIPS, 2021 (arXiv:2105.05233), "with minimal changes".
    Neither paper publishes a per-dataset architecture table for these
    experiments; the defaults here are the second paper's stated ones —
    128 base channels, two residual blocks per resolution, and
    multi-resolution attention with 64 channels per head.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.flow_matching import FlowMatchingConfig
    >>> from lucid.models.generative.flow_matching._model import _VelocityField
    >>> cfg = FlowMatchingConfig(sample_size=8, base_channels=16,
    ...                          channel_mult=(1, 2), num_res_blocks=1,
    ...                          attention_resolutions=(4,), resnet_groups=8)
    >>> field = _VelocityField(cfg).eval()
    >>> field(lucid.randn((2, 3, 8, 8)), lucid.tensor([0.3, 0.7])).shape
    (2, 3, 8, 8)
    """

    def __init__(self, config: FlowMatchingConfig) -> None:
        super().__init__()
        self._config = config
        levels = len(config.channel_mult)
        base = config.base_channels
        time_dim = base * 4
        self._act_name = config.act_fn

        self.time_mlp = nn.TimestepEmbedding(
            in_dim=base, out_dim=time_dim, flip_sin_to_cos=False
        )
        self.conv_in = nn.Conv2d(config.in_channels, base, 3, padding=1)

        size = config.sample_size
        spatial = size[0] if isinstance(size, tuple) else int(size)

        self.down_res: nn.ModuleList[_ResBlock] = nn.ModuleList()
        self.down_attn: nn.ModuleList[_AttentionBlock | nn.Identity] = nn.ModuleList()
        self.down_sample: nn.ModuleList[_Downsample | nn.Identity] = nn.ModuleList()
        skip_channels: list[int] = [base]
        ch = base
        for level, mult in enumerate(config.channel_mult):
            out_ch = base * mult
            for _ in range(config.num_res_blocks):
                self.down_res.append(
                    _ResBlock(
                        ch,
                        out_ch,
                        time_dim,
                        act_fn=config.act_fn,
                        dropout=config.dropout,
                        groups=config.resnet_groups,
                    )
                )
                ch = out_ch
                self.down_attn.append(
                    _AttentionBlock(
                        ch,
                        head_channels=config.num_head_channels,
                        groups=config.resnet_groups,
                    )
                    if spatial in config.attention_resolutions
                    else nn.Identity()
                )
                skip_channels.append(ch)
            if level != levels - 1:
                self.down_sample.append(_Downsample(ch))
                skip_channels.append(ch)
                spatial //= 2
            else:
                self.down_sample.append(nn.Identity())

        self.mid_block1 = _ResBlock(
            ch,
            ch,
            time_dim,
            act_fn=config.act_fn,
            dropout=config.dropout,
            groups=config.resnet_groups,
        )
        self.mid_attn = _AttentionBlock(
            ch, head_channels=config.num_head_channels, groups=config.resnet_groups
        )
        self.mid_block2 = _ResBlock(
            ch,
            ch,
            time_dim,
            act_fn=config.act_fn,
            dropout=config.dropout,
            groups=config.resnet_groups,
        )

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
                        act_fn=config.act_fn,
                        dropout=config.dropout,
                        groups=config.resnet_groups,
                    )
                )
                ch = out_ch
                self.up_attn.append(
                    _AttentionBlock(
                        ch,
                        head_channels=config.num_head_channels,
                        groups=config.resnet_groups,
                    )
                    if spatial in config.attention_resolutions
                    else nn.Identity()
                )
            if level != 0:
                self.up_sample.append(_Upsample(ch))
                spatial *= 2
            else:
                self.up_sample.append(nn.Identity())

        g_out = (
            min(config.resnet_groups, ch)
            if ch % config.resnet_groups != 0
            else config.resnet_groups
        )
        self.norm_out = nn.GroupNorm(g_out, ch, eps=1e-6)
        self.conv_out = nn.Conv2d(ch, config.in_channels, 3, padding=1)
        # The reference zero-initialises this projection so the block is
        # exactly the identity at step 0 — the standard diffusion U-Net
        # recipe, and what keeps a deep stack stable from the first step.
        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

        self._levels = levels
        self._blocks_per_down = config.num_res_blocks
        self._blocks_per_up = config.num_res_blocks + 1

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
        t_emb = self.time_mlp(t * _TIME_SCALE)

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
                h = self.down_sample[level](h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        idx = 0
        for level in range(self._levels - 1, -1, -1):
            for _ in range(self._blocks_per_up):
                h = lucid.cat([h, skips.pop()], dim=1)
                h = self.up_res[idx](h, t_emb)
                h = self.up_attn[idx](h)
                idx += 1
            if level != 0:
                h = self.up_sample[self._levels - 1 - level](h)

        h = generative_activation(self._act_name, self.norm_out(h))
        return self.conv_out(h)


# ─────────────────────────────────────────────────────────────────────────────
# Likelihood dynamics — only used at inference
# ─────────────────────────────────────────────────────────────────────────────


@final
class _LikelihoodDynamics:
    r"""Augmented right-hand side :math:`(x, \Delta)` for scoring density.

    Training never touches this.  It exists because a trained velocity
    field *is* a continuous normalizing flow, so the exact log-likelihood
    the paper reports is available by integrating

    .. math::

        \frac{dx}{dt} = v_t(x), \qquad
        \frac{d\Delta}{dt} = \operatorname{tr}
            \!\left(\frac{\partial v_t}{\partial x}\right)

    from :math:`t = 1` down to :math:`t = 0`, after which
    :math:`\log p_1(x_1) = \log p_0(x_0) + \Delta(0)`.
    """

    def __init__(self, field: _VelocityField, config: FlowMatchingConfig) -> None:
        # A plain reference, deliberately not a registered submodule: the
        # model already owns this field, and registering it twice would
        # put every weight in ``state_dict`` under two names and double
        # the size of every checkpoint.
        self.field = field
        self._dim = config.data_dim
        self._trace_method = config.resolved_trace_method
        self._trace_noise = config.trace_noise
        size = config.sample_size
        height, width = size if isinstance(size, tuple) else (size, size)
        self._image_shape = (config.in_channels, height, width)
        self.nfe = 0
        self._eps: Tensor | None = None
        self._seeds: list[Tensor] | None = None

        # No buffer: without Module machinery there is nothing to move
        # between devices, so the identity is built on first use from the
        # tensor that needs it and inherits its device and dtype.

    def reset(self, eps: Tensor | None = None) -> None:
        """Zero the counter and fix the probe for one solve."""
        self.nfe = 0
        self._eps = eps
        self._seeds = None

    def sample_noise(self, like: Tensor) -> Tensor:
        """Draw a Hutchinson probe shaped like ``like``."""
        return trace_probe(like, self._trace_noise)

    def velocity(self, t: Tensor, x: Tensor) -> Tensor:
        """Flat-state velocity — what sampling integrates."""
        self.nfe += 1
        batch = int(x.shape[0])
        image = x.reshape(batch, *self._image_shape)
        return self.field(image, t).reshape(batch, -1)

    def __call__(
        self, t: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        x, _ = state
        self.nfe += 1
        with lucid.enable_grad():
            # Detached under the adjoint's forward sweep, live under its
            # backward sweep and under direct differentiation; re-rooting
            # unconditionally would drop the state's own gradient path.
            source = x if x.requires_grad else x.detach().requires_grad_(True)
            batch = int(source.shape[0])
            image = source.reshape(batch, *self._image_shape)
            velocity = self.field(image, t).reshape(batch, -1)
            divergence = self._divergence(velocity, source)
        return velocity, divergence

    def _divergence(self, velocity: Tensor, x: Tensor) -> Tensor:
        # ``create_graph=False``: nothing differentiates this.  Flow
        # Matching's objective never touches the divergence — that is the
        # whole point of the method — so the density is a reported
        # quantity, not a trained one, and asking a convolutional field
        # for second derivatives it is never going to be asked for would
        # be pure cost.
        if self._trace_method == "exact":
            return exact_divergence(
                velocity, x, self._exact_seeds(velocity), create_graph=False
            )
        eps = self._eps
        if eps is None:
            eps = self.sample_noise(x)
            self._eps = eps
        return hutchinson_divergence(velocity, x, eps, create_graph=False)

    def _exact_seeds(self, velocity: Tensor) -> list[Tensor]:
        """Identity rows broadcast over the batch, built once per solve."""
        seeds = self._seeds
        if seeds is None or seeds[0].shape != velocity.shape:
            basis = lucid.eye(self._dim, dtype=velocity.dtype, device=velocity.device)
            ones = lucid.ones_like(velocity[:, :1])
            seeds = [
                basis[index].reshape(1, self._dim) * ones for index in range(self._dim)
            ]
            self._seeds = seeds
        return seeds


# ─────────────────────────────────────────────────────────────────────────────
# Direct model
# ─────────────────────────────────────────────────────────────────────────────


# ``lucid.diffeq`` solvers that take exactly the grid they are handed.
# Anything else is adaptive and re-chooses its own step sizes.
_FIXED_STEP_SOLVERS: frozenset[str] = frozenset(
    {"euler", "midpoint", "heun2", "heun3", "rk4", "rk4_classic"}
)


class FlowMatchingModel(PretrainedModel):
    r"""Velocity field trained by Conditional Flow Matching.

    ``forward(sample, t)`` evaluates :math:`v_t(x; \theta)`.  The pieces
    that make it a generative model are separate and each maps onto one
    part of the paper:

    * :meth:`path_sample` — draw :math:`x_t` from the conditional path.
    * :meth:`conditional_target` — the field Theorem 3 says generates it.
    * :meth:`flow_matching_loss` — the objective, with no solve in it.
    * :meth:`sample` — integrate :math:`t: 0 \to 1`.
    * :meth:`log_prob` / :meth:`bits_per_dim` — integrate back, tracking
      the divergence.

    Parameters
    ----------
    config : FlowMatchingConfig
        Architecture, path and solver settings.

    Attributes
    ----------
    field : nn.Module
        The velocity field itself.

    Notes
    -----
    Reference: Lipman, Chen, Ben-Hamu, Nickel, and Le, *"Flow Matching for
    Generative Modeling"*, ICLR, 2023 (arXiv:2210.02747).  Theorem 2 gives
    the objective's equivalence to the intractable one; Theorem 3 gives
    the conditional field; §4.2 gives the optimal-transport path.

    **Where to run it.**  Training is a plain U-Net forward and backward
    with no solve, so it behaves like any convolutional model and belongs
    on ``"metal"``.  Sampling and likelihood are solves, and inherit the
    same guidance as any continuous flow: the GPU wins once the batch is
    real.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.flow_matching import (
    ...     FlowMatchingConfig, FlowMatchingModel,
    ... )
    >>> cfg = FlowMatchingConfig(sample_size=8, base_channels=16,
    ...                          channel_mult=(1, 2), num_res_blocks=1,
    ...                          attention_resolutions=(), resnet_groups=8)
    >>> model = FlowMatchingModel(cfg).eval()
    >>> loss, _, _ = model.flow_matching_loss(lucid.randn((2, 3, 8, 8)))
    >>> loss.shape
    ()
    >>> model.sample(n_samples=2, steps=4).shape
    (2, 3, 8, 8)
    """

    config_class: ClassVar[type[FlowMatchingConfig]] = FlowMatchingConfig
    base_model_prefix: ClassVar[str] = "flow_matching"

    def __init__(self, config: FlowMatchingConfig) -> None:
        super().__init__(config)
        self.field = _VelocityField(config)
        self.dynamics = _LikelihoodDynamics(self.field, config)

        size = config.sample_size
        height, width = size if isinstance(size, tuple) else (size, size)
        self._image_shape = (config.in_channels, height, width)
        self._input_dim = config.data_dim

        self._path = config.path
        self._sigma_min = config.sigma_min
        self._beta_min = config.beta_min
        self._beta_max = config.beta_max
        self._solver = config.solver
        self._rtol = config.rtol
        self._atol = config.atol
        self._trace_method = config.resolved_trace_method

    @property
    def input_dim(self) -> int:
        """int: Flattened width ``D`` of one sample."""
        return self._input_dim

    @property
    def path(self) -> str:
        """str: Which conditional probability path the target comes from."""
        return self._path

    @property
    def trace_method(self) -> str:
        """str: How the divergence is obtained when scoring likelihood."""
        return self._trace_method

    @property
    def nfe(self) -> int:
        """int: Field evaluations spent by the most recent solve.

        A training step solves nothing and leaves the count where the last
        solve put it.  The number the paper cares about: straighter paths
        need fewer.
        """
        return self.dynamics.nfe

    def _check_image(self, x: Tensor) -> None:
        channels, height, width = self._image_shape
        if x.ndim != 4 or tuple(int(s) for s in x.shape[1:]) != (
            channels,
            height,
            width,
        ):
            raise ValueError(
                f"Flow Matching expects (B, {channels}, {height}, {width}) samples, "
                f"got shape {tuple(x.shape)}"
            )

    def coefficients(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""``(a_t, sigma_t, a_t', sigma_t')`` for the configured path.

        Exposed because it is the only place the two paths differ, and
        because checking a schedule's derivative against a difference
        quotient is the cheapest way to know a new one is right.
        """
        return _path_coefficients(
            self._path,
            t,
            sigma_min=self._sigma_min,
            beta_min=self._beta_min,
            beta_max=self._beta_max,
        )

    def path_sample(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        r"""Draw :math:`x_t \sim p_t(\cdot \mid x_1)`, given the noise ``x0``.

        Parameters
        ----------
        x1 : Tensor
            ``(B, C, H, W)`` data samples — the conditioning point.
        x0 : Tensor
            ``(B, C, H, W)`` standard-normal samples.
        t : Tensor
            ``(B,)`` times in ``[0, 1]``.

        Returns
        -------
        Tensor
            ``(B, C, H, W)`` point on the conditional path.  For the
            optimal-transport path this is the straight-line interpolation
            of paper eq. (22).

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.flow_matching import (
        ...     FlowMatchingConfig, FlowMatchingModel,
        ... )
        >>> cfg = FlowMatchingConfig(sample_size=8, base_channels=16,
        ...                          channel_mult=(1, 2), num_res_blocks=1,
        ...                          attention_resolutions=(), resnet_groups=8)
        >>> model = FlowMatchingModel(cfg).eval()
        >>> x1, x0 = lucid.randn((2, 3, 8, 8)), lucid.randn((2, 3, 8, 8))
        >>> model.path_sample(x1, x0, lucid.tensor([0.25, 0.75])).shape
        (2, 3, 8, 8)
        >>> bool(lucid.allclose(model.path_sample(x1, x0, lucid.zeros(2)), x0))
        True
        >>> end = model.path_sample(x1, x0, lucid.ones(2))
        >>> bool(lucid.allclose(end, x1 + cfg.sigma_min * x0))  # still sigma_min wide
        True
        """
        a, sigma, _, _ = self.coefficients(t.reshape(-1, 1, 1, 1))
        return a * x1 + sigma * x0

    def conditional_target(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        r"""The field :math:`u_t(x \mid x_1)` that generates the path.

        Theorem 3 states it as
        :math:`(\sigma_t'/\sigma_t)(x - a_t x_1) + a_t' x_1`.  Since
        :math:`x - a_t x_1 = \sigma_t x_0` for the sample actually drawn,
        that reduces to :math:`\sigma_t' x_0 + a_t' x_1` — the same
        quantity with **no division by** :math:`\sigma_t`, which matters
        because :math:`\sigma_t \to 0` at ``t = 1`` on the diffusion path.
        For the optimal-transport path it collapses further, to the
        :math:`t`-independent :math:`x_1 - (1 - \sigma_{\min}) x_0` of
        paper eq. (23).

        Parameters
        ----------
        x1, x0 : Tensor
            ``(B, C, H, W)`` data and noise samples.
        t : Tensor
            ``(B,)`` times in ``[0, 1]``.

        Returns
        -------
        Tensor
            ``(B, C, H, W)`` regression target.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.flow_matching import (
        ...     FlowMatchingConfig, FlowMatchingModel,
        ... )
        >>> cfg = FlowMatchingConfig(sample_size=8, base_channels=16,
        ...                          channel_mult=(1, 2), num_res_blocks=1,
        ...                          attention_resolutions=(), resnet_groups=8)
        >>> model = FlowMatchingModel(cfg).eval()
        >>> x1, x0 = lucid.randn((2, 3, 8, 8)), lucid.randn((2, 3, 8, 8))
        >>> t = lucid.tensor([0.3, 0.6])
        >>> target = model.conditional_target(x1, x0, t)
        >>> target.shape
        (2, 3, 8, 8)
        >>> step = 1e-2  # the target is the time derivative of the path
        >>> ahead = model.path_sample(x1, x0, t + step)
        >>> slope = (ahead - model.path_sample(x1, x0, t)) / step
        >>> bool(lucid.allclose(slope, target, atol=1e-3))
        True
        >>> bool(lucid.allclose(target, x1 - (1 - cfg.sigma_min) * x0))  # eq. (23)
        True
        """
        _, _, a_dot, sigma_dot = self.coefficients(t.reshape(-1, 1, 1, 1))
        return a_dot * x1 + sigma_dot * x0

    def flow_matching_loss(self, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""One Conditional Flow Matching step — no ODE is solved.

        Parameters
        ----------
        x1 : Tensor
            ``(B, C, H, W)`` data batch.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(loss, prediction, target)`` — the scalar mean squared
            error, the field's output at the sampled points, and what it
            was regressed against.

        Notes
        -----
        ``t`` is drawn uniformly per sample, which is the estimator the
        objective is written as an expectation over.  Sharing one ``t``
        across the batch would still be unbiased but noisier.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.flow_matching import (
        ...     FlowMatchingConfig, FlowMatchingModel,
        ... )
        >>> cfg = FlowMatchingConfig(sample_size=8, base_channels=16,
        ...                          channel_mult=(1, 2), num_res_blocks=1,
        ...                          attention_resolutions=(), resnet_groups=8)
        >>> model = FlowMatchingModel(cfg).eval()
        >>> loss, prediction, target = model.flow_matching_loss(
        ...     lucid.randn((2, 3, 8, 8)))
        >>> loss.shape, prediction.shape, target.shape
        ((), (2, 3, 8, 8), (2, 3, 8, 8))
        >>> bool(lucid.allclose(loss, ((prediction - target) ** 2).mean()))
        True
        """
        self._check_image(x1)
        batch = int(x1.shape[0])
        t = lucid.rand((batch,), dtype=x1.dtype, device=x1.device)
        x0 = lucid.randn(x1.shape, dtype=x1.dtype, device=x1.device)

        x_t = self.path_sample(x1, x0, t)
        target = self.conditional_target(x1, x0, t)
        prediction = self.field(x_t, t)
        loss = ((prediction - target) ** 2).mean()
        return loss, prediction, target

    def _integrate(
        self,
        func: Callable[..., object],
        state: _State,
        t0: float,
        t1: float,
    ) -> _State:
        """One solve, endpoint only."""
        return cast(
            _State,
            lucid.diffeq.odeint(
                func,
                state,
                [t0, t1],
                rtol=self._rtol,
                atol=self._atol,
                method=self._solver,
                return_trajectory=False,
            ),
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
            Force a fixed-step solve with this many uniform steps instead
            of the adaptive method.  The honest way to report the
            few-evaluation sampling the straight paths are supposed to
            allow: fix the budget rather than the tolerance.
        device : str, optional
            Where to allocate the prior draw.  Defaults to the device the
            model's parameters are on.
        noise : Tensor, optional
            ``(N, C, H, W)`` starting point, in place of a fresh
            standard-normal draw — pass the same tensor to two models to
            compare them on identical initial conditions.

        Returns
        -------
        Tensor
            ``(N, C, H, W)`` samples at ``t = 1``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.flow_matching import (
        ...     FlowMatchingConfig, FlowMatchingModel,
        ... )
        >>> cfg = FlowMatchingConfig(sample_size=8, base_channels=16,
        ...                          channel_mult=(1, 2), num_res_blocks=1,
        ...                          attention_resolutions=(), resnet_groups=8)
        >>> model = FlowMatchingModel(cfg).eval()
        >>> model.sample(n_samples=2, steps=4).shape
        (2, 3, 8, 8)
        >>> model.nfe  # the adaptive default falls back to Euler: one call a step
        4
        >>> noise = lucid.randn((3, 3, 8, 8))
        >>> model.sample(noise=noise, steps=4).shape  # the batch follows noise
        (3, 3, 8, 8)
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
            out = cast(Tensor, self._integrate(self.dynamics.velocity, flat, 0.0, 1.0))
        else:
            if steps <= 0:
                raise ValueError(f"steps must be positive, got {steps}")
            grid = [index / steps for index in range(steps + 1)]
            out = cast(
                Tensor,
                lucid.diffeq.odeint(
                    self.dynamics.velocity,
                    flat,
                    grid,
                    # ``config.solver`` is documented as "method handed to
                    # odeint for sampling and likelihood", and hard-coding
                    # rk4 ignored it — while also spending four field
                    # evaluations per requested step, so ``steps=N`` cost 4N.
                    #
                    # But an *adaptive* method (the ``dopri5`` default) on a
                    # fixed grid makes ``steps`` stop being a budget at all,
                    # which is the one thing this branch exists to guarantee.
                    # So honour the configured method when it is fixed-step,
                    # and otherwise fall back to Euler — one evaluation per
                    # step, and what the reference's ``ODESolver.sample``
                    # defaults to.
                    method=(
                        self._solver if self._solver in _FIXED_STEP_SOLVERS else "euler"
                    ),
                    return_trajectory=False,
                ),
            )
        return out.reshape(batch, *self._image_shape)

    def log_prob(self, x: Tensor) -> Tensor:
        r"""Exact per-sample :math:`\log p_1(x)` in nats.

        Integrates the field from ``t = 1`` back to ``t = 0`` alongside the
        accumulated divergence, then scores the arrival against the
        standard normal the path starts from.

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
        on the same input will not agree.  That is the same estimator the
        paper reports bits/dim with.

        Examples
        --------
        >>> import math
        >>> import lucid
        >>> from lucid.models.generative.flow_matching import (
        ...     FlowMatchingConfig, FlowMatchingModel,
        ... )
        >>> cfg = FlowMatchingConfig(sample_size=8, base_channels=16,
        ...                          channel_mult=(1, 2), num_res_blocks=1,
        ...                          attention_resolutions=(), resnet_groups=8)
        >>> model = FlowMatchingModel(cfg).eval()
        >>> x = lucid.randn((2, 3, 8, 8))
        >>> log_p = model.log_prob(x)
        >>> log_p.shape
        (2,)
        >>> # A fresh field is exactly zero (its output layer starts at zero),
        >>> # so the flow is the identity and x is scored as N(0, I) itself.
        >>> const = 0.5 * model.input_dim * math.log(2 * math.pi)
        >>> gauss = -0.5 * (x.reshape(2, -1) ** 2).sum(dim=-1) - const
        >>> bool(lucid.allclose(log_p, gauss, atol=1e-3))
        True
        """
        self._check_image(x)
        batch = int(x.shape[0])
        flat = x.reshape(batch, -1)
        delta = lucid.zeros((batch,), dtype=flat.dtype, device=flat.device)

        eps = (
            None if self._trace_method == "exact" else self.dynamics.sample_noise(flat)
        )
        self.dynamics.reset(eps)

        x0, delta0 = cast(
            tuple[Tensor, Tensor],
            self._integrate(self.dynamics, (flat, delta), 1.0, 0.0),
        )
        # log p_0 of a standard normal, summed over the flattened sample.
        prior = -0.5 * (x0 * x0).sum(dim=-1) - 0.5 * self._input_dim * math.log(
            2.0 * math.pi
        )
        return prior + delta0

    def bits_per_dim(self, x: Tensor) -> Tensor:
        """Per-sample negative log-likelihood in bits per dimension."""
        return -self.log_prob(x) / (self._input_dim * math.log(2.0))

    @override
    def forward(self, sample: Tensor, t: Tensor) -> DiffusionModelOutput:
        """Evaluate the velocity field at ``(t, sample)``."""
        return DiffusionModelOutput(sample=self.field(sample, t))


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper
# ─────────────────────────────────────────────────────────────────────────────


class FlowMatchingForImageGeneration(ImageGenerationModel):
    r"""Flow Matching with the CFM training loss and ``.generate()``.

    ``forward(x)`` returns a :class:`DiffusionModelOutput` whose ``loss``
    is the Conditional Flow Matching objective and whose ``sample`` is the
    field's prediction at the sampled path points.  ``generate(n)``
    integrates the field from a standard-normal draw to ``t = 1``.

    Parameters
    ----------
    config : FlowMatchingConfig
        Architecture, path and solver settings.

    Attributes
    ----------
    flow_matching : FlowMatchingModel
        Underlying velocity field, exposing the path and likelihood API.

    Notes
    -----
    Reference: Lipman, Chen, Ben-Hamu, Nickel, and Le, *"Flow Matching for
    Generative Modeling"*, ICLR, 2023 (arXiv:2210.02747).  Reported for
    the optimal-transport path: 6.35 FID / 2.99 bits per dimension on
    CIFAR-10, and 5.02 / 3.53, 14.45 / 3.31, 20.9 / 2.90 on ImageNet at
    32, 64 and 128.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.flow_matching import (
    ...     FlowMatchingConfig, FlowMatchingForImageGeneration,
    ... )
    >>> cfg = FlowMatchingConfig(sample_size=8, base_channels=16,
    ...                          channel_mult=(1, 2), num_res_blocks=1,
    ...                          attention_resolutions=(), resnet_groups=8)
    >>> model = FlowMatchingForImageGeneration(cfg).eval()
    >>> model(lucid.randn((2, 3, 8, 8))).loss.shape
    ()
    >>> model.generate(n_samples=2, steps=4).samples.shape
    (2, 3, 8, 8)
    """

    config_class: ClassVar[type[FlowMatchingConfig]] = FlowMatchingConfig
    base_model_prefix: ClassVar[str] = "flow_matching"

    def __init__(self, config: FlowMatchingConfig) -> None:
        super().__init__(config)
        self.flow_matching = FlowMatchingModel(config)

    @property
    def nfe(self) -> int:
        """int: Field evaluations spent by the most recent solve."""
        return self.flow_matching.nfe

    @override
    def forward(self, x: Tensor) -> DiffusionModelOutput:
        loss, prediction, _ = self.flow_matching.flow_matching_loss(x)
        return DiffusionModelOutput(sample=prediction, loss=loss)

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
            Fixed evaluation budget instead of an adaptive solve — the
            regime in which straighter paths pay off.
        device : str, optional
            Where to allocate the prior draw.  Defaults to the device the
            model's parameters are on.
        noise : Tensor, optional
            Starting point in place of a fresh draw.

        Returns
        -------
        GenerationOutput
            ``samples`` of shape ``(n_samples, C, H, W)``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.flow_matching import (
        ...     FlowMatchingConfig, FlowMatchingForImageGeneration,
        ... )
        >>> cfg = FlowMatchingConfig(sample_size=8, base_channels=16,
        ...                          channel_mult=(1, 2), num_res_blocks=1,
        ...                          attention_resolutions=(), resnet_groups=8)
        >>> model = FlowMatchingForImageGeneration(cfg).eval()
        >>> model.generate(n_samples=3, steps=2).samples.shape
        (3, 3, 8, 8)
        >>> model.nfe  # a fixed budget: two steps, two field evaluations
        2
        >>> noise = lucid.randn((1, 3, 8, 8))
        >>> model.generate(noise=noise, steps=2).samples.shape  # batch follows noise
        (1, 3, 8, 8)
        """
        device = resolve_generation_device(self, device)
        return GenerationOutput(
            samples=self.flow_matching.sample(
                n_samples, steps=steps, device=device, noise=noise
            )
        )


__all__ = ["FlowMatchingModel", "FlowMatchingForImageGeneration"]
