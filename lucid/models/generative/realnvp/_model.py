"""RealNVP model — multi-scale affine-coupling flow (Dinh et al., 2016).

Shape of the flow (paper §3.6 / §4.1, in order):

    x (B, C, H, W) in [0, 1]
      → logit squash (contributes to the log-determinant)
      → for each scale i < L:
            3 checkerboard couplings (alternating parity)
            squeeze  (B, C, H, W) → (B, 4C, H/2, W/2)
            3 channel-wise couplings (alternating parity)
            factor out half the channels as z_i
      → final scale: 4 checkerboard couplings (alternating parity) → z_L
      → z = concat(flatten(z_1), …, flatten(z_L))   (B, D)

Every stage is invertible and reports its own log-determinant, so the
exact log-likelihood is one forward pass and sampling is the same graph
run backwards from a Gaussian draw.
"""

import math
from typing import ClassVar, cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageGenerationModel
from lucid.models._output import GenerationOutput, NormalizingFlowOutput
from lucid.models._utils._generative import flow_prior_log_prob, generative_activation, start_noise
from lucid.models.generative.realnvp._config import RealNVPConfig

# ─────────────────────────────────────────────────────────────────────────────
# Volume-preserving reshapes and masks
# ─────────────────────────────────────────────────────────────────────────────


def _squeeze(x: Tensor) -> Tensor:
    """``(B, C, H, W) → (B, 4C, H/2, W/2)`` — trade space for depth.

    Each ``2 x 2 x c`` block becomes ``1 x 1 x 4c`` (paper §3.6), so a
    channel-wise mask applied afterwards partitions along a different axis
    than the checkerboard did before.
    """
    b, c, h, w = (int(s) for s in x.shape)
    out = x.reshape(b, c, h // 2, 2, w // 2, 2)
    return out.permute(0, 1, 3, 5, 2, 4).reshape(b, 4 * c, h // 2, w // 2)


def _unsqueeze(x: Tensor) -> Tensor:
    """Inverse of :func:`_squeeze`."""
    b, c4, h, w = (int(s) for s in x.shape)
    c = c4 // 4
    out = x.reshape(b, c, 2, 2, h, w)
    return out.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * 2, w * 2)


def _checkerboard_mask(height: int, width: int, parity: int) -> Tensor:
    """``(1, 1, H, W)`` mask, 1 where ``(i + j + parity)`` is odd."""
    rows = [[float((i + j + parity) % 2) for j in range(width)] for i in range(height)]
    return lucid.tensor(rows).reshape(1, 1, height, width)


def _channel_mask(channels: int, parity: int) -> Tensor:
    """``(1, C, 1, 1)`` mask, 1 on the first half of the channels."""
    half = channels // 2
    values = [1.0] * half + [0.0] * (channels - half)
    if parity:
        values = [1.0 - v for v in values]
    return lucid.tensor(values).reshape(1, channels, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Coupling network — residual conv stack producing (s, t)
# ─────────────────────────────────────────────────────────────────────────────


def _maybe_weight_norm(conv: nn.Conv2d, enabled: bool) -> nn.Conv2d:
    r"""Wrap ``conv`` in weight normalisation when the config asks for it.

    Section 4.1 uses weight normalisation throughout the coupling
    networks; it reparametrises each filter as :math:`g\,v/\lVert v\rVert`
    so the direction and the magnitude are learned separately.
    """
    return cast(nn.Conv2d, nn.utils.weight_norm(conv)) if enabled else conv


@final
class _ResBlock(nn.Module):
    """Pre-activation residual block (ReLU + skip, paper §4.1)."""

    def __init__(
        self, dim: int, act_fn: str, use_batch_norm: bool, use_weight_norm: bool = False
    ) -> None:
        super().__init__()
        self._act_name = act_fn
        self.norm1 = nn.BatchNorm2d(dim) if use_batch_norm else None
        self.conv1 = _maybe_weight_norm(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1), use_weight_norm
        )
        self.norm2 = nn.BatchNorm2d(dim) if use_batch_norm else None
        self.conv2 = _maybe_weight_norm(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1), use_weight_norm
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = x if self.norm1 is None else self.norm1(x)
        h = generative_activation(self._act_name, h)
        h = self.conv1(h)
        h = h if self.norm2 is None else self.norm2(h)
        h = generative_activation(self._act_name, h)
        return x + self.conv2(h)


@final
class _CouplingNet(nn.Module):
    r"""Residual conv stack emitting the coupling's :math:`(s, t)`.

    ``s`` is a :math:`\tanh` multiplied by a learned scalar — bounded, so
    ``exp(s)`` cannot blow up early in training — and starts at zero, which
    makes every coupling the identity at initialisation.  ``t`` is the raw
    affine output.
    """

    def __init__(self, channels: int, dim: int, config: RealNVPConfig) -> None:
        super().__init__()
        self._act_name = config.act_fn
        wn = config.use_weight_norm
        self.stem = _maybe_weight_norm(nn.Conv2d(channels, dim, kernel_size=3, padding=1), wn)
        self.blocks = nn.ModuleList(
            [
                _ResBlock(dim, config.act_fn, config.use_batch_norm, wn)
                for _ in range(config.residual_blocks)
            ]
        )
        self.norm_out = nn.BatchNorm2d(dim) if config.use_batch_norm else None
        # The head stays un-normalised: weight norm's ``g`` would fight the
        # zero initialisation below that makes step 0 the identity flow.
        self.head = nn.Conv2d(dim, 2 * channels, kernel_size=3, padding=1)
        # Zero head ⇒ s = t = 0 ⇒ identity flow at step 0.
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
        # ``rescale`` must NOT also start at zero.  With s = rescale·tanh(head),
        # zeroing both makes an exact saddle: ∂s/∂rescale = tanh(0) = 0 and
        # ∂s/∂head = rescale·(1−tanh²) = 0, so neither can ever move and every
        # coupling stays additive — the model silently degrades to NICE, losing
        # the affine scaling that is Real NVP's whole contribution.  One is the
        # reference initialisation (``Rescale``'s weight is ones), and it keeps
        # the step-0 identity because tanh(0) is still 0.
        # One scale *per channel*, as the reference's ``Rescale(num_channels)``
        # does.  A single global scalar forces every channel of the coupling
        # to share one saturation point on the tanh, so a channel that needs a
        # wide log-scale and one that needs a narrow one cannot both be served.
        self.rescale = nn.Parameter(lucid.ones(1, channels, 1, 1))

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        if self.norm_out is not None:
            h = self.norm_out(h)
        h = generative_activation(self._act_name, h)
        out = self.head(h)

        channels = int(x.shape[1])
        log_scale = out[:, :channels]
        shift = out[:, channels:]
        return self.rescale * log_scale.tanh(), shift


# ─────────────────────────────────────────────────────────────────────────────
# Bijective stages
# ─────────────────────────────────────────────────────────────────────────────


@final
class _AffineCoupling(nn.Module):
    r"""Masked affine coupling (paper eq. 11).

    .. math::

        y = b \odot x + (1 - b) \odot
            \big(x \odot \exp(s(b \odot x)) + t(b \odot x)\big)

    with ``b`` either a spatial checkerboard or a channel-wise split.  The
    Jacobian is triangular, so the log-determinant is
    :math:`\sum (1 - b) \odot s` regardless of the coupling network's
    depth.
    """

    mask: Tensor

    def __init__(
        self,
        channels: int,
        dim: int,
        config: RealNVPConfig,
        *,
        mask_kind: str,
        parity: int,
        height: int,
        width: int,
    ) -> None:
        super().__init__()
        if mask_kind == "checkerboard":
            mask = _checkerboard_mask(height, width, parity)
        else:
            mask = _channel_mask(channels, parity)
        self.register_buffer("mask", mask)
        self.net = _CouplingNet(channels, dim, config)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        log_scale, shift = self.net(x * self.mask)
        keep = 1.0 - self.mask
        log_scale = log_scale * keep
        shift = shift * keep
        y = x * self.mask + keep * (x * log_scale.exp() + shift)
        return y, log_scale.reshape(int(x.shape[0]), -1).sum(dim=-1)

    def inverse(self, y: Tensor) -> Tensor:
        """Exact inverse — the coupling net sees the untouched half."""
        log_scale, shift = self.net(y * self.mask)
        keep = 1.0 - self.mask
        log_scale = log_scale * keep
        shift = shift * keep
        return y * self.mask + keep * ((y - shift) * (-log_scale).exp())


@final
class _FlowBatchNorm(nn.Module):
    r"""Batch normalisation as a flow stage (paper §3.7).

    Rescaling by the batch statistics is linear per dimension, so it slots
    into the flow with log-determinant
    :math:`-\tfrac{1}{2}\sum_i \log(\tilde{\sigma}_i^2 + \epsilon)`.
    Training uses the batch statistics (and updates the running estimates);
    ``eval()`` and :meth:`inverse` use the running estimates, which is what
    makes inversion well-defined for a single sample.
    """

    running_mean: Tensor
    running_var: Tensor

    def __init__(
        self,
        channels: int,
        *,
        momentum: float = 0.1,
        eps: float = 1e-4,
        use_running_stats: bool = False,
    ):
        super().__init__()
        self._momentum = momentum
        self._eps = eps
        self._use_running_stats = use_running_stats
        self.register_buffer("running_mean", lucid.zeros(1, channels, 1, 1))
        self.register_buffer("running_var", lucid.ones(1, channels, 1, 1))

    def _track(self, mean: Tensor, var: Tensor) -> None:
        """Fold one batch's statistics into the running estimates.

        Assigned as attributes, which keeps each buffer in its slot and,
        under a compile trace, writes it in place so the compiled call
        carries the update back (``Module._write_traced_buffer``).  This
        used to write ``self._buffers[...]`` directly, which bypassed that:
        a compiled flow went on reading the statistics it was traced with.
        """
        with lucid.no_grad():
            self.running_mean = (1.0 - self._momentum) * self.running_mean + self._momentum * mean
            self.running_var = (1.0 - self._momentum) * self.running_var + self._momentum * var

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Normalise ``x`` and return it beside the log-determinant.

        Evaluation always uses the running estimates — that is what makes
        :meth:`inverse` well-defined for a single sample.

        Training has two modes.  Ordinary batch norm normalises by the
        current minibatch; §3.7 / Appendix E's variant normalises by the
        running average instead, which is steadier when the batch is small
        enough that its own statistics are noisy.  ``use_running_stats``
        selects the paper's variant.  Either way the running estimates are
        updated from the current batch, so the two differ in what is
        *used*, not in what is tracked.
        """
        mean, var = self.running_mean, self.running_var
        observed: tuple[Tensor, Tensor] | None = None
        if self.training:
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = ((x - batch_mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            observed = (batch_mean, batch_var)
            if not self._use_running_stats:
                mean, var = observed
        y = (x - mean) / (var + self._eps).sqrt()
        # log|det| = -½ Σ log(σ² + ε), broadcast over the spatial extent.
        per_channel = -0.5 * (var + self._eps).log()
        spatial = int(x.shape[2]) * int(x.shape[3])
        log_det = per_channel.reshape(-1).sum() * spatial
        if observed is not None:
            # Folded in only after normalising: Appendix E normalises by the
            # average over *recent* minibatches, which must not include this
            # one.  Under a trace the write lands in the buffer in place, so
            # reading it after the write would see this batch already in it.
            self._track(*observed)
        batch = int(x.shape[0])
        return y, log_det.reshape(1).expand(batch)

    def inverse(self, y: Tensor) -> Tensor:
        """Exact inverse using the running statistics."""
        return y * (self.running_var + self._eps).sqrt() + self.running_mean


@final
class _LogitTransform(nn.Module):
    r"""Squash-then-logit preprocessing (paper §4.1).

    Pixel data is bounded and quantised, so the flow is fitted to
    :math:`\mathrm{logit}(\alpha + (1 - 2\alpha) x)` rather than to ``x``
    itself; ``alpha = (1 - c) / 2`` for the configured constraint ``c``.
    Its log-determinant belongs to the reported likelihood, which is why
    it lives inside the model instead of in a transform pipeline.

    In training mode the input is first *dequantised* — one sample of
    uniform noise spread over each quantisation bin — so the continuous
    density being fitted has a finite optimum.  Evaluation leaves ``x``
    alone, keeping ``encode``/``decode`` an exact round-trip.
    """

    def __init__(self, data_constraint: float, num_bits: int = 8) -> None:
        super().__init__()
        self._c = data_constraint
        self._levels = float(2**num_bits)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.training:
            # Uniform dequantisation (§4.1).  Without it the flow is fitted to
            # a discrete measure it can drive to unbounded density, so the
            # likelihood improves without bound and the reported bits/dim is
            # meaningless.  ``bits_per_dim`` already adds back the num_bits
            # term this scaling costs.
            noise = lucid.rand(*x.shape, device=x.device.type)
            x = (x * (self._levels - 1.0) + noise) / self._levels
        s = (x - 0.5) * self._c + 0.5
        y = s.log() - (1.0 - s).log()
        # d/dx logit(s(x)) = c / (s (1 - s)); written with softplus for range.
        per_elem = F.softplus(y) + F.softplus(-y) + math.log(self._c)
        return y, per_elem.reshape(int(x.shape[0]), -1).sum(dim=-1)

    def inverse(self, y: Tensor) -> Tensor:
        """Back to ``[0, 1]`` — sigmoid, then undo the squash."""
        return (F.sigmoid(y) - 0.5) / self._c + 0.5


# ─────────────────────────────────────────────────────────────────────────────
# RealNVP model — the bare bijection + its prior
# ─────────────────────────────────────────────────────────────────────────────


class RealNVPModel(PretrainedModel):
    r"""Bare RealNVP flow — multi-scale bijection plus the latent prior.

    Entry points mirror the rest of the flow families:

    * ``encode(x) -> (z, log_det)`` — images to a flat ``(B, D)`` latent.
    * ``decode(z) -> x`` — the *exact* inverse, back to ``(B, C, H, W)``.
    * ``log_prob(x) -> (B,)`` — exact log-likelihood in nats.
    * ``bits_per_dim(x) -> (B,)`` — the paper's reported metric.

    Inputs live in ``[0, 1]``: the logit squash that opens the flow is part
    of the model, so feeding already-logit data double-counts it.

    Parameters
    ----------
    config : RealNVPConfig
        Architecture, prior and preprocessing knobs; see
        :class:`RealNVPConfig`.

    Attributes
    ----------
    logit : nn.Module
        Squash-then-logit stage (paper §4.1).
    scales : nn.ModuleList
        One entry per scale, each a ``nn.ModuleList`` of coupling (and
        optional batch-norm) stages.

    Notes
    -----
    Reference: Dinh, Sohl-Dickstein, and Bengio, *"Density Estimation Using
    Real NVP"*, ICLR, 2017 (arXiv:1605.08803).

    With batch normalisation enabled the forward pass uses batch statistics
    while training, exactly as in the paper; ``decode`` always uses the
    running estimates, so round-trip exactness is an ``eval()``-mode
    property.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.realnvp import RealNVPConfig, RealNVPModel
    >>> cfg = RealNVPConfig(sample_size=8, num_scales=2, residual_blocks=1,
    ...                     base_dim=8)
    >>> model = RealNVPModel(cfg).eval()
    >>> x = lucid.rand((2, 3, 8, 8))
    >>> z, log_det = model.encode(x)
    >>> z.shape, log_det.shape
    ((2, 192), (2,))
    """

    config_class: ClassVar[type[RealNVPConfig]] = RealNVPConfig
    base_model_prefix: ClassVar[str] = "realnvp"

    def __init__(self, config: RealNVPConfig) -> None:
        super().__init__(config)
        self._prior = config.prior
        self._image_shape = config.image_shape
        self._input_dim = config.input_dim
        self._num_bits = config.num_bits
        self._num_scales = config.num_scales
        self._use_batch_norm = config.use_batch_norm

        self.logit = _LogitTransform(config.data_constraint, config.num_bits)

        _c, height, width = config.image_shape
        scales: list[nn.ModuleList[_AffineCoupling | _FlowBatchNorm]] = []
        # Per-scale bookkeeping so ``encode`` can stay a flat loop.
        self._scale_shapes: list[tuple[int, int, int]] = []

        channels = config.in_channels
        for idx in range(config.num_scales):
            dim = config.base_dim * 2**idx
            stages: list[_AffineCoupling | _FlowBatchNorm] = []
            last = idx == config.num_scales - 1

            # Checkerboard block — three couplings, or four at the last scale.
            for step in range(4 if last else 3):
                stages.append(
                    _AffineCoupling(
                        channels,
                        dim,
                        config,
                        mask_kind="checkerboard",
                        parity=step % 2,
                        height=height,
                        width=width,
                    )
                )
                if config.use_batch_norm:
                    stages.append(
                        _FlowBatchNorm(
                            channels,
                            use_running_stats=(config.batch_norm_running_stats_in_training),
                        )
                    )

            self._scale_shapes.append((channels, height, width))
            if not last:
                # Squeeze, then three channel-wise couplings on 4C channels.
                # The squeeze quadruples the channel count, and the reference
                # doubles the coupling net's width to match (``2 * mid_channels``
                # for the channel-wise stack).  Reusing the pre-squeeze ``dim``
                # left those couplings modelling 4C channels with the capacity
                # sized for C.
                height, width = height // 2, width // 2
                squeezed = channels * 4
                channel_dim = dim * 2
                for step in range(3):
                    stages.append(
                        _AffineCoupling(
                            squeezed,
                            channel_dim,
                            config,
                            mask_kind="channel",
                            parity=step % 2,
                            height=height,
                            width=width,
                        )
                    )
                    if config.use_batch_norm:
                        stages.append(
                            _FlowBatchNorm(
                                squeezed,
                                use_running_stats=(config.batch_norm_running_stats_in_training),
                            )
                        )
                # Half the channels are factored out; the rest carry on.
                channels = squeezed // 2

            scales.append(nn.ModuleList(stages))

        self.scales = nn.ModuleList(scales)

    @property
    def input_dim(self) -> int:
        """Flattened dimensionality ``D`` of the latent."""
        return self._input_dim

    @property
    def prior(self) -> str:
        """Name of the factorised latent prior."""
        return self._prior

    def _check_image(self, x: Tensor) -> None:
        c, h, w = self._image_shape
        if x.ndim != 4 or tuple(int(s) for s in x.shape[1:]) != (c, h, w):
            raise ValueError(
                f"RealNVP expects (B, {c}, {h}, {w}) samples, " f"got shape {tuple(x.shape)}"
            )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        r"""Map images in ``[0, 1]`` to the latent space.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(z, log_det)`` with ``z`` of shape ``(B, D)`` — the
            factored-out parts concatenated in scale order — and
            ``log_det`` of shape ``(B,)``, summing the logit stage, every
            coupling and every flow batch-norm.
        """
        self._check_image(x)
        batch = int(x.shape[0])

        h, log_det = self.logit(x)
        parts: list[Tensor] = []

        for idx in range(self._num_scales):
            last = idx == self._num_scales - 1
            checker_stages = (4 if last else 3) * (2 if self._use_batch_norm else 1)
            stages = self.scales[idx]

            for stage in list(stages)[:checker_stages]:
                h, inc = stage(h)
                log_det = log_det + inc

            if last:
                parts.append(h.reshape(batch, -1))
                break

            h = _squeeze(h)
            for stage in list(stages)[checker_stages:]:
                h, inc = stage(h)
                log_det = log_det + inc

            # Reference divergence, not a paper violation: §3.6 / Fig. 4b
            # only say half the dimensions are factored out and do not fix
            # the partition.  The reference un-squeezes after the channel
            # couplings and re-squeezes with a checkerboard-derived ordering
            # before chunking, so its two halves interleave differently from
            # this raw squeezed-order split.  The flow is a valid bijection
            # either way; what differs is which dimensions travel together.
            split = int(h.shape[1]) // 2
            parts.append(h[:, :split].reshape(batch, -1))
            h = h[:, split:]

        return lucid.cat(parts, dim=-1), log_det

    def decode(self, z: Tensor) -> Tensor:
        """Invert the flow — latent ``(B, D)`` back to images ``(B, C, H, W)``.

        Exact rather than approximate: every stage is run backwards, so
        ``decode(encode(x)[0])`` recovers ``x`` up to float round-off (in
        ``eval()`` mode when batch normalisation is enabled).
        """
        if z.ndim != 2 or int(z.shape[1]) != self._input_dim:
            raise ValueError(
                f"RealNVP expects (B, {self._input_dim}) latents, " f"got shape {tuple(z.shape)}"
            )
        batch = int(z.shape[0])

        # Split the flat latent back into the per-scale blocks.
        blocks: list[Tensor] = []
        offset = 0
        for idx, (channels, height, width) in enumerate(self._scale_shapes):
            last = idx == self._num_scales - 1
            # Every scale but the last factors out half of its squeezed
            # channels at half the spatial extent.
            if last:
                size = channels * height * width
                shape = (batch, channels, height, width)
            else:
                size = channels * height * width // 2
                shape = (batch, channels * 2, height // 2, width // 2)
            blocks.append(z[:, offset : offset + size].reshape(*shape))
            offset += size

        h = blocks[-1]
        for idx in range(self._num_scales - 1, -1, -1):
            last = idx == self._num_scales - 1
            checker_stages = (4 if last else 3) * (2 if self._use_batch_norm else 1)
            stages = list(self.scales[idx])

            if not last:
                h = lucid.cat([blocks[idx], h], dim=1)
                for stage in reversed(stages[checker_stages:]):
                    h = self._inverse_stage(stage, h)
                h = _unsqueeze(h)

            for stage in reversed(stages[:checker_stages]):
                h = self._inverse_stage(stage, h)

        return self.logit.inverse(h)

    @staticmethod
    def _inverse_stage(stage: _AffineCoupling | _FlowBatchNorm, h: Tensor) -> Tensor:
        return stage.inverse(h)

    def log_prob(self, x: Tensor) -> Tensor:
        r"""Exact per-sample log-likelihood :math:`\log p_X(x)` in nats."""
        z, log_det = self.encode(x)
        return flow_prior_log_prob(self._prior, z).sum(dim=-1) + log_det

    def bits_per_dim(self, x: Tensor) -> Tensor:
        r"""Per-sample bits/dim — the metric the paper reports in Table 1.

        :meth:`log_prob` is the exact density over the model's own input
        space, ``[0, 1]^D``.  The paper's number is measured against the
        *discrete* 8-bit data, and the ``x / 256`` inside its squash is
        precisely that change of variables — worth
        :math:`-D \log 256` in nats, i.e. ``num_bits`` in bits/dim.  The
        canonical implementation applies the same offset in its loss.
        Without it the reported figure sits a full 8 bits/dim below Table 1
        and is not comparable to any published result.
        """
        nats = -self.log_prob(x) / (self._input_dim * math.log(2.0))
        return nats + float(self._num_bits)

    @override
    def forward(self, x: Tensor) -> NormalizingFlowOutput:
        z, log_det = self.encode(x)
        log_prob = flow_prior_log_prob(self._prior, z).sum(dim=-1) + log_det
        return NormalizingFlowOutput(latent=z, log_det_jacobian=log_det, log_prob=log_prob)


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper — maximum-likelihood loss + ancestral sampling
# ─────────────────────────────────────────────────────────────────────────────


class RealNVPForImageGeneration(ImageGenerationModel):
    r"""RealNVP with the bits/dim training loss and ``.generate()``.

    ``forward(x)`` returns a :class:`NormalizingFlowOutput` whose ``loss``
    is the mean negative log-likelihood in **bits per dimension** — the
    quantity the paper reports, and the scale-free form that keeps the
    gradient magnitude independent of image size.
    ``generate(n_samples)`` draws :math:`z \sim \mathcal{N}(0, I)` and
    returns :math:`f^{-1}(z)` in a single parallel pass, back in ``[0, 1]``
    pixel space.

    Parameters
    ----------
    config : RealNVPConfig
        Architecture, prior and preprocessing knobs.

    Attributes
    ----------
    realnvp : RealNVPModel
        Underlying bijection, exposing ``encode`` / ``decode`` /
        ``log_prob`` / ``bits_per_dim``.

    Notes
    -----
    Reference: Dinh, Sohl-Dickstein, and Bengio, *"Density Estimation Using
    Real NVP"*, ICLR, 2017 (arXiv:1605.08803).  Reported bits/dim: 3.49
    (CIFAR-10), 4.28 / 3.98 (Imagenet 32 / 64), 2.72–3.08 (LSUN), 3.02
    (CelebA).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.realnvp import (
    ...     RealNVPConfig, RealNVPForImageGeneration,
    ... )
    >>> cfg = RealNVPConfig(sample_size=8, num_scales=2, residual_blocks=1,
    ...                     base_dim=8)
    >>> model = RealNVPForImageGeneration(cfg).eval()
    >>> out = model(lucid.rand((2, 3, 8, 8)))
    >>> out.loss.shape                      # scalar bits/dim
    ()
    >>> model.generate(n_samples=3).samples.shape
    (3, 3, 8, 8)
    """

    config_class: ClassVar[type[RealNVPConfig]] = RealNVPConfig
    base_model_prefix: ClassVar[str] = "realnvp"

    def __init__(self, config: RealNVPConfig) -> None:
        super().__init__(config)
        self.realnvp = RealNVPModel(config)
        self._prior = config.prior
        self._input_dim = config.input_dim
        self._num_bits = config.num_bits

    @override
    def forward(self, x: Tensor) -> NormalizingFlowOutput:
        out = self.realnvp(x)
        # Same 8-bit discretisation offset ``bits_per_dim`` applies — a
        # constant, so gradients are untouched, but the reported loss is now
        # on the paper's scale.
        bits = -out.log_prob / (self._input_dim * math.log(2.0)) + float(self._num_bits)
        return NormalizingFlowOutput(
            latent=out.latent,
            log_det_jacobian=out.log_det_jacobian,
            log_prob=out.log_prob,
            loss=bits.mean(),
        )

    @lucid.no_grad()
    def generate(
        self, n_samples: int = 1, *, temperature: float = 1.0, device: str | None = None
    ) -> GenerationOutput:
        """Sample images by inverting a prior draw.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to draw.
        temperature : float, default=1.0
            Multiplies the prior sample before inversion.  Values below
            ``1`` trade diversity for typicality — a standard trick for
            flows, not part of the paper.
        device : str, optional
            Where to allocate the prior sample.  Defaults to the device
            the model's parameters are on.

        Returns
        -------
        GenerationOutput
            ``samples`` of shape ``(n_samples, C, H, W)`` in pixel space —
            the logit stage is undone on the way out.  The squash is not
            saturating, so values land just outside ``[0, 1]`` (within
            ``±(1 - c) / 2c``); clip before display.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.realnvp import (
        ...     RealNVPConfig, RealNVPForImageGeneration,
        ... )
        >>> cfg = RealNVPConfig(sample_size=8, num_scales=2, residual_blocks=1,
        ...                     base_dim=8)
        >>> model = RealNVPForImageGeneration(cfg).eval()
        >>> samples = model.generate(n_samples=3).samples
        >>> samples.shape
        (3, 3, 8, 8)
        >>> c = cfg.data_constraint
        >>> pad = (1 - c) / (2 * c)
        >>> bool(((samples >= -pad) & (samples <= 1 + pad)).all().item())
        True
        >>> model.generate(temperature=0.0)
        Traceback (most recent call last):
            ...
        ValueError: temperature must be positive, got 0.0
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        z = start_noise(self, (n_samples, self._input_dim), device, prior=self._prior)
        return GenerationOutput(samples=self.realnvp.decode(z * temperature))
