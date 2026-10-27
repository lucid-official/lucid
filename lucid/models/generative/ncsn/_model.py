"""NCSN model (Song & Ermon, 2019).

Reuses :class:`_DDPMUNet` for the score network (every NCSN variant from
v1 onward converged on the same modern U-Net architecture as diffusion).
The NCSN-specific pieces are:

    1. Geometric ``σ`` schedule (via :func:`make_sigma_schedule`).
    2. Denoising score matching (DSM) loss — Song 2019 Eq. (6).
    3. Annealed Langevin dynamics sampler — Algorithm 1.

Sampling uses the per-σ step-size rule from §4.3:
    ``ε_i = ε · σ_i² / σ_L²`` (where σ_L is the *smallest* noise).
"""

import math
from typing import ClassVar, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageGenerationModel
from lucid.models._output import DiffusionModelOutput, GenerationOutput
from lucid.models._utils._generative import (
    make_sigma_schedule,
    model_dtype,
    resolve_generation_device,
)
from lucid.models.generative.ddpm._config import DDPMConfig
from lucid.models.generative.ncsn._refinenet import RefineNetScoreNet
from lucid.models.generative.ddpm._model import _DDPMUNet
from lucid.models.generative.ncsn._config import NCSNConfig
from lucid.models._utils._common import host_floats

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _to_unet_config(cfg: NCSNConfig) -> DDPMConfig:
    """Build a :class:`DDPMConfig` shadow so we can instantiate the shared
    ``_DDPMUNet`` from an :class:`NCSNConfig` (the U-Net never sees the
    sigma-schedule / Langevin knobs)."""
    return DDPMConfig(
        sample_size=cfg.sample_size,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        act_fn=cfg.act_fn,
        base_channels=cfg.base_channels,
        channel_mult=cfg.channel_mult,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        resnet_groups=cfg.resnet_groups,
        # The DDPM-side knobs below aren't used by the U-Net but the config
        # still validates them — keep defaults that pass.
        num_train_timesteps=max(cfg.num_noise_levels, 2),
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        learn_sigma=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bare NCSN trunk
# ─────────────────────────────────────────────────────────────────────────────


class NCSNModel(PretrainedModel):
    r"""Score network :math:`s_\theta(\tilde{x}, \sigma_i)` over a geometric :math:`\sigma` schedule.

    Implements the noise-conditional score network of Song and Ermon, 2019
    (and the NCSNv2 / NCSN++ refinements of Song 2020, Song et al. 2021).
    Reuses :class:`_DDPMUNet` as the score backbone — modern NCSN variants
    converged on the same U-Net architecture as diffusion — but feeds the
    integer noise-level index :math:`i \in [0, L)` rather than a raw
    :math:`\sigma` value through the timestep embedding.  Divergence,
    deliberate: this matches none of the three published recipes exactly.
    NCSN v1 also conditions on the index, but through
    CondInstanceNorm++ per-:math:`\sigma` gain/bias rather than an
    additive MLP; NCSNv2's Technique 3 drops noise conditioning entirely
    and divides the output by :math:`\sigma`; NCSN++ conditions on
    *continuous* :math:`\sigma` via Fourier features of
    :math:`\log \sigma`.  Feeding the index through the timestep MLP
    keeps :class:`_DDPMUNet` reusable unchanged, which is what makes one
    backbone serve both families here.  The geometric :math:`\sigma` table itself is
    held as a non-persistent buffer so it travels with ``.to(device=...)``.

    Parameters
    ----------
    config : NCSNConfig
        Hyperparameters controlling the U-Net topology
        (``base_channels``, ``channel_mult``, ``num_res_blocks``,
        ``attention_resolutions``, ``num_heads``, ``dropout``,
        ``resnet_groups``) and the score-based extras
        (``num_noise_levels``, ``sigma_max``, ``sigma_min``).

    Attributes
    ----------
    unet : _DDPMUNet
        Shared U-Net backbone instantiated from a derived
        :class:`DDPMConfig`.
    sigmas : Tensor
        Buffer of shape ``(num_noise_levels,)`` holding the geometric
        :math:`\sigma_1 > \sigma_2 > \cdots > \sigma_L` schedule.

    Notes
    -----
    Reference: Song and Ermon, *"Generative Modeling by Estimating
    Gradients of the Data Distribution"*, NeurIPS, 2019
    (arXiv:1907.05600); NCSNv2 refinements in Song and Ermon, 2020
    (arXiv:2006.09011).

    The trained score satisfies

    .. math::

        s_\theta(\tilde{x}, \sigma)
            \;\approx\; \nabla_{\tilde{x}} \log p_\sigma(\tilde{x}),

    where :math:`p_\sigma` is the data distribution convolved with a
    Gaussian of standard deviation :math:`\sigma`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ncsn import NCSNConfig, NCSNModel
    >>> cfg = NCSNConfig(sample_size=32, base_channels=32,
    ...                  channel_mult=(1, 2), num_res_blocks=1,
    ...                  resnet_groups=16, num_noise_levels=10)
    >>> model = NCSNModel(cfg).eval()
    >>> x_tilde = lucid.randn((1, 3, 32, 32))
    >>> sigma_idx = lucid.tensor([3]).long()
    >>> out = model(x_tilde, sigma_idx)
    >>> out.sample.shape   # (1, 3, 32, 32) — raw score
    (1, 3, 32, 32)
    """

    config_class: ClassVar[type[NCSNConfig]] = NCSNConfig
    base_model_prefix: ClassVar[str] = "ncsn"

    sigmas: Tensor

    def __init__(self, config: NCSNConfig) -> None:
        super().__init__(config)
        # Divergence, deliberate: NCSN v1 §4.1 and NCSNv2 both specify a
        # RefineNet — a 4-cascade U-Net with CondInstanceNorm++ (v1) or
        # InstanceNorm++ (v2), ELU throughout, and dilated convolutions in
        # place of subsampling.  ``backbone="refinenet"`` builds that; the
        # default reuses the DDPM U-Net (GroupNorm, SiLU, self-attention,
        # sinusoidal timestep embedding), which is what NCSN++ later moved
        # to and what every factory here has always produced.  Neither
        # loads a released v1/v2 checkpoint — the weights are not published
        # in a form this can convert — so the choice is about inductive
        # bias, not about weight compatibility.
        self.unet: RefineNetScoreNet | _DDPMUNet
        if config.backbone == "refinenet":
            self.unet = RefineNetScoreNet(
                config.in_channels,
                base_channels=config.base_channels,
                num_classes=config.num_noise_levels,
                channel_mult=tuple(config.channel_mult),
            )
        else:
            self.unet = _DDPMUNet(_to_unet_config(config))
        sigmas = make_sigma_schedule(
            config.num_noise_levels,
            sigma_max=config.sigma_max,
            sigma_min=config.sigma_min,
        )
        self.register_buffer("sigmas", sigmas, persistent=False)
        self._num_levels = config.num_noise_levels
        self._scale_by_sigma = config.scale_by_sigma

    @property
    def num_noise_levels(self) -> int:
        return self._num_levels

    @override
    def forward(
        self,
        sample: Tensor,
        sigma_idx: Tensor,
    ) -> DiffusionModelOutput:
        """Predict the score at ``sample`` for the given noise-level indices.

        Args:
            sample:    ``(B, in_channels, H, W)`` perturbed image ``x̃``.
            sigma_idx: ``(B,)`` integer indices into the σ schedule.

        Returns:
            :class:`DiffusionModelOutput` whose ``sample`` field is the raw
            score prediction (same shape as ``sample``).
        """
        raw = self.unet(sample, sigma_idx)
        if not self._scale_by_sigma:
            return DiffusionModelOutput(sample=raw)

        # NCSNv2 Technique 3: s_theta(x, sigma) = s_theta(x) / sigma.
        #
        # The true score norm scales like 1/sigma, and every factory here runs
        # sigma from 50 down to 0.01 — a 5000x dynamic range.  Returning the
        # trunk output unscaled makes the network learn that range through a
        # shared conv stack fed only an additive level embedding, which is the
        # exact failure NCSNv2 sec. 3 identifies.  Dividing here also makes the
        # DSM residual ``sigma * score + z`` collapse to ``raw + z``, so the
        # regression target stops depending on sigma at all.
        used_sigmas = self.sigmas[sigma_idx].reshape(-1, *([1] * (len(sample.shape) - 1)))
        return DiffusionModelOutput(sample=raw / used_sigmas)


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper — DSM loss + annealed Langevin sampling
# ─────────────────────────────────────────────────────────────────────────────


class NCSNForImageGeneration(ImageGenerationModel):
    r"""NCSN with the DSM training loss and annealed Langevin ``.generate()``.

    Wraps :class:`NCSNModel` with denoising score matching (DSM) for
    training and annealed Langevin dynamics for inference.  Sampling does
    **not** reuse :class:`DiffusionMixin.generate` because NCSN's nested
    (per-:math:`\sigma` x per-step) loop has different semantics from the
    DDPM scheduler step.

    Training contract
        ``forward(x)`` samples a random noise-level index per image,
        perturbs ``x`` accordingly, runs the score network, and returns
        the DSM loss (Song 2019 Eq. 6).

    Sampling
        ``generate(n_samples)`` runs annealed Langevin dynamics across the
        full :math:`\sigma` schedule (Song 2019 Algorithm 1).

    Parameters
    ----------
    config : NCSNConfig
        Hyperparameters controlling architecture, noise schedule, and
        Langevin sampler.

    Attributes
    ----------
    ncsn : NCSNModel
        Underlying score network (see :class:`NCSNModel`).
    sigmas : Tensor
        Buffer of shape ``(num_noise_levels,)`` holding the geometric
        :math:`\sigma` schedule (mirrored from ``self.ncsn.sigmas`` for
        convenience in loss / sampling code).

    Notes
    -----
    Reference: Song and Ermon, *"Generative Modeling by Estimating
    Gradients of the Data Distribution"*, NeurIPS, 2019 (arXiv:1907.05600);
    NCSNv2 refinements in Song and Ermon, 2020 (arXiv:2006.09011).

    Denoising score matching loss:

    .. math::

        \mathcal{L}_{\text{DSM}}(\theta)
            = \tfrac{1}{2}\,
              \mathbb{E}_{\sigma, x, \tilde{x}}\!\left[
                \big\lVert
                    \sigma\, s_\theta(\tilde{x}, \sigma) + z
                \big\rVert^2
              \right],

    with :math:`\tilde{x} = x + \sigma z`, :math:`z \sim
    \mathcal{N}(0, \mathbf{I})`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ncsn import (
    ...     NCSNConfig, NCSNForImageGeneration,
    ... )
    >>> cfg = NCSNConfig(sample_size=32, base_channels=32,
    ...                  channel_mult=(1, 2), num_res_blocks=1,
    ...                  resnet_groups=16, num_noise_levels=10,
    ...                  langevin_steps=2)
    >>> model = NCSNForImageGeneration(cfg).eval()
    >>> x = lucid.randn((1, 3, 32, 32))
    >>> out = model(x)
    >>> out.sample.shape, out.loss.shape   # score field + scalar loss
    ((1, 3, 32, 32), ())
    """

    config_class: ClassVar[type[NCSNConfig]] = NCSNConfig
    base_model_prefix: ClassVar[str] = "ncsn"

    sigmas: Tensor

    def __init__(self, config: NCSNConfig) -> None:
        super().__init__(config)
        self.ncsn = NCSNModel(config)
        # Mirror the σ buffer for convenience in loss / sampling code.
        self.register_buffer(
            "sigmas",
            make_sigma_schedule(
                config.num_noise_levels,
                sigma_max=config.sigma_max,
                sigma_min=config.sigma_min,
            ),
            persistent=False,
        )
        self._num_levels = config.num_noise_levels
        self._sigma_min = config.sigma_min
        self._langevin_steps = config.langevin_steps
        self._langevin_eps = config.langevin_eps
        self._denoise = config.denoise
        self._in_channels = config.in_channels
        if isinstance(config.sample_size, tuple):
            self._h = int(config.sample_size[0])
            self._w = int(config.sample_size[1])
        else:
            self._h = self._w = int(config.sample_size)

    # ── Training ────────────────────────────────────────────────────────────

    @override
    def forward(self, sample: Tensor) -> DiffusionModelOutput:
        """Denoising Score Matching loss (Song 2019 Eq. 6).

        For each image we sample ``i ~ Uniform({0, …, L-1})``, perturb
        ``x̃ = x + σ_i · z`` with ``z ~ N(0, I)``, and minimise

            L = ½ · E ‖ σ_i · s_θ(x̃, i) + z ‖²

        which is the σ-weighted DSM loss.
        """
        B = int(sample.shape[0])
        dev = sample.device.type
        # A noise level per image, drawn and looked up on the device.  Drawn
        # one image at a time and read back with ``.item()`` — twice per
        # image — it stalled every step on a device sync and kept the
        # training step from compiling: a value read on the host is frozen
        # into a compiled graph.
        sigma_idx = lucid.randint(0, self._num_levels, (B,), device=dev).long()
        # (B, 1, 1, 1) for broadcast.
        sigma = self.sigmas[sigma_idx].reshape(B, 1, 1, 1)

        z = lucid.randn(sample.shape, dtype=sample.dtype, device=dev)
        x_tilde = sample + sigma * z
        score = self.ncsn(x_tilde, sigma_idx).sample  # ŝ_θ(x̃, i)

        residual = sigma * score + z  # (B, C, H, W)
        # The objective is a squared L2 *norm* — a sum over C.H.W — with the
        # mean taken over the batch only.  Reducing with .mean() over every
        # axis divides the paper's loss by C.H.W, which rescales the gradient
        # relative to any published learning rate by the same factor.
        per_sample = (residual * residual).reshape(B, -1).sum(dim=-1)
        loss = 0.5 * per_sample.mean()
        return DiffusionModelOutput(sample=score, loss=loss)

    # ── Sampling ────────────────────────────────────────────────────────────

    @lucid.no_grad()
    def generate(
        self,
        n_samples: int = 1,
        *,
        langevin_steps: int | None = None,
        return_intermediates: bool = False,
        device: str | None = None,
    ) -> GenerationOutput:
        """Annealed Langevin dynamics — Song 2019 Algorithm 1.

        Initialises ``x ~ N(0, σ_max² I)`` then iterates over the σ
        schedule from largest to smallest noise.  At each σ_i takes
        ``langevin_steps`` Langevin steps with step size
        ``α_i = ε · σ_i² / σ_L²``:

            x ← x + (α_i / 2) · s_θ(x, i) + √α_i · z      z ~ N(0, I)

        Args:
            n_samples:           Batch size of the generated images.
            langevin_steps:      Override ``config.langevin_steps`` for this
                                 call (typically lowered in unit tests).
            return_intermediates: If True, also returns one sample per σ
                                  level (after its inner Langevin loop).
            device:              Where to allocate the initial noise.

        Returns:
            :class:`GenerationOutput` with the final ``(n_samples, C, H, W)``
            samples and optional per-level intermediates.
        """
        device = resolve_generation_device(self, device)
        T = langevin_steps if langevin_steps is not None else self._langevin_steps

        shape = (n_samples, self._in_channels, self._h, self._w)
        # Algorithm 1 initialises from "some fixed prior (e.g. uniform
        # noise)", and both official repos seed annealed Langevin with
        # ``rand``.  The Gaussian N(0, sigma_1^2) start belongs to the later
        # VE-SDE formulation, not to NCSN or NCSNv2 — the previous comment
        # attributed it to NCSNv2, which is where the mix-up came from.
        dtype = model_dtype(self)
        x = lucid.rand(shape, dtype=dtype, device=device)

        intermediates: list[Tensor] = []
        sigma_min_sq = self._sigma_min**2
        for i, sigma_i in enumerate(host_floats(self.sigmas)):
            alpha_i = self._langevin_eps * (sigma_i**2) / sigma_min_sq
            sigma_idx = lucid.tensor([i] * n_samples, device=device).long()
            sqrt_alpha = math.sqrt(alpha_i)
            for _ in range(T):
                score = self.ncsn(x, sigma_idx).sample  # (B, C, H, W)
                z = lucid.randn(shape, dtype=dtype, device=device)
                x = x + (alpha_i / 2.0) * score + sqrt_alpha * z
            if return_intermediates:
                intermediates.append(x)

        if self._denoise:
            # NCSNv2 Technique 5: one noiseless correction at the end,
            # ``x_T + sigma_T^2 * s_theta(x_T, sigma_T)``.  Both released
            # configs set ``denoise: true``, and Table 1 reports FID with and
            # without it — the last Langevin step still leaves sigma_L-scale
            # noise on the sample otherwise.
            last = self._num_levels - 1
            sigma_last = float(self.sigmas[last].item())
            idx_last = lucid.tensor([last] * n_samples, device=device).long()
            score_last = self.ncsn(x, idx_last).sample
            x = x + (sigma_last**2) * score_last

        return GenerationOutput(
            samples=x,
            intermediates=tuple(intermediates) if return_intermediates else None,
        )
