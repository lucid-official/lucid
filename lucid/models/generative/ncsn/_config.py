"""NCSN configuration — Song & Ermon, 2019 ("Generative Modeling by
Estimating Gradients of the Data Distribution") + NCSNv2 (Song 2020).

Score-based generative model: learns ``s_θ(x̃, σ) ≈ ∇_x̃ log p_σ(x̃)`` for a
schedule of noise levels ``σ_1 > σ_2 > … > σ_L``, then samples via annealed
Langevin dynamics across that schedule.

Architecture-wise we reuse :class:`_DDPMUNet` (NCSNv2 / NCSN++ converged on
the same modern U-Net as diffusion).  The differences live in:

    * Noise schedule: geometric ``σ`` levels instead of diffusion ``β``.
    * Loss:           denoising score matching (DSM) — see
                      :meth:`NCSNForImageGeneration.forward`.
    * Sampling:       annealed Langevin dynamics — see
                      :meth:`NCSNForImageGeneration.generate`.
"""

from dataclasses import dataclass, field
from typing import ClassVar, override, Literal

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    GenerativeModelConfig,
    check_sample_space_channels,
)


@model_family_meta(
    canonical_name="NCSN",
    citation=(
        'Song, Yang, and Stefano Ermon. "Generative Modeling by '
        'Estimating Gradients of the Data Distribution." NeurIPS, 2019.'
    ),
    theory=r"""
    NCSN — *Noise-Conditional Score Network* — is the score-based
    counterpart of DDPM.  Rather than predicting noise inside a discrete
    Markov chain, it directly learns the **(Stein) score function**
    :math:`s_\theta(x, \sigma) \approx \nabla_x \log p_\sigma(x)`, where
    :math:`p_\sigma` is the data distribution convolved with Gaussian
    noise of standard deviation :math:`\sigma`.

    Training uses **denoising score matching** (Vincent 2011): for a
    noised sample
    :math:`\tilde{x} = x + \sigma \epsilon,\; \epsilon \sim
    \mathcal{N}(0, \mathbf{I})`, the conditional score is available in
    closed form,
    :math:`\nabla_{\tilde{x}} \log p_\sigma(\tilde{x} \mid x) =
    -(\tilde{x} - x)/\sigma^2`, giving the loss

    .. math::

        \mathcal{L}_{\text{DSM}}(\theta)
            = \mathbb{E}_{\sigma, x, \tilde{x}}\!\left[
                \frac{1}{2}\, \lambda(\sigma)
                \left\lVert
                    s_\theta(\tilde{x}, \sigma)
                    + \frac{\tilde{x} - x}{\sigma^2}
                \right\rVert^2
              \right],

    with the standard weighting :math:`\lambda(\sigma) = \sigma^2` so
    every noise level contributes comparable magnitude.  The
    architecture reuses the same U-Net as DDPM but conditions on
    :math:`\sigma` (continuous) rather than a discrete timestep
    :math:`t`.

    The key innovation is **noise annealing**: a geometric schedule
    :math:`\sigma_1 > \sigma_2 > \cdots > \sigma_L` (defaults
    :math:`\sigma_1 = 50`, :math:`\sigma_L = 0.01`, :math:`L = 10`)
    where the largest :math:`\sigma_1` smooths the data distribution
    enough that low-density regions become traversable, then sampling
    refines toward :math:`\sigma_L \approx 0` where the score
    approximates that of clean data.  Sampling uses **annealed Langevin
    dynamics** — for each level :math:`i` in order, run :math:`T_i`
    steps of

    .. math::

        x \leftarrow x + \frac{\alpha_i}{2}\, s_\theta(x, \sigma_i)
            + \sqrt{\alpha_i}\; z,
        \qquad z \sim \mathcal{N}(0, \mathbf{I}),

    with the per-level step size
    :math:`\alpha_i = \epsilon \cdot \sigma_i^2 / \sigma_L^2` (Song
    2019, §4.3) — large early on, decaying as noise shrinks.  The
    score-based view unifies with diffusion through the **continuous
    SDE framework** (Song et al., 2021), where DDPM and NCSN become
    different discretisations of the same reverse-time process.
    """,
)
@dataclass(frozen=True)
class NCSNConfig(GenerativeModelConfig):
    """Configuration for every NCSN variant.

    Args:
        base_channels: First U-Net stage width (paper: 128).
        channel_mult: Per-stage channel multiplier.  Same semantics as
            :class:`DDPMConfig`.
        num_res_blocks: ResBlocks per stage.
        attention_resolutions: Spatial sizes at which to insert
            self-attention.
        num_heads: Attention head count.
        dropout: Dropout inside ResBlocks.
        resnet_groups: GroupNorm group count.
        num_noise_levels: Number of σ levels ``L`` (paper L=10, NCSNv2 large
            L ≈ 200–500).
        sigma_max: Largest noise σ_1.  50.0 belongs to NCSNv2 (Technique 1)
            and goes with L = 232; NCSN v1 uses σ_1 = 1.0 with L = 10.
            Mixing the two breaks Technique 2's ratio criterion.
        sigma_min: Smallest noise σ_L (paper CIFAR-10: 0.01).
        langevin_steps: Sampler steps per σ level (paper T=100).
        langevin_eps: Sampler step-size base (paper ε=2e-5).  Per-σ step
            size is ``ε · σ_i² / σ_L²`` (Song 2019 §4.3).
        scale_by_sigma: Divide the network output by σ (NCSNv2 Technique 3).
            Default True; set False for the NCSN v1 parameterisation.

    Examples
    --------
    >>> from lucid.models.generative.ncsn._config import NCSNConfig
    >>> cfg = NCSNConfig()
    >>> cfg.model_type
    'ncsn'
    >>> cfg.in_channels
    3
    """

    model_type: ClassVar[str] = "ncsn"

    # U-Net architectural knobs (mirrored from DDPMConfig — same arch).
    base_channels: int = 128
    channel_mult: tuple[int, ...] = field(default_factory=lambda: (1, 2, 2, 2))
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = field(default_factory=lambda: (16,))
    num_heads: int = 1
    dropout: float = 0.0
    resnet_groups: int = 32
    # "ddpm_unet" reuses the diffusion U-Net (GroupNorm / SiLU / attention /
    # sinusoidal t), which is what NCSN++ moved to and what every factory
    # here has always built.  "refinenet" is the backbone v1/v2 actually
    # specify — a 4-cascade RefineNet with conditional InstanceNorm++, ELU,
    # and dilation in place of subsampling.  Default unchanged so no
    # existing run shifts under a caller.
    backbone: Literal["ddpm_unet", "refinenet"] = "ddpm_unet"

    # Score-based: sigma schedule + Langevin sampler.
    # v1 defaults: sigma_1 = 1.0 with L = 10.  The registered factories
    # override both together with the v2 pairs (50/232, 90/500) — pairing
    # sigma_max=50 with L=10 is neither paper's setting.
    num_noise_levels: int = 10
    sigma_max: float = 1.0
    sigma_min: float = 0.01
    langevin_steps: int = 100
    langevin_eps: float = 2e-5
    # NCSNv2 Technique 3 — parameterise the network as s_theta(x, sigma) =
    # s_theta(x) / sigma.  On by default because every factory here uses the
    # v2 noise range (50 -> 0.01), where the unscaled parameterisation is the
    # documented failure case.  Set False for the v1 formulation, in which the
    # score magnitude is learned per level via conditional normalisation.
    scale_by_sigma: bool = True
    # NCSNv2 Technique 5: a final noiseless correction after the last Langevin
    # step.  Both released configs set denoise: true.
    denoise: bool = True

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        check_sample_space_channels(self.in_channels, self.out_channels, "the score")
        if self.base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {self.base_channels}")
        if not self.channel_mult:
            raise ValueError("channel_mult must have at least one stage")
        if any(m <= 0 for m in self.channel_mult):
            raise ValueError(f"channel_mult entries must be positive, got {self.channel_mult}")
        if self.num_res_blocks <= 0:
            raise ValueError(f"num_res_blocks must be positive, got {self.num_res_blocks}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.resnet_groups <= 0:
            raise ValueError(f"resnet_groups must be positive, got {self.resnet_groups}")
        if self.num_noise_levels <= 0:
            raise ValueError(f"num_noise_levels must be positive, got {self.num_noise_levels}")
        if not 0.0 < self.sigma_min < self.sigma_max:
            raise ValueError(
                f"require 0 < sigma_min < sigma_max, got "
                f"min={self.sigma_min}, max={self.sigma_max}"
            )
        if self.langevin_steps <= 0:
            raise ValueError(f"langevin_steps must be positive, got {self.langevin_steps}")
        if self.langevin_eps <= 0:
            raise ValueError(f"langevin_eps must be positive, got {self.langevin_eps}")

        # Spatial divisibility — same rule as DDPM (no downsample after last stage).
        if isinstance(self.sample_size, tuple):
            h, w = self.sample_size
        else:
            h = w = self.sample_size
        factor = 2 ** (len(self.channel_mult) - 1)
        if h % factor != 0 or w % factor != 0:
            raise ValueError(
                f"sample_size {self.sample_size} must be divisible by "
                f"2 ** (len(channel_mult) - 1) = {factor}"
            )

        for mult in self.channel_mult:
            stage_ch = self.base_channels * mult
            if stage_ch % self.resnet_groups != 0:
                raise ValueError(
                    f"Stage channels {stage_ch} (base_channels {self.base_channels}"
                    f" × mult {mult}) must be divisible by resnet_groups "
                    f"{self.resnet_groups}"
                )
