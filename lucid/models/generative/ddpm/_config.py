"""DDPM configuration — Ho et al., 2020 ("Denoising Diffusion Probabilistic Models").

The paper introduced the now-canonical U-Net + sinusoidal-timestep architecture
for image diffusion.  We expose every hyperparameter from Appendix B (channel
multipliers, ResBlock count, attention resolutions, dropout) plus a couple of
Improved-DDPM (Nichol & Dhariwal 2021) extras (``learn_sigma``,
``v_prediction``) — superset of the legacy port, intentionally.

Inherits noise-schedule knobs from :class:`DiffusionModelConfig`
(``num_train_timesteps`` / ``beta_start`` / ``beta_end`` / ``beta_schedule`` /
``prediction_type``).
"""

from dataclasses import dataclass, field
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    DiffusionModelConfig,
    check_noise_prediction_channels,
)


@model_family_meta(
    canonical_name="DDPM",
    citation=(
        'Ho, Jonathan, et al. "Denoising Diffusion Probabilistic Models." '
        "Advances in Neural Information Processing Systems, 2020, "
        "pp. 6840–6851."
    ),
    theory=r"""
    DDPM — *Denoising Diffusion Probabilistic Models* — defines a latent
    variable generative model through a pair of Markov chains.  The
    **forward (noising) chain** progressively corrupts data
    :math:`x_0 \sim q(x_0)` over :math:`T` steps with a fixed Gaussian
    schedule :math:`\{\beta_t\}_{t=1}^{T}`:

    .. math::

        q(x_t \mid x_{t-1})
            = \mathcal{N}\!\left(
                x_t;\; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t\, \mathbf{I}
              \right).

    With :math:`\alpha_t = 1 - \beta_t` and
    :math:`\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s`, the marginal admits a
    closed form that supports cheap one-shot sampling at any step:

    .. math::

        q(x_t \mid x_0)
            = \mathcal{N}\!\left(
                x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\;
                (1 - \bar{\alpha}_t)\, \mathbf{I}
              \right),
        \qquad
        x_t = \sqrt{\bar{\alpha}_t}\, x_0 +
              \sqrt{1 - \bar{\alpha}_t}\, \epsilon,\;
              \epsilon \sim \mathcal{N}(0, \mathbf{I}).

    The **reverse (denoising) chain** is parameterised by a U-Net
    :math:`\epsilon_\theta(x_t, t)` trained to predict the injected noise.
    Ho et al. show that the variational ELBO

    .. math::

        \mathcal{L}_{\text{ELBO}} = \mathbb{E}_q\!\left[
            \log \frac{q(x_{1:T} \mid x_0)}{p_\theta(x_{0:T})}
        \right]

    decomposes into per-timestep KL divergences which, after a
    reweighting, simplify to the deceptively elegant noise-prediction
    objective

    .. math::

        \mathcal{L}_{\text{simple}}(\theta) =
            \mathbb{E}_{t, x_0, \epsilon}\!\left[
                \big\lVert \epsilon -
                    \epsilon_\theta(x_t, t)
                \big\rVert^2
            \right].

    Sampling iterates :math:`t = T, T-1, \dots, 1` via
    :math:`x_{t-1} = \tfrac{1}{\sqrt{\alpha_t}}
    \big(x_t - \tfrac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,
    \epsilon_\theta(x_t, t)\big) + \sigma_t z`.  The U-Net itself follows
    the original architecture (Ronneberger 2015) augmented with
    sinusoidal timestep embeddings, self-attention at low-resolution
    feature maps (default :math:`16 \times 16`), and GroupNorm-equipped
    residual blocks — the now-canonical recipe inherited by ADM, Imagen,
    Stable Diffusion, and beyond.
    """,
)
@dataclass(frozen=True)
class DDPMConfig(DiffusionModelConfig):
    """Configuration for every DDPM variant.

    Args:
        base_channels: Channel width of the first U-Net stage ``C_0``.  All
            other stages are integer multiples (see ``channel_mult``).
            Paper default: 128.
        channel_mult: Per-stage channel multiplier.  ``(1, 2, 2, 2)`` for
            CIFAR (Table 9), ``(1, 1, 2, 2, 4, 4)`` for LSUN/CelebA-HQ.
            Length = number of downsampling stages; spatial resolution
            divides by ``2 ** (L - 1)`` (no downsample after the final
            stage).
        num_res_blocks: ResBlocks per stage (paper: 2).
        attention_resolutions: Spatial sizes at which self-attention is
            applied.  Ho 2020 uses ``(16,)`` — attention only at the
            16×16 feature scale.
        num_heads: Attention head count (paper: 1 for CIFAR, can scale).
        dropout: Dropout inside ResBlocks.  Paper: 0.1 (CIFAR), 0.0
            (CelebA / LSUN).
        resnet_groups: ``GroupNorm`` group count inside ResBlocks (paper
            default: 32).  Must divide every per-stage channel count.
        learn_sigma: Improved-DDPM (Nichol 2021) — when ``True``, the
            network predicts the variance in addition to the mean.  Output
            channels become ``2 * in_channels`` — ``out_channels`` must say
            so — and training switches to the hybrid objective (see
            ``vlb_weight``), since the fixed simple loss carries no signal
            for a variance head.
        vlb_weight: The λ on the variational term of
            ``L_simple + λ·L_vlb`` (Nichol & Dhariwal 2021 §4; default
            ``1e-3``).  Only read when ``learn_sigma=True``.
        clip_denoised: Clip the predicted ``x_0`` to ``[-1, 1]`` during
            sampling (default True).

    Examples
    --------
    >>> from lucid.models.generative.ddpm._config import DDPMConfig
    >>> cfg = DDPMConfig()
    >>> cfg.model_type
    'ddpm'
    >>> cfg.in_channels
    3
    """

    model_type: ClassVar[str] = "ddpm"

    # U-Net architectural knobs.
    base_channels: int = 128
    channel_mult: tuple[int, ...] = field(default_factory=lambda: (1, 2, 2, 2))
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = field(default_factory=lambda: (16,))
    num_heads: int = 1
    dropout: float = 0.1
    resnet_groups: int = 32

    # Improved DDPM / Imagen-style extras.
    # Improved-DDPM's ResBlock conditions with FiLM rather than addition:
    # the time embedding projects to 2*out_channels and the block computes
    # ``norm(h) * (1 + scale) + shift``.  Ho 2020's original CIFAR/LSUN
    # models use the additive form, so this defaults off and the
    # Improved-DDPM-derived factories turn it on.
    use_scale_shift_norm: bool = False
    # ``rescale_timesteps`` feeds the network ``t * (1000 / T)`` instead of
    # the raw index, so a T=4000 model still sees magnitudes in [0, 1000)
    # — the range the sinusoidal embedding was designed for.
    rescale_timesteps: bool = False
    learn_sigma: bool = False
    # Weight λ on the variational term of Improved-DDPM's hybrid objective
    # ``L_simple + λ·L_vlb`` (Nichol & Dhariwal 2021 §4).  Only consulted when
    # ``learn_sigma=True`` — with a fixed variance there is nothing for the
    # bound to train.
    vlb_weight: float = 1e-3

    # Sampling.
    clip_denoised: bool = True

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        # Refused rather than derived: a config that names a width must be
        # the network it builds, and the value it was given is the one a
        # caller reads back (LCD-631).  DiT holds the same rule.
        check_noise_prediction_channels(
            self.in_channels, self.out_channels, self.learn_sigma
        )
        if self.base_channels <= 0:
            raise ValueError(
                f"base_channels must be positive, got {self.base_channels}"
            )
        if not self.channel_mult:
            raise ValueError("channel_mult must have at least one stage")
        if any(m <= 0 for m in self.channel_mult):
            raise ValueError(
                f"channel_mult entries must be positive, got {self.channel_mult}"
            )
        if self.num_res_blocks <= 0:
            raise ValueError(
                f"num_res_blocks must be positive, got {self.num_res_blocks}"
            )
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.resnet_groups <= 0:
            raise ValueError(
                f"resnet_groups must be positive, got {self.resnet_groups}"
            )
        if self.vlb_weight < 0.0:
            raise ValueError(f"vlb_weight must be non-negative, got {self.vlb_weight}")

        # Spatial divisibility: ``2 ** (L - 1)`` because the last stage has
        # no downsample.
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

        # Every per-stage channel count must be divisible by ``resnet_groups``
        # so GroupNorm doesn't error.
        for mult in self.channel_mult:
            stage_ch = self.base_channels * mult
            if stage_ch % self.resnet_groups != 0:
                raise ValueError(
                    f"Stage channels {stage_ch} (base_channels {self.base_channels}"
                    f" × mult {mult}) must be divisible by resnet_groups "
                    f"{self.resnet_groups}"
                )

    @property
    def out_channels_effective(self) -> int:
        """How many channels the U-Net actually emits (doubled when
        ``learn_sigma=True``)."""
        return 2 * self.in_channels if self.learn_sigma else self.in_channels
