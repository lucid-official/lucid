"""Stable Diffusion configuration (Rombach et al., 2022)."""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    BetaSchedule,
    DiffusionModelConfig,
    check_sample_space_channels,
)

__all__ = ["StableDiffusionConfig"]


def _check_unet_widths(widths: tuple[int, ...], head_dim: int) -> None:
    """Refuse U-Net widths the attention heads or the timestep embedding split unevenly."""
    if widths[0] % 2:
        raise ValueError(
            f"the first U-Net width {widths[0]} is the timestep embedding's, "
            f"which splits into cosine and sine halves: it must be even"
        )
    if any(w % head_dim != 0 for w in widths):
        raise ValueError(
            f"attention_head_dim {head_dim} must divide every U-Net width {widths}"
        )


@model_family_meta(
    canonical_name="Stable Diffusion",
    citation=(
        'Rombach, Robin, et al. "High-Resolution Image Synthesis with '
        'Latent Diffusion Models." CVPR, 2022, pp. 10684–10695.'
    ),
    theory=r"""
    Diffusion in pixel space spends most of its capacity on detail the
    eye discards.  Latent diffusion splits the problem in two: an
    autoencoder learns a perceptually equivalent space at a fraction of
    the resolution, and the diffusion model runs entirely inside it.

    The first stage is an encoder :math:`\mathcal{E}` and decoder
    :math:`\mathcal{D}` with a downsampling factor
    :math:`f = H/h = W/w`.  At :math:`f = 8` a
    :math:`512 \times 512 \times 3` image becomes a
    :math:`64 \times 64 \times 4` latent — 48 times fewer values, so
    every subsequent convolution and attention costs proportionally
    less.  The paper regularises this space rather than leaving it free:
    a slight KL penalty toward :math:`\mathcal{N}(0, I)` keeps the
    latents from drifting to arbitrary scale, which matters because the
    diffusion process assumes unit-ish variance.

    The second stage is a time-conditional U-Net trained on the usual
    denoising objective, but on latents:

    .. math::

        \mathcal{L} = \mathbb{E}_{\mathcal{E}(x),\, y,\,
        \epsilon \sim \mathcal{N}(0, 1),\, t}
        \Big[\big\| \epsilon - \epsilon_\theta\big(z_t, t,
        \tau_\theta(y)\big) \big\|_2^2\Big].

    Conditioning enters by cross-attention rather than concatenation,
    which is what makes the modality of :math:`y` irrelevant to the
    U-Net.  With :math:`\varphi_i(z_t)` the flattened intermediate
    representation,

    .. math::

        Q = W_Q^{(i)} \varphi_i(z_t), \quad
        K = W_K^{(i)} \tau_\theta(y), \quad
        V = W_V^{(i)} \tau_\theta(y),

    so the latent asks the questions and the conditioning answers them.
    A text encoder, a class embedding and a layout map all present the
    same interface — a sequence of vectors — and the U-Net never learns
    which it is reading.

    :math:`\tau_\theta` is deliberately left open by the paper, which
    evaluates several.  The released models use a frozen CLIP text
    encoder, so the width below is CLIP's and the conditioning sequence
    is its 77 tokens.
    """,
)
@dataclass(frozen=True)
class StableDiffusionConfig(DiffusionModelConfig):
    r"""Frozen configuration dataclass for every Stable Diffusion variant.

    Parameters
    ----------
    sample_size : int, default=512
        Image resolution.  The latent side is ``sample_size //
        downsample_factor``.
    latent_channels : int, default=4
        Channels of the first stage's latent.  Four is what the released
        autoencoders use; it is a *choice*, not a consequence of the
        downsampling factor.
    downsample_factor : int, default=8
        The paper's :math:`f`.  Fixed by the autoencoder's depth — one
        stride-2 stage per power of two — so it is validated against
        ``vae_block_out_channels`` rather than trusted.
    vae_block_out_channels : tuple of int, default=(128, 256, 512, 512)
        Encoder widths, one per resolution.  ``len - 1`` stride-2 stages,
        hence :math:`f = 2^{\,\mathrm{len}-1}`.
    vae_layers_per_block : int, default=2
        Residual blocks per resolution in the autoencoder.
    unet_block_out_channels : tuple of int, default=(320, 640, 1280, 1280)
        U-Net widths, one per resolution.
    unet_layers_per_block : int, default=2
        Residual blocks per resolution in the U-Net.
    attention_head_dim : int, default=8
        Attention **heads**, despite the name.  The released
        configuration calls it ``attention_head_dim`` and the reference
        reads it as a count: at ``8`` a 320-wide stage gets eight heads
        of forty channels.  Reading it as a dimension gives forty heads
        of eight, which has exactly the same parameters and different
        activations.
    cross_attention_dim : int, default=768
        Width of the conditioning sequence, and therefore of
        :math:`\tau_\theta`'s output.  768 is CLIP ViT-L/14's text width;
        1024 is OpenCLIP ViT-H's.
    context_length : int, default=77
        Length of the conditioning sequence.
    num_train_timesteps : int, default=1000
        Diffusion steps the noise schedule spans.
    beta_start, beta_end : float
        Endpoints of the schedule.
    steps_offset : int, default=1
        Added to every sampled timestep.  The released scheduler visits
        901, 801, … rather than 999, 899, …, and the offset is what
        produces the shift.  A trajectory that omits it takes correct
        steps between the wrong times.
    set_alpha_to_one : bool, default=False
        Whether the step past the end uses
        :math:`\bar\alpha = 1` or :math:`\bar\alpha_0`.  The released
        configuration says false, so the final step bootstraps from
        ``alphas_cumprod[0]`` — very close to 1, and not 1.
    beta_schedule : str, default="scaled_linear"
        ``"scaled_linear"`` interpolates linearly in
        :math:`\sqrt{\beta}` — not in :math:`\beta`.  Getting this wrong
        produces a schedule that looks right on a plot and denoises to
        mush, which is why it is a named field rather than a constant.
    norm_num_groups : int, default=32
        Groups for every :class:`~lucid.nn.GroupNorm`.

    Notes
    -----
    The defaults are the released v1 configuration, read from the
    published ``unet``, ``vae`` and ``scheduler`` configs rather than
    from memory.

    ``cross_attention_dim`` and ``context_length`` are not free: they
    must match whatever produces the conditioning.  At the defaults they
    are exactly :class:`~lucid.models.CLIPModel` ViT-L/14's
    ``text_width`` and ``context_length``, which is what the released
    model uses.

    Examples
    --------
    >>> from lucid.models.generative.stable_diffusion import StableDiffusionConfig
    >>> config = StableDiffusionConfig()
    >>> config.latent_size
    64
    >>> config.downsample_factor
    8
    """

    model_type: ClassVar[str] = "stable_diffusion"

    sample_size: int = 512
    in_channels: int = 3
    out_channels: int = 3

    latent_channels: int = 4
    downsample_factor: int = 8
    vae_block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    vae_layers_per_block: int = 2

    unet_block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280)
    unet_layers_per_block: int = 2
    attention_head_dim: int = 8
    cross_attention_dim: int = 768
    context_length: int = 77

    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: BetaSchedule = "scaled_linear"
    steps_offset: int = 1
    set_alpha_to_one: bool = False

    norm_num_groups: int = 32

    @property
    def latent_size(self) -> int:
        """Side of the square latent the U-Net operates on.

        Returns
        -------
        int
            ``sample_size // downsample_factor``.
        """
        return self.sample_size // self.downsample_factor

    @override
    def __post_init__(self) -> None:
        """Reject configurations that cannot be built."""
        super().__post_init__()
        # The first stage is an autoencoder: its decoder reconstructs the
        # image its encoder read (Rombach et al. 2022, §3.1).
        check_sample_space_channels(
            self.in_channels, self.out_channels, "the first stage's reconstruction"
        )
        object.__setattr__(
            self, "vae_block_out_channels", tuple(self.vae_block_out_channels)
        )
        object.__setattr__(
            self, "unet_block_out_channels", tuple(self.unet_block_out_channels)
        )

        expected = 2 ** (len(self.vae_block_out_channels) - 1)
        if self.downsample_factor != expected:
            raise ValueError(
                f"downsample_factor {self.downsample_factor} does not match "
                f"the autoencoder's depth: {len(self.vae_block_out_channels)} "
                f"resolutions give {len(self.vae_block_out_channels) - 1} "
                f"stride-2 stages, so f must be {expected}"
            )
        if not isinstance(self.sample_size, int):
            raise TypeError(
                f"sample_size must be a square int for this family, got "
                f"{self.sample_size}"
            )
        if self.sample_size % self.downsample_factor != 0:
            raise ValueError(
                f"sample_size {self.sample_size} is not divisible by "
                f"downsample_factor {self.downsample_factor}"
            )
        latent = self.latent_size
        if latent % 2 ** (len(self.unet_block_out_channels) - 1) != 0:
            raise ValueError(
                f"the U-Net halves its input {len(self.unet_block_out_channels) - 1} "
                f"times, which a latent of {latent} cannot survive"
            )
        for name, widths in (
            ("vae_block_out_channels", self.vae_block_out_channels),
            ("unet_block_out_channels", self.unet_block_out_channels),
        ):
            if not widths:
                raise ValueError(f"{name} must not be empty")
            bad = [w for w in widths if w % self.norm_num_groups != 0]
            if bad:
                raise ValueError(
                    f"every width in {name} must be divisible by "
                    f"norm_num_groups={self.norm_num_groups}, got {bad}"
                )
        _check_unet_widths(self.unet_block_out_channels, self.attention_head_dim)
        if self.beta_schedule not in ("linear", "scaled_linear"):
            raise ValueError(
                f"beta_schedule must be 'linear' or 'scaled_linear', got "
                f"{self.beta_schedule!r}"
            )
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError(
                f"betas must satisfy 0 < start < end < 1, got "
                f"{self.beta_start} and {self.beta_end}"
            )
        if self.context_length < 1:
            raise ValueError(
                f"context_length must be positive, got {self.context_length}"
            )
