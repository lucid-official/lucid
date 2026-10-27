r"""Stable Diffusion — the two stages, assembled.

The pieces are in their own modules because each is a model in its own
right: :class:`AutoencoderKL` learns the latent space,
:class:`UNet2DConditionModel` denoises inside it, and
:class:`PNDMScheduler` decides which timesteps to visit.  What this
file adds is the wiring, and the wiring has two decisions in it that
no shape check can see.

**The latent is rescaled.**  The first stage's latents have a standard
deviation of roughly 5, and the diffusion process assumes something near
1.  So the encoder's output is multiplied by a scaling factor before the
U-Net ever sees it, and divided back before decoding.  Omit it and the
forward process's noise is negligible against the signal; the model
trains, and samples nothing.

**Classifier-free guidance needs two predictions.**  One on the prompt
and one on an empty conditioning, then extrapolates away from the
empty one:

.. math::

    \tilde\epsilon = \epsilon_\varnothing
        + s\,(\epsilon_{\text{cond}} - \epsilon_\varnothing).

At :math:`s = 1` this is exactly the conditional prediction, which is
the identity worth testing — a guidance implementation that ignores its
scale still produces images.

Neither prediction depends on the other, so :meth:`generate` stacks the
two conditionings into one batch instead of calling the U-Net twice.
The arithmetic is identical; what it saves is the per-step cost of
driving the model a second time, which on an M4 Max is about 1.2× over
fifty steps.  The saving is *not* idle compute being filled — a batch of
two costs very nearly twice a batch of one at 512 pixels.
"""

from dataclasses import dataclass
from collections.abc import Callable
from typing import ClassVar, Self, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageGenerationModel
from lucid.models._output import GenerationOutput, ModelOutput
from lucid.models._utils._generative import model_dtype
from lucid.models.generative.stable_diffusion._autoencoder import AutoencoderKL
from lucid.models.generative.stable_diffusion._config import StableDiffusionConfig
from lucid.models.generative.stable_diffusion._scheduler import DDIMScheduler, PNDMScheduler
from lucid.models.generative.stable_diffusion._unet import UNet2DConditionModel

__all__ = ["StableDiffusionModel", "StableDiffusionForImageGeneration", "StableDiffusionOutput"]

# The released first stage's scaling constant.  Not a tuned value: it is
# 1/std of the latents the published autoencoder produces.
_LATENT_SCALE = 0.18215


@dataclass(slots=True)
class StableDiffusionOutput(ModelOutput):
    """What :class:`StableDiffusionModel` returns.

    Attributes
    ----------
    noise_pred : Tensor
        ``(B, latent_channels, h, w)`` — :math:`\\epsilon_\\theta`.
    latent : Tensor
        The noised latent the prediction was made from.
    loss : Tensor or None
        The denoising objective, present only when a target was given.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.stable_diffusion._model import (
    ...     StableDiffusionOutput,
    ... )
    >>> out = StableDiffusionOutput(
    ...     noise_pred=lucid.zeros(1, 4, 32, 32),
    ...     latent=lucid.zeros(1, 4, 32, 32),
    ... )
    >>> out.noise_pred.shape
    (1, 4, 32, 32)
    >>> out.loss is None
    True

    Both live in latent space, not pixels — decoding is a separate step,
    and doing it per training step would cost more than the training.
    """

    noise_pred: Tensor
    latent: Tensor
    loss: Tensor | None = None


class StableDiffusionModel(PretrainedModel):
    r"""Latent diffusion: an autoencoder, a conditional U-Net, a sampler.

    Parameters
    ----------
    config : StableDiffusionConfig
        The variant to build.

    Attributes
    ----------
    vae : AutoencoderKL
        First stage.
    unet : UNet2DConditionModel
        Second stage.
    scheduler : DDIMScheduler
        The paper's sampler, and what :meth:`forward` uses to noise a
        latent.  Sampling defaults to PNDM instead — see
        :meth:`StableDiffusionForImageGeneration.generate`.

    Notes
    -----
    Reference: Rombach et al., *"High-Resolution Image Synthesis with
    Latent Diffusion Models"*, CVPR, 2022
    (`arXiv:2112.10752 <https://arxiv.org/abs/2112.10752>`_).

    The text encoder is **not** held here.  The paper defines
    :math:`\tau_\theta` as a domain-specific encoder and evaluates
    several, and the released models freeze a CLIP text tower rather
    than training one — so conditioning arrives as an already-encoded
    ``(B, L, cross_attention_dim)`` sequence and the caller chooses what
    produced it.  At the default width that is
    :func:`~lucid.models.clip_vit_large_14`'s text tower.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.stable_diffusion import (
    ...     StableDiffusionConfig, StableDiffusionModel)
    >>> config = StableDiffusionConfig(sample_size=32, downsample_factor=4,
    ...                                vae_block_out_channels=(32, 64, 64),
    ...                                unet_block_out_channels=(32, 64),
    ...                                attention_head_dim=32,
    ...                                cross_attention_dim=16, context_length=4)
    >>> model = StableDiffusionModel(config).eval()
    >>> out = model(lucid.randn((1, 3, 32, 32)), lucid.randn((1, 4, 16)))
    >>> out.noise_pred.shape
    (1, 4, 8, 8)
    """

    config_class: ClassVar[type[StableDiffusionConfig]] = StableDiffusionConfig
    base_model_prefix = "stable_diffusion"

    def __init__(self, config: StableDiffusionConfig) -> None:
        """Initialise the model. See the class docstring for parameters."""
        super().__init__(config)
        self.config: StableDiffusionConfig = config
        self.vae = AutoencoderKL(config)
        self.unet = UNet2DConditionModel(config)
        self.scheduler = DDIMScheduler(config)

    @override
    def _apply(self, fn: Callable[[Tensor], Tensor]) -> Self:
        super()._apply(fn)
        # The scheduler is not a Module, but its noising tables are read
        # by every training forward, so they move and cast with the model.
        self.scheduler._apply(fn)
        return self

    def encode_image(self, images: Tensor, sample: bool = True) -> Tensor:
        """Image to scaled latent.

        Parameters
        ----------
        images : Tensor
            ``(B, in_channels, H, W)``.
        sample : bool, default=True
            Draw from the posterior, or take its mode.

        Returns
        -------
        Tensor
            ``(B, latent_channels, H/f, W/f)``, already scaled for the
            diffusion process.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.stable_diffusion import (
        ...     StableDiffusionConfig, StableDiffusionModel)
        >>> config = StableDiffusionConfig(sample_size=32, downsample_factor=4,
        ...                                vae_block_out_channels=(32, 64, 64),
        ...                                unet_block_out_channels=(32, 64),
        ...                                attention_head_dim=32,
        ...                                cross_attention_dim=16, context_length=4)
        >>> model = StableDiffusionModel(config).eval()
        >>> images = lucid.randn((1, 3, 32, 32))
        >>> latent = model.encode_image(images, sample=False)
        >>> latent.shape
        (1, 4, 8, 8)

        What this adds over the autoencoder is the scaling constant: the
        latent is the posterior's mode times 0.18215, the released first
        stage's inverse standard deviation.

        >>> raw = model.vae.encode(images).mode()
        >>> bool(lucid.allclose(latent, raw * 0.18215))
        True
        """
        posterior = self.vae.encode(images)
        latent = posterior.sample() if sample else posterior.mode()
        return latent * _LATENT_SCALE

    def decode_latent(self, latent: Tensor) -> Tensor:
        """Scaled latent back to an image.

        Parameters
        ----------
        latent : Tensor
            ``(B, latent_channels, h, w)``.

        Returns
        -------
        Tensor
            ``(B, out_channels, h*f, w*f)``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.stable_diffusion import (
        ...     StableDiffusionConfig, StableDiffusionModel)
        >>> config = StableDiffusionConfig(sample_size=32, downsample_factor=4,
        ...                                vae_block_out_channels=(32, 64, 64),
        ...                                unet_block_out_channels=(32, 64),
        ...                                attention_head_dim=32,
        ...                                cross_attention_dim=16, context_length=4)
        >>> model = StableDiffusionModel(config).eval()
        >>> images = lucid.randn((1, 3, 32, 32))
        >>> latent = model.encode_image(images, sample=False)
        >>> model.decode_latent(latent).shape
        (1, 3, 32, 32)

        The scale is divided back out before decoding, so this undoes the
        scaling :meth:`encode_image` applied: it matches decoding the
        unscaled mode directly.

        >>> raw = model.vae.encode(images).mode()
        >>> bool(lucid.allclose(model.decode_latent(latent),
        ...                     model.vae.decode(raw), atol=1e-5))
        True
        """
        return self.vae.decode(latent / _LATENT_SCALE)

    @override
    def forward(
        self,
        images: Tensor,
        context: Tensor,
        timestep: Tensor | None = None,
        return_loss: bool = False,
    ) -> StableDiffusionOutput:
        r"""One training step's worth of the denoising objective.

        Parameters
        ----------
        images : Tensor
            ``(B, in_channels, H, W)``.
        context : Tensor
            ``(B, L, cross_attention_dim)`` — :math:`\tau_\theta(y)`.
        timestep : Tensor or None, optional
            ``(B,)``.  Drawn uniformly when omitted, which is what
            training does.
        return_loss : bool, default=False
            Also compute :math:`\|\epsilon - \epsilon_\theta\|_2^2`.

        Returns
        -------
        StableDiffusionOutput
            Prediction, the noised latent, and optionally the loss.
        """
        latent = self.encode_image(images)
        shape = tuple(int(s) for s in latent.shape)
        noise = lucid.randn(shape, device=latent.device.type, dtype=latent.dtype)

        if timestep is None:
            drawn = lucid.rand((shape[0],), device=latent.device.type)
            timestep = (drawn * self.config.num_train_timesteps).to(lucid.int64)

        noised = self.scheduler.add_noise(latent, noise, timestep)
        prediction = self.unet(noised, timestep.to(latent.dtype), context)

        loss: Tensor | None = None
        if return_loss:
            loss = ((prediction - noise) ** 2).mean()
        return StableDiffusionOutput(noise_pred=prediction, latent=noised, loss=loss)


class StableDiffusionForImageGeneration(ImageGenerationModel):
    """Stable Diffusion posed as a diffusion model: the objective and a sampler.

    Parameters
    ----------
    config : StableDiffusionConfig
        The variant to build.

    Attributes
    ----------
    stable_diffusion : StableDiffusionModel
        The autoencoder, U-Net and noising schedule.

    Notes
    -----
    Reference: Rombach et al., CVPR 2022 (arXiv:2112.10752), §4.3.

    :meth:`forward` is one training step of the denoising objective, as
    in every other image-generation family; :meth:`generate` runs the
    reverse process and decodes. Conditioning arrives already encoded,
    and for classifier-free guidance the caller supplies the
    unconditional sequence too — usually the text encoder's output for
    the empty string, which is not the same thing as zeros.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.stable_diffusion import (
    ...     StableDiffusionConfig, StableDiffusionForImageGeneration)
    >>> config = StableDiffusionConfig(sample_size=32, downsample_factor=4,
    ...                                vae_block_out_channels=(32, 64, 64),
    ...                                unet_block_out_channels=(32, 64),
    ...                                attention_head_dim=32,
    ...                                cross_attention_dim=16, context_length=4)
    >>> model = StableDiffusionForImageGeneration(config).eval()
    >>> context = lucid.randn((1, 4, 16))
    >>> model.generate(context, num_inference_steps=2).samples.shape
    (1, 3, 32, 32)

    Calling the model trains it — the loss is always reported:

    >>> out = model(lucid.randn((1, 3, 32, 32)), context)
    >>> out.loss.shape, out.noise_pred.shape
    ((), (1, 4, 8, 8))
    """

    config_class: ClassVar[type[StableDiffusionConfig]] = StableDiffusionConfig
    base_model_prefix = "stable_diffusion"

    def __init__(self, config: StableDiffusionConfig) -> None:
        """Initialise the model. See the class docstring for parameters."""
        super().__init__(config)
        self.stable_diffusion = StableDiffusionModel(config)

    @lucid.no_grad()
    def generate(
        self,
        context: Tensor,
        uncond_context: Tensor | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        latent: Tensor | None = None,
        sampler: str = "pndm",
    ) -> GenerationOutput:
        r"""Sample an image from the conditioning.

        Parameters
        ----------
        context : Tensor
            ``(B, L, cross_attention_dim)``.
        uncond_context : Tensor or None, optional
            The unconditional sequence. Guidance is skipped when absent,
            which is equivalent to ``guidance_scale = 1``.
        num_inference_steps : int, default=50
            Network evaluations along the reverse trajectory.
        guidance_scale : float, default=7.5
            :math:`s` in
            :math:`\epsilon_\varnothing + s(\epsilon_c - \epsilon_\varnothing)`.
        eta : float, default=0.0
            DDIM at 0, DDPM at 1.  Ignored by PNDM, which is
            deterministic by construction.
        latent : Tensor or None, optional
            Starting noise. Drawn when absent.
        sampler : str, default="pndm"
            ``"pndm"`` reproduces the released pipeline — its
            ``model_index.json`` names ``PNDMScheduler``.  ``"ddim"`` is
            the paper's, and the one ``eta`` applies to.

        Returns
        -------
        GenerationOutput
            ``samples`` of shape ``(B, out_channels, sample_size,
            sample_size)`` — decoded pixels, not the latent.

        Raises
        ------
        ValueError
            If ``guidance_scale`` is negative, or the two conditioning
            sequences disagree in shape.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.stable_diffusion import (
        ...     StableDiffusionConfig, StableDiffusionForImageGeneration)
        >>> config = StableDiffusionConfig(sample_size=32, downsample_factor=4,
        ...                                vae_block_out_channels=(32, 64, 64),
        ...                                unet_block_out_channels=(32, 64),
        ...                                attention_head_dim=32,
        ...                                cross_attention_dim=16, context_length=4)
        >>> model = StableDiffusionForImageGeneration(config).eval()
        >>> context = lucid.randn((1, 4, 16))
        >>> start = lucid.randn((1, 4, 8, 8))
        >>> image = model.generate(
        ...     context, num_inference_steps=2, latent=start).samples
        >>> image.shape
        (1, 3, 32, 32)

        PNDM draws no noise of its own, so the starting latent is the whole
        seed: pass the same one and the same image comes back.

        >>> again = model.generate(
        ...     context, num_inference_steps=2, latent=start).samples
        >>> bool((again == image).all())
        True

        At ``guidance_scale=1`` the extrapolation is the conditional
        prediction itself, so an unconditional sequence changes nothing; at
        the default 7.5 it does.

        >>> uncond = lucid.randn((1, 4, 16))  # stands in for the empty caption
        >>> unguided = model.generate(context, uncond, num_inference_steps=2,
        ...                           guidance_scale=1.0, latent=start).samples
        >>> bool((unguided == image).all())
        True
        >>> guided = model.generate(context, uncond, num_inference_steps=2,
        ...                         latent=start).samples
        >>> bool((guided == image).all())
        False
        """
        if guidance_scale < 0.0:
            raise ValueError(f"guidance_scale must be non-negative, got {guidance_scale}")
        if sampler not in ("pndm", "ddim"):
            raise ValueError(f"sampler must be 'pndm' or 'ddim', got {sampler!r}")

        inner = self.stable_diffusion
        config = inner.config
        batch = int(context.shape[0])
        side = config.latent_size

        if uncond_context is not None and tuple(uncond_context.shape) != tuple(context.shape):
            raise ValueError(
                f"uncond_context {tuple(uncond_context.shape)} must match "
                f"context {tuple(context.shape)}"
            )

        if latent is None:
            shape = (batch, config.latent_channels, side, side)  # in the model dtype
            latent = lucid.randn(shape, dtype=model_dtype(inner), device=context.device)

        pndm = PNDMScheduler(config) if sampler == "pndm" else None
        scheduler = pndm if pndm is not None else inner.scheduler
        steps = scheduler.timesteps(num_inference_steps)

        both_context: Tensor | None = None
        if uncond_context is not None and guidance_scale != 1.0:
            both_context = lucid.cat([uncond_context, context], dim=0)
        width = batch if both_context is None else 2 * batch

        for index, step in enumerate(steps):
            timestep = lucid.full(
                (width,), float(step), device=latent.device.type, dtype=latent.dtype
            )
            if both_context is None:
                prediction = inner.unet(latent, timestep, context)
            else:
                pair = lucid.cat([latent, latent], dim=0)
                both = inner.unet(pair, timestep, both_context)
                unconditional, conditional = both[:batch], both[batch:]
                prediction = unconditional + guidance_scale * (conditional - unconditional)
            if pndm is not None:
                latent = pndm.step(prediction, step, latent)
            else:
                previous = steps[index + 1] if index + 1 < len(steps) else -1
                latent = inner.scheduler.step(prediction, step, previous, latent, eta)

        return GenerationOutput(samples=inner.decode_latent(latent))

    @override
    def forward(
        self, images: Tensor, context: Tensor, timestep: Tensor | None = None
    ) -> StableDiffusionOutput:
        r"""One training step of the denoising objective.

        Parameters
        ----------
        images : Tensor
            ``(B, in_channels, H, W)`` — pixels; the frozen first stage
            encodes them.
        context : Tensor
            ``(B, L, cross_attention_dim)`` — :math:`\tau_\theta(y)`.
        timestep : Tensor or None, optional
            ``(B,)``.  Drawn uniformly when omitted, which is what
            training does.

        Returns
        -------
        StableDiffusionOutput
            The prediction, the noised latent, and the loss
            :math:`\|\epsilon - \epsilon_\theta\|_2^2` — always present,
            since a task model is called to be trained.
        """
        return self.stable_diffusion(images, context, timestep, return_loss=True)
