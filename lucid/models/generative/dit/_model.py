r"""DiT — a diffusion backbone with no convolutions in it.

Patchify the latent, add a frozen sine-cosine table, run transformer
blocks conditioned on the timestep and class, decode each token back to
its patch.  The blocks themselves live at the domain layer
(:mod:`lucid.models.generative._common._transformers`) because MeanFlow's network
is the same one.

All four of the paper's conditioning designs are here.  Only adaLN-Zero
is used past Section 5 — it wins at every training budget and costs the
least — but the comparison *is* the paper's architectural finding, and a
family that could only express the winner could not reproduce the result
that made it the winner.
"""

from dataclasses import dataclass
from typing import ClassVar, cast, override

import lucid
import lucid.nn as nn
from lucid._dtype import _float32_or_wider
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import (
    DiffusionModelOutput,
    GenerationOutput,
    ModelOutput,
)
from lucid.models._tasks import ImageGenerationModel
from lucid.models._utils._generative import (
    make_beta_schedule,
    resolve_generation_device,
    start_noise,
)
from lucid.models.generative._common._transformers import (
    DiTBlock,
    DiTFinalLayer,
    sincos_position_embedding,
    timestep_embedding,
    unpatchify,
)
from lucid.models.generative.dit._config import DiTConfig

__all__ = [
    "DiTModel",
    "DiTForImageGeneration",
    "DiTOutput",
]


def _beta_schedule(config: DiTConfig) -> Tensor:
    """The config's ``β`` schedule, on the CPU in the default dtype."""
    return make_beta_schedule(
        config.num_train_timesteps,
        config.beta_schedule,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )


def _reverse_step(
    alpha_bar: Tensor, alpha_bar_prev: Tensor, eta: float, dtype: lucid.dtype
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Song et al.'s generalised reverse step, as coefficients in ``dtype``.

    Recover :math:`x_0`, then re-noise it to the previous timestep:
    ``(signal, noise_scale, keep, direction, sigma)`` with
    ``x0 = (z - noise_scale * eps) / signal`` and
    ``z' = keep * x0 + direction * eps + sigma * noise``.  ``eta`` scales
    how much of that re-noising is random — 0 is DDIM's deterministic
    step, 1 recovers DDPM's, which is what the paper samples with.

    Worked out in the dtype of the :math:`\bar\alpha` given, and each
    coefficient cast to ``dtype`` once: :math:`1 - \bar\alpha` is the
    subtraction half precision cannot do near :math:`\bar\alpha = 1`.
    """
    sigma = eta * (
        ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar)).sqrt()
        * (1.0 - alpha_bar / alpha_bar_prev).sqrt()
    )
    direction = (1.0 - alpha_bar_prev - sigma**2).clip(min=0.0).sqrt()
    return (
        alpha_bar.sqrt().to(dtype),
        (1.0 - alpha_bar).sqrt().to(dtype),
        alpha_bar_prev.sqrt().to(dtype),
        direction.to(dtype),
        sigma.to(dtype),
    )


class _CrossAttentionBlock(nn.Module):
    """A DiT block with a cross-attention layer for the conditioning.

    Parameters
    ----------
    hidden_size : int
        Residual stream width.
    num_heads : int
        Attention heads.
    mlp_ratio : float
        Feed-forward expansion.

    Notes
    -----
    The conditioning arrives as a length-two sequence — the timestep and
    the class — attended to by an extra layer after self-attention, as in
    the original encoder-decoder transformer.  Costs about 15% more
    Gflops than adaLN-Zero and scores worse, which is the point of
    keeping it.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        """Initialise the block. See the class docstring for parameters."""
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)
        inner = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner), nn.GELU(), nn.Linear(inner, hidden_size)
        )

    @override
    def forward(self, x: Tensor, cond_seq: Tensor) -> Tensor:
        """Self-attention, cross-attention onto ``cond_seq``, then the MLP."""
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        x = x + attended

        h = self.norm2(x)
        crossed, _ = self.cross(h, cond_seq, cond_seq, need_weights=False)
        x = x + crossed

        return x + self.mlp(self.norm3(x))


class _PlainBlock(nn.Module):
    """A standard ViT block, used by in-context conditioning.

    Parameters
    ----------
    hidden_size : int
        Residual stream width.
    num_heads : int
        Attention heads.
    mlp_ratio : float
        Feed-forward expansion.

    Notes
    -----
    In-context conditioning appends the timestep and class embeddings as
    two extra tokens, so the block needs no modification at all — which
    is exactly what the paper says about it, and why it adds negligible
    Gflops.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        """Initialise the block. See the class docstring for parameters."""
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        inner = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner), nn.GELU(), nn.Linear(inner, hidden_size)
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Attention then MLP, both on the residual stream."""
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


@dataclass(slots=True)
class DiTOutput(ModelOutput):
    r"""What a DiT training step reports.

    Attributes
    ----------
    loss : Tensor
        Mean squared error between the predicted and the true noise.
    noise_pred : Tensor
        ``(B, in_channels, H, W)`` — the predicted noise.
    variance_pred : Tensor or None
        The diagonal covariance the decoder emits beside it, present
        only when ``learn_sigma``.  Reported rather than dropped because
        it is half of what the network computed, and a caller training
        the full ADM objective needs it.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dit._model import DiTOutput
    >>> out = DiTOutput(
    ...     loss=lucid.zeros(()), noise_pred=lucid.zeros(1, 4, 32, 32)
    ... )
    >>> out.noise_pred.shape
    (1, 4, 32, 32)

    ``variance_pred`` stays empty unless the model learns the variance
    as well as the mean, which is optional and doubles the head.

    >>> out.variance_pred is None
    True
    """

    loss: Tensor
    noise_pred: Tensor
    variance_pred: Tensor | None = None


class DiTModel(PretrainedModel):
    r"""The denoising network: latent patches in, noise prediction out.

    Parameters
    ----------
    config : DiTConfig
        The variant to build.

    Attributes
    ----------
    patch_embed : lucid.nn.Conv2d
        Patchifying projection — stride equals kernel, so patches do not
        overlap.
    pos_embed : Tensor
        Frozen sine-cosine table, registered as a buffer.
    blocks : lucid.nn.ModuleList
        The transformer blocks, of whichever kind the conditioning names.
    final : DiTFinalLayer
        Modulated norm and the projection back to patch space.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748).  Backbone shapes are
    Table 1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dit import DiTConfig, DiTModel
    >>> config = DiTConfig(sample_size=8, patch_size=2, hidden_size=32,
    ...                    depth=2, num_heads=4, num_classes=10)
    >>> model = DiTModel(config).eval()
    >>> latent = lucid.randn((1, 4, 8, 8))
    >>> labels = lucid.tensor([3], dtype=lucid.int64)
    >>> model(latent, lucid.tensor([10.0]), labels).sample.shape
    (1, 8, 8, 8)
    """

    config_class: ClassVar[type[DiTConfig]] = DiTConfig
    base_model_prefix = "dit"

    def __init__(self, config: DiTConfig) -> None:
        """Initialise the network. See the class docstring for parameters."""
        super().__init__(config)
        self.config: DiTConfig = config
        side = (
            config.sample_size
            if isinstance(config.sample_size, int)
            else config.sample_size[0]
        )
        self.grid = side // config.patch_size

        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.register_buffer(
            "pos_embed",
            sincos_position_embedding(config.hidden_size, self.grid),
            persistent=False,
        )

        # ADM's way of admitting a timestep: sinusoid, then a two-layer
        # MLP.  The class label is a plain embedding with one extra row
        # for the null token guidance drops to.
        self.time_mlp = nn.Sequential(
            nn.Linear(config.frequency_embedding_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.label_embed = nn.Embedding(config.num_classes + 1, config.hidden_size)

        mode = config.conditioning
        self.blocks: nn.ModuleList[DiTBlock | _CrossAttentionBlock | _PlainBlock]
        if mode in ("adaln_zero", "adaln"):
            self.blocks = nn.ModuleList(
                [
                    DiTBlock(
                        config.hidden_size,
                        config.num_heads,
                        config.mlp_ratio,
                        gated=mode == "adaln_zero",
                    )
                    for _ in range(config.depth)
                ]
            )
        elif mode == "cross_attention":
            self.blocks = nn.ModuleList(
                [
                    _CrossAttentionBlock(
                        config.hidden_size, config.num_heads, config.mlp_ratio
                    )
                    for _ in range(config.depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    _PlainBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
                    for _ in range(config.depth)
                ]
            )

        self.final = DiTFinalLayer(
            config.hidden_size, config.patch_size, config.out_channels
        )
        if mode == "adaln_zero":
            for block in self.blocks:
                cast(DiTBlock, block).zero_conditioning()
        self.final.zero_conditioning()

    def _embeddings(
        self, timesteps: Tensor, labels: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """The timestep and label vectors, kept apart.

        Three of the four conditioning designs want their sum; in-context
        wants them as two separate tokens, so they are returned unmixed
        and the caller combines.
        """
        emb = timestep_embedding(
            timesteps,
            self.config.frequency_embedding_size,
            dtype=self.patch_embed.weight.dtype,
        )
        time = self.time_mlp(emb)
        if labels is None:
            labels = lucid.full(
                (int(timesteps.shape[0]),),
                float(self.config.num_classes),
                dtype=lucid.int64,
                device=timesteps.device.type,
            )
        return time, self.label_embed(labels)

    @override
    def forward(
        self,
        latent: Tensor,
        timesteps: Tensor,
        labels: Tensor | None = None,
    ) -> DiffusionModelOutput:
        r"""Predict the noise (and covariance) in a noised latent.

        Parameters
        ----------
        latent : Tensor
            ``(B, in_channels, H, W)`` — the noised latent :math:`z_t`.
        timesteps : Tensor
            ``(B,)`` diffusion timesteps.
        labels : Tensor or None, optional
            ``(B,)`` class indices.  ``None`` uses the null token.

        Returns
        -------
        DiffusionModelOutput
            ``sample`` of shape ``(B, out_channels, H, W)`` — noise, and
            the covariance beside it when ``learn_sigma``.  ``loss`` stays
            ``None``: the objective lives in
            :class:`DiTForImageGeneration`.
        """
        tokens = self.patch_embed(latent)
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1).permute(0, 2, 1)
        tokens = tokens + self.pos_embed

        time_emb, label_emb = self._embeddings(timesteps, labels)
        cond = time_emb + label_emb
        mode = self.config.conditioning

        if mode == "in_context":
            # The conditioning rides as two extra tokens and is stripped
            # again after the last block, so the blocks stay untouched.
            extra = lucid.stack([time_emb, label_emb], dim=1)
            tokens = lucid.cat([tokens, extra], dim=1)
            for block in self.blocks:
                tokens = cast(_PlainBlock, block)(tokens)
            tokens = tokens[:, : self.grid * self.grid]
        elif mode == "cross_attention":
            cond_seq = cond.reshape(cond.shape[0], 1, cond.shape[1])
            cond_seq = lucid.cat([cond_seq, cond_seq], dim=1)
            for block in self.blocks:
                tokens = cast(_CrossAttentionBlock, block)(tokens, cond_seq)
        else:
            for block in self.blocks:
                tokens = cast(DiTBlock, block)(tokens, cond)

        image = unpatchify(
            self.final(tokens, cond),
            self.grid,
            self.config.patch_size,
            self.config.out_channels,
        )
        return DiffusionModelOutput(sample=image)


class DiTForImageGeneration(ImageGenerationModel):
    r"""DiT posed as a diffusion model: the objective and a DDPM sampler.

    Parameters
    ----------
    config : DiTConfig
        The variant to build.

    Attributes
    ----------
    dit : DiTModel
        The denoising network.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023.  The diffusion
    hyperparameters are ADM's, which the paper retains wholesale: a
    linear variance schedule over 1000 steps from 1e-4 to 2e-2.

    The training objective here is the simple one — mean squared error on
    the noise.  The covariance the decoder emits is returned rather than
    trained; ADM's full objective adds a variational term for it, which
    the paper inherits but does not modify and which a caller can build
    from :attr:`DiTOutput.variance_pred`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dit import (
    ...     DiTConfig, DiTForImageGeneration)
    >>> config = DiTConfig(sample_size=8, patch_size=2, hidden_size=32,
    ...                    depth=2, num_heads=4, num_classes=10)
    >>> model = DiTForImageGeneration(config).eval()
    >>> model.generate(2, steps=3).samples.shape
    (2, 4, 8, 8)
    """

    config_class: ClassVar[type[DiTConfig]] = DiTConfig
    base_model_prefix = "dit"

    betas: Tensor
    alphas_cumprod: Tensor

    def __init__(self, config: DiTConfig) -> None:
        """Initialise the model. See the class docstring for parameters."""
        super().__init__(config)
        self.config: DiTConfig = config
        self.dit = DiTModel(config)
        betas = _beta_schedule(config)
        self.register_buffer("betas", betas, persistent=False)
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", lucid.cumprod(alphas), persistent=False)

    def _alphas_cumprod_in(self, dtype: lucid.dtype, device: str) -> Tensor:
        r""":math:`\bar\alpha` in ``dtype`` on ``device``, as precise as built.

        The buffer follows the model's dtype, and float16 rounds
        :math:`\bar\alpha_0` and :math:`\bar\alpha_1` to exactly 1 — after
        which the sampler's last step divides 0 by 0.  So a buffer narrower
        than ``dtype`` is rebuilt from the config rather than widened.
        """
        table = self.alphas_cumprod
        if table.dtype != dtype:
            table = lucid.cumprod(1.0 - _beta_schedule(self.config)).to(dtype)
        return table.to(device)

    def _split(self, prediction: Tensor) -> tuple[Tensor, Tensor | None]:
        """Separate the noise from the covariance the decoder emits."""
        if not self.config.learn_sigma:
            return prediction, None
        channels = self.config.in_channels
        return prediction[:, :channels], prediction[:, channels:]

    @override
    def forward(self, images: Tensor, labels: Tensor | None = None) -> DiTOutput:
        r"""One denoising step of training.

        Parameters
        ----------
        images : Tensor
            ``(B, in_channels, H, W)`` — the VAE latent of a batch.
        labels : Tensor or None, optional
            ``(B,)`` class indices, dropped to the null token with
            probability :attr:`DiTConfig.class_dropout`.

        Returns
        -------
        DiTOutput
            The loss, the predicted noise, and the covariance if learned.
        """
        config = self.config
        batch = int(images.shape[0])
        device = images.device.type

        steps = lucid.randint(
            0, config.num_train_timesteps, size=(batch,), device=device
        )
        dt = images.dtype
        noise = lucid.randn(images.shape, dtype=dt, device=device)
        # At float32 or wider, cast once: on the float16 buffer the first two
        # steps' 1 - alpha_bar is 0, and they added no noise.
        table = self._alphas_cumprod_in(_float32_or_wider(dt), device)
        alpha_bar = table[steps].reshape(-1, 1, 1, 1)
        signal, noise_scale = alpha_bar.sqrt().to(dt), (1.0 - alpha_bar).sqrt().to(dt)
        noised = signal * images + noise_scale * noise

        if labels is not None and config.class_dropout > 0.0:
            drop = lucid.rand((batch,), device=device) < config.class_dropout
            labels = lucid.where(
                drop,
                lucid.full(
                    (batch,),
                    float(config.num_classes),
                    dtype=labels.dtype,
                    device=device,
                ),
                labels,
            )

        prediction = self.dit(noised, steps.float(), labels).sample
        noise_pred, variance = self._split(prediction)
        loss = ((noise_pred - noise) ** 2).mean()
        return DiTOutput(loss=loss, noise_pred=noise_pred, variance_pred=variance)

    def generate(
        self,
        n_samples: int = 1,
        *,
        labels: Tensor | None = None,
        steps: int | None = None,
        eta: float = 0.0,
        noise: Tensor | None = None,
        device: str | None = None,
    ) -> GenerationOutput:
        r"""Sample by running the reverse diffusion process.

        Parameters
        ----------
        n_samples : int, default=1
            How many to draw.
        labels : Tensor or None, optional
            ``(n_samples,)`` class indices.  ``None`` samples the
            unconditional field.
        steps : int or None, optional
            Denoising steps.  Defaults to the full training schedule.
        eta : float, default=0.0
            Interpolates the reverse step between DDIM at ``0`` and DDPM
            at ``1``.  **The paper's numbers are at the DDPM end** — it
            follows ADM and reports FID over 250 DDPM steps — so
            ``steps=250, eta=1.0`` is the protocol to compare against.
            The default is deterministic instead, because a sampler that
            draws on every call cannot be reproduced without a seed and
            this family takes no seed.
        noise : Tensor or None, optional
            Starting latent.  Drawn when absent.
        device : str or None, optional
            Where to draw.  Defaults to the model's own device.

        Returns
        -------
        GenerationOutput
            ``samples`` of shape ``(n_samples, in_channels, H, W)`` — a
            latent, which a VAE decoder turns into pixels.

        Raises
        ------
        ValueError
            If ``steps`` is not positive, or ``eta`` is outside ``[0, 1]``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import dit_small_2_gen
        >>> model = dit_small_2_gen(
        ...     sample_size=8, hidden_size=32, depth=1, num_heads=4)
        >>> model.generate(1, steps=2).samples.shape
        (1, 4, 8, 8)

        The paper's protocol is the stochastic end of ``eta``:

        >>> model.generate(1, steps=2, eta=1.0).samples.shape
        (1, 4, 8, 8)
        """
        config = self.config
        total = config.num_train_timesteps
        steps = total if steps is None else steps
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}")
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta interpolates DDIM and DDPM, got {eta}")
        side = (
            config.sample_size
            if isinstance(config.sample_size, int)
            else config.sample_size[0]
        )
        device = resolve_generation_device(self, device)
        if noise is None:
            noise = start_noise(
                self, (n_samples, config.in_channels, side, side), device
            )

        z = noise
        dt = z.dtype
        alphas_cumprod = self._alphas_cumprod_in(_float32_or_wider(dt), device)
        grid = [int(round(i * total / steps)) for i in range(steps, 0, -1)]
        with lucid.no_grad():
            for index, step in enumerate(grid):
                t = min(step, total - 1)
                ts = lucid.full((n_samples,), float(t), device=device, dtype=dt)
                eps, _ = self._split(self.dit(z, ts, labels).sample)

                alpha_bar = alphas_cumprod[t]
                previous = grid[index + 1] if index + 1 < len(grid) else 0
                # The final step's alpha_bar_prev = 1 has to sit with the
                # schedule; a bare CPU scalar broke the last step on Metal.
                alpha_bar_prev = (
                    alphas_cumprod[previous]
                    if previous > 0
                    else lucid.ones_like(alpha_bar)
                )
                signal, noise_scale, keep, direction, sigma = _reverse_step(
                    alpha_bar, alpha_bar_prev, eta, dt
                )
                x0 = (z - noise_scale * eps) / signal
                z = keep * x0 + direction * eps
                if eta > 0.0:
                    z = z + sigma * lucid.randn(z.shape, dtype=dt, device=device)
        return GenerationOutput(samples=z)
