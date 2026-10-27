"""VAE model — Kingma & Welling (2013) + Sønderby et al. (2016) hierarchical.

Single class supports three modes selected by :attr:`VAEConfig.latent_dim`:

    * **Vanilla VAE** (``int`` latent_dim) — one bottleneck ``z`` projected
      from the flattened final encoder stage.
    * **β-VAE**       (``int`` latent_dim + ``kl_weight ≠ 1``) — same trunk,
      scaled KL.
    * **Hierarchical VAE / HVAE** (``tuple`` latent_dim) — one ``z_l`` per
      encoder stage, with the decoder injecting ``z_l`` back at the matching
      upsampling spatial scale.  Independent posteriors variant of Sønderby
      Ladder VAE (each ``z_l`` has prior ``N(0, I)`` and the ELBO sums KL
      across levels).

Architecture per shape (vanilla shown; HVAE adds per-stage taps + injects):

    encoder:
        x → [Conv2d↓2 → SiLU] × L → flat → 2 × Linear → (μ, logσ²)

    decoder:
        z → Linear → reshape(c_top, h_top, w_top)
          → [ConvT2d↑2 → SiLU] × L → Conv2d 3×3 → x̂

    loss = recon(x, x̂) + β · KL(q(z|x) ‖ N(0, I))

Hierarchical mode replaces the encoder's single (μ, logσ²) with one pair
per stage and the decoder's single ``z`` projection with a top-stage
``z_{L-1}`` projection plus per-stage spatial-broadcast injections of
``z_{L-2}, …, z_0``.
"""

from typing import ClassVar, cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageGenerationModel
from lucid.models._output import GenerationOutput, VAEOutput
from lucid.models._utils._generative import (
    gaussian_kl_divergence,
    generative_activation,
    reparameterize,
    start_noise,
)
from lucid.models.generative.vae._config import VAEConfig

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_hw(config: VAEConfig) -> tuple[int, int, int, int]:
    """Return ``(h_top, w_top, c_top, L)`` — top-of-encoder geometry."""
    if isinstance(config.sample_size, tuple):
        h0, w0 = config.sample_size
    else:
        h0 = w0 = int(config.sample_size)
    L = len(config.down_block_channels)
    factor = 2**L
    return h0 // factor, w0 // factor, config.down_block_channels[-1], L


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────


@final
class _VAEEncoder(nn.Module):
    """Conv-stack encoder with two operating modes.

    * Vanilla / β-VAE: single (μ, logσ²) projected from the flattened final
      stage.  ``forward`` returns a 2-tuple of ``(mu, logvar)``.
    * Hierarchical: (μ_l, logσ²_l) extracted per stage from each block's
      output via global-average pool + Linear.  ``forward`` returns
      ``(list[mu_l], list[logvar_l])``.
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self._act_name = config.act_fn
        self._hier = config.is_hierarchical
        h_top, w_top, c_top, levels = _resolve_hw(config)
        self._factor = 2**levels
        self._sample_hw = (h_top * self._factor, w_top * self._factor)
        self._h_top = h_top
        self._w_top = w_top
        self._c_top = c_top

        in_ch = config.in_channels
        blocks: list[nn.Conv2d] = []
        for out_ch in config.down_block_channels:
            blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            in_ch = out_ch
        self.down_blocks = nn.ModuleList(blocks)

        if self._hier:
            # Per-stage (μ_l, logvar_l) from global-average-pooled features.
            mus: list[nn.Linear] = []
            logs: list[nn.Linear] = []
            for stage_ch, lat_dim in zip(
                config.down_block_channels, config.latent_dims, strict=True
            ):
                mus.append(nn.Linear(stage_ch, lat_dim))
                logs.append(nn.Linear(stage_ch, lat_dim))
            self.fc_mus = nn.ModuleList(mus)
            self.fc_logvars = nn.ModuleList(logs)
        else:
            flat = c_top * h_top * w_top
            assert isinstance(config.latent_dim, int)
            self.fc_mu = nn.Linear(flat, config.latent_dim)
            self.fc_logvar = nn.Linear(flat, config.latent_dim)

    def _gap(self, h: Tensor) -> Tensor:
        """Global average pool over (H, W) → (B, C)."""
        return h.mean(dim=(-1, -2))

    def _check_input_size(self, height: int, width: int) -> None:
        """Refuse an image of a size other than the one the VAE models.

        The likelihood p(x | z) is over images of one size: the decoder
        starts from a fixed ``c_top x h_top x w_top`` map and up-samples it,
        so its reconstruction is ``sample_size`` whatever went in.  Before
        this check another size failed late, as a reshape into the
        encoder's flattened head or as the reconstruction loss broadcasting
        32x32 against 33x33, and a side the stride-2 convolutions round
        away (33 against a 32 build) answered ``VAEModel.forward`` silently
        with a 32x32 reconstruction.
        """
        built_h, built_w = self._sample_hw
        if (height, width) == (built_h, built_w):
            return
        rebuild = (
            f", or build the model with sample_size={height}"
            if height == width and height % self._factor == 0
            else ""
        )
        raise ValueError(
            f"VAE is built for {built_h}x{built_w} images: its decoder "
            f"reconstructs from a fixed {self._h_top}x{self._w_top} map, so a "
            f"reconstruction is always {built_h}x{built_w}.  Got "
            f"{height}x{width}.  Resize the input to {built_h}x{built_w}{rebuild}."
        )

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor] | tuple[list[Tensor], list[Tensor]]:
        self._check_input_size(int(x.shape[-2]), int(x.shape[-1]))
        h = x
        if self._hier:
            mus: list[Tensor] = []
            logvars: list[Tensor] = []
            for blk, fc_mu, fc_logvar in zip(
                self.down_blocks, self.fc_mus, self.fc_logvars, strict=True
            ):
                h = blk(h)
                h = generative_activation(self._act_name, h)
                pooled = self._gap(h)  # (B, c_l)
                mus.append(fc_mu(pooled))
                logvars.append(fc_logvar(pooled))
            return mus, logvars

        for blk in self.down_blocks:
            h = blk(h)
            h = generative_activation(self._act_name, h)
        B = int(h.shape[0])
        h = h.reshape(B, self._c_top * self._h_top * self._w_top)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────


@final
class _VAEDecoder(nn.Module):
    """Top-down decoder with two operating modes.

    * Vanilla / β-VAE: ``Linear(latent → c_top·h_top·w_top) → reshape →
      ConvT-stack → head``.
    * Hierarchical: top-stage ``Linear(latent_dims[-1] → c_top·h_top·w_top)``
      starts the trunk; for each upsample step ``l = L-2 … 0`` a
      ``Linear(latent_dims[l] → c_l)`` projection of ``z_l`` is broadcast
      across spatial dims and added to the post-upsample features.
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self._act_name = config.act_fn
        self._hier = config.is_hierarchical

        h_top, w_top, c_top, L = _resolve_hw(config)
        self._h_top = h_top
        self._w_top = w_top
        self._c_top = c_top
        self._L = L

        rev_channels = list(reversed(config.down_block_channels))
        # Initial Linear → top spatial.
        if self._hier:
            top_latent = config.latent_dims[-1]
            self.fc = nn.Linear(top_latent, c_top * h_top * w_top)
        else:
            assert isinstance(config.latent_dim, int)
            self.fc = nn.Linear(config.latent_dim, c_top * h_top * w_top)

        # Upsampling stack.  Channels go (c_top → ... → c_0) over L blocks;
        # the final ConvTranspose lifts to ``c_0`` so the head can mix.
        up_blocks: list[nn.ConvTranspose2d] = []
        in_ch = rev_channels[0]
        # rev_channels has L entries; we want L upsample blocks.  After the
        # last upsample the channel count stays at rev_channels[-1].
        for out_ch in rev_channels[1:]:
            up_blocks.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            in_ch = out_ch
        # Final upsample keeps channel count.
        up_blocks.append(nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.head = nn.Conv2d(in_ch, config.out_channels, kernel_size=3, padding=1)

        # Per-level injection projections (HVAE only).  Block l (0-indexed
        # in the *upsample* direction) consumes z at the *encoder*-side
        # level ``L-1-l`` (so the deepest z_{L-1} is consumed by the
        # initial Linear above; the next block consumes z_{L-2}; …).
        if self._hier:
            inject: list[nn.Linear] = []
            # We use the post-upsample feature's channel count to size the
            # injection.  For block index l in 0..L-1, post-upsample
            # channels are ``rev_channels[l+1]`` for l < L-1 and
            # ``rev_channels[-1]`` for l == L-1 (last block keeps channels).
            post_channels = tuple(rev_channels[1:]) + (rev_channels[-1],)
            # Levels matched: block l consumes z_{L-2-l} for l in 0..L-2;
            # block L-1 has no z to inject (already consumed all but z_{L-1}
            # which feeds the initial Linear).
            #
            # We register one projection *per* upsample step so iteration
            # order in forward stays simple — unused slot at the last
            # position is None-out via a placeholder Linear of size 0?  No,
            # cleaner: only register injections for the L-1 non-deepest
            # latents and skip in forward.
            for lvl in range(L - 1):
                # post_channels[lvl] is the channel count after this block.
                inject.append(nn.Linear(config.latent_dims[L - 2 - lvl], post_channels[lvl]))
            self.injectors = nn.ModuleList(inject)
        # Cache for forward dispatch.
        self._post_channels = tuple(rev_channels[1:]) + (rev_channels[-1],) if self._hier else ()

    @override
    def forward(
        self,
        z_or_zs: Tensor | list[Tensor],
    ) -> Tensor:
        if self._hier:
            assert isinstance(z_or_zs, list), "Hierarchical decoder expects a list of z_l"
            zs = z_or_zs
            # Bottom-up encoder produced (z_0, …, z_{L-1}); decoder consumes
            # them top-down.
            top_z = zs[-1]
            h = self.fc(top_z)
            B = int(h.shape[0])
            h = h.reshape(B, self._c_top, self._h_top, self._w_top)
            for lvl, blk in enumerate(self.up_blocks):
                h = blk(h)
                h = generative_activation(self._act_name, h)
                if lvl < self._L - 1:
                    z_inj = zs[self._L - 2 - lvl]  # (B, d_l)
                    proj = self.injectors[lvl](z_inj)  # (B, post_c)
                    proj = proj.reshape(B, self._post_channels[lvl], 1, 1).expand(
                        B, self._post_channels[lvl], int(h.shape[2]), int(h.shape[3])
                    )
                    h = h + proj
            return self.head(h)

        # Vanilla path
        assert isinstance(z_or_zs, Tensor), "Vanilla decoder expects a single Tensor z"
        h = self.fc(z_or_zs)
        B = int(h.shape[0])
        h = h.reshape(B, self._c_top, self._h_top, self._w_top)
        for blk in self.up_blocks:
            h = blk(h)
            h = generative_activation(self._act_name, h)
        return self.head(h)


# ─────────────────────────────────────────────────────────────────────────────
# VAE model — bare encoder-decoder
# ─────────────────────────────────────────────────────────────────────────────


class VAEModel(PretrainedModel):
    r"""Bare convolutional VAE — vanilla / :math:`\beta`-VAE / hierarchical.

    Implements the variational auto-encoder of Kingma and Welling, 2014
    in three operating modes selected implicitly by
    :attr:`VAEConfig.latent_dim`:

    * **Vanilla / :math:`\beta`-VAE** (``int`` latent_dim): a single
      bottleneck ``z`` is projected from the flattened final encoder
      stage.  ``encode(x) -> (mu, logvar)`` with shape ``(B, D)`` each;
      ``decode(z: (B, D)) -> x_hat``.
    * **Hierarchical / Ladder VAE** (``tuple`` latent_dim, length :math:`L`
      = ``len(down_block_channels)``): one ``z_l`` per encoder stage;
      the decoder injects ``z_l`` back at the matching upsampling spatial
      scale.  ``encode(x) -> (mus: list[Tensor], logvars: list[Tensor])``
      with ``mus[l]`` of shape ``(B, latent_dims[l])``; ``decode(zs:
      list[Tensor]) -> x_hat``.

    Reparameterisation is applied internally in :meth:`forward` —
    :math:`z = \mu + \sigma \odot \epsilon`,
    :math:`\epsilon \sim \mathcal{N}(0, \mathbf{I})` — so gradients flow
    pathwise through the stochastic sample.

    Parameters
    ----------
    config : VAEConfig
        Hyperparameters controlling sample size, channel widths
        (``down_block_channels``), latent topology (``latent_dim``), and
        loss-related options.  See :class:`VAEConfig` for the full list.

    Attributes
    ----------
    encoder : nn.Module
        Conv-stack encoder producing the posterior parameters
        :math:`(\mu, \log \sigma^2)`.  Output shape depends on the mode
        (see class summary).
    decoder : nn.Module
        Top-down ConvTranspose decoder mapping the latent (or list of
        per-level latents) back to image space.

    Notes
    -----
    Reference: Kingma and Welling, *"Auto-Encoding Variational Bayes"*,
    ICLR, 2014 (arXiv:1312.6114); hierarchical extension from Sønderby,
    Raiko, Maaløe, Sønderby, and Winther, *"Ladder Variational
    Autoencoders"*, NeurIPS, 2016 (arXiv:1602.02282).

    The closed-form KL term used by the loss head is

    .. math::

        \mathrm{KL}\!\big(
            \mathcal{N}(\mu, \sigma^2) \,\big\|\,
            \mathcal{N}(0, \mathbf{I})
        \big)
            = \tfrac{1}{2} \sum_i \!\big(
                \mu_i^2 + \sigma_i^2 - 1 - \log \sigma_i^2
              \big).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.vae import VAEConfig, VAEModel
    >>> cfg = VAEConfig(sample_size=32, in_channels=3, out_channels=3,
    ...                 latent_dim=16, down_block_channels=(16, 32, 64))
    >>> model = VAEModel(cfg).eval()
    >>> x = lucid.randn((1, 3, 32, 32))
    >>> out = model(x)
    >>> out.sample.shape, out.latent.shape   # reconstruction + latent
    ((1, 3, 32, 32), (1, 16))
    """

    config_class: ClassVar[type[VAEConfig]] = VAEConfig
    base_model_prefix: ClassVar[str] = "vae"

    def __init__(self, config: VAEConfig) -> None:
        super().__init__(config)
        self._is_hierarchical = config.is_hierarchical
        self._latent_dims = config.latent_dims
        self.encoder = _VAEEncoder(config)
        self.decoder = _VAEDecoder(config)

    @property
    def is_hierarchical(self) -> bool:
        return self._is_hierarchical

    @property
    def latent_dims(self) -> tuple[int, ...]:
        return self._latent_dims

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor] | tuple[list[Tensor], list[Tensor]]:
        """Return posterior parameters of ``q(z|x)``.

        Vanilla mode → ``(mu, logvar)``.  Hierarchical mode → ``(mus,
        logvars)`` with one entry per encoder stage.
        """
        return self.encoder(x)

    def decode(self, z_or_zs: Tensor | list[Tensor]) -> Tensor:
        """Map a latent (or list of per-level latents) back to image space."""
        return self.decoder(z_or_zs=z_or_zs)

    @override
    def forward(self, x: Tensor) -> VAEOutput:
        if self._is_hierarchical:
            mus, logvars = cast(tuple[list[Tensor], list[Tensor]], self.encode(x))
            zs = [reparameterize(m, lv) for m, lv in zip(mus, logvars, strict=True)]
            recon = self.decode(zs)
            # Pack per-level tensors into a flat (B, sum(D)) blob so the
            # VAEOutput contract (single mu / logvar / latent) holds for
            # both modes.  Callers that want per-level access read the model
            # directly via ``.encode`` / ``.decode``.
            mu_flat = lucid.cat(mus, dim=-1)
            logvar_flat = lucid.cat(logvars, dim=-1)
            z_flat = lucid.cat(zs, dim=-1)
            return VAEOutput(
                sample=recon,
                latent=z_flat,
                mu=mu_flat,
                logvar=logvar_flat,
            )

        mu, logvar = cast(tuple[Tensor, Tensor], self.encode(x))
        z = reparameterize(mu, logvar)
        recon = self.decode(z)
        return VAEOutput(sample=recon, latent=z, mu=mu, logvar=logvar)


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper — ELBO loss + prior sampling
# ─────────────────────────────────────────────────────────────────────────────


class VAEForImageGeneration(ImageGenerationModel):
    r"""VAE with the ELBO training loss and prior-sample ``.generate()``.

    Wraps :class:`VAEModel` with the standard reconstruction +
    :math:`\beta`-weighted KL training loss and a prior-sampler convenience
    method.  ``forward(x)`` returns a :class:`VAEOutput` carrying the
    reconstruction, latent (concatenated across levels in HVAE mode),
    posterior parameters, and the three loss components
    (``recon_loss``, ``kl_loss``, ``loss = recon + beta * kl``).  In
    hierarchical mode ``kl_loss`` is the sum of per-level KL terms.

    ``generate(n_samples)`` samples from the prior —
    :math:`z \sim \mathcal{N}(0, \mathbf{I})` for vanilla, one
    :math:`z_l \sim \mathcal{N}(0, \mathbf{I})` per level for HVAE — and
    returns the decoder output.

    Under ``recon_loss="bce"`` the decoder emits logits, and both
    ``forward(x).sample`` and ``generate().samples`` squash them through a
    sigmoid so every reconstruction this class hands back lives in the same
    ``[0, 1]`` space as the input.  The loss is still computed from the raw
    logits.

    Parameters
    ----------
    config : VAEConfig
        Hyperparameters.  ``config.kl_weight`` is the :math:`\beta`
        coefficient (Higgins 2017); ``config.recon_loss`` toggles between
        Gaussian (``"mse"``) and Bernoulli (``"bce"``) likelihoods.

    Attributes
    ----------
    vae : VAEModel
        Underlying VAE trunk providing ``encode`` and ``decode``.

    Notes
    -----
    Reference: Kingma and Welling, *"Auto-Encoding Variational Bayes"*,
    ICLR, 2014 (arXiv:1312.6114); :math:`\beta`-VAE in Higgins et al.,
    ICLR, 2017; hierarchical extension in Sønderby et al., NeurIPS, 2016
    (arXiv:1602.02282).

    Training objective — evidence lower bound:

    .. math::

        \mathcal{L}_{\mathrm{ELBO}}
            = \mathbb{E}_{q_\phi(z \mid x)}\!\big[\log p_\theta(x \mid z)\big]
              - \beta \cdot \mathrm{KL}\!\big(
                    q_\phi(z \mid x) \,\big\|\, p(z)
                \big).

    In hierarchical mode the KL term is the sum
    :math:`\sum_l \mathrm{KL}(q_\phi(z_l \mid x) \| p(z_l))`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.vae import (
    ...     VAEConfig, VAEForImageGeneration,
    ... )
    >>> cfg = VAEConfig(sample_size=32, in_channels=3, out_channels=3,
    ...                 latent_dim=16, down_block_channels=(16, 32, 64))
    >>> model = VAEForImageGeneration(cfg).eval()
    >>> x = lucid.randn((1, 3, 32, 32))
    >>> out = model(x)
    >>> out.sample.shape, out.loss.shape   # reconstruction + scalar loss
    ((1, 3, 32, 32), ())
    """

    config_class: ClassVar[type[VAEConfig]] = VAEConfig
    base_model_prefix: ClassVar[str] = "vae"

    def __init__(self, config: VAEConfig) -> None:
        super().__init__(config)
        self.vae = VAEModel(config)
        self._kl_weight = config.kl_weight
        self._recon_loss = config.recon_loss
        self._is_hierarchical = config.is_hierarchical
        self._latent_dims = config.latent_dims

    def _to_data_space(self, recon: Tensor) -> Tensor:
        """Map a raw decoder output into the space the data lives in.

        Under a Bernoulli likelihood the decoder emits logits — that is what
        the reconstruction loss needs — but the *reconstruction* a caller
        reads off the output should be comparable with the input it passed
        in.  ``generate`` already squashed its samples; ``forward`` did not,
        so ``out.sample`` and ``generate().samples`` lived in different
        spaces for the same model.
        """
        return F.sigmoid(recon) if self._recon_loss == "bce" else recon

    def _reconstruction_loss(self, recon: Tensor, target: Tensor) -> Tensor:
        if self._recon_loss == "mse":
            diff = (recon - target) ** 2
            B = int(diff.shape[0])
            return diff.reshape(B, -1).sum(dim=-1).mean()
        # bce
        recon_p = F.sigmoid(recon)
        eps = 1e-7
        recon_p = recon_p.clip(eps, 1.0 - eps)
        per_pixel = -(target * recon_p.log() + (1.0 - target) * (1.0 - recon_p).log())
        B = int(per_pixel.shape[0])
        return per_pixel.reshape(B, -1).sum(dim=-1).mean()

    @override
    def forward(self, x: Tensor) -> VAEOutput:
        if self._is_hierarchical:
            mus, logvars = cast(tuple[list[Tensor], list[Tensor]], self.vae.encode(x))
            zs = [reparameterize(m, lv) for m, lv in zip(mus, logvars, strict=True)]
            recon = self.vae.decode(zs)

            recon_l = self._reconstruction_loss(recon, x)
            # Sum closed-form KL across levels.
            kl_terms = [
                gaussian_kl_divergence(m, lv, reduction="mean")
                for m, lv in zip(mus, logvars, strict=True)
            ]
            kl_l = kl_terms[0]
            for term in kl_terms[1:]:
                kl_l = kl_l + term
            total = recon_l + self._kl_weight * kl_l

            mu_flat = lucid.cat(mus, dim=-1)
            logvar_flat = lucid.cat(logvars, dim=-1)
            z_flat = lucid.cat(zs, dim=-1)
            return VAEOutput(
                sample=self._to_data_space(recon),
                latent=z_flat,
                mu=mu_flat,
                logvar=logvar_flat,
                loss=total,
                recon_loss=recon_l,
                kl_loss=kl_l,
            )

        mu, logvar = cast(tuple[Tensor, Tensor], self.vae.encode(x))
        z = reparameterize(mu, logvar)
        recon = self.vae.decode(z)

        recon_l = self._reconstruction_loss(recon, x)
        kl_l = gaussian_kl_divergence(mu, logvar, reduction="mean")
        total = recon_l + self._kl_weight * kl_l

        return VAEOutput(
            sample=self._to_data_space(recon),
            latent=z,
            mu=mu,
            logvar=logvar,
            loss=total,
            recon_loss=recon_l,
            kl_loss=kl_l,
        )

    @lucid.no_grad()
    def generate(
        self,
        n_samples: int = 1,
        *,
        device: str | None = None,
    ) -> GenerationOutput:
        """Sample ``n_samples`` images from the prior ``N(0, I)``."""
        if self._is_hierarchical:
            zs = [start_noise(self, (n_samples, d), device) for d in self._latent_dims]
            samples = self.vae.decode(zs)
        else:
            z = start_noise(self, (n_samples, int(self._latent_dims[0])), device)
            samples = self.vae.decode(z)
        if self._recon_loss == "bce":
            samples = F.sigmoid(samples)
        return GenerationOutput(samples=samples)
