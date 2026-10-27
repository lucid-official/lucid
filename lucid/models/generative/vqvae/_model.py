"""VQ-VAE model + task wrapper + output dataclass.

Architecture per van den Oord et al., 2017 (Section 4.1):

    encoder:
        x -> [Conv2d(4x4, stride 2) -> act] x L -> [residual 3x3] x R
          -> Conv2d 1x1 -> z_e            (B, D, H/2^L, W/2^L)

    quantiser:
        z_e -> nearest codebook entry per spatial position -> z_q, indices

    decoder:
        z_q -> Conv2d 3x3 -> [residual 3x3] x R -> act
             -> [ConvTranspose2d(4x4, stride 2) -> act] x (L-1)
             -> ConvTranspose2d(4x4, stride 2) -> x_hat

    loss = recon(x, x_hat)
         + ||sg[z_e] - e||^2            (codebook)
         + beta * ||z_e - sg[e]||^2     (commitment)

The quantiser itself is not defined here: codebook lookup is a general
layer, not a VQ-VAE detail, so it lives in the core API as
:class:`lucid.nn.VectorQuantizer` (built on
:func:`lucid.nn.functional.straight_through`) and this family composes
it.  The only adaptation needed is layout — the core layer follows the
framework's trailing-axis convention while image models carry channels
second, so :meth:`VQVAEModel.quantize` permutes around the call.

The residual block below stays private: its ``act, 3x3, act, 1x1`` shape
is specific to this paper, and every vision family in the zoo defines its
own block rather than sharing one.
"""

from dataclasses import dataclass
from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageGenerationModel
from lucid.models._output import GenerationOutput, ModelOutput
from lucid.models._utils._generative import (
    generative_activation,
    resolve_generation_device,
)
from lucid.models.generative.vqvae._config import VQVAEConfig

# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class VQVAEOutput(ModelOutput):
    r"""Forward output of the discrete-latent auto-encoder.

    Attributes
    ----------
    sample : Tensor
        Reconstruction :math:`\hat{x}` shaped ``(B, C, H, W)``.
    latent : Tensor
        Quantised latent field :math:`z_q(x)` shaped
        ``(B, embedding_dim, H', W')``, carrying the straight-through
        gradient path back to the encoder.
    indices : Tensor
        Codebook assignment per spatial position, shaped ``(B, H', W')``
        with ``int64`` dtype and values in ``[0, num_embeddings)``.  This
        is the discrete token field a downstream prior would model.
    perplexity : Tensor
        Scalar :math:`\exp(-\sum_k p_k \log p_k)` over the batch's
        codebook usage histogram.  Ranges over ``[1, num_embeddings]``;
        a value collapsing toward 1 means the codebook has died down to a
        handful of live entries.
    loss : Tensor or None, optional
        Total objective — ``recon_loss + codebook_loss + beta *
        commitment_loss``.  ``None`` on the bare
        :class:`VQVAEModel`, which does not build a loss.
    recon_loss : Tensor or None, optional
        Reconstruction term alone.
    codebook_loss : Tensor or None, optional
        :math:`\| \mathrm{sg}[z_e] - e \|_2^2` — moves the codebook
        toward the encoder outputs assigned to it.
    commitment_loss : Tensor or None, optional
        :math:`\| z_e - \mathrm{sg}[e] \|_2^2` — moves the encoder
        toward the entry it selected.  Reported *unweighted*; the
        :math:`\beta` scaling is applied only inside ``loss``.

    Notes
    -----
    Returned by :meth:`VQVAEModel.forward` (losses ``None``) and by
    :meth:`VQVAEForImageGeneration.forward` (losses populated).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.vqvae import (
    ...     VQVAEConfig, VQVAEForImageGeneration,
    ... )
    >>> cfg = VQVAEConfig(sample_size=32, hidden_channels=16,
    ...                   residual_hidden_channels=16, embedding_dim=8,
    ...                   num_embeddings=32)
    >>> out = VQVAEForImageGeneration(cfg).eval()(lucid.randn((4, 3, 32, 32)))
    >>> out.sample.shape, out.indices.shape
    ((4, 3, 32, 32), (4, 8, 8))
    """

    sample: Tensor
    latent: Tensor
    indices: Tensor
    perplexity: Tensor
    loss: Tensor | None = None
    recon_loss: Tensor | None = None
    codebook_loss: Tensor | None = None
    commitment_loss: Tensor | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Private building blocks
# ─────────────────────────────────────────────────────────────────────────────


@final
class _VQVAEResidualBlock(nn.Module):
    """Pre-activation residual block — ``act, 3x3 conv, act, 1x1 conv``.

    Ordering follows the paper's parenthetical description of the image
    experiments; the leading activation is what makes it pre-activation,
    so a stack of these can be composed without an activation between
    them and the trunk's output stays unactivated — and nothing may
    activate immediately before one, or the boundary gets the activation
    twice.

    Both convolutions are bias-free.  The paper does not say either way;
    the bias is redundant here because every path out of this block is
    summed with ``x``, which carries its own, and dropping it is the
    common choice in residual stacks.
    """

    def __init__(self, channels: int, hidden: int, act_fn: str) -> None:
        super().__init__()
        self._act_name = act_fn
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = generative_activation(self._act_name, x)
        h = self.conv1(h)
        h = generative_activation(self._act_name, h)
        h = self.conv2(h)
        return x + h


@final
class _VQVAEEncoder(nn.Module):
    """Strided conv stack, residual blocks, then a 1x1 projection to ``D``.

    The trailing projection is what decouples the trunk width from the
    code dimension; at the paper's defaults (both 256) it is an extra
    learned 1x1 layer rather than a shape change.
    """

    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__()
        self._act_name = config.act_fn

        blocks: list[nn.Conv2d] = []
        in_ch = config.in_channels
        for _ in range(config.num_downsample_layers):
            blocks.append(
                nn.Conv2d(in_ch, config.hidden_channels, kernel_size=4, stride=2, padding=1)
            )
            in_ch = config.hidden_channels
        self.down_blocks = nn.ModuleList(blocks)

        self.residuals = nn.ModuleList(
            [
                _VQVAEResidualBlock(
                    config.hidden_channels,
                    config.residual_hidden_channels,
                    config.act_fn,
                )
                for _ in range(config.num_residual_layers)
            ]
        )
        self.proj = nn.Conv2d(config.hidden_channels, config.embedding_dim, kernel_size=1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = x
        last = len(self.down_blocks) - 1
        for i, blk in enumerate(self.down_blocks):
            h = blk(h)
            # Not after the last one.  What follows is either a
            # pre-activation residual block or the trailing activation
            # below, and both open with an activation of their own — so
            # activating here too applied it twice at that boundary.  With
            # the default ReLU that is idempotent and invisible; with the
            # ``silu`` / ``gelu`` this config also accepts it is not
            # (silu(silu(x)) differs from silu(x) by up to 0.26), and with
            # ``num_residual_layers=0`` it doubled for every activation.
            # The decoder never had this — nothing activates between its
            # lift and its residual stack.
            if i != last:
                h = generative_activation(self._act_name, h)
        for res in self.residuals:
            h = res(h)
        h = generative_activation(self._act_name, h)
        return self.proj(h)


@final
class _VQVAEDecoder(nn.Module):
    """Mirror of the encoder — 3x3 lift, residual blocks, transposed convs."""

    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__()
        self._act_name = config.act_fn

        self.lift = nn.Conv2d(
            config.embedding_dim, config.hidden_channels, kernel_size=3, padding=1
        )
        self.residuals = nn.ModuleList(
            [
                _VQVAEResidualBlock(
                    config.hidden_channels,
                    config.residual_hidden_channels,
                    config.act_fn,
                )
                for _ in range(config.num_residual_layers)
            ]
        )

        ups: list[nn.ConvTranspose2d] = []
        n_up = config.num_downsample_layers
        for i in range(n_up):
            last = i == n_up - 1
            ups.append(
                nn.ConvTranspose2d(
                    config.hidden_channels,
                    config.out_channels if last else config.hidden_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
        self.up_blocks = nn.ModuleList(ups)
        self._n_up = n_up

    @override
    def forward(self, z_q: Tensor) -> Tensor:
        h = self.lift(z_q)
        for res in self.residuals:
            h = res(h)
        h = generative_activation(self._act_name, h)
        for i, blk in enumerate(self.up_blocks):
            h = blk(h)
            # No activation after the final transposed conv: its output is
            # the reconstruction (or the Bernoulli logits) itself.
            if i < self._n_up - 1:
                h = generative_activation(self._act_name, h)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Direct model — encoder / quantiser / decoder, no loss
# ─────────────────────────────────────────────────────────────────────────────


class VQVAEModel(PretrainedModel):
    r"""Bare discrete-latent auto-encoder — encoder, codebook, decoder.

    Implements the VQ-VAE of van den Oord, Vinyals, and Kavukcuoglu, 2017.
    An image is encoded to a spatial field of continuous vectors, each
    snapped to its nearest entry in a learned codebook of
    :attr:`VQVAEConfig.num_embeddings` codes, and decoded back.  The
    quantisation step is non-differentiable, so :meth:`forward` routes the
    decoder's gradient onto the encoder with the straight-through
    estimator.

    This class computes no loss.  Use
    :class:`VQVAEForImageGeneration` for the training objective, or read
    the ``codebook_loss`` / ``commitment_loss`` terms off the returned
    :class:`VQVAEOutput` and combine them yourself.

    The pair :meth:`encode_indices` / :meth:`decode_indices` is the
    tokeniser interface: images in, an integer field out, and back again.
    Downstream discrete models over the latent grid — the paper's own
    PixelCNN prior among them — consume exactly that.

    Parameters
    ----------
    config : VQVAEConfig
        Hyperparameters controlling image shape, trunk width, codebook
        size, and the commitment coefficient.  See :class:`VQVAEConfig`.

    Attributes
    ----------
    encoder : nn.Module
        Strided-conv encoder emitting ``(B, embedding_dim, H', W')``.
    quantizer : nn.VectorQuantizer
        Core-API codebook layer holding ``(num_embeddings,
        embedding_dim)`` entries.
    decoder : nn.Module
        Transposed-conv decoder mapping the quantised field back to image
        space.

    Notes
    -----
    Reference: van den Oord, Vinyals, and Kavukcuoglu, *"Neural Discrete
    Representation Learning"*, NeurIPS, 2017 (arXiv:1711.00937).

    The straight-through estimator used here is

    .. math::

        z_q^{\mathrm{st}} = z_e + \mathrm{sg}\!\big[z_q - z_e\big],

    which equals :math:`z_q` in the forward pass and has the identity
    Jacobian with respect to :math:`z_e` in the backward pass.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.vqvae import VQVAEConfig, VQVAEModel
    >>> cfg = VQVAEConfig(sample_size=32, hidden_channels=32,
    ...                   residual_hidden_channels=32, embedding_dim=16,
    ...                   num_embeddings=64)
    >>> model = VQVAEModel(cfg).eval()
    >>> x = lucid.randn((1, 3, 32, 32))
    >>> out = model(x)
    >>> out.sample.shape, out.indices.shape
    ((1, 3, 32, 32), (1, 8, 8))
    """

    config_class: ClassVar[type[VQVAEConfig]] = VQVAEConfig
    base_model_prefix: ClassVar[str] = "vqvae"

    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__(config)
        self._latent_grid_size = config.latent_grid_size
        self._num_embeddings = config.num_embeddings

        self.encoder = _VQVAEEncoder(config)
        self.quantizer = nn.VectorQuantizer(
            config.num_embeddings, config.embedding_dim, config.commitment_cost
        )
        self.decoder = _VQVAEDecoder(config)

    @property
    def latent_grid_size(self) -> tuple[int, int]:
        """Spatial ``(H', W')`` of the discrete latent field."""
        return self._latent_grid_size

    @property
    def num_embeddings(self) -> int:
        """Codebook size :math:`K`."""
        return self._num_embeddings

    def encode(self, x: Tensor) -> Tensor:
        """Return the *continuous* pre-quantisation latent ``z_e(x)``."""
        return self.encoder(x)

    def decode(self, z_q: Tensor) -> Tensor:
        """Decode a quantised latent field back to image space."""
        return self.decoder(z_q)

    def quantize(self, z_e: Tensor) -> nn.VectorQuantizerOutput:
        """Quantise an ``(B, D, H', W')`` latent field.

        :class:`lucid.nn.VectorQuantizer` follows the framework's
        trailing-axis convention, so the channel axis is moved last for
        the lookup and moved back afterwards; the returned ``quantized``
        is in image layout, while ``indices`` is ``(B, H', W')``.
        """
        out = self.quantizer(z_e.permute(0, 2, 3, 1))
        return out._replace(quantized=out.quantized.permute(0, 3, 1, 2))

    @lucid.no_grad()
    def encode_indices(self, x: Tensor) -> Tensor:
        """Tokenise ``x`` into codebook indices shaped ``(B, H', W')``."""
        return self.quantizer.assign(self.encode(x).permute(0, 2, 3, 1))

    @lucid.no_grad()
    def decode_indices(self, indices: Tensor) -> Tensor:
        """Detokenise an index field ``(B, H', W')`` back to an image."""
        return self.decode(self.quantizer.lookup(indices).permute(0, 3, 1, 2))

    @override
    def forward(self, x: Tensor) -> VQVAEOutput:
        q = self.quantize(self.encode(x))
        z_q, indices = q.quantized, q.indices
        codebook_loss, commitment_loss = q.codebook_loss, q.commitment_loss
        perplexity = q.perplexity
        recon = self.decode(z_q)
        return VQVAEOutput(
            sample=recon,
            latent=z_q,
            indices=indices,
            perplexity=perplexity,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper — full objective + codebook-prior sampling
# ─────────────────────────────────────────────────────────────────────────────


class VQVAEForImageGeneration(ImageGenerationModel):
    r"""VQ-VAE with the full training objective and a codebook sampler.

    Wraps :class:`VQVAEModel` with the three-term objective of van den
    Oord et al., 2017 — reconstruction, codebook, and
    :math:`\beta`-weighted commitment — and a convenience sampler.

    ``forward(x)`` returns a :class:`VQVAEOutput` carrying the
    reconstruction, the quantised latent, the integer code field, the
    codebook perplexity, and all four loss tensors.

    Under ``recon_loss="bce"`` the decoder emits logits, which is what the
    Bernoulli likelihood needs; ``out.sample`` and ``generate().samples``
    are squashed through a sigmoid so every reconstruction handed back
    lives in the same ``[0, 1]`` space as the input, while the loss is
    still computed from the raw logits.

    Parameters
    ----------
    config : VQVAEConfig
        Hyperparameters.  ``config.commitment_cost`` is the :math:`\beta`
        coefficient; ``config.recon_loss`` selects the likelihood.

    Attributes
    ----------
    vqvae : VQVAEModel
        Underlying trunk providing ``encode`` / ``decode`` and the
        codebook.

    Notes
    -----
    Reference: van den Oord, Vinyals, and Kavukcuoglu, *"Neural Discrete
    Representation Learning"*, NeurIPS, 2017 (arXiv:1711.00937).

    Training objective:

    .. math::

        L = \log p(x \mid z_q(x))
            + \big\| \mathrm{sg}[z_e(x)] - e \big\|_2^2
            + \beta \big\| z_e(x) - \mathrm{sg}[e] \big\|_2^2 .

    **On ``generate``.** The prior over the discrete latents is uniform
    *during training* — that is the paper's choice, and it is why the KL
    term is the constant :math:`\log K` and drops out of the objective.
    ``generate`` samples from exactly that uniform prior, so it is
    faithful to what this model was trained against, but it is not how
    the paper produces its figures: there, a PixelCNN is fit over the
    latent grid afterwards and sampled autoregressively.  Expect
    incoherent images from the uniform sampler.  Fitting a prior is a
    separate model over ``encode_indices`` output, deliberately outside
    this family.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.vqvae import (
    ...     VQVAEConfig, VQVAEForImageGeneration,
    ... )
    >>> cfg = VQVAEConfig(sample_size=32, hidden_channels=32,
    ...                   residual_hidden_channels=32, embedding_dim=16,
    ...                   num_embeddings=64)
    >>> model = VQVAEForImageGeneration(cfg).eval()
    >>> x = lucid.randn((1, 3, 32, 32))
    >>> out = model(x)
    >>> out.sample.shape, out.loss.shape
    ((1, 3, 32, 32), ())
    """

    config_class: ClassVar[type[VQVAEConfig]] = VQVAEConfig
    base_model_prefix: ClassVar[str] = "vqvae"

    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__(config)
        self.vqvae = VQVAEModel(config)
        self._recon_loss = config.recon_loss
        self._latent_grid_size = config.latent_grid_size
        self._num_embeddings = config.num_embeddings

    def _to_data_space(self, recon: Tensor) -> Tensor:
        """Map a raw decoder output into the space the data lives in."""
        return F.sigmoid(recon) if self._recon_loss == "bce" else recon

    def _reconstruction_loss(self, recon: Tensor, target: Tensor) -> Tensor:
        if self._recon_loss == "mse":
            diff = (recon - target) ** 2
            b = int(diff.shape[0])
            return diff.reshape(b, -1).sum(dim=-1).mean()

        recon_p = F.sigmoid(recon)
        eps = 1e-7
        recon_p = recon_p.clip(eps, 1.0 - eps)
        per_pixel = -(target * recon_p.log() + (1.0 - target) * (1.0 - recon_p).log())
        b = int(per_pixel.shape[0])
        return per_pixel.reshape(b, -1).sum(dim=-1).mean()

    @override
    def forward(self, x: Tensor) -> VQVAEOutput:
        q = self.vqvae.quantize(self.vqvae.encode(x))
        recon = self.vqvae.decode(q.quantized)

        recon_l = self._reconstruction_loss(recon, x)
        # ``VectorQuantizer.loss`` applies the layer's own commitment_cost,
        # which the config seeded — so the beta lives in exactly one place.
        total = recon_l + self.vqvae.quantizer.loss(q)

        return VQVAEOutput(
            sample=self._to_data_space(recon),
            latent=q.quantized,
            indices=q.indices,
            perplexity=q.perplexity,
            loss=total,
            recon_loss=recon_l,
            codebook_loss=q.codebook_loss,
            commitment_loss=q.commitment_loss,
        )

    @lucid.no_grad()
    def generate(
        self,
        n_samples: int = 1,
        *,
        device: str | None = None,
    ) -> GenerationOutput:
        r"""Sample ``n_samples`` images from the uniform codebook prior.

        Draws an index field uniformly over :math:`[0, K)` and decodes it.

        Parameters
        ----------
        n_samples : int, default=1
            Number of images to draw.
        device : str or None, optional, keyword-only
            Device to sample on.  ``None`` resolves to the device the
            model's parameters already live on.

        Returns
        -------
        GenerationOutput
            ``samples`` of shape ``(n_samples, out_channels, H, W)``,
            squashed through a sigmoid when ``recon_loss="bce"`` so the
            result matches the space ``forward`` reports.

        Notes
        -----
        This is the *training-time* prior — uniform is what makes the KL
        term the constant :math:`\log K` and drops it from the objective.
        It is not the paper's generative prior, which is a PixelCNN fit
        over the latent grid afterwards, so these samples are expected to
        be incoherent.  Fitting such a prior is a separate model over
        :meth:`VQVAEModel.encode_indices` output.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.vqvae import VQVAEConfig
        >>> from lucid.models.generative.vqvae import VQVAEForImageGeneration
        >>> cfg = VQVAEConfig(sample_size=32, hidden_channels=16,
        ...                   residual_hidden_channels=16, embedding_dim=8,
        ...                   num_embeddings=32)
        >>> VQVAEForImageGeneration(cfg).eval().generate(2).samples.shape
        (2, 3, 32, 32)
        """
        device = resolve_generation_device(self, device)
        h, w = self._latent_grid_size
        indices = lucid.randint(0, self._num_embeddings, size=(n_samples, h, w), device=device)
        samples = self.vqvae.decode_indices(indices)
        if self._recon_loss == "bce":
            samples = F.sigmoid(samples)
        return GenerationOutput(samples=samples)
