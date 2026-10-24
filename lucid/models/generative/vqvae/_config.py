"""VQ-VAE configuration — van den Oord, Vinyals & Kavukcuoglu, 2017.

The discrete-latent auto-encoder of *Neural Discrete Representation
Learning*.  A convolutional encoder emits a spatial field of continuous
vectors; each is snapped to its nearest entry in a learned codebook, and
the decoder reconstructs from the quantised field.

Architecture knobs follow the paper's image experiments (Section 4.1):
two stride-2 downsampling convolutions, two residual blocks, 256 hidden
units throughout, a codebook of ``K = 512`` entries of dimension
``D = 256``, and a commitment coefficient of ``beta = 0.25``.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    GenerativeModelConfig,
    check_sample_space_channels,
)


@model_family_meta(
    canonical_name="VQ-VAE",
    citation=(
        "van den Oord, Aaron, Oriol Vinyals, and Koray Kavukcuoglu. "
        '"Neural Discrete Representation Learning." Advances in Neural '
        "Information Processing Systems, vol. 30, 2017, pp. 6306-6315."
    ),
    theory=r"""
    The Vector-Quantised Variational Auto-Encoder replaces the continuous
    Gaussian bottleneck of a VAE with a **discrete** one.  An encoder
    :math:`z_e(x)` maps an image to a spatial grid of :math:`D`-dimensional
    vectors, and each grid position is snapped to its nearest neighbour in
    a learned codebook :math:`e \in \mathbb{R}^{K \times D}`:

    .. math::

        q(z = k \mid x) =
            \begin{cases}
                1 & \text{if } k = \arg\min_j \| z_e(x) - e_j \|_2 \\
                0 & \text{otherwise},
            \end{cases}
        \qquad
        z_q(x) = e_k .

    The decoder then reconstructs from :math:`z_q(x)`.  Because the
    :math:`\arg\min` has zero gradient almost everywhere, training uses the
    **straight-through estimator** — the forward pass carries the quantised
    vector while the backward pass copies the decoder's gradient directly
    onto the encoder output, which is implemented as
    :math:`z_q = z_e + \mathrm{sg}[z_q - z_e]` with :math:`\mathrm{sg}`
    the stop-gradient operator.

    The codebook receives no gradient from that path, so it is trained by
    two additional terms.  The **codebook loss** pulls each selected entry
    toward the encoder output it was matched to, and the **commitment
    loss** pulls the encoder output toward the entry it chose, preventing
    the encoder's output space from growing without bound.  The full
    objective is

    .. math::

        L = \underbrace{\log p(x \mid z_q(x))}_{\text{reconstruction}}
            + \underbrace{\big\| \mathrm{sg}[z_e(x)] - e \big\|_2^2}
                        _{\text{codebook}}
            + \beta \underbrace{\big\| z_e(x) - \mathrm{sg}[e] \big\|_2^2}
                        _{\text{commitment}},

    with :math:`\beta = 0.25` in the paper's experiments.  Note the
    asymmetry: the codebook term is unweighted because the codebook has no
    other learning signal, while :math:`\beta` scales how strongly the
    encoder is held to its current assignment.

    Unlike a Gaussian VAE there is no KL term to anneal.  The prior over
    the discrete latents is held **uniform** during training, which makes
    its KL the constant :math:`\log K` and removes posterior collapse as a
    failure mode entirely — the paper's central practical claim.  A
    faithful *generative* prior is fit afterwards as a separate
    autoregressive model (PixelCNN over the latent grid), so the
    auto-encoder trained here is a representation learner and a discrete
    tokeniser rather than a stand-alone sampler.  That tokeniser role is
    what later made the architecture load-bearing across discrete
    generative modelling.
    """,
)
@dataclass(frozen=True)
class VQVAEConfig(GenerativeModelConfig):
    r"""Frozen configuration for the VQ-VAE family.

    Defaults reproduce the image experiments of van den Oord et al., 2017
    (Section 4.1): two stride-2 encoder convolutions with a 4x4 window,
    two residual blocks, 256 hidden units throughout, a ``512 x 256``
    codebook, and ``commitment_cost = 0.25``.

    Parameters
    ----------
    sample_size : int or tuple of int, default=32
        Input spatial resolution.  Inherited from
        :class:`GenerativeModelConfig`.  Must be divisible by
        ``2 ** num_downsample_layers``.
    in_channels : int, default=3
        Input image channels.
    out_channels : int, default=3
        Reconstruction channels.  Must equal ``in_channels``: the
        reconstruction loss compares the decoder's output with the input.
    act_fn : {"silu", "swish", "relu", "gelu"}, default="relu"
        Trunk activation.  The paper uses ReLU throughout, which is why
        this family overrides the generative-domain default of ``"silu"``.
    num_embeddings : int, default=512
        Codebook size :math:`K` — the number of discrete latent codes.
    embedding_dim : int, default=256
        Code dimension :math:`D`.  The encoder's final projection emits
        this many channels so encoder outputs and codebook entries live in
        the same space.
    hidden_channels : int, default=256
        Width of the encoder / decoder trunk.
    num_downsample_layers : int, default=2
        Number of stride-2 convolutions.  The latent grid is
        ``sample_size / 2 ** num_downsample_layers`` on a side.
    num_residual_layers : int, default=2
        Number of residual blocks after the downsampling stack (and,
        mirrored, before the decoder's upsampling stack).
    residual_hidden_channels : int, default=256
        Inner width of each residual block's 3x3 convolution.
    commitment_cost : float, default=0.25
        The coefficient :math:`\beta` weighting the commitment term.
    recon_loss : {"mse", "bce"}, default="mse"
        Reconstruction likelihood — Gaussian (``"mse"``) or Bernoulli
        (``"bce"``, for data in ``[0, 1]``).

    Attributes
    ----------
    latent_grid_size : tuple of int
        Spatial ``(H, W)`` of the discrete latent field, derived from
        ``sample_size`` and ``num_downsample_layers``.

    Notes
    -----
    Reference: van den Oord, Vinyals, and Kavukcuoglu, *"Neural Discrete
    Representation Learning"*, NeurIPS, 2017 (arXiv:1711.00937).

    The paper reports a single image architecture rather than a table of
    sized variants, so this family exposes one nominal factory rather than
    ``_small`` / ``_base`` / ``_large`` siblings.  Scale it by overriding
    the fields above at ``create_model`` time.

    The paper's Appendix A.1 also describes an exponential-moving-average
    alternative to the codebook loss.  That variant changes how the
    codebook is *updated* rather than what the network computes, so it
    belongs to a training loop rather than to the architecture, and is not
    modelled here.

    Examples
    --------
    >>> from lucid.models.generative.vqvae import VQVAEConfig
    >>> cfg = VQVAEConfig(sample_size=32)
    >>> cfg.latent_grid_size
    (8, 8)
    >>> cfg.num_embeddings, cfg.embedding_dim
    (512, 256)
    """

    model_type: ClassVar[str] = "vqvae"

    act_fn: Literal["silu", "swish", "relu", "gelu"] = "relu"

    num_embeddings: int = 512
    embedding_dim: int = 256

    hidden_channels: int = 256
    num_downsample_layers: int = 2
    num_residual_layers: int = 2
    residual_hidden_channels: int = 256

    commitment_cost: float = 0.25
    recon_loss: Literal["mse", "bce"] = "mse"

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        check_sample_space_channels(
            self.in_channels, self.out_channels, "the reconstruction"
        )
        if self.num_embeddings <= 0:
            raise ValueError(
                f"num_embeddings must be positive, got {self.num_embeddings}"
            )
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}"
            )
        if self.hidden_channels <= 0:
            raise ValueError(
                f"hidden_channels must be positive, got {self.hidden_channels}"
            )
        if self.num_downsample_layers <= 0:
            raise ValueError(
                "num_downsample_layers must be at least 1, got "
                f"{self.num_downsample_layers}"
            )
        if self.num_residual_layers < 0:
            raise ValueError(
                "num_residual_layers must be non-negative, got "
                f"{self.num_residual_layers}"
            )
        if self.residual_hidden_channels <= 0:
            raise ValueError(
                "residual_hidden_channels must be positive, got "
                f"{self.residual_hidden_channels}"
            )
        if self.commitment_cost < 0.0:
            raise ValueError(
                f"commitment_cost must be non-negative, got {self.commitment_cost}"
            )

        # A stride-2 stack that does not evenly divide the input silently
        # changes the reconstruction's shape, so reject it at config time
        # rather than at the decoder's output.
        factor = 2**self.num_downsample_layers
        h, w = self._spatial_size()
        if h % factor or w % factor:
            raise ValueError(
                f"sample_size {self.sample_size} must be divisible by "
                f"2 ** num_downsample_layers ({factor}) in both dimensions"
            )

    def _spatial_size(self) -> tuple[int, int]:
        """Return ``sample_size`` normalised to an explicit ``(H, W)``."""
        if isinstance(self.sample_size, tuple):
            return self.sample_size
        return int(self.sample_size), int(self.sample_size)

    @property
    def latent_grid_size(self) -> tuple[int, int]:
        """Spatial ``(H, W)`` of the discrete latent field."""
        factor = 2**self.num_downsample_layers
        h, w = self._spatial_size()
        return h // factor, w // factor
