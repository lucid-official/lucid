"""DiT configuration — Peebles & Xie, ICCV 2023.

The diffusion model that stopped using a U-Net.  DDPM, NCSN and the
latent diffusion models that followed all denoise with a convolutional
U-Net; DiT replaces it with a plain Vision Transformer over latent
patches and finds that the substitution is not merely viable but better
behaved — FID falls monotonically with the backbone's Gflops, whether
those come from depth, width, or a smaller patch.

The paper's contribution is as much a negative result as a positive one.
Nothing about the U-Net's inductive bias turned out to be necessary; what
mattered was how the conditioning entered the network, and the answer —
adaLN-Zero — is a few lines rather than an architecture.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    DiffusionModelConfig,
    check_noise_prediction_channels,
    check_patch_grid,
    patch_grid_side,
)

# How the timestep and class label reach the blocks.  The paper compares
# all four (Section 3.2, Figure 5) and adopts the last for everything
# afterwards; the others are kept because the comparison is the paper's
# main architectural finding and a configuration that cannot express it
# cannot reproduce it.
#
#   * ``"adaln_zero"`` — regress shift, scale *and* a residual gate, with
#     the projection zero-initialised so each block starts as the
#     identity.  Best at every training budget, and cheapest.
#   * ``"adaln"``      — the same without the gate.
#   * ``"cross_attention"`` — conditioning as a length-two sequence
#     attended to by an extra layer.  Costs about 15% more Gflops.
#   * ``"in_context"`` — conditioning appended as two extra tokens.
DiTConditioning = Literal["adaln_zero", "adaln", "cross_attention", "in_context"]


@model_family_meta(
    canonical_name="DiT",
    citation=(
        "Peebles, William, and Saining Xie. "
        '"Scalable Diffusion Models with Transformers." '
        "Proceedings of the IEEE/CVF International Conference on Computer "
        "Vision, 2023, pp. 4195-4205."
    ),
    theory=r"""
    A diffusion model needs a network that maps a noised latent and a
    timestep to a prediction of the noise.  Nothing about that job
    requires convolutions, and DiT tests the claim directly: patchify the
    latent into a sequence, run a standard Vision Transformer over it,
    and decode back.

    .. math::

        z \in \mathbb{R}^{I \times I \times C}
        \;\longrightarrow\;
        T = (I / p)^2 \text{ tokens of width } d
        \;\longrightarrow\;
        p \times p \times 2C \text{ per token}

    The decoder emits **twice** the input channels because the model
    predicts a noise and a diagonal covariance, following ADM's
    parameterisation of :math:`\Sigma_\theta`.

    **Where the conditioning enters is the finding.**  The paper compares
    four ways of admitting the timestep and class label and they are not
    close.  In-context conditioning appends them as tokens; cross
    attention gives them their own attention layer at about 15% more
    compute; adaptive layer norm regresses a per-channel shift and scale
    from their sum; and adaLN-Zero adds a third regressed vector, a gate
    :math:`\alpha` applied immediately before each residual add:

    .. math::

        x \leftarrow x + \alpha \cdot \mathrm{sublayer}\bigl(
            (1 + \gamma)\,\mathrm{LN}(x) + \beta \bigr).

    Initialising the projection that produces :math:`\alpha` at zero makes
    every block the identity at step zero.  That single change roughly
    halves the FID of the in-context variant at 400K steps, and it is the
    cheapest of the four in Gflops.

    **Scale is the other finding.**  Across twelve models — four widths
    crossed with three patch sizes — FID correlates with the backbone's
    Gflops at :math:`-0.93`, and parameter count is *not* the predictor:
    holding parameters fixed while shrinking the patch improves FID
    substantially, because more tokens mean more compute.  Sampling
    compute does not substitute for it; a small model given eight times
    the sampling steps still loses to a large one.
    """,
)
@dataclass(frozen=True)
class DiTConfig(DiffusionModelConfig):
    r"""Configuration for the DiT family.

    Parameters
    ----------
    sample_size : int, default=32
        Side of the latent the transformer sees — ``32`` for the
        256-pixel models and ``64`` for the 512-pixel ones, both after
        the VAE's factor-of-eight downsample.
    in_channels : int, default=4
        Latent channels.
    out_channels : int, default=8
        What the decoder emits: ``2 * in_channels``, a noise and a
        diagonal covariance.  Set it equal to ``in_channels`` for a model
        that predicts noise alone and fixes the variance.
    patch_size : int, default=2
        Side of the square patch each token covers.  The number after the
        slash in the paper's names — ``XL/2`` is the XLarge backbone at
        patch 2.  Halving it quadruples the token count and *at least*
        quadruples Gflops, which is the axis the scaling study rides.
    hidden_size : int, default=1152
        Transformer width.
    depth : int, default=28
        Number of blocks.
    num_heads : int, default=16
        Attention heads per block.
    mlp_ratio : float, default=4.0
        Feed-forward expansion.
    frequency_embedding_size : int, default=256
        Width of the sinusoid the timestep is expanded to before the
        two-layer MLP that maps it to ``hidden_size``.  The reference
        implementation fixes this at 256 independently of the model's
        width, so it is a field rather than a reuse of ``hidden_size`` —
        tying the two would add ``(hidden_size - 256) * hidden_size``
        parameters to every variant and make published checkpoints
        unloadable.
    num_classes : int, default=1000
        Label vocabulary.  Index ``num_classes`` is the null embedding
        classifier-free guidance drops to.
    class_dropout : float, default=0.1
        Probability of replacing the label with that null embedding
        during training.
    conditioning : {"adaln_zero", "adaln", "cross_attention", "in_context"}, default="adaln_zero"
        Which of the paper's four designs to build.
    learn_sigma : bool, default=True
        Whether the decoder predicts the covariance alongside the noise.
        Kept explicit because it is what makes ``out_channels`` twice
        ``in_channels``, and a reader who changes one without the other
        gets a shape error rather than an explanation.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748).  Backbone
    configurations are Table 1; the diffusion hyperparameters are ADM's,
    which is a linear variance schedule over 1000 steps from 1e-4 to
    2e-2 — the defaults :class:`DiffusionModelConfig` already carries.

    Examples
    --------
    >>> from lucid.models.generative.dit import DiTConfig
    >>> config = DiTConfig()
    >>> config.depth, config.hidden_size, config.num_heads
    (28, 1152, 16)

    The token count is what the scaling study varies, and it comes from
    the patch size rather than the width:

    >>> DiTConfig(patch_size=2).num_patches, DiTConfig(patch_size=8).num_patches
    (256, 16)
    """

    model_type: ClassVar[str] = "dit"

    sample_size: int | tuple[int, int] = 32
    in_channels: int = 4
    out_channels: int = 8

    patch_size: int = 2
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    frequency_embedding_size: int = 256

    num_classes: int = 1000
    class_dropout: float = 0.1

    conditioning: DiTConditioning = "adaln_zero"
    learn_sigma: bool = True

    @property
    def num_patches(self) -> int:
        """Tokens the latent is cut into — ``(sample_size / patch_size)^2``."""
        return (patch_grid_side(self.sample_size) // self.patch_size) ** 2

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        check_patch_grid(
            self.sample_size, self.patch_size, self.hidden_size, self.num_heads
        )
        if self.frequency_embedding_size % 2 != 0:
            raise ValueError(
                f"frequency_embedding_size must be even — the sinusoid "
                f"splits it into cosine and sine, got "
                f"{self.frequency_embedding_size}"
            )
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")
        if not 0.0 <= self.class_dropout <= 1.0:
            raise ValueError(
                f"class_dropout is a probability, got {self.class_dropout}"
            )
        check_noise_prediction_channels(
            self.in_channels, self.out_channels, self.learn_sigma
        )
