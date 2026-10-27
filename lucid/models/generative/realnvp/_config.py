"""RealNVP configuration — Dinh, Sohl-Dickstein & Bengio, 2016.

One multi-scale template parametrised per dataset.  The paper (§4.1) varies
only three things across its five experiments: how many scales the
recursion runs for, how deep each coupling network is, and how wide it is.
"""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    FlowPrior,
    GenerativeActivation,
    NormalizingFlowConfig,
    check_sample_space_channels,
)


@model_family_meta(
    canonical_name="RealNVP",
    citation=(
        "Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. "
        '"Density Estimation Using Real NVP." International Conference '
        "on Learning Representations, 2017."
    ),
    theory=r"""
    RealNVP takes the coupling layer of NICE and makes it *scale* as well
    as shift, which is what finally let exact-likelihood flows model
    natural images.  An **affine coupling layer** splits the input with a
    binary mask :math:`b` and rescales the unmasked half by a learned,
    input-dependent factor:

    .. math::

        y = b \odot x + (1 - b) \odot
            \Big(x \odot \exp\big(s(b \odot x)\big) + t(b \odot x)\Big).

    The Jacobian is still triangular, so its log-determinant is just
    :math:`\sum (1 - b) \odot s` — no matter how deep :math:`s` and
    :math:`t` are.  Unlike NICE's additive coupling this map is no longer
    volume-preserving, so the flow can concentrate and dilate density
    where the data needs it rather than relying on one global diagonal
    scaling.

    Two masks exploit image structure.  The **checkerboard** mask is 1
    where the sum of the spatial coordinates is odd; the **channel-wise**
    mask is 1 on the first half of the channels.  Between the two, a
    **squeeze** trades space for depth — each :math:`2 \times 2 \times c`
    block becomes :math:`1 \times 1 \times 4c`, turning an
    :math:`s \times s \times c` tensor into
    :math:`\tfrac{s}{2} \times \tfrac{s}{2} \times 4c` — so that
    channel-wise masking after the squeeze partitions along a different
    axis than the checkerboard did before it.

    The third idea is the **multi-scale architecture**.  Each scale runs
    three alternating checkerboard couplings, squeezes, runs three
    alternating channel-wise couplings, then *factors out* half the
    dimensions:

    .. math::

        h^{(0)} = x, \qquad
        \big(z^{(i+1)}, h^{(i+1)}\big) = f^{(i+1)}\big(h^{(i)}\big),
        \qquad z = \big(z^{(1)}, \dots, z^{(L)}\big).

    Factored-out dimensions are modelled as Gaussians immediately and
    stop paying compute, so the network spends its depth on the
    coarse-scale structure that still needs it; the final scale applies
    four alternating checkerboard couplings and emits the rest.
    Coupling networks are residual convolutional stacks (ReLU, skip
    connections) whose :math:`s` output is a :math:`\tanh` scaled by a
    learned factor — bounded, so the exponential cannot explode early in
    training.

    Because pixel data is bounded and quantised, the model is fitted in
    logit space: :math:`x \mapsto \mathrm{logit}\big(\alpha + (1 - 2
    \alpha) x\big)` with :math:`\alpha = 0.05`, whose log-determinant is
    part of the reported likelihood.  RealNVP reaches 3.49 bits/dim on
    CIFAR-10 with exact inference *and* exact single-pass sampling —
    neither of which a VAE or an autoregressive model gives you — and its
    coupling/multi-scale recipe is inherited almost unchanged by Glow.
    """,
)
@dataclass(frozen=True)
class RealNVPConfig(NormalizingFlowConfig):
    """Configuration for every RealNVP variant.

    Defaults reproduce the paper's Imagenet ``32 x 32`` setup (§4.1):
    four scales, four residual blocks per coupling network, 32 feature
    maps at the first scale, and a standard normal prior.

    Args:
        num_scales: Levels of the multi-scale recursion ``L``.  Every
            level except the last squeezes once and factors out half of
            its dimensions, so the spatial size halves ``L - 1`` times.
            The paper recurses "until the input of the last recursion is
            ``4 x 4 x c``" — four scales for a ``32 x 32`` input — and
            uses a single downscale for CIFAR-10.
        residual_blocks: Residual blocks inside each coupling network.
            Paper §4.1: ``4`` at ``32 x 32``, ``2`` at ``64 x 64``, ``8``
            for CIFAR-10.
        base_dim: Feature maps in the first scale's coupling networks;
            doubled at every subsequent scale, as in the released
            implementation.  Paper §4.1: ``32``, or ``64`` for CIFAR-10.
        use_weight_norm: Apply weight normalisation to every convolution
            inside the coupling networks (paper §4.1).
        use_batch_norm: Enable batch normalisation — inside the coupling
            networks *and* on each coupling layer's output, where it is
            an invertible rescaling that contributes to the
            log-determinant (paper §3.7).  Inversion uses the running
            statistics, so ``decode`` is exact in ``eval()`` mode.
        batch_norm_running_stats_in_training: Normalise by the running
            average during training rather than by the current batch —
            the paper's "novel variant of batch normalization ... based
            on a running average over recent minibatches, and is thus
            more robust when training with very small minibatches"
            (§3.7 / Appendix E).  Off by default: the two agree in the
            large-batch limit where most training happens, and turning
            it on changes the behaviour of runs already in flight.
        data_constraint: Width of the logit squash applied before the
            flow, matching the released implementation's
            ``data_constraint``: ``x`` in ``[0, 1]`` is mapped into
            ``[(1 - c) / 2, (1 + c) / 2]`` before the logit, i.e.
            ``alpha = 0.05`` at the default ``0.9``.
        num_bits: Bit depth of the discrete data the reported likelihood
            is measured against.  The flow is fitted to ``x`` in
            ``[0, 1]``, but the paper models
            ``logit(alpha + (1 - alpha) * x / 256)`` and reports
            bits/dim on the 8-bit pixel scale, so the change of
            variables from ``{0, ..., 255}`` back to ``[0, 1]``
            contributes ``num_bits`` to the metric.  Table 1 is not
            comparable without it.

    Notes:
        ``out_channels`` must equal ``in_channels`` — a flow is a
        bijection on its sample space.  ``sample_size`` must be divisible
        by ``2 ** (num_scales - 1)`` so every squeeze has an even spatial
        extent to work on.

    Examples
    --------
    >>> from lucid.models.generative.realnvp._config import RealNVPConfig
    >>> cfg = RealNVPConfig()
    >>> cfg.model_type
    'realnvp'
    >>> cfg.in_channels
    3
    """

    model_type: ClassVar[str] = "realnvp"

    # Paper §4.1 — Imagenet 32x32 column.
    sample_size: int | tuple[int, int] = 32
    in_channels: int = 3
    out_channels: int = 3
    act_fn: GenerativeActivation = "relu"
    prior: FlowPrior = "gaussian"

    num_scales: int = 4
    residual_blocks: int = 4
    base_dim: int = 32
    use_batch_norm: bool = True
    # §3.7 / Appendix E: "a novel variant of batch normalization which is
    # based on a running average over recent minibatches, and is thus more
    # robust when training with very small minibatches."  Ordinary batch
    # norm normalises by the *current* batch during training; this variant
    # normalises by the running average instead.  Defaults off so existing
    # runs keep their behaviour, and because the two agree in the
    # large-batch limit where most training happens.
    batch_norm_running_stats_in_training: bool = False
    # §4.1: "we use ... weight normalization" in the coupling networks.
    # Without it the s/t stack is a plain conv tower and the reported
    # optimisation behaviour does not reproduce.
    use_weight_norm: bool = True
    data_constraint: float = 0.9
    # Bit depth of the discrete data the reported likelihood is measured
    # against.  The flow itself is fitted to ``x`` in ``[0, 1]``, but the
    # paper models ``logit(alpha + (1 - alpha) * x / 256)`` and reports
    # bits/dim on the 8-bit pixel scale, so the change of variables from
    # ``{0, ..., 255}`` to ``[0, 1]`` contributes ``num_bits`` to the metric.
    # Table 1 is not comparable without it.
    num_bits: int = 8

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        check_sample_space_channels(
            self.in_channels, self.out_channels, "a flow is a bijection: its image"
        )
        if self.num_scales <= 0:
            raise ValueError(f"num_scales must be positive, got {self.num_scales}")
        if self.residual_blocks <= 0:
            raise ValueError(f"residual_blocks must be positive, got {self.residual_blocks}")
        if self.base_dim <= 0:
            raise ValueError(f"base_dim must be positive, got {self.base_dim}")
        if not 0.0 < self.data_constraint < 1.0:
            raise ValueError(f"data_constraint must lie in (0, 1), got {self.data_constraint}")

        h, w = self.image_shape[1:]
        factor = 2 ** (self.num_scales - 1)
        if h % factor != 0 or w % factor != 0:
            raise ValueError(
                f"sample_size {self.sample_size} must be divisible by "
                f"2 ** (num_scales - 1) = {factor} — every scale but the last "
                f"squeezes the spatial extent in half"
            )
        if (h // factor) % 2 != 0 and self.num_scales > 1:
            raise ValueError(
                f"sample_size {self.sample_size} leaves an odd {h // factor}-px "
                f"final scale; the checkerboard mask needs an even extent"
            )

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """``(C, H, W)`` of the data space the bijection is defined on."""
        if isinstance(self.sample_size, tuple):
            h, w = self.sample_size
        else:
            h = w = int(self.sample_size)
        return self.in_channels, h, w

    @property
    def input_dim(self) -> int:
        """Flattened dimensionality ``D = C · H · W`` of the latent."""
        c, h, w = self.image_shape
        return c * h * w

    @property
    def scale_channels(self) -> tuple[int, ...]:
        """Channel count entering each scale.

        Squeezing multiplies channels by four and factoring out halves
        them, so scale ``i`` sees ``in_channels * 2 ** i`` channels.
        """
        return tuple(self.in_channels * 2**i for i in range(self.num_scales))
