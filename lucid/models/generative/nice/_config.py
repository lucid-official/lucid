"""NICE configuration — Dinh, Krueger & Bengio, 2014.

One architecture template parametrised by dataset.  The paper's Figure 3
fixes four coupling layers for every experiment and varies only the
coupling network's width / depth and the choice of prior, so a single
config with per-dataset defaults covers all four reported models
(MNIST / TFD / SVHN / CIFAR-10).
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import GenerativeActivation, NormalizingFlowConfig


@model_family_meta(
    canonical_name="NICE",
    citation=(
        'Dinh, Laurent, David Krueger, and Yoshua Bengio. "NICE: '
        'Non-linear Independent Components Estimation." International '
        "Conference on Learning Representations, Workshop Track, 2015."
    ),
    theory=r"""
    NICE is the first *normalizing flow* trained on images: instead of
    bounding the likelihood (VAE) or factorising it over pixels
    (autoregressive models), it learns an invertible map
    :math:`h = f(x)` onto a latent space with a factorised prior, so the
    exact data log-density falls straight out of the change-of-variables
    formula

    .. math::

        \log p_X(x) = \log p_H\big(f(x)\big)
            + \log\Big|\det \frac{\partial f}{\partial x}\Big|.

    Nothing is approximated — no variational gap, no Markov chain — but
    every layer now has to be invertible *and* have a cheap Jacobian
    determinant, which is a severe architectural constraint.

    The construction that satisfies both is the **additive coupling
    layer**.  Partition the dimensions into two blocks
    :math:`(x_{I_1}, x_{I_2})` and update only the second, conditioned on
    the first:

    .. math::

        \begin{aligned}
        y_{I_1} &= x_{I_1}, \\
        y_{I_2} &= x_{I_2} + m(x_{I_1}).
        \end{aligned}

    The Jacobian is unit lower-triangular, so its determinant is
    :math:`1` no matter how deep the coupling network :math:`m` is, and
    the inverse is the same graph with one sign flipped,
    :math:`x_{I_2} = y_{I_2} - m(y_{I_1})`.  Since one block passes
    through untouched, consecutive layers must exchange the roles of the
    two blocks; at least three couplings are required before every
    dimension can influence every other, and the paper stacks four with
    deep rectified networks (linear output units) as the :math:`m`'s.

    A stack of volume-preserving maps is itself volume-preserving, so
    NICE closes the flow with a learned **diagonal scaling**
    :math:`h = \exp(s) \odot h^{(4)}` — the only stage that moves the
    determinant:

    .. math::

        \log p_X(x) = \sum_{i=1}^{D}
            \Big[\log p_{H_i}\big(f_i(x)\big) + s_i\Big].

    The scales :math:`\exp(s_i)` behave as learned per-dimension
    importance weights: directions the model captures well are given
    small scale, making :math:`s` the non-linear analogue of an
    eigenvalue spectrum in PCA.  The prior is a standard logistic
    (heavier tails, used for MNIST / SVHN / CIFAR-10) or a standard
    normal (used for TFD).

    Generation is exact and parallel — draw :math:`h \sim p_H`, return
    :math:`f^{-1}(h)` — with no decoder and no iterative refinement.  The
    coupling layer introduced here is the direct ancestor of RealNVP
    (affine couplings, multi-scale spatial masks) and Glow (invertible
    1x1 convolutions); the samples look blurry by modern standards, but
    the recipe defined how exact-likelihood generative models are built.
    """,
)
@dataclass(frozen=True)
class NICEConfig(NormalizingFlowConfig):
    """Configuration for every NICE variant.

    Defaults reproduce the paper's MNIST experiment (Figure 3): 784
    dimensions, four coupling layers whose coupling networks have five
    hidden layers of 1000 rectified units, and a standard logistic prior.

    Args:
        input_dim: Dimensionality ``D`` of the data the flow is defined
            on — the paper's "# dimensions" row and the released
            configs' ``nvis``.  Figure 3: ``784`` (MNIST), ``2304``
            (TFD), ``3072`` (SVHN, CIFAR-10).  Must be even, since every
            coupling layer splits it in half.
        num_coupling_layers: Number of stacked additive coupling layers.
            The paper uses ``4`` for every dataset; three is the minimum
            that lets all dimensions interact.
        num_hidden_layers: Hidden layers inside each coupling network
            ``m``.  Paper Figure 3: ``5`` for MNIST, ``4`` for TFD /
            SVHN / CIFAR-10.
        hidden_dim: Units per hidden layer of ``m``.  Paper Figure 3:
            ``1000`` (MNIST), ``5000`` (TFD), ``2000`` (SVHN,
            CIFAR-10).
        input_reorder: One-off permutation applied before the first
            coupling layer.  ``"tile"`` (default) reproduces the paper's
            odd/even partition; ``"none"`` splits in raster order, which
            is what the released configs for SVHN / CIFAR-10 do.
        init_range: Half-width of the uniform weight initialisation of
            every coupling network, matching the official release's
            ``irange = 0.01``.  Biases start at zero, and the diagonal
            scaling starts at ``s = 0`` (identity).

    Notes:
        NICE is defined on flat ``D``-vectors, not on a pixel grid: both
        the paper and the released configs parametrise each experiment by
        a single dimension count.  ``input_dim`` is therefore the only
        dimensional knob, and the spatial fields inherited from
        :class:`GenerativeModelConfig` (``sample_size``, ``in_channels``,
        ``out_channels``) hold the equivalent vector encoding rather than
        an image shape the sources never state: ``sample_size`` is pinned
        to ``(1, input_dim)``, and channels other than 1 are refused.

    Examples
    --------
    >>> from lucid.models.generative.nice._config import NICEConfig
    >>> cfg = NICEConfig()
    >>> cfg.model_type
    'nice'
    >>> cfg.in_channels
    1
    """

    model_type: ClassVar[str] = "nice"

    # The flat-vector encoding: one channel in, one out.
    in_channels: int = 1
    out_channels: int = 1

    # Paper Figure 3, MNIST column.
    input_dim: int = 784
    act_fn: GenerativeActivation = "relu"

    num_coupling_layers: int = 4
    num_hidden_layers: int = 5
    hidden_dim: int = 1000
    # ``"tile"`` sends even indices to the first block and odd indices to the
    # second, turning each coupling layer's contiguous half-split into the
    # paper's odd/even partition.  ``"none"`` keeps raster order, so the
    # split becomes top-half / bottom-half of the image.
    input_reorder: Literal["tile", "none"] = "tile"
    init_range: float = 0.01

    @override
    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")

        # The channel counts are refused rather than pinned: a value other
        # than the flat encoding's was accepted and silently replaced
        # (LCD-631).  The "(1, D)" extent is still pinned — bookkeeping, not
        # a claim about how the D values are laid out as an image.
        if (self.in_channels, self.out_channels) != (1, 1):
            raise ValueError(
                "NICE is defined on flat D-vectors, so in_channels and "
                "out_channels must both be 1 — set input_dim instead; got "
                f"{self.in_channels} and {self.out_channels}"
            )
        object.__setattr__(self, "sample_size", (1, self.input_dim))

        super().__post_init__()
        if self.num_coupling_layers <= 0:
            raise ValueError(
                f"num_coupling_layers must be positive, got {self.num_coupling_layers}"
            )
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.init_range <= 0.0:
            raise ValueError(f"init_range must be positive, got {self.init_range}")
        if self.input_dim % 2 != 0:
            raise ValueError(
                f"input_dim must be even, got {self.input_dim} — every coupling "
                f"layer splits the sample in half"
            )

    @property
    def split(self) -> int:
        """Index at which every coupling layer partitions the ``D`` dims."""
        return self.input_dim // 2
