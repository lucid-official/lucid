"""InceptionNeXt configuration (Yu et al., 2023)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive, check_stage_lengths


@model_family_meta(
    canonical_name="InceptionNeXt",
    citation=(
        'Yu, Weihao, et al. "InceptionNeXt: When Inception Meets '
        'ConvNeXt." Proceedings of the IEEE/CVF Conference on '
        "Computer Vision and Pattern Recognition, 2024."
    ),
    theory=r"""
    InceptionNeXt is a drop-in replacement for the ConvNeXt block that
    eliminates the cost of a single large :math:`7 \times 7` depthwise
    convolution by *factorizing* it into four parallel, low-cost
    Inception-style branches.  The motivation is that large-kernel
    depthwise convs dominate ConvNeXt's wall-clock latency on modern
    accelerators despite their relatively small FLOP count: depthwise
    convolutions are memory-bandwidth bound and large kernels amplify
    that bottleneck.

    The *InceptionDWConv2d* operator splits the input channels into
    four groups and applies in parallel: (i) an identity branch,
    (ii) a small :math:`3 \times 3` square depthwise conv, (iii) a
    horizontal *band* depthwise conv of shape :math:`1 \times K`, and
    (iv) a vertical band depthwise conv of shape :math:`K \times 1`,
    with :math:`K = 11` by default.  The outputs are concatenated along
    the channel dimension:

    .. math::

        \mathrm{IDWConv}(x) = \mathrm{Concat}\bigl(
            x^{(1)},\;
            \mathrm{DW}_{3 \times 3}(x^{(2)}),\;
            \mathrm{DW}_{1 \times K}(x^{(3)}),\;
            \mathrm{DW}_{K \times 1}(x^{(4)})\bigr).

    The factorization preserves a large effective receptive field
    (the band convolutions span :math:`K` pixels along each axis) while
    cutting depthwise FLOPs and — more importantly — memory traffic.
    All other elements of the ConvNeXt block (LayerNorm, inverted
    bottleneck MLP, layer scale, residual) are kept unchanged, so the
    Tiny / Small / Base variants reuse the same depth and width tables
    as ConvNeXt while running noticeably faster at equal accuracy.
    """,
)
@dataclass(frozen=True)
class InceptionNeXtConfig(ModelConfig):
    r"""Configuration dataclass for every InceptionNeXt variant.

    ``InceptionNeXtConfig`` is an immutable container that fully
    specifies the architecture of an InceptionNeXt (Yu et al., 2024).
    InceptionNeXt is a drop-in replacement for the ConvNeXt block that
    factorizes the single large :math:`7 \times 7` depthwise
    convolution into four parallel Inception-style branches:

    .. math::

        \mathrm{IDWConv}(x) = \mathrm{Concat}\bigl(
            x^{(1)},\;
            \mathrm{DW}_{3 \times 3}(x^{(2)}),\;
            \mathrm{DW}_{1 \times K}(x^{(3)}),\;
            \mathrm{DW}_{K \times 1}(x^{(4)})\bigr).

    The remaining elements of the ConvNeXt block (BatchNorm, inverted-
    bottleneck Conv-MLP, layer scale, residual) are kept unchanged, so
    the Tiny / Small / Base variants reuse the same depth and width
    tables as ConvNeXt while running noticeably faster.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes for the classification head.  Defaults
        to ``1000`` (ImageNet-1k).
    in_channels : int, optional
        Number of input image channels.  Defaults to ``3`` (RGB).
    depths : tuple of int, optional
        Number of MetaNeXt blocks in each of the four stages.  Defaults
        to ``(3, 3, 9, 3)`` (InceptionNeXt-T).
    dims : tuple of int, optional
        Channel width per stage.  Defaults to ``(96, 192, 384, 768)``
        (InceptionNeXt-T).
    band_kernel : int, optional
        Kernel length :math:`K` of the horizontal :math:`1 \times K`
        and vertical :math:`K \times 1` *band* depthwise convolutions
        inside the Inception token mixer.  Defaults to ``11``.
    mlp_ratios : int or tuple of int, optional
        Per-stage MLP expansion ratios; a scalar is broadcast across all
        stages, as the reference does.  Defaults to ``(4, 4, 4, 3)``
        — the reference recipe uses ratio 3 in the final stage to
        match the head's compute budget.  A tuple whose length differs
        from ``len(depths)`` is rejected here rather than failing later
        inside the builder.

    Attributes
    ----------
    model_type : ClassVar[str]
        Constant string ``"inception_next"`` used by the model registry.

    Notes
    -----
    Reference: Weihao Yu *et al.*, *"InceptionNeXt: When Inception
    Meets ConvNeXt"*, CVPR 2024,
    `arXiv:2303.16900 <https://arxiv.org/abs/2303.16900>`_.

    Examples
    --------
    Build an InceptionNeXt-T configuration with a 100-class head:

    >>> from lucid.models.vision.inception_next import InceptionNeXtConfig
    >>> cfg = InceptionNeXtConfig(num_classes=100)
    >>> cfg.depths, cfg.dims, cfg.band_kernel, cfg.num_classes
    ((3, 3, 9, 3), (96, 192, 384, 768), 11, 100)
    """

    model_type: ClassVar[str] = "inception_next"

    num_classes: int = 1000
    in_channels: int = 3
    depths: tuple[int, ...] = (3, 3, 9, 3)
    dims: tuple[int, ...] = (96, 192, 384, 768)
    band_kernel: int = 11
    # Token-mixer geometry.  The reference exposes both as ``InceptionDWConv2d``
    # arguments; hard-wiring them made the paper's Atto variant (which uses a
    # different branch ratio) unbuildable through ``create_model`` overrides.
    square_kernel: int = 3
    branch_ratio: float = 0.125
    # Per-stage MLP expansion ratios.  When None, defaults to (4, 4, 4, 3) which
    # matches timm inception_next_tiny / small / base (final stage uses 3×).
    mlp_ratios: int | tuple[int, ...] = (4, 4, 4, 3)
    # Stochastic depth.  The reference ramps the drop probability linearly
    # with the *global* block index, so deeper blocks are dropped more
    # often; a single flat rate is not the same regulariser.
    drop_path_rate: float = 0.0
    # Head dropout, applied between the head's norm and its final
    # projection — the reference's ``MlpHead(drop=...)``.
    drop_rate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "dims", tuple(self.dims))
        owner = type(self).__name__
        check_stage_lengths(owner, depths=self.depths, dims=self.dims)
        # The reference broadcasts a scalar across the stages
        # (``if not isinstance(mlp_ratios, (list, tuple)): [mlp_ratios] * n``)
        # and always has one entry per stage.  Accepting only a tuple made
        # ``mlp_ratios=4`` a TypeError, and a short/long tuple failed later
        # with an IndexError from inside the builder instead of here.
        n_stages = len(self.depths)
        ratios: tuple[int, ...] = (
            (self.mlp_ratios,) * n_stages
            if isinstance(self.mlp_ratios, int)
            else tuple(self.mlp_ratios)
        )
        if len(ratios) != n_stages:
            raise ValueError(
                f"mlp_ratios must have one entry per stage: got "
                f"{len(ratios)} for {n_stages} stages (depths={self.depths})"
            )
        object.__setattr__(self, "mlp_ratios", ratios)
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(owner, dims=self.dims)
