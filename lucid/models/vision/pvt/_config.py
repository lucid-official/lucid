"""PVT (Pyramid Vision Transformer) configuration (Wang et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive, check_stage_divisible, check_stage_lengths


@model_family_meta(
    canonical_name="PVT-v2",
    citation=(
        'Wang, Wenhai, et al. "PVT v2: Improved Baselines with '
        'Pyramid Vision Transformer." Computational Visual Media, '
        "vol. 8, no. 3, 2022, pp. 415-424.  Builds on Wang, Wenhai, "
        'et al. "Pyramid Vision Transformer: A Versatile Backbone '
        'for Dense Prediction without Convolutions." Proceedings of '
        "the IEEE/CVF International Conference on Computer Vision, "
        "2021, pp. 568-578."
    ),
    theory=r"""
    The Pyramid Vision Transformer (PVT) introduces a hierarchical
    feature pyramid into a *pure-transformer* backbone so that it can
    serve as a drop-in replacement for ResNet-style backbones in dense
    prediction tasks such as detection and segmentation.  Plain ViT
    produces a single-scale, low-resolution feature map and scales
    quadratically with the input resolution; PVT addresses both
    problems with two design changes.

    First, the backbone has four stages, each beginning with a
    *patch embedding* convolution (overlapping :math:`7 \times 7` in
    PVT v2, non-overlapping in v1) of stride :math:`s_i \in
    \{4, 2, 2, 2\}`, producing the canonical
    :math:`(\tfrac{H}{4}, \tfrac{W}{4}) \to \dots \to
    (\tfrac{H}{32}, \tfrac{W}{32})` pyramid.  Second, each transformer
    block uses *Spatial-Reduction Attention* (SRA): before computing
    keys and values, the token sequence is reshaped back to a 2-D feature
    map and downsampled by a stride-:math:`R_i` convolution, then
    flattened again.  Concretely,

    .. math::

        K, V \leftarrow \mathrm{LN}\!\bigl(
            W_R \cdot \mathrm{Reshape}_{R_i \times R_i}(x)\bigr),
        \qquad
        \mathrm{SRA}(x) = \mathrm{softmax}\!\left(
            \frac{Q K^\top}{\sqrt{d}}\right) V,

    where :math:`R_i` decreases with depth (e.g. :math:`8, 4, 2, 1`).
    This cuts the per-stage attention cost from
    :math:`\mathcal{O}((HW)^2)` to :math:`\mathcal{O}((HW)^2 / R_i^2)`,
    enabling high-resolution inputs at reasonable cost.  Variants
    PVT-Tiny / Small / Medium / Large scale stage depths along this
    shared pyramidal-SRA design.
    """,
)
@dataclass(frozen=True)
class PVTConfig(ModelConfig):
    r"""Configuration dataclass for every PVT (v2) variant.

    ``PVTConfig`` is an immutable container that fully specifies the
    architecture of a Pyramid Vision Transformer v2 (Wang et al., 2021
    and 2022).  The implementation in :mod:`lucid.models.vision.pvt`
    builds PVT v2 by default: overlapping convolutional patch
    embedding at each stage, a depthwise-conv-augmented MLP, and the
    Spatial-Reduction Attention (SRA) from PVT v1.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes for the classification head.  Defaults
        to ``1000`` (ImageNet-1k).
    in_channels : int, optional
        Number of input image channels.  Defaults to ``3`` (RGB).
    variant : str, optional
        Informational label identifying the canonical variant
        (``"pvt_v2_b0"`` through ``"pvt_v2_b5"``).  Defaults to
        ``"pvt_tiny"``.
    embed_dims : tuple of int, optional
        Output channel width per stage (4 stages).  Defaults to
        ``(64, 128, 320, 512)`` — PVT v2-B1.
    depths : tuple of int, optional
        Number of transformer blocks per stage.  Defaults to
        ``(2, 2, 2, 2)``.
    num_heads : tuple of int, optional
        Number of attention heads per stage.  Defaults to
        ``(1, 2, 5, 8)``.
    sr_ratios : tuple of int, optional
        Spatial-reduction ratio :math:`R_i` per stage; ``1`` disables
        SR for the deepest stage.  Defaults to ``(8, 4, 2, 1)``.
    mlp_ratios : tuple of float, optional
        Per-stage MLP expansion ratio.  Defaults to
        ``(8.0, 8.0, 4.0, 4.0)`` — early stages widen more than late
        stages, following PVT v2-B1.

    Attributes
    ----------
    model_type : ClassVar[str]
        Constant string ``"pvt"`` used by the model registry.

    Notes
    -----
    Spatial-Reduction Attention (SRA) reduces the per-stage attention
    cost from :math:`\mathcal{O}((HW)^2)` to
    :math:`\mathcal{O}((HW)^2 / R_i^2)` by downsampling the K / V
    feature map by stride :math:`R_i` before computing attention.

    Reference (v1): Wenhai Wang *et al.*, *"Pyramid Vision Transformer:
    A Versatile Backbone for Dense Prediction without Convolutions"*,
    ICCV 2021, `arXiv:2102.12122 <https://arxiv.org/abs/2102.12122>`_.
    Reference (v2): Wenhai Wang *et al.*, *"PVT v2: Improved Baselines
    with Pyramid Vision Transformer"*, CVMJ 2022,
    `arXiv:2106.13797 <https://arxiv.org/abs/2106.13797>`_.

    Examples
    --------
    Build a PVT v2-B1 configuration with a 100-class head:

    >>> from lucid.models.vision.pvt import PVTConfig
    >>> cfg = PVTConfig(num_classes=100)
    >>> cfg.embed_dims, cfg.depths, cfg.sr_ratios, cfg.num_classes
    ((64, 128, 320, 512), (2, 2, 2, 2), (8, 4, 2, 1), 100)
    """

    model_type: ClassVar[str] = "pvt"

    num_classes: int = 1000
    # Stochastic depth.  The reference ramps the drop probability linearly
    # with the *global* block index, so deeper blocks are dropped more
    # often; a single flat rate is not the same regulariser.
    drop_path_rate: float = 0.0
    in_channels: int = 3
    variant: str = "pvt_tiny"
    embed_dims: tuple[int, ...] = (64, 128, 320, 512)
    depths: tuple[int, ...] = (2, 2, 2, 2)
    num_heads: tuple[int, ...] = (1, 2, 5, 8)
    sr_ratios: tuple[int, ...] = (8, 4, 2, 1)
    # Section 3.5's *linear* SRA: instead of a stride-R convolution, average-
    # pool every stage to a fixed P x P grid (P = 7) before the 1x1
    # projection, making attention linear in the token count rather than
    # quadratic.  Table 1 carries a full "B2-Li" column and Table 2 reports
    # it at 22.6M / 3.9 GFLOPs / 82.1 top-1, so it is a paper variant with
    # published numbers — there was simply no way to build it.
    linear_sra: bool = False
    linear_pool_size: int = 7
    # Per-stage MLP expansion ratios (PVT v2-B1: 8,8,4,4 — stages 3&4 use 4)
    mlp_ratios: tuple[float, ...] = (8.0, 8.0, 4.0, 4.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "embed_dims", tuple(self.embed_dims))
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "num_heads", tuple(self.num_heads))
        object.__setattr__(self, "sr_ratios", tuple(self.sr_ratios))
        object.__setattr__(self, "mlp_ratios", tuple(self.mlp_ratios))
        check_stage_lengths(
            type(self).__name__,
            embed_dims=self.embed_dims,
            depths=self.depths,
            num_heads=self.num_heads,
            sr_ratios=self.sr_ratios,
            mlp_ratios=self.mlp_ratios,
        )
        check_positive(self, "num_classes", "in_channels", "embed_dims", "num_heads", "mlp_ratios")
        check_stage_divisible(
            type(self).__name__, "embed_dims", self.embed_dims, "num_heads", self.num_heads
        )
