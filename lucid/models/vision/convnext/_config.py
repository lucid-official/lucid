"""ConvNeXt configuration (Liu et al., 2022)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive, check_stage_lengths


@model_family_meta(
    canonical_name="ConvNeXt",
    citation=(
        'Liu, Zhuang, et al. "A ConvNet for the 2020s." Proceedings '
        "of the IEEE/CVF Conference on Computer Vision and Pattern "
        "Recognition, 2022, pp. 11976-11986."
    ),
    theory=r"""
    ConvNeXt is a *pure convolutional* architecture obtained by
    systematically *modernizing* a ResNet using design choices borrowed
    from the Swin Transformer.  Starting from a ResNet-50 the authors
    apply a one-at-a-time roadmap: change the stage compute ratio to
    :math:`(3, 3, 9, 3)`, swap the stem for a non-overlapping
    :math:`4 \times 4` patchify convolution, replace the bottleneck
    with a depthwise :math:`7 \times 7` convolution followed by an
    inverted bottleneck, move the activation and normalization to a
    *single* GELU + LayerNorm after the depthwise conv, and replace
    BatchNorm with LayerNorm.  The resulting block is

    .. math::

        \mathrm{Block}(x) = x + \gamma \odot \mathrm{Linear}_{4d \to d}\!\bigl(
            \mathrm{GELU}\!\bigl(\mathrm{Linear}_{d \to 4d}\!\bigl(
            \mathrm{LN}(\mathrm{DWConv}_{7 \times 7}(x))\bigr)\bigr)\bigr),

    where :math:`\gamma \in \mathbb{R}^d` is a *layer-scale* parameter
    initialized to :math:`10^{-6}` (Touvron et al., 2021).

    The pyramid follows the four-stage Swin layout — channel widths
    :math:`(C, 2C, 4C, 8C)` with downsampling between stages — and the
    canonical variants Tiny / Small / Base / Large / XLarge scale the
    stem width :math:`C` and the depth of stage 3 from :math:`9` to
    :math:`27`.  The paper's headline result is that the modernised
    ConvNet matches or exceeds Swin's ImageNet accuracy at the same
    FLOPs while remaining a simple all-convolutional design.
    """,
)
@dataclass(frozen=True)
class ConvNeXtConfig(ModelConfig):
    r"""Configuration dataclass for every ConvNeXt variant.

    ``ConvNeXtConfig`` is an immutable container that fully specifies
    the architecture of a ConvNeXt.  It is consumed by both
    :class:`ConvNeXt` (backbone) and
    :class:`ConvNeXtForImageClassification` (classifier).  All five
    canonical variants described in Liu et al. (2022) — Tiny / Small /
    Base / Large / XLarge — can be expressed by choosing different
    ``depths`` and ``dims``.

    Each ConvNeXt block applies a depthwise :math:`7 \times 7`
    convolution followed by a LayerNorm and an inverted-bottleneck MLP,
    with a learnable per-channel *layer scale* :math:`\gamma` on the
    residual branch:

    .. math::

        x \leftarrow x + \gamma \odot
        \mathrm{Linear}_{4d \to d}\!\bigl(
            \mathrm{GELU}\!\bigl(\mathrm{Linear}_{d \to 4d}\!\bigl(
            \mathrm{LN}(\mathrm{DWConv}_{7 \times 7}(x))\bigr)\bigr)\bigr).

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes for the classification head.  Defaults
        to ``1000`` (ImageNet-1k).
    in_channels : int, optional
        Number of input image channels.  Defaults to ``3`` (RGB).
    depths : tuple of int, optional
        Number of ConvNeXt blocks in each of the four stages.  Defaults
        to ``(3, 3, 9, 3)`` (ConvNeXt-T).
    dims : tuple of int, optional
        Output channel width of each of the four stages.  Defaults to
        ``(96, 192, 384, 768)`` (ConvNeXt-T).
    layer_scale_init : float, optional
        Initial value :math:`\gamma_0` of the per-channel layer-scale
        parameter applied to every residual branch.  Defaults to
        ``1e-6`` (Touvron et al., 2021).
    dropout : float, optional
        Maximum stochastic-depth rate; linearly scheduled across all
        blocks of the trunk (Liu et al., 2022, §3).  Defaults to
        ``0.0``.

    Attributes
    ----------
    model_type : ClassVar[str]
        Constant string ``"convnext"`` used by the model registry.

    Notes
    -----
    The canonical variants registered as factory functions in
    :mod:`lucid.models.vision.convnext` are:

    ================ ============= ===================================
    Variant          depths        dims
    ================ ============= ===================================
    ConvNeXt-T       (3, 3, 9, 3)  (96, 192, 384, 768)
    ConvNeXt-S       (3, 3, 27, 3) (96, 192, 384, 768)
    ConvNeXt-B       (3, 3, 27, 3) (128, 256, 512, 1024)
    ConvNeXt-L       (3, 3, 27, 3) (192, 384, 768, 1536)
    ConvNeXt-XL      (3, 3, 27, 3) (256, 512, 1024, 2048)
    ================ ============= ===================================

    Reference: Zhuang Liu *et al.*, *"A ConvNet for the 2020s"*,
    CVPR 2022, `arXiv:2201.03545 <https://arxiv.org/abs/2201.03545>`_.

    Examples
    --------
    Build a ConvNeXt-T configuration for CIFAR-10 (10 classes):

    >>> from lucid.models.vision.convnext import ConvNeXtConfig
    >>> cfg = ConvNeXtConfig(num_classes=10)
    >>> cfg.depths, cfg.dims, cfg.num_classes
    ((3, 3, 9, 3), (96, 192, 384, 768), 10)
    """

    model_type: ClassVar[str] = "convnext"

    num_classes: int = 1000
    in_channels: int = 3
    depths: tuple[int, ...] = (3, 3, 9, 3)
    dims: tuple[int, ...] = (96, 192, 384, 768)
    layer_scale_init: float = 1e-6
    # Every LayerNorm in the trunk uses ``eps=1e-6`` per the reference
    # ConvNeXt recipe (matches Facebook AI Research's official repo + the
    # reference_vision implementation).  Lucid's default LayerNorm eps is
    # ``1e-5`` which is fine for from-scratch training but diverges
    # numerically from pretrained checkpoints; setting it here keeps the
    # surface-level model match exact when loading converted weights.
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "dims", tuple(self.dims))
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(owner, dims=self.dims)
        check_stage_lengths(owner, depths=self.depths, dims=self.dims)
