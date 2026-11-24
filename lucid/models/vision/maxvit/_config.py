"""MaxViT configuration (Tu et al., 2022)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_divisible, check_positive, check_stage_lengths


@model_family_meta(
    canonical_name="MaxViT",
    citation=(
        'Tu, Zhengzhong, et al. "MaxViT: Multi-Axis Vision '
        'Transformer." Proceedings of the European Conference on '
        "Computer Vision, 2022, pp. 459-479."
    ),
    theory=r"""
    MaxViT combines a convolutional MBConv stage with two complementary
    sparse self-attention mechanisms — *block attention* and *grid
    attention* — to obtain global receptive fields at linear complexity
    in the number of tokens.  The backbone is a four-stage pyramid, and
    each stage stacks repeats of the MaxViT block consisting of a
    MobileNetV3-style MBConv followed by block and then grid attention.

    Given a feature map of shape :math:`(H, W, C)`, *block attention*
    partitions it into non-overlapping :math:`P \times P` windows
    (default :math:`P = 7`) and runs full self-attention *within* each
    window — identical to Swin's W-MSA.  *Grid attention* takes the
    same feature map and reshapes it into a *dilated* :math:`P \times P`
    grid: tokens that share the same intra-window position across all
    windows are gathered into a single attention group.  Formally, if
    we tile the spatial dimensions as :math:`H = h \cdot P`,
    :math:`W = w \cdot P`, block attention groups along the
    :math:`(h, w)` axis and grid attention groups along the
    :math:`(P, P)` axis:

    .. math::

        \mathrm{Block}: (H, W) \to (h \cdot w,\ P \cdot P),\quad
        \mathrm{Grid} : (H, W) \to (P \cdot P,\ h \cdot w).

    Together the two operations realise dense local plus dense global
    self-attention, but each individual attention call has
    :math:`\mathcal{O}(P^2)` cost per token, giving overall
    :math:`\mathcal{O}(HW)` complexity.  Variants Tiny / Small / Base /
    Large simply scale stage depths and channel widths along this
    shared multi-axis attention design.
    """,
)
@dataclass(frozen=True)
class MaxViTConfig(ModelConfig):
    r"""Configuration dataclass for every MaxViT variant.

    ``MaxViTConfig`` is an immutable container that fully specifies the
    architecture of a MaxViT (Tu et al., 2022).  MaxViT combines an
    MBConv convolutional stage with two complementary sparse
    self-attention mechanisms — *block attention* and *grid attention*
    — in every MaxViT block, realising dense local + dense global
    self-attention at linear complexity in the number of tokens.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes for the classification head.  Defaults
        to ``1000`` (ImageNet-1k).
    in_channels : int, optional
        Number of input image channels.  Defaults to ``3`` (RGB).
    depths : tuple of int, optional
        Number of MaxViT blocks in each of the four stages.  Defaults
        to ``(2, 2, 5, 2)`` (MaxViT-Tiny).
    dims : tuple of int, optional
        Output channel width of each of the four stages.  Defaults to
        ``(64, 128, 256, 512)`` (MaxViT-Tiny).
    window_size : int, optional
        Side length :math:`P` of the partition window for block and
        grid attention.  Defaults to ``7``.
    head_dim : int, optional
        Size of each attention head.  Table 11 fixes this at 32 and
        derives the head count per stage as ``dim // head_dim``, so this
        is the knob that actually sets attention granularity.  Defaults
        to ``32``.
    mlp_ratio : float, optional
        MLP expansion ratio inside the partition-attention blocks.
        Defaults to ``4.0``.
    stem_width : int or None, optional
        Output width of the convolutional stem.  The reference MaxViT
        decouples this from ``dims[0]``: Tiny/Large keep ``stem_width ==
        dims[0]`` while Small/Base use a narrower ``stem_width = 64`` and
        let the first block's shortcut expand to ``dims[0]``.  ``None``
        (the default) falls back to ``dims[0]``.

    Attributes
    ----------
    model_type : ClassVar[str]
        Constant string ``"maxvit"`` used by the model registry.

    Notes
    -----
    The canonical variants registered as factory functions in
    :mod:`lucid.models.vision.maxvit` are:

    ================ ================== ====================
    Variant          depths             dims
    ================ ================== ====================
    MaxViT-Tiny      (2, 2, 5, 2)       (64, 128, 256, 512)
    MaxViT-Small     (2, 2, 5, 2)       (96, 192, 384, 768)
    MaxViT-Base      (2, 6, 14, 2)      (96, 192, 384, 768)
    MaxViT-Large     (2, 6, 14, 2)      (128, 256, 512, 1024)
    MaxViT-XLarge    (2, 6, 14, 2)      (192, 384, 768, 1536)
    ================ ================== ====================

    Reference: Zhengzhong Tu *et al.*, *"MaxViT: Multi-Axis Vision
    Transformer"*, ECCV 2022,
    `arXiv:2204.01697 <https://arxiv.org/abs/2204.01697>`_.

    Examples
    --------
    Build a MaxViT-Tiny configuration with a 100-class head:

    >>> from lucid.models.vision.maxvit import MaxViTConfig
    >>> cfg = MaxViTConfig(num_classes=100)
    >>> cfg.depths, cfg.dims, cfg.window_size, cfg.num_classes
    ((2, 2, 5, 2), (64, 128, 256, 512), 7, 100)
    """

    model_type: ClassVar[str] = "maxvit"

    num_classes: int = 1000
    in_channels: int = 3
    depths: tuple[int, ...] = (2, 2, 5, 2)
    dims: tuple[int, ...] = (64, 128, 256, 512)
    window_size: int = 7
    # Table 11 fixes the head *size* at 32 and derives the head count per
    # stage as ``dim // head_dim``.  This field was named ``num_heads`` and
    # then ignored entirely — the builder hardcoded the same 32 — so setting
    # it changed nothing.  Renamed to what it actually is, and now wired in.
    head_dim: int = 32
    mlp_ratio: float = 4.0
    stem_width: int | None = None

    # Stochastic depth.  Table 12 makes this the primary per-variant
    # regulariser — 0.2 / 0.3 / 0.4 / 0.6 for MaxViT-T/S/B/L on ImageNet-1k —
    # and Appendix 0.B.1 says the variants differ mainly in this rate.  The
    # default stays 0.0 so eval and weight transfer are untouched; pass the
    # paper's value to reproduce the training recipe.
    drop_path_rate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "dims", tuple(self.dims))
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(
            owner,
            dims=self.dims,
            head_dim=self.head_dim,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
        )
        check_stage_lengths(owner, depths=self.depths, dims=self.dims)
        # Heads are ``dim // head_dim`` per stage, so each stage width must be a
        # whole number of heads.
        for stage, dim in enumerate(self.dims):
            check_divisible(owner, f"dims[{stage}]", dim, "head_dim", self.head_dim)
