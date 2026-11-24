"""CvT configuration dataclass (Wu et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import (
    check_non_negative,
    check_positive,
    check_stage_divisible,
    check_stage_lengths,
)


@model_family_meta(
    canonical_name="CvT",
    citation=(
        'Wu, Haiping, et al. "CvT: Introducing Convolutions to Vision '
        'Transformers." Proceedings of the IEEE/CVF International '
        "Conference on Computer Vision, 2021, pp. 22-31."
    ),
    theory=r"""
    The Convolutional vision Transformer (CvT) reinjects the locality
    inductive bias of CNNs into ViT through two changes: (i) replacing
    the non-overlapping patch embedding by *overlapping convolutional
    token embedding*, and (ii) replacing the linear projections inside
    self-attention with *depthwise-separable convolutional projections*.
    CvT also drops positional embeddings entirely — locality is supplied
    implicitly by the convolutional projections.

    Convolutional token embedding is performed at the start of each
    of the three stages: a strided convolution
    :math:`\mathrm{Conv}_{k \times k, \mathrm{stride}=s}` produces a
    new token grid of reduced spatial resolution and increased channel
    width, yielding a Swin-like hierarchical pyramid.  Inside every
    block the queries / keys / values are computed with depthwise
    convolutions rather than linear maps:

    .. math::

        Q = \mathrm{Flatten}(\mathrm{DWConv}_q(x_{2D})),\quad
        K = \mathrm{Flatten}(\mathrm{DWConv}_k(x_{2D})),\quad
        V = \mathrm{Flatten}(\mathrm{DWConv}_v(x_{2D})),

    where :math:`x_{2D}` is the spatially reshaped token map.  When the
    key/value DWConv uses stride 2 the attention becomes *strided*,
    reducing :math:`N` by 4 inside attention without losing the
    full-resolution output for the next block.  Variants CvT-13 / 21 /
    24 scale depth and width along this three-stage convolutional
    transformer skeleton.
    """,
)
@dataclass(frozen=True)
class CvTConfig(ModelConfig):
    r"""Configuration dataclass for every CvT variant.

    ``CvTConfig`` is an immutable container that fully specifies the
    architecture of a Convolutional Vision Transformer (Wu et al.,
    2021).  CvT replaces plain ViT's non-overlapping patch embedding
    with *overlapping convolutional token embedding* at each of three
    stages, and replaces the linear Q/K/V projections inside every
    self-attention with *depthwise-separable convolutional projections*.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes for the classification head.  Defaults
        to ``1000`` (ImageNet-1k).
    in_channels : int, optional
        Number of input image channels.  Defaults to ``3`` (RGB).
    variant : str, optional
        Informational label identifying the canonical variant
        (``"cvt_13"``, ``"cvt_21"``, ``"cvt_w24"``).  Defaults to
        ``"cvt_13"``.
    dims : tuple of int, optional
        Per-stage hidden width (3 stages).  Defaults to
        ``(64, 192, 384)`` (CvT-13).
    depths : tuple of int, optional
        Number of CvT blocks in each of the three stages.  Defaults
        to ``(1, 2, 10)`` (CvT-13).
    num_heads : tuple of int, optional
        Number of attention heads per stage.  Defaults to ``(1, 3, 6)``
        (CvT-13).
    embed_strides : tuple of int, optional
        Stride of each stage's convolutional token embedding (the
        spatial downsampling factor of the stage).  Defaults to
        ``(4, 2, 2)`` — total downsampling 16x.
    mlp_ratio : float, optional
        Expansion ratio of the MLP inside each CvT block.  Defaults to
        ``4.0``.
    dropout : float, optional
        Dropout probability inside MLP blocks.  Defaults to ``0.0``.

    Attributes
    ----------
    model_type : ClassVar[str]
        Constant string ``"cvt"`` used by the model registry.

    Notes
    -----
    The canonical variants registered as factory functions in
    :mod:`lucid.models.vision.cvt` are:

    =========== ==================== ============== ===============
    Variant     dims                 depths         num_heads
    =========== ==================== ============== ===============
    CvT-13      (64, 192, 384)       (1, 2, 10)     (1, 3, 6)
    CvT-21      (64, 192, 384)       (1, 4, 16)     (1, 3, 6)
    CvT-W24     (192, 768, 1024)     (2, 2, 20)     (3, 12, 16)
    =========== ==================== ============== ===============

    Reference: Haiping Wu *et al.*, *"CvT: Introducing Convolutions to
    Vision Transformers"*, ICCV 2021,
    `arXiv:2103.15808 <https://arxiv.org/abs/2103.15808>`_.

    Examples
    --------
    Build a CvT-13 configuration with a 10-class head:

    >>> from lucid.models.vision.cvt import CvTConfig
    >>> cfg = CvTConfig(num_classes=10)
    >>> cfg.dims, cfg.depths, cfg.num_heads, cfg.num_classes
    ((64, 192, 384), (1, 2, 10), (1, 3, 6), 10)
    """

    model_type: ClassVar[str] = "cvt"

    num_classes: int = 1000
    in_channels: int = 3
    variant: str = "cvt_13"
    # Per-stage configuration (3 stages)
    dims: tuple[int, ...] = (64, 192, 384)
    depths: tuple[int, ...] = (1, 2, 10)
    num_heads: tuple[int, ...] = (1, 3, 6)
    embed_strides: tuple[int, ...] = (4, 2, 2)
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # Table 1: 0.1 for CvT-13/21, 0.3 for CvT-W24.  Stochastic depth is a
    # training-time regulariser, so it is invisible to eval parity -- which
    # is exactly why its absence went unnoticed.  Default 0.0 keeps every
    # existing config numerically identical; the factories set the
    # paper-cited rates.
    drop_path_rate: float = 0.0
    # Per-stage class-token flag.  The reference CvT prepends a single
    # learnable CLS token only on the *last* stage; that token gathers
    # global information through the stage's attention and is what the
    # classifier reads (``layernorm(cls).mean(1)``).  Stages without a
    # CLS token attend over patch tokens only.
    cls_token: tuple[bool, ...] = (False, False, True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", tuple(self.dims))
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "num_heads", tuple(self.num_heads))
        object.__setattr__(self, "embed_strides", tuple(self.embed_strides))
        object.__setattr__(self, "cls_token", tuple(self.cls_token))
        owner = type(self).__name__
        check_stage_lengths(
            owner,
            dims=self.dims,
            depths=self.depths,
            num_heads=self.num_heads,
            embed_strides=self.embed_strides,
        )
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(
            owner,
            dims=self.dims,
            num_heads=self.num_heads,
            embed_strides=self.embed_strides,
            mlp_ratio=self.mlp_ratio,
        )
        check_non_negative(owner, depths=self.depths)
        check_stage_divisible(owner, "dims", self.dims, "num_heads", self.num_heads)
