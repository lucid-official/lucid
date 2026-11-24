"""CoAtNet configuration dataclass (Dai et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_divisible, check_positive, check_stage_lengths


@model_family_meta(
    canonical_name="CoAtNet",
    citation=(
        'Dai, Zihang, et al. "CoAtNet: Marrying Convolution and '
        'Attention for All Data Sizes." Advances in Neural Information '
        "Processing Systems, vol. 34, 2021."
    ),
    theory=r"""
    CoAtNet hybridises depthwise convolution and relative self-attention
    in a single backbone, motivated by the observation that
    convolutions excel at *generalization* (strong local inductive
    bias) while attention excels at *capacity* (data-dependent global
    mixing).  The network is a four-stage pyramid preceded by a
    two-layer convolutional stem; the early stages use *MBConv* blocks
    (squeeze-and-excitation, expansion ratio 4) and the later stages
    use *relative-attention* transformer blocks operating on
    :math:`\sqrt{HW}`-length token sequences.

    The key building block of the transformer stages is *relative*
    self-attention:

    .. math::

        \mathrm{Attn}(Q, K, V)_{ij} = \mathrm{softmax}\!\left(
            \frac{Q_i K_j^\top}{\sqrt{d}} + r_{i - j}
        \right) V_j,

    where :math:`r_{i-j}` is a learned bias indexed by the *relative*
    spatial offset between tokens.  This recovers the translation
    equivariance that convolutions provide for free while still
    permitting global, data-dependent mixing.  CoAtNet stacks two
    MBConv stages (C-stages) followed by two transformer stages
    (T-stages) so that strong local features are extracted first and
    long-range dependencies are modelled at lower resolutions where
    attention is cheap.  The CoAtNet-0 through CoAtNet-7 variants
    simply scale stage depths and channel widths along this skeleton.
    """,
)
@dataclass(frozen=True)
class CoAtNetConfig(ModelConfig):
    r"""Configuration dataclass for every CoAtNet variant.

    ``CoAtNetConfig`` is an immutable container that fully specifies
    the architecture of a CoAtNet (Dai et al., 2021).  CoAtNet uses
    *four stages* after a two-conv stem, with the first two stages
    being depthwise-convolutional (MBConv) and the last two stages
    being relative-attention transformer blocks.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes for the classification head.  Defaults
        to ``1000`` (ImageNet-1k).
    in_channels : int, optional
        Number of input image channels.  Defaults to ``3`` (RGB).
    image_size : int, optional
        Spatial side length of the square input image (used to build
        the fixed-size relative-position bias tables in T-stages).
        Defaults to ``224``.
    variant : str, optional
        Informational label identifying the canonical CoAtNet variant
        (``"coatnet_0"`` through ``"coatnet_7"``).  Defaults to
        ``"coatnet_0"``.
    blocks_per_stage : tuple of int, optional
        Number of blocks in each of the four stages
        :math:`(S_1, S_2, S_3, S_4)`.  Defaults to ``(2, 3, 5, 2)``
        (CoAtNet-0).
    dims : tuple of int, optional
        Channel widths for each of the four stages.  Defaults to
        ``(96, 192, 384, 768)`` (CoAtNet-0).
    stem_width : int, optional
        Output channels of the two-layer convolutional stem (before
        :math:`S_1`).  Defaults to ``64``.
    attn_heads : tuple of int, optional
        Number of attention heads for each of the two T-stages
        :math:`(S_3, S_4)`.  Defaults to ``(12, 24)``, derived from
        ``dim / 32`` with head dim 32.
    mbconv_expand : int, optional
        Expansion ratio of the inverted bottleneck inside MBConv blocks
        (``mid = out_ch * expand``).  Defaults to ``4``.
    head_hidden_size : int or None, optional
        If not ``None``, inserts an extra pre-classifier linear layer
        of the given width with a tanh activation between the pooled
        feature and the final classifier.  Defaults to ``768``
        (matches the reference recipe).
    dropout : float, optional
        Dropout probability inside the classifier head.  Defaults to
        ``0.0``.

    Attributes
    ----------
    model_type : ClassVar[str]
        Constant string ``"coatnet"`` used by the model registry.

    Notes
    -----
    The canonical CoAtNet-0 (registered as :func:`coatnet_0`) has
    approximately **25.6M parameters** and reaches **81.6% top-1 on
    ImageNet-1k** at 224x224 (Dai et al., 2021, Table 5).

    Reference: Zihang Dai *et al.*, *"CoAtNet: Marrying Convolution
    and Attention for All Data Sizes"*, NeurIPS 2021,
    `arXiv:2106.04803 <https://arxiv.org/abs/2106.04803>`_.

    Examples
    --------
    Build a CoAtNet-0 configuration with a 10-class head for CIFAR-10:

    >>> from lucid.models.vision.coatnet import CoAtNetConfig
    >>> cfg = CoAtNetConfig(num_classes=10)
    >>> cfg.dims, cfg.blocks_per_stage, cfg.num_classes
    ((96, 192, 384, 768), (2, 3, 5, 2), 10)
    """

    model_type: ClassVar[str] = "coatnet"

    num_classes: int = 1000
    in_channels: int = 3
    image_size: int = 224
    variant: str = "coatnet_0"

    # blocks_per_stage: (S1, S2, S3, S4) — 4 stages
    blocks_per_stage: tuple[int, ...] = (2, 3, 5, 2)

    # Channel dims for each of the 4 stages (S1 through S4)
    dims: tuple[int, ...] = (96, 192, 384, 768)

    # Stem output channels (before S1)
    stem_width: int = 64

    # Attention heads per T-stage (S3 head, S4 head); derived from dim/dim_head=32
    # S3 dim=384 → 384//32=12 heads; S4 dim=768 → 768//32=24 heads
    attn_heads: tuple[int, ...] = (12, 24)

    # MBConv expansion ratio (expand_output style: mid = out_ch * expand)
    mbconv_expand: int = 4

    # Optional pre-classifier hidden linear layer (timm head_hidden_size=768)
    head_hidden_size: int | None = 768

    dropout: float = 0.0

    # CoAtNet-6 / CoAtNet-7 mixed-S3 spec (Dai et al., 2021, §A.2 / Table 12):
    # When ``None`` (default for variants 0-5) the third stage is a uniform
    # transformer stage of ``blocks_per_stage[2]`` blocks at width ``dims[2]``.
    # When set to ``(L_mb, L_attn, D_attn)`` the third stage becomes a
    # *mixed-block* stage: first ``L_mb`` MBConv blocks at width ``dims[2]``
    # (entering S3 doubles channel from ``dims[1]``), then a 1×1 channel-
    # expansion conv from ``dims[2]`` to ``D_attn`` (no spatial change), then
    # ``L_attn`` relative-attention transformer blocks at width ``D_attn``.
    # S4 then enters from ``D_attn`` (not ``dims[2]``) and expands to
    # ``dims[3]`` as usual.  The paper rationale: "we move 2/3 of the MBConv
    # blocks of S2 into S3 and double its hidden dimension" — keeps FLOPs
    # comparable to a deeper homogeneous backbone while halving the memory
    # footprint of the deep transformer stack.
    mixed_s3: tuple[int, int, int] | None = None
    # Table 10 lists stochastic depth for every variant (0.2/0.3/0.5/0.7
    # for CoAtNet-0..3 on ImageNet-1k).  Default 0.0 leaves eval and
    # weight transfer untouched; set it to reproduce the recipe.
    drop_path_rate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks_per_stage", tuple(self.blocks_per_stage))
        object.__setattr__(self, "dims", tuple(self.dims))
        object.__setattr__(self, "attn_heads", tuple(self.attn_heads))
        owner = type(self).__name__
        # A stage always builds its first block, so a ``0`` in ``blocks_per_stage``
        # would silently build one: the requested depth is refused instead.
        check_stage_lengths(owner, count=4, blocks_per_stage=self.blocks_per_stage, dims=self.dims)
        check_stage_lengths(owner, count=2, attn_heads=self.attn_heads)
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(
            owner,
            image_size=self.image_size,
            stem_width=self.stem_width,
            mbconv_expand=self.mbconv_expand,
            blocks_per_stage=self.blocks_per_stage,
            dims=self.dims,
            attn_heads=self.attn_heads,
        )
        # The third stage's attention runs at ``dims[2]`` (or the mixed-S3
        # transformer width) and the fourth at ``dims[3]``, each split across
        # its own head count.
        s3_width = self.dims[2]
        if self.mixed_s3 is not None:
            if len(self.mixed_s3) != 3:
                raise ValueError(
                    f"{owner}: mixed_s3 is (MBConv blocks, attention blocks, attention "
                    f"width), so it holds 3 entries; got {len(self.mixed_s3)}: {self.mixed_s3}"
                )
            check_positive(owner, mixed_s3=self.mixed_s3)
            s3_width = self.mixed_s3[2]
        check_divisible(
            owner, "attention width of stage 3", s3_width, "attn_heads[0]", self.attn_heads[0]
        )
        check_divisible(owner, "dims[3]", self.dims[3], "attn_heads[1]", self.attn_heads[1])
