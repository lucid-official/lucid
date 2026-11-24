"""EfficientFormer configuration (Li et al., 2022)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive, check_stage_lengths


@model_family_meta(
    canonical_name="EfficientFormer",
    citation=(
        'Li, Yanyu, et al. "EfficientFormer: Vision Transformers at '
        'MobileNet Speed." Advances in Neural Information Processing '
        "Systems, vol. 35, 2022."
    ),
    theory=r"""
    EfficientFormer is a *mobile-grade* vision backbone designed so
    that its end-to-end on-device latency, rather than its FLOP count,
    matches the latency of MobileNet on the same hardware while
    retaining transformer-level accuracy.  The authors profile real
    iOS / NPU latency and identify three transformer operators that
    are unexpectedly slow: the reshape between spatial and token
    layouts, GeLU compared to ReLU, and LayerNorm compared to
    BatchNorm.  EfficientFormer is the result of a *latency-aware*
    architecture search that designs around these costs.

    The backbone is a four-stage pyramid.  Stages 1-3 use *pooling
    mixer* blocks — a MetaFormer-style token mixer that replaces the
    quadratic self-attention with a parameter-free :math:`3 \times 3`
    average pooling — and operate entirely in the 4-D
    :math:`(B, C, H, W)` layout so no expensive reshape is needed.
    Each pooling block computes

    .. math::

        x \leftarrow x + \gamma_1 \odot \bigl(\mathrm{AvgPool}_{3 \times 3}(x) - x\bigr),
        \qquad
        x \leftarrow x + \gamma_2 \odot \mathrm{MLP}(x),

    with :math:`\gamma_1, \gamma_2` initialised to :math:`10^{-5}` —
    the *layer scale* trick from CaiT.  Stage 4 is the only stage that
    pays the cost of a reshape and runs standard multi-head
    self-attention with relative position bias on the now-tiny token
    grid.  The resulting design (variants L1 / L3 / L7) matches
    ResNet-50 / DeiT-S accuracy at MobileNetV2-class latency on an
    iPhone 12 NPU.
    """,
)
@dataclass(frozen=True)
class EfficientFormerConfig(ModelConfig):
    r"""Configuration dataclass for every EfficientFormer variant.

    ``EfficientFormerConfig`` is an immutable container that fully
    specifies the architecture of an EfficientFormer (Li et al., 2022).
    The model is a *mobile-grade* vision backbone whose on-device
    latency, rather than its FLOP count, matches MobileNet on the same
    hardware while retaining transformer-level accuracy.

    Stages 1-3 use *pooling mixer* MetaFormer blocks (a parameter-free
    :math:`3 \times 3` average pool replacing self-attention) and
    operate entirely in :math:`(B, C, H, W)` layout.  Stage 4 is the
    only stage that pays the cost of a reshape and runs standard
    multi-head self-attention on the now-tiny token grid.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes for the classification head.  Defaults
        to ``1000`` (ImageNet-1k).
    in_channels : int, optional
        Number of input image channels.  Defaults to ``3`` (RGB).
    depths : tuple of int, optional
        Number of blocks per stage.  Defaults to ``(3, 2, 6, 4)``
        (EfficientFormer-L1).
    embed_dims : tuple of int, optional
        Output channel width per stage.  Defaults to
        ``(48, 96, 224, 448)`` (EfficientFormer-L1).
    mlp_ratios : tuple of float, optional
        Per-stage MLP expansion ratio.  Defaults to
        ``(4.0, 4.0, 4.0, 4.0)``.
    num_vit : int, optional
        Number of trailing blocks in the *last* stage that run 3-D
        multi-head self-attention (the remainder run the 4-D pooling
        token mixer).  Defaults to ``1`` (EfficientFormer-L1); L3 uses
        ``4`` and L7 uses ``8``.
    pool_size : int, optional
        Spatial extent of the average-pool token mixer in the 4-D
        pooling blocks.  Defaults to ``3``.
    key_dim : int, optional
        Per-head query/key dimension in the attention token mixer.
        Defaults to ``32`` (fixed across all paper variants).
    num_heads : int, optional
        Number of attention heads in the attention token mixer.
        Defaults to ``8`` (fixed across all paper variants).
    attn_ratio : float, optional
        Value/key dimension ratio in the attention token mixer; the
        per-head value dimension is ``attn_ratio * key_dim``.  Defaults
        to ``4.0``.
    resolution : int, optional
        Side length of the (square) token grid feeding the attention
        stage; the learned attention-bias table has
        ``resolution ** 2`` entries per head.  Defaults to ``7``
        (a ``7 x 7`` grid at the default ``224`` input).

        This is **coupled to the input size**: the trunk downsamples by
        32, so ``resolution`` must equal ``image_size / 32``.  It is not
        derived automatically because the bias table's shape — and hence
        the checkpoint's — depends on it; feeding a different input size
        without changing it raises a ``ValueError`` naming both.
    drop_path_rate : float, optional
        Maximum stochastic-depth rate; linearly scheduled across all
        blocks of the trunk.  The paper uses ``0.0`` for L1, ``0.1``
        for L3, ``0.2`` for L7.  Defaults to ``0.0``.
    layer_scale_init : float, optional
        Initial value of the layer-scale parameter on every residual
        branch.  Defaults to ``1e-5`` (Li et al., 2022, appendix).

    Attributes
    ----------
    model_type : ClassVar[str]
        Constant string ``"efficientformer"`` used by the model registry.

    Notes
    -----
    The canonical variants registered as factory functions in
    :mod:`lucid.models.vision.efficientformer` are:

    ============================ ================ ===================
    Variant                      depths           embed_dims
    ============================ ================ ===================
    EfficientFormer-L1           (3, 2, 6, 4)     (48, 96, 224, 448)
    EfficientFormer-L3           (4, 4, 12, 6)    (64, 128, 320, 512)
    EfficientFormer-L7           (6, 6, 18, 8)    (96, 192, 384, 768)
    ============================ ================ ===================

    Reference: Yanyu Li *et al.*, *"EfficientFormer: Vision
    Transformers at MobileNet Speed"*, NeurIPS 2022,
    `arXiv:2206.01191 <https://arxiv.org/abs/2206.01191>`_.

    Examples
    --------
    Build an EfficientFormer-L1 configuration with a 100-class head:

    >>> from lucid.models.vision.efficientformer import EfficientFormerConfig
    >>> cfg = EfficientFormerConfig(num_classes=100)
    >>> cfg.depths, cfg.embed_dims, cfg.num_classes
    ((3, 2, 6, 4), (48, 96, 224, 448), 100)
    """

    model_type: ClassVar[str] = "efficientformer"

    num_classes: int = 1000
    in_channels: int = 3
    depths: tuple[int, ...] = (3, 2, 6, 4)
    embed_dims: tuple[int, ...] = (48, 96, 224, 448)
    mlp_ratios: tuple[float, ...] = (4.0, 4.0, 4.0, 4.0)
    # Last-stage attention layout / hyperparameters (fixed across L1/L3/L7
    # except num_vit, which is 1 / 4 / 8 respectively).
    num_vit: int = 1
    pool_size: int = 3
    key_dim: int = 32
    num_heads: int = 8
    attn_ratio: float = 4.0
    resolution: int = 7
    # Regularization knobs (paper §4.1):
    #   drop_path_rate    — max stochastic depth rate (linear schedule across trunk).
    #                       Paper uses 0.0 for L1, 0.1 for L3, 0.2 for L7.
    #   layer_scale_init  — γ initialization for LayerScale on every residual
    #                       branch (paper appendix: 1e-5).
    drop_path_rate: float = 0.0
    layer_scale_init: float = 1e-5

    def __post_init__(self) -> None:
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "embed_dims", tuple(self.embed_dims))
        object.__setattr__(self, "mlp_ratios", tuple(self.mlp_ratios))
        check_stage_lengths(
            type(self).__name__,
            depths=self.depths,
            embed_dims=self.embed_dims,
            mlp_ratios=self.mlp_ratios,
        )
        check_positive(self, "num_classes", "in_channels", "embed_dims")
