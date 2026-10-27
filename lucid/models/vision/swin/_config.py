"""Swin Transformer configuration (Liu et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_stage_lengths


@model_family_meta(
    canonical_name="Swin Transformer",
    citation=(
        'Liu, Ze, et al. "Swin Transformer: Hierarchical Vision '
        'Transformer using Shifted Windows." Proceedings of the '
        "IEEE/CVF International Conference on Computer Vision, 2021, "
        "pp. 10012-10022."
    ),
    theory=r"""
    The Swin Transformer reintroduces the *hierarchical* feature
    pyramid of classical CNNs into a pure-transformer backbone, making
    it a general-purpose vision model that scales linearly with image
    resolution rather than quadratically (as plain ViT does).  The
    image is first split into non-overlapping :math:`4 \times 4`
    patches and projected to :math:`C` channels.  Four stages then
    follow, each halving the spatial resolution via a *patch merging*
    layer that concatenates :math:`2 \times 2` neighbours and projects
    :math:`4C \to 2C`, doubling the channel width — giving the
    canonical :math:`C, 2C, 4C, 8C` pyramid.

    Within each stage the tokens flow through pairs of Swin blocks.
    The first block computes self-attention inside non-overlapping
    *windows* of size :math:`M \times M` (default :math:`M = 7`), and
    the second performs *shifted-window* (SW-MSA) attention where the
    grid is cyclically shifted by :math:`\lfloor M/2 \rfloor`.  Because
    each window is processed independently, the attention cost per
    image is :math:`\mathcal{O}(M^2 H W)` instead of
    :math:`\mathcal{O}((HW)^2)`, but the shift introduces
    cross-window connections so the receptive field still grows with
    depth.  A learnable *relative position bias*
    :math:`B \in \mathbb{R}^{M^2 \times M^2}` is added inside softmax:

    .. math::

        \mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(
            \frac{Q K^\top}{\sqrt{d}} + B
        \right) V.

    The four canonical variants (Tiny / Small / Base / Large) scale
    :math:`C` and per-stage block counts from Table 7 of the paper,
    trading parameters for accuracy while reusing the same
    shifted-window mechanism end to end.
    """,
)
@dataclass(frozen=True)
class SwinConfig(ModelConfig):
    r"""Configuration dataclass for every Swin Transformer variant.

    ``SwinConfig`` is an immutable container that fully specifies the
    architecture of a Swin Transformer.  It is consumed by both
    :class:`SwinTransformer` (backbone) and
    :class:`SwinTransformerForImageClassification` (classifier).  All
    canonical variants described in Liu et al. (2021) — Tiny / Small /
    Base / Large — can be expressed by choosing different per-stage
    depths, channel widths, and head counts.

    The backbone is a four-stage pyramid: the stem patch embedding maps
    :math:`(H, W, 3) \to (H/p, W/p, C)`, then each stage applies a stack
    of pairs of Swin blocks (window + shifted-window attention) and a
    patch-merging downsampler that halves the spatial size while
    doubling the channel width, giving the canonical
    :math:`C, 2C, 4C, 8C` pyramid.

    Parameters
    ----------
    image_size : int, optional
        Spatial side length (in pixels) of the square input image.
        Defaults to ``224``.  Must be divisible by ``patch_size``.
    patch_size : int, optional
        Side length of the non-overlapping patches extracted by the stem
        convolution.  Defaults to ``4``.
    in_channels : int, optional
        Number of input image channels.  Defaults to ``3`` (RGB).
    num_classes : int, optional
        Number of output classes for the classification head.  Defaults
        to ``1000`` (ImageNet-1k).
    embed_dim : int, optional
        Channel width :math:`C` after patch embedding (stage-1 dim).
        Each subsequent stage doubles, so the final stage has width
        :math:`8C`.  Defaults to ``96`` (Swin-T).
    depths : tuple of int, optional
        Number of Swin blocks in each of the four stages.  Defaults to
        ``(2, 2, 6, 2)`` (Swin-T).
    num_heads : tuple of int, optional
        Number of attention heads in each stage.  Defaults to
        ``(3, 6, 12, 24)`` (Swin-T).
    window_size : int, optional
        Side length :math:`M` of the local attention window inside each
        Swin block.  Defaults to ``7``.
    mlp_ratio : float, optional
        Expansion ratio of the MLP block.  Defaults to ``4.0``.
    dropout : float, optional
        Dropout probability inside the MLP feed-forward layer and after
        the patch embedding.  Defaults to ``0.0``.
    attention_dropout : float, optional
        Dropout probability on post-softmax attention weights.  Defaults
        to ``0.0``.
    drop_path_rate : float, optional
        Stochastic-depth maximum rate; linearly scheduled across all
        blocks of the trunk.  Appendix A2.1 gives ``0.2`` / ``0.3`` /
        ``0.5`` for Swin-T / Swin-S / Swin-B on ImageNet-1k.  It gives no
        from-scratch rate for Swin-L, which the paper only ever
        pre-trains on ImageNet-22k (that fine-tune stage uses ``0.1``).
        Defaults to ``0.0``.

    Attributes
    ----------
    model_type : ClassVar[str]
        Constant string ``"swin"`` used by the model registry.

    Notes
    -----
    The canonical variants registered as factory functions in
    :mod:`lucid.models.vision.swin` are (see Table 7 of the paper):

    ============= ========== ================== ===================
    Variant       embed_dim  depths             num_heads
    ============= ========== ================== ===================
    Swin-T        96         (2, 2, 6, 2)       (3, 6, 12, 24)
    Swin-S        96         (2, 2, 18, 2)      (3, 6, 12, 24)
    Swin-B        128        (2, 2, 18, 2)      (4, 8, 16, 32)
    Swin-L        192        (2, 2, 18, 2)      (6, 12, 24, 48)
    ============= ========== ================== ===================

    Reference: Ze Liu *et al.*, *"Swin Transformer: Hierarchical Vision
    Transformer using Shifted Windows"*, ICCV 2021,
    `arXiv:2103.14030 <https://arxiv.org/abs/2103.14030>`_.

    Examples
    --------
    Build a Swin-Tiny configuration for CIFAR-100 (100 classes):

    >>> from lucid.models.vision.swin import SwinConfig
    >>> cfg = SwinConfig(num_classes=100)
    >>> cfg.embed_dim, cfg.depths, cfg.num_heads
    (96, (2, 2, 6, 2), (3, 6, 12, 24))

    Override the stem patch size for a higher-resolution variant:

    >>> cfg = SwinConfig(patch_size=8, image_size=384)
    >>> cfg.image_size // cfg.patch_size  # tokens per side
    48
    """

    model_type: ClassVar[str] = "swin"

    image_size: int = 224
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 1000
    embed_dim: int = 96
    depths: tuple[int, ...] = (2, 2, 6, 2)
    num_heads: tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "num_heads", tuple(self.num_heads))
        check_stage_lengths(type(self).__name__, depths=self.depths, num_heads=self.num_heads)
