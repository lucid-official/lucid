"""Vision Transformer (ViT) configuration (Dosovitskiy et al., 2020)."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="ViT",
    citation=(
        'Dosovitskiy, Alexey, et al. "An Image is Worth 16x16 Words: '
        'Transformers for Image Recognition at Scale." International '
        "Conference on Learning Representations, 2021."
    ),
    theory=r"""
    The Vision Transformer (ViT) shows that convolutions are *not*
    required for state-of-the-art image classification given enough
    data.  An input image
    :math:`x \in \mathbb{R}^{H \times W \times C}` is reshaped into
    a sequence of non-overlapping square patches
    :math:`x_p \in \mathbb{R}^{N \times (P^2 C)}` with
    :math:`N = HW / P^2`, each patch flattened and projected to a
    :math:`D`-dimensional token by a single linear map.  A learnable
    class token :math:`x_{\text{cls}}` is prepended, and learnable
    positional embeddings :math:`E_{\text{pos}}` are added:

    .. math::

        z_0 = [\,x_{\text{cls}};\; x_p^1 E;\; \dots;\; x_p^N E\,] + E_{\text{pos}}.

    The sequence then flows through :math:`L` stacked transformer
    encoder blocks — multi-head self-attention plus an MLP, each
    wrapped by LayerNorm and residual connections.  Classification
    reads the final state of the class token through a single linear
    head.

    The paper's central empirical claim is that the inductive biases
    convolutions provide (locality, translation equivariance) become
    *unnecessary* once pre-training data is large enough — ViT
    pre-trained on ImageNet-21k or JFT-300M matches or beats the best
    ResNet/EfficientNet baselines on downstream benchmarks while
    using fewer FLOPs.  Three canonical variants from Table 1 (Base /
    Large / Huge) trade compute for accuracy by scaling
    :math:`(D, L, H)` — the hidden width, depth, and number of
    attention heads.
    """,
)
@dataclass(frozen=True)
class ViTConfig(ModelConfig):
    r"""Configuration dataclass for every Vision Transformer (ViT) variant.

    ``ViTConfig`` is an immutable container that fully specifies the
    architecture of a Vision Transformer.  It is consumed by both
    :class:`ViT` (backbone) and :class:`ViTForImageClassification`
    (classifier head on top of the CLS token).  All canonical variants
    described in Dosovitskiy et al. (2020) — Base / Large / Huge at patch
    sizes 14, 16, and 32 — can be expressed by choosing different values for
    ``dim``, ``depth``, ``num_heads``, and ``patch_size``.

    The number of input tokens fed into the transformer encoder is
    :math:`N + 1`, where :math:`N = (\text{image\_size} / \text{patch\_size})^2`
    is the number of non-overlapping patches and the extra token is a
    learnable ``[CLS]`` embedding prepended to the sequence.

    Parameters
    ----------
    image_size : int, optional
        Spatial side length (in pixels) of the square input image.
        ``image_size`` must be divisible by ``patch_size``.  Defaults to
        ``224``, matching the ImageNet recipe in the original paper.
    patch_size : int, optional
        Side length of the non-overlapping square patches extracted by the
        patch-embedding convolution.  Common values are ``16``, ``32``
        (Base / Large) or ``14`` (Huge).  Defaults to ``16``.
    num_classes : int, optional
        Number of output logits produced by the classification head.
        Only used by :class:`ViTForImageClassification`.  Defaults to
        ``1000`` (ImageNet-1k).
    in_channels : int, optional
        Number of input image channels.  Defaults to ``3`` (RGB).
    dim : int, optional
        Hidden width :math:`D` of every transformer block — also the
        dimensionality of the patch embedding and of the CLS token.
        Defaults to ``768`` (ViT-Base).
    depth : int, optional
        Number of stacked transformer encoder blocks :math:`L`.
        Defaults to ``12`` (ViT-Base).
    num_heads : int, optional
        Number of attention heads inside each multi-head self-attention
        block.  ``dim`` must be divisible by ``num_heads``.  Defaults to
        ``12``.
    mlp_ratio : float, optional
        Expansion ratio of the MLP block.  The hidden width of the MLP is
        ``int(dim * mlp_ratio)``; the standard recipe is ``4.0`` (i.e. a
        4x expansion).  Defaults to ``4.0``.
    dropout : float, optional
        Dropout probability applied (i) after the patch embedding +
        positional embedding sum and (ii) inside the MLP feed-forward
        sub-layer.  Defaults to ``0.0``.
    attention_dropout : float, optional
        Dropout probability applied to the post-softmax attention weights
        inside every multi-head self-attention block.  Defaults to ``0.0``.
    layer_norm_eps : float, optional
        Epsilon added to the variance inside every :class:`nn.LayerNorm`
        (the two per-block norms and the final pre-head norm).  The
        original ViT formulation fixes this at ``1e-6``.  Defaults to
        ``1e-6``.

    Attributes
    ----------
    model_type : ClassVar[str]
        Constant string ``"vit"`` used by the model registry to associate
        config instances with the ViT model class.

    Notes
    -----
    The canonical variants registered as factory functions in
    :mod:`lucid.models.vision.vit` are (see Table 1 of the paper):

    ============== ===== ====== ======= ===========
    Variant        dim   depth  heads   patch sizes
    ============== ===== ====== ======= ===========
    ViT-Base       768   12     12      16, 32
    ViT-Large      1024  24     16      16, 32
    ViT-Huge       1280  32     16      14
    ============== ===== ====== ======= ===========

    Reference: Alexey Dosovitskiy *et al.*, *"An Image is Worth 16x16
    Words: Transformers for Image Recognition at Scale"*, ICLR 2021,
    `arXiv:2010.11929 <https://arxiv.org/abs/2010.11929>`_.

    Examples
    --------
    Build a custom ViT-Base/16 configuration for CIFAR-10 (10 classes,
    32x32 inputs, patch size 4 — yields 64 tokens):

    >>> from lucid.models.vision.vit import ViTConfig
    >>> cfg = ViTConfig(
    ...     image_size=32,
    ...     patch_size=4,
    ...     num_classes=10,
    ...     dim=768,
    ...     depth=12,
    ...     num_heads=12,
    ... )
    >>> cfg.image_size, cfg.patch_size, cfg.num_classes
    (32, 4, 10)

    Reuse a canonical variant and override only the number of classes:

    >>> from lucid.models.vision.vit._pretrained import _CFG_B16
    >>> from dataclasses import replace
    >>> cfg = replace(_CFG_B16, num_classes=100)
    >>> cfg.dim, cfg.depth, cfg.num_classes
    (768, 12, 100)
    """

    model_type: ClassVar[str] = "vit"

    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 1000
    in_channels: int = 3
    dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    # §3.1: during *pre-training* the head is an MLP with one hidden layer of
    # width ``representation_size`` followed by tanh; fine-tuning replaces it
    # with a single linear layer.  ``None`` selects the fine-tuning head,
    # which is what every released ImageNet checkpoint uses.
    representation_size: int | None = None
    # §3.1 uses the CLS token, but the official implementation also offers
    # global average pooling over the patch tokens ("gap"), which reaches the
    # same accuracy at a different learning rate (Appendix D.4 / Fig. 9).
    classifier: Literal["token", "gap"] = "token"

    def __post_init__(self) -> None:
        if self.classifier not in ("token", "gap"):
            raise ValueError(f"classifier must be 'token' or 'gap', got {self.classifier!r}")
        if self.representation_size is not None and self.representation_size <= 0:
            raise ValueError(
                "representation_size is the width of the pre-logits hidden "
                f"layer, so it must be positive; got {self.representation_size}. "
                "Use None for the single-linear fine-tuning head."
            )
