"""ResNet configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="ResNet",
    citation=(
        'He, Kaiming, et al. "Deep Residual Learning for Image '
        'Recognition." Proceedings of the IEEE Conference on Computer '
        "Vision and Pattern Recognition, 2016, pp. 770–778."
    ),
    theory=r"""
    Deep residual networks introduce *identity shortcuts* around every
    pair (or triple) of stacked convolutions.  A residual block computes

    .. math::

        y = \mathcal{F}(x, \{W_i\}) + x,

    where :math:`\mathcal{F}` is the stacked-convolution branch and
    the addition is the identity shortcut.  Reformulating the layer to
    learn a *residual* :math:`\mathcal{F}` rather than the full mapping
    :math:`\mathcal{H}(x) = \mathcal{F}(x) + x` makes the optimisation
    surface dramatically easier — the network only has to fit the
    *correction* to the identity, which is closer to zero at
    initialisation than the full mapping is to the target.

    Stacking these blocks lets gradient signals flow through dozens
    or hundreds of layers without vanishing.  The paper showed
    networks up to 152 layers deep training stably on ImageNet, with
    each successive depth setting a new state-of-the-art at the time
    (3.57% top-5 error, winning ILSVRC 2015).

    Two block topologies are used.  ``BasicBlock`` (two ``3×3`` convs)
    is used in ResNet-18/34.  ``Bottleneck`` (``1×1 → 3×3 → 1×1`` with
    a 4× channel expansion) is used in ResNet-50/101/152 and all
    deeper / wider variants.  The four ImageNet variants from Table 1
    of the paper differ only in ``(block, [n₁, n₂, n₃, n₄])`` where
    :math:`n_i` is the block-repetition count at stage :math:`i`.
    """,
)
@dataclass(frozen=True)
class ResNetConfig(ModelConfig):
    r"""Frozen configuration dataclass for every ResNet variant.

    A single ``ResNetConfig`` instance fully specifies the architecture of
    any model in the ResNet family (ResNet-18/34/50/101/152, Wide ResNet,
    or deeper bottleneck variants such as ResNet-200/269).  The
    :class:`ResNet` and :class:`ResNetForImageClassification` constructors
    take exactly one of these objects — no extra constructor arguments
    are ever needed.

    Fields are immutable (``frozen=True``) so the same config can be
    safely shared across training runs, serialised to ``config.json``
    via :meth:`ModelConfig.save`, and reloaded without copy-on-write
    surprises.  ``__post_init__`` re-coerces ``layers`` and
    ``hidden_sizes`` to ``tuple`` because JSON round-trips would
    otherwise hand back ``list`` objects, breaking the
    :class:`dataclasses.dataclass(frozen=True)` hash contract.

    Parameters
    ----------
    num_classes : int, optional, default=1000
        Number of logits produced by the classification head.  The
        default matches ImageNet-1k.  Ignored by the backbone-only
        :class:`ResNet` class.
    in_channels : int, optional, default=3
        Number of input image channels.  ``3`` for RGB inputs (the
        original paper setting); set to ``1`` for grayscale or higher
        for multi-spectral inputs.
    block_type : {"basic", "bottleneck"}, optional, default="bottleneck"
        Residual block topology.  ``"basic"`` selects the two-conv
        :class:`_BasicBlock` used in ResNet-18/34; ``"bottleneck"``
        selects the three-conv :class:`_Bottleneck` used in
        ResNet-50/101/152 and all wider/deeper variants.
    layers : tuple[int, ...], optional, default=(3, 4, 6, 3)
        Block-repetition count for each of the four stages (stage-1
        through stage-4).  The default ``(3, 4, 6, 3)`` is the ResNet-50
        configuration from Table 1 of He et al. (2015).
    stem_channels : int, optional, default=64
        Output channel count of the 7×7 stem convolution.  Also the
        input channel count of stage-1.  Almost never changed from the
        paper default.
    hidden_sizes : tuple[int, ...], optional, default=(64, 128, 256, 512)
        Base channel count per stage *before* expansion.  Each stage's
        final width is ``hidden_sizes[i] * block.expansion`` — so the
        last-stage feature map is 512 channels for BasicBlock or 2048
        channels for Bottleneck.
    bottleneck_width_mult : int, optional, default=1
        Multiplier applied to the *inner* 3×3 channels inside each
        :class:`_Bottleneck`.  Set to ``2`` for Wide ResNet-50-2 /
        101-2 (Zagoruyko & Komodakis, 2016); leaves the per-stage
        output width unchanged.
    dropout : float, optional, default=0.0
        Dropout probability applied before the final classification
        linear layer in :class:`ResNetForImageClassification`.
        Ignored by the backbone-only model.
    zero_init_residual : bool, optional, default=False
        If ``True``, initialise the final BatchNorm in every residual
        branch (``bn2`` for BasicBlock, ``bn3`` for Bottleneck) with
        zero weights so each block starts as the identity function.
        This trick, popularised by Goyal et al. ("Accurate, Large
        Minibatch SGD"), can improve training stability at large batch
        sizes.

    Attributes
    ----------
    model_type : ClassVar[str]
        Persistent family identifier ``"resnet"`` — embedded in
        ``config.json`` and used by directory-based ``from_pretrained``
        to dispatch to the correct registry entry.

    Notes
    -----
    The four canonical ImageNet variants from He et al., "Deep Residual
    Learning for Image Recognition", CVPR 2016 (arXiv:1512.03385,
    Table 1) correspond to the following ``(block_type, layers)``
    tuples:

    .. math::

        \begin{aligned}
            \text{ResNet-18}  \;&\to\; (\text{basic},\;     (2, 2, 2, 2))  \\\\
            \text{ResNet-34}  \;&\to\; (\text{basic},\;     (3, 4, 6, 3))  \\\\
            \text{ResNet-50}  \;&\to\; (\text{bottleneck},\; (3, 4, 6, 3))  \\\\
            \text{ResNet-101} \;&\to\; (\text{bottleneck},\; (3, 4, 23, 3)) \\\\
            \text{ResNet-152} \;&\to\; (\text{bottleneck},\; (3, 8, 36, 3))
        \end{aligned}

    Always prefer the factory functions in ``._pretrained`` (e.g.
    :func:`resnet_50`) over hand-rolling a config — they encode the
    exact paper-cited topology and integrate with the model registry.

    Examples
    --------
    Construct the ResNet-50 config and inspect it:

    >>> from lucid.models.vision.resnet import ResNetConfig
    >>> cfg = ResNetConfig(block_type="bottleneck", layers=(3, 4, 6, 3))
    >>> cfg.num_classes
    1000
    >>> cfg.model_type
    'resnet'

    Build a Wide ResNet-50-2 config (2x inner width, same output width):

    >>> wide = ResNetConfig(
    ...     block_type="bottleneck",
    ...     layers=(3, 4, 6, 3),
    ...     bottleneck_width_mult=2,
    ... )
    >>> wide.bottleneck_width_mult
    2

    JSON round-trip (lists get coerced back to tuples):

    >>> import json
    >>> d = cfg.to_dict()
    >>> restored = ResNetConfig.from_dict(json.loads(json.dumps(d)))
    >>> isinstance(restored.layers, tuple)
    True
    """

    model_type: ClassVar[str] = "resnet"

    num_classes: int = 1000
    in_channels: int = 3
    block_type: str = "bottleneck"
    layers: tuple[int, ...] = (3, 4, 6, 3)
    stem_channels: int = 64
    hidden_sizes: tuple[int, ...] = (64, 128, 256, 512)
    bottleneck_width_mult: int = 1
    dropout: float = 0.0
    zero_init_residual: bool = False

    def __post_init__(self) -> None:
        # JSON round-trips lists; coerce back to tuples for frozen-dataclass fields.
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        # ``block_type`` is documented as a two-value enumeration but typed as
        # a bare str.  Untouched, a typo silently selected the *other* branch —
        # ``block_type="Bottleneck"`` quietly built a BasicBlock network.
        if self.block_type not in ("basic", "bottleneck"):
            raise ValueError(
                f"block_type must be 'basic' or 'bottleneck', got " f"{self.block_type!r}"
            )
