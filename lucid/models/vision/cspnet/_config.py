"""CSPNet configuration dataclass (Wang et al., CVPRW 2020)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="CSPNet",
    citation=(
        'Wang, Chien-Yao, et al. "CSPNet: A New Backbone that can '
        'Enhance Learning Capability of CNN." Proceedings of the '
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition "
        "Workshops, 2020, pp. 390–391."
    ),
    theory=r"""
    CSPNet ("Cross-Stage Partial Network") is a generic backbone
    transformation that can be wrapped around almost any
    dense-or-residual stage — ResNet, ResNeXt, DenseNet, Darknet —
    to cut its compute and memory while *improving* its accuracy.

    The CSP transformation factors each stage as follows.  The
    input feature map (after an expansion conv) is split along the
    channel dimension into two halves:

    .. math::

        x = [x', x''], \qquad
        x' \in \mathbb{R}^{H \times W \times C/2}, \;
        x'' \in \mathbb{R}^{H \times W \times C/2}.

    Only :math:`x''` is fed through the heavy residual block to
    produce :math:`y'' = \mathcal{F}(x'')`; the other half
    :math:`x'` *bypasses* the block entirely.  At the end of the
    stage the two halves are concatenated and projected:

    .. math::

        y = \mathrm{Transition}\big( [x', \mathcal{F}(x'')] \big).

    Because :math:`x'` never enters the heavy block, the duplicated
    gradient is split into two truncated paths that share no
    intermediate activations — which both reduces redundant gradient
    information *and* halves the FLOPs of the heavy block.  Applied
    to ResNet-50 / ResNeXt-50 / Darknet-53 backbones, CSP cuts
    compute by 10–20% while improving ImageNet top-1 by a fraction
    of a percent.

    Lucid ships the three paper-cited base variants from Wang 2020:
    :func:`cspresnet_50`, :func:`cspresnext_50`, :func:`cspdarknet_53`.
    All three are built from ``CrossStage``; they differ only in the
    block placed inside it — a residual bottleneck for the first two,
    a Darknet block for :func:`cspdarknet_53`.
    """,
)
@dataclass(frozen=True)
class CSPNetConfig(ModelConfig):
    r"""Generic configuration for every paper-cited CSPNet variant.

    Parameters
    ----------
    num_classes : int, optional, default=1000
    in_channels : int, optional, default=3
    stem_out_chs : int, optional
        Output channels of the stem.  ``64`` for the CSPResNet
        variants (7×7 stride-4 + max-pool), ``32`` for CSPDarknet-53
        (3×3 stride-1, no pool).
    stem_kernel : int, optional, default=7
        Kernel size of the stem convolution.
    stem_stride : int, optional, default=4
        Stride of the stem (post-stem max-pool yields a further /2 for
        CSPResNet, none for CSPDarknet).
    stem_pool : str, optional, default="max"
        ``"max"`` (CSPResNet line) or ``""`` (CSPDarknet — no pool).
    block_type : str, optional, default="bottle"
        Per-stage block flavour: ``"bottle"`` = residual bottleneck
        (CSPResNet / CSPResNeXt), ``"dark"`` = Darknet block
        (CSPDarknet).
    depths : tuple of int, optional
        Block counts per stage.
    out_chs : tuple of int, optional
        Per-stage output channel widths.
    strides : tuple of int, optional
        Per-stage stride (1 = no downsample, 2 = halve spatial).
    groups : int, optional, default=1
        Per-bottleneck grouped-conv count (CSPResNeXt-50 uses 32).
    expand_ratio : float, optional, default=2.0
        How wide ``conv_exp`` is relative to the stage's input.
    bottle_ratio : float, optional, default=0.5
        Bottleneck reduction inside each block.
    block_ratio : float, optional, default=1.0
        Branch-output reduction inside CrossStage transitions.
    cross_linear : tuple of bool, optional, default=(True, True, True, True)
        Whether the ``conv_exp`` of each CrossStage skips its activation.
        The released CSPResNet-50 / CSPResNeXt-50 weights are linear in
        every stage, though darknet's cfg files make stages 1 and 4 leaky;
        CSPDarknet-53 is leaky throughout and sets its own.
    down_growth : bool, optional, default=False
        CSPDarknet only — whether the down-conv at each stage already
        applies the channel expansion.
    dropout : float, optional, default=0.0
        Head dropout probability.

    Examples
    --------
    >>> from lucid.models.vision.cspnet import CSPNetConfig
    >>> cfg = CSPNetConfig()
    >>> cfg.num_classes
    1000
    >>> cfg.model_type
    'cspnet'
    """

    model_type: ClassVar[str] = "cspnet"

    num_classes: int = 1000
    in_channels: int = 3

    # Stem
    stem_out_chs: int = 64
    stem_kernel: int = 7
    stem_stride: int = 2
    stem_pool: str = "max"

    # Per-stage hyperparameters.  Every per-stage field is a tuple whose
    # length must equal ``len(depths)`` — CSPResNet / CSPResNeXt are
    # 4-stage, CSPDarknet-53 is 5-stage.  ``block_type``/``stage_type``
    # vary across paper-cited variants too (``dark`` block inside the
    # CSP-wrapped stages for CSPDarknet, ``bottle`` for the others).
    depths: tuple[int, ...] = (3, 3, 5, 2)
    out_chs: tuple[int, ...] = (128, 256, 512, 1024)
    strides: tuple[int, ...] = (1, 2, 2, 2)
    groups: tuple[int, ...] = (1, 1, 1, 1)
    expand_ratio: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0)
    bottle_ratio: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5)
    block_ratio: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    # darknet's csresnet50.cfg / csresnext50.cfg make the stage-1 and
    # stage-4 split convs ``activation=leaky`` and only stages 2 and 3
    # ``activation=linear``, and this followed that.  The checkpoints do
    # not: measured against timm's cspresnet50 and cspresnext50, every
    # conv_exp is linear.  An activation carries no weights, so the two
    # extra ones cost nothing at load time and changed every prediction.
    # cspdarknet_53 is the other way round -- leaky throughout -- and
    # sets that itself.
    cross_linear: tuple[bool, ...] = (True, True, True, True)
    down_growth: tuple[bool, ...] = (False, False, False, False)
    block_type: tuple[str, ...] = ("bottle", "bottle", "bottle", "bottle")

    dropout: float = 0.0
    # The reference ends its constructor with a zero-init of the last BN in
    # each block and defaults it to *on*, so the residual branches start as
    # identities.
    zero_init_last: bool = True

    def __post_init__(self) -> None:
        for f in (
            "depths",
            "out_chs",
            "strides",
            "groups",
            "expand_ratio",
            "bottle_ratio",
            "block_ratio",
            "cross_linear",
            "down_growth",
            "block_type",
        ):
            object.__setattr__(self, f, tuple(getattr(self, f)))
        # Length consistency: every per-stage tuple must match ``depths``.
        n = len(self.depths)
        for f in (
            "out_chs",
            "strides",
            "groups",
            "expand_ratio",
            "bottle_ratio",
            "block_ratio",
            "cross_linear",
            "down_growth",
            "block_type",
        ):
            if len(getattr(self, f)) != n:
                raise ValueError(
                    f"CSPNetConfig: {f!r} length {len(getattr(self, f))} " f"!= depths length {n}"
                )
