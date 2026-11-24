"""SKNet configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import (
    check_choice,
    check_non_negative,
    check_positive,
    check_residual_layers,
)


@model_family_meta(
    canonical_name="SKNet",
    citation=(
        'Li, Xiang, et al. "Selective Kernel Networks." Proceedings '
        "of the IEEE/CVF Conference on Computer Vision and Pattern "
        "Recognition, 2019, pp. 510–519."
    ),
    theory=r"""
    SKNet builds on the biological observation that the receptive
    field size of cortical neurons is not fixed but is *modulated
    by the stimulus*.  Standard ConvNets bake the receptive field
    into the architecture (a :math:`3\times3` kernel always sees
    a :math:`3\times3` neighbourhood); SKNet replaces that with a
    **Selective Kernel** unit that dynamically chooses how much of
    each receptive-field size to use for every input, on a
    per-channel basis.

    A Selective Kernel block runs the input :math:`x` through two
    (or more) parallel branches with different effective receptive
    fields — typically a :math:`3\times3` convolution and a
    :math:`3\times3` dilation-2 convolution (giving an effective
    :math:`5\times5` receptive field).  Their outputs
    :math:`\tilde U_1` and :math:`\tilde U_2` are summed and pooled
    to produce a compact channel descriptor, which is then passed
    through a lightweight SE-style bottleneck and a *per-branch
    softmax* to yield soft attention weights:

    .. math::

        a_c, b_c \in (0, 1), \quad a_c + b_c = 1
        \qquad \forall c \in \{1, \dots, C\}.

    The final output of the block is the channel-wise convex
    combination

    .. math::

        V_c = a_c \cdot \tilde U_{1, c} + b_c \cdot \tilde U_{2, c}.

    Because :math:`a_c` and :math:`b_c` are computed from the input
    itself, every channel of every feature map can effectively
    choose its own receptive field at every spatial location — a
    form of soft kernel selection driven by data.

    Plugging Selective Kernel units into the ResNet bottleneck in
    place of the central :math:`3\times3` convolution gives the
    SK-ResNet family (SKNet-50 / 101 / 200 in the paper).  An
    aggressive ResNeXt-style cardinality / base-width split lets
    the same block scale up to SK-ResNeXt-50 32×4d.  In Lucid these
    are all parameterised by :class:`SKNetConfig`'s ``cardinality``,
    ``base_width``, ``split_input`` and ``rd_ratio`` fields.
    """,
)
@dataclass(frozen=True)
class SKNetConfig(ModelConfig):
    """Unified config for all SK-ResNet variants (Li et al., 2019).

    Paper: "Selective Kernel Networks"

    Architecture is identical to ResNet-50 (expansion=4, stages output
    256/512/1024/2048 channels) except each 3×3 conv in the bottleneck
    is replaced by a SelectiveKernel unit with two parallel branches
    (3×3 + 3×3 dilated-2, mimicking 5×5 receptive field).

    Key hyper-parameters:

    ``layers``
        Number of bottleneck blocks per stage (default ResNet-50 = 3/4/6/3).

    ``cardinality``
        Number of groups for the SK branch convolutions (G in the paper).
        Also used in the ResNeXt-style width formula:
          width = int(planes * (base_width / 64)) * cardinality
        Set to 1 for plain SK-ResNet (default).

    ``base_width``
        Base channel multiplier for the ResNeXt width formula.
        64 → plain ResNet widths (64/128/256/512 at each stage).
        4 with cardinality=32 → SK-ResNeXt-50 32×4d (SKNet-50 from paper).

    ``split_input``
        If True (timm default), each SK branch receives half the input
        channels, keeping the param count similar to a single grouped conv.

    ``rd_ratio``
        Reduction ratio for the SelectiveKernelAttn bottleneck.

    ``rd_divisor``
        Divisor for rounding the attention channel count.

    ``block_type``
        ``"bottleneck"`` (default, expansion=4) for SK-ResNet-50/101 or
        ``"basic"`` (expansion=1) for SK-ResNet-18/34.
    ``min_attn_channels``
        Eq. (4)'s :math:`L`, "the minimal value of :math:`d`", which the
        paper sets to 32 in every experiment.  It stops the attention
        bottleneck from collapsing on narrow stages, where
        :math:`C/r` alone would give 8.

        SK-ResNet-18 and -34 set it to 16 instead, because the
        checkpoints those two factories ship were trained without the
        floor: with 32 the first two stages want a 32-wide bottleneck
        and the weights carry a 16-wide one, so the load fails outright.
        The paper's geometry is still reachable --
        ``SKNetConfig(min_attn_channels=32)`` -- it simply has no
        published weights.

    Examples
    --------
    >>> from lucid.models.vision.sknet._config import SKNetConfig
    >>> cfg = SKNetConfig()
    >>> cfg.model_type
    'sknet'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "sknet"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 4, 6, 3)
    block_type: str = "bottleneck"
    cardinality: int = 1
    base_width: int = 64
    split_input: bool = True
    rd_ratio: float = 1.0 / 16
    rd_divisor: int = 8
    min_attn_channels: int = 32
    # §4.1: "Label-smoothing regularization is used during training."
    # Reproducing the paper's reported top-1 needs it, but it is a training
    # choice, so it defaults off and inference is unaffected either way.
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        owner = type(self).__name__
        check_residual_layers(owner, layers=self.layers)
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(
            owner,
            cardinality=self.cardinality,
            base_width=self.base_width,
            rd_ratio=self.rd_ratio,
            rd_divisor=self.rd_divisor,
        )
        # A floor of 0 is no floor, so only a negative one is meaningless.
        check_non_negative(owner, min_attn_channels=self.min_attn_channels)
        check_choice(owner, "block_type", self.block_type, ("basic", "bottleneck"))
