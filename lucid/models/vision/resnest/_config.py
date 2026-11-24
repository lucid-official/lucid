"""ResNeSt configuration (Zhang et al., 2020)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive, check_residual_layers


@model_family_meta(
    canonical_name="ResNeSt",
    citation=(
        'Zhang, Hang, et al. "ResNeSt: Split-Attention Networks." '
        "Proceedings of the IEEE/CVF Conference on Computer Vision "
        "and Pattern Recognition Workshops, 2022, pp. 2736–2746."
    ),
    theory=r"""
    ResNeSt fuses three previously separate ideas — ResNeXt's
    **cardinality**, SKNet's **soft kernel selection**, and SENet's
    **channel-wise attention** — into a single unified
    *Split-Attention* block that can replace the central
    :math:`3\times3` convolution of any ResNet bottleneck.

    The block first splits the input into :math:`K` groups
    (cardinality), and inside each group it further splits the
    feature map into :math:`R` parallel branches called *radixes*.
    Each radix runs a different transformation
    :math:`\mathcal{F}_r` and produces a :math:`C` -channel feature
    map :math:`U_r`.  The branch outputs are summed to form a
    single combined response :math:`U = \sum_r U_r`, which is then
    pooled and passed through a small bottleneck network to emit
    *per-radix, per-channel* attention weights:

    .. math::

        a_{r, c} = \frac{ \exp\big( z_{r, c} \big) }
                        { \sum_{r' = 1}^{R} \exp\big( z_{r', c} \big) },
        \qquad \sum_{r=1}^{R} a_{r, c} = 1 \;\; \forall c.

    The block's output is the radix-weighted sum

    .. math::

        V_c = \sum_{r=1}^{R} a_{r, c} \cdot U_{r, c},

    applied independently inside every cardinal group and then
    concatenated.  When :math:`R = 1` the block reduces to ResNeXt;
    when :math:`R \geq 2` it generalises SKNet to arbitrary radix
    counts with a softmax (rather than two-way sigmoid) gate.

    Two further refinements complete the architecture.  A
    **deep stem** replaces the original :math:`7\times7` ResNet
    stem with three :math:`3\times3` convolutions, and an
    **average-pool downsampling** path (controlled by the
    ``avg_down`` and ``avd`` fields) replaces strided
    convolutions in residual shortcuts and inside the Split-Attention
    block itself.  With these changes ResNeSt-50 outperforms a
    plain ResNet-50 by roughly 3 ImageNet top-1 percentage points
    at the same parameter count, and serves as a strong backbone
    for downstream detection / segmentation tasks.
    """,
)
@dataclass(frozen=True)
class ResNeStConfig(ModelConfig):
    """Configuration for ResNeSt.

    ``layers``           — per-stage block repetition counts.
    ``radix``            — number of split branches in SplitAttn conv.
    ``groups``           — cardinality (number of convolution groups).
    ``avg_down``         — use AvgPool + 1×1 Conv for downsampling shortcuts.
    ``avd``              — use averaged downsampling (AvgPool) around SplitAttn.
    ``avd_first``        — place the AvgPool before (True) or after (False) SplitAttn.
    ``stem_width``       — channel width of each deep-stem conv (output = stem_width*2).
    ``deep_stem``        — use a 3-convolution deep stem instead of a single 7×7 conv.

    Examples
    --------
    >>> from lucid.models.vision.resnest._config import ResNeStConfig
    >>> cfg = ResNeStConfig()
    >>> cfg.model_type
    'resnest'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "resnest"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 4, 6, 3)
    radix: int = 2
    groups: int = 1
    # Section 5.2 adjusts the width alongside cardinality "so that its overall
    # computational cost remains similar to the ResNet variants" — which is
    # what Table 1's ``2s2x40d`` (radix 2, cardinality 2, width 40) means, and
    # what timm's ``resnest50d_4s2x40d`` / ``resnest50d_1s4x24d`` express.
    # Without it the group width was ``planes * cardinality``, so raising
    # cardinality could only make the network wider.
    bottleneck_width: int = 64
    avg_down: bool = True
    avd: bool = True
    avd_first: bool = False
    stem_width: int = 32
    deep_stem: bool = True
    dropout: float = 0.0
    # Section 4.2: "We also apply DropBlock layers to the convolutional layers
    # at the last two stages of the network."  Training-time only, so 0.0
    # leaves every existing model numerically identical.
    drop_block_rate: float = 0.0
    drop_block_size: int = 3
    zero_init_residual: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        check_residual_layers(self)
        check_positive(
            self,
            "num_classes",
            "in_channels",
            "radix",
            "groups",
            "bottleneck_width",
            "stem_width",
            "drop_block_size",
        )
