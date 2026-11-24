"""SENet configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_choice, check_positive, check_residual_layers


@model_family_meta(
    canonical_name="SENet",
    citation=(
        'Hu, Jie, et al. "Squeeze-and-Excitation Networks." Proceedings '
        "of the IEEE Conference on Computer Vision and Pattern "
        "Recognition, 2018, pp. 7132–7141."
    ),
    theory=r"""
    SENet introduces the **Squeeze-and-Excitation block**, a tiny
    drop-in channel-attention module that recalibrates the relative
    importance of each feature-map channel after every convolutional
    stage.  The block runs in three phases.

    The **squeeze** phase reduces each channel to a single scalar by
    global average pooling over the spatial dimensions:

    .. math::

        z_c = \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W}
              u_c(i, j),

    yielding a channel descriptor :math:`z \in \mathbb{R}^{C}` that
    summarises the global response of every channel.

    The **excitation** phase passes :math:`z` through a small
    two-layer fully-connected bottleneck with a reduction ratio
    :math:`r` (typically 16) and a sigmoid output:

    .. math::

        s = \sigma\big( W_2 \,\delta( W_1 z ) \big),
        \qquad W_1 \in \mathbb{R}^{C/r \times C}, \;
        W_2 \in \mathbb{R}^{C \times C/r},

    where :math:`\delta` is ReLU and :math:`\sigma` is the
    sigmoid.  The output :math:`s \in (0, 1)^{C}` is interpreted
    as per-channel gating weights.

    Finally the **scale** phase modulates the original feature
    map channel-wise:

    .. math::

        \tilde{u}_c = s_c \cdot u_c.

    Plugging this block into the end of every residual unit of a
    ResNet-50 (giving SE-ResNet-50) adds only ~10% extra parameters
    and negligible FLOPs while reducing ImageNet top-5 error by
    roughly 1.5 percentage points — a result that won the ILSVRC
    2017 classification challenge.  In Lucid :class:`SENetConfig`
    parameterises every SE-ResNet variant (18 / 34 / 50 / 101 / 152)
    with the same ``layers`` and ``block_type`` knobs as ResNet plus
    the SE-specific ``reduction`` ratio.
    """,
)
@dataclass(frozen=True)
class SENetConfig(ModelConfig):
    """Unified config for all SE-ResNet variants (Hu et al., 2017).

    Paper: "Squeeze-and-Excitation Networks"

    ``block_type`` selects BasicBlock (SE-ResNet-18/34) or Bottleneck
    (SE-ResNet-50/101/152).  ``layers`` is the per-stage repetition count.
    ``reduction`` is the channel reduction ratio in the SE block (default 16).
    ``legacy_pool`` selects the original SENet stem max-pool geometry
    (``padding=0``, ``ceil_mode=True``) used by the canonical SE-ResNet
    line; ``False`` (default) uses the modern ResNet stem pool
    (``padding=1``, ``ceil_mode=False``).

    Examples
    --------
    >>> from lucid.models.vision.senet._config import SENetConfig
    >>> cfg = SENetConfig()
    >>> cfg.model_type
    'senet'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "senet"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 4, 6, 3)
    reduction: int = 16
    block_type: str = "bottleneck"
    legacy_pool: bool = False
    # Original Caffe SENet bottleneck: stride on the 1x1 conv1, not the 3x3
    # conv2.  The 101/152 checkpoints come from that lineage; the 50 comes
    # from a modern v1.5 retrain, so this cannot be a global switch.
    legacy_stride: bool = False
    # Zero the last BN gamma of every residual branch (bn3 on bottlenecks,
    # bn2 on basic blocks), as the sibling ResNet config already offers.  The
    # shipped se_resnet_50 checkpoint was itself trained with it enabled.
    zero_init_residual: bool = False
    # Dropout before the classifier.  The reference's legacy SENet class
    # defaults to 0.2 and its ``legacy_seresnet*`` entrypoints inherit that;
    # the original authors' own port passed no dropout for the se_resnet
    # line, so this defaults to 0.0 and is opt-in.
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        owner = type(self).__name__
        check_residual_layers(owner, layers=self.layers)
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(owner, reduction=self.reduction)
        check_choice(owner, "block_type", self.block_type, ("basic", "bottleneck"))
