"""MobileNet v2 configuration (Sandler et al., 2018)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive


@model_family_meta(
    canonical_name="MobileNet-v2",
    citation=(
        'Sandler, Mark, et al. "MobileNetV2: Inverted Residuals and '
        'Linear Bottlenecks." Proceedings of the IEEE Conference on '
        "Computer Vision and Pattern Recognition, 2018, pp. 4510–4520."
    ),
    theory=r"""
    MobileNet-v2 replaces the plain depthwise-separable stack of
    v1 with the **inverted residual bottleneck** block.  Each block
    receives a low-dimensional tensor of :math:`c` channels, projects
    it *up* to a much wider :math:`t \cdot c` channel space with a
    :math:`1 \times 1` convolution, applies a :math:`3 \times 3`
    depthwise convolution in that expanded space, then projects
    back *down* to :math:`c'` channels with another :math:`1 \times 1`.
    When the input and output shapes match, an identity shortcut is
    added around the whole block:

    .. math::

        y = x + \mathrm{Proj}\big(\mathrm{DW}\big(\mathrm{Expand}(x)\big)\big).

    This is the *inverse* of the classical bottleneck shape (which is
    wide → narrow → wide), and is why the authors call it an
    "inverted residual".

    The second insight is the **linear bottleneck**.  The final
    :math:`1 \times 1` projection is followed by *no* non-linearity —
    a deliberate departure from the ReLU-after-every-conv pattern.
    The paper's manifold argument shows that ReLU on a low-dimensional
    activation collapses information that cannot be recovered;
    keeping the bottleneck output linear preserves the representational
    capacity of the narrow channels while still letting the wide
    interior of the block use ReLU6 freely.

    These two ideas together give a network that, at comparable
    accuracy to MobileNet-v1, uses roughly :math:`30\%` fewer
    parameters and :math:`30\%` fewer multiply-adds, and runs
    measurably faster on mobile CPUs.  As with v1, a single
    **width multiplier** :math:`\alpha` scales every channel count
    uniformly, exposing the same accuracy/latency trade-off curve
    without retraining.
    """,
)
@dataclass(frozen=True)
class MobileNetV2Config(ModelConfig):
    """Configuration for MobileNet v2.

    ``width_mult`` — uniform channel multiplier; 1.0 = full model.
    ``dropout``    — classifier dropout probability.

    Examples
    --------
    >>> from lucid.models.vision.mobilenet_v2._config import MobileNetV2Config
    >>> cfg = MobileNetV2Config()
    >>> cfg.model_type
    'mobilenet_v2'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "mobilenet_v2"

    num_classes: int = 1000
    in_channels: int = 3
    width_mult: float = 1.0
    dropout: float = 0.2

    def __post_init__(self) -> None:
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(owner, width_mult=self.width_mult)
