"""MobileNet v3 configuration (Howard et al., 2019)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_choice, check_positive


@model_family_meta(
    canonical_name="MobileNet-v3",
    citation=(
        'Howard, Andrew, et al. "Searching for MobileNetV3." Proceedings '
        "of the IEEE/CVF International Conference on Computer Vision, "
        "2019, pp. 1314–1324."
    ),
    theory=r"""
    MobileNet-v3 is the result of an explicit *neural architecture
    search* layered on top of the inverted-residual block of v2.
    A platform-aware NAS (MnasNet-style) selects the per-stage
    expansion ratio, kernel size, and channel count to minimise
    a latency-and-accuracy joint objective on a real mobile CPU,
    after which a NetAdapt pass fine-tunes the channel counts of
    each layer individually under a hard latency budget.  The
    output is two hand-tuned variants — **Large** and **Small** —
    targeting different parts of the latency/accuracy frontier.

    Two architectural refinements give v3 its accuracy gain over v2.
    First, a lightweight **squeeze-and-excitation** module is inserted
    inside selected bottleneck blocks; it pools the spatial dimensions
    and re-weights the channel responses, providing a cheap
    channel-attention mechanism.  Second, a new activation
    function called **hard-swish**,

    .. math::

        \mathrm{h\text{-}swish}(x) = x \cdot \frac{\mathrm{ReLU6}(x + 3)}{6},

    replaces swish/SiLU in the deeper half of the network.  Hard-swish
    matches swish closely in shape but uses only piecewise-linear
    primitives, making it dramatically cheaper to evaluate on
    mobile-friendly fixed-point hardware.

    A further trick is rewriting the expensive final layers of v2.
    The :math:`1 \times 1` expansion before the global average pool
    is moved *after* the pool, where it operates on a single
    spatial location, saving roughly :math:`10\%` of total latency.
    Combined, these changes give MobileNet-v3-Large
    :math:`3.2\%` higher ImageNet top-1 accuracy than v2 at
    :math:`20\%` lower latency on a Pixel-1 CPU.
    """,
)
@dataclass(frozen=True)
class MobileNetV3Config(ModelConfig):
    """Configuration for MobileNet v3.

    ``variant``    — "large" or "small".
    ``width_mult`` — uniform channel multiplier.
    ``dropout``    — classifier dropout probability.

    Examples
    --------
    >>> from lucid.models.vision.mobilenet_v3._config import MobileNetV3Config
    >>> cfg = MobileNetV3Config()
    >>> cfg.model_type
    'mobilenet_v3'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "mobilenet_v3"

    num_classes: int = 1000
    in_channels: int = 3
    variant: str = "large"
    width_mult: float = 1.0
    dropout: float = 0.2

    def __post_init__(self) -> None:
        check_positive(self, "num_classes", "in_channels", "width_mult")
        check_choice(self, "variant", ("large", "small"))
