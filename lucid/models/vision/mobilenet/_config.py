"""MobileNet v1 configuration (Howard et al., 2017)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive


@model_family_meta(
    canonical_name="MobileNet",
    citation=(
        'Howard, Andrew G., et al. "MobileNets: Efficient Convolutional '
        'Neural Networks for Mobile Vision Applications." arXiv preprint '
        "arXiv:1704.04861, 2017."
    ),
    theory=r"""
    MobileNet is built around the *depthwise separable convolution*, a
    factorisation that decomposes a standard convolution into two much
    cheaper stages: a per-channel **depthwise** :math:`3 \times 3`
    spatial filter followed by a :math:`1 \times 1` **pointwise**
    convolution that mixes channels.  For an input of :math:`M`
    channels and an output of :math:`N` channels with kernel size
    :math:`D_K`, the cost drops from
    :math:`D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F` to

    .. math::

        D_K \cdot D_K \cdot M \cdot D_F \cdot D_F
        \;+\; M \cdot N \cdot D_F \cdot D_F,

    a reduction of roughly :math:`1/N + 1/D_K^2`.  For
    :math:`D_K = 3` and the typical :math:`N` of a few hundred,
    that is an 8–9× FLOPs saving with only a small accuracy loss
    on ImageNet.

    Two scalar hyper-parameters expose a smooth accuracy–latency
    trade-off curve at inference time without retraining the whole
    family.  The **width multiplier** :math:`\alpha \in (0, 1]`
    uniformly thins every layer (input and output channels become
    :math:`\alpha M` and :math:`\alpha N`), giving four standard
    variants at :math:`\alpha \in \{1.0, 0.75, 0.5, 0.25\}`.  The
    **resolution multiplier** :math:`\rho` rescales the input spatial
    size — it does not change the model itself, only the feature
    maps flowing through it, and is therefore not enforced inside
    the network.

    Together these two knobs let MobileNet-v1 cover a wide spectrum
    of mobile/embedded budgets from sub-mW edge devices up to
    desktop-class deployment, while keeping a single training recipe
    and a single :class:`MobileNetConfig` schema.
    """,
)
@dataclass(frozen=True)
class MobileNetConfig(ModelConfig):
    """Configuration for MobileNet v1.

    ``width_mult`` — uniform channel multiplier (α in the paper).
      1.0 → full model; 0.75 / 0.5 / 0.25 → slimmer variants.

    The paper's second knob, the resolution multiplier ρ, is *not* a
      field here: it scales the input image, not the architecture, so a
      caller applies it by feeding a smaller image.  It was previously
      documented as ``resolution_mult``, which never existed —
      constructing with it raised ``TypeError``.

    ``dropout`` — classifier dropout.  The paper never mentions dropout;
    §3.2 only says "we use less regularization ... because small models
    have less trouble with overfitting".  0.001 is TF-Slim's
    ``dropout_keep_prob=0.999`` restated as a drop probability, which is
    what the released model was trained with.

    Examples
    --------
    >>> from lucid.models.vision.mobilenet._config import MobileNetConfig
    >>> cfg = MobileNetConfig()
    >>> cfg.model_type
    'mobilenet'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "mobilenet"

    num_classes: int = 1000
    in_channels: int = 3
    width_mult: float = 1.0
    dropout: float = 0.001

    def __post_init__(self) -> None:
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(owner, width_mult=self.width_mult)
