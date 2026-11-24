"""EfficientNet configuration (Tan & Le, 2019)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive


@model_family_meta(
    canonical_name="EfficientNet",
    citation=(
        'Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model '
        'Scaling for Convolutional Neural Networks." Proceedings of the '
        "36th International Conference on Machine Learning, PMLR 97, "
        "2019, pp. 6105–6114."
    ),
    theory=r"""
    EfficientNet is built on the empirical observation that the three
    axes used to scale a ConvNet — **depth** :math:`d` (number of
    layers), **width** :math:`w` (number of channels per layer), and
    input **resolution** :math:`r` — are not independent.  Scaling any
    one of them in isolation gives rapidly diminishing returns, but
    scaling all three together along a carefully chosen ratio gives a
    much better accuracy/FLOPs frontier.

    The paper formalises this as **compound scaling**: a single
    user-facing scalar :math:`\phi` controls all three axes through

    .. math::

        d = \alpha^\phi, \quad
        w = \beta^\phi, \quad
        r = \gamma^\phi,

    subject to the FLOPs-balance constraint
    :math:`\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2`.  The
    constants :math:`\alpha, \beta, \gamma` are found once by a
    small grid search on the baseline network (giving roughly
    :math:`\alpha = 1.2, \beta = 1.1, \gamma = 1.15`); from then on,
    a single :math:`\phi` indexes the whole family.  Successive
    integer values produce the standard B0 → B7 variants, each
    requiring approximately :math:`2^\phi` more FLOPs than B0.

    The baseline B0 itself is *not* manually designed.  It is the
    output of a multi-objective neural-architecture search that
    optimises accuracy *and* FLOPs jointly.  Each block is an
    inverted-residual MBConv (MobileNet-v2 style) augmented with a
    squeeze-and-excitation module, and the network uses the swish
    activation throughout.  **Stochastic depth** with linearly
    increasing drop rate (the ``drop_connect_rate`` field) acts as
    a strong regulariser as depth grows, which is what allows
    B6/B7 to keep gaining accuracy where naively-deepened
    baselines would overfit or fail to train.
    """,
)
@dataclass(frozen=True)
class EfficientNetConfig(ModelConfig):
    """Configuration for EfficientNet B0–B7.

    Compound scaling coefficients:
      B0: width=1.0, depth=1.0, res=224, dropout=0.2
      B1: width=1.0, depth=1.1, res=240, dropout=0.2
      B2: width=1.1, depth=1.2, res=260, dropout=0.3
      B3: width=1.2, depth=1.4, res=300, dropout=0.3
      B4: width=1.4, depth=1.8, res=380, dropout=0.4
      B5: width=1.6, depth=2.2, res=456, dropout=0.4
      B6: width=1.8, depth=2.6, res=528, dropout=0.5
      B7: width=2.0, depth=3.1, res=600, dropout=0.5

    ``drop_connect_rate`` — stochastic depth rate applied linearly across blocks.
    ``se_ratio`` — squeeze-and-excitation reduction ratio (0.25 in the paper).

    Examples
    --------
    >>> from lucid.models.vision.efficientnet._config import EfficientNetConfig
    >>> cfg = EfficientNetConfig()
    >>> cfg.model_type
    'efficientnet'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "efficientnet"

    num_classes: int = 1000
    in_channels: int = 3
    width_mult: float = 1.0
    depth_mult: float = 1.0
    dropout: float = 0.2
    drop_connect_rate: float = 0.2
    se_ratio: float = 0.25
    # BatchNorm hyper-parameters.  The zoo that produced the shipped B5/B6/B7
    # checkpoints builds those three with eps=1e-3, momentum=0.01 and leaves
    # B0-B4 on the framework defaults, so this has to be settable per variant:
    # the checkpoints' running_var was fitted under their own eps, and eval
    # divides by sqrt(running_var + eps).
    bn_eps: float = 1e-5
    bn_momentum: float = 0.1

    def __post_init__(self) -> None:
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(owner, width_mult=self.width_mult, depth_mult=self.depth_mult)
