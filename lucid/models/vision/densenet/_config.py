"""DenseNet configuration (Huang et al., 2016)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_non_negative, check_positive


@model_family_meta(
    canonical_name="DenseNet",
    citation=(
        'Huang, Gao, et al. "Densely Connected Convolutional Networks." '
        "Proceedings of the IEEE Conference on Computer Vision and Pattern "
        "Recognition (CVPR), 2017, pp. 4700–4708."
    ),
    theory=r"""
    DenseNet takes the residual-shortcut idea to its logical extreme.
    Inside each *dense block*, every layer receives as input the
    concatenated feature maps of *all* previous layers in the block,
    not just the immediately preceding layer.  For a block of
    :math:`L` layers the :math:`\ell`-th layer computes

    .. math::

        x_\ell = H_\ell\bigl(
            [x_0,\; x_1,\; \dots,\; x_{\ell-1}]
        \bigr),

    where :math:`[\cdot]` denotes channel-wise concatenation and
    :math:`H_\ell` is a BN → ReLU → :math:`3\times3` Conv composite
    (optionally preceded by a :math:`1\times1` bottleneck).  Each
    :math:`H_\ell` contributes a small fixed number :math:`k` of new
    feature maps — the *growth rate* — so the input width of layer
    :math:`\ell` grows linearly as :math:`k_0 + (\ell - 1)\,k`.

    This dense connectivity has three notable consequences.  First,
    every layer has direct supervision-style access to the loss
    gradient through the concatenation paths, easing optimisation in
    very deep networks.  Second, parameter efficiency is excellent:
    because each layer only adds :math:`k` (typically 12, 24, or 32)
    new channels rather than transforming the full activation, a
    DenseNet with 0.8 M parameters matches the accuracy of a 1.7 M
    parameter ResNet on CIFAR.  Third, feature *reuse* is explicit —
    later layers can directly combine raw early features with refined
    deep features without re-encoding them.

    The four ImageNet variants — DenseNet-121, 169, 201, and 264 —
    differ only in the per-block layer counts (``block_config``).  All
    use four dense blocks separated by *transition* layers
    (:math:`1\times1` Conv + :math:`2\times2` average pool) that
    compress the channel count and halve the spatial resolution.
    DenseNet-121 reaches a top-1 ImageNet error of 25.0% with 8 M
    parameters, substantially better parameter efficiency than the
    contemporary ResNet-50.
    """,
)
@dataclass(frozen=True)
class DenseNetConfig(ModelConfig):
    """Configuration for all DenseNet variants (121/169/201/264).

    ``growth_rate`` (k) — number of feature maps each dense layer contributes.
    ``block_config`` — number of dense layers per block (4 blocks total).
    ``num_init_features`` — channels after the initial conv stem.
    ``bn_size`` — bottleneck expansion factor (each layer uses bn_size*k filters
      in its 1×1 branch before the 3×3 branch).
    ``dropout_rate`` — dropout after each dense layer (0 = disabled).

    Examples
    --------
    >>> from lucid.models.vision.densenet._config import DenseNetConfig
    >>> cfg = DenseNetConfig()
    >>> cfg.model_type
    'densenet'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "densenet"

    num_classes: int = 1000
    in_channels: int = 3
    growth_rate: int = 32
    block_config: tuple[int, ...] = (6, 12, 24, 16)  # DenseNet-121
    num_init_features: int = 64
    bn_size: int = 4
    dropout_rate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_config", tuple(self.block_config))
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(
            owner,
            growth_rate=self.growth_rate,
            num_init_features=self.num_init_features,
            bn_size=self.bn_size,
        )
        # A negative count builds an empty block yet still subtracts
        # ``n * growth_rate`` from the running width, so the next stage is
        # sized for channels that never arrive (a ShapeMismatch in the first
        # forward).  Zero is a block with no layers, which the model handles.
        check_non_negative(owner, block_config=self.block_config)
