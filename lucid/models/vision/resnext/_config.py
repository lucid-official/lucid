"""ResNeXt configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive, check_residual_layers


@model_family_meta(
    canonical_name="ResNeXt",
    citation=(
        'Xie, Saining, et al. "Aggregated Residual Transformations for '
        'Deep Neural Networks." Proceedings of the IEEE Conference on '
        "Computer Vision and Pattern Recognition (CVPR), 2017, "
        "pp. 1492–1500."
    ),
    theory=r"""
    ResNeXt augments the ResNet bottleneck with a *third* axis of
    network design that the authors term **cardinality**: the number
    of parallel transformation branches inside each block.  Where
    ResNet improves accuracy by making the network *deeper* and Wide
    ResNet improves it by making each layer *wider*, ResNeXt improves
    it by making each block more *parallel*.

    A ResNeXt block of cardinality :math:`C` computes

    .. math::

        y = x + \sum_{i=1}^{C} \mathcal{T}_i(x),

    where each :math:`\mathcal{T}_i` is a low-dimensional bottleneck
    (:math:`1\times1 \to 3\times3 \to 1\times1` with a small inner
    width :math:`d`).  The set of transformations are *aggregated* by
    summation and then added to the residual shortcut, exactly as in
    ResNet.  By construction this aggregation is equivalent to a
    single bottleneck whose :math:`3\times3` convolution is *grouped*
    into :math:`C` groups of :math:`d` channels each — so ResNeXt can
    be implemented as a one-line change to ResNet (replace ``groups=1``
    with ``groups=C``) without any custom block code.

    The empirical claim of the paper is that, holding the parameter
    and FLOP budget fixed, increasing cardinality is *more effective*
    than increasing depth or width.  ResNeXt-50 (32×4d:
    :math:`C=32`, :math:`d=4`) outperforms ResNet-50 by roughly one
    percentage point on ImageNet top-1 with the same compute budget,
    and ResNeXt-101 (64×4d) reaches 20.4% top-1 error.  The "split-
    transform-merge" pattern that ResNeXt makes explicit — independent
    parallel paths combined by aggregation — also turns out to be the
    structural backbone of every subsequent grouped-convolution
    architecture, from ShuffleNet to MobileNetV2's expand-project
    blocks.
    """,
)
@dataclass(frozen=True)
class ResNeXtConfig(ModelConfig):
    """Unified config for all ResNeXt variants (Xie et al., 2017).

    ResNeXt extends ResNet by replacing the plain 3×3 conv in each bottleneck
    with a grouped convolution of ``cardinality`` groups, where each group
    handles ``width_per_group`` channels.

    ``layers`` is the per-stage repetition count, e.g. ``(3, 4, 6, 3)`` for
    ResNeXt-50.  ``cardinality`` and ``width_per_group`` jointly determine the
    intermediate width inside each bottleneck.

    Examples
    --------
    >>> from lucid.models.vision.resnext._config import ResNeXtConfig
    >>> cfg = ResNeXtConfig()
    >>> cfg.model_type
    'resnext'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "resnext"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 4, 6, 3)
    cardinality: int = 32
    width_per_group: int = 4
    dropout: float = 0.0
    # Zero the last BN gamma of every bottleneck so each residual branch
    # starts as an identity — the sibling ResNet config exposes the same
    # switch, and this family's blocks have the identical ``bn3`` slot.
    zero_init_residual: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        check_residual_layers(self)
        check_positive(self, "num_classes", "in_channels", "cardinality", "width_per_group")
