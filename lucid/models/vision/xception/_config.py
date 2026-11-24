"""Xception configuration (Chollet, 2017)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive


@model_family_meta(
    canonical_name="Xception",
    citation=(
        'Chollet, François. "Xception: Deep Learning with Depthwise '
        'Separable Convolutions." Proceedings of the IEEE Conference '
        "on Computer Vision and Pattern Recognition, 2017, "
        "pp. 1251–1258."
    ),
    theory=r"""
    Xception ("Extreme Inception") interprets the Inception module
    as an *intermediate* point on a continuum between an ordinary
    convolution and a fully decoupled spatial/channel filter.  A
    standard Inception block splits the input into a few parallel
    branches, each of which is processed by a small :math:`1\times1`
    pointwise convolution followed by a spatial convolution, and the
    branch outputs are concatenated.  Xception pushes this idea to
    its limit: instead of a handful of branches, *every output
    channel* of the :math:`1\times1` pointwise convolution gets its
    own independent :math:`3\times3` spatial filter.  This is exactly
    the **depthwise separable convolution** — §1.2 defines it as a
    depthwise (per-channel spatial) convolution *followed by* a
    pointwise :math:`1\times1` mix in channel space, which is also the
    order the code builds.  (Xception's own blocks are the "extreme
    Inception" reading of that same pair, so the modules appear in the
    depthwise-then-pointwise order throughout.)

    The architectural hypothesis is that the cross-channel
    correlations and the spatial correlations in feature maps are
    sufficiently *decoupled* that they can — and should — be modeled
    independently.  A regular convolution conflates the two; a
    depthwise separable convolution separates them, dramatically
    reducing both the parameter count and the FLOPs:

    .. math::

        D_K^2 \cdot M \cdot N \;\;\longrightarrow\;\;
        M \cdot N \;+\; D_K^2 \cdot M.

    Xception stacks 36 such layers, organised into 14 modules.  Twelve
    of them carry a linear residual shortcut (similar to ResNet); the
    first and the last do not.
    With essentially the *same* parameter budget as Inception-v3, it
    outperforms it on ImageNet and substantially outperforms it on
    the much larger JFT dataset.  The architecture is designed for a
    :math:`299 \times 299` input — the same crop size as Inception-v3 —
    so the spatial-reduction schedule matches that of its
    predecessor exactly, isolating the depthwise-separable
    factorisation as the sole source of accuracy improvement.
    """,
)
@dataclass(frozen=True)
class XceptionConfig(ModelConfig):
    """Configuration for Xception (Extreme Inception).

    Replaces all Inception modules with depthwise separable convolutions.
    Designed for 299×299 inputs.

    ``dropout`` — head dropout rate (0.5 in the paper).

    Examples
    --------
    >>> from lucid.models.vision.xception._config import XceptionConfig
    >>> cfg = XceptionConfig()
    >>> cfg.model_type
    'xception'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "xception"

    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.5

    def __post_init__(self) -> None:
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
