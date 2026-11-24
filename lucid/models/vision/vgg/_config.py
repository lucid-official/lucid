"""VGG configuration (Simonyan & Zisserman, 2014)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive


@model_family_meta(
    canonical_name="VGG",
    citation=(
        'Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional '
        'Networks for Large-Scale Image Recognition." International '
        "Conference on Learning Representations (ICLR), 2015."
    ),
    theory=r"""
    VGG demonstrated that *uniformity and depth* — rather than
    bespoke layer designs — were the dominant factors driving image-
    classification accuracy in 2014.  The network is built from a
    single recipe: stack :math:`3\times3` convolutions with stride 1
    and padding 1, periodically halve the spatial resolution with a
    :math:`2\times2` max-pool, double the channel count at each
    pooling boundary, and end with three large fully-connected
    layers.

    The :math:`3\times3` kernel choice is theoretically motivated.
    Two stacked :math:`3\times3` convolutions have the same receptive
    field as a single :math:`5\times5` convolution but use only
    :math:`2 \cdot 9 C^2 = 18 C^2` parameters versus :math:`25 C^2`,
    while inserting an extra nonlinearity between them.  Three stacked
    :math:`3\times3` layers match a :math:`7\times7` receptive field at
    :math:`27 C^2` versus :math:`49 C^2` parameters and *two* extra
    nonlinearities.  Depth at a fixed receptive field therefore buys
    representational power essentially for free.

    The paper's variants (configurations A, B, D, E corresponding to
    VGG-11/13/16/19) differ only in how many :math:`3\times3` convs are
    stacked between pooling layers.  Despite a parameter count of
    138 M — most of it locked up in the 4096-dim fully-connected layers
    — VGG-16 achieves a top-5 ImageNet error of 7.3%, second only to
    GoogLeNet in ILSVRC-2014, and its features transferred so well
    that "VGG features" became the standard backbone for downstream
    tasks (object detection, segmentation, style transfer) for years
    after publication.
    """,
)
@dataclass(frozen=True)
class VGGConfig(ModelConfig):
    """Configuration for all VGG variants (A/B/D/E ≡ 11/13/16/19).

    ``arch`` encodes the per-block conv counts:
      - ``(1, 1, 2, 2, 2)`` → VGG-11
      - ``(2, 2, 2, 2, 2)`` → VGG-13
      - ``(2, 2, 3, 3, 3)`` → VGG-16
      - ``(2, 2, 4, 4, 4)`` → VGG-19

    ``batch_norm`` enables BatchNorm after each Conv+ReLU pair (VGG-BN).
    ``dropout`` applies to the two 4096-dim FC layers (0.5 in the paper).

    Examples
    --------
    >>> from lucid.models.vision.vgg._config import VGGConfig
    >>> cfg = VGGConfig()
    >>> cfg.model_type
    'vgg'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "vgg"

    num_classes: int = 1000
    in_channels: int = 3
    arch: tuple[int, ...] = (2, 2, 3, 3, 3)  # VGG-16 default
    batch_norm: bool = False
    dropout: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "arch", tuple(self.arch))
        # The paper fixes the macro topology at five conv blocks with widths
        # (64, 128, 256, 512, 512), and ``_build_features`` zips ``arch``
        # against that fixed 5-tuple.  A longer arch silently lost its tail
        # blocks; a shorter one built a truncated net that only failed later,
        # inside the classifier, with an unrelated-looking shape error.
        if len(self.arch) != 5:
            raise ValueError(
                f"arch must have exactly 5 entries, one per conv block, got "
                f"{len(self.arch)}: {self.arch!r}"
            )
        check_positive(self, "num_classes", "in_channels", "arch")
