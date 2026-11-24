"""AlexNet configuration (Krizhevsky, Sutskever & Hinton, 2012)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive


@model_family_meta(
    canonical_name="AlexNet",
    citation=(
        "Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "
        '"ImageNet Classification with Deep Convolutional Neural '
        'Networks." Advances in Neural Information Processing Systems, '
        "2012.  Single-stream channel widths from Krizhevsky, Alex. "
        '"One weird trick for parallelizing convolutional neural '
        'networks." arXiv:1404.5997 (2014).'
    ),
    theory=r"""
    AlexNet is the architecture that re-ignited deep learning, winning
    the ILSVRC-2012 ImageNet classification challenge with a top-5
    error of 15.3% — more than ten percentage points below the
    runner-up — and demonstrating for the first time that a large
    convolutional network trained end-to-end on raw pixels could
    decisively beat hand-engineered feature pipelines.

    The architecture stacks five convolutional layers followed by
    three fully-connected layers, taking :math:`3\times224\times224`
    RGB inputs and producing 1000-way softmax logits.  Three
    contributions made training such a deep network tractable on
    2012-era hardware: (i) the *rectified linear unit*
    :math:`\phi(x) = \max(0, x)` replaced saturating nonlinearities,
    cutting training time by several factors; (ii) *dropout* with
    :math:`p=0.5` in the two large 4096-dim fully-connected layers
    regularised the ≈61 M parameters against overfitting; and
    (iii) overlapping max-pooling combined with heavy data
    augmentation (random crops, horizontal flips, AlexNet-style PCA
    colour jitter) closed the train-test gap.

    The original 2012 paper split conv filters across two GPUs (model
    parallel) because a single GTX 580's 3 GB memory budget could not
    hold the network — Krizhevsky 2014 ("One weird trick for
    parallelizing convolutional neural networks") later re-derived
    the network as a single merged stream with adjusted channel widths
    :math:`(64, 192, 384, 256, 256)` and no local response
    normalisation; this single-stream variant is what Lucid (and every
    published reference checkpoint) ships.  AlexNet's success
    established the now-standard recipe of deep ConvNet + ReLU +
    dropout + heavy data augmentation + SGD with momentum, and every
    subsequent ImageNet-scale vision model descends from it directly.
    """,
)
@dataclass(frozen=True)
class AlexNetConfig(ModelConfig):
    """Configuration for AlexNet.

    The canonical AlexNet takes 3×224×224 inputs and produces 4096-dim
    embeddings before the final classifier.

    ``dropout`` controls the two dropout layers inside the classifier
    (0.5 in the original paper).

    Examples
    --------
    >>> from lucid.models.vision.alexnet._config import AlexNetConfig
    >>> cfg = AlexNetConfig()
    >>> cfg.model_type
    'alexnet'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "alexnet"

    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.5

    def __post_init__(self) -> None:
        check_positive(self, "num_classes", "in_channels")
