"""GoogLeNet (Inception v1) configuration (Szegedy et al., 2014)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive


@model_family_meta(
    canonical_name="GoogLeNet",
    citation=(
        'Szegedy, Christian, et al. "Going Deeper with Convolutions." '
        "Proceedings of the IEEE Conference on Computer Vision and Pattern "
        "Recognition (CVPR), 2015, pp. 1–9."
    ),
    theory=r"""
    GoogLeNet (Inception v1) is the ILSVRC-2014 classification winner
    and introduced the *Inception module* — a multi-branch building
    block that lets a single layer apply convolutions at several
    receptive-field scales in parallel and concatenate their outputs.
    The motivation is that natural images contain salient structure at
    multiple sizes, and forcing the network to commit to a single
    kernel per layer wastes capacity.

    Each Inception module computes four branches in parallel and
    concatenates them along the channel axis, giving the next layer an
    enriched, multi-scale representation.  To keep the computation
    tractable, *cheap* :math:`1\times1` bottlenecks sit before the
    larger spatial convolutions and after the pooling branch:

    .. math::

        y = \mathrm{concat}\bigl(
            f_{1\times1}(x),\; f_{3\times3}(g_{1\times1}(x)),\;
            f'_{3\times3}(h_{1\times1}(x)),\; p_{1\times1}(\mathrm{pool}(x))
        \bigr),

    where each :math:`f` is a learned convolution and each
    :math:`g, h, p_{1\times1}` is a dimensionality-reducing
    :math:`1\times1` projection.

    Divergence, deliberate: the paper's third branch is a
    :math:`5\times5` convolution.  The reference implementation the
    shipped weights come from uses a :math:`3\times3` there instead —
    a documented deviation on its side — and this build follows the
    weights, so the third branch is :math:`3\times3` in the code and in
    the equation above.

    GoogLeNet stacks nine such modules into a 22-layer network with
    only 5 M parameters — roughly **12×** fewer than AlexNet — while
    achieving a top-5 error of 6.67%.  Two auxiliary classifiers were
    attached to intermediate layers during training to combat
    vanishing gradients and add a regularising effect; they are
    discarded at inference.  The Inception family launched here led to
    Inception v2/v3 (factorised convolutions, label smoothing) and
    eventually Inception-ResNet.
    """,
)
@dataclass(frozen=True)
class GoogLeNetConfig(ModelConfig):
    """Configuration for GoogLeNet (Inception v1).

    ``aux_logits`` enables the two auxiliary classifiers that were used
    during training in the original paper.  They are ignored at inference
    (eval mode); the auxiliary loss is typically weighted 0.3 in the total
    training objective.

    ``dropout`` — main classifier dropout rate (0.4 in the paper).
    ``aux_dropout`` — auxiliary classifier dropout rate (0.7 in the paper).

    Examples
    --------
    >>> from lucid.models.vision.googlenet._config import GoogLeNetConfig
    >>> cfg = GoogLeNetConfig()
    >>> cfg.model_type
    'googlenet'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "googlenet"

    num_classes: int = 1000
    in_channels: int = 3
    aux_logits: bool = True
    # See InceptionConfig.transform_input — the ImageNet-1k checkpoint is a TF
    # port that expects (x-0.5)/0.5 inputs while the bundled preset emits
    # ImageNet-normalised ones.
    transform_input: bool = False
    dropout: float = 0.4
    aux_dropout: float = 0.7

    def __post_init__(self) -> None:
        check_positive(self, "num_classes", "in_channels")
