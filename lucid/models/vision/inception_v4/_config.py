"""Inception-v4 configuration (Szegedy et al., 2017)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive


@model_family_meta(
    canonical_name="Inception-v4",
    citation=(
        'Szegedy, Christian, et al. "Inception-v4, Inception-ResNet and '
        'the Impact of Residual Connections on Learning." Proceedings of '
        "the AAAI Conference on Artificial Intelligence, vol. 31, no. 1, "
        "2017, pp. 4278–4284."
    ),
    theory=r"""
    Inception-v4 is the purely *non-residual* member of the 2016
    Inception generation.  Where Inception v3 had grown piecemeal —
    each stage partitioned differently so the model could be trained
    across many machines under memory limits — Inception-v4 re-draws the
    whole network as a *uniform* design: one fixed stem, three kinds of
    Inception module repeated in blocks, and two dedicated reduction
    modules between them.  Every module is a set of parallel branches
    whose outputs are concatenated along the channel axis,

    .. math::

        y = \operatorname{concat}\big(
            \mathcal{B}_1(x),\ \mathcal{B}_2(x),\ \dots,\ \mathcal{B}_k(x)
        \big),

    where each branch :math:`\mathcal{B}_i` is a short chain of
    convolutions (or a pooling followed by a :math:`1\times1`
    projection), each followed by batch normalisation and a ReLU.

    The layout is Stem → 4× Inception-A (:math:`35\times35\times384`) →
    Reduction-A → 7× Inception-B (:math:`17\times17\times1024`) →
    Reduction-B → 3× Inception-C (:math:`8\times8\times1536`) → global
    average pool → dropout (keep 0.8) → linear classifier.  The stem
    itself is multi-branch: three filter-concatenation junctions take a
    :math:`299\times299` image to a :math:`35\times35\times384` grid, and
    every spatial reduction in it runs a stride-2 convolution and a
    stride-2 max pool side by side instead of choosing one.  Large
    kernels are factorised throughout — an :math:`n\times n` convolution
    becomes a :math:`1\times n` followed by an :math:`n\times1`, which
    covers the same receptive field for :math:`2n` rather than
    :math:`n^2` weights per input–output channel pair.

    The paper's point in building Inception-v4 alongside
    Inception-ResNet was a controlled comparison: with a comparable
    compute budget, residual connections speed training up markedly but
    are *not* what makes a very deep Inception network accurate —
    Inception-v4 reaches roughly the accuracy of Inception-ResNet-v2
    (single-crop top-5 error 5.0% on the ImageNet validation set,
    against 4.9%) with no shortcuts at all.
    """,
)
@dataclass(frozen=True)
class InceptionV4Config(ModelConfig):
    r"""Frozen configuration for Inception-v4.

    Inception-v4 is a single-size architecture: the paper (Figure 9)
    defines exactly one network, so this configuration carries only the
    input / output widths and the head dropout.  Every channel count and
    repeat count inside the body is fixed by the paper.

    Parameters
    ----------
    num_classes : int, optional, default=1000
        Width of the classifier head (ImageNet-1k by default).
    in_channels : int, optional, default=3
        Channels of the input image.
    dropout : float, optional, default=0.2
        Drop probability applied to the pooled features before the
        classifier.  Figure 9 specifies *keep* probability 0.8, i.e. a
        drop probability of 0.2.

    Attributes
    ----------
    model_type : str
        Registry identifier, ``"inception_v4"``.

    Notes
    -----
    This is the *TF-Slim* Inception-v4 — the network the released
    checkpoint was trained as — which follows the paper's figures in
    every channel width and repeat count.  One detail is dictated by the
    released tensors rather than by the drawings: where a figure lists an
    asymmetric factorised pair (:math:`1\times7` then :math:`7\times1`,
    or :math:`1\times3` then :math:`3\times1`) in one order and TF-Slim
    builds the other, the TF-Slim order is used, because the checkpoint's
    weight shapes fix it.

    Layout as built (``299×299`` input)::

      - Stem (Figure 3):  299×299×3  → 35×35×384
      - 4× Inception-A:   35×35×384
      - Reduction-A:      35×35×384  → 17×17×1024  (k, l, m, n = 192, 224, 256, 384)
      - 7× Inception-B:   17×17×1024
      - Reduction-B:      17×17×1024 →  8×8×1536
      - 3× Inception-C:    8×8×1536
      - AdaptiveAvgPool → Dropout(p=0.2) → Linear(1536, num_classes)

    Every convolution is followed by ``BatchNorm2d(eps=1e-3)`` (the
    TF-Slim default) and a ReLU.

    Examples
    --------
    >>> from lucid.models.vision.inception_v4 import InceptionV4Config
    >>> cfg = InceptionV4Config()
    >>> cfg.model_type
    'inception_v4'
    >>> cfg.num_classes, cfg.in_channels, cfg.dropout
    (1000, 3, 0.2)
    """

    model_type: ClassVar[str] = "inception_v4"

    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.2

    def __post_init__(self) -> None:
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
