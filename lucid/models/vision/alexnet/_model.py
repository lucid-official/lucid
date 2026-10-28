"""AlexNet backbone and classifier — "One Weird Trick" single-stream variant.

Implements the architecture used in Krizhevsky 2014 ("One weird trick for
parallelizing convolutional neural networks", arXiv:1404.5997), which is
the single-stream, no-LRN re-derivation of the original Krizhevsky 2012
two-GPU AlexNet ("ImageNet Classification with Deep Convolutional Neural
Networks", NIPS 2012).  The 2014 single-stream channel widths are what
the canonical reference-framework checkpoint targets, so they are what
Lucid ships — letting ``alexnet_cls(pretrained=True)`` load directly.

Architecture::

    Conv1 : 3→64,   11×11, stride=4, pad=2  → ReLU → MaxPool 3×3 s2
    Conv2 : 64→192,  5×5,  pad=2            → ReLU → MaxPool 3×3 s2
    Conv3 : 192→384, 3×3,  pad=1            → ReLU
    Conv4 : 384→256, 3×3,  pad=1            → ReLU
    Conv5 : 256→256, 3×3,  pad=1            → ReLU → MaxPool 3×3 s2
    AdaptiveAvgPool → 6×6
    Dropout → FC6 : 256*6*6 → 4096          → ReLU
    Dropout → FC7 : 4096    → 4096          → ReLU
    FC8   : 4096    → num_classes

Two deliberate divergences from arXiv:1404.5997, both inherited from the
canonical reference implementation these weights come from:

* **conv4 width.**  Footnote 1 of the 2014 paper gives conv4 384 filters;
  the widths built here are 64/192/384/256/256, so conv4 is 256.  That is
  what the reference implementation has always used and what the converted
  checkpoint's tensors are shaped for — changing it would break every
  shipped weight for a number no released model was ever trained with.
* **Local response normalisation.**  The 2012 paper introduced LRN; the
  2014 paper does not discuss normalisation at all, so it cannot be cited
  as having removed it.  The omission here follows the reference
  implementation, which drops LRN because later work found it adds compute
  without measurable accuracy gain once dropout, ReLU and heavy
  augmentation are present.
"""

from typing import ClassVar, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.alexnet._config import AlexNetConfig


def _build_features(cfg: AlexNetConfig) -> nn.Sequential:
    # Single-stream channel widths (64/192/384/256/256), no LRN.  Both
    # choices follow the reference implementation rather than the 2014
    # paper -- see the module docstring for why.
    # Indices in the resulting Sequential land at {0, 3, 6, 8, 10} for the
    # five convolutions — matching the reference-framework state_dict so
    # the converted checkpoint loads with a direct ``features.N.*`` map.
    return nn.Sequential(
        # Block 1
        nn.Conv2d(cfg.in_channels, 64, 11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2),
        # Block 2
        nn.Conv2d(64, 192, 5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2),
        # Block 3
        nn.Conv2d(192, 384, 3, padding=1),
        nn.ReLU(inplace=True),
        # Block 4
        nn.Conv2d(384, 256, 3, padding=1),
        nn.ReLU(inplace=True),
        # Block 5
        nn.Conv2d(256, 256, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2),
    )


# ---------------------------------------------------------------------------
# AlexNet backbone  (task="base")
# ---------------------------------------------------------------------------


def _init_paper_weights(model: nn.Module) -> None:
    r"""AlexNet's own initialisation (Krizhevsky et al., 2012, section 5).

    "We initialized the weights in each layer from a zero-mean Gaussian
    distribution with standard deviation 0.01.  We initialized the neuron
    biases in the second, fourth, and fifth convolutional layers, as well as
    in the fully-connected hidden layers, with the constant 1.  This
    initialization accelerates the early stages of learning by providing the
    ReLUs with positive inputs.  We initialized the neuron biases in the
    remaining layers with the constant 0."

    The bias-of-one on exactly those layers is the point: it is what keeps
    their ReLUs on at step 0.  Lucid's default gives every conv a
    ``kaiming_uniform(a=sqrt(5))`` weight and a uniform bias, so neither the
    scale nor the positive-input trick survives.
    """
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]

    for m in convs + linears:
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # Conv 2, 4 and 5 (1-indexed as the paper counts them).
    for idx in (1, 3, 4):
        if idx < len(convs):
            cbias = convs[idx].bias
            if cbias is not None:
                nn.init.ones_(cbias)
    # The fully-connected *hidden* layers — every Linear but the classifier.
    for lin in linears[:-1]:
        lbias = lin.bias
        if lbias is not None:
            nn.init.ones_(lbias)


class AlexNet(PretrainedModel, BackboneMixin):
    r"""AlexNet feature-extracting backbone (no fully-connected head).

    Implements the single-stream, no-LRN derivation from Krizhevsky,
    "One weird trick for parallelizing convolutional neural networks",
    arXiv:1404.5997 — the canonical re-derivation of the original
    Krizhevsky, Sutskever & Hinton 2012 two-GPU model into a single
    merged stream with adjusted channel widths
    :math:`(64, 192, 384, 256, 256)`.  Five convolutions
    (:math:`11\times11` stride-4 first; :math:`5\times5` second; three
    :math:`3\times3`), each followed by ReLU, with overlapping
    :math:`3\times3` stride-2 max-pools after blocks 1, 2, and 5.  A
    final :class:`~lucid.nn.AdaptiveAvgPool2d` collapses the feature
    map to :math:`6\times6` regardless of input resolution.

    Parameters
    ----------
    config : AlexNetConfig
        Frozen architecture spec.  Use :func:`alexnet` for the
        paper-cited single-stream configuration; pass a custom config
        to switch input channel count or to retarget the classifier
        variant.

    Attributes
    ----------
    config : AlexNetConfig
        Stored copy of the config that built this model.
    features : nn.Sequential
        The five conv blocks (Conv → ReLU → optional MaxPool) — see
        :func:`_build_features` for the exact ordering.
    avgpool : nn.AdaptiveAvgPool2d
        Global pool down to a :math:`6\times6` spatial map so the
        backbone produces a fixed-size feature regardless of input
        resolution.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed via
        :class:`BackboneMixin` for downstream decoder modules.

    Notes
    -----
    From Krizhevsky 2014, §3.  AlexNet's contribution to deep-learning
    history is threefold: the *rectified linear unit*
    :math:`\phi(x) = \max(0, x)` replaced saturating nonlinearities and
    cut training time by several factors; *dropout* with :math:`p=0.5`
    regularised the 4096-dim fully-connected layers against overfitting
    on a 1.2 M-image dataset; and *heavy data augmentation* (random
    crop, horizontal flip, AlexNet-style PCA colour jitter) was made
    central to the recipe.  The 2012 paper additionally used local
    response normalisation between blocks 1-2; this implementation omits
    it, following the reference implementation (the 2014 paper does not
    discuss normalisation).  The classifier variant has
    61,100,840 parameters, of which 58.6 M sit in the *three*
    fully-connected layers; the two 4096-dim hidden ones alone account
    for 54.5 M.  With the original ImageNet-1k
    training recipe the single-stream AlexNet reaches roughly 56.5%
    top-1 / 79.1% top-5 on the validation split.

    Examples
    --------
    Build the backbone and run a single forward pass:

    >>> import lucid
    >>> from lucid.models.vision.alexnet import alexnet
    >>> backbone = alexnet()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape   # (B, 256, 6, 6)
    (2, 256, 6, 6)

    Inspect per-stage feature descriptors:

    >>> info = backbone.feature_info
    >>> [(fi.stage, fi.num_channels, fi.reduction) for fi in info[:2]]
    [(1, 64, 4), (2, 192, 8)]
    """

    config_class: ClassVar[type[AlexNetConfig]] = AlexNetConfig
    base_model_prefix: ClassVar[str] = "alexnet"

    def __init__(self, config: AlexNetConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self._feature_info = [
            FeatureInfo(stage=1, num_channels=64, reduction=4),
            FeatureInfo(stage=2, num_channels=192, reduction=8),
            FeatureInfo(stage=3, num_channels=384, reduction=16),
            FeatureInfo(stage=4, num_channels=256, reduction=16),
            # conv5 emits a 13x13 map from a 224 input — stride 16.  The
            # /32 only appears after the following max-pool, which is not
            # part of this stage.
            FeatureInfo(stage=5, num_channels=256, reduction=16),
        ]

        _init_paper_weights(self)

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.avgpool(x)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# AlexNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class AlexNetForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""AlexNet with two 4096-dim fully-connected layers and a linear classifier head.

    Combines an :class:`AlexNet` convolutional backbone with the
    paper-cited three-layer classifier head: FC6 (256·6·6 → 4096),
    FC7 (4096 → 4096), and the final linear projection to
    ``config.num_classes``.  :class:`~lucid.nn.Dropout` with
    ``config.dropout`` is applied after both ReLU activations in the
    hidden layers — these two large FC layers dominate the parameter
    count and are the main overfitting risk that dropout was introduced
    to control.  When ``labels`` are supplied to :meth:`forward`, a
    cross-entropy loss is returned alongside the logits.

    Parameters
    ----------
    config : AlexNetConfig
        Architecture spec.  Use :func:`alexnet_cls` for the paper-cited
        ImageNet-1k configuration (1000-class head, ``dropout=0.5``).

    Attributes
    ----------
    config : AlexNetConfig
        Stored copy of the config that built this model.
    features, avgpool
        Same backbone components as on :class:`AlexNet`; see that class
        for shape semantics.
    fc6, fc7 : nn.Linear
        The two hidden fully-connected layers, both projecting to 4096
        dimensions.
    drop6, drop7 : nn.Dropout
        :class:`~lucid.nn.Dropout` layers applied before each FC in
        the hidden FC stack, controlled by ``config.dropout``.
    classifier : nn.Module
        Final linear projection to ``num_classes``, built by
        :meth:`ClassificationHeadMixin._build_classifier`.

    Notes
    -----
    From Krizhevsky 2014 (single-stream re-derivation of NIPS 2012).
    The two 4096-dim hidden layers alone account for roughly 54.5 M of
    the network's 61.1 M parameters — the original rationale for
    *dropout*, which randomly zeros out half of each FC activation
    during training so that no individual co-adapted neuron is critical
    for any single decision.  Loss is the standard cross-entropy

    .. math::

        \mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N}
            \log \operatorname{softmax}(\text{logits}_n)_{\,y_n}.

    Examples
    --------
    Run inference on a batch of 224x224 RGB images:

    >>> import lucid
    >>> from lucid.models.vision.alexnet import alexnet_cls
    >>> model = alexnet_cls()
    >>> x = lucid.randn(4, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (4, 1000)
    >>> out.loss is None
    True

    Compute a training loss given integer labels:

    >>> labels = lucid.tensor([0, 1, 2, 3], dtype=lucid.int64)
    >>> out = model(x, labels=labels)
    >>> out.loss.shape
    ()
    """

    config_class: ClassVar[type[AlexNetConfig]] = AlexNetConfig
    base_model_prefix: ClassVar[str] = "alexnet"

    def __init__(self, config: AlexNetConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.drop6 = nn.Dropout(p=config.dropout)
        self.drop7 = nn.Dropout(p=config.dropout)
        self._build_classifier(4096, config.num_classes)

        _init_paper_weights(self)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        # Dropout goes *in front of* each FC, matching the reference
        # classifier's Dropout -> Linear -> ReLU ordering.  Dropping after the
        # ReLU instead zeroes post-activation units, so the layer that follows
        # never sees the noise it is supposed to be regularised against; the
        # two orderings differ only in training mode, which is why eval-time
        # parity never caught it.
        x = F.relu(self.fc6(self.drop6(x)))
        x = F.relu(self.fc7(self.drop7(x)))
        return self._classify(x, labels)
