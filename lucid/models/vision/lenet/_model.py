"""LeNet-5 backbone and classifier (LeCun et al., 1998).

Original paper: "Gradient-Based Learning Applied to Document Recognition"
Architecture (canonical):
    Input   : 1 × 32 × 32
    C1      : Conv 1→6,  5×5, valid  → 6  × 28 × 28
    S2      : AvgPool 2×2             → 6  × 14 × 14
    C3      : Conv 6→16, 5×5, valid  → 16 × 10 × 10  (dense, see below)
    S4      : AvgPool 2×2             → 16 × 5  × 5
    C5      : Conv 16→120, 5×5, valid → 120 × 1 × 1   (fully-connected in disguise)
    F6      : Linear 120 → 84
    Output  : Linear 84  → num_classes

Two documented divergences from the 1998 paper, both matching universal
modern practice rather than the original:

* **C3 connectivity.**  Table I gives a *sparse* connection scheme in which
  each of the 16 C3 maps reads only a chosen subset of the 6 S2 maps —
  1,516 trainable parameters.  This implementation connects C3 densely to
  all 6 inputs (2,416 parameters).  LeCun's stated motivation for the
  sparse table was to break symmetry and to keep the connection count
  within 1998 compute budgets; neither concern survives, and every
  contemporary re-implementation uses the dense convolution.
* **Squashing function.**  ``activation="tanh"`` is plain :math:`\tanh`,
  not the paper's scaled :math:`1.7159\tanh(\tfrac{2}{3}a)`.

The ``activation`` and ``pooling`` config fields let callers switch to the
modern ReLU/MaxPool convention without changing the topology.
"""

from typing import ClassVar, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.lenet._config import LeNetConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _act(name: str) -> nn.ReLU | nn.Tanh:
    if name == "relu":
        return nn.ReLU(inplace=True)
    return nn.Tanh()


def _pool(name: str) -> nn.MaxPool2d | nn.AvgPool2d:
    """Sub-sampling layer for S2 / S4.

    Section II.B's sub-sampling unit is not a plain pool: it sums the four
    inputs, scales by a *trainable* coefficient, adds a trainable bias and
    passes the result through the squashing function.  Both parameters and
    the post-pool nonlinearity are dropped here — a parameter-free
    ``AvgPool2d`` (or ``MaxPool2d``) is what every modern
    re-implementation uses, and it is why the trunk's parameter count is
    slightly below a literal reading of the paper.
    """
    if name == "max":
        return nn.MaxPool2d(2, stride=2)
    return nn.AvgPool2d(2, stride=2)


# ---------------------------------------------------------------------------
# Input geometry
# ---------------------------------------------------------------------------

#: The input side the paper builds LeNet-5 for (MNIST's 28 px, padded by 2).
_PAPER_SIDE = 32


def _c5_side(side: int) -> int:
    """C5's output side for an input side: three valid 5x5 convs, two 2x pools."""
    return ((side - 4) // 2 - 4) // 2 - 4


def _check_input_size(height: int, width: int, *, classifier: bool) -> None:
    """Refuse an input side the trunk, or the head after it, cannot take.

    The trunk's convolutions are unpadded, so a side under 32 px leaves C5
    nothing to convolve over.  The classifier needs more: F6 is a Linear
    over C5's 120 channels, so C5 has to come out 1x1, which holds for a
    side of 32 to 35 px (the pools floor, so 33-35 drop their last rows).
    Before this check both failed inside the engine (a convolution larger
    than its input, or ``W.shape[1] != x.last_dim`` at F6), naming
    neither the input size nor the rule.
    """
    sides = (_c5_side(height), _c5_side(width))
    if classifier and sides == (1, 1):
        return
    if not classifier and min(sides) >= 1:
        return
    needs = (
        "a side of 32 to 35 px, which leaves C5 at the 1x1 map F6's 120 inputs need"
        if classifier
        else "at least 32 px per side for its three unpadded 5x5 convolutions"
    )
    raise ValueError(
        f"LeNet-5 is built for {_PAPER_SIDE}x{_PAPER_SIDE} inputs and takes "
        f"{needs}.  Got {height}x{width}.  Pad or resize the input to "
        f"{_PAPER_SIDE}x{_PAPER_SIDE}, the way the paper pads MNIST's 28x28 digits."
    )


# ---------------------------------------------------------------------------
# Shared feature extractor (C1→S2→C3→S4→C5)
# ---------------------------------------------------------------------------


def _build_features(cfg: LeNetConfig) -> nn.Sequential:
    return nn.Sequential(
        # C1
        nn.Conv2d(cfg.in_channels, 6, 5),
        _act(cfg.activation),
        # S2
        _pool(cfg.pooling),
        # C3
        nn.Conv2d(6, 16, 5),
        _act(cfg.activation),
        # S4
        _pool(cfg.pooling),
        # C5 — acts as a fully-connected conv over the 5×5 feature map
        nn.Conv2d(16, 120, 5),
        _act(cfg.activation),
    )


# ---------------------------------------------------------------------------
# LeNet backbone  (task="base")
# ---------------------------------------------------------------------------


class LeNet(PretrainedModel, BackboneMixin):
    r"""LeNet-5 feature extractor (no fully-connected head).

    Implements the canonical convolutional trunk of the original LeNet-5
    network from LeCun et al., "Gradient-Based Learning Applied to
    Document Recognition", Proc. IEEE 1998: three convolutional layers
    (C1, C3, C5) interleaved with two sub-sampling layers (S2, S4).  The
    receptive field grows from a single :math:`5\times5` patch in C1 to
    the entire :math:`32\times32` input by C5 — so for the canonical
    input size the C5 output collapses to a :math:`1\times1` spatial map
    of 120 channels, acting as a fully-connected layer in conv form.
    Used as the canonical "Hello World" of convolutional networks and
    as the smallest paper-cited baseline in the model zoo.

    Parameters
    ----------
    config : LeNetConfig
        Frozen architecture spec.  Use :func:`lenet_5` for the
        paper-cited tanh + average-pool variant; pass a custom config to
        switch to modern ``activation="relu"`` / ``pooling="max"`` or to
        accept RGB input via ``in_channels=3``.

    Attributes
    ----------
    config : LeNetConfig
        Stored copy of the config that built this model.
    features : nn.Sequential
        The C1-S2-C3-S4-C5 convolutional stack — see
        :func:`_build_features` for the exact layer chain.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed via
        :class:`BackboneMixin` for downstream decoder modules.  The
        reductions are measured on the canonical 32×32 input, where S2, S4
        and C5 emit 14×14, 5×5 and 1×1 maps; the unpadded convolutions make
        them depend on the input size.

    Notes
    -----
    From LeCun et al., "Gradient-Based Learning Applied to Document
    Recognition", Proc. IEEE 86(11):2278–2324, 1998, Figure 2.  Each
    convolution computes

    .. math::

        h_{i,j}^{(k)} = \phi\!\left(
            \sum_{c}\sum_{u,v} W_{u,v,c}^{(k)} \, x_{i+u,\,j+v,\,c}
            + b^{(k)}
        \right),

    where :math:`\phi` is :math:`\tanh` in the paper.  The original
    sub-sampling layer was a *trainable* :math:`2\times2` average pool
    with learnable scale and bias; this implementation uses standard
    parameter-free :class:`~lucid.nn.AvgPool2d` for compatibility with
    modern training recipes.  Total parameter count is approximately
    60 k — small enough to train to convergence on MNIST in minutes on
    a CPU.

    Examples
    --------
    Run a single forward pass on a batch of MNIST-shaped inputs:

    >>> import lucid
    >>> from lucid.models.vision.lenet import lenet_5
    >>> backbone = lenet_5()
    >>> x = lucid.randn(8, 1, 32, 32)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape   # (B, 120, 1, 1)
    (8, 120, 1, 1)

    Each stage's ``reduction`` is the 32-pixel input over the side of the map
    it emits (14, 5 and 1), so the last stage spans the whole input.

    >>> [f.reduction for f in backbone.feature_info]
    [2, 6, 32]
    """

    config_class: ClassVar[type[LeNetConfig]] = LeNetConfig
    base_model_prefix: ClassVar[str] = "lenet"

    def __init__(self, config: LeNetConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        # Measured on the paper's 32x32 input, where S2, S4 and C5 emit 14x14,
        # 5x5 and 1x1 maps.  Every convolution is valid (unpadded), so a map
        # shrinks faster than the two 2x subsamplings alone make it: counting
        # strides (2, 4, 4) would promise an 8x8 C5 map where there is one
        # pixel.  ``reduction`` is ``32 // side``, the rule the rest of the
        # zoo follows (the exact ratios 2.29 and 6.4 are not integers), and
        # it holds for 32x32 only -- unpadded convolutions make the ratio
        # depend on the input size.
        self._feature_info = [
            FeatureInfo(stage=1, num_channels=6, reduction=2),
            FeatureInfo(stage=2, num_channels=16, reduction=6),
            FeatureInfo(stage=3, num_channels=120, reduction=32),
        ]

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        _check_input_size(int(x.shape[-2]), int(x.shape[-1]), classifier=False)
        return self.features(x)

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# LeNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class LeNetForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""LeNet-5 with F6 fully-connected layer and final classifier.

    Combines a :class:`LeNet` convolutional trunk with the original
    paper's two-layer fully-connected head: a hidden layer F6 mapping
    120 → 84 followed by the output layer mapping 84 →
    ``config.num_classes``.  When ``labels`` are supplied to
    :meth:`forward`, a cross-entropy loss is returned alongside the
    logits; otherwise ``loss`` is ``None``.  Used as a teaching baseline
    on MNIST / Fashion-MNIST and as a parity reference for LeNet-style
    architectures.

    Parameters
    ----------
    config : LeNetConfig
        Architecture spec.  Use :func:`lenet_5_cls` for the paper-cited
        configuration (10-way classifier, tanh, average pooling); pass a
        custom config to retarget ``num_classes`` or to switch the
        activation/pooling family.

    Attributes
    ----------
    config : LeNetConfig
        Stored copy of the config that built this model.
    features : nn.Sequential
        The C1-S2-C3-S4-C5 convolutional stack inherited from the
        backbone.
    f6 : nn.Linear
        Hidden fully-connected layer projecting 120 → 84.
    act_f6 : nn.Module
        Activation applied after F6 — :class:`~lucid.nn.Tanh` in the
        paper, :class:`~lucid.nn.ReLU` if ``config.activation="relu"``.
    classifier : nn.Module
        Final linear projection 84 → ``num_classes``, built by
        :meth:`ClassificationHeadMixin._build_classifier`.

    Notes
    -----
    From LeCun et al., "Gradient-Based Learning Applied to Document
    Recognition", Proc. IEEE 86(11):2278–2324, 1998, Figure 2.  The
    original output layer was a Gaussian-RBF unit comparing F6
    activations against fixed digit templates; this implementation uses
    the modern linear + cross-entropy head, which gave essentially the
    same accuracy in subsequent reimplementations.  Loss is the standard
    categorical cross-entropy

    .. math::

        \mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N}
            \log \operatorname{softmax}(\text{logits}_n)_{\,y_n},

    computed only when ``labels`` is not ``None``.

    Examples
    --------
    Run inference on a small MNIST-style batch:

    >>> import lucid
    >>> from lucid.models.vision.lenet import lenet_5_cls
    >>> model = lenet_5_cls()
    >>> x = lucid.randn(4, 1, 32, 32)
    >>> out = model(x)
    >>> out.logits.shape
    (4, 10)
    >>> out.loss is None
    True

    Compute a training loss given labels:

    >>> labels = lucid.tensor([0, 1, 2, 3], dtype=lucid.int64)
    >>> out = model(x, labels=labels)
    >>> out.loss.shape
    ()
    """

    config_class: ClassVar[type[LeNetConfig]] = LeNetConfig
    base_model_prefix: ClassVar[str] = "lenet"

    def __init__(self, config: LeNetConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.f6 = nn.Linear(120, 84)
        self.act_f6 = _act(config.activation)
        self._build_classifier(84, config.num_classes)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        _check_input_size(int(x.shape[-2]), int(x.shape[-1]), classifier=True)
        x = self.features(x)
        # C5 output is (B, 120, 1, 1) — flatten spatial dims
        x = x.flatten(1)
        x = self.act_f6(self.f6(x))
        logits = self.classifier(x)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
