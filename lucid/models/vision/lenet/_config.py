"""LeNet-5 configuration (LeCun et al., 1998)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="LeNet",
    citation=(
        'LeCun, Yann, et al. "Gradient-based learning applied to document '
        'recognition." Proceedings of the IEEE, vol. 86, no. 11, 1998, '
        "pp. 2278–2324."
    ),
    theory=r"""
    LeNet-5 is the prototypical convolutional neural network and the
    direct ancestor of every modern vision architecture.  Designed for
    handwritten-digit recognition on 32×32 grayscale inputs, it
    interleaves *learned* :math:`5\times5` convolutions with fixed
    :math:`2\times2` sub-sampling (average pooling) layers, ending in
    two fully-connected layers and an output layer.  (The 1998 paper's
    output layer is a set of Gaussian RBF units scoring against fixed
    ASCII bitmap prototypes; this implementation uses an ordinary linear
    layer trained with cross-entropy, which is the universal modern
    substitute.)

    The key insight of LeCun et al. was that local receptive fields,
    shared weights, and spatial sub-sampling — together — give a model
    that is approximately invariant to small translations, scalings,
    and distortions of the input.  Convolution enforces *weight
    sharing*: the same :math:`5\times5` filter slides across the image,
    so a feature detector learned in one location applies everywhere.
    Sub-sampling layers then *pool* responses over a neighbourhood,
    yielding coarser feature maps that are robust to the exact position
    of each feature.

    Formally, a LeNet layer computes

    .. math::

        h_{i,j}^{(k)} = \phi\!\left(
            \sum_{c}\sum_{u,v} W_{u,v,c}^{(k)} \, x_{i+u,\,j+v,\,c}
            + b^{(k)}
        \right),

    where :math:`\phi` is :math:`\tanh` in the original paper (or ReLU
    in modern reimplementations) and the kernel :math:`W^{(k)}` is
    shared across all spatial positions :math:`(i, j)`.  Despite its
    small size by today's standards (≈60 k parameters), LeNet
    established the convolutional template — conv → nonlinearity →
    pool → repeat → flatten → MLP — that is still recognisable in
    every CNN that followed.
    """,
)
@dataclass(frozen=True)
class LeNetConfig(ModelConfig):
    r"""Configuration for LeNet-5.

    ``activation`` controls the nonlinearity:
      - ``"tanh"``    — plain :math:`\tanh`.  The 1998 paper's squashing
        function is the scaled :math:`f(a) = 1.7159\tanh(\tfrac{2}{3}a)`
        (Appendix A), chosen so :math:`f(\pm 1) = \pm 1`; plain tanh is
        what every modern re-implementation uses and what the shipped
        weights were trained with.
      - ``"relu"``    — modern convention

    ``pooling`` controls the sub-sampling layers:
      - ``"avg"``     — original paper (average pooling / sub-sampling)
      - ``"max"``     — modern convention

    ``in_channels`` defaults to 1 (grayscale). Set to 3 for RGB inputs,
    though the canonical use-case is MNIST / single-channel images.

    Examples
    --------
    >>> from lucid.models.vision.lenet._config import LeNetConfig
    >>> cfg = LeNetConfig()
    >>> cfg.model_type
    'lenet'
    >>> cfg.num_classes, cfg.in_channels
    (10, 1)
    """

    model_type: ClassVar[str] = "lenet"

    num_classes: int = 10
    in_channels: int = 1
    activation: str = "tanh"  # "tanh" | "relu"
    pooling: str = "avg"  # "avg" | "max"

    def __post_init__(self) -> None:
        # The builders test only for the non-default literal and fall back to
        # the default for anything else, so ``activation="ReLU"`` (wrong case)
        # or ``pooling="none"`` silently produced the *default* network — the
        # opposite of what the caller asked for, with no warning.
        if self.activation not in ("tanh", "relu"):
            raise ValueError(f"activation must be 'tanh' or 'relu', got {self.activation!r}")
        if self.pooling not in ("avg", "max"):
            raise ValueError(f"pooling must be 'avg' or 'max', got {self.pooling!r}")
