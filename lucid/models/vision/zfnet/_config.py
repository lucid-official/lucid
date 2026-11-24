"""ZFNet configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive


@model_family_meta(
    canonical_name="ZFNet",
    citation=(
        'Zeiler, Matthew D., and Rob Fergus. "Visualizing and Understanding '
        'Convolutional Networks." European Conference on Computer Vision '
        "(ECCV), 2014."
    ),
    theory=r"""
    ZFNet is the ILSVRC-2013 winner and is best understood as an
    AlexNet that was *tuned by looking at what the network had
    learned*.  Zeiler and Fergus introduced the *deconvolutional
    network* visualisation technique — projecting individual feature-
    map activations back to the input pixel space — and used the
    resulting diagnostic images to reshape the early layers of the
    network.

    Two specific changes distinguish ZFNet from AlexNet.  The first
    convolution was reduced from an :math:`11\times11` stride-4 kernel
    to a :math:`7\times7` stride-2 kernel, addressing the observation
    that AlexNet's first layer learned a mixture of extremely
    high-frequency and dead filters.  §4.1 attributes both changes to
    layer 1 ("we (i) reduced the 1st layer filter size from 11x11 to
    7x7 and (ii) made the stride of the convolution 2, rather than 4").
    §5.1 then describes the model as having "stride 2 convolutions in
    layers 1 & 2" — conv2's stride was *raised* from AlexNet's 1 to 2,
    which is also what Figure 3's 55 -> 26 transition requires
    arithmetically with a 5x5 kernel.  The remaining layers retain
    AlexNet's topology.

    These modest topological changes — together with the *methodology*
    of using visualisation to drive architectural choices — were
    influential out of proportion to their parameter cost.  ZFNet
    reports 16.5% top-5 validation error for the Figure 3 model
    (Table 2), and the
    deconvolutional-network analysis it popularised is the conceptual
    ancestor of every later interpretability tool from Grad-CAM to
    feature inversion.
    """,
)
@dataclass(frozen=True)
class ZFNetConfig(ModelConfig):
    """Configuration for ZFNet (Zeiler & Fergus, 2013).

    ZFNet is an AlexNet variant with modified first two conv layers:
    - Conv1: 7×7 stride=2 pad=1 (vs AlexNet's 11×11 stride=4)
    - Conv2: 5×5 stride=2 (vs AlexNet's 5×5 stride=1)

    Examples
    --------
    >>> from lucid.models.vision.zfnet._config import ZFNetConfig
    >>> cfg = ZFNetConfig()
    >>> cfg.model_type
    'zfnet'
    >>> cfg.num_classes, cfg.in_channels
    (1000, 3)
    """

    model_type: ClassVar[str] = "zfnet"

    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.5

    def __post_init__(self) -> None:
        check_positive(self, "num_classes", "in_channels")
