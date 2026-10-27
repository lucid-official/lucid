"""MobileNet v4 configuration (Qin et al., 2024)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta

#: The five architectures specified in the paper's Appendix D
#: (Tables 11-15), in the order the paper lists them.
MOBILENET_V4_VARIANTS: tuple[str, ...] = (
    "conv_small",
    "conv_medium",
    "conv_large",
    "hybrid_medium",
    "hybrid_large",
)


@model_family_meta(
    canonical_name="MobileNet-v4",
    citation=(
        'Qin, Danfeng, et al. "MobileNetV4: Universal Models for the Mobile '
        'Ecosystem." Computer Vision – ECCV 2024, Lecture Notes in Computer '
        "Science, vol. 15098, Springer, 2024."
    ),
    theory=r"""
    MobileNet-v4 is built around a single searchable block, the
    **Universal Inverted Bottleneck** (UIB).  It extends the inverted
    bottleneck of MobileNet-v2 with two *optional* depthwise convolutions:
    one before the :math:`1 \times 1` expansion and one between the
    expansion and the :math:`1 \times 1` projection,

    .. math::

        \mathrm{UIB}(x) = x + \gamma \odot
            P\bigl(D_{\mathrm{mid}}(E(D_{\mathrm{start}}(x)))\bigr),

    where :math:`E` is the pointwise expansion, :math:`P` the linear
    pointwise projection, and each :math:`D` is either a depthwise
    convolution or the identity.  The four on/off combinations recover four
    familiar blocks: the classic inverted bottleneck (**IB**, middle
    depthwise only), a **ConvNeXt-like** block (start depthwise only, a
    cheap large-kernel spatial mix before the expansion), a transformer
    **FFN** (neither — two pointwise layers), and the new **ExtraDW**
    block (both), which deepens the network and widens its receptive field
    at almost no cost.  Because the expansion and projection are shared by
    all four instantiations, a NAS super-network over UIB shares more than
    95% of its parameters, and the search simply decides which depthwise
    layers to keep at each position.  The stems use a *fused* inverted
    bottleneck — a dense :math:`3 \times 3` convolution into a
    :math:`1 \times 1` projection — which is faster than a depthwise pair
    at high resolution.

    The hybrid variants interleave UIB blocks with **Mobile MQA**, a
    multi-query attention block tuned for accelerators: every query head
    shares a single key and value head, which raises operational intensity
    when the token count is small relative to the channel width, and the
    keys and values can be spatially reduced by a stride-2
    :math:`3 \times 3` depthwise convolution,

    .. math::

        \mathrm{MQA}(X) = \mathrm{Concat}_{j=1}^{h}\!\left[
            \mathrm{softmax}\!\left(
                \frac{(X W^{Q}_{j})\,(\mathrm{SR}(X) W^{K})^{\top}}
                     {\sqrt{d_k}}\right)
            \mathrm{SR}(X) W^{V}\right] W^{O}.

    Sharing keys and values cuts attention latency by more than 39% on
    mobile accelerators relative to multi-head attention at a negligible
    accuracy cost.  Residual branches of the hybrids are scaled by a
    learnable per-channel layer scale :math:`\gamma` initialised at
    :math:`10^{-5}`.

    Five models come out of the search: **Conv-Small**, **Conv-Medium**
    and **Conv-Large**, built only from UIB and fused-IB blocks, and
    **Hybrid-Medium** and **Hybrid-Large**, which add Mobile MQA to the
    last two stages.  Every model ends with the MobileNet-v3 style head
    that moves the widest :math:`1 \times 1` layer after global pooling.
    They are mostly Pareto-optimal across mobile CPUs, DSPs, GPUs, the
    Apple Neural Engine and the Pixel EdgeTPU at once, ranging from 73.8%
    ImageNet-1k top-1 at 0.2 GMACs (Conv-Small) to 83.4% (Hybrid-Large).
    """,
)
@dataclass(frozen=True)
class MobileNetV4Config(ModelConfig):
    r"""Frozen configuration for every MobileNet-v4 variant.

    One dataclass describes the whole family; ``variant`` selects which of
    the paper's five searched architectures (Appendix D, Tables 11-15) is
    built, and the remaining fields cover the classifier and regularisation
    knobs that are not part of the searched topology.

    Parameters
    ----------
    num_classes : int, optional, default=1000
        Output classes of the classification head.
    in_channels : int, optional, default=3
        Channels of the input image.
    variant : str, optional, default="conv_small"
        Which searched architecture to build — one of ``"conv_small"``,
        ``"conv_medium"``, ``"conv_large"``, ``"hybrid_medium"`` or
        ``"hybrid_large"``.  The variant fixes the block sequence, the
        stem width (24 for the Large models, 32 otherwise), the activation
        (GELU for Hybrid-Large, ReLU for the rest) and whether residual
        branches carry a layer scale (hybrids only).
    dropout : float, optional, default=0.3
        Dropout probability before the final ``Linear``.  The paper trains
        Conv-Small with 0.3 and every other variant with 0.2 (Table 10);
        the factories set the matching value.
    drop_path_rate : float, optional, default=0.0
        Peak stochastic-depth rate.  Block ``i`` of ``n`` drops its
        residual branch with probability ``drop_path_rate * i / n``.  The
        paper's recipe peaks at 0 (Conv-Small), 0.075 (Medium) and 0.35
        (Large) during training (Table 10); the default leaves it off so
        the model behaves deterministically unless a recipe asks for it.

    Attributes
    ----------
    model_type : str
        Registry id, ``"mobilenet_v4"``.

    Raises
    ------
    ValueError
        If ``variant`` is not one of the five paper variants, or a rate
        lies outside :math:`[0, 1)`.

    Notes
    -----
    Block sequences follow the authors' released implementation, which is
    what the published checkpoints were trained against.  The four other
    variants match Tables 11, 12, 14 and 15 block for block.  For
    Hybrid-Medium the released model orders several UIB blocks differently
    from Table 13 and carries one extra block in the stride-32 stage; its
    parameter count (10.56 M in units of :math:`2^{20}`) is the 10.5 M
    that Table 6 reports.

    Examples
    --------
    >>> from lucid.models.vision.mobilenet_v4 import MobileNetV4Config
    >>> cfg = MobileNetV4Config()
    >>> cfg.model_type
    'mobilenet_v4'
    >>> cfg.variant, cfg.num_classes
    ('conv_small', 1000)
    >>> MobileNetV4Config(variant="hybrid_large").variant
    'hybrid_large'
    """

    model_type: ClassVar[str] = "mobilenet_v4"

    num_classes: int = 1000
    in_channels: int = 3
    variant: str = "conv_small"
    dropout: float = 0.3
    drop_path_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.variant not in MOBILENET_V4_VARIANTS:
            raise ValueError(
                f"MobileNetV4Config.variant must be one of "
                f"{MOBILENET_V4_VARIANTS}, got {self.variant!r}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"MobileNetV4Config.dropout must lie in [0, 1), got {self.dropout}")
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ValueError(
                f"MobileNetV4Config.drop_path_rate must lie in [0, 1), got "
                f"{self.drop_path_rate}"
            )
