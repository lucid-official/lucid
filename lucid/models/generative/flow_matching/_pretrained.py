"""Registry factories for Flow Matching (Lipman et al., 2023).

One factory per dataset the paper reports in Table 1:

    * ``flow_matching_cifar``       — CIFAR-10 32x32
    * ``flow_matching_imagenet32``  — ImageNet 32x32
    * ``flow_matching_imagenet64``  — ImageNet 64x64
    * ``flow_matching_imagenet128`` — ImageNet 128x128

and a ``_gen`` variant of each carrying the training objective and the
sampler.

**On the architecture.**  The paper states it uses the U-Net of Dhariwal
and Nichol (2021) "with minimal changes" and publishes no per-dataset
table for these runs, so there is no capacity figure here to cite from
it.  What is citable is the second paper's stated default — 128 base
channels, two residual blocks per resolution, and multi-resolution
attention with 64 channels per head — and that is what every config below
uses.  The four differ only in ``sample_size`` and in how many stages the
ladder needs: the multipliers are set so the encoder bottoms out at
``8 x 8``, which is what the cited attention resolutions ``(32, 16, 8)``
require, not a published table.

Smaller test models go through ``create_model("flow_matching_cifar",
sample_size=..., base_channels=..., channel_mult=...)``.
"""

from lucid.models._registry import register_model
from lucid.models.generative.flow_matching._config import FlowMatchingConfig
from lucid.models.generative.flow_matching._model import (
    FlowMatchingForImageGeneration,
    FlowMatchingModel,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# 32x32 — four stages, 32 → 16 → 8 → 4, attention at 32/16/8.
_CFG_CIFAR = FlowMatchingConfig(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_mult=(1, 2, 2, 2),
    num_res_blocks=2,
    attention_resolutions=(32, 16, 8),
    num_head_channels=64,
)

# ImageNet at 32x32 shares the ladder; kept separate so the reported
# numbers stay attached to the dataset they belong to.
_CFG_IMAGENET32 = FlowMatchingConfig(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_mult=(1, 2, 2, 2),
    num_res_blocks=2,
    attention_resolutions=(32, 16, 8),
    num_head_channels=64,
)

# 64x64 — one more stage so the bottleneck is still 8x8.
_CFG_IMAGENET64 = FlowMatchingConfig(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_mult=(1, 2, 3, 4),
    num_res_blocks=2,
    attention_resolutions=(32, 16, 8),
    num_head_channels=64,
)

# 128x128 — five stages, 128 → 64 → 32 → 16 → 8.
_CFG_IMAGENET128 = FlowMatchingConfig(
    sample_size=128,
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_mult=(1, 1, 2, 3, 4),
    num_res_blocks=2,
    attention_resolutions=(32, 16, 8),
    num_head_channels=64,
)


# ── Bare velocity fields ──────────────────────────────────────────────────────


@register_model(
    task="base",
    family="flow_matching",
    model_type="flow_matching",
    model_class=FlowMatchingModel,
    default_config=_CFG_CIFAR,
)
def flow_matching_cifar(pretrained: bool = False, **overrides: object) -> FlowMatchingModel:
    r"""Flow Matching velocity field for the CIFAR-10 setup.

    Trained by regressing onto the optimal-transport conditional field —
    no ODE is solved during training, which is the method's whole point.
    The solver appears only when sampling or scoring likelihood.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``flow_matching_cifar``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`FlowMatchingConfig` field overrides forwarded
        into the underlying config.

    Returns
    -------
    FlowMatchingModel
        Velocity field configured for ``32 x 32`` RGB.

    Notes
    -----
    Reference: Lipman, Chen, Ben-Hamu, Nickel, and Le, *"Flow Matching
    for Generative Modeling"*, ICLR, 2023 (arXiv:2210.02747), Table 1.
    Reported for the optimal-transport path: **6.35 FID / 2.99 bits per
    dimension**.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.flow_matching import flow_matching_cifar
    >>> model = flow_matching_cifar(base_channels=16, channel_mult=(1, 2),
    ...                             num_res_blocks=1, sample_size=8,
    ...                             attention_resolutions=(),
    ...                             resnet_groups=8).eval()
    >>> loss, _, _ = model.flow_matching_loss(lucid.randn((2, 3, 8, 8)))
    >>> model.nfe                      # training solves nothing
    0
    """
    if pretrained:
        reject_unavailable_pretrained("flow_matching_cifar")
    return FlowMatchingModel(apply_overrides(_CFG_CIFAR, overrides))


@register_model(
    task="base",
    family="flow_matching",
    model_type="flow_matching",
    model_class=FlowMatchingModel,
    default_config=_CFG_IMAGENET32,
)
def flow_matching_imagenet32(pretrained: bool = False, **overrides: object) -> FlowMatchingModel:
    r"""Flow Matching velocity field for ImageNet ``32 x 32``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``flow_matching_imagenet32``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`FlowMatchingConfig` field overrides.

    Returns
    -------
    FlowMatchingModel
        Velocity field configured for ImageNet at ``32 x 32``.

    Notes
    -----
    Reference: Lipman et al., *"Flow Matching for Generative Modeling"*,
    ICLR, 2023 (arXiv:2210.02747), Table 1.  Reported: **5.02 FID / 3.53
    bits per dimension**.

    Examples
    --------
    >>> from lucid.models.generative.flow_matching import flow_matching_imagenet32
    >>> flow_matching_imagenet32().path
    'ot'
    """
    if pretrained:
        reject_unavailable_pretrained("flow_matching_imagenet32")
    return FlowMatchingModel(apply_overrides(_CFG_IMAGENET32, overrides))


@register_model(
    task="base",
    family="flow_matching",
    model_type="flow_matching",
    model_class=FlowMatchingModel,
    default_config=_CFG_IMAGENET64,
)
def flow_matching_imagenet64(pretrained: bool = False, **overrides: object) -> FlowMatchingModel:
    r"""Flow Matching velocity field for ImageNet ``64 x 64``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``flow_matching_imagenet64``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`FlowMatchingConfig` field overrides.

    Returns
    -------
    FlowMatchingModel
        Velocity field configured for ImageNet at ``64 x 64``.

    Notes
    -----
    Reference: Lipman et al., *"Flow Matching for Generative Modeling"*,
    ICLR, 2023 (arXiv:2210.02747), Table 1.  Reported: **14.45 FID / 3.31
    bits per dimension**.

    Examples
    --------
    >>> from lucid.models.generative.flow_matching import flow_matching_imagenet64
    >>> flow_matching_imagenet64().input_dim
    12288
    """
    if pretrained:
        reject_unavailable_pretrained("flow_matching_imagenet64")
    return FlowMatchingModel(apply_overrides(_CFG_IMAGENET64, overrides))


@register_model(
    task="base",
    family="flow_matching",
    model_type="flow_matching",
    model_class=FlowMatchingModel,
    default_config=_CFG_IMAGENET128,
)
def flow_matching_imagenet128(pretrained: bool = False, **overrides: object) -> FlowMatchingModel:
    r"""Flow Matching velocity field for ImageNet ``128 x 128``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``flow_matching_imagenet128``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`FlowMatchingConfig` field overrides.

    Returns
    -------
    FlowMatchingModel
        Velocity field configured for ImageNet at ``128 x 128``.

    Notes
    -----
    Reference: Lipman et al., *"Flow Matching for Generative Modeling"*,
    ICLR, 2023 (arXiv:2210.02747), Table 1.  Reported: **20.9 FID / 2.90
    bits per dimension** — the best likelihood of the four, on the
    largest images.

    Examples
    --------
    >>> from lucid.models.generative.flow_matching import flow_matching_imagenet128
    >>> flow_matching_imagenet128().config.sample_size
    128
    """
    if pretrained:
        reject_unavailable_pretrained("flow_matching_imagenet128")
    return FlowMatchingModel(apply_overrides(_CFG_IMAGENET128, overrides))


# ── Generators ────────────────────────────────────────────────────────────────


@register_model(
    task="image-generation",
    family="flow_matching",
    model_type="flow_matching",
    model_class=FlowMatchingForImageGeneration,
    default_config=_CFG_CIFAR,
)
def flow_matching_cifar_gen(
    pretrained: bool = False, **overrides: object
) -> FlowMatchingForImageGeneration:
    r"""Flow Matching generator for CIFAR-10 ``32 x 32``.

    Same field as :func:`flow_matching_cifar`, wrapped with the
    Conditional Flow Matching objective and the sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``flow_matching_cifar_gen``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`FlowMatchingConfig` field overrides.

    Returns
    -------
    FlowMatchingForImageGeneration
        Generator configured for CIFAR-10.

    Notes
    -----
    Reference: Lipman et al., *"Flow Matching for Generative Modeling"*,
    ICLR, 2023 (arXiv:2210.02747).  Reported: 6.35 FID / 2.99 bits/dim.

    Examples
    --------
    >>> from lucid.models.generative.flow_matching import flow_matching_cifar_gen
    >>> model = flow_matching_cifar_gen(sample_size=8, base_channels=16,
    ...                                 channel_mult=(1, 2), num_res_blocks=1,
    ...                                 attention_resolutions=(),
    ...                                 resnet_groups=8).eval()
    >>> model.generate(n_samples=2, steps=4).samples.shape
    (2, 3, 8, 8)
    """
    if pretrained:
        reject_unavailable_pretrained("flow_matching_cifar_gen")
    return FlowMatchingForImageGeneration(apply_overrides(_CFG_CIFAR, overrides))


@register_model(
    task="image-generation",
    family="flow_matching",
    model_type="flow_matching",
    model_class=FlowMatchingForImageGeneration,
    default_config=_CFG_IMAGENET32,
)
def flow_matching_imagenet32_gen(
    pretrained: bool = False, **overrides: object
) -> FlowMatchingForImageGeneration:
    r"""Flow Matching generator for ImageNet ``32 x 32``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``flow_matching_imagenet32_gen``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`FlowMatchingConfig` field overrides.

    Returns
    -------
    FlowMatchingForImageGeneration
        Generator configured for ImageNet at ``32 x 32``.

    Notes
    -----
    Reference: Lipman et al., *"Flow Matching for Generative Modeling"*,
    ICLR, 2023 (arXiv:2210.02747).  Reported: 5.02 FID / 3.53 bits/dim.

    Examples
    --------
    >>> from lucid.models.generative.flow_matching import (
    ...     flow_matching_imagenet32_gen,
    ... )
    >>> flow_matching_imagenet32_gen().flow_matching.path
    'ot'
    """
    if pretrained:
        reject_unavailable_pretrained("flow_matching_imagenet32_gen")
    return FlowMatchingForImageGeneration(apply_overrides(_CFG_IMAGENET32, overrides))


@register_model(
    task="image-generation",
    family="flow_matching",
    model_type="flow_matching",
    model_class=FlowMatchingForImageGeneration,
    default_config=_CFG_IMAGENET64,
)
def flow_matching_imagenet64_gen(
    pretrained: bool = False, **overrides: object
) -> FlowMatchingForImageGeneration:
    r"""Flow Matching generator for ImageNet ``64 x 64``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``flow_matching_imagenet64_gen``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`FlowMatchingConfig` field overrides.

    Returns
    -------
    FlowMatchingForImageGeneration
        Generator configured for ImageNet at ``64 x 64``.

    Notes
    -----
    Reference: Lipman et al., *"Flow Matching for Generative Modeling"*,
    ICLR, 2023 (arXiv:2210.02747).  Reported: 14.45 FID / 3.31 bits/dim.

    Examples
    --------
    >>> from lucid.models.generative.flow_matching import (
    ...     flow_matching_imagenet64_gen,
    ... )
    >>> flow_matching_imagenet64_gen().flow_matching.input_dim
    12288
    """
    if pretrained:
        reject_unavailable_pretrained("flow_matching_imagenet64_gen")
    return FlowMatchingForImageGeneration(apply_overrides(_CFG_IMAGENET64, overrides))


@register_model(
    task="image-generation",
    family="flow_matching",
    model_type="flow_matching",
    model_class=FlowMatchingForImageGeneration,
    default_config=_CFG_IMAGENET128,
)
def flow_matching_imagenet128_gen(
    pretrained: bool = False, **overrides: object
) -> FlowMatchingForImageGeneration:
    r"""Flow Matching generator for ImageNet ``128 x 128``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``flow_matching_imagenet128_gen``, so
        ``True`` raises ``NotImplementedError`` rather than return randomly
        initialised weights.
    **overrides : object
        Optional :class:`FlowMatchingConfig` field overrides.

    Returns
    -------
    FlowMatchingForImageGeneration
        Generator configured for ImageNet at ``128 x 128``.

    Notes
    -----
    Reference: Lipman et al., *"Flow Matching for Generative Modeling"*,
    ICLR, 2023 (arXiv:2210.02747).  Reported: 20.9 FID / 2.90 bits/dim.

    Examples
    --------
    >>> from lucid.models.generative.flow_matching import (
    ...     flow_matching_imagenet128_gen,
    ... )
    >>> flow_matching_imagenet128_gen().flow_matching.config.sample_size
    128
    """
    if pretrained:
        reject_unavailable_pretrained("flow_matching_imagenet128_gen")
    return FlowMatchingForImageGeneration(apply_overrides(_CFG_IMAGENET128, overrides))


__all__ = [
    "flow_matching_cifar",
    "flow_matching_imagenet32",
    "flow_matching_imagenet64",
    "flow_matching_imagenet128",
    "flow_matching_cifar_gen",
    "flow_matching_imagenet32_gen",
    "flow_matching_imagenet64_gen",
    "flow_matching_imagenet128_gen",
]
