"""Registry factories for Rectified Flow (Liu, Gong & Liu, 2023).

One factory per dataset the paper trains on, and a ``_gen`` variant of
each carrying the objective, reflow and the sampler:

    * ``rectified_flow_cifar``      — CIFAR-10 ``32 x 32``
    * ``rectified_flow_bedroom``    — LSUN Bedroom ``256 x 256``
    * ``rectified_flow_church``     — LSUN Church ``256 x 256``
    * ``rectified_flow_celeba_hq``  — CelebA-HQ ``256 x 256``
    * ``rectified_flow_afhq_cat``   — AFHQ-Cat ``256 x 256``

**On the architecture.**  Unlike the two families before it, nothing here
is inferred: the authors released their configurations, and every number
below is copied from them rather than reconstructed.  The two groups
differ by more than resolution —

    ================  ============  ==================
    setting           CIFAR-10      256 x 256
    ================  ============  ==================
    channel_mult      (1,2,2,2)     (1,1,2,2,2,2,2)
    num_res_blocks    4             2
    dropout           0.15          0.0
    fir               off           on
    progressive       none          input + output
    embedding_type    positional    fourier
    ================  ============  ==================

so the four high-resolution factories carry the filtered resampling and
the resolution pyramid, and the CIFAR one does not.

The three ``256 x 256`` LSUN/face configurations are byte-identical to one
another; they are kept separate because they name different datasets and
different reported results, exactly as the released configs do.

Smaller test models go through ``create_model("rectified_flow_cifar",
sample_size=..., base_channels=..., channel_mult=...)``.
"""

from dataclasses import replace

from lucid.models._registry import register_model
from lucid.models.generative.rectified_flow._config import RectifiedFlowConfig
from lucid.models.generative.rectified_flow._model import (
    RectifiedFlowForImageGeneration,
    RectifiedFlowModel,
)
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# 32x32 — DDPM++: no filtering, no pyramid, positional time embedding,
# four residual blocks per resolution and attention at 16x16 only.
_CFG_CIFAR = RectifiedFlowConfig(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_mult=(1, 2, 2, 2),
    num_res_blocks=4,
    attention_resolutions=(16,),
    dropout=0.15,
    fir=False,
    progressive="none",
    progressive_input="none",
    embedding_type="positional",
)

# 256x256 — NCSN++: seven stages down to 4x4, filtered resampling, the
# input fed down and the output accumulated up through a pyramid, and
# random-Fourier time features.
_CFG_HIGH_RES = RectifiedFlowConfig(
    sample_size=256,
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_mult=(1, 1, 2, 2, 2, 2, 2),
    num_res_blocks=2,
    attention_resolutions=(16,),
    dropout=0.0,
    fir=True,
    progressive="output_skip",
    progressive_input="input_skip",
    embedding_type="fourier",
    fourier_scale=16.0,
    scale_by_sigma=True,
)

# LSUN bedroom/church inherit ``data.centered = False`` from
# default_lsun_configs, so those two released nets shift [0,1] inputs to
# [-1,1] internally.  CelebA-HQ and AFHQ set it True and do not.
_CFG_LSUN = replace(_CFG_HIGH_RES, data_centered=False)


# ── Bare velocity fields ──────────────────────────────────────────────────────


@register_model(
    task="base",
    family="rectified_flow",
    model_type="rectified_flow",
    model_class=RectifiedFlowModel,
    default_config=_CFG_CIFAR,
)
def rectified_flow_cifar(pretrained: bool = False, **overrides: object) -> RectifiedFlowModel:
    r"""Rectified Flow velocity field for the CIFAR-10 setup.

    The configuration the paper's headline result was produced with.
    Trained by regressing onto the constant velocity of the straight line
    between a noise sample and a data sample; reflow and distillation are
    the same call with paired inputs and a pinned time.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``rectified_flow_cifar``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`RectifiedFlowConfig` field overrides forwarded
        into the underlying config.

    Returns
    -------
    RectifiedFlowModel
        Velocity field configured for ``32 x 32`` RGB.

    Notes
    -----
    Reference: Liu, Gong, and Liu, *"Flow Straight and Fast"*, ICLR, 2023
    (arXiv:2209.03003).  Reported on CIFAR-10: **4.85 FID with a single
    function evaluation**, state of the art among one-step diffusion and
    flow models at publication.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.rectified_flow import rectified_flow_cifar
    >>> model = rectified_flow_cifar(sample_size=8, base_channels=16,
    ...                              channel_mult=(1, 2), num_res_blocks=1,
    ...                              attention_resolutions=()).eval()
    >>> loss, _, _ = model.rectified_flow_loss(lucid.randn((2, 3, 8, 8)))
    >>> model.nfe                      # training solves nothing
    0
    """
    if pretrained:
        reject_unavailable_pretrained("rectified_flow_cifar")
    return RectifiedFlowModel(apply_overrides(_CFG_CIFAR, overrides))


@register_model(
    task="base",
    family="rectified_flow",
    model_type="rectified_flow",
    model_class=RectifiedFlowModel,
    default_config=_CFG_LSUN,
)
def rectified_flow_bedroom(pretrained: bool = False, **overrides: object) -> RectifiedFlowModel:
    r"""Rectified Flow velocity field for LSUN Bedroom ``256 x 256``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``rectified_flow_bedroom``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`RectifiedFlowConfig` field overrides.

    Returns
    -------
    RectifiedFlowModel
        Velocity field with filtered resampling and a resolution pyramid.

    Notes
    -----
    Reference: Liu, Gong, and Liu, *"Flow Straight and Fast"*, ICLR, 2023
    (arXiv:2209.03003).  The paper reports high-resolution generation
    qualitatively; no FID is quoted for this dataset.

    Examples
    --------
    >>> from lucid.models.generative.rectified_flow import rectified_flow_bedroom
    >>> rectified_flow_bedroom().config.fir
    True
    """
    if pretrained:
        reject_unavailable_pretrained("rectified_flow_bedroom")
    return RectifiedFlowModel(apply_overrides(_CFG_LSUN, overrides))


@register_model(
    task="base",
    family="rectified_flow",
    model_type="rectified_flow",
    model_class=RectifiedFlowModel,
    default_config=_CFG_LSUN,
)
def rectified_flow_church(pretrained: bool = False, **overrides: object) -> RectifiedFlowModel:
    r"""Rectified Flow velocity field for LSUN Church ``256 x 256``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``rectified_flow_church``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`RectifiedFlowConfig` field overrides.

    Returns
    -------
    RectifiedFlowModel
        Velocity field with filtered resampling and a resolution pyramid.

    Notes
    -----
    Reference: Liu, Gong, and Liu, *"Flow Straight and Fast"*, ICLR, 2023
    (arXiv:2209.03003).

    Examples
    --------
    >>> from lucid.models.generative.rectified_flow import rectified_flow_church
    >>> rectified_flow_church().config.progressive
    'output_skip'
    """
    if pretrained:
        reject_unavailable_pretrained("rectified_flow_church")
    return RectifiedFlowModel(apply_overrides(_CFG_LSUN, overrides))


@register_model(
    task="base",
    family="rectified_flow",
    model_type="rectified_flow",
    model_class=RectifiedFlowModel,
    default_config=_CFG_HIGH_RES,
)
def rectified_flow_celeba_hq(pretrained: bool = False, **overrides: object) -> RectifiedFlowModel:
    r"""Rectified Flow velocity field for CelebA-HQ ``256 x 256``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``rectified_flow_celeba_hq``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`RectifiedFlowConfig` field overrides.

    Returns
    -------
    RectifiedFlowModel
        Velocity field with filtered resampling and a resolution pyramid.

    Notes
    -----
    Reference: Liu, Gong, and Liu, *"Flow Straight and Fast"*, ICLR, 2023
    (arXiv:2209.03003).  This is one of the datasets the paper's *transfer*
    experiments run on — the method is not restricted to a Gaussian
    source, and the same objective transports between two image domains.

    Examples
    --------
    >>> from lucid.models.generative.rectified_flow import rectified_flow_celeba_hq
    >>> rectified_flow_celeba_hq().config.embedding_type
    'fourier'
    """
    if pretrained:
        reject_unavailable_pretrained("rectified_flow_celeba_hq")
    return RectifiedFlowModel(apply_overrides(_CFG_HIGH_RES, overrides))


@register_model(
    task="base",
    family="rectified_flow",
    model_type="rectified_flow",
    model_class=RectifiedFlowModel,
    default_config=_CFG_HIGH_RES,
)
def rectified_flow_afhq_cat(pretrained: bool = False, **overrides: object) -> RectifiedFlowModel:
    r"""Rectified Flow velocity field for AFHQ-Cat ``256 x 256``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``rectified_flow_afhq_cat``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`RectifiedFlowConfig` field overrides.

    Returns
    -------
    RectifiedFlowModel
        Velocity field with filtered resampling and a resolution pyramid.

    Notes
    -----
    Reference: Liu, Gong, and Liu, *"Flow Straight and Fast"*, ICLR, 2023
    (arXiv:2209.03003).  The paper's Figure 1 uses this dataset for both
    generation and face-to-face transfer.

    Examples
    --------
    >>> from lucid.models.generative.rectified_flow import rectified_flow_afhq_cat
    >>> rectified_flow_afhq_cat().config.num_res_blocks
    2
    """
    if pretrained:
        reject_unavailable_pretrained("rectified_flow_afhq_cat")
    return RectifiedFlowModel(apply_overrides(_CFG_HIGH_RES, overrides))


# ── Generators ────────────────────────────────────────────────────────────────


@register_model(
    task="image-generation",
    family="rectified_flow",
    model_type="rectified_flow",
    model_class=RectifiedFlowForImageGeneration,
    default_config=_CFG_CIFAR,
)
def rectified_flow_cifar_gen(
    pretrained: bool = False, **overrides: object
) -> RectifiedFlowForImageGeneration:
    r"""Rectified Flow generator for CIFAR-10 ``32 x 32``.

    Same field as :func:`rectified_flow_cifar`, wrapped with the
    objective, :meth:`~RectifiedFlowForImageGeneration.reflow_pairs` and
    the sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``rectified_flow_cifar_gen``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`RectifiedFlowConfig` field overrides.  Pass
        ``t_schedule="t0"`` to build the one-step distillation stage.

    Returns
    -------
    RectifiedFlowForImageGeneration
        Generator configured for CIFAR-10.

    Notes
    -----
    Reference: Liu, Gong, and Liu, *"Flow Straight and Fast"*, ICLR, 2023
    (arXiv:2209.03003).  Reported: 4.85 FID at one function evaluation.

    Examples
    --------
    >>> from lucid.models.generative.rectified_flow import rectified_flow_cifar_gen
    >>> model = rectified_flow_cifar_gen(sample_size=8, base_channels=16,
    ...                                  channel_mult=(1, 2), num_res_blocks=1,
    ...                                  attention_resolutions=()).eval()
    >>> z0, z1 = model.reflow_pairs(n_samples=2, steps=4)
    >>> model(z1, noise=z0).loss.shape          # the reflow objective
    ()
    """
    if pretrained:
        reject_unavailable_pretrained("rectified_flow_cifar_gen")
    return RectifiedFlowForImageGeneration(apply_overrides(_CFG_CIFAR, overrides))


@register_model(
    task="image-generation",
    family="rectified_flow",
    model_type="rectified_flow",
    model_class=RectifiedFlowForImageGeneration,
    default_config=_CFG_HIGH_RES,
)
def rectified_flow_bedroom_gen(
    pretrained: bool = False, **overrides: object
) -> RectifiedFlowForImageGeneration:
    r"""Rectified Flow generator for LSUN Bedroom ``256 x 256``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``rectified_flow_bedroom_gen``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`RectifiedFlowConfig` field overrides.

    Returns
    -------
    RectifiedFlowForImageGeneration
        Generator configured for LSUN Bedroom.

    Notes
    -----
    Reference: Liu, Gong, and Liu, *"Flow Straight and Fast"*, ICLR, 2023
    (arXiv:2209.03003).

    Examples
    --------
    >>> from lucid.models.generative.rectified_flow import (
    ...     rectified_flow_bedroom_gen,
    ... )
    >>> rectified_flow_bedroom_gen().rectified_flow.input_dim
    196608
    """
    if pretrained:
        reject_unavailable_pretrained("rectified_flow_bedroom_gen")
    return RectifiedFlowForImageGeneration(apply_overrides(_CFG_HIGH_RES, overrides))


@register_model(
    task="image-generation",
    family="rectified_flow",
    model_type="rectified_flow",
    model_class=RectifiedFlowForImageGeneration,
    default_config=_CFG_HIGH_RES,
)
def rectified_flow_church_gen(
    pretrained: bool = False, **overrides: object
) -> RectifiedFlowForImageGeneration:
    r"""Rectified Flow generator for LSUN Church ``256 x 256``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``rectified_flow_church_gen``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`RectifiedFlowConfig` field overrides.

    Returns
    -------
    RectifiedFlowForImageGeneration
        Generator configured for LSUN Church.

    Notes
    -----
    Reference: Liu, Gong, and Liu, *"Flow Straight and Fast"*, ICLR, 2023
    (arXiv:2209.03003).

    Examples
    --------
    >>> from lucid.models.generative.rectified_flow import (
    ...     rectified_flow_church_gen,
    ... )
    >>> rectified_flow_church_gen().rectified_flow.t_schedule
    'uniform'
    """
    if pretrained:
        reject_unavailable_pretrained("rectified_flow_church_gen")
    return RectifiedFlowForImageGeneration(apply_overrides(_CFG_HIGH_RES, overrides))


@register_model(
    task="image-generation",
    family="rectified_flow",
    model_type="rectified_flow",
    model_class=RectifiedFlowForImageGeneration,
    default_config=_CFG_HIGH_RES,
)
def rectified_flow_celeba_hq_gen(
    pretrained: bool = False, **overrides: object
) -> RectifiedFlowForImageGeneration:
    r"""Rectified Flow generator for CelebA-HQ ``256 x 256``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``rectified_flow_celeba_hq_gen``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`RectifiedFlowConfig` field overrides.

    Returns
    -------
    RectifiedFlowForImageGeneration
        Generator configured for CelebA-HQ.

    Notes
    -----
    Reference: Liu, Gong, and Liu, *"Flow Straight and Fast"*, ICLR, 2023
    (arXiv:2209.03003).

    Examples
    --------
    >>> from lucid.models.generative.rectified_flow import (
    ...     rectified_flow_celeba_hq_gen,
    ... )
    >>> rectified_flow_celeba_hq_gen().config.sample_size
    256
    """
    if pretrained:
        reject_unavailable_pretrained("rectified_flow_celeba_hq_gen")
    return RectifiedFlowForImageGeneration(apply_overrides(_CFG_HIGH_RES, overrides))


@register_model(
    task="image-generation",
    family="rectified_flow",
    model_type="rectified_flow",
    model_class=RectifiedFlowForImageGeneration,
    default_config=_CFG_HIGH_RES,
)
def rectified_flow_afhq_cat_gen(
    pretrained: bool = False, **overrides: object
) -> RectifiedFlowForImageGeneration:
    r"""Rectified Flow generator for AFHQ-Cat ``256 x 256``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``rectified_flow_afhq_cat_gen``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`RectifiedFlowConfig` field overrides.

    Returns
    -------
    RectifiedFlowForImageGeneration
        Generator configured for AFHQ-Cat.

    Notes
    -----
    Reference: Liu, Gong, and Liu, *"Flow Straight and Fast"*, ICLR, 2023
    (arXiv:2209.03003).

    Examples
    --------
    >>> from lucid.models.generative.rectified_flow import (
    ...     rectified_flow_afhq_cat_gen,
    ... )
    >>> rectified_flow_afhq_cat_gen().config.progressive_input
    'input_skip'
    """
    if pretrained:
        reject_unavailable_pretrained("rectified_flow_afhq_cat_gen")
    return RectifiedFlowForImageGeneration(apply_overrides(_CFG_HIGH_RES, overrides))


__all__ = [
    "rectified_flow_cifar",
    "rectified_flow_bedroom",
    "rectified_flow_church",
    "rectified_flow_celeba_hq",
    "rectified_flow_afhq_cat",
    "rectified_flow_cifar_gen",
    "rectified_flow_bedroom_gen",
    "rectified_flow_church_gen",
    "rectified_flow_celeba_hq_gen",
    "rectified_flow_afhq_cat_gen",
]
