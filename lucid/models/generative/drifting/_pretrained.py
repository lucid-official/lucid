"""Registry factories for the Drifting Model family.

The four system-level generators the paper reports on ImageNet 256x256 —
``B/2`` and ``L/2`` in the SD-VAE latent space (Table 5), ``B/16`` and
``L/16`` in pixel space (Table 6) — each as a bare generator and as a
``_gen`` model that also carries the drifting objective.  H10 forbids the
paper's single-letter sizes, so ``B/2`` is ``drifting_base_2`` and
``L/16`` is ``drifting_large_16``.

No checkpoint is published for any of them, so ``pretrained=True`` is refused.

The ablation default of Table 8 is not a fifth factory: it *is* the
``B/2`` architecture, trained shorter against the narrower MAE-256 with
different guidance sampling, which ``drifting_base_2_gen(attn_fp32=True,
mae_base_channels=256, neg_cfg_pw=3.0, neg_per_sample=16)`` configures.

Each ``_CFG_*`` constant is the matching ``configs/gen/*.yaml`` of the
official release, field for field — ``model``, ``train.forward_dict``,
``train.loss_kwargs``, ``train.activation_kwargs``, the sample-queue sizes
and ``feature``.  The ``params`` are the paper's (Tables 5 and 6), which
round the built counts: ``B/2`` is 132,708,880.
"""

from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.generative.drifting._config import DriftingConfig
from lucid.models.generative.drifting._model import DriftingForImageGeneration, DriftingModel

__all__ = [
    "drifting_base_2",
    "drifting_large_2",
    "drifting_base_16",
    "drifting_large_16",
    "drifting_base_2_gen",
    "drifting_large_2_gen",
    "drifting_base_16_gen",
    "drifting_large_16_gen",
]

# configs/gen/latent_sota_B.yaml
_CFG_BASE_2 = DriftingConfig(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    patch_size=2,
    hidden_size=768,
    cond_dim=768,
    depth=12,
    num_heads=12,
    attn_fp32=False,
    neg_cfg_pw=5.0,
    no_cfg_frac=0.0,
    pos_per_sample=64,
    neg_per_sample=32,
    mae_base_channels=640,
    mae_input_patch_size=1,
)
# configs/gen/latent_sota_L.yaml — half the batch trains without guidance.
_CFG_LARGE_2 = DriftingConfig(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    patch_size=2,
    hidden_size=1024,
    cond_dim=1024,
    depth=24,
    num_heads=16,
    attn_fp32=False,
    neg_cfg_pw=3.0,
    no_cfg_frac=0.5,
    pos_per_sample=64,
    neg_per_sample=32,
    mae_base_channels=640,
    mae_input_patch_size=1,
)
# configs/gen/pixel_sota_B.yaml — MAE-640 (pixel) plus ConvNeXt-V2.
_CFG_BASE_16 = DriftingConfig(
    sample_size=256,
    in_channels=3,
    out_channels=3,
    patch_size=16,
    hidden_size=768,
    cond_dim=768,
    depth=12,
    num_heads=12,
    attn_fp32=True,
    use_remat=True,
    neg_cfg_pw=5.0,
    no_cfg_frac=0.0,
    pos_per_sample=128,
    neg_per_sample=32,
    use_convnext=True,
    mae_base_channels=640,
    mae_input_patch_size=8,
)
# configs/gen/pixel_sota_L.yaml
_CFG_LARGE_16 = DriftingConfig(
    sample_size=256,
    in_channels=3,
    out_channels=3,
    patch_size=16,
    hidden_size=1024,
    cond_dim=1024,
    depth=24,
    num_heads=16,
    attn_fp32=True,
    use_remat=True,
    neg_cfg_pw=5.0,
    no_cfg_frac=0.0,
    pos_per_sample=128,
    neg_per_sample=32,
    use_convnext=True,
    mae_base_channels=640,
    mae_input_patch_size=8,
)


@register_model(
    task="base",
    family="drifting",
    model_type="drifting",
    model_class=DriftingModel,
    default_config=_CFG_BASE_2,
    params=133_000_000,
    summary="auto",
)
def drifting_base_2(pretrained: bool = False, **overrides: object) -> DriftingModel:
    """Drifting ``B/2`` — the Base generator over the 32x32x4 SD-VAE latent.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``drifting_base_2``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        :class:`DriftingConfig` fields to change on the ``latent_sota_B``
        release config.

    Returns
    -------
    DriftingModel
        The ``B/2`` one-step generator; it emits an SD-VAE latent.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, *"Generative Modeling via
    Drifting"*, arXiv:2602.04770, 2026, Table 8 (``B/2, latent``) and
    Table 5 — 12 blocks, width 768, 12 heads, 16 in-context tokens,
    32 style tokens.  The paper reports FID 1.75 at one evaluation, guidance 1.1.

    Examples
    --------
    >>> from lucid.models import drifting_base_2
    >>> config = drifting_base_2().config
    >>> config.depth, config.hidden_size, config.patch_size
    (12, 768, 2)
    """
    if pretrained:
        reject_unavailable_pretrained("drifting_base_2")
    return DriftingModel(apply_overrides(_CFG_BASE_2, overrides))


@register_model(
    task="base",
    family="drifting",
    model_type="drifting",
    model_class=DriftingModel,
    default_config=_CFG_LARGE_2,
    params=463_000_000,
    summary="auto",
)
def drifting_large_2(pretrained: bool = False, **overrides: object) -> DriftingModel:
    """Drifting ``L/2`` — the Large latent generator, the paper's headline model.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``drifting_large_2``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        :class:`DriftingConfig` fields to change on the ``latent_sota_L``
        release config.

    Returns
    -------
    DriftingModel
        The ``L/2`` one-step generator; it emits an SD-VAE latent.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Table 8
    (``L/2, latent``) and Table 5 — 24 blocks, width 1024, 16 heads.
    The paper reports FID 1.54 at one evaluation, guidance 1.0.

    Examples
    --------
    >>> from lucid.models import drifting_large_2
    >>> config = drifting_large_2().config
    >>> config.depth, config.hidden_size, config.no_cfg_frac
    (24, 1024, 0.5)
    """
    if pretrained:
        reject_unavailable_pretrained("drifting_large_2")
    return DriftingModel(apply_overrides(_CFG_LARGE_2, overrides))


@register_model(
    task="base",
    family="drifting",
    model_type="drifting",
    model_class=DriftingModel,
    default_config=_CFG_BASE_16,
    params=134_000_000,
    summary="auto",
)
def drifting_base_16(pretrained: bool = False, **overrides: object) -> DriftingModel:
    """Drifting ``B/16`` — the Base generator directly in 256x256 pixel space.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``drifting_base_16``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        :class:`DriftingConfig` fields to change on the ``pixel_sota_B``
        release config.

    Returns
    -------
    DriftingModel
        The ``B/16`` one-step generator; it emits ``[-1, 1]`` RGB.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Table 8
    (``B/16, pixel``) and Table 6 — the ``B/2`` backbone at patch 16, so
    the sequence is again 256 tokens.  The paper reports FID 1.76, guidance 1.0.

    Examples
    --------
    >>> from lucid.models import drifting_base_16
    >>> config = drifting_base_16().config
    >>> config.sample_size, config.patch_size, config.num_patches
    (256, 16, 256)
    """
    if pretrained:
        reject_unavailable_pretrained("drifting_base_16")
    return DriftingModel(apply_overrides(_CFG_BASE_16, overrides))


@register_model(
    task="base",
    family="drifting",
    model_type="drifting",
    model_class=DriftingModel,
    default_config=_CFG_LARGE_16,
    params=464_000_000,
    summary="auto",
)
def drifting_large_16(pretrained: bool = False, **overrides: object) -> DriftingModel:
    """Drifting ``L/16`` — the Large generator in pixel space.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``drifting_large_16``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        :class:`DriftingConfig` fields to change on the ``pixel_sota_L``
        release config.

    Returns
    -------
    DriftingModel
        The ``L/16`` one-step generator; it emits ``[-1, 1]`` RGB.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Table 8
    (``L/16, pixel``) and Table 6.  The paper reports FID 1.61, guidance 1.0.

    Examples
    --------
    >>> from lucid.models import drifting_large_16
    >>> config = drifting_large_16().config
    >>> config.depth, config.hidden_size, config.use_convnext
    (24, 1024, True)
    """
    if pretrained:
        reject_unavailable_pretrained("drifting_large_16")
    return DriftingModel(apply_overrides(_CFG_LARGE_16, overrides))


@register_model(
    task="image-generation",
    family="drifting",
    model_type="drifting",
    model_class=DriftingForImageGeneration,
    default_config=_CFG_BASE_2,
    params=133_000_000,
    summary="auto",
)
def drifting_base_2_gen(
    pretrained: bool = False, **overrides: object
) -> DriftingForImageGeneration:
    """:func:`drifting_base_2` with the drifting objective and the one-step sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``drifting_base_2_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        :class:`DriftingConfig` fields to change, the loss's and the
        sample queue's included.

    Returns
    -------
    DriftingForImageGeneration
        The ``B/2`` training objective over latent-MAE features, and
        :meth:`~DriftingForImageGeneration.generate`.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Table 8.

    Examples
    --------
    >>> from lucid.models import drifting_base_2_gen
    >>> model = drifting_base_2_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("drifting_base_2_gen")
    return DriftingForImageGeneration(apply_overrides(_CFG_BASE_2, overrides))


@register_model(
    task="image-generation",
    family="drifting",
    model_type="drifting",
    model_class=DriftingForImageGeneration,
    default_config=_CFG_LARGE_2,
    params=463_000_000,
    summary="auto",
)
def drifting_large_2_gen(
    pretrained: bool = False, **overrides: object
) -> DriftingForImageGeneration:
    """:func:`drifting_large_2` with the drifting objective and the one-step sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``drifting_large_2_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        :class:`DriftingConfig` fields to change on ``latent_sota_L``; its
        ``no_cfg_frac=0.5`` is the half of the batch trained unguided.

    Returns
    -------
    DriftingForImageGeneration
        The ``L/2`` objective and one-step sampler.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Table 8.
    The configuration behind the 1.54 one-step FID.

    Examples
    --------
    >>> from lucid.models import drifting_large_2_gen
    >>> model = drifting_large_2_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("drifting_large_2_gen")
    return DriftingForImageGeneration(apply_overrides(_CFG_LARGE_2, overrides))


@register_model(
    task="image-generation",
    family="drifting",
    model_type="drifting",
    model_class=DriftingForImageGeneration,
    default_config=_CFG_BASE_16,
    params=134_000_000,
    summary="auto",
)
def drifting_base_16_gen(
    pretrained: bool = False, **overrides: object
) -> DriftingForImageGeneration:
    """:func:`drifting_base_16` with the drifting objective and the one-step sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``drifting_base_16_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        :class:`DriftingConfig` fields to change on ``pixel_sota_B`` —
        ``use_convnext=False`` drops the ConvNeXt-V2 term.

    Returns
    -------
    DriftingForImageGeneration
        The ``B/16`` pixel-space objective and one-step sampler.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Tables 8
    and 10.  Its loss reads MAE-640 (pixel) and ConvNeXt-V2-Base features.

    Examples
    --------
    >>> from lucid.models import drifting_base_16_gen
    >>> model = drifting_base_16_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("drifting_base_16_gen")
    return DriftingForImageGeneration(apply_overrides(_CFG_BASE_16, overrides))


@register_model(
    task="image-generation",
    family="drifting",
    model_type="drifting",
    model_class=DriftingForImageGeneration,
    default_config=_CFG_LARGE_16,
    params=464_000_000,
    summary="auto",
)
def drifting_large_16_gen(
    pretrained: bool = False, **overrides: object
) -> DriftingForImageGeneration:
    """:func:`drifting_large_16` with the drifting objective and the one-step sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``drifting_large_16_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        :class:`DriftingConfig` fields to change on ``pixel_sota_L``.

    Returns
    -------
    DriftingForImageGeneration
        The ``L/16`` objective — MAE-640 (pixel) plus ConvNeXt-V2-Base
        features — and its one-step sampler.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Tables 8
    and 10.

    Examples
    --------
    >>> from lucid.models import drifting_large_16_gen
    >>> model = drifting_large_16_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("drifting_large_16_gen")
    return DriftingForImageGeneration(apply_overrides(_CFG_LARGE_16, overrides))
