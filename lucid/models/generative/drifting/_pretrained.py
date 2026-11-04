"""Registry factories for the Drifting Model family.

The four system-level generators the paper reports on ImageNet 256x256 —
``B/2`` and ``L/2`` in the SD-VAE latent space (Table 5), ``B/16`` and
``L/16`` in pixel space (Table 6) — each as a bare generator and as a
``_gen`` model that also carries the drifting objective.  H10 forbids the
paper's single-letter sizes, so ``B/2`` is ``drifting_base_2`` and
``L/16`` is ``drifting_large_16``.

The ablation default of Table 8 is not a fifth factory: it *is* the
``B/2`` architecture, trained shorter against the narrower MAE-256 with
different guidance sampling.  It ships as the ``IMAGENET1K_256_ABLATION``
tag of :class:`DriftingBase2Weights`, whose ``config`` overrides the
factory applies.

Each ``_CFG_*`` constant is the matching ``configs/gen/*.yaml`` of the
official release, field for field — ``model``, ``train.forward_dict``,
``train.loss_kwargs``, ``train.activation_kwargs``, the sample-queue sizes
and ``feature``.  Parameter counts are the paper's (Tables 5 and 6) and
agree with the converted checkpoints: ``B/2`` is 132,708,880.
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides
from lucid.models.generative.drifting._config import DriftingConfig
from lucid.models.generative.drifting._model import DriftingForImageGeneration, DriftingModel
from lucid.models.generative.drifting._weights import (
    DriftingBase2Weights,
    DriftingBase16Weights,
    DriftingLarge2Weights,
    DriftingLarge16Weights,
)
from lucid.weights import WeightsEnum

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


def _tag_config(entry: WeightsEnum | None, overrides: dict[str, object]) -> dict[str, object]:
    """Merge a weight tag's own config under the caller's overrides.

    The ablation checkpoint shares ``B/2``'s architecture but was trained
    with a different feature encoder and guidance sampling; its tag records
    those fields so a loaded model reproduces its training objective.  An
    explicit override still wins.

    Parameters
    ----------
    entry : WeightsEnum or None
        The resolved weight tag, or ``None`` when building untrained.
    overrides : dict
        Config overrides the caller passed.

    Returns
    -------
    dict
        ``overrides`` on top of the tag's ``config``, if it has one.
    """
    if entry is None:
        return overrides
    tagged = entry.value.meta.get("config")
    return overrides if not isinstance(tagged, dict) else {**tagged, **overrides}


def _trunk(
    name: str,
    cfg: DriftingConfig,
    enum_cls: type[WeightsEnum],
    pretrained: bool | str,
    weights: WeightsEnum | None,
    overrides: dict[str, object],
) -> DriftingModel:
    entry = weights_mod.resolve_weights(enum_cls, pretrained, weights)
    model = DriftingModel(apply_overrides(cfg, _tag_config(entry, overrides)))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name=name)
    return model


def _gen(
    name: str,
    cfg: DriftingConfig,
    enum_cls: type[WeightsEnum],
    pretrained: bool | str,
    weights: WeightsEnum | None,
    overrides: dict[str, object],
) -> DriftingForImageGeneration:
    entry = weights_mod.resolve_weights(enum_cls, pretrained, weights)
    model = DriftingForImageGeneration(apply_overrides(cfg, _tag_config(entry, overrides)))
    if entry is not None:
        # The checkpoint holds the generator, which this wrapper keeps under
        # ``drifting`` — the feature encoder loads separately, on request.
        weights_mod.load_weight_entry(model.drifting, entry, name=name)
    return model


@register_model(
    task="base",
    family="drifting",
    model_type="drifting",
    model_class=DriftingModel,
    default_config=_CFG_BASE_2,
    params=133_000_000,
    summary="auto",
)
def drifting_base_2(
    pretrained: bool | str = False,
    *,
    weights: DriftingBase2Weights | None = None,
    **overrides: object,
) -> DriftingModel:
    """Drifting ``B/2`` — the Base generator over the 32x32x4 SD-VAE latent.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Load a released checkpoint.  ``True`` takes ``IMAGENET1K_256`` (the
        Table 5 model); ``"IMAGENET1K_256_ABLATION"`` takes the Table 8
        ablation default and applies the config it was trained with.
    weights : DriftingBase2Weights or None, optional, keyword-only
        An explicit tag, taking precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DriftingConfig` field overrides.

    Returns
    -------
    DriftingModel
        The one-step generator.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, *"Generative Modeling via
    Drifting"*, arXiv:2602.04770, 2026, Table 8 (``B/2, latent``) and
    Table 5 — 12 blocks, width 768, 12 heads, 16 in-context tokens,
    32 style tokens.  FID 1.75 at one evaluation, guidance 1.1.

    Examples
    --------
    >>> from lucid.models import drifting_base_2
    >>> config = drifting_base_2().config
    >>> config.depth, config.hidden_size, config.patch_size
    (12, 768, 2)
    """
    return _trunk(
        "drifting_base_2", _CFG_BASE_2, DriftingBase2Weights, pretrained, weights, overrides
    )


@register_model(
    task="base",
    family="drifting",
    model_type="drifting",
    model_class=DriftingModel,
    default_config=_CFG_LARGE_2,
    params=463_000_000,
    summary="auto",
)
def drifting_large_2(
    pretrained: bool | str = False,
    *,
    weights: DriftingLarge2Weights | None = None,
    **overrides: object,
) -> DriftingModel:
    """Drifting ``L/2`` — the Large latent generator, the paper's headline model.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Load the released checkpoint.  ``True`` takes ``IMAGENET1K_256``.
    weights : DriftingLarge2Weights or None, optional, keyword-only
        An explicit tag, taking precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DriftingConfig` field overrides.

    Returns
    -------
    DriftingModel
        The one-step generator.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Table 8
    (``L/2, latent``) and Table 5 — 24 blocks, width 1024, 16 heads.
    FID 1.54 at one evaluation, guidance 1.0.

    Examples
    --------
    >>> from lucid.models import drifting_large_2
    >>> config = drifting_large_2().config
    >>> config.depth, config.hidden_size, config.no_cfg_frac
    (24, 1024, 0.5)
    """
    return _trunk(
        "drifting_large_2", _CFG_LARGE_2, DriftingLarge2Weights, pretrained, weights, overrides
    )


@register_model(
    task="base",
    family="drifting",
    model_type="drifting",
    model_class=DriftingModel,
    default_config=_CFG_BASE_16,
    params=134_000_000,
    summary="auto",
)
def drifting_base_16(
    pretrained: bool | str = False,
    *,
    weights: DriftingBase16Weights | None = None,
    **overrides: object,
) -> DriftingModel:
    """Drifting ``B/16`` — the Base generator directly in 256x256 pixel space.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Load the released checkpoint.  ``True`` takes ``IMAGENET1K_256``.
    weights : DriftingBase16Weights or None, optional, keyword-only
        An explicit tag, taking precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DriftingConfig` field overrides.

    Returns
    -------
    DriftingModel
        The one-step generator; it emits ``[-1, 1]`` RGB.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Table 8
    (``B/16, pixel``) and Table 6 — the ``B/2`` backbone at patch 16, so
    the sequence is again 256 tokens.  FID 1.76, guidance 1.0.

    Examples
    --------
    >>> from lucid.models import drifting_base_16
    >>> config = drifting_base_16().config
    >>> config.sample_size, config.patch_size, config.num_patches
    (256, 16, 256)
    """
    return _trunk(
        "drifting_base_16", _CFG_BASE_16, DriftingBase16Weights, pretrained, weights, overrides
    )


@register_model(
    task="base",
    family="drifting",
    model_type="drifting",
    model_class=DriftingModel,
    default_config=_CFG_LARGE_16,
    params=464_000_000,
    summary="auto",
)
def drifting_large_16(
    pretrained: bool | str = False,
    *,
    weights: DriftingLarge16Weights | None = None,
    **overrides: object,
) -> DriftingModel:
    """Drifting ``L/16`` — the Large generator in pixel space.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Load the released checkpoint.  ``True`` takes ``IMAGENET1K_256``.
    weights : DriftingLarge16Weights or None, optional, keyword-only
        An explicit tag, taking precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DriftingConfig` field overrides.

    Returns
    -------
    DriftingModel
        The one-step generator; it emits ``[-1, 1]`` RGB.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Table 8
    (``L/16, pixel``) and Table 6.  FID 1.61, guidance 1.0.

    Examples
    --------
    >>> from lucid.models import drifting_large_16
    >>> config = drifting_large_16().config
    >>> config.depth, config.hidden_size, config.use_convnext
    (24, 1024, True)
    """
    return _trunk(
        "drifting_large_16", _CFG_LARGE_16, DriftingLarge16Weights, pretrained, weights, overrides
    )


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
    pretrained: bool | str = False,
    *,
    weights: DriftingBase2Weights | None = None,
    **overrides: object,
) -> DriftingForImageGeneration:
    """:func:`drifting_base_2` with the drifting objective and the one-step sampler.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Load a released generator checkpoint into ``model.drifting``.
        ``True`` takes ``IMAGENET1K_256``; ``"IMAGENET1K_256_ABLATION"``
        takes the ablation default.  The feature encoder is loaded
        separately by :meth:`DriftingForImageGeneration.load_feature_encoder`.
    weights : DriftingBase2Weights or None, optional, keyword-only
        An explicit tag, taking precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DriftingConfig` field overrides.

    Returns
    -------
    DriftingForImageGeneration
        Exposes the training objective and :meth:`generate`.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Table 8.

    Examples
    --------
    >>> from lucid.models import drifting_base_2_gen
    >>> model = drifting_base_2_gen()  # doctest: +SKIP
    """
    return _gen(
        "drifting_base_2", _CFG_BASE_2, DriftingBase2Weights, pretrained, weights, overrides
    )


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
    pretrained: bool | str = False,
    *,
    weights: DriftingLarge2Weights | None = None,
    **overrides: object,
) -> DriftingForImageGeneration:
    """:func:`drifting_large_2` with the drifting objective and the one-step sampler.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Load the released generator checkpoint into ``model.drifting``.
        ``True`` takes ``IMAGENET1K_256``.
    weights : DriftingLarge2Weights or None, optional, keyword-only
        An explicit tag, taking precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DriftingConfig` field overrides.

    Returns
    -------
    DriftingForImageGeneration
        Exposes the training objective and :meth:`generate`.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Table 8.
    The configuration behind the 1.54 one-step FID.

    Examples
    --------
    >>> from lucid.models import drifting_large_2_gen
    >>> model = drifting_large_2_gen()  # doctest: +SKIP
    """
    return _gen(
        "drifting_large_2", _CFG_LARGE_2, DriftingLarge2Weights, pretrained, weights, overrides
    )


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
    pretrained: bool | str = False,
    *,
    weights: DriftingBase16Weights | None = None,
    **overrides: object,
) -> DriftingForImageGeneration:
    """:func:`drifting_base_16` with the drifting objective and the one-step sampler.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Load the released generator checkpoint into ``model.drifting``.
        ``True`` takes ``IMAGENET1K_256``.
    weights : DriftingBase16Weights or None, optional, keyword-only
        An explicit tag, taking precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DriftingConfig` field overrides.

    Returns
    -------
    DriftingForImageGeneration
        Exposes the training objective and :meth:`generate`.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Tables 8
    and 10.  Its loss reads MAE-640 (pixel) and ConvNeXt-V2-Base features.

    Examples
    --------
    >>> from lucid.models import drifting_base_16_gen
    >>> model = drifting_base_16_gen()  # doctest: +SKIP
    """
    return _gen(
        "drifting_base_16", _CFG_BASE_16, DriftingBase16Weights, pretrained, weights, overrides
    )


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
    pretrained: bool | str = False,
    *,
    weights: DriftingLarge16Weights | None = None,
    **overrides: object,
) -> DriftingForImageGeneration:
    """:func:`drifting_large_16` with the drifting objective and the one-step sampler.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Load the released generator checkpoint into ``model.drifting``.
        ``True`` takes ``IMAGENET1K_256``.
    weights : DriftingLarge16Weights or None, optional, keyword-only
        An explicit tag, taking precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DriftingConfig` field overrides.

    Returns
    -------
    DriftingForImageGeneration
        Exposes the training objective and :meth:`generate`.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Tables 8
    and 10.

    Examples
    --------
    >>> from lucid.models import drifting_large_16_gen
    >>> model = drifting_large_16_gen()  # doctest: +SKIP
    """
    return _gen(
        "drifting_large_16", _CFG_LARGE_16, DriftingLarge16Weights, pretrained, weights, overrides
    )
