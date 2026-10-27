"""Registry factories for DiT.

The twelve models of the scaling study: four backbone widths from Table 1
crossed with the three patch sizes the paper adds to the design space.
H10 forbids the paper's single-letter shorthand, so ``DiT-XL/2`` is
``dit_xlarge_2`` here.

No parameter counts are registered.  The paper's own point is that
parameters do *not* determine quality — holding them fixed while
shrinking the patch improves FID substantially, because what changes is
Gflops.  Table 1 quotes Gflops rather than parameter counts for exactly
that reason, and a ``params`` pill would advertise the number the paper
argues against.  ``summary="auto"`` reports what is actually built.

The 512-pixel model is the same ``XL/2`` config over a 64-wide latent —
``dit_xlarge_2(sample_size=64)`` — not a separate variant, since the
paper states it uses "identical hyperparameters as the 256x256 model".
"""

import lucid.weights as weights_mod
from lucid.models.generative.dit._weights import DiTXLarge2Weights
from lucid.weights import WeightsEnum

from lucid.models._registry import register_model
from lucid.models.generative.dit._config import DiTConfig
from lucid.models.generative.dit._model import DiTForImageGeneration, DiTModel
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

__all__ = [
    "dit_small_2",
    "dit_small_4",
    "dit_small_8",
    "dit_base_2",
    "dit_base_4",
    "dit_base_8",
    "dit_large_2",
    "dit_large_4",
    "dit_large_8",
    "dit_xlarge_2",
    "dit_xlarge_4",
    "dit_xlarge_8",
    "dit_small_2_gen",
    "dit_small_4_gen",
    "dit_small_8_gen",
    "dit_base_2_gen",
    "dit_base_4_gen",
    "dit_base_8_gen",
    "dit_large_2_gen",
    "dit_large_4_gen",
    "dit_large_8_gen",
    "dit_xlarge_2_gen",
    "dit_xlarge_4_gen",
    "dit_xlarge_8_gen",
]


_CFG_SMALL_2 = DiTConfig(patch_size=2, hidden_size=384, depth=12, num_heads=6)
_CFG_SMALL_4 = DiTConfig(patch_size=4, hidden_size=384, depth=12, num_heads=6)
_CFG_SMALL_8 = DiTConfig(patch_size=8, hidden_size=384, depth=12, num_heads=6)
_CFG_BASE_2 = DiTConfig(patch_size=2, hidden_size=768, depth=12, num_heads=12)
_CFG_BASE_4 = DiTConfig(patch_size=4, hidden_size=768, depth=12, num_heads=12)
_CFG_BASE_8 = DiTConfig(patch_size=8, hidden_size=768, depth=12, num_heads=12)
_CFG_LARGE_2 = DiTConfig(patch_size=2, hidden_size=1024, depth=24, num_heads=16)
_CFG_LARGE_4 = DiTConfig(patch_size=4, hidden_size=1024, depth=24, num_heads=16)
_CFG_LARGE_8 = DiTConfig(patch_size=8, hidden_size=1024, depth=24, num_heads=16)
_CFG_XLARGE_2 = DiTConfig(patch_size=2, hidden_size=1152, depth=28, num_heads=16)
_CFG_XLARGE_4 = DiTConfig(patch_size=4, hidden_size=1152, depth=28, num_heads=16)
_CFG_XLARGE_8 = DiTConfig(patch_size=8, hidden_size=1152, depth=28, num_heads=16)


def _latent_for(entry: WeightsEnum | None, overrides: dict[str, object]) -> dict[str, object]:
    """Add the checkpoint's latent size to ``overrides`` when one applies.

    The two released checkpoints are the same architecture over
    different latents — 32 a side for the 256-pixel model, 64 for the
    512-pixel one — and the positional table is built from that number.
    Loading the 512 weights into a model configured for 32 would fail on
    a shape mismatch, so the tag carries the size and it is applied here
    rather than left for the caller to remember.

    An explicit ``sample_size`` override wins: someone who states a size
    means it, and the strict load will tell them if it disagrees with
    the weights.

    Parameters
    ----------
    entry : WeightsEnum or None
        The resolved weight tag, or ``None`` when building untrained.
    overrides : dict
        Config overrides the caller passed.

    Returns
    -------
    dict
        ``overrides``, plus ``sample_size`` when a tag supplied one.
    """
    if entry is None or "sample_size" in overrides:
        return overrides
    latent = entry.value.meta.get("latent_size")
    return overrides if latent is None else {**overrides, "sample_size": latent}


@register_model(
    task="base",
    family="dit",
    model_type="dit",
    model_class=DiTModel,
    default_config=_CFG_SMALL_2,
    summary="auto",
)
def dit_small_2(pretrained: bool = False, **overrides: object) -> DiTModel:
    """DiT-S/2 — the S backbone at patch 2.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTModel
        The denoising network.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748), Table 1 —
    12 layers, hidden 384, 6 heads.

    Examples
    --------
    >>> from lucid.models import dit_small_2
    >>> config = dit_small_2().config
    >>> config.depth, config.hidden_size, config.patch_size
    (12, 384, 2)
    """
    if pretrained:
        reject_unavailable_pretrained("dit_small_2", alternative="dit_xlarge_2")
    return DiTModel(apply_overrides(_CFG_SMALL_2, overrides))


@register_model(
    task="base",
    family="dit",
    model_type="dit",
    model_class=DiTModel,
    default_config=_CFG_SMALL_4,
    summary="auto",
)
def dit_small_4(pretrained: bool = False, **overrides: object) -> DiTModel:
    """DiT-S/4 — the S backbone at patch 4.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTModel
        The denoising network.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748), Table 1 —
    12 layers, hidden 384, 6 heads.

    Examples
    --------
    >>> from lucid.models import dit_small_4
    >>> config = dit_small_4().config
    >>> config.depth, config.hidden_size, config.patch_size
    (12, 384, 4)
    """
    if pretrained:
        reject_unavailable_pretrained("dit_small_4", alternative="dit_xlarge_2")
    return DiTModel(apply_overrides(_CFG_SMALL_4, overrides))


@register_model(
    task="base",
    family="dit",
    model_type="dit",
    model_class=DiTModel,
    default_config=_CFG_SMALL_8,
    summary="auto",
)
def dit_small_8(pretrained: bool = False, **overrides: object) -> DiTModel:
    """DiT-S/8 — the S backbone at patch 8.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTModel
        The denoising network.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748), Table 1 —
    12 layers, hidden 384, 6 heads.

    Examples
    --------
    >>> from lucid.models import dit_small_8
    >>> config = dit_small_8().config
    >>> config.depth, config.hidden_size, config.patch_size
    (12, 384, 8)
    """
    if pretrained:
        reject_unavailable_pretrained("dit_small_8", alternative="dit_xlarge_2")
    return DiTModel(apply_overrides(_CFG_SMALL_8, overrides))


@register_model(
    task="base",
    family="dit",
    model_type="dit",
    model_class=DiTModel,
    default_config=_CFG_BASE_2,
    summary="auto",
)
def dit_base_2(pretrained: bool = False, **overrides: object) -> DiTModel:
    """DiT-B/2 — the B backbone at patch 2.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTModel
        The denoising network.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748), Table 1 —
    12 layers, hidden 768, 12 heads.

    Examples
    --------
    >>> from lucid.models import dit_base_2
    >>> config = dit_base_2().config
    >>> config.depth, config.hidden_size, config.patch_size
    (12, 768, 2)
    """
    if pretrained:
        reject_unavailable_pretrained("dit_base_2", alternative="dit_xlarge_2")
    return DiTModel(apply_overrides(_CFG_BASE_2, overrides))


@register_model(
    task="base",
    family="dit",
    model_type="dit",
    model_class=DiTModel,
    default_config=_CFG_BASE_4,
    summary="auto",
)
def dit_base_4(pretrained: bool = False, **overrides: object) -> DiTModel:
    """DiT-B/4 — the B backbone at patch 4.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTModel
        The denoising network.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748), Table 1 —
    12 layers, hidden 768, 12 heads.

    Examples
    --------
    >>> from lucid.models import dit_base_4
    >>> config = dit_base_4().config
    >>> config.depth, config.hidden_size, config.patch_size
    (12, 768, 4)
    """
    if pretrained:
        reject_unavailable_pretrained("dit_base_4", alternative="dit_xlarge_2")
    return DiTModel(apply_overrides(_CFG_BASE_4, overrides))


@register_model(
    task="base",
    family="dit",
    model_type="dit",
    model_class=DiTModel,
    default_config=_CFG_BASE_8,
    summary="auto",
)
def dit_base_8(pretrained: bool = False, **overrides: object) -> DiTModel:
    """DiT-B/8 — the B backbone at patch 8.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTModel
        The denoising network.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748), Table 1 —
    12 layers, hidden 768, 12 heads.

    Examples
    --------
    >>> from lucid.models import dit_base_8
    >>> config = dit_base_8().config
    >>> config.depth, config.hidden_size, config.patch_size
    (12, 768, 8)
    """
    if pretrained:
        reject_unavailable_pretrained("dit_base_8", alternative="dit_xlarge_2")
    return DiTModel(apply_overrides(_CFG_BASE_8, overrides))


@register_model(
    task="base",
    family="dit",
    model_type="dit",
    model_class=DiTModel,
    default_config=_CFG_LARGE_2,
    summary="auto",
)
def dit_large_2(pretrained: bool = False, **overrides: object) -> DiTModel:
    """DiT-L/2 — the L backbone at patch 2.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTModel
        The denoising network.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748), Table 1 —
    24 layers, hidden 1024, 16 heads.

    Examples
    --------
    >>> from lucid.models import dit_large_2
    >>> config = dit_large_2().config
    >>> config.depth, config.hidden_size, config.patch_size
    (24, 1024, 2)
    """
    if pretrained:
        reject_unavailable_pretrained("dit_large_2", alternative="dit_xlarge_2")
    return DiTModel(apply_overrides(_CFG_LARGE_2, overrides))


@register_model(
    task="base",
    family="dit",
    model_type="dit",
    model_class=DiTModel,
    default_config=_CFG_LARGE_4,
    summary="auto",
)
def dit_large_4(pretrained: bool = False, **overrides: object) -> DiTModel:
    """DiT-L/4 — the L backbone at patch 4.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTModel
        The denoising network.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748), Table 1 —
    24 layers, hidden 1024, 16 heads.

    Examples
    --------
    >>> from lucid.models import dit_large_4
    >>> config = dit_large_4().config
    >>> config.depth, config.hidden_size, config.patch_size
    (24, 1024, 4)
    """
    if pretrained:
        reject_unavailable_pretrained("dit_large_4", alternative="dit_xlarge_2")
    return DiTModel(apply_overrides(_CFG_LARGE_4, overrides))


@register_model(
    task="base",
    family="dit",
    model_type="dit",
    model_class=DiTModel,
    default_config=_CFG_LARGE_8,
    summary="auto",
)
def dit_large_8(pretrained: bool = False, **overrides: object) -> DiTModel:
    """DiT-L/8 — the L backbone at patch 8.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTModel
        The denoising network.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748), Table 1 —
    24 layers, hidden 1024, 16 heads.

    Examples
    --------
    >>> from lucid.models import dit_large_8
    >>> config = dit_large_8().config
    >>> config.depth, config.hidden_size, config.patch_size
    (24, 1024, 8)
    """
    if pretrained:
        reject_unavailable_pretrained("dit_large_8", alternative="dit_xlarge_2")
    return DiTModel(apply_overrides(_CFG_LARGE_8, overrides))


@register_model(
    task="base",
    family="dit",
    model_type="dit",
    model_class=DiTModel,
    default_config=_CFG_XLARGE_2,
    summary="auto",
)
def dit_xlarge_2(
    pretrained: bool | str = False, *, weights: DiTXLarge2Weights | None = None, **overrides: object
) -> DiTModel:
    """DiT-XL/2 — the XL backbone at patch 2.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Load a released checkpoint.  ``True`` takes ``IMAGENET1K_256``;
        ``"IMAGENET1K_512"`` selects the 512-pixel model and moves
        ``sample_size`` to the 64-side latent it was trained on.  These
        are the only two DiT checkpoints ever published, and they are
        CC-BY-NC-4.0.
    weights : DiTXLarge2Weights or None, optional, keyword-only
        An explicit tag, taking precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTModel
        The denoising network.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748), Table 1 —
    28 layers, hidden 1152, 16 heads.

    Reports 2.27 FID with classifier-free guidance at scale 1.50 on ImageNet 256x256 — the state of the art when published, and 3.04 at 512x512 with the same config over a 64-wide latent.

    Examples
    --------
    >>> from lucid.models import dit_xlarge_2
    >>> config = dit_xlarge_2().config
    >>> config.depth, config.hidden_size, config.patch_size
    (28, 1152, 2)
    """
    entry = weights_mod.resolve_weights(DiTXLarge2Weights, pretrained, weights)
    model = DiTModel(apply_overrides(_CFG_XLARGE_2, _latent_for(entry, overrides)))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="dit_xlarge_2")
    return model


@register_model(
    task="base",
    family="dit",
    model_type="dit",
    model_class=DiTModel,
    default_config=_CFG_XLARGE_4,
    summary="auto",
)
def dit_xlarge_4(pretrained: bool = False, **overrides: object) -> DiTModel:
    """DiT-XL/4 — the XL backbone at patch 4.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTModel
        The denoising network.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748), Table 1 —
    28 layers, hidden 1152, 16 heads.

    Examples
    --------
    >>> from lucid.models import dit_xlarge_4
    >>> config = dit_xlarge_4().config
    >>> config.depth, config.hidden_size, config.patch_size
    (28, 1152, 4)
    """
    if pretrained:
        reject_unavailable_pretrained("dit_xlarge_4", alternative="dit_xlarge_2")
    return DiTModel(apply_overrides(_CFG_XLARGE_4, overrides))


@register_model(
    task="base",
    family="dit",
    model_type="dit",
    model_class=DiTModel,
    default_config=_CFG_XLARGE_8,
    summary="auto",
)
def dit_xlarge_8(pretrained: bool = False, **overrides: object) -> DiTModel:
    """DiT-XL/8 — the XL backbone at patch 8.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTModel
        The denoising network.

    Notes
    -----
    Reference: Peebles and Xie, *"Scalable Diffusion Models with
    Transformers"*, ICCV, 2023 (arXiv:2212.09748), Table 1 —
    28 layers, hidden 1152, 16 heads.

    Examples
    --------
    >>> from lucid.models import dit_xlarge_8
    >>> config = dit_xlarge_8().config
    >>> config.depth, config.hidden_size, config.patch_size
    (28, 1152, 8)
    """
    if pretrained:
        reject_unavailable_pretrained("dit_xlarge_8", alternative="dit_xlarge_2")
    return DiTModel(apply_overrides(_CFG_XLARGE_8, overrides))


@register_model(
    task="image-generation",
    family="dit",
    model_type="dit",
    model_class=DiTForImageGeneration,
    default_config=_CFG_SMALL_2,
    summary="auto",
)
def dit_small_2_gen(pretrained: bool = False, **overrides: object) -> DiTForImageGeneration:
    """:func:`dit_small_2` posed as a diffusion sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTForImageGeneration
        Exposes the denoising objective and :meth:`generate`.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023.

    Examples
    --------
    >>> from lucid.models import dit_small_2_gen
    >>> model = dit_small_2_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("dit_small_2_gen", alternative="dit_xlarge_2_gen")
    return DiTForImageGeneration(apply_overrides(_CFG_SMALL_2, overrides))


@register_model(
    task="image-generation",
    family="dit",
    model_type="dit",
    model_class=DiTForImageGeneration,
    default_config=_CFG_SMALL_4,
    summary="auto",
)
def dit_small_4_gen(pretrained: bool = False, **overrides: object) -> DiTForImageGeneration:
    """:func:`dit_small_4` posed as a diffusion sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTForImageGeneration
        Exposes the denoising objective and :meth:`generate`.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023.

    Examples
    --------
    >>> from lucid.models import dit_small_4_gen
    >>> model = dit_small_4_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("dit_small_4_gen", alternative="dit_xlarge_2_gen")
    return DiTForImageGeneration(apply_overrides(_CFG_SMALL_4, overrides))


@register_model(
    task="image-generation",
    family="dit",
    model_type="dit",
    model_class=DiTForImageGeneration,
    default_config=_CFG_SMALL_8,
    summary="auto",
)
def dit_small_8_gen(pretrained: bool = False, **overrides: object) -> DiTForImageGeneration:
    """:func:`dit_small_8` posed as a diffusion sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTForImageGeneration
        Exposes the denoising objective and :meth:`generate`.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023.

    Examples
    --------
    >>> from lucid.models import dit_small_8_gen
    >>> model = dit_small_8_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("dit_small_8_gen", alternative="dit_xlarge_2_gen")
    return DiTForImageGeneration(apply_overrides(_CFG_SMALL_8, overrides))


@register_model(
    task="image-generation",
    family="dit",
    model_type="dit",
    model_class=DiTForImageGeneration,
    default_config=_CFG_BASE_2,
    summary="auto",
)
def dit_base_2_gen(pretrained: bool = False, **overrides: object) -> DiTForImageGeneration:
    """:func:`dit_base_2` posed as a diffusion sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTForImageGeneration
        Exposes the denoising objective and :meth:`generate`.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023.

    Examples
    --------
    >>> from lucid.models import dit_base_2_gen
    >>> model = dit_base_2_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("dit_base_2_gen", alternative="dit_xlarge_2_gen")
    return DiTForImageGeneration(apply_overrides(_CFG_BASE_2, overrides))


@register_model(
    task="image-generation",
    family="dit",
    model_type="dit",
    model_class=DiTForImageGeneration,
    default_config=_CFG_BASE_4,
    summary="auto",
)
def dit_base_4_gen(pretrained: bool = False, **overrides: object) -> DiTForImageGeneration:
    """:func:`dit_base_4` posed as a diffusion sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTForImageGeneration
        Exposes the denoising objective and :meth:`generate`.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023.

    Examples
    --------
    >>> from lucid.models import dit_base_4_gen
    >>> model = dit_base_4_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("dit_base_4_gen", alternative="dit_xlarge_2_gen")
    return DiTForImageGeneration(apply_overrides(_CFG_BASE_4, overrides))


@register_model(
    task="image-generation",
    family="dit",
    model_type="dit",
    model_class=DiTForImageGeneration,
    default_config=_CFG_BASE_8,
    summary="auto",
)
def dit_base_8_gen(pretrained: bool = False, **overrides: object) -> DiTForImageGeneration:
    """:func:`dit_base_8` posed as a diffusion sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTForImageGeneration
        Exposes the denoising objective and :meth:`generate`.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023.

    Examples
    --------
    >>> from lucid.models import dit_base_8_gen
    >>> model = dit_base_8_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("dit_base_8_gen", alternative="dit_xlarge_2_gen")
    return DiTForImageGeneration(apply_overrides(_CFG_BASE_8, overrides))


@register_model(
    task="image-generation",
    family="dit",
    model_type="dit",
    model_class=DiTForImageGeneration,
    default_config=_CFG_LARGE_2,
    summary="auto",
)
def dit_large_2_gen(pretrained: bool = False, **overrides: object) -> DiTForImageGeneration:
    """:func:`dit_large_2` posed as a diffusion sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTForImageGeneration
        Exposes the denoising objective and :meth:`generate`.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023.

    Examples
    --------
    >>> from lucid.models import dit_large_2_gen
    >>> model = dit_large_2_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("dit_large_2_gen", alternative="dit_xlarge_2_gen")
    return DiTForImageGeneration(apply_overrides(_CFG_LARGE_2, overrides))


@register_model(
    task="image-generation",
    family="dit",
    model_type="dit",
    model_class=DiTForImageGeneration,
    default_config=_CFG_LARGE_4,
    summary="auto",
)
def dit_large_4_gen(pretrained: bool = False, **overrides: object) -> DiTForImageGeneration:
    """:func:`dit_large_4` posed as a diffusion sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTForImageGeneration
        Exposes the denoising objective and :meth:`generate`.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023.

    Examples
    --------
    >>> from lucid.models import dit_large_4_gen
    >>> model = dit_large_4_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("dit_large_4_gen", alternative="dit_xlarge_2_gen")
    return DiTForImageGeneration(apply_overrides(_CFG_LARGE_4, overrides))


@register_model(
    task="image-generation",
    family="dit",
    model_type="dit",
    model_class=DiTForImageGeneration,
    default_config=_CFG_LARGE_8,
    summary="auto",
)
def dit_large_8_gen(pretrained: bool = False, **overrides: object) -> DiTForImageGeneration:
    """:func:`dit_large_8` posed as a diffusion sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTForImageGeneration
        Exposes the denoising objective and :meth:`generate`.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023.

    Examples
    --------
    >>> from lucid.models import dit_large_8_gen
    >>> model = dit_large_8_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("dit_large_8_gen", alternative="dit_xlarge_2_gen")
    return DiTForImageGeneration(apply_overrides(_CFG_LARGE_8, overrides))


@register_model(
    task="image-generation",
    family="dit",
    model_type="dit",
    model_class=DiTForImageGeneration,
    default_config=_CFG_XLARGE_2,
    summary="auto",
)
def dit_xlarge_2_gen(
    pretrained: bool | str = False, *, weights: DiTXLarge2Weights | None = None, **overrides: object
) -> DiTForImageGeneration:
    """:func:`dit_xlarge_2` posed as a diffusion sampler.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Load a released checkpoint.  ``True`` takes ``IMAGENET1K_256``;
        ``"IMAGENET1K_512"`` selects the 512-pixel model and moves
        ``sample_size`` to the 64-side latent it was trained on.  These
        are the only two DiT checkpoints ever published, and they are
        CC-BY-NC-4.0.
    weights : DiTXLarge2Weights or None, optional, keyword-only
        An explicit tag, taking precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTForImageGeneration
        Exposes the denoising objective and :meth:`generate`.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023.

    Examples
    --------
    >>> from lucid.models import dit_xlarge_2_gen
    >>> model = dit_xlarge_2_gen()  # doctest: +SKIP
    """
    entry = weights_mod.resolve_weights(DiTXLarge2Weights, pretrained, weights)
    model = DiTForImageGeneration(apply_overrides(_CFG_XLARGE_2, _latent_for(entry, overrides)))
    if entry is not None:
        # The checkpoint holds the backbone, which this wrapper keeps
        # under ``dit`` — so the load targets that, not the wrapper.
        weights_mod.load_weight_entry(model.dit, entry, name="dit_xlarge_2")
    return model


@register_model(
    task="image-generation",
    family="dit",
    model_type="dit",
    model_class=DiTForImageGeneration,
    default_config=_CFG_XLARGE_4,
    summary="auto",
)
def dit_xlarge_4_gen(pretrained: bool = False, **overrides: object) -> DiTForImageGeneration:
    """:func:`dit_xlarge_4` posed as a diffusion sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTForImageGeneration
        Exposes the denoising objective and :meth:`generate`.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023.

    Examples
    --------
    >>> from lucid.models import dit_xlarge_4_gen
    >>> model = dit_xlarge_4_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("dit_xlarge_4_gen", alternative="dit_xlarge_2_gen")
    return DiTForImageGeneration(apply_overrides(_CFG_XLARGE_4, overrides))


@register_model(
    task="image-generation",
    family="dit",
    model_type="dit",
    model_class=DiTForImageGeneration,
    default_config=_CFG_XLARGE_8,
    summary="auto",
)
def dit_xlarge_8_gen(pretrained: bool = False, **overrides: object) -> DiTForImageGeneration:
    """:func:`dit_xlarge_8` posed as a diffusion sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No DiT checkpoint was published for this configuration; ``True``
        raises rather than returning a randomly initialised model.
    **overrides : object
        Optional :class:`DiTConfig` field overrides.

    Returns
    -------
    DiTForImageGeneration
        Exposes the denoising objective and :meth:`generate`.

    Notes
    -----
    Reference: Peebles and Xie, arXiv:2212.09748, 2023.

    Examples
    --------
    >>> from lucid.models import dit_xlarge_8_gen
    >>> model = dit_xlarge_8_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("dit_xlarge_8_gen", alternative="dit_xlarge_2_gen")
    return DiTForImageGeneration(apply_overrides(_CFG_XLARGE_8, overrides))
