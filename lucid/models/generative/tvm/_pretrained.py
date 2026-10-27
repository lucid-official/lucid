"""Registry factories for Terminal Velocity Matching.

The paper trains one architecture — DiT-XL/2 with the semi-Lipschitz
changes and 18 heads, 678M parameters (Table 6) — on two datasets, and
those are the factories: ``tvm_imagenet256`` over the 32-wide latent of
a 256-pixel image and ``tvm_imagenet512`` over the 64-wide latent of a
512-pixel one.  H11 keeps the size out of the name: the paper has no
size ladder, only ``TVM-XL/2``.

Each dataset was trained at two guidance weights, and those are training
settings rather than architectures, so they are overrides, not
factories.  The defaults are the runs the authors released and the
paper's headline one-step numbers come from:

=================  ======  ===============  ===========  ==========
factory            ``w``   scaled param.    time init    1-NFE FID
=================  ======  ===============  ===========  ==========
``imagenet256``    2.0     yes              spectral     3.29
(override)         1.75    yes              normal       4.58 (1.99 at 4 NFE)
``imagenet512``    2.5     no               normal       4.32
(override)         2.25    yes              spectral     5.37 (2.94 at 4 NFE)
=================  ======  ===============  ===========  ==========

The ``w = 1.75`` run is the paper's best four-step model; build it as
``tvm_imagenet256(guidance_scale=1.75, time_embedding_init="normal")``.
"""

from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.generative.tvm._config import TVMConfig
from lucid.models.generative.tvm._model import TVMForImageGeneration, TVMModel

__all__ = ["tvm_imagenet256", "tvm_imagenet512", "tvm_imagenet256_gen", "tvm_imagenet512_gen"]

# Table 6, first column — the released ``im256_w2`` run.
_CFG_IMAGENET256 = TVMConfig()

# Table 6, third column — the released ``im512_w2p5`` run.  The latent is
# twice as wide, and its whitening statistics are the 512-pixel ones.
_CFG_IMAGENET512 = TVMConfig(
    sample_size=64,
    guidance_scale=2.5,
    scaled_parameterization=False,
    time_embedding_init="normal",
    latent_mean=(1.56, -0.695, 0.483, 0.729),
    latent_std=(5.27, 5.91, 4.21, 4.31),
)

# Every count in Table 6 is the same network: 678M.  The positional table
# is a non-persistent buffer, so the two resolutions agree exactly.  Each
# decorator spells the count out: the docs site reads it as a literal.


@register_model(
    task="base",
    family="tvm",
    model_type="tvm",
    model_class=TVMModel,
    default_config=_CFG_IMAGENET256,
    params=678_000_000,
    summary="auto",
)
def tvm_imagenet256(pretrained: bool = False, **overrides: object) -> TVMModel:
    """The 256-pixel ImageNet network — one step to 3.29 FID.

    Parameters
    ----------
    pretrained : bool, default=False
        The authors' checkpoint has not been converted and hosted for
        Lucid yet (``tools/convert_weights/tvm.py`` does the conversion),
        so ``True`` raises rather than returning a randomly initialised
        model.
    **overrides : object
        Optional :class:`TVMConfig` field overrides.

    Returns
    -------
    TVMModel
        The two-time velocity network.

    Notes
    -----
    Reference: Zhou, Parger, Haque, and Song, *"Terminal Velocity
    Matching"*, ICLR 2026 (arXiv:2511.19797), Tables 1 and 6.  Trained at
    a constant guidance weight of 2: 3.29 FID at one function evaluation
    and 2.80 at two, against MeanFlow-XL/2's 3.43 and 2.93.

    Examples
    --------
    >>> from lucid.models import tvm_imagenet256
    >>> config = tvm_imagenet256().config  # doctest: +SKIP
    >>> config.num_patches, config.guidance_scale  # doctest: +SKIP
    (256, 2.0)
    """
    if pretrained:
        reject_unavailable_pretrained("tvm_imagenet256")
    return TVMModel(apply_overrides(_CFG_IMAGENET256, overrides))


@register_model(
    task="base",
    family="tvm",
    model_type="tvm",
    model_class=TVMModel,
    default_config=_CFG_IMAGENET512,
    params=678_000_000,
    summary="auto",
)
def tvm_imagenet512(pretrained: bool = False, **overrides: object) -> TVMModel:
    """The 512-pixel ImageNet network — one step to 4.32 FID.

    Parameters
    ----------
    pretrained : bool, default=False
        ``True`` raises: the released ``im512_w2p5`` weights are not
        hosted for Lucid yet.
    **overrides : object
        :class:`TVMConfig` fields to change from the 512-pixel defaults.

    Returns
    -------
    TVMModel
        The network over a 64-wide latent, conditioned on ``w = 2.5``.

    Notes
    -----
    Reference: Zhou et al., arXiv:2511.19797, Tables 2 and 6.  Trained at
    a constant guidance weight of 2.5 *without* the scaled
    parameterisation; it beats sCT-XL's 4.33 one-step FID at 61% of its
    parameters.

    Examples
    --------
    >>> from lucid.models import tvm_imagenet512
    >>> tvm_imagenet512().config.num_patches  # doctest: +SKIP
    1024
    """
    if pretrained:
        reject_unavailable_pretrained("tvm_imagenet512")
    return TVMModel(apply_overrides(_CFG_IMAGENET512, overrides))


def _gen(cfg: TVMConfig, overrides: dict[str, object]) -> TVMForImageGeneration:
    return TVMForImageGeneration(apply_overrides(cfg, overrides))


@register_model(
    task="image-generation",
    family="tvm",
    model_type="tvm",
    model_class=TVMForImageGeneration,
    default_config=_CFG_IMAGENET256,
    params=678_000_000,
    summary="auto",
)
def tvm_imagenet256_gen(pretrained: bool = False, **overrides: object) -> TVMForImageGeneration:
    """:func:`tvm_imagenet256` with its objective and sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        The authors' checkpoint has not been converted and hosted for
        Lucid yet, so ``True`` raises rather than returning a randomly
        initialised model.
    **overrides : object
        Optional :class:`TVMConfig` field overrides.

    Returns
    -------
    TVMForImageGeneration
        Exposes the training objective, :meth:`~TVMForImageGeneration.
        update_ema` and :meth:`~TVMForImageGeneration.generate`.

    Notes
    -----
    Reference: Zhou et al., arXiv:2511.19797, Algorithm 1 and Table 6.

    Examples
    --------
    >>> from lucid.models import tvm_imagenet256_gen
    >>> model = tvm_imagenet256_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("tvm_imagenet256_gen")
    return _gen(_CFG_IMAGENET256, overrides)


@register_model(
    task="image-generation",
    family="tvm",
    model_type="tvm",
    model_class=TVMForImageGeneration,
    default_config=_CFG_IMAGENET512,
    params=678_000_000,
    summary="auto",
)
def tvm_imagenet512_gen(pretrained: bool = False, **overrides: object) -> TVMForImageGeneration:
    """:func:`tvm_imagenet512` with its objective and sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        ``True`` raises until the 512-pixel weights are hosted for Lucid.
    **overrides : object
        :class:`TVMConfig` fields to change from the 512-pixel defaults.

    Returns
    -------
    TVMForImageGeneration
        The 512-pixel network, trainable with Algorithm 1 and sampled in
        one or a few jumps.

    Notes
    -----
    Reference: Zhou et al., arXiv:2511.19797, Algorithm 1 and Table 6.
    The 512-pixel configurations add gradient clipping at norm 1.0 in the
    released training recipe — an optimiser setting, left to the loop.

    Examples
    --------
    >>> from lucid.models import tvm_imagenet512_gen
    >>> model = tvm_imagenet512_gen()  # doctest: +SKIP
    """
    if pretrained:
        reject_unavailable_pretrained("tvm_imagenet512_gen")
    return _gen(_CFG_IMAGENET512, overrides)
