"""Registry factories for RealNVP (Dinh et al., 2016).

One multi-scale template per dataset in the paper's Table 1.  §4.1 fixes
the differences: how deep and wide the coupling networks are, and how far
the multi-scale recursion runs (it stops when the last scale sees a
``4 x 4`` map, except for CIFAR-10, which "downscale[s] only once").

    * ``realnvp_cifar``      — CIFAR-10 32x32, 8 blocks / 64 maps, 1 downscale
    * ``realnvp_imagenet32`` — Imagenet 32x32, 4 blocks / 32 maps
    * ``realnvp_imagenet64`` — Imagenet 64x64, 2 blocks / 32 maps
    * ``realnvp_lsun``       — LSUN bedroom / tower / church outdoor, 64x64
    * ``realnvp_celeba``     — CelebA 64x64

The three 64x64 setups share an architecture and differ only in data and
reported bits/dim, so each keeps its own name for discoverability.

Inputs are ``[0, 1]`` images: the logit squash the paper applies before
the flow lives inside the model, so its log-determinant is already part of
``log_prob`` and ``bits_per_dim``.

Smaller test-scale models go through ``create_model("realnvp_cifar",
sample_size=..., num_scales=..., residual_blocks=..., base_dim=...)``.
"""

from lucid.models._registry import register_model
from lucid.models.generative.realnvp._config import RealNVPConfig
from lucid.models.generative.realnvp._model import RealNVPForImageGeneration, RealNVPModel
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# Paper §4.1 — CIFAR-10: 8 residual blocks, 64 feature maps, one downscale.
_CFG_CIFAR = RealNVPConfig(
    sample_size=32, in_channels=3, out_channels=3, num_scales=2, residual_blocks=8, base_dim=64
)

# Paper §4.1 — Imagenet 32x32: 4 residual blocks, 32 feature maps, recursion
# down to a 4x4 final scale.
_CFG_IMAGENET32 = RealNVPConfig(
    sample_size=32, in_channels=3, out_channels=3, num_scales=4, residual_blocks=4, base_dim=32
)

# Paper §4.1 — Imagenet 64x64: 2 residual blocks; one more scale than the
# 32x32 setup so the recursion still ends at 4x4.
_CFG_IMAGENET64 = RealNVPConfig(
    sample_size=64, in_channels=3, out_channels=3, num_scales=5, residual_blocks=2, base_dim=32
)

# Paper §4.1 — LSUN 64x64 crops (bedroom / tower / church outdoor).
_CFG_LSUN = RealNVPConfig(
    sample_size=64, in_channels=3, out_channels=3, num_scales=5, residual_blocks=2, base_dim=32
)

# Paper §4.1 — CelebA, centre-cropped and resized to 64x64.
_CFG_CELEBA = RealNVPConfig(
    sample_size=64, in_channels=3, out_channels=3, num_scales=5, residual_blocks=2, base_dim=32
)


# ── Bare flows ────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="realnvp",
    model_type="realnvp",
    model_class=RealNVPModel,
    default_config=_CFG_CIFAR,
)
def realnvp_cifar(pretrained: bool = False, **overrides: object) -> RealNVPModel:
    r"""Construct the RealNVP flow for the CIFAR-10 setup.

    Paper-faithful CIFAR-10 configuration from Dinh et al., 2016 §4.1:
    ``32 x 32`` RGB, coupling networks of 8 residual blocks with 64
    feature maps, and — unlike every other experiment — only a single
    downscale, so the flow keeps working at ``16 x 16`` rather than
    recursing to ``4 x 4``.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``realnvp_cifar``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`RealNVPConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    RealNVPModel
        Bare bijection configured with the CIFAR-10 setup and any
        overrides.

    Notes
    -----
    Reference: Dinh, Sohl-Dickstein, and Bengio, *"Density Estimation
    Using Real NVP"*, ICLR, 2017 (arXiv:1605.08803), §4.1 and Table 1.
    Reported test result: **3.49 bits/dim**, with horizontal-flip
    augmentation and the ``alpha = 0.05`` logit preprocessing this model
    applies internally.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.realnvp import realnvp_cifar
    >>> model = realnvp_cifar().eval()
    >>> x = lucid.rand((1, 3, 32, 32))
    >>> model.bits_per_dim(x).shape
    (1,)
    """
    if pretrained:
        reject_unavailable_pretrained("realnvp_cifar")
    return RealNVPModel(apply_overrides(_CFG_CIFAR, overrides))


@register_model(
    task="base",
    family="realnvp",
    model_type="realnvp",
    model_class=RealNVPModel,
    default_config=_CFG_IMAGENET32,
)
def realnvp_imagenet32(pretrained: bool = False, **overrides: object) -> RealNVPModel:
    r"""Construct the RealNVP flow for the Imagenet ``32 x 32`` setup.

    Paper-faithful configuration from Dinh et al., 2016 §4.1: four scales
    (so the recursion ends on a ``4 x 4`` map), coupling networks of 4
    residual blocks with 32 feature maps at the first scale, doubling per
    scale.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``realnvp_imagenet32``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`RealNVPConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    RealNVPModel
        Bare bijection configured with the Imagenet ``32 x 32`` setup.

    Notes
    -----
    Reference: Dinh, Sohl-Dickstein, and Bengio, *"Density Estimation
    Using Real NVP"*, ICLR, 2017 (arXiv:1605.08803), §4.1 and Table 1.
    Reported validation result: **4.28 bits/dim** (4.26 train).

    Examples
    --------
    >>> from lucid.models.generative.realnvp import realnvp_imagenet32
    >>> model = realnvp_imagenet32().eval()
    >>> model.config.num_scales
    4
    """
    if pretrained:
        reject_unavailable_pretrained("realnvp_imagenet32")
    return RealNVPModel(apply_overrides(_CFG_IMAGENET32, overrides))


@register_model(
    task="base",
    family="realnvp",
    model_type="realnvp",
    model_class=RealNVPModel,
    default_config=_CFG_IMAGENET64,
)
def realnvp_imagenet64(pretrained: bool = False, **overrides: object) -> RealNVPModel:
    r"""Construct the RealNVP flow for the Imagenet ``64 x 64`` setup.

    Paper-faithful configuration from Dinh et al., 2016 §4.1: five scales
    (``64 → 32 → 16 → 8 → 4``) with coupling networks of 2 residual
    blocks, the depth the paper drops to at this resolution.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``realnvp_imagenet64``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`RealNVPConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    RealNVPModel
        Bare bijection configured with the Imagenet ``64 x 64`` setup.

    Notes
    -----
    Reference: Dinh, Sohl-Dickstein, and Bengio, *"Density Estimation
    Using Real NVP"*, ICLR, 2017 (arXiv:1605.08803), §4.1 and Table 1.
    Reported validation result: **3.98 bits/dim** (3.75 train).

    Examples
    --------
    >>> from lucid.models.generative.realnvp import realnvp_imagenet64
    >>> model = realnvp_imagenet64().eval()
    >>> model.input_dim
    12288
    """
    if pretrained:
        reject_unavailable_pretrained("realnvp_imagenet64")
    return RealNVPModel(apply_overrides(_CFG_IMAGENET64, overrides))


@register_model(
    task="base",
    family="realnvp",
    model_type="realnvp",
    model_class=RealNVPModel,
    default_config=_CFG_LSUN,
)
def realnvp_lsun(pretrained: bool = False, **overrides: object) -> RealNVPModel:
    r"""Construct the RealNVP flow for the LSUN ``64 x 64`` setup.

    Paper-faithful configuration from Dinh et al., 2016 §4.1 — the same
    five-scale, 2-residual-block architecture used for every ``64 x 64``
    dataset.  One factory covers the three reported LSUN categories
    (bedroom, tower, church outdoor); they differ only in training data.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``realnvp_lsun``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`RealNVPConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    RealNVPModel
        Bare bijection configured with the LSUN setup.

    Notes
    -----
    Reference: Dinh, Sohl-Dickstein, and Bengio, *"Density Estimation
    Using Real NVP"*, ICLR, 2017 (arXiv:1605.08803), §4.1 and Table 1.
    Reported validation results: **2.72** (bedroom), **2.81** (tower) and
    **3.08** (church outdoor) bits/dim.  Images are downsampled so the
    smallest side is 96 px, then randomly cropped to ``64 x 64``.

    Examples
    --------
    >>> from lucid.models.generative.realnvp import realnvp_lsun
    >>> model = realnvp_lsun().eval()
    >>> model.config.residual_blocks
    2
    """
    if pretrained:
        reject_unavailable_pretrained("realnvp_lsun")
    return RealNVPModel(apply_overrides(_CFG_LSUN, overrides))


@register_model(
    task="base",
    family="realnvp",
    model_type="realnvp",
    model_class=RealNVPModel,
    default_config=_CFG_CELEBA,
)
def realnvp_celeba(pretrained: bool = False, **overrides: object) -> RealNVPModel:
    r"""Construct the RealNVP flow for the CelebA ``64 x 64`` setup.

    Paper-faithful configuration from Dinh et al., 2016 §4.1 — five
    scales with 2-residual-block coupling networks, matching the other
    ``64 x 64`` experiments.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``realnvp_celeba``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`RealNVPConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    RealNVPModel
        Bare bijection configured with the CelebA setup.

    Notes
    -----
    Reference: Dinh, Sohl-Dickstein, and Bengio, *"Density Estimation
    Using Real NVP"*, ICLR, 2017 (arXiv:1605.08803), §4.1 and Table 1.
    Reported validation result: **3.02 bits/dim** (2.97 train), on an
    approximately central ``148 x 148`` crop resized to ``64 x 64``.

    Examples
    --------
    >>> from lucid.models.generative.realnvp import realnvp_celeba
    >>> model = realnvp_celeba().eval()
    >>> model.prior
    'gaussian'
    """
    if pretrained:
        reject_unavailable_pretrained("realnvp_celeba")
    return RealNVPModel(apply_overrides(_CFG_CELEBA, overrides))


# ── Image-generation heads ───────────────────────────────────────────────────


@register_model(
    task="image-generation",
    family="realnvp",
    model_type="realnvp",
    model_class=RealNVPForImageGeneration,
    default_config=_CFG_CIFAR,
)
def realnvp_cifar_gen(pretrained: bool = False, **overrides: object) -> RealNVPForImageGeneration:
    r"""RealNVP generator for CIFAR-10 (bits/dim loss + ``.generate()``).

    Same bijection as :func:`realnvp_cifar`, wrapped with the paper's
    training objective — exact negative log-likelihood reported in bits
    per dimension — and single-pass ancestral sampling.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``realnvp_cifar_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`RealNVPConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    RealNVPForImageGeneration
        Generator configured with the CIFAR-10 setup.

    Notes
    -----
    Reference: Dinh, Sohl-Dickstein, and Bengio, *"Density Estimation
    Using Real NVP"*, ICLR, 2017 (arXiv:1605.08803).  Reported: 3.49
    bits/dim.

    Examples
    --------
    >>> from lucid.models.generative.realnvp import realnvp_cifar_gen
    >>> model = realnvp_cifar_gen().eval()
    >>> model.generate(n_samples=2).samples.shape
    (2, 3, 32, 32)
    """
    if pretrained:
        reject_unavailable_pretrained("realnvp_cifar_gen")
    return RealNVPForImageGeneration(apply_overrides(_CFG_CIFAR, overrides))


@register_model(
    task="image-generation",
    family="realnvp",
    model_type="realnvp",
    model_class=RealNVPForImageGeneration,
    default_config=_CFG_IMAGENET32,
)
def realnvp_imagenet32_gen(
    pretrained: bool = False, **overrides: object
) -> RealNVPForImageGeneration:
    r"""RealNVP generator for Imagenet ``32 x 32``.

    Same bijection as :func:`realnvp_imagenet32`, wrapped with the
    bits/dim objective and the ancestral sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``realnvp_imagenet32_gen``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`RealNVPConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    RealNVPForImageGeneration
        Generator configured with the Imagenet ``32 x 32`` setup.

    Notes
    -----
    Reference: Dinh, Sohl-Dickstein, and Bengio, *"Density Estimation
    Using Real NVP"*, ICLR, 2017 (arXiv:1605.08803).  Reported: 4.28
    bits/dim.

    Examples
    --------
    >>> from lucid.models.generative.realnvp import realnvp_imagenet32_gen
    >>> model = realnvp_imagenet32_gen().eval()
    >>> model.generate(n_samples=2).samples.shape
    (2, 3, 32, 32)
    """
    if pretrained:
        reject_unavailable_pretrained("realnvp_imagenet32_gen")
    return RealNVPForImageGeneration(apply_overrides(_CFG_IMAGENET32, overrides))


@register_model(
    task="image-generation",
    family="realnvp",
    model_type="realnvp",
    model_class=RealNVPForImageGeneration,
    default_config=_CFG_IMAGENET64,
)
def realnvp_imagenet64_gen(
    pretrained: bool = False, **overrides: object
) -> RealNVPForImageGeneration:
    r"""RealNVP generator for Imagenet ``64 x 64``.

    Same bijection as :func:`realnvp_imagenet64`, wrapped with the
    bits/dim objective and the ancestral sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``realnvp_imagenet64_gen``, so ``True``
        raises ``NotImplementedError`` rather than return randomly initialised
        weights.
    **overrides : object
        Optional :class:`RealNVPConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    RealNVPForImageGeneration
        Generator configured with the Imagenet ``64 x 64`` setup.

    Notes
    -----
    Reference: Dinh, Sohl-Dickstein, and Bengio, *"Density Estimation
    Using Real NVP"*, ICLR, 2017 (arXiv:1605.08803).  Reported: 3.98
    bits/dim.

    Examples
    --------
    >>> from lucid.models.generative.realnvp import realnvp_imagenet64_gen
    >>> model = realnvp_imagenet64_gen().eval()
    >>> model.generate(n_samples=1).samples.shape
    (1, 3, 64, 64)
    """
    if pretrained:
        reject_unavailable_pretrained("realnvp_imagenet64_gen")
    return RealNVPForImageGeneration(apply_overrides(_CFG_IMAGENET64, overrides))


@register_model(
    task="image-generation",
    family="realnvp",
    model_type="realnvp",
    model_class=RealNVPForImageGeneration,
    default_config=_CFG_LSUN,
)
def realnvp_lsun_gen(pretrained: bool = False, **overrides: object) -> RealNVPForImageGeneration:
    r"""RealNVP generator for LSUN ``64 x 64``.

    Same bijection as :func:`realnvp_lsun`, wrapped with the bits/dim
    objective and the ancestral sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``realnvp_lsun_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`RealNVPConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    RealNVPForImageGeneration
        Generator configured with the LSUN setup.

    Notes
    -----
    Reference: Dinh, Sohl-Dickstein, and Bengio, *"Density Estimation
    Using Real NVP"*, ICLR, 2017 (arXiv:1605.08803).  Reported: 2.72
    (bedroom), 2.81 (tower), 3.08 (church outdoor) bits/dim.

    Examples
    --------
    >>> from lucid.models.generative.realnvp import realnvp_lsun_gen
    >>> model = realnvp_lsun_gen().eval()
    >>> model.generate(n_samples=1).samples.shape
    (1, 3, 64, 64)
    """
    if pretrained:
        reject_unavailable_pretrained("realnvp_lsun_gen")
    return RealNVPForImageGeneration(apply_overrides(_CFG_LSUN, overrides))


@register_model(
    task="image-generation",
    family="realnvp",
    model_type="realnvp",
    model_class=RealNVPForImageGeneration,
    default_config=_CFG_CELEBA,
)
def realnvp_celeba_gen(pretrained: bool = False, **overrides: object) -> RealNVPForImageGeneration:
    r"""RealNVP generator for CelebA ``64 x 64``.

    Same bijection as :func:`realnvp_celeba`, wrapped with the bits/dim
    objective and the ancestral sampler.  This is the setup the paper's
    latent-space interpolation figures come from — the flow is a
    bijection, so interpolating between two ``encode`` outputs and
    decoding gives a semantically smooth path with no encoder
    approximation.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``realnvp_celeba_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`RealNVPConfig` field overrides forwarded into
        the underlying config.

    Returns
    -------
    RealNVPForImageGeneration
        Generator configured with the CelebA setup.

    Notes
    -----
    Reference: Dinh, Sohl-Dickstein, and Bengio, *"Density Estimation
    Using Real NVP"*, ICLR, 2017 (arXiv:1605.08803).  Reported: 3.02
    bits/dim.

    Examples
    --------
    >>> from lucid.models.generative.realnvp import realnvp_celeba_gen
    >>> model = realnvp_celeba_gen().eval()
    >>> model.generate(n_samples=1).samples.shape
    (1, 3, 64, 64)
    """
    if pretrained:
        reject_unavailable_pretrained("realnvp_celeba_gen")
    return RealNVPForImageGeneration(apply_overrides(_CFG_CELEBA, overrides))
