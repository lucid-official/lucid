"""Registry factories for NCSN (Song & Ermon, 2019).

Two paper-faithful variants:

    * ``ncsn_cifar``  — CIFAR-10 setup from Song 2019 §4.1, with the
      NCSNv2 sigma range
    * ``ncsn_celeba`` — CelebA 64×64

Both reuse the DDPM U-Net architecture (NCSNv2 / NCSN++ uses the same
modern U-Net as diffusion).  ``sigma_max`` is dataset-specific — set
following NCSNv2 §3.2 "Technique 1" (use ``σ_max`` ≈ max pairwise data
distance).
"""

from lucid.models._registry import register_model
from lucid.models.generative.ncsn._config import NCSNConfig
from lucid.models.generative.ncsn._model import NCSNForImageGeneration, NCSNModel
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# Song & Ermon 2019 CIFAR-10 setup (§4.1) — L=10, σ_1=1, σ_10=0.01.  NCSNv2
# bumped σ_max up to 50 for CIFAR; we follow the v2 recommendation as the
# default since it produces sharper samples.  (NCSNv2's Table 1 is the
# Inception/FID comparison — it carries no architecture settings.)
_CFG_CIFAR = NCSNConfig(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_mult=(1, 2, 2, 2),
    num_res_blocks=2,
    attention_resolutions=(16,),
    num_heads=1,
    dropout=0.0,
    num_noise_levels=232,
    sigma_max=50.0,
    sigma_min=0.01,
    langevin_steps=5,  # NCSNv2 §3.3 — T=5 with σ-tuned step size
    # configs/cifar10.yml: step_lr 6.2e-6; in the alpha/2 + sqrt(alpha)
    # parameterisation used here that is eps = 2 * step_lr.  2e-5 is the
    # NCSN v1 value, which does not go with the v2 sigma schedule above.
    langevin_eps=1.24e-5,
)

# CelebA 64x64.  NCSNv2 has no §C architecture section: its per-dataset
# network settings live in Appendix B.1 and in the released configs,
# which pin only ngf=128, InstanceNorm++ and ELU.  The channel
# multipliers and attention resolutions below are this port's choice.
_CFG_CELEBA = NCSNConfig(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_mult=(1, 1, 2, 2, 4, 4),
    num_res_blocks=2,
    attention_resolutions=(16,),
    num_heads=1,
    dropout=0.0,
    num_noise_levels=500,
    sigma_max=90.0,
    sigma_min=0.01,
    langevin_steps=5,
    # configs/celeba.yml: step_lr 3.3e-6 -> eps = 6.6e-6.
    langevin_eps=6.6e-6,
)


# ── Bare score networks ───────────────────────────────────────────────────────


@register_model(
    task="base",
    family="ncsn",
    model_type="ncsn",
    model_class=NCSNModel,
    default_config=_CFG_CIFAR,
)
def ncsn_cifar(pretrained: bool = False, **overrides: object) -> NCSNModel:
    r"""Construct an NCSN score network for the CIFAR-10 setup.

    Paper-faithful CIFAR-10 configuration combining Song and Ermon, 2019
    §4.1 with the NCSNv2 (Song 2020) refinements: sample size
    32x32, ``base_channels=128``, ``channel_mult=(1, 2, 2, 2)``, 2
    ResBlocks per stage, self-attention at the 16x16 feature map,
    :math:`L = 232` geometric noise levels in :math:`[\sigma_{\min} = 0.01,
    \sigma_{\max} = 50]`, and the NCSNv2 §3.3 Langevin schedule
    (:math:`T = 5` steps per level, :math:`\epsilon = 2 \cdot 10^{-5}`).

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``ncsn_cifar``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NCSNConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    NCSNModel
        Score network configured with the CIFAR-10 setup and any
        overrides.

    Notes
    -----
    Reference: Song and Ermon, *"Generative Modeling by Estimating
    Gradients of the Data Distribution"*, NeurIPS, 2019 (arXiv:1907.05600);
    NCSNv2 refinements in Song and Ermon, 2020 (arXiv:2006.09011).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ncsn import ncsn_cifar
    >>> model = ncsn_cifar().eval()
    >>> x_tilde = lucid.randn((1, 3, 32, 32))
    >>> sigma_idx = lucid.tensor([3]).long()
    >>> out = model(x_tilde, sigma_idx)
    >>> out.sample.shape   # (1, 3, 32, 32) — raw score
    (1, 3, 32, 32)
    """
    if pretrained:
        reject_unavailable_pretrained("ncsn_cifar")
    return NCSNModel(apply_overrides(_CFG_CIFAR, overrides))


@register_model(
    task="base",
    family="ncsn",
    model_type="ncsn",
    model_class=NCSNModel,
    default_config=_CFG_CELEBA,
)
def ncsn_celeba(pretrained: bool = False, **overrides: object) -> NCSNModel:
    r"""Construct an NCSN score network for the CelebA 64x64 setup.

    Paper-faithful CelebA 64x64 configuration from NCSNv2 (Song 2020)
    Appendix C: sample size 64x64, ``base_channels=128``,
    ``channel_mult=(1, 1, 2, 2, 4, 4)``, 2 ResBlocks per stage,
    self-attention at the 16x16 feature map, :math:`L = 500` geometric
    noise levels in :math:`[\sigma_{\min} = 0.01, \sigma_{\max} = 90]`,
    and the NCSNv2 Langevin schedule (:math:`T = 5` steps per level,
    :math:`\epsilon = 2 \cdot 10^{-5}`).

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``ncsn_celeba``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NCSNConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    NCSNModel
        Score network configured with the CelebA 64x64 setup and any
        overrides.

    Notes
    -----
    Reference: Song and Ermon, *"Improved Techniques for Training
    Score-Based Generative Models"* (NCSNv2), NeurIPS, 2020
    (arXiv:2006.09011), Appendix C.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ncsn import ncsn_celeba
    >>> model = ncsn_celeba().eval()
    >>> x_tilde = lucid.randn((1, 3, 64, 64))
    >>> sigma_idx = lucid.tensor([10]).long()
    >>> out = model(x_tilde, sigma_idx)
    >>> out.sample.shape   # (1, 3, 64, 64)
    (1, 3, 64, 64)
    """
    if pretrained:
        reject_unavailable_pretrained("ncsn_celeba")
    return NCSNModel(apply_overrides(_CFG_CELEBA, overrides))


# ── Image-generation heads ───────────────────────────────────────────────────


@register_model(
    task="image-generation",
    family="ncsn",
    model_type="ncsn",
    model_class=NCSNForImageGeneration,
    default_config=_CFG_CIFAR,
)
def ncsn_cifar_gen(pretrained: bool = False, **overrides: object) -> NCSNForImageGeneration:
    r"""Construct an NCSN CIFAR-10 model with DSM loss and ``.generate()``.

    Same trunk as :func:`ncsn_cifar` (CIFAR-10 setup, :math:`L = 232`
    noise levels), wrapped with the denoising score-matching loss and
    annealed Langevin sampling.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``ncsn_cifar_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NCSNConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    NCSNForImageGeneration
        CIFAR-10 NCSN wrapped with the DSM loss head and Langevin sampler.

    Notes
    -----
    Reference: Song and Ermon, NeurIPS, 2019 (arXiv:1907.05600); NCSNv2 in
    Song and Ermon, 2020 (arXiv:2006.09011).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ncsn import ncsn_cifar_gen
    >>> model = ncsn_cifar_gen().eval()
    >>> x = lucid.randn((1, 3, 32, 32))
    >>> out = model(x)
    >>> out.sample.shape   # raw score field
    (1, 3, 32, 32)
    """
    if pretrained:
        reject_unavailable_pretrained("ncsn_cifar_gen")
    return NCSNForImageGeneration(apply_overrides(_CFG_CIFAR, overrides))


@register_model(
    task="image-generation",
    family="ncsn",
    model_type="ncsn",
    model_class=NCSNForImageGeneration,
    default_config=_CFG_CELEBA,
)
def ncsn_celeba_gen(pretrained: bool = False, **overrides: object) -> NCSNForImageGeneration:
    r"""Construct an NCSN CelebA 64x64 model with DSM loss and ``.generate()``.

    Same trunk as :func:`ncsn_celeba` (CelebA 64x64 setup, :math:`L = 500`
    noise levels), wrapped with the denoising score-matching loss and
    annealed Langevin sampling.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``ncsn_celeba_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NCSNConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    NCSNForImageGeneration
        CelebA 64x64 NCSN wrapped with the DSM loss head and Langevin
        sampler.

    Notes
    -----
    Reference: Song and Ermon, *"Improved Techniques for Training
    Score-Based Generative Models"* (NCSNv2), NeurIPS, 2020
    (arXiv:2006.09011), Appendix C.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ncsn import ncsn_celeba_gen
    >>> model = ncsn_celeba_gen().eval()
    >>> x = lucid.randn((1, 3, 64, 64))
    >>> out = model(x)
    >>> out.sample.shape   # raw score field
    (1, 3, 64, 64)
    """
    if pretrained:
        reject_unavailable_pretrained("ncsn_celeba_gen")
    return NCSNForImageGeneration(apply_overrides(_CFG_CELEBA, overrides))
