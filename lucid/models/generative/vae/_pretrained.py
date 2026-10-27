"""Registry factories for VAE.

Our convolutional VAE doesn't map to a single canonical paper variant table
(the way ResNet does for sizes 18 / 34 / 50 / …) — Kingma & Welling (2013)
specified an MLP encoder, and downstream variants (β-VAE, Ladder VAE, …)
each use bespoke per-dataset configs.  Following the project rule
("paper-named only, otherwise nominal"), we expose just two factories:

    * ``vae``  — vanilla / β-VAE topology (single bottleneck ``z``).
    * ``hvae`` — hierarchical / Ladder-VAE topology (one ``z_l`` per stage).

Switch β-VAE on by passing ``kl_weight=...`` at ``create_model`` time.
Specific dataset configs (DSprites, CelebA, MNIST 5-level, …) belong in
user code or in follow-up paper-faithful factories when we ship pretrained
weights.
"""

from lucid.models._registry import register_model
from lucid.models.generative.vae._config import VAEConfig
from lucid.models.generative.vae._model import VAEForImageGeneration, VAEModel
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# Vanilla / β-VAE default — single bottleneck ``z``.
_CFG_VAE = VAEConfig(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    latent_dim=128,
    down_block_channels=(64, 128, 256),
)

# Hierarchical (Sønderby et al., 2016 / Ladder VAE) default — one ``z_l``
# per encoder stage.
_CFG_HVAE = VAEConfig(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    latent_dim=(32, 64, 128),
    down_block_channels=(64, 128, 256),
)


# ── Bare encoder-decoder ──────────────────────────────────────────────────────


@register_model(
    task="base", family="vae", model_type="vae", model_class=VAEModel, default_config=_CFG_VAE
)
def vae(pretrained: bool = False, **overrides: object) -> VAEModel:
    r"""Construct a convolutional VAE with a single bottleneck latent.

    Vanilla VAE topology following Kingma and Welling, 2014, configured with
    a 32x32 input, ``down_block_channels=(64, 128, 256)`` (three stride-2
    encoder stages — 32 -> 16 -> 8 -> 4), and a 128-dimensional latent.
    Pass ``latent_dim`` as a tuple of integers to switch to the
    hierarchical topology, or call :func:`hvae` directly to use the
    Sønderby 3-level stack default.

    Pass ``kl_weight!=1.0`` to obtain a :math:`\beta`-VAE (Higgins 2017).

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``vae``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`VAEConfig` field overrides (e.g. ``latent_dim=...``,
        ``kl_weight=...``, ``recon_loss="bce"``) forwarded into the
        underlying config.

    Returns
    -------
    VAEModel
        Convolutional VAE trunk configured with the vanilla defaults and
        any overrides.

    Notes
    -----
    Reference: Kingma and Welling, *"Auto-Encoding Variational Bayes"*,
    ICLR, 2014 (arXiv:1312.6114); :math:`\beta`-VAE in Higgins et al.,
    *"beta-VAE: Learning Basic Visual Concepts with a Constrained
    Variational Framework"*, ICLR, 2017.

    Reparameterisation trick:

    .. math::

        z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon,
        \qquad
        \epsilon \sim \mathcal{N}(0, \mathbf{I}).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.vae import vae
    >>> model = vae().eval()
    >>> x = lucid.randn((1, 3, 32, 32))
    >>> out = model(x)
    >>> out.sample.shape, out.latent.shape
    ((1, 3, 32, 32), (1, 128))
    """
    if pretrained:
        reject_unavailable_pretrained("vae")
    return VAEModel(apply_overrides(_CFG_VAE, overrides))


@register_model(
    task="base", family="vae", model_type="vae", model_class=VAEModel, default_config=_CFG_HVAE
)
def hvae(pretrained: bool = False, **overrides: object) -> VAEModel:
    r"""Construct a hierarchical (Ladder-style) VAE with one latent per stage.

    Hierarchical VAE topology following Sønderby et al., 2016, configured
    with a 32x32 input, ``down_block_channels=(64, 128, 256)`` (three
    stride-2 encoder stages), and per-stage latent widths
    ``(32, 64, 128)`` — one :math:`z_l \in \mathbb{R}^{d_l}` extracted per
    stage and re-injected at the matching decoder resolution.  The KL
    becomes a sum over levels (independent-posterior Ladder variant).

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``hvae``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`VAEConfig` field overrides forwarded into the
        underlying config.  The tuple ``latent_dim`` must match the
        length of ``down_block_channels``.

    Returns
    -------
    VAEModel
        Hierarchical VAE trunk configured with the 3-level default and
        any overrides.

    Notes
    -----
    Reference: Sønderby, Raiko, Maaløe, Sønderby, and Winther, *"Ladder
    Variational Autoencoders"*, NeurIPS, 2016 (arXiv:1602.02282).

    Hierarchical ELBO:

    .. math::

        \mathcal{L}_{\mathrm{ELBO}}
            = \mathbb{E}_{q_\phi(z \mid x)}\!\big[\log p_\theta(x \mid z)\big]
              - \sum_{l = 1}^{L}
                \mathrm{KL}\!\big(
                    q_\phi(z_l \mid x) \,\big\|\, p(z_l)
                \big).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.vae import hvae
    >>> model = hvae().eval()
    >>> x = lucid.randn((1, 3, 32, 32))
    >>> out = model(x)
    >>> out.sample.shape, out.latent.shape   # latent is flat concat (32+64+128)
    ((1, 3, 32, 32), (1, 224))
    """
    if pretrained:
        reject_unavailable_pretrained("hvae")
    return VAEModel(apply_overrides(_CFG_HVAE, overrides))


# ── Image-generation heads ───────────────────────────────────────────────────


@register_model(
    task="image-generation",
    family="vae",
    model_type="vae",
    model_class=VAEForImageGeneration,
    default_config=_CFG_VAE,
)
def vae_gen(pretrained: bool = False, **overrides: object) -> VAEForImageGeneration:
    r"""Construct a vanilla VAE with the ELBO loss and prior-sample ``.generate()``.

    Same trunk as :func:`vae` (single 128-dim bottleneck, 3-stage encoder),
    wrapped with the ELBO training loss (reconstruction +
    :math:`\beta`-weighted KL) and a convenience prior sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``vae_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`VAEConfig` field overrides forwarded into the
        underlying config.  Pass ``kl_weight=...`` to engage
        :math:`\beta`-VAE behaviour; pass ``recon_loss="bce"`` for
        Bernoulli likelihoods.

    Returns
    -------
    VAEForImageGeneration
        Vanilla VAE wrapped with ELBO loss and prior sampler.

    Notes
    -----
    Reference: Kingma and Welling, *"Auto-Encoding Variational Bayes"*,
    ICLR, 2014 (arXiv:1312.6114).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.vae import vae_gen
    >>> model = vae_gen().eval()
    >>> x = lucid.randn((1, 3, 32, 32))
    >>> out = model(x)
    >>> out.sample.shape, out.kl_loss.shape   # reconstruction + scalar KL
    ((1, 3, 32, 32), ())
    """
    if pretrained:
        reject_unavailable_pretrained("vae_gen")
    return VAEForImageGeneration(apply_overrides(_CFG_VAE, overrides))


@register_model(
    task="image-generation",
    family="vae",
    model_type="vae",
    model_class=VAEForImageGeneration,
    default_config=_CFG_HVAE,
)
def hvae_gen(pretrained: bool = False, **overrides: object) -> VAEForImageGeneration:
    r"""Construct a hierarchical VAE with the ELBO loss and prior-sample ``.generate()``.

    Same trunk as :func:`hvae` (3-level latent stack, 3-stage encoder),
    wrapped with the hierarchical ELBO training loss (reconstruction +
    sum of per-level :math:`\beta`-weighted KLs) and a convenience prior
    sampler that draws one :math:`z_l \sim \mathcal{N}(0, \mathbf{I})`
    per level.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``hvae_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`VAEConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    VAEForImageGeneration
        Hierarchical VAE wrapped with ELBO loss and prior sampler.

    Notes
    -----
    Reference: Sønderby, Raiko, Maaløe, Sønderby, and Winther, *"Ladder
    Variational Autoencoders"*, NeurIPS, 2016 (arXiv:1602.02282).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.vae import hvae_gen
    >>> model = hvae_gen().eval()
    >>> x = lucid.randn((1, 3, 32, 32))
    >>> out = model(x)
    >>> out.sample.shape, out.kl_loss.shape
    ((1, 3, 32, 32), ())
    """
    if pretrained:
        reject_unavailable_pretrained("hvae_gen")
    return VAEForImageGeneration(apply_overrides(_CFG_HVAE, overrides))
