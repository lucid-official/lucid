"""Registry factories for NICE (Dinh et al., 2014).

The paper specifies one architecture template — four additive coupling
layers plus a diagonal scaling — parametrised per dataset.  We expose one
factory per column of Figure 3:

    * ``nice_mnist``  — MNIST (784 dims)
    * ``nice_tfd``    — Toronto Face Database (2304 dims)
    * ``nice_svhn``   — SVHN (3072 dims)
    * ``nice_cifar``  — CIFAR-10 (3072 dims)

All four share the coupling code; only the coupling network's depth /
width and the prior differ.  ``nice_svhn`` and ``nice_cifar`` are
architecturally identical (Figure 3 gives both 4 hidden layers of 2000
units and a logistic prior) and differ only in the reported likelihood and
the paper's preprocessing, so both names are kept for discoverability.

The configs here follow the *paper*, which is what the reported
likelihoods belong to.  The released reference configs diverge from it in
more places than one:

* TFD — 8 coupling layers of 2000 units with a logistic prior, against
  Figure 3's 4 layers of 5000 with a gaussian one.
* SVHN — 3 hidden layers per coupling network, against Figure 3's 4.
* CIFAR-10 — 8 coupling layers, 2400 units, a normal prior, and a
  learning rate of 2e-4 where the others use 1e-3.
* Input reordering is used only for MNIST and TFD.
* All four train with RMSProp + momentum, not the AdaM of §5.1.

Layer count, prior, reordering and width are all reachable through
overrides — e.g.
``create_model("nice_cifar", num_coupling_layers=8, hidden_dim=2400,
prior="gaussian", input_reorder="none")`` — with one exception: those
configs give each coupling network a *different* depth (CIFAR-10 runs
1, 1, 3, 3, 2, 2, 1, 1 hidden layers), whereas ``num_hidden_layers``
applies one depth to every layer, as the paper describes.

Every factory consumes and produces flat ``(B, D)`` batches — the dataset
names describe which experiment the hyperparameters come from, not an
image layout the flow knows about.

Smaller test-scale models go through ``create_model("nice_mnist",
input_dim=..., hidden_dim=..., num_hidden_layers=...)``.
"""

from lucid.models._registry import register_model
from lucid.models.generative.nice._config import NICEConfig
from lucid.models.generative.nice._model import NICEForImageGeneration, NICEModel
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# Paper Figure 3 — MNIST: 784 dims, 5 hidden layers of 1000, logistic prior.
_CFG_MNIST = NICEConfig(
    input_dim=784,
    num_coupling_layers=4,
    num_hidden_layers=5,
    hidden_dim=1_000,
    prior="logistic",
)

# Paper Figure 3 — TFD: 2304 dims, 4 hidden layers of 5000, normal prior.
_CFG_TFD = NICEConfig(
    input_dim=2_304,
    num_coupling_layers=4,
    num_hidden_layers=4,
    hidden_dim=5_000,
    prior="gaussian",
)

# Paper Figure 3 — SVHN: 3072 dims, 4 hidden layers of 2000, logistic prior.
_CFG_SVHN = NICEConfig(
    input_dim=3_072,
    num_coupling_layers=4,
    num_hidden_layers=4,
    hidden_dim=2_000,
    prior="logistic",
)

# Paper Figure 3 — CIFAR-10: same topology as SVHN.
_CFG_CIFAR = NICEConfig(
    input_dim=3_072,
    num_coupling_layers=4,
    num_hidden_layers=4,
    hidden_dim=2_000,
    prior="logistic",
)


# ── Bare flows ────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="nice",
    model_type="nice",
    model_class=NICEModel,
    default_config=_CFG_MNIST,
    params=19158352,
)
def nice_mnist(pretrained: bool = False, **overrides: object) -> NICEModel:
    r"""Construct the NICE flow for the MNIST setup.

    Paper-faithful MNIST configuration from Dinh et al., 2014 Figure 3:
    784 dimensions, four additive coupling layers whose coupling networks
    have five hidden layers of 1000 rectified units, an odd/even input
    partition, and a standard logistic prior.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``nice_mnist``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NICEConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    NICEModel
        Bare bijection configured with the MNIST setup and any overrides.

    Notes
    -----
    Reference: Dinh, Krueger, and Bengio, *"NICE: Non-linear Independent
    Components Estimation"*, ICLR Workshop, 2015 (arXiv:1410.8516),
    Figure 3.  Reported test log-likelihood: 1980.50 nats, on data
    dequantised with uniform ``1/256`` noise and rescaled to
    :math:`[0, 1]^D` (no whitening for MNIST).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.nice import nice_mnist
    >>> model = nice_mnist(input_dim=64, hidden_dim=16).eval()
    >>> x = lucid.rand((1, 64))
    >>> h, log_det = model.encode(x)
    >>> h.shape, log_det.shape
    ((1, 64), (1,))
    """
    if pretrained:
        reject_unavailable_pretrained("nice_mnist")
    return NICEModel(apply_overrides(_CFG_MNIST, overrides))


@register_model(
    task="base",
    family="nice",
    model_type="nice",
    model_class=NICEModel,
    default_config=_CFG_TFD,
    params=346166912,
)
def nice_tfd(pretrained: bool = False, **overrides: object) -> NICEModel:
    r"""Construct the NICE flow for the Toronto Face Database setup.

    Paper-faithful TFD configuration from Dinh et al., 2014 Figure 3:
    2304 dimensions, four coupling layers with four hidden layers of
    5000 units each, and — uniquely among the four experiments — a
    standard *normal* prior.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``nice_tfd``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NICEConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    NICEModel
        Bare bijection configured with the TFD setup and any overrides.

    Notes
    -----
    Reference: Dinh, Krueger, and Bengio, *"NICE: Non-linear Independent
    Components Estimation"*, ICLR Workshop, 2015 (arXiv:1410.8516),
    Figure 3.  Reported test log-likelihood: 5514.71 nats, on
    approximately whitened data — well ahead of the 5250 nats of the Deep
    MFA bound it was compared against.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.nice import nice_tfd
    >>> model = nice_tfd(input_dim=64, hidden_dim=16).eval()
    >>> model.prior
    'gaussian'
    >>> model.input_dim
    64
    """
    if pretrained:
        reject_unavailable_pretrained("nice_tfd")
    return NICEModel(apply_overrides(_CFG_TFD, overrides))


@register_model(
    task="base",
    family="nice",
    model_type="nice",
    model_class=NICEModel,
    default_config=_CFG_SVHN,
    params=72617216,
)
def nice_svhn(pretrained: bool = False, **overrides: object) -> NICEModel:
    r"""Construct the NICE flow for the SVHN setup.

    Paper-faithful SVHN configuration from Dinh et al., 2014 Figure 3:
    3072 dimensions, four coupling layers with four hidden layers of
    2000 units each, and a standard logistic prior.  Identical in
    topology to :func:`nice_cifar` — the two experiments differ in data
    and reported likelihood only.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``nice_svhn``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NICEConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    NICEModel
        Bare bijection configured with the SVHN setup and any overrides.

    Notes
    -----
    Reference: Dinh, Krueger, and Bengio, *"NICE: Non-linear Independent
    Components Estimation"*, ICLR Workshop, 2015 (arXiv:1410.8516),
    Figure 3.  Reported test log-likelihood: 11496.55 nats, on
    ZCA-whitened data — the whitening is what makes this number so much
    larger than the CIFAR-10 one, so likelihoods are comparable only
    within a preprocessing pipeline.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.nice import nice_svhn
    >>> model = nice_svhn(input_dim=64, hidden_dim=16).eval()
    >>> model.input_dim
    64
    """
    if pretrained:
        reject_unavailable_pretrained("nice_svhn")
    return NICEModel(apply_overrides(_CFG_SVHN, overrides))


@register_model(
    task="base",
    family="nice",
    model_type="nice",
    model_class=NICEModel,
    default_config=_CFG_CIFAR,
    params=72617216,
)
def nice_cifar(pretrained: bool = False, **overrides: object) -> NICEModel:
    r"""Construct the NICE flow for the CIFAR-10 setup.

    Paper-faithful CIFAR-10 configuration from Dinh et al., 2014
    Figure 3: 3072 dimensions, four coupling layers with four hidden
    layers of 2000 units each, and a standard logistic prior.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``nice_cifar``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NICEConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    NICEModel
        Bare bijection configured with the CIFAR-10 setup and any
        overrides.

    Notes
    -----
    Reference: Dinh, Krueger, and Bengio, *"NICE: Non-linear Independent
    Components Estimation"*, ICLR Workshop, 2015 (arXiv:1410.8516),
    Figure 3.  Reported test log-likelihood: 5371.78 nats, on ZCA-whitened
    data dequantised with uniform ``1/128`` noise and rescaled to
    :math:`[-1, 1]^D` (CIFAR-10 is the one dataset given the wider
    range).  The released reference config for this dataset instead
    stacks 8 coupling layers of 2400 units with a normal prior and no
    input reordering — see the module docstring for how far overrides
    reproduce it.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.nice import nice_cifar
    >>> model = nice_cifar(input_dim=64, hidden_dim=16).eval()
    >>> x = lucid.rand((1, 64)) * 2.0 - 1.0
    >>> model.log_prob(x).shape
    (1,)
    """
    if pretrained:
        reject_unavailable_pretrained("nice_cifar")
    return NICEModel(apply_overrides(_CFG_CIFAR, overrides))


# ── Image-generation heads ───────────────────────────────────────────────────


@register_model(
    task="image-generation",
    family="nice",
    model_type="nice",
    model_class=NICEForImageGeneration,
    default_config=_CFG_MNIST,
    params=19158352,
)
def nice_mnist_gen(pretrained: bool = False, **overrides: object) -> NICEForImageGeneration:
    r"""NICE generator for the MNIST setup (NLL loss + ``.generate()``).

    Same bijection as :func:`nice_mnist`, wrapped with the exact
    maximum-likelihood objective and the ancestral sampler.  Sampling is a
    single parallel pass: draw :math:`h` from the logistic prior, return
    :math:`f^{-1}(h)`.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``nice_mnist_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NICEConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    NICEForImageGeneration
        Generator configured with the MNIST setup and any overrides.

    Notes
    -----
    Reference: Dinh, Krueger, and Bengio, *"NICE: Non-linear Independent
    Components Estimation"*, ICLR Workshop, 2015 (arXiv:1410.8516),
    Figure 3.  Samples come back in the dequantised training space, not in
    ``[0, 255]``.

    Examples
    --------
    >>> from lucid.models.generative.nice import nice_mnist_gen
    >>> model = nice_mnist_gen(input_dim=64, hidden_dim=16).eval()
    >>> model.generate(n_samples=4).samples.shape
    (4, 64)
    """
    if pretrained:
        reject_unavailable_pretrained("nice_mnist_gen")
    return NICEForImageGeneration(apply_overrides(_CFG_MNIST, overrides))


@register_model(
    task="image-generation",
    family="nice",
    model_type="nice",
    model_class=NICEForImageGeneration,
    default_config=_CFG_TFD,
    params=346166912,
)
def nice_tfd_gen(pretrained: bool = False, **overrides: object) -> NICEForImageGeneration:
    r"""NICE generator for the TFD setup (NLL loss + ``.generate()``).

    Same bijection as :func:`nice_tfd` — four coupling layers of four
    5000-unit hidden layers over 2304 dims, with a standard normal prior
    — wrapped with the exact maximum-likelihood objective and the
    ancestral sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``nice_tfd_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NICEConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    NICEForImageGeneration
        Generator configured with the TFD setup and any overrides.

    Notes
    -----
    Reference: Dinh, Krueger, and Bengio, *"NICE: Non-linear Independent
    Components Estimation"*, ICLR Workshop, 2015 (arXiv:1410.8516),
    Figure 3.  Samples are in the approximately whitened space the model
    trains on — invert the whitening to view them as images.

    Examples
    --------
    >>> from lucid.models.generative.nice import nice_tfd_gen
    >>> model = nice_tfd_gen(input_dim=64, hidden_dim=16).eval()
    >>> model.generate(n_samples=2).samples.shape
    (2, 64)
    """
    if pretrained:
        reject_unavailable_pretrained("nice_tfd_gen")
    return NICEForImageGeneration(apply_overrides(_CFG_TFD, overrides))


@register_model(
    task="image-generation",
    family="nice",
    model_type="nice",
    model_class=NICEForImageGeneration,
    default_config=_CFG_SVHN,
    params=72617216,
)
def nice_svhn_gen(pretrained: bool = False, **overrides: object) -> NICEForImageGeneration:
    r"""NICE generator for the SVHN setup (NLL loss + ``.generate()``).

    Same bijection as :func:`nice_svhn`, wrapped with the exact
    maximum-likelihood objective and the ancestral sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``nice_svhn_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NICEConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    NICEForImageGeneration
        Generator configured with the SVHN setup and any overrides.

    Notes
    -----
    Reference: Dinh, Krueger, and Bengio, *"NICE: Non-linear Independent
    Components Estimation"*, ICLR Workshop, 2015 (arXiv:1410.8516),
    Figure 3.  Samples live in the ZCA-whitened space; apply the inverse
    ZCA transform to view them as images.

    Examples
    --------
    >>> from lucid.models.generative.nice import nice_svhn_gen
    >>> model = nice_svhn_gen().eval()
    >>> model.generate(n_samples=2).samples.shape
    (2, 3072)
    """
    if pretrained:
        reject_unavailable_pretrained("nice_svhn_gen")
    return NICEForImageGeneration(apply_overrides(_CFG_SVHN, overrides))


@register_model(
    task="image-generation",
    family="nice",
    model_type="nice",
    model_class=NICEForImageGeneration,
    default_config=_CFG_CIFAR,
    params=72617216,
)
def nice_cifar_gen(pretrained: bool = False, **overrides: object) -> NICEForImageGeneration:
    r"""NICE generator for the CIFAR-10 setup (NLL loss + ``.generate()``).

    Same bijection as :func:`nice_cifar`, wrapped with the exact
    maximum-likelihood objective and the ancestral sampler.  This is the
    weakest of the four experiments in perceptual terms — NICE's
    volume-preserving couplings cannot model natural-image texture — and
    the reason RealNVP replaced additive couplings with affine ones.

    Parameters
    ----------
    pretrained : bool, default=False
        No checkpoint is published for ``nice_cifar_gen``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Optional :class:`NICEConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    NICEForImageGeneration
        Generator configured with the CIFAR-10 setup and any overrides.

    Notes
    -----
    Reference: Dinh, Krueger, and Bengio, *"NICE: Non-linear Independent
    Components Estimation"*, ICLR Workshop, 2015 (arXiv:1410.8516),
    Figure 3.  Reported test log-likelihood: 5371.78 nats on ZCA-whitened
    data in :math:`[-1, 1]^D`.

    Examples
    --------
    >>> from lucid.models.generative.nice import nice_cifar_gen
    >>> model = nice_cifar_gen().eval()
    >>> out = model.generate(n_samples=2)
    >>> out.samples.shape
    (2, 3072)
    """
    if pretrained:
        reject_unavailable_pretrained("nice_cifar_gen")
    return NICEForImageGeneration(apply_overrides(_CFG_CIFAR, overrides))
