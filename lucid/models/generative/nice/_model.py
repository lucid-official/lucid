"""NICE model — additive-coupling normalizing flow (Dinh et al., 2014).

The flow is defined on flat ``D``-vectors, as in the paper (whose
experiments are parametrised by a dimension count) and in the released
code (whose input space is a ``VectorSpace``).  Callers flatten image data
themselves; nothing here assumes a pixel grid.

Shape of the flow (paper §5.1, in order):

    x (B, D) → tile-reorder (odd/even partition)
      → [additive coupling → reverse indices] × num_coupling_layers
      → diagonal scaling exp(s)
      → h (B, D)

Only the last stage changes volume, so ``log|det ∂f/∂x| = Σ_i s_i`` and the
exact log-likelihood is one forward pass away.  Generation runs the same
graph backwards from a prior sample, which is why ``decode`` is not an
approximate decoder but the true inverse.
"""

from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageGenerationModel
from lucid.models._output import GenerationOutput, NormalizingFlowOutput
from lucid.models._utils._generative import flow_prior_log_prob, generative_activation, start_noise
from lucid.models.generative.nice._config import NICEConfig

# ─────────────────────────────────────────────────────────────────────────────
# Volume-preserving index permutations (pure view ops, no gather)
# ─────────────────────────────────────────────────────────────────────────────


def _tile(x: Tensor) -> Tensor:
    """Send even indices to the first half, odd indices to the second.

    ``[x0, x1, x2, x3] → [x0, x2, x1, x3]`` — the ``mode="tile"``
    reordering of the official release, expressed as reshape + swap so it
    stays a view chain instead of a gather.
    """
    B, D = int(x.shape[0]), int(x.shape[1])
    return x.reshape(B, D // 2, 2).swapaxes(1, 2).reshape(B, D)


def _untile(x: Tensor) -> Tensor:
    """Inverse of :func:`_tile`."""
    B, D = int(x.shape[0]), int(x.shape[1])
    return x.reshape(B, 2, D // 2).swapaxes(1, 2).reshape(B, D)


# ─────────────────────────────────────────────────────────────────────────────
# Bijective building blocks
# ─────────────────────────────────────────────────────────────────────────────


@final
class _CouplingNet(nn.Module):
    """The coupling function ``m`` — a deep rectified net with linear output.

    ``m`` is unconstrained: it never has to be inverted, because the
    coupling layer around it only ever *adds* its output.  Initialisation
    follows the official release (uniform ``irange = 0.01``, zero biases),
    which keeps the flow near the identity at step 0.
    """

    def __init__(self, config: NICEConfig) -> None:
        super().__init__()
        self._act_name = config.act_fn

        split = config.split
        hidden: list[nn.Linear] = []
        in_dim = split
        for _ in range(config.num_hidden_layers):
            hidden.append(nn.Linear(in_dim, config.hidden_dim))
            in_dim = config.hidden_dim
        proj = nn.Linear(config.hidden_dim, config.input_dim - split)

        for linear in [*hidden, proj]:
            nn.init.uniform_(linear.weight, -config.init_range, config.init_range)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

        self.hidden = nn.ModuleList(list(hidden))
        self.proj = proj

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = x
        for layer in self.hidden:
            h = generative_activation(self._act_name, layer(h))
        return self.proj(h)


@final
class _AdditiveCoupling(nn.Module):
    r"""One additive coupling layer followed by an index reversal.

    Forward: :math:`y_{I_1} = x_{I_1}`,
    :math:`y_{I_2} = x_{I_2} + m(x_{I_1})`, then the ``D`` indices are
    reversed.  The reversal is what exchanges the roles of the two blocks
    between consecutive layers (the ``reverse=True`` default of the
    official release); on its own the coupling would leave the first block
    untouched forever.

    Both stages are volume-preserving — unit Jacobian determinant — so the
    layer has nothing to report to the log-likelihood.
    """

    def __init__(self, config: NICEConfig) -> None:
        super().__init__()
        self._split = config.split
        self.coupling = _CouplingNet(config)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x[:, : self._split], x[:, self._split :]
        y = lucid.cat([x1, x2 + self.coupling(x1)], dim=-1)
        return lucid.flip(y, 1)

    def inverse(self, y: Tensor) -> Tensor:
        """Exact inverse — undo the reversal, then subtract ``m``."""
        h = lucid.flip(y, 1)
        h1, h2 = h[:, : self._split], h[:, self._split :]
        return lucid.cat([h1, h2 - self.coupling(h1)], dim=-1)


@final
class _DiagonalScaling(nn.Module):
    r"""Learned per-dimension scaling :math:`h = \exp(s) \odot x`.

    The paper's top layer and the only non-volume-preserving stage of the
    flow: :math:`\log|\det| = \sum_i s_i`.  Parametrised exponentially so
    the scale stays positive under unconstrained optimisation, and
    initialised at :math:`s = 0` (identity), as in the official release.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(lucid.zeros(dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x * self.log_scale.exp()

    def inverse(self, h: Tensor) -> Tensor:
        """Exact inverse — divide out the scale."""
        return h * (-self.log_scale).exp()

    def log_det_jacobian(self) -> Tensor:
        r"""Scalar :math:`\sum_i s_i` — identical for every sample."""
        return self.log_scale.sum()


# ─────────────────────────────────────────────────────────────────────────────
# NICE model — the bare bijection + its prior
# ─────────────────────────────────────────────────────────────────────────────


class NICEModel(PretrainedModel):
    r"""Bare NICE flow — invertible map plus the factorised latent prior.

    The prior is part of the model rather than of the training loop
    (matching the official ``NICE(encoder, prior)`` construction): the
    flow only becomes a density once the latent space is measured against
    :math:`p_H`.  Three entry points cover the two directions and the
    density:

    * ``encode(x) -> (h, log_det)`` — data to latent, with the per-sample
      log-determinant of the transformation.
    * ``decode(h) -> x`` — the *exact* inverse, back to ``(B, D)``.
    * ``log_prob(x) -> (B,)`` — exact log-likelihood in nats.

    Data and latent are both flat ``(B, D)`` — the flow has no notion of
    spatial structure, so image batches are flattened by the caller.

    Parameters
    ----------
    config : NICEConfig
        Architecture and prior.  ``config.input_dim`` must be even; see
        :class:`NICEConfig` for the full field list.

    Attributes
    ----------
    couplings : nn.ModuleList
        ``config.num_coupling_layers`` additive coupling layers, each
        followed by an index reversal.
    scaling : nn.Module
        Diagonal scaling :math:`\exp(s)` closing the stack — the only
        stage contributing to the log-determinant.

    Notes
    -----
    Reference: Dinh, Krueger, and Bengio, *"NICE: Non-linear Independent
    Components Estimation"*, ICLR Workshop, 2015 (arXiv:1410.8516).

    The exact objective, with additive couplings contributing zero:

    .. math::

        \log p_X(x) = \sum_{i=1}^{D}
            \Big[\log p_{H_i}\big(f_i(x)\big) + s_i\Big].

    The model expects *dequantised* data — the paper adds uniform noise of
    ``1/256`` and rescales to :math:`[0, 1]^D` (:math:`1/128` and
    :math:`[-1, 1]^D` for CIFAR-10), and applies ZCA (SVHN, CIFAR-10) or
    approximate whitening (TFD) beforehand.  Feeding raw 8-bit integers
    lets the flow put unbounded density on the quantisation lattice, so
    the likelihood diverges.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.nice import NICEConfig, NICEModel
    >>> cfg = NICEConfig(input_dim=64, num_coupling_layers=2,
    ...                  num_hidden_layers=1, hidden_dim=16)
    >>> model = NICEModel(cfg).eval()
    >>> x = lucid.rand((2, 64))
    >>> h, log_det = model.encode(x)
    >>> h.shape, log_det.shape
    ((2, 64), (2,))
    >>> model.log_prob(x).shape
    (2,)
    """

    config_class: ClassVar[type[NICEConfig]] = NICEConfig
    base_model_prefix: ClassVar[str] = "nice"

    def __init__(self, config: NICEConfig) -> None:
        super().__init__(config)
        self._prior = config.prior
        self._reorder = config.input_reorder
        self._input_dim = config.input_dim

        self.couplings = nn.ModuleList(
            [_AdditiveCoupling(config) for _ in range(config.num_coupling_layers)]
        )
        self.scaling = _DiagonalScaling(config.input_dim)

    @property
    def input_dim(self) -> int:
        """Flattened dimensionality ``D`` the bijection acts on."""
        return self._input_dim

    @property
    def prior(self) -> str:
        """Name of the factorised latent prior (``"logistic"`` / ``"gaussian"``)."""
        return self._prior

    def _check_shape(self, x: Tensor, what: str) -> None:
        if x.ndim != 2 or int(x.shape[1]) != self._input_dim:
            raise ValueError(
                f"NICE expects (B, {self._input_dim}) {what}, " f"got shape {tuple(x.shape)}"
            )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        r"""Map data to the latent space.

        Parameters
        ----------
        x : Tensor
            Dequantised samples, shape ``(B, D)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(h, log_det)`` with ``h`` of shape ``(B, D)`` and
            ``log_det`` of shape ``(B,)``.  Every coupling layer and
            permutation contributes exactly zero to ``log_det``, so it is
            the diagonal scaling's :math:`\sum_i s_i` broadcast across the
            batch.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> from lucid.models.generative.nice import NICEConfig, NICEModel
        >>> cfg = NICEConfig(input_dim=64, num_coupling_layers=2,
        ...                  num_hidden_layers=1, hidden_dim=16)
        >>> model = NICEModel(cfg).eval()
        >>> _ = nn.init.normal_(model.scaling.log_scale, std=0.1)  # s = 0 at init
        >>> x = lucid.rand((2, 64))
        >>> h, log_det = model.encode(x)
        >>> h.shape, log_det.shape
        ((2, 64), (2,))
        >>> bool((log_det == model.scaling.log_scale.sum()).all().item())
        True
        >>> bool(lucid.allclose(model.decode(h), x, atol=1e-5))  # exact inverse
        True
        """
        self._check_shape(x, "samples")
        h = x
        if self._reorder == "tile":
            h = _tile(h)
        for layer in self.couplings:
            h = layer(h)
        h = self.scaling(h)

        batch = int(x.shape[0])
        log_det = self.scaling.log_det_jacobian()
        return h, log_det.reshape(1).expand(batch)

    def decode(self, h: Tensor) -> Tensor:
        """Invert the flow — latent ``(B, D)`` back to data ``(B, D)``.

        Exact rather than approximate: every stage is run backwards, so
        ``decode(encode(x)[0])`` recovers ``x`` up to float round-off.
        """
        self._check_shape(h, "latents")
        x = self.scaling.inverse(h)
        for idx in range(len(self.couplings) - 1, -1, -1):
            x = self.couplings[idx].inverse(x)
        if self._reorder == "tile":
            x = _untile(x)
        return x

    def _density(self, h: Tensor, log_det: Tensor) -> Tensor:
        """Change-of-variables density from a latent and its log-determinant."""
        return flow_prior_log_prob(self._prior, h).sum(dim=-1) + log_det

    def log_prob(self, x: Tensor) -> Tensor:
        r"""Exact per-sample log-likelihood :math:`\log p_X(x)` in nats."""
        h, log_det = self.encode(x)
        return self._density(h, log_det)

    @override
    def forward(self, x: Tensor) -> NormalizingFlowOutput:
        h, log_det = self.encode(x)
        return NormalizingFlowOutput(
            latent=h, log_det_jacobian=log_det, log_prob=self._density(h, log_det)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper — maximum-likelihood loss + ancestral sampling
# ─────────────────────────────────────────────────────────────────────────────


class NICEForImageGeneration(ImageGenerationModel):
    r"""NICE with the exact maximum-likelihood loss and ``.generate()``.

    Wraps :class:`NICEModel` with the paper's training objective — plain
    negative log-likelihood, no bound and no auxiliary term — and the
    ancestral sampler.  ``forward(x)`` returns a
    :class:`NormalizingFlowOutput` whose ``loss`` is
    ``-log_prob.mean()``; ``generate(n_samples)`` draws
    :math:`h \sim p_H` and returns :math:`f^{-1}(h)` in a single parallel
    pass.

    Parameters
    ----------
    config : NICEConfig
        Architecture and prior; see :class:`NICEConfig`.

    Attributes
    ----------
    nice : NICEModel
        Underlying bijection, exposing ``encode`` / ``decode`` /
        ``log_prob``.

    Notes
    -----
    Reference: Dinh, Krueger, and Bengio, *"NICE: Non-linear Independent
    Components Estimation"*, ICLR Workshop, 2015 (arXiv:1410.8516).

    Training objective (maximised in the paper, minimised here):

    .. math::

        \mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B}
            \Big[\log p_H\big(f(x^{(b)})\big) + \sum_{i=1}^{D} s_i\Big].

    The paper optimises this with Adam (learning rate :math:`10^{-3}`)
    for 1500 epochs and reports test log-likelihoods of 1980.50 (MNIST),
    5514.71 (TFD), 11496.55 (SVHN) and 5371.78 (CIFAR-10) nats — figures
    that are only comparable under the paper's own preprocessing.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.nice import (
    ...     NICEConfig, NICEForImageGeneration,
    ... )
    >>> cfg = NICEConfig(input_dim=64, num_coupling_layers=2,
    ...                  num_hidden_layers=1, hidden_dim=16)
    >>> model = NICEForImageGeneration(cfg).eval()
    >>> out = model(lucid.rand((2, 64)))
    >>> out.loss.shape                       # scalar NLL
    ()
    >>> model.generate(n_samples=3).samples.shape
    (3, 64)
    """

    config_class: ClassVar[type[NICEConfig]] = NICEConfig
    base_model_prefix: ClassVar[str] = "nice"

    def __init__(self, config: NICEConfig) -> None:
        super().__init__(config)
        self.nice = NICEModel(config)
        self._prior = config.prior
        self._input_dim = config.input_dim

    @override
    def forward(self, x: Tensor) -> NormalizingFlowOutput:
        out = self.nice(x)
        return NormalizingFlowOutput(
            latent=out.latent,
            log_det_jacobian=out.log_det_jacobian,
            log_prob=out.log_prob,
            loss=-out.log_prob.mean(),
        )

    @lucid.no_grad()
    def generate(self, n_samples: int = 1, *, device: str | None = None) -> GenerationOutput:
        """Sample ``n_samples`` vectors by inverting a prior draw.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to draw.
        device : str, optional
            Where to allocate the prior sample.  Defaults to the device the
            model's parameters are on.

        Returns
        -------
        GenerationOutput
            ``samples`` of shape ``(n_samples, D)``, in the same
            (dequantised, possibly whitened) space the model was trained
            on — no squashing is applied.  Reshape to the dataset's image
            grid for display.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.nice import (
        ...     NICEConfig, NICEForImageGeneration,
        ... )
        >>> cfg = NICEConfig(input_dim=64, num_coupling_layers=2,
        ...                  num_hidden_layers=1, hidden_dim=16)
        >>> model = NICEForImageGeneration(cfg).eval()
        >>> samples = model.generate(n_samples=3).samples
        >>> samples.shape  # flat, like the training data
        (3, 64)
        >>> samples.reshape(3, 1, 8, 8).shape  # onto an 8x8 grid for display
        (3, 1, 8, 8)
        >>> _ = lucid.manual_seed(0)
        >>> first = model.generate(n_samples=2).samples
        >>> _ = lucid.manual_seed(0)  # the prior draw is the only randomness
        >>> bool(lucid.allclose(first, model.generate(n_samples=2).samples))
        True
        """
        h = start_noise(self, (n_samples, self._input_dim), device, prior=self._prior)
        return GenerationOutput(samples=self.nice.decode(h))
