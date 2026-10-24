"""Neural ODE model — a continuous normalizing flow (Chen et al., 2018).

Shape of the flow, in order:

    x (B, C, H, W)
      → flatten to z(0) of shape (B, D)
      → integrate [z, Δ] from t = 0 to t = 1 under
            dz/dt = f(t, z),      dΔ/dt = tr(∂f/∂z)
      → z(1) is the latent, Δ(1) is the flow's log-determinant

Nothing about that is a stack of layers.  There is one vector field, and
:mod:`lucid.diffeq` decides how many evaluations of it the requested
accuracy needs — which is why :attr:`NeuralODEModel.nfe` is worth
watching during training, and why ``decode`` needs no hand-written
inverse: it is the same field integrated from ``t = 1`` back to ``t = 0``.
"""

import math
from collections.abc import Callable
from typing import ClassVar, cast, final, override

import lucid
import lucid.autograd
import lucid.diffeq
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageGenerationModel
from lucid.models._output import GenerationOutput, NormalizingFlowOutput
from lucid.models._utils._generative import (
    exact_divergence,
    flow_prior_log_prob,
    generative_activation,
    hutchinson_divergence,
    start_noise,
    trace_probe,
)
from lucid.models.generative.neural_ode._config import NeuralODEConfig

# What ``lucid.diffeq`` calls a state: one tensor, or a tuple of them when
# more than one quantity is integrated at once — here the sample and its
# accumulated log-density.
_State = Tensor | tuple[Tensor, ...]

# ─────────────────────────────────────────────────────────────────────────────
# Vector field — the paper's gated sum
# ─────────────────────────────────────────────────────────────────────────────


@final
class _GatedBlock(nn.Module):
    r"""One :math:`\sigma_n(t)\, f_n(\mathbf{z})` term of the vector field.

    The paper writes the dynamics as a sum of functions with learned
    time-dependent gates in :math:`(0, 1)` — a hypernetwork, so that what
    the field *does* changes along the trajectory instead of being one
    fixed map applied for longer.  Here the gate scales the block's linear
    response and a second time-driven term shifts it, which is the form
    that makes :math:`t` reach every unit rather than only the first
    layer.

    Only the ``z`` term is a matrix product.  The two time-driven terms
    take a *scalar* input, so writing them as dense layers would put two
    ``(1, 1) x (1, out)`` GEMMs on the hot path for every stage of every
    step — pure dispatch overhead, and measurably the largest single cost
    in the right-hand side.  Broadcasting a parameter against ``t``
    computes the same thing without the call.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        # Same statistics a dense layer of fan-in one would have had.
        bound = 1.0
        self.gate_weight = nn.Parameter(lucid.rand((1, out_dim)) * 2 * bound - bound)
        self.gate_bias = nn.Parameter(lucid.rand((1, out_dim)) * 2 * bound - bound)
        self.shift_weight = nn.Parameter(lucid.rand((1, out_dim)) * 2 * bound - bound)

    @override
    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        # t is 0-D, so each term broadcasts to (1, out) and then over the
        # batch — three elementwise ops in place of two matrix products.
        gate = lucid.sigmoid(self.gate_weight * t + self.gate_bias)
        return self.linear(z) * gate + self.shift_weight * t


@final
class _PlanarField(nn.Module):
    r"""Paper eq. (9) + (10): a gated sum of planar units.

    Equation 9 gives one unit as
    :math:`\mathbf{u}\, h(\mathbf{w}^{\top}\mathbf{z} + b)` — note the
    nonlinearity takes a *scalar*.  Equation 10 sums :math:`M` of them,
    each scaled by its own time gate :math:`\sigma_n(t) \in (0, 1)`:

    .. math::

        \frac{d\mathbf{z}}{dt}
            = \sum_{n=1}^{M} \sigma_n(t)\,
              \mathbf{u}_n\, h(\mathbf{w}_n^{\top}\mathbf{z} + b_n).

    The whole layer is two matrix products against ``(M, D)`` parameter
    blocks, so the ``M`` units cost one GEMM rather than ``M`` of them.
    Following the authors' ``examples/cnf.py``, the units are averaged
    rather than summed — the same field up to a constant, but it keeps the
    magnitude independent of ``M``.
    """

    def __init__(self, config: NeuralODEConfig) -> None:
        super().__init__()
        self._act_name = config.act_fn
        d, m = config.data_dim, config.hidden_dim
        bound = 1.0 / math.sqrt(d)
        self.w = nn.Parameter(lucid.rand((m, d)) * 2 * bound - bound)
        self.b = nn.Parameter(lucid.rand((1, m)) * 2 * bound - bound)
        self.u = nn.Parameter(lucid.rand((m, d)) * 2 * bound - bound)
        # One gate per unit, driven by t.  Scalar-input dense layers would be
        # two tiny GEMMs on the solver's hot path; broadcasting is the same
        # arithmetic without the dispatch.
        self.gate_weight = nn.Parameter(lucid.rand((1, m)) * 2 - 1.0)
        self.gate_bias = nn.Parameter(lucid.rand((1, m)) * 2 - 1.0)

    @override
    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        pre: Tensor = z @ self.w.mT + self.b  # (B, M)
        act: Tensor = generative_activation(self._act_name, pre)
        gate: Tensor = lucid.sigmoid(self.gate_weight * t + self.gate_bias)
        return (act * gate) @ self.u / float(self.w.shape[0])


@final
class _VectorField(nn.Module):
    r""":math:`f(t, \mathbf{z})` — ``num_blocks`` gated blocks and a readout.

    Width :math:`M` is ``hidden_dim``; the readout carries no activation,
    so the field can take either sign.  No masking, no ordering of the
    dimensions, no triangular structure: a continuous flow pays for its
    density with a trace rather than a determinant, and a trace puts no
    constraint on the architecture at all.
    """

    def __init__(self, config: NeuralODEConfig) -> None:
        super().__init__()
        self._act_name = config.act_fn
        widths = (
            [config.data_dim]
            + [config.hidden_dim] * config.num_blocks
            + [config.data_dim]
        )
        self.blocks = nn.ModuleList(
            [_GatedBlock(widths[i], widths[i + 1]) for i in range(len(widths) - 1)]
        )
        # The solver calls this thousands of times per epoch; the block
        # list is fixed, so materialise it once rather than per call.
        self._ordered = list(self.blocks)
        self._last = len(self._ordered) - 1

    @override
    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        h = z
        for index, block in enumerate(self._ordered):
            h = block(t, h)
            if index != self._last:
                h = generative_activation(self._act_name, h)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Augmented dynamics — state and log-density integrated together
# ─────────────────────────────────────────────────────────────────────────────


@final
class _CNFDynamics(nn.Module):
    r"""Right-hand side of the augmented system :math:`(\mathbf{z}, \Delta)`.

    Theorem 1 of the paper gives
    :math:`\partial \log p(\mathbf{z}(t)) / \partial t =
    -\operatorname{tr}(\partial f / \partial \mathbf{z})`, so integrating

    .. math::

        \frac{d\Delta}{dt} = \operatorname{tr}
            \!\left(\frac{\partial f}{\partial \mathbf{z}}\right),
        \qquad \Delta(0) = 0

    alongside the state accumulates exactly the log-determinant of the map
    :math:`x \mapsto \mathbf{z}(1)`.  Both components are handed to
    :func:`lucid.diffeq.odeint` as one tuple state, so the solver's error
    control sees the density term too.

    The trace itself comes from autograd, which is why the whole body runs
    under :func:`lucid.enable_grad` — the adjoint integrates its forward
    pass with gradients off, and the divergence would otherwise be
    unavailable exactly where it is needed.

    Attributes
    ----------
    nfe : int
        Function evaluations since the last :meth:`reset`.  The paper
        tracks this because it is the honest cost of a continuous flow:
        depth is not a hyper-parameter but whatever the solver spends.
    """

    basis: Tensor

    def __init__(self, config: NeuralODEConfig) -> None:
        super().__init__()
        self.field = (
            _PlanarField(config) if config.field == "planar" else _VectorField(config)
        )
        self._dim = config.data_dim
        self._trace_method = config.resolved_trace_method
        self._trace_noise = config.trace_noise
        self.nfe = 0
        self._eps: Tensor | None = None
        self._seeds: list[Tensor] | None = None

        if self._trace_method == "exact":
            # Rows of the identity are the seeds of the D vector-Jacobian
            # products the exact trace costs.  Not persistent: it is a
            # constant, and a checkpoint should not carry one.
            self.register_buffer("basis", lucid.eye(self._dim), persistent=False)

    def reset(self, eps: Tensor | None = None) -> None:
        """Zero the evaluation counter and fix the probe for one solve.

        The Hutchinson probe has to be drawn once and held for the whole
        solve.  Redrawing it per step would make the right-hand side
        stochastic, and an adaptive solver reading a different function on
        every attempt cannot converge — it would shrink the step to chase
        noise.

        Dropping the cached exact-trace seeds here is the other half: the
        batch is fixed within a solve but not across solves, so they are
        rebuilt on the first evaluation and then reused by every step.
        """
        self.nfe = 0
        self._eps = eps
        self._seeds = None

    def sample_noise(self, like: Tensor) -> Tensor:
        """Draw a probe vector shaped like ``like`` from the configured law."""
        return trace_probe(like, self._trace_noise)

    def velocity(self, t: Tensor, z: Tensor) -> Tensor:
        """The field alone — the state's dynamics with no density term.

        What ``decode`` integrates.  Going the other way needs no
        divergence, so the inverse costs one evaluation per step against
        the forward direction's one-plus-trace.
        """
        self.nfe += 1
        return self.field(t, z)

    @override
    def forward(self, t: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        z, _ = state
        self.nfe += 1
        with lucid.enable_grad():
            # Under the adjoint's forward sweep the state arrives detached,
            # so it has to be re-rooted before it can be differentiated.
            # Under the adjoint's *backward* sweep, and when the caller
            # differentiates through the solver directly, it is already
            # part of a live graph and must stay attached — re-rooting
            # there would silently drop the state's own gradient path.
            source = z if z.requires_grad else z.detach().requires_grad_(True)
            dz = self.field(t, source)
            divergence = self._divergence(dz, source)
        return dz, divergence

    def _divergence(self, dz: Tensor, z: Tensor) -> Tensor:
        # ``create_graph`` is only worth its cost when the trace itself has to
        # be differentiated — i.e. when the density is part of a training
        # objective.  Under ``no_grad`` (scoring: ``log_prob``,
        # ``bits_per_dim``, plain ``forward``) it demands a second derivative
        # of the field for a number nobody will backprop through, which is
        # what stops a convolutional field being scored at all.
        create_graph = lucid.is_grad_enabled()
        if self._trace_method == "exact":
            return exact_divergence(
                dz, z, self._exact_seeds(dz), create_graph=create_graph
            )
        eps = self._eps
        if eps is None:
            # A caller that reached the dynamics without going through
            # ``reset`` still gets an unbiased estimate; it just pays for a
            # draw here and holds it for the rest of the solve.
            eps = self.sample_noise(z)
            self._eps = eps
        return hutchinson_divergence(dz, z, eps, create_graph=create_graph)

    def _exact_seeds(self, dz: Tensor) -> list[Tensor]:
        """Identity rows broadcast over the batch, built once per solve.

        ``grad_outputs`` has to match the output shape exactly, so the rows
        cannot simply be broadcast in place; the batch is fixed within a
        solve, so every step after the first reuses these.
        """
        seeds = self._seeds
        if seeds is None or seeds[0].shape != dz.shape:
            basis = self.basis
            ones = lucid.ones_like(dz[:, :1])
            seeds = [
                basis[index].reshape(1, self._dim) * ones for index in range(self._dim)
            ]
            self._seeds = seeds
        return seeds


# ─────────────────────────────────────────────────────────────────────────────
# Direct model
# ─────────────────────────────────────────────────────────────────────────────


class NeuralODEModel(PretrainedModel):
    r"""Continuous normalizing flow — the generative half of Chen et al., 2018.

    ``encode`` integrates the augmented system forward and returns the
    latent together with the flow's log-determinant; ``decode`` integrates
    the same vector field with time reversed.  That symmetry is the point
    of the continuous formulation: a discrete flow has to be *built*
    invertible, whereas this one is invertible because an ODE is, and the
    inverse is the solver run backwards.

    Parameters
    ----------
    config : NeuralODEConfig
        Field width and depth, solver and tolerances, and how the trace is
        obtained.

    Attributes
    ----------
    dynamics : _CNFDynamics
        The augmented right-hand side, holding the vector field and the
        evaluation counter.

    Notes
    -----
    Reference: Chen, Rubanova, Bettencourt, and Duvenaud, *"Neural
    Ordinary Differential Equations"*, NeurIPS, 2018 (arXiv:1806.07366),
    §4.  The Hutchinson trace option follows Grathwohl et al., *"FFJORD"*,
    ICLR, 2019 (arXiv:1810.01367).

    **Where to run it.**  A solve is one small right-hand side evaluated
    tens of times, so which stream wins depends entirely on how much work
    each evaluation carries.  Measured on an M1 Pro, one training step
    (forward, loss, backward) of ``neural_ode_gen``:

    ==================  ===========  ===========
    setup               ``"cpu"``    ``"metal"``
    ==================  ===========  ===========
    B=8, D=192            16.8 ms      38.1 ms
    B=32, D=3072         175.0 ms      56.5 ms
    B=128, D=3072        587.5 ms      70.8 ms
    ==================  ===========  ===========

    Below a few hundred dimensions the tensors are too small to pay for a
    GPU dispatch and Accelerate wins; at anything resembling a real
    training batch the ratio reverses and keeps widening — ``8.3x`` at the
    bottom row, where the CPU cost grows with the batch and the GPU's
    barely does.  Train on ``"metal"``; keep the paper's two-dimensional
    density experiments on ``"cpu"``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.neural_ode import (
    ...     NeuralODEConfig, NeuralODEModel,
    ... )
    >>> cfg = NeuralODEConfig(sample_size=2, in_channels=1, out_channels=1,
    ...                       hidden_dim=8, num_blocks=1)
    >>> model = NeuralODEModel(cfg).eval()
    >>> z, log_det = model.encode(lucid.rand((2, 1, 2, 2)))
    >>> z.shape, log_det.shape
    ((2, 4), (2,))
    >>> model.decode(z).shape
    (2, 1, 2, 2)
    """

    config_class: ClassVar[type[NeuralODEConfig]] = NeuralODEConfig
    base_model_prefix: ClassVar[str] = "neural_ode"

    def __init__(self, config: NeuralODEConfig) -> None:
        super().__init__(config)
        self.dynamics = _CNFDynamics(config)
        self._nfe_forward = 0

        size = config.sample_size
        height, width = size if isinstance(size, tuple) else (size, size)
        self._image_shape = (config.in_channels, height, width)
        self._input_dim = config.data_dim
        self._prior = config.prior

        self._solver = config.solver
        self._rtol = config.rtol
        self._atol = config.atol
        self._use_adjoint = config.use_adjoint
        self._trace_method = config.resolved_trace_method
        # The adjoint needs the parameter list on every solve, and walking
        # the module tree for it each time is pure overhead in a training
        # loop where the set never changes.
        self._adjoint_params: list[Tensor] | None = None

    @property
    def input_dim(self) -> int:
        """int: Flattened dimensionality ``D`` the flow acts on."""
        return self._input_dim

    @property
    def prior(self) -> str:
        """str: Name of the factorised latent prior."""
        return self._prior

    @property
    def trace_method(self) -> str:
        """str: Whether the divergence is summed exactly or estimated."""
        return self._trace_method

    @property
    def nfe(self) -> int:
        """int: Vector-field evaluations spent by the most recent solve.

        A continuous flow has no layer count to report, so this is the
        closest thing to one — and it grows as the field stiffens during
        training, which the paper documents.

        This is the *forward* count, snapshotted when the solve returns.
        Reading the live counter instead let the adjoint's backward sweep
        keep incrementing it after the fact, so the number changed depending
        on whether ``.backward()`` had run yet — and §3 (Fig. 3c/3d) treats
        forward and backward NFE as two separate quantities.
        """
        return self._nfe_forward

    def _check_image(self, x: Tensor) -> None:
        channels, height, width = self._image_shape
        if x.ndim != 4 or tuple(int(s) for s in x.shape[1:]) != (
            channels,
            height,
            width,
        ):
            raise ValueError(
                f"Neural ODE expects (B, {channels}, {height}, {width}) samples, "
                f"got shape {tuple(x.shape)}"
            )

    def _integrate(
        self,
        func: Callable[..., object],
        state: _State,
        t0: float,
        t1: float,
    ) -> _State:
        """Run one solve, adjoint or direct, and return the endpoint only.

        The adjoint buys constant memory in the backward pass, so it is
        worth nothing when there is no backward pass to come — sampling
        and evaluation take the plain solver and skip the flattening and
        the custom autograd node it would otherwise set up.
        """
        span = [t0, t1]
        if self._use_adjoint and lucid.is_grad_enabled():
            if self._adjoint_params is None:
                self._adjoint_params = list(self.parameters())
            out = lucid.diffeq.odeint_adjoint(
                func,
                state,
                span,
                rtol=self._rtol,
                atol=self._atol,
                method=self._solver,
                adjoint_params=self._adjoint_params,
            )
            # The adjoint always reports the trajectory; the flow only ever
            # wants where it ended up.
            # Snapshot before any backward sweep can add to the live counter.
            self._nfe_forward = self.dynamics.nfe
            if isinstance(out, Tensor):
                return out[-1]
            return tuple(part[-1] for part in out)
        direct = cast(
            _State,
            lucid.diffeq.odeint(
                func,
                state,
                span,
                rtol=self._rtol,
                atol=self._atol,
                method=self._solver,
                return_trajectory=False,
            ),
        )
        self._nfe_forward = self.dynamics.nfe
        return direct

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        r"""Map samples to the latent space, accumulating the log-determinant.

        Integrates :math:`(\mathbf{z}, \Delta)` from ``t = 0`` to ``t = 1``
        with :math:`\Delta(0) = 0`, so that

        .. math::

            \log p_X(x) = \log p_Z(\mathbf{z}(1)) + \Delta(1).

        Parameters
        ----------
        x : Tensor
            Samples of shape ``(B, C, H, W)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(z, log_det)`` with ``z`` of shape ``(B, D)`` and ``log_det``
            of shape ``(B,)``.

        Notes
        -----
        When ``trace_method`` resolves to ``"hutchinson"`` the returned
        ``log_det`` is an unbiased *estimate*, not the exact value; its
        expectation is right, which is what a maximum-likelihood gradient
        needs, but two calls on the same input will not agree.

        Examples
        --------
        >>> import math
        >>> import lucid
        >>> from lucid.models.generative.neural_ode import (
        ...     NeuralODEConfig, NeuralODEModel,
        ... )
        >>> cfg = NeuralODEConfig(sample_size=2, in_channels=1, out_channels=1,
        ...                       hidden_dim=8, num_blocks=1)
        >>> model = NeuralODEModel(cfg).eval()
        >>> x = lucid.rand((2, 1, 2, 2))
        >>> z, log_det = model.encode(x)
        >>> z.shape, log_det.shape
        ((2, 4), (2,))
        >>> model.trace_method  # D = 4 is small enough for the exact trace
        'exact'
        >>> const = 0.5 * model.input_dim * math.log(2 * math.pi)
        >>> log_pz = -0.5 * (z**2).sum(dim=-1) - const  # the Gaussian prior
        >>> bool(lucid.allclose(model.log_prob(x), log_pz + log_det))
        True
        >>> bool(lucid.allclose(model.decode(z), x, atol=1e-4))  # solve it backwards
        True
        """
        self._check_image(x)
        batch = int(x.shape[0])
        z0 = x.reshape(batch, -1)
        delta0 = lucid.zeros((batch,), dtype=z0.dtype, device=z0.device)

        eps = None if self._trace_method == "exact" else self.dynamics.sample_noise(z0)
        self.dynamics.reset(eps)

        z1, delta1 = cast(
            tuple[Tensor, Tensor],
            self._integrate(self.dynamics, (z0, delta0), 0.0, 1.0),
        )
        return z1, delta1

    def decode(self, z: Tensor) -> Tensor:
        """Invert the flow — latent ``(B, D)`` back to samples ``(B, C, H, W)``.

        The same vector field, integrated from ``t = 1`` to ``t = 0``.  No
        separate inverse exists to get wrong, so the round trip is exact up
        to the tolerances the solver was given; tighten ``rtol`` / ``atol``
        and it tightens with them.
        """
        if z.ndim != 2 or int(z.shape[1]) != self._input_dim:
            raise ValueError(
                f"Neural ODE expects (B, {self._input_dim}) latents, "
                f"got shape {tuple(z.shape)}"
            )
        batch = int(z.shape[0])
        self.dynamics.reset()
        z0 = cast(Tensor, self._integrate(self.dynamics.velocity, z, 1.0, 0.0))
        return z0.reshape(batch, *self._image_shape)

    def log_prob(self, x: Tensor) -> Tensor:
        r"""Per-sample log-likelihood :math:`\log p_X(x)` in nats."""
        z, log_det = self.encode(x)
        return flow_prior_log_prob(self._prior, z).sum(dim=-1) + log_det

    def bits_per_dim(self, x: Tensor) -> Tensor:
        """Per-sample negative log-likelihood in bits per dimension."""
        return -self.log_prob(x) / (self._input_dim * math.log(2.0))

    @override
    def forward(self, x: Tensor) -> NormalizingFlowOutput:
        r"""Encode ``x`` and report the density along with it.

        Parameters
        ----------
        x : Tensor
            ``(B, C, H, W)`` data samples.

        Returns
        -------
        NormalizingFlowOutput
            ``latent`` :math:`z`, ``log_det_jacobian`` accumulated over the
            solve, and ``log_prob`` :math:`\log p_X(x)` in nats.  No
            ``loss`` — that is the task wrapper's to define.

        Notes
        -----
        Same computation as :meth:`log_prob`, returning the intermediates
        instead of discarding them.  With ``trace_method`` resolving to
        ``"hutchinson"`` the log-determinant is an unbiased *estimate*, so
        two calls on the same input will not agree.
        """
        z, log_det = self.encode(x)
        log_prob = flow_prior_log_prob(self._prior, z).sum(dim=-1) + log_det
        return NormalizingFlowOutput(
            latent=z,
            log_det_jacobian=log_det,
            log_prob=log_prob,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper — maximum-likelihood loss + sampling
# ─────────────────────────────────────────────────────────────────────────────


class NeuralODEForImageGeneration(ImageGenerationModel):
    r"""Neural ODE flow with a training loss and ``.generate()``.

    ``forward(x)`` returns a :class:`NormalizingFlowOutput` whose ``loss``
    is the mean negative log-likelihood in **bits per dimension**, the
    scale-free form that keeps the gradient magnitude independent of how
    wide the data is.  ``generate(n_samples)`` draws
    :math:`z \sim p_Z` and integrates the field backwards to ``t = 0``.

    Parameters
    ----------
    config : NeuralODEConfig
        Architecture, solver and trace-estimator knobs.

    Attributes
    ----------
    neural_ode : NeuralODEModel
        Underlying flow, exposing ``encode`` / ``decode`` / ``log_prob`` /
        ``bits_per_dim``.

    Notes
    -----
    Reference: Chen, Rubanova, Bettencourt, and Duvenaud, *"Neural
    Ordinary Differential Equations"*, NeurIPS, 2018 (arXiv:1806.07366).
    The paper's own continuous-flow experiments are two-dimensional
    density matching rather than images; sampling at image scale is what
    the Hutchinson trace of Grathwohl et al. (ICLR 2019) makes affordable,
    and is what ``trace_method`` selects by default above
    ``exact_trace_max_dim``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.neural_ode import (
    ...     NeuralODEConfig, NeuralODEForImageGeneration,
    ... )
    >>> cfg = NeuralODEConfig(sample_size=2, in_channels=1, out_channels=1,
    ...                       hidden_dim=8, num_blocks=1)
    >>> model = NeuralODEForImageGeneration(cfg).eval()
    >>> out = model(lucid.rand((2, 1, 2, 2)))
    >>> out.loss.shape                      # scalar bits/dim
    ()
    >>> model.generate(n_samples=3).samples.shape
    (3, 1, 2, 2)
    """

    config_class: ClassVar[type[NeuralODEConfig]] = NeuralODEConfig
    base_model_prefix: ClassVar[str] = "neural_ode"

    def __init__(self, config: NeuralODEConfig) -> None:
        super().__init__(config)
        self.neural_ode = NeuralODEModel(config)
        self._prior = config.prior
        self._input_dim = config.data_dim

    @property
    def nfe(self) -> int:
        """int: Vector-field evaluations spent by the most recent solve."""
        return self.neural_ode.nfe

    @override
    def forward(self, x: Tensor) -> NormalizingFlowOutput:
        r"""Run the flow and attach the training loss.

        Parameters
        ----------
        x : Tensor
            ``(B, C, H, W)`` data samples.

        Returns
        -------
        NormalizingFlowOutput
            The wrapped model's ``latent`` / ``log_det_jacobian`` /
            ``log_prob``, plus ``loss``: the mean negative log-likelihood
            in **bits per dimension**.

        Notes
        -----
        Bits/dim rather than nats so the gradient magnitude does not scale
        with the width of the data — the same quantity the likelihood
        literature reports.
        """
        out = self.neural_ode(x)
        bits = -out.log_prob / (self._input_dim * math.log(2.0))
        return NormalizingFlowOutput(
            latent=out.latent,
            log_det_jacobian=out.log_det_jacobian,
            log_prob=out.log_prob,
            loss=bits.mean(),
        )

    @lucid.no_grad()
    def generate(
        self,
        n_samples: int = 1,
        *,
        temperature: float = 1.0,
        device: str | None = None,
    ) -> GenerationOutput:
        """Sample by integrating a prior draw back to ``t = 0``.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to draw.
        temperature : float, default=1.0
            Multiplies the prior sample before inversion.  Below ``1`` it
            trades diversity for typicality — a standard trick for flows,
            not part of the paper.
        device : str, optional
            Where to allocate the prior sample.  Defaults to the device the
            model's parameters are on.

        Returns
        -------
        GenerationOutput
            ``samples`` of shape ``(n_samples, C, H, W)``.

        Notes
        -----
        Costs one solve, not one per step: the reverse direction carries no
        density term, so it is the cheaper of the two passes.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.neural_ode import (
        ...     NeuralODEConfig, NeuralODEForImageGeneration,
        ... )
        >>> cfg = NeuralODEConfig(sample_size=2, in_channels=1, out_channels=1,
        ...                       hidden_dim=8, num_blocks=1)
        >>> model = NeuralODEForImageGeneration(cfg).eval()
        >>> model.generate(n_samples=3).samples.shape
        (3, 1, 2, 2)
        >>> lucid.manual_seed(0)
        >>> cool = model.generate(n_samples=2, temperature=0.5).samples
        >>> lucid.manual_seed(0)
        >>> z = lucid.randn((2, 4))  # the same standard-normal prior draw
        >>> bool(lucid.allclose(cool, model.neural_ode.decode(z * 0.5)))
        True
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        z = start_noise(self, (n_samples, self._input_dim), device, prior=self._prior)
        return GenerationOutput(samples=self.neural_ode.decode(z * temperature))


__all__ = ["NeuralODEModel", "NeuralODEForImageGeneration"]
