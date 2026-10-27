"""Neural ODE configuration — Chen, Rubanova, Bettencourt & Duvenaud, 2018.

The generative half of the paper: a continuous normalizing flow.  Where NICE
and RealNVP stack a finite number of hand-designed bijections, this one
defines a single vector field and lets an ODE solver decide how many steps
that composition needs.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    FlowPrior,
    NormalizingFlowConfig,
    check_sample_space_channels,
)

# How the divergence of the vector field is obtained each step.
#
# ``"exact"`` sums the diagonal of the Jacobian, which costs one
# vector-Jacobian product per dimension — what the paper uses, and only
# affordable at the two dimensions it experiments with.  ``"hutchinson"``
# spends one product on an unbiased estimate instead, which is what makes
# the same flow reach image scale.
TraceMethod = Literal["exact", "hutchinson"]

# Distribution the Hutchinson probe vector is drawn from.  Both give an
# unbiased estimate; Rademacher has the lower variance of the two, and is
# the usual default.
TraceNoise = Literal["rademacher", "gaussian"]

#: Which vector field to build.  ``"planar"`` is the paper's eq. (9)+(10)
#: — a single layer of gated planar units — and is what the authors' own
#: ``examples/cnf.py`` builds.  ``"concat_squash"`` is the deeper MLP the
#: FFJORD follow-up popularised; it is a strictly larger family, but it is
#: not the model this paper describes.
VectorFieldKind = Literal["planar", "concat_squash"]


@model_family_meta(
    canonical_name="Neural ODE",
    citation=(
        "Chen, Ricky T. Q., et al. "
        '"Neural Ordinary Differential Equations." Advances in Neural '
        "Information Processing Systems, vol. 31, 2018, pp. 6571–6583."
    ),
    theory=r"""
    A residual network updates its state in discrete jumps,
    :math:`h_{t+1} = h_t + f(h_t, \theta_t)`.  Shrink the jump and add a
    step size and that recurrence is Euler's method; take the step to zero
    and the network stops being a sequence of layers and becomes a
    *differential equation*,

    .. math::

        \frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t, \theta),

    whose solution at :math:`t_1` is whatever an ODE solver says it is.
    Depth is no longer a hyper-parameter but an accuracy request handed to
    the solver, and the memory the backward pass needs stops growing with
    it — the adjoint method reconstructs what it needs by integrating
    backwards rather than storing every intermediate state.

    What that buys a generative model is **Theorem 1**, the instantaneous
    change of variables.  A discrete flow pays for every bijection with the
    log-determinant of its Jacobian, which is why NICE and RealNVP go to
    such lengths to keep that Jacobian triangular.  A continuous flow pays
    with a trace instead:

    .. math::

        \frac{\partial \log p(\mathbf{z}(t))}{\partial t}
            = -\operatorname{tr}\!\left(
                \frac{\partial f}{\partial \mathbf{z}(t)}\right).

    A determinant constrains the *architecture*; a trace does not.  The
    vector field can be any network at all — no masking, no ordering of the
    dimensions, no triangular structure — and the density still comes out
    exactly, by integrating the state and its log-density together as one
    augmented system.

    The paper's vector field is a sum of :math:`M` functions with learned
    time-dependent gates, :math:`d\mathbf{z}/dt = \sum_n \sigma_n(t)
    f_n(\mathbf{z})` with :math:`\sigma_n(t) \in (0, 1)` — a hypernetwork,
    so that the dynamics themselves change as the flow proceeds rather than
    being one fixed map applied for longer.

    Computing the trace exactly costs one vector-Jacobian product per
    dimension, which is why the paper's experiments are two-dimensional
    density matching.  Replacing it with the Hutchinson estimator,

    .. math::

        \operatorname{tr}(A) = \mathbb{E}_{p(\boldsymbol{\epsilon})}
            \left[\boldsymbol{\epsilon}^{\mathsf{T}} A
                  \boldsymbol{\epsilon}\right],
        \qquad
        \mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0},\;
        \operatorname{Cov}(\boldsymbol{\epsilon}) = I,

    turns that per-dimension cost into a single product for an unbiased
    estimate, and is what lets the same flow run on images — the step
    Grathwohl et al. take in FFJORD (*"FFJORD: Free-Form Continuous
    Dynamics for Scalable Reversible Generative Models"*, ICLR 2019).  Both
    are available here; ``trace_method`` chooses, and the default follows
    the dimension.
    """,
)
@dataclass(frozen=True)
class NeuralODEConfig(NormalizingFlowConfig):
    r"""Frozen configuration for every Neural ODE (CNF) variant.

    Parameters
    ----------
    sample_size : int or tuple of int, default=32
        Spatial resolution of a sample.  The flow itself is defined on the
        flattened vector, so this and ``in_channels`` only fix its width.
    in_channels : int, default=3
        Channels per sample.
    out_channels : int, default=3
        Kept equal to ``in_channels``: the flow is a bijection, so it
        cannot change the dimension it acts on.
    act_fn : {"silu", "swish", "relu", "gelu"}, default="silu"
        Activation between the vector field's gated blocks.  A smooth one
        is worth preferring here for a reason that does not apply to a
        discrete flow: the solver estimates its own error from how
        polynomial the trajectory looks locally, so a kinked right-hand
        side costs steps.
    prior : {"logistic", "gaussian"}, default="gaussian"
        Base distribution the latent is measured against.  Unlike the
        discrete flows, which use a logistic prior on dequantised pixels,
        the continuous flow integrates to a standard normal.
    hidden_dim : int, default=64
        Width :math:`M` of each function in the gated sum — the paper's CNF
        experiments use ``64``.
    num_blocks : int, default=2
        How many gated blocks the vector field stacks.
    solver : str, default="dopri5"
        Method name handed to :func:`lucid.diffeq.odeint`.  Any adaptive
        method works; the fixed-step ones trade accuracy for a predictable
        cost per sample.
    rtol, atol : float, default=1e-5, 1e-5
        Solver tolerances.  Looser than the ``lucid.diffeq`` defaults on
        purpose: the flow is trained, so solving it to twelve digits buys
        nothing the gradient can use.
    use_adjoint : bool, default=True
        Integrate the backward pass with :func:`lucid.diffeq.odeint_adjoint`
        rather than differentiating through the solver's own steps.  This is
        the paper's constant-memory result; turning it off costs memory
        proportional to the number of steps taken but gives gradients that
        are exact for the discretisation actually used.
    trace_method : {"exact", "hutchinson"} or None, default=None
        How the divergence is obtained.  ``None`` picks by dimension —
        exact below ``exact_trace_max_dim``, Hutchinson above it — so the
        default is the paper's method exactly where the paper's method is
        affordable.
    trace_noise : {"rademacher", "gaussian"}, default="rademacher"
        Probe distribution for the Hutchinson estimate.
    exact_trace_max_dim : int, default=32
        Dimension below which ``trace_method=None`` resolves to ``"exact"``.

    Notes
    -----
    ``in_channels`` and ``out_channels`` must agree: a flow is a bijection.

    Examples
    --------
    >>> from lucid.models.generative.neural_ode import NeuralODEConfig
    >>> NeuralODEConfig().hidden_dim
    64
    >>> NeuralODEConfig(sample_size=(1, 2), in_channels=1,
    ...                 out_channels=1).data_dim
    2
    """

    model_type: ClassVar[str] = "neural_ode"

    prior: FlowPrior = "gaussian"

    field: VectorFieldKind = "planar"
    hidden_dim: int = 64
    num_blocks: int = 2

    solver: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-5
    use_adjoint: bool = True

    trace_method: TraceMethod | None = None
    trace_noise: TraceNoise = "rademacher"
    exact_trace_max_dim: int = 32

    @property
    def data_dim(self) -> int:
        """int: Flattened width of one sample, the dimension the flow acts on."""
        size = self.sample_size
        height, width = size if isinstance(size, tuple) else (size, size)
        return self.in_channels * height * width

    @property
    def resolved_trace_method(self) -> TraceMethod:
        """TraceMethod: ``trace_method``, or the one implied by the dimension.

        Exact where exact is affordable and Hutchinson where it is not, so
        that the default reproduces the paper at the paper's scale without
        making the same choice unusable at image scale.
        """
        if self.trace_method is not None:
            return self.trace_method
        return "exact" if self.data_dim <= self.exact_trace_max_dim else "hutchinson"

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        check_sample_space_channels(
            self.in_channels, self.out_channels, "a flow is a bijection: its image"
        )
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError(
                f"solver tolerances must be positive, got rtol={self.rtol}, " f"atol={self.atol}"
            )
        if self.exact_trace_max_dim < 1:
            raise ValueError(
                f"exact_trace_max_dim must be at least 1, got " f"{self.exact_trace_max_dim}"
            )


__all__ = ["NeuralODEConfig", "TraceMethod", "TraceNoise"]
