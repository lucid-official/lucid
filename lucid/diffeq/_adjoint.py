"""Backpropagation through an ODE solve at constant memory.

Differentiating a solve the direct way keeps every stage of every step alive
for the backward pass, so memory grows with the number of steps — and an
adaptive solver decides that number at run time.  The adjoint method instead
throws the forward graph away and recovers the gradient by integrating a
second ODE backwards in time:

.. math::

    \\frac{da}{dt} = -a^{\\top} \\frac{\\partial f}{\\partial y},
    \\qquad
    \\frac{dg}{dt} = -a^{\\top} \\frac{\\partial f}{\\partial \\theta}

where :math:`a = \\partial L / \\partial y` is the adjoint and :math:`g`
accumulates the parameter gradient.  Memory is then set by the size of the
state and the parameters, not by how long the solve ran.

The price is accuracy and time: the backward solve is a numerical
approximation of the true gradient rather than the exact derivative of what
the forward pass computed, and it costs a second integration with a
vector-Jacobian product at every stage.
"""

from dataclasses import dataclass
from typing import Callable, Sequence, cast, final, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx
from lucid.diffeq import _flatten
from lucid.diffeq._solvers import (
    _make_callbacks,
    _pack_state,
    _resolve_grid,
    _unpack,
    _validate_method,
    odeint,
)
from lucid.diffeq._tableau import ButcherTableau
from lucid.diffeq._timegrad import RhsProbe, TimeGrid
from lucid.diffeq._typing import RightHandSide, State

__all__ = ["odeint_adjoint"]


@dataclass(frozen=True)
class _Config:
    """Everything the backward pass needs that is not a tensor."""

    func: RightHandSide
    grid: list[float]
    rtol: float
    atol: float
    method: str | ButcherTableau | None
    options: dict[str, object] | None
    adjoint_rtol: float
    adjoint_atol: float
    adjoint_method: str | ButcherTableau | None
    adjoint_options: dict[str, object] | None
    n_params: int


def _resolve_params(
    func: Callable[..., object], adjoint_params: Sequence[Tensor] | None
) -> tuple[Tensor, ...]:
    """Decide which tensors the adjoint should accumulate gradients for.

    Parameters
    ----------
    func : callable
        The right-hand side.  When it exposes ``parameters()`` — as a module
        does — that is used as the default set.
    adjoint_params : sequence of Tensor or None
        Explicit override.

    Returns
    -------
    tuple of Tensor
        The parameters to differentiate with respect to.

    Raises
    ------
    TypeError
        If any supplied parameter is not a tensor.

    Notes
    -----
    ``parameters()`` is found by duck-typing rather than by checking for a
    module type: ``lucid.diffeq`` must not import ``lucid.nn``, since the
    layer rules put the solver below it.
    """
    if adjoint_params is not None:
        params = tuple(adjoint_params)
        for index, p in enumerate(params):
            if not isinstance(p, Tensor):
                raise TypeError(
                    f"adjoint_params[{index}] must be a Tensor, " f"got {type(p).__name__}"
                )
        return params

    getter = getattr(func, "parameters", None)
    if callable(getter):
        return tuple(p for p in getter() if isinstance(p, Tensor))
    return ()


def _augmented_dynamics(
    cfg: _Config, params: tuple[Tensor, ...], shapes: list[tuple[int, ...]]
) -> RightHandSide:
    """Build the right-hand side of the augmented backward system.

    Parameters
    ----------
    cfg : _Config
        The solve's configuration; supplies the forward right-hand side.
    params : tuple of Tensor
        Parameters being differentiated.
    shapes : list of tuple of int
        Shapes of the flattened augmented state ``[y, a, *param_grads]``.

    Returns
    -------
    callable
        ``g(t, aug) -> d(aug)/dt`` operating on the flat augmented state.

    Notes
    -----
    One vector-Jacobian product per call gives both ``-a^T df/dy`` and
    ``-a^T df/dtheta`` at once, so the adjoint costs one backward through
    ``func`` per stage rather than a Jacobian.
    """

    def dynamics(t: Tensor, aug: Tensor) -> Tensor:
        y, adjoint = _flatten.unflatten(aug, shapes)[:2]

        with lucid.enable_grad():
            # Fresh leaves: the augmented state arrives detached from the
            # backward solve's own graph, and the VJP below must attach to
            # these, not to whatever produced them.
            y_leaf = y.detach().requires_grad_(True)
            f = cfg.func(t.detach(), y_leaf)
            vjps = lucid.autograd.grad(
                f, [y_leaf, *params], grad_outputs=[-adjoint.detach()], allow_unused=True
            )

        d_adjoint = vjps[0] if vjps[0] is not None else lucid.zeros_like(y)
        d_params = [
            v if v is not None else lucid.zeros_like(p)
            for v, p in zip(vjps[1:], params, strict=True)
        ]
        return _flatten.flatten([f.detach(), d_adjoint, *d_params])

    return dynamics


@final
class _AdjointSolve(Function):
    """Solve forwards without a graph, and recover gradients backwards."""

    _once_differentiable = True  # second derivatives through it are unverified

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, y0: Tensor, *params: Tensor, config: _Config) -> Tensor:
        """Integrate the trajectory with gradient tracking switched off."""
        ctx.config = config
        with lucid.no_grad():
            # ``odeint`` only returns a pair when given an ``event_fn``, which
            # the adjoint never does.
            ys = cast(
                Tensor,
                odeint(
                    config.func,
                    y0,
                    config.grid,
                    rtol=config.rtol,
                    atol=config.atol,
                    method=config.method,
                    options=config.options,
                ),
            )
        # The trajectory is saved detached — holding the returned tensor
        # itself would make the context and the graph node reference each
        # other.  The parameters, though, must be the originals: the
        # vector-Jacobian product needs the tensors that actually appear in
        # ``func``'s graph, and a detached stand-in would both come back
        # unused and leave the real parameter outside the ``inputs`` set.
        ctx.save_for_backward(ys.detach())
        ctx.params = params
        return ys

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_ys: Tensor) -> tuple[Tensor, ...]:
        """Integrate the augmented system backwards, interval by interval."""
        # ``FunctionCtx`` stores user attributes as ``object``; these two were
        # written by ``forward`` just above.
        cfg = cast(_Config, ctx.config)
        (ys,) = ctx.saved_tensors
        params = cast(tuple[Tensor, ...], ctx.params)
        grid = cfg.grid

        y_shape = tuple(ys.shape[1:])
        shapes = [y_shape, y_shape, *(tuple(p.shape) for p in params)]
        dynamics = _augmented_dynamics(cfg, params, shapes)

        adjoint = lucid.zeros_like(ys[0])
        grads = [lucid.zeros_like(p) for p in params]

        # Walk the output times from the end.  The state is re-anchored on the
        # stored forward value at each one instead of being carried backwards
        # through the whole span, which keeps the reverse solve from drifting
        # away from the trajectory the gradient belongs to.
        for index in range(len(grid) - 1, 0, -1):
            adjoint = adjoint + grad_ys[index]
            aug0 = _flatten.flatten([ys[index], adjoint, *grads])
            aug1 = cast(
                Tensor,
                odeint(
                    dynamics,
                    aug0,
                    [grid[index], grid[index - 1]],
                    rtol=cfg.adjoint_rtol,
                    atol=cfg.adjoint_atol,
                    method=cfg.adjoint_method,
                    options=cfg.adjoint_options,
                    return_trajectory=False,
                ),
            )
            pieces = _flatten.unflatten(aug1, shapes)
            adjoint = pieces[1]
            grads = pieces[2:]

        adjoint = adjoint + grad_ys[0]
        return (adjoint, *grads)


def _resolve_adjoint_norm(
    adjoint_options: dict[str, object] | None, state_elements: int
) -> dict[str, object] | None:
    """Translate ``norm="seminorm"`` into the private prefix the stepper reads.

    Parameters
    ----------
    adjoint_options : dict or None
        Options destined for the backward solve.
    state_elements : int
        Element count of ``y`` alone, so that ``y`` and the adjoint together
        occupy the first ``2 * state_elements`` of the augmented state.

    Returns
    -------
    dict or None
        The options with ``norm`` replaced by the internal prefix, or the
        input unchanged when no ``norm`` was given.

    Raises
    ------
    NotImplementedError
        For any other ``norm``, including a callable.

    Notes
    -----
    The augmented backward state is ``[y | adjoint | parameter gradients]``.
    The default norm judges the step on all of it; a seminorm drops the
    parameter block.  That block is a quadrature — it accumulates
    ``dL/dtheta`` but never feeds back into the dynamics — so holding it to
    the same tolerance buys no accuracy in the trajectory and costs steps.

    A caller-supplied norm cannot be honoured at all: the error norm is a
    fused engine kernel, and replacing it with a Python callable would give
    up the fusion and the single host synchronisation per step that goes with
    it.  Refusing loudly beats accepting and ignoring, which is what this
    option used to do.
    """
    if adjoint_options is None or "norm" not in adjoint_options:
        return adjoint_options

    norm = adjoint_options["norm"]
    if norm != "seminorm":
        raise NotImplementedError(
            f"adjoint_options['norm'] = {norm!r} is not supported; only "
            f"'seminorm' is. The error norm is a fused engine kernel, so a "
            f"caller-supplied norm cannot be honoured."
        )

    resolved = {k: v for k, v in adjoint_options.items() if k != "norm"}
    resolved["_norm_prefix"] = 2 * state_elements
    return resolved


def _apply(y0: Tensor, params: tuple[Tensor, ...], config: _Config) -> Tensor:
    """Apply the adjoint solve; ``apply`` is typed for tuples, this op returns one."""
    return cast(Tensor, _AdjointSolve.apply(y0, *params, config=config))


def _solve(
    t: Tensor | Sequence[float], y0: Tensor, params: tuple[Tensor, ...], config: _Config
) -> Tensor:
    """Run the adjoint solve, tying ``t`` into the graph when it asks for a gradient.

    Parameters
    ----------
    t : Tensor or sequence of float
        The caller's grid.
    y0 : Tensor
        Flat initial state.
    params : tuple of Tensor
        Parameters being differentiated.
    config : _Config
        The solve's configuration.

    Returns
    -------
    Tensor
        The trajectory, shape ``(len(t), *y0.shape)``.

    Notes
    -----
    A grid that asks for a gradient reaches the adjoint solve as a shifted
    start state and as edges on the output rows -- see ``_timegrad``.  The
    backward pass itself never sees ``t``.
    """
    times = TimeGrid.of(t, config.grid)
    if times is None:
        return _apply(y0, params, config)
    probe = RhsProbe(config.func, *_make_callbacks(y0))
    ys = _apply(times.shifted_start(y0, probe), params, config)
    return times.tie_outputs(ys, y0, probe)


def odeint_adjoint(
    func: Callable[..., object],
    y0: State,
    t: Tensor | Sequence[float],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: str | ButcherTableau | None = None,
    options: dict[str, object] | None = None,
    event_fn: Callable[..., Tensor] | None = None,
    adjoint_rtol: float | None = None,
    adjoint_atol: float | None = None,
    adjoint_method: str | ButcherTableau | None = None,
    adjoint_options: dict[str, object] | None = None,
    adjoint_params: Sequence[Tensor] | None = None,
) -> State:
    r"""Integrate ``dy/dt = f(t, y)`` and differentiate it at constant memory.

    Same result as :func:`odeint`, different gradient strategy.  ``odeint``
    differentiates the discretisation — exact for what it computed, but every
    stage of every step stays alive until backward.  This instead discards the
    forward graph and reconstructs the gradient by integrating the adjoint
    system backwards, so memory no longer grows with the number of steps.

    Use it when the solve is long or the tolerance tight enough that the
    retained stages dominate memory.  For a short solve, :func:`odeint` is
    both faster and more accurate.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y) -> dy/dt``.
    y0 : Tensor
        Initial state at ``t[0]``.
    t : Tensor or sequence of float
        Strictly monotonic 1-D grid of at least two finite time points.
    rtol, atol : float
        Tolerances for the forward solve.
    method : str or ButcherTableau or None, default=None
        Forward method; ``None`` selects ``"dopri5"``.
    options : dict or None, default=None
        Forward solver options.
    event_fn : callable or None, default=None
        Not implemented yet; passing one raises.
    adjoint_rtol, adjoint_atol : float or None, default=None
        Tolerances for the backward solve.  ``None`` inherits the forward
        values.
    adjoint_method : str or ButcherTableau or None, default=None
        Method for the backward solve.  ``None`` inherits ``method``.
    adjoint_options : dict or None, default=None
        Options for the backward solve.  ``None`` inherits ``options``.
        ``{"norm": "seminorm"}`` drops the parameter-gradient block from the
        backward solve's error norm, which is the one norm setting that has an
        effect here; any other value, including a callable, is refused.
    adjoint_params : sequence of Tensor or None, default=None
        Tensors to accumulate gradients for.  ``None`` uses
        ``func.parameters()`` when ``func`` exposes it, otherwise nothing.

    Returns
    -------
    Tensor
        Shape ``(len(t), *y0.shape)``; index 0 is ``y0`` itself.

    Raises
    ------
    NotImplementedError
        If ``event_fn`` is given.
    TypeError
        If any entry of ``adjoint_params`` is not a tensor.
    ValueError
        Anything :func:`odeint` rejects, on either solve.

    Notes
    -----
    The gradient is an approximation.  It solves the continuous adjoint
    equation numerically, so it converges to the true derivative as the
    tolerances tighten rather than matching :func:`odeint`'s gradient
    exactly — tightening ``adjoint_rtol`` / ``adjoint_atol`` is what closes
    the gap.

    Cost is a second integration whose right-hand side performs one
    vector-Jacobian product through ``func``, so expect roughly a doubling
    of solve time on top of the backward passes.

    A ``t`` that requires gradient receives one: ``dL/dt_i = <dL/dy_i,
    f(t_i, y_i)>`` at an output time and ``-<a(t_0), f(t_0, y_0)>`` at the
    start, with ``a`` the adjoint -- the exact solution's sensitivity, at one
    extra ``func`` call per output time.

    Examples
    --------
    >>> import lucid, lucid.diffeq as diffeq
    >>> k = lucid.tensor([0.5], dtype=lucid.float64, requires_grad=True)
    >>> y0 = lucid.tensor([1.0], dtype=lucid.float64)
    >>> ys = diffeq.odeint_adjoint(
    ...     lambda t, y: -k * y, y0, [0.0, 1.0], adjoint_params=[k]
    ... )
    >>> ys[-1].sum().backward()
    >>> bool(abs(float(k.grad.item()) + 0.60653066) < 1e-5)
    True

    See Also
    --------
    lucid.diffeq.odeint : Direct differentiation; exact but memory-hungry.
    """
    if event_fn is not None:
        raise NotImplementedError(
            "odeint_adjoint does not support event_fn yet; use odeint for "
            "event-free solves or track the gap in the diffeq roadmap"
        )

    grid = _resolve_grid(t)
    _validate_method(method)  # fail fast on a bad name, before any integration
    params = _resolve_params(func, adjoint_params)
    flat_func, flat_y0, _unused, shapes = _pack_state(func, y0, None)

    config = _Config(
        func=flat_func,
        grid=grid,
        rtol=rtol,
        atol=atol,
        method=method,
        options=options,
        adjoint_rtol=rtol if adjoint_rtol is None else adjoint_rtol,
        adjoint_atol=atol if adjoint_atol is None else adjoint_atol,
        adjoint_method=method if adjoint_method is None else adjoint_method,
        adjoint_options=_resolve_adjoint_norm(
            options if adjoint_options is None else adjoint_options, int(flat_y0.shape[0])
        ),
        n_params=len(params),
    )
    ys = _solve(t, flat_y0, params, config)
    return _unpack(ys, shapes)
