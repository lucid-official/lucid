"""Explicit Runge-Kutta integration of ``dy/dt = f(t, y)``.

The public entry point :func:`odeint` lives here and dispatches on the
tableau: a tableau carrying an embedded error estimate goes to the adaptive
stepper in :mod:`lucid.diffeq._adaptive`, everything else to the fixed-grid
loop below.

The loop stays in Python on purpose.  The right-hand side is a Python callable
— typically a neural network — so driving the loop from C++ would force the
``ops`` layer to call back up into Python, inverting the engine's layer DAG.
What *does* live in C++ is the per-step arithmetic: every stage input and
every state update is the same affine form, fused into a single engine op.
That collapses roughly ten temporaries per RK4 step down to four.

The dominant cost of a solve is still the ``stages x steps`` right-hand-side
evaluations; the fusion trims the bandwidth-bound arithmetic around them, not
the network forwards themselves.
"""

import bisect
import math
import warnings
from dataclasses import dataclass, replace
from typing import Callable, Sequence, SupportsFloat, cast

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._tensor.tensor import Tensor
from lucid.diffeq import _adaptive, _event, _fixed, _flatten, _fused, _implicit, _multistep
from lucid.diffeq._timegrad import RhsProbe, TimeGrid
from lucid.diffeq._tableau import RK4, ButcherTableau, _DEFAULT_METHOD, _METHODS
from lucid.diffeq._typing import EventFunction, RightHandSide, ScalarFactory, StageCheck, State

# The fixed-grid loop and the tests both reach for this; it is the fused
# stage combination that every explicit method reduces to.
_combine = _fused.combine

__all__ = ["odeint", "odeint_dense", "odeint_event"]


def _resolve_method(method: str | ButcherTableau | None) -> ButcherTableau:
    """Turn a method name, tableau instance, or ``None`` into a tableau.

    Parameters
    ----------
    method : str or ButcherTableau or None
        A registered method name, a tableau to use verbatim, or ``None`` for
        the default method.

    Returns
    -------
    ButcherTableau
        The resolved tableau.

    Raises
    ------
    ValueError
        If ``method`` is a string that names no registered method.
    TypeError
        If ``method`` is neither a string nor a :class:`ButcherTableau`.
    """
    if method is None:
        method = _DEFAULT_METHOD
    if isinstance(method, ButcherTableau):
        return method
    if isinstance(method, str):
        tableau = _METHODS.get(method)
        if tableau is None:
            known = ", ".join(sorted({*_METHODS, *_multistep.METHODS}))
            raise ValueError(f"unknown method {method!r}; expected one of: {known}")
        return tableau
    raise TypeError(f"method must be a str or ButcherTableau, got {type(method).__name__}")


def _multistep_kind(method: str | ButcherTableau | None) -> bool | None:
    """Report whether ``method`` names an Adams method, and which kind.

    Parameters
    ----------
    method : str or ButcherTableau or None
        The caller's ``method`` argument.

    Returns
    -------
    bool or None
        ``True`` for a predictor-corrector Adams method, ``False`` for the
        explicit one, and ``None`` when this is an ordinary Runge-Kutta
        method that a tableau describes.
    """
    if isinstance(method, str):
        return _multistep.METHODS.get(method)
    return None


def _validate_method(method: str | ButcherTableau | None) -> None:
    """Raise unless ``method`` names a method of either family.

    Both the tableau methods and the Adams methods are legal names, so a
    caller that only wants to fail fast on a typo has to check against both.

    Raises
    ------
    ValueError
        If ``method`` is a string naming no registered method.
    TypeError
        If ``method`` is neither a string nor a :class:`ButcherTableau`.
    """
    if _multistep_kind(method) is None:
        _resolve_method(method)


def _reject_multistep(method: str | ButcherTableau | None, entry: str) -> None:
    """Refuse an Adams method where no interpolant backs it.

    Raises
    ------
    NotImplementedError
        If ``method`` names an Adams method.  These carry no dense output of
        their own, and silently substituting a cruder interpolant would make
        ``{entry}`` quietly less accurate than the same method under
        :func:`odeint`.
    """
    if _multistep_kind(method) is not None:
        raise NotImplementedError(
            f"{entry} does not support method={method!r}: Adams methods carry "
            f"no dense output. Use a Runge-Kutta method such as 'dopri5'."
        )


def _resolve_grid(t: Tensor | Sequence[float]) -> list[float]:
    """Materialise the integration grid as host floats, exactly once.

    Parameters
    ----------
    t : Tensor or sequence of float
        1-D grid of time points.

    Returns
    -------
    list of float
        The grid as plain Python floats.

    Raises
    ------
    ValueError
        If the grid is not 1-D, holds fewer than two points, contains a
        non-finite value, or is not strictly monotonic.

    Notes
    -----
    A ``Tensor`` grid is read back with a single ``tolist()``.  Reading the
    step size off the device once per step instead would stall the GPU
    queue on every step of every solve.
    """
    if isinstance(t, Tensor):
        if t.ndim != 1:
            raise ValueError(f"t must be a 1-D grid, got shape {t.shape}")
        # ``tolist`` is typed for every rank; the ndim guard above already
        # establishes that this one returns a flat list.
        grid = [float(v) for v in cast(list[SupportsFloat], t.tolist())]
    else:
        grid = [float(v) for v in t]

    if len(grid) < 2:
        raise ValueError(f"t must hold at least 2 time points, got {len(grid)}")

    # Checked before monotonicity: every comparison against NaN is False, so a
    # NaN would slip through the ordering test and only surface as an all-NaN
    # result many right-hand-side evaluations later.
    for value in grid:
        if not math.isfinite(value):
            raise ValueError(f"t must hold finite time points, found {value!r}")

    ascending = grid[1] > grid[0]
    for prev, cur in zip(grid, grid[1:], strict=False):  # adjacent pairs: grid[1:] is 1 shorter
        if (cur > prev) != ascending or cur == prev:
            raise ValueError(
                "t must be strictly monotonic (ascending or descending); "
                f"found {prev!r} followed by {cur!r}"
            )
    return grid


def _pack_state(
    func: Callable[..., object], y0: State, event_fn: Callable[..., Tensor] | None
) -> tuple[RightHandSide, Tensor, EventFunction | None, list[tuple[int, ...]] | None]:
    """Reduce a possibly-tuple state to the flat one the solvers integrate.

    Parameters
    ----------
    func : callable
        The caller's right-hand side, over whichever state form they used.
    y0 : Tensor or tuple of Tensor
        The caller's initial state.
    event_fn : callable or None
        The caller's event function, if any.

    Returns
    -------
    tuple
        ``(func, y0, event_fn, shapes)`` in flat form.  ``shapes`` is
        ``None`` when the caller passed a plain tensor, which is the signal
        to hand results back unchanged.
    """
    if isinstance(y0, Tensor):
        return (cast(RightHandSide, func), y0, cast(EventFunction | None, event_fn), None)

    parts = _flatten.check_state(y0)
    shapes = _flatten.shapes_of(parts)
    flat_event = None if event_fn is None else _flatten.wrap_event(event_fn, shapes)
    return (
        _flatten.wrap_rhs(
            cast(Callable[[Tensor, tuple[Tensor, ...]], Sequence[Tensor]], func), shapes
        ),
        _flatten.flatten(parts),
        flat_event,
        shapes,
    )


def _unpack(stacked: Tensor, shapes: list[tuple[int, ...]] | None) -> State:
    """Split a stacked flat result back into the caller's state form.

    Parameters
    ----------
    stacked : Tensor
        Shape ``(n, total)`` — one flat state per time point.
    shapes : list of tuple of int or None
        Component shapes, or ``None`` when the caller used a plain tensor.

    Returns
    -------
    Tensor or tuple of Tensor
        The input unchanged when ``shapes`` is ``None``, otherwise one
        tensor of shape ``(n, *shape)`` per component.
    """
    if shapes is None:
        return stacked
    return tuple(_flatten.unflatten_rows(stacked, shapes))


def _make_callbacks(y0: Tensor) -> tuple[ScalarFactory, StageCheck]:
    """Build the time-tensor factory and right-hand-side validator for a solve.

    Parameters
    ----------
    y0 : Tensor
        Initial state, which fixes the dtype, device and shape every stage
        must agree with.

    Returns
    -------
    tuple
        ``(scalar, check)`` — the first turns a host time into the 0-D tensor
        the right-hand side expects, the second validates what it returns.

    Notes
    -----
    Shared by every entry point so the contract is stated once.  The checks
    live here rather than being left to the engine so a mistake in ``func``
    is reported against ``func``, not as a shape or device error deep inside
    an internal fused op.

    Both callbacks run once per stage per step, which on a small state is a
    measurable share of the whole solve, so everything about ``y0`` that
    cannot change during it is read once here.  The comparisons are made on
    the engine's own dtype and device values rather than on the user-facing
    wrappers: reading ``Tensor.device`` builds a fresh ``device`` object, and
    doing that twice per stage to compare two of them is the sort of cost
    that only shows up in a profile.
    """
    base = _unwrap(y0)
    dtype_enum = base.dtype
    device_enum = base.device
    shape = y0.shape

    def scalar(value: float) -> Tensor:
        # A 0-d constant straight from the engine, rather than the general
        # `lucid.tensor` path, which re-resolves dtype and device and goes
        # looking for a buffer protocol on a Python float.
        return _wrap(_C_engine.full([], value, dtype_enum, device_enum))

    def check(result: object, step: int, stage: int) -> Tensor:
        if not isinstance(result, Tensor):
            raise TypeError(
                f"func must return a Tensor, got {type(result).__name__} "
                f"at step {step}, stage {stage}"
            )
        impl = _unwrap(result)
        if tuple(impl.shape) != shape:
            raise ValueError(
                f"func returned shape {result.shape} but y0 has shape {shape} "
                f"(step {step}, stage {stage})"
            )
        if impl.device != device_enum:
            raise ValueError(
                f"func returned a tensor on {result.device} but y0 is on "
                f"{y0.device} (step {step}, stage {stage})"
            )
        return result

    return scalar, check


def _warn_coarse_interpolant(tableau: ButcherTableau, what: str) -> None:
    """Warn when a method interpolates far below the order it steps at.

    Parameters
    ----------
    tableau : ButcherTableau
        The method about to be used.
    what : str
        What the interpolant is being asked for, named in the message.

    Notes
    -----
    Everything between steps — dense output, and the event time found inside
    a step — comes from a quartic anchored on the tableau's midpoint
    estimate.  Some tableaux carry a genuinely derived one and some carry a
    first-order placeholder, and the method's own order says nothing about
    which: ``bosh3`` takes third-order steps and interpolates to first.

    The warning states that fact and stops there, deliberately.  It is
    tempting to say the interpolated value will be inaccurate, but measurement
    does not support it — how much the low order costs depends on the problem
    and on how large the steps end up being.  On a free-fall event at default
    tolerances ``bosh3`` lands 2.9e-02 from the true time against ``dopri5``'s
    2.6e-13, eleven orders apart; on an exponential decay at ``rtol=1e-10``
    the steps are small enough that ``adaptive_heun`` — also first-order here
    — is the most accurate of all six at 1.0e-12, while ``dopri8`` is the
    worst at 1.9e-07.  So silence is not a promise either: ``dopri8`` steps at
    order eight and still interpolates with a quartic, which is its own
    documented ceiling.

    What the caller cannot otherwise see is the order itself, and that is what
    is worth saying.  Inherited from the reference tableaux, so this is a
    caution rather than a defect — the coefficients are transcribed correctly.
    """
    if tableau.mid is None or tableau.mid_order > 1:
        return
    warnings.warn(
        f"method {tableau.name!r} takes order-{tableau.order} steps but "
        f"interpolates between them at order 1, so {what} is bounded by the "
        f"latter rather than by rtol/atol. How much that costs depends on the "
        f"step sizes the solve settles on. Use 'dopri5' or 'tsit5' for a "
        f"derived interpolant.",
        RuntimeWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class _Problem:
    """One flat initial-value problem and everything its solvers share."""

    func: RightHandSide
    y0: Tensor
    grid: list[float]
    tableau: ButcherTableau
    scalar: ScalarFactory
    check: StageCheck
    rtol: float
    atol: float
    options: dict[str, object] | None

    def run_event(self, event_fn: EventFunction) -> tuple[Tensor, Tensor]:
        """Integrate to the first sign change of ``event_fn``.

        Returns
        -------
        tuple of Tensor
            ``(event_time, state_at_event)``, both differentiable.
        """
        direction = 1.0 if self.grid[-1] > self.grid[0] else -1.0
        event_t, y_event = _event.integrate_until_event(
            self.func,
            self.y0,
            self.grid[0],
            event_fn,
            self.tableau,
            self.scalar,
            self.check,
            rtol=self.rtol,
            atol=self.atol,
            options=self.options,
            direction=direction,
        )
        # Bisection compares signs, so the event time arrives with no graph
        # behind it.  It is still a differentiable function of everything the
        # trajectory depends on, and this ties it back to the state at the
        # event, which is the tensor that carries that graph.
        return _event.differentiable_event_time(self.scalar(event_t), y_event, self.func, event_fn)

    def run_grid(
        self, multistep: bool | None, times: TimeGrid | None, return_trajectory: bool
    ) -> tuple[Tensor, list[Tensor]]:
        """Integrate across the output grid with the family ``tableau`` selects.

        Parameters
        ----------
        multistep : bool or None
            The Adams variant, or ``None`` for a Runge-Kutta method.
        times : TimeGrid or None
            When given, a fixed-step solve that keeps its step sizes
            differentiable; the caller has checked that applies.
        return_trajectory : bool
            Whether to collect every output state.
        """
        common = (self.func, self.y0, self.grid)
        callbacks = (self.scalar, self.check)
        if multistep is not None:
            return _multistep.integrate(
                *common,
                multistep,
                *callbacks,
                rtol=self.rtol,
                atol=self.atol,
                options=self.options,
                return_trajectory=return_trajectory,
            )
        if self.tableau.is_implicit:
            return _implicit.integrate(
                *common,
                self.tableau,
                *callbacks,
                options=self.options,
                return_trajectory=return_trajectory,
            )
        if self.tableau.is_adaptive:
            return _adaptive.integrate(
                *common,
                self.tableau,
                *callbacks,
                rtol=self.rtol,
                atol=self.atol,
                options=self.options,
                return_trajectory=return_trajectory,
            )
        if times is not None:
            return _fixed.integrate_on_tensor_grid(
                self.func,
                self.y0,
                times,
                self.tableau,
                self.check,
                return_trajectory=return_trajectory,
            )
        return _fixed.integrate(
            *common,
            self.tableau,
            *callbacks,
            options=self.options,
            return_trajectory=return_trajectory,
        )


def _differentiable_steps(
    times: TimeGrid | None,
    event_fn: EventFunction | None,
    multistep: bool | None,
    tableau: ButcherTableau,
    options: dict[str, object] | None,
) -> bool:
    """Whether the solve can differentiate its own steps with respect to ``t``.

    True only for an explicit fixed-step method stepping on the caller's own
    grid, where the output times are the step boundaries.  Everything else
    gets the exact solution's sensitivity instead.
    """
    return (
        times is not None
        and event_fn is None
        and multistep is None
        and not tableau.is_implicit
        and not tableau.is_adaptive
        and _fixed.steps_on_caller_grid(options)
    )


def odeint(
    func: Callable[..., object],
    y0: State,
    t: Tensor | Sequence[float],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: str | ButcherTableau | None = None,
    options: dict[str, object] | None = None,
    event_fn: Callable[..., Tensor] | None = None,
    return_trajectory: bool = True,
) -> State | tuple[Tensor, State]:
    r"""Integrate ``dy/dt = f(t, y)``.

    Four families are reachable through the same call, and which one runs
    follows from ``method``:

    **Adaptive** (the default, and anything carrying an embedded error
    estimate) treats ``t`` as *output* times.  The solver picks its own step
    sizes to hold the local error inside ``rtol`` / ``atol`` and interpolates
    to each requested time, so a coarse ``t`` costs nothing in accuracy.

    **Fixed step** (``euler`` / ``midpoint`` / ``heun2`` / ``heun3`` / ``rk4``
    / ``rk4_classic``)
    treats ``t`` as the integration grid itself by default: consecutive
    entries are one step each, with no sub-stepping and no interpolation.
    ``rtol`` / ``atol`` are unused there, and accuracy is controlled by
    choosing a finer grid — or by handing ``options["step_size"]``, which
    decouples the two the same way an adaptive method does.

    **Adams multistep** (``explicit_adams`` / ``implicit_adams`` /
    ``fixed_adams``) is fixed-step too, but reaches high order by reusing the
    derivatives from previous steps instead of taking more of them inside one
    step — one new evaluation per step regardless of order.  That pays off
    when ``func`` is expensive.  See ``options["max_order"]`` for the caveat
    that comes with it.

    Gradient mode is left entirely to the caller.  Under grad the whole solve
    is differentiable end-to-end (discretise-then-optimise) and every stage is
    retained for backward; wrap the call in :func:`lucid.no_grad` for sampling,
    where the stage graph is pure overhead.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y) -> dy/dt``.  Receives the stage time as a
        0-D tensor matching ``y0`` in dtype and device, and must return a
        tensor with the same shape and device as ``y0``.
    y0 : Tensor
        Initial state at ``t[0]``.  Any shape; must have a floating dtype.
    t : Tensor or sequence of float
        Strictly monotonic 1-D grid of at least two finite time points.
        Descending grids integrate backwards in time.
    rtol : float, default=1e-7
        Relative tolerance.  Adaptive methods only.
    atol : float, default=1e-9
        Absolute tolerance.  Adaptive methods only.
    method : str or ButcherTableau or None, default=None
        ``None`` selects ``"dopri5"``.  Otherwise one of ``"dopri5"``,
        ``"dopri8"``, ``"tsit5"``, ``"bosh3"``, ``"fehlberg2"``,
        ``"adaptive_heun"``,
        ``"euler"``, ``"midpoint"``, ``"heun2"``, ``"heun3"``, ``"rk4"``,
        ``"rk4_classic"``,
        ``"explicit_adams"``, ``"implicit_adams"``, ``"fixed_adams"``,
        ``"implicit_euler"``, ``"implicit_midpoint"``, ``"trapezoid"``,
        ``"radauIIA3"``, ``"radauIIA5"``, ``"gl4"``, ``"gl6"``, ``"sdirk2"``,
        ``"trbdf2"``, or a custom :class:`ButcherTableau`.
    options : dict or None, default=None
        Per-method settings.  Adaptive methods accept ``min_step``,
        ``max_step``, ``first_step``, ``step_t``, ``jump_t``, ``safety``,
        ``ifactor``, ``dfactor``, ``max_num_steps`` (plus ``dtype`` and
        ``norm``, accepted and ignored).  Fixed-step methods accept
        ``step_size``, ``grid_constructor``, ``interp`` and ``perturb`` —
        giving either of the first two decouples the integration grid from
        ``t``, which is then reached by interpolation.  Adams methods accept
        all four of those plus ``max_order`` (default 12) and ``max_iters``
        (default 4, corrector sweeps, ignored by ``explicit_adams``).

        A high ``max_order`` is not simply more accurate.  Explicit Adams
        loses stability as the order climbs — at the default 12 its stable
        step is small enough that ``explicit_adams`` diverges on problems
        ``rk4`` handles comfortably — so lower it, or use one of the
        corrected variants, whose stability holds up far better.  Order also
        ramps from a Runge-Kutta start, which caps the accuracy the first
        steps can contribute regardless of ``max_order``.

        Implicit methods accept the fixed-step keys plus ``max_iters``
        (default 100), the ceiling on iterations of the nonlinear solve.  A
        step whose solve runs out of iterations warns rather than passing off
        an unconverged iterate as a completed step.
    event_fn : callable or None, default=None
        ``g(t, y)`` returning a single-element tensor.  When given, the solve
        ignores every entry of ``t`` but the first, runs until ``g`` changes
        sign, and returns a ``(event_t, solution)`` pair instead of a
        trajectory.  The direction of ``t`` still decides which way it
        searches.
    return_trajectory : bool, default=True
        Return the state at every time in ``t``.  Set ``False`` to keep only
        the final state — for a sampling run of many steps over a batch of
        images, stacking the full trajectory multiplies peak memory by the
        number of steps.  Ignored when ``event_fn`` is given.  **Lucid
        extension**, not part of the reference interface.

    Returns
    -------
    Tensor or tuple
        Without ``event_fn``: shape ``(len(t), *y0.shape)`` when
        ``return_trajectory`` is ``True`` (index 0 is ``y0`` itself),
        otherwise ``y0.shape``.  The dtype is the promotion of ``y0`` with
        everything ``func`` returns, matching what ``y + dt * k`` would
        produce.

        With ``event_fn``: a ``(event_t, solution)`` pair, where ``event_t``
        is a 0-D tensor and ``solution`` has shape ``(2, *y0.shape)`` — the
        state at ``t[0]`` and the state at the event.

    Raises
    ------
    ValueError
        If ``t`` is not a strictly monotonic 1-D grid of at least two finite
        points, if ``y0`` has a non-floating dtype, if ``method`` names no
        registered method, if ``options`` holds a key the method does not
        accept, or if ``func`` returns a tensor whose shape or device differs
        from ``y0``.
    TypeError
        If ``method`` is neither a string nor a :class:`ButcherTableau`, or
        if ``func`` returns something other than a tensor.
    RuntimeError
        If an adaptive solve exceeds ``max_num_steps`` or its step size
        collapses.

    Notes
    -----
    A fixed-step solve costs exactly ``(len(t) - 1) * method.stages`` calls to
    ``func``.  An adaptive solve costs as many as the tolerances demand, and
    reads one scalar back to the host per step to decide whether to accept it
    — that host synchronisation is intrinsic to adaptivity, which is why the
    two families have different performance characters.

    ``t`` is differentiable when it requires gradient.  A fixed-step solve on
    its default grid (``t`` *is* the step boundaries) returns the derivative
    of its own arithmetic, step sizes included, with ``func`` seeing time
    tensors that carry the graph.  Every other route -- adaptive, implicit and
    Adams methods, or a grid decoupled from ``t`` by ``step_size`` -- returns
    the exact solution's sensitivity: ``dL/dt_i = <dL/dy_i, f(t_i, y_i)>`` for
    an output time, and ``-<dL/dy_0 through the solve, f(t_0, y_0)>`` for the
    start.  The two agree as a fixed grid refines.  Each costs one extra
    evaluation of ``func`` per output time, and only when ``t`` asks for it.
    With ``event_fn`` only ``t[0]`` is read, so only ``t[0]`` is differentiated.

    Higher-order differentiation works: the fused step opts into
    graph-recording backward, so ``create_graph=True`` through a solve behaves
    exactly as it would for the unfused arithmetic.

    Examples
    --------
    Exponential decay against its closed form:

    >>> import lucid, lucid.diffeq as diffeq
    >>> y0 = lucid.tensor([1.0], dtype=lucid.float64)
    >>> y = diffeq.odeint(lambda s, y: -y, y0, [0.0, 1.0], return_trajectory=False)
    >>> abs(float(y.item()) - 0.36787944117) < 1e-8
    True

    See Also
    --------
    lucid.diffeq.ButcherTableau : Coefficient table selected by ``method``.
    """
    multistep = _multistep_kind(method)
    tableau = RK4 if multistep is not None else _resolve_method(method)
    grid = _resolve_grid(t)
    func, y0, event_fn, shapes = _pack_state(func, y0, event_fn)

    if not y0.is_floating_point():
        raise ValueError(f"y0 must have a floating dtype for integration, got {y0.dtype}")

    scalar, check = _make_callbacks(y0)

    # A time grid that asks for a gradient is kept in the graph.  A fixed-step
    # solve that steps on that grid differentiates its own arithmetic; every
    # other route attaches the exact solution's sensitivity to the times.
    times = TimeGrid.of(t, grid)
    probe = RhsProbe(func, scalar, check)
    problem = _Problem(func, y0, grid, tableau, scalar, check, rtol, atol, options)
    exact_steps = _differentiable_steps(times, event_fn, multistep, tableau, options)
    if times is not None and not exact_steps:
        problem = replace(problem, y0=times.shifted_start(y0, probe))

    if event_fn is not None:
        _reject_multistep(method, "event detection")
        _warn_coarse_interpolant(tableau, "the event time")
        event_time, y_event = problem.run_event(event_fn)
        return event_time, _unpack(_event.solution_pair(y0, y_event), shapes)

    y, trajectory = problem.run_grid(multistep, times if exact_steps else None, return_trajectory)
    if not return_trajectory:
        if times is not None and not exact_steps:
            y = times.tie_final(y, y0, probe)
        return y if shapes is None else tuple(_flatten.unflatten(y, shapes))
    # Promotion can lift the state above y0's dtype (a float64 RHS over a
    # float32 y0), which would leave trajectory[0] as the odd one out and
    # make the stack fail.  Settle the whole trajectory on the final dtype.
    trajectory = [s if s.dtype == y.dtype else s.to(y.dtype) for s in trajectory]
    stacked = lucid.stack(trajectory, dim=0)
    if times is not None and not exact_steps:
        stacked = times.tie_outputs(stacked, y0, probe)
    return _unpack(stacked, shapes)


def odeint_dense(
    func: Callable[..., object],
    y0: State,
    t0: float,
    t1: float,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: str | ButcherTableau | None = None,
    options: dict[str, object] | None = None,
) -> Callable[[float | Tensor], State]:
    r"""Solve once across an interval and return a continuous solution.

    Where :func:`odeint` wants the output times up front, this integrates
    ``[t0, t1]`` a single time and hands back a function of ``t``.  Every
    step's interpolating polynomial is kept, so any later query costs a
    binary search and one polynomial evaluation — no re-integration, and no
    commitment to a grid before you know which times you need.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y) -> dy/dt``.  Receives the stage time as a
        0-D tensor matching ``y0`` in dtype and device, and must return a
        tensor with the same shape and device as ``y0``.
    y0 : Tensor
        Initial state at ``t0``.  Any shape; must have a floating dtype.
    t0, t1 : float
        Ends of the interval.  ``t1 < t0`` integrates backwards.
    rtol : float, default=1e-7
        Relative tolerance.  Adaptive methods only.
    atol : float, default=1e-9
        Absolute tolerance.  Adaptive methods only.
    method : str or ButcherTableau or None, default=None
        ``None`` selects ``"dopri5"``.  See :func:`odeint` for the full list.
    options : dict or None, default=None
        Per-method settings, exactly as :func:`odeint` takes them.

    Returns
    -------
    callable
        ``dense(t) -> Tensor``, accepting a float or a 0-D tensor and
        returning the state there.  Raises :class:`ValueError` for a time
        outside ``[t0, t1]``.

    Raises
    ------
    ValueError
        If ``t0`` and ``t1`` are equal or non-finite, if ``y0`` has a
        non-floating dtype, if ``method`` names no registered method, if
        ``options`` holds a key the method does not accept, or if ``func``
        returns a tensor whose shape or device differs from ``y0``.
    TypeError
        If ``method`` is neither a string nor a :class:`ButcherTableau`, or
        if ``func`` returns something other than a tensor.
    RuntimeError
        If an adaptive solve exceeds ``max_num_steps`` or its step size
        collapses.

    Notes
    -----
    Memory grows with the number of accepted steps: each keeps a handful of
    tensors the size of the state.  A long solve over a large state is
    therefore much heavier than :func:`odeint` with ``return_trajectory=False``.

    The returned function accepts a query-time tensor that requires gradient,
    and ``d state / d t`` is then ``f(t, state)``.

    A fixed-step method interpolates with whatever ``options["interp"]``
    says, and without ``step_size`` its grid is the single interval
    ``[t0, t1]`` — usually far too coarse, so pass one.

    Examples
    --------
    >>> import lucid, lucid.diffeq as diffeq
    >>> y0 = lucid.tensor([1.0], dtype=lucid.float64)
    >>> dense = diffeq.odeint_dense(lambda t, y: -y, y0, 0.0, 1.0)
    >>> abs(float(dense(0.5).item()) - 0.6065306597) < 1e-7
    True

    See Also
    --------
    lucid.diffeq.odeint : Same solvers, output times fixed up front.
    """
    if not math.isfinite(t0) or not math.isfinite(t1):
        raise ValueError(f"t0 and t1 must be finite, got {t0!r} and {t1!r}")
    if t0 == t1:
        raise ValueError(f"t0 and t1 must differ, both are {t0!r}")

    _reject_multistep(method, "odeint_dense")
    tableau = _resolve_method(method)
    _warn_coarse_interpolant(tableau, "the continuous solution")
    func, y0, _unused, shapes = _pack_state(func, y0, None)
    if not y0.is_floating_point():
        raise ValueError(f"y0 must have a floating dtype for integration, got {y0.dtype}")

    scalar, check = _make_callbacks(y0)

    if tableau.is_implicit:
        segments = _implicit.integrate_dense(
            func, y0, t0, t1, tableau, scalar, check, options=options
        )
    elif tableau.is_adaptive:
        segments = _adaptive.integrate_dense(
            func, y0, t0, t1, tableau, scalar, check, rtol=rtol, atol=atol, options=options
        )
    else:
        segments = _fixed.integrate_dense(func, y0, t0, t1, tableau, scalar, check, options=options)

    # Index by ascending interval start so a query is a binary search
    # regardless of which way the solve ran.
    ordered = segments if t1 > t0 else list(reversed(segments))
    starts = [min(a, b) for a, b, _ in ordered]
    low, high = min(t0, t1), max(t0, t1)

    probe = RhsProbe(func, scalar, check)

    def dense(t: float | Tensor) -> State:
        value = float(t.item()) if isinstance(t, Tensor) else float(t)
        if not math.isfinite(value):
            raise ValueError(f"query time must be finite, got {value!r}")
        if value < low or value > high:
            raise ValueError(
                f"query time {value!r} lies outside the solved interval " f"[{low!r}, {high!r}]"
            )
        index = bisect.bisect_right(starts, value) - 1
        index = min(max(index, 0), len(ordered) - 1)
        seg_t0, seg_t1, coeffs = ordered[index]
        state = _fused.poly_eval(coeffs, seg_t0, seg_t1, value)
        if isinstance(t, Tensor) and t.requires_grad and lucid.is_grad_enabled():
            state = probe.tie(value, state, t.to(dtype=state.dtype, device=state.device))
        return state if shapes is None else tuple(_flatten.unflatten(state, shapes))

    return dense


def odeint_event(
    func: Callable[..., object],
    y0: State,
    t0: float | Tensor,
    *,
    event_fn: Callable[..., Tensor],
    reverse_time: bool = False,
    odeint_interface: Callable[..., object] = odeint,
    **kwargs: object,
) -> tuple[Tensor, State]:
    r"""Integrate from ``t0`` until an event fires.

    The convenience form of ``odeint(..., event_fn=...)`` for the common case
    where there is no output grid at all — you have a starting time and a
    condition, and want to know when the condition is met.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y) -> dy/dt``.
    y0 : Tensor
        Initial state at ``t0``.
    t0 : float or Tensor
        Start time.
    event_fn : callable
        ``g(t, y)`` returning a single-element tensor.  The solve ends the
        moment its sign changes; a value of exactly zero at ``t0`` fires
        immediately.
    reverse_time : bool, default=False
        Search backwards in time instead of forwards.
    odeint_interface : callable, default=:func:`odeint`
        Solver to delegate to.  Pass :func:`odeint_adjoint` to get the event
        solve at constant memory.
    **kwargs
        Forwarded to ``odeint_interface`` — ``rtol``, ``atol``, ``method``,
        ``options``.

    Returns
    -------
    tuple
        ``(event_t, solution)``; ``event_t`` is a 0-D tensor and
        ``solution`` has shape ``(2, *y0.shape)`` — the state at ``t0`` and
        the state at the event.

    Raises
    ------
    ValueError
        If a fixed-step method is used without ``options["step_size"]``, or
        if ``event_fn`` does not return a single-element tensor.
    RuntimeError
        If no sign change is found within the step budget.

    Notes
    -----
    The event time itself is located by bisecting the interpolant of the
    step that brackets it, so it costs event-function calls but no extra
    right-hand-side evaluations.

    ``t0`` may be a tensor that requires gradient; the event time and the
    state at it then move with the start time (``d event_t / d t0 = 1`` for an
    autonomous system).

    ``event_t`` is differentiable, though bisection itself is not: the event
    time is pinned by ``g(t*, y(t*)) = 0``, and differentiating that identity
    routes its gradient onto the state at the event, which does carry a graph.
    Expect total derivatives -- the state at the event moves with the event
    time, so differentiating ``solution[-1]`` accounts for that too.

    Examples
    --------
    A body falling from rest hits the ground at :math:`\sqrt{2h/g}`:

    >>> import lucid, lucid.diffeq as diffeq
    >>> y0 = lucid.tensor([10.0, 0.0], dtype=lucid.float64)  # height, velocity
    >>> def fall(t, y):
    ...     return lucid.stack([y[1], lucid.tensor(-9.8, dtype=y.dtype)], dim=0)
    >>> event_t, sol = diffeq.odeint_event(
    ...     fall, y0, 0.0, event_fn=lambda t, y: y[0]
    ... )
    >>> abs(float(event_t.item()) - (2 * 10.0 / 9.8) ** 0.5) < 1e-6
    True

    See Also
    --------
    lucid.diffeq.odeint : Takes ``event_fn`` directly alongside an output grid.
    """
    start = float(t0.item()) if isinstance(t0, Tensor) else float(t0)
    # Only the first entry is read; the second exists to state the search
    # direction, which is the one thing the grid still carries.
    offset = -1.0 if reverse_time else 1.0
    grid: Tensor | list[float] = [start, start + offset]
    if isinstance(t0, Tensor) and t0.requires_grad:
        # A start time that asks for a gradient has to stay a tensor on its
        # way into ``odeint``, which is where it is tied to the trajectory.
        first = t0.reshape(1)
        grid = lucid.cat([first, first + offset], dim=0)
    result = odeint_interface(func, y0, grid, event_fn=event_fn, **kwargs)
    return cast(tuple[Tensor, State], result)
