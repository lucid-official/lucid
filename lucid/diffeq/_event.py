"""Integrating until a condition fires rather than to a fixed time.

Some problems do not end at a time you can name in advance — a trajectory ends
when it hits the ground, a simulation stops when a quantity crosses a
threshold.  ``event_fn(t, y)`` expresses that as a scalar whose sign change
marks the moment, and the solver runs until it flips.

Locating the moment precisely is what the interpolants built for dense output
are for: once a step brackets the crossing, the event time is found by
bisecting the polynomial that step already fitted, with no extra
right-hand-side evaluations along the way.
"""

import math
from typing import Sequence, cast, final, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.autograd._functional import vjp as _vjp
from lucid.autograd.function import Function, FunctionCtx
from lucid.diffeq import _adaptive, _fixed, _fused, _implicit
from lucid.diffeq._tableau import ButcherTableau
from lucid.diffeq._typing import (
    EventFunction,
    Interpolant,
    RightHandSide,
    ScalarFactory,
    StageCheck,
)

__all__: list[str] = []


# Bisection stops once the bracket is this narrow.  Matches the reference
# implementation's default.
_EVENT_TOL = 1e-12

# A solve that never fires has to end somehow; this bounds it.
_MAX_EVENT_STEPS = 100_000

# Keeps the implicit-function denominator from dividing by zero when the event
# is tangential.  Matches the reference implementation's guard.
_EVENT_DENOM_FLOOR = 1e-12


def _sign(value: float) -> float:
    """Sign of a float, with zero kept distinct from either side."""
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _evaluate(event_fn: EventFunction, scalar: ScalarFactory, t: float, y: Tensor) -> float:
    """Evaluate the event function and bring its scalar value to the host.

    Parameters
    ----------
    event_fn : callable
        ``g(t, y)`` whose sign change marks the event.
    scalar : callable
        Builds the 0-D time tensor the callback expects.
    t : float
        Time to evaluate at.
    y : Tensor
        State at that time.

    Returns
    -------
    float
        The event value.

    Raises
    ------
    ValueError
        If ``event_fn`` returns anything but a single-element tensor.
    """
    out = event_fn(scalar(t), y)
    if not isinstance(out, Tensor):
        raise TypeError(f"event_fn must return a Tensor, got {type(out).__name__}")
    if out.numel() != 1:
        raise ValueError(f"event_fn must return a single-element tensor, got shape {out.shape}")
    return float(out.item())


def find_event(
    interp: Interpolant,
    t0: float,
    t1: float,
    event_fn: EventFunction,
    scalar: ScalarFactory,
    tol: float = _EVENT_TOL,
) -> tuple[float, Tensor]:
    """Bisect a bracketing interval down to the event time.

    Parameters
    ----------
    interp : callable
        ``interp(t) -> y``, the step's polynomial.
    t0, t1 : float
        Ends of a step known to bracket the crossing.
    event_fn : callable
        The event function.
    scalar : callable
        Builds the 0-D time tensor the callback expects.
    tol : float, default=1e-12
        Width the bracket is narrowed to.

    Returns
    -------
    tuple
        The event time and the state there.

    Notes
    -----
    Costs event-function calls only -- no right-hand-side evaluations at all,
    since the state comes from the interpolant the step already fitted.

    The search is Illinois, a regula-falsi variant, rather than bisection.
    Both keep a bracket around the root and so are equally safe, but bisection
    needs one call per bit of the answer: narrowing a step of order 0.1 to
    1e-12 takes about thirty-seven of them.  Illinois interpolates instead and
    reaches the same tolerance in single digits on a smooth event function,
    which is what the interpolant makes it.

    That matters more than the call count suggests.  Each call queues GPU work
    and then reads a sign back, and on a lazily-evaluated backend reading
    anything waits for everything queued behind it -- so the iterations are
    serialised flushes, and this search was over half the cost of an event
    solve on Metal.

    Plain regula falsi can stall, approaching the root from one side while the
    opposite endpoint never moves.  The Illinois correction halves the stale
    endpoint's value each time it is retained, which restores fast convergence,
    and a width check falls back to bisection if a step ever fails to shrink
    the bracket usefully.  The iteration cap is bisection's own count, so this
    can never take more calls than the method it replaces.
    """
    width = abs(t1 - t0)
    max_steps = 1 if width <= tol else int(math.ceil(math.log2(width / tol)))

    low, high = t0, t1
    g_low = _evaluate(event_fn, scalar, low, interp(low))
    g_high = _evaluate(event_fn, scalar, high, interp(high))
    previous = math.inf

    for _ in range(max_steps):
        if abs(high - low) <= tol:
            break

        # Secant through the bracket endpoints, guarded: a denominator that
        # has collapsed, or a root outside the bracket, means the linear model
        # is useless here and bisection is the honest fallback.
        span = g_high - g_low
        guess = low + (high - low) * 0.5 if span == 0.0 else low - g_low * (high - low) / span
        margin = 0.01 * abs(high - low)
        if not (min(low, high) + margin <= guess <= max(low, high) - margin):
            guess = (low + high) / 2.0

        # Convergence is measured on how far the estimate moved, not on the
        # bracket width.  Regula falsi narrows the bracket from one side only,
        # so its width can stay wide long after the root itself has settled --
        # testing the width instead would throw away the speed this method
        # exists for.
        if abs(guess - previous) <= tol:
            low = high = guess
            break
        previous = guess

        g_guess = _evaluate(event_fn, scalar, guess, interp(guess))
        if g_guess == 0.0:
            low = high = guess
            break

        if _sign(g_guess) == _sign(g_low):
            low, g_low = guess, g_guess
            # Illinois: the retained endpoint's value is halved so it cannot
            # anchor the secant indefinitely.
            g_high *= 0.5
        else:
            high, g_high = guess, g_guess
            g_low *= 0.5

    event_t = (low + high) / 2.0
    return event_t, interp(event_t)


def integrate_until_event(
    func: RightHandSide,
    y0: Tensor,
    t0: float,
    event_fn: EventFunction,
    tableau: ButcherTableau,
    scalar: ScalarFactory,
    check: StageCheck,
    *,
    rtol: float,
    atol: float,
    options: dict[str, object] | None,
    direction: float,
) -> tuple[float, Tensor]:
    """Step from ``t0`` until the event function changes sign.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y)``.
    y0 : Tensor
        Initial state at ``t0``.
    t0 : float
        Start time.
    event_fn : callable
        ``g(t, y)``; the solve ends when its sign flips.
    tableau : ButcherTableau
        Adaptive or fixed; a fixed method needs ``options["step_size"]``
        since there is no end time to build a grid from.
    scalar, check : callable
        Time-tensor factory and right-hand-side validator.
    rtol, atol : float
        Tolerances, for an adaptive method.
    options : dict or None
        Solver options.
    direction : float
        ``+1`` to search forwards in time, ``-1`` backwards.

    Returns
    -------
    tuple
        The event time and the state there.

    Raises
    ------
    ValueError
        If a fixed-step method is used without ``step_size``.
    RuntimeError
        If no sign change is found within the step budget.

    Notes
    -----
    The event is bracketed by an accepted step and then located inside it,
    so accuracy comes from the interpolant rather than from making the
    steps small.
    """
    start_value = _evaluate(event_fn, scalar, t0, y0)
    sign0 = _sign(start_value)
    if sign0 == 0.0:
        return t0, y0

    if tableau.is_adaptive:
        opts = _adaptive.parse_options(options)
        stepper = _adaptive.Stepper(
            func, y0, t0, tableau, scalar, check, rtol, atol, opts, direction
        )
        for _ in range(_MAX_EVENT_STEPS):
            step = stepper.advance()
            if _sign(_evaluate(event_fn, scalar, step.t1, step.y1)) == sign0:
                continue
            coeffs = stepper.fit(step)

            def interp_adaptive(
                t: float, _c: list[Tensor] = coeffs, _s: _adaptive.Step = step
            ) -> Tensor:
                return _fused.poly_eval(_c, _s.t0, _s.t1, t)

            return find_event(interp_adaptive, step.t0, step.t1, event_fn, scalar)
    else:
        fixed_opts = _fixed.parse_options(options)
        if fixed_opts.step_size is None:
            raise ValueError(
                f"method {tableau.name!r} is a fixed-step method, so an event "
                f"solve needs options={{'step_size': ...}} — there is no end "
                f"time to build a grid from"
            )
        dt = direction * fixed_opts.step_size
        t_cur, y = t0, y0
        f0 = check(func(scalar(t0), y), 0, 0) if fixed_opts.interp == "cubic" else None
        # An implicit tableau's stage rows are square, so the explicit stage
        # recipe below would read coefficients that do not belong to it.  Its
        # steps come from the nonlinear solve instead; everything after that —
        # bracketing, the Hermite fit, the bisection — is shared.
        implicit_opts = _implicit.parse_options(options) if tableau.is_implicit else None

        for index in range(_MAX_EVENT_STEPS):
            t_next = t_cur + dt
            if implicit_opts is not None:
                y_next = _implicit._step(
                    func, y, t_cur, dt, tableau, scalar, check, index, implicit_opts
                )
            else:
                ks: list[Tensor] = []
                for stage in range(tableau.stages):
                    stage_y = _fused.combine(y, ks, tableau.a[stage], dt)
                    stage_t = scalar(t_cur + tableau.c[stage] * dt)
                    ks.append(check(func(stage_t, stage_y), index, stage))
                y_next = _fused.combine(y, ks, tableau.b, dt)

            if _sign(_evaluate(event_fn, scalar, t_next, y_next)) != sign0:
                if fixed_opts.interp == "cubic":
                    assert f0 is not None
                    f1 = check(func(scalar(t_next), y_next), index, tableau.stages)
                    coeffs = _fixed._interp_fit_cubic(y, y_next, f0, f1, dt)
                else:
                    coeffs = _fixed._interp_fit_linear(y, y_next)

                def interp_fixed(
                    t: float, _c: list[Tensor] = coeffs, _a: float = t_cur, _b: float = t_next
                ) -> Tensor:
                    return _fused.poly_eval(_c, _a, _b, t)

                return find_event(interp_fixed, t_cur, t_next, event_fn, scalar)

            if fixed_opts.interp == "cubic":
                f0 = check(func(scalar(t_next), y_next), index, tableau.stages)
            t_cur, y = t_next, y_next

    raise RuntimeError(
        f"event_fn did not change sign within {_MAX_EVENT_STEPS} steps from "
        f"t={t0!r}; check the event condition or the integration direction"
    )


def solution_pair(y0: Tensor, y_event: Tensor) -> Tensor:
    """Stack the initial and event states into the returned trajectory.

    Parameters
    ----------
    y0 : Tensor
        State the solve started from.
    y_event : Tensor
        State when the event fired.

    Returns
    -------
    Tensor
        Shape ``(2, *y0.shape)``.

    Notes
    -----
    An event solve has no output grid to report against — the only two
    times that mean anything are where it started and where it stopped.
    """
    states: Sequence[Tensor] = (y0, y_event)
    target = lucid.promote_types(y0.dtype, y_event.dtype)
    return lucid.stack([s if s.dtype == target else s.to(target) for s in states], dim=0)


@final
class _EventTimeGradient(Function):
    r"""Give the event time a derivative, by way of the implicit relation.

    Bisection reaches the event time through sign comparisons, and a
    comparison has no derivative, so the time arrives with no graph behind it.
    It is still a differentiable function of everything the trajectory depends
    on, because it is pinned by

    .. math::

        g(t^*, y(t^*)) = 0.

    Differentiating that identity along the trajectory and solving for
    :math:`t^*` gives

    .. math::

        \frac{\partial t^*}{\partial \theta}
        = -\frac{\partial_y g \cdot \partial_\theta y}
                {\partial_t g + \partial_y g \cdot f},

    which says a cotangent arriving at the event time can be turned into one
    on the state at the event -- and *that* is connected to ``y0`` and to any
    parameters.  So the node takes the state as its input and the time as its
    output, and its backward is the fraction above.

    The denominator is the total derivative of the event function along the
    trajectory.  It vanishes exactly when the event is tangential rather than
    transversal, which is the case where the event time genuinely has no
    derivative -- an arbitrarily small change can make the crossing disappear.
    """

    _once_differentiable = True  # second derivatives through it are unverified

    @override
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        state_t: Tensor,
        *,
        event_t: Tensor,
        func: RightHandSide,
        event_fn: EventFunction,
    ) -> Tensor:
        """Hand back the event time; the derivative is all in ``backward``."""
        ctx.save_for_backward(event_t.detach(), state_t.detach())
        ctx.func = func
        ctx.event_fn = event_fn
        return event_t.detach()

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_t: Tensor) -> Tensor:
        """Turn the event time's cotangent into the state's."""
        event_t, state_t = ctx.saved_tensors
        func = cast(RightHandSide, ctx.func)
        event_fn = cast(EventFunction, ctx.event_fn)

        # Both need gradient tracking for the partials of ``g`` to exist; the
        # graph built here is local to this backward pass.
        t_var = event_t.detach().requires_grad_(True)
        y_var = state_t.detach().requires_grad_(True)

        with lucid.enable_grad():
            f_val = func(t_var, y_var).detach()
            _, grads = _vjp(
                lambda tt, yy: event_fn(tt, yy).reshape(()),
                (t_var, y_var),
                lucid.ones((), dtype=state_t.dtype),
            )

        # A partial ``g`` does not depend on comes back as zeros: ignoring
        # the state entirely routes nothing to the trajectory.
        dg_dt, dg_dy = grads
        dcdt = dg_dt + (dg_dy * f_val).sum()
        return dg_dy * (-grad_t.reshape(()) / (dcdt + _EVENT_DENOM_FLOOR))


def differentiable_event_time(
    event_t: Tensor, state_t: Tensor, func: RightHandSide, event_fn: EventFunction
) -> tuple[Tensor, Tensor]:
    """Attach derivatives to an event time and to the state that shares it.

    Parameters
    ----------
    event_t : Tensor
        The event time found by bisection, carrying no graph.
    state_t : Tensor
        The state there -- the tensor holding the graph back to ``y0`` and any
        parameters.
    func : callable
        Right-hand side ``f(t, y)``.
    event_fn : callable
        The event function ``g(t, y)``.

    Returns
    -------
    tuple of Tensor
        ``(event_t, state_t)``, both differentiable.

    Notes
    -----
    Two derivatives are missing before this runs, and both are restored here.
    The event time gets one from :class:`_EventTimeGradient`.  The state gets
    the other from the correction below: the state was read off the
    interpolant at a *host* float, so nothing recorded that moving the event
    time moves the state with it, at rate ``f``.  Adding ``f * (t - t.detach())``
    changes no value -- the bracket is identically zero -- while giving
    autograd exactly that missing edge.

    When nothing upstream requires a gradient this is a pair of detaches and
    costs nothing.  Otherwise it costs one right-hand-side evaluation and one
    vector-Jacobian product through ``event_fn``, both only in ``backward``.
    """
    if not state_t.requires_grad:
        return event_t, state_t

    time = cast(
        Tensor, _EventTimeGradient.apply(state_t, event_t=event_t, func=func, event_fn=event_fn)
    )
    with lucid.no_grad():
        slope = func(event_t, state_t)
    return time, state_t + slope * (time - time.detach())
