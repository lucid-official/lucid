"""Adaptive-step explicit Runge-Kutta integration with dense output.

An adaptive method pairs two embedded solutions of different order.  Their
difference estimates the local error of the step, which a controller turns
into a step size: accept and grow when the estimate is comfortably inside
tolerance, shrink and retry when it is not.  The user's ``t`` grid stops being
the integration grid and becomes a set of *output* times, reached by
interpolating inside whichever step happens to span each one.

This costs one host synchronisation per step — the controller must branch on
the error ratio, so that scalar has to cross to the host. It is the
unavoidable price of adaptivity, and the reason this loop has a different
performance character from the fixed-grid one: the fused error-norm op exists
so the price stays at exactly one sync rather than several.
"""

import math
from dataclasses import dataclass
from typing import Sequence, SupportsFloat, cast

from lucid._tensor.tensor import Tensor
from lucid.diffeq import _fused
from lucid.diffeq._tableau import ButcherTableau
from lucid.diffeq._typing import RightHandSide, ScalarFactory, StageCheck

__all__: list[str] = []


# Upstream's controller defaults, kept identical so a solve reproduces the
# same step sequence.
_SAFETY = 0.9
_IFACTOR = 10.0
_DFACTOR = 0.2
_MAX_NUM_STEPS = 2**31 - 1


@dataclass(frozen=True)
class AdaptiveOptions:
    """Parsed ``options`` dict for an adaptive solver.

    Attributes
    ----------
    min_step, max_step : float
        Bounds on the step magnitude.
    first_step : float or None
        Initial step magnitude; chosen automatically when ``None``.
    step_t, jump_t : tuple of float
        Times the integrator must land on exactly.  ``step_t`` continues
        through; ``jump_t`` marks a discontinuity in the right-hand side, so
        the step after it restarts just past the time.
    safety, ifactor, dfactor : float
        Controller gains — the target margin, and the most a step may grow or
        shrink in one adjustment.
    max_num_steps : int
        Budget of accepted steps before the solve gives up.
    norm_prefix : int or None
        When set, the error norm covers only the first this-many elements of
        the state.  Private, and set only by :func:`odeint_adjoint` for
        ``adjoint_options={"norm": "seminorm"}``: the augmented backward state
        is ``[y | adjoint | parameter gradients]``, and the gradient block is
        a quadrature whose error never feeds back into the dynamics, so
        letting it drive the step size costs steps and buys nothing.
    """

    min_step: float = 0.0
    max_step: float = math.inf
    first_step: float | None = None
    step_t: tuple[float, ...] = ()
    jump_t: tuple[float, ...] = ()
    safety: float = _SAFETY
    ifactor: float = _IFACTOR
    dfactor: float = _DFACTOR
    max_num_steps: int = _MAX_NUM_STEPS
    norm_prefix: int | None = None


_ADAPTIVE_KEYS = frozenset(
    {
        "min_step",
        "max_step",
        "first_step",
        "step_t",
        "jump_t",
        "safety",
        "ifactor",
        "dfactor",
        "max_num_steps",
        "dtype",
        "norm",
        # Private, injected by `odeint_adjoint` for `norm="seminorm"`; see
        # `norm_prefix` on AdaptiveOptions.  Not part of the public option
        # surface -- the leading underscore marks it Tier 3, like every other
        # `_`-prefixed name in the package.
        "_norm_prefix",
    }
)


def parse_options(options: dict[str, object] | None) -> AdaptiveOptions:
    """Validate and convert an ``options`` mapping for an adaptive solver.

    Parameters
    ----------
    options : dict or None
        Caller-supplied options; ``None`` means all defaults.

    Returns
    -------
    AdaptiveOptions
        The parsed options.

    Raises
    ------
    ValueError
        If a key is not recognised, or a value is out of range.

    Notes
    -----
    ``dtype`` is accepted and ignored: step control here runs in host doubles
    unconditionally, which is what that option exists to request.

    ``norm`` is **rejected** rather than ignored.  The error norm is a fused
    engine kernel, so a caller-supplied one cannot be honoured, and accepting
    the key silently would leave someone believing they had changed the step
    control when they had not — the one value that does have an effect,
    ``"seminorm"``, is handled by :func:`odeint_adjoint` before the options
    reach here.
    """
    if options is None:
        return AdaptiveOptions()
    unknown = set(options) - _ADAPTIVE_KEYS
    if unknown:
        raise ValueError(
            f"unknown option(s) for an adaptive method: {sorted(unknown)}; "
            f"expected a subset of {sorted(_ADAPTIVE_KEYS)}"
        )
    if "norm" in options:
        raise NotImplementedError(
            "option 'norm' is not supported: the error norm is a fused engine "
            "kernel, so a caller-supplied norm cannot be honoured. The one "
            "value with an effect, 'seminorm', is accepted by odeint_adjoint "
            "through adjoint_options."
        )

    def _real(key: str, default: float) -> float:
        raw = options.get(key, default)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"option {key!r} must be a real number, got {type(raw).__name__}")
        return float(raw)

    def _count(key: str, default: int) -> int:
        raw = options.get(key, default)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(f"option {key!r} must be an int, got {type(raw).__name__}")
        return raw

    def _times(key: str) -> tuple[float, ...]:
        raw = options.get(key)
        if raw is None:
            return ()
        if isinstance(raw, Tensor):
            values = [float(v) for v in cast(list[SupportsFloat], raw.tolist())]
        elif isinstance(raw, (list, tuple)):
            values = [float(cast(SupportsFloat, v)) for v in raw]
        else:
            raise TypeError(
                f"option {key!r} must be a sequence of times or a 1-D tensor, "
                f"got {type(raw).__name__}"
            )
        return tuple(sorted(values))

    first_raw = options.get("first_step")
    if first_raw is None:
        first = None
    else:
        if isinstance(first_raw, bool) or not isinstance(first_raw, (int, float)):
            raise ValueError(
                f"option 'first_step' must be a real number, " f"got {type(first_raw).__name__}"
            )
        first = abs(float(first_raw))
        if first == 0.0:
            raise ValueError("first_step must be non-zero")

    parsed = AdaptiveOptions(
        min_step=abs(_real("min_step", 0.0)),
        max_step=abs(_real("max_step", math.inf)),
        first_step=first,
        step_t=_times("step_t"),
        jump_t=_times("jump_t"),
        safety=_real("safety", _SAFETY),
        ifactor=_real("ifactor", _IFACTOR),
        dfactor=_real("dfactor", _DFACTOR),
        max_num_steps=_count("max_num_steps", _MAX_NUM_STEPS),
        norm_prefix=(None if options.get("_norm_prefix") is None else _count("_norm_prefix", 0)),
    )
    if parsed.min_step > parsed.max_step:
        raise ValueError(
            f"min_step ({parsed.min_step}) must not exceed max_step ({parsed.max_step})"
        )
    if parsed.ifactor < 1.0:
        raise ValueError(f"ifactor must be >= 1, got {parsed.ifactor}")
    if not 0.0 < parsed.dfactor <= 1.0:
        raise ValueError(f"dfactor must lie in (0, 1], got {parsed.dfactor}")
    if parsed.max_num_steps < 1:
        raise ValueError(f"max_num_steps must be >= 1, got {parsed.max_num_steps}")
    if parsed.norm_prefix is not None and parsed.norm_prefix < 1:
        raise ValueError(f"_norm_prefix must be >= 1, got {parsed.norm_prefix}")
    return parsed


def optimal_step_size(
    last_step: float,
    ratio: float,
    safety: float,
    ifactor: float,
    dfactor: float,
    order: int,
) -> float:
    """Propose the next step magnitude from the last one and its error ratio.

    Parameters
    ----------
    last_step : float
        Magnitude of the step just attempted.
    ratio : float
        Its error ratio; at most ``1`` means the step was accepted.
    safety, ifactor, dfactor : float
        Controller gains.
    order : int
        Order of the method, the exponent applied to the ratio.

    Returns
    -------
    float
        The proposed magnitude, always positive.

    Notes
    -----
    A step that was accepted is never shrunk — ``dfactor`` is neutralised in
    that case, so a comfortable step can only grow or stay put.
    """
    if ratio == 0.0:
        return last_step * ifactor
    if ratio < 1.0:
        dfactor = 1.0
    # ``math.pow`` rather than ``**``: the ratio is strictly positive here, but
    # the operator is typed to allow a complex result for a negative base.
    factor = min(ifactor, max(safety / math.pow(ratio, 1.0 / order), dfactor))
    return last_step * factor


def select_initial_step(
    func: RightHandSide,
    t0: float,
    y0: Tensor,
    f0: Tensor,
    order: int,
    rtol: float,
    atol: float,
    direction: float,
    scalar: ScalarFactory,
) -> float:
    """Pick a starting step magnitude from the local scale of the problem.

    Parameters
    ----------
    func : callable
        The right-hand side.
    t0 : float
        Initial time.
    y0 : Tensor
        Initial state.
    f0 : Tensor
        ``func(t0, y0)``, already computed.
    order : int
        Order of the *error estimate*, one below the method's own order.  The
        trial step is sized so the estimated local error lands near the
        tolerance, and it is the estimate's order that governs how that error
        shrinks with the step.  Passing the method's order instead makes the
        first step too large by a factor that grows with the order, which a
        high-order method never fully recovers from: its step sequence stays
        shifted, and output times land further inside a step, where only the
        interpolant's accuracy is available.
    rtol, atol : float
        Tolerances.
    direction : float
        ``+1`` integrating forwards, ``-1`` backwards.
    scalar : callable
        Builds the 0-D time tensor the right-hand side expects.

    Returns
    -------
    float
        A positive step magnitude.

    Notes
    -----
    The standard two-probe heuristic: size a trial step from the ratio of the
    state to its derivative, take it, and refine from how much the derivative
    changed.  Costs one extra right-hand-side call at the start of a solve.

    All three norms it needs are the same fused kernel the step controller
    uses, with the estimate standing in for the state or derivative — so this
    adds no new engine surface.
    """
    d0 = _fused.error_ratio(y0, y0, [y0], [1.0], 1.0, rtol, atol)
    d1 = _fused.error_ratio(y0, y0, [f0], [1.0], 1.0, rtol, atol)

    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1

    # A state or derivative that is already non-finite drives `h0` to zero or
    # NaN, and every line below divides by it.  These norms are host floats,
    # so that is a ZeroDivisionError -- an arithmetic accident that says
    # nothing about the problem.  Returning a zero step instead lets the
    # stepper's underflow guard report it against the integration, which is
    # both the useful message and what the reference ends up doing (its norms
    # are tensors, so the same division yields an infinity and the step it
    # eventually derives is zero).
    if not (h0 > 0.0) or not math.isfinite(h0):
        return 0.0

    y1 = _fused.combine(y0, [f0], [1.0], direction * h0)
    f1 = func(scalar(t0 + direction * h0), y1)
    d2 = _fused.error_ratio(y0, y0, [f1, f0], [1.0, -1.0], 1.0, rtol, atol) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        # `max` picks the first argument when the comparison is false, so a
        # NaN here would slip through as `d1`; guard the divisor rather than
        # trusting it.  An infinite one needs no guard -- the division simply
        # gives zero, which is the right answer for "no step is small enough".
        largest = max(d1, d2)
        h1 = (0.01 / largest) ** (1.0 / (order + 1)) if largest > 0.0 else 0.0

    return min(100.0 * h0, h1)


def interp_fit(
    y0: Tensor, y1: Tensor, y_mid: Tensor, f0: Tensor, f1: Tensor, dt: float
) -> list[Tensor]:
    r"""Fit the quartic that reproduces a step's endpoints, midpoint and slopes.

    Parameters
    ----------
    y0, y1 : Tensor
        States at the two ends of the step.
    y_mid : Tensor
        State at the midpoint, from the tableau's ``mid`` weights.
    f0, f1 : Tensor
        Derivatives at the two ends.
    dt : float
        Step size.

    Returns
    -------
    list of Tensor
        Five coefficients in descending powers of the normalised time.

    Notes
    -----
    On :math:`x = (t - t_0) / \Delta t` the polynomial
    :math:`p(x) = ax^4 + bx^3 + cx^2 + dx + e` is pinned by five conditions —
    :math:`p(0) = y_0`, :math:`p(1) = y_1`, :math:`p(1/2) = y_{mid}`,
    :math:`p'(0) = \Delta t f_0`, :math:`p'(1) = \Delta t f_1` — and solving
    that system gives the coefficients below.

    The layout is an implementation detail, not part of any public contract,
    so it is simply whatever :func:`interp_evaluate` expects.

    The five inputs are promoted as a group first.  Under ``autocast`` the
    derivatives come back at a lower precision than the state, and a
    coefficient built from derivatives alone would otherwise carry that
    precision into the engine on its own — with nothing to promote against.
    """
    y0, y1, y_mid, f0, f1 = _fused._promote([y0, y1, y_mid, f0, f1])
    return [
        _fused.lincomb([f1, f0, y0, y1, y_mid], [2.0 * dt, -2.0 * dt, -8.0, -8.0, 16.0]),
        _fused.lincomb([f0, f1, y0, y1, y_mid], [5.0 * dt, -3.0 * dt, 18.0, 14.0, -32.0]),
        _fused.lincomb([f1, f0, y0, y1, y_mid], [dt, -4.0 * dt, -11.0, -5.0, 16.0]),
        _fused.lincomb([f0], [dt]),
        y0,
    ]


def interp_evaluate(coeffs: Sequence[Tensor], t0: float, t1: float, t: float) -> Tensor:
    """Evaluate a fitted step polynomial at a time inside the step.

    Parameters
    ----------
    coeffs : sequence of Tensor
        Output of :func:`interp_fit`.
    t0, t1 : float
        Ends of the step the polynomial was fitted on.
    t : float
        Query time, inside ``[t0, t1]`` (either orientation).

    Returns
    -------
    Tensor
        The interpolated state.
    """
    return _fused.poly_eval(coeffs, t0, t1, t)


def _clamp(step: float, opts: AdaptiveOptions) -> float:
    """Clamp a proposed step magnitude into the configured bounds."""
    return min(max(step, opts.min_step), opts.max_step)


def _next_forced(times: Sequence[float], t: float, direction: float) -> float | None:
    """Return the next time in ``times`` strictly ahead of ``t``, if any."""
    if direction > 0:
        ahead = [v for v in times if v > t]
        return min(ahead) if ahead else None
    ahead = [v for v in times if v < t]
    return max(ahead) if ahead else None


@dataclass(frozen=True)
class Step:
    """One accepted step, with everything an interpolant needs.

    Attributes
    ----------
    t0, t1 : float
        The times the step ran between.
    dt : float
        Signed step size, ``t1 - t0``.
    y0, y1 : Tensor
        States at those times.
    f0, f1 : Tensor
        Derivatives at those times.
    ks : list of Tensor
        The stage derivatives, needed for the midpoint state.
    """

    t0: float
    t1: float
    dt: float
    y0: Tensor
    y1: Tensor
    f0: Tensor
    f1: Tensor
    ks: list[Tensor]


class Stepper:
    """Drives one adaptive solve, one accepted step at a time.

    Both the output-times loop and the dense-output collector run on this, so
    the step body — stages, error ratio, accept/reject, the FSAL shortcut —
    exists once.  Duplicating it is how a stage input goes missing and a
    method silently loses its order.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y)``.
    y0 : Tensor
        Initial state.
    t0 : float
        Initial time.
    tableau : ButcherTableau
        An adaptive tableau.
    scalar : callable
        Builds the 0-D time tensor the right-hand side expects.
    check : callable
        Validates a right-hand-side result and returns it.
    rtol, atol : float
        Tolerances.
    opts : AdaptiveOptions
        Controller settings.
    direction : float
        ``+1`` integrating forwards, ``-1`` backwards.
    """

    def __init__(
        self,
        func: RightHandSide,
        y0: Tensor,
        t0: float,
        tableau: ButcherTableau,
        scalar: ScalarFactory,
        check: StageCheck,
        rtol: float,
        atol: float,
        opts: AdaptiveOptions,
        direction: float,
    ) -> None:
        self._func = func
        self._tableau = tableau
        self._scalar = scalar
        self._check = check
        self._rtol = rtol
        self._atol = atol
        self._opts = opts
        self._direction = direction
        self._forced = tuple(sorted({*opts.step_t, *opts.jump_t}))

        self.t = t0
        self.y = y0
        self.f = check(func(scalar(t0), y0), 0, 0)
        self.accepted = 0

        if opts.first_step is not None:
            step = opts.first_step
        else:
            step = select_initial_step(
                func, t0, y0, self.f, tableau.order - 1, rtol, atol, direction, scalar
            )
        self.step = _clamp(step, opts)

    def advance(self) -> Step:
        """Take one accepted step, retrying at smaller sizes until it passes.

        Returns
        -------
        Step
            The accepted step.  The stepper's own ``t`` / ``y`` / ``f`` have
            already moved to its end.

        Raises
        ------
        RuntimeError
            If the step size collapses, or the accepted-step budget runs out.
        """
        tableau = self._tableau
        while True:
            # Only times the caller pinned constrain the step.  Output times
            # deliberately do not: clipping to them would tie the step size to
            # the output grid, which is the coupling dense output exists to
            # remove, and would change the step sequence.
            limit = _next_forced(self._forced, self.t, self._direction)
            if limit is not None:
                self.step = min(self.step, abs(limit - self.t))

            dt = self._direction * self.step
            t_next = self.t + dt
            # `not (... > 0)` rather than `<= 0`, and `t_next == self.t`
            # rather than a threshold on the step: both survive a NaN, and a
            # NaN step is exactly what a diverging problem produces.  The
            # second condition is the one that matters -- a step can be
            # positive and still too small to move `t` at all, which loops
            # forever rather than failing.
            if not (self.step > 0.0) or t_next == self.t:
                raise RuntimeError(
                    f"underflow in dt {self.step!r} at t={self.t!r}: the step "
                    f"size can no longer advance the integration; the problem "
                    f"may be diverging, or too stiff for an explicit method"
                )
            ks: list[Tensor] = [self.f]
            for stage in range(1, tableau.stages):
                stage_y = _fused.combine(self.y, ks, tableau.a[stage], dt)
                stage_t = self._scalar(self.t + tableau.c[stage] * dt)
                ks.append(self._check(self._func(stage_t, stage_y), self.accepted, stage))

            y_next = _fused.combine(self.y, ks, tableau.b, dt)
            assert tableau.b_error is not None
            # A leading slice of a contiguous 1-D state is itself contiguous,
            # so restricting the norm costs metadata and no copy.
            prefix = self._opts.norm_prefix
            if prefix is None:
                ratio = _fused.error_ratio(
                    self.y, y_next, ks, tableau.b_error, dt, self._rtol, self._atol
                )
            else:
                ratio = _fused.error_ratio(
                    self.y[:prefix],
                    y_next[:prefix],
                    [k[:prefix] for k in ks],
                    tableau.b_error,
                    dt,
                    self._rtol,
                    self._atol,
                )

            proposed = _clamp(
                optimal_step_size(
                    self.step,
                    ratio,
                    self._opts.safety,
                    self._opts.ifactor,
                    self._opts.dfactor,
                    tableau.order,
                ),
                self._opts,
            )
            # `not (ratio <= 1.0)` rather than `ratio > 1.0`.  The difference
            # only shows when the ratio is NaN, and that is the case that
            # matters: a diverging problem drives the state to NaN, every
            # comparison against NaN is false, and written the other way round
            # the step would be *accepted* -- so the solver would march to the
            # end and hand back NaN without a word.  A non-finite ratio is a
            # rejection, and the guards below turn a rejection that cannot be
            # retried into an error.
            if not (ratio <= 1.0):
                if not (proposed < self.step):
                    # The step cannot shrink any further, so retrying would
                    # loop forever on the same rejected step.
                    if not math.isfinite(ratio):
                        raise RuntimeError(
                            f"non-finite values in the state at t={t_next!r}: "
                            f"the error estimate is {ratio!r}, so the step "
                            f"cannot be judged; the problem is diverging"
                        )
                    raise RuntimeError(
                        f"step rejected at the min_step floor "
                        f"({self._opts.min_step!r}) at t={self.t!r} with error "
                        f"ratio {ratio!r}; loosen rtol/atol or lower min_step"
                    )
                self.step = proposed
                continue

            self.accepted += 1
            if self.accepted > self._opts.max_num_steps:
                raise RuntimeError(
                    f"exceeded max_num_steps ({self._opts.max_num_steps}) at "
                    f"t={t_next!r}; loosen rtol/atol or raise the budget"
                )

            # The last stage of an FSAL tableau already evaluated the
            # right-hand side at the new state, so it is the next step's first
            # stage for free.  Otherwise that derivative has to be computed:
            # reusing the final stage would seed the next step from a point
            # that is not y1 and cost the method its order.
            if tableau.is_fsal:
                f_next = ks[-1]
            else:
                f_next = self._check(
                    self._func(self._scalar(t_next), y_next),
                    self.accepted,
                    tableau.stages,
                )

            step = Step(
                t0=self.t,
                t1=t_next,
                dt=dt,
                y0=self.y,
                y1=y_next,
                f0=self.f,
                f1=f_next,
                ks=ks,
            )
            self.t, self.y, self.f = t_next, y_next, f_next
            self.step = proposed
            return step

    def fit(self, step: Step) -> list[Tensor]:
        """Build the interpolating polynomial for an accepted step."""
        assert self._tableau.mid is not None
        y_mid = _fused.combine(step.y0, step.ks, self._tableau.mid, step.dt)
        return interp_fit(step.y0, step.y1, y_mid, step.f0, step.f1, step.dt)


def integrate(
    func: RightHandSide,
    y0: Tensor,
    grid: list[float],
    tableau: ButcherTableau,
    scalar: ScalarFactory,
    check: StageCheck,
    *,
    rtol: float,
    atol: float,
    options: dict[str, object] | None,
    return_trajectory: bool,
) -> tuple[Tensor, list[Tensor]]:
    """Integrate to every time in ``grid`` with adaptive steps and interpolation.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y)``.
    y0 : Tensor
        Initial state, at ``grid[0]``.
    grid : list of float
        Output times, strictly monotonic.
    tableau : ButcherTableau
        An adaptive tableau.
    scalar : callable
        Builds the 0-D time tensor the right-hand side expects.
    check : callable
        Validates a right-hand-side result and returns it.
    rtol, atol : float
        Tolerances.
    options : dict or None
        Controller options, see :class:`AdaptiveOptions`.
    return_trajectory : bool
        Whether to collect the state at every output time.

    Returns
    -------
    tuple
        The final state, and the collected trajectory (empty when not
        requested).

    Notes
    -----
    The interpolant is fitted lazily: only a step that actually spans an
    output time pays for it.
    """
    opts = parse_options(options)
    direction = 1.0 if grid[-1] > grid[0] else -1.0
    stepper = Stepper(func, y0, grid[0], tableau, scalar, check, rtol, atol, opts, direction)

    trajectory: list[Tensor] = [y0] if return_trajectory else []
    span: tuple[float, float] | None = None
    coeffs: list[Tensor] = []

    for target in grid[1:]:
        while direction * (stepper.t - target) < 0.0:
            step = stepper.advance()
            if direction * (step.t1 - target) >= 0.0:
                span = (step.t0, step.t1)
                coeffs = stepper.fit(step)
                break

        if return_trajectory:
            if stepper.t == target or span is None:
                trajectory.append(stepper.y)
            else:
                trajectory.append(interp_evaluate(coeffs, span[0], span[1], target))

    # ``stepper.y`` sits at the last accepted step, which generally overshoots
    # the final output time; the value asked for is the interpolated one.
    if stepper.t == grid[-1] or span is None:
        final = stepper.y
    else:
        final = interp_evaluate(coeffs, span[0], span[1], grid[-1])
    return final, trajectory


def integrate_dense(
    func: RightHandSide,
    y0: Tensor,
    t0: float,
    t1: float,
    tableau: ButcherTableau,
    scalar: ScalarFactory,
    check: StageCheck,
    *,
    rtol: float,
    atol: float,
    options: dict[str, object] | None,
) -> list[tuple[float, float, list[Tensor]]]:
    """Integrate once across ``[t0, t1]``, keeping every step's interpolant.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y)``.
    y0 : Tensor
        Initial state at ``t0``.
    t0, t1 : float
        Ends of the interval.  ``t1 < t0`` integrates backwards.
    tableau : ButcherTableau
        An adaptive tableau.
    scalar : callable
        Builds the 0-D time tensor the right-hand side expects.
    check : callable
        Validates a right-hand-side result and returns it.
    rtol, atol : float
        Tolerances.
    options : dict or None
        Controller options.

    Returns
    -------
    list of (float, float, list of Tensor)
        One entry per accepted step: its start, its end, and its polynomial.

    Notes
    -----
    Every step is fitted here, unlike :func:`integrate` — that is the whole
    point, since the caller may later ask for any time in the interval.
    """
    opts = parse_options(options)
    direction = 1.0 if t1 > t0 else -1.0
    stepper = Stepper(func, y0, t0, tableau, scalar, check, rtol, atol, opts, direction)

    segments: list[tuple[float, float, list[Tensor]]] = []
    while direction * (stepper.t - t1) < 0.0:
        step = stepper.advance()
        segments.append((step.t0, step.t1, stepper.fit(step)))
    return segments
