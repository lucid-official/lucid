"""Fixed-grid explicit Runge-Kutta integration.

By default the caller's ``t`` *is* the integration grid: one Runge-Kutta step
per interval, no interpolation, and accuracy controlled by choosing a finer
grid.  ``options`` decouples the two — give a ``step_size`` (or build the grid
yourself with ``grid_constructor``) and the solver steps on that grid instead,
interpolating to the times in ``t``.

That is the same separation the adaptive solver has, minus the error control:
there the step size comes from a tolerance, here it comes from the caller.
"""

import math
from dataclasses import dataclass
from typing import Callable, Sequence, SupportsFloat, cast

from lucid._tensor.tensor import Tensor
from lucid.diffeq import _fused
from lucid.diffeq._tableau import ButcherTableau
from lucid.diffeq._typing import (
    InterpKind,
    RightHandSide,
    ScalarFactory,
    StageCheck,
    is_interp_kind,
)

__all__: list[str] = []


# The complete set of accepted values, kept beside the validating helper
# so the runtime check and the static type cannot drift apart.
_INTERP_KINDS: tuple[InterpKind, ...] = ("linear", "cubic")


@dataclass(frozen=True)
class FixedOptions:
    """Parsed ``options`` dict for a fixed-step solver.

    Attributes
    ----------
    step_size : float or None
        Integration step magnitude.  ``None`` means the output grid is the
        integration grid.
    grid_constructor : callable or None
        ``f(func, y0, t) -> grid`` building the integration grid explicitly.
        Takes precedence over ``step_size``.
    interp : InterpKind
        ``"linear"`` or ``"cubic"``, how to reach output times that fall
        inside a step.  Unused when the integration grid is the output grid.
    perturb : bool
        Nudge the times handed to the right-hand side off the step endpoints
        by one float step, so a discontinuity sitting exactly on a grid point
        is evaluated on the side the step is coming from.
    """

    step_size: float | None = None
    grid_constructor: Callable[..., object] | None = None
    interp: InterpKind = "linear"
    perturb: bool = False


_FIXED_KEYS = frozenset({"step_size", "grid_constructor", "interp", "perturb"})


def parse_options(options: dict[str, object] | None) -> FixedOptions:
    """Validate and convert an ``options`` mapping for a fixed-step solver.

    Parameters
    ----------
    options : dict or None
        Caller-supplied options; ``None`` means all defaults.

    Returns
    -------
    FixedOptions
        The parsed options.

    Raises
    ------
    ValueError
        If a key is not recognised, or a value is out of range.
    """
    if options is None:
        return FixedOptions()
    unknown = set(options) - _FIXED_KEYS
    if unknown:
        raise ValueError(
            f"unknown option(s) for a fixed-step method: {sorted(unknown)}; "
            f"expected a subset of {sorted(_FIXED_KEYS)}"
        )

    raw_step = options.get("step_size")
    if raw_step is None:
        step: float | None = None
    else:
        if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
            raise ValueError(
                f"option 'step_size' must be a real number, " f"got {type(raw_step).__name__}"
            )
        step = abs(float(raw_step))
        if step == 0.0 or not math.isfinite(step):
            raise ValueError(f"step_size must be finite and non-zero, got {raw_step!r}")

    builder = options.get("grid_constructor")
    if builder is not None and not callable(builder):
        raise ValueError(
            f"option 'grid_constructor' must be callable, got {type(builder).__name__}"
        )

    interp = options.get("interp", "linear")
    if not is_interp_kind(interp):
        raise ValueError(f"interp must be one of {_INTERP_KINDS}, got {interp!r}")

    perturb = options.get("perturb", False)
    if not isinstance(perturb, bool):
        raise TypeError(f"option 'perturb' must be a bool, got {type(perturb).__name__}")

    return FixedOptions(
        step_size=step,
        grid_constructor=cast(Callable[..., object] | None, builder),
        interp=interp,
        perturb=perturb,
    )


def build_grid(opts: FixedOptions, func: RightHandSide, y0: Tensor, t: list[float]) -> list[float]:
    """Return the grid the solver actually steps on.

    Parameters
    ----------
    opts : FixedOptions
        Parsed options.
    func : callable
        Right-hand side, forwarded to a custom ``grid_constructor``.
    y0 : Tensor
        Initial state, forwarded to a custom ``grid_constructor``.
    t : list of float
        The output times.

    Returns
    -------
    list of float
        The integration grid.  Identical to ``t`` when neither ``step_size``
        nor ``grid_constructor`` was given.

    Raises
    ------
    ValueError
        If a custom ``grid_constructor`` returns a grid that does not start
        and end at the requested interval.
    """
    if opts.grid_constructor is not None:
        built = opts.grid_constructor(func, y0, t)
        if isinstance(built, Tensor):
            grid = [float(v) for v in cast(list[SupportsFloat], built.tolist())]
        else:
            grid = [float(cast(SupportsFloat, v)) for v in cast(Sequence[object], built)]
        if len(grid) < 2 or grid[0] != t[0] or grid[-1] != t[-1]:
            raise ValueError(
                f"grid_constructor must return a grid spanning t[0]..t[-1] "
                f"({t[0]!r}..{t[-1]!r}), got {grid[:1]}..{grid[-1:]}"
            )
        return grid

    if opts.step_size is None:
        return list(t)

    # Match the reference construction: ceil the span into whole steps, then
    # pin the last point back onto t[-1] so the grid ends exactly where the
    # caller asked rather than one rounding step past it.
    span = t[-1] - t[0]
    direction = 1.0 if span > 0 else -1.0
    count = int(math.ceil(abs(span) / opts.step_size)) + 1
    grid = [t[0] + i * direction * opts.step_size for i in range(count)]
    grid[-1] = t[-1]
    return grid


def _interp_fit_linear(y0: Tensor, y1: Tensor) -> list[Tensor]:
    """Coefficients of the straight line through a step's two endpoints."""
    return [_fused.lincomb([y1, y0], [1.0, -1.0]), y0]


def _interp_fit_cubic(y0: Tensor, y1: Tensor, f0: Tensor, f1: Tensor, dt: float) -> list[Tensor]:
    r"""Coefficients of the cubic Hermite through a step's endpoints and slopes.

    Parameters
    ----------
    y0, y1 : Tensor
        States at the two ends of the step.
    f0, f1 : Tensor
        Derivatives at the two ends.
    dt : float
        Step size.

    Returns
    -------
    list of Tensor
        Four coefficients in descending powers of the normalised time.

    Notes
    -----
    On :math:`x = (t - t_0) / \Delta t` the cubic
    :math:`p(x) = ax^3 + bx^2 + cx + d` is pinned by :math:`p(0) = y_0`,
    :math:`p(1) = y_1`, :math:`p'(0) = \Delta t f_0`,
    :math:`p'(1) = \Delta t f_1`; solving gives the coefficients below.
    """
    y0, y1, f0, f1 = _fused._promote([y0, y1, f0, f1])
    return [
        _fused.lincomb([f0, f1, y1, y0], [dt, dt, -2.0, 2.0]),
        _fused.lincomb([y1, y0, f0, f1], [3.0, -3.0, -2.0 * dt, -dt]),
        _fused.lincomb([f0], [dt]),
        y0,
    ]


def _interp_evaluate(coeffs: Sequence[Tensor], t0: float, t1: float, t: float) -> Tensor:
    """Evaluate a fitted step polynomial at a time inside the step."""
    return _fused.poly_eval(coeffs, t0, t1, t)


def integrate(
    func: RightHandSide,
    y0: Tensor,
    grid: list[float],
    tableau: ButcherTableau,
    scalar: ScalarFactory,
    check: StageCheck,
    *,
    options: dict[str, object] | None,
    return_trajectory: bool,
) -> tuple[Tensor, list[Tensor]]:
    """Step once per interval of the integration grid.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y)``.
    y0 : Tensor
        Initial state, at ``grid[0]``.
    grid : list of float
        The output times.
    tableau : ButcherTableau
        Any explicit tableau; embedded error weights are ignored here.
    scalar : callable
        Builds the 0-D time tensor the right-hand side expects.
    check : callable
        Validates a right-hand-side result and returns it.
    options : dict or None
        See :class:`FixedOptions`.
    return_trajectory : bool
        Whether to collect the state at every output time.

    Returns
    -------
    tuple
        The final state, and the collected trajectory (empty when not
        requested).
    """
    opts = parse_options(options)
    steps = build_grid(opts, func, y0, grid)
    # The common case: the caller's times are the step boundaries, so every
    # output is a step endpoint and no interpolation is involved at all.
    direct = steps == grid

    y = y0
    trajectory: list[Tensor] = [y0] if return_trajectory else []
    needs_slope = opts.interp == "cubic" and not direct
    f0: Tensor | None = None
    if needs_slope:
        f0 = check(func(scalar(steps[0]), y), 0, 0)

    target_index = 1
    for step in range(len(steps) - 1):
        t0, t1 = steps[step], steps[step + 1]
        dt = t1 - t0

        ks: list[Tensor] = []
        for stage in range(tableau.stages):
            stage_t = t0 + tableau.c[stage] * dt
            if opts.perturb:
                # Nudge off the endpoints so a discontinuity sitting exactly
                # on a grid point is sampled from inside this step.
                if tableau.c[stage] == 0.0:
                    stage_t = math.nextafter(t0, t1)
                elif tableau.c[stage] == 1.0:
                    stage_t = math.nextafter(t1, t0)
            if stage == 0 and f0 is not None and not opts.perturb:
                ks.append(f0)
                continue
            stage_y = _fused.combine(y, ks, tableau.a[stage], dt)
            ks.append(check(func(scalar(stage_t), stage_y), step, stage))

        y_next = _fused.combine(y, ks, tableau.b, dt)

        if direct:
            if return_trajectory:
                trajectory.append(y_next)
            y = y_next
            continue

        if opts.interp == "cubic":
            assert f0 is not None
            f1 = check(func(scalar(t1), y_next), step, tableau.stages)
            coeffs = _interp_fit_cubic(y, y_next, f0, f1, dt)
            f0 = f1
        else:
            coeffs = _interp_fit_linear(y, y_next)

        while target_index < len(grid) and (
            (grid[target_index] - t1) * (1.0 if dt > 0 else -1.0) <= 0.0
        ):
            if return_trajectory:
                trajectory.append(_interp_evaluate(coeffs, t0, t1, grid[target_index]))
            target_index += 1
        y = y_next

    if not direct and return_trajectory and len(trajectory) < len(grid):
        # Rounding can leave the final output time a hair past the last step;
        # it is the endpoint either way.
        trajectory.append(y)
    return y, trajectory


def integrate_dense(
    func: RightHandSide,
    y0: Tensor,
    t0: float,
    t1: float,
    tableau: ButcherTableau,
    scalar: ScalarFactory,
    check: StageCheck,
    *,
    options: dict[str, object] | None,
) -> list[tuple[float, float, list[Tensor]]]:
    """Step across ``[t0, t1]`` on the configured grid, keeping every interpolant.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y)``.
    y0 : Tensor
        Initial state at ``t0``.
    t0, t1 : float
        Ends of the interval.  ``t1 < t0`` integrates backwards.
    tableau : ButcherTableau
        Any explicit tableau.
    scalar : callable
        Builds the 0-D time tensor the right-hand side expects.
    check : callable
        Validates a right-hand-side result and returns it.
    options : dict or None
        See :class:`FixedOptions`.

    Returns
    -------
    list of (float, float, list of Tensor)
        One entry per step: its start, its end, and its polynomial.

    Notes
    -----
    Without ``step_size`` or ``grid_constructor`` the grid is just
    ``[t0, t1]`` — a single step across the whole interval, which is
    mathematically fine but usually far too coarse to interpolate through.
    A fixed-step dense solve normally wants one of those options.
    """
    opts = parse_options(options)
    steps = build_grid(opts, func, y0, [t0, t1])

    y = y0
    f0: Tensor | None = None
    if opts.interp == "cubic":
        f0 = check(func(scalar(steps[0]), y), 0, 0)

    segments: list[tuple[float, float, list[Tensor]]] = []
    for index in range(len(steps) - 1):
        a, b = steps[index], steps[index + 1]
        dt = b - a

        ks: list[Tensor] = []
        for stage in range(tableau.stages):
            stage_t = a + tableau.c[stage] * dt
            if opts.perturb:
                if tableau.c[stage] == 0.0:
                    stage_t = math.nextafter(a, b)
                elif tableau.c[stage] == 1.0:
                    stage_t = math.nextafter(b, a)
            if stage == 0 and f0 is not None and not opts.perturb:
                ks.append(f0)
                continue
            stage_y = _fused.combine(y, ks, tableau.a[stage], dt)
            ks.append(check(func(scalar(stage_t), stage_y), index, stage))

        y_next = _fused.combine(y, ks, tableau.b, dt)
        if opts.interp == "cubic":
            assert f0 is not None
            f1 = check(func(scalar(b), y_next), index, tableau.stages)
            segments.append((a, b, _interp_fit_cubic(y, y_next, f0, f1, dt)))
            f0 = f1
        else:
            segments.append((a, b, _interp_fit_linear(y, y_next)))
        y = y_next

    return segments
