"""Implicit Runge-Kutta integration.

An explicit method computes its stages in order, each from the ones already
known.  An implicit one cannot: its stage equations reference themselves,

.. math::

    K_i = f\\!\\left(t + c_i h,\\; y + h \\textstyle\\sum_j a_{ij} K_j\\right)

with the sum running over *all* stages.  So every step solves a nonlinear
system.  That is far more expensive per step than an explicit method — and it
is worth it exactly when stability, not accuracy, is what limits the step
size.  On a stiff problem an explicit method is forced into steps far smaller
than the solution's own timescale; an implicit one is not, and wins on total
cost despite losing badly per step.

The system is solved with Broyden's method: a quasi-Newton iteration that
builds up an approximate Jacobian from the residuals it has already seen,
rather than computing a real one.  That keeps the solver from having to
differentiate the right-hand side — which would mean running nested autograd
inside a solve that may itself already be under grad — and matches what the
reference implementation does.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Callable

import lucid
import lucid.linalg as linalg
from lucid._dtype import finfo
from lucid._tensor.tensor import Tensor
from lucid.diffeq import _fixed, _fused
from lucid.diffeq._tableau import ButcherTableau
from lucid.diffeq._typing import RightHandSide, ScalarFactory, StageCheck

__all__: list[str] = []


_DEFAULT_MAX_ITERS = 100

# Residual norm below which a stage counts as solved, per precision.
#
# The float64 entry is deliberately tighter than the reference's 1e-8.  A
# stage residual of eps leaves an error of about dt*eps in the step, which put
# a floor near 1e-9 on the result -- above what gl6 and radauIIA5 achieve in
# their own right, so the two highest-order methods were being held to roughly
# fourth-order accuracy no matter the step size.  Implementing them and then
# capping them there defeats the point of having them.
#
# These are targets, not promises: an absolute residual bound is not always
# reachable, because the residual's own noise floor scales with the problem
# rather than with the number written here.  What guarantees termination is
# the step test in :func:`_broyden`, not this table.
_TOL_BY_DTYPE = {"float64": 1e-12, "float32": 1e-6, "float16": 1e-3}
_FALLBACK_TOL = 1e-6

# Multiple of the state's machine epsilon at which a Broyden step stops being
# able to move the iterate.  Adding a step of eps*||x|| to x changes x by
# about one unit in the last place, so a step below that is arithmetic noise
# and iterating on it cannot improve anything.  The margin covers the residual
# noise the step is computed from, which is a small multiple of eps*||x||
# rather than eps*||x|| exactly.
_STEP_FLOOR_FACTOR = 16.0

_IMPLICIT_KEYS = frozenset({"max_iters"})


@dataclass(frozen=True)
class ImplicitOptions:
    """Parsed ``options`` for an implicit method.

    Attributes
    ----------
    max_iters : int
        Ceiling on Broyden iterations per stage solve.
    fixed : _fixed.FixedOptions
        The grid settings every fixed-step method shares.
    """

    max_iters: int = _DEFAULT_MAX_ITERS
    fixed: _fixed.FixedOptions = _fixed.FixedOptions()


def parse_options(options: dict[str, object] | None) -> ImplicitOptions:
    """Validate and convert an ``options`` mapping for an implicit method.

    Parameters
    ----------
    options : dict or None
        Caller-supplied options.  Accepts ``max_iters`` plus every fixed-grid
        key, since these are fixed-step methods.

    Returns
    -------
    ImplicitOptions
        The parsed options.

    Raises
    ------
    ValueError
        If a key is unrecognised or a value out of range.
    """
    if options is None:
        return ImplicitOptions()

    unknown = set(options) - _IMPLICIT_KEYS - _fixed._FIXED_KEYS
    if unknown:
        allowed = sorted(_IMPLICIT_KEYS | _fixed._FIXED_KEYS)
        raise ValueError(
            f"unknown option(s) for an implicit method: {sorted(unknown)}; "
            f"expected a subset of {allowed}"
        )

    raw = options.get("max_iters", _DEFAULT_MAX_ITERS)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"option 'max_iters' must be an int, got {type(raw).__name__}")
    if raw < 1:
        raise ValueError(f"max_iters must be >= 1, got {raw}")

    fixed_part = {k: v for k, v in options.items() if k not in _IMPLICIT_KEYS}
    return ImplicitOptions(max_iters=raw, fixed=_fixed.parse_options(fixed_part or None))


def _tolerance(y: Tensor) -> float:
    """Residual norm that counts as converged, given the state's precision."""
    return _TOL_BY_DTYPE.get(str(y.dtype).rsplit(".", 1)[-1], _FALLBACK_TOL)


def _broyden(
    residual: Callable[[Tensor], Tensor],
    guess: Tensor,
    max_iters: int,
    tol: float,
    where: str,
) -> Tensor:
    """Solve ``residual(x) == 0`` by Broyden's method, starting from ``guess``.

    Parameters
    ----------
    residual : callable
        The function to drive to zero.  Takes and returns a flat vector.
    guess : Tensor
        Starting point; a good one costs iterations, not correctness.
    max_iters : int
        Ceiling on iterations.
    tol : float
        Converged once the residual's Euclidean norm falls below this.
    where : str
        Human-readable location, used only in the non-convergence warning.

    Returns
    -------
    Tensor
        The best point found.

    Notes
    -----
    Broyden approximates the Jacobian instead of computing one, updating it by
    the rank-one correction that makes it consistent with the step just taken:

    .. math::

        J \\leftarrow J + \\frac{(\\Delta F - J \\Delta x)\\,\\Delta x^{T}}
                                {\\Delta x^{T} \\Delta x}

    It starts from the identity, which is a poor Jacobian but a safe one — the
    first step is then simply ``-F``, and the approximation improves from
    there.

    The iteration stops for one of two good reasons and one bad one.  It has
    converged when the residual meets ``tol``.  It has finished when the step
    falls below ``eps * (1 + ||x||)`` — a step that small cannot change the
    iterate at the state's precision, so further passes would burn iterations
    to return the same answer.  That second test is what makes an unreachable
    ``tol`` harmless: an absolute residual bound assumes the residual can be
    driven that low, and on a stiff problem in float32 it often cannot, since
    the residual's own noise floor rises with the stiffness.  Judging the step
    instead measures against the state's magnitude, which is where the
    precision limit actually lives, and needs no tuned patience count.

    A step test alone would not do, and neither would a stagnation test: the
    approximate Jacobian starts as the identity, so the first few passes make
    little progress before the approximation becomes useful, and reading that
    warm-up as failure makes the iteration abandon a step it was about to
    solve.  A short step is immune to that confusion in a way a flat residual
    is not — while the Jacobian is still the identity the step *is* the
    residual, so a warm-up pass is never a short one.

    Two further ways out, both returning the last trustworthy iterate rather
    than a wrong answer dressed up as a right one: a singular approximate
    Jacobian stops the iteration immediately, and exhausting ``max_iters``
    warns.  Silence in either case would mean a step that looks successful and
    is not.

    Each iteration needs several scalars on the host -- the residual norm, the
    solve's status, and the step and state norms -- and asking for them one at
    a time is what an implicit solve on the GPU actually spends its time on:
    MLX evaluates lazily, so every read waits on the whole queued pipeline.
    :func:`lucid.diffeq._fused.broyden_probe` returns them all from one pass,
    leaving a single synchronisation per iteration.
    """
    x = guess
    f = residual(x)
    size = int(f.shape[0])
    jacobian = lucid.eye(size, dtype=f.dtype, device=f.device)
    # Seeded once so the first pass has something to test; every later value
    # comes from the probe below, which measures the residual it has just
    # produced.  That is what lets one probe answer both this iteration's
    # step questions and the next one's convergence question.
    residual_sq, _, _, _ = _fused.broyden_probe(f, f, x)
    # Keyed on the state rather than on the promoted probe dtype: what limits
    # the iteration is the precision `x + step` is accumulated at.
    floor = _STEP_FLOOR_FACTOR * finfo(x.dtype).eps

    for _ in range(max_iters):
        if residual_sq <= tol * tol:
            return x

        column, info = linalg.solve_ex(jacobian, (-f).reshape(size, 1))
        step = column.reshape(size)
        x_next = x + step
        f_next = residual(x_next)

        residual_sq, step_sq, state_sq, singular = _fused.broyden_probe(
            f_next, column, x_next, info
        )

        # A singular approximation carries no information about where to go
        # next, and a zero step means the iteration has stalled; either way
        # the last trustworthy iterate is the one before this step.  The
        # residual just computed is discarded along with it -- evaluating it
        # before knowing the step was usable costs nothing on the common path
        # and only this one wasted call on a path that is already ending.
        if singular != 0.0 or step_sq == 0.0:
            return x

        # Below the floor the step is smaller than the rounding of the
        # addition that applied it, so `x_next` is as far as this precision
        # goes.  It is the better of the two iterates, so it is the one kept.
        limit = floor * (1.0 + math.sqrt(state_sq))
        if step_sq <= limit * limit:
            return x_next

        change = f_next - f
        correction = change - (jacobian @ column).reshape(size)
        outer = correction.reshape(size, 1) * step.reshape(1, size)
        jacobian = jacobian + outer / step_sq
        x, f = x_next, f_next

    if residual_sq > tol * tol:
        warnings.warn(
            f"implicit solve did not converge at {where} after {max_iters} "
            f"iterations; the step was taken with the best iterate found",
            RuntimeWarning,
            stacklevel=2,
        )
    return x


def _step(
    func: RightHandSide,
    y: Tensor,
    t0: float,
    dt: float,
    tableau: ButcherTableau,
    scalar: ScalarFactory,
    check: StageCheck,
    step_index: int,
    opts: ImplicitOptions,
) -> Tensor:
    """Advance one step, solving the tableau's stage equations.

    Notes
    -----
    Diagonally implicit tableaux are solved stage by stage; the rest are
    solved as one coupled system.  The distinction is worth making because
    Broyden carries a dense Jacobian: ``s`` solves over ``n`` unknowns need
    ``n**2`` entries, one solve over ``s*n`` unknowns needs ``s**2`` times
    that.
    """
    shape = y.shape
    width = math.prod(shape)
    tol = _tolerance(y)
    stage_t = [t0 + tableau.c[i] * dt for i in range(tableau.stages)]

    if tableau.is_dirk:
        ks: list[Tensor] = []
        for i in range(tableau.stages):
            diagonal = tableau.a[i][i]
            known = _fused.combine(y, ks, tableau.a[i][:i], dt)
            if diagonal == 0.0:
                # An explicit row inside an otherwise implicit tableau; there
                # is nothing to solve.
                ks.append(check(func(scalar(stage_t[i]), known), step_index, i))
                continue

            def residual(flat: Tensor, i: int = i, known: Tensor = known) -> Tensor:
                stage_y = known + (dt * tableau.a[i][i]) * flat.reshape(shape)
                value = check(func(scalar(stage_t[i]), stage_y), step_index, i)
                return flat - value.reshape(width)

            seed = check(func(scalar(stage_t[i]), known), step_index, i)
            solved = _broyden(
                residual,
                seed.reshape(width),
                opts.max_iters,
                tol,
                f"t={t0!r}, stage {i}",
            )
            ks.append(solved.reshape(shape))
        return _fused.combine(y, ks, tableau.b, dt)

    # Fully implicit: every stage sees every other, so they solve together.
    stages = tableau.stages

    def coupled(flat: Tensor) -> Tensor:
        rows = [flat[i * width : (i + 1) * width].reshape(shape) for i in range(stages)]
        out: list[Tensor] = []
        for i in range(stages):
            stage_y = _fused.combine(y, rows, tableau.a[i], dt)
            value = check(func(scalar(stage_t[i]), stage_y), step_index, i)
            out.append(rows[i].reshape(width) - value.reshape(width))
        return lucid.concat(out, dim=0)

    initial = check(func(scalar(t0), y), step_index, 0).reshape(width)
    solved = _broyden(
        coupled,
        lucid.concat([initial] * stages, dim=0),
        opts.max_iters,
        tol,
        f"t={t0!r}",
    )
    ks = [solved[i * width : (i + 1) * width].reshape(shape) for i in range(stages)]
    return _fused.combine(y, ks, tableau.b, dt)


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
    """Integrate on a fixed grid with an implicit method.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y)``.
    y0 : Tensor
        Initial state, at ``grid[0]``.
    grid : list of float
        Output times; also the integration grid unless ``options`` decouples
        them.
    tableau : ButcherTableau
        An implicit tableau.
    scalar, check : callable
        Time-tensor factory and right-hand-side validator.
    options : dict or None
        See :class:`ImplicitOptions`.
    return_trajectory : bool
        Whether to collect the state at every output time.

    Returns
    -------
    tuple
        The final state, and the collected trajectory (empty when not
        requested).
    """
    opts = parse_options(options)
    steps = _fixed.build_grid(opts.fixed, func, y0, grid)
    direct = steps == grid

    y = y0
    trajectory: list[Tensor] = [y0] if return_trajectory else []
    target_index = 1

    for index in range(len(steps) - 1):
        t0, t1 = steps[index], steps[index + 1]
        dt = t1 - t0
        eval_t0 = math.nextafter(t0, t1) if opts.fixed.perturb else t0
        y_next = _step(func, y, eval_t0, dt, tableau, scalar, check, index, opts)

        if direct:
            if return_trajectory:
                trajectory.append(y_next)
            y = y_next
            continue

        if opts.fixed.interp == "cubic":
            f0 = check(func(scalar(t0), y), index, 0)
            f1 = check(func(scalar(t1), y_next), index, tableau.stages)
            coeffs = _fixed._interp_fit_cubic(y, y_next, f0, f1, dt)
        else:
            coeffs = _fixed._interp_fit_linear(y, y_next)

        while target_index < len(grid) and (
            (grid[target_index] - t1) * (1.0 if dt > 0 else -1.0) <= 0.0
        ):
            if return_trajectory:
                trajectory.append(_fixed._interp_evaluate(coeffs, t0, t1, grid[target_index]))
            target_index += 1
        y = y_next

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
        An implicit tableau.
    scalar, check : callable
        Time-tensor factory and right-hand-side validator.
    options : dict or None
        See :class:`ImplicitOptions`.

    Returns
    -------
    list of (float, float, list of Tensor)
        One entry per step: its start, its end, and its polynomial.

    Notes
    -----
    Implicit tableaux carry no interpolant of their own, so the continuous
    solution is a Hermite fit through each step's endpoints — the same
    construction the explicit fixed-step methods use, and subject to the same
    ceiling: ``interp="linear"`` caps the interpolation at second order no
    matter how accurate the steps themselves are.

    Without ``step_size`` or ``grid_constructor`` the grid is just
    ``[t0, t1]``, a single step across the whole interval — usually far too
    coarse to interpolate through.
    """
    opts = parse_options(options)
    steps = _fixed.build_grid(opts.fixed, func, y0, [t0, t1])

    y = y0
    f0: Tensor | None = None
    if opts.fixed.interp == "cubic":
        f0 = check(func(scalar(steps[0]), y), 0, 0)

    segments: list[tuple[float, float, list[Tensor]]] = []
    for index in range(len(steps) - 1):
        a, b = steps[index], steps[index + 1]
        dt = b - a
        eval_a = math.nextafter(a, b) if opts.fixed.perturb else a
        y_next = _step(func, y, eval_a, dt, tableau, scalar, check, index, opts)

        if opts.fixed.interp == "cubic":
            assert f0 is not None
            f1 = check(func(scalar(b), y_next), index, tableau.stages)
            segments.append((a, b, _fixed._interp_fit_cubic(y, y_next, f0, f1, dt)))
            f0 = f1
        else:
            segments.append((a, b, _fixed._interp_fit_linear(y, y_next)))
        y = y_next

    return segments
