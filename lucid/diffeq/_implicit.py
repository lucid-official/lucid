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

The system is solved with a damped, simplified Newton iteration.  Its Jacobian
is a real one -- reverse-mode differentiation of the stage residual -- factorised
once and reused while the iteration keeps contracting, then re-taken when it
stalls.  A quasi-Newton iteration that *estimates* the Jacobian from residuals
(the reference implementation's Broyden) starts from the identity and has no
way to know a stiff problem's Jacobian is thousands of times larger; on
Robertson's kinetics or a stiff Van der Pol oscillator it takes steps that are
wrong by orders of magnitude and then settles for a "best iterate".  A solve
that cannot converge raises :class:`ImplicitSolveError` instead: these are
fixed-step methods, so there is no smaller step to retry with, and an iterate
that does not satisfy the stage equations is not a step.
"""

import math
from dataclasses import dataclass
from typing import Callable

import lucid
import lucid.linalg as linalg
from lucid._dtype import finfo
from lucid._tensor.tensor import Tensor
from lucid.autograd import jacobian
from lucid.diffeq import _fixed, _fused
from lucid.diffeq._tableau import ButcherTableau
from lucid.diffeq._typing import RightHandSide, ScalarFactory, StageCheck

__all__: list[str] = ["ImplicitSolveError"]


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
# the step test in :class:`_Newton`, not this table.
_TOL_BY_DTYPE = {"float64": 1e-12, "float32": 1e-6, "float16": 1e-3}
_FALLBACK_TOL = 1e-6

# Multiple of the state's machine epsilon at which a Newton step stops being
# able to move the iterate.  Adding a step of eps*||x|| to x changes x by
# about one unit in the last place, so a step below that is arithmetic noise
# and iterating on it cannot improve anything.  The margin covers the residual
# noise the step is computed from, which is a small multiple of eps*||x||
# rather than eps*||x|| exactly.
_STEP_FLOOR_FACTOR = 16.0

# A pass that leaves more than this fraction of the residual means the held
# Jacobian has gone stale; it is re-taken at the current iterate.  One half is
# the usual contraction test for simplified Newton.
_REFRESH_RATIO = 0.1

# The damped step is accepted once it lowers the residual norm by this
# fraction of the step length (Armijo's condition), and halved at most this
# many times before the pass is declared a failure.
_SUFFICIENT_DECREASE = 1e-4
_MAX_HALVINGS = 20

# Newton steps differentiated at the root to attach a gradient to it.  Each
# settles one order of derivative; three give second-order differentiation
# through the solve (create_graph) as well as the first.
_POLISH_PASSES = 3

# Dtypes the linear algebra does not factorise natively.
_HALF_DTYPES = (lucid.float16, lucid.bfloat16)

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


class ImplicitSolveError(RuntimeError):
    """An implicit method's nonlinear stage solve failed.

    A fixed-step implicit method has no step to shrink, so a stage system the
    Newton iteration cannot solve leaves nothing honest to return: the
    iterate it stopped on does not satisfy the stage equations, and carrying
    on from it is how a stiff solve ends in ``NaN`` fifty steps later.  The
    solve stops here instead and says where.

    Parameters
    ----------
    method : str
        Name of the implicit method.
    t : float
        Start time of the failing step.
    stage : int or None
        Index of the stage whose system failed, or ``None`` when all stages
        are solved together.
    reason : str
        What went wrong.

    Attributes
    ----------
    method : str
        Name of the implicit method.
    t : float
        Start time of the failing step.
    stage : int or None
        Failing stage, or ``None`` for a coupled solve.

    Notes
    -----
    Taking a smaller step (``options={"step_size": ...}`` or a finer ``t``
    grid) makes the stage system easier, and a larger ``max_iters`` helps
    when the reason names the iteration budget.
    """

    def __init__(self, method: str, t: float, stage: int | None, reason: str) -> None:
        self.method = method
        self.t = t
        self.stage = stage
        where = "" if stage is None else f", stage {stage}"
        super().__init__(
            f"implicit method {method!r} failed to solve its stage equations at "
            f"t={t!r}{where}: {reason}. Use a smaller step (a finer t grid or "
            f"options['step_size']) so the stage system stays within reach of "
            f"the Newton iteration"
        )


@dataclass(frozen=True)
class _Site:
    """Where a stage solve is happening, for the error it may have to raise."""

    method: str
    t: float
    stage: int | None

    def failure(self, reason: str) -> ImplicitSolveError:
        return ImplicitSolveError(self.method, self.t, self.stage, reason)


@dataclass(frozen=True)
class _Iterate:
    """A point of the Newton iteration, with the numbers it is judged by."""

    x: Tensor
    f: Tensor
    residual_sq: float
    state_sq: float


@dataclass(frozen=True)
class _Move:
    """An accepted Newton step, and whether it reached the precision floor."""

    iterate: _Iterate
    settled: bool


class _Newton:
    """Damped, simplified Newton iteration for one stage system.

    Parameters
    ----------
    residual : callable
        The function to drive to zero.  Takes and returns a flat vector.
    max_iters : int
        Ceiling on Newton iterations.
    tol : float
        Converged once the residual's Euclidean norm falls below this.
    site : _Site
        Where the solve is happening, named in the error if it fails.

    Notes
    -----
    The Jacobian is a real one, taken by reverse-mode differentiation of the
    residual at the current iterate and factorised once.  It is held fixed
    for as long as the iteration keeps contracting (*simplified* Newton, as
    Radau implementations do) and re-taken at the current iterate when a pass
    contracts the residual by less than :data:`_REFRESH_RATIO` or finds no
    descent at all.  The Jacobian is differentiated at a detached copy of the
    iterate, so no second-order graph is recorded; the iterates themselves
    stay in the caller's graph, which is what carries gradients through the
    solve to whatever the right-hand side closes over.

    Each pass is damped: the Newton step is halved until it lowers the
    residual norm.  With a current Jacobian the step is a descent direction
    for the squared residual, so a short enough step always exists, and that
    is what stops a far-off start on a stiff problem from overshooting into
    a region the right-hand side cannot be evaluated in.

    The iteration finishes for one of three reasons.  The residual meets
    ``tol``.  Or the step falls below ``eps * (1 + ||x||)`` of the state's
    precision: a step that small cannot change the iterate, so the residual's
    own noise floor, not ``tol``, is what is stopping it -- on a stiff
    problem in float32 an absolute ``tol`` is often unreachable.  Or it
    cannot converge, and raises :class:`ImplicitSolveError`: a singular
    Jacobian, no damped step that lowers the residual from a fresh Jacobian,
    or the iteration budget spent.  Handing back the last iterate instead
    would pass off a step that does not satisfy its own equations as a
    finished one.

    Each trial needs several scalars on the host -- the residual norm, the
    step and state norms -- and asking for them one at a time is what an
    implicit solve on the GPU actually spends its time on: MLX evaluates
    lazily, so every read waits on the whole queued pipeline.
    :func:`lucid.diffeq._fused.broyden_probe` returns them all from one pass.
    """

    def __init__(
        self, residual: Callable[[Tensor], Tensor], max_iters: int, tol: float, site: _Site
    ) -> None:
        self._residual = residual
        self._max_iters = max_iters
        self._tol_sq = tol * tol
        self._site = site

    def solve(self, guess: Tensor) -> Tensor:
        """Return the root of the residual found from ``guess``.

        Raises
        ------
        ImplicitSolveError
            If the iteration cannot reach a root.

        Notes
        -----
        The iteration runs outside the autograd graph, and the gradient is
        attached afterwards by differentiating a few more Newton steps taken
        at the root with a Jacobian taken there.  They return the same value
        to within the residual, and because the root satisfies ``F == 0``
        whatever Jacobian is used, they carry the derivative of the root
        itself, :math:`-J^{-1}\\,\\partial F / \\partial\\theta`.

        Differentiating the iterations instead would be right only to the
        extent they had stopped moving: a step taken far from the root
        evaluates :math:`\\partial F / \\partial\\theta` there, not at the
        root, and the derivative lags the value by a pass.  It would also keep
        every trial's graph alive.  Each pass at the root settles one more
        order of derivative, so :data:`_POLISH_PASSES` of them make
        ``create_graph=True`` through the solve right to second order.  They
        are taken only when something upstream requires a gradient.
        """
        with lucid.no_grad():
            root = self._converge(guess)
        if not lucid.is_grad_enabled():
            return root
        f = self._residual(root)
        if not f.requires_grad:
            return root
        packed, pivots = self._factorise(root)
        x = root - self._newton_step(packed, pivots, f)
        for _ in range(_POLISH_PASSES - 1):
            x = x - self._newton_step(packed, pivots, self._residual(x))
        return x

    def _newton_step(self, packed: Tensor, pivots: Tensor, f: Tensor) -> Tensor:
        """``J^-1 f`` from the factors of ``J``, in ``f``'s dtype."""
        size = int(f.shape[0])
        rhs = f.reshape(size, 1)
        if rhs.dtype != packed.dtype:
            rhs = rhs.to(packed.dtype)
        step = linalg.lu_solve(packed, pivots, rhs).reshape(size)
        return step if step.dtype == f.dtype else step.to(f.dtype)

    def _converge(self, guess: Tensor) -> Tensor:
        """Newton iteration proper; see :meth:`solve`."""
        current = self._evaluate(guess)
        # Keyed on the state rather than on the promoted probe dtype: what
        # limits the iteration is the precision `x + step` is accumulated at.
        floor = _STEP_FLOOR_FACTOR * finfo(guess.dtype).eps
        factor: tuple[Tensor, Tensor] | None = None
        fresh = False

        for _ in range(self._max_iters):
            if current.residual_sq <= self._tol_sq:
                return current.x
            if factor is None:
                factor, fresh = self._factorise(current.x), True

            move = self._damped_step(factor, current, floor)
            if move is None:
                if fresh:
                    raise self._site.failure(
                        "no damped Newton step lowers the residual from a freshly "
                        "evaluated Jacobian"
                    )
                factor = None
                continue

            slow = move.iterate.residual_sq > _REFRESH_RATIO**2 * current.residual_sq
            if move.settled:
                return move.iterate.x
            current = move.iterate
            if slow:
                factor = None
            fresh = False

        if current.residual_sq <= self._tol_sq:
            return current.x
        raise self._site.failure(
            f"the residual is still {math.sqrt(current.residual_sq):.3e} after {self._max_iters} iterations"
        )

    def _evaluate(self, x: Tensor) -> _Iterate:
        f = self._residual(x)
        residual_sq, _, state_sq, _ = _fused.broyden_probe(f, f, x)
        return _Iterate(x, f, residual_sq, state_sq)

    def _factorise(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """LU factors of the residual's Jacobian at ``x``."""
        jac = jacobian(self._residual, x.detach())
        assert isinstance(jac, Tensor)
        if jac.dtype in _HALF_DTYPES:
            jac = jac.to(lucid.float32)
        packed, pivots, info = linalg.lu_factor_ex(jac)
        if int(info.item()) != 0:
            raise self._site.failure("the Jacobian of the stage equations is singular")
        return packed, pivots

    def _damped_step(
        self, factor: tuple[Tensor, Tensor], current: _Iterate, floor: float
    ) -> _Move | None:
        """Halve the Newton step until it lowers the residual; ``None`` if none does."""
        step = -self._newton_step(*factor, current.f)

        damping = 1.0
        for _ in range(_MAX_HALVINGS + 1):
            x = current.x + damping * step
            f = self._residual(x)
            residual_sq, step_sq, state_sq, _ = _fused.broyden_probe(f, step, x)
            limit = floor * (1.0 + math.sqrt(state_sq))
            # A Newton step below the floor is smaller than the rounding of
            # the addition that applies it, so `x` is as far as this precision
            # goes whether or not the residual fell.  It is the *undamped*
            # step that is judged: a damped one is short because it was
            # shortened, which says nothing about how close the root is.
            if step_sq <= limit * limit:
                return _Move(_Iterate(x, f, residual_sq, state_sq), settled=True)
            if residual_sq <= (1.0 - _SUFFICIENT_DECREASE * damping) ** 2 * current.residual_sq:
                return _Move(_Iterate(x, f, residual_sq, state_sq), settled=False)
            damping *= 0.5
        return None


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

    Raises
    ------
    ImplicitSolveError
        If a stage system cannot be solved.

    Notes
    -----
    Diagonally implicit tableaux are solved stage by stage; the rest are
    solved as one coupled system.  The distinction is worth making because
    the Newton iteration carries a dense Jacobian: ``s`` solves over ``n``
    unknowns need ``n**2`` entries, one solve over ``s*n`` unknowns needs
    ``s**2`` times that.
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

            newton = _Newton(residual, opts.max_iters, tol, _Site(tableau.name, t0, i))
            ks.append(
                newton.solve(lucid.zeros(width, dtype=y.dtype, device=y.device)).reshape(shape)
            )
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

    newton = _Newton(coupled, opts.max_iters, tol, _Site(tableau.name, t0, None))
    solved = newton.solve(lucid.zeros(stages * width, dtype=y.dtype, device=y.device))
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
