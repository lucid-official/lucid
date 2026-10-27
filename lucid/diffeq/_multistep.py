"""Adams multistep integration.

A Runge-Kutta method buys accuracy by evaluating the right-hand side several
times inside one step.  A multistep method buys it by remembering the
derivatives it already computed: an order-``k`` Adams step reuses the last
``k`` values and costs a single new evaluation.  When the right-hand side is
expensive — a neural network, say — that trade is worth a lot.

The price is that it is not a one-step method.  There is no history at the
start, so the first few steps run on Runge-Kutta until enough has accumulated,
and the whole scheme is tied to a fixed grid: the coefficients below assume
evenly spaced past values.

Coefficients are derived rather than tabulated.  Adams weights are fixed by
requiring the interpolating polynomial through the remembered points to
integrate exactly, so they follow from that condition alone — computing them
in exact rationals removes any chance of a transcription slip in what would
otherwise be a wall of integer tables.
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from lucid._tensor.tensor import Tensor
from lucid.diffeq import _fixed, _fused
from lucid.diffeq._tableau import RK4, ButcherTableau
from lucid.diffeq._typing import RightHandSide, ScalarFactory, StageCheck

__all__: list[str] = []


# Upstream's ceiling; beyond this the coefficients grow large enough that the
# scheme loses stability faster than it gains accuracy.
_MAX_ORDER_CAP = 12

# Order below which there is not enough history yet and Runge-Kutta runs.
_MIN_ADAMS_ORDER = 4

_DEFAULT_MAX_ORDER = 12
_DEFAULT_MAX_ITERS = 4

# Method names, mapped to whether they carry an implicit corrector.  Both
# ``implicit_adams`` and ``fixed_adams`` name the predictor-corrector scheme,
# matching the reference library's two aliases for one solver.
METHODS: dict[str, bool] = {
    "explicit_adams": False,
    "implicit_adams": True,
    "fixed_adams": True,
}


@dataclass(frozen=True)
class AdamsOptions:
    """Parsed ``options`` for an Adams method.

    Attributes
    ----------
    max_order : int
        Highest order to ramp up to, capped at 12.
    max_iters : int
        Corrector iterations per step; ignored by the explicit method.
    fixed : _fixed.FixedOptions
        The grid settings every fixed-step method shares.
    """

    max_order: int = _DEFAULT_MAX_ORDER
    max_iters: int = _DEFAULT_MAX_ITERS
    fixed: _fixed.FixedOptions = _fixed.FixedOptions()


_ADAMS_KEYS = frozenset({"max_order", "max_iters"})


def parse_options(options: dict[str, object] | None) -> AdamsOptions:
    """Validate and convert an ``options`` mapping for an Adams method.

    Parameters
    ----------
    options : dict or None
        Caller-supplied options.  Accepts the Adams-specific keys plus every
        fixed-grid key, since these are fixed-step methods.

    Returns
    -------
    AdamsOptions
        The parsed options.

    Raises
    ------
    ValueError
        If a key is unrecognised or a value out of range.
    """
    if options is None:
        return AdamsOptions()

    # Check the union up front.  Delegating an unknown key to the fixed-step
    # parser would report a key list that leaves out the Adams keys, which
    # reads as though max_order were rejected.
    unknown = set(options) - _ADAMS_KEYS - _fixed._FIXED_KEYS
    if unknown:
        allowed = sorted(_ADAMS_KEYS | _fixed._FIXED_KEYS)
        raise ValueError(
            f"unknown option(s) for an Adams method: {sorted(unknown)}; "
            f"expected a subset of {allowed}"
        )

    adams_part = {k: v for k, v in options.items() if k in _ADAMS_KEYS}
    fixed_part = {k: v for k, v in options.items() if k not in _ADAMS_KEYS}
    fixed = _fixed.parse_options(fixed_part or None)

    def _count(key: str, default: int) -> int:
        raw = adams_part.get(key, default)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(f"option {key!r} must be an int, got {type(raw).__name__}")
        return raw

    max_order = _count("max_order", _DEFAULT_MAX_ORDER)
    if not 1 <= max_order <= _MAX_ORDER_CAP:
        raise ValueError(f"max_order must lie in [1, {_MAX_ORDER_CAP}], got {max_order}")
    max_iters = _count("max_iters", _DEFAULT_MAX_ITERS)
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1, got {max_iters}")

    return AdamsOptions(max_order=max_order, max_iters=max_iters, fixed=fixed)


def _polymul(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    """Multiply two polynomials given as ascending coefficient lists."""
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


@lru_cache(maxsize=None)
def coefficients(order: int, implicit: bool) -> tuple[float, ...]:
    r"""Adams weights for one step of the given order.

    Parameters
    ----------
    order : int
        Number of remembered derivatives the step combines.
    implicit : bool
        ``True`` for Adams-Moulton (the newest node is the step's end),
        ``False`` for Adams-Bashforth (the newest node is its start).

    Returns
    -------
    tuple of float
        Weights :math:`b_j` for ``y_{n+1} = y_n + h \sum_j b_j f_j``, newest
        derivative first.

    Notes
    -----
    Derived, not tabulated.  The weights are the integrals over one step of
    the Lagrange basis through the remembered nodes, so they follow from
    exactness on polynomials alone.  Computed in exact rationals and
    converted once, which is why there is no integer table here to mistype.

    They always sum to ``1`` — the consistency condition, and a cheap check
    that the derivation stayed correct.

    Memoised: the weights depend on nothing but the arguments, and the exact
    arithmetic is slow enough that re-deriving them every step dominated the
    solve outright.
    """
    nodes = [Fraction(1 - j) if implicit else Fraction(-j) for j in range(order)]
    weights: list[Fraction] = []
    for j in range(order):
        basis = [Fraction(1)]
        for i in range(order):
            if i == j:
                continue
            scale = nodes[j] - nodes[i]
            basis = _polymul(basis, [-nodes[i] / scale, Fraction(1) / scale])
        weights.append(sum((c / (power + 1) for power, c in enumerate(basis)), Fraction(0)))
    return tuple(float(w) for w in weights)


def _rk_step(
    func: RightHandSide,
    y: Tensor,
    t0: float,
    dt: float,
    scalar: ScalarFactory,
    check: StageCheck,
    step: int,
    f0: Tensor,
    tableau: ButcherTableau = RK4,
) -> Tensor:
    """Take one Runge-Kutta step, reusing the derivative already computed."""
    ks: list[Tensor] = [f0]
    for stage in range(1, tableau.stages):
        stage_y = _fused.combine(y, ks, tableau.a[stage], dt)
        ks.append(check(func(scalar(t0 + tableau.c[stage] * dt), stage_y), step, stage))
    return _fused.combine(y, ks, tableau.b, dt)


def integrate(
    func: RightHandSide,
    y0: Tensor,
    grid: list[float],
    implicit: bool,
    scalar: ScalarFactory,
    check: StageCheck,
    *,
    rtol: float,
    atol: float,
    options: dict[str, object] | None,
    return_trajectory: bool,
) -> tuple[Tensor, list[Tensor]]:
    """Integrate on a fixed grid, reusing remembered derivatives.

    Parameters
    ----------
    func : callable
        Right-hand side ``f(t, y)``.
    y0 : Tensor
        Initial state, at ``grid[0]``.
    grid : list of float
        Output times; also the integration grid unless ``options`` decouples
        them.
    implicit : bool
        Run the Adams-Moulton corrector after the Bashforth predictor.
    scalar, check : callable
        Time-tensor factory and right-hand-side validator.
    rtol, atol : float
        Used only to decide when the corrector has converged.
    options : dict or None
        See :class:`AdamsOptions`.
    return_trajectory : bool
        Whether to collect the state at every output time.

    Returns
    -------
    tuple
        The final state, and the collected trajectory (empty when not
        requested).

    Notes
    -----
    Order ramps up as history accumulates: the first steps run RK4 because
    an Adams step of order below four has nothing to gain over it, and the
    order then rises to whatever the history and ``max_order`` allow.

    Uneven grid spacing silently breaks the coefficients, which assume
    evenly spaced past values — so a non-uniform ``t`` falls back to RK4
    throughout rather than producing a quietly wrong answer.
    """
    opts = parse_options(options)
    steps = _fixed.build_grid(opts.fixed, func, y0, grid)
    direct = steps == grid

    spacings = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
    uniform = all(abs(h - spacings[0]) <= 1e-12 * max(1.0, abs(spacings[0])) for h in spacings)

    y = y0
    trajectory: list[Tensor] = [y0] if return_trajectory else []
    history: list[Tensor] = []
    target_index = 1

    for index, dt in enumerate(spacings):
        t0, t1 = steps[index], steps[index] + dt
        # Nudge off the endpoint so a discontinuity sitting exactly on a grid
        # point is sampled from inside this step, as the one-step solvers do.
        eval_t = math.nextafter(t0, t1) if opts.fixed.perturb else t0
        f0 = check(func(scalar(eval_t), y), index, 0)
        history.insert(0, f0)
        del history[opts.max_order :]

        order = min(len(history), opts.max_order)
        if not uniform or order < _MIN_ADAMS_ORDER:
            y_next = _rk_step(func, y, t0, dt, scalar, check, index, f0)
        else:
            predictor = coefficients(order, implicit=False)
            y_next = _fused.combine(y, history[:order], predictor, dt)
            if implicit:
                y_next = _correct(
                    func,
                    y,
                    y_next,
                    history,
                    t1,
                    dt,
                    order,
                    opts,
                    scalar,
                    check,
                    index,
                    rtol,
                    atol,
                )

        if direct:
            if return_trajectory:
                trajectory.append(y_next)
            y = y_next
            continue

        if opts.fixed.interp == "cubic":
            # f0 is the derivative this step already had to compute, so the
            # Hermite fit costs one extra evaluation rather than two.
            f1 = check(func(scalar(t1), y_next), index, order + opts.max_iters)
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


def _correct(
    func: RightHandSide,
    y: Tensor,
    predicted: Tensor,
    history: list[Tensor],
    t_next: float,
    dt: float,
    order: int,
    opts: AdamsOptions,
    scalar: ScalarFactory,
    check: StageCheck,
    step: int,
    rtol: float,
    atol: float,
) -> Tensor:
    """Refine a predictor with Adams-Moulton until it stops moving.

    Notes
    -----
    Convergence is measured with the same fused norm the adaptive controller
    uses, so "the correction is inside tolerance" means the same thing here
    as it does there.
    """
    corrector = coefficients(order, implicit=True)
    current = predicted
    for iteration in range(opts.max_iters):
        f_next = check(func(scalar(t_next), current), step, order + iteration)
        updated = _fused.combine(y, [f_next, *history[: order - 1]], corrector, dt)
        moved = _fused.error_ratio(
            current, updated, [updated, current], [1.0, -1.0], 1.0, rtol, atol
        )
        current = updated
        if moved <= 1.0:
            break
    return current
