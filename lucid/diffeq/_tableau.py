"""Butcher tableaux for the Runge-Kutta family.

A Butcher tableau is the complete specification of a Runge-Kutta method: the
stage coefficient matrix :math:`a`, the combination weights :math:`b`, and the
stage times :math:`c`.  Splitting it out from the solver means a new method is
data, not code — the twenty-one built-ins below and any user-supplied tableau
run through exactly the same integration loop.

Both explicit and implicit methods are representable, and the shape of
:math:`a` is what distinguishes them.  Strictly lower triangular means stage
:math:`i` depends only on stages before it, so the stages evaluate in one
forward sweep.  Anything on or above the diagonal means a stage depends on
itself, and the step becomes a nonlinear solve — expensive per step, and the
reason those methods can take large ones on a stiff problem.

The implicit tableaux are not transcribed.  Seven of the nine are collocation
methods, so :mod:`lucid.diffeq._collocation` derives them from the Legendre
roots at sixty decimal digits and checks the result against the theoretical
order at import: a transcription slip in an irrational coefficient would
otherwise cost order silently, with no error to notice.
"""

import math
from dataclasses import dataclass
from functools import cached_property
from decimal import Decimal

from lucid.diffeq import _collocation

__all__ = [
    "ButcherTableau",
    "EULER",
    "MIDPOINT",
    "HEUN2",
    "HEUN3",
    "RK4",
    "RK4_CLASSIC",
    "ADAPTIVE_HEUN",
    "FEHLBERG2",
    "BOSH3",
    "DOPRI5",
    "DOPRI8",
    "TSIT5",
    "IMPLICIT_EULER",
    "IMPLICIT_MIDPOINT",
    "TRAPEZOID",
    "RADAU_IIA3",
    "RADAU_IIA5",
    "GL4",
    "GL6",
    "SDIRK2",
    "TRBDF2",
]


# Consistency conditions are exact for the built-in tableaux and hold to
# round-off for hand-entered ones, so the slack only has to absorb the error
# of summing a handful of floats.
_TOL = 1e-12


@dataclass(frozen=True)
class ButcherTableau:
    r"""Coefficient table defining one explicit Runge-Kutta method.

    Given a step from :math:`t` to :math:`t + \Delta t`, the method evaluates
    :math:`s` stage derivatives and combines them into the new state:

    .. math::

        k_i    &= f\!\left(t + c_i \Delta t,\;
                   y + \Delta t \textstyle\sum_{j<i} a_{ij} k_j\right) \\
        y_{n+1} &= y_n + \Delta t \textstyle\sum_i b_i k_i

    Parameters
    ----------
    a : tuple of tuple of float
        Stage coefficients, row by row.  Two shapes are accepted.  An
        *explicit* method is written in ragged form with the zeros above the
        diagonal omitted — row ``i`` holds exactly ``i`` entries, so ``a[0]``
        is empty.  An *implicit* method needs those entries, so it is written
        square: every row holds all ``s``.  Sequences of any kind are accepted
        and coerced to nested tuples.
    b : tuple of float
        Combination weights, one per stage.  Must sum to ``1``.
    c : tuple of float
        Stage times as fractions of the step.  Entry ``i`` must equal
        ``sum(a[i])``, the standard consistency condition.
    order : int
        Order of accuracy :math:`p`: halving the step size shrinks the global
        error by roughly :math:`2^p`.  Also the exponent the adaptive
        controller uses when proposing the next step size.
    name : str
        Human-readable method name, surfaced in error messages.
    b_error : tuple of float, optional
        Difference between the two embedded solutions' weights,
        :math:`b - \hat{b}`, one per stage.  Present exactly on the adaptive
        methods; its presence is what :attr:`is_adaptive` reports.  Must sum
        to ``0``, since both solutions are consistent.
    mid : tuple of float, optional
        Weights producing the state at the midpoint of a step, used to anchor
        the quartic interpolant that gives dense output.  Required whenever
        ``b_error`` is given.

    Raises
    ------
    ValueError
        If ``a`` / ``b`` / ``c`` disagree on stage count, if any row of ``a``
        has the wrong length for a strictly lower-triangular matrix, if ``b``
        does not sum to ``1``, if any ``c[i] != sum(a[i])``, or if ``order``
        or ``name`` are unset.

    Notes
    -----
    Six derived properties read the table rather than being stored on it, so
    they cannot fall out of step with the coefficients:

    * :attr:`stages` — how many derivatives a step evaluates, ``len(b)``.
    * :attr:`is_adaptive` — whether ``b_error`` is present, which is what
      sends a solve to the adaptive stepper instead of the fixed-grid one.
    * :attr:`is_implicit` — whether any stage depends on itself or on a later
      one, which is what forces a nonlinear solve per step.
    * :attr:`is_dirk` — whether the stage matrix is lower triangular including
      the diagonal, so the stages can be solved one at a time.
    * :attr:`is_fsal` — whether the last stage already evaluated the
      right-hand side at the new state, in which case the next step reuses it
      and the method costs one evaluation less than it has stages.
    * :attr:`mid_order` — the order to which ``mid`` reproduces the true
      midpoint.  This bounds every interpolated value and is *not* implied by
      ``order``: a method can take third-order steps and interpolate to first.

    The constructor validates the table on the way in — stage counts agree,
    ``b`` sums to one, each ``c[i]`` equals ``sum(a[i])``, and ``b_error``
    sums to zero — so an inconsistent tableau fails where it is written rather
    than as a silent loss of order several steps into a solve.

    Examples
    --------
    The classical fourth-order method, written out in full:

    >>> import lucid.diffeq as diffeq
    >>> rk4 = diffeq.ButcherTableau(
    ...     a=((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)),
    ...     b=(1 / 6, 1 / 3, 1 / 3, 1 / 6),
    ...     c=(0.0, 0.5, 0.5, 1.0),
    ...     order=4,
    ...     name="rk4",
    ... )
    >>> rk4.stages
    4

    See Also
    --------
    lucid.diffeq.odeint : Consumes a tableau, by name or by instance.
    """

    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    c: tuple[float, ...]
    order: int
    name: str
    b_error: tuple[float, ...] | None = None
    mid: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        # Coerce first so that a tableau written with lists — the natural way
        # to type one out — still ends up frozen, hashable, and float-typed.
        object.__setattr__(self, "a", tuple(tuple(float(v) for v in row) for row in self.a))
        object.__setattr__(self, "b", tuple(float(v) for v in self.b))
        object.__setattr__(self, "c", tuple(float(v) for v in self.c))
        if self.b_error is not None:
            object.__setattr__(self, "b_error", tuple(float(v) for v in self.b_error))
        if self.mid is not None:
            object.__setattr__(self, "mid", tuple(float(v) for v in self.mid))

        n = len(self.b)
        if n == 0:
            raise ValueError("ButcherTableau requires at least one stage")
        if len(self.a) != n or len(self.c) != n:
            raise ValueError(
                f"ButcherTableau stage counts disagree: "
                f"len(a)={len(self.a)}, len(b)={n}, len(c)={len(self.c)}"
            )
        # Ragged rows spell an explicit method, square rows an implicit one.
        # Mixing the two is always a typo, and left unchecked it would shift
        # every coefficient in the offending row against its stage.
        ragged = all(len(row) == i for i, row in enumerate(self.a))
        square = all(len(row) == n for row in self.a)
        if not ragged and not square:
            widths = [len(row) for row in self.a]
            raise ValueError(
                f"ButcherTableau rows of a must be either ragged (row i holds "
                f"i entries, an explicit method) or square (every row holds "
                f"{n}, an implicit method); got widths {widths}"
            )
        weight_sum = sum(self.b)
        if abs(weight_sum - 1.0) > _TOL:
            raise ValueError(f"ButcherTableau weights b must sum to 1, got {weight_sum!r}")
        for i, row in enumerate(self.a):
            row_sum = sum(row)
            if abs(self.c[i] - row_sum) > _TOL:
                raise ValueError(
                    f"ButcherTableau consistency violated at stage {i}: "
                    f"c[{i}]={self.c[i]!r} but sum(a[{i}])={row_sum!r}"
                )
        if self.order < 1:
            raise ValueError(f"ButcherTableau order must be >= 1, got {self.order}")
        if not self.name:
            raise ValueError("ButcherTableau name must be a non-empty string")

        if self.b_error is not None:
            if len(self.b_error) != n:
                raise ValueError(
                    f"ButcherTableau b_error must hold {n} entries, " f"got {len(self.b_error)}"
                )
            error_sum = sum(self.b_error)
            if abs(error_sum) > _TOL:
                raise ValueError(
                    f"ButcherTableau b_error must sum to 0 (both embedded "
                    f"solutions are consistent), got {error_sum!r}"
                )
            if self.mid is None:
                raise ValueError(
                    "ButcherTableau with b_error also needs mid, the midpoint "
                    "weights the dense-output interpolant is anchored on"
                )
        if self.mid is not None and len(self.mid) != n:
            raise ValueError(f"ButcherTableau mid must hold {n} entries, got {len(self.mid)}")

    @cached_property
    def is_adaptive(self) -> bool:
        """bool: Whether the tableau carries an embedded error estimate.

        Adaptive methods pair two solutions of different order; their weight
        difference (``b_error``) estimates the local error, which is what a
        step-size controller needs.  A tableau without it can only be stepped
        on a fixed grid.
        """
        return self.b_error is not None

    @cached_property
    def is_implicit(self) -> bool:
        """bool: Whether any stage depends on itself or on a later stage.

        An explicit method computes its stages in order, each from the ones
        before it.  An implicit one does not: the stage equations are coupled,
        so a step has to solve a nonlinear system instead of evaluating a
        recipe.  That is a different solver, not a different coefficient set,
        which is what this flag selects.
        """
        return any(
            coefficient != 0.0
            for i, row in enumerate(self.a)
            for j, coefficient in enumerate(row)
            if j >= i
        )

    @cached_property
    def is_dirk(self) -> bool:
        """bool: Whether the stage matrix is lower triangular, diagonal included.

        Diagonally implicit methods still couple each stage to itself, but not
        to any stage after it, so the stages can be solved one at a time.  For
        a state of ``n`` elements that means ``s`` solves of size ``n`` rather
        than one of size ``s*n`` — the difference between a dense Jacobian of
        ``n**2`` entries and one of ``(s*n)**2``.
        """
        if not self.is_implicit:
            return False
        return all(
            coefficient == 0.0
            for i, row in enumerate(self.a)
            for j, coefficient in enumerate(row)
            if j > i
        )

    @cached_property
    def is_fsal(self) -> bool:
        """bool: Whether the last stage derivative equals ``f(t + dt, y_next)``.

        True when the final stage row reproduces the solution weights, so the
        stage already evaluated the right-hand side at the new state.  That
        derivative can then be reused as the next step's first stage, saving
        one call per accepted step — a real saving when the right-hand side is
        a neural network.
        """
        return self.b[-1] == 0.0 and self.a[-1] == self.b[:-1]

    @cached_property
    def mid_order(self) -> int:
        r"""int: Order to which ``mid`` reproduces the state at the midpoint.

        Zero when the tableau carries no ``mid`` at all.

        The dense output is a quartic anchored on the midpoint estimate, so
        this bounds how accurate any interpolated value can be — and it is not
        implied by ``order``.  Several tableaux pair a genuinely derived
        interpolant with their steps (``dopri5`` and ``tsit5`` reach 4 here,
        ``dopri8`` reaches 5) while others carry only a first-order
        placeholder: a half-step of Euler, which is what ``mid`` reduces to
        when its weights put ``1/2`` in a single slot.  ``bosh3`` takes third
        order steps and interpolates to first, and nothing about the method's
        name says so.

        Computed from the quadrature conditions the midpoint weights would
        have to satisfy, the same way the collocation tableaux verify their
        own order at import:

        .. math::

            \sum_i \text{mid}_i\, c_i^{\,k-1} = \frac{(1/2)^k}{k}
        """
        if self.mid is None:
            return 0
        reached = 0
        for k in range(1, len(self.mid) + 2):
            lhs = sum(m * c ** (k - 1) for m, c in zip(self.mid, self.c, strict=True))
            rhs = 0.5**k / k
            if abs(lhs - rhs) > 1e-10 * max(1.0, abs(rhs)):
                break
            reached = k
        return reached

    @property
    def stages(self) -> int:
        """int: Number of stage derivatives evaluated per step.

        Equals the number of right-hand-side calls the method makes for each
        step, which is the dominant cost when the right-hand side is a neural
        network.
        """
        return len(self.b)


EULER = ButcherTableau(a=((),), b=(1.0,), c=(0.0,), order=1, name="euler")
"""ButcherTableau: Forward Euler — one stage, first order."""

MIDPOINT = ButcherTableau(a=((), (0.5,)), b=(0.0, 1.0), c=(0.0, 0.5), order=2, name="midpoint")
"""ButcherTableau: Explicit midpoint — two stages, second order."""

HEUN2 = ButcherTableau(a=((), (1.0,)), b=(0.5, 0.5), c=(0.0, 1.0), order=2, name="heun2")
"""ButcherTableau: Heun's method (explicit trapezoid) — two stages, second order."""

HEUN3 = ButcherTableau(
    a=((), (1.0 / 3.0,), (0.0, 2.0 / 3.0)),
    b=(0.25, 0.0, 0.75),
    c=(0.0, 1.0 / 3.0, 2.0 / 3.0),
    order=3,
    name="heun3",
)
"""ButcherTableau: Heun's third-order method — three stages, third order."""

RK4 = ButcherTableau(
    a=((), (1.0 / 3.0,), (-1.0 / 3.0, 1.0), (1.0, -1.0, 1.0)),
    b=(0.125, 0.375, 0.375, 0.125),
    c=(0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
    order=4,
    name="rk4",
)
"""ButcherTableau: Kutta's 3/8 rule — four stages, fourth order.

This is what ``"rk4"`` resolves to, because it is what the reference ODE
library's ``"rk4"`` resolves to and a method name is exposed API: code ported
across should produce the same numbers.  For the textbook method that the name
more often denotes elsewhere, ask for :data:`RK4_CLASSIC`.

Both are fourth order, so no convergence-order test can tell them apart — only
a direct value comparison can, and the two disagree at the fifth-order term
(about 2e-7 relative on a smooth problem at a step of 1/32).
"""

RK4_CLASSIC = ButcherTableau(
    a=((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)),
    b=(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
    c=(0.0, 0.5, 0.5, 1.0),
    order=4,
    name="rk4_classic",
)
"""ButcherTableau: The classical Runge-Kutta method — four stages, fourth order.

A Lucid extension: the reference ODE library has no method name for this
tableau, having given ``"rk4"`` to :data:`RK4` instead.  It is kept reachable
because this is the method almost every other source means by "RK4".
"""


# ── Adaptive (embedded) pairs ────────────────────────────────────────────────
#
# Each carries ``b_error`` (the weight difference between the two embedded
# solutions) and ``mid`` (midpoint weights anchoring the dense-output
# interpolant).  Coefficients are written as exact ratios so the source reads
# like the published tableau and no decimal is silently truncated.

ADAPTIVE_HEUN = ButcherTableau(
    a=((), (1.0,)),
    b=(0.5, 0.5),
    c=(0.0, 1.0),
    order=2,
    name="adaptive_heun",
    b_error=(0.5, -0.5),
    mid=(0.5, 0.0),
)
"""ButcherTableau: Heun-Euler 2(1) — two stages, adaptive, second order."""

FEHLBERG2 = ButcherTableau(
    a=((), (0.5,), (1.0 / 256.0, 255.0 / 256.0)),
    b=(1.0 / 512.0, 255.0 / 256.0, 1.0 / 512.0),
    c=(0.0, 0.5, 1.0),
    order=2,
    name="fehlberg2",
    b_error=(-1.0 / 512.0, 0.0, 1.0 / 512.0),
    mid=(0.0, 0.5, 0.0),
)
"""ButcherTableau: Fehlberg 2(1) — three stages, adaptive, second order."""

BOSH3 = ButcherTableau(
    a=((), (0.5,), (0.0, 0.75), (2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0)),
    b=(2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0),
    c=(0.0, 0.5, 0.75, 1.0),
    order=3,
    name="bosh3",
    b_error=(2.0 / 9.0 - 7.0 / 24.0, 1.0 / 12.0, 1.0 / 9.0, -0.125),
    mid=(0.0, 0.5, 0.0, 0.0),
)
"""ButcherTableau: Bogacki-Shampine 3(2) — four stages, adaptive, third order."""

DOPRI5 = ButcherTableau(
    a=(
        (),
        (1.0 / 5.0,),
        (3.0 / 40.0, 9.0 / 40.0),
        (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
        (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
        (
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ),
        (
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ),
    ),
    b=(
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ),
    c=(0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0),
    order=5,
    name="dopri5",
    b_error=(
        35.0 / 384.0 - 1951.0 / 21600.0,
        0.0,
        500.0 / 1113.0 - 22642.0 / 50085.0,
        125.0 / 192.0 - 451.0 / 720.0,
        -2187.0 / 6784.0 + 12231.0 / 42400.0,
        11.0 / 84.0 - 649.0 / 6300.0,
        -1.0 / 60.0,
    ),
    mid=(
        6025192743.0 / 60171106304.0,
        0.0,
        51252292925.0 / 130801643196.0,
        -2691868925.0 / 90256659456.0,
        187940372067.0 / 3189068634112.0,
        -1776094331.0 / 39487288512.0,
        11237099.0 / 470086768.0,
    ),
)
"""ButcherTableau: Dormand-Prince 5(4) — seven stages, adaptive, fifth order."""

TSIT5 = ButcherTableau(
    a=(
        (),
        (0.161,),
        (-0.008480655492356989, 0.335480655492356989),
        (2.897153057105493, -6.359448489975075, 4.362295432869581),
        (
            5.325864828439257,
            -11.748883564062828,
            7.495539342889836,
            -0.09249506636175525,
        ),
        (
            5.861455442946420,
            -12.920969317847109,
            8.159367898576159,
            -0.07158497328140100,
            -0.02826905039406838,
        ),
        (
            0.09646076681806523,
            0.01,
            0.4798896504144996,
            1.379008574103742,
            -3.290069515436081,
            2.324710524099774,
        ),
    ),
    # The fifth-order solution, which is the one that propagates.  These two
    # vectors were swapped: ``b`` held the fourth-order embedded solution and
    # the fifth-order weights sat in the last row of ``a``, so the method
    # advertised order 5, raised the error ratio to the power 1/5, and took
    # fourth-order steps.  Every one of the nine fifth-order conditions was
    # violated by the vector that was propagating; all seventeen hold for this
    # one.  ``dopri5`` above has the same shape done right — ``b`` is the
    # high-order solution and ``a[-1] == b[:-1]``, which is what makes the pair
    # FSAL.  Tsitouras is FSAL too, and it was not, for the same reason.
    b=(
        0.09646076681806523,
        0.01,
        0.4798896504144996,
        1.379008574103742,
        -3.290069515436081,
        2.324710524099774,
        0.0,
    ),
    c=(0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0),
    order=5,
    name="tsit5",
    # ``b - b_hat``, the documented direction: the fifth-order solution minus
    # the fourth.  Only its magnitude reaches the controller (the error norm
    # squares it), but the sign is what the field means, and ``dopri5`` writes
    # its own that way round.
    b_error=(
        0.001780011052225771,
        0.0008164344596567469,
        -0.007880878010261996,
        0.1447110071732629,
        -0.5823571654525552,
        0.4580821059291869,
        -1.0 / 66.0,
    ),
    mid=(
        0.10741235230100377,
        0.01135625,
        0.39560903056045305,
        -0.34475214352593553,
        1.3161853649581645,
        -1.0170608542936508,
        0.031249999999999993,
    ),
)
"""ButcherTableau: Tsitouras 5(4) — seven stages, adaptive, fifth order."""


# The dense-output weights of Dopri8 are not published as ratios like the rest
# of the table.  Each is a quintic in the half-step evaluated at ``h = 1/2``,
# so the polynomial is written out and evaluated here rather than the value
# being pasted in: the coefficients below can be read straight off the
# published interpolant, while the number they produce cannot be checked by
# eye.  ``_dopri8_mid`` is the reference's ``poly(h) / (1 / h)``, i.e.
# ``poly(h) * h``.
def _dopri8_mid(*coeffs: float) -> float:
    """Evaluate a dense-output quintic (highest power first) at ``h = 1/2``."""
    h = 0.5
    acc = 0.0
    for coeff in coeffs:
        acc = acc * h + coeff
    return acc * h


DOPRI8 = ButcherTableau(
    a=(
        (),
        (1 / 18,),
        (1 / 48, 1 / 16),
        (1 / 32, 0.0, 3 / 32),
        (5 / 16, 0.0, -75 / 64, 75 / 64),
        (3 / 80, 0.0, 0.0, 3 / 16, 3 / 20),
        (
            29443841 / 614563906,
            0.0,
            0.0,
            77736538 / 692538347,
            -28693883 / 1125000000,
            23124283 / 1800000000,
        ),
        (
            16016141 / 946692911,
            0.0,
            0.0,
            61564180 / 158732637,
            22789713 / 633445777,
            545815736 / 2771057229,
            -180193667 / 1043307555,
        ),
        (
            39632708 / 573591083,
            0.0,
            0.0,
            -433636366 / 683701615,
            -421739975 / 2616292301,
            100302831 / 723423059,
            790204164 / 839813087,
            800635310 / 3783071287,
        ),
        (
            246121993 / 1340847787,
            0.0,
            0.0,
            -37695042795 / 15268766246,
            -309121744 / 1061227803,
            -12992083 / 490766935,
            6005943493 / 2108947869,
            393006217 / 1396673457,
            123872331 / 1001029789,
        ),
        (
            -1028468189 / 846180014,
            0.0,
            0.0,
            8478235783 / 508512852,
            1311729495 / 1432422823,
            -10304129995 / 1701304382,
            -48777925059 / 3047939560,
            15336726248 / 1032824649,
            -45442868181 / 3398467696,
            3065993473 / 597172653,
        ),
        (
            185892177 / 718116043,
            0.0,
            0.0,
            -3185094517 / 667107341,
            -477755414 / 1098053517,
            -703635378 / 230739211,
            5731566787 / 1027545527,
            5232866602 / 850066563,
            -4093664535 / 808688257,
            3962137247 / 1805957418,
            65686358 / 487910083,
        ),
        (
            403863854 / 491063109,
            0.0,
            0.0,
            -5068492393 / 434740067,
            -411421997 / 543043805,
            652783627 / 914296604,
            11173962825 / 925320556,
            -13158990841 / 6184727034,
            3936647629 / 1978049680,
            -160528059 / 685178525,
            248638103 / 1413531060,
            0.0,
        ),
        (
            14005451 / 335480064,
            0.0,
            0.0,
            0.0,
            0.0,
            -59238493 / 1068277825,
            181606767 / 758867731,
            561292985 / 797845732,
            -1041891430 / 1371343529,
            760417239 / 1151165299,
            118820643 / 751138087,
            -528747749 / 2220607170,
            1 / 4,
        ),
    ),
    b=(
        14005451 / 335480064,
        0.0,
        0.0,
        0.0,
        0.0,
        -59238493 / 1068277825,
        181606767 / 758867731,
        561292985 / 797845732,
        -1041891430 / 1371343529,
        760417239 / 1151165299,
        118820643 / 751138087,
        -528747749 / 2220607170,
        1 / 4,
        0.0,
    ),
    c=(
        0.0,
        1 / 18,
        1 / 12,
        1 / 8,
        5 / 16,
        3 / 8,
        59 / 400,
        93 / 200,
        5490023248 / 9719169821,
        13 / 20,
        1201146811 / 1299019798,
        1.0,
        1.0,
        1.0,
    ),
    order=8,
    name="dopri8",
    b_error=(
        14005451 / 335480064 - 13451932 / 455176623,
        0.0,
        0.0,
        0.0,
        0.0,
        -59238493 / 1068277825 + 808719846 / 976000145,
        181606767 / 758867731 - 1757004468 / 5645159321,
        561292985 / 797845732 - 656045339 / 265891186,
        -1041891430 / 1371343529 + 3867574721 / 1518517206,
        760417239 / 1151165299 - 465885868 / 322736535,
        118820643 / 751138087 - 53011238 / 667516719,
        -528747749 / 2220607170 - 2 / 45,
        1 / 4,
        0.0,
    ),
    mid=(
        _dopri8_mid(
            -6.3448349392860401388,
            22.1396504998094068976,
            -30.0610568289666450593,
            19.9990069333683970610,
            -6.6910181737837595697,
            1.0,
        ),
        0.0,
        0.0,
        0.0,
        0.0,
        _dopri8_mid(
            -39.6107919852202505218,
            116.4422149550342161651,
            -121.4999627731334642623,
            52.2273532792945524050,
            -7.6142658045872677172,
            0.0,
        ),
        _dopri8_mid(
            20.3761213808791436958,
            -67.1451318825957197185,
            83.1721004639847717481,
            -46.8919164181093621583,
            10.7281392630428866124,
            0.0,
        ),
        _dopri8_mid(
            7.3347098826795362023,
            -16.5672243527496524646,
            9.5724507555993664382,
            -0.1890893225010595467,
            0.5526637063753648783,
            0.0,
        ),
        _dopri8_mid(
            32.8801774352459155182,
            -89.9916014847245016028,
            87.8406057677205645007,
            -35.7075975946222072821,
            4.2186562625665153803,
            0.0,
        ),
        _dopri8_mid(
            -10.1588990526426760954,
            22.6237489648532849093,
            -17.4152107770762969005,
            6.2736448083240352160,
            -0.6627209125361597559,
            0.0,
        ),
        _dopri8_mid(
            -12.5401268098782561200,
            32.2362340167355370113,
            -28.5903289514790976966,
            10.3160881272450748458,
            -1.2636789001135462218,
            0.0,
        ),
        _dopri8_mid(
            29.5553001484516038033,
            -82.1020315488359848644,
            81.6630950584341412934,
            -34.7650769866611817349,
            5.4106037898590422230,
            0.0,
        ),
        _dopri8_mid(
            -41.7923486424390588923,
            116.2662185791119533462,
            -114.9375291377009418170,
            47.7457971078225540396,
            -7.0321379067945741781,
            0.0,
        ),
        _dopri8_mid(
            20.3006925822100825485,
            -53.9020777466385396792,
            50.2558364226176017553,
            -19.0082099341608028453,
            2.3537586759714983486,
            0.0,
        ),
    ),
)
"""ButcherTableau: Dormand-Prince 8(5,3) — fourteen stages, adaptive, eighth
order.  Method name ``"dopri8"``.

The highest-order method here, and the one to reach for when the tolerance is
tight: its cost per step is roughly twice ``dopri5``'s, but the step count
falls off as ``tol ** (-1/8)`` rather than ``tol ** (-1/5)``, so it wins once
the accuracy demanded is high enough to make step count dominate.
"""


# Name → tableau lookup backing ``odeint(method=...)``.  Private because the
# canonical way to reach a method is its name string or the module-level
# constant; a second public registry would be a second path to the same thing.

# ---------------------------------------------------------------------------
# Implicit methods.
#
# These are ordinary Butcher tableaux whose stage matrix is not strictly lower
# triangular, so a step solves a nonlinear system rather than evaluating a
# recipe.  Seven of the nine are collocation methods, meaning the whole table
# follows from the nodes -- see ``_collocation`` for why they are derived here
# rather than written out, and what that buys.
# ---------------------------------------------------------------------------


def _collocation_tableau(nodes: list[Decimal], order: int, name: str) -> ButcherTableau:
    """Build a collocation tableau and check it reached the intended order.

    Raises
    ------
    ValueError
        If the derived weights do not integrate polynomials exactly to
        ``order``.  A misplaced node does not otherwise announce itself — it
        silently costs accuracy — so the derivation is verified at import,
        every time, rather than trusted.
    """
    a, b = _collocation.collocation_tableau(nodes)
    c = tuple(float(node) for node in nodes)
    reached = _collocation.quadrature_order(b, c)
    if reached != order:
        raise ValueError(f"derived tableau {name!r} reached order {reached}, expected {order}")
    return ButcherTableau(a=a, b=b, c=c, order=order, name=name)


IMPLICIT_EULER = _collocation_tableau(_collocation.radau_nodes(1), order=1, name="implicit_euler")

IMPLICIT_MIDPOINT = _collocation_tableau(
    _collocation.gauss_nodes(1), order=2, name="implicit_midpoint"
)

TRAPEZOID = _collocation_tableau(_collocation.lobatto_nodes(2), order=2, name="trapezoid")

RADAU_IIA3 = _collocation_tableau(_collocation.radau_nodes(2), order=3, name="radauIIA3")

RADAU_IIA5 = _collocation_tableau(_collocation.radau_nodes(3), order=5, name="radauIIA5")

GL4 = _collocation_tableau(_collocation.gauss_nodes(2), order=4, name="gl4")

GL6 = _collocation_tableau(_collocation.gauss_nodes(3), order=6, name="gl6")


# The last two are not collocation methods, so there is nothing to derive them
# from -- but each is pinned by a single constant, and the order check below
# is the same one the collocation tableaux get.
_SDIRK2_GAMMA = 1.0 - math.sqrt(2.0) / 2.0

SDIRK2 = ButcherTableau(
    a=(
        (_SDIRK2_GAMMA, 0.0),
        (1.0 - _SDIRK2_GAMMA, _SDIRK2_GAMMA),
    ),
    b=(1.0 - _SDIRK2_GAMMA, _SDIRK2_GAMMA),
    c=(_SDIRK2_GAMMA, 1.0),
    order=2,
    name="sdirk2",
)

# TR-BDF2: a trapezoidal step to t + gamma*dt, then a BDF2 step to t + dt,
# with gamma chosen so the two share a Jacobian.
_TRBDF2_GAMMA = 2.0 - math.sqrt(2.0)
_TRBDF2_W = math.sqrt(2.0) / 4.0

TRBDF2 = ButcherTableau(
    a=(
        (0.0, 0.0, 0.0),
        (_TRBDF2_GAMMA / 2.0, _TRBDF2_GAMMA / 2.0, 0.0),
        (_TRBDF2_W, _TRBDF2_W, 1.0 - math.sqrt(2.0) / 2.0),
    ),
    b=(_TRBDF2_W, _TRBDF2_W, 1.0 - math.sqrt(2.0) / 2.0),
    c=(0.0, _TRBDF2_GAMMA, 1.0),
    order=2,
    name="trbdf2",
)

for _tableau in (SDIRK2, TRBDF2):
    _reached = _collocation.quadrature_order(_tableau.b, _tableau.c)
    if _reached != _tableau.order:
        raise ValueError(
            f"tableau {_tableau.name!r} reached order {_reached}, " f"expected {_tableau.order}"
        )


_METHODS: dict[str, ButcherTableau] = {
    EULER.name: EULER,
    MIDPOINT.name: MIDPOINT,
    HEUN2.name: HEUN2,
    HEUN3.name: HEUN3,
    RK4.name: RK4,
    RK4_CLASSIC.name: RK4_CLASSIC,
    ADAPTIVE_HEUN.name: ADAPTIVE_HEUN,
    FEHLBERG2.name: FEHLBERG2,
    BOSH3.name: BOSH3,
    DOPRI5.name: DOPRI5,
    DOPRI8.name: DOPRI8,
    TSIT5.name: TSIT5,
    IMPLICIT_EULER.name: IMPLICIT_EULER,
    IMPLICIT_MIDPOINT.name: IMPLICIT_MIDPOINT,
    TRAPEZOID.name: TRAPEZOID,
    RADAU_IIA3.name: RADAU_IIA3,
    RADAU_IIA5.name: RADAU_IIA5,
    GL4.name: GL4,
    GL6.name: GL6,
    SDIRK2.name: SDIRK2,
    TRBDF2.name: TRBDF2,
}

# The method ``odeint`` picks when the caller passes ``method=None``.
_DEFAULT_METHOD = DOPRI5.name
