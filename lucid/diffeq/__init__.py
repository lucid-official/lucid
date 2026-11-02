"""lucid.diffeq — differential-equation solvers.

Numerical integration of ordinary differential equations, ``dy/dt = f(t, y)``,
where the right-hand side is an arbitrary Lucid callable — most often a neural
network, as in continuous normalising flows, flow matching, and rectified flow.

Surface: :func:`odeint` integrates to a set of times you name up front and
:func:`odeint_dense` returns a continuous solution you can query afterwards, and
:func:`odeint_adjoint` trades exactness for constant memory when the solve is
long enough that retained stages dominate, and :func:`odeint_event` runs until a
condition fires rather than to a time you name;
:class:`ButcherTableau` is the coefficient table they dispatch on.  Twenty-one
built-in tableaux are exported, and a custom tableau is accepted anywhere a
method name is.

The tableau decides how ``t`` is read.  An **adaptive** method — the default
:data:`DOPRI5`, plus :data:`DOPRI8`, :data:`TSIT5`, :data:`BOSH3`,
:data:`FEHLBERG2` and :data:`ADAPTIVE_HEUN` —
carries an embedded error estimate, so it picks its own step sizes to hold the
error inside ``rtol`` / ``atol`` and interpolates to the times in ``t``.  A
**fixed-step** method — :data:`EULER`, :data:`MIDPOINT`, :data:`HEUN2`,
:data:`HEUN3`, :data:`RK4`, :data:`RK4_CLASSIC` — steps once per interval of
``t``, so ``t`` is the integration grid itself.

Two further fixed-step families trade a different way.  **Adams multistep** —
method names ``"explicit_adams"``, ``"implicit_adams"`` and ``"fixed_adams"``,
which are not tableaux — reaches high order by reusing derivatives from
earlier steps, so a step costs one evaluation whatever its order.  **Implicit**
methods — :data:`IMPLICIT_EULER`, :data:`IMPLICIT_MIDPOINT`, :data:`TRAPEZOID`,
:data:`RADAU_IIA3`, :data:`RADAU_IIA5`, :data:`GL4`, :data:`GL6`,
:data:`SDIRK2` and :data:`TRBDF2` — solve a nonlinear system every step, which
is expensive but is what lets them take large steps on a stiff problem, where
an explicit method is held to tiny ones by stability alone.

A fixed-step implicit method that cannot solve a step's stage equations raises
:class:`ImplicitSolveError`, naming the method, the time and the stage, rather
than carrying on from an iterate that does not satisfy them.

Everything here is reached as ``lucid.diffeq.*`` and nowhere else — there are
no top-level aliases and no ``Tensor`` methods, so each name has exactly one
canonical path.

Stochastic differential equations are not implemented yet.

Attributes
----------
DOPRI5 : ButcherTableau
    Dormand-Prince 5(4) — seven stages, adaptive, fifth order.  Method name
    ``"dopri5"``, and the default when ``method`` is left unset.  FSAL, so a
    step costs six right-hand-side evaluations rather than seven.  Its
    interpolant is derived rather than nominal (``mid_order`` 4), which makes it
    one of the two methods worth reaching for when values *between* steps
    matter — dense output and event times.
DOPRI8 : ButcherTableau
    Dormand-Prince 8(5,3) — fourteen stages, adaptive, eighth order.  Method
    name ``"dopri8"``.  The most accurate method here per step, and the most
    expensive; it earns that back on smooth problems at tight tolerances by
    taking far fewer steps.  Two things to know: its dense output is still the
    quartic every adaptive method uses, so interpolated values are capped well
    below eighth order, and at tight tolerances the embedded error estimate is
    dominated by round-off, which is a property of the method rather than a
    defect.
TSIT5 : ButcherTableau
    Tsitouras 5(4) — seven stages, adaptive, fifth order.  Method name
    ``"tsit5"``.  Coefficients tuned for a smaller error constant than
    :data:`DOPRI5` at the same cost; also FSAL, and also carries a derived
    interpolant (``mid_order`` 4), so it is the other of the two.
BOSH3 : ButcherTableau
    Bogacki-Shampine 3(2) — four stages, adaptive, third order.  Method name
    ``"bosh3"``.  Cheap per step and a reasonable choice at loose tolerances.
    Its interpolant is only first order, so dense output and event times are
    bounded by that rather than by ``rtol`` / ``atol`` — asking it for either
    raises a warning saying so.
FEHLBERG2 : ButcherTableau
    Fehlberg 2(1) — three stages, adaptive, second order.  Method name
    ``"fehlberg2"``.  First-order interpolant, as for :data:`BOSH3`.
ADAPTIVE_HEUN : ButcherTableau
    Heun-Euler 2(1) — two stages, adaptive, second order.  Method name
    ``"adaptive_heun"``.  The cheapest adaptive method; first-order
    interpolant.  It is not FSAL, and Lucid recomputes ``f(t1, y1)`` for the
    next step rather than reusing the final stage, which is why its values
    differ slightly from the reference ODE library's — at realistic tolerances
    Lucid's are three to four times more accurate.
EULER : ButcherTableau
    Forward Euler — one stage, first order.  Method name ``"euler"``.  Fixed
    step, so ``t`` is the integration grid itself and the solve takes exactly
    ``len(t) - 1`` steps.
MIDPOINT : ButcherTableau
    Explicit midpoint — two stages, second order.  Method name ``"midpoint"``.
HEUN2 : ButcherTableau
    Heun's method, the explicit trapezoid — two stages, second order.  Method
    name ``"heun2"``.
HEUN3 : ButcherTableau
    Heun's third-order method — three stages, third order.  Method name
    ``"heun3"``.
RK4 : ButcherTableau
    Kutta's 3/8 rule — four stages, fourth order.  Method name ``"rk4"``.
    That name resolves to this tableau and not to the textbook classical one,
    matching what it means in the reference ODE library: the two share an
    order, so only a direct value comparison distinguishes them.
RK4_CLASSIC : ButcherTableau
    The classical Runge-Kutta method — four stages, fourth order.  Method name
    ``"rk4_classic"``.  A Lucid extension: the reference has no name for this
    tableau, but it is what most other sources call "RK4".
IMPLICIT_EULER : ButcherTableau
    Backward Euler — one stage, first order, implicit.  Method name
    ``"implicit_euler"``.  The most stable method here and the least accurate:
    unconditionally stable, so no step size can make it blow up, at the cost
    of damping the solution along with the error.
IMPLICIT_MIDPOINT : ButcherTableau
    The implicit midpoint rule — one stage, second order, implicit.  Method
    name ``"implicit_midpoint"``.  Symplectic, so it conserves the invariants
    of a Hamiltonian system over long integrations where a non-symplectic
    method of the same order drifts.
TRAPEZOID : ButcherTableau
    The trapezoidal rule — two stages, second order, implicit.  Method name
    ``"trapezoid"``.
RADAU_IIA3 : ButcherTableau
    Radau IIA — two stages, third order, implicit.  Method name
    ``"radauIIA3"``.  L-stable: stiff components are damped rather than merely
    kept bounded, which is what a stiff problem usually wants.
RADAU_IIA5 : ButcherTableau
    Radau IIA — three stages, fifth order, implicit.  Method name
    ``"radauIIA5"``.  L-stable, and the highest-order stiff method here that
    still damps.  Coefficients are derived from the Legendre roots at sixty
    decimal digits rather than transcribed, and the derivation is checked
    against the theoretical order at import.
GL4 : ButcherTableau
    Gauss-Legendre — two stages, fourth order, implicit.  Method name
    ``"gl4"``.
GL6 : ButcherTableau
    Gauss-Legendre — three stages, sixth order, implicit.  Method name
    ``"gl6"``.  The highest order per stage any Runge-Kutta method can reach,
    and symplectic; A-stable but not L-stable, so it keeps stiff components
    bounded without damping them.
SDIRK2 : ButcherTableau
    Singly diagonally implicit — two stages, second order.  Method name
    ``"sdirk2"``.  Diagonally implicit, so each stage is solved on its own:
    ``s`` solves over ``n`` unknowns instead of one over ``s * n``, which is
    what keeps the quasi-Newton iteration's dense Jacobian small.
TRBDF2 : ButcherTableau
    TR-BDF2 — three stages, second order, implicit.  Method name
    ``"trbdf2"``.  A trapezoidal stage followed by a BDF2 stage; L-stable and
    diagonally implicit, a common default in circuit and chemical kinetics
    solvers.
"""

from lucid.diffeq._adjoint import odeint_adjoint
from lucid.diffeq._implicit import ImplicitSolveError
from lucid.diffeq._solvers import odeint, odeint_dense, odeint_event
from lucid.diffeq._tableau import (
    ADAPTIVE_HEUN,
    BOSH3,
    DOPRI5,
    DOPRI8,
    EULER,
    FEHLBERG2,
    GL4,
    GL6,
    HEUN2,
    HEUN3,
    IMPLICIT_EULER,
    IMPLICIT_MIDPOINT,
    MIDPOINT,
    RADAU_IIA3,
    RADAU_IIA5,
    RK4,
    RK4_CLASSIC,
    SDIRK2,
    TRAPEZOID,
    TRBDF2,
    TSIT5,
    ButcherTableau,
)

__all__ = [
    "odeint",
    "odeint_adjoint",
    "odeint_dense",
    "odeint_event",
    "ButcherTableau",
    "ImplicitSolveError",
    "DOPRI5",
    "DOPRI8",
    "TSIT5",
    "BOSH3",
    "FEHLBERG2",
    "ADAPTIVE_HEUN",
    "EULER",
    "MIDPOINT",
    "HEUN2",
    "HEUN3",
    "RK4",
    "RK4_CLASSIC",
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
