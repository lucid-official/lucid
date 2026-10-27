r"""An underactuated swing-up, in pure Lucid, seen through pixels.

:mod:`~lucid.test._fixtures.point_mass` answers "does the agent learn to do
anything", and it answers it with the easiest problem that still needs the
whole machine — the action moves the agent directly, so the optimal policy
is "go toward the target".  That is a low bar, and a measured one: a
*constant* action already scores 1.7 against an oracle's 3.6 there.

This one cannot be faked.  The pendulum is **underactuated** — the torque
limit is a fraction of the gravitational torque, so no action sequence
lifts the rod directly.  Solving it means pumping energy in over several
swings and then catching the rod at the top.  Measured over six seconds:
random 0.99, best constant 6.02, energy-shaping oracle 77.41, a **12.9x**
separation where the point mass offers 2.1x.

Nothing outside Lucid is used.  The dynamics are integrated with
:func:`lucid.diffeq.odeint` — the same tested Runge-Kutta the library
ships — and the frames are drawn with tensor arithmetic, as
:class:`~lucid.test._fixtures.point_mass.PointMass` draws its own.  Set
``device="metal"`` to render on the accelerator and skip a transfer per
step.

On getting the physics right
----------------------------
The equations are derived rather than copied, and then checked against
something a derivation can be held to: with no torque applied, total
mechanical energy must not drift.  Integrated in float64 the drift falls
from 7.3e-05 to 3.0e-09 as the step is quartered — clean convergence, so
what remains is truncation and not a wrong equation.  In the shipped
float32 path it sits at 3e-04 relative, which is the arithmetic's noise
floor rather than the physics.

That check is the point.  Florian (2007), writing about the neighbouring
cart-pole system, opens: *"The classic papers that introduced this problem
contain mistakes in the equations that govern the dynamics of the
cart-pole system, and these mistakes propagated in other studies that used
them."*  An invariant catches that; a citation does not.

What it measures, and one thing it measured
-------------------------------------------
DreamerV3 learns it.  Interleaved collection, 4000 gradient steps on
Metal, a 1.2M-parameter configuration: return 48.87 against random's 2.47
and the best constant's 6.02 — **8.1x the best constant** and 63% of the
oracle.  The point mass, for comparison, yields 2.7x.

Getting there took one change from the paper's defaults, and it is worth
recording because it qualifies a claim this repository implements
carefully.  At DreamerV3's ``actor_entropy`` of 3e-4 the policy collapses:
6000 steps returned *exactly* 6.02 at every checkpoint — the best constant,
to two decimals — with entropy pinned at -1.58, far below the uniform
limit of 0.69.  At 1e-2 the same model reaches 41 by step 1500.  The
paper's "one configuration spans every domain" is a claim about the
benchmarks it reports, and this task is not one of them; on a harder
exploration problem the entropy coefficient mattered by a factor of eight
in return.

A cart-pole swing-up was written alongside this and **is not shipped**.
Its physics passes the same energy check, but the only reference
controller found for it takes 27 seconds — some 680 steps — to raise the
pole, which is far outside what an agent with a 15-step imagination
horizon can be asked to discover.  Shipping a task without a demonstration
that it is solvable at the scale it will be used is the mistake this file
is trying not to make.  The working gains are recorded in the commit that
removed it, so a second attempt does not start from nothing.
"""

import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid.diffeq import odeint
from lucid.utils.rollout import StepResult

__all__ = ["Pendulum"]

_SIZE = 64


def _grid(device: str) -> tuple[Tensor, Tensor]:
    """Pixel-centre coordinates in ``[-1, 1]``, as ``(ys, xs)``."""
    axis = lucid.linspace(-1.0, 1.0, _SIZE, device=device)
    xs = axis.reshape(1, _SIZE).repeat(_SIZE, 1)
    ys = axis.reshape(_SIZE, 1).repeat(1, _SIZE)
    return ys, xs


def _blob(ys: Tensor, xs: Tensor, y: float, x: float, width: float) -> Tensor:
    """A soft disc at ``(y, x)`` — the same shape PointMass draws with."""
    squared = (xs - x) ** 2 + (ys - y) ** 2
    return lucid.exp(-squared / (2.0 * width**2))


def _segment(
    ys: Tensor, xs: Tensor, start: tuple[float, float], end: tuple[float, float], width: float
) -> Tensor:
    """A soft line from ``start`` to ``end``, both ``(y, x)``.

    Drawn as the distance from each pixel to the segment, so a rod appears
    at whatever angle the state puts it — the only thing a pixel-based
    agent can read the pole's angle from.
    """
    ay, ax = start
    by, bx = end
    dy, dx = by - ay, bx - ax
    length = dy * dy + dx * dx
    if length < 1e-12:
        return _blob(ys, xs, ay, ax, width)
    # Projection of each pixel onto the segment, clamped to its ends.
    t = ((ys - ay) * dy + (xs - ax) * dx) / length
    t = t.clip(0.0, 1.0)
    squared = (ys - (ay + t * dy)) ** 2 + (xs - (ax + t * dx)) ** 2
    return lucid.exp(-squared / (2.0 * width**2))


class Pendulum:
    r"""A rod that must be swung upright and held there, seen as pixels.

    Parameters
    ----------
    horizon : int, default=120
        Steps per episode — six seconds at the default step.  The oracle
        needs about three to raise the rod, so this leaves as long again
        to hold it, which is what the reward is actually measuring.  An
        earlier default of 40 was picked by analogy with the point mass
        and is 1.2 natural periods: too short to pump up at all, and the
        oracle scored 3.6 out of 40 because of it.
    dt : float, default=0.05
        Integration step.
    torque : float, default=0.35
        Limit on ``|u|``, as a fraction of the peak gravitational torque
        ``m g l / 2``.  Below ``1`` the pendulum **cannot** be lifted
        directly and has to be pumped, which is the whole point; the
        default leaves a comfortable margin.
    device : str, default="cpu"
        Where frames are rendered.  Set ``"metal"`` to keep observations on
        the accelerator and skip a transfer per step.

    Attributes
    ----------
    theta, omega : float
        Angle from upright in radians and its rate.  ``theta = 0`` is
        balanced, ``theta = pi`` is hanging, which is where an episode
        starts.

    Raises
    ------
    ValueError
        If ``torque`` is not in ``(0, 1)`` — at or above ``1`` the task
        stops being a swing-up and the tests that rest on that stop
        meaning anything.

    Notes
    -----
    A uniform rod of mass :math:`m` and length :math:`l` pivoted at one
    end has moment of inertia :math:`I = m l^2 / 3` and a gravitational
    torque of :math:`m g (l/2) \sin\theta` about that pivot, giving

    .. math::

        \ddot{\theta} = \frac{3g}{2l}\sin\theta + \frac{3u}{m l^2}.

    Derived rather than cited, and then pinned by the invariant a
    derivation can be checked against: with :math:`u = 0` the total energy
    :math:`\tfrac{1}{2}I\dot{\theta}^2 + mg(l/2)\cos\theta` is conserved.

    Examples
    --------
    >>> from lucid.test._fixtures.classic_control import Pendulum
    >>> env = Pendulum(horizon=5)
    >>> env.reset().shape
    (3, 64, 64)
    """

    def __init__(
        self, horizon: int = 120, dt: float = 0.05, torque: float = 0.35, device: str = "cpu"
    ) -> None:
        """Initialise the task. See the class docstring for parameters."""
        if not 0.0 < torque < 1.0:
            raise ValueError(
                f"torque must be in (0, 1) — at 1 the pendulum can be lifted "
                f"directly and this stops being a swing-up; got {torque}"
            )
        self.horizon = horizon
        self.dt = dt
        self.mass = 1.0
        self.length = 1.0
        self.gravity = 9.81
        # The peak gravitational torque, at the horizontal.
        self.max_torque = torque * self.mass * self.gravity * self.length / 2.0
        self.device = device
        self._ys, self._xs = _grid(device)
        self.theta = math.pi
        self.omega = 0.0
        self._t = 0

    # ── physics ──────────────────────────────────────────────────────────

    def acceleration(self, theta: float, omega: float, action: float) -> float:
        """Angular acceleration at a state — the right-hand side, in floats.

        Parameters
        ----------
        theta, omega : float
            Angle from upright, and its rate.
        action : float
            Applied torque, already clipped.

        Returns
        -------
        float
            :math:`\\ddot{\\theta}`.
        """
        gravity_term = 3.0 * self.gravity * math.sin(theta) / (2.0 * self.length)
        control_term = 3.0 * action / (self.mass * self.length**2)
        return gravity_term + control_term

    def energy(self) -> float:
        """Total mechanical energy — the invariant the physics is checked by.

        Returns
        -------
        float
            Kinetic plus potential, taking the pivot as the origin.  Held
            constant by an unforced trajectory, and only by a correct one.
        """
        inertia = self.mass * self.length**2 / 3.0
        kinetic = 0.5 * inertia * self.omega**2
        potential = self.mass * self.gravity * (self.length / 2.0) * math.cos(self.theta)
        return kinetic + potential

    def _integrate(self, action: float) -> None:
        """Advance one step with ``lucid.diffeq``'s Runge-Kutta."""

        def rhs(_: Tensor, state: Tensor) -> Tensor:
            theta, omega = float(state[0].item()), float(state[1].item())
            return lucid.stack([state[1], lucid.tensor(self.acceleration(theta, omega, action))])

        with lucid.no_grad():
            path = odeint(
                rhs,
                lucid.tensor([self.theta, self.omega]),
                lucid.tensor([0.0, self.dt]),
                method="rk4",
            )
        self.theta = float(path[-1][0].item())
        self.omega = float(path[-1][1].item())

    # ── environment protocol ─────────────────────────────────────────────

    def _render(self) -> Tensor:
        """Draw the rod, its tip and its pivot — ``(3, 64, 64)``."""
        # Screen coordinates: y grows downward, so upright is negative y.
        tip = (-0.8 * math.cos(self.theta), 0.8 * math.sin(self.theta))
        return lucid.stack(
            [
                _blob(self._ys, self._xs, tip[0], tip[1], 0.11),
                _segment(self._ys, self._xs, (0.0, 0.0), tip, 0.055),
                _blob(self._ys, self._xs, 0.0, 0.0, 0.07),
            ],
            dim=0,
        )

    def reset(self) -> Tensor:
        """Start hanging, at rest.

        Returns
        -------
        Tensor
            The first frame, ``(3, 64, 64)``.
        """
        self.theta = math.pi
        self.omega = 0.0
        self._t = 0
        return self._render()

    def step(self, action: Tensor) -> StepResult:
        """Apply a torque for one step.

        Parameters
        ----------
        action : Tensor
            One element in ``[-1, 1]``, scaled to the torque limit.  A
            one-hot of width two is also accepted, so a discrete policy can
            drive the same task.

        Returns
        -------
        StepResult
            Reward is ``(cos(theta) + 1) / 2``: one upright, zero hanging,
            and dense in between so a short run measures something.
        """
        flat = action.reshape(-1)
        if int(flat.shape[0]) == 2:  # one-hot: push one way or the other
            scalar = 1.0 if float(flat[1].item()) > float(flat[0].item()) else -1.0
        else:
            scalar = float(flat[0].item())
        torque = max(-1.0, min(1.0, scalar)) * self.max_torque

        self._integrate(torque)
        self._t += 1
        reward = (math.cos(self.theta) + 1.0) / 2.0
        return StepResult(
            observation=self._render(),
            reward=reward,
            terminated=False,
            truncated=self._t >= self.horizon,
        )

    def optimal_action(self) -> Tensor:
        """An energy-shaping swing-up, for a reference return.

        Returns
        -------
        Tensor
            One element in ``[-1, 1]``.

        Notes
        -----
        The textbook controller, and the reason this task is not solvable
        by a constant: away from the top it pumps energy in the direction
        the rod is already turning until the total matches what standing
        upright costs, and near the top it switches to damping.  Used to
        establish what a good return looks like, never as a training
        signal.
        """
        upright = self.mass * self.gravity * self.length / 2.0
        if math.cos(self.theta) > 0.85:
            command = -2.0 * self.omega - 6.0 * math.sin(self.theta)
        elif abs(self.omega) < 1e-3:
            # Break the symmetry deliberately.  Energy pumping is
            # proportional to the rate, so it is exactly zero at rest —
            # and hanging straight down is a stable equilibrium, so
            # without a kick the system sits there for ever.  An earlier
            # version relied on sin(pi) evaluating to 1.2e-16 to get
            # started, which worked and is not something to ship.
            command = 1.0
        else:
            command = 4.0 * self.omega * (upright - self.energy())
        return lucid.tensor([max(-1.0, min(1.0, command))])
