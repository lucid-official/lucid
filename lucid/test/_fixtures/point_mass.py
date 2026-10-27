"""A control task small enough to train inside the test suite.

The world models can be checked for shapes, gradients and falling losses
without an environment, and all of that was done before this existed. None
of it answers the question that matters: does the agent learn to *do*
anything. A latent dynamics model with a subtly wrong recurrence still
reconstructs frames and still reports a shrinking loss.

So: a dot that has to reach a target, seen only through pixels. Written in
pure Lucid, with no outside package anywhere, so it lives here rather than
behind an optional dependency.

The task is deliberately the easiest control problem that still requires
the whole machine — the action moves the agent directly, so the optimal
policy is "go toward the target", but reaching it still means the model
has to read the observation, carry a belief across steps, and connect
actions to reward.
"""

import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid.utils.rollout import StepResult

__all__ = ["PointMass"]


class PointMass:
    r"""A dot that must reach a fixed target, observed as pixels.

    Parameters
    ----------
    size : int, default=64
        Frame resolution.  The world models' convolutional schedule only
        produces 64, so that is the useful value.
    horizon : int, default=20
        Steps before the episode is truncated.
    target, start : tuple of float
        Positions in ``[-1, 1]^2``.  ``target`` is the fixed goal, and is
        ignored when ``random_target`` is set.
    random_target : bool, default=False
        Draw a new goal on every :meth:`reset`, uniformly in
        ``[-0.8, 0.8]^2``.

        This changes what passing the task means, which is the only
        reason it exists.  With a fixed goal the best constant action
        scores 15.78 against random's 5.77, so an agent that never reads
        its observation already clears any "beats random" bar — measured,
        and the reason the cheap control tests are honest about
        establishing "found a good constant" rather than control.  Move
        the goal and no constant is good for more than one episode, so
        the return can only be raised by *looking*.
    step_size : float, default=0.15
        How far a full-magnitude action moves the agent.
    sigma : float, default=1.0
        Width of the reward's Gaussian.  Wide on purpose: a narrow one
        makes the reward sparse, and then a short training run measures
        whether the agent got lucky rather than whether it learned.
    device : str, default="cpu"
        Where frames are rendered.  A model on the accelerator needs its
        observations there too, so this saves a transfer per step rather
        than being a performance option — the alternative is a device
        mismatch at the encoder's first convolution.

    Attributes
    ----------
    position : list of float
        Current position, mutated in place by :meth:`step`.

    Notes
    -----
    The episode is only ever *truncated*, never terminated — there is no
    failure state and no goal that ends things.  That makes it a poor
    exercise of a discount head, which is deliberate: ``pcont`` is tested
    directly elsewhere, and mixing the two would leave a failure
    ambiguous between them.

    Examples
    --------
    >>> from lucid.test._fixtures.point_mass import PointMass
    >>> env = PointMass(horizon=3)
    >>> obs = env.reset()
    >>> obs.shape
    (3, 64, 64)
    >>> result = env.step(lucid.tensor([1.0, -1.0]))
    >>> result.truncated
    False
    """

    def __init__(
        self,
        size: int = 64,
        horizon: int = 20,
        target: tuple[float, float] = (0.5, -0.5),
        start: tuple[float, float] = (-0.6, 0.6),
        step_size: float = 0.15,
        sigma: float = 1.0,
        device: str = "cpu",
        random_target: bool = False,
    ) -> None:
        """Initialise the environment. See the class docstring."""
        self.size = size
        self.horizon = horizon
        self.target = target
        self.start = start
        self.step_size = step_size
        self.sigma = sigma
        self.device = device
        self.random_target = random_target

        axis = lucid.linspace(-1.0, 1.0, size, device=device)
        self._xs = axis.reshape(1, size)
        self._ys = axis.reshape(size, 1)
        self._target_blob = self._blob(*target)
        self._empty = lucid.zeros((size, size), device=device)
        self.position = list(start)
        self._t = 0

    def _blob(self, x: float, y: float) -> Tensor:
        """A Gaussian bump at ``(x, y)`` over the frame."""
        squared = (self._xs - x) ** 2 + (self._ys - y) ** 2
        return lucid.exp(-squared / (2 * 0.12**2))

    def _render(self) -> Tensor:
        """Agent in channel 0, target in channel 1 — ``(3, size, size)``."""
        return lucid.stack([self._blob(*self.position), self._target_blob, self._empty], dim=0)

    def reset(self) -> Tensor:
        """Return the agent to its start, and re-draw the goal if asked.

        Returns
        -------
        Tensor
            The first observation, ``(3, size, size)``.  The goal is in
            channel 1, so a moved goal is visible rather than hidden
            state — the task stays fully observed and only stops
            rewarding constants.
        """
        if self.random_target:
            draw = lucid.rand((2,)) * 1.6 - 0.8
            self.target = (float(draw[0]), float(draw[1]))
            self._target_blob = self._blob(*self.target)
        self.position = list(self.start)
        self._t = 0
        return self._render()

    def step(self, action: Tensor) -> StepResult:
        """Move the agent and score how close it now is.

        Parameters
        ----------
        action : Tensor
            ``(2,)``, clipped into ``(-1, 1)``.

        Returns
        -------
        StepResult
            Reward is ``exp(-d^2 / 2 sigma^2)`` at the new position, so it
            is dense everywhere rather than only near the target.
        """
        for axis in (0, 1):
            delta = max(-1.0, min(1.0, float(action[axis]))) * self.step_size
            self.position[axis] = max(-1.0, min(1.0, self.position[axis] + delta))

        squared = sum((self.position[i] - self.target[i]) ** 2 for i in range(2))
        reward = math.exp(-squared / (2 * self.sigma**2))
        self._t += 1
        return StepResult(self._render(), reward, False, self._t >= self.horizon)

    def optimal_action(self) -> Tensor:
        """The greedy action — straight at the target.

        Returns
        -------
        Tensor
            ``(2,)``.  Used to establish what a good return looks like, so
            a learned policy can be judged against something better than
            "more than random".
        """
        return lucid.tensor(
            [max(-1.0, min(1.0, (self.target[i] - self.position[i]) * 10.0)) for i in range(2)],
            device=self.device,
        )
