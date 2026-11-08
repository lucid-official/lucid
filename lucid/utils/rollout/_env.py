"""The environment contract, and the episode it produces.

Lucid ships no environments.  It cannot: every compute path here is closed
to outside packages, so there is no adapter to a simulator suite and no
intention of writing one.  What it ships is the *shape* an environment has
to have, so the replay buffer and the rollout driver can be written against
something concrete and a user's own environment can slot in.
"""

from typing import NamedTuple, Protocol, runtime_checkable

from lucid._tensor.tensor import Tensor

__all__ = ["Environment", "Episode", "StepResult"]


class StepResult(NamedTuple):
    r"""What one environment step returns.

    Attributes
    ----------
    observation : Tensor
        What the agent sees after the step, ``(C, H, W)`` for the pixel
        world models.
    reward : float
        Reward for the transition.
    terminated : bool
        The episode reached a terminal state — a failure, a goal, an
        absorbing state.  Nothing follows it, so a discount head should
        learn ``0`` here.
    truncated : bool
        The episode was cut short from outside — a time limit, a step
        budget.  Something *would* have followed, so the value function
        should still bootstrap.

    Notes
    -----
    Termination and truncation are separated because a world model
    genuinely needs them apart.  Collapsing them into one ``done`` flag,
    which is the older convention, teaches a discount head that running
    out of clock is the same as dying — and then the planner refuses to
    look past the horizon it was trained under.
    """

    observation: Tensor
    reward: float
    terminated: bool
    truncated: bool


class Episode(NamedTuple):
    r"""One complete trajectory, laid out for the replay buffer.

    Attributes
    ----------
    observations : Tensor
        ``(T, C, H, W)``.
    actions : Tensor
        ``(T, action_dim)`` — the action taken *into* each step, so
        ``actions[t]`` produced ``observations[t]``.  This is the
        alignment the world models expect.
    rewards : Tensor
        ``(T,)``.
    discounts : Tensor
        ``(T,)``, ``0`` where the episode terminated and ``1`` everywhere
        else — including where it was merely truncated.  Feeds
        ``pcont``; ignored by models that hold the discount constant.

    Notes
    -----
    The first action is a zero vector: there is no action before the
    first observation, and the recurrence needs something of the right
    shape to start from.
    """

    observations: Tensor
    actions: Tensor
    rewards: Tensor
    discounts: Tensor

    @property
    def num_steps(self) -> int:
        """Number of transitions in the episode (``observations.shape[0]``).

        A named accessor rather than ``__len__``: the record unpacks into
        four fields, so ``len`` has to stay four for ``_replace`` / ``_make``
        / ``copy`` to keep working.
        """
        return int(self.observations.shape[0])


@runtime_checkable
class Environment(Protocol):
    r"""What a rollout needs an environment to do.

    Implement this on your own simulator; nothing here touches an outside
    package, and no base class has to be inherited — a plain object with
    these two methods satisfies it.

    Notes
    -----
    Actions are expected in ``(-1, 1)``, which is what a ``tanh``-squashed
    policy emits.  Rescale inside your environment rather than outside it,
    so the policy never has to know the units.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.rollout import Environment, StepResult
    >>> class Trivial:
    ...     def reset(self):
    ...         self.t = 0
    ...         return lucid.zeros((3, 64, 64))
    ...     def step(self, action):
    ...         self.t += 1
    ...         return StepResult(lucid.zeros((3, 64, 64)), 0.0, False, self.t >= 5)
    >>> isinstance(Trivial(), Environment)
    True
    """

    def reset(self) -> Tensor:
        """Start a new episode.

        Returns
        -------
        Tensor
            The first observation, ``(C, H, W)``.
        """
        ...

    def step(self, action: Tensor) -> StepResult:
        """Advance one step.

        Parameters
        ----------
        action : Tensor
            ``(action_dim,)``, expected in ``(-1, 1)``.

        Returns
        -------
        StepResult
            Observation, reward, and the two end-of-episode flags.
        """
        ...
