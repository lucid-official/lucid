"""Policies and the loop that runs them against an environment.

This is the smallest of the three pieces and the one that earns its place
least obviously — it is a ``while not done`` loop.  What justifies it is
that two of the papers' hyperparameters live nowhere else: **action
repeat**, which decides how many environment steps one decision covers,
and the **exploration noise** added while collecting.  Both change the
data the world model is fitted to, and both are stated in the papers, so
leaving them to every caller to reinvent is how implementations quietly
diverge.

Deliberately not here: checkpointing, logging, evaluation schedules,
vectorised environments, anything resembling a ``Trainer``.  The training
loop stays the caller's.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import lucid
from lucid._tensor.tensor import Tensor
from lucid.utils.rollout._env import Environment, Episode

if TYPE_CHECKING:
    # Import-time only: pulling lucid.models in here would close a cycle
    # (lucid.utils -> rollout -> models -> nn -> utils).  Nothing below
    # needs the class at runtime — LatentPolicy only calls methods on
    # whatever it is handed.
    from lucid.models.generative._common._rssm import RSSM, RSSMState

__all__ = ["Policy", "RandomPolicy", "LatentPolicy", "rollout"]


@runtime_checkable
class Policy(Protocol):
    """Something that chooses actions and can be told an episode restarted.

    The ``reset`` is what distinguishes this from a plain function.  A
    world model does not act from the frame in front of it — it acts from
    a belief it has been updating since the episode began — so the driver
    has to be able to say "that was a different episode, start again".  A
    stateless policy can implement it as a no-op.

    Notes
    -----
    Actions are expected in ``(-1, 1)``.  Anything narrower is fine; the
    environment is what decides how they are scaled.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.rollout import Policy, RandomPolicy
    >>> isinstance(RandomPolicy(2), Policy)
    True
    """

    def reset(self) -> None:
        """Forget whatever the previous episode left behind."""
        ...

    def __call__(self, observation: Tensor) -> Tensor:
        """Choose an action.

        Parameters
        ----------
        observation : Tensor
            ``(C, H, W)``.

        Returns
        -------
        Tensor
            ``(action_dim,)`` in ``(-1, 1)``.
        """
        ...


class RandomPolicy:
    r"""Uniform actions, in whichever shape the action space has.

    What fills the buffer before there is anything worth learning from.
    Both papers seed with a handful of random episodes for exactly this
    reason: a world model fitted on nothing has no opinion for a policy
    to improve on.

    Parameters
    ----------
    action_dim : int
        Width of the action vector, or the number of alternatives when
        ``discrete``.
    discrete : bool, default=False
        Draw a one-hot over ``action_dim`` choices instead of a vector in
        ``(-1, 1)``.

    Notes
    -----
    The flag is not cosmetic.  Seeding a discrete agent's buffer with
    continuous noise trains its dynamics on action vectors it will never
    be given again — a uniform draw in ``(-1, 1)`` looks nothing like a
    one-hot — so the model is fitted on one input distribution and asked
    to act under another.

    Examples
    --------
    >>> from lucid.utils.rollout import RandomPolicy
    >>> policy = RandomPolicy(3)
    >>> policy.reset()
    >>> policy(None).shape
    (3,)
    >>> onehot = RandomPolicy(4, discrete=True)
    >>> onehot.reset()
    >>> float(onehot(None).sum())
    1.0
    """

    def __init__(self, action_dim: int, discrete: bool = False) -> None:
        """Initialise the policy. See the class docstring for parameters."""
        if action_dim < 1:
            raise ValueError(f"action_dim must be at least 1, got {action_dim}")
        self.action_dim = action_dim
        self.discrete = discrete

    def reset(self) -> None:
        """No state to clear; present so this satisfies :class:`Policy`."""
        return None

    def __call__(self, observation: Tensor) -> Tensor:
        """Draw a uniform action, reading the observation only for its device.

        The draw ignores what is in the frame — that is the point of a
        random policy — but not *where* it is.  Seeding an accelerator
        environment with actions left on the CPU builds episodes whose
        five tensors do not agree, and the mismatch is only reported much
        later, inside whichever op a world model happens to reach first.
        """
        device = observation.device.type if isinstance(observation, Tensor) else "cpu"
        if self.discrete:
            index = int(float(lucid.rand(()).item()) * self.action_dim)
            index = min(index, self.action_dim - 1)
            return (
                lucid.nn.functional.one_hot(
                    lucid.tensor([index], device=device),
                    num_classes=self.action_dim,
                )
                .reshape(self.action_dim)
                .to(lucid.float32)
            )
        return lucid.rand((self.action_dim,), device=device) * 2.0 - 1.0


class LatentPolicy:
    r"""Carries an RSSM belief through an episode and asks the model to act.

    A world model does not act from the frame in front of it; it acts from
    a belief it has been updating since the episode began.  So this holds
    that belief, folds each new observation into it, and hands it to
    whatever the model uses to choose — which is not the same method
    across the family, and is therefore passed in rather than assumed:
    Dreamer has a learned actor, PlaNet searches with a planner.

    Parameters
    ----------
    encode : callable
        ``(B, T, C, H, W) -> (B, T, embed)``, normally ``model.encode``.
    rssm : RSSM
        The model's recurrent state-space model.
    act : callable
        ``RSSMState -> (1, action_dim)``, given the single-step posterior.
        For Dreamer this is ``lambda s: model.act(s, sample=False)``.  For
        PlaNet it is the *task wrapper's* planner — ``plan`` lives on
        :class:`~lucid.models.PlaNetForWorldModeling`, not on the trunk,
        so the encoder and RSSM come from ``wrapper.planet`` while the
        planner comes from ``wrapper``.
    action_dim : int
        Width of the action vector.
    noise : float, default=0.0
        Standard deviation of Gaussian exploration noise added to the
        chosen action.  The papers collect with ``0.3`` and evaluate with
        ``0``.

    Notes
    -----
    The action is clipped into ``(-1, 1)`` whether or not noise was added,
    because that is what :class:`Policy` promises an environment.  Dreamer
    emits a ``tanh``-squashed action and the clip does nothing; PlaNet's
    planner searches an unbounded Gaussian and relies on it, which is the
    clipping the paper describes.

    Notes
    -----
    Runs under :func:`lucid.no_grad` — acting is not a thing to
    differentiate, and holding a graph across a whole episode would keep
    every frame alive.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import dreamer
    >>> from lucid.utils.rollout import LatentPolicy
    >>> model = dreamer(action_dim=2, cnn_depth=2, stoch_size=4, deter_size=8,
    ...                 hidden_size=8, actor_hidden=8, value_hidden=8,
    ...                 reward_hidden=8).eval()
    >>> policy = LatentPolicy(model.encode, model.rssm,
    ...                       lambda s: model.act(s, sample=False), 2)
    >>> policy.reset()
    >>> policy(lucid.zeros((3, 64, 64))).shape
    (2,)
    """

    def __init__(
        self,
        encode: Callable[[Tensor], Tensor],
        rssm: RSSM,
        act: Callable[[RSSMState], Tensor],
        action_dim: int,
        noise: float = 0.0,
    ) -> None:
        """Initialise the policy. See the class docstring for parameters."""
        if noise < 0.0:
            raise ValueError(f"noise must be non-negative, got {noise}")
        self._encode = encode
        self.rssm = rssm
        self._act = act
        self.action_dim = action_dim
        self.noise = noise
        self._state: RSSMState | None = None
        self._previous: Tensor | None = None

    def reset(self) -> None:
        """Drop the belief so the next episode starts from the prior."""
        self._state = None
        self._previous = None

    @lucid.no_grad()
    def __call__(self, observation: Tensor) -> Tensor:
        """Fold in the observation, then choose."""
        if self._state is None:
            # `.type` — str(device) renders as "device('cpu')", which the
            # factories do not parse.
            self._state = self.rssm.initial(1, device=observation.device.type)
            self._previous = lucid.zeros((1, self.action_dim), device=observation.device)

        shape = tuple(int(s) for s in observation.shape)
        embed = self._encode(observation.reshape(1, 1, *shape))[:, 0]
        _, posterior = self.rssm.posterior_step(
            self._state, cast(Tensor, self._previous), embed, sample=False
        )
        self._state = posterior

        action = self._act(posterior)
        if self.noise > 0.0:
            action = (
                action
                + lucid.randn(
                    tuple(int(s) for s in action.shape),
                    device=action.device,
                    dtype=action.dtype,
                )
                * self.noise
            )
        # Always, not only when noise was added.  Dreamer's tanh already
        # satisfies this and the clip is a no-op; PlaNet's planner searches
        # an unbounded Gaussian and does not, which is what the paper
        # clips.  Letting the bound depend on whether exploration was
        # requested would make the same policy legal in one call and not
        # the other.
        action = lucid.clip(action, -1.0, 1.0)
        self._previous = action
        return action[0]


def rollout(
    env: Environment,
    policy: Policy,
    *,
    action_repeat: int = 1,
    max_steps: int | None = None,
) -> tuple[Episode, float]:
    r"""Run one episode and return it, ready for the replay buffer.

    Parameters
    ----------
    env : Environment
        Anything satisfying the protocol.
    policy : Policy
        Reset once here, then called per decision.
    action_repeat : int, default=1
        Environment steps per decision.  The papers use 2 for some
        Control Suite tasks and 4 for others, and it is not a free
        parameter — it sets what one step of the learned dynamics
        *means*, so the same number has to be used when collecting and
        when reporting a return.
    max_steps : int or None, optional
        Cut the episode off after this many recorded steps, reported as
        truncation rather than termination.  ``None`` runs until the
        environment says stop.

    Returns
    -------
    episode : Episode
        ``(T, ...)`` tensors, with ``actions[t]`` the action taken *into*
        step ``t`` — so the first entry is a zero vector, because nothing
        was taken into the first observation.
    total_reward : float
        Undiscounted sum over the episode.

    Notes
    -----
    A repeated action is recorded once per environment step, not once per
    decision, so the stored sequence is at the environment's own rate and
    a model trained on it predicts single steps.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.rollout import RandomPolicy, StepResult, rollout
    >>> class Trivial:
    ...     def reset(self):
    ...         self.t = 0
    ...         return lucid.zeros((3, 4, 4))
    ...     def step(self, action):
    ...         self.t += 1
    ...         return StepResult(lucid.zeros((3, 4, 4)), 1.0, False, self.t >= 5)
    >>> episode, total = rollout(Trivial(), RandomPolicy(2))
    >>> len(episode), total
    (5, 5.0)
    """
    if action_repeat < 1:
        raise ValueError(f"action_repeat must be at least 1, got {action_repeat}")
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be at least 1, got {max_steps}")

    policy.reset()
    observation = env.reset()
    zero = lucid.zeros_like(_probe_action(policy, observation))

    observations: list[Tensor] = []
    actions: list[Tensor] = [zero]
    rewards: list[float] = []
    discounts: list[float] = []
    total = 0.0

    action = zero
    done = False
    while not done:
        action = policy(observation)
        for _ in range(action_repeat):
            observations.append(observation)
            result = env.step(action)
            observation = result.observation
            rewards.append(float(result.reward))
            # Truncation is not termination: the episode would have gone
            # on, so nothing downstream should discount it to zero.
            discounts.append(0.0 if result.terminated else 1.0)
            total += float(result.reward)
            done = result.terminated or result.truncated
            if max_steps is not None and len(observations) >= max_steps:
                done = True
            if done:
                break
            actions.append(action)

    steps = len(observations)
    # Rewards and discounts arrive as Python floats, so they have no device
    # of their own and would default to the CPU.  An episode collected from
    # an accelerator environment would then carry three tensors there and
    # two here, and the mismatch does not surface until a world model reads
    # the batch — several layers away, reported against whichever op
    # happened to touch a reward first.  `.type` because `str(device)`
    # renders as "device('metal')", which the factories do not parse.
    device = observations[0].device.type
    episode = Episode(
        observations=lucid.stack(observations, dim=0),
        actions=lucid.stack(actions[:steps], dim=0),
        rewards=lucid.tensor(rewards[:steps], device=device),
        discounts=lucid.tensor(discounts[:steps], device=device),
    )
    return episode, total


def _probe_action(policy: Policy, observation: Tensor) -> Tensor:
    """Shape of the action this policy emits, without advancing anything.

    Parameters
    ----------
    policy : Policy
        Already reset.
    observation : Tensor
        The first observation.

    Returns
    -------
    Tensor
        An action of the right shape and device, used only as a template
        for the zero placed before the first observation.

    Notes
    -----
    ``LatentPolicy`` updates its belief when called, so it is reset again
    afterwards; the probe must not leave the policy a step ahead of the
    environment.
    """
    action = policy(observation)
    policy.reset()
    return action
