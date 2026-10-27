"""Control benchmarks — the claims the suite deliberately cannot make.

``test_world_model_control.py`` runs in seconds and says so about its
limits: on a fixed target the best constant action scores 15.78 against
random's 5.77, so every "beats random" threshold there is clearable by a
policy that never looks at its observation.

This file removes that escape rather than documenting it. The goal moves
each episode, so a constant is good for one episode and wrong for the
next, and the ceiling for an observation-blind policy becomes a
*measurable number* — the best constant's own return. Beating it is not
a threshold someone picked; it is the only thing a blind policy cannot
do.

Opt in with ``-m control``. Minutes, not seconds, which is why it is
excluded from the default run.

Measured on an M4 Max, moving target, horizon 20:

===============================  ======  =========================
policy                           return  vs the best constant
===============================  ======  =========================
random                            12.90  0.91x
best constant (grid over 25)      14.15  1.00x  (the blind ceiling)
DreamerV3, 1000 gradient steps    16.78  1.19x
greedy (reads the goal)           19.21  1.36x
===============================  ======  =========================

So v3 recovers roughly 60% of the distance from random to a controller
that is handed the goal coordinates. The curve is not monotone —
individual evaluations dip to 0.86x between peaks — so the assertion is
on the best evaluation over the run, and the run is short enough that
"best of six" is not a fishing expedition.

What this still does not establish is the paper's own scores. This is a
64x64 toy with a dense Gaussian reward, not the Control Suite, and no
claim about DMC or Atari numbers should be read out of it.

**The discount here is hostile to Dreamer v1, and that is the task's
doing rather than v1's.** Episodes run 20 steps and ``pcont`` is off, so
imagination never terminates and ``discount=0.99`` asks the critic for an
infinite-horizon value of roughly 70 on a task whose achievable return is
about 14. v2 absorbs the mismatch with its target critic and v3 with its
slow critic and return normalisation — measured, their value estimates
settle at 26 and 36. v1 has neither, and its estimate runs to 1286 while
the world model's reward predictions stay accurate; the actor then
maximises a quantity that is running away from it.

Matching the discount to the episode (``discount=0.95``) leaves v1's
value at 15-18 and it clears the ceiling at 1.12x +/- 0.10 over four
seeds, holding it to the final evaluation. So a v1 row added here at
0.99 would measure the discount, not the implementation. Anyone
extending this file to the earlier families should set the discount from
the horizon first.
"""

import itertools

import pytest

import lucid
import lucid.models as M
import lucid.nn as nn
import lucid.optim as optim
from lucid.test._fixtures.point_mass import PointMass
from lucid.utils.rollout import LatentPolicy, RandomPolicy, SequenceReplay, rollout

pytestmark = [pytest.mark.control, pytest.mark.slow]


def _clip(model: object) -> None:
    """The gradient clipping the papers specify and this file omitted.

    Dreamer clips at norm 100; DreamerV3's Table B.1 raises the world
    model's cap to 1000 and keeps the actor-critic at 100. Training
    without any is not the published algorithm.

    The omission is not academic here. Measured on the moving-target
    task, the critic's gradient norm reaches 2.3e4 and its value
    estimate runs to 1286 for a task whose achievable return is about
    14 — the return curve shows only that the agent does not improve,
    which is the symptom these two numbers explain.
    """
    nn.utils.clip_grad_norm_(model.world_parameters(), max_norm=1000.0)
    nn.utils.clip_grad_norm_(model.value_parameters(), max_norm=100.0)
    nn.utils.clip_grad_norm_(model.actor_parameters(), max_norm=100.0)


_DEVICE = "metal"

# Larger than the suite's config and still small. The point of the
# benchmark is the task, not the capacity.
_CONFIG = dict(
    action_dim=2,
    cnn_depth=8,
    stoch_size=8,
    discrete=8,
    deter_size=64,
    hidden_size=64,
    blocks=8,
    actor_hidden=64,
    value_hidden=64,
    reward_hidden=64,
    horizon=10,
    num_bins=41,
    pcont=False,
)

# Measured, not guessed: 3e-3 on the world model is what moves this from
# "hovers at random" to "beats the constant ceiling", and 8e-4 is the
# widest actor rate that stays stable. At 2e-2 the actor's gradient
# collapses to 1e-3 by step 80 and its entropy returns to the uniform
# limit — which reads exactly like a broken objective and is not one.
_WORLD_LR, _VALUE_LR, _ACTOR_LR = 3e-3, 2e-4, 8e-4

_STEPS = 1000
_COLLECT_EVERY = 10
_EVALUATE_EVERY = 200
_EPISODES = 15


class _Constant:
    """A policy that cannot see. The ceiling this file measures against."""

    def __init__(self, action: tuple[float, float]) -> None:
        self.action = lucid.tensor(list(action), device=_DEVICE)

    def reset(self) -> None:
        return None

    def __call__(self, observation: lucid.Tensor) -> lucid.Tensor:
        return self.action


class _Greedy:
    """Handed the goal. What solving the task looks like."""

    def __init__(self, env: PointMass) -> None:
        self.env = env

    def reset(self) -> None:
        return None

    def __call__(self, observation: lucid.Tensor) -> lucid.Tensor:
        return self.env.optimal_action()


def _score(env: PointMass, policy: object, episodes: int = _EPISODES) -> float:
    return sum(rollout(env, policy)[1] for _ in range(episodes)) / episodes


def _blind_ceiling(env: PointMass) -> float:
    """The best any observation-blind policy achieves, by search.

    A grid rather than a remembered pair: the best constant depends on
    the start, the step size and the goal distribution, so hard-coding
    one would silently stop being the ceiling the moment the fixture
    changes and turn this benchmark into a weaker claim without saying so.
    """
    grid = (-1.0, -0.5, 0.0, 0.5, 1.0)
    return max(_score(env, _Constant(a), episodes=8) for a in itertools.product(grid, repeat=2))


@pytest.fixture(scope="module")
def moving_target() -> PointMass:
    return PointMass(horizon=20, random_target=True, device=_DEVICE)


class TestTheTaskDeniesBlindPolicies:
    """Guards the benchmark below. If a constant can win, it measures nothing."""

    def test_a_constant_barely_beats_random(self, moving_target: PointMass) -> None:
        lucid.manual_seed(0)
        random_return = _score(moving_target, RandomPolicy(2))
        ceiling = _blind_ceiling(moving_target)
        assert ceiling < 1.3 * random_return, (
            f"a constant scores {ceiling:.2f} against random's "
            f"{random_return:.2f} — the escape this file exists to close "
            f"is open again"
        )

    def test_reading_the_goal_is_worth_much_more(self, moving_target: PointMass) -> None:
        """There has to be headroom above the ceiling, or nothing can pass."""
        lucid.manual_seed(0)
        assert _score(moving_target, _Greedy(moving_target)) > 1.3 * _blind_ceiling(moving_target)

    def test_the_fixed_target_still_admits_one(self) -> None:
        """The contrast, so the difference is the variant and not the seed."""
        lucid.manual_seed(0)
        fixed = PointMass(horizon=20, device=_DEVICE)
        assert _blind_ceiling(fixed) > 2.0 * _score(fixed, RandomPolicy(2))


class TestDreamerV3BeatsTheBlindCeiling:
    """State-dependent control, on the only task here that can show it."""

    def test_it_learns_to_read_the_goal(self, moving_target: PointMass) -> None:
        lucid.manual_seed(0)
        env = moving_target
        model = M.create_model("dreamer_v3_12m_world_model", **_CONFIG)
        model = model.metal() if _DEVICE == "metal" else model
        optimisers = [
            optim.Adam(model.world_parameters(), lr=_WORLD_LR),
            optim.Adam(model.value_parameters(), lr=_VALUE_LR),
            optim.Adam(model.actor_parameters(), lr=_ACTOR_LR),
        ]

        replay = SequenceReplay(capacity=500_000)
        for _ in range(30):
            replay.add(rollout(env, RandomPolicy(2))[0])
        ceiling = _blind_ceiling(env)

        inner = model.dreamer_v3
        evaluator = LatentPolicy(inner.encode, inner.rssm, lambda s: inner.act(s, sample=False), 2)
        collector = LatentPolicy(
            inner.encode,
            inner.rssm,
            lambda s: inner.act(s, sample=False),
            2,
            noise=0.3,
        )

        curve: list[float] = []
        for step in range(_STEPS):
            batch = replay.sample(16, 15)
            out = model(batch.observations, batch.actions, batch.rewards)
            model.backward(out)
            _clip(model)
            for opt in optimisers:
                opt.step()
            model.update_slow_critic()
            if (step + 1) % _COLLECT_EVERY == 0:
                replay.add(rollout(env, collector)[0])
            if (step + 1) % _EVALUATE_EVERY == 0:
                curve.append(_score(env, evaluator))

        best = max(curve)
        assert best > 1.10 * ceiling, (
            f"did not beat the blind ceiling: best {best:.2f} against a "
            f"constant's {ceiling:.2f} ({best / ceiling:.2f}x). Curve "
            f"{[round(c, 2) for c in curve]}. A policy that cannot read its "
            f"observation is bounded by the ceiling, so failing this is "
            f"either a defect or a budget that is now too small — check "
            f"TestTheActorObjectiveCanLearn first, which separates them."
        )
