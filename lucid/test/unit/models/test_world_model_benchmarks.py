"""Control benchmarks — the claims the suite deliberately cannot make.

``test_world_model_control.py`` runs in seconds and says so about its
limits: on a fixed target the best constant action scores 15.78 against
random's 5.77, so every "beats random" threshold there is clearable by a
policy that never looks at its observation.

This file removes that escape rather than documenting it. The goal moves
each episode, so the question becomes whether the agent *reads the goal*,
and two numbers answer it.

**The goal-blind ceiling.** A policy that never reads the goal earns, at
each step, the reward of its position averaged over the goals — and with
the goal uniform on a square centred at the origin, that average is
largest at the origin. So the best goal-blind policy of any kind
(constant, open-loop plan, or one that reads its own position) walks to
the origin and stops. It is a policy here, scored like the others.

**What the goal is worth to the policy.** The same goals again, each
drawn somewhere else — a decoy — while the reward still follows the real
one. A goal-blind policy scores exactly the same both ways; the
difference is what reading the goal bought, and nothing that ignores the
goal can make it positive.

Both are scored on a fixed panel of 36 goals rather than on random draws,
so each is a function of the policy alone and not of which goals a seed
happened to draw.

Opt in with ``-m control``. Minutes, not seconds, which is why it is
excluded from the default run.

Measured on the panel:

===================================================  ======
policy                                               return
===================================================  ======
random                                                11.93
best constant on a 5x5 grid of actions                13.91
best constant, ``(0.3, -0.3)``                        15.35
goal-blind ceiling (walk to the origin, stop)         16.18
reads the goal from the frame, heads straight to it   19.16
===================================================  ======

An earlier version took the best of that 5x5 grid as the ceiling, each
constant scored on 8 random goals, and asked the best of five 15-episode
evaluations to clear it by 10%. Neither side was what it said. The grid
misses constants that score 15.35 and the policy that scores 16.18, both
past the bar it set, so a policy could pass without reading the goal; and
the two estimates moved with the draw — the ceiling alone from 13.5 to
15.8 across seeds — by as much as the margin. The verdict on DreamerV3
was the draw's: 1.02x on a hosted runner, 1.06x on the Studio with Metal
as the default device, 1.12x with the CPU, one seed and one model.

**What decides a run is a race in its first few hundred steps.** The
actor learns a direction before the world model can tell one goal from
another, and the direction that pays while the goal is unknown is toward
the centre — ``(+1, -1)`` from the start. In 8 of 11 runs of 1000 steps
the actor's mean reached that corner before its output depended on the
state (spread across states 0.003) and stayed: the policy *is* the
constant ``(1, -1)``, which scores 10.54 on the panel and loses nothing
to a decoy. Runs left it after 400 to 1000 steps, one not within 1000,
so at 1000 steps whether a run reads the goal is a draw on when it got
out. At 3000 steps every run measured had, including the two seeds that
were worst at 1000 — which is the budget below.

**The discount here is hostile to Dreamer v1, and that is the task's
doing rather than v1's.** Episodes run 20 steps and ``pcont`` is off, so
imagination never terminates and ``discount=0.99`` asks the critic for an
infinite-horizon value of roughly 70 on a task whose achievable return is
about 14. v2 absorbs the mismatch with its target critic and v3 with its
slow critic and return normalisation. v1 has neither, and its estimate
runs to 1286 while the world model's reward predictions stay accurate;
the actor then maximises a quantity that is running away from it.
Matching the discount to the episode (``discount=0.95``) leaves v1's
value at 15-18, and it cleared the earlier grid ceiling at 1.12x +/- 0.10
over four seeds. So a v1 row added here at 0.99 would measure the
discount, not the implementation. Anyone extending this file to the
earlier families should set the discount from the horizon first, and
measure against the goal-blind ceiling rather than that grid.

What this still does not establish is the paper's own scores. This is a
64x64 toy with a dense Gaussian reward, not the Control Suite, and no
claim about DMC or Atari numbers should be read out of it.
"""

import itertools
import os

import pytest

import lucid
import lucid.models as M
import lucid.nn as nn
import lucid.optim as optim
from lucid.test._default_device import DEFAULT_DEVICE_ENV, DEFAULT_ZOO_DEVICE
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


# Models train on Metal, so this does unless the zoo is being verified on
# another device on purpose (``lucid-audit --models --zoo-device cpu``).
_DEVICE = os.environ.get(DEFAULT_DEVICE_ENV) or DEFAULT_ZOO_DEVICE

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

# Measured, not guessed: 3e-3 on the world model is what moved this from
# "hovers at random" to clearing the old grid ceiling, and 8e-4 the widest actor
# rate that stays stable. At 2e-2 the actor's gradient collapses to 1e-3
# by step 80 and its entropy returns to the uniform limit — which reads
# exactly like a broken objective and is not one. 3e-4 does not avoid the
# corner described in the module docstring: seed 2 reaches it at step 300
# rather than 150.
_WORLD_LR, _VALUE_LR, _ACTOR_LR = 3e-3, 2e-4, 8e-4

_STEPS = 3000
_COLLECT_EVERY = 10
_EVALUATE_EVERY = 500

# What the decoy has to cost the policy at its best evaluation. Set from
# the distribution measured in the test's docstring — the lowest of 12
# runs was 2.35 — not from the run it is applied to. A goal-blind policy's
# cost is exactly 0, the frame reader's 7.8.
_READ = 1.5

# The square the moving target draws from, ``[-0.8, 0.8]^2``, cut into a
# 6x6 grid and scored at the cell midpoints: the panel's mean is a
# quadrature of the expectation over the goal, not a sample of it.
_SIDE = 6
_POINTS = [-0.8 + 1.6 * (i + 0.5) / _SIDE for i in range(_SIDE)]
# Each goal's decoy is the panel point half the grid away on both axes,
# 0.8 off on each, so it is never close to the goal it stands in for.
_PANEL = [
    (
        (_POINTS[i], _POINTS[j]),
        (_POINTS[(i + _SIDE // 2) % _SIDE], _POINTS[(j + _SIDE // 2) % _SIDE]),
    )
    for i in range(_SIDE)
    for j in range(_SIDE)
]
_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _action(values: list[float], observation: lucid.Tensor) -> lucid.Tensor:
    return lucid.tensor([max(-1.0, min(1.0, v)) for v in values], device=observation.device.type)


class _Constant:
    """A policy that cannot see."""

    def __init__(self, action: tuple[float, float]) -> None:
        self.action = list(action)

    def reset(self) -> None:
        return None

    def __call__(self, observation: lucid.Tensor) -> lucid.Tensor:
        return _action(self.action, observation)


class _GoalBlind:
    """Handed its own position and never the goal. The ceiling this file measures against.

    Walks to the origin at full speed and stays there. The expected
    reward at a position falls off with its distance from the origin on
    each axis, and each axis moves on its own, so getting there fastest
    maximises every step's term at once — no policy without the goal
    does better.
    """

    def __init__(self, env: PointMass) -> None:
        self.env = env

    def reset(self) -> None:
        return None

    def __call__(self, observation: lucid.Tensor) -> lucid.Tensor:
        return _action([-x / self.env.step_size for x in self.env.position], observation)


class _ReadsTheFrame:
    """Finds the agent and the goal in the frame and heads straight at the goal.

    It reads what the agent is given and nothing else, so a decoy misleads
    it — which is what shows the decoy measures reading.
    """

    def reset(self) -> None:
        return None

    def __call__(self, observation: lucid.Tensor) -> lucid.Tensor:
        size = int(observation.shape[-1])
        peaks = lucid.argmax(observation[:2].reshape(2, -1), dim=1).tolist()
        (agent_y, agent_x), (goal_y, goal_x) = (divmod(int(p), size) for p in peaks)
        pixel = 2.0 / (size - 1)
        toward = [10.0 * (goal_x - agent_x) * pixel, 10.0 * (goal_y - agent_y) * pixel]
        return _action(toward, observation)


def _panel_return(env: PointMass, policy: object, *, decoy: bool = False) -> float:
    """Mean return over the panel, each goal drawn where it is or at its decoy."""
    total = 0.0
    for goal, elsewhere in _PANEL:
        env.place(goal, elsewhere if decoy else None)
        total += rollout(env, policy)[1]
    return total / len(_PANEL)


# The reference policies are scored on the CPU: their returns are Python
# arithmetic on positions, identical on any device, and rendering 36
# episodes per constant on an accelerator buys nothing.
@pytest.fixture(scope="module")
def reference_panel() -> PointMass:
    return PointMass(horizon=20, device="cpu")


@pytest.fixture(scope="module")
def goal_blind_ceiling(reference_panel: PointMass) -> float:
    return _panel_return(reference_panel, _GoalBlind(reference_panel))


class TestTheTaskDeniesBlindPolicies:
    """Guards the benchmark below. If a blind policy can pass, it measures nothing."""

    def test_no_constant_reaches_the_goal_blind_ceiling(
        self, reference_panel: PointMass, goal_blind_ceiling: float
    ) -> None:
        constants = [*itertools.product(_GRID, repeat=2), (0.3, -0.3)]
        best = max(_panel_return(reference_panel, _Constant(a)) for a in constants)
        assert best < goal_blind_ceiling, (
            f"a constant scores {best:.2f} against walking to the origin's "
            f"{goal_blind_ceiling:.2f} — that is no longer the best a "
            f"goal-blind policy can do, and the bar below is not a ceiling"
        )

    def test_a_decoy_costs_a_blind_policy_nothing(
        self, reference_panel: PointMass, goal_blind_ceiling: float
    ) -> None:
        blind = _GoalBlind(reference_panel)
        assert _panel_return(reference_panel, blind, decoy=True) == goal_blind_ceiling
        constant = _Constant((0.5, -0.5))
        shown = _panel_return(reference_panel, constant)
        assert _panel_return(reference_panel, constant, decoy=True) == shown

    def test_a_decoy_misleads_a_policy_that_reads_the_goal(
        self, reference_panel: PointMass
    ) -> None:
        reader = _ReadsTheFrame()
        shown = _panel_return(reference_panel, reader)
        assert shown - _panel_return(reference_panel, reader, decoy=True) > 5.0

    def test_reading_the_goal_is_worth_much_more(
        self, reference_panel: PointMass, goal_blind_ceiling: float
    ) -> None:
        """There has to be headroom above the ceiling, or nothing can pass."""
        assert _panel_return(reference_panel, _ReadsTheFrame()) > 1.15 * goal_blind_ceiling

    def test_the_fixed_target_still_admits_one(self) -> None:
        """The contrast, so the difference is the variant and not the seed."""
        lucid.manual_seed(0)
        fixed = PointMass(horizon=20, device=_DEVICE)
        best = max(rollout(fixed, _Constant(a))[1] for a in itertools.product(_GRID, repeat=2))
        random_return = sum(rollout(fixed, RandomPolicy(2))[1] for _ in range(15)) / 15
        assert best > 2.0 * random_return


class TestDreamerV3ReadsTheGoal:
    """State-dependent control, on the only task here that can show it."""

    def test_it_learns_to_read_the_goal(self, goal_blind_ceiling: float) -> None:
        """Train 3000 steps from random episodes; ask what the goal is worth to the policy.

        Measured on the Mac Studio (M4 Max), six evaluations per run — the
        decoy's cost at the best of them, and the best panel return:

        ====================  ====  =====================  =====================
        model on (default)    runs  decoy cost, low / mid  best return, low / mid
        ====================  ====  =====================  =====================
        Metal (CPU)           8     3.67 / 5.33            16.48 / 16.76
        Metal (Metal)         2     2.35 / 2.81            13.27 / 14.76
        CPU (CPU)             2     4.71 / 6.30            16.06 / 16.19
        ====================  ====  =====================  =====================

        The cost is asserted, the return is not. Every run read the goal;
        10 of 12 also beat walking to the origin (16.18). Of the other two,
        seed 1 with Metal as the default peaked at 13.27 and then sat below
        random while still paying for a decoy. A policy that reads the goal
        and acts on it badly is what the early corner leaves behind, and
        how long it takes to undo is the same lottery as the corner itself,
        so a bar on the return would be a bar on the seed.

        About 3.5 minutes on Metal and 8 on the CPU there.
        """
        lucid.manual_seed(0)
        env = PointMass(horizon=20, random_target=True, device=_DEVICE)
        panel = PointMass(horizon=20, device=_DEVICE)
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

        inner = model.dreamer_v3
        evaluator = LatentPolicy(inner.encode, inner.rssm, lambda s: inner.act(s, sample=False), 2)
        collector = LatentPolicy(
            inner.encode, inner.rssm, lambda s: inner.act(s, sample=False), 2, noise=0.3
        )

        shown: list[float] = []
        misled: list[float] = []
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
                # Neither the placed panel nor the noiseless evaluator draws
                # from the generator, so evaluating does not move training.
                shown.append(_panel_return(panel, evaluator))
                misled.append(_panel_return(panel, evaluator, decoy=True))

        # Each evaluation is exact — no goal is drawn — so the best of them
        # is a property of a policy the run actually reached, not the luck
        # of a sample; a goal-blind policy scores 0 at every one.
        read = max(s - m for s, m in zip(shown, misled, strict=True))
        curve = [(round(s, 2), round(m, 2)) for s, m in zip(shown, misled, strict=True)]
        assert read > _READ, (
            f"the policy never read the goal: at its best evaluation a decoy "
            f"cost it {read:.2f}, where a goal-blind policy loses exactly 0 "
            f"(needs {_READ}). Walking to the origin scores "
            f"{goal_blind_ceiling:.2f}; (goal shown, decoy shown) per "
            f"evaluation: {curve}"
        )
