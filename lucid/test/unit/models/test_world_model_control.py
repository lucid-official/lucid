"""Does a world model learn to act at all?

Everything else in the suite checks the parts. Shapes are right, gradients
reach the modules they should, losses fall, lambda-returns match the
paper's closed form. None of it would catch a recurrence that carries the
wrong state forward, because a model like that still reconstructs frames
and still reports a shrinking loss.

So this closes the loop: seed a replay buffer, train the three objectives,
collect with the policy being learned, and check the return beats random.

**What this does not establish.** The target here is fixed, so a policy
that ignores its observation entirely and pushes in one direction already
scores 15.78 against random's expected 6.21 — measured, and pinned below.
Passing these therefore shows the agent found a good *constant* action,
not that it learned to read a state and respond to it.

**Every bar is a property of the task, not of one draw** (LCD-872). The
tests used to compare against a five-episode random draw and a reward band
from one more random episode, both taken after seeding. Those draws differ
per device for the same seed — Metal keys MLX's sampler from the generator
where the CPU fills element by element — so the bars moved with the device
while the learning did not: Metal arithmetic fed the CPU's draws matches
the CPU run to about 1e-7 at the first step and to five decimals after
sixty, on every seed measured. Over 20 seeds the old DreamerV3 and PlaNet
bars alone failed 8 CPU seeds and 11 Metal seeds in one test or another,
and seed 0 happened to be one of the Metal ones. The bars now come from the
expectation (``_RANDOM_RETURN``), the rewards each model trained on
(``_reward_range``) and a shifted-reward control, so a seed's draws change
the run but never the bar it is held to.

State-dependent control **is** achieved, but not here and not this
cheaply. On a variant with the target moved each episode, where no
constant action works, DreamerV2 with proper interleaved collection
reaches 2.34x the best constant policy and 71% of the greedy optimum —
8000 gradient steps, seventeen minutes on Metal. That is a benchmark, not
a unit test, so what lives in the suite is this cheaper check with its
limitation written down rather than assumed away.

The tests are kept because "finds the best constant action" is still far
more than the rest of the suite establishes — a mis-wired recurrence
fails it — but the name of the file should not be read as more than that.

Cost, measured on an M1 Pro: about 45 seconds for both families.
"""

import math

import pytest

import lucid
import lucid.models as M
import lucid.nn as nn
import lucid.optim as optim
from lucid.models.generative._common._actor import Actor
from lucid.test._fixtures.classic_control import Pendulum
from lucid.test._fixtures.point_mass import PointMass
from lucid.utils.rollout import LatentPolicy, RandomPolicy, SequenceReplay, StepResult, rollout

pytestmark = pytest.mark.slow


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


_TINY = dict(
    action_dim=2,
    cnn_depth=8,
    stoch_size=8,
    deter_size=32,
    hidden_size=32,
    actor_hidden=32,
    value_hidden=32,
    reward_hidden=32,
    horizon=8,
)

#: Twice the 120 this used to run, for the reason in ``_train_dreamer``.
TRAIN_STEPS = 240
SEED_EPISODES = 5
BATCH, LENGTH = 8, 15

#: What a uniform random policy is expected to score on ``PointMass(horizon=20)``
#: — 6.2113 +- 0.0030 over 400,000 episodes of the same dynamics, and pinned
#: against the fixture itself in :class:`TestWhatTheThresholdActuallyRequires`.
#: Every "beats random" bar here is scaled from this rather than from a
#: five-episode draw: that draw has a standard deviation of 0.84, its largest
#: value over 200 Metal seeds (9.92) fell on seed 0, and its draws differ per
#: device, so the bar itself moved with the device (LCD-872).
_RANDOM_RETURN = 6.21
#: One random episode's standard deviation on the same task, measured with it.
_RANDOM_RETURN_SD = 1.87


def _policy(model: object, noise: float = 0.0) -> LatentPolicy:
    return LatentPolicy(
        model.encode, model.rssm, lambda state: model.act(state, sample=False), 2, noise=noise
    )


def _average_return(env: PointMass, policy: object, episodes: int = 2) -> float:
    return sum(rollout(env, policy)[1] for _ in range(episodes)) / episodes


class _FixedAction:
    """A policy that ignores what it sees — the thing to be beaten."""

    def __init__(self, action: lucid.Tensor) -> None:
        self.action = action

    def reset(self) -> None:
        pass

    def __call__(self, observation: lucid.Tensor) -> lucid.Tensor:
        return self.action


class _PendulumOracle:
    """The fixture's own reference controller, for a solvable return."""

    def __init__(self, env: Pendulum) -> None:
        self.env = env

    def reset(self) -> None:
        pass

    def __call__(self, observation: lucid.Tensor) -> lucid.Tensor:
        return self.env.optimal_action()


class _Greedy:
    """Straight at the target — what a solved task looks like."""

    def __init__(self, env: PointMass) -> None:
        self.env = env

    def reset(self) -> None:
        return None

    def __call__(self, observation: lucid.Tensor) -> lucid.Tensor:
        return self.env.optimal_action()


class TestPointMassEnvironment:
    """Guard the instrument before trusting what it measures."""

    def test_greedy_beats_random_by_a_wide_margin(self) -> None:
        """If it did not, the learning test could not distinguish anything."""
        env = PointMass()
        greedy_return = _average_return(env, _Greedy(env), episodes=2)
        assert greedy_return > 2.5 * _RANDOM_RETURN

    def test_reward_is_dense(self) -> None:
        """A sparse reward would measure luck over a run this short."""
        env = PointMass()
        env.reset()
        far = env.step(lucid.tensor([-1.0, 1.0])).reward
        assert far > 0.05

    def test_observation_shows_agent_and_target_apart(self) -> None:
        env = PointMass()
        observation = env.reset()
        assert observation.shape == (3, 64, 64)
        agent, target = observation[0], observation[1]
        assert float(agent.max().item()) > 0.9
        assert float(target.max().item()) > 0.9
        # Different places, or the task would be solved at t=0.
        assert float((agent * target).max().item()) < 0.5


class TestWhatTheThresholdActuallyRequires:
    """Pins the limitation of the tests below, so it cannot be forgotten.

    A threshold of 1.3x random reads like evidence of control. On a fixed
    target it is not: the numbers here show a constant action clearing it
    outright. Keeping the measurement in the tree is the difference
    between a known limitation and a wrong claim.
    """

    class _Constant:
        def __init__(self, value: list[float]) -> None:
            self.value = lucid.tensor(value)

        def reset(self) -> None:
            return None

        def __call__(self, observation: lucid.Tensor) -> lucid.Tensor:
            return self.value

    def test_the_random_return_is_what_the_bars_are_scaled_from(self) -> None:
        """Pins ``_RANDOM_RETURN`` against the fixture, to four standard errors.

        Drawn on the CPU from its own seed: the estimate is a property of
        the task, and a CPU frame costs a seventh of a Metal one.
        """
        lucid.manual_seed(0)
        env = PointMass(horizon=20, device="cpu")
        episodes = 400
        estimate = _average_return(env, RandomPolicy(2), episodes=episodes)
        tolerance = 4.0 * _RANDOM_RETURN_SD / math.sqrt(episodes)
        assert abs(estimate - _RANDOM_RETURN) < tolerance, (
            f"random scores {estimate:.3f} over {episodes} episodes, not "
            f"{_RANDOM_RETURN} +- {tolerance:.2f} — the fixture changed"
        )

    def test_a_constant_action_already_clears_the_bar(self) -> None:
        env = PointMass(horizon=20)
        best = max(
            _average_return(env, self._Constant(v), episodes=1)
            for v in ([1.0, -1.0], [1.0, 0.0], [0.0, -1.0], [0.7, -0.7])
        )
        assert best > 1.3 * _RANDOM_RETURN, (
            "if this ever fails the tests below became stronger than "
            "documented, and the module docstring needs revisiting"
        )

    def test_the_optimal_policy_is_still_better(self) -> None:
        """There is headroom above the constant — the task is not degenerate."""
        env = PointMass(horizon=20)
        greedy = _average_return(env, _Greedy(env), episodes=2)
        best_constant = _average_return(env, self._Constant([1.0, -1.0]), episodes=1)
        assert greedy > best_constant


def _train_dreamer(seed: int = 0) -> list[float]:
    """Dreamer on the fixed target; the greedy evaluator's return every 40 steps.

    At the paper's rates 120 steps is a coin: over 12 seeds 7 cleared the
    bar on the CPU and 5 on Metal, the rest still near random. 240 clears
    it on 19 of 20 CPU seeds and 17 of 20 Metal ones (best 6.2-15.7,
    median 15.0-15.4). The faster actor DreamerV2 needs does not help
    here: at 8e-4 over 120 steps 15 and 16 of 20 cleared it, and every
    miss sat on a wrong corner of the action box at some evaluation —
    5.86 is the return of a constant (1, 1), 2.25 that of (-1, 1).
    """
    lucid.manual_seed(seed)
    env = PointMass(horizon=20)
    model = M.create_model("dreamer_world_model", **_TINY)
    optimisers = [
        optim.Adam(model.world_parameters(), lr=6e-4),
        optim.Adam(model.value_parameters(), lr=8e-5),
        optim.Adam(model.actor_parameters(), lr=8e-5),
    ]

    replay = SequenceReplay(capacity=20_000)
    for _ in range(SEED_EPISODES):
        replay.add(rollout(env, RandomPolicy(2))[0])

    collector = _policy(model.dreamer, noise=0.3)
    evaluator = _policy(model.dreamer)

    curve = []
    for step in range(TRAIN_STEPS):
        batch = replay.sample(BATCH, LENGTH)
        out = model(batch.observations, batch.actions, batch.rewards)
        model.backward(out)
        _clip(model)
        for opt in optimisers:
            opt.step()

        if (step + 1) % 40 == 0:
            replay.add(rollout(env, collector)[0])
            curve.append(_average_return(env, evaluator))
    return curve


class TestDreamerLearnsControl:
    def test_return_improves_over_random(self) -> None:
        curve = _train_dreamer()
        best = max(curve)
        assert best > 1.3 * _RANDOM_RETURN, (
            f"policy did not learn: best {best:.2f} vs random {_RANDOM_RETURN}, "
            f"curve {[round(c, 2) for c in curve]}"
        )
        assert curve[-1] > curve[0], f"return did not trend upward: {curve}"

    def test_collected_episodes_feed_back_into_the_buffer(self) -> None:
        """The loop is closed — otherwise it is just offline training."""
        lucid.manual_seed(0)
        env = PointMass(horizon=10)
        model = M.create_model("dreamer_world_model", **_TINY)
        replay = SequenceReplay()
        for _ in range(2):
            replay.add(rollout(env, RandomPolicy(2))[0])
        before = replay.steps
        replay.add(rollout(env, _policy(model.dreamer, noise=0.3))[0])
        assert replay.steps > before

    def test_exploration_noise_changes_what_is_collected(self) -> None:
        lucid.manual_seed(0)
        env = PointMass(horizon=10)
        model = M.create_model("dreamer", **_TINY).eval()
        quiet = rollout(env, _policy(model))[0]
        loud = rollout(env, _policy(model, noise=0.5))[0]
        assert not bool((quiet.actions == loud.actions).all().item())


class _Cliff:
    """Terminates rather than truncating, so ``discounts`` reaches zero."""

    def __init__(self, at: int = 5) -> None:
        self.at = at

    def reset(self) -> lucid.Tensor:
        self.t = 0
        return lucid.rand((3, 64, 64))

    def step(self, action: lucid.Tensor) -> StepResult:
        self.t += 1
        return StepResult(lucid.rand((3, 64, 64)), 1.0, self.t >= self.at, False)


class TestDiscountHeadEndToEnd:
    """Termination has to survive the whole path, not just each end of it.

    ``pcont`` is tested against hand-written discounts elsewhere, and the
    driver's termination flag is tested without a model. This is the seam
    between them: an environment that really ends, through a real rollout,
    into a real buffer, into the loss.
    """

    def test_termination_reaches_the_loss(self) -> None:
        lucid.manual_seed(0)
        model = M.create_model("dreamer_world_model", pcont=True, **_TINY)
        replay = SequenceReplay()
        for _ in range(4):
            replay.add(rollout(_Cliff(5), RandomPolicy(2))[0])

        batch = replay.sample(4, 5)
        assert float(batch.discounts[0][-1]) == 0.0, "termination lost in the buffer"

        out = model(batch.observations, batch.actions, batch.rewards, batch.discounts)
        assert out.pcont_loss is not None
        assert out.behavior is not None and out.behavior.imagined_pcont is not None

        model.backward(out)
        grads = [p.grad for p in model.dreamer.pcont_head.parameters() if p.grad is not None]
        assert grads and sum(float(g.abs().sum().item()) for g in grads) > 0

    def test_a_truncating_environment_leaves_the_discount_alone(self) -> None:
        """PointMass only ever truncates; nothing should read as terminal."""
        replay = SequenceReplay()
        replay.add(rollout(PointMass(horizon=8), RandomPolicy(2))[0])
        batch = replay.sample(2, 8)
        assert bool((batch.discounts == 1.0).all().item())


_TINY_V2 = dict(
    action_dim=2,
    cnn_depth=4,
    stoch_size=6,
    discrete=6,
    deter_size=24,
    hidden_size=24,
    actor_hidden=32,
    value_hidden=32,
    reward_hidden=32,
    horizon=6,
    pcont=False,
)


def _v2_policy(model: object, noise: float = 0.0) -> LatentPolicy:
    return LatentPolicy(
        model.encode, model.rssm, lambda state: model.act(state, sample=False), 2, noise=noise
    )


def _train_dreamer_v2(seed: int = 0) -> tuple[list[float], float, float]:
    """DreamerV2 on the fixed target: the return curve, first and last policy entropy."""
    lucid.manual_seed(seed)
    env = PointMass(horizon=20)
    model = M.create_model("dreamer_v2_world_model", **_TINY_V2)
    optimisers = [
        optim.Adam(model.world_parameters(), lr=6e-4),
        optim.Adam(model.value_parameters(), lr=2e-4),
        # 8e-4, not the released 8e-5. DreamerV2's policy starts at
        # 99% of the maximum entropy the interval allows — std is
        # 2*sigmoid(0) + min_std ~= 1.1, which on [-1, 1] is nearly
        # uniform, so the mean barely moves the sample. Sharpening it
        # is most of the learning, and the published rate is set for
        # millions of steps rather than the sixty here. Measured: at
        # 8e-5 the policy is still uniform after 120 steps and scores
        # 0.70x random; at 8e-4 it reaches 2.9x by step 50.
        optim.Adam(model.actor_parameters(), lr=8e-4),
    ]

    replay = SequenceReplay(capacity=20_000)
    for _ in range(SEED_EPISODES):
        replay.add(rollout(env, RandomPolicy(2))[0])

    inner = model.dreamer_v2
    collector = _v2_policy(inner, noise=0.3)
    evaluator = _v2_policy(inner)

    curve = []
    entropies = []
    for step in range(60):
        batch = replay.sample(BATCH, LENGTH)
        out = model(batch.observations, batch.actions, batch.rewards)
        assert out.behavior is not None
        entropies.append(float(out.behavior.entropy.item()))
        model.backward(out)
        _clip(model)
        for opt in optimisers:
            opt.step()
        model.update_slow_target()
        if (step + 1) % 30 == 0:
            replay.add(rollout(env, collector)[0])
            curve.append(_average_return(env, evaluator))
    return curve, entropies[0], entropies[-1]


class TestDreamerV2LearnsControl:
    """The same question asked of the discrete latent.

    Worth asking separately rather than trusting the shared parts: the
    latent, the divergence, the critic's target and the policy's
    distribution are all different, and every one of them can be wired in
    a way that trains happily and controls nothing.

    **At this budget the run is a coin** (measured for LCD-872). Over 20
    seeds it passes on 9 CPU seeds and 11 Metal ones. Most misses sit on
    a wrong corner of the action box — 5.86 is a constant (1, 1), 2.25-2.4
    one of (-1, 1) — the early saturation LCD-870 tracks for DreamerV3;
    the rest did not sharpen by 0.04. Seed 0 passes on both devices, so
    this passes, but it holds the family to less than its name says
    until that is understood.
    """

    def test_return_improves_over_random(self) -> None:
        curve, entropy_first, entropy_last = _train_dreamer_v2()
        best = max(curve)
        assert best > 1.3 * _RANDOM_RETURN, (
            f"policy did not learn: best {best:.2f} vs random {_RANDOM_RETURN}, "
            f"curve {[round(c, 2) for c in curve]}"
        )
        # The mechanism, on the same run rather than a second one. A
        # policy that never leaves the uniform limit scores exactly
        # random however good the world model gets — which is what the
        # released actor rate produces over sixty steps.
        assert (
            entropy_last < entropy_first - 0.04
        ), f"policy never sharpened: {entropy_first:.3f} -> {entropy_last:.3f}"

    def test_it_starts_near_the_uniform_limit(self) -> None:
        """Pins the starting point the test above measures movement from."""
        model = M.create_model("dreamer_v2", **_TINY_V2).eval()
        feature = lucid.randn((4, 3, model.config.latent_size))
        entropy = float(model.actor.distribution(feature).entropy().mean().item())
        assert entropy > 0.9 * math.log(2.0)


class TestPendulumEnvironment:
    """The swing-up, and the two properties that make it worth having.

    The point-mass task above is honest about its ceiling: a constant
    action clears its threshold, so passing it shows an agent found a good
    constant, not that it reads a state. The pendulum removes that escape
    by physics rather than by tuning — the torque limit is a fraction of
    the gravitational torque, so no constant lifts the rod at all.

    The physics itself is checked against an invariant, not against a
    citation. An equation copied wrong still produces plausible motion; it
    does not conserve energy.
    """

    @staticmethod
    def _drift(dt: float, seconds: float = 6.0) -> float:
        env = Pendulum(horizon=10**6, dt=dt)
        env.reset()
        env.theta, env.omega = 2.0, 0.5  # away from equilibrium, so there is motion
        start = env.energy()
        worst = 0.0
        zero = lucid.tensor([0.0])
        for _ in range(int(seconds / dt)):
            env.step(zero)
            worst = max(worst, abs(env.energy() - start))
        return worst

    def test_unforced_motion_conserves_energy(self) -> None:
        """The check a citation cannot give: a wrong equation would drift."""
        env = Pendulum(horizon=10**6)
        env.reset()
        env.theta, env.omega = 2.0, 0.5
        start = env.energy()
        assert abs(self._drift(0.05)) / abs(start) < 1e-3

    def test_the_residual_is_truncation_and_shrinks_with_the_step(self) -> None:
        """Guards the test above.

        A tolerance alone would pass on a systematically wrong equation
        that happens to drift slowly. Truncation error falls with the step
        size; a wrong equation does not.
        """
        assert self._drift(0.02) < self._drift(0.08) / 4.0

    def test_the_torque_cannot_lift_the_rod(self) -> None:
        """Underactuation, as a property of the numbers rather than a hope."""
        env = Pendulum()
        peak_gravity_torque = env.mass * env.gravity * env.length / 2.0
        assert env.max_torque < peak_gravity_torque

    def test_no_constant_action_solves_it(self) -> None:
        """The property the point-mass task lacks.

        Measured: the best constant scores 6.02 where the oracle scores
        77.41. This is what lets a return on this task mean the agent is
        responding to what it sees.
        """
        env = Pendulum()
        best = max(
            rollout(env, _FixedAction(lucid.tensor([v])))[1] for v in (-1.0, -0.5, 0.0, 0.5, 1.0)
        )
        oracle = rollout(env, _PendulumOracle(env))[1]
        assert oracle > 5.0 * best

    def test_the_oracle_actually_swings_it_up(self) -> None:
        """Solvable, and by the controller the fixture ships."""
        env = Pendulum(horizon=200)
        env.reset()
        for _ in range(200):
            env.step(env.optimal_action())
        assert math.cos(env.theta) > 0.95

    def test_the_oracle_does_not_rely_on_floating_point_noise(self) -> None:
        """It starts at rest, where energy pumping is exactly zero.

        An earlier version bootstrapped off ``sin(pi)`` evaluating to
        1.2e-16 and would have stalled for ever on a system that reached
        the resting state exactly.
        """
        env = Pendulum()
        env.reset()
        assert abs(float(env.optimal_action()[0].item())) > 0.5

    def test_the_frame_moves_with_the_angle(self) -> None:
        """The rod's pixels are the only place the angle is observable."""
        env = Pendulum()
        hanging = env.reset()
        env.theta = 0.0
        upright = env._render()
        assert float((upright - hanging).abs().max().item()) > 0.5

    @pytest.mark.parametrize("device", ["cpu", "metal"])
    def test_it_renders_on_the_requested_device(self, device: str) -> None:
        env = Pendulum(device=device)
        assert str(env.reset().device) == f"device('{device}')"

    def test_it_satisfies_the_rollout_protocol(self) -> None:
        episode, total = rollout(Pendulum(horizon=10), RandomPolicy(1))
        assert episode.observations.shape == (10, 3, 64, 64)
        assert isinstance(total, float)

    @pytest.mark.parametrize("torque", [0.0, 1.0, 1.5])
    def test_it_rejects_a_torque_that_removes_the_challenge(self, torque: float) -> None:
        with pytest.raises(ValueError):
            Pendulum(torque=torque)


_TINY_V3 = dict(
    action_dim=2,
    cnn_depth=4,
    stoch_size=6,
    discrete=6,
    deter_size=32,
    hidden_size=24,
    blocks=8,
    actor_hidden=32,
    value_hidden=32,
    reward_hidden=32,
    horizon=6,
    num_bins=41,
    pcont=False,
)


#: Added to every reward a counterfactual run is trained on.  Real rewards
#: are ``exp(-d^2 / 2)`` on a box, so they lie in ``(0.105, 1]`` and a shift
#: of 1 moves the whole band clear of where it was.
_REWARD_SHIFT = 1.0


def _reward_range(replay: SequenceReplay) -> tuple[float, float]:
    """The rewards the world model was trained on — where real reward lives for it.

    Not a fresh episode drawn after training. That one was the old band,
    and over 20 seeds it put a calibrated model outside it on 2 Metal and
    2 CPU seeds (DreamerV3 and PlaNet): an episode that wandered close to
    the target gave a band of ``[0.375, 0.927]`` against a model whose
    training batch averaged 0.244, and it read 0.231.
    """
    rewards = lucid.cat([episode.rewards for episode in replay.episodes], dim=0)
    return float(rewards.min().item()), float(rewards.max().item())


def _train_v3(steps: int = 60, world: bool = True, seed: int = 0, shift: float = 0.0) -> tuple:
    """One short end-to-end run; returns the pieces the tests read.

    ``shift`` is added to the rewards the model is trained on, and to
    nothing else — the buffer keeps the task's own.
    """
    lucid.manual_seed(seed)
    env = PointMass(horizon=20)
    model = M.create_model("dreamer_v3_12m_world_model", **_TINY_V3)
    world_opt = optim.Adam(model.world_parameters(), lr=3e-3)
    value_opt = optim.Adam(model.value_parameters(), lr=2e-4)
    actor_opt = optim.Adam(model.actor_parameters(), lr=8e-4)

    replay = SequenceReplay(capacity=20_000)
    for _ in range(SEED_EPISODES):
        replay.add(rollout(env, RandomPolicy(2))[0])

    out = None
    for _ in range(steps):
        batch = replay.sample(BATCH, LENGTH)
        out = model(batch.observations, batch.actions, batch.rewards + shift)
        model.backward(out)
        _clip(model)
        if world:
            world_opt.step()
        value_opt.step()
        actor_opt.step()
        model.update_slow_critic()
    return model, env, replay, out


class TestDreamerV3EndToEnd:
    """v3 driven through the rollout layer, which nothing else does.

    **What this establishes.** That the loop closes and that the world
    model learns the task's reward from episodes it collected itself.
    The last part is the load-bearing one: the reward head is a
    distribution over symlog bins rather than a scalar, so "the number
    came out right" passes through two-hot encoding, a categorical
    cross-entropy and ``symexp`` — none of which a shape test exercises
    and all of which can be wired backwards while still training.

    **What it does not establish.** That the policy learns. Not because
    it does not — at ``lr`` 8e-4 on the actor and 3e-3 on the world
    model it reaches 2.16-2.84x random on five seeds out of five, and the
    imagined lambda-return it is actually maximising climbs from about
    zero to 3.4-5.6 on every one — but because on this task that number
    means less than it looks. The best constant action scores 15.78
    against random's expected 6.21, and 15.78 is precisely where all five runs
    land. So the honest reading is "found a good constant", the same
    reading the file's header gives v1 and v2.

    Two rates below and one above that window do *not* learn, which is
    worth knowing before reading a failure here as a defect: at 2e-2 the
    actor's gradient collapses to 1e-3 by step 80 with its entropy back
    at the uniform limit, and 200 gradient steps is short of the ~300
    this needs. Both were mistaken for a broken actor-critic until the
    objective was tested on its own, where a known advantage moves the
    continuous mode from -0.08 to +1.0000 inside 100 steps.

    State-dependent control needs a task no constant can win, so it is
    measured in ``test_world_model_benchmarks.py`` on the moving-target
    variant rather than asserted here.

    This is the seam that hid a real defect: episodes collected from an
    accelerator environment split across devices, because rewards arrive
    as Python floats and landed on the CPU while the frames did not. The
    rollout layer is tested without a model and the models are tested
    on-device without the rollout layer, so nothing saw it.
    """

    def test_the_collected_loop_closes(self) -> None:
        model, env, replay, _ = _train_v3(steps=10)
        before = replay.steps
        inner = model.dreamer_v3
        collector = LatentPolicy(
            inner.encode, inner.rssm, lambda state: inner.act(state, sample=False), 2, noise=0.3
        )
        replay.add(rollout(env, collector)[0])
        assert replay.steps > before

    def test_all_three_objectives_receive_gradient(self) -> None:
        """Off real collected data, not a synthetic batch."""
        model, _, _, _ = _train_v3(steps=10)
        for name, group in (
            ("world", model.world_parameters()),
            ("value", model.value_parameters()),
            ("actor", model.actor_parameters()),
        ):
            total = sum(float(p.grad.abs().sum().item()) for p in group if p.grad is not None)
            assert total > 0.0, f"{name} received no gradient"

    def test_the_reward_head_calibrates_to_the_task(self) -> None:
        """Imagined reward has to land where real reward lives."""
        _, _, replay, out = _train_v3()
        low, high = _reward_range(replay)
        imagined = float(out.behavior.imagined_reward.mean().item())
        assert low <= imagined <= high, (
            f"imagined reward {imagined:.3f} outside the rewards it was trained on "
            f"[{low:.3f}, {high:.3f}]"
        )

    def test_the_reading_follows_the_reward_it_was_trained_on(self) -> None:
        """Guards the test above.

        Rewards here are ``exp(-d^2)``, so they sit in a narrow positive
        band that an untrained head could fall inside by accident — which
        would make the check above pass without the model having learned
        anything. It does: this head is not zero-initialised, and over 20
        seeds an untrained one read -0.39 to 0.75, inside the band on 6 of
        20 seeds on the CPU and on Metal alike. Withholding the world
        optimiser, the guard this replaces, was therefore a draw on the
        initialisation, and seed 0 lost it on Metal only (LCD-872).

        So the guard moves the target instead of withholding the training:
        the same run on rewards shifted by ``_REWARD_SHIFT`` must read the
        shifted band, which no initialisation reaches by accident and
        which lies wholly outside the real one.
        """
        _, _, replay, out = _train_v3(shift=_REWARD_SHIFT)
        low, high = _reward_range(replay)
        assert high < low + _REWARD_SHIFT, "the shifted band overlaps the real one"
        imagined = float(out.behavior.imagined_reward.mean().item())
        assert low + _REWARD_SHIFT <= imagined <= high + _REWARD_SHIFT, (
            f"trained on rewards shifted by {_REWARD_SHIFT}, the head reads "
            f"{imagined:.3f} — not the shifted band "
            f"[{low + _REWARD_SHIFT:.3f}, {high + _REWARD_SHIFT:.3f}]"
        )


_TINY_PLANET = dict(
    action_dim=2, cnn_depth=4, stoch_size=8, deter_size=32, hidden_size=32, reward_hidden=32
)

# Far below the paper's 12/10/1000/100. The planner is the policy, so it
# runs once per environment step rather than once per training step, and
# the paper's budget makes a twenty-step episode the dominant cost of the
# whole file.
_PLAN = dict(horizon=6, iterations=4, candidates=64, elites=8)


def _train_planet(steps: int = 60, train: bool = True, seed: int = 0, shift: float = 0.0) -> tuple:
    """As :func:`_train_v3`, for PlaNet's single objective."""
    lucid.manual_seed(seed)
    env = PointMass(horizon=20)
    model = M.create_model("planet_world_model", **_TINY_PLANET)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    replay = SequenceReplay(capacity=20_000)
    for _ in range(SEED_EPISODES):
        replay.add(rollout(env, RandomPolicy(2))[0])

    first = out = None
    for step in range(steps):
        # Every step's own gradient. Without this they summed across all
        # sixty and Adam stepped along the running total: over 20 seeds
        # the reward head then read 0.02-0.40 against batches averaging
        # 0.20-0.59, and a fresh gradient each step reads 0.20-0.44.
        opt.zero_grad()
        batch = replay.sample(BATCH, LENGTH)
        out = model(batch.observations, batch.actions, batch.rewards + shift)
        first = float(out.recon_loss.item()) if step == 0 else first
        out.loss.backward()
        # PlaNet plans with CEM rather than an actor-critic, so the world
        # model's cap is the only one that applies.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0)
        if train:
            opt.step()
    return model, env, replay, out, first


def _planner(model: object) -> LatentPolicy:
    return LatentPolicy(
        model.planet.encode, model.planet.rssm, lambda state: model.plan(state, **_PLAN), 2
    )


class TestPlaNetPlansThroughTheLoop:
    """CEM as the policy, in a real episode.

    ``plan`` is unit-tested against a fixed state elsewhere. What was
    never done is the thing PlaNet actually is: no actor, the planner
    *is* the policy, so every environment step runs a fresh search
    against the learned model and feeds its own result back as the next
    belief. A planner that silently searches a stale or mis-shaped state
    still returns a well-formed action.

    Return is not asserted, and here the reason is sharper than for the
    actor: on the fixed-target task an **untrained** planner scored 2.00x
    random on one seed, and on another the trained one scored below the
    untrained. CEM searches hard enough to stumble onto a good constant
    through a model that has learned nothing, so on a task a constant can
    win, the score says nothing about the model. Ranking a planner needs
    the moving-target variant, so that lives in the benchmarks.
    """

    def test_the_planner_drives_a_real_episode(self) -> None:
        model, env, replay, _, _ = _train_planet(steps=10)
        before = replay.steps
        episode, total = rollout(env, _planner(model))
        replay.add(episode)
        assert replay.steps > before
        assert isinstance(total, float)

    def test_the_planner_respects_the_action_bound(self) -> None:
        """CEM searches an unbounded Gaussian; the driver is what clips.

        Worth asserting on a planned episode rather than a single call —
        the bound is applied where the action leaves the policy, and an
        actor-based model would satisfy it through its own tanh and hide
        a missing clip.
        """
        model, env, _, _, _ = _train_planet(steps=10)
        episode = rollout(env, _planner(model))[0]
        assert float(episode.actions.abs().max().item()) <= 1.0

    def test_the_reward_head_calibrates_to_the_task(self) -> None:
        _, _, replay, out, first = _train_planet()
        low, high = _reward_range(replay)
        predicted = float(out.reward.mean().item())
        assert (
            low <= predicted <= high
        ), f"predicted reward {predicted:.3f} outside [{low:.3f}, {high:.3f}]"
        assert float(out.recon_loss.item()) < first

    def test_the_reading_follows_the_reward_it_was_trained_on(self) -> None:
        """Guards the reward half of the test above, as for DreamerV3.

        An untrained head here read inside the band on 2 of 20 CPU seeds
        and 4 of 20 Metal ones, so withholding the optimiser guarded it
        only on the seeds where the initialisation happened to miss.

        Asserted as "leaves the real band", not "reaches the shifted one":
        this head regresses a scalar at a third of DreamerV3's rate, and in
        sixty steps it covers 0.62-1.10 of the shift (DreamerV3: 0.89-1.11)
        — inside the shifted band on 13 of 20 CPU seeds, but above the
        real band on all 40 seeds of both devices, by 0.30 at the least.
        """
        _, _, replay, out, _ = _train_planet(shift=_REWARD_SHIFT)
        _, high = _reward_range(replay)
        predicted = float(out.reward.mean().item())
        assert predicted > high, (
            f"trained on rewards shifted by {_REWARD_SHIFT}, the head reads "
            f"{predicted:.3f} — still inside the real rewards, which end at {high:.3f}"
        )

    def test_a_frozen_model_does_not_learn_to_reconstruct(self) -> None:
        """Guards the reconstruction half of the test above.

        Without a step the reconstruction only varies with the batch:
        over 20 seeds it ended at 0.96-1.05 of where it began, where a
        trained run ends at 0.27-0.73.
        """
        _, _, _, out, first = _train_planet(train=False)
        assert float(out.recon_loss.item()) >= first * 0.9


class TestTheDynamicsGradientCanLearn:
    """The other way an actor is trained here, which had no test.

    :class:`TestTheActorObjectiveCanLearn` covers the score function —
    the estimator DreamerV2 uses for discrete actions and DreamerV3 uses
    for both. Dreamer v1 does not use it for continuous ones. Its actor
    loss is ``-(weight * returns).mean()`` with nothing detached between
    the return and the policy, so the gradient arrives *through the
    sampled action* and the dynamics that consumed it, not through a
    log-probability.

    The two paths fail differently. A score-function bug leaves the
    log-probability intact and the advantage wrong; a pathwise bug
    severs the action from the objective, and then the actor's gradient
    is not small — it is exactly zero, or it is whatever the entropy
    term alone contributes. Neither is visible in a return curve, and
    neither is covered by the class above.

    So: no RSSM, no critic. A differentiable score that is a known
    function of the sampled action, and the question of whether the
    policy climbs it by backpropagating *into the sample*.
    """

    @staticmethod
    def _actor(action_dim: int = 1) -> Actor:
        lucid.manual_seed(0)
        return Actor(
            latent_size=4, hidden=32, layers=2, action_dim=action_dim, act_fn="silu", min_std=0.1
        )

    @staticmethod
    def _mode(actor: Actor, feature: lucid.Tensor) -> float:
        with lucid.no_grad():
            return float(actor.distribution(feature).mode[..., 0].mean().item())

    def test_a_reparameterised_sample_carries_gradient(self) -> None:
        """The property the whole path rests on.

        ``rsample`` has to be differentiable with respect to the actor's
        parameters. If it is not — if it silently detaches, as ``sample``
        would — every test below still runs and the actor never moves.
        """
        actor = self._actor()
        feature = lucid.ones((8, 1, 4))
        action = actor.distribution(feature).rsample()
        action[..., 0].mean().backward()
        total = sum(
            float((p.grad**2).sum().item()) for p in actor.parameters() if p.grad is not None
        )
        assert total > 0.0, "no gradient reached the actor through rsample()"

    def test_a_continuous_policy_climbs_the_pathwise_score(self) -> None:
        """v1's objective in miniature: maximise a differentiable score.

        The score is the action's own first component, so the optimum is
        +1 and the gradient is available only by differentiating through
        the sample.
        """
        actor = self._actor()
        opt = optim.Adam(actor.parameters(), lr=3e-3)
        feature = lucid.ones((64, 1, 4))
        before = self._mode(actor, feature)
        for _ in range(150):
            action = actor.distribution(feature).rsample()
            loss = -action[..., 0].mean()
            actor.zero_grad()
            loss.backward()
            opt.step()
        after = self._mode(actor, feature)
        assert after > before + 0.5, (
            f"the mode did not climb the pathwise score: " f"{before:+.4f} -> {after:+.4f}"
        )
        assert after > 0.9, f"did not approach the optimum: {after:+.4f}"

    def test_detaching_the_sample_stops_the_climb(self) -> None:
        """Guards the test above by breaking the one thing it tests.

        With the sample detached the loss is a constant as far as the
        actor is concerned — it has no graph at all, so ``backward()``
        refuses it rather than training nothing.  The climb above is
        therefore measuring the pathwise gradient and nothing else.
        """
        actor = self._actor()
        feature = lucid.ones((64, 1, 4))
        action = actor.distribution(feature).rsample().detach()
        loss = -action[..., 0].mean()
        assert not loss.requires_grad
        with pytest.raises(RuntimeError, match="does not require grad"):
            loss.backward()


class TestTheActorObjectiveCanLearn:
    """REINFORCE on its own, with an advantage that is known to be right.

    This exists because its absence cost real time. When the policy did
    not improve end-to-end, there was nothing between "the actor-critic
    is broken" and "the world model is not good enough yet" — the two
    look identical from a return curve, and the first guess was wrong.
    Every part of the objective was already covered (log-probabilities,
    entropy, lambda-returns, the advantage's scale) and none of them
    answers whether the thing *learns*.

    So: no RSSM, no critic, no imagination. A fixed feature, an advantage
    that is a known function of the sampled action, and the question of
    whether the policy moves toward the actions that pay. If this fails,
    a failure upstream is the actor's fault; if it passes, it is not.
    """

    @staticmethod
    def _ascend(discrete: bool, steps: int = 150) -> tuple[float, float]:
        lucid.manual_seed(0)
        action_dim = 3 if discrete else 1
        actor = Actor(
            latent_size=4,
            hidden=32,
            layers=2,
            action_dim=action_dim,
            act_fn="silu",
            min_std=0.1,
            discrete=discrete,
            unimix=0.01 if discrete else 0.0,
        )
        opt = optim.Adam(actor.parameters(), lr=3e-3)
        feature = lucid.ones((64, 1, 4))

        def preference() -> float:
            with lucid.no_grad():
                return float(actor.distribution(feature).mode[..., 0].mean().item())

        before = preference()
        for _ in range(steps):
            action = actor.distribution(feature).rsample().detach()
            # Paid for the first component: push toward +1 when
            # continuous, toward choosing alternative 0 when discrete.
            advantage = action[..., 0].detach()
            loss = -(actor.log_prob(feature, action) * advantage).mean()
            actor.zero_grad()
            loss.backward()
            opt.step()
        return before, preference()

    def test_a_continuous_policy_climbs(self) -> None:
        before, after = self._ascend(discrete=False)
        assert after > before + 0.5, (
            f"the mode did not move toward the paid action: " f"{before:+.4f} -> {after:+.4f}"
        )
        assert after > 0.9, f"did not approach the optimum: {after:+.4f}"

    def test_a_discrete_policy_concentrates(self) -> None:
        """The one-hot's entropy has to fall onto the paid alternative."""
        lucid.manual_seed(0)
        actor = Actor(
            latent_size=4,
            hidden=32,
            layers=2,
            action_dim=3,
            act_fn="silu",
            min_std=0.1,
            discrete=True,
            unimix=0.01,
        )
        opt = optim.Adam(actor.parameters(), lr=3e-3)
        feature = lucid.ones((64, 1, 4))
        start = float(actor.distribution(feature).entropy().mean().item())
        for _ in range(150):
            action = actor.distribution(feature).rsample().detach()
            loss = -(actor.log_prob(feature, action) * action[..., 0].detach()).mean()
            actor.zero_grad()
            loss.backward()
            opt.step()
        end = float(actor.distribution(feature).entropy().mean().item())
        chosen = float(actor.distribution(feature).mode[..., 0].mean().item())
        assert end < start - 0.3, f"policy never concentrated: {start:.3f} -> {end:.3f}"
        assert chosen > 0.9, f"concentrated on the wrong alternative: {chosen:.3f}"

    def test_the_climb_needs_the_advantage(self) -> None:
        """Guards both above.

        A policy left to the entropy term alone drifts, and a drifting
        mode can pass a "it moved" check. Zero the advantage and the
        objective has nothing to say about which action is better, so the
        mode must not arrive at the paid one.
        """
        lucid.manual_seed(0)
        actor = Actor(latent_size=4, hidden=32, layers=2, action_dim=1, act_fn="silu", min_std=0.1)
        opt = optim.Adam(actor.parameters(), lr=3e-3)
        feature = lucid.ones((64, 1, 4))
        for _ in range(150):
            action = actor.distribution(feature).rsample().detach()
            zero = lucid.zeros_like(action[..., 0])
            loss = -(actor.log_prob(feature, action) * zero).mean()
            actor.zero_grad()
            loss.backward()
            opt.step()
        with lucid.no_grad():
            mode = float(actor.distribution(feature).mode[..., 0].mean().item())
        assert mode < 0.9, f"reached the optimum without an advantage: {mode:+.4f}"
