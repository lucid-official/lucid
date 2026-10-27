"""Unit tests for the rollout layer.

Two things here are worth more than the rest.

**Chunk sampling** is the reason the replay buffer exists rather than a
list: a recurrent model handed loose transitions has nothing to be
recurrent about. So the tests check that a drawn window is contiguous and
lies inside one episode, not merely that the shapes come out right.

**Termination versus truncation** is checked separately in both
directions. Collapsing them is the classic world-model bug: a discount
head taught that running out of clock is the same as dying stops looking
past its own horizon.
"""

import pytest

import lucid
from lucid.utils.rollout import (
    Environment,
    Episode,
    LatentPolicy,
    Policy,
    RandomPolicy,
    SequenceReplay,
    StepResult,
    rollout,
    PrioritizedSequenceReplay,
    SumTree,
)


class _Counter:
    """Observations that encode the step index, so order is checkable."""

    def __init__(self, length: int = 12, terminate: bool = False) -> None:
        self.length = length
        self.terminate = terminate

    def reset(self) -> lucid.Tensor:
        self.t = 0
        return lucid.ones((1, 2, 2)) * self.t

    def step(self, action: lucid.Tensor) -> StepResult:
        self.t += 1
        last = self.t >= self.length
        return StepResult(
            lucid.ones((1, 2, 2)) * self.t,
            float(self.t),
            last and self.terminate,
            last and not self.terminate,
        )


class _DeviceCounter:
    """``_Counter``, but rendering wherever it is told to."""

    def __init__(self, device: str, length: int = 6) -> None:
        self.device, self.length = device, length

    def reset(self) -> lucid.Tensor:
        self.t = 0
        return lucid.ones((1, 2, 2), device=self.device)

    def step(self, action: lucid.Tensor) -> StepResult:
        self.t += 1
        last = self.t >= self.length
        return StepResult(lucid.ones((1, 2, 2), device=self.device), float(self.t), False, last)


class TestEpisodesAreDeviceConsistent:
    """An episode's five tensors have to end up in the same place.

    Rewards and discounts arrive from the environment as Python floats,
    so unlike observations and actions they carry no device of their own
    and default to the CPU.  Collect from an accelerator environment and
    the episode silently splits — three tensors there, two here — and
    nothing complains until a world model reads a batch, several layers
    away, against whichever op reached a reward first.

    This is the seam neither side's tests covered: the rollout layer was
    tested without a model and the models were tested on-device without
    the rollout layer.
    """

    @pytest.mark.parametrize("device", ["cpu", "metal"])
    def test_rollout_puts_every_field_on_one_device(self, device: str) -> None:
        episode, _ = rollout(_DeviceCounter(device), RandomPolicy(2))
        places = {
            name: getattr(episode, name).device.type
            for name in ("observations", "actions", "rewards", "discounts")
        }
        assert set(places.values()) == {device}, places

    @pytest.mark.parametrize("device", ["cpu", "metal"])
    def test_a_sampled_batch_keeps_them_together(self, device: str) -> None:
        """The buffer must not undo it either."""
        replay = SequenceReplay()
        for _ in range(2):
            replay.add(rollout(_DeviceCounter(device), RandomPolicy(2))[0])
        batch = replay.sample(2, 4)
        places = {
            getattr(batch, name).device.type
            for name in ("observations", "actions", "rewards", "discounts")
        }
        assert places == {device}, places

    @pytest.mark.parametrize("device", ["cpu", "metal"])
    def test_random_policy_follows_the_observation(self, device: str) -> None:
        """Guards the two above — they would also pass if the environment
        alone decided, and the seeding policy is the other half of it."""
        observation = lucid.zeros((1, 2, 2), device=device)
        assert RandomPolicy(2)(observation).device.type == device
        assert RandomPolicy(3, discrete=True)(observation).device.type == device

    def test_a_deviceless_observation_still_works(self) -> None:
        """``RandomPolicy(None)`` is in its own docstring; keep it valid."""
        assert RandomPolicy(3)(None).shape == (3,)


def _episode(length: int = 10, action_dim: int = 2) -> Episode:
    return Episode(
        observations=lucid.randn((length, 3, 4, 4)),
        actions=lucid.randn((length, action_dim)),
        rewards=lucid.randn((length,)),
        discounts=lucid.ones((length,)),
    )


class TestEnvironmentProtocol:
    def test_a_plain_object_satisfies_it(self) -> None:
        """No base class to inherit — that is the point of a Protocol."""
        assert isinstance(_Counter(), Environment)

    def test_random_policy_satisfies_policy(self) -> None:
        assert isinstance(RandomPolicy(3), Policy)

    def test_missing_step_is_rejected(self) -> None:
        class Broken:
            def reset(self) -> None:
                return None

        assert not isinstance(Broken(), Environment)


class TestRollout:
    def test_shapes_and_alignment(self) -> None:
        episode, total = rollout(_Counter(length=6), RandomPolicy(2))
        assert len(episode) == 6
        assert episode.observations.shape == (6, 1, 2, 2)
        assert episode.actions.shape == (6, 2)
        assert episode.rewards.shape == (6,)
        assert total == pytest.approx(1 + 2 + 3 + 4 + 5 + 6)

    def test_first_action_is_zero(self) -> None:
        """Nothing was taken *into* the first observation."""
        episode, _ = rollout(_Counter(), RandomPolicy(2))
        assert bool((episode.actions[0] == 0).all().item())

    def test_observations_are_in_order(self) -> None:
        episode, _ = rollout(_Counter(length=5), RandomPolicy(2))
        seen = [float(episode.observations[t].mean().item()) for t in range(5)]
        assert seen == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_termination_zeroes_the_last_discount(self) -> None:
        episode, _ = rollout(_Counter(length=4, terminate=True), RandomPolicy(2))
        discounts = [float(x) for x in episode.discounts]
        assert discounts == [1.0, 1.0, 1.0, 0.0]

    def test_truncation_does_not(self) -> None:
        """The episode would have continued; the critic must still bootstrap."""
        episode, _ = rollout(_Counter(length=4, terminate=False), RandomPolicy(2))
        assert all(float(x) == 1.0 for x in episode.discounts)

    def test_max_steps_truncates_rather_than_terminates(self) -> None:
        episode, _ = rollout(_Counter(length=100), RandomPolicy(2), max_steps=5)
        assert len(episode) == 5
        assert all(float(x) == 1.0 for x in episode.discounts)

    @pytest.mark.parametrize("repeat", [1, 2, 3])
    def test_action_repeat_records_at_the_environment_rate(self, repeat: int) -> None:
        """One decision, several steps — but every step is stored."""
        episode, _ = rollout(_Counter(length=6), RandomPolicy(2), action_repeat=repeat)
        assert len(episode) == 6

    def test_action_repeat_holds_the_action(self) -> None:
        episode, _ = rollout(_Counter(length=6), RandomPolicy(2), action_repeat=3)
        # Entries 1 and 2 come from the same decision as each other.
        assert bool((episode.actions[1] == episode.actions[2]).all().item())

    @pytest.mark.parametrize("kwargs", [{"action_repeat": 0}, {"max_steps": 0}])
    def test_rejects_bad_arguments(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValueError):
            rollout(_Counter(), RandomPolicy(2), **kwargs)  # type: ignore[arg-type]


class TestRandomPolicyActionSpaces:
    """Seeding has to match the space the agent will act in.

    Filling a discrete agent's buffer with continuous noise fits its
    dynamics on action vectors it is never given again — a uniform draw
    in ``(-1, 1)`` looks nothing like a one-hot — so the model trains on
    one input distribution and acts under another.
    """

    def test_continuous_is_the_default(self) -> None:
        drawn = RandomPolicy(3)(None)
        assert drawn.shape == (3,)
        assert bool((drawn.abs() <= 1.0).all().item())

    def test_discrete_draws_a_one_hot(self) -> None:
        policy = RandomPolicy(4, discrete=True)
        policy.reset()
        for _ in range(20):
            drawn = policy(None)
            assert drawn.shape == (4,)
            assert abs(float(drawn.sum().item()) - 1.0) < 1e-6

    def test_discrete_covers_every_choice(self) -> None:
        """A sampler stuck on one action would still pass the shape check."""
        policy = RandomPolicy(4, discrete=True)
        policy.reset()
        seen = {int(policy(None).argmax(dim=-1).item()) for _ in range(200)}
        assert seen == {0, 1, 2, 3}

    def test_continuous_is_not_a_one_hot(self) -> None:
        """Guards the pair above by showing the two really differ."""
        drawn = [RandomPolicy(4)(None) for _ in range(20)]
        assert not any(abs(float(a.sum().item()) - 1.0) < 1e-6 for a in drawn)

    def test_rejects_an_empty_action_space(self) -> None:
        with pytest.raises(ValueError):
            RandomPolicy(0)


class _Buttons:
    """Four choices; the observation encodes which was pressed."""

    def __init__(self, length: int = 5) -> None:
        self.length = length

    def reset(self) -> lucid.Tensor:
        self.t = 0
        return lucid.zeros((3, 64, 64))

    def step(self, action: lucid.Tensor) -> StepResult:
        self.t += 1
        return StepResult(lucid.zeros((3, 64, 64)), 1.0, False, self.t >= self.length)


class TestDiscreteAgentRollout:
    def test_a_one_hot_policy_drives_a_rollout(self) -> None:
        from lucid.models import dreamer_v2

        model = dreamer_v2(
            action_dim=4,
            action_space="discrete",
            cnn_depth=2,
            stoch_size=3,
            discrete=4,
            deter_size=8,
            hidden_size=8,
            actor_hidden=8,
            value_hidden=8,
            reward_hidden=8,
        ).eval()
        policy = LatentPolicy(model.encode, model.rssm, lambda s: model.act(s, sample=False), 4)
        episode, _ = rollout(_Buttons(5), policy)
        assert len(episode) == 5
        assert episode.actions.shape == (5, 4)
        # The zero placeholder at index 0 is the convention; everything
        # after it must still be a one-hot when it reaches the buffer.
        later = episode.actions[1:]
        assert float((later.sum(dim=-1) - 1.0).abs().max().item()) < 1e-4

    def test_the_buffer_keeps_them_one_hot(self) -> None:
        replay = SequenceReplay()
        policy = RandomPolicy(4, discrete=True)
        replay.add(rollout(_Buttons(6), policy)[0])
        batch = replay.sample(2, 4)
        assert batch.actions.shape == (2, 4, 4)


class TestSequenceReplay:
    def test_stores_and_counts(self) -> None:
        replay = SequenceReplay()
        replay.add(_episode(10))
        replay.add(_episode(7))
        assert len(replay) == 2 and replay.steps == 17

    def test_capacity_is_in_transitions(self) -> None:
        """Frames dominate the memory, so episodes are the wrong unit."""
        replay = SequenceReplay(capacity=25)
        for _ in range(5):
            replay.add(_episode(10))
        assert replay.steps <= 25
        assert len(replay) < 5

    def test_never_evicts_the_last_episode(self) -> None:
        replay = SequenceReplay(capacity=1)
        replay.add(_episode(10))
        assert len(replay) == 1

    def test_sample_shapes(self) -> None:
        replay = SequenceReplay()
        replay.add(_episode(20))
        batch = replay.sample(4, 6)
        assert batch.observations.shape == (4, 6, 3, 4, 4)
        assert batch.actions.shape == (4, 6, 2)
        assert batch.rewards.shape == (4, 6)
        assert batch.discounts.shape == (4, 6)

    def test_chunks_are_contiguous_and_within_one_episode(self) -> None:
        """The property the whole class exists for."""
        replay = SequenceReplay()
        marks = []
        for e in range(3):
            length = 12
            observations = lucid.stack(
                [lucid.ones((1, 2, 2)) * (e * 100 + t) for t in range(length)], dim=0
            )
            marks.append(observations)
            replay.add(
                Episode(
                    observations,
                    lucid.zeros((length, 2)),
                    lucid.zeros((length,)),
                    lucid.ones((length,)),
                )
            )

        batch = replay.sample(16, 5)
        for chunk in range(16):
            values = [round(float(batch.observations[chunk, t].mean().item())) for t in range(5)]
            assert values == list(range(values[0], values[0] + 5)), "not contiguous"
            assert values[0] // 100 == values[-1] // 100, "spans two episodes"

    def test_episodes_shorter_than_the_chunk_are_skipped(self) -> None:
        """Padding would feed the dynamics frames it never produced."""
        replay = SequenceReplay()
        replay.add(_episode(3))
        replay.add(_episode(20))
        batch = replay.sample(8, 10)
        assert batch.observations.shape[1] == 10

    def test_raises_when_nothing_is_long_enough(self) -> None:
        replay = SequenceReplay()
        replay.add(_episode(4))
        with pytest.raises(ValueError):
            replay.sample(2, 10)

    def test_rejects_ragged_episodes(self) -> None:
        replay = SequenceReplay()
        bad = Episode(
            lucid.randn((10, 3, 4, 4)), lucid.randn((9, 2)), lucid.randn((10,)), lucid.ones((10,))
        )
        with pytest.raises(ValueError):
            replay.add(bad)

    def test_rejects_an_empty_episode(self) -> None:
        replay = SequenceReplay()
        with pytest.raises(ValueError):
            replay.add(
                Episode(
                    lucid.zeros((0, 3, 4, 4)),
                    lucid.zeros((0, 2)),
                    lucid.zeros((0,)),
                    lucid.zeros((0,)),
                )
            )

    @pytest.mark.parametrize("kwargs", [{"capacity": 0}])
    def test_rejects_bad_capacity(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValueError):
            SequenceReplay(**kwargs)

    def test_sampling_follows_the_global_seed(self) -> None:
        """A whole run should be reproducible, buffer draws included."""
        replay = SequenceReplay()
        for _ in range(4):
            replay.add(_episode(12))
        lucid.manual_seed(7)
        first = replay.sample(3, 5).observations
        lucid.manual_seed(7)
        second = replay.sample(3, 5).observations
        assert bool((first == second).all().item())


class TestLatentPolicy:
    def _model(self) -> object:
        from lucid.models import dreamer

        return dreamer(
            action_dim=2,
            cnn_depth=2,
            stoch_size=4,
            deter_size=8,
            hidden_size=8,
            actor_hidden=8,
            value_hidden=8,
            reward_hidden=8,
        ).eval()

    def _policy(self, model: object, noise: float = 0.0) -> LatentPolicy:
        return LatentPolicy(
            model.encode, model.rssm, lambda state: model.act(state, sample=False), 2, noise=noise
        )

    def test_emits_a_bounded_action(self) -> None:
        policy = self._policy(self._model())
        policy.reset()
        action = policy(lucid.zeros((3, 64, 64)))
        assert action.shape == (2,)
        assert bool((action.abs() <= 1.0).all().item())

    def test_belief_changes_with_observations(self) -> None:
        """If it did not, the policy would be acting on the frame alone."""
        model = self._model()
        policy = self._policy(model)
        policy.reset()
        policy(lucid.zeros((3, 64, 64)))
        first = policy._state.deter.clone()
        policy(lucid.ones((3, 64, 64)))
        assert float((first - policy._state.deter).abs().max().item()) > 0

    def test_reset_returns_to_the_prior(self) -> None:
        model = self._model()
        policy = self._policy(model)
        policy.reset()
        first = policy(lucid.ones((3, 64, 64)))
        policy.reset()
        second = policy(lucid.ones((3, 64, 64)))
        assert bool((first == second).all().item())

    def test_noise_widens_the_action_and_stays_bounded(self) -> None:
        model = self._model()
        quiet, loud = self._policy(model), self._policy(model, noise=0.5)
        quiet.reset()
        loud.reset()
        frame = lucid.ones((3, 64, 64))
        assert not bool((quiet(frame) == loud(frame)).all().item())
        loud.reset()
        assert bool((loud(frame).abs() <= 1.0).all().item())

    def test_rejects_negative_noise(self) -> None:
        model = self._model()
        with pytest.raises(ValueError):
            LatentPolicy(model.encode, model.rssm, lambda s: None, 2, noise=-1.0)

    def test_rollout_does_not_leave_the_policy_a_step_ahead(self) -> None:
        """The zero-action probe must not advance the belief."""
        model = self._model()
        policy = self._policy(model)
        episode, _ = rollout(_Counter64(length=4), policy)
        assert len(episode) == 4
        assert bool((episode.actions[0] == 0).all().item())


class TestActionBounds:
    """``Policy`` promises ``(-1, 1)``; both families have to keep it.

    PlaNet does not get there by construction — its planner searches an
    unbounded Gaussian — so the bound is enforced here. Earlier it was
    applied only when exploration noise had been added, which made the
    same policy legal in one call and not the other.
    """

    def _dreamer(self) -> object:
        from lucid.models import dreamer

        return dreamer(
            action_dim=2,
            cnn_depth=2,
            stoch_size=4,
            deter_size=8,
            hidden_size=8,
            actor_hidden=8,
            value_hidden=8,
            reward_hidden=8,
        ).eval()

    @pytest.mark.parametrize("noise", [0.0, 0.5])
    def test_dreamer_actions_are_bounded(self, noise: float) -> None:
        model = self._dreamer()
        policy = LatentPolicy(
            model.encode, model.rssm, lambda state: model.act(state, sample=False), 2, noise=noise
        )
        episode, _ = rollout(_Counter64(length=4), policy)
        assert bool((episode.actions.abs() <= 1.0).all().item())

    @pytest.mark.parametrize("noise", [0.0, 0.5])
    def test_planet_actions_are_bounded(self, noise: float) -> None:
        """The planner is unbounded by design — the policy is what clips."""
        from lucid.models import create_model

        wrapper = create_model(
            "planet_world_model",
            action_dim=2,
            cnn_depth=2,
            stoch_size=4,
            deter_size=8,
            hidden_size=8,
            reward_hidden=8,
        ).eval()
        policy = LatentPolicy(
            wrapper.planet.encode,
            wrapper.planet.rssm,
            lambda state: wrapper.plan(state, horizon=3, iterations=2, candidates=16, elites=4),
            2,
            noise=noise,
        )
        episode, _ = rollout(_Counter64(length=3), policy)
        assert bool((episode.actions.abs() <= 1.0).all().item())

    def test_planner_itself_is_still_unbounded(self) -> None:
        """Guards the test above: the clip is doing real work, not nothing."""
        from lucid.models import create_model

        wrapper = create_model(
            "planet_world_model",
            action_dim=2,
            cnn_depth=2,
            stoch_size=4,
            deter_size=8,
            hidden_size=8,
            reward_hidden=8,
        ).eval()
        state = wrapper.planet.rssm.initial(8)
        planned = wrapper.plan(state, horizon=3, iterations=2, candidates=16, elites=4)
        assert float(planned.abs().max().item()) > 1.0


class _Counter64:
    """Same as :class:`_Counter` but at the resolution the pixel nets need."""

    def __init__(self, length: int = 4) -> None:
        self.length = length

    def reset(self) -> lucid.Tensor:
        self.t = 0
        return lucid.zeros((3, 64, 64))

    def step(self, action: lucid.Tensor) -> StepResult:
        self.t += 1
        return StepResult(lucid.zeros((3, 64, 64)), 1.0, False, self.t >= self.length)


class TestSumTree:
    """The structure that makes prioritized sampling affordable."""

    def test_it_finds_the_leaf_owning_a_prefix(self) -> None:
        tree = SumTree(4)
        tree.set(0, 1.0)
        tree.set(3, 3.0)
        assert tree.total == 4.0
        assert tree.find(0.5) == 0
        assert tree.find(2.0) == 3

    def test_updates_repair_every_sum_above(self) -> None:
        tree = SumTree(8)
        for i in range(8):
            tree.set(i, float(i))
        assert tree.total == pytest.approx(28.0)
        tree.set(3, 100.0)
        assert tree.total == pytest.approx(125.0)
        assert tree.get(3) == 100.0

    def test_the_search_is_proportional(self) -> None:
        """A leaf twice as heavy must own twice the range."""
        tree = SumTree(4)
        tree.set(0, 1.0)
        tree.set(1, 2.0)
        owned = sum(1 for i in range(300) if tree.find(i / 300.0 * 3.0) == 1)
        assert 0.6 < owned / 300 < 0.73

    @pytest.mark.parametrize("bad", [-1, 4])
    def test_it_rejects_an_out_of_range_leaf(self, bad: int) -> None:
        with pytest.raises(ValueError):
            SumTree(4).set(bad, 1.0)

    def test_it_rejects_a_negative_priority(self) -> None:
        with pytest.raises(ValueError):
            SumTree(4).set(0, -1.0)


class TestPrioritizedSequenceReplay:
    """Schaul et al. (2016), and the two details that are easy to omit."""

    @staticmethod
    def _episode(length: int, tag: float) -> Episode:
        return Episode(
            observations=lucid.ones((length, 3, 8, 8)) * tag,
            actions=lucid.zeros((length, 2)),
            rewards=lucid.zeros((length,)),
            discounts=lucid.ones((length,)),
        )

    def _filled(self, count: int = 5, alpha: float = 0.6, beta: float = 0.4):
        buffer = PrioritizedSequenceReplay(capacity=100_000, alpha=alpha, beta=beta)
        for i in range(count):
            buffer.add(self._episode(6, float(i)))
        return buffer

    def test_a_new_episode_enters_at_the_largest_priority(self) -> None:
        """Algorithm 1, line 6 — otherwise it can be buried before its
        first replay, and prioritization would starve fresh experience."""
        buffer = self._filled(4)
        buffer.update_priorities([0, 1, 2, 3], [10.0, 1e-3, 1e-3, 1e-3])
        buffer.add(self._episode(6, 4.0))
        newest = buffer._tree.get(len(buffer) - 1)
        assert newest > buffer._tree.get(1)

    def test_sampling_follows_priority(self) -> None:
        buffer = self._filled(4)
        buffer.update_priorities([0, 1, 2, 3], [10.0, 1e-3, 1e-3, 1e-3])
        counts: dict[int, int] = {}
        for _ in range(200):
            index = buffer.sample_prioritized(1, 5).indices[0]
            counts[index] = counts.get(index, 0) + 1
        assert counts.get(0, 0) > 150

    def test_alpha_zero_is_uniform(self) -> None:
        """Guards the test above: the paper says alpha=0 is the uniform case."""
        buffer = self._filled(4, alpha=0.0)
        buffer.update_priorities([0], [100.0])
        counts: dict[int, int] = {}
        for _ in range(400):
            index = buffer.sample_prioritized(1, 5).indices[0]
            counts[index] = counts.get(index, 0) + 1
        assert max(counts.values()) < 160

    def test_weights_are_normalised_by_their_maximum(self) -> None:
        """ "So that they only scale the update downwards.\" """
        buffer = self._filled(5)
        buffer.update_priorities([0, 1, 2, 3, 4], [50.0, 20.0, 5.0, 1.0, 0.1])
        weights = buffer.sample_prioritized(8, 5).weights.tolist()
        assert max(weights) == pytest.approx(1.0)
        assert all(0.0 < w <= 1.0 + 1e-6 for w in weights)

    def test_a_frequently_drawn_episode_is_down_weighted(self) -> None:
        """The correction has to actually correct something."""
        buffer = self._filled(5, beta=1.0)
        buffer.update_priorities([0, 1, 2, 3, 4], [50.0, 20.0, 5.0, 1.0, 0.1])
        seen: dict[int, list[float]] = {}
        for _ in range(60):
            batch = buffer.sample_prioritized(4, 5)
            for index, weight in zip(batch.indices, batch.weights.tolist()):
                seen.setdefault(index, []).append(weight)
        common = max(seen, key=lambda i: len(seen[i]))
        rare = min(seen, key=lambda i: len(seen[i]))
        mean = lambda i: sum(seen[i]) / len(seen[i])
        assert mean(common) < mean(rare)

    def test_beta_anneals_linearly(self) -> None:
        buffer = self._filled(3, beta=0.4)
        buffer.anneal_beta(0.0)
        assert buffer.beta == pytest.approx(0.4)
        buffer.anneal_beta(0.5)
        assert buffer.beta == pytest.approx(0.7)
        buffer.anneal_beta(1.0)
        assert buffer.beta == pytest.approx(1.0)

    def test_the_batch_matches_the_uniform_one_in_shape(self) -> None:
        buffer = self._filled(4)
        batch = buffer.sample_prioritized(3, 5)
        assert batch.episode.observations.shape == (3, 5, 3, 8, 8)
        assert len(batch.indices) == 3
        assert batch.weights.shape == (3,)

    def test_uniform_sampling_still_works(self) -> None:
        """It is a SequenceReplay; the inherited draw must be untouched."""
        assert self._filled(4).sample(2, 5).observations.shape == (2, 5, 3, 8, 8)

    def test_it_refuses_a_length_nothing_can_serve(self) -> None:
        with pytest.raises(ValueError):
            self._filled(2).sample_prioritized(1, 99)

    def test_mismatched_updates_are_refused(self) -> None:
        with pytest.raises(ValueError):
            self._filled(3).update_priorities([0, 1], [1.0])

    @pytest.mark.parametrize("kwargs", [{"alpha": -0.1}, {"beta": -0.1}, {"epsilon": 0.0}])
    def test_it_rejects_bad_hyperparameters(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            PrioritizedSequenceReplay(capacity=10, **kwargs)
