"""Unit tests for DreamerV3 (Hafner et al., Nature 640, 2025).

This family's changes are all of the kind that a shape test cannot see.
Every one of them produces a finite scalar whether or not it is correct,
so each test below is written against the *claim* rather than the output.

**Free bits** clip each step before averaging.  Clipping the mean instead
gives the same answer on a uniform batch and a different one on a mixed
batch, so the test uses a mixed batch — that is the only place the two
differ.

**Return normalisation** is meant to make the actor's objective
dimensionless.  The test scales returns by a thousand and asserts the
loss is unchanged, which is what "dimensionless" has to mean if it means
anything.

**The slow critic** plays the opposite role to DreamerV2's target critic:
the returns bootstrap from the *live* critic and the slow copy only
regularises it.  Nothing in the loss value reveals which way round it is,
so the test moves the slow copy and asserts the returns do not follow.

**The block recurrence** is what makes the size labels true.  A dense GRU
at the same widths trains perfectly well and is 2.2x the parameters, so
the test measures the count.

The actor's alignment is checked by rebuilding its objective from the
model's own submodules, in the shape that caught a real off-by-one in
DreamerV2: an estimator that credits the wrong step still trains, just
toward something else.
"""

from dataclasses import replace

import pytest

import lucid
import lucid.optim as optim
from lucid.models import (
    DreamerV3Config,
    DreamerV3ForWorldModeling,
    DreamerV3Model,
    create_model,
    dreamer_v3_12m,
    is_model,
    list_models,
)
from lucid.models.generative.dreamer_v3 import DreamerV3BehaviorOutput, DreamerV3Output
from lucid.models._utils._generative import generative_activation
from lucid.models.generative._common._actor import Actor
from lucid.models.generative._common._rssm import RSSM, BlockLinear, RSSMState, categorical_kl
from lucid.models.generative.dreamer_v3._config import DREAMER_V3_SIZES
from lucid.models.generative.dreamer_v3._objectives import (
    ReturnNormaliser,
    free_bits_kl,
    percentile,
)

_FACTORIES = [
    "dreamer_v3_12m",
    "dreamer_v3_25m",
    "dreamer_v3_50m",
    "dreamer_v3_100m",
    "dreamer_v3_200m",
    "dreamer_v3_400m",
]


def _tiny_cfg(**overrides: object) -> DreamerV3Config:
    base: dict[str, object] = {
        "action_dim": 2,
        "cnn_depth": 4,
        "stoch_size": 4,
        "discrete": 5,
        "deter_size": 16,
        "hidden_size": 8,
        "actor_hidden": 16,
        "value_hidden": 16,
        "reward_hidden": 16,
        "num_bins": 7,
        "horizon": 4,
        "blocks": 4,
        "pcont": False,
    }
    base.update(overrides)
    return DreamerV3Config(**base)  # type: ignore[arg-type]


def _batch(b: int = 2, t: int = 4) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    return lucid.randn((b, t, 3, 64, 64)), lucid.randn((b, t, 2)), lucid.randn((b, t))


class TestDreamerV3Config:
    def test_defaults_are_the_paper_values(self) -> None:
        cfg = DreamerV3Config()
        assert (cfg.stoch_size, cfg.discrete) == (32, 32)
        assert (cfg.free_nats, cfg.dyn_scale, cfg.rep_scale) == (1.0, 1.0, 0.1)
        assert (cfg.unimix, cfg.actor_entropy) == (0.01, 3e-4)
        assert (cfg.discount, cfg.lambda_, cfg.horizon) == (0.997, 0.95, 15)
        assert (cfg.num_bins, cfg.bin_range) == (255, 20.0)
        assert (cfg.critic_ema, cfg.replay_value_scale) == (0.02, 0.3)
        assert cfg.blocks == 8
        assert (cfg.reward_layers, cfg.pcont_layers) == (1, 1)
        assert (cfg.actor_layers, cfg.value_layers) == (3, 3)
        assert (cfg.critic_slowreg, cfg.pred_scale) == (1.0, 1.0)

    def test_the_ladder_follows_its_own_rule(self) -> None:
        """Table 3 derives every width from the model dimension."""
        for deter, hidden, classes, depth, units in DREAMER_V3_SIZES.values():
            assert deter == 8 * hidden, "recurrent units are 8d"
            assert classes == hidden // 16, "codes per latent are d/16"
            assert depth == hidden // 16, "base CNN channels are d/16"
            assert units == hidden, "the MLPs are d wide"

    def test_latent_is_the_flattened_grid(self) -> None:
        cfg = _tiny_cfg()
        assert cfg.stoch_width == 4 * 5
        assert cfg.latent_size == 16 + 20

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"blocks": 0},
            {"blocks": 5},  # does not divide deter_size
            {"num_bins": 1},
            {"discrete": 1},
            {"unimix": 1.0},
            {"discount": 0.0},
            {"lambda_": 1.5},
            {"bin_range": 0.0},
            {"critic_ema": 0.0},
            {"return_ema_decay": 1.0},
            {"return_low": 96.0},
            {"critic_slowreg": -1.0},
            {"action_space": "ternary"},
        ],
    )
    def test_rejects_bad_config(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            _tiny_cfg(**kwargs)


class TestBlockLinear:
    def test_a_block_reads_only_itself(self) -> None:
        lucid.manual_seed(0)
        layer = BlockLinear(12, 24, blocks=4)
        x = lucid.randn((2, 12))
        before = layer(x)
        moved = lucid.cat([x[:, :3] + 5.0, x[:, 3:]], dim=-1)
        delta = (layer(moved) - before).abs().reshape(2, 4, 6).sum(dim=(0, 2))
        assert float(delta[0].item()) > 0.0
        assert [float(v) for v in delta.tolist()[1:]] == [0.0, 0.0, 0.0]

    def test_it_costs_a_factor_of_blocks_less(self) -> None:
        layer = BlockLinear(64, 128, blocks=8)
        assert layer.weight.shape == (8, 8, 16)
        block_weights = 8 * 8 * 16
        assert block_weights * 8 == 64 * 128, "exactly 1/blocks of a dense map"

    @pytest.mark.parametrize("kwargs", [{"blocks": 0}, {"blocks": 5}])
    def test_rejects_a_grouping_that_does_not_divide(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValueError):
            BlockLinear(12, 24, **kwargs)


class TestBlockRecurrence:
    @staticmethod
    def _rssm(act_fn: str) -> RSSM:
        lucid.manual_seed(0)
        return RSSM(
            stoch_size=3,
            deter_size=16,
            hidden_size=8,
            action_dim=2,
            embed_size=6,
            act_fn=act_fn,
            discrete=4,
            unimix=0.01,
            blocks=4,
        ).eval()

    @pytest.mark.parametrize("act_fn", ["silu", "relu", "gelu"])
    def test_matches_a_composition_of_its_own_parts(self, act_fn: str) -> None:
        """Rebuilt from the module's submodules, so a duplicated or missing
        activation cannot hide behind an idempotent one."""
        model = self._rssm(act_fn)
        lucid.manual_seed(1)
        deter = lucid.randn((2, 16))
        stoch = lucid.randn((2, 12))
        action = lucid.randn((2, 2)) * 3.0

        def act(x: lucid.Tensor) -> lucid.Tensor:
            return generative_activation(act_fn, x)

        scaled = action / action.abs().clip(1.0, None).detach()
        x0 = act(model.deter_in_norm(model.deter_in(deter)))
        x1 = act(model.stoch_in_norm(model.stoch_in(stoch)))
        x2 = act(model.action_in_norm(model.action_in(scaled)))
        shared = lucid.cat([x0, x1, x2], dim=-1).reshape(2, 1, -1).repeat(1, 4, 1)
        grouped = lucid.cat([deter.reshape(2, 4, 4), shared], dim=-1)
        hidden = act(model.block_hidden_norm(model.block_hidden(grouped.reshape(2, -1))))
        gates = model.block_gate(hidden).reshape(2, 4, 12)
        reset = lucid.nn.functional.sigmoid(gates[..., :4].reshape(2, -1))
        candidate = lucid.tanh(reset * gates[..., 4:8].reshape(2, -1))
        update = lucid.nn.functional.sigmoid(gates[..., 8:].reshape(2, -1) - 1.0)
        expected = update * candidate + (1.0 - update) * deter

        got = model._recurrent(stoch, action, deter)
        assert float((got - expected).abs().max().item()) < 1e-5

    def test_the_update_gate_starts_biased_toward_keeping_state(self) -> None:
        """`sigmoid(u - 1)` is what makes a very wide recurrence trainable."""
        model = self._rssm("silu")
        lucid.manual_seed(2)
        deter = lucid.randn((64, 16)) * 10.0
        moved = model._recurrent(lucid.randn((64, 12)), lucid.randn((64, 2)), deter)
        kept = float(((moved - deter).abs() < deter.abs()).to(lucid.float32).mean().item())
        assert kept > 0.9, "a fresh model should mostly carry its state forward"

    def test_blocks_zero_is_the_dense_gru(self) -> None:
        dense = RSSM(stoch_size=3, deter_size=16, hidden_size=8, action_dim=2, embed_size=6)
        assert hasattr(dense, "cell") and not hasattr(dense, "block_gate")
        blocked = self._rssm("silu")
        assert hasattr(blocked, "block_gate") and not hasattr(blocked, "cell")


class TestSizeLabels:
    """The factory names are parameter counts; this is what makes them true."""

    @staticmethod
    def _count(model: lucid.nn.Module) -> int:
        total = 0
        for parameter in model.parameters():
            size = 1
            for dim in parameter.shape:
                size *= int(dim)
            total += size
        return total

    def test_the_smallest_rung_really_is_twelve_million(self) -> None:
        model = dreamer_v3_12m(action_dim=6)
        measured = self._count(model)
        assert 10e6 < measured < 14e6, f"labelled 12M, measured {measured / 1e6:.1f}M"

    def test_a_dense_recurrence_would_miss_by_more_than_double(self) -> None:
        """Guards the test above — otherwise it would pass on any recurrence."""
        cfg = replace(dreamer_v3_12m(action_dim=6).config, blocks=1)
        assert self._count(DreamerV3Model(cfg)) > 20e6


class TestFreeBits:
    @staticmethod
    def _states(spread: float) -> tuple[object, object]:
        lucid.manual_seed(0)
        model = DreamerV3Model(_tiny_cfg()).eval()
        observations, actions, _ = _batch(t=3)
        priors, posteriors = model.observe(observations * spread, actions)
        return posteriors, priors

    @staticmethod
    def _mixed_pair() -> tuple[RSSMState, RSSMState]:
        """Seven steps that agree, one that badly does not."""
        prior = lucid.zeros((1, 8, 2, 8), requires_grad=True)
        posterior = lucid.zeros((1, 8, 2, 8))
        posterior[0, 0, :, 0] = 20.0
        blank = lucid.zeros((1, 8, 16))
        return (
            RSSMState(deter=blank, stoch=blank, logits=posterior),
            RSSMState(deter=blank, stoch=blank, logits=prior),
        )

    def test_the_floor_applies_per_step_not_to_the_mean(self) -> None:
        """The whole point, and the easy thing to write backwards.

        On a batch whose *average* divergence is below the floor but which
        holds one step far above it, clipping the mean returns a flat
        constant — the diverging step stops contributing and stops
        receiving gradient.  Clipping each step first leaves it in.
        """
        posterior, prior = self._mixed_pair()
        assert posterior.logits is not None and prior.logits is not None
        raw = categorical_kl(posterior.logits.detach(), prior.logits)
        per_step = float(raw.clip(1.0, None).mean().item())
        mean_first = float(raw.mean().clip(1.0, None).item())
        assert mean_first < per_step, "the batch must be one where they disagree"

        dynamics, _ = free_bits_kl(posterior, prior, free_nats=1.0)
        assert abs(float(dynamics.item()) - per_step) < 1e-5
        assert abs(float(dynamics.item()) - mean_first) > 1e-3

    def test_the_diverging_step_keeps_its_gradient(self) -> None:
        """Restated as the thing that actually matters."""
        posterior, prior = self._mixed_pair()
        assert prior.logits is not None
        dynamics, _ = free_bits_kl(posterior, prior, free_nats=1.0)
        dynamics.backward()
        assert prior.logits.grad is not None
        assert float(prior.logits.grad[0, 0].abs().sum().item()) > 0.0

    def test_the_two_halves_carry_the_paper_ratio(self) -> None:
        posteriors, priors = self._states(1.0)
        dynamics, representation = free_bits_kl(posteriors, priors)  # type: ignore[arg-type]
        ratio = float(representation.item()) / float(dynamics.item())
        assert abs(ratio - 0.1) < 1e-4

    def test_free_nats_floors_both(self) -> None:
        posteriors, priors = self._states(1.0)
        dynamics, representation = free_bits_kl(posteriors, priors, free_nats=50.0)  # type: ignore[arg-type]
        assert abs(float(dynamics.item()) - 50.0) < 1e-3
        assert abs(float(representation.item()) - 5.0) < 1e-3

    def test_refuses_a_gaussian_state(self) -> None:
        lucid.manual_seed(0)
        rssm = RSSM(stoch_size=3, deter_size=8, hidden_size=8, action_dim=2, embed_size=6).eval()
        priors, posteriors = rssm.observe(lucid.randn((2, 3, 6)), lucid.randn((2, 3, 2)))
        with pytest.raises(ValueError):
            free_bits_kl(posteriors, priors)


class TestReturnNormalisation:
    def test_percentiles_are_exact_on_a_known_sample(self) -> None:
        values = lucid.tensor([float(i) for i in range(101)])
        for fraction in (0.0, 5.0, 50.0, 95.0, 100.0):
            assert abs(float(percentile(values, fraction).item()) - fraction) < 1e-5

    def test_it_converges_to_the_spread(self) -> None:
        lucid.manual_seed(0)
        returns = lucid.randn((256,)) * 1000.0
        normaliser = ReturnNormaliser()
        for _ in range(400):
            divisor = float(normaliser.update(returns))
        spread = float((percentile(returns, 95.0) - percentile(returns, 5.0)).item())
        assert abs(divisor - spread) / spread < 0.05

    def test_small_returns_are_left_alone(self) -> None:
        """The floor at one — dividing by a narrow spread amplifies noise."""
        lucid.manual_seed(0)
        normaliser = ReturnNormaliser()
        for _ in range(400):
            divisor = float(normaliser.update(lucid.randn((256,)) * 0.001))
        assert divisor == 1.0

    def test_evaluating_a_model_does_not_move_the_estimate(self) -> None:
        """Otherwise a run's numbers depend on how often it was measured."""
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg())
        observations, actions, rewards = _batch(t=3)
        model.train()
        model(observations, actions, rewards)
        trained = float(model.returns.spread)
        assert trained > 0.0

        model.eval()
        behavior = model(observations, actions, rewards * 500.0).behavior
        assert behavior is not None
        assert float(model.returns.spread) == trained
        assert float(behavior.return_scale) == max(1.0, trained)

        model.train()
        model(observations, actions, rewards * 500.0)
        assert float(model.returns.spread) != trained, "training must still update it"

    def test_the_actor_objective_is_scale_free(self) -> None:
        """The claim the fixed entropy coefficient rests on."""
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(actor_entropy=0.0))
        observations, actions, _ = _batch(t=3)
        _, posteriors = model.dreamer_v3.observe(observations, actions)
        flat = posteriors.map(lambda x: x.reshape(6, *(int(v) for v in x.shape[2:])))
        states, proposed = model.dreamer_v3.imagine(flat, 4)
        target = lucid.randn((6, 4))
        weight = lucid.ones((6, 4))

        small, _ = model._actor_objective(states, proposed, target, weight, 1.0)
        large, _ = model._actor_objective(states, proposed * 1.0, target * 1000.0, weight, 1000.0)
        assert abs(float(small.item()) - float(large.item())) < 1e-3

    def test_without_the_divisor_it_would_not_be(self) -> None:
        """Guards the test above."""
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(actor_entropy=0.0))
        observations, actions, _ = _batch(t=3)
        _, posteriors = model.dreamer_v3.observe(observations, actions)
        flat = posteriors.map(lambda x: x.reshape(6, *(int(v) for v in x.shape[2:])))
        states, proposed = model.dreamer_v3.imagine(flat, 4)
        target, weight = lucid.randn((6, 4)), lucid.ones((6, 4))
        small, _ = model._actor_objective(states, proposed, target, weight, 1.0)
        unscaled, _ = model._actor_objective(states, proposed, target * 1000.0, weight, 1.0)
        assert abs(float(unscaled.item()) - float(small.item())) > 1.0


class TestSlowCritic:
    def test_the_returns_come_from_the_live_critic(self) -> None:
        """DreamerV2's arrangement inverted, and invisible in the loss."""
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(mean_only=True))
        observations, actions, rewards = _batch(t=3)
        before = model(observations, actions, rewards).behavior
        assert isinstance(before, DreamerV3BehaviorOutput)

        with lucid.no_grad():
            for parameter in model.dreamer_v3.slow_value_head.parameters():
                parameter[:] = parameter + 3.0
        after = model(observations, actions, rewards).behavior
        assert after is not None
        moved = float((after.lambda_return - before.lambda_return).abs().max().item())
        assert moved < 1e-5, "returns must not follow the slow copy"

    def test_it_would_have_shown_if_they_did(self) -> None:
        """Guards the test above by moving the live critic instead."""
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(mean_only=True))
        observations, actions, rewards = _batch(t=3)
        before = model(observations, actions, rewards).behavior
        assert before is not None
        with lucid.no_grad():
            for parameter in model.dreamer_v3.value_head.parameters():
                parameter[:] = parameter + 3.0
        after = model(observations, actions, rewards).behavior
        assert after is not None
        assert float((after.lambda_return - before.lambda_return).abs().max().item()) > 0.1

    def test_the_first_update_copies(self) -> None:
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg())
        with lucid.no_grad():
            for parameter in model.dreamer_v3.value_head.parameters():
                parameter[:] = parameter + 1.0
        model.update_slow_critic()
        gap = sum(
            float((live - slow).abs().sum().item())
            for live, slow in zip(
                model.dreamer_v3.value_head.parameters(),
                model.dreamer_v3.slow_value_head.parameters(),
            )
        )
        assert gap < 1e-6

    def test_later_updates_move_by_the_configured_fraction(self) -> None:
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(critic_ema=0.02))
        model.update_slow_critic()
        anchor = [p.detach().clone() for p in model.dreamer_v3.slow_value_head.parameters()]
        with lucid.no_grad():
            for parameter in model.dreamer_v3.value_head.parameters():
                parameter[:] = parameter + 1.0
        model.update_slow_critic()
        moved = sum(
            float((now - was).abs().sum().item())
            for now, was in zip(model.dreamer_v3.slow_value_head.parameters(), anchor)
        )
        gap = sum(
            float((live - was).abs().sum().item())
            for live, was in zip(model.dreamer_v3.value_head.parameters(), anchor)
        )
        assert abs(moved / gap - 0.02) < 1e-3

    def test_the_slow_copy_is_in_no_optimiser_group(self) -> None:
        model = DreamerV3ForWorldModeling(_tiny_cfg())
        trained = {
            id(p)
            for group in (
                model.world_parameters(),
                model.actor_parameters(),
                model.value_parameters(),
            )
            for p in group
        }
        slow = [id(p) for p in model.dreamer_v3.slow_value_head.parameters()]
        assert slow and not trained.intersection(slow)


class TestActorObjective:
    def test_it_is_the_score_function_at_the_matching_step(self) -> None:
        """Rebuilt by hand, because a mis-aligned estimator still trains."""
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(actor_entropy=0.0))
        observations, actions, _ = _batch(t=3)
        _, posteriors = model.dreamer_v3.observe(observations, actions)
        flat = posteriors.map(lambda x: x.reshape(6, *(int(v) for v in x.shape[2:])))
        states, proposed = model.dreamer_v3.imagine(flat, 4)
        target, weight = lucid.randn((6, 4)), lucid.ones((6, 4))

        got, _ = model._actor_objective(states, proposed, target, weight, 1.0)

        scored = states.map(lambda x: x[:, :4])
        feature = scored.feature.detach()
        advantage = (target - model.dreamer_v3.predict_value(scored)).detach()
        log_prob = model.dreamer_v3.actor.log_prob(feature, proposed.detach())
        expected = -(log_prob * advantage).mean()
        assert abs(float(got.item()) - float(expected.item())) < 1e-5

    def test_a_shifted_pairing_would_read_differently(self) -> None:
        """Guards the alignment above."""
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(actor_entropy=0.0))
        observations, actions, _ = _batch(t=3)
        _, posteriors = model.dreamer_v3.observe(observations, actions)
        flat = posteriors.map(lambda x: x.reshape(6, *(int(v) for v in x.shape[2:])))
        states, proposed = model.dreamer_v3.imagine(flat, 4)
        target, weight = lucid.randn((6, 4)), lucid.ones((6, 4))
        aligned, _ = model._actor_objective(states, proposed, target, weight, 1.0)
        shifted, _ = model._actor_objective(
            states, lucid.cat([proposed[:, 1:], proposed[:, :1]], dim=1), target, weight, 1.0
        )
        assert abs(float(aligned.item()) - float(shifted.item())) > 1e-4

    def test_the_entropy_bonus_moves_the_loss(self) -> None:
        lucid.manual_seed(0)
        observations, actions, rewards = _batch(t=3)
        losses = []
        for bonus in (0.0, 1.0):
            lucid.manual_seed(1)
            model = DreamerV3ForWorldModeling(_tiny_cfg(actor_entropy=bonus, mean_only=True))
            behavior = model(observations, actions, rewards).behavior
            assert behavior is not None
            losses.append(float(behavior.actor_loss.item()))
        assert abs(losses[0] - losses[1]) > 1e-4

    @pytest.mark.parametrize("space", ["continuous", "discrete"])
    def test_both_action_spaces_train_the_actor(self, space: str) -> None:
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(action_space=space))
        observations, actions, rewards = _batch(t=3)
        if space == "discrete":
            actions = lucid.nn.functional.one_hot(
                lucid.zeros((2, 3)).to(lucid.int32), num_classes=2
            ).to(lucid.float32)
        output = model(observations, actions, rewards)
        model.backward(output)
        touched = sum(
            1
            for p in model.actor_parameters()
            if p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
        )
        assert touched == len(model.actor_parameters())

    def test_a_discrete_policy_keeps_every_action_reachable(self) -> None:
        """1% unimix — a probability at zero is a gradient that never returns."""
        lucid.manual_seed(0)
        actor = Actor(8, 16, 2, 4, "silu", 0.1, discrete=True, unimix=0.01)
        with lucid.no_grad():
            actor.head.out.weight[:] = actor.head.out.weight * 200.0
        probabilities = actor.distribution(lucid.randn((32, 1, 8))).probs
        assert float(probabilities.min().item()) >= 0.01 / 4 - 1e-6

    def test_without_unimix_it_does_not(self) -> None:
        lucid.manual_seed(0)
        actor = Actor(8, 16, 2, 4, "silu", 0.1, discrete=True, unimix=0.0)
        with lucid.no_grad():
            actor.head.out.weight[:] = actor.head.out.weight * 200.0
        probabilities = actor.distribution(lucid.randn((32, 1, 8))).probs
        assert float(probabilities.min().item()) < 1e-4


class TestForwardAndShapes:
    def test_direct_model_output(self) -> None:
        lucid.manual_seed(0)
        model = DreamerV3Model(_tiny_cfg()).eval()
        observations, actions, _ = _batch(t=3)
        output = model(observations, actions)
        assert isinstance(output, DreamerV3Output)
        assert output.observation.shape == (2, 3, 3, 64, 64)
        assert output.reward.shape == (2, 3) and output.value.shape == (2, 3)
        assert output.posterior_logits.shape == (2, 3, 4, 5)
        assert output.loss is None and output.behavior is None

    def test_wrapper_populates_every_loss(self) -> None:
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg())
        output = model(*_batch(t=3))
        for name in (
            "loss",
            "recon_loss",
            "reward_loss",
            "dynamics_loss",
            "representation_loss",
            "kl_loss",
        ):
            value = getattr(output, name)
            assert value is not None and value.ndim == 0
        assert output.behavior is not None
        assert output.behavior.value_loss.ndim == 0

    def test_the_reward_head_is_a_cross_entropy(self) -> None:
        """A reward of ten thousand must not produce a loss of a hundred million."""
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg())
        observations, actions, _ = _batch(t=3)
        huge = model(observations, actions, lucid.ones((2, 3)) * 10000.0)
        assert huge.reward_loss is not None
        assert 0.0 <= float(huge.reward_loss.item()) < 100.0

    def test_pcont_path(self) -> None:
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(pcont=True))
        observations, actions, rewards = _batch(t=4)
        with pytest.raises(ValueError):
            model(observations, actions, rewards)
        output = model(observations, actions, rewards, lucid.ones((2, 4)))
        assert output.pcont_loss is not None
        assert output.behavior is not None
        assert output.behavior.imagined_discount is not None

    def test_actions_are_bounded(self) -> None:
        lucid.manual_seed(0)
        model = DreamerV3Model(_tiny_cfg()).eval()
        _, posteriors = model.observe(*_batch(t=3)[:2])
        assert float(model.act(posteriors).abs().max().item()) <= 1.0


class TestTheImaginationWeight:
    """Every imagined step is worth the chance the episode reached it.

    That product runs over the steps *before* the one being weighted, so
    the first imagined step is worth exactly 1 — the trajectory has not
    had a chance to end yet. ``_behavior`` says its two branches "mean
    the same thing"; the constant-discount branch opens at
    ``gamma ** 0``, so the learned-discount branch has to open at 1 too.

    Folding ``pcont[0]`` in instead multiplies the actor's and the
    critic's whole objective by the head's opinion of the state the
    trajectory starts in — about 0.52 at initialisation, which is a
    halved learning rate that no loss curve identifies as one. It costs
    nothing in shape, in count or in gradient flow, because the weight is
    detached before it is used.
    """

    @staticmethod
    def _run(trainer: object) -> tuple[lucid.Tensor, lucid.Tensor]:
        """The weight and the ``pcont`` it came from, on one call.

        Both from the *same* forward: imagination samples, so a second
        call produces a different continuation prediction and the two
        would not be comparable.
        """
        seen: dict[str, lucid.Tensor] = {}
        original = trainer._actor_objective  # type: ignore[attr-defined]

        def spy(*args: object, **kwargs: object) -> object:
            seen["weight"] = args[3]  # type: ignore[assignment]
            return original(*args, **kwargs)

        trainer._actor_objective = spy  # type: ignore[attr-defined]
        try:
            observations, actions, rewards = _batch(t=4)
            output = trainer(  # type: ignore[operator]
                observations, actions, rewards, lucid.ones((2, 4))
            )
        finally:
            trainer._actor_objective = original  # type: ignore[attr-defined]

        assert output.behavior is not None
        pcont = output.behavior.imagined_discount
        assert pcont is not None
        return seen["weight"], pcont

    def test_the_first_imagined_step_is_worth_one(self) -> None:
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(pcont=True))
        weight, _ = self._run(model)
        assert float(weight[0][0].item()) == pytest.approx(1.0)

    def test_it_is_the_product_over_the_earlier_steps(self) -> None:
        """Not over the step itself — that is the off-by-one this pins."""
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(pcont=True))
        weight, pcont = self._run(model)

        running = 1.0
        for step in range(int(weight.shape[1])):
            assert float(weight[0][step].item()) == pytest.approx(running, rel=1e-5)
            running *= float(pcont[0][step].item())

    def test_it_decreases(self) -> None:
        """The guard: a weight of all ones would pass the first test."""
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(pcont=True))
        weight, _ = self._run(model)
        row = [float(v) for v in weight[0].numpy()]
        assert all(later < earlier for earlier, later in zip(row, row[1:]))


class TestReplayCritic:
    def test_it_is_reported_and_scaled(self) -> None:
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg())
        behavior = model(*_batch(t=3)).behavior
        assert behavior is not None and behavior.replay_value_loss is not None
        assert float(behavior.replay_value_loss.item()) > 0.0

    def test_switching_it_off_removes_it(self) -> None:
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(replay_value_scale=0.0))
        behavior = model(*_batch(t=3)).behavior
        assert behavior is not None and behavior.replay_value_loss is None

    def test_it_reaches_only_the_critic(self) -> None:
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(replay_value_scale=1.0))
        output = model(*_batch(t=3))
        assert output.behavior is not None
        model.zero_grad()
        output.behavior.value_loss.backward()
        leaked = [
            p
            for p in model.world_parameters()
            if p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
        ]
        assert not leaked


class TestGradientRouting:
    @staticmethod
    def _touched(parameters: list[lucid.nn.Parameter]) -> int:
        return sum(
            1 for p in parameters if p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
        )

    def test_backward_gives_each_group_its_own_loss(self) -> None:
        # Free bits off: at initialisation every KL term sits under the
        # 1-nat floor, so the prior head's only gradient is exactly zero.
        # This test passed with the floor on only while the CPU ``maximum``
        # backward read its broadcast operand past the buffer (3.15.0).
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg(pcont=True, free_nats=0.0))
        observations, actions, rewards = _batch(t=4)
        output = model(observations, actions, rewards, lucid.ones((2, 4)))
        model.backward(output)
        assert self._touched(model.world_parameters()) == len(model.world_parameters())
        assert self._touched(model.actor_parameters()) == len(model.actor_parameters())
        assert self._touched(model.value_parameters()) > 0

    def test_the_groups_partition_the_learnable_model(self) -> None:
        model = DreamerV3ForWorldModeling(_tiny_cfg())
        groups = [model.world_parameters(), model.actor_parameters(), model.value_parameters()]
        ids = [id(p) for group in groups for p in group]
        assert len(ids) == len(set(ids)), "no parameter in two optimisers"
        slow = {id(p) for p in model.dreamer_v3.slow_value_head.parameters()}
        assert set(ids) | slow == {id(p) for p in model.parameters()}

    def test_backward_rejects_an_output_without_losses(self) -> None:
        lucid.manual_seed(0)
        wrapper = DreamerV3ForWorldModeling(_tiny_cfg())
        bare = DreamerV3Model(_tiny_cfg()).eval()
        with pytest.raises(ValueError):
            wrapper.backward(bare(*_batch(t=3)[:2]))


class TestTrainingStep:
    def test_losses_fall_over_a_short_run(self) -> None:
        lucid.manual_seed(0)
        model = DreamerV3ForWorldModeling(_tiny_cfg())
        world = optim.Adam(model.world_parameters(), lr=3e-4)
        actor = optim.Adam(model.actor_parameters(), lr=3e-4)
        critic = optim.Adam(model.value_parameters(), lr=3e-4)
        observations, actions, rewards = _batch(t=3)

        first = last = 0.0
        for step in range(12):
            output = model(observations, actions, rewards)
            assert output.loss is not None
            model.backward(output)
            world.step()
            actor.step()
            critic.step()
            model.update_slow_critic()
            world.zero_grad()
            actor.zero_grad()
            critic.zero_grad()
            if step == 0:
                first = float(output.loss.item())
            last = float(output.loss.item())
        assert last < first


class TestRegistry:
    @pytest.mark.parametrize("name", _FACTORIES)
    def test_both_tasks_are_registered(self, name: str) -> None:
        registered = list_models()
        assert name in registered
        assert f"{name}_world_model" in registered
        assert is_model(name)

    def test_create_model_accepts_overrides(self) -> None:
        model = create_model("dreamer_v3_12m", **_tiny_cfg().__dict__)
        assert isinstance(model, DreamerV3Model)
        assert model.config.deter_size == 16

    def test_pretrained_weights_are_refused(self) -> None:
        with pytest.raises(NotImplementedError):
            dreamer_v3_12m(pretrained=True)
