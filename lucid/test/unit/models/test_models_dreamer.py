"""Unit tests for Dreamer (Hafner et al., 2020).

Two families of test carry the weight here.

The **lambda-return** tests check the recursion against the paper's closed
form — an exponentially-weighted sum over n-step returns — rather than
against a shape or a finite value. A wrong discount power or a
one-off index still produces a plausible scalar and a model that trains to
nothing, so the only useful check is the arithmetic itself.

The **gradient-routing** tests pin which modules each of the three losses
may reach. Dreamer's whole contribution is that the actor's gradient
travels *through* the learned dynamics, so "does it reach the RSSM" is a
correctness question with a specific answer per loss, not a matter of
taste: the world-model loss must not reach the actor or critic; the actor
loss must reach the dynamics; the critic loss must reach nothing but
itself.
"""

import pytest

import lucid
from lucid.models import (
    dreamer_discrete,
    DreamerConfig,
    DreamerForWorldModeling,
    DreamerModel,
    create_model,
    is_model,
    list_models,
)
from lucid.models.generative.dreamer import DreamerBehaviorOutput, DreamerOutput
from lucid.models.generative._common._rssm import RSSMState
from lucid.models.generative._common._returns import lambda_return


def _tiny_cfg(**overrides: object) -> DreamerConfig:
    base: dict[str, object] = {
        "action_dim": 2,
        "stoch_size": 4,
        "deter_size": 8,
        "hidden_size": 8,
        "cnn_depth": 4,
        "reward_hidden": 8,
        "actor_hidden": 8,
        "value_hidden": 8,
        "horizon": 4,
    }
    base.update(overrides)
    return DreamerConfig(**base)  # type: ignore[arg-type]


def _batch(
    b: int = 2, t: int = 3, action_dim: int = 2
) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    return (lucid.randn((b, t, 3, 64, 64)), lucid.randn((b, t, action_dim)), lucid.randn((b, t)))


class TestDreamerConfig:
    def test_defaults_match_paper(self) -> None:
        cfg = DreamerConfig(action_dim=6)
        assert (cfg.horizon, cfg.lambda_, cfg.discount) == (15, 0.95, 0.99)
        # Appendix A, verbatim: "We use the convolutional encoder and
        # decoder networks from Ha and Schmidhuber (2018), the RSSM of
        # Hafner et al. (2018), and implement *all other functions* as
        # three dense layers of size 300 with ELU activations."  All
        # other functions is reward, action and value alike — an earlier
        # version of this test paraphrased it as "the action and value
        # models" and used that to justify a two-layer reward head.
        # The released code disagrees with the paper on every one of
        # these (400 units; reward 2, value 3, actor 4).
        assert (cfg.actor_hidden, cfg.actor_layers) == (300, 3)
        assert (cfg.value_hidden, cfg.value_layers) == (300, 3)
        assert (cfg.reward_hidden, cfg.reward_layers) == (300, 3)
        assert cfg.act_fn == "elu"

    def test_head_depths_reach_the_modules(self) -> None:
        """A cited depth is worth nothing if the head is built some other way."""
        model = DreamerModel(DreamerConfig(action_dim=2))
        assert len(model.reward_head.layers) == 3
        assert len(model.value_head.layers) == 3
        assert len(model.actor.head.layers) == 3

    def test_pcont_defaults_off(self) -> None:
        """The paper introduces it for early termination; DMC has none."""
        cfg = DreamerConfig(action_dim=2)
        assert cfg.pcont is False
        assert cfg.pcont_scale == 10.0 and cfg.pcont_layers == 3
        assert cfg.detach_actor_input is True

    def test_world_model_fields_inherited(self) -> None:
        """The state geometry comes from the shared base, not from Dreamer."""
        cfg = DreamerConfig(action_dim=6)
        assert (cfg.stoch_size, cfg.deter_size, cfg.hidden_size) == (30, 200, 200)
        assert (cfg.cnn_depth, cfg.min_std, cfg.free_nats) == (32, 0.1, 3.0)
        assert (cfg.embed_size, cfg.latent_size) == (1024, 230)

    def test_matches_planet_where_the_papers_agree(self) -> None:
        """Every field promoted to the shared base holds one value in both.

        This is the promotion rule, enforced. A field only belongs on
        ``WorldModelConfig`` if PlaNet and Dreamer were *observed* to state
        the same value — hoisting one they disagree on would force the base
        to invent a number neither paper backs. ``act_fn`` is the standing
        example of a field kept out for exactly that reason: ReLU in one
        paper, ELU in the other.
        """
        from lucid.models import PlaNetConfig
        from lucid.models.generative._common._config import WorldModelConfig

        promoted = set(WorldModelConfig.__annotations__) - {"model_type"}
        assert "act_fn" not in promoted
        planet, dreamer = PlaNetConfig(), DreamerConfig()
        differing = {name for name in promoted if getattr(planet, name) != getattr(dreamer, name)}
        assert not differing, f"shared base disagrees with a family on {differing}"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sample_size": 32},
            {"horizon": 0},
            {"discount": 0.0},
            {"discount": 1.5},
            {"lambda_": -0.1},
            {"lambda_": 1.1},
            {"actor_min_std": 0.0},
            {"actor_mean_scale": -1.0},
            {"actor_layers": 0},
            {"value_hidden": 0},
        ],
    )
    def test_rejects_bad_config(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            DreamerConfig(**kwargs)  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        cfg = _tiny_cfg()
        with pytest.raises(Exception):
            cfg.horizon = 3  # type: ignore[misc]


class TestDreamerForward:
    def test_output_shapes(self) -> None:
        model = DreamerModel(_tiny_cfg())
        obs, actions, _ = _batch()
        out = model(obs, actions)
        assert isinstance(out, DreamerOutput)
        assert out.observation.shape == (2, 3, 3, 64, 64)
        assert out.reward.shape == (2, 3)
        assert out.value.shape == (2, 3)
        assert out.posterior_stoch.shape == (2, 3, 4)
        assert out.deter.shape == (2, 3, 8)
        assert out.behavior is None

    def test_act_is_bounded(self) -> None:
        model = DreamerModel(_tiny_cfg())
        obs, actions, _ = _batch()
        _, posteriors = model.observe(obs, actions)
        a = model.act(posteriors)
        assert a.shape == (2, 3, 2)
        assert bool((a.abs() <= 1.0).all().item())

    def test_act_mean_only_is_deterministic(self) -> None:
        model = DreamerModel(_tiny_cfg())
        obs, actions, _ = _batch()
        _, posteriors = model.observe(obs, actions, sample=False)
        first = model.act(posteriors, sample=False)
        second = model.act(posteriors, sample=False)
        assert bool((first == second).all().item())

    def test_act_sampled_is_not_deterministic(self) -> None:
        model = DreamerModel(_tiny_cfg())
        obs, actions, _ = _batch()
        _, posteriors = model.observe(obs, actions)
        assert not bool((model.act(posteriors) == model.act(posteriors)).all().item())


class TestActorHead:
    def test_untrained_std_starts_near_init_std(self) -> None:
        """``softplus(raw + c) + min_std`` with ``c = log(exp(init_std) - 1)``."""
        cfg = _tiny_cfg(actor_init_std=5.0, actor_min_std=1e-4)
        actor = DreamerModel(cfg).actor
        _, std = actor.distribution(lucid.zeros((2, 3, cfg.latent_size)))
        assert 3.0 < float(std.mean().item()) < 7.0

    def test_init_std_moves_the_scale(self) -> None:
        wide = DreamerModel(_tiny_cfg(actor_init_std=5.0)).actor
        narrow = DreamerModel(_tiny_cfg(actor_init_std=0.5)).actor
        feature = lucid.zeros((2, 3, _tiny_cfg().latent_size))
        assert float(wide.distribution(feature)[1].mean().item()) > float(
            narrow.distribution(feature)[1].mean().item()
        )

    def test_mean_is_bounded_by_mean_scale(self) -> None:
        """``s * tanh(x / s)`` keeps the mean inside ``(-s, s)`` for any input."""
        cfg = _tiny_cfg(actor_mean_scale=5.0)
        actor = DreamerModel(cfg).actor
        huge = lucid.ones((2, 3, cfg.latent_size)) * 1e4
        mean, _ = actor.distribution(huge)
        assert bool((mean.abs() <= 5.0).all().item())


class TestLambdaReturn:
    @staticmethod
    def _closed_form(r: list[float], v: list[float], gamma: float, lam: float) -> list[float]:
        """The paper's definition, built directly from n-step returns."""
        horizon = len(r) - 1

        def n_step(t: int, k: int) -> float:
            return sum(gamma**i * r[t + i] for i in range(k)) + gamma**k * v[t + k]

        out = []
        for t in range(horizon):
            longest = horizon - t
            value = sum((1 - lam) * lam ** (n - 1) * n_step(t, n) for n in range(1, longest))
            out.append(value + lam ** (longest - 1) * n_step(t, longest))
        return out

    R = [0.5, -1.2, 2.0, 0.3, 1.1, -0.4]
    V = [0.9, 0.2, -0.7, 1.5, 0.1, 0.8]

    @pytest.mark.parametrize("gamma", [1.0, 0.99, 0.5])
    @pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
    def test_matches_closed_form(self, gamma: float, lam: float) -> None:
        mine = lambda_return(lucid.tensor([self.R]), lucid.tensor([self.V]), gamma, lam)
        expected = self._closed_form(self.R, self.V, gamma, lam)
        for got, want in zip([float(x) for x in mine[0]], expected):
            assert abs(got - want) < 1e-4

    def test_lambda_zero_is_one_step_td(self) -> None:
        mine = lambda_return(lucid.tensor([self.R]), lucid.tensor([self.V]), 0.99, 0.0)
        for t, got in enumerate([float(x) for x in mine[0]]):
            assert abs(got - (self.R[t] + 0.99 * self.V[t + 1])) < 1e-4

    def test_lambda_one_is_bootstrapped_monte_carlo(self) -> None:
        mine = lambda_return(lucid.tensor([self.R]), lucid.tensor([self.V]), 0.99, 1.0)
        horizon = len(self.R) - 1
        for t, got in enumerate([float(x) for x in mine[0]]):
            want = (
                sum(0.99**i * self.R[t + i] for i in range(horizon - t))
                + 0.99 ** (horizon - t) * self.V[horizon]
            )
            assert abs(got - want) < 1e-4

    def test_targets_include_the_starting_state(self) -> None:
        """Paper Algorithm 1 anchors the sum at the state you are in.

        "Imagine trajectories {(s_tau, a_tau)}_{tau=t}^{t+H} from each s_t
        ... Compute value estimates V_lambda(s_tau)" — tau starts at t, so
        the observed state gets a target. The released implementation's
        indexing starts one step in; this follows the paper. Imagining H
        steps from N starts therefore yields H targets, not H-1.
        """
        model = DreamerForWorldModeling(_tiny_cfg(horizon=5))
        out = model(*_batch(b=2, t=3))
        assert out.behavior is not None
        assert out.behavior.imagined_reward.shape == (6, 6)  # start + 5
        assert out.behavior.lambda_return.shape == (6, 5)  # one per step, incl. start

    def test_shape_drops_the_bootstrap(self) -> None:
        out = lambda_return(lucid.randn((3, 6)), lucid.randn((3, 6)), 0.99, 0.95)
        assert out.shape == (3, 5)

    def test_needs_two_states(self) -> None:
        with pytest.raises(ValueError):
            lambda_return(lucid.randn((2, 1)), lucid.randn((2, 1)), 0.99, 0.95)


class TestImagination:
    def test_shapes(self) -> None:
        model = DreamerModel(_tiny_cfg())
        obs, actions, _ = _batch()
        _, posteriors = model.observe(obs, actions)
        start = RSSMState(
            deter=posteriors.deter[:, -1],
            stoch=posteriors.stoch[:, -1],
            mean=posteriors.mean[:, -1],
            std=posteriors.std[:, -1],
        )
        states, imagined = model.imagine(start, 5)
        assert states.deter.shape == (2, 6, 8)
        assert states.stoch.shape == (2, 6, 4)
        assert imagined.shape == (2, 5, 2)

    def test_start_state_is_carried_unchanged(self) -> None:
        """Index 0 is the state handed in, not a step taken from it."""
        model = DreamerModel(_tiny_cfg())
        start = model.rssm.initial(2)
        states, _ = model.imagine(start, 3)
        assert bool((states.deter[:, 0] == start.deter).all().item())
        assert bool((states.stoch[:, 0] == start.stoch).all().item())

    def test_actions_come_from_the_actor(self) -> None:
        """The first imagined action is what the actor proposes at the start."""
        model = DreamerModel(_tiny_cfg(mean_only=True))
        start = model.rssm.initial(2)
        _, imagined = model.imagine(start, 3, sample=False)
        feature = start.feature.reshape(2, 1, -1)
        expected = model.actor(feature, sample=False)[:, 0]
        assert bool((imagined[:, 0] - expected).abs().max().item() < 1e-5)

    def test_rejects_zero_horizon(self) -> None:
        model = DreamerModel(_tiny_cfg())
        with pytest.raises(ValueError):
            model.imagine(model.rssm.initial(2), 0)


class TestMeanOnly:
    """``mean_only`` must reach *every* draw, not just the filtering one.

    This slipped through the first time: ``observe`` honoured the config
    and ``imagine`` did not, so a model declared deterministic still
    sampled its way through imagination — the one place Dreamer does most
    of its drawing. Nothing about the shapes or the losses looked wrong.
    """

    def test_imagine_is_reproducible(self) -> None:
        model = DreamerModel(_tiny_cfg(mean_only=True))
        start = model.rssm.initial(2)
        first_states, first_actions = model.imagine(start, 4)
        second_states, second_actions = model.imagine(start, 4)
        assert bool((first_states.stoch == second_states.stoch).all().item())
        assert bool((first_actions == second_actions).all().item())

    def test_sampling_model_is_not_reproducible(self) -> None:
        """Guards the test above — the default must actually draw."""
        model = DreamerModel(_tiny_cfg(mean_only=False))
        start = model.rssm.initial(2)
        first, _ = model.imagine(start, 4)
        second, _ = model.imagine(start, 4)
        assert not bool((first.stoch == second.stoch).all().item())

    def test_explicit_argument_overrides_the_config(self) -> None:
        model = DreamerModel(_tiny_cfg(mean_only=False))
        start = model.rssm.initial(2)
        first, _ = model.imagine(start, 4, sample=False)
        second, _ = model.imagine(start, 4, sample=False)
        assert bool((first.stoch == second.stoch).all().item())

    def test_behaviour_pass_is_reproducible(self) -> None:
        """The whole objective, end to end, with every draw pinned."""
        model = DreamerForWorldModeling(_tiny_cfg(mean_only=True))
        obs, actions, rewards = _batch()
        first = model(obs, actions, rewards)
        second = model(obs, actions, rewards)
        assert first.behavior is not None and second.behavior is not None
        assert float(first.loss.item()) == float(second.loss.item())
        assert float(first.behavior.actor_loss.item()) == float(second.behavior.actor_loss.item())


class TestObjective:
    def test_world_loss_decomposes(self) -> None:
        model = DreamerForWorldModeling(_tiny_cfg())
        out = model(*_batch())
        assert out.loss is not None and out.kl_loss is not None
        total = (
            float(out.recon_loss.item())
            + float(out.reward_loss.item())
            + model.config.kl_weight * float(out.kl_loss.item())
        )
        assert abs(float(out.loss.item()) - total) < 1e-2

    def test_behavior_is_populated(self) -> None:
        model = DreamerForWorldModeling(_tiny_cfg())
        out = model(*_batch())
        assert isinstance(out.behavior, DreamerBehaviorOutput)
        b = out.behavior
        assert b.lambda_return.shape == (6, 4)
        assert b.imagined_reward.shape == (6, 5)
        assert b.imagined_value.shape == (6, 5)
        assert b.imagined_action.shape == (6, 4, 2)
        assert b.actor_loss.ndim == 0 and b.value_loss.ndim == 0

    def test_every_filtered_step_starts_an_imagination(self) -> None:
        """``B * T`` starts, so the behaviour sees every state the model filtered."""
        model = DreamerForWorldModeling(_tiny_cfg())
        out = model(*_batch(b=3, t=4))
        assert out.behavior is not None
        assert out.behavior.lambda_return.shape[0] == 12

    def test_actor_loss_is_negative_mean_return(self) -> None:
        model = DreamerForWorldModeling(_tiny_cfg(discount=1.0))
        out = model(*_batch())
        assert out.behavior is not None
        expected = -float(out.behavior.lambda_return.mean().item())
        assert abs(float(out.behavior.actor_loss.item()) - expected) < 1e-3

    def test_value_loss_is_non_negative(self) -> None:
        model = DreamerForWorldModeling(_tiny_cfg())
        out = model(*_batch())
        assert out.behavior is not None
        assert float(out.behavior.value_loss.item()) >= 0.0

    def test_free_nats_can_zero_the_kl(self) -> None:
        model = DreamerForWorldModeling(_tiny_cfg(free_nats=1e6))
        out = model(*_batch())
        assert out.kl_loss is not None
        assert float(out.kl_loss.item()) == pytest.approx(0.0, abs=1e-5)


class TestDiscountHead:
    """``pcont`` — the discount the model predicts instead of assuming.

    The paper: "In tasks with early termination, the world model also
    predicts the discount factor from each latent state." Without it a
    constant gamma has the planner keep collecting reward past the end of
    the episode.
    """

    def test_absent_by_default(self) -> None:
        model = DreamerForWorldModeling(_tiny_cfg())
        assert model.dreamer.pcont_head is None
        out = model(*_batch())
        assert out.pcont_loss is None
        assert out.behavior is not None and out.behavior.imagined_pcont is None

    def test_predict_pcont_raises_without_a_head(self) -> None:
        model = DreamerModel(_tiny_cfg())
        with pytest.raises(ValueError):
            model.predict_pcont(model.rssm.initial(2))

    def test_requires_discounts(self) -> None:
        """Defaulting to 'never terminates' would train the head to a constant."""
        model = DreamerForWorldModeling(_tiny_cfg(pcont=True))
        obs, actions, rewards = _batch()
        with pytest.raises(ValueError):
            model(obs, actions, rewards)

    def test_populates_its_loss_and_prediction(self) -> None:
        model = DreamerForWorldModeling(_tiny_cfg(pcont=True))
        obs, actions, rewards = _batch()
        out = model(obs, actions, rewards, lucid.ones((2, 3)))
        assert out.pcont_loss is not None
        assert out.behavior is not None
        assert out.behavior.imagined_pcont is not None
        assert out.behavior.imagined_pcont.shape == (4, 5)

    def test_drops_the_last_filtered_step(self) -> None:
        """It may be terminal, and imagining onward from it trains a fiction."""
        plain = DreamerForWorldModeling(_tiny_cfg())
        with_pcont = DreamerForWorldModeling(_tiny_cfg(pcont=True))
        obs, actions, rewards = _batch(b=2, t=3)
        a = plain(obs, actions, rewards)
        b = with_pcont(obs, actions, rewards, lucid.ones((2, 3)))
        assert a.behavior is not None and b.behavior is not None
        assert a.behavior.lambda_return.shape[0] == 6  # B * T
        assert b.behavior.lambda_return.shape[0] == 4  # B * (T - 1)

    def test_rejects_a_sequence_too_short_to_drop_from(self) -> None:
        model = DreamerForWorldModeling(_tiny_cfg(pcont=True))
        obs, actions, rewards = _batch(b=2, t=1)
        with pytest.raises(ValueError):
            model(obs, actions, rewards, lucid.ones((2, 1)))

    def test_head_joins_the_world_parameter_group(self) -> None:
        model = DreamerForWorldModeling(_tiny_cfg(pcont=True))
        world = {id(p) for p in model.world_parameters()}
        assert {id(p) for p in model.dreamer.pcont_head.parameters()} <= world
        actor = {id(p) for p in model.actor_parameters()}
        value = {id(p) for p in model.value_parameters()}
        assert world | actor | value == {id(p) for p in model.parameters()}
        assert not world & actor and not world & value

    def test_termination_cuts_the_future(self) -> None:
        """The semantics, not the plumbing: gamma == 0 leaves only the reward."""
        reward = lucid.tensor([[0.5, -1.2, 2.0, 0.3]])
        value = lucid.tensor([[0.9, 0.2, -0.7, 1.5]])
        dead = lambda_return(reward, value, lucid.zeros((1, 4)), 0.95)
        for t, got in enumerate([float(x) for x in dead[0]]):
            assert abs(got - float(reward[0, t])) < 1e-5

    def test_constant_tensor_matches_a_scalar_discount(self) -> None:
        reward, value = lucid.randn((3, 6)), lucid.randn((3, 6))
        scalar = lambda_return(reward, value, 0.99, 0.95)
        tensor = lambda_return(reward, value, lucid.ones((3, 6)) * 0.99, 0.95)
        assert float((scalar - tensor).abs().max().item()) < 1e-4


class TestActorInputDetach:
    """``detach_actor_input`` — the one behavioural fork with the reference.

    The released implementation feeds the actor a ``stop_gradient``-ed
    state during imagination. Both settings must still train the actor;
    what changes is which chain-rule terms survive.
    """

    def _actor_grads(self, detach: bool, weights: object) -> list[lucid.Tensor]:
        model = DreamerForWorldModeling(_tiny_cfg(detach_actor_input=detach, mean_only=True))
        model.load_state_dict(weights)
        out = model(*_batch())
        assert out.behavior is not None
        model.zero_grad()
        out.behavior.actor_loss.backward()
        return [p.grad.clone() for p in model.actor_parameters()]

    def test_flag_changes_the_gradient(self) -> None:
        weights = DreamerForWorldModeling(
            _tiny_cfg(detach_actor_input=True, mean_only=True)
        ).state_dict()
        on = self._actor_grads(True, weights)
        off = self._actor_grads(False, weights)
        assert max(float((a - b).abs().max().item()) for a, b in zip(on, off)) > 1e-9

    @pytest.mark.parametrize("detach", [True, False])
    def test_actor_still_learns_either_way(self, detach: bool) -> None:
        weights = DreamerForWorldModeling(
            _tiny_cfg(detach_actor_input=True, mean_only=True)
        ).state_dict()
        grads = self._actor_grads(detach, weights)
        assert sum(float((g**2).sum().item()) for g in grads) > 0


class TestGradientRouting:
    GROUPS = ("encoder", "rssm", "decoder", "reward_head", "value_head", "actor")

    def _reached(self, loss_name: str) -> set[str]:
        model = DreamerForWorldModeling(_tiny_cfg())
        out = model(*_batch())
        assert out.behavior is not None and out.loss is not None
        loss = out.loss if loss_name == "world" else getattr(out.behavior, loss_name)
        model.zero_grad()
        loss.backward()
        return {
            name
            for name in self.GROUPS
            if any(
                p.grad is not None and float(abs(p.grad).sum().item()) > 0
                for p in getattr(model.dreamer, name).parameters()
            )
        }

    def test_world_loss_reaches_only_the_world_model(self) -> None:
        assert self._reached("world") == {"encoder", "rssm", "decoder", "reward_head"}

    def test_actor_loss_flows_through_the_dynamics(self) -> None:
        """The paper's analytic path — not an accident, and not to the encoder."""
        reached = self._reached("actor_loss")
        assert "actor" in reached
        assert "rssm" in reached, "actor gradient must reach the learned dynamics"
        assert "reward_head" in reached and "value_head" in reached
        assert "encoder" not in reached, "imagination starts detached"
        assert "decoder" not in reached

    def test_value_loss_reaches_only_the_critic(self) -> None:
        assert self._reached("value_loss") == {"value_head"}


class TestParameterGroups:
    def test_groups_partition_the_model(self) -> None:
        model = DreamerForWorldModeling(_tiny_cfg())
        world = {id(p) for p in model.world_parameters()}
        actor = {id(p) for p in model.actor_parameters()}
        value = {id(p) for p in model.value_parameters()}
        everything = {id(p) for p in model.parameters()}
        assert not world & actor and not world & value and not actor & value
        assert world | actor | value == everything

    def test_groups_are_non_empty(self) -> None:
        model = DreamerForWorldModeling(_tiny_cfg())
        assert model.world_parameters()
        assert model.actor_parameters()
        assert model.value_parameters()


class TestComposition:
    """Pin the composition, not the shape.

    A duplicated or dropped activation is invisible under an idempotent
    one. Dreamer's default is ELU, which is not idempotent, but these run
    across several activations anyway so the check cannot come to depend on
    that.
    """

    @pytest.mark.parametrize("act_fn", ["elu", "relu", "silu", "gelu"])
    def test_actor_equals_hand_assembled(self, act_fn: str) -> None:
        cfg = _tiny_cfg(act_fn=act_fn)
        model = DreamerModel(cfg)
        actor = model.actor
        feature = lucid.randn((2, 3, cfg.latent_size))

        from lucid.models._utils._generative import generative_activation

        h = feature.reshape(6, cfg.latent_size)
        for layer in actor.head.layers:
            h = generative_activation(act_fn, layer(h))
        raw = actor.head.out(h).reshape(2, 3, 2 * cfg.action_dim)
        expected_mean = cfg.actor_mean_scale * lucid.tanh(
            raw[..., : cfg.action_dim] / cfg.actor_mean_scale
        )

        mean, _ = actor.distribution(feature)
        assert float((mean - expected_mean).abs().max().item()) < 1e-5

    def test_activation_choice_actually_changes_the_actor(self) -> None:
        """Guards the test above: it must be able to tell activations apart.

        Same weights, different activation. If these agreed, the
        composition test would pass on a model that ignored ``act_fn``
        entirely — which is the shape the VQ-VAE encoder bug took.
        """
        feature = lucid.randn((2, 3, _tiny_cfg().latent_size))
        reference = DreamerModel(_tiny_cfg(act_fn="relu"))
        weights = reference.state_dict()
        baseline = reference.actor.distribution(feature)[0]

        for act_fn in ("elu", "silu", "gelu"):
            other = DreamerModel(_tiny_cfg(act_fn=act_fn))
            other.load_state_dict(weights)
            spread = float((baseline - other.actor.distribution(feature)[0]).abs().max())
            assert spread > 1e-4, f"{act_fn} is indistinguishable from relu"


class TestRegistry:
    def test_factories_are_registered(self) -> None:
        assert is_model("dreamer") and is_model("dreamer_world_model")
        listed = list_models()
        assert "dreamer" in listed and "dreamer_world_model" in listed

    def test_create_model_applies_overrides(self) -> None:
        model = create_model("dreamer", action_dim=4, horizon=7)
        assert isinstance(model, DreamerModel)
        assert model.config.action_dim == 4 and model.config.horizon == 7

    def test_world_modeling_factory(self) -> None:
        model = create_model("dreamer_world_model", **_tiny_cfg().__dict__)
        assert isinstance(model, DreamerForWorldModeling)

    def test_pretrained_is_refused(self) -> None:
        with pytest.raises(Exception):
            create_model("dreamer", pretrained=True)


class TestTrainingStep:
    """The end-to-end contract: three losses, three groups, one graph.

    Both hand-rolled ways of spending these losses are wrong, so the
    tests below pin what ``backward`` must produce rather than merely
    that it runs.
    """

    KW = dict(mean_only=True)

    def _reference_grads(
        self, weights: object, batch: tuple[lucid.Tensor, ...]
    ) -> dict[str, list[object]]:
        """Each loss backpropagated alone, in its own model, on one batch."""
        picks = {
            "world": lambda o: o.loss,
            "actor": lambda o: o.behavior.actor_loss,
            "value": lambda o: o.behavior.value_loss,
        }
        out: dict[str, list[object]] = {}
        for name, pick in picks.items():
            model = DreamerForWorldModeling(_tiny_cfg(**self.KW))
            model.load_state_dict(weights)
            result = model(*batch)
            model.zero_grad()
            pick(result).backward()
            group = getattr(model, f"{name}_parameters")()
            out[name] = [None if p.grad is None else p.grad.clone() for p in group]
        return out

    def test_matches_three_independent_backward_passes(self) -> None:
        model = DreamerForWorldModeling(_tiny_cfg(**self.KW))
        weights = model.state_dict()
        # The same batch on both sides — different data would produce
        # different gradients and the comparison would mean nothing.
        batch = _batch()
        reference = self._reference_grads(weights, batch)

        out = model(*batch)
        model.backward(out)

        for name, expected in reference.items():
            got = getattr(model, f"{name}_parameters")()
            assert any(p.grad is not None for p in got), f"{name} got no gradient"
            for want, param in zip(expected, got):
                if want is None:
                    continue
                # Compared relatively, with a tolerance set from measurement
                # rather than taste. Over eight trials the two paths — one
                # backward alone, one with the graph retained across three —
                # differ by at most 0.9% on these gradients, which are small
                # enough (1e-4) that float32 noise on O(1) intermediates
                # lands there. A mis-routed gradient differs by its whole
                # magnitude, so 5% sits 5x above the noise and 20x below
                # anything real. `test_the_comparison_catches_contamination`
                # holds that second claim up.
                scale = float(want.abs().max().item())
                if scale < 1e-8:
                    continue
                error = float((want - param.grad).abs().max().item())
                assert error / scale < 5e-2, f"{name}: {error / scale:.3e}"

    def test_world_group_is_not_contaminated_by_the_actor(self) -> None:
        """The failure this method exists to prevent, asserted directly."""
        model = DreamerForWorldModeling(_tiny_cfg(**self.KW))
        weights = model.state_dict()
        batch = _batch()

        alone = DreamerForWorldModeling(_tiny_cfg(**self.KW))
        alone.load_state_dict(weights)
        result = alone(*batch)
        alone.zero_grad()
        result.loss.backward()
        clean = [
            float(p.grad.abs().sum().item())
            for p in alone.dreamer.rssm.parameters()
            if p.grad is not None
        ]

        out = model(*batch)
        model.backward(out)
        via = [
            float(p.grad.abs().sum().item())
            for p in model.dreamer.rssm.parameters()
            if p.grad is not None
        ]
        assert clean and len(clean) == len(via)
        for a, b in zip(clean, via):
            assert abs(a - b) < 1e-6

    def test_naive_accumulation_really_would_contaminate(self) -> None:
        """Guards the test above — otherwise it proves nothing."""
        model = DreamerForWorldModeling(_tiny_cfg(**self.KW))
        out = model(*_batch())
        params = [p for p in model.dreamer.rssm.parameters()]

        model.zero_grad()
        out.loss.backward(retain_graph=True)
        clean = [float(p.grad.abs().sum().item()) for p in params if p.grad is not None]
        out.behavior.actor_loss.backward(retain_graph=True)
        mixed = [float(p.grad.abs().sum().item()) for p in params if p.grad is not None]
        assert any(abs(a - b) > 1e-9 for a, b in zip(clean, mixed))

    def test_rejects_an_output_without_losses(self) -> None:
        wrapper = DreamerForWorldModeling(_tiny_cfg())
        plain = DreamerModel(_tiny_cfg())
        with pytest.raises(ValueError):
            wrapper.backward(plain(*_batch()[:2]))

    def test_three_optimisers_move_their_own_group(self) -> None:
        import lucid.optim as optim

        model = DreamerForWorldModeling(_tiny_cfg())
        groups = {
            "world": model.world_parameters(),
            "actor": model.actor_parameters(),
            "value": model.value_parameters(),
        }
        before = {k: [p.data.clone() for p in v] for k, v in groups.items()}
        optimisers = {k: optim.Adam(v, lr=1e-2) for k, v in groups.items()}

        out = model(*_batch())
        model.backward(out)
        for opt in optimisers.values():
            opt.step()

        for name, params in groups.items():
            moved = any(float(abs(p.data - old).sum()) > 0 for p, old in zip(params, before[name]))
            assert moved, f"{name} did not move"

    def test_losses_fall_over_a_short_run(self) -> None:
        """A few steps on fixed data must reduce what the world model fits."""
        import lucid.optim as optim

        lucid.manual_seed(0)
        model = DreamerForWorldModeling(_tiny_cfg(horizon=3))
        optimisers = [
            optim.Adam(model.world_parameters(), lr=6e-4),
            optim.Adam(model.value_parameters(), lr=8e-5),
            optim.Adam(model.actor_parameters(), lr=8e-5),
        ]
        obs = lucid.rand((2, 3, 3, 64, 64))
        actions = lucid.rand((2, 3, 2)) * 2 - 1
        rewards = obs.reshape(2, 3, -1).mean(dim=-1) * 10.0

        first = last = None
        for step in range(12):
            out = model(obs, actions, rewards)
            if step == 0:
                first = (float(out.recon_loss.item()), float(out.reward_loss.item()))
            last = (float(out.recon_loss.item()), float(out.reward_loss.item()))
            model.backward(out)
            for opt in optimisers:
                opt.step()

        assert last[0] < first[0], "reconstruction did not improve"
        assert last[1] < first[1], "reward prediction did not improve"


class TestDiscreteControl:
    """Appendix A's second setting, and the two halves of it that matter.

    The paper gives Atari and DeepMind Lab their own paragraph: "the action
    model predicts the logits of a categorical distribution.  We use
    straight-through gradients for the sampling step during latent
    imagination.  ... we use an imagination horizon of H = 10, scale the KL
    regularizers by beta = 0.1, and bound rewards using tanh.  We predict
    the discount factor from the latent state with a binary classifier."

    Both halves of the policy sentence are load-bearing and neither shows
    up in a shape check. The one-hot is what an Atari button is; the
    straight-through draw is what keeps the actor trainable, because
    Dreamer's gradient arrives *through* the action and a hard sample
    severs it. So the tests below check that the sample is one-hot **and**
    that a gradient survives it.
    """

    @staticmethod
    def _cfg(**overrides: object) -> DreamerConfig:
        return _tiny_cfg(action_dim=4, action_space="discrete", **overrides)

    @staticmethod
    def _actions(b: int = 2, t: int = 3, action_dim: int = 4) -> lucid.Tensor:
        index = lucid.zeros((b, t)).to(lucid.int32)
        return lucid.nn.functional.one_hot(index, num_classes=action_dim).to(lucid.float32)

    def test_the_action_is_a_one_hot(self) -> None:
        lucid.manual_seed(0)
        model = DreamerModel(self._cfg()).eval()
        _, posteriors = model.observe(lucid.randn((2, 3, 3, 64, 64)), self._actions())
        action = model.act(posteriors)
        assert action.shape == (2, 3, 4)
        assert float((action.sum(dim=-1) - 1.0).abs().max().item()) < 1e-5
        assert set(float(v) for v in action.reshape(-1).tolist()) <= {0.0, 1.0}

    def test_the_mode_is_also_a_one_hot(self) -> None:
        lucid.manual_seed(0)
        model = DreamerModel(self._cfg()).eval()
        _, posteriors = model.observe(lucid.randn((2, 3, 3, 64, 64)), self._actions())
        action = model.act(posteriors, sample=False)
        assert float((action.sum(dim=-1) - 1.0).abs().max().item()) < 1e-5

    def test_the_draw_carries_a_gradient(self) -> None:
        """Straight-through, which is the half a shape check cannot see.

        A hard one-hot has zero gradient everywhere. If the sample were
        drawn without the straight-through estimator the actor would still
        produce valid actions and never learn.
        """
        lucid.manual_seed(0)
        model = DreamerModel(self._cfg())
        feature = lucid.randn((2, 3, model.config.latent_size), requires_grad=True)
        model.actor(feature).sum().backward()
        assert feature.grad is not None
        assert float(feature.grad.abs().sum().item()) > 0.0

    def test_a_hard_draw_would_not(self) -> None:
        """Guards the test above — otherwise it would pass on any sampler."""
        lucid.manual_seed(0)
        model = DreamerModel(self._cfg())
        feature = lucid.randn((2, 3, model.config.latent_size), requires_grad=True)
        logits = model.actor.logits(feature)
        index = logits.argmax(dim=-1)
        hard = lucid.nn.functional.one_hot(index, num_classes=4).to(lucid.float32)
        assert not hard.requires_grad
        with pytest.raises(RuntimeError, match="does not require grad"):
            hard.sum().backward()

    def test_the_two_parameterisations_are_exclusive(self) -> None:
        """Each raises on the other's accessor rather than returning nonsense."""
        discrete = DreamerModel(self._cfg()).actor
        continuous = DreamerModel(_tiny_cfg()).actor
        feature_d = lucid.zeros((1, 1, discrete.head.out.in_features))
        with pytest.raises(ValueError):
            discrete.distribution(feature_d)
        feature_c = lucid.zeros((1, 1, continuous.head.out.in_features))
        with pytest.raises(ValueError):
            continuous.logits(feature_c)

    def test_the_head_is_narrower_than_the_gaussian_one(self) -> None:
        """One score per button, not a location and a scale per dimension."""
        assert DreamerModel(self._cfg()).actor.head.out.out_features == 4
        assert DreamerModel(_tiny_cfg(action_dim=4)).actor.head.out.out_features == 8

    def test_the_objectives_run_and_route(self) -> None:
        lucid.manual_seed(0)
        model = DreamerForWorldModeling(self._cfg(pcont=True))
        output = model(
            lucid.randn((2, 3, 3, 64, 64)), self._actions(), lucid.randn((2, 3)), lucid.ones((2, 3))
        )
        assert output.loss is not None and output.pcont_loss is not None
        model.backward(output)
        touched = sum(
            1
            for p in model.actor_parameters()
            if p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
        )
        assert touched == len(model.actor_parameters())

    @pytest.mark.parametrize("space", ["continuous", "discrete"])
    def test_the_continuous_setting_is_untouched(self, space: str) -> None:
        """Adding the second setting must not move the first."""
        cfg = _tiny_cfg() if space == "continuous" else self._cfg()
        assert cfg.action_space == space
        assert DreamerConfig(action_dim=2).action_space == "continuous"
        assert DreamerConfig(action_dim=2).horizon == 15
        assert DreamerConfig(action_dim=2).kl_weight == 1.0

    def test_rejects_an_unknown_action_space(self) -> None:
        with pytest.raises(ValueError):
            _tiny_cfg(action_space="ternary")


class TestDiscreteRegistry:
    def test_the_factories_carry_the_papers_four_changes(self) -> None:
        model = create_model("dreamer_discrete", action_dim=18)
        cfg = model.config
        assert cfg.action_space == "discrete"
        assert cfg.horizon == 10  # "an imagination horizon of H = 10"
        assert cfg.kl_weight == 0.1  # "scale the KL regularizers by beta = 0.1"
        assert cfg.pcont is True  # "predict the discount factor ..."

    def test_it_differs_from_the_control_suite_setting_in_exactly_those(self) -> None:
        """Nothing else about the network changes between the two settings."""
        from dataclasses import fields

        base = create_model("dreamer", action_dim=18).config
        discrete = create_model("dreamer_discrete", action_dim=18).config
        moved = {f.name for f in fields(base) if getattr(base, f.name) != getattr(discrete, f.name)}
        assert moved == {"action_space", "horizon", "kl_weight", "pcont"}

    def test_both_tasks_are_registered(self) -> None:
        assert "dreamer_discrete" in list_models()
        assert "dreamer_discrete_world_model" in list_models()

    def test_pretrained_weights_are_refused(self) -> None:
        with pytest.raises(NotImplementedError):
            dreamer_discrete(pretrained=True)
