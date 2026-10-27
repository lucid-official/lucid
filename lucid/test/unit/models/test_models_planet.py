"""Unit tests for PlaNet (Hafner et al., 2019).

The gradient-routing tests are the load-bearing ones, for the same reason
they were in VQ-VAE: the reconstruction path never touches the prior head,
so a model with the KL term removed or mis-wired still returns correct
shapes, a finite loss, and a plausible reconstruction — and never learns to
predict anything.

The composition tests exist because a duplicated activation is invisible
under ReLU, which is idempotent and is this family's default. They pin the
exact composition instead of the output shape.
"""

import pytest

import lucid
import lucid.nn.functional as F
from lucid.models import (
    PlaNetConfig,
    PlaNetForWorldModeling,
    PlaNetModel,
    create_model,
    is_model,
    list_models,
)
from lucid.models.generative.planet import PlaNetOutput
from lucid.models._utils._generative import generative_activation
from lucid.models.generative._common._rssm import RSSM, _gumbel_argmax, categorical_kl, rssm_kl


def _tiny_cfg(**overrides: object) -> PlaNetConfig:
    base: dict[str, object] = {
        "action_dim": 2,
        "stoch_size": 4,
        "deter_size": 8,
        "hidden_size": 8,
        "cnn_depth": 4,
        "reward_hidden": 8,
    }
    base.update(overrides)
    return PlaNetConfig(**base)  # type: ignore[arg-type]


def _batch(b: int = 2, t: int = 3, action_dim: int = 2) -> tuple[lucid.Tensor, ...]:
    return (lucid.randn((b, t, 3, 64, 64)), lucid.randn((b, t, action_dim)), lucid.randn((b, t)))


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestPlaNetConfig:
    def test_defaults_match_the_paper(self) -> None:
        cfg = PlaNetConfig()
        assert cfg.stoch_size == 30
        assert cfg.deter_size == 200
        assert cfg.hidden_size == 200
        assert cfg.reward_hidden == 200
        assert cfg.reward_layers == 2
        assert cfg.free_nats == 3.0
        assert cfg.min_std == 0.1
        assert cfg.act_fn == "relu"
        assert cfg.sample_size == 64
        assert cfg.model_type == "planet"

    def test_derived_sizes(self) -> None:
        cfg = PlaNetConfig()
        assert cfg.embed_size == 32 * cfg.cnn_depth == 1024
        assert cfg.latent_size == cfg.deter_size + cfg.stoch_size == 230

    def test_only_64px_is_accepted(self) -> None:
        # The decoder kernel schedule (5, 5, 6, 6) reaches 64 and nothing
        # else; refusing beats silently reconstructing the wrong shape.
        with pytest.raises(ValueError, match="64x64"):
            PlaNetConfig(sample_size=32)
        with pytest.raises(ValueError, match="64x64"):
            PlaNetConfig(sample_size=(64, 32))
        assert PlaNetConfig(sample_size=(64, 64)).sample_size == (64, 64)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("action_dim", 0),
            ("stoch_size", 0),
            ("deter_size", 0),
            ("hidden_size", 0),
            ("cnn_depth", 0),
            ("reward_hidden", 0),
            ("reward_layers", 0),
            ("min_std", 0.0),
            ("free_nats", -1.0),
            ("kl_weight", -1.0),
        ],
    )
    def test_rejects_invalid_fields(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field):
            PlaNetConfig(**{field: value})  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        cfg = PlaNetConfig()
        with pytest.raises(Exception):
            cfg.stoch_size = 4  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Shapes
# ─────────────────────────────────────────────────────────────────────────────


class TestPlaNetForward:
    def test_forward_shapes(self) -> None:
        cfg = _tiny_cfg()
        model = PlaNetModel(cfg).eval()
        obs, act, _ = _batch()
        out = model(obs, act)

        assert isinstance(out, PlaNetOutput)
        assert out.observation.shape == (2, 3, 3, 64, 64)
        assert out.reward.shape == (2, 3)
        assert out.posterior_stoch.shape == (2, 3, cfg.stoch_size)
        assert out.prior_mean.shape == (2, 3, cfg.stoch_size)
        assert out.deter.shape == (2, 3, cfg.deter_size)

    def test_encode_decode_reward_shapes(self) -> None:
        cfg = _tiny_cfg()
        model = PlaNetModel(cfg).eval()
        obs, act, _ = _batch()

        assert model.encode(obs).shape == (2, 3, cfg.embed_size)
        _, posteriors = model.observe(obs, act)
        assert model.decode(posteriors).shape == (2, 3, 3, 64, 64)
        assert model.predict_reward(posteriors).shape == (2, 3)
        assert posteriors.feature.shape == (2, 3, cfg.latent_size)

    def test_bare_model_builds_no_objective(self) -> None:
        out = PlaNetModel(_tiny_cfg()).eval()(*_batch()[:2])
        assert out.loss is None
        assert out.recon_loss is None
        assert out.kl_loss is None

    def test_std_is_always_positive(self) -> None:
        cfg = _tiny_cfg()
        out = PlaNetModel(cfg).eval()(*_batch()[:2])
        assert float(out.posterior_std.min().item()) >= cfg.min_std
        assert float(out.prior_std.min().item()) >= cfg.min_std

    def test_imagine_runs_without_observations(self) -> None:
        model = PlaNetModel(_tiny_cfg()).eval()
        actions = lucid.randn((2, 5, 2))
        imagined = model.imagine(model.rssm.initial(2), actions)
        assert imagined.stoch.shape == (2, 5, 4)
        assert imagined.deter.shape == (2, 5, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics semantics
# ─────────────────────────────────────────────────────────────────────────────


class TestDynamics:
    def test_prior_and_posterior_share_the_deterministic_path(self) -> None:
        # The observation refines the belief about s_t; it does not change
        # the path that led there. A shape check cannot see this.
        model = PlaNetModel(_tiny_cfg()).eval()
        priors, posteriors = model.observe(*_batch()[:2])
        assert float((priors.deter - posteriors.deter).abs().max().item()) == 0.0

    def test_imagine_matches_a_manual_prior_unroll(self) -> None:
        model = PlaNetModel(_tiny_cfg()).eval()
        actions = lucid.randn((2, 4, 2))

        lucid.manual_seed(0)
        rolled = model.imagine(model.rssm.initial(2), actions)

        lucid.manual_seed(0)
        state = model.rssm.initial(2)
        steps = []
        for t in range(4):
            state = model.rssm.prior_step(state, actions[:, t])
            steps.append(state.stoch)
        manual = lucid.stack(steps, dim=1)

        assert float((rolled.stoch - manual).abs().max().item()) < 1e-6

    def test_actions_change_the_state(self) -> None:
        # Pins the action-alignment contract: actions[:, t] is consumed at
        # step t. If the unroll dropped or mis-indexed them, this passes
        # nothing and every shape still checks out.
        model = PlaNetModel(_tiny_cfg()).eval()
        obs = lucid.randn((2, 1, 3, 64, 64))

        lucid.manual_seed(0)
        _, zero = model.observe(obs, lucid.zeros(2, 1, 2))
        lucid.manual_seed(0)
        _, nonzero = model.observe(obs, lucid.ones(2, 1, 2) * 5.0)

        assert float((zero.deter - nonzero.deter).abs().max().item()) > 1e-4

    def test_kl_is_non_negative_and_free_nats_clamps_it(self) -> None:
        model = PlaNetModel(_tiny_cfg()).eval()
        priors, posteriors = model.observe(*_batch()[:2])

        raw = rssm_kl(posteriors, priors, free_nats=0.0)
        assert float(raw.item()) >= 0.0
        assert float(rssm_kl(posteriors, priors, free_nats=1e9).item()) == 0.0
        # Not exactly zero: normal_kl's terms cancel analytically but not in
        # float32, leaving round-off around 1e-8.
        assert float(rssm_kl(posteriors, posteriors, free_nats=0.0).item()) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Gradient routing — what makes the objective necessary
# ─────────────────────────────────────────────────────────────────────────────


class TestGradientRouting:
    def test_reconstruction_alone_never_reaches_the_prior_head(self) -> None:
        # Nothing reconstructed is computed from the prior — the decoder
        # and the reward head both read the posterior. So the KL is the
        # prior's only teacher, and a model trained without it predicts
        # nothing while every shape and loss still looks healthy.
        model = PlaNetModel(_tiny_cfg())
        obs, act, _ = _batch()
        out = model(obs, act)
        ((out.observation - obs) ** 2).mean().backward()

        prior_grads = [p.grad for n, p in model.named_parameters() if "prior_head" in n]
        assert prior_grads and all(g is None for g in prior_grads)

        posterior_grads = [p.grad for n, p in model.named_parameters() if "posterior_head" in n]
        assert posterior_grads and all(g is not None for g in posterior_grads)

    def test_the_kl_is_what_trains_the_prior_head(self) -> None:
        model = PlaNetModel(_tiny_cfg())
        priors, posteriors = model.observe(*_batch()[:2])
        rssm_kl(posteriors, priors, free_nats=0.0).backward()

        reached = [n for n, p in model.named_parameters() if p.grad is not None]
        assert any("prior_head" in n for n in reached)

    def test_full_objective_reaches_every_parameter(self) -> None:
        model = PlaNetForWorldModeling(_tiny_cfg())
        obs, act, rew = _batch()
        out = model(obs, act, rewards=rew)
        assert out.loss is not None
        out.loss.backward()

        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, f"no gradient reached: {missing}"

    def test_gradient_reaches_the_recurrence_from_the_last_step(self) -> None:
        # A broken unroll — an accidental detach, or rebuilding the carried
        # state from a stacked tensor — still returns the right shapes and
        # simply stops learning across time.
        model = PlaNetModel(_tiny_cfg())
        obs, act, _ = _batch(t=4)
        _, posteriors = model.observe(obs, act)
        posteriors.stoch[:, -1].sum().backward()

        assert model.rssm.cell.weight_hh.grad is not None
        assert float(abs(model.rssm.cell.weight_hh.grad).sum()) > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────


class TestObjective:
    def test_losses_populated_only_with_rewards(self) -> None:
        model = PlaNetForWorldModeling(_tiny_cfg()).eval()
        obs, act, rew = _batch()

        without = model(obs, act)
        assert without.reward_loss is None
        assert without.loss is not None

        with_rewards = model(obs, act, rewards=rew)
        assert with_rewards.reward_loss is not None

    def test_total_is_the_sum_of_its_terms(self) -> None:
        cfg = _tiny_cfg(free_nats=0.0, kl_weight=1.0, overshoot_distance=1)
        out = PlaNetForWorldModeling(cfg).eval()(*_batch())

        assert out.loss is not None and out.recon_loss is not None
        assert out.kl_loss is not None and out.reward_loss is not None
        expected = (
            float(out.recon_loss.item()) + float(out.kl_loss.item()) + float(out.reward_loss.item())
        )
        assert abs(float(out.loss.item()) - expected) < 1e-2

    def test_kl_weight_scales_the_divergence(self) -> None:
        obs, act, rew = _batch()
        lucid.manual_seed(0)
        a = PlaNetForWorldModeling(
            _tiny_cfg(free_nats=0.0, kl_weight=0.0, overshoot_distance=1)
        ).eval()(obs, act, rewards=rew)
        assert a.loss is not None and a.kl_loss is not None
        assert a.recon_loss is not None and a.reward_loss is not None
        # With the weight at zero the KL is reported but does not enter.
        assert (
            abs(float(a.loss.item()) - (float(a.recon_loss.item()) + float(a.reward_loss.item())))
            < 1e-3
        )


# ─────────────────────────────────────────────────────────────────────────────
# Composition — what a shape check cannot see
# ─────────────────────────────────────────────────────────────────────────────


class TestComposition:
    @pytest.mark.parametrize("act", ["relu", "silu", "gelu"])
    def test_encoder_activates_once_per_convolution(self, act: str) -> None:
        model = PlaNetModel(_tiny_cfg(act_fn=act)).eval()
        enc = model.encoder
        x = lucid.randn((1, 1, 3, 64, 64))

        h = x.reshape(1, 3, 64, 64)
        for conv in enc.convs:
            h = generative_activation(act, conv(h))
        want = h.reshape(1, 1, model.config.embed_size)

        assert float((enc(x) - want).abs().max().item()) < 1e-4

    @pytest.mark.parametrize("act", ["relu", "silu", "gelu"])
    def test_decoder_leaves_its_last_layer_unactivated(self, act: str) -> None:
        # The final transposed convolution emits the reconstruction mean of
        # a unit-variance Gaussian, not a hidden layer — activating it
        # would clamp the output to a half-line.
        cfg = _tiny_cfg(act_fn=act)
        model = PlaNetModel(cfg).eval()
        dec = model.decoder
        feature = lucid.randn((1, 1, cfg.latent_size))

        h = dec.lift(feature.reshape(1, cfg.latent_size))
        h = h.reshape(1, cfg.embed_size, 1, 1)
        last = len(dec.deconvs) - 1
        for i, deconv in enumerate(dec.deconvs):
            h = deconv(h)
            if i != last:
                h = generative_activation(act, h)
        want = h.reshape(1, 1, 3, 64, 64)

        assert float((dec(feature) - want).abs().max().item()) < 1e-4

    def test_decoder_output_is_not_one_sided(self) -> None:
        out = PlaNetModel(_tiny_cfg()).eval()(*_batch()[:2])
        assert float(out.observation.min().item()) < 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestRSSMCategorical:
    """The seam DreamerV2 needs, tested where the RSSM lives.

    PlaNet never asks for a categorical latent, but it shares the class
    that provides one — so the Gaussian path has to keep behaving exactly
    as before, and the discrete path has to actually work. Both are
    asserted here rather than waiting for a family to depend on them.
    """

    def _rssm(self, discrete: int) -> RSSM:
        return RSSM(
            stoch_size=4, deter_size=8, hidden_size=8, action_dim=2, embed_size=6, discrete=discrete
        )

    def test_gaussian_is_unchanged(self) -> None:
        model = self._rssm(0).eval()
        _, posterior = model.observe(lucid.randn((2, 3, 6)), lucid.randn((2, 3, 2)))
        assert posterior.stoch.shape == (2, 3, 4)
        assert not posterior.is_discrete
        assert posterior.logits is None
        assert posterior.mean is not None and posterior.std is not None

    def test_categorical_shapes(self) -> None:
        model = self._rssm(5).eval()
        _, posterior = model.observe(lucid.randn((2, 3, 6)), lucid.randn((2, 3, 2)))
        assert posterior.is_discrete
        assert posterior.logits.shape == (2, 3, 4, 5)
        assert posterior.stoch.shape == (2, 3, 20)  # flattened 4 x 5
        assert posterior.mean is None and posterior.std is None
        assert posterior.feature.shape == (2, 3, 28)  # deter 8 + stoch 20

    def test_every_variable_is_one_hot(self) -> None:
        model = self._rssm(5).eval()
        _, posterior = model.observe(lucid.randn((2, 3, 6)), lucid.randn((2, 3, 2)))
        grid = posterior.stoch.reshape(2, 3, 4, 5)
        assert float((grid.sum(dim=-1) - 1.0).abs().max().item()) < 1e-5

    def test_straight_through_reaches_the_head(self) -> None:
        """A one-hot has no gradient; the estimator is what supplies one.

        Weighted, not summed: softmax rows total 1 whatever the logits
        are, so ``stoch.sum()`` has zero gradient by construction and
        would pass on a model whose latent was cut off entirely.
        """
        model = self._rssm(5)
        _, posterior = model.observe(lucid.randn((2, 3, 6)), lucid.randn((2, 3, 2)))
        weights = lucid.randn((2, 3, 20))
        model.zero_grad()
        (posterior.stoch * weights).sum().backward()
        reached = sum(
            float(p.grad.abs().sum().item())
            for p in model.posterior_head.parameters()
            if p.grad is not None
        )
        assert reached > 1e-3

    def test_mean_only_is_deterministic_and_binary(self) -> None:
        model = self._rssm(5).eval()
        embed, actions = lucid.randn((2, 2, 6)), lucid.randn((2, 2, 2))
        first = model.observe(embed, actions, sample=False)[1].stoch
        second = model.observe(embed, actions, sample=False)[1].stoch
        assert bool((first == second).all().item())
        assert bool(((first == 0) | (first == 1)).all().item())

    def test_sampling_varies(self) -> None:
        """Guards the test above — otherwise it would pass on a dead sampler."""
        model = self._rssm(5).eval()
        embed, actions = lucid.randn((2, 2, 6)), lucid.randn((2, 2, 2))
        first = model.observe(embed, actions, sample=True)[1].stoch
        second = model.observe(embed, actions, sample=True)[1].stoch
        assert not bool((first == second).all().item())

    @pytest.mark.parametrize("discrete", [-1, 1])
    def test_rejects_a_degenerate_grid(self, discrete: int) -> None:
        with pytest.raises(ValueError):
            self._rssm(discrete)

    def test_kl_refuses_a_mismatched_pair(self) -> None:
        """A Gaussian prior against a categorical posterior is a bug, not a number."""
        gaussian = self._rssm(0).eval()
        discrete = self._rssm(5).eval()
        _, g_post = gaussian.observe(lucid.randn((2, 2, 6)), lucid.randn((2, 2, 2)))
        _, d_post = discrete.observe(lucid.randn((2, 2, 6)), lucid.randn((2, 2, 2)))
        with pytest.raises(ValueError):
            rssm_kl(d_post, g_post)

    def test_categorical_kl_matches_a_hand_computation(self) -> None:
        import math

        q = [[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]]
        p = [[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]]
        expected = sum(sum(a * math.log(a / b) for a, b in zip(qv, pv)) for qv, pv in zip(q, p))
        got = float(
            categorical_kl(
                lucid.tensor([[[math.log(x) for x in v] for v in q]]),
                lucid.tensor([[[math.log(x) for x in v] for v in p]]),
            ).item()
        )
        assert abs(got - expected) < 1e-5

    def test_categorical_kl_is_zero_against_itself(self) -> None:
        logits = lucid.randn((2, 3, 4, 5))
        assert float(categorical_kl(logits, logits).abs().max().item()) < 1e-6


class TestUnimix:
    """A uniform floor under every categorical.

    Without it a class can reach probability zero, and a divergence
    measured against a zero is unbounded — which is the tuning problem
    DreamerV3 removes rather than tunes around.
    """

    def _peaked(self, unimix: float) -> RSSM:
        """An RSSM whose head is scaled up until its output saturates."""
        model = RSSM(
            stoch_size=2,
            deter_size=8,
            hidden_size=8,
            action_dim=2,
            embed_size=4,
            discrete=4,
            unimix=unimix,
        ).eval()
        for parameter in list(model.posterior_head.parameters()) + list(
            model.prior_head.parameters()
        ):
            parameter.data.copy_(parameter.data * 60.0)
        return model

    def test_the_floor_binds(self) -> None:
        model = self._peaked(0.1)
        _, posterior = model.observe(lucid.randn((300, 1, 4)), lucid.randn((300, 1, 2)))
        probabilities = F.softmax(posterior.logits, dim=-1)
        assert float(probabilities.min().item()) >= 0.1 / 4 - 1e-6

    def test_without_it_a_class_reaches_zero(self) -> None:
        """Guards the test above — otherwise it passes on any model."""
        model = self._peaked(0.0)
        _, posterior = model.observe(lucid.randn((300, 1, 4)), lucid.randn((300, 1, 2)))
        probabilities = F.softmax(posterior.logits, dim=-1)
        assert float(probabilities.min().item()) < 1e-6

    def test_it_bounds_the_divergence(self) -> None:
        """The point of the floor, not a side effect of it."""
        unbounded = self._peaked(0.0)
        floored = self._peaked(0.01)
        embed, actions = lucid.randn((300, 1, 4)), lucid.randn((300, 1, 2))
        wild = float(
            categorical_kl(*(s.logits for s in reversed(unbounded.observe(embed, actions))))
            .max()
            .item()
        )
        tame = float(
            categorical_kl(*(s.logits for s in reversed(floored.observe(embed, actions))))
            .max()
            .item()
        )
        assert tame < wild / 10.0

    def test_it_stays_a_distribution(self) -> None:
        model = self._peaked(0.1)
        _, posterior = model.observe(lucid.randn((50, 1, 4)), lucid.randn((50, 1, 2)))
        probabilities = F.softmax(posterior.logits, dim=-1)
        assert float((probabilities.sum(dim=-1) - 1.0).abs().max().item()) < 1e-5

    def test_the_sample_is_still_one_hot(self) -> None:
        model = self._peaked(0.1)
        _, posterior = model.observe(lucid.randn((50, 1, 4)), lucid.randn((50, 1, 2)))
        grid = posterior.stoch.reshape(50, 1, 2, 4)
        assert float((grid.sum(dim=-1) - 1.0).abs().max().item()) < 1e-5

    @pytest.mark.parametrize("unimix", [-0.1, 1.0, 1.5])
    def test_rejects_a_bad_mixture(self, unimix: float) -> None:
        with pytest.raises(ValueError):
            RSSM(
                stoch_size=2,
                deter_size=8,
                hidden_size=8,
                action_dim=2,
                embed_size=4,
                discrete=4,
                unimix=unimix,
            )

    def test_default_is_off(self) -> None:
        """The earlier families must be untouched by this."""
        model = RSSM(
            stoch_size=2, deter_size=8, hidden_size=8, action_dim=2, embed_size=4, discrete=4
        )
        assert model.unimix == 0.0


class TestGumbelArgmax:
    """The categorical draw, and why it is not ``multinomial``.

    ``multinomial`` is data-dependent, so it returns on the CPU. A
    recurrence pays that synchronisation once per step and the imagination
    pays it again — measured at 11.5 s per DreamerV2 training step on
    Metal against 0.7 s on the CPU, which made the accelerator sixteen
    times slower than not using it. Gumbel-max is the same draw with
    ``rand``, ``log`` and ``argmax``, all device-resident: 38 ms.

    So the tests here are about the draw being *exact*, since that is what
    the substitution has to preserve.
    """

    def test_matches_the_target_distribution(self) -> None:
        import collections
        import math

        probabilities = [0.5, 0.3, 0.15, 0.05]
        logits = lucid.tensor([[math.log(p) for p in probabilities]] * 4000)
        drawn = _gumbel_argmax(logits)
        counts = collections.Counter(int(v) for v in drawn)
        for index, expected in enumerate(probabilities):
            empirical = counts[index] / 4000
            assert abs(empirical - expected) < 0.03, f"class {index}: {empirical:.3f} vs {expected}"

    def test_a_deterministic_logit_is_drawn_every_time(self) -> None:
        """Guards the test above: a broken draw could pass it by chance."""
        logits = lucid.tensor([[100.0, 0.0, 0.0]] * 200)
        assert all(int(v) == 0 for v in _gumbel_argmax(logits))

    def test_it_is_not_deterministic(self) -> None:
        """And the other way — a plain argmax would pass the test above."""
        logits = lucid.zeros((200, 4))
        first, second = _gumbel_argmax(logits), _gumbel_argmax(logits)
        assert not bool((first == second).all().item())

    def test_shape_drops_the_class_axis(self) -> None:
        assert _gumbel_argmax(lucid.zeros((3, 5, 7))).shape == (3, 5)

    def test_the_latent_stays_one_hot(self) -> None:
        model = RSSM(
            stoch_size=3, deter_size=8, hidden_size=8, action_dim=2, embed_size=6, discrete=4
        ).eval()
        _, posterior = model.observe(lucid.randn((4, 3, 6)), lucid.randn((4, 3, 2)))
        grid = posterior.stoch.reshape(4, 3, 3, 4)
        assert float((grid.sum(dim=-1) - 1.0).abs().max().item()) < 1e-5

    def test_every_class_gets_used(self) -> None:
        """A draw collapsed onto one class is still a valid one-hot."""
        model = RSSM(
            stoch_size=1, deter_size=8, hidden_size=8, action_dim=2, embed_size=4, discrete=4
        ).eval()
        _, posterior = model.observe(lucid.randn((800, 1, 4)), lucid.randn((800, 1, 2)))
        chosen = {int(v) for v in posterior.stoch.reshape(800, 4).argmax(dim=-1)}
        assert chosen == {0, 1, 2, 3}


class TestRegistry:
    @pytest.mark.parametrize("name", ["planet", "planet_world_model"])
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    def test_task_bucket(self) -> None:
        assert "planet_world_model" in list_models(task="world-modeling")
        assert "planet" in list_models(task="base")

    def test_create_model_defaults(self) -> None:
        model = create_model("planet")
        assert isinstance(model, PlaNetModel)
        assert model.config.deter_size == 200

    def test_config_override_through_registry(self) -> None:
        model = create_model("planet", action_dim=6, stoch_size=8)
        assert model.config.action_dim == 6
        assert model.config.stoch_size == 8

    @pytest.mark.parametrize("name", ["planet", "planet_world_model"])
    def test_pretrained_is_refused_rather_than_faked(self, name: str) -> None:
        with pytest.raises(NotImplementedError, match="No pretrained weights"):
            create_model(name, pretrained=True)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic rollouts
# ─────────────────────────────────────────────────────────────────────────────


class TestMeanOnly:
    def test_mean_only_is_reproducible(self) -> None:
        model = PlaNetModel(_tiny_cfg(mean_only=True)).eval()
        obs, act, _ = _batch()
        a = model.observe(obs, act)[1]
        b = model.observe(obs, act)[1]
        assert float((a.stoch - b.stoch).abs().max().item()) == 0.0

    def test_mean_only_takes_the_mean(self) -> None:
        model = PlaNetModel(_tiny_cfg(mean_only=True)).eval()
        _, post = model.observe(*_batch()[:2])
        assert float((post.stoch - post.mean).abs().max().item()) == 0.0

    def test_sampling_is_the_default(self) -> None:
        model = PlaNetModel(_tiny_cfg()).eval()
        obs, act, _ = _batch()
        a = model.observe(obs, act)[1]
        b = model.observe(obs, act)[1]
        assert float((a.stoch - b.stoch).abs().max().item()) > 0.0

    def test_per_call_override_beats_the_config(self) -> None:
        model = PlaNetModel(_tiny_cfg()).eval()  # config says sample
        obs, act, _ = _batch()
        a = model.observe(obs, act, sample=False)[1]
        b = model.observe(obs, act, sample=False)[1]
        assert float((a.stoch - b.stoch).abs().max().item()) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Latent overshooting
# ─────────────────────────────────────────────────────────────────────────────


class TestOvershooting:
    def test_one_step_rollout_reproduces_the_observed_prior(self) -> None:
        # The invariant that pins the indexing: advancing every posterior by
        # a single step must land exactly on what ``observe`` already called
        # the prior. Off-by-one in either the start state or the action
        # would still produce correctly shaped garbage.
        model = PlaNetModel(_tiny_cfg(mean_only=True)).eval()
        obs, act, _ = _batch(t=5)
        priors, posts = model.observe(obs, act)

        b, span = 2, 4
        state = posts.map(lambda x: x[:, :span].reshape(b * span, -1))
        step = model.rssm.prior_step(
            state, act[:, 1 : 1 + span].reshape(b * span, -1), sample=False
        )
        rolled = step.map(lambda x: x.reshape(b, span, -1))

        assert float((rolled.mean - priors.mean[:, 1:]).abs().max().item()) == 0.0
        assert float((rolled.deter - priors.deter[:, 1:]).abs().max().item()) == 0.0

    def test_overshooting_adds_a_term(self) -> None:
        obs, act, rew = _batch(t=5)
        lucid.manual_seed(0)
        on = PlaNetForWorldModeling(_tiny_cfg()).eval()(obs, act, rewards=rew)
        lucid.manual_seed(0)
        off = PlaNetForWorldModeling(_tiny_cfg(overshoot_distance=1)).eval()(obs, act, rewards=rew)
        assert on.loss is not None and off.loss is not None
        assert float(on.loss.item()) > float(off.loss.item())

    def test_reward_overshooting_adds_a_term(self) -> None:
        obs, act, rew = _batch(t=5)
        lucid.manual_seed(0)
        on = PlaNetForWorldModeling(_tiny_cfg()).eval()(obs, act, rewards=rew)
        lucid.manual_seed(0)
        off = PlaNetForWorldModeling(_tiny_cfg(overshoot_reward_weight=0.0)).eval()(
            obs, act, rewards=rew
        )
        assert on.loss is not None and off.loss is not None
        assert float(on.loss.item()) > float(off.loss.item())

    def test_too_short_a_sequence_skips_overshooting(self) -> None:
        # T = 2 leaves no distance beyond the ordinary one-step KL.
        obs, act, rew = _batch(t=2)
        lucid.manual_seed(0)
        a = PlaNetForWorldModeling(_tiny_cfg()).eval()(obs, act, rewards=rew)
        lucid.manual_seed(0)
        b = PlaNetForWorldModeling(_tiny_cfg(overshoot_distance=1)).eval()(obs, act, rewards=rew)
        assert a.loss is not None and b.loss is not None
        assert abs(float(a.loss.item()) - float(b.loss.item())) < 1e-4

    def test_overshooting_trains_the_dynamics_not_the_encoder(self) -> None:
        # The posterior is stop-gradiented in the multi-step term, so it
        # teaches the transition and never pulls the encoder toward being
        # easier to predict.
        model = PlaNetForWorldModeling(_tiny_cfg(overshoot_reward_weight=0.0))
        obs, act, _ = _batch(t=5)
        _, posts = model.planet.observe(obs, act)
        term, _ = model._overshooting(posts, act, None)
        assert term is not None
        term.backward()

        assert model.planet.rssm.prior_head[0].weight.grad is not None
        assert model.planet.encoder.convs[0].weight.grad is None

    def test_reward_loss_scale(self) -> None:
        obs, act, rew = _batch()
        lucid.manual_seed(0)
        one = PlaNetForWorldModeling(_tiny_cfg()).eval()(obs, act, rewards=rew)
        lucid.manual_seed(0)
        ten = PlaNetForWorldModeling(_tiny_cfg(reward_loss_scale=10.0)).eval()(
            obs, act, rewards=rew
        )
        assert one.reward_loss is not None and ten.reward_loss is not None
        ratio = float(ten.reward_loss.item()) / float(one.reward_loss.item())
        assert abs(ratio - 10.0) < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Planning
# ─────────────────────────────────────────────────────────────────────────────


class TestPlanner:
    def test_the_planner_defaults_are_the_papers(self) -> None:
        """The four numbers that decide what "planning" costs here.

        ``test_generative_provenance`` requires a recorded source for
        every field of every config, and these are not config fields —
        they are :meth:`plan`'s own defaults. So the inventory gate that
        catches a wrong ``horizon`` in ``PlaNetConfig`` does not reach
        the planning horizon, which is the one the paper's own results
        were produced with. Changing any of them silently would leave
        every gate green.

        Appendix A: "planning horizon H = 12, optimization iterations
        I = 10, candidate samples J = 1000, and refitting to the best
        K = 100".
        """
        import inspect

        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(PlaNetForWorldModeling.plan).parameters.items()
        }
        assert defaults["horizon"] == 12
        assert defaults["iterations"] == 10
        assert defaults["candidates"] == 1000
        assert defaults["elites"] == 100

    def test_returns_one_action_per_batch_item(self) -> None:
        model = PlaNetForWorldModeling(_tiny_cfg()).eval()
        action = model.plan(
            model.planet.rssm.initial(3), horizon=3, iterations=2, candidates=8, elites=2
        )
        assert action.shape == (3, 2)

    def test_rollouts_are_deterministic(self) -> None:
        # plan() imagines with sample=False; without that the search ranks
        # its own sampling noise rather than the actions.
        lucid.manual_seed(0)
        model = PlaNetForWorldModeling(_tiny_cfg()).eval()
        kw = dict(horizon=3, iterations=2, candidates=8, elites=2)
        lucid.manual_seed(1)
        a = model.plan(model.planet.rssm.initial(1), **kw)  # type: ignore[arg-type]
        lucid.manual_seed(1)
        b = model.plan(model.planet.rssm.initial(1), **kw)  # type: ignore[arg-type]
        assert float((a - b).abs().max().item()) == 0.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"elites": 9, "candidates": 8}, "cannot exceed"),
            ({"horizon": 0}, "horizon"),
            ({"iterations": 0}, "iterations"),
            ({"candidates": 0}, "candidates"),
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs: dict[str, object], match: str) -> None:
        model = PlaNetForWorldModeling(_tiny_cfg()).eval()
        base: dict[str, object] = {"horizon": 3, "iterations": 2, "candidates": 8, "elites": 2}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            model.plan(model.planet.rssm.initial(1), **base)  # type: ignore[arg-type]

    def test_planner_finds_high_reward_actions(self) -> None:
        """The only way to check a planner *plans*.

        On an untrained model the reward head barely responds to actions —
        measured, the action-driven share of return variance is ~0 — so the
        search has nothing to climb and "beats random" is unprovable. Give
        the reward a signal it can chase and the planner must find it.
        """
        lucid.manual_seed(0)
        model = PlaNetForWorldModeling(_tiny_cfg(reward_hidden=16, overshoot_distance=1))
        opt = lucid.optim.Adam(model.parameters(), lr=3e-3)
        model.train()
        for _ in range(120):
            act = lucid.randn((8, 4, 2))
            out = model(lucid.rand((8, 4, 3, 64, 64)), act, rewards=act[:, :, 0])
            opt.zero_grad()
            assert out.loss is not None
            out.loss.backward()
            opt.step()
        model.eval()

        chosen = model.plan(
            model.planet.rssm.initial(1), horizon=4, iterations=6, candidates=128, elites=16
        )
        assert float(chosen[0, 0].item()) > 0.5
