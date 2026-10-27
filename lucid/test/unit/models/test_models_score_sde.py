"""Unit tests for Score-SDE (Song et al., ICLR 2021).

Everything here trains and samples whether or not it is right — a sign
error in a drift, a square root where the paper has a square, a prior
with the wrong width all leave a model that runs. So the tests are built
on two things a wrong implementation cannot fake.

**The perturbation kernels are checked against integrating their own
SDEs.** Each SDE ships a closed form for `p_0t(x(t) | x(0))` so that
training needs one network call rather than a trajectory; if that form
and the SDE disagree, the model is trained on one process and sampled
from another.

**The samplers are checked against an analytic score.** For Gaussian
data under a linear SDE the true score is available in closed form, so
the reverse SDE and the probability-flow ODE can be asked to recover a
distribution that is known exactly. This is the only test here that
would catch a sign error in the reverse-time drift, and it catches it in
all three SDEs at once.
"""

import pytest

import lucid
from lucid.diffeq import odeint
from lucid.models import (
    ScoreSDEConfig,
    ScoreSDEForImageGeneration,
    ScoreSDEModel,
    create_model,
    is_model,
    list_models,
    score_sde_vp,
)
from lucid.models.generative.score_sde._sde import SubVPSDE, VPSDE, make_sde

_TINY = dict(
    sample_size=8,
    base_channels=8,
    channel_mult=(1,),
    num_res_blocks=1,
    resnet_groups=4,
    attention_resolutions=(),
    num_scales=20,
)
_KINDS = ["vp", "ve", "subvp"]


class TestScoreSDEConfig:
    def test_defaults_are_the_papers(self) -> None:
        cfg = ScoreSDEConfig()
        assert cfg.sde_type == "vp"
        assert (cfg.beta_min, cfg.beta_max) == (0.1, 20.0)
        assert cfg.sigma_min == 0.01
        assert cfg.num_scales == 1000

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sde_type": "heat"},
            {"sigma_min": 0.0},
            {"sigma_max": 0.001},
            {"beta_min": 30.0},
            {"num_scales": 0},
            {"snr": 0.0},
            {"corrector_steps": -1},
        ],
    )
    def test_rejects_bad_config(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            ScoreSDEConfig(**kwargs)  # type: ignore[arg-type]


class TestSDEs:
    """The three processes, against invariants rather than against prose."""

    @pytest.mark.parametrize("kind", _KINDS)
    def test_the_kernel_matches_integrating_the_sde(self, kind: str) -> None:
        """A closed form that disagrees with its own SDE trains one model
        and samples another."""
        lucid.manual_seed(0)
        sde = make_sde(kind)
        count, end, steps = 4000, 0.6, 1500
        x = lucid.ones((count, 1))
        step = (end - sde.t_min) / steps
        t = lucid.ones((count,)) * sde.t_min
        for _ in range(steps):
            g = sde.diffusion(t).reshape(count, 1)
            x = x + sde.drift(x, t) * step + g * (step**0.5) * lucid.randn((count, 1))
            t = t + step
        mean, std = sde.marginal_prob(lucid.ones((count, 1)), lucid.ones((count,)) * end)
        assert abs(float(x.mean().item()) - float(mean.mean().item())) < 0.05
        assert abs(float(x.std().item()) / float(std.mean().item()) - 1.0) < 0.06

    def test_sub_vp_variance_is_bounded_by_vp(self) -> None:
        """The paper's reason for the name, and for the SDE existing."""
        vp, sub = VPSDE(), SubVPSDE()
        x = lucid.ones((16, 1))
        for moment in (0.2, 0.5, 0.9):
            t = lucid.ones((16,)) * moment
            assert float(sub.marginal_prob(x, t)[1].mean().item()) < float(
                vp.marginal_prob(x, t)[1].mean().item()
            )

    def test_ve_leaves_the_mean_alone(self) -> None:
        """'Variance exploding' — the drift is zero by construction."""
        sde = make_sde("ve")
        x = lucid.randn((8, 3))
        mean, _ = sde.marginal_prob(x, lucid.ones((8,)) * 0.7)
        assert float((mean - x).abs().max().item()) == 0.0
        assert float(sde.drift(x, lucid.ones((8,)) * 0.7).abs().max().item()) == 0.0

    def test_the_priors_have_the_widths_the_sdes_need(self) -> None:
        """A VE sampler started from a standard normal would silently fail."""
        lucid.manual_seed(0)
        wide = make_sde("ve").prior_sampling((4000, 1), "cpu")
        unit = make_sde("vp").prior_sampling((4000, 1), "cpu")
        assert 45.0 < float(wide.std().item()) < 55.0
        assert 0.9 < float(unit.std().item()) < 1.1

    def test_it_rejects_an_unknown_sde(self) -> None:
        with pytest.raises(ValueError):
            make_sde("brownian")


class TestSamplersAgainstAnAnalyticScore:
    r"""Gaussian data has a closed-form score, so the answer is known.

    Under a linear SDE the marginal of :math:`\mathcal{N}(\mu, s_0^2)`
    stays Gaussian, so the true score can be written down and handed to
    the samplers. Recovering the distribution then tests the reverse-time
    drift, the diffusion coefficient and the prior at once — none of
    which a shape check or a falling loss can reach.
    """

    MU, S0, COUNT = 2.0, 0.5, 8000

    def _score(self, sde: object, x: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
        mean_coeff, std = sde.marginal_prob(lucid.ones((self.COUNT, 1)), t)  # type: ignore[attr-defined]
        scale = mean_coeff.reshape(self.COUNT, 1)
        variance = std.reshape(self.COUNT, 1) ** 2 + (scale * self.S0) ** 2
        return -(x - scale * self.MU) / variance

    @pytest.mark.parametrize("kind", _KINDS)
    def test_the_reverse_sde_recovers_the_distribution(self, kind: str) -> None:
        lucid.manual_seed(0)
        sde = make_sde(kind)
        steps = 300
        x = sde.prior_sampling((self.COUNT, 1), "cpu")
        times = lucid.linspace(1.0, sde.t_min, steps)
        step = (1.0 - sde.t_min) / (steps - 1)
        for i in range(steps):
            t = lucid.ones((self.COUNT,)) * float(times[i].item())
            g = sde.diffusion(t).reshape(self.COUNT, 1)
            drift = sde.drift(x, t) - g**2 * self._score(sde, x, t)
            x = x + drift * (-step) + g * (step**0.5) * lucid.randn((self.COUNT, 1))
        assert abs(float(x.mean().item()) - self.MU) < 0.1
        assert abs(float(x.std().item()) - self.S0) < 0.08

    @pytest.mark.parametrize("kind", _KINDS)
    def test_the_probability_flow_ode_recovers_it_too(self, kind: str) -> None:
        """Equation 13's claim: the deterministic process shares the marginals."""
        lucid.manual_seed(0)
        sde = make_sde(kind)

        def rhs(moment: lucid.Tensor, flat: lucid.Tensor) -> lucid.Tensor:
            x = flat.reshape(self.COUNT, 1)
            t = lucid.ones((self.COUNT,)) * float(moment.item())
            g = sde.diffusion(t).reshape(self.COUNT, 1)
            return (sde.drift(x, t) - 0.5 * g**2 * self._score(sde, x, t)).reshape(-1)

        start = sde.prior_sampling((self.COUNT, 1), "cpu").reshape(-1)
        path = odeint(rhs, start, lucid.linspace(1.0, sde.t_min, 150), method="rk4")
        x = path[-1]
        assert abs(float(x.mean().item()) - self.MU) < 0.12
        assert abs(float(x.std().item()) - self.S0) < 0.08

    def test_a_flipped_reverse_drift_would_not(self) -> None:
        """Guards the two tests above.

        Dropping the score term leaves a valid-looking integration that
        does not recover anything — without this, a sampler that ignored
        the model entirely could pass.
        """
        lucid.manual_seed(0)
        sde = make_sde("vp")
        steps = 300
        x = sde.prior_sampling((self.COUNT, 1), "cpu")
        times = lucid.linspace(1.0, sde.t_min, steps)
        step = (1.0 - sde.t_min) / (steps - 1)
        for i in range(steps):
            t = lucid.ones((self.COUNT,)) * float(times[i].item())
            g = sde.diffusion(t).reshape(self.COUNT, 1)
            x = x + sde.drift(x, t) * (-step) + g * (step**0.5) * lucid.randn((self.COUNT, 1))
        assert abs(float(x.mean().item()) - self.MU) > 0.5


class TestModel:
    @pytest.mark.parametrize("kind", _KINDS)
    def test_the_score_is_minus_noise_over_sigma(self, kind: str) -> None:
        """The parameterisation every sampler here depends on."""
        lucid.manual_seed(0)
        model = create_model(f"score_sde_{kind}", **_TINY).eval()
        x = lucid.randn((2, 3, 8, 8))
        t = lucid.ones((2,)) * 0.5
        _, std = model.sde.marginal_prob(x, t)
        expected = -model.predict_noise(x, t) / std.reshape(2, 1, 1, 1)
        assert float((model.score(x, t) - expected).abs().max().item()) < 1e-5

    def test_the_loss_is_the_noise_error(self) -> None:
        """Sigma-squared weighting cancels both sigmas — DDPM's objective."""
        lucid.manual_seed(0)
        model = create_model("score_sde_vp_gen", **_TINY)
        out = model(lucid.randn((2, 3, 8, 8)))
        expected = ((out.noise - out.target) ** 2).reshape(2, -1).sum(dim=-1).mean()
        assert abs(float(out.loss.item()) - float(expected.item())) < 1e-4

    def test_the_loss_falls(self) -> None:
        # Each step draws its own time and noise, and two steps' losses
        # differ by more than ten steps of training move them.  The first
        # step's draws are replayed after training, so both numbers are the
        # same sample of the objective.
        import lucid.optim as optim

        lucid.manual_seed(0)
        model = create_model("score_sde_vp_gen", **_TINY)
        model.train()
        optimiser = optim.Adam(model.parameters(), lr=1e-3)
        x = lucid.randn((2, 3, 8, 8))
        draws = lucid.get_rng_state()
        first = 0.0
        for i in range(10):
            out = model(x)
            optimiser.zero_grad()
            out.loss.backward()
            optimiser.step()
            if i == 0:
                first = float(out.loss.item())
        lucid.set_rng_state(draws)
        with lucid.no_grad():
            after = float(model(x).loss.item())
        assert after < first

    @pytest.mark.parametrize("method", ["euler", "pc", "ode"])
    def test_every_sampler_runs(self, method: str) -> None:
        lucid.manual_seed(0)
        model = create_model("score_sde_vp_gen", **_TINY).eval()
        samples = model.generate(2, method=method, steps=5).samples
        assert samples.shape == (2, 3, 8, 8)
        assert not bool(lucid.isnan(samples).any().item())

    def test_it_rejects_an_unknown_sampler(self) -> None:
        model = create_model("score_sde_vp_gen", **_TINY).eval()
        with pytest.raises(ValueError):
            model.generate(1, method="heun")

    def test_the_step_count_is_a_sampling_choice(self) -> None:
        """The framework's point: it is not baked into the trained model."""
        lucid.manual_seed(0)
        model = create_model("score_sde_vp_gen", **_TINY).eval()
        for steps in (3, 11):
            assert model.generate(1, method="euler", steps=steps).samples.shape == (
                1,
                3,
                8,
                8,
            )


class TestRegistry:
    @pytest.mark.parametrize("kind", _KINDS)
    def test_both_tasks_are_registered(self, kind: str) -> None:
        assert f"score_sde_{kind}" in list_models()
        assert f"score_sde_{kind}_gen" in list_models()
        assert is_model(f"score_sde_{kind}")

    def test_the_classes_are_what_the_registry_says(self) -> None:
        assert isinstance(create_model("score_sde_vp", **_TINY), ScoreSDEModel)
        assert isinstance(create_model("score_sde_ve_gen", **_TINY), ScoreSDEForImageGeneration)

    def test_pretrained_weights_are_refused(self) -> None:
        with pytest.raises(NotImplementedError):
            score_sde_vp(pretrained=True)


class TestExactLikelihood:
    r"""Appendix D.2 — what the deterministic sampler buys.

    The probability-flow ODE is a neural ODE, so the instantaneous change
    of variables gives an *exact* density rather than a bound. Both the
    integral and the Skilling-Hutchinson trace estimate can be subtly
    wrong and still return a plausible number, so the test is convergence
    to a case where the answer is known.
    """

    @staticmethod
    def _tiny() -> object:
        return create_model("score_sde_vp_gen", **_TINY).eval()

    def _analytic(self, model: object, x: lucid.Tensor) -> float:
        """With a zero score the flow is a linear contraction, so the
        likelihood is available in closed form."""
        import math

        sde = model.sde  # type: ignore[attr-defined]
        dims = 3 * 8 * 8
        integral = 0.5 * (sde.beta_max - sde.beta_min) + sde.beta_min
        end = x * math.exp(-0.5 * integral)
        prior = -0.5 * (
            dims * math.log(2.0 * math.pi) + float((end.reshape(1, -1) ** 2).sum().item())
        )
        return prior - 0.5 * integral * dims

    def test_it_converges_to_the_analytic_value(self) -> None:
        """A zero-initialised head makes the score exactly zero, which is
        the one case where the answer can be written down."""
        lucid.manual_seed(0)
        model = self._tiny()
        x = lucid.randn((1, 3, 8, 8))
        assert float(model.score_sde.score(x, lucid.ones((1,)) * 0.5).abs().max().item()) == 0.0
        target = self._analytic(model, x)
        coarse = abs(float(model.log_likelihood(x, steps=16).item()) - target)
        fine = abs(float(model.log_likelihood(x, steps=128).item()) - target)
        assert fine < coarse / 4.0

    def test_the_error_is_first_order_in_the_step(self) -> None:
        """Guards the test above: a tolerance alone would pass on a
        systematically wrong integral that happened to land close."""
        lucid.manual_seed(0)
        model = self._tiny()
        x = lucid.randn((1, 3, 8, 8))
        target = self._analytic(model, x)
        errors = [abs(float(model.log_likelihood(x, steps=n).item()) - target) for n in (32, 128)]
        assert 3.0 < errors[0] / errors[1] < 5.0

    def test_it_returns_one_number_per_sample(self) -> None:
        lucid.manual_seed(0)
        assert self._tiny().log_likelihood(lucid.randn((3, 3, 8, 8)), steps=3).shape == (3,)

    def test_the_prior_width_follows_the_sde(self) -> None:
        """VE's prior is as wide as sigma_max; assuming unit variance
        would silently bias every VE likelihood."""
        assert make_sde("ve").prior_variance == 50.0**2
        assert make_sde("vp").prior_variance == 1.0


class TestControllableGeneration:
    """The paper's conditional sampling, which needs no retraining.

    ``grad log p(x | y) = grad log p(x) + grad log p(y | x)``, so
    conditioning is one added term — "all achievable using a single
    unconditional score-based model without re-training".
    """

    @pytest.mark.parametrize("method", ["euler", "pc", "ode"])
    def test_guidance_changes_the_samples(self, method: str) -> None:
        lucid.manual_seed(0)
        model = create_model("score_sde_vp_gen", **_TINY).eval()

        def pull_to_one(x: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
            return (lucid.ones_like(x) - x) * 5.0

        lucid.manual_seed(1)
        plain = model.generate(1, method=method, steps=5).samples
        lucid.manual_seed(1)
        guided = model.generate(1, method=method, steps=5, guidance=pull_to_one).samples
        assert float((guided - plain).abs().max().item()) > 1e-4

    def test_guidance_pulls_in_the_direction_it_is_given(self) -> None:
        """Guards the test above — 'different' could be noise."""
        lucid.manual_seed(0)
        model = create_model("score_sde_vp_gen", **_TINY).eval()
        toward = lambda target: (lambda x, t: (lucid.ones_like(x) * target - x) * 20.0)
        lucid.manual_seed(2)
        high = model.generate(1, method="ode", steps=8, guidance=toward(3.0)).samples
        lucid.manual_seed(2)
        low = model.generate(1, method="ode", steps=8, guidance=toward(-3.0)).samples
        assert float(high.mean().item()) > float(low.mean().item())
