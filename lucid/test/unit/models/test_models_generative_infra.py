"""Unit tests for Phase 5 base layer — generative-domain infrastructure.

Mirrors ``test_models_text_infra.py``: covers configs, output dataclasses,
shared utils, schedulers, and the :class:`DiffusionMixin` sampling loop.
The first concrete family (VAE / DDPM / NCSN) gets its own test file.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.models import (
    DiffusionModelConfig,
    DiffusionModelOutput,
    GenerationOutput,
    GenerativeModelConfig,
    VAEOutput,
)
from lucid.models._mixins import DiffusionMixin
from lucid.models.generative import DDPMScheduler, DiffusionScheduler
from lucid.test._fixtures.devices import metal_available
from lucid.models._base import PretrainedModel
from lucid.models.generative._common._rssm import RSSM, rssm_kl
from lucid.models._utils import (
    extract_into_tensor,
    gaussian_kl_divergence,
    generative_activation,
    make_beta_schedule,
    reparameterize,
)
import dataclasses
from pathlib import Path

import lucid.models.generative as _generative
from lucid.models._registry import _REGISTRY

# Every generative family's network emits something the objective or the
# sampler adds to, compares with, or splits at the input's width — a velocity,
# a score, a noise, a reconstruction, a bijection's image, the next frame — so
# each config refuses an ``out_channels`` that does not fit (LCD-620,
# LCD-631).  A family that legitimately accepts another width, or has no
# channel fields at all, is listed here with the reason.
_CHANNELS_NOT_CHECKED: dict[str, str] = {
    "vjepa2_ac": "predicts V-JEPA 2 feature tokens; its config has no "
    "in_channels / out_channels pair to disagree"
}


def _generative_families() -> list[str]:
    root = Path(_generative.__file__).parent
    return sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))


def _default_config(family: str) -> GenerativeModelConfig:
    for entry in _REGISTRY.values():
        if entry.family == family and isinstance(entry.default_config, GenerativeModelConfig):
            return entry.default_config
    raise AssertionError(f"{family}: no registered generative default config")


_CHECKED_FAMILIES = [f for f in _generative_families() if f not in _CHANNELS_NOT_CHECKED]

# ─────────────────────────────────────────────────────────────────────────────
# Configs
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerativeConfigs:
    def test_generative_base_defaults(self) -> None:
        cfg = GenerativeModelConfig()
        assert cfg.sample_size == 32
        assert cfg.in_channels == 3
        assert cfg.out_channels == 3
        assert cfg.act_fn == "silu"
        assert cfg.model_type == "generative"

    def test_tuple_sample_size(self) -> None:
        cfg = GenerativeModelConfig(sample_size=(16, 24))
        assert cfg.sample_size == (16, 24)

    def test_negative_sample_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="sample_size"):
            GenerativeModelConfig(sample_size=0)

    def test_bad_tuple_rejected(self) -> None:
        with pytest.raises(ValueError, match="sample_size tuple"):
            GenerativeModelConfig(sample_size=(16, 0))

    def test_diffusion_defaults(self) -> None:
        cfg = DiffusionModelConfig()
        assert cfg.num_train_timesteps == 1000
        assert cfg.beta_schedule == "linear"
        assert cfg.prediction_type == "epsilon"

    def test_beta_invariant(self) -> None:
        with pytest.raises(ValueError, match="beta_start"):
            DiffusionModelConfig(beta_start=0.5, beta_end=0.3)

    def test_negative_timesteps_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_train_timesteps"):
            DiffusionModelConfig(num_train_timesteps=0)

    @pytest.mark.parametrize("family", _CHECKED_FAMILIES)
    def test_every_family_refuses_an_output_width_that_does_not_fit(self, family: str) -> None:
        # Accepted, the mismatch surfaced as a broadcast error deep in the
        # forward pass or the sampler (LCD-620), or was silently replaced by
        # the width the family derived (LCD-631).
        config = _default_config(family)
        with pytest.raises(ValueError, match="out_channels"):
            dataclasses.replace(config, out_channels=config.in_channels + 1)

    def test_the_unchecked_list_names_real_families(self) -> None:
        assert set(_CHANNELS_NOT_CHECKED) <= set(_generative_families())


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestBetaSchedule:
    def test_linear_endpoints(self) -> None:
        betas = make_beta_schedule(100, "linear", beta_start=1e-4, beta_end=0.02)
        assert tuple(betas.shape) == (100,)
        assert abs(float(betas[0].item()) - 1e-4) < 1e-6
        assert abs(float(betas[-1].item()) - 0.02) < 1e-6

    def test_cosine_monotonic_alpha_bar(self) -> None:
        # Cosine schedule's cumulative-α (ᾱ_t) must be monotonically decreasing.
        betas = make_beta_schedule(50, "cosine")
        alphas = 1.0 - betas
        cum: list[float] = []
        acc = 1.0
        for i in range(50):
            acc *= float(alphas[i].item())
            cum.append(acc)
        for i in range(1, 50):
            assert cum[i] < cum[i - 1], f"ᾱ not decreasing at i={i}"

    def test_unknown_schedule_rejected(self) -> None:
        with pytest.raises(ValueError, match="schedule"):
            make_beta_schedule(10, "exp")


class TestExtractIntoTensor:
    def test_shape_broadcast(self) -> None:
        arr = lucid.tensor([0.1, 0.2, 0.3, 0.4]).float()
        ts = lucid.tensor([0, 2, 3]).long()
        out = extract_into_tensor(arr, ts, (3, 4, 8, 8))
        assert tuple(out.shape) == (3, 1, 1, 1)
        assert abs(float(out[0, 0, 0, 0].item()) - 0.1) < 1e-6
        assert abs(float(out[1, 0, 0, 0].item()) - 0.3) < 1e-6
        assert abs(float(out[2, 0, 0, 0].item()) - 0.4) < 1e-6


class TestGaussianKL:
    def test_zero_for_standard_normal(self) -> None:
        # mu=0, logvar=0 → KL(N(0,I) ‖ N(0,I)) = 0
        mu = lucid.zeros((8, 4))
        logvar = lucid.zeros((8, 4))
        kl = gaussian_kl_divergence(mu, logvar)
        assert abs(float(kl.item())) < 1e-6

    def test_positive_for_displaced_mean(self) -> None:
        mu = lucid.ones((4, 4))  # μ = 1
        logvar = lucid.zeros((4, 4))
        kl = gaussian_kl_divergence(mu, logvar)
        # KL = 0.5 * sum(μ²) per sample = 0.5 * 4 = 2
        assert abs(float(kl.item()) - 2.0) < 1e-5

    def test_reduction_modes(self) -> None:
        mu = lucid.ones((4, 4))
        logvar = lucid.zeros((4, 4))
        per = gaussian_kl_divergence(mu, logvar, reduction="none")
        assert tuple(per.shape) == (4,)
        s = gaussian_kl_divergence(mu, logvar, reduction="sum")
        assert abs(float(s.item()) - 8.0) < 1e-5


class TestReparameterize:
    def test_shape_preserved(self) -> None:
        mu = lucid.zeros((4, 8))
        logvar = lucid.zeros((4, 8))
        z = reparameterize(mu, logvar)
        assert tuple(z.shape) == (4, 8)


class TestGenerativeActivation:
    def test_silu_alias(self) -> None:
        x = lucid.tensor([1.0, -1.0]).float()
        out_silu = generative_activation("silu", x)
        out_swish = generative_activation("swish", x)
        diff = float(((out_silu - out_swish) ** 2).sum().item())
        assert diff < 1e-10

    def test_unknown_rejected(self) -> None:
        with pytest.raises(ValueError, match="activation"):
            generative_activation("nope", lucid.tensor([0.0]).float())


# ─────────────────────────────────────────────────────────────────────────────
# TimestepEmbedding
# ─────────────────────────────────────────────────────────────────────────────


class TestTimestepEmbedding:
    def test_shape(self) -> None:
        te = nn.TimestepEmbedding(in_dim=16, out_dim=64).eval()
        out = te(lucid.tensor([0, 10, 100, 500]).long())
        assert tuple(out.shape) == (4, 64)

    def test_odd_in_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="even in_dim"):
            nn.TimestepEmbedding(in_dim=7, out_dim=32)

    def test_distinct_per_timestep(self) -> None:
        """Different t must map to different embedding vectors."""
        te = nn.TimestepEmbedding(in_dim=16, out_dim=32).eval()
        out = te(lucid.tensor([0, 1, 2, 3]).long())
        for i in range(1, 4):
            diff = float(((out[i] - out[0]) ** 2).sum().item())
            assert diff > 1e-6, f"row {i} equals row 0"

    def test_flip_sin_to_cos_swaps_sinusoid_halves(self) -> None:
        """``flip_sin_to_cos`` controls the raw sinusoid ordering: ``True``
        (default) → ``[cos, sin]``; ``False`` (DDPM / Ho 2020) → ``[sin, cos]``.
        The two halves must be swapped between the two settings."""
        cos_first = nn.TimestepEmbedding(in_dim=8, out_dim=8)
        sin_first = nn.TimestepEmbedding(in_dim=8, out_dim=8, flip_sin_to_cos=False)
        t = lucid.tensor([5, 250]).long()
        a = sin_first._sinusoidal(t).numpy()  # [sin, cos]
        b = cos_first._sinusoidal(t).numpy()  # [cos, sin]
        half = 4
        import numpy as np

        # sin_first's first half == cos_first's second half, and vice-versa.
        assert float(np.abs(a[:, :half] - b[:, half:]).max()) < 1e-6
        assert float(np.abs(a[:, half:] - b[:, :half]).max()) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# DDPM scheduler
# ─────────────────────────────────────────────────────────────────────────────


class TestDDPMScheduler:
    def test_default_timesteps(self) -> None:
        sched = DDPMScheduler(num_train_timesteps=100)
        assert tuple(sched.timesteps.shape) == (100,)
        # Default schedule is reverse (99 → 0).
        assert int(sched.timesteps[0].item()) == 99
        assert int(sched.timesteps[-1].item()) == 0

    def test_set_inference_timesteps(self) -> None:
        sched = DDPMScheduler(num_train_timesteps=100)
        sched.set_timesteps(10)
        assert tuple(sched.timesteps.shape) == (10,)

    def test_inference_exceeds_train_rejected(self) -> None:
        sched = DDPMScheduler(num_train_timesteps=10)
        with pytest.raises(ValueError, match="exceeds"):
            sched.set_timesteps(20)

    def test_add_noise_shape(self) -> None:
        sched = DDPMScheduler(num_train_timesteps=50)
        x0 = lucid.randn((2, 3, 8, 8))
        noise = lucid.randn((2, 3, 8, 8))
        t = lucid.tensor([10, 40]).long()
        xt = sched.add_noise(x0, noise, t)
        assert tuple(xt.shape) == (2, 3, 8, 8)

    def test_step_shape(self) -> None:
        sched = DDPMScheduler(num_train_timesteps=50)
        sample = lucid.randn((1, 3, 8, 8))
        pred = lucid.randn((1, 3, 8, 8))
        prev = sched.step(pred, 25, sample)
        assert tuple(prev.shape) == (1, 3, 8, 8)

    def test_step_t_zero_is_deterministic(self) -> None:
        """At t=0 the reverse step returns mean only (no Gaussian noise)."""
        sched = DDPMScheduler(num_train_timesteps=10)
        sample = lucid.zeros((1, 1, 4, 4))
        pred = lucid.zeros((1, 1, 4, 4))  # predicting zero noise
        out1 = sched.step(pred, 0, sample)
        out2 = sched.step(pred, 0, sample)
        diff = float(((out1 - out2) ** 2).sum().item())
        assert diff < 1e-10, "t=0 step should be deterministic"

    def test_prediction_type_validation(self) -> None:
        with pytest.raises(ValueError, match="prediction_type"):
            DDPMScheduler(prediction_type="bogus")

    def test_step_reads_the_stored_float32_schedule(self) -> None:
        """The host copy ``step`` reads is the table, value for value."""
        sched = DDPMScheduler(num_train_timesteps=50)
        assert sched._alphas_cumprod_host == sched.alphas_cumprod.tolist()


_METAL = ["cpu", "metal"] if metal_available() else ["cpu"]


class TestDDPMSchedulerDevice:
    """LCD-383: the schedule had no device — every table stayed on the CPU."""

    def _tables(self, sched: DDPMScheduler) -> set[str]:
        return {t.device.type for t in vars(sched).values() if isinstance(t, lucid.Tensor)}

    @pytest.mark.parametrize("device", _METAL)
    def test_built_on_a_device(self, device: str) -> None:
        sched = DDPMScheduler(num_train_timesteps=20, device=device)
        assert self._tables(sched) == {device}
        sched.set_timesteps(5)
        assert sched.timesteps.device.type == device

    @pytest.mark.parametrize("device", _METAL)
    def test_set_timesteps_moves_every_table(self, device: str) -> None:
        """``generate`` passes its device here; moving ``timesteps`` alone
        reported that device while ``add_noise`` kept copying CPU tables."""
        sched = DDPMScheduler(num_train_timesteps=20)
        sched.set_timesteps(5, device=device)
        assert self._tables(sched) == {device}
        assert sched.device.type == device

    def test_the_model_builds_its_scheduler_where_it_lives(self) -> None:
        from lucid.models.generative.ddpm import DDPMConfig, DDPMForImageGeneration

        cfg = DDPMConfig(
            sample_size=8,
            base_channels=16,
            channel_mult=(1,),
            num_res_blocks=1,
            resnet_groups=8,
            attention_resolutions=(),
            num_train_timesteps=10,
        )
        device = _METAL[-1]
        model = DDPMForImageGeneration(cfg).to(device)
        assert self._tables(model.make_scheduler()) == {device}

    @pytest.mark.parametrize("device", _METAL)
    def test_to_moves_every_table(self, device: str) -> None:
        sched = DDPMScheduler(num_train_timesteps=20).to(device)
        assert self._tables(sched) == {device}
        assert sched.to("cpu") is sched and self._tables(sched) == {"cpu"}

    @pytest.mark.parametrize("device", _METAL)
    def test_add_noise_follows_the_sample_once(self, device: str) -> None:
        sched = DDPMScheduler(num_train_timesteps=20)
        x0 = lucid.randn((2, 1, 4, 4), device=device)
        t = lucid.tensor([3, 17], device=device).long()
        first = sched.add_noise(x0, lucid.zeros_like(x0), t)
        assert first.device.type == device
        assert self._tables(sched) == {device}
        ref = DDPMScheduler(num_train_timesteps=20)
        want = ref.add_noise(x0.to("cpu"), lucid.zeros((2, 1, 4, 4)), t.to("cpu"))
        assert float((first.to("cpu") - want).abs().max().item()) < 1e-6

    @pytest.mark.parametrize("device", _METAL)
    def test_step_stays_on_the_sample_device(self, device: str) -> None:
        sched = DDPMScheduler(num_train_timesteps=20).to(device)
        sample = lucid.randn((1, 1, 4, 4), device=device)
        out = sched.step(lucid.zeros_like(sample), 7, sample)
        assert out.device.type == device


# ─────────────────────────────────────────────────────────────────────────────
# DiffusionMixin — end-to-end sampling loop
# ─────────────────────────────────────────────────────────────────────────────


class _DummyDiffusionModel(PretrainedModel, DiffusionMixin):
    """Identity-noise predictor over a single conv — minimal mixin host."""

    config_class = DiffusionModelConfig
    base_model_prefix = "dummy"

    def __init__(self, config: DiffusionModelConfig) -> None:
        super().__init__(config)
        self.conv = nn.Conv2d(config.in_channels, config.in_channels, 3, padding=1)

    def forward(self, sample, timestep):  # type: ignore[override]
        return DiffusionModelOutput(sample=self.conv(sample))


class TestDiffusionMixin:
    def _tiny_cfg(self) -> DiffusionModelConfig:
        return DiffusionModelConfig(sample_size=8, in_channels=3, num_train_timesteps=20)

    def test_generate_shape(self) -> None:
        cfg = self._tiny_cfg()
        m = _DummyDiffusionModel(cfg).eval()
        sched = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)
        out = m.generate(sched, n_samples=2, num_inference_steps=4)
        assert isinstance(out, GenerationOutput)
        assert tuple(out.samples.shape) == (2, 3, 8, 8)
        assert out.intermediates is None

    def test_generate_intermediates(self) -> None:
        cfg = self._tiny_cfg()
        m = _DummyDiffusionModel(cfg).eval()
        sched = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)
        out = m.generate(sched, n_samples=1, num_inference_steps=3, return_intermediates=True)
        assert out.intermediates is not None
        assert len(out.intermediates) == 3
        for inter in out.intermediates:
            assert tuple(inter.shape) == (1, 3, 8, 8)

    def test_the_model_can_build_the_scheduler_its_config_describes(self) -> None:
        """``clip_denoised`` had no route from the config to the sampler.

        Every other schedule field was on the config too and had to be
        retyped by the caller; a scheduler built with a different
        schedule from the trained one produces plausible noise rather
        than an error, so the mismatch is invisible.
        """
        from lucid.models.generative.ddpm import DDPMConfig, DDPMForImageGeneration

        cfg = DDPMConfig(
            sample_size=8,
            base_channels=16,
            channel_mult=(1,),
            num_res_blocks=1,
            resnet_groups=8,
            num_train_timesteps=20,
            beta_start=2e-4,
            beta_end=0.03,
            clip_denoised=False,
        )
        sched = DDPMForImageGeneration(cfg).make_scheduler()
        assert sched.clip_denoised is False
        assert (sched.num_train_timesteps, sched.beta_start, sched.beta_end) == (
            cfg.num_train_timesteps,
            cfg.beta_start,
            cfg.beta_end,
        )
        assert sched.prediction_type == cfg.prediction_type
        assert sched.beta_schedule == cfg.beta_schedule

    def test_generate_no_config_no_shape_raises(self) -> None:
        cfg = self._tiny_cfg()
        m = _DummyDiffusionModel(cfg).eval()
        m.config = None  # type: ignore[assignment]
        sched = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)
        with pytest.raises(RuntimeError, match="generator_shape"):
            m.generate(sched, n_samples=1, num_inference_steps=2)


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclasses sanity
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerativeOutputs:
    def test_diffusion_output(self) -> None:
        out = DiffusionModelOutput(sample=lucid.zeros((1, 3, 8, 8)))
        assert out.loss is None

    def test_vae_output_fields(self) -> None:
        out = VAEOutput(
            sample=lucid.zeros((1, 3, 8, 8)),
            latent=lucid.zeros((1, 8)),
            mu=lucid.zeros((1, 8)),
            logvar=lucid.zeros((1, 8)),
        )
        assert out.recon_loss is None and out.kl_loss is None

    def test_generation_output(self) -> None:
        out = GenerationOutput(samples=lucid.zeros((4, 3, 8, 8)))
        assert out.intermediates is None


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler ABC sanity
# ─────────────────────────────────────────────────────────────────────────────


class TestSchedulerABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            DiffusionScheduler()  # type: ignore[abstract]

    def test_ddpm_satisfies_contract(self) -> None:
        assert issubclass(DDPMScheduler, DiffusionScheduler)


class TestRSSM:
    """The recurrent state-space model shared by the world-model families.

    The structural properties here are the ones a shape check cannot see:
    that prior and posterior share the deterministic path, that the unroll
    keeps the autograd chain intact across time, and that the KL is the
    only thing training the prior.
    """

    @staticmethod
    def _rssm() -> RSSM:
        return RSSM(stoch_size=4, deter_size=8, hidden_size=8, action_dim=2, embed_size=6)

    def test_observe_shapes(self) -> None:
        rssm = self._rssm().eval()
        priors, posteriors = rssm.observe(lucid.randn((2, 5, 6)), lucid.randn((2, 5, 2)))
        for state in (priors, posteriors):
            assert state.stoch.shape == (2, 5, 4)
            assert state.deter.shape == (2, 5, 8)
            assert state.mean.shape == (2, 5, 4)
            assert state.std.shape == (2, 5, 4)

    def test_posterior_shares_the_deterministic_path(self) -> None:
        rssm = self._rssm().eval()
        priors, posteriors = rssm.observe(lucid.randn((2, 3, 6)), lucid.randn((2, 3, 2)))
        assert float((priors.deter - posteriors.deter).abs().max().item()) == 0.0

    def test_std_is_floored_at_min_std(self) -> None:
        rssm = RSSM(
            stoch_size=4, deter_size=8, hidden_size=8, action_dim=2, embed_size=6, min_std=0.5
        ).eval()
        _, posteriors = rssm.observe(lucid.randn((2, 3, 6)), lucid.randn((2, 3, 2)))
        assert float(posteriors.std.min().item()) >= 0.5

    def test_feature_concatenates_deter_then_stoch(self) -> None:
        rssm = self._rssm().eval()
        _, posteriors = rssm.observe(lucid.randn((1, 2, 6)), lucid.randn((1, 2, 2)))
        assert posteriors.feature.shape == (1, 2, 12)
        assert float((posteriors.feature[..., :8] - posteriors.deter).abs().max().item()) == 0.0

    def test_logvar_round_trips_the_std(self) -> None:
        rssm = self._rssm().eval()
        _, posteriors = rssm.observe(lucid.randn((1, 2, 6)), lucid.randn((1, 2, 2)))
        recovered = (0.5 * posteriors.logvar).exp()
        assert float((recovered - posteriors.std).abs().max().item()) < 1e-5

    def test_imagine_matches_a_manual_prior_unroll(self) -> None:
        rssm = self._rssm().eval()
        actions = lucid.randn((2, 4, 2))

        lucid.manual_seed(0)
        rolled = rssm.imagine(rssm.initial(2), actions)
        lucid.manual_seed(0)
        state = rssm.initial(2)
        steps = [(state := rssm.prior_step(state, actions[:, t])).stoch for t in range(4)]

        assert float((rolled.stoch - lucid.stack(steps, dim=1)).abs().max().item()) < 1e-6

    def test_gradient_survives_the_whole_unroll(self) -> None:
        # A detach anywhere in the loop, or rebuilding the carried state
        # from a stacked tensor, leaves every shape correct and silently
        # stops the model learning across time.
        rssm = self._rssm()
        _, posteriors = rssm.observe(lucid.randn((2, 4, 6)), lucid.randn((2, 4, 2)))
        posteriors.stoch[:, -1].sum().backward()

        assert rssm.cell.weight_hh.grad is not None
        assert float(abs(rssm.cell.weight_hh.grad).sum()) > 0.0

    def test_reconstruction_path_never_reaches_the_prior_head(self) -> None:
        rssm = self._rssm()
        _, posteriors = rssm.observe(lucid.randn((2, 3, 6)), lucid.randn((2, 3, 2)))
        posteriors.stoch.sum().backward()

        prior_grads = [p.grad for n, p in rssm.named_parameters() if "prior_head" in n]
        assert prior_grads and all(g is None for g in prior_grads)

    def test_kl_trains_the_prior_head(self) -> None:
        rssm = self._rssm()
        priors, posteriors = rssm.observe(lucid.randn((2, 3, 6)), lucid.randn((2, 3, 2)))
        rssm_kl(posteriors, priors, free_nats=0.0).backward()

        reached = [n for n, p in rssm.named_parameters() if p.grad is not None]
        assert any("prior_head" in n for n in reached)

    def test_rssm_kl_clamps_at_free_nats(self) -> None:
        rssm = self._rssm().eval()
        priors, posteriors = rssm.observe(lucid.randn((2, 3, 6)), lucid.randn((2, 3, 2)))

        assert float(rssm_kl(posteriors, priors, free_nats=0.0).item()) >= 0.0
        assert float(rssm_kl(posteriors, priors, free_nats=1e9).item()) == 0.0

    def test_initial_state_is_zero(self) -> None:
        state = self._rssm().initial(3)
        assert state.deter.shape == (3, 8)
        assert state.stoch.shape == (3, 4)
        assert float(state.deter.abs().max().item()) == 0.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"stoch_size": 0}, "stoch_size"),
            ({"deter_size": 0}, "deter_size"),
            ({"action_dim": 0}, "action_dim"),
            ({"min_std": 0.0}, "min_std"),
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs: dict[str, object], match: str) -> None:
        base: dict[str, object] = {
            "stoch_size": 4,
            "deter_size": 8,
            "hidden_size": 8,
            "action_dim": 2,
            "embed_size": 6,
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            RSSM(**base)  # type: ignore[arg-type]

    def test_observe_rejects_mismatched_sequence_lengths(self) -> None:
        with pytest.raises(ValueError, match="agree on T"):
            self._rssm().observe(lucid.randn((2, 5, 6)), lucid.randn((2, 3, 2)))

    def test_observe_rejects_non_sequence_input(self) -> None:
        with pytest.raises(ValueError, match=r"\(B, T"):
            self._rssm().observe(lucid.randn((2, 6)), lucid.randn((2, 2)))
