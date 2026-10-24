"""Unit tests for DDPM (Ho et al., 2020).

Validates the U-Net architecture (encoder/middle/decoder skip topology),
ResBlock + AttentionBlock building blocks, training loss path, and the
end-to-end sampling loop via :class:`DiffusionMixin`.

Uses a tiny config (16×16, base_channels=16, 2-stage) so the full suite
runs in under a second.
"""

import pytest

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._utils._generative import (
    diffusion_posterior,
    diffusion_posterior_constants,
    diffusion_vlb_term,
    make_beta_schedule,
)
from lucid.models import (
    DDPMConfig,
    DDPMForImageGeneration,
    DDPMModel,
    DiffusionModelOutput,
    GenerationOutput,
    create_model,
    is_model,
)
from lucid.models.generative import DDPMScheduler
from lucid.models.generative.ddpm._model import _DDPMUNet


def _tiny_cfg(**overrides: object) -> DDPMConfig:
    base = {
        "sample_size": 16,
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 16,
        "channel_mult": (1, 2),
        "num_res_blocks": 1,
        "attention_resolutions": (8,),
        "num_heads": 2,
        "resnet_groups": 8,
        "num_train_timesteps": 20,
    }
    base.update(overrides)
    return DDPMConfig(**base)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestDDPMConfig:
    def test_paper_defaults(self) -> None:
        cfg = DDPMConfig()
        # Ho 2020 CIFAR setup defaults
        assert cfg.base_channels == 128
        assert cfg.channel_mult == (1, 2, 2, 2)
        assert cfg.num_res_blocks == 2
        assert cfg.attention_resolutions == (16,)
        assert cfg.num_heads == 1
        assert cfg.dropout == 0.1
        assert cfg.resnet_groups == 32
        assert cfg.num_train_timesteps == 1000
        assert cfg.beta_schedule == "linear"
        assert cfg.learn_sigma is False
        assert cfg.model_type == "ddpm"

    def test_sample_size_must_divide(self) -> None:
        # sample_size=30 not divisible by 2^(L-1) = 2^3 = 8 (with default mult).
        with pytest.raises(ValueError, match="divisible"):
            DDPMConfig(sample_size=30)

    def test_invalid_base_channels(self) -> None:
        with pytest.raises(ValueError, match="base_channels"):
            DDPMConfig(base_channels=0)

    def test_empty_channel_mult(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            DDPMConfig(channel_mult=())

    def test_dropout_bounds(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            DDPMConfig(dropout=1.0)

    def test_groups_must_divide_channels(self) -> None:
        # 17 doesn't divide 128 cleanly
        with pytest.raises(ValueError, match="resnet_groups"):
            DDPMConfig(base_channels=128, resnet_groups=17)

    def test_learn_sigma_doubles_output(self) -> None:
        cfg = DDPMConfig(learn_sigma=True, out_channels=6)
        assert cfg.out_channels_effective == 2 * cfg.in_channels

    @pytest.mark.parametrize("learn_sigma,width", [(True, 3), (False, 6)])
    def test_an_output_width_the_variance_does_not_fix_is_refused(
        self, learn_sigma: bool, width: int
    ) -> None:
        # The width was accepted and silently replaced by the derived one,
        # so the config said one thing and the network built another
        # (LCD-631).
        with pytest.raises(ValueError, match=f"out_channels must be {9 - width}"):
            DDPMConfig(learn_sigma=learn_sigma, out_channels=width)

    def test_no_learn_sigma_keeps_output(self) -> None:
        cfg = DDPMConfig(learn_sigma=False)
        assert cfg.out_channels_effective == cfg.in_channels


# ─────────────────────────────────────────────────────────────────────────────
# U-Net architecture
# ─────────────────────────────────────────────────────────────────────────────


class TestDDPMUNet:
    def test_forward_shape(self) -> None:
        cfg = _tiny_cfg()
        unet = _DDPMUNet(cfg).eval()
        x = lucid.randn((2, 3, 16, 16))
        t = lucid.tensor([5, 15]).long()
        out = unet(x, t)
        assert tuple(out.shape) == (2, 3, 16, 16)

    def test_scalar_timestep_is_broadcast(self) -> None:
        cfg = _tiny_cfg()
        unet = _DDPMUNet(cfg).eval()
        x = lucid.randn((2, 3, 16, 16))
        # Scalar tensor (rank-0) — should expand to batch.
        t = lucid.tensor(5).long()
        out = unet(x, t)
        assert tuple(out.shape) == (2, 3, 16, 16)

    def test_learn_sigma_doubles_channels(self) -> None:
        cfg = _tiny_cfg(learn_sigma=True, out_channels=6)
        unet = _DDPMUNet(cfg).eval()
        x = lucid.randn((1, 3, 16, 16))
        out = unet(x, lucid.tensor([0]).long())
        assert tuple(out.shape) == (1, 6, 16, 16)

    def test_attention_only_at_listed_resolutions(self) -> None:
        """Counting attention blocks: only stages whose spatial size is in
        ``attention_resolutions`` should have non-Identity attention slots."""
        cfg = _tiny_cfg(attention_resolutions=(8,))  # only the 8×8 stage
        unet = _DDPMUNet(cfg).eval()
        # Encoder side: stage 0 (16×16) → no attn; stage 1 (8×8 after down) → attn
        non_id_down = sum(
            1 for blk in unet.down_attn if not isinstance(blk, lucid.nn.Identity)
        )
        # Tiny config has num_res_blocks=1 per stage.  Stage-0 spatial == 16
        # (no attn), stage-1 spatial == 8 (attn).  So expect 1.
        assert non_id_down == 1

    def test_different_sample_size(self) -> None:
        cfg = _tiny_cfg(sample_size=32)
        unet = _DDPMUNet(cfg).eval()
        x = lucid.randn((1, 3, 32, 32))
        out = unet(x, lucid.tensor([0]).long())
        assert tuple(out.shape) == (1, 3, 32, 32)


# ─────────────────────────────────────────────────────────────────────────────
# DDPMModel + ForImageGeneration
# ─────────────────────────────────────────────────────────────────────────────


class TestDDPMModel:
    def test_returns_diffusion_output(self) -> None:
        cfg = _tiny_cfg()
        m = DDPMModel(cfg).eval()
        out = m(
            lucid.randn((1, 3, 16, 16)),
            lucid.tensor([0]).long(),
        )
        assert isinstance(out, DiffusionModelOutput)
        assert tuple(out.sample.shape) == (1, 3, 16, 16)
        assert out.loss is None


class TestDDPMForImageGeneration:
    def test_loss_when_target_provided(self) -> None:
        cfg = _tiny_cfg()
        m = DDPMForImageGeneration(cfg).eval()
        x = lucid.randn((2, 3, 16, 16))
        t = lucid.tensor([5, 15]).long()
        target = lucid.randn((2, 3, 16, 16))
        out = m(x, t, target=target)
        assert out.loss is not None
        assert float(out.loss.item()) >= 0.0

    def test_no_loss_without_target(self) -> None:
        cfg = _tiny_cfg()
        m = DDPMForImageGeneration(cfg).eval()
        out = m(
            lucid.randn((1, 3, 16, 16)),
            lucid.tensor([0]).long(),
        )
        assert out.loss is None


class TestDDPMHybridObjective:
    """Improved-DDPM §4 — ``L_simple + λ·L_vlb`` for the learned variance."""

    @staticmethod
    def _fixture() -> tuple[DDPMConfig, DDPMForImageGeneration]:
        cfg = _tiny_cfg(learn_sigma=True, out_channels=6)
        return cfg, DDPMForImageGeneration(cfg)

    @staticmethod
    def _batch() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            lucid.randn((2, 3, 16, 16)),  # x_t
            lucid.tensor([4, 17]).long(),  # t
            lucid.randn((2, 3, 16, 16)),  # target noise
            lucid.randn((2, 3, 16, 16)).clip(-1, 1),  # x_0
        )

    def test_bound_is_zero_only_at_the_true_posterior(self) -> None:
        """KL(q‖q) = 0, and any other model scores strictly worse.

        This pins the sign and the normalisation of the whole bound: a
        term that were mis-derived would not bottom out exactly here.
        """
        betas = make_beta_schedule(20, "linear", beta_start=1e-4, beta_end=0.02)
        post = diffusion_posterior_constants(betas)
        x0 = lucid.randn((3, 3, 8, 8)).clip(-1, 1)
        x_t = lucid.randn((3, 3, 8, 8))
        t = lucid.tensor([5, 11, 19]).long()

        mean, logvar = diffusion_posterior(x_start=x0, x_t=x_t, t=t, posterior=post)
        logvar = logvar + lucid.zeros_like(x0)
        at_optimum = diffusion_vlb_term(
            x_start=x0,
            x_t=x_t,
            t=t,
            model_mean=mean,
            model_log_variance=logvar,
            posterior=post,
        )
        assert float(at_optimum.abs().max().item()) < 1e-6

        shifted = diffusion_vlb_term(
            x_start=x0,
            x_t=x_t,
            t=t,
            model_mean=mean + 0.5,
            model_log_variance=logvar,
            posterior=post,
        )
        assert float(shifted.min().item()) > 0.0

    def test_t_zero_uses_the_decoder_likelihood(self) -> None:
        """At ``t=0`` there is no ``x_{-1}``, so the KL is replaced by the
        discretised decoder NLL — which is *not* zero at the optimum."""
        betas = make_beta_schedule(20, "linear", beta_start=1e-4, beta_end=0.02)
        post = diffusion_posterior_constants(betas)
        x0 = lucid.randn((2, 3, 8, 8)).clip(-1, 1)
        x_t = lucid.randn((2, 3, 8, 8))
        t = lucid.zeros(2).long()

        mean, logvar = diffusion_posterior(x_start=x0, x_t=x_t, t=t, posterior=post)
        term = diffusion_vlb_term(
            x_start=x0,
            x_t=x_t,
            t=t,
            model_mean=mean,
            model_log_variance=logvar + lucid.zeros_like(x0),
            posterior=post,
        )
        assert float(term.mean().item()) > 0.1

    def test_posterior_at_t_zero_is_x_start_exactly(self) -> None:
        """``coef1 = β_0/(1-ᾱ_0) = 1`` and ``coef2 = 0`` — algebraically
        exact.  Taking ``ᾱ`` from a float32 tensor instead of accumulating
        it in double loses so much of ``1-ᾱ_0`` that this lands ~5e-4 off.
        """
        betas = make_beta_schedule(20, "linear", beta_start=1e-4, beta_end=0.02)
        post = diffusion_posterior_constants(betas)
        x0 = lucid.randn((2, 3, 8, 8))
        mean, _ = diffusion_posterior(
            x_start=x0,
            x_t=lucid.randn((2, 3, 8, 8)),
            t=lucid.zeros(2).long(),
            posterior=post,
        )
        assert float((mean - x0).abs().max().item()) == 0.0

    def test_learn_sigma_constructs_and_reports_both_terms(self) -> None:
        cfg, model = self._fixture()
        x_t, t, target, x0 = self._batch()
        out = model(x_t, t, target=target, x_start=x0)

        assert out.loss_simple is not None and out.loss_vlb is not None
        assert float(out.loss_simple.item()) > 0.0
        expected = float(
            out.loss_simple.item()
            + cfg.vlb_weight * cfg.num_train_timesteps * out.loss_vlb.item()
        )
        assert abs(float(out.loss.item()) - expected) < 1e-5

    def test_fixed_variance_reports_no_vlb_term(self) -> None:
        """With ``learn_sigma=False`` there is no variance to train, so the
        bound is not computed at all."""
        model = DDPMForImageGeneration(_tiny_cfg()).eval()
        x_t, t, target, _ = self._batch()
        out = model(x_t, t, target=target)
        assert out.loss is not None
        assert out.loss_vlb is None

    def test_sampling_runs_with_a_learned_variance(self) -> None:
        """``generate`` feeds the scheduler ``out.sample``.  With the
        variance head on, the network emits ``2*in_channels`` — handing the
        whole thing to the sampler would double the image's channels."""
        _, model = self._fixture()
        model.eval()
        out = model.generate(
            DDPMScheduler(num_train_timesteps=20),
            n_samples=1,
            num_inference_steps=4,
        )
        assert tuple(out.samples.shape) == (1, 3, 16, 16)

    def test_vlb_gradient_reaches_the_variance_head_only(self) -> None:
        """Nichol & Dhariwal §4 stop-gradient the mean inside ``L_vlb``.

        Without it the noisy variational term steers the mean too and
        training destabilises — so the first half of ``conv_out``'s
        gradient must be exactly zero, not merely small.
        """
        _, model = self._fixture()
        x_t, t, target, x0 = self._batch()
        out = model(x_t, t, target=target, x_start=x0)
        assert out.loss_vlb is not None
        out.loss_vlb.backward()

        bias = model.unet.conv_out.bias
        assert bias is not None and bias.grad is not None
        grads = [float(bias.grad[i].item()) for i in range(int(bias.shape[0]))]
        mean_half, var_half = grads[:3], grads[3:]
        assert all(g == 0.0 for g in mean_half), mean_half
        assert any(g != 0.0 for g in var_half), var_half


class TestDDPMSampling:
    def test_generate_via_diffusion_mixin(self) -> None:
        cfg = _tiny_cfg()
        m = DDPMForImageGeneration(cfg).eval()
        sched = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)
        out = m.generate(sched, n_samples=2, num_inference_steps=4)
        assert isinstance(out, GenerationOutput)
        assert tuple(out.samples.shape) == (2, 3, 16, 16)

    def test_generate_intermediates(self) -> None:
        cfg = _tiny_cfg()
        m = DDPMForImageGeneration(cfg).eval()
        sched = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)
        out = m.generate(
            sched, n_samples=1, num_inference_steps=3, return_intermediates=True
        )
        assert out.intermediates is not None
        assert len(out.intermediates) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestDDPMRegistry:
    @pytest.mark.parametrize(
        "name",
        [
            "ddpm_cifar",
            "ddpm_lsun",
            "ddpm_imagenet64",
            "ddpm_cifar_gen",
            "ddpm_lsun_gen",
            "ddpm_imagenet64_gen",
        ],
    )
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    def test_factory_with_override(self) -> None:
        m = create_model(
            "ddpm_cifar",
            sample_size=16,
            base_channels=16,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(8,),
            num_heads=2,
            resnet_groups=8,
            num_train_timesteps=20,
        )
        assert isinstance(m, DDPMModel)
        out = m.eval()(
            lucid.randn((1, 3, 16, 16)),
            lucid.tensor([0]).long(),
        )
        assert tuple(out.sample.shape) == (1, 3, 16, 16)

    def test_auto_image_generation_dispatch(self) -> None:
        from lucid.models import AutoModelForImageGeneration
        from lucid.models._registry import _registry_lookup

        entry = _registry_lookup(
            "ddpm_cifar_gen", task=AutoModelForImageGeneration._task
        )
        assert entry.model_class is DDPMForImageGeneration


# ─────────────────────────────────────────────────────────────────────────────
# Pretrained weights — official google/ddpm-* (Apache-2.0) checkpoints
# ─────────────────────────────────────────────────────────────────────────────


class TestDDPMWeights:
    """``ddpm_cifar`` / ``ddpm_lsun`` (+ their ``_gen`` wrappers) ship the
    official ``google/ddpm-*`` UNet checkpoints; ``pretrained=True`` is an
    inference-ready generator.  Network-free: enum/registry + the parity-
    critical model conventions (sin-first time embed, asymmetric downsample,
    GroupNorm eps 1e-6).
    """

    def _enum(self, name: str) -> type:
        import lucid.models.weights as weights_ns

        return getattr(weights_ns, name)

    @pytest.mark.parametrize(
        ("enum_name", "tag", "slug"),
        [
            ("DDPMCifarWeights", "CIFAR10", "ddpm-cifar10"),
            ("DDPMChurchWeights", "LSUN_CHURCH", "ddpm-church"),
        ],
    )
    def test_entry_fields(self, enum_name: str, tag: str, slug: str) -> None:
        cls = self._enum(enum_name)
        assert cls.DEFAULT is cls[tag]
        e = cls[tag].entry
        assert e.num_classes == 3
        assert len(e.sha256) == 64 and set(e.sha256) != {"0"}
        assert f"lucid-dl/{slug}" in e.url and f"/{tag}/" in e.url
        assert e.meta["license"] == "apache-2.0"

    @pytest.mark.parametrize(
        ("factory", "enum_name"),
        [
            ("ddpm_cifar", "DDPMCifarWeights"),
            ("ddpm_cifar_gen", "DDPMCifarWeights"),
            ("ddpm_lsun", "DDPMChurchWeights"),
            ("ddpm_lsun_gen", "DDPMChurchWeights"),
        ],
    )
    def test_registered_for_factories(self, factory: str, enum_name: str) -> None:
        from lucid.weights import weights_for

        resolved = weights_for(factory)
        assert resolved is not None
        assert resolved.__name__ == enum_name

    def test_model_uses_ddpm_canonical_conventions(self) -> None:
        # sin-first time embedding + eps 1e-6 GroupNorm + asymmetric downsample
        # are required for checkpoint parity.
        m = create_model(
            "ddpm_cifar",
            sample_size=16,
            base_channels=16,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(8,),
            resnet_groups=8,
        )
        assert m.unet.time_mlp.flip_sin_to_cos is False
        assert abs(m.unet.down_res[0].norm1.eps - 1e-6) < 1e-12
        # asymmetric downsample: padding-0 conv (the (0,1,0,1) pad is in forward)
        assert m.unet.down_sample[0].op.padding == (0, 0)
