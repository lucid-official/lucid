"""Unit tests for VAE (first concrete generative family — Phase 5)."""

import pytest

import lucid
from lucid.models import (
    GenerationOutput,
    VAEConfig,
    VAEForImageGeneration,
    VAEModel,
    VAEOutput,
    create_model,
    is_model,
)


def _tiny_cfg(**overrides: object) -> VAEConfig:
    base = {
        "sample_size": 16,
        "in_channels": 3,
        "out_channels": 3,
        "latent_dim": 8,
        "down_block_channels": (8, 16),  # 16 / 4 = 4-px bottleneck
    }
    base.update(overrides)
    return VAEConfig(**base)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestVAEConfig:
    def test_defaults(self) -> None:
        cfg = VAEConfig()
        assert cfg.latent_dim == 128
        assert cfg.down_block_channels == (64, 128, 256)
        assert cfg.kl_weight == 1.0
        assert cfg.recon_loss == "mse"
        assert cfg.model_type == "vae"

    def test_sample_size_must_divide(self) -> None:
        # 30 isn't divisible by 2 ** 3 = 8.
        with pytest.raises(ValueError, match="divisible"):
            VAEConfig(sample_size=30, down_block_channels=(8, 16, 32))

    def test_latent_dim_positive(self) -> None:
        with pytest.raises(ValueError, match="latent_dim"):
            VAEConfig(latent_dim=0)

    def test_empty_down_block_channels(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            VAEConfig(down_block_channels=())

    def test_negative_kl_weight(self) -> None:
        with pytest.raises(ValueError, match="kl_weight"):
            VAEConfig(kl_weight=-0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Forward shapes
# ─────────────────────────────────────────────────────────────────────────────


class TestVAEModel:
    def test_encode_decode_shapes(self) -> None:
        cfg = _tiny_cfg()
        m = VAEModel(cfg).eval()
        x = lucid.randn((2, cfg.in_channels, 16, 16))
        mu, logvar = m.encode(x)
        assert tuple(mu.shape) == (2, cfg.latent_dim)
        assert tuple(logvar.shape) == (2, cfg.latent_dim)
        recon = m.decode(mu)
        assert tuple(recon.shape) == (2, cfg.out_channels, 16, 16)

    def test_full_forward(self) -> None:
        cfg = _tiny_cfg()
        m = VAEModel(cfg).eval()
        x = lucid.randn((1, 3, 16, 16))
        out = m(x)
        assert isinstance(out, VAEOutput)
        assert tuple(out.sample.shape) == (1, 3, 16, 16)
        assert tuple(out.latent.shape) == (1, cfg.latent_dim)


class TestVAEForImageGeneration:
    def test_mse_loss_components(self) -> None:
        cfg = _tiny_cfg(recon_loss="mse")
        m = VAEForImageGeneration(cfg).eval()
        x = lucid.randn((2, 3, 16, 16))
        out = m(x)
        assert out.loss is not None
        assert out.recon_loss is not None
        assert out.kl_loss is not None
        # ELBO = recon + β · KL with β = 1 by default.
        diff = float((out.loss - out.recon_loss - cfg.kl_weight * out.kl_loss).item())
        assert abs(diff) < 1e-3, f"loss decomposition mismatch: {diff}"

    def test_bce_loss_branch(self) -> None:
        cfg = _tiny_cfg(recon_loss="bce")
        m = VAEForImageGeneration(cfg).eval()
        # BCE expects targets in [0, 1] — use uniform random.
        x = lucid.rand((1, 3, 16, 16))
        out = m(x)
        assert out.loss is not None

    def test_beta_vae_weight(self) -> None:
        cfg_a = _tiny_cfg(kl_weight=1.0)
        cfg_b = _tiny_cfg(kl_weight=4.0)
        m_a = VAEForImageGeneration(cfg_a).eval()
        m_b = VAEForImageGeneration(cfg_b).eval()
        # Seed RNG so reparameterise noise is comparable.
        lucid.manual_seed(0)
        x = lucid.randn((1, 3, 16, 16))
        lucid.manual_seed(1)
        out_a = m_a(x)
        lucid.manual_seed(1)
        out_b = m_b(x)
        # Higher β → loss is more KL-dominated.  Verify the *decomposition*
        # contract holds for both (architecture is initialised fresh per
        # model so absolute values diverge; we only check the formula).
        diff_a = float((out_a.loss - out_a.recon_loss - cfg_a.kl_weight * out_a.kl_loss).item())
        diff_b = float((out_b.loss - out_b.recon_loss - cfg_b.kl_weight * out_b.kl_loss).item())
        assert abs(diff_a) < 1e-3 and abs(diff_b) < 1e-3

    def test_generate_shape(self) -> None:
        cfg = _tiny_cfg()
        m = VAEForImageGeneration(cfg).eval()
        out = m.generate(n_samples=3)
        assert isinstance(out, GenerationOutput)
        assert tuple(out.samples.shape) == (3, cfg.out_channels, 16, 16)

    def test_generate_bce_clipped_to_0_1(self) -> None:
        cfg = _tiny_cfg(recon_loss="bce")
        m = VAEForImageGeneration(cfg).eval()
        out = m.generate(n_samples=2)
        # After sigmoid the samples must live in (0, 1).
        s = out.samples
        mn = float(s.min().item())
        mx = float(s.max().item())
        assert 0.0 <= mn and mx <= 1.0, f"BCE generate not in [0,1]: [{mn}, {mx}]"


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestVAERegistry:
    @pytest.mark.parametrize(
        "name",
        ["vae", "hvae", "vae_gen", "hvae_gen"],
    )
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    def test_factory_with_override(self) -> None:
        m = create_model(
            "vae",
            sample_size=16,
            in_channels=3,
            latent_dim=8,
            down_block_channels=(8, 16),
        )
        assert isinstance(m, VAEModel)
        out = m.eval()(lucid.randn((1, 3, 16, 16)))
        assert tuple(out.sample.shape) == (1, 3, 16, 16)

    def test_auto_image_generation_dispatch(self) -> None:
        """Generative factories must be visible through AutoModelForImageGeneration."""
        from lucid.models import AutoModelForImageGeneration
        from lucid.models._registry import _registry_lookup

        entry = _registry_lookup("vae_gen", task=AutoModelForImageGeneration._task)
        assert entry.model_class is VAEForImageGeneration


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical VAE — Sønderby et al., 2016 mode (latent_dim=tuple)
# ─────────────────────────────────────────────────────────────────────────────


def _hier_cfg(**overrides: object) -> VAEConfig:
    base = {
        "sample_size": 16,
        "in_channels": 3,
        "out_channels": 3,
        "latent_dim": (4, 8, 16),
        "down_block_channels": (8, 16, 32),  # 16 / 8 = 2-px bottleneck
    }
    base.update(overrides)
    return VAEConfig(**base)  # type: ignore[arg-type]


class TestHierarchicalVAEConfig:
    def test_is_hierarchical_property(self) -> None:
        cfg = _hier_cfg()
        assert cfg.is_hierarchical is True
        assert cfg.latent_dims == (4, 8, 16)
        assert cfg.total_latent_dim == 28

    def test_vanilla_mode_property(self) -> None:
        cfg = VAEConfig(latent_dim=64)
        assert cfg.is_hierarchical is False
        assert cfg.latent_dims == (64,)
        assert cfg.total_latent_dim == 64

    def test_length_mismatch_rejected(self) -> None:
        # Tuple of length 2 vs 3 down blocks.
        with pytest.raises(ValueError, match="length must equal"):
            VAEConfig(latent_dim=(8, 16), down_block_channels=(16, 32, 64))

    def test_zero_entry_rejected(self) -> None:
        with pytest.raises(ValueError, match="latent_dim entries"):
            VAEConfig(
                latent_dim=(8, 0, 16),
                down_block_channels=(16, 32, 64),
                sample_size=32,
            )


class TestHierarchicalVAEModel:
    def test_encode_returns_per_level_lists(self) -> None:
        cfg = _hier_cfg()
        m = VAEModel(cfg).eval()
        mus, logvars = m.encode(lucid.randn((2, 3, 16, 16)))
        assert isinstance(mus, list) and isinstance(logvars, list)
        assert len(mus) == 3 and len(logvars) == 3
        for level, dim in enumerate(cfg.latent_dims):
            assert tuple(mus[level].shape) == (2, dim)
            assert tuple(logvars[level].shape) == (2, dim)

    def test_decode_consumes_per_level_zs(self) -> None:
        cfg = _hier_cfg()
        m = VAEModel(cfg).eval()
        zs = [lucid.randn((1, d)) for d in cfg.latent_dims]
        recon = m.decode(zs)
        assert tuple(recon.shape) == (1, 3, 16, 16)

    def test_forward_concatenates_for_output_dataclass(self) -> None:
        cfg = _hier_cfg()
        m = VAEModel(cfg).eval()
        out = m(lucid.randn((2, 3, 16, 16)))
        # mu / logvar / latent are flattened across levels: 4 + 8 + 16 = 28.
        assert tuple(out.mu.shape) == (2, 28)
        assert tuple(out.logvar.shape) == (2, 28)
        assert tuple(out.latent.shape) == (2, 28)
        assert tuple(out.sample.shape) == (2, 3, 16, 16)

    def test_is_hierarchical_flag(self) -> None:
        cfg = _hier_cfg()
        m = VAEModel(cfg).eval()
        assert m.is_hierarchical is True
        m_v = VAEModel(_tiny_cfg()).eval()
        assert m_v.is_hierarchical is False


class TestHierarchicalVAEForImageGeneration:
    def test_loss_decomposition(self) -> None:
        cfg = _hier_cfg(kl_weight=1.0)
        m = VAEForImageGeneration(cfg).eval()
        x = lucid.randn((2, 3, 16, 16))
        out = m(x)
        diff = float((out.loss - out.recon_loss - cfg.kl_weight * out.kl_loss).item())
        assert abs(diff) < 1e-3

    def test_kl_sums_across_levels(self) -> None:
        """KL term in the wrapper output equals the sum of per-level KLs
        when computed manually."""
        from lucid.models._utils import gaussian_kl_divergence

        cfg = _hier_cfg()
        m = VAEForImageGeneration(cfg).eval()
        x = lucid.randn((1, 3, 16, 16))
        # Get the per-level posteriors via the bare model (deterministic
        # path — same encoder weights).
        mus, logvars = m.vae.encode(x)
        manual = sum(
            float(gaussian_kl_divergence(mu, lv, reduction="mean").item())
            for mu, lv in zip(mus, logvars)
        )
        out = m(x)
        # The wrapper resamples z, but encoder outputs are deterministic so
        # KL terms agree with the manual sum.
        assert abs(float(out.kl_loss.item()) - manual) < 1e-4

    def test_generate_uses_per_level_prior(self) -> None:
        cfg = _hier_cfg()
        m = VAEForImageGeneration(cfg).eval()
        out = m.generate(n_samples=3)
        assert tuple(out.samples.shape) == (3, cfg.out_channels, 16, 16)


class TestHierarchicalVAERegistry:
    @pytest.mark.parametrize("name", ["hvae", "hvae_gen"])
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    def test_factory_dispatches_hierarchical(self) -> None:
        m = create_model(
            "hvae",
            sample_size=16,
            in_channels=3,
            latent_dim=(4, 8, 16),
            down_block_channels=(8, 16, 32),
        )
        assert isinstance(m, VAEModel)
        assert m.is_hierarchical is True
