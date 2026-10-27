"""Unit tests for NCSN (Song & Ermon, 2019).

Validates the sigma-schedule contract, the U-Net wiring (shared with
DDPM via ``_to_unet_config``), the denoising-score-matching loss, and the
annealed Langevin sampler.
"""

import math

import pytest

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models import (
    DiffusionModelOutput,
    GenerationOutput,
    NCSNConfig,
    NCSNForImageGeneration,
    NCSNModel,
    create_model,
    is_model,
)
from lucid.models._utils import make_sigma_schedule


def _tiny_cfg(**overrides: object) -> NCSNConfig:
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
        "num_noise_levels": 5,
        "sigma_max": 10.0,
        "sigma_min": 0.1,
        "langevin_steps": 2,
        "langevin_eps": 1e-4,
    }
    base.update(overrides)
    return NCSNConfig(**base)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestNCSNConfig:
    def test_paper_defaults(self) -> None:
        """The defaults must be one paper's settings, not a mix of two.

        ``sigma_max=50`` is NCSNv2 (Technique 1) and goes with ``L=232``;
        ``L=10`` is NCSN v1 and goes with ``sigma_max=1.0``.  Pairing 50 with
        10 gives a schedule ratio of 0.388, which fails Technique 2's
        criterion outright — the registered factories carry the v2 pairs.
        """
        cfg = NCSNConfig()
        assert cfg.base_channels == 128
        assert cfg.num_noise_levels == 10
        assert cfg.sigma_max == 1.0
        assert cfg.sigma_min == 0.01
        assert cfg.langevin_steps == 100
        assert cfg.langevin_eps == 2e-5
        assert cfg.model_type == "ncsn"

    def test_sigma_invariant(self) -> None:
        with pytest.raises(ValueError, match="sigma_min < sigma_max"):
            NCSNConfig(sigma_min=10.0, sigma_max=1.0)

    def test_num_noise_levels_positive(self) -> None:
        with pytest.raises(ValueError, match="num_noise_levels"):
            NCSNConfig(num_noise_levels=0)

    def test_langevin_eps_positive(self) -> None:
        with pytest.raises(ValueError, match="langevin_eps"):
            NCSNConfig(langevin_eps=0.0)

    def test_resnet_groups_must_divide(self) -> None:
        with pytest.raises(ValueError, match="resnet_groups"):
            NCSNConfig(base_channels=128, resnet_groups=17)


# ─────────────────────────────────────────────────────────────────────────────
# Sigma schedule
# ─────────────────────────────────────────────────────────────────────────────


class TestSigmaSchedule:
    def test_geometric_decreasing(self) -> None:
        sigmas = make_sigma_schedule(10, sigma_max=10.0, sigma_min=0.1)
        assert tuple(sigmas.shape) == (10,)
        for i in range(9):
            assert float(sigmas[i].item()) > float(sigmas[i + 1].item())

    def test_endpoints(self) -> None:
        sigmas = make_sigma_schedule(5, sigma_max=50.0, sigma_min=0.01)
        assert abs(float(sigmas[0].item()) - 50.0) < 1e-4
        assert abs(float(sigmas[-1].item()) - 0.01) < 1e-4

    def test_geometric_ratio(self) -> None:
        """Consecutive ratio must be constant for a geometric schedule."""
        sigmas = make_sigma_schedule(5, sigma_max=10.0, sigma_min=0.001)
        ratios = [float(sigmas[i + 1].item()) / float(sigmas[i].item()) for i in range(4)]
        for r in ratios[1:]:
            assert math.isclose(r, ratios[0], rel_tol=1e-4)

    def test_single_level(self) -> None:
        sigmas = make_sigma_schedule(1, sigma_max=5.0, sigma_min=0.1)
        assert tuple(sigmas.shape) == (1,)
        assert float(sigmas[0].item()) == 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Model forward
# ─────────────────────────────────────────────────────────────────────────────


class TestNCSNModel:
    def test_forward_shape(self) -> None:
        cfg = _tiny_cfg()
        m = NCSNModel(cfg).eval()
        x = lucid.randn((2, 3, 16, 16))
        idx = lucid.tensor([0, 4]).long()
        out = m(x, idx)
        assert isinstance(out, DiffusionModelOutput)
        assert tuple(out.sample.shape) == (2, 3, 16, 16)

    def test_sigmas_buffer_descending(self) -> None:
        cfg = _tiny_cfg()
        m = NCSNModel(cfg).eval()
        # sigmas[0] is largest (sigma_max), sigmas[-1] is smallest (sigma_min)
        assert float(m.sigmas[0].item()) > float(m.sigmas[-1].item())
        assert abs(float(m.sigmas[0].item()) - cfg.sigma_max) < 1e-5
        assert abs(float(m.sigmas[-1].item()) - cfg.sigma_min) < 1e-5

    def test_num_noise_levels_property(self) -> None:
        cfg = _tiny_cfg(num_noise_levels=7)
        m = NCSNModel(cfg).eval()
        assert m.num_noise_levels == 7


# ─────────────────────────────────────────────────────────────────────────────
# DSM loss
# ─────────────────────────────────────────────────────────────────────────────


class TestNCSNForImageGeneration:
    def test_dsm_loss_positive(self) -> None:
        cfg = _tiny_cfg()
        m = NCSNForImageGeneration(cfg).eval()
        x = lucid.randn((2, 3, 16, 16))
        out = m(x)
        assert out.loss is not None
        assert float(out.loss.item()) > 0.0

    def test_dsm_loss_target_free(self) -> None:
        """DSM doesn't need an external target — the noise z plays that
        role internally.  Just verify forward(x) suffices."""
        cfg = _tiny_cfg()
        m = NCSNForImageGeneration(cfg).eval()
        out = m(lucid.randn((1, 3, 16, 16)))
        assert out.loss is not None


# ─────────────────────────────────────────────────────────────────────────────
# Annealed Langevin sampling
# ─────────────────────────────────────────────────────────────────────────────


class TestLangevinSampling:
    def test_generate_shape(self) -> None:
        cfg = _tiny_cfg()
        m = NCSNForImageGeneration(cfg).eval()
        out = m.generate(n_samples=2)
        assert isinstance(out, GenerationOutput)
        assert tuple(out.samples.shape) == (2, 3, 16, 16)

    def test_generate_intermediates_one_per_level(self) -> None:
        cfg = _tiny_cfg(num_noise_levels=4)
        m = NCSNForImageGeneration(cfg).eval()
        out = m.generate(n_samples=1, return_intermediates=True)
        assert out.intermediates is not None
        # One intermediate per σ level.
        assert len(out.intermediates) == 4
        for inter in out.intermediates:
            assert tuple(inter.shape) == (1, 3, 16, 16)

    def test_langevin_steps_override(self) -> None:
        """Override ``langevin_steps`` to make tests faster."""
        cfg = _tiny_cfg(langevin_steps=100)
        m = NCSNForImageGeneration(cfg).eval()
        out = m.generate(n_samples=1, langevin_steps=2)
        assert tuple(out.samples.shape) == (1, 3, 16, 16)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestNCSNRegistry:
    @pytest.mark.parametrize(
        "name",
        ["ncsn_cifar", "ncsn_celeba", "ncsn_cifar_gen", "ncsn_celeba_gen"],
    )
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    def test_factory_with_override(self) -> None:
        m = create_model(
            "ncsn_cifar",
            sample_size=16,
            base_channels=16,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(8,),
            num_heads=2,
            resnet_groups=8,
            num_noise_levels=5,
            sigma_max=10.0,
        )
        assert isinstance(m, NCSNModel)
        out = m.eval()(
            lucid.randn((1, 3, 16, 16)),
            lucid.tensor([0]).long(),
        )
        assert tuple(out.sample.shape) == (1, 3, 16, 16)

    def test_auto_image_generation_dispatch(self) -> None:
        from lucid.models import AutoModelForImageGeneration
        from lucid.models._registry import _registry_lookup

        entry = _registry_lookup("ncsn_cifar_gen", task=AutoModelForImageGeneration._task)
        assert entry.model_class is NCSNForImageGeneration


class TestNCSNRefineNetBackbone:
    """The backbone v1/v2 specify, behind ``backbone="refinenet"``."""

    @staticmethod
    def _config(**over: object):
        from lucid.models.generative.ncsn._config import NCSNConfig

        base: dict[str, object] = dict(
            in_channels=3,
            base_channels=16,
            channel_mult=(1, 2, 2, 2),
            num_noise_levels=4,
            num_res_blocks=1,
            attention_resolutions=(),
            resnet_groups=8,
        )
        base.update(over)
        return NCSNConfig(**base)  # type: ignore[arg-type]

    def test_default_backbone_is_unchanged(self) -> None:
        assert self._config().backbone == "ddpm_unet"

    def test_refinenet_produces_a_score_of_the_input_shape(self) -> None:
        from lucid.models.generative.ncsn._model import NCSNModel

        lucid.manual_seed(0)
        m = NCSNModel(self._config(backbone="refinenet"))
        m.eval()
        out = m(lucid.randn(2, 3, 32, 32), lucid.tensor([0, 3]).long())
        assert tuple(out.sample.shape) == (2, 3, 32, 32)
        assert bool(out.sample.isfinite().all().item())

    def test_the_two_backbones_are_actually_different_networks(self) -> None:
        """Not merely a renamed flag: the layer kinds must not overlap."""
        from lucid.models.generative.ncsn._model import NCSNModel

        default = {type(mod).__name__ for mod in NCSNModel(self._config()).modules()}
        refine = {
            type(mod).__name__ for mod in NCSNModel(self._config(backbone="refinenet")).modules()
        }
        assert "GroupNorm" in default and "CondInstanceNormPlusPlus" not in default
        assert "CondInstanceNormPlusPlus" in refine and "GroupNorm" not in refine

    def test_conditional_gains_all_receive_gradient(self) -> None:
        """A per-noise-level gain that never trains is a dead parameter."""
        from lucid.models.generative.ncsn._model import NCSNModel

        lucid.manual_seed(0)
        m = NCSNModel(self._config(backbone="refinenet"))
        m.train()
        m(lucid.randn(2, 3, 32, 32), lucid.tensor([0, 3]).long()).sample.sum().backward()
        gains = [n for n, _ in m.named_parameters() if "gamma" in n or "alpha" in n]
        assert gains
        for name in gains:
            grad = m.get_parameter(name).grad
            assert grad is not None and float(grad.abs().sum().item()) > 0.0

    def test_instance_norm_plus_plus_keeps_the_brightness_pattern(self) -> None:
        """§3.3's whole point: plain instance norm throws this away."""
        from lucid.models.generative.ncsn._refinenet import (
            CondInstanceNormPlusPlus,
        )

        def make(means: list[float]) -> Tensor:
            return lucid.tensor([[[[m - 0.5, m + 0.5]] for m in means]])

        norm = CondInstanceNormPlusPlus(3, num_classes=1)
        label = lucid.tensor([0]).long()
        a, b = make([1.0, 2.0, 3.0]), make([1.0, 2.0, 9.0])

        def plain(x: Tensor) -> Tensor:
            mu = x.mean(dim=(2, 3), keepdim=True)
            var = ((x - mu) ** 2).mean(dim=(2, 3), keepdim=True)
            return (x - mu) / var.sqrt()

        # The two differ only in their channel-mean pattern, which plain
        # instance norm removes entirely.
        assert float((plain(a) - plain(b)).abs().max().item()) == 0.0
        assert float((norm(a, label) - norm(b, label)).abs().max().item()) > 0.1
