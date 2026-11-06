"""Drifting Models — the identities the method rests on, not just the shapes.

The loss is the method, so most of what is pinned here is the loss: its
values against the official release on fixed inputs, the normalisation
identity that makes one temperature's loss average exactly one, and the
stop-gradient that keeps the field out of the backward pass.  The rest
pins what a ported checkpoint depends on — parameter counts equal to the
released files, the release's feature-set keys, and the factory configs.
"""

import math

import pytest

import lucid
from lucid.models import DiffusionModelOutput
from lucid.models._utils._common import apply_overrides
from lucid.models.generative.drifting import (
    DriftingBase2Weights,
    DriftingConfig,
    DriftingFeatureEncoder,
    DriftingForImageGeneration,
    DriftingMAEWeights,
    DriftingModel,
    DriftingSampleQueue,
)
from lucid.models.generative.drifting._model import _drift_loss, _pairwise_distance, _self_mask
from lucid.models.generative.drifting._pretrained import (
    _CFG_BASE_2,
    _CFG_BASE_16,
    _CFG_LARGE_2,
    _CFG_LARGE_16,
    _tag_config,
)
from lucid.nn._shadow import shadow_alloc


def _tiny(**overrides: object) -> DriftingConfig:
    """A model small enough to instantiate per test, drifting in raw space."""
    base: dict[str, object] = dict(
        sample_size=8,
        patch_size=2,
        hidden_size=32,
        cond_dim=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        noise_classes=4,
        noise_coords=2,
        n_cls_tokens=2,
        gen_per_label=4,
        use_mae=False,
    )
    base.update(overrides)
    return DriftingConfig(**base)  # type: ignore[arg-type]


def _grid(shape: tuple[int, ...], freq: float, phase: float) -> lucid.Tensor:
    """Deterministic, RNG-free inputs: ``sin(freq * i + phase)``."""
    n = math.prod(shape)
    return lucid.sin(lucid.arange(n, dtype=lucid.float32) * freq + phase).reshape(shape)


class TestDriftLoss:
    """The release's ``drift_loss``, reproduced."""

    TEMPS = (0.2, 0.05, 0.02)

    def _inputs(self) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor, lucid.Tensor]:
        gen = _grid((2, 3, 5), 0.37, 0.0)
        pos = _grid((2, 4, 5), 0.53, 1.0) + 0.25
        neg = _grid((2, 2, 5), 0.71, 2.0)
        weight = lucid.tensor([[0.5, 1.5], [2.0, 0.25]])
        return gen, pos, neg, weight

    def test_matches_the_release_with_guidance_negatives(self) -> None:
        # Reference values from ``drift_loss.py`` of the official release on
        # these exact inputs (float32).  In float64 the same inputs give
        # 0.484186572 and a scale of 1.94359325, so the release is the exact
        # value to 1e-6; float32 here lands within 1.2e-6 of it on the CPU
        # and on Metal.
        gen, pos, neg, weight = self._inputs()
        loss, info = _drift_loss(gen, pos, neg, weight, self.TEMPS)
        assert loss.tolist() == pytest.approx([15.9986839, 0.4841861], rel=1e-5)
        assert float(info["scale"].item()) == pytest.approx(1.9435929, rel=1e-5)

    def test_matches_the_release_without_negatives(self) -> None:
        gen, pos, _, _ = self._inputs()
        loss, _ = _drift_loss(gen, pos, None, None, self.TEMPS)
        assert loss.tolist() == pytest.approx([10.8916798, 3.2236001], rel=1e-5)

    @staticmethod
    def _random(seed: int) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor, lucid.Tensor]:
        lucid.manual_seed(seed)
        gen, pos, neg = lucid.randn(4, 16, 64), lucid.randn(4, 8, 64), lucid.randn(4, 6, 64)
        return gen, pos, neg, lucid.rand(4, 6) + 0.5

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_a_sample_s_distance_to_itself_is_the_floor_exactly(self, seed: int) -> None:
        # The first C_g targets are the samples themselves, so their squared
        # distance is zero and the floor makes it sqrt(1e-8).  Left to
        # ``xx + yy - 2 xy`` it is the root of whatever ulp the matmul and
        # the sum round apart by.  In float32, over these three seeds, 62 of
        # the 192 were off the floor on the CPU (up to 7.8e-3) and 22 on Metal.
        gen, pos, _, _ = self._random(seed)
        targets = lucid.cat([gen, pos], dim=1)
        dist = _pairwise_distance(gen, targets, _self_mask(gen, targets))
        own = dist[:, :, :16].diagonal(dim1=1, dim2=2).reshape(-1).tolist()
        assert own == pytest.approx([1e-4] * len(own), rel=1e-6)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_float32_normalisation_is_the_float64_one(self, seed: int) -> None:
        # The scale is one mean over every pair, self pairs included, so a
        # self distance off by ulps moves it.  Over 12 seeds the float32
        # scale sits within 1.8e-7 of float64 on the CPU and on Metal; with
        # the self block left to cancellation it was 3.5e-6 to 6.3e-6 off on
        # the CPU and 1.0e-6 to 2.6e-6 on Metal.
        inputs = self._random(seed)
        _, info = _drift_loss(*inputs, self.TEMPS)
        _, exact = _drift_loss(*(t.to("cpu").to(lucid.float64) for t in inputs), self.TEMPS)
        assert float(info["scale"].item()) == pytest.approx(float(exact["scale"].item()), rel=5e-7)

    def test_one_temperature_averages_exactly_one(self) -> None:
        # A.6: the drift is divided by its own RMS over the batch, and the
        # loss is the squared drift, so its mean is one by construction.
        # A per-row (rather than global) normalisation would break this.
        gen, pos, _, _ = self._inputs()
        loss, _ = _drift_loss(gen, pos, None, None, (0.05,))
        assert float(loss.mean().item()) == pytest.approx(1.0, abs=1e-5)

    def test_gradient_flows_through_the_sample_only(self) -> None:
        # The target is stop-gradiented (Eq. 6): positives receive nothing,
        # and the generated features receive exactly 2 (x - goal) / (C S).
        gen, pos, _, _ = self._inputs()
        gen.requires_grad_(True)
        pos.requires_grad_(True)
        loss, _ = _drift_loss(gen, pos, None, None, self.TEMPS)
        loss.sum().backward()
        assert gen.grad is not None and bool(gen.grad.isfinite().all().item())
        assert pos.grad is None or float(pos.grad.abs().max().item()) == 0.0


class TestGenerator:
    def test_forward_maps_noise_to_a_sample_of_its_shape(self) -> None:
        model = DriftingModel(_tiny()).eval()
        out = model(lucid.randn(2, 4, 8, 8), lucid.tensor([0, 1]), 1.5)
        assert isinstance(out, DiffusionModelOutput)
        assert tuple(out.sample.shape) == (2, 4, 8, 8)

    def test_adaln_zero_starts_at_zero(self) -> None:
        # Every adaLN projection and the output projection are zeroed, so a
        # freshly built generator emits exactly zero.
        model = DriftingModel(_tiny()).eval()
        out = model(lucid.randn(2, 4, 8, 8), lucid.tensor([0, 1])).sample
        assert float(out.abs().max().item()) == 0.0

    def test_style_indices_are_part_of_the_noise(self) -> None:
        model = DriftingModel(_tiny()).eval()
        with lucid.no_grad():
            for p in model.parameters():
                p.copy_(lucid.randn(*p.shape) * 0.05)
        x, y = lucid.randn(1, 4, 8, 8), lucid.tensor([3])
        a = model(x, y, style_indices=lucid.tensor([[0, 1]])).sample
        b = model(x, y, style_indices=lucid.tensor([[2, 3]])).sample
        assert float((a - b).abs().max().item()) > 0.0

    @pytest.mark.parametrize(
        ("cfg", "count"),
        [
            (_CFG_BASE_2, 132_708_880),  # == the released ablation / B/2 files
            (_CFG_LARGE_2, 462_909_456),
            (_CFG_BASE_16, 133_864_704),
            (_CFG_LARGE_16, 464_450_304),
        ],
        ids=["base_2", "large_2", "base_16", "large_16"],
    )
    def test_parameter_counts_equal_the_released_checkpoints(
        self, cfg: DriftingConfig, count: int
    ) -> None:
        with shadow_alloc():
            model = DriftingModel(cfg)
            assert sum(p.numel() for p in model.parameters()) == count

    def test_swiglu_width_follows_the_release(self) -> None:
        assert _CFG_BASE_2.swiglu_hidden_size == 2048
        assert _CFG_LARGE_2.swiglu_hidden_size == 2752


class TestObjective:
    def test_a_training_step_reaches_every_generator_parameter(self) -> None:
        model = DriftingForImageGeneration(_tiny())
        positives = lucid.randn(2, 3, 4, 8, 8)
        negatives = lucid.randn(2, 2, 4, 8, 8)
        out = model(positives, lucid.tensor([0, 1]), negatives)
        assert out.loss.ndim == 0 and bool(out.loss.isfinite().item())
        assert tuple(out.samples.shape) == (8, 4, 8, 8)
        assert set(out.feature_losses) == {"global"}
        out.loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, missing

    def test_guidance_scale_is_one_without_negatives(self) -> None:
        model = DriftingForImageGeneration(_tiny())
        out = model(lucid.randn(2, 4, 8, 8), lucid.tensor([0, 1]))
        assert out.cfg_scale.tolist() == [1.0, 1.0]

    def test_sampled_guidance_stays_in_range(self) -> None:
        model = DriftingForImageGeneration(_tiny(cfg_min=1.0, cfg_max=4.0, neg_cfg_pw=5.0))
        alpha = model._sample_cfg(512, "cpu", lucid.float32)
        assert float(alpha.min().item()) >= 1.0 - 1e-6
        assert float(alpha.max().item()) <= 4.0 + 1e-6
        # p(alpha) ~ alpha^-5 piles its mass near the lower end.
        assert float((alpha < 1.5).float().mean().item()) > 0.6

    def test_learned_feature_space_needs_an_encoder(self) -> None:
        model = DriftingForImageGeneration(_tiny(use_mae=True))
        with pytest.raises(RuntimeError, match="load_feature_encoder"):
            model(lucid.randn(2, 4, 8, 8), lucid.tensor([0, 1]))

    def test_the_loaded_encoder_is_frozen_and_trains_the_generator(self) -> None:
        cfg = _tiny(use_mae=True, mae_base_channels=16, mae_layers=(1, 1, 1, 1))
        model = DriftingForImageGeneration(cfg)
        encoder = model.load_feature_encoder(pretrained=False)
        assert all(not p.requires_grad for p in encoder.parameters())
        out = model(lucid.randn(2, 2, 4, 8, 8), lucid.tensor([0, 1]))
        out.loss.backward()
        assert all(p.grad is not None for p in model.drifting.parameters())

    def test_generate_is_one_evaluation(self) -> None:
        model = DriftingForImageGeneration(_tiny()).eval()
        out = model.generate(3, labels=lucid.tensor([0, 1, 2]), cfg_scale=1.5)
        assert tuple(out.samples.shape) == (3, 4, 8, 8)


class TestFeatureEncoder:
    def test_feature_sets_match_the_release(self) -> None:
        # The release's MAEResNetJAX.get_activations on a 32x32 input yields
        # these 85 sets, in this order; the encoder adds "global" first.
        cfg = DriftingConfig(sample_size=32, mae_base_channels=16)
        feats = DriftingFeatureEncoder(cfg)(lucid.randn(1, 4, 32, 32))
        keys = list(feats)
        assert keys[:2] == ["global", "norm_x"]
        assert len(keys) == 86
        assert keys[2:12] == [
            "conv1",
            "conv1_mean",
            "conv1_std",
            "conv1_mean_2",
            "conv1_mean_4",
            "conv1_std_2",
            "conv1_std_4",
            "layer1",
            "layer1_mean",
            "layer1_std",
        ]
        assert "layer3_blk6" in feats and "layer4_blk2" in feats
        # layer4 is 4x4: 2x2 pooling leaves four locations, 4x4 pooling one.
        assert tuple(feats["layer4_mean_2"].shape) == (1, 4, 128)
        assert tuple(feats["layer4_mean_4"].shape) == (1, 1, 128)

    def test_pixel_mae_reads_a_space_to_depth_input(self) -> None:
        cfg = DriftingConfig(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            patch_size=4,
            mae_base_channels=16,
            mae_layers=(1, 1, 1, 1),
            mae_input_patch_size=8,
        )
        feats = DriftingFeatureEncoder(cfg)(lucid.randn(1, 3, 64, 64))
        assert tuple(feats["norm_x"].shape) == (1, 1, 192)
        assert tuple(feats["conv1"].shape) == (1, 64, 16)

    @pytest.mark.parametrize(
        ("cfg", "member"),
        [
            (_CFG_BASE_2, DriftingMAEWeights.LATENT_640),
            (_CFG_LARGE_2, DriftingMAEWeights.LATENT_640),
            (
                apply_overrides(
                    _CFG_BASE_2, _tag_config(DriftingBase2Weights.IMAGENET1K_256_ABLATION, {})
                ),
                DriftingMAEWeights.LATENT_256,
            ),
            (_CFG_BASE_16, DriftingMAEWeights.PIXEL_640),
            (_CFG_LARGE_16, DriftingMAEWeights.PIXEL_640),
        ],
        ids=["base_2", "large_2", "base_2_ablation", "base_16", "large_16"],
    )
    def test_each_release_finds_the_mae_it_was_trained_against(
        self, cfg: DriftingConfig, member: DriftingMAEWeights
    ) -> None:
        # load_feature_encoder(pretrained=True) loads exactly this tag.
        with shadow_alloc():
            tag = DriftingForImageGeneration(cfg)._mae_tag()
        assert tag == member.tag
        assert DriftingMAEWeights[tag] is member

    def test_no_released_mae_matches_an_unreleased_width(self) -> None:
        model = DriftingForImageGeneration(_tiny(mae_base_channels=32))
        with pytest.raises(ValueError, match="no released MAE encoder"):
            model._mae_tag()


class TestSampleQueue:
    def test_ring_buffer_keeps_the_newest(self) -> None:
        queue = DriftingSampleQueue(num_classes=1, max_size=2)
        queue.add(lucid.tensor([[1.0], [2.0], [3.0]]), lucid.tensor([0, 0, 0]))
        drawn = queue.sample(lucid.tensor([0]), 2)
        assert sorted(drawn.reshape(-1).tolist()) == [2.0, 3.0]

    def test_empty_class_yields_zeros(self) -> None:
        queue = DriftingSampleQueue(num_classes=2, max_size=4)
        queue.add(lucid.ones(1, 3), lucid.tensor([0]))
        assert float(queue.sample(lucid.tensor([1]), 2).abs().max().item()) == 0.0

    def test_the_model_queues_feed_a_training_batch(self) -> None:
        cfg = _tiny(pos_per_sample=3, neg_per_sample=2, positive_bank_size=4, negative_bank_size=5)
        model = DriftingForImageGeneration(cfg)
        positive_queue, unconditional_queue = model.make_sample_queues()
        assert (positive_queue.num_classes, positive_queue.max_size) == (10, 4)
        assert (unconditional_queue.num_classes, unconditional_queue.max_size) == (1, 5)

        # Sample ``i`` is filled with ``i``, so a drawn sample names its source.
        ids = lucid.arange(7, dtype=lucid.float64).reshape(7, 1, 1, 1)
        samples = ids * lucid.ones(7, 4, 8, 8, dtype=lucid.float64)
        # Class 2 receives 0, 2, 4, 5, 6 and keeps the newest four; class 5 holds 1, 3.
        positive_queue.add(samples, lucid.tensor([2, 5, 2, 5, 2, 2, 2]))
        unconditional_queue.add(samples[:3] + 100.0, lucid.zeros(3, dtype=lucid.int64))

        positives, negatives = model.draw_training_batch(
            positive_queue, unconditional_queue, lucid.tensor([5, 2])
        )
        assert tuple(positives.shape) == (2, 3, 4, 8, 8)
        assert tuple(negatives.shape) == (2, 2, 4, 8, 8)
        assert positives.dtype == negatives.dtype == lucid.float64
        assert positives.device == negatives.device == samples.device
        # Every drawn sample is one queued sample, whole.
        for drawn in (positives, negatives):
            assert bool((drawn == drawn[:, :, :1, :1, :1]).all().item())
        drawn_ids = positives[:, :, 0, 0, 0].tolist()
        # Two queued for class 5 and three asked for: drawn with replacement.
        assert set(drawn_ids[0]) <= {1.0, 3.0}
        # Four queued for class 2: three distinct, never the overwritten 0.
        assert len(set(drawn_ids[1])) == 3 and set(drawn_ids[1]) <= {2.0, 4.0, 5.0, 6.0}
        for row in negatives[:, :, 0, 0, 0].tolist():
            assert len(set(row)) == 2 and set(row) <= {100.0, 101.0, 102.0}


class TestConfig:
    def test_the_ablation_tag_reconfigures_base_2(self) -> None:
        entry = DriftingBase2Weights.IMAGENET1K_256_ABLATION
        merged = _tag_config(entry, {})
        assert merged["mae_base_channels"] == 256 and merged["neg_cfg_pw"] == 3.0
        assert _tag_config(entry, {"neg_cfg_pw": 2.0})["neg_cfg_pw"] == 2.0

    def test_released_configs(self) -> None:
        assert (_CFG_LARGE_2.no_cfg_frac, _CFG_LARGE_2.neg_cfg_pw) == (0.5, 3.0)
        assert (_CFG_BASE_16.use_convnext, _CFG_BASE_16.mae_input_patch_size) == (True, 8)
        assert _CFG_BASE_16.num_patches == _CFG_BASE_2.num_patches == 256

    @pytest.mark.parametrize(
        "bad",
        [
            {"in_channels": 4, "out_channels": 3},
            {"temperatures": ()},
            {"cfg_min": 2.0, "cfg_max": 1.0},
            {"no_cfg_frac": 1.5},
            {"every_k_block": 0},
            {"use_convnext": True},
        ],
    )
    def test_rejects_invalid(self, bad: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            _tiny(**bad)
