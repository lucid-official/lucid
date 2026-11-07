"""Drifting Models — the identities the method rests on, not just the shapes.

The loss is the method, so much of what is pinned here is the loss: its
values against the official release on fixed inputs, the normalisation
identity that makes one temperature's loss average exactly one, and the
stop-gradient that keeps the field out of the backward pass.  The rest pins
the wiring around it — which positives, negatives and guidance scale reach
the loss, what ``generate`` does with each argument, the sample queues, the
encoder's place outside the checkpoint — and what a later port depends on:
parameter counts, the release's feature-set keys, and the factory configs.
Fixed-input values of the generator and the encoders are in
``test_models_drifting_reference.py``.
"""

import copy
import math
import pathlib
import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
import lucid.weights
from lucid.models import DiffusionModelOutput, create_model, list_models
from lucid.models._utils._generative import start_noise
from lucid.models.generative.drifting import (
    DriftingConfig,
    DriftingForImageGeneration,
    DriftingModel,
)
from lucid.models.generative.drifting._feature_encoder import _DriftingFeatureEncoder
from lucid.models.generative.drifting._model import (
    _drift_loss,
    _DriftingSampleQueue,
    _pairwise_distance,
    _self_mask,
)
from lucid.models.generative.drifting._pretrained import (
    _CFG_BASE_2,
    _CFG_BASE_16,
    _CFG_LARGE_2,
    _CFG_LARGE_16,
)
from lucid.nn._shadow import shadow_alloc
from lucid.test.unit.models.test_models_drifting_reference import fill_parameters


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


def _with_encoder(**overrides: object) -> DriftingConfig:
    """``_tiny`` in a learned feature space: an MAE small enough to build per test."""
    return _tiny(use_mae=True, mae_base_channels=16, mae_layers=(1, 1, 1, 1), **overrides)


def _load_encoder(model: DriftingForImageGeneration) -> _DriftingFeatureEncoder:
    """``load_feature_encoder`` where the warning about its random weights is not the point."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return model.load_feature_encoder()


def _generator_output(model: DriftingForImageGeneration) -> lucid.Tensor:
    """The generator on fixed inputs, so two models are compared bit for bit."""
    generator = model.drifting.eval()
    device = next(generator.parameters()).device.type
    x = _grid((2, 4, 8, 8), 0.19, 0.5).to(device)
    labels = lucid.tensor([1, 2], device=device)
    styles = lucid.tensor([[0, 1], [2, 3]], device=device)
    return generator(x, labels, 1.5, styles).sample


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
        # The target is stop-gradiented (Eq. 6): positives receive nothing.  With x the
        # generated features in coordinates divided by s = scale / sqrt(S), each row's
        # gradient is exactly 2 (x - goal) / (s C S), and x - goal is what its loss averages
        # the square of.  Differentiating through the goal would break that identity.
        gen, pos, _, _ = self._inputs()
        gen.requires_grad_(True)
        pos.requires_grad_(True)
        loss, info = _drift_loss(gen, pos, None, None, self.TEMPS)
        loss.sum().backward()
        assert gen.grad is not None and pos.grad is None
        c, width = int(gen.shape[1]), int(gen.shape[2])
        s = float(info["scale"].item()) / math.sqrt(width)
        residual = gen.grad * (s * c * width / 2.0)
        assert ((residual * residual).mean(dim=(1, 2))).tolist() == pytest.approx(
            loss.tolist(), rel=1e-5
        )


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
            (_CFG_BASE_2, 132_708_880),
            (_CFG_LARGE_2, 462_909_456),
            (_CFG_BASE_16, 133_864_704),
            (_CFG_LARGE_16, 464_450_304),
        ],
        ids=["base_2", "large_2", "base_16", "large_16"],
    )
    def test_parameter_counts_round_to_the_paper_s(self, cfg: DriftingConfig, count: int) -> None:
        # Tables 5 and 6 give 133M, 463M, 134M and 464M.
        with shadow_alloc():
            model = DriftingModel(cfg)
            assert sum(p.numel() for p in model.parameters()) == count

    def test_swiglu_width_follows_the_release(self) -> None:
        assert _CFG_BASE_2.swiglu_hidden_size == 2048
        assert _CFG_LARGE_2.swiglu_hidden_size == 2752

    def test_the_position_table_starts_as_the_sine_cosine_grid(self) -> None:
        # The release's ``get_2d_sincos_pos_embed``, written out: half the width for the
        # column, half for the row, sines then cosines, over a ladder of ``width / 4`` steps.
        grid, width = 4, 32
        quarter = width // 4
        omega = [1.0 / 10000 ** (k / quarter) for k in range(quarter)]
        want: list[float] = []
        for row in range(grid):
            for column in range(grid):
                for position in (column, row):
                    want += [math.sin(position * w) for w in omega]
                    want += [math.cos(position * w) for w in omega]
        got = DriftingModel(_tiny()).pos_embed.reshape(-1).tolist()
        assert got == pytest.approx(want, rel=None, abs=1e-6)

    @pytest.mark.parametrize("columns", [1, 3])
    def test_style_indices_need_one_column_per_codebook(self, columns: int) -> None:
        # ``_tiny`` has two codebooks; extra columns used to be dropped without a word.
        model = DriftingModel(_tiny()).eval()
        styles = lucid.zeros((2, columns), dtype=lucid.int64)
        with pytest.raises(ValueError, match=r"style_indices must be \(B, 2\)"):
            model(lucid.randn(2, 4, 8, 8), lucid.tensor([0, 1]), style_indices=styles)

    def test_a_model_with_no_codebook_reads_no_styles(self) -> None:
        model = DriftingModel(_tiny(noise_classes=0)).eval()
        styles = lucid.zeros((2, 7), dtype=lucid.int64)
        assert (
            model(lucid.randn(2, 4, 8, 8), lucid.tensor([0, 1]), style_indices=styles).sample.ndim
            == 4
        )


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
        assert out.guidance_scale.tolist() == [1.0, 1.0]

    @pytest.mark.parametrize("power", [5.0, 3.0, 1.0, 0.0])
    def test_sampled_guidance_follows_the_power_law(self, power: float) -> None:
        # Measured against the inverse CDF the release samples: within 0.016 at these points
        # for this seed.  The floor of 2 keeps a reset-to-one scale out of it.
        lo, hi = 2.0, 4.0
        model = DriftingForImageGeneration(_tiny(cfg_min=lo, cfg_max=hi, neg_cfg_pw=power))
        lucid.manual_seed(0)
        alpha = model._sample_guidance(4096, "cpu", lucid.float32)
        assert float(alpha.min().item()) >= lo - 1e-6 and float(alpha.max().item()) <= hi + 1e-6

        def cdf(a: float) -> float:
            if power == 1.0:
                return math.log(a / lo) / math.log(hi / lo)
            e = 1.0 - power
            return (a**e - lo**e) / (hi**e - lo**e)

        for point in (2.25, 2.5, 3.0, 3.5):
            drawn = float((alpha < point).float().mean().item())
            assert drawn == pytest.approx(cdf(point), abs=0.03)

    @pytest.mark.parametrize(
        ("fraction", "low", "high"), [(0.0, 0.0, 0.0), (0.5, 0.45, 0.55), (1.0, 1.0, 1.0)]
    )
    def test_no_cfg_frac_resets_that_share_to_no_guidance(
        self, fraction: float, low: float, high: float
    ) -> None:
        model = DriftingForImageGeneration(_tiny(cfg_min=2.0, cfg_max=4.0, no_cfg_frac=fraction))
        lucid.manual_seed(0)
        alpha = model._sample_guidance(2048, "cpu", lucid.float32)
        share = float((alpha == 1.0).float().mean().item())
        assert low <= share <= high

    def test_learned_feature_space_needs_an_encoder(self) -> None:
        model = DriftingForImageGeneration(_tiny(use_mae=True))
        with pytest.raises(RuntimeError, match="load_feature_encoder"):
            model(lucid.randn(2, 4, 8, 8), lucid.tensor([0, 1]))

    def test_the_loaded_encoder_is_frozen_and_trains_the_generator(self) -> None:
        cfg = _tiny(use_mae=True, mae_base_channels=16, mae_layers=(1, 1, 1, 1))
        model = DriftingForImageGeneration(cfg)
        encoder = _load_encoder(model)
        assert all(not p.requires_grad for p in encoder.parameters())
        out = model(lucid.randn(2, 2, 4, 8, 8), lucid.tensor([0, 1]))
        out.loss.backward()
        assert all(p.grad is not None for p in model.drifting.parameters())

    def test_generate_is_one_evaluation(self) -> None:
        model = DriftingForImageGeneration(_tiny()).eval()
        out = model.generate(3, labels=lucid.tensor([0, 1, 2]), guidance_scale=1.5)
        assert tuple(out.samples.shape) == (3, 4, 8, 8)

    @staticmethod
    def _trained_model(**overrides: object) -> DriftingForImageGeneration:
        # adaLN-Zero starts at zero, so an untouched generator's samples all coincide.
        model = DriftingForImageGeneration(_tiny(**overrides))
        fill_parameters(model.drifting)
        return model

    @staticmethod
    def _labelled_batch() -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
        """Positives and negatives ``(2, n, 4, 8, 8)`` no two of which coincide, and labels."""
        return (
            _grid((2, 3, 4, 8, 8), 0.17, 0.2) + 0.5,
            _grid((2, 2, 4, 8, 8), 0.29, 1.1) - 0.5,
            lucid.tensor([3, 7]),
        )

    @pytest.mark.parametrize("scales", [[2.0, 3.5], [1.0, 1.0]])
    def test_the_loss_pairs_each_label_with_its_own_batch(self, scales: list[float]) -> None:
        # Worked out here from the outputs the step reports: each label's generated samples
        # against that label's positives and negatives, at w = (alpha - 1)(g - 1) / N_unc.
        # Positives of the wrong class, dropped negatives and a zero weight all move it.
        model = self._trained_model(temperatures=(0.2, 0.05))
        positives, negatives, labels = self._labelled_batch()
        out = model(positives, labels, negatives, guidance_scale=lucid.tensor(scales))
        assert out.guidance_scale.tolist() == scales
        want, _ = _drift_loss(
            out.samples.reshape(2, 4, -1).detach(),
            positives.reshape(2, 3, -1),
            negatives.reshape(2, 2, -1),
            lucid.tensor([[(a - 1.0) * (4 - 1) / 2] * 2 for a in scales]),
            (0.2, 0.05),
        )
        assert float(out.loss.item()) == pytest.approx(float(want.mean().item()), rel=1e-5)
        assert set(out.feature_losses) == {"global"}

    def test_every_location_of_every_set_pairs_with_its_own_label(self) -> None:
        # Rows are (label, location); built one by one here rather than by the reshape the
        # forward uses, so a transposed layout would show.
        model = self._trained_model(use_mae=True, mae_base_channels=16, mae_layers=(1, 1, 1, 1))
        encoder = _load_encoder(model)
        positives, negatives, labels = self._labelled_batch()
        out = model(positives, labels, negatives, guidance_scale=lucid.tensor([2.0, 3.5]))
        real = lucid.cat([positives, negatives], dim=1).reshape(10, 4, 8, 8)
        with lucid.no_grad():
            real_feats, gen_feats = encoder(real), encoder(out.samples)
        assert list(out.feature_losses) == list(gen_feats)
        for key in ("norm_x", "conv1", "layer1_mean_2", "layer3", "layer4_std"):
            f, width = int(gen_feats[key].shape[1]), int(gen_feats[key].shape[2])
            gen_rows, pos_rows, neg_rows, weights = [], [], [], []
            for label, boost in enumerate((1.0, 2.5)):
                for where in range(f):
                    gen_rows.append(gen_feats[key][label * 4 : (label + 1) * 4, where])
                    pos_rows.append(real_feats[key][label * 5 : label * 5 + 3, where])
                    neg_rows.append(real_feats[key][label * 5 + 3 : (label + 1) * 5, where])
                    weights.append([boost * 3 / 2] * 2)
            want, _ = _drift_loss(
                lucid.stack(gen_rows),
                lucid.stack(pos_rows),
                lucid.stack(neg_rows),
                lucid.tensor(weights),
                model.config.temperatures,
            )
            assert width > 0
            got = float(out.feature_losses[key].item())
            assert got == pytest.approx(float(want.mean().item()), rel=1e-4), key

    def test_the_generator_gets_each_label_and_scale_gen_per_label_times(self) -> None:
        model = self._trained_model()
        seen: list[tuple[list[int], list[float]]] = []

        def record(module: nn.Module, args: tuple[lucid.Tensor, ...]) -> None:
            del module
            seen.append((args[1].tolist(), args[2].tolist()))

        model.drifting.register_forward_pre_hook(record)
        positives, _, labels = self._labelled_batch()
        model(positives, labels, guidance_scale=lucid.tensor([1.5, 2.5]))
        assert seen == [([3, 3, 3, 3, 7, 7, 7, 7], [1.5] * 4 + [2.5] * 4)]

    def test_recomputed_blocks_give_the_same_loss_and_gradients(self) -> None:
        # The released pixel models train with ``use_remat``.
        positives, negatives, labels = self._labelled_batch()
        results = []
        for remat in (False, True):
            model = self._trained_model(use_remat=remat).train()
            lucid.manual_seed(2)
            out = model(positives, labels, negatives)
            out.loss.backward()
            results.append((float(out.loss.item()), model))
        (plain_loss, plain), (remat_loss, remat_model) = results
        assert remat_loss == pytest.approx(plain_loss, rel=1e-6)
        for (name, a), b in zip(plain.named_parameters(), remat_model.parameters(), strict=True):
            assert a.grad is not None and b.grad is not None, name
            assert float((a.grad - b.grad).abs().max().item()) <= 1e-4, name

    def test_autocast_trains_float32_parameters_with_every_element_moving(self) -> None:
        # What the family documents in place of casting the model to bfloat16.
        model = self._trained_model()
        before = [p.detach().clone() for p in model.parameters()]
        positives, negatives, labels = self._labelled_batch()
        opt = optim.AdamW(model.parameters(), lr=4e-4)
        with lucid.amp.autocast(dtype=lucid.bfloat16):
            loss = model(positives, labels, negatives).loss
        loss.backward()
        opt.step()
        assert all(p.dtype == lucid.float32 for p in model.parameters())
        changed = sum(
            int((p != b).sum().item()) for p, b in zip(model.parameters(), before, strict=True)
        )
        # Unused style-codebook rows and class rows see no gradient; AdamW's decay moves them.
        assert changed == sum(p.numel() for p in model.parameters())

    @pytest.mark.parametrize(
        ("images", "labels", "negatives", "message"),
        [
            ((2, 0, 4, 8, 8), (2,), None, "at least one"),
            ((0, 3, 4, 8, 8), (0,), None, "at least one"),
            ((4, 8, 8), (2,), None, "images must be"),
            ((2, 3, 4, 8, 8), (1,), None, "labels must be"),
            ((2, 3, 4, 8, 8), (3,), None, "labels must be"),
            ((2, 3, 4, 8, 8), (2,), (1, 2, 4, 8, 8), "negatives must be"),
            ((2, 3, 4, 8, 8), (2,), (2, 2, 4, 4, 4), "negatives must be"),
            ((2, 3, 4, 8, 8), (2,), (2, 4, 8, 8), "negatives must be"),
        ],
    )
    def test_a_batch_that_does_not_fit_is_refused_at_the_boundary(
        self,
        images: tuple[int, ...],
        labels: tuple[int, ...],
        negatives: tuple[int, ...] | None,
        message: str,
    ) -> None:
        model = DriftingForImageGeneration(_tiny())
        with pytest.raises(ValueError, match=message):
            model(
                lucid.randn(*images),
                lucid.zeros(labels, dtype=lucid.int64),
                None if negatives is None else lucid.randn(*negatives),
            )

    def test_no_negatives_is_a_switch_all_the_way_through(self) -> None:
        # ``neg_per_sample=0``: the unconditional queue is not read, so it need not be fed.
        model = self._trained_model(pos_per_sample=2, neg_per_sample=0)
        positive_queue, unconditional_queue = model.make_sample_queues()
        positive_queue.add(lucid.randn(4, 4, 8, 8), lucid.tensor([0, 0, 1, 1]))
        labels = lucid.tensor([0, 1])
        positives, negatives = model.draw_training_batch(
            positive_queue, unconditional_queue, labels
        )
        assert tuple(negatives.shape) == (2, 0, 4, 8, 8)
        out = model(positives, labels, negatives)
        assert out.guidance_scale.tolist() == [1.0, 1.0]
        out.loss.backward()
        assert all(p.grad is not None for p in model.parameters())


class TestGenerate:
    """Each argument of ``generate`` does what it says, shown against the generator itself."""

    STYLES = [[0, 1], [2, 3]]

    @staticmethod
    def _model() -> DriftingForImageGeneration:
        model = DriftingForImageGeneration(_tiny())
        fill_parameters(model.drifting)
        return model.eval()

    def _generator(
        self,
        model: DriftingForImageGeneration,
        noise: lucid.Tensor,
        labels: lucid.Tensor,
        guidance: float | lucid.Tensor = 1.0,
    ) -> list[float]:
        styles = lucid.tensor(self.STYLES)
        values: list[float] = model.drifting(noise, labels, guidance, styles).sample.tolist()
        return values

    def _generated(
        self,
        model: DriftingForImageGeneration,
        noise: lucid.Tensor,
        labels: lucid.Tensor,
        **kw: object,
    ) -> list[float]:
        styles = lucid.tensor(self.STYLES)
        out = model.generate(2, labels=labels, noise=noise, style_indices=styles, **kw)  # type: ignore[arg-type]
        values: list[float] = out.samples.tolist()
        return values

    def test_labels_noise_and_styles_reach_the_generator(self) -> None:
        model, noise, labels = self._model(), _grid((2, 4, 8, 8), 0.19, 0.5), lucid.tensor([1, 2])
        assert self._generated(model, noise, labels) == self._generator(model, noise, labels)
        other_labels = lucid.tensor([4, 5])
        assert self._generated(model, noise, other_labels) != self._generated(model, noise, labels)
        other_noise = _grid((2, 4, 8, 8), 0.41, 0.9)
        assert self._generated(model, other_noise, labels) != self._generated(model, noise, labels)

    @pytest.mark.parametrize("as_tensor", [False, True])
    def test_the_guidance_scale_reaches_the_generator(self, as_tensor: bool) -> None:
        model, noise, labels = self._model(), _grid((2, 4, 8, 8), 0.19, 0.5), lucid.tensor([1, 2])
        scale: float | lucid.Tensor = lucid.tensor([2.5, 2.5]) if as_tensor else 2.5
        guided = self._generated(model, noise, labels, guidance_scale=scale)
        assert guided == self._generator(model, noise, labels, scale)
        assert guided != self._generated(model, noise, labels)

    def test_temperature_scales_the_drawn_noise_only(self) -> None:
        model, labels = self._model(), lucid.tensor([1, 2])
        styles = lucid.tensor(self.STYLES)
        lucid.manual_seed(3)
        cooled = model.generate(2, labels=labels, style_indices=styles, temperature=0.5)
        lucid.manual_seed(3)
        drawn = start_noise(model, (2, 4, 8, 8), None)
        assert cooled.samples.tolist() == self._generator(model, drawn * 0.5, labels)
        lucid.manual_seed(3)
        full = model.generate(2, labels=labels, style_indices=styles)
        assert full.samples.tolist() != cooled.samples.tolist()
        # Noise the caller passes is used as given.
        noise = _grid((2, 4, 8, 8), 0.19, 0.5)
        assert self._generated(model, noise, labels, temperature=0.5) == self._generated(
            model, noise, labels
        )

    def test_what_is_not_given_is_drawn(self) -> None:
        model = self._model()
        lucid.manual_seed(4)
        first = model.generate(4)
        second = model.generate(4)
        assert tuple(first.samples.shape) == (4, 4, 8, 8)
        assert first.samples.tolist() != second.samples.tolist()
        # Drawn from the seed, so the same seed draws them again.
        lucid.manual_seed(4)
        assert model.generate(4).samples.tolist() == first.samples.tolist()

    def test_it_records_no_graph(self) -> None:
        assert not self._model().generate(2).samples.requires_grad


class TestNoWeightsArePublished:
    """Nothing is hosted, so nothing may read as if it were."""

    FACTORIES = sorted(n for n in list_models() if n.startswith("drifting_"))

    def test_there_are_eight_factories(self) -> None:
        assert len(self.FACTORIES) == 8

    @pytest.mark.parametrize("name", FACTORIES)
    def test_nothing_is_listed_and_the_flag_is_refused(self, name: str) -> None:
        assert lucid.weights.list_pretrained(name) == []
        with pytest.raises(NotImplementedError, match=name):
            create_model(name, pretrained=True)

    @pytest.mark.parametrize("flag", [True, "LATENT_640"])
    def test_the_encoder_flag_is_refused_before_anything_is_allocated(
        self, flag: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # At real size the encoder is about 8.5 GB, so a refusal comes before the first build.
        model = DriftingForImageGeneration(_with_encoder())
        built: list[object] = []
        monkeypatch.setattr(_DriftingFeatureEncoder, "build", built.append)
        with pytest.raises(NotImplementedError, match="load_feature_encoder"):
            model.load_feature_encoder(flag)  # type: ignore[arg-type]
        assert built == []

    def test_a_random_encoder_says_it_is_not_the_paper_s_objective(self) -> None:
        model = DriftingForImageGeneration(_with_encoder())
        with pytest.warns(UserWarning, match="not the paper's objective"):
            model.load_feature_encoder()

    def test_the_raw_space_has_no_encoder_to_warn_about(self) -> None:
        model = DriftingForImageGeneration(_tiny())
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            model.load_feature_encoder()


class TestEncoderDtype:
    @pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
    def test_the_encoder_is_built_in_the_generator_s_dtype(self, dtype: lucid.dtype) -> None:
        # Loaded after the cast it stayed float32 and the first call raised DtypeMismatch.
        model = DriftingForImageGeneration(_with_encoder()).to(dtype)
        encoder = _load_encoder(model)
        assert next(encoder.parameters()).dtype == dtype
        device = next(model.parameters()).device.type
        x = lucid.randn(2, 2, 4, 8, 8, dtype=dtype, device=device)
        assert bool(model(x, lucid.tensor([0, 1], device=device)).loss.isfinite().item())


class TestFeatureEncoder:
    def test_feature_sets_match_the_release(self) -> None:
        # The release's MAEResNetJAX.get_activations on a 32x32 input yields
        # these 85 sets, in this order; the encoder adds "global" first.
        cfg = DriftingConfig(sample_size=32, mae_base_channels=16)
        feats = _DriftingFeatureEncoder(cfg)(lucid.randn(1, 4, 32, 32))
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
        feats = _DriftingFeatureEncoder(cfg)(lucid.randn(1, 3, 64, 64))
        assert tuple(feats["norm_x"].shape) == (1, 1, 192)
        assert tuple(feats["conv1"].shape) == (1, 64, 16)


class TestSampleQueue:
    def test_ring_buffer_keeps_the_newest(self) -> None:
        queue = _DriftingSampleQueue(num_classes=1, max_size=2)
        queue.add(lucid.tensor([[1.0], [2.0], [3.0]]), lucid.tensor([0, 0, 0]))
        drawn = queue.sample(lucid.tensor([0]), 2)
        assert sorted(drawn.reshape(-1).tolist()) == [2.0, 3.0]

    def test_empty_class_yields_zeros(self) -> None:
        queue = _DriftingSampleQueue(num_classes=2, max_size=4)
        queue.add(lucid.ones(1, 3), lucid.tensor([0]))
        assert float(queue.sample(lucid.tensor([1]), 2).abs().max().item()) == 0.0

    def test_the_model_queues_feed_a_training_batch(self) -> None:
        cfg = _tiny(pos_per_sample=3, neg_per_sample=2, positive_bank_size=4, negative_bank_size=5)
        model = DriftingForImageGeneration(cfg)
        positive_queue, unconditional_queue = model.make_sample_queues()
        assert (positive_queue.num_classes, positive_queue.max_size) == (10, 4)
        assert (unconditional_queue.num_classes, unconditional_queue.max_size) == (1, 5)

        # Sample ``i`` is filled with ``i``, so a drawn sample names its source.
        # float16 is not the default dtype and exists on Metal as well as the
        # CPU (float64 does not), and it holds these ids exactly.
        ids = lucid.arange(7, dtype=lucid.float16).reshape(7, 1, 1, 1)
        samples = ids * lucid.ones(7, 4, 8, 8, dtype=lucid.float16)
        # Class 2 receives 0, 2, 4, 5, 6 and keeps the newest four; class 5 holds 1, 3.
        positive_queue.add(samples, lucid.tensor([2, 5, 2, 5, 2, 2, 2]))
        unconditional_queue.add(samples[:3] + 100.0, lucid.zeros(3, dtype=lucid.int64))

        positives, negatives = model.draw_training_batch(
            positive_queue, unconditional_queue, lucid.tensor([5, 2])
        )
        assert tuple(positives.shape) == (2, 3, 4, 8, 8)
        assert tuple(negatives.shape) == (2, 2, 4, 8, 8)
        assert positives.dtype == negatives.dtype == lucid.float16
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

    def test_an_entry_does_not_pin_its_batch(self) -> None:
        # A view of the batch kept its whole storage alive: 4.2 MB of live entries held
        # 1 GB, on the CPU and on Metal.  The release's bank copies too.
        queue = _DriftingSampleQueue(num_classes=2, max_size=4)
        batch = lucid.randn(16, 3, 8, 8)
        queue.add(batch, lucid.zeros(16, dtype=lucid.int64))
        entry = queue._items[0][0]
        assert entry.untyped_storage().nbytes() == entry.nbytes == 3 * 8 * 8 * 4

    def test_an_entry_does_not_see_later_writes_to_its_batch(self) -> None:
        queue = _DriftingSampleQueue(num_classes=1, max_size=4)
        batch = lucid.ones(2, 3)
        queue.add(batch, lucid.zeros(2, dtype=lucid.int64))
        batch.zero_()
        assert queue.sample(lucid.zeros(1, dtype=lucid.int64), 2).tolist() == [[[1.0] * 3] * 2]

    @pytest.mark.parametrize("label", [-1, 3])
    def test_a_label_outside_the_queues_changes_nothing(self, label: int) -> None:
        # -1 used to land in the last class without a word, after the ones before it.
        queue = _DriftingSampleQueue(num_classes=3, max_size=4)
        with pytest.raises(IndexError, match="out of range"):
            queue.add(lucid.ones(2, 3), lucid.tensor([0, label]))
        assert [len(q) for q in queue._items] == [0, 0, 0] and queue._like is None
        queue.add(lucid.ones(1, 3), lucid.tensor([0]))
        with pytest.raises(IndexError, match="out of range"):
            queue.sample(lucid.tensor([label]), 1)

    @pytest.mark.parametrize("labels", [[0], [0, 1, 2]])
    def test_a_length_mismatch_changes_nothing(self, labels: list[int]) -> None:
        queue = _DriftingSampleQueue(num_classes=3, max_size=4)
        with pytest.raises(ValueError, match="one label per sample"):
            queue.add(lucid.ones(2, 3), lucid.tensor(labels))
        assert [len(q) for q in queue._items] == [0, 0, 0] and queue._like is None

    def test_a_sample_of_another_shape_changes_nothing(self) -> None:
        queue = _DriftingSampleQueue(num_classes=1, max_size=4)
        queue.add(lucid.ones(2, 3), lucid.zeros(2, dtype=lucid.int64))
        with pytest.raises(ValueError, match="holds samples of shape"):
            queue.add(lucid.ones(2, 5), lucid.zeros(2, dtype=lucid.int64))
        assert len(queue._items[0]) == 2

    def test_labels_must_be_integers(self) -> None:
        with pytest.raises(TypeError, match="integer"):
            _DriftingSampleQueue(num_classes=1, max_size=4).add(
                lucid.ones(1, 3), lucid.tensor([0.0])
            )

    def test_an_empty_batch_adds_nothing(self) -> None:
        queue = _DriftingSampleQueue(num_classes=1, max_size=4)
        queue.add(lucid.ones(0, 3), lucid.zeros(0, dtype=lucid.int64))
        assert queue._like is None

    @pytest.mark.parametrize(("labels", "count"), [([0], 0), ([0], -1), ([], 2)])
    def test_an_empty_draw_is_refused(self, labels: list[int], count: int) -> None:
        queue = _DriftingSampleQueue(num_classes=1, max_size=4)
        queue.add(lucid.ones(1, 3), lucid.zeros(1, dtype=lucid.int64))
        with pytest.raises(ValueError, match="at least one label"):
            queue.sample(lucid.tensor(labels, dtype=lucid.int64), count)


class TestConfig:
    def test_released_configs(self) -> None:
        assert (_CFG_LARGE_2.no_cfg_frac, _CFG_LARGE_2.neg_cfg_pw) == (0.5, 3.0)
        assert (_CFG_BASE_16.use_convnext, _CFG_BASE_16.mae_input_patch_size) == (True, 8)
        assert _CFG_BASE_16.num_patches == _CFG_BASE_2.num_patches == 256

    # Copied by hand from the release's ``configs/gen/*.yaml``.
    @pytest.mark.parametrize(
        ("factory", "fields"),
        [
            (
                "drifting_base_2",
                dict(
                    sample_size=32,
                    in_channels=4,
                    patch_size=2,
                    hidden_size=768,
                    cond_dim=768,
                    depth=12,
                    num_heads=12,
                    attn_fp32=False,
                    use_remat=False,
                    gen_per_label=64,
                    cfg_min=1.0,
                    cfg_max=4.0,
                    neg_cfg_pw=5.0,
                    no_cfg_frac=0.0,
                    pos_per_sample=64,
                    neg_per_sample=32,
                    positive_bank_size=128,
                    negative_bank_size=1000,
                    use_mae=True,
                    use_convnext=False,
                    mae_base_channels=640,
                    mae_input_patch_size=1,
                ),
            ),
            (
                "drifting_large_2",
                dict(
                    sample_size=32,
                    in_channels=4,
                    patch_size=2,
                    hidden_size=1024,
                    cond_dim=1024,
                    depth=24,
                    num_heads=16,
                    attn_fp32=False,
                    use_remat=False,
                    gen_per_label=64,
                    cfg_min=1.0,
                    cfg_max=4.0,
                    neg_cfg_pw=3.0,
                    no_cfg_frac=0.5,
                    pos_per_sample=64,
                    neg_per_sample=32,
                    positive_bank_size=128,
                    negative_bank_size=1000,
                    use_mae=True,
                    use_convnext=False,
                    mae_base_channels=640,
                    mae_input_patch_size=1,
                ),
            ),
            (
                "drifting_base_16",
                dict(
                    sample_size=256,
                    in_channels=3,
                    patch_size=16,
                    hidden_size=768,
                    cond_dim=768,
                    depth=12,
                    num_heads=12,
                    attn_fp32=True,
                    use_remat=True,
                    gen_per_label=64,
                    cfg_min=1.0,
                    cfg_max=4.0,
                    neg_cfg_pw=5.0,
                    no_cfg_frac=0.0,
                    pos_per_sample=128,
                    neg_per_sample=32,
                    positive_bank_size=128,
                    negative_bank_size=1000,
                    use_mae=True,
                    use_convnext=True,
                    mae_base_channels=640,
                    mae_input_patch_size=8,
                ),
            ),
            (
                "drifting_large_16",
                dict(
                    sample_size=256,
                    in_channels=3,
                    patch_size=16,
                    hidden_size=1024,
                    cond_dim=1024,
                    depth=24,
                    num_heads=16,
                    attn_fp32=True,
                    use_remat=True,
                    gen_per_label=64,
                    cfg_min=1.0,
                    cfg_max=4.0,
                    neg_cfg_pw=5.0,
                    no_cfg_frac=0.0,
                    pos_per_sample=128,
                    neg_per_sample=32,
                    positive_bank_size=128,
                    negative_bank_size=1000,
                    use_mae=True,
                    use_convnext=True,
                    mae_base_channels=640,
                    mae_input_patch_size=8,
                ),
            ),
        ],
    )
    def test_each_factory_is_its_release_config(
        self, factory: str, fields: dict[str, object]
    ) -> None:
        for name in (factory, f"{factory}_gen"):
            with shadow_alloc():
                config = create_model(name).config
            assert {k: getattr(config, k) for k in fields} == fields, name

    def test_the_ablation_default_is_base_2_with_overrides(self) -> None:
        # Table 8 and ``configs/gen/latent_ablation.yaml``; the module docstring spells it out.
        with shadow_alloc():
            config = create_model(
                "drifting_base_2_gen",
                attn_fp32=True,
                mae_base_channels=256,
                neg_cfg_pw=3.0,
                neg_per_sample=16,
            ).config
        assert (config.hidden_size, config.depth, config.pos_per_sample) == (768, 12, 64)

    @pytest.mark.parametrize(
        "bad",
        [
            {"in_channels": 4, "out_channels": 3},
            {"temperatures": ()},
            {"cfg_min": 2.0, "cfg_max": 1.0},
            {"no_cfg_frac": 1.5},
            {"every_k_block": 0},
            {"use_convnext": True},
            {"pos_per_sample": 0},
            {"neg_per_sample": -1},
        ],
    )
    def test_rejects_invalid(self, bad: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            _tiny(**bad)


class TestEncoderIsNotPersistentState:
    """The frozen encoder is about 8.5 GB at real size and has its own entry point.

    A generator checkpoint holds the generator, whichever order the encoder
    was loaded in, on both public paths; ``from_pretrained`` builds an empty
    shell and loads strictly, so one stray key made a saved model unreadable.
    """

    @staticmethod
    def _restore(
        path: str, source: DriftingForImageGeneration, target_encoder: bool, where: pathlib.Path
    ) -> DriftingForImageGeneration:
        if path == "save_pretrained":
            source.save_pretrained(str(where))
            restored = DriftingForImageGeneration.from_pretrained(str(where))
            assert isinstance(restored, DriftingForImageGeneration)
            if target_encoder:
                _load_encoder(restored)
            return restored
        restored = DriftingForImageGeneration(source.config)
        if target_encoder:
            _load_encoder(restored)
        restored.load_state_dict(source.state_dict())
        return restored

    @pytest.mark.parametrize("path", ["save_pretrained", "state_dict"])
    @pytest.mark.parametrize(
        ("config", "encoder_before_save", "encoder_after_load"),
        [
            pytest.param(_tiny, False, False, id="raw_space"),
            pytest.param(_with_encoder, False, False, id="encoder_never_loaded"),
            pytest.param(_with_encoder, True, False, id="encoder_before_save"),
            pytest.param(_with_encoder, False, True, id="encoder_after_load"),
            pytest.param(_with_encoder, True, True, id="encoder_on_both_sides"),
        ],
    )
    def test_a_checkpoint_round_trips(
        self,
        path: str,
        config: Callable[[], DriftingConfig],
        encoder_before_save: bool,
        encoder_after_load: bool,
        tmp_path: pathlib.Path,
    ) -> None:
        source = DriftingForImageGeneration(config())
        if encoder_before_save:
            _load_encoder(source)
        keys = list(source.state_dict())
        assert keys and all(k.startswith("drifting.") for k in keys), keys
        restored = self._restore(path, source, encoder_after_load, tmp_path)
        assert list(restored.state_dict()) == keys
        device = next(source.parameters()).device.type
        same = _generator_output(source).tolist() == _generator_output(restored.to(device)).tolist()
        assert same
        # What the restored model trains in is the encoder it was given.
        built = restored.feature_encoder.built
        assert built == (config is _tiny or encoder_after_load)
        if built:
            out = restored.train()(
                lucid.randn(2, 2, 4, 8, 8, device=device), lucid.tensor([0, 1], device=device)
            )
            out.loss.backward()
            assert all(p.grad is not None for p in restored.drifting.parameters())

    def test_a_loaded_checkpoint_leaves_the_encoder_alone(self) -> None:
        source = DriftingForImageGeneration(_with_encoder())
        target = DriftingForImageGeneration(_with_encoder())
        encoder = _load_encoder(target)
        before = {k: v.clone() for k, v in encoder.state_dict().items()}
        target.load_state_dict(source.state_dict())
        assert all(bool((v == before[k]).all().item()) for k, v in encoder.state_dict().items())

    def test_the_encoder_stays_out_of_every_walk_over_the_model(self) -> None:
        model = DriftingForImageGeneration(_with_encoder())
        encoder = _load_encoder(model)
        assert not any(n.startswith("feature_encoder") for n, _ in model.named_parameters())
        assert not any(n.startswith("feature_encoder") for n, _ in model.named_modules())
        assert all(p.requires_grad for p in model.parameters())
        # It stays frozen and evaluating whatever the model is switched to.
        model.requires_grad_(True)
        model.train()
        assert not encoder.training
        assert not any(p.requires_grad for p in encoder.parameters())

    @pytest.mark.parametrize("target", ["cpu", "metal", lucid.float16])
    def test_the_encoder_follows_to(self, target: str | lucid.dtype) -> None:
        model = DriftingForImageGeneration(_with_encoder())
        encoder = _load_encoder(model)
        assert model.to(target) is model
        assert model.feature_encoder is encoder
        generator, frozen = next(model.drifting.parameters()), next(encoder.parameters())
        assert (frozen.device, frozen.dtype) == (generator.device, generator.dtype)
        assert not frozen.requires_grad
        # Moved together, the objective runs.
        x = lucid.randn(2, 2, 4, 8, 8, dtype=generator.dtype, device=generator.device.type)
        labels = lucid.tensor([0, 1], device=generator.device.type)
        assert bool(model(x, labels).loss.isfinite().item())

    def test_assigning_an_encoder_does_not_put_it_in_the_checkpoint(self) -> None:
        model = DriftingForImageGeneration(_with_encoder())
        keys = list(model.state_dict())
        other = _DriftingFeatureEncoder(model.config)
        model.feature_encoder = other
        assert model.feature_encoder is other
        assert list(model.state_dict()) == keys

    def test_a_deep_copy_takes_the_encoder_with_it(self) -> None:
        model = DriftingForImageGeneration(_with_encoder())
        encoder = _load_encoder(model)
        twin = copy.deepcopy(model)
        assert twin.feature_encoder is not encoder and twin.feature_encoder.built
        assert all(
            a is not b
            for a, b in zip(twin.feature_encoder.parameters(), encoder.parameters(), strict=True)
        )
