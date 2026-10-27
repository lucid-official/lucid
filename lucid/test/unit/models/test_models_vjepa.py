"""V-JEPA — what video adds to I-JEPA, asserted.

Three of V-JEPA's decisions are invisible unless tested.  A mask that
does not run the full depth of the clip leaves the answer in the next
frame and the loss falls for the wrong reason.  A position table whose
width is split the other way loads released weights onto positions they
never saw, and nothing raises.  And the paper's probe is not a linear
one — reporting an average-pooled linear probe against its tables would
give away 17 points on Kinetics-400 by construction.

This file also trains the family (``test_models_train_step.ELSEWHERE``),
because that suite refuses a model with frozen parameters and V-JEPA
freezes its target encoder by design.
"""

import math
import random

import pytest

import lucid
from lucid.models import AutoConfig, list_models
from lucid.models.vision.vjepa import (
    VJEPAConfig,
    VJEPAForVideoClassification,
    VJEPAModel,
    vjepa_huge_16,
    vjepa_huge_16_384,
    vjepa_huge_16_384_cls,
    vjepa_huge_16_cls,
    vjepa_large_16,
    vjepa_large_16_cls,
)
from lucid.models.vision.vjepa._model import (
    _sample_collection,
    _sincos_3d,
    _tube_indices,
)


def _tiny(**overrides: object) -> VJEPAConfig:
    """A model small enough to instantiate per test."""
    base: dict[str, object] = dict(
        image_size=32,
        patch_size=8,
        tubelet_size=2,
        num_frames=4,
        dim=24,
        depth=1,
        num_heads=2,
        predictor_dim=12,
        predictor_depth=1,
        short_range_blocks=2,
        long_range_blocks=1,
    )
    base.update(overrides)
    return VJEPAConfig(**base)  # type: ignore[arg-type]


def _clips(batch: int = 2, config: VJEPAConfig | None = None) -> lucid.Tensor:
    config = _tiny() if config is None else config
    return lucid.rand(
        batch,
        config.num_frames,
        config.in_channels,
        config.image_size,
        config.image_size,
    )


class TestPaperConfigurations:
    @pytest.mark.parametrize(
        ("factory", "dim", "depth", "size"),
        [
            ("vjepa_large_16", 1024, 24, 224),
            ("vjepa_huge_16", 1280, 32, 224),
            ("vjepa_huge_16_384", 1280, 32, 384),
        ],
    )
    def test_table_6_models(self, factory: str, dim: int, depth: int, size: int) -> None:
        config = AutoConfig.from_pretrained(factory)
        assert isinstance(config, VJEPAConfig)
        assert (config.dim, config.depth, config.image_size) == (dim, depth, size)
        assert config.num_heads == 16

    def test_every_model_carries_the_same_predictor(self) -> None:
        # Table 8: twelve blocks, 384 wide, at all three encoder sizes —
        # the predictor's job does not grow with the encoder's.
        for factory in ("vjepa_large_16", "vjepa_huge_16", "vjepa_huge_16_384"):
            config = AutoConfig.from_pretrained(factory)
            assert isinstance(config, VJEPAConfig)
            assert (config.predictor_dim, config.predictor_depth) == (384, 12)
            assert config.resolved_predictor_heads == 16

    def test_a_clip_is_sixteen_frames_of_tubelets_two_deep(self) -> None:
        config = AutoConfig.from_pretrained("vjepa_large_16")
        assert isinstance(config, VJEPAConfig)
        assert (config.num_frames, config.tubelet_size, config.sampling_rate) == (
            16,
            2,
            4,
        )
        assert config.token_grid == (8, 14, 14)
        assert config.num_tokens == 1568

    def test_the_masks_are_the_paper_s_two_collections(self) -> None:
        config = AutoConfig.from_pretrained("vjepa_huge_16")
        assert isinstance(config, VJEPAConfig)
        assert (config.short_range_blocks, config.short_range_scale) == (
            8,
            (0.15, 0.15),
        )
        assert (config.long_range_blocks, config.long_range_scale) == (2, (0.7, 0.7))
        assert config.aspect_ratio == (0.75, 1.5)

    def test_the_objective_is_l1(self) -> None:
        # Section 3.1: "an l1 regression, which we found to be more
        # stable".  Paper and released code agree here, unlike I-JEPA's.
        assert AutoConfig.from_pretrained("vjepa_large_16").objective == "l1"

    def test_no_weights_are_published_here(self) -> None:
        for factory in (
            vjepa_large_16,
            vjepa_large_16_cls,
            vjepa_huge_16,
            vjepa_huge_16_cls,
            vjepa_huge_16_384,
            vjepa_huge_16_384_cls,
        ):
            with pytest.raises(NotImplementedError):
                factory(pretrained=True)

    def test_the_factories_register_under_their_tasks(self) -> None:
        assert {"vjepa_large_16", "vjepa_huge_16"} <= set(list_models(task="base"))
        assert {"vjepa_large_16_cls"} <= set(list_models(task="image-classification"))


class TestConfigValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"image_size": 30, "patch_size": 8},
            {"num_frames": 5, "tubelet_size": 2},
            {"dim": 18, "num_heads": 4},
            {"predictor_dim": 9, "predictor_heads": 2},
            {"short_range_scale": (0.5, 0.2)},
            {"long_range_scale": (0.0, 1.0)},
            {"aspect_ratio": (1.5, 0.75)},
            {"ema": (1.0, 0.9)},
            {"ema_schedule_scale": 0.0},
        ],
    )
    def test_a_config_no_model_can_be_built_from_is_refused(
        self, overrides: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError):
            _tiny(**overrides)

    def test_a_clip_must_divide_into_whole_tubelets(self) -> None:
        with pytest.raises(ValueError, match="straddle"):
            _tiny(num_frames=5, tubelet_size=2)


class TestPositions:
    def test_the_table_has_three_axes(self) -> None:
        # Time, row and column, each with its own frequencies: two tokens
        # at the same place in different frames must differ.
        table = _sincos_3d(24, (2, 4, 4), uniform_power=True)[0]
        assert tuple(int(s) for s in table.shape) == (32, 24)
        first_frame, second_frame = table[0], table[16]
        assert float((first_frame - second_frame).abs().max().item()) > 1e-6

    def test_neighbouring_positions_differ_on_every_axis(self) -> None:
        table = _sincos_3d(24, (2, 4, 4), uniform_power=True)[0]
        for a, b in ((0, 1), (0, 4), (0, 16)):  # column, row, time
            assert float((table[a] - table[b]).abs().max().item()) > 1e-6

    def test_uniform_power_changes_the_split(self) -> None:
        # Released configs set it; the code's default is the other one,
        # and the difference is silent on loaded weights.
        even = _sincos_3d(24, (2, 4, 4), uniform_power=True)
        weighted = _sincos_3d(24, (2, 4, 4), uniform_power=False)
        assert float((even - weighted).abs().max().item()) > 1e-6

    def test_the_table_is_not_learned(self) -> None:
        model = VJEPAModel(_tiny())
        assert "encoder.pos_embed" not in dict(model.named_parameters())
        assert "pos_embed" in dict(model.encoder.named_buffers())


class TestTubeMasking:
    def test_a_block_runs_the_whole_clip_in_time(self) -> None:
        # Section 3.2.  A mask that leaves a tubelet visible in the next
        # frame asks nothing: the answer is next door.
        grid = (4, 6, 6)
        indices = _tube_indices(grid, 2, 3, random.Random(0))
        duration, rows, cols = grid
        spatial = {index % (rows * cols) for index in indices}
        assert len(spatial) == 6
        for frame in range(duration):
            for offset in spatial:
                assert frame * rows * cols + offset in indices

    def test_the_context_is_what_the_blocks_left(self) -> None:
        config = _tiny()
        context, targets = _sample_collection(
            config, 3, config.short_range_blocks, config.short_range_scale, "cpu"
        )
        for clip in range(3):
            assert not (set(context[clip].tolist()) & set(targets[clip].tolist()))

    def test_the_two_collections_hide_different_amounts(self) -> None:
        # Eight small blocks against two large ones.
        lucid.manual_seed(0)
        config = VJEPAConfig(image_size=224, patch_size=16, num_frames=16, tubelet_size=2)
        _, short = _sample_collection(
            config, 2, config.short_range_blocks, config.short_range_scale, "cpu"
        )
        _, long = _sample_collection(
            config, 2, config.long_range_blocks, config.long_range_scale, "cpu"
        )
        assert int(long.shape[1]) > int(short.shape[1])

    def test_every_index_lies_inside_the_clip(self) -> None:
        config = _tiny()
        context, targets = _sample_collection(
            config, 3, config.long_range_blocks, config.long_range_scale, "cpu"
        )
        assert int(context.max().item()) < config.num_tokens
        assert int(targets.max().item()) < config.num_tokens
        assert 0 <= int(context.min().item())

    def test_the_batch_collates(self) -> None:
        # Sizes are drawn once per batch and the lists truncated to the
        # shortest, as the released collator does.
        config = _tiny()
        context, targets = _sample_collection(
            config, 5, config.short_range_blocks, config.short_range_scale, "cpu"
        )
        assert int(context.shape[0]) == int(targets.shape[0]) == 5


class TestWhatEachNetworkSees:
    def test_the_context_encoder_reads_a_shorter_sequence(self) -> None:
        # Masking here drops tokens; it is not an attention mask.
        config = _tiny()
        model = VJEPAModel(config).eval()
        indices = lucid.tensor([[0, 5, 9]], dtype=lucid.int64)
        with lucid.no_grad():
            kept = model.encoder(_clips(1, config), indices)
            whole = model.encoder(_clips(1, config))
        assert int(kept.shape[1]) == 3
        assert int(whole.shape[1]) == config.num_tokens

    def test_the_target_encoder_reads_the_whole_clip(self) -> None:
        config = _tiny()
        model = VJEPAModel(config).eval()
        assert int(model.tokens(_clips(2, config)).shape[1]) == config.num_tokens

    def test_the_targets_are_normalised(self) -> None:
        # In the released code, not in the paper: a non-affine layer norm
        # over the feature axis before any block is taken.
        lucid.manual_seed(1)
        out = VJEPAModel(_tiny()).eval()(_clips(2))
        flat = out.short_target.reshape(-1, int(out.short_target.shape[-1]))
        assert abs(float(flat.mean().item())) < 1e-4
        assert abs(float(flat.std().item()) - 1.0) < 0.05

    def test_each_collection_has_its_own_mask_token(self) -> None:
        # The released predictor carries one per collection and picks by
        # index; they start at zero.
        model = VJEPAModel(_tiny())
        assert len(model.predictor.mask_tokens) == 2
        for token in model.predictor.mask_tokens:
            assert float(token.abs().max().item()) == 0.0
        with pytest.raises(ValueError, match="mask_index"):
            model.predictor(
                lucid.rand(1, 3, 24),
                lucid.tensor([[0, 1, 2]], dtype=lucid.int64),
                lucid.tensor([[3]], dtype=lucid.int64),
                2,
            )

    def test_the_predictor_answers_at_the_encoder_s_width(self) -> None:
        config = _tiny()
        out = VJEPAModel(config)(_clips(2, config))
        assert int(out.short_prediction.shape[-1]) == config.dim
        assert out.short_prediction.shape == out.short_target.shape
        assert out.long_prediction.shape == out.long_target.shape


class TestTheMovingAverage:
    def test_the_target_encoder_is_frozen_and_starts_as_a_copy(self) -> None:
        model = VJEPAModel(_tiny())
        assert all(not p.requires_grad for p in model.target_encoder.parameters())
        for live, averaged in zip(model.encoder.parameters(), model.target_encoder.parameters()):
            assert float((live - averaged).abs().max().item()) == 0.0

    def test_no_gradient_reaches_the_target_encoder(self) -> None:
        model = VJEPAModel(_tiny())
        model(_clips()).loss.backward()
        assert all(p.grad is None for p in model.target_encoder.parameters())
        assert all(p.grad is not None for p in model.encoder.parameters())
        assert all(p.grad is not None for p in model.predictor.parameters())

    def test_the_schedule_is_stretched_past_the_run(self) -> None:
        # The released runs give the momentum a schedule a quarter longer
        # than training, so it never reaches 1.  Reproducing it with a
        # schedule the length of the run is a different experiment.
        model = VJEPAModel(_tiny())
        assert model.momentum(0, 1000) == pytest.approx(0.998)
        assert model.momentum(1000, 1000) < 1.0
        assert model.momentum(1000, 1000) == pytest.approx(0.998 + 0.002 * 0.8)
        assert model.momentum(1250, 1000) == pytest.approx(1.0)

    def test_the_average_lands_where_the_momentum_says(self) -> None:
        model = VJEPAModel(_tiny())
        with lucid.no_grad():
            for parameter in model.encoder.parameters():
                parameter[:] = parameter + 1.0
        before = [p.detach().clone() for p in model.target_encoder.parameters()]
        model.update_target(0.5)
        for averaged, start in zip(model.target_encoder.parameters(), before):
            assert float((averaged - (start + 0.5)).abs().max().item()) < 1e-6

    def test_the_optimiser_never_sees_the_target_encoder(self) -> None:
        model = VJEPAModel(_tiny())
        trained = {id(p) for p in model.trainable_parameters()}
        assert not trained & {id(p) for p in model.target_encoder.parameters()}


class TestObjective:
    def test_l1_is_the_default_and_differs_from_the_others(self) -> None:
        model = VJEPAModel(_tiny())
        a, b = lucid.zeros(2, 3), lucid.zeros(2, 3) + 4.0
        assert float(model._discrepancy(a, b).item()) == pytest.approx(4.0)
        model.config = _tiny(objective="l2")
        assert float(model._discrepancy(a, b).item()) == pytest.approx(16.0)
        model.config = _tiny(objective="smooth_l1")
        assert float(model._discrepancy(a, b).item()) == pytest.approx(3.5)

    def test_the_loss_is_the_two_collections_averaged(self) -> None:
        lucid.manual_seed(2)
        model = VJEPAModel(_tiny())
        out = model(_clips(2))
        expected = (
            model._discrepancy(out.short_prediction, out.short_target)
            + model._discrepancy(out.long_prediction, out.long_target)
        ) / 2.0
        assert float((out.loss - expected).abs().item()) < 1e-6

    def test_the_forward_loss_is_finite_and_scalar(self) -> None:
        out = VJEPAModel(_tiny())(_clips())
        assert out.loss.ndim == 0
        assert math.isfinite(float(out.loss.item()))

    @pytest.mark.parametrize("shape", [(2, 3, 32, 32), (2, 8, 3, 32, 32), (2, 4, 3, 16, 16)])
    def test_a_clip_of_the_wrong_shape_is_refused(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            VJEPAModel(_tiny())(lucid.rand(*shape))


class TestTraining:
    def test_a_step_trains_the_encoder_and_the_predictor(self) -> None:
        lucid.manual_seed(0)
        config = _tiny()
        model = VJEPAModel(config)
        clips = _clips(4, config)
        optimizer = lucid.optim.Adam(model.trainable_parameters(), lr=1e-3)

        start = float(model(clips).loss.item())
        for step in range(30):
            optimizer.zero_grad()
            model(clips).loss.backward()
            optimizer.step()
            model.update_target(model.momentum(step, 30))
        assert float(model(clips).loss.item()) < start


class TestAttentiveProbe:
    def test_the_probe_reads_the_whole_feature_map(self) -> None:
        # Not an average: Section 4.3's probe attends over the tokens, and
        # the paper puts the gap at 17 points on Kinetics-400.
        lucid.manual_seed(3)
        config = _tiny(num_classes=10)
        model = VJEPAForVideoClassification(config).eval()
        clips = _clips(2, config)
        with lucid.no_grad():
            pooled = model.pooler(model.vjepa.tokens(clips))
            averaged = model.vjepa.encode(clips)
        assert pooled.shape == averaged.shape
        assert float((pooled - averaged).abs().max().item()) > 1e-6

    def test_the_pooler_depends_on_every_token(self) -> None:
        lucid.manual_seed(4)
        config = _tiny(num_classes=10)
        model = VJEPAForVideoClassification(config).eval()
        tokens = lucid.rand(1, config.num_tokens, config.dim)
        changed = tokens.detach().clone()
        changed[:, -1] = lucid.rand(config.dim)
        with lucid.no_grad():
            before, after = model.pooler(tokens), model.pooler(changed)
        assert float((before - after).abs().max().item()) > 1e-6

    def test_only_the_probe_learns(self) -> None:
        config = _tiny(num_classes=10)
        model = VJEPAForVideoClassification(config)
        labels = lucid.tensor([0, 3], dtype=lucid.int64)
        out = model(_clips(2, config), labels)
        assert out.loss is not None
        out.loss.backward()
        assert model.head.weight.grad is not None
        assert all(p.grad is not None for p in model.pooler.parameters())
        assert all(p.grad is None for p in model.vjepa.target_encoder.parameters())

    def test_the_head_gives_one_logit_per_class(self) -> None:
        config = _tiny(num_classes=7)
        out = VJEPAForVideoClassification(config).eval()(_clips(3, config))
        assert out.logits.shape == (3, 7)
        assert out.loss is None


class TestInitialisation:
    """The same initialisation as I-JEPA's tower, plus this family's own.

    The tubelet convolution is drawn like the image side's patch
    convolution, the probe is initialised as the one-block tower it is,
    and the mask tokens stay at zero — the one place the released code
    departs from I-JEPA.
    """

    def test_the_tower_and_the_tubelet_start_at_the_released_width(self) -> None:
        lucid.manual_seed(0)
        model = VJEPAModel(_tiny(dim=32, depth=4, num_heads=4))
        block = model.encoder.blocks[0]
        assert float(block.attn.qkv.weight.std().item()) == pytest.approx(0.02, rel=0.15)
        assert float(model.encoder.patch_embed.proj.weight.std().item()) == pytest.approx(
            0.02, rel=0.15
        )
        assert float(block.attn.qkv.bias.abs().max().item()) == 0.0

    def test_each_block_narrows_its_residual_writes_by_its_depth(self) -> None:
        lucid.manual_seed(1)
        model = VJEPAModel(_tiny(dim=32, depth=4, num_heads=4))
        for depth_index, block in enumerate(model.encoder.blocks, start=1):
            want = 0.02 / math.sqrt(2.0 * depth_index)
            assert float(block.mlp.fc2.weight.std().item()) == pytest.approx(
                want, rel=0.15
            ), depth_index

    def test_the_deepest_block_writes_the_least(self) -> None:
        """Guards the test above — without the scaling every block matches."""
        lucid.manual_seed(2)
        model = VJEPAModel(_tiny(dim=32, depth=4, num_heads=4))
        widths = [float(block.mlp.fc2.weight.std().item()) for block in model.encoder.blocks]
        assert widths == sorted(widths, reverse=True), widths
        assert widths[0] > 1.7 * widths[-1]

    def test_the_probe_is_initialised_as_a_one_block_tower(self) -> None:
        lucid.manual_seed(3)
        config = _tiny(dim=32, num_heads=4, num_classes=5)
        model = VJEPAForVideoClassification(config)
        assert float(model.pooler.proj.weight.std().item()) == pytest.approx(
            0.02 / math.sqrt(2.0), rel=0.2
        )
        assert float(model.pooler.key.weight.std().item()) == pytest.approx(0.02, rel=0.2)
        assert float(model.pooler.query_token.std().item()) == pytest.approx(0.02, rel=0.3)

    def test_the_mask_tokens_stay_at_zero(self) -> None:
        # The sweep visits linear, convolutional and norm layers only, so
        # the zeroed mask tokens keep the value the released code gives
        # them — this family's one initialisation difference from I-JEPA.
        model = VJEPAModel(_tiny(dim=32, depth=2, num_heads=4))
        for token in model.predictor.mask_tokens:
            assert float(token.abs().max().item()) == 0.0
