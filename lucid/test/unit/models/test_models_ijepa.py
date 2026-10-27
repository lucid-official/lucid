"""I-JEPA — the claims that make it self-supervised, asserted.

Every one of I-JEPA's ideas is a constraint on what a network is allowed
to see or be trained by, and each of them fails silently when broken: a
target encoder that receives gradient collapses onto a constant and the
loss *falls*; targets read from the encoder's input instead of its output
cost 11 points of ImageNet accuracy and nothing else; a context block that
keeps its targets makes the task trivial and the loss small.  A shape test
sees none of it.

This file also trains the family (``test_models_train_step.ELSEWHERE``),
because that suite asserts no parameter is frozen and I-JEPA freezes its
target encoder by design.
"""

import math
import random

import pytest

import lucid
import lucid.nn.functional as F
from lucid.models import AutoConfig, list_models
from lucid.models.vision.ijepa import (
    IJEPAConfig,
    IJEPAForImageClassification,
    IJEPAModel,
    ijepa_base_16,
    ijepa_base_16_cls,
    ijepa_huge_14,
    ijepa_huge_14_cls,
    ijepa_huge_16_448,
    ijepa_huge_16_448_cls,
    ijepa_large_16,
    ijepa_large_16_cls,
)
from lucid.models.vision.ijepa._model import _block_shape, _sample_masks


def _tiny(**overrides: object) -> IJEPAConfig:
    """A model small enough to instantiate per test."""
    base: dict[str, object] = dict(
        image_size=32,
        patch_size=8,
        dim=16,
        depth=1,
        num_heads=2,
        predictor_dim=8,
        predictor_depth=1,
        min_keep=1,
        target_scale=(0.1, 0.2),
        context_scale=(0.8, 1.0),
    )
    base.update(overrides)
    return IJEPAConfig(**base)  # type: ignore[arg-type]


def _images(batch: int = 2, config: IJEPAConfig | None = None) -> lucid.Tensor:
    config = _tiny() if config is None else config
    side = config.image_size
    return lucid.rand(batch, config.in_channels, side, side)


class TestPaperConfigurations:
    @pytest.mark.parametrize(
        ("factory", "dim", "depth", "heads", "patch", "size", "predictor_depth"),
        [
            ("ijepa_base_16", 768, 12, 12, 16, 224, 6),
            ("ijepa_large_16", 1024, 24, 16, 16, 224, 12),
            ("ijepa_huge_14", 1280, 32, 16, 14, 224, 12),
            ("ijepa_huge_16_448", 1280, 32, 16, 16, 448, 12),
        ],
    )
    def test_table_1_sizes(
        self,
        factory: str,
        dim: int,
        depth: int,
        heads: int,
        patch: int,
        size: int,
        predictor_depth: int,
    ) -> None:
        config = AutoConfig.from_pretrained(factory)
        assert isinstance(config, IJEPAConfig)
        assert (config.dim, config.depth, config.num_heads) == (dim, depth, heads)
        assert (config.patch_size, config.image_size) == (patch, size)
        assert config.predictor_depth == predictor_depth

    def test_the_predictor_stays_narrow_at_every_size(self) -> None:
        # Appendix A.1 keeps it at 384 for all of them; Table 14 measures
        # 384 against 1024 and prefers the narrow one.
        for factory in ("ijepa_base_16", "ijepa_large_16", "ijepa_huge_14"):
            assert AutoConfig.from_pretrained(factory).predictor_dim == 384

    def test_the_predictor_borrows_the_encoder_head_count(self) -> None:
        # Which gives a 384-wide predictor 24-dimensional heads at ViT-H.
        config = AutoConfig.from_pretrained("ijepa_huge_14")
        assert isinstance(config, IJEPAConfig)
        assert config.predictor_heads is None
        assert config.resolved_predictor_heads == 16
        assert config.predictor_dim % config.resolved_predictor_heads == 0

    def test_a_fourteen_pixel_patch_still_tiles_224(self) -> None:
        config = AutoConfig.from_pretrained("ijepa_huge_14")
        assert isinstance(config, IJEPAConfig)
        assert (config.grid_size, config.num_patches) == (16, 256)

    def test_there_is_no_class_token(self) -> None:
        # Appendix A.1: pretrained without one, and evaluation average-pools
        # the patches.  A positional table sized num_patches + 1 is the
        # usual way this creeps back in.
        model = IJEPAModel(_tiny())
        positions = model.encoder.pos_embed
        assert tuple(int(s) for s in positions.shape) == (
            1,
            model.config.num_patches,
            16,
        )
        assert not any("cls" in name for name, _ in model.named_parameters())

    def test_no_weights_are_published_here(self) -> None:
        for factory in (
            ijepa_base_16,
            ijepa_base_16_cls,
            ijepa_large_16,
            ijepa_large_16_cls,
            ijepa_huge_14,
            ijepa_huge_14_cls,
            ijepa_huge_16_448,
            ijepa_huge_16_448_cls,
        ):
            with pytest.raises(NotImplementedError):
                factory(pretrained=True)

    def test_the_factories_register_under_their_tasks(self) -> None:
        base = set(list_models(task="base"))
        heads = set(list_models(task="image-classification"))
        assert {"ijepa_base_16", "ijepa_large_16", "ijepa_huge_14"} <= base
        assert {"ijepa_base_16_cls", "ijepa_huge_16_448_cls"} <= heads


class TestConfigValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"image_size": 30, "patch_size": 8},
            {"dim": 18, "num_heads": 4},
            {"predictor_dim": 9, "predictor_heads": 2},
            {"target_scale": (0.5, 0.2)},
            {"context_scale": (0.0, 1.0)},
            {"target_aspect": (1.5, 0.75)},
            {"ema": (1.0, 0.9)},
            {"min_keep": 10_000},
            {"mlp_ratio": 0.0},
            {"smooth_l1_beta": 0.0},
        ],
    )
    def test_a_config_no_model_can_be_built_from_is_refused(
        self, overrides: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError):
            _tiny(**overrides)

    def test_a_json_list_range_becomes_a_tuple(self) -> None:
        config = _tiny(target_scale=[0.1, 0.2], ema=[0.99, 1.0])  # type: ignore[arg-type]
        assert config.target_scale == (0.1, 0.2)
        assert config.ema == (0.99, 1.0)


class TestMasking:
    def test_a_target_block_covers_the_paper_s_fraction(self) -> None:
        # Section 3: 0.15 to 0.2 of the image, aspect 0.75 to 1.5.
        config = IJEPAConfig(image_size=224, patch_size=16)
        grid = config.grid_size
        rng = random.Random(0)
        for _ in range(20):
            height, width = _block_shape(
                config.target_scale, config.target_aspect, grid, config.min_keep, rng
            )
            fraction = height * width / float(grid * grid)
            assert 0.10 <= fraction <= 0.26, (height, width, fraction)
            ratio = height / width
            assert 0.6 <= ratio <= 1.8, (height, width, ratio)

    def test_a_context_block_is_square_and_large(self) -> None:
        # The paper gives the context a unit aspect ratio; the released
        # code hard-codes it while leaving the config's range for targets.
        config = IJEPAConfig(image_size=224, patch_size=16)
        grid = config.grid_size
        rng = random.Random(0)
        for _ in range(20):
            height, width = _block_shape(
                config.context_scale, (1.0, 1.0), grid, config.min_keep, rng
            )
            assert height == width
            assert 0.80 <= height * width / float(grid * grid) <= 1.0

    def test_the_context_drops_every_patch_a_target_holds(self) -> None:
        config = _tiny()
        context, targets = _sample_masks(config, 4, "cpu")
        for image in range(4):
            taken = {int(i) for block in targets[image].tolist() for i in block}
            assert not (set(context[image].tolist()) & taken)

    def test_allowing_overlap_keeps_them(self) -> None:
        # The flag exists in the released config; the paper always removes.
        lucid.manual_seed(0)
        config = _tiny(allow_overlap=True, context_scale=(1.0, 1.0))
        context, targets = _sample_masks(config, 4, "cpu")
        shared = sum(
            len(set(context[i].tolist()) & {int(j) for b in targets[i].tolist() for j in b})
            for i in range(4)
        )
        assert shared > 0

    def test_there_are_four_target_blocks(self) -> None:
        config = _tiny()
        _, targets = _sample_masks(config, 3, "cpu")
        assert int(targets.shape[1]) == config.num_target_blocks == 4

    def test_every_index_lies_inside_the_grid(self) -> None:
        config = _tiny()
        context, targets = _sample_masks(config, 3, "cpu")
        assert 0 <= int(context.min().item())
        assert int(context.max().item()) < config.num_patches
        assert int(targets.max().item()) < config.num_patches

    def test_one_block_size_serves_the_whole_batch(self) -> None:
        # The released code samples the size once per iteration so every
        # image's block is the same length, and only the position varies.
        config = _tiny()
        context, targets = _sample_masks(config, 5, "cpu")
        assert int(context.shape[0]) == 5
        assert len({tuple(row) for row in targets[:, 0].tolist()}) > 1


class TestWhatEachNetworkSees:
    def test_targets_are_read_from_the_encoder_output(self) -> None:
        """Section 3 calls this distinction crucial; Table 11 prices it.

        Masking the *input* would leave a target block ignorant of the rest
        of the image.  Because the targets are taken from the output, a
        patch far from a block still moves that block's target.
        """
        lucid.manual_seed(0)
        config = _tiny()
        model = IJEPAModel(config).eval()
        images = _images(1, config)
        with lucid.no_grad():
            tokens = model.target_encoder(images)
            changed = images.detach().clone()
            changed[:, :, :8, :8] = lucid.rand(1, 3, 8, 8)
            after = model.target_encoder(changed)
        # Patch 0 is the corner that changed; patch 15 is the far corner.
        far = float((tokens[:, -1] - after[:, -1]).abs().max().item())
        assert far > 1e-6

    def test_the_target_representations_are_normalised(self) -> None:
        # Not in the paper: the released code layer-norms the target
        # encoder's output, without affine terms, before taking blocks.
        lucid.manual_seed(1)
        model = IJEPAModel(_tiny()).eval()
        out = model(_images(2))
        flat = out.target.reshape(-1, int(out.target.shape[-1]))
        assert abs(float(flat.mean().item())) < 1e-4
        assert abs(float(flat.std().item()) - 1.0) < 0.05

    def test_the_context_encoder_is_permutation_equivariant(self) -> None:
        # Positions are added before the patches are selected, so reordering
        # the kept indices reorders the outputs and changes nothing else.
        lucid.manual_seed(2)
        config = _tiny()
        model = IJEPAModel(config).eval()
        images = _images(1, config)
        indices = lucid.tensor([[0, 3, 5, 9]], dtype=lucid.int64)
        shuffled = lucid.tensor([[9, 0, 5, 3]], dtype=lucid.int64)
        with lucid.no_grad():
            straight = model.encoder(images, indices)
            mixed = model.encoder(images, shuffled)
        order = [3, 0, 2, 1]
        for got, want in enumerate(order):
            gap = float((mixed[:, got] - straight[:, want]).abs().max().item())
            assert gap < 1e-5, (got, want, gap)

    def test_the_predictor_is_told_where_to_look(self) -> None:
        # Its only information about a target is that position: the mask
        # token is shared, so two different positions must differ.
        lucid.manual_seed(3)
        config = _tiny()
        model = IJEPAModel(config).eval()
        context_index = lucid.tensor([[0, 1, 2, 3]], dtype=lucid.int64)
        with lucid.no_grad():
            context = model.encoder(_images(1, config), context_index)
            here = model.predictor(context, context_index, lucid.tensor([[8]], dtype=lucid.int64))
            there = model.predictor(context, context_index, lucid.tensor([[15]], dtype=lucid.int64))
        assert float((here - there).abs().max().item()) > 1e-6

    def test_the_predictor_answers_at_the_encoder_s_width(self) -> None:
        config = _tiny()
        out = IJEPAModel(config)(_images(2, config))
        assert int(out.prediction.shape[-1]) == config.dim
        assert out.prediction.shape == out.target.shape


class TestTheMovingAverage:
    def test_the_target_encoder_starts_as_a_copy(self) -> None:
        model = IJEPAModel(_tiny())
        for live, averaged in zip(model.encoder.parameters(), model.target_encoder.parameters()):
            assert float((live - averaged).abs().max().item()) == 0.0

    def test_the_target_encoder_is_frozen(self) -> None:
        # Structural, not a call site someone might forget — and the reason
        # this family trains here rather than in the shared step suite.
        model = IJEPAModel(_tiny())
        assert all(not p.requires_grad for p in model.target_encoder.parameters())
        assert all(p.requires_grad for p in model.encoder.parameters())
        assert len(model.trainable_parameters()) == len(list(model.encoder.parameters())) + len(
            list(model.predictor.parameters())
        )

    def test_no_gradient_reaches_the_target_encoder(self) -> None:
        # If one did, the pair would be free to agree on a constant.
        model = IJEPAModel(_tiny())
        model(_images()).loss.backward()
        assert all(p.grad is None for p in model.target_encoder.parameters())
        assert all(p.grad is not None for p in model.encoder.parameters())
        assert all(p.grad is not None for p in model.predictor.parameters())

    def test_the_average_lands_where_the_momentum_says(self) -> None:
        model = IJEPAModel(_tiny())
        with lucid.no_grad():
            for parameter in model.encoder.parameters():
                parameter[:] = parameter + 1.0
        before = [p.detach().clone() for p in model.target_encoder.parameters()]
        model.update_target(0.5)
        for averaged, start in zip(model.target_encoder.parameters(), before):
            assert float((averaged - (start + 0.5)).abs().max().item()) < 1e-6

    def test_a_momentum_of_one_freezes_the_target(self) -> None:
        model = IJEPAModel(_tiny())
        with lucid.no_grad():
            for parameter in model.encoder.parameters():
                parameter[:] = parameter + 1.0
        before = [p.detach().clone() for p in model.target_encoder.parameters()]
        model.update_target(1.0)
        for averaged, start in zip(model.target_encoder.parameters(), before):
            assert float((averaged - start).abs().max().item()) == 0.0

    def test_the_schedule_runs_from_996_to_one(self) -> None:
        # Appendix A.1: linear over training, ending at 1 so the target
        # stops moving exactly when training does.
        model = IJEPAModel(_tiny())
        assert model.momentum(0, 100) == pytest.approx(0.996)
        assert model.momentum(50, 100) == pytest.approx(0.998)
        assert model.momentum(100, 100) == pytest.approx(1.0)
        assert model.momentum(200, 100) == pytest.approx(1.0)

    @pytest.mark.parametrize("momentum", [-0.1, 1.5])
    def test_a_momentum_outside_the_unit_interval_is_refused(self, momentum: float) -> None:
        with pytest.raises(ValueError):
            IJEPAModel(_tiny()).update_target(momentum)

    def test_a_schedule_with_no_steps_is_refused(self) -> None:
        with pytest.raises(ValueError):
            IJEPAModel(_tiny()).momentum(0, 0)


class TestObjective:
    def test_the_two_losses_are_different_numbers(self) -> None:
        # The paper writes L2; the released code uses smooth L1, and the
        # default follows the code.
        model = IJEPAModel(_tiny())
        a = lucid.rand(2, 3, 4)
        b = a + 0.5
        smooth = float(model._discrepancy(a, b).item())
        model.config = _tiny(objective="l2")
        squared = float(model._discrepancy(a, b).item())
        assert smooth == pytest.approx(0.5 * 0.5 * 0.5)  # beta=1 keeps it quadratic
        assert squared == pytest.approx(0.25)
        assert smooth != squared

    def test_the_loss_is_zero_when_the_prediction_is_exact(self) -> None:
        model = IJEPAModel(_tiny())
        same = lucid.rand(2, 3, 4)
        assert float(model._discrepancy(same, same).item()) == 0.0

    def test_the_forward_loss_is_finite_and_scalar(self) -> None:
        out = IJEPAModel(_tiny())(_images())
        assert out.loss.ndim == 0
        assert math.isfinite(float(out.loss.item()))

    def test_a_wrongly_sized_image_is_refused(self) -> None:
        with pytest.raises(ValueError):
            IJEPAModel(_tiny())(lucid.rand(2, 3, 16, 16))


class TestTraining:
    def test_a_step_trains_the_encoder_and_the_predictor(self) -> None:
        lucid.manual_seed(0)
        config = _tiny()
        model = IJEPAModel(config)
        images = _images(4, config)
        optimizer = lucid.optim.Adam(model.trainable_parameters(), lr=1e-3)

        start = float(model(images).loss.item())
        for step in range(30):
            optimizer.zero_grad()
            model(images).loss.backward()
            optimizer.step()
            model.update_target(model.momentum(step, 30))
        end = float(model(images).loss.item())
        assert end < start, (start, end)

    def test_the_optimiser_never_sees_the_target_encoder(self) -> None:
        model = IJEPAModel(_tiny())
        trained = {id(p) for p in model.trainable_parameters()}
        assert not trained & {id(p) for p in model.target_encoder.parameters()}


class TestLinearProbe:
    def test_the_probe_reads_the_target_encoder(self) -> None:
        lucid.manual_seed(4)
        config = _tiny(num_classes=10)
        model = IJEPAForImageClassification(config).eval()
        images = _images(2, config)
        with lucid.no_grad():
            pooled = model.ijepa.encode(images)
            direct = model.head(pooled)
        assert float((model(images).logits - direct).abs().max().item()) < 1e-6

    def test_only_the_head_learns(self) -> None:
        # Appendix A.2's protocol is a probe on a frozen encoder: the
        # target encoder must take no gradient even through the head's loss.
        config = _tiny(num_classes=10)
        model = IJEPAForImageClassification(config)
        labels = lucid.tensor([0, 3], dtype=lucid.int64)
        out = model(_images(2, config), labels)
        assert out.loss is not None
        out.loss.backward()
        assert model.head.weight.grad is not None
        assert all(p.grad is None for p in model.ijepa.target_encoder.parameters())

    def test_the_head_gives_one_logit_per_class(self) -> None:
        config = _tiny(num_classes=7)
        out = IJEPAForImageClassification(config).eval()(_images(3, config))
        assert out.logits.shape == (3, 7)
        assert out.loss is None

    def test_labels_bring_a_cross_entropy(self) -> None:
        config = _tiny(num_classes=5)
        model = IJEPAForImageClassification(config).eval()
        images, labels = _images(2, config), lucid.tensor([1, 4], dtype=lucid.int64)
        out = model(images, labels)
        assert out.loss is not None
        expected = F.cross_entropy(out.logits, labels)
        assert float((out.loss - expected).abs().item()) < 1e-6


class TestInitialisation:
    """The released code's initialisation, which the paper does not state.

    Only a from-scratch run can tell the difference, and this family is
    for from-scratch runs: the targets are the encoder's own averaged
    output, so the width it starts at is not a detail that washes out.
    """

    def test_the_tower_starts_at_the_released_width(self) -> None:
        lucid.manual_seed(0)
        model = IJEPAModel(_tiny(dim=32, depth=4, num_heads=4))
        block = model.encoder.blocks[0]

        assert float(block.attn.qkv.weight.std().item()) == pytest.approx(0.02, rel=0.15)
        assert float(block.mlp.fc1.weight.std().item()) == pytest.approx(0.02, rel=0.15)
        assert float(model.encoder.patch_embed.proj.weight.std().item()) == pytest.approx(
            0.02, rel=0.15
        )

    def test_biases_start_at_zero_and_norms_at_one(self) -> None:
        model = IJEPAModel(_tiny(dim=32, depth=2, num_heads=4))
        block = model.encoder.blocks[0]
        assert float(block.attn.qkv.bias.abs().max().item()) == 0.0
        assert float(block.mlp.fc2.bias.abs().max().item()) == 0.0
        assert float((block.norm1.weight - 1.0).abs().max().item()) == 0.0
        assert float(block.norm1.bias.abs().max().item()) == 0.0

    def test_each_block_narrows_its_residual_writes_by_its_depth(self) -> None:
        # A block adds to the residual stream twice, so the two projections
        # it adds through are drawn at 0.02 / sqrt(2 * depth).
        lucid.manual_seed(1)
        model = IJEPAModel(_tiny(dim=32, depth=4, num_heads=4))
        for depth_index, block in enumerate(model.encoder.blocks, start=1):
            want = 0.02 / math.sqrt(2.0 * depth_index)
            assert float(block.attn.proj.weight.std().item()) == pytest.approx(
                want, rel=0.15
            ), depth_index
            assert float(block.mlp.fc2.weight.std().item()) == pytest.approx(
                want, rel=0.15
            ), depth_index

    def test_the_deepest_block_writes_the_least(self) -> None:
        """Guards the test above — without the scaling every block matches."""
        lucid.manual_seed(2)
        model = IJEPAModel(_tiny(dim=32, depth=4, num_heads=4))
        widths = [float(block.mlp.fc2.weight.std().item()) for block in model.encoder.blocks]
        assert widths == sorted(widths, reverse=True), widths
        assert widths[0] > 1.7 * widths[-1]

    def test_the_predictor_is_initialised_as_its_own_tower(self) -> None:
        lucid.manual_seed(3)
        model = IJEPAModel(_tiny(dim=32, depth=2, num_heads=4, predictor_dim=32))
        first = model.predictor.predictor_blocks[0]
        assert float(first.attn.proj.weight.std().item()) == pytest.approx(
            0.02 / math.sqrt(2.0), rel=0.15
        )
        # The mask token is drawn before the sweep and must survive it:
        # the sweep visits linear, convolutional and norm layers only.
        assert float(model.predictor.mask_token.std().item()) == pytest.approx(0.02, rel=0.25)
