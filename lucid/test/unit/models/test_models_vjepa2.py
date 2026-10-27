"""Shape, masking, initialisation and conditioning contracts for V-JEPA 2.

The two families share an encoder and a rotary attention, so what is
checked here is mostly what the released implementation does that a shape
assertion cannot see: which residual writes are narrowed and by how much,
what the probe is made of, whether dropout stops at ``eval()``, and
whether an action at one step can reach a prediction before it.
"""

import math
from collections.abc import Callable

import pytest

import lucid
import lucid.optim as optim
from lucid.models import create_model, list_models
from lucid.models.generative.vjepa2_ac import (
    VJEPA2ACConfig,
    VJEPA2ACModel,
    VJEPA2ACWeights,
)
from lucid.models.vision.vjepa2 import (
    VJEPA2Config,
    VJEPA2ForVideoClassification,
    VJEPA2Model,
    VJEPA2ViTGiant384Weights,
    VJEPA2ViTGiantWeights,
    VJEPA2ViTHugeWeights,
    VJEPA2ViTLargeWeights,
)
from lucid.models.vision.vjepa2._model import _sincos_3d


def _vjepa_config(**overrides: object) -> VJEPA2Config:
    values: dict[str, object] = {
        "image_size": 16,
        "patch_size": 8,
        "tubelet_size": 2,
        "num_frames": 4,
        "dim": 24,
        "depth": 1,
        "num_heads": 2,
        "predictor_dim": 12,
        "predictor_depth": 1,
        "predictor_heads": 2,
        "predictor_num_mask_tokens": 2,
    }
    values.update(overrides)
    return VJEPA2Config(**values)  # type: ignore[arg-type]


def _ac_config(**overrides: object) -> VJEPA2ACConfig:
    values: dict[str, object] = {
        "image_size": 16,
        "patch_size": 8,
        "tubelet_size": 2,
        "num_frames": 4,
        "encoder_dim": 24,
        "encoder_depth": 1,
        "encoder_heads": 2,
        "predictor_dim": 24,
        "predictor_depth": 1,
        "predictor_heads": 2,
        "action_dim": 3,
        "state_dim": 3,
        "extrinsics_dim": 2,
        "use_extrinsics": True,
    }
    values.update(overrides)
    return VJEPA2ACConfig(**values)  # type: ignore[arg-type]


_SMALL: dict[str, object] = {
    "image_size": 16,
    "patch_size": 8,
    "num_frames": 4,
    "dim": 24,
    "depth": 1,
    "num_heads": 2,
    "predictor_dim": 12,
    "predictor_depth": 1,
    "predictor_heads": 2,
    "num_classes": 5,
    "num_pooler_layers": 2,
}


class TestRegistry:
    """What ``create_model`` can reach, and under which task."""

    def test_every_released_variant_is_registered(self) -> None:
        assert {
            "vjepa2_vit_large",
            "vjepa2_vit_huge",
            "vjepa2_vit_giant",
            "vjepa2_vit_giant_384",
        } <= set(list_models(task="base"))

    def test_every_backbone_also_comes_under_the_probe(self) -> None:
        """A task wrapper no factory builds is one no caller can use."""
        assert {
            "vjepa2_vit_large_cls",
            "vjepa2_vit_huge_cls",
            "vjepa2_vit_giant_cls",
            "vjepa2_vit_giant_384_cls",
        } <= set(list_models(task="image-classification"))

    def test_the_action_model_registers_as_a_world_model(self) -> None:
        assert "vjepa2_ac_vit_giant_world_model" in set(list_models(task="world-modeling"))

    def test_the_probe_refuses_a_checkpoint_it_does_not_have(self) -> None:
        """Only the backbones are published; the probe's head is not."""
        with pytest.raises(NotImplementedError, match="vjepa2_vit_large"):
            create_model("vjepa2_vit_large_cls", pretrained=True)


class TestReleasedGeometry:
    """Configuration facts taken from the release, not from a default."""

    def test_the_predictor_ratio_is_independent_of_the_encoder_s(self) -> None:
        assert VJEPA2Config(mlp_ratio=48.0 / 11.0).predictor_mlp_ratio == 4.0
        assert VJEPA2ACConfig().predictor_mlp_ratio == 4.0

    def test_the_probe_carries_three_blocks(self) -> None:
        """Every released classifier config says ``num_pooler_layers: 3``."""
        assert VJEPA2Config().num_pooler_layers == 3

    def test_the_momentum_is_constant_in_the_released_recipe(self) -> None:
        assert VJEPA2Config().ema == (0.99925, 0.99925)

    def test_the_backbone_tags_carry_their_checksums(self) -> None:
        assert VJEPA2ViTLargeWeights.DEFAULT.entry.sha256
        assert VJEPA2ViTHugeWeights.DEFAULT.entry.sha256
        assert VJEPA2ViTGiantWeights.DEFAULT.entry.sha256
        assert VJEPA2ViTGiant384Weights.DEFAULT.entry.sha256

    def test_the_action_tag_names_the_official_checkpoint(self) -> None:
        source = VJEPA2ACWeights.DEFAULT.entry.meta["source"]
        assert isinstance(source, str) and source.endswith("vjepa2-ac-vitg.pt")

    def test_the_action_tag_carries_its_checksum(self) -> None:
        """An entry with an empty hash points at nothing that can load."""
        entry = VJEPA2ACWeights.DEFAULT.entry
        assert len(entry.sha256) == 64
        assert float(entry.meta["file_size_mb"]) > 0.0

    def test_the_action_checkpoint_counts_what_the_model_holds(self) -> None:
        """Encoder plus predictor — the release's 1B counts the encoder."""
        assert VJEPA2ACWeights.DEFAULT.entry.meta["num_params"] == 1_317_394_944

    def test_a_config_from_json_keeps_its_pair(self) -> None:
        """Round-tripping through JSON hands back a list, not a tuple."""
        config = VJEPA2Config(ema=[0.99, 1.0])  # type: ignore[arg-type]
        assert config.ema == (0.99, 1.0)

    @pytest.mark.parametrize("field", ["ema", "ema_schedule_scale", "num_pooler_layers"])
    def test_a_configuration_no_model_can_be_built_from_is_refused(self, field: str) -> None:
        bad: dict[str, object] = {
            "ema": (1.5, 1.0),
            "ema_schedule_scale": 0.0,
            "num_pooler_layers": -1,
        }
        with pytest.raises(ValueError):
            VJEPA2Config(**{field: bad[field]})  # type: ignore[arg-type]


class TestPositionTable:
    """The non-rotary table, which only the ``use_rope=False`` path reads."""

    def test_an_odd_axis_width_is_padded_not_truncated(self) -> None:
        assert tuple(
            _sincos_3d(12, (2, 2, 2), False, device=lucid.device("cpu"), dtype=lucid.float32).shape
        ) == (1, 8, 12)

    def test_a_float32_cpu_table_is_the_float64_one_rounded_once(self) -> None:
        # The reference builds its table in float64; a float32 build drifts
        # by ~1e-6 at a released grid, which a checkpoint would see.
        cpu = lucid.device("cpu")
        narrow = _sincos_3d(1024, (8, 16, 16), True, device=cpu, dtype=lucid.float32)
        wide = _sincos_3d(1024, (8, 16, 16), True, device=cpu, dtype=lucid.float64)
        assert narrow.dtype == lucid.float32
        assert bytes(narrow._impl.to_bytes()) == bytes(wide.to(lucid.float32)._impl.to_bytes())

    def test_uniform_power_changes_the_split(self) -> None:
        even = _sincos_3d(24, (2, 2, 2), True, device=lucid.device("cpu"), dtype=lucid.float32)
        split = _sincos_3d(24, (2, 2, 2), False, device=lucid.device("cpu"), dtype=lucid.float32)
        assert float((even - split).abs().max().item()) > 0.0

    def test_the_config_chooses_the_split(self) -> None:
        """``uniform_power`` was read from the config, not hard-coded."""
        video = lucid.randn(1, 4, 3, 16, 16)
        plain = VJEPA2Model(_vjepa_config(use_rope=False)).eval()
        uniform = VJEPA2Model(_vjepa_config(use_rope=False, uniform_power=True)).eval()
        lucid.nn.utils.copy_parameters_and_buffers(plain, uniform)
        difference = (plain.tokens(video) - uniform.tokens(video)).abs().max()
        assert float(difference.item()) > 0.0


class TestPredictorStaysOnDevice:
    """LCD-373: the predictor read its token indices back to the host to size
    the time axis, a sync no compile trace can follow.  The depth now comes
    from the config, as the reference's ``grid_depth`` does."""

    @pytest.mark.parametrize("use_rope", [True, False])
    def test_forward_reads_nothing_to_the_host(
        self, use_rope: bool, host_reads: Callable[[Callable[[], object]], list[str]]
    ) -> None:
        model = VJEPA2Model(_vjepa_config(use_rope=use_rope))
        video = lucid.randn(1, 4, 3, 16, 16)
        ctx = lucid.arange(6, dtype=lucid.int32).reshape(1, 6)
        tgt = lucid.arange(6, 8, dtype=lucid.int32).reshape(1, 2)
        reads = host_reads(lambda: model(video, ctx, tgt))
        assert reads == []

    def test_a_target_in_a_partial_last_frame_is_positioned(self) -> None:
        """The old depth, ``(max index + 1) // tokens per frame``, floored a
        partial frame away, so its tokens indexed past the position table."""
        model = VJEPA2Model(_vjepa_config(use_rope=False)).eval()
        context = lucid.randn(1, 4, 24)
        ctx = lucid.arange(4, dtype=lucid.int32).reshape(1, 4)  # frame 0
        tgt = lucid.tensor([[5]], dtype=lucid.int32)  # one token of frame 1
        out = model.predictor(context, ctx, tgt)
        assert tuple(out.shape) == (1, 1, 24)
        assert bool(lucid.isfinite(out).all().item())


class TestMaskedPrediction:
    """One masked step: what is frozen, what is predicted, what is compared."""

    def test_the_prediction_answers_at_the_held_out_positions(self) -> None:
        config = _vjepa_config()
        model = VJEPA2Model(config)
        video = lucid.randn(1, 4, 3, 16, 16)
        context_indices = lucid.arange(6, dtype=lucid.int32).reshape(1, 6)
        target_indices = lucid.arange(6, 8, dtype=lucid.int32).reshape(1, 2)

        output = model(video, context_indices, target_indices)

        assert output.prediction is not None and output.target is not None
        assert tuple(output.prediction.shape) == (1, 2, config.dim)
        assert tuple(output.target.shape) == (1, 2, config.dim)
        assert output.loss is not None and output.loss.ndim == 0
        assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())

    def test_a_step_trains_the_encoder_and_the_predictor(self) -> None:
        """The family's training step, through the public factory."""
        model = create_model("vjepa2_vit_large", **_SMALL)
        optimiser = optim.SGD(model.trainable_parameters(), lr=0.1)
        video = lucid.randn(2, 4, 3, 16, 16)
        context = lucid.arange(6, dtype=lucid.int32).repeat(2, 1)
        target = lucid.arange(6, 8, dtype=lucid.int32).repeat(2, 1)

        before = [p.detach().clone() for p in model.trainable_parameters()]
        output = model(video, context, target)
        assert output.loss is not None
        assert math.isfinite(float(output.loss.item()))
        output.loss.backward()
        optimiser.step()

        # Every mask-token slot but the one this step asked for stays out
        # of the graph, which is what ten slots are for.
        unused = {
            id(token) for index, token in enumerate(model.predictor.mask_tokens) if index != 1
        }
        for parameter in model.trainable_parameters():
            if id(parameter) in unused:
                continue
            assert parameter.grad is not None
            assert math.isfinite(float(parameter.grad.abs().sum().item()))
        moved = any(
            float((now - was).abs().max().item()) > 0.0
            for now, was in zip(model.trainable_parameters(), before)
        )
        assert moved
        assert all(p.grad is None for p in model.target_encoder.parameters())

    def test_the_returned_tokens_can_include_the_context(self) -> None:
        """``predictor_return_all_tokens`` is a config field, not a constant."""
        config = _vjepa_config(predictor_return_all_tokens=True)
        model = VJEPA2Model(config).eval()
        video = lucid.randn(1, 4, 3, 16, 16)
        context = lucid.arange(6, dtype=lucid.int32).reshape(1, 6)
        target = lucid.arange(6, 8, dtype=lucid.int32).reshape(1, 2)

        output = model(video, context, target)
        assert output.prediction is not None
        assert tuple(output.prediction.shape) == (1, 8, config.dim)
        # Only the held-out positions have a target to be scored against.
        assert output.target is not None
        assert tuple(output.target.shape) == (1, 2, config.dim)
        assert output.loss is not None and output.loss.ndim == 0


class TestMomentum:
    """The EMA schedule, which the released recipe holds flat."""

    def test_the_released_schedule_is_flat(self) -> None:
        model = VJEPA2Model(_vjepa_config())
        assert model.momentum(0, 1000) == pytest.approx(0.99925)
        assert model.momentum(1000, 1000) == pytest.approx(0.99925)

    def test_a_ramp_is_still_a_ramp(self) -> None:
        """The arguments are read, so another recipe can be given."""
        model = VJEPA2Model(_vjepa_config(ema=(0.9, 1.0), ema_schedule_scale=1.0))
        assert model.momentum(0, 100) == pytest.approx(0.9)
        assert model.momentum(50, 100) == pytest.approx(0.95)
        assert model.momentum(100, 100) == pytest.approx(1.0)

    def test_the_average_lands_where_the_momentum_says(self) -> None:
        model = VJEPA2Model(_vjepa_config())
        with lucid.no_grad():
            for parameter in model.encoder.parameters():
                parameter[:] = parameter + 1.0
        target = next(iter(model.target_encoder.parameters())).detach().clone()
        live = next(iter(model.encoder.parameters())).detach().clone()

        model.update_target(0.75)

        moved = next(iter(model.target_encoder.parameters()))
        expected = 0.75 * target + 0.25 * live
        assert float((moved - expected).abs().max().item()) < 1e-6


class TestInitialisation:
    """What the released code draws, which loading weights would hide."""

    def test_each_block_narrows_its_residual_writes_by_its_depth(self) -> None:
        config = _vjepa_config(dim=64, depth=4, num_heads=4)
        model = VJEPA2Model(config)
        for index, block in enumerate(model.encoder.blocks, start=1):
            wanted = config.init_std / math.sqrt(2.0 * index)
            assert float(block.attn.proj.weight.std().item()) == pytest.approx(wanted, rel=0.15)
            assert float(block.mlp.fc2.weight.std().item()) == pytest.approx(wanted, rel=0.15)

    def test_the_tubelet_embedding_starts_at_the_released_width(self) -> None:
        model = VJEPA2Model(_vjepa_config(dim=64, num_heads=4))
        weight = model.encoder.patch_embed.weight
        assert float(weight.std().item()) == pytest.approx(0.02, rel=0.25)

    def test_the_probe_is_initialised_as_a_tower(self) -> None:
        """Left out, the probe would start at the framework's fan-in default."""
        config = _vjepa_config(dim=64, num_heads=4, num_classes=5, num_pooler_layers=3)
        model = VJEPA2ForVideoClassification(config)
        assert float(model.pooler.query.weight.std().item()) == pytest.approx(0.02, rel=0.25)
        for index, block in enumerate(model.pooler.blocks, start=1):
            wanted = config.init_std / math.sqrt(2.0 * index)
            assert float(block.proj.weight.std().item()) == pytest.approx(wanted, rel=0.2)

    def test_the_mask_tokens_stay_at_zero(self) -> None:
        model = VJEPA2Model(_vjepa_config())
        for token in model.predictor.mask_tokens:
            assert float(token.abs().max().item()) == 0.0


class TestRegularisation:
    """Fields a config declares have to reach the forward pass."""

    def test_stochastic_depth_rises_with_depth(self) -> None:
        model = VJEPA2Model(_vjepa_config(depth=4, drop_path_rate=0.3))
        rates = [float(block.drop_path.drop_prob) for block in model.encoder.blocks]
        assert rates == pytest.approx([0.0, 0.1, 0.2, 0.3])

    def test_attention_dropout_stops_at_eval(self) -> None:
        """A rate forwarded unconditionally keeps dropping at inference."""
        model = VJEPA2Model(_vjepa_config(attn_drop_rate=0.5)).eval()
        video = lucid.randn(1, 4, 3, 16, 16)
        first = model.tokens(video)
        second = model.tokens(video)
        assert float((first - second).abs().max().item()) == 0.0


class TestProbe:
    """The probe V-JEPA 2 evaluates through, which is not V-JEPA 1's."""

    def test_the_cross_attention_has_no_output_projection(self) -> None:
        """The released implementation carries that ``Linear`` commented out."""
        model = VJEPA2ForVideoClassification(_vjepa_config(num_classes=5))
        assert not hasattr(model.pooler, "proj")

    def test_the_probe_carries_the_released_tensors(self) -> None:
        """Pinned against ``facebook/vjepa2-vitl-fpc16-256-ssv2``.

        Its ``pooler.*`` tensors map onto these one-for-one — three
        self-attention blocks with an output projection, then a cross
        attention without one.  Checked numerically at ``4.3e-6`` when
        that checkpoint's probe was loaded into this class; frozen here
        so the structure cannot drift without a test saying so.
        """
        config = _vjepa_config(num_classes=5, num_pooler_layers=3)
        model = VJEPA2ForVideoClassification(config)
        probe = {key for key in model.state_dict() if key.startswith(("pooler.", "head."))}
        tower = {
            f"pooler.blocks.{index}.{leaf}"
            for index in range(3)
            for leaf in (
                "norm1.weight",
                "norm1.bias",
                "norm2.weight",
                "norm2.bias",
                "qkv.weight",
                "qkv.bias",
                "proj.weight",
                "proj.bias",
                "mlp.fc1.weight",
                "mlp.fc1.bias",
                "mlp.fc2.weight",
                "mlp.fc2.bias",
            )
        }
        cross = {
            "pooler.query_token",
            "pooler.norm_keys.weight",
            "pooler.norm_keys.bias",
            "pooler.norm_out.weight",
            "pooler.norm_out.bias",
            "pooler.query.weight",
            "pooler.query.bias",
            "pooler.key.weight",
            "pooler.key.bias",
            "pooler.value.weight",
            "pooler.value.bias",
            "pooler.mlp.fc1.weight",
            "pooler.mlp.fc1.bias",
            "pooler.mlp.fc2.weight",
            "pooler.mlp.fc2.bias",
        }
        assert probe == tower | cross | {"head.weight", "head.bias"}
        assert len(probe) == 53

    def test_the_tower_runs_before_the_query_attends(self) -> None:
        config = _vjepa_config(num_classes=5, num_pooler_layers=2)
        model = VJEPA2ForVideoClassification(config)
        assert len(model.pooler.blocks) == 2

    def test_the_head_gives_one_logit_per_class(self) -> None:
        model = create_model("vjepa2_vit_large_cls", **_SMALL)
        logits = model(lucid.randn(2, 4, 3, 16, 16)).logits
        assert tuple(logits.shape) == (2, 5)

    def test_a_step_trains_the_probe_and_leaves_the_backbone(self) -> None:
        model = create_model("vjepa2_vit_large_cls", **_SMALL)
        probe = list(model.pooler.parameters()) + list(model.head.parameters())
        optimiser = optim.SGD(probe, lr=0.1)
        video = lucid.randn(2, 4, 3, 16, 16)
        labels = lucid.randint(0, 5, (2,))

        before = [p.detach().clone() for p in probe]
        output = model(video, labels=labels)
        assert output.loss is not None
        assert math.isfinite(float(output.loss.item()))
        output.loss.backward()
        optimiser.step()

        for parameter in probe:
            assert parameter.grad is not None
        assert any(float((now - was).abs().max().item()) > 0.0 for now, was in zip(probe, before))
        assert all(parameter.grad is None for parameter in model.vjepa2.target_encoder.parameters())

    def test_the_pooled_vector_depends_on_every_token(self) -> None:
        model = VJEPA2ForVideoClassification(_vjepa_config(num_classes=5)).eval()
        tokens = lucid.randn(1, 8, 24)
        base = model.pooler(tokens)
        changed = tokens.detach().clone()
        changed[0, 5] = changed[0, 5] + 5.0
        assert float((model.pooler(changed) - base).abs().max().item()) > 0.0


class TestActionConditioned:
    """The AC predictor: one step per frame, next-frame targets."""

    def test_one_step_is_one_frame(self) -> None:
        """The released loop repeats a frame to fill the tubelet.

        Pairing two distinct frames instead would halve the temporal
        resolution and take one action per two frames.
        """
        config = _ac_config()
        model = VJEPA2ACModel(config).eval()
        with lucid.no_grad():
            encoded = model.encode(lucid.randn(1, 5, 3, 16, 16))
        assert tuple(encoded.shape) == (1, 5 * config.tokens_per_step, 24)

    def test_an_odd_frame_count_is_accepted(self) -> None:
        """A step is a frame, so nothing has to divide by tubelet_size."""
        model = VJEPA2ACModel(_ac_config()).eval()
        with lucid.no_grad():
            encoded = model.encode(lucid.randn(1, 3, 3, 16, 16))
        assert int(encoded.shape[1]) == 3 * _ac_config().tokens_per_step

    def test_a_step_trains_the_predictor(self) -> None:
        config = _ac_config()
        model = VJEPA2ACModel(config)
        video = lucid.randn(1, 4, 3, 16, 16)
        actions = lucid.randn(1, 3, 3)
        states = lucid.randn(1, 3, 3)
        extrinsics = lucid.randn(1, 3, 2)

        output = model(video, actions, states, extrinsics)
        assert output.loss is not None
        assert math.isfinite(float(output.loss.item()))
        output.loss.backward()

        gradients = [p.grad for p in model.predictor.parameters()]
        assert gradients and all(g is not None for g in gradients)

    def test_the_conditioning_is_per_transition(self) -> None:
        """``T`` frames give ``T - 1`` transitions, and that many rows."""
        config = _ac_config(use_extrinsics=False)
        model = VJEPA2ACModel(config).eval()
        video = lucid.randn(1, 4, 3, 16, 16)
        with lucid.no_grad():
            output = model(video, lucid.randn(1, 3, 3), lucid.randn(1, 3, 3))
        spatial = config.tokens_per_step
        assert tuple(output.prediction.shape) == (1, 3 * spatial, config.encoder_dim)
        assert tuple(output.context.shape) == (1, 3 * spatial, config.encoder_dim)

    def test_the_target_is_the_next_frame_not_the_present(self) -> None:
        """Scoring the unshifted latents is satisfied by copying the input."""
        config = _ac_config(use_extrinsics=False)
        model = VJEPA2ACModel(config).eval()
        video = lucid.randn(1, 4, 3, 16, 16)
        with lucid.no_grad():
            output = model(video, lucid.randn(1, 3, 3), lucid.randn(1, 3, 3))
            encoded = model.normalize(model.encode(video))
        spatial = config.tokens_per_step
        assert output.target is not None
        shifted = encoded[:, spatial:]
        assert float((output.target - shifted).abs().max().item()) < 1e-6
        present = encoded[:, :-spatial]
        assert float((output.target - present).abs().max().item()) > 1e-3

    def test_both_sides_of_the_loss_are_normalised(self) -> None:
        config = _ac_config(use_extrinsics=False)
        model = VJEPA2ACModel(config).eval()
        with lucid.no_grad():
            output = model(
                lucid.randn(1, 4, 3, 16, 16),
                lucid.randn(1, 3, 3),
                lucid.randn(1, 3, 3),
            )
        assert output.target is not None
        for side in (output.prediction, output.target):
            assert float(side.mean(dim=-1).abs().max().item()) < 1e-4

    def test_a_later_action_cannot_reach_an_earlier_step(self) -> None:
        """What ``frame_causal`` buys, and the mask is easy to get backwards."""
        config = _ac_config(use_extrinsics=False)
        model = VJEPA2ACModel(config).eval()
        video = lucid.randn(1, 5, 3, 16, 16)
        actions = lucid.randn(1, 4, 3)
        states = lucid.randn(1, 4, 3)
        spatial = config.tokens_per_step

        with lucid.no_grad():
            base = model(video, actions, states).prediction
            later = actions.detach().clone()
            later[0, 3] = later[0, 3] + 10.0
            changed = model(video, later, states).prediction

        head = (changed - base)[:, : 3 * spatial]
        tail = (changed - base)[:, 3 * spatial :]
        assert float(head.abs().max().item()) < 1e-5
        assert float(tail.abs().max().item()) > 1e-5

    def test_a_single_frame_is_not_a_transition(self) -> None:
        model = VJEPA2ACModel(_ac_config())
        with pytest.raises(ValueError, match="two frames"):
            model(lucid.randn(1, 1, 3, 16, 16), lucid.randn(1, 1, 3), lucid.randn(1, 1, 3))

    def test_a_video_of_the_wrong_shape_is_refused(self) -> None:
        model = VJEPA2ACModel(_ac_config())
        with pytest.raises(ValueError, match=r"\(C, H, W\)"):
            model.encode(lucid.randn(1, 4, 3, 8, 8))
