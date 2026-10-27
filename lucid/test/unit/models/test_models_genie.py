"""Genie — the paper's structural claims, asserted rather than assumed.

Genie is three transformers whose value lies entirely in what each one is
*not* allowed to see.  The latent action model's decoder must not see the
frame it predicts, or the eight codes learn nothing.  Temporal attention
must not see the future, or a generated frame depends on frames not yet
generated.  The dynamics model must be conditioned on the action that
leads *to* a frame, not the one that leaves it.  A shape test passes
every one of those mis-wirings, so each test here names the one it
catches.

This file also trains the family (``test_models_train_step.ELSEWHERE``).
"""

import math
from typing import cast

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.models import AutoConfig, list_models
from lucid.models.generative.genie import (
    GenieConfig,
    GenieForWorldModeling,
    GenieModel,
    genie,
    genie_coinrun,
    genie_coinrun_world_model,
    genie_world_model,
)
from lucid.models.generative.genie._model import (
    _Codebook,
    _patchify,
    _STBlock,
    _STTransformer,
    _unpatchify,
)
from lucid.nn._shadow import shadow_alloc


def _tiny(**overrides: object) -> GenieConfig:
    """A model small enough to instantiate per test."""
    base: dict[str, object] = dict(
        sample_size=(8, 8),
        num_frames=4,
        num_codes=16,
        code_dim=4,
        tokenizer_encoder_layers=1,
        tokenizer_encoder_dim=16,
        tokenizer_encoder_heads=2,
        tokenizer_encoder_head_dim=8,
        tokenizer_decoder_layers=1,
        tokenizer_decoder_dim=16,
        tokenizer_decoder_heads=2,
        tokenizer_decoder_head_dim=8,
        action_patch_size=4,
        action_dim=4,
        action_encoder_layers=1,
        action_encoder_dim=16,
        action_encoder_heads=2,
        action_decoder_layers=1,
        action_decoder_dim=16,
        action_decoder_heads=2,
        dynamics_layers=1,
        dynamics_dim=16,
        dynamics_heads=2,
        dynamics_head_dim=8,
        maskgit_steps=4,
    )
    base.update(overrides)
    return GenieConfig(**base)  # type: ignore[arg-type]


def _video(batch: int = 2, config: GenieConfig | None = None) -> lucid.Tensor:
    config = _tiny() if config is None else config
    h, w = config.frame_shape
    return lucid.rand(batch, config.num_frames, config.in_channels, h, w)


def _close(a: lucid.Tensor, b: lucid.Tensor, tol: float = 1e-5) -> bool:
    return float((a - b).abs().max().item()) <= tol


def _count(module: nn.Module) -> int:
    return sum(math.prod(int(s) for s in p.shape) for p in module.parameters())


class TestPaperConfigurations:
    def test_platformers_is_tables_5_7_and_12(self) -> None:
        config = AutoConfig.from_pretrained("genie")
        assert isinstance(config, GenieConfig)
        assert (
            config.tokenizer_encoder_layers,
            config.tokenizer_encoder_dim,
            config.tokenizer_encoder_heads,
            config.tokenizer_decoder_layers,
            config.tokenizer_decoder_dim,
            config.tokenizer_decoder_heads,
        ) == (12, 512, 8, 20, 1024, 16)
        assert (config.num_codes, config.code_dim, config.tokenizer_patch_size) == (
            1024,
            32,
            4,
        )
        assert (
            config.num_latent_actions,
            config.action_dim,
            config.action_patch_size,
        ) == (
            8,
            32,
            16,
        )
        assert (
            config.dynamics_layers,
            config.dynamics_dim,
            config.dynamics_heads,
            config.dynamics_head_dim,
        ) == (48, 5120, 36, 128)
        assert (config.maskgit_steps, config.temperature) == (25, 2.0)

    def test_coinrun_is_appendix_f(self) -> None:
        config = AutoConfig.from_pretrained("genie_coinrun")
        assert isinstance(config, GenieConfig)
        for stack in (
            "tokenizer_encoder",
            "tokenizer_decoder",
            "action_encoder",
            "action_decoder",
        ):
            assert (
                getattr(config, f"{stack}_layers"),
                getattr(config, f"{stack}_dim"),
                getattr(config, f"{stack}_heads"),
            ) == (8, 512, 8), stack
        assert (config.dynamics_layers, config.dynamics_dim, config.dynamics_heads) == (
            12,
            512,
            8,
        )
        assert (
            config.num_latent_actions,
            config.temperature,
            config.maskgit_steps,
        ) == (
            6,
            1.0,
            25,
        )

    def test_the_attention_is_narrower_than_the_model_where_the_table_says(
        self,
    ) -> None:
        # 36 heads of 128: tying the inner width to d_model would build a
        # different network from the one Table 12 describes.
        config = GenieConfig()
        assert config.dynamics_heads * config.attention_head_dim("dynamics") == 4608
        assert config.attention_head_dim("action_encoder") == 1024 // 16

    def test_ninety_pixels_pad_to_twenty_three_rows(self) -> None:
        # 942B tokens over 512 x 125k x 16 frames is 920 per frame: 40x23.
        assert GenieConfig().token_grid == (23, 40)
        assert 512 * 125_000 * 16 * 23 * 40 == 942_080_000_000

    def test_the_build_is_the_tables_not_the_reported_counts(self) -> None:
        """Pins the parameter counts, and why they are not the paper's.

        The paper reports 200M, 300M and 10.1B.  With the feed-forward
        layers all but removed, the latent action model is still over its
        300M, while the tokenizer and dynamics model fall under theirs by
        margins that would need different feed-forward widths — so no
        choice of the values the paper leaves out matches all three.  A
        change to these numbers is a change to the network.
        """
        with shadow_alloc():
            platformers = genie()
            coinrun = genie_coinrun()
            narrowest = genie(mlp_ratio=1e-9)
        assert _count(platformers) == 20_208_496_240
        assert _count(coinrun) == 187_836_976
        attention = {
            "tokenizer": _count(narrowest.tokenizer),
            "latent_action_model": _count(narrowest.latent_action_model),
            "dynamics": _count(narrowest.dynamics),
        }
        assert attention["latent_action_model"] > 300_000_000
        full = {
            "tokenizer": _count(platformers.tokenizer),
            "dynamics": _count(platformers.dynamics),
        }
        reported = {"tokenizer": 200_000_000, "dynamics": 10_100_000_000}
        # The fraction of the 4x feed-forward each would have to keep.
        needed = {k: (reported[k] - attention[k]) / (full[k] - attention[k]) for k in reported}
        assert abs(needed["tokenizer"] - needed["dynamics"]) > 0.05

    def test_no_weights_were_released(self) -> None:
        for factory in (
            genie,
            genie_world_model,
            genie_coinrun,
            genie_coinrun_world_model,
        ):
            with pytest.raises(NotImplementedError):
                factory(pretrained=True)

    def test_the_factories_register_under_their_tasks(self) -> None:
        assert {"genie", "genie_coinrun"} <= set(list_models(task="base"))
        assert {"genie_world_model", "genie_coinrun_world_model"} <= set(
            list_models(task="world-modeling")
        )


class TestConfigValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"num_frames": 1},
            {"num_latent_actions": 1},
            {"mask_ratio_min": 0.0},
            {"mask_ratio_min": 0.8, "mask_ratio_max": 0.6},
            {"temperature": 0.0},
            {"action_encoder_dim": 30, "action_encoder_heads": 4},
            {"dynamics_head_dim": 0},
        ],
    )
    def test_a_config_no_network_can_be_built_from_is_refused(
        self, overrides: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError):
            _tiny(**overrides)

    def test_a_json_list_frame_shape_becomes_a_tuple(self) -> None:
        assert _tiny(sample_size=[8, 12]).frame_shape == (8, 12)  # type: ignore[arg-type]


class TestCodebookCollapse:
    """Eight codes collapse to two, and nothing in the loss says so.

    A latent action codebook that has lost most of its entries still
    trains and still reconstructs — the surviving codes take up the slack
    — so the model quietly has three buttons instead of eight.  These
    tests are the only place that failure is visible.
    """

    @staticmethod
    def _two_clusters() -> lucid.Tensor:
        # Two codes can explain this; the rest of the codebook has nothing
        # to do and, left alone, would never move again.
        return lucid.tensor([[5.0, 5.0], [-5.0, -5.0]] * 16)

    def _run(self, codebook: _Codebook, steps: int) -> lucid.Tensor:
        points = self._two_clusters()
        out = codebook(points)
        for _ in range(steps):
            out = codebook(points)
            codebook.revive()
        return out.indices

    def test_a_dead_code_is_moved_onto_the_data(self) -> None:
        lucid.manual_seed(0)
        codebook = _Codebook(6, 2, commitment_cost=0.25, reset_threshold=0.1)

        def on_the_data() -> int:
            return sum(
                max(abs(abs(v) - 5.0) for v in row) < 1e-6 for row in codebook.weight.tolist()
            )

        assert on_the_data() == 0
        self._run(codebook, 400)
        # Two codes explain this data, so a revived one lands on a cluster
        # a live code already holds, loses every assignment and is revived
        # again later — at any instant one may be mid-cycle.  What must not
        # happen is codes sitting forever where initialisation left them.
        assert on_the_data() >= 4

    def test_a_code_in_use_is_left_where_it_is(self) -> None:
        # Nothing here runs an optimiser, so a code that moves moved
        # because it was revived.  The two live ones must not.
        lucid.manual_seed(1)
        codebook = _Codebook(6, 2, commitment_cost=0.25, reset_threshold=0.1)
        indices = self._run(codebook, 400)
        live = sorted({int(i) for i in indices.reshape(-1).tolist()})
        assert len(live) == 2
        before = [codebook.weight[i].tolist() for i in live]
        self._run(codebook, 200)
        assert [codebook.weight[i].tolist() for i in live] == before

    def test_the_revival_is_slow_enough_to_ignore_one_quiet_batch(self) -> None:
        # A code the batch merely did not reach must survive: a batch holds
        # fewer tokens than the tokenizer has codes.
        lucid.manual_seed(2)
        codebook = _Codebook(6, 2, commitment_cost=0.25, reset_threshold=0.1)
        before = codebook.weight.detach().clone()
        self._run(codebook, 20)
        assert float((codebook.weight - before).abs().max().item()) == 0.0

    def test_forward_alone_never_moves_a_code(self) -> None:
        """The whole reason reviving is a call of its own.

        A forward that wrote the codebook would sever it from any graph
        already holding it, and the gradient would stop arriving with
        nothing raised — see the accumulation test below.
        """
        lucid.manual_seed(3)
        codebook = _Codebook(6, 2, commitment_cost=0.25, reset_threshold=0.9)
        before = codebook.weight.detach().clone()
        for _ in range(400):
            codebook(self._two_clusters())
        assert float((codebook.weight - before).abs().max().item()) == 0.0

    def test_a_zero_threshold_switches_it_off(self) -> None:
        lucid.manual_seed(4)
        codebook = _Codebook(6, 2, commitment_cost=0.25, reset_threshold=0.0)
        before = codebook.weight.detach().clone()
        self._run(codebook, 400)
        assert float((codebook.weight - before).abs().max().item()) == 0.0

    def test_eval_watches_nothing(self) -> None:
        lucid.manual_seed(5)
        codebook = _Codebook(6, 2, commitment_cost=0.25, reset_threshold=0.1).eval()
        before = cast(lucid.Tensor, codebook.usage).detach().clone()
        for _ in range(50):
            codebook(self._two_clusters())
        codebook.revive()
        assert float((cast(lucid.Tensor, codebook.usage) - before).abs().max().item()) == 0.0

    def test_the_usage_buffer_is_saved_with_the_model(self) -> None:
        # It carries which codes were alive; a checkpoint that dropped it
        # would revive healthy codes on the first batch after loading.
        keys = set(GenieModel(_tiny()).state_dict())
        assert "latent_action_model.quantizer.usage" in keys
        assert "tokenizer.quantizer.usage" in keys
        # Where a revived code would land is one old batch, of no use to a
        # checkpoint, so it is not carried.
        assert "latent_action_model.quantizer.candidates" not in keys

    def test_accumulating_gradients_does_not_cost_the_codebook_its_own(self) -> None:
        """Two forwards, one backward — how a model this size is trained.

        Reviving inside ``forward`` used to move the codebook between the
        two, which severed it from the first graph and left both
        codebooks with no gradient at all, silently.  Reviving is its own
        call now, made when no graph is alive — and it hands back the
        gradient the in-place write would otherwise drop, so a loop that
        revives before stepping still applies one.
        """
        config = _tiny(code_reset_threshold=0.5)
        model = GenieModel(config)
        first = model(_video(2, config)).loss
        second = model(_video(2, config)).loss
        (first + second).backward()
        model.revive_codes()
        for name in ("tokenizer", "latent_action_model"):
            weight = getattr(model, name).quantizer.weight
            assert weight.grad is not None, name
            assert float(weight.grad.abs().sum().item()) > 0.0, name

    def test_reviving_before_the_step_still_leaves_a_step_to_take(self) -> None:
        # Writing a parameter in place drops its gradient.  A loop that
        # revives before the optimiser would otherwise apply nothing, and
        # nothing would say so.
        lucid.manual_seed(10)
        config = _tiny(code_reset_threshold=0.9)
        model = GenieModel(config)
        model(_video(2, config)).loss.backward()
        model.revive_codes()
        book = model.tokenizer.quantizer.weight
        assert book.grad is not None
        before = book.detach().clone()
        optimizer = lucid.optim.Adam(model.parameters(), lr=1e-2)
        optimizer.step()
        assert float((book - before).abs().max().item()) > 0.0

    def test_a_rollout_leaves_the_codebooks_alone_in_training_mode(self) -> None:
        # A rollout run to look at the model must not age the averages the
        # revival reads, nor move a code the user's buttons index.
        config = _tiny(code_reset_threshold=0.5)
        model = GenieForWorldModeling(config)
        assert model.training
        book = model.genie.latent_action_model.quantizer
        weight, usage = (
            book.weight.detach().clone(),
            cast(lucid.Tensor, book.usage).detach().clone(),
        )
        model(lucid.rand(1, 1, 3, 8, 8), lucid.tensor([[2, 5]], dtype=lucid.int64))
        assert float((book.weight - weight).abs().max().item()) == 0.0
        assert float((cast(lucid.Tensor, book.usage) - usage).abs().max().item()) == 0.0

    def test_the_frozen_pass_inside_dynamics_loss_is_not_counted(self) -> None:
        # dynamics_loss re-tokenizes under no_grad; counting that would age
        # both codebooks twice per step in the paper's staged schedule.
        config = _tiny(code_reset_threshold=0.5)
        model = GenieModel(config)
        usage = cast(lucid.Tensor, model.tokenizer.quantizer.usage).detach().clone()
        model.dynamics_loss(_video(2, config))
        after = cast(lucid.Tensor, model.tokenizer.quantizer.usage)
        assert float((after - usage).abs().max().item()) == 0.0

    def test_the_usage_average_is_the_histogram_of_the_batch(self) -> None:
        # Computed from a histogram rather than a dense one-hot: at the
        # paper's sizes the indicator of one batch would be tens of GB.
        lucid.manual_seed(9)
        codebook = _Codebook(4, 2, commitment_cost=0.25, reset_threshold=0.1)
        points = lucid.tensor([[5.0, 5.0], [-5.0, -5.0]] * 8)
        out = codebook(points)
        counts = [0.0] * 4
        for i in out.indices.reshape(-1).tolist():
            counts[int(i)] += 1.0 / 16.0
        fair = 0.25
        expected = [0.99 * fair + 0.01 * c for c in counts]
        got = cast(lucid.Tensor, codebook.usage).tolist()
        assert max(abs(a - b) for a, b in zip(expected, got)) < 1e-6

    def test_the_model_revives_both_codebooks(self) -> None:
        lucid.manual_seed(6)
        config = _tiny(num_latent_actions=6, code_reset_threshold=0.9)
        model = GenieModel(config)
        video = _video(2, config)
        before = model.latent_action_model.quantizer.weight.detach().clone()
        for _ in range(30):
            model.latent_action_loss(video)
            model.revive_codes()
        after = model.latent_action_model.quantizer.weight
        assert float((after - before).abs().max().item()) > 0.0


class TestPatches:
    def test_padding_is_added_and_cropped_back(self) -> None:
        video = lucid.rand(2, 3, 3, 10, 12)
        patches = _patchify(video, 4)
        assert patches.shape == (2, 3, 9, 48)
        restored = _unpatchify(patches, 4, (3, 3), 3, (10, 12))
        assert _close(restored, video, 0.0)


class TestSpatiotemporalBlock:
    def test_time_is_causal(self) -> None:
        # A frame's output must not depend on a later frame, or generation
        # would read frames that do not exist yet.
        config = _tiny()
        stack = _STTransformer(config, "tokenizer_encoder", 4).eval()
        x = lucid.randn(1, 4, 4, 16)
        y = x.detach().clone()
        y[:, 3] = lucid.randn(1, 4, 16)
        with lucid.no_grad():
            a, b = stack(x), stack(y)
        assert _close(a[:, :3], b[:, :3])
        assert not _close(a[:, 3], b[:, 3])

    def test_space_is_not_causal(self) -> None:
        # Every patch of a frame sees every other: the last patch changing
        # moves the first patch's output within the same frame.
        config = _tiny()
        stack = _STTransformer(config, "tokenizer_encoder", 4).eval()
        x = lucid.randn(1, 4, 4, 16)
        y = x.detach().clone()
        y[:, 0, 3] = lucid.randn(16)
        with lucid.no_grad():
            a, b = stack(x), stack(y)
        assert not _close(a[:, 0, 0], b[:, 0, 0])

    def test_one_feed_forward_after_both_attentions(self) -> None:
        # Section 2 omits the post-spatial FFW: four projections per
        # attention and two feed-forward layers, ten in all.
        block = _STBlock(16, 2, 8, 64, nn.GELU, qk_norm=False)
        linears = [m for m in block.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 10

    def test_only_the_dynamics_model_normalises_queries_and_keys(self) -> None:
        model = GenieModel(_tiny())
        assert model.dynamics.transformer.blocks[0].spatial.query_norm is not None
        assert model.tokenizer.encoder.blocks[0].spatial.query_norm is None
        assert model.latent_action_model.encoder.blocks[0].temporal.key_norm is None


class TestLatentActionModel:
    def test_an_action_does_not_read_frames_after_its_transition(self) -> None:
        config = _tiny()
        lam = GenieModel(config).latent_action_model.eval()
        video = _video(1, config)
        later = video.detach().clone()
        later[:, 3] = lucid.rand(1, 3, 8, 8)

        def continuous(v: lucid.Tensor) -> lucid.Tensor:
            hidden = lam.encoder(lam.encoder_in(_patchify(v, config.action_patch_size)))
            return lam.to_action(hidden[:, 1:].mean(dim=2))

        with lucid.no_grad():
            a, b = continuous(video), continuous(later)
        # Transitions 0->1 and 1->2 are fixed; 2->3 is the one that changed.
        assert _close(a[:, :2], b[:, :2])
        assert not _close(a[:, 2], b[:, 2])

    def test_the_decoder_never_sees_the_frame_it_predicts(self) -> None:
        # If it did, the action codes would carry nothing: the frame itself
        # would be the cheaper route to a perfect reconstruction.
        config = _tiny()
        lam = GenieModel(config).latent_action_model.eval()
        video = _video(1, config)
        actions = lucid.randn(1, 3, config.action_dim)
        changed = video.detach().clone()
        changed[:, 2] = lucid.rand(1, 3, 8, 8)
        with lucid.no_grad():
            a, b = lam.decode(video, actions), lam.decode(changed, actions)
        # prediction i is frame i + 1: predictions of frames 1 and 2 hold.
        assert _close(a[:, :2], b[:, :2])
        assert not _close(a[:, 2], b[:, 2])

    def test_the_action_reaches_the_frame_it_leads_to(self) -> None:
        config = _tiny()
        lam = GenieModel(config).latent_action_model.eval()
        video = _video(1, config)
        actions = lucid.randn(1, 3, config.action_dim)
        other = actions.detach().clone()
        other[:, 1] = lucid.randn(config.action_dim)
        with lucid.no_grad():
            a, b = lam.decode(video, actions), lam.decode(video, other)
        assert _close(a[:, 0], b[:, 0])
        assert not _close(a[:, 1], b[:, 1])


class TestDynamicsModel:
    def test_action_t_conditions_frame_t_plus_one(self) -> None:
        # Not the frame it leaves: that frame is already known.
        config = _tiny()
        dynamics = GenieModel(config).dynamics.eval()
        tokens = lucid.randint(0, config.num_codes, (1, 4, 4))
        actions = lucid.randn(1, 3, config.action_dim)
        other = actions.detach().clone()
        other[:, 1] = lucid.randn(config.action_dim)
        with lucid.no_grad():
            a, b = dynamics(tokens, actions), dynamics(tokens, other)
        assert _close(a[:, :2], b[:, :2])
        assert not _close(a[:, 2], b[:, 2])

    def test_the_prompt_frame_is_never_masked(self) -> None:
        config = _tiny(mask_ratio_min=1.0, mask_ratio_max=1.0)
        model = GenieModel(config)
        seen: list[lucid.Tensor] = []
        forward = model.dynamics.forward

        def capture(tokens: lucid.Tensor, actions: lucid.Tensor) -> lucid.Tensor:
            seen.append(tokens)
            return forward(tokens, actions)

        model.dynamics.forward = capture  # type: ignore[method-assign]
        tokens = lucid.randint(0, config.num_codes, (2, 4, 4))
        model._dynamics_objective(tokens, lucid.randn(2, 3, config.action_dim))
        inputs = seen[0]
        assert inputs[:, 0].tolist() == tokens[:, 0].tolist()
        assert int((inputs[:, 1:] == config.num_codes).sum().item()) == 2 * 3 * 4

    def test_the_loss_is_over_masked_tokens_of_later_frames(self) -> None:
        # At a rate of 1 the masked mean and the plain mean over frames
        # 2..T are the same number, so the rate has to be one that leaves
        # some tokens visible for this to say anything.
        lucid.manual_seed(7)
        config = _tiny(mask_ratio_min=0.5, mask_ratio_max=0.5)
        model = GenieModel(config)
        tokens = lucid.randint(0, config.num_codes, (2, 4, 4))
        loss, logits, mask = model._dynamics_objective(tokens, lucid.randn(2, 3, config.action_dim))
        flat_logits = logits.reshape(-1, config.num_codes)
        flat_tokens = tokens.reshape(-1)
        per_token = F.cross_entropy(flat_logits, flat_tokens, reduction="none")
        flat_mask = mask.reshape(-1)
        masked_mean = float(((per_token * flat_mask).sum() / flat_mask.sum()).item())
        assert abs(float(loss.item()) - masked_mean) < 1e-5

        # And it is not the mean over every token of frames 2..T, which is
        # what an implementation that forgot the mask would report.
        later = float(
            F.cross_entropy(
                logits[:, 1:].reshape(-1, config.num_codes),
                tokens[:, 1:].reshape(-1),
            ).item()
        )
        assert abs(float(loss.item()) - later) > 1e-3
        assert 0.0 < float(flat_mask.sum().item()) < float(flat_mask.shape[0])

    def test_the_mask_covers_only_frames_after_the_prompt(self) -> None:
        lucid.manual_seed(8)
        config = _tiny(mask_ratio_min=0.5, mask_ratio_max=0.5)
        model = GenieModel(config)
        _, _, mask = model._dynamics_objective(
            lucid.randint(0, config.num_codes, (2, 4, 4)),
            lucid.randn(2, 3, config.action_dim),
        )
        assert float(mask[:, 0].sum().item()) == 0.0
        assert float(mask[:, 1:].sum().item()) > 0.0


class TestObjectives:
    def test_the_shapes_of_one_training_pass(self) -> None:
        config = _tiny()
        out = GenieModel(config)(_video(2, config))
        assert out.tokens.shape == (2, 4, 4)
        assert out.actions.shape == (2, 3)
        assert out.logits.shape == (2, 4, 4, config.num_codes)
        assert out.reconstruction.shape == (2, 4, 3, 8, 8)
        assert out.prediction.shape == (2, 3, 3, 8, 8)
        assert math.isfinite(float(out.loss.item()))

    def test_one_backward_trains_all_three_networks(self) -> None:
        model = GenieModel(_tiny())
        model(_video()).loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing

    @pytest.mark.parametrize(
        ("objective", "network"),
        [
            ("tokenizer_loss", "tokenizer."),
            ("latent_action_loss", "latent_action_model."),
            ("dynamics_loss", "dynamics."),
        ],
    )
    def test_each_objective_reaches_only_its_own_network(
        self, objective: str, network: str
    ) -> None:
        # The dynamics model reads codes and stop-gradient actions: if
        # either leaked a gradient, training it would move the tokenizer.
        model = GenieModel(_tiny())
        getattr(model, objective)(_video()).backward()
        leaked = [
            n
            for n, p in model.named_parameters()
            if p.grad is not None and not n.startswith(network)
        ]
        trained = [n for n, p in model.named_parameters() if p.grad is not None]
        assert trained and not leaked

    def test_the_parameter_groups_partition_the_model(self) -> None:
        model = GenieModel(_tiny())
        groups = (
            model.tokenizer_parameters()
            + model.latent_action_parameters()
            + model.dynamics_parameters()
        )
        assert len(groups) == len(list(model.parameters()))
        assert len({id(p) for p in groups}) == len(groups)

    def test_training_lowers_every_objective(self) -> None:
        lucid.manual_seed(0)
        config = _tiny(mask_ratio_min=1.0, mask_ratio_max=1.0)
        model = GenieModel(config)
        video = _video(4, config)
        optimizer = lucid.optim.Adam(model.parameters(), lr=3e-3)
        first = model(video)
        start = (
            float(first.tokenizer_loss.item()),
            float(first.latent_action_loss.item()),
            float(first.dynamics_loss.item()),
        )
        # Joint training updates the tokenizer codebook while the dynamics
        # model chases its discrete assignments: around steps 25-35 the
        # tokenizer breaks out of initial single-code collapse into ~12
        # active codes, which transiently raises the target entropy before
        # the dynamics transformer fits the settled codes by step 60.
        for _ in range(60):
            optimizer.zero_grad()
            model(video).loss.backward()
            optimizer.step()
        last = model(video)
        end = (
            float(last.tokenizer_loss.item()),
            float(last.latent_action_loss.item()),
            float(last.dynamics_loss.item()),
        )
        assert all(e < s for s, e in zip(start, end)), (start, end)


class TestPlaying:
    def test_every_token_of_a_generated_frame_is_revealed(self) -> None:
        config = _tiny()
        model = GenieForWorldModeling(config).eval()
        tokens = lucid.randint(0, config.num_codes, (2, 2, 4))
        with lucid.no_grad():
            frame = model.predict_frame(tokens, lucid.randn(2, 2, config.action_dim))
        assert frame.shape == (2, 4)
        assert int((frame >= config.num_codes).sum().item()) == 0

    def test_a_frame_is_revealed_a_few_tokens_at_a_time(self) -> None:
        # MaskGIT, not one-shot prediction: under the cosine schedule four
        # tokens over four steps are revealed one per step.
        config = _tiny(maskgit_steps=4)
        model = GenieForWorldModeling(config).eval()
        hidden: list[int] = []
        forward = model.genie.dynamics.forward

        def capture(tokens: lucid.Tensor, actions: lucid.Tensor) -> lucid.Tensor:
            hidden.append(int((tokens[:, -1] == config.num_codes).sum().item()))
            return forward(tokens, actions)

        model.genie.dynamics.forward = capture  # type: ignore[method-assign]
        tokens = lucid.randint(0, config.num_codes, (1, 2, 4))
        with lucid.no_grad():
            model.predict_frame(tokens, lucid.randn(1, 2, config.action_dim))
        assert hidden == [4, 3, 2, 1]

    def test_the_memory_is_num_frames(self) -> None:
        config = _tiny()
        model = GenieForWorldModeling(config).eval()
        lengths: list[int] = []
        forward = model.genie.dynamics.forward

        def capture(tokens: lucid.Tensor, actions: lucid.Tensor) -> lucid.Tensor:
            lengths.append(int(tokens.shape[1]))
            return forward(tokens, actions)

        model.genie.dynamics.forward = capture  # type: ignore[method-assign]
        tokens = lucid.randint(0, config.num_codes, (1, 9, 4))
        with lucid.no_grad():
            model.predict_frame(tokens, lucid.randn(1, 9, config.action_dim))
        assert set(lengths) == {config.num_frames}

    def test_a_rollout_longer_than_the_memory(self) -> None:
        config = _tiny()
        model = GenieForWorldModeling(config).eval()
        actions = lucid.tensor([[0, 1, 2, 3, 4, 5, 6]], dtype=lucid.int64)
        out = model(lucid.rand(1, 1, 3, 8, 8), actions)
        assert out.frames.shape == (1, 7, 3, 8, 8)
        assert out.tokens.shape == (1, 8, 4)
        assert 0.0 <= float(out.frames.min().item())
        assert float(out.frames.max().item()) <= 1.0

    def test_a_frame_decodes_the_same_however_the_rollout_is_split(self) -> None:
        # Within the first window the one-pass decode and a pass per frame
        # are the same computation — the decoder is causal and the frame
        # sits at the same position either way.  Past the window a frame
        # can only be decoded at the end of the most recent codes, which
        # is the most context it can be given.
        config = _tiny()
        model = GenieForWorldModeling(config).eval()
        tokens = lucid.randint(0, config.num_codes, (1, config.num_frames, 4))
        with lucid.no_grad():
            together = model._decode(tokens, 0)
            apart = lucid.cat(
                [
                    model.genie.detokenize(tokens[:, : j + 1])[:, -1:]
                    for j in range(config.num_frames)
                ],
                dim=1,
            )
        assert float((together - apart).abs().max().item()) < 1e-5

    def test_the_first_window_is_decoded_in_one_pass(self) -> None:
        # It used to cost one decoder pass per frame once the rollout grew
        # past the window, including for the frames that did not need it.
        config = _tiny()
        model = GenieForWorldModeling(config).eval()
        passes: list[int] = []
        detokenize = model.genie.detokenize

        def capture(tokens: lucid.Tensor) -> lucid.Tensor:
            passes.append(int(tokens.shape[1]))
            return detokenize(tokens)

        model.genie.detokenize = capture  # type: ignore[method-assign]
        with lucid.no_grad():
            model._decode(lucid.randint(0, config.num_codes, (1, 6, 4)), 0)
        # One pass for frames 0..3, then one each for 4 and 5.
        assert len(passes) == 3

    def test_a_button_is_a_row_of_the_action_codebook(self) -> None:
        # Section 2.2: at play time the latent action model is discarded
        # except for its codebook, and a user's integer indexes it.
        config = _tiny()
        model = GenieForWorldModeling(config).eval()
        received: list[lucid.Tensor] = []
        predict = model.predict_frame

        def capture(tokens: lucid.Tensor, actions: lucid.Tensor) -> lucid.Tensor:
            received.append(actions)
            return predict(tokens, actions)

        model.predict_frame = capture  # type: ignore[method-assign]
        prompt = lucid.rand(1, 2, 3, 8, 8)
        model(prompt, lucid.tensor([[5]], dtype=lucid.int64))
        codebook = model.genie.latent_action_model.quantizer.weight
        inferred = model.genie.latent_action_model.quantize(prompt).quantized
        (actions,) = received
        assert actions.shape == (1, 2, config.action_dim)
        assert _close(actions[:, 0], inferred[:, 0])
        assert _close(actions[0, 1], codebook[5])

    @pytest.mark.parametrize(
        "actions",
        [
            lucid.tensor([[8]], dtype=lucid.int64),
            lucid.tensor([[-1]], dtype=lucid.int64),
            lucid.tensor([[0], [1]], dtype=lucid.int64),
        ],
    )
    def test_an_action_the_codebook_cannot_hold_is_refused(self, actions: lucid.Tensor) -> None:
        model = GenieForWorldModeling(_tiny()).eval()
        with pytest.raises(ValueError):
            model(lucid.rand(1, 1, 3, 8, 8), actions)

    def test_an_action_per_transition_is_refused_where_one_per_frame_is_meant(
        self,
    ) -> None:
        # Every other action API here takes one per transition.  Truncation
        # would make this silent once the window is full, so it is refused.
        config = _tiny()
        model = GenieForWorldModeling(config).eval()
        tokens = lucid.randint(0, config.num_codes, (1, 4, 4))
        with pytest.raises(ValueError, match="one per frame"):
            model.predict_frame(tokens, lucid.randn(1, 3, config.action_dim))

    def test_a_decoder_that_cannot_emit_a_frame_is_refused(self) -> None:
        with pytest.raises(ValueError, match="out_channels"):
            _tiny(out_channels=1)

    def test_a_wrongly_sized_frame_is_refused(self) -> None:
        model = GenieModel(_tiny())
        with pytest.raises(ValueError):
            model(lucid.rand(1, 4, 3, 8, 9))
