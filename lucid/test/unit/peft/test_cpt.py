"""Context-aware prompt tuning: a prompt of token embeddings plus a bounded delta, and its own loss.

The pieces the method defines: the token types (modulo 4) decide which deltas train and how far
each may move, the delta is projected onto its ball before every use, and the loss covers the
output tokens of the prompt and of the batch, optionally decayed per example.  The loss and the
projection are checked against plain-ops rewrites of the formulas.
"""

import math
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.models import create_model
from lucid.peft import CPTConfig, CPTEmbedding, CPTModel, PeftModel, PeftType, get_peft_model
from lucid.peft import get_peft_model_state_dict, set_peft_model_state_dict
from lucid.peft.tuners.cpt._loss import _call_labels, _call_types, _cpt_loss

_LM: dict[str, object] = {
    "vocab_size": 32,
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 24,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
}
_IDS = [3, 4, 5, 6]
_TYPES = [1, 2, 3, 4]


def _lm(device: str = "cpu", factory: str = "gpt2_small_lm") -> nn.Module:
    lucid.manual_seed(0)
    return create_model(factory, **_LM).to(device)


def _config(**fields: object) -> CPTConfig:
    return CPTConfig.from_dict(
        {"cpt_token_ids": _IDS, "cpt_tokens_type_mask": _TYPES, "task_type": "CAUSAL_LM", **fields}
    )


def _tokens(device: str = "cpu") -> lucid.Tensor:
    lucid.manual_seed(1)
    return lucid.randint(0, 32, (2, 6), device=device)


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _delta(prompt: CPTEmbedding) -> lucid.Tensor:
    """The weight of a trainable prompt's delta."""
    delta = prompt.delta_embedding
    assert delta is not None
    return delta.weight


def _row_norms(weight: lucid.Tensor) -> list[float]:
    return [float(x) for x in (weight * weight).sum(dim=1).sqrt().tolist()]


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = CPTConfig(task_type="CAUSAL_LM")
        assert config.cpt_token_ids == [0] and config.cpt_mask == [1]
        assert config.cpt_tokens_type_mask == [1] and config.num_virtual_tokens == 1
        assert config.opt_weighted_loss_type == "none" and config.opt_loss_decay_factor == 1.0
        assert config.opt_projection_epsilon == 0.1 == config.opt_projection_format_epsilon
        assert config.num_transformer_submodules == 1 and config.peft_type == PeftType.CPT

    def test_the_virtual_token_count_follows_the_ids(self) -> None:
        config = _config(num_virtual_tokens=99)
        assert config.num_virtual_tokens == 4
        assert config.cpt_mask == [1, 1, 1, 1]

    def test_the_fields_are_the_references(self) -> None:
        assert set(_config().to_dict()) == {
            *("auto_mapping", "base_model_name_or_path", "inference_mode", "modules_to_save"),
            *("num_attention_heads", "num_layers", "num_transformer_submodules", "peft_type"),
            *("num_virtual_tokens", "revision", "task_type", "token_dim", "tokenizer_name_or_path"),
            *("cpt_mask", "cpt_token_ids", "cpt_tokens_type_mask", "opt_loss_decay_factor"),
            *("opt_projection_epsilon", "opt_projection_format_epsilon", "opt_weighted_loss_type"),
        }

    def test_a_file_round_trips(self) -> None:
        config = _config(opt_weighted_loss_type="decay", opt_loss_decay_factor=0.5)
        assert CPTConfig.from_dict(config.to_dict()) == config

    @pytest.mark.parametrize("task", [None, "SEQ_CLS", "SEQ_2_SEQ_LM"])
    def test_only_a_causal_language_model_is_served(self, task: str | None) -> None:
        with pytest.raises(ValueError, match="only supports task_type = CAUSAL_LM"):
            CPTConfig(task_type=task)

    def test_the_per_token_lists_must_agree(self) -> None:
        with pytest.raises(ValueError, match="must have the same length"):
            _config(cpt_mask=[1, 1])
        with pytest.raises(ValueError, match="must have the same length"):
            _config(cpt_tokens_type_mask=[1])

    def test_bad_values_are_refused(self) -> None:
        with pytest.raises(ValueError, match="one of 'none', 'decay'"):
            _config(opt_weighted_loss_type="linear")
        with pytest.raises(ValueError, match="negative"):
            _config(opt_projection_epsilon=-1.0)
        with pytest.raises(TypeError, match="int"):
            _config(cpt_token_ids=[1.5])


class TestEmbedding:
    def _embedding(self, **fields: object) -> tuple[CPTEmbedding, nn.Embedding]:
        lucid.manual_seed(3)
        table = nn.Embedding(32, 16)
        config = _config(token_dim=16, **fields)
        return CPTEmbedding(config, table), table

    def test_it_starts_as_the_token_embeddings(self) -> None:
        prompt, table = self._embedding()
        rows = prompt(lucid.arange(4).long())
        assert _same(rows, table(lucid.tensor(_IDS)))
        assert float(_delta(prompt).abs().max().item()) == 0.0

    def test_only_the_delta_trains(self) -> None:
        prompt, _ = self._embedding()
        assert {n for n, p in prompt.named_parameters() if p.requires_grad} == {
            "delta_embedding.weight"
        }
        prompt.requires_grad_(True)
        assert not prompt.embedding.weight.requires_grad

    def test_only_template_and_input_deltas_get_a_gradient(self) -> None:
        prompt, _ = self._embedding()
        prompt(lucid.arange(4).long()).sum().backward()
        grad = _delta(prompt).grad
        assert grad is not None
        live = [float(x) for x in grad.abs().sum(dim=1).tolist()]
        assert live[0] > 0 and live[1] > 0 and live[2] > 0 and live[3] == 0  # types 1, 2, 3 | 4

    def test_the_delta_is_projected_onto_its_ball_before_use(self) -> None:
        prompt, table = self._embedding(
            opt_projection_epsilon=0.4, opt_projection_format_epsilon=0.2
        )
        scale = math.sqrt(16 / 2048)
        with lucid.no_grad():
            _delta(prompt).copy_(lucid.full((4, 16), 1.0))
        rows = prompt(lucid.arange(4).long())
        norms = _row_norms(_delta(prompt))
        # type 1 and 3 (templates) 0.2, type 2 (input) 0.4, scaled; type 4 is projected to ~0.
        assert norms[0] == pytest.approx(0.2 * scale, rel=1e-4)
        assert norms[1] == pytest.approx(0.4 * scale, rel=1e-4)
        assert norms[2] == pytest.approx(0.2 * scale, rel=1e-4)
        assert norms[3] < 1e-8
        # The prompt is the embedding plus the (masked) projected delta.
        expected = table(lucid.tensor(_IDS)) + _delta(prompt) * prompt.update_mask
        assert bool(lucid.allclose(rows, expected, atol=1e-6))

    def test_a_row_inside_its_ball_is_left_alone(self) -> None:
        prompt, _ = self._embedding()
        small = lucid.full((4, 16), 1e-4)
        with lucid.no_grad():
            _delta(prompt).copy_(small)
        prompt.project()
        kept = _delta(prompt)
        assert bool(lucid.allclose(kept[:3], small[:3]))

    def test_an_inference_adapter_holds_the_prompt_alone(self) -> None:
        lucid.manual_seed(3)
        prompt = CPTEmbedding(_config(token_dim=16, inference_mode=True), nn.Embedding(32, 16))
        assert prompt.delta_embedding is None and prompt.stores_prompt
        assert [n for n, _ in prompt.named_parameters()] == ["embedding.weight"]

    def test_a_table_that_is_not_an_embedding_is_refused(self) -> None:
        with pytest.raises(TypeError, match="nn.Embedding"):
            CPTEmbedding(_config(token_dim=16), None)


class TestTypesAndLabels:
    def _config(self) -> CPTConfig:
        return _config(cpt_token_ids=[7, 8, 9], cpt_tokens_type_mask=[1, 2, 4])

    def test_the_input_types_are_shifted_past_the_prompts(self) -> None:
        given = lucid.tensor([[0, 2, 4], [4, 4, 0]])
        types = _call_types(self._config(), given, 2, 3, given)
        assert types.tolist() == [[1, 2, 4, 0, 6, 8], [1, 2, 4, 8, 8, 0]]
        assert given.tolist() == [[0, 2, 4], [4, 4, 0]]  # the caller's mask is not changed

    def test_every_input_token_is_an_output_by_default(self) -> None:
        labels = lucid.tensor([[1, 2, 3]])
        types = _call_types(self._config(), None, 1, 3, labels)
        assert types.tolist() == [[1, 2, 4, 8, 8, 8]]

    def test_a_mask_of_the_wrong_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match="input_type_mask"):
            _call_types(self._config(), lucid.ones(2, 2).long(), 2, 3, lucid.ones(2, 3).long())

    def test_only_output_positions_keep_their_label(self) -> None:
        labels = lucid.tensor([[10, 11, 12]])
        types = lucid.tensor([[1, 2, 4, 8, 0, 8]])
        kept = _call_labels(self._config(), labels, types)
        # prompt ids 7 8 9 | labels: the prompt's type-4 token and the input's type-8 tokens stay.
        assert kept.tolist() == [[-100, -100, 9, 10, -100, 12]]


class TestLoss:
    def _call(self) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
        lucid.manual_seed(4)
        logits = lucid.randn(2, 6, 11)
        labels = lucid.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
        types = lucid.tensor([[1, 2, 4, 4, 8, 8], [2, 4, 4, 8, 8, 12]])
        return logits, _call_labels(_config(), labels[:, :2], types), types

    def test_it_is_the_mean_cross_entropy_of_the_labelled_next_tokens(self) -> None:
        logits, labels, types = self._call()
        loss = _cpt_loss(logits, labels, types, _config())
        shifted = labels[:, 1:]
        mask = shifted != -100
        picked = F.cross_entropy(
            logits[:, :-1].reshape(-1, 11), shifted.reshape(-1), ignore_index=-100, reduction="none"
        ).reshape(shifted.shape)
        assert float(loss.item()) == pytest.approx(float(picked[mask].mean().item()), rel=1e-5)

    def test_the_decay_halves_each_earlier_example(self) -> None:
        logits, labels, types = self._call()
        loss = _cpt_loss(
            logits,
            labels,
            types,
            _config(opt_weighted_loss_type="decay", opt_loss_decay_factor=0.5),
        )
        shifted, kinds = labels[:, 1:], types[:, 1:].tolist()
        picked = F.cross_entropy(
            logits[:, :-1].reshape(-1, 11), shifted.reshape(-1), ignore_index=-100, reduction="none"
        ).reshape(shifted.shape)
        total, count = 0.0, 0
        for row in range(2):
            # The row's output types, last segment weighing one, the one before 0.5, and so on.
            segments = sorted({t for t in kinds[row] if t > 0 and t % 4 == 0}, reverse=True)
            for column, kind in enumerate(kinds[row]):
                if shifted[row, column].item() != -100:
                    total += float(picked[row, column].item()) * 0.5 ** segments.index(kind)
                    count += 1
        assert float(loss.item()) == pytest.approx(total / count, rel=1e-5)

    def test_a_call_without_an_output_token_is_refused(self) -> None:
        logits = lucid.randn(1, 4, 11)
        labels = lucid.full((1, 4), -100, dtype=lucid.int64)
        with pytest.raises(ValueError, match="no position of the call is an output token"):
            _cpt_loss(logits, labels, lucid.ones(1, 4).long(), _config())


class TestTuner:
    def test_the_prompt_goes_in_front_and_the_loss_is_cpts(self) -> None:
        peft = get_peft_model(_lm(), _config()).eval()
        ids = _tokens()
        out = peft(ids, labels=ids)
        assert tuple(out.logits.shape) == (2, 10, 32)  # 4 prompt positions + 6
        types = _call_types(_config(), None, 2, 6, ids)
        expected = _cpt_loss(out.logits, _call_labels(_config(), ids, types), types, _config())
        assert float(out.loss.item()) == pytest.approx(float(expected.item()), rel=1e-6)

    def test_a_call_without_labels_is_prompt_tuning(self) -> None:
        peft = get_peft_model(_lm(), _config()).eval()
        out = peft(_tokens())
        assert tuple(out.logits.shape) == (2, 10, 32) and out.loss is None

    def test_the_input_type_mask_chooses_the_labelled_tokens(self) -> None:
        peft = get_peft_model(_lm(), _config()).eval()
        ids = _tokens()
        everything = peft(ids, labels=ids).loss
        half = lucid.cat([lucid.zeros(2, 3), lucid.full((2, 3), 4.0)], dim=1).long()
        partial = peft(ids, labels=ids, input_type_mask=half).loss
        assert float(partial.item()) != float(everything.item())
        assert half.tolist()[0] == [0, 0, 0, 4, 4, 4]  # not changed in place

    def test_a_non_tensor_label_is_refused(self) -> None:
        peft = get_peft_model(_lm(), _config())
        with pytest.raises(TypeError, match="Tensors"):
            peft(_tokens(), labels=[[1, 2]])

    def test_a_disabled_adapter_is_the_base_with_its_own_loss(self) -> None:
        base, ids = _lm(), _tokens()
        expected = base(ids, labels=ids)
        peft = get_peft_model(base, _config()).eval()
        with peft.disable_adapter():
            out = peft(ids, labels=ids)
        assert _same(out.logits, expected.logits) and float(out.loss.item()) == float(
            expected.loss.item()
        )

    def test_the_delta_alone_trains_and_the_loss_falls(self, device: str) -> None:
        peft = get_peft_model(_lm(device), _config(opt_weighted_loss_type="decay")).train()
        ids = _tokens(device)
        trained = [n for n, p in peft.named_parameters() if p.requires_grad]
        assert trained == ["base_model.prompt_encoder.default.delta_embedding.weight"]
        before = {n: p.clone() for n, p in peft.named_parameters() if not p.requires_grad}
        opt = optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=2e-2)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            loss = peft(ids, labels=ids).loss
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        assert losses[-1] < losses[0], losses
        assert all(_same(p, before[n]) for n, p in peft.named_parameters() if n in before)
        encoder = peft.base_model.prompt_encoder["default"]
        assert max(_row_norms(encoder.delta_embedding.weight)) > 0

    def test_generate_carries_the_prompt(self) -> None:
        peft = get_peft_model(_lm(), _config()).eval()
        assert tuple(peft.generate(_tokens(), max_new_tokens=3, do_sample=False).shape) == (2, 9)

    def test_it_works_on_a_gpt_model_too(self) -> None:
        peft = get_peft_model(_lm(factory="gpt_lm"), _config()).eval()
        ids = _tokens()
        assert tuple(peft(ids, labels=ids).logits.shape) == (2, 10, 32)

    def test_a_compiled_call_follows_the_delta_and_its_projection(
        self, device_gpu_only: str
    ) -> None:
        device = device_gpu_only
        peft = get_peft_model(_lm(device), _config()).eval()
        ids = _tokens(device)
        delta = peft.base_model.prompt_encoder["default"].delta_embedding.weight
        with lucid.no_grad():
            compiled = lucid.compile(peft)
            for scale in (0.0, 0.003, 3.0):
                delta.copy_(lucid.randn(*delta.shape, device=device) * scale)
                assert bool(lucid.allclose(compiled(ids).logits, peft(ids).logits, atol=1e-6))

    def test_the_class_is_the_registered_tuner(self) -> None:
        assert isinstance(get_peft_model(_lm(), _config()).base_model, CPTModel)


class TestFile:
    def test_the_file_holds_the_finished_prompt(self) -> None:
        peft = get_peft_model(_lm(), _config())
        encoder = peft.base_model.prompt_encoder["default"]
        with lucid.no_grad():
            encoder.delta_embedding.weight.copy_(lucid.full((4, 16), 0.01))
        state = get_peft_model_state_dict(peft)
        assert list(state) == ["prompt_embeddings"]
        rows = encoder(lucid.arange(4).long())
        assert bool(lucid.allclose(state["prompt_embeddings"], rows, atol=1e-6))

    def test_a_saved_adapter_reloads_to_the_same_logits(self, tmp_path: Path) -> None:
        trained = get_peft_model(_lm(), _config()).eval()
        with lucid.no_grad():
            trained.base_model.prompt_encoder["default"].delta_embedding.weight.copy_(
                lucid.randn(4, 16) * 0.01
            )
        ids = _tokens()
        expected = trained(ids).logits
        trained.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_lm(), tmp_path)
        assert type(loaded).__name__ == "PeftModelForCausalLM"
        assert loaded.base_model.prompt_encoder["default"].delta_embedding is None
        assert bool(lucid.allclose(loaded(ids).logits, expected, atol=1e-5))

    def test_a_training_adapter_does_not_take_a_stored_prompt(self) -> None:
        state = get_peft_model_state_dict(get_peft_model(_lm(), _config()))
        other = get_peft_model(_lm(), _config())
        with pytest.raises(ValueError, match="built to train"):
            set_peft_model_state_dict(other, state)
