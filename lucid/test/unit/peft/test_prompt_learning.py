"""Prompt tuning, prefix tuning and p-tuning on the text zoo (tiny configs, CPU and Metal).

The prompt is applied by hooks on the model's own call sites, so the checks are about what the model
sees and returns: the virtual positions in the output and the masks, only the prompt (and a head)
training, the prompt equal to hand-built inputs at a known initialisation, ``generate`` carrying the
prompt through every decoding step (cache or none), the adapter file in the reference's layout.
"""

import json
import warnings
from collections.abc import Callable
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.models import create_model
from lucid.peft import (
    PeftConfig,
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    get_layer_status,
    get_model_status,
    get_peft_model,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from lucid.peft.tuners._prompt._sites import _Applied, _put_prefix
from lucid.serialization import load_safetensors
from lucid.utils.cache import DynamicCache

_V = 3  # virtual tokens
_TEXT: dict[str, object] = {
    "vocab_size": 50,
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 64,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
}
_T5: dict[str, object] = {
    "vocab_size": 50,
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_decoder_layers": 2,
    "num_attention_heads": 2,
    "head_dim": 8,
    "intermediate_size": 32,
    "hidden_dropout": 0.0,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "decoder_start_token_id": 0,
}

_Make = Callable[[str], PeftConfig]
_METHODS: dict[str, Callable[[str | None], PeftConfig]] = {
    "prompt": lambda task: PromptTuningConfig(num_virtual_tokens=_V, task_type=task),
    "prefix": lambda task: PrefixTuningConfig(num_virtual_tokens=_V, task_type=task),
    "prefix-projection": lambda task: PrefixTuningConfig(
        num_virtual_tokens=_V, task_type=task, prefix_projection=True, encoder_hidden_size=8
    ),
    "p-tuning-mlp": lambda task: PromptEncoderConfig(
        num_virtual_tokens=_V, task_type=task, encoder_hidden_size=8
    ),
    "p-tuning-lstm": lambda task: PromptEncoderConfig(
        num_virtual_tokens=_V,
        task_type=task,
        encoder_hidden_size=8,
        encoder_reparameterization_type="LSTM",
    ),
}
_IN_FRONT = ["prompt", "p-tuning-mlp", "p-tuning-lstm"]
_ALL = list(_METHODS)


def _ids(device: str, shape: tuple[int, ...], seed: int = 0) -> lucid.Tensor:
    lucid.manual_seed(seed)
    return lucid.randint(2, 50, shape, device=device)


def _gpt2(device: str) -> nn.Module:
    lucid.manual_seed(0)
    return create_model("gpt2_small_lm", use_cache=False, **_TEXT).eval().to(device)


def _t5(device: str) -> nn.Module:
    lucid.manual_seed(0)
    return create_model("t5_small_seq2seq", **_T5).eval().to(device)


def _classifier(family: str, device: str) -> nn.Module:
    lucid.manual_seed(0)
    return create_model(family, num_labels=3, **_TEXT).eval().to(device)


def _assign(parameter: lucid.Tensor, value: lucid.Tensor) -> None:
    with lucid.no_grad():
        parameter.copy_(value)


def _prompt_parameters(peft: PeftModel) -> list[str]:
    return [n for n, p in peft.named_parameters() if p.requires_grad]


class TestInFront:
    @pytest.mark.parametrize("method", _ALL)
    def test_the_output_has_a_row_per_virtual_token_for_embeddings_only(
        self, method: str, device: str
    ) -> None:
        ids = _ids(device, (2, 5))
        peft = get_peft_model(_gpt2(device), _METHODS[method]("CAUSAL_LM"))
        rows = peft(ids).logits.shape[1]
        assert rows == 5 + (_V if method in _IN_FRONT else 0)
        assert type(peft) is PeftModelForCausalLM

    @pytest.mark.parametrize("method", _ALL)
    def test_the_attention_mask_is_widened_to_the_prompt(self, method: str, device: str) -> None:
        ids = _ids(device, (2, 5))
        peft = get_peft_model(_gpt2(device), _METHODS[method]("CAUSAL_LM"))
        plain = peft(ids).logits
        masked = peft(ids, attention_mask=lucid.ones(2, 5, device=device).long()).logits
        assert bool(lucid.allclose(plain, masked, atol=1e-5))
        # A padded key is not attended to: zeroing the first real token changes the result.
        padded = lucid.ones(2, 5, device=device).long()
        padded[:, 0] = 0
        assert not bool(lucid.allclose(peft(ids, attention_mask=padded).logits, plain, atol=1e-5))

    @pytest.mark.parametrize("method", _IN_FRONT)
    def test_labels_skip_the_virtual_positions(self, method: str, device: str) -> None:
        ids = _ids(device, (2, 5))
        peft = get_peft_model(_gpt2(device), _METHODS[method]("CAUSAL_LM"))
        out = peft(ids, labels=ids)
        # The first real label is predicted from the last virtual position.
        logits = out.logits[:, _V - 1 : -1, :]
        want = nn.functional.cross_entropy(logits.reshape(-1, 50), ids.reshape(-1).long())
        assert abs(float(out.loss.item()) - float(want.item())) < 1e-5

    def test_token_types_and_positions_follow_the_prompt(self, device: str) -> None:
        lucid.manual_seed(0)
        bert = create_model("bert_base", **_TEXT).eval().to(device)
        peft = get_peft_model(bert, _METHODS["prompt"]("FEATURE_EXTRACTION"))
        ids = _ids(device, (2, 5))
        types = lucid.zeros(2, 5, device=device).long()
        out = peft(ids, token_type_ids=types, attention_mask=lucid.ones(2, 5, device=device).long())
        assert out.last_hidden_state.shape == (2, 5 + _V, 16)
        with pytest.warns(UserWarning, match="Position ids"):
            peft(ids, position_ids=lucid.arange(5, device=device).long().reshape(1, 5))

    def test_a_prefix_hooks_the_trunk_alone_and_a_prompt_the_model_too(self, device: str) -> None:
        prefix = get_peft_model(_gpt2(device), _METHODS["prefix"]("CAUSAL_LM"))
        prompt = get_peft_model(_gpt2(device), _METHODS["prompt"]("CAUSAL_LM"))
        assert len(prefix.base_model._handles) == 1
        assert len(prompt.base_model._handles) == 2


class TestKnownInit:
    @pytest.mark.parametrize("method", ["prompt", "p-tuning-mlp"])
    def test_a_prompt_of_known_rows_is_the_same_as_input_ids_in_front(
        self, method: str, device: str
    ) -> None:
        base = _gpt2(device)
        ids = _ids(device, (2, 5))
        front = _ids(device, (_V,), seed=7)
        config = _METHODS[method]("CAUSAL_LM")
        config.inference_mode = True  # the encoder holds the prompt rows themselves
        peft = get_peft_model(_gpt2(device), config)
        rows = peft.get_base_model().transformer.wte(front)
        _assign(peft.prompt_encoder["default"].embedding.weight, rows)
        joined = lucid.cat([front.reshape(1, _V).repeat(2, 1), ids], dim=1)
        want = base(joined).logits[:, _V:]
        assert bool(lucid.allclose(peft(ids).logits[:, _V:], want, atol=1e-5))

    def test_text_initialisation_is_the_embedding_of_the_text(self, device: str) -> None:
        class Chars:
            def encode(self, text: str) -> list[int]:
                return [ord(c) % 50 for c in text]

        from lucid.peft import PromptEmbedding

        model = _gpt2(device)
        config = PromptTuningConfig(
            num_virtual_tokens=4,
            token_dim=16,
            num_transformer_submodules=1,
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text="ab",
            tokenizer_name_or_path="chars",
        )
        weight = PromptEmbedding(config, model.transformer.wte, tokenizer=Chars()).embedding.weight
        table = model.transformer.wte.weight
        ids = lucid.tensor([97 % 50, 98 % 50] * 2, device=device).long()
        assert bool(lucid.allclose(weight, table[ids]))

    def test_a_known_prefix_is_the_same_as_a_cache_that_holds_it(self, device: str) -> None:
        base = _gpt2(device)
        ids = _ids(device, (2, 5))
        peft = get_peft_model(_gpt2(device), _METHODS["prefix"]("CAUSAL_LM"))
        tuner = peft.base_model
        applied = tuner._applying()
        assert isinstance(applied, _Applied)
        cache = DynamicCache()
        _put_prefix(cache, applied.key_values(2, tuner._model_dtype()), _V)
        want = base(ids, past_key_values=cache).logits
        assert bool(lucid.allclose(peft(ids).logits, want, atol=1e-5))

    def test_a_zero_prefix_still_changes_the_output(self, device: str) -> None:
        # Zero keys score 0 against every query and zero values add nothing: not a no-op.
        base = _gpt2(device)
        ids = _ids(device, (1, 5))
        config = PrefixTuningConfig(num_virtual_tokens=_V, init_weights="zero")
        peft = get_peft_model(_gpt2(device), config)
        assert not bool(lucid.allclose(peft(ids).logits, base(ids).logits, atol=1e-6))


class TestTraining:
    @pytest.mark.parametrize("method", _ALL)
    def test_only_the_prompt_trains(self, method: str, device: str) -> None:
        ids = _ids(device, (2, 5))
        peft = get_peft_model(_gpt2(device), _METHODS[method]("CAUSAL_LM"))
        named = _prompt_parameters(peft)
        assert named and all(n.startswith("base_model.prompt_encoder.default.") for n in named)
        before = {n: p.clone() for n, p in peft.named_parameters()}
        optimizer = optim.SGD([p for p in peft.parameters() if p.requires_grad], lr=0.5)
        peft(ids, labels=ids).loss.backward()
        assert {n for n, p in peft.named_parameters() if p.grad is not None} == set(named)
        optimizer.step()
        for n, p in peft.named_parameters():
            moved = not bool(lucid.allclose(p, before[n]))
            assert moved == (n in named), n

    def test_the_trainable_share_is_the_prompt(self, device: str) -> None:
        peft = get_peft_model(_gpt2(device), _METHODS["prompt"]("CAUSAL_LM"))
        trainable, total = peft.get_nb_trainable_parameters()
        assert trainable == _V * 16 and total > trainable

    def test_a_loss_goes_down(self, device: str) -> None:
        ids = _ids(device, (4, 6))
        peft = get_peft_model(_gpt2(device), _METHODS["p-tuning-mlp"]("CAUSAL_LM"))
        optimizer = optim.Adam([p for p in peft.parameters() if p.requires_grad], lr=5e-2)
        losses = []
        for _ in range(12):
            optimizer.zero_grad()
            loss = peft(ids, labels=ids).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        assert losses[-1] < losses[0]

    @pytest.mark.parametrize("method", _ALL)
    def test_disabling_runs_the_base_and_unloading_restores_it(
        self, method: str, device: str
    ) -> None:
        base = _gpt2(device)
        ids = _ids(device, (2, 5))
        model = _gpt2(device)
        flags = [p.requires_grad for p in model.parameters()]
        peft = get_peft_model(model, _METHODS[method]("CAUSAL_LM"))
        with peft.disable_adapter():
            assert bool(lucid.allclose(peft(ids).logits, base(ids).logits, atol=1e-6))
        assert peft(ids).logits.shape != base(ids).logits.shape or method.startswith("prefix")
        assert peft.unload() is model
        assert bool(lucid.allclose(model(ids).logits, base(ids).logits, atol=1e-6))
        assert [p.requires_grad for p in model.parameters()] == flags
        assert not any(m._forward_pre_hooks for m in model.modules())
        assert "peft_config" not in model.__dict__

    def test_status_reports_the_method(self, device: str) -> None:
        peft = get_peft_model(_classifier("gpt2_small_cls", device), _METHODS["prompt"]("SEQ_CLS"))
        status = get_model_status(peft)
        assert status.adapter_model_type == "PromptTuningModel"
        assert status.peft_types == {"default": "PROMPT_TUNING"}
        assert [layer.module_type for layer in get_layer_status(peft)] == ["ModulesToSaveWrapper"]


class TestGenerate:
    @pytest.mark.parametrize("method", _ALL)
    def test_the_prompt_is_in_every_step_with_or_without_a_cache(
        self, method: str, device: str
    ) -> None:
        ids = _ids(device, (2, 4))
        peft = get_peft_model(_gpt2(device), _METHODS[method]("CAUSAL_LM"))
        cached = peft.generate(ids, max_new_tokens=5)
        plain = peft.generate(ids, max_new_tokens=5, use_cache=False)
        assert cached.shape == (2, 9)  # the prompt tokens are not part of the answer
        assert bool((cached == plain).all().item())
        assert bool((cached[:, :4] == ids).all().item())

    @pytest.mark.parametrize("method", _ALL)
    def test_the_prompt_changes_what_is_generated_through_the_hooks(
        self, method: str, device: str
    ) -> None:
        ids = _ids(device, (2, 4))
        peft = get_peft_model(_gpt2(device), _METHODS[method]("CAUSAL_LM"))
        for p in peft.base_model.prompt_encoder.parameters():
            _assign(p, p + 2.0)
        adapted = peft.generate(ids, max_new_tokens=6)
        with peft.disable_adapter():
            base = peft.generate(ids, max_new_tokens=6)
        assert not bool((adapted == base).all().item())

    @pytest.mark.parametrize("method", _ALL)
    def test_seq2seq_generate_carries_the_prompt(self, method: str, device: str) -> None:
        src = _ids(device, (2, 5))
        mask = lucid.ones(2, 5, device=device).long()
        peft = get_peft_model(_t5(device), _METHODS[method]("SEQ_2_SEQ_LM"))
        assert type(peft) is PeftModelForSeq2SeqLM
        cached = peft.generate(src, max_length=7, attention_mask=mask)
        plain = peft.generate(src, max_length=7, attention_mask=mask, use_cache=False)
        assert cached.shape == (2, 7) and bool((cached == plain).all().item())


class TestEncoderDecoder:
    @pytest.mark.parametrize("method", _ALL)
    def test_the_encoder_sees_the_prompt_and_the_labels_are_untouched(
        self, method: str, device: str
    ) -> None:
        src, labels = _ids(device, (2, 5)), _ids(device, (2, 4), seed=3)
        peft = get_peft_model(_t5(device), _METHODS[method]("SEQ_2_SEQ_LM"))
        out = peft(src, labels=labels, attention_mask=lucid.ones(2, 5, device=device).long())
        encoded = out.encoder_last_hidden_state.shape[1]
        assert encoded == 5 + (_V if method in _IN_FRONT else 0)
        assert out.logits.shape[:2] == (2, 4) and bool(lucid.isfinite(out.loss).item())

    def test_a_prompt_of_known_rows_is_the_encoder_input_in_front(self, device: str) -> None:
        base = _t5(device)
        src = _ids(device, (2, 5))
        config = PromptTuningConfig(num_virtual_tokens=_V, task_type="SEQ_2_SEQ_LM")
        config.inference_mode = True
        peft = get_peft_model(_t5(device), config)
        encoder = peft.prompt_encoder["default"]
        assert (
            encoder.num_embeddings == 2 * _V
        )  # one block of rows per stack, the encoder reads the first
        front = base.t5.shared(_ids(device, (_V,), seed=5))
        _assign(encoder.embedding.weight, lucid.cat([front, front], dim=0))
        joined = lucid.cat([front.reshape(1, _V, 16).repeat(2, 1, 1), base.t5.shared(src)], dim=1)
        want = base.t5.encoder(joined)
        got = peft(src, labels=src[:, :3]).encoder_last_hidden_state
        assert bool(lucid.allclose(got, want, atol=1e-5))

    def test_the_decoder_prefix_is_the_same_as_a_cache_that_holds_it(self, device: str) -> None:
        from lucid.utils.cache import EncoderDecoderCache

        base = _t5(device)
        src, labels = _ids(device, (2, 5)), _ids(device, (2, 4), seed=3)
        peft = get_peft_model(_t5(device), _METHODS["prefix"]("SEQ_2_SEQ_LM"))
        tuner = peft.base_model
        applied = tuner._applying()
        assert isinstance(applied, _Applied)
        cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        _put_prefix(cache, applied.key_values(2, tuner._model_dtype()), _V)
        decoder_in = base._shift_right(labels)
        memory = base.t5.encode(src)
        hidden = base.t5.decode(decoder_in, memory, past_key_value=cache, use_cache=True)
        want = base._logits(hidden)
        assert bool(lucid.allclose(peft(src, labels=labels).logits, want, atol=1e-5))


class TestFile:
    @pytest.mark.parametrize("method", _ALL)
    def test_a_saved_prompt_loads_to_the_same_model(
        self, method: str, device: str, tmp_path: Path
    ) -> None:
        ids = _ids(device, (2, 5))
        peft = get_peft_model(_gpt2(device), _METHODS[method]("CAUSAL_LM"))
        for p in peft.base_model.prompt_encoder.parameters():
            _assign(p, p + 0.5)
        peft.save_pretrained(tmp_path)
        saved = load_safetensors(str(tmp_path / "adapter_model.safetensors"))
        assert set(saved) == {"prompt_embeddings"}
        encoder = peft.base_model.prompt_encoder["default"]
        assert tuple(saved["prompt_embeddings"].shape) == tuple(encoder.prompt_embeddings().shape)
        assert bool(lucid.allclose(saved["prompt_embeddings"], encoder.prompt_embeddings().cpu()))
        config = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
        assert config["num_virtual_tokens"] == _V and config["token_dim"] == 16
        assert config["num_layers"] == 2 and config["inference_mode"] is True
        loaded = PeftModel.from_pretrained(_gpt2(device), tmp_path)
        assert type(loaded) is PeftModelForCausalLM
        assert bool(lucid.allclose(loaded(ids).logits, peft(ids).logits, atol=1e-5))
        assert loaded.peft_config["default"].inference_mode
        assert not _prompt_parameters(loaded)

    def test_a_trainable_load_is_refused(self, device: str, tmp_path: Path) -> None:
        peft = get_peft_model(_gpt2(device), _METHODS["prompt"]("CAUSAL_LM"))
        peft.save_pretrained(tmp_path)
        with pytest.raises(ValueError, match="Cannot set a prompt learning adapter to trainable"):
            PeftModel.from_pretrained(_gpt2(device), tmp_path, is_trainable=True)
        with pytest.raises(ValueError, match="Cannot set a prompt learning adapter to trainable"):
            peft.load_adapter(tmp_path, "again", is_trainable=True)

    def test_a_file_loads_into_a_second_adapter_slot(self, device: str, tmp_path: Path) -> None:
        peft = get_peft_model(_gpt2(device), _METHODS["prompt"]("CAUSAL_LM"))
        peft.save_pretrained(tmp_path)
        other = get_peft_model(_gpt2(device), _METHODS["prompt"]("CAUSAL_LM"))
        other.load_adapter(tmp_path, "from-file")
        other.set_adapter("from-file")
        ids = _ids(device, (2, 5))
        assert bool(lucid.allclose(other(ids).logits, peft(ids).logits, atol=1e-5))

    def test_a_training_encoder_does_not_take_a_stored_prompt(self, device: str) -> None:
        peft = get_peft_model(_gpt2(device), _METHODS["p-tuning-mlp"]("CAUSAL_LM"))
        stored = get_peft_model_state_dict(peft)
        with pytest.raises(ValueError, match="built to train"):
            set_peft_model_state_dict(peft, stored)

    def test_the_head_and_the_prompt_share_a_file_in_the_reference_layout(
        self, device: str, tmp_path: Path
    ) -> None:
        peft = get_peft_model(_classifier("gpt2_small_cls", device), _METHODS["prompt"]("SEQ_CLS"))
        peft.save_pretrained(tmp_path)
        saved = load_safetensors(str(tmp_path / "adapter_model.safetensors"))
        assert set(saved) == {"prompt_embeddings", "base_model.classifier.weight"}
        loaded = PeftModel.from_pretrained(_classifier("gpt2_small_cls", device), tmp_path)
        ids = _ids(device, (2, 5))
        assert bool(lucid.allclose(loaded(ids).logits, peft(ids).logits, atol=1e-5))


class TestSequenceClassification:
    @pytest.mark.parametrize("family", ["gpt2_small_cls", "gpt_cls", "bert_base_cls"])
    @pytest.mark.parametrize("method", ["prompt", "p-tuning-lstm", "prefix"])
    def test_the_head_trains_beside_the_prompt(self, family: str, method: str, device: str) -> None:
        if method == "prefix" and family == "bert_base_cls":
            with pytest.raises(NotImplementedError, match="cached keys and values"):
                get_peft_model(_classifier(family, device), _METHODS[method]("SEQ_CLS"))
            return
        ids = _ids(device, (2, 5))
        labels = lucid.tensor([0, 2], device=device).long()
        peft = get_peft_model(_classifier(family, device), _METHODS[method]("SEQ_CLS"))
        assert type(peft) is PeftModelForSequenceClassification
        assert peft.peft_config["default"].modules_to_save == ["classifier"]
        mask = lucid.ones(2, 5, device=device).long()
        out = peft(ids, attention_mask=mask, labels=labels)
        assert out.logits.shape == (2, 3)
        out.loss.backward()
        named = _prompt_parameters(peft)
        assert all("prompt_encoder" in n or "classifier.modules_to_save" in n for n in named)
        assert any("classifier.modules_to_save" in n for n in named)
        assert {n for n, p in peft.named_parameters() if p.grad is not None} == set(named)


class TestAdapters:
    def test_a_second_prompt_is_switched_to(self, device: str) -> None:
        ids = _ids(device, (2, 5))
        peft = get_peft_model(_gpt2(device), _METHODS["prompt"]("CAUSAL_LM"))
        first = peft(ids).logits
        peft.add_adapter("two", PromptTuningConfig(num_virtual_tokens=5, task_type="CAUSAL_LM"))
        assert peft(ids).logits.shape[1] == 5 + _V  # still the first
        assert not _prompt_parameters_of(peft, "two")
        peft.set_adapter("two")
        assert peft(ids).logits.shape[1] == 5 + 5
        assert _prompt_parameters_of(peft, "two") and not _prompt_parameters_of(peft, "default")
        peft.set_adapter("default")
        assert bool(lucid.allclose(peft(ids).logits, first))
        peft.delete_adapter("two")
        assert sorted(peft.peft_config) == ["default"]

    def test_deleting_the_active_prompt_hands_over(self, device: str) -> None:
        peft = get_peft_model(_gpt2(device), _METHODS["prompt"]("CAUSAL_LM"))
        peft.add_adapter("two", PromptTuningConfig(num_virtual_tokens=2, task_type="CAUSAL_LM"))
        with pytest.warns(UserWarning, match="Setting active adapter to two"):
            peft.delete_adapter("default")
        assert peft.active_adapter == "two" and _prompt_parameters(peft)
        peft.delete_adapter("two")
        assert peft.active_adapters == []
        ids = _ids(device, (1, 4))
        assert peft(ids).logits.shape[1] == 4  # nothing is applied

    def test_other_methods_do_not_mix(self, device: str) -> None:
        peft = get_peft_model(_gpt2(device), _METHODS["prompt"]("CAUSAL_LM"))
        with pytest.raises(ValueError, match="different peft types"):
            peft.add_adapter("two", _METHODS["prefix"]("CAUSAL_LM"))
        with pytest.raises(ValueError, match="already exists"):
            peft.add_adapter("default", _METHODS["prompt"]("CAUSAL_LM"))
        with pytest.raises(ValueError, match="one adapter at a time"):
            peft.set_adapter(["default", "default"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="not found"):
            peft.set_adapter("missing")
        with pytest.raises(ValueError, match="does not exist"):
            peft.delete_adapter("missing")

    def test_requires_grad_can_be_set_by_name(self, device: str) -> None:
        peft = get_peft_model(_gpt2(device), _METHODS["prompt"]("CAUSAL_LM"))
        peft.set_requires_grad("default", False)
        assert not _prompt_parameters(peft)
        peft.set_requires_grad(["default"], True)
        assert _prompt_parameters(peft)


def _prompt_parameters_of(peft: PeftModel, name: str) -> list[str]:
    return [
        n for n, p in peft.named_parameters() if p.requires_grad and f"prompt_encoder.{name}." in n
    ]


class TestForwarding:
    @pytest.mark.parametrize("method", _ALL)
    def test_a_name_the_tuner_lacks_is_read_from_the_model(self, method: str, device: str) -> None:
        peft = get_peft_model(_gpt2(device), _METHODS[method]("CAUSAL_LM"))
        tuner = peft.base_model
        assert tuner.config is peft.get_base_model().config
        assert callable(tuner.generate)
        with pytest.raises(AttributeError):
            tuner.no_such_attribute
        assert not hasattr(tuner, "no_such_attribute") and not hasattr(peft, "no_such_attribute")


class TestRefusals:
    def test_there_is_nothing_to_merge(self, device: str) -> None:
        peft = get_peft_model(_gpt2(device), _METHODS["prompt"]("CAUSAL_LM"))
        with pytest.raises(NotImplementedError, match="nothing to merge"):
            peft.merge_and_unload()

    def test_a_batch_cannot_mix_prompts(self, device: str) -> None:
        peft = get_peft_model(_gpt2(device), _METHODS["prompt"]("CAUSAL_LM"))
        with pytest.raises(NotImplementedError, match="different prompts"):
            peft(_ids(device, (2, 4)), adapter_names=["default", "default"])

    def test_a_bare_model_has_no_place_for_a_prompt(self, device: str) -> None:
        with pytest.raises(TypeError, match="get_peft_model"):
            inject_adapter_in_model(_METHODS["prompt"]("CAUSAL_LM"), _gpt2(device))

    def test_a_model_holding_adapters_is_not_adapted_again(self, device: str) -> None:
        model = _gpt2(device)
        get_peft_model(model, _METHODS["prompt"]("CAUSAL_LM"))
        with pytest.raises(ValueError, match="already holds adapters"):
            get_peft_model(model, _METHODS["prefix"]("CAUSAL_LM"))

    @pytest.mark.parametrize("task", ["TOKEN_CLS", "QUESTION_ANS"])
    def test_tasks_with_a_row_per_position_are_refused(self, task: str, device: str) -> None:
        with pytest.raises(NotImplementedError, match="token classification"):
            get_peft_model(_gpt2(device), _METHODS["prompt"](task))

    def test_the_length_of_the_prompt_must_be_given(self, device: str) -> None:
        with pytest.raises(ValueError, match="num_virtual_tokens"):
            get_peft_model(_gpt2(device), PromptTuningConfig(task_type="CAUSAL_LM"))

    def test_a_model_that_cannot_take_embeddings_is_named(self) -> None:
        net = nn.Sequential(nn.Embedding(5, 4), nn.Linear(4, 4))
        net.config = type("C", (), {"hidden_size": 4, "num_hidden_layers": 1, "num_attention_heads": 1})()  # type: ignore[assignment]
        with pytest.raises(NotImplementedError, match="inputs_embeds"):
            get_peft_model(net, PromptTuningConfig(num_virtual_tokens=2))

    def test_sizes_the_model_does_not_state_must_be_given(self) -> None:
        with pytest.raises(ValueError, match="`token_dim`"):
            get_peft_model(nn.Sequential(nn.Linear(2, 2)), PromptTuningConfig(num_virtual_tokens=2))

    def test_an_encoder_decoder_must_hand_out_its_stacks(self, device: str) -> None:
        with pytest.raises(NotImplementedError, match="get_encoder"):
            get_peft_model(_gpt2(device), _METHODS["prompt"]("SEQ_2_SEQ_LM"))

    def test_a_config_of_another_method_is_a_type_error(self, device: str) -> None:
        from lucid.peft.tuners.prompt_tuning import PromptTuningModel

        with pytest.raises(TypeError, match="PromptTuningConfig"):
            PromptTuningModel(_gpt2(device), _METHODS["prefix"]("CAUSAL_LM"))
        with pytest.raises(TypeError, match="nn.Module"):
            PromptTuningModel(object(), _METHODS["prompt"]("CAUSAL_LM"))  # type: ignore[arg-type]

    def test_a_call_with_neither_ids_nor_embeddings_is_refused(self, device: str) -> None:
        peft = get_peft_model(_gpt2(device), _METHODS["prompt"]("CAUSAL_LM"))
        with pytest.raises(ValueError, match="input_ids or inputs_embeds"):
            peft.get_base_model().transformer()

    def test_embeddings_can_be_given_instead_of_ids(self, device: str) -> None:
        base = _gpt2(device)
        peft = get_peft_model(_gpt2(device), _METHODS["prompt"]("CAUSAL_LM"))
        ids = _ids(device, (2, 5))
        embeds = peft.get_base_model().transformer.wte(ids)
        got = peft.get_base_model().transformer(inputs_embeds=embeds).last_hidden_state
        assert got.shape == (2, 5 + _V, 16)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            base.transformer(inputs_embeds=embeds)


class TestModelHooks:
    def test_a_trunk_takes_ids_or_embeddings_not_both(self, device: str) -> None:
        model = _gpt2(device)
        ids = _ids(device, (2, 5))
        embeds = model.transformer.wte(ids)
        with_ids = model.transformer(ids).last_hidden_state
        with_embeds = model.transformer(inputs_embeds=embeds).last_hidden_state
        assert bool(lucid.allclose(with_ids, with_embeds, atol=1e-6))
        with pytest.raises(ValueError, match="exactly one"):
            model.transformer(ids, inputs_embeds=embeds)
        with pytest.raises(ValueError, match="exactly one"):
            model.transformer()

    @pytest.mark.parametrize("family", ["gpt_cls"])
    def test_the_other_decoder_family_agrees(self, family: str, device: str) -> None:
        model = _classifier(family, device)
        ids = _ids(device, (2, 5))
        embeds = model.transformer.tokens_embed(ids)
        a = model.transformer(ids).last_hidden_state
        b = model.transformer(inputs_embeds=embeds).last_hidden_state
        assert bool(lucid.allclose(a, b, atol=1e-6))

    def test_roformer_takes_embeddings(self, device: str) -> None:
        from lucid.models.text.roformer import RoFormerConfig, RoFormerModel

        lucid.manual_seed(0)
        config = RoFormerConfig(
            vocab_size=50, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=32, max_position_embeddings=32,
        )  # fmt: skip
        model = RoFormerModel(config).eval().to(device)
        ids = _ids(device, (2, 5))
        embeds = model.embeddings.word_embeddings(ids)
        a = model(ids).last_hidden_state
        b = model(inputs_embeds=embeds).last_hidden_state
        assert bool(lucid.allclose(a, b, atol=1e-6))
        with pytest.raises(ValueError, match="exactly one"):
            model(ids, inputs_embeds=embeds)

    def test_roformer_prompt_tuning(self, device: str) -> None:
        from lucid.models.text.roformer import RoFormerConfig, RoFormerForSequenceClassification

        lucid.manual_seed(0)
        config = RoFormerConfig(
            vocab_size=50, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=32, max_position_embeddings=32, num_labels=3,
        )  # fmt: skip
        peft = get_peft_model(
            RoFormerForSequenceClassification(config).eval().to(device),
            _METHODS["prompt"]("SEQ_CLS"),
        )
        assert peft(_ids(device, (2, 5))).logits.shape == (2, 3)

    def test_the_encoder_decoder_hands_out_its_stacks(self, device: str) -> None:
        model = _t5(device)
        assert model.get_encoder() is model.t5.encoder is model.t5.get_encoder()
        assert model.get_decoder() is model.t5.decoder is model.t5.get_decoder()
