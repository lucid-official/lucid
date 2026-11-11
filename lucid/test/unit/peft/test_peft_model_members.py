"""The reference's ``PeftModel`` members that were missing: what each reports, builds, or refuses.

``test_reference_signatures`` accounts for every reference member of every exported class (present with
the reference's signature, or owed).  These tests hold the members this issue added to what the reference
computes: the config of the adapter in use, the union of the ``modules_to_save`` names, the enabled flag
inside ``disable_adapter()``, the base model's dtype (spelled without the reference's framework name),
the prompt a prompt-learning adapter puts in front of a batch and the one it saves, and the two members
that have no meaning here (a model card, and a constructor-time gradient-checkpointing call).  The
``parity`` tests compare them with the reference library's on the same model.
"""

from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    CartridgeConfig,
    LoraConfig,
    MultitaskPromptTuningConfig,
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
    PrefixTuningConfig,
    get_peft_model,
)
from lucid.peft.tuners._prompt._tuner import _PromptLearningTuner
from lucid.peft.tuners.prefix_tuning import PrefixEncoder
from lucid.serialization import load_safetensors
from lucid.test.unit.peft._parity import adapter_library
from lucid.test.unit.peft.test_prompt_learning import (
    _ALL,
    _IN_FRONT,
    _METHODS,
    _V,
    _gpt2,
    _ids,
    _t5,
)
from lucid.utils.cache import DynamicCache, EncoderDecoderCache

_TASK_MODELS = [
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
]

_KEY_VALUE = ["prefix", "prefix-projection"]


def _net() -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2))


def _lora(modules_to_save: list[str] | None = None) -> LoraConfig:
    return LoraConfig(r=2, target_modules=["0"], modules_to_save=modules_to_save)


def _holder(peft: PeftModel) -> _PromptLearningTuner:
    host = peft.base_model
    assert isinstance(host, _PromptLearningTuner)
    return host


class TestTheConfigInUse:
    def test_active_peft_config_is_the_config_of_the_adapter_in_use(self) -> None:
        peft = get_peft_model(_net(), _lora())
        peft.add_adapter("wide", LoraConfig(r=4, target_modules=["0"]))
        assert peft.active_peft_config is peft.peft_config["default"]
        peft.set_adapter("wide")
        config = peft.active_peft_config
        assert isinstance(config, LoraConfig) and config.r == 4

    def test_it_names_the_problem_when_no_single_adapter_is_in_use(self) -> None:
        peft = get_peft_model(_net(), _lora())
        peft.delete_adapter("default")
        with pytest.raises(ValueError, match="no single adapter in use"):
            peft.active_peft_config

    def test_modules_to_save_is_the_union_over_every_adapter(self) -> None:
        peft = get_peft_model(_net(), _lora(modules_to_save=["2"]))
        assert peft.modules_to_save == {"2"}
        peft.add_adapter("other", _lora(modules_to_save=["1"]))
        assert peft.modules_to_save == {"1", "2"}

    def test_modules_to_save_is_none_when_no_adapter_trains_a_module_in_full(self) -> None:
        assert get_peft_model(_net(), _lora()).modules_to_save is None


class TestEnabledFlag:
    @pytest.mark.parametrize("method", ["lora", "prefix"])
    def test_it_is_false_exactly_inside_disable_adapter(self, method: str) -> None:
        if method == "lora":
            peft = get_peft_model(_net(), _lora())
        else:
            peft = get_peft_model(_gpt2("cpu"), _METHODS["prefix"]("CAUSAL_LM"))
        assert peft.has_active_enabled_adapter
        with peft.disable_adapter():
            assert not peft.has_active_enabled_adapter
            with peft.disable_adapter():  # blocks nest: the outermost exit re-enables
                assert not peft.has_active_enabled_adapter
            assert not peft.has_active_enabled_adapter
        assert peft.has_active_enabled_adapter


class TestBaseModelDtype:
    @pytest.mark.parametrize("model_class", _TASK_MODELS)
    def test_every_model_class_has_it_under_the_name_the_library_may_spell(
        self, model_class: type[PeftModel]
    ) -> None:
        assert isinstance(model_class.base_model_dtype, property)

    def test_it_is_the_dtype_of_the_first_floating_point_parameter(self) -> None:
        peft = get_peft_model(_net(), _lora())
        assert peft.base_model_dtype == lucid.float32
        half = get_peft_model(_net().half(), _lora())
        assert half.base_model_dtype == lucid.float16


class TestGradientCheckpointingPrep:
    def test_a_frozen_embedding_gets_an_output_that_requires_a_gradient(self, device: str) -> None:
        model = _gpt2(device)
        peft = get_peft_model(model, _METHODS["prompt"]("CAUSAL_LM"))
        ids = _ids(device, (2, 5))
        table = model.get_input_embeddings()
        assert not table(ids).requires_grad
        assert peft.prepare_model_for_gradient_checkpointing(model) is model
        assert table(ids).requires_grad

    def test_a_quantized_model_is_left_as_it_is(self, device: str) -> None:
        model = _gpt2(device)
        peft = get_peft_model(model, _METHODS["prompt"]("CAUSAL_LM"))
        model.is_loaded_in_8bit = True
        peft.prepare_model_for_gradient_checkpointing(model)
        assert not model.get_input_embeddings()(_ids(device, (2, 5))).requires_grad


class TestModelCard:
    def test_it_is_refused_and_writes_nothing(self, tmp_path: Path) -> None:
        peft = get_peft_model(_net(), _lora())
        with pytest.raises(NotImplementedError, match="model card"):
            peft.create_or_update_model_card(str(tmp_path))
        assert list(tmp_path.iterdir()) == []


class TestPromptRows:
    @pytest.mark.parametrize("method", _IN_FRONT)
    def test_the_rows_are_what_the_model_is_given_in_front_of_the_input(
        self, method: str, device: str
    ) -> None:
        base = _gpt2(device)
        peft = get_peft_model(_gpt2(device), _METHODS[method]("CAUSAL_LM")).eval()
        ids = _ids(device, (3, 5))
        rows = peft.get_prompt(batch_size=3)
        assert isinstance(rows, lucid.Tensor)
        assert tuple(rows.shape) == (3, _V, 16)
        embeds = lucid.cat([rows, base.get_input_embeddings()(ids)], dim=1)
        want = base.lm_head(base.transformer(inputs_embeds=embeds).last_hidden_state)
        assert bool(lucid.allclose(peft(ids).logits, want, atol=1e-5))

    def test_an_encoder_decoders_rows_hold_a_block_for_each_stack(self, device: str) -> None:
        peft = get_peft_model(_t5(device), _METHODS["prompt"]("SEQ_2_SEQ_LM"))
        rows = peft.get_prompt(batch_size=2)
        assert isinstance(rows, lucid.Tensor)
        assert tuple(rows.shape) == (2, 2 * _V, 16)

    def test_each_task_has_its_own_rows(self, device: str) -> None:
        config = MultitaskPromptTuningConfig(
            task_type="CAUSAL_LM", num_virtual_tokens=_V, num_tasks=3, num_ranks=2
        )
        peft = get_peft_model(_gpt2(device), config)
        with lucid.no_grad():
            encoder = _holder(peft).prompt_encoder["default"]
            for name in ("prefix_task_cols", "prefix_task_rows"):
                lucid.nn.init.normal_(getattr(encoder, name), std=0.5)
        tasks = lucid.tensor([0, 2, 1]).long().to(device)
        rows = peft.get_prompt(3, task_ids=tasks)
        assert isinstance(rows, lucid.Tensor)
        assert tuple(rows.shape) == (3, _V, 16)
        alone = peft.get_prompt(1, task_ids=tasks[1:2])
        assert isinstance(alone, lucid.Tensor)
        assert bool(lucid.allclose(rows[1:2], alone, atol=1e-6))
        with pytest.raises(ValueError, match="task_ids cannot be None"):
            peft.get_prompt(3)


class TestPromptCache:
    @pytest.mark.parametrize("method", _KEY_VALUE)
    def test_the_cache_is_the_prefix_a_decoder_starts_from(self, method: str, device: str) -> None:
        base = _gpt2(device)
        peft = get_peft_model(_gpt2(device), _METHODS[method]("CAUSAL_LM")).eval()
        cache = peft.get_prompt(batch_size=2)
        assert isinstance(cache, DynamicCache)
        assert len(cache) == 2 and cache.get_seq_length() == _V
        key, value = cache[0]
        assert tuple(key.shape) == tuple(value.shape) == (2, 2, _V, 8)
        ids = _ids(device, (2, 5))
        mask = lucid.ones(2, _V + 5, device=device).long()
        want = base(input_ids=ids, past_key_values=cache, attention_mask=mask).logits
        assert bool(lucid.allclose(peft(ids).logits, want, atol=1e-5))

    def test_an_encoder_decoders_cache_holds_the_prefix_on_the_decoders_self_attention(
        self, device: str
    ) -> None:
        peft = get_peft_model(_t5(device), _METHODS["prefix"]("SEQ_2_SEQ_LM"))
        cache = peft.get_prompt(batch_size=2)
        assert isinstance(cache, EncoderDecoderCache)
        own = cache.self_attention_cache
        assert isinstance(own, DynamicCache)
        assert own.get_seq_length() == _V and cache.cross_attention_cache.get_seq_length() == 0
        assert tuple(own[0][0].shape) == (2, 2, _V, 8)

    def test_a_cartridge_gives_its_whole_cache(self, device: str) -> None:
        config = CartridgeConfig(num_virtual_tokens=_V, num_frozen_tokens=1, task_type="CAUSAL_LM")
        peft = get_peft_model(_gpt2(device), config)
        cache = peft.get_prompt(batch_size=3)
        assert isinstance(cache, DynamicCache) and cache.get_seq_length() == _V
        assert tuple(cache[1][1].shape) == (3, 2, _V, 8)

    def test_the_cache_is_in_the_dtype_of_the_base_model(self, device: str) -> None:
        peft = get_peft_model(_gpt2(device).half(), _METHODS["prefix"]("CAUSAL_LM"))
        cache = peft.get_prompt(batch_size=1)
        assert isinstance(cache, DynamicCache)
        assert cache[0][0].dtype == peft.base_model_dtype == lucid.float16


class TestPromptRefusals:
    def test_a_weight_adapter_has_no_prompt(self) -> None:
        peft = get_peft_model(_net(), _lora())
        with pytest.raises(TypeError, match="only a prompt-learning adapter has a prompt"):
            peft.get_prompt(2)
        with pytest.raises(TypeError, match="only a prompt-learning adapter has a prompt"):
            peft.get_prompt_embedding_to_save("default")

    def test_a_fixed_size_cache_is_refused(self) -> None:
        peft = get_peft_model(_gpt2("cpu"), _METHODS["prefix"]("CAUSAL_LM"))
        with pytest.raises(NotImplementedError, match="max_cache_len"):
            peft.get_prompt(2, max_cache_len=16)

    def test_an_unknown_adapter_is_named(self) -> None:
        peft = get_peft_model(_gpt2("cpu"), _METHODS["prompt"]("CAUSAL_LM"))
        with pytest.raises(ValueError, match="Adapter nope not found"):
            peft.get_prompt_embedding_to_save("nope")

    def test_no_prompt_is_given_without_an_adapter_in_use(self) -> None:
        peft = get_peft_model(_gpt2("cpu"), _METHODS["prompt"]("CAUSAL_LM"))
        peft.delete_adapter("default")
        with pytest.raises(ValueError, match="no single adapter in use"):
            peft.get_prompt(2)


class TestPromptToSave:
    @pytest.mark.parametrize("method", _ALL)
    def test_it_is_what_save_pretrained_writes(
        self, method: str, device: str, tmp_path: Path
    ) -> None:
        peft = get_peft_model(_gpt2(device), _METHODS[method]("CAUSAL_LM"))
        saved = peft.get_prompt_embedding_to_save("default")
        assert saved.device.type == "cpu"
        peft.save_pretrained(tmp_path)
        written = load_safetensors(str(tmp_path / "adapter_model.safetensors"))["prompt_embeddings"]
        assert tuple(saved.shape) == tuple(written.shape)
        assert bool(lucid.allclose(saved, written, atol=1e-6))

    def test_a_cartridge_joins_its_frozen_rows_and_its_trained_ones(self, device: str) -> None:
        config = CartridgeConfig(num_virtual_tokens=_V, num_frozen_tokens=1, task_type="CAUSAL_LM")
        peft = get_peft_model(_gpt2(device), config)
        saved = peft.get_prompt_embedding_to_save("default")
        assert tuple(saved.shape) == (_V, 2 * 2 * 16)
        encoder = _holder(peft).prompt_encoder["default"]
        assert bool(lucid.allclose(saved, encoder.prompt_embeddings().cpu(), atol=0))

    def test_a_multitask_adapters_is_the_shared_prompt_its_tasks_scale(self, device: str) -> None:
        config = MultitaskPromptTuningConfig(
            task_type="CAUSAL_LM", num_virtual_tokens=_V, num_tasks=2, num_ranks=2
        )
        peft = get_peft_model(_gpt2(device), config)
        shared = _holder(peft).prompt_encoder["default"].embedding.weight
        assert bool(lucid.allclose(peft.get_prompt_embedding_to_save("default"), shared.cpu()))


class TestLoadPromptEmbeddings:
    """``PrefixEncoder.load_prompt_embeddings``: a flattened prefix into an unprojected encoder."""

    def _encoder(self, device: str, projection: bool = False) -> PrefixEncoder:
        config = PrefixTuningConfig(
            num_virtual_tokens=_V,
            task_type="CAUSAL_LM",
            prefix_projection=projection,
            encoder_hidden_size=8 if projection else None,
        )
        encoder = _holder(get_peft_model(_gpt2(device), config)).prompt_encoder["default"]
        assert isinstance(encoder, PrefixEncoder)
        return encoder

    def test_the_rows_are_copied_in(self, device: str) -> None:
        encoder = self._encoder(device)
        rows = lucid.arange(_V * 2 * 2 * 16).float().reshape(_V, 64) / 100.0
        encoder.load_prompt_embeddings(rows)
        assert bool(lucid.allclose(encoder.embedding.weight.cpu(), rows))
        assert encoder.embedding.weight.device.type == device

    def test_a_projection_cannot_take_them(self, device: str) -> None:
        encoder = self._encoder(device, projection=True)
        with pytest.raises(ValueError, match="prefix_projection=True"):
            encoder.load_prompt_embeddings(lucid.zeros(_V, 64))

    def test_a_wrong_shape_is_named_and_changes_nothing(self, device: str) -> None:
        encoder = self._encoder(device)
        before = encoder.embedding.weight.clone()
        with pytest.raises(ValueError, match=r"Expected \(3, 64\), got \(3, 8\)"):
            encoder.load_prompt_embeddings(lucid.zeros(_V, 8))
        assert bool(lucid.allclose(encoder.embedding.weight, before))


# ── the reference library ────────────────────────────────────────────────────

_SIZES: dict[str, int] = {
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "vocab_size": 50,
    "max_position_embeddings": 64,
}


def _reference_gpt(ref: ModuleType) -> Any:
    transformers = pytest.importorskip("transformers")
    ref.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=50, n_embd=16, n_layer=2, n_head=2, n_positions=64, n_inner=32
    )
    return transformers.GPT2LMHeadModel(config).eval()


@pytest.mark.parity
class TestAgainstTheReference:
    @pytest.mark.parametrize("method", ["prompt", "prefix"])
    def test_the_prompt_and_what_is_saved_are_the_references(
        self, method: str, ref: ModuleType
    ) -> None:
        library = adapter_library()
        reference = _reference_gpt(ref)
        theirs_config = (
            library.PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=_V)
            if method == "prompt"
            else library.PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=_V)
        )
        theirs = library.get_peft_model(reference, theirs_config).eval()
        ours = get_peft_model(_gpt2("cpu"), _METHODS[method]("CAUSAL_LM")).eval()
        source = theirs.prompt_encoder["default"].embedding.weight.detach().numpy()
        with lucid.no_grad():
            _holder(ours).prompt_encoder["default"].embedding.weight.copy_(lucid.tensor(source))
        assert np.array_equal(
            ours.get_prompt_embedding_to_save("default").numpy(),
            theirs.get_prompt_embedding_to_save("default").numpy(),
        )
        if method == "prompt":
            want = theirs.get_prompt(batch_size=3).detach().numpy()
            got = ours.get_prompt(batch_size=3)
            assert isinstance(got, lucid.Tensor)
            assert np.array_equal(got.numpy(), want)
            return
        want_cache = theirs.get_prompt(batch_size=3)
        got_cache = ours.get_prompt(batch_size=3)
        assert isinstance(got_cache, DynamicCache)
        for layer, layer_want in enumerate(want_cache.layers):
            key, value = got_cache[layer]
            assert np.array_equal(key.numpy(), layer_want.keys.detach().numpy())
            assert np.array_equal(value.numpy(), layer_want.values.detach().numpy())

    def test_the_flags_and_names_are_the_references(self, ref: ModuleType) -> None:
        library = adapter_library()
        theirs = library.get_peft_model(
            _reference_gpt(ref),
            library.PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=_V),
        )
        ours = get_peft_model(_gpt2("cpu"), _METHODS["prompt"]("CAUSAL_LM"))
        assert ours.has_active_enabled_adapter == theirs.has_active_enabled_adapter
        with theirs.disable_adapter(), ours.disable_adapter():
            assert ours.has_active_enabled_adapter == theirs.has_active_enabled_adapter
        assert ours.has_active_enabled_adapter == theirs.has_active_enabled_adapter
        assert ours.modules_to_save == theirs.modules_to_save is None
        assert ours.active_peft_config.peft_type == theirs.active_peft_config.peft_type

    def test_a_weight_adapters_names_and_flag_are_the_references(self, ref: ModuleType) -> None:
        library = adapter_library()
        net = ref.nn.Sequential(ref.nn.Linear(4, 4), ref.nn.Linear(4, 4), ref.nn.Linear(4, 2))
        theirs = library.get_peft_model(
            net, library.LoraConfig(r=2, target_modules=["0"], modules_to_save=["2"])
        )
        ours = get_peft_model(_net(), _lora(modules_to_save=["2"]))
        assert ours.modules_to_save == theirs.modules_to_save == {"2"}
        assert ours.has_active_enabled_adapter == theirs.has_active_enabled_adapter is True
        with theirs.disable_adapter(), ours.disable_adapter():
            assert ours.has_active_enabled_adapter == theirs.has_active_enabled_adapter is False
        assert ours.has_active_enabled_adapter == theirs.has_active_enabled_adapter is True
