"""A key/value prefix is sized to the attention it is cached for, not to the hidden size.

Prefix tuning and Cartridge put keys and values in front of every attention layer, so their rows are
``heads x head_dim`` wide.  That equals the hidden size for most models, and does not for T5-3B and
T5-11B (a head width of 128 on a model width of 1024).  The tuner that builds the prompt reads the
sizes it was not given from the model's config, and for these methods ``token_dim`` has to come from
the attention.  The tests run the registered methods on a T5 whose heads and head width disagree with
its hidden size — forward, backward, save, load — hold the rule where a model declares its attention
(``head_dim``, ``num_key_value_heads``), and check the rule is stated for every method that injects keys
and values.  The reference library fails on this geometry (it reads ``token_dim`` as the model width);
its documented override, ``token_dim=heads x head_dim``, is the config Lucid derives, and the ``parity``
test holds the sizes to it.
"""

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import lucid
import lucid.nn as nn
from lucid.models import create_model
from lucid.peft import (
    PEFT_TYPE_TO_TUNER_MAPPING,
    CartridgeConfig,
    PeftConfig,
    PeftModel,
    PeftModelForSeq2SeqLM,
    PeftType,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptLearningConfig,
    get_peft_model,
)
from lucid.peft.tuners._prompt._encoder import _Injection
from lucid.peft.tuners._prompt._geometry import _fill_sizes, _fit_key_values
from lucid.peft.tuners._prompt._tuner import _PromptLearningTuner
from lucid.test.unit.peft._parity import adapter_library
from lucid.test.unit.peft.test_prompt_learning import _T5, _V, _ids
from lucid.utils.cache import DynamicCache, EncoderDecoderCache

#: T5-3B's shape in small: two heads of eight make 16, on a model width of 32.
_WIDE: dict[str, object] = {**_T5, "hidden_size": 32, "num_attention_heads": 2, "head_dim": 8}
_ATTENTION = 2 * 8

_KEY_VALUE_METHODS: dict[str, PromptLearningConfig] = {
    "prefix": PrefixTuningConfig(num_virtual_tokens=_V, task_type="SEQ_2_SEQ_LM"),
    "prefix-projection": PrefixTuningConfig(
        num_virtual_tokens=_V,
        task_type="SEQ_2_SEQ_LM",
        prefix_projection=True,
        encoder_hidden_size=8,
    ),
    "cartridge": CartridgeConfig(
        num_virtual_tokens=_V, num_frozen_tokens=1, task_type="SEQ_2_SEQ_LM"
    ),
}


def _wide_t5(device: str) -> nn.Module:
    lucid.manual_seed(0)
    return create_model("t5_small_seq2seq", **_WIDE).eval().to(device)


def _peft(method: str, device: str) -> PeftModelForSeq2SeqLM:
    peft = get_peft_model(_wide_t5(device), _KEY_VALUE_METHODS[method])
    assert isinstance(peft, PeftModelForSeq2SeqLM)
    return peft


def _model_and_config(**sizes: int) -> tuple[nn.Module, PromptLearningConfig]:
    """A model that only has a ``config`` with ``sizes``, and a prefix config that sets none of its own."""
    model = nn.Module()
    model.config = SimpleNamespace(num_hidden_layers=2, **sizes)
    return model, PrefixTuningConfig(num_virtual_tokens=_V)


def _fitted(config: PromptLearningConfig) -> tuple[int | None, int | None]:
    return config.token_dim, config.num_attention_heads


class TestTheMethodsThatInjectKeysAndValues:
    def test_every_method_that_does_is_run_below(self) -> None:
        """A new key/value method has to join ``_KEY_VALUE_METHODS``, or this fails."""
        injecting = {
            peft_type
            for peft_type, tuner in PEFT_TYPE_TO_TUNER_MAPPING.items()
            if isinstance(tuner, type)
            and issubclass(tuner, _PromptLearningTuner)
            and tuner._encoder_class.injection == _Injection.KEY_VALUES
        }
        assert injecting == {PeftType(config.peft_type) for config in _KEY_VALUE_METHODS.values()}


@pytest.mark.parametrize("method", list(_KEY_VALUE_METHODS))
class TestAWideT5:
    def test_the_prefix_is_as_wide_as_the_attention(self, method: str, device: str) -> None:
        config = _peft(method, device).active_peft_config
        assert isinstance(config, PromptLearningConfig)
        assert (config.token_dim, config.num_attention_heads) == (_ATTENTION, 2)

    def test_the_forward_and_the_backward_run(self, method: str, device: str) -> None:
        peft = _peft(method, device).train()
        src, labels = _ids(device, (2, 5)), _ids(device, (2, 4), seed=3)
        out = peft(src, labels=labels)
        assert tuple(out.logits.shape) == (2, 4, 50)
        out.loss.backward()
        trained = [name for name, p in peft.named_parameters() if p.requires_grad]
        assert trained and all("prompt_encoder" in name for name in trained)
        for name, parameter in peft.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name

    def test_it_generates(self, method: str, device: str) -> None:
        peft = _peft(method, device)
        out = peft.generate(input_ids=_ids(device, (2, 5)), max_length=6)
        assert out.shape[0] == 2

    def test_the_saved_adapter_loads_into_a_fresh_model(
        self, method: str, device: str, tmp_path: Path
    ) -> None:
        peft = _peft(method, device)
        peft.save_pretrained(tmp_path)
        assert json.loads((tmp_path / "adapter_config.json").read_text())["token_dim"] == _ATTENTION
        loaded = PeftModel.from_pretrained(_wide_t5(device), tmp_path)
        src, dec = _ids(device, (2, 5)), _ids(device, (2, 4), seed=3)
        want = peft(src, decoder_input_ids=dec).logits
        assert bool(lucid.allclose(loaded(src, decoder_input_ids=dec).logits, want, atol=1e-5))

    def test_the_prompt_the_model_is_given_has_the_models_head_width(
        self, method: str, device: str
    ) -> None:
        cache = _peft(method, device).get_prompt(batch_size=2)
        assert isinstance(cache, EncoderDecoderCache)
        own = cache.self_attention_cache
        assert isinstance(own, DynamicCache)
        assert tuple(own[0][0].shape) == (2, 2, _V, 8)


class TestOtherMethodsKeepTheHiddenSize:
    def test_a_prompt_in_front_of_the_embeddings_is_as_wide_as_the_embeddings(
        self, device: str
    ) -> None:
        peft = get_peft_model(
            _wide_t5(device), PromptTuningConfig(num_virtual_tokens=_V, task_type="SEQ_2_SEQ_LM")
        )
        config = peft.active_peft_config
        assert isinstance(config, PromptLearningConfig) and config.token_dim == 32
        out = peft(_ids(device, (2, 5)), decoder_input_ids=_ids(device, (2, 4), seed=3))
        assert tuple(out.logits.shape) == (2, 4, 50)


class TestWhereTheModelDeclaresItsAttention:
    def test_a_model_without_a_declared_head_width_is_sized_as_before(self) -> None:
        model, config = _model_and_config(hidden_size=16, num_attention_heads=2)
        _fit_key_values(model, config)
        assert _fitted(config) == (16, 2)

    def test_a_head_width_is_read_from_the_config(self) -> None:
        model, config = _model_and_config(hidden_size=32, num_attention_heads=2, head_dim=8)
        _fit_key_values(model, config)
        assert _fitted(config) == (16, 2)

    def test_grouped_heads_give_the_key_value_head_count_of_full_width_heads(self) -> None:
        model, config = _model_and_config(
            hidden_size=32, num_attention_heads=8, num_key_value_heads=2
        )
        _fit_key_values(model, config)
        assert _fitted(config) == (2 * 4, 2)

    def test_grouped_heads_with_a_declared_head_width(self) -> None:
        model, config = _model_and_config(
            hidden_size=32, num_attention_heads=8, num_key_value_heads=2, head_dim=16
        )
        _fit_key_values(model, config)
        assert _fitted(config) == (32, 2)

    def test_the_references_override_for_a_wide_head_is_accepted(self) -> None:
        model, config = _model_and_config(hidden_size=32, num_attention_heads=2, head_dim=8)
        config.token_dim = 16
        _fit_key_values(model, config)
        assert _fitted(config) == (16, 2)

    def test_a_token_dim_that_is_the_hidden_size_names_both_widths(self) -> None:
        model, config = _model_and_config(hidden_size=32, num_attention_heads=2, head_dim=8)
        config.token_dim = 32
        with pytest.raises(ValueError, match=r"2 heads of 8 \(16 wide\).*token_dim=32"):
            _fit_key_values(model, config)

    def test_a_head_count_that_is_not_the_models_is_refused(self) -> None:
        model, config = _model_and_config(
            hidden_size=32, num_attention_heads=8, num_key_value_heads=2
        )
        config.num_attention_heads = 8
        with pytest.raises(ValueError, match=r"2 heads of 4 \(8 wide\).*num_attention_heads=8"):
            _fit_key_values(model, config)

    def test_a_hidden_size_that_does_not_split_into_heads_is_refused(self) -> None:
        model, config = _model_and_config(hidden_size=30, num_attention_heads=4)
        with pytest.raises(ValueError, match="30 does not split into 4 attention heads"):
            _fit_key_values(model, config)

    def test_sizes_the_caller_gave_a_model_that_declares_nothing_are_kept(self) -> None:
        model, config = _model_and_config(hidden_size=32, num_attention_heads=4)
        config.token_dim, config.num_attention_heads = 12, 3
        _fit_key_values(model, config)
        assert _fitted(config) == (12, 3)

    def test_a_model_with_no_layer_count_is_asked_for_it(self) -> None:
        model = nn.Module()
        model.config = SimpleNamespace(hidden_size=16, num_attention_heads=2)
        with pytest.raises(ValueError, match="`num_layers`"):
            _fill_sizes(model, PrefixTuningConfig(num_virtual_tokens=_V), key_values=True)

    def test_a_prompt_in_front_of_the_embeddings_is_the_hidden_size_whatever_the_head_width(
        self,
    ) -> None:
        model, config = _model_and_config(hidden_size=32, num_attention_heads=2, head_dim=8)
        _fill_sizes(model, config, key_values=False)
        assert _fitted(config) == (32, 2)


# ── the reference library ────────────────────────────────────────────────────


def _reference_t5(ref: ModuleType) -> Any:
    transformers = pytest.importorskip("transformers")
    ref.manual_seed(0)
    config = transformers.T5Config(
        vocab_size=50,
        d_model=32,
        d_kv=8,
        num_heads=2,
        d_ff=32,
        num_layers=2,
        num_decoder_layers=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    return transformers.T5ForConditionalGeneration(config).eval()


@pytest.mark.parity
class TestAgainstTheReference:
    def test_the_reference_fails_on_this_geometry_and_works_with_the_sizes_lucid_derives(
        self, ref: ModuleType
    ) -> None:
        library = adapter_library()
        fields: dict[str, Any] = {"task_type": "SEQ_2_SEQ_LM", "num_virtual_tokens": _V}
        src = ref.randint(2, 50, (2, 5))
        dec = ref.randint(2, 50, (2, 4))
        naive = library.get_peft_model(_reference_t5(ref), library.PrefixTuningConfig(**fields))
        assert naive.peft_config["default"].token_dim == 32
        with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
            naive(input_ids=src, decoder_input_ids=dec)

        theirs = library.get_peft_model(
            _reference_t5(ref), library.PrefixTuningConfig(token_dim=_ATTENTION, **fields)
        )
        theirs(input_ids=src, decoder_input_ids=dec)
        ours = _peft("prefix", "cpu")
        mine, want = ours.active_peft_config, theirs.peft_config["default"]
        assert isinstance(mine, PrefixTuningConfig)
        sizes = ("token_dim", "num_attention_heads", "num_layers", "num_transformer_submodules")
        for field in sizes:
            assert getattr(mine, field) == getattr(want, field), field
        assert (
            ours.get_prompt_embedding_to_save("default").shape
            == theirs.get_prompt_embedding_to_save("default").shape
        )
        theirs_cache = theirs.get_prompt(batch_size=2).self_attention_cache
        ours_cache = ours.get_prompt(batch_size=2)
        assert isinstance(ours_cache, EncoderDecoderCache)
        own = ours_cache.self_attention_cache
        assert isinstance(own, DynamicCache)
        for layer in range(len(theirs_cache.layers)):
            assert tuple(own[layer][0].shape) == tuple(theirs_cache.layers[layer].keys.shape)
        assert PeftConfig.from_dict(mine.to_dict()).token_dim == want.token_dim
