"""Cartridge: the trained cache, its frozen rows, the file, and the four utilities that start and join caches.

A Cartridge is the keys and values a model would hold for a document, as trainable rows, put in front of
the model's own cache.  The tests hold it to that: the rows are laid out as prefix tuning's are (the same
rows in a prefix adapter give the same logits), the first ``num_frozen_tokens`` never train, the adapter
file is one ``prompt_embeddings`` matrix that splits into the frozen and the trained rows on load and
joins again on save, and the utilities pack a cache (a layout written out here with plain loops), read a
text through the base model with the adapter off, and lay saved caches end to end.  The ``parity`` tests
compare the encoder, the packing and the composition with the reference library's.
"""

import json
from pathlib import Path
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.models import create_model
from lucid.peft import (
    CartridgeConfig,
    CartridgeEncoder,
    PeftModel,
    PeftModelForCausalLM,
    PrefixTuningConfig,
    compose_cartridge_adapters,
    get_peft_model,
    get_peft_model_state_dict,
    initialize_kv_prefix_from_past_key_values,
    initialize_kv_prefix_from_text,
    prompt_embeddings_from_past_key_values,
)
from lucid.peft.tuners.cartridge import CartridgeModel
from lucid.serialization import load_safetensors, save_safetensors
from lucid.test.unit.peft._parity import adapter_library
from lucid.test.unit.peft._tuner_recipes import same, span
from lucid.utils.cache import DynamicCache

_V, _FROZEN = 4, 1
_SIZES = {
    "vocab_size": 50,
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 64,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
}
#: The width of a row: ``num_layers * 2 * token_dim`` of that model.
_WIDTH = 2 * 2 * 16


def _lm(device: str = "cpu") -> nn.Module:
    lucid.manual_seed(0)
    return create_model("gpt2_small_lm", **_SIZES).eval().to(device)


def _config(**kw: object) -> CartridgeConfig:
    fields: dict[str, object] = {
        "num_virtual_tokens": _V,
        "num_frozen_tokens": _FROZEN,
        "task_type": "CAUSAL_LM",
    } | kw
    return CartridgeConfig(**fields)  # type: ignore[arg-type]


def _cartridge(device: str = "cpu", **kw: object) -> PeftModelForCausalLM:
    peft = get_peft_model(_lm(device), _config(**kw))
    assert isinstance(peft, PeftModelForCausalLM)
    return peft


def _encoder(peft: PeftModel, name: str = "default") -> CartridgeEncoder:
    encoder = peft.base_model.prompt_encoder[name]  # type: ignore[attr-defined]
    assert isinstance(encoder, CartridgeEncoder)
    return encoder


def _ids(device: str = "cpu") -> lucid.Tensor:
    lucid.manual_seed(1)
    return lucid.randint(2, 50, (2, 5), device=device)


def _rows(seed: int = 4, count: int = _V) -> lucid.Tensor:
    lucid.manual_seed(seed)
    return lucid.randn(count, _WIDTH)


def _legacy(
    layers: int = 2, heads: int = 2, tokens: int = 6, dim: int = 3, seed: int = 0
) -> tuple[tuple[lucid.Tensor, lucid.Tensor], ...]:
    """A cache of one sequence: ``layers`` pairs of ``(1, heads, tokens, dim)`` tensors."""
    lucid.manual_seed(seed)
    return tuple(
        (lucid.randn(1, heads, tokens, dim), lucid.randn(1, heads, tokens, dim))
        for _ in range(layers)
    )


class TestEncoder:
    def _encoder(self, **kw: object) -> CartridgeEncoder:
        fields: dict[str, object] = {"num_virtual_tokens": 5, "num_layers": 3, "token_dim": 4} | kw
        return CartridgeEncoder(CartridgeConfig(**fields))  # type: ignore[arg-type]

    def test_the_rows_are_frozen_then_trained_each_of_the_cache_width(self) -> None:
        encoder = self._encoder(num_frozen_tokens=2)
        assert encoder.frozen_embedding is not None
        assert tuple(encoder.frozen_embedding.shape) == (2, 24)
        assert tuple(encoder.trainable_embedding.shape) == (3, 24)
        assert tuple(encoder.weight.shape) == (5, 24) and encoder.num_embeddings == 5
        assert not encoder.frozen_embedding.requires_grad
        assert encoder.trainable_embedding.requires_grad

    def test_with_no_frozen_token_there_is_no_frozen_parameter(self) -> None:
        encoder = self._encoder(num_frozen_tokens=0)
        assert encoder.frozen_embedding is None
        assert sorted(encoder.state_dict()) == ["trainable_embedding"]

    def test_everything_frozen_leaves_no_trained_row(self) -> None:
        encoder = self._encoder(num_frozen_tokens=5)
        assert tuple(encoder.trainable_embedding.shape) == (0, 24)
        assert tuple(encoder(lucid.zeros(1, 5).long()).shape) == (1, 5, 24)

    def test_inference_mode_trains_nothing(self) -> None:
        encoder = self._encoder(inference_mode=True)
        assert not any(p.requires_grad for p in encoder.parameters())

    def test_the_rows_start_as_a_standard_normal_draw(self) -> None:
        lucid.manual_seed(0)
        encoder = CartridgeEncoder(
            CartridgeConfig(
                num_virtual_tokens=400, num_frozen_tokens=100, num_layers=2, token_dim=8
            )
        )
        weight = encoder.weight.detach()
        assert abs(float(weight.mean().item())) < 0.05
        assert abs(float(weight.std().item()) - 1.0) < 0.05

    def test_a_call_gives_the_whole_cache_to_every_row_of_the_batch(self) -> None:
        encoder = self._encoder()
        out = encoder(lucid.zeros(3, 7).long())  # the ids are not read
        assert tuple(out.shape) == (3, 5, 24)
        for row in range(3):
            assert same(out[row], encoder.weight)

    def test_turning_the_encoder_on_trains_the_trained_rows_only(self) -> None:
        encoder = self._encoder(num_frozen_tokens=2)
        encoder.requires_grad_(False)
        assert not any(p.requires_grad for p in encoder.parameters())
        encoder.requires_grad_(True)
        assert encoder.frozen_embedding is not None
        assert not encoder.frozen_embedding.requires_grad
        assert encoder.trainable_embedding.requires_grad

    def test_loading_a_cache_fills_the_frozen_rows_first(self) -> None:
        encoder = self._encoder(num_frozen_tokens=2)
        cache = lucid.randn(5, 24)
        encoder.load_prompt_embeddings(cache)
        assert encoder.frozen_embedding is not None
        assert same(encoder.frozen_embedding, cache[:2])
        assert same(encoder.trainable_embedding, cache[2:])
        assert same(encoder.weight, cache)

    @pytest.mark.parametrize("shape", [(4, 24), (5, 23), (24,), (1, 5, 24)])
    def test_a_cache_of_another_shape_is_refused_and_changes_nothing(
        self, shape: tuple[int, ...]
    ) -> None:
        encoder = self._encoder()
        before = encoder.weight.clone()
        with pytest.raises(ValueError, match="Invalid `prompt_embeddings` shape"):
            encoder.load_prompt_embeddings(lucid.zeros(*shape))
        assert same(encoder.weight, before)

    def test_resetting_draws_again(self) -> None:
        encoder = self._encoder()
        before = encoder.weight.clone()
        encoder.reset_parameters()
        assert span(encoder.weight, before) > 0.1

    def test_a_frozen_count_outside_the_range_is_refused(self) -> None:
        config = CartridgeConfig(num_virtual_tokens=3, num_layers=1, token_dim=2)
        config.num_frozen_tokens = 4  # past the config's own check
        with pytest.raises(ValueError, match="num_frozen_tokens"):
            CartridgeEncoder(config)


class TestTuner:
    def test_the_adapter_is_a_cartridge_encoder_in_a_prompt_tuner(self) -> None:
        peft = _cartridge()
        assert isinstance(peft.base_model, CartridgeModel)
        encoder = _encoder(peft)
        assert tuple(encoder.weight.shape) == (_V, _WIDTH)
        # The sizes the config leaves out are read off the model.
        config = peft.peft_config["default"]
        assert (config.num_layers, config.token_dim, config.num_attention_heads) == (2, 16, 2)  # type: ignore[attr-defined]

    def test_only_the_trained_rows_train_and_only_they_get_a_gradient(self, device: str) -> None:
        peft = _cartridge(device)
        trainable = [n for n, p in peft.named_parameters() if p.requires_grad]
        assert trainable == ["base_model.prompt_encoder.default.trainable_embedding"]
        peft(_ids(device)).logits.sum().backward()
        encoder = _encoder(peft)
        assert encoder.trainable_embedding.grad is not None
        assert float(encoder.trainable_embedding.grad.abs().max().item()) > 0
        assert encoder.frozen_embedding is not None and encoder.frozen_embedding.grad is None
        assert all(p.grad is None for p in peft.base_model.model.parameters())  # type: ignore[attr-defined]

    def test_the_cache_changes_the_logits_and_disabling_it_gives_the_base_back(
        self, device: str
    ) -> None:
        base = _lm(device)
        ids = _ids(device)
        plain = base(input_ids=ids).logits
        peft = get_peft_model(_lm(device), _config())
        assert span(peft(input_ids=ids).logits, plain) > 1e-3
        with peft.disable_adapter():
            assert span(peft(input_ids=ids).logits, plain) <= 1e-6

    def test_the_cache_is_laid_out_as_prefix_tunings_is(self, device: str) -> None:
        """The same rows in a prefix adapter and in a Cartridge give the same logits."""
        rows = _rows().to(device)
        ids = _ids(device)
        cartridge = get_peft_model(_lm(device), _config())
        _encoder(cartridge).load_prompt_embeddings(rows)
        prefix = get_peft_model(
            _lm(device), PrefixTuningConfig(num_virtual_tokens=_V, task_type="CAUSAL_LM")
        )
        with lucid.no_grad():
            prefix.base_model.prompt_encoder["default"].embedding.weight.copy_(rows)  # type: ignore[attr-defined]
            assert span(cartridge(input_ids=ids).logits, prefix(input_ids=ids).logits) <= 1e-5

    def test_a_cartridge_runs_through_generate_with_the_cache_in_front(self) -> None:
        peft = _cartridge()
        out = peft.generate(input_ids=_ids()[:, :3], max_new_tokens=3, do_sample=False)
        assert tuple(out.shape) == (2, 6)

    def test_a_second_cartridge_is_added_frozen_and_inactive_and_only_one_applies(self) -> None:
        peft = _cartridge()
        peft.add_adapter("b", _config(num_frozen_tokens=0))
        assert peft.active_adapter == "default"
        assert not any(p.requires_grad for p in _encoder(peft, "b").parameters())
        with pytest.raises(ValueError, match="one adapter at a time"):
            peft.set_adapter(["default", "b"])
        peft.set_adapter("b")
        assert _encoder(peft, "b").trainable_embedding.requires_grad
        assert _encoder(peft, "b").frozen_embedding is None

    def test_a_config_with_no_virtual_tokens_is_refused(self) -> None:
        with pytest.raises(ValueError, match="num_virtual_tokens"):
            get_peft_model(_lm(), CartridgeConfig(task_type="CAUSAL_LM"))

    def test_merging_is_refused(self) -> None:
        peft = _cartridge()
        with pytest.raises(NotImplementedError, match="nothing to merge"):
            peft.merge_and_unload()

    def test_a_trained_cartridge_unloads_to_the_plain_model(self) -> None:
        net = _lm()
        peft = get_peft_model(net, _config())
        assert peft.unload() is net
        assert "peft_config" not in net.__dict__


class TestFile:
    def test_the_file_holds_the_whole_cache_as_one_matrix(self) -> None:
        peft = _cartridge()
        state = get_peft_model_state_dict(peft)
        assert {k: tuple(v.shape) for k, v in state.items()} == {"prompt_embeddings": (_V, _WIDTH)}
        assert same(state["prompt_embeddings"], _encoder(peft).weight)

    def test_a_save_and_load_round_trip_reproduces_the_cache_and_the_logits(
        self, tmp_path: Path, device: str
    ) -> None:
        peft = _cartridge(device)
        _encoder(peft).load_prompt_embeddings(_rows().to(device))
        ids = _ids(device)
        with lucid.no_grad():
            expected = peft(input_ids=ids).logits
        peft.save_pretrained(tmp_path)
        saved = load_safetensors(str(tmp_path / "adapter_model.safetensors"))
        assert sorted(saved) == ["prompt_embeddings"]
        loaded = PeftModel.from_pretrained(_lm(device), tmp_path)
        encoder = _encoder(loaded)
        assert encoder.frozen_embedding is not None and encoder.num_frozen_tokens == _FROZEN
        assert same(encoder.weight, _encoder(peft).weight)
        with lucid.no_grad():
            assert span(loaded(input_ids=ids).logits, expected) <= 1e-6
        # A loaded adapter is for inference: nothing in it trains.
        assert not any(p.requires_grad for p in encoder.parameters())

    def test_a_file_written_in_the_references_layout_loads_into_the_two_parameters(
        self, tmp_path: Path
    ) -> None:
        rows = _rows(seed=9)
        save_safetensors(
            {"prompt_embeddings": rows},
            str(tmp_path / "adapter_model.safetensors"),
            metadata={"format": "pt"},
        )
        (tmp_path / "adapter_config.json").write_text(json.dumps(_REFERENCE_CONFIG))
        loaded = PeftModel.from_pretrained(_lm(), tmp_path)
        encoder = _encoder(loaded)
        assert encoder.frozen_embedding is not None
        assert same(encoder.frozen_embedding, rows[:1]) and same(
            encoder.trainable_embedding, rows[1:]
        )
        assert loaded.peft_config["default"].num_frozen_tokens == 1  # type: ignore[attr-defined]

    def test_a_file_of_another_size_is_refused_and_the_model_is_left_as_it_was(
        self, tmp_path: Path
    ) -> None:
        save_safetensors(
            {"prompt_embeddings": _rows(count=_V + 1)}, str(tmp_path / "adapter_model.safetensors")
        )
        (tmp_path / "adapter_config.json").write_text(json.dumps(_REFERENCE_CONFIG))
        net = _lm()
        with pytest.raises(ValueError, match="Invalid `prompt_embeddings` shape"):
            PeftModel.from_pretrained(net, tmp_path)
        assert "peft_config" not in net.__dict__

    def test_a_saved_cartridge_cannot_be_loaded_to_train(self, tmp_path: Path) -> None:
        _cartridge().save_pretrained(tmp_path)
        with pytest.raises(ValueError, match="prompt learning adapter to trainable"):
            PeftModel.from_pretrained(_lm(), tmp_path, is_trainable=True)

    def test_a_cartridge_with_no_frozen_token_round_trips_too(self, tmp_path: Path) -> None:
        peft = _cartridge(num_frozen_tokens=0)
        peft.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_lm(), tmp_path)
        assert _encoder(loaded).frozen_embedding is None
        assert same(_encoder(loaded).weight, _encoder(peft).weight)


_REFERENCE_CONFIG = {
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "inference_mode": True,
    "modules_to_save": None,
    "num_attention_heads": 2,
    "num_frozen_tokens": 1,
    "num_layers": 2,
    "num_transformer_submodules": 1,
    "num_virtual_tokens": 4,
    "peft_type": "CARTRIDGE",
    "peft_version": "0.21.2",
    "revision": None,
    "task_type": "CAUSAL_LM",
    "token_dim": 16,
}


class TestPacking:
    def test_the_rows_hold_each_layers_key_then_value_per_token(self) -> None:
        cache = _legacy(layers=2, heads=2, tokens=6, dim=3)
        packed = prompt_embeddings_from_past_key_values(cache, num_virtual_tokens=4)
        assert tuple(packed.shape) == (4, 2 * 2 * 2 * 3)
        rows = packed.tolist()
        for token in range(4):
            expected: list[float] = []
            for key, value in cache:
                for tensor in (key, value):
                    for head in range(2):
                        expected += tensor[0, head, token].tolist()
            assert rows[token] == expected

    def test_a_cache_object_a_list_and_an_iterable_all_pack_alike(self) -> None:
        legacy = _legacy()
        dynamic = DynamicCache.from_legacy_cache(legacy)
        packed = [
            prompt_embeddings_from_past_key_values(form, num_virtual_tokens=3)
            for form in (legacy, list(legacy), dynamic, iter(list(dynamic)))
        ]
        assert all(same(packed[0], p) for p in packed[1:])

    def test_the_dtype_and_the_device_of_the_cache_are_kept(self) -> None:
        cache = tuple((k.to(lucid.float16), v.to(lucid.float16)) for k, v in _legacy())
        assert (
            prompt_embeddings_from_past_key_values(cache, num_virtual_tokens=2).dtype
            == lucid.float16
        )

    @pytest.mark.parametrize(
        ("cache", "tokens", "match"),
        [
            ((), 2, "Empty"),
            (_legacy(tokens=3), 4, "at least 4 cached tokens"),
            (((lucid.zeros(2, 2, 5), lucid.zeros(2, 2, 5)),), 2, "rank 4"),
            (((lucid.zeros(2, 2, 5, 3), lucid.zeros(2, 2, 5, 3)),), 2, "single-sequence"),
        ],
    )
    def test_a_cache_that_does_not_fit_is_refused(
        self, cache: object, tokens: int, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            prompt_embeddings_from_past_key_values(cache, num_virtual_tokens=tokens)

    def test_layers_that_disagree_are_refused(self) -> None:
        cache = (_legacy(layers=1, heads=2)[0], _legacy(layers=1, heads=3)[0])
        with pytest.raises(ValueError, match="Inconsistent head shapes"):
            prompt_embeddings_from_past_key_values(cache, num_virtual_tokens=2)
        cache = (_legacy(layers=1, dim=3)[0], _legacy(layers=1, dim=4)[0])
        with pytest.raises(ValueError, match="Inconsistent head_dim"):
            prompt_embeddings_from_past_key_values(cache, num_virtual_tokens=2)

    def test_a_thing_that_is_not_a_cache_is_refused(self) -> None:
        with pytest.raises(TypeError, match="Unsupported `past_key_values` type"):
            prompt_embeddings_from_past_key_values(5, num_virtual_tokens=2)

    def test_a_layer_that_is_not_a_pair_of_tensors_is_refused(self) -> None:
        with pytest.raises(TypeError, match="must be a \\(key, value\\) pair of tensors"):
            prompt_embeddings_from_past_key_values(
                ((lucid.zeros(1, 2, 5, 3),),), num_virtual_tokens=2
            )
        with pytest.raises(TypeError, match="must be a \\(key, value\\) pair of tensors"):
            prompt_embeddings_from_past_key_values((([1.0], [2.0]),), num_virtual_tokens=2)


class _Tokens:
    """A tokenizer by its one method, ``encode(text) -> list[int]``: the byte values, folded into the vocabulary."""

    def encode(self, text: str) -> list[int]:
        return [2 + b % 40 for b in text.encode()]


class _Templated(_Tokens):
    """A tokenizer that also has a chat template: it frames the text with two ids."""

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> list[int]:
        assert tokenize and not add_generation_prompt and messages[0]["role"] == "system"
        return [1, *self.encode(messages[0]["content"]), 1]


class _BrokenTemplate(_Tokens):
    def apply_chat_template(self, *args: object, **kw: object) -> list[int]:
        raise ValueError("this tokenizer defines no chat template")


def _base_cache(ids: list[int], count: int) -> lucid.Tensor:
    """The rows of the first ``count`` tokens of what the base model caches when it reads ``ids``."""
    out = _lm()(input_ids=lucid.tensor([ids]), use_cache=True)
    return prompt_embeddings_from_past_key_values(out.past_key_values, num_virtual_tokens=count)


class TestInitialisation:
    def test_from_a_cache_the_adapter_takes_the_first_tokens(self) -> None:
        peft = _cartridge()
        out = _lm()(input_ids=lucid.tensor([[3, 4, 5, 6, 7, 8]]), use_cache=True)
        rows = initialize_kv_prefix_from_past_key_values(peft, past_key_values=out.past_key_values)
        assert tuple(rows.shape) == (_V, _WIDTH)
        assert same(_encoder(peft).weight, rows)
        assert same(_encoder(peft).weight, _base_cache([3, 4, 5, 6, 7, 8], _V))

    def test_a_count_other_than_the_adapters_is_refused_by_the_encoder(self) -> None:
        peft = _cartridge()
        cache = _lm()(input_ids=lucid.tensor([[3, 4, 5, 6, 7, 8]]), use_cache=True).past_key_values
        before = _encoder(peft).weight.clone()
        with pytest.raises(ValueError, match="Invalid `prompt_embeddings` shape"):
            initialize_kv_prefix_from_past_key_values(
                peft, past_key_values=cache, num_virtual_tokens=2
            )
        assert same(_encoder(peft).weight, before)

    def test_a_named_adapter_is_initialised_and_the_others_are_not(self) -> None:
        peft = _cartridge()
        peft.add_adapter("b", _config())
        before = _encoder(peft).weight.clone()
        cache = _lm()(input_ids=lucid.tensor([[3, 4, 5, 6, 7]]), use_cache=True).past_key_values
        initialize_kv_prefix_from_past_key_values(peft, adapter_name="b", past_key_values=cache)
        assert same(_encoder(peft).weight, before)
        assert not same(_encoder(peft, "b").weight, before)

    def test_an_unprojected_prefix_is_started_from_a_cache_too(self) -> None:
        peft = get_peft_model(
            _lm(), PrefixTuningConfig(num_virtual_tokens=_V, task_type="CAUSAL_LM")
        )
        cache = _lm()(input_ids=lucid.tensor([[3, 4, 5, 6, 7]]), use_cache=True).past_key_values
        rows = initialize_kv_prefix_from_past_key_values(peft, past_key_values=cache)
        assert same(peft.base_model.prompt_encoder["default"].embedding.weight, rows)  # type: ignore[attr-defined]

    def test_a_projected_prefix_and_other_methods_are_refused(self) -> None:
        cache = _legacy()
        projected = get_peft_model(
            _lm(),
            PrefixTuningConfig(
                num_virtual_tokens=_V,
                task_type="CAUSAL_LM",
                prefix_projection=True,
                encoder_hidden_size=8,
            ),
        )
        with pytest.raises(ValueError, match="prefix_projection"):
            initialize_kv_prefix_from_past_key_values(projected, past_key_values=cache)

    def test_a_text_is_read_by_the_base_model_with_the_adapter_off(self) -> None:
        peft = _cartridge()
        text = "a long document"
        rows = initialize_kv_prefix_from_text(peft, _Tokens(), text=text)
        assert same(_encoder(peft).weight, _base_cache(_Tokens().encode(text), _V))
        assert same(rows, _encoder(peft).weight)

    def test_the_chat_template_is_used_when_there_is_one_and_it_works(self) -> None:
        peft = _cartridge()
        initialize_kv_prefix_from_text(peft, _Templated(), text="a long document")
        assert same(
            _encoder(peft).weight, _base_cache([1, *_Tokens().encode("a long document"), 1], _V)
        )

    def test_without_a_working_template_or_when_told_not_to_the_plain_encoding_is_read(
        self,
    ) -> None:
        text = "a long document"
        expected = _base_cache(_Tokens().encode(text), _V)
        for tokenizer, kw in (
            (_BrokenTemplate(), {}),
            (_Templated(), {"use_chat_template": False}),
        ):
            peft = _cartridge()
            initialize_kv_prefix_from_text(peft, tokenizer, text=text, **kw)  # type: ignore[arg-type]
            assert same(_encoder(peft).weight, expected)

    def test_max_length_cuts_the_ids_before_the_model_reads_them(self) -> None:
        peft = _cartridge()
        initialize_kv_prefix_from_text(peft, _Tokens(), text="a long document", max_length=_V)
        assert same(
            _encoder(peft).weight, _base_cache(_Tokens().encode("a long document")[:_V], _V)
        )
        with pytest.raises(ValueError, match="at least 4 cached tokens"):
            initialize_kv_prefix_from_text(peft, _Tokens(), text="a long document", max_length=2)

    def test_a_tokenizer_that_cannot_encode_is_refused_with_what_it_needs(self) -> None:
        with pytest.raises(TypeError, match="encode"):
            initialize_kv_prefix_from_text(_cartridge(), object(), text="x")  # type: ignore[arg-type]

    def test_a_tokenizer_that_returns_something_else_is_refused(self) -> None:
        class Words:
            def encode(self, text: str) -> list[str]:
                return text.split()

        with pytest.raises(TypeError, match="token ids"):
            initialize_kv_prefix_from_text(_cartridge(), Words(), text="a b")  # type: ignore[arg-type]

    def test_an_empty_text_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no tokens"):
            initialize_kv_prefix_from_text(_cartridge(), _Tokens(), text="")

    def test_a_model_that_returns_no_cache_is_refused(self) -> None:
        class NoCache(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = _lm().config
                self.proj = nn.Linear(2, 2)

            def forward(self, input_ids: lucid.Tensor, **kw: object) -> object:
                class Out:
                    past_key_values = None

                return Out()

        base = _lm()
        peft = get_peft_model(base, _config())
        peft.base_model.model = NoCache()  # type: ignore[attr-defined,assignment]
        with pytest.raises(ValueError, match="no `past_key_values`"):
            initialize_kv_prefix_from_text(peft, _Tokens(), text="abc")

    def test_an_adapter_that_is_not_a_cache_is_refused(self) -> None:
        from lucid.peft import PromptTuningConfig

        peft = get_peft_model(
            _lm(), PromptTuningConfig(num_virtual_tokens=2, task_type="CAUSAL_LM")
        )
        with pytest.raises(ValueError, match="CARTRIDGE or PREFIX_TUNING"):
            initialize_kv_prefix_from_text(peft, _Tokens(), text="abc")


class TestComposition:
    def _saved(self, tmp_path: Path, name: str, rows: lucid.Tensor, frozen: int) -> Path:
        peft = _cartridge(num_virtual_tokens=int(rows.shape[0]), num_frozen_tokens=frozen)
        _encoder(peft).load_prompt_embeddings(rows)
        peft.save_pretrained(tmp_path / name)
        return tmp_path / name

    def test_the_caches_are_laid_end_to_end_with_the_first_ones_frozen_tokens(
        self, tmp_path: Path
    ) -> None:
        a = self._saved(tmp_path, "a", _rows(1, 4), 1)
        b = self._saved(tmp_path, "b", _rows(2, 3), 2)
        compose_cartridge_adapters([a, b], output_path=tmp_path / "joined")
        weights = load_safetensors(str(tmp_path / "joined" / "adapter_model.safetensors"))
        assert same(weights["prompt_embeddings"], lucid.cat([_rows(1, 4), _rows(2, 3)], dim=0))
        loaded = PeftModel.from_pretrained(_lm(), tmp_path / "joined")
        config = loaded.peft_config["default"]
        assert (config.num_virtual_tokens, config.num_frozen_tokens) == (7, 1)  # type: ignore[attr-defined]
        assert same(_encoder(loaded).weight, weights["prompt_embeddings"])

    def test_several_cartridges_compose_in_the_order_given(self, tmp_path: Path) -> None:
        dirs = [self._saved(tmp_path, n, _rows(i, 2), 1) for i, n in enumerate("abc")]
        compose_cartridge_adapters(dirs[::-1], output_path=tmp_path / "joined")
        weights = load_safetensors(str(tmp_path / "joined" / "adapter_model.safetensors"))
        assert same(
            weights["prompt_embeddings"], lucid.cat([_rows(i, 2) for i in (2, 1, 0)], dim=0)
        )

    def test_fewer_than_two_adapters_are_refused(self, tmp_path: Path) -> None:
        a = self._saved(tmp_path, "a", _rows(1, 4), 1)
        with pytest.raises(ValueError, match="at least 2"):
            compose_cartridge_adapters([a], output_path=tmp_path / "joined")

    def test_an_adapter_of_another_method_is_refused(self, tmp_path: Path) -> None:
        a = self._saved(tmp_path, "a", _rows(1, 4), 1)
        other = get_peft_model(
            _lm(), PrefixTuningConfig(num_virtual_tokens=_V, task_type="CAUSAL_LM")
        )
        other.save_pretrained(tmp_path / "prefix")
        with pytest.raises(ValueError, match="not a CARTRIDGE adapter"):
            compose_cartridge_adapters([a, tmp_path / "prefix"], output_path=tmp_path / "joined")

    def test_cartridges_of_different_sizes_are_refused(self, tmp_path: Path) -> None:
        a = self._saved(tmp_path, "a", _rows(1, 4), 1)
        lucid.manual_seed(0)
        wide = create_model("gpt2_small_lm", **{**_SIZES, "hidden_size": 8})
        get_peft_model(wide, _config()).save_pretrained(tmp_path / "wide")
        with pytest.raises(
            ValueError, match="Incompatible CARTRIDGE configs for attribute 'token_dim'"
        ):
            compose_cartridge_adapters([a, tmp_path / "wide"], output_path=tmp_path / "joined")
        assert not (tmp_path / "joined").exists()

    def test_a_pickle_is_not_written(self, tmp_path: Path) -> None:
        a = self._saved(tmp_path, "a", _rows(1, 4), 1)
        with pytest.raises(NotImplementedError, match="safetensors only"):
            compose_cartridge_adapters(
                [a, a], output_path=tmp_path / "joined", safe_serialization=False
            )


# ── the reference adapter library ────────────────────────────────────────────


def _pair(ref: ModuleType, **kw: object) -> tuple[CartridgeEncoder, object]:
    """The same encoder in both libraries, with the rows of ours copied into theirs."""
    library = adapter_library()
    fields: dict[str, object] = {
        "num_virtual_tokens": 5,
        "num_frozen_tokens": 2,
        "num_layers": 3,
        "token_dim": 4,
    } | kw
    mine = CartridgeEncoder(CartridgeConfig(**fields))  # type: ignore[arg-type]
    theirs = library.CartridgeEncoder(library.CartridgeConfig(**fields))
    theirs.load_prompt_embeddings(ref.tensor(mine.weight.tolist()))
    return mine, theirs


@pytest.mark.parity
@pytest.mark.parametrize("frozen", [0, 2, 5])
def test_the_encoder_matches_the_reference(ref: ModuleType, frozen: int) -> None:
    mine, theirs = _pair(ref, num_frozen_tokens=frozen)
    assert {k: tuple(v.shape) for k, v in mine.state_dict().items()} == {
        k: tuple(v.shape) for k, v in theirs.state_dict().items()  # type: ignore[attr-defined]
    }
    assert [n for n, p in mine.named_parameters() if p.requires_grad] == [
        n for n, p in theirs.named_parameters() if p.requires_grad  # type: ignore[attr-defined]
    ]
    assert float((ref.tensor(mine.weight.tolist()) - theirs.weight.detach()).abs().max()) == 0.0  # type: ignore[attr-defined]
    ids = lucid.zeros(3, 7).long()
    out = theirs(ref.zeros(3, 7).long())  # type: ignore[operator]
    assert tuple(mine(ids).shape) == tuple(out.shape)
    assert float((ref.tensor(mine(ids).tolist()) - out).abs().max()) == 0.0


@pytest.mark.parity
def test_the_packing_matches_the_reference(ref: ModuleType) -> None:
    adapter_library()
    from peft.tuners.cartridge.utils import prompt_embeddings_from_past_key_values as theirs

    cache = _legacy(layers=3, heads=2, tokens=7, dim=5, seed=3)
    mine = prompt_embeddings_from_past_key_values(cache, num_virtual_tokens=5)
    twin = tuple((ref.tensor(k.tolist()), ref.tensor(v.tolist())) for k, v in cache)
    other = theirs(twin, num_virtual_tokens=5)
    assert tuple(mine.shape) == tuple(other.shape)
    assert float((ref.tensor(mine.tolist()) - other).abs().max()) == 0.0


@pytest.mark.parity
def test_the_composition_matches_the_reference(ref: ModuleType, tmp_path: Path) -> None:
    library = adapter_library()
    paths = []
    for name, seed, count, frozen in (("a", 1, 4, 1), ("b", 2, 3, 2)):
        directory = tmp_path / name
        directory.mkdir()
        config = {**_REFERENCE_CONFIG, "num_virtual_tokens": count, "num_frozen_tokens": frozen}
        (directory / "adapter_config.json").write_text(json.dumps(config))
        save_safetensors(
            {"prompt_embeddings": _rows(seed, count)},
            str(directory / "adapter_model.safetensors"),
            metadata={"format": "pt"},
        )
        paths.append(directory)
    compose_cartridge_adapters(paths, output_path=tmp_path / "mine")
    library.compose_cartridge_adapters(
        [str(p) for p in paths], output_path=str(tmp_path / "theirs")
    )
    mine = json.loads((tmp_path / "mine" / "adapter_config.json").read_text())
    theirs = json.loads((tmp_path / "theirs" / "adapter_config.json").read_text())
    assert mine == theirs
    ours = load_safetensors(str(tmp_path / "mine" / "adapter_model.safetensors"))[
        "prompt_embeddings"
    ]
    from safetensors import safe_open

    with safe_open(
        str(tmp_path / "theirs" / "adapter_model.safetensors"), framework="pt"
    ) as handle:
        other = handle.get_tensor("prompt_embeddings")
    assert float((ref.tensor(ours.tolist()) - other).abs().max()) == 0.0
