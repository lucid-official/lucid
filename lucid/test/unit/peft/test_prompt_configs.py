"""The configs and encoders of the three prompt-learning methods, before any model is involved."""

import json
import warnings
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    PeftConfig,
    PeftType,
    PrefixEncoder,
    PrefixTuningConfig,
    PromptEmbedding,
    PromptEncoder,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
    PromptLearningConfig,
    PromptTuningConfig,
    PromptTuningInit,
)

_SIZES = {
    "token_dim": 8,
    "num_layers": 2,
    "num_attention_heads": 2,
    "num_transformer_submodules": 1,
}


class _Chars:
    """A tokenizer that turns each character into its code point, for TEXT initialisation."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 10 for c in text]


class TestConfigs:
    def test_each_method_registers_under_its_peft_type(self) -> None:
        assert PromptTuningConfig(num_virtual_tokens=2).peft_type == PeftType.PROMPT_TUNING
        assert PrefixTuningConfig(num_virtual_tokens=2).peft_type == PeftType.PREFIX_TUNING
        assert PromptEncoderConfig(num_virtual_tokens=2).peft_type == PeftType.P_TUNING

    @pytest.mark.parametrize(
        "config",
        [
            PromptTuningConfig(num_virtual_tokens=4, task_type="CAUSAL_LM", **_SIZES),
            PrefixTuningConfig(num_virtual_tokens=4, prefix_projection=True, encoder_hidden_size=5),
            PromptEncoderConfig(
                num_virtual_tokens=4, encoder_reparameterization_type="LSTM", encoder_dropout=0.1
            ),
        ],
        ids=["prompt", "prefix", "p-tuning"],
    )
    def test_a_file_names_its_class_and_round_trips(
        self, config: PeftConfig, tmp_path: Path
    ) -> None:
        config.save_pretrained(tmp_path)
        loaded = PeftConfig.from_pretrained(tmp_path)
        assert type(loaded) is type(config) and loaded == config
        assert isinstance(loaded, PromptLearningConfig)

    def test_a_file_from_the_reference_reads(self, tmp_path: Path) -> None:
        (tmp_path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "auto_mapping": None,
                    "base_model_name_or_path": "gpt2",
                    "inference_mode": True,
                    "num_attention_heads": 12,
                    "num_layers": 12,
                    "num_transformer_submodules": 1,
                    "num_virtual_tokens": 20,
                    "peft_type": "PROMPT_TUNING",
                    "prompt_tuning_init": "RANDOM",
                    "prompt_tuning_init_text": None,
                    "revision": None,
                    "task_type": "CAUSAL_LM",
                    "token_dim": 768,
                    "tokenizer_kwargs": None,
                    "tokenizer_name_or_path": None,
                    "peft_version": "0.21.2",
                }
            )
        )
        config = PeftConfig.from_pretrained(tmp_path)
        assert isinstance(config, PromptTuningConfig)
        assert (config.num_virtual_tokens, config.token_dim) == (20, 768)

    def test_text_initialisation_needs_its_text_and_tokenizer(self) -> None:
        with pytest.raises(ValueError, match="tokenizer_name_or_path"):
            PromptTuningConfig(num_virtual_tokens=2, prompt_tuning_init="TEXT")
        with pytest.raises(ValueError, match="prompt_tuning_init_text"):
            PromptTuningConfig(
                num_virtual_tokens=2, prompt_tuning_init="TEXT", tokenizer_name_or_path="t"
            )
        with pytest.raises(ValueError, match="tokenizer_kwargs"):
            PromptTuningConfig(num_virtual_tokens=2, tokenizer_kwargs={"a": 1})
        with pytest.raises(TypeError, match="tokenizer_kwargs"):
            PromptTuningConfig(num_virtual_tokens=2, tokenizer_kwargs=[1])  # type: ignore[arg-type]
        assert (
            PromptTuningConfig(
                num_virtual_tokens=2,
                prompt_tuning_init="TEXT",
                prompt_tuning_init_text="hi",
                tokenizer_name_or_path="t",
                tokenizer_kwargs={"use_fast": True},
            ).prompt_tuning_init
            is PromptTuningInit.TEXT
        )

    @pytest.mark.parametrize(
        ("build", "match"),
        [
            (lambda: PromptTuningConfig(num_virtual_tokens=0), "positive"),
            (lambda: PromptTuningConfig(num_virtual_tokens=True), "int"),  # type: ignore[arg-type]
            (lambda: PromptTuningConfig(prompt_tuning_init="nope"), "must be one of"),
            (lambda: PrefixTuningConfig(init_weights="one"), "'zero' or None"),
            (lambda: PrefixTuningConfig(prefix_projection="yes"), "bool"),  # type: ignore[arg-type]
            (lambda: PromptEncoderConfig(encoder_reparameterization_type="GRU"), "must be one of"),
            (lambda: PromptEncoderConfig(encoder_dropout=1.0), r"\[0, 1\)"),
        ],
    )
    def test_a_field_out_of_range_is_refused(self, build: object, match: str) -> None:
        assert callable(build)
        with pytest.raises((TypeError, ValueError), match=match):
            build()

    def test_the_base_is_not_built_directly(self) -> None:
        with pytest.raises(TypeError, match="base of the adapter configs"):
            PromptLearningConfig()


class TestEncoders:
    def _ids(self, rows: int) -> lucid.Tensor:
        return lucid.arange(rows).long().reshape(1, rows)

    def test_prompt_embedding_is_a_table_of_the_virtual_tokens(self) -> None:
        config = PromptTuningConfig(
            num_virtual_tokens=3, **{**_SIZES, "num_transformer_submodules": 2}
        )
        encoder = PromptEmbedding(config, nn.Embedding(10, 8))
        assert encoder.num_embeddings == 6 and encoder.stores_prompt
        assert encoder(self._ids(6)).shape == (1, 6, 8)
        assert encoder.prompt_embeddings().shape == (6, 8)

    def test_sample_vocab_starts_from_vocabulary_rows(self) -> None:
        table = nn.Embedding(10, 8)
        config = PromptTuningConfig(
            num_virtual_tokens=4, prompt_tuning_init="SAMPLE_VOCAB", **_SIZES
        )
        rows = PromptEmbedding(config, table).embedding.weight
        for row in rows:
            assert any(bool((row == t).all().item()) for t in table.weight)

    def test_text_starts_from_the_texts_tokens_cut_or_repeated(self) -> None:
        table = nn.Embedding(10, 8)
        for text, expected in (("abc", [7, 8, 9, 7, 8]), ("abcdefgh", [7, 8, 9, 0, 1])):
            config = PromptTuningConfig(
                num_virtual_tokens=5,
                prompt_tuning_init="TEXT",
                prompt_tuning_init_text=text,
                tokenizer_name_or_path="chars",
                **_SIZES,
            )
            weight = PromptEmbedding(config, table, tokenizer=_Chars()).embedding.weight
            want = table.weight[lucid.tensor(expected).long()]
            assert bool(lucid.allclose(weight, want))

    def test_text_without_a_tokenizer_says_what_to_do(self) -> None:
        config = PromptTuningConfig(
            num_virtual_tokens=2,
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text="hi",
            tokenizer_name_or_path="gpt2",
            **_SIZES,
        )
        with pytest.raises(ValueError, match="no loader for tokenizer_name_or_path='gpt2'"):
            PromptEmbedding.from_config(config, nn.Embedding(10, 8))
        silent = PromptTuningConfig(
            num_virtual_tokens=2,
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text="",
            tokenizer_name_or_path="x",
            **_SIZES,
        )
        with pytest.raises(ValueError, match="tokenizes to nothing"):
            PromptEmbedding(silent, nn.Embedding(10, 8), tokenizer=_Chars())
        with pytest.raises(TypeError, match="nn.Embedding token table"):
            PromptEmbedding(config, None)

    def test_an_inference_encoder_skips_the_initialisation(self) -> None:
        config = PromptTuningConfig(
            num_virtual_tokens=2, prompt_tuning_init="SAMPLE_VOCAB", inference_mode=True, **_SIZES
        )
        assert PromptEmbedding(config, None).embedding.weight.shape == (2, 8)

    def test_prefix_rows_hold_every_layers_key_and_value(self) -> None:
        config = PrefixTuningConfig(num_virtual_tokens=3, encoder_hidden_size=6, **_SIZES)
        encoder = PrefixEncoder(config)
        assert encoder.stores_prompt and encoder.num_embeddings == 3
        assert encoder(self._ids(3)).shape == (1, 3, 2 * 2 * 8)

    def test_prefix_projection_trains_through_an_mlp_and_stores_the_result(self) -> None:
        config = PrefixTuningConfig(
            num_virtual_tokens=3, prefix_projection=True, encoder_hidden_size=6, **_SIZES
        )
        encoder = PrefixEncoder(config)
        assert not encoder.stores_prompt and encoder.embedding.weight.shape == (3, 8)
        assert encoder.prompt_embeddings().shape == (3, 32)
        stored = PrefixEncoder(PrefixTuningConfig(**{**vars(config), "inference_mode": True}))
        assert stored.stores_prompt and stored.embedding.weight.shape == (3, 32)

    @pytest.mark.parametrize("projection", [False, True])
    def test_zero_init_makes_the_prefix_null(self, projection: bool) -> None:
        config = PrefixTuningConfig(
            num_virtual_tokens=3,
            prefix_projection=projection,
            init_weights="zero",
            encoder_hidden_size=6,
            **_SIZES,
        )
        assert float(PrefixEncoder(config).prompt_embeddings().abs().sum().item()) == 0.0

    @pytest.mark.parametrize("kind", ["MLP", "LSTM"])
    def test_p_tuning_maps_the_embeddings_through_its_encoder(self, kind: str) -> None:
        config = PromptEncoderConfig(
            num_virtual_tokens=3,
            encoder_reparameterization_type=kind,
            encoder_hidden_size=6,
            **_SIZES,
        )
        encoder = PromptEncoder(config)
        assert not encoder.stores_prompt
        assert (encoder.lstm_head is not None) == (kind == "LSTM")
        assert encoder(self._ids(3)).shape == (1, 3, 8)
        stored = PromptEncoder(PromptEncoderConfig(**{**vars(config), "inference_mode": True}))
        assert stored.stores_prompt and stored.lstm_head is None

    def test_an_mlp_encoder_warns_of_ignored_layers(self) -> None:
        config = PromptEncoderConfig(num_virtual_tokens=2, encoder_num_layers=3, **_SIZES)
        with pytest.warns(UserWarning, match="encoder_num_layers"):
            PromptEncoder(config)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PromptEncoder(PromptEncoderConfig(num_virtual_tokens=2, **_SIZES))

    def test_the_reparameterization_names_are_the_references(self) -> None:
        assert [t.value for t in PromptEncoderReparameterizationType] == ["MLP", "LSTM"]
        assert [t.value for t in PromptTuningInit] == ["TEXT", "SAMPLE_VOCAB", "RANDOM"]
