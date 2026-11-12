"""The reference parameters ``lucid.peft`` accepts by name and refuses, or honours, by value.

``test_reference_signatures`` holds the signatures to the reference's; this module holds what the
parameters do: one that means nothing here (a meta-device build, a hub download, a framework
dtype switch) is accepted at its default and raises ``NotImplementedError`` naming itself at any
other value, never silently ignored; one with a meaning (``key_mapping``, ``is_main_process``,
``device``, ``revision``, ``auto_mapping_dict``) does it.
"""

import dataclasses
import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    AdaLoraConfig,
    AdaLoraModel,
    AdaptionPromptConfig,
    AdaptionPromptModel,
    ArrowConfig,
    AutoPeftModelForCausalLM,
    BdLoraConfig,
    KasaConfig,
    LoraConfig,
    LoraModel,
    MontecloraConfig,
    PeftConfig,
    PeftModel,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
    VeloraConfig,
    XLoraConfig,
    XLoraModel,
    get_peft_model,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from lucid.peft._refusals import refuse_kwargs, refuse_value, reject_unexpected
from lucid.peft.tuners.lora import LoraRuntimeConfig


def _net() -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


def _config(**kw: object) -> LoraConfig:
    return LoraConfig(r=2, target_modules=["0"], **kw)  # type: ignore[arg-type]


def _saved(path: Path, **kw: object) -> Path:
    get_peft_model(_net(), _config(**kw)).save_pretrained(path)
    return path


# -- the helpers ------------------------------------------------------------


class TestRefusals:
    def test_the_default_passes_and_anything_else_names_the_parameter(self) -> None:
        refuse_value("f", "flag", False, False, "why")
        refuse_value("f", "tokens", [], [], "why")
        with pytest.raises(NotImplementedError, match="f: `flag=True` is not supported.*why"):
            refuse_value("f", "flag", True, False, "why")

    def test_every_extra_keyword_is_named(self) -> None:
        refuse_kwargs("f", {}, "why")
        with pytest.raises(NotImplementedError, match="`a`, `b` are not supported.*why"):
            refuse_kwargs("f", {"b": 1, "a": 2}, "why")
        with pytest.raises(NotImplementedError, match="`a` is not supported"):
            refuse_kwargs("f", {"a": 2}, "why")

    def test_an_unexpected_keyword_is_a_type_error(self) -> None:
        reject_unexpected("f", {})
        with pytest.raises(TypeError, match=r"f\(\) got unexpected keyword arguments \['x'\]"):
            reject_unexpected("f", {"x": 1})


# -- configs ----------------------------------------------------------------

_LORA_ONLY = sorted(LoraConfig._all_unsupported)

#: A value of the option dataclass an extension's field holds: the field reads it, then refuses it.
_EXTENSION_OPTIONS: dict[str, Callable[[], object]] = {
    "arrow_config": ArrowConfig,
    "kasa_config": KasaConfig,
    "use_bdlora": BdLoraConfig,
    "monteclora_config": MontecloraConfig,
    "velora_config": VeloraConfig,
}


class TestConfig:
    @pytest.mark.parametrize("name", _LORA_ONLY)
    def test_a_field_the_library_lacks_is_accepted_at_its_default_only(self, name: str) -> None:
        field = {f.name: f for f in dataclasses.fields(LoraConfig)}[name]
        default = (
            field.default if field.default is not dataclasses.MISSING else field.default_factory()
        )
        assert getattr(LoraConfig(), name) == default
        other: object = (not default) if isinstance(default, bool) else "other"
        if isinstance(default, int) and not isinstance(default, bool):
            other = default + 1
        if name in _EXTENSION_OPTIONS:
            other = _EXTENSION_OPTIONS[name]()
        with pytest.raises(NotImplementedError, match=name):
            LoraConfig(**{name: other})  # type: ignore[arg-type]

    def test_a_file_that_turns_such_a_feature_on_is_refused_by_name(self) -> None:
        with pytest.raises(NotImplementedError, match="use_qalora"):
            LoraConfig.from_dict({"peft_type": "LORA", "use_qalora": True})

    def test_adalora_takes_the_lora_fields_it_does_not_use_at_their_default(self) -> None:
        assert AdaLoraConfig(total_step=10).use_dora is False
        for name, value in (("use_dora", True), ("use_rslora", True), ("lora_bias", True)):
            with pytest.raises(NotImplementedError, match=name):
                AdaLoraConfig(total_step=10, **{name: value})  # type: ignore[arg-type]

    def test_the_runtime_config_is_the_default_never_written_and_never_on(self) -> None:
        assert LoraConfig().runtime_config == LoraRuntimeConfig()
        assert "runtime_config" not in LoraConfig().to_dict()
        assert LoraConfig(runtime_config={}).runtime_config == LoraRuntimeConfig()  # type: ignore[arg-type]
        with pytest.raises(NotImplementedError, match="ephemeral_gpu_offload"):
            LoraRuntimeConfig(ephemeral_gpu_offload=True)
        with pytest.raises(TypeError, match="LoraRuntimeConfig"):
            LoraConfig(runtime_config=3)  # type: ignore[arg-type]

    def test_the_fields_are_positional_in_the_references_order(self) -> None:
        config = LoraConfig(None, None, None, "0.99", "gpt2")
        assert (config.peft_version, config.base_model_name_or_path) == ("0.99", "gpt2")

    def test_peft_version_is_metadata_kept_when_given_and_left_out_when_not(
        self, tmp_path: Path
    ) -> None:
        assert "peft_version" not in LoraConfig().to_dict()
        config = LoraConfig(peft_version="0.17.0", target_modules=["q"])
        assert config.to_dict()["peft_version"] == "0.17.0"
        config.save_pretrained(tmp_path)
        loaded = LoraConfig.from_pretrained(tmp_path)
        assert loaded.peft_version == "0.17.0"
        assert loaded == LoraConfig(target_modules=["q"])  # not part of what a config is

    def test_save_pretrained_takes_the_auto_mapping_dict_and_refuses_the_hub_options(
        self, tmp_path: Path
    ) -> None:
        mapping = {"base_model_class": "GPT2LMHeadModel", "parent_library": "lucid.models"}
        _config().save_pretrained(tmp_path, auto_mapping_dict=mapping)
        assert json.loads((tmp_path / "adapter_config.json").read_text())["auto_mapping"] == mapping
        with pytest.raises(NotImplementedError, match="push_to_hub"):
            _config().save_pretrained(tmp_path, push_to_hub=True)

    def test_from_pretrained_and_from_json_file_refuse_the_hub_options(
        self, tmp_path: Path
    ) -> None:
        _config().save_pretrained(tmp_path)
        with pytest.raises(NotImplementedError, match="revision"):
            PeftConfig.from_pretrained(tmp_path, revision="main")
        path = os.path.join(tmp_path, "adapter_config.json")
        with pytest.raises(NotImplementedError, match="token"):
            PeftConfig.from_json_file(path, token="t")
        assert PeftConfig.from_json_file(path)["peft_type"] == "LORA"
        assert isinstance(PeftConfig.__dict__["from_json_file"], classmethod)

    def test_adaption_prompt_has_no_targeting_fields_it_never_used(self) -> None:
        with pytest.raises(TypeError, match="exclude_modules"):
            AdaptionPromptConfig(exclude_modules=["x"])  # type: ignore[call-arg]
        assert AdaptionPromptConfig(modules_to_save=["head"]).modules_to_save == ["head"]


# -- PeftModel --------------------------------------------------------------

_OPTIONS = [("autocast_adapter_dtype", False), ("low_cpu_mem_usage", True)]


class TestPeftModelOptions:
    @pytest.mark.parametrize("name,value", _OPTIONS)
    def test_the_constructor_refuses_what_it_cannot_do(self, name: str, value: bool) -> None:
        net = _net()
        before = dict(net.state_dict())
        with pytest.raises(NotImplementedError, match=name):
            PeftModel(net, _config(), **{name: value})
        assert list(net.state_dict()) == list(before)  # the model was not touched

    @pytest.mark.parametrize("name,value", _OPTIONS)
    def test_get_peft_model_refuses_them_too(self, name: str, value: bool) -> None:
        with pytest.raises(NotImplementedError, match=name):
            get_peft_model(_net(), _config(), **{name: value})

    @pytest.mark.parametrize("name,value", _OPTIONS)
    def test_add_adapter_refuses_them(self, name: str, value: bool) -> None:
        peft = get_peft_model(_net(), _config())
        with pytest.raises(NotImplementedError, match=name):
            peft.add_adapter("b", _config(), **{name: value})
        assert list(peft.peft_config) == ["default"]

    def test_mixed_models_are_built_and_the_revision_is_recorded(self) -> None:
        mixed = get_peft_model(_net(), _config(), mixed=True, revision="abc123")
        assert type(mixed).__name__ == "PeftMixedModel"
        assert mixed.peft_config["default"].revision == "abc123"
        peft = get_peft_model(_net(), _config(), revision="abc123")
        assert peft.peft_config["default"].revision == "abc123"

    def test_inject_adapter_in_model_refuses_what_it_cannot_do(self) -> None:
        with pytest.raises(NotImplementedError, match="low_cpu_mem_usage"):
            inject_adapter_in_model(_config(), _net(), low_cpu_mem_usage=True)
        with pytest.raises(NotImplementedError, match="state_dict"):
            inject_adapter_in_model(_config(), _net(), state_dict={})
        assert inject_adapter_in_model(_config(), _net(), "default", False, None) is not None

    def test_a_tuner_model_refuses_them_and_wants_its_adapter_name(self) -> None:
        with pytest.raises(NotImplementedError, match="low_cpu_mem_usage"):
            LoraModel(_net(), _config(), "default", low_cpu_mem_usage=True)
        with pytest.raises(NotImplementedError, match="state_dict"):
            LoraModel(_net(), _config(), "default", state_dict={})
        with pytest.raises(TypeError, match="adapter_name"):
            LoraModel(_net(), _config())  # type: ignore[call-arg]
        tuner = AdaLoraModel(_net(), AdaLoraConfig(total_step=10, target_modules=["0"]), "d")
        assert tuner.active_adapters == ["d"]
        with pytest.raises(NotImplementedError, match="low_cpu_mem_usage"):
            AdaLoraModel(_net(), AdaLoraConfig(total_step=10), "d", low_cpu_mem_usage=True)
        with pytest.raises(NotImplementedError, match="state_dict"):
            AdaLoraModel(_net(), AdaLoraConfig(total_step=10), "d", state_dict={})
        with pytest.raises(TypeError, match="surprise"):
            AdaLoraModel(_net(), AdaLoraConfig(total_step=10), "d", surprise=1)

    def test_adaption_prompt_takes_the_configs_by_adapter_name_as_the_reference_does(self) -> None:
        from lucid.models.text.gpt2 import GPT2Config, GPT2Model

        def gpt() -> GPT2Model:
            return GPT2Model(
                GPT2Config(
                    num_hidden_layers=2,
                    hidden_size=16,
                    num_attention_heads=2,
                    intermediate_size=32,
                    vocab_size=20,
                )
            )

        config = AdaptionPromptConfig(adapter_len=2, adapter_layers=1)
        by_name = AdaptionPromptModel(gpt(), {"a": config}, "a")
        alone = AdaptionPromptModel(gpt(), config, "a")
        assert by_name.targeted_module_names == alone.targeted_module_names

    def test_xlora_refuses_the_options_it_cannot_do_before_it_loads_anything(self) -> None:
        config = XLoraConfig(adapters={"a": "nowhere"})
        for name, value in (
            ("device", "cpu"),
            ("ephemeral_gpu_offload", True),
            ("autocast_adapter_dtype", False),
        ):
            with pytest.raises(NotImplementedError, match=name):
                XLoraModel(_net(), config, "x", **{name: value})  # type: ignore[arg-type]
        with pytest.raises(NotImplementedError, match="trust"):
            XLoraModel(_net(), config, "x", trust=True)


class TestSaveAndLoad:
    def test_only_the_main_process_writes(self, tmp_path: Path) -> None:
        peft = get_peft_model(_net(), _config())
        peft.save_pretrained(tmp_path / "quiet", is_main_process=False)
        assert not (tmp_path / "quiet").exists()
        peft.save_pretrained(tmp_path / "loud", is_main_process=True)
        assert (tmp_path / "loud" / "adapter_model.safetensors").is_file()

    def test_the_embeddings_are_not_saved_and_the_hub_options_are_refused(
        self, tmp_path: Path
    ) -> None:
        peft = get_peft_model(_net(), _config())
        peft.save_pretrained(tmp_path / "a", save_embedding_layers=False)
        peft.save_pretrained(tmp_path / "b", save_embedding_layers="auto")
        with pytest.raises(NotImplementedError, match="save_embedding_layers"):
            peft.save_pretrained(tmp_path / "c", save_embedding_layers=True)
        with pytest.raises(NotImplementedError, match="push_to_hub"):
            peft.save_pretrained(tmp_path / "d", push_to_hub=True)
        assert not (tmp_path / "c").exists() and not (tmp_path / "d").exists()

    @pytest.mark.parametrize(
        "name,value",
        [
            ("autocast_adapter_dtype", False),
            ("ephemeral_gpu_offload", True),
            ("low_cpu_mem_usage", True),
        ],
    )
    def test_loading_refuses_what_it_cannot_do_with_the_model_untouched(
        self, tmp_path: Path, name: str, value: bool
    ) -> None:
        _saved(tmp_path)
        net = _net()
        with pytest.raises(NotImplementedError, match=name):
            PeftModel.from_pretrained(net, tmp_path, **{name: value})
        assert all("lora_" not in key for key in net.state_dict())
        peft = get_peft_model(_net(), _config())
        with pytest.raises(NotImplementedError, match=name):
            peft.load_adapter(tmp_path, "b", **{name: value})
        assert list(peft.peft_config) == ["default"]

    def test_the_hub_options_and_skipped_tensors_are_refused(self, tmp_path: Path) -> None:
        _saved(tmp_path)
        with pytest.raises(NotImplementedError, match="revision"):
            PeftModel.from_pretrained(_net(), tmp_path, revision="main")
        peft = get_peft_model(_net(), _config())
        with pytest.raises(NotImplementedError, match="token"):
            peft.load_adapter(tmp_path, "b", token="t")
        with pytest.raises(NotImplementedError, match="ignore_mismatched_sizes"):
            peft.load_adapter(tmp_path, "b", ignore_mismatched_sizes=True)
        assert list(peft.peft_config) == ["default"]
        peft.load_adapter(tmp_path, "b", ignore_mismatched_sizes=False)

    def test_load_adapter_reads_onto_the_device_it_is_given(self, tmp_path: Path) -> None:
        _saved(tmp_path, init_lora_weights=False)
        peft = get_peft_model(_net(), _config())
        peft.load_adapter(tmp_path, "b", device="cpu")
        assert "b" in peft.peft_config

    def test_key_mapping_renames_the_files_keys_before_they_are_loaded(
        self, tmp_path: Path
    ) -> None:
        trained = get_peft_model(_net(), _config(init_lora_weights=False))
        trained.save_pretrained(tmp_path)
        # Rename the layer in the file so that it only loads through the mapping.
        from lucid.serialization import load_safetensors, save_safetensors

        weights = dict(load_safetensors(str(tmp_path / "adapter_model.safetensors")))
        renamed = {k.replace(".0.", ".first."): v for k, v in weights.items()}
        save_safetensors(renamed, str(tmp_path / "adapter_model.safetensors"))
        with pytest.raises(ValueError, match="do not fit"):
            PeftModel.from_pretrained(_net(), tmp_path)
        loaded = PeftModel.from_pretrained(
            _net(), tmp_path, key_mapping={r"^first\.": "0.", r"\.first\.": ".0."}
        )
        x = lucid.randn(3, 4)
        assert bool(lucid.allclose(loaded(x), trained(x), atol=1e-6))
        # load_adapter takes the same mapping
        peft = get_peft_model(_net(), _config())
        peft.load_adapter(tmp_path, "b", key_mapping={r"^first\.": "0."})
        assert sorted(peft.peft_config) == ["b", "default"]

    def test_a_key_outside_the_adapter_prefix_cannot_be_mapped(self) -> None:
        from lucid.peft._peft_model import _remapped

        with pytest.raises(ValueError, match="base_model"):
            _remapped({"lone": lucid.zeros(1)}, {"a": "b"})
        weights = {"base_model.model.a": lucid.zeros(1)}
        assert _remapped(weights, None) is weights
        assert list(_remapped(weights, {"a": "z"})) == ["base_model.model.z"]
        assert list(_remapped(weights, {"nomatch": "z"})) == ["base_model.model.a"]


class TestStateDict:
    @pytest.mark.parametrize(
        "name,value", [("unwrap_compiled", True), ("save_embedding_layers", True)]
    )
    def test_extracting_refuses_what_it_cannot_do(self, name: str, value: bool) -> None:
        peft = get_peft_model(_net(), _config())
        with pytest.raises(NotImplementedError, match=name):
            get_peft_model_state_dict(peft, **{name: value})
        assert get_peft_model_state_dict(peft, None, "default", False, "auto")
        assert get_peft_model_state_dict(peft, save_embedding_layers=False)

    @pytest.mark.parametrize("name", ["ignore_mismatched_sizes", "low_cpu_mem_usage"])
    def test_loading_refuses_what_it_cannot_do(self, name: str) -> None:
        peft = get_peft_model(_net(), _config())
        state = get_peft_model_state_dict(peft)
        with pytest.raises(NotImplementedError, match=name):
            set_peft_model_state_dict(peft, state, **{name: True})
        set_peft_model_state_dict(peft, state, "default", False, False)


# -- calls ------------------------------------------------------------------


class TestForward:
    def test_the_routing_keywords_are_checked(self) -> None:
        peft = get_peft_model(_net(), _config())
        with pytest.raises(TypeError, match="adapter_names"):
            peft(lucid.randn(2, 4), adapter_names="default")
        with pytest.raises(TypeError, match="adapter_names"):
            peft(lucid.randn(2, 4), adapter_names=[1, 2])
        with pytest.raises(TypeError, match="task_ids"):
            peft(lucid.randn(2, 4), task_ids=[0, 1])

    def test_adapter_names_route_through_the_keywords(self) -> None:
        peft = get_peft_model(_net(), _config(init_lora_weights=False))
        peft.add_adapter("b", _config(init_lora_weights=False))
        peft.eval()
        x = lucid.randn(2, 4)
        mixed = peft(x, adapter_names=["default", "__base__"])
        assert tuple(mixed.shape) == (2, 2)


class _Encoder(nn.Module):
    """A model whose ``forward`` names the arguments a task wrapper hands over."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 3)

    def forward(
        self,
        input_ids: lucid.Tensor,
        attention_mask: lucid.Tensor | None = None,
        labels: lucid.Tensor | None = None,
        decoder_input_ids: lucid.Tensor | None = None,
        start_positions: lucid.Tensor | None = None,
        token_type_ids: lucid.Tensor | None = None,
    ) -> tuple[object, ...]:
        return (
            self.proj(input_ids),
            attention_mask,
            labels,
            decoder_input_ids,
            start_positions,
            token_type_ids,
        )


class TestTaskModels:
    def _wrap(self, wrapper: type[PeftModel], task: str) -> PeftModel:
        config = LoraConfig(r=2, target_modules=["proj"], task_type=task)
        return wrapper(_Encoder(), config)

    def test_the_first_input_goes_positionally_and_the_rest_by_name(self) -> None:
        x, mask, labels = lucid.randn(2, 3), lucid.ones(2, 3), lucid.zeros(2)
        peft = self._wrap(PeftModelForSequenceClassification, "SEQ_CLS")
        _, got_mask, got_labels, *rest = peft(x, attention_mask=mask, labels=labels)  # type: ignore[misc]
        assert got_mask is mask and got_labels is labels and rest == [None, None, None]
        _, none_mask, *_ = peft(input_ids=x)  # type: ignore[misc]
        assert none_mask is None

    def test_a_task_with_its_own_arguments_names_them(self) -> None:
        x, dec = lucid.randn(2, 3), lucid.zeros(2, 3)
        seq = self._wrap(PeftModelForSeq2SeqLM, "SEQ_2_SEQ_LM")
        assert seq(x, decoder_input_ids=dec)[3] is dec  # type: ignore[index]
        qa = self._wrap(PeftModelForQuestionAnswering, "QUESTION_ANS")
        start = lucid.zeros(2)
        assert qa(x, token_type_ids=dec, start_positions=start)[4] is start  # type: ignore[index]
        feat = self._wrap(PeftModelForFeatureExtraction, "FEATURE_EXTRACTION")
        assert feat(x, attention_mask=dec)[1] is dec  # type: ignore[index]

    def test_an_argument_the_model_has_no_parameter_for_is_an_error(self) -> None:
        peft = self._wrap(PeftModelForSequenceClassification, "SEQ_CLS")
        with pytest.raises(TypeError, match="inputs_embeds"):
            peft(inputs_embeds=lucid.randn(2, 3))

    def test_adapter_names_and_unknown_keywords_pass_through(self) -> None:
        peft = self._wrap(PeftModelForSequenceClassification, "SEQ_CLS")
        peft.add_adapter("b", LoraConfig(r=2, target_modules=["proj"]))
        peft.eval()
        out = peft(lucid.randn(2, 3), adapter_names=["default", "b"])
        assert tuple(out[0].shape) == (2, 3)  # type: ignore[index]

    def test_the_constructor_takes_the_options_of_peft_model_and_nothing_else(self) -> None:
        config = LoraConfig(r=2, target_modules=["proj"])
        PeftModelForSeq2SeqLM(_Encoder(), config, "x", autocast_adapter_dtype=True)
        for wrapper in (PeftModelForSeq2SeqLM, PeftModelForTokenClassification):
            with pytest.raises(NotImplementedError, match="low_cpu_mem_usage"):
                wrapper(_Encoder(), config, low_cpu_mem_usage=True)
            with pytest.raises(NotImplementedError, match="autocast_adapter_dtype"):
                wrapper(_Encoder(), config, autocast_adapter_dtype=False)
            with pytest.raises(TypeError, match="unexpected"):
                wrapper(_Encoder(), config, surprise=True)

    def test_token_classification_keeps_the_references_default_but_needs_a_config(self) -> None:
        with pytest.raises(TypeError, match="peft_config"):
            PeftModelForTokenClassification(_Encoder())

    def test_seq2seq_generate_is_keywords_only(self) -> None:
        peft = self._wrap(PeftModelForSeq2SeqLM, "SEQ_2_SEQ_LM")
        with pytest.raises(TypeError):
            peft.generate(lucid.randn(1, 3))  # type: ignore[call-arg]


class TestGenerate:
    class _Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(2, 2)

        def generate(self, x: lucid.Tensor, steps: int = 1) -> lucid.Tensor:
            for _ in range(steps):
                x = self.proj(x)
            return x

    def test_the_routing_keywords_are_checked_and_not_given_to_the_model(self) -> None:
        peft = PeftModel(self._Decoder(), LoraConfig(r=2, target_modules=["proj"]))
        assert tuple(peft.generate(lucid.randn(1, 2), steps=2).shape) == (1, 2)  # type: ignore[attr-defined]
        with pytest.raises(TypeError, match="adapter_names"):
            peft.generate(lucid.randn(1, 2), adapter_names="default")  # type: ignore[attr-defined]
        with pytest.raises(TypeError, match="task_ids"):
            peft.generate(lucid.randn(1, 2), task_ids="0")  # type: ignore[attr-defined]


class TestAutoPeft:
    def test_the_hub_options_are_refused_before_the_directory_is_read(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError, match="revision"):
            AutoPeftModelForCausalLM.from_pretrained(tmp_path, revision="main")
        with pytest.raises(NotImplementedError, match="token"):
            AutoPeftModelForCausalLM.from_pretrained(tmp_path, token="t")
