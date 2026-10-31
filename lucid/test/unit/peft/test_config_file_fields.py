"""``adapter_config.json`` holds the reference's fields; a field only this library has is written when it is set.

IA3's reference config has no ``layers_to_transform`` / ``layers_pattern``
and VeRA's none ``exclude_modules``; ours accepts them.  A file that does not
use them must be one the reference reads, so the keys are left out until they
carry a value — and a file with them still loads here.  Messages that every
method shares name the method of the adapter, not LoRA.
"""

import dataclasses
import json
from pathlib import Path

import pytest

import lucid.nn as nn
from lucid.peft import (
    IA3Config,
    LoraConfig,
    PeftConfig,
    PeftModel,
    VeraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from lucid.peft._config import _CONFIG_CLASSES

#: The fields of the reference's configs (v0.21.2) that ours has no field for are listed here.
_REFERENCE_KEYS = {
    "IA3": {
        "target_modules",
        "exclude_modules",
        "feedforward_modules",
        "fan_in_fan_out",
        "modules_to_save",
        "init_ia3_weights",
    },
    "VERA": {
        "r",
        "target_modules",
        "projection_prng_key",
        "save_projection",
        "vera_dropout",
        "d_initial",
        "fan_in_fan_out",
        "bias",
        "modules_to_save",
        "init_weights",
        "layers_to_transform",
        "layers_pattern",
    },
}
_BASE_KEYS = {
    "task_type",
    "peft_type",
    "auto_mapping",
    "base_model_name_or_path",
    "revision",
    "inference_mode",
}


def _ia3_net() -> nn.Module:
    return nn.Sequential(nn.Linear(6, 6), nn.Linear(6, 3))


class TestExtensionFields:
    @pytest.mark.parametrize("config_class", list(_CONFIG_CLASSES.values()))
    def test_an_extension_field_is_a_field_of_the_class(
        self, config_class: type[PeftConfig]
    ) -> None:
        names = {f.name for f in dataclasses.fields(config_class)}
        assert config_class._extension_fields <= names

    @pytest.mark.parametrize("peft_type", sorted(_REFERENCE_KEYS))
    def test_a_default_file_holds_only_the_reference_keys(self, peft_type: str) -> None:
        config_class = next(c for c in _CONFIG_CLASSES.values() if c._peft_type == peft_type)
        keys = set(config_class(target_modules=["x"]).to_dict())
        assert keys == _REFERENCE_KEYS[peft_type] | _BASE_KEYS

    @pytest.mark.parametrize(
        ("config_class", "field", "value"),
        [
            (IA3Config, "layers_to_transform", [0]),
            (IA3Config, "layers_pattern", "layers"),
            (VeraConfig, "exclude_modules", ["x"]),
        ],
    )
    def test_a_field_that_is_set_is_written_and_read_back(
        self, config_class: type[PeftConfig], field: str, value: object
    ) -> None:
        options: dict[str, object] = {"target_modules": ["a", "b"], field: value}
        if field == "layers_pattern":
            options["layers_to_transform"] = [0]
        config = config_class(**options)
        assert config.to_dict()[field] is not None
        assert config_class.from_dict(json.loads(json.dumps(config.to_dict()))) == config

    def test_other_methods_keep_every_field(self) -> None:
        assert {"layers_to_transform", "layers_pattern", "exclude_modules"} <= set(
            LoraConfig().to_dict()
        )


class TestSavedFile:
    def test_an_ia3_adapter_file_has_no_layer_selection_keys(self, tmp_path: Path) -> None:
        config = IA3Config(target_modules=["0"], feedforward_modules=[])
        get_peft_model(_ia3_net(), config).save_pretrained(tmp_path)
        saved = json.loads((tmp_path / "adapter_config.json").read_text())
        assert "layers_to_transform" not in saved and "layers_pattern" not in saved

    def test_a_file_with_or_without_the_keys_loads(self, tmp_path: Path) -> None:
        config = IA3Config(target_modules=["0"], feedforward_modules=[])
        get_peft_model(_ia3_net(), config).save_pretrained(tmp_path)
        path = tmp_path / "adapter_config.json"
        bare = PeftModel.from_pretrained(_ia3_net(), tmp_path)
        saved = json.loads(path.read_text())
        path.write_text(json.dumps({**saved, "layers_to_transform": None, "layers_pattern": None}))
        explicit = PeftModel.from_pretrained(_ia3_net(), tmp_path)
        assert bare.peft_config["default"] == explicit.peft_config["default"]

    def test_a_layer_selection_survives_the_file(self, tmp_path: Path) -> None:
        config = IA3Config(target_modules=["0"], feedforward_modules=[], layers_to_transform=[0])
        config.save_pretrained(tmp_path)
        saved = json.loads((tmp_path / "adapter_config.json").read_text())
        assert saved["layers_to_transform"] == [0]
        assert PeftConfig.from_pretrained(tmp_path) == config


class TestMethodNamedMessages:
    def test_a_model_without_the_adapter_names_what_it_holds(self) -> None:
        peft = get_peft_model(_ia3_net(), IA3Config(target_modules=["0"], feedforward_modules=[]))
        with pytest.raises(ValueError, match=r"no adapter named 'other' \(it holds 'default' \(IA3\)\)"):
            get_peft_model_state_dict(peft, adapter_name="other")
        with pytest.raises(ValueError, match="no adapter named 'default'"):
            get_peft_model_state_dict(_ia3_net())

    @pytest.mark.parametrize("finish", ["merge_and_unload", "unload"])
    def test_a_spent_model_names_its_method(self, finish: str, tmp_path: Path) -> None:
        peft = get_peft_model(_ia3_net(), IA3Config(target_modules=["0"], feedforward_modules=[]))
        getattr(peft, finish)()
        with pytest.raises(RuntimeError, match=r"no IA3 layer left"):
            peft.save_pretrained(tmp_path / "out")

    def test_a_spent_lora_model_still_says_lora(self, tmp_path: Path) -> None:
        peft = get_peft_model(_ia3_net(), LoraConfig(target_modules=["0"]))
        peft.unload()
        with pytest.raises(RuntimeError, match=r"no LORA layer left"):
            peft.save_pretrained(tmp_path / "out")
