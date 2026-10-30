"""How ``lucid.peft`` is put together: the registries, the bases and the slot protocol.

A method is a config class, a tuner and layer classes, each registered
as it is defined; the entry points dispatch on the method a config or a
file names.  These hold that wiring to what LoRA registers, and the
contracts the generic code relies on.
"""

import json
from pathlib import Path

import pytest

import lucid.nn as nn
from lucid.peft import (
    LoraConfig,
    LoraModel,
    PeftConfig,
    PeftModel,
    PeftType,
    get_peft_model,
    inject_adapter_in_model,
)
from lucid.peft._config import _CONFIG_CLASSES, _TargetedConfig
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft._slots import _adapter_slots, _AdapterSlot
from lucid.peft.tuners._base import _TUNERS, _tuner_for
from lucid.peft.tuners.lora import Linear, LoraLayer


def _net() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


class TestRegistries:
    def test_lora_registers_its_config_tuner_and_layer(self) -> None:
        assert _CONFIG_CLASSES[PeftType.LORA] is LoraConfig
        assert _TUNERS[PeftType.LORA] is LoraModel
        assert LoraLayer._layer_types == {nn.Linear: Linear}
        assert LoraModel._layer_types is LoraLayer._layer_types

    def test_the_entry_points_dispatch_on_the_config(self) -> None:
        config = LoraConfig(target_modules=["0"])
        assert _tuner_for(config) is LoraModel
        assert type(get_peft_model(_net(), config).base_model) is LoraModel
        net = _net()
        inject_adapter_in_model(config, net)
        assert isinstance(net[0], Linear)

    def test_a_config_of_no_tuner_is_refused(self) -> None:
        config = LoraConfig(target_modules=["0"])
        config.peft_type = PeftType.IA3
        with pytest.raises(NotImplementedError, match="no tuner for IA3"):
            _tuner_for(config)

    def test_peft_config_reads_the_method_a_file_names(self, tmp_path: Path) -> None:
        LoraConfig(r=2, target_modules=["q"], base_model_name_or_path="m").save_pretrained(tmp_path)
        config = PeftConfig.from_pretrained(tmp_path)
        assert type(config) is LoraConfig and config.r == 2
        assert type(PeftConfig.from_dict({"target_modules": ["q"]})) is LoraConfig
        with pytest.raises(NotImplementedError, match="IA3"):
            PeftConfig.from_dict({"peft_type": "IA3"})

    def test_peft_model_reads_the_method_a_file_names(self, tmp_path: Path) -> None:
        get_peft_model(_net(), LoraConfig(target_modules=["0"])).save_pretrained(tmp_path)
        data = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
        assert data["peft_type"] == "LORA"
        assert type(PeftModel.from_pretrained(_net(), tmp_path).base_model) is LoraModel


class TestBases:
    @pytest.mark.parametrize("base", [PeftConfig, _TargetedConfig])
    def test_a_base_config_is_not_built_directly(self, base: type[PeftConfig]) -> None:
        with pytest.raises(TypeError, match="base of the adapter configs; build one of LoraConfig"):
            base()

    def test_the_base_fields_are_checked_by_the_table(self) -> None:
        config = LoraConfig(revision="v1.2", base_model_name_or_path="gpt2")
        assert (config.revision, config.base_model_name_or_path) == ("v1.2", "gpt2")
        with pytest.raises(TypeError, match="`revision` must be a str or None"):
            LoraConfig(revision=3)  # type: ignore[arg-type]

    def test_the_field_checks_are_merged_down_the_hierarchy(self) -> None:
        assert set(PeftConfig._all_checks) < set(_TargetedConfig._all_checks)
        assert set(_TargetedConfig._all_checks) < set(LoraConfig._all_checks)
        assert "r" in LoraConfig._all_checks and "r" not in _TargetedConfig._all_checks

    def test_lora_layer_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            LoraLayer(nn.Linear(2, 2), "default", r=1)  # type: ignore[abstract]

    def test_the_layers_say_what_they_hold(self) -> None:
        peft = get_peft_model(
            _net(), LoraConfig(r=2, lora_alpha=4, target_modules=["0"], modules_to_save=["2"])
        )
        model = peft.get_base_model()
        assert "adapter='default', r=2, lora_alpha=4, merged=False, disabled=False" in repr(
            model[0]
        )
        assert "active_adapter='default', disabled=False" in repr(model[2])


class TestSlots:
    def test_both_kinds_of_slot_answer_the_protocol(self) -> None:
        peft = get_peft_model(_net(), LoraConfig(target_modules=["0"], modules_to_save=["2"]))
        model = peft.get_base_model()
        slots = dict(_adapter_slots(model))
        assert set(slots) == {"0", "2"}
        assert isinstance(model[0], _AdapterSlot) and isinstance(model[2], _AdapterSlot)
        assert not isinstance(nn.Linear(2, 2), _AdapterSlot)
        wrapper = model[2]
        assert isinstance(wrapper, ModulesToSaveWrapper)
        assert wrapper.get_base_layer() is wrapper.original_module
        assert model[0].get_base_layer() is model[0].base_layer


class TestDisableAdapter:
    def test_blocks_nest(self) -> None:
        peft = get_peft_model(_net(), LoraConfig(target_modules=["0"]))
        layer = peft.get_base_model()[0]
        with peft.disable_adapter():
            with peft.disable_adapter():
                assert layer.disable_adapters
            assert layer.disable_adapters
        assert not layer.disable_adapters

    def test_the_context_manager_is_reusable(self) -> None:
        peft = get_peft_model(_net(), LoraConfig(target_modules=["0"]))
        layer = peft.get_base_model()[0]
        block = peft.disable_adapter()
        for _ in range(2):
            with block:
                assert layer.disable_adapters
            assert not layer.disable_adapters
