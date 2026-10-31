"""``get_layer_status`` / ``get_model_status``: what each adapter layer, and the model, is doing.

A crafted model — three linears, adapter ``a`` on the first two and ``b`` on
the last two, plus a ``modules_to_save`` head — is put into each state the
reports tell apart (disabled, merged, frozen, switched on one layer only), and
the layer report and the folded model report are read back against the
state set by hand.
"""

import dataclasses

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    LoraConfig,
    PeftModel,
    TunerLayerStatus,
    TunerModelStatus,
    get_layer_status,
    get_model_status,
    get_peft_model,
    inject_adapter_in_model,
)
from lucid.peft.tuners.lora import Linear


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l0 = nn.Linear(4, 4)
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.l2(self.l1(self.l0(x))))


def _peft(device: str) -> PeftModel:
    lucid.manual_seed(0)
    config_a = LoraConfig(r=2, target_modules=["l0", "l1"], init_lora_weights=False)
    config_b = LoraConfig(r=2, target_modules=["l1", "l2"], init_lora_weights=False)
    peft = get_peft_model(_Net().to(device), config_a, adapter_name="a")
    peft.add_adapter("b", config_b)
    return peft


def _layers(peft: PeftModel) -> dict[str, Linear]:
    return {n: m for n, m in peft.base_model.named_modules() if isinstance(m, Linear)}


class TestLayerStatus:
    def test_one_record_per_adapter_layer(self, device: str) -> None:
        peft = _peft(device)
        statuses = get_layer_status(peft)
        assert [s.name for s in statuses] == ["model.l0", "model.l1", "model.l2"]
        assert {s.module_type for s in statuses} == {"lora.Linear"}
        by_name = {s.name: s for s in statuses}
        assert by_name["model.l0"].available_adapters == ["a"]
        assert by_name["model.l1"].available_adapters == ["a", "b"]
        assert by_name["model.l2"].available_adapters == ["b"]

    def test_a_fresh_pair_has_a_active_and_trainable(self, device: str) -> None:
        for status in get_layer_status(_peft(device)):
            assert status.enabled and status.merged_adapters == []
            assert status.active_adapters == ["a"]
            assert status.requires_grad.get("a", True) is True
            assert status.requires_grad.get("b", False) is False  # added frozen
            assert set(status.devices) == set(status.available_adapters)

    def test_the_device_of_each_adapter(self, device: str) -> None:
        for status in get_layer_status(_peft(device)):
            assert all(kinds == [device] for kinds in status.devices.values())

    def test_a_layer_in_a_mixed_state_reads_irregular(self, device: str) -> None:
        peft = _peft(device)
        layer = _layers(peft)["model.l1"]
        layer.lora_A["a"].weight.requires_grad_(False)  # A frozen, B still trains
        status = next(s for s in get_layer_status(peft) if s.name == "model.l1")
        assert status.requires_grad["a"] == "irregular"

    def test_the_modules_to_save_copy_is_a_layer_too(self, device: str) -> None:
        lucid.manual_seed(0)
        config = LoraConfig(r=2, target_modules=["l0"], modules_to_save=["head"])
        peft = get_peft_model(_Net().to(device), config)
        by_name = {s.name: s for s in get_layer_status(peft)}
        head = by_name["model.head"]
        assert head.module_type == "ModulesToSaveWrapper"
        assert head.available_adapters == ["default"] and head.active_adapters == ["default"]
        assert head.merged_adapters == [] and head.requires_grad == {"default": True}
        peft.base_model.disable_adapter_layers()
        assert not {s.name: s for s in get_layer_status(peft)}["model.head"].enabled

    def test_no_adapter_layer_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="No adapter layers found"):
            get_layer_status(nn.Sequential(nn.Linear(2, 2)))

    def test_a_model_adapted_in_place_is_read_directly(self, device: str) -> None:
        net = _Net().to(device)
        inject_adapter_in_model(LoraConfig(r=2, target_modules=["l0"]), net)
        (status,) = get_layer_status(net)
        assert status.name == "l0" and status.active_adapters == ["default"]

    def test_the_peft_model_has_the_methods_too(self, device: str) -> None:
        peft = _peft(device)
        assert peft.get_layer_status() == get_layer_status(peft)
        assert peft.get_model_status() == get_model_status(peft)

    def test_records_are_frozen(self, device: str) -> None:
        status = get_layer_status(_peft(device))[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            status.enabled = False  # type: ignore[misc]
        assert isinstance(status, TunerLayerStatus)


class TestModelStatus:
    def test_a_regular_model(self, device: str) -> None:
        peft = _peft(device)
        status = get_model_status(peft)
        assert isinstance(status, TunerModelStatus)
        assert (status.base_model_type, status.adapter_model_type) == ("_Net", "LoraModel")
        assert status.peft_types == {"a": "LORA", "b": "LORA"}
        assert (status.trainable_params, status.total_params) == peft.get_nb_trainable_parameters()
        assert status.num_adapter_layers == 3
        assert status.enabled is True and status.active_adapters == ["a"]
        assert status.merged_adapters == [] and status.available_adapters == ["a", "b"]
        assert status.requires_grad == {"a": True, "b": False}
        assert status.devices == {"a": [device], "b": [device]}

    def test_switching_follows(self, device: str) -> None:
        peft = _peft(device)
        peft.set_adapter("b")
        status = get_model_status(peft)
        assert status.active_adapters == ["b"]
        assert status.requires_grad == {"a": False, "b": True}

    def test_disabling_one_layer_is_irregular(self, device: str) -> None:
        peft = _peft(device)
        _layers(peft)["model.l0"].enable_adapters(False)
        assert get_model_status(peft).enabled == "irregular"
        peft.base_model.disable_adapter_layers()
        assert get_model_status(peft).enabled is False

    def test_a_different_active_adapter_on_one_layer_is_irregular(self, device: str) -> None:
        peft = _peft(device)
        _layers(peft)["model.l1"]._activate(["a", "b"])
        assert get_model_status(peft).active_adapters == "irregular"

    def test_merged_on_every_layer_that_holds_it(self, device: str) -> None:
        peft = _peft(device)
        peft.base_model.merge_adapter(["a"])
        status = get_model_status(peft)
        assert status.merged_adapters == ["a"]
        peft.base_model.unmerge_adapter()
        assert get_model_status(peft).merged_adapters == []

    def test_merged_on_one_layer_only_is_irregular(self, device: str) -> None:
        peft = _peft(device)
        _layers(peft)["model.l0"].merge(adapter_names=["a"])  # a is also on l1, unmerged
        assert get_model_status(peft).merged_adapters == "irregular"

    def test_requires_grad_that_differs_between_layers_is_irregular(self, device: str) -> None:
        peft = _peft(device)
        _layers(peft)["model.l0"].set_requires_grad("a", False)
        status = get_model_status(peft)
        assert status.requires_grad["a"] == "irregular" and status.requires_grad["b"] is False

    def test_a_bare_model_is_other(self, device: str) -> None:
        net = _Net().to(device)
        inject_adapter_in_model(LoraConfig(r=2, target_modules=["l0"]), net)
        status = get_model_status(net)
        assert (status.base_model_type, status.adapter_model_type) == ("other", "None")
        assert status.peft_types == {} and status.num_adapter_layers == 1
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        assert status.trainable_params == trainable == 16

    def test_no_adapter_layer_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="No adapter layers found"):
            get_model_status(nn.Sequential(nn.Linear(2, 2)))
