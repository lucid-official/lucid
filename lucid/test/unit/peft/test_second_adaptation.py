"""Adapting a model that is already adapted: the new adapter joins the old, with the reference's warning.

``get_peft_model`` / ``inject_adapter_in_model`` / a tuner constructor on a
model that holds adapters of the same method adds the new one beside them
(onto the layers they made, and new layers where it reaches more), makes it
the one in use, and answers for all of them.  A name already taken, or
another method's layers, is refused with the model untouched.
"""

import warnings
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    IA3Config,
    LoraConfig,
    LoraModel,
    PeftModel,
    get_peft_model,
    inject_adapter_in_model,
)
from lucid.peft.tuners.lora import Linear

_WARNING = "modify a model with PEFT for a second time"


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


def _config(targets: list[str], **kwargs: object) -> LoraConfig:
    return LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=targets,
        init_lora_weights=False,
        **kwargs,  # type: ignore[arg-type]
    )


def _net(device: str) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3)).to(device)


class TestSecondGetPeftModel:
    def test_the_new_adapter_joins_and_is_in_use(self, device: str) -> None:
        net = _net(device)
        first = get_peft_model(net, _config(["0"]), adapter_name="one")
        with pytest.warns(UserWarning, match=_WARNING):
            second = get_peft_model(net, _config(["0", "2"]), adapter_name="two")
        assert isinstance(second, PeftModel) and second is not first
        assert sorted(second.peft_config) == ["one", "two"]
        assert second.active_adapter == "two"
        layer0, layer2 = net[0], net[2]
        assert isinstance(layer0, Linear) and isinstance(layer2, Linear)
        assert sorted(layer0.lora_A) == ["one", "two"] and sorted(layer2.lora_A) == ["two"]
        # The older adapter is frozen, the newest trains.
        assert not layer0.lora_A["one"].weight.requires_grad
        assert layer0.lora_A["two"].weight.requires_grad

    def test_the_new_adapter_computes_what_it_computes_alone(self, device: str) -> None:
        net = _net(device)
        get_peft_model(net, _config(["0"]), adapter_name="one")
        with pytest.warns(UserWarning, match=_WARNING):
            second = get_peft_model(net, _config(["0"]), adapter_name="two")
        layer = net[0]
        assert isinstance(layer, Linear)
        x = lucid.randn(4, 6, device=device)
        with lucid.no_grad():
            expected = layer.base_layer(x) + x @ layer.get_delta_weight("two").T
            assert _close(net[0](x), expected)
            second.set_adapter("one")
            expected = layer.base_layer(x) + x @ layer.get_delta_weight("one").T
            assert _close(net[0](x), expected)

    def test_the_wrapper_answers_for_every_adapter(self, device: str, tmp_path: Path) -> None:
        net = _net(device)
        get_peft_model(net, _config(["0"]), adapter_name="one")
        with pytest.warns(UserWarning, match=_WARNING):
            second = get_peft_model(net, _config(["0"]), adapter_name="two")
        second.save_pretrained(str(tmp_path))
        assert (tmp_path / "one" / "adapter_model.safetensors").is_file()
        assert (tmp_path / "two" / "adapter_model.safetensors").is_file()
        second.delete_adapter("one")
        assert sorted(second.peft_config) == ["two"]
        assert sorted(net[0].lora_A) == ["two"]

    def test_a_taken_name_is_refused_and_nothing_changes(self, device: str) -> None:
        net = _net(device)
        get_peft_model(net, _config(["0"]))
        before = sorted(net.state_dict())
        with pytest.raises(ValueError, match="'default' already exists"):
            get_peft_model(net, _config(["2"]))
        assert sorted(net.state_dict()) == before and type(net[2]) is nn.Linear

    def test_another_methods_layers_are_refused_and_nothing_changes(self, device: str) -> None:
        net = _net(device)
        get_peft_model(net, _config(["0"]))
        before = sorted(net.state_dict())
        with pytest.raises(ValueError, match="different peft types"):
            get_peft_model(
                net, IA3Config(target_modules=["2"], feedforward_modules=[]), adapter_name="x"
            )
        assert sorted(net.state_dict()) == before

    def test_a_merged_earlier_adapter_is_unmerged_first(self, device: str) -> None:
        net = _net(device)
        first = get_peft_model(net, _config(["0"]), adapter_name="one")
        first.base_model.merge_adapter()
        layer = net[0]
        assert isinstance(layer, Linear) and layer.merged
        with pytest.warns(UserWarning, match=_WARNING):
            get_peft_model(net, _config(["0"]), adapter_name="two")
        assert not layer.merged and layer.merged_adapters == []
        x = lucid.randn(2, 6, device=device)
        with lucid.no_grad():
            expected = layer.base_layer(x) + x @ layer.get_delta_weight("two").T
            assert _close(net[0](x), expected)

    def test_unload_gives_back_the_original_requires_grad(self, device: str) -> None:
        net = _net(device)
        first = get_peft_model(net, _config(["0"]), adapter_name="one")
        with pytest.warns(UserWarning, match=_WARNING):
            second = get_peft_model(net, _config(["2"]), adapter_name="two")
        assert not any(p.requires_grad for n, p in net.named_parameters() if "base_layer" in n)
        second.unload()
        assert all(p.requires_grad for p in net.parameters())
        assert type(net[0]) is nn.Linear and type(net[2]) is nn.Linear
        assert "peft_config" not in net.__dict__ and not hasattr(net, "_peft_requires_grad_before")
        del first

    def test_a_head_in_modules_to_save_is_shared_and_copied(self, device: str) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = nn.Linear(6, 6)
                self.head = nn.Linear(6, 2)

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return self.head(self.body(x))

        lucid.manual_seed(0)
        net = Net().to(device)
        get_peft_model(net, _config(["body"], modules_to_save=["head"]), adapter_name="one")
        with pytest.warns(UserWarning, match=_WARNING):
            second = get_peft_model(
                net, _config(["body"], modules_to_save=["head"]), adapter_name="two"
            )
        assert sorted(net.head.modules_to_save) == ["one", "two"]
        assert net.head.active_adapter == "two"
        assert sorted(second.peft_config) == ["one", "two"]


class TestOtherEntryPoints:
    def test_inject_adapter_in_model_twice(self, device: str) -> None:
        net = _net(device)
        inject_adapter_in_model(_config(["0"]), net, "one")
        with pytest.warns(UserWarning, match=_WARNING):
            assert inject_adapter_in_model(_config(["2"]), net, "two") is net
        assert sorted(net.peft_config) == ["one", "two"]
        layer = net[0]
        assert isinstance(layer, Linear) and layer.active_adapters == ["two"]

    def test_a_tuner_constructor_adopts_too(self, device: str) -> None:
        net = _net(device)
        LoraModel(net, _config(["0"]), adapter_name="one")
        with pytest.warns(UserWarning, match=_WARNING):
            tuner = LoraModel(net, _config(["2"]), adapter_name="two")
        assert tuner.active_adapters == ["two"] and sorted(tuner.peft_config) == ["one", "two"]

    def test_the_first_adaptation_does_not_warn(self, device: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            get_peft_model(_net(device), _config(["0"]))
