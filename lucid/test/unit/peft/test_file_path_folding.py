"""Where an adapter lives in a file when another method's layer stands beneath it.

A layer built on another method's layer sits one ``base_layer`` down in memory (``0.base_layer``);
the adapter file is laid out as a model of the adapter's own method alone (``0.hada_w1_a.weight``).
``_file_path`` takes that step off, and only that step: the wrappers one method nests around one module
(a ``ParamWrapper`` per parameter) are named in the file, ``base_layer`` and all.  These tests build the
stacks by hand, with no host, so they hold the folding alone; ``test_mixed_model_contract`` holds
the files a host writes.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import LoHaConfig, LoraConfig, ShiraConfig
from lucid.peft._slots import _file_path
from lucid.peft._state_dict import _adapter_keys, _file_context
from lucid.peft.tuners import loha, lora, shira
from lucid.peft.tuners._layer import _BaseTunerLayer
from lucid.peft.tuners._files import _FileContext

_TARGETS = ["0"]


def _net() -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 4))


def _lora(base: nn.Module, name: str) -> _BaseTunerLayer:
    return lora.Linear(base, name, r=2, lora_alpha=4)


def _loha(base: nn.Module, name: str) -> _BaseTunerLayer:
    return loha.Linear(base, name, LoHaConfig(target_modules=_TARGETS), r=2, alpha=4)


def _shira(base: nn.Module, name: str) -> _BaseTunerLayer:
    return shira.Linear(base, name, ShiraConfig(r=1, random_seed=3, target_modules=_TARGETS))


class _Gate(nn.Module):
    """A module with two raw parameters, each adapted by its own wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Parameter(lucid.randn(4, 6))
        self.second = nn.Parameter(lucid.randn(4, 6))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        out: lucid.Tensor = x @ self.first.mT + x @ self.second.mT
        return out


class TestWhichStepsAreTaken:
    def test_a_layer_nothing_stands_beneath_is_filed_where_it_is(self) -> None:
        net = _net()
        net[0] = _lora(net[0], "a")
        assert _file_path(net, "0") == "0"

    def test_a_layer_on_another_methods_layer_is_filed_at_the_modules_path(self) -> None:
        net = _net()
        net[0] = _loha(_lora(net[0], "a"), "b")
        assert [_file_path(net, path) for path in ("0", "0.base_layer")] == ["0", "0"]

    def test_every_step_between_methods_is_taken(self) -> None:
        net = _net()
        net[0] = _shira(_loha(_lora(net[0], "a"), "b"), "c")
        paths = ("0", "0.base_layer", "0.base_layer.base_layer")
        assert [_file_path(net, path) for path in paths] == ["0", "0", "0"]

    def test_the_wrappers_one_method_nests_keep_their_step(self) -> None:
        holder = nn.Sequential(_Gate())
        inner = lora.ParamWrapper(holder[0], "a", "first", r=2, lora_alpha=4)
        holder[0] = lora.ParamWrapper(inner, "a", "second", r=2, lora_alpha=4)
        assert _file_path(holder, "0.base_layer") == "0.base_layer"

    def test_a_path_that_is_not_a_slot_stays_as_it_is(self) -> None:
        net = _net()
        net[0] = _lora(net[0], "a")
        assert _file_path(net, "0.base_layer") == "0.base_layer"  # the user's own module


class TestWhatTheFileContextDoesWithIt:
    def test_the_key_of_a_tensor_in_a_layer_standing_on_another_names_the_module(self) -> None:
        net = _net()
        net[0] = _loha(_lora(net[0], "a"), "b")
        context = _FileContext(net, "", LoraConfig(target_modules=_TARGETS), "a")
        assert context.key("0.base_layer", "lora_A") == "0.lora_A"
        assert context.key("0", "hada_w1_a") == "0.hada_w1_a"

    def test_a_prefix_is_kept_and_not_folded_into(self) -> None:
        net = _net()
        net[0] = _loha(_lora(net[0], "a"), "b")
        holder = nn.Sequential()
        holder.model = net
        context = _FileContext(holder, "model.", LoraConfig(target_modules=_TARGETS), "a")
        assert context.key("0.base_layer", "lora_A") == "model.0.lora_A"
        assert context.file_path("0.base_layer") == "0"

    def test_the_layers_are_still_listed_where_they_are_in_memory(self) -> None:
        """A tuner that derives an in-memory key from a layer's path (SHiRA's indices) needs the memory path."""
        net = _net()
        net[0] = _loha(_shira(net[0], "a"), "b")
        context = _FileContext(net, "", ShiraConfig(r=1, target_modules=_TARGETS), "a")
        assert [path for path, _ in context.layers] == ["0", "0.base_layer"]
        assert context.memory_key("0.base_layer", "shira_indices") == "0.base_layer.shira_indices.a"
        assert context.key("0.base_layer", "shira_indices") == "0.shira_indices"


class TestTheKeysOfAnAdapterInAStack:
    def _keys(
        self, net: nn.Sequential, config: LoraConfig | LoHaConfig, name: str
    ) -> dict[str, str]:
        net.peft_config = {name: config}
        return _adapter_keys(_file_context(net, name))

    def test_a_stacked_adapters_file_keys_are_those_of_the_adapter_alone(self) -> None:
        alone = _net()
        alone[0] = _lora(alone[0], "a")
        stacked = _net()
        stacked[0] = _loha(_lora(stacked[0], "a"), "b")
        config = LoraConfig(target_modules=_TARGETS)
        assert sorted(self._keys(alone, config, "a").values()) == sorted(
            self._keys(stacked, config, "a").values()
        )

    def test_the_memory_keys_are_the_stacks(self) -> None:
        stacked = _net()
        stacked[0] = _loha(_lora(stacked[0], "a"), "b")
        keys = self._keys(stacked, LoraConfig(target_modules=_TARGETS), "a")
        assert set(keys) == {
            "0.base_layer.lora_A.a.weight",
            "0.base_layer.lora_B.a.weight",
        }
        assert set(keys.values()) == {"0.lora_A.weight", "0.lora_B.weight"}

    @pytest.mark.parametrize("name", ["a", "b"])
    def test_each_adapter_of_a_stack_is_filed_under_the_module(self, name: str) -> None:
        stacked = _net()
        stacked[0] = _loha(_lora(stacked[0], "a"), "b")
        config: LoraConfig | LoHaConfig = (
            LoraConfig(target_modules=_TARGETS)
            if name == "a"
            else LoHaConfig(target_modules=_TARGETS)
        )
        files = self._keys(stacked, config, name).values()
        assert files and all(key.startswith("0.") and "base_layer" not in key for key in files)
