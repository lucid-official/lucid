"""``_FileContext.buffer_keys``: how a layer's frozen, saved tensor gets into an adapter file.

A tensor the adapter file carries but no slot table holds (AdaMSS's projection) is a buffer beside the
layer's tables.  ``buffer_keys`` maps it to the key the reference writes: the layer's path and the table's
name, the adapter's name taken out, under whatever prefix the model is wrapped in.  A table that is not
persistent is not in the file, and neither is a layer that holds nothing of the adapter.
"""

import lucid.nn as nn
from lucid.peft import AdamssConfig, get_peft_model, inject_adapter_in_model
from lucid.peft._state_dict import _adapter_keys, _file_context


def _config(*targets: str) -> AdamssConfig:
    return AdamssConfig(r=3, num_subspaces=2, target_modules=list(targets))


def _net() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 6))


def test_a_persistent_table_maps_each_layers_buffer_to_its_file_key() -> None:
    peft = get_peft_model(_net(), _config("0", "2"))
    keys = _file_context(peft, "default").buffer_keys("adamss_newB")
    assert keys == {
        f"base_model.model.{index}.adamss_newB.default": f"base_model.model.{index}.adamss_newB"
        for index in (0, 2)
    }


def test_a_table_that_is_not_saved_has_no_keys() -> None:
    peft = get_peft_model(_net(), _config("0", "2"))
    assert _file_context(peft, "default").buffer_keys("adamss_rows") == {}


def test_a_layer_that_holds_nothing_of_the_adapter_is_left_out() -> None:
    peft = get_peft_model(_net(), _config("0", "2"))
    peft.add_adapter("b", _config("0"))
    context = _file_context(peft, "b")
    assert context.buffer_keys("adamss_newB") == {
        "base_model.model.0.adamss_newB.b": "base_model.model.0.adamss_newB"
    }


def test_a_table_the_layers_lack_has_no_keys() -> None:
    peft = get_peft_model(_net(), _config("0"))
    context = _file_context(peft, "default")
    assert context.buffer_keys("not_a_table") == {}
    assert context.buffer_keys("adamss_A") == {}  # a ModuleDict of parameters is not a buffer table


def test_a_model_adapted_in_place_has_no_prefix() -> None:
    net = inject_adapter_in_model(_config("2"), _net())
    keys = _file_context(net, "default").buffer_keys("adamss_newB")
    assert keys == {"2.adamss_newB.default": "2.adamss_newB"}


def test_the_tuner_puts_them_among_the_adapters_file_keys() -> None:
    peft = get_peft_model(_net(), _config("0", "2"))
    keys = _adapter_keys(_file_context(peft, "default"))
    assert keys["base_model.model.0.adamss_newB.default"] == "base_model.model.0.adamss_newB"
    assert keys["base_model.model.2.adamss_A.default.0"] == "base_model.model.2.adamss_A.0"
