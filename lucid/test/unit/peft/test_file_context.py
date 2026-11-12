"""``_FileContext.buffer_keys``: how a layer's frozen, saved tensor gets into an adapter file.

A tensor the adapter file carries but no slot table holds (AdaMSS's projection) is a buffer beside the
layer's tables.  ``buffer_keys`` maps it to the key the reference writes: the layer's path and the table's
name, the adapter's name taken out, under whatever prefix the model is wrapped in.  A table that is not
persistent is not in the file, and neither is a layer that holds nothing of the adapter.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    AdamssConfig,
    DeloraConfig,
    FrodConfig,
    PeftConfig,
    PveraConfig,
    RandLoraConfig,
    ShiraConfig,
    SupertuningConfig,
    TinyLoraConfig,
    UniLoraConfig,
    VeraConfig,
    get_peft_model,
    inject_adapter_in_model,
)
from lucid.peft._state_dict import _adapter_keys, _file_context
from lucid.peft.tuners._shared import _BufferDict


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


# ── persistence is per buffer ─────────────────────────────────────────────


def test_a_buffer_is_saved_as_the_table_says_unless_entered_with_a_choice() -> None:
    table = _BufferDict(persistent=True)
    table["a"] = lucid.zeros(2)
    table.register("b", lucid.zeros(2), persistent=False)
    assert (table.persistent, table.persists("a"), table.persists("b")) == (True, True, False)
    assert list(table.state_dict()) == ["a"]


def test_a_buffer_entered_again_takes_the_new_choice() -> None:
    table = _BufferDict(persistent=False)
    table["a"] = lucid.zeros(2)
    assert not table.persists("a")
    table.register("a", lucid.ones(2), persistent=True)
    assert table.persists("a") and list(table.state_dict()) == ["a"]
    del table["a"]
    assert not table.persists("a") and list(table.state_dict()) == []


def test_a_name_the_table_does_not_hold_is_not_persisted() -> None:
    assert not _BufferDict(persistent=True).persists("missing")


def test_buffer_keys_follow_the_adapters_own_choice() -> None:
    first = UniLoraConfig(r=2, theta_d_length=8, target_modules=["0", "2"], save_indices=False)
    peft = get_peft_model(_net(), first, "a")
    peft.add_adapter(
        "b", UniLoraConfig(r=2, theta_d_length=8, target_modules=["0", "2"], save_indices=True)
    )
    assert _file_context(peft, "a").buffer_keys("unilora_indices_A") == {}
    assert _file_context(peft, "b").buffer_keys("unilora_indices_A") == {
        f"base_model.model.{index}.unilora_indices_A.b": f"base_model.model.{index}.unilora_indices_A"
        for index in (0, 2)
    }


_TARGETS = ["0", "2"]

#: Every kind of ``_BufferDict`` a method holds, as a config that builds it.
_CONTAINER_CONFIGS: dict[str, Callable[[], PeftConfig]] = {
    "shira": lambda: ShiraConfig(r=1, target_modules=_TARGETS),
    "supertuning": lambda: SupertuningConfig(sparsity=0.5, target_modules=_TARGETS),
    "adamss": lambda: _config(*_TARGETS),
    "delora": lambda: DeloraConfig(r=2, target_modules=_TARGETS),
    "tinylora": lambda: TinyLoraConfig(r=2, u=3, target_modules=_TARGETS),
    "tinylora_unsaved_projection": lambda: TinyLoraConfig(
        r=2, u=3, target_modules=_TARGETS, save_projection=False
    ),
    "vera": lambda: VeraConfig(r=4, target_modules=_TARGETS),
    "vera_unsaved": lambda: VeraConfig(r=4, target_modules=_TARGETS, save_projection=False),
    "pvera": lambda: PveraConfig(r=4, target_modules=_TARGETS),
    "randlora": lambda: RandLoraConfig(r=4, target_modules=_TARGETS),
    "frod": lambda: FrodConfig(target_modules=_TARGETS),
    "unilora": lambda: UniLoraConfig(r=2, theta_d_length=8, target_modules=_TARGETS),
    "unilora_saved": lambda: UniLoraConfig(
        r=2, theta_d_length=8, target_modules=_TARGETS, save_indices=True
    ),
}


@pytest.mark.filterwarnings("ignore:Specified to not")
@pytest.mark.parametrize("name", list(_CONTAINER_CONFIGS))
def test_a_method_whose_adapters_do_not_choose_keeps_every_buffer_as_its_table_says(
    name: str,
) -> None:
    """Persistence per buffer refines the table's: where the config fixes it, the two agree."""
    peft = get_peft_model(_net(), _CONTAINER_CONFIGS[name]())
    tables = [(path, m) for path, m in peft.named_modules() if isinstance(m, _BufferDict)]
    assert tables
    for path, table in tables:
        for adapter in table:
            assert table.persists(adapter) == table.persistent, f"{path}.{adapter}"
