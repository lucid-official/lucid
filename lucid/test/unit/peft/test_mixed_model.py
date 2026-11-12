"""``PeftMixedModel``: the public wrapper of a model that holds adapters of several methods.

``test_mixed_model_contract`` holds the host (``MixedModel``) to the maths of a stack.  This holds the
wrapper to its own promises, on the CPU and on Metal where tensors are computed:

* **Its API and its refusals**: a method outside the mapping, a model that holds adapters, a bad
  name and the options the library cannot honour are refused with the model as it was; a failed add
  or load is rolled back; ``set_adapter`` and ``delete_adapter`` take a name or a list, check every
  name before changing anything, and say what is available.
* **What it folds**: ``merge_and_unload`` is the stacked forward pass, partial by ``adapter_names``;
  ``unload`` gives the model back with its ``requires_grad``.
* **What it does not do**: it saves nothing (``NotImplementedError``, nothing written) and reports
  no layer status (``TypeError``), as the reference's mixed model does not.
* **The way in**: ``get_peft_model(mixed=True)`` builds it and stacks a second method; the files a
  single-method ``PeftModel`` wrote load into it, before and after other methods are stacked on
  their modules, with the output they were saved with.
* **The mapping** of the methods it holds is read-only and is the registry the host reads.
"""

from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import pytest

import lucid
import lucid.nn as nn
import lucid.peft as peft
from lucid.peft import (
    PEFT_TYPE_TO_MIXED_MODEL_MAPPING,
    PEFT_TYPE_TO_TUNER_MAPPING,
    IA3Config,
    LoHaConfig,
    LoraConfig,
    PeftConfig,
    PeftMixedModel,
    PeftModel,
    PeftType,
    PromptTuningConfig,
    VeraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from lucid.peft.tuners import MixedModel
from lucid.peft.tuners._base import _TUNERS
from lucid.peft.tuners._layer import _tuner_layers
from lucid.peft.tuners._tuner import _MIXED_TUNERS
from lucid.test._default_device import metal_available
from lucid.test.unit.peft._tuner_recipes import RECIPES, magnitude, perturb_adapter, same, span

_F32 = lucid.float32
#: A float32 sum of a few products of order one; Metal sums in another order, to ~1e-6.
_TOLERANCE = 1e-5
#: What a layer's own adapter must move the output by (the contract's floor, 5000 times the tolerance).
_EFFECT = 0.05


def _net(device: str) -> nn.Module:
    """The two-linear-layer model every recipe adapts, at sites ``"0"`` and ``"2"``."""
    return RECIPES[PeftType.LORA].base(_F32, device)


def _call(model: nn.Module, device: str) -> lucid.Tensor:
    return RECIPES[PeftType.LORA].call(model, _F32, device)


def _lora(**fields: object) -> LoraConfig:
    return LoraConfig(r=2, target_modules=["0", "2"], **fields)


def _loha() -> LoHaConfig:
    return LoHaConfig(r=2, target_modules=["0", "2"])


def _ia3() -> IA3Config:
    return IA3Config(target_modules=["0", "2"], feedforward_modules=["2"])


def _move(model: nn.Module, name: str, seed: int) -> None:
    """Move ``name``'s tensors well off their start, three times (a LoRA ``B`` starts at zero)."""
    for step in range(3):
        perturb_adapter(model, name, 10 * seed + step)


class _Stack(NamedTuple):
    """A LoRA adapter with a LoHa layer on it, both moved off their start and in use; the base's output."""

    net: nn.Module
    mixed: PeftMixedModel
    clean: lucid.Tensor


def _stack(device: str) -> _Stack:
    net = _net(device)
    clean = _call(net, device)
    mixed = PeftMixedModel(net, _lora(), "lora")
    mixed.add_adapter("loha", _loha())
    _move(net, "lora", 1)
    _move(net, "loha", 2)
    mixed.set_adapter(["lora", "loha"])
    return _Stack(net, mixed, clean)


class _Structure(NamedTuple):
    """Everything a refused call must leave alone."""

    tree: list[tuple[str, str]]
    state: dict[str, lucid.Tensor]
    grads: dict[str, bool]
    adapters: list[str]
    active: list[str]


def _structure(net: nn.Module, mixed: PeftMixedModel | None = None) -> _Structure:
    return _Structure(
        [(name, type(module).__name__) for name, module in net.named_modules()],
        {k: v.detach().clone() for k, v in net.state_dict().items()},
        {name: p.requires_grad for name, p in net.named_parameters()},
        sorted(mixed.peft_config) if mixed is not None else [],
        mixed.active_adapters if mixed is not None else [],
    )


def _unchanged(net: nn.Module, mixed: PeftMixedModel | None, before: _Structure) -> None:
    after = _structure(net, mixed)
    assert after.tree == before.tree
    assert after.grads == before.grads
    assert (after.adapters, after.active) == (before.adapters, before.active)
    assert after.state.keys() == before.state.keys()
    assert all(same(after.state[k], v) for k, v in before.state.items())


def _saved(config: PeftConfig, directory: Path, device: str, seed: int) -> lucid.Tensor:
    """Save a single-method ``PeftModel`` of ``config`` (moved off its start) in ``directory``; its output."""
    single = get_peft_model(_net(device), config)
    _move(single, "default", seed)
    single.save_pretrained(directory)
    with lucid.no_grad():
        return _call(single, device)


def _only(net: nn.Module, mixed: PeftMixedModel, name: str, device: str) -> lucid.Tensor:
    mixed.set_adapter([name])
    with lucid.no_grad():
        return _call(net, device)


def _held(net: nn.Module, name: str) -> list[lucid.Tensor]:
    return [p for _, layer in _tuner_layers(net) for p in layer._adapter_parameters(name)]


# ── what it is ────────────────────────────────────────────────────────────


class TestWhatItIs:
    def test_it_holds_a_mixed_model_and_is_no_peft_model(self) -> None:
        net = _net("cpu")
        mixed = PeftMixedModel(net, _lora(), "lora")
        assert not isinstance(mixed, PeftModel)
        assert isinstance(mixed.base_model, MixedModel)
        assert mixed.base_model.model is net

    def test_its_state_dict_reads_as_a_peft_models_does(self) -> None:
        stack = _stack("cpu")
        keys = list(stack.mixed.state_dict())
        assert keys and all(k.startswith("base_model.model.") for k in keys)
        assert "base_model.model.0.base_layer.base_layer.weight" in keys

    def test_it_reads_the_host(self) -> None:
        stack = _stack("cpu")
        host = stack.mixed.base_model
        assert stack.mixed.peft_config is host.peft_config
        assert stack.mixed.active_adapter == host.active_adapter == ["lora", "loha"]
        assert stack.mixed.active_adapters == ["lora", "loha"]

    def test_the_default_adapter_name_is_default(self) -> None:
        assert list(PeftMixedModel(_net("cpu"), _lora()).peft_config) == ["default"]

    def test_it_is_exported_with_its_mapping(self) -> None:
        assert peft.PeftMixedModel is PeftMixedModel
        assert "PeftMixedModel" in peft.__all__
        assert "PEFT_TYPE_TO_MIXED_MODEL_MAPPING" in peft.__all__


# ── what it refuses ───────────────────────────────────────────────────────


class TestWhatItRefuses:
    @pytest.mark.parametrize(
        "config", [VeraConfig(r=2, target_modules=["0"]), PromptTuningConfig(num_virtual_tokens=2)]
    )
    def test_a_method_outside_the_mapping_is_refused_as_the_first_adapter(
        self, config: PeftConfig
    ) -> None:
        net = _net("cpu")
        before = _structure(net)
        with pytest.raises(ValueError, match="is not compatible with the `PeftMixedModel`"):
            PeftMixedModel(net, config)
        _unchanged(net, None, before)

    def test_and_as_a_later_one_with_the_model_as_it_was(self) -> None:
        stack = _stack("cpu")
        before = _structure(stack.net, stack.mixed)
        with pytest.raises(ValueError, match="Compatible types are: ADALORA, BEFT, HIRA"):
            stack.mixed.add_adapter("x", VeraConfig(r=2, target_modules=["0", "2"]))
        _unchanged(stack.net, stack.mixed, before)

    def test_something_that_is_no_config_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="config must be a PeftConfig"):
            PeftMixedModel(_net("cpu"), "lora")  # a str where the config goes
        stack = _stack("cpu")
        with pytest.raises(TypeError, match="peft_config must be an adapter config"):
            stack.mixed.add_adapter("x", "loha")

    def test_a_model_that_holds_adapters_is_refused(self) -> None:
        net = _net("cpu")
        get_peft_model(net, _lora())
        with pytest.raises(ValueError, match="already holds adapters"):
            PeftMixedModel(net, _loha())

    def test_a_name_that_is_taken_is_refused(self) -> None:
        stack = _stack("cpu")
        before = _structure(stack.net, stack.mixed)
        with pytest.raises(ValueError, match="'lora' already exists"):
            stack.mixed.add_adapter("lora", _loha())
        _unchanged(stack.net, stack.mixed, before)

    @pytest.mark.parametrize("option", ["autocast_adapter_dtype", "low_cpu_mem_usage"])
    def test_an_option_the_library_cannot_honour_is_refused_by_value(self, option: str) -> None:
        stack = _stack("cpu")
        before = _structure(stack.net, stack.mixed)
        value = option == "low_cpu_mem_usage"  # the opposite of each one's default
        with pytest.raises(NotImplementedError, match=f"PeftMixedModel.add_adapter: `{option}"):
            stack.mixed.add_adapter("x", _ia3(), **{option: value})
        _unchanged(stack.net, stack.mixed, before)

    def test_an_adapter_that_cannot_stand_where_it_lands_is_rolled_back(self) -> None:
        net = _net("cpu")
        mixed = PeftMixedModel(net, _loha(), "loha")
        before = _structure(net, mixed)
        with pytest.raises(ValueError, match="DoRA"):
            mixed.add_adapter("dora", _lora(use_dora=True))  # a DoRA layer reads the weight beneath
        _unchanged(net, mixed, before)

    def test_a_load_that_does_not_fit_leaves_no_adapter_and_the_same_computation(
        self, tmp_path: Path
    ) -> None:
        _saved(_lora(), tmp_path, "cpu", seed=1)
        wide = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 5))
        mixed = PeftMixedModel(wide, _loha(), "loha")
        flags = [p.requires_grad for p in wide.parameters()]
        with lucid.no_grad():
            wanted = _call(wide, "cpu")
        with pytest.raises(ValueError, match="do not fit this model"):
            mixed.load_adapter(tmp_path, "lora")
        assert (sorted(mixed.peft_config), mixed.active_adapters) == (["loha"], ["loha"])
        assert not _held(wide, "lora")
        assert [p.requires_grad for p in wide.parameters()] == flags
        with lucid.no_grad():
            assert same(_call(wide, "cpu"), wanted)

    @pytest.mark.xfail(
        strict=True, reason="LCD-972: a failed load of a new adapter leaves the layers it added"
    )
    def test_a_load_that_does_not_fit_leaves_the_tree_as_it_was(self, tmp_path: Path) -> None:
        _saved(_lora(), tmp_path, "cpu", seed=1)
        wide = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 5))
        mixed = PeftMixedModel(wide, _loha(), "loha")
        before = _structure(wide, mixed)
        with pytest.raises(ValueError, match="do not fit this model"):
            mixed.load_adapter(tmp_path, "lora")
        _unchanged(wide, mixed, before)

    def test_a_from_pretrained_that_does_not_fit_leaves_the_model(self, tmp_path: Path) -> None:
        _saved(_lora(), tmp_path, "cpu", seed=1)
        wide = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 5))
        before = _structure(wide)
        with pytest.raises(ValueError, match="do not fit this model"):
            PeftMixedModel.from_pretrained(wide, tmp_path)
        _unchanged(wide, None, before)

    def test_the_layer_status_is_not_reported(self) -> None:
        mixed = _stack("cpu").mixed
        with pytest.raises(TypeError, match="get_layer_status is not supported for PeftMixedModel"):
            mixed.get_layer_status()
        with pytest.raises(TypeError, match="get_model_status is not supported for PeftMixedModel"):
            mixed.get_model_status()

    def test_it_is_not_saved_and_nothing_is_written(self, tmp_path: Path) -> None:
        mixed = _stack("cpu").mixed
        for call in (
            lambda: mixed.save_pretrained(tmp_path),
            lambda: mixed.save_pretrained(
                tmp_path, safe_serialization=True, selected_adapters=["lora"]
            ),
            lambda: mixed.save_pretrained(tmp_path, anything="goes"),
        ):
            with pytest.raises(NotImplementedError, match=r"PeftMixedModel.save_pretrained\(\)"):
                call()
        assert list(tmp_path.iterdir()) == []

    def test_there_is_no_model_card(self, tmp_path: Path) -> None:
        mixed = _stack("cpu").mixed
        with pytest.raises(NotImplementedError, match="create_or_update_model_card"):
            mixed.create_or_update_model_card(str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_weighted_adapters_are_refused_by_the_host_it_forwards_to(self) -> None:
        stack = _stack("cpu")
        before = _structure(stack.net, stack.mixed)
        with pytest.raises(NotImplementedError, match="add_weighted_adapter is not supported"):
            stack.mixed.add_weighted_adapter(["lora", "loha"], [0.5, 0.5], "mix")
        _unchanged(stack.net, stack.mixed, before)

    def test_the_host_methods_the_reference_lacks_are_not_on_the_wrapper(self) -> None:
        mixed = _stack("cpu").mixed
        assert not any(hasattr(PeftMixedModel, n) for n in ("merge_adapter", "unmerge_adapter"))
        assert callable(mixed.base_model.merge_adapter)  # still there on the host


# ── choosing the adapters in use ──────────────────────────────────────────


class TestSwitching:
    def test_a_name_or_a_list_names_the_adapters_in_use(self, device: str) -> None:
        stack = _stack(device)
        both = _call(stack.net, device)
        stack.mixed.set_adapter("lora")
        assert (stack.mixed.active_adapter, stack.mixed.active_adapters) == ("lora", ["lora"])
        lora_only = _call(stack.net, device)
        stack.mixed.set_adapter(["loha", "lora"])
        assert stack.mixed.active_adapters == ["loha", "lora"]
        assert span(_call(stack.net, device), both) < _TOLERANCE
        assert span(lora_only, both) > _EFFECT
        assert span(lora_only, stack.clean) > _EFFECT

    def test_the_adapters_named_train_and_the_rest_are_frozen(self) -> None:
        stack = _stack("cpu")
        stack.mixed.set_adapter(["loha"])
        assert all(p.requires_grad for p in _held(stack.net, "loha"))
        assert not any(p.requires_grad for p in _held(stack.net, "lora"))
        assert not any(
            p.requires_grad for p in stack.net.get_submodule("0").base_layer.base_layer.parameters()
        )

    def test_inference_mode_trains_none(self) -> None:
        stack = _stack("cpu")
        stack.mixed.set_adapter(["lora", "loha"], inference_mode=True)
        assert not any(p.requires_grad for p in stack.net.parameters())
        stack.mixed.set_adapter(["lora", "loha"])
        assert all(p.requires_grad for n in ("lora", "loha") for p in _held(stack.net, n))

    def test_a_name_the_model_lacks_is_refused_naming_what_it_has(self) -> None:
        stack = _stack("cpu")
        before = _structure(stack.net, stack.mixed)
        with pytest.raises(
            ValueError,
            match=r"Adapter\(s\) \['nope'\] not found, available adapters: \['loha', 'lora'\]",
        ):
            stack.mixed.set_adapter(["lora", "nope"])
        _unchanged(stack.net, stack.mixed, before)

    def test_a_new_adapter_is_frozen_and_not_in_use_until_named(self, device: str) -> None:
        stack = _stack(device)
        with lucid.no_grad():
            before = _call(stack.net, device)
        stack.mixed.add_adapter("ia3", _ia3())
        assert stack.mixed.active_adapters == ["lora", "loha"]
        assert not any(p.requires_grad for p in _held(stack.net, "ia3"))
        with lucid.no_grad():
            assert span(_call(stack.net, device), before) < _TOLERANCE

    def test_disabling_the_adapters_runs_the_base_and_nests(self, device: str) -> None:
        stack = _stack(device)
        with lucid.no_grad():
            adapted = _call(stack.net, device)
            with stack.mixed.disable_adapter():
                assert same(_call(stack.net, device), stack.clean)
                with stack.mixed.disable_adapter():
                    assert same(_call(stack.net, device), stack.clean)
                assert same(_call(stack.net, device), stack.clean)  # the inner exit leaves it off
            assert span(_call(stack.net, device), adapted) < _TOLERANCE

    def test_the_adapters_come_back_after_an_exception(self, device: str) -> None:
        stack = _stack(device)
        with lucid.no_grad():
            adapted = _call(stack.net, device)
        with pytest.raises(RuntimeError, match="boom"), stack.mixed.disable_adapter():
            raise RuntimeError("boom")
        with lucid.no_grad():
            assert span(_call(stack.net, device), adapted) < _TOLERANCE

    @pytest.mark.skipif(not metal_available(), reason="compiles for Metal")
    def test_a_compiled_wrapper_follows_the_adapters_in_use(self) -> None:
        stack = _stack("metal")
        stack.mixed.eval()
        compiled = lucid.compile(stack.mixed)
        x = lucid.randn(3, 6, device="metal")
        outputs = []
        for active in (["lora", "loha"], ["lora"], ["loha"]):
            stack.mixed.set_adapter(active)
            with lucid.no_grad():
                got, want = compiled(x), stack.mixed(x)
            assert span(got, want) < _TOLERANCE * max(1.0, magnitude(want)), active
            outputs.append(got)
        assert span(outputs[0], outputs[1]) > _EFFECT and span(outputs[1], outputs[2]) > _EFFECT
        assert not compiled.cache_info()["eager_only"], "the compiled model fell back to eager"


# ── deleting ──────────────────────────────────────────────────────────────


class TestDeleting:
    def test_a_name_or_a_list_removes_the_adapters(self, device: str) -> None:
        stack = _stack(device)
        stack.mixed.add_adapter("ia3", _ia3())
        stack.mixed.delete_adapter("ia3")
        assert sorted(stack.mixed.peft_config) == ["loha", "lora"]
        stack.mixed.delete_adapter(["loha", "lora", "loha"])  # a name twice is deleted once
        assert stack.mixed.peft_config == {}
        with lucid.no_grad():
            assert span(_call(stack.net, device), stack.clean) < _TOLERANCE

    def test_what_stays_still_computes_what_it_computed(self, device: str) -> None:
        stack = _stack(device)
        wanted = _only(stack.net, stack.mixed, "lora", device)
        stack.mixed.set_adapter(["lora", "loha"])
        stack.mixed.delete_adapter(["loha"])
        assert stack.mixed.active_adapters == ["lora"]
        with lucid.no_grad():
            assert span(_call(stack.net, device), wanted) < _TOLERANCE

    def test_one_unknown_name_deletes_nothing(self) -> None:
        stack = _stack("cpu")
        before = _structure(stack.net, stack.mixed)
        with pytest.raises(
            ValueError,
            match=r"Adapter\(s\) \['nope'\] not found, available adapters: \['loha', 'lora'\]",
        ):
            stack.mixed.delete_adapter(["loha", "nope"])
        _unchanged(stack.net, stack.mixed, before)

    @pytest.mark.parametrize("names", [["lora"], ["loha", "lora"], "loha"])
    def test_a_merged_adapter_is_not_deleted_and_nothing_is_lost(
        self, names: str | list[str]
    ) -> None:
        stack = _stack("cpu")
        stack.mixed.base_model.merge_adapter()
        before = _structure(stack.net, stack.mixed)
        with pytest.raises(
            ValueError, match=r"Cannot delete adapter\(s\) \[.*\] while they are merged"
        ):
            stack.mixed.delete_adapter(names)
        _unchanged(stack.net, stack.mixed, before)

    def test_a_merged_adapter_late_in_the_list_leaves_the_earlier_ones_held(self) -> None:
        stack = _stack("cpu")
        stack.mixed.base_model.merge_adapter(["lora"])  # the lower layer alone; the LoHa stays free
        before = _structure(stack.net, stack.mixed)
        with pytest.raises(ValueError, match=r"Cannot delete adapter\(s\) \['lora'\] while they"):
            stack.mixed.delete_adapter(["loha", "lora"])
        _unchanged(stack.net, stack.mixed, before)
        assert sorted(stack.mixed.peft_config) == ["loha", "lora"]

    def test_removing_the_adapter_in_use_hands_over_to_the_first_other_with_a_warning(self) -> None:
        stack = _stack("cpu")
        stack.mixed.add_adapter("ia3", _ia3())
        stack.mixed.set_adapter("lora")
        with pytest.warns(UserWarning, match="Setting active adapter to ia3"):
            stack.mixed.delete_adapter("lora")
        assert stack.mixed.active_adapters == ["ia3"]


# ── unloading and merging ─────────────────────────────────────────────────


class TestUnloading:
    def test_unload_gives_the_model_back_as_it_was(self, device: str) -> None:
        net = _net(device)
        grads = {n: p.requires_grad for n, p in net.named_parameters()}
        clean = _call(net, device)
        mixed = PeftMixedModel(net, _lora(), "lora")
        mixed.add_adapter("loha", _loha())
        mixed.set_adapter(["lora", "loha"])
        assert not all(p.requires_grad for p in net.parameters())
        assert mixed.unload() is net
        assert all(type(net.get_submodule(site)) is nn.Linear for site in ("0", "2"))
        assert {n: p.requires_grad for n, p in net.named_parameters()} == grads
        assert same(_call(net, device), clean)

    @pytest.mark.parametrize(
        "call", [lambda m: m.unload(1), lambda m: m.unload(adapter_names=["a"])]
    )
    def test_the_reference_s_catch_all_arguments_are_refused_here(
        self, call: Callable[[PeftMixedModel], nn.Module]
    ) -> None:
        stack = _stack("cpu")
        before = _structure(stack.net, stack.mixed)
        with pytest.raises(TypeError, match=r"unload\(\) takes no arguments"):
            call(stack.mixed)
        _unchanged(stack.net, stack.mixed, before)


class TestMerging:
    def test_merge_and_unload_is_the_stacked_forward_pass(self, device: str) -> None:
        stack = _stack(device)
        with lucid.no_grad():
            stacked = _call(stack.net, device)
        plain = stack.mixed.merge_and_unload()
        assert all(type(plain.get_submodule(site)) is nn.Linear for site in ("0", "2"))
        with lucid.no_grad():
            assert span(_call(plain, device), stacked) < _TOLERANCE
        assert span(stacked, stack.clean) > _EFFECT

    def test_safe_merge_and_a_progress_bar_are_accepted_by_position_or_name(
        self, device: str
    ) -> None:
        stack = _stack(device)
        with lucid.no_grad():
            stacked = _call(stack.net, device)
        plain = stack.mixed.merge_and_unload(True, safe_merge=True)
        with lucid.no_grad():
            assert span(_call(plain, device), stacked) < _TOLERANCE

    def test_adapter_names_fold_those_and_drop_the_rest_with_their_layers(
        self, device: str
    ) -> None:
        stack = _stack(device)
        wanted = _only(stack.net, stack.mixed, "lora", device)
        stack.mixed.set_adapter(["lora", "loha"])
        plain = stack.mixed.merge_and_unload(adapter_names=["lora"])
        assert all(type(plain.get_submodule(site)) is nn.Linear for site in ("0", "2"))
        with lucid.no_grad():
            assert span(_call(plain, device), wanted) < _TOLERANCE
            assert span(_call(plain, device), stack.clean) > _EFFECT

    @pytest.mark.parametrize(
        "call, message",
        [
            (
                lambda m: m.merge_and_unload(False, False, None, 1),
                "takes at most 3 more positional",
            ),
            (lambda m: m.merge_and_unload(False, progressbar=True), "multiple values for argument"),
            (lambda m: m.merge_and_unload(progressbar=1), "`progressbar` must be a bool"),
            (
                lambda m: m.merge_and_unload(adapter_names="lora"),
                "adapter_names must be a sequence",
            ),
            (lambda m: m.merge_and_unload(nope=1), "unexpected keyword arguments"),
        ],
    )
    def test_arguments_are_checked_before_anything_is_merged(
        self, call: Callable[[PeftMixedModel], nn.Module], message: str
    ) -> None:
        stack = _stack("cpu")
        before = _structure(stack.net, stack.mixed)
        with pytest.raises(TypeError, match=message):
            call(stack.mixed)
        _unchanged(stack.net, stack.mixed, before)

    def test_an_adapter_the_model_lacks_is_refused_before_a_weight_is_written(self) -> None:
        stack = _stack("cpu")
        before = _structure(stack.net, stack.mixed)
        with pytest.raises(ValueError, match="nope"):
            stack.mixed.merge_and_unload(adapter_names=["nope"])
        _unchanged(stack.net, stack.mixed, before)


# ── the way in ────────────────────────────────────────────────────────────


class TestGetPeftModel:
    def test_it_returns_a_mixed_model_that_stacks_a_second_method(self, device: str) -> None:
        net = _net(device)
        mixed = get_peft_model(net, _lora(), mixed=True)
        assert type(mixed) is PeftMixedModel
        mixed.add_adapter("loha", _loha())
        _move(net, "default", 1)
        _move(net, "loha", 2)
        mixed.set_adapter(["default", "loha"])
        assert type(net.get_submodule("0")).__module__.endswith("loha._layer")
        assert type(net.get_submodule("0").base_layer).__module__.endswith("lora._layer")
        with lucid.no_grad():
            stacked = _call(net, device)
        assert span(stacked, _call(_net(device), device)) > _EFFECT

    def test_the_stacked_output_is_the_hosts(self, device: str) -> None:
        net = _net(device)
        mixed = get_peft_model(net, _lora(), mixed=True)
        mixed.add_adapter("loha", _loha())
        _move(net, "default", 1)
        _move(net, "loha", 2)
        mixed.set_adapter(["default", "loha"])
        twin = _net(device)
        host = MixedModel(twin, _lora(), "default")
        host.add_adapter("loha", _loha())
        for name in ("default", "loha"):
            keys = get_peft_model_state_dict(mixed, adapter_name=name)
            filed = {k.replace("base_model.model.", "model.", 1): v for k, v in keys.items()}
            peft.set_peft_model_state_dict(host, filed, adapter_name=name)
        host.set_adapter(["default", "loha"])
        with lucid.no_grad():
            assert span(_call(net, device), _call(twin, device)) < _TOLERANCE
            assert span(_call(twin, device), _call(_net(device), device)) > _EFFECT

    def test_the_config_is_stored_as_get_peft_model_stores_it(self) -> None:
        net = _net("cpu")
        net.name_or_path = "tiny-mlp"
        config = _lora()
        mixed = get_peft_model(net, config, mixed=True, revision="abc")
        stored = mixed.peft_config["default"]
        assert (stored.base_model_name_or_path, stored.revision) == ("tiny-mlp", "abc")
        assert config.base_model_name_or_path is None and config.revision is None

    def test_a_task_type_does_not_pick_another_class(self) -> None:
        mixed = get_peft_model(
            _net("cpu"), LoraConfig(r=2, target_modules=["0"], task_type="CAUSAL_LM"), mixed=True
        )
        assert type(mixed) is PeftMixedModel

    @pytest.mark.parametrize("option", ["autocast_adapter_dtype", "low_cpu_mem_usage"])
    def test_an_option_the_library_cannot_honour_is_refused_by_value(self, option: str) -> None:
        net = _net("cpu")
        before = _structure(net)
        with pytest.raises(NotImplementedError, match=f"get_peft_model: `{option}"):
            get_peft_model(net, _lora(), mixed=True, **{option: option == "low_cpu_mem_usage"})
        _unchanged(net, None, before)

    def test_without_mixed_it_is_the_peft_model_it_was(self) -> None:
        assert type(get_peft_model(_net("cpu"), _lora())) is PeftModel
        assert type(get_peft_model(_net("cpu"), _lora(), mixed=False)) is PeftModel


# ── the files ─────────────────────────────────────────────────────────────


class TestFromPretrained:
    @pytest.mark.parametrize("peft_type", list(_MIXED_TUNERS), ids=lambda t: t.name)
    def test_the_file_of_every_mixable_method_reads_back_with_the_output_it_was_saved_with(
        self, peft_type: PeftType, device: str, tmp_path: Path
    ) -> None:
        config = RECIPES[peft_type].config(_F32, device, tmp_path)
        wanted = _saved(config, tmp_path, device, seed=3)
        loaded = PeftMixedModel.from_pretrained(_net(device), tmp_path)
        with lucid.no_grad():
            got = _call(loaded.base_model.model, device)
        assert type(loaded) is PeftMixedModel
        assert span(got, wanted) < _TOLERANCE * max(1.0, float(wanted.abs().max().item()))
        assert span(wanted, _call(_net(device), device)) > _EFFECT

    def test_it_is_frozen_and_in_eval_mode_unless_trainable(self, tmp_path: Path) -> None:
        _saved(_lora(), tmp_path, "cpu", seed=3)
        frozen = PeftMixedModel.from_pretrained(_net("cpu"), tmp_path)
        assert not frozen.training and not any(p.requires_grad for p in frozen.parameters())
        trainable = PeftMixedModel.from_pretrained(_net("cpu"), tmp_path, is_trainable=True)
        assert any(p.requires_grad for p in trainable.parameters())
        assert not trainable.peft_config["default"].inference_mode

    def test_it_names_the_adapter_and_reads_a_sub_directory(
        self, device: str, tmp_path: Path
    ) -> None:
        single = get_peft_model(_net(device), _lora())
        single.add_adapter("style", _lora())
        single.set_adapter("style")
        _move(single, "style", 5)
        single.save_pretrained(tmp_path)
        with lucid.no_grad():
            wanted = _call(single, device)
        loaded = PeftMixedModel.from_pretrained(
            _net(device), tmp_path, adapter_name="mine", subfolder="style"
        )
        assert loaded.active_adapters == ["mine"]
        with lucid.no_grad():
            assert span(_call(loaded.base_model.model, device), wanted) < _TOLERANCE

    def test_a_method_outside_the_mapping_is_refused_and_the_model_is_untouched(
        self, tmp_path: Path
    ) -> None:
        saved = get_peft_model(_net("cpu"), VeraConfig(r=2, target_modules=["0", "2"]))
        saved.save_pretrained(tmp_path)
        net = _net("cpu")
        before = _structure(net)
        with pytest.raises(
            ValueError, match="Adapter of type VERA is not supported for mixed models."
        ):
            PeftMixedModel.from_pretrained(net, tmp_path)
        _unchanged(net, None, before)

    @pytest.mark.parametrize("is_trainable", [False, True])
    def test_a_prompt_learning_file_is_refused_trainable_or_not(
        self, is_trainable: bool, tmp_path: Path
    ) -> None:
        recipe = RECIPES[PeftType.PROMPT_TUNING]
        get_peft_model(
            recipe.base(_F32, "cpu"), recipe.config(_F32, "cpu", tmp_path)
        ).save_pretrained(tmp_path)
        with pytest.raises(ValueError, match="PROMPT_TUNING is not supported for mixed models"):
            PeftMixedModel.from_pretrained(
                recipe.base(_F32, "cpu"), tmp_path, is_trainable=is_trainable
            )

    def test_a_config_given_is_checked_like_the_files(self, tmp_path: Path) -> None:
        _saved(_lora(), tmp_path, "cpu", seed=3)
        with pytest.raises(ValueError, match="Adapter of type VERA is not supported"):
            PeftMixedModel.from_pretrained(
                _net("cpu"), tmp_path, config=VeraConfig(r=2, target_modules=["0", "2"])
            )
        with pytest.raises(TypeError, match="config must be an adapter config"):
            PeftMixedModel.from_pretrained(
                _net("cpu"), tmp_path, config="lora"
            )  # a str, not a config
        loaded = PeftMixedModel.from_pretrained(_net("cpu"), tmp_path, config=_lora())
        assert loaded.peft_config["default"].inference_mode

    def test_a_directory_without_an_adapter_is_a_file_not_found_error(self, tmp_path: Path) -> None:
        net = _net("cpu")
        before = _structure(net)
        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            PeftMixedModel.from_pretrained(net, tmp_path / "nowhere")
        _unchanged(net, None, before)

    @pytest.mark.parametrize(
        "kwargs, error, message",
        [
            ({"token": "t"}, NotImplementedError, "`token` is not supported"),
            (
                {"autocast_adapter_dtype": False},
                NotImplementedError,
                "autocast_adapter_dtype=False",
            ),
            ({"ephemeral_gpu_offload": True}, NotImplementedError, "ephemeral_gpu_offload=True"),
            (
                {"ignore_mismatched_sizes": True},
                NotImplementedError,
                "ignore_mismatched_sizes=True",
            ),
            ({"subfolder": 3}, TypeError, "`subfolder` must be a str"),
            ({"convert": "maybe"}, TypeError, "`convert` must be True, False or 'auto'"),
            ({"key_mapping": {"a": 1}}, TypeError, "`key_mapping` must map str"),
            ({"low_cpu_mem_usage": "no"}, TypeError, "`low_cpu_mem_usage` must be a bool"),
        ],
    )
    def test_an_option_the_library_cannot_honour_or_a_value_of_the_wrong_type_is_refused(
        self, kwargs: dict[str, object], error: type[Exception], message: str, tmp_path: Path
    ) -> None:
        _saved(_lora(), tmp_path, "cpu", seed=3)
        net = _net("cpu")
        before = _structure(net)
        with pytest.raises(error, match=message):
            PeftMixedModel.from_pretrained(net, tmp_path, **kwargs)
        _unchanged(net, None, before)

    def test_convert_is_accepted_off(self, tmp_path: Path) -> None:
        wanted = _saved(_lora(), tmp_path, "cpu", seed=3)
        loaded = PeftMixedModel.from_pretrained(_net("cpu"), tmp_path, convert=False)
        with lucid.no_grad():
            assert span(_call(loaded.base_model.model, "cpu"), wanted) < _TOLERANCE


class TestLoadAdapter:
    @pytest.mark.parametrize("when", ["before", "after"])
    def test_a_single_method_file_loads_before_and_after_the_stacking(
        self, when: str, device: str, tmp_path: Path
    ) -> None:
        wanted = {
            "lora": _saved(_lora(), tmp_path / "lora", device, seed=7),
            "loha": _saved(_loha(), tmp_path / "loha", device, seed=8),
        }
        net = _net(device)
        mixed = PeftMixedModel(net, _lora(), "lora")
        if when == "before":
            mixed.load_adapter(
                tmp_path / "lora", "lora"
            )  # the name is held: its values are replaced
        mixed.load_adapter(tmp_path / "loha", "loha")  # a new name: added, stacked, filled
        if when == "after":
            mixed.load_adapter(tmp_path / "lora", "lora")  # beneath the LoHa layer by now
        assert mixed.active_adapters == ["lora"]  # a loaded adapter is not put in use
        for name, saved in wanted.items():
            assert span(_only(net, mixed, name, device), saved) < _TOLERANCE, name
            assert span(saved, _call(_net(device), device)) > _EFFECT

    def test_a_file_loaded_into_the_adapter_below_changes_the_stack_the_reference_would_not(
        self, device: str, tmp_path: Path
    ) -> None:
        wanted = _saved(_lora(), tmp_path, device, seed=7)
        stack = _stack(device)
        with lucid.no_grad():
            before = _only(stack.net, stack.mixed, "lora", device)
        stack.mixed.load_adapter(tmp_path, "lora")
        with lucid.no_grad():
            after = _only(stack.net, stack.mixed, "lora", device)
        assert span(after, before) > _EFFECT
        assert span(after, wanted) < _TOLERANCE

    def test_from_pretrained_then_a_second_method_stacks_on_it(
        self, device: str, tmp_path: Path
    ) -> None:
        wanted = _saved(_loha(), tmp_path / "loha", device, seed=8)
        loaded = PeftMixedModel.from_pretrained(
            _net(device), tmp_path / "loha", adapter_name="loha"
        )
        loaded.add_adapter("lora", _lora())
        assert type(loaded.base_model.model.get_submodule("0")).__module__.endswith("lora._layer")
        loaded.set_adapter(["loha"])
        with lucid.no_grad():
            assert span(_call(loaded.base_model.model, device), wanted) < _TOLERANCE

    def test_an_adapter_is_frozen_unless_it_is_loaded_trainable(self, tmp_path: Path) -> None:
        _saved(_loha(), tmp_path, "cpu", seed=8)
        stack = _stack("cpu")
        stack.mixed.load_adapter(tmp_path, "frozen")
        stack.mixed.load_adapter(tmp_path, "trained", True)  # is_trainable, by position
        assert not any(p.requires_grad for p in _held(stack.net, "frozen"))
        assert all(p.requires_grad for p in _held(stack.net, "trained"))
        assert stack.mixed.active_adapters == ["lora", "loha"]

    def test_it_returns_what_the_state_dict_loader_returns(self, tmp_path: Path) -> None:
        _saved(_lora(), tmp_path, "cpu", seed=7)
        loaded = _stack("cpu").mixed.load_adapter(tmp_path, "again")
        assert loaded.unexpected_keys == []

    @pytest.mark.parametrize(
        "args, kwargs, error, message",
        [
            (
                (True, "cpu", True, False, False, None, "x"),
                {},
                TypeError,
                "takes at most 6 more positional",
            ),
            (
                (True,),
                {"is_trainable": True},
                TypeError,
                "multiple values for argument 'is_trainable'",
            ),
            ((), {"is_trainable": 1}, TypeError, "`is_trainable` must be a bool"),
            ((), {"device": 3}, TypeError, "`device` must be a str"),
            ((), {"token": "t"}, NotImplementedError, "`token` is not supported"),
            (
                (),
                {"ignore_mismatched_sizes": True},
                NotImplementedError,
                "ignore_mismatched_sizes=True",
            ),
            ((True, "cpu", False), {}, NotImplementedError, "autocast_adapter_dtype=False"),
        ],
    )
    def test_arguments_are_checked_before_the_model_changes(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        error: type[Exception],
        message: str,
        tmp_path: Path,
    ) -> None:
        _saved(_lora(), tmp_path, "cpu", seed=7)
        stack = _stack("cpu")
        before = _structure(stack.net, stack.mixed)
        with pytest.raises(error, match=message):
            stack.mixed.load_adapter(tmp_path, "again", *args, **kwargs)
        _unchanged(stack.net, stack.mixed, before)

    def test_a_device_a_subfolder_and_a_renaming_are_accepted(
        self, device: str, tmp_path: Path
    ) -> None:
        wanted = _saved(_lora(), tmp_path / "inner", device, seed=7)
        mixed = PeftMixedModel(_net(device), _loha(), "loha")
        mixed.load_adapter(
            tmp_path, "lora", device="cpu", subfolder="inner", key_mapping={"zzz": "y"}
        )
        net = mixed.base_model.model
        assert span(_only(net, mixed, "lora", device), wanted) < _TOLERANCE


# ── what it forwards ──────────────────────────────────────────────────────


class _Generating(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: lucid.Tensor, scale: float = 1.0) -> lucid.Tensor:
        return self.lin(x) * scale

    def generate(self, x: lucid.Tensor, steps: int = 2) -> lucid.Tensor:
        for _ in range(steps):
            x = self(x)
        return x


class TestForwarding:
    def test_forward_and_generate_pass_their_arguments_to_the_model(self) -> None:
        model = _Generating()
        mixed = PeftMixedModel(model, LoraConfig(r=2, target_modules=["lin"]))
        x = lucid.randn(2, 4)
        assert same(mixed(x, scale=2.0), model(x, scale=2.0))
        assert same(mixed.generate(x, steps=3), model.generate(x, steps=3))

    def test_a_keyword_the_model_does_not_know_is_not_swallowed(self) -> None:
        mixed = PeftMixedModel(_Generating(), LoraConfig(r=2, target_modules=["lin"]))
        with pytest.raises(TypeError):
            mixed(lucid.randn(2, 4), adapter_names=["default"])  # a PeftModel's routing; none here

    def test_a_model_without_generate_says_so(self) -> None:
        mixed = _stack("cpu").mixed
        with pytest.raises(AttributeError, match="Sequential has no generate"):
            mixed.generate(lucid.randn(1, 6))

    def test_a_public_name_it_lacks_is_read_from_the_model(self) -> None:
        model = _Generating()
        model.tag = "wrapped"
        mixed = PeftMixedModel(model, LoraConfig(r=2, target_modules=["lin"]))
        assert mixed.tag == "wrapped"
        assert not hasattr(mixed, "_missing")


class TestParameterCounts:
    def test_trainable_and_total_count_each_parameter_once(self) -> None:
        stack = _stack("cpu")
        total = sum(p.numel() for p in stack.mixed.parameters())
        trainable = sum(p.numel() for p in stack.mixed.parameters() if p.requires_grad)
        assert stack.mixed.get_nb_trainable_parameters() == (trainable, total)
        assert 0 < trainable < total

    def test_the_report_is_the_peft_models(self, capsys: pytest.CaptureFixture[str]) -> None:
        net = nn.Sequential(nn.Linear(8, 8))
        PeftMixedModel(net, LoraConfig(r=2, target_modules=["0"])).print_trainable_parameters()
        assert capsys.readouterr().out == (
            "trainable params: 32 || all params: 104 || trainable%: 30.7692\n"
        )


class TestModulesToSave:
    def test_a_module_trained_in_full_is_listed_and_confirmed(self) -> None:
        net = _net("cpu")
        mixed = PeftMixedModel(
            net, LoraConfig(r=2, target_modules=["0"], modules_to_save=["2"]), "a"
        )
        mixed.add_adapter("b", LoHaConfig(r=2, target_modules=["0"]))
        assert mixed.modules_to_save == {"2"}
        mixed.set_modules_to_save(mixed.peft_config["a"], "a")  # held: nothing to do

    def test_none_is_none_and_an_adapter_without_the_module_is_refused(self) -> None:
        net = _net("cpu")
        mixed = PeftMixedModel(
            net, LoraConfig(r=2, target_modules=["0"], modules_to_save=["2"]), "a"
        )
        mixed.add_adapter("b", LoHaConfig(r=2, target_modules=["0"]))
        assert PeftMixedModel(_net("cpu"), _lora()).modules_to_save is None
        mixed.set_modules_to_save(_lora(), "b")  # names none
        with pytest.raises(ValueError, match="does not train \\['2'\\] in full"):
            mixed.set_modules_to_save(mixed.peft_config["a"], "b")
        with pytest.raises(ValueError, match="Adapter\\(s\\) \\['nope'\\] not found"):
            mixed.set_modules_to_save(mixed.peft_config["a"], "nope")


# ── the mapping ───────────────────────────────────────────────────────────


class TestTheMapping:
    def test_it_is_a_read_only_view_of_the_registry_the_host_reads(self) -> None:
        assert isinstance(PEFT_TYPE_TO_MIXED_MODEL_MAPPING, MappingProxyType)
        assert PEFT_TYPE_TO_MIXED_MODEL_MAPPING == _MIXED_TUNERS
        with pytest.raises(TypeError):
            PEFT_TYPE_TO_MIXED_MODEL_MAPPING[PeftType.VERA] = _TUNERS[PeftType.VERA]
        with pytest.raises(TypeError):
            del PEFT_TYPE_TO_MIXED_MODEL_MAPPING[PeftType.LORA]

    def test_it_holds_the_ten_methods_the_pair_contract_verifies(self) -> None:
        names = ["LORA", "LOHA", "LOKR", "ADALORA", "SHIRA", "OFT", "IA3", "BEFT", "ROAD", "HIRA"]
        assert sorted(t.name for t in PEFT_TYPE_TO_MIXED_MODEL_MAPPING) == sorted(names)

    def test_its_tuners_are_the_methods_own(self) -> None:
        for peft_type, tuner in PEFT_TYPE_TO_MIXED_MODEL_MAPPING.items():
            assert PEFT_TYPE_TO_TUNER_MAPPING[peft_type] is tuner

    @pytest.mark.parametrize("peft_type", list(_TUNERS), ids=lambda t: t.name)
    def test_exactly_the_methods_in_it_can_start_a_mixed_model(
        self, peft_type: PeftType, tmp_path: Path
    ) -> None:
        recipe = RECIPES[peft_type]
        config = recipe.config(_F32, "cpu", tmp_path)
        if peft_type in PEFT_TYPE_TO_MIXED_MODEL_MAPPING:
            mixed = PeftMixedModel(recipe.base(_F32, "cpu"), config)
            assert mixed.peft_config["default"].peft_type == peft_type
        else:
            with pytest.raises(ValueError, match="is not compatible with the `PeftMixedModel`"):
                PeftMixedModel(recipe.base(_F32, "cpu"), config)
