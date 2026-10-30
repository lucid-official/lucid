"""A model with several adapters: switching, deleting, combining, saving and loading them.

Closed forms on a tiny MLP, on CPU and Metal: ``y = W x + b + s B A x`` for
the adapter in use and for it alone (a model that holds an adapter it does
not use computes what it would without it), a weighted adapter is exactly
the sum the combination type names (or, for ``"svd"``, its rank-``r``
truncation with the discarded singular values as the error), and the
on-disk layout is the reference's: the ``"default"`` adapter in the
directory itself, every other in a sub-directory of its own name.
"""

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid.peft import (
    LoraConfig,
    LoraModel,
    PeftConfig,
    PeftModel,
    PeftType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from lucid.peft._config import _CONFIG_CLASSES
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft.tuners import _base
from lucid.peft.tuners.lora import Linear, LoraLayer
from lucid.peft.tuners.lora._merging import _Combination
from lucid.serialization import load_safetensors, save_safetensors


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


def _mlp(device: str) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3)).to(device)


def _config(r: int, targets: list[str], **kwargs: object) -> LoraConfig:
    """A config whose update is non-zero from the start (``B`` is not zeroed)."""
    return LoraConfig(
        r=r,
        lora_alpha=2 * r,
        target_modules=targets,
        init_lora_weights=False,
        **kwargs,  # type: ignore[arg-type]
    )


def _two(device: str) -> tuple[PeftModel, nn.Sequential]:
    """``default`` (r=2) on both Linears, ``other`` (r=3) on the first only."""
    net = _mlp(device)
    peft = get_peft_model(net, _config(2, ["0", "2"]))
    peft.add_adapter("other", _config(3, ["0"]))
    return peft, net


def _layer(net: nn.Sequential, index: int) -> Linear:
    layer = net[index]
    assert isinstance(layer, Linear)
    return layer


def _delta(layer: LoraLayer, name: str) -> lucid.Tensor:
    """``s B A`` read off the matrices, not off ``get_delta_weight``."""
    scale = layer.lora_alpha[name] / layer.r[name]
    return (layer.lora_B[name].weight @ layer.lora_A[name].weight) * scale


def _trainable(model: nn.Module) -> set[str]:
    return {name for name, p in model.named_parameters() if p.requires_grad}


# ── two adapters on one model ────────────────────────────────────────────────


class TestSwitching:
    def test_each_adapter_computes_what_it_computes_alone(self, device: str) -> None:
        peft, _ = _two(device)
        x = lucid.randn(4, 6, device=device)
        configs = {"default": _config(2, ["0", "2"]), "other": _config(3, ["0"])}
        for name, config in configs.items():
            # The same adapter's matrices, alone on a fresh copy of the same base.
            alone = get_peft_model(_mlp(device), config)
            set_peft_model_state_dict(alone, get_peft_model_state_dict(peft, adapter_name=name))
            peft.set_adapter(name)
            with lucid.no_grad():
                assert _same(peft(x), alone(x)), name

    def test_the_new_adapter_is_inactive_until_set(self, device: str) -> None:
        peft, net = _two(device)
        x = lucid.randn(4, 6, device=device)
        with lucid.no_grad():
            before = peft(x)
        assert peft.active_adapter == "default" and peft.active_adapters == ["default"]
        assert sorted(peft.peft_config) == ["default", "other"]
        # A rank-3 adapter with a non-zero update is there and unused.
        assert _layer(net, 0).r == {"default": 2, "other": 3}
        peft.set_adapter("other")
        with lucid.no_grad():
            assert not _same(peft(x), before)
            peft.set_adapter("default")
            assert _same(peft(x), before)

    def test_an_adapter_aimed_elsewhere_leaves_the_other_layer_to_the_base(
        self, device: str
    ) -> None:
        peft, net = _two(device)
        x = lucid.randn(4, 6, device=device)
        peft.set_adapter("other")
        first, last = _layer(net, 0), _layer(net, 2)
        with lucid.no_grad():
            hidden = first.base_layer(x) + x @ _delta(first, "other").T
            expected = last.base_layer(lucid.relu(hidden))
            assert _close(peft(x), expected)

    def test_a_list_applies_every_adapter_it_names(self, device: str) -> None:
        net = nn.Sequential(nn.Linear(6, 4)).to(device)
        lora = LoraModel(net, _config(2, ["0"]), adapter_name="a")
        lora.add_adapter("b", _config(3, ["0"]))
        lora.set_adapter(["a", "b"])
        layer = _layer(net, 0)
        x = lucid.randn(5, 6, device=device)
        assert lora.active_adapter == ["a", "b"] and layer.active_adapter == ["a", "b"]
        with lucid.no_grad():
            expected = layer.base_layer(x) + x @ (_delta(layer, "a") + _delta(layer, "b")).T
            assert _close(net(x), expected)

    def test_disable_adapter_is_the_base_with_several_adapters(self, device: str) -> None:
        peft, _ = _two(device)
        x = lucid.randn(4, 6, device=device)
        with lucid.no_grad():
            adapted = peft(x)
            with peft.disable_adapter():
                assert _same(peft(x), _mlp(device)(x))
            assert _same(peft(x), adapted)

    def test_set_adapter_trains_the_chosen_adapter_only(self, device: str) -> None:
        peft, _ = _two(device)
        has = lambda name: {n for n in _trainable(peft) if f".{name}." in n}  # noqa: E731
        assert has("default") and not has("other")
        peft.set_adapter("other")
        assert has("other") and not has("default")
        peft.set_adapter("default", inference_mode=True)
        assert not has("default") and not has("other")
        assert _trainable(peft) == set()

    def test_the_base_stays_frozen_through_every_switch(self, device: str) -> None:
        peft, net = _two(device)
        for name in ("other", "default", "other"):
            peft.set_adapter(name)
            assert not any(p.requires_grad for p in net[0].base_layer.parameters())
            assert not any(p.requires_grad for p in net[2].base_layer.parameters())

    def test_conversions_reach_every_adapter(self, device: str) -> None:
        peft, net = _two(device)
        peft.half()
        for name in ("default", "other"):
            assert _layer(net, 0).lora_A[name].weight.dtype == lucid.float16
            assert _layer(net, 0).lora_B[name].weight.dtype == lucid.float16
        assert _layer(net, 2).lora_A["default"].weight.dtype == lucid.float16

    def test_setting_an_adapter_the_model_lacks(self, device: str) -> None:
        peft, _ = _two(device)
        with pytest.raises(ValueError, match="Adapter nope not found"):
            peft.set_adapter("nope")
        assert peft.active_adapter == "default"

    def test_setting_a_merged_model_unmerges_it_first(self, device: str) -> None:
        peft, net = _two(device)
        x = lucid.randn(4, 6, device=device)
        with lucid.no_grad():
            adapted = peft(x)
            peft.base_model.merge_adapter()
            assert _layer(net, 0).merged
            with pytest.warns(UserWarning, match="Unmerging the model first"):
                peft.set_adapter("other")
            assert not _layer(net, 0).merged
            peft.set_adapter("default")
            assert _close(peft(x), adapted)

    def test_a_layer_that_is_set_to_nothing_runs_its_base(self, device: str) -> None:
        peft, net = _two(device)
        peft.base_model.set_adapter([])
        x = lucid.randn(4, 6, device=device)
        with lucid.no_grad():
            assert _same(peft(x), _mlp(device)(x))
        assert peft.active_adapters == []


class TestAdding:
    def test_the_existing_parameters_keep_their_trainability(self, device: str) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, _config(2, ["0"]))
        net[2].bias.requires_grad_(True)  # the user opened one base bias
        # By identity: wrapping "2" renames its parameters.
        before = {id(p): p.requires_grad for p in peft.parameters()}
        peft.add_adapter("other", _config(2, ["0", "2"]))
        assert {id(p): p.requires_grad for p in peft.parameters() if id(p) in before} == before
        added = [p for p in peft.parameters() if id(p) not in before]
        assert added and not any(p.requires_grad for p in added)

    def test_a_target_no_adapter_wrapped_yet_is_wrapped(self, device: str) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, _config(2, ["0"]))
        assert type(net[2]) is nn.Linear
        peft.add_adapter("other", _config(2, ["2"]))
        assert isinstance(net[2], Linear) and net[2].r == {"other": 2}
        assert sorted(peft.base_model.targeted_module_names) == ["0", "2"]
        # It is inactive there: the model still computes the first adapter alone.
        assert net[2].active_adapter == "default"
        x = lucid.randn(3, 6, device=device)
        with lucid.no_grad():
            assert _close(peft(x), peft(x))

    def test_all_linear_reaches_wrapped_layers_and_not_their_insides(self, device: str) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, _config(2, ["0"]))
        peft.add_adapter("every", LoraConfig(r=2, target_modules="all-linear"))
        assert peft.peft_config["every"].target_modules == {"0", "2"}
        assert sorted(net[0].lora_A) == ["default", "every"]
        assert sorted(net[2].lora_A) == ["every"]

    def test_a_failed_add_leaves_the_model_as_it_was(self, device: str) -> None:
        peft, net = _two(device)
        keys = sorted(peft.state_dict())
        flags = {n: p.requires_grad for n, p in peft.named_parameters()}
        with pytest.raises(ValueError, match="not found in the base model"):
            peft.add_adapter("bad", _config(2, ["nowhere"]))
        assert sorted(peft.state_dict()) == keys and "bad" not in peft.peft_config
        assert {n: p.requires_grad for n, p in peft.named_parameters()} == flags

    def test_a_failure_while_applying_takes_the_new_adapter_back_out(
        self, device: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, _config(2, ["0"], modules_to_save=["2"]))
        keys = sorted(peft.state_dict())

        def fail(self: ModulesToSaveWrapper, adapter_name: str) -> None:
            raise RuntimeError("out of memory")

        # The layer takes the adapter first; the copy for the head then fails.
        monkeypatch.setattr(ModulesToSaveWrapper, "update", fail)
        with pytest.raises(RuntimeError, match="out of memory"):
            peft.add_adapter("second", _config(2, ["0"], modules_to_save=["2"]))
        assert sorted(peft.state_dict()) == keys
        assert sorted(peft.peft_config) == ["default"] and _layer(net, 0).r == {"default": 2}

    def test_a_failure_while_placing_restores_the_modules(
        self, device: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        net = _mlp(device)
        placed: list[str] = []
        real = _base._set_submodule

        def flaky(model: nn.Module, key: str, module: nn.Module) -> None:
            placed.append(key)
            if len(placed) == 2:
                raise RuntimeError("cannot place")
            real(model, key, module)

        monkeypatch.setattr(_base, "_set_submodule", flaky)
        with pytest.raises(RuntimeError, match="cannot place"):
            LoraModel(net, _config(2, ["0", "2"]))
        assert type(net[0]) is nn.Linear and type(net[2]) is nn.Linear
        assert "peft_config" not in vars(net) and all(p.requires_grad for p in net.parameters())

    @pytest.mark.parametrize(
        ("name", "error", "match"),
        [
            ("default", ValueError, "already exists"),
            ("a.b", ValueError, "without '.'"),
            ("", ValueError, "non-empty"),
            ("double", ValueError, "attribute of nn.ModuleDict"),
            ("keys", ValueError, "attribute of nn.ModuleDict"),
            (3, TypeError, "must be a str"),
        ],
    )
    def test_a_bad_name_is_refused(self, name: object, error: type[Exception], match: str) -> None:
        peft = get_peft_model(_mlp("cpu"), _config(2, ["0"]))
        with pytest.raises(error, match=match):
            peft.add_adapter(name, _config(2, ["0"]))  # type: ignore[arg-type]

    def test_the_config_is_checked(self) -> None:
        peft = get_peft_model(_mlp("cpu"), _config(2, ["0"]))
        with pytest.raises(TypeError, match="adapter config"):
            peft.add_adapter("x", "not a config")  # type: ignore[arg-type]
        assert sorted(peft.peft_config) == ["default"]

    def test_a_config_of_another_method_is_refused(self) -> None:
        peft = get_peft_model(_mlp("cpu"), _config(2, ["0"]))
        registered = dict(_CONFIG_CLASSES)
        try:

            @dataclass(kw_only=True)
            class OtherConfig(PeftConfig, peft_type=PeftType.IA3):
                pass

            with pytest.raises(ValueError, match="different peft types. Found LORA and IA3"):
                peft.add_adapter("x", OtherConfig())
        finally:
            # The registry is global: leave it as the other tests expect it.
            _CONFIG_CLASSES.clear()
            _CONFIG_CLASSES.update(registered)
        assert sorted(peft.peft_config) == ["default"]

    def test_only_one_adapter_trains_biases(self) -> None:
        peft = get_peft_model(_mlp("cpu"), _config(2, ["0"], bias="all"))
        with pytest.raises(ValueError, match="supports only 1 adapter with bias"):
            peft.add_adapter("x", _config(2, ["2"], bias="lora_only"))
        peft.add_adapter("plain", _config(2, ["2"]))  # bias="none" sits beside it

    def test_the_bias_follows_the_adapter_that_asked_for_it(self) -> None:
        net = _mlp("cpu")
        peft = get_peft_model(net, _config(2, ["0"]))
        peft.add_adapter("biased", _config(2, ["0"], bias="lora_only"))
        bias = net[0].base_layer.bias
        assert not bias.requires_grad
        peft.set_adapter("biased")
        assert bias.requires_grad
        peft.set_adapter("default")
        assert not bias.requires_grad

    def test_a_wrapper_already_around_the_model_is_not_stacked(self) -> None:
        net = _mlp("cpu")
        get_peft_model(net, _config(2, ["0"]))
        with pytest.raises(NotImplementedError, match="add more with PeftModel.add_adapter"):
            get_peft_model(net, _config(2, ["2"]))


class TestDeleting:
    def test_deleting_the_extra_adapter_restores_the_model(self, device: str) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, _config(2, ["0", "2"]))
        x = lucid.randn(4, 6, device=device)
        with lucid.no_grad():
            before = peft(x)
        keys = sorted(peft.state_dict())
        modules = [name for name, _ in peft.named_modules()]
        peft.add_adapter("other", _config(3, ["0"]))
        peft.delete_adapter("other")
        assert sorted(peft.state_dict()) == keys
        assert [name for name, _ in peft.named_modules()] == modules  # dropout modules too
        assert sorted(peft.peft_config) == ["default"] and "other" not in net[0].r
        assert peft.active_adapter == "default"
        with lucid.no_grad():
            assert _same(peft(x), before)

    def test_deleting_the_active_adapter_hands_over_with_a_warning(self, device: str) -> None:
        peft, net = _two(device)
        peft.set_adapter("other")
        with pytest.warns(UserWarning, match="Setting active adapter to default"):
            peft.delete_adapter("other")
        assert peft.active_adapter == "default"
        assert _layer(net, 0).active_adapter == "default"
        assert any(p.requires_grad for p in net[0].lora_A["default"].parameters())

    def test_deleting_the_inactive_one_changes_nothing_else(self, device: str) -> None:
        peft, net = _two(device)
        flags = {n: p.requires_grad for n, p in peft.named_parameters() if ".other." not in n}
        peft.delete_adapter("other")
        assert {n: p.requires_grad for n, p in peft.named_parameters()} == flags

    def test_deleting_one_of_several_active_keeps_the_rest(self, device: str) -> None:
        net = nn.Sequential(nn.Linear(6, 4)).to(device)
        lora = LoraModel(net, _config(2, ["0"]), adapter_name="a")
        lora.add_adapter("b", _config(2, ["0"]))
        lora.add_adapter("c", _config(2, ["0"]))
        lora.set_adapter(["a", "b"])
        lora.delete_adapter("a")
        assert lora.active_adapters == ["b"] and _layer(net, 0).active_adapter == "b"

    def test_the_last_adapter_can_go_and_the_layers_apply_nothing(self, device: str) -> None:
        peft, net = _two(device)
        peft.delete_adapter("other")
        peft.delete_adapter("default")
        x = lucid.randn(4, 6, device=device)
        assert peft.peft_config == {} and peft.active_adapters == []
        with lucid.no_grad():
            assert _same(peft(x), _mlp(device)(x))
        peft.unload()
        assert type(net[0]) is nn.Linear and type(net[2]) is nn.Linear

    def test_a_merged_adapter_cannot_be_deleted(self, device: str) -> None:
        peft, net = _two(device)
        peft.base_model.merge_adapter(adapter_names=["other"])
        with pytest.raises(ValueError, match="while they are merged"):
            peft.delete_adapter("other")
        assert "other" in peft.peft_config
        peft.base_model.unmerge_adapter()
        peft.delete_adapter("other")

    def test_deleting_what_is_not_there(self, device: str) -> None:
        peft, _ = _two(device)
        with pytest.raises(ValueError, match="Adapter nope does not exist"):
            peft.delete_adapter("nope")

    def test_modules_to_save_copies_go_with_their_adapter(self, device: str) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, _config(2, ["0"], modules_to_save=["2"]))
        peft.add_adapter("b", _config(2, ["0"], modules_to_save=["2"]))
        wrapper = net[2]
        assert isinstance(wrapper, ModulesToSaveWrapper)
        assert sorted(wrapper.modules_to_save) == ["b", "default"]
        peft.delete_adapter("b")
        assert sorted(wrapper.modules_to_save) == ["default"]
        peft.delete_adapter("default")
        assert wrapper.active_adapter == "" and sorted(wrapper.modules_to_save) == []
        x = lucid.randn(2, 8, device=device)
        assert _same(wrapper(x), wrapper.original_module(x))


class TestModulesToSave:
    def _heads(self, device: str) -> tuple[PeftModel, nn.Sequential, ModulesToSaveWrapper]:
        net = _mlp(device)
        peft = get_peft_model(net, _config(2, ["0"], modules_to_save=["2"]))
        peft.add_adapter("b", _config(2, ["0"], modules_to_save=["2"]))
        peft.add_adapter("c", _config(2, ["0"]))  # trains no head
        wrapper = net[2]
        assert isinstance(wrapper, ModulesToSaveWrapper)
        with lucid.no_grad():
            wrapper.modules_to_save["default"].weight.add_(1.0)
            wrapper.modules_to_save["b"].weight.add_(-1.0)
        return peft, net, wrapper

    def test_each_adapter_runs_its_own_copy(self, device: str) -> None:
        peft, net, wrapper = self._heads(device)
        x = lucid.randn(2, 8, device=device)
        for name in ("default", "b"):
            peft.set_adapter(name)
            assert wrapper.active_adapter == name
            assert _same(wrapper(x), wrapper.modules_to_save[name](x))
        assert not _same(
            wrapper.modules_to_save["default"].weight, wrapper.modules_to_save["b"].weight
        )

    def test_an_adapter_without_a_copy_runs_the_original(self, device: str) -> None:
        peft, net, wrapper = self._heads(device)
        peft.set_adapter("c")
        x = lucid.randn(2, 8, device=device)
        assert wrapper.active_adapter == "" and _same(wrapper(x), wrapper.original_module(x))

    def test_only_the_active_copy_trains(self, device: str) -> None:
        peft, net, wrapper = self._heads(device)
        copies = wrapper.modules_to_save
        peft.set_adapter("b")
        assert all(p.requires_grad for p in copies["b"].parameters())
        assert not any(p.requires_grad for p in copies["default"].parameters())
        assert not any(p.requires_grad for p in wrapper.original_module.parameters())
        peft.set_adapter("c")
        assert not any(p.requires_grad for p in copies["b"].parameters())

    def test_two_adapters_cannot_share_the_module_when_active_together(self, device: str) -> None:
        peft, net, wrapper = self._heads(device)
        with pytest.raises(ValueError, match="Only one adapter can be set at a time"):
            peft.base_model.set_adapter(["default", "b"])
        assert peft.active_adapter == "default" and wrapper.active_adapter == "default"
        peft.base_model.set_adapter(["default", "c"])  # only one of them trains a head
        assert wrapper.active_adapter == "default"

    def test_merge_and_unload_keeps_the_active_adapters_copy(self, device: str) -> None:
        peft, net, wrapper = self._heads(device)
        peft.set_adapter("b")
        kept = wrapper.modules_to_save["b"]
        merged = peft.merge_and_unload()
        assert merged[2] is kept and type(merged[0]) is nn.Linear

    def test_unload_gives_the_original_back(self, device: str) -> None:
        peft, net, wrapper = self._heads(device)
        original = wrapper.original_module
        assert peft.unload()[2] is original

    def test_the_copies_save_and_load_by_adapter(self, device: str, tmp_path: Path) -> None:
        peft, net, wrapper = self._heads(device)
        peft.save_pretrained(tmp_path)
        stored = load_safetensors(str(tmp_path / "b" / "adapter_model.safetensors"))
        assert "base_model.model.2.weight" in stored
        fresh = get_peft_model(_mlp(device), _config(2, ["0"], modules_to_save=["2"]))
        fresh.load_adapter(tmp_path / "b", "b")
        fresh_wrapper = fresh.get_base_model()[2]
        assert isinstance(fresh_wrapper, ModulesToSaveWrapper)
        assert _same(fresh_wrapper.modules_to_save["b"].weight, wrapper.modules_to_save["b"].weight)


class TestMerging:
    def test_merging_several_adapters_adds_all_their_updates(self, device: str) -> None:
        net = nn.Sequential(nn.Linear(6, 4)).to(device)
        lora = LoraModel(net, _config(2, ["0"]), adapter_name="a")
        lora.add_adapter("b", _config(3, ["0"]))
        layer = _layer(net, 0)
        base = layer.base_layer.weight.clone()
        lora.merge_adapter(adapter_names=["a", "b"])
        assert layer.merged and layer.merged_adapters == ["a", "b"]
        with lucid.no_grad():
            assert _close(layer.base_layer.weight, base + _delta(layer, "a") + _delta(layer, "b"))
            lora.unmerge_adapter()
            assert _close(layer.base_layer.weight, base) and not layer.merged

    def test_the_active_adapter_is_the_default_to_merge(self, device: str) -> None:
        peft, net = _two(device)
        peft.set_adapter("other")
        peft.base_model.merge_adapter()
        assert _layer(net, 0).merged_adapters == ["other"]
        assert not _layer(net, 2).merged  # "other" is not on this layer

    def test_merge_and_unload_folds_the_adapters_it_is_given(self, device: str) -> None:
        net = nn.Sequential(nn.Linear(6, 4)).to(device)
        peft = get_peft_model(net, _config(2, ["0"]))
        peft.add_adapter("b", _config(3, ["0"]))
        x = lucid.randn(3, 6, device=device)
        layer = _layer(net, 0)
        with lucid.no_grad():
            expected = layer.base_layer(x) + x @ _delta(layer, "b").T
            merged = peft.merge_and_unload(adapter_names=["b"])
            assert type(merged[0]) is nn.Linear
            assert _close(merged(x), expected)

    def test_merging_an_unknown_adapter(self, device: str) -> None:
        peft, _ = _two(device)
        with pytest.raises(ValueError, match="Adapter ghost not found"):
            peft.base_model.merge_adapter(adapter_names=["ghost"])

    def test_safe_merge_checks_every_adapter(self, device: str) -> None:
        peft, net = _two(device)
        with lucid.no_grad():
            net[0].lora_B["other"].weight.mul_(float("nan"))
        with pytest.raises(ValueError, match="NaNs detected"):
            peft.base_model.merge_adapter(adapter_names=["default", "other"], safe_merge=True)
        assert not _layer(net, 0).merged

    def test_unload_drops_every_adapter(self, device: str) -> None:
        peft, net = _two(device)
        model = peft.unload()
        assert model is net and type(net[0]) is nn.Linear and "peft_config" not in vars(net)
        assert not any("lora" in n for n in net.state_dict())


# ── weighted adapters ────────────────────────────────────────────────────────


def _pair(device: str, r_a: int = 2, r_b: int = 3) -> tuple[LoraModel, Linear]:
    lucid.manual_seed(1)
    net = nn.Sequential(nn.Linear(6, 5)).to(device)
    lora = LoraModel(net, _config(r_a, ["0"]), adapter_name="a")
    lora.add_adapter("b", _config(r_b, ["0"]))
    return lora, _layer(net, 0)


class TestWeightedAdapters:
    def test_cat_is_the_exact_weighted_sum_at_the_summed_rank(self, device: str) -> None:
        lora, layer = _pair(device)
        lora.add_weighted_adapter(["a", "b"], [0.7, -0.4], "mix", combination_type="cat")
        expected = _delta(layer, "a") * 0.7 + _delta(layer, "b") * -0.4
        assert layer.r["mix"] == 5 and layer.scaling["mix"] == 1.0
        assert tuple(layer.lora_A["mix"].weight.shape) == (5, 6)
        assert tuple(layer.lora_B["mix"].weight.shape) == (5, 5)
        with lucid.no_grad():
            assert _close(layer.get_delta_weight("mix"), expected)

    def test_cat_stacks_the_sources_side_by_side(self, device: str) -> None:
        lora, layer = _pair(device)
        lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "mix", combination_type="cat")
        stacked = layer.lora_B["mix"].weight
        assert _close(stacked[:, :2], layer.lora_B["a"].weight)
        assert _close(stacked[:, 2:], layer.lora_B["b"].weight)

    def test_linear_weights_a_and_b_apart(self, device: str) -> None:
        lora, layer = _pair(device, 3, 3)
        lora.add_weighted_adapter(["a", "b"], [0.5, 2.0], "mix", combination_type="linear")
        scales = {n: layer.lora_alpha[n] / layer.r[n] for n in "ab"}
        roots = {n: math.sqrt(abs(w * scales[n])) for n, w in (("a", 0.5), ("b", 2.0))}
        a_sum = layer.lora_A["a"].weight * roots["a"] + layer.lora_A["b"].weight * roots["b"]
        b_sum = layer.lora_B["a"].weight * roots["a"] + layer.lora_B["b"].weight * roots["b"]
        assert layer.r["mix"] == 3
        with lucid.no_grad():
            assert _close(layer.get_delta_weight("mix"), b_sum @ a_sum)

    def test_linear_of_one_adapter_is_that_adapter_scaled(self, device: str) -> None:
        lora, layer = _pair(device)
        lora.add_weighted_adapter(["b"], [-1.5], "scaled", combination_type="cat")
        # One adapter has nothing to stack: every type is "linear".
        assert layer.r["scaled"] == 3
        with lucid.no_grad():
            assert _close(layer.get_delta_weight("scaled"), _delta(layer, "b") * -1.5)

    def test_linear_needs_one_rank(self, device: str) -> None:
        lora, layer = _pair(device)
        with pytest.raises(ValueError, match="same r value"):
            lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "mix", combination_type="linear")
        assert "mix" not in lora.peft_config

    def test_svd_is_the_truncation_of_the_summed_update(self, device: str) -> None:
        lora, layer = _pair(device)
        weights = [0.8, 0.6]
        total = _delta(layer, "a") * weights[0] + _delta(layer, "b") * weights[1]
        lora.add_weighted_adapter(["a", "b"], weights, "mix", svd_rank=2)
        sigma = linalg.svdvals(total)
        mix = layer.get_delta_weight("mix")
        assert layer.r["mix"] == 2 and layer.scaling["mix"] == 1.0
        with lucid.no_grad():
            kept = linalg.svdvals(mix)
            # rank 2: nothing left past the second singular value.
            assert float(kept[2:].max().item()) < 1e-5
            assert _close(kept[:2], sigma[:2], atol=1e-4)
            # Eckart-Young: the error is exactly the discarded singular values.
            error = float(((mix - total) ** 2).sum().sqrt().item())
            discarded = float((sigma[2:] ** 2).sum().sqrt().item())
            assert abs(error - discarded) < 1e-4

    def test_svd_defaults_to_the_largest_source_rank(self, device: str) -> None:
        lora, layer = _pair(device)
        lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "mix")
        assert layer.r["mix"] == 3 and layer.lora_alpha["mix"] == 3

    def test_svd_at_full_rank_is_exact(self, device: str) -> None:
        lora, layer = _pair(device)
        total = _delta(layer, "a") + _delta(layer, "b")
        lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "mix", svd_rank=5)
        with lucid.no_grad():
            assert _close(layer.get_delta_weight("mix"), total, atol=1e-4)

    def test_a_rank_beyond_the_matrix_is_padded_with_zeros(self, device: str) -> None:
        lora, layer = _pair(device)
        total = _delta(layer, "a") + _delta(layer, "b")
        lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "mix", svd_rank=8)
        assert layer.r["mix"] == 8 and tuple(layer.lora_A["mix"].weight.shape) == (8, 6)
        with lucid.no_grad():
            assert _close(layer.get_delta_weight("mix"), total, atol=1e-4)

    def test_reduced_and_full_matrices_agree(self, device: str) -> None:
        lora, layer = _pair(device)
        lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "full", svd_rank=3)
        lora.add_weighted_adapter(
            ["a", "b"], [1.0, 1.0], "thin", svd_rank=3, svd_full_matrices=False
        )
        with lucid.no_grad():
            assert _close(layer.get_delta_weight("full"), layer.get_delta_weight("thin"), atol=1e-4)

    def test_svd_clamp_bounds_the_factors(self, device: str) -> None:
        lora, layer = _pair(device)
        lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "free", svd_rank=3)
        lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "tight", svd_rank=3, svd_clamp=0.9)
        pooled = lucid.cat(
            [layer.lora_B["free"].weight.flatten(), layer.lora_A["free"].weight.flatten()]
        )
        bound = float(lucid.quantile(pooled, 0.9).item())
        for matrix in (layer.lora_A["tight"].weight, layer.lora_B["tight"].weight):
            assert float(matrix.abs().max().item()) <= bound + 1e-6
        assert float(layer.lora_B["free"].weight.abs().max().item()) > bound

    def test_a_layer_one_source_misses_combines_the_others(self, device: str) -> None:
        peft, net = _two(device)  # "other" is on layer 0 only, "default" on both
        lora = peft.base_model
        assert isinstance(lora, LoraModel)
        lora.add_weighted_adapter(["default", "other"], [1.0, 1.0], "mix", combination_type="cat")
        first, last = _layer(net, 0), _layer(net, 2)
        assert first.r["mix"] == 5 and last.r["mix"] == 5
        with lucid.no_grad():
            assert _close(
                first.get_delta_weight("mix"), _delta(first, "default") + _delta(first, "other")
            )
            # Only "default" reaches layer 2; the rest of the stacked rank is zero.
            assert _close(last.get_delta_weight("mix"), _delta(last, "default"))
        assert peft.peft_config["mix"].target_modules == {"0", "2"}

    def test_the_new_adapter_is_frozen_and_inactive(self, device: str) -> None:
        lora, layer = _pair(device)
        lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "mix", combination_type="cat")
        assert lora.active_adapters == ["a"]
        assert not any(p.requires_grad for p in layer.lora_A["mix"].parameters())
        assert not any(p.requires_grad for p in layer.lora_B["mix"].parameters())

    def test_a_weighted_adapter_is_a_normal_adapter(self, device: str) -> None:
        lora, layer = _pair(device)
        lora.add_weighted_adapter(["a", "b"], [0.5, 0.5], "mix", combination_type="cat")
        lora.set_adapter("mix")
        x = lucid.randn(3, 6, device=device)
        with lucid.no_grad():
            expected = layer.base_layer(x) + x @ layer.get_delta_weight("mix").T
            assert _close(lora.model(x), expected)

    def test_the_task_vector_family_is_not_built_yet(self, device: str) -> None:
        lora, _ = _pair(device, 2, 2)
        for name in ("ties", "dare_linear", "dare_ties", "magnitude_prune", "ties_svd"):
            with pytest.raises(NotImplementedError, match="not implemented in lucid.peft yet"):
                lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "mix", combination_type=name)  # type: ignore[arg-type]
        assert "mix" not in lora.peft_config

    def test_one_adapter_makes_any_type_linear(self, device: str) -> None:
        lora, layer = _pair(device)
        lora.add_weighted_adapter(["a"], [2.0], "twice", combination_type="ties")
        with lucid.no_grad():
            assert _close(layer.get_delta_weight("twice"), _delta(layer, "a") * 2.0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"adapters": [], "weights": []}, "Found no adapter"),
            ({"adapters": ["a", "ghost"], "weights": [1.0, 1.0]}, "Adapter ghost does not exist"),
            ({"adapters": ["a", "b"], "weights": [1.0]}, "one weight per adapter"),
            (
                {"adapters": ["a", "b"], "weights": [1.0, 1.0], "adapter_name": "a"},
                "already exists",
            ),
            (
                {"adapters": ["a", "b"], "weights": [1.0, 1.0], "combination_type": "mean"},
                "Invalid combination_type",
            ),
            (
                {"adapters": ["a", "b"], "weights": [1.0, 1.0], "majority_sign_method": "x"},
                "majority_sign_method",
            ),
            ({"adapters": ["a", "b"], "weights": [1.0, 1.0], "adapter_name": "m.x"}, "without '.'"),
        ],
    )
    def test_bad_arguments_are_refused_before_anything_is_added(
        self, kwargs: dict[str, object], match: str, device: str
    ) -> None:
        lora, _ = _pair(device)
        call: dict[str, object] = {"adapter_name": "mix"} | kwargs
        with pytest.raises(ValueError, match=match):
            lora.add_weighted_adapter(**call)  # type: ignore[arg-type]
        assert sorted(lora.peft_config) == ["a", "b"]

    def test_svd_driver_goes_to_the_linear_algebra(self, device: str) -> None:
        lora, _ = _pair(device)
        with pytest.raises(NotImplementedError):
            lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "mix", svd_driver="gesvd")
        assert "mix" not in lora.peft_config

    def test_two_adapters_training_one_module_cannot_be_combined(self, device: str) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, _config(2, ["0"], modules_to_save=["2"]))
        peft.add_adapter("b", _config(2, ["0"], modules_to_save=["2"]))
        lora = peft.base_model
        assert isinstance(lora, LoraModel)
        with pytest.raises(ValueError, match="same module with modules_to_save"):
            lora.add_weighted_adapter(["default", "b"], [1.0, 1.0], "mix", combination_type="cat")

    def test_target_modules_of_mixed_kinds_cannot_be_combined(self, device: str) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, _config(2, ["0"]))
        peft.add_adapter("by_regex", LoraConfig(r=2, target_modules=r"2"))
        lora = peft.base_model
        assert isinstance(lora, LoraModel)
        with pytest.raises(ValueError, match="same target modules type"):
            lora.add_weighted_adapter(
                ["default", "by_regex"], [1.0, 1.0], "mix", combination_type="cat"
            )

    def test_regex_targets_are_joined(self, device: str) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, LoraConfig(r=2, target_modules=r"0", init_lora_weights=False))
        peft.add_adapter("two", LoraConfig(r=2, target_modules=r"2", init_lora_weights=False))
        lora = peft.base_model
        assert isinstance(lora, LoraModel)
        lora.add_weighted_adapter(["default", "two"], [1.0, 1.0], "mix", combination_type="cat")
        assert peft.peft_config["mix"].target_modules == "(0)|(2)"

    def test_rank_stabilised_sources_still_give_a_unit_scale(self, device: str) -> None:
        lucid.manual_seed(2)
        net = nn.Sequential(nn.Linear(6, 5)).to(device)
        lora = LoraModel(net, _config(3, ["0"], use_rslora=True), adapter_name="a")
        layer = _layer(net, 0)
        lora.add_weighted_adapter(["a"], [1.0], "mix")
        assert layer.scaling["mix"] == 1.0
        with lucid.no_grad():
            assert _close(layer.get_delta_weight("mix"), layer.get_delta_weight("a"), atol=1e-5)

    def test_a_failure_while_filling_removes_the_half_made_adapter(
        self, device: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lora, layer = _pair(device)

        def fail(*args: object, **kwargs: object) -> tuple[lucid.Tensor, lucid.Tensor]:
            raise RuntimeError("combine failed")

        monkeypatch.setattr(_Combination, "weights_for", fail)
        with pytest.raises(RuntimeError, match="combine failed"):
            lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "mix")
        assert "mix" not in lora.peft_config and "mix" not in layer.lora_A


# ── saving and loading several adapters ──────────────────────────────────────


def _three(device: str) -> tuple[PeftModel, nn.Sequential]:
    net = _mlp(device)
    peft = get_peft_model(net, _config(2, ["0", "2"]))
    peft.add_adapter("style", _config(3, ["0"]))
    peft.add_adapter("tone", _config(1, ["2"]))
    return peft, net


class TestSaveLoad:
    def test_the_default_is_the_directory_and_the_others_are_sub_directories(
        self, device: str, tmp_path: Path
    ) -> None:
        peft, _ = _three(device)
        peft.save_pretrained(tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "adapter_config.json",
            "adapter_model.safetensors",
            "style",
            "tone",
        ]
        for name in ("style", "tone"):
            assert sorted(p.name for p in (tmp_path / name).iterdir()) == [
                "adapter_config.json",
                "adapter_model.safetensors",
            ]
        keys = sorted(load_safetensors(str(tmp_path / "style" / "adapter_model.safetensors")))
        assert keys == ["base_model.model.0.lora_A.weight", "base_model.model.0.lora_B.weight"]
        config = json.loads((tmp_path / "tone" / "adapter_config.json").read_text(encoding="utf-8"))
        assert config["r"] == 1 and config["target_modules"] == ["2"]
        assert config["inference_mode"] is True

    def test_selected_adapters_write_only_those(self, device: str, tmp_path: Path) -> None:
        peft, _ = _three(device)
        peft.save_pretrained(tmp_path, selected_adapters=["tone"])
        assert [p.name for p in tmp_path.iterdir()] == ["tone"]
        flat = tmp_path / "flat"
        peft.save_pretrained(flat, selected_adapters=["default"])
        assert sorted(p.name for p in flat.iterdir()) == [
            "adapter_config.json",
            "adapter_model.safetensors",
        ]

    def test_an_unknown_selection_writes_nothing(self, device: str, tmp_path: Path) -> None:
        peft, _ = _three(device)
        target = tmp_path / "out"
        with pytest.raises(ValueError, match="invalid `selected_adapters`"):
            peft.save_pretrained(target, selected_adapters=["tone", "ghost"])
        assert not target.exists()

    def test_nothing_is_saved_from_a_spent_model(self, device: str, tmp_path: Path) -> None:
        peft, _ = _three(device)
        peft.merge_and_unload()
        with pytest.raises(RuntimeError, match="merged or unloaded"):
            peft.save_pretrained(tmp_path / "out")
        assert not (tmp_path / "out").exists()

    def test_every_adapter_round_trips(self, device: str, tmp_path: Path) -> None:
        peft, _ = _three(device)
        x = lucid.randn(4, 6, device=device)
        expected = {}
        with lucid.no_grad():
            for name in ("default", "style", "tone"):
                peft.set_adapter(name)
                expected[name] = peft(x)
        peft.save_pretrained(tmp_path)
        fresh = PeftModel.from_pretrained(_mlp(device), tmp_path)
        fresh.load_adapter(tmp_path / "style", "style")
        fresh.load_adapter(tmp_path / "tone", "tone")
        assert sorted(fresh.peft_config) == ["default", "style", "tone"]
        assert fresh.active_adapter == "default"
        with lucid.no_grad():
            for name, output in expected.items():
                fresh.set_adapter(name)
                assert _same(fresh(x), output), name

    def test_a_loaded_adapter_is_frozen_unless_asked(self, device: str, tmp_path: Path) -> None:
        peft, _ = _three(device)
        peft.save_pretrained(tmp_path)
        fresh = PeftModel.from_pretrained(_mlp(device), tmp_path)
        fresh.load_adapter(tmp_path / "style", "style")
        assert not any(p.requires_grad for p in fresh.parameters())
        assert not fresh.training
        assert fresh.peft_config["style"].inference_mode is True
        fresh.load_adapter(tmp_path / "tone", "tone", is_trainable=True)
        names = {n for n, p in fresh.named_parameters() if p.requires_grad}
        assert names and all(".tone." in n for n in names)
        assert fresh.peft_config["tone"].inference_mode is False

    def test_a_loaded_adapter_does_not_take_over(self, device: str, tmp_path: Path) -> None:
        peft, _ = _three(device)
        peft.save_pretrained(tmp_path)
        fresh = PeftModel.from_pretrained(_mlp(device), tmp_path, is_trainable=True)
        before = _trainable(fresh)
        fresh.load_adapter(tmp_path / "style", "style")
        assert fresh.active_adapter == "default" and _trainable(fresh) == before

    def test_loading_into_a_held_adapter_replaces_its_values(
        self, device: str, tmp_path: Path
    ) -> None:
        peft, _ = _three(device)
        peft.save_pretrained(tmp_path)
        fresh = PeftModel.from_pretrained(_mlp(device), tmp_path)
        fresh.load_adapter(tmp_path / "style", "other")
        with lucid.no_grad():
            for param in fresh.parameters():
                param.mul_(0.0)
        fresh.load_adapter(tmp_path / "style", "other")
        assert sorted(fresh.peft_config) == ["default", "other"]
        saved = get_peft_model_state_dict(peft, adapter_name="style")
        loaded = get_peft_model_state_dict(fresh, adapter_name="other")
        assert sorted(loaded) == sorted(saved)
        assert all(_same(loaded[k], saved[k]) for k in saved)

    def test_a_directory_that_does_not_fit_adds_nothing(self, device: str, tmp_path: Path) -> None:
        peft, _ = _three(device)
        peft.save_pretrained(tmp_path)
        other = get_peft_model(
            nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3)).to(device),
            _config(2, ["0"]),
        )
        weights = tmp_path / "style" / "adapter_model.safetensors"
        stored = load_safetensors(str(weights))
        stored["base_model.model.0.lora_A.weight"] = lucid.zeros(3, 7)
        save_safetensors(stored, str(weights), metadata={"format": "pt"})
        keys = sorted(other.state_dict())
        with pytest.raises(ValueError, match="do not fit"):
            other.load_adapter(tmp_path / "style", "style")
        assert sorted(other.state_dict()) == keys and sorted(other.peft_config) == ["default"]

    def test_a_missing_directory_changes_nothing(self, device: str, tmp_path: Path) -> None:
        peft, _ = _three(device)
        with pytest.raises(FileNotFoundError):
            peft.load_adapter(tmp_path / "nowhere", "x")
        assert sorted(peft.peft_config) == ["default", "style", "tone"]

    def test_a_pickled_adapter_is_refused_before_the_model_changes(
        self, device: str, tmp_path: Path
    ) -> None:
        peft, _ = _three(device)
        peft.save_pretrained(tmp_path)
        (tmp_path / "style" / "adapter_model.safetensors").rename(
            tmp_path / "style" / "adapter_model.bin"
        )
        fresh = PeftModel.from_pretrained(_mlp(device), tmp_path)
        with pytest.raises(NotImplementedError, match="pickle"):
            fresh.load_adapter(tmp_path / "style", "style")
        assert sorted(fresh.peft_config) == ["default"]

    def test_an_adapter_written_by_the_reference_loads_beside_another(
        self, device: str, tmp_path: Path
    ) -> None:
        """Keys and config spelled as the reference writes them, not as Lucid does."""
        lucid.manual_seed(3)
        lora_a, lora_b = lucid.randn(2, 6), lucid.randn(8, 2)
        save_safetensors(
            {
                "base_model.model.0.lora_A.weight": lora_a,
                "base_model.model.0.lora_B.weight": lora_b,
            },
            str(tmp_path / "adapter_model.safetensors"),
            metadata={"format": "pt"},
        )
        (tmp_path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "alpha_pattern": {},
                    "auto_mapping": None,
                    "base_model_name_or_path": None,
                    "bias": "none",
                    "fan_in_fan_out": False,
                    "inference_mode": True,
                    "init_lora_weights": True,
                    "layers_pattern": None,
                    "layers_to_transform": None,
                    "lora_alpha": 6,
                    "lora_bias": False,
                    "lora_dropout": 0.0,
                    "modules_to_save": None,
                    "peft_type": "LORA",
                    "peft_version": "0.17.1",
                    "r": 2,
                    "rank_pattern": {},
                    "revision": None,
                    "target_modules": ["0"],
                    "task_type": None,
                    "use_dora": False,
                    "use_rslora": False,
                }
            ),
            encoding="utf-8",
        )
        net = _mlp(device)
        peft = get_peft_model(net, _config(2, ["0", "2"]))
        peft.load_adapter(tmp_path, "reference")
        x = lucid.randn(4, 6, device=device)
        layer = _layer(net, 0)
        peft.set_adapter("reference")
        with lucid.no_grad():
            hidden = layer.base_layer(x) + (x @ lora_a.to(device).T @ lora_b.to(device).T) * 3.0
            expected = net[2](lucid.relu(hidden))
            assert _close(peft(x), expected)

    def test_each_adapters_state_holds_only_what_reaches_it(self, device: str) -> None:
        peft, _ = _three(device)
        assert sorted(get_peft_model_state_dict(peft, adapter_name="tone")) == [
            "base_model.model.2.lora_A.weight",
            "base_model.model.2.lora_B.weight",
        ]
        assert len(get_peft_model_state_dict(peft)) == 4


def test_the_happy_path_raises_no_warning(device: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        peft, _ = _two(device)
        peft.set_adapter("other")
        peft.set_adapter("default")
        peft.delete_adapter("other")
