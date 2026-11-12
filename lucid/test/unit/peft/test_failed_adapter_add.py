"""An add, a load or a build that fails leaves its model as it was (every registered method).

Adding an adapter puts more into a model than the adapter's tensors: a layer on each module the
adapter targets and no earlier adapter did, a ``modules_to_save`` wrapper, names the tuner records.
``delete_adapter`` keeps the layers (another adapter may stand on one), so a flow that adds an adapter
and then fails — filling it from a file that does not fit — has to take back what the add put in, not
delete the adapter.  The host holds that undo (``_add_adapter`` returns it); these tests hold every host
to it, with a snapshot of the whole model before and after:

* **A load of a new adapter that does not fit**: a ``PeftModel`` of every method whose second adapter
  targets a module the first did not, the prompt hosts and ``AdaptionPrompt`` whose second adapter
  trains a module in full, and a ``PeftMixedModel`` of the ten mixable methods beside and on top of
  another method's layers.  The file holds a tensor the model has no place for.
* **A build over a model that holds adapters**: ``from_pretrained`` of a file that does not fit takes
  out the adapter it added and keeps the one the model held; so does a constructor that raises.
* **An add that fails late** and a weighted adapter that cannot be filled.

A method added later is held to all of it without a row here.
"""

import dataclasses
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    PeftConfig,
    PeftMixedModel,
    PeftModel,
    PeftType,
    get_peft_model,
    inject_adapter_in_model,
)
from lucid.peft._slots import _adapter_slots, _SlotState
from lucid.peft.tuners._base import _TUNERS
from lucid.peft.tuners._layer import _BaseTunerLayer
from lucid.peft.tuners._prompt import _tuner as prompt_tuner
from lucid.peft.tuners._prompt._tuner import _PromptLearningTuner
from lucid.peft.tuners._tuner import _FLAGS_KEY, _MIXED_TUNERS
from lucid.serialization import load_safetensors, save_safetensors
from lucid.test.unit.peft._tuner_recipes import RECIPES, Recipe, same

pytestmark = pytest.mark.filterwarnings(
    "ignore:You are trying to modify a model with PEFT for a second time:UserWarning"
)

_F32 = lucid.float32
_STRAY = "base_model.model.no_such_module.weight"

#: What a tuner sets on the model it adapts: the config table, and the flags it had before the first adapter.
_ADAPTED_ATTRIBUTES = frozenset({"peft_config", _FLAGS_KEY})
#: A prompt in front of a language model, or the attention prompt: the methods run on the text model.
_PROMPTS = sorted(
    (t for t, host in _TUNERS.items() if issubclass(host, _PromptLearningTuner)), key=str
)
_TEXT = [PeftType.ADAPTION_PROMPT, *_PROMPTS]
#: Every method whose adapters wrap the modules its config names (X-LoRA mixes saved experts).
_TARGETED = sorted((t for t, r in RECIPES.items() if r.adds and t not in _TEXT), key=str)
_MIXED = sorted(_MIXED_TUNERS, key=str)
#: A shared projection is sized by the largest layer of the first adapter, so a second adapter that
#: reaches a smaller or larger layer than the first's is refused for another reason: these three
#: run on layers of one shape.
_SQUARE = frozenset({PeftType.VERA, PeftType.PVERA, PeftType.RANDLORA})
#: AdaLoRA trains one adapter at a time: a second one that trains is refused before the add.
_REFUSED_EARLY: dict[tuple[PeftType, bool], str] = {
    (PeftType.ADALORA, True): "supports only 1 trainable adapter"
}
#: HiRA reads the weight beneath it, so no other method's layer is beneath it: the add is refused
#: before the model is touched, which holds the model as it was just the same.
_REFUSED_ON_TOP: dict[PeftType, str] = {PeftType.HIRA: "cannot add an adapter"}


def _square(dtype: lucid.dtype, device: str) -> nn.Module:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)).to(device).to(dtype)


def _recipe(peft_type: PeftType) -> Recipe:
    recipe = RECIPES[peft_type]
    return dataclasses.replace(recipe, base=_square) if peft_type in _SQUARE else recipe


def _narrowed(config: PeftConfig, module: str) -> PeftConfig:
    """``config`` aimed at the module ``module`` alone: the recipes' configs name the audit's two."""
    targets = getattr(config, "target_modules", None)
    assert set(targets or ()) == {"0", "2"}, "the recipe's config no longer names the two layers"
    return dataclasses.replace(config, target_modules=[module])


def _saved(config: PeftConfig, recipe: Recipe, device: str, directory: Path) -> Path:
    """A ``PeftModel`` of ``config`` over the recipe's base, saved in ``directory``."""
    get_peft_model(recipe.base(_F32, device), config).save_pretrained(directory)
    return directory


def _unfit(directory: Path, into: Path) -> Path:
    """A copy of the saved adapter in ``directory`` whose file also holds a tensor no model has a place for."""
    shutil.copytree(directory, into)
    path = into / "adapter_model.safetensors"
    weights = load_safetensors(path)
    weights[_STRAY] = lucid.zeros(2, 2)
    save_safetensors(weights, str(path))
    return into


class _Snapshot(NamedTuple):
    """Everything a failed add, load or build must leave alone."""

    modules: list[tuple[str, int, type[nn.Module]]]
    state: dict[str, lucid.Tensor]
    parameters: list[tuple[str, int, bool]]
    attributes: list[str]
    adapted: list[tuple[str, int]]
    hooks: list[int]
    configs: dict[str, int]
    active: list[str]
    slots: list[tuple[str, int, _SlotState]]
    targeted: tuple[list[str], list[str]]
    encoders: list[str]


def _snapshot(wrapper: PeftModel | PeftMixedModel) -> _Snapshot:
    host = wrapper.base_model
    model = host.model
    return _Snapshot(
        [(n, id(m), type(m)) for n, m in model.named_modules(remove_duplicate=False)],
        {k: v.detach().clone() for k, v in model.state_dict().items()},
        [(n, id(p), p.requires_grad) for n, p in model.named_parameters(remove_duplicate=False)],
        sorted(model.__dict__),
        [(k, id(v)) for k, v in sorted(model.__dict__.items()) if k in _ADAPTED_ATTRIBUTES],
        [len(m._forward_pre_hooks) for _, m in model.named_modules(remove_duplicate=False)],
        {name: id(config) for name, config in wrapper.peft_config.items()},
        list(wrapper.active_adapters),
        [(key, id(slot), slot._state()) for key, slot in _adapter_slots(model)],
        (
            list(getattr(host, "targeted_module_names", [])),
            list(getattr(host, "targeted_parameter_names", [])),
        ),
        list(getattr(host, "prompt_encoder", [])),
    )


def _assert_unchanged(before: _Snapshot, after: _Snapshot) -> None:
    """Field by field, so the failure says which part of the model changed."""
    for field in _Snapshot._fields:
        if field == "state":
            assert after.state.keys() == before.state.keys(), "state_dict keys"
            for key, value in before.state.items():
                assert same(after.state[key], value), f"{key} was rewritten"
        else:
            assert getattr(after, field) == getattr(before, field), field


def _refused(call: Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


# ── a load of a new adapter that does not fit ─────────────────────────────


def test_every_registered_method_has_a_row_in_the_recipes() -> None:
    assert set(_TUNERS) == set(RECIPES)
    assert {*_TARGETED, *_TEXT, PeftType.XLORA} == set(RECIPES)
    assert len(_MIXED) == 10 and set(_MIXED) <= set(_TARGETED)


@pytest.mark.parametrize("is_trainable", [False, True])
@pytest.mark.parametrize("peft_type", _TARGETED, ids=lambda t: t.name)
def test_a_load_that_does_not_fit_leaves_a_peft_model_as_it_was(
    peft_type: PeftType, is_trainable: bool, device: str, tmp_path: Path
) -> None:
    recipe = _recipe(peft_type)
    first = _narrowed(recipe.config(_F32, device, tmp_path), "2")  # the second one reaches "0" too
    good = _saved(recipe.second(_F32, device, tmp_path), recipe, device, tmp_path / "good")
    bad = _unfit(good, tmp_path / "bad")
    peft = get_peft_model(recipe.base(_F32, device), first, adapter_name="first")
    before = _snapshot(peft)
    match = _REFUSED_EARLY.get((peft_type, is_trainable), "do not fit this model")
    _refused(lambda: peft.load_adapter(bad, "second", is_trainable=is_trainable), match)
    _assert_unchanged(before, _snapshot(peft))
    # Nothing it left behind stands in the way of the same adapter coming from a good file.
    peft.load_adapter(good, "second", is_trainable=False)
    assert sorted(peft.peft_config) == ["first", "second"]


@pytest.mark.parametrize("peft_type", _TEXT, ids=lambda t: t.name)
def test_a_load_whose_adapter_trains_a_module_leaves_a_text_model_as_it_was(
    peft_type: PeftType, device: str, tmp_path: Path
) -> None:
    recipe = RECIPES[peft_type]
    # The second adapter trains the head in full: a wrapper goes in the tree.
    second = dataclasses.replace(recipe.second(_F32, device, tmp_path), modules_to_save=["lm_head"])
    good = _saved(second, recipe, device, tmp_path / "good")
    bad = _unfit(good, tmp_path / "bad")
    peft = get_peft_model(recipe.base(_F32, device), recipe.config(_F32, device, tmp_path))
    before = _snapshot(peft)
    _refused(lambda: peft.load_adapter(bad, "second"), "do not fit this model")
    _assert_unchanged(before, _snapshot(peft))
    peft.load_adapter(good, "second")
    assert sorted(peft.peft_config) == ["default", "second"]


@pytest.mark.parametrize("stacking", ["beside", "on top"])
@pytest.mark.parametrize("peft_type", _MIXED, ids=lambda t: t.name)
def test_a_load_that_does_not_fit_leaves_a_mixed_model_as_it_was(
    peft_type: PeftType, stacking: str, device: str, tmp_path: Path
) -> None:
    recipe = RECIPES[peft_type]
    if stacking == "beside":
        # The same method on the module the second adapter reaches besides: a layer of its own.
        first = _narrowed(recipe.config(_F32, device, tmp_path), "2")
        match = "do not fit this model"
    else:
        # Another method's layer on both modules: the second adapter's layers go on top of them.
        under = PeftType.LORA if peft_type is not PeftType.LORA else PeftType.LOHA
        first = RECIPES[under].config(_F32, device, tmp_path)
        match = _REFUSED_ON_TOP.get(peft_type, "do not fit this model")
    good = _saved(recipe.second(_F32, device, tmp_path), recipe, device, tmp_path / "good")
    bad = _unfit(good, tmp_path / "bad")
    mixed = PeftMixedModel(recipe.base(_F32, device), first, "first")
    before = _snapshot(mixed)
    _refused(lambda: mixed.load_adapter(bad, "second"), match)
    _assert_unchanged(before, _snapshot(mixed))
    if match == "do not fit this model":
        mixed.load_adapter(good, "second")
        assert sorted(mixed.peft_config) == ["first", "second"]


@pytest.mark.parametrize("kind", [PeftModel, PeftMixedModel], ids=lambda k: k.__name__)
def test_a_load_into_a_merged_model_leaves_it_merged(
    kind: type[PeftModel] | type[PeftMixedModel], device: str, tmp_path: Path
) -> None:
    recipe = RECIPES[PeftType.LORA]
    first = _narrowed(recipe.config(_F32, device, tmp_path), "2")
    good = _saved(recipe.config(_F32, device, tmp_path), recipe, device, tmp_path / "good")
    bad = _unfit(good, tmp_path / "bad")
    wrapped = kind(recipe.base(_F32, device), first, "first")
    wrapped.base_model.merge_adapter()
    before = _snapshot(wrapped)
    assert any(state.merged for _, _, state in before.slots)
    _refused(lambda: wrapped.load_adapter(bad, "second"), "do not fit this model")
    _assert_unchanged(before, _snapshot(wrapped))


# ── a build over a model that holds an adapter ────────────────────────────


@pytest.mark.parametrize("peft_type", _TARGETED, ids=lambda t: t.name)
def test_a_from_pretrained_that_does_not_fit_keeps_the_adapter_the_model_held(
    peft_type: PeftType, device: str, tmp_path: Path
) -> None:
    recipe = _recipe(peft_type)
    first = _narrowed(recipe.config(_F32, device, tmp_path), "2")
    good = _saved(recipe.second(_F32, device, tmp_path), recipe, device, tmp_path / "good")
    bad = _unfit(good, tmp_path / "bad")
    net = recipe.base(_F32, device)
    held = get_peft_model(net, first, adapter_name="held")
    before = _snapshot(held)
    _refused(
        lambda: PeftModel.from_pretrained(net, bad, adapter_name="new"), "do not fit this model"
    )
    _assert_unchanged(before, _snapshot(held))
    PeftModel.from_pretrained(net, good, adapter_name="new")  # the model takes it from a good file


@pytest.mark.parametrize("peft_type", _TARGETED, ids=lambda t: t.name)
def test_a_constructor_that_raises_keeps_the_adapter_the_model_held(
    peft_type: PeftType, device: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe(peft_type)
    first = _narrowed(recipe.config(_F32, device, tmp_path), "2")
    net = recipe.base(_F32, device)
    held = get_peft_model(net, first, adapter_name="held")
    before = _snapshot(held)

    def refuse(self: object) -> None:
        raise RuntimeError("refused in the last step")

    monkeypatch.setattr(_TUNERS[peft_type], "_mark_only_adapters_as_trainable", refuse)
    with pytest.raises(RuntimeError, match="refused in the last step"):
        inject_adapter_in_model(recipe.second(_F32, device, tmp_path), net, "new")
    _assert_unchanged(before, _snapshot(held))


# ── an add that fails late ────────────────────────────────────────────────


@pytest.mark.parametrize("peft_type", _TARGETED, ids=lambda t: t.name)
def test_an_add_that_fails_on_its_second_layer_leaves_a_peft_model_as_it_was(
    peft_type: PeftType, device: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe(peft_type)
    first = _narrowed(recipe.config(_F32, device, tmp_path), "2")
    peft = get_peft_model(recipe.base(_F32, device), first, adapter_name="first")
    before = _snapshot(peft)
    widened = _BaseTunerLayer._widen_adapter
    calls: list[str] = []

    def widen(self: _BaseTunerLayer, adapter_name: str) -> None:
        calls.append(adapter_name)
        if len(calls) == 2:
            raise RuntimeError("refused on the second layer")
        widened(self, adapter_name)

    monkeypatch.setattr(_BaseTunerLayer, "_widen_adapter", widen)
    with pytest.raises(RuntimeError, match="refused on the second layer"):
        peft.add_adapter("second", recipe.second(_F32, device, tmp_path))
    monkeypatch.undo()
    _assert_unchanged(before, _snapshot(peft))
    peft.add_adapter("second", recipe.second(_F32, device, tmp_path))  # and takes it again
    assert sorted(peft.peft_config) == ["first", "second"]


def test_a_prompt_host_that_cannot_place_its_second_wrapper_leaves_the_model_as_it_was(
    device: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = RECIPES[PeftType.PROMPT_TUNING]
    peft = get_peft_model(recipe.base(_F32, device), recipe.config(_F32, device, tmp_path))
    before = _snapshot(peft)
    placed = prompt_tuner._set_submodule
    calls: list[str] = []

    def place(model: nn.Module, key: str, module: nn.Module) -> None:
        calls.append(key)
        if len(calls) == 2:
            raise RuntimeError("refused on the second wrapper")
        placed(model, key, module)

    monkeypatch.setattr(prompt_tuner, "_set_submodule", place)
    second = dataclasses.replace(
        recipe.second(_F32, device, tmp_path), modules_to_save=["lm_head", "transformer.ln_f"]
    )
    with pytest.raises(RuntimeError, match="refused on the second wrapper"):
        peft.add_adapter("second", second)
    monkeypatch.undo()
    _assert_unchanged(before, _snapshot(peft))


@pytest.mark.parametrize(
    ("peft_type", "fill"),
    [(PeftType.LORA, "_fill_weighted_adapter"), (PeftType.IA3, "_sum_vectors")],
    ids=["LORA", "IA3"],
)
def test_a_weighted_adapter_that_cannot_be_filled_leaves_the_model_as_it_was(
    peft_type: PeftType, fill: str, device: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = RECIPES[peft_type]
    peft = get_peft_model(
        recipe.base(_F32, device), recipe.config(_F32, device, tmp_path), adapter_name="a"
    )
    peft.add_adapter("b", recipe.second(_F32, device, tmp_path))
    before = _snapshot(peft)

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("refused while filling")

    monkeypatch.setattr(type(peft.base_model), fill, refuse)
    with pytest.raises(RuntimeError, match="refused while filling"):
        peft.base_model.add_weighted_adapter(["a", "b"], [0.5, 0.5], "mix")
    _assert_unchanged(before, _snapshot(peft))
