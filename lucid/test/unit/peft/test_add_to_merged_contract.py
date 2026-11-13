"""An adapter added to a model that holds merged adapters starts as it would on the unmerged model.

A layer that is merged holds the base weight with its adapters' updates folded in.  An adapter whose
start reads the weight (a factorisation, a norm, an identity, a magnitude) must read it as it was
before any merge, or it starts from the sum: a FRoD adapter's identity was another weight (1.2
relative error), DEFT's removal missed by 5e-4, DeLoRA's norms, TinyLoRA's and PSOFT's frozen factors,
SuperTuning's positions and DoRA's magnitude came from the merged weight, and PiSSA / OLoRA wrote a
residual of it back into the base (LCD-975).  The tuner owns it (``_BaseTuner._inject``): the merged
layers are held unmerged while the adapter is built and folded again, as they were, after.

Every method with a recipe that can merge and hold a second adapter is held to it, so a method added
later is checked without a row here:

* what the model holds is unchanged by the add — which layers are merged and with what, every base
  weight bit for bit, the forward pass bit for bit (the new adapter is not in use);
* the new adapter is the one the unmerged model gets, at its start and moved off it (a frozen factor
  derived from the wrong weight shows only once the adapter is trained), whether it is built by
  ``add_adapter``, loaded from a file or made by a second ``get_peft_model``;
* an initialisation that writes the base weights is refused, before anything changes, where there is
  a fold to write into — the model stays as it was, and takes the adapter once it is unmerged;
* a stack of two methods' layers, both merged, is taken apart highest first: LoRA beneath LoHa, IA3, OFT
  or BEFT, and the new adapter reads the weight beneath both;
* what a merge keeps beside the weights (DoRA's row gains) is what it was after the add.
"""

import warnings
from pathlib import Path

import pytest

import lucid
from lucid.peft import LoraConfig, LoraGAConfig, PeftMixedModel, PeftModel, PeftType, get_peft_model
from lucid.peft.tuners._layer import _BaseTunerLayer, _tuner_layers
from lucid.test.unit.peft._tuner_recipes import (
    RECIPES,
    Recipe,
    magnitude,
    perturb_adapter,
    same,
    span,
)

#: Relative to the larger of 1 and the output: a float32 weight loses a ulp or two folded out and in.
_TOLERANCE = 1e-5

_ADDS = sorted((t for t, r in RECIPES.items() if r.mergeable and r.adds), key=str)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Adapter cannot be set when the model is merged:UserWarning",
    "ignore:You are trying to modify a model with PEFT for a second time:UserWarning",
    "ignore:Unmerge result can be inaccurate:UserWarning",
)


def _first(recipe: Recipe, device: str, scratch: Path, *, merged: bool) -> PeftModel:
    """The recipe's model with its first adapter moved off the start, folded into the weights if ``merged``."""
    model = get_peft_model(
        recipe.base(lucid.float32, device), recipe.config(lucid.float32, device, scratch)
    )
    perturb_adapter(model, "default", seed=5)
    if merged:
        model.base_model.merge_adapter()
    return model


def _base_weights(model: PeftModel) -> dict[str, lucid.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if "base_layer" in n}


def _folded(model: PeftModel) -> list[tuple[str, bool, list[str]]]:
    return [(key, layer.merged, list(layer.merged_adapters)) for key, layer in _tuner_layers(model)]


def _unfolded(model: PeftModel) -> bool:
    return not any(merged for _, merged, _ in _folded(model))


def _output_of_the_new(
    recipe: Recipe, model: PeftModel, device: str, *, trained: bool
) -> lucid.Tensor:
    """What the model computes with only adapter ``b`` in use, ``b`` moved off its start if ``trained``."""
    model.set_adapter("b")
    if trained:
        perturb_adapter(model, "b", seed=11)
    with lucid.no_grad():
        return recipe.call(model, lucid.float32, device)


@pytest.mark.parametrize("peft_type", _ADDS, ids=lambda t: t.name)
def test_adding_an_adapter_leaves_a_merged_model_as_it_was(
    peft_type: PeftType, device: str, tmp_path: Path
) -> None:
    recipe = RECIPES[peft_type]
    model = _first(recipe, device, tmp_path, merged=True)
    assert not _unfolded(model)
    weights, folded = _base_weights(model), _folded(model)
    with lucid.no_grad():
        out = recipe.call(model, lucid.float32, device)

    lucid.manual_seed(7)
    model.add_adapter("b", recipe.second(lucid.float32, device, tmp_path))

    assert _folded(model) == folded
    after = _base_weights(model)
    assert after.keys() == weights.keys()
    assert all(same(after[name], value) for name, value in weights.items())
    with lucid.no_grad():
        assert same(recipe.call(model, lucid.float32, device), out)


@pytest.mark.parametrize("trained", [False, True], ids=["at its start", "moved off it"])
@pytest.mark.parametrize("peft_type", _ADDS, ids=lambda t: t.name)
def test_the_adapter_added_to_a_merged_model_is_the_one_an_unmerged_model_gets(
    peft_type: PeftType, trained: bool, device: str, tmp_path: Path
) -> None:
    recipe = RECIPES[peft_type]
    outs = {}
    for merged in (False, True):
        model = _first(recipe, device, tmp_path, merged=merged)
        lucid.manual_seed(7)
        model.add_adapter("b", recipe.second(lucid.float32, device, tmp_path))
        outs[merged] = _output_of_the_new(recipe, model, device, trained=trained)
    out = outs[False]
    assert span(outs[True], out) <= _TOLERANCE * max(1.0, magnitude(out))


#: LN tuning runs one adapter at a time, so ``default`` and ``b`` cannot be in use together.
_TOGETHER = [t for t in _ADDS if t is not PeftType.LN_TUNING]


def _what_the_new_adds(recipe: Recipe, model: PeftModel, device: str) -> lucid.Tensor:
    """What adapter ``b``, at its start, adds to what ``default`` computes alone."""
    model.set_adapter("default")
    with lucid.no_grad():
        alone = recipe.call(model, lucid.float32, device)
        model.set_adapter(["default", "b"])
        return recipe.call(model, lucid.float32, device) - alone


@pytest.mark.parametrize("peft_type", _TOGETHER, ids=lambda t: t.name)
def test_the_new_adapter_adds_to_a_merged_model_what_it_adds_to_an_unmerged_one(
    peft_type: PeftType, device: str, tmp_path: Path
) -> None:
    recipe = RECIPES[peft_type]
    adds = {}
    for merged in (False, True):
        model = _first(recipe, device, tmp_path, merged=merged)
        lucid.manual_seed(7)
        model.add_adapter("b", recipe.second(lucid.float32, device, tmp_path))
        adds[merged] = _what_the_new_adds(recipe, model, device)
    assert span(adds[True], adds[False]) <= _TOLERANCE * max(1.0, magnitude(adds[False]))


@pytest.mark.parametrize("peft_type", [PeftType.FROD, PeftType.DEFT], ids=lambda t: t.name)
def test_an_adapter_that_starts_as_the_identity_adds_nothing_to_a_merged_model(
    peft_type: PeftType, device: str, tmp_path: Path
) -> None:
    """FRoD and DEFT re-express the weight: at the start the new adapter is the weight it was derived from."""
    recipe = RECIPES[peft_type]
    model = _first(recipe, device, tmp_path, merged=True)
    lucid.manual_seed(7)
    model.add_adapter("b", recipe.second(lucid.float32, device, tmp_path))
    adds = _what_the_new_adds(recipe, model, device)
    assert magnitude(adds) <= _TOLERANCE


@pytest.mark.parametrize("peft_type", _ADDS, ids=lambda t: t.name)
def test_a_merge_that_changes_no_weight_would_prove_nothing(
    peft_type: PeftType, device: str, tmp_path: Path
) -> None:
    """The two worlds differ in the weights the second adapter could read, for every method held to the rest."""
    recipe = RECIPES[peft_type]
    plain = _base_weights(_first(recipe, device, tmp_path, merged=False))
    merged = _base_weights(_first(recipe, device, tmp_path, merged=True))
    assert any(span(merged[name], plain[name]) > 1e-3 for name in plain), "the merge wrote nothing"


@pytest.mark.parametrize("peft_type", _ADDS, ids=lambda t: t.name)
def test_an_adapter_loaded_into_a_merged_model_is_the_one_an_unmerged_model_gets(
    peft_type: PeftType, device: str, tmp_path: Path
) -> None:
    """The file carries what training moved; what is derived from the weight is made again when it is loaded."""
    recipe = RECIPES[peft_type]
    saved = get_peft_model(
        recipe.base(lucid.float32, device), recipe.second(lucid.float32, device, tmp_path)
    )
    perturb_adapter(saved, "default", seed=3)
    saved.save_pretrained(tmp_path / "file")
    outs = {}
    for merged in (False, True):
        model = _first(recipe, device, tmp_path, merged=merged)
        model.load_adapter(tmp_path / "file", "b", is_trainable=False)
        outs[merged] = _output_of_the_new(recipe, model, device, trained=False)
    out = outs[False]
    assert span(outs[True], out) <= _TOLERANCE * max(1.0, magnitude(out))


@pytest.mark.parametrize(
    "peft_type",
    [PeftType.FROD, PeftType.DEFT, PeftType.DELORA, PeftType.PSOFT],
    ids=lambda t: t.name,
)
def test_a_second_get_peft_model_on_a_merged_model_adds_the_unmerged_models_adapter(
    peft_type: PeftType, device: str, tmp_path: Path
) -> None:
    """The constructor of a model that holds adapters builds through the same start, then unmerges."""
    recipe = RECIPES[peft_type]
    outs = {}
    for merged in (False, True):
        first = _first(recipe, device, tmp_path, merged=merged)
        net = first.get_base_model()
        lucid.manual_seed(7)
        second = get_peft_model(
            net, recipe.second(lucid.float32, device, tmp_path), adapter_name="b"
        )
        assert _unfolded(second)
        perturb_adapter(second, "b", seed=11)
        with lucid.no_grad():
            outs[merged] = recipe.call(second, lucid.float32, device)
    out = outs[False]
    assert span(outs[True], out) <= _TOLERANCE * max(1.0, magnitude(out))


def test_only_the_merged_layers_the_adapter_reaches_are_taken_apart(
    device: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A layer whose weight the new adapter never reads is left alone, not folded out and in."""
    recipe = RECIPES[PeftType.LORA]
    model = _first(recipe, device, tmp_path, merged=True)
    layers = dict(_tuner_layers(model))
    reached, apart = layers["base_model.model.2"], layers["base_model.model.0"]
    taken: list[int] = []
    unmerge = type(reached).unmerge

    def record(self: _BaseTunerLayer) -> None:
        taken.append(id(self))
        unmerge(self)

    monkeypatch.setattr(type(reached), "unmerge", record)
    model.add_adapter("b", _lora(targets=("2",)))
    assert taken == [id(reached)]
    assert reached.merged_adapters == apart.merged_adapters == ["default"]


# ── LoRA's starts: a factor of the weight, and a write into it ──────────────────────────────────


def _lora(
    *,
    targets: tuple[str, ...] = ("0", "2"),
    use_dora: bool = False,
    init: bool | str = True,
    ga: LoraGAConfig | None = None,
) -> LoraConfig:
    return LoraConfig(
        r=2,
        target_modules=list(targets),
        use_dora=use_dora,
        init_lora_weights=init,
        lora_ga_config=ga,
    )


def test_a_dora_adapter_added_to_a_merged_model_takes_its_magnitude_from_the_weight_before_the_merge(
    device: str, tmp_path: Path
) -> None:
    recipe = RECIPES[PeftType.LORA]
    outs = {}
    for merged in (False, True):
        model = _first(recipe, device, tmp_path, merged=merged)
        lucid.manual_seed(7)
        model.add_adapter("b", _lora(use_dora=True))
        outs[merged] = _output_of_the_new(recipe, model, device, trained=False)
    out = outs[False]
    assert span(outs[True], out) <= _TOLERANCE * max(1.0, magnitude(out))


_WRITING = {
    "pissa": lambda: _lora(init="pissa"),
    "pissa_niter_4": lambda: _lora(init="pissa_niter_4"),
    "olora": lambda: _lora(init="olora"),
    "lora_ga": lambda: _lora(init="lora_ga", ga=LoraGAConfig()),
}


@pytest.mark.parametrize("init", list(_WRITING))
def test_an_initialisation_that_writes_the_base_is_refused_where_adapters_are_folded_in(
    init: str, device: str, tmp_path: Path
) -> None:
    """A rewrite of the base has no single meaning beside a fold, so nothing is changed and the model is told what to do."""
    recipe = RECIPES[PeftType.LORA]
    model = _first(recipe, device, tmp_path, merged=True)
    weights, folded = _base_weights(model), _folded(model)
    with pytest.raises(ValueError, match=r"rewrites the base weights.*unmerge_adapter\(\)"):
        model.add_adapter("b", _WRITING[init]())
    assert sorted(model.peft_config) == ["default"]
    assert _folded(model) == folded
    assert all(same(p, weights[n]) for n, p in model.named_parameters() if n in weights)

    model.base_model.unmerge_adapter()
    if init != "lora_ga":  # LoRA-GA needs gradients it is given elsewhere
        model.add_adapter("b", _WRITING[init]())
        assert sorted(model.peft_config) == ["b", "default"]


# ── a stack: the layers are taken apart highest first ────────────────────────────────────────────

#: The layer above LoRA's, each folded after it: LoHa adds to the output and commutes with it, IA3
#: scales it, OFT and BEFT rotate it, so a stack taken apart lowest first reads another weight.
_UPPER = [PeftType.LOHA, PeftType.IA3, PeftType.OFT, PeftType.BEFT]


def _stack(upper: PeftType, device: str, scratch: Path, *, merged: bool) -> PeftMixedModel:
    recipe = RECIPES[upper]
    model = PeftMixedModel(recipe.base(lucid.float32, device), _lora(), "lora")
    model.add_adapter("upper", recipe.config(lucid.float32, device, scratch))
    perturb_adapter(model, "lora", seed=5)
    perturb_adapter(model, "upper", seed=6)
    model.set_adapter(["lora", "upper"])
    if merged:
        model.base_model.merge_adapter()
    return model


@pytest.mark.parametrize("upper", _UPPER, ids=lambda t: t.name)
def test_an_adapter_added_to_a_merged_stack_takes_the_weight_from_beneath_every_layer(
    upper: PeftType, device: str, tmp_path: Path
) -> None:
    outs = {}
    for merged in (False, True):
        model = _stack(upper, device, tmp_path, merged=merged)
        weights, folded = _base_weights(model), _folded(model)
        lucid.manual_seed(7)
        model.add_adapter("dora", _lora(use_dora=True))
        assert _folded(model) == folded
        assert all(same(p, weights[n]) for n, p in model.named_parameters() if n in weights)
        model.set_adapter(["dora"])
        with lucid.no_grad():
            outs[merged] = model(lucid.randn(3, 6, device=device))
    assert span(outs[True], outs[False]) <= _TOLERANCE * max(1.0, magnitude(outs[False]))


# ── what a merge keeps beside the weights ───────────────────────────────────────────────────────


def test_the_row_gains_of_merged_dora_adapters_are_what_they_were_after_an_add(device: str) -> None:
    """Folding DoRA again recomputes its gains from a weight that lost bits unmerging: they are put back."""
    recipe = RECIPES[PeftType.LORA]
    model = get_peft_model(
        recipe.base(lucid.float32, device), _lora(use_dora=True), adapter_name="one"
    )
    model.add_adapter("two", _lora(use_dora=True))
    perturb_adapter(model, "one", seed=5)
    perturb_adapter(model, "two", seed=6)
    model.set_adapter(["one", "two"])
    original = {n: p.detach().clone() for n, p in model.named_parameters() if "base_layer" in n}
    model.base_model.merge_adapter()
    layers = [layer for _, layer in _tuner_layers(model)]
    gains = [
        {name: layer.lora_magnitude_vector[name].merged_gain for name in layer.merged_adapters}
        for layer in layers
    ]
    assert all(gain is not None for held in gains for gain in held.values())

    model.add_adapter("b", _lora())

    for layer, held in zip(layers, gains, strict=True):
        assert layer.merged_adapters == ["one", "two"]
        for name, gain in held.items():
            assert gain is not None
            assert same(layer.lora_magnitude_vector[name].merged_gain, gain)
    model.base_model.unmerge_adapter()
    restored = _TOLERANCE * max(magnitude(w) for w in original.values())
    assert all(
        span(p, original[n]) <= restored for n, p in model.named_parameters() if n in original
    )


def test_adding_to_a_merged_model_does_not_warn_that_an_unmerge_is_inexact(
    device: str, tmp_path: Path
) -> None:
    """IA3's unmerge divides and says so; here the weights it read are written back as they were."""
    recipe = RECIPES[PeftType.IA3]
    model = _first(recipe, device, tmp_path, merged=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.add_adapter("b", recipe.second(lucid.float32, device, tmp_path))
