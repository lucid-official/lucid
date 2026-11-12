"""Several active adapters: the merged model is the adapted one, whatever order they are listed in.

Merging folds the active adapters into the base weights; the forward pass applies them to the
input.  The two must be one function for any number of adapters in any order — an adapter that
rotates or rewrites the weight does not commute with another, so "any order" is the case that
can differ.  Every method in ``_TUNERS`` is held to it with two adapters, both away from their
starting values, active in each order: merged output equals the adapted output, unmerging gives
the base weights and the adapted output back (to rounding: a float sum or product is not
undone bit for bit), and ``merge_and_unload`` returns a plain model that computes the adapted
output.  A method that applies one adapter at a time, or whose adapter cannot be folded, must say
so at the point it refuses (``_REFUSED``); one that is neither is held to the contract, so a
method added to the registry is checked without anyone remembering to.

Merging only some of the active adapters is held to the same rule (LCD-971): the layer goes on
applying the rest, so the model computes what it did unmerged, whichever adapters were folded.
The reference runs a merged layer's base alone and drops them; Lucid departs from it on purpose.
A method whose adapters apply in order (``_IN_ORDER``) can fold only the first of the active
ones, and refuses the rest before it writes anything.
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.peft import (
    BOFTConfig,
    BoneConfig,
    HRAConfig,
    LNTuningConfig,
    LoHaConfig,
    LoraConfig,
    MissConfig,
    OFTConfig,
    PeftConfig,
    PeftModel,
    PeftType,
    RoadConfig,
)
from lucid.peft import get_peft_model
from lucid.peft.tuners._base import _TUNERS
from lucid.peft.tuners._layer import _BaseTunerLayer, _tuner_layers
from lucid.test.unit.peft._tuner_recipes import (
    RECIPES,
    Recipe,
    adapted,
    magnitude,
    perturb_adapter,
    same,
    span,
)

_ORDERS = [("default", "b"), ("b", "default")]

#: Relative to the largest output: a float32 sum or product, and the inverse of a rotation, round.
_TOLERANCE = 1e-5

_ONE_AT_A_TIME = "one adapter at a time"
#: Where a method that cannot hold or fold several adapters refuses, with what error and message.
_REFUSED: dict[PeftType, tuple[str, type[Exception], str]] = {
    PeftType.LN_TUNING: ("forward", ValueError, "more than one adapter at a time"),
    PeftType.POLY: ("merge", NotImplementedError, "cannot be folded"),
    PeftType.LILY: ("merge", NotImplementedError, "cannot be folded"),
    PeftType.ADAPTION_PROMPT: ("activate", ValueError, _ONE_AT_A_TIME),
    PeftType.PROMPT_TUNING: ("activate", ValueError, _ONE_AT_A_TIME),
    PeftType.PREFIX_TUNING: ("activate", ValueError, _ONE_AT_A_TIME),
    PeftType.P_TUNING: ("activate", ValueError, _ONE_AT_A_TIME),
    PeftType.MULTITASK_PROMPT_TUNING: ("activate", ValueError, _ONE_AT_A_TIME),
    PeftType.CPT: ("activate", ValueError, _ONE_AT_A_TIME),
    PeftType.CARTRIDGE: ("activate", ValueError, _ONE_AT_A_TIME),
    PeftType.XLORA: ("add", NotImplementedError, "holds one adapter"),
}

#: The audit's configs rotate blocks of two, which commute: these rotate blocks of four, which do
#: not, so the order the adapters are listed in changes the function and a wrong fold shows.
_NON_COMMUTING: dict[PeftType, Callable[[], PeftConfig]] = {
    PeftType.OFT: lambda: OFTConfig(
        oft_block_size=4, use_cayley_neumann=False, target_modules=["2"]
    ),
    PeftType.BOFT: lambda: BOFTConfig(boft_block_size=4, target_modules=["2"]),
}


def _recipe(peft_type: PeftType) -> Recipe:
    recipe = RECIPES[peft_type]
    make = _NON_COMMUTING.get(peft_type)
    if make is None:
        return recipe
    return dataclasses.replace(recipe, config=lambda dtype, device, scratch: make())


def _two(
    peft_type: PeftType, device: str, scratch: Path
) -> tuple[Recipe, list[lucid.Tensor], PeftModel]:
    """``(recipe, own, model)``: the method with adapters ``default`` and ``b`` in float32."""
    recipe = _recipe(peft_type)
    own, model = adapted(recipe, lucid.float32, device, scratch)
    model.add_adapter("b", recipe.second(lucid.float32, device, scratch))
    return recipe, own, model


def test_no_method_is_registered_without_a_recipe_or_a_refusal_for_nothing() -> None:
    assert set(_TUNERS) <= set(RECIPES), sorted(t.name for t in set(_TUNERS) - set(RECIPES))
    assert set(_REFUSED) <= set(_TUNERS), sorted(t.name for t in set(_REFUSED) - set(_TUNERS))


@pytest.mark.parametrize("peft_type", list(_REFUSED), ids=lambda t: t.name)
def test_a_method_that_cannot_hold_several_says_where(
    peft_type: PeftType, device: str, tmp_path: Path
) -> None:
    stage, error, reason = _REFUSED[peft_type]
    recipe = _recipe(peft_type)
    _, model = adapted(recipe, lucid.float32, device, tmp_path)
    second = recipe.second(lucid.float32, device, tmp_path)
    if stage == "add":
        with pytest.raises(error, match=reason):
            model.add_adapter("b", second)
        return
    model.add_adapter("b", second)
    if stage == "activate":
        with pytest.raises(error, match=reason):
            model.set_adapter(list(_ORDERS[0]))
        return
    model.set_adapter(list(_ORDERS[0]))
    if stage == "forward":
        with pytest.raises(error, match=reason):
            recipe.call(model, lucid.float32, device)
        return
    recipe.call(model, lucid.float32, device)  # both apply; only folding them is refused
    with pytest.raises(error, match=reason):
        model.base_model.merge_adapter()


@pytest.mark.parametrize(
    "peft_type", [t for t in _TUNERS if t not in _REFUSED], ids=lambda t: t.name
)
@pytest.mark.parametrize("order", _ORDERS, ids="-".join)
@pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
def test_merged_several_adapters_are_the_adapted_model(
    peft_type: PeftType, order: tuple[str, str], device: str, tmp_path: Path
) -> None:
    recipe, own, model = _two(peft_type, device, tmp_path)
    weights = [p.detach().clone() for p in own]
    perturb_adapter(model, "default", seed=5)
    perturb_adapter(model, "b", seed=9)
    model.set_adapter(list(order))
    model.eval()
    with lucid.no_grad():
        out = recipe.call(model, lucid.float32, device)
        tolerance = _TOLERANCE * magnitude(out)

        model.base_model.merge_adapter()
        assert span(recipe.call(model, lucid.float32, device), out) <= tolerance
        model.base_model.unmerge_adapter()
        assert span(recipe.call(model, lucid.float32, device), out) <= tolerance
        # What a weight loses folding an update in and out is rounding at the size of the update,
        # which the output's size stands for (RandLoRA's is in the thousands).
        restored = _TOLERANCE * max(1.0, magnitude(out))
        assert all(span(p, w) <= restored for p, w in zip(own, weights, strict=True))

        merged = model.merge_and_unload()
        assert span(recipe.call(merged, lucid.float32, device), out) <= tolerance


class TestOrthogonalComposition:
    """The order of rotations: the weight is ``W D_1 D_2``, ``D_i`` being the adapter's matrix."""

    @staticmethod
    def _two_oft(device: str) -> tuple[PeftModel, lucid.Tensor]:
        def config() -> OFTConfig:
            return OFTConfig(
                oft_block_size=4, target_modules=["0"], use_cayley_neumann=False, init_weights=False
            )

        lucid.manual_seed(0)
        model = get_peft_model(nn.Sequential(nn.Linear(8, 8)).to(device), config())
        model.add_adapter("b", config())
        model.set_adapter(["default", "b"])
        return model, lucid.randn(5, 8, device=device)

    def test_two_oft_adapters_merge_to_what_the_forward_computes(self, device: str) -> None:
        model, x = self._two_oft(device)
        with lucid.no_grad():
            adapted_out = model(x)
            model.base_model.merge_adapter()
            assert span(model(x), adapted_out) <= _TOLERANCE * magnitude(adapted_out)

    def test_the_weight_is_rewritten_by_the_adapters_in_the_order_listed(self, device: str) -> None:
        model, x = self._two_oft(device)
        layer = model.base_model.model[0]
        weight, bias = layer.get_base_layer().weight, layer.get_base_layer().bias
        first, second = layer.get_delta_weight("default").mT, layer.get_delta_weight("b").mT
        with lucid.no_grad():
            expected = F.linear(x, weight @ first @ second, bias)
            reverse = F.linear(x, weight @ second @ first, bias)
            assert span(model(x), expected) <= _TOLERANCE * magnitude(expected)
            # The guard is not vacuous: the other order is a different function.
            assert span(expected, reverse) > 1e-3 * magnitude(expected)
            model.set_adapter(["b", "default"])
            assert span(model(x), reverse) <= _TOLERANCE * magnitude(reverse)


_CONV_CONFIGS: dict[str, Callable[[], PeftConfig]] = {
    "oft": lambda: OFTConfig(oft_block_size=4, use_cayley_neumann=False, target_modules=["0", "2"]),
    "boft": lambda: BOFTConfig(boft_block_size=4, target_modules=["0", "2"]),
    "hra": lambda: HRAConfig(r=2, target_modules=["0", "2"]),
}


@pytest.mark.parametrize("method", list(_CONV_CONFIGS))
@pytest.mark.parametrize("order", _ORDERS, ids="-".join)
def test_two_adapters_of_a_convolution_merge_to_the_adapted_forward(
    method: str, order: tuple[str, str], device: str
) -> None:
    """The weight-rewriting methods on ``nn.Conv2d``: the same contract as on ``nn.Linear``."""
    lucid.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(4, 8, 3), nn.ReLU(), nn.Conv2d(8, 8, 1)).to(device)
    model = get_peft_model(net, _CONV_CONFIGS[method]())
    model.add_adapter("b", _CONV_CONFIGS[method]())
    perturb_adapter(model, "default", seed=5)
    perturb_adapter(model, "b", seed=9)
    model.set_adapter(list(order))
    x = lucid.randn(2, 4, 6, 6, device=device)
    with lucid.no_grad():
        out = model(x)
        model.base_model.merge_adapter()
        assert span(model(x), out) <= _TOLERANCE * magnitude(out)
        model.base_model.unmerge_adapter()
        assert span(model(x), out) <= _TOLERANCE * magnitude(out)
        assert span(model.merge_and_unload()(x), out) <= _TOLERANCE * magnitude(out)


# ── merging some of the active adapters (LCD-971) ───────────────────────────────────────────────

#: The methods whose adapters apply one after another, each turning what the ones before it made
#: (a rotation of the weight): of these, only the first of the active adapters, in their order, can
#: be folded while the rest go on applying.  RoAd's ``road_1`` and MiSS's shard variants commute and
#: take any split; their other variants are held below.
_IN_ORDER = {PeftType.OFT, PeftType.BOFT, PeftType.HRA}

_MERGEABLE = [t for t in _TUNERS if t not in _REFUSED]
_FIRST = pytest.mark.parametrize("first", [0, 1], ids=["first-listed", "second-listed"])


def _layers(model: PeftModel) -> list[_BaseTunerLayer]:
    return [layer for _, layer in _tuner_layers(model)]


def _unmerged(model: PeftModel) -> bool:
    return not any(layer.merged or layer.merged_adapters for layer in _layers(model))


def _prepared(
    peft_type: PeftType, order: tuple[str, str], device: str, scratch: Path
) -> tuple[Recipe, list[lucid.Tensor], PeftModel]:
    """``_two`` with both adapters moved off their start, in use in ``order``, in eval mode."""
    recipe, own, model = _two(peft_type, device, scratch)
    perturb_adapter(model, "default", seed=5)
    perturb_adapter(model, "b", seed=9)
    model.set_adapter(list(order))
    model.eval()
    return recipe, own, model


@pytest.mark.parametrize("peft_type", _MERGEABLE, ids=lambda t: t.name)
def test_the_methods_that_apply_in_order_are_the_ones_that_declare_it(
    peft_type: PeftType, device: str, tmp_path: Path
) -> None:
    _, _, model = _two(peft_type, device, tmp_path)
    assert {type(layer)._ordered for layer in _layers(model)} == {peft_type in _IN_ORDER}


@pytest.mark.parametrize("peft_type", _MERGEABLE, ids=lambda t: t.name)
@pytest.mark.parametrize("order", _ORDERS, ids="-".join)
@_FIRST
@pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
def test_merging_some_of_the_active_adapters_does_not_change_the_function(
    peft_type: PeftType, order: tuple[str, str], first: int, device: str, tmp_path: Path
) -> None:
    """The layer keeps applying what it did not fold: forward is the unmerged forward, in every split.

    A method that applies in order refuses the split that folds an adapter ahead of one listed
    before it, before it writes anything; every other method takes any split.
    """
    recipe, own, model = _prepared(peft_type, order, device, tmp_path)
    weights = [p.detach().clone() for p in own]
    folded, rest = order[first], order[1 - first]
    with lucid.no_grad():
        out = recipe.call(model, lucid.float32, device)
        tolerance = _TOLERANCE * magnitude(out)
        model.set_adapter([folded])
        without_rest = recipe.call(model, lucid.float32, device)
        model.set_adapter(list(order))
        # Not vacuous: dropping the adapter that stays unmerged is a far bigger change.
        assert span(without_rest, out) > 1e3 * tolerance

        if peft_type in _IN_ORDER and first == 1:
            with pytest.raises(ValueError, match=rf"cannot merge \['{folded}'\] while .* unmerged"):
                model.base_model.merge_adapter(adapter_names=[folded])
            assert _unmerged(model)
            assert all(same(p, w) for p, w in zip(own, weights, strict=True))
            assert same(recipe.call(model, lucid.float32, device), out)
            return

        model.base_model.merge_adapter(adapter_names=[folded])
        assert all(
            layer.merged_adapters == [folded]
            for layer in _layers(model)
            if folded in layer._adapter_names()
        )
        assert span(recipe.call(model, lucid.float32, device), out) <= tolerance
        model.base_model.unmerge_adapter()
        assert _unmerged(model)
        restored = _TOLERANCE * max(1.0, magnitude(out))
        assert all(span(p, w) <= restored for p, w in zip(own, weights, strict=True))
        assert span(recipe.call(model, lucid.float32, device), out) <= tolerance

        # One and then the other, in two calls: the function holds after each.
        model.base_model.merge_adapter(adapter_names=[folded])
        model.base_model.merge_adapter(adapter_names=[rest])
        assert span(recipe.call(model, lucid.float32, device), out) <= tolerance
        model.base_model.unmerge_adapter()
        assert all(span(p, w) <= restored for p, w in zip(own, weights, strict=True))

        # Fold one, then unload: the rest is folded by the unload, and the model is the adapted one.
        model.base_model.merge_adapter(adapter_names=[folded])
        plain = model.merge_and_unload()
        assert span(recipe.call(plain, lucid.float32, device), out) <= tolerance


@pytest.mark.parametrize("peft_type", _MERGEABLE, ids=lambda t: t.name)
@pytest.mark.parametrize("order", _ORDERS, ids="-".join)
@pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
def test_merging_every_active_adapter_is_taken_in_any_order_and_nothing_stays_to_apply(
    peft_type: PeftType, order: tuple[str, str], device: str, tmp_path: Path
) -> None:
    """Naming all of the active adapters folds them in the order named, as before LCD-971."""
    recipe, _, model = _prepared(peft_type, order, device, tmp_path)
    names = list(order[::-1])
    with lucid.no_grad():
        model.base_model.merge_adapter(adapter_names=names)
        assert all(layer.merged_adapters == names for layer in _layers(model))
        assert all(not layer._applied(layer._adapter_names()) for layer in _layers(model))
        folded = recipe.call(model, lucid.float32, device)
        with model.disable_adapter():  # unmerges first: the base alone
            base = recipe.call(model, lucid.float32, device)
        assert span(folded, base) > 0  # the folded weights are not the base's


@pytest.mark.parametrize("peft_type", _MERGEABLE, ids=lambda t: t.name)
@pytest.mark.parametrize("order", _ORDERS, ids="-".join)
@_FIRST
def test_merge_and_unload_may_name_any_part_of_the_active_adapters(
    peft_type: PeftType, order: tuple[str, str], first: int, device: str, tmp_path: Path
) -> None:
    """The layers are dropped, so the weights alone are what is asked for: the refusal does not reach it."""
    recipe, _, model = _prepared(peft_type, order, device, tmp_path)
    name = order[first]
    with lucid.no_grad():
        model.set_adapter([name])
        alone = recipe.call(model, lucid.float32, device)
        model.set_adapter(list(order))
        plain = model.merge_and_unload(adapter_names=[name])
        assert not any(isinstance(m, _BaseTunerLayer) for m in plain.modules())
        out = recipe.call(plain, lucid.float32, device)
        assert span(out, alone) <= _TOLERANCE * magnitude(alone)


@pytest.mark.parametrize("dora", [(True, False), (False, True), (True, True)], ids=str)
@pytest.mark.parametrize("order", _ORDERS, ids="-".join)
@pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
def test_a_dora_adapter_that_stays_unmerged_blocks_the_merge_of_the_others(
    dora: tuple[bool, bool], order: tuple[str, str], device: str
) -> None:
    """DoRA divides by the norm of the weight it reads and takes its update from the base output.

    Folded adapters change both, so a DoRA adapter cannot go on applying after them; a plain one
    reads neither, so it can.
    """
    is_dora = dict(zip(("default", "b"), dora, strict=True))

    def config(use_dora: bool) -> LoraConfig:
        return LoraConfig(r=2, lora_alpha=4, target_modules=["0", "2"], use_dora=use_dora)

    lucid.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)
    own = list(net.parameters())
    model = get_peft_model(net, config(is_dora["default"]))
    model.add_adapter("b", config(is_dora["b"]))
    perturb_adapter(model, "default", seed=5)
    perturb_adapter(model, "b", seed=9)
    model.set_adapter(list(order))
    model.eval()
    weights = [p.detach().clone() for p in own]
    x = lucid.randn(3, 6, device=device)
    with lucid.no_grad():
        out = model(x)
        tolerance = _TOLERANCE * magnitude(out)
        for folded, rest in (order, order[::-1]):
            if is_dora[rest]:
                before = [p.detach().clone() for p in own]  # a round trip earlier rounded them
                before_out = model(x)
                with pytest.raises(ValueError, match=rf"DoRA adapters \['{rest}'\]"):
                    model.base_model.merge_adapter(adapter_names=[folded])
                assert _unmerged(model)
                assert all(same(p, w) for p, w in zip(own, before, strict=True))
                assert same(model(x), before_out)
                continue
            model.base_model.merge_adapter(adapter_names=[folded])
            assert span(model(x), out) <= tolerance
            model.base_model.unmerge_adapter()
            assert span(model(x), out) <= tolerance
            assert all(span(p, w) <= tolerance for p, w in zip(own, weights, strict=True))
        # Naming every adapter folds them, whatever they are.
        model.base_model.merge_adapter(adapter_names=list(order))
        assert span(model(x), out) <= tolerance


def _miss(init: bool | Literal["mini", "bat"]) -> PeftConfig:
    return MissConfig(r=2, init_weights=init, target_modules=["0", "2"])


def _bone(init: bool | Literal["bat"]) -> PeftConfig:
    return BoneConfig(r=2, init_weights=init, target_modules=["0", "2"])


def _road(variant: str, group_size: int = 2) -> PeftConfig:
    return RoadConfig(variant=variant, group_size=group_size, target_modules=["0", "2"])


#: Pairs of a variant that may not commute with its partner: MiSS and Bone's shard variants add to
#: the output and commute, ``bat`` rewrites the weight it is applied to; RoAd's ``road_1`` turns each
#: pair of features by one scaled rotation, which commutes, and the other two fill the whole 2x2 block.
#: ``(first, second, applies in order)``.
_PAIRS: dict[str, tuple[Callable[[], PeftConfig], Callable[[], PeftConfig], bool]] = {
    "road_1 + road_1": (lambda: _road("road_1"), lambda: _road("road_1"), False),
    "road_1 + road_1 in fours": (lambda: _road("road_1", 4), lambda: _road("road_1", 4), False),
    "road_2 + road_2": (lambda: _road("road_2"), lambda: _road("road_2"), True),
    "road_4 + road_4": (lambda: _road("road_4"), lambda: _road("road_4"), True),
    "road_1 + road_2": (lambda: _road("road_1"), lambda: _road("road_2"), True),
    "road_2 + road_4 in fours": (lambda: _road("road_2", 4), lambda: _road("road_4", 4), True),
    "miss balance + mini": (lambda: _miss(True), lambda: _miss("mini"), False),
    "miss balance + balance": (lambda: _miss(True), lambda: _miss(True), False),
    "miss bat + bat": (lambda: _miss("bat"), lambda: _miss("bat"), True),
    "miss balance + bat": (lambda: _miss(True), lambda: _miss("bat"), True),
    "miss bat + mini": (lambda: _miss("bat"), lambda: _miss("mini"), True),
    "bone balance + balance": (lambda: _bone(True), lambda: _bone(True), False),
    "bone bat + bat": (lambda: _bone("bat"), lambda: _bone("bat"), True),
    "bone balance + bat": (lambda: _bone(True), lambda: _bone("bat"), True),
}


@pytest.mark.parametrize("pair", list(_PAIRS))
@pytest.mark.parametrize("order", _ORDERS, ids="-".join)
@_FIRST
def test_a_variant_that_may_not_commute_applies_in_order_and_one_that_does_not(
    pair: str, order: tuple[str, str], first: int, device: str
) -> None:
    first_config, second_config, in_order = _PAIRS[pair]
    lucid.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)
    own = list(net.parameters())
    model = get_peft_model(net, first_config())
    model.add_adapter("b", second_config())
    perturb_adapter(model, "default", seed=5)
    perturb_adapter(model, "b", seed=9)
    model.set_adapter(list(order))
    model.eval()
    weights = [p.detach().clone() for p in own]
    x = lucid.randn(3, 6, device=device)
    folded = order[first]
    with lucid.no_grad():
        out = model(x)
        tolerance = _TOLERANCE * magnitude(out)
        if in_order and first == 1:
            with pytest.raises(ValueError, match=r"apply one after another"):
                model.base_model.merge_adapter(adapter_names=[folded])
            assert _unmerged(model)
            assert all(same(p, w) for p, w in zip(own, weights, strict=True))
            return
        model.base_model.merge_adapter(adapter_names=[folded])
        assert span(model(x), out) <= tolerance
        model.base_model.unmerge_adapter()
        assert all(span(p, w) <= tolerance for p, w in zip(own, weights, strict=True))


@pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
def test_an_adapter_of_ln_tuning_runs_whether_or_not_another_is_merged(
    device: str, tmp_path: Path
) -> None:
    """LN tuning applies one adapter's copy: another adapter's merge exchanges the base's values, not the copy's."""
    recipe = RECIPES[PeftType.LN_TUNING]
    own, model = adapted(recipe, lucid.float32, device, tmp_path)
    model.add_adapter("b", LNTuningConfig(target_modules=["0", "2"]))
    perturb_adapter(model, "default", seed=5)
    perturb_adapter(model, "b", seed=9)
    model.set_adapter(["default"])
    model.eval()
    weights = [p.detach().clone() for p in own]
    with lucid.no_grad():
        out = recipe.call(model, lucid.float32, device)
        model.base_model.merge_adapter(adapter_names=["b"])
        assert all(layer.merged_adapters == ["b"] for layer in _layers(model))
        assert same(recipe.call(model, lucid.float32, device), out)
        model.base_model.unmerge_adapter()
        assert same(recipe.call(model, lucid.float32, device), out)
        assert all(same(p, w) for p, w in zip(own, weights, strict=True))


class _Attention(nn.Module):
    """A module that reads its attention weights inside its own forward: the LoRA weight-space layer."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        out, _ = self.attn(x, x, x)
        return out


def _convs() -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Conv2d(4, 4, 1))


type _Kind = tuple[Callable[[], nn.Module], Callable[[], lucid.Tensor], Callable[[], PeftConfig]]

#: ``(base, input, config)`` for the layer kinds the MLP recipes do not reach.
_KINDS: dict[str, _Kind] = {
    "lora conv2d": (
        _convs,
        lambda: lucid.randn(2, 3, 6, 6),
        lambda: LoraConfig(r=2, target_modules=["0", "2"]),
    ),
    "lora conv-transpose": (
        lambda: nn.Sequential(nn.ConvTranspose2d(3, 4, 3), nn.ReLU()),
        lambda: lucid.randn(2, 3, 6, 6),
        lambda: LoraConfig(r=2, target_modules=["0"]),
    ),
    "lora embedding": (
        lambda: nn.Sequential(nn.Embedding(10, 6), nn.Linear(6, 3)),
        lambda: lucid.tensor([[1, 2, 3], [4, 5, 6]]),
        lambda: LoraConfig(r=2, target_modules=["0", "1"]),
    ),
    "lora attention": (
        _Attention,
        lambda: lucid.randn(2, 5, 8),
        lambda: LoraConfig(r=2, target_modules=["attn"]),
    ),
    "loha conv2d": (
        _convs,
        lambda: lucid.randn(2, 3, 6, 6),
        lambda: LoHaConfig(r=2, target_modules=["0", "2"]),
    ),
}


@pytest.mark.parametrize("kind", list(_KINDS))
@pytest.mark.parametrize("order", _ORDERS, ids="-".join)
@_FIRST
def test_a_layer_of_every_kind_goes_on_applying_the_adapters_it_did_not_fold(
    kind: str, order: tuple[str, str], first: int, device: str
) -> None:
    make_base, make_input, make_config = _KINDS[kind]
    lucid.manual_seed(0)
    model = get_peft_model(make_base().to(device), make_config())
    model.add_adapter("b", make_config())
    perturb_adapter(model, "default", seed=5)
    perturb_adapter(model, "b", seed=9)
    model.set_adapter(list(order))
    model.eval()
    x = make_input().to(device)
    with lucid.no_grad():
        out = model(x)
        tolerance = _TOLERANCE * magnitude(out)
        model.base_model.merge_adapter(adapter_names=[order[first]])
        assert span(model(x), out) <= tolerance
        model.base_model.unmerge_adapter()
        assert span(model(x), out) <= tolerance
