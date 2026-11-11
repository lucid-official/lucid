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
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.peft import BOFTConfig, HRAConfig, OFTConfig, PeftConfig, PeftModel, PeftType
from lucid.peft import get_peft_model
from lucid.peft.tuners._base import _TUNERS
from lucid.test.unit.peft._tuner_recipes import (
    RECIPES,
    Recipe,
    adapted,
    magnitude,
    perturb_adapter,
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
