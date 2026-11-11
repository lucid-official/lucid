"""The placement contract of every registered tuner: an adapter lives where the weight it adapts does.

An adapter's tensors are allocated beside the base weight (``_Home`` and the layers' own homes), not
where the process's default device happens to be.  A tensor made without a device follows the
default, so a Metal default with a CPU base (a model built on the CPU, adapted in a session that
trains on Metal) or the other way round mixes two devices in one expression and the engine refuses
it (LCD-963: FRoD's basis built its Gram sum on the default device from CPU pieces).  The class-wide
shadow-allocation sweep (``nn/test_shadow_alloc_surface``) builds each class under each default
device but in a phantom world where nothing is placed by weight; this file builds real adapters on a
real base and holds every method in ``_TUNERS`` to the rule, so a method added to the registry is
held to it too: one with no recipe fails loudly.
"""

from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import PeftModel, PeftType
from lucid.peft.tuners._base import _TUNERS
from lucid.test._default_device import default_device, metal_available
from lucid.test.unit.peft._tuner_recipes import RECIPES, Recipe, adapted

#: ``(default device, base device)``: the pairs where the two differ, and Metal under both.
_PAIRS = [("metal", "cpu"), ("cpu", "metal"), ("metal", "metal")]

pytestmark = pytest.mark.skipif(not metal_available(), reason="compares the CPU with Metal")


@pytest.fixture(params=list(_TUNERS), ids=lambda t: t.name)
def recipe(request: pytest.FixtureRequest) -> Recipe:
    peft_type: PeftType = request.param
    if peft_type not in RECIPES:
        pytest.fail(
            f"{peft_type.name} is registered but has no recipe: give it a row in RECIPES "
            "(lucid/test/unit/peft/_tuner_recipes.py), or in the audit's _EXAMPLE_CONFIGS if it "
            "adapts the two linear layers, so its adapter placement is held to the contract"
        )
    return RECIPES[peft_type]


def _strays(model: nn.Module, where: str) -> list[str]:
    """Every parameter and buffer of ``model`` that is not on ``where``, with the device it is on."""
    held = [*model.named_parameters(), *model.named_buffers()]
    return sorted(f"{name}: {t.device.type}" for name, t in held if t.device.type != where)


@pytest.mark.parametrize(("default", "where"), _PAIRS, ids=lambda p: str(p))
class TestPlacement:
    def test_every_tensor_of_the_adapter_is_on_the_base_weights_device(
        self, recipe: Recipe, default: str, where: str, tmp_path: Path
    ) -> None:
        with default_device(default):
            _, model = adapted(recipe, lucid.float32, where, tmp_path)
            assert _strays(model, where) == []

    def test_the_adapted_model_runs_forward_and_backward_there(
        self, recipe: Recipe, default: str, where: str, tmp_path: Path
    ) -> None:
        with default_device(default):
            _, model = adapted(recipe, lucid.float32, where, tmp_path)
            out = recipe.call(model, lucid.float32, where)
            assert out.device.type == where
            out.float().sum().backward()

    def test_an_adapter_added_later_is_placed_the_same_way(
        self, recipe: Recipe, default: str, where: str, tmp_path: Path
    ) -> None:
        if not recipe.adds:
            pytest.skip("this method holds one adapter")
        with default_device(default):
            _, model = adapted(recipe, lucid.float32, where, tmp_path)
            assert isinstance(model, PeftModel)
            model.add_adapter("b", recipe.second(lucid.float32, where, tmp_path))
            assert _strays(model, where) == []
