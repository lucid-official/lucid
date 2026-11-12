"""A compiled adapted model follows an in-place change of any parameter or buffer (LCD-986).

A compiled executable does not key on tensor values: whatever Python derives from a
parameter's values outside the graph (a prototype, a factorisation, a norm kept on the
module) is replayed as it was at the trace.  ``Arrow`` is the one method that does, and it
refuses the trace itself; a sweep of the rest found none.  This holds every registered
tuner to that: after a compiled call, every float parameter and buffer of the adapted
model is moved in place (as an optimizer step, a merge or a load would), and the compiled
model must answer as an eager one does, or have been run eagerly.

One small model per method (``_tuner_recipes.RECIPES``, which holds a row for each
registered method), on Metal, where executables run.  A method added to the registry with
no recipe fails loudly, as in ``test_adapter_dtype_contract``.
"""

import warnings
from pathlib import Path

import pytest

import lucid
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.peft import PeftModel, PeftType
from lucid.peft.tuners._base import _TUNERS
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.peft._tuner_recipes import (
    RECIPES,
    Recipe,
    adapted,
    magnitude,
    perturb,
    span,
)

pytestmark = [
    pytest.mark.skipif(not metal_available(), reason="compiled executables run on Metal"),
    pytest.mark.filterwarnings("ignore:lucid.compile"),
]

DEVICE = "metal"
#: Relative to the largest output: a perturbed method's output may be large (RandLoRA's reaches 3e4).
TOLERANCE = 1e-4
#: How far the second perturbation must move the eager answer for the comparison to mean anything.
MOVED = 1e-3


@pytest.fixture(params=list(_TUNERS), ids=lambda t: t.name)
def peft_type(request: pytest.FixtureRequest) -> PeftType:
    chosen: PeftType = request.param
    if chosen not in RECIPES:
        pytest.fail(
            f"{chosen.name} is registered but has no recipe: give it a row in RECIPES "
            "(lucid/test/unit/peft/_tuner_recipes.py), or in the audit's _EXAMPLE_CONFIGS if it "
            "adapts the two linear layers, so a compiled call is held to follow its parameters"
        )
    return chosen


def _allowed(reference: lucid.Tensor) -> float:
    return TOLERANCE * max(1.0, magnitude(reference))


def _move_everything(model: PeftModel, seed: int) -> None:
    """Move every float parameter and buffer of ``model`` in place, trainable or not."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for tensor in [*model.parameters(), *model.buffers()]:
            if tensor.is_floating_point():
                tensor.add_(0.2 * lucid.randn(*tensor.shape, device=tensor.device))


def _build(recipe: Recipe, scratch: Path) -> PeftModel:
    _, model = adapted(recipe, lucid.float32, DEVICE, scratch)
    perturb(model)
    return model.eval()


def test_a_compiled_call_follows_an_in_place_change_of_every_tensor(
    peft_type: PeftType, tmp_path: Path
) -> None:
    recipe = RECIPES[peft_type]
    model = _build(recipe, tmp_path)
    with lucid.no_grad():
        compiled = lucid.compile(model)
        first = recipe.call(compiled, lucid.float32, DEVICE)
        assert span(first, recipe.call(model, lucid.float32, DEVICE)) <= _allowed(first)

        before = recipe.call(model, lucid.float32, DEVICE)
        _move_everything(model, seed=7)
        # The compiled call comes first: no eager pass has had the chance to refresh what
        # the method derives from its parameters on the compiled one's behalf.
        got = recipe.call(compiled, lucid.float32, DEVICE)
        want = recipe.call(model, lucid.float32, DEVICE)

    assert span(want, before) > MOVED, "the change did not reach the output: the test is blind"
    assert span(got, want) <= _allowed(want), (
        f"{peft_type.name}: the compiled call answers from the tensors of its trace "
        f"(off eager by {span(got, want):.3g}); refuse the trace "
        "(lucid._engine_scope.refuse_trace) or compute it inside the graph"
    )


def test_the_guard_sees_a_compiled_call_where_one_can_run(tmp_path: Path) -> None:
    """LoRA's two linear layers compile, so the comparison above is not made on eager alone."""
    recipe = RECIPES[PeftType.LORA]
    model = _build(recipe, tmp_path)
    with lucid.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        compiled = lucid.compile(model)
        recipe.call(compiled, lucid.float32, DEVICE)
        _move_everything(model, seed=7)
        recipe.call(compiled, lucid.float32, DEVICE)
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
