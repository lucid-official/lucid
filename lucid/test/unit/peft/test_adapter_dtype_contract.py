"""The dtype contract of every registered tuner: a half-precision base gets a float32 adapter.

``lucid.peft`` documents that an adapter beside a float16 or bfloat16 base trains in float32 and
computes there, and that the layer hands back the base's dtype.  An adapter left in half precision
cannot train: its update vanishes under the ulp of a weight near one (SGD) and Adam's second
moment overflows (NaN on Metal).  The rule has one owner, ``_BaseTuner._inject``, and this file
holds every method in ``_TUNERS`` to it — on a float16 and a bfloat16 base, on CPU and Metal —
so a method added to the registry is held to it too: one with no recipe fails loudly.

A float64 base keeps float64 adapters (CPU only: Metal holds no double precision), as the reference
does; only half precision is widened.  A prompt is the exception: the reference's is float32
whatever the model's dtype, and the layer casts it where it enters.
"""

from pathlib import Path

import pytest

import lucid
import lucid.optim as optim
from lucid.peft import PeftModel, PeftType
from lucid.peft.tuners._base import _TUNERS
from lucid.test.unit.peft._tuner_recipes import (
    RECIPES,
    Recipe,
    adapted,
    finite,
    magnitude,
    perturb,
    same,
    span,
    trainable,
)

_HALF = [lucid.float16, lucid.bfloat16]
_NAMES = {lucid.float16: "float16", lucid.bfloat16: "bfloat16"}

#: The relative spacing of each half dtype.  Merging rounds the base weights to it, so a merged
#: model may sit a few of these from the adapted one, relative to its largest output (the worst
#: method measures under two; three leaves room for the second layer and for Metal).
_EPSILON = {lucid.float16: 2.0**-10, lucid.bfloat16: 2.0**-7}
_MERGE_EPSILONS = 3


@pytest.fixture(params=list(_TUNERS), ids=lambda t: t.name)
def recipe(request: pytest.FixtureRequest) -> Recipe:
    peft_type: PeftType = request.param
    if peft_type not in RECIPES:
        pytest.fail(
            f"{peft_type.name} is registered but has no recipe: give it a row in RECIPES "
            "(lucid/test/unit/peft/_tuner_recipes.py), or in the audit's _EXAMPLE_CONFIGS if it "
            "adapts the two linear layers, so its adapter dtype is held to the contract"
        )
    return RECIPES[peft_type]


def test_every_registered_tuner_has_a_recipe() -> None:
    assert set(_TUNERS) <= set(RECIPES), sorted(t.name for t in set(_TUNERS) - set(RECIPES))


@pytest.mark.parametrize("dtype", _HALF, ids=_NAMES.get)
class TestHalfPrecisionBase:
    def test_the_adapter_trains_in_float32(
        self, recipe: Recipe, dtype: lucid.dtype, device: str, tmp_path: Path
    ) -> None:
        _, model = adapted(recipe, dtype, device, tmp_path)
        assert {p.dtype for p in trainable(model)} == {lucid.float32}

    def test_the_output_keeps_the_base_dtype(
        self, recipe: Recipe, dtype: lucid.dtype, device: str, tmp_path: Path
    ) -> None:
        _, model = adapted(recipe, dtype, device, tmp_path)
        perturb(model)
        out = recipe.call(model, dtype, device)
        assert out.dtype == dtype and finite(out)

    def test_an_adapter_added_later_trains_in_float32_too(
        self, recipe: Recipe, dtype: lucid.dtype, device: str, tmp_path: Path
    ) -> None:
        _, model = adapted(recipe, dtype, device, tmp_path)
        if not recipe.adds:
            with pytest.raises(NotImplementedError):
                model.add_adapter("b", recipe.second(dtype, device, tmp_path))
            return
        model.add_adapter("b", recipe.second(dtype, device, tmp_path))
        model.set_adapter("b")
        assert {p.dtype for p in trainable(model)} == {lucid.float32}

    @pytest.mark.parametrize("name", ["sgd", "adam"])
    def test_steps_move_the_adapter_and_keep_it_finite(
        self, recipe: Recipe, dtype: lucid.dtype, device: str, tmp_path: Path, name: str
    ) -> None:
        _, model = adapted(recipe, dtype, device, tmp_path)
        params = trainable(model)
        start = [p.detach().clone() for p in params]
        # A step below a float16 weight's spacing near one: a half-precision adapter would not move.
        optimizer = optim.SGD(params, lr=1e-5) if name == "sgd" else optim.Adam(params, lr=1e-5)
        for _ in range(5):
            # Linear in the output, so no method's scale can make the steps diverge.
            loss = recipe.call(model, dtype, device).to(lucid.float32).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        assert all(p.dtype == lucid.float32 for p in params)
        assert all(finite(p) for p in params)
        assert any(span(p, s) > 0 for p, s in zip(params, start, strict=True))

    def test_a_saved_adapter_loads_back_to_the_same_output(
        self, recipe: Recipe, dtype: lucid.dtype, device: str, tmp_path: Path
    ) -> None:
        _, model = adapted(recipe, dtype, device, tmp_path)
        perturb(model)
        model.eval()
        expected = recipe.call(model, dtype, device)
        model.save_pretrained(tmp_path / "adapter")
        fresh = recipe.base(dtype, device)
        own = {id(p) for p in fresh.parameters()}  # the adapter is built into ``fresh`` itself
        loaded = PeftModel.from_pretrained(fresh, str(tmp_path / "adapter"))
        loaded.eval()
        assert {p.dtype for p in loaded.parameters() if id(p) not in own} == {lucid.float32}
        assert same(recipe.call(loaded, dtype, device), expected)

    @pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
    def test_merging_into_the_half_base(
        self, recipe: Recipe, dtype: lucid.dtype, device: str, tmp_path: Path
    ) -> None:
        own, model = adapted(recipe, dtype, device, tmp_path)
        if not recipe.mergeable:
            with pytest.raises(NotImplementedError):
                model.merge_and_unload()
            return
        weights = [p.detach().clone() for p in own]
        perturb(model)
        model.eval()
        out = recipe.call(model, dtype, device)
        with model.disable_adapter():
            unadapted = recipe.call(model, dtype, device)
        tolerance = _MERGE_EPSILONS * _EPSILON[dtype] * magnitude(out)
        assert span(out, unadapted) > tolerance  # the adapter does something

        model.base_model.merge_adapter()
        assert span(recipe.call(model, dtype, device), out) <= tolerance
        model.base_model.unmerge_adapter()
        assert span(recipe.call(model, dtype, device), out) <= tolerance
        assert all(span(p, w) <= tolerance for p, w in zip(own, weights, strict=True))

        merged = model.merge_and_unload()
        assert {p.dtype for p in merged.parameters()} == {dtype}
        assert span(recipe.call(merged, dtype, device), out) <= tolerance


class TestFloat64Base:
    """A float64 base is not half precision: the adapter stays in its dtype (CPU: Metal has no float64)."""

    def test_the_adapter_follows_the_base(
        self, recipe: Recipe, device_cpu_only: str, tmp_path: Path
    ) -> None:
        dtype = lucid.float64
        _, model = adapted(recipe, dtype, device_cpu_only, tmp_path)
        assert {p.dtype for p in trainable(model)} == {recipe.on_float64}
        out = recipe.call(model, dtype, device_cpu_only)
        assert out.dtype == dtype
        out.square().mean().backward()
        assert all(
            p.grad is not None and p.grad.dtype == recipe.on_float64 for p in trainable(model)
        )
