"""A compiled adapted model follows every adapter switch, on Metal (LCD-703).

The compiled cache keys an executable on each submodule's ``training``
flag and *public scalar* attributes.  The LoRA layer's forward branches on
whether the adapter is disabled, merged and which adapter is active, so
those must be public scalar attributes — kept that way on every path that
changes them — or a compiled model replays the executable of a state it
has left: disabled, it still added the update; merged, it added it twice.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid.compile._core import signature
from lucid.peft import LoraConfig, PeftModel, get_peft_model
from lucid.peft.tuners.lora import Linear
from lucid.test._fixtures.devices import metal_available

metal_only = pytest.mark.skipif(not metal_available(), reason="compiled executables run on Metal")


def _peft(
    modules_to_save: list[str] | None = None, device: str = "metal"
) -> tuple[PeftModel, nn.Sequential]:
    lucid.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 3)).to(device)
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["0"],
        init_lora_weights=False,
        modules_to_save=modules_to_save,
    )
    peft = get_peft_model(model, config)
    return peft, model


def _close(a: object, b: object) -> bool:
    assert isinstance(a, lucid.Tensor) and isinstance(b, lucid.Tensor)
    return bool(lucid.allclose(a, b, atol=1e-5))


@metal_only
def test_compiled_follows_disable_enable_merge_unmerge() -> None:
    peft, _ = _peft()
    compiled = lucid.compile(peft)
    x = lucid.randn(4, 8).to("metal")
    with lucid.no_grad():
        adapted = peft(x)
        assert _close(compiled(x), adapted)
        with peft.disable_adapter():
            base = peft(x)
            assert not _close(base, adapted)
            assert _close(compiled(x), base)
        assert _close(compiled(x), adapted)
        peft.base_model.merge_adapter()
        assert _close(peft(x), adapted)
        assert _close(compiled(x), adapted)
        peft.base_model.unmerge_adapter()
        assert _close(compiled(x), adapted)
        peft.base_model.merge_adapter(safe_merge=True)
        assert _close(compiled(x), adapted)
        with peft.disable_adapter():
            assert _close(compiled(x), base)
        assert _close(compiled(x), adapted)


@metal_only
def test_compiled_follows_the_modules_to_save_switch() -> None:
    peft, model = _peft(modules_to_save=["2"])
    with lucid.no_grad():
        model[2].modules_to_save["default"].weight.add_(1.0)
    compiled = lucid.compile(peft)
    x = lucid.randn(4, 8).to("metal")
    with lucid.no_grad():
        adapted = peft(x)
        assert _close(compiled(x), adapted)
        with peft.disable_adapter():
            base = peft(x)
            assert _close(compiled(x), base)
        assert _close(compiled(x), adapted)


def _branch_state(model: nn.Sequential) -> tuple[object, ...]:
    """What the LoRA layer and the wrapper branch on, read off their ``__dict__``."""
    layer, wrapper = vars(model[0]), vars(model[2])
    return (
        layer["active_adapter"],
        layer["disable_adapters"],
        layer["merged"],
        wrapper["active_adapter"],
        wrapper["disable_adapters"],
    )


def _disable(peft: PeftModel) -> None:
    peft.base_model.disable_adapter_layers()


def _enable(peft: PeftModel) -> None:
    peft.base_model.enable_adapter_layers()


def _merge(peft: PeftModel) -> None:
    peft.base_model.merge_adapter()


def _safe_merge(peft: PeftModel) -> None:
    peft.base_model.merge_adapter(safe_merge=True)


def _unmerge(peft: PeftModel) -> None:
    peft.base_model.unmerge_adapter()


#: Every path that moves the branch state, with the state it leaves:
#: (active adapter, layer disabled, layer merged, copy's adapter, copy disabled).
STEPS: list[tuple[str, Callable[[PeftModel], None], tuple[object, ...]]] = [
    ("disable", _disable, ("default", True, False, "default", True)),
    ("enable", _enable, ("default", False, False, "default", False)),
    ("merge", _merge, ("default", False, True, "default", False)),
    ("unmerge", _unmerge, ("default", False, False, "default", False)),
    ("safe merge", _safe_merge, ("default", False, True, "default", False)),
    ("disable while merged", _disable, ("default", True, False, "default", True)),
    ("enable again", _enable, ("default", False, False, "default", False)),
]


def test_every_switch_lands_in_the_compile_key() -> None:
    peft, model = _peft(modules_to_save=["2"], device="cpu")
    assert _branch_state(model) == ("default", False, False, "default", False)
    seen = {signature._module_state(peft)}
    for name, step, expected in STEPS:
        before = signature._module_state(peft)
        step(peft)
        assert _branch_state(model) == expected, name
        after = signature._module_state(peft)
        assert (after != before) == (expected != _state_before(name)), name
        seen.add(after)
    # Three branch states — live, disabled, merged (never disabled and
    # merged at once) — and three keys.
    assert len(seen) == 3


def _state_before(name: str) -> tuple[object, ...]:
    index = [step[0] for step in STEPS].index(name)
    return STEPS[index - 1][2] if index else ("default", False, False, "default", False)


def test_the_disable_context_round_trips_the_state() -> None:
    peft, model = _peft(modules_to_save=["2"], device="cpu")
    _merge(peft)
    with peft.disable_adapter():
        assert _branch_state(model) == ("default", True, False, "default", True)
    assert _branch_state(model) == ("default", False, False, "default", False)


def test_merging_a_disabled_adapter_is_refused_before_any_write() -> None:
    peft, model = _peft(device="cpu")
    weight = model[0].base_layer.weight.clone()
    with peft.disable_adapter():
        with pytest.raises(RuntimeError, match="disabled"):
            peft.base_model.merge_adapter()
        with pytest.raises(RuntimeError, match="disabled"):
            peft.merge_and_unload()
    assert bool((model[0].base_layer.weight == weight).all().item())
    assert vars(model[0])["merged"] is False


def test_unload_and_merge_and_unload_leave_no_branch_state() -> None:
    peft, model = _peft(modules_to_save=["2"], device="cpu")
    layer = model[0]
    assert isinstance(layer, Linear)
    peft.merge_and_unload()
    assert type(model[0]) is nn.Linear and vars(layer)["merged"] is True
    again, model = _peft(modules_to_save=["2"], device="cpu")
    layer = model[0]
    assert isinstance(layer, Linear)
    _merge(again)
    again.unload()
    assert vars(layer)["merged"] is False and type(model[0]) is nn.Linear
