"""A conversion that changes nothing leaves every tensor where it was (LCD-601, LCD-546).

``Module._apply`` put each parameter's gradient back through ``.grad =`` and
told the compiled wrappers that the model's tensors were replaced.  It did
both even when ``fn`` handed every tensor back unchanged: ``model.to()`` to
the device the model is on, or a cast to the dtype it has.  So a held
``p.grad`` stopped being the gradient, and on the CPU an in-place write to
``p.grad`` was refused while the holder lived.  A compiled module also
traced again on every call that followed such a ``.to()``.  Now a slot keeps
its tensor when ``fn`` returns the same engine tensor, and the wrappers hear
only of real replacements.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid.compile._entry.module import CompiledModule
from lucid.test._fixtures.devices import metal_available

type Change = Callable[[nn.Module, str], object]

# Conversions that hand every tensor back as it is.
_NO_OPS: dict[str, Change] = {
    "to(same device)": lambda m, device: m.to(device),
    "to(float32)": lambda m, device: m.to(lucid.float32),
    "float()": lambda m, device: m.float(),
    "type(float32)": lambda m, device: m.type(lucid.float32),
}


def _model(device: str) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3)).to(device)


@pytest.mark.parametrize("change", _NO_OPS.values(), ids=_NO_OPS.keys())
def test_a_no_op_conversion_keeps_parameters_gradients_and_buffers(
    change: Change, device: str
) -> None:
    m = _model(device)
    m(lucid.randn(2, 4, device=device)).sum().backward()
    weight = m[0].weight
    impl = weight._impl
    g = weight.grad
    mean = m[1].running_mean
    assert g is not None and mean is not None
    change(m, device)
    assert m[0].weight is weight and weight._impl is impl
    assert weight.grad is not None and weight.grad._impl is g._impl
    assert m[1].running_mean is mean


def test_the_gradient_takes_an_in_place_write_after_a_no_op_move() -> None:
    m = _model("cpu")
    m(lucid.randn(2, 4)).sum().backward()
    weight = m[0].weight
    g = weight.grad
    assert g is not None
    before = g.tolist()
    m.to("cpu")
    with lucid.no_grad():
        assert weight.grad is not None
        weight.grad.mul_(2.0)
    assert g.tolist() == [[2.0 * v for v in row] for row in before]


def _compiles(change: Change, monkeypatch: pytest.MonkeyPatch) -> int:
    """How many times a compiled model traces over four calls, ``change`` before each."""
    m = _model("metal").eval()
    compiled = lucid.compile(m)
    assert isinstance(compiled, CompiledModule)
    count = 0
    trace = compiled._compile_for

    def spy(args: tuple[object, ...], kwargs: dict[str, object]) -> object:
        nonlocal count
        count += 1
        return trace(args, kwargs)

    monkeypatch.setattr(compiled, "_compile_for", spy)
    x = lucid.randn(2, 4, device="metal")
    with lucid.no_grad():
        for _ in range(4):
            change(m, "metal")
            compiled(x)
    return count


def _round_trip(m: nn.Module, device: str) -> None:
    del device  # the round trip changes the dtype, not the device
    m.half()
    m.float()


@pytest.mark.skipif(not metal_available(), reason="compile traces on Metal")
@pytest.mark.parametrize("change", _NO_OPS.values(), ids=_NO_OPS.keys())
def test_a_no_op_conversion_does_not_retrace_a_compiled_model(
    change: Change, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _compiles(change, monkeypatch) == 1


@pytest.mark.skipif(not metal_available(), reason="compile traces on Metal")
def test_a_real_conversion_still_retraces(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _compiles(_round_trip, monkeypatch) == 4
