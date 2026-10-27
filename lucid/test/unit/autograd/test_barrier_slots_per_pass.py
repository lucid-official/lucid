"""A barrier's gradients belong to the pass that delivered them (LCD-168).

A barrier — a hooked module's outputs or inputs, a custom Function with
several outputs, an LSTM — holds what reaches each of its slots until its
turn.  Two passes must not share those slots:

* a pass that delivered and never gave the barrier its turn — an exception,
  a pruned ``autograd.grad`` — left the multi-output Function's slots full,
  and the next ``backward`` added onto them;
* a pass opened inside another (a hook calling ``autograd.grad``) that
  crossed the same barrier wiped what the outer pass had delivered, and the
  outer pass resumed without it.

The reference keeps the input buffers of each pass apart.  Each scenario is
written once over a module ``m`` and run on Lucid and the reference.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

Scenario = Callable[[ModuleType, str], object]


def _values(t: object) -> object:
    if t is None:
        return None
    return [round(float(v), 5) for v in t.detach().cpu().reshape(-1).tolist()]


def _split(m: ModuleType) -> type:
    class Split(m.autograd.Function):
        @staticmethod
        def forward(ctx: object, a: object) -> object:
            return a * 2.0, a * 3.0

        @staticmethod
        def backward(ctx: object, g0: object, g1: object) -> object:
            return g0 * 2.0 + g1 * 3.0

    return Split


def _raising_hook(flag: dict[str, bool]) -> Callable[[object], object]:
    def hook(g: object) -> object:
        if flag["raise"]:
            raise ValueError("boom")
        return g

    return hook


def _aborted_backward_then_backward(m: ModuleType, device: str) -> object:
    a = m.ones(1, requires_grad=True, device=device)
    y0, y1 = _split(m).apply(a)
    flag = {"raise": True}
    # y1's hook runs when the Function is reached, after y0's gradient was
    # delivered to it.
    y1.register_hook(_raising_hook(flag))
    loss = (y0 * 2.0).sum() + (y1 * 1.0).sum()
    with pytest.raises(ValueError, match="boom"):
        loss.backward(retain_graph=True)
    flag["raise"] = False
    loss.backward()
    return _values(a.grad)


def _aborted_grad_then_grad(m: ModuleType, device: str) -> object:
    a = m.ones(1, requires_grad=True, device=device)
    y0, y1 = _split(m).apply(a)
    flag = {"raise": True}
    y1.register_hook(_raising_hook(flag))
    loss = (y0 * 2.0).sum() + (y1 * 1.0).sum()
    with pytest.raises(ValueError, match="boom"):
        m.autograd.grad(loss, a, retain_graph=True)
    flag["raise"] = False
    return [_values(g) for g in m.autograd.grad(loss, a)]


def _nested_through_module(m: ModuleType, device: str, hook_on_b: bool) -> object:
    class Square(m.nn.Module):
        def forward(self, x: object) -> object:
            return x * x

    mod = Square()
    seen: list[object] = []
    mod.register_full_backward_hook(lambda _m, gi, go: seen.append(_values(go[0])))
    x = m.tensor([1.0, 3.0], requires_grad=True, device=device)
    y = mod(x)
    a, b = y * 2.0, y * 5.0
    inner: list[object] = []

    def hook(g: object) -> object:
        with m.enable_grad():
            (gi,) = m.autograd.grad((y * 3.0).sum(), x, retain_graph=True)
        inner.append(_values(gi))
        return g

    (b if hook_on_b else a).register_hook(hook)
    (a.sum() + b.sum()).backward()
    return [_values(x.grad), inner, seen]


def _nested_through_function(m: ModuleType, device: str, hook_on_b: bool) -> object:
    x = m.tensor([1.0, 3.0], requires_grad=True, device=device)
    y0, y1 = _split(m).apply(x)
    a, b = y0 * 2.0, y0 * 5.0
    inner: list[object] = []

    def hook(g: object) -> object:
        with m.enable_grad():
            (gi,) = m.autograd.grad((y0 * 3.0 + y1).sum(), x, retain_graph=True)
        inner.append(_values(gi))
        return g

    (b if hook_on_b else a).register_hook(hook)
    (a.sum() + b.sum() + y1.sum()).backward()
    return [_values(x.grad), inner]


_SCENARIOS: dict[str, Scenario] = {
    "aborted-backward-then-backward": _aborted_backward_then_backward,
    "aborted-grad-then-grad": _aborted_grad_then_grad,
    "nested-through-module-first": lambda m, d: _nested_through_module(m, d, False),
    "nested-through-module-second": lambda m, d: _nested_through_module(m, d, True),
    "nested-through-function-first": lambda m, d: _nested_through_function(m, d, False),
    "nested-through-function-second": lambda m, d: _nested_through_function(m, d, True),
}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, ref: ModuleType, device: str) -> None:
    assert _SCENARIOS[name](lucid, device) == _SCENARIOS[name](ref, "cpu")


@pytest.mark.parametrize("hook_on", ["output", "hidden"])
def test_a_nested_pass_through_an_lstm_leaves_the_outer_gradient(hook_on: str, device: str) -> None:
    lucid.manual_seed(0)
    lstm = lucid.nn.LSTM(3, 4).to(device)
    x = lucid.randn(2, 1, 3, device=device).requires_grad_(True)

    def run(nested: bool) -> object:
        x.grad = None
        out, (h, _) = lstm(x)
        a, b = out * 2.0, h * 5.0

        def hook(g: object) -> object:
            if nested:
                with lucid.enable_grad():
                    lucid.autograd.grad((h * 3.0).sum(), x, retain_graph=True)
            return g

        (a if hook_on == "output" else b).register_hook(hook)
        (a.sum() + b.sum()).backward()
        return _values(x.grad)

    assert run(nested=True) == run(nested=False)
