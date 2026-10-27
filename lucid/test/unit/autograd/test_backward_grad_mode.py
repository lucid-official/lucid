"""A backward pass runs with grad mode on exactly when it records a graph (LCD-152).

The reference runs every pass with grad mode set to ``create_graph``: a
hook, a module backward hook or a custom Function's ``backward`` then builds
the graph of the gradients when one is asked for, and otherwise builds none.
Tensor hooks already ran that way; a module's hooks and a custom Function's
``backward`` inherited whatever the caller's mode was — on in a plain
``backward()``, so a tensor made there required grad and grew a graph nobody
would use, and off under ``no_grad`` with ``create_graph=True``, so the
gradient they built had no graph.  The engine now sets the mode for the
whole pass.

Each scenario is written once over a module ``m`` and run on Lucid and the
reference.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

Scenario = Callable[[ModuleType, str, bool, str], object]


def _caller_mode(m: ModuleType, mode: str) -> object:
    return m.no_grad() if mode == "no_grad" else m.enable_grad()


def _module_hooks(m: ModuleType, device: str, create_graph: bool, mode: str) -> object:
    class Double(m.nn.Module):
        def forward(self, a: object) -> object:
            return a * 2.0

    mod = Double()
    seen: dict[str, bool] = {}
    mod.register_full_backward_pre_hook(lambda _m, go: seen.__setitem__("pre", m.is_grad_enabled()))
    mod.register_full_backward_hook(
        lambda _m, gi, go: seen.__setitem__("full", m.is_grad_enabled())
    )
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    loss = mod(x).sum()
    with _caller_mode(m, mode):
        loss.backward(create_graph=create_graph)
    return seen


def _tensor_hook(m: ModuleType, device: str, create_graph: bool, mode: str) -> object:
    seen: list[bool] = []
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    y = x * x
    y.register_hook(lambda g: seen.append(m.is_grad_enabled()))
    loss = y.sum()
    with _caller_mode(m, mode):
        loss.backward(create_graph=create_graph)
    return seen


def _function_backward(m: ModuleType, device: str, create_graph: bool, mode: str) -> object:
    seen: list[bool] = []

    class Twice(m.autograd.Function):
        @staticmethod
        def forward(ctx: object, a: object) -> object:
            return a * 2.0

        @staticmethod
        def backward(ctx: object, g: object) -> object:
            seen.append(m.is_grad_enabled())
            return g * 2.0

    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    loss = Twice.apply(x).sum()
    with _caller_mode(m, mode):
        loss.backward(create_graph=create_graph)
    return seen


def _multi_output_function_backward(
    m: ModuleType, device: str, create_graph: bool, mode: str
) -> object:
    seen: list[bool] = []

    class Split(m.autograd.Function):
        @staticmethod
        def forward(ctx: object, a: object) -> object:
            return a * 2.0, a * 3.0

        @staticmethod
        def backward(ctx: object, g0: object, g1: object) -> object:
            seen.append(m.is_grad_enabled())
            return g0 * 2.0 + g1 * 3.0

    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    y0, y1 = Split.apply(x)
    loss = (y0 * y1).sum()
    with _caller_mode(m, mode):
        loss.backward(create_graph=create_graph)
    return seen


def _hook_built_tensor(m: ModuleType, device: str, create_graph: bool, mode: str) -> object:
    class Double(m.nn.Module):
        def forward(self, a: object) -> object:
            return a * 2.0

    mod = Double()
    w = m.ones(2, requires_grad=True, device=device)
    seen: list[bool] = []
    mod.register_full_backward_hook(lambda _m, gi, go: seen.append(bool((go[0] * w).requires_grad)))
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    loss = mod(x).sum()
    with _caller_mode(m, mode):
        loss.backward(create_graph=create_graph)
    return seen


def _grad_create_graph_under_no_grad(
    m: ModuleType, device: str, create_graph: bool, mode: str
) -> object:
    class Cube(m.autograd.Function):
        @staticmethod
        def forward(ctx: object, a: object) -> object:
            ctx.save_for_backward(a)
            return a * a * a

        @staticmethod
        def backward(ctx: object, g: object) -> object:
            (a,) = ctx.saved_tensors
            return g * 3.0 * a * a

    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    y = Cube.apply(x).sum()
    with _caller_mode(m, mode):
        (g,) = m.autograd.grad(y, x, create_graph=create_graph)
    return [bool(g.requires_grad), [round(float(v), 5) for v in g.detach().cpu().tolist()]]


_SCENARIOS: dict[str, Scenario] = {
    "module-hooks": _module_hooks,
    "tensor-hook": _tensor_hook,
    "function-backward": _function_backward,
    "multi-output-function-backward": _multi_output_function_backward,
    "hook-built-tensor": _hook_built_tensor,
    "grad-through-function": _grad_create_graph_under_no_grad,
}


@pytest.mark.parametrize("mode", ["enable_grad", "no_grad"])
@pytest.mark.parametrize("create_graph", [False, True], ids=["eager", "create_graph"])
@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(
    name: str, create_graph: bool, mode: str, ref: ModuleType, device: str
) -> None:
    scenario = _SCENARIOS[name]
    assert scenario(lucid, device, create_graph, mode) == scenario(ref, "cpu", create_graph, mode)


@pytest.mark.parametrize("create_graph", [False, True])
def test_the_callers_grad_mode_is_restored(create_graph: bool, device: str) -> None:
    x = lucid.tensor([1.0, 2.0], requires_grad=True, device=device)
    loss = (x * x).sum()
    with lucid.no_grad():
        loss.backward(create_graph=create_graph, retain_graph=True)
        assert not lucid.is_grad_enabled()
    loss.backward(create_graph=create_graph)
    assert lucid.is_grad_enabled()
