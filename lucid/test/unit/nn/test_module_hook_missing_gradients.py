"""What a hooked module's backward does with a gradient that is not there (LCD-152).

The reference runs a module's backward hooks through a Function wrapped
around its inputs and one around its outputs, and three rules follow:

* ``None`` in the tuple a hook returns drops that gradient — the tuple is
  what flows on.  Lucid kept the gradient the hook was handed.
* The inputs' Function hands zeros for an input no gradient reached, so the
  full hooks and the input itself see zeros, not ``None`` — an input the
  body never used included.
* A gradient that reaches the inputs without passing through the outputs —
  a double backward through what the body saved — has no ``grad_output``,
  and the full hooks do not run.  Lucid ran them with ``grad_output=(None,)``.

Each scenario is written once over a module ``m`` and run on Lucid and the
reference, which states what is expected.
"""

import warnings
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

Scenario = Callable[[ModuleType, str], object]


def _values(t: object) -> object:
    if t is None:
        return None
    return [round(float(v), 5) for v in t.detach().cpu().reshape(-1).tolist()]


def _module(
    m: ModuleType,
    body: Callable[[object, object], object],
    pre: Callable[..., object] | None = None,
    full: Callable[..., object] | None = None,
) -> object:
    class Body(m.nn.Module):
        def forward(self, a: object, b: object) -> object:
            return body(a, b)

    mod = Body()
    if pre is not None:
        mod.register_full_backward_pre_hook(pre)
    if full is not None:
        mod.register_full_backward_hook(full)
    return mod


def _operands(m: ModuleType, device: str) -> tuple[object, object]:
    a = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    b = m.tensor([3.0, 4.0], requires_grad=True, device=device)
    return a, b


def _full_hook_drops_one(m: ModuleType, device: str) -> object:
    mod = _module(m, lambda a, b: a * b, full=lambda _m, gi, go: (None, gi[1]))
    a, b = _operands(m, device)
    mod(a, b).sum().backward()
    return [_values(a.grad), _values(b.grad)]


def _full_hook_drops_all(m: ModuleType, device: str) -> object:
    mod = _module(m, lambda a, b: a * b, full=lambda _m, gi, go: (None, None))
    a, b = _operands(m, device)
    mod(a, b).sum().backward()
    return [_values(a.grad), _values(b.grad)]


def _full_hook_drops_under_create_graph(m: ModuleType, device: str) -> object:
    mod = _module(m, lambda a, b: a * b, full=lambda _m, gi, go: (None, gi[1]))
    a, b = _operands(m, device)
    ga, gb = m.autograd.grad(mod(a, b).sum(), (a, b), create_graph=True, allow_unused=True)
    return [_values(ga), _values(gb)]


def _pre_hook_drops_one(m: ModuleType, device: str) -> object:
    seen: list[object] = []
    mod = _module(
        m,
        lambda a, b: (a * 2.0, b * 3.0),
        pre=lambda _m, go: (None, go[1]),
        full=lambda _m, gi, go: seen.append([[_values(g) for g in gi], [_values(g) for g in go]]),
    )
    a, b = _operands(m, device)
    y1, y2 = mod(a, b)
    (y1.sum() + y2.sum()).backward()
    return [seen, _values(a.grad), _values(b.grad)]


def _unused_input(m: ModuleType, device: str) -> object:
    seen: list[object] = []
    mod = _module(
        m, lambda a, b: a * 2.0, full=lambda _m, gi, go: seen.append([_values(g) for g in gi])
    )
    a, b = _operands(m, device)
    mod(a, b).sum().backward()
    return [seen, _values(a.grad), _values(b.grad)]


def _unused_input_pre_hook_only(m: ModuleType, device: str) -> object:
    mod = _module(m, lambda a, b: a * 2.0, pre=lambda _m, go: None)
    a, b = _operands(m, device)
    mod(a, b).sum().backward()
    return [_values(a.grad), _values(b.grad)]


def _double_backward(m: ModuleType, device: str) -> object:
    calls: list[object] = []

    class Square(m.nn.Module):
        def forward(self, x: object) -> object:
            return x * x

    mod = Square()
    mod.register_full_backward_pre_hook(lambda _m, go: calls.append(("pre", _values(go[0]))))
    mod.register_full_backward_hook(
        lambda _m, gi, go: calls.append(("full", _values(gi[0]), _values(go[0])))
    )
    x = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    (g,) = m.autograd.grad(mod(x).sum(), x, create_graph=True)
    first = list(calls)
    (gg,) = m.autograd.grad(g.sum(), x)
    return [first, calls[len(first) :], _values(gg)]


def _empty_gradient(m: ModuleType, device: str) -> object:
    # A gradient of no elements is a gradient, not a missing one: on the CPU
    # it owns no memory and read as None, so the hooks and the body were
    # skipped (LCD-168).
    seen: list[object] = []

    class Double(m.nn.Module):
        def forward(self, x: object) -> object:
            return x * 2

    mod = Double()
    mod.register_full_backward_hook(lambda _m, gi, go: seen.append(tuple(go[0].shape)))
    x = m.ones(0, 3, requires_grad=True, device=device)
    mod(x).sum().backward()
    return [seen, None if x.grad is None else tuple(x.grad.shape)]


def _output_feeding_a_sibling_output(m: ModuleType, device: str) -> object:
    # The output barrier handed its slot's buffer on while the slot still
    # held it, and a later add into it changed what the full hook saw
    # (LCD-168: [5, 5] on the CPU, [3, 3] in the reference).
    seen: list[object] = []

    class Branch(m.nn.Module):
        def forward(self, x: object) -> object:
            h = x * 1.0
            return h, h * 2

    mod = Branch()
    mod.register_full_backward_hook(lambda _m, gi, go: seen.append(_values(go[0])))
    x = m.ones(2, requires_grad=True, device=device)
    h, y = mod(x)
    (3 * h.sum() + y.sum()).backward()
    return [seen, _values(x.grad)]


def _pre_hook_drops_all(m: ModuleType, device: str) -> object:
    mod = _module(m, lambda a, b: a * b, pre=lambda _m, go: (None,))
    a, b = _operands(m, device)
    mod(a, b).sum().backward()
    return [_values(a.grad), _values(b.grad)]


def _pre_hook_drops_all_under_create_graph(m: ModuleType, device: str) -> object:
    seen: list[object] = []
    mod = _module(
        m,
        lambda a, b: a * b,
        pre=lambda _m, go: (None,),
        full=lambda _m, gi, go: seen.append([[_values(g) for g in gi], [_values(g) for g in go]]),
    )
    a, b = _operands(m, device)
    ga, gb = m.autograd.grad(mod(a, b).sum(), (a, b), create_graph=True)
    return [seen, _values(ga), _values(gb)]


_SCENARIOS: dict[str, Scenario] = {
    "empty-gradient": _empty_gradient,
    "output-feeding-a-sibling-output": _output_feeding_a_sibling_output,
    "full-hook-drops-one": _full_hook_drops_one,
    "full-hook-drops-all": _full_hook_drops_all,
    "full-hook-drops-under-create-graph": _full_hook_drops_under_create_graph,
    "pre-hook-drops-one": _pre_hook_drops_one,
    "pre-hook-drops-all": _pre_hook_drops_all,
    "pre-hook-drops-all-under-create-graph": _pre_hook_drops_all_under_create_graph,
    "unused-input": _unused_input,
    "unused-input-pre-hook-only": _unused_input_pre_hook_only,
    "double-backward": _double_backward,
}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, ref: ModuleType, device: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = _SCENARIOS[name](ref, "cpu")
    assert _SCENARIOS[name](lucid, device) == want


def test_a_pre_hook_dropping_every_gradient_leaves_zeros(ref: ModuleType, device: str) -> None:
    def run(m: ModuleType, dev: str) -> object:
        seen: list[object] = []

        class Weighted(m.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = m.nn.Parameter(m.tensor([2.0, 3.0], device=dev))

            def forward(self, a: object, b: object) -> object:
                return a * b * self.w

        mod = Weighted()
        mod.register_full_backward_pre_hook(lambda _m, go: (None,))
        mod.register_full_backward_hook(
            lambda _m, gi, go: seen.append([[_values(g) for g in gi], [_values(g) for g in go]])
        )
        a, b = _operands(m, dev)
        mod(a, b).sum().backward()
        return [
            _pre_hook_drops_all(m, dev),
            seen,
            _values(a.grad),
            _values(b.grad),
            _values(mod.w.grad),
        ]

    assert run(lucid, device) == run(ref, "cpu")
