"""``backward(inputs=)`` accumulates only into the tensors it names (LCD-308).

The argument was accepted and ignored: ``backward(loss, inputs=[a])`` filled
every leaf's ``.grad``.  Now the traversal is the engine's ``grad`` — only the
paths to ``inputs`` run — and each gradient is accumulated into its tensor;
a non-leaf input is marked to retain its gradient, as the reference does.

Each scenario is written once over a module ``m`` and run on Lucid and the
reference, so the reference states the expected values.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

Scenario = Callable[[ModuleType, str], dict[str, object]]


def _grad_of(t: object) -> object:
    g = getattr(t, "grad")
    return (
        None if g is None else [round(float(v), 6) for v in g.detach().cpu().reshape(-1).tolist()]
    )


def _leaf_subset(m: ModuleType, device: str) -> dict[str, object]:
    a = m.ones(2, requires_grad=True, device=device)
    b = m.ones(2, requires_grad=True, device=device)
    m.autograd.backward((a * b * 3).sum(), inputs=[a])
    return {"a": _grad_of(a), "b": _grad_of(b)}


def _single_tensor_input(m: ModuleType, device: str) -> dict[str, object]:
    a = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    b = m.tensor([3.0, 4.0], requires_grad=True, device=device)
    (a * b).sum().backward(inputs=b)
    return {"a": _grad_of(a), "b": _grad_of(b)}


def _non_leaf_input(m: ModuleType, device: str) -> dict[str, object]:
    a = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    b = m.tensor([3.0, 4.0], requires_grad=True, device=device)
    h = a * b
    (h * h).sum().backward(inputs=[h, b])
    return {"a": _grad_of(a), "b": _grad_of(b), "h": _grad_of(h)}


def _accumulates(m: ModuleType, device: str) -> dict[str, object]:
    a = m.tensor([1.0, 2.0], requires_grad=True, device=device)
    b = m.tensor([3.0, 4.0], requires_grad=True, device=device)
    y = (a * a * b).sum()
    y.backward(inputs=[a], retain_graph=True)
    y.backward(inputs=[a, b])
    return {"a": _grad_of(a), "b": _grad_of(b)}


def _unreached_input(m: ModuleType, device: str) -> dict[str, object]:
    a = m.ones(2, requires_grad=True, device=device)
    c = m.ones(2, requires_grad=True, device=device)
    (a * 2).sum().backward(inputs=[a, c])
    return {"a": _grad_of(a), "c": _grad_of(c)}


def _named_twice(m: ModuleType, device: str) -> dict[str, object]:
    a = m.ones(2, requires_grad=True, device=device)
    (a * 2).sum().backward(inputs=[a, a])
    return {"a": _grad_of(a)}


def _leaf_hook_once(m: ModuleType, device: str) -> dict[str, object]:
    a = m.ones(2, requires_grad=True, device=device)
    b = m.ones(2, requires_grad=True, device=device)
    seen: list[str] = []
    a.register_hook(lambda g: seen.append("a"))
    b.register_hook(lambda g: seen.append("b"))
    (a * b).sum().backward(inputs=[a])
    return {"seen": seen, "a": _grad_of(a)}


def _several_roots(m: ModuleType, device: str) -> dict[str, object]:
    # Two roots sharing a node: the first pass must not free it.
    x = m.ones(2, requires_grad=True, device=device)
    w = m.ones(2, requires_grad=True, device=device)
    h = x * w
    m.autograd.backward([h.sum(), (h * 3).sum()], inputs=[x])
    y = m.ones(2, requires_grad=True, device=device)
    k = y * 2
    m.autograd.backward([k.sum(), (k * 3).sum()])
    return {"x": _grad_of(x), "w": _grad_of(w), "y": _grad_of(y)}


_SCENARIOS: dict[str, Scenario] = {
    "leaf-subset": _leaf_subset,
    "single-tensor-input": _single_tensor_input,
    "non-leaf-input": _non_leaf_input,
    "accumulates-with-retain-graph": _accumulates,
    "unreached-input": _unreached_input,
    "named-twice": _named_twice,
    "leaf-hook-once": _leaf_hook_once,
    "several-roots": _several_roots,
}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, ref: ModuleType, device: str) -> None:
    assert _SCENARIOS[name](lucid, device) == _SCENARIOS[name](ref, "cpu")


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_the_values(name: str, device: str) -> None:
    # Without the reference installed, the values the reference gives.
    expected: dict[str, dict[str, object]] = {
        "leaf-subset": {"a": [3.0, 3.0], "b": None},
        "single-tensor-input": {"a": None, "b": [1.0, 2.0]},
        "non-leaf-input": {"a": None, "b": [6.0, 32.0], "h": [6.0, 16.0]},
        "accumulates-with-retain-graph": {"a": [12.0, 32.0], "b": [1.0, 4.0]},
        "unreached-input": {"a": [2.0, 2.0], "c": None},
        "named-twice": {"a": [2.0, 2.0]},
        "leaf-hook-once": {"seen": ["a"], "a": [1.0, 1.0]},
        "several-roots": {"x": [4.0, 4.0], "w": None, "y": [8.0, 8.0]},
    }
    assert _SCENARIOS[name](lucid, device) == expected[name]


def test_a_freed_graph_still_refuses_a_second_pass(device: str) -> None:
    a = lucid.ones(2, requires_grad=True, device=device)
    y = (a * a).sum()
    y.backward(inputs=[a])
    with pytest.raises(RuntimeError, match="second time"):
        y.backward(inputs=[a])


@pytest.mark.parametrize("module", ["lucid", "ref"])
def test_empty_inputs_are_refused(module: str, device: str, request: pytest.FixtureRequest) -> None:
    m: ModuleType = lucid if module == "lucid" else request.getfixturevalue("ref")
    a = m.ones(2, requires_grad=True, device=device if module == "lucid" else "cpu")
    with pytest.raises(RuntimeError, match="empty"):
        m.autograd.backward((a * 2).sum(), inputs=[])
    assert a.grad is None


@pytest.mark.parametrize("module", ["lucid", "ref"])
def test_an_input_that_does_not_require_grad_is_refused(
    module: str, request: pytest.FixtureRequest
) -> None:
    m: ModuleType = lucid if module == "lucid" else request.getfixturevalue("ref")
    a = m.ones(2, requires_grad=True)
    c = m.ones(2)
    with pytest.raises(RuntimeError) as info:
        (a * c).sum().backward(inputs=[c])
    if m is lucid:
        assert "does not require grad" in str(info.value)


def test_create_graph_with_inputs_accumulates_across_passes(device: str) -> None:
    a = lucid.tensor([1.0, 2.0], requires_grad=True, device=device)
    b = lucid.tensor([1.0, 1.0], requires_grad=True, device=device)
    (a * a * b).sum().backward(inputs=[a], create_graph=True)
    (a * a * a).sum().backward(inputs=[a], create_graph=True)
    assert a.grad is not None and a.grad.tolist() == [5.0, 16.0]
    assert b.grad is None
    (g,) = lucid.autograd.grad(a.grad.sum(), a)
    assert g.tolist() == [8.0, 14.0]


def test_create_graph_with_inputs_gives_a_differentiable_grad() -> None:
    # The reference's answer: d/da sum(a^3) = 3a^2, and its derivative 6a.
    a = lucid.tensor([1.0, 2.0], requires_grad=True)
    b = lucid.tensor([1.0, 1.0], requires_grad=True)
    (a * a * a * b).sum().backward(inputs=[a], create_graph=True)
    assert a.grad is not None and a.grad.tolist() == [3.0, 12.0]
    assert b.grad is None
    (g,) = lucid.autograd.grad(a.grad.sum(), a)
    assert g.tolist() == [6.0, 12.0]
