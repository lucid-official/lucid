"""``autograd.grad`` fills every requested tensor at a shared node (CHA-123).

``autograd.grad`` finds each requested tensor by the node that produced
it, and kept one requested index per node.  Two requests at one node —
both outputs of a multi-output Function, both outputs of a hooked module,
or simply the same tensor asked for twice — answered only the first; the
rest were reported "not reachable" and raised.  The reference answers
every one of them.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.test._helpers.compare import assert_close


def _compare(got: tuple[object, ...], want: tuple[object, ...]) -> None:
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if w is None or g is None:
            assert g is None and w is None, i
        else:
            assert_close(g, w, msg=str(i))


def _two_outputs(lib: ModuleType) -> type:
    class Two(lib.autograd.Function):  # type: ignore[name-defined, misc]
        @staticmethod
        def forward(ctx: object, a: object) -> object:
            return a * 2, a * 3

        @staticmethod
        def backward(ctx: object, g1: object, g2: object) -> object:
            return g1 * 2 + g2 * 3  # type: ignore[operator]

    return Two


def _a(lib: ModuleType, device: str) -> object:
    return lib.tensor([1.0, -0.5], device=device, requires_grad=True)


# ── scenarios: run on Lucid and on the reference ──────────────────────────


def _function_outputs(lib: ModuleType, device: str, create_graph: bool) -> tuple[object, ...]:
    o1, o2 = _two_outputs(lib).apply(_a(lib, device))
    return lib.autograd.grad((5 * o2).sum() + (o1 * o1).sum(), (o1, o2), create_graph=create_graph)


def _hooked_module_outputs(lib: ModuleType, device: str, create_graph: bool) -> tuple[object, ...]:
    class Pair(lib.nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, a: object) -> object:
            return a * 2, a * a  # type: ignore[operator]

    m = Pair()
    m.register_full_backward_hook(lambda mod, gi, go: None)
    o1, o2 = m(_a(lib, device))
    return lib.autograd.grad((5 * o2).sum() + (o1 * o1).sum(), (o2, o1), create_graph=create_graph)


def _same_leaf_twice(lib: ModuleType, device: str, create_graph: bool) -> tuple[object, ...]:
    x = _a(lib, device)
    return lib.autograd.grad((3 * x * x).sum(), (x, x), create_graph=create_graph)


def _same_interior_twice(lib: ModuleType, device: str, create_graph: bool) -> tuple[object, ...]:
    x = _a(lib, device)
    y = x * 4
    return lib.autograd.grad((y * y).sum(), (y, x, y), create_graph=create_graph)


def _leaf_root_twice(lib: ModuleType, device: str, create_graph: bool) -> tuple[object, ...]:
    x = lib.tensor(2.0, device=device, requires_grad=True)
    return lib.autograd.grad(x, (x, x), create_graph=create_graph)


def _unused_alongside(lib: ModuleType, device: str, create_graph: bool) -> tuple[object, ...]:
    o1, o2 = _two_outputs(lib).apply(_a(lib, device))
    return lib.autograd.grad(
        (o1 * o1).sum(), (o1, o2, o1), allow_unused=True, create_graph=create_graph
    )


_SCENARIOS: dict[str, Callable[[ModuleType, str, bool], tuple[object, ...]]] = {
    "function-outputs": _function_outputs,
    "hooked-module-outputs": _hooked_module_outputs,
    "same-leaf-twice": _same_leaf_twice,
    "same-interior-twice": _same_interior_twice,
    "leaf-root-twice": _leaf_root_twice,
    "unused-alongside": _unused_alongside,
}


@pytest.mark.parametrize("create_graph", [False, True], ids=["eager", "create-graph"])
@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, create_graph: bool, ref: ModuleType, device: str) -> None:
    got = _SCENARIOS[name](lucid, device, create_graph)
    want = _SCENARIOS[name](ref, "cpu", create_graph)
    _compare(tuple(got), tuple(want))


# ── without the reference ──────────────────────────────────────────────────


def test_the_reported_case(device: str) -> None:
    x = lucid.ones(2, requires_grad=True, device=device)
    gx, gx_again = lucid.autograd.grad((3 * x).sum(), (x, x))
    assert gx.tolist() == [3.0, 3.0]
    assert gx_again.tolist() == [3.0, 3.0]


def test_an_unused_output_still_needs_allow_unused(device: str) -> None:
    o1, o2 = _two_outputs(lucid).apply(_a(lucid, device))
    with pytest.raises(RuntimeError, match="allow_unused"):
        lucid.autograd.grad(o1.sum(), (o1, o2))
