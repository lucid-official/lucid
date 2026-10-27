"""A hooked module call lives exactly as long as its graph (LCD-167).

A module with a backward hook or pre-hook hands the engine, at every call,
a state whose runners call back into Python.  The runner object also held
that engine state: runner -> engine state -> runner, a cycle through C++
that the garbage collector cannot see.  Every call leaked the runner and,
through it, the module — with or without a backward, so a training loop
over a hooked module grew without bound.

The engine state is now owned by the call's graph alone.  Pinned here: the
module and every call's runner die once the module and the graph are
dropped, for each kind of hook and each way a graph can end (backward,
none, a retained graph run twice), on one module called many times, and
the hooks still see and return what the reference's do — including on a
module that only the pending graph still reaches.
"""

import gc
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.nn.module import _ModuleBackwardState

# ── harness ────────────────────────────────────────────────────────────────


def _full_hook(module: object, grad_input: object, grad_output: object) -> None:
    del module, grad_input, grad_output  # the hook only has to be there


def _pre_hook(module: object, grad_output: object) -> None:
    del module, grad_output


@contextmanager
def _global_hook(kind: str) -> Iterator[None]:
    """One global hook of ``kind`` (if it is a global one) for the block."""
    handle = None
    if kind == "global_full_backward_hook":
        handle = nn.register_module_full_backward_hook(_full_hook)
    elif kind == "global_full_backward_pre_hook":
        handle = nn.register_module_full_backward_pre_hook(_pre_hook)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def _hooked_linear(kind: str, device: str) -> nn.Module:
    """A ``Linear`` with one hook of ``kind`` if it is a per-module one.

    Built here rather than in a context manager so no frame but the
    test's holds the module.
    """
    module = nn.Linear(3, 3, device=device)
    if kind == "full_backward_hook":
        module.register_full_backward_hook(_full_hook)
    elif kind == "full_backward_pre_hook":
        module.register_full_backward_pre_hook(_pre_hook)
    return module


def _states_of(module: nn.Module) -> list[_ModuleBackwardState]:
    gc.collect()
    return [
        obj
        for obj in gc.get_objects()
        if isinstance(obj, _ModuleBackwardState) and obj.module is module
    ]


def _step(module: nn.Module, mode: str, device: str) -> None:
    x = lucid.randn(2, 3, device=device, requires_grad=True)
    y = module(x)
    if mode == "backward":
        y.sum().backward()
    elif mode == "retain-graph-twice":
        loss = y.sum()
        loss.backward(retain_graph=True)
        loss.backward()


_KINDS = [
    "full_backward_hook",
    "full_backward_pre_hook",
    "global_full_backward_hook",
    "global_full_backward_pre_hook",
]
_MODES = ["backward", "no-backward", "retain-graph-twice"]

# A hooked call's runners live on the engine state its input and output hook
# nodes share, and the input node is owned through a tensor an op saved.  A
# node reports the tensors it keeps for backward (Node::for_each_owned_tensor),
# so the collector can prove the state is reachable only through the garbage
# and a cycle closed before backward ever runs is collected.  Two cases —
# several kept tensors from one graph, a ctx saving its own output — are
# owned jointly, so no one wrapper's region holds them, and they leak in the
# reference too.
_JOINTLY_OWNED_LEAK = pytest.mark.xfail(
    strict=True,
    reason="LCD-459: owned jointly by several wrappers, so no region holds the "
    "cycle; the reference leaks it too",
)


# ── lifetime ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("kind", _KINDS)
def test_the_module_and_its_states_are_freed(kind: str, mode: str, device: str) -> None:
    with _global_hook(kind):
        module = _hooked_linear(kind, device)
        x = lucid.randn(2, 3, device=device, requires_grad=True)
        y = module(x)
        states = [weakref.ref(s) for s in _states_of(module)]
        assert len(states) == 1  # the call made one, and the graph holds it
        if mode == "backward":
            y.sum().backward()
        elif mode == "retain-graph-twice":
            loss = y.sum()
            loss.backward(retain_graph=True)
            loss.backward()
            del loss
        module_ref = weakref.ref(module)
        del module, x, y
        gc.collect()
        assert module_ref() is None, "the module outlived every reference to it"
        assert all(s() is None for s in states), "a call's state outlived its graph"


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("kind", _KINDS)
def test_one_module_called_many_times_holds_no_state(kind: str, mode: str, device: str) -> None:
    with _global_hook(kind):
        module = _hooked_linear(kind, device)
        for _ in range(10):
            _step(module, mode, device)
            assert _states_of(module) == []


def test_a_state_lives_as_long_as_its_graph(device: str) -> None:
    module = nn.Linear(3, 3, device=device)
    module.register_full_backward_hook(_full_hook)
    outputs = [module(lucid.randn(2, 3, device=device, requires_grad=True))]
    outputs.append(module(lucid.randn(2, 3, device=device, requires_grad=True)))
    assert len(_states_of(module)) == 2
    outputs.pop().sum().backward()
    assert len(_states_of(module)) == 1
    outputs.clear()
    assert _states_of(module) == []


def test_a_module_holding_its_hooked_output_is_freed(device: str) -> None:
    module = nn.Linear(3, 3, device=device)
    module.register_full_backward_hook(_full_hook)
    module.last = module(lucid.randn(2, 3, device=device, requires_grad=True))
    module_ref = weakref.ref(module)
    del module
    gc.collect()
    assert module_ref() is None


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("kind", _KINDS)
def test_a_module_holding_its_hooked_output_across_a_deeper_graph_is_freed(
    kind: str, mode: str, device: str
) -> None:
    with _global_hook(kind):
        module = _hooked_linear(kind, device)
        x = lucid.randn(2, 3, device=device, requires_grad=True)
        y = module(x) * 2 + 1
        module.last = y
        if mode == "backward":
            y.sum().backward()
        elif mode == "retain-graph-twice":
            loss = y.sum()
            loss.backward(retain_graph=True)
            loss.backward()
            del loss
        module_ref = weakref.ref(module)
        output_ref = weakref.ref(y)
        del module, x, y
        gc.collect()
        assert module_ref() is None
        assert output_ref() is None


@_JOINTLY_OWNED_LEAK
def test_a_module_holding_multiple_tensors_from_the_same_hooked_graph_is_freed(
    device: str,
) -> None:
    module = nn.Linear(3, 3, device=device)
    module.register_full_backward_hook(_full_hook)
    module.register_full_backward_pre_hook(_pre_hook)
    x = lucid.randn(2, 3, device=device, requires_grad=True)
    y = module(x)
    z = y * 2 + 1
    module.y = y
    module.z = z
    module_ref = weakref.ref(module)
    y_ref = weakref.ref(y)
    z_ref = weakref.ref(z)
    del module, x, y, z
    gc.collect()
    assert module_ref() is None
    assert y_ref() is None
    assert z_ref() is None


def test_a_live_downstream_branch_keeps_a_cyclic_hooked_module_alive_until_dropped(
    device: str,
) -> None:
    calls: list[int] = []
    module = nn.Linear(3, 3, device=device)
    module.register_full_backward_hook(lambda mod, gi, go: calls.append(1))
    x = lucid.randn(2, 3, device=device, requires_grad=True)
    y = module(x)
    z = y * 2
    module.last = y
    module_ref = weakref.ref(module)
    del module, y
    gc.collect()
    # ``z`` is still reachable and its backward path runs through the module.
    assert module_ref() is not None
    z.sum().backward()
    assert calls == [1]
    del z, x
    gc.collect()
    assert module_ref() is None


@pytest.mark.parametrize(
    "run_backward",
    [
        pytest.param(False, id="no-backward"),
        pytest.param(True, id="backward"),
    ],
)
def test_custom_function_cycle_through_ctx_or_module_is_freed(
    run_backward: bool, device: str
) -> None:
    class SelfSavingFn(lucid.autograd.Function):
        @staticmethod
        def forward(ctx: object, x: lucid.Tensor, owner: object) -> lucid.Tensor:
            out = x * 2
            ctx.save_for_backward(out)  # type: ignore[attr-defined]
            ctx.owner = owner  # type: ignore[attr-defined]
            return out

        @staticmethod
        def backward(ctx: object, grad_out: lucid.Tensor) -> tuple[lucid.Tensor, None]:
            (out,) = ctx.saved_tensors  # type: ignore[attr-defined]
            return grad_out * (out * 0 + 2), None

    module = nn.Linear(3, 3, device=device)
    x = lucid.randn(2, 3, device=device, requires_grad=True)
    y = SelfSavingFn.apply(module(x), module)
    assert isinstance(y, lucid.Tensor)
    module.last = y * 3
    if run_backward:
        module.last.sum().backward()
    module_ref = weakref.ref(module)
    y_ref = weakref.ref(y)
    del module, x, y
    gc.collect()
    assert module_ref() is None
    assert y_ref() is None


@pytest.mark.parametrize("leaf", [False, True], ids=["non-leaf", "leaf"])
def test_tensor_hook_cycle_through_module_is_freed(leaf: bool, device: str) -> None:
    module = nn.Linear(3, 3, device=device)
    x = lucid.randn(2, 3, device=device, requires_grad=True)
    target = x if leaf else module(x) * 2
    target.register_hook(lambda g, m=module, t=target: g)
    module.held = target
    module_ref = weakref.ref(module)
    target_ref = weakref.ref(target)
    del module, x, target
    gc.collect()
    assert module_ref() is None
    assert target_ref() is None


def test_repeated_gc_collect_cycles_with_hooked_modules(device: str) -> None:
    refs: list[weakref.ref[nn.Module]] = []
    for i in range(12):
        module = nn.Linear(3, 3, device=device)
        module.register_full_backward_hook(_full_hook)
        module.register_full_backward_pre_hook(_pre_hook)
        x = lucid.randn(2, 3, device=device, requires_grad=True)
        y = module(x) * 2 + 1
        module.last = y
        if i % 2 == 0:
            y.sum().backward()
        refs.append(weakref.ref(module))
        del module, x, y
        gc.collect()
    assert all(r() is None for r in refs)


# ── hook semantics against the reference ───────────────────────────────────


def _plain(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value.detach().tolist()  # type: ignore[attr-defined]


def _assert_close(got: object, want: object, tol: float = 1e-5) -> None:
    if isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), (got, want)
        for g, w in zip(got, want, strict=True):
            _assert_close(g, w, tol)
    elif isinstance(want, float):
        assert isinstance(got, float), (got, want)
        assert abs(got - want) <= tol * max(1.0, abs(want)), (got, want)
    else:
        assert got == want, (got, want)


def _linear(lib: ModuleType, device: str) -> object:
    lin = lib.nn.Linear(2, 2, bias=False)
    with lib.no_grad():
        lin.weight.copy_(lib.tensor([[1.0, 0.5], [-1.0, 2.0]]))
    return lin.to(device)


def _input(lib: ModuleType, values: list[object], device: str) -> object:
    return lib.tensor(values, device=device, requires_grad=True)


def _hook_all(lib: ModuleType, lin: object, log: list[object]) -> None:
    """Two pre-hooks and two full hooks; the second of each returns."""

    def pre_record(module: object, grad_output: object) -> None:
        log.append(["pre", type(module).__name__, _plain(grad_output)])

    def pre_scale(module: object, grad_output: tuple[object, ...]) -> object:
        del module
        return tuple(3.0 * g for g in grad_output)  # type: ignore[operator]

    def full_record(module: object, grad_input: object, grad_output: object) -> None:
        log.append(["full", type(module).__name__, _plain(grad_input), _plain(grad_output)])

    def full_negate(module: object, grad_input: tuple[object, ...], grad_output: object) -> object:
        del module, grad_output
        return tuple(None if g is None else -g for g in grad_input)  # type: ignore[operator]

    lin.register_full_backward_pre_hook(pre_record)  # type: ignore[attr-defined]
    lin.register_full_backward_pre_hook(pre_scale)  # type: ignore[attr-defined]
    lin.register_full_backward_hook(full_record)  # type: ignore[attr-defined]
    lin.register_full_backward_hook(full_negate)  # type: ignore[attr-defined]


def _plain_backward(lib: ModuleType, device: str) -> dict[str, object]:
    log: list[object] = []
    lin = _linear(lib, device)
    _hook_all(lib, lin, log)
    x = _input(lib, [[1.0, 2.0]], device)
    (lin(x) * lib.tensor([[1.0, -2.0]], device=device)).sum().backward()  # type: ignore[operator]
    return {"x": _plain(x.grad), "w": _plain(lin.weight.grad), "hooks": log}  # type: ignore[attr-defined]


def _retained_twice(lib: ModuleType, device: str) -> dict[str, object]:
    log: list[object] = []
    lin = _linear(lib, device)
    _hook_all(lib, lin, log)
    x = _input(lib, [[1.0, 2.0]], device)
    loss = (lin(x) ** 2).sum()  # type: ignore[operator]
    loss.backward(retain_graph=True)
    loss.backward()
    return {"x": _plain(x.grad), "w": _plain(lin.weight.grad), "hooks": log}  # type: ignore[attr-defined]


def _module_dropped_before_backward(lib: ModuleType, device: str) -> dict[str, object]:
    # Only the pending graph still reaches the module: its hooks run anyway.
    log: list[object] = []
    lin = _linear(lib, device)
    _hook_all(lib, lin, log)
    x = _input(lib, [[1.0, 2.0]], device)
    loss = lin(x).sum()  # type: ignore[operator]
    del lin
    gc.collect()
    loss.backward()
    return {"x": _plain(x.grad), "hooks": log}


def _two_calls_two_graphs(lib: ModuleType, device: str) -> dict[str, object]:
    # Each call keeps its own state: backward of the later call first.
    log: list[object] = []
    lin = _linear(lib, device)
    _hook_all(lib, lin, log)
    a = _input(lib, [[1.0, 2.0]], device)
    b = _input(lib, [[-1.0, 0.5]], device)
    first = lin(a).sum()  # type: ignore[operator]
    second = (2 * lin(b)).sum()  # type: ignore[operator]
    second.backward()
    first.backward()
    return {"a": _plain(a.grad), "b": _plain(b.grad), "hooks": log}


def _forward_only_then_backward(lib: ModuleType, device: str) -> dict[str, object]:
    # Calls whose graph is dropped unrun leave nothing behind for later ones.
    log: list[object] = []
    lin = _linear(lib, device)
    _hook_all(lib, lin, log)
    for _ in range(3):
        lin(_input(lib, [[5.0, 5.0]], device))  # type: ignore[operator]
    x = _input(lib, [[1.0, 2.0]], device)
    lin(x).sum().backward()  # type: ignore[operator]
    return {"x": _plain(x.grad), "hooks": log}


_SCENARIOS: dict[str, Callable[[ModuleType, str], dict[str, object]]] = {
    "backward": _plain_backward,
    "retain-graph-twice": _retained_twice,
    "module-dropped-before-backward": _module_dropped_before_backward,
    "two-calls-two-graphs": _two_calls_two_graphs,
    "forward-only-then-backward": _forward_only_then_backward,
}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_hooks_match_the_reference(name: str, ref: ModuleType, device: str) -> None:
    got = _SCENARIOS[name](lucid, device)
    want = _SCENARIOS[name](ref, "cpu")
    assert got.keys() == want.keys()
    for key in want:
        _assert_close(got[key], want[key])
