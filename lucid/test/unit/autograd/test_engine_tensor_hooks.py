"""The engine runs a tensor's gradient hooks where its gradient is complete (CHA-151).

``Tensor.register_hook`` was a registry Python walked after ``backward()``
returned, reading each tensor's ``.grad``.  A non-leaf has no ``.grad``, so its
hooks never ran; a hook's return value reached ``.grad`` but nothing upstream
of it; a leaf's hooks ran again on every later backward, unrelated or not, on
the accumulated gradient; and ``autograd.grad`` ran none of them.

The engine now runs them where the reference does: a non-leaf's on its
producer's output slot, once the whole gradient of that slot has arrived and
before the producer runs, so what a hook returns is what flows on; a leaf's
just before the gradient is added into ``.grad``.  These tests drive the
engine half directly — ``_tensor_hook_runner`` with a runner written here —
so they pin what the engine does whatever the Python wrapper over it does.
"""

import gc
import json
import subprocess
import sys
import weakref
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.test._fixtures.devices import metal_available
from lucid.test._helpers.compare import assert_close

Hook = Callable[[lucid.Tensor], lucid.Tensor | None]
Register = Callable[[object, Callable[[object], object]], None]

_needs_metal = pytest.mark.skipif(not metal_available(), reason="needs Metal")


class _Runner:
    """One slot's runner: its hooks, in registration order, on the gradient.

    Anything a hook returns that is not a tensor is handed to the engine as
    it is, so the engine's own check is what the tests see.
    """

    def __init__(self) -> None:
        self.hooks: list[Hook] = []

    def __call__(self, grad: _C_engine.TensorImpl) -> object:
        g = _wrap(grad)
        replaced: object = None
        for hook in self.hooks:
            result = hook(g)
            if result is not None:
                replaced = result
                if not isinstance(result, lucid.Tensor):
                    return result
                g = result
        return _unwrap(g) if replaced is not None else None


def _hook(t: object, hook: Callable[[object], object]) -> None:
    assert isinstance(t, lucid.Tensor)
    runner = _C_engine._tensor_hook_runner(_unwrap(t), _Runner)
    assert isinstance(runner, _Runner)
    runner.hooks.append(hook)  # type: ignore[arg-type]


def _ref_hook(t: object, hook: Callable[[object], object]) -> None:
    t.register_hook(hook)  # type: ignore[attr-defined]


def _compare(got: dict[str, object], want: dict[str, object]) -> None:
    assert got.keys() == want.keys()
    for key, w in want.items():
        g = got[key]
        if w is None or isinstance(w, (bool, int, list, tuple)):
            assert g == w, key
        else:
            assert_close(g, w, msg=key)


def _x(lib: ModuleType, device: str) -> object:
    return lib.tensor([1.0, -2.0, 0.5], device=device, requires_grad=True)


def _two_outputs(lib: ModuleType) -> type:
    class Two(lib.autograd.Function):  # type: ignore[name-defined, misc]
        @staticmethod
        def forward(ctx: object, a: object) -> object:
            return a * 2, a * 3

        @staticmethod
        def backward(ctx: object, g1: object, g2: object) -> object:
            return g1 * 2 + g2 * 3  # type: ignore[operator]

    return Two


def _hooked_linear(lib: ModuleType, device: str) -> object:
    lin = lib.nn.Linear(2, 2, bias=False)
    with lib.no_grad():
        lin.weight.copy_(lib.tensor([[1.0, 0.5], [-1.0, 2.0]]))
    lin = lin.to(device)
    lin.register_full_backward_hook(lambda mod, gi, go: None)
    return lin


# ── scenarios: run on Lucid and on the reference ──────────────────────────


def _consumers_before_and_after(lib: ModuleType, device: str, hook: Register) -> dict[str, object]:
    # A consumer made before the hook still sends its gradient through it:
    # the hook is on y's producer, not on the edges out of y.
    x = _x(lib, device)
    y = x * 2
    early = y * 3
    seen: list[object] = []
    hook(y, lambda g: seen.append(g.tolist()))  # type: ignore[attr-defined]
    late = y * y
    (early.sum() + late.sum()).backward()
    return {"seen": seen, "x": x.grad}  # type: ignore[attr-defined]


def _return_value_flows_upstream(lib: ModuleType, device: str, hook: Register) -> dict[str, object]:
    x = _x(lib, device)
    y = x * 2
    hook(y, lambda g: g * 10)  # type: ignore[operator]
    (y * 1.5).sum().backward()
    return {"x": x.grad}  # type: ignore[attr-defined]


def _leaf_two_passes(lib: ModuleType, device: str, hook: Register) -> dict[str, object]:
    # A leaf's hook runs on each pass's gradient, before it is added — not on
    # the accumulated one, and not during an unrelated backward.
    x = _x(lib, device)
    calls: list[object] = []

    def double(g: object) -> object:
        calls.append(g.tolist())  # type: ignore[attr-defined]
        return g * 2  # type: ignore[operator]

    hook(x, double)
    (x * 3).sum().backward()
    (x * 3).sum().backward()
    w = lib.ones(3, device=device, requires_grad=True)
    (w * 5).sum().backward()
    return {"x": x.grad, "calls": calls, "w": w.grad}  # type: ignore[attr-defined]


def _functional_grad(lib: ModuleType, device: str, hook: Register) -> dict[str, object]:
    x = lib.ones(3, device=device, requires_grad=True)
    y = x * 2
    z = y * 3
    u = lib.ones(3, device=device, requires_grad=True)
    v = u * 5
    off_path: list[object] = []
    hook(y, lambda g: g * 10)  # type: ignore[operator]
    hook(x, lambda g: g + 1)  # type: ignore[operator]
    hook(v, lambda g: off_path.append(1))
    loss = z.sum() + v.sum()
    (gy,) = lib.autograd.grad(loss, [y], retain_graph=True)
    (gx,) = lib.autograd.grad(loss, [x])
    return {
        "gy": gy,
        "gx": gx,
        "off_path": off_path,
        "x_grad": x.grad,  # type: ignore[attr-defined]
        "u_grad": u.grad,  # type: ignore[attr-defined]
    }


def _leaf_root(lib: ModuleType, device: str, hook: Register) -> dict[str, object]:
    s = lib.tensor(2.0, device=device, requires_grad=True)
    hook(s, lambda g: g * 5)  # type: ignore[operator]
    (g,) = lib.autograd.grad(s, s)
    return {"g": g}


def _create_graph(lib: ModuleType, device: str, hook: Register) -> dict[str, object]:
    # The hook runs with grad mode on and its result keeps its graph, so the
    # gradient it shaped differentiates again — through y's producer, whose
    # hook then runs once more, in an eager pass, with grad mode off.
    x = _x(lib, device)
    y = x * x
    state: list[object] = []

    def triple(g: object) -> object:
        state.append((g.requires_grad, lib.is_grad_enabled()))  # type: ignore[attr-defined]
        return g * 3  # type: ignore[operator]

    hook(y, triple)
    (y * y).sum().backward(create_graph=True)
    gx = x.grad  # type: ignore[attr-defined]
    (h,) = lib.autograd.grad(gx.sum(), x)
    return {"state": state, "gx": gx.detach(), "h": h}


def _eager_grad_mode(lib: ModuleType, device: str, hook: Register) -> dict[str, object]:
    x = _x(lib, device)
    y = x * x
    state: list[object] = []
    hook(y, lambda g: state.append((g.requires_grad, lib.is_grad_enabled())))  # type: ignore[attr-defined]
    hook(x, lambda g: state.append((g.requires_grad, lib.is_grad_enabled())))  # type: ignore[attr-defined]
    (y * y).sum().backward()
    return {"state": state, "after": lib.is_grad_enabled()}


def _function_output(lib: ModuleType, device: str, hook: Register) -> dict[str, object]:
    # One output of a two-output Function: its slot's whole gradient, once.
    a = lib.ones(2, device=device, requires_grad=True)
    o1, o2 = _two_outputs(lib).apply(a)
    seen: list[object] = []

    def scale(g: object) -> object:
        seen.append(g.tolist())  # type: ignore[attr-defined]
        return g * 10  # type: ignore[operator]

    hook(o2, scale)
    loss = o1.sum() + (o2 * 5).sum() + o2.sum()
    (g2,) = lib.autograd.grad(loss, o2, retain_graph=True)
    loss.backward()
    return {"seen": seen, "g2": g2, "a": a.grad}  # type: ignore[attr-defined]


def _module_output(lib: ModuleType, device: str, hook: Register) -> dict[str, object]:
    lin = _hooked_linear(lib, device)
    x = lib.tensor([[1.0, -2.0]], device=device, requires_grad=True)
    y = lin(x)  # type: ignore[operator]
    seen: list[object] = []

    def double(g: object) -> object:
        seen.append(g.tolist())  # type: ignore[attr-defined]
        return g * 2  # type: ignore[operator]

    hook(y, double)
    ((3 * y).sum() + (y * y).sum()).backward()
    return {"seen": seen, "x": x.grad}  # type: ignore[attr-defined]


def _hook_before_retain(lib: ModuleType, device: str, hook: Register) -> dict[str, object]:
    x = _x(lib, device)
    y = x * 2
    y.retain_grad()  # type: ignore[attr-defined]
    hook(y, lambda g: g * 10)  # type: ignore[operator]
    (y * y).sum().backward()
    return {"y": y.grad, "x": x.grad}  # type: ignore[attr-defined]


def _before_an_in_place_write(lib: ModuleType, device: str, hook: Register) -> dict[str, object]:
    # A hook registered before an in-place op stays with the values it was
    # registered on: it sees the gradient with respect to y before the write.
    x = _x(lib, device)
    y = x * 1.0
    seen: list[object] = []
    hook(y, lambda g: seen.append(g.tolist()))  # type: ignore[attr-defined]
    y.mul_(2)  # type: ignore[attr-defined]
    (y * y).sum().backward()
    return {"seen": seen, "x": x.grad}  # type: ignore[attr-defined]


_SCENARIOS: dict[str, Callable[[ModuleType, str, Register], dict[str, object]]] = {
    "consumers-before-and-after": _consumers_before_and_after,
    "return-value-flows-upstream": _return_value_flows_upstream,
    "leaf-two-passes": _leaf_two_passes,
    "functional-grad": _functional_grad,
    "leaf-root": _leaf_root,
    "create-graph": _create_graph,
    "eager-grad-mode": _eager_grad_mode,
    "function-output": _function_output,
    "module-output": _module_output,
    "hook-before-retain": _hook_before_retain,
    "before-an-in-place-write": _before_an_in_place_write,
}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, ref: ModuleType, device: str) -> None:
    _compare(_SCENARIOS[name](lucid, device, _hook), _SCENARIOS[name](ref, "cpu", _ref_hook))


# ── the engine's contract, without the reference ───────────────────────────


def test_the_reported_cases(device: str) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    y = x * 2
    calls: list[list[float]] = []
    _hook(y, lambda g: calls.append(g.tolist()))
    (y * 3).sum().backward()
    assert calls == [[3.0, 3.0]]

    x = lucid.ones(3, device=device, requires_grad=True)
    y = x * 2
    _hook(y, lambda g: g * 10)
    y.sum().backward()
    assert x.grad is not None and x.grad.tolist() == [20.0, 20.0, 20.0]


def test_a_leaf_hook_runs_per_pass_on_that_pass_only(device: str) -> None:
    x = lucid.ones(3, device=device, requires_grad=True)
    calls: list[list[float]] = []

    def double(g: lucid.Tensor) -> lucid.Tensor:
        calls.append(g.tolist())
        return g * 2

    _hook(x, double)
    (x * 3).sum().backward()
    (x * 3).sum().backward()
    assert x.grad is not None and x.grad.tolist() == [12.0, 12.0, 12.0]
    other = lucid.ones(3, device=device, requires_grad=True)
    (other * 5).sum().backward()
    assert calls == [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]


def test_a_leaf_root_without_a_graph(device: str) -> None:
    # backward() on a leaf that never took part in an op: the seed is its
    # gradient, through its hooks.
    x = lucid.ones(3, device=device, requires_grad=True)
    _hook(x, lambda g: g * 4)
    x.backward(lucid.ones(3, device=device))
    assert x.grad is not None and x.grad.tolist() == [4.0, 4.0, 4.0]


def test_only_nodes_on_the_path_run_under_grad(device: str) -> None:
    x = lucid.ones(3, device=device, requires_grad=True)
    y = x * 2
    w = lucid.ones(3, device=device, requires_grad=True)
    calls: list[str] = []
    _hook(y, lambda g: calls.append("y"))
    _hook(x, lambda g: calls.append("x"))
    _hook(w, lambda g: calls.append("w"))
    (g,) = lucid.autograd.grad((y * w).sum(), [y])
    assert g.tolist() == [1.0, 1.0, 1.0]
    assert calls == ["y"]
    assert x.grad is None and w.grad is None


def test_a_hook_that_keeps_its_gradient_keeps_it_unchanged(device: str) -> None:
    x = lucid.ones(3, device=device, requires_grad=True)
    kept: list[lucid.Tensor] = []
    _hook(x, lambda g: kept.append(g))
    (x * 2).sum().backward()
    (x * 2).sum().backward()
    assert [k.tolist() for k in kept] == [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    assert x.grad is not None and x.grad.tolist() == [4.0, 4.0, 4.0]


def test_a_returned_tensor_held_elsewhere_is_not_accumulated_into(device: str) -> None:
    x = lucid.ones(3, device=device, requires_grad=True)
    elsewhere = lucid.full((3,), 5.0, device=device)
    _hook(x, lambda g: elsewhere)
    (x * 2).sum().backward()
    (x * 2).sum().backward()
    assert x.grad is not None and x.grad.tolist() == [10.0, 10.0, 10.0]
    assert elsewhere.tolist() == [5.0, 5.0, 5.0]


def test_an_in_place_write_in_the_hook_is_what_flows_on(device: str) -> None:
    x = lucid.ones(3, device=device, requires_grad=True)
    y = x * 2
    _hook(y, lambda g: g.mul_(3))
    (y * 1.0).sum().backward()
    assert x.grad is not None and x.grad.tolist() == [6.0, 6.0, 6.0]


def test_hooks_run_in_registration_order(device: str) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    y = x * 1.0
    _hook(y, lambda g: g + 1)
    _hook(y, lambda g: g * 10)
    y.sum().backward()
    assert x.grad is not None and x.grad.tolist() == [20.0, 20.0]


def test_the_runner_is_installed_once_per_slot() -> None:
    x = lucid.ones(2, requires_grad=True)
    y = x * 2
    assert not _C_engine._has_tensor_hooks(_unwrap(y))
    first = _C_engine._tensor_hook_runner(_unwrap(y), _Runner)
    second = _C_engine._tensor_hook_runner(_unwrap(y), _Runner)
    assert first is second
    assert _C_engine._has_tensor_hooks(_unwrap(y))
    assert not _C_engine._has_tensor_hooks(_unwrap(x))
    _C_engine._tensor_hook_runner(_unwrap(x), _Runner)
    assert _C_engine._has_tensor_hooks(_unwrap(x))


def test_a_tensor_that_does_not_require_grad_is_refused() -> None:
    with pytest.raises(RuntimeError, match="does not require grad"):
        _C_engine._tensor_hook_runner(_unwrap(lucid.ones(2)), _Runner)


# ── what a hook may return ─────────────────────────────────────────────────


@pytest.mark.parametrize("leaf", [True, False], ids=["leaf", "non-leaf"])
def test_another_dtype_is_refused(device: str, leaf: bool) -> None:
    x = lucid.ones(3, device=device, requires_grad=True)
    y = x if leaf else x * 1.0
    _hook(y, lambda g: g.to(lucid.float16))
    with pytest.raises(_C_engine.DtypeMismatch):
        (y * 2.0).sum().backward()


@pytest.mark.parametrize("leaf", [True, False], ids=["leaf", "non-leaf"])
def test_another_shape_is_refused(device: str, leaf: bool) -> None:
    x = lucid.ones(3, device=device, requires_grad=True)
    y = x if leaf else x * 1.0
    _hook(y, lambda g: g[:2])
    with pytest.raises(_C_engine.ShapeMismatch):
        (y * 2.0).sum().backward()


@_needs_metal
def test_another_device_is_refused() -> None:
    x = lucid.ones(3, device="metal", requires_grad=True)
    y = x * 1.0
    _hook(y, lambda g: g.to("cpu"))
    with pytest.raises(_C_engine.DeviceMismatch):
        (y * 2.0).sum().backward()


def test_another_dtype_is_refused_under_create_graph() -> None:
    x = lucid.ones(3, requires_grad=True)
    y = x * x
    _hook(y, lambda g: g.to(lucid.float16))
    with pytest.raises(_C_engine.DtypeMismatch):
        y.sum().backward(create_graph=True)


def test_a_non_tensor_is_a_type_error() -> None:
    x = lucid.ones(3, requires_grad=True)
    y = x * 1.0
    _hook(y, lambda g: [1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="tensor or None"):
        y.sum().backward()


def test_an_exception_in_a_hook_keeps_its_type() -> None:
    x = lucid.ones(3, requires_grad=True)
    y = x * 1.0

    def boom(g: lucid.Tensor) -> None:
        raise ValueError("boom from the hook")

    _hook(y, boom)
    with pytest.raises(ValueError, match="boom from the hook"):
        y.sum().backward()
    assert lucid.is_grad_enabled()


# ── a hook that runs a backward through its own node ───────────────────────
#
# A hook on y that runs a backward of its own through y's producer, without
# retain_graph, frees what that producer saved.  The engine used to check the
# node before running its hooks and apply it after: the node then read freed
# storage and the process died (SIGSEGV in the CPU mul).  The checks now come
# after the hooks, so the outer pass is refused the way the reference refuses
# it.  A crash cannot fail an in-process test, so each case runs in a child.


def _nested_backward_outcome(
    lib: ModuleType, device: str, hook: Register, outer: str, inner: str
) -> dict[str, object]:
    x = lib.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    y = x * x
    done: list[int] = []

    def run_inner(g: object) -> None:
        if done:
            return
        done.append(1)
        with lib.enable_grad():
            if inner == "backward":
                (y * 1.0).sum().backward()
            else:
                lib.autograd.grad((y * 1.0).sum(), [x])

    hook(y, run_inner)
    loss = (y * 2.0).sum()
    try:
        if outer == "backward":
            loss.backward()
        elif outer == "backward-create-graph":
            loss.backward(create_graph=True)
        elif outer == "grad":
            lib.autograd.grad(loss, [x])
        else:
            lib.autograd.grad(loss, [x], create_graph=True)
    except RuntimeError as e:
        return {"raised": True, "second_pass": "a second time" in str(e).lower()}
    return {"raised": False, "second_pass": False}


_CHILD = """
import json, sys
import lucid
from lucid.test.unit.autograd.test_engine_tensor_hooks import _hook, _nested_backward_outcome
print(json.dumps(_nested_backward_outcome(lucid, *sys.argv[1:2], _hook, *sys.argv[2:4])))
"""

_NESTED = [
    ("backward", "backward"),
    ("backward", "grad"),
    ("backward-create-graph", "backward"),
    ("grad", "backward"),
    ("grad-create-graph", "grad"),
]


@pytest.mark.parametrize(("outer", "inner"), _NESTED, ids=[f"{o}-around-{i}" for o, i in _NESTED])
def test_a_hook_that_frees_its_own_node_is_refused_cleanly(
    outer: str, inner: str, device: str, ref: ModuleType
) -> None:
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, device, outer, inner],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode == 0, f"the process died ({child.returncode}):\n{child.stderr[-2000:]}"
    got = json.loads(child.stdout.strip().splitlines()[-1])
    assert got == {"raised": True, "second_pass": True}
    assert _nested_backward_outcome(ref, "cpu", _ref_hook, outer, inner) == got


# ── what grad() leaves behind ──────────────────────────────────────────────


def test_grad_leaves_nothing_in_a_function_it_did_not_run(device: str) -> None:
    # autograd.grad by one output of a two-output Function captures there and
    # does not run the Function.  It used to hand the Function both outputs'
    # gradients anyway; the Function kept them, having no notion of a pass,
    # and the next backward() added them in again: a.grad came out doubled.
    a = lucid.ones(2, device=device, requires_grad=True)
    o1, o2 = _two_outputs(lucid).apply(a)
    loss = o1.sum() + (o2 * 5).sum() + o2.sum()
    (g2,) = lucid.autograd.grad(loss, o2, retain_graph=True)
    assert g2.tolist() == [6.0, 6.0]
    loss.backward()
    assert a.grad is not None and a.grad.tolist() == [20.0, 20.0]


# ── lifetime ───────────────────────────────────────────────────────────────


def test_a_spent_node_lets_go_of_its_hooks() -> None:
    # tensor -> node -> runner -> closure -> tensor is a cycle Python cannot
    # see through the engine; once backward frees what the node saved, the
    # node can never run again, so its hooks go with it.
    x = lucid.ones(3, requires_grad=True)
    y = x * x
    _hook(y, lambda g: g * y)
    assert _C_engine._has_tensor_hooks(_unwrap(y))
    y.sum().backward()
    assert not _C_engine._has_tensor_hooks(_unwrap(y))


def test_a_tensor_whose_hook_refers_to_it_is_freed_after_backward() -> None:
    def run() -> weakref.ref[lucid.Tensor]:
        x = lucid.ones(3, requires_grad=True)
        y = x * x
        _hook(y, lambda g: g * y)
        y.sum().backward()
        return weakref.ref(y)

    gone = run()
    gc.collect()
    assert gone() is None


def test_a_backward_without_hooks_calls_no_python() -> None:
    # The engine pays a null test per node for hooks; Python is never entered.
    x = lucid.ones(3, requires_grad=True)
    y = x * 2
    y.retain_grad()
    loss = (lucid.cat([y, x]) * 3).sum()
    root = _unwrap(loss)
    events: list[str] = []

    def profile(frame: object, event: str, arg: object) -> None:
        if event == "call":
            events.append(str(frame))

    sys.setprofile(profile)
    try:
        _C_engine.engine_backward([root], [None])
    finally:
        sys.setprofile(None)
    assert events == []
    assert y.grad is not None and y.grad.tolist() == [3.0, 3.0, 3.0]


def test_retain_graph_keeps_the_hooks() -> None:
    x = lucid.ones(3, requires_grad=True)
    y = x * x
    calls: list[int] = []
    _hook(y, lambda g: calls.append(1))
    loss = y.sum()
    loss.backward(retain_graph=True)
    loss.backward()
    assert calls == [1, 1]
