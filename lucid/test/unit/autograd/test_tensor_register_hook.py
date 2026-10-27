"""``Tensor.register_hook`` runs inside backward, where the gradient is complete.

The public method used to put hooks in a Python registry keyed by ``id()``
and walk it after ``backward()`` returned, reading each tensor's ``.grad``:
a non-leaf's hook never ran (it has no ``.grad``), a hook's return value
reached ``.grad`` but nothing upstream, a leaf's hook ran again on every
later, unrelated backward with the accumulated gradient, and
``autograd.grad`` ran none of them (LCD-151).

The method now installs one runner per gradient slot on the engine
(``_C_engine._tensor_hook_runner``), which runs it where the reference runs
a tensor's hooks.  The engine's half is pinned by
``test_engine_tensor_hooks.py`` with a runner of its own; this file pins
the public method over it: every kind of tensor a hook can sit on × every
answer a hook can give × both devices × both graph modes, against the
reference, plus the handle, the order with ``retain_grad`` and the checks
on what a hook returns.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._helpers.compare import assert_close

# ── the matrix: {tensor kind} × {hook answer} × {device} × {create_graph} ────

_X = [1.0, -2.0, 0.5]


def _leaf(lib: ModuleType, x: object) -> tuple[object, Callable[[object], object]]:
    return x, lambda t: (t * t).sum() * 1.5


def _non_leaf(lib: ModuleType, x: object) -> tuple[object, Callable[[object], object]]:
    t = x * 2
    return t, lambda t: (t * t).sum()


def _view(lib: ModuleType, x: object) -> tuple[object, Callable[[object], object]]:
    t = (x * 2).view(3, 1)
    return t, lambda t: (t * t * t).sum()


def _multi_use(lib: ModuleType, x: object) -> tuple[object, Callable[[object], object]]:
    # One consumer made before the hook, two after: every one of them sends
    # its gradient through the hook.
    t = x * 2
    early = t * t
    return t, lambda t: early.sum() + (t * 3).sum() + t.sin().sum()


_KINDS = {"leaf": _leaf, "non-leaf": _non_leaf, "view": _view, "multi-use": _multi_use}
_ANSWERS = ["none", "tensor", "removed"]


def _scenario(
    lib: ModuleType, device: str, kind: str, answer: str, create_graph: bool
) -> dict[str, object]:
    x = lib.tensor(_X, device=device, requires_grad=True)
    t, loss_of = _KINDS[kind](lib, x)
    seen: list[object] = []

    def hook(g: object) -> object:
        seen.append(g.detach().clone())
        return g * 2 if answer == "tensor" else None

    handle = t.register_hook(hook)
    if answer == "removed":
        handle.remove()
    loss = loss_of(t)
    loss.backward(create_graph=create_graph)
    out: dict[str, object] = {"x.grad": x.grad.detach().clone()}
    if create_graph:
        # The gradient carries a graph — through the hook's result, when the
        # hook returned one — so it can be differentiated once more.
        (second,) = lib.autograd.grad(x.grad.sum(), x)
        out["second"] = second.detach()
    out["calls"] = len(seen)
    out["seen"] = seen
    return out


def _compare(got: dict[str, object], want: dict[str, object]) -> None:
    assert got.keys() == want.keys()
    for key, w in want.items():
        g = got[key]
        if isinstance(w, int):
            assert g == w, key
        elif isinstance(w, list):
            assert isinstance(g, list) and len(g) == len(w), key
            for a, b in zip(g, w, strict=True):
                assert_close(a, b, msg=key)
        else:
            assert_close(g, w, msg=key)


@pytest.mark.parity
@pytest.mark.parametrize("create_graph", [False, True], ids=["eager", "create-graph"])
@pytest.mark.parametrize("answer", _ANSWERS)
@pytest.mark.parametrize("kind", list(_KINDS))
def test_matches_the_reference(
    kind: str, answer: str, create_graph: bool, device: str, ref: ModuleType
) -> None:
    _compare(
        _scenario(lucid, device, kind, answer, create_graph),
        _scenario(ref, "cpu", kind, answer, create_graph),
    )


# ── the reported cases, without the reference ──────────────────────────────


def test_a_non_leaf_hook_runs_with_the_complete_gradient(device: str) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    y = x * 2
    calls: list[list[float]] = []
    y.register_hook(lambda g: calls.append(g.tolist()))
    (y * 3).sum().backward()
    assert calls == [[3.0, 3.0]]


def test_a_returned_tensor_flows_upstream(device: str) -> None:
    x = lucid.ones(3, device=device, requires_grad=True)
    y = x * 2
    y.register_hook(lambda g: g * 10)
    y.sum().backward()
    assert x.grad is not None and x.grad.tolist() == [20.0, 20.0, 20.0]


def test_a_leaf_hook_does_not_run_on_an_unrelated_backward(device: str) -> None:
    w = lucid.ones(3, device=device, requires_grad=True)
    seen: list[list[float]] = []
    w.register_hook(lambda g: seen.append(g.tolist()))
    (w * 2).sum().backward()
    other = lucid.ones(3, device=device, requires_grad=True)
    (other * 3).sum().backward()
    assert seen == [[2.0, 2.0, 2.0]]
    assert w.grad is not None and w.grad.tolist() == [2.0, 2.0, 2.0]


def test_a_leaf_hook_sees_each_pass_alone(device: str) -> None:
    # Not the accumulated ``.grad``: the second pass's hook sees 2, not 4,
    # and what it returns is what is added.
    w = lucid.ones(3, device=device, requires_grad=True)
    seen: list[list[float]] = []

    def hook(g: lucid.Tensor) -> lucid.Tensor:
        seen.append(g.tolist())
        return g * 3

    w.register_hook(hook)
    (w * 2).sum().backward()
    (w * 2).sum().backward()
    assert seen == [[2.0] * 3, [2.0] * 3]
    assert w.grad is not None and w.grad.tolist() == [12.0] * 3


def test_autograd_grad_runs_the_hooks_on_its_path(device: str) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    y = x * 2
    seen: list[list[float]] = []

    def hook(g: lucid.Tensor) -> lucid.Tensor:
        seen.append(g.tolist())
        return g * 5

    y.register_hook(hook)
    (gx,) = lucid.autograd.grad((y * 3).sum(), x)
    assert seen == [[3.0, 3.0]]
    assert gx is not None and gx.tolist() == [30.0, 30.0]
    assert x.grad is None


# ── the handle ──────────────────────────────────────────────────────────────


def test_the_handle_removes_the_hook(device: str) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    calls: list[int] = []
    handle = x.register_hook(lambda g: calls.append(1))
    handle.remove()
    handle.remove()
    (x * 2).sum().backward()
    assert calls == []


def test_the_handle_as_a_context_manager(device: str) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    with x.register_hook(lambda g: g * 10):
        (x * 2).sum().backward()
    (x * 2).sum().backward()
    assert x.grad is not None and x.grad.tolist() == [22.0, 22.0]


def test_removing_twice_removes_one_registration_of_a_repeated_hook() -> None:
    x = lucid.ones(2, requires_grad=True)
    calls: list[int] = []

    def hook(g: lucid.Tensor) -> None:
        calls.append(1)

    first = x.register_hook(hook)
    x.register_hook(hook)
    first.remove()
    first.remove()
    (x * 2).sum().backward()
    assert calls == [1]


def test_hooks_run_in_registration_order_each_on_the_last_result(device: str) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    y = x * 1.0
    seen: list[tuple[str, list[float]]] = []

    def first(g: lucid.Tensor) -> lucid.Tensor:
        seen.append(("first", g.tolist()))
        return g + 1

    def second(g: lucid.Tensor) -> None:
        seen.append(("second", g.tolist()))

    def third(g: lucid.Tensor) -> lucid.Tensor:
        seen.append(("third", g.tolist()))
        return g * 10

    for hook in (first, second, third):
        y.register_hook(hook)
    y.sum().backward()
    assert seen == [("first", [1.0, 1.0]), ("second", [2.0, 2.0]), ("third", [2.0, 2.0])]
    assert x.grad is not None and x.grad.tolist() == [20.0, 20.0]


def test_a_hook_that_registers_another_does_not_run_it_this_pass() -> None:
    x = lucid.ones(2, requires_grad=True)
    calls: list[str] = []

    def late(g: lucid.Tensor) -> None:
        calls.append("late")

    def outer(g: lucid.Tensor) -> None:
        calls.append("outer")
        x.register_hook(late)

    x.register_hook(outer)
    (x * 2).sum().backward()
    assert calls == ["outer"]


# ── retain_grad ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "retain_first", [True, False], ids=["retain-then-hook", "hook-then-retain"]
)
def test_retain_grad_keeps_the_hooked_gradient(retain_first: bool, device: str) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    y = x * 2
    if retain_first:
        y.retain_grad()
    y.register_hook(lambda g: g * 7)
    if not retain_first:
        y.retain_grad()
    (y * 3).sum().backward()
    assert y.grad is not None and y.grad.tolist() == [21.0, 21.0]
    assert x.grad is not None and x.grad.tolist() == [42.0, 42.0]


def test_retain_grad_on_a_tensor_that_does_not_require_grad_is_refused() -> None:
    with pytest.raises(RuntimeError, match="does not require grad"):
        lucid.ones(2).retain_grad()


# ── what a hook may return ──────────────────────────────────────────────────


def test_register_hook_on_a_tensor_that_does_not_require_grad_is_refused() -> None:
    with pytest.raises(RuntimeError, match="does not require grad"):
        lucid.ones(2).register_hook(lambda g: None)


@pytest.mark.parametrize("leaf", [True, False], ids=["leaf", "non-leaf"])
@pytest.mark.parametrize("value", [3, 1.5, [1.0, 1.0]], ids=["int", "float", "list"])
def test_a_non_tensor_result_is_a_type_error_naming_the_hook(
    value: object, leaf: bool, device: str
) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    t = x if leaf else x * 1.0

    def bad_hook(g: lucid.Tensor) -> object:
        return value

    t.register_hook(bad_hook)
    with pytest.raises(TypeError, match="bad_hook"):
        (t * 2).sum().backward()
    assert x.grad is None


@pytest.mark.parametrize(
    ("make", "error"),
    [
        (lambda g: g[:1], _C_engine.ShapeMismatch),
        (lambda g: g.to(lucid.float16), _C_engine.DtypeMismatch),
    ],
    ids=["shape", "dtype"],
)
def test_a_result_of_another_kind_is_refused(
    make: Callable[[lucid.Tensor], lucid.Tensor], error: type[BaseException], device: str
) -> None:
    x = lucid.ones(2, device=device, requires_grad=True)
    y = x * 1.0
    y.register_hook(make)
    with pytest.raises(error):
        (y * 2).sum().backward()
    assert x.grad is None


def test_the_hooks_registry_is_gone() -> None:
    import lucid.autograd._hooks as hooks

    assert not hasattr(hooks, "_TENSOR_HOOKS")
    assert not hasattr(hooks, "_dispatch_tensor_grad_hooks")
