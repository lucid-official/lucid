"""A hooked module's gradients are summed, slot by slot, like any other (CHA-45).

A module with a full backward hook or pre-hook is wrapped in two barrier
nodes: one where gradients arrive at its outputs, one where they leave
through its inputs.  A slot of either barrier can be reached more than once
— an output read by two consumers, an input the body uses twice — and the
reference sums those arrivals before any hook sees them.  The barriers used
to keep only the last one, so the hooks were shown part of the gradient and
every input and parameter behind the module got the wrong one, silently.

Each case runs the same code against Lucid (CPU and Metal) and against the
reference, and compares the gradients *and* what each hook was shown.
Also pinned here: the barrier state belongs to one backward pass — a second
pass over a retained graph fires the hooks again and starts from nothing,
even after an ``autograd.grad`` that delivered to a barrier and skipped it,
and a pass opened inside a hook does not disturb the one around it — and
``create_graph`` through a hooked module, which had no graph-mode path.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

# ── harness ────────────────────────────────────────────────────────────────


def _plain(value: object) -> object:
    """A gradient, a tuple of them, or ``None``, as nested lists."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value.detach().tolist()  # type: ignore[attr-defined]


def _assert_close(got: object, want: object, tol: float = 1e-5) -> None:
    if want is None or got is None:
        assert got is None and want is None, (got, want)
        return
    if isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), (got, want)
        for g, w in zip(got, want, strict=True):
            _assert_close(g, w, tol)
        return
    if isinstance(want, str):
        assert got == want
        return
    assert isinstance(got, float) and isinstance(want, float), (got, want)
    assert abs(got - want) <= tol * max(1.0, abs(want)), (got, want)


def _recorder(log: list[object]) -> Callable[..., None]:
    def hook(module: object, grad_input: object, grad_output: object) -> None:
        log.append(["full", _plain(grad_input), _plain(grad_output)])

    return hook


def _dtype(lib: ModuleType, name: str) -> object:
    return getattr(lib, name)


def _run(
    scenario: Callable[..., dict[str, object]],
    ref: ModuleType,
    device: str,
    dtype: str = "float32",
) -> None:
    if device == "metal" and dtype == "float64":
        pytest.skip("Metal holds no float64")
    got = scenario(lucid, device, dtype)
    want = scenario(ref, "cpu", dtype)
    assert got.keys() == want.keys()
    for key in want:
        _assert_close(got[key], want[key])


def _identity_linear(lib: ModuleType, device: str, dtype: str) -> object:
    lin = lib.nn.Linear(2, 2, bias=False)
    with lib.no_grad():
        lin.weight.copy_(lib.tensor([[1.0, 0.5], [-1.0, 2.0]]))
    return lin.to(device=device, dtype=_dtype(lib, dtype))


def _input(lib: ModuleType, values: list[object], device: str, dtype: str) -> object:
    return lib.tensor(values, dtype=_dtype(lib, dtype), device=device, requires_grad=True)


# ── scenarios ──────────────────────────────────────────────────────────────


def _input_used_twice(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    class Twice(lib.nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, x: object) -> object:
            return x + 2 * x

    log: list[object] = []
    m = Twice()
    m.register_full_backward_hook(_recorder(log))
    x = _input(lib, [1.0, -2.0, 0.5], device, dtype)
    (m(x) * lib.tensor([1.0, 2.0, 3.0], dtype=_dtype(lib, dtype), device=device)).sum().backward()
    return {"x": _plain(x.grad), "hooks": log}


def _output_read_twice(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    log: list[object] = []
    lin = _identity_linear(lib, device, dtype)
    lin.register_full_backward_hook(_recorder(log))
    x = _input(lib, [[1.0, 2.0]], device, dtype)
    y = lin(x)
    (y.sum() + (2 * y).sum()).backward()
    return {"x": _plain(x.grad), "w": _plain(lin.weight.grad), "hooks": log}


def _residual(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    log: list[object] = []
    lin = _identity_linear(lib, device, dtype)
    lin.register_full_backward_hook(_recorder(log))
    x = _input(lib, [[1.0, 2.0]], device, dtype)
    h = lin(x)
    y = x + h
    (y * y + h).sum().backward()
    return {"x": _plain(x.grad), "w": _plain(lin.weight.grad), "hooks": log}


def _hook_replaces_the_sum(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    class Twice(lib.nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, x: object) -> object:
            return x * x + 2 * x

    log: list[object] = []

    def halve(module: object, grad_input: tuple[object, ...], grad_output: object) -> object:
        log.append(["full", _plain(grad_input), _plain(grad_output)])
        return (grad_input[0] * 0.5,)  # type: ignore[operator]

    m = Twice()
    m.register_full_backward_hook(halve)
    x = _input(lib, [1.0, -2.0, 0.5], device, dtype)
    y = m(x)
    (y.sum() + (3 * y).sum()).backward()
    return {"x": _plain(x.grad), "hooks": log}


def _pre_hook_sees_the_sum(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    class Twice(lib.nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, x: object) -> object:
            return x + 2 * x

    log: list[object] = []

    def scale(module: object, grad_output: tuple[object, ...]) -> object:
        log.append(["pre", _plain(grad_output)])
        return (grad_output[0] * 10,)  # type: ignore[operator]

    m = Twice()
    m.register_full_backward_pre_hook(scale)
    m.register_full_backward_hook(_recorder(log))
    x = _input(lib, [1.0, 2.0, 3.0], device, dtype)
    y = m(x)
    (y.sum() + (2 * y).sum()).backward()
    return {"x": _plain(x.grad), "hooks": log}


def _two_inputs_two_outputs(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    class Pair(lib.nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, a: object, b: object) -> object:
            return a * b + a, b * 2

    log: list[object] = []
    m = Pair()
    m.register_full_backward_hook(_recorder(log))
    a = _input(lib, [2.0, -1.0], device, dtype)
    b = _input(lib, [3.0, 0.5], device, dtype)
    o1, o2 = m(a, b)
    (o1.sum() + 3 * o1.sum() + 5 * o2.sum() + (o2 * o2).sum()).backward()
    return {"a": _plain(a.grad), "b": _plain(b.grad), "hooks": log}


def _create_graph_one_slot(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    class Twice(lib.nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, x: object) -> object:
            return x + 2 * x

    log: list[object] = []
    m = Twice()
    m.register_full_backward_hook(_recorder(log))
    x = _input(lib, [1.0, -2.0, 0.5], device, dtype)
    (g,) = lib.autograd.grad((m(x) ** 3).sum(), x, create_graph=True)
    (h,) = lib.autograd.grad(g.sum(), x)
    return {"g": _plain(g), "h": _plain(h), "hooks": log}


def _create_graph_two_slots(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    # Two outputs of different shapes: graph mode used to add every arrival
    # at a node into one tensor, whatever its slot.
    class Pair(lib.nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, a: object, b: object) -> object:
            return a * a, b * b * b

    log: list[object] = []
    m = Pair()
    m.register_full_backward_hook(_recorder(log))
    a = _input(lib, [2.0, -1.0], device, dtype)
    b = _input(lib, [3.0, 0.5, -1.5], device, dtype)
    o1, o2 = m(a, b)
    # Every output gradient depends on the inputs, so the second pass reaches
    # both output slots again.
    ga, gb = lib.autograd.grad(
        (o1 * o1).sum() + o1.sum() + (o2 * o2).sum(), (a, b), create_graph=True
    )
    (ga.sum() + (gb * gb).sum()).backward()
    return {
        "ga": _plain(ga),
        "gb": _plain(gb),
        "a": _plain(a.grad),
        "b": _plain(b.grad),
        "hooks": log,
    }


def _backward_create_graph(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    log: list[object] = []
    lin = _identity_linear(lib, device, dtype)
    lin.register_full_backward_hook(_recorder(log))
    x = _input(lib, [[1.0, 2.0]], device, dtype)
    y = lin(x)
    ((y * y).sum() + y.sum()).backward(create_graph=True)
    (h,) = lib.autograd.grad(x.grad.sum(), x)
    return {"x": _plain(x.grad), "h": _plain(h), "hooks": log}


def _retained_graph_twice(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    log: list[object] = []
    lin = _identity_linear(lib, device, dtype)
    lin.register_full_backward_hook(_recorder(log))
    x = _input(lib, [[1.0, 2.0]], device, dtype)
    y = lin(x)
    loss = y.sum() + (2 * y).sum()
    loss.backward(retain_graph=True)
    loss.backward()
    return {"x": _plain(x.grad), "w": _plain(lin.weight.grad), "hooks": log}


def _grad_skips_a_barrier(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    # autograd.grad by b hands the output barrier a gradient and then skips
    # it: nothing upstream of it was asked for.  The next pass must not add
    # onto what that one left.
    log: list[object] = []
    lin = _identity_linear(lib, device, dtype)
    lin.register_full_backward_hook(_recorder(log))
    x = _input(lib, [[1.0, 2.0]], device, dtype)
    b = _input(lib, [[0.5, -0.5]], device, dtype)
    out = lin(x) + b
    loss = (out * out).sum()
    (gb,) = lib.autograd.grad(loss, b, retain_graph=True)
    loss.backward()
    return {"gb": _plain(gb), "x": _plain(x.grad), "hooks": log}


def _pass_inside_a_hook(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    class Twice(lib.nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, x: object) -> object:
            return x * x + x

    log: list[object] = []

    def nested(module: object, grad_output: tuple[object, ...]) -> None:
        # The reference runs hooks with grad mode off.
        with lib.enable_grad():
            z = _input(lib, [1.0, 3.0], device, dtype)
            (gz,) = lib.autograd.grad((z * z).sum(), z)
        log.append(["nested", _plain(gz), _plain(grad_output)])

    m = Twice()
    m.register_full_backward_pre_hook(nested)
    m.register_full_backward_hook(_recorder(log))
    x = _input(lib, [1.0, -2.0], device, dtype)
    y = m(x)
    (y.sum() + (2 * y).sum()).backward()
    return {"x": _plain(x.grad), "hooks": log}


_SCENARIOS = {
    "input-used-twice": _input_used_twice,
    "output-read-twice": _output_read_twice,
    "residual": _residual,
    "hook-replaces-the-sum": _hook_replaces_the_sum,
    "pre-hook-sees-the-sum": _pre_hook_sees_the_sum,
    "two-inputs-two-outputs": _two_inputs_two_outputs,
    "create-graph-one-slot": _create_graph_one_slot,
    "create-graph-two-slots": _create_graph_two_slots,
    "backward-create-graph": _backward_create_graph,
    "retained-graph-twice": _retained_graph_twice,
    "grad-skips-a-barrier": _grad_skips_a_barrier,
    "pass-inside-a-hook": _pass_inside_a_hook,
}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, ref: ModuleType, device: str) -> None:
    _run(_SCENARIOS[name], ref, device)


@pytest.mark.parametrize(
    "name", ["input-used-twice", "output-read-twice", "create-graph-two-slots"]
)
def test_matches_the_reference_in_float64(name: str, ref: ModuleType) -> None:
    _run(_SCENARIOS[name], ref, "cpu", "float64")


# ── without the reference ──────────────────────────────────────────────────


def test_the_reported_case(device: str) -> None:
    class Twice(lucid.nn.Module):
        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return x + 2 * x

    m = Twice()
    m.register_full_backward_hook(lambda mod, gi, go: None)
    x = lucid.ones(3, requires_grad=True, device=device)
    m(x).sum().backward()
    assert x.grad is not None
    assert x.grad.tolist() == [3.0, 3.0, 3.0]


def test_hooks_fire_once_per_pass(device: str) -> None:
    calls: list[int] = []
    lin = lucid.nn.Linear(2, 2).to(device)
    lin.register_full_backward_pre_hook(lambda mod, go: calls.append(0))
    lin.register_full_backward_hook(lambda mod, gi, go: calls.append(1))
    x = lucid.ones(1, 2, requires_grad=True, device=device)
    y = lin(x)
    loss = y.sum() + (2 * y).sum()
    for _ in range(3):
        loss.backward(retain_graph=True)
    assert calls == [0, 1] * 3
