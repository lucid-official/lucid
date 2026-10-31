"""A custom ``Function`` builds its constants on the device it runs on.

``lucid.ones(())`` with no ``device=`` is a CPU tensor.  Inside a ``backward``
that is a seed for a vector-Jacobian product, and a CPU seed against a Metal
output is a ``DeviceMismatch`` -- but only on Metal, only when the backward
actually runs, and never on the CPU stream the suite mostly exercises
(LCD-729: ``odeint_event`` backward).  The invariant is that every tensor a
``Function`` creates follows an input's device, so the guard here reads the
source rather than waiting for a device that happens to be absent to run it:
no factory call in a ``Function`` method -- or in a module helper such a
method calls -- may omit ``device=``.
"""

import ast
import pathlib

import pytest

import lucid
import lucid.diffeq as diffeq

#: Factories whose result lands on the default device unless told otherwise.
#: ``*_like`` constructors inherit their argument's device and are fine.
_FACTORIES = frozenset(
    {
        "ones",
        "zeros",
        "full",
        "tensor",
        "arange",
        "eye",
        "empty",
        "linspace",
        "logspace",
        "rand",
        "randn",
        "randint",
        "randperm",
        "as_tensor",
        "scalar_tensor",
        "tril_indices",
        "triu_indices",
    }
)

_ROOT = pathlib.Path(lucid.__file__).parent


def _is_function_subclass(node: ast.ClassDef) -> bool:
    bases = (ast.unparse(b).split("[")[0].split(".")[-1] for b in node.bases)
    return any(b in ("Function", "_Function") for b in bases)


def _factory_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id == "lucid":
            return func.attr if func.attr in _FACTORIES else None
    if isinstance(func, ast.Name) and func.id in _FACTORIES:
        return func.id
    return None


def deviceless_factory_calls(source: str) -> list[tuple[int, str]]:
    """Return ``(line, call)`` for every offending call inside a ``Function``."""
    tree = ast.parse(source)
    helpers = {
        n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    scopes: list[ast.AST] = []
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_function_subclass(node):
            scopes.extend(node.body)
            called.update(
                c.func.id
                for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            )
    scopes.extend(helpers[name] for name in sorted(called & helpers.keys()))

    found: list[tuple[int, str]] = []
    for scope in scopes:
        for node in ast.walk(scope):
            if not isinstance(node, ast.Call) or _factory_name(node) is None:
                continue
            if any(k.arg == "device" or k.arg is None for k in node.keywords):
                continue
            found.append((node.lineno, ast.unparse(node)))
    return found


class TestScannerSees:
    """The guard is only worth having if it can fail."""

    def test_flags_a_seed_in_backward(self) -> None:
        source = (
            "class F(Function):\n"
            "    @staticmethod\n"
            "    def backward(ctx, g):\n"
            "        return vjp(fn, x, lucid.ones((), dtype=g.dtype))\n"
        )
        assert [c for _, c in deviceless_factory_calls(source)] == ["lucid.ones((), dtype=g.dtype)"]

    def test_follows_a_module_helper(self) -> None:
        source = (
            "def seed(g):\n"
            "    return zeros(3)\n"
            "class F(Function):\n"
            "    @staticmethod\n"
            "    def backward(ctx, g):\n"
            "        return seed(g)\n"
        )
        assert [c for _, c in deviceless_factory_calls(source)] == ["zeros(3)"]

    def test_accepts_a_device_and_a_like_constructor(self) -> None:
        source = (
            "class F(Function):\n"
            "    @staticmethod\n"
            "    def backward(ctx, g):\n"
            "        a = lucid.ones((), device=g.device)\n"
            "        return a + lucid.ones_like(g)\n"
        )
        assert deviceless_factory_calls(source) == []

    def test_ignores_factories_outside_a_function(self) -> None:
        assert deviceless_factory_calls("def build():\n    return lucid.ones(3)\n") == []


class TestEveryFunctionFollowsItsInputsDevice:
    def test_no_function_method_builds_a_device_less_constant(self) -> None:
        offenders: list[str] = []
        for path in sorted(_ROOT.rglob("*.py")):
            rel = path.relative_to(_ROOT)
            if rel.parts[0] in ("test", "_C"):
                continue
            for line, call in deviceless_factory_calls(path.read_text()):
                offenders.append(f"{rel}:{line}: {call}")
        assert offenders == [], "device-less constant inside a custom Function:\n" + "\n".join(
            offenders
        )


def _fall(t: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
    return lucid.stack([y[1], -9.8 * lucid.ones_like(y[1])])


class TestEventGradientOnBothDevices:
    """The backward the scan above was written for, run where it crashed."""

    def test_event_backward_runs_on_metal_and_matches_cpu(self, device_gpu_only: str) -> None:
        grads = {}
        for device in ("cpu", device_gpu_only):
            y0 = lucid.tensor([2.0, 3.0], device=device, requires_grad=True)
            event_t, solution = diffeq.odeint_event(
                _fall, y0, lucid.tensor(0.0, device=device), event_fn=lambda t, y: y[0]
            )
            (event_t + (solution[-1] ** 2).sum()).backward()
            assert y0.grad is not None
            assert str(y0.grad.device) == str(y0.device)
            grads[device] = y0.grad.tolist()

        assert grads[device_gpu_only] == pytest.approx(grads["cpu"], rel=1e-4)
        assert grads["cpu"] == pytest.approx([19.744, 6.146], rel=1e-3)

    def test_event_backward_with_a_parameter_matches_across_devices(
        self, device_gpu_only: str
    ) -> None:
        grads = {}
        for device in ("cpu", device_gpu_only):
            k = lucid.tensor([2.0], device=device, requires_grad=True)
            event_t, solution = diffeq.odeint_event(
                lambda t, y: -k * y,
                lucid.tensor([1.0], device=device),
                lucid.tensor(0.0, device=device),
                event_fn=lambda t, y: y[0] - 0.5,
            )
            (event_t + solution[-1].sum()).backward()
            assert k.grad is not None
            grads[device] = float(k.grad.item())

        assert grads[device_gpu_only] == pytest.approx(grads["cpu"], rel=1e-4)
