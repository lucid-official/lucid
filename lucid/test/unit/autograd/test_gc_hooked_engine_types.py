"""Every engine type with a garbage-collector hook survives a collection.

LCD-392 gave six engine types ``tp_traverse`` / ``tp_clear`` so that a cycle
through the autograd graph can be freed.  The hooks found the C++ object
behind a Python one with a pybind11 lookup that answers "not found" for an
instance of a registered *subclass* — ``AccumulateGrad``,
``_PythonBackwardNode`` — and then read through the empty answer.  Any
collection that reached such a node was a segfault: it killed the CI audit
sweep inside ``linalg.qr``'s custom function, and ``x.grad_fn`` on a leaf
followed by ``gc.collect()`` killed a plain script the same way.

Each case runs in its own interpreter, so a crash fails one test rather than
the whole session.  The table of cases is checked against every GC-flagged
pybind11 type the engine exposes, so a type that gains a hook without a case
here fails as well.
"""

import subprocess
import sys
import textwrap
import types

import pytest

from lucid._C import engine as _C_engine

_HAVE_GC = 1 << 14  # Py_TPFLAGS_HAVE_GC

_PRELUDE = """
import gc
import weakref
import lucid
import lucid.nn as nn
from lucid._C import engine as E
"""

# How to reach a live instance of each hooked type, binding it to ``obj``.
_INSTANCES = {
    "TensorImpl": "obj = lucid.randn(3)._impl",
    "Node": "x = lucid.randn(3, requires_grad=True)\nobj = (x * 2).grad_fn",
    "AccumulateGrad": ("x = lucid.randn(3, requires_grad=True)\ny = x * 2\nobj = x._impl.grad_fn"),
    "_PythonBackwardNode": "obj = E._PythonBackwardNode()\nobj.ctx = E.FunctionCtx()",
    "FunctionCtx": "obj = E.FunctionCtx()\nobj.payload = [1, 2]",
    "_ModuleBackwardHookState": ("obj = E._create_module_backward_hook_state(1, None, None)"),
}

# A Python subclass resolves to its pybind11 base; a C++ subclass does not.
_SUBCLASSES = {
    "_PythonBackwardNode": "class Sub(E._PythonBackwardNode):\n    pass\n"
    "obj = Sub()\nobj.ctx = E.FunctionCtx()",
    "FunctionCtx": "class Sub(E.FunctionCtx):\n    pass\nobj = Sub()",
}

_EXERCISE = """
assert type(obj).__name__ in ({name!r}, "Sub"), type(obj)
gc.get_referents(obj)
gc.get_referrers(obj)
gc.collect()
gc.collect()
print("survived")
"""


def _run(body: str) -> subprocess.CompletedProcess[str]:
    code = _PRELUDE + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _assert_survived(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0 and "survived" in proc.stdout, (
        f"exit {proc.returncode} (negative: killed by that signal)\n" f"{proc.stderr[-3000:]}"
    )


def _hooked_types() -> set[str]:
    """Every pybind11 type under the engine that the collector traverses."""
    found: set[str] = set()
    seen: set[int] = set()

    def walk(module: types.ModuleType) -> None:
        for value in vars(module).values():
            if isinstance(value, types.ModuleType):
                if value.__name__.startswith("lucid._C") and id(value) not in seen:
                    seen.add(id(value))
                    walk(value)
            elif (
                isinstance(value, type)
                and value.__flags__ & _HAVE_GC
                and any(base.__name__ == "pybind11_object" for base in value.__mro__)
            ):
                found.add(value.__name__)

    walk(_C_engine)
    return found


def test_every_hooked_engine_type_has_a_case() -> None:
    assert _hooked_types() == set(_INSTANCES)


@pytest.mark.parametrize("name", sorted(_INSTANCES))
def test_a_live_instance_survives_traversal_and_collection(name: str) -> None:
    _assert_survived(_run(_INSTANCES[name] + _EXERCISE.format(name=name)))


@pytest.mark.parametrize("name", sorted(_SUBCLASSES))
def test_a_python_subclass_instance_survives_traversal_and_collection(
    name: str,
) -> None:
    _assert_survived(_run(_SUBCLASSES[name] + _EXERCISE.format(name=name)))


def test_a_hooked_cycle_is_still_freed_while_subclass_nodes_are_alive() -> None:
    # LCD-392's own case — a module holding its hooked output — collected
    # while an AccumulateGrad and a _PythonBackwardNode are reachable.  Here
    # after backward; before it the cycle is collected too, now that a node
    # shows the collector the tensors it keeps (see
    # test_module_backward_hook_lifetime).
    body = """
    module = nn.Linear(3, 3)
    module.register_full_backward_hook(lambda mod, gi, go: None)
    x = lucid.randn(2, 3, requires_grad=True)
    y = module(x)
    module.last = y
    y.sum().backward()
    leaf_node = x._impl.grad_fn
    python_node = E._PythonBackwardNode()
    assert type(leaf_node).__name__ == "AccumulateGrad"
    module_ref = weakref.ref(module)
    output_ref = weakref.ref(y)
    del module, y
    gc.collect()
    assert module_ref() is None, "the hooked cycle outlived its last reference"
    assert output_ref() is None
    print("survived")
    """
    _assert_survived(_run(body))


def test_a_collection_inside_function_apply_survives() -> None:
    # The crash in CI: a collection the interpreter started on its own while
    # ``Function.apply`` (``linalg.qr``'s, there) held its fresh
    # _PythonBackwardNode.  A profile hook pins that moment — a full
    # collection on entry to every call ``_register`` makes once the node
    # exists — so it no longer depends on where the allocation count stands.
    body = """
    import sys

    class Double(lucid.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * 2

        @staticmethod
        def backward(ctx, grad):
            return grad * 2

    fired = []

    def collect_inside_register(frame, event, arg):
        caller = frame.f_back
        if event == "call" and caller is not None and caller.f_code.co_name == "_register":
            if "node" in caller.f_locals:
                fired.append(frame.f_code.co_name)
                gc.collect()

    module = nn.Linear(3, 3)
    module.register_full_backward_hook(lambda mod, gi, go: None)
    x = lucid.randn(2, 3, requires_grad=True)
    sys.setprofile(collect_inside_register)
    y = Double.apply(module(x))
    q, r = lucid.linalg.qr(lucid.randn(3, 3, requires_grad=True))
    sys.setprofile(None)
    assert fired, "no collection ran while a node was alive — the case is vacuous"
    (y.sum() + q.sum() + r.sum()).backward()
    print("survived")
    """
    _assert_survived(_run(body))
