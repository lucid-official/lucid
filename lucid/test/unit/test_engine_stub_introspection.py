"""Regression tests for ``lucid._C.engine`` stub introspection and 1:1 parity (LCD-220).

Verifies that ``tools/gen_pyi.py`` introspects bound classes, methods,
properties (including setters), sub-namespaces, constants, and functions
directly from ``lucid._C.engine``, and that ``tools/check_stubs.py``'s
:func:`engine_drift` catches both missing runtime bindings and phantom
stub entries.
"""

import ast
from pathlib import Path

from lucid._C import engine as _C_engine

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENGINE_STUB = _REPO_ROOT / "lucid" / "_C" / "engine.pyi"


def test_engine_stub_has_zero_drift_against_runtime() -> None:
    """Committed ``lucid/_C/engine.pyi`` has 1:1 parity with ``lucid._C.engine``."""
    from tools.check_stubs import engine_drift

    stub_text = _ENGINE_STUB.read_text(encoding="utf-8")
    assert engine_drift(stub_text, _C_engine) == []


def test_generator_next_uint32x4_and_counter_setter_stubbed() -> None:
    """``Generator.next_uint32x4`` and ``Generator.counter`` setter are emitted."""
    stub_text = _ENGINE_STUB.read_text(encoding="utf-8")
    tree = ast.parse(stub_text)
    gen_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Generator")
    fn_defs = [n for n in gen_cls.body if isinstance(n, ast.FunctionDef)]
    names = {n.name for n in fn_defs}
    assert "next_uint32x4" in names
    assert "next_uniform_float" in names
    assert "set_seed" in names

    setters = {
        n.name for n in fn_defs if any(ast.unparse(d) == "counter.setter" for d in n.decorator_list)
    }
    assert "counter" in setters


def test_all_runtime_classes_present_and_phantoms_absent() -> None:
    """Every bound class in ``lucid._C.engine`` is stubbed and legacy phantoms are gone."""
    from tools.check_stubs import _collect_runtime_classes

    stub_text = _ENGINE_STUB.read_text(encoding="utf-8")
    tree = ast.parse(stub_text)
    stub_classes = {
        n.name: n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and not n.name.endswith("_namespace")
    }
    runtime_classes = _collect_runtime_classes(_C_engine)
    assert set(stub_classes) == set(runtime_classes)

    assert "_EngineOptimizer" not in stub_classes
    assert "Optimizer" in stub_classes
    assert "LRScheduler" in stub_classes
    assert "NoGradGuard" in stub_classes

    profiler_members = {
        n.name for n in stub_classes["Profiler"].body if isinstance(n, ast.FunctionDef)
    }
    assert "is_active" in profiler_members
    assert "reset" not in profiler_members

    tensor_members = {
        n.name for n in stub_classes["TensorImpl"].body if isinstance(n, ast.FunctionDef)
    }
    assert "_make_view" in tensor_members
    assert "bump_version" not in tensor_members


def test_engine_drift_catches_missing_and_phantom_members() -> None:
    """``engine_drift`` reports missing methods, phantom methods, and parameter drift."""
    from tools.check_stubs import engine_drift

    stub_text = _ENGINE_STUB.read_text(encoding="utf-8")

    # 1. Remove Generator.next_uint32x4 from stub -> reported as missing in stub
    missing_method_stub = stub_text.replace(
        "    def next_uint32x4(self) -> tuple[int, int, int, int]: ...\n", ""
    )
    assert missing_method_stub != stub_text
    assert engine_drift(missing_method_stub, _C_engine) == [
        "Generator.next_uint32x4: in the engine, not in the stub"
    ]

    # 2. Add phantom Profiler.reset to stub -> reported as phantom in stub
    phantom_method_stub = stub_text.replace(
        "class Profiler:\n",
        "class Profiler:\n    def reset(self) -> None: ...\n",
    )
    assert engine_drift(phantom_method_stub, _C_engine) == [
        "Profiler.reset: in the stub, not in the engine"
    ]

    # 3. Remove Generator.counter.setter -> reported as missing setter
    missing_setter_stub = stub_text.replace(
        "    @counter.setter\n    def counter(self, value: SupportsInt) -> None: ...\n",
        "",
    )
    assert missing_setter_stub != stub_text
    assert engine_drift(missing_setter_stub, _C_engine) == [
        "Generator.counter.setter: in the engine, not in the stub"
    ]

    # 4. Rename parameter on Generator.set_seed -> reported as parameter mismatch
    renamed_param_stub = stub_text.replace(
        "def set_seed(self, seed: SupportsInt) -> None: ...",
        "def set_seed(self, wrong_name: SupportsInt) -> None: ...",
    )
    assert renamed_param_stub != stub_text
    assert engine_drift(renamed_param_stub, _C_engine) == [
        "Generator.set_seed: stub (wrong_name), engine (seed)"
    ]

    # 5. Add phantom top-level class and function -> both reported
    phantom_top_stub = stub_text + "\nclass _EngineOptimizer: ...\ndef no_grad() -> object: ...\n"
    assert engine_drift(phantom_top_stub, _C_engine) == [
        "_EngineOptimizer: in the stub, not in the engine",
        "no_grad: in the stub, not in the engine",
    ]
