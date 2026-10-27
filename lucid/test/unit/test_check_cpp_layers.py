"""Unit tests for ``tools/check_cpp_layers.py`` (AE-8 / GitHub #314).

Verifies that:
1. The current ``lucid/_C/`` tree passes with 0 new violations and 0 stale
   ``_ALLOWLIST`` entries.
2. Injecting each of the 6 violation classes (upward ``#include``, unauthorized
   ``extern ... _op(``, ``cblas_`` outside ``Blas.cpp``, LAPACK symbol outside
   ``Lapack.cpp``, ``cpu::lapack_`` in ``backend/gpu``, and ``goto``) into a
   tree causes ``check_cpp_layers`` to report the violation and return 1.
3. Canonical owner files (``autograd/GraphOps.h``, ``backend/cpu/Blas.cpp``,
   ``backend/cpu/Lapack.cpp``) and downward ``#include``s are permitted.
4. Resolving a listed violation triggers the stale ``_ALLOWLIST`` ratchet.
"""

import sys
from pathlib import Path

import pytest

_TOOLS_DIR = str(Path(__file__).resolve().parents[3] / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import check_cpp_layers
import check_layers


def _write(engine: Path, rel_path: str, content: str) -> Path:
    target = engine / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def test_real_engine_tree_passes() -> None:
    new_violations, stale, allowed_hits = check_cpp_layers.scan_engine()
    assert new_violations == []
    assert stale == []
    assert len(allowed_hits) == len(check_cpp_layers._ALLOWLIST)
    assert check_cpp_layers.check() == 0
    assert check_layers.check() == 0


@pytest.mark.parametrize(
    ("rel_path", "snippet", "expected_rule", "expected_target"),
    [
        # 1. Upward #include against the C++ layer DAG
        (
            "core/BadInclude.cpp",
            '#include "../ops/bfunc/Add.h"\n',
            "upward-include",
            "../ops/bfunc/Add.h",
        ),
        (
            "backend/cpu/BadInclude.cpp",
            '#include "../../autograd/Engine.h"\n',
            "upward-include",
            "../../autograd/Engine.h",
        ),
        (
            "backend/gpu/CrossBackend.h",
            '#include "../cpu/Lapack.h"\n',
            "upward-include",
            "../cpu/Lapack.h",
        ),
        (
            "optim/CrossPeer.cpp",
            '#include "../ops/bfunc/Add.h"\n',
            "upward-include",
            "../ops/bfunc/Add.h",
        ),
        # 2. Unauthorized extern ... _op( forward declarations
        (
            "autograd/SomeNode.cpp",
            "extern TensorImplPtr mul_op(const TensorImplPtr&, const TensorImplPtr&);\n",
            "extern-op",
            "mul_op",
        ),
        (
            "core/BadForwardDecl.cpp",
            "TensorImplPtr sub_op(const TensorImplPtr& a, const TensorImplPtr& b);\n",
            "extern-op",
            "sub_op",
        ),
        # 3. cblas_ outside backend/cpu/Blas.cpp
        (
            "backend/cpu/CpuBackend.h",
            "void f() { cblas_sgemm(0, 0, 0, 1, 1, 1, 1.0f, nullptr, 1, nullptr, 1, 0.0f, nullptr, 1); }\n",
            "cblas-outside-blas",
            "cblas_sgemm",
        ),
        # 4. LAPACK symbols outside backend/cpu/Lapack.cpp
        (
            "ops/linalg/BadLapack.cpp",
            "void f() { int info = 0; sgetrf_(nullptr, nullptr, nullptr, nullptr, nullptr, &info); }\n",
            "lapack-outside-lapack",
            "sgetrf_",
        ),
        (
            "backend/cpu/CpuBackend.h",
            "using lapack_int = __LAPACK_int;\n",
            "lapack-outside-lapack",
            "__LAPACK_int",
        ),
        # 5. cpu::lapack_ inside backend/gpu/
        (
            "backend/gpu/GpuBackend.h",
            "void f(float* a, int n, int* info) { cpu::lapack_inv_f32(a, n, info); }\n",
            "gpu-cpu-lapack",
            "cpu::lapack_inv_f32",
        ),
        # 6. Rule H13: goto statements
        ("ops/ufunc/BadGoto.cpp", "void f() { goto cleanup; cleanup: return; }\n", "goto", "goto"),
    ],
)
def test_injected_violation_is_reported(
    tmp_path: Path, rel_path: str, snippet: str, expected_rule: str, expected_target: str
) -> None:
    engine = tmp_path / "lucid" / "_C"
    _write(engine, rel_path, snippet)

    new_violations, stale, _ = check_cpp_layers.scan_engine(engine, allowlist={})
    assert stale == []
    assert len(new_violations) == 1
    v = new_violations[0]
    assert v.path == rel_path
    assert v.rule == expected_rule
    assert v.target == expected_target
    assert check_cpp_layers.check(engine, allowlist={}) == 1


def test_canonical_owner_files_and_downward_includes_are_allowed(tmp_path: Path) -> None:
    engine = tmp_path / "lucid" / "_C"
    _write(engine, "api.h", "#pragma once\n")
    _write(engine, "core/TensorImpl.h", '#include "../api.h"\n')
    _write(engine, "tensor/TensorMeta.h", '#include "../core/TensorImpl.h"\n')
    _write(
        engine,
        "backend/cpu/Blas.cpp",
        '#include "../../core/TensorImpl.h"\n'
        "void gemm() { cblas_sgemm(0, 0, 0, 1, 1, 1, 1.0f, nullptr, 1, nullptr, 1, 0.0f, nullptr, 1); }\n",
    )
    _write(
        engine,
        "backend/cpu/Lapack.cpp",
        "using i32 = __LAPACK_int;\n"
        "void lu() { sgetrf_(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr); }\n",
    )
    _write(
        engine,
        "autograd/GraphOps.h",
        '#include "../core/TensorImpl.h"\n'
        "extern TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b);\n",
    )
    _write(
        engine,
        "ops/bfunc/Add.h",
        '#include "../../autograd/GraphOps.h"\n'
        "// cblas_sgemm sgetrf_ goto cpu::lapack_inv_f32 in a comment are ignored\n"
        'const char* doc = "cblas_sgemm sgetrf_ goto cpu::lapack_inv_f32";\n'
        "TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b);\n",
    )
    _write(engine, "nn/Linear.cpp", '#include "../ops/bfunc/Add.h"\n')
    _write(engine, "bindings/bind_ops.cpp", '#include "../nn/Linear.cpp"\n')

    new_violations, stale, _ = check_cpp_layers.scan_engine(engine, allowlist={})
    assert new_violations == []
    assert stale == []
    assert check_cpp_layers.check(engine, allowlist={}) == 0


def test_allowlist_ratchet_detects_stale_and_duplicate_entries(tmp_path: Path) -> None:
    engine = tmp_path / "lucid" / "_C"
    _write(
        engine,
        "autograd/Engine.cpp",
        "extern TensorImplPtr add_op(const TensorImplPtr&, const TensorImplPtr&);\n",
    )
    allowlist = {
        ("autograd/Engine.cpp", "extern-op", "add_op"): ("2026-10-14", "AE-7"),
        ("autograd/Engine.cpp", "upward-include", "../ops/gfunc/Gfunc.h"): ("2026-10-14", "AE-7"),
    }

    # The upward-include entry is no longer present -> stale allow-list failure.
    new_violations, stale, allowed_hits = check_cpp_layers.scan_engine(engine, allowlist=allowlist)
    assert new_violations == []
    assert len(allowed_hits) == 1
    assert stale == [("autograd/Engine.cpp", "upward-include", "../ops/gfunc/Gfunc.h")]
    assert check_cpp_layers.check(engine, allowlist=allowlist) == 1
