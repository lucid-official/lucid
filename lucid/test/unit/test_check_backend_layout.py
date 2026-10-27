"""Unit tests for ``tools/check_backend_layout.py`` (LCD-264-QG).

Verifies that:
1. The real repository tree passes ``check_backend_layout`` once ``methods/*.cpp``
   are tracked (or when checked against the expected git-tracked set).
2. Adding an unapproved inline method body or a >10-line ``KEEP_INLINE`` body in
   ``CpuBackend.h`` / ``GpuBackend.h`` triggers ``inline-body`` / ``inline-body-too-long``.
3. Adding an untracked ``.cpp`` file under ``methods/`` triggers ``untracked-methods-cpp``.
4. Adding a namespace-scope free function or an extra ``#include`` inside
   ``methods/<Family>.cpp`` triggers ``non-member-definition`` / ``methods-preamble``.
"""

from pathlib import Path
import shutil

from tools import check_backend_layout

_CLEAN_CPU_HDR = """\
#pragma once
namespace lucid {
namespace backend {
class CpuBackend final : public IBackend {
public:
    Device device() const noexcept override { return Device::CPU; }
    Storage add(const Storage& a, const Storage& b, const Shape& s, Dtype dt) override;
};
}  // namespace backend
}  // namespace lucid
"""

_CLEAN_GPU_HDR = """\
#pragma once
namespace lucid {
namespace backend {
class GpuBackend final : public IBackend {
public:
    Device device() const noexcept override { return Device::GPU; }
    Storage add(const Storage& a, const Storage& b, const Shape& s, Dtype dt) override;
};
}  // namespace backend
}  // namespace lucid
"""

_CLEAN_CPU_BINARY = """\
#include "../CpuBackend.h"
namespace lucid {
namespace backend {
Storage CpuBackend::add(const Storage& a, const Storage& b, const Shape& s, Dtype dt) {
    (void)b;
    (void)s;
    (void)dt;
    return a;
}
}  // namespace backend
}  // namespace lucid
"""

_CLEAN_GPU_BINARY = """\
#include "../GpuBackend.h"
namespace lucid {
namespace backend {
Storage GpuBackend::add(const Storage& a, const Storage& b, const Shape& s, Dtype dt) {
    (void)b;
    (void)s;
    (void)dt;
    return a;
}
}  // namespace backend
}  // namespace lucid
"""


def _write_clean_fixture(tmp_path: Path) -> set[str]:
    files = {
        "lucid/_C/backend/cpu/CpuBackend.h": _CLEAN_CPU_HDR,
        "lucid/_C/backend/gpu/GpuBackend.h": _CLEAN_GPU_HDR,
        "lucid/_C/backend/cpu/methods/Binary.cpp": _CLEAN_CPU_BINARY,
        "lucid/_C/backend/gpu/methods/Binary.cpp": _CLEAN_GPU_BINARY,
    }
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return {"lucid/_C/backend/cpu/methods/Binary.cpp", "lucid/_C/backend/gpu/methods/Binary.cpp"}


def test_real_backend_tree_passes() -> None:
    violations = check_backend_layout.scan_backend_layout(root=check_backend_layout.ROOT)
    assert violations == []
    assert check_backend_layout.check() == 0


def test_rejects_inline_non_keep_inline_and_long_keep_inline(tmp_path: Path) -> None:
    tracked = _write_clean_fixture(tmp_path)
    assert check_backend_layout.scan_backend_layout(tmp_path, tracked) == []

    # 1. Non-KEEP_INLINE body in CpuBackend.h
    cpu_hdr = tmp_path / "lucid/_C/backend/cpu/CpuBackend.h"
    cpu_hdr.write_text(
        _CLEAN_CPU_HDR.replace(
            "Storage add(const Storage& a, const Storage& b, const Shape& s, Dtype dt) override;",
            "Storage add(const Storage& a, const Storage& b, const Shape& s, Dtype dt) override {\n"
            "    return a;\n"
            "}",
        ),
        encoding="utf-8",
    )
    violations = check_backend_layout.scan_backend_layout(tmp_path, tracked)
    assert [v.rule for v in violations] == ["inline-body"]

    # 2. >10-line KEEP_INLINE body in CpuBackend.h
    long_lines = "\n".join(f"        int v{i} = {i}; (void)v{i};" for i in range(12))
    cpu_hdr.write_text(
        _CLEAN_CPU_HDR.replace(
            "Device device() const noexcept override { return Device::CPU; }",
            f"Device device() const noexcept override {{\n{long_lines}\n        return Device::CPU;\n    }}",
        ),
        encoding="utf-8",
    )
    violations = check_backend_layout.scan_backend_layout(tmp_path, tracked)
    assert [v.rule for v in violations] == ["inline-body-too-long"]


def test_rejects_untracked_cpp_non_member_and_extra_include(tmp_path: Path) -> None:
    tracked = _write_clean_fixture(tmp_path)

    # 1. Untracked .cpp file under methods/
    stray = tmp_path / "lucid/_C/backend/cpu/methods/Stray.cpp"
    shutil.copy2(tmp_path / "lucid/_C/backend/cpu/methods/Binary.cpp", stray)
    violations = check_backend_layout.scan_backend_layout(tmp_path, tracked)
    assert [v.rule for v in violations] == ["untracked-methods-cpp"]
    stray.unlink()

    # 2. Namespace-scope free function in methods/Binary.cpp
    cpu_bin = tmp_path / "lucid/_C/backend/cpu/methods/Binary.cpp"
    cpu_bin.write_text(
        _CLEAN_CPU_BINARY.replace(
            "namespace backend {\n",
            "namespace backend {\nstatic int helper(int x) { return x + 1; }\n",
        ),
        encoding="utf-8",
    )
    violations = check_backend_layout.scan_backend_layout(tmp_path, tracked)
    assert [v.rule for v in violations] == ["non-member-definition"]

    # 3. Extra #include in methods/Binary.cpp
    cpu_bin.write_text("#include <vector>\n" + _CLEAN_CPU_BINARY, encoding="utf-8")
    violations = check_backend_layout.scan_backend_layout(tmp_path, tracked)
    assert [v.rule for v in violations] == ["methods-preamble"]

    # 4. Unallowed std::tanh / std::sin in CPU backend compute file (LCD-608)
    cpu_bin.write_text(
        _CLEAN_CPU_BINARY.replace(
            "return a;", "float y = std::tanh(0.5f) + std::sin(0.5f); (void)y;\n    return a;"
        ),
        encoding="utf-8",
    )
    violations = check_backend_layout.scan_backend_layout(tmp_path, tracked)
    assert [v.rule for v in violations] == ["cpu-scalar-libm", "cpu-scalar-libm"]
    cpu_bin.write_text(_CLEAN_CPU_BINARY, encoding="utf-8")

    # 5. Narrowing custom Metal kernel element count to int (LCD-591)
    gpu_k = tmp_path / "lucid/_C/backend/gpu/Pow.h"
    gpu_k.write_text(
        "void run(std::size_t n) {\n"
        '    auto k = mx::fast::metal_kernel("k", {"a"}, {"out"}, "");\n'
        "    int threads = static_cast<int>(n);\n"
        "    (void)k; (void)threads;\n"
        "}\n",
        encoding="utf-8",
    )
    violations = check_backend_layout.scan_backend_layout(tmp_path, tracked)
    assert [v.rule for v in violations] == ["metal-narrow-launch"]
