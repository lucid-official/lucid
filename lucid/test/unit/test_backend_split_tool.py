"""Unit tests for ``tools/backend_split.py`` and ``check_dtype_dispatch`` backend unit keys."""

from pathlib import Path

import pytest

from tools import backend_split, check_dtype_dispatch

_SYNTHETIC_CPU_HEADER = """\
#pragma once
#include "../IBackend.h"

namespace lucid {
namespace backend {

class CpuBackend final : public IBackend {
public:
    // Registers this backend with the Dispatcher.
    static void register_self() {
        Dispatcher::register_backend(Device::CPU, nullptr);
    }

    // Returns Device::CPU.
    Device device() const noexcept override { return Device::CPU; }

    // Doc comment for add.
    Storage add(const Storage& a, const Storage& b, const Shape& s, Dtype dt) override {
        // Inside add comment
        auto fn = [](int x) { return x + 1; };
        (void)fn;
        return a;
    }

    // Doc comment for astype with internal macro.
    Storage astype(const Storage& a, const Shape& shape, Dtype /*src_dt*/, Dtype dst_dt) override {
#define LOCAL_CAST_MACRO(X) (X)
        Storage out = LOCAL_CAST_MACRO(a);
#undef LOCAL_CAST_MACRO
        return out;
    }

    // Doc comment for linalg_eigh with default parameter.
    StoragePair linalg_eigh(const Storage& a,
                            const Shape& shape,
                            Dtype dt,
                            bool upper = false) override {
        (void)upper;
        return {a, a};
    }

    // Doc comment for fused_linear_relu_forward with #ifdef.
    Storage fused_linear_relu_forward(const Storage& x,
                                      const Storage& w,
                                      const Storage& b,
                                      const Shape& out_shape,
                                      Dtype dt) override {
#ifdef __APPLE__
        return x;
#else
        return w;
#endif
    }

private:
    struct AxisScatterWalk {
        int outer = 0;
    };

    static constexpr std::int64_t kColumnBudget = std::int64_t{1} << 27;

    // Helper returning nested struct.
    static AxisScatterWalk axis_scatter_walk(const Shape& base_shape, int dim) {
        (void)base_shape;
        return AxisScatterWalk{dim};
    }

    template <typename T>
    static T sum_values(const T* values, std::size_t n) {
        T acc = T{0};
        for (std::size_t i = 0; i < n; ++i)
            acc += values[i];
        return acc;
    }

    static void ctnd_im2col(const float* x, float* col, int n) {
        for (int i = 0; i < n; ++i)
            col[i] = x[i];
    }

    static void ctnd_im2col(const double* x, double* col, int n) {
        for (int i = 0; i < n; ++i)
            col[i] = x[i];
    }
};

}  // namespace backend
}  // namespace lucid
"""


def _write_synthetic_tree(tmp_path: Path) -> tuple[Path, Path]:
    base_dir = tmp_path / "base"
    work_dir = tmp_path / "work"
    for root in (base_dir, work_dir):
        cpu_hdr = root / "lucid/_C/backend/cpu/CpuBackend.h"
        cpu_hdr.parent.mkdir(parents=True, exist_ok=True)
        cpu_hdr.write_text(_SYNTHETIC_CPU_HEADER, encoding="utf-8")
    return base_dir, work_dir


def test_synthetic_move_and_verify_covers_all_cpp_constructs(tmp_path: Path) -> None:
    base_dir, work_dir = _write_synthetic_tree(tmp_path)

    counts = backend_split.move_families(
        backend="cpu", families=["all"], root=work_dir, format_files=False
    )
    assert counts == {
        "Factory": 1,
        "Binary": 1,
        "Indexing": 1,
        "Linalg": 1,
        "Matmul": 1,
        "Loss": 1,
        "Conv": 2,
    }

    moved, errors, keep_only = backend_split.verify_backend(
        backend="cpu", base=str(base_dir), root=work_dir
    )
    assert errors == []
    assert moved == 8
    assert keep_only is True

    # Check nested return type qualification and default-arg stripping
    indexing_cpp = (work_dir / "lucid/_C/backend/cpu/methods/Indexing.cpp").read_text(
        encoding="utf-8"
    )
    assert "CpuBackend::AxisScatterWalk CpuBackend::axis_scatter_walk(" in indexing_cpp
    assert "static " not in indexing_cpp

    linalg_cpp = (work_dir / "lucid/_C/backend/cpu/methods/Linalg.cpp").read_text(encoding="utf-8")
    assert "bool upper)" in linalg_cpp
    assert "= false" not in linalg_cpp
    assert "override" not in linalg_cpp

    # Doc comments remain in header above declarations
    hdr_after = (work_dir / "lucid/_C/backend/cpu/CpuBackend.h").read_text(encoding="utf-8")
    assert "// Doc comment for add." in hdr_after
    assert "bool upper = false) override;" in hdr_after


def test_verify_rejects_mutated_token_comment_and_extra_definition(tmp_path: Path) -> None:
    base_dir, work_dir = _write_synthetic_tree(tmp_path)
    backend_split.move_families(
        backend="cpu", families=["Binary"], root=work_dir, format_files=False
    )

    binary_path = work_dir / "lucid/_C/backend/cpu/methods/Binary.cpp"
    orig_binary = binary_path.read_text(encoding="utf-8")

    # 1. Mutate a single token in the moved body (x + 1 -> x + 2)
    binary_path.write_text(orig_binary.replace("x + 1", "x + 2"), encoding="utf-8")
    _, errors, _ = backend_split.verify_backend("cpu", str(base_dir), root=work_dir)
    assert any("body token diff" in e for e in errors)

    # 2. Mutate an interior comment in the moved body
    binary_path.write_text(
        orig_binary.replace("Inside add comment", "Altered comment"), encoding="utf-8"
    )
    _, errors, _ = backend_split.verify_backend("cpu", str(base_dir), root=work_dir)
    assert any("body comments changed" in e for e in errors)

    # 3. Inject a namespace-scope helper into Binary.cpp
    binary_path.write_text(
        orig_binary.replace(
            "namespace backend {\n",
            "namespace backend {\nstatic int stray_helper(int x) { return x; }\n",
        ),
        encoding="utf-8",
    )
    _, errors, _ = backend_split.verify_backend("cpu", str(base_dir), root=work_dir)
    assert any("non-CpuBackend definition" in e for e in errors)

    # Restore and verify clean
    binary_path.write_text(orig_binary, encoding="utf-8")
    _, errors, _ = backend_split.verify_backend("cpu", str(base_dir), root=work_dir)
    assert errors == []


def test_parse_header_rejects_class_scope_ifdef_and_raw_string() -> None:
    with_class_ifdef = _SYNTHETIC_CPU_HEADER.replace(
        "class CpuBackend final : public IBackend {\npublic:",
        "class CpuBackend final : public IBackend {\npublic:\n#ifdef FOO\n#endif",
    )
    with pytest.raises(ValueError, match="Preprocessor directive #ifdef at class scope"):
        backend_split.parse_header(with_class_ifdef, "CpuBackend")

    with_raw_str = _SYNTHETIC_CPU_HEADER.replace(
        "return a;", 'const char* s = R"(raw)"; (void)s; return a;'
    )
    with pytest.raises(ValueError, match="Raw string literal"):
        backend_split.parse_header(with_raw_str, "CpuBackend")


def test_check_dtype_dispatch_unit_key_tracks_moved_methods() -> None:
    assert (
        check_dtype_dispatch._known_unit("lucid/_C/backend/cpu/CpuBackend.h")
        == check_dtype_dispatch._CPU
    )
    assert (
        check_dtype_dispatch._known_unit("lucid/_C/backend/cpu/methods/Indexing.cpp")
        == check_dtype_dispatch._CPU
    )
    assert (
        check_dtype_dispatch._known_unit("lucid/_C/backend/gpu/methods/Loss.cpp")
        == check_dtype_dispatch._GPU
    )
    assert check_dtype_dispatch.main(["--root", str(backend_split.ROOT)]) == 0
