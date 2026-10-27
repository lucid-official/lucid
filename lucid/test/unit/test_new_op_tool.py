"""Unit tests for ``tools/new_op.py`` across inline and split backend layouts."""

from pathlib import Path
import shutil

import pytest

from tools import new_op

_INLINE_CPU_HEADER = """\
#pragma once
namespace lucid::backend {
class CpuBackend final : public IBackend {
public:
    Storage reciprocal(const Storage& a, const Shape& shape, Dtype dt) override {
        return a;
    }

    Storage add(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return a;
    }
};
}  // namespace lucid::backend
"""

_INLINE_GPU_HEADER = """\
#pragma once
namespace lucid::backend {
class GpuBackend final : public IBackend {
public:
    Storage erfinv(const Storage& a, const Shape& shape, Dtype dt) override {
        return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::erfinv(x); });
    }

    Storage add(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return mlx_binary(a, b, shape, dt, [](auto& x, auto& y) { return ::mlx::core::add(x, y); });
    }
};
}  // namespace lucid::backend
"""


def _copy_real_split_backends(tmp_path: Path) -> Path:
    for rel in (
        "lucid/_C/backend/cpu/CpuBackend.h",
        "lucid/_C/backend/cpu/methods/Unary.cpp",
        "lucid/_C/backend/cpu/methods/Binary.cpp",
        "lucid/_C/backend/gpu/GpuBackend.h",
        "lucid/_C/backend/gpu/methods/Unary.cpp",
        "lucid/_C/backend/gpu/methods/Binary.cpp",
    ):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(new_op.ROOT / rel, dst)
    return tmp_path / "lucid/_C/backend"


def test_new_op_supports_inline_backend_headers(tmp_path: Path) -> None:
    be_dir = tmp_path / "lucid/_C/backend"
    (be_dir / "cpu").mkdir(parents=True, exist_ok=True)
    (be_dir / "gpu").mkdir(parents=True, exist_ok=True)
    (be_dir / "cpu/CpuBackend.h").write_text(_INLINE_CPU_HEADER, encoding="utf-8")
    (be_dir / "gpu/GpuBackend.h").write_text(_INLINE_GPU_HEADER, encoding="utf-8")

    new_op._mod_cpu_unary("dummy_uop", dry_run=False, be_dir=be_dir, root=tmp_path)
    new_op._mod_cpu_binary("dummy_bop", dry_run=False, be_dir=be_dir, root=tmp_path)
    new_op._mod_gpu_unary("dummy_uop", dry_run=False, be_dir=be_dir, root=tmp_path)
    new_op._mod_gpu_binary("dummy_bop", dry_run=False, be_dir=be_dir, root=tmp_path)

    cpu_hdr = (be_dir / "cpu/CpuBackend.h").read_text(encoding="utf-8")
    gpu_hdr = (be_dir / "gpu/GpuBackend.h").read_text(encoding="utf-8")
    assert "Storage dummy_uop(const Storage& a, const Shape& shape, Dtype dt) override {" in cpu_hdr
    assert "Storage dummy_bop(const Storage& a, const Storage& b," in cpu_hdr
    assert "Storage dummy_uop(const Storage& a, const Shape& shape, Dtype dt) override {" in gpu_hdr
    assert "Storage dummy_bop(const Storage& a, const Storage& b," in gpu_hdr


def test_new_op_supports_split_backend_headers_and_methods_cpp(tmp_path: Path) -> None:
    be_dir = _copy_real_split_backends(tmp_path)

    new_op._mod_cpu_unary("dummy_uop", dry_run=False, be_dir=be_dir, root=tmp_path)
    new_op._mod_cpu_binary("dummy_bop", dry_run=False, be_dir=be_dir, root=tmp_path)
    new_op._mod_gpu_unary("dummy_uop", dry_run=False, be_dir=be_dir, root=tmp_path)
    new_op._mod_gpu_binary("dummy_bop", dry_run=False, be_dir=be_dir, root=tmp_path)

    cpu_hdr = (be_dir / "cpu/CpuBackend.h").read_text(encoding="utf-8")
    gpu_hdr = (be_dir / "gpu/GpuBackend.h").read_text(encoding="utf-8")
    cpu_unary = (be_dir / "cpu/methods/Unary.cpp").read_text(encoding="utf-8")
    cpu_binary = (be_dir / "cpu/methods/Binary.cpp").read_text(encoding="utf-8")
    gpu_unary = (be_dir / "gpu/methods/Unary.cpp").read_text(encoding="utf-8")
    gpu_binary = (be_dir / "gpu/methods/Binary.cpp").read_text(encoding="utf-8")

    assert "Storage dummy_uop(const Storage& a, const Shape& shape, Dtype dt) override;" in cpu_hdr
    assert (
        "Storage dummy_bop(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;"
        in cpu_hdr
    )
    assert (
        "Storage CpuBackend::dummy_uop(const Storage& a, const Shape& shape, Dtype dt) {"
        in cpu_unary
    )
    assert "Storage CpuBackend::dummy_bop(const Storage& a, const Storage& b," in cpu_binary

    assert "Storage dummy_uop(const Storage& a, const Shape& shape, Dtype dt) override;" in gpu_hdr
    assert (
        "Storage dummy_bop(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;"
        in gpu_hdr
    )
    assert (
        "Storage GpuBackend::dummy_uop(const Storage& a, const Shape& shape, Dtype dt) {"
        in gpu_unary
    )
    assert "Storage GpuBackend::dummy_bop(const Storage& a, const Storage& b," in gpu_binary


def test_new_op_raises_when_anchor_is_missing(tmp_path: Path) -> None:
    be_dir = tmp_path / "lucid/_C/backend"
    (be_dir / "cpu").mkdir(parents=True, exist_ok=True)
    (be_dir / "cpu/CpuBackend.h").write_text("// empty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Anchor not found"):
        new_op._mod_cpu_unary("dummy_uop", dry_run=True, be_dir=be_dir, root=tmp_path)
