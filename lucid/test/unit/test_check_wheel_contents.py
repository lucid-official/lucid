"""The publish gate ships runtime files under ``lucid/`` and nothing else.

``tools/check_wheel.py`` allowlists the package's contents: Python, stubs,
``py.typed``, the engine and the MLX it bundles.  LCD-447: with
``include-package-data`` on, the wheel carried the test tree's JSON
baselines, the benchmark baseline and the whole C++ source tree.  Each test
adds one such entry to an otherwise complete stand-in wheel.
"""

import zipfile
from pathlib import Path

import pytest

from tools import check_wheel

_FLOOR = "26.2"
_RUNTIME = (
    "lucid/__init__.py",
    "lucid/__init__.pyi",
    "lucid/py.typed",
    "lucid/_C/__init__.py",
    "lucid/_C/engine.pyi",
    "lucid/_C/engine.cpython-314-darwin.so",
    *(f"lucid/_C/{name}" for name in check_wheel.SHIPPED),
    "lucid/models/",
    "lucid_dl-0.0.0.dist-info/METADATA",
    "lucid_dl-0.0.0.dist-info/licenses/THIRD_PARTY_NOTICES.md",
    "lucid_dl-0.0.0.dist-info/entry_points.txt",
)


def _wheel(tmp_path: Path, members: tuple[str, ...]) -> Path:
    path = tmp_path / f"lucid_dl-0.0.0-cp314-cp314-{check_wheel.platform_tag(_FLOOR)}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name in members:
            archive.writestr(name, b"")
    return path


def test_runtime_files_and_metadata_pass(tmp_path: Path) -> None:
    assert check_wheel.check(_wheel(tmp_path, _RUNTIME), _FLOOR) == ([], [])


@pytest.mark.parametrize(
    "entry",
    [
        "lucid/test/audit/coverage.json",
        "lucid/test/unit/cost/step_cost.json",
        "lucid/test/conftest.py",
        "lucid/benchmarks/baseline/baseline.json",
        "lucid/benchmarks/run_all.py",
    ],
)
def test_the_test_and_benchmark_trees_are_refused(tmp_path: Path, entry: str) -> None:
    # Their .py files too: packages.find excluding them is one setting away.
    problems, _ = check_wheel.check(_wheel(tmp_path, (*_RUNTIME, entry)), _FLOOR)
    assert problems == [f"ships {entry}: the test and benchmark trees stay in the checkout"]


@pytest.mark.parametrize(
    "entry",
    [
        "lucid/_C/core/TensorImpl.cpp",
        "lucid/_C/core/TensorImpl.h",
        "lucid/_C/backend/gpu/MetalAllocator.mm",
        "lucid/_C/CMakeLists.txt",
        "lucid/_C/test/cmake_test_discovery_34e61b15ea.json",
        "lucid/models/demo.ipynb",
        "lucid/nn/__pycache__/module.cpython-314.pyc",
    ],
)
def test_source_build_and_cache_files_are_refused(tmp_path: Path, entry: str) -> None:
    problems, _ = check_wheel.check(_wheel(tmp_path, (*_RUNTIME, entry)), _FLOOR)
    assert problems == [f"ships {entry}: not a runtime file (Python, stub or py.typed)"]


def test_a_foreign_mlx_library_is_refused_once(tmp_path: Path) -> None:
    # It is not a runtime file either; the gate says the reason that matters.
    problems, _ = check_wheel.check(_wheel(tmp_path, (*_RUNTIME, "lucid/_C/libmlx.dylib")), _FLOOR)
    assert len(problems) == 1 and "dyld would share" in problems[0], problems
