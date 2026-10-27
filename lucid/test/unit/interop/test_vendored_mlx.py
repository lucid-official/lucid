"""The engine runs on its own pinned MLX, beside whatever ``mlx`` is installed.

MLX's C++ ABI changes between patch releases, so ``engine*.so`` links the
build pinned in ``pyproject.toml [tool.lucid.mlx]``, renamed to
``liblucid_mlx.dylib`` (``tools/fetch_mlx.py``).  The rename is what lets a
user's ``mlx`` package — any version — load its own ``libmlx.dylib`` into the
same process: dyld reuses an image by install name, so two libraries both
called ``@rpath/libmlx.dylib`` would collapse into whichever loaded first and
the other side would die on a missing symbol.  Import order is therefore the
axis that matters, and each order needs a fresh process.

Which ``mlx`` is installed is the other axis: scripts/ci_mlx_compat.sh runs
this file against an installed wheel with none, the [mlx] extra's floor, and
the newest release (ci.yml's MLX compatibility job).
"""

import gc
import os
import shutil
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import metal_available

_PYPROJECT = Path(lucid.__file__).resolve().parent.parent / "pyproject.toml"

_requires_metal = pytest.mark.skipif(not metal_available(), reason="needs a Metal device")


def _pinned_version() -> str:
    if _PYPROJECT.is_file():
        with _PYPROJECT.open("rb") as fh:
            return str(tomllib.load(fh)["tool"]["lucid"]["mlx"]["version"])
    # An installed wheel has no pyproject beside it; scripts/ci_mlx_compat.sh
    # passes the pin it was built with.
    pin = os.environ.get("LUCID_MLX_PIN")
    if not pin:
        pytest.skip("pyproject.toml is not beside the lucid package and LUCID_MLX_PIN is unset")
    return pin


def test_the_engine_runs_the_pinned_mlx() -> None:
    assert _C_engine.mlx_runtime_version() == _pinned_version()


def test_the_engine_links_only_the_renamed_library() -> None:
    otool = shutil.which("otool")
    if otool is None:
        pytest.skip("otool (Xcode command line tools) is not installed")
    linked = subprocess.run(
        [otool, "-L", _C_engine.__file__], capture_output=True, text=True, check=True
    ).stdout
    assert "@rpath/liblucid_mlx.dylib" in linked
    assert "libmlx.dylib" not in linked


@_requires_metal
def test_metal_cache_queries_answer_for_the_engines_mlx() -> None:
    x = lucid.randn(512, 512, device="metal")
    float((x @ x).relu().sum().item())
    del x
    gc.collect()
    _C_engine.synchronize_gpu()
    # Freed buffers land in the engine's cache, so a read of 0 here would
    # mean the query looks at another MLX instance.
    assert _C_engine.metal_cache_memory() > 0
    _C_engine.metal_clear_cache()
    assert _C_engine.metal_cache_memory() == 0


@_requires_metal
def test_metal_memory_counters_track_the_engines_allocations() -> None:
    _C_engine.synchronize_gpu()
    _C_engine.metal_reset_peak_memory()
    before = _C_engine.metal_active_memory()
    x = lucid.randn(1024, 1024, device="metal")
    float(x.sum().item())
    assert _C_engine.metal_active_memory() >= before + 4 * 1024 * 1024
    assert _C_engine.metal_peak_memory() >= _C_engine.metal_active_memory()
    del x


@_requires_metal
def test_metal_device_info_names_the_device() -> None:
    info = _C_engine.metal_device_info()
    assert isinstance(info["device_name"], str) and info["device_name"]
    assert all(isinstance(v, (str, int)) for v in info.values())


# Runs in a fresh interpreter: argv[1] says which side is imported first.
_COEXIST = textwrap.dedent("""
    import sys

    if sys.argv[1] == "mlx_first":
        import mlx.core as mx
        import lucid
    else:
        import lucid
        import mlx.core as mx
    from lucid._C import engine as _C_engine

    a = lucid.arange(6, dtype=lucid.float32).reshape(2, 3).to("metal")
    product = a @ a.mT
    assert product.tolist() == [[5.0, 14.0], [14.0, 50.0]], product.tolist()

    m = mx.array([[1.0, 2.0], [3.0, 4.0]])
    square = m @ m
    mx.eval(square)
    assert square.tolist() == [[7.0, 10.0], [15.0, 22.0]], square.tolist()

    shared = mx.from_dlpack(product)
    mx.eval(shared)
    assert shared.tolist() == product.tolist()

    adopted = lucid.from_dlpack(square)
    assert adopted.is_metal
    assert adopted.tolist() == [[7.0, 10.0], [15.0, 22.0]]

    print(_C_engine.mlx_runtime_version(), mx.__version__)
    """)


@_requires_metal
@pytest.mark.parametrize("order", ["mlx_first", "lucid_first"])
def test_a_python_mlx_coexists_in_either_import_order(order: str) -> None:
    pytest.importorskip("mlx.core", reason="coexistence needs the mlx package")
    proc = subprocess.run(
        [sys.executable, "-c", _COEXIST, order],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    engine_mlx, _python_mlx = proc.stdout.split()
    assert engine_mlx == _pinned_version()
