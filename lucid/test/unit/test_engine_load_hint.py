"""The engine refuses an unsupported macOS by name, and explains a missing MLX.

The engine ships an MLX build compiled for macOS 26.2, but its wheel can only
be tagged 26_0 (pip knows no minor versions past macOS 11), so 26.0 and 26.1
must be turned away at import with the floor in the message.  A source build
links the MLX staged in a per-machine cache; when that entry is gone — purged,
or a checkout synced from another machine — the load error names the command
that restores it.
"""

import tomllib
from pathlib import Path

import pytest

import lucid
from lucid._C import _MACOS_FLOOR, _engine_load_hint, _macos_floor_error


@pytest.mark.parametrize("macos", ["26.0", "26", "26.1", "26.1.1", "15.5"])
def test_macos_below_the_floor_is_refused_by_name(macos: str) -> None:
    error = _macos_floor_error(macos)
    assert error is not None
    assert "26.2" in error
    assert macos in error


@pytest.mark.parametrize("macos", ["26.2", "26.2.1", "26.10", "27.0", ""])
def test_the_floor_and_above_load(macos: str) -> None:
    assert _macos_floor_error(macos) is None


def test_the_floor_is_the_pinned_mlx_builds() -> None:
    pyproject = Path(lucid.__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml is not beside the lucid package (installed wheel)")
    with pyproject.open("rb") as fh:
        pinned = tomllib.load(fh)["tool"]["lucid"]["mlx"]["macos"]
    assert ".".join(map(str, _MACOS_FLOOR)) == pinned


def test_a_missing_bundled_mlx_points_at_fetch_mlx() -> None:
    message = (
        "dlopen(.../engine.cpython-314-darwin.so, 0x0002): Library not loaded: "
        "@rpath/liblucid_mlx.dylib\n  Referenced from: <...> engine.cpython-314-darwin.so"
    )
    hint = _engine_load_hint(message)
    assert hint is not None
    assert "python3 tools/fetch_mlx.py" in hint
    assert "mlx<" not in hint


def test_an_unknown_failure_gets_no_hint() -> None:
    assert _engine_load_hint("symbol not found in flat namespace '__ZN3mlx4core'") is None
