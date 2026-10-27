"""The built engine links the MLX build it ships (``tools/check_engine_link.py``).

MLX's C++ symbols change between patch releases, so an engine built against
one pin and run beside another's library dies at dlopen.  The check reads the
binaries instead; these tests run it on this checkout's engine and prove each
failure it exists for is reported, on copies of the engine that dyld resolves
differently.
"""

import shutil
from pathlib import Path

import pytest

from tools import check_engine_link, fetch_mlx

_ENGINE = check_engine_link.default_engine()

pytestmark = [
    pytest.mark.skipif(_ENGINE is None, reason="the engine is not built in this checkout"),
    pytest.mark.skipif(shutil.which("otool") is None, reason="needs the Xcode command line tools"),
]


@pytest.fixture(scope="module")
def engine() -> Path:
    assert _ENGINE is not None
    return _ENGINE


@pytest.fixture(scope="module")
def floor() -> str:
    return fetch_mlx.read_pin().macos


def _beside(engine: Path, directory: Path, mlx: Path) -> Path:
    """A copy of *engine* whose ``@loader_path`` rpath finds *mlx* first."""
    copy = directory / engine.name
    shutil.copyfile(engine, copy)
    shutil.copyfile(mlx, directory / fetch_mlx.MLX_LIB)
    return copy


def _shipped_mlx(engine: Path) -> Path:
    commands = check_engine_link.load_commands(engine)
    mlx = check_engine_link.resolve_mlx(engine, commands.rpaths)
    if mlx is None:
        pytest.skip("the engine's liblucid_mlx is not staged (python3 tools/fetch_mlx.py)")
    return mlx


def test_the_built_engine_links_its_pinned_mlx(engine: Path, floor: str) -> None:
    assert check_engine_link.check(engine, floor) == []


def test_every_mlx_import_is_bound_to_the_renamed_library(engine: Path) -> None:
    imports = check_engine_link.undefined_symbols(engine)
    bound = [name for name, lib in imports.items() if lib == "liblucid_mlx"]
    # ~200 for MLX 0.32: far below means the parse found nothing to check.
    assert len(bound) > 100


def test_a_library_missing_the_engines_symbols_is_reported(
    engine: Path, floor: str, tmp_path: Path
) -> None:
    # liblucid_jaccl exports none of mlx::core: the shape of a stale pin.
    jaccl = _shipped_mlx(engine).with_name(fetch_mlx.JACCL_LIB)
    problems = check_engine_link.check(_beside(engine, tmp_path, jaccl), floor)
    assert any("lacks" in p and "MLX symbols" in p for p in problems), problems


def _engine_minos_problems(engine: Path, floor: str, wheel: bool) -> list[str]:
    problems = check_engine_link.check(engine, floor, wheel=wheel)
    return [p for p in problems if p.startswith(f"{engine.name}: minos")]


def test_an_engine_below_the_floor_is_reported(engine: Path) -> None:
    assert _engine_minos_problems(engine, "99.0", wheel=False)


def test_a_developer_build_may_target_a_newer_macos(engine: Path) -> None:
    # MACOSX_DEPLOYMENT_TARGET=27.0 on a dev machine: above the floor is fine.
    older = check_engine_link.load_commands(engine).minos.split(".")[0] + ".0"
    assert _engine_minos_problems(engine, older, wheel=False) == []


def test_a_wheel_must_target_exactly_the_floor(engine: Path) -> None:
    older = check_engine_link.load_commands(engine).minos.split(".")[0] + ".0"
    if older == check_engine_link.load_commands(engine).minos:
        pytest.skip("the engine targets X.0: no lower floor of the same major to test")
    problems = _engine_minos_problems(engine, older, wheel=True)
    assert any("a wheel's must be" in p for p in problems), problems


def test_a_wheel_must_not_name_a_build_machine_path(engine: Path, floor: str) -> None:
    rpaths = check_engine_link.load_commands(engine).rpaths
    if rpaths == [check_engine_link.LOADER_PATH]:
        pytest.skip("this engine is a wheel build: it has no absolute rpath to refuse")
    problems = check_engine_link.check(engine, floor, wheel=True)
    assert any("absolute rpath" in p for p in problems), problems
