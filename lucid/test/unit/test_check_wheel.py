"""The publish gate refuses a wheel that cannot run on a clean machine.

``tools/check_wheel.py`` runs on every wheel before it is smoke-tested and
published.  Each test builds a small stand-in wheel with one defect.
"""

import zipfile
from pathlib import Path

import pytest

from tools import check_wheel

_FLOOR = "26.2"
_TAG = check_wheel.platform_tag(_FLOOR)
_COMPLETE = (
    "lucid/__init__.py",
    "lucid/_C/engine.cpython-314-darwin.so",
    *(f"lucid/_C/{name}" for name in check_wheel.SHIPPED),
)


def _wheel(tmp_path: Path, members: tuple[str, ...], tag: str = _TAG) -> Path:
    path = tmp_path / f"lucid_dl-0.0.0-cp314-cp314-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name in members:
            archive.writestr(name, b"")
    return path


def test_a_complete_wheel_passes(tmp_path: Path) -> None:
    assert check_wheel.check(_wheel(tmp_path, _COMPLETE), _FLOOR) == ([], [])


def test_the_tag_names_the_floors_major_only() -> None:
    # pip has no 26_2 tag; lucid._C enforces the minor at import instead.
    assert _TAG == "macosx_26_0_arm64"


@pytest.mark.parametrize("missing", check_wheel.SHIPPED)
def test_a_missing_mlx_runtime_file_is_refused(tmp_path: Path, missing: str) -> None:
    members = tuple(m for m in _COMPLETE if not m.endswith(missing))
    problems, _ = check_wheel.check(_wheel(tmp_path, members), _FLOOR)
    assert any(missing in p for p in problems), problems


@pytest.mark.parametrize("foreign", check_wheel.FOREIGN)
def test_a_python_mlx_library_name_is_refused(tmp_path: Path, foreign: str) -> None:
    problems, _ = check_wheel.check(_wheel(tmp_path, (*_COMPLETE, f"lucid/_C/{foreign}")), _FLOOR)
    assert any(foreign in p for p in problems), problems


def test_a_universal2_tag_is_refused(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, _COMPLETE, tag="macosx_26_0_universal2")
    problems, _ = check_wheel.check(wheel, _FLOOR)
    assert any("platform tag" in p for p in problems), problems


def test_size_warns_near_the_pypi_limit_and_fails_past_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = _wheel(tmp_path, _COMPLETE)
    size = wheel.stat().st_size
    monkeypatch.setattr(check_wheel, "WARN_AT", size - 1)
    monkeypatch.setattr(check_wheel, "PYPI_LIMIT", size + 1)
    problems, warnings = check_wheel.check(wheel, _FLOOR)
    assert problems == [] and len(warnings) == 1
    monkeypatch.setattr(check_wheel, "PYPI_LIMIT", size - 1)
    problems, _ = check_wheel.check(wheel, _FLOOR)
    assert any("PyPI" in p for p in problems), problems
