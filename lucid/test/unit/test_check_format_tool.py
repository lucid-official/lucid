"""``tools/check_format.sh`` gives a verdict only from a clang-format that ran.

With no clang-format to run it reported every one of the 476 C++ files as
``NEEDS FORMAT`` and exited 1, the verdict a real format difference gets
(LCD-572).  Inside a worktree, whose light venv has no clang-format, that
also failed ``scripts/ci_local.sh checks``.  A tool that never ran now exits
2 with ``cannot check`` and the reason.  Exit 2 is non-zero, so the CI stage
still fails, but nobody is told to reformat files that may be fine.

Each case runs the script in a scratch checkout: a copy of the script and
``.clang-format`` plus one C++ file.  Nothing in it is a git repository, and
``PATH`` and ``HOMEBREW_PREFIX`` point nowhere useful, so the only
clang-format the script can find is the one a case hands it.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "tools" / "check_format.sh"
_FORMATTED = "int f() { return 1; }\n"
_UNFORMATTED = "int  f( ){return 1;}\n"


def _checkout(tmp_path: Path, source: str, style: str | None = None) -> Path:
    (tmp_path / "tools").mkdir()
    shutil.copy(_SCRIPT, tmp_path / "tools" / "check_format.sh")
    clang_format = tmp_path / ".clang-format"
    if style is None:
        shutil.copy(_ROOT / ".clang-format", clang_format)
    else:
        clang_format.write_text(style)
    (tmp_path / "lucid" / "_C").mkdir(parents=True)
    (tmp_path / "lucid" / "_C" / "a.cpp").write_text(source)
    return tmp_path


def _run(checkout: Path, clang_format: str | None) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/tmp"),
        "HOMEBREW_PREFIX": str(checkout / "no-homebrew"),
        "GIT_CEILING_DIRECTORIES": str(checkout.parent),
    }
    if clang_format is not None:
        env["CLANG_FORMAT"] = clang_format
    return subprocess.run(
        ["/bin/bash", str(checkout / "tools" / "check_format.sh")],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.fixture(scope="module")
def clang_format() -> str:
    """The clang-format the real checkout's check runs, if there is one."""
    found = subprocess.run(
        ["/bin/bash", str(_SCRIPT), "--which"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if found.returncode != 0:
        pytest.skip(f"no clang-format on this machine: {found.stderr.strip()}")
    return found.stdout.strip()


def _formatted(clang_format: str, source: str) -> str:
    """``source`` as the repository's style lays it out."""
    return subprocess.run(
        [clang_format, f"--style=file:{_ROOT / '.clang-format'}"],
        input=source,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout


def test_no_clang_format_cannot_check(tmp_path: Path) -> None:
    done = _run(_checkout(tmp_path, _UNFORMATTED), None)
    assert done.returncode == 2, done
    assert "cannot check: clang-format not found" in done.stderr
    assert "CLANG_FORMAT=" in done.stderr
    assert "NEEDS FORMAT" not in done.stdout


@pytest.mark.parametrize(
    "given,said",
    [
        ("/nonexistent/clang-format", "clang-format not found at /nonexistent"),
        ("/usr/bin/false", "/usr/bin/false does not run as clang-format"),
    ],
    ids=["missing", "not-clang-format"],
)
def test_a_clang_format_that_does_not_run_cannot_check(
    tmp_path: Path, given: str, said: str
) -> None:
    # $CLANG_FORMAT names the tool exactly: no fallback to another one.
    done = _run(_checkout(tmp_path, _UNFORMATTED), given)
    assert done.returncode == 2, done
    assert f"cannot check: {said}" in done.stderr
    assert "NEEDS FORMAT" not in done.stdout


def test_a_formatted_file_passes(tmp_path: Path, clang_format: str) -> None:
    source = _formatted(clang_format, _FORMATTED)
    done = _run(_checkout(tmp_path, source), clang_format)
    assert done.returncode == 0, done
    assert "all C++ files formatted correctly" in done.stdout


def test_an_unformatted_file_needs_format(tmp_path: Path, clang_format: str) -> None:
    done = _run(_checkout(tmp_path, _UNFORMATTED), clang_format)
    assert done.returncode == 1, done
    assert "NEEDS FORMAT: lucid/_C/a.cpp" in done.stdout


def test_a_style_clang_format_cannot_read_cannot_check(tmp_path: Path, clang_format: str) -> None:
    # The tool runs, but fails before it compares anything: no verdict.
    done = _run(_checkout(tmp_path, _FORMATTED, style="NoSuchKey: 3\n"), clang_format)
    assert done.returncode == 2, done
    assert "cannot check" in done.stderr
    assert "NEEDS FORMAT" not in done.stdout
