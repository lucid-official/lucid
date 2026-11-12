#!/usr/bin/env python3
"""Tests for ``run_tests.py``, the runner every docs workflow uses (LCD-970).

The runner is the gate for every other test in this directory, so its two
reasons to fail -- a test that exits non-zero, and a test file that would run
nothing -- and its two refusals (a ``--skip`` that names no file, a directory
with no tests) are pinned against throwaway directories.

Standalone like the other ``web/scripts/test_*.py``: ``python3
web/scripts/test_run_tests.py``, no pytest needed, also pytest-collectable.
"""

import contextlib
import importlib.util
import io
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load_runner():
    path = _HERE / "run_tests.py"
    spec = importlib.util.spec_from_file_location("_run_tests", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_R = _load_runner()

_PASSES = 'import sys\n\nif __name__ == "__main__":\n    sys.exit(0)\n'
_FAILS = 'import sys\n\nif __name__ == "__main__":\n    sys.exit(3)\n'
# What a pytest-only file looks like: importable, collectable, and a no-op as a script.
_PYTEST_ONLY = "def test_x() -> None:\n    assert False\n"


@contextlib.contextmanager
def _directory(**files: str) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, text in files.items():
            (root / f"{name}.py").write_text(text)
        yield root


def _run(root: Path, skip: tuple[str, ...] = ()) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        status = _R.run(root, skip)
    return status, out.getvalue(), err.getvalue()


def test_all_passing_files_pass() -> None:
    with _directory(test_a=_PASSES, test_b=_PASSES) as root:
        status, out, _err = _run(root)
    assert status == 0
    assert "2/2 test files passed" in out, out


def test_a_failing_file_fails_the_run_and_the_others_still_run() -> None:
    with _directory(test_a=_FAILS, test_b=_PASSES) as root:
        status, out, err = _run(root)
    assert status == 1
    assert "1/2 test files passed" in out, out
    assert "test_a.py: exit status 3" in err, err
    assert "ok    test_b.py" in out, "a failure must not stop the files after it"


def test_a_file_with_no_entry_point_is_a_failure_not_a_pass() -> None:
    """Run as a script a pytest-only file exits 0 having run nothing -- the
    silent pass that let two tests sit unrun in this very directory."""
    with _directory(test_a=_PASSES, test_b=_PYTEST_ONLY) as root:
        status, _out, err = _run(root)
    assert status == 1
    assert "test_b.py" in err and "__main__" in err, err


def test_skip_leaves_a_file_out_and_says_so() -> None:
    with _directory(test_a=_PASSES, test_b=_FAILS) as root:
        status, out, _err = _run(root, skip=("test_b.py",))
    assert status == 0
    assert "1/1 test files passed" in out, out
    assert "skipped: test_b.py" in out, out


def test_skip_naming_no_file_is_refused() -> None:
    """A typo in a workflow's ``--skip`` must not quietly skip nothing."""
    with _directory(test_a=_PASSES) as root:
        status, _out, err = _run(root, skip=("test_missing.py",))
    assert status == 2
    assert "test_missing.py" in err, err


def test_a_directory_with_no_tests_is_refused() -> None:
    with _directory(helper=_PASSES) as root:
        status, _out, err = _run(root)
    assert status == 2
    assert "no test_*.py" in err, err


def test_the_files_beside_the_runner_all_have_an_entry_point() -> None:
    """Cheap and instant: the directory the workflows run holds no pytest-only
    file (the runner would fail on it, but not until a workflow got there)."""
    missing = [p.name for p in sorted(_HERE.glob("test_*.py")) if not _R._has_entry_point(p)]
    assert not missing, f"no standalone entry point: {missing}"


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    failed = 0
    for t in _TESTS:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}\n        {exc!r}", file=sys.stderr)
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} run-tests tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
