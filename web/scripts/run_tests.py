#!/usr/bin/env python3
"""Run every ``web/scripts/test_*.py`` as a standalone script (LCD-970).

The docs runners install no pytest, so each test file in this directory is a
standalone script: a ``main()`` that prints its results and exits non-zero on
failure.  The workflows used to name the files they ran, one step each, so a
new test file ran nowhere until somebody remembered to edit a workflow --
``test_api_data_cached.py`` and ``test_build_api_data_subcategory.py`` ran in
no workflow at all.  This runner finds the files by glob instead: a test is
run from the moment it exists.

It reports two things a bare ``for f in test_*.py`` loop would let through:

* a ``test_*.py`` with no ``if __name__ == "__main__":`` block (a pytest-only
  file).  Run as a script it imports, runs nothing and exits 0 -- a test that
  passes by not being run, which is the defect this runner exists to close;
* ``--skip`` naming a file that is not there, and a glob that finds nothing
  (a moved directory would otherwise read as green).

``--skip NAME`` leaves a file out of this run.  A caller says what to leave out
and why, so a new test defaults to running everywhere: ``docs-api-data.yml``
skips ``test_peft_docs.py`` because it judges the committed JSON that job is
about to regenerate.

Usage:  python3 web/scripts/run_tests.py [--skip test_x.py ...]
"""

import argparse
import ast
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# A hung test should be a named failure, not a job that sits until the
# workflow's own timeout.  The slowest file (the Griffe build guards) takes
# ~15 s.
_TIMEOUT_S = 300


def _is_main_guard(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _has_entry_point(path: Path) -> bool:
    """Whether running ``path`` as a script runs anything."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return any(isinstance(n, ast.If) and _is_main_guard(n.test) for n in tree.body)


def _run_one(path: Path) -> str | None:
    """Run one test file; the failure reason, or None when it passed."""
    if not _has_entry_point(path):
        return (
            'no `if __name__ == "__main__":` block -- run as a script it executes '
            "nothing and exits 0; give it a standalone main() like its siblings"
        )
    try:
        status = subprocess.run([sys.executable, str(path)], timeout=_TIMEOUT_S).returncode
    except subprocess.TimeoutExpired:
        return f"timed out after {_TIMEOUT_S} s"
    return None if status == 0 else f"exit status {status}"


def run(directory: Path, skip: Sequence[str] = ()) -> int:
    """Run the tests under ``directory``; 0 when all passed, 1 on a failure,
    2 when the invocation itself is wrong."""
    tests = sorted(directory.glob("test_*.py"))
    unknown = sorted(set(skip) - {t.name for t in tests})
    if unknown:
        print(f"run_tests: --skip names no test file in {directory}: {unknown}", file=sys.stderr)
        return 2
    selected = [t for t in tests if t.name not in skip]
    if not selected:
        print(f"run_tests: no test_*.py to run in {directory}", file=sys.stderr)
        return 2

    failures: list[tuple[str, str]] = []
    timings: list[tuple[str, float]] = []
    for path in selected:
        print(f"\n== {path.name}", flush=True)
        started = time.monotonic()
        reason = _run_one(path)
        timings.append((path.name, time.monotonic() - started))
        if reason is not None:
            failures.append((path.name, reason))
            print(f"== FAIL {path.name}: {reason}", file=sys.stderr, flush=True)

    print("\n-- summary")
    failed = {name for name, _ in failures}
    for name, seconds in timings:
        print(f"  {'FAIL' if name in failed else 'ok  '}  {name:44} {seconds:5.1f}s")
    if skip:
        print(f"  skipped: {', '.join(sorted(skip))}")
    print(f"{len(selected) - len(failures)}/{len(selected)} test files passed")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="NAME",
        help="test file name (e.g. test_peft_docs.py) to leave out of this run; repeatable",
    )
    args = parser.parse_args(argv)
    return run(_HERE, args.skip)


if __name__ == "__main__":
    sys.exit(main())
