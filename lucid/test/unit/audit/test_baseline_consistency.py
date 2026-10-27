"""The audit's recorded baselines agree with themselves, and a commit cannot break that.

``coverage.json`` / ``doctest.json`` / ``suite.json`` each hold a header total
beside the body it sums.  The audit's writers derive one from the other; a
hand edit changes one of them.  LCD-358 was that: a deleted ``answered`` line
left ``cells`` one high and a unit test red on main.
``tools/check_audit_baselines.py`` is the guard, run by the pre-commit hook
on staged baselines and by ``tools/agent_ws.py land`` (whose own test is in
``tools/test_agent_ws.py``).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lucid.test.audit import _doctests, _suite
from lucid.test.audit.__main__ import save_coverage
from lucid.test.audit._result import Finding, Report, Status
from tools.check_audit_baselines import BASELINES, on_disk, problems

REPO = Path(__file__).resolve().parents[4]
COVERAGE = "lucid/test/audit/coverage.json"
DOCTEST = "lucid/test/audit/doctest.json"
SUITE = "lucid/test/audit/suite.json"


def _coverage(cells: int, answered: dict[str, str]) -> str:
    return json.dumps({"cells": cells, "answered": answered})


GOOD = {
    COVERAGE: _coverage(2, {"grad::lucid.exp": "pass", "grad::lucid.log": "vacuous"}),
    DOCTEST: json.dumps({"failed": 3, "attempted": 10, "failing": {"lucid.a": 1, "lucid.b": 2}}),
    SUITE: json.dumps(
        {
            "line_covered": 3,
            "line_total": 4,
            "percent": 75.0,
            "files": {"a.py": {"covered": 1, "total": 1}, "b.py": {"covered": 2, "total": 3}},
        }
    ),
}


def test_the_repository_baselines_agree_with_themselves() -> None:
    found = on_disk(REPO)
    assert set(found) == set(BASELINES), "a baseline is missing from this checkout"
    assert {rel: problems(rel, text) for rel, text in found.items()} == {rel: [] for rel in found}


@pytest.mark.parametrize("rel", sorted(GOOD))
def test_a_consistent_baseline_passes(rel: str) -> None:
    assert problems(rel, GOOD[rel]) == []


def test_what_the_audit_writes_passes(tmp_path: Path) -> None:
    """The guard and the writers agree on the format — consistent by construction."""
    report = Report()
    report.add(Finding("grad", "lucid.exp", Status.PASS, "detail"))
    report.add(Finding("grad", "lucid.log", Status.FAIL, "detail"))
    report.add(Finding("grad", "lucid.sin", Status.SKIP, "detail"))
    save_coverage(tmp_path / "coverage.json", report)

    _doctests.save_floor(
        tmp_path / "doctest.json",
        _doctests.DoctestResult(3, 10, {"lucid.a": 1, "lucid.b": 2}, ran=True),
    )
    result = _suite.SuiteResult(
        ran=True, line_covered=3, line_total=7, per_file={"a.py": (1, 4), "b.py": (2, 3)}
    )
    _suite.save_floor(tmp_path / "suite.json", result)

    for rel in BASELINES:
        written = (tmp_path / Path(rel).name).read_text()
        assert problems(rel, written) == [], rel


INCONSISTENT = [
    pytest.param(
        COVERAGE,
        _coverage(3, {"grad::lucid.exp": "pass", "grad::lucid.log": "pass"}),
        "'cells' is 3 but 'answered' holds 2 entries",
        id="coverage-cells-high",
    ),
    pytest.param(
        COVERAGE,
        _coverage(1, {"grad::lucid.exp": "pass", "grad::lucid.log": "pass"}),
        "'cells' is 1 but 'answered' holds 2 entries",
        id="coverage-cells-low",
    ),
    pytest.param(
        COVERAGE,
        _coverage(1, {"grad::lucid.exp": "skip"}),
        "not a verdict",
        id="coverage-unrecorded-status",
    ),
    pytest.param(
        COVERAGE, _coverage(1, {"lucid.exp": "pass"}), "not 'axis::symbol'", id="coverage-key"
    ),
    pytest.param(
        COVERAGE, json.dumps({"answered": {}}), "'cells' is not a count", id="coverage-no-cells"
    ),
    pytest.param(
        DOCTEST,
        json.dumps({"failed": 2, "attempted": 10, "failing": {"lucid.a": 1}}),
        "'failed' is 2 but 'failing' sums to 1",
        id="doctest-failed",
    ),
    pytest.param(
        DOCTEST,
        json.dumps({"failed": 5, "attempted": 4, "failing": {"lucid.a": 5}}),
        "exceeds 'attempted'",
        id="doctest-attempted",
    ),
    pytest.param(
        SUITE,
        GOOD[SUITE].replace('"line_covered": 3', '"line_covered": 4'),
        "'line_covered' is 4 but 'files' sums to 3",
        id="suite-covered",
    ),
    pytest.param(
        SUITE,
        GOOD[SUITE].replace('"line_total": 4', '"line_total": 5'),
        "'line_total' is 5 but 'files' sums to 4",
        id="suite-total",
    ),
    pytest.param(
        SUITE,
        GOOD[SUITE].replace('"percent": 75.0', '"percent": 74.0'),
        "'percent' is 74.0 but the counts give 75.0",
        id="suite-percent",
    ),
    pytest.param(COVERAGE, "{ not json", "not JSON", id="not-json"),
    pytest.param(SUITE, "[]", "the top level is not an object", id="not-an-object"),
]


@pytest.mark.parametrize(("rel", "text", "expected"), INCONSISTENT)
def test_an_inconsistent_baseline_is_refused(rel: str, text: str, expected: str) -> None:
    found = problems(rel, text)
    assert any(expected in line for line in found), found


# ── through a real `git commit` and the repository's pre-commit hook ─────────


@pytest.fixture
def scratch(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A repository running this checkout's pre-commit hook, with one consistent baseline."""
    hook = REPO / ".githooks" / "pre-commit"
    if not hook.exists() or shutil.which("git") is None:
        pytest.skip("needs git and the repository's hooks")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "LUCID_SKIP"))}
    # Only the gate under test: the badge refresh would write a README.
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", LUCID_SKIP_LOC_BADGE="1")
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    (root / "hooks").mkdir()
    shutil.copy2(hook, root / "hooks" / "pre-commit")
    shutil.copy2(REPO / "tools" / "check_audit_baselines.py", root / "tools")
    # The hook prefers <repo>/.venv/bin/python3; point it at this interpreter.
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python3").symlink_to(sys.executable)
    (root / ".gitignore").write_text(".venv/\n")
    (root / COVERAGE).parent.mkdir(parents=True)
    (root / COVERAGE).write_text(GOOD[COVERAGE])
    for args in (
        ["init", "-q", "-b", "main"],
        ["config", "user.name", "scratch"],
        ["config", "user.email", "scratch@example.invalid"],
        ["config", "core.hooksPath", "hooks"],
        ["add", "."],
        ["commit", "-q", "-m", "init"],
    ):
        proc = subprocess.run(
            ["git", *args], cwd=root, env=env, capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, (args, proc.stderr)
    return root, env


def _commit(root: Path, env: dict[str, str], mode: str) -> subprocess.CompletedProcess[str]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=root, env=env, capture_output=True, text=True, check=False
        )

    if mode == "staged":
        assert git("add", COVERAGE).returncode == 0
        return git("commit", "-q", "-m", "baseline")
    # `git commit -- <path>`: git hands the hook a temporary index.
    return git("commit", "-q", "-m", "baseline", "--", COVERAGE)


MODES = ("staged", "partial")


@pytest.mark.parametrize("mode", MODES)
def test_the_hook_refuses_an_inconsistent_staged_baseline(
    scratch: tuple[Path, dict[str, str]], mode: str
) -> None:
    root, env = scratch
    (root / COVERAGE).write_text(_coverage(3, {"grad::lucid.exp": "pass"}))
    result = _commit(root, env, mode)
    assert result.returncode != 0
    assert "'cells' is 3 but 'answered' holds 1 entries" in result.stderr


@pytest.mark.parametrize("mode", MODES)
def test_the_hook_takes_a_consistent_staged_baseline(
    scratch: tuple[Path, dict[str, str]], mode: str
) -> None:
    root, env = scratch
    (root / COVERAGE).write_text(_coverage(1, {"grad::lucid.exp": "pass"}))
    result = _commit(root, env, mode)
    assert result.returncode == 0, result.stderr


def test_the_hook_judges_the_index_not_the_working_tree(
    scratch: tuple[Path, dict[str, str]],
) -> None:
    root, env = scratch
    (root / COVERAGE).write_text(_coverage(1, {"grad::lucid.exp": "pass"}))
    assert subprocess.run(["git", "add", COVERAGE], cwd=root, env=env, check=False).returncode == 0
    (root / COVERAGE).write_text("{ an unstaged edit that is not even JSON")
    result = subprocess.run(
        ["git", "commit", "-q", "-m", "baseline"], cwd=root, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
