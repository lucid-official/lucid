"""C++ leaves for main only with a recorded pass of the suites that see it, on an engine built from it.

The suites take minutes, so the run and the check are apart: ``attest`` runs them where
the engine is fresh and records the result under ``HEAD:lucid/_C``, the C++ tree's hash,
and pre-push only looks it up.  One record covers any number of commits that leave the
C++ as it is, a Python-only change needs none, and every worktree reads the same ledger.
"""

import json
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ElementTree
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Self

# One rule for "is this engine built from these sources", shared with the workspace tool.
from tools.agent_ws import _engine_is_stale, _header_abi_version
from tools.rules.checks import Workspace
from tools.rules.model import Exit, read
from tools.rules.push import _git

ENGINE: Final = "lucid/_C"
# The suites that see what the C++ does to a traced graph and what each op costs.
SUITES: Final = ("lucid/test/unit/compile", "lucid/test/unit/cost")
# What the suites read: an edit there that is not committed is not what a record names.
_CLEAN: Final = ("lucid", "pyproject.toml")
_COMMAND: Final = ".venv/bin/python3 -m tools.rules attest"
_PROBE: Final = (
    "import lucid; from lucid import _C_engine as e; "
    "print(lucid.__file__, e.__file__, e.ABI_VERSION, sep='\\n')"
)


@dataclass(frozen=True)
class SuiteRun:
    path: str
    returncode: int
    tests: int
    failures: int  # and errors
    skipped: int

    @classmethod
    def of(cls, path: str, returncode: int, report: Path) -> Self:
        suites = list(ElementTree.parse(report).iter("testsuite")) if report.is_file() else []
        count = {
            key: sum(int(s.get(key, "0")) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        return cls(
            path, returncode, count["tests"], count["failures"] + count["errors"], count["skipped"]
        )

    @property
    def line(self) -> str:
        return f"{self.path}: {self.tests} tests, {self.failures} failed, {self.skipped} skipped"


@dataclass(frozen=True)
class Attestation:
    tree: str
    commit: str
    host: str
    at: str
    seconds: float
    abi: int
    suites: tuple[SuiteRun, ...]

    @property
    def passed(self) -> bool:
        ran = all(run.returncode == 0 and run.tests > 0 for run in self.suites)
        return ran and tuple(run.path for run in self.suites) == SUITES

    @property
    def line(self) -> str:
        word = "passed" if self.passed else "FAILED"
        return (
            f"{word} on {self.host} at {self.at} in {self.seconds:.0f}s (commit {self.commit[:9]})"
        )


def engine_tree(ws: Workspace) -> str:
    return _git(ws, "rev-parse", "--verify", f"HEAD:{ENGINE}")


@dataclass(frozen=True)
class Ledger:
    """``<git common dir>/lucid-attest/<tree>.json``: local, and shared by every worktree."""

    directory: Path

    @classmethod
    def of(cls, ws: Workspace) -> Self:
        common = _git(ws, "rev-parse", "--path-format=absolute", "--git-common-dir")
        return cls(Path(common) / "lucid-attest")

    def record(self, attestation: Attestation) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{attestation.tree}.json"
        part = path.with_suffix(".part")
        part.write_text(json.dumps(asdict(attestation), indent=2), encoding="utf-8")
        part.replace(path)  # a reader sees the old record or the new one, never half
        return path

    def find(self, tree: str) -> Attestation | None:
        """The record for ``tree``; one that does not read as a record is none."""
        path = self.directory / f"{tree}.json"
        try:
            table = json.loads(path.read_text(encoding="utf-8"))
            runs = tuple(read(SuiteRun, run, str(path)) for run in table.pop("suites"))
            return read(Attestation, table, str(path), suites=runs)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None


def _engine_problem(ws: Workspace) -> str | None:
    """Why the suites here would not run this tree's C++, or None."""
    if _engine_is_stale(ws.root):
        return f"the engine here is missing or older than {ENGINE}: tools/agent_ws.py build"
    done = ws.spawn([ws.python, "-c", _PROBE])
    found = done.stdout.splitlines()
    if done.returncode != 0 or len(found) != 3:
        return f"{ws.python} cannot import the engine: {done.stderr.strip()[-300:]}"
    package, engine = Path(found[0]).parent, Path(found[1]).parent
    if package != ws.root / "lucid" or engine != ws.root / ENGINE:
        return f"{ws.python} imports lucid from {package} and its engine from {engine}"
    if found[2] != str(_header_abi_version(ws.root)):
        return f"the engine says ABI {found[2]}, and {ENGINE}/version.h does not"
    return None


def _dirty(ws: Workspace) -> str:
    return _git(ws, "status", "--porcelain", "--untracked-files=normal", "--", *_CLEAN)


def _run_suites(ws: Workspace) -> list[SuiteRun]:
    """One process each, its output on the terminal: these take minutes."""
    runs = []
    for suite in SUITES:
        with tempfile.TemporaryDirectory(prefix="lucid-attest-") as tmp:
            report = Path(tmp) / "junit.xml"
            argv = [
                ws.python,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={report}",
            ]
            done = subprocess.run([*argv, suite], cwd=ws.root, check=False)
            runs.append(SuiteRun.of(suite, done.returncode, report))
    return runs


def attest(ws: Workspace) -> Exit:
    """Run the suites on this checkout's engine and record the result under its C++ tree."""
    tree, dirty = engine_tree(ws), _dirty(ws)
    problem = f"what the suites read is not committed:\n{dirty}" if dirty else _engine_problem(ws)
    start = time.monotonic()
    runs = [] if problem else _run_suites(ws)
    idle = [run.path for run in runs if run.tests == 0]
    if idle:
        problem = f"{', '.join(idle)} selected no test: a run of nothing proves nothing"
    elif runs and (engine_tree(ws) != tree or _dirty(ws)):
        problem = f"{ENGINE} or what the suites read changed during the run"
    if problem:
        print(f"[rules] attest refused: {problem}", file=sys.stderr)
        return Exit.UNMEASURED
    commit, host = _git(ws, "rev-parse", "HEAD"), platform.node().split(".")[0]
    at, seconds = time.strftime("%Y-%m-%dT%H:%M:%S%z"), round(time.monotonic() - start, 1)
    made = Attestation(
        tree, commit, host, at, seconds, _header_abi_version(ws.root) or 0, tuple(runs)
    )
    path = Ledger.of(ws).record(made)
    print("".join(f"  {run.line}\n" for run in runs), end="")
    print(f"attest: {ENGINE} tree {tree[:9]} {made.line}\n  recorded in {path}")
    return Exit.PASS if made.passed else Exit.VIOLATION


def verify(ws: Workspace) -> Exit:
    """What pre-push asks: has this C++ tree a passing record?"""
    tree, ledger = engine_tree(ws), Ledger.of(ws)
    found = ledger.find(tree)
    if found is not None and found.passed:
        print(f"attested: {ENGINE} tree {tree[:9]} {found.line}")
        return Exit.PASS
    said = "has no record" if found is None else f"has a record that {found.line}"
    print(
        f"{ENGINE} at HEAD (tree {tree[:9]}) {said}:\n"
        f"{' and '.join(SUITES)} have not passed on an engine built from it.\n"
        f"Where this commit is checked out and its engine is built, run\n\n    {_COMMAND}\n\n"
        f"then push again.  The record goes to {ledger.directory},\nwhich every worktree shares."
    )
    return Exit.VIOLATION
