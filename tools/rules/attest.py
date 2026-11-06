"""C++ leaves for main only with a recorded pass of the suites that see it, on an engine built from it.

The suites take minutes, so the run and the check are apart: ``attest`` runs them where
the engine is fresh and records the result under a key for what the engine is built from
(``HEAD:lucid/_C``, the native inputs outside it, the MLX pin), and pre-push only looks the
key up.  One record covers any number of commits that leave those as they are, a
Python-only change needs none, and every worktree reads the same ledger.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import tomllib
import xml.etree.ElementTree as ElementTree
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Self

# One rule for "is this engine built from these sources", shared with the workspace tool.
from tools.agent_ws import _engine_is_stale, _header_abi_version
from tools.rules.checks import CannotMeasure, Workspace
from tools.rules.model import Exit, read
from tools.rules.push import _git

ENGINE: Final = "lucid/_C"
# What builds the engine besides lucid/_C; pyproject.toml's [tool.lucid.mlx] pins the MLX.
NATIVE: Final = ("setup.py", "tools/fetch_mlx.py")
# The suites that see what the C++ does to a traced graph and what each op costs.
SUITES: Final = ("lucid/test/unit/compile", "lucid/test/unit/cost")
# What the suites read: an edit there that is not committed is not what a record names.
_CLEAN: Final = ("lucid", "pyproject.toml", *NATIVE)
# A selection from outside the repository: a --deselect or a plugin would pass what it hid.
# Autoload off drops the benchmark plugin a compile test's fixture needs: a red run then
# overwrites a green record (scripts/ci_local.sh sets it).
_SELECTION_ENV: Final = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD")
# Measured on a loaded Studio, 2026-11-06: unit/compile skips 13 of 7497 (0.17 %, xfails
# included), unit/cost 0 of 42.  A run under memory pressure (lucid/test/conftest.py's
# governor) skips by the hundred; 5 % is some 30 times what a sound run skips.
_MOST_SKIPPED: Final = 0.05
_COMMAND: Final = ".venv/bin/python3 -m tools.rules attest"
_PROBE: Final = (
    "import lucid; from lucid import _C_engine; "
    "print(lucid.__file__, _C_engine.__file__, _C_engine.ABI_VERSION, sep='\\n')"
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

    @property
    def unproven(self) -> str | None:
        """Why a clean exit here would prove nothing, or None.  junit's tests count skips."""
        if self.tests - self.skipped <= 0:
            return f"{self.path} ran no test ({self.tests} selected, {self.skipped} skipped)"
        if self.skipped > _MOST_SKIPPED * self.tests:
            return (
                f"{self.path} skipped {self.skipped} of {self.tests}, over {_MOST_SKIPPED:.0%}: "
                "read the memory governor's summary in its output, and run it again when "
                "memory is free"
            )
        return None


@dataclass(frozen=True)
class Attestation:
    key: str
    commit: str
    host: str
    at: str
    seconds: float
    abi: int
    suites: tuple[SuiteRun, ...]

    @property
    def passed(self) -> bool:
        ran = all(run.returncode == 0 and run.unproven is None for run in self.suites)
        return ran and tuple(run.path for run in self.suites) == SUITES

    @property
    def line(self) -> str:
        word = "passed" if self.passed else "FAILED"
        return (
            f"{word} on {self.host} at {self.at} in {self.seconds:.0f}s (commit {self.commit[:9]})"
        )


def engine_key(ws: Workspace) -> str:
    """One digest of what HEAD builds the engine from: lucid/_C, NATIVE and the MLX pin."""
    blobs = [_git(ws, "rev-parse", "--verify", f"HEAD:{ENGINE}")]
    blobs += [ws.spawn(["git", "rev-parse", "-q", "--verify", f"HEAD:{p}"]).stdout for p in NATIVE]
    project = ws.spawn(["git", "show", "HEAD:pyproject.toml"]).stdout
    try:
        pin = tomllib.loads(project).get("tool", {}).get("lucid", {}).get("mlx", {})
    except tomllib.TOMLDecodeError as exc:
        raise CannotMeasure(f"pyproject.toml at HEAD: {exc}") from exc
    text = "\0".join([*(blob.strip() for blob in blobs), json.dumps(pin, sort_keys=True)])
    return hashlib.sha1(text.encode()).hexdigest()


def _head(ws: Workspace) -> str:
    return _git(ws, "rev-parse", "--verify", "HEAD")


@dataclass(frozen=True)
class Ledger:
    """``<git common dir>/lucid-attest/<key>.json``: local, and shared by every worktree."""

    directory: Path

    @classmethod
    def of(cls, ws: Workspace) -> Self:
        common = _git(ws, "rev-parse", "--path-format=absolute", "--git-common-dir")
        return cls(Path(common) / "lucid-attest")

    def record(self, attestation: Attestation) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{attestation.key}.json"
        part = path.with_suffix(".part")
        part.write_text(json.dumps(asdict(attestation), indent=2), encoding="utf-8")
        part.replace(path)  # a reader sees the old record or the new one, never half
        return path

    def find(self, key: str) -> Attestation | None:
        """The record for ``key``; one that does not read as a record is none."""
        path = self.directory / f"{key}.json"
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
    env = {name: value for name, value in os.environ.items() if name not in _SELECTION_ENV}
    for name in sorted(os.environ.keys() - env.keys()):
        print(f"[rules] attest ignores {name}={os.environ[name]!r}: suites run as committed")
    runs = []
    for suite in SUITES:
        with tempfile.TemporaryDirectory(prefix="lucid-attest-") as tmp:
            report = Path(tmp) / "junit.xml"
            argv = [ws.python, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            done = subprocess.run(
                [*argv, f"--junitxml={report}", suite], cwd=ws.root, env=env, check=False
            )
            runs.append(SuiteRun.of(suite, done.returncode, report))
    return runs


def attest(ws: Workspace) -> Exit:
    """Run the suites on this checkout's engine and record the result under its key."""
    key, commit, dirty = engine_key(ws), _head(ws), _dirty(ws)
    problem = f"what the suites read is not committed:\n{dirty}" if dirty else _engine_problem(ws)
    start = time.monotonic()
    runs = [] if problem else _run_suites(ws)
    unproven = [why for run in runs if (why := run.unproven)]
    if unproven:
        problem = "; ".join(unproven)
    elif runs and (engine_key(ws), _head(ws), _dirty(ws)) != (key, commit, ""):
        problem = f"HEAD or what the suites read changed during the run, from {commit[:9]}"
    if problem:
        print(f"[rules] attest refused: {problem}", file=sys.stderr)
        return Exit.UNMEASURED
    host, at = platform.node().split(".")[0], time.strftime("%Y-%m-%dT%H:%M:%S%z")
    abi, seconds = _header_abi_version(ws.root) or 0, round(time.monotonic() - start, 1)
    made = Attestation(key, commit, host, at, seconds, abi, tuple(runs))
    path = Ledger.of(ws).record(made)
    print("".join(f"  {run.line}\n" for run in runs), end="")
    print(f"attest: key {key[:9]} {made.line}\n  recorded in {path}")
    return Exit.PASS if made.passed else Exit.VIOLATION


def verify(ws: Workspace) -> Exit:
    """What pre-push asks: has what HEAD builds the engine from a passing record?"""
    key, ledger = engine_key(ws), Ledger.of(ws)
    found = ledger.find(key)
    if found is not None and found.passed:
        print(f"attested: key {key[:9]} {found.line}")
        return Exit.PASS
    said = "has no record" if found is None else f"has a record that {found.line}"
    print(
        f"{ENGINE} and the native inputs at HEAD (key {key[:9]}) {said}:\n"
        f"{' and '.join(SUITES)} have not passed on an engine built from it.\n"
        f"Where this commit is checked out and its engine is built, run\n\n    {_COMMAND}\n\n"
        f"then push again.  The record goes to {ledger.directory},\nwhich every worktree shares."
    )
    return Exit.VIOLATION
