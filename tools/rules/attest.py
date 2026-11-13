"""C++ leaves for main only with a recorded pass of the suites that see it, on an engine built from it.

The suites take minutes, so the run and the check are apart: ``attest`` runs them where
the engine is fresh and records the result under a key for what the engine is built from
(``HEAD:lucid/_C``, the native inputs outside it, the MLX pin), and pre-push only looks the
key up.  One record covers any number of commits that leave those as they are, a
Python-only change needs none, and every worktree reads the same ledger.

A suite runs as several processes, each over a run of its test files, and their junit
totals are one ``SuiteRun``: the record is what it always was, whatever the cut.
"""

import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import xml.etree.ElementTree as ElementTree
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Final, Self

# One rule for "is this engine built from these sources": the build stamp, which the
# workspace tool reads too.
from tools.agent_ws import _header_abi_version
from tools.engine_stamp import ENGINE, NATIVE, EngineTree, mlx_pin
from tools.rules.checks import CannotMeasure, Workspace
from tools.rules.model import Exit, read
from tools.rules.push import _git

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
# One process holds what its tests leave behind, so a suite of thousands outgrows the machine:
# unit/compile, 14 280 tests in one process, was SIGKILLed on the 16 GB MacBook at about 80 %,
# with no traceback and no report.  Each pytest process takes at most this many test files.
# Measured 2026-11-13 (peak memory footprint, /usr/bin/time -l): 6 files make 31 chunks of
# unit/compile, 18 minutes in all, and only four chunks peak above 2 GB.  Those hold the files
# that peak that high alone (rank_canonical_widening 38.9 GB, norm_pool_rank_canonical 5.9,
# rank_canonical_movement 4.9, rank_canonical_attention 2.7): fewer files per chunk keeps such
# files apart, and cannot make one of them smaller.
CHUNK_FILES: Final = 6
# What pytest collects by file name when no ``python_files`` says otherwise.
_TEST_FILES: Final = ("test_*.py", "*_test.py")
# pytest's exit code for "no test was collected": a chunk whose tests are all deselected by
# the markers in addopts says it, and the suite's totals are what must be non-empty.
_NO_TESTS: Final = 5
# A chunk's pytest, which leaves the id of the test it is on in the file named first: a process
# the kernel kills writes no traceback and no report, and this is what it can still be asked.
_CHUNK_RUNNER: Final = """\
import sys, pytest
class OnNow:
    def pytest_runtest_logstart(self, nodeid, location):
        open(sys.argv[1], "w").write(nodeid)
sys.exit(pytest.main(sys.argv[2:], plugins=[OnNow()]))
"""
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

    @classmethod
    def merged(cls, path: str, parts: Sequence[Self]) -> Self:
        """One suite's totals from its chunks'.  A chunk that collected nothing failed nothing."""
        failed = next((p.returncode for p in parts if p.returncode not in (0, _NO_TESTS)), 0)
        tests = sum(p.tests for p in parts)
        return cls(
            path,
            failed or (0 if tests else _NO_TESTS),
            tests,
            sum(p.failures for p in parts),
            sum(p.skipped for p in parts),
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
        pin = mlx_pin(project)
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
    # The stamp its build leaves, equal to this tree's digest: an engine without one may be
    # a ``cp -c`` or a sync of a binary nobody built from here, and mtimes cannot tell.
    if why := EngineTree(ws.root).unfit():
        return why
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


@dataclass(frozen=True)
class Chunk:
    """The test files of one suite that one pytest process takes."""

    suite: str
    number: int
    of: int
    files: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.suite} chunk {self.number}/{self.of}"

    @property
    def span(self) -> str:
        return f"{len(self.files)} files: {', '.join(Path(f).name for f in self.files)}"


def _test_files(ws: Workspace, suite: str) -> list[str]:
    """The test files HEAD holds under ``suite``, sorted: one tree is always cut the same way."""
    names = _git(ws, "ls-files", "-z", "--", suite).split("\0")
    return sorted(n for n in names if any(fnmatch(Path(n).name, pat) for pat in _TEST_FILES))


def chunks_of(ws: Workspace, suite: str, most: int) -> tuple[Chunk, ...]:
    """``suite`` in the fewest runs of at most ``most`` files, the sorted files dealt out in turn.

    In turn, not in runs: files named alike cost alike (the ``rank_canonical_*`` family holds
    the three that peak above 2 GB), and dealt out they land in different processes.
    """
    files = _test_files(ws, suite)
    count = -(-len(files) // most)
    return tuple(Chunk(suite, n + 1, count, tuple(files[n::count])) for n in range(count))


def _death(returncode: int, report: Path) -> str | None:
    """How a pytest process ended without finishing, or None.  What it ran proved nothing."""
    if returncode < 0:
        try:
            name = f" ({signal.Signals(-returncode).name})"
        except ValueError:
            name = ""
        likely = ", likely memory" if -returncode == signal.SIGKILL else ""
        return f"killed by signal {-returncode}{name}{likely}"
    if not report.is_file():
        shell = " (above 128 a shell reports a signal)" if returncode > 128 else ""
        return f"exited {returncode}{shell} and wrote no report"
    try:
        ElementTree.parse(report)
    except ElementTree.ParseError:
        return f"exited {returncode} and wrote a report that does not parse"
    return None


@dataclass(frozen=True)
class ChunkRun:
    chunk: Chunk
    run: SuiteRun
    died: str | None
    on_now: str  # the test it was on when it died
    seconds: float


def _run_chunk(ws: Workspace, chunk: Chunk, env: dict[str, str]) -> ChunkRun:
    """One process, its output on the terminal."""
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="lucid-attest-") as tmp:
        report, marker = Path(tmp) / "junit.xml", Path(tmp) / "on-now"
        argv = [ws.python, "-c", _CHUNK_RUNNER, str(marker), "-q", "-p", "no:cacheprovider"]
        done = subprocess.run(
            [*argv, f"--junitxml={report}", *chunk.files], cwd=ws.root, env=env, check=False
        )
        died = _death(done.returncode, report)
        on_now = marker.read_text() if died and marker.is_file() else ""
        run = (
            SuiteRun(chunk.name, done.returncode, 0, 0, 0)
            if died
            else SuiteRun.of(chunk.name, done.returncode, report)
        )
    return ChunkRun(chunk, run, died, on_now, time.monotonic() - start)


def _run_suites(ws: Workspace, most: int) -> tuple[list[SuiteRun], str | None]:
    """Each suite as its chunks, one after another; a chunk that died ends the run.

    Dead is not failed: a process the kernel killed says nothing about its tests, and the
    chunks after it say nothing about it.  The reason comes back, and no record is made.
    """
    env = {name: value for name, value in os.environ.items() if name not in _SELECTION_ENV}
    for name in sorted(os.environ.keys() - env.keys()):
        print(f"[rules] attest ignores {name}={os.environ[name]!r}: suites run as committed")
    plans = [chunks_of(ws, suite, most) for suite in SUITES]
    runs: list[SuiteRun] = []
    for suite, chunks in zip(SUITES, plans, strict=True):
        parts = []
        for chunk in chunks:
            print(f"[rules] attest: {chunk.name}: {len(chunk.files)} files", flush=True)
            made = _run_chunk(ws, chunk, env)
            if made.died:
                half = len(chunk.files) // 2
                split = f", or split it with `attest --chunk-files {half}`" if half else ""
                on_now = f"; it was running {made.on_now}" if made.on_now else ""
                return runs, (
                    f"{chunk.name} {made.died}{on_now} ({chunk.span}): "
                    f"rerun when memory is free{split}"
                )
            print(f"[rules] attest: {made.run.line} in {made.seconds:.0f}s", flush=True)
            parts.append(made.run)
        runs.append(SuiteRun.merged(suite, parts))
    return runs, None


def attest(ws: Workspace, most: int = CHUNK_FILES) -> Exit:
    """Run the suites on this checkout's engine and record the result under its key."""
    key, commit, dirty = engine_key(ws), _head(ws), _dirty(ws)
    problem = f"what the suites read is not committed:\n{dirty}" if dirty else _engine_problem(ws)
    start = time.monotonic()
    runs, died = ([], None) if problem else _run_suites(ws, most)
    unproven = [why for run in runs if (why := run.unproven)]
    if died:
        problem = died
    elif unproven:
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


def plan(ws: Workspace, most: int = CHUNK_FILES) -> Exit:
    """What ``attest`` would run, process by process, without running it."""
    for suite in SUITES:
        chunks = chunks_of(ws, suite, most)
        print(f"{suite}: {sum(len(c.files) for c in chunks)} files in {len(chunks)} chunks")
        print("".join(f"  {c.number}/{c.of}: {c.span}\n" for c in chunks), end="")
    return Exit.PASS


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
