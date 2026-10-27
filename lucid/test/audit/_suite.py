"""The test suite as a stage of the audit rather than a separate errand.

The audit walks the package and asks whether each reachable symbol keeps
its contract.  The suite asks whether specific values are the right
values.  They fail independently — and this session's evidence is that
most defects are only visible to one of them:

* the sweep found the gradient that was never wired and the sampler
  drawing at the wrong concentration;
* the suite found the assignment writing a rectangle instead of a
  diagonal, the histogram binning onto the wrong grid, the transform
  whose inverse was NaN, and the window that deleted its own frame.

So a gate that runs only one of them reports "clean" while half the
framework is unchecked.  This module runs the other half and folds its
result into the same verdict, which is what makes ``lucid-audit`` a
single command rather than the first of two.

Line coverage rides along because it costs nothing: the suite takes
9m09s uninstrumented and 8m29s under ``coverage`` — the wall clock is
MLX and Accelerate, not Python line tracing.  A floor that is free to
check should be checked on every run rather than remembered.
"""

import contextlib
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lucid.test.audit._console import Console


#: How far a module's line coverage may fall before it is called a
#: regression.  Not zero: adding statements to a well-covered file lowers
#: its percentage honestly, and a gate that fires on that teaches people
#: to ignore it.
_FILE_TOLERANCE = 2.0

#: The same, for the total.  Tighter, because the total absorbs the noise
#: of individual files and a real drop shows up here first.
_TOTAL_TOLERANCE = 0.10

#: Files below this many statements are not worth a per-file verdict —
#: one uncovered line in a twelve-line module is a 8% swing.
_MIN_STATEMENTS = 20

_COUNT = re.compile(r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)")
_FAILURE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)")

#: pytest's closing line always carries the elapsed time — "92 passed, 6
#: skipped in 0.25s".  The *collection* line carries counts and no time
#: ("collected 19146 items / 10831 deselected / 14 skipped"), so this is
#: what separates "the run ended" from "the run had started".  Judging on
#: counts alone is how a suite killed at 37% reported itself clean.
_TERMINAL = re.compile(r"\bin\s[\d.]+\s*s")


@dataclass
class SuiteResult:
    """What one run of the suite established."""

    ran: bool = False
    returncode: int = 0
    duration: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    line_covered: int | None = None
    line_total: int | None = None
    per_file: dict[str, tuple[int, int]] = field(default_factory=dict)
    unavailable: str | None = None
    #: Chunks that ended without reporting a summary — killed, crashed, or
    #: otherwise unable to say what they found.  Named rather than
    #: counted: "one chunk died" is only actionable with the name of it.
    unfinished: list[str] = field(default_factory=list)
    #: Chunks not started because the machine was out of memory, with the
    #: reading that decided it.  A run that skipped these checked less
    #: than a full one and has to say so.
    skipped_chunks: list[tuple[str, str]] = field(default_factory=list)

    @property
    def percent(self) -> float | None:
        """Line coverage as a percentage, or ``None`` if not measured."""
        if self.line_total in (None, 0) or self.line_covered is None:
            return None
        assert self.line_total is not None
        return 100.0 * self.line_covered / self.line_total

    @property
    def finished(self) -> bool:
        """Whether pytest reached a summary line at all.

        Every pytest run that completes reports ``passed``, ``failed`` or
        ``error``.  Those three and no others: the *collection* line
        already says ``collected 19146 items / 10831 deselected / 14
        skipped / 8315 selected``, so a check for "any counts at all" is
        satisfied before a single test has run — which is exactly how a
        suite killed at 37% reported itself clean twice.

        With the tree split across chunks the same trap reappears one
        level up: sixty-four chunks reporting and one dying still leaves
        plenty of counts, so a chunk that vanished has to veto the whole
        stage rather than be outvoted by its neighbours.
        """
        if self.unfinished:
            return False
        # Any counts at all is enough *here* because liveness is already
        # decided per chunk, on the exit code and the closing line rather
        # than on which words appeared.  Requiring passed/failed/error
        # instead would call a subtree whose every test is skipped — a
        # parity chunk with no reference framework installed — a suite
        # that never ran.
        return bool(self.counts)

    @property
    def broken(self) -> int:
        """Failures plus errors — the count that makes the verdict red.

        A stage that **died** counts as one problem rather than none.
        This was the reverse: ``returncode`` was recorded and never
        consulted, so a pytest killed at 37% — no summary line, no
        counts, no coverage data — produced ``broken == 0`` and the
        verdict printed ``suite failures 0`` over a suite that had not
        run.  A gate whose stage can vanish silently is worse than no
        gate, because it answers.
        """
        if not self.finished:
            return 1
        # A chunk that was skipped for memory is not a pass.  It is the
        # gate declining to answer for part of the tree, and it goes red
        # for the same reason a dead chunk does: the alternative is a
        # green run over a suite that checked less than the last one.
        return (
            self.counts.get("failed", 0)
            + self.counts.get("error", self.counts.get("errors", 0))
            + len(self.skipped_chunks)
        )


def chunks_for(root: Path, path: str, ignore: Sequence[str] = ()) -> list[str]:
    """Split ``path`` into subtrees, each run in its own interpreter.

    One process cannot finish this suite on a 16 GB machine.  Resident
    size ratchets up and does not come back: over 321 model-zoo tests an
    explicit ``gc.collect()`` returned **1 MB** of RSS in total, so the
    growth is retention the process cannot undo, not garbage.  It reaches
    3.1 GB in the model zoo alone, and by the time the whole tree has run
    ahead of it macOS jetsam sends ``SIGKILL`` — the ``exited -9`` with no
    failing test and no traceback.

    Splitting is the only lever that costs no coverage.  Each chunk starts
    at ~50 MB, and the line data is combined afterwards, so the run
    measures exactly what a single process would have measured had it
    survived.  Skipping tests under pressure was the alternative and it
    buys the same memory by checking less.

    Directories are cut one level below any that has subdirectories, which
    puts ``unit/models`` — the heavy one — in a chunk of its own rather
    than inside a ``unit`` chunk that would reproduce the problem.
    """
    base = root / path
    if not base.is_dir():
        return [path]
    skip = {str(Path(s)) for s in ignore}

    def usable(folder: Path) -> bool:
        rel = str(folder.relative_to(root))
        if rel in skip or folder.name == "__pycache__":
            return False
        return any(folder.rglob("test_*.py"))

    def subdirs(folder: Path) -> list[Path]:
        return [d for d in sorted(folder.iterdir()) if d.is_dir() and usable(d)]

    def files(folder: Path) -> list[str]:
        return [str(f.relative_to(root)) for f in sorted(folder.glob("test_*.py"))]

    # Chunks must partition the tree, never overlap: naming a directory
    # *and* one of its descendants would run the descendant twice and
    # double its counts.  So a directory that gets split contributes its
    # subdirectories plus its own loose files — never itself.
    out: list[str] = []
    for child in subdirs(base):
        grand = subdirs(child)
        if grand:
            out.extend(str(g.relative_to(root)) for g in grand)
            out.extend(files(child))
        else:
            out.append(str(child.relative_to(root)))
    out.extend(files(base))
    return out or [path]


def run_suite(
    console: Console,
    path: str = "lucid/test",
    *,
    with_coverage: bool = True,
    root: Path | None = None,
    ignore: Sequence[str] | None = None,
) -> SuiteResult:
    """Run the suite in a subprocess and return what it established.

    A subprocess rather than ``pytest.main`` in-process: the audit has
    already imported the whole package and, on some axes, patched parts of
    it.  Collecting the suite into that interpreter would let one stage's
    state decide the other stage's result, which is the one thing a gate
    made of two independent checks must not allow.

    Parameters
    ----------
    console : Console
        Where progress is echoed while the child runs.
    path : str, optional
        What to hand pytest.  Default ``"lucid/test"`` — the whole tree.
    ignore : sequence of str, optional
        Subtrees to leave out, as ``--ignore=`` paths.  A gate is worth
        having only if it can be run, and the model-zoo parity suite is
        slow enough and optional enough that excluding it has to be one
        flag rather than a reason to skip the whole stage.
        ``pytest.ini_options`` already deselects the audit sweep (marked
        ``audit``) and the heavy models, so this is "everything except
        the stage that is running right now".
    with_coverage : bool, optional
        Measure line coverage alongside.  Default ``True``.
    root : Path, optional
        Working directory.  Defaults to the repository root inferred from
        this file.

    Returns
    -------
    SuiteResult
        ``ran=False`` with ``unavailable`` set when the suite could not be
        started at all — a wheel install, or ``coverage`` missing.
    """
    root = root or Path(__file__).resolve().parents[3]
    if not (root / path).exists():
        return SuiteResult(unavailable=f"{path} is not in this checkout")

    argv = [sys.executable]
    if with_coverage:
        try:
            import coverage  # noqa: F401
        except ImportError:
            with_coverage = False
            console.always(
                console.paint(
                    "  coverage is not installed — running without the line floor",
                    "grey",
                )
            )

    # The scratch files go to a system temporary directory rather than the
    # repository root.  Running the gate must leave the checkout exactly as
    # it found it: a stray ``.coverage.audit`` is picked up by the sync
    # harness, shows in ``git status`` next to real work, and survives any
    # run that is interrupted before the read-back.  ``parallel`` mode also
    # writes ``<data-file>.<host>.<pid>.<random>`` siblings that deleting
    # one known name would miss; a directory takes all of them with it.
    scratch = contextlib.ExitStack()
    folder: Path | None = None
    if with_coverage:
        folder = Path(scratch.enter_context(tempfile.TemporaryDirectory()))
    prefix = list(argv)
    pytest_args = [
        "-m",
        "pytest",
        *[f"--ignore={sub}" for sub in (ignore or ())],
        "-q",
        "-rfE",
        "--no-header",
        "-p",
        "no:cacheprovider",
        # The child colours its own output, so it has to be told what the
        # parent decided — otherwise ``--no-color`` produces a transcript
        # full of escape sequences, which is exactly what that flag is for.
        f"--color={'yes' if console.colour else 'no'}",
    ]

    chunks = chunks_for(root, path, ignore or ())
    scope = path + ("".join(f"  (without {sub})" for sub in (ignore or ())))
    console.always(
        console.paint(
            f"  running {scope} in {len(chunks)} chunk(s) — this is the long stage",
            "grey",
        )
    )

    started = time.time()
    result = SuiteResult(ran=True)
    with scratch:
        data_files: list[Path] = []
        for index, chunk in enumerate(chunks):
            reason = _afford(chunk, console)
            if reason is not None:
                result.skipped_chunks.append((chunk, reason))
                continue
            argv = list(prefix)
            if folder is not None:
                data = folder / f"data.{index}"
                data_files.append(data)
                argv += ["-m", "coverage", "run", f"--data-file={data}"]
            argv += [*pytest_args, chunk]
            _run_chunk(argv, root, chunk, result, console)
        result.duration = time.time() - started

        if folder is not None:
            _read_coverage(result, root, folder, data_files, console)
    return result


#: How long to let the OS take back a finished chunk's pages before
#: concluding there is no room for the next one.  Process exit frees them
#: eventually, not instantly, and deciding in that window would skip a
#: chunk that had memory waiting for it.
_SETTLE_S = 2.0


def _afford(chunk: str, console: Console) -> str | None:
    """Decide whether the machine can afford ``chunk``.

    The same three steps the per-test governor uses, at the level that
    now actually governs.  Reclaiming inside one process turned out to
    return almost nothing — an explicit ``gc.collect()`` gave back 1 MB
    across 321 model-zoo tests — so here step one is to let the *previous
    chunk's exit* land, which returns all of it.  Only if the machine is
    still short does the chunk get skipped, and it is always named.

    Returns the reason to skip, or ``None`` to run it.
    """
    from lucid.test import _memory

    if not _memory.ENABLED or _memory.FLOOR_MB <= 0:
        return None
    free = _memory.available_mb(force=True)
    if free < 0 or free >= _memory.FLOOR_MB:
        return None

    time.sleep(_SETTLE_S)
    free = _memory.available_mb(force=True)
    if free < 0 or free >= _memory.FLOOR_MB:
        return None

    console.always(
        console.paint(
            f"  skipping {chunk} — {free:.0f} MB available, floor is "
            f"{_memory.FLOOR_MB} MB (LUCID_TEST_MEM_FLOOR_MB)",
            "yellow",
        )
    )
    return f"{free:.0f} MB available, floor {_memory.FLOOR_MB} MB"


def _run_chunk(
    argv: list[str],
    root: Path,
    chunk: str,
    result: SuiteResult,
    console: Console,
) -> None:
    """Run one chunk and fold what it reported into ``result``.

    Counts are *summed* across chunks rather than overwritten — each
    child prints its own summary line, and keeping only the last would
    report the final chunk's handful of tests as the whole suite.
    """
    proc = subprocess.Popen(
        argv,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    counts: dict[str, int] = {}
    ended = False
    for line in proc.stdout:
        line = line.rstrip("\n")
        match = _FAILURE.match(line)
        if match:
            result.failures.append(match.group(1))
        found = _COUNT.findall(line)
        for count, label in found:
            if label.startswith("error"):
                label = "error"
            counts[label] = int(count)
        if found and _TERMINAL.search(line):
            ended = True
        if line.strip():
            # ``write`` rather than ``always``: under ``--quiet`` the
            # child's chatter is exactly what should disappear, and the
            # counts and failures below survive because they go through
            # ``always``.
            console.write(console.paint(f"  │ {line[:160]}", "grey"))
    code = proc.wait()
    for label, value in counts.items():
        result.counts[label] = result.counts.get(label, 0) + value

    # Liveness is judged on the exit code here, not on which words the
    # summary contained.  ``0`` is a complete run and ``5`` is "nothing
    # was collected", both of which a partitioned tree produces
    # legitimately — a parity chunk with the reference framework absent
    # skips every test and prints "16 skipped" with no passed, failed or
    # errored among them, which the passed/failed/error test read as a
    # chunk that had died.
    #
    # What cannot be waved through is a chunk that exited badly *and*
    # said nothing: that is the ``-9`` this whole split exists to
    # survive, and it has to be named rather than averaged away by the
    # sixty-odd chunks that did report.
    if code in (0, 5) or ended:
        if code not in (0, 5):
            result.returncode = code
        return
    result.unfinished.append(f"{chunk} (exit {code})")
    result.returncode = code


def _read_coverage(
    result: SuiteResult,
    root: Path,
    folder: Path,
    data_files: Sequence[Path],
    console: Console,
) -> None:
    """Combine the per-chunk data and turn it into covered/total counts.

    Everything lives in a temporary directory the caller owns, so none of
    it outlives the stage and none of it is written into the checkout.

    The combine step is what makes splitting free: each chunk measures
    only the lines its own tests reached, and the union of those is what
    a single process would have recorded had it survived to the end.
    """
    present = [f for f in data_files if f.exists()]
    if not present:
        console.always(console.paint("  no coverage data was produced", "grey"))
        return
    combined = folder / "combined"
    out = folder / "coverage.json"
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "combine",
                f"--data-file={combined}",
                *[str(f) for f in present],
            ],
            cwd=root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "json",
                f"--data-file={combined}",
                "-o",
                str(out),
                "-q",
            ],
            cwd=root,
            check=True,
            capture_output=True,
        )
        payload = json.loads(out.read_text())
    except (subprocess.CalledProcessError, OSError, json.JSONDecodeError) as exc:
        console.always(console.paint(f"  could not read the coverage data: {exc!r}", "grey"))
        return

    totals = payload["totals"]
    result.line_covered = totals["covered_lines"]
    result.line_total = totals["num_statements"]
    result.per_file = {
        name: (entry["summary"]["covered_lines"], entry["summary"]["num_statements"])
        for name, entry in payload["files"].items()
    }


# ── the recorded floor ───────────────────────────────────────────────────────


def load_floor(path: Path) -> dict[str, object] | None:
    """Read the recorded line-coverage floor, or ``None`` when absent."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def save_floor(path: Path, result: SuiteResult) -> None:
    """Record this run as the floor later runs are measured against."""
    payload = {
        "line_covered": result.line_covered,
        "line_total": result.line_total,
        "percent": result.percent,
        "files": {
            name: {"covered": covered, "total": total}
            for name, (covered, total) in sorted(result.per_file.items())
        },
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def report_line_coverage_diff(
    result: SuiteResult, floor: dict[str, object], console: Console
) -> int:
    """Compare against the floor and report what moved.

    Percentages rather than raw counts, because both numerator and
    denominator move legitimately: deleting dead code lowers the covered
    count without lowering coverage, and adding a well-tested module
    raises both.  A tolerance rather than equality, because a file that
    grows by a statement drops a fraction of a point honestly and a gate
    that fires on that gets muted.

    Returns
    -------
    int
        How many regressions were found.  Non-zero makes the run red.
    """
    now = result.percent
    was = floor.get("percent")
    if now is None or not isinstance(was, (int, float)):
        return 0

    recorded = floor.get("files")
    lost: list[tuple[str, float, float]] = []
    if isinstance(recorded, dict):
        for name, entry in recorded.items():
            if not isinstance(entry, dict):
                continue
            old_total = entry.get("total", 0)
            if not isinstance(old_total, int) or old_total < _MIN_STATEMENTS:
                continue
            old_covered = entry.get("covered", 0)
            assert isinstance(old_covered, int)
            before = 100.0 * old_covered / old_total
            current = result.per_file.get(name)
            if current is None:
                continue  # moved or deleted — not a coverage regression
            covered, total = current
            after = 100.0 * covered / total if total else 100.0
            if before - after > _FILE_TOLERANCE:
                lost.append((name, before, after))

    if not lost and now >= was - _TOTAL_TOLERANCE:
        console.always(
            console.paint(
                f"  line coverage {now:.2f}% — at or above the recorded " f"{was:.2f}%",
                "green",
            )
        )
        if now > was + _TOTAL_TOLERANCE:
            console.always(
                console.paint("  run with --update-suite to record the new floor", "grey")
            )
        return 0

    console.rule(f"line coverage regressions · {len(lost) or 1}", "red")
    if now < was - _TOTAL_TOLERANCE:
        console.always(console.paint(f"  TOTAL  {was:.2f}% → {now:.2f}%", "red"))
    for name, before, after in sorted(lost, key=lambda r: r[1] - r[2], reverse=True):
        console.always(console.paint(f"  LOST   {name}  {before:.1f}% → {after:.1f}%", "red"))
    console.always(
        console.paint(
            "  a module that stopped being exercised is what a refactor "
            "breaks without breaking a test",
            "grey",
        )
    )
    console.always(console.paint("  run with --update-suite to accept the new floor", "grey"))
    return len(lost) or 1


def report_suite(result: SuiteResult, console: Console) -> None:
    """Print what the suite established, defects first."""
    if not result.ran:
        console.always(console.paint(f"  suite not run: {result.unavailable}", "grey"))
        return

    counts = result.counts
    # Reported before the two "nothing established" branches below, not
    # after: when every chunk is skipped there are no counts either, and
    # returning early on that would print "did not finish" while hiding
    # the reason it did not.
    if result.skipped_chunks:
        console.always("")
        console.always(
            console.paint(
                f"  {len(result.skipped_chunks)} chunk(s) were SKIPPED for memory — "
                f"this run checked less than a full one:",
                "yellow",
                "bold",
            )
        )
        for name, why in result.skipped_chunks:
            console.always(console.paint(f"    {name}  ({why})", "yellow"))
    if result.unfinished:
        console.always("")
        console.always(
            console.paint(
                f"  {len(result.unfinished)} chunk(s) died without reporting a "
                f"result — this is not a clean run:",
                "red",
                "bold",
            )
        )
        for name in result.unfinished:
            console.always(console.paint(f"    {name}", "red"))
        return
    if not result.finished:
        console.always("")
        console.always(
            console.paint(
                f"  the suite did not finish — pytest exited {result.returncode} "
                f"after {result.duration:.0f}s without reporting a single "
                "passed, failed or errored test. Nothing was established; "
                "this is not a clean run.",
                "red",
                "bold",
            )
        )
        return
    parts = [f"{counts.get('passed', 0)} passed"]
    if result.broken:
        parts.append(f"{result.broken} failed")
    for label in ("skipped", "xfailed", "xpassed"):
        if counts.get(label):
            parts.append(f"{counts[label]} {label}")

    colour = "red" if result.broken else "green"
    console.always("")
    console.always(
        console.paint("  suite".ljust(22), "grey")
        + console.paint(", ".join(parts), colour)
        + console.paint(f"   {result.duration:.0f}s", "grey")
    )
    if result.percent is not None:
        console.always(
            console.paint("  line coverage".ljust(22), "grey")
            + console.paint(
                f"{result.percent:.2f}%  " f"({result.line_covered}/{result.line_total})",
                "cyan",
            )
        )

    if result.failures:
        console.rule(f"suite failures · {len(result.failures)}", "red")
        for nodeid in result.failures[:40]:
            console.always(console.paint(f"  FAIL  {nodeid}", "red"))
        if len(result.failures) > 40:
            console.always(console.paint(f"  … and {len(result.failures) - 40} more", "grey"))
    elif result.broken:
        # Counts said something broke but no nodeid was parsed — say so
        # rather than printing a clean list and implying there is none.
        console.always(
            console.paint(
                f"  {result.broken} failure(s) reported without a parseable "
                "node id — run pytest directly",
                "red",
            )
        )


__all__ = [
    "SuiteResult",
    "load_floor",
    "report_line_coverage_diff",
    "report_suite",
    "run_suite",
    "save_floor",
]
