"""Run test files one process each, with nothing filtered, and every skip
held to a reason.

The nightly suites exist so that no test goes unrun.  Three things used to let
one slip: a marker the default ``addopts`` deselects (``heavy``, ``control``),
an environment switch nobody set (``LUCID_TEST_NETWORK``), and a skip — an
oracle not installed, an API that drifted, a combination that does not apply.
This runs every given file with the markers cleared and the network on.

The skip rule.  A skip fails the run unless :data:`DESIGNED` lists it.  An
entry names the test file, a pattern the skip's reason matches, how many such
skips there are, and why the test cannot run there by design: a dtype the
device does not have, a question a case cannot ask.  Any other skip is a test
that has silently stopped running, and it fails the file with its reason.
The list is held both ways, so it cannot go stale:

* a skip in any file that matches none of that file's entries fails
  ("unexpected skip");
* an entry with a count fails when its skips number anything else — more
  means something new stopped running, fewer that the entry is stale.  A
  count of ``None`` is for a class of skip that is the design whatever its
  size, such as one per zoo family without the feature a test asks about.
  It is held one way only — a skip that joins the class is not seen — and
  that is the trade: a pinned count would fail on every family the zoo adds;
* an entry none of whose skips occurred fails: delete it.

Designed skips are listed at the end of the report, so a green run still
says what it did not run.  A file that runs no test at all fails.

Treating every skip as a failure kept the nightly red on files whose skips
were all by design (LCD-530), and a gate that is always red reports nothing.

One child process per file, because a single interpreter carrying the model
zoo or the parity tier from file to file outgrows a hosted runner's memory.

usage: python scripts/ci_nightly_suite.py PATH [PATH ...] [--exclude PATH ...]
       (a directory is expanded to its ``test_*.py`` files; ``--exclude``
       drops files so another job can run them — ci.yml gives the long
       training parity its own)
"""

import dataclasses
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SUMMARY = re.compile(r"^(?:=+ )?(\d+ .*?) in [0-9.]+s")
# ``-rs`` groups identical skips: ``SKIPPED [count] path:line: reason``.
_SKIP = re.compile(r"^SKIPPED \[(\d+)\] (\S+?)(?::\d+)?: (.*)$")
_TIMEOUT = 3600


@dataclasses.dataclass(frozen=True)
class Designed:
    """Skips that are the design of a test, not a test that stopped running."""

    #: Searched for in each skip's reason.
    reason: str
    #: How many skips match; ``None`` when the class is the design at any size.
    count: int | None
    #: Why the test cannot run there — what a reader of the report is told.
    why: str


#: Test file (repository-relative) → the skips it is designed to make.
DESIGNED: dict[str, tuple[Designed, ...]] = {
    "lucid/test/parity/models/vision/test_resnet.py": (
        Designed(
            r"^the reference answers with \|logits\| up to \S+, far past the ",
            3,
            "three deep variants amplify the randomised norm buffers past where "
            "an elementwise comparison describes the model "
            "(lucid/test/parity/models/_utils.py)",
        ),
    ),
    "lucid/test/parity/nn/test_interpolate_sample_positions_parity.py": (
        Designed(r"^Metal has no float64$", None, "every float64 case runs on the CPU only"),
    ),
    "lucid/test/unit/models/test_models_inplace_independence.py": (
        # None, deliberately, and so held one way only: a family that loses
        # its last inplace module starts skipping unseen.  A pinned count, or
        # a pinned list of families, would instead fail on every family added
        # to the zoo without one and copy the zoo's roster into a CI script;
        # the test itself still checks every family that has one.
        Designed(
            r"^\w+: no module built with inplace=True$",
            None,
            "the test asks what inplace=False changes; a family with no inplace "
            "module has nothing to change",
        ),
    ),
}


def _files(args: list[str]) -> list[Path]:
    found: list[Path] = []
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            found.extend(sorted(path.rglob("test_*.py")))
        else:
            found.append(path)
    return found


def _relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(_ROOT):
        return resolved.relative_to(_ROOT).as_posix()
    return path.as_posix()


def skips_of(lines: list[str]) -> list[tuple[int, str]]:
    """``(count, reason)`` for every skip line of a ``-rs`` report."""
    return [(int(m.group(1)), m.group(3)) for ln in lines if (m := _SKIP.match(ln))]


def judge_skips(
    path: str, skips: list[tuple[int, str]], designed: dict[str, tuple[Designed, ...]] | None = None
) -> tuple[list[str], list[str]]:
    """``(problems, designed lines)`` for one file's skips.

    No problems means every skip is one ``designed`` (by default
    :data:`DESIGNED`) lists for ``path``, in the number it lists; the
    designed lines then say what did not run.
    """
    entries = (DESIGNED if designed is None else designed).get(path, ())
    seen = [0] * len(entries)
    problems: list[str] = []
    for count, reason in skips:
        hit = next((i for i, e in enumerate(entries) if re.search(e.reason, reason)), None)
        if hit is None:
            problems.append(f"unexpected skip ({count}): {reason}")
        else:
            seen[hit] += count
    lines: list[str] = []
    for entry, got in zip(entries, seen, strict=True):
        if got == 0:
            problems.append(f"designed skip no longer happens — delete its entry: {entry.reason!r}")
        elif entry.count is not None and got != entry.count:
            problems.append(
                f"{got} skips match {entry.reason!r}, {entry.count} designed — "
                "a new skip is a test that stopped running; fewer, a stale count"
            )
        else:
            lines.append(f"{got} × {entry.reason!r} — {entry.why}")
    return problems, lines


def _run(path: Path) -> tuple[str, list[str], float]:
    """Run one file; return (verdict, detail lines, seconds)."""
    env = dict(os.environ, LUCID_TEST_NETWORK="1")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(path),
        "-q",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "-rsfE",
        "--tb=short",
        "--color=no",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return "timeout", [f"exceeded {_TIMEOUT}s"], time.time() - t0
    secs = time.time() - t0
    lines = proc.stdout.splitlines()
    fails = [ln for ln in lines if ln.startswith(("FAILED", "ERROR"))]
    summary = next((m.group(1) for ln in reversed(lines) if (m := _SUMMARY.search(ln))), "")
    if proc.returncode not in (0, 5) or fails:
        detail = fails or lines[-25:]
        return "fail", [summary, *detail], secs
    if proc.returncode == 5:
        return "skip", [summary, "no test ran"], secs
    problems, designed = judge_skips(_relative(path), skips_of(lines))
    if problems:
        return "skip", [summary, *problems], secs
    return "ok", [summary, *designed], secs


def main() -> int:
    args = sys.argv[1:]
    excluded: set[Path] = set()
    if "--exclude" in args:
        cut = args.index("--exclude")
        args, excluded = args[:cut], {p.resolve() for p in _files(args[cut + 1 :])}
    files = [p for p in _files(args) if p.resolve() not in excluded]
    if not files:
        # An empty selection reporting "0 not clean" would pass a job that
        # ran nothing.
        print("no test files selected", file=sys.stderr)
        return 1
    bad: list[tuple[Path, str, list[str]]] = []
    designed: list[tuple[Path, list[str]]] = []
    started = time.time()
    flaky: list[Path] = []
    for path in files:
        verdict, detail, secs = _run(path)
        if verdict in ("fail", "timeout"):
            # Once more before calling it: a first download of a large
            # checkpoint can stall past the reader's timeout and pass on the
            # next try.  Listed as flaky either way, never hidden.
            verdict2, detail2, secs2 = _run(path)
            if verdict2 == "ok":
                flaky.append(path)
                verdict, detail, secs = "flaky", [detail2[0], *detail], secs + secs2
            else:
                verdict, detail, secs = verdict2, detail2, secs + secs2
        print(f"{verdict:7s} {secs:7.1f}s {path} — {detail[0] if detail else ''}", flush=True)
        if verdict not in ("ok", "flaky"):
            bad.append((path, verdict, detail))
        elif verdict == "ok" and len(detail) > 1:
            designed.append((path, detail[1:]))
    print(f"\n{len(files)} files in {(time.time() - started) / 60:.1f} min, {len(bad)} not clean")
    for path, lines in designed:
        print(f"\nDESIGNED SKIPS: {path}")
        for line in lines:
            print(f"    {line}")
    for path in flaky:
        print(f"FLAKY (failed, then passed on retry): {path}")
    for path, verdict, detail in bad:
        print(f"\n{verdict.upper()}: {path}")
        for line in detail[1:]:
            print(f"    {line}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
