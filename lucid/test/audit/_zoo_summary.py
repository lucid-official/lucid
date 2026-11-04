"""What the zoo stage reports at the end, short on screen and complete on disk.

A full run has seventy-odd families.  When most of them fail, listing every
failing stage in the verdict scrolls the numbers that matter off the screen —
and cutting the list at an arbitrary row ("... and 79 more") loses the rest.  So
the screen carries the counts, labelled with what they count, and the first few
rows; the complete list is a file, and the screen names it.

What the numbers count:

``failing stages``
    stages whose status is a failure (``fail``, ``timeout``, ``mem-guard``,
    ``machine-short``).  This is the number the verdict adds to the exit code.
    One family usually has several (its tests, its probe), so it is larger than
    the number of families — and it is not a count of failing *tests*, which only
    each stage's own log has.
``failing families``
    groups with at least one failing stage.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from lucid.test.audit import _zoo_record
from lucid.test.audit._console import fmt_duration

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lucid.test.audit._console import Console
    from lucid.test.audit._zoo_budget import MachineBudget
    from lucid.test.audit._zoo_stage import StageResult

#: Rows shown on screen; the file has them all.
SHOWN = 8


@dataclass
class ZooReport:
    """What the stage established, in the form the verdict reads."""

    groups: int = 0
    resumed: int = 0
    seconds: float = 0.0
    peak_gb: float = 0.0
    failed: list[tuple[str, StageResult]] = field(default_factory=list)
    adapted: list[tuple[str, str, str]] = field(default_factory=list)
    #: Groups that passed what they were asked and were asked for less than the
    #: zoo's full verification (see :func:`~lucid.test.audit._zoo_record.overall`).
    reduced: list[str] = field(default_factory=list)
    notes_written: int = 0
    #: Where the complete lists were written, once the run has finished.
    summary_path: Path | None = None

    @property
    def failures(self) -> int:
        """Stages that failed, a killed stage counting unless its retry passed."""
        return len(self.failed)

    @property
    def failing_groups(self) -> dict[str, list[StageResult]]:
        found: dict[str, list[StageResult]] = {}
        for group, result in self.failed:
            found.setdefault(group, []).append(result)
        return found

    def add(self, group: str, results: Sequence[StageResult], *, resumed: bool) -> None:
        self.groups += 1
        self.resumed += resumed
        if _zoo_record.overall(results) == "reduced":
            self.reduced.append(group)
        for result in results:
            self.peak_gb = max(self.peak_gb, result.peak_gb)
            if result.failed:
                self.failed.append((group, result))
            self.adapted.extend((group, result.name, note) for note in result.notes + result.events)


def _first_failure(results: Sequence[StageResult]) -> str:
    for result in results:
        if result.failures:
            return result.failures[0]
    return ""


def render_summary(report: ZooReport) -> str:
    """The complete lists, as the file holds them."""
    groups = report.failing_groups
    lines = [
        f"failing stages {report.failures} in {len(groups)} of {report.groups} group(s)",
        "(a stage is one process: its tests, a probe, the contract; for the failing tests "
        "of a stage read logs/<group>/<stage>.log)",
        "",
        f"== failing stages · {report.failures}",
    ]
    for group, result in report.failed:
        first = result.failures[0] if result.failures else ""
        lines.append(f"{group}/{result.name}  ({result.status})  {first}")
    lines += ["", f"== groups that passed in reduced form · {len(report.reduced)}"]
    lines += list(report.reduced)
    lines += ["", f"== skipped or adapted for this machine · {len(report.adapted)}"]
    lines += [f"{group}/{stage}  {note}" for group, stage, note in report.adapted]
    return "\n".join(lines) + "\n"


def summarise(
    console: Console, report: ZooReport, budget: MachineBudget, record_dir: Path | None = None
) -> None:
    """The stage's closing lines: the counts, the first few rows, and where the rest is."""
    groups = report.failing_groups
    console.always("")
    console.rule("model zoo", "red" if report.failures else "cyan")
    console.kv("groups", f"{report.groups} ({report.resumed} recorded for this commit)")
    if report.failures:
        console.kv(
            "failing stages",
            f"{report.failures} in {len(groups)} of {report.groups} group(s)  "
            "(stages, not tests: a stage is a process)",
        )
    console.kv("peak footprint", f"{report.peak_gb:.1f} GB of {budget.limit_gb:g} GB allowed")
    console.kv("elapsed", fmt_duration(report.seconds))
    if record_dir is None:
        console.always(
            console.paint(
                "  no notes written: this checkout has no obsidian/zoo-verify; "
                "pass --zoo-record DIR to keep them",
                "yellow",
            )
        )
    if report.reduced:
        shown = ", ".join(report.reduced[:SHOWN])
        more = f" and {len(report.reduced) - SHOWN} more" if len(report.reduced) > SHOWN else ""
        console.always(
            console.paint(
                f"  {len(report.reduced)} group(s) passed in reduced form: {shown}{more}", "yellow"
            )
        )
    if report.adapted:
        console.always("")
        console.rule(f"skipped or adapted for this machine · {len(report.adapted)}", "yellow")
        for group, stage, note in report.adapted[:SHOWN]:
            console.always("  " + console.paint(f"{group}/{stage}", "yellow") + f"  {note[:110]}")
    if groups:
        console.always("")
        console.rule(f"failing groups · {len(groups)}", "red")
        for group, results in list(groups.items())[:SHOWN]:
            stages = ", ".join(f"{r.name} ({r.status})" for r in results)
            console.always(
                "  "
                + console.paint("FAIL", "red", "bold")
                + f"  {group}: {stages}  {_first_failure(results)[:70]}"
            )
    hidden = max(len(groups) - SHOWN, 0) + max(len(report.adapted) - SHOWN, 0)
    if hidden and report.summary_path is not None:
        console.always("")
        console.always(
            console.paint(f"  the complete lists: {report.summary_path}", "grey")
            + console.paint("   (cat it; each stage's own log is beside it in logs/)", "grey")
        )


__all__ = ["SHOWN", "ZooReport", "render_summary", "summarise"]
