"""The model zoo as a stage of the audit: every family, one process at a time.

CI runs the zoo's tests in pieces, on shrunken configs, and a cancelled nightly
leaves no record of how far it got.  This runs everything the zoo has — the
contract validator, a real-size build of every factory, each family's own test
files, the zoo-wide tests and the compiled-training matrix — strictly one
process at a time, with the markers cleared and the network off, so nothing is
downloaded and nothing is kept.  After each family it records the result for
this commit, so an interrupted run resumes and a finished one is not repeated.

It replaces ``tools/verify_zoo.py``.  What changed is the owner of the numbers:
the memory limits come from the machine (:class:`~lucid.test.audit._zoo_budget.MachineBudget`)
instead of a constant tuned on one Mac Studio, and whatever a small machine
therefore skips is *said* — in the display, in the family's note and in the
verdict — rather than left to look like a pass.

    lucid-audit --models                      # the whole zoo
    lucid-audit --models --family lenet       # one family (repeatable)
    lucid-audit --models --zoo-record DIR     # where the notes go
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from lucid.test.audit import _zoo_record, _zoo_report
from lucid.test.audit._console import fmt_duration
from lucid.test.audit._plan import PlanError
from lucid.test.audit._zoo_budget import HEAVY_HEADROOM, HEAVY_PEAK_GB, MachineBudget
from lucid.test.audit._zoo_guard import POLL_S, StageGuard, StageRunner
from lucid.test.audit._zoo_stage import ZOO_WIDE, Stage, StageResult, ZooCatalog

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterable, Sequence

    from lucid.test.audit._console import Console
    from lucid.test.audit._zoo_report import ZooReporter


def add_options(p: argparse.ArgumentParser) -> None:
    """The zoo stage's command-line options, as an argument group of ``p``."""
    zoo = p.add_argument_group(
        "model zoo", "the zoo stage runs under --models, and by default; --no-models leaves it out"
    )
    zoo.add_argument(
        "--family",
        action="append",
        default=[],
        metavar="NAME",
        help="only this family, or zoo-wide for the multi-family files; repeatable",
    )
    zoo.add_argument(
        "--zoo-record",
        type=Path,
        default=None,
        metavar="DIR",
        help="write zoo-verify-<family>.md and the ledger here "
        "(default: obsidian/zoo-verify when this checkout has it, else no notes)",
    )
    zoo.add_argument(
        "--fresh",
        action="store_true",
        help="run every family again even if this commit already recorded it",
    )
    zoo.add_argument(
        "--mem-limit-gb",
        type=float,
        default=None,
        metavar="GB",
        help="footprint that kills a zoo stage (default: derived from this machine's memory)",
    )


@dataclass(frozen=True)
class ZooSettings:
    """What the command line decided about the zoo stage."""

    families: tuple[str, ...] = ()
    limit: int = 0
    fresh: bool = False
    record_dir: Path | None = None
    mem_limit_gb: float | None = None
    command: str = "lucid-audit --models"
    state_root: Path = _zoo_record.STATE_ROOT
    poll_s: float = POLL_S

    @classmethod
    def from_args(cls, args: argparse.Namespace, root: Path, command: str) -> ZooSettings:
        vault = root / "obsidian" / "zoo-verify"
        record = args.zoo_record or (vault if vault.is_dir() else None)
        return cls(
            families=tuple(args.family or ()),
            limit=args.limit,
            fresh=args.fresh,
            record_dir=record,
            mem_limit_gb=args.mem_limit_gb,
            command=command,
        )


@dataclass
class ZooReport:
    """What the stage established, in the form the verdict reads."""

    groups: int = 0
    resumed: int = 0
    seconds: float = 0.0
    peak_gb: float = 0.0
    failed: list[tuple[str, StageResult]] = field(default_factory=list)
    adapted: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def failures(self) -> int:
        """Stages that failed, a killed stage counting unless its retry passed."""
        return len(self.failed)

    def add(self, group: str, results: Sequence[StageResult], *, resumed: bool) -> None:
        self.groups += 1
        self.resumed += resumed
        for result in results:
            self.peak_gb = max(self.peak_gb, result.peak_gb)
            if result.failed:
                self.failed.append((group, result))
            self.adapted.extend((group, result.name, note) for note in result.notes)


def validate_families(root: Path, names: Iterable[str]) -> None:
    """Refuse a family name that is not one, before any other stage has run for hours."""
    known = {*ZooCatalog(root).families(), ZOO_WIDE}
    unknown = sorted(set(names) - known)
    if unknown:
        raise PlanError(
            f"--family: no such model family {', '.join(unknown)}; "
            f"{len(known) - 1} families are known (lucid-audit --list-packages names the zoo)"
        )


class ZooRun:
    """One execution of the zoo stage."""

    def __init__(
        self,
        root: Path,
        console: Console,
        settings: ZooSettings,
        *,
        budget: MachineBudget | None = None,
        reporter: ZooReporter | None = None,
        catalog: ZooCatalog | None = None,
        revision: tuple[str, bool] | None = None,
    ) -> None:
        self.root = root
        self.console = console
        self.settings = settings
        self.budget = budget or MachineBudget.detect(settings.mem_limit_gb)
        self._reporter = reporter
        self._catalog = catalog or ZooCatalog(root)
        self._revision = revision

    def _groups(self, families: dict[str, str]) -> list[str]:
        names = list(self.settings.families) or [*families, ZOO_WIDE]
        return names[: self.settings.limit] if self.settings.limit else names

    def _stages(self, group: str, catalog: ZooCatalog, temp: Path) -> list[Stage]:
        stages = (
            catalog.wide_stages(temp)
            if group == ZOO_WIDE
            else catalog.family_stages(group, temp, self.budget)
        )
        if self.budget.admits_heavy:
            return stages
        why = (
            f"heavy tests deselected up front: the footprint limit here is "
            f"{self.budget.limit_gb:g} GB and the largest heavy test peaks at {HEAVY_PEAK_GB:g} GB "
            f"(needs {HEAVY_PEAK_GB * HEAVY_HEADROOM:g} GB of limit)"
        )
        return [stage.without_heavy(why) or stage for stage in stages]

    def run(self) -> ZooReport:
        sha, dirty = self._revision or _zoo_record.repo_state(self.root)
        state = _zoo_record.ZooState(
            self.settings.state_root, sha, dirty=dirty, resume=not self.settings.fresh
        )
        catalog = self._catalog
        families = catalog.families()
        groups = self._groups(families)
        facts = _zoo_record.RunFacts(sha, self.settings.command, self.budget, dirty)
        notebook = None
        if self.settings.record_dir is not None:
            notebook = _zoo_record.Notebook(self.settings.record_dir, state, families, facts)

        self.console.banner(
            "Model zoo", f"{len(groups)} group(s) · {self.budget.total_gb:.0f} GB machine"
        )
        for key, line in zip(("limits", "adapts"), self.budget.limits(), strict=True):
            self.console.kv(key, line)
        if dirty:
            self.console.always(
                self.console.paint("  uncommitted changes: no earlier record is reused", "yellow")
            )
        reporter = self._reporter or _zoo_report.build_reporter(self.console, self.budget.limit_gb)
        report = ZooReport()
        began = time.monotonic()
        with reporter:
            reporter.start(len(groups))
            guard = StageGuard(
                self.root, self.budget, on_tick=reporter.tick, poll_s=self.settings.poll_s
            )
            runner = StageRunner(guard, state.logs, state.temp)
            for index, group in enumerate(groups, start=1):
                reporter.family(index, group)
                results = state.load(group)
                resumed = results is not None
                if results is None:
                    results = self._run_group(group, catalog, state, runner, reporter)
                    state.save(group, results)
                    if notebook is not None:
                        notebook.write(group, results)
                reporter.family_done(group, results, resumed=resumed)
                report.add(group, results, resumed=resumed)
        report.seconds = time.monotonic() - began
        if not self.settings.families and not self.settings.limit:
            (state.root / "DONE").write_text(sha + "\n")
        return report

    def _run_group(
        self,
        group: str,
        catalog: ZooCatalog,
        state: _zoo_record.ZooState,
        runner: StageRunner,
        reporter: ZooReporter,
    ) -> list[StageResult]:
        results = []
        for stage in self._stages(group, catalog, state.temp):
            reporter.stage(stage.name)
            result = runner.run(stage)
            reporter.stage_done(result)
            results.append(result)
        return results


def summarise(console: Console, report: ZooReport, budget: MachineBudget) -> None:
    """The stage's closing lines: what failed, and everything this machine skipped."""
    console.always("")
    console.rule("model zoo", "red" if report.failures else "cyan")
    console.kv("groups", f"{report.groups} ({report.resumed} recorded for this commit)")
    console.kv("peak footprint", f"{report.peak_gb:.1f} GB of {budget.limit_gb:g} GB allowed")
    console.kv("elapsed", fmt_duration(report.seconds))
    if report.adapted:
        console.always("")
        console.rule(f"skipped or adapted for this machine · {len(report.adapted)}", "yellow")
        for group, stage, note in report.adapted[:20]:
            console.always("  " + console.paint(f"{group}/{stage}", "yellow") + f"  {note}")
        if len(report.adapted) > 20:
            console.always(console.paint(f"  ... and {len(report.adapted) - 20} more", "grey"))
    if report.failed:
        console.always("")
        console.rule(f"failing stages · {len(report.failed)}", "red")
        for group, result in report.failed[:40]:
            first = result.failures[0] if result.failures else ""
            console.always(
                "  "
                + console.paint("FAIL", "red", "bold")
                + f"  {group}/{result.name}  ({result.status})  {first[:100]}"
            )
        if len(report.failed) > 40:
            console.always(console.paint(f"  ... and {len(report.failed) - 40} more", "grey"))


def run_zoo_stage(args: argparse.Namespace, console: Console, root: Path, command: str) -> int:
    """Run the zoo stage for ``args``.  Returns the number of failed stages."""
    run = ZooRun(root, console, ZooSettings.from_args(args, root, command))
    report = run.run()
    summarise(console, report, run.budget)
    return report.failures


__all__ = ["ZooReport", "ZooRun", "ZooSettings", "run_zoo_stage", "summarise", "validate_families"]
