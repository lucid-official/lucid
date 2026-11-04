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

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from lucid.test._fixtures.devices import metal_available
from lucid.test.audit import _zoo_record, _zoo_report
from lucid.test.audit._console import fmt_duration
from lucid.test.audit._plan import PlanError
from lucid.test.audit._zoo_budget import HEAVY_HEADROOM, HEAVY_PEAK_GB, MachineBudget
from lucid.test.audit._zoo_guard import POLL_S, StageGuard, StageRunner, exit_on_terminate
from lucid.test.audit._zoo_stage import ZOO_WIDE, Stage, StageResult, ZooCatalog

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterable, Sequence

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
        "--zoo-device",
        choices=("metal", "cpu"),
        default="metal",
        help="the device every stage runs on (default metal: models train there). "
        "A machine without Metal is an error, not a CPU run",
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
    device: str = "metal"

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
            device=args.zoo_device,
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
    #: Groups that passed what they were asked and were asked for less than the
    #: zoo's full verification (see :func:`~lucid.test.audit._zoo_record.overall`).
    reduced: list[str] = field(default_factory=list)
    notes_written: int = 0

    @property
    def failures(self) -> int:
        """Stages that failed, a killed stage counting unless its retry passed."""
        return len(self.failed)

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


def validate_options(root: Path, args: argparse.Namespace) -> None:
    """Refuse a zoo option that cannot work, before any other stage has run for hours.

    ``--mem-limit-gb`` would otherwise reach :class:`MachineBudget` hours in, and
    ``--zoo-record`` would fail after the first family had finished — the one
    moment a run has the most to lose.
    """
    validate_families(root, args.family)
    if args.zoo_device == "metal" and not metal_available():
        raise PlanError(
            "--zoo-device metal: this machine has no usable Metal device. "
            "Pass --zoo-device cpu to verify the zoo on CPU on purpose"
        )
    if args.mem_limit_gb is not None and not args.mem_limit_gb > 0:
        raise PlanError(f"--mem-limit-gb must be positive, got {args.mem_limit_gb:g}")
    if args.zoo_record is not None:
        record = Path(args.zoo_record)
        if record.exists() and not record.is_dir():
            raise PlanError(f"--zoo-record {record} is a file, not a directory")
        nearest = next((p for p in (record, *record.parents) if p.exists()), None)
        if nearest is None or not nearest.is_dir() or not os.access(nearest, os.W_OK):
            raise PlanError(f"--zoo-record {record} cannot be created: no writable parent")


def validate_families(root: Path, names: Iterable[str]) -> None:
    """Refuse a family name that is not one, before any other stage has run for hours."""
    known = {*ZooCatalog(root).families(), ZOO_WIDE}
    unknown = sorted(set(names) - known)
    if unknown:
        raise PlanError(
            f"--family: no such model family {', '.join(unknown)}; "
            f"{len(known) - 1} families are known (lucid-audit --list-packages names the zoo)"
        )


class TreeMoved(RuntimeError):
    """The checkout changed while the zoo stage was running."""


class ZooRun:
    """One execution of the zoo stage.

    Run it from a scratch clone, not a checkout people are working in: it reads
    the test tree for hours, and results can only be filed under the commit the
    tree was at when it *finished* a family.  A tree whose HEAD or working copy
    changes mid-run stops the run (:class:`TreeMoved`) rather than filing later
    results under the earlier commit.
    """

    def __init__(
        self,
        root: Path,
        console: Console,
        settings: ZooSettings,
        *,
        budget: MachineBudget | None = None,
        reporter: ZooReporter | None = None,
        catalog: ZooCatalog | None = None,
        revision: Callable[[], tuple[str, bool]] | None = None,
    ) -> None:
        self.root = root
        self.console = console
        self.settings = settings
        self.budget = budget or MachineBudget.detect(settings.mem_limit_gb)
        self._reporter = reporter
        self._catalog = catalog or ZooCatalog(root)
        self._revision = revision or (lambda: _zoo_record.repo_state(root))

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

    def _still_at(self, sha: str, dirty: bool) -> None:
        """Stop if the tree is no longer the one this run began on."""
        now = self._revision()
        if now != (sha, dirty):
            raise TreeMoved(
                f"the tree changed during the run (was {sha[:9]}{' dirty' if dirty else ''}, "
                f"now {now[0][:9]}{' dirty' if now[1] else ''}); results already recorded stand, "
                "later ones would be filed under the wrong commit. Run from a scratch clone."
            )

    def run(self) -> ZooReport:
        sha, dirty = self._revision()
        state = _zoo_record.ZooState(
            self.settings.state_root, sha, dirty=dirty, resume=not self.settings.fresh
        )
        catalog = self._catalog
        families = catalog.families()
        groups = self._groups(families)
        device = self.settings.device
        facts = _zoo_record.RunFacts(sha, self.settings.command, self.budget, dirty, device)
        notebook = None
        if self.settings.record_dir is not None:
            notebook = _zoo_record.Notebook(self.settings.record_dir, state, families, facts)

        self.console.banner(
            "Model zoo", f"{len(groups)} group(s) · {self.budget.total_gb:.0f} GB machine"
        )
        self.console.kv("device", device)
        for key, line in zip(("limits", "adapts"), self.budget.limits(), strict=True):
            self.console.kv(key, line)
        if dirty:
            self.console.always(
                self.console.paint("  uncommitted changes: no earlier record is reused", "yellow")
            )
        reporter = self._reporter or _zoo_report.build_reporter(self.console, self.budget.limit_gb)
        report = ZooReport()
        began = time.monotonic()
        with exit_on_terminate(), state.lock(), reporter:
            reporter.start(len(groups))
            guard = StageGuard(
                self.root,
                self.budget,
                on_tick=reporter.tick,
                on_wait=reporter.wait,
                poll_s=self.settings.poll_s,
                device=device,
            )
            runner = StageRunner(guard, state.logs, state.temp)
            for index, group in enumerate(groups, start=1):
                self._still_at(sha, dirty)
                reporter.family(index, group)
                results = state.load(group, self.budget, device)
                resumed = results is not None
                if results is None:
                    results = self._run_group(group, catalog, state, runner, reporter)
                    self._still_at(sha, dirty)
                    state.save(group, results, self.budget, device)
                    if notebook is not None:
                        notebook.write(group, results)
                        report.notes_written += 1
                elif notebook is not None and not notebook.note_path(group).exists():
                    notebook.write_note(group, results)
                    report.notes_written += 1
                reporter.family_done(group, results, resumed=resumed)
                report.add(group, results, resumed=resumed)
            if notebook is not None:
                notebook.write_ledger()
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
            result = runner.run(stage, group)
            reporter.stage_done(result)
            results.append(result)
        return results


def summarise(
    console: Console, report: ZooReport, budget: MachineBudget, record_dir: Path | None = None
) -> None:
    """The stage's closing lines: what failed, and everything this machine skipped."""
    console.always("")
    console.rule("model zoo", "red" if report.failures else "cyan")
    console.kv("groups", f"{report.groups} ({report.resumed} recorded for this commit)")
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
        console.always(
            console.paint(
                f"  {len(report.reduced)} group(s) passed in reduced form (see below): "
                + ", ".join(report.reduced[:12])
                + (" ..." if len(report.reduced) > 12 else ""),
                "yellow",
            )
        )
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
    settings = ZooSettings.from_args(args, root, command)
    run = ZooRun(root, console, settings)
    report = run.run()
    summarise(console, report, run.budget, settings.record_dir)
    return report.failures


__all__ = [
    "TreeMoved",
    "ZooReport",
    "ZooRun",
    "ZooSettings",
    "add_options",
    "run_zoo_stage",
    "summarise",
    "validate_families",
    "validate_options",
]
