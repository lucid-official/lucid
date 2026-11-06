"""The model zoo as a stage of the audit: every family, one process at a time.

CI runs the zoo's tests in pieces, on shrunken configs, and a cancelled nightly
leaves no record of how far it got.  This runs everything the zoo has — the
contract validator, a real-size build of every factory, each family's own test
files, the zoo-wide tests and the compiled-training matrix — strictly one
process at a time, with the markers cleared and the network off, so nothing is
downloaded and nothing is kept.  After each family it records the result for
this commit, so an interrupted run resumes and a finished one is not repeated.

The zoo is a set of deep-learning models, so by design it is verified on both
devices a model runs on: Metal first (where models train, and the quick pass),
then the CPU.  ``--zoo-device`` narrows that on purpose.  Each stage declares
which devices it is for (:mod:`~lucid.test.audit._zoo_device`), results are kept
per (family, device), and the verdict is the worse of the two.

It replaces ``tools/verify_zoo.py``.  What changed is the owner of the numbers:
the memory limits come from the machine (:class:`~lucid.test.audit._zoo_budget.MachineBudget`)
instead of a constant tuned on one Mac Studio, and whatever a small machine
therefore skips is *said* — in the display, in the family's note and in the
verdict — rather than left to look like a pass.

    lucid-audit --models                      # the whole zoo
    lucid-audit --models --family lenet       # one family (repeatable)
    lucid-audit --models --zoo-record DIR     # where the notes go
    lucid-audit --models --zoo-device cpu     # one device, on purpose
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lucid.test._default_device import metal_available
from lucid.test.audit import _zoo_record, _zoo_report
from lucid.test.audit._plan import PlanError
from lucid.test.audit._zoo_budget import HEAVY_HEADROOM, HEAVY_PEAK_GB, MachineBudget
from lucid.test.audit._zoo_device import CHOICES, DEFAULT_CHOICE, ORDER, devices_for
from lucid.test.audit._zoo_guard import POLL_S, StageGuard, StageRunner, exit_on_terminate
from lucid.test.audit._zoo_stage import ZOO_WIDE, Stage, StageResult, ZooCatalog
from lucid.test.audit._zoo_summary import ZooReport, render_summary, summarise

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterable

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
        choices=CHOICES,
        default=None,
        help="the devices the zoo is verified on (default both: Metal first, where models "
        "train, then the CPU; metal or cpu narrows it on purpose). Each stage says which "
        "devices it is for: the contract runs once, compiled training is Metal only, tests "
        "and real-size builds run per device. A machine without Metal is an error, not a "
        "CPU run; pass --zoo-device cpu there",
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
    #: What ``--zoo-device`` asked for: ``both``, ``metal`` or ``cpu``.
    device: str = DEFAULT_CHOICE

    @property
    def devices(self) -> tuple[str, ...]:
        """The devices the run takes, in the order it takes them."""
        return devices_for(self.device)

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
            device=args.zoo_device or DEFAULT_CHOICE,
        )


def validate_options(root: Path, args: argparse.Namespace) -> None:
    """Refuse a zoo option that cannot work, before any other stage has run for hours.

    ``--mem-limit-gb`` would otherwise reach :class:`MachineBudget` hours in, and
    ``--zoo-record`` would fail after the first family had finished — the one
    moment a run has the most to lose.
    """
    validate_families(root, args.family)
    choice = args.zoo_device or DEFAULT_CHOICE
    if "metal" in devices_for(choice) and not metal_available():
        given = "" if args.zoo_device else " (the default)"
        raise PlanError(
            f"--zoo-device {choice}{given}: this machine has no usable Metal device. "
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

    def _stages(self, group: str, catalog: ZooCatalog, temp: Path, device: str) -> list[Stage]:
        """The stages ``group`` goes through in the pass for ``device``: those whose scope covers it."""
        stages = (
            catalog.wide_stages(temp)
            if group == ZOO_WIDE
            else catalog.family_stages(group, temp, self.budget)
        )
        stages = [s for s in stages if s.scope.covers(device, self.settings.devices)]
        if self.budget.admits_heavy:
            return stages
        why = (
            f"heavy tests deselected up front on {device}: the footprint limit here is "
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
        devices = self.settings.devices
        facts = _zoo_record.RunFacts(
            sha, self.settings.command, self.budget, dirty, self.settings.device
        )
        notebook = None
        if self.settings.record_dir is not None:
            notebook = _zoo_record.Notebook(self.settings.record_dir, state, families, facts)

        self.console.banner(
            "Model zoo", f"{len(groups)} group(s) · {self.budget.total_gb:.0f} GB machine"
        )
        self.console.kv("devices", " → ".join(devices))
        for key, line in zip(("limits", "adapts"), self.budget.limits(), strict=True):
            self.console.kv(key, line)
        if dirty:
            self.console.always(
                self.console.paint("  uncommitted changes: no earlier record is reused", "yellow")
            )
        reporter = self._reporter or _zoo_report.build_reporter(self.console, self.budget.limit_gb)
        report = ZooReport(devices=devices)
        merged: dict[str, list[StageResult]] = {group: [] for group in groups}
        recorded: dict[str, bool] = {group: True for group in groups}
        began = time.monotonic()
        with exit_on_terminate(), state.lock(), reporter:
            reporter.start(len(groups))
            for device in devices:
                runner = StageRunner(
                    StageGuard(
                        self.root,
                        self.budget,
                        on_tick=reporter.tick,
                        on_wait=reporter.wait,
                        poll_s=self.settings.poll_s,
                        device=device,
                    ),
                    state.logs,
                    state.temp,
                )
                reporter.begin_device(device)
                for index, group in enumerate(groups, start=1):
                    self._still_at(sha, dirty)
                    reporter.family(index, group)
                    results = state.load(group, self.budget, device)
                    resumed = results is not None
                    if results is None:
                        results = self._run_group(group, catalog, state, runner, reporter, device)
                        self._still_at(sha, dirty)
                        state.save(group, results, self.budget, device)
                        if notebook is not None:
                            notebook.write(group)
                            report.notes_written += 1
                    elif notebook is not None and not notebook.note_path(group).exists():
                        notebook.write_note(group)
                        report.notes_written += 1
                    reporter.family_done(group, results, resumed=resumed)
                    merged[group].extend(results)
                    recorded[group] &= resumed
            if notebook is not None:
                notebook.write_ledger()
        for group in groups:
            report.add(group, merged[group], resumed=recorded[group])
        report.seconds = time.monotonic() - began
        report.summary_path = state.root / "summary.txt"
        _zoo_record.write_atomic(report.summary_path, render_summary(report))
        if not self.settings.families and not self.settings.limit and devices == ORDER:
            (state.root / "DONE").write_text(sha + "\n")
        return report

    def _run_group(
        self,
        group: str,
        catalog: ZooCatalog,
        state: _zoo_record.ZooState,
        runner: StageRunner,
        reporter: ZooReporter,
        device: str,
    ) -> list[StageResult]:
        results = []
        for stage in self._stages(group, catalog, state.temp, device):
            reporter.stage(stage.name)
            result = runner.run(stage, group)
            reporter.stage_done(result)
            results.append(result)
        return results


def run_zoo_stage(
    args: argparse.Namespace, console: Console, root: Path, command: str
) -> ZooReport:
    """Run the zoo stage for ``args``; the report holds the failing stages and the reduced groups."""
    settings = ZooSettings.from_args(args, root, command)
    run = ZooRun(root, console, settings)
    report = run.run()
    summarise(console, report, run.budget, settings.record_dir)
    return report


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
