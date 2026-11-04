"""One process of the zoo stage: what to run, and how its output is read.

A *stage* is a single child process — the contract validator, the real-size
build probe, the compiled-training probe, or one pytest invocation.  Stages run
strictly one at a time, with the markers cleared and the network off, so
nothing is downloaded and nothing is kept.

:class:`Stage` is the unit the memory guard kills and may retry, which is why
it is a small value object rather than a command line: a stage knows whether
it has ``heavy`` tests it could drop (:meth:`Stage.without_heavy`), and the
runner needs only that to decide on the one retry the card allows.
"""

import dataclasses
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from lucid.test.audit._zoo_budget import MachineBudget

DOMAINS = ("vision", "text", "generative", "multimodal")
ZOO_WIDE = "zoo-wide"

#: Statuses that make a stage a failure.  ``none`` (pytest exit 5, nothing
#: collected) is not one: a family with no tests of its own has not failed —
#: though a stage that ran nothing *and says nothing about why* is reduced.
#: ``mem-guard`` is the stage passing its own footprint limit; ``machine-short``
#: is the machine running out of memory under it, which is not the stage's doing
#: and so is never answered by running less of it.
FAILED_STATUSES = frozenset({"fail", "timeout", "mem-guard", "machine-short"})

_KILLED = {
    "mem-guard": "killed by the memory guard",
    "machine-short": "killed because the machine ran short of memory",
    "timeout": "timed out",
}

_SUMMARY = re.compile(r"^(?:=+ )?(\d+ .*?) in [0-9.]+s")
_FAILED = re.compile(r"^(?:FAILED|ERROR) (.+)$")
_SKIPPED = re.compile(r"^SKIPPED \[(\d+)\] (.+)$")
_CHECK = re.compile(r"^CHECK (.+)$")
_ADAPTIVE = re.compile(r"^ADAPTIVE (.+)$")
_MEMBER = re.compile(r"create_model\(")


@dataclass(frozen=True, kw_only=True)
class Stage(ABC):
    """One child process: what to run and what this machine has already decided about it."""

    name: str
    #: Decisions made before the process started (``heavy`` deselected up front).
    notes: tuple[str, ...] = ()

    @abstractmethod
    def argv(self) -> tuple[str, ...]:
        """The command line, run from the repository root."""

    def without_heavy(self, why: str) -> Stage | None:
        """This stage with the ``heavy`` tests deselected, or ``None`` if it has no such lever."""
        del why  # only a stage that runs tests has a reason to record
        return None


@dataclass(frozen=True, kw_only=True)
class CommandStage(Stage):
    """A fixed command: the contract validator or one of the probes."""

    command: tuple[str, ...]

    @override
    def argv(self) -> tuple[str, ...]:
        return self.command


@dataclass(frozen=True, kw_only=True)
class PytestStage(Stage):
    """One pytest invocation, with the project's default marker expression cleared.

    ``marker`` replaces ``addopts``' ``-m 'not heavy and not audit and not
    control'``: ``""`` runs everything, ``"not control"`` runs everything but
    the tens-of-minutes benchmarks, which belong to the zoo-wide group alone.
    """

    files: tuple[str, ...]
    basetemp: Path
    marker: str = ""
    #: A family's factory names and labels; ``()`` runs every test in ``files``.
    names: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    heavy_off: bool = False

    @override
    def argv(self) -> tuple[str, ...]:
        marker = self.marker
        if self.heavy_off:
            marker = f"({marker}) and not heavy" if marker else "not heavy"
        select = (
            (
                "-p",
                "lucid.test.audit._zoo_select",
                f"--zoo-names={','.join(self.names)}",
                f"--zoo-labels={','.join(self.labels)}",
            )
            if self.names or self.labels
            else ()
        )
        return (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "-rsfE",
            "-p",
            "no:cacheprovider",
            f"--basetemp={self.basetemp}",
            "-o",
            "faulthandler_timeout=900",
            "-m",
            marker,
            *select,
            *self.files,
        )

    @override
    def without_heavy(self, why: str) -> PytestStage | None:
        if self.heavy_off:
            return None
        return dataclasses.replace(self, heavy_off=True, notes=(*self.notes, why))


@dataclass
class StageResult:
    """What one stage did, in the form the notes and the resume check read."""

    name: str
    status: str
    seconds: float
    peak_gb: float
    summary: str = ""
    failures: list[str] = field(default_factory=list)
    skips: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    #: What this machine decided *to leave out*: tests dropped, builds skipped,
    #: a retry with less.  A stage with any of these verified less than it was
    #: asked to, which is what ``reduced`` means.
    notes: list[str] = field(default_factory=list)
    #: What happened to the stage without making it verify less: a wait for
    #: memory that ran out, a rerun after the machine ran short.
    events: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status in FAILED_STATUSES

    @property
    def reduced(self) -> bool:
        """Whether the stage verified less than it was asked to.

        Leaving something out because of the machine is reduced; so is a stage
        that ran nothing and does not say why.  A skip with a reason on record
        (a test that needs the network, a matrix pair with a stated gap) is a
        designed skip, not a reduction.
        """
        return bool(self.notes) or (self.status == "none" and not self.skips)


def read_result(
    stage: Stage, log: Path, status: str, seconds: float, peak_gb: float
) -> StageResult:
    """Fold a finished stage's log into a :class:`StageResult`."""
    result = StageResult(stage.name, status, round(seconds, 1), round(peak_gb, 2))
    result.notes.extend(stage.notes)
    lines = log.read_text(errors="replace").splitlines()
    for line in lines:
        if failed := _FAILED.match(line):
            result.failures.append(failed.group(1)[:300])
        elif skipped := _SKIPPED.match(line):
            result.skips.append(f"{skipped.group(1)} × {skipped.group(2)[:200]}")
        elif check := _CHECK.match(line):
            result.checks.append(check.group(1)[:300])
        elif adaptive := _ADAPTIVE.match(line):
            result.notes.append(adaptive.group(1)[:300])
        elif summary := _SUMMARY.match(line):
            result.summary = summary.group(1)
    if status in _KILLED:
        # The process never got to say why; the guard did, so that comes first.
        tail = " / ".join(lines[-3:])
        reason = f"{_KILLED[status]} at {peak_gb:.1f} GB after {seconds:.0f}s; the log ends: {tail}"
        result.failures.insert(0, reason[:300])
    elif result.failed and not result.failures:
        result.failures.append(" / ".join(lines[-3:])[:300])
    return result


class ZooCatalog:
    """The zoo as the filesystem has it: families, their test files, their stages."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._models = root / "lucid" / "models"
        self._unit = root / "lucid" / "test" / "unit"
        self._parity = root / "lucid" / "test" / "parity" / "models"
        self._wide: list[Path] | None = None

    def families(self) -> dict[str, str]:
        """Every family directory, ``name -> domain``, in a stable order."""
        found: dict[str, str] = {}
        for domain in DOMAINS:
            for path in sorted((self._models / domain).iterdir()):
                if path.is_dir() and not path.name.startswith("_"):
                    found[path.name] = domain
        return found

    def own_files(self, family: str) -> list[Path]:
        """The test files named for ``family`` alone."""
        own = sorted((self._unit / "models").glob(f"test_models_{family}.py"))
        return own + sorted(self._parity.glob(f"*/test_{family}.py"))

    def wide_files(self) -> list[Path]:
        """Files reaching several families: every model test no family owns, then the rest."""
        if self._wide is None:
            owned = {path for family in self.families() for path in self.own_files(family)}
            wide = [p for p in sorted((self._unit / "models").glob("test_*.py")) if p not in owned]
            wide += [p for p in sorted(self._parity.rglob("test_*.py")) if p not in owned]
            for path in sorted(self._unit.rglob("test_*.py")):
                if path.parent != self._unit / "models" and _MEMBER.search(
                    path.read_text(errors="replace")
                ):
                    wide.append(path)
            self._wide = wide
        return self._wide

    @staticmethod
    def _pytest(
        name: str,
        temp: Path,
        files: list[Path],
        marker: str = "",
        names: tuple[str, ...] = (),
        labels: tuple[str, ...] = (),
    ) -> PytestStage:
        return PytestStage(
            name=name,
            files=tuple(map(str, files)),
            basetemp=temp,
            marker=marker,
            names=names,
            labels=labels,
        )

    def family_stages(self, family: str, temp: Path, budget: MachineBudget) -> list[Stage]:
        """The stages one family goes through, in order."""
        py = sys.executable
        probe = (py, "-m", "lucid.test.audit._zoo_probes")
        stages: list[Stage] = [
            CommandStage(
                name="contract",
                command=(py, "-m", "tools.validate_model_zoo", "--family", family, "--runtime"),
            ),
            CommandStage(
                name="build",
                command=(*probe, "build", family, f"--budget-gb={budget.build_budget_gb}"),
            ),
            CommandStage(name="compiled-training", command=(*probe, "matrix", family)),
        ]
        if own := self.own_files(family):
            stages.append(self._pytest("own-tests", temp, own, marker=""))
        # Control benchmarks take tens of minutes and run once, in the zoo-wide group.
        models = [p for p in self.wide_files() if p.parent == self._unit / "models"]
        names, labels = self.selection_for(family)
        stages.append(
            self._pytest(
                "zoo-tests", temp, models, marker="not control", names=names, labels=labels
            )
        )
        return stages

    def selection_for(self, family: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The factory names and labels that identify ``family``'s tests in shared files."""
        from lucid.test.audit._zoo_probes import factories_by_family

        found = factories_by_family().get(family)
        names = tuple(found.names) if found else ()
        labels = tuple(sorted({family, *(found.labels if found else ())}))
        return names, labels

    def wide_stages(self, temp: Path) -> list[Stage]:
        """The zoo-wide group: every multi-family file once, ``control`` benchmarks included."""
        base = self._unit.parent
        return [
            self._pytest(str(path.relative_to(base)), temp, [path], marker="")
            for path in self.wide_files()
        ]


__all__ = [
    "DOMAINS",
    "FAILED_STATUSES",
    "ZOO_WIDE",
    "CommandStage",
    "PytestStage",
    "Stage",
    "StageResult",
    "ZooCatalog",
    "read_result",
]
