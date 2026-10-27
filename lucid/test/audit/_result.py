"""Verdicts, findings and the report the audit produces.

The status vocabulary is the part worth reading.  A survey that only
knows PASS and FAIL over-reports: most of what a first pass flags is an
artefact of the probe rather than a defect in the code, and a tool that
cannot say so trains its user to ignore it.

    PASS         checked and correct
    FAIL         checked and wrong — the only status that is a defect
    TRUNCATION   the finite-difference probe was the limiting factor, not
                 the op; proved by refining the step and watching the
                 disagreement shrink quadratically
    GAUGE        the quantity is only defined up to a symmetry (SVD's
                 singular vectors), so a difference means nothing
    VACUOUS      the check ran but could not have failed — a gradient
                 that is identically zero, a mask with no live entries.
                 Reported separately because a vacuous pass is worse
                 than no pass: it reads as coverage and is not
    UNSUPPORTED  the op refused, loudly and by design (no graph formula,
                 dtype not implemented).  A limitation, not a defect
    NOT_APPLICABLE
                 the question does not exist for this symbol — a broadcast
                 check on a unary op, an entry-point comparison for a name
                 with one spelling, float64 on a device that has no float64.
                 Excluded from the coverage denominator, because counting a
                 question that cannot be asked as one that went unanswered
                 understates how much a run actually verified.  Distinct
                 from SKIP, which is a real gap in the harness and stays in
                 the denominator as one.
    SKIP         the harness could not build inputs.  **Counted and
                 listed**, because this is exactly where a census
                 quietly stops being one
    KNOWN        matches an accepted entry in the baseline file
    ERROR        the probe itself broke
"""

import dataclasses
import enum
import json
import platform
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class Status(enum.Enum):
    """Outcome of one axis applied to one symbol."""

    PASS = "pass"
    FAIL = "fail"
    TRUNCATION = "truncation"
    GAUGE = "gauge"
    VACUOUS = "vacuous"
    UNSUPPORTED = "unsupported"
    SKIP = "skip"
    NOT_APPLICABLE = "not_applicable"
    KNOWN = "known"
    ERROR = "error"

    @property
    def is_defect(self) -> bool:
        """bool: Whether this outcome should fail the run."""
        return self is Status.FAIL

    @property
    def is_coverage(self) -> bool:
        """bool: Whether this outcome counts as the symbol having been checked."""
        return self in (
            Status.PASS,
            Status.FAIL,
            Status.TRUNCATION,
            Status.GAUGE,
            Status.KNOWN,
        )


#: Rendering order and colour for each status.
STATUS_STYLE: dict[Status, tuple[str, str]] = {
    Status.FAIL: ("FAIL", "red"),
    Status.ERROR: ("ERR ", "red"),
    Status.VACUOUS: ("VAC ", "yellow"),
    Status.KNOWN: ("KNWN", "magenta"),
    Status.TRUNCATION: ("TRNC", "blue"),
    Status.GAUGE: ("GAUG", "blue"),
    Status.UNSUPPORTED: ("UNSP", "grey"),
    Status.SKIP: ("SKIP", "grey"),
    Status.NOT_APPLICABLE: ("N/A ", "grey"),
    Status.PASS: ("PASS", "green"),
}


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """One (axis, symbol) outcome."""

    axis: str
    symbol: str
    status: Status
    detail: str = ""
    evidence: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def key(self) -> str:
        """str: Stable identity for baseline matching."""
        return f"{self.axis}::{self.symbol}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "symbol": self.symbol,
            "status": self.status.value,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclasses.dataclass
class Report:
    """Everything one run produced."""

    findings: list[Finding] = dataclasses.field(default_factory=list)
    started: float = dataclasses.field(default_factory=time.time)
    duration: float = 0.0
    surface_total: int = 0
    #: (symbol, axis) pairs where the axis is *meaningful* for the symbol.
    #: The denominator for depth.  Using ``symbols x axes`` instead would
    #: punish adding a subsystem-specific axis, since it dilutes every
    #: symbol the new axis cannot express — which is backwards.
    applicable_cells: int = 0
    axes_run: list[str] = dataclasses.field(default_factory=list)
    config: dict[str, Any] = dataclasses.field(default_factory=dict)

    def add(self, finding: Finding) -> Finding:
        self.findings.append(finding)
        return finding

    # ── slicing ──────────────────────────────────────────────────────────────

    def by_status(self, status: Status) -> list[Finding]:
        return [f for f in self.findings if f.status is status]

    def counts(self) -> dict[Status, int]:
        out = dict.fromkeys(Status, 0)
        for f in self.findings:
            out[f.status] += 1
        return out

    def counts_by_axis(self) -> dict[str, dict[Status, int]]:
        out: dict[str, dict[Status, int]] = {}
        for f in self.findings:
            out.setdefault(f.axis, dict.fromkeys(Status, 0))[f.status] += 1
        return out

    @property
    def defects(self) -> list[Finding]:
        return [f for f in self.findings if f.status.is_defect]

    @property
    def covered_symbols(self) -> set[str]:
        """Symbols that at least one axis actually reached."""
        return {f.symbol for f in self.findings if f.status.is_coverage}

    @property
    def unreached_symbols(self) -> set[str]:
        """Symbols every axis skipped — the honest hole in the census."""
        reached = self.covered_symbols
        return {f.symbol for f in self.findings if f.symbol not in reached}

    def coverage(self) -> tuple[int, int, float]:
        """``(reached, total, fraction)`` over the enumerated surface."""
        reached = len(self.covered_symbols)
        total = self.surface_total or reached
        return reached, total, (reached / total if total else 0.0)

    def cell_coverage(self) -> tuple[int, int, float]:
        """Applicable (symbol, axis) pairs that produced a real verdict.

        The measure that does not flatter: a SKIP or an UNSUPPORTED does
        not count, so a run that reaches every symbol but only smoke-tests
        them still reports a low number.
        """
        filled = sum(1 for f in self.findings if f.status.is_coverage)
        total = self.applicable_cells or (self.surface_total * max(len(self.axes_run), 1))
        # A question that cannot be asked is not one that went unanswered.
        # 1,856 cells were being counted against the framework for asking a
        # unary op how it broadcasts, a single-spelling name how its two
        # entry points compare, and Metal for a float64 it does not have.
        total -= sum(1 for f in self.findings if f.status is Status.NOT_APPLICABLE)
        return filled, total, (filled / total if total else 0.0)

    # ── serialisation ────────────────────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        reached, total, frac = self.coverage()
        filled, cells, cell_frac = self.cell_coverage()
        return {
            "schema": 1,
            "started": self.started,
            "duration_seconds": round(self.duration, 3),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "config": self.config,
            "axes": self.axes_run,
            "coverage": {
                "symbols_reached": reached,
                "symbols_total": total,
                "symbol_fraction": round(frac, 4),
                "cells_filled": filled,
                "cells_total": cells,
                "cell_fraction": round(cell_frac, 4),
            },
            "counts": {s.value: n for s, n in self.counts().items()},
            "findings": [f.as_dict() for f in self.findings],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=False) + "\n")


class Baseline:
    """Accepted deviations, so a repeat run reports only what is new.

    An audit is only worth automating if its output goes to zero when
    nothing is wrong.  Some findings are decisions rather than bugs —
    ``sign(NaN)`` returning 0 is consistent across both of Lucid's
    devices and inherited from the GPU's own primitive, so it is a
    convention the project has taken, not a defect to re-report every
    run.  Those live here, each with a reason.
    """

    def __init__(self, entries: dict[str, str] | None = None) -> None:
        self.entries = dict(entries or {})

    @classmethod
    def load(cls, path: Path | None) -> Baseline:
        if path is None or not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls({k: str(v) for k, v in raw.get("accepted", {}).items()})

    def save(self, path: Path, findings: Iterable[Finding]) -> None:
        merged = dict(self.entries)
        for f in findings:
            merged.setdefault(f.key, f.detail or "accepted by --update-known")
        payload = {
            "comment": (
                "Findings accepted as decisions rather than defects. "
                "Each key is 'axis::symbol'; the value is why it is accepted. "
                "Delete an entry to make the audit report it again."
            ),
            "accepted": dict(sorted(merged.items())),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        self.entries = merged

    def reason(self, finding: Finding) -> str | None:
        return self.entries.get(finding.key)

    def apply(self, finding: Finding) -> Finding:
        """Downgrade a defect to KNOWN when the baseline accepts it."""
        if not finding.status.is_defect:
            return finding
        why = self.reason(finding)
        if why is None:
            return finding
        return dataclasses.replace(
            finding, status=Status.KNOWN, detail=f"{finding.detail}  [accepted: {why}]"
        )


__all__ = ["Baseline", "Finding", "Report", "STATUS_STYLE", "Status"]
