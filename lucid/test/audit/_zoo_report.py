"""What the zoo stage shows while it runs.

Seventy-five families and hours of wall clock: the question a person has part
way through is not *what passed* but *where is it, is it alive, and how is it
going*.  So the display carries exactly that — the family ``n/N``, the stage in
flight, how long it has been going, the footprint against the limit, and a
running tally — and nothing else while a stage runs.

Three renderings of one reporter, chosen the way the sweep's are:

* **plain lines** when there is no terminal to animate (a detached run, a
  pipe, ``--no-color``): each stage start and result is a line, and a stage
  that runs long says it is still alive every few minutes, because a silent
  log is indistinguishable from a hung one;
* **a live block** on a terminal, redrawn in place from the stdlib console;
* **a ``rich`` live region** when that extra is installed
  (:func:`~lucid.test.audit._console_rich.build_zoo`).

Finished stages are always lines — they are the record, and the live region is
cleared around them so they scroll like any other output.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from lucid.test.audit._console import fmt_duration

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from lucid.test.audit._console import Console
    from lucid.test.audit._zoo_stage import StageResult

#: Seconds between "still running" lines when there is no live region.
HEARTBEAT_S = 300.0

_COLOUR = {"pass": "green", "none": "grey", "fail": "red", "timeout": "red", "mem-guard": "red"}


@dataclass
class ZooTally:
    """The running counts the display and the verdict both read."""

    families_ok: int = 0
    families_failed: int = 0
    stages_ok: int = 0
    stages_failed: int = 0


class ZooReporter(ABC):
    """The zoo stage's progress, independent of how it is drawn.

    The run calls the public methods in order — :meth:`start`, then per family
    :meth:`family`, per stage :meth:`stage` / :meth:`tick` / :meth:`stage_done`,
    then :meth:`family_done` — and a subclass only draws and emits.
    """

    def __init__(self, console: Console, limit_gb: float) -> None:
        self.console = console
        self.limit_gb = limit_gb
        self.tally = ZooTally()
        self.total = 0
        self.index = 0
        self.family_name = ""
        self.stage_name = ""
        self.stage_seconds = 0.0
        self.footprint_gb = 0.0
        self.stage_peak_gb = 0.0
        self.run_peak_gb = 0.0
        self._began = time.monotonic()

    # ── what the run calls ───────────────────────────────────────────────────

    def start(self, total: int) -> None:
        self.total = total
        self._began = time.monotonic()
        self._draw()

    def family(self, index: int, name: str) -> None:
        self.index = index
        self.family_name = name
        self._family_started()
        self._draw()

    def stage(self, name: str) -> None:
        self.stage_name = name
        self.stage_seconds = 0.0
        self.footprint_gb = 0.0
        self.stage_peak_gb = 0.0
        self._stage_started()
        self._draw()

    def tick(self, elapsed_s: float, footprint_gb: float) -> None:
        self.stage_seconds = elapsed_s
        self.footprint_gb = footprint_gb
        self.stage_peak_gb = max(self.stage_peak_gb, footprint_gb)
        self.run_peak_gb = max(self.run_peak_gb, footprint_gb)
        self._draw()

    def stage_done(self, result: StageResult) -> None:
        if result.failed:
            self.tally.stages_failed += 1
        else:
            self.tally.stages_ok += 1
        self.run_peak_gb = max(self.run_peak_gb, result.peak_gb)
        self.stage_name = ""
        self._emit(self.format_stage(result))
        self._draw()

    def family_done(self, name: str, results: Sequence[StageResult], *, resumed: bool) -> None:
        bad = any(result.failed for result in results)
        if bad:
            self.tally.families_failed += 1
        else:
            self.tally.families_ok += 1
        if resumed:
            for result in results:
                self.tally.stages_failed += result.failed
                self.tally.stages_ok += not result.failed
        self._emit(self.format_family(name, results, resumed))
        self._draw()

    def close(self) -> None:
        """Release whatever the drawing holds.  Safe to call more than once."""

    def __enter__(self) -> ZooReporter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ── what a subclass provides ─────────────────────────────────────────────

    @abstractmethod
    def _draw(self) -> None:
        """Show the current state: the live region, or a heartbeat line."""

    @abstractmethod
    def _emit(self, line: str) -> None:
        """Put one permanent line above the live region."""

    def _family_started(self) -> None:
        """Hook for a reporter that wants a line when a family begins."""

    def _stage_started(self) -> None:
        """Hook for a reporter that wants a line when a stage begins."""

    # ── shared text ──────────────────────────────────────────────────────────

    def format_stage(self, result: StageResult) -> str:
        paint = self.console.paint
        return (
            "    "
            + paint(result.name.ljust(24)[:24], "white")
            + paint(result.status.ljust(10), _COLOUR.get(result.status, "white"))
            + paint(f"{result.seconds:>6.0f}s  {result.peak_gb:>5.1f} GB  ", "grey")
            + result.summary[:60]
        )

    def format_family(self, name: str, results: Sequence[StageResult], resumed: bool) -> str:
        paint = self.console.paint
        bad = any(result.failed for result in results)
        status = "fail" if bad else "pass"
        minutes = sum(result.seconds for result in results) / 60
        peak = max((result.peak_gb for result in results), default=0.0)
        label = f"{self.position()} {name}: " + paint(status, _COLOUR[status], "bold")
        extra = "  (recorded for this commit)" if resumed else ""
        return f"{label}  {minutes:.1f} min  peak {peak:.1f} GB{extra}  " + self.format_tally()

    def format_tally(self) -> str:
        t = self.tally
        paint = self.console.paint
        return paint(
            f"[families ok {t.families_ok} fail {t.families_failed} · "
            f"stages ok {t.stages_ok} fail {t.stages_failed}]",
            "red" if t.stages_failed else "grey",
        )

    def position(self) -> str:
        """``[ 3/75]``, padded so a column of them lines up."""
        return f"[{self.index:>{len(str(self.total))}}/{self.total}]"

    def header(self) -> str:
        return f"family {self.index}/{self.total}  {self.family_name}"

    def status(self) -> str:
        if not self.stage_name:
            return "between stages"
        return (
            f"{self.stage_name}  {fmt_duration(self.stage_seconds)}  "
            f"footprint {self.footprint_gb:.1f} GB "
            f"(stage peak {self.stage_peak_gb:.1f}, limit {self.limit_gb:g})"
        )

    @property
    def elapsed(self) -> str:
        return fmt_duration(time.monotonic() - self._began)


class LineReporter(ZooReporter):
    """The stdlib rendering: a live block on a terminal, plain lines otherwise."""

    def __init__(self, console: Console, limit_gb: float) -> None:
        super().__init__(console, limit_gb)
        self._frames = 0
        self._last_beat = time.monotonic()

    @override
    def _family_started(self) -> None:
        if not self.console.live:
            self.console.write(f"{self.position()} {self.family_name}")

    @override
    def _stage_started(self) -> None:
        self._last_beat = time.monotonic()
        if not self.console.live:
            self.console.write(f"    {self.stage_name} ...")

    @override
    def _emit(self, line: str) -> None:
        self.console.live_clear()
        self.console.write(line)

    @override
    def _draw(self) -> None:
        console = self.console
        if console.live:
            self._frames += 1
            console.live_block(self._frame())
        elif self.stage_name and time.monotonic() - self._last_beat >= HEARTBEAT_S:
            self._last_beat = time.monotonic()
            console.write(f"    ... {self.status()}")

    def _frame(self) -> list[str]:
        c = self.console
        done = max(self.index - 1, 0)
        return [
            "  "
            + c.paint(c.spinner(self._frames), "cyan")
            + "  "
            + c.paint(self.header(), "bold")
            + "  "
            + c.paint(c.bar(done, self.total, 24), "cyan")
            + c.paint(f"  {self.elapsed}", "grey"),
            "     " + c.paint(self.status(), "grey"),
            "     " + self.format_tally(),
        ]


def build_reporter(console: Console, limit_gb: float) -> ZooReporter:
    """A ``rich`` reporter when it can be used, the stdlib one otherwise."""
    from lucid.test.audit import _console_rich

    return _console_rich.build_zoo(console, limit_gb) or LineReporter(console, limit_gb)


__all__ = ["HEARTBEAT_S", "LineReporter", "ZooReporter", "ZooTally", "build_reporter"]
