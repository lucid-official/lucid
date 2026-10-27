"""A live per-subsystem display, when ``rich`` is installed.

The stdlib console in :mod:`~lucid.test.audit._console` shows one bar at
a time, for the axis currently running.  That is the wrong axis of the
work to watch: a sweep is 27 axes over 34 subsystems, and what a person
wants to know part-way through is *which package is being hammered and
how it is doing*, not which of 27 questions is currently being asked.

This module renders instead:

* one overall bar over every (symbol x axis) cell, with elapsed and ETA;
* one bar per subsystem, each with its own running pass / fail / skip,
  as many as the terminal is tall enough to hold;
* a footer naming the axis and symbol in flight, and the running defect
  count.

Nothing else is written while the sweep runs.  A live region can only be
redrawn in place, so anything printed into it pushes it down the screen,
and a display that jumps every time a defect is found is worse at
conveying "9 defects so far" than a number that counts up.  The findings
themselves are listed in full when the run ends, where the whole list can
be read at once; the per-subsystem totals are reprinted there too, as
static lines, so the bars that did not fit on screen are still accounted
for.

``rich`` is optional.  :func:`build` returns ``None`` when it is not
installed, when the output is redirected, or under ``--quiet`` /
``--no-color``, and the caller keeps the stdlib display — the audit has
never needed a third-party package to run and still does not.

Install it with ``pip install lucid-dl[audit]``.
"""

import sys
import time
from typing import TYPE_CHECKING, Any

from lucid.test.audit._result import Status

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from lucid.test.audit._axes import Axis
    from lucid.test.audit._console import Console
    from lucid.test.audit._result import Finding
    from lucid.test.audit._surface import Symbol

try:  # pragma: no cover - exercised by whether the extra is installed
    from rich.console import Console as RichConsole
    from rich.console import Group
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text

    _AVAILABLE = True
except ImportError:  # pragma: no cover
    _AVAILABLE = False


def available() -> bool:
    """bool: Whether ``rich`` can be imported."""
    return _AVAILABLE


#: Cells between redraws.  A full re-render of every bar costs more than
#: some of the cells being measured, and nothing is lost by batching:
#: the counters are exact, only the frame rate is coarse.
_REFRESH_EVERY = 5

#: Seconds between redraws, on top of the cell count.  A fast sweep
#: finishes 10,000 cells in 23 seconds, which is still ~90 frames a
#: second after the count throttle alone — far past what a terminal can
#: draw, and the backlog is what makes the bars look like they are
#: stuttering rather than moving.  Dropping frames costs nothing: the
#: counters are read at draw time, not accumulated per frame.
_REFRESH_INTERVAL = 1 / 12

#: Lines the display needs for everything that is not a subsystem bar:
#: the overall bar, the blank spacer, the footer, the truncation notice,
#: and a margin so the last line never lands on the bottom row.
_CHROME_LINES = 6


class RichReporter:
    """Live progress, one bar per subsystem.

    Parameters
    ----------
    axes : sequence of Axis
        The axes this run will execute, used only for the total.
    symbols : sequence of Symbol
        Every symbol in scope.

    Notes
    -----
    Totals are computed the same way the runner computes its own work
    total — ``axis.applies(symbol)`` summed over both — so the bars
    cannot drift from what actually runs.
    """

    def __init__(self, axes: Sequence[Axis], symbols: Sequence[Symbol]) -> None:
        self._counts: dict[str, dict[str, int]] = {}
        totals: dict[str, int] = {}
        for symbol in symbols:
            applicable = sum(1 for axis in axes if axis.applies(symbol))
            if applicable:
                totals[symbol.subsystem] = totals.get(symbol.subsystem, 0) + applicable
        self._totals = dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))

        self._overall = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=34, complete_style="cyan", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta"),
            TimeRemainingColumn(),
            expand=False,
        )
        self._per_subsystem = Progress(
            TextColumn("  [dim]{task.description}"),
            BarColumn(bar_width=22, complete_style="blue", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("{task.fields[tally]}"),
            expand=False,
        )
        self._footer = Text("")
        self._defects = 0
        self._overall_task = self._overall.add_task("overall", total=sum(self._totals.values()))
        self._console = RichConsole(file=sys.stdout)

        # Only as many bars as the terminal can hold.
        #
        # ``rich`` redraws a live region by moving the cursor back over
        # the lines it wrote last time, which it cannot do once that
        # region is taller than the screen — the top has already scrolled
        # away.  Every refresh then appends a fresh copy instead of
        # replacing the old one, and 34 subsystems in a 24-row terminal
        # turn a progress display into a transcript of itself.  That is
        # the stray half-frames, and the bars that suddenly jump down.
        #
        # The cap is taken once, from the height at startup, and the same
        # subsystems stay on screen for the whole run: a display that
        # reorders itself to follow whatever is busy is harder to read
        # than one that holds still.  Nothing is lost by not showing the
        # rest — every subsystem appears in the tally printed at the end.
        room = max(1, self._console.size.height - _CHROME_LINES)
        self._shown = set(list(self._totals)[:room])
        self._hidden = len(self._totals) - len(self._shown)

        self._tasks: dict[str, Any] = {}
        width = max(len(k) for k in self._totals) if self._totals else 0
        for name, total in self._totals.items():
            self._counts[name] = {"pass": 0, "fail": 0, "other": 0}
            self._tasks[name] = self._per_subsystem.add_task(
                name.ljust(width),
                total=total,
                tally=Text(""),
                visible=name in self._shown,
            )
        # Pinned to the real stdout and refreshed by hand.  The runner
        # wraps every probe in ``Suppress``, which swaps ``sys.stdout``
        # for /dev/null so an op's warnings do not bury the display — and
        # the sweep spends nearly all of its time inside that window, so
        # a background refresh thread writes almost every frame into the
        # void.  The first version of this drew exactly one frame, at the
        # start, and then sat still for the whole run.
        #
        # ``transient``: the animated region is erased when the run ends
        # and a static tally is printed in its place.  A live region left
        # behind is the one frame nobody can scroll back through cleanly,
        # and it was also stranding its footer — the last symbol probed —
        # above the summary as though that symbol meant something.
        self._live = Live(
            self._render(),
            console=self._console,
            transient=True,
            auto_refresh=False,
        )
        self._since_refresh = 0
        self._last_drawn = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def __enter__(self) -> RichReporter:
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._live.__exit__(exc_type, exc, tb)
        self.print_tally()

    def print_tally(self) -> None:
        """The final per-subsystem counts, as static lines.

        Printed once the live region is gone, so this is the copy that
        stays in the scrollback — and it covers every subsystem, not only
        the ones the animated display had room for.
        """
        width = max(len(k) for k in self._totals) if self._totals else 0
        for name, total in self._totals.items():
            line = Text("  ")
            line.append(name.ljust(width), style="dim")
            line.append(f"  {total:>6}", style="white")
            line.append_text(self._tally(self._counts[name]))
            self._console.print(line)

    # ── updates ──────────────────────────────────────────────────────────────

    def record(self, symbol: Symbol, axis_name: str, finding: Finding) -> None:
        """Advance the bars for one completed cell."""
        bucket = self._counts.get(symbol.subsystem)
        if bucket is None:  # a subsystem with no applicable axis
            return
        if finding.status is Status.PASS:
            bucket["pass"] += 1
        elif finding.status.is_defect:
            bucket["fail"] += 1
        else:
            bucket["other"] += 1

        task = self._tasks[symbol.subsystem]
        self._per_subsystem.update(task, advance=1, tally=self._tally(bucket))
        self._overall.advance(self._overall_task, 1)
        if finding.status.is_defect:
            self._defects += 1
        self._footer = self._footer_text(axis_name, symbol.qualname)

        # Throttled twice: by cells, because a full re-render costs more
        # than some of the cells do, and by wall clock, because the cell
        # throttle alone still asks for more frames than a terminal can
        # draw and the backlog is what makes the bars look unsteady.
        self._since_refresh += 1
        if self._since_refresh < _REFRESH_EVERY:
            return
        now = time.monotonic()
        if now - self._last_drawn < _REFRESH_INTERVAL:
            return
        self._since_refresh = 0
        self._last_drawn = now
        self._live.update(self._render(), refresh=True)

    # ── rendering ────────────────────────────────────────────────────────────

    @staticmethod
    def _tally(bucket: dict[str, int]) -> Text:
        out = Text()
        out.append(f"  {bucket['pass']:>4} ok", style="green")
        if bucket["fail"]:
            out.append(f"  {bucket['fail']:>3} fail", style="bold red")
        if bucket["other"]:
            out.append(f"  {bucket['other']:>4} –", style="dim")
        return out

    def _footer_text(self, axis_name: str, qualname: str) -> Text:
        """What is in flight, and how many defects so far.

        A running count and nothing else while the sweep is going.  Each
        defect used to be printed the moment it was found, which scrolled
        the live region every time and then said the same thing again in
        the FAIL list at the end.  One place is enough, and the end is
        the place where the whole list can be read at once.
        """
        line = Text("  ")
        line.append(axis_name, style="bold cyan")
        line.append("  ")
        line.append(qualname, style="dim")
        if self._defects:
            line.append(f"    {self._defects} defect(s)", style="bold red")
        return line

    def _render(self) -> Group:
        parts: list[Any] = [self._overall, Text(""), self._per_subsystem]
        if self._hidden:
            parts.append(Text(f"  … {self._hidden} more, tallied below", style="dim"))
        parts.append(self._footer)
        return Group(*parts)


def build(console: Console, axes: Sequence[Axis], symbols: Sequence[Symbol]) -> RichReporter | None:
    """A live reporter, or ``None`` to keep the stdlib display.

    ``console.live`` already encodes "a terminal that can be animated and
    a user who did not ask for quiet", so this defers to it rather than
    re-deriving the same conditions and disagreeing with it.
    """
    if not _AVAILABLE or not console.live:
        return None
    return RichReporter(axes, symbols)


__all__ = ["RichReporter", "available", "build"]
