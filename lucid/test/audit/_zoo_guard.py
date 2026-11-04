"""Running a stage under the memory guard, and the one retry the guard allows.

Memory is guarded from *outside* the process: a stage whose physical footprint
passes the budget's limit, or that leaves the machine short, is killed and
recorded as ``mem-guard``, and the run moves on.  Inside the process the same
condition is a jetsam ``SIGKILL`` with no failing test and no record of why.

A killed stage is a failure unless it can be made to fit.  The only lever the
guard pulls is the one the task describes: a stage with ``heavy`` tests is run
once more with them deselected, and its record says so.  One retry, never a
loop — a stage that still does not fit is a failure worth reading.
"""

import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lucid.test.audit import _zoo_budget
from lucid.test.audit._zoo_stage import StageResult, read_result

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from lucid.test.audit._zoo_budget import MachineBudget
    from lucid.test.audit._zoo_stage import Stage

STAGE_TIMEOUT_S = 2 * 3600
POLL_S = 2.0
ROOM_WAIT_S = 30 * 60
_ROOM_POLL_S = 30.0


@dataclass(frozen=True)
class Outcome:
    """How a process ended, before its log is read."""

    status: str
    seconds: float
    peak_gb: float


def child_env(root: Path) -> dict[str, str]:
    """The children's environment: this tree, no network, bytecode out of the tree."""
    env = dict(os.environ)
    env.pop("LUCID_TEST_NETWORK", None)
    env.update(
        PYTHONPATH=str(root),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPYCACHEPREFIX=str(Path.home() / "Library" / "Caches" / "lucid-pycache"),
    )
    return env


class StageGuard:
    """Runs one process and kills it when it passes the budget."""

    def __init__(
        self,
        root: Path,
        budget: MachineBudget,
        *,
        on_tick: Callable[[float, float], None] | None = None,
        poll_s: float = POLL_S,
        timeout_s: float = STAGE_TIMEOUT_S,
        room_wait_s: float = ROOM_WAIT_S,
    ) -> None:
        self.root = root
        self.budget = budget
        self._on_tick = on_tick
        self._poll_s = poll_s
        self._timeout_s = timeout_s
        self._room_wait_s = room_wait_s

    def _tick(self, elapsed: float, footprint: float) -> None:
        if self._on_tick is not None:
            self._on_tick(elapsed, footprint)

    def _wait_for_room(self, log: Path) -> None:
        """Hold the stage back until the machine has room, since a build may share it."""
        began = time.monotonic()
        while (
            _zoo_budget.available_gb() < self.budget.start_room_gb
            and time.monotonic() - began < self._room_wait_s
        ):
            self._tick(time.monotonic() - began, 0.0)
            time.sleep(_ROOM_POLL_S)
        with log.open("a") as out:
            out.write(f"[audit-zoo] starting with {_zoo_budget.available_gb():.1f} GB available\n")

    def _verdict(self, peak: float, elapsed: float) -> str:
        """Why the process must die now, or ``""``."""
        if peak > self.budget.limit_gb or _zoo_budget.available_gb() < self.budget.floor_gb:
            return "mem-guard"
        return "timeout" if elapsed > self._timeout_s else ""

    def _intervals(self) -> Iterator[float]:
        """Poll quickly at first, then every ``poll_s``.

        A stage that ends inside its first interval is never measured, and a
        build that took a second reported a peak of 0.0 GB.  The first samples
        come early (a quarter second, doubling) so a short stage still has a
        footprint, and a long one costs the usual two seconds a look.
        """
        wait = min(0.25, self._poll_s)
        while True:
            yield wait
            wait = min(wait * 2, self._poll_s)

    @staticmethod
    def _exited(proc: subprocess.Popen[bytes], timeout: float) -> bool:
        """Wait up to ``timeout``; whether the process ended in it."""
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def run(self, argv: tuple[str, ...], log: Path) -> Outcome:
        """Run ``argv`` with its output in ``log``; kill it past the budget."""
        self._wait_for_room(log)
        started = time.monotonic()
        peak = 0.0
        verdict = ""
        with log.open("a") as out:
            proc = subprocess.Popen(
                argv,
                cwd=self.root,
                env=child_env(self.root),
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                for interval in self._intervals():
                    if self._exited(proc, interval):
                        break
                    footprint = _zoo_budget.footprint_gb(proc.pid)
                    peak = max(peak, footprint)
                    self._tick(time.monotonic() - started, footprint)
                    verdict = self._verdict(peak, time.monotonic() - started)
                    if verdict:
                        break
            finally:
                # The child has its own session, so ^C on the parent does not
                # reach it; without this an interrupted run leaves a stage
                # holding tens of gigabytes behind.
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        seconds = time.monotonic() - started
        if verdict:
            return Outcome(verdict, seconds, peak)
        return Outcome({0: "pass", 5: "none"}.get(proc.returncode, "fail"), seconds, peak)


class StageRunner:
    """Runs stages one at a time, with the one retry a killed stage is owed."""

    def __init__(self, guard: StageGuard, logs: Path, temp: Path) -> None:
        self._guard = guard
        self._logs = logs
        self._temp = temp

    def _attempt(self, stage: Stage, suffix: str = "") -> StageResult:
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stage.name) + suffix
        log = self._logs / f"{slug}.log"
        log.write_text("")
        outcome = self._guard.run(stage.argv(), log)
        return read_result(stage, log, outcome.status, outcome.seconds, outcome.peak_gb)

    def run(self, stage: Stage) -> StageResult:
        """Run ``stage``; if the memory guard killed it, run it once more without ``heavy``."""
        result = self._attempt(stage)
        retry = None
        if result.status == "mem-guard":
            retry = stage.without_heavy(
                f"the memory guard killed the first attempt at {result.peak_gb:.1f} GB; "
                "this is the retry with heavy tests deselected"
            )
        if retry is not None:
            first = result
            result = self._attempt(retry, "-retry")
            result.seconds = round(first.seconds + result.seconds, 1)
            result.peak_gb = max(first.peak_gb, result.peak_gb)
            result.notes.append(f"retry with heavy deselected: {result.status}")
        shutil.rmtree(self._temp, ignore_errors=True)
        return result


__all__ = ["Outcome", "StageGuard", "StageRunner", "child_env"]
