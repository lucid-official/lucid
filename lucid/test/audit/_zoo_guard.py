"""Running a stage under the memory guard, and the one retry the guard allows.

Memory is guarded from *outside* the process: a stage whose physical footprint
(the whole process group's, not just the leader's) passes the budget's limit, or
that leaves the machine short, is killed and the run moves on.  Inside the
process the same condition is a jetsam ``SIGKILL`` with no failing test and no
record of why.

The two kills are not the same event, and the stage is answered differently:

``mem-guard``
    the stage passed *its own* limit.  It is the one case where running less of
    it is an answer: a stage with ``heavy`` tests is run once more with them
    deselected, and its record says so.
``machine-short``
    the *machine* ran low, which another process may have caused.  Running less
    would only hide that — a family would read ``pass`` with its heavy tests
    never run because someone else's build dipped the floor — so the stage waits
    for room and is run again unreduced, once.  If the machine is short again it
    fails, and that record is never reused.

One retry either way, never a loop.
"""

import contextlib
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lucid.test.audit import _zoo_budget
from lucid.test._default_device import DEFAULT_DEVICE_ENV
from lucid.test.audit._console import fmt_duration
from lucid.test.audit._zoo_stage import StageResult, read_result

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import FrameType

    from lucid.test.audit._zoo_budget import MachineBudget
    from lucid.test.audit._zoo_stage import Stage

STAGE_TIMEOUT_S = 2 * 3600
POLL_S = 2.0
ROOM_WAIT_S = 30 * 60
ROOM_POLL_S = 30.0


@dataclass(frozen=True)
class Outcome:
    """How a process ended, before its log is read."""

    status: str
    seconds: float
    peak_gb: float
    #: Set when the wait for free memory ran out and the stage started anyway.
    wait_note: str = ""


def child_env(root: Path, device: str = "cpu") -> dict[str, str]:
    """The children's environment: this tree, no network, bytecode out of the tree.

    ``device`` is stated, never inherited: whatever the caller's shell had in
    ``LUCID_TEST_DEFAULT_DEVICE`` is replaced, so the record's device is the
    device the stage ran on.
    """
    env = dict(os.environ)
    env.pop("LUCID_TEST_NETWORK", None)
    env.update(
        {DEFAULT_DEVICE_ENV: device},
        PYTHONPATH=str(root),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPYCACHEPREFIX=str(Path.home() / "Library" / "Caches" / "lucid-pycache"),
    )
    return env


@contextlib.contextmanager
def exit_on_terminate() -> Iterator[None]:
    """Turn ``SIGTERM`` and ``SIGHUP`` into ``SystemExit`` for the length of a run.

    The stage child lives in a session of its own, so a signal to the parent
    does not reach it; and a signal's default action ends the parent without
    running the ``finally`` that kills the child.  ``^C`` was handled and
    ``kill`` and a closed terminal were not, which left a stage holding tens of
    gigabytes behind a dead run.

    ``SIGHUP`` is left alone when it is *ignored*.  ``nohup`` ignores it on
    purpose, so that a run started for the night survives its terminal closing;
    installing a handler over that ignore turned the closed terminal back into
    the death ``nohup`` exists to prevent.
    """

    def leave(signum: int, frame: FrameType | None) -> None:
        del frame  # the signal handler protocol passes it; nothing here needs it
        raise SystemExit(128 + signum)

    if threading.current_thread() is not threading.main_thread():
        yield  # only the main thread may install handlers
        return
    wanted = [signal.SIGTERM]
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        wanted.append(signal.SIGHUP)
    previous = {sig: signal.signal(sig, leave) for sig in wanted}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class StageGuard:
    """Runs one process and kills it when it passes the budget."""

    def __init__(
        self,
        root: Path,
        budget: MachineBudget,
        *,
        on_tick: Callable[[float, float], None] | None = None,
        on_wait: Callable[[float, float, float], None] | None = None,
        poll_s: float = POLL_S,
        timeout_s: float = STAGE_TIMEOUT_S,
        room_wait_s: float = ROOM_WAIT_S,
        room_poll_s: float = ROOM_POLL_S,
        device: str = "cpu",
    ) -> None:
        self.root = root
        self.budget = budget
        self.device = device
        self._on_tick = on_tick
        self._on_wait = on_wait
        self._poll_s = poll_s
        self._timeout_s = timeout_s
        self._room_wait_s = room_wait_s
        self._room_poll_s = room_poll_s

    def _tick(self, elapsed: float, footprint: float) -> None:
        if self._on_tick is not None:
            self._on_tick(elapsed, footprint)

    def wait_for_room(self, log: Path) -> str:
        """Hold the stage back until the machine has room, since a build may share it.

        Returns a description of a wait that ran out, or ``""``.  The wait is
        shown as a wait (``on_wait``), not as a stage at ``0.0 GB``, and a wait
        that gave up is put on the record, because a stage that started with
        little free memory is the one most likely to be killed for it.
        """
        began = time.monotonic()
        needed = self.budget.start_room_gb
        note = ""
        while (free := _zoo_budget.available_gb()) < needed:
            waited = time.monotonic() - began
            if waited >= self._room_wait_s:
                note = (
                    f"waited {fmt_duration(waited)} for {needed:g} GB of free memory and "
                    f"started with {free:.1f} GB"
                )
                break
            if self._on_wait is not None:
                self._on_wait(waited, free, needed)
            time.sleep(self._room_poll_s)
        with log.open("a") as out:
            out.write(f"[audit-zoo] starting with {_zoo_budget.available_gb():.1f} GB available\n")
            if note:
                out.write(f"[audit-zoo] {note}\n")
        return note

    def _verdict(self, peak: float, elapsed: float) -> str:
        """Why the process must die now, or ``""``."""
        if peak > self.budget.limit_gb:
            return "mem-guard"
        if _zoo_budget.available_gb() < self.budget.floor_gb:
            return "machine-short"
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
        wait_note = self.wait_for_room(log)
        started = time.monotonic()
        peak = 0.0
        verdict = ""
        with log.open("a") as out:
            proc = subprocess.Popen(
                argv,
                cwd=self.root,
                env=child_env(self.root, self.device),
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                for interval in self._intervals():
                    if self._exited(proc, interval):
                        break
                    footprint = _zoo_budget.group_footprint_gb(proc.pid)
                    peak = max(peak, footprint)
                    self._tick(time.monotonic() - started, footprint)
                    verdict = self._verdict(peak, time.monotonic() - started)
                    if verdict:
                        break
            finally:
                # The whole group, always: a worker that outlives a leader that
                # died (or finished) is still holding the memory the next stage
                # needs.  The group may be empty by now; that is not an error.
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        seconds = time.monotonic() - started
        if verdict:
            return Outcome(verdict, seconds, peak, wait_note)
        status = {0: "pass", 5: "none"}.get(proc.returncode, "fail")
        return Outcome(status, seconds, peak, wait_note)


class StageRunner:
    """Runs stages one at a time, with the one retry a killed stage is owed."""

    def __init__(self, guard: StageGuard, logs: Path, temp: Path) -> None:
        self._guard = guard
        self._logs = logs
        self._temp = temp

    def _attempt(self, stage: Stage, group: str, suffix: str = "") -> StageResult:
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stage.name) + suffix
        folder = self._logs / re.sub(r"[^a-zA-Z0-9_.-]+", "_", group)
        folder.mkdir(parents=True, exist_ok=True)
        log = folder / f"{slug}.log"
        log.write_text("")
        outcome = self._guard.run(stage.argv(), log)
        result = read_result(
            stage, log, outcome.status, outcome.seconds, outcome.peak_gb, self._guard.device
        )
        if outcome.wait_note:
            result.events.append(outcome.wait_note)
        return result

    @staticmethod
    def _merge(first: StageResult, result: StageResult) -> None:
        result.seconds = round(first.seconds + result.seconds, 1)
        result.peak_gb = max(first.peak_gb, result.peak_gb)

    def run(self, stage: Stage, group: str = "") -> StageResult:
        """Run ``stage``; if the guard killed it, answer the kind of kill it was."""
        result = self._attempt(stage, group)
        if result.status == "machine-short":
            first = result
            result = self._attempt(stage, group, "-again")
            self._merge(first, result)
            result.events.append(
                f"the machine ran short of memory during the first attempt on {first.device} "
                f"({first.peak_gb:.1f} GB footprint); waited for room and reran it "
                f"unreduced: {result.status}"
            )
        elif result.status == "mem-guard":
            retry = stage.without_heavy(
                f"the memory guard killed the first attempt on {result.device} "
                f"at {result.peak_gb:.1f} GB; "
                "this is the retry with heavy tests deselected"
            )
            if retry is not None:
                first = result
                result = self._attempt(retry, group, "-retry")
                self._merge(first, result)
                result.notes.append(f"retry with heavy deselected: {result.status}")
        shutil.rmtree(self._temp, ignore_errors=True)
        return result


__all__ = ["Outcome", "StageGuard", "StageRunner", "child_env", "exit_on_terminate"]
