"""Reading a process's footprint until the memory its released executables pinned is back.

``MPSGraphExecutable`` memory does not leave the process footprint when its executable is
released.  The Metal runtime returns it later, and *when* varies a great deal (LCD-1112).  On
an idle M1 Pro a ``session_cache_clear()`` showed its drop after 0 ms to 2.3 s; on a Mac Studio
running another test run (``load1`` 40-60) the footprint sat flat for 7 s or for 13 s with no
activity and then fell by itself, in steps.  A reading taken after a fixed ``sleep`` therefore
lands before the release about half the time (2 of 3 runs failed, with the capped footprint at
88, 144 and -53 MB where 45 is right), and the baseline read the same way is as often too
high: the control then reads small, or negative, and the ratio of the two is noise.

Touching the GPU while waiting (an MLX op, or a compiled-graph run, every half second) made
the byte-cap child's readings reproducible (12 of 12 runs read 149 and 45 MB at load 40-60) but
it does not release what a module held: the baseline after ``del`` of a module with eight
executables still carried their 1.5 GB, and the live-module control then read 0 in 3 of 4
runs.  So this waits idle, and the wait ends when the reading says so, not when a clock does.

:class:`Footprints` reads every ``POLL_S`` after a quiet ``QUIET_S`` until a condition on the
reading holds, up to a timeout.  The condition is the assertion's own (the control grew; the
capped footprint is under half the control), so a release that comes late is waited for and a
hold, which never comes, runs to the timeout and fails.  Every reading is judged against the
lowest footprint the process has shown, which a baseline read too high cannot move.  The
graphs are large (:data:`LEAD`) so the transient block the footprint also carries (about
144 MB on the Studio: it appears with the first graph run after a quiet spell and goes by a
timer) cannot reach the threshold while the reading waits for it.
"""

import gc
import os
import time
from collections.abc import Callable

import lucid
from lucid.test.audit._zoo_budget import footprint_gb

__all__ = ["GRAPHS", "LEAD", "PINNED_MB", "Footprints", "footprint_mb"]

#: The graphs a footprint child builds, and the leading dimension of the first one's input
#: (``(LEAD + 32 * i, 8, 16, 16)`` float32 for graph ``i``).  Each graph writes out a 2x
#: broadcast of it, which its executable then pins: 128 MB at this size, a gigabyte over
#: eight graphs.  At 1024 rows (16 MB each) the transient was as large as the whole signal.
GRAPHS = 8
LEAD = 8192
#: What one graph of the first input size pins, in MB: the 2x broadcast, in float32.
PINNED_MB = 2 * LEAD * 8 * 16 * 16 * 4 / 2**20

#: Idle seconds before the first reading, and between the readings that follow it.
QUIET_S = 3.0
POLL_S = 0.5
#: How long a reading waits for its condition: a release was seen to take 13 s under load.
TIMEOUT_S = 40.0


def footprint_mb() -> float:
    """This process's physical footprint in MB (Metal buffers count, resident size does not)."""
    return footprint_gb(os.getpid()) * 1024


def _free_garbage() -> None:
    """What a reading must not count: Python garbage and MLX's cache of freed buffers."""
    gc.collect()
    lucid.metal.empty_cache()  # freed inputs sit in MLX's cache, not in an executable


class Footprints:
    """The footprint readings of one process, each taken once its condition holds.

    ``floor`` is the lowest footprint any reading has shown: what the process holds with
    nothing pinned (the Metal set-up of one graph stays).  A condition is a function of
    ``(reading, floor)``, both in MB, so it is judged against the floor known when it is asked.
    """

    def __init__(
        self,
        read: Callable[[], float] = footprint_mb,
        sleep: Callable[[float], None] = time.sleep,
        prepare: Callable[[], None] = _free_garbage,
    ) -> None:
        self._read = read
        self._sleep = sleep
        self._prepare = prepare
        self.floor = float("inf")

    def _reading(self) -> float:
        value = self._read()
        self.floor = min(self.floor, value)
        return value

    def settle(
        self,
        until: Callable[[float, float], bool] | None = None,
        timeout_s: float = TIMEOUT_S,
    ) -> float:
        """The footprint after ``QUIET_S`` of idle, and, with ``until``, once it holds.

        With a condition the footprint is read every ``POLL_S`` until ``until(reading,
        floor)`` is true, or ``timeout_s`` has passed; the last reading is returned either
        way, so the caller's assertion judges what is there.
        """
        self._prepare()
        self._sleep(QUIET_S)
        reading = self._reading()
        for _ in range(round(timeout_s / POLL_S)):
            if until is None or until(reading, self.floor):
                break
            self._sleep(POLL_S)
            reading = self._reading()
        return reading
