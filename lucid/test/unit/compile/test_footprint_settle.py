"""The footprint tests wait for the release their assertion needs, and no longer (LCD-1112).

``_footprint.Footprints.settle`` is what every footprint child in ``test_cache_memory.py``
reads through.  These run it against a stand-in for the process, so the contract the Studio
needed is held without a GPU: a release that arrives late is waited for, a hold is read as
held when the wait runs out, and a baseline that was read too high does not become the floor.
"""

import pytest

from lucid.test.unit.compile._footprint import POLL_S, QUIET_S, TIMEOUT_S, Footprints


class _Process:
    """A footprint that answers with ``readings`` in turn, the last one for ever after."""

    def __init__(self, *readings: float) -> None:
        self._readings = list(readings)
        self.reads = 0
        self.slept = 0.0
        self.prepared = 0

    def script(self, *readings: float) -> None:
        """From now on answer with ``readings`` instead."""
        self._readings = list(readings)
        self.reads = 0

    def read(self) -> float:
        value = self._readings[min(self.reads, len(self._readings) - 1)]
        self.reads += 1
        return value

    def sleep(self, seconds: float) -> None:
        self.slept += seconds

    def prepare(self) -> None:
        self.prepared += 1

    def footprints(self) -> Footprints:
        return Footprints(self.read, self.sleep, self.prepare)


def test_no_condition_is_one_reading_after_the_quiet_wait() -> None:
    process = _Process(120.0, 90.0)
    assert process.footprints().settle() == 120.0
    assert (process.reads, process.slept, process.prepared) == (1, QUIET_S, 1)


def test_a_condition_that_already_holds_is_one_reading() -> None:
    process = _Process(45.0)
    assert process.footprints().settle(until=lambda reading, floor: reading < 100) == 45.0
    assert process.reads == 1


def test_a_release_that_comes_late_is_waited_for() -> None:
    """Flat for three polls (the Studio's was flat for 13 s), then it falls: that is read."""
    process = _Process(150.0, 150.0, 150.0, 45.0)
    fp = process.footprints()
    assert fp.settle(until=lambda reading, floor: reading < 100) == 45.0
    assert process.slept == pytest.approx(QUIET_S + 3 * POLL_S)
    assert fp.floor == 45.0


def test_a_hold_runs_to_the_timeout_and_is_read_as_held() -> None:
    """A release that never comes is not hidden: the last reading is what the caller judges."""
    process = _Process(150.0)
    got = process.footprints().settle(until=lambda reading, floor: reading < 100)
    assert got == 150.0
    assert process.slept == pytest.approx(QUIET_S + TIMEOUT_S)
    assert process.reads == 1 + round(TIMEOUT_S / POLL_S)


def test_the_timeout_is_the_callers() -> None:
    process = _Process(150.0)
    process.footprints().settle(until=lambda reading, floor: False, timeout_s=1.0)
    assert process.reads == 1 + 2


def test_a_baseline_read_too_high_does_not_become_the_floor() -> None:
    """The first reading still carried a released module; the floor is the lowest seen."""
    process = _Process(300.0, 120.0, 160.0)
    fp = process.footprints()
    fp.settle()
    assert fp.floor == 300.0
    fp.settle()
    fp.settle()
    assert fp.floor == 120.0


def test_the_condition_sees_the_floor_at_the_moment_it_is_asked() -> None:
    """A control that reads 0 above a dirty baseline passes once the baseline drains."""
    process = _Process(1500.0, 1500.0, 1500.0, 1500.0)
    fp = process.footprints()
    fp.settle()  # the baseline still carries the module that died: floor 1500
    seen: list[tuple[float, float]] = []

    def grew(reading: float, floor: float) -> bool:
        seen.append((reading, floor))
        return reading - floor > 700

    process.script(1500.0, 1500.0, 40.0, 1540.0)  # the old module goes, the new one stays
    assert fp.settle(until=grew) == 1540.0
    assert seen[-1] == (1540.0, 40.0)
