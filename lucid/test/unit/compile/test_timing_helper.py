"""``compare_ms`` fails on a large cost and ignores a runner's noise (LCD-935, LCD-981).

A timing guard that flakes costs a CI rerun, and one that never fails guards nothing.  These
tests drive the helper with a scripted clock: each call advances it by the cost it is scripted
to have, so what a bound lets through is exact, on any machine, and nothing here sleeps.
"""

from collections.abc import Callable

import pytest

from lucid.test.unit.compile import _helpers
from lucid.test.unit.compile._helpers import (
    MIN_RATIO_BOUND,
    MIN_SLACK_MS,
    Calls,
    assert_ratio_within,
    compare_ms,
)


class _Clock:
    """Moves only when a scripted call runs."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    scripted = _Clock()
    monkeypatch.setattr(_helpers, "time", scripted)
    return scripted


def _cost(clock: _Clock, ms: float | Callable[[int], float]) -> Callable[[], None]:
    """A call that takes ``ms`` of the clock (``ms(i)`` on its i-th call)."""
    calls = [0]

    def call() -> None:
        clock.now += (ms(calls[0]) if callable(ms) else ms) / 1e3
        calls[0] += 1

    return call


def _pair(clock: _Clock, num_ms: float | Callable[[int], float], den_ms: float) -> Calls:
    return {"num": _cost(clock, num_ms), "den": _cost(clock, den_ms)}


def test_a_2x_slowdown_fails_and_a_1_4x_one_does_not(clock: _Clock) -> None:
    held = compare_ms(_pair(clock, 14.0, 10.0), "num", "den", bound=1.5)
    assert held.within and held.ratio == pytest.approx(1.4)
    broken = compare_ms(_pair(clock, 20.0, 10.0), "num", "den", bound=1.5)
    assert not broken.within and broken.ratio == pytest.approx(2.0)


def test_a_spike_on_one_side_is_not_a_regression(clock: _Clock) -> None:
    # Every other call of the guarded side stalls for ten times its cost, the last one of the
    # attempt included: a loaded runner's scheduler.  Each side keeps its minimum, so the spikes
    # never reach the ratio (one attempt, so a retry cannot be what rescues it).
    spiky = _pair(clock, lambda i: 100.0 if i % 2 == 0 else 10.0, 10.0)
    timing = compare_ms(spiky, "num", "den", bound=1.5, attempts=1)
    assert timing.within and timing.ratio == pytest.approx(1.0)


def test_a_fresh_build_gets_another_attempt(clock: _Clock) -> None:
    builds: list[Calls] = []

    def build() -> Calls:
        # The first build lands on a slow executable, the second does not.
        builds.append(_pair(clock, 30.0 if not builds else 10.0, 10.0))
        return builds[-1]

    timing = compare_ms(build, "num", "den", bound=1.5)
    assert timing.within and timing.attempts == 2 and len(builds) == 2


def test_a_persistent_slowdown_fails_every_attempt(clock: _Clock) -> None:
    builds: list[Calls] = []

    def build() -> Calls:
        builds.append(_pair(clock, 30.0, 10.0))
        return builds[-1]

    timing = compare_ms(build, "num", "den", bound=1.5, attempts=3)
    assert not timing.within and len(builds) == 3


def test_a_guard_with_no_slack_of_its_own_gets_the_floor(clock: _Clock) -> None:
    # 0.3 ms against 0.1 ms is 3x, but a sub-millisecond side is launch jitter: the floor holds it.
    assert 1.5 * 0.1 + MIN_SLACK_MS >= 0.3
    timing = compare_ms(_pair(clock, 0.3, 0.1), "num", "den", bound=1.5)
    assert timing.within and f"+ {MIN_SLACK_MS} ms" in timing.report
    # ... and the floor is a floor: a cost past it still fails.
    assert not compare_ms(_pair(clock, 0.5, 0.1), "num", "den", bound=1.5).within


@pytest.mark.parametrize("slack_ms", [0.0, 0.05])
def test_a_slack_stated_by_the_guard_is_taken_as_stated(clock: _Clock, slack_ms: float) -> None:
    timing = compare_ms(_pair(clock, 0.3, 0.1), "num", "den", bound=1.5, slack_ms=slack_ms)
    assert not timing.within and f"+ {slack_ms} ms" in timing.report


def test_a_bound_of_one_is_an_absolute_guard(clock: _Clock) -> None:
    assert compare_ms(_pair(clock, 0.28, 0.2), "num", "den", bound=1.0, slack_ms=0.1).within
    assert not compare_ms(_pair(clock, 0.35, 0.2), "num", "den", bound=1.0, slack_ms=0.1).within


@pytest.mark.parametrize("bound", [1.01, 1.1, 1.3, MIN_RATIO_BOUND - 0.01])
def test_a_bound_a_shared_runner_cannot_resolve_is_refused(bound: float) -> None:
    def never() -> None:
        raise AssertionError("a refused guard must not run anything")

    with pytest.raises(ValueError, match="shared runner"):
        compare_ms({"num": never, "den": never}, "num", "den", bound=bound)


@pytest.mark.parametrize("bound", [0.1, 0.6, 1.0, MIN_RATIO_BOUND, 5.0])
def test_a_speedup_or_a_coarse_bound_is_accepted(clock: _Clock, bound: float) -> None:
    assert compare_ms(_pair(clock, 1.0, 10.0), "num", "den", bound=bound, slack_ms=0.0).within


def test_the_failure_message_carries_the_measurement(clock: _Clock) -> None:
    with pytest.raises(AssertionError, match=r"mine: num 20\.000 ms vs den 10\.000 ms = 2\.00x"):
        assert_ratio_within(_pair(clock, 20.0, 10.0), "num", "den", bound=1.5, what="mine")
