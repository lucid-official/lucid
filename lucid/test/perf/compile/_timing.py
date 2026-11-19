"""Interleaved, synchronized timings for the recorded performance tier."""

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

Calls = Mapping[str, Callable[[], object]]


def checked_twin_shape(wide: tuple[int, ...], twin: tuple[int, ...]) -> tuple[int, ...]:
    """Check equal work when the timed calls allocate their operands independently."""
    assert math.prod(wide) == math.prod(twin), (wide, twin)
    return twin


#: The smallest ratio bound a guard may state (``1.0`` itself is allowed: it says "no ratio,
#: only ``slack_ms``").  The bounds that held on the runner run after run are 1.5x and up; the
#: ones that failed there were 1.1x and a 3x that a graph at 2.1x here measured 4.4x on it.
MIN_RATIO_BOUND = 1.5
#: What a guard that states no slack of its own gets: the launch and sync jitter of a
#: millisecond-scale graph, which a ratio of a ~1 ms side does not cover.
MIN_SLACK_MS = 0.25


@dataclass(frozen=True)
class Timing:
    """The best comparison ``compare_ms`` saw: the first to hold, else the lowest ratio."""

    ratio: float
    ms: Mapping[str, float]
    attempts: int
    within: bool
    report: str


def _measure(calls: Calls, rounds: int, reps: int, warmup: int) -> dict[str, float]:
    for call in calls.values():
        for _ in range(warmup):
            call()
    best = dict.fromkeys(calls, float("inf"))
    names = list(calls)
    for rnd in range(rounds):
        for name in names if rnd % 2 == 0 else reversed(names):
            for _ in range(reps):
                t0 = time.perf_counter()
                calls[name]()
                best[name] = min(best[name], time.perf_counter() - t0)
    return {name: t * 1e3 for name, t in best.items()}


def _checked_slack(bound: float, slack_ms: float | None) -> float:
    """The slack a guard runs with, after refusing a ratio bound a runner cannot resolve."""
    if 1.0 < bound < MIN_RATIO_BOUND:
        raise ValueError(
            f"a wall-clock ratio bound of {bound}x is below what a shared runner resolves "
            f"({MIN_RATIO_BOUND}x): the same fixed graph measured 0.75x here and 1.11x on the "
            "hosted runner (LCD-935).  Bound a large cost with a ratio >= "
            f"{MIN_RATIO_BOUND}, or assert a small one on the graph; bound=1.0 with slack_ms "
            "is an absolute guard."
        )
    return MIN_SLACK_MS if slack_ms is None else slack_ms


def compare_ms(
    calls: Calls | Callable[[], Calls],
    num: str,
    den: str,
    *,
    bound: float,
    slack_ms: float | None = None,
    attempts: int = 3,
    rounds: int = 4,
    reps: int = 5,
    warmup: int = 3,
) -> Timing:
    """Hold ``calls[num]`` to ``bound * calls[den] + slack_ms``, on a loaded machine too.

    ``calls`` maps a name to a callable that does its own work *and* waits for it (the
    device runs asynchronously).  Pass a zero-argument factory instead of the mapping when
    each attempt needs a fresh build (a cache clear, a recompile under an environment).
    Each attempt warms every callable, then times them in rounds that alternate the order,
    ``reps`` calls apiece, and keeps each side's minimum.  The first attempt to hold ends
    the search; otherwise the lowest-ratio attempt is returned with ``within=False``.

    ``bound`` is ``1.0`` (absolute only) or at least ``MIN_RATIO_BOUND``; ``slack_ms`` left
    out is ``MIN_SLACK_MS``, an explicit value (``0.0`` too) is taken as stated.
    """
    slack_ms = _checked_slack(bound, slack_ms)
    best: Timing | None = None
    for attempt in range(1, attempts + 1):
        ms = _measure(calls if isinstance(calls, Mapping) else calls(), rounds, reps, warmup)
        ratio = ms[num] / ms[den]
        within = ms[num] <= bound * ms[den] + slack_ms
        report = (
            f"{num} {ms[num]:.3f} ms vs {den} {ms[den]:.3f} ms = {ratio:.2f}x "
            f"(bound {bound}x + {slack_ms} ms, attempt {attempt}/{attempts})"
        )
        if best is None or within or ratio < best.ratio:
            best = Timing(ratio, ms, attempt, within, report)
        if within:
            break
    assert best is not None, "attempts must be at least 1"
    return best


def assert_ratio_within(
    calls: Calls | Callable[[], Calls],
    num: str,
    den: str,
    *,
    bound: float,
    slack_ms: float | None = None,
    attempts: int = 3,
    rounds: int = 4,
    reps: int = 5,
    warmup: int = 3,
    what: str = "",
) -> Timing:
    """``compare_ms`` that fails the test with the report when no attempt held."""
    timing = compare_ms(
        calls,
        num,
        den,
        bound=bound,
        slack_ms=slack_ms,
        attempts=attempts,
        rounds=rounds,
        reps=reps,
        warmup=warmup,
    )
    assert timing.within, f"{what + ': ' if what else ''}{timing.report}"
    return timing
