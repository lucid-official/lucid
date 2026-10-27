"""Performance fixtures — built on top of ``pytest-benchmark``.

The ``bench`` fixture wraps the upstream ``benchmark`` fixture so tests
degrade to a no-op timing call when ``pytest-benchmark`` isn't
installed.  This keeps the rest of the suite green even in the
slimmest dev environments while making the perf tier first-class
when the dep is present.
"""

import functools
import importlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tools import _bench_timing


@functools.lru_cache(maxsize=1)
def _benchmark_available() -> bool:
    try:
        importlib.import_module("pytest_benchmark")
        return True
    except ImportError:
        return False


@pytest.fixture
def bench(request: pytest.FixtureRequest) -> Callable[..., Any]:
    """Return a ``benchmark``-compatible callable.

    Both providers materialize returned Lucid outputs and synchronize. Callers
    must return outputs and gradients they want timed, not discard lazy work.
    ``last_elapsed`` is the plugin median, or a one-shot fallback/disabled-plugin
    observation. These are host-observed end-to-end timings, not kernel timings.
    """
    benchmark = request.getfixturevalue("benchmark") if _benchmark_available() else None

    def measured(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        def observed() -> Any:
            _bench_timing.lucid.metal.synchronize()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            _bench_timing.materialize(out)
            _bench_timing.lucid.metal.synchronize()
            measured.last_elapsed = time.perf_counter() - t0  # type: ignore[attr-defined]
            return out

        out = benchmark(observed) if benchmark is not None else observed()
        stats = getattr(benchmark, "stats", None)
        if stats:
            measured.last_elapsed = float(stats["median"])  # type: ignore[attr-defined]
        return out

    return measured


def load_thresholds(area_dir: Path) -> dict[str, float]:
    """Load golden-timing thresholds from ``<area_dir>/_thresholds.json``.

    Returns an empty dict when the file is missing — perf tests can
    then run without enforcing a regression threshold (useful for
    bootstrapping a new area)."""
    path = area_dir / "_thresholds.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def assert_no_regression(
    name: str, elapsed_s: float, thresholds: dict[str, float], *, tolerance: float = 0.25
) -> None:
    """Fail when ``elapsed_s`` exceeds ``thresholds[name] * (1 + tolerance)``.

    Missing entries are treated as "no threshold" so new perf tests
    don't have to gate the suite until a baseline is recorded.
    """
    threshold = thresholds.get(name)
    if threshold is None:
        return
    limit = threshold * (1.0 + tolerance)
    if elapsed_s > limit:
        raise AssertionError(
            f"perf regression {name!r}: {elapsed_s * 1e3:.3f} ms "
            f"> {limit * 1e3:.3f} ms (threshold {threshold * 1e3:.3f} ms "
            f"+{int(tolerance * 100)}%)"
        )
