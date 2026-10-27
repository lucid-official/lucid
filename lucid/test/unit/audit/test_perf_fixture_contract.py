"""Both benchmark providers must observe outputs and expose usable timings."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lucid.test._fixtures import perf


@pytest.mark.parametrize("provider", ["fallback", "plugin", "plugin-disabled"])
def test_benchmark_provider_observes_output_and_exposes_elapsed(monkeypatch, provider: str) -> None:
    from tools import _bench_timing

    events = []
    monkeypatch.setattr(_bench_timing.lucid.metal, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(
        _bench_timing,
        "materialize",
        lambda value: events.append(("materialize", value)),
    )
    monkeypatch.setattr(perf, "_benchmark_available", lambda: provider != "fallback")
    plugin = Mock(side_effect=lambda fn: fn())
    plugin.stats = {"median": 0.0125} if provider == "plugin" else None
    request = SimpleNamespace(getfixturevalue=lambda name: plugin)
    benchmark = perf.bench.__wrapped__(request)
    assert benchmark(lambda: 42) == 42
    assert events == ["sync", ("materialize", 42), "sync"]
    assert benchmark.last_elapsed >= 0
    if provider == "plugin":
        assert benchmark.last_elapsed == 0.0125
        with pytest.raises(AssertionError, match="perf regression"):
            perf.assert_no_regression(
                "plugin-median", benchmark.last_elapsed, {"plugin-median": 0.001}
            )
    with pytest.raises(AssertionError, match="perf regression"):
        perf.assert_no_regression("negative-control", 2, {"negative-control": 1})
