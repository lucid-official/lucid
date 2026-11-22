"""The gamma-family benches still run: every case builds, differentiates and times on a tiny batch."""

import math

import pytest

import lucid
from lucid.test._fixtures.devices import device_dtype_params
from lucid.test.perf.distributions import bench_gamma_family as bench
from lucid.test.perf.distributions import bench_gamma_special as special


@pytest.mark.perf
@pytest.mark.parametrize(("device", "dtype"), device_dtype_params([lucid.float32]))
def test_every_case_times_on_a_tiny_batch(device: str, dtype: lucid.dtype) -> None:
    rows = bench.run(256, [device], dtype, repeats=1, cases=bench.CASE_NAMES)
    assert [r["case"] for r in rows] == list(bench.CASE_NAMES)
    for row in rows:
        for key in ("plain_ms", "current_ms", "ratio"):
            assert math.isfinite(float(str(row[key]))) and float(str(row[key])) > 0.0, row


@pytest.mark.perf
def test_an_unknown_case_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown case"):
        bench.run(16, ["cpu"], lucid.float32, repeats=1, cases=["no_such_case"])


@pytest.mark.perf
@pytest.mark.parametrize(("device", "dtype"), device_dtype_params([lucid.float32]))
@pytest.mark.parametrize("forward_only", [False, True])
def test_every_special_function_times_on_a_tiny_batch(
    device: str, dtype: lucid.dtype, forward_only: bool
) -> None:
    rows = special.run(
        256,
        [device],
        dtype,
        repeats=1,
        cases=special.CASE_NAMES,
        forward_only=forward_only,
        events=True,
    )
    assert [r["case"] for r in rows] == list(special.CASE_NAMES)
    for row in rows:
        for key in ("public_ms", "positive_ms", "ratio"):
            assert math.isfinite(float(str(row[key]))) and float(str(row[key])) > 0.0, row
        assert 0 < int(str(row["positive_events"])) < int(str(row["public_events"])), row


@pytest.mark.perf
def test_an_unknown_special_function_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown case"):
        special.run(16, ["cpu"], lucid.float32, repeats=1, cases=["no_such_case"])
