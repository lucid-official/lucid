"""The gamma-family bench still runs: every case builds, differentiates and times on a tiny batch."""

import math

import pytest

import lucid
from lucid.test._fixtures.devices import device_dtype_params
from lucid.test.perf.distributions import bench_gamma_family as bench


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
