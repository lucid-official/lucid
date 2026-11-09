"""A perf ceiling is recorded from many hosted runs, never from one sample (LCD-867)."""

import json
import re
from pathlib import Path

import pytest

from tools import record_perf_ceilings as rec


def _run(tmp_path: Path, n: int, medians: dict[str, float]) -> Path:
    path = tmp_path / f"run{n}.json"
    path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {"name": f"test_{k}", "stats": {"median": v}} for k, v in medians.items()
                ]
            }
        )
    )
    return path


def test_case_key_matches_the_name_the_tests_assert_with() -> None:
    assert rec.case_key("test_fft_64[cpu]") == "fft_64_cpu"
    assert rec.case_key("test_roll_256x4096[metal]") == "roll_256x4096_metal"
    assert rec.case_key("not_a_test") is None


def test_quantile_interpolates_and_ignores_input_order() -> None:
    assert rec.quantile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.5
    assert rec.quantile([5.0], 0.95) == 5.0
    assert rec.quantile(list(map(float, range(101))), 0.95) == 95.0


def test_ceiling_puts_the_checked_limit_at_the_quantile() -> None:
    runs = [{"fft_cpu": float(v)} for v in range(1, 22)]  # 1..21
    got = rec.record(runs, {"fft_cpu": 0.5}, re.compile("fft"), q=0.95, tolerance=0.25)
    n, limit, ceiling = got["fft_cpu"]
    assert (n, limit) == (21, 20.0)
    assert ceiling * 1.25 == pytest.approx(limit, rel=1e-2)


def test_only_selected_cases_present_in_the_file_are_recorded() -> None:
    runs = [{"fft_cpu": 1.0, "add_cpu": 1.0, "new_cpu": 1.0}] * 3
    got = rec.record(
        runs, {"fft_cpu": 9.0, "add_cpu": 9.0}, re.compile("^fft"), q=0.5, tolerance=0.0
    )
    assert list(got) == ["fft_cpu"]


def test_refuses_too_few_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    thresholds = tmp_path / "_thresholds.json"
    thresholds.write_text(json.dumps({"fft_cpu": 1.0}))
    runs = [_run(tmp_path, i, {"fft[cpu]": 2.0}) for i in range(3)]
    before = thresholds.read_text()
    args = [str(thresholds), *map(str, runs), "--cases", "fft", "--issue", "LCD-0", "--write"]
    assert rec.main(args) == 2
    assert "single-sample" in capsys.readouterr().err
    assert thresholds.read_text() == before


def test_write_records_ceilings_with_provenance_and_keeps_the_rest(tmp_path: Path) -> None:
    thresholds = tmp_path / "_thresholds.json"
    thresholds.write_text(json.dumps({"_comment": "kept", "fft_cpu": 1.0, "add_cpu": 5.0}))
    runs = [_run(tmp_path, i, {"fft[cpu]": 0.001 * (i + 1), "add[cpu]": 9.9}) for i in range(10)]
    args = [str(thresholds), *map(str, runs), "--cases", "^fft", "--issue", "LCD-867", "--write"]
    assert rec.main(args) == 0
    out = json.loads(thresholds.read_text())
    assert out["_comment"] == "kept" and out["add_cpu"] == 5.0
    assert out["fft_cpu"] == pytest.approx(0.00955 / 1.25, rel=1e-2)
    assert out["_provenance"]["fft_cpu"].startswith("LCD-867: p95 of 10 hosted runs")
    # a second pass over the same runs is stable
    assert rec.main(args) == 0
    assert json.loads(thresholds.read_text()) == out


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    thresholds = tmp_path / "_thresholds.json"
    thresholds.write_text(json.dumps({"fft_cpu": 1.0}))
    runs = [_run(tmp_path, i, {"fft[cpu]": 2.0}) for i in range(10)]
    before = thresholds.read_text()
    assert rec.main([str(thresholds), *map(str, runs), "--cases", "fft", "--issue", "X"]) == 0
    assert thresholds.read_text() == before
