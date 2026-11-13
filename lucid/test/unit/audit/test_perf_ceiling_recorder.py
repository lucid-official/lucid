"""A perf ceiling is recorded from many hosted runs, never from one sample (LCD-867, LCD-947)."""

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


def test_headroom_lifts_the_limit_above_the_slowest_run() -> None:
    runs = [{"fft_cpu": float(v)} for v in range(1, 22)]
    got = rec.record(
        runs, {"fft_cpu": 0.5}, re.compile("fft"), q=1.0, tolerance=0.25, headroom=1.15
    )
    n, slowest, ceiling = got["fft_cpu"]
    assert (n, slowest) == (21, 21.0)
    assert ceiling * 1.25 == pytest.approx(21.0 * 1.15, rel=1e-2)


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
    assert rec.main([*args, "--allow-lower"]) == 0
    out = json.loads(thresholds.read_text())
    assert out["_comment"] == "kept" and out["add_cpu"] == 5.0
    assert out["fft_cpu"] == pytest.approx(0.010 * 1.15 / 1.25, rel=1e-2)
    assert out["_provenance"]["fft_cpu"].startswith("LCD-867: p100 of 10 hosted runs")
    # a second pass over the same runs is stable
    assert rec.main([*args, "--allow-lower"]) == 0
    assert json.loads(thresholds.read_text()) == out


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    thresholds = tmp_path / "_thresholds.json"
    thresholds.write_text(json.dumps({"fft_cpu": 1.0}))
    runs = [_run(tmp_path, i, {"fft[cpu]": 2.0}) for i in range(10)]
    before = thresholds.read_text()
    assert rec.main([str(thresholds), *map(str, runs), "--cases", "fft", "--issue", "X"]) == 0
    assert thresholds.read_text() == before


def test_write_refuses_to_lower_a_ceiling_unless_allowed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """LCD-947: ten artifacts topped out at 13.0 ms where the old ceiling came from 55 runs
    reading 16.6 ms, so recording them alone would have re-opened the flake."""
    thresholds = tmp_path / "_thresholds.json"
    thresholds.write_text(json.dumps({"sort_cpu": 0.0153}))
    runs = [_run(tmp_path, i, {"sort[cpu]": 0.0100 + 0.0003 * i}) for i in range(10)]
    before = thresholds.read_text()
    args = [str(thresholds), *map(str, runs), "--cases", "^sort", "--issue", "LCD-947", "--write"]
    assert rec.main(args) == 2
    assert "refusing to lower sort_cpu" in capsys.readouterr().err
    assert thresholds.read_text() == before
    assert rec.main([*args, "--allow-lower"]) == 0
    assert json.loads(thresholds.read_text())["sort_cpu"] < 0.0153


def test_write_raises_a_ceiling_without_the_flag(tmp_path: Path) -> None:
    thresholds = tmp_path / "_thresholds.json"
    thresholds.write_text(json.dumps({"sort_cpu": 0.0100}))
    runs = [_run(tmp_path, i, {"sort[cpu]": 0.0150}) for i in range(10)]
    args = [str(thresholds), *map(str, runs), "--cases", "^sort", "--issue", "LCD-947", "--write"]
    assert rec.main(args) == 0
    assert json.loads(thresholds.read_text())["sort_cpu"] > 0.0100


def test_a_selected_case_no_run_measured_is_reported_not_skipped_silently(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    thresholds = tmp_path / "_thresholds.json"
    thresholds.write_text(json.dumps({"fft_cpu": 1.0, "fft_renamed_cpu": 1.0}))
    runs = [_run(tmp_path, i, {"fft[cpu]": 2.0}) for i in range(10)]
    args = [str(thresholds), *map(str, runs), "--cases", "^fft", "--issue", "X"]
    assert rec.main(args) == 0
    assert "no run measured fft_renamed_cpu" in capsys.readouterr().err


def test_recording_needs_cases_and_issue(tmp_path: Path) -> None:
    thresholds = tmp_path / "_thresholds.json"
    thresholds.write_text(json.dumps({"fft_cpu": 1.0}))
    runs = [_run(tmp_path, i, {"fft[cpu]": 2.0}) for i in range(10)]
    with pytest.raises(SystemExit):
        rec.main([str(thresholds), *map(str, runs), "--cases", "fft"])


def _audit(
    runs: list[dict[str, float]], ceilings: dict[str, float], *, min_runs: int = 3
) -> dict[str, rec.Audit]:
    return rec.audit(
        runs,
        ceilings,
        re.compile(""),
        tolerance=0.25,
        headroom=1.15,
        exceed_rate=0.02,
        min_runs=min_runs,
    )


def test_audit_flags_each_way_a_ceiling_can_fail_the_runs() -> None:
    runs = [{"slow_cpu": 1.0 + 0.1 * i, "tight_cpu": 1.0, "fine_cpu": 1.0} for i in range(5)]
    got = _audit(runs, {"slow_cpu": 0.8, "tight_cpu": 0.85, "fine_cpu": 1.0, "gone_cpu": 1.0})
    assert got["slow_cpu"].flags == ("EXCEEDS",)  # limit 1.0 < 1.1 .. 1.4: 4 of 5 runs over
    assert got["slow_cpu"].exceeded == 4
    assert got["tight_cpu"].flags == ("TIGHT",)  # limit 1.0625 never failed but < 1.15 x slowest
    assert got["tight_cpu"].exceeded == 0
    assert got["fine_cpu"].flags == ()  # limit 1.25 >= 1.15
    assert got["gone_cpu"].flags == ("NO-HISTORY",) and got["gone_cpu"].runs == 0


def test_audit_counts_few_runs_but_does_not_fail_on_them() -> None:
    got = _audit([{"a_cpu": 1.0}] * 2, {"a_cpu": 2.0}, min_runs=10)
    assert got["a_cpu"].flags == ("FEW-RUNS",)


def test_audit_exit_status_follows_the_failing_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    thresholds = tmp_path / "_thresholds.json"
    runs = [_run(tmp_path, i, {"op[cpu]": 1.0}) for i in range(3)]
    args = [str(thresholds), *map(str, runs), "--audit"]
    thresholds.write_text(json.dumps({"op_cpu": 2.0, "_provenance": {"op_cpu": "LCD-0: x"}}))
    assert rec.main(args) == 0
    out = capsys.readouterr().out
    assert "op_cpu" in out and "recorded" in out and "FEW-RUNS" in out
    thresholds.write_text(json.dumps({"op_cpu": 2.0, "unread_cpu": 1.0}))
    assert rec.main(args) == 1
    assert "NO-HISTORY" in capsys.readouterr().out
    thresholds.write_text(json.dumps({"op_cpu": 0.5}))
    assert rec.main(args) == 1  # limit 0.625 < every run
    assert "EXCEEDS" in capsys.readouterr().out


def test_audit_never_writes(tmp_path: Path) -> None:
    thresholds = tmp_path / "_thresholds.json"
    thresholds.write_text(json.dumps({"op_cpu": 2.0}))
    runs = [_run(tmp_path, 0, {"op[cpu]": 1.0})]
    before = thresholds.read_text()
    rec.main([str(thresholds), *map(str, runs), "--audit"])
    assert thresholds.read_text() == before
    with pytest.raises(SystemExit):
        rec.main([str(thresholds), *map(str, runs), "--audit", "--write"])
