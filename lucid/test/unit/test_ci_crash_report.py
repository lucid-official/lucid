"""Unit tests for ``tools/ci_crash_report.py`` (LCD-1022)."""

import json
import os
from pathlib import Path

import pytest

from tools import ci_crash_report


def _ips_text(
    *,
    app_name: str = "Python",
    faulting_thread: int | bool = 1,
    threads: list[dict[str, object]] | None = None,
    images: list[dict[str, object]] | None = None,
    exception: dict[str, object] | None = None,
) -> str:
    header = {"app_name": app_name, "bug_type": "309"}
    if images is None:
        images = [
            {"name": "libsystem_c.dylib"},
            {"name": "engine.cpython-314-darwin.so"},
            {"name": "liblucid_mlx.dylib"},
        ]
    if threads is None:
        threads = [
            {
                "id": 100,
                "queue": "com.apple.main-thread",
                "frames": [
                    {
                        "imageIndex": 0,
                        "imageOffset": 0x1234,
                        "symbol": "__cxa_finalize_ranges",
                        "symbolLocation": 44,
                    },
                    {
                        "imageIndex": 0,
                        "imageOffset": 0x1400,
                        "symbol": "exit",
                        "symbolLocation": 20,
                    },
                ],
            },
            {
                "id": 101,
                "name": "worker-1",
                "frames": [
                    {
                        "imageIndex": 1,
                        "imageOffset": 0xABCD0,
                        "symbol": "lucid::compile::worker_run",
                        "symbolLocation": 128,
                    },
                    {"imageIndex": 2, "imageOffset": 0x4560},
                ],
            },
        ]
    body = {
        "exception": exception or {"type": "EXC_BAD_ACCESS", "signal": "SIGSEGV"},
        "termination": {"signal": 11, "indicator": "Segmentation fault: 11"},
        "faultingThread": faulting_thread,
        "usedImages": images,
        "threads": threads,
    }
    return f"{json.dumps(header)}\n{json.dumps(body)}\n"


def test_format_report_prints_faulting_with_offsets_and_main_thread() -> None:
    out = ci_crash_report.format_report(_ips_text(faulting_thread=1))
    assert "process: Python |" in out
    assert "faulting thread 1 (name=worker-1):" in out
    assert "lucid::compile::worker_run + 128 (imageOffset=0xabcd0)" in out
    assert "liblucid_mlx.dylib                   imageOffset=0x4560" in out
    assert "main thread 0 (queue=com.apple.main-thread):" in out
    assert "__cxa_finalize_ranges + 44 (imageOffset=0x1234)" in out
    assert "exit + 20 (imageOffset=0x1400)" in out


def test_format_report_caps_main_thread_at_15_and_faulting_at_30() -> None:
    main_frames = [{"imageIndex": 0, "imageOffset": i, "symbol": f"main_f{i}"} for i in range(25)]
    fault_frames = [{"imageIndex": 1, "imageOffset": i, "symbol": f"fault_f{i}"} for i in range(40)]
    text = _ips_text(
        faulting_thread=1,
        threads=[
            {"queue": "com.apple.main-thread", "frames": main_frames},
            {"frames": fault_frames},
        ],
    )
    out = ci_crash_report.format_report(text)
    assert "fault_f29 (imageOffset=0x1d)" in out
    assert "fault_f30" not in out
    assert "main_f14 (imageOffset=0xe)" in out
    assert "main_f15" not in out


def test_format_report_does_not_duplicate_when_main_thread_faults() -> None:
    out = ci_crash_report.format_report(_ips_text(faulting_thread=0))
    assert "faulting thread 0 (queue=com.apple.main-thread, main thread):" in out
    assert "\nmain thread " not in out


def test_format_report_handles_bool_json_fields_and_null_image_name() -> None:
    text = _ips_text(
        faulting_thread=True,
        images=[{"name": None}, {"name": "wrong_image"}],
        threads=[
            {
                "queue": "com.apple.main-thread",
                "frames": [
                    {
                        "imageIndex": True,
                        "imageOffset": True,
                        "symbol": "fn_sym",
                        "symbolLocation": True,
                    },
                    {"imageIndex": 0, "imageOffset": 0x10},
                ],
            },
            {"frames": [{"imageIndex": 1, "imageOffset": 0x20}]},
        ],
    )
    out = ci_crash_report.format_report(text)
    # faultingThread=True must not coerce to 1; falls back to 0 (main thread).
    assert "faulting thread 0 (queue=com.apple.main-thread, main thread):" in out
    assert "wrong_image" not in out
    assert "?                                    fn_sym (imageOffset=?)" in out
    assert "?                                    imageOffset=0x10" in out


def test_format_report_handles_jetsam_and_malformed_text() -> None:
    jetsam = _ips_text(threads=[])
    assert "not a crash report" in ci_crash_report.format_report(jetsam)
    malformed = "not json\nline 2\nline 3\n"
    assert ci_crash_report.format_report(malformed) == "not json\nline 2\nline 3"


def test_collect_and_stage_filters_by_marker_and_keeps_all_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dir_a = tmp_path / "user_reports"
    dir_b = tmp_path / "sys_reports"
    dir_a.mkdir()
    dir_b.mkdir()

    old = dir_a / "old.ips"
    old.write_text(_ips_text(app_name="OldProc"), encoding="utf-8")
    os.utime(old, ns=(1_000_000_000, 1_000_000_000))

    marker = tmp_path / "marker"
    marker.write_text("", encoding="utf-8")
    os.utime(marker, ns=(2_000_000_000, 2_000_000_000))

    # A directory matching *.ips must be ignored by find_reports.
    (dir_a / "not_a_file.ips").mkdir()

    first = dir_a / "crash.ips"
    first.write_text(_ips_text(app_name="FirstCrash"), encoding="utf-8")
    os.utime(first, ns=(3_000_000_000, 3_000_000_000))

    # Same basename in second directory tests collision disambiguation.
    second = dir_b / "crash.ips"
    second.write_text(_ips_text(app_name="SecondCrash"), encoding="utf-8")
    os.utime(second, ns=(4_000_000_000, 4_000_000_000))

    dest = tmp_path / "staged"
    rc = ci_crash_report.main(
        [
            "--since",
            str(marker),
            "--copy-to",
            str(dest),
            "--wait-seconds",
            "0",
            "--search-dir",
            str(dir_a),
            "--search-dir",
            str(dir_b),
        ]
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "OldProc" not in stdout
    assert "FirstCrash" in stdout
    assert "SecondCrash" in stdout
    assert stdout.index("FirstCrash") < stdout.index("SecondCrash")

    staged = sorted(p.name for p in dest.glob("*.ips"))
    assert staged == ["crash.ips", "crash_1.ips"]
    assert "FirstCrash" in (dest / "crash.ips").read_text(encoding="utf-8")
    assert "SecondCrash" in (dest / "crash_1.ips").read_text(encoding="utf-8")


def test_unreadable_report_does_not_abort_staging_remaining_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    unreadable = reports_dir / "a_root.ips"
    unreadable.write_text(_ips_text(app_name="RootOnly"), encoding="utf-8")
    os.utime(unreadable, ns=(1_000_000_000, 1_000_000_000))
    os.chmod(unreadable, 0o000)

    readable = reports_dir / "b_lucid.ips"
    readable.write_text(_ips_text(app_name="LucidProc"), encoding="utf-8")
    os.utime(readable, ns=(2_000_000_000, 2_000_000_000))

    dest = tmp_path / "staged"
    try:
        rc = ci_crash_report.main(
            [
                "--copy-to",
                str(dest),
                "--wait-seconds",
                "0",
                "--search-dir",
                str(reports_dir),
            ]
        )
    finally:
        os.chmod(unreadable, 0o644)

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "could not read" in stdout
    assert "could not stage" in stdout
    assert "LucidProc" in stdout
    assert [p.name for p in dest.glob("*.ips")] == ["b_lucid.ips"]


def test_missing_since_marker_and_nonpositive_poll_interval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    preexisting = reports_dir / "old.ips"
    preexisting.write_text(_ips_text(app_name="PreExisting"), encoding="utf-8")

    missing_marker = tmp_path / "does_not_exist"
    assert (
        ci_crash_report.collect_reports([reports_dir], since=missing_marker, wait_seconds=1.0) == []
    )
    assert "marker missing" in capsys.readouterr().out

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert ci_crash_report.collect_reports([empty_dir], wait_seconds=1.0, poll_interval=0.0) == []
    assert ci_crash_report.collect_reports([empty_dir], wait_seconds=1.0, poll_interval=-5.0) == []


def test_main_reports_when_no_crash_report_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    dest = tmp_path / "staged"
    rc = ci_crash_report.main(
        [
            "--copy-to",
            str(dest),
            "--wait-seconds",
            "0",
            "--search-dir",
            str(empty_dir),
        ]
    )
    assert rc == 0
    assert "no crash report was written" in capsys.readouterr().out
    assert not dest.exists()
