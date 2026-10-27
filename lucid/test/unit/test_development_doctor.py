"""A broken native installation must still explain how to recover."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import doctor


def test_stale_engine_import_gives_rebuild_command() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys, types
engine = types.ModuleType('lucid._C.engine')
engine.ABI_VERSION = -1
sys.modules['lucid._C.engine'] = engine
import lucid
""",
        ],
        cwd=doctor.ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "got -1" in result.stderr
    assert doctor.REBUILD in result.stderr
    assert "tools.doctor" in result.stderr


def test_doctor_detects_source_contract_drift(tmp_path: Path) -> None:
    (tmp_path / "lucid/_C").mkdir(parents=True)
    (tmp_path / "lucid/version.py").write_text("_EXPECTED_ABI: int = 12\n")
    (tmp_path / "lucid/_C/version.h").write_text("#define LUCID_ABI_VERSION 11\n")
    assert doctor.source_abi(tmp_path) == (12, 11)


@pytest.mark.parametrize("returncode", [1, -6])
def test_doctor_reports_failed_or_crashed_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    returncode: int,
) -> None:
    def failed_probe(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["python"], returncode, "", "native load failed")

    monkeypatch.setattr(doctor.subprocess, "run", failed_probe)
    assert doctor.main(["--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    runtime = next(check for check in report["checks"] if check["name"] == "runtime")
    assert runtime["status"] == "error"
    assert f"exit={returncode}" in runtime["detail"]
    assert doctor.REBUILD in runtime["detail"]


def test_doctor_reports_hung_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def hung_probe(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired("python", 0.1)

    monkeypatch.setattr(doctor.subprocess, "run", hung_probe)
    runtime = next(check for check in doctor.diagnose(0.1) if check["name"] == "runtime")
    assert runtime["status"] == "error"
    assert "timed out" in runtime["detail"]


# ── The vendored MLX (LCD-292) ───────────────────────────────────────────

_PIN = doctor.fetch_mlx.Pin("0.32.3", "w.whl", "a" * 64, "https://x/w.whl", "26.2")


def _named(checks: list[doctor.Check], name: str) -> doctor.Check:
    return next(check for check in checks if check["name"] == name)


def test_an_mlx_below_the_interop_floor_warns_without_failing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_version = doctor.importlib.metadata.version

    def old_mlx(name: str) -> str:
        return "0.31.2" if name == "mlx" else real_version(name)

    monkeypatch.setattr(doctor.importlib.metadata, "version", old_mlx)
    check = doctor._interop_mlx_check()
    assert check["status"] == "warn"
    assert "mlx>=" in check["detail"]

    ok: doctor.Check = {"name": "python", "status": "ok", "detail": ""}
    monkeypatch.setattr(doctor, "diagnose", lambda timeout: [ok, check])
    assert doctor.main([]) == 0
    assert "[warn] mlx (interop)" in capsys.readouterr().out


def test_a_macos_below_the_pinned_floor_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor.fetch_mlx, "read_pin", lambda path: _PIN)
    monkeypatch.setattr(doctor.platform, "mac_ver", lambda: ("26.1", ("", "", ""), ""))
    monkeypatch.setattr(doctor, "_vendored_mlx_checks", lambda pin: [])
    monkeypatch.setattr(doctor, "_runtime_checks", lambda pin, timeout: [])
    check = _named(doctor.diagnose(), "platform")
    assert check["status"] == "error"
    assert "26.2" in check["detail"]


def test_an_engine_on_another_mlx_names_the_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def probe(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["python"], 0, '{"mlx_runtime": "0.32.0"}', "")

    monkeypatch.setattr(doctor.subprocess, "run", probe)
    check = _named(doctor._runtime_checks(_PIN, 1.0), "mlx_runtime")
    assert check["status"] == "error"
    assert "0.32.0" in check["detail"] and doctor.REBUILD in check["detail"]


def test_a_missing_cache_entry_names_fetch_mlx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LUCID_MLX_CACHE", str(tmp_path))
    monkeypatch.setattr(doctor.check_engine_link, "default_engine", lambda: None)
    check = _named(doctor._vendored_mlx_checks(_PIN), "mlx_cache")
    assert check["status"] == "error"
    assert "tools/fetch_mlx.py" in check["detail"]


def test_a_broken_engine_link_is_an_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LUCID_MLX_CACHE", str(tmp_path))
    monkeypatch.setattr(doctor.check_engine_link, "default_engine", lambda: tmp_path / "engine.so")
    monkeypatch.setattr(
        doctor.check_engine_link,
        "check",
        lambda engine, floor: ["liblucid_mlx.dylib: lacks 166 of the 200 MLX symbols"],
    )
    check = _named(doctor._vendored_mlx_checks(_PIN), "engine_link")
    assert check["status"] == "error"
    assert "lacks 166" in check["detail"] and "fetch_mlx" in check["detail"]
