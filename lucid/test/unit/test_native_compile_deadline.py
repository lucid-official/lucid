"""The watchdog that turns a native compile blow-up into a failure (LCD-786)."""

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from lucid.test._fixtures import deadline

_UNIT = Path(__file__).resolve().parent


def test_compile_directory_has_a_deadline() -> None:
    got = deadline.deadline_seconds(_UNIT / "compile" / "test_anything.py")
    assert got == deadline.COMPILE_DIR_DEADLINE_S


def test_slow_compile_test_gets_the_long_deadline() -> None:
    got = deadline.deadline_seconds(_UNIT / "compile" / "test_anything.py", slow=True)
    assert got == deadline.SLOW_DEADLINE_S


def test_explicit_marker_wins() -> None:
    assert deadline.deadline_seconds(_UNIT / "compile" / "x.py", explicit=7.0, slow=True) == 7.0


def test_module_that_names_make_step_has_one_and_others_do_not(tmp_path: Path) -> None:
    reaches = tmp_path / "test_reaches.py"
    reaches.write_text("from lucid.compile import make_step\n")
    plain = tmp_path / "test_plain.py"
    plain.write_text("import lucid\n")
    assert deadline.deadline_seconds(reaches) == deadline.ELSEWHERE_DEADLINE_S
    assert deadline.deadline_seconds(plain) is None


def test_scale_zero_switches_it_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(deadline.SCALE_ENV, "0")
    assert deadline.deadline_seconds(_UNIT / "compile" / "x.py", explicit=5.0) is None
    monkeypatch.setenv(deadline.SCALE_ENV, "2")
    assert deadline.deadline_seconds(_UNIT / "compile" / "x.py") == 2 * deadline.COMPILE_DIR_DEADLINE_S


def test_overrunning_test_ends_the_run_with_a_stack(tmp_path: Path) -> None:
    """Through the real fixture, pytest capture included: the run fails fast and says where."""
    (tmp_path / "conftest.py").write_text(
        "from lucid.test._fixtures.deadline import (\n"
        "    _deadline_stderr_fd, _native_compile_deadline,\n"
        ")  # noqa: F401\n"
    )
    (tmp_path / "test_stuck.py").write_text(
        textwrap.dedent(
            """
            import time

            import pytest


            @pytest.mark.deadline(1)
            def test_stuck_in_native_code():
                time.sleep(60)
            """
        )
    )
    root = str(Path(deadline.__file__).resolve().parents[3])
    start = time.perf_counter()
    done = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-p", "no:cacheprovider", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    elapsed = time.perf_counter() - start
    assert done.returncode != 0
    assert elapsed < 30, f"the watchdog let it run {elapsed:.0f} s"
    out = done.stdout + done.stderr
    assert "Timeout" in out and "test_stuck_in_native_code" in out, out[-2000:]
