"""No test inherits a fallback notice an earlier one consumed (LCD-915)."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from lucid.compile._core import fallback
from lucid.test._fixtures import compile_notices

_ROOT = str(Path(compile_notices.__file__).resolve().parents[3])

# Two tests in one process, in file order: the first leaves state behind (a
# consumed notice, or the verbose switch set the way the matrix helpers do), the
# second asserts the notice is still a warning.
_CONSUMES = textwrap.dedent("""
    import pytest

    from lucid.compile._core import fallback


    def test_first_consumes_the_notice():
        with pytest.warns(fallback.CompileFallbackWarning):
            fallback._fallback_notice("lowering", "op has no emitter")


    def test_second_still_hears_it():
        with pytest.warns(fallback.CompileFallbackWarning):
            fallback._fallback_notice("lowering", "op has no emitter")
    """)

_SWITCHES_VERBOSE_ON = textwrap.dedent("""
    import os

    import pytest

    from lucid.compile._core import fallback


    def test_first_sets_verbose_the_bare_way():
        os.environ["LUCID_COMPILE_VERBOSE"] = "1"


    def test_second_still_hears_it():
        with pytest.warns(fallback.CompileFallbackWarning):
            fallback._fallback_notice("lowering", "op has no emitter")
    """)

_WIRED = "from lucid.test._fixtures.compile_notices import _fresh_compile_notices  # noqa: F401\n"


def _run_pair(tmp_path: Path, conftest: str, pair: str) -> subprocess.CompletedProcess[str]:
    tmp_path.mkdir()
    (tmp_path / "conftest.py").write_text(conftest)
    (tmp_path / "test_pair.py").write_text(pair)
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-p", "no:cacheprovider", "-q"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_nothing_is_reported_when_a_test_starts() -> None:
    assert fallback._NOTICED == set()
    assert compile_notices.VERBOSE_ENV not in os.environ


@pytest.mark.parametrize("kind", ["lowering", "first-run", "hooks"])
def test_a_notice_this_test_consumes_does_not_reach_the_next(kind: str) -> None:
    """Each parametrized case must find its kind unannounced."""
    assert kind not in fallback._NOTICED
    with pytest.warns(fallback.CompileFallbackWarning):
        fallback._fallback_notice(kind, "a reason")
    assert kind in fallback._NOTICED


@pytest.mark.parametrize("pair", [_CONSUMES, _SWITCHES_VERBOSE_ON], ids=["notice", "verbose"])
def test_the_fixture_is_what_isolates_two_tests_in_one_process(tmp_path: Path, pair: str) -> None:
    """With the fixture the pair passes; without it the second fails (the control)."""
    wired = _run_pair(tmp_path / "wired", _WIRED, pair)
    assert wired.returncode == 0, (wired.stdout + wired.stderr)[-1500:]
    assert "2 passed" in wired.stdout, wired.stdout[-500:]

    bare = _run_pair(tmp_path / "bare", "", pair)
    assert bare.returncode != 0, "the control passed: the pair no longer shows the leak"
    assert "DID NOT WARN" in bare.stdout, bare.stdout[-800:]


def test_the_notice_set_is_cleared_in_place() -> None:
    """Tests hold ``_NOTICED`` by import; a rebound set would orphan them."""
    held = fallback._NOTICED
    held.add("first-run")
    compile_notices.forget_reports()
    assert fallback._NOTICED is held
    assert held == set()


def test_verbose_switched_on_the_bare_way_is_undone_between_tests() -> None:
    """The matrix helpers set ``os.environ`` directly; the fixture takes it back."""
    assert compile_notices.VERBOSE_ENV not in os.environ
    os.environ[compile_notices.VERBOSE_ENV] = "1"
    compile_notices.forget_reports()
    assert compile_notices.VERBOSE_ENV not in os.environ
