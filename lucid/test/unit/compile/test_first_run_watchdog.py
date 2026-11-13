"""The first run of a compiled executable has a time budget (LCD-847).

MPSGraph specializes an executable inside its first run, and one pass there
(``MemrefDeallocationPass``) never returns on a long chain of
``cat(head view, computed middle, tail view)``.  The call cannot be interrupted and the user's
process hangs with it.  The engine now runs a first run on a worker thread and waits for it
under ``LUCID_COMPILE_FIRST_RUN_TIMEOUT_S`` (120 s; 0 turns the guard off):

* the run that overruns is given up on: ``FirstRunTimeout`` (a ``RuntimeError`` whose message
  starts "MPSGraph failed:", so every entry that answers a first-run failure eagerly answers
  this one eagerly too), the signature refused for the rest of the process, and the worker left
  running -- it is inside Apple's code and cannot be stopped -- holding one core;
* four such runs open a circuit breaker: a cold first run is no longer started at all;
* a warm executable never touches a worker, so steady-state runs are as they were;
* ``compiled_step`` has no eager path and keeps raising.

A first run given up on leaves a thread behind and counts toward the breaker, so no test may do
it in the pytest process: the scenarios in ``_first_run_probes.py`` run in child processes, each
group in one child, and report one JSON line per scenario.  They make a first run last as long as
they need with the engine's test seam instead of relying on how slow MPSGraph is: a held worker
waits before it runs.  The seam only ever delays a run, and refuses to hold unless the process
says it is under test (``LUCID_COMPILE_TEST_HOOKS=1``).
"""

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.compile import _tracing
from lucid.compile._core.fallback import is_first_run_failure
from lucid.test.unit.compile._first_run_probes import _GROUPS, GROUP_ENV, STALL_BUDGET_ENV
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, assert_ratio_within

_C = _C_engine.compile
_ROOT = Path(lucid.__file__).resolve().parent.parent
#: A group of scenarios is a few seconds of work and a few seconds of start-up; a child that
#: has not exited by now is stuck.
_CHILD_CEILING_S = 240
_REAL_BUDGET_S = 3.0
_SCENARIOS = [
    (group, name) for group, scenarios in _GROUPS.items() if group != "real" for name in scenarios
]


@dataclass(frozen=True)
class _Child:
    results: dict[str, dict[str, object]]
    exited: bool
    returncode: int | None
    stderr: str


def _parse(stdout: str) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for line in stdout.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a library's own output between the probes' lines
        if isinstance(record, dict) and "scenario" in record:
            results[str(record["scenario"])] = record
    return results


@cache
def _run_group(group: str, budget_env: str | None = None, stall_s: float | None = None) -> _Child:
    env = dict(os.environ, LUCID_COMPILE_TEST_HOOKS="1", **GROUP_ENV.get(group, {}))
    env.pop("LUCID_COMPILE_FIRST_RUN_TIMEOUT_S", None)
    env.pop(STALL_BUDGET_ENV, None)
    if budget_env is not None:
        env["LUCID_COMPILE_FIRST_RUN_TIMEOUT_S"] = budget_env
    if stall_s is not None:
        env[STALL_BUDGET_ENV] = repr(stall_s)
    with subprocess.Popen(
        [sys.executable, "-m", "lucid.test.unit.compile._first_run_probes", group],
        cwd=_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            out, err = child.communicate(timeout=_CHILD_CEILING_S)
            exited = True
        except subprocess.TimeoutExpired:
            child.kill()
            out, err = child.communicate()
            exited = False
    return _Child(_parse(out), exited, child.returncode if exited else None, err[-1500:])


@pytest.fixture(autouse=True)
def _the_test_process_gives_up_on_nothing() -> Iterator[None]:
    """Fail the test that leaves a thread running in this process, or the budget changed.

    Past four, no compiled call in this process would run a cold first run at all.
    """
    before = _C.first_run_status()["abandoned"]
    yield
    _C.set_first_run_budget_s(None)
    assert _C.first_run_status()["abandoned"] == before, "a test gave up on a first run in-process"


@pytest.mark.parametrize(("group", "scenario"), _SCENARIOS, ids=[f"{g}-{n}" for g, n in _SCENARIOS])
def test_scenario(group: str, scenario: str, device_gpu_only: str) -> None:
    del device_gpu_only
    child = _run_group(group)
    assert (
        scenario in child.results
    ), f"the child never reached it (exit {child.returncode}): {child.stderr}"
    assert child.results[scenario]["ok"], child.results[scenario]["detail"]


# A scenario holds one run and gives up on it at a small budget (0.3 s).  Any other first run
# in the same scenario -- an optimizer step's own compile, the eager rerun's -- is an ordinary
# one, and a runner that needs more than the small budget for it fails the scenario on timing
# (LCD-981: ``fused_step_answers_eagerly`` on a hosted runner).  A stall budget no real first
# run can beat makes that dependence deterministic instead of a flake.
_UNBEATABLE_STALL_S = 0.001
#: The groups whose stalled runs use the small budget (``breaker`` and ``herd`` set their own).
_STALL_GROUPS = ("engine", "entries", "raises", "pinned", "inplace_target")


@pytest.mark.parametrize("group", _STALL_GROUPS)
def test_a_scenario_does_not_depend_on_the_runner_beating_the_stall_budget(
    group: str, device_gpu_only: str
) -> None:
    del device_gpu_only
    child = _run_group(group, stall_s=_UNBEATABLE_STALL_S)
    missing = set(_GROUPS[group]) - set(child.results)
    assert not missing, f"the child never reached {sorted(missing)} (exit {child.returncode})"
    failed = {name: rec["detail"] for name, rec in child.results.items() if not rec["ok"]}
    assert not failed, failed
    assert child.exited and child.returncode == 0, (child.returncode, child.stderr)


@pytest.mark.parametrize("group", [g for g in _GROUPS if g != "real"])
def test_a_child_with_given_up_runs_still_exits(group: str, device_gpu_only: str) -> None:
    """The group's worker threads (some still held, one stuck for good) cannot keep a process up."""
    del device_gpu_only
    child = _run_group(group)
    assert child.exited and child.returncode == 0, (child.returncode, child.stderr)


# The hang depends on the machine's MPSGraph: it never returns on an M1 Pro or an M4 Max, and
# returned inside a few seconds on CI's macos-26 runner, where there is nothing to give up on
# (the test skips there).  The budget is small, so this does not need the default 120 s.
@pytest.mark.slow
def test_the_real_hang_is_given_up_on_and_answered_eagerly(device_gpu_only: str) -> None:
    """LCD-847's repro through ``lucid.compile``, with the budget from the environment."""
    del device_gpu_only
    child = _run_group("real", f"{_REAL_BUDGET_S:g}")
    name = "the_real_hang_is_given_up_on_and_answered_eagerly"
    assert (
        name in child.results
    ), f"the child never reached it (exit {child.returncode}): {child.stderr}"
    assert child.results[name]["ok"], child.results[name]["detail"]
    detail = child.results["real_hang_detail"]["detail"]
    assert isinstance(detail, dict)
    if not detail["gave_up"]:
        pytest.skip(
            f"the hang does not reproduce on this machine (first call {detail['elapsed_s']:.1f} s)"
        )
    # Tracing and the compile come on top of the budget; an unbounded wait would be minutes.
    assert detail["elapsed_s"] < 20 * _REAL_BUDGET_S, detail
    # The process ended with MPSGraph's own thread still inside the pass.
    assert child.exited and child.returncode == 0, (child.returncode, child.stderr)


# ── what needs no thread left behind: run in this process ─────────────────────────────────


def _cached_executable(scale: float) -> tuple[_C.CompiledExecutable, list[object]]:
    """An executable of ``x * scale + 1`` through the session cache, and its feeds."""
    x = lucid.ones(4, device=COMPILE_DEVICE)
    step = lucid.ones_like(x)
    with lucid.no_grad(), _tracing() as tracer:
        y = x * scale + step
    lucid.eval(x, step, y)
    exe = _C.compile_or_cached(tracer.graph, tracer.external_feeds, False, [], [])
    assert exe is not None
    by_id = {int(tid): impl for tid, impl in tracer.external_feeds.items()}
    return exe, [by_id[int(tid)] for tid in exe.input_ids]


@pytest.fixture
def _clean_session_cache() -> Iterator[None]:
    _C.session_cache_clear()
    yield
    _C.session_cache_clear()


def test_a_first_run_warms_every_copy_of_the_executable(
    _clean_session_cache: None, device_gpu_only: str
) -> None:
    del device_gpu_only
    first, first_feeds = _cached_executable(2.5)
    second, _second_feeds = _cached_executable(2.5)  # a hit: a relabelled copy of the first
    assert first is not second
    assert first.first_run_state == second.first_run_state == "cold"
    _C.run_executable(first, first_feeds)
    # The ledger hangs off the MPSGraph executable the copies share.
    assert first.first_run_state == second.first_run_state == "warm"


def test_the_budget_is_not_part_of_the_cache_key(
    _clean_session_cache: None, device_gpu_only: str
) -> None:
    """The budget changes when a first run is given up on, never what is emitted (LCD-922)."""
    del device_gpu_only
    policy = _C.compile_policy()
    _C.set_first_run_budget_s(0.5)
    assert _C.compile_policy() == policy
    _cached_executable(3.5)
    _C.set_first_run_budget_s(90.0)
    assert _C.compile_policy() == policy
    _cached_executable(3.5)
    assert _C.session_cache_size() == 1


def test_a_timeout_is_a_first_run_failure() -> None:
    assert issubclass(_C.FirstRunTimeout, RuntimeError)
    assert is_first_run_failure(_C.FirstRunTimeout("run_executable: MPSGraph failed: gave up"))
    # A bare timeout, without the engine's prefix, still counts: it is recognised by type.
    assert is_first_run_failure(_C.FirstRunTimeout("took too long"))
    assert not is_first_run_failure(RuntimeError("took too long"))


def test_holding_workers_is_refused_outside_a_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUCID_COMPILE_TEST_HOOKS", raising=False)
    with pytest.raises(RuntimeError, match="test hook"):
        _C._hold_first_run_workers(True)
    _C._hold_first_run_workers(False)  # releasing is always safe


def test_the_budget_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LUCID_COMPILE_FIRST_RUN_TIMEOUT_S", raising=False)
    _C.set_first_run_budget_s(None)
    assert _C.first_run_budget_s() == 120.0
    for text, seconds in (("0.25", 0.25), ("45", 45.0), ("0", 0.0)):
        monkeypatch.setenv("LUCID_COMPILE_FIRST_RUN_TIMEOUT_S", text)
        _C.set_first_run_budget_s(None)
        assert _C.first_run_budget_s() == seconds
    capfd.readouterr()
    for unusable in ("soon", "-3", "inf", ""):
        monkeypatch.setenv("LUCID_COMPILE_FIRST_RUN_TIMEOUT_S", unusable)
        _C.set_first_run_budget_s(None)
        # An unusable value is said once and the default stands: a typo must not turn the guard
        # off, and must not fail a run that did nothing wrong.
        assert _C.first_run_budget_s() == 120.0
        assert ("ignoring" in capfd.readouterr().err) == (unusable != "")
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="seconds"):
            _C.set_first_run_budget_s(bad)


def test_the_guard_adds_no_measurable_steady_state_cost(
    _clean_session_cache: None, device_gpu_only: str
) -> None:
    """A warm run looks the ledger up and runs on the caller's thread; off, it does neither.

    Absolute, not a ratio (LCD-935): the cost is a microsecond-scale lookup on a ~0.3 ms run, so
    a ratio bound would measure the runner.  The run may cost 0.1 ms more, a third of itself: a
    per-call wait or lock of that size fails.  That a warm run never asks for a worker (a
    hand-off is ~0.02 ms, below any wall-clock bound) is held by the probe
    ``a_warm_executable_does_not_need_a_worker``.
    """
    del device_gpu_only
    exe, feeds = _cached_executable(1.5)
    _C.run_executable(exe, feeds)
    assert exe.first_run_state == "warm"

    def guarded() -> None:
        _C.set_first_run_budget_s(120.0)
        _C.run_executable(exe, feeds)

    def unguarded() -> None:
        _C.set_first_run_budget_s(0.0)
        _C.run_executable(exe, feeds)

    assert_ratio_within(
        {"guarded": guarded, "unguarded": unguarded},
        "guarded",
        "unguarded",
        bound=1.0,
        slack_ms=0.1,
        what="a warm run with the watchdog on against off",
    )
