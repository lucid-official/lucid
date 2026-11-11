"""Child-process probes behind ``test_first_run_watchdog.py`` (LCD-847).

Run as ``python -m lucid.test.unit.compile._first_run_probes <group>`` with
``LUCID_COMPILE_TEST_HOOKS=1``.  A group runs its scenarios in order and prints one JSON
line per scenario (``{"scenario": ..., "ok": ..., "detail": ...}``), so the parent reports
each as its own test.  They run in a child because every first run the watchdog gives up on
leaves a thread behind and counts toward its circuit breaker (four ends all first runs in
the process): a pytest process must never abandon one.

A scenario makes a first run last as long as it needs with the engine's test seam
(``_hold_first_run_workers``: a held worker waits before it starts its run) and a small
budget, so nothing here depends on how slow MPSGraph is on the machine.

A child exits through the interpreter's normal exit, with workers possibly still held: that
is the check that an abandoned worker cannot keep the process from ending.
"""

import gc
import json
import os
import signal
import sys
import threading
import time
import traceback
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import lucid
import lucid.metal
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.compile import _tracing, compiled_step, fused_step, make_step
from lucid.compile._core.fallback import CompileFallbackWarning, is_first_run_failure
from lucid.test._fixtures.compile_notices import forget_reports
from lucid.test.unit.compile._jacobi_probes import _rotation_chain

_C = _C_engine.compile
_METAL = "metal"
#: A budget a test outlasts without waiting long.
_SMALL_BUDGET_S = 0.3
#: Slack for a thread's wake-up on a loaded machine.
_SLACK_S = 2.0


class _Executable:
    """A fresh cold executable computing ``x * scale + 1`` over ``n`` floats, and its feeds.

    Through the session cache when ``cached``, else on its own.
    """

    def __init__(self, scale: float, n: int = 4, *, cached: bool = False) -> None:
        self.x = lucid.ones(n, device=_METAL)
        step = lucid.ones_like(self.x)
        with lucid.no_grad(), _tracing() as tracer:
            y = self.x * scale + step
        lucid.eval(self.x, step, y)
        if cached:
            exe = _C.compile_or_cached(tracer.graph, tracer.external_feeds, False, [], [])
        else:
            exe = _C.compile_trace(tracer.graph, tracer.external_feeds)
        assert exe is not None
        self.exe = exe
        self.scale = scale
        by_id = {int(tid): impl for tid, impl in tracer.external_feeds.items()}
        self.feeds = [by_id[int(tid)] for tid in exe.input_ids]
        self._x_impl = _unwrap(self.x)

    def feeds_for(self, x: lucid.Tensor) -> list[object]:
        """The feeds with the input replaced by ``x`` (the other feeds are constants)."""
        impl = _unwrap(x)
        return [impl if feed is self._x_impl else feed for feed in self.feeds]

    def run(self, x: lucid.Tensor | None = None) -> list[float]:
        feeds = self.feeds if x is None else self.feeds_for(x)
        return [float(v) for v in _wrap(_C.run_executable(self.exe, feeds)[0]).tolist()]

    def expect(self, x: lucid.Tensor | None = None) -> list[float]:
        base = self.x if x is None else x
        return [float(v) * self.scale + 1.0 for v in base.tolist()]


def _hold(on: bool) -> None:
    _C._hold_first_run_workers(on)


def _budget(seconds: float | None) -> None:
    _C.set_first_run_budget_s(seconds)


def _abandoned() -> int:
    return int(_C.first_run_status()["abandoned"])


def _wait_for(done: Callable[[], bool], what: str, within_s: float = 10.0) -> None:
    deadline = time.monotonic() + within_s
    while not done():
        assert time.monotonic() < deadline, f"{what} did not happen within {within_s} s"
        time.sleep(0.01)


def _raises_timeout(call: Callable[[], object]) -> tuple[_C.FirstRunTimeout, float]:
    start = time.perf_counter()
    try:
        call()
    except _C.FirstRunTimeout as timeout:
        return timeout, time.perf_counter() - start
    raise AssertionError("expected FirstRunTimeout, the call returned")


def _close(got: list[float], want: list[float]) -> None:
    assert len(got) == len(want), (got, want)
    for a, b in zip(got, want, strict=True):
        assert abs(a - b) <= 1e-5, (got, want)


# ── group "engine": the state machine, through run_executable ───────────────────────────


def _timeout_is_typed_and_refusal_is_sticky() -> None:
    e = _Executable(2.0)
    before = _abandoned()
    _hold(True)
    _budget(_SMALL_BUDGET_S)
    timeout, took = _raises_timeout(e.run)
    assert isinstance(timeout, RuntimeError)
    text = str(timeout)
    assert text.startswith("run_executable: MPSGraph failed:"), text
    assert "LUCID_COMPILE_FIRST_RUN_TIMEOUT_S" in text, text
    assert is_first_run_failure(timeout)
    assert _SMALL_BUDGET_S * 0.8 <= took < _SMALL_BUDGET_S + _SLACK_S, took
    assert e.exe.first_run_state == "refused"
    assert _abandoned() == before + 1

    # The abandoned worker finishes once released; the refusal stays, and is instant.
    _hold(False)
    time.sleep(0.5)
    assert e.exe.first_run_state == "refused"
    _budget(60.0)
    again, took_again = _raises_timeout(e.run)
    assert "given up on earlier" in str(again), again
    assert took_again < 1.0, took_again
    assert _abandoned() == before + 1


def _a_warm_executable_does_not_need_a_worker() -> None:
    e = _Executable(3.0)
    _budget(60.0)
    _close(e.run(), e.expect())
    assert e.exe.first_run_state == "warm"
    _hold(True)
    try:
        # A cold executable would wait on its held worker; a warm one never asks for one.
        _close(e.run(), e.expect())
        assert e.exe.first_run_state == "warm"
    finally:
        _hold(False)


def _a_budget_of_zero_is_the_old_path() -> None:
    e = _Executable(4.0)
    before = _abandoned()
    _budget(0.0)
    _hold(True)
    try:
        _close(e.run(), e.expect())
    finally:
        _hold(False)
        _budget(60.0)
    # Nothing was tracked: the run was an ordinary run on this thread.
    assert e.exe.first_run_state == "cold"
    assert _abandoned() == before


def _a_second_caller_joins_the_first_run() -> None:
    e = _Executable(5.0)
    before = _abandoned()
    other = lucid.full((4,), 7.0, device=_METAL)
    _budget(60.0)
    _hold(True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        starter = pool.submit(e.run)
        _wait_for(lambda: e.exe.first_run_state == "running", "the first run to start")
        joiner = pool.submit(e.run, other)
        time.sleep(0.3)  # the joiner is waiting on the same run, not starting another
        _hold(False)
        starter_answer, joiner_answer = starter.result(30), joiner.result(30)
    assert e.exe.first_run_state == "warm"
    # Each caller got its own call's outputs: the joiner ran its own input, not the starter's.
    _close(starter_answer, e.expect())
    _close(joiner_answer, e.expect(other))
    assert _abandoned() == before


def _joined_callers_share_one_deadline_and_one_count() -> None:
    e = _Executable(6.0)
    before = _abandoned()
    budget = 0.8
    _budget(budget)
    _hold(True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        starter = pool.submit(_raises_timeout, e.run)
        _wait_for(lambda: e.exe.first_run_state == "running", "the first run to start")
        time.sleep(0.4)
        joiner = pool.submit(_raises_timeout, e.run)
        _starter_timeout, _starter_took = starter.result(30)
        _joiner_timeout, joiner_took = joiner.result(30)
    assert e.exe.first_run_state == "refused"
    # Counted once, and the joiner waited for the starter's deadline, not for a fresh budget.
    assert _abandoned() == before + 1
    assert joiner_took < budget - 0.2, joiner_took
    _hold(False)
    _budget(60.0)


def _an_abandoned_run_outlives_its_executable() -> None:
    e = _Executable(8.0)
    before = _abandoned()
    _hold(True)
    _budget(_SMALL_BUDGET_S)
    _raises_timeout(e.run)
    assert _abandoned() == before + 1
    # The worker holds the only references it needs: dropping everything the caller held
    # (the executable, the session cache, the tensors) before it finishes must be safe.
    del e
    _C.session_cache_clear()
    gc.collect()
    _hold(False)
    time.sleep(1.0)
    fresh = _Executable(9.0)
    _budget(60.0)
    _close(fresh.run(), fresh.expect())


def _an_interrupt_leaves_the_run_going() -> None:
    e = _Executable(10.0)
    before = _abandoned()
    _budget(60.0)
    _hold(True)
    timer = threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGINT))
    timer.start()
    start = time.perf_counter()
    try:
        e.run()
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Ctrl-C did not reach a wait on a first run")
    finally:
        timer.cancel()
    assert time.perf_counter() - start < _SLACK_S + 0.3
    # Interrupting is not giving up: the run is still going, and the next call joins it.
    assert e.exe.first_run_state == "running"
    assert _abandoned() == before
    _hold(False)
    _wait_for(lambda: e.exe.first_run_state == "warm", "the interrupted run to finish")
    _close(e.run(), e.expect())


_ENGINE = {
    "timeout_is_typed_and_refusal_is_sticky": _timeout_is_typed_and_refusal_is_sticky,
    "a_warm_executable_does_not_need_a_worker": _a_warm_executable_does_not_need_a_worker,
    "a_budget_of_zero_is_the_old_path": _a_budget_of_zero_is_the_old_path,
    "a_second_caller_joins_the_first_run": _a_second_caller_joins_the_first_run,
    "joined_callers_share_one_deadline_and_one_count": _joined_callers_share_one_deadline_and_one_count,
    "an_abandoned_run_outlives_its_executable": _an_abandoned_run_outlives_its_executable,
    "an_interrupt_leaves_the_run_going": _an_interrupt_leaves_the_run_going,
}


# ── group "entries": every entry point but compiled_step answers eagerly ────────────────


def _trunk() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2)).to(_METAL)


def _twin(model: nn.Sequential) -> nn.Sequential:
    other = _trunk()
    other.load_state_dict(model.state_dict())
    return other


def _fallback_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, CompileFallbackWarning)]


def _stalled_first_run() -> None:
    _hold(True)
    _budget(_SMALL_BUDGET_S)


def _compile_answers_eagerly() -> None:
    lucid.manual_seed(1)
    model = _trunk()
    eager = _twin(model)
    compiled = lucid.compile(model)
    x = lucid.randn(5, 4, device=_METAL)
    before = _abandoned()
    _stalled_first_run()
    with lucid.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = compiled(x)
        _close(got.reshape(-1).tolist(), eager(x).reshape(-1).tolist())
        start = time.perf_counter()
        compiled(x)  # remembered as eager-only: no second wait
        assert time.perf_counter() - start < 1.0
    said = _fallback_messages(caught)
    assert len(said) == 1 and "did not finish within" in said[0], said
    assert _abandoned() == before + 1


def _make_step_answers_eagerly() -> None:
    lucid.manual_seed(2)
    model = _trunk()
    eager = _twin(model)
    x, t = lucid.randn(5, 4, device=_METAL), lucid.randn(5, 2, device=_METAL)
    step = make_step(model, F.mse_loss)
    before = _abandoned()
    _stalled_first_run()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss = step(x, t)
        loss.backward()
        want = F.mse_loss(eager(x), t)
        want.backward()
    _close([float(loss.item())], [float(want.item())])
    for got_p, want_p in zip(model.parameters(), eager.parameters(), strict=True):
        assert got_p.grad is not None and want_p.grad is not None
        _close(got_p.grad.reshape(-1).tolist(), want_p.grad.reshape(-1).tolist())
    said = _fallback_messages(caught)
    assert len(said) == 1 and "did not finish within" in said[0], said
    assert _abandoned() == before + 1


def _fused_step_answers_eagerly() -> None:
    lucid.manual_seed(3)
    model = _trunk()
    eager = _twin(model)
    x, t = lucid.randn(5, 4, device=_METAL), lucid.randn(5, 2, device=_METAL)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    eager_optimizer = optim.SGD(eager.parameters(), lr=0.1)
    step = fused_step(model, F.mse_loss, optimizer)
    before = _abandoned()
    _stalled_first_run()
    said: list[str] = []

    def release_on_notice(message: object, category: type[Warning], *_rest: object) -> None:
        # The eager rerun updates the parameters through a compiled optimizer step of its
        # own, which is an ordinary first run: only the fused executable is to stall.
        if issubclass(category, CompileFallbackWarning):
            said.append(str(message))
            _hold(False)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = release_on_notice
        loss = step(x, t)
    eager_optimizer.zero_grad()
    want = F.mse_loss(eager(x), t)
    want.backward()
    eager_optimizer.step()
    _close([float(loss.item())], [float(want.item())])
    for got_p, want_p in zip(model.parameters(), eager.parameters(), strict=True):
        _close(got_p.reshape(-1).tolist(), want_p.reshape(-1).tolist())
    assert len(said) == 1 and "did not finish within" in said[0], said
    assert _abandoned() == before + 1
    _budget(60.0)


_ENTRIES = {
    "compile_answers_eagerly": _compile_answers_eagerly,
    "make_step_answers_eagerly": _make_step_answers_eagerly,
    "fused_step_answers_eagerly": _fused_step_answers_eagerly,
}


# ── group "raises": compiled_step keeps raising, and signatures are per batch size ──────


def _compiled_step_raises() -> None:
    model = _trunk()
    x, t = lucid.randn(5, 4, device=_METAL), lucid.randn(5, 2, device=_METAL)
    before = _abandoned()
    _stalled_first_run()

    def loss_fn(out: lucid.Tensor) -> lucid.Tensor:
        return F.mse_loss(out, t)

    timeout, _took = _raises_timeout(lambda: compiled_step(model, x, loss_fn))
    assert "LUCID_COMPILE_FIRST_RUN_TIMEOUT_S" in str(timeout), timeout
    assert _abandoned() == before + 1
    _hold(False)
    _budget(60.0)


def _a_dynamic_batch_specializes_once_per_batch_size() -> None:
    model = nn.Sequential(nn.Linear(4, 3)).to(_METAL)
    compiled = lucid.compile(model, dynamic=True)
    before = _abandoned()
    _budget(60.0)
    with lucid.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        compiled(lucid.randn(4, 4, device=_METAL))
        # Warm for batch 4: it runs with workers held; batch 8 is a new specialization.
        _hold(True)
        compiled(lucid.randn(4, 4, device=_METAL))
    _budget(_SMALL_BUDGET_S)
    with lucid.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x8 = lucid.randn(8, 4, device=_METAL)
        got = compiled(x8)
        _close(got.reshape(-1).tolist(), model(x8).reshape(-1).tolist())
    said = _fallback_messages(caught)
    assert len(said) == 1 and "did not finish within" in said[0], said
    assert _abandoned() == before + 1
    _hold(False)
    _budget(60.0)


_RAISES = {
    "compiled_step_raises": _compiled_step_raises,
    "a_dynamic_batch_specializes_once_per_batch_size": _a_dynamic_batch_specializes_once_per_batch_size,
}


# ── group "breaker": four given up on, then no more are started ─────────────────────────


def _the_breaker_stops_new_first_runs() -> None:
    warm = _Executable(11.0)
    _budget(60.0)
    _close(warm.run(), warm.expect())
    _hold(True)
    _budget(0.2)
    for scale in (12.0, 13.0, 14.0, 15.0):
        _raises_timeout(_Executable(scale).run)
    status = _C.first_run_status()
    assert status["abandoned"] == 4 and status["breaker_open"] is True, status

    # Open: a cold first run is not started at all, however long the budget.
    _budget(30.0)
    fifth = _Executable(16.0)
    refused, took = _raises_timeout(fifth.run)
    assert "have already been given up on" in str(refused), refused
    assert is_first_run_failure(refused)
    assert took < 1.0, took
    assert fifth.exe.first_run_state == "cold"
    assert _abandoned() == 4

    # What is warm runs, and a budget of zero is the old path whatever the breaker says.
    _close(warm.run(), warm.expect())
    _budget(0.0)
    sixth = _Executable(17.0)
    _close(sixth.run(), sixth.expect())
    _budget(60.0)


def _an_entry_answers_eagerly_with_the_breaker_open() -> None:
    assert _C.first_run_status()["breaker_open"] is True
    lucid.manual_seed(4)
    model = _trunk()
    eager = _twin(model)
    x = lucid.randn(5, 4, device=_METAL)
    with lucid.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = lucid.compile(model)(x)
        _close(got.reshape(-1).tolist(), eager(x).reshape(-1).tolist())
    said = _fallback_messages(caught)
    assert len(said) == 1 and "have already been given up on" in said[0], said
    # The child now exits with four workers still held.


_BREAKER = {
    "the_breaker_stops_new_first_runs": _the_breaker_stops_new_first_runs,
    "an_entry_answers_eagerly_with_the_breaker_open": _an_entry_answers_eagerly_with_the_breaker_open,
}


# ── group "pinned": a refusal outlives the session cache's eviction (the group runs with
#    LUCID_COMPILE_MAX_CACHE=2) ──────────────────────────────────────────────────────────


def _a_refused_executable_is_not_evicted() -> None:
    refused = _Executable(18.0, 4, cached=True)
    before = _abandoned()
    _hold(True)
    _budget(_SMALL_BUDGET_S)
    _raises_timeout(refused.run)
    assert _abandoned() == before + 1
    _hold(False)
    _budget(60.0)
    # Four other structures through a cache of two: each insert evicts the oldest entry
    # that is not refused, so the refusal stays and the cache does not grow past it.
    for size in (5, 6, 7, 8):
        other = _Executable(19.0, size, cached=True)
        _close(other.run(), other.expect())
    assert _C.session_cache_size() == 2, _C.session_cache_size()
    # A module meeting the same graph again is handed the refusal, not a cold executable.
    again = _Executable(18.0, 4, cached=True)
    assert again.exe.first_run_state == "refused"
    refusal, took = _raises_timeout(again.run)
    assert "given up on earlier" in str(refusal), refusal
    assert took < 1.0, took
    assert _abandoned() == before + 1


_PINNED = {"a_refused_executable_is_not_evicted": _a_refused_executable_is_not_evicted}


# ── group "alias": LUCID_COMPILE_ALIAS_INPLACE=1 binds a result slot to the caller's own
#    buffer, which a first run given up on must never write once the caller has moved on ──


def _run_inplace(e: _Executable, target: lucid.Tensor) -> None:
    _C.run_executable_inplace(e.exe, e.feeds, [_unwrap(target)])


def _an_abandoned_first_run_leaves_the_target_alone() -> None:
    e = _Executable(2.0, 64)
    target = lucid.zeros(64, device=_METAL)
    lucid.eval(target)
    before = _abandoned()
    _hold(True)
    _budget(_SMALL_BUDGET_S)
    _raises_timeout(lambda: _run_inplace(e, target))
    assert _abandoned() == before + 1
    # The caller carries on with the target and the inputs; the held worker is then let go.
    target.add_(7.0)
    lucid.eval(target)
    _hold(False)
    time.sleep(1.5)
    assert all(v == 7.0 for v in target.tolist()), target.tolist()
    _budget(60.0)


def _a_cold_run_takes_fresh_buffers_and_a_warm_one_aliases() -> None:
    e = _Executable(3.0, 64)
    _budget(60.0)
    target = lucid.zeros(64, device=_METAL)
    lucid.eval(target)
    address = target.data_ptr()
    _run_inplace(e, target)
    # Cold: the outputs came in fresh buffers and were swapped into the target.
    assert target.data_ptr() != address
    assert e.exe.first_run_state == "warm"
    assert all(v == 4.0 for v in target.tolist()), target.tolist()
    warm_target = lucid.zeros(64, device=_METAL)
    lucid.eval(warm_target)
    address = warm_target.data_ptr()
    _run_inplace(e, warm_target)
    # Warm: the run wrote the target's own buffer.
    assert warm_target.data_ptr() == address
    assert all(v == 4.0 for v in warm_target.tolist()), warm_target.tolist()


def _an_abandoned_run_survives_its_inputs_being_freed() -> None:
    n = 1 << 20
    e = _Executable(2.0, n)
    before = _abandoned()
    _hold(True)
    _budget(_SMALL_BUDGET_S)
    _raises_timeout(e.run)
    assert _abandoned() == before + 1
    # Everything the caller held goes, the allocator gives its cache back, and other
    # buffers churn through the freed memory; only then does the worker run.
    del e
    _C.session_cache_clear()
    gc.collect()
    lucid.metal.synchronize()
    lucid.metal.empty_cache()
    junk = [lucid.full((n,), float(i), device=_METAL) for i in range(40)]
    lucid.eval(*junk)
    del junk
    lucid.metal.empty_cache()
    _hold(False)
    time.sleep(2.0)
    probe = lucid.full((n,), 3.0, device=_METAL)
    assert float((probe * 2).sum().item()) == 6.0 * n
    _budget(60.0)


_ALIAS = {
    "an_abandoned_first_run_leaves_the_target_alone": _an_abandoned_first_run_leaves_the_target_alone,
    "a_cold_run_takes_fresh_buffers_and_a_warm_one_aliases": _a_cold_run_takes_fresh_buffers_and_a_warm_one_aliases,
    "an_abandoned_run_survives_its_inputs_being_freed": _an_abandoned_run_survives_its_inputs_being_freed,
}


# ── group "herd": the breaker is read when a first run starts, not reserved ──────────────


def _the_bound_is_the_limit_plus_the_runs_in_flight() -> None:
    herd = 2 * _C.first_run_status()["abandon_limit"]
    executables = [_Executable(1.0 + i, 8) for i in range(herd)]
    # Long enough that every thread has started its run before the first one gives up.
    _budget(2.0)
    _hold(True)
    barrier = threading.Barrier(herd)

    def start(e: _Executable) -> None:
        barrier.wait()
        _raises_timeout(e.run)

    with ThreadPoolExecutor(max_workers=herd) as pool:
        for done in [pool.submit(start, e) for e in executables]:
            done.result(30)
    status = _C.first_run_status()
    # Every one of them saw the breaker closed: more were given up on than the limit.
    assert status["abandoned"] == herd > status["abandon_limit"], status
    assert status["breaker_open"] is True, status
    # And now it is open: another cold executable is not started.
    late = _Executable(99.0, 8)
    refusal, _took = _raises_timeout(late.run)
    assert f"{herd} first runs have already been given up on" in str(refusal), refusal
    assert late.exe.first_run_state == "cold"
    assert _abandoned() == herd


_HERD = {
    "the_bound_is_the_limit_plus_the_runs_in_flight": _the_bound_is_the_limit_plus_the_runs_in_flight
}


# ── group "real": the hang the guard exists for ─────────────────────────────────────────


def _the_real_hang() -> None:
    """LCD-847's repro through ``lucid.compile``: either it hangs and is given up on, or not."""
    rounds = 384
    x = lucid.randn(8, device=_METAL)
    with lucid.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        got = lucid.compile(lambda t: _rotation_chain(t, rounds))(x)
        elapsed = time.perf_counter() - start
    want = _rotation_chain(x, rounds)
    _close(got.tolist(), want.tolist())
    said = _fallback_messages(caught)
    print(
        json.dumps(
            {
                "scenario": "real_hang_detail",
                "ok": True,
                "detail": {
                    "gave_up": any("did not finish within" in m for m in said),
                    "elapsed_s": elapsed,
                    "budget_s": _C.first_run_budget_s(),
                },
            }
        ),
        flush=True,
    )


_REAL = {"the_real_hang_is_given_up_on_and_answered_eagerly": _the_real_hang}

_GROUPS: dict[str, dict[str, Callable[[], None]]] = {
    "engine": _ENGINE,
    "entries": _ENTRIES,
    "raises": _RAISES,
    "breaker": _BREAKER,
    "pinned": _PINNED,
    "inplace_target": _ALIAS,
    "herd": _HERD,
    "real": _REAL,
}
#: What a group's child process needs in its environment, before the engine starts.
GROUP_ENV = {
    "pinned": {"LUCID_COMPILE_MAX_CACHE": "2"},
    "inplace_target": {"LUCID_COMPILE_ALIAS_INPLACE": "1"},
}
GROUP_NAMES = tuple(_GROUPS)


def main(group: str) -> None:
    # A child started from a backgrounded or detached shell inherits SIGINT as ignored, and
    # Python then installs no KeyboardInterrupt handler: the interrupt scenario's signal
    # would be dropped and its wait would run the whole budget.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    for name, scenario in _GROUPS[group].items():
        forget_reports()
        try:
            scenario()
        except Exception:  # a failed scenario is reported and the next one still runs
            print(
                json.dumps(
                    {"scenario": name, "ok": False, "detail": traceback.format_exc()[-1500:]}
                ),
                flush=True,
            )
        else:
            print(json.dumps({"scenario": name, "ok": True, "detail": ""}), flush=True)
        finally:
            gc.collect()


if __name__ == "__main__":
    main(sys.argv[1])
