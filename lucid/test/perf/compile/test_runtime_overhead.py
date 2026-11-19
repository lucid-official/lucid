"""Steady-state watchdog and module-signature costs."""

import pytest

import lucid
import lucid.nn._tree_version as tree_version
from lucid._C import engine as _C_engine
from lucid.compile._core import signature as sig
from lucid.test.unit.compile._helpers import assert_ratio_within
from lucid.test.unit.compile._runtime_cases import _cached_executable, _deep_model

pytestmark = pytest.mark.perf
_C = _C_engine.compile


def test_the_guard_adds_no_measurable_steady_state_cost(device_gpu_only: str) -> None:
    """A warm run looks the ledger up and runs on the caller's thread; off, it does neither.

    Absolute, not a ratio (LCD-935): the cost is a microsecond-scale lookup on a ~0.3 ms run, so
    a ratio bound would measure the runner.  The run may cost 0.1 ms more, a third of itself: a
    per-call wait or lock of that size fails.  That a warm run never asks for a worker (a
    hand-off is ~0.02 ms, below any wall-clock bound) is held by the probe
    ``a_warm_executable_does_not_need_a_worker``.
    """
    del device_gpu_only
    _C.session_cache_clear()
    try:
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
    finally:
        _C.set_first_run_budget_s(None)
        _C.session_cache_clear()


def test_keying_a_call_no_longer_costs_a_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Time, loosely: the cached key against one that walks every call."""
    model = _deep_model()
    x = lucid.randn(2, 4)
    fp = sig._param_fingerprint(model)

    def keying() -> None:
        for _ in range(20):
            sig.signature_of(model, (x,), {}, dynamic=False, param_fingerprint=fp)
            sig.model_has_active_hooks(model)

    versions = iter(range(10**9, 2 * 10**9))

    def keying_each_call_a_new_version() -> None:
        # The version is patched inside the call, so the other side keeps its cache.
        with monkeypatch.context() as patch:
            patch.setattr(tree_version, "_tree_version", lambda: next(versions))
            keying()

    assert_ratio_within(
        {"cached": keying, "walking": keying_each_call_a_new_version},
        "cached",
        "walking",
        bound=0.1,
    )
