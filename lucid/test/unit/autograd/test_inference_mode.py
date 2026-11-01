"""``inference_mode(mode)`` and ``is_inference_mode_enabled`` match the reference (LCD-743).

The reference signature is ``inference_mode(mode=True)``; it is a context
manager and a decorator (bare and called).  ``mode=False`` clears the
inference flag *and* re-enables grad tracking, even inside a ``no_grad``
region.  Each scenario is written once over a module ``m`` and run on Lucid
and the reference.

Not covered: ``Tensor.is_inference()``, the "inference tensors cannot be
saved for backward" error and the in-place check outside the mode.  Those
need a TensorImpl flag in the engine (LCD-743 FIX PROPOSAL).
"""

import inspect
import threading
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

Scenario = Callable[[ModuleType], object]


def _state(m: ModuleType) -> tuple[bool, bool]:
    return (m.is_inference_mode_enabled(), m.is_grad_enabled())


def _plain(m: ModuleType) -> object:
    return _state(m)


def _default_scope(m: ModuleType) -> object:
    with m.inference_mode():
        inside = _state(m)
    return [inside, _state(m)]


def _explicit_true(m: ModuleType) -> object:
    with m.inference_mode(True):
        inside = _state(m)
    return [inside, _state(m)]


def _keyword_false(m: ModuleType) -> object:
    with m.inference_mode(mode=False):
        inside = _state(m)
    return [inside, _state(m)]


def _false_under_no_grad(m: ModuleType) -> object:
    with m.no_grad():
        with m.inference_mode(False):
            inside = _state(m)
        after = _state(m)
    return [inside, after, _state(m)]


def _true_under_enable_grad(m: ModuleType) -> object:
    with m.enable_grad():
        with m.inference_mode():
            inside = _state(m)
        after = _state(m)
    return [inside, after]


def _false_inside_true(m: ModuleType) -> object:
    with m.inference_mode():
        outer = _state(m)
        with m.inference_mode(False):
            inner = _state(m)
        restored = _state(m)
    return [outer, inner, restored, _state(m)]


def _deep_nesting(m: ModuleType) -> object:
    seen = []
    with m.inference_mode():
        with m.inference_mode(False):
            with m.inference_mode():
                seen.append(_state(m))
            seen.append(_state(m))
        seen.append(_state(m))
    seen.append(_state(m))
    return seen


def _enable_grad_inside(m: ModuleType) -> object:
    with m.inference_mode():
        with m.enable_grad():
            inside = _state(m)
        after = _state(m)
    return [inside, after]


def _bare_decorator(m: ModuleType) -> object:
    @m.inference_mode
    def f() -> tuple[bool, bool]:
        return _state(m)

    return [f(), f.__name__, _state(m)]


def _called_decorator(m: ModuleType) -> object:
    @m.inference_mode()
    def f() -> tuple[bool, bool]:
        return _state(m)

    return [f(), f.__name__, _state(m)]


def _false_decorator(m: ModuleType) -> object:
    @m.inference_mode(False)
    def f() -> tuple[bool, bool]:
        return _state(m)

    with m.inference_mode():
        inside = f()
        after = _state(m)
    return [inside, after, f(), _state(m)]


def _decorator_is_reentrant(m: ModuleType) -> object:
    @m.inference_mode()
    def f(depth: int) -> list[tuple[bool, bool]]:
        out = [_state(m)]
        if depth:
            out += f(depth - 1)
            out.append(_state(m))
        return out

    return [f(2), _state(m)]


def _decorator_restores_after_raise(m: ModuleType) -> object:
    @m.inference_mode
    def f() -> None:
        raise ValueError("boom")

    try:
        f()
    except ValueError:
        pass
    return _state(m)


def _scope_restores_after_raise(m: ModuleType) -> object:
    try:
        with m.inference_mode():
            raise ValueError("boom")
    except ValueError:
        pass
    return _state(m)


def _reused_instance(m: ModuleType) -> object:
    ctx = m.inference_mode()
    seen = []
    with ctx:
        seen.append(_state(m))
        with ctx:
            seen.append(_state(m))
        seen.append(_state(m))
    seen.append(_state(m))
    return seen


def _requires_grad_of_results(m: ModuleType) -> object:
    w = m.ones(2, requires_grad=True)
    with m.inference_mode():
        y = w * 2
        with m.inference_mode(False):
            z = w * 2
    return [y.requires_grad, z.requires_grad]


_SCENARIOS: dict[str, Scenario] = {
    "plain": _plain,
    "default-scope": _default_scope,
    "explicit-true": _explicit_true,
    "keyword-false": _keyword_false,
    "false-under-no-grad": _false_under_no_grad,
    "true-under-enable-grad": _true_under_enable_grad,
    "false-inside-true": _false_inside_true,
    "deep-nesting": _deep_nesting,
    "enable-grad-inside": _enable_grad_inside,
    "bare-decorator": _bare_decorator,
    "called-decorator": _called_decorator,
    "false-decorator": _false_decorator,
    "decorator-reentrant": _decorator_is_reentrant,
    "decorator-raise": _decorator_restores_after_raise,
    "scope-raise": _scope_restores_after_raise,
    "result-requires-grad": _requires_grad_of_results,
}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, ref: ModuleType) -> None:
    scenario = _SCENARIOS[name]
    assert scenario(lucid) == scenario(ref)


@pytest.mark.parity
def test_signature_matches_the_reference(ref: ModuleType) -> None:
    mine = inspect.signature(lucid.inference_mode)
    theirs = inspect.signature(ref.inference_mode)
    assert list(mine.parameters) == list(theirs.parameters) == ["mode"]
    assert mine.parameters["mode"].default is theirs.parameters["mode"].default is True


def test_a_reused_instance_nests_correctly() -> None:
    # The reference keeps one saved state per instance, so re-entering the
    # same object leaves the thread stuck in the inner state; Lucid keeps a
    # stack and restores each level.
    assert _reused_instance(lucid) == [
        (True, False),
        (True, False),
        (True, False),
        (False, True),
    ]


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_scenarios_leave_the_thread_unchanged(name: str) -> None:
    _SCENARIOS[name](lucid)
    assert (lucid.is_inference_mode_enabled(), lucid.is_grad_enabled()) == (False, True)


def test_mode_attribute_is_recorded() -> None:
    assert lucid.inference_mode().mode is True
    assert lucid.inference_mode(False).mode is False


def test_non_bool_non_callable_is_rejected() -> None:
    with pytest.raises(TypeError, match="bool or a callable"):
        lucid.inference_mode(3)  # type: ignore[call-overload]  # deliberate wrong type


def test_autograd_namespace_exposes_the_predicate() -> None:
    assert lucid.autograd.is_inference_mode_enabled is lucid.is_inference_mode_enabled


def test_the_flag_is_per_thread() -> None:
    seen: list[bool] = []
    with lucid.inference_mode():
        t = threading.Thread(target=lambda: seen.append(lucid.is_inference_mode_enabled()))
        t.start()
        t.join()
        assert lucid.is_inference_mode_enabled()
    assert seen == [False]
