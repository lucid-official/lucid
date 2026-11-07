"""``set_grad_enabled`` is a function, a scope and a decorator like the reference (LCD-906).

``set_grad_enabled(mode)`` switches the mode when it is constructed.  Used in a
``with`` block it restores the previous mode on exit, also when the body raises;
as a decorator it runs each call in a scope of ``mode`` and leaves the mode alone
at decoration time.  ``no_grad`` and ``enable_grad`` share its owner, so one
instance can be entered again and each exit restores what its entry saw.  Each
scenario is written once over a module ``m`` and run on Lucid and the reference.
"""

import inspect
import threading
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid

Scenario = Callable[[ModuleType], object]


@pytest.fixture(autouse=True)
def _restore_grad_mode() -> object:
    yield
    lucid.set_grad_enabled(True)


def _function_form(m: ModuleType) -> object:
    m.set_grad_enabled(False)
    off = m.is_grad_enabled()
    m.set_grad_enabled(True)
    return [off, m.is_grad_enabled()]


def _with_form(m: ModuleType) -> object:
    with m.set_grad_enabled(False):
        inside = m.is_grad_enabled()
    return [inside, m.is_grad_enabled()]


def _with_true_inside_no_grad(m: ModuleType) -> object:
    with m.no_grad():
        with m.set_grad_enabled(True):
            inside = m.is_grad_enabled()
        after = m.is_grad_enabled()
    return [inside, after, m.is_grad_enabled()]


def _nesting(m: ModuleType) -> object:
    seen = []
    with m.set_grad_enabled(False):
        seen.append(m.is_grad_enabled())
        with m.set_grad_enabled(True):
            seen.append(m.is_grad_enabled())
            with m.set_grad_enabled(False):
                seen.append(m.is_grad_enabled())
            seen.append(m.is_grad_enabled())
        seen.append(m.is_grad_enabled())
    seen.append(m.is_grad_enabled())
    return seen


def _with_restores_after_raise(m: ModuleType) -> object:
    try:
        with m.set_grad_enabled(False):
            raise ValueError("boom")
    except ValueError:
        pass
    return m.is_grad_enabled()


def _constructed_switches_at_once(m: ModuleType) -> object:
    scope = m.set_grad_enabled(False)
    built = m.is_grad_enabled()
    with scope:
        inside = m.is_grad_enabled()
    return [built, inside, m.is_grad_enabled()]


def _decorator(m: ModuleType) -> object:
    @m.set_grad_enabled(False)
    def f() -> bool:
        return m.is_grad_enabled()

    return [f(), f.__name__, m.is_grad_enabled()]


def _decorator_true_under_no_grad(m: ModuleType) -> object:
    @m.set_grad_enabled(True)
    def f() -> bool:
        return m.is_grad_enabled()

    with m.no_grad():
        inside = f()
        after = m.is_grad_enabled()
    return [inside, after, m.is_grad_enabled()]


def _decorator_restores_after_raise(m: ModuleType) -> object:
    @m.set_grad_enabled(False)
    def f() -> None:
        raise ValueError("boom")

    try:
        f()
    except ValueError:
        pass
    return m.is_grad_enabled()


def _decorator_is_reentrant(m: ModuleType) -> object:
    @m.set_grad_enabled(False)
    def f(depth: int) -> list[bool]:
        out = [m.is_grad_enabled()]
        if depth:
            out += f(depth - 1)
            out.append(m.is_grad_enabled())
        return out

    return [f(2), m.is_grad_enabled()]


def _conditional(m: ModuleType) -> object:
    seen = []
    for training in (True, False, True):
        with m.set_grad_enabled(training):
            seen.append(m.is_grad_enabled())
    return seen + [m.is_grad_enabled()]


def _result_requires_grad(m: ModuleType) -> object:
    w = m.ones(2, requires_grad=True)
    with m.set_grad_enabled(False):
        off = w * 2
    on = w * 2
    return [off.requires_grad, on.requires_grad]


def _no_grad_and_enable_grad_forms(m: ModuleType) -> object:
    @m.no_grad()
    def quiet() -> bool:
        return m.is_grad_enabled()

    @m.enable_grad()
    def loud() -> bool:
        return m.is_grad_enabled()

    with m.no_grad():
        both = [quiet(), loud(), m.is_grad_enabled()]
    return both + [m.is_grad_enabled()]


_SCENARIOS: dict[str, Scenario] = {
    "function": _function_form,
    "with": _with_form,
    "true-inside-no-grad": _with_true_inside_no_grad,
    "nesting": _nesting,
    "with-raise": _with_restores_after_raise,
    "constructed-switches": _constructed_switches_at_once,
    "decorator": _decorator,
    "decorator-true-under-no-grad": _decorator_true_under_no_grad,
    "decorator-raise": _decorator_restores_after_raise,
    "decorator-reentrant": _decorator_is_reentrant,
    "conditional": _conditional,
    "result-requires-grad": _result_requires_grad,
    "no-grad-enable-grad": _no_grad_and_enable_grad_forms,
}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, ref: ModuleType) -> None:
    scenario = _SCENARIOS[name]
    assert scenario(lucid) == scenario(ref)
    lucid.set_grad_enabled(True)
    ref.set_grad_enabled(True)


@pytest.mark.parity
def test_signature_matches_the_reference(ref: ModuleType) -> None:
    mine = inspect.signature(lucid.set_grad_enabled)
    theirs = inspect.signature(ref.set_grad_enabled)
    assert list(mine.parameters) == ["mode"]
    assert mine.parameters["mode"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert theirs.parameters["mode"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


@pytest.mark.parity
def test_a_non_bool_is_refused_like_the_reference(ref: ModuleType) -> None:
    for bad in (1, 0, "x", None):
        with pytest.raises(TypeError):
            ref.set_grad_enabled(bad)
        with pytest.raises(TypeError, match="expects a bool"):
            lucid.set_grad_enabled(bad)  # type: ignore[arg-type]  # deliberate wrong type
    assert lucid.is_grad_enabled()


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_scenarios_leave_the_thread_in_grad_mode(name: str) -> None:
    _SCENARIOS[name](lucid)
    assert lucid.is_grad_enabled()


def test_with_no_longer_leaves_grad_disabled() -> None:
    with lucid.set_grad_enabled(False):
        assert not lucid.is_grad_enabled()
    assert lucid.is_grad_enabled()


def test_the_scope_restores_a_mode_that_was_already_off() -> None:
    with lucid.no_grad():
        with lucid.set_grad_enabled(False):
            pass
        assert not lucid.is_grad_enabled()
    assert lucid.is_grad_enabled()


def test_mode_attribute_is_recorded() -> None:
    assert lucid.set_grad_enabled(False).mode is False
    assert lucid.set_grad_enabled(True).mode is True
    assert lucid.no_grad().mode is False
    assert lucid.enable_grad().mode is True


def test_decoration_alone_changes_nothing() -> None:
    @lucid.set_grad_enabled(False)
    def f() -> None: ...

    assert lucid.is_grad_enabled()
    f()
    assert lucid.is_grad_enabled()


def test_a_scope_can_be_entered_again_after_it_exited() -> None:
    scope = lucid.set_grad_enabled(False)
    seen = []
    for _ in range(2):
        with scope:
            seen.append(lucid.is_grad_enabled())
        seen.append(lucid.is_grad_enabled())
    assert seen == [False, True, False, True]


@pytest.mark.parametrize("make", [lucid.no_grad, lucid.enable_grad])
def test_one_instance_nests_and_reenters(make: Callable[[], object]) -> None:
    # The reference keeps one saved state per instance, so re-entering the same
    # object leaves the thread in the inner state; the owner keeps a stack.
    scope = make()
    start = lucid.is_grad_enabled()
    with scope:
        with scope:
            pass
    assert lucid.is_grad_enabled() is start
    with scope:
        pass
    assert lucid.is_grad_enabled() is start


def test_a_reused_instance_restores_each_level() -> None:
    scope = lucid.no_grad()
    seen = []
    with scope:
        seen.append(lucid.is_grad_enabled())
        with scope:
            seen.append(lucid.is_grad_enabled())
        seen.append(lucid.is_grad_enabled())
    seen.append(lucid.is_grad_enabled())
    assert seen == [False, False, False, True]


def test_the_mode_is_per_thread() -> None:
    seen: list[bool] = []

    def worker() -> None:
        seen.append(lucid.is_grad_enabled())
        with lucid.set_grad_enabled(False):
            seen.append(lucid.is_grad_enabled())
        seen.append(lucid.is_grad_enabled())

    with lucid.set_grad_enabled(False):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert not lucid.is_grad_enabled()
    assert seen == [True, False, True]


def test_the_function_form_does_not_leak_into_another_thread() -> None:
    seen: list[bool] = []
    t = threading.Thread(target=lambda: (lucid.set_grad_enabled(False), seen.append(False)))
    t.start()
    t.join()
    assert lucid.is_grad_enabled()
    assert seen == [False]


def test_inference_mode_false_re_enables_inside_a_set_grad_enabled_scope() -> None:
    with lucid.set_grad_enabled(False):
        with lucid.inference_mode(False):
            assert lucid.is_grad_enabled()
        assert not lucid.is_grad_enabled()
    assert lucid.is_grad_enabled()


def test_autograd_namespace_exposes_the_same_classes() -> None:
    assert lucid.autograd.set_grad_enabled is lucid.set_grad_enabled
    assert lucid.autograd.no_grad is lucid.no_grad
    assert lucid.autograd.enable_grad is lucid.enable_grad
