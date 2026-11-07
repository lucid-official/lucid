"""A removed gradient hook is no hook, under every entry (LCD-905).

``Tensor.register_hook`` installs one runner on a gradient slot, and the runner
stays there once every hook is removed (the next registration reuses it).  The
engine's ``has_tensor_hooks`` — what the tracer asks to refuse a write over a
hooked value, and what every entry asks of a traced tensor — counted that empty
runner, so ``y = s.a(x); h = y.register_hook(f); h.remove(); y += x`` ran
eagerly although no hook was left.  The answer was right; the compilation was
lost.

The table is every place a hook can sit x what became of it x every entry.  A
hook that is gone must compile and match eager; one that is there must still
send the call to eager, with eager's gradient.
"""

from collections.abc import Callable, Iterator

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.autograd._hooks import RemovableHandle
from lucid.compile._core import fallback
from lucid.test.unit.compile.test_write_on_a_tensor_the_call_made import (
    ENTRIES,
    _metal_ok,
    _run,
    _sequential,
)

pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

Tensor = lucid.Tensor
Register = Callable[[Callable[[Tensor], Tensor | None]], RemovableHandle]


def _doubled(g: Tensor) -> Tensor:
    return g * 2.0


def _tripled(g: Tensor) -> Tensor:
    return g * 3.0


def _removed(reg: Register) -> None:
    reg(_doubled).remove()


def _removed_twice(reg: Register) -> None:
    reg(_doubled).remove()
    reg(_tripled).remove()


def _removed_twice_over(reg: Register) -> None:
    handle = reg(_doubled)
    handle.remove()
    handle.remove()


def _removed_in_a_with(reg: Register) -> None:
    with reg(_doubled):
        pass


def _registered(reg: Register) -> None:
    reg(_doubled)


def _removed_then_readded(reg: Register) -> None:
    reg(_doubled).remove()
    reg(_tripled)


def _one_of_two_removed(reg: Register) -> None:
    reg(_doubled)
    reg(_tripled).remove()


#: What became of the hooks on a slot, and whether none is left.
STATES: dict[str, tuple[bool, Callable[[Register], None]]] = {
    "registered": (False, _registered),
    "removed": (True, _removed),
    "removed_twice": (True, _removed_twice),
    "removed_twice_over": (True, _removed_twice_over),
    "removed_in_a_with": (True, _removed_in_a_with),
    "removed_then_readded": (False, _removed_then_readded),
    "one_of_two_removed": (False, _one_of_two_removed),
}


class _HookedValue(nn.Module):
    """A hook on a value the call computes, or on a view of one."""

    def __init__(self, site: str, state: str) -> None:
        super().__init__()
        self.site, self.state = site, state

    def forward(self, x: Tensor) -> Tensor:
        apply = STATES[self.state][1]
        y = x * 2.0
        if self.site == "value":
            apply(y.register_hook)
            return y * 2.0
        if self.site == "value_written":
            apply(y.register_hook)
            y += x
            return y * 2.0
        if self.site == "value_beside_a_write":
            apply(y.register_hook)
            z = y * 1.0
            z += x
            return z * 2.0 + y
        v = y[:2]
        apply(v.register_hook)
        if self.site == "view":
            return v.sum() * 3.0 + y * 2.0
        read = v.sum() * 3.0
        y += x
        return y * 2.0 + read


class _HookedParameter(nn.Module):
    """A hook on a parameter, a leaf the module owns, with a write after it or not."""

    def __init__(self, state: str, writes: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.state, self.writes = state, writes

    def arm(self) -> None:
        """Hook the weight; after ``.to``, which makes the parameter it keeps."""
        STATES[self.state][1](self.linear.weight.register_hook)

    def forward(self, x: Tensor) -> Tensor:
        y = self.linear(x)
        if self.writes:
            y += x
        return y * 2.0


SITES: dict[str, Callable[[str], nn.Module]] = {
    "value": lambda state: _HookedValue("value", state),
    "value_written": lambda state: _HookedValue("value_written", state),
    "value_beside_a_write": lambda state: _HookedValue("value_beside_a_write", state),
    "view": lambda state: _HookedValue("view", state),
    "view_written": lambda state: _HookedValue("view_written", state),
    "parameter": lambda state: _HookedParameter(state, writes=False),
    "parameter_then_written": lambda state: _HookedParameter(state, writes=True),
}

# ``module_no_grad`` runs under ``no_grad``, where nothing can be hooked.
ENTRY_NAMES = [name for name in ENTRIES if name != "module_no_grad"]


def _build(middle: Callable[[], nn.Module]) -> nn.Module:
    model = _sequential(middle)
    for sub in model.modules():
        if isinstance(sub, _HookedParameter):
            sub.arm()
    return model


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    _C_engine.compile.session_cache_clear()
    yield


@pytest.mark.parametrize("entry", ENTRY_NAMES)
@pytest.mark.parametrize("state", list(STATES))
@pytest.mark.parametrize("site", list(SITES))
def test_a_hook_that_is_gone_compiles_and_one_that_is_there_runs_eagerly(
    site: str, state: str, entry: str
) -> None:
    gone = STATES[state][0]
    if entry == "compiled_step" and not gone:
        # The one entry that has no eager to fall back on refuses the hook.
        with pytest.raises(RuntimeError, match="gradient hook"):
            _run(lambda: SITES[site](state), entry, _build)
        return
    ran = _run(lambda: SITES[site](state), entry, _build)
    stayed = ran.stayed_compiled()
    if stayed is None:
        return
    if gone:
        assert stayed, "a removed hook left the call eager"
    elif not site.startswith("parameter"):
        # A parameter's hook is found before any trace: the entry runs eagerly
        # without a record, and ``_run`` already matched eager's gradient, which
        # the hook changes.
        assert not stayed, "a hook was dropped"


@pytest.mark.parametrize("state", list(STATES))
def test_the_engine_counts_what_a_slot_holds(state: str) -> None:
    gone, apply = STATES[state]
    non_leaf = lucid.ones(3, requires_grad=True) * 2.0
    leaf = lucid.ones(3, requires_grad=True)
    for tensor in (non_leaf, leaf):
        assert not _C_engine._has_tensor_hooks(_unwrap(tensor))
        apply(tensor.register_hook)
        assert _C_engine._has_tensor_hooks(_unwrap(tensor)) is (not gone)


def test_a_hook_removing_itself_during_backward_leaves_no_hook() -> None:
    x = lucid.ones(3, requires_grad=True)
    y = x * 2.0
    handles: list[RemovableHandle] = []

    def once(g: Tensor) -> Tensor:
        handles[0].remove()
        return g * 5.0

    handles.append(y.register_hook(once))
    assert _C_engine._has_tensor_hooks(_unwrap(y))
    y.sum().backward(retain_graph=True)
    assert x.grad is not None and x.grad.tolist() == [10.0, 10.0, 10.0]
    assert not _C_engine._has_tensor_hooks(_unwrap(y))
    x.grad = None
    y.sum().backward()
    assert x.grad is not None and x.grad.tolist() == [2.0, 2.0, 2.0]
