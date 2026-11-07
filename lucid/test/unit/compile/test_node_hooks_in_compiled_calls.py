"""A hook on a graph node is never dropped by a compiled call (LCD-914).

``Node.register_hook`` / ``register_prehook`` (``y.grad_fn.register_hook(f)``)
install a runner on the node, not on a tensor slot.  The tracer asked only
``has_tensor_hooks`` about the values a call touched, and that looked at the
slot, so a node hook registered inside a compiled forward — or before the call,
on the graph an input comes from — was silently never run: the compiled
backward does not go through the node.  The call was right but for the hook's
side effect and the gradient edit it makes.

The table is where the hook sits x which kind x what became of it x every
entry.  Each cell is compared against an eager twin on the number of hook calls,
the loss, the gradients and the parameters.  A hook that is gone must compile
like a call that never had one; one that is there must send the call to eager.
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
    CALLS,
    ENTRIES,
    TOLERANCE,
    WIDTH,
    _batches,
    _close,
    _eager_call,
    _metal_ok,
    _plain_sgd,
    _sequential,
)

pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

Tensor = lucid.Tensor
Register = Callable[[Callable[..., object]], RemovableHandle]
Hooks = tuple[Callable[..., object], Callable[..., object]]


def _post_hooks(calls: list[str]) -> Hooks:
    """Two node post-hooks, each of which counts itself and scales the gradients."""

    def scaled(factor: float) -> Callable[..., object]:
        def hook(grad_inputs: tuple[Tensor | None, ...], grad_outputs: object) -> object:
            calls.append("post")
            return tuple(None if g is None else g * factor for g in grad_inputs)

        return hook

    return scaled(2.0), scaled(3.0)


def _pre_hooks(calls: list[str]) -> Hooks:
    """Two node prehooks, each of which counts itself and scales the gradient."""

    def scaled(factor: float) -> Callable[..., object]:
        def hook(grad_outputs: tuple[Tensor | None, ...]) -> object:
            calls.append("pre")
            return tuple(None if g is None else g * factor for g in grad_outputs)

        return hook

    return scaled(2.0), scaled(3.0)


KINDS: dict[str, tuple[str, Callable[[list[str]], Hooks]]] = {
    "post_hook": ("register_hook", _post_hooks),
    "prehook": ("register_prehook", _pre_hooks),
}


def _none(reg: Register, a: Callable[..., object], b: Callable[..., object]) -> None:
    return None


def _registered(reg: Register, a: Callable[..., object], b: Callable[..., object]) -> None:
    reg(a)


def _removed(reg: Register, a: Callable[..., object], b: Callable[..., object]) -> None:
    reg(a).remove()


def _removed_twice_over(reg: Register, a: Callable[..., object], b: Callable[..., object]) -> None:
    handle = reg(a)
    handle.remove()
    handle.remove()


def _removed_in_a_with(reg: Register, a: Callable[..., object], b: Callable[..., object]) -> None:
    with reg(a):
        pass


def _removed_then_readded(
    reg: Register, a: Callable[..., object], b: Callable[..., object]
) -> None:
    reg(a).remove()
    reg(b)


def _one_of_two_removed(reg: Register, a: Callable[..., object], b: Callable[..., object]) -> None:
    reg(a)
    reg(b).remove()


#: What became of the hooks on a node, and whether none is left.
STATES: dict[
    str, tuple[bool, Callable[[Register, Callable[..., object], Callable[..., object]], None]]
] = {
    "none": (True, _none),
    "registered": (False, _registered),
    "removed": (True, _removed),
    "removed_twice_over": (True, _removed_twice_over),
    "removed_in_a_with": (True, _removed_in_a_with),
    "removed_then_readded": (False, _removed_then_readded),
    "one_of_two_removed": (False, _one_of_two_removed),
}

#: Where the hook sits.  ``inside*`` are registered by the forward: on the node
#: of one of its values, of the value before that, or of one it has let go.  The
#: others are registered by the caller, before the call, on the graph the input
#: comes from (``input``: the input's producer; ``input_deep``: the node before
#: it).
SITES = ("inside", "inside_upstream", "inside_dead", "input", "input_deep")


class _Inside(nn.Module):
    """A forward that registers its hook on the node that made one of its values."""

    def __init__(self, site: str, kind: str, state: str, calls: list[str]) -> None:
        super().__init__()
        self.site, self.kind, self.state, self.calls = site, kind, state, calls

    def forward(self, x: Tensor) -> Tensor:
        y = x * 2.0
        z = y * 2.0
        attr, make = KINDS[self.kind]
        a, b = make(self.calls)
        apply = STATES[self.state][1]
        if self.site == "inside":
            fn = y.grad_fn
        elif self.site == "inside_upstream":
            fn = z.grad_fn.next_functions[0][0]
        else:
            # The node of a value nothing holds by the time the hook is set,
            # though the result still flows through it.
            held = y * 3.0
            fn = held.grad_fn
            z = z + held * 2.0
            del held
        assert fn is not None
        apply(getattr(fn, attr), a, b)
        return z


class _Plain(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * 2.0


def _hooked_input(
    site: str, kind: str, state: str, calls: list[str], batch: Tensor
) -> tuple[Tensor, Tensor | None]:
    """The call's input, with the hook registered on the graph it comes from."""
    if site.startswith("inside"):
        return batch.clone(), None
    raw = batch.clone()
    raw.requires_grad_(True)
    middle = raw * 2.0
    x = middle * 3.0
    fn = x.grad_fn if site == "input" else x.grad_fn.next_functions[0][0]
    assert fn is not None
    attr, make = KINDS[kind]
    a, b = make(calls)
    STATES[state][1](getattr(fn, attr), a, b)
    return x, raw


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    _C_engine.compile.session_cache_clear()
    yield


class _Ran:
    """What one table cell did, for the assertions that read it."""

    def __init__(self, stayed: bool | None, hook_calls: int) -> None:
        self.stayed, self.hook_calls = stayed, hook_calls


def _run(site: str, kind: str, state: str, entry_name: str) -> _Ran:
    """Run ``entry_name`` against an eager twin over ``CALLS`` batches, comparing each."""
    calls: dict[str, list[str]] = {"model": [], "twin": []}

    def build(which: str) -> nn.Module:
        if site.startswith("inside"):
            return _sequential(lambda: _Inside(site, kind, state, calls[which]))
        return _sequential(_Plain)

    model, twin = build("model"), build("twin")
    optimizer, twin_optimizer = _plain_sgd(model), _plain_sgd(twin)
    entry = ENTRIES[entry_name](model, optimizer)
    for i, batch in enumerate(_batches()[:CALLS]):
        optimizer.zero_grad()
        twin_optimizer.zero_grad()
        x, raw = _hooked_input(site, kind, state, calls["model"], batch)
        twin_x, twin_raw = _hooked_input(site, kind, state, calls["twin"], batch)
        got = entry.call(x)
        want = _eager_call(twin, twin_x, True)
        assert abs(got - want) / max(1.0, abs(want)) < TOLERANCE, f"call {i}: {got} vs {want}"
        assert len(calls["model"]) == len(
            calls["twin"]
        ), f"call {i}: {len(calls['model'])} hook calls, eager made {len(calls['twin'])}"
        if raw is not None and entry.compares_grads:
            assert _close(
                raw.grad, twin_raw.grad if twin_raw is not None else None
            ), f"call {i}: the gradient past the input's hook"
        if entry.compares_grads:
            for p, q in zip(model.parameters(), twin.parameters(), strict=True):
                assert _close(p.grad, q.grad), f"call {i}: gradient"
        if not entry.steps_itself:
            optimizer.step()
        twin_optimizer.step()
        for p, q in zip(model.parameters(), twin.parameters(), strict=True):
            assert _close(p.detach(), q.detach()), f"call {i}: parameters"
    return _Ran(entry.stayed_compiled(), len(calls["model"]))


def _cells() -> list[tuple[str, str]]:
    """Every (site, entry) the table covers.

    ``module_no_grad`` runs under ``no_grad``, where no node exists to hook, and
    ``fused_step`` refuses an input that requires grad, so a hook on the graph an
    input comes from has no fused call to test.
    """
    return [
        (site, entry)
        for site in SITES
        for entry in ENTRIES
        if entry != "module_no_grad" and not (site.startswith("input") and entry == "fused_step")
    ]


@pytest.mark.parametrize(("site", "entry"), _cells())
@pytest.mark.parametrize("state", list(STATES))
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_node_hook_is_run_and_a_removed_one_costs_no_compilation(
    site: str, kind: str, state: str, entry: str
) -> None:
    gone = STATES[state][0]
    inside = site.startswith("inside")
    if site != "input_deep" and entry == "compiled_step" and not gone:
        # The one entry that has no eager to fall back on refuses the hook (a
        # hook two nodes up an input's graph is not asked about: the eager
        # backward continuing past the call runs it).
        with pytest.raises(RuntimeError, match="hook"):
            _run(site, kind, state, entry)
        return
    # ``_run`` matches eager's hook calls, loss, gradients and parameters.
    ran = _run(site, kind, state, entry)
    if not gone:
        assert ran.hook_calls > 0, "the hook was never run"
        if inside:
            # A hook on the graph an input comes from runs in the eager backward
            # that continues past the compiled call; one on a node the call made
            # has no place in a compiled backward.
            assert ran.stayed is not True, "a node hook was dropped by a compiled call"
    elif state != "none":
        # What compiles with no hook at all compiles with one that was removed.
        assert ran.stayed == _run(site, kind, "none", entry).stayed


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("state", list(STATES))
def test_the_engine_counts_what_a_node_holds(kind: str, state: str) -> None:
    gone, apply = STATES[state]
    attr, make = KINDS[kind]
    x = lucid.ones(WIDTH, requires_grad=True)
    y = x * 2.0
    z = y * 3.0
    assert y.grad_fn is not None
    assert not _C_engine._has_tensor_hooks(_unwrap(y))
    a, b = make([])
    apply(getattr(y.grad_fn, attr), a, b)
    assert _C_engine._has_tensor_hooks(_unwrap(y)) is (not gone)
    # The hook sits on the node that made ``y``, not on what consumes it.
    assert not _C_engine._has_tensor_hooks(_unwrap(x))
    assert not _C_engine._has_tensor_hooks(_unwrap(z))
