"""Forward hooks under ``lucid.compile`` (LCD-776), CPU and Metal.

A trace runs the forward through ``Module.__call__``, so a module's forward hooks and pre-hooks run
while tracing and their tensor ops are recorded with the rest.  What a replay cannot do is run
Python, so :class:`~lucid.compile._core.hook_contract.HookContract` traces a hook only when it
rewrites tensors and acts by no side effect, and puts the hook set and the state each hook reads in
the key.  The classes below pin each half:

* ``TestTraced`` — a hooked module compiles and equals eager; a hook added, removed, replaced or
  reordered, and every kind of state a hook branches on, retraces.
* ``TestSideEffects`` — a hook that acts by side effect still runs eagerly, each call, with the
  fallback notice.
* ``TestTraining`` — ``make_step`` (monolithic and segmented) and ``fused_step`` trace hooks and
  train like eager.

The CPU has no executables: there the same cases check the key and the eager result.
"""

from collections.abc import Callable
from functools import partial

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile._core import fallback
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.compile._core.hook_contract import HookContract, traceable_hook
from lucid.compile._core.signature import hook_block, model_has_active_hooks, signature_of
from lucid.nn.hooks import RemovableHandle

_IN, _OUT = 4, 3


def _linear(device: str) -> nn.Linear:
    lucid.manual_seed(0)
    return nn.Linear(_IN, _OUT).to(device).eval()


def _x(device: str, rows: int = 2, cols: int = _IN) -> lucid.Tensor:
    lucid.manual_seed(1)
    return lucid.randn(rows, cols, device=device)


def _close(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool(lucid.allclose(a, b, atol=1e-5))


def _entries(compiled: lucid.compile.CompiledModule[..., lucid.Tensor], device: str) -> int:
    """How many executables ``compiled`` holds; the CPU holds none, whatever the key says."""
    return int(compiled.cache_info()["entries"]) if device == "metal" else 0


@traceable_hook
def _forward(m: nn.Module, _: object, out: lucid.Tensor) -> lucid.Tensor:
    return out * 2.0


@traceable_hook
def _pre(m: nn.Module, args: tuple[lucid.Tensor, ...]) -> tuple[lucid.Tensor]:
    return (args[0] + 1.0,)


@traceable_hook
def _forward_kw(
    m: nn.Module, args: object, kwargs: dict[str, object], out: lucid.Tensor
) -> lucid.Tensor:
    return out - 3.0


@traceable_hook
def _pre_kw(
    m: nn.Module, args: tuple[lucid.Tensor, ...], kwargs: dict[str, object]
) -> tuple[tuple[lucid.Tensor], dict[str, object]]:
    return (args[0] * 0.5,), kwargs


_KINDS: dict[str, Callable[[nn.Module], RemovableHandle]] = {
    "forward": lambda m: m.register_forward_hook(_forward),
    "pre": lambda m: m.register_forward_pre_hook(_pre),
    "forward-kwargs": lambda m: m.register_forward_hook(_forward_kw, with_kwargs=True),
    "pre-kwargs": lambda m: m.register_forward_pre_hook(_pre_kw, with_kwargs=True),
}


@traceable_hook
class _Scale:
    """A callable-object hook whose factor is its attribute."""

    def __init__(self, k: float) -> None:
        self.k = k

    def __call__(self, m: nn.Module, args: object, out: lucid.Tensor) -> lucid.Tensor:
        return out * self.k


@traceable_hook
class _Bias:
    """A hook that adds a tensor it holds."""

    def __init__(self, bias: lucid.Tensor) -> None:
        self.bias = bias

    def __call__(self, m: nn.Module, args: object, out: lucid.Tensor) -> lucid.Tensor:
        return out + self.bias


@pytest.mark.usefixtures("inference")
class TestTraced:
    @pytest.mark.parametrize("kind", list(_KINDS))
    def test_a_hooked_module_compiles_and_equals_eager(self, kind: str, device: str) -> None:
        m = _linear(device)
        x = _x(device)
        plain = m(x)
        _KINDS[kind](m)
        assert not model_has_active_hooks(m)
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile(m)
        got = compiled(x)
        assert _close(got, m(x))
        assert not _close(got, plain)  # the hook did something
        if device == "metal":
            assert not compiled.cache_info()["eager_only"]
            assert _C_engine.compile.session_cache_size() > 0
            assert _entries(compiled, device) == 1
            assert _close(compiled(x), got)
            assert _entries(compiled, device) == 1  # a hit

    def test_a_hook_on_a_submodule(self, device: str) -> None:
        net = nn.Sequential(nn.Linear(_IN, 5), nn.ReLU(), nn.Linear(5, _OUT)).to(device).eval()
        x = _x(device)
        net[0].register_forward_hook(_forward)
        net[2].register_forward_pre_hook(_pre)
        compiled = lucid.compile(net)
        assert _close(compiled(x), net(x))
        assert _entries(compiled, device) == (1 if device == "metal" else 0)

    def test_adding_and_removing_a_hook_retraces(self, device: str) -> None:
        m = _linear(device)
        x = _x(device)
        compiled = lucid.compile(m)
        bare = compiled(x)
        assert _entries(compiled, device) == (1 if device == "metal" else 0)
        handle = m.register_forward_hook(_forward)
        hooked = compiled(x)
        assert _close(hooked, m(x))
        assert not _close(hooked, bare)  # a stale graph would have answered `bare`
        if device == "metal":
            assert _entries(compiled, device) == 2
        handle.remove()
        assert _close(compiled(x), bare)
        if device == "metal":
            assert _entries(compiled, device) == 2  # the bare graph is reused
        # The same hook registered again is another registration: another graph, same answer.
        again = m.register_forward_hook(_forward)
        assert _close(compiled(x), hooked)
        again.remove()

    def test_a_replaced_hook_retraces(self, device: str) -> None:
        m = _linear(device)
        x = _x(device)
        compiled = lucid.compile(m)
        first = m.register_forward_hook(traceable_hook(lambda mod, args, out: out * 2.0))
        a = compiled(x)
        first.remove()
        second = m.register_forward_hook(traceable_hook(lambda mod, args, out: out * 5.0))
        b = compiled(x)
        assert _close(b, m(x))
        assert not _close(a, b)
        second.remove()

    def test_the_order_of_the_hooks_is_in_the_key(self, device: str) -> None:
        m = _linear(device)
        x = _x(device)
        compiled = lucid.compile(m)
        m.register_forward_hook(traceable_hook(lambda mod, args, out: out + 1.0))
        m.register_forward_hook(traceable_hook(lambda mod, args, out: out * 3.0))
        a = compiled(x)
        assert _close(a, m(x))
        late = m.register_forward_hook(
            traceable_hook(lambda mod, args, out: out - 7.0), prepend=True
        )
        b = compiled(x)
        assert _close(b, m(x))
        assert not _close(a, b)
        late.remove()
        assert _close(compiled(x), a)

    def test_a_cell_a_hook_closes_over_is_in_the_key(self, device: str) -> None:
        m = _linear(device)
        x = _x(device)
        config = {"scale": 2.0}
        m.register_forward_hook(traceable_hook(lambda mod, args, out: out * config["scale"]))
        compiled = lucid.compile(m)
        a = compiled(x)
        config["scale"] = 4.0  # an in-place change moves no tree version
        b = compiled(x)
        assert _close(a * 2.0, b)
        assert _close(b, m(x))

    def test_an_attribute_of_a_hook_object_is_in_the_key(self, device: str) -> None:
        m = _linear(device)
        x = _x(device)
        hook = _Scale(2.0)
        m.register_forward_hook(hook)
        compiled = lucid.compile(m)
        a = compiled(x)
        hook.k = 3.0
        b = compiled(x)
        assert _close(b, m(x))
        assert _close(a * 1.5, b)

    def test_a_tensor_a_hook_holds_is_read_live_and_rebinding_it_retraces(
        self, device: str
    ) -> None:
        m = _linear(device)
        x = _x(device)
        hook = _Bias(lucid.ones(_OUT, device=device))
        m.register_forward_hook(hook)
        compiled = lucid.compile(m)
        a = compiled(x)
        hook.bias.copy_(hook.bias * 10.0)  # new values, the same tensor: the same graph
        b = compiled(x)
        assert _close(b, m(x))
        assert not _close(a, b)
        assert _entries(compiled, device) == (1 if device == "metal" else 0)
        hook.bias = lucid.full((_OUT,), -4.0, device=device)  # another tensor: another graph
        assert _close(compiled(x), m(x))
        if device == "metal":
            assert _entries(compiled, device) == 2

    def test_the_state_a_bound_method_hook_reads_is_in_the_key(self, device: str) -> None:
        class Switch(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.on = True

            @traceable_hook
            def rewrite(self, mod: nn.Module, args: object, out: lucid.Tensor) -> lucid.Tensor:
                return out * 2.0 if self.on else out

        m = _linear(device)
        x = _x(device)
        switch = Switch()  # not in m's tree: the hook reaches it
        m.register_forward_hook(switch.rewrite)
        compiled = lucid.compile(m)
        a = compiled(x)
        switch.on = False
        b = compiled(x)
        assert _close(b, m(x))
        assert _close(a, b * 2.0)
        switch.on = True
        assert _close(compiled(x), a)

    def test_the_hook_set_is_part_of_the_module_state(self) -> None:
        m = _linear("cpu")
        bare = signature_of(m, (_x("cpu"),), {}, dynamic=False)
        handle = m.register_forward_hook(_forward)
        hooked = signature_of(m, (_x("cpu"),), {}, dynamic=False)
        assert hooked != bare
        handle.remove()
        assert signature_of(m, (_x("cpu"),), {}, dynamic=False) == bare


def _appends(calls: list[int]) -> Callable[..., lucid.Tensor]:
    def hook(m: nn.Module, args: object, out: lucid.Tensor) -> lucid.Tensor:
        calls.append(1)
        return out * 2.0

    return hook


def _item_store(calls: list[int]) -> Callable[..., lucid.Tensor]:
    def hook(m: nn.Module, args: object, out: lucid.Tensor) -> lucid.Tensor:
        calls[0] += 1
        return out * 2.0

    return hook


def _attribute_store(calls: list[int]) -> Callable[..., lucid.Tensor]:
    box = _Box()

    def hook(m: nn.Module, args: object, out: lucid.Tensor) -> lucid.Tensor:
        box.n += 1
        calls.append(box.n)
        return out * 2.0

    return hook


def _rebinds(calls: list[int]) -> Callable[..., lucid.Tensor]:
    count = 0

    def hook(m: nn.Module, args: object, out: lucid.Tensor) -> lucid.Tensor:
        nonlocal count
        count += 1
        calls.append(count)
        return out * 2.0

    return hook


def _observer(calls: list[int]) -> Callable[..., None]:
    def hook(m: nn.Module, args: object, out: lucid.Tensor) -> None:
        calls.append(len(out))

    return hook


def _prints(calls: list[int]) -> Callable[..., lucid.Tensor]:
    def hook(m: nn.Module, args: object, out: lucid.Tensor) -> lucid.Tensor:
        calls.append(0)
        print(out.shape)
        return out

    return hook


class _Box:
    n = 0


_SIDE_EFFECTS: dict[str, Callable[[list[int]], Callable[..., object]]] = {
    "mutator": _appends,
    "item-store": _item_store,
    "attribute-store": _attribute_store,
    "nonlocal": _rebinds,
    "observer": _observer,
    "print": _prints,
}


def _hidden_in_helper(calls: list[int]) -> Callable[..., lucid.Tensor]:
    """A hook that reads as a pure rewrite; its side effect lives in a helper it calls."""

    def record(out: lucid.Tensor) -> None:
        calls.append(len(out))

    def hook(m: nn.Module, args: object, out: lucid.Tensor) -> lucid.Tensor:
        record(out)
        return out * 2.0

    return hook


@pytest.mark.usefixtures("inference")
class TestOnlyDeclaredHooksAreTraced:
    @pytest.mark.parametrize("marked", [False, True], ids=["unmarked", "marked-but-impure"])
    @pytest.mark.parametrize("name", list(_SIDE_EFFECTS))
    def test_a_side_effect_runs_eagerly_every_call_and_says_so(
        self, name: str, marked: bool, device: str, capfd: pytest.CaptureFixture[str]
    ) -> None:
        m = _linear(device)
        x = _x(device)
        calls: list[int] = [0] if name == "item-store" else []
        hook = _SIDE_EFFECTS[name](calls)
        m.register_forward_hook(traceable_hook(hook) if marked else hook)
        assert model_has_active_hooks(m)
        reason = hook_block(m)
        assert reason is not None and "hook" in reason
        compiled = lucid.compile(m)
        fallback._NOTICED.discard("hooks")
        with pytest.warns(CompileFallbackWarning, match="forward hook"):
            got = compiled(x)
        before = len(calls) if name != "item-store" else calls[0]
        compiled(x)
        compiled(x)
        after = len(calls) if name != "item-store" else calls[0]
        assert after - before == 2  # the hook ran on every call: nothing was replayed
        assert _close(got, m(x)) or name == "observer"
        assert compiled.cache_info()["entries"] == 0
        capfd.readouterr()

    def test_a_pure_looking_user_hook_is_not_traced_without_a_declaration(
        self, device: str
    ) -> None:
        m = _linear(device)
        x = _x(device)
        m.register_forward_hook(lambda mod, args, out: out * 2.0)
        reason = hook_block(m)
        assert reason is not None and "not declared" in reason
        compiled = lucid.compile(m)
        fallback._NOTICED.discard("hooks")
        with pytest.warns(CompileFallbackWarning, match="not declared"):
            assert _close(compiled(x), m(x))
        assert compiled.cache_info()["entries"] == 0

    def test_a_side_effect_hidden_in_a_helper_runs_on_every_call(self, device: str) -> None:
        m = _linear(device)
        x = _x(device)
        calls: list[int] = []
        m.register_forward_hook(_hidden_in_helper(calls))
        compiled = lucid.compile(m)
        for _ in range(3):
            assert _close(compiled(x), m(x))
        assert len(calls) == 6  # three compiled calls and three eager ones, none replayed
        assert compiled.cache_info()["entries"] == 0

    def test_the_marker_and_the_guard(self) -> None:
        def clean(m: nn.Module, args: object, out: lucid.Tensor) -> lucid.Tensor:
            return (lambda t: t * 2.0)(out) + 1.0

        assert HookContract.blocker(clean) is not None  # unmarked
        assert HookContract.blocker(traceable_hook(clean)) is None
        assert HookContract.blocker(_Scale(2.0)) is None  # a marked class
        assert HookContract.blocker(partial(traceable_hook(clean))) is None
        assert HookContract.blocker(print) is not None  # a builtin: not Python we can read

    def test_a_marked_hook_does_not_hide_an_impure_one_behind_a_wrapper(self) -> None:
        calls: list[int] = []

        def observe(m: nn.Module, args: object, out: lucid.Tensor) -> None:
            calls.append(1)

        def wrapper(m: nn.Module, args: object, out: lucid.Tensor) -> lucid.Tensor | None:
            observe(m, args, out)
            return None

        def forwarding(m: nn.Module, args: object, out: lucid.Tensor) -> None:
            return wrapper(m, args, out)

        assert HookContract.blocker(traceable_hook(forwarding)) is not None

    def test_backward_and_global_hooks_still_run_eagerly(self, device: str) -> None:
        m = _linear(device)
        handle = m.register_full_backward_hook(lambda mod, gi, go: None)
        assert hook_block(m) is not None and "backward" in hook_block(m)  # type: ignore[operator]
        handle.remove()
        assert hook_block(m) is None
        gh = nn.hooks.register_module_forward_hook(lambda mod, args, out: out)
        try:
            assert hook_block(m) is not None
        finally:
            gh.remove()


def _mlp(device: str) -> nn.Sequential:
    lucid.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)
    net[0].register_forward_hook(traceable_hook(lambda mod, args, out: out * 2.0))
    net[2].register_forward_pre_hook(traceable_hook(lambda mod, args: (args[0] + 1.0,)))
    return net


def _loss(y: lucid.Tensor) -> lucid.Tensor:
    return (y * y).sum()


def _same_grads(a: nn.Module, b: nn.Module) -> bool:
    pairs = list(zip(a.parameters(), b.parameters(), strict=True))
    return all(
        p.grad is not None
        and q.grad is not None
        and bool(lucid.allclose(p.grad, q.grad, atol=1e-4))
        for p, q in pairs
    )


class TestTraining:
    def test_make_step_traces_the_hooks_and_trains_like_eager(self, device: str) -> None:
        net, ref = _mlp(device), _mlp(device)
        x = _x(device, 4, 8)
        step = lucid.compile.make_step(net, _loss)
        loss = step(x)
        loss.backward()
        want = _loss(ref(x))
        want.backward()
        assert abs(float(loss.item()) - float(want.item())) < 1e-3 * max(1.0, float(want.item()))
        assert _same_grads(net, ref)
        if device == "metal":
            assert len(step.cache) == 1

    def test_a_segmented_step(self, device: str) -> None:
        net, ref = _mlp(device), _mlp(device)
        x = _x(device, 4, 8)
        step = lucid.compile.make_step(net, _loss, segments=2)
        step(x).backward()
        _loss(ref(x)).backward()
        assert _same_grads(net, ref)
        if device == "metal":
            assert all(len(c) == 1 for c in step.cache)

    def test_fused_step(self, device_gpu_only: str) -> None:
        net, ref = _mlp(device_gpu_only), _mlp(device_gpu_only)
        x = _x(device_gpu_only, 4, 8)
        step = lucid.compile.fused_step(net, _loss, lucid.optim.SGD(net.parameters(), lr=0.1))
        got = step(x)
        opt = lucid.optim.SGD(ref.parameters(), lr=0.1)
        want = _loss(ref(x))
        want.backward()
        opt.step()
        assert abs(float(got.item()) - float(want.item())) < 1e-3 * max(1.0, float(want.item()))
        assert step.cache_info()["entries"] == 1
        for a, b in zip(net.parameters(), ref.parameters(), strict=True):
            assert _close(a, b)
