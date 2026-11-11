"""A bound method of a plain object shows its modules to the compile walk (LCD-912).

``compiled_decode_step(rig.forward, ...)`` left a batch norm's
``num_batches_tracked`` at 1 where eager counted 4: ``rig`` is not a module,
and the walk looked at a bound method's ``__code__`` cells and globals but not
at the object it is bound to — while a closure over ``rig`` was walked.  The
statistics travelled as feeds, so only the counter (and, below, a swapped
parameter) went wrong, silently.

Each case reaches the same two modules, a ``Linear`` and a training-mode
``BatchNorm1d``, from a plain object by one of the routes a callable has, and
runs through each entry that walks a callable (``lucid.compile`` and
``compiled_decode_step``; ``make_step`` / ``fused_step`` take a module).  The
compiled run is held to the eager one: outputs, the counter, the running
statistics, and a parameter replaced halfway through.
"""

import types
from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.compile._core.signature import _indirect_module_paths, _path_children, _resolve_path
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.test._fixtures.devices import metal_available
from lucid.utils.cache import StaticCache

pytestmark = [
    pytest.mark.skipif(not metal_available(), reason="metal unavailable"),
    pytest.mark.filterwarnings("ignore::lucid.compile._core.fallback.CompileFallbackWarning"),
]

_W, _LEN, _CALLS = 4, 8, 4


class _Plain:
    """What every holder offers: a pure body, and a decode step on the cache."""

    cache: StaticCache

    def body(self, hidden: lucid.Tensor) -> lucid.Tensor:
        raise NotImplementedError

    def decode(self, hidden: lucid.Tensor, cache_position: lucid.Tensor) -> lucid.Tensor:
        raise NotImplementedError


class _Rig(_Plain):
    """A plain object holding modules and a cache: not a ``Module`` itself."""

    def __init__(self) -> None:
        lucid.manual_seed(0)
        self.cache = StaticCache(_LEN)
        self.lin = nn.Linear(_W, _W).to("metal")
        self.bn = nn.BatchNorm1d(_W).to("metal").train()

    def body(self, hidden: lucid.Tensor) -> lucid.Tensor:
        return self.bn(self.lin(hidden)) * 1.0

    def decode(self, hidden: lucid.Tensor, cache_position: lucid.Tensor) -> lucid.Tensor:
        # One row, not the batch mean: a training batch norm's output has a zero mean per
        # feature, so a mean here would make every cached key roundoff and the check vacuous.
        key = (self.body(hidden)[0] * 1.0).reshape(1, 1, 1, _W)
        keys, _ = self.cache.update(key, key, 0, {"cache_position": cache_position})
        return keys * 1.0


class _Outer(_Plain):
    """A plain object whose modules sit one plain object further down."""

    def __init__(self, inner: _Rig) -> None:
        self.inner = inner
        self.cache = inner.cache

    def body(self, hidden: lucid.Tensor) -> lucid.Tensor:
        return self.inner.body(hidden)

    def decode(self, hidden: lucid.Tensor, cache_position: lucid.Tensor) -> lucid.Tensor:
        return self.inner.decode(hidden, cache_position)


_GLOBAL_HOLDER: _Plain = _Plain()


def _via_global_body(hidden: lucid.Tensor) -> lucid.Tensor:
    return _GLOBAL_HOLDER.body(hidden)


def _via_global_decode(hidden: lucid.Tensor, cache_position: lucid.Tensor) -> lucid.Tensor:
    return _GLOBAL_HOLDER.decode(hidden, cache_position)


# How a callable reaches its holder: (holder, "body" | "decode", monkeypatch) -> callable.
type _Reach = Callable[[_Plain, str, pytest.MonkeyPatch], Callable[..., lucid.Tensor]]


def _bound(holder: _Plain, name: str, mp: pytest.MonkeyPatch) -> Callable[..., lucid.Tensor]:
    del mp
    return holder.body if name == "body" else holder.decode


def _closure_over_method(
    holder: _Plain, name: str, mp: pytest.MonkeyPatch
) -> Callable[..., lucid.Tensor]:
    method = _bound(holder, name, mp)
    return lambda *args: method(*args)


def _closure_over_holder(
    holder: _Plain, name: str, mp: pytest.MonkeyPatch
) -> Callable[..., lucid.Tensor]:
    del mp
    if name == "body":
        return lambda hidden: holder.body(hidden)
    return lambda hidden, cp: holder.decode(hidden, cp)


class _Callable:
    """A callable instance: its state is the ``holder`` it forwards to."""

    def __init__(self, holder: _Plain, name: str) -> None:
        self.holder = holder
        self.name = name

    def __call__(self, *args: lucid.Tensor) -> lucid.Tensor:
        method: Callable[..., lucid.Tensor] = getattr(self.holder, self.name)
        return method(*args)


def _callable_instance(
    holder: _Plain, name: str, mp: pytest.MonkeyPatch
) -> Callable[..., lucid.Tensor]:
    del mp
    return _Callable(holder, name)


def _global(holder: _Plain, name: str, mp: pytest.MonkeyPatch) -> Callable[..., lucid.Tensor]:
    mp.setitem(globals(), "_GLOBAL_HOLDER", holder)
    return _via_global_body if name == "body" else _via_global_decode


# (how the callable reaches the holder, whether the modules sit one holder deeper)
_REACHES: dict[str, tuple[_Reach, bool]] = {
    "bound-method": (_bound, False),
    "bound-method-of-nested-holder": (_bound, True),
    "closure-over-holder": (_closure_over_holder, False),
    "closure-over-nested-holder": (_closure_over_holder, True),
    "closure-over-bound-method": (_closure_over_method, False),
    "callable-instance": (_callable_instance, False),
    "global-holder": (_global, False),
    "global-nested-holder": (_global, True),
}


def _hidden(pos: int) -> lucid.Tensor:
    return (lucid.arange(3 * _W, device="metal").reshape(3, _W) * 0.1) * (pos + 1.0)


def _run(
    entry: str, method: str, reach: str, swap: bool, mp: pytest.MonkeyPatch
) -> dict[str, object]:
    """``entry`` ("eager", "compile" or "decode") over ``method`` ("body" or "decode")."""
    factory, nested = _REACHES[reach]
    rig = _Rig()
    holder: _Plain = _Outer(rig) if nested else rig
    fn = factory(holder, method, mp)
    step = fn
    if entry == "compile":
        step = lucid.compile(fn)
    elif entry == "decode":
        step = compiled_decode_step(fn, rig.cache)
    outs: list[np.ndarray] = []
    with lucid.no_grad():
        for pos in range(_CALLS):
            if swap and pos == _CALLS // 2:
                lucid.manual_seed(5)
                rig.lin.weight = nn.Parameter(lucid.randn(_W, _W, device="metal"))
            h = _hidden(pos)
            if method == "decode":
                out = step(h, lucid.tensor([pos], device="metal").long())
            else:
                out = step(h)
            outs.append(out.numpy())
    return {
        "out": np.concatenate([o.reshape(-1) for o in outs]),
        "counter": int(rig.bn.num_batches_tracked.item()),
        "mean": rig.bn.running_mean.numpy(),
        "var": rig.bn.running_var.numpy(),
    }


@pytest.mark.parametrize("swap", [False, True], ids=["steady", "param-swap"])
@pytest.mark.parametrize("reach", list(_REACHES))
@pytest.mark.parametrize("entry", ["compile", "decode"])
def test_the_walk_reaches_modules_held_by_a_plain_object(
    entry: str, reach: str, swap: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = "decode" if entry == "decode" else "body"
    expected = _run("eager", method, reach, swap, monkeypatch)
    got = _run(entry, method, reach, swap, monkeypatch)
    assert expected["counter"] == _CALLS
    assert got["counter"] == _CALLS
    for key in ("out", "mean", "var"):
        np.testing.assert_allclose(got[key], expected[key], rtol=1e-5, atol=1e-5, err_msg=key)


def test_the_bound_object_is_a_hop_of_the_path() -> None:
    rig = _Rig()
    paths = {path: mod for path, mod in _indirect_module_paths(rig.body)}
    self_hop = ("self", "")
    assert paths[(self_hop, ("attr", "bn"))] is rig.bn
    assert paths[(self_hop, ("attr", "lin"))] is rig.lin
    for path, mod in paths.items():
        assert _resolve_path(rig.body, path) is mod
    # Rebinding the attribute is seen the next time the path is followed.
    other = nn.BatchNorm1d(_W)
    rig.bn = other
    assert _resolve_path(rig.body, (self_hop, ("attr", "bn"))) is other


def test_a_callable_instance_is_its_own_self_hop() -> None:
    rig = _Rig()
    held = _Callable(rig, "body")
    paths = {path: mod for path, mod in _indirect_module_paths(held)}
    assert paths[(("self", ""), ("attr", "holder"), ("attr", "bn"))] is rig.bn
    for path, mod in paths.items():
        assert _resolve_path(held, path) is mod


def test_a_builtin_methods_module_is_not_walked() -> None:
    # ``math.sin.__self__`` is the ``math`` module: not the object of a bound method.
    import math

    assert _indirect_module_paths(math.sin) == []


_A_CLASS_GLOBAL = types.FunctionType
_A_MODULE_GLOBAL: nn.Module = nn.Identity()


def _reads_a_class_and_a_module() -> object:
    return _A_CLASS_GLOBAL, _A_MODULE_GLOBAL


def test_a_function_type_is_not_walked_as_a_function() -> None:
    # ``callable(types.FunctionType) and hasattr(.., "__code__")`` holds — the
    # class's ``__code__`` is a member descriptor — and ``callable_slots`` then
    # iterated it (TypeError).  Only a real code object makes a function.
    assert list(_path_children(types.FunctionType, set())) == []
    assert _indirect_module_paths(_reads_a_class_and_a_module) == []
