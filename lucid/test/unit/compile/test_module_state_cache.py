"""The compiled cache key's walk of a module tree is cached (LCD-462).

Every compiled call keys its executable on each submodule's ``training``
flag, public scalar attributes and hooks.  Walking the tree for that cost
about a millisecond a call on a 200-module model, so the walk is kept per
model against the tree version (``lucid/nn/_tree_version.py``), which moves
after an attribute set or deleted through ``Module.__setattr__`` /
``__delattr__`` and after any write to a submodule or hook registry.

The matrix below makes each kind of change after the walk was cached and
holds the cached answer to a fresh walk: a kind that does not move the
version would leave a compiled model replaying the executable of the tree
it used to be — hooks that never fire, a ``p`` that never changes.
"""

import copy
import pickle
import time
from collections import OrderedDict
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn._tree_version as tree_version
from lucid.compile._core import signature as sig
from lucid.nn._tree_version import _TrackedDict
from lucid.nn.utils import parametrize
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = pytest.mark.usefixtures("inference")


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
        self.mlist = nn.ModuleList([nn.Dropout(0.1), nn.Linear(4, 4)])
        self.mdict = nn.ModuleDict({"a": nn.Dropout(0.2), "b": nn.Linear(4, 4)})
        self.head = nn.Linear(4, 2)
        self.flag = 1


class _Double(nn.Module):
    def forward(self, w: lucid.Tensor) -> lucid.Tensor:
        return w * 2.0


def _hook(*args: object) -> None:
    del args


def _answer(model: nn.Module) -> tuple[tuple[object, ...], bool]:
    return sig._module_state(model), sig.model_has_active_hooks(model)


def _fresh(model: nn.Module) -> tuple[tuple[object, ...], bool]:
    sig._WALKS.pop(model, None)
    return _answer(model)


def _none(m: _Net) -> object:
    del m
    return None


def _clear_hooks(m: _Net) -> None:
    m.seq[0]._forward_hooks.clear()


_Change = tuple[Callable[[_Net], object], Callable[[_Net], object]]

# (setup before the walk is cached, change after it)
CHANGES: dict[str, _Change] = {
    "deep scalar set": (_none, lambda m: setattr(m.seq[1], "p", 0.25)),
    "scalar delete": (_none, lambda m: delattr(m, "flag")),
    "train flip": (_none, lambda m: m.mlist.eval()),
    "submodule set": (_none, lambda m: setattr(m, "extra", nn.Dropout(0.3))),
    "submodule delete": (_none, lambda m: delattr(m, "head")),
    "add_module": (_none, lambda m: m.seq.add_module("tail", nn.Dropout(0.7))),
    "Sequential.append": (_none, lambda m: m.seq.append(nn.Dropout(0.2))),
    "Sequential del": (_none, lambda m: m.seq.__delitem__(1)),
    "ModuleList.append": (_none, lambda m: m.mlist.append(nn.Dropout(0.6))),
    "ModuleList.insert": (_none, lambda m: m.mlist.insert(0, nn.Dropout(0.8))),
    "ModuleDict.update": (_none, lambda m: m.mdict.update({"z": nn.Dropout(0.4)})),
    "ModuleDict.pop": (_none, lambda m: m.mdict.pop("a")),
    "ModuleDict.clear": (_none, lambda m: m.mdict.clear()),
    "_modules write": (_none, lambda m: m.seq._modules.__setitem__("x", nn.Dropout(0.9))),
    "_modules popitem": (_none, lambda m: m.mdict._modules.popitem()),
    "_modules move_to_end": (_none, lambda m: m._modules.move_to_end("seq")),
    "forward hook": (_none, lambda m: m.seq[0].register_forward_hook(_hook)),
    "forward pre hook": (_none, lambda m: m.head.register_forward_pre_hook(_hook)),
    "backward hook": (_none, lambda m: m.mlist[1].register_full_backward_hook(_hook)),
    "backward pre hook": (_none, lambda m: m.mdict["b"].register_full_backward_pre_hook(_hook)),
    "hook handle remove": (
        lambda m: m.seq[0].register_forward_hook(_hook),
        lambda m: m._handle.remove(),  # type: ignore[union-attr]
    ),
    "hook registry clear": (lambda m: m.seq[0].register_forward_hook(_hook), _clear_hooks),
    "parametrization": (
        _none,
        lambda m: parametrize.register_parametrization(m.head, "weight", _Double()),
    ),
}


@pytest.mark.parametrize("name", list(CHANGES))
def test_the_cached_walk_follows_every_change(name: str) -> None:
    setup, change = CHANGES[name]
    model = _Net()
    handle = setup(model)
    object.__setattr__(model, "_handle", handle)
    before = _answer(model)
    assert _answer(model) == before  # served from the cache
    change(model)
    after = _answer(model)
    assert after == _fresh(model), f"{name}: the cached walk is stale"
    assert after != before, f"{name}: the change is not visible to the key"


def test_a_closure_that_names_another_module_is_another_tree() -> None:
    first, second = nn.Dropout(0.1), nn.Dropout(0.9)
    current = first

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return current(x)

    wrapper = lucid.compile(fn)._model  # type: ignore[attr-defined]
    before = _answer(wrapper)
    current = second
    after = _answer(wrapper)
    assert after != before and after == _fresh(wrapper)


def test_copies_and_unpickled_modules_keep_tracked_registries() -> None:
    model = _Net()
    data = pickle.dumps(model)
    assert b"_TrackedDict" not in data, "a checkpoint names the private type"
    for clone in (copy.deepcopy(model), pickle.loads(data)):
        for name in ("_modules", "_forward_hooks", "_backward_hooks"):
            assert isinstance(getattr(clone.seq, name), _TrackedDict)
        before = _answer(clone)
        clone.seq[0].register_forward_hook(_hook)
        assert _answer(clone) != before


def test_a_module_pickled_with_plain_registries_comes_back_tracked() -> None:
    """A pickle written before the registries were tracked holds plain
    ``OrderedDict``s; loading converts them."""
    old = nn.Dropout(0.5)
    for name in ("_modules", "_forward_pre_hooks", "_forward_hooks"):
        object.__setattr__(old, name, OrderedDict(getattr(old, name)))
    restored = pickle.loads(pickle.dumps(old))
    assert isinstance(restored._forward_hooks, _TrackedDict)
    before = _answer(restored)
    restored.register_forward_hook(_hook)
    assert _answer(restored) != before and _answer(restored)[1]


def _deep_model(n: int = 200) -> nn.Module:
    return nn.Sequential(*[nn.Linear(4, 4) for _ in range(n)])


def test_replays_do_not_walk_the_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    walks: list[int] = []
    walk = sig._iter_modules

    def counted(roots: list[nn.Module]) -> list[nn.Module]:
        walks.append(len(roots))
        return walk(roots)

    monkeypatch.setattr(sig, "_iter_modules", counted)
    model = _deep_model(50).to(COMPILE_DEVICE)
    compiled = lucid.compile(model)
    x = metal_tensor(2, 4)
    compiled(x)
    walks.clear()
    for _ in range(5):
        compiled(x)
    assert walks == [], "a compiled replay walked the module tree"

    model[7].flag = 3  # type: ignore[attr-defined]
    compiled(x)
    assert walks, "a change to the tree did not reach the key"
    assert compiled.cache_info()["entries"] == 2
    walks.clear()
    for _ in range(5):
        compiled(x)
    assert walks == []


def test_keying_a_call_no_longer_costs_a_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Time, loosely: the cached key against one that walks every call."""
    model = _deep_model()
    x = lucid.randn(2, 4)
    fp = sig._param_fingerprint(model)

    def keying() -> float:
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(20):
                sig.signature_of(model, (x,), {}, dynamic=False, param_fingerprint=fp)
                sig.model_has_active_hooks(model)
            best = min(best, time.perf_counter() - t0)
        return best

    cached = keying()
    versions = iter(range(10**9, 2 * 10**9))
    monkeypatch.setattr(tree_version, "_tree_version", lambda: next(versions))
    walking = keying()
    assert cached * 10 < walking, f"cached {cached:.6f}s vs walking {walking:.6f}s"
