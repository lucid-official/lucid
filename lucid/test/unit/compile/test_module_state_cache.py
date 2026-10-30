"""The compiled cache key's walk of a module tree is cached (LCD-462).

Every compiled call keys its executable on each submodule's ``training``
flag, state attributes (``_StateContract``: public or private scalars and
short containers of them) and hooks.  Walking the tree for that cost about a
millisecond a call on a 200-module model, so the walk is kept per model
against the tree version (``lucid/nn/_tree_version.py``), which moves after an
attribute set or deleted through ``Module.__setattr__`` / ``__delattr__`` and
after any write to a submodule or hook registry.  A list, set or dict changed
in place moves nothing, so the walk holds those and the key reads them on
every call.

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
from enum import Enum

import pytest

import lucid
import lucid.nn as nn
import lucid.nn._tree_version as tree_version
from lucid.compile._core import signature as sig
from lucid.compile._core.module_state import StateContract
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
        self._hidden = False
        self.names: list[str] = []
        self.table: dict[str, int] = {}
        self.kinds: set[str] = set()
        self.groups: list[list[str]] = [[]]


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
    "private scalar set": (_none, lambda m: setattr(m, "_hidden", True)),
    "list changed in place": (_none, lambda m: m.names.append("shift")),
    "dict changed in place": (_none, lambda m: m.table.update(rank=4)),
    "set changed in place": (_none, lambda m: m.kinds.add("lora")),
    "nested list changed in place": (_none, lambda m: m.groups[0].append("x")),
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


# ── what the walk keys a module's attributes by (LCD-704) ────────────────


class _Mode(Enum):
    ON = 1
    OFF = 2


# value → whether it is state (``_StateContract.value_sig`` keys it).
_CONTRACT: dict[str, tuple[object, bool]] = {
    "None": (None, True),
    "bool": (False, True),
    "complex": (1j, True),
    "bytes": (b"x", True),
    "enum member": (_Mode.ON, True),
    "dtype": (lucid.float16, True),
    "tuple of tuples": (((1, 2), (3, 4)), True),
    "list of names": (["default", "other"], True),
    "dict of scalars": ({"rank": 4, "alpha": 8.0}, True),
    "set of names": ({"a", "b"}, True),
    "64 items": (list(range(64)), True),
    "65 items": (list(range(65)), False),
    "three containers deep": ([[[1]]], True),
    "four containers deep": ([[[[1]]]], False),
    "a list holding a tensor": ([lucid.zeros(1)], False),
    "a dict with an int key": ({1: 2}, False),
    "a class": (nn.Linear, False),
    "a function": (len, False),
}


@pytest.mark.parametrize("name", list(_CONTRACT))
def test_the_state_a_module_is_keyed_by(name: str) -> None:
    value, keyed = _CONTRACT[name]
    assert (StateContract.value_sig(value) is not None) is keyed


def test_state_signatures_tell_apart_what_a_branch_can() -> None:
    value_sig = StateContract.value_sig
    assert value_sig({1, 2}) == value_sig({2, 1})
    assert value_sig([1, 2]) != value_sig((1, 2))
    assert value_sig([1, 2]) != value_sig([2, 1])
    assert value_sig({"a": 1, "b": 2}) == value_sig({"b": 2, "a": 1})
    assert value_sig(_Mode.ON) != value_sig(_Mode.OFF)
    assert value_sig(lucid.float16) != value_sig(lucid.bfloat16)


def test_the_bookkeeping_of_every_module_is_not_its_state() -> None:
    names = {name for name, _ in StateContract.attrs(nn.Linear(2, 2))}
    assert names == {"in_features", "out_features"}
