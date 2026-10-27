"""Writing a registered name again keeps its slot in the module's registries.

``Module.__setattr__`` deleted a registered name and added it back, which
moved it to the end of its registry.  ``layer.weight += 1`` under
``no_grad`` is such a write — Python stores what ``__iadd__`` returns back
on the attribute — so a Linear's parameters came out ``['bias', 'weight']``
and whatever paired two modules' parameters by position paired the wrong
tensors: the JEPA families' target-encoder average, an optimizer state
loaded into an optimizer built afterwards, a checkpoint's key order
(LCD-551).  The reference writes the new value into the old slot, and for
a registered name refuses a value of another kind with a ``TypeError``
where Lucid demoted the name to a plain attribute without a word.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.nn._tree_version import _tree_version

_NAMES = ["a", "b", "c"]


@dataclass(frozen=True)
class _Kind:
    registry: str
    make: Callable[[str], object]
    register: Callable[[nn.Module, str, object], None]
    addable: bool


_KINDS = {
    "parameter": _Kind(
        "_parameters",
        lambda d: nn.Parameter(lucid.zeros(2, device=d)),
        lambda m, n, v: m.register_parameter(n, v),
        True,
    ),
    "buffer": _Kind(
        "_buffers",
        lambda d: lucid.zeros(2, device=d),
        lambda m, n, v: m.register_buffer(n, v),
        True,
    ),
    "non-persistent buffer": _Kind(
        "_buffers",
        lambda d: lucid.zeros(2, device=d),
        lambda m, n, v: m.register_buffer(n, v, persistent=False),
        True,
    ),
    "submodule": _Kind(
        "_modules", lambda d: nn.Linear(2, 2, device=d), lambda m, n, v: m.add_module(n, v), False
    ),
}


def _filled(kind: _Kind, device: str) -> nn.Module:
    module = nn.Module()
    for name in _NAMES:
        kind.register(module, name, kind.make(device))
    return module


def _assign_new(module: nn.Module, kind: _Kind, device: str) -> None:
    module.a = kind.make(device)


def _assign_same(module: nn.Module, kind: _Kind, device: str) -> None:
    module.a = module.a


def _add_in_place(module: nn.Module, kind: _Kind, device: str) -> None:
    with lucid.no_grad():
        module.a += 1.0


def _register_again(module: nn.Module, kind: _Kind, device: str) -> None:
    kind.register(module, "a", kind.make(device))


def _empty_then_fill(module: nn.Module, kind: _Kind, device: str) -> None:
    module.a = None
    module.a = kind.make(device)


_WAYS: dict[str, Callable[[nn.Module, _Kind, str], None]] = {
    "assigned a new value": _assign_new,
    "assigned its own value": _assign_same,
    "added to in place": _add_in_place,
    "registered again": _register_again,
    "emptied then filled": _empty_then_fill,
}

_CASES = [
    pytest.param(kind, way, id=f"{kind}-{way}")
    for kind in _KINDS
    for way in _WAYS
    if _KINDS[kind].addable or way != "added to in place"
]


@pytest.mark.parametrize("kind, way", _CASES)
def test_writing_a_registered_name_again_keeps_its_place(kind, way, device):
    spec = _KINDS[kind]
    module = _filled(spec, device)
    keys = list(module.state_dict())
    saved = [name for name, _ in module.named_buffers()]

    _WAYS[way](module, spec, device)

    assert list(getattr(module, spec.registry)) == _NAMES
    assert list(module.state_dict()) == keys
    assert [name for name, _ in module.named_buffers()] == saved
    assert "a" not in module.__dict__


def test_an_in_place_update_leaves_a_layers_parameters_in_order(device):
    layer = nn.Linear(3, 2, device=device)
    before = list(layer.parameters())
    with lucid.no_grad():
        layer.weight += 1.0
    assert [name for name, _ in layer.named_parameters()] == ["weight", "bias"]
    assert all(now is then for now, then in zip(layer.parameters(), before, strict=True))
    assert list(layer.state_dict()) == ["weight", "bias"]


def test_an_optimizer_state_loads_onto_the_same_parameters_after_an_update(device):
    # The state is keyed by position in ``parameters()``: an optimizer built
    # after the update reads it back in the order the update left.
    layer = nn.Linear(3, 2, device=device)
    first = optim.Adam(layer.parameters())
    layer(lucid.ones(4, 3, device=device)).sum().backward()
    first.step()
    with lucid.no_grad():
        layer.weight += 0.0
    second = optim.Adam(layer.parameters())
    second.load_state_dict(first.state_dict())
    state = second.state_dict()["state"]
    for index, parameter in enumerate(second.param_groups[0]["params"]):
        assert state[index]["exp_avg"].shape == parameter.shape


def test_a_sequential_child_assigned_by_name_keeps_its_turn_in_the_forward():
    # ``_modules`` order is a Sequential's execution order: with the conv
    # moved to the end, the BatchNorm ran on the raw input.  Quantization's
    # ``convert`` hit this and has written through ``_modules`` since.
    block = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU())
    setattr(block, "0", nn.Conv2d(3, 8, 3))
    assert list(block._modules) == ["0", "1", "2"]
    assert block(lucid.randn(2, 3, 8, 8)).shape == (2, 8, 6, 6)


def test_a_name_deleted_and_assigned_again_is_registered_at_the_end():
    module = _filled(_KINDS["parameter"], "cpu")
    del module.a
    module.a = nn.Parameter(lucid.zeros(2))
    assert list(module._parameters) == ["b", "c", "a"]


# ── a value of another kind for a registered name ───────────────────────────

_REFUSALS = [
    pytest.param("parameter", lambda d: lucid.zeros(2, device=d), id="parameter-tensor"),
    pytest.param("parameter", lambda d: 3, id="parameter-int"),
    pytest.param("parameter", lambda d: nn.Linear(2, 2, device=d), id="parameter-module"),
    pytest.param("buffer", lambda d: 3, id="buffer-int"),
    pytest.param("submodule", lambda d: lucid.zeros(2, device=d), id="submodule-tensor"),
    pytest.param("submodule", lambda d: 3, id="submodule-int"),
]


@pytest.mark.parametrize("kind, value", _REFUSALS)
def test_a_value_of_another_kind_is_refused_and_changes_nothing(kind, value, device):
    spec = _KINDS[kind]
    module = _filled(spec, device)
    held = module.a
    with pytest.raises(TypeError, match="cannot assign .* as .*'a'"):
        module.a = value(device)
    assert module.a is held
    assert list(getattr(module, spec.registry)) == _NAMES
    assert "a" not in module.__dict__


def test_none_empties_a_slot_without_giving_it_up():
    layer = nn.Linear(3, 2)
    layer.bias = None
    assert layer.bias is None
    assert list(layer._parameters) == ["weight", "bias"]
    assert list(layer.state_dict()) == ["weight"]
    layer.bias = nn.Parameter(lucid.zeros(2))
    assert [name for name, _ in layer.named_parameters()] == ["weight", "bias"]


# ── compiled children ───────────────────────────────────────────────────────


def test_a_compiled_child_keeps_its_place_and_can_be_replaced():
    parent = nn.Module()
    parent.first = nn.Linear(2, 2)
    parent.second = nn.Linear(2, 2)
    names = [name for name, _ in parent.named_parameters()]

    parent.first = lucid.compile(parent.first)
    assert [name for name, _ in parent.named_parameters()] == names

    replacement = nn.Linear(2, 2)
    parent.add_module("first", replacement)
    assert parent.first is replacement
    assert list(parent._modules) == ["first", "second"]


# ── deletion ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", list(_KINDS))
def test_deleting_a_registered_name_moves_the_tree_version(kind):
    module = _filled(_KINDS[kind], "cpu")
    version = _tree_version()
    del module.a
    assert _tree_version() != version
    assert "a" not in getattr(module, _KINDS[kind].registry)


# ── each transition, against the reference ─────────────────────────────────


def _value(lib: ModuleType, what: str) -> object:
    return {
        "parameter": lambda: lib.nn.Parameter(lib.zeros(1)),
        "tensor": lambda: lib.zeros(1),
        "module": lambda: lib.nn.Linear(1, 1),
        "none": lambda: None,
        "int": lambda: 3,
    }[what]()


def _start(lib: ModuleType, start: str) -> object:
    module = lib.nn.Module()
    for name in _NAMES:
        if start == "parameters":
            module.register_parameter(name, _value(lib, "parameter"))
        elif start in ("buffers", "non-persistent buffers"):
            persistent = start == "buffers"
            module.register_buffer(name, _value(lib, "tensor"), persistent=persistent)
        elif start == "submodules":
            module.add_module(name, _value(lib, "module"))
    if start == "plain":
        module.a = _value(lib, "tensor")
        module.b = _value(lib, "parameter")
        module.c = _value(lib, "module")
    return module


def _step(lib: ModuleType, module: object, op: str, what: str) -> None:
    if op == "set":
        setattr(module, "a", _value(lib, what))
    elif op == "del":
        delattr(module, "a")
    elif op == "register_buffer_np":
        module.register_buffer("a", _value(lib, what), persistent=False)
    else:
        getattr(module, op)("a", _value(lib, what))


def _layout(module: object) -> object:
    return (
        list(module._parameters),
        list(module._buffers),
        list(module._modules),
        [name for name in _NAMES if name in module.__dict__],
        list(module.state_dict()),
    )


def _outcome(lib: ModuleType, start: str, steps: list[tuple[str, str]]) -> object:
    module = _start(lib, start)
    try:
        for op, what in steps:
            _step(lib, module, op, what)
    except (TypeError, KeyError) as error:
        return type(error).__name__, _layout(module)
    return _layout(module)


_TRANSITIONS = {
    "parameter <- parameter": ("parameters", [("set", "parameter")]),
    "parameter <- None": ("parameters", [("set", "none")]),
    "parameter <- None <- parameter": ("parameters", [("set", "none"), ("set", "parameter")]),
    "parameter <- tensor": ("parameters", [("set", "tensor")]),
    "parameter <- int": ("parameters", [("set", "int")]),
    "parameter <- module": ("parameters", [("set", "module")]),
    "parameter deleted <- parameter": ("parameters", [("del", ""), ("set", "parameter")]),
    "buffer <- tensor": ("buffers", [("set", "tensor")]),
    "buffer <- None": ("buffers", [("set", "none")]),
    "buffer <- int": ("buffers", [("set", "int")]),
    "buffer <- parameter": ("buffers", [("set", "parameter")]),
    "buffer <- module": ("buffers", [("set", "module")]),
    "non-persistent buffer <- tensor": ("non-persistent buffers", [("set", "tensor")]),
    "non-persistent buffer <- None <- tensor": (
        "non-persistent buffers",
        [("set", "none"), ("set", "tensor")],
    ),
    "non-persistent buffer <- module": ("non-persistent buffers", [("set", "module")]),
    "non-persistent buffer deleted <- tensor": (
        "non-persistent buffers",
        [("del", ""), ("set", "tensor")],
    ),
    "submodule <- module": ("submodules", [("set", "module")]),
    "submodule <- None": ("submodules", [("set", "none")]),
    "submodule <- None <- module": ("submodules", [("set", "none"), ("set", "module")]),
    "submodule <- tensor": ("submodules", [("set", "tensor")]),
    "submodule <- int": ("submodules", [("set", "int")]),
    "submodule <- parameter": ("submodules", [("set", "parameter")]),
    "plain tensor <- tensor": ("plain", [("set", "tensor")]),
    "plain tensor <- parameter": ("plain", [("set", "parameter")]),
    "plain tensor <- module": ("plain", [("set", "module")]),
    "register_parameter over a parameter": ("parameters", [("register_parameter", "parameter")]),
    "register_parameter None over a parameter": ("parameters", [("register_parameter", "none")]),
    "register_buffer over a buffer": ("buffers", [("register_buffer", "tensor")]),
    "register_buffer non-persistent over a buffer": ("buffers", [("register_buffer_np", "tensor")]),
    "add_module over a submodule": ("submodules", [("add_module", "module")]),
    "add_module None over a submodule": ("submodules", [("add_module", "none")]),
    "register_parameter over a buffer": ("buffers", [("register_parameter", "parameter")]),
    "register_buffer over a parameter": ("parameters", [("register_buffer", "tensor")]),
    "add_module over a parameter": ("parameters", [("add_module", "module")]),
    "register_parameter over a plain attribute": ("plain", [("register_parameter", "parameter")]),
}


@pytest.mark.parity
@pytest.mark.parametrize("transition", list(_TRANSITIONS))
def test_each_transition_matches_the_reference(ref, transition):
    start, steps = _TRANSITIONS[transition]
    assert _outcome(lucid, start, steps) == _outcome(ref, start, steps)


# ── containers: indexed assignment goes through the same rules ──────────────


def _container(lib: ModuleType, kind: str) -> object:
    if kind == "Sequential":
        return lib.nn.Sequential(lib.nn.Linear(1, 1), lib.nn.ReLU(), lib.nn.Linear(1, 1))
    if kind == "ModuleList":
        return lib.nn.ModuleList([lib.nn.Linear(1, 1), lib.nn.ReLU(), lib.nn.Linear(1, 1)])
    return lib.nn.ParameterList([lib.nn.Parameter(lib.zeros(1)) for _ in range(3)])


def _item(lib: ModuleType, kind: str, what: str) -> object:
    if what == "own kind":
        what = "parameter" if kind == "ParameterList" else "module"
    return _value(lib, what)


def _indexed(lib: ModuleType, kind: str, index: object, what: str) -> object:
    container = _container(lib, kind)
    try:
        container[index] = _item(lib, kind, what)
    except (TypeError, IndexError) as error:
        return type(error).__name__
    return [
        (name, type(value).__name__)
        for registry in (container._modules, container._parameters)
        for name, value in registry.items()
    ]


_INDEXED = [
    (kind, index, what)
    for kind in ("Sequential", "ModuleList", "ParameterList")
    for index, what in [
        (0, "own kind"),
        (-1, "own kind"),
        (3, "own kind"),
        (-4, "own kind"),
        ("0", "own kind"),
        (1.0, "own kind"),
        (0, "int"),
        (0, "tensor"),
        (0, "module" if kind == "ParameterList" else "parameter"),
    ]
]


@pytest.mark.parity
@pytest.mark.parametrize(
    "kind, index, what", _INDEXED, ids=[f"{k}[{i!r}]={w}" for k, i, w in _INDEXED]
)
def test_indexed_assignment_matches_the_reference(ref, kind, index, what):
    assert _indexed(lucid, kind, index, what) == _indexed(ref, kind, index, what)


@pytest.mark.parametrize("kind", ["Sequential", "ModuleList"])
def test_indexed_assignment_refuses_what_is_not_a_module(kind, device):
    container = _container(lucid, kind).to(device)
    held = container[0]
    with pytest.raises(TypeError, match="cannot assign 'int' as child module '0'"):
        container[0] = 5
    assert container[0] is held
    assert list(container._modules) == ["0", "1", "2"]


def test_a_plain_tensor_put_in_a_parameter_list_is_trained(device):
    params = nn.ParameterList([nn.Parameter(lucid.zeros(2, device=device))])
    params[0] = lucid.ones(2, device=device)
    assert type(params[0]) is nn.Parameter and params[0].requires_grad
    assert list(params.parameters())[0] is params[0]


def test_replacing_a_compiled_child_by_index_replaces_what_answers():
    block = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    block[0] = lucid.compile(block[0])
    plain = nn.Linear(2, 2)
    block[0] = plain
    assert getattr(block, "0") is plain
    assert block[0] is plain
    assert list(block._modules) == ["0", "1"]
