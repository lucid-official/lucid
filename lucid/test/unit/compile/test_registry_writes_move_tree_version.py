"""Every write to a module's tensor registries moves the tree version (LCD-560).

``lucid.compile`` keys an executable on the tree version and, when it moves,
compares the tree with the one it traced (``ParamFingerprint``).  The
parameter and buffer registries used to be plain dicts: a tensor put there
by ``register_parameter`` / ``register_buffer``, an ``assign=True`` load, a
reparametrisation or a parameter container left the version where it was,
and a compiled call went on reading the old tensor.  They are
``_TrackedDict``\\ s now, like the submodule and hook registries.

The table makes each kind of write and asserts the version moved; the
copies below keep the registries tracked.
"""

import copy
import pickle
from collections import OrderedDict
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid.nn._tree_version import _TrackedDict, _tree_version
from lucid.nn.utils import (
    parametrize,
    prune,
    remove_spectral_norm,
    remove_weight_norm,
    spectral_norm,
    weight_norm,
)


class _Holder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.plist = nn.ParameterList([nn.Parameter(lucid.ones(2))])
        self.pdict = nn.ParameterDict({"a": nn.Parameter(lucid.ones(2))})
        self.register_buffer("buf", lucid.zeros(2))


class _Double(nn.Module):
    def forward(self, w: lucid.Tensor) -> lucid.Tensor:
        return w * 2.0


def _param() -> nn.Parameter:
    return nn.Parameter(lucid.full((2,), 3.0))


def _assign_load(m: _Holder) -> None:
    m.lin.load_state_dict(
        {k: v.detach().clone() for k, v in m.lin.state_dict().items()}, assign=True
    )


def _none(m: _Holder) -> None:
    del m


# (setup, the write); setup runs before the version is read.
_WRITES: dict[str, tuple[Callable[[_Holder], object], Callable[[_Holder], object]]] = {
    "setattr a parameter": (_none, lambda m: setattr(m, "w", _param())),
    "setattr a parameter over one": (
        _none,
        lambda m: setattr(m.lin, "bias", nn.Parameter(lucid.zeros(4))),
    ),
    "setattr a buffer": (_none, lambda m: setattr(m, "buf", lucid.ones(2))),
    "register_parameter": (_none, lambda m: m.register_parameter("w", _param())),
    "register_parameter over one": (
        _none,
        lambda m: m.lin.register_parameter("bias", nn.Parameter(lucid.zeros(4))),
    ),
    "register_parameter None": (_none, lambda m: m.register_parameter("w", None)),
    "register_buffer": (_none, lambda m: m.register_buffer("b2", lucid.ones(2))),
    "register_buffer over one": (
        _none,
        lambda m: m.register_buffer("buf", lucid.ones(2), persistent=False),
    ),
    "delattr a parameter": (_none, lambda m: delattr(m.lin, "bias")),
    "delattr a buffer": (_none, lambda m: delattr(m, "buf")),
    "parameter .data =": (_none, lambda m: setattr(m.lin.weight, "data", lucid.ones(4, 4))),
    "buffer .data =": (_none, lambda m: setattr(m.buf, "data", lucid.ones(2))),
    "assign load": (_none, _assign_load),
    ".to() a buffer": (_none, lambda m: m.half()),
    "_parameters write": (
        _none,
        lambda m: m.lin._parameters.__setitem__("bias", nn.Parameter(lucid.zeros(4))),
    ),
    "_buffers pop": (_none, lambda m: m._buffers.pop("buf")),
    "ParameterList append": (_none, lambda m: m.plist.append(_param())),
    "ParameterList item": (_none, lambda m: m.plist.__setitem__(0, _param())),
    "ParameterList delete": (_none, lambda m: m.plist.__delitem__(0)),
    "ParameterDict item": (_none, lambda m: m.pdict.__setitem__("a", _param())),
    "ParameterDict new key": (_none, lambda m: m.pdict.__setitem__("b", _param())),
    "ParameterDict pop": (_none, lambda m: m.pdict.pop("a")),
    "ParameterDict update": (_none, lambda m: m.pdict.update({"c": _param()})),
    "ParameterDict clear": (_none, lambda m: m.pdict.clear()),
    "register_parametrization": (
        _none,
        lambda m: parametrize.register_parametrization(m.lin, "weight", _Double()),
    ),
    "remove_parametrizations": (
        lambda m: parametrize.register_parametrization(m.lin, "weight", _Double()),
        lambda m: parametrize.remove_parametrizations(m.lin, "weight"),
    ),
    "prune": (_none, lambda m: prune.l1_unstructured(m.lin, "weight", amount=0.5)),
    "prune remove": (
        lambda m: prune.l1_unstructured(m.lin, "weight", amount=0.5),
        lambda m: prune.remove(m.lin, "weight"),
    ),
    "weight_norm": (_none, lambda m: weight_norm(m.lin)),
    "remove_weight_norm": (lambda m: weight_norm(m.lin), lambda m: remove_weight_norm(m.lin)),
    "spectral_norm": (_none, lambda m: spectral_norm(m.lin)),
    "remove_spectral_norm": (lambda m: spectral_norm(m.lin), lambda m: remove_spectral_norm(m.lin)),
}


@pytest.mark.parametrize("write", list(_WRITES))
def test_a_registry_write_moves_the_tree_version(write: str) -> None:
    setup, apply = _WRITES[write]
    model = _Holder()
    setup(model)
    version = _tree_version()
    apply(model)
    assert _tree_version() != version, f"{write}: the tree version did not move"


def test_copies_and_unpickled_modules_keep_tracked_tensor_registries() -> None:
    model = _Holder()
    data = pickle.dumps(model)
    assert b"_TrackedDict" not in data, "a checkpoint names the private type"
    keys = list(model.state_dict())
    for clone in (copy.deepcopy(model), pickle.loads(data)):
        for sub in clone.modules():
            assert isinstance(sub._parameters, _TrackedDict)
            assert isinstance(sub._buffers, _TrackedDict)
        assert list(clone.state_dict()) == keys
        version = _tree_version()
        clone.register_buffer("buf", lucid.ones(2))
        assert _tree_version() != version


def test_a_state_dict_round_trip_keeps_the_order_and_the_values() -> None:
    model, other = _Holder(), _Holder()
    other.load_state_dict(model.state_dict())
    assert list(other.state_dict()) == list(model.state_dict())
    for (k, a), b in zip(model.state_dict().items(), other.state_dict().values(), strict=True):
        assert a.tolist() == b.tolist(), k


def test_a_module_pickled_with_plain_tensor_registries_comes_back_tracked() -> None:
    old = nn.Linear(2, 2)
    for name in ("_parameters", "_buffers"):
        object.__setattr__(old, name, OrderedDict(getattr(old, name)))
    restored = pickle.loads(pickle.dumps(old))
    assert isinstance(restored._parameters, _TrackedDict)
    assert isinstance(restored._buffers, _TrackedDict)
    version = _tree_version()
    restored.register_parameter("bias", nn.Parameter(lucid.zeros(2)))
    assert _tree_version() != version
