"""Which in-place writes a leaf that requires grad refuses, through its view family.

A leaf's values are where its gradient accumulates, so while autograd records
a write to the leaf, or through a view made from it, is refused.  The rule
follows the chain of views the written tensor was made from — not the whole
family that shares the buffer.  Writing the leaf's own base, or a view made
beside the leaf, is allowed, as the reference allows it: the engine refused
any write into a family with such a leaf in it (LCD-76).
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

_DEVICES = [
    "cpu",
    pytest.param("metal", marks=pytest.mark.skipif(not metal_available(), reason="needs Metal")),
]

_WRITES: dict[str, Callable[[lucid.Tensor], object]] = {
    "add_": lambda t: t.add_(1.0),
    "mul_": lambda t: t.mul_(2.0),
    "fill_": lambda t: t.fill_(2.0),
    "zero_": lambda t: t.zero_(),
    "copy_": lambda t: t.copy_(t.detach() * 0.0 + 3.0),
}


def _leaf_view_of_plain_base(lib: ModuleType, device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
    """``X`` needs no grad; ``x = X[:]`` is a leaf that does."""
    X = lib.ones(4, 3, device=device)
    x = X[:]
    x.requires_grad_(True)
    return X, x


def _make(lib: ModuleType, device: str, case: str) -> tuple[lucid.Tensor, lucid.Tensor | None]:
    """The tensor ``case`` writes into, and the leaf it is about (kept alive)."""
    if case == "base of a leaf view":
        return _leaf_view_of_plain_base(lib, device)
    if case == "view beside a leaf view":
        X = lib.ones(4, 3, device=device)
        x = X[:2]
        x.requires_grad_(True)
        return X[2:], x
    if case == "view of a leaf view":
        _, x = _leaf_view_of_plain_base(lib, device)
        return x[:], x
    if case == "view of a dropped leaf":
        return lib.ones(4, 3, device=device, requires_grad=True)[0], None
    W = lib.ones(4, 3, device=device, requires_grad=True)
    if case == "view of a leaf":
        return W[0], W
    if case == "view through a dropped view":
        return W[0:2][0], W
    if case == "view of a leaf's data":
        return W.data[0], W
    raise ValueError(case)


#: case -> whether the write goes through while autograd records
_ALLOWED = {
    "base of a leaf view": True,
    "view beside a leaf view": True,
    "view of a leaf view": False,
    "view of a leaf": False,
    "view of a dropped leaf": False,
    "view through a dropped view": False,
    "view of a leaf's data": True,
}


def _outcome(lib: ModuleType, device: str, case: str, write: str) -> bool:
    target, leaf = _make(lib, device, case)
    assert leaf is None or (leaf.is_leaf and leaf.requires_grad)
    try:
        _WRITES[write](target)
    except RuntimeError as err:
        assert "leaf" in str(err), err
        return False
    return True


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("write", list(_WRITES))
@pytest.mark.parametrize("case", list(_ALLOWED))
def test_a_write_is_refused_only_through_a_view_of_the_leaf(
    case: str, write: str, device: str
) -> None:
    assert _outcome(lucid, device, case, write) is _ALLOWED[case]


@pytest.mark.parity
@pytest.mark.parametrize("write", list(_WRITES))
@pytest.mark.parametrize("case", list(_ALLOWED))
def test_the_rule_is_the_reference_rule(case: str, write: str, ref: ModuleType) -> None:
    assert _outcome(lucid, "cpu", case, write) is _outcome(ref, "cpu", case, write)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("write", list(_WRITES))
def test_the_leaf_view_reads_the_base_write_and_keeps_its_gradient(write: str, device: str) -> None:
    X, x = _leaf_view_of_plain_base(lucid, device)
    _WRITES[write](X)
    assert X.to("cpu").tolist() == x.to("cpu").tolist()
    assert x.is_leaf and x.requires_grad
    (x * x).sum().backward()
    assert x.grad is not None
    assert x.grad.to("cpu").tolist() == (2.0 * X).to("cpu").tolist()


@pytest.mark.parametrize("device", _DEVICES)
def test_a_base_write_still_moves_what_saved_the_leaf_view(device: str) -> None:
    # Allowed now, the write still counts as one: a node that saved ``x``
    # before it refuses to run on the new values, here or at backward.
    X, x = _leaf_view_of_plain_base(lucid, device)
    y = (x * x).sum()
    with pytest.raises(RuntimeError):
        X.add_(1.0)
        y.backward()
