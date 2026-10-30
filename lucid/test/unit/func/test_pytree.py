"""The pytree ``lucid.func`` flattens its arguments and results with."""

import dataclasses
from collections import OrderedDict, namedtuple
from collections.abc import Hashable, Sequence

import pytest

from lucid.func._pytree import (
    Node,
    StructureError,
    TreeSpec,
    tree_flatten,
    tree_map,
    tree_unflatten,
)

Point = namedtuple("Point", ["x", "y"])  # noqa: PYI024


class _Tagged(list[int]):
    """A list subclass: it may carry state, so it is a leaf, not a node."""


_TREES = {
    "leaf": 7,
    "none": None,
    "string": "text",
    "tuple": (1, 2),
    "list": [1, 2],
    "dict": {"a": 1, "b": 2},
    "namedtuple": Point(1, 2),
    "empty tuple": (),
    "empty list": [],
    "empty dict": {},
    "nested": {"w": (1, [2, 3]), "p": Point(4, {"k": 5}), "n": None},
    "singleton": ([()],),
}


@pytest.mark.parametrize("name", _TREES)
def test_flatten_unflatten_round_trip(name: str) -> None:
    tree = _TREES[name]
    leaves, spec = tree_flatten(tree)
    rebuilt = tree_unflatten(leaves, spec)
    assert rebuilt == tree and type(rebuilt) is type(tree)
    assert spec.num_leaves == len(leaves)


@pytest.mark.parametrize(
    ("name", "leaves"),
    [
        ("leaf", [7]),
        ("none", [None]),
        ("tuple", [1, 2]),
        ("dict", [1, 2]),
        ("namedtuple", [1, 2]),
        ("empty dict", []),
        ("nested", [1, 2, 3, 4, 5, None]),
        ("singleton", []),
    ],
)
def test_leaves_come_out_in_order(name: str, leaves: list[object]) -> None:
    assert tree_flatten(_TREES[name])[0] == leaves


def test_every_kind_of_node_is_told_apart() -> None:
    specs = [tree_flatten(t)[1] for t in ((1, 2), [1, 2], {"a": 1, "b": 2}, Point(1, 2), 1)]
    assert len(set(specs)) == len(specs)
    assert tree_flatten((1, 2))[1] == tree_flatten((3, 4))[1]


def test_a_namedtuple_is_rebuilt_as_its_own_class() -> None:
    leaves, spec = tree_flatten({"p": Point(1, 2)})
    assert tree_unflatten([10, 20], spec) == {"p": Point(10, 20)}
    assert isinstance(tree_unflatten([10, 20], spec), dict)
    assert type(tree_unflatten(leaves, spec)["p"]) is Point


def test_dict_keeps_its_order_and_order_is_part_of_the_structure() -> None:
    forward, backward = {"a": 1, "b": 2}, {"b": 2, "a": 1}
    assert list(tree_unflatten(*tree_flatten(backward))) == ["b", "a"]
    assert tree_flatten(forward)[1] != tree_flatten(backward)[1]


@pytest.mark.parametrize("subclass", [_Tagged([1, 2]), OrderedDict(a=1)])
def test_a_container_subclass_is_a_leaf(subclass: object) -> None:
    leaves, spec = tree_flatten(subclass)
    assert leaves == [subclass] and spec.num_leaves == 1 and spec.node is None


def test_unflatten_checks_the_leaf_count() -> None:
    spec = tree_flatten((1, [2, 3]))[1]
    with pytest.raises(StructureError, match="expected 3 leaves"):
        spec.unflatten([1, 2])
    with pytest.raises(StructureError, match="expected 3 leaves"):
        spec.unflatten([1, 2, 3, 4])


def test_match_returns_the_leaves_of_the_same_structure() -> None:
    spec = tree_flatten({"a": (1, 2), "b": 3})[1]
    assert spec.match({"a": ("x", "y"), "b": "z"}) == ["x", "y", "z"]
    for other in ({"a": (1, 2)}, {"a": [1, 2], "b": 3}, {"b": 3, "a": (1, 2)}, 5):
        with pytest.raises(StructureError, match="expected a pytree of structure"):
            spec.match(other)


@pytest.mark.parametrize(
    ("prefix", "entries"),
    [
        (0, [0, 0, 0]),
        (None, [None, None, None]),
        ({"a": 0, "b": None}, [0, 0, None]),
        ({"a": (1, None), "b": 2}, [1, None, 2]),
        ({"a": 5, "b": [0]}, [5, 5, 0]),
    ],
)
def test_broadcast_spreads_a_prefix_over_the_leaves(prefix: object, entries: list[object]) -> None:
    spec = tree_flatten({"a": (1, 2), "b": [3]})[1]
    assert spec.broadcast(prefix) == entries


@pytest.mark.parametrize(
    "prefix",
    [{"a": 0}, {"a": 0, "c": 0}, (0, 0), {"a": [0, 0], "b": [0]}, {"a": (0, 0, 0), "b": [0]}],
)
def test_broadcast_refuses_a_prefix_that_does_not_fit(prefix: object) -> None:
    spec = tree_flatten({"a": (1, 2), "b": [3]})[1]
    with pytest.raises(StructureError, match="cannot be given a prefix"):
        spec.broadcast(prefix)


def test_broadcast_refuses_a_node_where_there_is_a_leaf() -> None:
    with pytest.raises(StructureError, match="cannot be given a prefix"):
        tree_flatten(1)[1].broadcast((0,))


def test_tree_map_keeps_the_structure() -> None:
    tree = {"a": (1, [2]), "p": Point(3, None)}
    assert tree_map(lambda x: x * 10 if isinstance(x, int) else x, tree) == {
        "a": (10, [20]),
        "p": Point(30, None),
    }


def test_the_spec_describes_itself() -> None:
    assert str(tree_flatten(1)[1]) == "*"
    assert str(tree_flatten((1, [2], {"k": 3}, Point(4, 5)))[1]) == (
        "tuple(*, list(*), dict['k'](*), Point(*, *))"
    )


def test_a_spec_is_frozen_and_hashable_and_has_no_dict() -> None:
    spec = tree_flatten((1, 2))[1]
    assert isinstance(spec, Hashable) and hash(spec) == hash(tree_flatten((8, 9))[1])
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.context = 1
    assert not hasattr(spec, "__dict__")


def test_a_new_container_is_one_registered_class() -> None:
    """Defining a ``Node`` subclass is all it takes to make a type a node."""

    class _Pair:
        def __init__(self, left: object, right: object) -> None:
            self.left, self.right = left, right

    class _PairNode(Node, node_type=_Pair):
        def flatten(self, obj: object) -> tuple[tuple[object, ...], Hashable]:
            assert isinstance(obj, _Pair)
            return (obj.left, obj.right), None

        def unflatten(self, context: Hashable, children: Sequence[object]) -> object:
            return _Pair(*children)

    try:
        leaves, spec = tree_flatten({"p": _Pair(1, (2, 3))})
        assert leaves == [1, 2, 3] and str(spec) == "dict['p'](_Pair(*, tuple(*, *)))"
        rebuilt = tree_unflatten([4, 5, 6], spec)
        assert isinstance(rebuilt, dict) and isinstance(rebuilt["p"], _Pair)
        assert (rebuilt["p"].left, rebuilt["p"].right) == (4, (5, 6))
    finally:
        Node._registry.remove(next(n for n in Node._registry if isinstance(n, _PairNode)))
    assert tree_flatten(_Pair(1, 2))[0][0].left == 1  # a leaf again, once it is gone


def test_an_incomplete_node_cannot_be_defined() -> None:
    before = list(Node._registry)
    with pytest.raises(TypeError, match="abstract"):

        class _Incomplete(Node, node_type=frozenset):
            pass

    assert Node._registry == before


def test_a_subclass_without_a_node_type_is_not_a_container_kind() -> None:
    before = list(Node._registry)

    class _Abstract(Node):
        pass

    assert Node._registry == before
    assert TreeSpec.of(1)[1].node is None
