"""Pytrees for ``lucid.func``: nested tuples, lists and dicts of tensors.

A transform sees its inputs and outputs as *pytrees* — a tensor is a leaf,
and a tuple, a list, a dict or a namedtuple is a node holding more
pytrees.  ``tree_flatten`` takes a pytree apart into its leaves and a
:class:`TreeSpec`, the shape they were in; ``tree_unflatten`` puts leaves
back into a spec.  Every transform works on the flat list of leaves and
uses the spec to give the result the structure of the thing it came from.

Each kind of node is a :class:`Node` subclass that declares the container
type it owns and is registered by being defined::

    class _SetNode(Node, node_type=frozenset):
        ...

so a new container is one class, and nothing here branches on the type of
the object it is given.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Final, final, override

__all__ = [
    "DictNode",
    "ListNode",
    "NamedTupleNode",
    "Node",
    "StructureError",
    "TreeSpec",
    "TupleNode",
    "tree_flatten",
    "tree_map",
    "tree_unflatten",
]


class StructureError(ValueError):
    """Two pytrees that had to share a structure do not.

    A ``ValueError``: the arguments are of the right types, but their
    nesting does not fit — tangents that are not shaped like their primals,
    an ``in_dims`` that names a key the argument lacks.
    """


class Node(ABC):
    """One kind of container a pytree descends into.

    A subclass names the type it owns — ``class ListNode(Node,
    node_type=list)`` — and is entered in the registry when the class
    body ends.  One instance per subclass is kept: a node holds no state,
    the :class:`TreeSpec` carries the per-object ``context`` (the keys of
    a dict, the class of a namedtuple) it needs to rebuild the container.
    """

    node_type: ClassVar[type]
    _registry: ClassVar[list[Node]] = []

    @override
    def __init_subclass__(cls, *, node_type: type | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if node_type is None:
            return  # an abstract intermediate, not a container kind
        # ``ABCMeta`` only learns which methods are still abstract after this
        # hook returns, so instantiating here would not refuse an incomplete kind.
        unimplemented = sorted(
            name
            for name in Node.__abstractmethods__
            if getattr(getattr(cls, name), "__isabstractmethod__", False)
        )
        if unimplemented:
            raise TypeError(
                f"cannot define the pytree node {cls.__name__}: abstract method(s) "
                f"{', '.join(unimplemented)} not implemented"
            )
        cls.node_type = node_type
        Node._registry.append(cls())

    @staticmethod
    def of(obj: object) -> Node | None:
        """The node kind that ``obj`` is an instance of, or ``None`` for a leaf."""
        return next((kind for kind in Node._registry if kind.claims(obj)), None)

    def claims(self, obj: object) -> bool:
        """Whether ``obj`` is a container of this kind.

        Exact type, not ``isinstance``: a subclass of ``list`` may carry
        state that rebuilding it from its children would drop.
        """
        return type(obj) is self.node_type

    @abstractmethod
    def flatten(self, obj: object) -> tuple[tuple[object, ...], Hashable]:
        """``obj``'s children, in order, and what rebuilding it needs besides."""

    @abstractmethod
    def unflatten(self, context: Hashable, children: Sequence[object]) -> object:
        """A container of this kind, from what :meth:`flatten` returned."""

    def label(self, context: Hashable) -> str:
        """The name this node goes by in a structure's description."""
        del context  # the kinds with no per-object state
        return self.node_type.__name__


@final
class TupleNode(Node, node_type=tuple):
    @override
    def flatten(self, obj: object) -> tuple[tuple[object, ...], Hashable]:
        assert isinstance(obj, tuple)
        return obj, None

    @override
    def unflatten(self, context: Hashable, children: Sequence[object]) -> object:
        return tuple(children)


@final
class ListNode(Node, node_type=list):
    @override
    def flatten(self, obj: object) -> tuple[tuple[object, ...], Hashable]:
        assert isinstance(obj, list)
        return tuple(obj), None

    @override
    def unflatten(self, context: Hashable, children: Sequence[object]) -> object:
        return list(children)


@final
class DictNode(Node, node_type=dict):
    @override
    def flatten(self, obj: object) -> tuple[tuple[object, ...], Hashable]:
        assert isinstance(obj, dict)
        return tuple(obj.values()), tuple(obj)

    @override
    def unflatten(self, context: Hashable, children: Sequence[object]) -> object:
        assert isinstance(context, tuple)
        return dict(zip(context, children, strict=True))

    @override
    def label(self, context: Hashable) -> str:
        assert isinstance(context, tuple)
        return f"dict[{', '.join(map(repr, context))}]"


@final
class NamedTupleNode(Node, node_type=tuple):
    @override
    def claims(self, obj: object) -> bool:
        return isinstance(obj, tuple) and type(obj) is not tuple and hasattr(obj, "_fields")

    @override
    def flatten(self, obj: object) -> tuple[tuple[object, ...], Hashable]:
        assert isinstance(obj, tuple)
        return obj, type(obj)

    @override
    def unflatten(self, context: Hashable, children: Sequence[object]) -> object:
        assert callable(context)
        return context(*children)

    @override
    def label(self, context: Hashable) -> str:
        return getattr(context, "__name__", "namedtuple")


@final
@dataclass(frozen=True, slots=True)
class TreeSpec:
    """The shape of a pytree with its leaves taken out.

    A leaf has ``node=None``.  Anything else records the :class:`Node` kind
    it was, the ``context`` that kind needs to rebuild it, and the specs of
    its children.  Frozen and hashable, so two structures compare with ``==``.
    """

    node: Node | None
    context: Hashable
    children: tuple[TreeSpec, ...]
    num_leaves: int = field(init=False, compare=False)

    def __post_init__(self) -> None:
        count = 1 if self.node is None else sum(c.num_leaves for c in self.children)
        object.__setattr__(self, "num_leaves", count)

    @staticmethod
    def of(obj: object) -> tuple[list[object], TreeSpec]:
        """``obj``'s leaves, in order, and the spec to put them back with."""
        kind = Node.of(obj)
        if kind is None:
            return [obj], _LEAF
        children, context = kind.flatten(obj)
        leaves: list[object] = []
        specs: list[TreeSpec] = []
        for child in children:
            child_leaves, child_spec = TreeSpec.of(child)
            leaves.extend(child_leaves)
            specs.append(child_spec)
        return leaves, TreeSpec(kind, context, tuple(specs))

    def unflatten(self, leaves: Sequence[object]) -> object:
        """The pytree of this shape whose leaves are ``leaves``."""
        if len(leaves) != self.num_leaves:
            raise StructureError(f"expected {self.num_leaves} leaves for {self}, got {len(leaves)}")
        return self._build(iter(leaves))

    def _build(self, leaves: Iterator[object]) -> object:
        if self.node is None:
            return next(leaves)
        return self.node.unflatten(self.context, [c._build(leaves) for c in self.children])

    def match(self, obj: object) -> list[object]:
        """``obj``'s leaves, which must have exactly this structure."""
        leaves, spec = TreeSpec.of(obj)
        if spec != self:
            raise StructureError(f"expected a pytree of structure {self}, got {spec}")
        return leaves

    def broadcast(self, prefix: object) -> list[object]:
        """One entry per leaf of this spec, from a pytree that is a prefix of it.

        A leaf of ``prefix`` stands for every leaf of the subtree it sits
        over — ``in_dims=0`` for all arguments, ``{"w": None}`` for one
        key — and a node of ``prefix`` has to be the node of the same kind
        and context that is there.
        """
        kind = Node.of(prefix)
        if kind is None:
            return [prefix] * self.num_leaves
        children, context = kind.flatten(prefix)
        if kind is not self.node or context != self.context or len(children) != len(self.children):
            raise StructureError(f"{self} cannot be given a prefix {TreeSpec.of(prefix)[1]}")
        return [
            entry
            for spec, child in zip(self.children, children, strict=True)
            for entry in spec.broadcast(child)
        ]

    @override
    def __str__(self) -> str:
        if self.node is None:
            return "*"
        inner = ", ".join(map(str, self.children))
        return f"{self.node.label(self.context)}({inner})"


_LEAF: Final = TreeSpec(None, None, ())


def tree_flatten(obj: object) -> tuple[list[object], TreeSpec]:
    """``obj``'s leaves, in order, and the :class:`TreeSpec` that rebuilds it.

    A tensor, ``None`` or any object that is not a registered container is
    a leaf; the others are walked.
    """
    return TreeSpec.of(obj)


def tree_unflatten(leaves: Sequence[object], spec: TreeSpec) -> object:
    """The pytree of structure ``spec`` holding ``leaves``, in order.

    Parameters
    ----------
    leaves : sequence of object
        Exactly ``spec.num_leaves`` of them.
    spec : TreeSpec
        The structure to build.

    Returns
    -------
    object
        The rebuilt pytree.

    Raises
    ------
    StructureError
        The number of leaves is not ``spec.num_leaves``.
    """
    return spec.unflatten(leaves)


def tree_map(fn: Callable[[object], object], obj: object) -> object:
    """``obj`` with ``fn`` applied to each leaf, in the same structure.

    Parameters
    ----------
    fn : Callable
        Applied to each leaf, in the order :func:`tree_flatten` lists them.
    obj : object
        The pytree.

    Returns
    -------
    object
        A pytree of the structure of ``obj``.
    """
    leaves, spec = TreeSpec.of(obj)
    return spec.unflatten([fn(leaf) for leaf in leaves])
