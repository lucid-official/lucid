"""What every ``lucid.func`` transform has in common.

A transform takes a function and gives back a function.  Here that "function"
is an object: a :class:`Transform` holds the user's ``func`` and what the
transform was configured with, and calling it runs the transform.  The
public ``grad`` / ``jacrev`` / ... in :mod:`lucid.func` just build one.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

__all__ = ["Dims", "Isolated", "PyTree", "Transform"]

#: A tensor, or a tuple / list / dict / namedtuple nesting tensors — what a
#: transform takes apart with :mod:`lucid.func._pytree` and puts back
#: together.  The result of a transform has the structure of the thing it
#: differentiates or returns, which no static type can say.
type PyTree = object

#: ``in_dims`` / ``out_dims``: an axis (``None`` for "not batched") for every
#: tensor of a pytree, or a pytree that is a prefix of it, whose leaves each
#: stand for the whole subtree they sit over.
type Dims = int | None | tuple[Dims, ...] | list[Dims] | dict[str, Dims]


class Isolated:
    """Marks what :func:`~lucid.func.vmap` must call once per batch element.

    The result of a derivative transform has to be built from one example at
    a time: ``jacrev`` over a batched input would differentiate every
    output with respect to every batch element, giving ``(B, out, B, in)``
    where ``(B, out, in)`` is meant.  ``vmap`` looks for this flag on the
    function it is given and, when it is set, slices the batch itself
    instead of moving the batch axis to the front.
    """

    __slots__ = ()

    _lucid_needs_vmap_isolation: ClassVar[bool] = True


class Transform[**P, R](Isolated, ABC):
    """A function transform: the user's ``func`` in, a transformed function out.

    A subclass is configured in ``__init__`` and defines what calling the
    transformed function does in ``__call__``, with the arguments
    ``func`` itself takes.
    """

    __slots__ = ("func",)

    def __init__(self, func: Callable[P, object]) -> None:
        self.func = func

    @abstractmethod
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Run the transformed function."""
