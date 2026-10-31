"""A user function seen through its pytrees.

The derivative machinery in :mod:`lucid.autograd` works on a flat tuple of
tensors in and a flat tuple of tensors out.  What a ``lucid.func`` caller
writes is a function of nested arguments returning nested results.
:class:`Selection` is the bridge on the way in — which arguments are
differentiated, flattened — and :class:`FlatFunction` /
:class:`LossFunction` the bridge on the way out; :func:`pack_jacobian`
puts a flat answer back into the structure it belongs to.
"""

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import override

import lucid
from lucid._ctx import _TransformLevel, transform_depth
from lucid._tensor.tensor import Tensor
from lucid.func._pytree import TreeSpec, tree_flatten, tree_map

__all__ = [
    "FlatFunction",
    "LossFunction",
    "ScalarLoss",
    "ScalarOutputError",
    "Selection",
    "as_tensors",
    "detach_aux",
    "keeps_graph",
    "pack_jacobian",
]

_MIN_AUX_TUPLE_LEN = 2


def as_tensors(leaves: Sequence[object], name: str, expected: str) -> list[Tensor]:
    """``leaves`` as tensors; ``TypeError`` for the first that is not one.

    Parameters
    ----------
    leaves : sequence of object
        The flattened leaves of a pytree.
    name : str
        The transform to name in the error.
    expected : str
        What it wanted in the place of the leaf it refuses.

    Returns
    -------
    list of Tensor
        The same leaves, typed as tensors.
    """
    tensors: list[Tensor] = []
    for leaf in leaves:
        if not isinstance(leaf, Tensor):
            raise TypeError(f"lucid.func.{name}: {expected}, found a {type(leaf).__name__}")
        tensors.append(leaf)
    return tensors


def keeps_graph(leaves: Sequence[Tensor]) -> bool:
    """Whether a reverse-mode transform at ``leaves`` must keep its result
    differentiable.

    It must when something outside can still differentiate through it: a
    leaf requires grad, so the caller's own graph runs through the result,
    or the transform sits inside another transform's function
    (:func:`~lucid._ctx.transform_depth`), whose derivative is taken of
    this result.  Otherwise nothing will ask for a second derivative, and
    the graph of the first is dropped as soon as it is computed.  Under
    ``no_grad`` nothing is kept.
    """
    if not lucid.is_grad_enabled():
        return False
    return transform_depth() > 0 or any(t.requires_grad for t in leaves)


def split_aux(
    output: object, *, has_aux: bool, name: str, label: str = "output"
) -> tuple[object, object]:
    """``(output, aux)`` from what ``func`` returned — ``aux`` is ``None``
    unless ``has_aux``, and then everything after the first item.

    Parameters
    ----------
    output : object
        What ``func`` returned.
    has_aux : bool
        Whether ``func`` was declared to return ``(output, aux)``.
    name : str
        The transform, for the error message.
    label : str
        What the first item is called in that message.

    Returns
    -------
    tuple
        ``(output, aux)``.
    """
    if not has_aux:
        return output, None
    if not isinstance(output, tuple) or len(output) < _MIN_AUX_TUPLE_LEN:
        raise ValueError(f"lucid.func.{name}: has_aux=True requires func to return ({label}, aux)")
    aux = output[1] if len(output) == _MIN_AUX_TUPLE_LEN else output[1:]
    return output[0], aux


class Selection:
    """The arguments of one call that a transform differentiates, flattened.

    ``argnums`` is an ``int`` for one argument or a tuple for several; what is
    chosen is that argument's pytree, or the tuple of them.  Its leaves are
    the tensors the derivative is taken at, and its :class:`TreeSpec` is the
    structure every derivative of them comes back in — which is why an int
    ``argnums`` gives a bare result and a tuple gives a tuple.

    A leaf that is not a tensor is refused, unless ``allow_fixed`` — then it
    is held at its value, as ``jvp`` holds a plain number among its primals.
    """

    __slots__ = ("argnums", "scalar", "spec", "leaves", "_name", "_flat", "_tensor_at")

    def __init__(
        self,
        args: Sequence[object],
        argnums: int | tuple[int, ...],
        name: str,
        *,
        allow_fixed: bool = False,
    ) -> None:
        self._name = name
        self.scalar = isinstance(argnums, int)
        self.argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
        for i in self.argnums:
            if not -len(args) <= i < len(args):
                raise IndexError(
                    f"lucid.func.{name}: argnums {i} is out of range for "
                    f"{len(args)} positional arguments"
                )
        chosen = tuple(args[i] for i in self.argnums)
        self._flat, self.spec = tree_flatten(chosen[0] if self.scalar else chosen)
        self._tensor_at = tuple(i for i, leaf in enumerate(self._flat) if isinstance(leaf, Tensor))
        if not allow_fixed:
            for leaf in self._flat:
                if not isinstance(leaf, Tensor):
                    raise TypeError(
                        f"lucid.func.{name}: argnums {argnums} select a pytree of Tensors to "
                        f"differentiate, found a {type(leaf).__name__}"
                    )
        self.leaves = tuple(leaf for leaf in self._flat if isinstance(leaf, Tensor))

    def splice(self, args: Sequence[object], tensors: Sequence[Tensor]) -> tuple[object, ...]:
        """``args`` with the chosen arguments rebuilt around ``tensors``."""
        flat = list(self._flat)
        for position, tensor in zip(self._tensor_at, tensors, strict=True):
            flat[position] = tensor
        chosen = self.spec.unflatten(flat)
        out = list(args)
        if self.scalar:
            out[self.argnums[0]] = chosen
        else:
            assert isinstance(chosen, tuple)
            for i, value in zip(self.argnums, chosen, strict=True):
                out[i] = value
        return tuple(out)

    def tracked(self) -> list[Tensor]:
        """The leaves as tensors that require grad: a leaf that already does
        is used as it is, so the graph through it stays the caller's; any
        other is a fresh leaf that shares its storage."""
        return [t if t.requires_grad else t.detach().requires_grad_(True) for t in self.leaves]

    def tangents(self, tangents: object) -> tuple[Tensor, ...]:
        """The tensors of ``tangents`` — a pytree of exactly the chosen
        structure — that belong to the chosen tensors."""
        flat = self.spec.match(tangents)
        return tuple(
            as_tensors([flat[i] for i in self._tensor_at], self._name, "a tangent must be a Tensor")
        )


class _Traced:
    """One run of ``func`` at the selected tensors, as every transform makes it.

    Splices the tensors into the call, runs ``func`` as a level of
    transform nesting (so a transform inside it knows it is inside one),
    and keeps the ``has_aux`` side output of the first run.
    """

    __slots__ = (
        "_func",
        "_args",
        "_kwargs",
        "_selection",
        "_has_aux",
        "_name",
        "_ran",
        "_out_spec",
        "aux",
    )

    def __init__(
        self,
        func: Callable[..., object],
        args: Sequence[object],
        kwargs: dict[str, object],
        selection: Selection,
        *,
        has_aux: bool,
        name: str,
    ) -> None:
        self._func = func
        self._args = tuple(args)
        self._kwargs = kwargs
        self._selection = selection
        self._has_aux = has_aux
        self._name = name
        self._ran = False
        self._out_spec: TreeSpec | None = None
        self.aux: object = None

    def _run(self, tensors: Sequence[Tensor], label: str, *, differentiated: bool = True) -> object:
        level: AbstractContextManager[None] = _TransformLevel() if differentiated else nullcontext()
        with level:
            raw = self._func(*self._selection.splice(self._args, tensors), **self._kwargs)
        first, aux = split_aux(raw, has_aux=self._has_aux, name=self._name, label=label)
        if not self._ran:
            self._ran = True
            self.aux = aux
        return first


class FlatFunction(_Traced):
    """``func`` as a function of the selected tensors, returning a flat tuple
    of tensors — its output pytree taken apart.  :attr:`out_spec`, set by
    the first call, is the structure to give the results back."""

    __slots__ = ()

    @property
    def out_spec(self) -> TreeSpec:
        """The structure of ``func``'s output, known once it has run."""
        assert self._out_spec is not None, "out_spec is read after func has run"
        return self._out_spec

    def __call__(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        return self._flat_run(tensors, differentiated=True)

    def observe(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        """A run no derivative is taken through — to learn the output's
        shapes and whether it reaches the caller's graph — so a transform
        inside ``func`` is not counted as nested in this one."""
        return self._flat_run(tensors, differentiated=False)

    def _flat_run(self, tensors: Sequence[Tensor], *, differentiated: bool) -> tuple[Tensor, ...]:
        raw = self._run(tensors, "output", differentiated=differentiated)
        leaves, spec = tree_flatten(raw)
        outputs = as_tensors(leaves, self._name, "func must return a Tensor or a pytree of Tensors")
        if self._out_spec is None:
            self._out_spec = spec
        return tuple(outputs)


class ScalarOutputError(RuntimeError, ValueError):
    """``func`` returned a tensor that is not 0-d where the transform
    differentiates a scalar.

    A ``RuntimeError`` as the reference framework raises it, and a
    ``ValueError`` because the output's shape is a value the caller chose.
    """


class LossFunction(_Traced):
    """``func`` as a function of the selected tensors returning the one tensor
    it must return — a loss, a scalar to take the gradient or Hessian of."""

    __slots__ = ()

    def __call__(self, *tensors: Tensor) -> Tensor:
        value = self._run(tensors, "loss")
        if not isinstance(value, Tensor):
            kind = f"a tuple of {len(value)}" if isinstance(value, tuple) else type(value).__name__
            raise TypeError(
                f"lucid.func.{self._name}: func must return a single Tensor, got {kind}"
            )
        self._check(value)
        return value

    def _check(self, value: Tensor) -> None:
        """Refuse a ``value`` this transform cannot differentiate."""


class ScalarLoss(LossFunction):
    """A :class:`LossFunction` whose tensor has to be 0-d — what ``grad``
    takes: one number to differentiate, not one per output element."""

    __slots__ = ()

    @override
    def _check(self, value: Tensor) -> None:
        if value.ndim != 0:
            raise ScalarOutputError(
                f"{self._name}(f)(*args): Expected f(*args) to return a scalar Tensor, got "
                f"tensor with {value.ndim} dims. Maybe you wanted to use the vjp or jacrev "
                "APIs instead?"
            )


def detach_aux(aux: object, *, keep: bool) -> object:
    """``aux`` — the side output of a ``has_aux`` function — as the caller
    gets it: attached to the graph only where the transform's own results
    are (``keep``), otherwise a plain value.

    Parameters
    ----------
    aux : object
        A tensor or a pytree of them, or anything else.
    keep : bool
        Whether the transform keeps the graph of its results.

    Returns
    -------
    object
        ``aux`` itself when ``keep``, else the same structure with each
        tensor detached.
    """
    if keep:
        return aux
    return tree_map(lambda leaf: leaf.detach() if isinstance(leaf, Tensor) else leaf, aux)


def pack_jacobian(
    out_spec: TreeSpec, in_spec: TreeSpec, rows: Sequence[Sequence[object]]
) -> object:
    """A Jacobian's blocks in the structure the reference framework gives it.

    ``rows[i][j]`` is the block of output leaf ``i`` and input leaf ``j``.
    The result has the output's structure, and each of its leaves is the
    input's structure holding that output leaf's blocks — so a dict-valued
    ``func`` of a tuple ``argnums`` gives ``{"y": (J_x0, J_x1)}``.

    Parameters
    ----------
    out_spec : TreeSpec
        The structure of ``func``'s output.
    in_spec : TreeSpec
        The structure of the differentiated arguments.
    rows : sequence of sequence of object
        The blocks, one row per output leaf and one entry per input leaf.

    Returns
    -------
    object
        The Jacobian as a pytree.
    """
    return out_spec.unflatten([in_spec.unflatten(list(row)) for row in rows])
