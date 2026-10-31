"""``vmap``: a function of one example, run over a batch of them.

``vmap`` takes its arguments apart as pytrees, so a dict or a list of
tensors is batched leaf by leaf, and puts the results back together in
whatever structure ``func`` returned.  :class:`_Batch` is the arguments
once flattened — the leaves, the axis each is batched along and the batch
size — and :class:`_Outputs` is the way back.

What ``vmap`` is asked is checked where it is asked: ``in_dims`` /
``out_dims`` that are not dims, a ``chunk_size`` that is not positive, an
unknown ``strategy`` or ``randomness`` are refused when the transform is
built, and a batch that does not fit the ``in_dims`` when it is called.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import ClassVar, Final, final, override

import lucid
from lucid._ctx import _RandomnessGuard
from lucid._tensor.tensor import Tensor
from lucid.func._pytree import Node, StructureError, TreeSpec, tree_flatten
from lucid.func._randomness import Draws, Randomness
from lucid.func._transform import Isolated, PyTree, Transform

__all__ = ["NonTensorError", "PerElement", "Vmap", "WholeBatch"]


class NonTensorError(ValueError):
    """``vmap`` was given, or ``func`` returned, a value that had to be a
    tensor and is not.

    A ``ValueError`` and not a ``TypeError``, as in the reference
    framework: it is the *use* of the value (batched along an axis) that is
    refused, and the same object is fine at an ``in_dim`` / ``out_dim`` of
    ``None``.
    """


def _axis(x: Tensor, dim: int) -> int:
    """``dim`` as a non-negative axis of ``x``."""
    return dim if dim >= 0 else len(x.shape) + dim


# ── in_dims / out_dims ───────────────────────────────────────────────────────


class _Dims(ABC):
    """``in_dims`` or ``out_dims`` as the caller gave it: an int or ``None``
    for every tensor of a pytree, or a pytree that is a prefix of it."""

    __slots__ = ("_prefix",)
    name: ClassVar[str]
    subject: ClassVar[str]

    def __init__(self, prefix: object) -> None:
        leaves, _ = tree_flatten(prefix)
        if not self._fits(prefix) or any(
            leaf is not None and not isinstance(leaf, int) for leaf in leaves
        ):
            raise ValueError(self._expected(prefix))
        self._prefix = prefix

    def _fits(self, prefix: object) -> bool:
        """Whether ``prefix`` may stand at the top, whatever it holds."""
        del prefix
        return True

    @abstractmethod
    def _expected(self, prefix: object) -> str:
        """The refusal for a ``prefix`` that is not dims."""

    def _prefix_for(self, spec: TreeSpec) -> object:
        """The prefix to lay over ``spec``."""
        del spec
        return self._prefix

    def entries(self, spec: TreeSpec) -> list[int | None]:
        """One entry per leaf of ``spec`` — the dim each is batched along."""
        try:
            broadcast = spec.broadcast(self._prefix_for(spec))
        except StructureError as error:
            raise StructureError(
                f"vmap: {self.name} is not compatible with the structure of {self.subject}: {error}"
            ) from None
        dims: list[int | None] = []
        for entry in broadcast:
            assert entry is None or isinstance(entry, int)
            dims.append(entry)
        return dims


@final
class _InDims(_Dims):
    __slots__ = ()
    name = "in_dims"
    subject = "`inputs`"

    @override
    def _fits(self, prefix: object) -> bool:
        return Node.of(prefix) is None or isinstance(prefix, tuple)

    @override
    def _expected(self, prefix: object) -> str:
        return (
            "vmap: expected `in_dims` to be int or a (potentially nested) tuple "
            f"matching the structure of inputs, got: {prefix!r}."
        )


@final
class _OutDims(_Dims):
    __slots__ = ()
    name = "out_dims"
    subject = "`outputs`"

    @override
    def _expected(self, prefix: object) -> str:
        return (
            "vmap: `out_dims` must be an int, None or a python collection of ints "
            "representing where in the outputs the vmapped dimension should appear."
        )

    @override
    def _prefix_for(self, spec: TreeSpec) -> object:
        # ``out_dims=(1,)`` for the one tensor ``func`` returns
        if spec.node is None and isinstance(self._prefix, tuple) and len(self._prefix) == 1:
            return self._prefix[0]
        return self._prefix


# ── the arguments ────────────────────────────────────────────────────────────


@final
class _Batch:
    """The positional arguments of one ``vmap`` call, flattened."""

    __slots__ = ("_leaves", "_spec", "_dims", "size")

    def __init__(self, args: tuple[object, ...], in_dims: _InDims) -> None:
        self._leaves, self._spec = tree_flatten(args)
        self._dims = in_dims.entries(self._spec)
        self.size = self._batch_size()

    def _batch_size(self) -> int | None:
        """The batch length shared by the batched leaves, or ``None`` if none is."""
        size: int | None = None
        for leaf, in_dim in zip(self._leaves, self._dims, strict=True):
            if in_dim is None:
                continue
            if not isinstance(leaf, Tensor):
                raise NonTensorError(
                    f"vmap: Got in_dim={in_dim} for an input but the input is of type "
                    f"{type(leaf).__name__}. We cannot vmap over non-Tensor arguments, "
                    "please use None as the respective in_dim"
                )
            ndim = len(leaf.shape)
            axis = _axis(leaf, in_dim)
            if not 0 <= axis < ndim:
                raise ValueError(
                    f"vmap: in_dim {in_dim} is out of range for an input that is a Tensor of "
                    f"dimensionality {ndim}; expected in_dim to satisfy "
                    f"-{ndim} <= in_dim < {ndim}."
                )
            n = int(leaf.shape[axis])
            if size is not None and size != n:
                raise ValueError(
                    "vmap: Expected all tensors to have the same size in the mapped "
                    f"dimension (inconsistent batch sizes {size} vs {n})"
                )
            size = n
        return size

    def _map(self, fn: Callable[[Tensor, int], Tensor]) -> tuple[object, ...]:
        """The arguments with ``fn(leaf, axis)`` in place of each batched leaf."""
        mapped = [
            fn(leaf, dim) if dim is not None and isinstance(leaf, Tensor) else leaf
            for leaf, dim in zip(self._leaves, self._dims, strict=True)
        ]
        rebuilt = self._spec.unflatten(mapped)
        assert isinstance(rebuilt, tuple)
        return rebuilt

    def element(self, index: int) -> tuple[object, ...]:
        """The arguments of batch element ``index``."""

        def select(x: Tensor, dim: int) -> Tensor:
            where: list[int | slice] = [slice(None)] * len(x.shape)
            where[_axis(x, dim)] = index
            return x[tuple(where)]

        return self._map(select)

    def chunk(self, indices: range) -> tuple[object, ...]:
        """The arguments of batch elements ``indices``, still batched, the
        batch axis at the front."""

        def window(x: Tensor, dim: int) -> Tensor:
            axis = _axis(x, dim)
            where: list[slice] = [slice(None)] * len(x.shape)
            where[axis] = slice(indices.start, indices.stop)
            part = x[tuple(where)]
            return part if axis == 0 else lucid.moveaxis(part, axis, 0)

        return self._map(window)

    def front(self) -> tuple[object, ...]:
        """The arguments with every batch axis moved to the front."""

        def to_front(x: Tensor, dim: int) -> Tensor:
            axis = _axis(x, dim)
            return x if axis == 0 else lucid.moveaxis(x, axis, 0)

        return self._map(to_front)

    def prototype(self) -> tuple[object, ...]:
        """The arguments of one example that is not there: zeros of the
        logical shape and dtype — what an empty batch is traced with."""

        def blank(x: Tensor, dim: int) -> Tensor:
            shape = list(x.shape)
            del shape[_axis(x, dim)]
            return lucid.zeros(tuple(shape), dtype=x.dtype, device=x.device)

        return self._map(blank)


# ── the results ──────────────────────────────────────────────────────────────


class _Joiner(ABC):
    """How the same output leaf of several runs becomes one tensor."""

    @abstractmethod
    def join(self, tensors: list[Tensor], dim: int) -> Tensor:
        """``tensors`` — one per run — as one tensor batched along ``dim``."""


@final
class _Stack(_Joiner):
    """One tensor per batch element: a new axis at ``dim``."""

    @override
    def join(self, tensors: list[Tensor], dim: int) -> Tensor:
        return lucid.stack(tensors, dim=dim)


@final
class _Concatenate(_Joiner):
    """One tensor per chunk, already batched: joined along ``dim``."""

    @override
    def join(self, tensors: list[Tensor], dim: int) -> Tensor:
        return lucid.cat(tensors, dim=dim)


_STACK: Final = _Stack()
_CONCATENATE: Final = _Concatenate()


def _identical(a: Tensor, b: Tensor) -> bool:
    """Whether two tensors hold the same values (``NaN`` equal to ``NaN``)."""
    if tuple(a.shape) != tuple(b.shape):
        return False
    return lucid.equal(a, b) or lucid.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True)


def _unbatched(values: Sequence[object]) -> object:
    """The one value of an output that is not batched (``out_dims=None``).

    ``func`` may not let it depend on the batch, which the runs prove: a
    tensor that differs between them is a batched one.
    """
    first = values[0]
    if isinstance(first, Tensor):
        for other in values[1:]:
            if not isinstance(other, Tensor) or not _identical(first, other):
                raise ValueError(
                    "vmap: func returned a Tensor that depends on the batch where out_dims "
                    "is None; give it an out_dim"
                )
    return first


def _tensor(value: object) -> Tensor:
    """``value``, which an output with an ``out_dim`` has to be a tensor."""
    if not isinstance(value, Tensor):
        raise NonTensorError(
            f"vmap: func must only return Tensors, got type {type(value).__name__}. "
            "Did you mean to set out_dims= to None for output?"
        )
    return value


@final
class _Outputs:
    """``func``'s results, as the batched structure ``vmap`` returns."""

    __slots__ = ("_out_dims",)

    def __init__(self, out_dims: object) -> None:
        self._out_dims = _OutDims(out_dims)

    def combine(self, results: Sequence[object], joiner: _Joiner) -> PyTree:
        """One batched result from the ``results`` of several elements (or
        chunks): ``joiner`` joins the tensors found at each leaf position
        along that leaf's ``out_dims`` axis."""
        rows: list[list[object]] = []
        spec: TreeSpec | None = None
        for result in results:
            leaves, shape = tree_flatten(result)
            if spec is not None and shape != spec:
                raise StructureError(
                    f"vmap: func returned {spec} for one batch element and {shape} for another"
                )
            spec = shape
            rows.append(leaves)
        assert spec is not None
        joined: list[object] = []
        for k, dim in enumerate(self._out_dims.entries(spec)):
            column = [row[k] for row in rows]
            if dim is None:
                joined.append(_unbatched(column))
            else:
                joined.append(joiner.join([_tensor(value) for value in column], dim))
        return spec.unflatten(joined)

    def move(self, output: object) -> PyTree:
        """``output`` of one call on the whole batch, with the batch axis of
        each tensor moved from the front to its ``out_dims`` axis.  A leaf
        whose ``out_dims`` is ``None`` is not batched and stays."""
        leaves, spec = tree_flatten(output)
        moved: list[object] = []
        for leaf, dim in zip(leaves, self._out_dims.entries(spec), strict=True):
            if dim is None:
                moved.append(leaf)
            else:
                tensor = _tensor(leaf)
                moved.append(tensor if dim == 0 else lucid.moveaxis(tensor, 0, dim))
        return spec.unflatten(moved)

    def empty(self, example: object) -> PyTree:
        """The result for a batch of no elements, from what ``func`` returned
        for one that is not there: each tensor with an axis of length 0 at
        its ``out_dims`` position."""
        leaves, spec = tree_flatten(example)
        blanks: list[object] = []
        for leaf, dim in zip(leaves, self._out_dims.entries(spec), strict=True):
            if dim is None:
                blanks.append(leaf)
                continue
            tensor = _tensor(leaf)
            shape = list(tensor.shape)
            axis = dim if dim >= 0 else dim + len(shape) + 1
            if not 0 <= axis <= len(shape):
                raise IndexError(
                    f"vmap: out_dim {dim} is out of range for an output of dimensionality "
                    f"{len(shape)}"
                )
            shape.insert(axis, 0)
            blanks.append(lucid.zeros(tuple(shape), dtype=tensor.dtype, device=tensor.device))
        return spec.unflatten(blanks)


# ── how func is run ──────────────────────────────────────────────────────────


class _Execution(ABC):
    """One value of ``strategy=``: whether ``func`` is called once per
    batch element or once on the whole batch."""

    _registry: ClassVar[dict[str, _Execution]] = {}

    @override
    def __init_subclass__(cls, *, names: tuple[str, ...] | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for name in names or ():
            _Execution._registry[name] = cls()

    @staticmethod
    def named(name: str) -> _Execution:
        """The strategy called ``name``; ``ValueError`` for any other."""
        try:
            return _Execution._registry[name]
        except KeyError:
            raise ValueError(
                f"vmap: strategy must be 'auto', 'isolated', or 'vectorized'; got {name!r}"
            ) from None

    @abstractmethod
    def isolates(self, func: object, randomness: Randomness) -> bool:
        """Whether ``func`` is called once per batch element."""


@final
class PerElement(_Execution, names=("auto", "isolated")):
    """``func`` always sees the logical per-example shapes, reductions and
    nested transforms."""

    @override
    def isolates(self, func: object, randomness: Randomness) -> bool:
        return True


@final
class WholeBatch(_Execution, names=("vectorized",)):
    """The batch axis moved to the front and ``func`` called once — except
    where that cannot be what was asked: a derivative transform has to see
    one example at a time, and independent random draws need one run each."""

    @override
    def isolates(self, func: object, randomness: Randomness) -> bool:
        return isinstance(func, Isolated) or not randomness.whole_batch


class Vmap(Transform[..., PyTree]):
    """The transform behind :func:`lucid.func.vmap`: ``func`` run over a batch axis.

    Holds ``func`` and how it was asked to be batched; calling it flattens
    the arguments, runs ``func`` on the batch (once per element, or once on
    the whole batch when ``strategy`` is ``"vectorized"``) and gives the
    results back in the structure ``func`` returned them in.
    """

    __slots__ = ("in_dims", "_outputs", "_randomness", "chunk_size", "strategy", "_execution")

    def __init__(
        self,
        func: Callable[..., object],
        in_dims: object,
        out_dims: object,
        randomness: str,
        chunk_size: int | None,
        strategy: str,
    ) -> None:
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(
                f"vmap: chunk_size should be None or greater than 0. (got {chunk_size})"
            )
        super().__init__(func)
        self.in_dims = _InDims(in_dims)
        self._outputs = _Outputs(out_dims)
        self._randomness = Randomness.named(randomness)
        self.chunk_size = chunk_size
        self.strategy = strategy
        self._execution = _Execution.named(strategy)

    @property
    def randomness(self) -> str:
        """The ``randomness`` the transform was built with."""
        return self._randomness.mode

    @override
    def __call__(self, *args: object, **kwargs: object) -> PyTree:
        batch = _Batch(args, self.in_dims)
        with _RandomnessGuard(self._randomness.mode), self._randomness.draws() as draws:
            if batch.size is None:
                return self.func(*args, **kwargs)
            if batch.size == 0:
                return self._outputs.empty(self.func(*batch.prototype(), **kwargs))
            spans = self._spans(batch.size)
            if self._execution.isolates(self.func, self._randomness):
                return self._isolated(batch, spans, kwargs, draws)
            if len(spans) > 1:
                return self._chunked(batch, spans, kwargs, draws)
            return self._whole(batch.front(), kwargs)

    def _spans(self, size: int) -> list[range]:
        """The batch indices ``chunk_size`` at a time (all of them at once
        without a ``chunk_size``)."""
        step = self.chunk_size or size
        return [range(start, min(start + step, size)) for start in range(0, size, step)]

    def _whole(self, front: tuple[object, ...], kwargs: dict[str, object]) -> PyTree:
        """``func`` on arguments batched along their front axis, the results
        with the batch axis where ``out_dims`` puts it."""
        return self._outputs.move(self.func(*front, **kwargs))

    def _chunked(
        self, batch: _Batch, spans: list[range], kwargs: dict[str, object], draws: Draws
    ) -> PyTree:
        """``func`` on ``chunk_size`` elements at a time, the results joined."""
        chunks = [draws.element(partial(self._whole, batch.chunk(span), kwargs)) for span in spans]
        return self._outputs.combine(chunks, _CONCATENATE)

    def _each(self, batch: _Batch, span: range, kwargs: dict[str, object], draws: Draws) -> PyTree:
        """The elements of ``span``, each through ``func`` on its own, stacked."""
        return self._outputs.combine(
            [draws.element(partial(self.func, *batch.element(b), **kwargs)) for b in span],
            _STACK,
        )

    def _isolated(
        self, batch: _Batch, spans: list[range], kwargs: dict[str, object], draws: Draws
    ) -> PyTree:
        """``func`` once per element, so it sees the logical per-example
        shapes and a transform inside it never sees the batch axis.

        With a ``chunk_size`` the elements of each chunk are stacked as soon
        as they are done, to free their autograd graphs before the next
        chunk starts.
        """
        parts = [self._each(batch, span, kwargs, draws) for span in spans]
        return parts[0] if len(parts) == 1 else self._outputs.combine(parts, _CONCATENATE)
