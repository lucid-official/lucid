"""``vmap``: a function of one example, run over a batch of them.

``vmap`` takes its arguments apart as pytrees, so a dict or a list of
tensors is batched leaf by leaf, and puts the results back together in
whatever structure ``func`` returned.  :class:`_Batch` is the arguments
once flattened — the leaves, the axis each is batched along and the batch
size — and :class:`_Outputs` is the way back.
"""

from collections.abc import Callable, Sequence
from typing import Final, final, override

import lucid
from lucid._ctx import _RandomnessGuard
from lucid._tensor.tensor import Tensor
from lucid.func._function import as_tensors
from lucid.func._pytree import StructureError, TreeSpec, tree_flatten
from lucid.func._transform import Isolated, PyTree, Transform

__all__ = ["Vmap"]

_STRATEGIES: Final = ("auto", "isolated", "vectorized")


def _axis(x: Tensor, dim: int) -> int:
    """``dim`` as a non-negative axis of ``x``."""
    return dim if dim >= 0 else len(x.shape) + dim


def _stack(tensors: list[Tensor], dim: int) -> Tensor:
    return lucid.stack(tensors, dim=dim)


def _cat(tensors: list[Tensor], dim: int) -> Tensor:
    return lucid.cat(tensors, dim=dim)


def _dims_of(entries: Sequence[object], what: str) -> list[int | None]:
    """``entries`` — what a ``dims`` pytree broadcast to — as ints and ``None``."""
    dims: list[int | None] = []
    for entry in entries:
        if entry is not None and not isinstance(entry, int):
            raise TypeError(
                f"vmap: {what} must hold ints or None, got {type(entry).__name__}: {entry!r}"
            )
        dims.append(entry)
    return dims


def _per_argument(in_dims: object, n_args: int) -> object:
    """``in_dims`` as a prefix of the ``n_args`` positional arguments.

    A tuple gives one entry per argument, a shorter one padded with ``0`` —
    batch the remaining arguments along the first axis — and a longer one
    cut; anything else (an ``int``) stands for every argument.
    """
    if not isinstance(in_dims, tuple):
        return in_dims
    return in_dims[:n_args] + (0,) * (n_args - len(in_dims))


@final
class _Batch:
    """The positional arguments of one ``vmap`` call, flattened."""

    __slots__ = ("_leaves", "_spec", "_dims", "size")

    def __init__(self, args: tuple[object, ...], in_dims: object) -> None:
        self._leaves, self._spec = tree_flatten(args)
        entries = self._spec.broadcast(_per_argument(in_dims, len(args)))
        self._dims = _dims_of(entries, "in_dims")
        self.size = self._batch_size()

    def _batch_size(self) -> int | None:
        """The batch length shared by the batched leaves, or ``None`` if none is."""
        size: int | None = None
        for leaf, in_dim in zip(self._leaves, self._dims, strict=True):
            if in_dim is None:
                continue
            if not isinstance(leaf, Tensor):
                raise TypeError(
                    f"vmap: in_dim {in_dim} was given for an input of type "
                    f"{type(leaf).__name__}; only Tensors can be batched — "
                    "pass None to hand it to func unchanged"
                )
            ndim = len(leaf.shape)
            axis = _axis(leaf, in_dim)
            if not 0 <= axis < ndim:
                raise ValueError(f"vmap: in_dim {in_dim} is out of range for a {ndim}-D tensor")
            n = int(leaf.shape[axis])
            if size is not None and size != n:
                raise ValueError(f"vmap: inconsistent batch sizes {size} vs {n}")
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

    def chunk(self, start: int, stop: int) -> tuple[object, ...]:
        """The arguments of batch elements ``start`` to ``stop``, still batched."""

        def window(x: Tensor, dim: int) -> Tensor:
            where: list[slice] = [slice(None)] * len(x.shape)
            where[_axis(x, dim)] = slice(start, stop)
            return x[tuple(where)]

        return self._map(window)

    def front(self) -> tuple[object, ...]:
        """The arguments with every batch axis moved to the front."""

        def to_front(x: Tensor, dim: int) -> Tensor:
            axis = _axis(x, dim)
            return x if axis == 0 else lucid.moveaxis(x, axis, 0)

        return self._map(to_front)


@final
class _Outputs:
    """``func``'s results, as the batched structure ``vmap`` returns."""

    __slots__ = ("_out_dims",)

    def __init__(self, out_dims: object) -> None:
        self._out_dims = out_dims

    def _dims(self, spec: TreeSpec) -> list[int]:
        """The batch axis of each output leaf of structure ``spec``."""
        prefix = self._out_dims
        if spec.node is None and isinstance(prefix, tuple) and len(prefix) == 1:
            prefix = prefix[0]  # ``out_dims=(1,)`` for the one tensor ``func`` returns
        dims: list[int] = []
        for dim in _dims_of(spec.broadcast(prefix), "out_dims"):
            if dim is None:
                raise TypeError("vmap: out_dims must hold ints, got None")
            dims.append(dim)
        return dims

    def combine(
        self, results: Sequence[object], reduce: Callable[[list[Tensor], int], Tensor]
    ) -> PyTree:
        """One batched result from the ``results`` of several elements (or
        chunks): ``reduce`` joins the tensors found at each leaf position
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
        dims = self._dims(spec)
        joined = [
            reduce(
                as_tensors([row[k] for row in rows], "vmap", "func must return Tensors"), dims[k]
            )
            for k in range(spec.num_leaves)
        ]
        return spec.unflatten(joined)

    def move(self, output: object) -> PyTree:
        """``output`` of one call on the whole batch, with the batch axis of
        each tensor moved from the front to its ``out_dims`` axis.  What is
        not a tensor passes through."""
        leaves, spec = tree_flatten(output)
        moved = [
            lucid.moveaxis(leaf, 0, dim) if isinstance(leaf, Tensor) and dim != 0 else leaf
            for leaf, dim in zip(leaves, self._dims(spec), strict=True)
        ]
        return spec.unflatten(moved)


class Vmap(Transform[..., PyTree]):
    """The transform behind :func:`lucid.func.vmap`: ``func`` run over a batch axis.

    Holds ``func`` and how it was asked to be batched; calling it flattens
    the arguments, runs ``func`` on the batch (once per element, or once on
    the whole batch when ``strategy`` is ``"vectorized"``) and gives the
    results back in the structure ``func`` returned them in.
    """

    __slots__ = ("in_dims", "_outputs", "randomness", "chunk_size", "strategy")

    def __init__(
        self,
        func: Callable[..., object],
        in_dims: object,
        out_dims: object,
        randomness: str,
        chunk_size: int | None,
        strategy: str,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"vmap: strategy must be 'auto', 'isolated', or 'vectorized'; got {strategy!r}"
            )
        super().__init__(func)
        self.in_dims = in_dims
        self._outputs = _Outputs(out_dims)
        self.randomness = randomness
        self.chunk_size = chunk_size
        self.strategy = strategy

    @override
    def __call__(self, *args: object, **kwargs: object) -> PyTree:
        batch = _Batch(args, self.in_dims)
        with _RandomnessGuard(self.randomness):
            if batch.size is None:
                return self.func(*args, **kwargs)
            if batch.size > 0 and (
                self.strategy != "vectorized" or isinstance(self.func, Isolated)
            ):
                return self._isolated(batch, kwargs)
            if self.chunk_size is not None and batch.size > self.chunk_size:
                return self._chunked(batch, kwargs)
            return self._outputs.move(self.func(*batch.front(), **kwargs))

    def _chunked(self, batch: _Batch, kwargs: dict[str, object]) -> PyTree:
        """``func`` on ``chunk_size`` elements at a time, the results joined."""
        assert batch.size is not None and self.chunk_size is not None
        chunks = [
            self.func(*batch.chunk(start, min(start + self.chunk_size, batch.size)), **kwargs)
            for start in range(0, batch.size, self.chunk_size)
        ]
        return self._outputs.combine(chunks, _cat)

    def _isolated(self, batch: _Batch, kwargs: dict[str, object]) -> PyTree:
        """``func`` once per element, so it sees the logical per-example
        shapes and a transform inside it never sees the batch axis.

        With a ``chunk_size`` the elements of each chunk are stacked as soon
        as they are done, to free their autograd graphs before the next
        chunk starts.
        """
        assert batch.size is not None
        if self.chunk_size is None or batch.size <= self.chunk_size:
            return self._outputs.combine(
                [self.func(*batch.element(b), **kwargs) for b in range(batch.size)], _stack
            )
        parts = [
            self._outputs.combine(
                [
                    self.func(*batch.element(b), **kwargs)
                    for b in range(start, min(start + self.chunk_size, batch.size))
                ],
                _stack,
            )
            for start in range(0, batch.size, self.chunk_size)
        ]
        return self._outputs.combine(parts, _cat)
