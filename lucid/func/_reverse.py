"""Reverse-mode transforms: ``grad``, ``vjp``, ``jacrev`` and ``hessian``.

All four differentiate with the reverse passes of :mod:`lucid.autograd`, and
all four decide the same thing the same way before they do: whether the
result has to stay attached to a graph (:func:`~lucid.func._function.keeps_graph`).
That one rule is what lets them nest — the derivative inside is an
intermediate of the derivative outside, and has to carry its graph out.
"""

from collections.abc import Callable
from typing import ClassVar, override

from lucid._tensor.tensor import Tensor
from lucid.autograd._backward import grad as _autograd_grad
from lucid.autograd._functional import hessian as _hessian, jacobian as _jacobian
from lucid.autograd._grad_mode import enable_grad
from lucid.func._function import FlatFunction, LossFunction, Selection, keeps_graph, pack_jacobian
from lucid.func._pytree import StructureError, TreeSpec
from lucid.func._transform import Isolated, PyTree, Transform

__all__ = ["Grad", "GradAndValue", "Hessian", "Jacrev", "Pullback", "vjp_at"]


class _GradBase[**P, R](Transform[P, R]):
    """What :class:`Grad` and :class:`GradAndValue` share: differentiating a
    scalar ``func`` once, with the graph kept when it has to be."""

    __slots__ = ("argnums", "has_aux")
    name: ClassVar[str]

    def __init__(
        self, func: Callable[P, object], argnums: int | tuple[int, ...], has_aux: bool
    ) -> None:
        super().__init__(func)
        self.argnums = argnums
        self.has_aux = has_aux

    def _evaluate(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> tuple[PyTree, Tensor, object]:
        """``(gradient, loss, aux)`` of one call."""
        selection = Selection(args, self.argnums, self.name)
        graph = keeps_graph(selection.leaves)
        tracked = selection.tracked()
        loss_fn = LossFunction(
            self.func, args, kwargs, selection, has_aux=self.has_aux, name=self.name
        )
        with enable_grad():
            loss = loss_fn(*tracked)
        grads = (
            _autograd_grad(
                [loss], tracked, retain_graph=graph, create_graph=graph, allow_unused=True
            )
            if tracked
            else ()
        )
        return selection.spec.unflatten(grads), (loss if graph else loss.detach()), loss_fn.aux


class Grad[**P](_GradBase[P, PyTree]):
    """The transform behind :func:`lucid.func.grad`: the gradient of a scalar ``func``.

    Differentiates the selected arguments (each a tensor or a pytree of them)
    with one reverse pass and returns the gradient in their structure.
    """

    __slots__ = ()
    name: ClassVar[str] = "grad"

    @override
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> PyTree:
        grads, _, aux = self._evaluate(args, kwargs)
        return (grads, aux) if self.has_aux else grads


class GradAndValue[**P](_GradBase[P, tuple[PyTree, Tensor | tuple[Tensor, object]]]):
    """The transform behind :func:`lucid.func.grad_and_value`: gradient and loss.

    The same pass as :class:`Grad`, returning the loss it evaluated beside
    the gradient instead of making the caller evaluate ``func`` twice.
    """

    __slots__ = ()
    name: ClassVar[str] = "grad_and_value"

    @override
    def __call__(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> tuple[PyTree, Tensor | tuple[Tensor, object]]:
        grads, value, aux = self._evaluate(args, kwargs)
        return (grads, (value, aux)) if self.has_aux else (grads, value)


class Pullback(Isolated):
    """The ``vjp_fn`` that :func:`vjp_at` returns: ``v ↦ vᵀ J`` at the primals.

    The cotangent has the structure of the output.  For convenience the
    output's leaves may also be given one by one as positional arguments —
    the form every earlier release took — and, for a scalar output, left out.
    """

    __slots__ = ("_outputs", "_out_spec", "_tracked", "_selection", "_graph")

    def __init__(
        self,
        outputs: tuple[Tensor, ...],
        out_spec: TreeSpec,
        tracked: list[Tensor],
        selection: Selection,
        graph: bool,
    ) -> None:
        self._outputs = outputs
        self._out_spec = out_spec
        self._tracked = tracked
        self._selection = selection
        self._graph = graph

    def _seeds(self, cotangents: tuple[object, ...]) -> list[Tensor | None]:
        if not cotangents:
            return [None] * len(self._outputs)
        structure = cotangents[0] if len(cotangents) == 1 else cotangents
        try:
            leaves = self._out_spec.match(structure)
        except StructureError:
            if len(cotangents) != self._out_spec.num_leaves:
                raise
            leaves = list(cotangents)
        seeds: list[Tensor | None] = []
        for leaf in leaves:
            if leaf is not None and not isinstance(leaf, Tensor):
                raise TypeError(
                    f"lucid.func.vjp: a cotangent must be a Tensor, got {type(leaf).__name__}"
                )
            seeds.append(leaf)
        return seeds

    def __call__(self, *cotangents: object) -> PyTree:
        grads = _autograd_grad(
            list(self._outputs),
            self._tracked,
            grad_outputs=self._seeds(cotangents),
            retain_graph=True,
            create_graph=self._graph,
            allow_unused=True,
        )
        return self._selection.spec.unflatten(grads)


def vjp_at(
    func: Callable[..., object], primals: tuple[object, ...], has_aux: bool
) -> tuple[PyTree, Pullback]:
    """``(output, pullback)`` — or ``((output, aux), pullback)`` — of ``func``
    at ``primals``."""
    selection = Selection(primals, tuple(range(len(primals))), "vjp")
    graph = keeps_graph(selection.leaves)
    tracked = selection.tracked()
    flat = FlatFunction(func, primals, {}, selection, has_aux=has_aux, name="vjp")
    with enable_grad():
        outputs = flat(*tracked)
    pullback = Pullback(outputs, flat.out_spec, tracked, selection, graph)
    result = flat.out_spec.unflatten(outputs)
    return ((result, flat.aux) if has_aux else result), pullback


def _rows(blocks: Tensor | tuple[Tensor | tuple[Tensor, ...], ...]) -> list[tuple[Tensor, ...]]:
    """The nested blocks :mod:`lucid.autograd` returns for a flat tuple of
    inputs, as one row of blocks per output."""
    assert isinstance(blocks, tuple)
    rows: list[tuple[Tensor, ...]] = []
    for row in blocks:
        assert isinstance(row, tuple)
        rows.append(row)
    return rows


class Jacrev[**P](Transform[P, PyTree]):
    """``jacrev(func, argnums, has_aux, chunk_size=)``: the Jacobian of
    ``func``, a row at a time by reverse passes."""

    __slots__ = ("argnums", "has_aux")

    def __init__(
        self,
        func: Callable[P, object],
        argnums: int | tuple[int, ...],
        has_aux: bool,
        chunk_size: int | None,
    ) -> None:
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("jacrev: chunk_size must be positive")
        super().__init__(func)
        self.argnums = argnums
        self.has_aux = has_aux

    @override
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> PyTree:
        selection = Selection(args, self.argnums, "jacrev")
        flat = FlatFunction(self.func, args, kwargs, selection, has_aux=self.has_aux, name="jacrev")
        blocks = _jacobian(flat, selection.leaves, create_graph=keeps_graph(selection.leaves))
        packed = pack_jacobian(flat.out_spec, selection.spec, _rows(blocks))
        return (packed, flat.aux) if self.has_aux else packed


class Hessian[**P](Transform[P, PyTree]):
    """``hessian(func, argnums)``: the second derivatives of a scalar ``func``.

    The Jacobian of the gradient — the reverse-over-reverse the autograd
    ``hessian`` takes — which is the value ``jacfwd(jacrev(func))`` has and
    costs ``n`` reverse passes through the gradient's graph rather than
    ``n`` forward ones through it.
    """

    __slots__ = ("argnums",)

    def __init__(self, func: Callable[P, object], argnums: int | tuple[int, ...]) -> None:
        super().__init__(func)
        self.argnums = argnums

    @override
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> PyTree:
        selection = Selection(args, self.argnums, "hessian")
        loss_fn = LossFunction(self.func, args, kwargs, selection, has_aux=False, name="hessian")
        blocks = _hessian(loss_fn, selection.leaves, create_graph=keeps_graph(selection.leaves))
        return pack_jacobian(selection.spec, selection.spec, _rows(blocks))
