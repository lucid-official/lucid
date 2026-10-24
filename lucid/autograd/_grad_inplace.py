"""Every write that changes a gradient, and why most land in place.

``t.grad`` reads the gradient's own buffer, and the reference framework's
``.grad`` is one tensor that all its holders share: a hook, a
gradient-accumulation buffer, a training loop that kept ``g = p.grad``.  Code
that changes a gradient therefore changes that tensor.  Assigning
``t.grad = new`` instead installs a second tensor and leaves every holder
reading the old values.  That is what ``zero_grad(set_to_none=False)``,
``clip_grad_norm_``, ``clip_grad_value_`` and ``GradScaler.unscale_`` did.

Two kinds of gradient still get a new tensor (LCD-622):

* a gradient held apart from the plain slot (see :func:`_held_apart`);
* the negation a ``maximize`` step installs.

Both are spelled here, so the change that brings them in place lands in one
file.  The writes go to the engine tensors directly: through
``Tensor.copy_`` each one cost some twenty times the old assignment.
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid.autograd._grad_mode import no_grad


def _held_apart(impl: _C_engine.TensorImpl) -> bool:
    """Whether ``impl``'s gradient lives outside the slot the optimizers read.

    Two kinds of gradient are kept apart from the plain gradient slot:

    * one from ``backward(create_graph=True)``;
    * a tensor that requires grad, assigned to ``.grad``.

    The engine's optimizers read only that slot (LCD-622).  A first-order
    backward adds into such a gradient out of place, making a new tensor.
    So a write that should reach the next step goes into the slot, as a new
    gradient.

    Nothing writes into the tensor held apart, either.  The engine keeps the
    first graph-mode contribution as it comes, so that tensor can be another
    parameter's gradient, the ``gradient=`` given to ``backward``, or a
    Parameter.  The reference clones it.
    """
    return impl.grad_as_impl() is not None


def _zeros(like: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    return _C_engine.full(list(like.shape), 0.0, like.dtype, like.device)


def _assign(grad: _C_engine.TensorImpl, value: _C_engine.TensorImpl, name: str) -> None:
    """Write ``value`` into ``grad``'s buffer: ``Tensor.copy_``'s write, by itself.

    A gradient in the slot never requires grad nor has a graph, so the write
    records nothing under grad mode either.  ``value`` has ``grad``'s shape,
    dtype and device.
    """
    if grad.is_aliased():
        _C_engine.assign_inplace(grad, value, name)
    else:
        grad.assign_from(value, name)


def zero_grads_(tensors: Iterable[Tensor], *, set_to_none: bool) -> None:
    """Clear the gradient of each of ``tensors``, as ``zero_grad`` does.

    Parameters
    ----------
    tensors : Iterable[Tensor]
        The tensors whose gradients are cleared.  Those without one are
        skipped.
    set_to_none : bool
        ``True`` drops each gradient.  ``False`` zeroes each one in place.
        A gradient held apart (:func:`_held_apart`) is replaced by zeros
        instead.
    """
    for t in tensors:
        impl = t._impl
        if set_to_none:
            impl.zero_grad()
        elif _held_apart(impl):
            impl.set_grad(_zeros(impl))
        elif (g := impl.grad_to_tensor()) is not None:
            _assign(g, _zeros(g), "zero_grad")


def write_grad_(t: Tensor, value: _C_engine.TensorImpl, name: str) -> None:
    """Write ``value`` into ``t``'s gradient, in place where it can.

    A gradient held apart from the optimizers' slot (:func:`_held_apart`)
    gets ``value`` as a new gradient instead, so the next step reads it.

    Parameters
    ----------
    t : Tensor
        The tensor whose gradient changes.
    value : lucid._C.engine.TensorImpl
        The new values, of the gradient's shape, dtype and device.  A result
        computed in a wider dtype is cast by the caller, which knows whether
        it was; reading a dtype costs more than the write itself.
    name : str
        The public operation writing, for the engine's refusals.

    Raises
    ------
    RuntimeError
        If ``t`` has no gradient to write into.
    """
    impl = t._impl
    if _held_apart(impl):
        impl.set_grad(value)
        return
    g = impl.grad_to_tensor()
    if g is None:
        raise RuntimeError(f"{name}: the tensor has no gradient to write into")
    _assign(g, value, name)


@contextmanager
def negated_grads(tensors: Iterable[Tensor]) -> Iterator[None]:
    """Hold each non-empty gradient of ``tensors`` negated for the block.

    The negation is installed as a new gradient, and the original goes back
    afterwards, even when the block raises.  A held ``p.grad`` is therefore
    not the gradient after the block (LCD-622).  Negating in place is refused
    for a CPU gradient that another gradient shares, and it would not reach
    a gradient held apart from the optimizers' slot.

    Parameters
    ----------
    tensors : Iterable[Tensor]
        The tensors whose gradients the block sees negated.
    """
    saved: list[tuple[Tensor, Tensor]] = []
    with no_grad():
        for t in tensors:
            g = t.grad
            if g is not None and g.numel() > 0:
                saved.append((t, g))
                t.grad = -g
    try:
        yield
    finally:
        for t, g in saved:
            t.grad = g
