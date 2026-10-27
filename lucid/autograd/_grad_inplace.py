"""Every write that changes a gradient, and why it lands in place.

``t.grad`` reads the gradient's own buffer, and the reference framework's
``.grad`` is one tensor that all its holders share: a hook, a
gradient-accumulation buffer, a training loop that kept ``g = p.grad``.  Code
that changes a gradient therefore changes that tensor.  Assigning
``t.grad = new`` instead installs a second tensor and leaves every holder
reading the old values.  That is what ``zero_grad(set_to_none=False)``,
``clip_grad_norm_``, ``clip_grad_value_``, ``GradScaler.unscale_`` and a
``maximize`` step did.

The writes go to the engine tensors directly: through ``Tensor.copy_`` each
one cost some twenty times the old assignment.
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import math

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid.autograd._grad_mode import enable_grad, no_grad


def _aliased_tracked(g: _C_engine.TensorImpl) -> bool:
    """Whether ``g`` is a view of a tensor that participates in autograd."""
    return g.is_aliased() and (g.requires_grad or g.grad_fn is not None)


def _zeros(like: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    return _C_engine.full(list(like.shape), 0.0, like.dtype, like.device)


def _assign(grad: _C_engine.TensorImpl, value: _C_engine.TensorImpl, name: str) -> None:
    """Write ``value`` into ``grad``'s buffer: ``Tensor.copy_``'s write, by itself.

    When ``value`` carries no graph, a graph-mode or leaf ``grad`` is detached
    in place as it is written.  ``value`` has ``grad``'s shape, dtype and
    device.
    """
    if grad.requires_grad and grad.grad_fn is None:
        grad._set_requires_grad(False)
    if not value.requires_grad and value.grad_fn is None and grad.grad_fn is not None:
        with enable_grad():
            grad.assign_from(value, name)
        return
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
        A gradient assigned as a view of a leaf that requires grad is replaced
        by zeros instead so the leaf's buffer is left alone.
    """
    for t in tensors:
        impl = t._impl
        if set_to_none:
            impl.zero_grad()
        elif (g := impl.grad_to_tensor()) is not None:
            if _aliased_tracked(g):
                impl.set_grad(_zeros(impl))
            else:
                _assign(g, _zeros(g), "zero_grad")


def write_grad_(t: Tensor, value: _C_engine.TensorImpl, name: str) -> None:
    """Write ``value`` into ``t``'s gradient in place.

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
    g = impl.grad_to_tensor()
    if g is None:
        raise RuntimeError(f"{name}: the tensor has no gradient to write into")
    if _aliased_tracked(g):
        impl.set_grad(value)
        return
    _assign(g, value, name)


def _negate_inplace(g: _C_engine.TensorImpl) -> None:
    neg = _C_engine.neg(g)
    if g.is_aliased():
        _C_engine.assign_inplace(g, neg, "maximize")
    else:
        g.assign_from(neg, "maximize")


@contextmanager
def negated_grads(tensors: Iterable[Tensor]) -> Iterator[None]:
    """Hold each non-empty gradient of ``tensors`` negated in place for the block.

    Each gradient is negated in place on entry and negated back on exit, even
    when the block raises, so a held ``p.grad`` stays the gradient after the
    block.

    Parameters
    ----------
    tensors : Iterable[Tensor]
        The tensors whose gradients the block sees negated.
    """
    grads: list[_C_engine.TensorImpl] = []
    seen: set[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = set()
    with no_grad():
        for t in tensors:
            g = t._impl.grad_to_tensor()
            if g is not None and math.prod(g.shape) > 0:
                key = (
                    g.family_id() or g.storage_data_ptr() or id(g),
                    g.storage_offset_bytes(),
                    tuple(g.shape),
                    tuple(g.stride),
                )
                if key in seen:
                    continue
                seen.add(key)
                grads.append(g)
                _negate_inplace(g)
    try:
        yield
    finally:
        with no_grad():
            for g in grads:
                _negate_inplace(g)
