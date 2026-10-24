"""Writes that change a gradient write into it, so whoever holds it sees them.

``t.grad`` reads the gradient's own buffer, and the reference framework's
``.grad`` is one tensor that all its holders share: a hook, a
gradient-accumulation buffer, a training loop that kept ``g = p.grad``.  Code
that changes a gradient therefore changes that tensor.  Assigning
``t.grad = new`` instead installs a second tensor and leaves every holder
reading the old values.  That is what ``zero_grad(set_to_none=False)``,
``clip_grad_norm_``, ``clip_grad_value_`` and ``GradScaler.unscale_`` did.
All of them write through here now.
"""

from collections.abc import Iterable

from lucid._factories.creation import zeros_like
from lucid._tensor.tensor import Tensor
from lucid.autograd._grad_mode import enable_grad, no_grad


def zero_grads_(tensors: Iterable[Tensor], *, set_to_none: bool) -> None:
    """Clear the gradient of each of ``tensors``, as ``zero_grad`` does.

    Parameters
    ----------
    tensors : Iterable[Tensor]
        The tensors whose gradients are cleared.  Those without one are
        skipped.
    set_to_none : bool
        ``True`` drops each gradient.  ``False`` zeroes each one in place
        and takes it out of the graph, the way the reference's
        ``zero_grad`` takes it out: a gradient with a ``grad_fn`` (from
        ``backward(create_graph=True)``) is detached, and one that requires
        grad stops requiring it.
    """
    with no_grad():
        for t in tensors:
            g = t.grad
            if g is None:
                continue
            if set_to_none:
                t.grad = None
                continue
            if g.grad_fn is not None:
                # ``detach_`` would rebind only this wrapper and leave the
                # gradient in the graph.  A copy recorded under grad mode puts
                # the tensor where its source is in the graph, and zeros are
                # nowhere, so this write detaches and zeroes the gradient
                # itself.
                with enable_grad():
                    g.copy_(zeros_like(g))
            else:
                if g.requires_grad:
                    g.requires_grad_(False)
                g.zero_()
            if t._impl.grad_as_impl() is not None:
                # A gradient that has a graph, or that was assigned a tensor
                # requiring grad, is kept apart from the plain gradient slot.
                # The engine's optimizers and first-order accumulation read
                # only that slot, so zeros go there as well.  The held tensor
                # has been reset above, but from here on ``t.grad`` is a
                # different tensor.
                t.grad = zeros_like(g)


def write_grad_(t: Tensor, value: Tensor) -> None:
    """Write ``value`` into ``t``'s gradient in place.

    Parameters
    ----------
    t : Tensor
        The tensor whose gradient changes.
    value : Tensor
        The new values, of the gradient's shape.  The gradient keeps its
        dtype: ``value`` is cast to it, so a result computed in a wider
        dtype is rounded into the gradient once.

    Raises
    ------
    RuntimeError
        If ``t`` has no gradient to write into.
    """
    g = t.grad
    if g is None:
        raise RuntimeError("write_grad_: the tensor has no gradient to write into")
    with no_grad():
        g.copy_(value)
