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
file.
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from lucid._factories.creation import zeros_like
from lucid._tensor.tensor import Tensor
from lucid.autograd._grad_mode import enable_grad, no_grad


def _held_apart(t: Tensor) -> bool:
    """Whether ``t``'s gradient lives outside the slot the optimizers read.

    A gradient from ``backward(create_graph=True)``, or a tensor that
    requires grad assigned to ``.grad``, is kept apart from the plain
    gradient slot.  The engine's optimizers read only that slot (LCD-622).
    A first-order backward adds into such a gradient out of place, making a
    new tensor.  So a write that should reach the next step has to go into
    the slot, as a new gradient.
    """
    return t._impl.grad_as_impl() is not None


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
                # ``Tensor.detach_`` rebinds only its wrapper and leaves the
                # gradient in the graph (LCD-621).  A copy recorded under grad
                # mode moves the tensor to its source's place in the graph,
                # and zeros have none, so this one write detaches and zeroes
                # the gradient itself.
                with enable_grad():
                    g.copy_(zeros_like(g))
            else:
                if g.requires_grad:
                    g.requires_grad_(False)
                g.zero_()
            if _held_apart(t):
                # The held tensor has been reset above.  From here on
                # ``t.grad`` is a new zero tensor in the slot.
                t.grad = zeros_like(g)


def write_grad_(t: Tensor, value: Tensor) -> None:
    """Write ``value`` into ``t``'s gradient, in place where it can.

    A gradient held apart from the optimizers' slot (:func:`_held_apart`)
    gets ``value`` as a new gradient instead, so the next step reads it.

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
    if _held_apart(t):
        t.grad = value if value.dtype == g.dtype else value.to(g.dtype)
        return
    with no_grad():
        g.copy_(value)


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
