"""
lucid.autograd.graph — utilities for inspecting and tweaking the
autograd backward graph.

Currently exposes:

  * ``allow_mutation_on_saved_tensors`` — opt-in escape hatch that
    suppresses the version-mismatch check during backward.  When the
    user knows they are mutating a saved tensor in a way that does not
    corrupt the gradient (rare, expert-only), this lets them proceed
    rather than raising ``VersionMismatch``.

  * ``saved_tensors_hooks`` — context manager that routes every tensor a
    custom :class:`~lucid.autograd.Function` saves for backward through a
    ``pack_hook`` when it is saved and an ``unpack_hook`` when ``backward``
    reads it back.  Tensors the built-in ops save are not routed through it.

  * ``save_on_cpu`` — a ``saved_tensors_hooks`` that holds what a custom
    ``Function`` saves in host memory while the graph lives, freeing the
    accelerator's at the cost of a transfer each way.
"""

import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Iterator, NamedTuple, TYPE_CHECKING

from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._device import device as _Device
    from lucid._tensor.tensor import Tensor


@contextmanager
def allow_mutation_on_saved_tensors() -> Iterator[None]:
    """Disable the autograd version-mismatch check inside the block.

    By default, mutating a tensor in-place after it has been saved for
    backward raises ``VersionMismatch`` — the saved activation no longer
    reflects the value the forward pass observed, so gradients would be
    silently wrong.  Some advanced workflows (custom in-place forward
    paths that are mathematically safe to mutate) need to bypass this
    check.

    Inside this context the engine skips the version check.  Outside,
    the previous setting is restored.  **The user takes full
    responsibility for not corrupting gradients** while the flag is on.

    Example::

        x = lucid.tensor([1.0], requires_grad=True)
        y = x * x
        # y saved x for backward.  Mutating x normally raises here:
        with lucid.autograd.graph.allow_mutation_on_saved_tensors():
            x.add_(1.0)         # ok — no VersionMismatch
            y.backward()        # gradients use the original saved x
    """
    prev = _C_engine.is_mutation_on_saved_allowed()
    _C_engine.set_mutation_on_saved_allowed(True)
    try:
        yield
    finally:
        _C_engine.set_mutation_on_saved_allowed(prev)


class _SavedHooksStack(threading.local):
    """The ``saved_tensors_hooks`` entered on this thread, innermost last."""

    def __init__(self) -> None:
        self.entered: list[saved_tensors_hooks] = []


_STACK = _SavedHooksStack()


def _current_saved_hooks() -> saved_tensors_hooks | None:
    """The innermost ``saved_tensors_hooks`` entered on this thread, or ``None``."""
    entered = _STACK.entered
    return entered[-1] if entered else None


class _Packed(NamedTuple):
    """One saved tensor after its ``pack_hook`` ran.

    Attributes
    ----------
    packed : object
        What ``pack_hook`` returned.
    unpack_hook : callable
        The hook of the context that packed it, not of whichever is innermost
        when the tensor is read back.
    """

    packed: object
    unpack_hook: Callable[[object], Tensor]

    def unpack(self) -> Tensor:
        """Run the ``unpack_hook``; it must give a tensor back.

        Raises
        ------
        TypeError
            The hook returned something that is not a tensor.
        """
        # Imported here: ``lucid._tensor`` imports this package's hooks.
        from lucid._tensor.tensor import Tensor

        out = self.unpack_hook(self.packed)
        if not isinstance(out, Tensor):
            raise TypeError(
                "Output of saved tensor unpack_hook expected to be a Tensor but "
                f"got result of type {type(out).__name__}"
            )
        return out


class _HostCopy(NamedTuple):
    """A saved tensor ``save_on_cpu`` moved to the host."""

    origin: _Device
    host: Tensor


class saved_tensors_hooks:
    r"""Route the tensors a custom ``Function`` saves through a pair of hooks.

    Inside the block, each tensor
    :meth:`~lucid.autograd.FunctionCtx.save_for_backward` is given is passed
    to ``pack_hook`` as the call is recorded, and what the hook returns is
    kept in the tensor's place.  Reading
    :attr:`~lucid.autograd.FunctionCtx.saved_tensors` in ``backward`` passes
    that object to ``unpack_hook``, which must return a tensor.

    Parameters
    ----------
    pack_hook : callable
        ``pack_hook(tensor) -> object``, run once per saved tensor.
    unpack_hook : callable
        ``unpack_hook(object) -> Tensor``, run each time the tensor is read.

    Notes
    -----
    Contexts nest: a tensor is packed by the innermost one entered when it is
    saved, and unpacked by that same context's hook later, wherever it is
    read from.  The stack is per thread.

    A tensor saved through a hook is not version-checked against writes made
    after ``forward`` — the hook, not the original, is what ``backward``
    reads.

    Coverage: only the tensors saved by custom
    :class:`~lucid.autograd.Function` calls made inside the block go through
    the hooks.  The built-in ops keep what they save inside the engine and
    are not routed through them.  A tensor ``forward`` returns and saves is
    handed to ``pack_hook`` as ``forward`` returned it, without the call's
    ``grad_fn``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import Function
    >>> from lucid.autograd.graph import saved_tensors_hooks
    >>> class Square(Function):
    ...     @staticmethod
    ...     def forward(ctx, x):
    ...         ctx.save_for_backward(x)
    ...         return x * x
    ...     @staticmethod
    ...     def backward(ctx, grad):
    ...         (x,) = ctx.saved_tensors
    ...         return 2 * x * grad
    >>> seen = []
    >>> def pack(t):
    ...     seen.append("pack")
    ...     return t
    >>> def unpack(t):
    ...     seen.append("unpack")
    ...     return t
    >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
    >>> with saved_tensors_hooks(pack, unpack):
    ...     y = Square.apply(x)
    >>> y.sum().backward()
    >>> seen
    ['pack', 'unpack']
    """

    def __init__(
        self,
        pack_hook: Callable[[Tensor], object],
        unpack_hook: Callable[[object], Tensor],
    ) -> None:
        """Remember the hooks; nothing is routed until the block is entered."""
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self) -> None:
        """Make this the innermost hook pair on the calling thread."""
        _STACK.entered.append(self)

    def __exit__(self, *exc_info: object) -> None:
        """Leave the block; the previously entered pair, if any, is innermost again."""
        _STACK.entered.pop()

    def _pack(self, tensor: Tensor) -> _Packed:
        """Run ``pack_hook`` on one saved tensor."""
        return _Packed(self.pack_hook(tensor), self.unpack_hook)


class save_on_cpu(saved_tensors_hooks):
    """Hold the tensors a custom ``Function`` saves in host memory.

    A :class:`saved_tensors_hooks` whose ``pack_hook`` copies the tensor to
    the CPU and whose ``unpack_hook`` copies it back to the device it came
    from, freeing the accelerator's memory for the life of the graph at the
    cost of a transfer each way.

    Parameters
    ----------
    pin_memory : bool, optional
        Kept for the reference signature.  The host and the accelerator
        share memory on Apple silicon, so there is no pinned staging to ask
        for and the flag changes nothing.

    Notes
    -----
    Like every :class:`saved_tensors_hooks`, it covers the tensors saved by
    custom :class:`~lucid.autograd.Function` calls only; the built-in ops'
    saved tensors stay where the engine put them.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import Function
    >>> from lucid.autograd.graph import save_on_cpu
    >>> class Cube(Function):
    ...     @staticmethod
    ...     def forward(ctx, x):
    ...         ctx.save_for_backward(x)
    ...         return x**3
    ...     @staticmethod
    ...     def backward(ctx, grad):
    ...         (x,) = ctx.saved_tensors
    ...         return 3 * x**2 * grad
    >>> x = lucid.tensor([2.0], requires_grad=True)
    >>> with save_on_cpu():
    ...     y = Cube.apply(x)
    >>> y.sum().backward()
    >>> x.grad.tolist()
    [12.0]
    """

    def __init__(self, pin_memory: bool = False) -> None:
        """Build the hook pair that moves a saved tensor to the CPU and back."""

        def to_host(tensor: Tensor) -> _HostCopy:
            return _HostCopy(tensor.device, tensor.to("cpu"))

        def to_origin(packed: object) -> Tensor:
            if not isinstance(packed, _HostCopy):
                raise TypeError(f"save_on_cpu cannot unpack a {type(packed).__name__}")
            return packed.host.to(packed.origin)

        super().__init__(to_host, to_origin)
        self.pin_memory = pin_memory


__all__ = ["allow_mutation_on_saved_tensors", "save_on_cpu", "saved_tensors_hooks"]
