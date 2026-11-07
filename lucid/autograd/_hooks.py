"""
Tensor-level gradient hooks (``Tensor.register_hook``).

A hook runs inside the backward pass, where the tensor's gradient is
complete: for a non-leaf, on its producer's output slot once every
consumer's contribution has arrived and before the producer runs; for a
leaf, just before the gradient is added into ``.grad``.  What a hook
returns is the gradient that flows on.  The engine owns where and when
(``_C_engine._tensor_hook_runner``); this module owns what a slot's hooks
are — their list, their order, their removal, and the check that each one
returned a tensor or ``None``.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING
import weakref

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_HOOK_VERSION: int = 0


def _tensor_hook_version() -> int:
    """Monotonic counter bumped whenever a tensor gradient hook is added or removed."""
    return _HOOK_VERSION


class RemovableHandle:
    r"""Handle returned by :meth:`~lucid.Tensor.register_hook`.

    A ``RemovableHandle`` keeps a reference to the list of hooks
    attached to a tensor's gradient and to the specific callable
    that was just registered, providing two ways to take the hook
    back off:

    * Imperative — call :meth:`remove` whenever the hook is no
      longer needed.
    * RAII — bind the handle in a ``with`` block; the hook is
      automatically removed when control leaves the block.

    This class is *distinct* from
    ``lucid.nn.hooks.RemovableHandle``: that one manages
    forward/backward hooks on :class:`~lucid.nn.Module`, whereas
    this one operates on the per-tensor hooks the engine runs on a
    tensor's gradient during backward.

    Parameters
    ----------
    hooks_list : list of callable
        The hook list of the tensor's gradient slot; on
        :meth:`remove` the hook is dropped from this list.
    hook : callable
        The exact callable to remove. Identity (``is``) is used
        for matching.

    Attributes
    ----------
    _hooks_list : list of callable
        The slot's hook list (private).
    _hook : callable
        The registered hook (private).

    Notes
    -----
    The hook contract is

    .. math::

        \bar x \leftarrow h(\bar x),

    where :math:`\bar x = \partial \mathcal{L} / \partial x` is
    the gradient reaching :math:`x` in the current backward pass
    and :math:`h` is the user-supplied hook. A hook returning
    ``None`` leaves :math:`\bar x` unchanged; returning a tensor
    replaces it for everything upstream of :math:`x`.

    :meth:`remove` is idempotent — calling it twice is harmless, and
    removes one registration even when the same callable was
    registered more than once.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
    >>> handle = x.register_hook(lambda g: print('grad =', g))
    >>> (x * x).sum().backward()
    grad = tensor([2., 4.])
    >>> handle.remove()

    As a context manager:

    >>> with x.register_hook(lambda g: g * 2) as h:
    ...     (x * x).sum().backward()
    >>> x.grad
    tensor([6., 12.])
    """

    def __init__(
        self, hooks_list: list[Callable[..., object]], hook: Callable[..., object]
    ) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        self._hooks_list = hooks_list
        self._hook = hook
        self._removed = False

    def remove(self) -> None:
        """Remove the hook from the tensor."""
        if self._removed:
            return
        self._removed = True
        for i, registered in enumerate(self._hooks_list):
            if registered is self._hook:
                del self._hooks_list[i]
                global _HOOK_VERSION
                _HOOK_VERSION += 1
                return

    def __enter__(self) -> RemovableHandle:
        """Enter the context.  Returns self so the value can be bound via ``with ... as``."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context, restoring any state that was modified on entry."""
        self.remove()


class _HookRunner:
    """The hooks of one gradient slot, run in registration order.

    The engine holds one per slot and calls it with the slot's gradient.
    It holds no reference to the tensor: the engine keeps it on the
    producer node (or the leaf's autograd metadata), and a reference back
    would be a cycle the collector cannot see through the engine.
    """

    __slots__ = ("hooks",)

    def __init__(self) -> None:
        self.hooks: list[Callable[..., object]] = []

    def __len__(self) -> int:
        """How many hooks the slot holds; the engine reads zero as no hook at all.

        The runner stays installed on its slot once every hook is removed, so
        ``_C_engine._has_tensor_hooks`` — the one test of whether a tensor is
        hooked — asks the runner rather than its presence.
        """
        return len(self.hooks)

    def __call__(self, grad: _C_engine.TensorImpl) -> _C_engine.TensorImpl | None:
        """Run every hook on the gradient; the last tensor returned flows on.

        Returns ``None`` when no hook replaced the gradient, so the engine
        keeps the one it handed over, with any in-place write a hook made.

        Raises
        ------
        TypeError
            A hook returned something that is neither a tensor nor ``None``.
        """
        from lucid._tensor.tensor import Tensor

        g = _wrap(grad)
        replaced = False
        # A snapshot: a hook may register or remove hooks on this slot.
        for hook in tuple(self.hooks):
            result = hook(g)
            if result is None:
                continue
            if not isinstance(result, Tensor):
                raise TypeError(
                    f"a tensor hook must return a Tensor or None, but "
                    f"{getattr(hook, '__qualname__', repr(hook))} returned "
                    f"{type(result).__name__}"
                )
            g = result
            replaced = True
        return _unwrap(g) if replaced else None


def _register_tensor_hook(
    tensor: Tensor, hook: Callable[[Tensor], Tensor | None]
) -> RemovableHandle:
    """Add *hook* to *tensor*'s gradient slot and return a removable handle.

    Called by :meth:`~lucid.Tensor.register_hook`.  The slot's runner is
    installed on the first registration and shared by every later one.
    """
    global _HOOK_VERSION
    runner = _C_engine._tensor_hook_runner(_unwrap(tensor), _HookRunner)
    runner.hooks.append(hook)
    _HOOK_VERSION += 1
    return RemovableHandle(runner.hooks, hook)


def _has_active_tensor_hooks(tensor: Tensor) -> bool:
    """Return ``True`` if *tensor* has at least one active gradient hook."""
    return _C_engine._has_tensor_hooks(_unwrap(tensor))


class _PostAccumulateGradHookRunner:
    """The post-accumulate-grad hooks of a leaf tensor, run in registration order."""

    __slots__ = ("hooks", "tensor_ref")

    def __init__(self) -> None:
        self.hooks: list[Callable[..., object]] = []
        self.tensor_ref: weakref.ReferenceType[Tensor] | None = None

    def __call__(self, tensor_impl: _C_engine.TensorImpl) -> None:
        t = self.tensor_ref() if self.tensor_ref is not None else None
        if t is None or _unwrap(t) is not tensor_impl:
            t = _wrap(tensor_impl)
        for hook in tuple(self.hooks):
            res = hook(t)
            if res is not None:
                raise RuntimeError("post accumulate grad hooks should return None.")


def _register_post_accumulate_grad_hook(
    tensor: Tensor, hook: Callable[[Tensor], None]
) -> RemovableHandle:
    """Add *hook* to *tensor*'s post-accumulate-grad hook list."""
    global _HOOK_VERSION
    runner = _C_engine._tensor_post_accumulate_grad_hook_runner(
        _unwrap(tensor), _PostAccumulateGradHookRunner
    )
    runner.tensor_ref = weakref.ref(tensor)
    runner.hooks.append(hook)
    _HOOK_VERSION += 1
    return RemovableHandle(runner.hooks, hook)


class _NodePreHookRunner:
    """The backward prehooks of a graph node, run in registration order."""

    __slots__ = ("hooks",)

    def __init__(self) -> None:
        self.hooks: list[Callable[..., object]] = []

    def __call__(self, grad_out: _C_engine.TensorImpl) -> _C_engine.TensorImpl | None:
        from lucid._tensor.tensor import Tensor

        go_tuple: tuple[Tensor | None, ...] = (_wrap(grad_out),)
        replaced = False
        for hook in tuple(self.hooks):
            res = hook(go_tuple)
            if res is None:
                continue
            if isinstance(res, Tensor):
                go_tuple = (res,)
                replaced = True
                continue
            if not isinstance(res, (tuple, list)) or len(res) != 1:
                raise TypeError(
                    "Node prehook must return None or a sequence of length 1 containing "
                    f"a Tensor or None, got {type(res).__name__}"
                )
            item = res[0]
            if item is not None and not isinstance(item, Tensor):
                raise TypeError(
                    f"Node prehook output element 0 must be a Tensor or None, "
                    f"got {type(item).__name__}"
                )
            if item is not None:
                go_tuple = (item,)
                replaced = True
        if not replaced or go_tuple[0] is None:
            return None
        return _unwrap(go_tuple[0])


def _coerce_node_hook_return(res: object, n: int) -> tuple[Tensor | None, ...]:
    """Validate and normalize the return value of a node post-hook."""
    from lucid._tensor.tensor import Tensor

    if isinstance(res, Tensor) and n == 1:
        return (res,)
    if not isinstance(res, (tuple, list)):
        raise TypeError(
            f"Node hook must return None, a tuple, or a list of gradients, got {type(res).__name__}"
        )
    if len(res) != n:
        raise RuntimeError(f"Node hook returned {len(res)} grad_inputs, but node has {n} inputs")
    new_gi: list[Tensor | None] = []
    for idx, item in enumerate(res):
        if item is not None and not isinstance(item, Tensor):
            raise TypeError(
                f"Node hook grad_input[{idx}] must be a Tensor or None, got {type(item).__name__}"
            )
        new_gi.append(item)
    return tuple(new_gi)


class _NodeHookRunner:
    """The backward post-hooks of a graph node, run in registration order."""

    __slots__ = ("hooks",)

    def __init__(self) -> None:
        self.hooks: list[Callable[..., object]] = []

    def __call__(
        self, grad_inputs: list[_C_engine.TensorImpl | None], grad_out: _C_engine.TensorImpl | None
    ) -> list[_C_engine.TensorImpl | None] | None:
        gi_tuple: tuple[Tensor | None, ...] = tuple(
            None if g is None else _wrap(g) for g in grad_inputs
        )
        go_tuple: tuple[Tensor | None, ...] = (None if grad_out is None else _wrap(grad_out),)
        replaced = False
        n = len(gi_tuple)
        for hook in tuple(self.hooks):
            res = hook(gi_tuple, go_tuple)
            if res is None:
                continue
            gi_tuple = _coerce_node_hook_return(res, n)
            replaced = True
        if not replaced:
            return None
        return [None if g is None else _unwrap(g) for g in gi_tuple]


def _register_node_prehook(
    node: _C_engine.Node,
    hook: Callable[[tuple[Tensor | None, ...]], Sequence[Tensor | None] | Tensor | None],
) -> RemovableHandle:
    """Register a backward prehook on *node*."""
    if not callable(hook):
        raise TypeError(f"register_prehook expected a callable, got {type(hook).__name__}")
    global _HOOK_VERSION
    runner = _C_engine._node_prehook_runner(node, _NodePreHookRunner)
    runner.hooks.append(hook)
    _HOOK_VERSION += 1
    return RemovableHandle(runner.hooks, hook)


def _register_node_hook(
    node: _C_engine.Node,
    hook: Callable[
        [tuple[Tensor | None, ...], tuple[Tensor | None, ...]],
        Sequence[Tensor | None] | Tensor | None,
    ],
) -> RemovableHandle:
    """Register a backward post-hook on *node*."""
    if not callable(hook):
        raise TypeError(f"register_hook expected a callable, got {type(hook).__name__}")
    global _HOOK_VERSION
    runner = _C_engine._node_hook_runner(node, _NodeHookRunner)
    runner.hooks.append(hook)
    _HOOK_VERSION += 1
    return RemovableHandle(runner.hooks, hook)
