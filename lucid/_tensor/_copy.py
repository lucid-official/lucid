"""``copy.deepcopy`` for :class:`Tensor` and its subclasses.

``Tensor.__deepcopy__`` (injected from ``_methods``, which keeps ``tensor.py``
from growing) is the one owner of the rule; ``Parameter``, ``Buffer`` and a
user's subclass inherit it.  The one difference is the reference's too: a
``Parameter`` does not carry its ``.grad`` over.  Only a graph leaf can be
copied -- a tensor with a ``grad_fn`` would lose the history that makes it what
it is, so it is refused.
"""

import copy
from typing import TYPE_CHECKING

from lucid._dispatch import _impl_with_grad

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

#: ``__dict__`` key of a tensor a forward pre-hook recomputes before every
#: forward (the legacy reparametrisations'), which a module copy may therefore
#: carry as a detached value instead of refusing.
_RECOMPUTED = "_recomputed_before_forward"

#: Memo slot of the gradients already copied in this ``deepcopy`` call.  Keyed
#: by the id of this sentinel, which no real object can share.
_GRAD_MEMO_KEY = object()


class _CopiedGrads:
    """The gradients copied so far in one ``deepcopy`` call, by buffer."""

    __slots__ = ("by_buffer",)

    def __init__(self) -> None:
        self.by_buffer: dict[tuple[object, ...], Tensor] = {}


def _mark_recomputed(t: Tensor) -> None:
    """Record that a pre-hook rebuilds ``t`` before each forward.

    The reference refuses to copy a module holding such a tensor, because it is
    not a leaf.  Here a copy of the module keeps its value as a leaf and the
    hook rebuilds it on the copy's first forward, so there is nothing to refuse.
    """
    t.__dict__[_RECOMPUTED] = True


def _clone_detached[T: Tensor](t: T, requires_grad: bool) -> T:
    """An independent copy of ``t``'s values, as ``type(t)``, with no history."""
    impl = _impl_with_grad(t.data.clone()._impl, requires_grad)
    return type(t)._from_pickle_impl(impl)


def _copied_grad(t: Tensor, memo: dict[int, object]) -> Tensor | None:
    """``t.grad`` copied once per ``deepcopy`` call, however many owners share it.

    A gradient set on two tensors (``b.grad = a.grad``) is one buffer; the
    reference keeps that identity through ``memo``.  Here ``.grad`` hands out a
    fresh wrapper on every read, so the buffer itself is the key.
    """
    grad = t.grad
    if grad is None:
        return None
    seen = memo.get(id(_GRAD_MEMO_KEY))
    if not isinstance(seen, _CopiedGrads):
        seen = memo[id(_GRAD_MEMO_KEY)] = _CopiedGrads()
    impl = grad._impl
    key = (impl.storage_data_ptr(), impl.data_ptr(), tuple(impl.shape), tuple(impl.stride))
    copied = seen.by_buffer.get(key)
    if copied is None:
        copied = seen.by_buffer[key] = _clone_detached(grad, False)
    return copied


def _tensor_deepcopy[T: Tensor](self: T, memo: dict[int, object]) -> T:
    r"""Deep-copy into an independent tensor of the same type.

    The copy owns its storage, keeps ``requires_grad``, its ``.grad``
    (copied, and still one buffer if two tensors shared it) and the
    instance attributes (deep-copied through ``memo``), and is a plain
    graph leaf.  Subclasses (:class:`~lucid.nn.Parameter`,
    :class:`~lucid.nn.Buffer`, a user's) inherit this and come back as
    themselves; a :class:`~lucid.nn.Parameter` leaves ``.grad`` behind.

    Parameters
    ----------
    memo : dict
        The memo ``copy.deepcopy`` threads through one call, so a tensor
        met twice is copied once.

    Returns
    -------
    Tensor
        The copy, of ``type(self)``.

    Raises
    ------
    RuntimeError
        If ``self`` is not a graph leaf (it has a ``grad_fn``): copying
        it would drop the history that defines it.  Use
        ``t.detach().clone()`` to copy its values.  The tensor a legacy
        reparametrisation (``weight_norm``, ``spectral_norm``, pruning)
        recomputes before each forward is the one exception: it is copied
        as its value, and the hook rebuilds it.

    Examples
    --------
    >>> import copy, lucid
    >>> x = lucid.ones(2, requires_grad=True)
    >>> _ = (x * 2).sum().backward()
    >>> y = copy.deepcopy(x)
    >>> y.requires_grad, y.grad.tolist()
    (True, [2.0, 2.0])
    """
    existing = memo.get(id(self))
    if isinstance(existing, type(self)):
        return existing
    if not self.is_leaf and not self.__dict__.get(_RECOMPUTED):
        raise RuntimeError(
            "Only Tensors created explicitly by the user (graph leaves) support "
            "the deepcopy protocol at the moment.  Detach it first "
            "(tensor.detach().clone()) to copy its values."
        )
    result = _clone_detached(self, self.requires_grad)
    memo[id(self)] = result
    if not self._is_parameter:
        grad = _copied_grad(self, memo)
        if grad is not None:
            result.grad = grad
    for name, value in self.__dict__.items():
        if name != "_impl":
            result.__dict__[name] = copy.deepcopy(value, memo)
    return result
