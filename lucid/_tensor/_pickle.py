"""Pickle support for :class:`Tensor` and its subclasses.

The wire format mirrors the ``lucid.serialization`` v3 contract (raw bytes plus
shape, dtype name and device), so ``multiprocessing`` under ``spawn`` moves
tensors without NumPy.  A plain ``Tensor`` rebuilds through
:func:`_tensor_unpickle`; any subclass (``Parameter``, ``Buffer``, a user's)
rebuilds as itself through :func:`_subclass_unpickle`, which asks the class's
``_from_pickle_impl`` for the instance and lets pickle restore its ``__dict__``.
Both live here so that ``tensor.py`` carries only the two-line ``__reduce__``.
"""

from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _reduce_tensor(t: Tensor) -> tuple[object, ...]:
    """The ``__reduce__`` result for ``t``: how to rebuild it, as its own type."""
    from lucid._tensor.tensor import Tensor

    impl = t._impl
    wire = (impl.to_bytes(), list(impl.shape), t.dtype._name, impl.device, impl.requires_grad)
    cls = type(t)
    if cls is Tensor:
        return (_tensor_unpickle, wire)
    state = {k: v for k, v in t.__dict__.items() if k != "_impl"}
    return (_subclass_unpickle, (cls, *wire), state)


def _rebuild_impl(
    raw_bytes: bytes,
    shape: list[int],
    dtype_name: str,
    device: _C_engine.Device,
    requires_grad: bool,
) -> _C_engine.TensorImpl:
    from lucid._dtype import _resolve_dtype_name, to_engine_dtype

    eng_dtype = to_engine_dtype(_resolve_dtype_name(dtype_name))
    return _C_engine.TensorImpl.from_bytes(raw_bytes, list(shape), eng_dtype, device, requires_grad)


def _tensor_unpickle(
    raw_bytes: bytes,
    shape: list[int],
    dtype_name: str,
    device: _C_engine.Device,
    requires_grad: bool,
) -> Tensor:
    """Top-level helper so multiprocessing (spawn) can pickle/unpickle Tensor."""
    from lucid._tensor.tensor import Tensor

    impl = _rebuild_impl(raw_bytes, shape, dtype_name, device, requires_grad)
    return Tensor.__new_from_impl__(impl)


def _subclass_unpickle(
    cls: type[Tensor],
    raw_bytes: bytes,
    shape: list[int],
    dtype_name: str,
    device: _C_engine.Device,
    requires_grad: bool,
) -> Tensor:
    """Unpickle a :class:`Tensor` subclass as itself (``Parameter``, ``Buffer``, ...)."""
    return cls._from_pickle_impl(_rebuild_impl(raw_bytes, shape, dtype_name, device, requires_grad))
