"""Helpers shared across the ``lucid._ops.composite`` submodules.

Composite ops are pure-Python — they layer on top of the registered
engine primitives via the ``lucid`` top-level namespace and Tensor
operator overloads.

Conventions used by every composite:

* prefer **operator overloads** (``+ - * / == < >``) — they go through
  ``Tensor.__add__`` etc., broadcast, and accept Python scalars;
* the free-function forms in ``lucid`` (``lucid.add``, ``lucid.sub``, ...)
  are thin engine wrappers and reject Python scalars — only use them
  when both operands are already tensors;
* for axis swaps, use :func:`_swap_dims` — the engine's ``transpose``
  with no dims reverses *all* of them, not just two.
"""

from typing import TypeGuard

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor


def _is_tensor(x: object) -> TypeGuard[Tensor]:
    """``TypeGuard`` for runtime-typed ``Tensor`` narrowing.

    Returns ``True`` when ``x`` is a :class:`Tensor` instance, narrowing
    its static type for callers that branch on the result.
    """
    return isinstance(x, Tensor)


def _swap_dims(t: Tensor, d0: int, d1: int) -> Tensor:
    """Swap exactly two dims, through the same op as ``Tensor.swapaxes``.

    A permutation built here from the raw rank had no slot for a 0-d
    tensor's notional axis (``list index out of range``), and let the free
    function and the method answer differently.  The engine's ``swapaxes``
    resolves both dims by its ``wrap_dim`` rule.
    """
    return t.swapaxes(d0, d1)


def _binary_device(op: str, a: Tensor, b: Tensor) -> _C_engine.Device:
    """The device a binary op of ``a`` and ``b`` answers on, as the engine picks it.

    For a composite that computes off the engine — a host loop over the
    values — and so never meets the engine's own device check.  Two
    operands on one device answer there; a 0-d CPU tensor is a scalar and
    follows the other operand's device; any other pair on two devices is
    refused, as the engine refuses it.

    Raises
    ------
    lucid._C.engine.DeviceMismatch
        If the operands are on two devices and neither is a 0-d CPU tensor.
    """
    da, db = a._impl.device, b._impl.device
    if da == db:
        return da
    cpu = _C_engine.Device.CPU
    if b.ndim == 0 and db == cpu:
        return da
    if a.ndim == 0 and da == cpu:
        return db
    raise _C_engine.DeviceMismatch(
        f"{op}: expected both operands on one device, got {a.device.type} and "
        f"{b.device.type} — only a 0-d cpu tensor follows the other operand's"
    )
