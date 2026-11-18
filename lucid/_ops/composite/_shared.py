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

import lucid
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap_or_scalar, _wrap
from lucid._tensor.tensor import Tensor
from lucid._types import TensorLike


def _is_tensor(x: object) -> TypeGuard[Tensor]:
    """``TypeGuard`` for runtime-typed ``Tensor`` narrowing.

    Returns ``True`` when ``x`` is a :class:`Tensor` instance, narrowing
    its static type for callers that branch on the result.
    """
    return isinstance(x, Tensor)


def _operand_beside(x: Tensor | TensorLike, like: Tensor | None) -> Tensor:
    """``x`` as a tensor on the device of ``like``, the tensor it is paired with.

    The composite-level counterpart of :func:`lucid._dispatch._unwrap_or_scalar`
    for a composite that also takes a Python sequence.  Built by
    ``lucid.tensor(x)`` alone, a number or a list was a tensor on the default
    device, so beside a Metal tensor it met the engine as a CPU one and
    ``isin(metal_tensor, 2)`` raised ``DeviceMismatch`` (LCD-1021).

    A tensor comes back as it is.  A Python number is a 0-d constant on
    ``like``'s device, weak beside it as the reference takes it
    (``int8_tensor`` beside ``300`` stays int8).  A Python ``bool`` is the
    exception: it keeps its own dtype, so an op that refuses a bool operand
    (``isin``) still sees it.  Any other value is a sequence or an array, and
    is built by ``lucid.tensor`` on ``like``'s device.

    Parameters
    ----------
    x : Tensor or sequence or bool or int or float
        The operand.
    like : Tensor or None
        The operand it is paired with, when that is a tensor; ``None`` leaves
        ``x`` on the default device, as ``lucid.tensor`` does.

    Returns
    -------
    Tensor
        ``x`` as a tensor.
    """
    if _is_tensor(x):
        return x
    if like is None:
        return lucid.tensor(x)
    number = _promotion.as_number(x)
    if isinstance(number, _promotion.NUMBER) and not isinstance(number, bool):
        return _wrap(_unwrap_or_scalar(number, like._impl))
    return lucid.tensor(x, device=like.device)


def _operands_together(*operands: Tensor | TensorLike) -> tuple[Tensor, ...]:
    """``operands`` as tensors on one device, the first tensor's.

    For a composite that compares or joins its operands and so wants a Python
    number or list among them to meet the engine where the tensors are, not on
    the default device (:func:`_operand_beside`).  With no tensor among them
    every operand is built on the default device.

    Parameters
    ----------
    *operands : Tensor or sequence or bool or int or float
        The operands, in the order the caller wants them back.

    Returns
    -------
    tuple of Tensor
        One tensor per operand.
    """
    like = next((x for x in operands if _is_tensor(x)), None)
    return tuple(_operand_beside(x, like) for x in operands)


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
