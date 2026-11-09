"""The rescaling the norms of :mod:`lucid.linalg` share.

``vector_norm`` is the one place a norm reduces entries, ``matrix_norm``'s
entry-wise orders and ``norm``'s dispatch go through it, so the guard against
a power leaving the floating-point range lives here and nowhere else
(:func:`_scaled_p_norm`); the singular-value orders have their own
(:func:`_scaled_by_max`).
"""

import math
from collections.abc import Callable

import lucid
from lucid._C import engine as _C_engine
from lucid._ops._reduce import _p_norm
from lucid._tensor.tensor import Tensor

_Impl = _C_engine.TensorImpl

#: Largest |exponent| of a power of two the reduction may form: ``|x|^p`` is
#: kept inside ``2^±E`` so a sum of up to ``2^20`` of them still fits.  float32
#: stops at ``2^±126`` (and Metal flushes the subnormals below that to zero),
#: float64 at ``2^±1022``.
_RANGE_EXPONENT = {_C_engine.F32: 100.0, _C_engine.F64: 900.0}
_HALF = frozenset({_C_engine.F16, _C_engine.BF16})


def _scaled_p_norm(r: _Impl, p: float, axes: list[int], keepdim: bool) -> _Impl:
    """``(sum |r|^p)^(1/p)`` over ``axes`` for a real ``r``, safe from underflow.

    The norm is absolutely homogeneous, so each slice's largest magnitude can
    be divided out first and multiplied back: every ratio lands in [0, 1] and
    nothing powers out of range.  ``norm([1e-20, 1e-20])`` in float32 is
    2e-20, an ordinary number, and the direct evaluation squares to 1e-40 and
    answers 0 on Metal.

    The rescale costs several passes over the input, and only a slice whose
    magnitude is outside the window where ``|x|^p`` is representable needs it.
    On the CPU the reduction therefore runs plain first and its answers are
    read: when every one is inside the window the plain result stands, bit for
    bit what it was before the rescale existed, and only an underflowed,
    overflowed or all-zero slice has the call redone rescaled.  The read is one
    boolean of the *result*, and costs nothing there — the CPU is eager.  On
    Metal the same read would be a device sync per norm (about 200 us, against
    a norm that is queued in a few), so there the rescale is recorded
    unconditionally, with the divisor 1 for any slice that is inside the window
    — the same bits, no sync.  A compile trace or a phantom tensor has no value
    to read either, and records the rescaled form, which is right for any later
    input.

    The divisor is a constant of the call — computed without the graph — so
    the gradient is the plain norm's, and an all-zero, ``nan`` or infinite
    slice takes the unscaled evaluation, which is right there already.  A
    half input reduces in float32, as :func:`_p_norm` does.
    """
    if r.dtype in _HALF:
        out = _scaled_p_norm(_C_engine.astype(r, _C_engine.F32), p, axes, keepdim)
        return _C_engine.astype(out, r.dtype)
    # ``p == 0`` counts entries and ``±inf`` picks one: neither powers anything.
    # An empty tensor has no largest magnitude (``max`` refuses it).
    if p == 0 or math.isinf(p) or r.numel() == 0:
        return _p_norm(r, p, axes, keepdim)
    exponent = _RANGE_EXPONENT[r.dtype] / max(abs(p), 1.0)
    if not _readable(r):
        return _rescaled_p_norm(r, p, axes, keepdim, exponent)
    out = _p_norm(r, p, axes, keepdim)
    with lucid.no_grad():
        outside = _C_engine.any(
            _C_engine.logical_or(
                _C_engine.less(out, _C_engine.full_like(out, 2.0**-exponent)),
                _C_engine.greater(out, _C_engine.full_like(out, 2.0**exponent)),
            )
        )
        if not outside.item():
            return out
    return _rescaled_p_norm(r, p, axes, keepdim, exponent)


def _readable(r: _Impl) -> bool:
    """Whether reading a value of ``r``'s reduction is free: an eager CPU tensor."""
    return (
        r.device == _C_engine.Device.CPU
        and not r.is_phantom
        and _C_engine.compile.current_tracer() is None
    )


def _rescaled_p_norm(r: _Impl, p: float, axes: list[int], keepdim: bool, exponent: float) -> _Impl:
    """:func:`_p_norm` of ``r`` over ratios to each slice's largest magnitude.

    A slice's magnitude is divided out only when it is outside ``2^±exponent``
    and finite and positive; every other slice has divisor 1.
    """
    with lucid.no_grad():
        top = _C_engine.max(_C_engine.abs(r), axes, True)
        outside = _C_engine.logical_or(
            _C_engine.less(top, _C_engine.full_like(top, 2.0**-exponent)),
            _C_engine.greater(top, _C_engine.full_like(top, 2.0**exponent)),
        )
        usable = _C_engine.logical_and(
            _C_engine.logical_and(
                _C_engine.isfinite(top), _C_engine.greater(top, _C_engine.zeros_like(top))
            ),
            outside,
        )
        divisor = _C_engine.where(usable, top, _C_engine.ones_like(top))
    out = _p_norm(_C_engine.div(r, divisor), p, axes, keepdim)
    if not keepdim:
        divisor = _C_engine.reshape(divisor, list(out.shape))
    return _C_engine.mul(out, divisor)


def _scaled_by_max(x: Tensor, compute: Callable[[Tensor], Tensor]) -> Tensor:
    """Run ``compute`` on ``x`` divided by its largest magnitude, then undo it.

    The singular-value orders of :func:`matrix_norm` are absolutely
    homogeneous — ``‖αx‖ = |α|‖x‖`` — and an SVD of entries near the float
    range's edges loses them to overflow or underflow, so they are taken on
    ratios in [0, 1].  The entry-wise orders rescale per slice, and only where
    a power would leave the range, in :func:`_scaled_p_norm`.

    The scale is only used when it is finite and positive.  An all-zero
    input has scale 0, and an input containing an infinity has scale inf —
    dividing by either turns the answer into NaN, which is how the first
    version of this rescaling reported ``norm([inf, 1])``.  Both fall back
    to 1, where the unscaled evaluation is already right.
    """
    # Nothing to rescale, and no largest magnitude to find: ``max`` of an
    # empty tensor is refused, as it has no identity.
    if x.numel() == 0:
        return compute(x)
    scale = lucid.max(lucid.abs(x))
    usable = lucid.isfinite(scale) & (scale > lucid.zeros_like(scale))
    safe = lucid.where(usable, scale, lucid.ones_like(scale))
    return compute(x / safe) * safe
