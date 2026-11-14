"""The rescaling the norms of :mod:`lucid.linalg` share.

``vector_norm`` is the one place a norm reduces entries, and ``matrix_norm``'s
entry-wise orders go through it, so the guard against a power leaving the
floating-point range on the CPU lives here (:func:`_scaled_p_norm`); where the
range-safe kernel serves the call (a finite ``p > 1`` but 2, on Metal and
the CPU: ``backend/gpu/PNorm.h``, ``cpu::PNorm``) the reduction itself is
range-safe.  ``norm``
additionally rescales whatever it dispatches by the largest magnitude
(:func:`_scaled`), on every device, as it always has.
"""

import math
from collections.abc import Callable

import lucid
from lucid._C import engine as _C_engine
from lucid._ops._reduce import _p_norm, _takes_scaled_kernel
from lucid._tensor.tensor import Tensor

_Impl = _C_engine.TensorImpl

#: Largest |exponent| of a power of two the reduction may form: ``|x|^p`` is
#: kept inside ``2^±E`` so a sum of up to ``2^20`` of them still fits.  float32
#: stops at ``2^±126`` (and Metal flushes the subnormals below that to zero),
#: float64 at ``2^±1022``.
_RANGE_EXPONENT = {_C_engine.F32: 100.0, _C_engine.F64: 900.0}
_HALF = frozenset({_C_engine.F16, _C_engine.BF16})


def _scaled_p_norm(r: _Impl, p: float, axes: list[int], keepdim: bool) -> _Impl:
    """``(sum |r|^p)^(1/p)`` over ``axes`` for a real ``r``; CPU safe from underflow.

    The norm is absolutely homogeneous, so each slice's largest magnitude can
    be divided out first and multiplied back: every ratio lands in [0, 1] and
    nothing powers out of range.  ``vector_norm([1e-20, 1e-20])`` in float32 is
    2e-20, an ordinary number, and the direct evaluation squares to 1e-40.

    A finite ``p > 1`` other than 2 of a float32 or float64 CPU tensor is the
    range-safe kernel itself (:func:`_takes_scaled_kernel`): it keeps the
    largest magnitude in the pass that reads the data, and its backward only
    forms ratios to the norm, so the value and every derivative are in range
    and nothing is read or redone here.  The same holds on Metal, which has
    its own scaled kernel (``backend/gpu/PNorm.h``), so no read of the result
    (a ~200 us device sync) and no extra pass is needed.

    What is left on the CPU is ``p = 2``, whose plain graph keeps its range
    wherever the forward does, and ``0 < p < 1`` and ``p < 0``.  The rescale
    costs several passes over the input, and only a slice whose magnitude is
    outside the window where ``|x|^p`` is representable needs it.  The
    reduction therefore runs plain first and its answers are read: when every
    one is inside the window the plain result stands, bit for bit what it was
    before, and only an underflowed, overflowed or all-zero slice has the call
    redone rescaled.  The read is one boolean of the *result*, and costs
    nothing there — the CPU is eager.  A compile trace and a phantom tensor
    have no value to read and take the plain graph.  ``norm`` still rescales
    on every device.

    The divisor is a constant of the call — computed without the graph — so
    the gradient is the plain norm's, and an all-zero, ``nan`` or infinite
    slice takes the unscaled evaluation, which is right there already.  A
    half input reduces in float32, as :func:`_p_norm` does.
    """
    # ``p == 0`` counts entries and ``±inf`` picks one: neither powers anything.
    # An empty tensor has no largest magnitude (``max`` refuses it).
    if not _readable(r) or p == 0 or math.isinf(p) or r.numel() == 0:
        return _p_norm(r, p, axes, keepdim)
    if r.dtype in _HALF:
        out = _scaled_p_norm(_C_engine.astype(r, _C_engine.F32), p, axes, keepdim)
        return _C_engine.astype(out, r.dtype)
    if _takes_scaled_kernel(r, p):
        return _p_norm(r, p, axes, keepdim)
    out = _p_norm(r, p, axes, keepdim)
    exponent = _RANGE_EXPONENT[r.dtype] / max(abs(p), 1.0)
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


def _scaled(
    x: Tensor, ord: int | float | str | None, compute: Callable[[Tensor], Tensor]
) -> Tensor:
    """Run ``compute`` on a rescaled ``x`` and undo the scaling.

    Every norm order here is absolutely homogeneous — ``‖αx‖ = |α|‖x‖`` —
    so the largest magnitude can be divided out first and multiplied back
    at the end.  Every order but one: ``ord=0`` counts non-zero entries,
    which no positive scaling changes, so multiplying the scale back in
    inflates the count by it — ``norm([1, -2, 3], ord=0)`` answered 9
    instead of 3.  It also cannot overflow, so it skips the rescaling
    rather than being corrected for it.  That matters because the squaring in a 2-norm reaches
    infinity before the answer does: ``norm([1e200, 1e200])`` is 1.41e200,
    an ordinary double, and the direct evaluation says ``inf``.  The other
    end underflows to zero.  Dividing by the largest magnitude puts every
    ratio in [0, 1], so nothing squares out of range.  BLAS's ``nrm2`` has
    rescaled for the same reason since the seventies.

    The scale is only used when it is finite and positive.  An all-zero
    input has scale 0, and an input containing an infinity has scale inf —
    dividing by either turns the answer into NaN, which is how the first
    version of this rescaling reported ``norm([inf, 1])``.  Both fall back
    to 1, where the unscaled evaluation is already right.
    """
    if ord == 0 and not isinstance(ord, str):
        return compute(x)
    # Nothing to rescale, and no largest magnitude to find: ``max`` of an
    # empty tensor is refused, as it has no identity.
    if x.numel() == 0:
        return compute(x)
    scale = lucid.max(lucid.abs(x))
    usable = lucid.isfinite(scale) & (scale > lucid.zeros_like(scale))
    safe = lucid.where(usable, scale, lucid.ones_like(scale))
    return compute(x / safe) * safe
