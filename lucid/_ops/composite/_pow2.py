"""Exact scaling by a power of two: the kernels of ``ldexp`` and ``frexp``.

For an integral ``k``, ``x * 2**k`` is exact whenever the answer is
representable, and both ops answer exactly.  A fractional ``k`` costs the
rounding of ``2**frac``.  Three things stand in the way:

* **Range.**  ``2**k`` leaves float32 for ``|k| > 127`` where
  ``x * 2**k`` need not (``ldexp(1e30, -200)``), so the scaling runs in
  three bounded steps (:func:`_scale_pow2`).
* **Negative powers on the GPU.**  Metal's ``pow`` is inexact for a
  negative exponent (``2 ** -1`` is ``0.49999994``), so ``2**-m`` is the
  reciprocal of ``2**m`` (:func:`_pow2`).  It is taken by ``reciprocal``,
  not by ``1 / ...``: MPSGraph's simplifier rewrites ``1 / pow(2, m)`` into
  that inexact ``pow(2, -m)``.  A compiled ``ldexp(1, -126)`` came out
  just below the smallest normal float, and the GPU flushed it to zero.
* **float32 subnormals on Metal.**  The GPU reads a subnormal operand as
  zero and flushes a subnormal result.  The eager Metal path reads a
  subnormal operand from its bits (:func:`_normalize_metal_f32`) and writes
  a subnormal answer back the same way
  (:func:`_restore_metal_f32_subnormals`).  Any shift by an integer is
  applied to a whole exponent only (:func:`_split_exponent`), in the
  forward's subnormal shift, the subnormal answer and ``d/de``.
  ``e - 149`` or ``e + 149`` rounds away most of a fractional ``e``, and a
  subnormal answer from ``trunc(e)`` drops it entirely.

Two gradients cannot come from differentiating those compositions.  For
them the derivatives are written out, in :class:`_Ldexp` and
:class:`_MetalFrexp`:

* The bits a Metal subnormal is read from carry no gradient, so eager
  Metal float32 always takes these classes.
* The exponent's gradient, differentiated through the three steps, takes
  the reciprocal's derivative ``-(2**-m)**2``.  At a large negative
  exponent that rounds through a subnormal and comes out 0, even though
  the gradient itself is normal.  So ``ldexp`` is :class:`_Ldexp` wherever
  its exponent needs a gradient.

The written-out derivatives are three-step products themselves.  ``d/dx``
is exact for an integral exponent, and ``d/de`` is within a few ulp,
wherever the gradient is a normal number.  A subnormal gradient flushes on
Metal, as every Metal backward flushes one.  Everywhere else the ops are
the plain differentiable compositions: the CPU with an exponent that needs
no gradient, and any ``lucid.compile`` trace.  A trace records no bitcast,
so float32 subnormals flush there.
"""

import math
from typing import final, override

import lucid
from lucid import _promotion
from lucid._C import _C_engine
from lucid._dispatch import _wrap
from lucid._ops.composite._shared import _is_tensor
from lucid._tensor.tensor import Tensor
from lucid._types import Scalar
from lucid.autograd.function import Function, FunctionCtx

_LN2 = math.log(2.0)

#: The largest ``|k|`` the three steps of :func:`_scale_pow2` take, per
#: dtype: each step's ``2**(k/3)`` stays finite and normal.
_F32_LIMIT = 300.0
_F64_LIMIT = 2400.0


def _pow2(k: Tensor) -> Tensor:
    """``2**k``, exact for an integral ``k`` within the dtype's exponents."""
    return lucid.where(k < 0.0, lucid.reciprocal(2.0 ** (-k)), 2.0**k)


def _scale_pow2(work: Tensor, exp_t: Tensor, limit: float) -> Tensor:
    """``work * 2**exp_t`` in three steps of at most ``limit / 3`` each.

    The exponent is clipped to ``±limit``, which only moves an answer that
    is already zero or infinite.  A non-integral exponent's fraction rides
    on the last step, so the result stays differentiable in ``exp_t``.
    """
    k = lucid.clip(lucid.trunc(exp_t), -limit, limit)
    k1 = lucid.trunc(k / 3.0)
    k2 = lucid.trunc((k - k1) * 0.5)
    k3 = lucid.where(lucid.abs(exp_t) > limit, k - k1 - k2, exp_t - k1 - k2)
    return ((work * _pow2(k1)) * _pow2(k2)) * _pow2(k3)


def _split_exponent(e: Tensor) -> tuple[Tensor, Tensor]:
    """``e`` as its whole part and its fraction (``0`` for ``±inf`` and NaN)."""
    whole = lucid.trunc(e)
    return whole, lucid.where(lucid.isfinite(e), e - whole, lucid.zeros_like(e))


def _limit(work: Tensor) -> float:
    """The largest exponent :func:`_scale_pow2` takes at ``work``'s dtype."""
    return _F64_LIMIT if work.dtype == lucid.float64 else _F32_LIMIT


def _traced() -> bool:
    """Whether a ``lucid.compile`` trace is recording."""
    return _C_engine.compile.current_tracer() is not None


def _reads_bits(work: Tensor) -> bool:
    """Whether ``work``'s float32 subnormals are read from its bits.

    Only eager Metal does this.  A trace records no bitcast, and a phantom
    tensor has no bits to read.
    """
    return (
        work.device.type == "metal"
        and work.dtype == lucid.float32
        and not work._impl.is_phantom
        and not _traced()
    )


def _normalize_metal_f32(work: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``work`` with each subnormal scaled by ``2**149``, the shift, the bits.

    A subnormal's magnitude bits are its value in units of ``2**-149``, so
    the integer they spell, as a float, is the scaled value exactly.  Every
    other element passes through, still differentiable.
    """
    with lucid.no_grad():
        bits = _wrap(_C_engine._bitcast(work._impl, _C_engine.Dtype.I32))
        abs_bits = bits & 0x7FFFFFFF
        is_sub = ((abs_bits >> 23) == 0) & (abs_bits != 0)
        signed_frac = lucid.where(bits < 0, -abs_bits, abs_bits).to(dtype=lucid.float32)
        shift = lucid.where(
            is_sub, lucid.full_like(work, 149.0), lucid.zeros_like(work)
        )
    return lucid.where(is_sub, signed_frac, work), shift, bits


def _restore_metal_f32_subnormals(
    work_norm: Tensor, exp_eff: Tensor, bits: Tensor, out: Tensor, integral: bool
) -> Tensor:
    """``out``, with each subnormal answer the GPU flushed written from bits.

    The answer in units of ``2**-149`` is rounded to an integer (half to
    even, as IEEE rounds), and that integer becomes the magnitude bits.  A
    fractional exponent is split first: ``exp_eff + 149`` can be coarser
    than its fraction (``2**-126`` and ``-0.01`` meet at ``148.99``).  The
    whole part is an exact shift, and the fraction scales the result.  It
    cannot scale ``work_norm`` first, which may then be subnormal itself.
    """
    with lucid.no_grad():
        magnitude = lucid.abs(work_norm)
        if integral:
            scaled = _scale_pow2(magnitude, exp_eff + 149.0, _F32_LIMIT)
        else:
            whole, frac = _split_exponent(exp_eff)
            scaled = _scale_pow2(magnitude, whole + 149.0, _F32_LIMIT) * _pow2(frac)
        finite_nz = (
            (lucid.abs(work_norm) > 0.0)
            & (~lucid.isinf(work_norm))
            & (~lucid.isnan(work_norm))
            & (~lucid.isnan(exp_eff))
        )
        flushed = (
            finite_nz
            & (lucid.abs(out) == 0.0)
            & (scaled >= 0.5)
            & (scaled <= float(1 << 23))
        )
        sign_bit = (bits & (-0x80000000)).broadcast_to(out.shape)
        sub_bits = sign_bit | lucid.round(scaled).to(dtype=lucid.int32)
        sub_f32 = _wrap(_C_engine._bitcast(sub_bits._impl, _C_engine.Dtype.F32))
    return lucid.where(flushed, sub_f32, out)


def _metal_f32_ldexp(x: Tensor, e: Tensor, integral: bool) -> Tensor:
    """``x * 2**e``, with ``x``'s subnormals read and written as bits.

    A subnormal ``x`` is scaled up by ``2**149`` and ``e`` down by 149.  A
    fractional ``e`` would lose its fraction to that shift, so for those
    elements the fraction scales ``x`` first and only the whole part
    shifts.  Every other element computes as main always has.
    """
    x_n, shift, bits = _normalize_metal_f32(x)
    e_n = e - shift
    if not integral:
        whole, frac = _split_exponent(e)
        is_sub = shift != 0.0
        x_n = lucid.where(is_sub, x_n * _pow2(frac), x_n)
        e_n = lucid.where(is_sub, whole - shift, e)
    out = _scale_pow2(x_n, e_n, _F32_LIMIT)
    return _restore_metal_f32_subnormals(x_n, e_n, bits, out, integral)


def _frexp_arith(work: Tensor) -> tuple[Tensor, Tensor]:
    """``(m, e)`` with ``work = m * 2**e`` in one pass, ``e`` a float.

    ``log2`` only estimates the exponent (the GPU's ``log2(8)`` is
    ``2.9999998``).  So the mantissa it implies is formed, and the pair
    moves by one where that mantissa falls outside ``[0.5, 1)``.  Zero,
    infinity and NaN pass through as given, with exponent ``0``.
    """
    abs_x = lucid.abs(work)
    special = (abs_x == 0.0) | lucid.isinf(work) | lucid.isnan(work)
    safe_abs = lucid.where(special, lucid.full_like(work, 1.0), abs_x)
    e = lucid.where(
        special, lucid.zeros_like(work), lucid.floor(lucid.log2(safe_abs)) + 1.0
    )
    e1 = lucid.trunc(e * 0.5)
    m = (work * _pow2(-e1)) * _pow2(-(e - e1))
    hi = (~special) & (lucid.abs(m) >= 1.0)
    m, e = lucid.where(hi, m * 0.5, m), lucid.where(hi, e + 1.0, e)
    lo = (~special) & (lucid.abs(m) < 0.5)
    m, e = lucid.where(lo, m * 2.0, m), lucid.where(lo, e - 1.0, e)
    return lucid.where(special, work, m), e


@final
class _Ldexp(Function):
    """``x * 2**e``, its derivatives written out.

    ``d/dx = 2**e`` and ``d/de = x * 2**e * ln 2``, as forward three-step
    products.  The second scales ``x``'s mantissa, so a subnormal ``x`` is
    not read as zero, by ``e``'s fraction first and then by the whole
    exponents.  ``x`` is saved only when that second one is needed.
    On eager Metal float32 the value's subnormals are read and written as
    bits.
    """

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, e: Tensor, integral: bool) -> Tensor:
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(e, x)
        else:
            ctx.save_for_backward(e)
        # ``no_grad``, not ``detach``: an alias of ``x`` taken here makes the
        # write of ``x.ldexp_(e)`` count as two versions.
        with lucid.no_grad():
            if not _reads_bits(x):
                return _scale_pow2(x, e, _limit(x))
            return _metal_f32_ldexp(x, e, integral)

    @override
    @staticmethod
    def backward(
        ctx: FunctionCtx, grad: Tensor
    ) -> tuple[Tensor | None, Tensor | None, None]:
        saved = ctx.saved_tensors
        e = saved[0]
        needs_x, needs_e, _ = ctx.needs_input_grad
        limit = _limit(grad)
        grad_x = _scale_pow2(grad, e, limit) if needs_x else None
        grad_e = None
        if needs_e:
            # The fraction first (it is in (-1, 1), so one power does):
            # ``e + e_x`` would round it away.
            m, e_x = _frexp_parts(saved[1])
            whole, frac = _split_exponent(e)
            scaled = grad * (_LN2 * m) * _pow2(frac)
            grad_e = _scale_pow2(scaled, whole + e_x, limit)
        return grad_x, grad_e, None


@final
class _MetalFrexp(Function):
    """``frexp`` on eager Metal float32, subnormals read as bits.

    The exponent is not differentiable.  The mantissa ``x * 2**-e`` has
    ``dm/dx = 2**-e``, which is the reference's ``grad / exp2(exponent)``.
    """

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> tuple[Tensor, Tensor]:
        with lucid.no_grad():
            x_n, shift, _ = _normalize_metal_f32(x)
            m, e = _frexp_arith(x_n)
            e = e - shift
        ctx.save_for_backward(e)
        ctx.mark_non_differentiable(e)
        return m, e

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_m: Tensor, grad_e: Tensor) -> Tensor:
        del grad_e  # the exponent is integral: no gradient flows through it
        (e,) = ctx.saved_tensors
        return _scale_pow2(grad_m, -e, _F32_LIMIT)


def _ldexp(work: Tensor, other: Tensor | Scalar) -> Tensor:
    """``work * 2**other``, with ``work`` float32 or float64.

    Parameters
    ----------
    work : Tensor
        The float32 or float64 mantissa.
    other : Tensor or Scalar
        The exponent, broadcast against ``work``.

    Returns
    -------
    Tensor
        The broadcast product at ``work``'s dtype.  For an integral
        exponent it is rounded once.
    """
    if _is_tensor(other):
        integral = not other.is_floating_point()
        exp_t = other.to(dtype=work.dtype)
    else:
        value = _promotion.as_number(other)
        integral = float(value).is_integer()
        exp_t = lucid.tensor(value, dtype=work.dtype, device=work.device)
    if _reads_bits(work) or (
        exp_t.requires_grad and lucid.is_grad_enabled() and not _traced()
    ):
        out = _Ldexp.apply(work, exp_t, integral)
        assert isinstance(out, Tensor)
        return out
    return _scale_pow2(work, exp_t, _limit(work))


def _frexp_parts(work: Tensor) -> tuple[Tensor, Tensor]:
    """``(m, e)`` with ``work = m * 2**e``, with ``e`` a float of ``work``'s dtype."""
    if not _reads_bits(work):
        return _frexp_arith(work)
    out = _MetalFrexp.apply(work)
    assert isinstance(out, tuple)
    m, e = out
    assert isinstance(m, Tensor) and isinstance(e, Tensor)
    return m, e


def _frexp(work: Tensor) -> tuple[Tensor, Tensor]:
    """``(m, e)`` with ``work = m * 2**e``, with ``work`` float32 or float64.

    Parameters
    ----------
    work : Tensor
        The float32 or float64 operand.

    Returns
    -------
    tuple of Tensor
        The mantissa at ``work``'s dtype, with ``|m|`` in ``[0.5, 1)``
        (zero, infinity and NaN pass through as given), and the int32
        exponent.  The mantissa is differentiable, ``dm/dx = 2**-e``; the
        exponent is not.
    """
    m, e = _frexp_parts(work)
    return m, e.to(dtype=lucid.int32)
