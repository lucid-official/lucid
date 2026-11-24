"""``lerp`` split on the weight, as the reference's CPU kernel splits it (LCD-1101).

``start + weight * (end - start)`` is exact at ``weight = 0`` but not at ``1``: it rounds
``end - start`` and adds it back, so ``lerp(1, 1e-8, 1)`` was ``0``.  Its answer for an
operand that is not finite is the form's too: ``lerp(0, inf, 0.5)`` was ``inf`` where the
reference's is ``NaN``.  The reference's kernel picks the form by the weight::

    |weight| <  0.5:  start + weight * (end - start)
    |weight| >= 0.5:  end - (end - start) * (1 - weight)

so ``lerp(a, b, 1) == b`` and ``lerp(a, b, 0) == a`` hold exactly for finite operands, a
``NaN`` weight takes the second form (``NaN < 0.5`` is false), and an infinite operand
gives what the form it falls in gives (``inf - inf`` for ``lerp(0, inf, 0.5)``).

The second form is written ``end + (weight - 1) * (end - start)``: ``weight - 1`` is
``-(1 - weight)`` in floating point, and ``end + (-x)`` is ``end - x``, so it is the same
number up to the sign of a zero.  At ``weight = 1`` it is ``+0`` where the other is ``-0``, and
a zero ``end`` can come back as ``+0`` (``lerp(-2, -0, 1)``); the reference's vector body takes
this form and its scalar tail the other, and they differ only there.  The whole is the first
form with another base (``end``) and coefficient.  A tensor weight picks base and coefficient
with a ``where`` on ``|weight| < 0.5`` and never leaves the device, so there is no host read
and no sync on Metal; a Python number picks them at once, on the host.  Picking the *operands*
of one product, not the results of two, keeps ``0 * inf`` out of the form that was not taken.

Metal follows the reference's CPU kernel, not its ``mps`` one.  That one uses
``start + weight * (end - start)`` for every weight, which misses ``lerp(a, b, 1) == b`` and
answers ``inf`` where the CPU kernel answers ``NaN``; one op answers alike on both devices.

The gradient is the reference's and ignores the branch: ``grad * (1 - weight)`` for
``start``, ``grad * weight`` for ``end`` and ``grad * (end - start)`` for ``weight``
(conjugated for a complex tensor).  Differentiating the value gives the same numbers up to a
rounding (``g + g * (weight - 1)`` is not ``g * weight``), so :class:`_Lerp` writes the three
out.
"""

from typing import final, override

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._scalar_cast import ScalarRead, cast_scalar
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx

#: ``|weight|`` below this takes ``start + weight * (end - start)``.  The rule has three
#: copies: here, ``lerp_toward`` in ``lucid/_C/optim/Prop.cpp`` and
#: ``_CompiledRMSprop._flags`` in ``lucid/compile/_optim/compiler.py``; change all three.
_SMALL_WEIGHT = 0.5


def _step(dtype: _C_engine.Dtype, weight: float | complex) -> float:
    """``weight`` as the kernel reads it: rounded to the operands' opmath dtype.

    ``lerp`` takes floating tensors only, so the weight read for one is real.
    """
    return float(cast_scalar(dtype, weight, read=ScalarRead.OP_MATH).real)


def _conj(t: Tensor) -> Tensor:
    """``t.conj()`` for a complex tensor, ``t`` itself for a real one."""
    return t.conj() if t.is_complex() else t


def _constant(value: float, like: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    """``value`` as a 0-d tensor of ``like``'s dtype, on its device."""
    return _C_engine.full([], value, like.dtype, like.device)


def _value(
    start: _C_engine.TensorImpl,
    end: _C_engine.TensorImpl,
    weight: _C_engine.TensorImpl | float | complex,
) -> _C_engine.TensorImpl:
    """The split value over engine tensors: each op is the engine's own, so it records a gradient.

    The engine's ops, not the tensor operators: a call to ``lerp`` is a handful of ops on a
    small tensor, and the operators' dtype and device rules cost several times what the ops do.
    """
    diff = _C_engine.sub(end, start)
    if isinstance(weight, _C_engine.TensorImpl):
        magnitude = _C_engine.abs(weight)
        small = _C_engine.less(magnitude, _constant(_SMALL_WEIGHT, magnitude))
        coeff = _C_engine.where(small, weight, _C_engine.sub(weight, _constant(1.0, weight)))
        return _C_engine.add(_C_engine.where(small, start, end), _C_engine.mul(coeff, diff))
    step = _step(start.dtype, weight)
    if abs(step) < _SMALL_WEIGHT:
        return _C_engine.add(start, _C_engine.mul(diff, _constant(step, diff)))
    return _C_engine.add(end, _C_engine.mul(diff, _constant(step - 1, diff)))


def _lerp_value(start: Tensor, end: Tensor, weight: Tensor | float | complex) -> Tensor:
    """``lerp(start, end, weight)`` from differentiable ops, split on ``|weight| < 0.5``.

    ``start`` and ``end`` share a floating dtype and a tensor ``weight`` is at it
    too, as :func:`lucid._ops.composite.blas._lerp` leaves them.  A ``lucid.compile``
    recording takes this form in place of the node (:attr:`_Lerp._trace_form`).
    """
    return _wrap(
        _value(
            _unwrap(start),
            _unwrap(end),
            _unwrap(weight) if isinstance(weight, Tensor) else weight,
        )
    )


@final
class _Lerp(Function):
    """``lerp`` with the reference's two-sided value and its one-sided derivative.

    A number ``weight`` is kept as it was given, not as the kernel reads it: the
    reference's ``1 - weight`` is taken from it.  A tensor ``weight`` is saved for the
    ``start`` and ``end`` gradients and ``start`` and ``end`` for the ``weight`` one.
    """

    _trace_form = staticmethod(_lerp_value)

    @override
    @staticmethod
    def forward(ctx: FunctionCtx, start: Tensor, end: Tensor, weight: Tensor | float) -> Tensor:
        needs_start, needs_end, needs_weight = ctx.needs_input_grad
        ctx.number = None
        if isinstance(weight, Tensor):
            saved = [weight] if needs_start or needs_end else []
            ctx.save_for_backward(*saved, *([start, end] if needs_weight else []))
        else:
            ctx.number = float(complex(weight).real)
        return _lerp_value(start, end, weight)

    @override
    @staticmethod
    def backward(
        ctx: FunctionCtx, grad: Tensor
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        needs_start, needs_end, needs_weight = ctx.needs_input_grad
        number = ctx.number
        if isinstance(number, float):
            return (
                grad * (1 - number) if needs_start else None,
                grad * number if needs_end else None,
                None,
            )
        saved = list(ctx.saved_tensors)
        weight = saved.pop(0) if needs_start or needs_end else None
        return (
            grad * _conj(1 - weight) if needs_start and weight is not None else None,
            grad * _conj(weight) if needs_end and weight is not None else None,
            grad * _conj(saved[1] - saved[0]) if needs_weight else None,
        )


def _apply_lerp(start: Tensor, end: Tensor, weight: Tensor | float | complex) -> Tensor:
    """``lerp`` of tensors of one floating dtype: the node when a gradient is wanted.

    Nothing records without one, and the node's bookkeeping is most of the time a small
    tensor's ``lerp`` takes, so the value alone is returned then.
    """
    if not (
        _C_engine.grad_enabled()
        and (
            start.requires_grad
            or end.requires_grad
            or (isinstance(weight, Tensor) and weight.requires_grad)
        )
    ):
        return _lerp_value(start, end, weight)
    out = _Lerp.apply(start, end, weight)
    assert isinstance(out, Tensor)
    return out
