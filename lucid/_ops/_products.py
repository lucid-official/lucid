"""``dot`` and ``inner``: the vector products that rank their operands by rank.

``Tensor.dot`` and ``lucid.linalg.dot`` are the same adapter, as are
``Tensor.inner`` and ``lucid.linalg.inner``, so a rule cannot hold for one
spelling and not the other (and ``vdot``, which forwards to ``dot``, inherits it).

``dot`` takes two 1-D tensors and nothing else.  The engine's CPU kernel also
answered a pair of 2-D tensors and refused a 0-d pair with a kernel message,
where the reference refuses every rank but one with ``1D tensors expected``
(LCD-1061).  The refusal is a ``ShapeMismatch``, which is a ``RuntimeError`` as
the reference's is.

``inner`` contracts the last axes of two tensors, and a 0-d operand has none:
the reference answers ``a * b`` for it, with that product's promotion -- a 0-d
``int16`` beside an ``int8`` vector computes in ``int8``, where the engine's
single-dtype kernel refused the pair (LCD-1068).  Two tensors that both have a
dimension still meet at one dtype or are refused, as the reference refuses them.
"""

from lucid._C import engine as _C_engine
from lucid._ops._adapters import (
    _keep_engine_signature,
    _make_arith_adapter,
    _require_tensor_operands,
)

_Impl = _C_engine.TensorImpl

# ``a * b`` -- the full operator path (promotion tiers, half-precision op-math).
_times = _make_arith_adapter(_C_engine.mul)


def _dot_adapter(a_impl: _Impl, b_impl: _Impl) -> _Impl:
    """dot(a, b) -- the inner product of two 1-D tensors.

    Raises
    ------
    TypeError
        An operand is a Python number.
    lucid._C.engine.ShapeMismatch
        An operand is not 1-D.  The message is the reference's.
    """
    _require_tensor_operands("dot", a_impl, b_impl)
    ranks = (len(a_impl.shape), len(b_impl.shape))
    if ranks != (1, 1):
        raise _C_engine.ShapeMismatch(
            f"1D tensors expected, but got {ranks[0]}D and {ranks[1]}D tensors"
        )
    return _C_engine.dot(a_impl, b_impl)


def _inner_adapter(a_impl: _Impl, b_impl: _Impl) -> _Impl:
    """inner(a, b) -- the last axes contracted; ``a * b`` for a 0-d operand.

    Raises
    ------
    TypeError
        An operand is a Python number.
    """
    _require_tensor_operands("inner", a_impl, b_impl)
    if not a_impl.shape or not b_impl.shape:
        return _times(a_impl, b_impl)
    return _C_engine.inner(a_impl, b_impl)


_keep_engine_signature(_dot_adapter, _C_engine.dot)
_keep_engine_signature(_inner_adapter, _C_engine.inner)
