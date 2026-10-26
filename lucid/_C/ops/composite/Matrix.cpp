// lucid/_C/ops/composite/Matrix.cpp
//
// ``mm`` and ``bmm`` validate the rank up front then defer to ``matmul_op``;
// the rank check makes their error messages crisper than letting matmul's
// internal planner reject the input.
//
// ``kron`` builds the block structure with the standard outer-product trick:
// expand each input to a 2·ndim-rank tensor where every dimension of ``a`` is
// followed by a size-1 placeholder for ``b`` and vice versa, multiply with
// broadcasting, then collapse the interleaved pairs back into the result
// shape (sa[i] · sb[i]).
//
// Each opens with an :class:`AmpScope` under its own name, so autocast
// applies the composite's row to its operands once.

#include "Matrix.h"

#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
#include "../bfunc/Matmul.h"
#include "../bfunc/Mul.h"
#include "../utils/Promote.h"
#include "../utils/View.h"

namespace lucid {

TensorImplPtr mm_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (!a || !b)
        ErrorBuilder("mm").fail("null input");
    if (a->shape().size() != 2 || b->shape().size() != 2)
        ErrorBuilder("mm").fail("both operands must be 2-D");
    const AmpScope amp{"mm", a->device()};
    return matmul_op(amp.cast(a), amp.cast(b));
}

TensorImplPtr bmm_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (!a || !b)
        ErrorBuilder("bmm").fail("null input");
    if (a->shape().size() != 3 || b->shape().size() != 3)
        ErrorBuilder("bmm").fail("both operands must be 3-D");
    if (a->shape()[0] != b->shape()[0])
        ErrorBuilder("bmm").fail("batch dimensions must match");
    const AmpScope amp{"bmm", a->device()};
    return matmul_op(amp.cast(a), amp.cast(b));
}

TensorImplPtr kron_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in) {
    if (!a_in || !b_in)
        ErrorBuilder("kron").fail("null input");
    const AmpScope amp{"kron", a_in->device()};
    const auto cast = amp.cast_widest({a_in, b_in});
    const TensorImplPtr& a = cast[0];
    const TensorImplPtr& b = cast[1];
    const auto& sa = a->shape();
    const auto& sb = b->shape();

    // Pad the lower-rank operand's shape with leading 1s to the common rank,
    // then interleave size-1 placeholders so the multiply broadcasts to the
    // full 2·ndim-rank block tensor.
    const std::size_t ndim = std::max(sa.size(), sb.size());
    Shape a_padded(ndim - sa.size(), 1);
    a_padded.insert(a_padded.end(), sa.begin(), sa.end());
    Shape b_padded(ndim - sb.size(), 1);
    b_padded.insert(b_padded.end(), sb.begin(), sb.end());

    Shape a_expanded;
    Shape b_expanded;
    a_expanded.reserve(2 * ndim);
    b_expanded.reserve(2 * ndim);
    for (std::size_t i = 0; i < ndim; ++i) {
        a_expanded.push_back(a_padded[i]);
        a_expanded.push_back(1);
        b_expanded.push_back(1);
        b_expanded.push_back(b_padded[i]);
    }
    auto a_r = reshape_op(a, a_expanded);
    auto b_r = reshape_op(b, b_expanded);
    auto product = mul_op(a_r, b_r);

    // Collapse each (a_padded[i], b_padded[i]) pair into a single axis.
    Shape out_shape;
    out_shape.reserve(ndim);
    for (std::size_t i = 0; i < ndim; ++i)
        out_shape.push_back(a_padded[i] * b_padded[i]);
    return reshape_op(product, out_shape);
}

}  // namespace lucid
