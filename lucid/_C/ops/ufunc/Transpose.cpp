// lucid/_C/ops/ufunc/Transpose.cpp
//
// Forward and backward implementations for the permute family.
// Two anonymous-namespace helpers handle all axis bookkeeping so that the
// public functions are kept short:
//   validate_perm — normalises user-supplied axis indices and asserts that the
//                   result is a valid permutation of 0..ndim-1.
//   inverse_perm  — builds the inverse permutation used by the backward pass.

#include "Transpose.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"

namespace lucid {

// KeepInput: permute is valid for any dtype; no promotion is required.
const OpSchema PermuteBackward::schema_v1{"permute", 1, AmpPolicy::KeepInput, true};

namespace {

// Normalise perm_user by wrapping negative indices and verify that the result
// is a bijection on {0, …, ndim-1}.  Returns the normalised permutation.
// Throws a descriptive error for invalid inputs.
std::vector<int> validate_perm(const std::vector<int>& perm, int ndim) {
    if (static_cast<int>(perm.size()) != ndim) {
        ErrorBuilder("permute").invalid_argument("perm length must equal tensor ndim");
    }
    std::vector<int> normalized;
    normalized.reserve(ndim);
    // A 0-d tensor takes only the empty permutation (its length check above),
    // as in the reference, so no entry reaches its notional axis.
    for (int p : perm)
        normalized.push_back(wrap_dim(p, ndim, "permute"));
    // Sort a copy and check for 0..ndim-1 to detect duplicates and gaps.
    std::vector<int> sorted = normalized;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < ndim; ++i) {
        if (sorted[i] != i) {
            ErrorBuilder("permute").invalid_argument("perm must be a permutation of 0..ndim-1");
        }
    }
    return normalized;
}

// Compute the inverse permutation: inv[perm[i]] = i.
// Applying the inverse to a permuted tensor restores the original axis order.
std::vector<int> inverse_perm(const std::vector<int>& perm) {
    std::vector<int> inv(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        inv[static_cast<std::size_t>(perm[i])] = static_cast<int>(i);
    }
    return inv;
}

}  // namespace

// Build the output shape/stride, dispatch the backend permute, and wire the
// backward node with the normalised perm saved on it.
// wire_autograd is called with save_output=false because PermuteBackward does
// not need saved_output_; it only needs perm_ and out_shape_.
TensorImplPtr PermuteBackward::forward(const TensorImplPtr& a, const std::vector<int>& perm_user) {
    Validator::input(a, "permute.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());
    const auto perm = validate_perm(perm_user, ndim);

    Shape out_shape;
    Stride out_stride;
    out_shape.reserve(ndim);
    out_stride.reserve(ndim);
    for (int p : perm) {
        out_shape.push_back(a->shape()[static_cast<std::size_t>(p)]);
        out_stride.push_back(a->stride()[static_cast<std::size_t>(p)]);
    }

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), out_shape};
    // 3.5 Phase 1.2 step 4: report the permutation to the trace so the
    // compile-path Permute emitter can rebuild the same axis ordering
    // inside MPSGraph.  No-op outside any _tracing() scope.
    {
        std::vector<std::int64_t> perm64(perm.begin(), perm.end());
        scope.set_attr("permutation", std::move(perm64));
    }

    // On the CPU the result is a view: the same buffer, the axes' strides
    // reordered, so a write through either side reaches the other.  Every
    // consumer reads a strided CPU tensor correctly (``TensorImpl::storage``
    // packs it), and an in-place write follows its strides.  Metal keeps
    // copy semantics — its arrays cannot see one another's writes.
    TensorImplPtr out;
    if (a->views_by_stride()) {
        out = TensorImpl::make_view(a, out_shape, out_stride, 0);
    } else {
        Storage out_storage = backend::Dispatcher::for_device(a->device())
                                  .permute(a->storage(), a->shape(), perm, a->dtype());
        out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, a->dtype(),
                                           a->device(), false);
    }

    auto bwd = std::make_shared<PermuteBackward>();
    bwd->perm_ = perm;
    kernel::NaryKernel<PermuteBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

// dL/dx = permute(dL/dy, inverse(perm_)).
// out_shape_ here is the permuted output shape; it becomes the input shape of
// the inverse permute call.
std::vector<Storage> PermuteBackward::apply(Storage grad_out) {
    const auto inv = inverse_perm(perm_);
    Storage dx =
        backend::Dispatcher::for_device(device_).permute(grad_out, out_shape_, inv, dtype_);
    return {std::move(dx)};
}

// Graph-mode: apply the inverse permutation using permute_op so the result
// is tracked in the autograd graph for second-order differentiation.
std::vector<TensorImplPtr> PermuteBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    const auto inv = inverse_perm(perm_);
    return {permute_op(grad_out, inv)};
}

TensorImplPtr permute_op(const TensorImplPtr& a, const std::vector<int>& perm) {
    return PermuteBackward::forward(a, perm);
}

// Build the full-reversal permutation [ndim-1, ndim-2, ..., 0].
TensorImplPtr transpose_op(const TensorImplPtr& a) {
    Validator::input(a, "transpose.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());
    std::vector<int> perm(ndim);
    for (int i = 0; i < ndim; ++i)
        perm[i] = ndim - 1 - i;
    return PermuteBackward::forward(a, perm);
}

// .T property: same as transpose_op.
TensorImplPtr T_op(const TensorImplPtr& a) {
    return transpose_op(a);
}

// .mT property: swap the last two axes only, leaving all leading axes intact.
TensorImplPtr mT_op(const TensorImplPtr& a) {
    Validator::input(a, "mT.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());
    if (ndim < 2)
        ErrorBuilder("mT").invalid_argument("requires ndim >= 2");
    std::vector<int> perm(ndim);
    for (int i = 0; i < ndim; ++i)
        perm[i] = i;
    std::swap(perm[ndim - 1], perm[ndim - 2]);
    return PermuteBackward::forward(a, perm);
}

// Build the identity permutation and swap the two requested axis positions.
TensorImplPtr swapaxes_op(const TensorImplPtr& a, int axis1, int axis2) {
    Validator::input(a, "swapaxes.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());
    const int w1 = wrap_dim(axis1, ndim, "swapaxes");
    const int w2 = wrap_dim(axis2, ndim, "swapaxes");
    std::vector<int> perm(ndim);
    for (int i = 0; i < ndim; ++i)
        perm[i] = i;
    // A 0-d tensor swaps its notional axis with itself: the empty permutation.
    if (ndim > 0)
        std::swap(perm[w1], perm[w2]);
    return PermuteBackward::forward(a, perm);
}

LUCID_REGISTER_OP(PermuteBackward)

}  // namespace lucid
