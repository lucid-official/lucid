// lucid/_C/nn/Linear.cpp
//
// Implementation of the fully-connected layer forward pass and its autograd
// backward node.  The forward flattens all leading dimensions of x into a
// single M-row batch, calls IBackend::linear() for the GEMM, and records
// shapes for the backward pass.  The backward delegates to
// IBackend::linear_backward(), which computes dW, db, and dx.

#include "Linear.h"

#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/BinaryKernel.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/Matmul.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/gfunc/Gfunc.h"
#include "../ops/ufunc/Astype.h"
#include "../ops/ufunc/Reductions.h"
#include "../ops/ufunc/Transpose.h"
#include "../ops/utils/View.h"

namespace lucid {

const OpSchema LinearBackward::schema_v1{"linear", 1, AmpPolicy::Promote, true};

namespace {

// Helper carrying the (M, K) pair derived by flattening x's leading dims.
struct FlatX {
    std::size_t M;
    std::size_t K;
};

// Collapse all dimensions of x except the last one into M rows.
// The last dimension becomes K (input features).
// Throws if x is 0-D (scalar).
FlatX flatten_x(const Shape& x_shape) {
    if (x_shape.empty()) {
        ErrorBuilder("linear").invalid_argument("x must be at least 1-D");
    }
    std::size_t m = 1;
    for (std::size_t d = 0; d + 1 < x_shape.size(); ++d) {
        m *= static_cast<std::size_t>(x_shape[d]);
    }
    return {m, static_cast<std::size_t>(x_shape.back())};
}

}  // namespace

TensorImplPtr
LinearBackward::forward(const TensorImplPtr& x, const TensorImplPtr& W, const TensorImplPtr& b) {
    Validator::input(x, "linear.x").non_null();
    Validator::input(W, "linear.W").non_null().ndim(2);
    Validator::input(b, "linear.b").non_null().ndim(1);
    // Device check first — AMP cast doesn't change device.
    Validator::pair(x, W, "linear: x/weight").same_device();
    Validator::pair(x, b, "linear: x/bias").same_device();

    // 3.3 AMP plumbing: schema_v1.amp_policy == Promote.  Under an
    // ``AutocastGuard(F16)`` scope, SchemaGuard resolves ``eff_dt`` to the
    // autocast target (F32 on CPU per Accelerate-no-F16 carve-out, F16 on
    // GPU).  All three operands are cast to ``eff_dt`` before the matmul
    // so MLX's conv/matmul kernels can take their native F16 fast path —
    // previously the autocast scope was completely ignored by Linear, so
    // F32 was used even inside ``with amp.autocast(F16):``.  Under
    // autocast, the surrounding op has cast x to F16 while W / b on the
    // Parameter slots are still F32; after the cast all three share
    // ``eff_dt``.
    //
    // ``astype_op`` (not ``maybe_cast_for_kernel``) is used because it
    // wires an ``AstypeBackward`` node when the input has requires_grad.
    // Without this the cast tensors carry requires_grad=false → the
    // downstream ``NaryKernel::wire_autograd`` any_grad check fires
    // false → the entire backward chain breaks at every AMP cast
    // boundary.  ``astype_op`` is a same-dtype no-op (returns input as
    // ``shared_ptr``), so the F32 baseline pays no cost.
    SchemaGuard sg{schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr x_eff = astype_op(x, eff_dt);
    const TensorImplPtr W_eff = astype_op(W, eff_dt);
    const TensorImplPtr b_eff = astype_op(b, eff_dt);

    const auto fx = flatten_x(x_eff->shape());
    const std::size_t M = fx.M;
    const std::size_t K = fx.K;
    const std::size_t N = static_cast<std::size_t>(W_eff->shape()[0]);

    if (W_eff->shape()[1] != static_cast<std::int64_t>(K))
        throw ShapeMismatch(W_eff->shape(), x_eff->shape(), "linear: W.shape[1] != x.last_dim");
    if (b_eff->shape()[0] != static_cast<std::int64_t>(N))
        throw ShapeMismatch(b_eff->shape(), W_eff->shape(), "linear: b.shape[0] != W.shape[0]");

    // Output shape mirrors x but with the last dim replaced by N (out features).
    Shape out_shape = x_eff->shape();
    out_shape.back() = static_cast<std::int64_t>(N);

    OpScopeFull scope{schema_v1.name, x_eff->device(), eff_dt, out_shape};
    // Each output element requires 2*K FLOPs (multiply-add pairs).
    scope.set_flops(static_cast<std::int64_t>(2 * M * N * K));

    Storage out_storage = backend::Dispatcher::for_device(x_eff->device())
                              .linear(x_eff->storage(), W_eff->storage(), b_eff->storage(),
                                      x_eff->shape(), W_eff->shape(), out_shape, eff_dt);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape), eff_dt,
                                            x_eff->device(), false);

    // Wire the backward node; saved_inputs_ holds {x, W} at the
    // effective (cast) dtype so apply() can do the matmul transposes
    // entirely at F16 — matching the forward path.  The bias is read by
    // shape alone (db sums the gradient), so it is not saved and a write
    // into it after forward is not refused.
    kernel::NaryKernel<LinearBackward, 3>::wire_autograd(
        std::make_shared<LinearBackward>(), {x_eff, W_eff, b_eff}, out, {true, true, false});
    return out;
}

std::vector<Storage> LinearBackward::apply(Storage grad_out) {
    // saved_inputs_[0..2] are {x, W, b}; input_shapes_[0..2] are their shapes.
    return backend::Dispatcher::for_device(device_).linear_backward(
        grad_out, saved_or_zeros(0, grad_out), saved_or_zeros(1, grad_out), input_shapes_[0],
        input_shapes_[1], input_shapes_[2], dtype_);
}

std::vector<TensorImplPtr> LinearBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // The eager path hands the whole job to one fused backend call, which
    // returns Storages and therefore no graph.  A second derivative has to
    // come from the three GEMMs written out, where each factor keeps its
    // own backward node.
    const auto& x = saved_impl_inputs_[0];
    const auto& W = saved_impl_inputs_[1];
    const auto& b = saved_impl_inputs_[2];
    if ((needs_input_grad(0) && !W) || (needs_input_grad(1) && !x) || !grad_out)
        ErrorBuilder("linear").fail("graph-mode backward is missing a saved input");

    const FlatX flat = flatten_x(input_shapes_[0]);
    const auto M = static_cast<std::int64_t>(flat.M);
    const auto K = static_cast<std::int64_t>(flat.K);
    const std::int64_t N = input_shapes_[1][0];

    // Collapse the leading batch dims so all three products are plain 2-D
    // GEMMs, exactly as the fused kernel does internally.
    auto g_flat = reshape_op(grad_out, {M, N});

    // dx = g W, back to x's original shape.
    TensorImplPtr dx;
    if (needs_input_grad(0)) {
        std::vector<std::int64_t> x_shape(input_shapes_[0].begin(), input_shapes_[0].end());
        dx = reshape_op(matmul_op(g_flat, W), x_shape);
    }
    // dW = g^T x.
    TensorImplPtr dW =
        needs_input_grad(1) ? matmul_op(transpose_op(g_flat), reshape_op(x, {M, K})) : nullptr;
    // db = the gradient summed over every row.  A caller that passed an
    // empty bias (the C++ unbiased form) gets a matching empty gradient
    // rather than a length-N one it has no slot for.
    TensorImplPtr db;
    if (needs_input_grad(2)) {
        db = shape_numel(input_shapes_[2]) == 0
                 ? zeros_like_op(b)
                 : sum_op(g_flat, std::vector<int>{0}, /*keepdims=*/false);
    }

    return {dx, dW, db};
}

TensorImplPtr linear_op(const TensorImplPtr& x, const TensorImplPtr& W, const TensorImplPtr& b) {
    return LinearBackward::forward(x, W, b);
}

LUCID_REGISTER_OP(LinearBackward)

}  // namespace lucid
