// lucid/_C/ops/bfunc/Dot.cpp
//
// Implements the 1-D dot product and 2-D matrix-multiply cases of dot_op, each
// with a dedicated backward node.

#include "Dot.h"

#include <variant>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/AutogradNode.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/NaryKernel.h"
#include "../ufunc/Transpose.h"
#include "../utils/Promote.h"
#include "Matmul.h"
#include "Mul.h"
#include "_BinaryOp.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

// Backward node for the 1-D inner product: c = sum(a * b).
//
// The forward output is a scalar (empty shape).  During backward, the scalar
// gradient is first broadcast back to the vector shape, then element-wise
// multiplied by the other operand to recover the per-element gradient:
//   dA[i] = grad_out * b[i]
//   dB[i] = grad_out * a[i]
class Dot1DBackward : public AutogradNode<Dot1DBackward, 2> {
public:
    static const OpSchema schema_v1;
    // Each operand's gradient reads the other operand alone.
    static constexpr GradReads<2> kGradReads{{{false, true}, {true, false}}};

    std::size_t numel_;  // Length of both input vectors.

    // Broadcast the scalar grad_out to [numel_], then multiply by the other
    // saved operand to get per-element gradients.
    std::vector<Storage> apply(Storage grad_out) override {
        Shape vec_shape{static_cast<std::int64_t>(numel_)};
        auto& be = backend::Dispatcher::for_device(device_);
        // Scalar gradient (shape {}) → vector shape {numel_}.
        Storage scaled_grad = be.broadcast(grad_out, Shape{}, vec_shape, dtype_);
        Storage da = needs_input_grad(0) ? be.mul(saved_inputs_[1], scaled_grad, vec_shape, dtype_)
                                         : Storage{};
        Storage db = needs_input_grad(1) ? be.mul(saved_inputs_[0], scaled_grad, vec_shape, dtype_)
                                         : Storage{};
        return {std::move(da), std::move(db)};
    }

    // The same two products, recorded: each operand's gradient is the other
    // operand, so the second derivative of a dot is its cross term — the
    // one the storage path above cannot carry.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        const auto& a = saved_impl_inputs_[0];
        const auto& b = saved_impl_inputs_[1];
        if ((needs_input_grad(0) && !b) || (needs_input_grad(1) && !a))
            ErrorBuilder("dot").fail("graph-mode backward is missing its saved inputs");
        TensorImplPtr da = needs_input_grad(0) ? mul_op(grad_out, b) : nullptr;
        TensorImplPtr db = needs_input_grad(1) ? mul_op(grad_out, a) : nullptr;
        return {std::move(da), std::move(db)};
    }
};

// Backward node for the 2-D matrix multiply: C = A @ B.
//
// Identical in structure to MatmulBackward but without batch dimensions.
//   dA = grad_out @ B^T   [M×N @ N×K = M×K]
//   dB = A^T @ grad_out   [K×M @ M×N = K×N]
class Dot2DBackward : public AutogradNode<Dot2DBackward, 2> {
public:
    static const OpSchema schema_v1;
    // Each operand's gradient reads the other operand alone.
    static constexpr GradReads<2> kGradReads{{{false, true}, {true, false}}};

    Shape a_shape_, b_shape_;

    // Compute dA = grad_out @ B^T and dB = A^T @ grad_out.
    std::vector<Storage> apply(Storage grad_out) override {
        const std::int64_t M = a_shape_[0], K = a_shape_[1], N = b_shape_[1];

        // grad_out [M×N] @ B^T → dA [M×K]
        backend::MatmulOpts da_opts;
        da_opts.M = static_cast<int>(M);
        da_opts.K = static_cast<int>(N);
        da_opts.N = static_cast<int>(K);
        da_opts.transB = true;

        // A^T @ grad_out [M×N] → dB [K×N]
        backend::MatmulOpts db_opts;
        db_opts.M = static_cast<int>(K);
        db_opts.K = static_cast<int>(M);
        db_opts.N = static_cast<int>(N);
        db_opts.transA = true;

        auto& be = backend::Dispatcher::for_device(device_);
        Storage da = needs_input_grad(0) ? be.matmul(grad_out, saved_inputs_[1], da_opts, dtype_)
                                         : Storage{};
        Storage db = needs_input_grad(1) ? be.matmul(saved_inputs_[0], grad_out, db_opts, dtype_)
                                         : Storage{};
        return {std::move(da), std::move(db)};
    }

    // Graph mode: the same two products as matmuls, differentiable again.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        const auto& a = saved_impl_inputs_[0];
        const auto& b = saved_impl_inputs_[1];
        if ((needs_input_grad(0) && !b) || (needs_input_grad(1) && !a))
            ErrorBuilder("dot").fail("graph-mode backward is missing a saved input");
        const std::vector<int> swap{1, 0};
        TensorImplPtr da = needs_input_grad(0) ? matmul_op(grad_out, permute_op(b, swap)) : nullptr;
        TensorImplPtr db = needs_input_grad(1) ? matmul_op(permute_op(a, swap), grad_out) : nullptr;
        return {std::move(da), std::move(db)};
    }
};

// AmpPolicy::KeepInput: do not promote dtype.  Dot products on integer types
// are valid; forcing FP promotion would silently change the result type.
const OpSchema Dot1DBackward::schema_v1{"dot_1d", 1, AmpPolicy::KeepInput, true};
const OpSchema Dot2DBackward::schema_v1{"dot_2d", 1, AmpPolicy::KeepInput, true};

}  // namespace

TensorImplPtr dot_op(const TensorImplPtr& a_in, const TensorImplPtr& b_in) {
    // The vector dot's row is WIDEST under Metal autocast: a float16 and a
    // float32 vector meet in float32, as the reference's do, instead of
    // failing the pair check.  The matrix form has no row.
    const bool vectors = a_in && b_in && a_in->shape().size() == 1 && b_in->shape().size() == 1;
    const auto operands = vectors ? promote_for_schema(Dot1DBackward::schema_v1, {a_in, b_in})
                                  : std::vector<TensorImplPtr>{a_in, b_in};
    const TensorImplPtr& a = operands[0];
    const TensorImplPtr& b = operands[1];
    validate_pair(a, b, "dot");
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"dot", device, dt, Shape{}};

    // wire_grad attaches the appropriate backward node to 'out' after the
    // forward storage has been computed.  Defined as a lambda to share the
    // autograd wiring code between the 1-D and 2-D branches.
    auto wire_grad = [&](const TensorImplPtr& out) {
        if (a->shape().size() == 1 && b->shape().size() == 1) {
            auto n = std::make_shared<Dot1DBackward>();
            n->numel_ = static_cast<std::size_t>(a->shape()[0]);
            kernel::NaryKernel<Dot1DBackward, 2>::wire_autograd(std::move(n), {a, b}, out);
        } else if (a->shape().size() == 2 && b->shape().size() == 2) {
            auto n = std::make_shared<Dot2DBackward>();
            n->a_shape_ = a->shape();
            n->b_shape_ = b->shape();
            kernel::NaryKernel<Dot2DBackward, 2>::wire_autograd(std::move(n), {a, b}, out);
        }
    };

    // 1-D case: inner product → scalar.
    //   1. Element-wise multiply (reuses the mul backend primitive).
    //   2. Sum-reduce over axis 0.
    if (a->shape().size() == 1 && b->shape().size() == 1) {
        if (a->shape()[0] != b->shape()[0])
            throw ShapeMismatch(a->shape(), b->shape(), "dot");
        Shape out_shape{};
        auto& be = backend::Dispatcher::for_device(device);
        Storage values = be.mul(a->storage(), b->storage(), a->shape(), dt);
        backend::ReduceOpts opts{{0}, false};
        Storage out = be.reduce_sum(values, a->shape(), opts, dt);
        auto t = fresh(std::move(out), out_shape, dt, device);
        wire_grad(t);
        return t;
    }

    // 2-D case: matrix multiply.
    if (a->shape().size() == 2 && b->shape().size() == 2) {
        const std::int64_t M = a->shape()[0], K = a->shape()[1];
        const std::int64_t Kb = b->shape()[0], N = b->shape()[1];
        if (K != Kb)
            throw ShapeMismatch(a->shape(), b->shape(), "dot");
        Shape out_shape{M, N};
        backend::MatmulOpts opts;
        opts.M = static_cast<int>(M);
        opts.K = static_cast<int>(K);
        opts.N = static_cast<int>(N);
        Storage out =
            backend::Dispatcher::for_device(device).matmul(a->storage(), b->storage(), opts, dt);
        auto t = fresh(std::move(out), out_shape, dt, device);
        wire_grad(t);
        return t;
    }

    ErrorBuilder("dot").not_implemented("CPU supports only 1-D × 1-D and 2-D × 2-D");
}

}  // namespace lucid
