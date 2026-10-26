// lucid/_C/ops/bfunc/Matmul.cpp
//
// Implements MatmulBackward::forward, MatmulBackward::apply, and the matmul_op
// free function.

#include "Matmul.h"

#include <utility>
#include <variant>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/SchemaGuard.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/BinaryKernel.h"
#include "../../kernel/NaryKernel.h"
#include "../../kernel/primitives/BatchedMatmul.h"
#include "../bfunc/_BinaryOp.h"
#include "../ufunc/Astype.h"
#include "../utils/View.h"

using lucid::kernel::primitives::NdMatmulInfo;
using lucid::kernel::primitives::plan_nd_matmul;

namespace lucid {

const OpSchema MatmulBackward::schema_v1{"matmul", 1, AmpPolicy::Promote, true};

namespace {

// Build a MatmulOpts struct for use with IBackend::matmul from a pre-computed
// NdMatmulInfo, optionally transposing either operand.
backend::MatmulOpts matmul_opts(const NdMatmulInfo& info, bool transA, bool transB) {
    backend::MatmulOpts opts;
    opts.transA = transA;
    opts.transB = transB;
    opts.M = info.M;
    opts.K = info.K;
    opts.N = info.N;
    opts.batch = info.batch;
    return opts;
}

// Return storage broadcast to dst_shape if src_shape differs; otherwise return
// the original storage unchanged to avoid an unnecessary allocation.
Storage broadcast_for_matmul(const Storage& storage,
                             const Shape& src_shape,
                             const Shape& dst_shape,
                             Dtype dt,
                             Device device) {
    if (src_shape == dst_shape)
        return storage;
    return backend::Dispatcher::for_device(device).broadcast(storage, src_shape, dst_shape, dt);
}

}  // namespace

// Compute dA and dB from the upstream gradient tensor.
//
// Given C = A @ B  (shapes [batch, M, K] and [batch, K, N]):
//   dA = grad_out @ B^T   → [batch, M, K]
//   dB = A^T @ grad_out   → [batch, K, N]
//
// Both saved inputs are first broadcast to the shapes that were used during the
// forward pass (info.a_bcast_shape and info.b_bcast_shape).  After the matmuls,
// reduce_grad_to_shape collapses any batch dimensions that were broadcast-
// expanded, recovering tensors with the original input shapes.
std::vector<Storage> MatmulBackward::apply(Storage grad_out) {
    const auto info = plan_nd_matmul(input_shapes_[0], input_shapes_[1]);
    auto& be = backend::Dispatcher::for_device(device_);

    Storage dA_s;
    if (needs_input_grad(0)) {
        Storage b_use = broadcast_for_matmul(saved_inputs_[1], input_shapes_[1], info.b_bcast_shape,
                                             dtype_, device_);
        backend::MatmulOpts dA_opts;
        dA_opts.M = info.M;
        dA_opts.K = info.N;
        dA_opts.N = info.K;
        dA_opts.batch = info.batch;
        dA_opts.transB = true;
        dA_s = be.matmul(grad_out, b_use, dA_opts, dtype_);
        dA_s = reduce_grad_to_shape(dA_s, info.a_bcast_shape, input_shapes_[0], dtype_, device_);
    }

    Storage dB_s;
    if (needs_input_grad(1)) {
        Storage a_use = broadcast_for_matmul(saved_inputs_[0], input_shapes_[0], info.a_bcast_shape,
                                             dtype_, device_);
        backend::MatmulOpts dB_opts;
        dB_opts.M = info.K;
        dB_opts.K = info.M;
        dB_opts.N = info.N;
        dB_opts.batch = info.batch;
        dB_opts.transA = true;
        dB_s = be.matmul(a_use, grad_out, dB_opts, dtype_);
        dB_s = reduce_grad_to_shape(dB_s, info.b_bcast_shape, input_shapes_[1], dtype_, device_);
    }

    return {std::move(dA_s), std::move(dB_s)};
}

namespace {

// The dtype two matmul operands meet at: complex over float over integer
// over bool, and the wider within a kind — the rule the Python operators
// already follow for every other arithmetic op.  Two different 16-bit
// floats meet at float32.
Dtype common_dtype(Dtype a, Dtype b) {
    if (a == b)
        return a;
    const auto kind = [](Dtype d) {
        return is_complex(d) ? 3 : is_floating_point(d) ? 2 : is_integral(d) ? 1 : 0;
    };
    if (kind(a) != kind(b))
        return kind(a) > kind(b) ? a : b;
    if (dtype_size(a) != dtype_size(b))
        return dtype_size(a) > dtype_size(b) ? a : b;
    return Dtype::F32;
}

}  // namespace

// Execute the forward matmul, open a profiler scope, compute FLOPs, and wire
// the backward node if gradient tracking is active.
//
// Validation requires:
//   - both inputs non-null, same dtype and device
//   - both inputs at least 2-D (plan_nd_matmul handles batch broadcasting)
TensorImplPtr MatmulBackward::forward(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (!a || !b)
        ErrorBuilder("matmul").fail("null input");
    if (a->device() != b->device())
        throw DeviceMismatch(std::string(device_name(a->device())),
                             std::string(device_name(b->device())), "matmul");
    if (a->shape().empty() || b->shape().empty())
        throw ShapeMismatch(a->shape(), b->shape(), "matmul: both operands must be at least 1-D");
    // A vector is a matrix of one row on the left or one column on the
    // right, and the axis it gained is dropped from the product again — so
    // ``A @ v`` is a vector, as the reference and NumPy both have it.  The
    // reshapes carry the gradient back to the vector's own shape.
    if (a->shape().size() == 1 || b->shape().size() == 1) {
        const bool row = a->shape().size() == 1;
        const bool col = b->shape().size() == 1;
        const auto out = forward(row ? reshape_op(a, {1, a->shape()[0]}) : a,
                                 col ? reshape_op(b, {b->shape()[0], 1}) : b);
        Shape shape = out->shape();
        if (col)
            shape.pop_back();
        if (row)
            shape.erase(shape.end() - (col ? 1 : 2));
        return reshape_op(out, shape);
    }

    // 3.3 AMP plumbing: schema_v1.amp_policy == Promote.  Under
    // ``AutocastGuard(F16)`` SchemaGuard resolves ``eff_dt`` to the
    // autocast target (F32 on CPU per Accelerate-no-F16 carve-out;
    // F16 on GPU where ``mlx::core::matmul`` runs natively at half
    // throughput).  Cast precedes the dtype-match check so a/b can
    // arrive at differing dtypes under autocast (e.g. a from an upstream
    // Conv cast to F16, b a Parameter still F32).
    //
    // ``astype_op`` is autograd-aware (wires ``AstypeBackward`` when the
    // input has requires_grad), which keeps the backward chain intact
    // through the AMP cast.  ``maybe_cast_for_kernel`` would produce a
    // requires_grad=false cast tensor → wire_autograd would short-circuit
    // → backward would drop every gradient.  No-op fast path on F32.
    // The operands meet at their common dtype.  The guard used to key off
    // ``a`` alone, so ``int @ float`` cast the float operand to int and
    // truncated it (``[[1, 2]] @ [[0.5], [0.25]]`` was 0) while
    // ``float @ int`` was right.
    SchemaGuard sg{MatmulBackward::schema_v1, common_dtype(a->dtype(), b->dtype()), a->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr a_eff = astype_op(a, eff_dt);
    const TensorImplPtr b_eff = astype_op(b, eff_dt);
    if (a_eff->dtype() != b_eff->dtype())
        throw DtypeMismatch(std::string(dtype_name(a_eff->dtype())),
                            std::string(dtype_name(b_eff->dtype())), "matmul");

    const auto info = plan_nd_matmul(a_eff->shape(), b_eff->shape());
    const int M = info.M, N = info.N, K = info.K;

    OpScopeFull scope{MatmulBackward::schema_v1.name, a_eff->device(), eff_dt, info.out_shape};
    // Each output element requires K multiplications and K-1 additions ≈ 2*K MACs.
    scope.set_flops(static_cast<std::int64_t>(2) * info.batch * M * N * K);

    Storage a_use = broadcast_for_matmul(a_eff->storage(), a_eff->shape(), info.a_bcast_shape,
                                         eff_dt, a_eff->device());
    Storage b_use = broadcast_for_matmul(b_eff->storage(), b_eff->shape(), info.b_bcast_shape,
                                         eff_dt, a_eff->device());
    Storage out_storage = backend::Dispatcher::for_device(a_eff->device())
                              .matmul(a_use, b_use, matmul_opts(info, false, false), eff_dt);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), info.out_shape, eff_dt,
                                            a_eff->device(), false);

    kernel::NaryKernel<MatmulBackward, 2>::wire_autograd({a_eff, b_eff}, out);
    return out;
}

// Graph-mode matmul backward: dA = grad_out @ B^T,  dB = A^T @ grad_out.
std::vector<TensorImplPtr> MatmulBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    auto& a = saved_impl_inputs_[0];
    auto& b = saved_impl_inputs_[1];
    if ((needs_input_grad(0) && !b) || (needs_input_grad(1) && !a)) {
        throw std::runtime_error("apply_for_graph: saved_impl_inputs_ not set for matmul.");
    }

    extern TensorImplPtr mT_op(const TensorImplPtr&);
    extern TensorImplPtr sum_op(const TensorImplPtr&, const std::vector<int>&, bool);
    extern TensorImplPtr reshape_op(const TensorImplPtr&, const Shape&);

    auto reduce = [&](TensorImplPtr g, const Shape& target) -> TensorImplPtr {
        if (!g || g->shape() == target)
            return g;
        std::vector<int> axes;
        const int ng = static_cast<int>(g->shape().size());
        const int nt = static_cast<int>(target.size());
        for (int i = 0; i < ng - nt; ++i)
            axes.push_back(i);
        for (int i = 0; i < nt; ++i) {
            if (target[static_cast<std::size_t>(i)] == 1 &&
                g->shape()[static_cast<std::size_t>(i + ng - nt)] != 1)
                axes.push_back(i + ng - nt);
        }
        if (!axes.empty())
            g = sum_op(g, axes, false);
        if (g->shape() != target)
            g = reshape_op(g, target);
        return g;
    };

    TensorImplPtr da =
        needs_input_grad(0) ? reduce(matmul_op(grad_out, mT_op(b)), input_shapes_[0]) : nullptr;
    TensorImplPtr db =
        needs_input_grad(1) ? reduce(matmul_op(mT_op(a), grad_out), input_shapes_[1]) : nullptr;
    return {std::move(da), std::move(db)};
}

TensorImplPtr matmul_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    return MatmulBackward::forward(a, b);
}

LUCID_REGISTER_OP(MatmulBackward)

}  // namespace lucid
