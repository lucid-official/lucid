// lucid/_C/ops/bfunc/Inner.cpp
//
// Implements inner_op.  For gradient-tracked tensors, the inner product is
// expressed as an equivalent einsum string and forwarded to einsum_op so that
// the full einsum autograd machinery handles the backward pass.  For non-
// gradient paths the backend inner primitive is called directly.

#include "Inner.h"

#include <string>
#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../compile/Tracer.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../einops/Einops.h"
#include "../ufunc/Transpose.h"
#include "../utils/View.h"
#include "Matmul.h"
#include "Mul.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

// Build an einsum contraction string equivalent to numpy.inner for tensors
// with rank na and nb.
//
// The last axis of each tensor is labelled 'z' (the contracted axis).  Free
// axes of A receive letters starting at 'a'; free axes of B receive letters
// starting at 'p'.  The two alphabets are kept disjoint so that no free axis
// of A aliases a free axis of B.
//
// Example — inner_einsum_pattern(3, 2):
//   A has axes [a, b, z], B has axes [p, z]  → "abz,pz->abp"
std::string inner_einsum_pattern(std::size_t na, std::size_t nb) {
    std::string a_lhs, b_lhs, rhs;
    for (std::size_t i = 0; i + 1 < na; ++i) {
        char c = static_cast<char>('a' + i);
        a_lhs.push_back(c);
        rhs.push_back(c);
    }
    a_lhs.push_back('z');
    for (std::size_t i = 0; i + 1 < nb; ++i) {
        char c = static_cast<char>('p' + i);
        b_lhs.push_back(c);
        rhs.push_back(c);
    }
    b_lhs.push_back('z');
    return a_lhs + "," + b_lhs + "->" + rhs;
}

}  // namespace

TensorImplPtr inner_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    validate_pair(a, b, "inner");

    const Dtype dt = a->dtype();
    const Device device = a->device();
    const auto& sa = a->shape();
    const auto& sb = b->shape();

    // When either operand is a 0-D scalar, inner(a, b) is elementwise a * b
    // (after refusing bool, matching the non-scalar contract).
    if (dt == Dtype::Bool)
        ErrorBuilder("inner").not_implemented("not implemented for bool");
    if (sa.empty() || sb.empty())
        return mul_op(a, b);
    if (sa.back() != sb.back())
        throw ShapeMismatch(sa, sb, "inner");

    // When gradient tracking is enabled, delegate to einsum_op so that the
    // backward pass is handled automatically by the einsum backward node.
    if (GradMode::is_enabled() && (a->requires_grad() || b->requires_grad())) {
        return einsum_op(inner_einsum_pattern(sa.size(), sb.size()), {a, b});
    }

    // Output shape: all-but-last dims of A followed by all-but-last dims of B.
    Shape out_shape(sa.begin(), sa.end() - 1);
    out_shape.insert(out_shape.end(), sb.begin(), sb.end() - 1);

    // Integers have no BLAS or MLX inner kernel: the CPU refused them and
    // Metal passed MLX's own ValueError through.  The same contraction runs
    // as ``A @ B^T`` over the flattened leading axes, and matmul takes
    // integers on both devices.
    if (!is_floating_point(dt) && !is_complex(dt)) {
        const std::int64_t k = sa.back();
        const std::int64_t rows_a =
            static_cast<std::int64_t>(shape_numel(Shape(sa.begin(), sa.end() - 1)));
        const std::int64_t rows_b =
            static_cast<std::int64_t>(shape_numel(Shape(sb.begin(), sb.end() - 1)));
        auto a2 = reshape_op(a, Shape{rows_a, k});
        auto b2 = reshape_op(b, Shape{rows_b, k});
        return reshape_op(matmul_op(a2, swapaxes_op(b2, 0, 1)), out_shape);
    }

    // Opened only here: the integer path above is a composition whose ops
    // trace themselves, and an ``inner`` scope around them would record an
    // op with no trace I/O, which ``lucid.compile`` refuses.
    OpScopeFull scope{"inner", device, dt, Shape{}};

    if (device == Device::GPU) {
        auto out_storage = backend::Dispatcher::for_device(device).inner(a->storage(), b->storage(),
                                                                         sa, sb, out_shape, dt);
        // MLX may adjust the output shape slightly (e.g. scalar squeeze), so
        // read the actual shape back from the returned GpuStorage.
        const auto& gs = storage_gpu(out_storage);
        Shape actual_shape;
        for (auto d : gs.arr->shape())
            actual_shape.push_back(static_cast<std::int64_t>(d));
        auto result = fresh(std::move(out_storage), std::move(actual_shape), dt, device);
        // The grad path above goes through ``einsum_op`` and traces
        // itself; this one bypasses ``wire_autograd``, which is also what
        // records a traced op's operands.  Without this ``lucid.compile``
        // sees ``inner`` with no inputs and refuses the whole graph.
        if (auto* trc = ::lucid::compile::current_tracer())
            trc->on_op_io({a, b}, result);
        return result;
    }

    auto out_storage = backend::Dispatcher::for_device(device).inner(a->storage(), b->storage(), sa,
                                                                     sb, out_shape, dt);
    auto result = fresh(std::move(out_storage), out_shape, dt, device);
    if (auto* trc = ::lucid::compile::current_tracer())
        trc->on_op_io({a, b}, result);
    return result;
}

}  // namespace lucid
