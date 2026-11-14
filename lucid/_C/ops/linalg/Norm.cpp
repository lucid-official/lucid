// lucid/_C/ops/linalg/Norm.cpp
//
// Implementation of the tensor norm op and its autograd backward node.
//
// Forward: dispatches to IBackend::linalg_norm() via Dispatcher.
//   CPU path: Apple Accelerate vDSP routines for dot products / sum-of-abs.
//   GPU path: MLX reduction ops (sum, abs, sqrt as needed).
//
// Backward:
//   ord=2: ∂L/∂A = (A / clip(N, ε, ∞)) ⊙ expand(∂L/∂N)
//          Derived from d/dA [sqrt(sum Aᵢ²)] = A / ‖A‖.
//   ord=1: ∂L/∂A = sign(A) ⊙ expand(∂L/∂N)
//          Derived from d/dA [sum |Aᵢ|] = sign(Aᵢ).
//
// The expand_back lambda inside grad_from() handles the general case of
// re-expanding the reduced gradient to the original shape regardless of
// whether keepdims was true/false or the reduction was partial/full.
//
// File-local reduced_shape() computes the output shape from the input shape
// and the (axes, keepdims) arguments; it is called both in norm_op (to
// construct the output TensorImpl) and implicitly drives the backward shape
// inference via out_shape_.

#include "Norm.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <variant>
#include <vector>

#include "../../autograd/Engine.h"
#include "../../autograd/Helpers.h"
#include "../../backend/Dispatcher.h"
#include "../../compile/Tracer.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/SchemaGuard.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../../ops/bfunc/Compare.h"
#include "../../ops/bfunc/Div.h"
#include "../../ops/bfunc/Mul.h"
#include "../../ops/gfunc/Gfunc.h"
#include "../../ops/ufunc/Arith.h"
#include "../../ops/ufunc/Astype.h"
#include "../../ops/ufunc/ScalarParam.h"
#include "../../ops/utils/Layout.h"
#include "../../ops/utils/Select.h"
#include "../../ops/utils/View.h"
#include "_Detail.h"

namespace lucid {

// AmpPolicy::KeepInput: preserve the input dtype so that the A/N division in
// the backward does not silently lose precision through AMP downcasting.
const OpSchema NormBackward::schema_v1{"norm", 1, AmpPolicy::KeepInput, true, "", true};

namespace {

// ``t`` — shaped like a reduction over ``axis`` of a ``full``-shaped tensor — with
// every reduced axis present as size 1, so a binary op broadcasts it against the
// full shape.  Two cases:
//   - keepdims or a full reduction (axis empty): the dimensions are all
//     present (possibly all-ones), so ``t`` is already that.
//   - otherwise the missing axes are re-inserted.  Indices are normalised
//     (negative -> positive) and sorted ascending so that each unsqueeze_op
//     call inserts at the right logical position even as previously inserted
//     axes shift subsequent dimension indices.
TensorImplPtr keepdim_view(const TensorImplPtr& t,
                           const Shape& full,
                           const std::vector<int>& axis,
                           bool keepdims) {
    if (keepdims || axis.empty())
        return t;
    const int ndim = static_cast<int>(full.size());
    std::vector<int> sorted_axes;
    for (int a : axis)
        sorted_axes.push_back(a < 0 ? a + ndim : a);
    std::sort(sorted_axes.begin(), sorted_axes.end());
    auto result = t;
    for (int a : sorted_axes)
        result = unsqueeze_op(result, a);
    return result;
}

// ``t`` broadcast back to ``full``, re-inserting the reduced axes first.
TensorImplPtr expand_reduced(const TensorImplPtr& t,
                             const Shape& full,
                             const std::vector<int>& axis,
                             bool keepdims) {
    return broadcast_to_op(keepdim_view(t, full, axis, keepdims), full);
}

// The eager gradient of a norm node: ``grad_from`` over detached views of the
// saved input ``a``, the saved norm ``n`` and the incoming gradient.
template <class GradFrom>
std::vector<Storage> detached_norm_grad(const Storage& a,
                                        const Shape& a_shape,
                                        const Storage& n,
                                        const Shape& n_shape,
                                        Storage grad_out,
                                        Dtype dt,
                                        Device device,
                                        GradFrom&& grad_from) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;
    auto A = fresh(Storage{a}, a_shape, dt, device);
    auto N = fresh(Storage{n}, n_shape, dt, device);
    auto dN = fresh(std::move(grad_out), n_shape, dt, device);
    return {grad_from(A, N, dN)->storage()};
}

}  // namespace

// Backward pass for norm_op.
TensorImplPtr NormBackward::grad_from(const TensorImplPtr& A,
                                      const TensorImplPtr& N,
                                      const TensorImplPtr& dN) const {
    auto expand_back = [&](const TensorImplPtr& t) {
        return expand_reduced(t, input_shapes_[0], axis_, keepdims_);
    };

    if (ord_ == 2.0) {
        // L2 gradient: dA = (A / clip(N, ε)) * expand(dN)
        // The clip prevents NaN when A = 0 (‖A‖ = 0 means the gradient of ‖A‖
        // is technically undefined; we follow reference framework in returning 0 there).
        auto N_exp = expand_back(clip_op(N, 1e-12, 1e30));
        auto dN_exp = expand_back(dN);
        return mul_op(div_op(A, N_exp), dN_exp);
    }
    if (ord_ == 1.0) {
        // L1 gradient: dA = sign(A) * expand(dN)
        return mul_op(sign_op(A), expand_back(dN));
    }
    ErrorBuilder("norm_backward").not_implemented("gradient only implemented for ord=1 and ord=2");
    return nullptr;
}

std::vector<Storage> NormBackward::apply(Storage grad_out) {
    return detached_norm_grad(
        saved_inputs_[0], input_shapes_[0], saved_output_, out_shape_, std::move(grad_out), dtype_,
        device_,
        [this](const auto& A, const auto& N, const auto& dN) { return grad_from(A, N, dN); });
}

std::vector<TensorImplPtr> NormBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // The norm is recomputed from the tracked input so ord=2's A / N carries
    // N's own dependence on A into the second derivative; ord=1's sign(A)
    // has derivative zero but stays in the graph, as sign is an op of A.
    const auto& A = saved_impl_inputs_[0];
    if (!A)
        ErrorBuilder("norm").fail("graph-mode backward is missing its saved input");
    auto N = norm_op(A, ord_, axis_, keepdims_);
    return {zero_derivative_through(grad_from(A, N, grad_out), next_edges()[0])};
}

// Register NormBackward in the global op registry.
LUCID_REGISTER_OP(NormBackward)

namespace {

// Compute the output shape after reducing sh over axes with optional keepdims.
//
// If axes is empty the entire tensor collapses: to [] (scalar) without
// keepdims, or to a rank-N all-ones shape with keepdims.
// Otherwise, each reduced axis is removed or replaced by 1 depending on
// keepdims.  Negative axis values are normalised to positive here.
Shape reduced_shape(const Shape& sh, const std::vector<int>& axes, bool keepdims) {
    if (axes.empty()) {
        if (keepdims)
            return Shape(sh.size(), 1);  // keep all dims as size-1
        return Shape{};                  // global reduction to scalar
    }
    // Mark which dimensions are reduced.
    std::vector<bool> mask(sh.size(), false);
    for (int a : axes) {
        int p = a < 0 ? a + static_cast<int>(sh.size()) : a;
        mask[p] = true;
    }
    Shape out;
    for (std::size_t i = 0; i < sh.size(); ++i) {
        if (mask[i]) {
            if (keepdims)
                out.push_back(1);  // collapsed but kept as a stub
            // else: dimension is simply removed from the output shape
        } else {
            out.push_back(sh[i]);  // non-reduced dimension passes through
        }
    }
    return out;
}

}  // namespace

// Compute the p-norm of tensor a over the specified axes.
//
// The backward hyperparameters (ord, axis, keepdims) are stored on the
// NormBackward node so it can reconstruct expand_back at backward time.
// saved_output_ captures the forward norm value N so the backward can
// compute A/N without re-running the forward.
// save_inputs=true stores A in saved_inputs_[0] for the L2 numerator.
TensorImplPtr norm_op(const TensorImplPtr& a_in, double ord, std::vector<int> axis, bool keepdims) {
    using namespace linalg_detail;
    Validator::input(a_in, "norm.a").non_null();
    SchemaGuard sg{NormBackward::schema_v1, a_in->dtype(), a_in->device()};
    const TensorImplPtr a = astype_op(a_in, sg.effective_dtype());
    require_float(a->dtype(), "norm");
    // Checked here, for both devices: the CPU kernel took an axis the input
    // lacks without a word — ``dims=[1]`` on a 1-D input returned ``|x|``
    // unreduced — while Metal raised MLX's own "Invalid axis".
    // A 0-d input's one axis is notional (``wrap_dim``): norm over it is
    // the norm over nothing, the empty axis list.
    distinct_dims(axis, static_cast<int>(a->shape().size()), "norm");
    if (a->shape().empty())
        axis.clear();
    Shape out_shape = reduced_shape(a->shape(), axis, keepdims);
    OpScopeFull scope{"norm", a->device(), a->dtype(), out_shape};
    scope.set_attr("ord", ord);
    {
        std::vector<std::int64_t> ax_v;
        ax_v.reserve(axis.size());
        for (int v : axis)
            ax_v.push_back(static_cast<std::int64_t>(v));
        scope.set_attr("axis_list", std::move(ax_v));
    }
    scope.set_attr("keepdims", keepdims);

    Storage out_storage =
        backend::Dispatcher::for_device(a->device())
            .linalg_norm(a->storage(), a->shape(), ord, axis, keepdims, a->dtype());
    auto out = fresh(std::move(out_storage), out_shape, a->dtype(), a->device());
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, out);
    }
    auto bwd = std::make_shared<NormBackward>();
    // Copy the hyperparameters onto the backward node.
    bwd->ord_ = ord;
    bwd->axis_ = axis;
    bwd->keepdims_ = keepdims;
    bwd->keep_output(out->storage());
    // save_inputs=true: NormBackward reads A from saved_inputs_[0] for ord=2.
    kernel::NaryKernel<NormBackward, 1>::wire_autograd(std::move(bwd), {a}, out, true);
    return out;
}

// ``internal``: ``lucid.linalg.vector_norm`` is the one public entry, and it
// reaches this kernel only through ``_p_norm`` (H8: one canonical path).
const OpSchema PNormBackward::schema_v1{"p_norm", 1,   AmpPolicy::KeepInput, true, "", 1, 1, {},
                                        true,     true};

TensorImplPtr PNormBackward::grad_from(const TensorImplPtr& A,
                                       const TensorImplPtr& N,
                                       const TensorImplPtr& dN) const {
    // A zero norm has the subgradient 0 (its ratio is 0 / 0); a NaN norm is not
    // zero and keeps its NaN.  Both are settled on the reduced shape: a zero
    // slice divides by 1 and takes a zero cotangent, so its gradient is
    // ``0 * 0`` with no full-size mask, and the derivative through the cotangent
    // and the norm is 0 there, as the masked gradient's was.
    auto zero = equal_op(N, zeros_like_op(N));
    auto n_safe = masked_fill_op(N, zero, 1.0);
    auto d_live = masked_fill_op(dN, zero, 0.0);
    const Shape& full = input_shapes_[0];
    // ``u = A / N``, in [-1, 1]: the slope never forms a power of the norm.
    auto u = div_op(A, keepdim_view(n_safe, full, axis_, keepdims_));
    // ``sign(u) |u|^(p-1)``, written ``u |u|^(p-2)`` where that is as good: one op
    // fewer.  From p = 3 up, and from p > 2 when nothing is recorded, the slope of
    // ``|u|^(p-2)`` is finite at 0 (or only the value is wanted).  In (2, 3) a
    // recorded graph keeps ``|u|^(p-1)``, whose slope ``(p-1) |u|^(p-2)`` is finite
    // there, where ``|u|^(p-2)``'s infinite slope would meet the zero factor ``u``.
    // Below 2 even ``|u|^(p-1)`` has an infinite slope at a zero entry, and that
    // ``inf * 0`` (through ``sign``'s zero slope) is a NaN that the cotangent sum of
    // ``N`` carries to every entry of the slice, so a recorded graph takes the power
    // of 1 at a zero entry instead: its slope is ``sign(0) = 0`` either way, the
    // value is the same, and only that entry's own second derivative differs (0,
    // where the true Hessian is infinite).
    const bool recorded = GradMode::is_enabled();
    TensorImplPtr slope;
    if (p_ == 2.0) {
        slope = u;
    } else if (p_ == 3.0) {
        slope = mul_op(u, abs_op(u));
    } else if (p_ > 3.0 || (p_ > 2.0 && !recorded)) {
        slope = mul_op(u, pow_scalar_op(abs_op(u), p_ - 2.0));
    } else {
        auto magnitude = abs_op(u);
        if (recorded && p_ < 2.0)
            magnitude =
                masked_fill_op(magnitude, equal_op(magnitude, zeros_like_op(magnitude)), 1.0);
        slope = mul_op(sign_op(u), pow_scalar_op(magnitude, p_ - 1.0));
    }
    return mul_op(slope, keepdim_view(d_live, full, axis_, keepdims_));
}

std::vector<Storage> PNormBackward::apply(Storage grad_out) {
    return detached_norm_grad(
        saved_inputs_[0], input_shapes_[0], saved_output_, out_shape_, std::move(grad_out), dtype_,
        device_,
        [this](const auto& A, const auto& N, const auto& dN) { return grad_from(A, N, dN); });
}

std::vector<TensorImplPtr> PNormBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // The norm is recomputed from the tracked input, so the ratio |A| / N
    // carries N's own dependence on A into the second derivative.
    const auto& A = saved_impl_inputs_[0];
    if (!A)
        ErrorBuilder("p_norm").fail("graph-mode backward is missing its saved input");
    auto N = p_norm_op(A, p_, axis_, keepdims_);
    return {zero_derivative_through(grad_from(A, N, grad_out), next_edges()[0])};
}

LUCID_REGISTER_OP(PNormBackward)

TensorImplPtr p_norm_op(const TensorImplPtr& a_in, double p, std::vector<int> axis, bool keepdims) {
    using namespace linalg_detail;
    Validator::input(a_in, "p_norm.a").non_null();
    if (!(p > 1.0) || !std::isfinite(p))
        ErrorBuilder("p_norm").invalid_argument("p must be finite and greater than 1, got " +
                                                std::to_string(p));
    SchemaGuard sg{PNormBackward::schema_v1, a_in->dtype(), a_in->device()};
    const TensorImplPtr a = astype_op(a_in, sg.effective_dtype());
    require_float(a->dtype(), "p_norm");
    axis = normalize_axes(axis, static_cast<int>(a->shape().size()), "p_norm");
    const Shape out_shape = reduced_shape(a->shape(), axis, keepdims);
    OpScopeFull scope{"p_norm", a->device(), a->dtype(), out_shape};
    scope.set_attr("p", p);
    {
        std::vector<std::int64_t> ax_v(axis.begin(), axis.end());
        scope.set_attr("axis_list", std::move(ax_v));
    }
    scope.set_attr("keepdims", keepdims);

    Storage out_storage =
        backend::Dispatcher::for_device(a->device())
            .p_norm_forward(a->storage(), a->shape(), p, axis, keepdims, a->dtype());
    auto out = fresh(std::move(out_storage), out_shape, a->dtype(), a->device());
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, out);
    }
    auto bwd = std::make_shared<PNormBackward>();
    bwd->p_ = p;
    bwd->axis_ = axis;
    bwd->keepdims_ = keepdims;
    bwd->keep_output(out->storage());
    kernel::NaryKernel<PNormBackward, 1>::wire_autograd(std::move(bwd), {a}, out, true);
    return out;
}

}  // namespace lucid
