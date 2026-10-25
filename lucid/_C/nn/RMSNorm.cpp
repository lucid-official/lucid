// lucid/_C/nn/RMSNorm.cpp
//
// Implementation of Root Mean Square Layer Normalization.
//
// Shape resolution is performed inline (not via a helper): gamma's shape must
// match the trailing Dn dims of x, and the outer/N split is computed manually.
// Forward calls IBackend::rms_norm_forward, returning {y, rstd}.
// Backward calls IBackend::rms_norm_backward, returning {dx, d_gamma}.
// FLOP estimate: 4 * outer * N (rms + scale).

#include "RMSNorm.h"

#include <vector>

#include "../backend/Dispatcher.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/OpRegistry.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/Add.h"
#include "../ops/bfunc/Mul.h"
#include "../ops/bfunc/Sub.h"
#include "../ops/gfunc/Gfunc.h"
#include "../ops/ufunc/Astype.h"
#include "../ops/ufunc/Exponential.h"
#include "../ops/ufunc/Reductions.h"

namespace lucid {

const OpSchema RMSNormBackward::schema_v1{"rms_norm", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr
RMSNormBackward::forward(const TensorImplPtr& x, const TensorImplPtr& gamma, double eps) {
    Validator::input(x, "rms_norm.x").non_null();
    Validator::input(gamma, "rms_norm.weight").non_null();
    Validator::pair(x, gamma, "rms_norm: x/weight").same_device();

    // 3.4+ Phase A.8: AMP plumbing (same as BatchNorm / LayerNorm).
    // schema_v1.amp_policy = ForceFP32 — under autocast(F16) we upcast x
    // and gamma to F32 before the kernel so RMS variance over normalised
    // dims doesn't lose precision.
    SchemaGuard sg{RMSNormBackward::schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr x_eff = astype_op(x, eff_dt);
    const TensorImplPtr gamma_eff = astype_op(gamma, eff_dt);

    // Validate that gamma's shape matches the trailing dims of x.
    if (gamma_eff->shape().size() > x_eff->shape().size())
        throw ShapeMismatch(x_eff->shape(), gamma_eff->shape(), "rms_norm: γ has more dims than x");
    const std::size_t Dn = gamma_eff->shape().size();
    const std::size_t lead = x_eff->shape().size() - Dn;
    for (std::size_t i = 0; i < Dn; ++i) {
        if (x_eff->shape()[lead + i] != gamma_eff->shape()[i]) {
            throw ShapeMismatch(x_eff->shape(), gamma_eff->shape(),
                                "rms_norm: γ must match trailing dims of x");
        }
    }
    std::size_t outer = 1, N = 1;
    for (std::size_t i = 0; i < lead; ++i)
        outer *= static_cast<std::size_t>(x_eff->shape()[i]);
    for (std::size_t i = 0; i < Dn; ++i)
        N *= static_cast<std::size_t>(gamma_eff->shape()[i]);

    OpScopeFull scope{schema_v1.name, x_eff->device(), eff_dt, x_eff->shape()};
    // 3.5 Phase 1.2: report eps for the compile-path RMSNorm emitter.
    scope.set_attr("eps", eps);

    // rms_norm_forward returns {y, rstd}.
    auto forward = backend::Dispatcher::for_device(x_eff->device())
                       .rms_norm_forward(x_eff->storage(), gamma_eff->storage(), outer, N, eps,
                                         x_eff->shape(), eff_dt);
    scope.set_flops(static_cast<std::int64_t>(outer * N) * 4);

    auto out = std::make_shared<TensorImpl>(std::move(forward.first), x_eff->shape(), eff_dt,
                                            x_eff->device(), false);

    auto bwd = std::make_shared<RMSNormBackward>();
    bwd->saved_rstd_ = std::move(forward.second);
    bwd->outer_ = outer;
    bwd->N_ = N;
    bwd->eps_ = eps;
    // saved_inputs_[0..1] hold {x, gamma} at eff_dt.
    kernel::NaryKernel<RMSNormBackward, 2>::wire_autograd(std::move(bwd), {x_eff, gamma_eff}, out);
    return out;
}

std::vector<Storage> RMSNormBackward::apply(Storage grad_out) {
    // rms_norm_backward returns {dx, d_gamma} as a pair.
    auto grads = backend::Dispatcher::for_device(device_).rms_norm_backward(
        saved_inputs_[0], saved_inputs_[1], saved_rstd_, grad_out, outer_, N_, input_shapes_[0],
        input_shapes_[1], dtype_);
    return {std::move(grads.first), std::move(grads.second)};
}

std::vector<TensorImplPtr> RMSNormBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // y = x * rstd * gamma, rstd = 1 / sqrt(mean(x^2) + eps) over gamma's
    // trailing axes.  With xhat = x * rstd and gg = g * gamma:
    //   dx     = rstd * (gg - xhat * mean(gg * xhat))
    //   dgamma = sum over the leading axes of g * xhat
    const auto& x = saved_impl_inputs_[0];
    const auto& gamma = saved_impl_inputs_[1];
    if (!x || !gamma)
        ErrorBuilder("rms_norm").fail("graph-mode backward is missing its saved inputs");
    const int nd = static_cast<int>(x->shape().size());
    const int dn = static_cast<int>(gamma->shape().size());
    std::vector<int> normed, leading;
    for (int d = 0; d < nd; ++d)
        (d < nd - dn ? leading : normed).push_back(d);

    auto mean_sq = mean_op(mul_op(x, x), normed, true);
    auto rstd = rsqrt_op(add_op(mean_sq, full_like_op(mean_sq, eps_)));
    auto xhat = mul_op(x, rstd);
    auto gg = mul_op(grad_out, gamma);
    auto dx = mul_op(rstd, sub_op(gg, mul_op(xhat, mean_op(mul_op(gg, xhat), normed, true))));
    TensorImplPtr dgamma = mul_op(grad_out, xhat);
    if (!leading.empty())
        dgamma = sum_op(dgamma, leading, false);
    return {dx, dgamma};
}

TensorImplPtr rms_norm_op(const TensorImplPtr& x, const TensorImplPtr& gamma, double eps) {
    return RMSNormBackward::forward(x, gamma, eps);
}

LUCID_REGISTER_OP(RMSNormBackward)

}  // namespace lucid
