// lucid/_C/nn/GroupNorm.cpp
//
// Implementation of Group Normalization.
//
// Layout expected: (B, C, S_0, ..., S_{ndim-3}) where ndim >= 2.
// The C channels are partitioned into G groups of (C/G) channels each.
// Statistics are computed per (b, g) slice spanning all channels in the
// group plus all spatial positions.
//
// Forward calls IBackend::group_norm_forward, returning [y, mean, rstd].
// Backward calls IBackend::group_norm_backward, returning [dx, d_gamma, d_beta].

#include "GroupNorm.h"

#include <string>
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
#include "../ops/utils/View.h"

namespace lucid {

const OpSchema GroupNormBackward::schema_v1{"group_norm", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr GroupNormBackward::forward(const TensorImplPtr& x,
                                         const TensorImplPtr& gamma,
                                         const TensorImplPtr& beta,
                                         int G,
                                         double eps) {
    if (!x || !gamma || !beta)
        ErrorBuilder("group_norm").fail("null input");
    Validator::pair(x, gamma, "group_norm: x/weight").same_device();
    Validator::pair(x, beta, "group_norm: x/bias").same_device();

    // 3.4+ Phase A.8: AMP plumbing (same as BatchNorm / LayerNorm / RMSNorm).
    // schema_v1.amp_policy = ForceFP32 — cast x / γ / β to F32 under autocast(F16).
    SchemaGuard sg{GroupNormBackward::schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr x_eff = astype_op(x, eff_dt);
    const TensorImplPtr gamma_eff = astype_op(gamma, eff_dt);
    const TensorImplPtr beta_eff = astype_op(beta, eff_dt);

    require_min_rank(x_eff->shape(), 2, "group_norm: x", "(N, C, *)");
    const int B = static_cast<int>(x_eff->shape()[0]);
    const int C = static_cast<int>(x_eff->shape()[1]);
    require_at_least(G, 1, "group_norm: num_groups");
    if (C % G != 0)
        ErrorBuilder("group_norm")
            .invalid_argument("C=" + std::to_string(C) +
                              " must be divisible by num_groups=" + std::to_string(G));
    require_length(gamma_eff->shape(), C, "group_norm: weight");
    require_length(beta_eff->shape(), C, "group_norm: bias");

    const int N_spatial = static_cast<int>(x_eff->shape().size()) - 2;
    std::vector<int> S(N_spatial);
    int spatial_total = 1;
    for (int i = 0; i < N_spatial; ++i) {
        S[i] = static_cast<int>(x_eff->shape()[2 + i]);
        spatial_total *= S[i];
    }
    OpScopeFull scope{schema_v1.name, x_eff->device(), eff_dt, x_eff->shape()};
    // 3.5 Phase 1.2: report eps + num_groups for the compile-path GN emitter.
    scope.set_attr("eps", eps);
    scope.set_attr("num_groups", static_cast<std::int64_t>(G));

    // group_norm_forward returns [y, mean, rstd].
    auto forward =
        backend::Dispatcher::for_device(x_eff->device())
            .group_norm_forward(x_eff->storage(), gamma_eff->storage(), beta_eff->storage(), B, C,
                                spatial_total, G, S, eps, x_eff->shape(), eff_dt);

    auto out = std::make_shared<TensorImpl>(std::move(forward[0]), x_eff->shape(), eff_dt,
                                            x_eff->device(), false);
    auto bwd = std::make_shared<GroupNormBackward>();
    bwd->saved_mean_ = std::move(forward[1]);
    bwd->saved_rstd_ = std::move(forward[2]);
    bwd->B_ = B;
    bwd->C_ = C;
    bwd->G_ = G;
    bwd->spatial_dims_ = std::move(S);
    bwd->eps_ = eps;
    // saved_inputs_[0..1] hold {x, gamma} at eff_dt; beta is read by shape
    // alone and is not saved.
    kernel::NaryKernel<GroupNormBackward, 3>::wire_autograd(
        std::move(bwd), {x_eff, gamma_eff, beta_eff}, out, {true, true, false});
    return out;
}

std::vector<Storage> GroupNormBackward::apply(Storage grad_out) {
    int spatial_total = 1;
    for (int s : spatial_dims_)
        spatial_total *= s;

    // Returns [dx, d_gamma, d_beta].
    return backend::Dispatcher::for_device(device_).group_norm_backward(
        saved_inputs_[0], saved_inputs_[1], saved_mean_, saved_rstd_, grad_out, B_, C_,
        spatial_total, G_, spatial_dims_, input_shapes_[0], dtype_);
}

std::vector<TensorImplPtr> GroupNormBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    const auto& x = saved_impl_inputs_[0];
    const auto& gamma = saved_impl_inputs_[1];
    if (!x || !gamma)
        ErrorBuilder("group_norm").fail("graph-mode backward is missing a saved input");

    // Group normalisation is layer normalisation over a reshaped view: fold
    // (C, *spatial) into (G, C/G * spatial) and the statistics are a plain
    // trailing-axis reduction.  Reshape carries a graph-mode formula, so the
    // whole derivation stays differentiable.
    const Shape& xs = input_shapes_[0];
    std::int64_t spatial = 1;
    for (std::size_t i = 2; i < xs.size(); ++i)
        spatial *= xs[i];
    const std::int64_t per_group = static_cast<std::int64_t>(C_ / G_) * spatial;

    std::vector<std::int64_t> grouped{static_cast<std::int64_t>(B_), static_cast<std::int64_t>(G_),
                                      per_group};
    std::vector<std::int64_t> original(xs.begin(), xs.end());
    const std::vector<int> dims{2};

    auto xg = reshape_op(x, grouped);
    auto mu = mean_op(xg, dims, /*keepdims=*/true);
    auto xc = sub_op(xg, mu);
    auto var = mean_op(mul_op(xc, xc), dims, /*keepdims=*/true);
    auto rstd = rsqrt_op(add_op(var, full_like_op(var, eps_)));
    auto xhat = reshape_op(mul_op(xc, rstd), original);

    // gamma is per-channel: (C,) broadcast over batch and spatial.
    std::vector<std::int64_t> chan(xs.size(), 1);
    chan[1] = static_cast<std::int64_t>(C_);
    auto gamma_b = reshape_op(gamma, chan);

    auto gg = reshape_op(mul_op(grad_out, gamma_b), grouped);
    auto xhat_g = reshape_op(xhat, grouped);
    auto term = sub_op(gg, mean_op(gg, dims, /*keepdims=*/true));
    term = sub_op(term, mul_op(xhat_g, mean_op(mul_op(gg, xhat_g), dims, /*keepdims=*/true)));
    auto dx = reshape_op(mul_op(rstd, term), original);

    // gamma / beta are summed over everything but the channel axis.
    std::vector<int> reduce;
    for (std::size_t i = 0; i < xs.size(); ++i)
        if (i != 1)
            reduce.push_back(static_cast<int>(i));
    auto d_gamma = sum_op(mul_op(grad_out, xhat), reduce, /*keepdims=*/false);
    auto d_beta = sum_op(grad_out, reduce, /*keepdims=*/false);
    return {dx, d_gamma, d_beta};
}

TensorImplPtr group_norm_op(const TensorImplPtr& x,
                            const TensorImplPtr& gamma,
                            const TensorImplPtr& beta,
                            int num_groups,
                            double eps) {
    return GroupNormBackward::forward(x, gamma, beta, num_groups, eps);
}

LUCID_REGISTER_OP(GroupNormBackward)

}  // namespace lucid
