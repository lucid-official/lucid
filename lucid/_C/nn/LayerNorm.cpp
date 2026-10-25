// lucid/_C/nn/LayerNorm.cpp
//
// Implementation of the Layer Normalization forward pass and its autograd node.
//
// Shape resolution: given x of shape (d0, d1, ..., dk, n0, n1, ..., nm) and
// gamma of shape (n0, n1, ..., nm), the operation normalizes each of the
// outer = d0*...*dk slices of length N = n0*...*nm independently.
//
// Forward calls IBackend::layer_norm_forward, which returns [y, mean, rstd].
// Backward calls IBackend::layer_norm_backward, which computes [dx, d_gamma, d_beta].
// FLOP estimate: 5 * outer * N (normalize + affine).

#include "LayerNorm.h"

#include <vector>

#include "../autograd/Helpers.h"
#include "../backend/Dispatcher.h"
#include "../core/AmpPolicy.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
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

const OpSchema LayerNormBackward::schema_v1{"layer_norm", 1, AmpPolicy::ForceFP32, true};

namespace {

// Decomposed shape into (outer, N) where outer is the product of leading dims
// that are not normalized, and N is the product of normalized trailing dims.
struct LayerNormShapes {
    std::size_t outer;
    std::size_t N;
};

// Verify that gamma_shape matches the trailing Dn dims of x_shape and
// compute the outer/N split.
// Throws ShapeMismatch on any mismatch.
LayerNormShapes resolve_shapes(const Shape& x_shape, const Shape& gamma_shape) {
    if (gamma_shape.size() > x_shape.size()) {
        throw ShapeMismatch(x_shape, gamma_shape, "layer_norm: γ has more dims than x");
    }
    const std::size_t Dn = gamma_shape.size();
    const std::size_t lead = x_shape.size() - Dn;
    for (std::size_t i = 0; i < Dn; ++i) {
        if (x_shape[lead + i] != gamma_shape[i]) {
            throw ShapeMismatch(x_shape, gamma_shape,
                                "layer_norm: γ shape must match trailing dims of x");
        }
    }
    std::size_t outer = 1;
    for (std::size_t i = 0; i < lead; ++i)
        outer *= static_cast<std::size_t>(x_shape[i]);
    std::size_t N = 1;
    for (std::size_t i = 0; i < Dn; ++i)
        N *= static_cast<std::size_t>(gamma_shape[i]);
    return {outer, N};
}

}  // namespace

TensorImplPtr LayerNormBackward::forward(const TensorImplPtr& x,
                                         const TensorImplPtr& gamma,
                                         const TensorImplPtr& beta,
                                         double eps) {
    if (!x || !gamma || !beta)
        ErrorBuilder("layer_norm").fail("null input");
    Validator::pair(x, gamma, "layer_norm: x/weight").same_device();
    Validator::pair(x, beta, "layer_norm: x/bias").same_device();

    // 3.4+ AMP plumbing: schema_v1.amp_policy = ForceFP32.  Under
    // ``AutocastGuard(F16)`` SchemaGuard returns ``F32`` regardless of x's
    // input dtype — LayerNorm's per-sample mean/var reductions are
    // numerically sensitive and running them in F16 risks catastrophic
    // cancellation on the variance.  Under autocast the preceding op
    // (typically Embedding with ``AmpPolicy::Promote``) emits F16 while
    // ``gamma``/``beta`` Parameter slots are still F32; after the cast all
    // three operands share ``eff_dt``.  Outside an autocast scope this is a
    // no-op (``astype_op`` returns the input unchanged when dtypes already
    // match).
    //
    // ``astype_op`` (not ``maybe_cast_for_kernel``) is used so the cast
    // tensors carry an ``AstypeBackward`` grad_fn — without that the F32-
    // cast x_eff has requires_grad=false and ``wire_autograd`` would drop
    // the LayerNorm backward chain under AMP.
    SchemaGuard sg{LayerNormBackward::schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr x_eff = astype_op(x, eff_dt);
    const TensorImplPtr gamma_eff = astype_op(gamma, eff_dt);
    const TensorImplPtr beta_eff = astype_op(beta, eff_dt);

    // The kernel reads N entries of β, N being γ's element count.
    if (gamma_eff->shape() != beta_eff->shape())
        throw ShapeMismatch(gamma_eff->shape(), beta_eff->shape(),
                            "layer_norm: γ and β must have the same shape");

    const auto [outer, N] = resolve_shapes(x_eff->shape(), gamma_eff->shape());

    OpScopeFull scope{schema_v1.name, x_eff->device(), eff_dt, x_eff->shape()};
    // 3.5 Phase 1.2: report eps for the compile-path LayerNorm emitter.
    scope.set_attr("eps", eps);

    // layer_norm_forward returns {y, mean, rstd}.
    auto forward =
        backend::Dispatcher::for_device(x_eff->device())
            .layer_norm_forward(x_eff->storage(), gamma_eff->storage(), beta_eff->storage(), outer,
                                N, eps, x_eff->shape(), eff_dt);
    scope.set_flops(static_cast<std::int64_t>(outer * N) * 5);

    auto out = std::make_shared<TensorImpl>(std::move(forward[0]), x_eff->shape(), eff_dt,
                                            x_eff->device(), false);

    auto bwd = std::make_shared<LayerNormBackward>();
    bwd->saved_mean_ = std::move(forward[1]);
    bwd->saved_rstd_ = std::move(forward[2]);
    bwd->outer_ = outer;
    bwd->N_ = N;
    bwd->eps_ = eps;
    // saved_inputs_[0..1] hold {x, gamma} at eff_dt; beta is read by shape
    // alone and is not saved.
    kernel::NaryKernel<LayerNormBackward, 3>::wire_autograd(
        std::move(bwd), {x_eff, gamma_eff, beta_eff}, out, {true, true, false});
    return out;
}

std::vector<Storage> LayerNormBackward::apply(Storage grad_out) {
    // Returns [dx, d_gamma, d_beta].
    return backend::Dispatcher::for_device(device_).layer_norm_backward(
        saved_inputs_[0], saved_inputs_[1], saved_mean_, saved_rstd_, grad_out, outer_, N_,
        input_shapes_[0], input_shapes_[1], input_shapes_[2], dtype_);
}

std::vector<TensorImplPtr> LayerNormBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    const auto& x = saved_impl_inputs_[0];
    const auto& gamma = saved_impl_inputs_[1];
    if (!x || !gamma)
        ErrorBuilder("layer_norm").fail("graph-mode backward is missing a saved input");

    // The normalised axes are the trailing ones gamma spans.
    const std::size_t rank = input_shapes_[0].size();
    const std::size_t norm_rank = input_shapes_[1].size();
    std::vector<int> dims;
    for (std::size_t i = rank - norm_rank; i < rank; ++i)
        dims.push_back(static_cast<int>(i));
    std::vector<int> outer_dims;
    for (std::size_t i = 0; i + norm_rank < rank; ++i)
        outer_dims.push_back(static_cast<int>(i));

    // Rebuild the normalisation rather than reusing the saved mean/rstd:
    // both depend on x, and a Storage stand-in would drop that path.
    auto mu = mean_op(x, dims, /*keepdims=*/true);
    auto xc = sub_op(x, mu);
    auto var = mean_op(mul_op(xc, xc), dims, /*keepdims=*/true);
    auto rstd = rsqrt_op(add_op(var, full_like_op(var, eps_)));
    auto xhat = mul_op(xc, rstd);

    auto gg = mul_op(grad_out, gamma);
    auto term = sub_op(gg, mean_op(gg, dims, /*keepdims=*/true));
    term = sub_op(term, mul_op(xhat, mean_op(mul_op(gg, xhat), dims, /*keepdims=*/true)));
    auto dx = mul_op(rstd, term);

    auto d_gamma = mul_op(grad_out, xhat);
    auto d_beta = grad_out;
    if (!outer_dims.empty()) {
        d_gamma = sum_op(d_gamma, outer_dims, /*keepdims=*/false);
        d_beta = sum_op(d_beta, outer_dims, /*keepdims=*/false);
    }
    return {dx, d_gamma, d_beta};
}

TensorImplPtr layer_norm_op(const TensorImplPtr& x,
                            const TensorImplPtr& gamma,
                            const TensorImplPtr& beta,
                            double eps) {
    return LayerNormBackward::forward(x, gamma, beta, eps);
}

LUCID_REGISTER_OP(LayerNormBackward)

}  // namespace lucid
