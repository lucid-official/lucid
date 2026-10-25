// lucid/_C/nn/NormExt.cpp
//
// Implementations of BatchNormEval, LpNormalize, and GlobalResponseNorm.
//
// BatchNormEval: uses externally supplied running statistics; does not update
//   them.  IBackend::batch_norm_eval_forward returns [y, rstd].
//   Backward: IBackend::batch_norm_eval_backward returns [dx, d_mean, d_var,
//   d_gamma, d_beta] (mean/var grads are zero in standard BN eval).
//
// LpNormalize: IBackend::lp_normalize_forward returns [y, norm].
//   Backward: IBackend::lp_normalize_backward returns dx.
//
// GlobalResponseNorm: IBackend::global_response_norm_forward returns [y, Nx].
//   Backward: IBackend::global_response_norm_backward returns [dx, d_gamma, d_beta].

#include "NormExt.h"

#include <cmath>
#include <limits>
#include <vector>

#include "../autograd/Engine.h"
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
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/Compare.h"
#include "../ops/bfunc/Div.h"
#include "../ops/bfunc/Mul.h"
#include "../ops/bfunc/Sub.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/gfunc/Gfunc.h"
#include "../ops/ufunc/Arith.h"
#include "../ops/ufunc/Astype.h"
#include "../ops/ufunc/Exponential.h"
#include "../ops/ufunc/Reductions.h"
#include "../ops/ufunc/ScalarParam.h"
#include "../ops/utils/Layout.h"
#include "../ops/utils/Promote.h"
#include "../ops/utils/ScalarAxis.h"
#include "../ops/utils/Select.h"
#include "../ops/utils/View.h"

namespace lucid {

const OpSchema BatchNormEvalBackward::schema_v1{"batch_norm_eval", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr BatchNormEvalBackward::forward(const TensorImplPtr& x,
                                             const TensorImplPtr& mean,
                                             const TensorImplPtr& var,
                                             const TensorImplPtr& gamma,
                                             const TensorImplPtr& beta,
                                             double eps) {
    if (!x || !mean || !var || !gamma || !beta)
        ErrorBuilder("batch_norm_eval").fail("null input");
    Validator::pair(x, mean, "batch_norm_eval: x/running_mean").same_device();
    Validator::pair(x, var, "batch_norm_eval: x/running_var").same_device();
    Validator::pair(x, gamma, "batch_norm_eval: x/weight").same_device();
    Validator::pair(x, beta, "batch_norm_eval: x/bias").same_device();
    require_min_rank(x->shape(), 2, "batch_norm_eval: x", "(N, C, *)");

    const int C = static_cast<int>(x->shape()[1]);
    require_length(mean->shape(), C, "batch_norm_eval: running_mean");
    require_length(var->shape(), C, "batch_norm_eval: running_var");
    require_length(gamma->shape(), C, "batch_norm_eval: weight");
    require_length(beta->shape(), C, "batch_norm_eval: bias");
    int spatial = 1;
    for (std::size_t i = 2; i < x->shape().size(); ++i)
        spatial *= static_cast<int>(x->shape()[i]);

    // 3.3 AMP plumbing: schema_v1.amp_policy == ForceFP32.  Under
    // ``AutocastGuard(F16)`` SchemaGuard returns ``Dtype::F32`` regardless of
    // input dtype — the eval-mode normalisation is numerically sensitive in
    // exactly the same way the training-mode reduction is, and the running
    // ``mean`` / ``var`` buffers live on the ``nn.BatchNorm`` Parameter slots
    // at F32 anyway.  Without this plumbing ``model.eval()`` under
    // ``with amp.autocast(F16):`` would feed F16 activations into F32 running
    // stats — the backend doesn't validate dtype mismatch and silently
    // produces NaN outputs (Mac Studio ResNet-18 / CIFAR-10 train loop
    // converged on train but reported ``test_loss=nan, test_acc=10.00 %``).
    SchemaGuard sg{schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr x_eff = astype_op(x, eff_dt);
    const TensorImplPtr mean_eff = astype_op(mean, eff_dt);
    const TensorImplPtr var_eff = astype_op(var, eff_dt);
    const TensorImplPtr gamma_eff = astype_op(gamma, eff_dt);
    const TensorImplPtr beta_eff = astype_op(beta, eff_dt);

    OpScopeFull scope{schema_v1.name, x_eff->device(), eff_dt, x_eff->shape()};
    // 3.5 Phase 1.2: report eps for the compile-path BatchNormEval emitter.
    scope.set_attr("eps", eps);

    auto& be = backend::Dispatcher::for_device(x_eff->device());
    // batch_norm_eval_forward returns [y, rstd].
    auto fwd = be.batch_norm_eval_forward(x_eff->storage(), mean_eff->storage(), var_eff->storage(),
                                          gamma_eff->storage(), beta_eff->storage(), x_eff->shape(),
                                          C, spatial, eps, eff_dt);
    Storage out_storage = std::move(fwd[0]);
    Storage rstd_storage = std::move(fwd[1]);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), x_eff->shape(), eff_dt,
                                            x_eff->device(), false);

    auto bwd = std::make_shared<BatchNormEvalBackward>();
    bwd->eps_ = eps;
    bwd->rstd_ = std::move(rstd_storage);
    // saved_inputs_ hold {x, mean, gamma} at eff_dt in their slots 0, 1 and 3.
    // ``var`` reaches the backward as ``rstd_``, computed here, and beta is
    // read by shape alone; neither is saved.
    kernel::NaryKernel<BatchNormEvalBackward, 5>::wire_autograd(
        std::move(bwd), {x_eff, mean_eff, var_eff, gamma_eff, beta_eff}, out,
        {true, true, false, true, false});
    return out;
}

std::vector<Storage> BatchNormEvalBackward::apply(Storage grad_out) {
    const Shape& xs = this->input_shapes_[0];
    const int C = static_cast<int>(xs[1]);
    int spatial = 1;
    for (std::size_t i = 2; i < xs.size(); ++i)
        spatial *= static_cast<int>(xs[i]);
    auto& be = backend::Dispatcher::for_device(this->device_);
    // saved_inputs_[3] is gamma (index 3 of the five saved tensors).
    return be.batch_norm_eval_backward(this->saved_inputs_[0], this->saved_inputs_[1],
                                       this->saved_inputs_[3], this->rstd_, grad_out, xs, C,
                                       spatial, this->dtype_);
}

std::vector<TensorImplPtr> BatchNormEvalBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // y = (x - mean) * rstd * gamma + beta with rstd fixed at forward.
    // dx is affine in nothing but gamma, so a second derivative through x
    // is zero; the reference still keeps dx (and dbeta) in the graph — its
    // backward is an op of x — so both carry a zero derivative.
    const auto& edges = this->next_edges();
    auto needs = [&](std::size_t i) { return i < edges.size() && edges[i].is_valid(); };
    if (needs(2))
        ErrorBuilder("batch_norm_eval")
            .not_implemented("create_graph=True through a running_var that requires grad: the "
                             "node keeps only rstd, and the reference does not differentiate "
                             "running statistics either");
    const auto& x = this->saved_impl_inputs_[0];
    const auto& mean = this->saved_impl_inputs_[1];
    const auto& gamma = this->saved_impl_inputs_[3];
    if (!x || !mean || !gamma)
        ErrorBuilder("batch_norm_eval").fail("graph-mode backward is missing a saved input");

    const Shape& xs = this->input_shapes_[0];
    Shape per_channel(xs.size(), 1);
    per_channel[1] = xs[1];
    std::vector<int> axes;
    for (std::size_t i = 0; i < xs.size(); ++i)
        if (i != 1)
            axes.push_back(static_cast<int>(i));

    auto rstd =
        std::make_shared<TensorImpl>(this->rstd_, Shape{xs[1]}, this->dtype_, this->device_, false);
    auto scale = mul_op(gamma, rstd);
    auto sum_g = sum_op(grad_out, axes, false);
    std::vector<TensorImplPtr> grads(5);
    if (needs(0))
        grads[0] =
            zero_derivative_through(mul_op(grad_out, reshape_op(scale, per_channel)), edges[0]);
    if (needs(1))
        grads[1] = neg_op(mul_op(scale, sum_g));
    if (needs(3)) {
        auto centered = sub_op(x, reshape_op(mean, per_channel));
        grads[3] = mul_op(sum_op(mul_op(grad_out, centered), axes, false), rstd);
    }
    // The zero derivative goes back along beta's own edge: it has to have
    // the gradient's shape, and x's is (N, C, *) where dbeta is (C,).
    if (needs(4))
        grads[4] = zero_derivative_through(sum_g, edges[4]);
    return grads;
}

TensorImplPtr batch_norm_eval_op(const TensorImplPtr& x,
                                 const TensorImplPtr& mean,
                                 const TensorImplPtr& var,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps) {
    return BatchNormEvalBackward::forward(x, mean, var, gamma, beta, eps);
}

LUCID_REGISTER_OP(BatchNormEvalBackward)

const OpSchema LpNormalizeBackward::schema_v1{"lp_normalize", 1, AmpPolicy::ForceFP32, true};

namespace {

TensorImplPtr real_non_trivial_p_norm(const TensorImplPtr& r, double ord, int axis) {
    const std::vector<int> axes{axis};
    TensorImplPtr powered;
    if (ord == 2.0) {
        powered = mul_op(r, r);
    } else if (ord > 1.0 || !r->requires_grad()) {
        powered = pow_scalar_op(r, ord);
    } else {
        auto nz = not_equal_op(r, zeros_like_op(r));
        auto safe = where_op(nz, r, ones_like_op(r));
        auto raised = pow_scalar_op(safe, ord);
        auto fill = full_like_op(raised, ord > 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
        powered = where_op(nz, raised, fill);
    }
    auto s = sum_op(powered, axes, true);
    if (ord <= 0.0)
        return pow_scalar_op(s, 1.0 / ord);
    if (!s->requires_grad())
        return (ord == 2.0) ? sqrt_op(s) : pow_scalar_op(s, 1.0 / ord);
    auto at_zero = equal_op(s, zeros_like_op(s));
    auto safe_s = where_op(at_zero, ones_like_op(s), s);
    auto root = (ord == 2.0) ? sqrt_op(safe_s) : pow_scalar_op(safe_s, 1.0 / ord);
    return where_op(at_zero, zeros_like_op(root), root);
}

TensorImplPtr real_p_norm_keepdim(const TensorImplPtr& r, double ord, int axis) {
    const std::vector<int> axes{axis};
    if (ord == 0.0) {
        auto nz = not_equal_op(r, zeros_like_op(r));
        return astype_op(sum_op(nz, axes, true), r->dtype());
    }
    if (ord == std::numeric_limits<double>::infinity())
        return max_op(r, axes, true);
    if (ord == -std::numeric_limits<double>::infinity())
        return min_op(r, axes, true);
    if (ord == 1.0)
        return sum_op(r, axes, true);
    return real_non_trivial_p_norm(r, ord, axis);
}

}  // namespace

TensorImplPtr
LpNormalizeBackward::forward(const TensorImplPtr& x0, double ord, int axis, double eps) {
    Validator::input(x0, "lp_normalize.x").non_null();
    if (x0->shape().empty())
        return on_scalar_axis(x0, axis, "lp_normalize", [&](const TensorImplPtr& v, int ax) {
            return forward(v, ord, ax, eps);
        });
    // Builds its own forward, so it asks for the schema dtype the kernel
    // templates would have applied.  See promote_for_schema.
    const TensorImplPtr x = promote_for_schema(schema_v1, x0);
    const int rank = static_cast<int>(x->shape().size());
    axis = wrap_dim(axis, rank, "lp_normalize");

    if (is_complex(x->dtype()) || !std::isfinite(ord)) {
        // A complex Lp norm is a real reduction of the magnitudes |x_i|
        // (ComplexAbsBackward owns their Wirtinger derivative), and the
        // normalised output divides the input by the clamped norm — the same
        // graph multi-axis F.normalize builds via vector_norm, which also
        // handles +/-inf orders where pow(|x|, inf)^(1/inf) would yield 1.
        auto norm = real_p_norm_keepdim(abs_op(x), ord, axis);
        auto floor = full_like_op(norm, eps);
        auto denom = where_op(less_op(norm, floor), floor, norm);
        return div_op(x, astype_op(denom, x->dtype()));
    }

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), x->shape()};
    scope.set_attr("ord", ord);
    scope.set_attr("axis", static_cast<std::int64_t>(axis));
    scope.set_attr("eps", eps);

    auto& be = backend::Dispatcher::for_device(x->device());
    // lp_normalize_forward returns [y, norm].
    auto fwd = be.lp_normalize_forward(x->storage(), x->shape(), ord, axis, eps, x->dtype());
    Storage y_storage = std::move(fwd[0]);
    Storage norm_storage = std::move(fwd[1]);

    auto out = std::make_shared<TensorImpl>(std::move(y_storage), x->shape(), x->dtype(),
                                            x->device(), false);
    // wire_autograd records on_op_io internally — no explicit call.
    {
        auto bwd = std::make_shared<LpNormalizeBackward>();
        bwd->ord_ = ord;
        bwd->axis_ = axis;
        bwd->eps_ = eps;
        bwd->saved_norm_ = std::move(norm_storage);
        kernel::NaryKernel<LpNormalizeBackward, 1>::wire_autograd(std::move(bwd), {x}, out);
    }
    return out;
}

std::vector<Storage> LpNormalizeBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(this->device_);
    return {be.lp_normalize_backward(this->saved_inputs_[0], this->saved_norm_, grad_out,
                                     this->out_shape_, ord_, axis_, this->dtype_)};
}

std::vector<TensorImplPtr> LpNormalizeBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // y = x / d, d = max(n, eps), n = (sum |x|^p)^(1/p) along the axis.
    //   dx = g / d - x * sum(g * x) / d^2 * dn/dx,  dn/dx = sign(x) |x|^(p-1) / n^(p-1)
    // where the norm is above eps; below it d is the constant eps and the
    // second term vanishes.  Everything is recomputed from x so the second
    // derivative sees its dependence on the input.
    const auto& x = this->saved_impl_inputs_[0];
    if (!x)
        ErrorBuilder("lp_normalize").fail("graph-mode backward is missing its saved input");
    if (!std::isfinite(ord_) || ord_ < 1.0)
        ErrorBuilder("lp_normalize")
            .not_implemented("create_graph=True needs a finite ord of at least 1");
    const std::vector<int> axis{axis_};
    auto magnitude = abs_op(x);
    auto norm = pow_scalar_op(sum_op(pow_scalar_op(magnitude, ord_), axis, true), 1.0 / ord_);
    auto floor = full_like_op(norm, eps_);
    // The reference's floor is a clamp_min: a norm equal to eps keeps its
    // derivative, a NaN norm passes through the forward and gets none.
    // maximum_op would halve the derivative at the tie.
    auto denom = where_op(less_op(norm, floor), floor, norm);
    auto active = greater_equal_op(norm, floor);
    auto dnorm = div_op(mul_op(sign_op(x), pow_scalar_op(magnitude, ord_ - 1.0)),
                        pow_scalar_op(norm, ord_ - 1.0));
    dnorm = where_op(broadcast_to_op(active, x->shape()), dnorm, zeros_like_op(dnorm));
    auto along = div_op(sum_op(mul_op(grad_out, x), axis, true), mul_op(denom, denom));
    return {sub_op(div_op(grad_out, denom), mul_op(dnorm, along))};
}

TensorImplPtr lp_normalize_op(const TensorImplPtr& x, double ord, int axis, double eps) {
    return LpNormalizeBackward::forward(x, ord, axis, eps);
}

LUCID_REGISTER_OP(LpNormalizeBackward)

const OpSchema GlobalResponseNormBackward::schema_v1{"global_response_norm", 1,
                                                     AmpPolicy::ForceFP32, true};

TensorImplPtr GlobalResponseNormBackward::forward(const TensorImplPtr& x,
                                                  const TensorImplPtr& gamma,
                                                  const TensorImplPtr& beta,
                                                  double eps) {
    if (!x || !gamma || !beta)
        ErrorBuilder("global_response_norm").fail("null input");
    // The kernels read gamma and beta in x's dtype.
    Validator::pair(x, gamma, "global_response_norm: x/weight").same_device().same_dtype();
    Validator::pair(x, beta, "global_response_norm: x/bias").same_device().same_dtype();
    require_rank(x->shape(), 4, "global_response_norm: x", "(N, C, H, W)");
    const int C = static_cast<int>(x->shape()[1]);
    if (gamma->numel() != static_cast<std::size_t>(C) ||
        beta->numel() != static_cast<std::size_t>(C))
        throw ShapeMismatch(gamma->shape(), x->shape(),
                            "global_response_norm: gamma/beta must have C elements");

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), x->shape()};
    scope.set_attr("eps", eps);

    auto& be = backend::Dispatcher::for_device(x->device());
    // global_response_norm_forward returns [y, Nx] where Nx is the
    // normalized global response used for backprop.
    auto fwd = be.global_response_norm_forward(x->storage(), gamma->storage(), beta->storage(),
                                               x->shape(), eps, x->dtype());
    Storage out_storage = std::move(fwd[0]);
    Storage nx_storage = std::move(fwd[1]);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), x->shape(), x->dtype(),
                                            x->device(), false);
    // wire_autograd records on_op_io internally — no explicit call.
    auto bwd = std::make_shared<GlobalResponseNormBackward>();
    bwd->eps_ = eps;
    bwd->saved_Nx_ = std::move(nx_storage);
    // saved_inputs_[0..2] will hold {x, gamma, beta}.  Unlike the other
    // norms' beta, this one is read: the op is gamma * (x * Nx) + beta * x,
    // so dx carries a beta * g term (both backends read it).
    kernel::NaryKernel<GlobalResponseNormBackward, 3>::wire_autograd(std::move(bwd),
                                                                     {x, gamma, beta}, out);
    return out;
}

std::vector<Storage> GlobalResponseNormBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(this->device_);
    // Returns [dx, d_gamma, d_beta].
    return be.global_response_norm_backward(this->saved_inputs_[0], this->saved_inputs_[1],
                                            this->saved_inputs_[2], this->saved_Nx_, grad_out,
                                            this->input_shapes_[0], eps_, this->dtype_);
}

TensorImplPtr global_response_norm_op(const TensorImplPtr& x,
                                      const TensorImplPtr& gamma,
                                      const TensorImplPtr& beta,
                                      double eps) {
    return GlobalResponseNormBackward::forward(x, gamma, beta, eps);
}

LUCID_REGISTER_OP(GlobalResponseNormBackward)

}  // namespace lucid
