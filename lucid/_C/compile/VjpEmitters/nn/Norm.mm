// lucid/_C/compile/VjpEmitters/nn/Norm.mm
//
// VJP for ``layer_norm``.  Forward (see :file:`OpEmitters/nn/Norm.mm`):
//
//     y = ((x - mean) / sqrt(var + eps)) * gamma + beta
//
// where the reduction axes are the *trailing K* dims (K = gamma.rank()).
// inputs = [x, gamma, beta], attrs = {eps}.
//
// Backward (standard LayerNorm formulas):
//
//   x_hat   = (x - mean) * rstd                 where rstd = 1/√(var+eps)
//   g_hat   = grad * gamma   (broadcast over leading dims)
//   sum1    = sum_norm(g_hat,           keepdim=true)
//   sum2    = sum_norm(g_hat * x_hat,   keepdim=true)
//   dx      = rstd * (g_hat - (sum1 + x_hat * sum2) / N)
//   dgamma  = sum_leading(grad * x_hat)         reshape→ gamma_shape
//   dbeta   = sum_leading(grad)                 reshape→ gamma_shape
//
// where N = ∏ x.shape[trailing K dims].
//
// ``rms_norm`` is similar but with no centering / beta; deferred until
// we have a test that exercises it.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"
#include "../../OpEmitters/_GroupNormLayout.h"
#include "../../OpEmitters/_NormLayout.h"
#include "../../OpEmitters/_ScaledPNorm.h"

namespace lucid::compile {

namespace {

// What layer_norm / rms_norm's backward works on.  ``x`` / ``grad`` / ``gamma`` are the
// operands as given, or — for a concrete rank-5+ input — their rank-2 canonical view
// (_NormLayout.h), as the forward runs.  ``norm_axes`` are the axes the statistics reduce,
// ``lead_axes`` the ones dgamma / dbeta sum over, ``N`` the size of one normalised row.
struct TrailingNormGrad {
    MPSGraphTensor* x = nil;
    MPSGraphTensor* grad = nil;
    MPSGraphTensor* gamma = nil;
    NSMutableArray<NSNumber*>* norm_axes = nil;
    NSMutableArray<NSNumber*>* lead_axes = nil;
    std::vector<std::int64_t> x_shape;
    std::vector<std::int64_t> g_shape;
    double N = 1.0;
    bool canonical = false;
};

// ``false`` when the shapes do not describe a norm over the trailing axes ``gamma`` spans.
bool trailing_norm_grad(MPSGraph* graph,
                        MPSGraphTensor* x,
                        MPSGraphTensor* grad,
                        MPSGraphTensor* gamma,
                        TrailingNormGrad& out) {
    out.x_shape = shape_of_mps(x);
    out.g_shape = shape_of_mps(gamma);
    if (out.x_shape.empty() || out.g_shape.empty() || out.g_shape.size() > out.x_shape.size())
        return false;
    const std::size_t x_rank = out.x_shape.size();
    const std::size_t lead = x_rank - out.g_shape.size();
    out.norm_axes = [NSMutableArray arrayWithCapacity:out.g_shape.size()];
    for (std::size_t i = lead; i < x_rank; ++i) {
        [out.norm_axes addObject:[NSNumber numberWithLongLong:(long long)i]];
        out.N *= (double)out.x_shape[i];
    }
    out.lead_axes = [NSMutableArray arrayWithCapacity:lead];
    for (std::size_t i = 0; i < lead; ++i)
        [out.lead_axes addObject:[NSNumber numberWithLongLong:(long long)i]];

    out.x = x;
    out.grad = grad;
    out.gamma = gamma;
    const auto layout = trailing_norm_layout(out.x_shape, out.g_shape.size());
    if (layout) {
        out.canonical = true;
        NSArray<NSNumber*>* work = shape_to_ns(layout->work);
        out.x = [graph reshapeTensor:x withShape:work name:nil];
        out.grad = [graph reshapeTensor:grad withShape:work name:nil];
        out.gamma = [graph reshapeTensor:gamma withShape:shape_to_ns(layout->affine) name:nil];
        out.norm_axes = [NSMutableArray arrayWithObject:@1];
        out.lead_axes = [NSMutableArray arrayWithObject:@0];
    }
    return true;
}

// ``dx`` back at the input's shape when the backward ran on the canonical view.
MPSGraphTensor* trailing_norm_dx(MPSGraph* graph,
                                 MPSGraphTensor* dx,
                                 const TrailingNormGrad& t,
                                 NSString* name) {
    return t.canonical ? [graph reshapeTensor:dx withShape:shape_to_ns(t.x_shape) name:name] : dx;
}

class LayerNormVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "layer_norm"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 3 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId g_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || g_id < 0 || b_id < 0) return false;

        double eps = 1e-5;
        if (auto it = node.attrs.find("eps"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) eps = *p;
        }
        if (eps < 0) return false;

        MPSGraph* graph = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        MPSGraphTensor* gamma = as_tensor(bctx.forward(g_id));
        if (graph == nil || grad == nil || x == nil || gamma == nil) return false;

        // Mixed-dtype reconciliation (autocast): all arithmetic runs
        // in grad's dtype; cast forward activations up front.
        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(graph, x, chain_dt);
        gamma = cast_if_needed(graph, gamma, chain_dt);

        TrailingNormGrad t;
        if (!trailing_norm_grad(graph, x, grad, gamma, t))
            return false;
        x = t.x;
        grad = t.grad;
        gamma = t.gamma;
        const double N = t.N;

        MPSDataType dt = chain_dt;
        MPSGraphTensor* eps_t = [graph constantWithScalar:eps dataType:dt];

        // mean, var, x_centered, rstd, x_hat.
        MPSGraphTensor* mean = [graph meanOfTensor:x axes:t.norm_axes name:nil];
        MPSGraphTensor* x_centered =
            [graph subtractionWithPrimaryTensor:x secondaryTensor:mean name:nil];
        MPSGraphTensor* var = [graph varianceOfTensor:x axes:t.norm_axes name:nil];
        MPSGraphTensor* var_eps =
            [graph additionWithPrimaryTensor:var secondaryTensor:eps_t name:nil];
        MPSGraphTensor* std_t = [graph squareRootWithTensor:var_eps name:nil];
        MPSGraphTensor* one = [graph constantWithScalar:1.0 dataType:dt];
        MPSGraphTensor* rstd =
            [graph divisionWithPrimaryTensor:one secondaryTensor:std_t name:nil];
        MPSGraphTensor* x_hat =
            [graph multiplicationWithPrimaryTensor:x_centered
                                    secondaryTensor:rstd
                                               name:nil];

        // g_hat = grad * gamma (broadcast over leading dims).
        MPSGraphTensor* g_hat =
            [graph multiplicationWithPrimaryTensor:grad
                                    secondaryTensor:gamma
                                               name:nil];

        // sum1 = sum_norm(g_hat, keepdim=true),
        // sum2 = sum_norm(g_hat * x_hat, keepdim=true).
        MPSGraphTensor* sum1 =
            [graph reductionSumWithTensor:g_hat axes:t.norm_axes name:nil];
        MPSGraphTensor* gh_xh =
            [graph multiplicationWithPrimaryTensor:g_hat
                                    secondaryTensor:x_hat
                                               name:nil];
        MPSGraphTensor* sum2 =
            [graph reductionSumWithTensor:gh_xh axes:t.norm_axes name:nil];

        // dx = rstd * (g_hat - (sum1 + x_hat * sum2) / N).
        MPSGraphTensor* inv_N =
            [graph constantWithScalar:(1.0 / N) dataType:dt];
        MPSGraphTensor* xh_sum2 =
            [graph multiplicationWithPrimaryTensor:x_hat
                                    secondaryTensor:sum2
                                               name:nil];
        MPSGraphTensor* sum_total =
            [graph additionWithPrimaryTensor:sum1 secondaryTensor:xh_sum2 name:nil];
        MPSGraphTensor* avg_term =
            [graph multiplicationWithPrimaryTensor:sum_total
                                    secondaryTensor:inv_N
                                               name:nil];
        MPSGraphTensor* g_hat_minus_avg =
            [graph subtractionWithPrimaryTensor:g_hat
                                  secondaryTensor:avg_term
                                             name:nil];
        MPSGraphTensor* dx =
            [graph multiplicationWithPrimaryTensor:rstd
                                    secondaryTensor:g_hat_minus_avg
                                               name:@"layer_norm_vjp_dx"];
        dx = trailing_norm_dx(graph, dx, t, @"layer_norm_vjp_dx");
        bctx.accumulate_grad(x_id, from_tensor(dx));

        // dgamma = sum_leading(grad * x_hat), reshape to gamma_shape.
        MPSGraphTensor* g_xh =
            [graph multiplicationWithPrimaryTensor:grad
                                    secondaryTensor:x_hat
                                               name:nil];
        MPSGraphTensor* dgamma_raw = g_xh;
        if (t.lead_axes.count > 0)
            dgamma_raw = [graph reductionSumWithTensor:dgamma_raw axes:t.lead_axes name:nil];
        // dgamma_raw has shape (1,1,...,1, gamma_shape...).  Reshape to gamma_shape.
        NSArray<NSNumber*>* g_shape_ns = shape_to_ns(t.g_shape);
        MPSGraphTensor* dgamma =
            [graph reshapeTensor:dgamma_raw withShape:g_shape_ns name:@"layer_norm_vjp_dgamma"];
        bctx.accumulate_grad(g_id, from_tensor(dgamma));

        // dbeta = sum_leading(grad), reshape to gamma_shape.
        MPSGraphTensor* dbeta_raw = grad;
        if (t.lead_axes.count > 0)
            dbeta_raw = [graph reductionSumWithTensor:dbeta_raw axes:t.lead_axes name:nil];
        MPSGraphTensor* dbeta =
            [graph reshapeTensor:dbeta_raw withShape:g_shape_ns name:@"layer_norm_vjp_dbeta"];
        bctx.accumulate_grad(b_id, from_tensor(dbeta));

        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// batch_norm_eval — inference-mode BatchNorm.
//
// Forward (see :file:`OpEmitters/nn/Norm.mm`):
//   inputs = [x, running_mean, running_var, gamma, beta]
//   y = gamma * (x - rm) / sqrt(rv + eps) + beta
//
// Backward (running stats treated as constants):
//   rstd_c = 1 / sqrt(running_var + eps)            shape (1, C, 1, ...)
//   x_hat  = (x - running_mean) * rstd_c
//   dx     = grad * gamma_c * rstd_c                shape == x
//   dgamma = sum(grad * x_hat, all-axes-except-C)   shape (C,)
//   dbeta  = sum(grad,         all-axes-except-C)   shape (C,)
//   d(running_mean) = d(running_var) = 0  (constants for bwd).
// ────────────────────────────────────────────────────────────────────
class BatchNormEvalVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "batch_norm_eval"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 5 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId rm_id = node.inputs[1];   // running_mean (no grad)
        TensorId rv_id = node.inputs[2];   // running_var  (no grad)
        TensorId g_id = node.inputs[3];
        TensorId b_id = node.inputs[4];
        if (x_id < 0 || rm_id < 0 || rv_id < 0 || g_id < 0 || b_id < 0)
            return false;
        double eps = 1e-5;
        if (auto it = node.attrs.find("eps"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) eps = *p;
        }
        if (eps < 0) return false;

        MPSGraph* graph = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        MPSGraphTensor* rm = as_tensor(bctx.forward(rm_id));
        MPSGraphTensor* rv = as_tensor(bctx.forward(rv_id));
        MPSGraphTensor* gamma = as_tensor(bctx.forward(g_id));
        if (graph == nil || grad == nil || x == nil || rm == nil || rv == nil ||
            gamma == nil)
            return false;

        // Mixed-dtype reconciliation (autocast): all forward
        // activations get cast to grad's dtype so downstream
        // arithmetic runs in one precision.  See cast_if_needed.
        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(graph, x, chain_dt);
        rm = cast_if_needed(graph, rm, chain_dt);
        rv = cast_if_needed(graph, rv, chain_dt);
        gamma = cast_if_needed(graph, gamma, chain_dt);

        std::vector<std::int64_t> x_shape = shape_of_mps(x);
        if (x_shape.size() < 2) {
            if (!node.outputs.empty()) x_shape = node.outputs[0].shape;
            if (x_shape.size() < 2) return false;
        }
        const std::size_t rank = x_shape.size();
        const std::int64_t C = x_shape[1];

        // Reshape (C,) -> (1, C, 1, ...) for broadcast over NCHW.
        std::vector<std::int64_t> affine(rank, 1);
        affine[1] = C;
        NSArray<NSNumber*>* affine_ns = shape_to_ns(affine);
        MPSGraphTensor* rm_b = [graph reshapeTensor:rm withShape:affine_ns name:nil];
        MPSGraphTensor* rv_b = [graph reshapeTensor:rv withShape:affine_ns name:nil];
        MPSGraphTensor* g_b = [graph reshapeTensor:gamma withShape:affine_ns name:nil];

        MPSDataType dt = chain_dt;
        MPSGraphTensor* eps_t = [graph constantWithScalar:eps dataType:dt];
        MPSGraphTensor* var_eps =
            [graph additionWithPrimaryTensor:rv_b secondaryTensor:eps_t name:nil];
        MPSGraphTensor* rstd =
            [graph reciprocalSquareRootWithTensor:var_eps name:nil];
        MPSGraphTensor* x_centered =
            [graph subtractionWithPrimaryTensor:x secondaryTensor:rm_b name:nil];
        MPSGraphTensor* x_hat =
            [graph multiplicationWithPrimaryTensor:x_centered
                                    secondaryTensor:rstd name:nil];

        // dx = grad * gamma * rstd
        MPSGraphTensor* g_rstd =
            [graph multiplicationWithPrimaryTensor:g_b
                                    secondaryTensor:rstd name:nil];
        MPSGraphTensor* dx =
            [graph multiplicationWithPrimaryTensor:grad
                                    secondaryTensor:g_rstd
                                               name:@"bn_eval_vjp_dx"];
        bctx.accumulate_grad(x_id, from_tensor(dx));

        // axes_non_C = all axes except 1
        NSMutableArray<NSNumber*>* non_C =
            [NSMutableArray arrayWithCapacity:rank - 1];
        for (std::size_t i = 0; i < rank; ++i) {
            if (i == 1) continue;
            [non_C addObject:[NSNumber numberWithLongLong:(long long)i]];
        }

        // dgamma = sum_non_C(grad * x_hat), reshape to (C,)
        MPSGraphTensor* g_xh =
            [graph multiplicationWithPrimaryTensor:grad
                                    secondaryTensor:x_hat name:nil];
        MPSGraphTensor* dg_keep =
            [graph reductionSumWithTensor:g_xh axes:non_C name:nil];
        NSArray<NSNumber*>* c_shape = @[ [NSNumber numberWithLongLong:C] ];
        MPSGraphTensor* dgamma =
            [graph reshapeTensor:dg_keep withShape:c_shape name:@"bn_eval_vjp_dgamma"];
        bctx.accumulate_grad(g_id, from_tensor(dgamma));

        // dbeta = sum_non_C(grad), reshape to (C,)
        MPSGraphTensor* db_keep =
            [graph reductionSumWithTensor:grad axes:non_C name:nil];
        MPSGraphTensor* dbeta =
            [graph reshapeTensor:db_keep withShape:c_shape name:@"bn_eval_vjp_dbeta"];
        bctx.accumulate_grad(b_id, from_tensor(dbeta));
        // running_mean / running_var: no gradient (treated as constants).
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// batch_norm — training-mode BatchNorm.
//
// Forward (see :file:`OpEmitters/nn/Norm.mm`):
//   inputs = [x, gamma, beta]; reduction axes = all-except-C.
//   y = gamma * (x - batch_mean) / sqrt(batch_var + eps) + beta
//
// Backward (same closed-form as LayerNorm, with axes = non-C):
//   N      = ∏ x.shape[axes-except-C]
//   x_hat  = (x - batch_mean) * rstd
//   g_hat  = grad * gamma_b
//   sum1   = sum_non_C(g_hat,         keepdim=true)
//   sum2   = sum_non_C(g_hat * x_hat, keepdim=true)
//   dx     = rstd * (g_hat - (sum1 + x_hat * sum2) / N)
//   dgamma = sum_non_C(grad * x_hat)           → (C,)
//   dbeta  = sum_non_C(grad)                    → (C,)
// ────────────────────────────────────────────────────────────────────
class BatchNormTrainVjp final : public VjpEmitter {
public:
    explicit BatchNormTrainVjp(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        // 3.5 compile: training BN may be a 3-input node (batch-stats only) or a
        // 5-input node (also carries running_mean/running_var as inputs[3]/[4]
        // for the running-stats EMA outputs).  The backward is identical for
        // both — x/gamma/beta stay inputs[0..2], the running-stat inputs carry
        // no gradient, and grad_outs[1]/[2] (for new_rm/new_rv) are never read.
        if ((node.inputs.size() != 3 && node.inputs.size() != 5) || grad_outs.empty() ||
            grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId g_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || g_id < 0 || b_id < 0) return false;
        double eps = 1e-5;
        if (auto it = node.attrs.find("eps"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) eps = *p;
        }
        if (eps < 0) return false;

        MPSGraph* graph = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        MPSGraphTensor* gamma = as_tensor(bctx.forward(g_id));
        if (graph == nil || grad == nil || x == nil || gamma == nil) return false;

        // Mixed-dtype reconciliation: cast forward activations to
        // grad's dtype so downstream arithmetic stays in one
        // precision under autocast.  See cast_if_needed in
        // _VjpHelpers.h.
        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(graph, x, chain_dt);
        gamma = cast_if_needed(graph, gamma, chain_dt);

        std::vector<std::int64_t> x_shape = shape_of_mps(x);
        if (x_shape.size() < 2) {
            if (!node.outputs.empty()) {
                x_shape = node.outputs[0].shape;
            }
            if (x_shape.size() < 2) return false;
        }
        // Rank 5 (BatchNorm3d) folds to rank 4, (N, C, D·H, W): the same
        // per-channel statistics over axes (0, 2, 3).  MPSGraph fuses these
        // gradient kernels into a stitched RMS-norm reduction that asserts on
        // axis 4 — "This class only supports axis = 0, 1, 2 or 3" — and the
        // assertion killed the process of any compiled training step through
        // a train-mode BatchNorm3d.
        const std::vector<std::int64_t> full_shape = x_shape;
        if (x_shape.size() > 4) {
            for (std::int64_t d : x_shape)
                if (d < 0)
                    return false;
            std::int64_t mid = 1;
            for (std::size_t i = 2; i + 1 < x_shape.size(); ++i)
                mid *= x_shape[i];
            x_shape = {x_shape[0], x_shape[1], mid, x_shape.back()};
            NSArray<NSNumber*>* folded = shape_to_ns(x_shape);
            x = [graph reshapeTensor:x withShape:folded name:nil];
            grad = [graph reshapeTensor:grad withShape:folded name:nil];
        }
        const std::size_t rank = x_shape.size();
        const std::int64_t C = x_shape[1];

        // axes_non_C = all axes except 1, computed once.
        NSMutableArray<NSNumber*>* non_C =
            [NSMutableArray arrayWithCapacity:rank - 1];
        for (std::size_t i = 0; i < rank; ++i) {
            if (i == 1) continue;
            [non_C addObject:[NSNumber numberWithLongLong:(long long)i]];
        }
        // gamma broadcast to the per-channel keepdim shape (1, C, 1, …).
        std::vector<std::int64_t> affine(rank, 1);
        affine[1] = C;
        NSArray<NSNumber*>* affine_ns = shape_to_ns(affine);
        MPSGraphTensor* gamma_b =
            [graph reshapeTensor:gamma withShape:affine_ns name:nil];

        // PERF (see obsidian/perf/perf-rootcause-speed-mem-deepdive): emit
        // MPSGraph's DEDICATED normalization-gradient kernels instead of the
        // ~18-node hand-expanded closed form.  The hand-rolled per-channel
        // reductions interleaved with broadcasts dominated the compiled
        // backward — BN cost ~3.9× the reference's (compiled whole-model
        // backward 269 vs 144 ms; stripping BN made compiled beat the
        // reference).  These three fused kernels are the analog of what
        // Conv2dVjp does with convolution2DData/WeightsGradient.
        // batch_mean / batch_var are keepdim per-channel (1, C, 1, …); the grad
        // op adds eps to var internally, so the RAW variance is passed.
        MPSGraphTensor* mean =
            [graph meanOfTensor:x axes:non_C name:nil];
        MPSGraphTensor* var =
            [graph varianceOfTensor:x axes:non_C name:nil];

        MPSGraphTensor* dgamma_t =
            [graph normalizationGammaGradientWithIncomingGradientTensor:grad
                                                          sourceTensor:x
                                                            meanTensor:mean
                                                        varianceTensor:var
                                                         reductionAxes:non_C
                                                               epsilon:(float)eps
                                                                  name:nil];
        MPSGraphTensor* dbeta_t =
            [graph normalizationBetaGradientWithIncomingGradientTensor:grad
                                                         sourceTensor:x
                                                        reductionAxes:non_C
                                                                 name:nil];
        MPSGraphTensor* dx =
            [graph normalizationGradientWithIncomingGradientTensor:grad
                                                      sourceTensor:x
                                                        meanTensor:mean
                                                    varianceTensor:var
                                                       gammaTensor:gamma_b
                                               gammaGradientTensor:dgamma_t
                                                betaGradientTensor:dbeta_t
                                                     reductionAxes:non_C
                                                           epsilon:(float)eps
                                                              name:@"bn_train_vjp_dx"];
        if (full_shape.size() > 4)
            dx = [graph reshapeTensor:dx withShape:shape_to_ns(full_shape) name:nil];
        bctx.accumulate_grad(x_id, from_tensor(dx));

        // dgamma / dbeta come out per-channel keepdim (1, C, 1, …) → (C,).
        NSArray<NSNumber*>* c_shape = @[ [NSNumber numberWithLongLong:C] ];
        MPSGraphTensor* dgamma =
            [graph reshapeTensor:dgamma_t withShape:c_shape name:@"bn_train_vjp_dgamma"];
        bctx.accumulate_grad(g_id, from_tensor(dgamma));
        MPSGraphTensor* dbeta =
            [graph reshapeTensor:dbeta_t withShape:c_shape name:@"bn_train_vjp_dbeta"];
        bctx.accumulate_grad(b_id, from_tensor(dbeta));
        return true;
    }

private:
    std::string name_;
};

// ────────────────────────────────────────────────────────────────────
// rms_norm — root-mean-square norm.
//
// Forward (see :file:`OpEmitters/nn/Norm.mm`):
//   inputs = [x, gamma]
//   rstd = 1 / sqrt(mean(x², trailing_K_axes) + eps)
//   norm_x = x * rstd
//   y = norm_x * gamma
//
// Backward (closed-form, derived in retro):
//   g_hat = grad * gamma                  (broadcast over leading dims)
//   s     = sum_norm(g_hat * norm_x, keepdim=true)
//   dx    = rstd * (g_hat - norm_x * s / N)
//   dgamma = sum_leading(grad * norm_x)   → gamma_shape
//
// where N = ∏ x.shape[trailing K dims].
// ────────────────────────────────────────────────────────────────────
class RmsNormVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "rms_norm"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 2 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId g_id = node.inputs[1];
        if (x_id < 0 || g_id < 0) return false;

        double eps = 1e-5;
        if (auto it = node.attrs.find("eps"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) eps = *p;
        }
        if (eps < 0) return false;

        MPSGraph* graph = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        MPSGraphTensor* gamma = as_tensor(bctx.forward(g_id));
        if (graph == nil || grad == nil || x == nil || gamma == nil) return false;

        // Mixed-dtype reconciliation (autocast).
        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(graph, x, chain_dt);
        gamma = cast_if_needed(graph, gamma, chain_dt);

        TrailingNormGrad t;
        if (!trailing_norm_grad(graph, x, grad, gamma, t))
            return false;
        x = t.x;
        grad = t.grad;
        gamma = t.gamma;
        const double N = t.N;

        MPSDataType dt = chain_dt;
        MPSGraphTensor* eps_t = [graph constantWithScalar:eps dataType:dt];

        // Recompute rstd + norm_x.
        MPSGraphTensor* x_sq =
            [graph multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
        MPSGraphTensor* mean_sq =
            [graph meanOfTensor:x_sq axes:t.norm_axes name:nil];
        MPSGraphTensor* var_eps =
            [graph additionWithPrimaryTensor:mean_sq secondaryTensor:eps_t name:nil];
        MPSGraphTensor* rstd =
            [graph reciprocalSquareRootWithTensor:var_eps name:nil];
        MPSGraphTensor* norm_x =
            [graph multiplicationWithPrimaryTensor:x secondaryTensor:rstd name:nil];

        // g_hat = grad * gamma (gamma broadcasts over leading dims).
        MPSGraphTensor* g_hat =
            [graph multiplicationWithPrimaryTensor:grad
                                    secondaryTensor:gamma name:nil];
        // s = sum_norm(g_hat * norm_x, keepdim=true)
        MPSGraphTensor* gh_nx =
            [graph multiplicationWithPrimaryTensor:g_hat
                                    secondaryTensor:norm_x name:nil];
        MPSGraphTensor* s =
            [graph reductionSumWithTensor:gh_nx axes:t.norm_axes name:nil];
        // dx = rstd * (g_hat - norm_x * s / N)
        MPSGraphTensor* inv_N =
            [graph constantWithScalar:(1.0 / N) dataType:dt];
        MPSGraphTensor* s_over_N =
            [graph multiplicationWithPrimaryTensor:s secondaryTensor:inv_N name:nil];
        MPSGraphTensor* nx_s_N =
            [graph multiplicationWithPrimaryTensor:norm_x
                                    secondaryTensor:s_over_N name:nil];
        MPSGraphTensor* diff =
            [graph subtractionWithPrimaryTensor:g_hat
                              secondaryTensor:nx_s_N name:nil];
        MPSGraphTensor* dx =
            [graph multiplicationWithPrimaryTensor:rstd
                                    secondaryTensor:diff
                                               name:@"rms_norm_vjp_dx"];
        dx = trailing_norm_dx(graph, dx, t, @"rms_norm_vjp_dx");
        bctx.accumulate_grad(x_id, from_tensor(dx));

        // dgamma = sum_leading(grad * norm_x) → gamma_shape.
        MPSGraphTensor* g_nx =
            [graph multiplicationWithPrimaryTensor:grad
                                    secondaryTensor:norm_x name:nil];
        MPSGraphTensor* dg_raw = g_nx;
        if (t.lead_axes.count > 0)
            dg_raw = [graph reductionSumWithTensor:dg_raw axes:t.lead_axes name:nil];
        MPSGraphTensor* dgamma =
            [graph reshapeTensor:dg_raw
                       withShape:shape_to_ns(t.g_shape)
                            name:@"rms_norm_vjp_dgamma"];
        bctx.accumulate_grad(g_id, from_tensor(dgamma));
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// group_norm — splits C channels into G groups, normalizes per-(N, G).
//
// Forward (see :file:`OpEmitters/nn/Norm.mm`):
//   inputs = [x, gamma, beta];  attrs: num_groups, eps.
//   x_grouped = reshape(x, (N, G, C/G, *spatial))  — or the rank-2
//               (N*G, C/G*spatial) form, see _GroupNormLayout.h
//   mean, var = per-group (axes 2+)
//   normed_grouped = (x_grouped - mean) * rstd     (gamma=1, beta=0)
//   normed = reshape(normed_grouped, (N, C, *spatial))
//   y = normed * gamma_b + beta_b   (gamma/beta broadcast on C axis)
//
// Backward:
//   1. dbeta  = sum_non_C(grad)               → (C,)
//   2. dgamma = sum_non_C(grad * normed)      → (C,)
//   3. d_normed = grad * gamma_b              (broadcast on C)
//      d_normed_grouped = reshape(d_normed, (N, G, C/G, *spatial))
//   4. Apply LN-style formula within each (N, G) group:
//      g_hat = d_normed_grouped
//      s1 = sum_group(g_hat,          keepdim=true)
//      s2 = sum_group(g_hat * x_hat,  keepdim=true)
//      dx_grouped = rstd * (g_hat - (s1 + x_hat * s2) / N_group)
//      where N_group = (C/G) * ∏spatial.
//   5. dx = reshape(dx_grouped, x.shape)
// ────────────────────────────────────────────────────────────────────
class GroupNormVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "group_norm"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 3 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId g_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || g_id < 0 || b_id < 0) return false;

        double eps = 1e-5;
        if (auto it = node.attrs.find("eps"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) eps = *p;
        }
        if (eps < 0) return false;
        auto ng_it = node.attrs.find("num_groups");
        if (ng_it == node.attrs.end()) return false;
        const auto* G_p = std::get_if<std::int64_t>(&ng_it->second);
        if (G_p == nullptr || *G_p <= 0) return false;
        const std::int64_t G = *G_p;

        MPSGraph* graph = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        MPSGraphTensor* gamma = as_tensor(bctx.forward(g_id));
        if (graph == nil || grad == nil || x == nil || gamma == nil) return false;

        // Mixed-dtype reconciliation (autocast).
        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(graph, x, chain_dt);
        gamma = cast_if_needed(graph, gamma, chain_dt);

        std::vector<std::int64_t> x_shape = shape_of_mps(x);
        if (x_shape.size() < 2) {
            if (!node.outputs.empty()) x_shape = node.outputs[0].shape;
            if (x_shape.size() < 2) return false;
        }
        const std::size_t rank = x_shape.size();
        const auto layout = group_norm_layout(x_shape, G);
        if (!layout) return false;
        const std::int64_t C = x_shape[1];
        double N_group = (double)(C / G);
        for (std::size_t i = 2; i < rank; ++i)
            N_group *= (double)x_shape[i];
        NSArray<NSNumber*>* grp_axes = shape_to_ns(layout->work_axes);
        // The layout the upstream gradient and the per-channel sums live in.
        NSArray<NSNumber*>* chan_shape = shape_to_ns(layout->chan);
        if (layout->canonical)
            grad = [graph reshapeTensor:grad withShape:chan_shape name:nil];

        MPSGraphTensor* gamma_b =
            [graph reshapeTensor:gamma withShape:shape_to_ns(layout->affine) name:nil];

        MPSDataType dt = chain_dt;
        MPSGraphTensor* eps_t = [graph constantWithScalar:eps dataType:dt];

        // Forward recomputation (in grouped layout): mean, var, rstd, x_hat.
        MPSGraphTensor* x_grouped =
            [graph reshapeTensor:x withShape:shape_to_ns(layout->work) name:nil];
        MPSGraphTensor* mean =
            [graph meanOfTensor:x_grouped axes:grp_axes name:nil];
        MPSGraphTensor* var =
            [graph varianceOfTensor:x_grouped axes:grp_axes name:nil];
        MPSGraphTensor* var_eps =
            [graph additionWithPrimaryTensor:var secondaryTensor:eps_t name:nil];
        MPSGraphTensor* rstd =
            [graph reciprocalSquareRootWithTensor:var_eps name:nil];
        MPSGraphTensor* x_centered =
            [graph subtractionWithPrimaryTensor:x_grouped
                                secondaryTensor:mean name:nil];
        MPSGraphTensor* x_hat_grp =
            [graph multiplicationWithPrimaryTensor:x_centered
                                    secondaryTensor:rstd name:nil];
        // normed (channel layout) for dgamma later.
        MPSGraphTensor* normed =
            [graph reshapeTensor:x_hat_grp withShape:chan_shape name:nil];

        // Channel-axes reduction list for dgamma / dbeta = all-except-C.
        NSArray<NSNumber*>* non_C = shape_to_ns(layout->chan_sum_axes);
        NSArray<NSNumber*>* c_shape = @[ [NSNumber numberWithLongLong:C] ];

        // dbeta = sum_non_C(grad) → (C,)
        MPSGraphTensor* db_keep =
            [graph reductionSumWithTensor:grad axes:non_C name:nil];
        MPSGraphTensor* dbeta =
            [graph reshapeTensor:db_keep withShape:c_shape name:@"gn_vjp_dbeta"];
        bctx.accumulate_grad(b_id, from_tensor(dbeta));

        // dgamma = sum_non_C(grad * normed) → (C,)
        MPSGraphTensor* g_normed =
            [graph multiplicationWithPrimaryTensor:grad
                                    secondaryTensor:normed name:nil];
        MPSGraphTensor* dg_keep =
            [graph reductionSumWithTensor:g_normed axes:non_C name:nil];
        MPSGraphTensor* dgamma =
            [graph reshapeTensor:dg_keep withShape:c_shape name:@"gn_vjp_dgamma"];
        bctx.accumulate_grad(g_id, from_tensor(dgamma));

        // d_normed = grad * gamma_b (broadcast on C).
        MPSGraphTensor* d_normed =
            [graph multiplicationWithPrimaryTensor:grad
                                    secondaryTensor:gamma_b name:nil];
        // d_normed in grouped layout.
        MPSGraphTensor* d_normed_grp =
            [graph reshapeTensor:d_normed
                       withShape:shape_to_ns(layout->work) name:nil];

        // s1 = sum_group(g_hat,          keepdim)
        // s2 = sum_group(g_hat * x_hat,  keepdim)
        MPSGraphTensor* s1 =
            [graph reductionSumWithTensor:d_normed_grp axes:grp_axes name:nil];
        MPSGraphTensor* gh_xh =
            [graph multiplicationWithPrimaryTensor:d_normed_grp
                                    secondaryTensor:x_hat_grp name:nil];
        MPSGraphTensor* s2 =
            [graph reductionSumWithTensor:gh_xh axes:grp_axes name:nil];

        // dx_grouped = rstd * (g_hat - (s1 + x_hat * s2) / N_group)
        MPSGraphTensor* inv_Ng =
            [graph constantWithScalar:(1.0 / N_group) dataType:dt];
        MPSGraphTensor* xh_s2 =
            [graph multiplicationWithPrimaryTensor:x_hat_grp
                                    secondaryTensor:s2 name:nil];
        MPSGraphTensor* sum_total =
            [graph additionWithPrimaryTensor:s1 secondaryTensor:xh_s2 name:nil];
        MPSGraphTensor* avg_term =
            [graph multiplicationWithPrimaryTensor:sum_total
                                    secondaryTensor:inv_Ng name:nil];
        MPSGraphTensor* diff =
            [graph subtractionWithPrimaryTensor:d_normed_grp
                                secondaryTensor:avg_term name:nil];
        MPSGraphTensor* dx_grp =
            [graph multiplicationWithPrimaryTensor:rstd
                                    secondaryTensor:diff name:nil];
        // Reshape back to original x.shape.
        MPSGraphTensor* dx =
            [graph reshapeTensor:dx_grp withShape:shape_to_ns(x_shape)
                            name:@"gn_vjp_dx"];
        bctx.accumulate_grad(x_id, from_tensor(dx));
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// lp_normalize (ord=2 only) — y = x / max(‖x‖_2, eps) along ``axis``.
//
// Backward (denom = max(norm, eps), u = x / denom, dot = sum(grad*u, axis, keepdim=true)):
//   dx = (grad - u * dot) / denom
// The same value as ``inv * (grad - x * <grad, x> * inv²)`` with ``inv = 1 / denom``, which
// squares an ``inv`` that flushes to zero from denom ~ 1e19 up (LCD-955).
// ────────────────────────────────────────────────────────────────────

// The norm the forward divided by: its own (``kLpNormSavedSlot``) when the same graph holds
// it, else built the same way, so the backward never reduces ``x`` a second time.
MPSGraphTensor* lp_normalize_norm(BackwardContext& bctx, const OpNode& node, MPSGraph* g,
                                  MPSGraphTensor* x, std::int64_t axis) {
    void* saved = node.outputs.empty() ? nullptr
                                       : bctx.forward_ctx().resolve_saved(node.outputs[0].id,
                                                                          kLpNormSavedSlot);
    return saved != nullptr ? as_tensor(saved) : euclidean_norm_keepdim(g, x, @[ @(axis) ]);
}

class LpNormalizeVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "lp_normalize"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        double ord = 2.0;
        if (auto it = node.attrs.find("ord"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) ord = *p;
        }
        if (ord != 2.0) return false;
        std::int64_t axis = -1;
        if (auto it = node.attrs.find("axis"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<std::int64_t>(&it->second)) axis = *p;
        }
        double eps = 1e-12;
        if (auto it = node.attrs.find("eps"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) eps = *p;
        }

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        if (g == nil || grad == nil || x == nil) return false;

        // Mixed-dtype reconciliation (autocast).
        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(g, x, chain_dt);

        std::vector<std::int64_t> x_shape = shape_of_mps(x);
        if (x_shape.empty()) return false;
        const std::int64_t nd = (std::int64_t)x_shape.size();
        if (axis < 0) axis += nd;
        if (axis < 0 || axis >= nd) return false;

        MPSDataType dt = chain_dt;
        MPSGraphTensor* norm = cast_if_needed(g, lp_normalize_norm(bctx, node, g, x, axis), dt);
        MPSGraphTensor* eps_t = [g constantWithScalar:eps dataType:dt];
        MPSGraphTensor* denom =
            [g maximumWithPrimaryTensor:norm secondaryTensor:eps_t name:nil];
        // dx = (grad - u * <grad, u>) / denom with u = x / denom.  The textbook
        // ``inv * (grad - x * <grad, x> * inv^2)`` squares ``1 / denom``, which
        // is subnormal (flushed to 0) from denom ~ 1e19 up; ``u`` is at most 1
        // in magnitude and the one division by denom never forms ``inv``.
        MPSGraphTensor* u = [g divisionWithPrimaryTensor:x secondaryTensor:denom name:nil];
        MPSGraphTensor* g_times_u =
            [g multiplicationWithPrimaryTensor:grad secondaryTensor:u name:nil];
        MPSGraphTensor* dot =
            [g reductionSumWithTensor:g_times_u axis:(NSInteger)axis name:nil];
        MPSGraphTensor* u_term =
            [g multiplicationWithPrimaryTensor:u secondaryTensor:dot name:nil];
        MPSGraphTensor* diff =
            [g subtractionWithPrimaryTensor:grad secondaryTensor:u_term name:nil];
        MPSGraphTensor* dx = [g divisionWithPrimaryTensor:diff
                                          secondaryTensor:denom
                                                     name:@"lp_normalize_vjp"];
        bctx.accumulate_grad(x_id, from_tensor(dx));
        return true;
    }
};

struct NormVjpRegistrar {
    NormVjpRegistrar() {
        register_vjp_emitter(std::make_unique<LayerNormVjp>());
        register_vjp_emitter(std::make_unique<BatchNormEvalVjp>());
        register_vjp_emitter(std::make_unique<BatchNormTrainVjp>("batch_norm"));
        register_vjp_emitter(std::make_unique<BatchNormTrainVjp>("batch_norm1d"));
        register_vjp_emitter(std::make_unique<BatchNormTrainVjp>("batch_norm3d"));
        register_vjp_emitter(std::make_unique<RmsNormVjp>());
        register_vjp_emitter(std::make_unique<GroupNormVjp>());
        register_vjp_emitter(std::make_unique<LpNormalizeVjp>());
        // global_response_norm (ConvNeXt-V2): VJP deferred — multi-term
        // chain rule through Gx + per-(N) channel-mean + affine; ~200
        // additional lines.  Niche.  Until added, GRN traces fall
        // through to MPSGraph autograd via the soft-fallback policy.
    }
};

[[maybe_unused]] static const NormVjpRegistrar g_norm_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
