// lucid/_C/compile/OpEmitters/Norm.mm
//
// LayerNorm + RMSNorm emitters.  BatchNorm / GroupNorm carry running
// stats / channel splits that need a longer composition; deferred to
// the next chunk.
//
// LayerNorm:
//     y = ((x - mean) / sqrt(var + eps)) * gamma + beta
//   reduction axes = trailing K dims (K = gamma.rank())
//
// RMSNorm:
//     y = x / sqrt(mean(x²) + eps) * gamma
//   reduction axes = trailing K dims (K = gamma.rank())
//
// Each op reports a single double "eps" attribute via OpScopeFull::set_attr.
// The reduction axes are inferred from the trailing dim count of
// the gamma input tensor's MPSGraph shape.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../_AttrHelpers.h"
#include "../_NanPropagation.h"

namespace lucid::compile {

namespace {

inline double eps_attr(const OpNode& node) {
    auto it = node.attrs.find("eps");
    if (it == node.attrs.end())
        return -1.0;
    const auto* p = std::get_if<double>(&it->second);
    return p ? *p : -1.0;
}

// Momentum for the BN-train running-stats EMA.  Returns -1.0 when the attr is
// absent so the 5-input emit path bails to eager rather than silently freezing
// the buffers (a 5-input node always carries it — see BatchNorm.cpp set_attr).
inline double momentum_attr(const OpNode& node) {
    auto it = node.attrs.find("momentum");
    if (it == node.attrs.end())
        return -1.0;
    const auto* p = std::get_if<double>(&it->second);
    return p ? *p : -1.0;
}

// Build [rank-K, rank-K+1, …, rank-1] as the trailing-K axes list.
inline NSArray<NSNumber*>* trailing_axes(NSUInteger rank, NSUInteger K) {
    NSMutableArray<NSNumber*>* axes = [NSMutableArray arrayWithCapacity:K];
    for (NSUInteger i = rank - K; i < rank; ++i)
        [axes addObject:[NSNumber numberWithUnsignedInteger:i]];
    return axes;
}

class LayerNormEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "layer_norm"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 3)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId g_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || g_id < 0 || b_id < 0)
            return false;
        const double eps = eps_attr(node);
        if (eps < 0)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* g_t = (__bridge MPSGraphTensor*)ctx.resolve(g_id);
        MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
        if (x_t == nil || g_t == nil || b_t == nil || graph == nil)
            return false;

        NSUInteger x_rank = x_t.shape.count;
        NSUInteger g_rank = g_t.shape.count;
        if (g_rank == 0 || g_rank > x_rank)
            return false;
        NSArray<NSNumber*>* axes = trailing_axes(x_rank, g_rank);

        MPSGraphTensor* mean = [graph meanOfTensor:x_t axes:axes name:nil];
        MPSGraphTensor* var = [graph varianceOfTensor:x_t axes:axes name:nil];
        MPSGraphTensor* y =
            [graph normalizationWithTensor:x_t
                                meanTensor:mean
                            varianceTensor:var
                               gammaTensor:g_t
                                betaTensor:b_t
                                   epsilon:static_cast<float>(eps)
                                      name:@"layer_norm"];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

class RMSNormEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "rms_norm"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 2)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId g_id = node.inputs[1];
        if (x_id < 0 || g_id < 0)
            return false;
        const double eps = eps_attr(node);
        if (eps < 0)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* g_t = (__bridge MPSGraphTensor*)ctx.resolve(g_id);
        if (x_t == nil || g_t == nil || graph == nil)
            return false;

        NSUInteger x_rank = x_t.shape.count;
        NSUInteger g_rank = g_t.shape.count;
        if (g_rank == 0 || g_rank > x_rank)
            return false;
        NSArray<NSNumber*>* axes = trailing_axes(x_rank, g_rank);

        MPSDataType dt = x_t.dataType;
        MPSGraphTensor* eps_t =
            [graph constantWithScalar:eps dataType:dt];

        // RMS: rstd = 1 / sqrt(mean(x²) + eps)
        MPSGraphTensor* x_sq =
            [graph multiplicationWithPrimaryTensor:x_t secondaryTensor:x_t name:nil];
        MPSGraphTensor* mean_sq = [graph meanOfTensor:x_sq axes:axes name:nil];
        MPSGraphTensor* var_plus_eps =
            [graph additionWithPrimaryTensor:mean_sq secondaryTensor:eps_t name:nil];
        MPSGraphTensor* rstd =
            [graph reciprocalSquareRootWithTensor:var_plus_eps name:nil];
        MPSGraphTensor* x_scaled =
            [graph multiplicationWithPrimaryTensor:x_t secondaryTensor:rstd name:nil];
        MPSGraphTensor* y =
            [graph multiplicationWithPrimaryTensor:x_scaled
                                   secondaryTensor:g_t
                                              name:@"rms_norm"];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

// BatchNormEval — inference-mode BN (no running-stats update).
// Inputs: x (N, C, H, W), running_mean (C,), running_var (C,), gamma (C,), beta (C,).
// Reduction axes: implicit on the channel dim.
// y_nchw = ((x - mean[None,:,None,None]) / sqrt(var[None,:,None,None] + eps)) *
//          gamma[None,:,None,None] + beta[None,:,None,None]
class BatchNormEvalEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "batch_norm_eval"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 5)
            return false;
        for (TensorId id : node.inputs)
            if (id < 0)
                return false;
        const double eps = eps_attr(node);
        if (eps < 0)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        MPSGraphTensor* m_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[1]);
        MPSGraphTensor* v_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[2]);
        MPSGraphTensor* g_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[3]);
        MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[4]);
        if (x_t == nil || m_t == nil || v_t == nil || g_t == nil || b_t == nil || graph == nil)
            return false;

        // Reshape (C,) → (1, C, 1, …, 1) so the affine + normalisation
        // broadcast cleanly against an N-D NCHW input.  Channel-dim size
        // is read from the input ``x_t`` (NCHW) so we don't depend on
        // the running-stats placeholder rank — a placeholder might
        // arrive as (1, C) instead of (C,) after a cast or reshape op.
        NSUInteger rank = x_t.shape.count;
        if (rank < 2)
            return false;
        // MPSGraph's normalization does not carry a symbolic batch through:
        // every batch size came back the traced one, (4, 3, 5) for a batch
        // of 6.  Declined, the compile retries per-shape static.
        if (symbolic_batch_at_dim0(x_t))
            return false;
        NSMutableArray<NSNumber*>* affine_shape = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger i = 0; i < rank; ++i)
            [affine_shape addObject:[NSNumber numberWithLongLong:1]];
        affine_shape[1] = x_t.shape[1];  // channel dim from NCHW input

        MPSGraphTensor* mean = [graph reshapeTensor:m_t withShape:affine_shape name:nil];
        MPSGraphTensor* var = [graph reshapeTensor:v_t withShape:affine_shape name:nil];
        MPSGraphTensor* gamma = [graph reshapeTensor:g_t withShape:affine_shape name:nil];
        MPSGraphTensor* beta = [graph reshapeTensor:b_t withShape:affine_shape name:nil];

        MPSGraphTensor* y =
            [graph normalizationWithTensor:x_t
                                meanTensor:mean
                            varianceTensor:var
                               gammaTensor:gamma
                                betaTensor:beta
                                   epsilon:static_cast<float>(eps)
                                      name:@"batch_norm_eval"];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

// BatchNorm train mode — inputs (x, gamma, beta), reduce axes = all
// except channel dim 1.  Engine schema "batch_norm" (BatchNorm2dBackward).
class BatchNormTrainEmitter final : public OpEmitter {
public:
    explicit BatchNormTrainEmitter(std::string_view n) : name_(n) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        // 3.5 compile: 3-input = batch-stats only; 5-input = training BN that
        // also carries running_mean/running_var (inputs[3]/[4]) and emits the
        // running-stats EMA as 2 extra outputs (new_rm/new_rv).
        const std::size_t nin = node.inputs.size();
        if (nin != 3 && nin != 5)
            return false;
        for (TensorId id : node.inputs)
            if (id < 0)
                return false;
        const double eps = eps_attr(node);
        if (eps < 0)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        MPSGraphTensor* g_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[1]);
        MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[2]);
        if (x_t == nil || g_t == nil || b_t == nil || graph == nil)
            return false;

        // Source of truth for rank/shape: the trace IR's recorded
        // output meta.  ``x_t.shape`` from MPSGraph is unreliable —
        // ops like ``convolution2DWithSourceTensor:`` produce
        // tensors whose ``shape`` property may be empty (MPSGraph
        // populates shape lazily based on input shapes at
        // compile-time, not at op-construction-time).  When such an
        // unsized tensor flows into ``astype`` → BN, the BN emit
        // previously gave up at the rank check.  The trace's
        // ``node.outputs[0].shape`` always reflects the real shape
        // since it's recorded at op-dispatch time in Lucid eager.
        const auto& out_shape = node.outputs.empty()
                                    ? std::vector<std::int64_t>{}
                                    : node.outputs[0].shape;
        const NSUInteger rank = static_cast<NSUInteger>(out_shape.size());
        if (rank < 2)
            return false;

        // Reduction over all axes except channel dim 1.
        NSMutableArray<NSNumber*>* reduce_axes = [NSMutableArray array];
        for (NSUInteger i = 0; i < rank; ++i)
            if (i != 1)
                [reduce_axes addObject:[NSNumber numberWithUnsignedInteger:i]];

        MPSGraphTensor* mean = [graph meanOfTensor:x_t axes:reduce_axes name:nil];
        MPSGraphTensor* var = [graph varianceOfTensor:x_t axes:reduce_axes name:nil];

        // Reshape gamma / beta (C,) → (1, C, 1, ...) — use the
        // trace's recorded channel count, not ``x_t.shape[1]``
        // which may be unavailable (see comment above).
        NSMutableArray<NSNumber*>* affine_shape = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger i = 0; i < rank; ++i)
            [affine_shape addObject:[NSNumber numberWithLongLong:1]];
        affine_shape[1] = [NSNumber numberWithLongLong:out_shape[1]];
        MPSGraphTensor* gamma = [graph reshapeTensor:g_t withShape:affine_shape name:nil];
        MPSGraphTensor* beta = [graph reshapeTensor:b_t withShape:affine_shape name:nil];

        // **AMP/mixed-dtype reconciliation** (mirror of the Arith
        // emitters' fix).  Under autocast, ``x_t`` may be F16 (cast
        // by the autocast SchemaGuard) while gamma/beta stay F32 as
        // master weights.  MPSGraph's ``normalizationWithTensor:``
        // internally multiplies these and requires matching dtypes
        // — cast gamma/beta to match x BEFORE the normalization call.
        if (gamma.dataType != x_t.dataType) {
            gamma = [graph castTensor:gamma toType:x_t.dataType name:@"bn_gamma_cast"];
        }
        if (beta.dataType != x_t.dataType) {
            beta = [graph castTensor:beta toType:x_t.dataType name:@"bn_beta_cast"];
        }

        MPSGraphTensor* y =
            [graph normalizationWithTensor:x_t
                                meanTensor:mean
                            varianceTensor:var
                               gammaTensor:gamma
                                betaTensor:beta
                                   epsilon:static_cast<float>(eps)
                                      name:@"batch_norm"];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));

        // 3.5 compile: a 5-input training BN node carries running_mean
        // (inputs[3]) + running_var (inputs[4]) and declares 2 extra outputs.
        // Emit the EMA in-graph so the executable write-back advances the
        // buffers EVERY compiled run — otherwise the running stats freeze at
        // their trace-time value and eval() reads stale stats.  Matches the
        // eager formula (BatchNorm.cpp): the y-normalization keeps the biased
        // population variance, while the running-var blend applies the Bessel
        // n/(n-1) correction.  ``var`` here is varianceOfTensor = biased var.
        if (nin == 5 && node.outputs.size() >= 3) {
            MPSGraphTensor* rm_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[3]);
            MPSGraphTensor* rv_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[4]);
            if (rm_t == nil || rv_t == nil)
                return false;
            const double m = momentum_attr(node);
            if (m < 0.0)
                return false;

            // n = product of every axis except the channel dim (1); use the
            // trace-recorded out_shape (x_t.shape may be empty after a conv).
            double n = 1.0;
            for (NSUInteger i = 0; i < rank; ++i)
                if (i != 1)
                    n *= static_cast<double>(out_shape[i]);
            const double bessel = (n > 1.0) ? n / (n - 1.0) : 1.0;

            // mean/var are keepdim (1,C,1,...); reshape to (C,) to match buffers.
            NSArray<NSNumber*>* c_shape =
                @[ [NSNumber numberWithLongLong:static_cast<long long>(out_shape[1])] ];
            MPSGraphTensor* mean_C = [graph reshapeTensor:mean withShape:c_shape name:nil];
            MPSGraphTensor* var_C = [graph reshapeTensor:var withShape:c_shape name:nil];
            // Defensive dtype reconcile — a no-op today (x is F32 under AMP via
            // the BN ForceFP32 cast, so mean/var already match the buffer dtype).
            if (mean_C.dataType != rm_t.dataType)
                mean_C = [graph castTensor:mean_C toType:rm_t.dataType name:@"bn_meanC_cast"];
            if (var_C.dataType != rv_t.dataType)
                var_C = [graph castTensor:var_C toType:rv_t.dataType name:@"bn_varC_cast"];

            MPSGraphTensor* one_m_rm = [graph constantWithScalar:(1.0 - m) dataType:rm_t.dataType];
            MPSGraphTensor* m_rm = [graph constantWithScalar:m dataType:rm_t.dataType];
            MPSGraphTensor* one_m_rv = [graph constantWithScalar:(1.0 - m) dataType:rv_t.dataType];
            MPSGraphTensor* m_bessel =
                [graph constantWithScalar:(m * bessel) dataType:rv_t.dataType];

            // new_rm = (1-m)*running_mean + m*batch_mean
            MPSGraphTensor* new_rm = [graph
                additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:rm_t
                                                                 secondaryTensor:one_m_rm
                                                                            name:nil]
                          secondaryTensor:[graph multiplicationWithPrimaryTensor:mean_C
                                                                 secondaryTensor:m_rm
                                                                            name:nil]
                                     name:@"bn_new_rm"];
            // new_rv = (1-m)*running_var + (m*n/(n-1))*batch_var
            MPSGraphTensor* new_rv = [graph
                additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:rv_t
                                                                 secondaryTensor:one_m_rv
                                                                            name:nil]
                          secondaryTensor:[graph multiplicationWithPrimaryTensor:var_C
                                                                 secondaryTensor:m_bessel
                                                                            name:nil]
                                     name:@"bn_new_rv"];
            if (ctx.is_consumed(node.outputs[1].id))
                ctx.bind(node.outputs[1].id, (__bridge void*)new_rm);
            if (ctx.is_consumed(node.outputs[2].id))
                ctx.bind(node.outputs[2].id, (__bridge void*)new_rv);
        }
        return true;
    }

private:
    std::string name_;
};

// GroupNorm — inputs (x, gamma, beta), splits channels into G groups,
// reduces over (per-sample, per-group, all spatial).
// Implementation: reshape x to (N, G, C/G, *spatial), reduce over
// (G-index axis + spatial), normalize, reshape back, then apply
// per-channel gamma/beta broadcast.
class GroupNormEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "group_norm"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 3)
            return false;
        for (TensorId id : node.inputs)
            if (id < 0)
                return false;
        const double eps = eps_attr(node);
        if (eps < 0)
            return false;
        auto it = node.attrs.find("num_groups");
        if (it == node.attrs.end())
            return false;
        const auto* G_p = std::get_if<std::int64_t>(&it->second);
        if (G_p == nullptr || *G_p <= 0)
            return false;
        const NSInteger G = static_cast<NSInteger>(*G_p);

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        MPSGraphTensor* g_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[1]);
        MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[2]);
        if (x_t == nil || g_t == nil || b_t == nil || graph == nil)
            return false;

        NSArray<NSNumber*>* x_shape = x_t.shape;
        NSUInteger rank = x_shape.count;
        if (rank < 2)
            return false;
        const NSInteger N = [x_shape[0] integerValue];
        const NSInteger C = [x_shape[1] integerValue];
        if (C % G != 0)
            return false;
        const NSInteger C_per_G = C / G;

        // Reshape x from (N, C, *spatial) → (N, G, C/G, *spatial).
        NSMutableArray<NSNumber*>* grouped_shape =
            [NSMutableArray arrayWithCapacity:rank + 1];
        [grouped_shape addObject:@(N)];
        [grouped_shape addObject:@(G)];
        [grouped_shape addObject:@(C_per_G)];
        for (NSUInteger i = 2; i < rank; ++i)
            [grouped_shape addObject:x_shape[i]];
        MPSGraphTensor* x_grouped =
            [graph reshapeTensor:x_t withShape:grouped_shape name:nil];

        // Reduction axes = [2, 3, …, rank] (C/G + spatial).  Group axis (1)
        // and batch axis (0) are kept.
        NSMutableArray<NSNumber*>* reduce_axes = [NSMutableArray array];
        for (NSUInteger i = 2; i < grouped_shape.count; ++i)
            [reduce_axes addObject:[NSNumber numberWithUnsignedInteger:i]];
        MPSGraphTensor* mean =
            [graph meanOfTensor:x_grouped axes:reduce_axes name:nil];
        MPSGraphTensor* var =
            [graph varianceOfTensor:x_grouped axes:reduce_axes name:nil];

        // Per-group normalisation (no affine yet).  Pass identity gamma=1, beta=0
        // (we'll multiply per-channel gamma/beta after the reshape back).
        MPSDataType dt = x_t.dataType;
        MPSGraphTensor* one_t = [graph constantWithScalar:1.0 dataType:dt];
        MPSGraphTensor* zero_t = [graph constantWithScalar:0.0 dataType:dt];
        MPSGraphTensor* normed_grouped =
            [graph normalizationWithTensor:x_grouped
                                meanTensor:mean
                            varianceTensor:var
                               gammaTensor:one_t
                                betaTensor:zero_t
                                   epsilon:static_cast<float>(eps)
                                      name:nil];

        // Reshape back to original (N, C, *spatial).
        MPSGraphTensor* normed =
            [graph reshapeTensor:normed_grouped withShape:x_shape name:nil];

        // Apply per-channel gamma/beta broadcast.
        NSMutableArray<NSNumber*>* affine_shape =
            [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger i = 0; i < rank; ++i)
            [affine_shape addObject:[NSNumber numberWithLongLong:1]];
        affine_shape[1] = x_shape[1];
        MPSGraphTensor* gamma_r =
            [graph reshapeTensor:g_t withShape:affine_shape name:nil];
        MPSGraphTensor* beta_r =
            [graph reshapeTensor:b_t withShape:affine_shape name:nil];
        MPSGraphTensor* scaled =
            [graph multiplicationWithPrimaryTensor:normed
                                   secondaryTensor:gamma_r
                                              name:nil];
        MPSGraphTensor* y =
            [graph additionWithPrimaryTensor:scaled
                              secondaryTensor:beta_r
                                         name:@"group_norm"];
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }
};

// ── lp_normalize (ord=2 only) — x / max(‖x‖_2, eps) along ``axis``.
// Other orders (1, inf, generic p) defer to eager fallback.
class LpNormalizeEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "lp_normalize"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        double ord = double_attr(node, "ord", 2.0);
        if (ord != 2.0) return false;
        std::int64_t axis = int_attr(node, "axis", -1);
        double eps = double_attr(node, "eps", 1e-12);
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        NSInteger nd = (NSInteger)x.shape.count;
        if (axis < 0) axis += nd;
        if (axis < 0 || axis >= nd) return false;
        MPSGraphTensor* x_sq = [g squareWithTensor:x name:nil];
        MPSGraphTensor* sum = [g reductionSumWithTensor:x_sq
                                                    axis:(NSInteger)axis
                                                    name:nil];
        MPSGraphTensor* norm = [g squareRootWithTensor:sum name:nil];
        MPSGraphTensor* eps_c = [g constantWithScalar:eps dataType:norm.dataType];
        MPSGraphTensor* denom =
            keep_nan_of(g, [g maximumWithPrimaryTensor:norm secondaryTensor:eps_c name:nil], norm);
        ctx.bind(node.outputs[0].id, (__bridge void*)([g divisionWithPrimaryTensor:x
                                             secondaryTensor:denom
                                                        name:@"lp_normalize"]));
        return true;
    }
};

// ── global_response_norm — ConvNeXt-V2 GRN.
// Inputs: {x: (N, C, H, W), gamma: (C,), beta: (C,)}.  Formula
// matching :file:`CpuBackend.h::grn_forward`::
//
//     Gx[b, c]   = sqrt(Σ_s x[b, c, s]^2)            # L2 over (H, W)
//     mean[b]    = Σ_c Gx[b, c] / C
//     Nx[b, c]   = Gx[b, c] / (mean[b] + eps)
//     y[b, c, s] = x[b, c, s] * (gamma[c] * Nx[b, c] + beta[c])
class GlobalResponseNormEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "global_response_norm"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 3 || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        TensorId g_id = node.inputs[1];
        TensorId b_id = node.inputs[2];
        if (x_id < 0 || g_id < 0 || b_id < 0) return false;
        double eps = double_attr(node, "eps", 1e-6);
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* gamma = (__bridge MPSGraphTensor*)ctx.resolve(g_id);
        MPSGraphTensor* beta = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
        if (g == nil || x == nil || gamma == nil || beta == nil) return false;
        NSArray<NSNumber*>* x_sh = x.shape;
        if (x_sh.count != 4) return false;
        NSNumber* C = x_sh[1];
        MPSGraphTensor* x_sq = [g squareWithTensor:x name:nil];
        MPSGraphTensor* sum_sp =
            [g reductionSumWithTensor:x_sq axes:@[@2, @3] name:nil];
        sum_sp = [g reshapeTensor:sum_sp
                        withShape:@[x_sh[0], C, @1, @1]
                             name:nil];
        MPSGraphTensor* Gx = [g squareRootWithTensor:sum_sp name:nil];
        MPSGraphTensor* mean = [g meanOfTensor:Gx axes:@[@1] name:nil];
        mean = [g reshapeTensor:mean
                      withShape:@[x_sh[0], @1, @1, @1]
                           name:nil];
        MPSGraphTensor* eps_c = [g constantWithScalar:eps dataType:mean.dataType];
        MPSGraphTensor* denom =
            [g additionWithPrimaryTensor:mean secondaryTensor:eps_c name:nil];
        MPSGraphTensor* Nx =
            [g divisionWithPrimaryTensor:Gx secondaryTensor:denom name:nil];
        NSArray<NSNumber*>* g_sh = @[@1, C, @1, @1];
        MPSGraphTensor* gamma_r = [g reshapeTensor:gamma withShape:g_sh name:nil];
        MPSGraphTensor* beta_r = [g reshapeTensor:beta withShape:g_sh name:nil];
        MPSGraphTensor* term1 =
            [g multiplicationWithPrimaryTensor:gamma_r secondaryTensor:Nx name:nil];
        MPSGraphTensor* scale =
            [g additionWithPrimaryTensor:term1 secondaryTensor:beta_r name:nil];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g multiplicationWithPrimaryTensor:x
                                                   secondaryTensor:scale
                                                              name:@"grn"]));
        return true;
    }
};

struct NormEmitterRegistrar {
    NormEmitterRegistrar() {
        register_emitter(std::make_unique<LayerNormEmitter>());
        register_emitter(std::make_unique<RMSNormEmitter>());
        register_emitter(std::make_unique<BatchNormEvalEmitter>());
        // BN train: register all three N-dim names since they share the
        // same forward path (Lucid uses N=1/2/3 explicit schemas).
        register_emitter(std::make_unique<BatchNormTrainEmitter>("batch_norm"));
        register_emitter(std::make_unique<BatchNormTrainEmitter>("batch_norm1d"));
        register_emitter(std::make_unique<BatchNormTrainEmitter>("batch_norm3d"));
        register_emitter(std::make_unique<GroupNormEmitter>());
        register_emitter(std::make_unique<LpNormalizeEmitter>());
        register_emitter(std::make_unique<GlobalResponseNormEmitter>());
    }
};

[[maybe_unused]] static const NormEmitterRegistrar g_norm_registrar;

}  // namespace

}  // namespace lucid::compile
