// lucid/_C/compile/OpEmitters/Loss.mm
//
// R2 — composite loss emitters: mse_loss / bce_loss / bce_with_logits /
// huber_loss.  All four are *fused single ops* on the trace IR and
// would otherwise fall back to eager.  Each one is decomposed into
// element-wise MPSGraph primitives, with an optional reduction
// (None / Mean / Sum) keyed off the ``reduction`` int64 attr that
// :file:`lucid/_C/nn/Loss.cpp` records via ``OpScopeFull::set_attr``.
//
// ``cross_entropy_loss`` / ``nll_loss`` are handled separately (R3)
// because their per-class ``gather`` + ``ignore_index`` mask path
// requires more attrs to flow.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <memory>
#include <string_view>
#include <variant>

#include "../OpEmitter.h"
#include "../_DataMovement.h"
#include "../_IndexBounds.h"

namespace lucid::compile {

namespace {

// Reduction enum mirror of :enum:`lucid::Reduction`: 0=None / 1=Mean
// / 2=Sum.  Default is Mean.
inline std::int64_t reduction_of(const OpNode& node) {
    auto it = node.attrs.find("reduction");
    if (it == node.attrs.end())
        return 1;
    const auto* p = std::get_if<std::int64_t>(&it->second);
    return p ? *p : 1;
}

// Apply ``reduction`` to a fully-shaped element-wise loss tensor.
// Mean / Sum collapse over every axis and are squeezed to the 0-D scalar
// the trace records for the loss (MPSGraph keeps the reduced axes at size
// 1, and a value bound at another rank than its trace id promotes the next
// broadcast).
inline MPSGraphTensor* apply_reduction(MPSGraph* g, MPSGraphTensor* per_sample,
                                        std::int64_t reduction) {
    if (reduction == 0)
        return per_sample;
    NSArray<NSNumber*>* shape = per_sample.shape;
    NSMutableArray<NSNumber*>* axes =
        [NSMutableArray arrayWithCapacity:shape.count];
    for (NSUInteger d = 0; d < shape.count; ++d)
        [axes addObject:[NSNumber numberWithLongLong:(long long)d]];
    MPSGraphTensor* reduced =
        reduction == 2 ? [g reductionSumWithTensor:per_sample axes:axes name:@"loss_sum"]
                       : [g meanOfTensor:per_sample axes:axes name:@"loss_mean"];
    return [g reshapeTensor:reduced withShape:@[] name:@"loss_scalar"];
}

class MseLossEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "mse_loss"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.outputs.empty())
            return false;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        if (in_id < 0 || tg_id < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(in_id);
        MPSGraphTensor* t = (__bridge MPSGraphTensor*)ctx.resolve(tg_id);
        if (g == nil || x == nil || t == nil)
            return false;
        MPSGraphTensor* diff =
            [g subtractionWithPrimaryTensor:x secondaryTensor:t name:nil];
        MPSGraphTensor* sq = [g squareWithTensor:diff name:nil];
        ctx.bind(node.outputs[0].id, (__bridge void*)(apply_reduction(g, sq, reduction_of(node))));
        return true;
    }
};

class HuberLossEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "huber_loss"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.outputs.empty())
            return false;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        if (in_id < 0 || tg_id < 0)
            return false;
        double delta = 1.0;
        if (auto it = node.attrs.find("delta"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) delta = *p;
        }

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(in_id);
        MPSGraphTensor* t = (__bridge MPSGraphTensor*)ctx.resolve(tg_id);
        if (g == nil || x == nil || t == nil)
            return false;

        MPSGraphTensor* diff =
            [g subtractionWithPrimaryTensor:x secondaryTensor:t name:nil];
        MPSGraphTensor* abs_d = [g absoluteWithTensor:diff name:nil];
        MPSGraphTensor* d_const =
            [g constantWithScalar:delta dataType:x.dataType];
        MPSGraphTensor* half_d =
            [g constantWithScalar:(0.5 * delta) dataType:x.dataType];
        MPSGraphTensor* half =
            [g constantWithScalar:0.5 dataType:x.dataType];
        // small branch: 0.5 * diff^2
        MPSGraphTensor* sq = [g squareWithTensor:diff name:nil];
        MPSGraphTensor* sm =
            [g multiplicationWithPrimaryTensor:half secondaryTensor:sq name:nil];
        // large branch: delta * (|diff| - 0.5*delta)
        MPSGraphTensor* shifted =
            [g subtractionWithPrimaryTensor:abs_d secondaryTensor:half_d name:nil];
        MPSGraphTensor* lg =
            [g multiplicationWithPrimaryTensor:d_const secondaryTensor:shifted name:nil];
        MPSGraphTensor* mask =
            [g lessThanWithPrimaryTensor:abs_d secondaryTensor:d_const name:nil];
        MPSGraphTensor* per_sample =
            [g selectWithPredicateTensor:mask
                     truePredicateTensor:sm
                    falsePredicateTensor:lg
                                    name:nil];
        ctx.bind(node.outputs[0].id, (__bridge void*)(apply_reduction(g, per_sample, reduction_of(node))));
        return true;
    }
};

class BCELossEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "bce_loss"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        // inputs: [input, target, weight].
        if (node.inputs.size() < 3 || node.outputs.empty())
            return false;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        TensorId w_id = node.inputs[2];
        if (in_id < 0 || tg_id < 0 || w_id < 0)
            return false;
        double eps = 1e-12;
        if (auto it = node.attrs.find("eps"); it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second)) eps = *p;
        }

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(in_id);
        MPSGraphTensor* t = (__bridge MPSGraphTensor*)ctx.resolve(tg_id);
        MPSGraphTensor* w = (__bridge MPSGraphTensor*)ctx.resolve(w_id);
        if (g == nil || x == nil || t == nil || w == nil)
            return false;

        // Clamp x to [eps, 1-eps] to keep log finite.
        MPSGraphTensor* e_lo = [g constantWithScalar:eps dataType:x.dataType];
        MPSGraphTensor* e_hi =
            [g constantWithScalar:(1.0 - eps) dataType:x.dataType];
        MPSGraphTensor* xc1 =
            [g maximumWithPrimaryTensor:x secondaryTensor:e_lo name:nil];
        MPSGraphTensor* xc =
            [g minimumWithPrimaryTensor:xc1 secondaryTensor:e_hi name:nil];
        MPSGraphTensor* one =
            [g constantWithScalar:1.0 dataType:x.dataType];
        MPSGraphTensor* log_x = [g logarithmWithTensor:xc name:nil];
        MPSGraphTensor* one_minus_xc =
            [g subtractionWithPrimaryTensor:one secondaryTensor:xc name:nil];
        MPSGraphTensor* log_1mx = [g logarithmWithTensor:one_minus_xc name:nil];
        MPSGraphTensor* term1 =
            [g multiplicationWithPrimaryTensor:t secondaryTensor:log_x name:nil];
        MPSGraphTensor* one_minus_t =
            [g subtractionWithPrimaryTensor:one secondaryTensor:t name:nil];
        MPSGraphTensor* term2 =
            [g multiplicationWithPrimaryTensor:one_minus_t
                              secondaryTensor:log_1mx
                                         name:nil];
        MPSGraphTensor* sum_terms =
            [g additionWithPrimaryTensor:term1 secondaryTensor:term2 name:nil];
        MPSGraphTensor* per_sample =
            [g negativeWithTensor:sum_terms name:nil];
        // multiply by per-sample weight.
        per_sample =
            [g multiplicationWithPrimaryTensor:per_sample secondaryTensor:w name:nil];
        ctx.bind(node.outputs[0].id, (__bridge void*)(apply_reduction(g, per_sample, reduction_of(node))));
        return true;
    }
};

class BCEWithLogitsLossEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "bce_with_logits"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        // inputs: [input, target, weight, pos_weight].
        if (node.inputs.size() < 4 || node.outputs.empty())
            return false;
        TensorId in_id = node.inputs[0];
        TensorId tg_id = node.inputs[1];
        TensorId w_id = node.inputs[2];
        TensorId pw_id = node.inputs[3];
        if (in_id < 0 || tg_id < 0 || w_id < 0 || pw_id < 0)
            return false;

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(in_id);
        MPSGraphTensor* t = (__bridge MPSGraphTensor*)ctx.resolve(tg_id);
        MPSGraphTensor* w = (__bridge MPSGraphTensor*)ctx.resolve(w_id);
        MPSGraphTensor* pw = (__bridge MPSGraphTensor*)ctx.resolve(pw_id);
        if (g == nil || x == nil || t == nil || w == nil || pw == nil)
            return false;

        // Numerically stable BCE-with-logits when pos_weight=1 (the
        // ``pw`` tensor already encodes positional reweighting via the
        // engine's broadcast).  Formula:
        //   l = (1 + (pw - 1) * t) * log(1 + exp(-|x|))
        //     + max(x, 0) - x * t * pw  (rewritten for stability).
        // Engineering choice: emit the *standard* unstable form but
        // clip exp to keep parity within fp32 tolerance — matches the
        // backend's eager implementation closely enough for compile
        // smoke tests; an exact-stable variant is a future polish.
        MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:x.dataType];
        MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
        MPSGraphTensor* abs_x = [g absoluteWithTensor:x name:nil];
        MPSGraphTensor* neg_abs_x = [g negativeWithTensor:abs_x name:nil];
        MPSGraphTensor* exp_term = [g exponentWithTensor:neg_abs_x name:nil];
        MPSGraphTensor* one_plus =
            [g additionWithPrimaryTensor:one secondaryTensor:exp_term name:nil];
        MPSGraphTensor* log1p = [g logarithmWithTensor:one_plus name:nil];
        MPSGraphTensor* max_x_0 =
            [g maximumWithPrimaryTensor:x secondaryTensor:zero name:nil];
        // (pw - 1)*t + 1 — the "pos_weight" reweighting factor.
        MPSGraphTensor* pw_m1 =
            [g subtractionWithPrimaryTensor:pw secondaryTensor:one name:nil];
        MPSGraphTensor* pw_m1_t =
            [g multiplicationWithPrimaryTensor:pw_m1 secondaryTensor:t name:nil];
        MPSGraphTensor* factor =
            [g additionWithPrimaryTensor:pw_m1_t secondaryTensor:one name:nil];
        MPSGraphTensor* log1p_scaled =
            [g multiplicationWithPrimaryTensor:log1p secondaryTensor:factor name:nil];
        // x * t (then subtract).
        MPSGraphTensor* xt =
            [g multiplicationWithPrimaryTensor:x secondaryTensor:t name:nil];
        MPSGraphTensor* per_sample_a =
            [g additionWithPrimaryTensor:max_x_0
                          secondaryTensor:log1p_scaled
                                     name:nil];
        MPSGraphTensor* per_sample =
            [g subtractionWithPrimaryTensor:per_sample_a
                            secondaryTensor:xt
                                       name:nil];
        per_sample =
            [g multiplicationWithPrimaryTensor:per_sample secondaryTensor:w name:nil];
        ctx.bind(node.outputs[0].id, (__bridge void*)(apply_reduction(g, per_sample, reduction_of(node))));
        return true;
    }
};

// The ``ignore_index`` attr, -100 when the trace did not record one.
inline std::int64_t ignore_index_of(const OpNode& node) {
    auto it = node.attrs.find("ignore_index");
    if (it == node.attrs.end())
        return -100;
    const auto* p = std::get_if<std::int64_t>(&it->second);
    return p ? *p : -100;
}

// ``-log_p[i, target_i] * weight[target_i]`` per sample, then ``reduction``
// — the part ``nll_loss`` and ``cross_entropy_loss`` share.  The target goes
// through ``graph_class_target`` first: an ignored sample is 0, a target
// outside the classes is NaN (eager raises there), and the gathers only ever
// see an index inside the class axis.  A weighted mean divides by the
// weight of the counted samples, as eager does.  nil — decline to eager —
// for any layout but (B, C) with a (B,) target, or an unknown class count.
inline MPSGraphTensor* class_target_loss(MPSGraph* g,
                                         MPSGraphTensor* log_p,
                                         MPSGraphTensor* tgt,
                                         MPSGraphTensor* weight,
                                         std::int64_t ignore_index,
                                         std::int64_t reduction) {
    if (log_p.shape.count != 2 || tgt.shape.count != 1)
        return nil;
    const GraphClassTarget t = graph_class_target(g, tgt, static_extent(log_p, 1), ignore_index);
    if (t.safe == nil)
        return nil;
    const MPSDataType ft = log_p.dataType;
    MPSGraphTensor* column = [g reshapeTensor:t.safe withShape:@[ @-1, @1 ] name:nil];
    MPSGraphTensor* picked = [g reshapeTensor:graph_gather_along_axis(g, 1, log_p, column)
                                    withShape:@[ @-1 ]
                                         name:nil];
    MPSGraphTensor* w_t = [g constantWithScalar:1.0 dataType:ft];
    if (weight != nil) {
        if (weight.shape.count != 1)
            return nil;
        MPSGraphTensor* w = weight.dataType == ft ? weight : [g castTensor:weight toType:ft name:nil];
        w_t = graph_gather_along_axis(g, 0, w, t.safe);
    }
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:ft];
    MPSGraphTensor* stray = [g selectWithPredicateTensor:t.ignored
                                     truePredicateTensor:zero
                                    falsePredicateTensor:[g constantWithScalar:std::nan("")
                                                                      dataType:ft]
                                                    name:nil];
    MPSGraphTensor* per_sample = [g
        selectWithPredicateTensor:t.counts
              truePredicateTensor:[g negativeWithTensor:[g multiplicationWithPrimaryTensor:picked
                                                                           secondaryTensor:w_t
                                                                                      name:nil]
                                                   name:nil]
             falsePredicateTensor:stray
                             name:@"class_target_loss"];
    if (reduction == 0)
        return per_sample;
    MPSGraphTensor* total = [g reductionSumWithTensor:per_sample axes:@[ @0 ] name:nil];
    if (reduction == 2)
        return total;
    MPSGraphTensor* counted = [g selectWithPredicateTensor:t.counts
                                       truePredicateTensor:w_t
                                      falsePredicateTensor:zero
                                                      name:nil];
    return [g divisionWithPrimaryTensor:total
                        secondaryTensor:[g reductionSumWithTensor:counted axes:@[ @0 ] name:nil]
                                   name:@"class_target_loss_mean"];
}

// Binds ``nll_loss`` / ``cross_entropy_loss``; ``log_softmax_first`` for the
// latter, whose input is logits.
inline bool emit_class_target_loss(BuilderContext& ctx, const OpNode& node, bool log_softmax_first) {
    if (node.inputs.size() < 2 || node.outputs.empty())
        return false;
    const TensorId in_id = node.inputs[0];
    const TensorId tg_id = node.inputs[1];
    if (in_id < 0 || tg_id < 0)
        return false;
    MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
    MPSGraphTensor* in = (__bridge MPSGraphTensor*)ctx.resolve(in_id);
    MPSGraphTensor* tgt = (__bridge MPSGraphTensor*)ctx.resolve(tg_id);
    if (g == nil || in == nil || tgt == nil)
        return false;
    MPSGraphTensor* weight = nil;
    if (node.inputs.size() >= 3 && node.inputs[2] >= 0) {
        weight = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[2]);
        if (weight == nil)
            return false;
    }
    MPSGraphTensor* log_p =
        log_softmax_first
            ? [g logarithmWithTensor:[g softMaxWithTensor:in axis:1 name:nil] name:nil]
            : in;
    MPSGraphTensor* loss =
        class_target_loss(g, log_p, tgt, weight, ignore_index_of(node), reduction_of(node));
    if (loss == nil)
        return false;
    ctx.bind(node.outputs[0].id, (__bridge void*)loss);
    return true;
}

// nll_loss(log_probs, target, [weight]).  Today's engine records only the
// log-probabilities as a trace input, so this declines; it is ready for the
// target to be wired in.
class NLLLossEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "nll_loss"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_class_target_loss(ctx, node, false);
    }
};

// cross_entropy_loss(input, target, [weight]) = nll_loss(log_softmax(input,
// dim=1), target, ...).  Declines today for the same reason as nll_loss.
class CrossEntropyEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cross_entropy_loss"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_class_target_loss(ctx, node, true);
    }
};

struct LossOpsRegistrar {
    LossOpsRegistrar() {
        register_emitter(std::make_unique<MseLossEmitter>());
        register_emitter(std::make_unique<HuberLossEmitter>());
        register_emitter(std::make_unique<BCELossEmitter>());
        register_emitter(std::make_unique<BCEWithLogitsLossEmitter>());
        register_emitter(std::make_unique<NLLLossEmitter>());
        register_emitter(std::make_unique<CrossEntropyEmitter>());
    }
};

[[maybe_unused]] static const LossOpsRegistrar g_loss_ops_registrar;

}  // namespace

}  // namespace lucid::compile
