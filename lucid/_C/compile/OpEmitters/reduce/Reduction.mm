// lucid/_C/compile/OpEmitters/Reduction.mm
//
// Reduction emitters — sum / mean / max / min.
//
// All require the ``dims`` (vector<int64>) + ``keepdim`` (bool)
// attributes that :file:`kernel/ReduceKernel.h`'s forward path
// reports via :func:`OpScopeFull::set_attr`.  Op names match the
// engine schemas in :file:`ops/ufunc/Reductions.cpp` ("sum", "mean",
// "max", "min").
//
// MPSGraph builders used:
//   - reductionSumWithTensor:axes:name:
//   - reductionMeanWithTensor:axes:name:
//
// ``keepdim=false`` collapses the reduced axes; MPSGraph's builders
// always keep size-1 axes, so a trailing reshape to the trace's
// recorded output shape handles the squeeze.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../OpEmitter.h"
#include "../_AttrHelpers.h"
#include "../_SignedZero.h"

namespace lucid::compile {

namespace {

template <class BuilderBlock>
inline bool emit_reduce(BuilderContext& ctx, const OpNode& node, BuilderBlock builder) {
    if (node.inputs.size() != 1 || node.outputs.empty())
        return false;
    TensorId x_id = node.inputs[0];
    if (x_id < 0)
        return false;

    // Required attrs.
    auto dims_it = node.attrs.find("dims");
    auto kd_it = node.attrs.find("keepdim");
    if (dims_it == node.attrs.end() || kd_it == node.attrs.end())
        return false;
    const auto* dims =
        std::get_if<std::vector<std::int64_t>>(&dims_it->second);
    const auto* keepdim = std::get_if<bool>(&kd_it->second);
    if (dims == nullptr || keepdim == nullptr)
        return false;

    MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
    MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
    if (x_t == nil || graph == nil)
        return false;

    NSMutableArray<NSNumber*>* axes =
        [NSMutableArray arrayWithCapacity:dims->size()];
    for (std::int64_t d : *dims)
        [axes addObject:[NSNumber numberWithLongLong:d]];

    x_t = promote_to_float_output(graph, x_t, node);
    MPSGraphTensor* reduced = builder(graph, x_t, axes);

    // MPSGraph's reductions keep the reduced axes at size 1, so with
    // ``keepdim=false`` the result is reshaped to the trace's output shape:
    // a tensor bound to a trace id must have exactly that shape, or the first
    // op that broadcasts it against a lower-rank operand promotes its result
    // (``(4,) + (1,1)`` is ``(1,4)``) and everything downstream is built from
    // the wrong rank.
    //
    // A full reduction (empty trace shape) used to stay ``(1,..,1)`` so that
    // ``gradientForPrimaryTensor:`` could chain through a loss; that
    // differentiation now goes through a rank-1 alias made where the loss is
    // handed to it (``autodiff_primary`` in ``MpsBuilder.mm``).
    const auto& out_shape = node.outputs[0].shape;
    if (!*keepdim) {
        if (out_shape.empty()) {
            reduced = [graph reshapeTensor:reduced withShape:@[] name:@"reduce_scalar"];
        } else {
            // Dynamic-batch-aware squeeze: a reduction over non-batch axes keeps the
            // symbolic (-1) batch, so pin it -1 in the target instead of the
            // trace-time concrete value (which would abort the MLIR pass).  nil =
            // symbolic batch not provably at dim 0 → bail (compile falls back).
            reduced = reshape_dynamic_aware(graph, reduced, out_shape, @"reduce_squeeze");
        }
        if (reduced == nil)
            return false;
    }
    ctx.bind(node.outputs[0].id, (__bridge void*)(reduced));
        return true;
}

class SumEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "sum"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_reduce(ctx, node, [](MPSGraph* g, MPSGraphTensor* x,
                                          NSArray<NSNumber*>* axes) {
            return [g reductionSumWithTensor:x axes:axes name:@"sum"];
        });
    }
};

class MeanEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "mean"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_reduce(ctx, node, [](MPSGraph* g, MPSGraphTensor* x,
                                          NSArray<NSNumber*>* axes) {
            MPSGraphTensor* m = [g meanOfTensor:x axes:axes name:@"mean"];
            if (m != nil && (m.dataType & MPSDataTypeFloatBit) != 0) {
                m = sub_pos_zero(g, m, [g constantWithScalar:0.0 dataType:m.dataType]);
            }
            return m;
        });
    }
};

// max / min: a slice holding a NaN reduces to NaN, as eager's does.  The
// plain ``reductionMaximum`` skips NaN — ``[1, nan, 1]`` gave 1 and an
// all-NaN slice -inf — so a compiled loss was finite where eager's is NaN,
// and the max VJP's tie mask picked elements eager's does not (LCD-865).
// An integer or bool input holds no NaN and keeps the plain reduction.
template <bool IS_MAX>
class ExtremumReduceEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return IS_MAX ? "max" : "min"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_reduce(ctx, node, [](MPSGraph* g, MPSGraphTensor* x, NSArray<NSNumber*>* axes) {
            NSString* name = IS_MAX ? @"max" : @"min";
            if ((x.dataType & MPSDataTypeFloatBit) == 0)
                return IS_MAX ? [g reductionMaximumWithTensor:x axes:axes name:name]
                              : [g reductionMinimumWithTensor:x axes:axes name:name];
            return IS_MAX ? [g reductionMaximumPropagateNaNWithTensor:x axes:axes name:name]
                          : [g reductionMinimumPropagateNaNWithTensor:x axes:axes name:name];
        });
    }
};

class ProdEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "prod"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_reduce(ctx, node, [](MPSGraph* g, MPSGraphTensor* x,
                                          NSArray<NSNumber*>* axes) {
            return [g reductionProductWithTensor:x axes:axes name:@"prod"];
        });
    }
};

// var — biased variance (matches ``var_op`` semantics; the ``unbiased``
// branch is composed on top in eager Python via ``var · n/(n-1)`` and
// shows up as a separate ``mul`` node in the trace).
class VarEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "var"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_reduce(ctx, node, [](MPSGraph* g, MPSGraphTensor* x,
                                          NSArray<NSNumber*>* axes) {
            return [g varianceOfTensor:x axes:axes name:@"var"];
        });
    }
};

struct ReductionEmitterRegistrar {
    ReductionEmitterRegistrar() {
        register_emitter(std::make_unique<SumEmitter>());
        register_emitter(std::make_unique<MeanEmitter>());
        register_emitter(std::make_unique<ExtremumReduceEmitter<true>>());
        register_emitter(std::make_unique<ExtremumReduceEmitter<false>>());
        register_emitter(std::make_unique<ProdEmitter>());
        register_emitter(std::make_unique<VarEmitter>());
    }
};

[[maybe_unused]] static const ReductionEmitterRegistrar g_reduction_registrar;

}  // namespace

}  // namespace lucid::compile
