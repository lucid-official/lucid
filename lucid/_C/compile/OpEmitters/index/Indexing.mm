// lucid/_C/compile/OpEmitters/index/Indexing.mm
//
// Data-dependent value-selection emitters:
//
//   - ``where``        — element-wise select via predicate
//   - ``masked_fill``  — where(mask, full(v), x) shortcut
//   - ``searchsorted`` — static unrolled binary search over sorted sequence

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <variant>

#include "../OpEmitter.h"
#include "../_DataMovement.h"
#include "../_IndexBounds.h"

namespace lucid::compile {

namespace {

class WhereEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "where"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 3 || node.outputs.empty())
            return false;
        TensorId c_id = node.inputs[0];
        TensorId x_id = node.inputs[1];
        TensorId y_id = node.inputs[2];
        if (c_id < 0 || x_id < 0 || y_id < 0)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* c_t = (__bridge MPSGraphTensor*)ctx.resolve(c_id);
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* y_t = (__bridge MPSGraphTensor*)ctx.resolve(y_id);
        // Two branch dtypes abort MPSGraph's select; the engine never
        // records one (see Compare.mm).
        if (graph == nil || c_t == nil || x_t == nil || y_t == nil || x_t.dataType != y_t.dataType)
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)([graph selectWithPredicateTensor:c_t
                                                                   truePredicateTensor:x_t
                                                                  falsePredicateTensor:y_t
                                                                                  name:@"where"]));
        return true;
    }
};

class MaskedFillEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "masked_fill"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 2 || node.outputs.empty())
            return false;
        TensorId a_id = node.inputs[0];
        TensorId m_id = node.inputs[1];
        if (a_id < 0 || m_id < 0)
            return false;
        auto it = node.attrs.find("fill_value");
        if (it == node.attrs.end())
            return false;
        const auto* vp = std::get_if<double>(&it->second);
        if (vp == nullptr)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* a_t = (__bridge MPSGraphTensor*)ctx.resolve(a_id);
        MPSGraphTensor* m_t = (__bridge MPSGraphTensor*)ctx.resolve(m_id);
        if (graph == nil || a_t == nil || m_t == nil)
            return false;
        MPSGraphTensor* fill = [graph constantWithScalar:*vp dataType:a_t.dataType];
        ctx.bind(node.outputs[0].id,
                 (__bridge void*)([graph selectWithPredicateTensor:m_t
                                               truePredicateTensor:fill
                                              falsePredicateTensor:a_t
                                                              name:@"masked_fill"]));
        return true;
    }
};

std::pair<MPSGraphTensor*, MPSGraphTensor*> emit_searchsorted_step(MPSGraph* g,
                                                                   MPSGraphTensor* seq,
                                                                   MPSGraphTensor* val,
                                                                   MPSGraphTensor* lo,
                                                                   MPSGraphTensor* hi,
                                                                   MPSGraphTensor* one,
                                                                   MPSGraphTensor* max_idx,
                                                                   NSInteger last_axis,
                                                                   bool is_1d_seq,
                                                                   bool right) {
    MPSGraphTensor* diff = [g subtractionWithPrimaryTensor:hi secondaryTensor:lo name:nil];
    MPSGraphTensor* half = [g bitwiseRightShiftWithPrimaryTensor:diff secondaryTensor:one name:nil];
    MPSGraphTensor* mid = [g additionWithPrimaryTensor:lo secondaryTensor:half name:nil];
    MPSGraphTensor* safe_mid = [g minimumWithPrimaryTensor:mid secondaryTensor:max_idx name:nil];
    MPSGraphTensor* seq_mid = is_1d_seq ? graph_gather(g, seq, safe_mid, 0, nil)
                                        : graph_gather_along_axis(g, last_axis, seq, safe_mid, nil);
    if (seq_mid == nil)
        return {nil, nil};
    MPSGraphTensor* cmp =
        right ? [g greaterThanOrEqualToWithPrimaryTensor:val secondaryTensor:seq_mid name:nil]
              : [g greaterThanWithPrimaryTensor:val secondaryTensor:seq_mid name:nil];
    if ((val.dataType & MPSDataTypeFloatBit) != 0) {
        MPSGraphTensor* nan_pred = [g logicalORWithPrimaryTensor:[g isNaNWithTensor:val name:nil]
                                                 secondaryTensor:[g isNaNWithTensor:seq_mid
                                                                               name:nil]
                                                            name:nil];
        cmp = [g logicalORWithPrimaryTensor:cmp secondaryTensor:nan_pred name:nil];
    }
    MPSGraphTensor* active = [g lessThanWithPrimaryTensor:lo secondaryTensor:hi name:nil];
    MPSGraphTensor* go_right = [g logicalANDWithPrimaryTensor:active secondaryTensor:cmp name:nil];
    MPSGraphTensor* mid_plus_one = [g additionWithPrimaryTensor:mid secondaryTensor:one name:nil];
    MPSGraphTensor* next_lo = [g selectWithPredicateTensor:go_right
                                       truePredicateTensor:mid_plus_one
                                      falsePredicateTensor:lo
                                                      name:nil];
    MPSGraphTensor* next_hi = [g selectWithPredicateTensor:go_right
                                       truePredicateTensor:hi
                                      falsePredicateTensor:mid
                                                      name:nil];
    return {next_lo, next_hi};
}

MPSGraphTensor* emit_searchsorted_binary_search(MPSGraph* g,
                                                MPSGraphTensor* seq_t,
                                                MPSGraphTensor* val_t,
                                                std::int64_t n_seq,
                                                NSInteger last_axis,
                                                bool is_1d_seq,
                                                bool right) {
    MPSGraphTensor* seq_work = seq_t;
    MPSGraphTensor* val_work = val_t;
    if (seq_work.dataType == MPSDataTypeBool) {
        seq_work = [g castTensor:seq_work toType:MPSDataTypeInt8 name:nil];
        val_work = [g castTensor:val_work toType:MPSDataTypeInt8 name:nil];
    }
    const bool scalar_val = val_t.shape.count == 0;
    if (scalar_val)
        val_work = [g expandDimsOfTensor:val_work axis:0 name:nil];

    MPSGraphTensor* lo = data_movement_detail::int32_filled(g, val_work, 0.0);
    MPSGraphTensor* hi =
        data_movement_detail::int32_filled(g, val_work, static_cast<double>(n_seq));
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:MPSDataTypeInt32];
    MPSGraphTensor* max_idx = [g constantWithScalar:static_cast<double>(n_seq - 1)
                                           dataType:MPSDataTypeInt32];
    for (std::int64_t span = 1; span <= n_seq; span <<= 1) {
        auto [next_lo, next_hi] = emit_searchsorted_step(g, seq_work, val_work, lo, hi, one,
                                                         max_idx, last_axis, is_1d_seq, right);
        if (next_lo == nil || next_hi == nil)
            return nil;
        lo = next_lo;
        hi = next_hi;
    }
    if (scalar_val)
        lo = [g squeezeTensor:lo axis:0 name:nil];
    return [g castTensor:lo toType:MPSDataTypeInt64 name:@"searchsorted"];
}

bool resolve_searchsorted_right(const OpNode& node, bool& right) {
    if (node.inputs.size() != 2 || node.outputs.empty())
        return false;
    if (node.inputs[0] < 0 || node.inputs[1] < 0)
        return false;
    auto r_it = node.attrs.find("right");
    if (r_it == node.attrs.end())
        return false;
    const auto* rp = std::get_if<std::int64_t>(&r_it->second);
    if (rp == nullptr)
        return false;
    right = *rp != 0;
    return true;
}

bool validate_searchsorted_tensors(MPSGraph* graph,
                                   MPSGraphTensor* seq_t,
                                   MPSGraphTensor* val_t,
                                   bool& is_1d_seq,
                                   NSInteger& last_axis,
                                   std::int64_t& n_seq) {
    if (graph == nil || seq_t == nil || val_t == nil)
        return false;
    if (seq_t.dataType != val_t.dataType || (seq_t.dataType & MPSDataTypeComplexBit) != 0)
        return false;
    if (seq_t.shape == nil || val_t.shape == nil || seq_t.shape.count == 0)
        return false;
    is_1d_seq = seq_t.shape.count == 1;
    last_axis = static_cast<NSInteger>(seq_t.shape.count) - 1;
    if (!is_1d_seq && !shapes_agree_off_axis(seq_t, val_t, last_axis))
        return false;
    n_seq = static_extent(seq_t, last_axis);
    return n_seq >= 0 && n_seq < std::numeric_limits<std::int32_t>::max();
}

bool has_zero_extent(NSArray<NSNumber*>* shape) {
    for (NSNumber* dim in shape) {
        if (dim.longLongValue == 0)
            return true;
    }
    return false;
}

class SearchsortedEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "searchsorted"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        bool right = false;
        if (!resolve_searchsorted_right(node, right))
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* seq_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        MPSGraphTensor* val_t = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[1]);
        bool is_1d_seq = false;
        NSInteger last_axis = 0;
        std::int64_t n_seq = 0;
        if (!validate_searchsorted_tensors(graph, seq_t, val_t, is_1d_seq, last_axis, n_seq))
            return false;

        if (n_seq == 0 || has_zero_extent(val_t.shape)) {
            MPSGraphTensor* zero =
                val_t.shape.count == 0
                    ? [graph constantWithScalar:0.0 dataType:MPSDataTypeInt64]
                    : [graph castTensor:data_movement_detail::int32_filled(graph, val_t, 0.0)
                                 toType:MPSDataTypeInt64
                                   name:@"searchsorted_empty"];
            ctx.bind(node.outputs[0].id, (__bridge void*)zero);
            return true;
        }
        MPSGraphTensor* out = emit_searchsorted_binary_search(graph, seq_t, val_t, n_seq, last_axis,
                                                              is_1d_seq, right);
        if (out == nil)
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)out);
        return true;
    }
};

struct IndexingEmitterRegistrar {
    IndexingEmitterRegistrar() {
        register_emitter(std::make_unique<WhereEmitter>());
        register_emitter(std::make_unique<MaskedFillEmitter>());
        register_emitter(std::make_unique<SearchsortedEmitter>());
    }
};

[[maybe_unused]] static const IndexingEmitterRegistrar g_indexing_registrar;

}  // namespace

}  // namespace lucid::compile
