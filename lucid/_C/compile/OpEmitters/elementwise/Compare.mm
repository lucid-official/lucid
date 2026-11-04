// lucid/_C/compile/OpEmitters/elementwise/Compare.mm
//
// Element-wise comparison emitters: equal / not_equal / greater /
// greater_equal / less / less_equal + bitwise invert.  All six
// comparison forms produce a Bool output and share a single
// MPSGraph builder per op.
//
// Op names match the engine schemas:
//   - equal / not_equal / greater / greater_equal / less / less_equal
//     (lucid/_C/ops/bfunc/Compare.cpp) — share ``cmp_dispatch``
//   - invert (lucid/_C/ops/bfunc/Predicate.cpp) — bitwise NOT, also
//     acts as logical NOT for Bool dtype.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string>
#include <string_view>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

template <class BuilderBlock>
inline bool emit_cmp(BuilderContext& ctx, const OpNode& node, BuilderBlock builder) {
    if (node.inputs.size() != 2 || node.outputs.empty())
        return false;
    TensorId a_id = node.inputs[0];
    TensorId b_id = node.inputs[1];
    if (a_id < 0 || b_id < 0)
        return false;
    MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
    MPSGraphTensor* a_t = (__bridge MPSGraphTensor*)ctx.resolve(a_id);
    MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
    if (graph == nil || a_t == nil || b_t == nil)
        return false;
    // MPSGraph aborts the process on a comparison of two dtypes (LCD-270).
    // The builder holds every value at its declared dtype and the engine
    // compares one dtype only, so a mismatch here is a broken trace:
    // decline it to eager rather than let MPSGraph see it.
    if (a_t.dataType != b_t.dataType)
        return false;
    ctx.bind(node.outputs[0].id, (__bridge void*)(builder(graph, a_t, b_t)));
    return true;
}

class EqualEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "equal"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_cmp(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g equalWithPrimaryTensor:a secondaryTensor:b name:@"equal"];
        });
    }
};

class NotEqualEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "not_equal"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_cmp(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g notEqualWithPrimaryTensor:a secondaryTensor:b name:@"not_equal"];
        });
    }
};

class GreaterEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "greater"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_cmp(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g greaterThanWithPrimaryTensor:a secondaryTensor:b name:@"greater"];
        });
    }
};

class GreaterEqualEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "greater_equal"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_cmp(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g greaterThanOrEqualToWithPrimaryTensor:a
                                            secondaryTensor:b
                                                       name:@"greater_equal"];
        });
    }
};

class LessEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "less"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_cmp(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g lessThanWithPrimaryTensor:a secondaryTensor:b name:@"less"];
        });
    }
};

class LessEqualEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "less_equal"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_cmp(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g lessThanOrEqualToWithPrimaryTensor:a secondaryTensor:b name:@"less_equal"];
        });
    }
};

// invert — bitwise NOT (also acts as logical NOT for Bool dtype).
class InvertEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "invert"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return false;
        // ``~`` on bool is logical not.  MPSGraph's bitwise NOT takes integers
        // only and, handed an ``i1``, aborts the process rather than decline.
        MPSGraphTensor* y = x_t.dataType == MPSDataTypeBool
                                ? [graph notWithTensor:x_t name:@"invert"]
                                : [graph bitwiseNOTWithTensor:x_t name:@"invert"];
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }
};

inline int mps_shift_bit_width(MPSDataType dt) {
    if (dt == MPSDataTypeInt8 || dt == MPSDataTypeUInt8)
        return 8;
    if (dt == MPSDataTypeInt16 || dt == MPSDataTypeUInt16)
        return 16;
    if (dt == MPSDataTypeInt64 || dt == MPSDataTypeUInt64)
        return 64;
    return 32;
}

inline MPSGraphTensor*
emit_clamped_shift(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b, bool is_left) {
    const MPSDataType dt = a.dataType;
    const int width = mps_shift_bit_width(dt);
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
    MPSGraphTensor* w_const = [g constantWithScalar:static_cast<double>(width) dataType:dt];
    MPSGraphTensor* w_minus_1 = [g constantWithScalar:static_cast<double>(width - 1) dataType:dt];
    MPSGraphTensor* in_range = [g
        logicalANDWithPrimaryTensor:[g greaterThanOrEqualToWithPrimaryTensor:b
                                                             secondaryTensor:zero
                                                                        name:nil]
                    secondaryTensor:[g lessThanWithPrimaryTensor:b secondaryTensor:w_const name:nil]
                               name:nil];
    if (is_left) {
        MPSGraphTensor* safe_b = [g selectWithPredicateTensor:in_range
                                          truePredicateTensor:b
                                         falsePredicateTensor:zero
                                                         name:nil];
        MPSGraphTensor* shifted = [g bitwiseLeftShiftWithPrimaryTensor:a
                                                       secondaryTensor:safe_b
                                                                  name:nil];
        return [g selectWithPredicateTensor:in_range
                        truePredicateTensor:shifted
                       falsePredicateTensor:zero
                                       name:nil];
    }
    MPSGraphTensor* safe_b = [g selectWithPredicateTensor:in_range
                                      truePredicateTensor:b
                                     falsePredicateTensor:w_minus_1
                                                     name:nil];
    return [g bitwiseRightShiftWithPrimaryTensor:a secondaryTensor:safe_b name:nil];
}

// ── bitwise_and / or / xor / shifts.  ``Bitwise.cpp`` records the operands
// by hand (no gradient, so no ``wire_autograd``) but nothing emitted them, so
// every graph with ``&`` / ``|`` / ``^`` fell back — ``linalg.norm``'s default
// path among them.  Bool takes the logical forms: MPSGraph's bitwise ops
// take integers only.  OP: 0=and 1=or 2=xor 3=<< 4=>>.
template <int OP>
class BitwiseEmitterT final : public OpEmitter {
public:
    explicit BitwiseEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 2 || node.outputs.empty())
            return false;
        if (node.inputs[0] < 0 || node.inputs[1] < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* a = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        MPSGraphTensor* b = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[1]);
        if (g == nil || a == nil || b == nil || a.dataType != b.dataType)
            return false;
        const bool is_bool = a.dataType == MPSDataTypeBool;
        if (a.dataType & MPSDataTypeFloatBit)
            return false;
        MPSGraphTensor* y = nil;
        switch (OP) {
        case 0:
            y = is_bool ? [g logicalANDWithPrimaryTensor:a secondaryTensor:b name:nil]
                        : [g bitwiseANDWithPrimaryTensor:a secondaryTensor:b name:nil];
            break;
        case 1:
            y = is_bool ? [g logicalORWithPrimaryTensor:a secondaryTensor:b name:nil]
                        : [g bitwiseORWithPrimaryTensor:a secondaryTensor:b name:nil];
            break;
        case 2:
            y = is_bool ? [g logicalXORWithPrimaryTensor:a secondaryTensor:b name:nil]
                        : [g bitwiseXORWithPrimaryTensor:a secondaryTensor:b name:nil];
            break;
        case 3:
        case 4:
            if (is_bool)
                return false;
            y = emit_clamped_shift(g, a, b, OP == 3);
            break;
        default:
            return false;
        }
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }

private:
    std::string name_;
};

struct CompareEmitterRegistrar {
    CompareEmitterRegistrar() {
        register_emitter(std::make_unique<BitwiseEmitterT<0>>("bitwise_and"));
        register_emitter(std::make_unique<BitwiseEmitterT<1>>("bitwise_or"));
        register_emitter(std::make_unique<BitwiseEmitterT<2>>("bitwise_xor"));
        register_emitter(std::make_unique<BitwiseEmitterT<3>>("bitwise_left_shift"));
        register_emitter(std::make_unique<BitwiseEmitterT<4>>("bitwise_right_shift"));
        register_emitter(std::make_unique<EqualEmitter>());
        register_emitter(std::make_unique<NotEqualEmitter>());
        register_emitter(std::make_unique<GreaterEmitter>());
        register_emitter(std::make_unique<GreaterEqualEmitter>());
        register_emitter(std::make_unique<LessEmitter>());
        register_emitter(std::make_unique<LessEqualEmitter>());
        register_emitter(std::make_unique<InvertEmitter>());
    }
};

[[maybe_unused]] static const CompareEmitterRegistrar g_compare_registrar;

}  // namespace

}  // namespace lucid::compile
