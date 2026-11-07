// lucid/_C/compile/OpEmitters/Cumulative.mm
//
// R3 — cumulative reductions: cumsum / cumprod / cummax / cummin.
// All four are 1-input → 1-output with an ``axis`` (int64) attribute
// set on the engine forward.  MPSGraph exposes each as a single
// builder (``cumulativeSumWithTensor:axis:...``).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../OpEmitter.h"
#include "../_NanPropagation.h"

namespace lucid::compile {

namespace {

template <class BuilderBlock>
inline bool emit_scan(BuilderContext& ctx, const OpNode& node, BuilderBlock builder) {
    if (node.inputs.empty() || node.outputs.empty())
        return false;
    TensorId x_id = node.inputs[0];
    if (x_id < 0)
        return false;
    auto it = node.attrs.find("axis");
    if (it == node.attrs.end())
        return false;
    const auto* axp = std::get_if<std::int64_t>(&it->second);
    if (axp == nullptr)
        return false;

    MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
    MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
    if (graph == nil || x_t == nil)
        return false;
    ctx.bind(node.outputs[0].id, (__bridge void*)(builder(graph, x_t, (NSInteger)*axp)));
        return true;
}

class CumsumEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cumsum"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_scan(ctx, node, [](MPSGraph* g, MPSGraphTensor* x, NSInteger ax) {
            return [g cumulativeSumWithTensor:x axis:ax name:@"cumsum"];
        });
    }
};

class CumprodEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cumprod"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_scan(ctx, node, [](MPSGraph* g, MPSGraphTensor* x, NSInteger ax) {
            return [g cumulativeProductWithTensor:x axis:ax name:@"cumprod"];
        });
    }
};

// cummax / cummin: a NaN poisons every later position of its scan, as eager's
// does.  MPSGraph's scan skips it (``cummax([1, nan, 2])`` is ``[1, nan, 2]``),
// so the NaN is put back through a running sum of the NaN carrier (LCD-873).
template <bool IS_MAX>
class CumExtremumEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return IS_MAX ? "cummax" : "cummin"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_scan(ctx, node, [](MPSGraph* g, MPSGraphTensor* x, NSInteger ax) {
            MPSGraphTensor* y = IS_MAX ? [g cumulativeMaximumWithTensor:x axis:ax name:@"cummax"]
                                       : [g cumulativeMinimumWithTensor:x axis:ax name:@"cummin"];
            return scan_keeping_nan(g, x, y, ax);
        });
    }
};

struct CumulativeEmitterRegistrar {
    CumulativeEmitterRegistrar() {
        register_emitter(std::make_unique<CumsumEmitter>());
        register_emitter(std::make_unique<CumprodEmitter>());
        register_emitter(std::make_unique<CumExtremumEmitter<true>>());
        register_emitter(std::make_unique<CumExtremumEmitter<false>>());
    }
};

[[maybe_unused]] static const CumulativeEmitterRegistrar g_cum_registrar;

}  // namespace

}  // namespace lucid::compile
