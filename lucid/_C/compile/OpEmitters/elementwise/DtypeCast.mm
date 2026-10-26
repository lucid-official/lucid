// lucid/_C/compile/OpEmitters/elementwise/DtypeCast.mm
//
// Element-wise dtype conversion: ``astype``.  Each output element
// comes from the same-position input element with the target dtype.
//
// Engine schema: ``astype`` (lucid/_C/ops/ufunc/Astype.cpp) — attr
// ``dst_dtype`` (int).  Falls back to the output meta's dtype when
// the attr is missing (always set by the tracer).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../../../core/Dtype.h"
#include "../../MpsDtype.h"
#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

struct CastInputs {
    MPSGraph* graph = nil;
    MPSGraphTensor* x = nil;
    MPSDataType dst = MPSDataTypeInvalid;
};

bool resolve_cast_inputs(BuilderContext& ctx, const OpNode& node, CastInputs& out) {
    if (node.inputs.size() != 1 || node.outputs.empty() || node.inputs[0] < 0)
        return false;
    out.graph = (__bridge MPSGraph*)ctx.graph();
    out.x = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
    if (out.graph == nil || out.x == nil)
        return false;
    Dtype target = node.outputs[0].dtype;
    if (auto it = node.attrs.find("dst_dtype"); it != node.attrs.end()) {
        if (const auto* p = std::get_if<std::int64_t>(&it->second))
            target = static_cast<Dtype>(static_cast<int>(*p));
    }
    const auto dst = mps_dtype_of(target);
    if (!dst)
        return false;
    out.dst = *dst;
    return true;
}

class AstypeEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "astype"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        CastInputs in;
        if (!resolve_cast_inputs(ctx, node, in))
            return false;
        ctx.bind(node.outputs[0].id,
                 (__bridge void*)cast_mps_tensor(in.graph, in.x, in.dst, @"astype"));
        return true;
    }
};

class BitcastEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "bitcast"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        CastInputs in;
        if (!resolve_cast_inputs(ctx, node, in))
            return false;
        const bool scalar = (in.x.shape != nil && in.x.shape.count == 0);
        MPSGraphTensor* r = scalar ? [in.graph expandDimsOfTensor:in.x axis:0 name:nil] : in.x;
        MPSGraphTensor* cast = [in.graph reinterpretCastTensor:r toType:in.dst name:@"bitcast"];
        MPSGraphTensor* out = scalar ? [in.graph squeezeTensor:cast axis:0 name:nil] : cast;
        ctx.bind(node.outputs[0].id, (__bridge void*)out);
        return true;
    }
};

struct DtypeCastEmitterRegistrar {
    DtypeCastEmitterRegistrar() {
        register_emitter(std::make_unique<AstypeEmitter>());
        register_emitter(std::make_unique<BitcastEmitter>());
    }
};

[[maybe_unused]] static const DtypeCastEmitterRegistrar g_dtype_cast_registrar;

}  // namespace

}  // namespace lucid::compile
