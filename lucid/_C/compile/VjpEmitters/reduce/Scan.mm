// lucid/_C/compile/VjpEmitters/reduce/Scan.mm
//
// VJPs for ``prod``, ``var``, ``cumprod``, ``cummax`` and ``cummin``.
//
// Each had no manual VJP, so a training step through it relied on
// MPSGraph's autodiff — which a train-mode batch norm or an interpolation
// elsewhere in the graph rules out, sending the whole step eager.  Every
// formula here is eager's own, edge cases included:
//
//   prod:    dx = g · (y / x)                 (y broadcast back; a zero in
//                                              x gives what eager gives)
//   var:     dx = g · 2 (x − mean) / (N − c)  (c = 1 when unbiased)
//   cumprod: dx = rcumsum(g · y) / x          (eager's reverse-cumsum form)
//   cummax / cummin: each g[k] goes to the position that *first* reached
//            the running extreme at k — a tie does not move it.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../VjpEmitter.h"
#include "../../OpEmitters/_NanPropagation.h"
#include "../../OpEmitters/_ReduceCanon.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

const std::vector<std::int64_t>* vec_attr(const OpNode& node, const char* key) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return nullptr;
    return std::get_if<std::vector<std::int64_t>>(&it->second);
}

bool bool_or(const OpNode& node, const char* key, bool def) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return def;
    if (const auto* b = std::get_if<bool>(&it->second))
        return *b;
    if (const auto* i = std::get_if<std::int64_t>(&it->second))
        return *i != 0;
    return def;
}

bool axis_attr(const OpNode& node, std::size_t rank, NSInteger& axis) {
    auto it = node.attrs.find("axis");
    if (it == node.attrs.end())
        return false;
    const auto* v = std::get_if<std::int64_t>(&it->second);
    if (v == nullptr)
        return false;
    std::int64_t a = *v < 0 ? *v + (std::int64_t)rank : *v;
    if (a < 0 || a >= (std::int64_t)rank)
        return false;
    axis = (NSInteger)a;
    return true;
}

// The axes ``dims`` names, every axis when it names none; an out-of-range one is
// ignored.  Only a rank-5+ input is rewritten (``ReduceWork``).
std::vector<bool> reduced_mask_or_all(std::size_t rank, const std::vector<std::int64_t>& dims) {
    return dims.empty() ? std::vector<bool>(rank, true)
                        : reduce_canon_mask(rank, dims).value_or(std::vector<bool>(rank, false));
}

class ProdVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "prod"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        const auto* dims = vec_attr(node, "dims");
        if (dims == nullptr || node.outputs.empty())
            return false;
        MPSGraphTensor* y = as_tensor(bctx.forward(node.outputs[0].id));
        if (y == nil)
            return false;
        return emit_unary_vjp(bctx, node, grad_outs,
            [&](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                const auto in_shape = shape_of_mps(x);
                const ReduceWork work(in_shape, reduced_mask_or_all(in_shape.size(), *dims));
                NSArray<NSNumber*>* full = shape_to_ns(work.work_shape());
                MPSGraphTensor* yk = work.keep(g, cast_if_needed(g, y, go.dataType));
                MPSGraphTensor* gk = work.keep(g, go);
                MPSGraphTensor* ratio = [g divisionWithPrimaryTensor:[g broadcastTensor:yk toShape:full name:nil]
                                                     secondaryTensor:work.enter(g, x)
                                                                name:nil];
                return work.leave(g, [g multiplicationWithPrimaryTensor:[g broadcastTensor:gk toShape:full name:nil]
                                                        secondaryTensor:ratio
                                                                   name:@"prod_vjp"]);
            });
    }
};

class VarVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "var"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        const auto* dims = vec_attr(node, "dims");
        if (dims == nullptr)
            return false;
        const bool unbiased = bool_or(node, "unbiased", true);
        return emit_unary_vjp(bctx, node, grad_outs,
            [&](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                const auto in_shape = shape_of_mps(x);
                const auto reduced = reduced_mask_or_all(in_shape.size(), *dims);
                double n = 1.0;
                for (std::size_t i = 0; i < in_shape.size(); ++i) {
                    if (reduced[i])
                        n *= (double)in_shape[i];
                }
                const double divisor = n - (unbiased ? 1.0 : 0.0);
                if (divisor <= 0.0)
                    return nil;
                const ReduceWork work(in_shape, reduced);
                MPSGraphTensor* x_e = work.enter(g, x);
                MPSGraphTensor* mean = [g meanOfTensor:x_e axes:work.axes_ns() name:nil];
                MPSGraphTensor* centred = [g subtractionWithPrimaryTensor:x_e secondaryTensor:mean name:nil];
                MPSGraphTensor* gk = work.keep(g, go);
                MPSGraphTensor* scale = [g constantWithScalar:2.0 / divisor dataType:go.dataType];
                return work.leave(g, [g multiplicationWithPrimaryTensor:[g multiplicationWithPrimaryTensor:centred
                                                                                          secondaryTensor:scale
                                                                                                     name:nil]
                                                        secondaryTensor:gk
                                                                   name:@"var_vjp"]);
            });
    }
};

class CumprodVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "cumprod"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.outputs.empty())
            return false;
        MPSGraphTensor* y = as_tensor(bctx.forward(node.outputs[0].id));
        if (y == nil)
            return false;
        return emit_unary_vjp(bctx, node, grad_outs,
            [&](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                NSInteger axis = 0;
                if (!axis_attr(node, x.shape.count, axis))
                    return nil;
                MPSGraphTensor* weighted = [g multiplicationWithPrimaryTensor:go
                                                              secondaryTensor:cast_if_needed(g, y, go.dataType)
                                                                         name:nil];
                MPSGraphTensor* acc = [g cumulativeSumWithTensor:weighted
                                                            axis:axis
                                                       exclusive:NO
                                                         reverse:YES
                                                            name:nil];
                return [g divisionWithPrimaryTensor:acc secondaryTensor:x name:@"cumprod_vjp"];
            });
    }
};

template <bool IS_MAX>
class CumExtremumVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return IS_MAX ? "cummax" : "cummin"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.outputs.empty())
            return false;
        MPSGraphTensor* y = as_tensor(bctx.forward(node.outputs[0].id));
        if (y == nil)
            return false;
        return emit_unary_vjp(bctx, node, grad_outs,
            [&](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                NSInteger axis = 0;
                if (!axis_attr(node, x.shape.count, axis))
                    return nil;
                NSArray<NSNumber*>* shape = x.shape;
                for (NSNumber* d in shape)
                    if (d.longLongValue <= 0)
                        return nil;
                const long long n = shape[(NSUInteger)axis].longLongValue;
                // A new extreme starts where the running value changes; the
                // first position always does.
                MPSGraphTensor* is_new = nil;
                if (n == 1) {
                    is_new = [g constantWithScalar:1 shape:shape dataType:MPSDataTypeInt32];
                } else {
                    MPSGraphTensor* head = [g sliceTensor:y dimension:axis start:0 length:n - 1 name:nil];
                    MPSGraphTensor* tail = [g sliceTensor:y dimension:axis start:1 length:n - 1 name:nil];
                    MPSGraphTensor* moved = [g notEqualWithPrimaryTensor:tail
                                                         secondaryTensor:head
                                                                    name:nil];
                    // A NaN is never overtaken (``nan != nan``): the scan stays on
                    // the first one, which is where eager sends the gradient.
                    if (holds_nan(y))
                        moved = [g logicalANDWithPrimaryTensor:moved
                                               secondaryTensor:[g notWithTensor:[g isNaNWithTensor:head
                                                                                              name:nil]
                                                                           name:nil]
                                                          name:nil];
                    MPSGraphTensor* changed = [g castTensor:moved
                                                     toType:MPSDataTypeInt32
                                                       name:nil];
                    NSMutableArray<NSNumber*>* first_shape = [shape mutableCopy];
                    first_shape[(NSUInteger)axis] = @1;
                    MPSGraphTensor* first = [g constantWithScalar:1 shape:first_shape dataType:MPSDataTypeInt32];
                    is_new = [g concatTensors:@[ first, changed ] dimension:axis name:nil];
                }
                // Position along the axis where each element's extreme was
                // reached: the running maximum of (is_new ? k : 0).
                MPSGraphTensor* pos = [g coordinateAlongAxis:axis withShape:shape name:nil];
                pos = [g castTensor:pos toType:MPSDataTypeInt32 name:nil];
                MPSGraphTensor* marked = [g multiplicationWithPrimaryTensor:pos secondaryTensor:is_new name:nil];
                MPSGraphTensor* claim = [g cumulativeMaximumWithTensor:marked axis:axis name:nil];
                MPSGraphTensor* zeros = [g constantWithScalar:0.0 shape:shape dataType:go.dataType];
                return [g scatterAlongAxis:axis
                            withDataTensor:zeros
                             updatesTensor:go
                             indicesTensor:claim
                                      mode:MPSGraphScatterModeAdd
                                      name:IS_MAX ? @"cummax_vjp" : @"cummin_vjp"];
            });
    }
};

struct ScanVjpRegistrar {
    ScanVjpRegistrar() {
        register_vjp_emitter(std::make_unique<ProdVjp>());
        register_vjp_emitter(std::make_unique<VarVjp>());
        register_vjp_emitter(std::make_unique<CumprodVjp>());
        register_vjp_emitter(std::make_unique<CumExtremumVjp<true>>());
        register_vjp_emitter(std::make_unique<CumExtremumVjp<false>>());
    }
};

[[maybe_unused]] static const ScanVjpRegistrar g_scan_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
