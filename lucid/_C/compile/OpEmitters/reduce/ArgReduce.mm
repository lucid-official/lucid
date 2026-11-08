// lucid/_C/compile/OpEmitters/reduce/ArgReduce.mm
//
// Reductions that return indices or boolean aggregates instead of
// element-wise values:
//
//   - ``sort``    — full-tensor sort along an axis
//   - ``argsort`` — indices that would sort along an axis
//   - ``argmax``  — index of the maximum along an axis (squeezed unless keepdim)
//   - ``argmin``  — same for the minimum
//   - ``all``     — boolean AND-reduce across every axis
//   - ``any``     — boolean OR-reduce across every axis
//
// ``sort``, ``argsort``, ``argmax`` and ``argmin`` take their order from
// ``_SortOrder.h`` (the order eager takes: ties first index first, NaN
// as the extreme); ``all`` / ``any`` route 1:1 to MPSGraph builders.  ``topk``
// is a multi-output op in :file:`../shape/Split.mm`.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>

#include "../_AttrHelpers.h"
#include "../_SortOrder.h"

namespace lucid::compile {

namespace {

class SortEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "sort"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        std::int64_t ax = int_attr(node, "axis", -1);
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        // Values gathered by the stable order, not ``sortWithTensor``: the
        // same order eager takes, -0.0 kept where it was (``_SortOrder.h``).
        // nil: a shape MPSGraph sorts far slower than eager, so the graph
        // runs eager (``graph_sort_order``).
        MPSGraphTensor* order = graph_sort_order(g, x, (NSInteger)ax, false);
        if (order == nil) return false;
        MPSGraphTensor* sorted = graph_gather_sorted(g, x, order, (NSInteger)ax);
        if (sorted == nil) return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)sorted);
        return true;
    }
};

class ArgsortEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "argsort"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        std::int64_t ax = int_attr(node, "axis", -1);
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        MPSGraphTensor* order = graph_sort_order(g, x, (NSInteger)ax, false);
        if (order == nil) return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)order);
        return true;
    }
};

// ── argmax / argmin — same shape: reduction + optional squeeze.
template <bool IS_MAX>
class ArgReduceEmitterT final : public OpEmitter {
public:
    explicit ArgReduceEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        std::int64_t ax = int_attr(node, "axis", 0);
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        MPSGraphTensor* r = graph_arg_extreme(g, x, (NSInteger)ax, IS_MAX);
        if (r == nil) return false;
        if (!bool_attr(node, "keepdim", false)) {
            NSArray<NSNumber*>* sh = r.shape;
            NSMutableArray<NSNumber*>* new_shape =
                [NSMutableArray arrayWithCapacity:sh.count];
            for (NSUInteger d = 0; d < sh.count; ++d) {
                if ((std::int64_t)d == ax) continue;
                [new_shape addObject:sh[d]];
            }
            r = [g reshapeTensor:r withShape:new_shape name:nil];
        }
        ctx.bind(node.outputs[0].id, (__bridge void*)(r));
        return true;
    }

private:
    std::string name_;
};

// ── all / any — boolean reductions over every axis at once.
template <bool IS_AND>
class BoolReduceEmitterT final : public OpEmitter {
public:
    explicit BoolReduceEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        NSArray<NSNumber*>* in_shape = x.shape;
        NSMutableArray<NSNumber*>* all_axes = [NSMutableArray array];
        for (NSUInteger d = 0; d < in_shape.count; ++d)
            [all_axes addObject:[NSNumber numberWithLongLong:(long long)d]];
        // Reduce over int32 truth values and cast the scalar to bool last.
        // A bool-typed ``reductionAnd`` / ``reductionOr`` that is itself the
        // graph output reaches an MPS reduction kernel that asserts "Source
        // and destination must have same data type" and aborts the process;
        // min / max over 0-1 integers are the same AND / OR without it.
        MPSGraphTensor* truth = [g castTensor:[g castTensor:x toType:MPSDataTypeBool name:nil]
                                       toType:MPSDataTypeInt32
                                         name:nil];
        MPSGraphTensor* r = IS_AND ? [g reductionMinimumWithTensor:truth axes:all_axes name:nil]
                                   : [g reductionMaximumWithTensor:truth axes:all_axes name:nil];
        // The engine's all / any reduce every axis to a 0-d bool.
        r = [g reshapeTensor:r withShape:@[] name:nil];
        ctx.bind(node.outputs[0].id,
                 (__bridge void*)([g castTensor:r toType:MPSDataTypeBool name:IS_AND ? @"all" : @"any"]));
        return true;
    }

private:
    std::string name_;
};

struct ArgReduceRegistrar {
    ArgReduceRegistrar() {
        register_emitter(std::make_unique<SortEmitter>());
        register_emitter(std::make_unique<ArgsortEmitter>());
        register_emitter(std::make_unique<ArgReduceEmitterT<true>>("argmax"));
        register_emitter(std::make_unique<ArgReduceEmitterT<false>>("argmin"));
        register_emitter(std::make_unique<BoolReduceEmitterT<true>>("all"));
        register_emitter(std::make_unique<BoolReduceEmitterT<false>>("any"));
    }
};

[[maybe_unused]] static const ArgReduceRegistrar g_arg_reduce_registrar;

}  // namespace

}  // namespace lucid::compile
