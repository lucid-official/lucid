// lucid/_C/compile/OpEmitters/Reshape.mm
//
// Shape-only emitters: view + contiguous.
//
// Op names match the engine schemas in:
//   - lucid/_C/ops/utils/View.cpp        ("view")
//   - lucid/_C/ops/utils/Contiguous.cpp  ("contiguous")
//
// Both ops carry their full target metadata in :class:`OpNode::outputs[0]`
// (the new shape and dtype), so no extra attribute payload is required.
// Permute / transpose are deferred until the trace IR grows an
// attribute map, since their axis permutation can't be recovered from
// shapes alone.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../OpEmitter.h"
#include "../_AttrHelpers.h"
#include "../_AxisSplitLayout.h"

namespace lucid::compile {

namespace {

// All view-family ops (view / reshape / squeeze / unsqueeze / flatten)
// share the same forward — reshape the input to ``node.outputs[0].shape``.
// We register one instance per op_name so the engine's OpScopeFull
// strings (which carry the call-site name like ``"reshape"`` rather
// than the schema's ``"view"``) all resolve via the registry.
class ReshapeFamilyEmitter final : public OpEmitter {
public:
    explicit ReshapeFamilyEmitter(std::string_view name) : name_(name) {}

    std::string_view op_name() const override { return name_; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (x_t == nil || graph == nil)
            return false;
        // Dynamic-batch: keep the symbolic (-1) batch flowing instead of
        // pinning the trace-time concrete batch (which aborts the MLIR pass).
        // nil = symbolic batch not provably at dim 0 (e.g. unsqueeze-at-front);
        // bail so the compile falls back rather than mis-shaping the graph.
        MPSGraphTensor* y =
            reshape_dynamic_aware(graph, x_t, node.outputs[0].shape, nil);
        if (y == nil)
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }

private:
    std::string name_;
};

// ``contiguous`` and ``detach``: the value itself under a new identity.
// ``detach`` differs only backward, where its VJP passes nothing on.
class ContiguousEmitter final : public OpEmitter {
public:
    explicit ContiguousEmitter(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        // MPSGraph tensors have no observable stride layout — every
        // intermediate is already "contiguous" from the graph's
        // perspective.  Emit a reshape-to-same-shape so the output
        // tensor gets its own MPSGraphTensor identity (matches eager
        // semantics where `contiguous()` returns a new TensorImpl).
        if (node.inputs.empty() || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (x_t == nil || graph == nil)
            return false;
        // Reshape-to-same-shape (new tensor identity); dynamic-batch-aware so a
        // contiguous() under a symbolic batch keeps the -1 dim.  (Same-shape, so
        // always batch-preserving — the nil guard is for uniformity.)
        MPSGraphTensor* y =
            reshape_dynamic_aware(graph, x_t, node.outputs[0].shape, @"contiguous");
        if (y == nil)
            return false;
        ctx.bind(node.outputs[0].id, (__bridge void*)(y));
        return true;
    }

private:
    std::string name_;
};

// R1 — broadcast_to / pad / tile / repeat.
//
// ``broadcast_to`` and ``pad`` carry shape / pad info attribute-side;
// ``repeat`` and ``tile`` carry the repetition pattern.  All four are
// 1-input → 1-output and trace through the explicit ``on_op_io`` call
// in the engine forward.

class BroadcastToEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "broadcast_to"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return false;
        // Target shape is the recorded output shape on the OpNode.
        const auto& out_shape = node.outputs[0].shape;
        NSMutableArray<NSNumber*>* target =
            [NSMutableArray arrayWithCapacity:out_shape.size()];
        for (std::int64_t d : out_shape)
            [target addObject:[NSNumber numberWithLongLong:d]];
        ctx.bind(node.outputs[0].id, (__bridge void*)([graph broadcastTensor:x_t
                                              toShape:target
                                                 name:@"broadcast_to"]));
        return true;
    }
};

class PadEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "pad"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        auto pads_it = node.attrs.find("pads");
        if (pads_it == node.attrs.end())
            return false;
        const auto* pads =
            std::get_if<std::vector<std::int64_t>>(&pads_it->second);
        if (pads == nullptr || pads->size() % 2 != 0)
            return false;
        double constant_value = 0.0;
        if (auto c_it = node.attrs.find("constant"); c_it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&c_it->second)) constant_value = *p;
        }

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return false;

        const std::size_t ndim = pads->size() / 2;
        NSMutableArray<NSNumber*>* leading =
            [NSMutableArray arrayWithCapacity:ndim];
        NSMutableArray<NSNumber*>* trailing =
            [NSMutableArray arrayWithCapacity:ndim];
        for (std::size_t d = 0; d < ndim; ++d) {
            [leading addObject:[NSNumber numberWithLongLong:(*pads)[2 * d]]];
            [trailing addObject:[NSNumber numberWithLongLong:(*pads)[2 * d + 1]]];
        }
        ctx.bind(node.outputs[0].id,
                 (__bridge void*)(pad_or_identity(graph,
                                                  x_t,
                                                  leading,
                                                  trailing,
                                                  MPSGraphPaddingModeConstant,
                                                  constant_value,
                                                  @"pad")));
        return true;
    }
};

class TileEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "tile"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        auto it = node.attrs.find("reps");
        if (it == node.attrs.end())
            return false;
        const auto* reps = std::get_if<std::vector<std::int64_t>>(&it->second);
        if (reps == nullptr)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return false;

        NSMutableArray<NSNumber*>* mul =
            [NSMutableArray arrayWithCapacity:reps->size()];
        for (std::int64_t r : *reps)
            [mul addObject:[NSNumber numberWithLongLong:r]];
        ctx.bind(node.outputs[0].id, (__bridge void*)([graph tileTensor:x_t
                                 withMultiplier:mul
                                           name:@"tile"]));
        return true;
    }
};

// ``repeat`` — element-wise repetition (numpy semantics: [a,b,c] →
// [a,a,b,b,c,c] with n=2 along axis 0).  Realised by inserting a
// size-1 axis after the chosen one, broadcasting to size n, and
// reshaping so the new axis collapses into the original.  The shapes come
// from ``repeat_layout``: with a lead worth it the axes around the repeated
// one merge, so the intermediate stays at rank <= 4 (LCD-995).
class RepeatEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "repeat"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        auto ax_it = node.attrs.find("axis");
        auto r_it = node.attrs.find("repeats");
        if (ax_it == node.attrs.end() || r_it == node.attrs.end())
            return false;
        const auto* axp = std::get_if<std::int64_t>(&ax_it->second);
        const auto* rp = std::get_if<std::int64_t>(&r_it->second);
        if (axp == nullptr || rp == nullptr)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return false;

        const std::vector<std::int64_t> in_shape = shape_of_mps(x_t);
        const std::int64_t ndim = static_cast<std::int64_t>(in_shape.size());
        std::int64_t ax = *axp;
        if (ax < 0) ax += ndim;
        if (ax < 0 || ax >= ndim)
            return false;
        const RepeatLayout layout = repeat_layout(in_shape, static_cast<std::size_t>(ax), *rp);

        // 1) insert a size-1 axis after ``ax``: shape (..., D, 1, ...).
        MPSGraphTensor* shaped =
            [graph reshapeTensor:x_t withShape:shape_to_ns(layout.unit) name:nil];

        // 2) broadcast the inserted size-1 axis to ``reps``.
        MPSGraphTensor* tiled =
            [graph broadcastTensor:shaped toShape:shape_to_ns(layout.spread) name:nil];

        // 3) collapse the two axes back: (..., D*reps, ...).
        std::vector<std::int64_t> final_shape = in_shape;
        final_shape[static_cast<std::size_t>(ax)] = in_shape[static_cast<std::size_t>(ax)] * *rp;
        ctx.bind(node.outputs[0].id, (__bridge void*)([graph reshapeTensor:tiled
                                          withShape:shape_to_ns(final_shape)
                                               name:@"repeat"]));
        return true;
    }
};

struct ReshapeEmitterRegistrar {
    ReshapeEmitterRegistrar() {
        // The engine attaches OpScopeFull with the call-site op_name,
        // not the schema name — so we register one instance per
        // observed call name (lucid/_C/ops/utils/View.cpp's
        // ``build_view_output`` passes "view" / "reshape" / "squeeze" /
        // "unsqueeze" / "flatten" depending on the caller).
        register_emitter(std::make_unique<ReshapeFamilyEmitter>("view"));
        register_emitter(std::make_unique<ReshapeFamilyEmitter>("reshape"));
        register_emitter(std::make_unique<ReshapeFamilyEmitter>("squeeze"));
        register_emitter(std::make_unique<ReshapeFamilyEmitter>("unsqueeze"));
        register_emitter(std::make_unique<ReshapeFamilyEmitter>("flatten"));
        register_emitter(std::make_unique<ContiguousEmitter>("contiguous"));
        register_emitter(std::make_unique<ContiguousEmitter>("detach"));
        // R1 additions.
        register_emitter(std::make_unique<BroadcastToEmitter>());
        register_emitter(std::make_unique<PadEmitter>());
        register_emitter(std::make_unique<TileEmitter>());
        register_emitter(std::make_unique<RepeatEmitter>());
    }
};

[[maybe_unused]] static const ReshapeEmitterRegistrar g_reshape_registrar;

}  // namespace

}  // namespace lucid::compile
