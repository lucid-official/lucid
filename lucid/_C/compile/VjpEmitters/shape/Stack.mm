// lucid/_C/compile/VjpEmitters/shape/Stack.mm
//
// Misc small VJPs for data-rearrangement ops (P5):
//   * ``stack``     → bwd = unbind (slice + squeeze along stack-axis)
//   * ``unbind``    → bwd = stack (unsqueeze + concat along axis)
//   * ``chunk``     → bwd = concat (same shape as split bwd)
//   * ``pad``       → bwd = slice (strip the padded borders)
//   * ``tile``      → bwd = reduction (sum over tiled-out axes)
//   * ``topk``      → bwd = scatter on indices (sparse one-hot grad)
//
// Each is multi-input or multi-output in varying ways — care taken to
// match the forward emitter's I/O conventions verbatim.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../../OpEmitters/_AxisSplitLayout.h"
#include "../../OpEmitters/_SortOrder.h"
#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

// ────────────────────────────────────────────────────────────────────
// stack — N inputs along a NEW axis.  Output shape = inputs[0].shape
// with a new axis of size N inserted at ``axis``.
//
// Backward: slice grad along ``axis`` at each k, squeeze the axis out.
// Each input gets the squeezed slice as its gradient.
// ────────────────────────────────────────────────────────────────────
class StackVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "stack"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        auto ax_it = node.attrs.find("axis");
        if (ax_it == node.attrs.end()) return false;
        const auto* axp = std::get_if<std::int64_t>(&ax_it->second);
        if (axp == nullptr) return false;
        const std::int64_t axis = *axp;

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        if (g == nil || grad == nil) return false;

        std::vector<std::int64_t> grad_shape = shape_of_mps(grad);
        if (grad_shape.empty()) return false;
        // Build the squeezed-input shape (drop axis dim from grad_shape).
        std::vector<std::int64_t> piece_shape;
        piece_shape.reserve(grad_shape.size() - 1);
        for (std::size_t i = 0; i < grad_shape.size(); ++i) {
            if ((std::int64_t)i == axis) continue;
            piece_shape.push_back(grad_shape[i]);
        }
        NSArray<NSNumber*>* piece_ns = shape_to_ns(piece_shape);

        for (std::size_t k = 0; k < node.inputs.size(); ++k) {
            TensorId iid = node.inputs[k];
            if (iid < 0) return false;
            MPSGraphTensor* slab =
                [g sliceTensor:grad
                     dimension:(NSInteger)axis
                         start:(NSInteger)k
                        length:1
                          name:[NSString stringWithFormat:@"stack_vjp_slab%zu", k]];
            MPSGraphTensor* squeezed =
                [g reshapeTensor:slab
                       withShape:piece_ns
                            name:[NSString stringWithFormat:@"stack_vjp_p%zu", k]];
            bctx.accumulate_grad(iid, from_tensor(squeezed));
        }
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// unbind — 1 input → N outputs along ``axis`` (each output is a slice
// with the axis squeezed).  Multi-output op.
//
// Backward: unsqueeze each output's grad to insert the missing axis,
// then concat the N pieces.  Dead-slot grads (no demand from
// downstream) get zeros at the squeezed-piece shape.
// ────────────────────────────────────────────────────────────────────
class UnbindVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "unbind"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        auto ax_it = node.attrs.find("axis");
        if (ax_it == node.attrs.end()) return false;
        const auto* axp = std::get_if<std::int64_t>(&ax_it->second);
        if (axp == nullptr) return false;
        const std::int64_t axis = *axp;
        if (grad_outs.size() != node.outputs.size()) return false;

        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        if (g == nil || x == nil) return false;

        std::vector<std::int64_t> x_shape = shape_of_mps(x);
        if (x_shape.empty()) return false;
        // Piece shape = x_shape with the axis dim squeezed.
        std::vector<std::int64_t> piece_shape;
        piece_shape.reserve(x_shape.size() - 1);
        for (std::size_t i = 0; i < x_shape.size(); ++i) {
            if ((std::int64_t)i == axis) continue;
            piece_shape.push_back(x_shape[i]);
        }
        // Lifted shape = piece_shape with a size-1 axis at ``axis``.
        std::vector<std::int64_t> lifted_shape = x_shape;
        lifted_shape[axis] = 1;
        NSArray<NSNumber*>* lifted_ns = shape_to_ns(lifted_shape);

        // Find the dtype to use for zeros (use any present grad's dtype;
        // fall back to x's dtype).
        MPSDataType dt = x.dataType;
        for (void* gp : grad_outs) {
            if (gp != nullptr) { dt = as_tensor(gp).dataType; break; }
        }

        NSMutableArray<MPSGraphTensor*>* pieces =
            [NSMutableArray arrayWithCapacity:node.outputs.size()];
        for (std::size_t k = 0; k < node.outputs.size(); ++k) {
            MPSGraphTensor* p = nil;
            if (grad_outs[k] != nullptr) {
                MPSGraphTensor* gk = as_tensor(grad_outs[k]);
                p = [g reshapeTensor:gk withShape:lifted_ns name:nil];
            } else {
                p = [g constantWithScalar:0.0
                                    shape:lifted_ns
                                 dataType:dt];
            }
            [pieces addObject:p];
        }
        MPSGraphTensor* dx =
            [g concatTensors:pieces dimension:(NSInteger)axis name:@"unbind_vjp"];
        bctx.accumulate_grad(x_id, from_tensor(dx));
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// pad — strip the padded borders from grad.  Forward attrs include
// ``pads`` which is [left0, right0, left1, right1, ...] across all
// dims.  Backward: slice each dim by (left, length=input_dim).
// ────────────────────────────────────────────────────────────────────
class PadVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "pad"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        auto pads_it = node.attrs.find("pads");
        if (pads_it == node.attrs.end()) return false;
        const auto* pads =
            std::get_if<std::vector<std::int64_t>>(&pads_it->second);
        if (pads == nullptr || pads->size() % 2 != 0) return false;

        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        if (g == nil || grad == nil || x == nil) return false;

        std::vector<std::int64_t> x_shape = shape_of_mps(x);
        if (x_shape.empty()) return false;
        const std::size_t ndim = pads->size() / 2;
        if (ndim != x_shape.size()) return false;

        // Slice each dim by (left, length=x_shape[d]).
        MPSGraphTensor* dx = grad;
        for (std::size_t d = 0; d < ndim; ++d) {
            const std::int64_t left = (*pads)[2 * d];
            const std::int64_t length = x_shape[d];
            if (left == 0 && length == (std::int64_t)shape_of_mps(dx)[d])
                continue;  // no-op slice on this dim
            dx = [g sliceTensor:dx
                      dimension:(NSInteger)d
                          start:(NSInteger)left
                         length:(NSInteger)length
                           name:nil];
        }
        bctx.accumulate_grad(x_id, from_tensor(dx));
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// chunk — equal split with possibly uneven last piece.  Multi-output
// op; forward records ``axis`` + ``num_chunks``.  Backward = concat
// all output grads along axis (same as split bwd).
// ────────────────────────────────────────────────────────────────────
class ChunkVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "chunk"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        auto ax_it = node.attrs.find("axis");
        if (ax_it == node.attrs.end()) return false;
        const auto* axp = std::get_if<std::int64_t>(&ax_it->second);
        if (axp == nullptr) return false;
        const std::int64_t axis = *axp;
        if (grad_outs.size() != node.outputs.size()) return false;

        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        if (g == nil || x == nil) return false;
        // Pick dtype for any synthesised zero pieces — prefer the
        // chain dtype (from any present grad) so concat doesn't mix
        // F16 grad slices with F32 zero slices under autocast.
        MPSDataType dt = x.dataType;
        for (void* gp : grad_outs) {
            if (gp != nullptr) { dt = as_tensor(gp).dataType; break; }
        }

        NSMutableArray<MPSGraphTensor*>* pieces =
            [NSMutableArray arrayWithCapacity:node.outputs.size()];
        for (std::size_t k = 0; k < node.outputs.size(); ++k) {
            MPSGraphTensor* p = nil;
            if (grad_outs[k] != nullptr) {
                p = as_tensor(grad_outs[k]);
            } else {
                const auto& sh = node.outputs[k].shape;
                if (sh.empty()) return false;
                p = [g constantWithScalar:0.0
                                    shape:shape_to_ns(sh)
                                 dataType:dt];
            }
            [pieces addObject:p];
        }
        MPSGraphTensor* dx =
            [g concatTensors:pieces dimension:(NSInteger)axis name:@"chunk_vjp"];
        bctx.accumulate_grad(x_id, from_tensor(dx));
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// tile — repeats input along each dim by ``reps[d]``.  Backward: sum
// over the repeated dimensions to collapse back to the original shape.
//
// Implementation: reshape grad from (rep0 * d0, rep1 * d1, ...) to
// (rep0, d0, rep1, d1, ...) so the rep-axes are explicit, then sum
// over the rep-axes, then reshape back to input shape.
// ────────────────────────────────────────────────────────────────────
class TileVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "tile"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        auto it = node.attrs.find("reps");
        if (it == node.attrs.end()) return false;
        const auto* reps = std::get_if<std::vector<std::int64_t>>(&it->second);
        if (reps == nullptr) return false;

        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        if (g == nil || grad == nil || x == nil) return false;

        std::vector<std::int64_t> x_shape = shape_of_mps(x);
        if (x_shape.size() != reps->size()) return false;

        // A lead worth it: one rank <= 4 reduction per tiled axis, instead of one over
        // the rank-2n interleaved shape below (a slice loop in MPSGraph).
        if (const auto folds = tile_folds(x_shape, *reps)) {
            MPSGraphTensor* cur = grad;
            for (const TileFold& fold : *folds) {
                MPSGraphTensor* split =
                    [g reshapeTensor:cur withShape:shape_to_ns(fold.split) name:nil];
                MPSGraphTensor* summed = [g reductionSumWithTensor:split
                                                              axis:(NSInteger)fold.sum_axis
                                                              name:nil];
                cur = [g reshapeTensor:summed withShape:shape_to_ns(fold.folded) name:nil];
            }
            bctx.accumulate_grad(
                x_id, from_tensor([g reshapeTensor:cur
                                         withShape:shape_to_ns(x_shape)
                                              name:@"tile_vjp"]));
            return true;
        }

        // Build the interleaved shape [r0, d0, r1, d1, ...] and the
        // axes-to-reduce list [0, 2, 4, ...].
        std::vector<std::int64_t> interleaved;
        interleaved.reserve(2 * x_shape.size());
        NSMutableArray<NSNumber*>* rep_axes =
            [NSMutableArray arrayWithCapacity:x_shape.size()];
        for (std::size_t d = 0; d < x_shape.size(); ++d) {
            interleaved.push_back((*reps)[d]);
            interleaved.push_back(x_shape[d]);
            [rep_axes addObject:[NSNumber numberWithLongLong:(long long)(2 * d)]];
        }
        MPSGraphTensor* g_reshape =
            [g reshapeTensor:grad withShape:shape_to_ns(interleaved) name:nil];
        MPSGraphTensor* g_summed =
            [g reductionSumWithTensor:g_reshape axes:rep_axes name:nil];
        MPSGraphTensor* dx =
            [g reshapeTensor:g_summed withShape:shape_to_ns(x_shape) name:@"tile_vjp"];
        bctx.accumulate_grad(x_id, from_tensor(dx));
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// topk — multi-output (values + indices).  Backward: scatter the
// values-grad onto a zero buffer at the indices.  d(indices) = none
// (integer).
// ────────────────────────────────────────────────────────────────────
class TopkVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "topk"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || node.outputs.size() != 2) return false;
        if (grad_outs.size() != 2) return false;
        // Grad on values (output[0]) is the differentiable path;
        // indices (output[1]) gets no grad.  If grad_outs[0] is null,
        // d(x) = 0 effectively — skip.
        if (grad_outs[0] == nullptr) return true;

        auto ax_it = node.attrs.find("axis");
        if (ax_it == node.attrs.end()) return false;
        const auto* axp = std::get_if<std::int64_t>(&ax_it->second);
        if (axp == nullptr) return false;
        const NSInteger axis = (NSInteger)*axp;

        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        if (![g respondsToSelector:@selector(scatterAlongAxis:withDataTensor:updatesTensor:indicesTensor:mode:name:)])
            return false;

        MPSGraphTensor* v_grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        // The forward emitter binds outputs[1].id to the indices tensor;
        // re-read it via forward() if it's been bound.
        TensorId idx_id = node.outputs[1].id;
        MPSGraphTensor* indices = as_tensor(bctx.forward(idx_id));
        // The forward binds the indices only when something reads them, so a
        // caller that used only the values — the usual case — left this nil
        // and every such graph fell back.  Rank again for them.
        if (indices == nil && x != nil) {
            auto k_it = node.attrs.find("k");
            const auto* kp = k_it == node.attrs.end() ? nullptr : std::get_if<std::int64_t>(&k_it->second);
            if (kp == nullptr || *kp <= 0) return false;
            indices = graph_topk(g, x, axis, (NSInteger)*kp).indices;
        }
        if (g == nil || x == nil || v_grad == nil || indices == nil) return false;

        std::vector<std::int64_t> x_shape = shape_of_mps(x);
        if (x_shape.empty()) return false;

        // Base zero buffer dtype = grad's chain dtype so scatter-add
        // matches its updates dtype (autocast: x may be F32, grad F16).
        MPSGraphTensor* base =
            [g constantWithScalar:0.0
                            shape:shape_to_ns(x_shape)
                         dataType:v_grad.dataType];
        MPSGraphTensor* dx =
            [g scatterAlongAxis:axis
                  withDataTensor:base
                   updatesTensor:v_grad
                   indicesTensor:indices
                            mode:MPSGraphScatterModeAdd
                            name:@"topk_vjp"];
        bctx.accumulate_grad(x_id, from_tensor(dx));
        return true;
    }
};

struct StackVjpRegistrar {
    StackVjpRegistrar() {
        register_vjp_emitter(std::make_unique<StackVjp>());
        register_vjp_emitter(std::make_unique<UnbindVjp>());
        register_vjp_emitter(std::make_unique<PadVjp>());
        register_vjp_emitter(std::make_unique<ChunkVjp>());
        register_vjp_emitter(std::make_unique<TileVjp>());
        register_vjp_emitter(std::make_unique<TopkVjp>());
    }
};

[[maybe_unused]] static const StackVjpRegistrar g_stack_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
