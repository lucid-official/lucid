// lucid/_C/compile/VjpEmitters/_VjpHelpers.h
//
// Shared inline helpers for VJP ``.mm`` files: cast helpers,
// broadcast-aware unreduce, ones-like-loss seed, axis arithmetic.
//
// Mirrors the role of :file:`OpEmitters/_AttrHelpers.h` but on the
// backward side.  Pure C++/Objective-C++ — safe from ``.mm`` only.

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <vector>

#include "../../core/Dtype.h"
#include "../MpsDtype.h"
#include "../OpEmitters/OpEmitter.h"
#include "../OpEmitters/_AttrHelpers.h"
#include "../TraceIR.h"
#include "VjpEmitter.h"

namespace lucid::compile {

// Cast helper: ``void*`` ↔ ``MPSGraphTensor*``.  Each VJP body would
// otherwise litter every line with ``(__bridge MPSGraphTensor*)``.
[[maybe_unused]] inline MPSGraphTensor* as_tensor(void* p) {
    return (__bridge MPSGraphTensor*)p;
}

[[maybe_unused]] inline void* from_tensor(MPSGraphTensor* t) {
    return (__bridge void*)t;
}

// Cast ``t`` to ``target`` if its dtype doesn't already match — no-op
// otherwise.  Returns the (possibly-cast) tensor.  Used by VJPs to
// reconcile mixed-dtype operands under autocast: every VJP convention
// is that all intermediate arithmetic runs in ``grad.dataType``
// (the chain's dominant dtype).  Forward activations like ``x`` /
// ``gamma`` / ``beta`` cast to that dtype at the VJP prologue; the
// astype VJP at the param boundary handles the final cast back to
// the param's master dtype on the way to the optimizer.
//
// Without this reconciliation, under ``with autocast(F16)`` the
// downstream binary builders fail MPSGraph's same-dtype check
// (``'mps.multiply' op requires the same element type for all
// operands and results``), exposing every VJP that calls
// ``multiplicationWithPrimaryTensor:`` / ``additionWithPrimaryTensor:``
// / etc. on a mix of grad (F16) + forward activation (F32 master).
//
// The convention is grad-dominant (not output-dominant) because the
// grad's dtype determines the downstream chain's dtype — we minimize
// per-VJP casts by aligning to the side that's already in the right
// precision for whatever comes next in the backward walk.
[[maybe_unused]] inline MPSGraphTensor*
cast_if_needed(MPSGraph* g, MPSGraphTensor* t, MPSDataType target) {
    return cast_mps_tensor(g, t, target, @"vjp_dtype_cast");
}

// Resolve the compute dtype for ``node``'s backward pass: the forward
// output's floating MPSDataType when recorded, else ``fallback``.  Matches
// eager autograd, which executes each backward formula at the forward op's
// effective dtype (``eff_dt``) and casts each input gradient back to its
// input's dtype upon accumulation.
[[maybe_unused]] inline MPSDataType vjp_compute_dtype(const OpNode& node, MPSDataType fallback) {
    if (!node.outputs.empty()) {
        if (const auto dt = mps_dtype_of(node.outputs[0].dtype)) {
            if ((*dt & MPSDataTypeFloatBit) != 0)
                return *dt;
        }
    }
    return fallback;
}

[[maybe_unused]] inline bool is_vjp_half_f32_0d_pair(const OpNode& node,
                                                     MPSDataType target_dt,
                                                     MPSGraphTensor* a_t,
                                                     MPSGraphTensor* b_t) {
    if (target_dt != MPSDataTypeFloat16 && target_dt != MPSDataTypeBFloat16)
        return false;
    if (node.name != "mul" && node.name != "div" && node.name != "floordiv")
        return false;
    auto is_0d_f32 = [](MPSGraphTensor* t) {
        return t != nil && t.dataType == MPSDataTypeFloat32 && t.shape != nil && t.shape.count == 0;
    };
    return (a_t.dataType == target_dt && is_0d_f32(b_t)) ||
           (b_t.dataType == target_dt && is_0d_f32(a_t));
}

// Lift a ``Shape`` (``std::vector<int64_t>``) to an ``NSArray<NSNumber*>``.
[[maybe_unused]] inline NSArray<NSNumber*>* shape_to_ns(const std::vector<std::int64_t>& s) {
    NSMutableArray<NSNumber*>* out = [NSMutableArray arrayWithCapacity:s.size()];
    for (std::int64_t d : s)
        [out addObject:[NSNumber numberWithLongLong:d]];
    return out;
}

// ``[graph constantWithScalar:1.0 shape:loss_shape dataType:dt]``.
//
// Used by the walker to seed ``∂loss/∂loss = 1`` before iterating the
// trace in reverse.  Explicit shape so an N-D loss (per-sample loss
// w/ ``reduction='none'``) seeds correctly; using
// ``constantWithScalar:dataType:`` (0-D) would broadcast on the first
// VJP and confuse the unreduce.
[[maybe_unused]] inline MPSGraphTensor*
ones_like_loss(MPSGraph* graph, const std::vector<std::int64_t>& loss_shape, Dtype dtype) {
    // A loss is a real float, which always has an MPSGraph type.
    return [graph constantWithScalar:1.0
                               shape:shape_to_ns(loss_shape)
                            dataType:mps_dtype_of(dtype).value_or(MPSDataTypeFloat32)];
}

// Compute the axes that need to be reduce-summed in order to go from
// ``broadcast`` shape to ``target`` shape.  Mirrors the eager
// :func:`broadcast_back_for_reduce`'s axis computation.  Broadcast
// rules: align right, treat missing leading dims as 1, dims that are
// 1 in ``target`` but > 1 in ``broadcast`` were broadcast.
[[maybe_unused]] inline std::vector<std::int64_t>
axes_for_unreduce(const std::vector<std::int64_t>& broadcast,
                  const std::vector<std::int64_t>& target) {
    std::vector<std::int64_t> axes;
    const std::size_t br = broadcast.size();
    const std::size_t tr = target.size();
    // Leading axes that exist in broadcast but not in target → all reduced.
    for (std::size_t i = 0; i + tr < br; ++i)
        axes.push_back(static_cast<std::int64_t>(i));
    // Aligned trailing axes: size-1 in target but >1 in broadcast → reduce.
    for (std::size_t j = 0; j < tr; ++j) {
        const std::size_t i = br - tr + j;
        if (target[j] == 1 && broadcast[i] != 1)
            axes.push_back(static_cast<std::int64_t>(i));
    }
    return axes;
}

// Broadcast-back: reduce ``grad`` (shape == ``broadcast``) back to
// ``target`` by summing over the broadcast-expanded axes, then reshape
// to match ``target`` rank exactly.
//
// If ``broadcast == target`` (no broadcast happened) this returns
// ``grad`` unchanged.
[[maybe_unused]] inline MPSGraphTensor* unreduce_impl(MPSGraph* graph,
                                                      MPSGraphTensor* grad,
                                                      const std::vector<std::int64_t>& broadcast,
                                                      const std::vector<std::int64_t>& target) {
    if (broadcast == target)
        return grad;
    std::vector<std::int64_t> axes = axes_for_unreduce(broadcast, target);
    MPSGraphTensor* reduced = grad;
    if (!axes.empty()) {
        NSMutableArray<NSNumber*>* axes_ns = [NSMutableArray arrayWithCapacity:axes.size()];
        for (std::int64_t a : axes)
            [axes_ns addObject:[NSNumber numberWithLongLong:a]];
        reduced = [graph reductionSumWithTensor:reduced axes:axes_ns name:nil];
    }
    // Reshape to exact target shape (collapse the size-1 axes left
    // behind by keepdim=true reductionSum, plus any leading axes).
    NSArray<NSNumber*>* target_ns = shape_to_ns(target);
    return [graph reshapeTensor:reduced withShape:target_ns name:nil];
}

// Shape accessor that pulls the producer's recorded output shape for
// ``tid`` out of the trace.  Used when a VJP needs to know "what shape
// was input k of this op?" without materialising a forward tensor.
//
// Walks ``trace.ops`` once looking for the node whose outputs include
// ``tid``.  Returns an empty Shape if ``tid`` is an external feed (not
// produced by any node) — callers must handle that fallback (typically
// by looking at the forward tensor's ``.shape`` directly).
[[maybe_unused]] inline std::vector<std::int64_t> shape_of_tid(const TraceGraph& trace,
                                                               TensorId tid) {
    for (const auto& node : trace.ops) {
        for (const auto& out : node.outputs) {
            if (out.id == tid)
                return out.shape;
        }
    }
    return {};
}

// As above but pulls the dtype.  Defaults to F32 if not found.
[[maybe_unused]] inline Dtype dtype_of_tid(const TraceGraph& trace, TensorId tid) {
    for (const auto& node : trace.ops) {
        for (const auto& out : node.outputs) {
            if (out.id == tid)
                return out.dtype;
        }
    }
    return Dtype::F32;
}

// Extract the int64 vector at ``shape`` from an MPSGraphTensor's
// ``.shape`` NSArray<NSNumber*>.  Convenient when only the live
// MPSGraph shape is available (e.g. forward-feed placeholders).
[[maybe_unused]] inline std::vector<std::int64_t> shape_of_mps(MPSGraphTensor* t) {
    std::vector<std::int64_t> out;
    if (t == nil || t.shape == nil)
        return out;
    out.reserve(t.shape.count);
    for (NSNumber* n in t.shape)
        out.push_back((std::int64_t)n.longLongValue);
    return out;
}

// ════════════════════════════════════════════════════════════════════
// Per-arity skeletons.  Most VJPs follow one of two patterns: single
// input (activation / reshape / reduction) or binary input (arith,
// matmul).  These helpers absorb the resolve-forward, nil-guard, and
// cast boilerplate so the per-VJP body stays focused on the math.
// ════════════════════════════════════════════════════════════════════

// Unary VJP skeleton — for ops with exactly one differentiable input.
//
// ``body`` is invoked as ``body(graph, x_t, grad_t)`` and must return
// the gradient w.r.t. ``x``, OR ``nil`` to abort the VJP.  On success
// the helper calls :func:`BackwardContext::accumulate_grad(x_id, dx)`
// and returns ``true``; on any guard failure it returns ``false``.
//
// Mixed-dtype reconciliation under autocast (the chain-dominant
// convention — see ``cast_if_needed``): by default the forward
// activation ``x_t`` is cast to ``grad_t.dataType`` before the body
// runs, so all body arithmetic is in one dtype.  VJPs that genuinely
// need the original forward dtype (e.g. ``AstypeVjp`` which casts the
// grad *back* to the input dtype) opt out by passing
// ``cast_forward_to_grad=false``.
template <class Body>
inline bool emit_unary_vjp(BackwardContext& bctx,
                           const OpNode& node,
                           const std::vector<void*>& grad_outs,
                           Body body,
                           bool cast_forward_to_grad = true) {
    if (node.inputs.size() != 1 || grad_outs.empty() || grad_outs[0] == nullptr)
        return false;
    TensorId x_id = node.inputs[0];
    if (x_id < 0)
        return false;
    MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
    MPSGraphTensor* go = as_tensor(grad_outs[0]);
    MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
    if (g == nil || x == nil || go == nil)
        return false;
    if (cast_forward_to_grad) {
        const MPSDataType dt = vjp_compute_dtype(node, go.dataType);
        x = cast_if_needed(g, x, dt);
        go = cast_if_needed(g, go, dt);
    }
    MPSGraphTensor* dx = body(g, x, go);
    if (dx == nil)
        return false;
    bctx.accumulate_grad(x_id, from_tensor(dx));
    return true;
}

// Resolved context for a single-output binary VJP.  Filled by
// :func:`unpack_binary`; consumed by the per-op gradient body.  Holds
// graph + cached forward tensors + the input shapes and the
// (post-broadcast) output shape so unreduce can be applied per input.
struct BinaryVjpCtx {
    MPSGraph* g = nil;
    MPSGraphTensor* go = nil;  // incoming grad
    MPSGraphTensor* a = nil;   // forward a
    MPSGraphTensor* b = nil;   // forward b
    TensorId a_id = -1;
    TensorId b_id = -1;
    std::vector<std::int64_t> a_shape;
    std::vector<std::int64_t> b_shape;
    std::vector<std::int64_t> out_shape;
    bool ok = false;
};

// Resolve a two-input single-output VJP node and return a populated
// :class:`BinaryVjpCtx`.  ``ok == false`` signals any guard failure
// (wrong arity, missing forward binding, etc.).
inline BinaryVjpCtx
unpack_binary_vjp(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) {
    BinaryVjpCtx c;
    if (node.inputs.size() != 2 || grad_outs.empty() || grad_outs[0] == nullptr)
        return c;
    c.a_id = node.inputs[0];
    c.b_id = node.inputs[1];
    if (c.a_id < 0 || c.b_id < 0)
        return c;
    c.g = (__bridge MPSGraph*)bctx.graph();
    c.go = as_tensor(grad_outs[0]);
    c.a = as_tensor(bctx.forward(c.a_id));
    c.b = as_tensor(bctx.forward(c.b_id));
    if (c.g == nil || c.go == nil || c.a == nil || c.b == nil)
        return c;
    const MPSDataType target_dt = vjp_compute_dtype(node, c.go.dataType);
    const MPSDataType op_dt =
        is_vjp_half_f32_0d_pair(node, target_dt, c.a, c.b) ? MPSDataTypeFloat32 : target_dt;
    c.a_shape = shape_of_mps(c.a);
    c.b_shape = shape_of_mps(c.b);
    c.go = cast_if_needed(c.g, c.go, op_dt);
    c.a = cast_if_needed(c.g, c.a, op_dt);
    c.b = cast_if_needed(c.g, c.b, op_dt);
    if (!node.outputs.empty())
        c.out_shape = node.outputs[0].shape;
    if (c.out_shape.empty())
        c.out_shape = shape_of_mps(c.go);
    c.ok = true;
    return c;
}

}  // namespace lucid::compile
