// lucid/_C/compile/OpEmitters/_SignedZero.h
//
// Where a zero constant meets an addition inside a compiled graph (LCD-446).
//
// MPSGraph rewrites ``x + c`` and ``x - c`` to ``x`` whenever ``c`` is a
// constant splat of either zero — at every optimization level, before anything
// runs.  IEEE 754 agrees only for the zero that is the identity: ``x + (-0)``
// and ``x - (+0)`` are ``x`` for every ``x``, but ``(-0) + (+0)`` and
// ``(-0) - (-0)`` are ``+0``, so the rewrite hands back ``-0`` where eager
// gives ``+0``.  ``0 - x``, ``x * 1``, ``x * 0`` and ``x / 1`` were measured
// too and MPSGraph keeps them exact.
//
// A trace's zero constants come from the ``full`` / ``zeros`` factories (a
// Python scalar operand is materialised by ``full``).  Those emitters mark a
// floating ±0 fill in the builder's saved-state table, keyed by their output
// id under slot names no VJP uses, and ``emit_node`` (MpsBuilder.mm) carries
// the mark across an op that only moves or widens values — a reshape of a
// zero splat is still one, and MPSGraph folds it just the same.  The add and
// sub emitters look the mark up and, when the operand is the zero that is not
// the identity, spell the sum as ``x + floor(min(|x|, +0))``.  The second term
// is +0 for every non-NaN ``x`` but is not a constant, so MPSGraph has nothing
// to fold and the hardware addition gives IEEE's answer.
//
// The term must also be invisible to MPSGraph's autodiff, the fallback when a
// manual VJP is missing.  Its true derivative is 0, but ``min``'s gradient
// multiplies the incoming one by a 0/1 mask, so an infinite gradient became
// NaN.  MPSGraph has no public stop-gradient; ``floor``'s gradient is built as
// zeros, not as a product, so it cuts the path and ``x`` receives the incoming
// gradient untouched, as the manual add VJP gives it.  A select on ``x == 0``
// would be cheaper still, but autodiff returns a wrong gradient through it.
//
// Objective-C++ only.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <string_view>
#include <unordered_set>

#include "../../core/Dtype.h"
#include "../TraceIR.h"
#include "OpEmitter.h"

namespace lucid::compile {

inline constexpr const char* kPosZeroSplatSlot = "signed_zero:pos_splat";
inline constexpr const char* kNegZeroSplatSlot = "signed_zero:neg_splat";
inline constexpr const char* kPosOneSplatSlot = "signed_zero:pos_one_splat";
inline constexpr const char* kNegOneSplatSlot = "signed_zero:neg_one_splat";

// Record that ``out_id`` is a splat of ``fill``, if that is a floating ±0 or ±1.
inline void mark_zero_splat(
    BuilderContext& ctx, TensorId out_id, Dtype dtype, double fill, MPSGraphTensor* splat) {
    if (!is_floating_point(dtype))
        return;
    const char* slot = nullptr;
    if (fill == 0.0)
        slot = std::signbit(fill) ? kNegZeroSplatSlot : kPosZeroSplatSlot;
    else if (fill == 1.0)
        slot = kPosOneSplatSlot;
    else if (fill == -1.0)
        slot = kNegOneSplatSlot;
    if (slot != nullptr)
        ctx.stash_saved(out_id, slot, (__bridge void*)splat);
}

inline bool is_pos_zero_splat(const BuilderContext& ctx, TensorId id) {
    return ctx.resolve_saved(id, kPosZeroSplatSlot) != nullptr;
}

inline bool is_neg_zero_splat(const BuilderContext& ctx, TensorId id) {
    return ctx.resolve_saved(id, kNegZeroSplatSlot) != nullptr;
}

inline bool is_pos_one_splat(const BuilderContext& ctx, TensorId id) {
    return ctx.resolve_saved(id, kPosOneSplatSlot) != nullptr;
}

inline bool is_neg_one_splat(const BuilderContext& ctx, TensorId id) {
    return ctx.resolve_saved(id, kNegOneSplatSlot) != nullptr;
}

inline MPSGraphTensor* constant_splat_like(MPSGraph* g, MPSGraphTensor* t, double value) {
    MPSGraphTensor* c = [g constantWithScalar:value dataType:t.dataType];
    if (t.shape == nil || t.shape.count == 0)
        return c;
    for (NSNumber* dim in t.shape) {
        if (dim.longLongValue <= 0)
            return [g broadcastTensor:c toShapeTensor:[g shapeOfTensor:t name:nil] name:nil];
    }
    return [g constantWithScalar:value shape:t.shape dataType:t.dataType];
}

inline MPSGraphTensor* zero_splat_like(MPSGraph* g, MPSGraphTensor* t) {
    return constant_splat_like(g, t, 0.0);
}

// ``x + pos_zero`` as IEEE computes it, where ``pos_zero`` is a +0 splat (it
// also carries the broadcast shape of the sum).
inline MPSGraphTensor* add_pos_zero(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* pos_zero) {
    MPSGraphTensor* zero_unless_nan = [g minimumWithPrimaryTensor:[g absoluteWithTensor:x name:nil]
                                                  secondaryTensor:pos_zero
                                                             name:nil];
    return [g additionWithPrimaryTensor:x
                        secondaryTensor:[g floorWithTensor:zero_unless_nan name:nil]
                                   name:@"add_pos_zero"];
}

// ``x - pos_zero`` (and ``x + (-0)``, ``x * 1``, ``x / 1``): keeps ``-0`` as
// ``-0`` while executing a real hardware op so f32/bf16 subnormals flush to
// signed zero as in eager Metal (LCD-452).
inline MPSGraphTensor* sub_pos_zero(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* pos_zero) {
    MPSGraphTensor* zero_unless_nan = [g minimumWithPrimaryTensor:[g absoluteWithTensor:x name:nil]
                                                  secondaryTensor:pos_zero
                                                             name:nil];
    return [g subtractionWithPrimaryTensor:x
                           secondaryTensor:[g floorWithTensor:zero_unless_nan name:nil]
                                      name:@"sub_pos_zero"];
}

// Ops whose output holds only values of their first input: a splat in is a
// splat out.  ``astype`` qualifies only into a floating dtype, which keeps a
// zero's sign; a marked input is already floating.
inline bool moves_values_only(const OpNode& node) {
    static const std::unordered_set<std::string_view> kMoves{
        "broadcast_to", "contiguous", "detach",  "flatten", "flip",      "permute", "repeat",
        "reshape",      "roll",       "squeeze", "tile",    "unsqueeze", "view"};
    if (node.name == "astype")
        return !node.outputs.empty() && is_floating_point(node.outputs[0].dtype);
    return kMoves.count(node.name) != 0;
}

// Carry a ±0 / ±1 splat mark from ``node``'s input to its output.  ``emit_node``
// calls it after every op, so the rule lives in one place.
inline void propagate_zero_splat(BuilderContext& ctx, const OpNode& node) {
    if (node.inputs.empty() || node.outputs.size() != 1 || !moves_values_only(node))
        return;
    void* out = ctx.resolve(node.outputs[0].id);
    if (out == nullptr)
        return;
    for (const char* slot :
         {kPosZeroSplatSlot, kNegZeroSplatSlot, kPosOneSplatSlot, kNegOneSplatSlot})
        if (ctx.resolve_saved(node.inputs[0], slot) != nullptr)
            ctx.stash_saved(node.outputs[0].id, slot, out);
}

}  // namespace lucid::compile
