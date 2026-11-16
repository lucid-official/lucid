// lucid/_C/compile/OpEmitters/_CancellingSums.h
//
// Where an addition and a subtraction cancel inside a compiled graph (LCD-1033).
//
// MPSGraph's algebraic simplifier cancels a sum against one of its own terms:
//
//     (a + b) - a  ->  b          a - (a - b)  ->  b
//     (a + b) - b  ->  a          (a - b) + b  ->  a        b + (a - b)  ->  a
//
// and, with constants, ``(x + C) - C`` and ``(x + C) + (-C)`` go to ``x``.  In
// real arithmetic that is the same value.  In float32 the sum rounds before the
// subtraction reads it, so the left side is the *rounding error* of the sum,
// and the right side is not: ``(1e-8 + 1) - 1`` is ``0`` where the rewrite
// hands back ``1e-8``.  Anything that recovers that error is cancelled with it
// — ``log1p`` (``u = 1 + x``, then ``u - 1``), the Kahan / Fast2Sum
// compensation ``(t - s) - y``, the ``(x + 2^k) - 2^k`` round-to-integer trick.
// Measured on 4096 random float32 pairs, 1900 of them differ from eager by
// the whole error term, and ``(x + 1e6) - 1e6`` differs in all of them.
//
// The simplifier matches the operands by identity, so it needs the sum itself
// to be what the subtraction reads.  ``commit_rounding`` puts one value-neutral
// op between the two (the launder ``sub_pos_zero`` already uses against the
// zero fold, _SignedZero.h), and the pattern no longer matches.  The pass is
// local: each add / sub emitter records its operands under slot names no VJP
// uses, keyed by its output id, and the add / sub that reads that output asks
// ``commit_before`` whether its own operands close a cancelling pair.  Only
// those pairs pay for the extra ops.
//
// A constant is told from another constant by kind, not by value: any ``full``
// splat marked in _SignedZero.h counts.  ``(x + C1) - C2`` with C1 != C2 is not
// cancelled by the simplifier, but folding it to ``x + (C1 - C2)`` rounds once
// where eager rounds twice, so committing it too is the faithful answer, and the
// pair is rare enough that the three extra ops are not a cost anyone sees.
//
// What the guard does not reach (each pinned by a strict xfail in
// test_cancelling_sums_compiled.py):
//
// * float16 / bfloat16: the commit is emitted, but Lucid's half pipeline keeps a
//   fused chain's intermediates at float32 and rounds once, so ``(x + 1) - 1``
//   still comes back ``x``; the cause is the width of the op-math, not a rewrite.
// * complex64: ``is_floating_point`` excludes it, so an add / sub of complex
//   values is neither recorded nor committed.
// * a view between the sum and the subtraction: a moves-only op (``reshape``,
//   ``detach``, a full slice) binds a new tensor, so the operands no longer
//   match by identity, and MPSGraph sees through the view when it simplifies.
// * a term recomputed rather than reused (``(x * k) - ((x * k) - x)``, or
//   ``((a + b) + c) - (a + b)`` with two add nodes): two nodes are two tensors
//   here, and MPSGraph merges them before it simplifies.
//
// Objective-C++ only.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "../../core/Dtype.h"
#include "../TraceIR.h"
#include "OpEmitter.h"
#include "_SignedZero.h"

namespace lucid::compile {

inline constexpr const char* kSumLhsSlot = "cancel:sum_lhs";
inline constexpr const char* kSumRhsSlot = "cancel:sum_rhs";
inline constexpr const char* kDiffLhsSlot = "cancel:diff_lhs";
inline constexpr const char* kDiffRhsSlot = "cancel:diff_rhs";
inline constexpr const char* kConstTermSlot = "cancel:const_term";

enum class Additive { Add, Sub };

// Which operands of the add / sub about to be emitted must be committed.
struct RoundingCommit {
    bool lhs = false;
    bool rhs = false;
};

// ``x`` itself, as a value no later rewrite can see through.
inline MPSGraphTensor* commit_rounding(MPSGraph* g, MPSGraphTensor* x) {
    return sub_pos_zero(g, x, zero_splat_like(g, x));
}

namespace cancelling_sums_detail {

inline bool is_traced_value(const BuilderContext& ctx, TensorId id, void* value) {
    return value != nullptr && ctx.resolve(id) == value;
}

// ``id`` is a sum / difference with a constant term (a bias, an offset).
inline bool has_const_term(const BuilderContext& ctx, TensorId id) {
    return ctx.resolve_saved(id, kConstTermSlot) != nullptr;
}

// ``id`` is a difference ``x - y`` and ``other`` is ``y``: adding ``other`` back
// returns ``x``.
inline bool difference_returns_with(const BuilderContext& ctx, TensorId diff, TensorId other) {
    return is_traced_value(ctx, other, ctx.resolve_saved(diff, kDiffRhsSlot));
}

// ``id`` is a sum ``x + y`` and ``other`` is ``x`` or ``y``: taking ``other`` off
// returns the rest.
inline bool sum_cancels_with(const BuilderContext& ctx, TensorId sum, TensorId other) {
    return is_traced_value(ctx, other, ctx.resolve_saved(sum, kSumLhsSlot)) ||
           is_traced_value(ctx, other, ctx.resolve_saved(sum, kSumRhsSlot));
}

inline bool is_floating(const OpNode& node) {
    return !node.outputs.empty() && is_floating_point(node.outputs[0].dtype);
}

}  // namespace cancelling_sums_detail

// The commits the add / sub about to be emitted owes its operands ``P`` and ``Q``.
// ``add``: a difference read together with the term it removed (``(x - y) + y``),
// or a constant offset read together with a constant.  ``sub``: a sum read
// together with one of its terms (``(x + y) - y``), a constant offset read
// together with a constant, ``x - (x - y)``, or a constant minus a constant
// offset (``C - (C - x)``).
inline RoundingCommit commit_before(const BuilderContext& ctx, const OpNode& node, Additive kind) {
    using namespace cancelling_sums_detail;
    RoundingCommit out;
    if (node.inputs.size() != 2 || !is_floating(node))
        return out;
    const TensorId p = node.inputs[0];
    const TensorId q = node.inputs[1];
    const bool const_offset_meets_const_lhs = has_const_term(ctx, p) && is_const_splat(ctx, q);
    if (kind == Additive::Sub) {
        out.lhs = sum_cancels_with(ctx, p, q) || const_offset_meets_const_lhs;
        out.rhs = is_traced_value(ctx, p, ctx.resolve_saved(q, kDiffLhsSlot)) ||
                  (is_const_splat(ctx, p) && has_const_term(ctx, q));
        return out;
    }
    out.lhs = difference_returns_with(ctx, p, q) || const_offset_meets_const_lhs;
    out.rhs =
        difference_returns_with(ctx, q, p) || (has_const_term(ctx, q) && is_const_splat(ctx, p));
    return out;
}

// Note the operands of a floating add / sub under its output id, for the next
// add / sub that reads it.
inline void record_additive(BuilderContext& ctx, const OpNode& node, Additive kind) {
    if (node.inputs.size() != 2 || node.outputs.empty() ||
        !cancelling_sums_detail::is_floating(node))
        return;
    void* lhs = ctx.resolve(node.inputs[0]);
    void* rhs = ctx.resolve(node.inputs[1]);
    const TensorId out = node.outputs[0].id;
    ctx.stash_saved(out, kind == Additive::Add ? kSumLhsSlot : kDiffLhsSlot, lhs);
    ctx.stash_saved(out, kind == Additive::Add ? kSumRhsSlot : kDiffRhsSlot, rhs);
    const bool lhs_const = is_const_splat(ctx, node.inputs[0]);
    const bool rhs_const = is_const_splat(ctx, node.inputs[1]);
    if (lhs_const != rhs_const)
        ctx.stash_saved(out, kConstTermSlot, lhs_const ? lhs : rhs);
}

}  // namespace lucid::compile
