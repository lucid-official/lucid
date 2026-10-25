// lucid/_C/compile/OpEmitters/_AttrHelpers.h
//
// Shared inline helpers for emitter ``.mm`` files: typed accessors for
// the :type:`AttributeValue` variant on :class:`OpNode`, plus a small
// utility to build an ``NSArray<NSNumber*>*`` axis list from a
// :class:`Shape`.
//
// Each .mm in a sub-package (core/ linalg/ shape/ reduce/ nn/ misc/)
// includes this header to avoid duplicating the same five getters at
// the top of every file.  Marked ``inline`` so a copy is emitted in
// each translation unit and the linker picks one — no ODR conflicts.
//
// Leading underscore in the filename keeps the header sorted to the
// top of directory listings and signals "internal to this directory"
// to anyone browsing the tree.
//
// Header is pure C++/Objective-C++ — safe to include from ``.mm``
// files but not from plain ``.cpp`` (since it touches NSArray /
// NSNumber).

#pragma once

#import <Foundation/Foundation.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <variant>
#include <vector>

#include "../TraceIR.h"
#include "OpEmitter.h"

namespace lucid::compile {

// Pull an ``int64_t`` attribute from ``node`` by key, falling back to
// ``def`` if the key is absent or the stored variant alternative
// doesn't match.  Used for axis / dim / kernel_size / padding / etc.
inline std::int64_t int_attr(const OpNode& node, const char* key, std::int64_t def) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return def;
    const auto* p = std::get_if<std::int64_t>(&it->second);
    return p ? *p : def;
}

// A 0/1 flag the forward may record as an int, a bool, or a one-element
// list.  Pooling records ``ceil_mode`` and ``count_include_pad`` as ``[1]``,
// which ``int_attr`` does not read — it answered the default, so every
// compiled ceil-mode pool floored and every ``count_include_pad=False``
// average counted the padding.
[[maybe_unused]] inline bool flag_attr(const OpNode& node, const char* key, bool def) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return def;
    if (const auto* p = std::get_if<std::int64_t>(&it->second))
        return *p != 0;
    if (const auto* p = std::get_if<bool>(&it->second))
        return *p;
    if (const auto* p = std::get_if<std::vector<std::int64_t>>(&it->second))
        return p->empty() ? def : (*p)[0] != 0;
    return def;
}

// Ceil-mode average pooling that counts the padding.  The reference divides
// a window hanging past the padded input by the part inside it; MPSGraph,
// counting the padding, by the whole kernel — no option matches.  Without
// padding there is nothing to count and the reference's divisor is the
// in-bounds part, which is MPSGraph's divisor when it does not count the
// padding.  Returns false when the pairing must stay eager, else settles
// ``include_zero_pad`` (ResNeSt's shortcut pool is the unpadded case).
[[maybe_unused]] inline bool
settle_ceil_divisor(const OpNode& node, bool ceil_mode, bool& include_zero_pad) {
    if (!ceil_mode || !include_zero_pad)
        return true;
    auto it = node.attrs.find("padding");
    if (it != node.attrs.end()) {
        if (const auto* v = std::get_if<std::vector<std::int64_t>>(&it->second)) {
            for (std::int64_t p : *v)
                if (p != 0)
                    return false;
        } else if (const auto* p = std::get_if<std::int64_t>(&it->second)) {
            if (*p != 0)
                return false;
        }
    }
    include_zero_pad = false;
    return true;
}

// ``t`` reshaped to the shape the trace recorded for the node's first
// output — for an op whose MPSGraph result has no static shape.  nil when
// the recording has a symbolic axis (the dynamic batch).
[[maybe_unused]] inline MPSGraphTensor*
reshape_to_recorded(MPSGraph* g, MPSGraphTensor* t, const OpNode& node) {
    if (node.outputs.empty())
        return nil;
    NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
    for (const auto d : node.outputs[0].shape) {
        if (d < 0)
            return nil;
        [shape addObject:@(d)];
    }
    return [g reshapeTensor:t withShape:shape name:nil];
}

// True when ``t``'s static shape is the shape the trace recorded for the
// node's first output.  An emitter whose op is lowered with options MPSGraph
// may size differently (ceil-mode pooling) declines rather than hand the
// next op a tensor of another shape.
[[maybe_unused]] inline bool matches_recorded_shape(MPSGraphTensor* t, const OpNode& node) {
    if (node.outputs.empty())
        return false;
    const auto& want = node.outputs[0].shape;
    NSArray<NSNumber*>* got = t.shape;
    if (got.count != want.size())
        return false;
    // A symbolic axis (the dynamic batch) is -1 in the graph and concrete
    // in the recording; it cannot disagree.
    for (NSUInteger d = 0; d < got.count; ++d)
        if (got[d].longLongValue >= 0 && got[d].longLongValue != want[d])
            return false;
    return true;
}

// Pull a ``bool`` attribute (e.g. ``keepdim``, ``align_corners``).
inline bool bool_attr(const OpNode& node, const char* key, bool def) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return def;
    const auto* p = std::get_if<bool>(&it->second);
    return p ? *p : def;
}

// Pull a ``double`` attribute (e.g. ``eps``, ``ord``, ``scale``).
[[maybe_unused]] inline double double_attr(const OpNode& node, const char* key, double def) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return def;
    const auto* p = std::get_if<double>(&it->second);
    return p ? *p : def;
}

// Pull a ``vector<int64_t>`` attribute (stride / padding / dilation /
// kernel_size / axis_list).  Returns ``nullptr`` if absent or the
// stored alternative isn't a vector — caller decides whether to
// fall back to a default or bail out.
[[maybe_unused]] inline const std::vector<std::int64_t>* int_vec_attr(const OpNode& node,
                                                                      const char* key) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return nullptr;
    return std::get_if<std::vector<std::int64_t>>(&it->second);
}

// Eager promotes the integer or bool operand of a floating op — ``exp`` of
// an int32 tensor is float32 — and the trace records that promoted output
// dtype, but the operand reaches the emitter un-promoted.  Without the cast
// the graph ran ``exp`` on integers and wrote the result into a float32
// buffer: ``exp(int32)`` was off by 403, ``mean(int64)`` by 0.83.  Only a
// floating output over a non-floating input is touched, so ``abs(int)`` and
// ``sum(int)`` keep their integer arithmetic.
[[maybe_unused]] inline MPSGraphTensor*
promote_to_float_output(MPSGraph* g, MPSGraphTensor* x, const OpNode& node) {
    if (node.outputs.empty() || x == nil)
        return x;
    MPSDataType target = x.dataType;
    switch (node.outputs[0].dtype) {
    case Dtype::F32:
        target = MPSDataTypeFloat32;
        break;
    case Dtype::F16:
        target = MPSDataTypeFloat16;
        break;
    case Dtype::BF16:
        target = MPSDataTypeBFloat16;
        break;
    default:
        return x;
    }
    return x.dataType == target ? x : [g castTensor:x toType:target name:nil];
}

// Build ``[0, 1, …, rank-1]`` as an Objective-C array of NSNumber.
// Used for "reduce over every axis" code paths.
[[maybe_unused]] inline NSArray<NSNumber*>* shape_to_axes(const Shape& s) {
    NSMutableArray<NSNumber*>* a = [NSMutableArray arrayWithCapacity:s.size()];
    for (std::size_t d = 0; d < s.size(); ++d)
        [a addObject:[NSNumber numberWithLongLong:(long long)d]];
    return a;
}

// ── Dynamic-batch shape helpers ──────────────────────────────────────
//
// Under ``dynamic_batch`` (lucid.compile(..., dynamic=True) with the symbolic
// opt-in), MpsBuilder marks the LEADING dim (dim 0) of every user-input
// placeholder -1 so one executable serves any batch size.  That -1 must keep
// flowing through the graph; the *view* ops (reshape / flatten / squeeze /
// contiguous / reduce-squeeze) copy a target shape verbatim from the trace,
// pinning the batch to its concrete trace-time value — which mismatches the
// symbolic input and aborts MPSGraph's MLIR pass.  ``reshape_dynamic_aware``
// re-introduces the symbolic dim, but ONLY when the view provably keeps the
// batch at dim 0.

// True if ``t``'s leading (batch) dim is the symbolic (-1) one.  Only dim 0 is
// ever symbolised (MpsBuilder), so this is the precise dynamic-batch predicate
// — checking dim 0 specifically, NOT "any dim is -1".
[[maybe_unused]] inline bool symbolic_batch_at_dim0(MPSGraphTensor* t) {
    return t != nil && t.shape != nil && t.shape.count > 0 && t.shape[0].longLongValue < 0;
}

// Product of all dims EXCEPT the leading one (the "non-batch" numel).
[[maybe_unused]] inline long long trailing_numel(NSArray<NSNumber*>* s) {
    long long p = 1;
    for (NSUInteger i = 1; i < s.count; ++i)
        p *= s[i].longLongValue;
    return p;
}
[[maybe_unused]] inline long long trailing_numel(const Shape& s) {
    long long p = 1;
    for (std::size_t i = 1; i < s.size(); ++i)
        p *= s[i];
    return p;
}

// Reshape ``x_t`` to the trace's concrete ``out_shape``.
//
// When ``x_t`` has a symbolic batch (leading dim -1) AND the view keeps the
// batch at dim 0, mark dim 0 of the target -1 so MPSGraph INFERS it at runtime
// instead of pinning the trace-time batch.  "Keeps the batch at dim 0" is
// proven conservatively by ``trailing_numel(in) == trailing_numel(out)``: equal
// non-leading numel forces ``out_dim0 == in_dim0 == batch`` (the view only
// restructures the non-batch part — flatten, attention head-split / merge,
// reduce-squeeze all satisfy this).  MPSGraph allows exactly one inferred (-1)
// dim, so the remaining concrete dims determine it uniquely.
//
// Returns **nil** when ``x_t`` is symbolic-batch but the view does NOT provably
// preserve the batch at dim 0 — e.g. ``unsqueeze(0)`` (batch slides to dim 1) or
// a transpose-then-reshape, and also batch-merging views like
// ``(B,T,E)->(B*T,E)`` which we conservatively reject.  Such a view cannot be
// represented under a dim-0 symbolic batch, so the caller MUST fall back
// (``return false`` → per-shape / eager) rather than emit a mis-shaped graph
// that would silently produce wrong numerics at non-trace batch sizes.
//
// Static (non-dynamic) compiles are unaffected — ``symbolic_batch_at_dim0`` is
// false, so ``out_shape`` is used verbatim, identical to the prior behaviour.
[[maybe_unused]] inline MPSGraphTensor* reshape_dynamic_aware(MPSGraph* graph,
                                                              MPSGraphTensor* x_t,
                                                              const Shape& out_shape,
                                                              NSString* name) {
    NSMutableArray<NSNumber*>* tgt = [NSMutableArray arrayWithCapacity:out_shape.size()];
    for (std::int64_t d : out_shape)
        [tgt addObject:[NSNumber numberWithLongLong:d]];
    if (symbolic_batch_at_dim0(x_t)) {
        if (tgt.count == 0 || trailing_numel(x_t.shape) != trailing_numel(out_shape))
            return nil;  // batch not provably at dim 0 → caller falls back
        tgt[0] = @(-1);
    }
    return [graph reshapeTensor:x_t withShape:tgt name:name];
}

// Return ``x`` unchanged when every padding amount in ``left`` and ``right``
// is zero; otherwise emit the MPSGraph pad op.  All-zero pad ops lower
// through MPSGraph's ANEC padding conversion and abort compilation on macOS 26.
[[maybe_unused]] inline MPSGraphTensor* pad_or_identity(MPSGraph* g,
                                                        MPSGraphTensor* x,
                                                        NSArray<NSNumber*>* left,
                                                        NSArray<NSNumber*>* right,
                                                        MPSGraphPaddingMode mode,
                                                        double value,
                                                        NSString* name = nil) {
    bool has_padding = false;
    for (NSNumber* n in left) {
        if (n.longLongValue != 0) {
            has_padding = true;
            break;
        }
    }
    if (!has_padding) {
        for (NSNumber* n in right) {
            if (n.longLongValue != 0) {
                has_padding = true;
                break;
            }
        }
    }
    if (!has_padding)
        return x;
    return [g padTensor:x
        withPaddingMode:mode
            leftPadding:left
           rightPadding:right
          constantValue:value
                   name:name];
}

}  // namespace lucid::compile
