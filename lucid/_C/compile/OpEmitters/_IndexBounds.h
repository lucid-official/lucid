// lucid/_C/compile/OpEmitters/_IndexBounds.h
//
// Where a caller's index values meet an MPSGraph gather or scatter inside a
// compiled graph, and the one place that keeps them inside the buffer.
//
// The compiled graph is a second implementation of the eager index ops, so it
// follows the eager Metal contract, policy B (LCD-228,
// ``backend/gpu/AxisIndex.h``):
//
//   * a gather answers NaN at an out-of-range position, or 0 when the result
//     is an integer or bool;
//   * a scatter drops an out-of-range update, and the base keeps its value;
//   * no index outside the axis reaches an MPSGraph gather or scatter.
//
// The graph is built from the first call's indices, and every later call
// feeds new ones that nothing on the host reads, so the check has to be in
// the graph.  Before LCD-278 the emitters handed the indices straight to
// MPSGraph, which does not compare them with the axis either.
//
// Objective-C++ only.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "_DataMovement.h"

namespace lucid::compile {

// What an index below zero means.
enum class NegativeIndex {
    Wrap,        // ``-k`` names ``extent - k``, once: gather, scatter
    OutOfRange,  // nothing lies below zero: a table row, a class
};

// An index that is safe to hand to an MPSGraph gather or scatter.  Both nil
// when the index cannot be checked (not an integer, or the axis length is not
// known when the graph is built), and the emitter declines to eager then.
struct GraphAxisIndex {
    // In [0, extent) everywhere, and the index itself where it is in range.
    MPSGraphTensor* safe = nil;
    // Bool, the index's shape: whether the index named a real position.
    MPSGraphTensor* in_range = nil;
};

// ``idx`` checked against an axis of ``extent`` positions, at a width that
// holds every index value and ``extent``: int64 stays int64 (narrowing first
// is how 2^32 + 1 read row 1), anything narrower is widened to int32.  The
// safe index comes back as int32 whenever ``extent`` fits, as eager's does.
inline GraphAxisIndex
graph_axis_index(MPSGraph* g, MPSGraphTensor* idx, std::int64_t extent, NegativeIndex negative) {
    const MPSDataType dt = idx.dataType;
    const bool integer = dt == MPSDataTypeInt8 || dt == MPSDataTypeInt16 ||
                         dt == MPSDataTypeInt32 || dt == MPSDataTypeInt64 || dt == MPSDataTypeBool;
    if (!integer || extent <= 0)
        return {};
    const bool fits32 = extent <= std::numeric_limits<std::int32_t>::max();
    const MPSDataType work =
        (dt == MPSDataTypeInt64 || !fits32) ? MPSDataTypeInt64 : MPSDataTypeInt32;
    MPSGraphTensor* i = dt == work ? idx : [g castTensor:idx toType:work name:nil];
    MPSGraphTensor* n = [g constantWithScalar:static_cast<double>(extent) dataType:work];
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:work];
    MPSGraphTensor* negative_side = [g lessThanWithPrimaryTensor:i secondaryTensor:zero name:nil];
    MPSGraphTensor* lowest =
        negative == NegativeIndex::Wrap ? [g negativeWithTensor:n name:nil] : zero;
    MPSGraphTensor* in_range =
        [g logicalANDWithPrimaryTensor:[g greaterThanOrEqualToWithPrimaryTensor:i
                                                                secondaryTensor:lowest
                                                                           name:nil]
                       secondaryTensor:[g lessThanWithPrimaryTensor:i secondaryTensor:n name:nil]
                                  name:@"index_in_range"];
    MPSGraphTensor* wrapped = negative == NegativeIndex::Wrap
                                  ? [g selectWithPredicateTensor:negative_side
                                             truePredicateTensor:[g additionWithPrimaryTensor:i
                                                                              secondaryTensor:n
                                                                                         name:nil]
                                            falsePredicateTensor:i
                                                            name:nil]
                                  : i;
    MPSGraphTensor* safe = [g selectWithPredicateTensor:in_range
                                    truePredicateTensor:wrapped
                                   falsePredicateTensor:zero
                                                   name:@"index_safe"];
    if (fits32 && work != MPSDataTypeInt32)
        safe = [g castTensor:safe toType:MPSDataTypeInt32 name:nil];
    return {safe, in_range};
}

// ``gathered`` with every out-of-range position replaced by what policy B
// answers there: NaN for a floating result, 0 for an integer or bool one.
// ``in_range`` broadcasts against ``gathered``.  nil for a complex result,
// whose NaN this layer does not build; the emitter declines then.
inline MPSGraphTensor*
graph_fill_out_of_range(MPSGraph* g, MPSGraphTensor* gathered, MPSGraphTensor* in_range) {
    const MPSDataType dt = gathered.dataType;
    if (dt & MPSDataTypeComplexBit)
        return nil;
    const double fill = (dt & MPSDataTypeFloatBit) ? std::nan("") : 0.0;
    return [g selectWithPredicateTensor:in_range
                    truePredicateTensor:gathered
                   falsePredicateTensor:[g constantWithScalar:fill dataType:dt]
                                   name:@"index_out_of_range_fill"];
}

// The reductions an MPSGraph scatter applies.
enum class GraphScatter { Add, Max, Min, Mul };

// The update that leaves a held value as it was, which is what an
// out-of-range update becomes before it is scattered at the safe index.
// For a sum that is -0.0, not +0.0: ``x + -0.0`` is ``x`` for every ``x``,
// -0.0 included.  nil when the dtype has no such value here (complex).
inline MPSGraphTensor* graph_scatter_identity(MPSGraph* g, GraphScatter op, MPSDataType dt) {
    if (dt & MPSDataTypeComplexBit)
        return nil;
    const bool floating = (dt & MPSDataTypeFloatBit) != 0;
    double v = 0.0;
    switch (op) {
    case GraphScatter::Add:
        v = floating ? -0.0 : 0.0;
        break;
    case GraphScatter::Mul:
        v = 1.0;
        break;
    case GraphScatter::Max:
    case GraphScatter::Min: {
        const bool lowest = op == GraphScatter::Max;
        if (floating) {
            v = lowest ? -INFINITY : INFINITY;
        } else if (dt == MPSDataTypeInt32) {
            v = lowest ? std::numeric_limits<std::int32_t>::lowest()
                       : std::numeric_limits<std::int32_t>::max();
        } else if (dt == MPSDataTypeInt16) {
            v = lowest ? std::numeric_limits<std::int16_t>::lowest()
                       : std::numeric_limits<std::int16_t>::max();
        } else if (dt == MPSDataTypeInt8) {
            v = lowest ? std::numeric_limits<std::int8_t>::lowest()
                       : std::numeric_limits<std::int8_t>::max();
        } else {
            return nil;  // int64's bounds do not survive the double constant
        }
        break;
    }
    }
    return [g constantWithScalar:v dataType:dt];
}

// ``updates`` with each out-of-range entry replaced by ``op``'s identity, so
// a reducing scatter at the safe index leaves that position as it was.  nil
// when ``op`` has no identity for the dtype.
inline MPSGraphTensor* graph_drop_out_of_range(MPSGraph* g,
                                               MPSGraphTensor* updates,
                                               MPSGraphTensor* in_range,
                                               GraphScatter op) {
    MPSGraphTensor* identity = graph_scatter_identity(g, op, updates.dataType);
    if (identity == nil)
        return nil;
    return [g selectWithPredicateTensor:in_range
                    truePredicateTensor:updates
                   falsePredicateTensor:identity
                                   name:@"index_out_of_range_drop"];
}

// An overwrite scatter under policy B.  No in-range position can take a
// write and keep its value, so an out-of-range write is aimed at a one-wide
// sink slab appended past the end of ``axis`` and cut off again — eager's
// ``gpu_sink_index`` / ``with_sink_slab``.  nil when MPSGraph cannot write
// the dtype exactly.
inline MPSGraphTensor* graph_scatter_set_dropping(MPSGraph* g,
                                                  MPSGraphTensor* base,
                                                  const GraphAxisIndex& ix,
                                                  MPSGraphTensor* src,
                                                  NSInteger axis,
                                                  std::int64_t extent) {
    if (base.dataType != src.dataType)
        return nil;
    // Fast path for single-slice overwrite along ``axis`` (e.g. single-token
    // KV-cache ``index_copy`` or ``x[:, :, idx] = v``): when the update has
    // extent 1 along ``axis``, there are no duplicate-index collisions, and
    // ``src`` + ``ix`` broadcast directly against ``base`` along ``axis``.
    // Emitting a coordinate comparison + elementwise ``select`` avoids
    // allocating a ``(max_len + 1)`` padded concat buffer, running a scatter
    // kernel, and slicing off the sink slab on every layer.
    NSArray<NSNumber*>* src_shape = src.shape;
    if (extent > 0 && src_shape != nil && axis >= 0 &&
        static_cast<NSUInteger>(axis) < src_shape.count &&
        src_shape[static_cast<NSUInteger>(axis)].longLongValue == 1 &&
        !(base.dataType & MPSDataTypeComplexBit)) {
        MPSGraphTensor* coord = [g coordinateAlongAxis:axis
                                       withShapeTensor:[g shapeOfTensor:base name:nil]
                                                  name:nil];
        if (coord.dataType != ix.safe.dataType)
            coord = [g castTensor:coord toType:ix.safe.dataType name:nil];
        MPSGraphTensor* match = [g logicalANDWithPrimaryTensor:ix.in_range
                                               secondaryTensor:[g equalWithPrimaryTensor:coord
                                                                         secondaryTensor:ix.safe
                                                                                    name:nil]
                                                          name:nil];
        return [g selectWithPredicateTensor:match
                        truePredicateTensor:src
                       falsePredicateTensor:base
                                       name:@"scatter_set_single"];
    }
    // The slab is cut off again, so what it holds does not matter: a slice
    // of ``base`` has its shape even where the batch is symbolic, which a
    // constant does not ("shape[0] (-1) should be a strictly positive value
    // for constant operation" aborted a dynamic-batch scatter).
    MPSGraphTensor* padded = [g concatTensor:base
                                  withTensor:[g sliceTensor:base
                                                  dimension:axis
                                                      start:0
                                                     length:1
                                                       name:nil]
                                   dimension:axis
                                        name:nil];
    MPSGraphTensor* past_end = [g constantWithScalar:static_cast<double>(extent)
                                            dataType:ix.safe.dataType];
    MPSGraphTensor* sink = [g selectWithPredicateTensor:ix.in_range
                                    truePredicateTensor:ix.safe
                                   falsePredicateTensor:past_end
                                                   name:@"scatter_sink_index"];
    MPSGraphTensor* written = graph_scatter_along_axis(g, axis, padded, src, sink,
                                                       MPSGraphScatterModeSet, @"scatter_axis");
    if (written == nil)
        return nil;
    return [g sliceTensor:written dimension:axis start:0 length:extent name:@"scatter_cut_sink"];
}

// A class-index target (``nll_loss``, ``cross_entropy_loss``) checked against
// ``classes`` classes, one per sample.  Eager raises IndexError for a target
// that is neither a class nor ``ignore_index``; a compiled graph cannot raise
// on a value fed after it was built, so that sample's loss is NaN (policy B)
// and nothing outside the class axis is read.
struct GraphClassTarget {
    MPSGraphTensor* safe = nil;     // in [0, classes) everywhere: what to gather at
    MPSGraphTensor* counts = nil;   // bool: names a class and is not ignored
    MPSGraphTensor* ignored = nil;  // bool: equals ``ignore_index`` (contributes 0)
};

inline GraphClassTarget graph_class_target(MPSGraph* g,
                                           MPSGraphTensor* target,
                                           std::int64_t classes,
                                           std::int64_t ignore_index) {
    const GraphAxisIndex ix = graph_axis_index(g, target, classes, NegativeIndex::OutOfRange);
    if (ix.safe == nil)
        return {};
    MPSGraphTensor* ignored =
        [g equalWithPrimaryTensor:target
                  secondaryTensor:[g constantWithScalar:static_cast<double>(ignore_index)
                                               dataType:target.dataType]
                             name:@"target_ignored"];
    MPSGraphTensor* counts = [g logicalANDWithPrimaryTensor:ix.in_range
                                            secondaryTensor:[g notWithTensor:ignored name:nil]
                                                       name:@"target_counts"];
    return {ix.safe, counts, ignored};
}

// One pixel of an ``H x W`` image named by a whole-numbered floating
// coordinate pair (``grid_sample``'s corners): its index into the flattened
// image, and whether it lies inside.
struct GraphPixel {
    MPSGraphTensor* flat = nil;    // int32, in [0, H * W) everywhere
    MPSGraphTensor* inside = nil;  // bool; nil when ``zero_padding`` is false
};

// The pixel at ``(px, py)``, clamped onto the image for the read.  A NaN
// coordinate is taken as 0 first: casting NaN to an integer is undefined and
// comparing it is false everywhere, so it otherwise read an unknown pixel.
// Eager casts after rounding, and Apple silicon's float-to-int conversion
// turns NaN into 0 on both devices, so eager reads position 0 there and
// counts it inside — a NaN nearest sample is that pixel, a NaN bilinear
// sample NaN through its weights.  ``inside`` is computed only for zero
// padding, the one mode whose outside reads are replaced.
inline GraphPixel graph_pixel(MPSGraph* g,
                              MPSGraphTensor* px,
                              MPSGraphTensor* py,
                              long long H,
                              long long W,
                              bool zero_padding) {
    const MPSDataType dt = px.dataType;
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
    const auto without_nan = [&](MPSGraphTensor* p) {
        return [g selectWithPredicateTensor:[g isNaNWithTensor:p name:nil]
                        truePredicateTensor:zero
                       falsePredicateTensor:p
                                       name:@"coordinate_nan_to_zero"];
    };
    MPSGraphTensor* x = without_nan(px);
    MPSGraphTensor* y = without_nan(py);
    MPSGraphTensor* x_hi = [g constantWithScalar:static_cast<double>(W - 1) dataType:dt];
    MPSGraphTensor* y_hi = [g constantWithScalar:static_cast<double>(H - 1) dataType:dt];
    GraphPixel out;
    if (zero_padding) {
        const auto within = [&](MPSGraphTensor* p, MPSGraphTensor* hi) {
            return [g logicalANDWithPrimaryTensor:[g greaterThanOrEqualToWithPrimaryTensor:p
                                                                           secondaryTensor:zero
                                                                                      name:nil]
                                  secondaryTensor:[g lessThanOrEqualToWithPrimaryTensor:p
                                                                        secondaryTensor:hi
                                                                                   name:nil]
                                             name:nil];
        };
        out.inside = [g logicalANDWithPrimaryTensor:within(x, x_hi)
                                    secondaryTensor:within(y, y_hi)
                                               name:@"pixel_inside"];
    }
    // Whole numbers already (floor / rint), so the cast discards nothing.
    const auto index = [&](MPSGraphTensor* p, MPSGraphTensor* hi) {
        return [g castTensor:[g clampWithTensor:p minValueTensor:zero maxValueTensor:hi name:nil]
                      toType:MPSDataTypeInt32
                        name:nil];
    };
    MPSGraphTensor* row =
        [g multiplicationWithPrimaryTensor:index(y, y_hi)
                           secondaryTensor:[g constantWithScalar:static_cast<double>(W)
                                                        dataType:MPSDataTypeInt32]
                                      name:nil];
    out.flat = [g additionWithPrimaryTensor:row secondaryTensor:index(x, x_hi) name:@"pixel_flat"];
    return out;
}

// ``p`` clamped into [lo, hi] with NaN left NaN, as eager's clamp leaves it
// (a maximum then a minimum, each of which propagates NaN).  MPSGraph's
// clamp answers a bound for NaN, which turned a NaN border-padded bilinear
// sample into a finite one.
inline MPSGraphTensor*
graph_clamp_keeping_nan(MPSGraph* g, MPSGraphTensor* p, MPSGraphTensor* lo, MPSGraphTensor* hi) {
    return [g selectWithPredicateTensor:[g isNaNWithTensor:p name:nil]
                    truePredicateTensor:p
                   falsePredicateTensor:[g clampWithTensor:p
                                            minValueTensor:lo
                                            maxValueTensor:hi
                                                      name:nil]
                                   name:@"clamp_keeping_nan"];
}

// MPSGraph's along-axis gather and scatter want the index to match the data
// on every axis but ``skip_axis`` (pass -1 for none).  The reference — and
// Lucid's eager kernels — allow an index shorter there, and MPSGraph handed
// one aborted the process: "updates shape and indices shape must match
// except at axis".  Such a call is declined and runs eagerly.
inline bool shapes_agree_off_axis(MPSGraphTensor* a, MPSGraphTensor* b, NSInteger skip_axis) {
    NSArray<NSNumber*>* sa = a.shape;
    NSArray<NSNumber*>* sb = b.shape;
    if (sa == nil || sb == nil || sa.count != sb.count)
        return false;
    for (NSUInteger i = 0; i < sa.count; ++i) {
        if (static_cast<NSInteger>(i) == skip_axis)
            continue;
        if (sa[i].longLongValue != sb[i].longLongValue)
            return false;
    }
    return true;
}

// The static length of ``t`` along ``axis``, or -1 when it is not known when
// the graph is built (the symbolic batch).
inline std::int64_t static_extent(MPSGraphTensor* t, NSInteger axis) {
    NSArray<NSNumber*>* s = t.shape;
    if (s == nil || axis < 0 || static_cast<NSUInteger>(axis) >= s.count)
        return -1;
    return s[static_cast<NSUInteger>(axis)].longLongValue;
}

}  // namespace lucid::compile
