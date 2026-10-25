// lucid/_C/compile/OpEmitters/_SortOrder.h
//
// The order ``sort``, ``argsort``, ``topk``, ``argmax`` and ``argmin`` read
// a tensor in inside a compiled graph — the compiled side of
// ``backend/gpu/SortOrder.h``, so the two name the same positions.
//
// Eager orders stably: equal values keep their index order, so ``topk``
// names ties lowest index first (LCD-407), and NaN sorts as the largest
// value.  MPSGraph's ``topK`` does neither: ``topk([3, 3, 3, 3], 2)`` named
// 3 and 1, a NaN came back as 0.0 at a repeated index, and an int64 source
// was ranked through a narrower type, so 2^40 + 3 lost to 2^40 (LCD-425).
// It also reads -inf as 0.0 at position 0 and ranks int32 through float32
// (2^30 + 3 tied with 2^30), so nothing here calls it (LCD-461).
// Its ``argSort`` is stable in both directions — descending lists equal
// values lowest index first and NaN first, ascending NaN last — and exact on
// int64 (measured on macOS 26 up to 2^20 elements per axis; the parity tests
// in ``test_sort_order_compiled.py`` hold it to that).  So every ordering op
// sorts with ``argSort`` and gathers its values by the order; ``topk`` with
// a small ``k`` selects instead (``graph_topk``).
//
// Two inputs still differ from eager before the sort, and the key removes
// them: MPSGraph puts -0.0 below +0.0 where eager calls them equal, and it
// has no bool sort.  The values a caller gathers come from the original
// tensor, so -0.0 keeps its sign.
//
// Objective-C++ only.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <limits>

#include "_DataMovement.h"

namespace lucid::compile {

// ``x`` as the key the order is taken on: zeros of either sign made +0.0 and
// bool widened to int32 (false < true, as eager orders it).  Half precision
// is widened to float32 first — exact, so the order is the same — because
// MPSGraph's compare-to-zero pass rejects a float16 ``equal`` with a warning
// on every compile ("ConvertBinaryCompareToZero expects the second operand
// to be zero").
inline MPSGraphTensor* graph_sort_key(MPSGraph* g, MPSGraphTensor* x) {
    MPSDataType dt = x.dataType;
    if (dt == MPSDataTypeBool)
        return [g castTensor:x toType:MPSDataTypeInt32 name:nil];
    if (dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16) {
        x = [g castTensor:x toType:MPSDataTypeFloat32 name:nil];
        dt = MPSDataTypeFloat32;
    }
    if (dt != MPSDataTypeFloat32)
        return x;
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
    MPSGraphTensor* is_zero = [g equalWithPrimaryTensor:x secondaryTensor:zero name:nil];
    return [g selectWithPredicateTensor:is_zero
                    truePredicateTensor:zero
                   falsePredicateTensor:x
                                   name:nil];
}

// The positions that order ``x`` along ``axis`` (int32), equal values in
// index order: ascending puts NaN last, descending puts it first.
inline MPSGraphTensor*
graph_sort_order(MPSGraph* g, MPSGraphTensor* x, NSInteger axis, bool descending) {
    return [g argSortWithTensor:graph_sort_key(g, x)
                           axis:axis
                     descending:descending ? YES : NO
                           name:nil];
}

// ``x`` read at ``order`` along ``axis`` — a sort's values, every bit of
// them: MPSGraph's own gather rounded int64 through float32, so 2^40 + 3
// came back as 2^40 (LCD-435, ``_DataMovement.h``).
inline MPSGraphTensor*
graph_gather_sorted(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* order, NSInteger axis) {
    return graph_gather_along_axis(g, axis, x, order);
}

// The first position of the largest (``is_max``) or smallest value along
// ``axis`` (kept, length one).  MPSGraph's arg reductions start from the
// type's extreme and keep the first value that passes it, so a slice that is
// all -inf (argmax), all +inf (argmin), or all the integer type's lowest /
// highest value named -1 (LCD-461).  Every value of such a slice is equal,
// and the first position, 0, is the answer.
inline MPSGraphTensor*
graph_first_extreme(MPSGraph* g, MPSGraphTensor* x, NSInteger axis, bool is_max) {
    if (x.dataType == MPSDataTypeBool)
        x = [g castTensor:x toType:MPSDataTypeInt32 name:nil];
    MPSGraphTensor* r = is_max ? [g reductionArgMaximumWithTensor:x axis:axis name:nil]
                               : [g reductionArgMinimumWithTensor:x axis:axis name:nil];
    if (r == nil)
        return nil;
    return [g maximumWithPrimaryTensor:r
                       secondaryTensor:[g constantWithScalar:0.0 dataType:r.dataType]
                                  name:nil];
}

// ``argmax`` / ``argmin`` along ``axis`` (kept, length one): the first
// extreme position, and the first NaN where the slice holds one — NaN is
// both the largest and the smallest value to eager, as to the reference.
// MPSGraph's arg reductions skip NaN, so ``argmax([2, nan, 2, nan])`` named
// 0 where eager names 1.
inline MPSGraphTensor*
graph_arg_extreme(MPSGraph* g, MPSGraphTensor* x, NSInteger axis, bool is_max) {
    MPSGraphTensor* r = graph_first_extreme(g, x, axis, is_max);
    const MPSDataType dt = x.dataType;
    if (r == nil ||
        (dt != MPSDataTypeFloat32 && dt != MPSDataTypeFloat16 && dt != MPSDataTypeBFloat16))
        return r;
    MPSGraphTensor* nan = [g castTensor:[g isNaNWithTensor:x name:nil]
                                 toType:MPSDataTypeInt32
                                   name:nil];
    MPSGraphTensor* first_nan = [g reductionArgMaximumWithTensor:nan axis:axis name:nil];
    MPSGraphTensor* any_nan =
        [g greaterThanWithPrimaryTensor:[g reductionMaximumWithTensor:nan axis:axis name:nil]
                        secondaryTensor:[g constantWithScalar:0 dataType:MPSDataTypeInt32]
                                   name:nil];
    first_nan = [g castTensor:first_nan toType:r.dataType name:nil];
    return [g selectWithPredicateTensor:any_nan
                    truePredicateTensor:first_nan
                   falsePredicateTensor:r
                                   name:nil];
}

// An int32 key that orders ``x`` as eager orders it — ±0 equal, -inf the
// lowest value, every NaN equal and above +inf — and never takes INT32_MIN,
// so a selection can retire a position by writing INT32_MIN there without
// meeting a real value.  nil for a dtype with no such key: int32 and wider
// integers fill every int32 value, and bool is not ordered by ``topk``.
//
// A float is keyed by its bits.  A non-negative float's bits already order
// as int32; a negative float's grow with its magnitude, so their low 31 bits
// are flipped.  -inf keys to INT32_MIN + 2^23 - 1, the lowest key, and only
// a NaN's bits could reach INT32_MIN.
inline MPSGraphTensor* graph_order_key(MPSGraph* g, MPSGraphTensor* x) {
    switch (x.dataType) {
    case MPSDataTypeInt8:
    case MPSDataTypeInt16:
        return [g castTensor:x toType:MPSDataTypeInt32 name:nil];
    case MPSDataTypeFloat32:
    case MPSDataTypeFloat16:
    case MPSDataTypeBFloat16:
        break;
    default:
        return nil;
    }
    MPSGraphTensor* xf = graph_sort_key(g, x);  // float32, -0.0 made +0.0
    MPSGraphTensor* bits = [g reinterpretCastTensor:xf toType:MPSDataTypeInt32 name:nil];
    MPSGraphTensor* int32_max =
        [g constantWithScalar:static_cast<double>(std::numeric_limits<std::int32_t>::max())
                     dataType:MPSDataTypeInt32];
    MPSGraphTensor* negative = [g lessThanWithPrimaryTensor:bits
                                            secondaryTensor:[g constantWithScalar:0.0
                                                                         dataType:MPSDataTypeInt32]
                                                       name:nil];
    MPSGraphTensor* key = [g selectWithPredicateTensor:negative
                                   truePredicateTensor:[g bitwiseXORWithPrimaryTensor:bits
                                                                      secondaryTensor:int32_max
                                                                                 name:nil]
                                  falsePredicateTensor:bits
                                                  name:nil];
    return [g selectWithPredicateTensor:[g isNaNWithTensor:xf name:nil]
                    truePredicateTensor:int32_max
                   falsePredicateTensor:key
                                   name:nil];
}

// ``topk``: the positions of the ``k`` largest values along ``axis``, largest
// first and equal values lowest index first, and the values at them.
// ``k`` must not exceed the axis.
struct GraphTopK {
    MPSGraphTensor* values = nil;
    MPSGraphTensor* indices = nil;  // int32
};

inline GraphTopK graph_topk_full_sort(MPSGraph* g, MPSGraphTensor* x, NSInteger axis, NSInteger k) {
    MPSGraphTensor* order = graph_sort_order(g, x, axis, true);
    MPSGraphTensor* indices = [g sliceTensor:order dimension:axis start:0 length:k name:nil];
    if (indices == nil)
        return {};
    return {graph_gather_sorted(g, x, indices, axis), indices};
}

// ``k`` first-maximum reductions of the order ``key``, each taken position
// then set to INT32_MIN.  No key is INT32_MIN, so every reduction finds an
// untaken position (``k`` does not exceed the axis) and none is named twice
// — the -inf rows of a masked softmax included.
inline GraphTopK graph_topk_select(
    MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* key, NSInteger axis, NSInteger k) {
    MPSGraphTensor* pos = [g coordinateAlongAxis:axis
                                 withShapeTensor:[g shapeOfTensor:x name:nil]
                                            name:nil];
    if (pos.dataType != MPSDataTypeInt32)
        pos = [g castTensor:pos toType:MPSDataTypeInt32 name:nil];
    MPSGraphTensor* taken =
        [g constantWithScalar:static_cast<double>(std::numeric_limits<std::int32_t>::min())
                     dataType:MPSDataTypeInt32];
    NSMutableArray<MPSGraphTensor*>* picked = [NSMutableArray arrayWithCapacity:(NSUInteger)k];
    MPSGraphTensor* rest = key;
    for (NSInteger s = 0; s < k; ++s) {
        MPSGraphTensor* at = [g reductionArgMaximumWithTensor:rest axis:axis name:nil];
        if (at == nil)
            return graph_topk_full_sort(g, x, axis, k);
        if (at.dataType != MPSDataTypeInt32)
            at = [g castTensor:at toType:MPSDataTypeInt32 name:nil];
        [picked addObject:at];
        if (s + 1 < k)
            rest = [g selectWithPredicateTensor:[g equalWithPrimaryTensor:pos
                                                          secondaryTensor:at
                                                                     name:nil]
                            truePredicateTensor:taken
                           falsePredicateTensor:rest
                                           name:nil];
    }
    MPSGraphTensor* indices = k == 1 ? picked[0] : [g concatTensors:picked dimension:axis name:nil];
    return {graph_gather_sorted(g, x, indices, axis), indices};
}

// The largest ``k`` that ``topk`` selects rather than sorts.  Selection makes
// ``k`` passes over the tensor; past eight the sort is as fast or faster on a
// long axis (M1 Pro, LCD-461: (64, 1000) k = 5 selects in 0.6 ms against
// 1.6 ms sorted; (8, 32000) k = 16 takes 2.7 ms either way, and (1, 2^20)
// k = 32 sorts faster).
constexpr NSInteger kTopKSelectMax = 8;

inline GraphTopK graph_topk(MPSGraph* g, MPSGraphTensor* x, NSInteger axis, NSInteger k) {
    if (g == nil || x == nil || x.shape == nil)
        return {};
    const NSInteger rank = (NSInteger)x.shape.count;
    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= rank || k <= 0)
        return {};
    bool selects = k <= kTopKSelectMax && k <= [x.shape[(NSUInteger)axis] integerValue];
    for (NSNumber* dim in x.shape)
        selects = selects && dim.integerValue != 0;
    MPSGraphTensor* key = selects ? graph_order_key(g, x) : nil;
    if (key != nil)
        return graph_topk_select(g, x, key, axis, k);
    return graph_topk_full_sort(g, x, axis, k);
}

}  // namespace lucid::compile
