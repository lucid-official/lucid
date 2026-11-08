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
// orders with ``graph_sort_order`` and gathers its values by the order;
// ``topk`` with a small ``k`` selects instead (``graph_topk``).
//
// ``argSort`` is slow, though, in the way that matters: it costs per row
// (about 4 us a row at any length), so a compiled sort of (32768, 16) ran 140
// ms against 1.3 ms eager (LCD-926).  ``graph_sort_order`` therefore uses it
// only for few rows, ranks a short axis by comparing every row with itself
// (``graph_rank_order``), and declines the rest so the graph runs eager.
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

#include "../MpsDtype.h"
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

// The positions that order ``x`` along ``axis`` (int32) through MPSGraph's
// ``argSort``, equal values in index order: ascending puts NaN last,
// descending puts it first.  Its cost is per row — about 4 us a row at any
// row length — so ``graph_sort_order`` below picks it only for few rows.
inline MPSGraphTensor*
graph_argsort_order(MPSGraph* g, MPSGraphTensor* x, NSInteger axis, bool descending) {
    return [g argSortWithTensor:graph_sort_key(g, x)
                           axis:axis
                     descending:descending ? YES : NO
                           name:nil];
}

// MPSGraph runs a tensor of rank five or more as a loop over the dims above the
// last four, one dispatch per slice (LCD-918), so a static rank-5+ ``x`` is
// read as ``[before, n, after]`` — the dims on each side of the axis fused.
struct SortFold {
    NSInteger before = 1;
    NSInteger after = 1;
};

// ``shape`` as ``before * n * after`` around ``axis``; false for a dynamic dim.
inline bool sort_fold(NSArray<NSNumber*>* shape, NSInteger axis, SortFold& fold) {
    fold = SortFold{};
    for (NSUInteger d = 0; d < shape.count; ++d) {
        const NSInteger size = shape[d].integerValue;
        if (size < 0)
            return false;
        if ((NSInteger)d < axis)
            fold.before *= size;
        else if ((NSInteger)d > axis)
            fold.after *= size;
    }
    return true;
}

// ``x`` read at ``order`` along ``axis`` — a sort's values, every bit of
// them: MPSGraph's own gather rounded int64 through float32, so 2^40 + 3
// came back as 2^40 (LCD-435, ``_DataMovement.h``).
// A static rank-5+ ``x`` is gathered folded (``SortFold``).
inline MPSGraphTensor*
graph_gather_sorted(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* order, NSInteger axis) {
    const NSInteger rank = (NSInteger)x.shape.count;
    if (axis < 0)
        axis += rank;
    SortFold fold, order_fold;
    if (rank < 5 || axis < 0 || axis >= rank || order.shape.count != x.shape.count ||
        !sort_fold(x.shape, axis, fold) || !sort_fold(order.shape, axis, order_fold) ||
        fold.before != order_fold.before || fold.after != order_fold.after)
        return graph_gather_along_axis(g, axis, x, order);
    NSArray<NSNumber*>* x3 = @[ @(fold.before), x.shape[(NSUInteger)axis], @(fold.after) ];
    NSArray<NSNumber*>* o3 = @[ @(fold.before), order.shape[(NSUInteger)axis], @(fold.after) ];
    MPSGraphTensor* out = graph_gather_along_axis(g, 1, [g reshapeTensor:x withShape:x3 name:nil],
                                                  [g reshapeTensor:order withShape:o3 name:nil]);
    return out == nil ? nil : [g reshapeTensor:out withShape:order.shape name:nil];
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
    MPSGraphTensor* bits = reinterpret_cast_mps_tensor(g, xf, MPSDataTypeInt32);
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

// An int32 (or int64) key that ``graph_rank_order`` compares: the order
// ``graph_order_key`` gives a float, an integer as itself, and bool / the
// narrow unsigned types widened.  nil for a dtype it cannot compare exactly
// (uint32, uint64).
inline MPSGraphTensor* graph_rank_key(MPSGraph* g, MPSGraphTensor* x) {
    switch (x.dataType) {
    case MPSDataTypeBool:
        return graph_sort_key(g, x);
    case MPSDataTypeUInt8:
    case MPSDataTypeUInt16:
        return [g castTensor:x toType:MPSDataTypeInt32 name:nil];
    case MPSDataTypeInt32:
    case MPSDataTypeInt64:
        return x;
    default:
        return graph_order_key(g, x);
    }
}

// The positions that order ``x`` along ``axis`` (int32), counted instead of
// sorted: the position of element ``i`` is how many elements come before it
// (compare less, or equal at a lower index; greater for descending), and the
// order is those positions inverted by one scatter.  Every row is compared
// with itself at once, so the cost is ``rows * n * n`` element compares and
// does not depend on how many rows there are — the shape ``argSort`` is slow
// on.  ``key`` is ``graph_rank_key(x)``; the shape of ``x`` is static.
inline MPSGraphTensor* graph_rank_order(
    MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* key, NSInteger axis, bool descending) {
    // Folded to ``[before, n, after]`` so the compared grid ``[before, n, n,
    // after]`` is rank four whatever the rank of ``x`` (see ``SortFold``).
    SortFold fold;
    if (!sort_fold(x.shape, axis, fold))
        return nil;
    NSNumber* n = x.shape[(NSUInteger)axis];
    NSArray<NSNumber*>* folded = @[ @(fold.before), n, @(fold.after) ];
    key = [g reshapeTensor:key withShape:folded name:nil];

    // ``mine`` is element i on the first of the two compared axes, ``other``
    // is element j on the second.
    NSArray<NSNumber*>* pair = @[ @1, n, n, @1 ];
    MPSGraphTensor* mine = [g expandDimsOfTensor:key axes:@[ @2 ] name:nil];
    MPSGraphTensor* other = [g expandDimsOfTensor:key axes:@[ @1 ] name:nil];
    MPSGraphTensor* i_pos = [g coordinateAlongAxis:1 withShape:pair name:nil];
    MPSGraphTensor* j_pos = [g coordinateAlongAxis:2 withShape:pair name:nil];
    // ``other`` comes before ``mine`` when it is lower (higher, descending), or
    // equal at a lower index: ``<=`` where j < i, ``<`` elsewhere.
    MPSGraphTensor* earlier = [g lessThanWithPrimaryTensor:j_pos secondaryTensor:i_pos name:nil];
    MPSGraphTensor* strictly =
        descending ? [g greaterThanWithPrimaryTensor:other secondaryTensor:mine name:nil]
                   : [g lessThanWithPrimaryTensor:other secondaryTensor:mine name:nil];
    MPSGraphTensor* loosely =
        descending ? [g greaterThanOrEqualToWithPrimaryTensor:other secondaryTensor:mine name:nil]
                   : [g lessThanOrEqualToWithPrimaryTensor:other secondaryTensor:mine name:nil];
    MPSGraphTensor* before = [g selectWithPredicateTensor:earlier
                                      truePredicateTensor:loosely
                                     falsePredicateTensor:strictly
                                                     name:nil];
    MPSGraphTensor* position = [g reductionSumWithTensor:[g castTensor:before
                                                                toType:MPSDataTypeInt32
                                                                  name:nil]
                                                    axis:2
                                                    name:nil];
    position = [g squeezeTensor:position axis:2 name:nil];

    MPSGraphTensor* index = [g coordinateAlongAxis:1 withShape:folded name:nil];
    MPSGraphTensor* zeros = [g broadcastTensor:[g constantWithScalar:0.0 dataType:MPSDataTypeInt32]
                                       toShape:folded
                                          name:nil];
    MPSGraphTensor* order =
        graph_scatter_along_axis(g, 1, zeros, index, position, MPSGraphScatterModeSet);
    return order == nil ? nil : [g reshapeTensor:order withShape:x.shape name:nil];
}

// Rows up to which ``argSort`` is taken as it is: it costs about 4 us a row
// on top of a 0.6 ms floor (M1 Pro, LCD-926), so a few rows are as fast as
// anything.
constexpr std::int64_t kArgSortFewRows = 16;
// Rows up to which ``argSort`` is still worth taking where ranking does not
// fit: 128 rows cost 1.5 to 2.5 ms against 0.3 to 1.4 ms eager.
constexpr std::int64_t kArgSortMaxRows = 128;
// The most element compares ``graph_rank_order`` may make (2^25: a bool and
// an int32 temporary of 128 MB at most).  At 2^25, (8192, 64) ranks in
// 1.6 ms against 0.9 ms eager.
constexpr std::int64_t kRankMaxPairs = std::int64_t{1} << 25;

// The positions that order ``x`` along ``axis`` (int32), equal values in
// index order: ascending puts NaN last, descending puts it first.
//
// MPSGraph's ``argSort`` costs per row, not per element: (32768, 16) took
// 140 ms against 1.3 ms eager, (131072, 4) 556 ms against 4.2 ms, and a
// compiled sort at 2^25 elements did not finish in two minutes (LCD-926).  So
//   * few rows               -> ``argSort``;
//   * rows * n * n compares fit -> ``graph_rank_order``, which a short axis
//                               on many rows needs;
//   * up to 128 rows         -> ``argSort``;
//   * otherwise              -> nil, and the emitter declines: the graph runs
//     eager, where MLX's block sort is faster than anything MPSGraph builds.
// ``always`` takes ``argSort`` over nil, for a caller with no eager to fall
// back on.  A dynamic dimension is unknown at build time and takes ``argSort``.
inline MPSGraphTensor* graph_sort_order(
    MPSGraph* g, MPSGraphTensor* x, NSInteger axis, bool descending, bool always = false) {
    NSArray<NSNumber*>* shape = x.shape;
    const NSInteger rank = shape == nil ? 0 : (NSInteger)shape.count;
    if (axis < 0)
        axis += rank;
    if (rank == 0 || axis < 0 || axis >= rank)
        return graph_argsort_order(g, x, axis, descending);
    std::int64_t total = 1;
    for (NSNumber* dim in shape) {
        if (dim.integerValue < 0)
            return graph_argsort_order(g, x, axis, descending);
        total *= dim.integerValue;
    }
    const std::int64_t n = shape[(NSUInteger)axis].integerValue;
    if (total == 0)
        return graph_argsort_order(g, x, axis, descending);
    const std::int64_t rows = total / n;
    if (rows <= kArgSortFewRows)
        return graph_argsort_order(g, x, axis, descending);
    if (n <= kRankMaxPairs / total) {  // rows * n * n compares
        MPSGraphTensor* key = graph_rank_key(g, x);
        MPSGraphTensor* ranked = key == nil ? nil : graph_rank_order(g, x, key, axis, descending);
        if (ranked != nil)
            return ranked;
    }
    if (rows <= kArgSortMaxRows || always)
        return graph_argsort_order(g, x, axis, descending);
    return nil;
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
    if (order == nil)
        return {};
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
