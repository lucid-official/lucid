// lucid/_C/compile/OpEmitters/_DataMovement.h
//
// The one place a compiled forward graph hands values to an MPSGraph gather
// or scatter, so that every bit of every value arrives (LCD-435).
//
// MPSGraph's gather family (``gatherAlongAxis``, ``gather``, ``gatherND``)
// carries 32- and 64-bit integers through float32: 2^40 + 3 came back as
// 2^40, 2^62 + 5 as 2^62 and even int32 2^24 + 1 as 2^24, whatever the index
// type.  Its scatter family (``scatterAlongAxis``, ``scatter``, ``scatterND``)
// keeps int32 but cuts int64 to its low 32 bits (2^62 + 5 wrote 5), rounds
// uint32 / uint64 through float32 and writes bool as false.  Moving the same
// bytes as int16 or int8 is exact in every mode, as are float32, bfloat16 and
// every reshape, concat, slice, select, transpose, broadcast, pad and tile
// (measured on macOS 27, M1 Pro; ``test_wide_integer_data_movement_compiled.py``
// holds the compiled ops to eager bit for bit).
// float16 gathers keep every value but answer one canonical NaN for every
// NaN payload, which no Lucid op observes.
//
// So a value MPSGraph would round is carried instead: a wide integer as its
// int16 lanes on a new trailing axis (little endian, so lane 0 is the low
// half-word), bool as int8 0 / 1.  An along-axis index gains the lane axis
// by broadcasting, an axis gather needs none.  Every other dtype goes to
// MPSGraph untouched, so float graphs are the graphs they were.
//
// The lane axis is one more axis, and MPSGraph runs a rank >= 5 gather as a
// loop over the dims above the last four (~4.5 us a slice): int32
// ``(8192, 8, 16, 16)`` gathered in 53 ms against 12 at rank 3.  So the
// operand and the index are read as ``(pre, d, post)`` around the gathered
// axis before the lanes are added, and the answer is read back
// (``_AxisSplitLayout.h``, LCD-997; the policy is the rank pass's gate).
//
// An overwrite cannot run on lanes either: each lane is its own scatter
// element, so where two updates aim at one slot each lane picked its own
// winner and the slot held half of one update and half of another.  It
// scatters the update's int32 position instead (one element, one winner),
// gathers the winning update whole and keeps ``data`` where nothing landed.
//
// A scatter that sums cannot run on lanes.  An int64 sum splits each update
// into its high word and two low half-words, sums each in int32 and adds
// them back with int64 arithmetic, which is exact: the high word is needed
// only modulo 2^32, which int32 keeps however often it wraps, and a slot
// that takes at most 2^15 updates sums each half-word below 2^31.  A sum over
// a longer update axis, and every other reduction of a dtype MPSGraph does
// not move exactly, answers nil and the emitter declines to eager.
//
// Objective-C++ only.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <optional>

#include "_AttrHelpers.h"
#include "_AxisSplitLayout.h"

namespace lucid::compile {

enum class GraphMove { Gather, Scatter };

// Whether MPSGraph's ``op`` family moves every value of ``dt`` unchanged.
inline bool graph_moves_exactly(MPSDataType dt, GraphMove op) {
    switch (dt) {
    case MPSDataTypeInt64:
    case MPSDataTypeUInt64:
    case MPSDataTypeUInt32:
        return false;
    case MPSDataTypeInt32:
        return op == GraphMove::Scatter;
    case MPSDataTypeBool:
        return op == GraphMove::Gather;
    default:
        return true;
    }
}

namespace data_movement_detail {

// int16 lanes per element of a wide integer, 0 for a carrier without a lane
// axis (bool).
inline NSUInteger lanes_of(MPSDataType dt) {
    if (dt == MPSDataTypeBool)
        return 0;
    return (dt & 0xFFFF) / 16;
}

// ``x`` in the type that carries it exactly.  nil when ``x`` has no rank.
inline MPSGraphTensor* carry(MPSGraph* g, MPSGraphTensor* x) {
    if (x.dataType == MPSDataTypeBool)
        return [g castTensor:x toType:MPSDataTypeInt8 name:@"bool_as_int8"];
    if (x.shape == nil)
        return nil;
    MPSGraphTensor* one = [g expandDimsOfTensor:x axis:(NSInteger)x.shape.count name:nil];
    return [g reinterpretCastTensor:one toType:MPSDataTypeInt16 name:@"int16_lanes"];
}

// The carried ``c`` read back as ``dt``.
inline MPSGraphTensor* uncarry(MPSGraph* g, MPSGraphTensor* c, MPSDataType dt) {
    if (c == nil)
        return nil;
    if (dt == MPSDataTypeBool)
        return [g castTensor:c toType:MPSDataTypeBool name:nil];
    MPSGraphTensor* whole = [g reinterpretCastTensor:c toType:dt name:nil];
    return [g squeezeTensor:whole axis:(NSInteger)whole.shape.count - 1 name:@"from_int16_lanes"];
}

// An along-axis index for data carried with ``lanes`` lanes: the same
// position for every lane of an element.
inline MPSGraphTensor* lane_index(MPSGraph* g, MPSGraphTensor* idx, NSUInteger lanes) {
    if (lanes == 0)
        return idx;
    if (idx.shape == nil)
        return nil;
    const NSUInteger rank = idx.shape.count;
    NSMutableArray<NSNumber*>* spread = [NSMutableArray arrayWithCapacity:rank + 1];
    for (NSUInteger d = 0; d < rank; ++d)
        [spread addObject:@1];
    [spread addObject:@(lanes)];
    return [g additionWithPrimaryTensor:[g expandDimsOfTensor:idx axis:(NSInteger)rank name:nil]
                        secondaryTensor:[g constantWithScalar:0.0
                                                        shape:spread
                                                     dataType:idx.dataType]
                                   name:nil];
}

// ``x`` read as ``shape`` (a reshape: an alias of a computed tensor).  nil in, nil out.
inline MPSGraphTensor*
read_as(MPSGraph* g, MPSGraphTensor* x, const std::vector<std::int64_t>& shape) {
    return x == nil ? nil : [g reshapeTensor:x withShape:shape_to_ns(shape) name:nil];
}

// An int32 tensor of ``like``'s shape holding ``value``; the shape is read
// from the graph, so a symbolic batch is fine.
inline MPSGraphTensor* int32_filled(MPSGraph* g, MPSGraphTensor* like, double value) {
    return [g broadcastTensor:[g constantWithScalar:value dataType:MPSDataTypeInt32]
                toShapeTensor:[g shapeOfTensor:like name:nil]
                         name:nil];
}

// The most updates one slot of an int32 half-word sum may take.
constexpr std::int64_t kMaxHalfWordUpdates = std::int64_t{1} << 15;

// ``data`` plus the int64 ``updates`` summed at ``idx`` along ``axis``.
inline MPSGraphTensor* scatter_add_int64(MPSGraph* g,
                                         NSInteger axis,
                                         MPSGraphTensor* data,
                                         MPSGraphTensor* updates,
                                         MPSGraphTensor* idx) {
    NSArray<NSNumber*>* us = updates.shape;
    if (us == nil || axis < 0 || static_cast<NSUInteger>(axis) >= us.count)
        return nil;
    const std::int64_t count = us[static_cast<NSUInteger>(axis)].longLongValue;
    if (count < 0 || count > kMaxHalfWordUpdates)
        return nil;
    const NSInteger last = static_cast<NSInteger>(us.count);
    MPSGraphTensor* words = [g reinterpretCastTensor:[g expandDimsOfTensor:updates
                                                                      axis:last
                                                                      name:nil]
                                              toType:MPSDataTypeInt32
                                                name:nil];
    const auto word = [&](NSInteger k) {
        return [g squeezeTensor:[g sliceTensor:words dimension:last start:k length:1 name:nil]
                           axis:last
                           name:nil];
    };
    MPSGraphTensor* low = word(0);
    MPSGraphTensor* mask = [g constantWithScalar:65535.0 dataType:MPSDataTypeInt32];
    MPSGraphTensor* low_half = [g bitwiseANDWithPrimaryTensor:low secondaryTensor:mask name:nil];
    MPSGraphTensor* high_half =
        [g bitwiseANDWithPrimaryTensor:
                [g bitwiseRightShiftWithPrimaryTensor:low
                                      secondaryTensor:[g constantWithScalar:16.0
                                                                   dataType:MPSDataTypeInt32]
                                                 name:nil]
                       secondaryTensor:mask
                                  name:nil];
    MPSGraphTensor* zeros = int32_filled(g, data, 0.0);
    const auto sum = [&](MPSGraphTensor* part, double scale) {
        MPSGraphTensor* s = [g scatterAlongAxis:axis
                                 withDataTensor:zeros
                                  updatesTensor:part
                                  indicesTensor:idx
                                           mode:MPSGraphScatterModeAdd
                                           name:nil];
        return [g multiplicationWithPrimaryTensor:[g castTensor:s toType:MPSDataTypeInt64 name:nil]
                                  secondaryTensor:[g constantWithScalar:scale
                                                               dataType:MPSDataTypeInt64]
                                             name:nil];
    };
    MPSGraphTensor* total = [g additionWithPrimaryTensor:sum(word(1), 4294967296.0)
                                         secondaryTensor:sum(high_half, 65536.0)
                                                    name:nil];
    total = [g additionWithPrimaryTensor:total secondaryTensor:sum(low_half, 1.0) name:nil];
    return [g additionWithPrimaryTensor:data secondaryTensor:total name:@"scatter_add_int64"];
}

}  // namespace data_movement_detail

// ``gatherAlongAxis``, exact for every dtype.
inline MPSGraphTensor* graph_gather_along_axis(
    MPSGraph* g, NSInteger axis, MPSGraphTensor* data, MPSGraphTensor* idx, NSString* name = nil) {
    namespace d = data_movement_detail;
    const MPSDataType dt = data.dataType;
    if (graph_moves_exactly(dt, GraphMove::Gather))
        return [g gatherAlongAxis:axis withUpdatesTensor:data indicesTensor:idx name:name];
    const NSUInteger lanes = d::lanes_of(dt);
    const auto merged = lanes == 0 ? std::nullopt
                                   : lane_gather_along_axis(shape_of_mps(data), shape_of_mps(idx),
                                                            axis, static_cast<std::int64_t>(lanes));
    MPSGraphTensor* c = d::carry(g, merged ? d::read_as(g, data, merged->data) : data);
    MPSGraphTensor* i = d::lane_index(g, merged ? d::read_as(g, idx, merged->index) : idx, lanes);
    if (c == nil || i == nil)
        return nil;
    MPSGraphTensor* out =
        d::uncarry(g,
                   [g gatherAlongAxis:merged ? static_cast<NSInteger>(merged->axis) : axis
                       withUpdatesTensor:c
                           indicesTensor:i
                                    name:name],
                   dt);
    return merged ? d::read_as(g, out, merged->result) : out;
}

// ``gather`` (the axis gather, no batch axes), exact for every dtype.
inline MPSGraphTensor* graph_gather(
    MPSGraph* g, MPSGraphTensor* data, MPSGraphTensor* idx, NSUInteger axis, NSString* name = nil) {
    namespace d = data_movement_detail;
    const MPSDataType dt = data.dataType;
    if (graph_moves_exactly(dt, GraphMove::Gather))
        return [g gatherWithUpdatesTensor:data
                            indicesTensor:idx
                                     axis:axis
                          batchDimensions:0
                                     name:name];
    const NSUInteger lanes = d::lanes_of(dt);
    const auto merged =
        lanes == 0 ? std::nullopt
                   : lane_take(shape_of_mps(data), shape_of_mps(idx),
                               static_cast<std::int64_t>(axis), static_cast<std::int64_t>(lanes));
    MPSGraphTensor* src = d::carry(g, merged ? d::read_as(g, data, merged->data) : data);
    if (src == nil)
        return nil;
    MPSGraphTensor* out =
        d::uncarry(g,
                   [g gatherWithUpdatesTensor:src
                                indicesTensor:merged ? d::read_as(g, idx, merged->index) : idx
                                         axis:merged ? merged->axis : axis
                              batchDimensions:0
                                         name:name],
                   dt);
    return merged ? d::read_as(g, out, merged->result) : out;
}

namespace data_movement_detail {

// An overwrite of a multi-lane dtype: which update wins each slot is
// decided once, on its int32 position along ``axis``, and that update is
// then read whole.
inline MPSGraphTensor* scatter_set_whole(MPSGraph* g,
                                         NSInteger axis,
                                         MPSGraphTensor* data,
                                         MPSGraphTensor* updates,
                                         MPSGraphTensor* idx,
                                         NSString* name) {
    MPSGraphTensor* position = [g coordinateAlongAxis:axis
                                      withShapeTensor:[g shapeOfTensor:updates name:nil]
                                                 name:nil];
    if (position.dataType != MPSDataTypeInt32)
        position = [g castTensor:position toType:MPSDataTypeInt32 name:nil];
    MPSGraphTensor* winner = [g scatterAlongAxis:axis
                                  withDataTensor:int32_filled(g, data, -1.0)
                                   updatesTensor:position
                                   indicesTensor:idx
                                            mode:MPSGraphScatterModeSet
                                            name:name];
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:MPSDataTypeInt32];
    MPSGraphTensor* read_at = [g maximumWithPrimaryTensor:winner secondaryTensor:zero name:nil];
    MPSGraphTensor* written = graph_gather_along_axis(g, axis, updates, read_at);
    if (written == nil)
        return nil;
    return [g selectWithPredicateTensor:[g greaterThanOrEqualToWithPrimaryTensor:winner
                                                                 secondaryTensor:zero
                                                                            name:nil]
                    truePredicateTensor:written
                   falsePredicateTensor:data
                                   name:@"scatter_set_whole"];
}

}  // namespace data_movement_detail

// ``scatterAlongAxis`` of ``updates`` into ``data`` (one dtype), exact for
// every dtype it answers.  nil when MPSGraph cannot apply ``mode`` to the
// dtype exactly; the emitter declines then.
inline MPSGraphTensor* graph_scatter_along_axis(MPSGraph* g,
                                                NSInteger axis,
                                                MPSGraphTensor* data,
                                                MPSGraphTensor* updates,
                                                MPSGraphTensor* idx,
                                                MPSGraphScatterMode mode,
                                                NSString* name = nil) {
    namespace d = data_movement_detail;
    const MPSDataType dt = data.dataType;
    if (updates.dataType != dt)
        return nil;
    if (graph_moves_exactly(dt, GraphMove::Scatter))
        return [g scatterAlongAxis:axis
                    withDataTensor:data
                     updatesTensor:updates
                     indicesTensor:idx
                              mode:mode
                              name:name];
    if (mode == MPSGraphScatterModeAdd && dt == MPSDataTypeInt64)
        return d::scatter_add_int64(g, axis, data, updates, idx);
    if (mode != MPSGraphScatterModeSet)
        return nil;
    if (d::lanes_of(dt) > 0)
        return d::scatter_set_whole(g, axis, data, updates, idx, name);
    // bool: one int8 element per value, so a slot cannot be torn.
    MPSGraphTensor* c = d::carry(g, data);
    MPSGraphTensor* u = d::carry(g, updates);
    if (c == nil || u == nil)
        return nil;
    return d::uncarry(g,
                      [g scatterAlongAxis:axis
                           withDataTensor:c
                            updatesTensor:u
                            indicesTensor:idx
                                     mode:mode
                                     name:name],
                      dt);
}

}  // namespace lucid::compile
