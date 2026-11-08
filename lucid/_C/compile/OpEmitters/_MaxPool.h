// lucid/_C/compile/OpEmitters/_MaxPool.h
//
// A max pool that keeps a window of nothing but ``-inf`` at ``-inf`` (LCD-916).
//
// MPSGraph's pooling kernel starts its running maximum at the dtype's lowest
// finite value and only takes an element that is *greater*, so a window whose
// every element is ``-inf`` (or exactly the lowest finite value) is an empty
// window to it and comes back ``0`` — eager, and the reference, say ``-inf``.
// A masked attention score or a log-probability pooled over a masked span is
// exactly such a window.
//
// ``stencil`` with a Max reduction is the same pool but its empty window comes
// back as that start value, ``lowest``, which no real window answers (see the
// one exception below).  That is what makes it usable: the empty window is
// recognisable at the *output*, where one ``select`` per output element (fused
// into the stencil, measured free) turns it into ``-inf``.  The stencil also
// fuses a producer into itself, which the pooling kernel does not.
//
// The one value the kernel cannot tell from an empty window is an element equal
// to ``lowest`` itself.  A prologue moves it one step toward zero (``next``) and
// the epilogue moves a result of ``next`` back, so genuine ``lowest`` windows
// stay exact; the cost is that a window whose true maximum is exactly ``next``
// (the second-lowest finite value, ``-3.4028233e38`` in float32) answers
// ``lowest``, one step off.  Pigeonhole: nothing in one dtype can keep every
// value and add an "empty" code, so the unavoidable collision sits where no
// data lives.
//
// Padding is ``-inf`` and never contributes.  A NaN is still skipped, as it is
// by every MPSGraph extremum (LCD-913); a window of nothing but NaN reads as
// ``-inf`` here.
//
// Measured (MacBook M1 Pro, Metal, GPU-serial, 32x64x112x112 float32, standalone
// MPSGraph): plain pool 0.705 / 0.879 ms (k2 s2 / k3 s2 p1), this form 0.736 /
// 0.861 ms.  A separate ``select`` after the plain pool is +60% instead: only the
// stencil fuses it.
//
// Objective-C++ only.

#pragma once

#import <Foundation/Foundation.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace lucid::compile {

namespace max_pool_detail {

// The lowest finite value of a float dtype and the step from it toward zero.
struct Range {
    double lowest;
    double ulp;
};

inline bool range_of(MPSDataType dt, Range& r) {
    switch (dt) {
    case MPSDataTypeFloat32:
        r = {-3.4028234663852886e38, std::ldexp(1.0, 104)};
        return true;
    case MPSDataTypeFloat16:
        r = {-65504.0, 32.0};
        return true;
    case MPSDataTypeBFloat16:
        r = {-3.3895313892515355e38, std::ldexp(1.0, 120)};
        return true;
    default:
        return false;
    }
}

// ``stencil`` is rank 4 only.  A 2-D pool is as it is; a 3-D pool folds the batch and channel
// axes into one; a 1-D pool becomes ``1, B, C, L``.  The layout matters to the kernel: on
// 64x64x8192 (with the reshapes in the graph) ``B, C, 1, L`` was +17% over the plain pool,
// ``B, 1, C, L`` +38%, ``1, 1, B*C, L`` +50% and ``1, B, C, L`` -16%, and it is no slower on any
// size tried (0.02-2.2 ms).  ``nil`` for a 2-D pool: no reshape.
inline NSArray<NSNumber*>* lifted_shape(NSArray<NSNumber*>* xs, std::size_t nd) {
    if (nd == 1)
        return @[ @1, xs[0], xs[1], xs[2] ];
    if (nd == 3)
        return @[ @(-1), xs[2], xs[3], xs[4] ];
    return nil;
}

// The stencil's descriptor and window for the pool over ``nd`` trailing axes, the leading
// axes of the rank-4 operand being windowed by 1.  ``nil`` when an extent is dynamic.
inline MPSGraphStencilOpDescriptor* descriptor(NSArray<NSNumber*>* xs,
                                               const std::vector<std::int64_t>& kernel,
                                               const std::vector<std::int64_t>& stride,
                                               const std::vector<std::int64_t>& pad,
                                               const std::vector<std::int64_t>& out_spatial,
                                               NSMutableArray<NSNumber*>* window) {
    const std::size_t nd = kernel.size();
    NSMutableArray<NSNumber*>* strides = [NSMutableArray array];
    NSMutableArray<NSNumber*>* padding = [NSMutableArray array];
    for (std::size_t i = nd; i < 4; ++i) {
        [strides addObject:@1];
        [padding addObject:@0];
        [padding addObject:@0];
        [window addObject:@1];
    }
    for (std::size_t a = 0; a < nd; ++a) {
        const long long in = xs[a + 2].longLongValue;
        if (in < 0 || out_spatial[a] < 0)
            return nil;
        // Past the symmetric padding, the overhang a ceil-mode window reaches.
        const long long reach = (out_spatial[a] - 1) * stride[a] + kernel[a] - in - 2 * pad[a];
        [padding addObject:@(pad[a])];
        [padding addObject:@(pad[a] + (reach > 0 ? reach : 0))];
        [strides addObject:@(stride[a])];
        [window addObject:@(kernel[a])];
    }
    return [MPSGraphStencilOpDescriptor descriptorWithReductionMode:MPSGraphReductionModeMax
                                                            offsets:@[ @0, @0, @0, @0 ]
                                                            strides:strides
                                                      dilationRates:@[ @1, @1, @1, @1 ]
                                                    explicitPadding:padding
                                                       boundaryMode:MPSGraphPaddingModeConstant
                                                       paddingStyle:MPSGraphPaddingStyleExplicit
                                                    paddingConstant:-INFINITY];
}

}  // namespace max_pool_detail

// ``x`` max-pooled over its trailing ``kernel.size()`` axes, an all-``-inf`` window
// answering ``-inf``.  ``out_spatial`` is the pooled extent of those axes (the trace's
// recorded shape — it carries ceil mode).  ``nil`` when this dtype or shape cannot take
// the stencil path; the caller then uses the plain pool.
[[maybe_unused]] inline MPSGraphTensor*
exact_max_pool(MPSGraph* g,
               MPSGraphTensor* x,
               const std::vector<std::int64_t>& kernel,
               const std::vector<std::int64_t>& stride,
               const std::vector<std::int64_t>& pad,
               const std::vector<std::int64_t>& out_spatial) {
    NSArray<NSNumber*>* xs = x.shape;
    const std::size_t nd = kernel.size();
    max_pool_detail::Range range{};
    if (nd < 1 || nd > 3 || xs.count != nd + 2 || stride.size() != nd || pad.size() != nd ||
        out_spatial.size() != nd || !max_pool_detail::range_of(x.dataType, range))
        return nil;

    NSMutableArray<NSNumber*>* window = [NSMutableArray array];
    MPSGraphStencilOpDescriptor* d =
        max_pool_detail::descriptor(xs, kernel, stride, pad, out_spatial, window);
    if (d == nil)
        return nil;

    const MPSDataType dt = x.dataType;
    MPSGraphTensor* lo = [g constantWithScalar:range.lowest dataType:dt];
    MPSGraphTensor* next = [g constantWithScalar:range.lowest + range.ulp dataType:dt];
    MPSGraphTensor* ninf = [g constantWithScalar:-INFINITY dataType:dt];
    MPSGraphTensor* ones = [g constantWithScalar:1.0 shape:window dataType:dt];

    NSArray<NSNumber*>* lifted = max_pool_detail::lifted_shape(xs, nd);
    MPSGraphTensor* src = lifted != nil ? [g reshapeTensor:x withShape:lifted name:nil] : x;
    MPSGraphTensor* is_lowest = [g equalWithPrimaryTensor:src secondaryTensor:lo name:nil];
    MPSGraphTensor* held = [g selectWithPredicateTensor:is_lowest
                                    truePredicateTensor:next
                                   falsePredicateTensor:src
                                                   name:nil];
    MPSGraphTensor* y = [g stencilWithSourceTensor:held weightsTensor:ones descriptor:d name:nil];
    if (y == nil)
        return nil;

    // ``<= lowest`` is an empty window (an fp16 / bf16 stencil may round its start value to
    // ``-inf`` or keep it as a float); ``next`` is a held ``lowest``.
    MPSGraphTensor* empty = [g lessThanOrEqualToWithPrimaryTensor:y secondaryTensor:lo name:nil];
    MPSGraphTensor* was_lowest = [g equalWithPrimaryTensor:y secondaryTensor:next name:nil];
    MPSGraphTensor* restored = [g selectWithPredicateTensor:was_lowest
                                        truePredicateTensor:lo
                                       falsePredicateTensor:y
                                                       name:nil];
    MPSGraphTensor* out = [g selectWithPredicateTensor:empty
                                   truePredicateTensor:ninf
                                  falsePredicateTensor:restored
                                                  name:nil];
    if (lifted == nil)
        return out;
    // Back to the input's rank: the lifted output is (1, B, C, Lout) or (B*C, D, H, W).
    NSMutableArray<NSNumber*>* back = [NSMutableArray arrayWithObjects:xs[0], xs[1], nil];
    for (std::size_t a = 0; a < nd; ++a)
        [back addObject:@(out_spatial[a])];
    return [g reshapeTensor:out withShape:back name:nil];
}

}  // namespace lucid::compile
