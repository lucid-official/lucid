// lucid/_C/compile/OpEmitters/_NanPropagation.h
//
// Where an extremum meets a NaN inside a compiled graph (LCD-873).
//
// MPSGraph's plain ``maximum`` / ``minimum``, ``reLU``, ``clamp``, pooling and
// cumulative extremum all *skip* NaN: ``maximum(1, nan)`` is 1, ``relu(nan)``
// is 0, ``max_pool([1, nan])`` is 1 and ``cummax([1, nan, 2])`` carries 2 past
// the NaN.  Eager's kernels propagate it — a NaN anywhere in a window, a scan
// prefix or an operand pair is the answer — so a compiled model hid a layer
// that had gone bad: its loss stayed finite where eager's is NaN.
//
// The plain result is kept and a *NaN carrier* is added to it: a float16 tensor
// that is NaN where the operand is and ``-0`` elsewhere, built as
// ``-floor(sigmoid(x) * 0)``.  ``sigmoid`` maps ``+-inf`` to 0 and 1 so only a
// NaN survives the product with 0 (``tanh`` would not do: on this GPU it maps NaN to
// 1).  ``y + (-0)`` is ``y`` for every ``y``, ``-0`` included, and a carrier carried
// through a running sum is NaN exactly for the prefixes that held a NaN.  The carrier
// is a half-width tensor, fused into the elementwise kernel it joins.
//
// Max pooling does not use this carrier: it would need a second (average) pooling over
// the whole input (+58-65% on a compiled ``max_pool2d``).  It encodes the NaN in the
// stencil's fused prologue and epilogue instead (``_MaxPool.h``, LCD-913).
//
// Why not MPSGraph's own ``maximumWithNaNPropagation`` / ``minimumWithNaNPropagation``,
// ``select`` or ``isNaN``: the autodiff fallback (``kAutodiffSafeOps``,
// MpsBuilder.mm) differentiates the forward graph, and it aborts the process on
// all of them, even when the term sits behind a ``floor`` (measured: "Couldn't
// get gradient Tensor for tensor of op : feed").  ``floor``'s gradient is built
// as zeros, so it cuts the carrier off from the gradient, which then runs
// through the plain operation as it always did; only ``sigmoid``,
// ``multiplication`` and ``floor``, all differentiable, stand behind it.
//
// An integer or bool tensor holds no NaN and keeps the plain operation.
//
// Objective-C++ only.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace lucid::compile {

inline bool holds_nan(MPSGraphTensor* t) {
    return (t.dataType & MPSDataTypeFloatBit) != 0;
}

// NaN where ``x`` is NaN, ``-0`` elsewhere, in float16 (``nil`` for a dtype with no NaN).
// A value past float16's range becomes ``inf``, which ``sigmoid`` maps to 1.
inline MPSGraphTensor* nan_carrier(MPSGraph* g, MPSGraphTensor* x) {
    if (!holds_nan(x))
        return nil;
    MPSGraphTensor* h =
        x.dataType == MPSDataTypeFloat16 ? x : [g castTensor:x toType:MPSDataTypeFloat16 name:nil];
    MPSGraphTensor* zero_unless_nan =
        [g multiplicationWithPrimaryTensor:[g sigmoidWithTensor:h name:nil]
                           secondaryTensor:[g constantWithScalar:0.0 dataType:MPSDataTypeFloat16]
                                      name:nil];
    return [g negativeWithTensor:[g floorWithTensor:zero_unless_nan name:nil] name:nil];
}

// ``y`` with NaN wherever ``carrier`` has one.
inline MPSGraphTensor* add_nan_carrier(MPSGraph* g, MPSGraphTensor* y, MPSGraphTensor* carrier) {
    if (carrier == nil || y == nil)
        return y;
    MPSGraphTensor* c = carrier.dataType == y.dataType
                            ? carrier
                            : [g castTensor:carrier toType:y.dataType name:nil];
    return [g additionWithPrimaryTensor:y secondaryTensor:c name:nil];
}

// ``y`` with NaN wherever ``x`` has one (``y`` an elementwise result of ``x``).
inline MPSGraphTensor* keep_nan_of(MPSGraph* g, MPSGraphTensor* y, MPSGraphTensor* x) {
    return add_nan_carrier(g, y, nan_carrier(g, x));
}

inline MPSGraphTensor*
maximum_keeping_nan(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b, NSString* name) {
    MPSGraphTensor* y = [g maximumWithPrimaryTensor:a secondaryTensor:b name:name];
    return keep_nan_of(g, keep_nan_of(g, y, a), b);
}

inline MPSGraphTensor*
minimum_keeping_nan(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b, NSString* name) {
    MPSGraphTensor* y = [g minimumWithPrimaryTensor:a secondaryTensor:b name:name];
    return keep_nan_of(g, keep_nan_of(g, y, a), b);
}

inline MPSGraphTensor* relu_keeping_nan(MPSGraph* g, MPSGraphTensor* x, NSString* name) {
    return keep_nan_of(g, [g reLUWithTensor:x name:name], x);
}

// ``y``, a running extremum of ``x`` along ``axis``, with NaN from the first NaN on.
inline MPSGraphTensor*
scan_keeping_nan(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* y, NSInteger axis) {
    MPSGraphTensor* carrier = nan_carrier(g, x);
    if (carrier == nil)
        return y;
    return add_nan_carrier(g, y, [g cumulativeSumWithTensor:carrier axis:axis name:nil]);
}

}  // namespace lucid::compile
