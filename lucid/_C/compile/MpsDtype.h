// lucid/_C/compile/MpsDtype.h
//
// The one table from a Lucid dtype to the MPSGraph type that holds it, and
// the rule that every value the compiled graph computes is held at the dtype
// the trace declared for it.
//
// Seven copies of the table used to live in the builder, the executable and
// five emitters.  They disagreed: some answered float32 for a dtype they did
// not know, so a ``full`` of int8 built a float32 constant and an ``astype``
// to an unmapped dtype cast to float32, each under a trace that said
// otherwise.  An emitter whose MPSGraph primitive picks its own result type
// did the same: ``argmax`` came out int32 where Lucid types it int64, and the
// next comparison against an int64 constant aborted the process inside
// MPSGraph ("input types 'tensor<4xsi32>' and 'tensor<si64>' are not
// broadcast compatible", LCD-270).  MPSGraph aborts rather than declines on a
// binary op over two dtypes, and the engine never dispatches one — eager
// inserts an ``astype`` first — so the graph is safe exactly when each value
// has its declared dtype.  ``held_at`` is how the builder enforces that.
//
// Objective-C++ only.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <optional>
#include <stdexcept>

#include "../core/Dtype.h"

namespace lucid::compile {

// The MPSGraph type holding ``dt``, or nothing when MPSGraph has none.
// float64 and complex128 have none (Metal has no 64-bit float lanes); a value
// of either cannot be compiled, and nothing may stand in for it.
inline std::optional<MPSDataType> mps_dtype_of(Dtype dt) {
    switch (dt) {
    case Dtype::F32:
        return MPSDataTypeFloat32;
    case Dtype::F16:
        return MPSDataTypeFloat16;
    case Dtype::BF16:
        return MPSDataTypeBFloat16;
    case Dtype::I64:
        return MPSDataTypeInt64;
    case Dtype::I32:
        return MPSDataTypeInt32;
    case Dtype::I16:
        return MPSDataTypeInt16;
    case Dtype::I8:
        return MPSDataTypeInt8;
    case Dtype::Bool:
        return MPSDataTypeBool;
    // Lucid stores complex64 interleaved, a pair of float32 lanes in one
    // storage, which is exactly MPSGraph's ``MPSDataTypeComplexFloat32``.
    case Dtype::C64:
        return MPSDataTypeComplexFloat32;
    default:
        return std::nullopt;
    }
}

// ``mps_dtype_of`` for a feed or a bound buffer, where a dtype without an
// MPSGraph type is a caller error rather than a reason to decline an op.
inline MPSDataType mps_dtype_or_throw(Dtype dt) {
    if (const auto mdt = mps_dtype_of(dt))
        return *mdt;
    if (dt == Dtype::C128)
        throw std::runtime_error(
            "lucid::compile: complex128 has no MPSGraph type (complex is float32 / "
            "float16 lanes only) — cast to complex64 to compile this graph");
    throw std::runtime_error("lucid::compile: dtype not supported on the MPSGraph compile path");
}

// Materialise a float16 / bfloat16 tensor's 16-bit representation before
// widening it to float32.  MPSGraph's MLIR optimizer otherwise folds
// ``cast(op_f16(...), f32)`` into ``op_f32(...)``, eliding the IEEE binary16 /
// bfloat16 rounding and finite-range clamp (65504 for float16).
inline MPSGraphTensor* commit_half_before_widen(MPSGraph* g, MPSGraphTensor* x, MPSDataType dst) {
    if (g == nil || x == nil || dst != MPSDataTypeFloat32 ||
        (x.dataType != MPSDataTypeFloat16 && x.dataType != MPSDataTypeBFloat16))
        return x;
    const bool scalar = (x.shape != nil && x.shape.count == 0);
    MPSGraphTensor* r = scalar ? [g expandDimsOfTensor:x axis:0 name:nil] : x;
    MPSGraphTensor* i16 = [g reinterpretCastTensor:r toType:MPSDataTypeInt16 name:nil];
    MPSGraphTensor* back = [g reinterpretCastTensor:i16 toType:x.dataType name:nil];
    return scalar ? [g squeezeTensor:back axis:0 name:nil] : back;
}

inline MPSGraphTensor*
cast_mps_tensor(MPSGraph* g, MPSGraphTensor* x, MPSDataType dst, NSString* name = nil) {
    if (x == nil || g == nil || x.dataType == dst)
        return x;
    return [g castTensor:commit_half_before_widen(g, x, dst) toType:dst name:name];
}

// ``t`` held at the declared dtype ``dt``: ``t`` itself when it already is,
// a cast when only the width or the kind of a real value differs, and nil
// when ``dt`` has no MPSGraph type or the cast would cross between real and
// complex (which a cast does not mean).  nil fails the build, so the trace
// runs eagerly instead of reaching MPSGraph with two dtypes.
inline MPSGraphTensor* held_at(MPSGraph* g, MPSGraphTensor* t, Dtype dt) {
    const auto want = mps_dtype_of(dt);
    if (!want)
        return nil;
    if (t.dataType == *want)
        return t;
    const bool t_complex = (t.dataType & MPSDataTypeComplexBit) != 0;
    const bool want_complex = (*want & MPSDataTypeComplexBit) != 0;
    if (t_complex != want_complex)
        return nil;
    return cast_mps_tensor(g, t, *want, @"held_at_declared_dtype");
}

}  // namespace lucid::compile
