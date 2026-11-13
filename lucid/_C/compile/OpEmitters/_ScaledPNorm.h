// lucid/_C/compile/OpEmitters/_ScaledPNorm.h
//
// The range-safe entrywise p-norm of a compiled graph (LCD-955), as MPSGraph ops.
//
// ``sqrt(sum(x * x))`` squares a float32 magnitude below ~1e-19 into the
// subnormals, which Metal flushes to zero, and one above ~1.8e19 into infinity,
// though the norm itself is an ordinary float.  Eager Metal answers this with a
// fused kernel (``backend/gpu/PNorm.h``) that keeps a running largest magnitude;
// a compiled graph has no kernel to call, so it spells out the same algorithm
// BLAS's ``nrm2`` has used since the seventies:
//
//     m = max |x|          over the reduced axes
//     d = clamp(m, FLT_MIN, FLT_MAX)
//     s = sum (|x| / d)^p
//     N = d * s^(1/p)
//
// Every ratio of a finite slice is in [0, 1], so no power leaves the float
// range.  The clamp stands where the eager kernel branches, and is one node on
// the reduced tensor where a compare-and-select was four: a compiled step is
// priced per kernel at these sizes, and the select form cost +2.5 ms on a
// 22 ms ``F.normalize`` step against the clamp's +0.4 (LCD-955).
//
//   * all zero   m = 0 would divide 0 by 0; d = FLT_MIN gives 0 / d = 0, s = 0,
//                N = 0 * d = 0
//   * infinity   d = FLT_MAX, so inf / d = inf, s = inf and N = inf * d = inf
//   * NaN        the NaN stays in |x| / d, and a sum holding a NaN is NaN,
//                whatever the plain ``reductionMaximum`` (which skips NaN) made
//                of ``m``
//
// so the special values come out as the eager kernel's.  The VJP of the op uses
// the same ratio (``|x| / N`` is in [0, 1] too), :func:`p_norm_grad`.
//
// Objective-C++ only.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "../TraceIR.h"
#include "_AttrHelpers.h"
#include "_ReduceCanon.h"

namespace lucid::compile {

// The saved-state slot (``BuilderContext::stash_saved``) under which a compiled
// ``lp_normalize`` hands its norm, keepdim-shaped, to its VJP.
inline constexpr const char* kLpNormSavedSlot = "lp_normalize_norm";

// ``(sum |x|^p)^(1/p)`` of float32 ``x`` over ``axes`` for a finite ``p > 1``,
// reduced axes kept at size 1.  The caller reshapes to the shape it needs.
[[maybe_unused]] inline MPSGraphTensor*
scaled_p_norm_keepdim(MPSGraph* g, MPSGraphTensor* x, NSArray<NSNumber*>* axes, double p) {
    const MPSDataType dt = x.dataType;
    MPSGraphTensor* a = [g absoluteWithTensor:x name:nil];
    MPSGraphTensor* top = [g reductionMaximumWithTensor:a axes:axes name:@"p_norm_top"];
    MPSGraphTensor* divisor =
        [g clampWithTensor:top
            minValueTensor:[g constantWithScalar:std::numeric_limits<float>::min() dataType:dt]
            maxValueTensor:[g constantWithScalar:std::numeric_limits<float>::max() dataType:dt]
                      name:@"p_norm_divisor"];
    MPSGraphTensor* ratio = [g divisionWithPrimaryTensor:a secondaryTensor:divisor name:nil];
    const bool two = (p == 2.0);
    MPSGraphTensor* powered = two ? [g squareWithTensor:ratio name:nil]
                                  : [g powerWithPrimaryTensor:ratio
                                              secondaryTensor:[g constantWithScalar:p dataType:dt]
                                                         name:nil];
    MPSGraphTensor* s = [g reductionSumWithTensor:powered axes:axes name:nil];
    MPSGraphTensor* root = two ? [g squareRootWithTensor:s name:nil]
                               : [g powerWithPrimaryTensor:s
                                           secondaryTensor:[g constantWithScalar:(1.0 / p)
                                                                        dataType:dt]
                                                      name:nil];
    return [g multiplicationWithPrimaryTensor:root secondaryTensor:divisor name:@"p_norm"];
}

// The Euclidean norm ``lp_normalize`` divides by: the scaled form for float32,
// the plain ``sqrt(sum(x * x))`` for any other float (autocast hands it half).
[[maybe_unused]] inline MPSGraphTensor*
euclidean_norm_keepdim(MPSGraph* g, MPSGraphTensor* x, NSArray<NSNumber*>* axes) {
    if (x.dataType == MPSDataTypeFloat32)
        return scaled_p_norm_keepdim(g, x, axes, 2.0);
    MPSGraphTensor* sum = [g reductionSumWithTensor:[g squareWithTensor:x name:nil]
                                               axes:axes
                                               name:nil];
    return [g squareRootWithTensor:sum name:nil];
}

// What a recorded ``p_norm`` node says: the order and the axes it reduced.
struct PNormSpec {
    double p = 2.0;
    std::vector<bool> reduced;
};

// ``node``'s order and reduced axes over an input of ``x_shape`` (no ``axis_list``
// = every axis); nullopt for what the lowering does not serve — an order that is
// not finite and above 1, a 0-d input, an axis outside the rank.
[[maybe_unused]] inline std::optional<PNormSpec>
p_norm_spec(const OpNode& node, const std::vector<std::int64_t>& x_shape) {
    const double p = double_attr(node, "p", 0.0);
    if (!(p > 1.0) || std::isinf(p) || x_shape.empty())
        return std::nullopt;
    const auto* dims = int_vec_attr(node, "axis_list");
    auto reduced = dims == nullptr || dims->empty()
                       ? std::optional<std::vector<bool>>(std::vector<bool>(x_shape.size(), true))
                       : reduce_canon_mask(x_shape.size(), *dims);
    if (!reduced)
        return std::nullopt;
    return PNormSpec{p, std::move(*reduced)};
}

// The engine's ``PNormBackward``: ``sign(x) * (|x| / N)^(p-1) * dN``, 0 where
// ``N == 0`` (the subgradient; its ratio is 0 / 0) and NaN where ``N`` is.  ``n``
// and ``go`` are the norm and its gradient in keepdim form, ``go``'s dtype the
// chain's.
[[maybe_unused]] inline MPSGraphTensor*
p_norm_grad(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* n, MPSGraphTensor* go, double p) {
    const MPSDataType dt = go.dataType;
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
    MPSGraphTensor* slope = nil;
    if (p == 2.0) {
        // sign(x) * |x| / N is x / N, one pass shorter.
        slope = [g divisionWithPrimaryTensor:x secondaryTensor:n name:nil];
    } else {
        MPSGraphTensor* ratio = [g divisionWithPrimaryTensor:[g absoluteWithTensor:x name:nil]
                                             secondaryTensor:n
                                                        name:nil];
        MPSGraphTensor* powered = [g powerWithPrimaryTensor:ratio
                                            secondaryTensor:[g constantWithScalar:p - 1.0
                                                                         dataType:dt]
                                                       name:nil];
        slope = [g multiplicationWithPrimaryTensor:[g signWithTensor:x name:nil]
                                   secondaryTensor:powered
                                              name:nil];
    }
    MPSGraphTensor* grad = [g multiplicationWithPrimaryTensor:slope secondaryTensor:go name:nil];
    return [g selectWithPredicateTensor:[g notEqualWithPrimaryTensor:n
                                                     secondaryTensor:zero
                                                                name:nil]
                    truePredicateTensor:grad
                   falsePredicateTensor:zero
                                   name:@"p_norm_vjp"];
}

}  // namespace lucid::compile
