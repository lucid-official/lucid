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
// An order below 1 (a quasi-norm, LCD-1015) has the same shape with three differences,
// each one eager Metal's (``backend/gpu/PNorm.h``, LCD-1009):
//
//   * Its power is not negligible where the ratio underflows: ``r^p > r``, so for
//     ``p < 1/2`` an entry 40 decades below the largest still adds ``1e-40^p`` (1e-4
//     at ``p = 0.1``), and the division flushes that ratio to zero.  Such a term is
//     the power of a rescaled ratio instead (:func:`quasi_ratio_power`); the ratios
//     above the floor, and every ratio for ``p >= 1/2`` (whose underflowed power is
//     below 1e-19), keep the plain ``pow``.
//   * ``s`` is in [1, n] and the root ``s^(1/p)`` reaches ``n^(1/p)`` (1e40 for 1e4
//     equal entries at ``p = 0.1``) while ``d * s^(1/p)`` is an ordinary float, so it
//     is taken in two halves, ``(d * h) * h`` with ``h = s^(1/(2p)) >= 1`` and
//     ``d * h`` no larger than the answer.
//   * Its gradient has an infinite slope at a zero entry, which is defined as 0
//     (:func:`quasi_p_norm_grad`).
//
// The plain chain a trace recorded for these orders kept the value and the first
// gradient in range but not accurate: ``pow`` is ``exp2(y log2 x)`` in float32, whose
// error grows with ``|y log2 x|``, so a slice at 1e+-25 lost up to 20 ulp at ``p = 0.75``
// where a slice at 1 lost one.  The ratio form's exponents are small.
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

// ``(|x| / d)^q`` of the quasi-norm's terms (``q = p``) and slopes (``q = p - 1``), where
// ``a = |x|``, ``d`` is the slice's divisor and ``ratio = a / d``.  A ratio below float32's
// smallest normal is flushed to zero by the division, and its power is not negligible (the
// file header), so with ``rescue`` those entries take ``((a 2^126) / d)^q 2^(-126 q)``: the
// ratio of a power-of-two multiple of ``a`` (below 4 there, so exact and finite), whose
// exponent ``q log2`` stays small for the ratios that matter, where the difference of two
// float32 logarithms of magnitude ~100 loses ~1e-5 of the entry (``2^(q (log2 a - log2 d))``
// measured 120 ulp on a slope).  The select is on the base and the factor, so the one ``pow``
// is the only transcendental pass.  An entry of 0 gets ``0^q``: 0 for the terms, ``inf`` for
// the slopes (the caller masks it).
[[maybe_unused]] inline MPSGraphTensor* quasi_ratio_power(MPSGraph* g,
                                                          MPSGraphTensor* a,
                                                          MPSGraphTensor* d,
                                                          MPSGraphTensor* ratio,
                                                          double q,
                                                          bool rescue) {
    const MPSDataType dt = a.dataType;
    MPSGraphTensor* exponent = [g constantWithScalar:q dataType:dt];
    if (!rescue)
        return [g powerWithPrimaryTensor:ratio secondaryTensor:exponent name:nil];
    constexpr double kRescale = 0x1p126;
    MPSGraphTensor* lifted = [g
        divisionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:a
                                                     secondaryTensor:[g constantWithScalar:kRescale
                                                                                  dataType:dt]
                                                                name:nil]
                  secondaryTensor:d
                             name:nil];
    MPSGraphTensor* flushed =
        [g lessThanWithPrimaryTensor:ratio
                     secondaryTensor:[g constantWithScalar:std::numeric_limits<float>::min()
                                                  dataType:dt]
                                name:nil];
    MPSGraphTensor* base = [g selectWithPredicateTensor:flushed
                                    truePredicateTensor:lifted
                                   falsePredicateTensor:ratio
                                                   name:nil];
    MPSGraphTensor* factor =
        [g selectWithPredicateTensor:flushed
                 truePredicateTensor:[g constantWithScalar:std::exp2(-126.0 * q) dataType:dt]
                falsePredicateTensor:[g constantWithScalar:1.0 dataType:dt]
                                name:nil];
    return [g multiplicationWithPrimaryTensor:[g powerWithPrimaryTensor:base
                                                        secondaryTensor:exponent
                                                                   name:nil]
                              secondaryTensor:factor
                                         name:nil];
}

// What the quasi-norm's forward computes and its VJP reads again: the divisor
// ``d = clamp(max |x|)``, ``s = sum (|x| / d)^p`` and the norm ``d s^(1/p)``, all
// keepdim-shaped.
struct QuasiNormParts {
    MPSGraphTensor* divisor = nil;
    MPSGraphTensor* sum = nil;
    MPSGraphTensor* norm = nil;
};

// The saved-state slots (``BuilderContext::stash_saved``) under which a compiled quasi-norm
// hands ``d`` and ``s`` to its VJP, so the backward does not reduce ``x`` again.
inline constexpr const char* kQuasiNormDivisorSlot = "p_norm_quasi_divisor";
inline constexpr const char* kQuasiNormSumSlot = "p_norm_quasi_sum";

// ``(sum |x|^p)^(1/p)`` of float32 ``x`` over ``axes`` for ``0 < p < 1``, reduced axes kept at
// size 1.  ``d`` is the largest magnitude as :func:`scaled_p_norm_keepdim` takes it (the
// clamp keeps a zero slice's 0 / 0 and an infinity's inf / inf out of the ratio), every term
// is ``(|x| / d)^p`` in [0, 1] with the largest entry's exactly 1, and the root is
// ``(d h) h`` for ``h = s^(1/(2p))``: ``s^(1/p)`` alone leaves float32 where the norm does not.
[[maybe_unused]] inline QuasiNormParts
quasi_p_norm_keepdim(MPSGraph* g, MPSGraphTensor* x, NSArray<NSNumber*>* axes, double p) {
    const MPSDataType dt = x.dataType;
    MPSGraphTensor* a = [g absoluteWithTensor:x name:nil];
    MPSGraphTensor* top = [g reductionMaximumWithTensor:a axes:axes name:@"p_norm_top"];
    QuasiNormParts out;
    out.divisor =
        [g clampWithTensor:top
            minValueTensor:[g constantWithScalar:std::numeric_limits<float>::min() dataType:dt]
            maxValueTensor:[g constantWithScalar:std::numeric_limits<float>::max() dataType:dt]
                      name:@"p_norm_divisor"];
    MPSGraphTensor* ratio = [g divisionWithPrimaryTensor:a secondaryTensor:out.divisor name:nil];
    out.sum = [g reductionSumWithTensor:quasi_ratio_power(g, a, out.divisor, ratio, p, p < 0.5)
                                   axes:axes
                                   name:nil];
    // At p = 1/2 the half root is s itself.
    MPSGraphTensor* half = p == 0.5 ? out.sum
                                    : [g powerWithPrimaryTensor:out.sum
                                                secondaryTensor:[g constantWithScalar:0.5 / p
                                                                             dataType:dt]
                                                           name:nil];
    out.norm = [g multiplicationWithPrimaryTensor:[g multiplicationWithPrimaryTensor:out.divisor
                                                                     secondaryTensor:half
                                                                                name:nil]
                                  secondaryTensor:half
                                             name:@"p_norm"];
    return out;
}

// The engine's ``PNormBackward`` of an order below 1 (``quasi_grad``, ``ops/linalg/Norm.cpp``):
//
//     sign(x) (|x| / N)^(p-1) dN  =  sign(x) (|x| / d)^(p-1) s^((1-p)/p) dN
//
// with ``N = d s^(1/p)``, 0 at a zero entry (the power's slope is infinite there; the
// reference's gradient is 0) and where ``|x|`` is below float32's smallest normal (the
// arithmetic of the stream reads it as 0).  Neither factor underflows, since ``|x| / d``
// is at most 1 and ``s`` at least 1 where the slope is live, so the slope is as range-safe
// as the forward.  Where eager forms ``N / d`` for the second factor, which is ``s^(1/p)``,
// past float32 for 1e4 equal entries at ``p = 0.1`` though the gradient is an ordinary
// 1e36, this takes ``s^((1-p)/p)`` itself, in two halves with the cotangent
// (``(dN h) h``, as the norm's root is) before it meets the slope.  A NaN slice keeps its
// NaN, and an all-zero slice is 0.
//
// ``x``, ``parts`` (:func:`quasi_p_norm_keepdim`'s, of ``x``) and ``go`` (the cotangent of
// the norm, keepdim-shaped) are all float32.
[[maybe_unused]] inline MPSGraphTensor* quasi_p_norm_grad(
    MPSGraph* g, MPSGraphTensor* x, const QuasiNormParts& parts, MPSGraphTensor* go, double p) {
    const MPSDataType dt = x.dataType;
    MPSGraphTensor* a = [g absoluteWithTensor:x name:nil];
    MPSGraphTensor* ratio = [g divisionWithPrimaryTensor:a secondaryTensor:parts.divisor name:nil];
    MPSGraphTensor* slope = quasi_ratio_power(g, a, parts.divisor, ratio, p - 1.0, true);
    MPSGraphTensor* floor = [g constantWithScalar:std::numeric_limits<float>::min() dataType:dt];
    MPSGraphTensor* live_slope = [g selectWithPredicateTensor:[g lessThanWithPrimaryTensor:a
                                                                           secondaryTensor:floor
                                                                                      name:nil]
                                          truePredicateTensor:[g constantWithScalar:0.0 dataType:dt]
                                         falsePredicateTensor:slope
                                                         name:nil];
    // dN s^((1-p)/p); at p = 1/2 the exponent is 1.
    MPSGraphTensor* scale = nil;
    if (p == 0.5) {
        scale = [g multiplicationWithPrimaryTensor:go secondaryTensor:parts.sum name:nil];
    } else {
        MPSGraphTensor* half = [g powerWithPrimaryTensor:parts.sum
                                         secondaryTensor:[g constantWithScalar:(1.0 - p) / (2.0 * p)
                                                                      dataType:dt]
                                                    name:nil];
        scale = [g multiplicationWithPrimaryTensor:[g multiplicationWithPrimaryTensor:go
                                                                      secondaryTensor:half
                                                                                 name:nil]
                                   secondaryTensor:half
                                              name:nil];
    }
    return [g multiplicationWithPrimaryTensor:[g signWithTensor:x name:nil]
                              secondaryTensor:[g multiplicationWithPrimaryTensor:live_slope
                                                                 secondaryTensor:scale
                                                                            name:nil]
                                         name:@"p_norm_quasi_vjp"];
}

// What a recorded ``p_norm`` node says: the order and the axes it reduced.
struct PNormSpec {
    double p = 2.0;
    std::vector<bool> reduced;
};

// ``node``'s order and reduced axes over an input of ``x_shape`` (no ``axis_list``
// = every axis); nullopt for what the lowering does not serve — an order that is
// not finite and positive, or is 1 (a sum of magnitudes powers nothing), a 0-d
// input, an axis outside the rank.
[[maybe_unused]] inline std::optional<PNormSpec>
p_norm_spec(const OpNode& node, const std::vector<std::int64_t>& x_shape) {
    const double p = double_attr(node, "p", 0.0);
    if (!(p > 0.0) || p == 1.0 || std::isinf(p) || x_shape.empty())
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
