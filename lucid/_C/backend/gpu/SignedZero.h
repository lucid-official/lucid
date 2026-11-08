// lucid/_C/backend/gpu/SignedZero.h
//
// The sign of a zero extremum in element-wise GPU ops.
//
// ``max`` of a signed-zero tie is +0 and ``min`` is -0, whichever operand came
// first (the reference's vectorised kernels, and the CPU stream).  MLX hands a
// tie to its second operand, so ``maximum([-0], [0])`` was ``+0`` but
// ``maximum([0], [-0])`` was ``-0``.  ``a + b`` of two zeros is -0 only when
// both are, which is the max rule; ``-((-a) + (-b))`` is the min rule.  Compiled
// with the op, the whole ranking is one kernel (+1..6% at 16M).
//
// Not applied to the reductions (LCD-879): an exact form needs a second pass over
// the input (the minimum / maximum of its unsigned bit view), +47..77% on every
// Metal max / amax / amin, so a Metal (and compiled) reduction keeps MLX's /
// MPSGraph's zero order and is a documented difference.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <mlx/array.h>
#include <mlx/compile.h>
#include <mlx/ops.h>

namespace lucid {
namespace gpu {

// float16, bfloat16 and float32 are the floating dtypes the GPU stream holds
// (it has no float64); each has a same-width unsigned integer to view it as.
inline bool has_signed_zero(::mlx::core::Dtype dt) {
    return dt == ::mlx::core::float16 || dt == ::mlx::core::bfloat16 || dt == ::mlx::core::float32;
}

// ``m`` = the plain element-wise max of ``a`` and ``b`` (NaN-propagating), with
// a tie of two zeros given the sign the rule asks for.
inline ::mlx::core::array rank_zero_pair(const ::mlx::core::array& a,
                                         const ::mlx::core::array& b,
                                         const ::mlx::core::array& m,
                                         bool is_max) {
    namespace mx = ::mlx::core;
    const mx::array zero(0.0, m.dtype());
    const mx::array tied_zero =
        is_max ? mx::add(a, b) : mx::negative(mx::add(mx::negative(a), mx::negative(b)));
    return mx::where(mx::equal(a, zero), mx::where(mx::equal(a, b), tied_zero, m), m);
}

// ``minimum(maximum(x, lo), hi)`` with each tie of zeros ranked (SignedZero.h),
// one fused kernel, for the clamps whose zero bound is the wrong zero (below).
inline ::mlx::core::array clip_ranked(const ::mlx::core::array& x,
                                      const ::mlx::core::array& lo,
                                      const ::mlx::core::array& hi) {
    namespace mx = ::mlx::core;
    static const std::function<std::vector<mx::array>(const std::vector<mx::array>&)> fused =
        mx::compile(
            [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
                const mx::array m =
                    rank_zero_pair(in[0], in[1], mx::maximum(in[0], in[1]), /*is_max=*/true);
                return {rank_zero_pair(m, in[2], mx::minimum(m, in[2]), /*is_max=*/false)};
            },
            /*shapeless=*/true);
    return fused({x, lo, hi})[0];
}

}  // namespace gpu
}  // namespace lucid
