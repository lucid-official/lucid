// lucid/_C/backend/gpu/PNorm.h
//
// The scaled p-norm reduction of the GPU stream: ``(sum |x|^p)^(1/p)`` for a
// finite p > 1, taken relative to each slice's largest magnitude so no power
// leaves the float32 range.
//
// ``sum(x * x)`` squares a magnitude below ~1e-19 into the subnormals, which
// Metal flushes to zero (the norm of ``full(4, 1e-20)`` came back 0), and one
// above ~1.8e19 into infinity — while the norm itself, 2e-20 or 2e25, is an
// ordinary float.  BLAS's ``nrm2`` has scaled by the largest magnitude for the
// same reason since the seventies, and this is that algorithm as Metal kernels.
//
// A simdgroup takes a chunk of one slice: it reads the chunk twice — the second
// read is served from cache — first for the largest magnitude m (``simd_max``),
// then for s = sum (|x| / m)^p (``simd_sum``).  Every ratio is in [0, 1], so
// nothing overflows, and one underflows only when it is negligible next to the 1
// the largest entry contributes.  A slice too long for one simdgroup is split into
// chunks, each writing its pair (m, s), and a second launch folds a slice's pairs
// by rescaling (m_a / m)^p.  The answer is m * s^(1/p).  ``sum(x * x)`` writes a
// temporary the size of the input and reads it back, where this kernel only reads
// the input, so it is not slower than the plain reduction (the benchmark is in the
// LCD-944 notes).
//
// A NaN anywhere in a slice is the answer, an infinity is the largest magnitude
// (the answer is inf unless a NaN is there), and an all-zero slice has m = 0 and
// answers 0.
//
// Float32 arithmetic on Metal flushes subnormal operands, so a magnitude below
// float32's smallest normal (1.18e-38) reads as zero here as it does in every
// other op of the stream; the kernel is exact down to that floor.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/ops.h>

#include "../../core/ErrorBuilder.h"
#include "Pow.h"

namespace lucid {
namespace backend {
namespace pnorm {

// A row of at most this many entries is one thread's: fewer lanes than that
// would idle in a simdgroup.
inline constexpr int kRowThreadMax = 64;
// Entries a simdgroup takes from a long row before another simdgroup shares it.
inline constexpr int kChunk = 512;
// The most simdgroups sharing one row.
inline constexpr int kMaxChunks = 1024;
// The most (row, chunk) pairs a launch holds: the grid is 32 threads each.
inline constexpr std::int64_t kMaxItems = 1 << 22;
// The most rows the simdgroup kernels take; the grid is a 32-bit count of threads.
inline constexpr std::int64_t kMaxSimdRows = 1 << 24;

// Metal helpers: the ratio power, the root, and the fold of two (m, s) pairs.
// ``p`` is the order; the ``two`` forms avoid ``pow``.
inline const std::string& header() {
    static const std::string h = "#define PN_CHUNK " + std::to_string(kChunk) +
                                 "u\n#define PN_MAX_CHUNKS " + std::to_string(kMaxChunks) +
                                 "u\n#define PN_MAX_ITEMS " + std::to_string(kMaxItems) + "u\n" +
                                 gpu_pow_detail::header_src() + R"(
// ``r`` in [0, 1] to the power ``p``.  The exponents with their own exact
// operation (2, 3, ...) take it; any other is ``metal::pow``, whose 3 ulp is on
// terms that are below the largest entry's 1 and that average out in the sum
// (measured: the norm is within the plain reduction's own error).
inline float pn_pow(float r, float p) {
    return lucid_pow_is_special(p) ? lucid_pow(r, p) : metal::pow(r, p);
}
inline float pn_ratio(float a, float m) {
    return (a == m) ? 1.0f : metal::precise::divide(a, m);
}
inline float pn_root(float m, float s, float p, bool two) {
    return m * (two ? metal::sqrt(s) : lucid_pow(s, 1.0f / p));
}
inline void pn_merge(thread float& m, thread float& s, float m2, float s2, float p, bool two) {
    if (m != m || m2 != m2) { m = NAN; s = NAN; return; }
    float big = metal::max(m, m2);
    if (!(big > 0.0f)) { m = 0.0f; s = 0.0f; return; }
    float r1 = pn_ratio(m, big);
    float r2 = pn_ratio(m2, big);
    if (two) { r1 *= r1; r2 *= r2; } else { r1 = pn_pow(r1, p); r2 = pn_pow(r2, p); }
    s = s * r1 + s2 * r2;
    m = big;
}
)";
    return h;
}

// One thread per row: the largest magnitude, then the sum of the ratios.
inline const ::mlx::core::fast::CustomKernelFunction& row_thread_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel("lucid_pnorm_row_thread", {"x", "pw"}, {"out"}, R"(
    uint row = thread_position_in_grid.x;
    uint B = uint(x_shape[0]), N = uint(x_shape[1]);
    if (row >= B) return;
    auto xr = x + size_t(row) * N;
    float p = pw[0];
    float m = 0.0f;
    bool nan_seen = false;
    for (uint i = 0; i < N; ++i) {
        float a = metal::abs(xr[i]);
        nan_seen = nan_seen || (a != a);
        m = metal::max(m, a);
    }
    float s = 0.0f;
    if (m > 0.0f && !nan_seen) {
        for (uint i = 0; i < N; ++i) {
            float r = pn_ratio(metal::abs(xr[i]), m);
            s += Two ? r * r : pn_pow(r, p);
        }
    }
    if (nan_seen) { m = NAN; s = NAN; }
    out[row] = pn_root(m, s, p, Two);
)",
                                                 header(), true, false, mx::CompileOptions{});
    return k;
}

// One simdgroup per (row, chunk).  With one chunk per row (``Single``) it
// writes the answer; otherwise the pair (m, s), for ``combine_kernel``.
inline const ::mlx::core::fast::CustomKernelFunction& simd_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel("lucid_pnorm_simd", {"x", "pw"}, {"out"}, R"(
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    uint B = uint(x_shape[0]), N = uint(x_shape[1]);
    uint G = metal::min(PN_MAX_CHUNKS, (N + PN_CHUNK - 1u) / PN_CHUNK);
    G = metal::max(1u, metal::min(G, PN_MAX_ITEMS / B));
    uint w = threadgroup_position_in_grid.x * 8u + sg;
    if (w >= B * G) return;
    uint row = w / G, chunk = w % G;
    uint per = (N + G - 1u) / G;
    uint lo = chunk * per, hi = metal::min(N, lo + per);
    auto xr = x + size_t(row) * N;
    float p = pw[0];
    float m = 0.0f;
    bool nan_seen = false;
    for (uint i = lo + lane; i < hi; i += 32u) {
        float a = metal::abs(xr[i]);
        nan_seen = nan_seen || (a != a);
        m = metal::max(m, a);
    }
    m = metal::simd_max(m);
    nan_seen = metal::simd_any(nan_seen);
    float s = 0.0f;
    if (m > 0.0f && !nan_seen) {
        for (uint i = lo + lane; i < hi; i += 32u) {
            float r = pn_ratio(metal::abs(xr[i]), m);
            s += Two ? r * r : pn_pow(r, p);
        }
        s = metal::simd_sum(s);
    }
    if (nan_seen) { m = NAN; s = NAN; }
    if (lane == 0u) {
        if (Single) { out[row] = pn_root(m, s, p, Two); }
        else { out[w * 2u] = m; out[w * 2u + 1u] = s; }
    }
)",
                                                 header(), true, false, mx::CompileOptions{});
    return k;
}

// One simdgroup per row folds that row's chunk pairs into the answer.
inline const ::mlx::core::fast::CustomKernelFunction& combine_kernel() {
    namespace mx = ::mlx::core;
    static const auto k =
        mx::fast::metal_kernel("lucid_pnorm_combine", {"part", "pw", "meta"}, {"out"}, R"(
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    uint G = uint(meta[0]), B = uint(meta[1]);
    uint row = threadgroup_position_in_grid.x * 8u + sg;
    if (row >= B) return;
    float p = pw[0];
    float m = 0.0f, s = 0.0f;
    for (uint g = lane; g < G; g += 32u) {
        uint at = (uint(row) * G + g) * 2u;
        pn_merge(m, s, part[at], part[at + 1u], p, Two);
    }
    for (uint off = 16u; off > 0u; off >>= 1) {
        float m2 = metal::simd_shuffle_down(m, off);
        float s2 = metal::simd_shuffle_down(s, off);
        pn_merge(m, s, m2, s2, p, Two);
    }
    if (lane == 0u) out[row] = pn_root(m, s, p, Two);
)",
                               header(), true, false, mx::CompileOptions{});
    return k;
}

// Chunks per row, as the simdgroup kernel counts them itself.
inline int chunks_per_row(std::int64_t rows, std::int64_t n) {
    std::int64_t g = std::min<std::int64_t>(kMaxChunks, (n + kChunk - 1) / kChunk);
    g = std::min<std::int64_t>(g, kMaxItems / rows);
    return static_cast<int>(std::max<std::int64_t>(1, g));
}

// ``(sum |x|^p)^(1/p)`` of each row of the (rows, n) float32 matrix ``x``,
// a (rows,) float32 array.  ``p`` > 1; ``rows`` and ``n`` are not 0.
inline ::mlx::core::array
scaled_rows(const ::mlx::core::array& x, std::int64_t rows, std::int64_t n, double p) {
    namespace mx = ::mlx::core;
    const int b = static_cast<int>(rows);
    const bool two = p == 2.0;
    const float p_value = static_cast<float>(p);
    const mx::array pw(&p_value, {1}, mx::float32);
    const std::vector<std::pair<std::string, mx::fast::TemplateArg>> two_arg{{"Two", two}};
    if (n <= kRowThreadMax || rows >= kMaxSimdRows) {
        return row_thread_kernel()({x, pw}, {{b}}, {mx::float32}, {b, 1, 1}, {256, 1, 1}, two_arg,
                                   std::nullopt, false, {})[0];
    }
    const int g = chunks_per_row(rows, n);
    const int items = b * g;
    const int tg_count = (items + 7) / 8;
    if (g == 1) {
        return simd_kernel()({x, pw}, {{b}}, {mx::float32}, {tg_count * 256, 1, 1}, {256, 1, 1},
                             {{"Two", two}, {"Single", true}}, std::nullopt, false, {})[0];
    }
    auto part =
        simd_kernel()({x, pw}, {{items * 2}}, {mx::float32}, {tg_count * 256, 1, 1}, {256, 1, 1},
                      {{"Two", two}, {"Single", false}}, std::nullopt, false, {})[0];
    const std::vector<int> meta_v{g, b};
    const mx::array meta(meta_v.data(), {2}, mx::int32);
    return combine_kernel()({part, pw, meta}, {{b}}, {mx::float32}, {((b + 7) / 8) * 256, 1, 1},
                            {256, 1, 1}, two_arg, std::nullopt, false, {})[0];
}

// :func:`scaled_rows` over the ``axes`` of a float32 ``x``, with the reduced
// axes removed or kept as size 1.  ``axes`` are distinct and in range.
inline ::mlx::core::array
scaled_p_norm(const ::mlx::core::array& x, double p, const std::vector<int>& axes, bool keepdims) {
    namespace mx = ::mlx::core;
    const int ndim = static_cast<int>(x.ndim());
    std::vector<bool> reduced(static_cast<std::size_t>(ndim), false);
    for (int a : axes)
        reduced[static_cast<std::size_t>(a)] = true;
    std::vector<int> perm;
    mx::Shape out_shape;
    std::int64_t rows = 1;
    std::int64_t cols = 1;
    for (int i = 0; i < ndim; ++i) {
        if (!reduced[static_cast<std::size_t>(i)]) {
            perm.push_back(i);
            out_shape.push_back(x.shape(i));
            rows *= x.shape(i);
        } else if (keepdims) {
            out_shape.push_back(1);
        }
    }
    for (int i = 0; i < ndim; ++i) {
        if (reduced[static_cast<std::size_t>(i)]) {
            perm.push_back(i);
            cols *= x.shape(i);
        }
    }
    if (rows == 0 || cols == 0)
        return mx::zeros(out_shape, mx::float32);
    // The matrix is an MLX shape, whose dimensions are 32-bit.
    constexpr std::int64_t kMaxDim = std::numeric_limits<std::int32_t>::max();
    if (rows > kMaxDim || cols > kMaxDim)
        ErrorBuilder("p_norm").not_implemented(
            "the scaled p-norm takes at most 2^31 - 1 rows of at most 2^31 - 1 entries");
    const bool in_place = std::is_sorted(perm.begin(), perm.end());
    auto flat = mx::reshape(in_place ? x : mx::transpose(x, perm),
                            {static_cast<int>(rows), static_cast<int>(cols)});
    return mx::reshape(scaled_rows(flat, rows, cols, p), out_shape);
}

}  // namespace pnorm
}  // namespace backend
}  // namespace lucid
