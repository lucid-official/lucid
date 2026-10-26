// lucid/_C/backend/gpu/HalfAccumulation.h
//
// Float32 accumulation for float16 / bfloat16 reductions and normalisation
// layers on the GPU stream via custom Metal kernels.
//
// MLX accumulates its built-in sum, mean, var, prod, and scan primitives in
// their output dtype.  Rather than widening half inputs to a temporary float32
// buffer (moving 10 bytes per element instead of 2), sum_widened,
// mean_widened, var_widened, and the BatchNorm / GroupNorm / LayerNorm
// statistics use custom ``mlx::core::fast::metal_kernel`` shaders that read
// float16 / bfloat16 at native width, accumulate in float32 registers and
// threadgroup memory, and round once into the output dtype.
//
// The contract every plan keeps: a half reduction is the float32 reduction
// of the widened input, rounded once.  A sum or mean kernel may differ from
// MLX's float32 one only in summation order; a variance kernel is a different
// algorithm — shifted sums per group of threads, merged by moments (see
// ``kMomentsMetal``) — with its own error bound.  GPU float32 arithmetic
// flushes subnormals, so a bfloat16 subnormal result of a sum of two or more
// elements, or of any mean or variance, is a signed zero, as it is for float32
// and in ``lucid.compile``'s graphs (LCD-452).  A float16 subnormal is a
// normal float32 and survives everywhere.
//
// One deliberate departure from the reference's GPU stream, which flushes a
// one-element float32 / bfloat16 sum too: here that sum is a copy and keeps a
// subnormal, bit-identical to MLX's float32 sum and to the compiled graph.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <mlx/array.h>
#include <mlx/compile.h>
#include <mlx/fast.h>
#include <mlx/ops.h>

#include "ComplexParts.h"
#include "MlxBridge.h"

namespace lucid {
namespace backend {

// True for float16 and bfloat16.
inline bool is_half(::mlx::core::Dtype dt) {
    return dt == ::mlx::core::float16 || dt == ::mlx::core::bfloat16;
}

// ``x`` in float32 when it is float16 or bfloat16, else ``x`` itself.
inline ::mlx::core::array widen_half(const ::mlx::core::array& x) {
    return is_half(x.dtype()) ? ::mlx::core::astype(x, ::mlx::core::float32) : x;
}

// ``x`` in ``dt`` — a no-op when it already is.
inline ::mlx::core::array narrow_to(const ::mlx::core::array& x, ::mlx::core::Dtype dt) {
    return x.dtype() == dt ? x : ::mlx::core::astype(x, dt);
}

inline ::mlx::core::array widen_gpu(const Storage& s) {
    return widen_half(*std::get<GpuStorage>(s).arr);
}

inline Storage wrap_narrow(const ::mlx::core::array& a, Dtype dt) {
    return Storage{::lucid::gpu::wrap_mlx_array(narrow_to(a, ::lucid::gpu::to_mlx_dtype(dt)), dt)};
}

// The dtype a reduction over ``dt`` accumulates in.
inline ::mlx::core::Dtype accumulation_dtype(::mlx::core::Dtype dt) {
    return is_half(dt) ? ::mlx::core::float32 : dt;
}

// A result computed from ``widen_half`` of an ``in_dt`` input, back in
// ``in_dt`` when that was a half dtype.
inline ::mlx::core::array narrow_half(const ::mlx::core::array& r, ::mlx::core::Dtype in_dt) {
    return is_half(in_dt) ? narrow_to(r, in_dt) : r;
}

// ``reduce(x)`` with a half ``x`` accumulated in float32, returned in
// ``x``'s dtype.
template <class Reduce>
::mlx::core::array reduce_widened(const ::mlx::core::array& x, Reduce&& reduce) {
    return narrow_half(reduce(widen_half(x)), x.dtype());
}

inline ::mlx::core::array reshape_contig(const ::mlx::core::array& x,
                                         const ::mlx::core::Shape& shape) {
    namespace mx = ::mlx::core;
    return mx::reshape(x.flags().row_contiguous ? x : mx::contiguous(x), shape);
}

struct ReducePlan {
    enum class Kind { None, Identity, Omi, Krk };
    Kind kind = Kind::None;
    std::int64_t O = 1;
    std::int64_t M = 1;
    std::int64_t I = 1;
    ::mlx::core::Shape out_shape;
};

struct ReduceSeg {
    bool red;
    std::int64_t ext;
};

inline void assign_reduce_plan_2(ReducePlan& plan, const std::vector<ReduceSeg>& segs) {
    if (!segs[0].red) {
        plan.M = segs[0].ext;
        plan.I = segs[1].ext;
        plan.kind = (plan.I == 1) ? ReducePlan::Kind::Identity : ReducePlan::Kind::Omi;
        return;
    }
    plan.M = (segs[1].ext == 1) ? 1 : segs[0].ext;
    plan.I = (segs[1].ext == 1) ? segs[0].ext : segs[1].ext;
    plan.kind = (segs[0].ext == 1)
                    ? ReducePlan::Kind::Identity
                    : ((segs[1].ext == 1) ? ReducePlan::Kind::Omi : ReducePlan::Kind::Krk);
}

inline void assign_reduce_plan_3(ReducePlan& plan, const std::vector<ReduceSeg>& segs) {
    const bool tail1 = segs[2].ext == 1;
    plan.O = tail1 ? 1 : segs[0].ext;
    plan.M = tail1 ? segs[0].ext : segs[1].ext;
    plan.I = tail1 ? segs[1].ext : segs[2].ext;
    if (segs[0].red) {
        plan.kind = (segs[0].ext * segs[2].ext == 1)
                        ? ReducePlan::Kind::Identity
                        : (tail1 ? ReducePlan::Kind::Krk : ReducePlan::Kind::Omi);
        return;
    }
    plan.kind = (segs[1].ext == 1) ? ReducePlan::Kind::Identity
                                   : (tail1 ? ReducePlan::Kind::Omi : ReducePlan::Kind::Krk);
}

inline void assign_reduce_plan(ReducePlan& plan, const std::vector<ReduceSeg>& segs) {
    const auto sz = segs.size();
    if (sz == 1 && segs[0].red) {
        plan.I = segs[0].ext;
        plan.kind = (plan.I == 1) ? ReducePlan::Kind::Identity : ReducePlan::Kind::Omi;
    } else if (sz == 2) {
        assign_reduce_plan_2(plan, segs);
    } else if (sz == 3) {
        assign_reduce_plan_3(plan, segs);
    }
}

inline bool normalize_reduce_axes(int nd, std::vector<int>& axes) {
    if (nd == 0 || axes.empty())
        return false;
    for (int& ax : axes)
        ax = ax < 0 ? ax + nd : ax;
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    return axes.front() >= 0 && axes.back() < nd;
}

inline ReducePlan
classify_reduction(const ::mlx::core::array& x, std::vector<int> axes, bool keepdims) {
    ReducePlan plan;
    const int nd = static_cast<int>(x.ndim());
    if (x.size() == 0 || !normalize_reduce_axes(nd, axes))
        return plan;

    std::vector<ReduceSeg> segs;
    for (int d = 0; d < nd; ++d) {
        const bool red = std::binary_search(axes.begin(), axes.end(), d);
        const auto ext = static_cast<std::int64_t>(x.shape(d));
        if (!red || keepdims)
            plan.out_shape.push_back(red ? 1 : x.shape(d));
        if (!segs.empty() && segs.back().red == red)
            segs.back().ext *= ext;
        else
            segs.push_back({red, ext});
    }
    assign_reduce_plan(plan, segs);
    return plan;
}

inline int pick_omi_groups(std::int64_t O, std::int64_t M, std::int64_t I) {
    if (M >= 512 || O * I <= 1024)
        return 1;
    const int target = std::min<int>(256, std::max<int>(1, static_cast<int>(1024 / M)));
    if (target <= O)
        return target;
    const int max_i = static_cast<int>(std::max<std::int64_t>(1, (I + 1023) / 1024));
    const int g_per_o = std::min<int>(max_i, std::max<int>(1, static_cast<int>(target / O)));
    return static_cast<int>(O) * g_per_o;
}

inline const ::mlx::core::fast::CustomKernelFunction& row_sum_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel(
        "lucid_half_row_sum", {"x"}, {"out"},
        "uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;"
        "uint m=threadgroup_position_in_grid.x*8u+sg,M=uint(x_shape[1]),I=uint(x_shape[2]);"
        "if(m>=M)return;uint base=m*I;float acc=0.0f;"
        "for(uint i=lane;i<I;i+=32u){acc+=float(x[base+i]);}"
        "acc=simd_sum(acc);"
        "if(lane==0u){if(ScaleMean)acc/=float(I);out[m]=OutT(acc);}",
        "", true, false, mx::CompileOptions{});
    return k;
}

inline const ::mlx::core::fast::CustomKernelFunction& omi_sum_s1_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel(
        "lucid_half_omi_sum_s1", {"x"}, {"out"},
        "uint tid=thread_position_in_threadgroup.x,lane=thread_index_in_simdgroup;"
        "uint sg=simdgroup_index_in_threadgroup,g=threadgroup_position_in_grid.x;"
        "uint m=threadgroup_position_in_grid.y,G=threadgroups_per_grid.x;"
        "uint O=uint(x_shape[0]),M=uint(x_shape[1]),I=uint(x_shape[2]);float acc=0.0f;"
        "if(G<=O){for(uint o=g;o<O;o+=G){uint base=(o*M+m)*I;"
        "for(uint i=tid;i<I;i+=256u){acc+=float(x[base+i]);}}}"
        "else{uint gpo=G/O,o=g/gpo,gi=g%gpo;if(o<O){uint base=(o*M+m)*I,step=gpo<<8;"
        "for(uint i=(gi<<8)+tid;i<I;i+=step){acc+=float(x[base+i]);}}}"
        "acc=simd_sum(acc);threadgroup float sh[8];if(lane==0u)sh[sg]=acc;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "if(sg==0u){float v=(lane<8u)?sh[lane]:0.0f;v=simd_sum(v);"
        "if(lane==0u){if(ScaleMean)v/=(float(O)*float(I));out[m*G+g]=OutT(v);}}",
        "", true, false, mx::CompileOptions{});
    return k;
}

inline const ::mlx::core::fast::CustomKernelFunction& omi_sum_s2_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel(
        "lucid_half_omi_sum_s2", {"p", "x"}, {"out"},
        "uint tid=thread_position_in_threadgroup.x,lane=thread_index_in_simdgroup;"
        "uint sg=simdgroup_index_in_threadgroup,m=threadgroup_position_in_grid.y;"
        "uint G=uint(p_shape[1]),O=uint(x_shape[0]),I=uint(x_shape[2]),base=m*G;float acc=0.0f;"
        "for(uint g=tid;g<G;g+=256u){acc+=p[base+g];}"
        "acc=simd_sum(acc);threadgroup float sh[8];if(lane==0u)sh[sg]=acc;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "if(sg==0u){float v=(lane<8u)?sh[lane]:0.0f;v=simd_sum(v);"
        "if(lane==0u){if(ScaleMean)v/=(float(O)*float(I));out[m]=OutT(v);}}",
        "", true, false, mx::CompileOptions{});
    return k;
}

inline const ::mlx::core::fast::CustomKernelFunction& krk_sum_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel(
        "lucid_half_krk_sum", {"x"}, {"out"},
        "uint tid=thread_position_in_threadgroup.x,tx=tid&7u,ty=tid>>3;"
        "uint gm=threadgroup_position_in_grid.x,bo=threadgroup_position_in_grid.y;"
        "uint R=uint(x_shape[1]),K=uint(x_shape[2]),col=(gm<<3)+tx;float acc=0.0f;"
        "if(col<K){uint base=bo*R*K+col;for(uint r=ty;r<R;r+=32u){acc+=float(x[base+r*K]);}}"
        "threadgroup float sh[32][9];sh[ty][tx]=acc;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "if(ty<8u){sh[ty][tx]=sh[ty][tx]+sh[ty+8u][tx]+sh[ty+16u][tx]+sh[ty+24u][tx];}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "if(ty==0u&&col<K){float v=(sh[0][tx]+sh[1][tx])+(sh[2][tx]+sh[3][tx])+"
        "(sh[4][tx]+sh[5][tx])+(sh[6][tx]+sh[7][tx]);"
        "if(ScaleMean)v/=float(R);out[bo*K+col]=OutT(v);}",
        "", true, false, mx::CompileOptions{});
    return k;
}

// Metal helpers of the variance and normalisation-statistics kernels.
//
// A group of threads sums the deviations ``d`` of its n values from one
// anchor a: s1 = sum(d), s2 = sum(d^2), and ``lucid_moments`` takes M2 = s2 -
// s1^2 / n.  The rounding of the sums is magnified by s2 / M2 = 1 + n (a -
// mean)^2 / M2, and if a is the mean of k of the group's values,
// Cauchy-Schwarz gives (a - mean)^2 <= M2 / k: the magnification is at most
// 1 + n / k whatever the data.  The rounding itself grows with the values
// one thread sums in a row, so the error is about eps * run * (1 + n / k).
// The anchor used to be the first element of the whole reduction, so an
// outlier there magnified it by the reduction's size.  Now:
//
//   * a row kernel simdgroup anchors at the mean of its row's first 32
//     values: n <= 4096, so at most 129;
//   * a split-kernel simdgroup anchors at the mean of its lanes' first
//     values: at most 1 + n / 32 over the simdgroup's n, an eighth of its
//     threadgroup's, on every route, the one-threadgroup route too;
//   * a column-kernel thread re-anchors at the first of every 64 of its
//     rows: at most 65, however many rows.
//
// The groups' moments (count, mean, M2) merge by Chan, Golub and LeVeque's
// update (``lucid_merge``) or about a pivot mean (``lucid_simd_combine``),
// adding no cancellation of their own.  A sample is scaled down before it is
// summed, so k of them cannot overflow.  The bandwidth-bound loops of the row
// and split kernels are the ones a single anchor needs (per-thread moments in
// them held more registers and cost occupancy: a variance of 32M float16
// values went from 437 to 612 us on an M1 Pro).  What the sampled anchor
// still costs is its wait: the loop starts once the lanes' first loads are in.
//
// ``lucid_nonneg`` clamps M2 at +0: ``s2 - s1 * md`` is contracted into an fma
// that can round a flushed square to -0, and a variance has no sign.
inline constexpr const char* kMomentsMetal = R"(
struct LucidMoments { float n; float mean; float m2; };

inline float lucid_nonneg(float m2) { return (m2 > 0.0f || isnan(m2)) ? m2 : 0.0f; }

// A non-finite first value would turn every deviation into NaN, while the
// mean of an input holding one infinity is that infinity.
inline float lucid_anchor(float v) { return isfinite(v) ? v : 0.0f; }

// The mean of a simdgroup's samples, ``k`` of whose lanes hold one.
inline float lucid_simd_anchor(bool has, float v, uint k) {
    float s = simd_sum(has ? lucid_anchor(v) * 0.03125f : 0.0f);
    return (k > 0u) ? s * (32.0f / float(k)) : 0.0f;
}

inline LucidMoments lucid_moments(float n, float anchor, float s1, float s2) {
    if (!(n > 0.0f)) return {0.0f, 0.0f, 0.0f};
    float md = s1 / n;
    return {n, anchor + md, lucid_nonneg(s2 - s1 * md)};
}

inline LucidMoments lucid_merge(LucidMoments a, LucidMoments b) {
    if (b.n == 0.0f) return a;
    if (a.n == 0.0f) return b;
    float n = a.n + b.n, wb = b.n / n, delta = b.mean - a.mean;
    float mean = isfinite(delta) ? metal::fma(delta, wb, a.mean)
                                 : a.mean * (a.n / n) + b.mean * wb;
    return {n, mean, a.m2 + b.m2 + delta * delta * a.n * wb};
}

// A simdgroup's moments.  M2 adds the spread of the lanes' means about lane
// 0's, the first group's, which always holds values; any pivot gives the
// same M2 in exact arithmetic, and one of the means keeps the correction
// small.  The mean is the plain weighted sum, which keeps an infinity.  The
// sums are independent and reduce side by side.
inline LucidMoments lucid_simd_combine(LucidMoments t) {
    bool has = t.n > 0.0f;
    float c = simd_broadcast_first(t.mean), dm = has ? t.mean - c : 0.0f;
    float n = simd_sum(t.n), total = simd_sum(has ? t.n * t.mean : 0.0f), sa = simd_sum(t.m2);
    float sb = simd_sum(t.n * dm * dm), sc = simd_sum(t.n * dm);
    float mean = (n > 0.0f) ? total / n : 0.0f;
    return {n, mean, sa + lucid_nonneg(sb - sc * ((n > 0.0f) ? sc / n : 0.0f))};
}

inline float lucid_variance(LucidMoments r, int ddof) {
    float denom = r.n - float(ddof);
    return denom > 0.0f ? r.m2 / denom : NAN;
}

// How many of [0, I) fall at offsets [lo, lo + 32) of the 256-wide blocks
// b with b % step == first (step = 1: every block).
inline uint lucid_lanes_count(uint I, uint lo, uint first, uint step) {
    uint full = I >> 8, rest = I & 255u;
    uint blocks = (full > first) ? (full - first - 1u) / step + 1u : 0u;
    uint tail = (rest > lo && full % step == first) ? min(rest - lo, 32u) : 0u;
    return blocks * 32u + tail;
}
)";

inline const ::mlx::core::fast::CustomKernelFunction& row_var_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel(
        "lucid_half_row_var", {"x"}, {"out"},
        "uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;"
        "uint m=threadgroup_position_in_grid.x*8u+sg,M=uint(x_shape[1]),I=uint(x_shape[2]);"
        "if(m>=M)return;uint base=m*I;"
        "float anchor=lucid_simd_anchor(lane<I,float(x[base+min(lane,I-1u)]),min(I,32u));"
        "float s1=0.0f,s2=0.0f;"
        "for(uint i=lane;i<I;i+=32u){float d=float(x[base+i])-anchor;s1+=d;s2=metal::fma(d,d,s2);}"
        "s1=simd_sum(s1);s2=simd_sum(s2);"
        "if(lane==0u){out[m]=OutT(lucid_variance(lucid_moments(float(I),anchor,s1,s2),Ddof));}",
        kMomentsMetal, true, false, mx::CompileOptions{});
    return k;
}

// Moments over (O, ., I) per middle index ``m``, as ``G`` partials per ``m``:
// ``p`` is (M, G, 3) of (count, mean, M2).  A threadgroup covers whole
// 256-wide blocks of a row — every block of ``ceil((O - g) / G)`` rows when
// G <= O, else the blocks of its one row congruent to ``g % (G / O)`` — so
// a simdgroup's count has a closed form and the loop counts nothing.
inline const ::mlx::core::fast::CustomKernelFunction& omi_stats_s1_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel(
        "lucid_half_omi_stats_s1", {"x"}, {"p"},
        "uint tid=thread_position_in_threadgroup.x,lane=thread_index_in_simdgroup;"
        "uint sg=simdgroup_index_in_threadgroup,g=threadgroup_position_in_grid.x;"
        "uint m=threadgroup_position_in_grid.y,G=threadgroups_per_grid.x;"
        "uint O=uint(x_shape[0]),M=uint(x_shape[1]),I=uint(x_shape[2]),lo=sg*32u,row,start,n;"
        "if(G<=O){row=g;start=tid;n=((O-g+G-1u)/G)*lucid_lanes_count(I,lo,0u,1u);}"
        "else{uint gpo=G/O;row=g/gpo;start=((g%gpo)<<8)+tid;"
        "n=lucid_lanes_count(I,lo,g%gpo,gpo);}"
        "uint s0=start-lane,k=(I>s0)?min(I-s0,32u):0u;"
        "float anchor=lucid_simd_anchor(start<I,float(x[(row*M+m)*I+min(start,I-1u)]),k);"
        "float s1=0.0f,s2=0.0f;"
        "if(G<=O){for(uint o=g;o<O;o+=G){uint base=(o*M+m)*I;"
        "for(uint i=tid;i<I;i+=256u){"
        "float d=float(x[base+i])-anchor;s1+=d;s2=metal::fma(d,d,s2);}}}"
        "else{uint gpo=G/O,base=(row*M+m)*I,step=gpo<<8;"
        "for(uint i=start;i<I;i+=step){"
        "float d=float(x[base+i])-anchor;s1+=d;s2=metal::fma(d,d,s2);}}"
        "s1=simd_sum(s1);s2=simd_sum(s2);"
        "threadgroup LucidMoments sh[8];if(lane==0u)sh[sg]=lucid_moments(float(n),anchor,s1,s2);"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "if(sg==0u){LucidMoments v=lucid_simd_combine((lane<8u)?sh[lane]:"
        "LucidMoments{0.0f,0.0f,0.0f});"
        "if(lane==0u){uint idx=(m*G+g)*3u;p[idx]=v.n;p[idx+1u]=v.mean;p[idx+2u]=v.m2;}}",
        kMomentsMetal, true, false, mx::CompileOptions{});
    return k;
}

// The merge of ``omi_stats_s1_kernel``'s partials ``p`` into one moment set
// ``v`` per ``m`` by a single simdgroup (launched as 32 threads per ``m``; one
// partial is the answer as it stands), valid in lane 0 — the body every kernel
// reading ``p`` starts with.  ``p`` is read inline: MLX binds a small input in
// the ``constant`` address space, a large one in ``device``.
inline constexpr const char* kMergePartialsMetal =
    "uint lane=thread_index_in_simdgroup,m=threadgroup_position_in_grid.y;"
    "uint G=uint(p_shape[1]);LucidMoments v={p[m*G*3u],p[m*G*3u+1u],p[m*G*3u+2u]};"
    "if(G>1u){LucidMoments a={0.0f,0.0f,0.0f};"
    "for(uint g=lane;g<G;g+=32u){uint i=(m*G+g)*3u;"
    "a=lucid_merge(a,LucidMoments{p[i],p[i+1u],p[i+2u]});}"
    "v=lucid_simd_combine(a);}";

inline const ::mlx::core::fast::CustomKernelFunction& omi_var_s2_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel(
        "lucid_half_omi_var_s2", {"p"}, {"out"},
        std::string(kMergePartialsMetal) + "if(lane==0u){out[m]=OutT(lucid_variance(v,Ddof));}",
        kMomentsMetal, true, false, mx::CompileOptions{});
    return k;
}

// One column per ``tx``, its rows split over the 32 ``ty``.  A thread sums
// its rows in runs of 64, each anchored at its first row, and merges the
// runs' moments.  The column then takes two passes over shared floats: the
// mean, then each thread's M2 plus n (its mean - the mean)^2.  Merging the
// threads' moments pairwise in shared memory instead cost 2.6x.
inline const ::mlx::core::fast::CustomKernelFunction& krk_var_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel(
        "lucid_half_krk_var", {"x"}, {"out"},
        "uint tid=thread_position_in_threadgroup.x,tx=tid&7u,ty=tid>>3;"
        "uint gm=threadgroup_position_in_grid.x,bo=threadgroup_position_in_grid.y;"
        "uint R=uint(x_shape[1]),K=uint(x_shape[2]),col=(gm<<3)+tx;"
        "LucidMoments t={0.0f,0.0f,0.0f};"
        "if(col<K){uint base=bo*R*K+col;"
        "for(uint r0=ty;r0<R;r0+=2048u){uint end=min(R,r0+2048u);"
        "float anchor=lucid_anchor(float(x[base+r0*K])),s1=0.0f,s2=0.0f;"
        "for(uint r=r0;r<end;r+=32u){float "
        "d=float(x[base+r*K])-anchor;s1+=d;s2=metal::fma(d,d,s2);}"
        "t=lucid_merge(t,lucid_moments(float((end-r0+31u)/32u),anchor,s1,s2));}}"
        "threadgroup float sh[32][9],mean_of[8];sh[ty][tx]=t.n*t.mean;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "if(ty<8u){sh[ty][tx]+=sh[ty+8u][tx]+sh[ty+16u][tx]+sh[ty+24u][tx];}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "if(ty==0u){mean_of[tx]=((sh[0][tx]+sh[1][tx])+(sh[2][tx]+sh[3][tx])+"
        "(sh[4][tx]+sh[5][tx])+(sh[6][tx]+sh[7][tx]))/float(R);}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "float "
        "mean=mean_of[tx],dm=t.mean-mean;sh[ty][tx]=(t.n>0.0f)?metal::fma(t.n*dm,dm,t.m2):0.0f;"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "if(ty<8u){sh[ty][tx]+=sh[ty+8u][tx]+sh[ty+16u][tx]+sh[ty+24u][tx];}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "if(ty==0u&&col<K){float m2=(sh[0][tx]+sh[1][tx])+(sh[2][tx]+sh[3][tx])+"
        "(sh[4][tx]+sh[5][tx])+(sh[6][tx]+sh[7][tx]);"
        "out[bo*K+col]=OutT(lucid_variance(LucidMoments{float(R),mean,m2},Ddof));}",
        kMomentsMetal, true, false, mx::CompileOptions{});
    return k;
}

inline ::mlx::core::array run_omi_sum(const ::mlx::core::array& x,
                                      std::int64_t O,
                                      std::int64_t M,
                                      std::int64_t I,
                                      bool scale_mean,
                                      ::mlx::core::Dtype out_dt) {
    namespace mx = ::mlx::core;
    auto x3 = reshape_contig(x, {static_cast<int>(O), static_cast<int>(M), static_cast<int>(I)});
    const int m_i = static_cast<int>(M);
    if (O == 1 && M >= 256 && I <= 4096) {
        const int tgs = (m_i + 7) / 8;
        return row_sum_kernel()({x3}, {{m_i}}, {out_dt}, {tgs * 256, 1, 1}, {256, 1, 1},
                                {{"T", x.dtype()}, {"OutT", out_dt}, {"ScaleMean", scale_mean}},
                                std::nullopt, false, {})[0];
    }
    const int G = pick_omi_groups(O, M, I);
    if (G == 1) {
        return omi_sum_s1_kernel()({x3}, {{m_i}}, {out_dt}, {256, m_i, 1}, {256, 1, 1},
                                   {{"T", x.dtype()}, {"OutT", out_dt}, {"ScaleMean", scale_mean}},
                                   std::nullopt, false, {})[0];
    }
    auto p = omi_sum_s1_kernel()({x3}, {{m_i, G}}, {mx::float32}, {G * 256, m_i, 1}, {256, 1, 1},
                                 {{"T", x.dtype()}, {"OutT", mx::float32}, {"ScaleMean", false}},
                                 std::nullopt, false, {})[0];
    return omi_sum_s2_kernel()({p, x3}, {{m_i}}, {out_dt}, {256, m_i, 1}, {256, 1, 1},
                               {{"OutT", out_dt}, {"ScaleMean", scale_mean}}, std::nullopt, false,
                               {})[0];
}

inline ::mlx::core::array
run_krk_kernel(const ::mlx::core::fast::CustomKernelFunction& fn,
               const ::mlx::core::array& x,
               const ReducePlan& plan,
               const std::vector<std::pair<std::string, ::mlx::core::fast::TemplateArg>>& targs,
               ::mlx::core::Dtype out_dt) {
    namespace mx = ::mlx::core;
    auto x3 = reshape_contig(
        x, {static_cast<int>(plan.O), static_cast<int>(plan.M), static_cast<int>(plan.I)});
    const int gm = static_cast<int>((plan.I + 7) / 8);
    const int bo = static_cast<int>(plan.O);
    const int m_out = static_cast<int>(plan.O * plan.I);
    auto out = fn({x3}, {{m_out}}, {out_dt}, {gm * 256, bo, 1}, {256, 1, 1}, targs, std::nullopt,
                  false, {})[0];
    return mx::reshape(out, plan.out_shape);
}

inline std::optional<ReducePlan>
plan_half_reduction(const ::mlx::core::array& x, const std::vector<int>& axes, bool keepdims) {
    if (!is_half(x.dtype()))
        return std::nullopt;
    auto plan = classify_reduction(x, axes, keepdims);
    if (plan.kind == ReducePlan::Kind::None)
        return std::nullopt;
    return plan;
}

// The one-element plan, element-wise in ``x``'s dtype — what the widened
// float32 op computes, without its float32 copy.  A sum is a copy.  A mean
// still multiplies by 1/1, which flushes a bfloat16 subnormal on the GPU (the
// ``x * 1`` that LCD-452 pins eager against compiled); a float16 subnormal is
// a normal float32, so a float16 mean is the element too.
inline ::mlx::core::array one_element_mean_or_sum(const ::mlx::core::array& x, bool scale_mean) {
    namespace mx = ::mlx::core;
    if (!scale_mean || x.dtype() == mx::float16)
        return x;
    return mx::multiply(x, mx::array(1.0f, x.dtype()));
}

// The one-element variance: ``x - x`` is +0, or NaN for an infinity or NaN,
// and flushes as the widened ``var`` does; with ``ddof >= 1`` the divisor
// ``1 - ddof`` leaves only NaN.
inline ::mlx::core::array
one_element_variance(const ::mlx::core::array& x, int ddof, ::mlx::core::Dtype out_dt) {
    namespace mx = ::mlx::core;
    if (ddof >= 1)
        return mx::full(x.shape(), std::numeric_limits<float>::quiet_NaN(), out_dt);
    auto diff = mx::subtract(x, x);
    return diff.dtype() == out_dt ? diff : mx::astype(diff, out_dt);
}

inline std::optional<::mlx::core::array> half_reduce_sum_or_mean(const ::mlx::core::array& x,
                                                                 const std::vector<int>& axes,
                                                                 bool keepdims,
                                                                 bool scale_mean) {
    namespace mx = ::mlx::core;
    auto maybe = plan_half_reduction(x, axes, keepdims);
    if (!maybe)
        return std::nullopt;
    const auto& plan = *maybe;
    if (plan.kind == ReducePlan::Kind::Identity)
        return mx::reshape(one_element_mean_or_sum(x, scale_mean), plan.out_shape);
    if (plan.kind == ReducePlan::Kind::Omi) {
        auto out = run_omi_sum(x, plan.O, plan.M, plan.I, scale_mean, x.dtype());
        return mx::reshape(out, plan.out_shape);
    }
    return run_krk_kernel(krk_sum_kernel(), x, plan,
                          {{"T", x.dtype()}, {"OutT", x.dtype()}, {"ScaleMean", scale_mean}},
                          x.dtype());
}

inline std::optional<::mlx::core::array> half_reduce_var(const ::mlx::core::array& x,
                                                         const std::vector<int>& axes,
                                                         bool keepdims,
                                                         int ddof,
                                                         ::mlx::core::Dtype out_dt) {
    namespace mx = ::mlx::core;
    auto maybe = plan_half_reduction(x, axes, keepdims);
    if (!maybe)
        return std::nullopt;
    const auto& plan = *maybe;
    if (plan.kind == ReducePlan::Kind::Identity)
        return mx::reshape(one_element_variance(x, ddof, out_dt), plan.out_shape);
    if (plan.kind == ReducePlan::Kind::Omi) {
        auto x3 = reshape_contig(
            x, {static_cast<int>(plan.O), static_cast<int>(plan.M), static_cast<int>(plan.I)});
        const int m_i = static_cast<int>(plan.M);
        if (plan.O == 1 && plan.M >= 256 && plan.I <= 4096) {
            const int tgs = (m_i + 7) / 8;
            auto out = row_var_kernel()({x3}, {{m_i}}, {out_dt}, {tgs * 256, 1, 1}, {256, 1, 1},
                                        {{"T", x.dtype()}, {"OutT", out_dt}, {"Ddof", ddof}},
                                        std::nullopt, false, {})[0];
            return mx::reshape(out, plan.out_shape);
        }
        const int G = pick_omi_groups(plan.O, plan.M, plan.I);
        auto p = omi_stats_s1_kernel()({x3}, {{m_i, G, 3}}, {mx::float32}, {G * 256, m_i, 1},
                                       {256, 1, 1}, {{"T", x.dtype()}}, std::nullopt, false, {})[0];
        auto out =
            omi_var_s2_kernel()({p}, {{m_i}}, {out_dt}, {32, m_i, 1}, {32, 1, 1},
                                {{"OutT", out_dt}, {"Ddof", ddof}}, std::nullopt, false, {})[0];
        return mx::reshape(out, plan.out_shape);
    }
    return run_krk_kernel(krk_var_kernel(), x, plan,
                          {{"T", x.dtype()}, {"OutT", out_dt}, {"Ddof", ddof}}, out_dt);
}

// ``sum`` accumulated in float32 for a half input, returned in its dtype.
inline ::mlx::core::array
sum_widened(const ::mlx::core::array& x, const std::vector<int>& axes, bool keepdims) {
    if (auto custom = half_reduce_sum_or_mean(x, axes, keepdims, false))
        return *std::move(custom);
    return reduce_widened(
        x, [&](const ::mlx::core::array& w) { return ::mlx::core::sum(w, axes, keepdims); });
}

// ``mean`` accumulated in float32 for a half input, returned in its dtype.
inline ::mlx::core::array
mean_widened(const ::mlx::core::array& x, const std::vector<int>& axes, bool keepdims) {
    if (auto custom = half_reduce_sum_or_mean(x, axes, keepdims, true))
        return *std::move(custom);
    return reduce_widened(
        x, [&](const ::mlx::core::array& w) { return ::mlx::core::mean(w, axes, keepdims); });
}

// ``var`` accumulated in float32 for a half input, returned in float32 so
// Bessel correction and ``std``'s square root round to half only once.
inline ::mlx::core::array var_widened(const ::mlx::core::array& x,
                                      const std::vector<int>& axes,
                                      bool keepdims,
                                      int ddof = 0) {
    if (auto custom = half_reduce_var(x, axes, keepdims, ddof, ::mlx::core::float32))
        return *std::move(custom);
    if (!is_half(x.dtype()))
        return ::mlx::core::var(x, axes, keepdims, ddof);
    return ::mlx::core::var(::mlx::core::astype(x, ::mlx::core::float32), axes, keepdims, ddof);
}

// Full-tensor sum in float32 without widening a half input first.
inline ::mlx::core::array sum_all_f32(const ::mlx::core::array& x) {
    namespace mx = ::mlx::core;
    if (!is_half(x.dtype()) || x.size() == 0)
        return mx::sum(widen_half(x), false);
    auto out = run_omi_sum(x, 1, 1, static_cast<std::int64_t>(x.size()), false, mx::float32);
    return mx::reshape(out, {});
}

// ── Normalisation kernels (BatchNorm / GroupNorm / LayerNorm) ────────────────

inline const ::mlx::core::fast::CustomKernelFunction& omi_norm_stats_s2_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel(
        "lucid_omi_norm_stats_s2", {"p", "eps"}, {"stats_f32", "saved_mean", "saved_rstd"},
        std::string(kMergePartialsMetal) +
            "if(lane==0u){uint M=uint(p_shape[0]);float "
            "rstd=metal::precise::divide(1.0f,metal::precise::sqrt(metal::precise::divide(v.m2,v.n)"
            "+eps[0]));"
            "stats_f32[m]=v.mean;stats_f32[M+m]=rstd;"
            "saved_mean[m]=OutT(v.mean);saved_rstd[m]=OutT(rstd);}",
        kMomentsMetal, true, false, mx::CompileOptions{});
    return k;
}

inline const ::mlx::core::fast::CustomKernelFunction& bn_affine_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel(
        "lucid_bn_affine", {"x", "stats_f32", "gamma", "beta"}, {"y"},
        "uint tid=thread_position_in_grid.x,c=threadgroup_position_in_grid.y;"
        "uint b=threadgroup_position_in_grid.z,C=uint(x_shape[1]),S=uint(x_shape[2]);"
        "uint base=(b*C+c)*S,i=tid*4u;"
        "float mc=stats_f32[c],rc=stats_f32[C+c],gc=float(gamma[c]),bc=float(beta[c]);"
        "for(uint j=i;j<S&&j<i+4u;++j){"
        "uint idx=base+j;y[idx]=T(metal::fma((float(x[idx])-mc)*rc,gc,bc));}",
        "", true, false, mx::CompileOptions{});
    return k;
}

inline const ::mlx::core::fast::CustomKernelFunction& bn_bwd_s1_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel(
        "lucid_bn_bwd_s1", {"g_in", "x_in", "stats_f32"}, {"p"},
        "uint tid=thread_position_in_threadgroup.x,lane=thread_index_in_simdgroup;"
        "uint sg=simdgroup_index_in_threadgroup,g=threadgroup_position_in_grid.x;"
        "uint m=threadgroup_position_in_grid.y,G=threadgroups_per_grid.x;"
        "uint O=uint(g_in_shape[0]),M=uint(g_in_shape[1]),I=uint(g_in_shape[2]);"
        "float mc=stats_f32[m],rc=stats_f32[M+m],sg_acc=0.0f,sgx_acc=0.0f;"
        "if(G<=O){for(uint o=g;o<O;o+=G){uint base=(o*M+m)*I;"
        "for(uint i=tid;i<I;i+=256u){float gv=float(g_in[base+i]),xn=(float(x_in[base+i])-mc)*rc;"
        "sg_acc+=gv;sgx_acc=metal::fma(gv,xn,sgx_acc);}}}"
        "else{uint gpo=G/O,o=g/gpo,gi=g%gpo;if(o<O){uint base=(o*M+m)*I,step=gpo<<8;"
        "for(uint i=(gi<<8)+tid;i<I;i+=step){"
        "float gv=float(g_in[base+i]),xn=(float(x_in[base+i])-mc)*rc;"
        "sg_acc+=gv;sgx_acc=metal::fma(gv,xn,sgx_acc);}}}"
        "sg_acc=simd_sum(sg_acc);sgx_acc=simd_sum(sgx_acc);threadgroup float sh1[8],sh2[8];"
        "if(lane==0u){sh1[sg]=sg_acc;sh2[sg]=sgx_acc;}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "if(sg==0u){sg_acc=(lane<8u)?sh1[lane]:0.0f;sgx_acc=(lane<8u)?sh2[lane]:0.0f;"
        "sg_acc=simd_sum(sg_acc);sgx_acc=simd_sum(sgx_acc);"
        "if(lane==0u){uint idx=(m*G+g)*2u;p[idx]=sg_acc;p[idx+1u]=sgx_acc;}}",
        "", true, false, mx::CompileOptions{});
    return k;
}

inline const ::mlx::core::fast::CustomKernelFunction& bn_bwd_s2_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel(
        "lucid_bn_bwd_s2", {"p", "g_in", "gamma", "stats_f32"}, {"dgamma", "dbeta", "bwd_f32"},
        "uint tid=thread_position_in_threadgroup.x,lane=thread_index_in_simdgroup;"
        "uint sg=simdgroup_index_in_threadgroup,m=threadgroup_position_in_grid.y;"
        "uint M=uint(g_in_shape[1]),G=uint(p_shape[1]),O=uint(g_in_shape[0]),I=uint(g_in_shape[2]);"
        "uint base=m*G*2u;float s1=0.0f,s2=0.0f;"
        "for(uint g=tid;g<G;g+=256u){s1+=p[base+g*2u];s2+=p[base+g*2u+1u];}"
        "s1=simd_sum(s1);s2=simd_sum(s2);threadgroup float sh1[8],sh2[8];"
        "if(lane==0u){sh1[sg]=s1;sh2[sg]=s2;}"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        "if(sg==0u){s1=(lane<8u)?sh1[lane]:0.0f;s2=(lane<8u)?sh2[lane]:0.0f;"
        "s1=simd_sum(s1);s2=simd_sum(s2);"
        "if(lane==0u){float inv_n=1.0f/(float(O)*float(I));"
        "dbeta[m]=OutT(s1);dgamma[m]=TGamma(s2);"
        "bwd_f32[m]=s1*inv_n;bwd_f32[M+m]=s2*inv_n;"
        "bwd_f32[2u*M+m]=float(gamma[m])*stats_f32[M+m];}}",
        "", true, false, mx::CompileOptions{});
    return k;
}

inline const ::mlx::core::fast::CustomKernelFunction& bn_dx_kernel() {
    namespace mx = ::mlx::core;
    static const auto k = mx::fast::metal_kernel(
        "lucid_bn_dx", {"g_in", "x_in", "stats_f32", "bwd_f32"}, {"dx"},
        "uint tid=thread_position_in_grid.x,c=threadgroup_position_in_grid.y;"
        "uint b=threadgroup_position_in_grid.z,C=uint(g_in_shape[1]),S=uint(g_in_shape[2]);"
        "uint base=(b*C+c)*S,i=tid*4u;"
        "float mc=stats_f32[c],rc=stats_f32[C+c],mg=bwd_f32[c],mgx=bwd_f32[C+c],sc=bwd_f32[2u*C+c];"
        "for(uint j=i;j<S&&j<i+4u;++j){uint idx=base+j;"
        "float gv=float(g_in[idx]),xn=(float(x_in[idx])-mc)*rc;"
        "dx[idx]=T(sc*((gv-mg)-xn*mgx));}",
        "", true, false, mx::CompileOptions{});
    return k;
}

// Single-pass float32 mean and rstd over (O, M, I) -> {stats_f32, saved_mean, saved_rstd}.
inline std::vector<::mlx::core::array> omi_norm_stats(const ::mlx::core::array& x,
                                                      int O,
                                                      int M,
                                                      int I,
                                                      double eps,
                                                      const ::mlx::core::Shape& saved_shape) {
    namespace mx = ::mlx::core;
    auto x3 = reshape_contig(x, {O, M, I});
    const int G = pick_omi_groups(O, M, I);
    auto p = omi_stats_s1_kernel()({x3}, {{M, G, 3}}, {mx::float32}, {G * 256, M, 1}, {256, 1, 1},
                                   {{"T", x.dtype()}}, std::nullopt, false, {})[0];
    mx::array eps_arr({static_cast<float>(eps)}, {1}, mx::float32);
    return omi_norm_stats_s2_kernel()({p, eps_arr}, {{2, M}, saved_shape, saved_shape},
                                      {mx::float32, x.dtype(), x.dtype()}, {32, M, 1}, {32, 1, 1},
                                      {{"OutT", x.dtype()}}, std::nullopt, false, {});
}

// Fuse an element-wise composite with any number of inputs and outputs
// into one Metal kernel.
template <class Fn>
std::vector<::mlx::core::array> mlx_fused(const std::vector<::mlx::core::array>& ins, Fn) {
    for (const auto& a : ins) {
        if (a.size() == 0)
            return Fn{}(ins);
    }
    static const std::function<std::vector<::mlx::core::array>(
        const std::vector<::mlx::core::array>&)>
        compiled = ::mlx::core::compile(
            [](const std::vector<::mlx::core::array>& in) -> std::vector<::mlx::core::array> {
                return Fn{}(in);
            },
            /*shapeless=*/true);
    return compiled(ins);
}

// ``(x - mean)^2`` in float32 for a half ``x``.
inline ::mlx::core::array squared_deviation(const ::mlx::core::array& x,
                                            const ::mlx::core::array& mean) {
    return mlx_fused({x, mean}, [](const std::vector<::mlx::core::array>& in) {
        return std::vector<::mlx::core::array>{
            ::mlx::core::square(::mlx::core::subtract(widen_half(in[0]), in[1]))};
    })[0];
}

// ``a * b`` in float32 for half inputs.
inline ::mlx::core::array product_widened(const ::mlx::core::array& a,
                                          const ::mlx::core::array& b) {
    return mlx_fused({a, b}, [](const std::vector<::mlx::core::array>& in) {
        return std::vector<::mlx::core::array>{
            ::mlx::core::multiply(widen_half(in[0]), widen_half(in[1]))};
    })[0];
}

// ``scale * (g - mean_g - xnorm * mean_g_xn)`` in the accumulation dtype,
// rounded to ``g``'s dtype once, as one kernel.
inline ::mlx::core::array normalized_input_grad(const ::mlx::core::array& g,
                                                const ::mlx::core::array& xnorm,
                                                const ::mlx::core::array& mean_g,
                                                const ::mlx::core::array& mean_g_xn,
                                                const ::mlx::core::array& scale) {
    return mlx_fused(
        {g, xnorm, mean_g, mean_g_xn, scale}, [](const std::vector<::mlx::core::array>& in) {
            auto inner = ::mlx::core::subtract(::mlx::core::subtract(widen_half(in[0]), in[2]),
                                               ::mlx::core::multiply(widen_half(in[1]), in[3]));
            return std::vector<::mlx::core::array>{
                narrow_to(::mlx::core::multiply(widen_half(in[4]), inner), in[0].dtype())};
        })[0];
}

inline int spatial_extent(const ::mlx::core::array& x, int ndim) {
    int S = 1;
    for (int i = 0; i < ndim; ++i)
        S *= static_cast<int>(x.shape(2 + i));
    return S;
}

inline std::vector<Storage> bn_forward_gpu(const ::mlx::core::array& x,
                                           const ::mlx::core::array& gamma,
                                           const ::mlx::core::array& beta,
                                           int channels,
                                           int ndim,
                                           double eps,
                                           Dtype dt) {
    namespace mx = ::mlx::core;
    mx::Shape br_c(static_cast<std::size_t>(ndim) + 2, 1);
    br_c[1] = static_cast<mx::ShapeElem>(channels);
    const int B = static_cast<int>(x.shape(0));
    const int S = spatial_extent(x, ndim);
    if (x.size() > 0) {
        auto stats = omi_norm_stats(x, B, channels, S, eps, br_c);
        auto x3 = reshape_contig(x, {B, channels, S});
        auto g1 = reshape_contig(gamma, {channels});
        auto b1 = reshape_contig(beta, {channels});
        const int s_grid = (((S + 3) / 4 + 255) / 256) * 256;
        auto y3 = bn_affine_kernel()({x3, stats[0], g1, b1}, {{B, channels, S}}, {x.dtype()},
                                     {s_grid, channels, B}, {256, 1, 1}, {{"T", x.dtype()}},
                                     std::nullopt, false, {})[0];
        auto y = mx::reshape(y3, x.shape());
        return {Storage{gpu::wrap_mlx_array(std::move(y), dt)},
                Storage{gpu::wrap_mlx_array(std::move(stats[1]), dt)},
                Storage{gpu::wrap_mlx_array(std::move(stats[2]), dt)},
                Storage{gpu::wrap_mlx_array(std::move(stats[0]), Dtype::F32)}};
    }
    std::vector<int> axes{0};
    for (int i = 0; i < ndim; ++i)
        axes.push_back(2 + i);
    const auto x_dt = x.dtype();
    const auto acc_dt = accumulation_dtype(x_dt);
    const mx::array inv_N(1.0 / static_cast<double>(static_cast<std::int64_t>(B) * S), acc_dt);
    auto mean_acc = mx::multiply(mx::sum(widen_half(x), axes, true), inv_N);
    auto var_acc = mx::multiply(mx::sum(squared_deviation(x, mean_acc), axes, true), inv_N);
    auto rstd_acc = mx::rsqrt(mx::add(var_acc, gpu::mlx_scalar(eps, acc_dt)));
    auto y = mx::zeros(x.shape(), x_dt);
    return {Storage{gpu::wrap_mlx_array(std::move(y), dt)},
            Storage{gpu::wrap_mlx_array(narrow_to(mean_acc, x_dt), dt)},
            Storage{gpu::wrap_mlx_array(narrow_to(rstd_acc, x_dt), dt)}};
}

inline ::mlx::core::array extract_bn_stats_f32(const Storage& saved_xnorm,
                                               const ::mlx::core::array& saved_mean,
                                               const ::mlx::core::array& saved_rstd,
                                               int channels) {
    namespace mx = ::mlx::core;
    const auto* xn = std::get_if<GpuStorage>(&saved_xnorm);
    if (xn && xn->arr && xn->arr->size() == static_cast<std::size_t>(2 * channels) &&
        xn->arr->dtype() == mx::float32) {
        return *xn->arr;
    }
    return mx::stack({mx::reshape(widen_half(saved_mean), {channels}),
                      mx::reshape(widen_half(saved_rstd), {channels})},
                     0);
}

inline std::vector<Storage> bn_backward_gpu(const ::mlx::core::array& x,
                                            const ::mlx::core::array& gamma,
                                            const ::mlx::core::array& saved_mean,
                                            const ::mlx::core::array& saved_rstd,
                                            const Storage& saved_xnorm,
                                            const ::mlx::core::array& grad,
                                            int channels,
                                            int ndim,
                                            Dtype dt) {
    namespace mx = ::mlx::core;
    const int B = static_cast<int>(x.shape(0));
    const int S = spatial_extent(x, ndim);
    if (x.size() > 0 && grad.dtype() == x.dtype()) {
        auto stats_f32 = extract_bn_stats_f32(saved_xnorm, saved_mean, saved_rstd, channels);
        auto g3 = reshape_contig(grad, {B, channels, S});
        auto x3 = reshape_contig(x, {B, channels, S});
        auto gam1 = reshape_contig(gamma, {channels});
        const int G = pick_omi_groups(B, channels, S);
        auto p = bn_bwd_s1_kernel()({g3, x3, stats_f32}, {{channels, G, 2}}, {mx::float32},
                                    {G * 256, channels, 1}, {256, 1, 1}, {{"T", x.dtype()}},
                                    std::nullopt, false, {})[0];
        auto s2 = bn_bwd_s2_kernel()(
            {p, g3, gam1, stats_f32}, {{channels}, {channels}, {3, channels}},
            {gamma.dtype(), grad.dtype(), mx::float32}, {256, channels, 1}, {256, 1, 1},
            {{"T", x.dtype()}, {"OutT", grad.dtype()}, {"TGamma", gamma.dtype()}}, std::nullopt,
            false, {});
        const int s_grid = (((S + 3) / 4 + 255) / 256) * 256;
        auto dx3 = bn_dx_kernel()({g3, x3, stats_f32, s2[2]}, {{B, channels, S}}, {grad.dtype()},
                                  {s_grid, channels, B}, {256, 1, 1}, {{"T", grad.dtype()}},
                                  std::nullopt, false, {})[0];
        return {Storage{gpu::wrap_mlx_array(mx::reshape(dx3, x.shape()), dt)},
                Storage{gpu::wrap_mlx_array(std::move(s2[0]), dt)},
                Storage{gpu::wrap_mlx_array(std::move(s2[1]), dt)}};
    }
    mx::Shape br_c(static_cast<std::size_t>(ndim) + 2, 1);
    br_c[1] = static_cast<mx::ShapeElem>(channels);
    auto gamma_view = mx::reshape(gamma, br_c);
    auto xnorm = mx::multiply(mx::subtract(x, saved_mean), saved_rstd);
    std::vector<int> axes{0};
    for (int i = 0; i < ndim; ++i)
        axes.push_back(2 + i);
    const auto g_dt = grad.dtype();
    const auto acc_dt = accumulation_dtype(g_dt);
    auto sum_grad_kept = mx::sum(widen_half(grad), axes, true);
    auto sum_grad_xn_kept = mx::sum(product_widened(grad, xnorm), axes, true);
    const mx::array inv_N(1.0 / static_cast<double>(static_cast<std::int64_t>(B) * S), acc_dt);
    auto mean_g = mx::multiply(sum_grad_kept, inv_N);
    auto mean_g_xn = mx::multiply(sum_grad_xn_kept, inv_N);
    auto scale = mx::multiply(widen_half(gamma_view), widen_half(saved_rstd));
    auto dx = normalized_input_grad(grad, xnorm, mean_g, mean_g_xn, scale);
    mx::Shape c_only{static_cast<mx::ShapeElem>(channels)};
    auto dgamma = narrow_to(mx::reshape(sum_grad_xn_kept, c_only), g_dt);
    auto dbeta = narrow_to(mx::reshape(sum_grad_kept, c_only), g_dt);
    return {Storage{gpu::wrap_mlx_array(std::move(dx), dt)},
            Storage{gpu::wrap_mlx_array(std::move(dgamma), dt)},
            Storage{gpu::wrap_mlx_array(std::move(dbeta), dt)}};
}

// ``scatter_add_axis`` accumulated in float32 for a half input, returned in
// ``base``'s dtype.
inline ::mlx::core::array scatter_add_axis_widened(const ::mlx::core::array& base,
                                                   const ::mlx::core::array& idx,
                                                   const ::mlx::core::array& updates,
                                                   int axis) {
    namespace mx = ::mlx::core;
    if (base.dtype() == mx::complex64)
        return by_complex_parts(base, updates, [&](const auto& b, const auto& u) {
            return mx::scatter_add_axis(b, idx, u, axis);
        });
    return narrow_half(mx::scatter_add_axis(widen_half(base), idx, widen_half(updates), axis),
                       base.dtype());
}

// Pooled bag ``sum`` (``mode == 0``) or ``mean`` (``mode == 1``) for
// ``embedding_bag_forward``, accumulated in float32 when ``emb`` is a half
// dtype, then rounded back to ``emb``'s dtype once.
inline ::mlx::core::array embedding_bag_sum_or_mean(const ::mlx::core::array& emb,
                                                    const ::mlx::core::array& seg,
                                                    const ::mlx::core::array& valid,
                                                    const ::mlx::core::array& oh,
                                                    int B,
                                                    int D,
                                                    int n_idx,
                                                    int mode) {
    namespace mx = ::mlx::core;
    const auto mdt = emb.dtype();
    const auto acc_dt = accumulation_dtype(mdt);
    auto keep_row = mx::broadcast_to(
        mx::reshape(mx::greater(valid, mx::array(0.0f, mdt)), {n_idx, 1}), {n_idx, D});
    auto contributions = mx::where(keep_row, widen_half(emb), mx::zeros({n_idx, D}, acc_dt));
    auto seg_col = mx::broadcast_to(mx::reshape(seg, {n_idx, 1}), {n_idx, D});
    auto out = mx::scatter_add_axis(mx::zeros({B, D}, acc_dt), mx::astype(seg_col, mx::int32),
                                    contributions, 0);
    if (mode == 1) {
        auto sizes = mx::reshape(mx::sum(widen_half(oh), std::vector<int>{1}, false), {B, 1});
        auto denom = mx::broadcast_to(mx::maximum(sizes, mx::array(1.0f, acc_dt)), {B, D});
        out = mx::divide(out, denom);
    }
    return narrow_half(out, mdt);
}

}  // namespace backend
}  // namespace lucid
