// lucid/_C/backend/cpu/Vforce.cpp
//
// Implements the vForce wrappers declared in Vforce.h.  All non-power functions
// hand off to the corresponding vv*f / vv* symbol from Accelerate.  The count
// argument required by vForce is a signed int*, so every function captures a
// local int before the call via the helper N().
//
// vpow_f32 / vpow_f64 note: vvpowf and vvpow take (out, exponent, base, &n),
// which is the opposite order from pow(base, exp) in C.  The wrappers accept
// (base, expo, out, n) to match the Lucid calling convention and internally
// pass expo before base to the vForce API.
//
// The LUCID_VFORCE_UNARY macro at the bottom expands pairs of f32/f64 wrappers
// for functions that share a uniform (out, in, &count) signature (log2, fabs,
// floor, ceil, round).  LUCID_VFORCE_EDGED does the same for the
// trigonometric and hyperbolic functions, which also post-fix the special
// points vForce gets wrong (see split_edges).
//
// Square root and reciprocal are IEEE operations the reference rounds
// correctly, and the float vForce entries for them do not, so their float
// wrappers are plain loops (see vsqrt_f32 and vrec_f32).

#include "Vforce.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include "ErfPoly.h"
#include "Parallel.h"

namespace lucid::backend::cpu {

namespace {
// Converts a std::size_t element count to the signed int* expected by all
// vForce vector math functions.
inline int N(std::size_t n) {
    return static_cast<int>(n);
}

// Elements of a vForce call worth a core of their own: at a few ns an
// element, 16k of them are tens of microseconds, well past a dispatch.
constexpr std::size_t kGrain = 16384;

// vvsinf, vvcosf and vvtanf work in blocks of this many elements from the
// start of a call.  A block holding a non-finite or large argument (from
// 1e10 for sin / cos, 1e4 for tan) takes a slower path, and every value in
// that block comes out with other bits: tiny ones are flushed or a few ulp
// off, the rest move by an ulp (LCD-472).  So one element's bits depend on
// where the call started, and also on the output pointer: an output that
// is not 16-byte aligned shifts the blocks.  Engine buffers are 64-byte
// aligned (kCpuAlignment).  The other vForce entries here are element-wise.
constexpr std::size_t kVforceBlock = 16;

// ``fn(lo, hi)`` over [0, n) in chunks across cores, each starting on a
// kVforceBlock boundary, so every element sits in the same block as it
// would in one call over the whole array.  The bits then match a single
// call at any core count.  One call on one core was what kept
// ``lucid.exp`` at twice the reference and ``lucid.erf`` at 14x.  A call
// from inside a parallel chunk (GELU's tiles) is below the grain and runs
// inline.
template <class F>
inline void for_blocks(std::size_t n, F&& fn) {
    const std::size_t blocks = (n + kVforceBlock - 1) / kVforceBlock;
    parallel_for(blocks, kGrain / kVforceBlock, [&](std::size_t lo, std::size_t hi) {
        fn(lo * kVforceBlock, std::min(hi * kVforceBlock, n));
    });
}

// Runs ``fn(out, in, &count)`` over [0, n) in chunks across cores.
template <class T, class F>
inline void split(const T* in, T* out, std::size_t n, F&& fn) {
    for_blocks(n, [&](std::size_t lo, std::size_t hi) {
        int count = N(hi - lo);
        fn(out + lo, in + lo, &count);
    });
}

// ``out[i] = fn(in[i])`` over [0, n), chunked across cores like split().
// Used where the vForce entry is not exact, and for erf, which vForce lacks.
template <class T, class F>
inline void split_map(const T* in, T* out, std::size_t n, F fn) {
    parallel_for(n, kGrain, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i)
            out[i] = fn(in[i]);
    });
}

// The trigonometric and hyperbolic kernels are vForce, then a post-fix of the
// special points where the reference returns the exact value and vForce does
// not (LCD-472).  Measured on macOS 26 (M1 Pro) over every float in
// [-4pi, 4pi], a stride over the rest of the float range and 4.2e7 doubles,
// against the reference and the correctly rounded value:
//
// - cos rounds to 1 for |x| <= 2^-12 (float) and |x| < 2^-26.5 (double).
//   vvcosf / vvcos return 1 - 1 ulp there, so cos(0) was 0.99999994.
// - tan and tanh round to x for |x| <= 3.52e-4 (float) and 1.35e-8
//   (double), and so does float sin.  vvtanf / vvtan return +0 for -0 and
//   are 1 ulp off further in; vvtanhf / vvtanh are 1 ulp off from 4.2e-8 /
//   4e-11 up; in a slow block (see kVforceBlock) vvsinf and vvtanf flush
//   subnormals to 0, lose the sign of -0 and are 1-3 ulp off tiny values.
//
// asin, acos, atan, sinh and cosh were exact at their special points (the
// sign of -0, subnormals, cosh(0) = 1, +-inf, NaN), and vvtanhf saturates to
// +-1 exactly where the rounded value does, so they take no fix.  Elsewhere
// vForce stays within the error test_cpu_trig_vforce.py records (at most
// 4 ulp of the reference); libm would be within 1 ulp but is 7-14x slower a
// core and is not Accelerate (H3).  The ±1 plateaus of sin / cos away from 0
// come back 1-2 ulp short in float (up to 3 in double) and are left:
// telling them apart needs a range reduction, which measured 24-73% on top
// of vvcosf.
enum class Edge { None, One, Identity };

// For |x| at or under the limit, ``One`` writes 1 and ``Identity`` writes x.
// Each limit is the last value where the correctly rounded result is still
// that for every function given it (sin's own would be 4.44e-4 in float),
// so the fix only ever writes the correctly rounded answer.
template <class T>
struct EdgeRule {
    Edge kind;
    T limit;
};

struct EdgeSpec {
    EdgeRule<float> f32;
    EdgeRule<double> f64;
};

constexpr EdgeSpec kNoEdge{{Edge::None, 0.0f}, {Edge::None, 0.0}};
constexpr EdgeSpec kCosEdge{{Edge::One, 0x1p-12f}, {Edge::One, 0x1.6a09e667f3bccp-27}};
constexpr EdgeSpec kOddEdge{{Edge::Identity, 0x1.713744p-12f},
                            {Edge::Identity, 0x1.d12ed0af1a27ep-27}};
// vvsin is exact near 0; only vvsinf's slow blocks get tiny values wrong.
constexpr EdgeSpec kSinEdge{kOddEdge.f32, kNoEdge.f64};

// GELU's tanh calls in with tiles of this size, so it stays one vForce call.
constexpr std::size_t kEdgeTile = 1024;

inline float min_magnitude(const float* x, std::size_t n) {
    float m = 0.0f;
    vDSP_minmgv(x, 1, &m, n);
    return m;
}

// Not vDSP_minmgvD, which took 0.32 ms on 1e6 doubles (one core) against
// 0.10 ms for this loop, and made the gate cost +26% on vvcos.  The float
// entry measured the same as a NEON loop and stays.
inline double min_magnitude(const double* x, std::size_t n) {
    float64x2_t a = vdupq_n_f64(INFINITY);
    float64x2_t b = a;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        a = vminq_f64(a, vabsq_f64(vld1q_f64(x + i)));
        b = vminq_f64(b, vabsq_f64(vld1q_f64(x + i + 2)));
    }
    double m = vminvq_f64(vminq_f64(a, b));
    for (; i < n; ++i)
        m = std::fmin(m, std::fabs(x[i]));
    return m;
}

// out = |x| <= limit ? edge value : y.  NaN fails the compare and keeps
// vForce's NaN.
template <Edge kKind>
inline void select_edges(const float* x, const float* y, float* out, std::size_t n, float limit) {
    const float32x4_t lim = vdupq_n_f32(limit);
    const float32x4_t one = vdupq_n_f32(1.0f);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t xv = vld1q_f32(x + i);
        const float32x4_t fix = kKind == Edge::One ? one : xv;
        vst1q_f32(out + i, vbslq_f32(vcaleq_f32(xv, lim), fix, vld1q_f32(y + i)));
    }
    for (; i < n; ++i)
        out[i] = std::fabs(x[i]) <= limit ? (kKind == Edge::One ? 1.0f : x[i]) : y[i];
}

template <Edge kKind>
inline void
select_edges(const double* x, const double* y, double* out, std::size_t n, double limit) {
    const float64x2_t lim = vdupq_n_f64(limit);
    const float64x2_t one = vdupq_n_f64(1.0);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const float64x2_t xv = vld1q_f64(x + i);
        const float64x2_t fix = kKind == Edge::One ? one : xv;
        vst1q_f64(out + i, vbslq_f64(vcaleq_f64(xv, lim), fix, vld1q_f64(y + i)));
    }
    for (; i < n; ++i)
        out[i] = std::fabs(x[i]) <= limit ? (kKind == Edge::One ? 1.0 : x[i]) : y[i];
}

// split() with the edge post-fix, a tile at a time.  A select over every
// element cost +31% on vvcosf (1e6, one core); most tiles have no |x| under
// the limit, so a min_magnitude scan sends those straight through vForce
// and the cost drops to +10% (cos) and +6% (tanh) in float.  A tile that
// needs the fix computes into a stack buffer, so a caller working in place
// (GELU's tanh) still has x to compare against.  Tiles start on block
// boundaries, so either way a value gets the bits one whole call would give
// it.
template <EdgeSpec kSpec, class T, class F>
inline void split_edges(const T* in, T* out, std::size_t n, F&& fn) {
    static_assert(kEdgeTile % kVforceBlock == 0);
    constexpr EdgeRule<T> rule = [] {
        if constexpr (std::is_same_v<T, float>)
            return kSpec.f32;
        else
            return kSpec.f64;
    }();
    if constexpr (rule.kind == Edge::None) {
        split(in, out, n, fn);
    } else {
        constexpr T limit = rule.limit;
        for_blocks(n, [&](std::size_t lo, std::size_t hi) {
            for (std::size_t b = lo; b < hi; b += kEdgeTile) {
                const std::size_t m = std::min(kEdgeTile, hi - b);
                int count = N(m);
                if (min_magnitude(in + b, m) > limit) {
                    fn(out + b, in + b, &count);
                    continue;
                }
                // Aligned like the tensor's own buffer, so vForce cuts the
                // same blocks here as when it writes to out directly.
                alignas(64) T tile[kEdgeTile];
                fn(tile, in + b, &count);
                select_edges<rule.kind>(in + b, tile, out + b, m, limit);
            }
        });
    }
}
}  // namespace

void vexp_f32(const float* in, float* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvexpf(o, i, c); });
}

void vlog_f32(const float* in, float* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvlogf(o, i, c); });
}

// Deliberately not vvsqrtf, which is not correctly rounded: on 1e6 gaussian
// magnitudes 41% of its roots were 1–2 ulp off the IEEE root.  The
// reference's square root is exact, and so is the one on Metal.  This loop
// compiles to NEON ``fsqrt``, which IEEE 754 requires to be correctly
// rounded, and it measured faster on one core (0.17 ms against 0.31 ms on
// 1e6 elements).  The double path stays on ``vvsqrt``, which was exact on the
// same data.
void vsqrt_f32(const float* in, float* out, std::size_t n) {
    split_map(in, out, n, [](float x) { return std::sqrt(x); });
}

void vexp_f64(const double* in, double* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvexp(o, i, c); });
}

void vlog_f64(const double* in, double* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvlog(o, i, c); });
}

void vsqrt_f64(const double* in, double* out, std::size_t n) {
    split(in, out, n, [](auto* o, const auto* i, int* c) { vvsqrt(o, i, c); });
}

void vpow_f32(const float* base, const float* expo, float* out, std::size_t n) {
    int count = N(n);

    vvpowf(out, expo, base, &count);
}

void vpow_f64(const double* base, const double* expo, double* out, std::size_t n) {
    int count = N(n);
    vvpow(out, expo, base, &count);
}

// Expands a matching f32 and f64 wrapper pair for any vForce function that
// follows the (output_ptr, input_ptr, &count) calling convention.
#define LUCID_VFORCE_UNARY(NAME, F32, F64)                                                         \
    void NAME##_f32(const float* in, float* out, std::size_t n) {                                  \
        split(in, out, n, [](float* o, const float* i, int* c) { F32(o, i, c); });                 \
    }                                                                                              \
    void NAME##_f64(const double* in, double* out, std::size_t n) {                                \
        split(in, out, n, [](double* o, const double* i, int* c) { F64(o, i, c); });               \
    }

// The trigonometric and hyperbolic pairs, each with its edge post-fix.
#define LUCID_VFORCE_EDGED(NAME, F32, F64, SPEC)                                                   \
    void NAME##_f32(const float* in, float* out, std::size_t n) {                                  \
        split_edges<SPEC>(in, out, n, [](float* o, const float* i, int* c) { F32(o, i, c); });     \
    }                                                                                              \
    void NAME##_f64(const double* in, double* out, std::size_t n) {                                \
        split_edges<SPEC>(in, out, n, [](double* o, const double* i, int* c) { F64(o, i, c); });   \
    }

LUCID_VFORCE_EDGED(vsin, vvsinf, vvsin, kSinEdge)
LUCID_VFORCE_EDGED(vcos, vvcosf, vvcos, kCosEdge)
LUCID_VFORCE_EDGED(vtan, vvtanf, vvtan, kOddEdge)
LUCID_VFORCE_EDGED(vasin, vvasinf, vvasin, kNoEdge)
LUCID_VFORCE_EDGED(vacos, vvacosf, vvacos, kNoEdge)
LUCID_VFORCE_EDGED(vatan, vvatanf, vvatan, kNoEdge)
LUCID_VFORCE_EDGED(vsinh, vvsinhf, vvsinh, kNoEdge)
LUCID_VFORCE_EDGED(vcosh, vvcoshf, vvcosh, kNoEdge)
LUCID_VFORCE_EDGED(vtanh, vvtanhf, vvtanh, kOddEdge)

#undef LUCID_VFORCE_EDGED

LUCID_VFORCE_UNARY(vtanh_unfixed, vvtanhf, vvtanh)
LUCID_VFORCE_UNARY(vlog2, vvlog2f, vvlog2)
LUCID_VFORCE_UNARY(vfabs, vvfabsf, vvfabs)
LUCID_VFORCE_UNARY(vfloor, vvfloorf, vvfloor)
LUCID_VFORCE_UNARY(vceil, vvceilf, vvceil)
LUCID_VFORCE_UNARY(vround, vvnintf, vvnint)

#undef LUCID_VFORCE_UNARY

// Deliberately not vvrecf, which is not correctly rounded: on 1e6 gaussian
// values 34% of its reciprocals were 1 ulp off ``1 / x``.  The reference
// divides, and sigmoid, rsqrt and the half reciprocals all go through here.
// ``fdiv`` is exact and measured faster on one core (0.085 ms against
// 0.15 ms on 1e6 elements).  ``vvrec`` was exact on the same data and stays.
void vrec_f32(const float* in, float* out, std::size_t n) {
    split_map(in, out, n, [](float x) { return 1.0f / x; });
}

void vrec_f64(const double* in, double* out, std::size_t n) {
    split(in, out, n, [](double* o, const double* i, int* c) { vvrec(o, i, c); });
}

// erf — Apple Accelerate does not expose a vForce erf symbol (no vverff /
// vverf in <Accelerate/Accelerate.h>).  float32 uses erf_f32 (ErfPoly.h):
// within 1 ulp of the correctly rounded value and branch-free, so this loop
// vectorises, which a loop over libm's erff did not.  float64 keeps std::erf
// across parallel_for chunks (0.37 ms on 1e6 elements on M4 Max; allowed under
// the 2026-10-24 H3 ruling since Accelerate has no double erf routine).
void verf_f32(const float* in, float* out, std::size_t n) {
    parallel_for(n, kGrain, [&](std::size_t lo, std::size_t hi) {
        const float* __restrict x = in;
        float* __restrict y = out;
        for (std::size_t i = lo; i < hi; ++i)
            y[i] = erf_f32(x[i]);
    });
}
void verf_f64(const double* in, double* out, std::size_t n) {
    split_map(in, out, n, [](double x) { return std::erf(x); });
}

}  // namespace lucid::backend::cpu
