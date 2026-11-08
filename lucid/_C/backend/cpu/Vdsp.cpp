// lucid/_C/backend/cpu/Vdsp.cpp
//
// Implements the vDSP wrapper functions declared in Vdsp.h.  Each function is
// a one-line delegate to the corresponding Apple Accelerate vDSP_* intrinsic
// with unit stride (ia=1, ib=1, ic=1).  The anonymous helper L() converts the
// Lucid std::size_t element count to vDSP_Length (also std::size_t on Apple
// Silicon, but the cast documents the intent).
//
// Note on vDSP_vsub argument order: vDSP uses (B, ib, A, ia, C, ic, N) with
// the convention C = A - B.  vsub_f32 and vsub_f64 therefore swap the a/b
// argument order when calling vDSP so that the wrapper's semantics match the
// caller's expectation (out = a - b).

#include "Vdsp.h"

#include <cmath>
#include <complex>
#include <cstring>
#include <limits>

#include <Accelerate/Accelerate.h>

namespace lucid::backend::cpu {

namespace {

// Converts a std::size_t element count to the vDSP_Length type expected by
// the Accelerate framework APIs.
inline vDSP_Length L(std::size_t n) {
    return static_cast<vDSP_Length>(n);
}

// Clamp to ``[lo, hi]``; a zero that ties a zero bound takes the sign the
// extremum rule gives it (max of a signed-zero tie is +0, min is -0, as
// ``maximum`` / ``minimum``): a +0 floor also takes -0 (``<=``), a -0 cap also
// takes +0 (``>=``).  The comparison is chosen once, outside the loop, so the
// plain loops stay vectorised.  NaN fails every comparison and passes through.
template <bool kLoTies, bool kHiTies, typename T>
void clip_run(const T* src, T* dst, std::size_t n, T lo, T hi) {
    for (std::size_t i = 0; i < n; ++i) {
        T v = src[i];
        if (kLoTies ? v <= lo : v < lo)
            v = lo;
        else if (kHiTies ? v >= hi : v > hi)
            v = hi;
        dst[i] = v;
    }
}

template <typename T>
void clip_dispatch(const T* src, T* dst, std::size_t n, T lo, T hi) {
    const bool lo_ties = lo == T(0) && !std::signbit(lo);
    const bool hi_ties = hi == T(0) && std::signbit(hi);
    if (lo_ties && hi_ties)
        clip_run<true, true>(src, dst, n, lo, hi);
    else if (lo_ties)
        clip_run<true, false>(src, dst, n, lo, hi);
    else if (hi_ties)
        clip_run<false, true>(src, dst, n, lo, hi);
    else
        clip_run<false, false>(src, dst, n, lo, hi);
}

}  // namespace

void vadd_f32(const float* a, const float* b, float* out, std::size_t n) {
    vDSP_vadd(a, 1, b, 1, out, 1, L(n));
}

void vsub_f32(const float* a, const float* b, float* out, std::size_t n) {
    vDSP_vsub(b, 1, a, 1, out, 1, L(n));
}

void vmul_f32(const float* a, const float* b, float* out, std::size_t n) {
    vDSP_vmul(a, 1, b, 1, out, 1, L(n));
}

// Deliberately not vDSP_vdiv / vDSP_vdivD, neither of which is correctly
// rounded across supported macOS releases (LCD-541).  On macOS 26 ``vDSP_vdiv``
// is 1–2 ulp off on 40.3% of 1e6 gaussian float32 pairs and flips the sign of
// zero / infinity on subnormal and signed-zero divisors, just as ``vDSP_vdivD``
// is 1–2 ulp off on 41% of float64 pairs (``1 / 0.001`` came out
// 999.9999999999999, ``0 / 5e-324`` gave NaN, and ``1 / 5e-324`` gave -inf).
//
// Plain loops compile to NEON ``fdiv``, which IEEE 754 requires to be
// correctly rounded at both widths.
void vdiv_f32(const float* a, const float* b, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] / b[i];
}

void vadd_f64(const double* a, const double* b, double* out, std::size_t n) {
    vDSP_vaddD(a, 1, b, 1, out, 1, L(n));
}

void vsub_f64(const double* a, const double* b, double* out, std::size_t n) {
    vDSP_vsubD(b, 1, a, 1, out, 1, L(n));
}

void vmul_f64(const double* a, const double* b, double* out, std::size_t n) {
    vDSP_vmulD(a, 1, b, 1, out, 1, L(n));
}

void vdiv_f64(const double* a, const double* b, double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] / b[i];
}

void vneg_f32(const float* in, float* out, std::size_t n) {
    vDSP_vneg(in, 1, out, 1, L(n));
}

void vabs_f32(const float* in, float* out, std::size_t n) {
    vDSP_vabs(in, 1, out, 1, L(n));
}

void vsq_f32(const float* in, float* out, std::size_t n) {
    vDSP_vsq(in, 1, out, 1, L(n));
}

void vneg_f64(const double* in, double* out, std::size_t n) {
    vDSP_vnegD(in, 1, out, 1, L(n));
}

void vabs_f64(const double* in, double* out, std::size_t n) {
    vDSP_vabsD(in, 1, out, 1, L(n));
}

void vsq_f64(const double* in, double* out, std::size_t n) {
    vDSP_vsqD(in, 1, out, 1, L(n));
}

void vsadd_f32(const float* in, float scalar, float* out, std::size_t n) {
    vDSP_vsadd(in, 1, &scalar, out, 1, L(n));
}

void vsmul_f32(const float* in, float scalar, float* out, std::size_t n) {
    vDSP_vsmul(in, 1, &scalar, out, 1, L(n));
}

void vsadd_f64(const double* in, double scalar, double* out, std::size_t n) {
    vDSP_vsaddD(in, 1, &scalar, out, 1, L(n));
}

void vsmul_f64(const double* in, double scalar, double* out, std::size_t n) {
    vDSP_vsmulD(in, 1, &scalar, out, 1, L(n));
}

void widen_f32_f64(const float* in, double* out, std::size_t n) {
    vDSP_vspdp(in, 1, out, 1, L(n));
}

void vsma_f32(const float* a, float scalar, const float* b, float* out, std::size_t n) {
    vDSP_vsma(a, 1, &scalar, b, 1, out, 1, L(n));
}

void vsma_f64(const double* a, double scalar, const double* b, double* out, std::size_t n) {
    vDSP_vsmaD(a, 1, &scalar, b, 1, out, 1, L(n));
}

// Deliberately not vDSP_vthres.  That primitive returns the *threshold*
// for a NaN input, so relu(NaN) came out 0 on the CPU while the Metal
// path returned NaN — the same op disagreeing across devices, and a NaN
// entering a network silently becoming a zero at the first activation.
// Accelerate's own vDSP_vmax propagates NaN, so the two primitives
// disagree and this one inherited the wrong half.
//
// The branchless form below is IEEE-correct (``NaN < 0`` is false, so the
// NaN passes through) and measured *faster* than vDSP_vthres on 4M
// floats: 0.54 ms against 0.73 ms, with vDSP_vmax against a zero buffer
// slowest at 0.88 ms.
// ``<=`` rather than ``<``: relu(-0) is +0 (max(-0, +0) of the reference), and a NaN
// still fails both and passes through.
void vrelu_f32(const float* in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = (in[i] <= 0.0f) ? 0.0f : in[i];
}

void vrelu_f64(const double* in, double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = (in[i] <= 0.0) ? 0.0 : in[i];
}

void vclip_f32(const float* in, float* out, std::size_t n, float lo, float hi) {
    clip_dispatch(in, out, n, lo, hi);
}

void vclip_f64(const double* in, double* out, std::size_t n, double lo, double hi) {
    clip_dispatch(in, out, n, lo, hi);
}

void vmax_f32(const float* a, const float* b, float* out, std::size_t n) {
    vDSP_vmax(a, 1, b, 1, out, 1, L(n));
}

void vmin_f32(const float* a, const float* b, float* out, std::size_t n) {
    vDSP_vmin(a, 1, b, 1, out, 1, L(n));
}

// ``vDSP_vmaxD`` and ``vDSP_vminD`` do not propagate NaN — their
// single-precision counterparts do.  Same primitive family, opposite
// answer, and nothing in the documentation distinguishes them:
// ``minimum([nan, 1, 3], [2, nan, 2])`` came back ``[2, nan, 2]`` in
// double and ``[nan, nan, 2]`` in float, so whether an op saw a NaN
// depended on the dtype it was called at.  The reference propagates from
// either operand at both widths, and so does the float path here, so the
// double one is the one that was wrong.
//
// Written out rather than corrected in a second pass: a fix-up would
// re-read the whole buffer to repair what one compare avoids.  The float
// versions stay on vDSP because they are already right and already
// vectorised — the test pins both, so a change in Accelerate is a test
// failure rather than a silent one.
void vmax_f64(const double* a, const double* b, double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(a[i]) || std::isnan(b[i]))
            out[i] = std::numeric_limits<double>::quiet_NaN();
        else
            out[i] = a[i] > b[i] ? a[i] : b[i];
    }
}

void vmin_f64(const double* a, const double* b, double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(a[i]) || std::isnan(b[i]))
            out[i] = std::numeric_limits<double>::quiet_NaN();
        else
            out[i] = a[i] < b[i] ? a[i] : b[i];
    }
}

void vge_mask_f32(const float* a, const float* b, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = (a[i] >= b[i]) ? 1.0f : 0.0f;
}

void vle_mask_f32(const float* a, const float* b, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
}

void vge_mask_f64(const double* a, const double* b, double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = (a[i] >= b[i]) ? 1.0 : 0.0;
}

void vle_mask_f64(const double* a, const double* b, double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = (a[i] < b[i]) ? 1.0 : 0.0;
}

float vsum_f32(const float* in, std::size_t n) {
    float s = 0.f;
    vDSP_sve(in, 1, &s, L(n));
    return s;
}
double vsum_f64(const double* in, std::size_t n) {
    double s = 0.0;
    vDSP_sveD(in, 1, &s, L(n));
    return s;
}
float vmean_f32(const float* in, std::size_t n) {
    float m = 0.f;
    vDSP_meanv(in, 1, &m, L(n));
    return m;
}
double vmean_f64(const double* in, std::size_t n) {
    double m = 0.0;
    vDSP_meanvD(in, 1, &m, L(n));
    return m;
}
float vmaxval_f32(const float* in, std::size_t n) {
    float m = 0.f;
    vDSP_maxv(in, 1, &m, L(n));
    return m;
}
double vmaxval_f64(const double* in, std::size_t n) {
    double m = 0.0;
    vDSP_maxvD(in, 1, &m, L(n));
    return m;
}
float vdotpr_f32(const float* a, const float* b, std::size_t n) {
    float s = 0.f;
    vDSP_dotpr(a, 1, b, 1, &s, L(n));
    return s;
}
double vdotpr_f64(const double* a, const double* b, std::size_t n) {
    double s = 0.0;
    vDSP_dotprD(a, 1, b, 1, &s, L(n));
    return s;
}
void vmadd_f32(const float* a, const float* b, const float* c, float* out, std::size_t n) {
    vDSP_vma(a, 1, b, 1, c, 1, out, 1, L(n));
}
void vmadd_f64(const double* a, const double* b, const double* c, double* out, std::size_t n) {
    vDSP_vmaD(a, 1, b, 1, c, 1, out, 1, L(n));
}

void vadd_i32(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

void vadd_i64(const std::int64_t* a, const std::int64_t* b, std::int64_t* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

void vzmul_c64(const float* a, const float* b, float* out, std::size_t n) {
    // vDSP's ``zvmul`` operates on split-complex ``DSPSplitComplex`` (separate
    // real / imag arrays), but Lucid's C64 storage is interleaved
    // ``[re, im, re, im, ...]``.  Going through ``vDSP_ctoz`` + ``zvmul``
    // + ``vDSP_ztoc`` would mean two extra full passes over the data; a
    // straight ``std::complex<float>`` loop is simpler and the compiler
    // auto-vectorises it with NEON on Apple Silicon.
    const auto* ac = reinterpret_cast<const std::complex<float>*>(a);
    const auto* bc = reinterpret_cast<const std::complex<float>*>(b);
    auto* oc = reinterpret_cast<std::complex<float>*>(out);
    for (std::size_t i = 0; i < n; ++i)
        oc[i] = ac[i] * bc[i];
}

namespace {

// ``out[i] = op(a[i], b[i])`` over interleaved complex64, the loop
// :func:`vzmul_c64` spells out by hand.
template <typename Scalar, typename Op>
void zv_binary(const Scalar* a, const Scalar* b, Scalar* out, std::size_t n, Op op) {
    const auto* ac = reinterpret_cast<const std::complex<Scalar>*>(a);
    const auto* bc = reinterpret_cast<const std::complex<Scalar>*>(b);
    auto* oc = reinterpret_cast<std::complex<Scalar>*>(out);
    for (std::size_t i = 0; i < n; ++i)
        oc[i] = op(ac[i], bc[i]);
}

}  // namespace

void vzadd_c64(const float* a, const float* b, float* out, std::size_t n) {
    zv_binary(a, b, out, n, [](std::complex<float> x, std::complex<float> y) { return x + y; });
}

void vzsub_c64(const float* a, const float* b, float* out, std::size_t n) {
    zv_binary(a, b, out, n, [](std::complex<float> x, std::complex<float> y) { return x - y; });
}

void vzdiv_c64(const float* a, const float* b, float* out, std::size_t n) {
    zv_binary(a, b, out, n, [](std::complex<float> x, std::complex<float> y) { return x / y; });
}

void vzconj_c64(const float* a, float* out, std::size_t n) {
    // Copy the full interleaved buffer, then negate only the imag halves
    // via a stride-2 view starting at offset 1.  ``vDSP_vneg`` handles the
    // stride natively.
    std::memcpy(out, a, n * 2 * sizeof(float));
    vDSP_vneg(out + 1, 2, out + 1, 2, L(n));
}

void vzadd_c128(const double* a, const double* b, double* out, std::size_t n) {
    zv_binary(a, b, out, n, [](std::complex<double> x, std::complex<double> y) { return x + y; });
}

void vzsub_c128(const double* a, const double* b, double* out, std::size_t n) {
    zv_binary(a, b, out, n, [](std::complex<double> x, std::complex<double> y) { return x - y; });
}

void vzmul_c128(const double* a, const double* b, double* out, std::size_t n) {
    zv_binary(a, b, out, n, [](std::complex<double> x, std::complex<double> y) { return x * y; });
}

void vzdiv_c128(const double* a, const double* b, double* out, std::size_t n) {
    zv_binary(a, b, out, n, [](std::complex<double> x, std::complex<double> y) { return x / y; });
}

}  // namespace lucid::backend::cpu
