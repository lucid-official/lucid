// lucid/_C/backend/cpu/Reduce.cpp
//
// Implements the single-axis reduction primitives declared in Reduce.h.
//
// The generic axis_reduce template handles arbitrary outer/reduce/inner triples
// using a three-level nested loop.  For sum_axis_f32/f64 when inner == 1
// (i.e. the reduction is over the last, contiguous dimension), vDSP_sve /
// vDSP_sveD is used instead because it performs a numerically stable
// compensated summation in a single pass and is vectorised by Accelerate.
// The other reductions (max, min, prod) use the generic template even for
// inner == 1 because there is no vDSP equivalent with the same semantics.

#include "Reduce.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

namespace lucid::backend::cpu {

namespace {

// Generic single-axis reduction over the [outer, reduce_dim, inner] layout.
// identity is the neutral element of op (0 for sum, -inf for max, +inf for
// min, 1 for prod).
//
// The reduced axis is walked outermost and the ``inner`` row innermost, so
// each step folds one contiguous row into a row of accumulators — a vector
// loop — rather than chasing one strided element at a time.  Every output
// still folds its inputs in the same order (r = 0, 1, ...), so the result is
// the same bits.
template <typename T, typename Op>
void axis_reduce(const T* in,
                 T* out,
                 std::size_t outer,
                 std::size_t reduce_dim,
                 std::size_t inner,
                 T identity,
                 Op op) {
    for (std::size_t o = 0; o < outer; ++o) {
        T* __restrict acc = out + o * inner;
        std::fill_n(acc, inner, identity);
        for (std::size_t r = 0; r < reduce_dim; ++r) {
            const T* __restrict row = in + (o * reduce_dim + r) * inner;
            for (std::size_t i = 0; i < inner; ++i)
                acc[i] = op(acc[i], row[i]);
        }
    }
}

// The extreme of two non-NaN values, with -0 below +0: ``max`` of a signed-zero
// tie is +0 and ``min`` is -0, whichever came first.  ``a > b ? a : b`` keeps
// the first of a tie, so the sign of a zero result followed the element order;
// NEON's FMAX / FMIN (the vector body of ``extreme_*`` below) already rank the
// zeros, and this is the scalar form of the same rule.
template <bool kMax, typename T>
inline T extreme_of(T a, T b) {
    if constexpr (kMax)
        return a > b ? a : (b > a ? b : (std::signbit(a) ? b : a));
    else
        return a < b ? a : (b < a ? b : (std::signbit(a) ? a : b));
}

// ``extreme_of`` that propagates NaN: the combine step of the strided reductions.
template <bool kMax, typename T>
inline T extreme_nan(T a, T b) {
    if (std::isnan(a) || std::isnan(b))
        return std::numeric_limits<T>::quiet_NaN();
    return extreme_of<kMax>(a, b);
}

// The max or min of ``n`` contiguous values, NaN if any is NaN.
//
// NEON's FMAX / FMIN (and the across-lane FMAXV / FMINV) propagate NaN —
// the contract below — so four independent 4-lane accumulators run the
// row at full width.  The scalar ``a > b ? a : b`` chain, with its NaN
// tests, was one serial dependency: ``x.max()`` took 25 times ``x.sum()``.
template <bool kMax>
float extreme_f32(const float* p, std::size_t n) {
    constexpr float kStart =
        kMax ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    auto pick = [](float32x4_t a, float32x4_t b) {
        if constexpr (kMax)
            return vmaxq_f32(a, b);
        else
            return vminq_f32(a, b);
    };
    float32x4_t a0 = vdupq_n_f32(kStart), a1 = a0, a2 = a0, a3 = a0;
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = pick(a0, vld1q_f32(p + i));
        a1 = pick(a1, vld1q_f32(p + i + 4));
        a2 = pick(a2, vld1q_f32(p + i + 8));
        a3 = pick(a3, vld1q_f32(p + i + 12));
    }
    for (; i + 4 <= n; i += 4)
        a0 = pick(a0, vld1q_f32(p + i));
    const float32x4_t all = pick(pick(a0, a1), pick(a2, a3));
    float acc = kMax ? vmaxvq_f32(all) : vminvq_f32(all);
    for (; i < n; ++i) {
        if (std::isnan(p[i]) || std::isnan(acc))
            return std::numeric_limits<float>::quiet_NaN();
        acc = extreme_of<kMax>(acc, p[i]);
    }
    return acc;
}

template <bool kMax>
double extreme_f64(const double* p, std::size_t n) {
    constexpr double kStart =
        kMax ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    auto pick = [](float64x2_t a, float64x2_t b) {
        if constexpr (kMax)
            return vmaxq_f64(a, b);
        else
            return vminq_f64(a, b);
    };
    float64x2_t a0 = vdupq_n_f64(kStart), a1 = a0, a2 = a0, a3 = a0;
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 = pick(a0, vld1q_f64(p + i));
        a1 = pick(a1, vld1q_f64(p + i + 2));
        a2 = pick(a2, vld1q_f64(p + i + 4));
        a3 = pick(a3, vld1q_f64(p + i + 6));
    }
    for (; i + 2 <= n; i += 2)
        a0 = pick(a0, vld1q_f64(p + i));
    const float64x2_t all = pick(pick(a0, a1), pick(a2, a3));
    double acc = kMax ? vmaxvq_f64(all) : vminvq_f64(all);
    for (; i < n; ++i) {
        if (std::isnan(p[i]) || std::isnan(acc))
            return std::numeric_limits<double>::quiet_NaN();
        acc = extreme_of<kMax>(acc, p[i]);
    }
    return acc;
}

// The strided reductions, ``inner`` contiguous lanes folded one row at a time.
// NEON's FMAX / FMIN are NaN-propagating and rank the zeros, so a full
// register of lanes is one instruction; the scalar ``extreme_nan`` only
// finishes the lanes that do not fill one.
template <bool kMax>
void extreme_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    constexpr float kStart =
        kMax ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    for (std::size_t o = 0; o < outer; ++o) {
        float* __restrict acc = out + o * inner;
        std::fill_n(acc, inner, kStart);
        for (std::size_t r = 0; r < reduce_dim; ++r) {
            const float* __restrict row = in + (o * reduce_dim + r) * inner;
            std::size_t i = 0;
            for (; i + 16 <= inner; i += 16) {
                for (std::size_t k = 0; k < 16; k += 4) {
                    const float32x4_t a = vld1q_f32(acc + i + k), b = vld1q_f32(row + i + k);
                    vst1q_f32(acc + i + k, kMax ? vmaxq_f32(a, b) : vminq_f32(a, b));
                }
            }
            for (; i + 4 <= inner; i += 4) {
                const float32x4_t a = vld1q_f32(acc + i), b = vld1q_f32(row + i);
                vst1q_f32(acc + i, kMax ? vmaxq_f32(a, b) : vminq_f32(a, b));
            }
            for (; i < inner; ++i)
                acc[i] = extreme_nan<kMax>(acc[i], row[i]);
        }
    }
}

template <bool kMax>
void extreme_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    constexpr double kStart =
        kMax ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    for (std::size_t o = 0; o < outer; ++o) {
        double* __restrict acc = out + o * inner;
        std::fill_n(acc, inner, kStart);
        for (std::size_t r = 0; r < reduce_dim; ++r) {
            const double* __restrict row = in + (o * reduce_dim + r) * inner;
            std::size_t i = 0;
            for (; i + 8 <= inner; i += 8) {
                for (std::size_t k = 0; k < 8; k += 2) {
                    const float64x2_t a = vld1q_f64(acc + i + k), b = vld1q_f64(row + i + k);
                    vst1q_f64(acc + i + k, kMax ? vmaxq_f64(a, b) : vminq_f64(a, b));
                }
            }
            for (; i + 2 <= inner; i += 2) {
                const float64x2_t a = vld1q_f64(acc + i), b = vld1q_f64(row + i);
                vst1q_f64(acc + i, kMax ? vmaxq_f64(a, b) : vminq_f64(a, b));
            }
            for (; i < inner; ++i)
                acc[i] = extreme_nan<kMax>(acc[i], row[i]);
        }
    }
}

}  // namespace

void sum_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    if (inner == 1) {
        for (std::size_t o = 0; o < outer; ++o) {
            float acc = 0.f;
            vDSP_sve(in + o * reduce_dim, 1, &acc, static_cast<vDSP_Length>(reduce_dim));
            out[o] = acc;
        }
        return;
    }
    axis_reduce<float>(in, out, outer, reduce_dim, inner, 0.f,
                       [](float a, float b) { return a + b; });
}

void sum_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    if (inner == 1) {
        for (std::size_t o = 0; o < outer; ++o) {
            double acc = 0.0;
            vDSP_sveD(in + o * reduce_dim, 1, &acc, static_cast<vDSP_Length>(reduce_dim));
            out[o] = acc;
        }
        return;
    }
    axis_reduce<double>(in, out, outer, reduce_dim, inner, 0.0,
                        [](double a, double b) { return a + b; });
}

// NaN propagates through max and min, as IEEE-754's maximum/minimum
// operations and the reference framework both do.
//
// ``a > b ? a : b`` does not: every comparison against a NaN is false, so
// the NaN loses and vanishes.  ``lucid.max`` on a tensor containing one
// answered 3.0 while Metal — and everyone else — answered nan, which is
// the worst kind of disagreement: a plausible number in place of the
// signal that the data was bad.  A poisoned batch reduced to a healthy
// looking maximum and training carried on.
void max_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    if (inner == 1) {
        for (std::size_t o = 0; o < outer; ++o)
            out[o] = extreme_f32<true>(in + o * reduce_dim, reduce_dim);
        return;
    }
    extreme_axis_f32<true>(in, out, outer, reduce_dim, inner);
}

void max_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    if (inner == 1) {
        for (std::size_t o = 0; o < outer; ++o)
            out[o] = extreme_f64<true>(in + o * reduce_dim, reduce_dim);
        return;
    }
    extreme_axis_f64<true>(in, out, outer, reduce_dim, inner);
}

void min_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    if (inner == 1) {
        for (std::size_t o = 0; o < outer; ++o)
            out[o] = extreme_f32<false>(in + o * reduce_dim, reduce_dim);
        return;
    }
    extreme_axis_f32<false>(in, out, outer, reduce_dim, inner);
}

void min_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    if (inner == 1) {
        for (std::size_t o = 0; o < outer; ++o)
            out[o] = extreme_f64<false>(in + o * reduce_dim, reduce_dim);
        return;
    }
    extreme_axis_f64<false>(in, out, outer, reduce_dim, inner);
}

void prod_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    axis_reduce<float>(in, out, outer, reduce_dim, inner, 1.f,
                       [](float a, float b) { return a * b; });
}

void prod_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    axis_reduce<double>(in, out, outer, reduce_dim, inner, 1.0,
                        [](double a, double b) { return a * b; });
}

}  // namespace lucid::backend::cpu
