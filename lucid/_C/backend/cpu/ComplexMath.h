// lucid/_C/backend/cpu/ComplexMath.h
//
// Elementwise complex math that Accelerate has no vector entry for: vForce
// is real-only, and vDSP's complex routines are split-complex (separate real
// and imaginary arrays) while Lucid stores ``[re, im, re, im, ...]``.  Each
// function reads and writes the interleaved buffer as ``std::complex``, which
// the standard allows for an array of ``Scalar``, and has the signature
// ``CpuBackend::complex_binary`` takes, so a binary one plugs in beside
// ``vzmul_c64``.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>

#include "Parallel.h"
#include "Vforce.h"

namespace lucid::backend::cpu {

// A 16-byte element — a complex128 — read as its bytes.  ``where`` and its
// gradient only copy elements, so they dispatch on element width, and no
// scalar type is 16 bytes wide.
struct Word128 {
    std::uint64_t lo;
    std::uint64_t hi;
};

// Elements of a complex log or power worth a core of their own.  They cost
// tens of nanoseconds an element (a real ``exp`` is ~5), so a quarter of the
// real transcendental grain is already past the cost of a dispatch.
inline constexpr std::size_t kComplexMathGrain = 4096;

// Elements per vForce call inside a parallel chunk of ``vzlog``: the tile of
// magnitudes lives on the stack (8 KiB of double).
inline constexpr std::size_t kComplexLogTile = 1024;

// The real logarithm of a buffer of magnitudes, through vForce.
inline void vlog_real(const float* in, float* out, std::size_t n) {
    vlog_f32(in, out, n);
}

inline void vlog_real(const double* in, double* out, std::size_t n) {
    vlog_f64(in, out, n);
}

// The principal logarithm over interleaved complex values:
// $\ln z = \ln |z| + i \arg z$.  The magnitude is ``std::abs`` (a hypot, so
// it neither overflows nor underflows where $x^2 + y^2$ would) and the
// argument is ``std::arg``; the logarithm of the magnitudes, the
// transcendental part, is one vForce call per tile.
template <typename Scalar>
void vzlog(const Scalar* in, Scalar* out, std::size_t n) {
    const auto* src = reinterpret_cast<const std::complex<Scalar>*>(in);
    auto* dst = reinterpret_cast<std::complex<Scalar>*>(out);
    parallel_for(n, kComplexMathGrain, [&](std::size_t lo, std::size_t hi) {
        Scalar magnitude[kComplexLogTile];
        for (std::size_t start = lo; start < hi; start += kComplexLogTile) {
            const std::size_t count = std::min(kComplexLogTile, hi - start);
            for (std::size_t i = 0; i < count; ++i)
                magnitude[i] = std::abs(src[start + i]);
            vlog_real(magnitude, magnitude, count);
            for (std::size_t i = 0; i < count; ++i)
                dst[start + i] = std::complex<Scalar>(magnitude[i], std::arg(src[start + i]));
        }
    });
}

// ``out[i] = std::pow(base[i], expo[i])`` over interleaved complex values.
// There is no vForce power for a complex base, and ``std::pow`` is the
// function the reference's CPU kernel calls, so the special cases agree with
// it: ``pow(0, 0)`` is NaN and ``pow(0, 1)`` is 0 in both.
template <typename Scalar>
void vzpow(const Scalar* base, const Scalar* expo, Scalar* out, std::size_t n) {
    const auto* bp = reinterpret_cast<const std::complex<Scalar>*>(base);
    const auto* ep = reinterpret_cast<const std::complex<Scalar>*>(expo);
    auto* dst = reinterpret_cast<std::complex<Scalar>*>(out);
    parallel_for(n, kComplexMathGrain, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i)
            dst[i] = std::pow(bp[i], ep[i]);
    });
}

// ``out[i] = in[i] + scalar`` — a real scalar adds to the real part only, the
// imaginary part passes through.
template <typename Scalar>
void vzsadd(const Scalar* in, Scalar scalar, Scalar* out, std::size_t n) {
    const auto* src = reinterpret_cast<const std::complex<Scalar>*>(in);
    auto* dst = reinterpret_cast<std::complex<Scalar>*>(out);
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = std::complex<Scalar>(src[i].real() + scalar, src[i].imag());
}

}  // namespace lucid::backend::cpu
