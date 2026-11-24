// lucid/_C/backend/cpu/Gemm.h
//
// The matrix product of the dtypes ``CpuBackend::matmul`` does not hand to the
// real ``sgemm`` / ``dgemm``: the complex ones, which Accelerate multiplies
// with ``cblas_cgemm`` / ``cblas_zgemm``, and the integer ones, which have no
// BLAS and take Lucid's own wrapping kernel (``IntGemm.h``).

#pragma once

#include <cstddef>

#include "../IBackend.h"
#include "Blas.h"
#include "IntGemm.h"

namespace lucid::backend::cpu {

// One complex GEMM per batch entry, writing the products into ``c``.
//
// ``T`` is the element type and ``gemm`` the BLAS routine for it.  An empty
// extent leaves ``c`` as the caller zeroed it: that is already the answer (a
// contraction over nothing sums to 0), and BLAS exits the process on a zero
// leading dimension.
template <typename T, typename Gemm>
void complex_gemm_batches(
    const MatmulOpts& opts, const std::byte* a, const std::byte* b, std::byte* c, Gemm gemm) {
    const int M = opts.M, K = opts.K, N = opts.N;
    if (M <= 0 || N <= 0 || K <= 0)
        return;
    const auto* ap = reinterpret_cast<const T*>(a);
    const auto* bp = reinterpret_cast<const T*>(b);
    auto* cp = reinterpret_cast<T*>(c);
    for (std::size_t bi = 0; bi < opts.batch; ++bi) {
        gemm(opts.transA, opts.transB, M, N, K, T{1}, ap + bi * M * K, opts.transA ? M : K,
             bp + bi * K * N, opts.transB ? K : N, T{0}, cp + bi * M * N, N);
    }
}

// The product for ``dt`` when it is complex or integer.
//
// Returns
// -------
// bool
//     ``false`` for any other dtype, with nothing written; the caller refuses
//     it in its own words.
inline bool gemm_by_dtype(
    Dtype dt, const MatmulOpts& opts, const std::byte* a, const std::byte* b, std::byte* out) {
    switch (dt) {
    case Dtype::C64:
        complex_gemm_batches<Complex64>(opts, a, b, out, cgemm);
        return true;
    case Dtype::C128:
        complex_gemm_batches<Complex128>(opts, a, b, out, zgemm);
        return true;
    default:
        return int_gemm_by_dtype(dt, opts.transA, opts.transB, opts.batch,
                                 static_cast<std::size_t>(opts.M), static_cast<std::size_t>(opts.N),
                                 static_cast<std::size_t>(opts.K), a, b, out);
    }
}

}  // namespace lucid::backend::cpu
