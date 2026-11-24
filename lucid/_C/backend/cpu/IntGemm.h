// lucid/_C/backend/cpu/IntGemm.h
//
// The integer matrix product on the CPU stream, exact at the dtype's width.
//
// Accelerate has no integer GEMM, so an integer ``matmul`` was widened to
// double, multiplied by ``cblas_dgemm`` and rounded back (LCD-1067).  A double
// holds 53 bits, and the product of two integer matrices is an integer matrix
// that wraps at the operand width, as the reference framework's does: an int64
// dot product past 2^53 lost its low bits before the wrap ran
// (``mm([[2**53 + 1]], [[1]])`` gave ``2**53``), and so did an int32 one summed
// over a long K.
//
// The arithmetic runs in an unsigned word of 32 or 64 bits.  Unsigned, because
// a signed product or sum that overflows is undefined behaviour while the wrap
// at 2^N of an unsigned one is the language's own rule; the low ``width`` bits
// of a sum or product of integers depend only on the low ``width`` bits of its
// terms, so one 32-bit word serves int8, int16 and int32 and the result is cut
// to the dtype at the store.  The word is never narrower than ``unsigned int``:
// two ``uint16_t`` operands would otherwise promote to *signed* ``int``, and
// 65535 * 65535 overflows it.
//
// Every output element is the dot product of one row of ``A`` and one column of
// ``B`` taken along ``K``, so the kernel lays both out contiguous along ``K`` —
// a transposed copy for the operand whose layout is the other way round — and
// every dot is a plain reduction the compiler vectorises.  A transposed copy is
// O(MK + KN) against the O(MNK) product, and an integer product is rare.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "../../core/Dtype.h"
#include "Parallel.h"

namespace lucid::backend::cpu {

namespace int_gemm_detail {

// The unsigned type the arithmetic of a ``T`` product runs in.
template <typename T>
using Word = std::conditional_t<(sizeof(T) <= sizeof(std::uint32_t)), std::uint32_t, std::uint64_t>;

// Multiply-accumulates one task is worth: below it a dispatch costs more than
// the rows it hands out.
inline constexpr std::size_t kGrainMacs = std::size_t{1} << 15;

// A row-major ``rows x cols`` matrix reads the same flat sequence as its
// transpose when either extent is 1, so only the others need a copy.
inline bool needs_transpose(std::size_t rows, std::size_t cols) {
    return rows > 1 && cols > 1;
}

// ``dst[b][c][r] = src[b][r][c]`` over a batch of ``rows x cols`` matrices.
template <typename T>
void transpose_batch(const T* src, T* dst, std::size_t batch, std::size_t rows, std::size_t cols) {
    for (std::size_t bi = 0; bi < batch; ++bi) {
        const T* in = src + bi * rows * cols;
        T* out = dst + bi * rows * cols;
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c)
                out[c * rows + r] = in[r * cols + c];
        }
    }
}

// The dot product of two length-``k`` rows, wrapped to ``T``.
template <typename T>
T dot_wrapped(const T* a, const T* b, std::size_t k) {
    using W = Word<T>;
    W acc = 0;
    for (std::size_t i = 0; i < k; ++i)
        acc += static_cast<W>(a[i]) * static_cast<W>(b[i]);
    return static_cast<T>(acc);
}

}  // namespace int_gemm_detail

// ``out[b] = A[b] @ B[b]`` for ``batch`` row-major matrices, wrapped at ``T``'s
// width.
//
// Parameters
// ----------
// trans_a, trans_b : bool
//     ``A`` is stored ``[batch, K, M]`` / ``B`` as ``[batch, N, K]`` when set,
//     as in ``MatmulOpts``.
// batch, M, N, K : std::size_t
//     ``A`` is ``M x K`` and ``B`` is ``K x N`` before any transposition; the
//     output is ``batch x M x N``, row-major and contiguous.
// a, b, out : pointers
//     Operand and result buffers.  ``out`` is written in full, never read, and
//     must not alias an operand.
//
// Notes
// -----
// ``K == 0`` is an empty sum and writes zeros; an empty output touches nothing.
// A task writes only the output rows it owns and cannot throw, as
// ``parallel_for`` requires.
template <typename T>
void int_gemm(bool trans_a,
              bool trans_b,
              std::size_t batch,
              std::size_t M,
              std::size_t N,
              std::size_t K,
              const T* a,
              const T* b,
              T* out) {
    static_assert(std::is_integral_v<T> && std::is_signed_v<T>,
                  "int_gemm takes the signed integer dtypes");
    const std::size_t out_n = batch * M * N;
    if (out_n == 0)
        return;
    if (K == 0) {
        std::fill_n(out, out_n, T{0});
        return;
    }
    // Rows of A and columns of B, contiguous along K.
    std::vector<T> a_rows;
    std::vector<T> b_cols;
    const T* ap = a;
    const T* bp = b;
    if (trans_a && int_gemm_detail::needs_transpose(K, M)) {
        a_rows.resize(batch * M * K);
        int_gemm_detail::transpose_batch(a, a_rows.data(), batch, K, M);
        ap = a_rows.data();
    }
    if (!trans_b && int_gemm_detail::needs_transpose(K, N)) {
        b_cols.resize(batch * N * K);
        int_gemm_detail::transpose_batch(b, b_cols.data(), batch, K, N);
        bp = b_cols.data();
    }
    const std::size_t rows = batch * M;
    const std::size_t grain = std::max<std::size_t>(1, int_gemm_detail::kGrainMacs / (N * K));
    parallel_for(rows, grain, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t r = lo; r < hi; ++r) {
            const T* a_row = ap + r * K;
            const T* b_mat = bp + (r / M) * N * K;
            T* out_row = out + r * N;
            for (std::size_t j = 0; j < N; ++j)
                out_row[j] = int_gemm_detail::dot_wrapped(a_row, b_mat + j * K, K);
        }
    });
}

// ``int_gemm`` for the dtype ``dt``, over raw buffers.
//
// Returns
// -------
// bool
//     ``false`` when ``dt`` is not one of ``I8`` / ``I16`` / ``I32`` / ``I64``,
//     with nothing written; the caller refuses it in its own words.
inline bool int_gemm_by_dtype(Dtype dt,
                              bool trans_a,
                              bool trans_b,
                              std::size_t batch,
                              std::size_t M,
                              std::size_t N,
                              std::size_t K,
                              const std::byte* a,
                              const std::byte* b,
                              std::byte* out) {
    auto run = [&](auto tag) {
        using T = decltype(tag);
        int_gemm<T>(trans_a, trans_b, batch, M, N, K, reinterpret_cast<const T*>(a),
                    reinterpret_cast<const T*>(b), reinterpret_cast<T*>(out));
    };
    switch (dt) {
    case Dtype::I8:
        run(std::int8_t{});
        return true;
    case Dtype::I16:
        run(std::int16_t{});
        return true;
    case Dtype::I32:
        run(std::int32_t{});
        return true;
    case Dtype::I64:
        run(std::int64_t{});
        return true;
    default:
        return false;
    }
}

}  // namespace lucid::backend::cpu
