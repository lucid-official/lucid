// lucid/_C/backend/gpu/IntGemm.h
//
// The integer matrix product on the GPU stream, exact at the dtype's width.
//
// MLX refuses an integer ``matmul`` ("[matmul] Only inexact types are
// supported"), and the engine answered it by widening to float32, multiplying
// and rounding back (LCD-1067).  Float32 holds 24 bits, so a dot product past
// 2^24 lost its low bits (``mm([[16777217]], [[1]])`` gave 16777216), and the
// cast back to the integer dtype *saturated*: an int8 product that overflowed
// came out 127 where the reference framework, and the CPU stream, wrap at the
// dtype's width (``[[100, 100], [100, 100]]`` squared is 32, not 127).
//
// MLX's integer ``multiply`` and ``sum`` do wrap, so the product is taken in
// the integer dtype itself as a broadcast multiply and a sum over ``K``:
//
//     C[.., i, j] = sum_k  A[.., i, k, None] * B[.., None, k, j]
//
// A sum of int8 or int16 products is accumulated in int32, which cannot lose
// what the final cast to the dtype keeps: the low ``width`` bits of a sum
// depend only on the low ``width`` bits of its terms.  The cast from int32 to
// int8 / int16 wraps, which is what the product wants and what makes the
// result equal the CPU kernel's (``backend/cpu/IntGemm.h``) bit for bit.
//
// The broadcast product holds ``batch * M * K * N`` elements, so it is formed
// a block of rows at a time (and, past one row's worth, a block of ``K`` at a
// time, the partial sums added), each block no larger than ``kChunkElems``
// elements.  The blocks of one product stay lazy and MLX evaluates them in
// order, freeing each block's products once its sum is made.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <mlx/array.h>
#include <mlx/ops.h>

#include "../../core/Dtype.h"
#include "MlxBridge.h"

namespace lucid::gpu {

// Most elements of the broadcast product one block forms: 16 MiB of int32.
inline constexpr std::int64_t kIntGemmChunkElems = std::int64_t{1} << 22;

namespace int_gemm_detail {

// ``x`` restricted to ``[lo, hi)`` along ``axis``.
inline ::mlx::core::array
slice_axis(const ::mlx::core::array& x, int axis, std::int64_t lo, std::int64_t hi) {
    ::mlx::core::Shape start(x.ndim(), 0);
    ::mlx::core::Shape stop = x.shape();
    start[axis] = static_cast<::mlx::core::ShapeElem>(lo);
    stop[axis] = static_cast<::mlx::core::ShapeElem>(hi);
    return ::mlx::core::slice(x, start, stop);
}

// Elements in the broadcast of the leading (batch) dimensions of ``a`` and
// ``b``, both of rank at least 2.
inline std::int64_t batch_elems(const ::mlx::core::array& a, const ::mlx::core::array& b) {
    const int a_rank = static_cast<int>(a.ndim()) - 2;
    const int b_rank = static_cast<int>(b.ndim()) - 2;
    const int rank = std::max(a_rank, b_rank);
    std::int64_t total = 1;
    for (int i = 0; i < rank; ++i) {
        const int ia = i - (rank - a_rank);
        const int ib = i - (rank - b_rank);
        const std::int64_t da = ia >= 0 ? a.shape(ia) : 1;
        const std::int64_t db = ib >= 0 ? b.shape(ib) : 1;
        total *= (da == 1 ? db : da);
    }
    return total;
}

// ``A @ B`` for one block, with ``A`` ``[.., m, k]`` and ``B`` ``[.., k, n]``,
// as the broadcast multiply and sum over ``k``.
inline ::mlx::core::array product_block(const ::mlx::core::array& a, const ::mlx::core::array& b) {
    auto products =
        ::mlx::core::multiply(::mlx::core::expand_dims(a, -1), ::mlx::core::expand_dims(b, -3));
    return ::mlx::core::sum(products, std::vector<int>{static_cast<int>(products.ndim()) - 2},
                            false);
}

// The blocked product, before the cast to the operands' dtype: int32 for an
// int8 or int16 operand pair, the operand dtype otherwise.
inline ::mlx::core::array wide_matmul(const ::mlx::core::array& a,
                                      const ::mlx::core::array& b,
                                      std::int64_t max_block_elems) {
    const int a_rank = static_cast<int>(a.ndim());
    const int b_rank = static_cast<int>(b.ndim());
    const std::int64_t M = a.shape(a_rank - 2);
    const std::int64_t K = a.shape(a_rank - 1);
    const std::int64_t N = b.shape(b_rank - 1);
    const std::int64_t batch = batch_elems(a, b);

    // An empty product has nothing to split, and the one-block form answers
    // it: a sum over no ``k`` is zeros, and an empty batch or row stays empty.
    if (batch * M * K * N == 0)
        return product_block(a, b);

    const std::int64_t row_elems = batch * K * N;
    std::int64_t rows_per_block = M;
    std::int64_t k_per_block = K;
    if (row_elems > max_block_elems) {
        rows_per_block = 1;
        k_per_block = std::max<std::int64_t>(1, max_block_elems / (batch * N));
    } else {
        rows_per_block = std::min(M, std::max<std::int64_t>(1, max_block_elems / row_elems));
    }

    std::vector<::mlx::core::array> row_blocks;
    for (std::int64_t m0 = 0; m0 < M; m0 += rows_per_block) {
        const std::int64_t m1 = std::min(M, m0 + rows_per_block);
        const auto a_rows = rows_per_block < M ? slice_axis(a, a_rank - 2, m0, m1) : a;
        std::optional<::mlx::core::array> acc;
        for (std::int64_t k0 = 0; k0 < K; k0 += k_per_block) {
            const std::int64_t k1 = std::min(K, k0 + k_per_block);
            const bool whole_k = k_per_block >= K;
            auto piece = product_block(whole_k ? a_rows : slice_axis(a_rows, a_rank - 1, k0, k1),
                                       whole_k ? b : slice_axis(b, b_rank - 2, k0, k1));
            acc = acc ? ::mlx::core::add(*acc, piece) : piece;
        }
        row_blocks.push_back(*acc);
    }
    if (row_blocks.size() == 1)
        return row_blocks.front();
    const int row_axis = static_cast<int>(row_blocks.front().ndim()) - 2;
    return ::mlx::core::concatenate(std::move(row_blocks), row_axis);
}

}  // namespace int_gemm_detail

// ``A @ B`` for the signed integer dtype of ``a`` and ``b``, wrapped at that
// dtype's width.
//
// Parameters
// ----------
// a, b : mlx::core::array
//     Operands ``[.., M, K]`` and ``[.., K, N]`` of one integer dtype (a view
//     from ``swapaxes`` is read through its strides).  Their batch dimensions
//     broadcast.
// max_block_elems : std::int64_t
//     Most elements of the broadcast product one block forms.  The default is
//     what the engine uses; a smaller value makes a small product take the
//     blocked path, which is how the tests reach it.
//
// Returns
// -------
// mlx::core::array
//     The ``[.., M, N]`` product, in the operands' dtype.  Lazy, as every MLX
//     result is.
inline ::mlx::core::array int_matmul(const ::mlx::core::array& a,
                                     const ::mlx::core::array& b,
                                     std::int64_t max_block_elems = kIntGemmChunkElems) {
    return ::mlx::core::astype(int_gemm_detail::wide_matmul(a, b, max_block_elems), a.dtype());
}

// ``tensordot(a, b, axes_a, axes_b)`` for the signed integer dtype of ``a`` and
// ``b``, as the product of the two operands laid out as matrices: ``a``'s free
// axes by the contracted ones, and the contracted ones by ``b``'s free axes.
//
// Parameters
// ----------
// a, b : mlx::core::array
//     Operands of one integer dtype.
// axes_a, axes_b : std::vector<int>
//     Contracted axes, paired by position; a negative axis counts from the end.
//
// Returns
// -------
// mlx::core::array
//     Shape ``a``'s free dimensions followed by ``b``'s, in the operands' dtype.
inline ::mlx::core::array int_tensordot(const ::mlx::core::array& a,
                                        const ::mlx::core::array& b,
                                        const std::vector<int>& axes_a,
                                        const std::vector<int>& axes_b) {
    // The axes of ``x`` the contraction leaves, then (``contracted_first``
    // aside) the contracted ones, as the permutation that lays ``x`` out as
    // ``free x contracted`` or ``contracted x free``.
    auto layout = [](const ::mlx::core::array& x, const std::vector<int>& axes,
                     bool contracted_first, ::mlx::core::Shape& free_dims,
                     std::int64_t& contracted_size) {
        const int rank = static_cast<int>(x.ndim());
        std::vector<int> contracted;
        std::vector<bool> is_contracted(static_cast<std::size_t>(rank), false);
        contracted_size = 1;
        for (int axis : axes) {
            const int ax = axis < 0 ? axis + rank : axis;
            contracted.push_back(ax);
            is_contracted[static_cast<std::size_t>(ax)] = true;
            contracted_size *= x.shape(ax);
        }
        std::vector<int> free_axes;
        for (int ax = 0; ax < rank; ++ax) {
            if (!is_contracted[static_cast<std::size_t>(ax)]) {
                free_axes.push_back(ax);
                free_dims.push_back(x.shape(ax));
            }
        }
        std::vector<int> perm;
        const auto& head = contracted_first ? contracted : free_axes;
        const auto& tail = contracted_first ? free_axes : contracted;
        perm.insert(perm.end(), head.begin(), head.end());
        perm.insert(perm.end(), tail.begin(), tail.end());
        return perm;
    };
    ::mlx::core::Shape a_free;
    ::mlx::core::Shape b_free;
    std::int64_t k_a = 1;
    std::int64_t k_b = 1;
    const auto perm_a = layout(a, axes_a, false, a_free, k_a);
    const auto perm_b = layout(b, axes_b, true, b_free, k_b);
    std::int64_t m = 1;
    for (auto d : a_free)
        m *= d;
    std::int64_t n = 1;
    for (auto d : b_free)
        n *= d;
    const auto a_mat =
        ::mlx::core::reshape(::mlx::core::transpose(a, perm_a),
                             ::mlx::core::Shape{static_cast<::mlx::core::ShapeElem>(m),
                                                static_cast<::mlx::core::ShapeElem>(k_a)});
    const auto b_mat =
        ::mlx::core::reshape(::mlx::core::transpose(b, perm_b),
                             ::mlx::core::Shape{static_cast<::mlx::core::ShapeElem>(k_b),
                                                static_cast<::mlx::core::ShapeElem>(n)});
    ::mlx::core::Shape out_shape = a_free;
    out_shape.insert(out_shape.end(), b_free.begin(), b_free.end());
    return ::mlx::core::reshape(int_matmul(a_mat, b_mat), out_shape);
}

}  // namespace lucid::gpu
