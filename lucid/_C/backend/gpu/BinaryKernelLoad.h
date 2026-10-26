// lucid/_C/backend/gpu/BinaryKernelLoad.h
//
// How a two-input element-wise ``mx::fast::metal_kernel`` reads its
// operands: the Metal source that puts this thread's element index in
// ``elem`` and the operands in ``av`` and ``bv`` (of the kernel's template
// type ``T``), for the custom kernels that share that shape (DivMod.h,
// Pow.h).
//
// The contiguous form reads ``a`` and ``b`` flat, or element 0 of an operand
// whose ``AScalar`` / ``BScalar`` template flag is set; the kernel is built
// with ``ensure_row_contiguous``.  The strided form reads both through
// ``elem_to_loc``, so broadcast (stride-0) and permuted operands need no
// copy; the kernel is built without ``ensure_row_contiguous`` and receives
// ``a_shape`` / ``a_strides`` / ``a_ndim`` (and ``b_*``).
//
// Either way one thread computes one element.  Because MLX's grid axes are
// signed 32-bit ``int``s while a tensor's element count ``n`` is 64-bit,
// :func:`metal_launch_1d` uses a 1-D grid ``(n, 1, 1)`` up to
// ``kMaxMetalGridX`` and tiles across ``grid.y`` past ``kMaxMetalGridX``,
// while the shader computes a 64-bit ``elem`` index and guards tail threads
// when ``threads_per_grid.y > 1``.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>

namespace lucid::backend {

// Largest multiple of 256 that fits in a signed 32-bit MLX grid axis.
inline constexpr std::size_t kMaxMetalGridX = 0x7fffff00ULL;

struct MetalLaunch1D {
    std::tuple<int, int, int> grid;
    std::tuple<int, int, int> threadgroup;
};

// Build a ``(grid, threadgroup)`` pair covering ``n`` elements without
// narrowing ``n`` to 32 bits.
inline constexpr MetalLaunch1D metal_launch_1d(std::size_t n,
                                               std::size_t max_tg = 256,
                                               std::size_t max_grid_x = kMaxMetalGridX) noexcept {
    if (n == 0)
        return {{0, 1, 1}, {1, 1, 1}};
    const int gx = (n <= max_grid_x) ? static_cast<int>(n) : static_cast<int>(max_grid_x);
    const std::size_t gy_sz = (n <= max_grid_x) ? 1 : ((n + max_grid_x - 1) / max_grid_x);
    const int gy = static_cast<int>(gy_sz);
    const int tg = static_cast<int>((n < max_tg) ? n : max_tg);
    return {{gx, gy, 1}, {tg, 1, 1}};
}

// 64-bit linear element index matching the MSL ``elem`` computation in
// :func:`unary_elem_index_src` and :func:`binary_kernel_load_src`.
inline constexpr std::uint64_t
metal_linear_elem(std::uint32_t tid_x, std::uint32_t tid_y, std::uint32_t grid_x) noexcept {
    return static_cast<std::uint64_t>(tid_y) * static_cast<std::uint64_t>(grid_x) +
           static_cast<std::uint64_t>(tid_x);
}

// MSL preamble computing 64-bit ``elem`` and guarding tail threads when
// ``threads_per_grid.y > 1`` against ``a_shape[0 .. a_ndim - 1]``.
inline const char* unary_elem_index_src() {
    return R"(
        uint64_t elem = uint64_t(thread_position_in_grid.y) * uint64_t(threads_per_grid.x) +
                        uint64_t(thread_position_in_grid.x);
        if (threads_per_grid.y > 1) {
            uint64_t n_total = 1;
            for (int d = 0; d < a_ndim; ++d)
                n_total *= uint64_t(uint(a_shape[d]));
            if (elem >= n_total)
                return;
        }
    )";
}

inline std::string binary_kernel_load_src(bool strided) {
    if (strided) {
        return std::string(unary_elem_index_src()) + R"(
        T av = a[elem_to_loc(int64_t(elem), a_shape, a_strides, a_ndim)];
        T bv = b[elem_to_loc(int64_t(elem), b_shape, b_strides, b_ndim)];
    )";
    }
    return R"(
        uint64_t elem = uint64_t(thread_position_in_grid.y) * uint64_t(threads_per_grid.x) +
                        uint64_t(thread_position_in_grid.x);
        if (threads_per_grid.y > 1) {
            const constant int* out_s = AScalar ? b_shape : a_shape;
            int out_d = AScalar ? b_ndim : a_ndim;
            uint64_t n_total = 1;
            for (int d = 0; d < out_d; ++d)
                n_total *= uint64_t(uint(out_s[d]));
            if (elem >= n_total)
                return;
        }
        T av = AScalar ? a[0] : a[elem];
        T bv = BScalar ? b[0] : b[elem];
    )";
}

}  // namespace lucid::backend
