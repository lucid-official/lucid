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
#pragma once

namespace lucid::backend {

inline const char* binary_kernel_load_src(bool strided) {
    return strided ? R"(
        uint elem = thread_position_in_grid.x;
        T av = a[elem_to_loc(int64_t(elem), a_shape, a_strides, a_ndim)];
        T bv = b[elem_to_loc(int64_t(elem), b_shape, b_strides, b_ndim)];
    )"
                   : R"(
        uint elem = thread_position_in_grid.x;
        T av = AScalar ? a[0] : a[elem];
        T bv = BScalar ? b[0] : b[elem];
    )";
}

}  // namespace lucid::backend
