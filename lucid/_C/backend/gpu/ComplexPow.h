// lucid/_C/backend/gpu/ComplexPow.h
//
// ``x ** y`` for complex64 on Metal, with the special values of ``std::pow``
// at a zero base (LCD-1082).
//
// ``mx::power``'s complex kernel has a branch for a base of ``0 + 0j`` that
// answers ``0 + 0j`` whatever the exponent, and NaN only when the exponent has
// a NaN part.  The CPU stream and the reference compute ``std::pow(x, y)`` =
// ``exp(y * log(x))``, whose value at a zero base is one of three:
//
//     0 ** 0         = nan + nan j        0 ** 2.5       = 0
//     0 ** -1        = inf + nan j        0 ** (1 + 1j)  = 0
//
// ``log(+-0 +- 0j)`` is ``-inf + theta j`` with ``theta`` in ``{+0, -0, pi,
// -pi}`` by the signs of the base's parts, and the product with the exponent
// ``c + d j`` then follows C99 Annex G (an infinity times a zero is NaN, and
// ``libc++`` recovers an infinity from a NaN pair).  Working that product and
// ``exp`` through gives a table over the signs and classes of ``c`` and ``d``
// (``lucid_zero_base_pow`` below); it was checked against the reference over a
// grid of 169 exponents per signed zero (finite, +-0, +-inf, NaN, 1e+-30, and
// +-3e38, whose product with pi overflows float32) in both widths, and the CPU
// stream is held to the same grid.
//
// The rest of the tensor is ``mx::power``'s own answer, unchanged: a one-pass
// patch kernel replaces the lanes MLX answered from its zero branch, so no
// other lane changes by a bit.  The extra pass over the operands and the result
// costs 1.5-1.7x of ``mx::power`` alone (M1 Pro, 64K to 16M elements), the price
// of leaving MLX's transcendental formula as it is.
//
// A custom ``mx::fast::metal_kernel`` rather than array ops, because the table
// is a function of the NaN / infinity / sign class of each part of the
// exponent, and an ``mx::compile``-fused expression of those tests is not
// reliable on them (see DivMod.h).  The classes are read from the float bits,
// which do not depend on how the library was compiled.
#pragma once

#include <optional>
#include <string>

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/ops.h>

#include "BinaryKernelLoad.h"

namespace lucid::backend {

namespace gpu_complex_pow_detail {

namespace mx = ::mlx::core;

// ``lucid_zero_base_pow(x, y)``: ``x ** y`` for a base ``x`` whose parts are
// both zero (MLX's own test of its zero branch).
//
// With ``c = Re y`` and ``d = Im y`` (``N`` = nan + nan j, ``I`` = inf + nan j,
// ``Z`` = 0 + 0 j; every zero in a result is +0):
//
//   * ``c`` NaN or +-0                          -> N
//   * ``d`` not infinite                        -> Z for ``c > 0``, I for ``c < 0``
//   * ``d`` infinite, ``Re x`` +0 (theta is +-0) -> N for a finite ``c``, else
//                                                 Z for ``c > 0``, I for ``c < 0``
//   * ``d`` infinite, ``Re x`` -0 (theta is +-pi): with ``q`` true when the signs
//     of ``d`` and ``Im x`` are the same,  (a finite ``d`` whose product with pi
//     overflows float32 counts as infinite here)
//                                                 q: Z for ``c > 0``, N for ``c < 0``
//                                                 not q: N for ``c > 0``, I for ``c < 0``
inline const char* header_src() {
    return R"(
inline bool lucid_cpow_is_nan(float x) {
    return (as_type<uint>(x) & 0x7fffffffu) > 0x7f800000u;
}

inline bool lucid_cpow_is_inf(float x) {
    return (as_type<uint>(x) & 0x7fffffffu) == 0x7f800000u;
}

inline bool lucid_cpow_is_zero(float x) {
    return (as_type<uint>(x) & 0x7fffffffu) == 0u;
}

inline bool lucid_cpow_sign(float x) {
    return (as_type<uint>(x) >> 31) != 0u;
}

inline complex64_t lucid_zero_base_pow(complex64_t x, complex64_t y) {
    const float quiet_nan = as_type<float>(0x7fc00000u);
    const float pos_inf = as_type<float>(0x7f800000u);
    const complex64_t n_val = complex64_t(quiet_nan, quiet_nan);
    const complex64_t i_val = complex64_t(pos_inf, quiet_nan);
    const complex64_t z_val = complex64_t(0.0f, 0.0f);

    const float c = y.real;
    const float d = y.imag;
    if (lucid_cpow_is_nan(c) || lucid_cpow_is_zero(c))
        return n_val;
    const bool c_pos = !lucid_cpow_sign(c);
    // At Re x = -0 theta is +-pi, and libc++ forms d * theta in float32: a finite
    // |d| above ~1.083e38 overflows it to +-inf and takes the infinite-d path.
    const bool d_over =
        lucid_cpow_is_inf(d) ||
        (lucid_cpow_sign(x.real) && lucid_cpow_is_inf(metal::fabs(d) * 3.14159274f));
    if (!d_over)
        return c_pos ? z_val : i_val;
    if (!lucid_cpow_sign(x.real))
        return !lucid_cpow_is_inf(c) ? n_val : (c_pos ? z_val : i_val);
    if (lucid_cpow_sign(d) == lucid_cpow_sign(x.imag))
        return c_pos ? z_val : n_val;
    return c_pos ? n_val : i_val;
}
)";
}

// ``out = r`` except where the base is a zero, where it is the ``std::pow``
// value.  ``r`` is ``mx::power(a, b)`` at the broadcast shape; ``a`` and ``b``
// are read through their own strides, so a broadcast or permuted operand costs
// no copy, and ``b`` is read only for a zero base.
//
// Metal flushes float32 subnormals, so a subnormal base compares equal to zero
// here as in MLX's own branch (a known limit, not a change).
inline mx::fast::CustomKernelFunction make_kernel() {
    const std::string body = R"(
        T av = a[elem_to_loc(int64_t(elem), a_shape, a_strides, a_ndim)];
        complex64_t rv = r[elem];
        if (av.real == 0.0f && av.imag == 0.0f) {
            T bv = b[elem_to_loc(int64_t(elem), b_shape, b_strides, b_ndim)];
            rv = lucid_zero_base_pow(av, bv);
        }
        out[elem] = rv;
    )";
    return mx::fast::metal_kernel("lucid_cpow_zero_base", {"a", "b", "r"}, {"out"},
                                  std::string(unary_elem_index_src()) + body, header_src(),
                                  /*ensure_row_contiguous=*/false,
                                  /*atomic_outputs=*/false, mx::CompileOptions{});
}

inline const mx::fast::CustomKernelFunction& patch_kernel() {
    static const auto k = make_kernel();
    return k;
}

}  // namespace gpu_complex_pow_detail

// ``a ** b`` for complex64 operands, broadcast: ``mx::power`` with every lane
// of a zero base answered as ``std::pow`` answers it.
inline ::mlx::core::array gpu_complex_pow(const ::mlx::core::array& a,
                                          const ::mlx::core::array& b_in) {
    namespace mx = ::mlx::core;
    const mx::array b = (b_in.dtype() == a.dtype()) ? b_in : mx::astype(b_in, a.dtype());
    const mx::array r = mx::power(a, b);
    const mx::Shape out_shape = r.shape();
    if (r.size() == 0)
        return r;
    // A 0-d result runs as one element; every other shape as itself.
    const mx::Shape k_shape = out_shape.empty() ? mx::Shape{1} : out_shape;
    const auto launch = metal_launch_1d(r.size());
    auto outs = gpu_complex_pow_detail::patch_kernel()(
        {mx::broadcast_to(a, k_shape), mx::broadcast_to(b, k_shape),
         mx::reshape(mx::contiguous(r), k_shape)},
        {k_shape}, {mx::complex64}, launch.grid, launch.threadgroup, {{"T", mx::complex64}},
        std::nullopt, false, {});
    return out_shape.empty() ? mx::reshape(outs[0], out_shape) : outs[0];
}

}  // namespace lucid::backend
