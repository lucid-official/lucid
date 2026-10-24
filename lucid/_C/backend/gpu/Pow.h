// lucid/_C/backend/gpu/Pow.h
//
// The Metal kernel of floating ``pow``: ``x ** y`` element-wise for float32,
// float16 and bfloat16, with a tensor exponent, a scalar exponent or a
// scalar base (LCD-549).
//
// Why a custom ``mx::fast::metal_kernel``:
//
//   * ``mx::power`` is ``metal::pow``, up to 3 ulp off for every float32
//     exponent and wrong where the answer is representable:
//     ``0.5 ** 2`` was 0.24999997 and ``2 ** -1`` 0.49999994.  The
//     exponents the reference answers with one or two arithmetic
//     operations take those operations here, element by element, so a
//     tensor exponent of 2 is ``x * x`` too:
//     0 → 1, 1 → x, 2 → x*x, 3 → x*x*x, -1 → 1/x, -2 → 1/(x*x),
//     0.5 → sqrt, -0.5 → rsqrt.  Metal's ``*``, divide, sqrt and rsqrt are
//     correctly rounded, so 0, 1, 2, -1 and 0.5 are the exact power.
//   * 0.5 and -0.5 keep IEEE ``pow`` at -0 and -inf (+0 / +inf, and
//     +inf / +0), where sqrt and rsqrt answer -0, NaN and -inf; the CPU
//     backend answers the same.
//   * -2 squares the reciprocal when ``x * x`` would be subnormal, since Metal
//     flushes it to zero and ``1 / 0`` is inf where the answer is finite.
//   * Every other exponent is ``metal::pow`` in the operands' own dtype, bit
//     for bit the ``mx::power`` answer, including its IEEE special values;
//     a scalar exponent of that kind skips the kernel and calls
//     ``mx::power`` itself.  That keeps its up-to-3-ulp error (``0.5 ** 4``,
//     ``2 ** -3``, ``10 ** 7``): LCD-557.  An exact path for a power-of-two
//     base raised to an integer cost ``2 ** x`` with a float ``x`` 1.2-1.9x,
//     every element of such a base paying the integer test.
//   * The special exponents are computed in float32 and rounded once, so a
//     16-bit float gets the correctly rounded value.
//   * A scalar special exponent is a compile-time constant of the kernel
//     (``FixedY``), so the kernel is that one operation: testing it per
//     element, the small cases are if-converted and every element computes
//     all of them.
//   * A float16 tensor exponent stays ``mx::power`` (``takes_tensor_exponent``):
//     float16's half pow is within 1 ulp of the special exponents' exact
//     answer (LCD-557), and the per-element dispatch cost it 1.6x.
//
// Metal flushes float32 subnormals to zero, so a subnormal operand reads as
// a signed zero and a subnormal answer comes back zero.  ``x ** 1`` copies
// the operand and keeps it.
#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <string>

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/ops.h>

#include "../../core/Half.h"
#include "BinaryKernelLoad.h"

namespace lucid::backend {

namespace gpu_pow_detail {

namespace mx = ::mlx::core;

// The exponents with their own operation.  The caller has already answered
// ``y == 0``, ``y == 1`` and ``x == 1``.
inline const char* special_src() {
    return R"(
inline bool lucid_pow_is_special(float y) {
    return y == 2.0f || y == 3.0f || y == -1.0f || y == -2.0f || y == 0.5f || y == -0.5f;
}

inline float lucid_pow_special(float x, float y) {
    if (y == 2.0f)
        return x * x;
    if (y == 3.0f)
        return x * x * x;
    if (y == -1.0f)
        return metal::precise::divide(1.0f, x);
    if (y == -2.0f) {
        float sq = x * x;
        float inv = metal::precise::divide(1.0f, x);
        return (sq >= 0x1p-126f) ? metal::precise::divide(1.0f, sq) : inv * inv;
    }
    if (y == 0.5f)
        return (x == -INFINITY) ? INFINITY : metal::precise::sqrt(x) + 0.0f;
    if (x == 0.0f)
        return INFINITY;
    return (x == -INFINITY) ? 0.0f : metal::precise::rsqrt(x);
}
)";
}

// Any other exponent: ``metal::pow`` in the operands' dtype, as
// ``mx::power`` computes it (float16 in half, bfloat16 through float32).
inline const char* general_src() {
    return R"(
inline float lucid_pow_native(float a, float b) {
    return metal::precise::pow(a, b);
}

inline half lucid_pow_native(half a, half b) {
    return metal::pow(a, b);
}

inline bfloat16_t lucid_pow_native(bfloat16_t a, bfloat16_t b) {
    return static_cast<bfloat16_t>(
        metal::precise::pow(static_cast<float>(a), static_cast<float>(b)));
}

template <typename T>
inline T lucid_pow(T a, float y) {
    float x = static_cast<float>(a);
    if (y == 0.0f || x == 1.0f)
        return static_cast<T>(1.0f);
    if (y == 1.0f)
        return a;
    if (lucid_pow_is_special(y))
        return static_cast<T>(lucid_pow_special(x, y));
    return lucid_pow_native(a, static_cast<T>(y));
}
)";
}

inline std::string header_src() {
    return std::string(special_src()) + general_src();
}

// ``FixedY``: the exponent is the constant with float32 magnitude bits
// ``YBits`` and sign ``YNeg``, so the per-element tests fold away; otherwise
// it is ``bv``.  The sign is its own flag because a template value becomes
// part of the kernel's name, where a minus sign does not compile.
inline mx::fast::CustomKernelFunction make_kernel(const char* name, bool strided) {
    const std::string store = R"(
        float y = FixedY ? as_type<float>(static_cast<uint>(YBits) | (YNeg ? 0x80000000u : 0u))
                         : static_cast<float>(bv);
        out[elem] = lucid_pow(av, y);
    )";
    return mx::fast::metal_kernel(name, {"a", "b"}, {"out"},
                                  binary_kernel_load_src(strided) + store, header_src(),
                                  /*ensure_row_contiguous=*/!strided,
                                  /*atomic_outputs=*/false, mx::CompileOptions{});
}

inline const mx::fast::CustomKernelFunction& pow_kernel(bool strided) {
    static const auto k_contig = make_kernel("lucid_pow_c", false);
    static const auto k_strided = make_kernel("lucid_pow_s", true);
    return strided ? k_strided : k_contig;
}

// The dtypes whose per-element exponent runs the kernel.  float16 is not
// one: its own power is cheap enough (``mx::power`` sits at the memory
// floor) that the special operations, which every element pays for once
// the compiler predicates them, cost 1.6x, and a SIMD-uniform skip still
// 1.3x.  float16's half pow is within 1 ulp there (LCD-557); a scalar
// special exponent still takes the kernel in every dtype.
inline bool takes_tensor_exponent(mx::Dtype dt) {
    return dt == mx::float32 || dt == mx::bfloat16;
}

inline bool is_kernel_dtype(mx::Dtype dt) {
    return dt == mx::float32 || dt == mx::float16 || dt == mx::bfloat16;
}

// One element seen through any shape: a 0-d array, or one MLX broadcast
// from a single value (``full`` is), whose strided read would cost a
// 64-bit index per element (``x ** full_like(x, 0.37)`` in float16 took
// 1.7x ``mx::power``).  Strides and ``data_size`` are only known once the
// array is available; before that the kernel reads it as it is.
inline bool is_one_value(const mx::array& x) {
    return x.size() == 1 || (x.is_available() && x.flags().contiguous && x.data_size() == 1);
}

inline mx::array one_value(const mx::array& x) {
    if (x.ndim() == 0)
        return mx::reshape(x, {1});
    if (x.size() == 1)
        return x;
    const auto zeros = mx::Shape(x.ndim(), 0);
    const auto ones = mx::Shape(x.ndim(), 1);
    return mx::reshape(mx::slice(x, zeros, ones), {1});
}

// An operand the contiguous kernel reads as it is: one value, or the
// output's shape laid out row-major.
inline bool reads_flat(const mx::array& x, const mx::Shape& out_shape) {
    return is_one_value(x) || (x.shape() == out_shape && x.flags().row_contiguous);
}

// The exponent compiled into the kernel: ``fixed``, its float32 magnitude
// bits and its sign.
struct Exponent {
    bool fixed = false;
    int bits = 0;
    bool negative = false;
};

inline mx::array run(const mx::array& a, const mx::array& b_in, Exponent ex) {
    const mx::Dtype dt = a.dtype();
    const mx::array b = (b_in.dtype() == dt) ? b_in : mx::astype(b_in, dt);
    const mx::Shape out_shape =
        (a.shape() == b.shape()) ? a.shape() : mx::broadcast_shapes(a.shape(), b.shape());
    std::size_t n = 1;
    for (const auto d : out_shape)
        n *= static_cast<std::size_t>(d);
    if (n == 0)
        return mx::zeros(out_shape, dt);
    const int threads = static_cast<int>(n);
    const int tg = static_cast<int>(std::min<std::size_t>(n, 256));
    if (reads_flat(a, out_shape) && reads_flat(b, out_shape)) {
        const mx::Shape k_shape = out_shape.empty() ? mx::Shape{1} : out_shape;
        const bool a_one = is_one_value(a);
        const bool b_one = is_one_value(b);
        auto outs = pow_kernel(false)({a_one ? one_value(a) : a, b_one ? one_value(b) : b},
                                      {k_shape}, {dt}, {threads, 1, 1}, {tg, 1, 1},
                                      {{"T", dt},
                                       {"AScalar", a_one},
                                       {"BScalar", b_one},
                                       {"FixedY", ex.fixed},
                                       {"YBits", ex.bits},
                                       {"YNeg", ex.negative}},
                                      std::nullopt, false, {});
        return out_shape.empty() ? mx::reshape(outs[0], out_shape) : outs[0];
    }
    const mx::array ab = (a.shape() == out_shape) ? a : mx::broadcast_to(a, out_shape);
    const mx::array bb = (b.shape() == out_shape) ? b : mx::broadcast_to(b, out_shape);
    auto outs = pow_kernel(true)(
        {ab, bb}, {out_shape}, {dt}, {threads, 1, 1}, {tg, 1, 1},
        {{"T", dt}, {"FixedY", ex.fixed}, {"YBits", ex.bits}, {"YNeg", ex.negative}}, std::nullopt,
        false, {});
    return outs[0];
}

// ``e`` as the kernel sees it: rounded to the operand's dtype.
inline float rounded_exponent(double e, mx::Dtype dt) {
    const auto f = static_cast<float>(e);
    if (dt == mx::float16)
        return detail::half_bits_to_float(detail::float_to_half_bits(f));
    if (dt == mx::bfloat16)
        return detail::bfloat16_bits_to_float(detail::float_to_bfloat16_bits(f));
    return f;
}

inline bool is_special_exponent(float e) {
    return e == 0.0f || e == 1.0f || e == 2.0f || e == 3.0f || e == -1.0f || e == -2.0f ||
           e == 0.5f || e == -0.5f;
}

}  // namespace gpu_pow_detail

// ``a ** b`` element-wise and broadcast, ``b`` a tensor (or ``a`` a scalar
// base).  float32 and bfloat16 run the kernel above; float16 (see
// ``takes_tensor_exponent``), integer and complex dtypes stay ``mx::power``
// (exact integer powers by squaring, and the complex principal branch).
inline ::mlx::core::array gpu_pow(const ::mlx::core::array& a, const ::mlx::core::array& b) {
    if (!gpu_pow_detail::takes_tensor_exponent(a.dtype()))
        return ::mlx::core::power(a, b);
    return gpu_pow_detail::run(a, b, {});
}

// ``a ** exponent`` for a scalar exponent.  A special exponent is compiled
// into the kernel; any other is ``mx::power``, which is what the kernel
// would compute.
inline ::mlx::core::array gpu_pow(const ::mlx::core::array& a, double exponent) {
    namespace mx = ::mlx::core;
    const mx::Dtype dt = a.dtype();
    const mx::array b = mx::astype(mx::array(static_cast<float>(exponent)), dt);
    if (!gpu_pow_detail::is_kernel_dtype(dt))
        return mx::power(a, b);
    const float e = gpu_pow_detail::rounded_exponent(exponent, dt);
    if (gpu_pow_detail::is_special_exponent(e)) {
        const auto bits = std::bit_cast<std::uint32_t>(e);
        return gpu_pow_detail::run(
            a, b, {true, static_cast<int>(bits & 0x7fffffffu), (bits >> 31) != 0u});
    }
    return mx::power(a, b);
}

}  // namespace lucid::backend
