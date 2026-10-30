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
//     0.5 → sqrt, -0.5 → 1/sqrt.  Metal's ``*``, divide and sqrt are
//     correctly rounded, so 0, 1, 2, -1 and 0.5 are the exact power, and
//     3, -2 and -0.5 are the reference's float32 bits.
//   * 0.5 and -0.5 keep IEEE ``pow`` at -0 and -inf (+0 / +inf, and
//     +inf / +0), where sqrt and 1/sqrt answer -0, NaN and -inf; the CPU
//     backend answers the same.
//   * -2 squares the reciprocal when ``x * x`` would be subnormal, since Metal
//     flushes it to zero and ``1 / 0`` is inf where the answer is finite.
//   * Every other float32 exponent runs double-float ``exp2(b * log2(|a|))``
//     (<= 1 ulp, 99.97% 0 ulp, and exact on representable powers such as
//     ``0.5 ** 4``, ``2 ** -3``, ``3 ** -4``, ``10 ** 7``, ``3 ** 9``:
//     LCD-557), delegating IEEE special values and negative bases with
//     non-integer exponents to ``metal::precise::pow``.  The 16-bit floats
//     use ``metal::pow`` in their native precision (already <= 1 ulp).
//   * The special exponents are computed in float32 and rounded once, so a
//     16-bit float gets the correctly rounded value.  In the tensor-exponent
//     path, ``metal::simd_any`` guards the special-exponent block so general
//     SIMD groups skip the divide/sqrt path completely (LCD-557).
//   * A scalar exponent in float32 or a special scalar exponent in any
//     floating dtype is a compile-time constant of the kernel (``FixedY``).
//
// Metal flushes float32 subnormals to zero, so a subnormal operand reads as
// a signed zero and a subnormal answer comes back zero.  ``x ** 1`` copies
// the operand and keeps it.
#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

#include <mlx/array.h>
#include <mlx/compile.h>
#include <mlx/fast.h>
#include <mlx/ops.h>

#include "../../core/Half.h"
#include "BinaryKernelLoad.h"
#include "MlxBridge.h"

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

// The compiler predicates these cases for a per-element exponent, so every
// element pays for all of them: the reciprocals of -1, -2 and -0.5 share one
// divide (of ``x``, ``x * x`` or ``sqrt(x)``) and 0.5 and -0.5 one sqrt.
// Four divides and two sqrts cost a general bfloat16 exponent 17%.
inline float lucid_pow_special(float x, float y) {
    float sq = x * x;
    if (!metal::simd_any(y != 2.0f && y != 3.0f))
        return (y == 2.0f) ? sq : sq * x;
    if (!metal::simd_any(y != -1.0f))
        return metal::precise::divide(1.0f, x);
    if (!metal::simd_any(y != 0.5f))
        return (x == -INFINITY) ? INFINITY : metal::precise::sqrt(x) + 0.0f;
    float s = metal::precise::sqrt(x);
    bool tiny = (y == -2.0f) && !(sq >= 0x1p-126f);
    float inv = metal::precise::divide(1.0f, (y == -2.0f) ? (tiny ? x : sq) : ((y == -0.5f) ? s : x));
    if (y == 2.0f)
        return sq;
    if (y == 3.0f)
        return sq * x;
    if (y == -1.0f)
        return inv;
    if (y == -2.0f)
        return tiny ? inv * inv : inv;
    if (y == 0.5f)
        return (x == -INFINITY) ? INFINITY : s + 0.0f;
    if (x == 0.0f)
        return INFINITY;
    return (x == -INFINITY) ? 0.0f : inv;
}
)";
}

// Double-float ``log2(ax)`` for normal positive float32 ``ax`` (LCD-557).
inline const char* f32_log2_src() {
    return R"(
inline float2 lucid_log2_df(float ax) {
    uint ux = as_type<uint>(ax);
    int exp_i = int((ux >> 23) & 0xffu) - 127;
    float m = as_type<float>((ux & 0x007fffffu) | 0x3f800000u);
    if (m >= 1.41421356f) {
        m *= 0.5f;
        exp_i += 1;
    }
    float ef = float(exp_i);

    float u = m - 1.0f;
    float v_hi = m + 1.0f;
    float v_lo = m - (v_hi - 1.0f);
    float rcp = 1.0f / v_hi;
    float t0 = u * rcp;
    float t1 = (metal::fma(-t0, v_hi, u) - t0 * v_lo) * rcp;
    float t_hi = t0 + t1;
    float t_lo = (t0 - t_hi) + t1;

    float s_hi = t_hi * t_hi;
    float s_lo = metal::fma(t_hi, t_hi, -s_hi) + 2.0f * t_hi * t_lo;
    float t3_hi = s_hi * t_hi;
    float t3_lo = metal::fma(s_hi, t_hi, -t3_hi) + s_lo * t_hi + s_hi * t_lo;

    const float c3_hi = 0.9617967009544373f;
    const float c3_lo = -7.028191294579134e-09f;
    const float c5_hi = 0.5770780444145203f;
    const float c5_lo = -2.805797798502579e-08f;
    const float c7 = 0.412198583112f;
    const float c9 = 0.320598897976f;
    const float c11 = 0.262308188524f;
    const float c13 = 0.221953082597f;
    float p7 = metal::fma(s_hi, metal::fma(s_hi, metal::fma(s_hi, c13, c11), c9), c7);
    float t5_hi = t3_hi * s_hi;
    float t5_lo = metal::fma(t3_hi, s_hi, -t5_hi) + t3_lo * s_hi + t3_hi * s_lo;
    float tail5 = metal::fma(t5_hi * s_hi, p7, c5_hi * t5_lo + c5_lo * t5_hi);
    float c5_term = c5_hi * t5_hi;
    float c3_term_hi = c3_hi * t3_hi;
    float c3_term_lo = metal::fma(c3_hi, t3_hi, -c3_term_hi) + c3_hi * t3_lo + c3_lo * t3_hi;
    float poly_hi = c3_term_hi + c5_term;
    float poly_lo = (c5_term - (poly_hi - c3_term_hi)) + c3_term_lo + tail5;

    const float c1_hi = 2.885390043258667f;
    const float c1_lo = 3.273447448356849e-08f;
    float l1_hi = c1_hi * t_hi;
    float l1_lo = metal::fma(c1_hi, t_hi, -l1_hi) + metal::fma(c1_hi, t_lo, c1_lo * t_hi);

    float lm_hi = l1_hi + poly_hi;
    float zm = lm_hi - l1_hi;
    float lm_lo = (l1_hi - (lm_hi - zm)) + (poly_hi - zm) + l1_lo + poly_lo;

    float L0 = ef + lm_hi;
    float z = L0 - ef;
    float L1 = (ef - (L0 - z)) + (lm_hi - z) + lm_lo;
    float L_hi = L0 + L1;
    return float2(L_hi, (L0 - L_hi) + L1);
}
)";
}

// Any other exponent: double-float ``exp2(b * log2(|a|))`` in float32 (<= 1 ulp,
// 99.97% 0 ulp, and exact on representable powers; LCD-557), and ``metal::pow``
// in the 16-bit dtypes (float16 in half, bfloat16 through float32).
inline const char* general_src() {
    return R"(
inline float lucid_pow_native(float a, float b) {
    float ax = metal::fabs(a);
    float rb = metal::rint(b);
    if (!(ax >= 0x1p-126f && ax < INFINITY && metal::isfinite(b)) || (a < 0.0f && rb != b))
        return metal::precise::pow(a, b);
    bool neg = (a < 0.0f) && (metal::rint(0.5f * b) != 0.5f * b);
    float2 L = lucid_log2_df(ax);
    float p0 = b * L.x;
    float p1 = metal::fma(b, L.x, -p0) + b * L.y;
    float p_hi = p0 + p1;
    float p_lo = (p0 - p_hi) + p1;

    if (p_hi > 128.0f)
        return neg ? -INFINITY : INFINITY;
    if (p_hi < -127.0f)
        return neg ? -0.0f : 0.0f;

    float k = metal::rint(p_hi);
    float r_hi = p_hi - k;
    float r = r_hi + p_lo;
    float r_lo = (r_hi - r) + p_lo;

    const float ln2_hi = 0.6931471824645996f;
    const float ln2_lo = -1.904654323148236e-09f;
    const float d2_hi = 0.24022650718688965f;
    const float d2_lo = -2.277889376645144e-10f;
    const float d3_hi = 0.05550410971045494f;
    const float d3_lo = -1.0456333622217998e-09f;
    const float d4 = 0.009618129107628477f;
    const float d5 = 0.0013333558146428443f;
    const float d6 = 0.0001540353039338161f;
    const float d7 = 1.525273380405984e-05f;
    const float d8 = 1.321548679014431e-06f;
    float q4 = metal::fma(r, metal::fma(r, metal::fma(r, metal::fma(r, d8, d7), d6), d5), d4);
    float r2_hi = r * r;
    float r2_lo = metal::fma(r, r, -r2_hi) + 2.0f * r * r_lo;
    float r3_hi = r2_hi * r;
    float r3_lo = metal::fma(r2_hi, r, -r3_hi) + r2_lo * r + r2_hi * r_lo;
    float d3_term_hi = d3_hi * r3_hi;
    float d3_term_lo = metal::fma(d3_hi, r3_hi, -d3_term_hi) + d3_hi * r3_lo + d3_lo * r3_hi +
                       (r2_hi * r2_hi) * q4;
    float d2_term_hi = d2_hi * r2_hi;
    float d2_term_lo = metal::fma(d2_hi, r2_hi, -d2_term_hi) + d2_hi * r2_lo + d2_lo * r2_hi;
    float q2_hi = d2_term_hi + d3_term_hi;
    float q2_lo = (d3_term_hi - (q2_hi - d2_term_hi)) + d2_term_lo + d3_term_lo;

    float t_lin = r * ln2_hi;
    float t_lin_lo = metal::fma(r, ln2_hi, -t_lin) + (r_lo * ln2_hi + r * ln2_lo);
    float s1 = 1.0f + t_lin;
    float e1 = (t_lin - (s1 - 1.0f)) + t_lin_lo;
    float s2 = s1 + q2_hi;
    float e2 = (q2_hi - (s2 - s1)) + e1 + q2_lo;
    float exp2_r = s2 + e2;

    int ki = int(k);
    int k1 = ki >> 1;
    int k2 = ki - k1;
    float s_k1 = as_type<float>(uint(k1 + 127) << 23);
    float s_k2 = as_type<float>(uint(k2 + 127) << 23);
    float mag = (exp2_r * s_k1) * s_k2;
    if (!(mag >= 0x1p-126f))
        mag = 0.0f;
    return neg ? -mag : mag;
}

inline half lucid_pow_native(half a, half b) {
    return metal::pow(a, b);
}

inline bfloat16_t lucid_pow_native(bfloat16_t a, bfloat16_t b) {
    float fa = static_cast<float>(a);
    float fb = static_cast<float>(b);
    // ``exp2``: ``metal::precise::pow(2, -126)`` is 0 on an M1, although
    // ``2**-126`` is bfloat16's smallest normal; the float32 kernel is exact on
    // every power of two (LCD-651).  Other bases keep ``precise::pow``.
    if (fa == 2.0f)
        return static_cast<bfloat16_t>(lucid_pow_native(fa, fb));
    return static_cast<bfloat16_t>(metal::precise::pow(fa, fb));
}
)";
}

inline const char* dispatch_src() {
    return R"(
template <typename T>
inline T lucid_pow_special_in(T a, float x, float y) {
    return static_cast<T>(lucid_pow_special(x, y));
}

// float16's own rsqrt is correctly rounded on all 65536 values, as the
// float32 1/sqrt rounded once is, so it gives the same bits at the cost of
// MLX's half pow; the float32 divide and sqrt cost x ** -0.5 18%.
template <>
inline half lucid_pow_special_in<half>(half a, float x, float y) {
    if (y != -0.5f)
        return static_cast<half>(lucid_pow_special(x, y));
    if (x == 0.0f)
        return half(INFINITY);
    return (x == -INFINITY) ? half(0.0f) : metal::precise::rsqrt(a);
}

template <typename T>
inline T lucid_pow(T a, float y) {
    if (y == 1.0f)
        return a;
    float x = static_cast<float>(a);
    if (y == 0.0f || x == 1.0f)
        return static_cast<T>(1.0f);
    bool is_spec = lucid_pow_is_special(y);
    if (metal::simd_any(is_spec) && is_spec)
        return lucid_pow_special_in(a, x, y);
    return lucid_pow_native(a, static_cast<T>(y));
}
)";
}

inline std::string header_src() {
    return std::string(special_src()) + f32_log2_src() + general_src() + dispatch_src();
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

// The dtypes whose per-element exponent runs the kernel (float32, float16,
// and bfloat16; ``metal::simd_any`` skips the special-exponent block for
// non-special SIMD groups: LCD-557).
inline bool takes_tensor_exponent(mx::Dtype dt) {
    return dt == mx::float32 || dt == mx::float16 || dt == mx::bfloat16;
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
    const auto launch = metal_launch_1d(n);
    if (reads_flat(a, out_shape) && reads_flat(b, out_shape)) {
        const mx::Shape k_shape = out_shape.empty() ? mx::Shape{1} : out_shape;
        const bool a_one = is_one_value(a);
        const bool b_one = is_one_value(b);
        auto outs = pow_kernel(false)({a_one ? one_value(a) : a, b_one ? one_value(b) : b},
                                      {k_shape}, {dt}, launch.grid, launch.threadgroup,
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
        {ab, bb}, {out_shape}, {dt}, launch.grid, launch.threadgroup,
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

// ``a ** b`` for a signed integer ``a``: exact powers by squaring for a
// non-negative ``b`` (``mx::power`` wraps modulo $2^w$), and for a negative
// ``b`` the reference's value, the one ``cpu::ipow_scalar`` computes:
// 1 for a base of 1, $\pm 1$ by the parity of ``b`` for a base of -1, else 0.
// ``mx::power`` alone answers 0 for $1^{-n}$ and $(-1)^{-n}$.  All of it is
// array arithmetic, so a negative exponent costs no host sync, and
// ``mx::compile`` fuses it into one kernel instead of a pass per step.
inline std::vector<mx::array> int_pow_steps(const std::vector<mx::array>& in) {
    const mx::array& a = in[0];
    const mx::array& b = in[1];
    const mx::Dtype dt = a.dtype();
    const mx::array zero(0, dt);
    const mx::array one(1, dt);
    const mx::array minus_one(-1, dt);
    const mx::array odd = mx::not_equal(mx::bitwise_and(b, one), zero);
    const mx::array inverse =
        mx::where(mx::equal(a, one), one,
                  mx::where(mx::equal(a, minus_one), mx::where(odd, minus_one, one), zero));
    return {mx::where(mx::less(b, zero), inverse, mx::power(a, mx::maximum(b, zero)))};
}

inline mx::array int_pow(const mx::array& a, const mx::array& b) {
    static const auto fused = mx::compile(&int_pow_steps, /*shapeless=*/true);
    return fused({a, b})[0];
}

inline bool is_signed_integer(mx::Dtype dt) {
    return dt == mx::int8 || dt == mx::int16 || dt == mx::int32 || dt == mx::int64;
}

}  // namespace gpu_pow_detail

// ``a ** b`` element-wise and broadcast, ``b`` a tensor (or ``a`` a scalar
// base).  float32, float16, and bfloat16 run the kernel above; signed
// integers run ``int_pow``; bool and complex dtypes stay ``mx::power`` (the
// complex principal branch).
inline ::mlx::core::array gpu_pow(const ::mlx::core::array& a, const ::mlx::core::array& b) {
    if (gpu_pow_detail::is_signed_integer(a.dtype()))
        return gpu_pow_detail::int_pow(a, b);
    if (!gpu_pow_detail::takes_tensor_exponent(a.dtype()))
        return ::mlx::core::power(a, b);
    return gpu_pow_detail::run(a, b, {});
}

// ``a ** exponent`` for a scalar exponent.  Every float32 exponent and every
// special exponent in the 16-bit floats is compiled into the kernel; any
// other 16-bit scalar exponent is ``mx::power``.
inline ::mlx::core::array gpu_pow(const ::mlx::core::array& a, double exponent) {
    namespace mx = ::mlx::core;
    const mx::Dtype dt = a.dtype();
    const mx::array b = gpu::mlx_scalar(exponent, dt);
    if (gpu_pow_detail::is_signed_integer(dt))
        return gpu_pow_detail::int_pow(a, b);
    if (!gpu_pow_detail::is_kernel_dtype(dt))
        return mx::power(a, b);
    const float e = gpu_pow_detail::rounded_exponent(exponent, dt);
    if (dt == mx::float32 || gpu_pow_detail::is_special_exponent(e)) {
        const auto bits = std::bit_cast<std::uint32_t>(e);
        return gpu_pow_detail::run(
            a, b, {true, static_cast<int>(bits & 0x7fffffffu), (bits >> 31) != 0u});
    }
    return mx::power(a, b);
}

}  // namespace lucid::backend
