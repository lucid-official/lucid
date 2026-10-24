// lucid/_C/backend/gpu/DivMod.h
//
// The Metal kernels of the floor-division family: ``//``, ``%`` and
// ``fmod`` on equal-shape or broadcast operands, exactly, in the operands'
// own dtype.  The rule is in ../DivMod.h; these are its steps in one custom
// Metal kernel (LCD-159, LCD-448).
//
// Why a custom ``mx::fast::metal_kernel``:
//
//   * ``mx::compile`` fuses ops with fast-math enabled, turning
//     ``remainder(5, inf)`` into NaN and stripping signed zeros; running
//     the float path unfused launched 6–15 separate kernels per call.
//   * Inside the custom kernel, ``metal::precise::fmod``,
//     ``metal::precise::divide`` and ``metal::precise::floor`` keep IEEE-754
//     semantics, and ``as_type<uint>`` preserves signed zero bit patterns
//     across ``float16``, ``bfloat16`` and ``float32`` in a single launch.
//   * Integer 64-bit division on Apple GPUs has no native hardware divider;
//     when both magnitudes fit in 32 bits the kernel uses 32-bit hardware
//     division, and when either exceeds 2**32 it refines the quotient in
//     four ``float32`` reciprocal steps (~20 bits per step) before at most
//     one final remainder subtraction.
//   * Metal flushes float32 subnormals to zero, so an operand or result
//     below 2**-126 differs from the CPU (``1e-40 % 3e-41`` is NaN).
//
// The integer quotient's ``min / -1`` wraps on Metal (int8 -128 // -1 is
// -128, as in the reference).  The 16-bit floats are computed in float32
// and rounded once.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/ops.h>

#include "../DivMod.h"
#include "BinaryKernelLoad.h"

namespace lucid::backend {

namespace gpu_divmod_detail {

namespace mx = ::mlx::core;

inline const char* float_body_src() {
    return R"(
        float af = static_cast<float>(av);
        float bf = static_cast<float>(bv);
        uint sign_a = as_type<uint>(af) & 0x80000000u;
        uint sign_b = as_type<uint>(bf) & 0x80000000u;
        float abs_a = as_type<float>(as_type<uint>(af) & 0x7fffffffu);
        float abs_b = as_type<float>(as_type<uint>(bf) & 0x7fffffffu);
        float r = (isinf(abs_b) && !isinf(abs_a) && !isnan(abs_a))
                      ? abs_a
                      : metal::precise::fmod(abs_a, abs_b);
        float mod = isnan(r) ? r : as_type<float>(as_type<uint>(r) | sign_a);
        float res = mod;
        if (Mode != 2) {
            bool adjust = (r != 0.0f) && !isnan(r) && !isnan(bf) && (sign_a != sign_b);
            if (Mode == 1) {
                res = adjust ? (mod + bf) : mod;
            } else if (bf == 0.0f) {
                res = metal::precise::divide(af, bf);
            } else if (isnan(mod) || isnan(bf)) {
                res = mod + bf;
            } else if (isinf(bf)) {
                res = (af != 0.0f && (sign_a != sign_b)) ? -1.0f : as_type<float>(sign_a ^ sign_b);
            } else {
                float div = metal::precise::divide(af - mod, bf) - (adjust ? 1.0f : 0.0f);
                float whole = metal::precise::floor(div);
                if (div - whole > 0.5f)
                    whole += 1.0f;
                res = (div == 0.0f) ? as_type<float>(sign_a ^ sign_b) : whole;
            }
        }
        out[elem] = static_cast<T>(res);
    )";
}

inline const char* int_body_src() {
    return R"(
        if (bv == T(0)) {
            out[elem] = (Mode == 0) ? T(-1) : av;
        } else if (bv == T(-1)) {
            uint64_t ua = static_cast<uint64_t>(static_cast<int64_t>(av));
            out[elem] = (Mode == 0) ? static_cast<T>(0ull - ua) : T(0);
        } else {
            int64_t ai = static_cast<int64_t>(av);
            int64_t bi = static_cast<int64_t>(bv);
            int64_t q = 0;
            int64_t r = 0;
            if (sizeof(T) < 8) {
                int32_t q32 = static_cast<int32_t>(ai) / static_cast<int32_t>(bi);
                q = q32;
                r = static_cast<int32_t>(ai) - q32 * static_cast<int32_t>(bi);
            } else {
                uint64_t ua = (ai < 0) ? (0ull - static_cast<uint64_t>(ai)) : static_cast<uint64_t>(ai);
                uint64_t ub = (bi < 0) ? (0ull - static_cast<uint64_t>(bi)) : static_cast<uint64_t>(bi);
                uint64_t uq = 0;
                uint64_t rem = ua;
                if ((ua | ub) <= 0xffffffffull) {
                    uint32_t q32 = static_cast<uint32_t>(ua) / static_cast<uint32_t>(ub);
                    uq = q32;
                    rem = static_cast<uint32_t>(ua) - q32 * static_cast<uint32_t>(ub);
                } else {
                    float inv_b = metal::precise::divide(1.0f, static_cast<float>(ub));
                    for (int step = 0; step < 4 && rem >= ub; ++step) {
                        uint64_t q_est = static_cast<uint64_t>((static_cast<float>(rem) * inv_b) * 0x1.fffff0p-1f);
                        if (q_est == 0)
                            q_est = 1;
                        uint64_t prod = q_est * ub;
                        if (prod <= rem) {
                            rem -= prod;
                            uq += q_est;
                        }
                    }
                    while (rem >= ub) {
                        rem -= ub;
                        ++uq;
                    }
                }
                q = static_cast<int64_t>(((ai < 0) != (bi < 0)) ? (0ull - uq) : uq);
                r = static_cast<int64_t>((ai < 0) ? (0ull - rem) : rem);
            }
            bool adjust = (r != 0) && ((r < 0) != (bi < 0));
            int64_t ans = (Mode == 2) ? r : ((Mode == 0) ? (q - (adjust ? 1 : 0)) : (r + (adjust ? bi : 0)));
            out[elem] = static_cast<T>(ans);
        }
    )";
}

inline mx::fast::CustomKernelFunction make_kernel(const char* name, bool is_float, bool strided) {
    const std::string load = binary_kernel_load_src(strided);
    return mx::fast::metal_kernel(name, {"a", "b"}, {"out"},
                                  load + (is_float ? float_body_src() : int_body_src()), "",
                                  /*ensure_row_contiguous=*/!strided,
                                  /*atomic_outputs=*/false, mx::CompileOptions{});
}

inline const mx::fast::CustomKernelFunction& divmod_kernel(bool is_float, bool strided) {
    static const auto k_fc = make_kernel("lucid_divmod_fc", true, false);
    static const auto k_fs = make_kernel("lucid_divmod_fs", true, true);
    static const auto k_ic = make_kernel("lucid_divmod_ic", false, false);
    static const auto k_is = make_kernel("lucid_divmod_is", false, true);
    return is_float ? (strided ? k_fs : k_fc) : (strided ? k_is : k_ic);
}

inline mx::array as_1d_if_scalar(const mx::array& x) {
    return (x.ndim() == 0) ? mx::reshape(x, {1}) : x;
}

inline mx::array run_contig(const mx::array& a_in,
                            const mx::array& b_in,
                            const mx::Shape& out_shape,
                            int mode_int,
                            int threads,
                            int tg) {
    const mx::Dtype dt = a_in.dtype();
    const bool is_float = mx::issubdtype(dt, mx::floating);
    const mx::Shape k_shape = out_shape.empty() ? mx::Shape{1} : out_shape;
    auto outs = divmod_kernel(is_float, false)({as_1d_if_scalar(a_in), as_1d_if_scalar(b_in)},
                                               {k_shape}, {dt}, {threads, 1, 1}, {tg, 1, 1},
                                               {{"T", dt},
                                                {"Mode", mode_int},
                                                {"AScalar", a_in.size() == 1},
                                                {"BScalar", b_in.size() == 1}},
                                               std::nullopt, false, {});
    return out_shape.empty() ? mx::reshape(outs[0], out_shape) : outs[0];
}

template <DivMod M>
mx::array run(const mx::array& a_in, const mx::array& b_in) {
    const mx::Shape out_shape = (a_in.shape() == b_in.shape())
                                    ? a_in.shape()
                                    : mx::broadcast_shapes(a_in.shape(), b_in.shape());
    const mx::Dtype dt = a_in.dtype();
    const mx::array a = (a_in.shape() == out_shape) ? a_in : mx::broadcast_to(a_in, out_shape);
    const std::size_t n = a.size();
    if (n == 0)
        return mx::zeros(out_shape, dt);
    const int mode_int = (M == DivMod::FloorDiv) ? 0 : ((M == DivMod::Remainder) ? 1 : 2);
    const int threads = static_cast<int>(n);
    const int tg = static_cast<int>(std::min<std::size_t>(n, 256));
    if ((a_in.shape() == out_shape || a_in.size() == 1) &&
        (b_in.shape() == out_shape || b_in.size() == 1))
        return run_contig(a_in, b_in, out_shape, mode_int, threads, tg);
    const mx::array b = (b_in.shape() == out_shape) ? b_in : mx::broadcast_to(b_in, out_shape);
    auto outs = divmod_kernel(mx::issubdtype(dt, mx::floating),
                              true)({a, b}, {out_shape}, {dt}, {threads, 1, 1}, {tg, 1, 1},
                                    {{"T", dt}, {"Mode", mode_int}}, std::nullopt, false, {});
    return outs[0];
}

}  // namespace gpu_divmod_detail

// ``a (op) b`` element-wise; ``a`` and ``b`` have one shape and one dtype.
inline ::mlx::core::array
gpu_divmod(DivMod mode, const ::mlx::core::array& a, const ::mlx::core::array& b) {
    switch (mode) {
    case DivMod::FloorDiv:
        return gpu_divmod_detail::run<DivMod::FloorDiv>(a, b);
    case DivMod::Remainder:
        return gpu_divmod_detail::run<DivMod::Remainder>(a, b);
    case DivMod::Fmod:
        break;
    }
    return gpu_divmod_detail::run<DivMod::Fmod>(a, b);
}

// The floored quotient of float32 ``a`` and ``b``, broadcast.
inline ::mlx::core::array gpu_floor_divide_float32(const ::mlx::core::array& a,
                                                   const ::mlx::core::array& b) {
    return gpu_divmod_detail::run<DivMod::FloorDiv>(a, b);
}

}  // namespace lucid::backend
