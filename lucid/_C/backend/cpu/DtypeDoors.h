// lucid/_C/backend/cpu/DtypeDoors.h
//
// Half (float16 / bfloat16) and narrow-integer (int8 / int16 / int32 / bool)
// widen-reuse-narrow helpers and parallel loop grain constants for CpuBackend.

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../core/Allocator.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Half.h"
#include "../../core/Shape.h"
#include "../IBackend.h"
#include "Parallel.h"
#include "Vforce.h"

namespace lucid {
namespace backend {

// Half conversion lives in ``core/Half.h`` — see the note there.

namespace detail {

// Run a float-only CPU path in single precision on behalf of a half input.
//
// Accelerate has no half kernels, so every entry point below relu, sigmoid,
// clip, variance and friends refused F16 while Metal ran it — the "same
// code, different device" failure this project already treats as a bug.
// Widening once at the door and rounding once on the way out is what the
// astype path does anyway.
//: The two 16-bit float formats.  Accelerate has kernels for neither, so
//: both take the same widen-reuse-narrow route through float32 — they
//: differ only in how the sixteen bits are read.  The predicate itself
//: lives in core beside ``is_floating_point``, because the random
//: generators need it too and they are not part of this backend.
// The layout ``gather`` can walk as (outer, taken, inner) blocks: every
// non-gathered axis of the output is as wide as the input's.  ``extent`` is
// the input's length along the gathered axis.  Empty when any other axis
// differs, which leaves the general coordinate walk to handle it.
struct GatherRows {
    std::size_t outer = 1;
    std::size_t taken = 0;
    std::size_t inner = 1;
    std::size_t extent = 0;

    static std::optional<GatherRows> of(const Shape& in, const Shape& out, int axis) {
        if (in.empty() || in.size() != out.size() || axis < 0 ||
            axis >= static_cast<int>(in.size()))
            return std::nullopt;
        GatherRows rows;
        for (std::size_t d = 0; d < in.size(); ++d) {
            if (static_cast<int>(d) == axis)
                continue;
            if (out[d] != in[d])
                return std::nullopt;
            (static_cast<int>(d) < axis ? rows.outer : rows.inner) *=
                static_cast<std::size_t>(in[d]);
        }
        rows.taken = static_cast<std::size_t>(out[static_cast<std::size_t>(axis)]);
        rows.extent = static_cast<std::size_t>(in[static_cast<std::size_t>(axis)]);
        return rows;
    }
};

inline bool is_half_like(Dtype dt) {
    return is_half_float(dt);
}

inline CpuStorage widen_half(const CpuStorage& in, std::size_t n) {
    auto p = allocate_aligned_bytes(n * sizeof(float), Device::CPU);
    half_or_bfloat16_to_float_bulk(reinterpret_cast<const std::uint16_t*>(in.ptr.get()),
                                   reinterpret_cast<float*>(p.get()), n, in.dtype == Dtype::BF16);
    return CpuStorage{p, n * sizeof(float), Dtype::F32};
}

inline Storage narrow_half(const Storage& wide, std::size_t n, Dtype to) {
    const auto& w = std::get<CpuStorage>(wide);
    auto p = allocate_aligned_bytes(n * sizeof(std::uint16_t), Device::CPU);
    float_to_half_or_bfloat16_bulk(reinterpret_cast<const float*>(w.ptr.get()),
                                   reinterpret_cast<std::uint16_t*>(p.get()), n, to == Dtype::BF16);
    return Storage{CpuStorage{p, n * sizeof(std::uint16_t), to}};
}

// Widen a narrow integer storage to int64, exactly.
//
// The integer counterpart of widen_half, and exact rather than merely
// close: every value of int8 / int16 / int32 / bool fits in an int64, so
// a comparison or a sum computed wide is the same one the narrow type
// would have produced had a kernel existed for it.
inline CpuStorage widen_int(const CpuStorage& in, std::size_t n, Dtype from) {
    auto p = allocate_aligned_bytes(n * sizeof(std::int64_t), Device::CPU);
    auto* dst = reinterpret_cast<std::int64_t*>(p.get());
    const auto* raw = in.ptr.get();
    for (std::size_t i = 0; i < n; ++i) {
        switch (from) {
        case Dtype::I32:
            dst[i] = reinterpret_cast<const std::int32_t*>(raw)[i];
            break;
        case Dtype::I16:
            dst[i] = reinterpret_cast<const std::int16_t*>(raw)[i];
            break;
        case Dtype::I8:
            dst[i] = reinterpret_cast<const std::int8_t*>(raw)[i];
            break;
        default:  // Bool
            dst[i] = reinterpret_cast<const std::uint8_t*>(raw)[i] ? 1 : 0;
            break;
        }
    }
    return CpuStorage{p, n * sizeof(std::int64_t), Dtype::I64};
}

// The inverse.  Truncates on overflow, which cannot happen for the
// reductions that use it: max and min of narrow values stay in range, and
// a sum that overflows the caller's dtype would have overflowed it in a
// native kernel too.
inline Storage narrow_int(const Storage& wide, std::size_t n, Dtype to) {
    const auto& w = std::get<CpuStorage>(wide);
    const std::size_t width = dtype_size(to);
    auto p = allocate_aligned_bytes(n * width, Device::CPU);
    const auto* src = reinterpret_cast<const std::int64_t*>(w.ptr.get());
    auto* raw = p.get();
    for (std::size_t i = 0; i < n; ++i) {
        switch (to) {
        case Dtype::I32:
            reinterpret_cast<std::int32_t*>(raw)[i] = static_cast<std::int32_t>(src[i]);
            break;
        case Dtype::I16:
            reinterpret_cast<std::int16_t*>(raw)[i] = static_cast<std::int16_t>(src[i]);
            break;
        case Dtype::I8:
            reinterpret_cast<std::int8_t*>(raw)[i] = static_cast<std::int8_t>(src[i]);
            break;
        default:  // Bool
            reinterpret_cast<std::uint8_t*>(raw)[i] = src[i] ? 1 : 0;
            break;
        }
    }
    return Storage{CpuStorage{p, n * width, to}};
}

//: The narrow integers, as a set — the ones with no native kernel.
inline bool is_narrow_int(Dtype dt) {
    return dt == Dtype::I32 || dt == Dtype::I16 || dt == Dtype::I8 || dt == Dtype::Bool;
}

//: Every integer dtype, including I64 and bool.
inline bool is_int_like(Dtype dt) {
    return dt == Dtype::I64 || is_narrow_int(dt);
}

// Apply an elementwise function at the caller's integer width.
//
// These ops — relu6, clip, masked_fill — are comparisons and selections,
// not arithmetic, so there is nothing to be gained by widening and a
// dtype to be lost by it.  ``fn`` is a generic lambda instantiated once
// per width; bool is handled as uint8 so a mask stays a mask.
template <typename Fn>
void for_each_int(Dtype dt, const std::byte* in, std::byte* out, std::size_t n, Fn fn) {
    auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* ip = reinterpret_cast<const T*>(in);
        T* op = reinterpret_cast<T*>(out);
        for (std::size_t i = 0; i < n; ++i)
            op[i] = fn(ip[i]);
    };
    switch (dt) {
    case Dtype::I64:
        run(std::int64_t{});
        break;
    case Dtype::I32:
        run(std::int32_t{});
        break;
    case Dtype::I16:
        run(std::int16_t{});
        break;
    case Dtype::I8:
        run(std::int8_t{});
        break;
    default:  // Bool
        run(std::uint8_t{});
        break;
    }
}

// ``out[i] = mask[i] ? fill : in[i]`` at the caller's integer width.
inline void for_each_int_masked(Dtype dt,
                                const std::byte* in,
                                const std::uint8_t* mask,
                                std::byte* out,
                                std::size_t n,
                                std::int64_t fill) {
    auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* __restrict ip = reinterpret_cast<const T*>(in);
        const std::uint8_t* __restrict mp = mask;
        T* __restrict op = reinterpret_cast<T*>(out);
        const T f = static_cast<T>(fill);
        // Load, then select: a conditional load is a branch on the mask.
        for (std::size_t i = 0; i < n; ++i) {
            const T v = ip[i];
            op[i] = mp[i] ? f : v;
        }
    };
    switch (dt) {
    case Dtype::I64:
        run(std::int64_t{});
        break;
    case Dtype::I32:
        run(std::int32_t{});
        break;
    case Dtype::I16:
        run(std::int16_t{});
        break;
    case Dtype::I8:
        run(std::int8_t{});
        break;
    default:  // Bool
        run(std::uint8_t{});
        break;
    }
}

// Widen a storage to int64 iff it is a narrow integer; pass anything
// else through.  The integer twin of as_f32, and dtype-driven for the
// same reason: a call site hands it every operand and the floats, masks
// and indices travel through unchanged.
inline Storage as_i64(const Storage& s) {
    const auto* c = std::get_if<CpuStorage>(&s);
    if (c == nullptr || !is_narrow_int(c->dtype))
        return s;
    return Storage{widen_int(*c, c->nbytes / dtype_size(c->dtype), c->dtype)};
}

// The inverse, back to a stated dtype.  Unlike back_to_f16 the target
// cannot be inferred from the result — every narrow integer widens to
// the same I64 — so the caller names it.
inline Storage back_to_int(const Storage& wide, Dtype to) {
    const auto* c = std::get_if<CpuStorage>(&wide);
    if (c == nullptr || c->dtype != Dtype::I64 || !is_narrow_int(to))
        return wide;
    return narrow_int(wide, c->nbytes / sizeof(std::int64_t), to);
}

// Widen any integer storage to double; pass anything else through.
// Used where the only kernel is a BLAS one — see matmul.
inline Storage as_f64(const Storage& s) {
    const auto* c = std::get_if<CpuStorage>(&s);
    if (c == nullptr || !is_int_like(c->dtype))
        return s;
    const std::size_t n = c->nbytes / dtype_size(c->dtype);
    auto p = allocate_aligned_bytes(n * sizeof(double), Device::CPU);
    auto* dst = reinterpret_cast<double*>(p.get());
    const auto* raw = c->ptr.get();
    for (std::size_t i = 0; i < n; ++i) {
        switch (c->dtype) {
        case Dtype::I64:
            dst[i] = static_cast<double>(reinterpret_cast<const std::int64_t*>(raw)[i]);
            break;
        case Dtype::I32:
            dst[i] = reinterpret_cast<const std::int32_t*>(raw)[i];
            break;
        case Dtype::I16:
            dst[i] = reinterpret_cast<const std::int16_t*>(raw)[i];
            break;
        case Dtype::I8:
            dst[i] = reinterpret_cast<const std::int8_t*>(raw)[i];
            break;
        default:  // Bool
            dst[i] = reinterpret_cast<const std::uint8_t*>(raw)[i] ? 1.0 : 0.0;
            break;
        }
    }
    return Storage{CpuStorage{p, n * sizeof(double), Dtype::F64}};
}

//: How a double result becomes an integer.  The two callers want
//: different things and the difference is visible in the answers.
//:
//: ``Nearest`` is for arithmetic whose exact result *is* an integer and
//: which was only routed through double because Accelerate had no
//: integer kernel — an integer matmul, say.  There the double may sit a
//: few ulp off the true integer, and rounding to nearest repairs that;
//: truncating would turn 5 into 4.
//:
//: ``TowardZero`` is for a result that is genuinely fractional and is
//: being narrowed because the caller asked for an integer dtype — the
//: mean of an integer pooling window.  That is a *cast*, and the
//: framework's float-to-integer cast truncates: the reference pools
//: [3,1,5,9] to 4, not 5, and Metal's own cast already agreed with it
//: while the CPU rounded and disagreed.
enum class IntRounding { Nearest, TowardZero };

// Convert a double result back to an integer dtype.
inline Storage
f64_to_int(const Storage& wide, std::size_t n, Dtype to, IntRounding how = IntRounding::Nearest) {
    const auto& w = std::get<CpuStorage>(wide);
    const std::size_t width = dtype_size(to);
    auto p = allocate_aligned_bytes(n * width, Device::CPU);
    const auto* src = reinterpret_cast<const double*>(w.ptr.get());
    auto* raw = p.get();
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t v = how == IntRounding::Nearest
                                   ? static_cast<std::int64_t>(std::llround(src[i]))
                                   : static_cast<std::int64_t>(src[i]);
        switch (to) {
        case Dtype::I64:
            reinterpret_cast<std::int64_t*>(raw)[i] = v;
            break;
        case Dtype::I32:
            reinterpret_cast<std::int32_t*>(raw)[i] = static_cast<std::int32_t>(v);
            break;
        case Dtype::I16:
            reinterpret_cast<std::int16_t*>(raw)[i] = static_cast<std::int16_t>(v);
            break;
        case Dtype::I8:
            reinterpret_cast<std::int8_t*>(raw)[i] = static_cast<std::int8_t>(v);
            break;
        default:  // Bool
            reinterpret_cast<std::uint8_t*>(raw)[i] = v ? 1 : 0;
            break;
        }
    }
    return Storage{CpuStorage{p, n * width, to}};
}

// The inverse of ``as_f64``, dtype-driven in the same way ``back_to_f16``
// is: an F64 result rounds back to the caller's integer width, and a
// result that is already an index or a mask passes through as it is.  A
// pooling call returns both at once, which is exactly why this keys off
// the storage rather than the position.
inline Storage f64_back_to_int(const Storage& s, Dtype to, IntRounding how = IntRounding::Nearest) {
    const auto* c = std::get_if<CpuStorage>(&s);
    if (c == nullptr || c->dtype != Dtype::F64 || !is_int_like(to))
        return s;
    return f64_to_int(s, c->nbytes / sizeof(double), to, how);
}

inline std::vector<Storage>
f64_back_to_int(const std::vector<Storage>& v, Dtype to, IntRounding how = IntRounding::Nearest) {
    std::vector<Storage> out;
    out.reserve(v.size());
    for (const auto& s : v)
        out.push_back(f64_back_to_int(s, to, how));
    return out;
}

// Widen a storage to float iff it is half; pass anything else through
// untouched.  Dtype-driven rather than position-driven, so a call site can
// hand it every operand it has and the index / mask / condition tensors
// travel through unchanged — there is no list of "which argument is the
// float one" to get wrong.
inline Storage as_f32(const Storage& s) {
    const auto* c = std::get_if<CpuStorage>(&s);
    if (c == nullptr || !is_half_like(c->dtype))
        return s;
    return Storage{widen_half(*c, c->nbytes / sizeof(std::uint16_t))};
}

// The same for an optional operand — an attention mask, a class weight.
// Those used to be handed through raw while their siblings widened, and the
// float kernel then read half bytes as floats.  ``hold`` owns the copy.
inline const Storage* as_f32(const Storage* s, std::optional<Storage>& hold) {
    if (s == nullptr)
        return nullptr;
    hold = as_f32(*s);
    return &*hold;
}

// The inverse, equally dtype-driven: a float result rounds back to half, and
// a result that is a mask (bool) or an index (int64) is returned as it is.
inline Storage back_to_f16(const Storage& s, Dtype to) {
    const auto* c = std::get_if<CpuStorage>(&s);
    if (c == nullptr || c->dtype != Dtype::F32)
        return s;
    return narrow_half(s, c->nbytes / sizeof(float), to);
}

inline std::vector<Storage> back_to_f16(const std::vector<Storage>& v, Dtype to) {
    std::vector<Storage> out;
    out.reserve(v.size());
    for (const auto& s : v)
        out.push_back(back_to_f16(s, to));
    return out;
}

inline std::pair<Storage, Storage> back_to_f16(const std::pair<Storage, Storage>& p, Dtype to) {
    return {back_to_f16(p.first, to), back_to_f16(p.second, to)};
}

inline StoragePair back_to_f16(const StoragePair& p, Dtype to) {
    return {back_to_f16(p.first, to), back_to_f16(p.second, to)};
}

}  // namespace detail

// Elements of a scalar transcendental loop (``std::erf``, ``std::exp``) worth
// a core of their own: at ~5 ns an element, 16k of them are ~80 us, far past
// the few microseconds a dispatch costs.  See cpu::parallel_for.
inline constexpr std::size_t kTranscendentalGrain = 16384;

// Elements of a memory-bound loop (a comparison, a select) worth a core: at
// ~0.4 ns an element, 64k of them are ~25 us, still well past a dispatch.
inline constexpr std::size_t kStreamGrain = 65536;

// Elements per vForce call inside a parallel chunk: the tile lives on the
// stack and stays in L1 (8 KiB of double) between the passes over it.
inline constexpr std::size_t kTanhTile = 1024;

namespace detail {

// ``out[i] = mask[i] ? value : in[i]``, without a branch.
//
// Written as a conditional, the element was loaded only when the mask was
// clear, so the compiler branched on the mask — and a data mask (``x > 0``)
// is a coin flip: 150 us for 64k floats, thirty times an add.  Loading first
// and selecting makes it a vector blend.
template <typename T>
inline void select_fill(const T* in, const std::uint8_t* mask, T value, T* out, std::size_t n) {
    cpu::parallel_for(n, kStreamGrain, [&](std::size_t lo, std::size_t hi) {
        // Locals, not the closure's references: a store through ``y`` could
        // alias a captured ``value``, which would keep the loop scalar.
        const T* __restrict x = in;
        const std::uint8_t* __restrict m = mask;
        T* __restrict y = out;
        const T fill = value;
        for (std::size_t i = lo; i < hi; ++i) {
            const T v = x[i];
            y[i] = m[i] ? fill : v;
        }
    });
}

// IEEE ``pow(v, 0.5)`` from ``s = sqrt(v)``: sqrt, except at -0 (sqrt keeps
// the sign, the power is +0) and -inf (sqrt is NaN, the power +inf).
template <typename T>
inline T sqrt_power_of(T v, T s) {
    constexpr T kInf = std::numeric_limits<T>::infinity();
    return v == -kInf ? kInf : s + T(0);
}

template <typename T>
inline T sqrt_power(T v) {
    return sqrt_power_of(v, std::sqrt(v));
}

// IEEE ``pow(v, -0.5)`` from ``inv = 1 / sqrt(v)``, the reference's formula,
// except at -0 (that is -inf, the power +inf) and -inf (NaN, the power +0).
template <typename T>
inline T rsqrt_power_of(T v, T inv) {
    constexpr T kInf = std::numeric_limits<T>::infinity();
    return v == T(0) ? kInf : (v == -kInf ? T(0) : inv);
}

template <typename T>
inline T rsqrt_power(T v) {
    return rsqrt_power_of(v, T(1) / std::sqrt(v));
}

// Calls ``f`` with the element operation of an exponent the reference
// answers with one or two IEEE operations, and returns false for any other
// exponent.  0, 1, 2, -1 and 0.5 are one correctly rounded operation, so
// the exact power; 3, -2 and -0.5 are the reference's own formulas
// (``x*x*x``, ``1/(x*x)``, ``1/sqrt(x)``), so its bits.
template <typename T, typename F>
inline bool with_exact_power(T e, F&& f) {
    if (e == T(2))
        f([](T v) { return v * v; });
    else if (e == T(1))
        f([](T v) { return v; });
    else if (e == T(0))
        f([](T) { return T(1); });
    else if (e == T(-1))
        f([](T v) { return T(1) / v; });
    else if (e == T(0.5))
        f([](T v) { return sqrt_power(v); });
    else if (e == T(3))
        f([](T v) { return v * v * v; });
    else if (e == T(-2))
        f([](T v) { return T(1) / (v * v); });
    else if (e == T(-0.5))
        f([](T v) { return rsqrt_power(v); });
    else
        return false;
    return true;
}

template <typename T>
inline void vpow(const T* x, const T* y, T* out, std::size_t n) {
    if constexpr (std::is_same_v<T, float>)
        cpu::vpow_f32(x, y, out, n);
    else
        cpu::vpow_f64(x, y, out, n);
}

// ``out = x ** e`` for a scalar exponent.
//
// vForce's pow is within 1 ulp, not exact (24-30% of float32 answers are 1
// ulp off, for every exponent: ``x ** 1`` was not ``x``), and ~8 ns an
// element; backward reaches it for every power (``d/dx x**2`` is
// ``2 * x**1``).  The exponents of ``with_exact_power`` take their
// operations instead.  Any other exponent runs vForce in L1 tiles across
// cores.
template <typename T>
inline void pow_by_scalar(const T* in, T e, T* out, std::size_t n) {
    auto each = [&](auto f) {
        cpu::parallel_for(n, kStreamGrain, [&](std::size_t lo, std::size_t hi) {
            const T* __restrict x = in;
            T* __restrict y = out;
            for (std::size_t i = lo; i < hi; ++i)
                y[i] = f(x[i]);
        });
    };
    if (with_exact_power(e, each))
        return;
    cpu::parallel_for(n, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
        T expo[kTanhTile];
        std::fill_n(expo, std::min(kTanhTile, hi - lo), e);
        for (std::size_t s = lo; s < hi; s += kTanhTile)
            vpow(in + s, expo, out + s, std::min(kTanhTile, hi - s));
    });
}

// Whether any of ``y[0, n)`` can be an exponent of ``with_exact_power``.
// Every one of them has a mantissa that is zero below its top bit (3 is
// 1.5 * 2), so a tile of general exponents costs one AND and min an element.
template <typename T>
inline bool may_hold_exact_exponent(const T* y, std::size_t n) {
    using Bits = std::conditional_t<std::is_same_v<T, float>, std::uint32_t, std::uint64_t>;
    constexpr Bits kLow = (Bits{1} << (std::numeric_limits<T>::digits - 2)) - 1;
    const T* __restrict ys = y;
    Bits low = kLow;
    for (std::size_t i = 0; i < n; ++i)
        low = std::min(low, static_cast<Bits>(std::bit_cast<Bits>(ys[i]) & kLow));
    return low == 0;
}

template <typename T>
inline bool all_equal(const T* y, std::size_t n) {
    const T* __restrict ys = y;
    const T first = ys[0];
    bool same = true;
    for (std::size_t i = 1; i < n; ++i)
        same &= ys[i] == first;
    return same;
}

// ``out[i]`` rewritten with ``with_exact_power``'s operation for each
// element whose exponent has one; the others keep vForce's answer.  It
// selects instead of branching, so it vectorises (a branch per element
// tripled the time of a mixed exponent tensor), and the three reciprocals
// share one divide: ``1 / d`` with ``d`` the element's ``x``, ``x * x`` or
// ``sqrt(x)``.
template <typename T>
inline void exact_powers(const T* x, const T* y, T* out, std::size_t n) {
    const T* __restrict xs = x;
    const T* __restrict ys = y;
    T* __restrict o = out;
    for (std::size_t i = 0; i < n; ++i) {
        const T e = ys[i];
        const T v = xs[i];
        const T sq = v * v;
        const T s = std::sqrt(v);
        const T inv = T(1) / ((e == T(-2)) ? sq : ((e == T(-0.5)) ? s : v));
        T r = o[i];
        r = (e == T(2)) ? sq : r;
        r = (e == T(1)) ? v : r;
        r = (e == T(0)) ? T(1) : r;
        r = (e == T(-1) || e == T(-2)) ? inv : r;
        r = (e == T(0.5)) ? sqrt_power_of(v, s) : r;
        r = (e == T(3)) ? sq * v : r;
        r = (e == T(-0.5)) ? rsqrt_power_of(v, inv) : r;
        o[i] = r;
    }
}

// ``out = x ** y`` over one L1 tile of a tensor exponent: vForce, with the
// exponents of ``with_exact_power`` taking their operations.  A tile of one
// exponent (``x ** full_like(x, e)``) takes that exponent's operation, or
// vForce alone, so a uniform 4 or 1.5, which the mantissa test lets through,
// pays no select pass.
template <typename T>
inline void pow_tile(const T* x, const T* y, T* out, std::size_t n) {
    if (!may_hold_exact_exponent(y, n)) {
        vpow(x, y, out, n);
        return;
    }
    auto each = [&](auto f) {
        const T* __restrict xs = x;
        T* __restrict o = out;
        for (std::size_t i = 0; i < n; ++i)
            o[i] = f(xs[i]);
    };
    if (all_equal(y, n)) {
        if (!with_exact_power(y[0], each))
            vpow(x, y, out, n);
        return;
    }
    vpow(x, y, out, n);
    exact_powers(x, y, out, n);
}

// Inclusive scan along ``axis`` for ``cumsum`` and ``cumprod``.  An empty scan
// axis returns an empty buffer without reading ``src[0]``.
template <typename Op>
inline Storage
scan_axis(const Storage& a, const Shape& shape, int axis, Dtype dt, const char* op_name, Op op) {
    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t nb = cs.nbytes;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const std::size_t ax = static_cast<std::size_t>(axis);
    const std::size_t L = static_cast<std::size_t>(shape[ax]);
    if (L == 0)
        return Storage{CpuStorage{ptr, nb, dt}};
    std::size_t outer = 1, inner = 1;
    for (std::size_t d = 0; d < shape.size(); ++d) {
        if (d < ax)
            outer *= static_cast<std::size_t>(shape[d]);
        else if (d > ax)
            inner *= static_cast<std::size_t>(shape[d]);
    }

    auto step = [&]<typename T>() {
        auto* dst = reinterpret_cast<T*>(ptr.get());
        const auto* src = reinterpret_cast<const T*>(cs.ptr.get());
        for (std::size_t o = 0; o < outer; ++o)
            for (std::size_t j = 0; j < inner; ++j) {
                T acc = src[(o * L) * inner + j];
                dst[(o * L) * inner + j] = acc;
                for (std::size_t k = 1; k < L; ++k) {
                    acc = op(acc, src[(o * L + k) * inner + j]);
                    dst[(o * L + k) * inner + j] = acc;
                }
            }
    };

    switch (dt) {
    case Dtype::F32:
        step.template operator()<float>();
        break;
    case Dtype::F64:
        step.template operator()<double>();
        break;
    case Dtype::I32:
        step.template operator()<std::int32_t>();
        break;
    case Dtype::I64:
        step.template operator()<std::int64_t>();
        break;
    case Dtype::C64:
        step.template operator()<std::complex<float>>();
        break;
    case Dtype::C128:
        step.template operator()<std::complex<double>>();
        break;
    default:
        ErrorBuilder(op_name).not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

}  // namespace detail
}  // namespace backend
}  // namespace lucid
