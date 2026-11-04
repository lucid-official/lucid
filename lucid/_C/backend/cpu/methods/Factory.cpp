// lucid/_C/backend/cpu/methods/Factory.cpp
// Out-of-line CpuBackend method definitions for the Factory family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

Storage CpuBackend::from_cpu(CpuStorage cpu, const Shape&) {
    return Storage{std::move(cpu)};
}

Storage CpuBackend::ones(const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(ones(shape, Dtype::F32), dt);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    fill_ones(ptr.get(), n, dt);
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::clone(const Storage& src, const Shape& shape, Dtype dt) {
    const auto& cs = std::get<CpuStorage>(src);
    std::size_t nb = cs.nbytes;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    std::memcpy(ptr.get(), cs.ptr.get(), nb);
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::contiguous(const Storage& src,
                               const Shape& shape,
                               const Stride& stride,
                               std::size_t storage_offset,
                               bool already_contiguous,
                               Dtype dt) {
    // SharedStorage (MetalKernelRunner output) is CPU-readable via cpu_view().
    // Create a temporary CpuStorage view and proceed normally.
    CpuStorage tmp_view;
    const CpuStorage* cs_ptr = nullptr;
    if (storage_is_metal_shared(src)) {
        tmp_view = std::get<SharedStorage>(src).cpu_view();
        cs_ptr = &tmp_view;
    } else {
        cs_ptr = &std::get<CpuStorage>(src);
    }
    const auto& cs = *cs_ptr;
    const std::size_t elem = dtype_size(dt);
    const std::size_t n = shape_numel(shape);
    const std::size_t nbytes = n * elem;
    auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
    if (nbytes == 0)
        return Storage{CpuStorage{ptr, nbytes, dt}};

    if (already_contiguous && storage_offset == 0) {
        std::memcpy(ptr.get(), cs.ptr.get(), nbytes);
        return Storage{CpuStorage{ptr, nbytes, dt}};
    }

    // Axes merged, runs memcpy'd, a 2-D transpose tiled (core/StridedCopy.h)
    // — this walked every element with a coordinate step and a memcpy.
    strided::pack(cs.ptr.get() + storage_offset, ptr.get(), shape, stride, elem);
    return Storage{CpuStorage{ptr, nbytes, dt}};
}

std::vector<Storage> CpuBackend::meshgrid(const std::vector<Storage>& xs,
                                          const Shape& out_shape,
                                          Dtype dt,
                                          bool indexing_xy) {
    const std::size_t N = xs.size();
    const std::size_t total = shape_numel(out_shape);
    const std::size_t elem = dtype_size(dt);
    std::vector<Storage> result;
    result.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        const auto& cv = std::get<CpuStorage>(xs[i]);
        std::size_t out_nbytes = total * elem;
        auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
        std::size_t carry_axis = i;
        if (indexing_xy && N >= 2 && i < 2)
            carry_axis = 1 - i;

        std::vector<std::int64_t> coord(N, 0);
        for (std::size_t f = 0; f < total; ++f) {
            const std::int64_t k = coord[carry_axis];
            std::memcpy(ptr.get() + f * elem, cv.ptr.get() + static_cast<std::size_t>(k) * elem,
                        elem);
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(N) - 1; d >= 0; --d) {
                if (++coord[static_cast<std::size_t>(d)] < out_shape[static_cast<std::size_t>(d)]) {
                    break;
                }
                coord[static_cast<std::size_t>(d)] = 0;
            }
        }
        result.push_back(Storage{CpuStorage{ptr, out_nbytes, dt}});
    }
    return result;
}

Storage CpuBackend::astype(const Storage& a, const Shape& shape, Dtype src_dt, Dtype dst_dt) {
    const auto& ca = std::get<CpuStorage>(a);
    const std::size_t n = shape_numel(shape);
    const std::size_t dsz = dtype_size(dst_dt);
    auto out_ptr = allocate_aligned_bytes(n * dsz, Device::CPU);

    using I8 = std::int8_t;
    using I16 = std::int16_t;
    using I32 = std::int32_t;
    using I64 = std::int64_t;

    // Generic static_cast loop — used for every (From, To) pair where both
    // sides have a 1-to-1 C++ scalar type.  ``bool`` is supported as From
    // and To by C++ static_cast rules (0 / 1 conversion at both ends).
    auto run = [&]<typename From, typename To>() {
        // Restrict locals, the count among them: through the closure, a
        // store of int64 or of a byte may alias ``n`` (same-width and
        // char aliasing), so the bound was reloaded every element and
        // the loop stayed scalar — bool to int64 cost three adds.
        const From* __restrict src = reinterpret_cast<const From*>(ca.ptr.get());
        To* __restrict dst = reinterpret_cast<To*>(out_ptr.get());
        const std::size_t n = shape_numel(shape);
        if constexpr (std::is_floating_point_v<From> && std::is_integral_v<To> &&
                      !std::is_same_v<To, bool>) {
            // Float to integer saturates and sends NaN to 0 — what the
            // hardware conversion does and what the other device matches
            // (unit/ops/test_float_to_int_casts.py).  Spelled out because
            // a ``static_cast`` of NaN, ±inf or anything out of range is
            // undefined: the UBSan build halted on it.
            //
            // Clamp first, convert, then select: no branch, so the loop
            // is a vector one (an if-chain per element was 14 times an
            // add).  ``top`` is the largest From that converts in range —
            // To's max itself when representable, else the From below
            // 2^(bits-1) — and the selects restore max and 0 exactly.
            constexpr auto lo = static_cast<From>(std::numeric_limits<To>::lowest());
            constexpr auto hi = static_cast<From>(std::numeric_limits<To>::max());
            const From top = hi < -lo ? hi : std::nextafter(-lo, From(0));
            for (std::size_t i = 0; i < n; ++i) {
                const From v = src[i];
                // fmax answers lo for a NaN, so c is always in range.
                const From c = std::fmin(std::fmax(v, lo), top);
                To r = static_cast<To>(c);
                r = v >= hi ? std::numeric_limits<To>::max() : r;
                dst[i] = v != v ? To(0) : r;
            }
        } else {
            for (std::size_t i = 0; i < n; ++i)
                dst[i] = static_cast<To>(src[i]);
        }
    };

#define CPU_CAST(F, T) run.template operator()<F, T>()

    // F16 has no ``static_cast``-able host scalar, so the main table below
    // cannot express it.  The legs are converted explicitly here.
    //
    // The previous version delegated both legs back into ``astype`` and
    // claimed the recursion would land in the main switch — it never did:
    // every recursive call still had F16 on one side, so it re-entered this
    // same branch and the process died of stack exhaustion.  CPU F16 casts
    // (``.half()``, ``.to(float16)``, and the way back) therefore SIGSEGV'd
    // in both directions and had never worked.  Recurse only on the
    // F16-free leg, which the main table does handle.
    // Both 16-bit float formats take this route, and for the same
    // reason: neither has a host scalar the Cartesian table below can
    // ``static_cast`` through.  bfloat16 joined F16 here rather than
    // getting its own branch because the shape is identical — decode to
    // float32, recurse on the leg that has no 16-bit float in it, encode
    // — and two copies of that would be two places to get the recursion
    // wrong.
    const bool src_half = detail::is_half_like(src_dt);
    const bool dst_half = detail::is_half_like(dst_dt);
    if (src_dt != dst_dt && (src_half || dst_half)) {
        if (src_half) {
            if (dst_dt == Dtype::F32) {
                detail::half_or_bfloat16_to_float_bulk(
                    reinterpret_cast<const std::uint16_t*>(ca.ptr.get()),
                    reinterpret_cast<float*>(out_ptr.get()), n, src_dt == Dtype::BF16);
                return Storage{CpuStorage{out_ptr, n * dsz, Dtype::F32}};
            }
            return astype(Storage{detail::widen_half(ca, n)}, shape, Dtype::F32, dst_dt);
        }
        Storage as_f32 = (src_dt == Dtype::F32) ? a : astype(a, shape, src_dt, Dtype::F32);
        const auto& cf = std::get<CpuStorage>(as_f32);
        detail::float_to_half_or_bfloat16_bulk(reinterpret_cast<const float*>(cf.ptr.get()),
                                               reinterpret_cast<std::uint16_t*>(out_ptr.get()), n,
                                               dst_dt == Dtype::BF16);
        return Storage{CpuStorage{out_ptr, n * dsz, dst_dt}};
    }

    // Cast complex lanes separately. Routing C128 through F32 loses
    // precision; routing complex-to-complex through a real tensor loses
    // its imaginary component entirely. Recursion below is real-only.
    if (src_dt != dst_dt && (is_complex(src_dt) || is_complex(dst_dt))) {
        if (is_complex(src_dt) && dst_dt == Dtype::Bool) {
            auto truth = [&]<typename T>() {
                const auto* values = reinterpret_cast<const std::complex<T>*>(ca.ptr.get());
                auto* output = reinterpret_cast<std::uint8_t*>(out_ptr.get());
                for (std::size_t i = 0; i < n; ++i)
                    output[i] = values[i].real() != T{} || values[i].imag() != T{};
            };
            if (src_dt == Dtype::C128)
                truth.template operator()<double>();
            else
                truth.template operator()<float>();
            return Storage{CpuStorage{out_ptr, n * dsz, dst_dt}};
        }
        const auto src_lane = is_complex(src_dt) ? real_lane_of(src_dt) : src_dt;
        Storage re = is_complex(src_dt) ? complex_real(a, shape) : a;
        if (!is_complex(dst_dt))
            return src_lane == dst_dt ? re : astype(re, shape, src_lane, dst_dt);
        const auto dst_lane = real_lane_of(dst_dt);
        Storage im = is_complex(src_dt) ? complex_imag(a, shape) : zeros(shape, src_lane);
        if (src_lane != dst_lane) {
            re = astype(re, shape, src_lane, dst_lane);
            im = astype(im, shape, src_lane, dst_lane);
        }
        return complex_combine(re, im, shape);
    }

    // Main Cartesian dispatch.  Outer = src, inner = dst.  Every
    // {F32, F64, I8, I16, I32, I64, Bool} × {same} pair is enumerated;
    // diagonal (src == dst) is handled at the op layer (no-op share).
    bool ok = true;
#define HANDLE_DST(F)                                                                              \
    switch (dst_dt) {                                                                              \
    case Dtype::F32:                                                                               \
        CPU_CAST(F, float);                                                                        \
        break;                                                                                     \
    case Dtype::F64:                                                                               \
        CPU_CAST(F, double);                                                                       \
        break;                                                                                     \
    case Dtype::I8:                                                                                \
        CPU_CAST(F, I8);                                                                           \
        break;                                                                                     \
    case Dtype::I16:                                                                               \
        CPU_CAST(F, I16);                                                                          \
        break;                                                                                     \
    case Dtype::I32:                                                                               \
        CPU_CAST(F, I32);                                                                          \
        break;                                                                                     \
    case Dtype::I64:                                                                               \
        CPU_CAST(F, I64);                                                                          \
        break;                                                                                     \
    case Dtype::Bool:                                                                              \
        CPU_CAST(F, bool);                                                                         \
        break;                                                                                     \
    default:                                                                                       \
        ok = false;                                                                                \
    }

    switch (src_dt) {
    case Dtype::F32:
        HANDLE_DST(float);
        break;
    case Dtype::F64:
        HANDLE_DST(double);
        break;
    case Dtype::I8:
        HANDLE_DST(I8);
        break;
    case Dtype::I16:
        HANDLE_DST(I16);
        break;
    case Dtype::I32:
        HANDLE_DST(I32);
        break;
    case Dtype::I64:
        HANDLE_DST(I64);
        break;
    case Dtype::Bool:
        HANDLE_DST(bool);
        break;
    default:
        ok = false;
    }

#undef HANDLE_DST
#undef CPU_CAST
    if (!ok)
        ErrorBuilder("astype").not_implemented(std::string(dtype_name(src_dt)) + " -> " +
                                               std::string(dtype_name(dst_dt)));
    return Storage{CpuStorage{out_ptr, n * dsz, dst_dt}};
}

Storage CpuBackend::cast(const Storage& a, const Shape& shape, Dtype src_dt, Dtype dst_dt) {
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dst_dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    cast_impl(cs.ptr.get(), ptr.get(), n, src_dt, dst_dt);
    return Storage{CpuStorage{ptr, nb, dst_dt}};
}

CpuStorage CpuBackend::to_cpu(const Storage& a, const Shape&) {
    return std::get<CpuStorage>(a);
}

void CpuBackend::fill_ones(std::byte* ptr, std::size_t n, Dtype dt) {
    switch (dt) {
    case Dtype::C128: {
        auto* p = reinterpret_cast<std::complex<double>*>(ptr);
        std::fill_n(p, n, std::complex<double>{1.0, 0.0});
        break;
    }
    case Dtype::F32: {
        float* p = reinterpret_cast<float*>(ptr);
        for (std::size_t i = 0; i < n; ++i)
            p[i] = 1.f;
        break;
    }
    case Dtype::F64: {
        double* p = reinterpret_cast<double*>(ptr);
        for (std::size_t i = 0; i < n; ++i)
            p[i] = 1.0;
        break;
    }
    case Dtype::I32: {
        std::int32_t* p = reinterpret_cast<std::int32_t*>(ptr);
        for (std::size_t i = 0; i < n; ++i)
            p[i] = 1;
        break;
    }
    case Dtype::I64: {
        std::int64_t* p = reinterpret_cast<std::int64_t*>(ptr);
        for (std::size_t i = 0; i < n; ++i)
            p[i] = 1;
        break;
    }
    case Dtype::C64: {
        // Interleaved [re, im, re, im, ...] — set re=1, im=0 per element.
        float* p = reinterpret_cast<float*>(ptr);
        for (std::size_t i = 0; i < n; ++i) {
            p[2 * i] = 1.0f;
            p[2 * i + 1] = 0.0f;
        }
        break;
    }
    case Dtype::Bool: {
        auto* p = reinterpret_cast<std::uint8_t*>(ptr);
        for (std::size_t i = 0; i < n; ++i)
            p[i] = 1;
        break;
    }
    case Dtype::I8: {
        auto* p = reinterpret_cast<std::int8_t*>(ptr);
        for (std::size_t i = 0; i < n; ++i)
            p[i] = 1;
        break;
    }
    case Dtype::I16: {
        auto* p = reinterpret_cast<std::int16_t*>(ptr);
        for (std::size_t i = 0; i < n; ++i)
            p[i] = 1;
        break;
    }
    case Dtype::F16:
    case Dtype::BF16: {
        // 1.0 is 0x3C00 in IEEE binary16 and 0x3F80 in bfloat16.  Writing
        // the pattern avoids needing a half type for what is only a fill.
        const std::uint16_t one = dt == Dtype::BF16 ? 0x3F80 : 0x3C00;
        std::fill_n(reinterpret_cast<std::uint16_t*>(ptr), n, one);
        break;
    }
    default:
        ErrorBuilder("cpu_backend::ones").not_implemented("dtype not supported");
    }
}

void CpuBackend::cast_impl(
    const std::byte* src, std::byte* dst, std::size_t n, Dtype src_dt, Dtype dst_dt) {
    auto cast_loop = [&](auto from_tag, auto to_tag) {
        using Src = decltype(from_tag);
        using Dst = decltype(to_tag);
        const Src* sp = reinterpret_cast<const Src*>(src);
        Dst* dp = reinterpret_cast<Dst*>(dst);
        for (std::size_t i = 0; i < n; ++i)
            dp[i] = static_cast<Dst>(sp[i]);
    };

#define CAST_CASE(S, D, st, dt_)                                                                   \
    if (src_dt == Dtype::S && dst_dt == Dtype::D) {                                                \
        cast_loop(st{}, dt_{});                                                                    \
        return;                                                                                    \
    }

    CAST_CASE(F32, F64, float, double)
    CAST_CASE(F64, F32, double, float)
    CAST_CASE(F32, I32, float, std::int32_t)
    CAST_CASE(F32, I64, float, std::int64_t)
    CAST_CASE(F64, I32, double, std::int32_t)
    CAST_CASE(F64, I64, double, std::int64_t)
    CAST_CASE(I32, F32, std::int32_t, float)
    CAST_CASE(I32, F64, std::int32_t, double)
    CAST_CASE(I64, F32, std::int64_t, float)
    CAST_CASE(I64, F64, std::int64_t, double)
    CAST_CASE(I32, I64, std::int32_t, std::int64_t)
    CAST_CASE(I64, I32, std::int64_t, std::int32_t)
#undef CAST_CASE

    if (src_dt == dst_dt) {
        std::memcpy(dst, src, n * dtype_size(dst_dt));
        return;
    }
    if (detail::is_half_like(src_dt) && dst_dt == Dtype::F32) {
        detail::half_or_bfloat16_to_float_bulk(reinterpret_cast<const std::uint16_t*>(src),
                                               reinterpret_cast<float*>(dst), n,
                                               src_dt == Dtype::BF16);
        return;
    }
    if (src_dt == Dtype::F32 && detail::is_half_like(dst_dt)) {
        detail::float_to_half_or_bfloat16_bulk(reinterpret_cast<const float*>(src),
                                               reinterpret_cast<std::uint16_t*>(dst), n,
                                               dst_dt == Dtype::BF16);
        return;
    }

    // The twelve pairs above are a hand-written list, and it was
    // missing every pair involving int8, int16, bool or half — 40 of
    // the 56 combinations.  Nothing noticed while the ops that would
    // have used them refused those dtypes anyway; promoting integers
    // for the real-valued ops made ``sqrt(int8)`` reach here and fail
    // on the cast rather than the maths.
    //
    // Rather than extend the list, go through double: every dtype the
    // framework has converts to and from it, so one pair of switches
    // covers all of them.  int64 above 2^53 is the one lossy case, and
    // it keeps its exact path in the table above.
    auto load = [&](std::size_t i) -> double {
        switch (src_dt) {
        case Dtype::F64:
            return reinterpret_cast<const double*>(src)[i];
        case Dtype::F32:
            return reinterpret_cast<const float*>(src)[i];
        case Dtype::F16:
            return detail::half_bits_to_float(reinterpret_cast<const std::uint16_t*>(src)[i]);
        case Dtype::BF16:
            return detail::bfloat_bits_to_float(reinterpret_cast<const std::uint16_t*>(src)[i]);
        case Dtype::I64:
            return static_cast<double>(reinterpret_cast<const std::int64_t*>(src)[i]);
        case Dtype::I32:
            return reinterpret_cast<const std::int32_t*>(src)[i];
        case Dtype::I16:
            return reinterpret_cast<const std::int16_t*>(src)[i];
        case Dtype::I8:
            return reinterpret_cast<const std::int8_t*>(src)[i];
        default:  // Bool
            return reinterpret_cast<const std::uint8_t*>(src)[i] ? 1.0 : 0.0;
        }
    };
    auto store = [&](std::size_t i, double v) {
        switch (dst_dt) {
        case Dtype::F64:
            reinterpret_cast<double*>(dst)[i] = v;
            break;
        case Dtype::F32:
            reinterpret_cast<float*>(dst)[i] = static_cast<float>(v);
            break;
        case Dtype::F16:
            reinterpret_cast<std::uint16_t*>(dst)[i] =
                detail::float_to_half_bits(static_cast<float>(v));
            break;
        case Dtype::BF16:
            reinterpret_cast<std::uint16_t*>(dst)[i] =
                detail::float_to_bfloat_bits(static_cast<float>(v));
            break;
        case Dtype::I64:
            reinterpret_cast<std::int64_t*>(dst)[i] = static_cast<std::int64_t>(v);
            break;
        case Dtype::I32:
            reinterpret_cast<std::int32_t*>(dst)[i] = static_cast<std::int32_t>(v);
            break;
        case Dtype::I16:
            reinterpret_cast<std::int16_t*>(dst)[i] = static_cast<std::int16_t>(v);
            break;
        case Dtype::I8:
            reinterpret_cast<std::int8_t*>(dst)[i] = static_cast<std::int8_t>(v);
            break;
        default:  // Bool — anything non-zero is true
            reinterpret_cast<std::uint8_t*>(dst)[i] = (v != 0.0) ? 1 : 0;
            break;
        }
    };
    for (std::size_t i = 0; i < n; ++i)
        store(i, load(i));
}

Storage CpuBackend::full(const Shape& shape, Dtype dt, double fill_value) {
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    auto* p = ptr.get();
    switch (dt) {
    case Dtype::C128: {
        auto* out = reinterpret_cast<std::complex<double>*>(p);
        std::fill_n(out, n, std::complex<double>{fill_value, 0.0});
        break;
    }
    case Dtype::Bool: {
        auto v = fill_value != 0.0 ? std::uint8_t{1} : std::uint8_t{0};
        for (std::size_t i = 0; i < n; ++i)
            reinterpret_cast<std::uint8_t*>(p)[i] = v;
        break;
    }
    case Dtype::I8: {
        auto v = static_cast<std::int8_t>(fill_value);
        for (std::size_t i = 0; i < n; ++i)
            reinterpret_cast<std::int8_t*>(p)[i] = v;
        break;
    }
    case Dtype::I16: {
        auto v = static_cast<std::int16_t>(fill_value);
        for (std::size_t i = 0; i < n; ++i)
            reinterpret_cast<std::int16_t*>(p)[i] = v;
        break;
    }
    case Dtype::I32: {
        auto v = static_cast<std::int32_t>(fill_value);
        for (std::size_t i = 0; i < n; ++i)
            reinterpret_cast<std::int32_t*>(p)[i] = v;
        break;
    }
    case Dtype::I64: {
        auto v = static_cast<std::int64_t>(fill_value);
        for (std::size_t i = 0; i < n; ++i)
            reinterpret_cast<std::int64_t*>(p)[i] = v;
        break;
    }
    case Dtype::F32: {
        auto v = static_cast<float>(fill_value);
        for (std::size_t i = 0; i < n; ++i)
            reinterpret_cast<float*>(p)[i] = v;
        break;
    }
    case Dtype::F64: {
        auto v = fill_value;
        for (std::size_t i = 0; i < n; ++i)
            reinterpret_cast<double*>(p)[i] = v;
        break;
    }
    case Dtype::C64: {
        // Interleaved [re, im, re, im, ...]; ``fill_value`` becomes the
        // real part, imag is set to 0.  Users wanting non-zero imag
        // should construct the array via ``complex_combine(real, imag)``.
        auto v = static_cast<float>(fill_value);
        float* fp = reinterpret_cast<float*>(p);
        for (std::size_t i = 0; i < n; ++i) {
            fp[2 * i] = v;
            fp[2 * i + 1] = 0.0f;
        }
        break;
    }
    case Dtype::F16: {
        // 102 ops reached this gate: a composite that builds a constant
        // of its input's dtype could not do so in half, so the whole
        // composite refused float16 on the CPU while Metal ran it.
        const auto bits = detail::float_to_half_bits(static_cast<float>(fill_value));
        auto* hp = reinterpret_cast<std::uint16_t*>(p);
        for (std::size_t i = 0; i < n; ++i)
            hp[i] = bits;
        break;
    }
    case Dtype::BF16: {
        const auto bits = detail::float_to_bfloat_bits(static_cast<float>(fill_value));
        auto* hp = reinterpret_cast<std::uint16_t*>(p);
        for (std::size_t i = 0; i < n; ++i)
            hp[i] = bits;
        break;
    }
    default:
        ErrorBuilder("cpu_backend::full").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::eye(std::int64_t N, std::int64_t M, std::int64_t k, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(eye(N, M, k, Dtype::F32), dt);
    std::size_t nb = static_cast<std::size_t>(N * M) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    std::memset(ptr.get(), 0, nb);
    auto set_one = [&](auto* p) {
        using T = std::remove_pointer_t<decltype(p)>;
        for (std::int64_t i = 0; i < N; ++i) {
            if (const std::int64_t j = i + k; j >= 0 && j < M)
                p[i * M + j] = static_cast<T>(1);
        }
    };
    switch (dt) {
    case Dtype::Bool:
        set_one(reinterpret_cast<std::uint8_t*>(ptr.get()));
        break;
    case Dtype::I8:
        set_one(reinterpret_cast<std::int8_t*>(ptr.get()));
        break;
    case Dtype::I16:
        set_one(reinterpret_cast<std::int16_t*>(ptr.get()));
        break;
    case Dtype::I32:
        set_one(reinterpret_cast<std::int32_t*>(ptr.get()));
        break;
    case Dtype::I64:
        set_one(reinterpret_cast<std::int64_t*>(ptr.get()));
        break;
    case Dtype::F32:
        set_one(reinterpret_cast<float*>(ptr.get()));
        break;
    case Dtype::F64:
        set_one(reinterpret_cast<double*>(ptr.get()));
        break;
    case Dtype::C64:
        set_one(reinterpret_cast<std::complex<float>*>(ptr.get()));
        break;
    case Dtype::C128:
        set_one(reinterpret_cast<std::complex<double>*>(ptr.get()));
        break;
    default:
        ErrorBuilder("cpu_backend::eye").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::diag(
    const Storage& v, const Shape& v_shape, std::int64_t k, Dtype dt, Shape& out_shape) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(diag(detail::as_f32(v), v_shape, k, Dtype::F32, out_shape), dt);
    const auto& cv = std::get<CpuStorage>(v);
    const std::size_t elem = dtype_size(dt);
    if (v_shape.size() == 1) {
        const std::int64_t L = v_shape[0];
        const std::int64_t side = L + std::abs(k);
        out_shape = {side, side};
        std::size_t nb = static_cast<std::size_t>(side * side) * elem;
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        std::memset(ptr.get(), 0, nb);
        auto fill = [&](auto* dst, const auto* src) {
            using T = std::remove_pointer_t<decltype(dst)>;
            for (std::int64_t i = 0; i < L; ++i) {
                const std::int64_t row = (k >= 0) ? i : (i - k);
                const std::int64_t col = (k >= 0) ? (i + k) : i;
                dst[row * side + col] = static_cast<T>(src[i]);
            }
        };
        switch (dt) {
        case Dtype::F32:
            fill(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(cv.ptr.get()));
            break;
        case Dtype::F64:
            fill(reinterpret_cast<double*>(ptr.get()),
                 reinterpret_cast<const double*>(cv.ptr.get()));
            break;
        case Dtype::I32:
            fill(reinterpret_cast<std::int32_t*>(ptr.get()),
                 reinterpret_cast<const std::int32_t*>(cv.ptr.get()));
            break;
        case Dtype::I64:
            fill(reinterpret_cast<std::int64_t*>(ptr.get()),
                 reinterpret_cast<const std::int64_t*>(cv.ptr.get()));
            break;
        default:
            ErrorBuilder("cpu_backend::diag").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    const std::int64_t Mv = v_shape[0], Nv = v_shape[1];
    const std::int64_t r0 = (k >= 0) ? 0 : -k;
    const std::int64_t c0 = (k >= 0) ? k : 0;
    const std::int64_t Lv = std::min(Mv - r0, Nv - c0);
    const std::int64_t out_len = std::max<std::int64_t>(Lv, 0);
    out_shape = {out_len};
    std::size_t nb = static_cast<std::size_t>(out_len) * elem;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    auto extract = [&](auto* dst, const auto* src) {
        using T = std::remove_pointer_t<decltype(dst)>;
        for (std::int64_t i = 0; i < out_len; ++i)
            dst[i] = static_cast<T>(src[(r0 + i) * Nv + (c0 + i)]);
    };
    switch (dt) {
    case Dtype::F32:
        extract(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(cv.ptr.get()));
        break;
    case Dtype::F64:
        extract(reinterpret_cast<double*>(ptr.get()),
                reinterpret_cast<const double*>(cv.ptr.get()));
        break;
    case Dtype::I32:
        extract(reinterpret_cast<std::int32_t*>(ptr.get()),
                reinterpret_cast<const std::int32_t*>(cv.ptr.get()));
        break;
    case Dtype::I64:
        extract(reinterpret_cast<std::int64_t*>(ptr.get()),
                reinterpret_cast<const std::int64_t*>(cv.ptr.get()));
        break;
    default:
        ErrorBuilder("cpu_backend::diag").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::tri(const Storage& input, const Shape& shape, Dtype dt, int k, bool upper) {
    const auto& ca = std::get<CpuStorage>(input);
    if (shape.size() < 2)
        ErrorBuilder("cpu_backend::tri").invalid_argument("input must have ndim >= 2");
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const std::int64_t M = shape[shape.size() - 2];
    const std::int64_t N = shape[shape.size() - 1];
    std::size_t batch = 1;
    for (std::size_t d = 0; d < shape.size() - 2; ++d)
        batch *= static_cast<std::size_t>(shape[d]);
    const std::size_t plane = static_cast<std::size_t>(M * N);
    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        const T* src = reinterpret_cast<const T*>(ca.ptr.get());
        T* dst = reinterpret_cast<T*>(ptr.get());
        for (std::size_t b = 0; b < batch; ++b)
            for (std::int64_t i = 0; i < M; ++i)
                for (std::int64_t j = 0; j < N; ++j) {
                    const std::size_t f = b * plane + static_cast<std::size_t>(i * N + j);
                    dst[f] = (upper ? (j - i >= k) : (j - i <= k)) ? src[f] : T{};
                }
    };
    // Every dtype the framework has, because ``tri`` is masking rather
    // than arithmetic: it copies an element or writes a zero, so the
    // only thing that matters is how wide the element is.  This was
    // F32/F64 only, with half routed through a widen-and-round —
    // pointless for a memcpy, and lossy in principle — while integers
    // were refused outright even though ``tril`` on an integer matrix
    // is an ordinary thing to want and Metal did it.
    switch (dt) {
    case Dtype::F32:
        run(float{});
        break;
    case Dtype::F64:
        run(double{});
        break;
    case Dtype::F16:
    case Dtype::BF16:
        run(std::uint16_t{});  // bit pattern; zero is all-zero bits in both
        break;
    case Dtype::I64:
        run(std::int64_t{});
        break;
    case Dtype::C64:
        run(std::complex<float>{});
        break;
    case Dtype::C128:
        run(std::complex<double>{});
        break;
    case Dtype::I32:
        run(std::int32_t{});
        break;
    case Dtype::I16:
        run(std::int16_t{});
        break;
    case Dtype::I8:
    case Dtype::Bool:
        run(std::uint8_t{});
        break;
    default:
        ErrorBuilder("cpu_backend::tri").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

}  // namespace backend
}  // namespace lucid
