// lucid/_C/backend/cpu/methods/Binary.cpp
// Out-of-line CpuBackend method definitions for the Binary family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

template <typename Scalar>
Storage CpuBackend::complex_binary(const Storage& a,
                                   const Storage& b,
                                   const Shape& shape,
                                   void (*fn)(const Scalar*, const Scalar*, Scalar*, std::size_t)) {
    const auto& ca = std::get<CpuStorage>(a);
    const auto& cb = std::get<CpuStorage>(b);
    const std::size_t n = shape_numel(shape);
    constexpr auto dt = sizeof(Scalar) == sizeof(double) ? Dtype::C128 : Dtype::C64;
    const std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    fn(reinterpret_cast<const Scalar*>(ca.ptr.get()), reinterpret_cast<const Scalar*>(cb.ptr.get()),
       reinterpret_cast<Scalar*>(ptr.get()), n);
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::add(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    if (dt == Dtype::C128)
        return complex_binary(a, b, shape, cpu::vzadd_c128);
    if (dt == Dtype::C64)
        return complex_binary(a, b, shape, cpu::vzadd_c64);
    return binary_op(
        a, b, shape, dt,
        [](const float* ap, const float* bp, float* op, std::size_t n) {
            cpu::vadd_f32(ap, bp, op, n);
        },
        [](const double* ap, const double* bp, double* op, std::size_t n) {
            cpu::vadd_f64(ap, bp, op, n);
        },
        [](const std::int32_t* ap, const std::int32_t* bp, std::int32_t* op, std::size_t n) {
            cpu::vadd_i32(ap, bp, op, n);
        },
        [](const std::int64_t* ap, const std::int64_t* bp, std::int64_t* op, std::size_t n) {
            cpu::vadd_i64(ap, bp, op, n);
        });
}

Storage CpuBackend::sub(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    if (dt == Dtype::C128)
        return complex_binary(a, b, shape, cpu::vzsub_c128);
    if (dt == Dtype::C64)
        return complex_binary(a, b, shape, cpu::vzsub_c64);
    return binary_op(
        a, b, shape, dt,
        [](const float* ap, const float* bp, float* op, std::size_t n) {
            cpu::vsub_f32(ap, bp, op, n);
        },
        [](const double* ap, const double* bp, double* op, std::size_t n) {
            cpu::vsub_f64(ap, bp, op, n);
        },
        [](const std::int32_t* ap, const std::int32_t* bp, std::int32_t* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = ap[i] - bp[i];
        },
        [](const std::int64_t* ap, const std::int64_t* bp, std::int64_t* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = ap[i] - bp[i];
        });
}

Storage CpuBackend::mul(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    if (dt == Dtype::C128)
        return complex_binary(a, b, shape, cpu::vzmul_c128);
    if (dt == Dtype::C64)
        return complex_binary(a, b, shape, cpu::vzmul_c64);
    return binary_op(
        a, b, shape, dt,
        [](const float* ap, const float* bp, float* op, std::size_t n) {
            cpu::vmul_f32(ap, bp, op, n);
        },
        [](const double* ap, const double* bp, double* op, std::size_t n) {
            cpu::vmul_f64(ap, bp, op, n);
        },
        [](const std::int32_t* ap, const std::int32_t* bp, std::int32_t* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = ap[i] * bp[i];
        },
        [](const std::int64_t* ap, const std::int64_t* bp, std::int64_t* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = ap[i] * bp[i];
        });
}

Storage CpuBackend::div(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    if (dt == Dtype::C128)
        return complex_binary(a, b, shape, cpu::vzdiv_c128);
    if (dt == Dtype::C64)
        return complex_binary(a, b, shape, cpu::vzdiv_c64);
    return binary_op(
        a, b, shape, dt,
        [](const float* ap, const float* bp, float* op, std::size_t n) {
            cpu::vdiv_f32(ap, bp, op, n);
        },
        [](const double* ap, const double* bp, double* op, std::size_t n) {
            cpu::vdiv_f64(ap, bp, op, n);
        },
        [](const std::int32_t* ap, const std::int32_t* bp, std::int32_t* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = ap[i] / bp[i];
        },
        [](const std::int64_t* ap, const std::int64_t* bp, std::int64_t* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = ap[i] / bp[i];
        });
}

Storage CpuBackend::pow(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    return binary_op(
        a, b, shape, dt,
        // vForce's pow is ~8 ns an element: split it across cores, an L1
        // tile at a time (see detail::pow_tile for the exact exponents).
        [](const float* ap, const float* bp, float* op, std::size_t n) {
            cpu::parallel_for(n, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
                for (std::size_t s = lo; s < hi; s += kTanhTile)
                    detail::pow_tile(ap + s, bp + s, op + s, std::min(kTanhTile, hi - s));
            });
        },
        [](const double* ap, const double* bp, double* op, std::size_t n) {
            cpu::parallel_for(n, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
                for (std::size_t s = lo; s < hi; s += kTanhTile)
                    detail::pow_tile(ap + s, bp + s, op + s, std::min(kTanhTile, hi - s));
            });
        },
        cpu::ipow_vec<std::int32_t>, cpu::ipow_vec<std::int64_t>);
}

Storage CpuBackend::bitwise_binary(
    const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) {
    const auto& ca = std::get<CpuStorage>(a);
    const auto& cb = std::get<CpuStorage>(b);
    const std::size_t n = shape_numel(shape);
    const std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);

    if (op < 0 || op > 4)
        ErrorBuilder("cpu_backend::bitwise_binary").fail("unknown op");
    // The op is chosen once, outside the loop: inside, the if-chain and
    // its error path kept ``a & b`` scalar — twelve times an add.
    auto run = [&](auto* dst_in, const auto* lhs_in, const auto* rhs_in) {
        using T = std::remove_pointer_t<decltype(dst_in)>;
        // Width of T in bits — used to clamp shift amounts so we don't
        // hit the C++ undefined-behaviour zone (shifts ≥ width are UB).
        constexpr std::int64_t kWidth = static_cast<std::int64_t>(sizeof(T) * 8);
        T* __restrict dst = dst_in;
        const T* __restrict lhs = lhs_in;
        const T* __restrict rhs = rhs_in;
        auto each = [&](auto f) {
            for (std::size_t i = 0; i < n; ++i)
                dst[i] = static_cast<T>(
                    f(static_cast<std::int64_t>(lhs[i]), static_cast<std::int64_t>(rhs[i])));
        };
        switch (op) {
        case 0:
            each([](std::int64_t x, std::int64_t y) { return x & y; });
            break;
        case 1:
            each([](std::int64_t x, std::int64_t y) { return x | y; });
            break;
        case 2:
            each([](std::int64_t x, std::int64_t y) { return x ^ y; });
            break;
        case 3:
            // Left-shift: clamp y into [0, width) — out-of-range shifts
            // are UB in C++, but the reference framework returns 0 for
            // y ≥ width and 0 for y < 0.
            each([](std::int64_t x, std::int64_t y) -> std::int64_t {
                if (y < 0 || y >= kWidth)
                    return 0;
                return static_cast<std::int64_t>(static_cast<T>(x) << y);
            });
            break;
        default:
            // Arithmetic right-shift on the *narrow* type so sign
            // extension matches the input dtype.  An amount outside
            // [0, width) shifts every bit out, leaving the sign fill —
            // negative amounts included, as in the reference.
            each([](std::int64_t x, std::int64_t y) -> std::int64_t {
                if (y < 0 || y >= kWidth)
                    return (x < 0) ? -1 : 0;
                return static_cast<std::int64_t>(static_cast<T>(x) >> y);
            });
            break;
        }
    };

    if (dt == Dtype::I32) {
        run(reinterpret_cast<std::int32_t*>(ptr.get()),
            reinterpret_cast<const std::int32_t*>(ca.ptr.get()),
            reinterpret_cast<const std::int32_t*>(cb.ptr.get()));
    } else if (dt == Dtype::I64) {
        run(reinterpret_cast<std::int64_t*>(ptr.get()),
            reinterpret_cast<const std::int64_t*>(ca.ptr.get()),
            reinterpret_cast<const std::int64_t*>(cb.ptr.get()));
    } else if (dt == Dtype::I16) {
        run(reinterpret_cast<std::int16_t*>(ptr.get()),
            reinterpret_cast<const std::int16_t*>(ca.ptr.get()),
            reinterpret_cast<const std::int16_t*>(cb.ptr.get()));
    } else if (dt == Dtype::I8) {
        run(reinterpret_cast<std::int8_t*>(ptr.get()),
            reinterpret_cast<const std::int8_t*>(ca.ptr.get()),
            reinterpret_cast<const std::int8_t*>(cb.ptr.get()));
    } else if (dt == Dtype::Bool) {
        run(reinterpret_cast<std::uint8_t*>(ptr.get()),
            reinterpret_cast<const std::uint8_t*>(ca.ptr.get()),
            reinterpret_cast<const std::uint8_t*>(cb.ptr.get()));
    } else {
        ErrorBuilder("cpu_backend::bitwise_binary")
            .not_implemented("dtype must be integer or bool");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::compare_binary(
    const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) {
    // Comparison answers in Bool but has to read the operands, and
    // Accelerate has no half compare.  Widen both and compare in
    // single — the result is exact, since half is a subset of float.
    if (detail::is_half_like(dt)) {
        const std::size_t half_n = shape_numel(shape);
        return compare_binary(Storage{detail::widen_half(std::get<CpuStorage>(a), half_n)},
                              Storage{detail::widen_half(std::get<CpuStorage>(b), half_n)}, shape,
                              Dtype::F32, op);
    }
    const auto& ca = std::get<CpuStorage>(a);
    const auto& cb = std::get<CpuStorage>(b);
    const std::size_t n = shape_numel(shape);
    auto ptr = allocate_aligned_bytes(n, Device::CPU);
    auto* dst = reinterpret_cast<std::uint8_t*>(ptr.get());

    // The comparison is chosen once, outside the loop, and the loop is
    // split across cores.  An if-chain over ``op`` with an error path in
    // every element kept it scalar and on one core: 0.52 ms for ``x > 0``
    // on 524k floats, twenty times ``x + y``.
    if (op < 0 || op > 5)
        ErrorBuilder("cpu_backend::compare_binary").fail("unknown op");
    auto run = [&](auto tag) {
        using T = decltype(tag);
        const auto* lhs_in = reinterpret_cast<const T*>(ca.ptr.get());
        const auto* rhs_in = reinterpret_cast<const T*>(cb.ptr.get());
        auto each = [&](auto cmp) {
            cpu::parallel_for(n, kStreamGrain, [&](std::size_t lo, std::size_t hi) {
                // Local restrict pointers: a byte store may alias
                // anything, so pointers read through the closure would
                // be reloaded every element and the loop stay scalar.
                const T* __restrict lhs = lhs_in;
                const T* __restrict rhs = rhs_in;
                std::uint8_t* __restrict out = dst;
                for (std::size_t i = lo; i < hi; ++i)
                    out[i] = static_cast<std::uint8_t>(cmp(lhs[i], rhs[i]));
            });
        };
        if (op == 0) {
            each([](T l, T r) { return l == r; });
        } else if (op == 1) {
            each([](T l, T r) { return l != r; });
        } else if constexpr (requires(T l, T r) { l < r; }) {
            if (op == 2)
                each([](T l, T r) { return l > r; });
            else if (op == 3)
                each([](T l, T r) { return l >= r; });
            else if (op == 4)
                each([](T l, T r) { return l < r; });
            else
                each([](T l, T r) { return l <= r; });
        } else {
            ErrorBuilder("cpu_backend::compare_binary")
                .not_implemented("ordering comparison is not supported for complex dtypes");
        }
    };

    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else if (dt == Dtype::I32)
        run(std::int32_t{});
    else if (dt == Dtype::I64)
        run(std::int64_t{});
    else if (dt == Dtype::I16)
        run(std::int16_t{});
    else if (dt == Dtype::I8)
        run(std::int8_t{});
    else if (dt == Dtype::Bool)
        run(std::uint8_t{});
    else if (dt == Dtype::C64)
        run(std::complex<float>{});
    else if (dt == Dtype::C128)
        run(std::complex<double>{});
    else
        ErrorBuilder("cpu_backend::compare_binary").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, n, Dtype::Bool}};
}

Storage CpuBackend::maximum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    return binary_op(
        a, b, shape, dt,
        [](const float* ap, const float* bp, float* op, std::size_t n) {
            cpu::vmax_f32(ap, bp, op, n);
        },
        [](const double* ap, const double* bp, double* op, std::size_t n) {
            cpu::vmax_f64(ap, bp, op, n);
        },
        [](const std::int32_t* ap, const std::int32_t* bp, std::int32_t* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = std::max(ap[i], bp[i]);
        },
        [](const std::int64_t* ap, const std::int64_t* bp, std::int64_t* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = std::max(ap[i], bp[i]);
        });
}

Storage CpuBackend::minimum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    return binary_op(
        a, b, shape, dt,
        [](const float* ap, const float* bp, float* op, std::size_t n) {
            cpu::vmin_f32(ap, bp, op, n);
        },
        [](const double* ap, const double* bp, double* op, std::size_t n) {
            cpu::vmin_f64(ap, bp, op, n);
        },
        [](const std::int32_t* ap, const std::int32_t* bp, std::int32_t* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = std::min(ap[i], bp[i]);
        },
        [](const std::int64_t* ap, const std::int64_t* bp, std::int64_t* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = std::min(ap[i], bp[i]);
        });
}

Storage CpuBackend::where_branch(
    const Storage& grad, const Storage& cond, const Shape& shape, Dtype dt, bool true_branch) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(where_branch(detail::as_f32(grad), detail::as_f32(cond), shape,
                                                Dtype::F32, true_branch),
                                   dt);
    const auto& g = std::get<CpuStorage>(grad);
    const auto& c = std::get<CpuStorage>(cond);
    auto out = zeros(shape, dt);
    const std::size_t n = shape_numel(shape);
    const auto* cp = reinterpret_cast<const std::uint8_t*>(c.ptr.get());
    // Loaded, then selected — a branch around the load mispredicted on
    // every mask that is not sorted (see relu_backward).
    if (dt == Dtype::F32) {
        const float* __restrict gp = reinterpret_cast<const float*>(g.ptr.get());
        float* __restrict dst = reinterpret_cast<float*>(std::get<CpuStorage>(out).ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const float v = gp[i];
            dst[i] = ((cp[i] != 0) == true_branch) ? v : 0.0f;
        }
    } else if (dt == Dtype::F64) {
        const double* __restrict gp = reinterpret_cast<const double*>(g.ptr.get());
        double* __restrict dst = reinterpret_cast<double*>(std::get<CpuStorage>(out).ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const double v = gp[i];
            dst[i] = ((cp[i] != 0) == true_branch) ? v : 0.0;
        }
    } else {
        ErrorBuilder("cpu_backend::where_branch").not_implemented("dtype not supported");
    }
    return out;
}

Storage CpuBackend::masked_fill(
    const Storage& a, const Storage& mask, const Shape& shape, Dtype dt, double value) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            masked_fill(detail::as_f32(a), detail::as_f32(mask), shape, Dtype::F32, value), dt);
    const auto& as = std::get<CpuStorage>(a);
    const auto& ms = std::get<CpuStorage>(mask);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const auto* mp = reinterpret_cast<const std::uint8_t*>(ms.ptr.get());
    if (dt == Dtype::F32) {
        detail::select_fill(reinterpret_cast<const float*>(as.ptr.get()), mp,
                            static_cast<float>(value), reinterpret_cast<float*>(ptr.get()), n);
    } else if (dt == Dtype::F64) {
        detail::select_fill(reinterpret_cast<const double*>(as.ptr.get()), mp, value,
                            reinterpret_cast<double*>(ptr.get()), n);
    } else if (detail::is_int_like(dt)) {
        // Structural, not arithmetic: it selects between an element and
        // a constant.  Integers keep their dtype in the reference
        // framework and on Metal.
        const std::int64_t fill = static_cast<std::int64_t>(value);
        detail::for_each_int_masked(dt, as.ptr.get(), mp, ptr.get(), n, fill);
    } else {
        ErrorBuilder("cpu_backend::masked_fill").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::pow_scalar(const Storage& a, const Shape& shape, Dtype dt, double exp) {
    // Accelerate's scalar/vector primitives are f32 and f64 only.  Widen at
    // the door, reuse, round once.  Reached constantly from backward, where
    // chain-rule constants arrive as scalars.
    if (detail::is_half_like(dt)) {
        const std::size_t half_n = shape_numel(shape);
        return detail::narrow_half(
            pow_scalar(Storage{detail::widen_half(std::get<CpuStorage>(a), half_n)}, shape,
                       Dtype::F32, exp),
            half_n, dt);
    }
    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t numel = shape_numel(shape);
    std::size_t nb = numel * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    switch (dt) {
    case Dtype::F32:
        detail::pow_by_scalar(reinterpret_cast<const float*>(cs.ptr.get()), static_cast<float>(exp),
                              reinterpret_cast<float*>(ptr.get()), numel);
        break;
    case Dtype::F64:
        detail::pow_by_scalar(reinterpret_cast<const double*>(cs.ptr.get()), exp,
                              reinterpret_cast<double*>(ptr.get()), numel);
        break;
    default:
        ErrorBuilder("cpu_backend::pow_scalar").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::rpow_scalar(const Storage& a, const Shape& shape, Dtype dt, double base) {
    // Accelerate's scalar/vector primitives are f32 and f64 only.  Widen at
    // the door, reuse, round once.  Reached constantly from backward, where
    // chain-rule constants arrive as scalars.
    if (detail::is_half_like(dt)) {
        const std::size_t half_n = shape_numel(shape);
        return detail::narrow_half(
            rpow_scalar(Storage{detail::widen_half(std::get<CpuStorage>(a), half_n)}, shape,
                        Dtype::F32, base),
            half_n, dt);
    }
    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t numel = shape_numel(shape);
    std::size_t nb = numel * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    // The tensor power's tiles, so an exponent element of 2 is ``base * base``
    // here as in ``pow``.
    auto run = [&](auto b) {
        using T = decltype(b);
        const T* y = reinterpret_cast<const T*>(cs.ptr.get());
        T* out = reinterpret_cast<T*>(ptr.get());
        cpu::parallel_for(numel, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
            T bases[kTanhTile];
            std::fill_n(bases, std::min(kTanhTile, hi - lo), b);
            for (std::size_t s = lo; s < hi; s += kTanhTile)
                detail::pow_tile(bases, y + s, out + s, std::min(kTanhTile, hi - s));
        });
    };
    switch (dt) {
    case Dtype::F32:
        run(static_cast<float>(base));
        break;
    case Dtype::F64:
        run(base);
        break;
    default:
        ErrorBuilder("cpu_backend::rpow_scalar").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage
CpuBackend::clip(const Storage& a, const Shape& shape, Dtype dt, double min_v, double max_v) {
    // Clamping an integer tensor keeps its dtype in the reference
    // framework, and Metal already did it.  Exact through int64.
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(clip(detail::as_f32(a), shape, Dtype::F32, min_v, max_v), dt);
    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t numel = shape_numel(shape);
    std::size_t nb = numel * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    switch (dt) {
    case Dtype::F32: {
        const auto* src = reinterpret_cast<const float*>(cs.ptr.get());
        auto* dst = reinterpret_cast<float*>(ptr.get());
        const float lo = static_cast<float>(min_v);
        const float hi = static_cast<float>(max_v);
        for (std::size_t i = 0; i < numel; ++i) {
            float v = src[i];
            if (v < lo)
                v = lo;
            else if (v > hi)
                v = hi;
            dst[i] = v;
        }
        break;
    }
    case Dtype::F64: {
        const auto* src = reinterpret_cast<const double*>(cs.ptr.get());
        auto* dst = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < numel; ++i) {
            double v = src[i];
            if (v < min_v)
                v = min_v;
            else if (v > max_v)
                v = max_v;
            dst[i] = v;
        }
        break;
    }
    default:
        if (detail::is_int_like(dt)) {
            // Clamping keeps the dtype for integers.  The bounds arrive
            // as doubles; they are compared in int64 so a bound outside
            // the caller's width still clamps rather than wrapping.
            const std::int64_t lo = static_cast<std::int64_t>(min_v);
            const std::int64_t hi = static_cast<std::int64_t>(max_v);
            detail::for_each_int(dt, cs.ptr.get(), ptr.get(), numel, [lo, hi](auto v) {
                using T = decltype(v);
                const std::int64_t w = static_cast<std::int64_t>(v);
                return static_cast<T>(w < lo ? lo : (w > hi ? hi : w));
            });
            break;
        }
        ErrorBuilder("cpu_backend::clip").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

std::pair<Storage, Storage> CpuBackend::expand_and_multiply(const Storage& mask,
                                                            const Storage& x,
                                                            const Shape& mask_shape,
                                                            const Shape& x_shape,
                                                            Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(expand_and_multiply(detail::as_f32(mask), detail::as_f32(x),
                                                       mask_shape, x_shape, Dtype::F32),
                                   dt);
    const std::size_t B_sz = static_cast<std::size_t>(x_shape[0]);

    const bool is_channel_mask = (mask_shape.size() >= 2 && mask_shape[1] > 1);
    const std::size_t C_sz = is_channel_mask ? static_cast<std::size_t>(mask_shape[1])
                                             : static_cast<std::size_t>(x_shape[1]);
    std::size_t spatial = 1;
    for (std::size_t i = 2; i < x_shape.size(); ++i)
        spatial *= static_cast<std::size_t>(x_shape[i]);

    const std::size_t numel = shape_numel(x_shape);
    CpuStorage exp_out = alloc_cpu(numel, dt);
    const auto& m_cpu = std::get<CpuStorage>(mask);
    const auto& x_cpu = std::get<CpuStorage>(x);
    CpuStorage y_out = alloc_cpu(numel, dt);

    if (is_channel_mask) {
        if (dt == Dtype::F32) {
            const float* mp = reinterpret_cast<const float*>(m_cpu.ptr.get());
            const float* xp = reinterpret_cast<const float*>(x_cpu.ptr.get());
            float* ep = reinterpret_cast<float*>(exp_out.ptr.get());
            float* yp = reinterpret_cast<float*>(y_out.ptr.get());
            for (std::size_t b = 0; b < B_sz; ++b)
                for (std::size_t c = 0; c < C_sz; ++c) {
                    const float v = mp[b * C_sz + c];
                    for (std::size_t s = 0; s < spatial; ++s) {
                        const std::size_t idx = (b * C_sz + c) * spatial + s;
                        ep[idx] = v;
                        yp[idx] = xp[idx] * v;
                    }
                }
        } else if (dt == Dtype::F64) {
            const double* mp = reinterpret_cast<const double*>(m_cpu.ptr.get());
            const double* xp = reinterpret_cast<const double*>(x_cpu.ptr.get());
            double* ep = reinterpret_cast<double*>(exp_out.ptr.get());
            double* yp = reinterpret_cast<double*>(y_out.ptr.get());
            for (std::size_t b = 0; b < B_sz; ++b)
                for (std::size_t c = 0; c < C_sz; ++c) {
                    const double v = mp[b * C_sz + c];
                    for (std::size_t s = 0; s < spatial; ++s) {
                        const std::size_t idx = (b * C_sz + c) * spatial + s;
                        ep[idx] = v;
                        yp[idx] = xp[idx] * v;
                    }
                }
        } else {
            ErrorBuilder("cpu_backend::expand_and_multiply").not_implemented("dtype not supported");
        }
    } else {
        const std::size_t per = numel / B_sz;
        if (dt == Dtype::F32) {
            const float* mp = reinterpret_cast<const float*>(m_cpu.ptr.get());
            const float* xp = reinterpret_cast<const float*>(x_cpu.ptr.get());
            float* ep = reinterpret_cast<float*>(exp_out.ptr.get());
            float* yp = reinterpret_cast<float*>(y_out.ptr.get());
            for (std::size_t b = 0; b < B_sz; ++b) {
                const float v = mp[b];
                for (std::size_t s = 0; s < per; ++s) {
                    const std::size_t idx = b * per + s;
                    ep[idx] = v;
                    yp[idx] = xp[idx] * v;
                }
            }
        } else if (dt == Dtype::F64) {
            const double* mp = reinterpret_cast<const double*>(m_cpu.ptr.get());
            const double* xp = reinterpret_cast<const double*>(x_cpu.ptr.get());
            double* ep = reinterpret_cast<double*>(exp_out.ptr.get());
            double* yp = reinterpret_cast<double*>(y_out.ptr.get());
            for (std::size_t b = 0; b < B_sz; ++b) {
                const double v = mp[b];
                for (std::size_t s = 0; s < per; ++s) {
                    const std::size_t idx = b * per + s;
                    ep[idx] = v;
                    yp[idx] = xp[idx] * v;
                }
            }
        } else {
            ErrorBuilder("cpu_backend::expand_and_multiply").not_implemented("dtype not supported");
        }
    }
    return {Storage{std::move(exp_out)}, Storage{std::move(y_out)}};
}

template <class F32Fn, class F64Fn, class I32Fn, class I64Fn>
Storage CpuBackend::binary_op(const Storage& a,
                              const Storage& b,
                              const Shape& shape,
                              Dtype dt,
                              F32Fn fn32,
                              F64Fn fn64,
                              I32Fn fni32,
                              I64Fn fni64) {
    const auto& ca = std::get<CpuStorage>(a);
    const auto& cb = std::get<CpuStorage>(b);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32)
        fn32(reinterpret_cast<const float*>(ca.ptr.get()),
             reinterpret_cast<const float*>(cb.ptr.get()), reinterpret_cast<float*>(ptr.get()), n);
    else if (dt == Dtype::F64)
        fn64(reinterpret_cast<const double*>(ca.ptr.get()),
             reinterpret_cast<const double*>(cb.ptr.get()), reinterpret_cast<double*>(ptr.get()),
             n);
    else if (dt == Dtype::I32)
        fni32(reinterpret_cast<const std::int32_t*>(ca.ptr.get()),
              reinterpret_cast<const std::int32_t*>(cb.ptr.get()),
              reinterpret_cast<std::int32_t*>(ptr.get()), n);
    else if (dt == Dtype::I64)
        fni64(reinterpret_cast<const std::int64_t*>(ca.ptr.get()),
              reinterpret_cast<const std::int64_t*>(cb.ptr.get()),
              reinterpret_cast<std::int64_t*>(ptr.get()), n);
    else if (dt == Dtype::Bool || dt == Dtype::I8 || dt == Dtype::I16) {
        // Accelerate has no vector arithmetic below 32 bits, so these
        // widen, reuse the I32 kernel, and narrow back.  Bool follows the
        // reference convention that ``True + True`` is ``True``: the sum
        // is taken as a number and then read as "non-zero".
        std::vector<std::int32_t> wa(n), wb(n), wo(n);
        const auto load = [&](const std::byte* src, std::vector<std::int32_t>& dst) {
            if (dt == Dtype::I16) {
                const auto* p = reinterpret_cast<const std::int16_t*>(src);
                for (std::size_t i = 0; i < n; ++i)
                    dst[i] = p[i];
            } else if (dt == Dtype::I8) {
                const auto* p = reinterpret_cast<const std::int8_t*>(src);
                for (std::size_t i = 0; i < n; ++i)
                    dst[i] = p[i];
            } else {
                const auto* p = reinterpret_cast<const std::uint8_t*>(src);
                for (std::size_t i = 0; i < n; ++i)
                    dst[i] = p[i] ? 1 : 0;
            }
        };
        load(reinterpret_cast<const std::byte*>(ca.ptr.get()), wa);
        load(reinterpret_cast<const std::byte*>(cb.ptr.get()), wb);
        fni32(wa.data(), wb.data(), wo.data(), n);
        if (dt == Dtype::I16) {
            auto* o = reinterpret_cast<std::int16_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                o[i] = static_cast<std::int16_t>(wo[i]);
        } else if (dt == Dtype::I8) {
            auto* o = reinterpret_cast<std::int8_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                o[i] = static_cast<std::int8_t>(wo[i]);
        } else {
            auto* o = reinterpret_cast<std::uint8_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                o[i] = wo[i] != 0 ? 1 : 0;
        }
    } else if (detail::is_half_like(dt)) {
        // Half has no host scalar to compute in; widen through float and
        // round once on the way back, which is what the astype path does.
        const auto* pa = reinterpret_cast<const std::uint16_t*>(ca.ptr.get());
        const auto* pb = reinterpret_cast<const std::uint16_t*>(cb.ptr.get());
        const bool brain = dt == Dtype::BF16;
        std::vector<float> wa(n), wb(n), wo(n);
        detail::half_or_bfloat16_to_float_bulk(pa, wa.data(), n, brain);
        detail::half_or_bfloat16_to_float_bulk(pb, wb.data(), n, brain);
        fn32(wa.data(), wb.data(), wo.data(), n);
        auto* o = reinterpret_cast<std::uint16_t*>(ptr.get());
        detail::float_to_half_or_bfloat16_bulk(wo.data(), o, n, brain);
    } else
        ErrorBuilder("cpu_backend::binary").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::ge_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(ge_mask(detail::as_f32(a), detail::as_f32(b), shape, Dtype::F32),
                                   dt);
    const auto& ca = std::get<CpuStorage>(a);
    const auto& cb = std::get<CpuStorage>(b);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    auto run_f = [&](auto tag_a, auto tag_b, auto tag_out) {
        using A = decltype(tag_a);
        using B = decltype(tag_b);
        using O = decltype(tag_out);
        const A* ap = reinterpret_cast<const A*>(ca.ptr.get());
        const B* bp = reinterpret_cast<const B*>(cb.ptr.get());
        O* op = reinterpret_cast<O*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = (ap[i] >= bp[i]) ? O{1} : O{0};
    };
    if (dt == Dtype::F32)
        run_f(float{}, float{}, float{});
    else if (dt == Dtype::F64)
        run_f(double{}, double{}, double{});
    else
        ErrorBuilder("cpu_backend::ge_mask").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::lt_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(lt_mask(detail::as_f32(a), detail::as_f32(b), shape, Dtype::F32),
                                   dt);
    const auto& ca = std::get<CpuStorage>(a);
    const auto& cb = std::get<CpuStorage>(b);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    auto run_f = [&](auto tag_a, auto tag_b, auto tag_out) {
        using A = decltype(tag_a);
        using B = decltype(tag_b);
        using O = decltype(tag_out);
        const A* ap = reinterpret_cast<const A*>(ca.ptr.get());
        const B* bp = reinterpret_cast<const B*>(cb.ptr.get());
        O* op = reinterpret_cast<O*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = (ap[i] < bp[i]) ? O{1} : O{0};
    };
    if (dt == Dtype::F32)
        run_f(float{}, float{}, float{});
    else if (dt == Dtype::F64)
        run_f(double{}, double{}, double{});
    else
        ErrorBuilder("cpu_backend::lt_mask").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::add_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) {
    // Accelerate's scalar/vector primitives are f32 and f64 only.  Widen at
    // the door, reuse, round once.  Reached constantly from backward, where
    // chain-rule constants arrive as scalars.
    if (detail::is_half_like(dt)) {
        const std::size_t half_n = shape_numel(shape);
        return detail::narrow_half(
            add_scalar(Storage{detail::widen_half(std::get<CpuStorage>(a), half_n)}, shape,
                       Dtype::F32, scalar),
            half_n, dt);
    }
    const auto& ca = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32)
        cpu::vsadd_f32(reinterpret_cast<const float*>(ca.ptr.get()), static_cast<float>(scalar),
                       reinterpret_cast<float*>(ptr.get()), n);
    else if (dt == Dtype::F64)
        cpu::vsadd_f64(reinterpret_cast<const double*>(ca.ptr.get()), scalar,
                       reinterpret_cast<double*>(ptr.get()), n);
    else
        ErrorBuilder("cpu_backend::add_scalar").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::mul_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) {
    // Accelerate's scalar/vector primitives are f32 and f64 only.  Widen at
    // the door, reuse, round once.  Reached constantly from backward, where
    // chain-rule constants arrive as scalars.
    if (detail::is_half_like(dt)) {
        const std::size_t half_n = shape_numel(shape);
        return detail::narrow_half(
            mul_scalar(Storage{detail::widen_half(std::get<CpuStorage>(a), half_n)}, shape,
                       Dtype::F32, scalar),
            half_n, dt);
    }
    const auto& ca = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32)
        cpu::vsmul_f32(reinterpret_cast<const float*>(ca.ptr.get()), static_cast<float>(scalar),
                       reinterpret_cast<float*>(ptr.get()), n);
    else if (dt == Dtype::F64)
        cpu::vsmul_f64(reinterpret_cast<const double*>(ca.ptr.get()), scalar,
                       reinterpret_cast<double*>(ptr.get()), n);
    else
        ErrorBuilder("cpu_backend::mul_scalar").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::in_range_mask(
    const Storage& a, const Shape& shape, Dtype dt, double lo, double hi, bool inclusive) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            in_range_mask(detail::as_f32(a), shape, Dtype::F32, lo, hi, inclusive), dt);
    const auto& ca = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const auto* p = reinterpret_cast<const float*>(ca.ptr.get());
        auto* q = reinterpret_cast<float*>(ptr.get());
        const auto flo = static_cast<float>(lo), fhi = static_cast<float>(hi);
        for (std::size_t i = 0; i < n; ++i)
            q[i] =
                (inclusive ? (p[i] >= flo && p[i] <= fhi) : (p[i] > flo && p[i] < fhi)) ? 1.f : 0.f;
    } else if (dt == Dtype::F64) {
        const auto* p = reinterpret_cast<const double*>(ca.ptr.get());
        auto* q = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = (inclusive ? (p[i] >= lo && p[i] <= hi) : (p[i] > lo && p[i] < hi)) ? 1.0 : 0.0;
    } else {
        ErrorBuilder("cpu_backend::in_range_mask").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::leaky_mask(const Storage& a, const Shape& shape, Dtype dt, double slope) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(leaky_mask(detail::as_f32(a), shape, Dtype::F32, slope), dt);
    const auto& ca = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const auto* p = reinterpret_cast<const float*>(ca.ptr.get());
        auto* q = reinterpret_cast<float*>(ptr.get());
        const float fs = static_cast<float>(slope);
        for (std::size_t i = 0; i < n; ++i)
            q[i] = (p[i] > 0.f) ? 1.f : fs;
    } else if (dt == Dtype::F64) {
        const auto* p = reinterpret_cast<const double*>(ca.ptr.get());
        auto* q = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = (p[i] > 0.0) ? 1.0 : slope;
    } else {
        ErrorBuilder("cpu_backend::leaky_mask").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::positive_mask(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(positive_mask(detail::as_f32(a), shape, Dtype::F32), dt);
    const auto& ca = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const auto* p = reinterpret_cast<const float*>(ca.ptr.get());
        auto* q = reinterpret_cast<float*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = (p[i] > 0.f) ? 1.f : 0.f;
    } else if (dt == Dtype::F64) {
        const auto* p = reinterpret_cast<const double*>(ca.ptr.get());
        auto* q = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = (p[i] > 0.0) ? 1.0 : 0.0;
    } else {
        ErrorBuilder("cpu_backend::positive_mask").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::floordiv(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    // Exact, in ``dt`` (DivMod.h).  floordiv_op no longer calls this.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            floordiv(detail::as_f32(a), detail::as_f32(b), shape, Dtype::F32), dt);
    return Storage{cpu::divmod(DivMod::FloorDiv, std::get<CpuStorage>(a), std::get<CpuStorage>(b),
                               shape_numel(shape), dt)};
}

Storage CpuBackend::where_op(
    const Storage& cond, const Storage& x, const Storage& y, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            where_op(detail::as_f32(cond), detail::as_f32(x), detail::as_f32(y), shape, Dtype::F32),
            dt);
    const auto& cc = std::get<CpuStorage>(cond);
    const auto& cx = std::get<CpuStorage>(x);
    const auto& cy = std::get<CpuStorage>(y);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const auto* c = reinterpret_cast<const std::uint8_t*>(cc.ptr.get());
    // Both branches are loaded and one selected: a conditional load is a
    // branch on the condition, and a data condition (``x > 0``) is a coin
    // flip.  Restrict locals let the select run as a vector blend.
    auto run = [&](auto* dst_in, const auto* xp_in, const auto* yp_in) {
        using T = std::remove_pointer_t<decltype(dst_in)>;
        cpu::parallel_for(n, kStreamGrain, [&](std::size_t lo, std::size_t hi) {
            const std::uint8_t* __restrict cp = c;
            const T* __restrict xp = xp_in;
            const T* __restrict yp = yp_in;
            T* __restrict dst = dst_in;
            for (std::size_t i = lo; i < hi; ++i) {
                const T a = xp[i];
                const T b = yp[i];
                dst[i] = cp[i] ? a : b;
            }
        });
    };
    // Selection copies bits; it never reads them as numbers.  Dispatching
    // on element width rather than dtype therefore covers every type —
    // bool, the integers and both float widths — with four branches and
    // no per-type code.  The float-only version this replaces was why a
    // masked integer index had to be written as a clamp elsewhere.
    switch (dtype_size(dt)) {
    case 1:
        run(reinterpret_cast<std::uint8_t*>(ptr.get()),
            reinterpret_cast<const std::uint8_t*>(cx.ptr.get()),
            reinterpret_cast<const std::uint8_t*>(cy.ptr.get()));
        break;
    case 2:
        run(reinterpret_cast<std::uint16_t*>(ptr.get()),
            reinterpret_cast<const std::uint16_t*>(cx.ptr.get()),
            reinterpret_cast<const std::uint16_t*>(cy.ptr.get()));
        break;
    case 4:
        run(reinterpret_cast<std::uint32_t*>(ptr.get()),
            reinterpret_cast<const std::uint32_t*>(cx.ptr.get()),
            reinterpret_cast<const std::uint32_t*>(cy.ptr.get()));
        break;
    case 8:
        run(reinterpret_cast<std::uint64_t*>(ptr.get()),
            reinterpret_cast<const std::uint64_t*>(cx.ptr.get()),
            reinterpret_cast<const std::uint64_t*>(cy.ptr.get()));
        break;
    default:
        ErrorBuilder("cpu_backend::where_op").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

}  // namespace backend
}  // namespace lucid
