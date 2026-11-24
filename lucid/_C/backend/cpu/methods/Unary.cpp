// lucid/_C/backend/cpu/methods/Unary.cpp
// Out-of-line CpuBackend method definitions for the Unary family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

Storage CpuBackend::exp(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::vexp_f32, cpu::vexp_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::exp(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::log(const Storage& a, const Shape& shape, Dtype dt) {
    if (is_complex(dt)) {
        // vForce is real-only, so ``unary_op`` has no complex branch.
        const auto& cs = std::get<CpuStorage>(a);
        const std::size_t n = shape_numel(shape);
        const std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::C128)
            cpu::vzlog(reinterpret_cast<const double*>(cs.ptr.get()),
                       reinterpret_cast<double*>(ptr.get()), n);
        else
            cpu::vzlog(reinterpret_cast<const float*>(cs.ptr.get()),
                       reinterpret_cast<float*>(ptr.get()), n);
        return Storage{CpuStorage{ptr, nb, dt}};
    }
    return unary_op(a, shape, dt, cpu::vlog_f32, cpu::vlog_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::log(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::sqrt(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::vsqrt_f32, cpu::vsqrt_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::sqrt(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::rsqrt(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(rsqrt(detail::as_f32(a), shape, Dtype::F32), dt);
    auto sq = sqrt(a, shape, dt);
    const auto& cs = std::get<CpuStorage>(sq);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32)
        cpu::vrec_f32(reinterpret_cast<const float*>(cs.ptr.get()),
                      reinterpret_cast<float*>(ptr.get()), n);
    else if (dt == Dtype::F64)
        cpu::vrec_f64(reinterpret_cast<const double*>(cs.ptr.get()),
                      reinterpret_cast<double*>(ptr.get()), n);
    else
        ErrorBuilder("cpu_backend::rsqrt").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::abs(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(
        a, shape, dt, [](const float* ip, float* op, std::size_t n) { cpu::vabs_f32(ip, op, n); },
        [](const double* ip, double* op, std::size_t n) { cpu::vabs_f64(ip, op, n); },
        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = std::abs(ip[i]);
        });
}

Storage CpuBackend::neg(const Storage& a, const Shape& shape, Dtype dt) {
    if (is_complex(dt))
        return sub(zeros(shape, dt), a, shape, dt);
    // A bool has no negation.  It went through the integer path as
    // uint8, where ``-1`` wraps to 255 and 255 reads back as ``true``,
    // so ``-tensor([True, False])`` answered ``[True, False]`` — the
    // input, unchanged, with nothing raised.  Metal refuses this and
    // says to use logical_not; so does the reference.  Refuse here too,
    // and point at the same op, so the two devices agree on both the
    // outcome and the reason.
    if (dt == Dtype::Bool)
        ErrorBuilder("cpu_backend::neg")
            .not_implemented("negation is not defined for bool — use logical_not");
    return unary_op(
        a, shape, dt, [](const float* ip, float* op, std::size_t n) { cpu::vneg_f32(ip, op, n); },
        [](const double* ip, double* op, std::size_t n) { cpu::vneg_f64(ip, op, n); },
        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = -ip[i];
        });
}

Storage CpuBackend::sign(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernel here; widen at the door and round
    // once on the way out.  See detail::widen_half.
    if (detail::is_half_like(dt)) {
        const std::size_t half_n = shape_numel(shape);
        return detail::narrow_half(
            sign(Storage{detail::widen_half(std::get<CpuStorage>(a), half_n)}, shape, Dtype::F32),
            half_n, dt);
    }
    // Integers take the same road, through int64.  Selection, movement
    // and addition are all exact there, so this is the answer a native
    // kernel would have produced rather than an approximation of it.
    if (detail::is_narrow_int(dt))
        return detail::back_to_int(sign(detail::as_i64(a), shape, Dtype::I64), dt);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const auto& cs = std::get<CpuStorage>(a);
    if (dt == Dtype::F32) {
        const float* ip = reinterpret_cast<const float*>(cs.ptr.get());
        float* op = reinterpret_cast<float*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = (ip[i] > 0.f) ? 1.f : ((ip[i] < 0.f) ? -1.f : 0.f);
    } else if (dt == Dtype::F64) {
        const double* ip = reinterpret_cast<const double*>(cs.ptr.get());
        double* op = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = (ip[i] > 0.0) ? 1.0 : ((ip[i] < 0.0) ? -1.0 : 0.0);
    } else if (dt == Dtype::I32) {
        const std::int32_t* ip = reinterpret_cast<const std::int32_t*>(cs.ptr.get());
        std::int32_t* op = reinterpret_cast<std::int32_t*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = (ip[i] > 0) ? 1 : ((ip[i] < 0) ? -1 : 0);
    } else if (dt == Dtype::I64) {
        const std::int64_t* ip = reinterpret_cast<const std::int64_t*>(cs.ptr.get());
        std::int64_t* op = reinterpret_cast<std::int64_t*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = (ip[i] > 0) ? 1 : ((ip[i] < 0) ? -1 : 0);
    } else {
        ErrorBuilder("cpu_backend::sign").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::floor(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(
        a, shape, dt, cpu::vfloor_f32, cpu::vfloor_f64,
        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) { std::copy(ip, ip + n, op); });
}

Storage CpuBackend::ceil(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(
        a, shape, dt, cpu::vceil_f32, cpu::vceil_f64,
        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) { std::copy(ip, ip + n, op); });
}

Storage CpuBackend::round(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(
        a, shape, dt, cpu::vround_f32, cpu::vround_f64,
        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) { std::copy(ip, ip + n, op); });
}

Storage CpuBackend::sin(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::vsin_f32, cpu::vsin_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::sin(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::cos(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::vcos_f32, cpu::vcos_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::cos(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::tanh(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::vtanh_f32, cpu::vtanh_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::tanh(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::log1p(const Storage& a, const Shape& shape, Dtype dt) {
    const auto n = shape_numel(shape);
    const auto bytes = n * dtype_size(dt);
    auto data = allocate_aligned_bytes(bytes, Device::CPU);
    const auto& src = std::get<CpuStorage>(a).ptr;
    if (dt == Dtype::F32)
        cpu::vlog1p_f32(reinterpret_cast<const float*>(src.get()),
                        reinterpret_cast<float*>(data.get()), n);
    else if (dt == Dtype::F64)
        cpu::vlog1p_f64(reinterpret_cast<const double*>(src.get()),
                        reinterpret_cast<double*>(data.get()), n);
    else
        ErrorBuilder("cpu_backend::log1p").not_implemented("requires float32 or float64");
    return Storage{CpuStorage{std::move(data), bytes, dt}};
}

Storage CpuBackend::log2(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::vlog2_f32, cpu::vlog2_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::log2(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::erf(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::verf_f32, cpu::verf_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::erf(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::erfinv(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(erfinv(detail::as_f32(a), shape, Dtype::F32), dt);
    // erfinv via Winitzki (2008) initial guess + 3 Newton steps.
    static constexpr double kTwoOverSqrtPi = 1.1283791670955126;
    static constexpr double kA = 0.147;  // Winitzki constant

    auto scalar_erfinv = [&](double x) -> double {
        // erf maps onto (-1, 1), so only the two endpoints are infinite;
        // anything outside is undefined and must be NaN.  Collapsing
        // ``x >= 1`` to +inf swallowed the whole out-of-domain half, and
        // an inf is more dangerous than a NaN downstream — ``1/inf`` and
        // ``exp(-inf)`` quietly become finite, so a domain error could
        // vanish instead of propagating.  (Metal already matched SciPy.)
        if (std::isnan(x))
            return x;
        if (x > 1.0 || x < -1.0)
            return std::numeric_limits<double>::quiet_NaN();
        if (x == 1.0)
            return std::numeric_limits<double>::infinity();
        if (x == -1.0)
            return -std::numeric_limits<double>::infinity();
        if (x == 0.0)
            return x;
        double s = (x > 0) ? 1.0 : -1.0;
        double ax = std::abs(x);
        double ln1 = std::log(1.0 - ax * ax);
        double b = 2.0 / (M_PI * kA) + ln1 * 0.5;
        double y = s * std::sqrt(std::sqrt(b * b - ln1 / kA) - b);
        // Newton refinement: y_new = y - (erf(y) - x) / ((2/sqrt(pi))*exp(-y^2))
        for (int k = 0; k < 3; ++k) {
            double ey = std::erf(y);
            double dy = (ey - x) / (kTwoOverSqrtPi * std::exp(-y * y));
            y -= dy;
        }
        return y;
    };

    auto run_f32 = [&](const float* ip, float* op, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i)
            op[i] = static_cast<float>(scalar_erfinv(static_cast<double>(ip[i])));
    };
    auto run_f64 = [&](const double* ip, double* op, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i)
            op[i] = scalar_erfinv(ip[i]);
    };

    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32)
        run_f32(reinterpret_cast<const float*>(cs.ptr.get()), reinterpret_cast<float*>(ptr.get()),
                n);
    else if (dt == Dtype::F64)
        run_f64(reinterpret_cast<const double*>(cs.ptr.get()), reinterpret_cast<double*>(ptr.get()),
                n);
    else
        ErrorBuilder("cpu_backend::erfinv").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::reciprocal(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernel here; widen at the door and round
    // once on the way out.  See detail::widen_half.
    if (detail::is_half_like(dt)) {
        const std::size_t half_n = shape_numel(shape);
        return detail::narrow_half(
            reciprocal(Storage{detail::widen_half(std::get<CpuStorage>(a), half_n)}, shape,
                       Dtype::F32),
            half_n, dt);
    }
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32)
        cpu::vrec_f32(reinterpret_cast<const float*>(cs.ptr.get()),
                      reinterpret_cast<float*>(ptr.get()), n);
    else if (dt == Dtype::F64)
        cpu::vrec_f64(reinterpret_cast<const double*>(cs.ptr.get()),
                      reinterpret_cast<double*>(ptr.get()), n);
    else
        ErrorBuilder("cpu_backend::reciprocal").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::square(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::vsq_f32, cpu::vsq_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = ip[i] * ip[i];
                    });
}

Storage CpuBackend::cube(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(cube(detail::as_f32(a), shape, Dtype::F32), dt);
    // x·x·x is arithmetic, not a selection, so it goes through int64 —
    // exact for every narrower width whose cube fits, and wrapping the
    // same way a native kernel would for the ones that do not, which is
    // what naming the width asks for.
    if (detail::is_narrow_int(dt))
        return detail::back_to_int(cube(detail::as_i64(a), shape, Dtype::I64), dt);
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        auto* p = reinterpret_cast<const float*>(cs.ptr.get());
        auto* q = reinterpret_cast<float*>(ptr.get());
        cpu::vsq_f32(p, q, n);
        cpu::vmul_f32(p, q, q, n);
    } else if (dt == Dtype::F64) {
        auto* p = reinterpret_cast<const double*>(cs.ptr.get());
        auto* q = reinterpret_cast<double*>(ptr.get());
        cpu::vsq_f64(p, q, n);
        cpu::vmul_f64(p, q, q, n);
    } else if (dt == Dtype::I64) {
        // Written out rather than routed through double: the cube of an
        // int64 leaves double's exact range at |x| > 2^17, so the widen
        // that works for a selection would quietly round here.
        const auto* p = reinterpret_cast<const std::int64_t*>(cs.ptr.get());
        auto* q = reinterpret_cast<std::int64_t*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = p[i] * p[i] * p[i];
    } else {
        ErrorBuilder("cpu_backend::cube").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::cube_root(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(
        a, shape, dt,
        [](const float* ip, float* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = std::cbrt(ip[i]);
        },
        [](const double* ip, double* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = std::cbrt(ip[i]);
        },
        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i)
                op[i] = static_cast<std::int32_t>(std::cbrt(static_cast<float>(ip[i])));
        });
}

Storage CpuBackend::tan(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::vtan_f32, cpu::vtan_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::tan(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::asin(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::vasin_f32, cpu::vasin_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::asin(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::acos(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::vacos_f32, cpu::vacos_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::acos(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::atan(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::vatan_f32, cpu::vatan_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::atan(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::sinh(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::vsinh_f32, cpu::vsinh_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::sinh(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::cosh(const Storage& a, const Shape& shape, Dtype dt) {
    return unary_op(a, shape, dt, cpu::vcosh_f32, cpu::vcosh_f64,
                    [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                        for (std::size_t i = 0; i < n; ++i)
                            op[i] = static_cast<std::int32_t>(std::cosh(static_cast<float>(ip[i])));
                    });
}

Storage CpuBackend::invert(const Storage& a, const Shape& shape, Dtype dt) {
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::Bool) {
        const std::uint8_t* ip = reinterpret_cast<const std::uint8_t*>(cs.ptr.get());
        std::uint8_t* op = reinterpret_cast<std::uint8_t*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = ip[i] ? 0 : 1;
    } else if (dt == Dtype::I8) {
        const std::int8_t* ip = reinterpret_cast<const std::int8_t*>(cs.ptr.get());
        std::int8_t* op = reinterpret_cast<std::int8_t*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = ~ip[i];
    } else if (dt == Dtype::I16) {
        const std::int16_t* ip = reinterpret_cast<const std::int16_t*>(cs.ptr.get());
        std::int16_t* op = reinterpret_cast<std::int16_t*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = ~ip[i];
    } else if (dt == Dtype::I32) {
        const std::int32_t* ip = reinterpret_cast<const std::int32_t*>(cs.ptr.get());
        std::int32_t* op = reinterpret_cast<std::int32_t*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = ~ip[i];
    } else if (dt == Dtype::I64) {
        const std::int64_t* ip = reinterpret_cast<const std::int64_t*>(cs.ptr.get());
        std::int64_t* op = reinterpret_cast<std::int64_t*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = ~ip[i];
    } else {
        ErrorBuilder("cpu_backend::invert").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::project_lane(const Storage& a, const Shape& shape, std::size_t lane) {
    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t n = shape_numel(shape);
    const Dtype out_dt = real_lane_of(cs.dtype);
    const std::size_t nb = n * dtype_size(out_dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (out_dt == Dtype::F64) {
        const auto* ip = reinterpret_cast<const double*>(cs.ptr.get());
        auto* op = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = ip[2 * i + lane];
    } else {
        const auto* ip = reinterpret_cast<const float*>(cs.ptr.get());
        auto* op = reinterpret_cast<float*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = ip[2 * i + lane];
    }
    return Storage{CpuStorage{ptr, nb, out_dt}};
}

Storage CpuBackend::complex_combine(const Storage& re, const Storage& im, const Shape& shape) {
    const auto& re_s = std::get<CpuStorage>(re);
    const auto& im_s = std::get<CpuStorage>(im);
    const std::size_t n = shape_numel(shape);

    // The lane width follows the inputs, and used to not.
    //
    // Both halves were read as ``float*`` whatever they were, so a
    // pair of f64 inputs was interleaved out of the wrong bytes:
    // ``complex([1., 2.], [3., 4.])`` answered
    // ``[0+0j, 1.875+2.125j]``.  Not an error, not a NaN — four
    // plausible numbers assembled from the halves of four doubles.
    // f16 was the same, differently wrong.  Only f32 ever worked, and
    // f32 is what every test used.
    //
    // The op layer widens the half formats to f32 before dispatch, so
    // only the two widths that have a complex type reach here.
    const Dtype out_dt = complex_for(re_s.dtype);
    const std::size_t nb = n * dtype_size(out_dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);

    if (out_dt == Dtype::C128) {
        const auto* rp = reinterpret_cast<const double*>(re_s.ptr.get());
        const auto* ip = reinterpret_cast<const double*>(im_s.ptr.get());
        auto* op = reinterpret_cast<double*>(ptr.get());
        DSPDoubleSplitComplex split{const_cast<double*>(rp), const_cast<double*>(ip)};
        vDSP_ztocD(&split, 1, reinterpret_cast<DSPDoubleComplex*>(op), 2,
                   static_cast<vDSP_Length>(n));
    } else {
        const auto* rp = reinterpret_cast<const float*>(re_s.ptr.get());
        const auto* ip = reinterpret_cast<const float*>(im_s.ptr.get());
        auto* op = reinterpret_cast<float*>(ptr.get());
        // vDSP's ``vDSP_ztoc`` interleaves from split-complex form
        // (separate re / im pointers); the inputs match that layout.
        DSPSplitComplex split{const_cast<float*>(rp), const_cast<float*>(ip)};
        vDSP_ztoc(&split, 1, reinterpret_cast<DSPComplex*>(op), 2, static_cast<vDSP_Length>(n));
    }
    return Storage{CpuStorage{ptr, nb, out_dt}};
}

Storage CpuBackend::complex_conj(const Storage& a, const Shape& shape, Dtype dt) {
    // Real dtypes: the conjugate is the identity, and returning the
    // input storage directly avoids a needless allocation and copy.
    //
    // The test was ``dt != Dtype::C64`` when C64 was the only complex
    // type, which made "not C64" and "real" the same statement.  They
    // stopped being the same the moment C128 existed, and a complex128
    // tensor took the identity branch: ``conj`` returned its argument
    // unchanged, with no error and no clue.
    if (!is_complex(dt))
        return a;

    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t n = shape_numel(shape);
    const std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::C128) {
        const auto* ip = reinterpret_cast<const double*>(cs.ptr.get());
        auto* op = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            op[2 * i] = ip[2 * i];
            op[2 * i + 1] = -ip[2 * i + 1];
        }
    } else {
        cpu::vzconj_c64(reinterpret_cast<const float*>(cs.ptr.get()),
                        reinterpret_cast<float*>(ptr.get()), n);
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

template <cpu::NonFiniteProbe P>
Storage CpuBackend::nonfinite_probe(const Storage& a, const Shape& shape, Dtype dt) {
    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t n = shape_numel(shape);
    auto ptr = allocate_aligned_bytes(n, Device::CPU);
    auto* dst = reinterpret_cast<std::uint8_t*>(ptr.get());
    const void* src = cs.ptr.get();
    switch (dt) {
    case Dtype::F16:
        cpu::probe_nonfinite<P, cpu::HalfFormat>(src, dst, n, kStreamGrain);
        break;
    case Dtype::BF16:
        cpu::probe_nonfinite<P, cpu::BrainFormat>(src, dst, n, kStreamGrain);
        break;
    case Dtype::F32:
        cpu::probe_nonfinite<P, cpu::SingleFormat>(src, dst, n, kStreamGrain);
        break;
    case Dtype::F64:
        cpu::probe_nonfinite<P, cpu::DoubleFormat>(src, dst, n, kStreamGrain);
        break;
    case Dtype::C64:
        cpu::probe_nonfinite<P, cpu::SingleFormat, true>(src, dst, n, kStreamGrain);
        break;
    case Dtype::C128:
        cpu::probe_nonfinite<P, cpu::DoubleFormat, true>(src, dst, n, kStreamGrain);
        break;
    default:
        std::fill_n(dst, n, P == cpu::NonFiniteProbe::IsFinite ? std::uint8_t{1} : std::uint8_t{0});
        break;
    }
    return Storage{CpuStorage{ptr, n, Dtype::Bool}};
}

Storage CpuBackend::isinf(const Storage& a, const Shape& shape, Dtype dt) {
    return nonfinite_probe<cpu::NonFiniteProbe::IsInf>(a, shape, dt);
}

Storage CpuBackend::isnan(const Storage& a, const Shape& shape, Dtype dt) {
    return nonfinite_probe<cpu::NonFiniteProbe::IsNan>(a, shape, dt);
}

Storage CpuBackend::isfinite(const Storage& a, const Shape& shape, Dtype dt) {
    return nonfinite_probe<cpu::NonFiniteProbe::IsFinite>(a, shape, dt);
}

bool CpuBackend::element_nonzero(const std::byte* base, std::size_t i, Dtype dt) {
    switch (dt) {
    case Dtype::Bool:
    case Dtype::I8: {
        std::int8_t v;
        std::memcpy(&v, base + i, 1);
        return v != 0;
    }
    case Dtype::I16: {
        std::int16_t v;
        std::memcpy(&v, base + i * 2, 2);
        return v != 0;
    }
    case Dtype::I32: {
        std::int32_t v;
        std::memcpy(&v, base + i * 4, 4);
        return v != 0;
    }
    case Dtype::I64: {
        std::int64_t v;
        std::memcpy(&v, base + i * 8, 8);
        return v != 0;
    }
    case Dtype::F16:
    case Dtype::BF16: {
        std::uint16_t v;
        std::memcpy(&v, base + i * 2, 2);
        return (v & 0x7FFFu) != 0;
    }
    case Dtype::F32: {
        float v;
        std::memcpy(&v, base + i * 4, 4);
        return v != 0.f;
    }
    case Dtype::F64: {
        double v;
        std::memcpy(&v, base + i * 8, 8);
        return v != 0.0;
    }
    case Dtype::C64: {
        float v[2];
        std::memcpy(v, base + i * 8, 8);
        return v[0] != 0.f || v[1] != 0.f;
    }
    case Dtype::C128: {
        double v[2];
        std::memcpy(v, base + i * 16, 16);
        return v[0] != 0.0 || v[1] != 0.0;
    }
    }
    return false;
}

Storage CpuBackend::any(const Storage& a, const Shape& shape, Dtype dt) {
    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t n = shape_numel(shape);
    auto ptr = allocate_aligned_bytes(1, Device::CPU);
    auto* dst = reinterpret_cast<std::uint8_t*>(ptr.get());
    const auto* base = reinterpret_cast<const std::byte*>(cs.ptr.get());
    bool found = false;
    for (std::size_t i = 0; i < n && !found; ++i)
        found = element_nonzero(base, i, dt);
    dst[0] = found ? 1u : 0u;
    return Storage{CpuStorage{ptr, 1, Dtype::Bool}};
}

Storage CpuBackend::all(const Storage& a, const Shape& shape, Dtype dt) {
    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t n = shape_numel(shape);
    auto ptr = allocate_aligned_bytes(1, Device::CPU);
    auto* dst = reinterpret_cast<std::uint8_t*>(ptr.get());
    const auto* base = reinterpret_cast<const std::byte*>(cs.ptr.get());
    bool all_nz = true;
    for (std::size_t i = 0; i < n && all_nz; ++i)
        all_nz = element_nonzero(base, i, dt);
    dst[0] = all_nz ? 1u : 0u;
    return Storage{CpuStorage{ptr, 1, Dtype::Bool}};
}

Storage CpuBackend::nan_to_num(const Storage& a,
                               const Shape& shape,
                               Dtype dt,
                               double nan_val,
                               double posinf_val,
                               double neginf_val) {
    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t n = shape_numel(shape);
    const std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const void* src = cs.ptr.get();
    void* dst = ptr.get();
    // Each replacement is rounded into the dtype once, as a cast rounds
    // it (through float for the 16-bit formats, as the reference
    // framework's half types do).  A value the dtype cannot hold becomes
    // its infinity, as an explicit out-of-range replacement does in the
    // reference framework.  When no replacement is given, the op passes
    // the dtype's own extremes, which every format holds exactly.
    //
    // Only float32 and float64 used to be replaced.  float16 and bfloat16
    // came back untouched.  A complex tensor is replaced part by part:
    // its 2n interleaved floats go through the real kernel.
    auto run = [&](auto format, std::size_t words, auto to_bits) {
        using Format = decltype(format);
        cpu::replace_nonfinite<Format>(src, dst, words, to_bits(nan_val), to_bits(posinf_val),
                                       to_bits(neginf_val), kStreamGrain);
    };
    auto half_bits = [](double v) { return detail::float_to_half_bits(static_cast<float>(v)); };
    auto brain_bits = [](double v) { return detail::float_to_bfloat_bits(static_cast<float>(v)); };
    auto single_bits = [](double v) { return std::bit_cast<std::uint32_t>(static_cast<float>(v)); };
    auto double_bits = [](double v) { return std::bit_cast<std::uint64_t>(v); };
    switch (dt) {
    case Dtype::F16:
        run(cpu::HalfFormat{}, n, half_bits);
        break;
    case Dtype::BF16:
        run(cpu::BrainFormat{}, n, brain_bits);
        break;
    case Dtype::F32:
        run(cpu::SingleFormat{}, n, single_bits);
        break;
    case Dtype::F64:
        run(cpu::DoubleFormat{}, n, double_bits);
        break;
    case Dtype::C64:
        run(cpu::SingleFormat{}, 2 * n, single_bits);
        break;
    case Dtype::C128:
        run(cpu::DoubleFormat{}, 2 * n, double_bits);
        break;
    default:
        // An integer or bool tensor holds nothing to replace.
        if (nb != 0)
            std::memcpy(dst, src, nb);
        break;
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

template <class F32Fn, class F64Fn, class I32Fn>
Storage CpuBackend::unary_op(
    const Storage& a, const Shape& shape, Dtype dt, F32Fn fn32, F64Fn fn64, I32Fn fni32) {
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32)
        fn32(reinterpret_cast<const float*>(cs.ptr.get()), reinterpret_cast<float*>(ptr.get()), n);
    else if (dt == Dtype::F64)
        fn64(reinterpret_cast<const double*>(cs.ptr.get()), reinterpret_cast<double*>(ptr.get()),
             n);
    else if (dt == Dtype::I32)
        fni32(reinterpret_cast<const std::int32_t*>(cs.ptr.get()),
              reinterpret_cast<std::int32_t*>(ptr.get()), n);
    else if (dt == Dtype::I64) {
        // This branch used to read
        //     op[i] = (std::int64_t)(double)(double)ip[i];
        // which is an identity: the function was never applied, so
        // every unary op on I64 returned its own input.  I64 is the
        // default integer dtype, so ``exp(tensor([2, 3]))`` answered
        // ``[2, 3]`` instead of ``[7, 20]`` — silently, on the CPU
        // only, while Metal was correct.
        const std::int64_t* ip = reinterpret_cast<const std::int64_t*>(cs.ptr.get());
        std::int64_t* op = reinterpret_cast<std::int64_t*>(ptr.get());
        std::vector<double> wi(n), wo(n);
        for (std::size_t i = 0; i < n; ++i)
            wi[i] = static_cast<double>(ip[i]);
        fn64(wi.data(), wo.data(), n);
        for (std::size_t i = 0; i < n; ++i)
            op[i] = static_cast<std::int64_t>(wo[i]);
    } else if (dt == Dtype::Bool) {
        // Bool goes through *float*, not through the integer kernel.
        // Truncating first makes ``tanh(True)`` land on 0 and read back
        // as False, while a float-to-bool cast is "non-zero" and gives
        // True — which is what Metal and the reference both do.
        const auto* p = reinterpret_cast<const std::uint8_t*>(cs.ptr.get());
        std::vector<float> wi(n), wo(n);
        for (std::size_t i = 0; i < n; ++i)
            wi[i] = p[i] ? 1.0f : 0.0f;
        fn32(wi.data(), wo.data(), n);
        auto* o = reinterpret_cast<std::uint8_t*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            o[i] = wo[i] != 0.0f ? 1 : 0;
    } else if (dt == Dtype::I8 || dt == Dtype::I16) {
        // Accelerate has no vector maths below 32 bits, so these widen,
        // reuse the I32 kernel and narrow back — the same shape as
        // ``binary_op``.  Metal supported all three and the CPU did
        // not, so the same call worked on one device and raised on the
        // other.
        std::vector<std::int32_t> wi(n), wo(n);
        if (dt == Dtype::I16) {
            const auto* p = reinterpret_cast<const std::int16_t*>(cs.ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                wi[i] = p[i];
        } else if (dt == Dtype::I8) {
            const auto* p = reinterpret_cast<const std::int8_t*>(cs.ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                wi[i] = p[i];
        } else {
            const auto* p = reinterpret_cast<const std::int8_t*>(cs.ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                wi[i] = p[i];
        }
        fni32(wi.data(), wo.data(), n);
        if (dt == Dtype::I16) {
            auto* o = reinterpret_cast<std::int16_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                o[i] = static_cast<std::int16_t>(wo[i]);
        } else if (dt == Dtype::I8) {
            auto* o = reinterpret_cast<std::int8_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                o[i] = static_cast<std::int8_t>(wo[i]);
        } else {
            auto* o = reinterpret_cast<std::int8_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                o[i] = static_cast<std::int8_t>(wo[i]);
        }
    } else if (detail::is_half_like(dt)) {
        // Half has no host scalar to compute in; widen through float
        // and round once on the way back, which is what the astype
        // path does.
        const auto* p = reinterpret_cast<const std::uint16_t*>(cs.ptr.get());
        const bool brain = dt == Dtype::BF16;
        std::vector<float> wi(n), wo(n);
        detail::half_or_bfloat16_to_float_bulk(p, wi.data(), n, brain);
        fn32(wi.data(), wo.data(), n);
        auto* o = reinterpret_cast<std::uint16_t*>(ptr.get());
        detail::float_to_half_or_bfloat16_bulk(wo.data(), o, n, brain);
    } else {
        ErrorBuilder("cpu_backend::unary").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

}  // namespace backend
}  // namespace lucid
