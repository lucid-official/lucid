// lucid/_C/backend/cpu/methods/Activation.cpp
// Out-of-line CpuBackend method definitions for the Activation family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

Storage CpuBackend::sigmoid(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernel here; widen at the door and
    // round once on the way out.  See detail::widen_half.
    if (detail::is_half_like(dt)) {
        const std::size_t half_n = shape_numel(shape);
        const Storage wide = Storage{detail::widen_half(std::get<CpuStorage>(a), half_n)};
        return detail::narrow_half(sigmoid(wide, shape, Dtype::F32), half_n, dt);
    }
    // One pass over L1 tiles, split across cores.  The chain this replaces
    // — neg, exp, ones, add, reciprocal — wrote three full-size
    // temporaries and ran on one core, 2.4 times the reference.  Each
    // element still takes the same operations in the same order (vForce
    // exp of -x, plus one, the correctly rounded reciprocal
    // cpu::vrec_f32/f64), so the bits are unchanged.
    const auto& ca = std::get<CpuStorage>(a);
    const std::size_t n = shape_numel(shape);
    const std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    auto run = [&](const auto* in, auto* out, auto vexp, auto vrec) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(in)>>;
        cpu::parallel_for(n, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
            T tile[kTanhTile];
            for (std::size_t s = lo; s < hi; s += kTanhTile) {
                const std::size_t m = std::min(kTanhTile, hi - s);
                for (std::size_t i = 0; i < m; ++i)
                    tile[i] = -in[s + i];
                vexp(tile, tile, m);
                for (std::size_t i = 0; i < m; ++i)
                    tile[i] = T(1) + tile[i];
                vrec(tile, out + s, m);
            }
        });
    };
    if (dt == Dtype::F32) {
        run(reinterpret_cast<const float*>(ca.ptr.get()), reinterpret_cast<float*>(ptr.get()),
            &cpu::vexp_f32, &cpu::vrec_f32);
    } else if (dt == Dtype::F64) {
        run(reinterpret_cast<const double*>(ca.ptr.get()), reinterpret_cast<double*>(ptr.get()),
            &cpu::vexp_f64, &cpu::vrec_f64);
    } else {
        ErrorBuilder("cpu_backend::sigmoid").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::relu(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernel here; widen at the door and
    // round once on the way out.  See detail::widen_half.
    if (detail::is_half_like(dt)) {
        const std::size_t half_n = shape_numel(shape);
        const Storage wide = Storage{detail::widen_half(std::get<CpuStorage>(a), half_n)};
        return detail::narrow_half(relu(wide, shape, Dtype::F32), half_n, dt);
    }
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const auto& cs = std::get<CpuStorage>(a);
    // ``max(0, x)`` needs no arithmetic beyond a comparison, so every
    // integer width computes it in itself.  Written out rather than
    // delegated to a wide path: there is no I64 kernel here to delegate
    // *to*, which is what a first attempt at widening ran into.
    // The reference framework answers relu on integers and keeps the
    // dtype; Metal already did.
    auto run_int = [&](auto tag) {
        using T = decltype(tag);
        const T* ip = reinterpret_cast<const T*>(cs.ptr.get());
        T* op = reinterpret_cast<T*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            op[i] = ip[i] < T{} ? T{} : ip[i];
    };
    switch (dt) {
    case Dtype::F32:
        cpu::vrelu_f32(reinterpret_cast<const float*>(cs.ptr.get()),
                       reinterpret_cast<float*>(ptr.get()), n);
        break;
    case Dtype::F64:
        cpu::vrelu_f64(reinterpret_cast<const double*>(cs.ptr.get()),
                       reinterpret_cast<double*>(ptr.get()), n);
        break;
    case Dtype::I64:
        run_int(std::int64_t{});
        break;
    case Dtype::I32:
        run_int(std::int32_t{});
        break;
    case Dtype::I16:
        run_int(std::int16_t{});
        break;
    case Dtype::I8:
        run_int(std::int8_t{});
        break;
    case Dtype::Bool:
        // Already non-negative; relu is the identity on a mask.
        std::memcpy(ptr.get(), cs.ptr.get(), nb);
        break;
    default:
        ErrorBuilder("cpu_backend::relu").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::silu(const Storage& a, const Shape& shape, Dtype dt) {
    auto sig = sigmoid(a, shape, dt);
    return mul(a, sig, shape, dt);
}

Storage CpuBackend::gelu(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(gelu(detail::as_f32(a), shape, Dtype::F32), dt);
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    constexpr double kC1 = 0.7978845608028654;
    constexpr double kC2 = 0.044715;
    // The tanh runs through vForce a tile at a time, on every core.
    // This was one scalar ``tanhf`` call per element on one core —
    // 4.1 ms on 524k elements, ten times the exact GELU it is meant to
    // be a cheaper stand-in for.
    if (dt == Dtype::F32) {
        const float* p = reinterpret_cast<const float*>(cs.ptr.get());
        float* q = reinterpret_cast<float*>(ptr.get());
        const float c1 = static_cast<float>(kC1);
        const float c2 = static_cast<float>(kC2);
        cpu::parallel_for(n, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
            alignas(64) float tile[kTanhTile];
            for (std::size_t base = lo; base < hi; base += kTanhTile) {
                const std::size_t m = std::min(kTanhTile, hi - base);
                for (std::size_t i = 0; i < m; ++i) {
                    const float x = p[base + i];
                    tile[i] = c1 * (x + c2 * x * x * x);
                }
                cpu::vtanh_unfixed_f32(tile, tile, m);
                for (std::size_t i = 0; i < m; ++i)
                    q[base + i] = 0.5f * p[base + i] * (1.f + tile[i]);
            }
        });
    } else if (dt == Dtype::F64) {
        const double* p = reinterpret_cast<const double*>(cs.ptr.get());
        double* q = reinterpret_cast<double*>(ptr.get());
        cpu::parallel_for(n, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
            alignas(64) double tile[kTanhTile];
            for (std::size_t base = lo; base < hi; base += kTanhTile) {
                const std::size_t m = std::min(kTanhTile, hi - base);
                for (std::size_t i = 0; i < m; ++i) {
                    const double x = p[base + i];
                    tile[i] = kC1 * (x + kC2 * x * x * x);
                }
                cpu::vtanh_unfixed_f64(tile, tile, m);
                for (std::size_t i = 0; i < m; ++i)
                    q[base + i] = 0.5 * p[base + i] * (1.0 + tile[i]);
            }
        });
    } else {
        ErrorBuilder("cpu_backend::gelu").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage
CpuBackend::gelu_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            gelu_backward(detail::as_f32(a), detail::as_f32(grad), shape, Dtype::F32), dt);
    constexpr double kC1 = 0.7978845608028654;
    constexpr double kC2 = 0.044715;
    const auto& cs = std::get<CpuStorage>(a);
    const auto& gs = std::get<CpuStorage>(grad);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const float* x = reinterpret_cast<const float*>(cs.ptr.get());
        const float* g = reinterpret_cast<const float*>(gs.ptr.get());
        float* q = reinterpret_cast<float*>(ptr.get());
        const float c1 = static_cast<float>(kC1);
        const float c2 = static_cast<float>(kC2);
        // As the forward: vForce tanh a tile at a time, on every core.
        cpu::parallel_for(n, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
            alignas(64) float tile[kTanhTile];
            for (std::size_t base = lo; base < hi; base += kTanhTile) {
                const std::size_t m = std::min(kTanhTile, hi - base);
                for (std::size_t i = 0; i < m; ++i) {
                    const float xi = x[base + i];
                    tile[i] = c1 * (xi + c2 * xi * xi * xi);
                }
                cpu::vtanh_unfixed_f32(tile, tile, m);
                for (std::size_t i = 0; i < m; ++i) {
                    const float xi = x[base + i];
                    const float t = tile[i];
                    const float dinner = c1 * (1.f + 3.f * c2 * xi * xi);
                    const float dx = 0.5f * (1.f + t) + 0.5f * xi * (1.f - t * t) * dinner;
                    q[base + i] = dx * g[base + i];
                }
            }
        });
    } else if (dt == Dtype::F64) {
        const double* x = reinterpret_cast<const double*>(cs.ptr.get());
        const double* g = reinterpret_cast<const double*>(gs.ptr.get());
        double* q = reinterpret_cast<double*>(ptr.get());
        cpu::parallel_for(n, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
            alignas(64) double tile[kTanhTile];
            for (std::size_t base = lo; base < hi; base += kTanhTile) {
                const std::size_t m = std::min(kTanhTile, hi - base);
                for (std::size_t i = 0; i < m; ++i) {
                    const double xi = x[base + i];
                    tile[i] = kC1 * (xi + kC2 * xi * xi * xi);
                }
                cpu::vtanh_unfixed_f64(tile, tile, m);
                for (std::size_t i = 0; i < m; ++i) {
                    const double xi = x[base + i];
                    const double t = tile[i];
                    const double dinner = kC1 * (1.0 + 3.0 * kC2 * xi * xi);
                    const double dx = 0.5 * (1.0 + t) + 0.5 * xi * (1.0 - t * t) * dinner;
                    q[base + i] = dx * g[base + i];
                }
            }
        });
    } else {
        ErrorBuilder("cpu_backend::gelu_backward").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage
CpuBackend::silu_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            silu_backward(detail::as_f32(a), detail::as_f32(grad), shape, Dtype::F32), dt);
    const auto& cs = std::get<CpuStorage>(a);
    const auto& gs = std::get<CpuStorage>(grad);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const float* x = reinterpret_cast<const float*>(cs.ptr.get());
        const float* g = reinterpret_cast<const float*>(gs.ptr.get());
        float* q = reinterpret_cast<float*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const float xi = x[i];
            const float sx = 1.f / (1.f + std::exp(-xi));
            const float dx = sx * (1.f + xi * (1.f - sx));
            q[i] = dx * g[i];
        }
    } else if (dt == Dtype::F64) {
        const double* x = reinterpret_cast<const double*>(cs.ptr.get());
        const double* g = reinterpret_cast<const double*>(gs.ptr.get());
        double* q = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const double xi = x[i];
            const double sx = 1.0 / (1.0 + std::exp(-xi));
            const double dx = sx * (1.0 + xi * (1.0 - sx));
            q[i] = dx * g[i];
        }
    } else {
        ErrorBuilder("cpu_backend::silu_backward").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::gelu_exact(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(gelu_exact(detail::as_f32(a), shape, Dtype::F32), dt);
    // 0.5 * x * (1 + erf(x / sqrt(2))) — exact Gaussian-CDF GELU.
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    constexpr double kInvSqrt2 = 0.7071067811865476;
    if (dt == Dtype::F32) {
        const float* p = reinterpret_cast<const float*>(cs.ptr.get());
        float* q = reinterpret_cast<float*>(ptr.get());
        const float k = static_cast<float>(kInvSqrt2);
        // Not an approximation: erf_f32 is within 1 ulp of the correctly
        // rounded erf (see ErfPoly.h) — the GPT training parity holds to
        // 1.3e-7 over 3,000 steps on an exact erf, and an Abramowitz-
        // Stegun-class one (1.5e-7) would not do.  Unlike libm's erff
        // it is branch-free, so this loop vectorises.
        cpu::parallel_for(n, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
            for (std::size_t i = lo; i < hi; ++i) {
                const float x = p[i];
                q[i] = 0.5f * x * (1.f + cpu::erf_f32(x * k));
            }
        });
    } else if (dt == Dtype::F64) {
        const double* p = reinterpret_cast<const double*>(cs.ptr.get());
        double* q = reinterpret_cast<double*>(ptr.get());
        cpu::parallel_for(n, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
            for (std::size_t i = lo; i < hi; ++i) {
                const double x = p[i];
                q[i] = 0.5 * x * (1.0 + std::erf(x * kInvSqrt2));
            }
        });
    } else {
        ErrorBuilder("cpu_backend::gelu_exact").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::gelu_exact_backward(const Storage& a,
                                        const Storage& grad,
                                        const Shape& shape,
                                        Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            gelu_exact_backward(detail::as_f32(a), detail::as_f32(grad), shape, Dtype::F32), dt);
    // dy/dx = 0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2π)
    const auto& cs = std::get<CpuStorage>(a);
    const auto& gs = std::get<CpuStorage>(grad);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    constexpr double kInvSqrt2 = 0.7071067811865476;
    constexpr double kInvSqrt2Pi = 0.3989422804014327;  // 1 / sqrt(2π)
    if (dt == Dtype::F32) {
        const float* x = reinterpret_cast<const float*>(cs.ptr.get());
        const float* g = reinterpret_cast<const float*>(gs.ptr.get());
        float* q = reinterpret_cast<float*>(ptr.get());
        // Two passes over an L1 tile, neither of which calls out: the
        // CDF (erf_f32), then the density (normal_pdf_f32), the sum and
        // the incoming gradient.  See GeluGrad.h for why two.
        cpu::parallel_for(n, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
            float cdf[kTanhTile];
            for (std::size_t base = lo; base < hi; base += kTanhTile) {
                const std::size_t m = std::min(kTanhTile, hi - base);
                cpu::gelu_exact_cdf_f32(x + base, cdf, m);
                cpu::gelu_exact_grad_f32(x + base, cdf, g + base, q + base, m);
            }
        });
    } else if (dt == Dtype::F64) {
        const double* x = reinterpret_cast<const double*>(cs.ptr.get());
        const double* g = reinterpret_cast<const double*>(gs.ptr.get());
        double* q = reinterpret_cast<double*>(ptr.get());
        cpu::parallel_for(n, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
            for (std::size_t i = lo; i < hi; ++i) {
                const double xi = x[i];
                const double cdf = 0.5 * (1.0 + std::erf(xi * kInvSqrt2));
                const double pdf = kInvSqrt2Pi * std::exp(-0.5 * xi * xi);
                q[i] = (cdf + xi * pdf) * g[i];
            }
        });
    } else {
        ErrorBuilder("cpu_backend::gelu_exact_backward").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::leaky_relu(const Storage& a, const Shape& shape, Dtype dt, double slope) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(leaky_relu(detail::as_f32(a), shape, Dtype::F32, slope), dt);
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const float* p = reinterpret_cast<const float*>(cs.ptr.get());
        float* q = reinterpret_cast<float*>(ptr.get());
        const float fs = static_cast<float>(slope);
        for (std::size_t i = 0; i < n; ++i)
            q[i] = p[i] >= 0.f ? p[i] : fs * p[i];
    } else if (dt == Dtype::F64) {
        const double* p = reinterpret_cast<const double*>(cs.ptr.get());
        double* q = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = p[i] >= 0.0 ? p[i] : slope * p[i];
    } else {
        ErrorBuilder("cpu_backend::leaky_relu").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::softplus(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernel here; widen at the door and
    // round once on the way out.  See detail::widen_half.
    if (detail::is_half_like(dt)) {
        const std::size_t half_n = shape_numel(shape);
        const Storage wide = Storage{detail::widen_half(std::get<CpuStorage>(a), half_n)};
        return detail::narrow_half(softplus(wide, shape, Dtype::F32), half_n, dt);
    }
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const float* p = reinterpret_cast<const float*>(cs.ptr.get());
        float* q = reinterpret_cast<float*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const float x = p[i];
            q[i] = std::max(x, 0.f) + std::log1p(std::exp(-std::abs(x)));
        }
    } else if (dt == Dtype::F64) {
        const double* p = reinterpret_cast<const double*>(cs.ptr.get());
        double* q = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const double x = p[i];
            q[i] = std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
        }
    } else {
        ErrorBuilder("cpu_backend::softplus").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::elu(const Storage& a, const Shape& shape, Dtype dt, double alpha) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(elu(detail::as_f32(a), shape, Dtype::F32, alpha), dt);
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const float* p = reinterpret_cast<const float*>(cs.ptr.get());
        float* q = reinterpret_cast<float*>(ptr.get());
        const float fa = static_cast<float>(alpha);
        for (std::size_t i = 0; i < n; ++i) {
            const float x = p[i];
            q[i] = x >= 0.f ? x : fa * (std::exp(x) - 1.f);
        }
    } else if (dt == Dtype::F64) {
        const double* p = reinterpret_cast<const double*>(cs.ptr.get());
        double* q = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const double x = p[i];
            q[i] = x >= 0.0 ? x : alpha * (std::exp(x) - 1.0);
        }
    } else {
        ErrorBuilder("cpu_backend::elu").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::elu_backward(
    const Storage& a, const Storage& grad, const Shape& shape, Dtype dt, double alpha) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            elu_backward(detail::as_f32(a), detail::as_f32(grad), shape, Dtype::F32, alpha), dt);
    const auto& cs = std::get<CpuStorage>(a);
    const auto& gs = std::get<CpuStorage>(grad);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const float* xp = reinterpret_cast<const float*>(cs.ptr.get());
        const float* gp = reinterpret_cast<const float*>(gs.ptr.get());
        float* qp = reinterpret_cast<float*>(ptr.get());
        const float fa = static_cast<float>(alpha);
        for (std::size_t i = 0; i < n; ++i) {
            const float x = xp[i];
            qp[i] = (x > 0.f ? 1.f : fa * std::exp(x)) * gp[i];
        }
    } else if (dt == Dtype::F64) {
        const double* xp = reinterpret_cast<const double*>(cs.ptr.get());
        const double* gp = reinterpret_cast<const double*>(gs.ptr.get());
        double* qp = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const double x = xp[i];
            qp[i] = (x > 0.0 ? 1.0 : alpha * std::exp(x)) * gp[i];
        }
    } else {
        ErrorBuilder("cpu_backend::elu_backward").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::selu(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernel here; widen at the door and round
    // once on the way out.  See detail::widen_half.
    if (detail::is_half_like(dt)) {
        const std::size_t half_n = shape_numel(shape);
        return detail::narrow_half(
            selu(Storage{detail::widen_half(std::get<CpuStorage>(a), half_n)}, shape, Dtype::F32),
            half_n, dt);
    }
    constexpr double kScale = 1.0507009873554805;
    constexpr double kAlpha = 1.6732632423543772;
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const float* p = reinterpret_cast<const float*>(cs.ptr.get());
        float* q = reinterpret_cast<float*>(ptr.get());
        const float s = static_cast<float>(kScale);
        const float al = static_cast<float>(kAlpha);
        for (std::size_t i = 0; i < n; ++i)
            q[i] = s * (p[i] >= 0.f ? p[i] : al * (std::exp(p[i]) - 1.f));
    } else if (dt == Dtype::F64) {
        const double* p = reinterpret_cast<const double*>(cs.ptr.get());
        double* q = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = kScale * (p[i] >= 0.0 ? p[i] : kAlpha * (std::exp(p[i]) - 1.0));
    } else {
        ErrorBuilder("cpu_backend::selu").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage
CpuBackend::selu_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            selu_backward(detail::as_f32(a), detail::as_f32(grad), shape, Dtype::F32), dt);
    constexpr double kScale = 1.0507009873554805;
    constexpr double kAlpha = 1.6732632423543772;
    const auto& cs = std::get<CpuStorage>(a);
    const auto& gs = std::get<CpuStorage>(grad);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const float* xp = reinterpret_cast<const float*>(cs.ptr.get());
        const float* gp = reinterpret_cast<const float*>(gs.ptr.get());
        float* qp = reinterpret_cast<float*>(ptr.get());
        const float s = static_cast<float>(kScale);
        const float sa = static_cast<float>(kScale * kAlpha);
        for (std::size_t i = 0; i < n; ++i)
            qp[i] = (xp[i] > 0.f ? s : sa * std::exp(xp[i])) * gp[i];
    } else if (dt == Dtype::F64) {
        const double* xp = reinterpret_cast<const double*>(cs.ptr.get());
        const double* gp = reinterpret_cast<const double*>(gs.ptr.get());
        double* qp = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const double dx = xp[i] > 0.0 ? kScale : kScale * kAlpha * std::exp(xp[i]);
            qp[i] = dx * gp[i];
        }
    } else {
        ErrorBuilder("cpu_backend::selu_backward").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::mish(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(mish(detail::as_f32(a), shape, Dtype::F32), dt);
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    auto run = [&]<typename T>() {
        const T* p = reinterpret_cast<const T*>(cs.ptr.get());
        T* q = reinterpret_cast<T*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = -std::abs(p[i]);
        cpu::vexp(q, q, n);
        for (std::size_t i = 0; i < n; ++i)
            q[i] = std::max(p[i], T(0)) + std::log1p(q[i]);
        cpu::vtanh(q, q, n);
        for (std::size_t i = 0; i < n; ++i)
            q[i] *= p[i];
    };
    if (dt == Dtype::F32) {
        run.template operator()<float>();
    } else if (dt == Dtype::F64) {
        run.template operator()<double>();
    } else {
        ErrorBuilder("cpu_backend::mish").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage
CpuBackend::mish_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            mish_backward(detail::as_f32(a), detail::as_f32(grad), shape, Dtype::F32), dt);
    const auto& cs = std::get<CpuStorage>(a);
    const auto& gs = std::get<CpuStorage>(grad);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    auto run = [&]<typename T>() {
        const T* xp = reinterpret_cast<const T*>(cs.ptr.get());
        const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
        T* qp = reinterpret_cast<T*>(ptr.get());
        cpu::parallel_for(n, kTranscendentalGrain, [&](std::size_t lo, std::size_t hi) {
            alignas(64) T tile[kTanhTile], sig_tile[kTanhTile];
            for (std::size_t base = lo; base < hi; base += kTanhTile) {
                const std::size_t m = std::min(kTanhTile, hi - base);
                for (std::size_t i = 0; i < m; ++i) {
                    const T x = xp[base + i];
                    tile[i] = -std::abs(x);
                    sig_tile[i] = -x;
                }
                cpu::vexp(tile, tile, m);
                cpu::vexp(sig_tile, sig_tile, m);
                for (std::size_t i = 0; i < m; ++i) {
                    const T x = xp[base + i];
                    tile[i] = std::max(x, T(0)) + std::log1p(tile[i]);
                }
                cpu::vtanh(tile, tile, m);
                for (std::size_t i = 0; i < m; ++i) {
                    const T x = xp[base + i];
                    const T t = tile[i];
                    const T sig = T(1) / (T(1) + sig_tile[i]);
                    qp[base + i] = (t + x * (T(1) - t * t) * sig) * gp[base + i];
                }
            }
        });
    };
    if (dt == Dtype::F32) {
        run.template operator()<float>();
    } else if (dt == Dtype::F64) {
        run.template operator()<double>();
    } else {
        ErrorBuilder("cpu_backend::mish_backward").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::hard_sigmoid(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(hard_sigmoid(detail::as_f32(a), shape, Dtype::F32), dt);
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const float* p = reinterpret_cast<const float*>(cs.ptr.get());
        float* q = reinterpret_cast<float*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = std::min(std::max((p[i] + 3.f) / 6.f, 0.f), 1.f);
    } else if (dt == Dtype::F64) {
        const double* p = reinterpret_cast<const double*>(cs.ptr.get());
        double* q = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = std::min(std::max((p[i] + 3.0) / 6.0, 0.0), 1.0);
    } else {
        ErrorBuilder("cpu_backend::hard_sigmoid").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::hard_sigmoid_backward(const Storage& a,
                                          const Storage& grad,
                                          const Shape& shape,
                                          Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            hard_sigmoid_backward(detail::as_f32(a), detail::as_f32(grad), shape, Dtype::F32), dt);
    const auto& cs = std::get<CpuStorage>(a);
    const auto& gs = std::get<CpuStorage>(grad);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const float* xp = reinterpret_cast<const float*>(cs.ptr.get());
        const float* gp = reinterpret_cast<const float*>(gs.ptr.get());
        float* qp = reinterpret_cast<float*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const float g = gp[i] / 6.f;  // loaded first: a select, not a branch
            qp[i] = (xp[i] > -3.f && xp[i] < 3.f) ? g : 0.f;
        }
    } else if (dt == Dtype::F64) {
        const double* xp = reinterpret_cast<const double*>(cs.ptr.get());
        const double* gp = reinterpret_cast<const double*>(gs.ptr.get());
        double* qp = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const double g = gp[i] / 6.0;
            qp[i] = (xp[i] > -3.0 && xp[i] < 3.0) ? g : 0.0;
        }
    } else {
        ErrorBuilder("cpu_backend::hard_sigmoid_backward").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::hard_swish(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(hard_swish(detail::as_f32(a), shape, Dtype::F32), dt);
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const float* p = reinterpret_cast<const float*>(cs.ptr.get());
        float* q = reinterpret_cast<float*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = p[i] * std::min(std::max((p[i] + 3.f) / 6.f, 0.f), 1.f);
    } else if (dt == Dtype::F64) {
        const double* p = reinterpret_cast<const double*>(cs.ptr.get());
        double* q = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = p[i] * std::min(std::max((p[i] + 3.0) / 6.0, 0.0), 1.0);
    } else {
        ErrorBuilder("cpu_backend::hard_swish").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::hard_swish_backward(const Storage& a,
                                        const Storage& grad,
                                        const Shape& shape,
                                        Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            hard_swish_backward(detail::as_f32(a), detail::as_f32(grad), shape, Dtype::F32), dt);
    const auto& cs = std::get<CpuStorage>(a);
    const auto& gs = std::get<CpuStorage>(grad);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const float* xp = reinterpret_cast<const float*>(cs.ptr.get());
        const float* gp = reinterpret_cast<const float*>(gs.ptr.get());
        float* qp = reinterpret_cast<float*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const float x = xp[i];
            float dx;
            if (x <= -3.f)
                dx = 0.f;
            else if (x >= 3.f)
                dx = 1.f;
            else
                dx = x / 3.f + 0.5f;
            qp[i] = dx * gp[i];
        }
    } else if (dt == Dtype::F64) {
        const double* xp = reinterpret_cast<const double*>(cs.ptr.get());
        const double* gp = reinterpret_cast<const double*>(gs.ptr.get());
        double* qp = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const double x = xp[i];
            double dx;
            if (x <= -3.0)
                dx = 0.0;
            else if (x >= 3.0)
                dx = 1.0;
            else
                dx = x / 3.0 + 0.5;
            qp[i] = dx * gp[i];
        }
    } else {
        ErrorBuilder("cpu_backend::hard_swish_backward").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::relu6(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(relu6(detail::as_f32(a), shape, Dtype::F32), dt);
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (dt == Dtype::F32) {
        const float* p = reinterpret_cast<const float*>(cs.ptr.get());
        float* q = reinterpret_cast<float*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = std::min(std::max(p[i], 0.f), 6.f);
    } else if (dt == Dtype::F64) {
        const double* p = reinterpret_cast<const double*>(cs.ptr.get());
        double* q = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            q[i] = std::min(std::max(p[i], 0.0), 6.0);
    } else if (detail::is_int_like(dt)) {
        // Clamping to [0, 6] is a pair of comparisons, so each integer
        // width does it in itself.  The reference framework answers
        // relu6 on integers and keeps the dtype.
        detail::for_each_int(dt, cs.ptr.get(), ptr.get(), n, [](auto v) {
            using T = decltype(v);
            return v < T{} ? T{} : (v > T{6} ? T{6} : v);
        });
    } else {
        ErrorBuilder("cpu_backend::relu6").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::softmax(const Storage& a, const Shape& shape, int axis, Dtype dt) {
    // Accelerate has no half kernel here; widen at the door and round
    // once on the way out.  See detail::widen_half.
    if (detail::is_half_like(dt)) {
        const std::size_t half_n = shape_numel(shape);
        return detail::narrow_half(
            softmax(Storage{detail::widen_half(std::get<CpuStorage>(a), half_n)}, shape, axis,
                    Dtype::F32),
            half_n, dt);
    }
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t nb = cs.nbytes;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const std::size_t axis_dim = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
    const auto [outer, inner] = cpu::around_axis(shape, axis);
    if (dt == Dtype::F32)
        cpu::softmax_cpu(reinterpret_cast<const float*>(cs.ptr.get()),
                         reinterpret_cast<float*>(ptr.get()), outer, axis_dim, inner);
    else if (dt == Dtype::F64)
        cpu::softmax_cpu(reinterpret_cast<const double*>(cs.ptr.get()),
                         reinterpret_cast<double*>(ptr.get()), outer, axis_dim, inner);
    else
        ErrorBuilder("cpu_backend::softmax").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::softmax_backward(
    const Storage& z, const Storage& grad_out, const Shape& shape, int axis, Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            softmax_backward(detail::as_f32(z), detail::as_f32(grad_out), shape, axis, Dtype::F32),
            dt);
    const auto& z_cpu = std::get<CpuStorage>(z);
    const auto& g_cpu = std::get<CpuStorage>(grad_out);
    std::size_t nb = z_cpu.nbytes;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const std::size_t axis_dim = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
    const auto [outer, inner] = cpu::around_axis(shape, axis);
    if (dt == Dtype::F32)
        cpu::softmax_backward_cpu(reinterpret_cast<const float*>(z_cpu.ptr.get()),
                                  reinterpret_cast<const float*>(g_cpu.ptr.get()),
                                  reinterpret_cast<float*>(ptr.get()), outer, axis_dim, inner);
    else if (dt == Dtype::F64)
        cpu::softmax_backward_cpu(reinterpret_cast<const double*>(z_cpu.ptr.get()),
                                  reinterpret_cast<const double*>(g_cpu.ptr.get()),
                                  reinterpret_cast<double*>(ptr.get()), outer, axis_dim, inner);
    else
        ErrorBuilder("cpu_backend::softmax_backward").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::log_softmax_backward(
    const Storage& y, const Storage& grad_out, const Shape& shape, int axis, Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(log_softmax_backward(detail::as_f32(y), detail::as_f32(grad_out),
                                                        shape, axis, Dtype::F32),
                                   dt);
    const auto& yc = std::get<CpuStorage>(y);
    const auto& gc = std::get<CpuStorage>(grad_out);
    std::size_t nb = gc.nbytes;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const std::size_t axis_dim = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
    const auto [outer, inner] = cpu::around_axis(shape, axis);
    if (dt == Dtype::F32)
        cpu::log_softmax_backward_cpu(reinterpret_cast<const float*>(yc.ptr.get()),
                                      reinterpret_cast<const float*>(gc.ptr.get()),
                                      reinterpret_cast<float*>(ptr.get()), outer, axis_dim, inner);
    else if (dt == Dtype::F64)
        cpu::log_softmax_backward_cpu(reinterpret_cast<const double*>(yc.ptr.get()),
                                      reinterpret_cast<const double*>(gc.ptr.get()),
                                      reinterpret_cast<double*>(ptr.get()), outer, axis_dim, inner);
    else
        ErrorBuilder("cpu_backend::log_softmax_backward").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::log_softmax(const Storage& a, const Shape& shape, int axis, Dtype dt) {
    if (detail::is_half_like(dt)) {
        const std::size_t half_n = shape_numel(shape);
        return detail::narrow_half(
            log_softmax(Storage{detail::widen_half(std::get<CpuStorage>(a), half_n)}, shape, axis,
                        Dtype::F32),
            half_n, dt);
    }
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t nb = cs.nbytes;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const std::size_t axis_dim = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
    const auto [outer, inner] = cpu::around_axis(shape, axis);
    if (dt == Dtype::F32)
        cpu::log_softmax_cpu(reinterpret_cast<const float*>(cs.ptr.get()),
                             reinterpret_cast<float*>(ptr.get()), outer, axis_dim, inner);
    else if (dt == Dtype::F64)
        cpu::log_softmax_cpu(reinterpret_cast<const double*>(cs.ptr.get()),
                             reinterpret_cast<double*>(ptr.get()), outer, axis_dim, inner);
    else
        ErrorBuilder("cpu_backend::log_softmax").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage
CpuBackend::relu_backward(const Storage& g, const Storage& x, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            relu_backward(detail::as_f32(g), detail::as_f32(x), shape, Dtype::F32), dt);
    const auto& cg = std::get<CpuStorage>(g);
    const auto& cx = std::get<CpuStorage>(x);
    std::size_t n = shape_numel(shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    // The gradient is loaded before the comparison picks it, so the
    // choice compiles to a select.  Written as ``x > 0 ? g[i] : 0`` the
    // load sat on one side of a branch, and on activations of random
    // sign — every real network's — half the branches mispredicted:
    // 2.25 ms on 524k elements against 0.26 when every x was positive.
    if (dt == Dtype::F32) {
        const float* __restrict gp = reinterpret_cast<const float*>(cg.ptr.get());
        const float* __restrict xp = reinterpret_cast<const float*>(cx.ptr.get());
        float* __restrict op = reinterpret_cast<float*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const float g = gp[i];
            op[i] = xp[i] <= 0.f ? 0.f : g;
        }
    } else if (dt == Dtype::F64) {
        const double* __restrict gp = reinterpret_cast<const double*>(cg.ptr.get());
        const double* __restrict xp = reinterpret_cast<const double*>(cx.ptr.get());
        double* __restrict op = reinterpret_cast<double*>(ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            const double g = gp[i];
            op[i] = xp[i] <= 0.0 ? 0.0 : g;
        }
    } else {
        ErrorBuilder("cpu_backend::relu_backward").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

}  // namespace backend
}  // namespace lucid
