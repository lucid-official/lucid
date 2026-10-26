// lucid/_C/backend/gpu/methods/Activation.cpp
// Out-of-line GpuBackend method definitions for the Activation family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

Storage GpuBackend::sigmoid(const Storage& a, const Shape& shape, Dtype dt) {
    if (is_complex(dt))
        ErrorBuilder("gpu_backend::sigmoid").not_implemented("dtype not supported");
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::sigmoid(x); });
}

Storage GpuBackend::relu(const Storage& a, const Shape& shape, Dtype dt) {
    // Use a scalar zero instead of zeros_like to avoid allocating a
    // full-size zero tensor on the GPU.  zeros_like(x) pays an extra
    // 40 MB allocation + fill kernel for a 10M-element float32 input;
    // a scalar broadcasts inside the MLX kernel at zero allocation cost.
    return mlx_unary(a, shape, dt, [dt](auto& x) {
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
        return ::mlx::core::maximum(x, zero);
    });
}

Storage GpuBackend::silu(const Storage& a, const Shape&, Dtype dt) {
    if (is_complex(dt))
        ErrorBuilder("gpu_backend::silu").not_implemented("dtype not supported");
    return mlx_unary_fused(a, dt, [](const ::mlx::core::array& x) {
        return ::mlx::core::multiply(x, ::mlx::core::sigmoid(x));
    });
}

Storage GpuBackend::silu_backward(const Storage& a, const Storage& grad, const Shape&, Dtype dt) {
    // dL/dx = σ(x) * (1 + x*(1 - σ(x))) * dL/dy — fused into one kernel.
    return mlx_binary_fused(a, grad, dt,
                            [](const ::mlx::core::array& x, const ::mlx::core::array& g) {
                                ::mlx::core::array one(1.0, x.dtype());
                                auto sx = ::mlx::core::sigmoid(x);
                                auto one_m_sx = ::mlx::core::subtract(one, sx);
                                auto x_omsx = ::mlx::core::multiply(x, one_m_sx);
                                auto one_p = ::mlx::core::add(one, x_omsx);
                                auto dx = ::mlx::core::multiply(sx, one_p);
                                return ::mlx::core::multiply(dx, g);
                            });
}

Storage GpuBackend::gelu(const Storage& a, const Shape&, Dtype dt) {
    return mlx_unary_fused(a, dt, [](const ::mlx::core::array& x) {
        auto d = x.dtype();
        ::mlx::core::array half(0.5, d);
        ::mlx::core::array one(1.0, d);
        ::mlx::core::array a044(0.044715, d);
        ::mlx::core::array kk(0.7978845608028654, d);  // sqrt(2/pi)
        auto x3 = ::mlx::core::multiply(::mlx::core::square(x), x);
        auto inner =
            ::mlx::core::multiply(kk, ::mlx::core::add(x, ::mlx::core::multiply(a044, x3)));
        auto t = ::mlx::core::tanh(inner);
        return ::mlx::core::multiply(::mlx::core::multiply(half, x), ::mlx::core::add(one, t));
    });
}

Storage GpuBackend::gelu_exact(const Storage& a, const Shape&, Dtype dt) {
    // y = 0.5 * x * (1 + erf(x / sqrt(2))) — fused into one kernel.
    return mlx_unary_fused(a, dt, [](const ::mlx::core::array& x) {
        auto d = x.dtype();
        ::mlx::core::array half(0.5, d);
        ::mlx::core::array one(1.0, d);
        ::mlx::core::array inv_sqrt2(0.7071067811865476, d);
        auto z = ::mlx::core::multiply(x, inv_sqrt2);
        auto cdf = ::mlx::core::multiply(half, ::mlx::core::add(one, ::mlx::core::erf(z)));
        return ::mlx::core::multiply(x, cdf);
    });
}

Storage GpuBackend::gelu_exact_backward(const Storage& a,
                                        const Storage& grad,
                                        const Shape& shape,
                                        Dtype dt) {
    // dy/dx = 0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2π) — fused.
    return mlx_binary_fused(
        a, grad, dt, [](const ::mlx::core::array& x, const ::mlx::core::array& g) {
            auto d = x.dtype();
            ::mlx::core::array half(0.5, d);
            ::mlx::core::array one(1.0, d);
            ::mlx::core::array neg_half(-0.5, d);
            ::mlx::core::array inv_sqrt2(0.7071067811865476, d);
            ::mlx::core::array inv_sqrt2pi(0.3989422804014327, d);
            auto z = ::mlx::core::multiply(x, inv_sqrt2);
            auto cdf = ::mlx::core::multiply(half, ::mlx::core::add(one, ::mlx::core::erf(z)));
            auto x2 = ::mlx::core::multiply(x, x);
            auto pdf = ::mlx::core::multiply(inv_sqrt2pi,
                                             ::mlx::core::exp(::mlx::core::multiply(neg_half, x2)));
            auto dx = ::mlx::core::add(cdf, ::mlx::core::multiply(x, pdf));
            return ::mlx::core::multiply(dx, g);
        });
}

Storage GpuBackend::gelu_backward(const Storage& a, const Storage& grad, const Shape&, Dtype dt) {
    return mlx_binary_fused(
        a, grad, dt, [](const ::mlx::core::array& x, const ::mlx::core::array& g) {
            auto d = x.dtype();
            ::mlx::core::array c1(0.7978845608028654, d);  // sqrt(2/pi)
            ::mlx::core::array c2(0.044715, d);
            ::mlx::core::array three(3.0, d);
            ::mlx::core::array half(0.5, d);
            ::mlx::core::array one(1.0, d);
            auto x2 = ::mlx::core::multiply(x, x);
            auto x3 = ::mlx::core::multiply(x2, x);
            auto inner =
                ::mlx::core::multiply(c1, ::mlx::core::add(x, ::mlx::core::multiply(c2, x3)));
            auto t = ::mlx::core::tanh(inner);
            auto dinner = ::mlx::core::multiply(
                c1,
                ::mlx::core::add(one, ::mlx::core::multiply(three, ::mlx::core::multiply(c2, x2))));
            auto t2 = ::mlx::core::multiply(t, t);
            auto term1 = ::mlx::core::multiply(half, ::mlx::core::add(one, t));
            auto term2 = ::mlx::core::multiply(
                half, ::mlx::core::multiply(
                          x, ::mlx::core::multiply(::mlx::core::subtract(one, t2), dinner)));
            auto dx = ::mlx::core::add(term1, term2);
            return ::mlx::core::multiply(dx, g);
        });
}

Storage GpuBackend::leaky_relu(const Storage& a, const Shape& shape, Dtype dt, double slope) {
    return mlx_unary(a, shape, dt, [dt, slope](auto& x) {
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array slope_arr(slope, gpu::to_mlx_dtype(dt));
        auto pos_mask = ::mlx::core::greater_equal(x, zero);
        auto neg_branch = ::mlx::core::multiply(slope_arr, x);
        return ::mlx::core::where(pos_mask, x, neg_branch);
    });
}

Storage GpuBackend::softplus(const Storage& a, const Shape&, Dtype dt) {
    return mlx_unary_fused(a, dt, [](const ::mlx::core::array& x) {
        ::mlx::core::array zero(0.0, x.dtype());
        auto pos = ::mlx::core::maximum(x, zero);
        auto neg_abs = ::mlx::core::negative(::mlx::core::abs(x));
        auto log1p = ::mlx::core::log1p(::mlx::core::exp(neg_abs));
        return ::mlx::core::add(pos, log1p);
    });
}

Storage GpuBackend::elu(const Storage& a, const Shape& shape, Dtype dt, double alpha) {
    return mlx_unary(a, shape, dt, [dt, alpha](auto& x) {
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array alpha_arr(alpha, gpu::to_mlx_dtype(dt));
        auto pos_mask = ::mlx::core::greater_equal(x, zero);
        auto neg =
            ::mlx::core::multiply(alpha_arr, ::mlx::core::subtract(::mlx::core::exp(x), one));
        return ::mlx::core::where(pos_mask, x, neg);
    });
}

Storage GpuBackend::elu_backward(
    const Storage& a, const Storage& grad, const Shape& shape, Dtype dt, double alpha) {
    return mlx_binary(a, grad, shape, dt, [dt, alpha](auto& x, auto& g) {
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));  // scalar — no full-tensor alloc
        ::mlx::core::array alpha_arr(alpha, gpu::to_mlx_dtype(dt));
        auto pos_mask = ::mlx::core::greater(x, zero);
        auto neg_branch = ::mlx::core::multiply(alpha_arr, ::mlx::core::exp(x));
        auto deriv = ::mlx::core::where(pos_mask, one, neg_branch);
        return ::mlx::core::multiply(deriv, g);
    });
}

Storage GpuBackend::selu(const Storage& a, const Shape&, Dtype dt) {
    return mlx_unary_fused(a, dt, [](const ::mlx::core::array& x) {
        constexpr double kS = 1.0507009873554805;
        constexpr double kA = 1.6732632423543772;
        auto d = x.dtype();
        ::mlx::core::array zero(0.0, d);
        ::mlx::core::array one(1.0, d);
        ::mlx::core::array s(kS, d);
        ::mlx::core::array sa(kS * kA, d);
        auto pos_branch = ::mlx::core::multiply(s, x);
        auto neg_branch =
            ::mlx::core::multiply(sa, ::mlx::core::subtract(::mlx::core::exp(x), one));
        auto pos_mask = ::mlx::core::greater_equal(x, zero);
        return ::mlx::core::where(pos_mask, pos_branch, neg_branch);
    });
}

Storage GpuBackend::selu_backward(const Storage& a, const Storage& grad, const Shape&, Dtype dt) {
    return mlx_binary_fused(
        a, grad, dt, [](const ::mlx::core::array& x, const ::mlx::core::array& g) {
            constexpr double kS = 1.0507009873554805;
            constexpr double kA = 1.6732632423543772;
            auto d = x.dtype();
            ::mlx::core::array zero(0.0, d);
            ::mlx::core::array s_arr(kS, d);
            ::mlx::core::array sa_arr(kS * kA, d);
            auto pos_mask = ::mlx::core::greater(x, zero);
            auto pos_branch = ::mlx::core::broadcast_to(s_arr, x.shape());
            auto neg_branch = ::mlx::core::multiply(sa_arr, ::mlx::core::exp(x));
            auto deriv = ::mlx::core::where(pos_mask, pos_branch, neg_branch);
            return ::mlx::core::multiply(deriv, g);
        });
}

Storage GpuBackend::mish(const Storage& a, const Shape&, Dtype dt) {
    return mlx_unary_fused(a, dt, [](const ::mlx::core::array& x) {
        ::mlx::core::array zero(0.0, x.dtype());
        auto pos = ::mlx::core::maximum(x, zero);
        auto neg_abs = ::mlx::core::negative(::mlx::core::abs(x));
        auto sp = ::mlx::core::add(pos, ::mlx::core::log1p(::mlx::core::exp(neg_abs)));
        return ::mlx::core::multiply(x, ::mlx::core::tanh(sp));
    });
}

Storage GpuBackend::mish_backward(const Storage& a, const Storage& grad, const Shape&, Dtype dt) {
    return mlx_binary_fused(
        a, grad, dt, [](const ::mlx::core::array& x, const ::mlx::core::array& g) {
            ::mlx::core::array zero(0.0, x.dtype());
            ::mlx::core::array one(1.0, x.dtype());
            auto pos = ::mlx::core::maximum(x, zero);
            auto neg_abs = ::mlx::core::negative(::mlx::core::abs(x));
            auto sp = ::mlx::core::add(pos, ::mlx::core::log1p(::mlx::core::exp(neg_abs)));
            auto t = ::mlx::core::tanh(sp);
            auto sig = ::mlx::core::sigmoid(x);
            auto one_minus_t2 = ::mlx::core::subtract(one, ::mlx::core::square(t));
            auto deriv = ::mlx::core::add(
                t, ::mlx::core::multiply(x, ::mlx::core::multiply(one_minus_t2, sig)));
            return ::mlx::core::multiply(deriv, g);
        });
}

Storage GpuBackend::hard_sigmoid(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [dt](auto& x) {
        ::mlx::core::array three(3.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array six(6.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
        auto v = ::mlx::core::divide(::mlx::core::add(x, three), six);
        return ::mlx::core::minimum(::mlx::core::maximum(v, zero), one);
    });
}

Storage GpuBackend::hard_sigmoid_backward(const Storage& a,
                                          const Storage& grad,
                                          const Shape& shape,
                                          Dtype dt) {
    return mlx_binary(a, grad, shape, dt, [dt](auto& x, auto& g) {
        ::mlx::core::array m3(-3.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array p3(3.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array s(1.0 / 6.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
        auto in_range =
            ::mlx::core::logical_and(::mlx::core::greater(x, m3), ::mlx::core::less(x, p3));
        auto s_b = ::mlx::core::broadcast_to(s, x.shape());
        auto z_b = ::mlx::core::broadcast_to(zero, x.shape());
        auto deriv = ::mlx::core::where(in_range, s_b, z_b);
        return ::mlx::core::multiply(deriv, g);
    });
}

Storage GpuBackend::hard_swish(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [dt](auto& x) {
        ::mlx::core::array three(3.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array six(6.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
        auto v = ::mlx::core::divide(::mlx::core::add(x, three), six);
        auto h = ::mlx::core::minimum(::mlx::core::maximum(v, zero), one);
        return ::mlx::core::multiply(x, h);
    });
}

Storage GpuBackend::hard_swish_backward(const Storage& a,
                                        const Storage& grad,
                                        const Shape& shape,
                                        Dtype dt) {
    return mlx_binary(a, grad, shape, dt, [dt](auto& x, auto& g) {
        ::mlx::core::array m3(-3.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array p3(3.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array half(0.5, gpu::to_mlx_dtype(dt));
        ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array third(1.0 / 3.0, gpu::to_mlx_dtype(dt));
        auto mid_branch = ::mlx::core::add(::mlx::core::multiply(x, third), half);
        auto le_m3 = ::mlx::core::less_equal(x, m3);
        auto ge_p3 = ::mlx::core::greater_equal(x, p3);
        auto z_b = ::mlx::core::broadcast_to(zero, x.shape());
        auto o_b = ::mlx::core::broadcast_to(one, x.shape());
        auto step1 = ::mlx::core::where(ge_p3, o_b, mid_branch);
        auto deriv = ::mlx::core::where(le_m3, z_b, step1);
        return ::mlx::core::multiply(deriv, g);
    });
}

Storage GpuBackend::relu6(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [dt](auto& x) {
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array six(6.0, gpu::to_mlx_dtype(dt));
        return ::mlx::core::minimum(::mlx::core::maximum(x, zero), six);
    });
}

Storage GpuBackend::softmax(const Storage& a, const Shape&, int axis, Dtype dt) {
    const auto& gs = std::get<GpuStorage>(a);
    return Storage{gpu::wrap_mlx_array(::mlx::core::softmax(*gs.arr, axis, true), dt)};
}

Storage GpuBackend::softmax_backward(
    const Storage& z, const Storage& grad_out, const Shape& shape_param, int axis, Dtype dt) {
    const auto& z_gpu = std::get<GpuStorage>(z);
    const auto& g_gpu = std::get<GpuStorage>(grad_out);
    (void)shape_param;
    auto gz = ::mlx::core::multiply(*g_gpu.arr, *z_gpu.arr);
    auto sum_gz = ::mlx::core::sum(gz, std::vector<int>{axis}, true);
    auto diff = ::mlx::core::subtract(*g_gpu.arr, sum_gz);
    auto result = ::mlx::core::multiply(*z_gpu.arr, diff);
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

Storage GpuBackend::log_softmax_backward(
    const Storage& y, const Storage& grad_out, const Shape&, int axis, Dtype dt) {
    const auto& yg = std::get<GpuStorage>(y);
    const auto& gg = std::get<GpuStorage>(grad_out);
    auto p = ::mlx::core::exp(*yg.arr);
    // ``sum(g)`` is not bounded the way softmax's ``sum(z * g)`` is, so
    // a half gradient is summed in float32 (see ``widen_half``).
    auto sum_g = sum_widened(*gg.arr, std::vector<int>{axis}, true);
    auto result = ::mlx::core::subtract(*gg.arr, ::mlx::core::multiply(p, sum_g));
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

Storage GpuBackend::log_softmax(const Storage& a, const Shape&, int axis, Dtype dt) {
    const auto& gs = std::get<GpuStorage>(a);
    return Storage{gpu::wrap_mlx_array(gpu::log_softmax_along(*gs.arr, axis), dt)};
}

Storage GpuBackend::relu_backward(const Storage& g, const Storage& x, const Shape&, Dtype dt) {
    const auto& gg = std::get<GpuStorage>(g);
    const auto& gx = std::get<GpuStorage>(x);
    if (!gg.arr || !gx.arr)
        ErrorBuilder("gpu_backend::relu_backward").fail("null GPU array");
    ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
    auto out = ::mlx::core::where(::mlx::core::less_equal(*gx.arr, zero), zero, *gg.arr);
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

}  // namespace backend
}  // namespace lucid
