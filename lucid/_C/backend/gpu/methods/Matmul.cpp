// lucid/_C/backend/gpu/methods/Matmul.cpp
// Out-of-line GpuBackend method definitions for the Matmul family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

Storage GpuBackend::matmul(const Storage& a, const Storage& b, const MatmulOpts& opts, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    const auto& gb = std::get<GpuStorage>(b);
    ::mlx::core::array a_arr = *ga.arr;
    ::mlx::core::array b_arr = *gb.arr;
    if (opts.transA)
        a_arr = ::mlx::core::swapaxes(a_arr, -2, -1);
    if (opts.transB)
        b_arr = ::mlx::core::swapaxes(b_arr, -2, -1);
    // MLX refuses integer operands outright — "[matmul] Only inexact
    // types are supported" — while the product of two integer matrices
    // is an integer matrix, which the CPU and the reference framework
    // both answer.  Widen to float32, multiply, round back: exact while
    // each dot product stays inside 2^24, which is the same bound the
    // CPU's own float path carries.
    const bool discrete = dt != Dtype::F16 && dt != Dtype::BF16 && dt != Dtype::F32 &&
                          dt != Dtype::F64 && dt != Dtype::C64;
    if (discrete) {
        a_arr = ::mlx::core::astype(a_arr, ::mlx::core::float32);
        b_arr = ::mlx::core::astype(b_arr, ::mlx::core::float32);
        auto wide = ::mlx::core::matmul(a_arr, b_arr);
        auto rounded = ::mlx::core::round(wide);
        return Storage{
            gpu::wrap_mlx_array(::mlx::core::astype(rounded, gpu::to_mlx_dtype(dt)), dt)};
    }
    auto result = ::mlx::core::matmul(a_arr, b_arr);
    // PERF: matmul produces a fresh contiguous buffer; the defensive
    // wrap-in-contiguous was forcing a redundant memcpy and breaking
    // MLX's fusion chain.  Same anti-pattern that 3.0.3 fixed for
    // Conv2d output (cf perf-conv2d-deferred-contig).
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

Storage GpuBackend::linear(const Storage& x,
                           const Storage& weight,
                           const Storage& bias,
                           const Shape&,
                           const Shape&,
                           const Shape& out_shape,
                           Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gw = std::get<GpuStorage>(weight);
    const auto& gb = std::get<GpuStorage>(bias);
    if (shape_numel(out_shape) == 0) {
        auto out = ::mlx::core::zeros(gpu::to_mlx_shape(out_shape), gpu::to_mlx_dtype(dt));
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }
    auto out =
        ::mlx::core::add(::mlx::core::matmul(*gx.arr, ::mlx::core::transpose(*gw.arr)), *gb.arr);
    // PERF: add(matmul(...), bias) produces a fresh contiguous buffer.
    // Dropping the defensive contiguous() unlocks fusion with downstream
    // relu / next linear / etc.  Same fix as 3.0.3 Conv2d.
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

std::vector<Storage> GpuBackend::linear_backward(const Storage& grad,
                                                 const Storage& x,
                                                 const Storage& weight,
                                                 const Shape& x_shape,
                                                 const Shape& weight_shape,
                                                 const Shape& bias_shape,
                                                 Dtype dt) {
    const auto& gg = std::get<GpuStorage>(grad);
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gw = std::get<GpuStorage>(weight);
    const std::size_t M = linear_batch(x_shape);
    const std::size_t K = static_cast<std::size_t>(x_shape.back());
    const std::size_t N = static_cast<std::size_t>(weight_shape[0]);
    if (M == 0 || N == 0 || K == 0) {
        auto dx = ::mlx::core::zeros(gpu::to_mlx_shape(x_shape), gpu::to_mlx_dtype(dt));
        auto dW = ::mlx::core::zeros(gpu::to_mlx_shape(weight_shape), gpu::to_mlx_dtype(dt));
        auto db = ::mlx::core::zeros(gpu::to_mlx_shape(bias_shape), gpu::to_mlx_dtype(dt));
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dt)},
                Storage{gpu::wrap_mlx_array(std::move(dW), dt)},
                Storage{gpu::wrap_mlx_array(std::move(db), dt)}};
    }
    ::mlx::core::Shape flat_x{static_cast<int>(M), static_cast<int>(K)};
    ::mlx::core::Shape flat_g{static_cast<int>(M), static_cast<int>(N)};
    auto g_2d = ::mlx::core::reshape(*gg.arr, flat_g);
    auto x_2d = ::mlx::core::reshape(*gx.arr, flat_x);
    auto dx_flat = ::mlx::core::matmul(g_2d, *gw.arr);
    auto dx = ::mlx::core::reshape(dx_flat, gpu::to_mlx_shape(x_shape));
    auto dW = ::mlx::core::matmul(::mlx::core::transpose(g_2d), x_2d);
    auto db = sum_widened(g_2d, std::vector<int>{0}, false);
    // PERF: matmul / sum produce fresh contiguous buffers.  reshape on
    // a contiguous matmul result is also contiguous.  Drop redundant
    // contiguous() wraps so dx/dW/db can fuse into the autograd graph.
    return {Storage{gpu::wrap_mlx_array(std::move(dx), dt)},
            Storage{gpu::wrap_mlx_array(std::move(dW), dt)},
            Storage{gpu::wrap_mlx_array(std::move(db), dt)}};
}

std::size_t GpuBackend::linear_batch(const Shape& x_shape) {
    std::size_t M = 1;
    for (std::size_t d = 0; d + 1 < x_shape.size(); ++d)
        M *= static_cast<std::size_t>(x_shape[d]);
    return M;
}

Storage GpuBackend::bilinear_layer_forward(const Storage& x1,
                                           const Storage& x2,
                                           const Storage& weight,
                                           const Storage& bias,
                                           bool has_bias,
                                           const Shape& x1_shape,
                                           const Shape& x2_shape,
                                           const Shape& w_shape,
                                           Dtype dt) {
    std::size_t B = 1;
    for (std::size_t i = 0; i + 1 < x1_shape.size(); ++i)
        B *= static_cast<std::size_t>(x1_shape[i]);
    const int iB = static_cast<int>(B);
    const int iD1 = static_cast<int>(x1_shape.back());
    const int iD2 = static_cast<int>(x2_shape.back());
    const int iDout = static_cast<int>(w_shape[0]);

    const auto& gx1 = std::get<GpuStorage>(x1);
    const auto& gx2 = std::get<GpuStorage>(x2);
    const auto& gw = std::get<GpuStorage>(weight);

    auto x1_b = ::mlx::core::reshape(*gx1.arr, {iB, iD1});
    auto x2_b = ::mlx::core::reshape(*gx2.arr, {iB, iD2});

    auto W_rs = ::mlx::core::transpose(*gw.arr, {1, 0, 2});
    auto W_rs2 = ::mlx::core::reshape(W_rs, {iD1, iDout * iD2});
    auto tmp = ::mlx::core::matmul(x1_b, W_rs2);
    auto tmp3d = ::mlx::core::reshape(tmp, {iB, iDout, iD2});
    auto x2_e = ::mlx::core::reshape(x2_b, {iB, iD2, 1});
    auto y_e = ::mlx::core::matmul(tmp3d, x2_e);
    auto y_2d = ::mlx::core::reshape(y_e, {iB, iDout});
    if (has_bias) {
        const auto& gb = std::get<GpuStorage>(bias);
        y_2d = ::mlx::core::add(y_2d, *gb.arr);
    }

    ::mlx::core::Shape out_mlx_shape = gpu::to_mlx_shape(x1_shape);
    out_mlx_shape.back() = iDout;
    auto y_out = ::mlx::core::reshape(y_2d, out_mlx_shape);
    return Storage{gpu::wrap_mlx_array(std::move(y_out), dt)};
}

std::vector<Storage> GpuBackend::bilinear_layer_backward(const Storage& grad_out,
                                                         const Storage& x1,
                                                         const Storage& x2,
                                                         const Storage& weight,
                                                         const Shape& x1_shape,
                                                         const Shape& x2_shape,
                                                         const Shape& w_shape,
                                                         bool has_bias,
                                                         Dtype dt) {
    const std::size_t B = 1;
    std::size_t b = 1;
    for (std::size_t i = 0; i + 1 < x1_shape.size(); ++i)
        b *= static_cast<std::size_t>(x1_shape[i]);
    const std::size_t D1 = static_cast<std::size_t>(x1_shape.back());
    const std::size_t D2 = static_cast<std::size_t>(x2_shape.back());
    const std::size_t Dout = static_cast<std::size_t>(w_shape[0]);
    (void)B;
    const int iB = static_cast<int>(b);
    const int iD1 = static_cast<int>(D1), iD2 = static_cast<int>(D2),
              iDout = static_cast<int>(Dout);
    const auto& gx1 = std::get<GpuStorage>(x1);
    const auto& gx2 = std::get<GpuStorage>(x2);
    const auto& gw = std::get<GpuStorage>(weight);
    const auto& gg = std::get<GpuStorage>(grad_out);
    auto x1_f = ::mlx::core::reshape(*gx1.arr, {iB, iD1});
    auto x2_f = ::mlx::core::reshape(*gx2.arr, {iB, iD2});
    auto W = ::mlx::core::reshape(*gw.arr, {iDout, iD1, iD2});
    auto G = ::mlx::core::reshape(*gg.arr, {iB, iDout});
    auto W_perm = ::mlx::core::transpose(W, {1, 0, 2});
    auto W_flat = ::mlx::core::reshape(W_perm, {iD1, iDout * iD2});
    auto Q_flat = ::mlx::core::matmul(x1_f, W_flat);
    auto Q = ::mlx::core::reshape(Q_flat, {iB, iDout, iD2});
    auto W_ki = ::mlx::core::reshape(W, {iDout * iD1, iD2});
    auto x2_t = ::mlx::core::transpose(x2_f, {1, 0});
    auto P_pre = ::mlx::core::matmul(W_ki, x2_t);
    auto P_3 = ::mlx::core::reshape(P_pre, {iDout, iD1, iB});
    auto P = ::mlx::core::transpose(P_3, {2, 0, 1});
    auto G_b1 = ::mlx::core::reshape(G, {iB, 1, iDout});
    auto dx1_3 = ::mlx::core::matmul(G_b1, P);
    auto dx1r = ::mlx::core::reshape(dx1_3, gpu::to_mlx_shape(x1_shape));
    auto dx2_3 = ::mlx::core::matmul(G_b1, Q);
    auto dx2r = ::mlx::core::reshape(dx2_3, gpu::to_mlx_shape(x2_shape));
    auto x1_b = ::mlx::core::reshape(x1_f, {iB, iD1, 1});
    auto x2_b = ::mlx::core::reshape(x2_f, {iB, 1, iD2});
    auto outer = ::mlx::core::multiply(::mlx::core::broadcast_to(x1_b, {iB, iD1, iD2}),
                                       ::mlx::core::broadcast_to(x2_b, {iB, iD1, iD2}));
    auto outer_flat = ::mlx::core::reshape(outer, {iB, iD1 * iD2});
    auto G_t = ::mlx::core::transpose(G, {1, 0});
    auto dW_flat = ::mlx::core::matmul(G_t, outer_flat);
    auto dWr = ::mlx::core::reshape(dW_flat, gpu::to_mlx_shape(w_shape));
    std::vector<Storage> out;
    out.push_back(Storage{gpu::wrap_mlx_array(std::move(dx1r), dt)});
    out.push_back(Storage{gpu::wrap_mlx_array(std::move(dx2r), dt)});
    out.push_back(Storage{gpu::wrap_mlx_array(std::move(dWr), dt)});
    if (has_bias) {
        auto db = sum_widened(G, std::vector<int>{0}, false);
        out.push_back(Storage{gpu::wrap_mlx_array(std::move(db), dt)});
    } else {
        CpuStorage empty;
        empty.dtype = dt;
        empty.nbytes = 0;
        out.push_back(Storage{std::move(empty)});
    }
    return out;
}

Storage GpuBackend::inner(
    const Storage& a, const Storage& b, const Shape&, const Shape&, const Shape&, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    const auto& gb = std::get<GpuStorage>(b);
    auto out = ::mlx::core::inner(*ga.arr, *gb.arr);
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

Storage GpuBackend::tensordot(const Storage& a,
                              const Storage& b,
                              const Shape&,
                              const Shape&,
                              const Shape&,
                              const std::vector<int>& axes_a,
                              const std::vector<int>& axes_b,
                              Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    const auto& gb = std::get<GpuStorage>(b);
    const bool discrete = dt != Dtype::F16 && dt != Dtype::BF16 && dt != Dtype::F32 &&
                          dt != Dtype::F64 && dt != Dtype::C64;
    if (discrete) {
        auto a_arr = ::mlx::core::astype(*ga.arr, ::mlx::core::float32);
        auto b_arr = ::mlx::core::astype(*gb.arr, ::mlx::core::float32);
        auto wide = ::mlx::core::tensordot(a_arr, b_arr, axes_a, axes_b);
        auto rounded = ::mlx::core::round(wide);
        return Storage{
            gpu::wrap_mlx_array(::mlx::core::astype(rounded, gpu::to_mlx_dtype(dt)), dt)};
    }
    auto out = ::mlx::core::tensordot(*ga.arr, *gb.arr, axes_a, axes_b);
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

}  // namespace backend
}  // namespace lucid
