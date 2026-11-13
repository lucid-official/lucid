// lucid/_C/backend/gpu/methods/Norm.cpp
// Out-of-line GpuBackend method definitions for the Norm family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

StoragePair GpuBackend::rms_norm_forward(const Storage& x,
                                         const Storage& gamma,
                                         std::size_t outer,
                                         std::size_t normalized_size,
                                         double eps,
                                         const Shape& x_shape,
                                         Dtype dt) {
    // Replace the 4-op chain (square → mean → add+rsqrt → multiply×2)
    // with MLX's fused ``fast::rms_norm`` for the main output, plus a
    // small standalone ``mean(square)`` + ``rsqrt`` for the saved rstd
    // tensor that the backward needs.  Llama-scale: Mac Studio fwd
    // 1.5 ms (chain) → ~0.5 ms (target, matching the reference framework).
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(gamma);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    ::mlx::core::Shape flat_x{static_cast<int>(outer), static_cast<int>(normalized_size)};
    auto x_2d = ::mlx::core::reshape(*gx.arr, flat_x);
    auto g_1d = ::mlx::core::reshape(*gg.arr, {static_cast<int>(normalized_size)});

    auto y_2d = ::mlx::core::fast::rms_norm(x_2d, std::optional<::mlx::core::array>{g_1d},
                                            static_cast<float>(eps));
    auto y = ::mlx::core::reshape(y_2d, gpu::to_mlx_shape(x_shape));

    // Saved tensor for backward: rstd = 1 / sqrt(mean(x^2) + eps), in
    // float32 for a half input (see ``widen_half``) — a float16 square
    // overflows past |x| = 256 and saved an rstd of 0.
    const auto acc_dt = accumulation_dtype(mlx_dt);
    auto ms = ::mlx::core::mean(squared_deviation(x_2d, ::mlx::core::zeros({}, acc_dt)),
                                std::vector<int>{1}, true);
    auto rstd =
        narrow_to(::mlx::core::rsqrt(::mlx::core::add(ms, gpu::mlx_scalar(eps, acc_dt))), mlx_dt);
    return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(y), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(rstd), dt)}};
}

StoragePair GpuBackend::rms_norm_backward(const Storage& x,
                                          const Storage& gamma,
                                          const Storage& saved_rstd,
                                          const Storage& grad,
                                          std::size_t outer,
                                          std::size_t normalized_size,
                                          const Shape& x_shape,
                                          const Shape& gamma_shape,
                                          Dtype dt,
                                          GradMask2 grad_mask) {
    if (!grad_mask.any())
        return {Storage{}, Storage{}};
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gr = std::get<GpuStorage>(saved_rstd);
    const auto& ggrad = std::get<GpuStorage>(grad);
    ::mlx::core::Shape flat_x{static_cast<int>(outer), static_cast<int>(normalized_size)};
    auto x_2d = ::mlx::core::reshape(*gx.arr, flat_x);
    auto grad_2d = ::mlx::core::reshape(*ggrad.arr, flat_x);
    auto xnorm = ::mlx::core::multiply(x_2d, *gr.arr);
    // Both sums accumulate in float32 for a half gradient (see
    // ``widen_half``); the dx chain runs fused in that dtype.
    const auto g_dt = grad_2d.dtype();
    const auto acc_dt = accumulation_dtype(g_dt);
    return {
        grad_mask.make(
            0,
            [&] {
                auto gx_scaled = scaled_by_gamma(grad_2d, gamma, normalized_size);
                // 3.4+ Phase A.7: mean -> sum x 1/N (same as Phase A.2 for BN).
                // Helps Llama-style RMSNorm-using transformer workloads.
                const ::mlx::core::array inv_N(1.0 / static_cast<double>(normalized_size), acc_dt);
                auto sum_gx_xn_kept =
                    ::mlx::core::sum(product_widened(gx_scaled, xnorm), std::vector<int>{1}, true);
                auto m = ::mlx::core::multiply(sum_gx_xn_kept, inv_N);
                // dx = rstd * (gx_scaled - xnorm * m): RMSNorm subtracts no mean.
                auto dx_2d = normalized_input_grad(gx_scaled, xnorm, ::mlx::core::zeros({}, acc_dt),
                                                   m, *gr.arr);
                return grad_storage(::mlx::core::reshape(dx_2d, gpu::to_mlx_shape(x_shape)), dt);
            }),
        grad_mask.make(1, [&] {
            auto dgamma_2d = scale_grad(grad_2d, xnorm, std::vector<int>{0});
            return grad_storage(::mlx::core::reshape(dgamma_2d, gpu::to_mlx_shape(gamma_shape)),
                                dt);
        })};
}

std::vector<Storage> GpuBackend::layer_norm_forward(const Storage& x,
                                                    const Storage& gamma,
                                                    const Storage& beta,
                                                    std::size_t outer,
                                                    std::size_t normalized_size,
                                                    double eps,
                                                    const Shape& x_shape,
                                                    Dtype dt) {
    // Replace the 7-op chain (mean → subtract → square → mean → add+rsqrt
    // → multiply → multiply+add) with MLX's fused ``fast::layer_norm`` for
    // the main output, plus a small ``mean`` + ``var`` reduction pair
    // for the saved-tensor rstd that the backward needs.  Llama-scale:
    // Mac Studio fwd 2.4 ms (chain) → ~0.6 ms (target, matching the
    // reference framework).
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(gamma);
    const auto& gb = std::get<GpuStorage>(beta);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    ::mlx::core::Shape flat_x{static_cast<int>(outer), static_cast<int>(normalized_size)};
    auto x_2d = ::mlx::core::reshape(*gx.arr, flat_x);
    auto g_1d = ::mlx::core::reshape(*gg.arr, {static_cast<int>(normalized_size)});
    auto b_1d = ::mlx::core::reshape(*gb.arr, {static_cast<int>(normalized_size)});

    auto y_2d = ::mlx::core::fast::layer_norm(x_2d, std::optional<::mlx::core::array>{g_1d},
                                              std::optional<::mlx::core::array>{b_1d},
                                              static_cast<float>(eps));
    auto y = ::mlx::core::reshape(y_2d, gpu::to_mlx_shape(x_shape));

    // Saved tensors for backward: mean and rstd, the population variance
    // ``mean(square(x - mean))``.  ``fast::layer_norm`` accumulates in
    // float32 internally; these accumulate in float32 for a half input
    // too (see ``widen_half``) — a float16 row of 4096 values of 16
    // summed to inf and saved an rstd of 0.
    if (x_2d.size() > 0) {
        auto stats =
            omi_norm_stats(x_2d, 1, static_cast<int>(outer), static_cast<int>(normalized_size), eps,
                           {static_cast<int>(outer), 1});
        return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(y), dt)},
                Storage{gpu::wrap_mlx_array(std::move(stats[1]), dt)},
                Storage{gpu::wrap_mlx_array(std::move(stats[2]), dt)}};
    }
    const auto acc_dt = accumulation_dtype(mlx_dt);
    auto mean_acc = ::mlx::core::mean(widen_half(x_2d), std::vector<int>{1}, true);
    auto var_acc = ::mlx::core::mean(squared_deviation(x_2d, mean_acc), std::vector<int>{1}, true);
    auto mean = narrow_to(mean_acc, mlx_dt);
    auto rstd = narrow_to(
        ::mlx::core::rsqrt(::mlx::core::add(var_acc, gpu::mlx_scalar(eps, acc_dt))), mlx_dt);
    return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(y), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(mean), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(rstd), dt)}};
}

std::vector<Storage> GpuBackend::layer_norm_backward(const Storage& x,
                                                     const Storage& gamma,
                                                     const Storage& saved_mean,
                                                     const Storage& saved_rstd,
                                                     const Storage& grad,
                                                     std::size_t outer,
                                                     std::size_t normalized_size,
                                                     const Shape& x_shape,
                                                     const Shape& gamma_shape,
                                                     const Shape& beta_shape,
                                                     Dtype dt,
                                                     GradMask3 grad_mask) {
    if (!grad_mask.any())
        return GradMask3::none();
    const auto& ggrad = std::get<GpuStorage>(grad);
    ::mlx::core::Shape flat_x{static_cast<int>(outer), static_cast<int>(normalized_size)};
    auto grad_2d = ::mlx::core::reshape(*ggrad.arr, flat_x);
    const auto g_dt = grad_2d.dtype();
    const auto acc_dt = accumulation_dtype(g_dt);

    Storage dbeta_out = grad_mask.make(2, [&] {
        auto dbeta_2d = sum_widened(grad_2d, std::vector<int>{0}, false);
        return grad_storage(::mlx::core::reshape(dbeta_2d, gpu::to_mlx_shape(beta_shape)), dt);
    });
    if (!grad_mask.want_any(0, 1))
        return {Storage{}, Storage{}, std::move(dbeta_out)};

    const auto& gx = std::get<GpuStorage>(x);
    const auto& gm = std::get<GpuStorage>(saved_mean);
    const auto& gr = std::get<GpuStorage>(saved_rstd);
    auto x_2d = ::mlx::core::reshape(*gx.arr, flat_x);
    auto centered = ::mlx::core::subtract(x_2d, *gm.arr);
    auto xnorm = ::mlx::core::multiply(centered, *gr.arr);

    return {
        grad_mask.make(
            0,
            [&] {
                auto gx_scaled = scaled_by_gamma(grad_2d, gamma, normalized_size);
                const ::mlx::core::array inv_N(1.0 / static_cast<double>(normalized_size), acc_dt);
                auto sum_gx_kept =
                    ::mlx::core::sum(widen_half(gx_scaled), std::vector<int>{1}, true);
                auto sum_gx_xn_kept =
                    ::mlx::core::sum(product_widened(gx_scaled, xnorm), std::vector<int>{1}, true);
                auto mean1 = ::mlx::core::multiply(sum_gx_kept, inv_N);
                auto mean2 = ::mlx::core::multiply(sum_gx_xn_kept, inv_N);
                auto dx_2d = normalized_input_grad(gx_scaled, xnorm, mean1, mean2, *gr.arr);
                return grad_storage(::mlx::core::reshape(dx_2d, gpu::to_mlx_shape(x_shape)), dt);
            }),
        grad_mask.make(1,
                       [&] {
                           auto dgamma_2d = scale_grad(grad_2d, xnorm, std::vector<int>{0});
                           return grad_storage(
                               ::mlx::core::reshape(dgamma_2d, gpu::to_mlx_shape(gamma_shape)), dt);
                       }),
        std::move(dbeta_out)};
}

std::vector<Storage> GpuBackend::batch_norm_forward(const Storage& x,
                                                    const Storage& gamma,
                                                    const Storage& beta,
                                                    int,
                                                    int channels,
                                                    int,
                                                    int ndim,
                                                    double eps,
                                                    const Shape& x_shape_param,
                                                    Dtype dt) {
    (void)x_shape_param;
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(gamma);
    const auto& gb = std::get<GpuStorage>(beta);
    return bn_forward_gpu(*gx.arr, *gg.arr, *gb.arr, channels, ndim, eps, dt);
}

std::vector<Storage> GpuBackend::batch_norm_backward(const Storage& x,
                                                     const Storage& gamma,
                                                     const Storage& saved_mean,
                                                     const Storage& saved_rstd,
                                                     const Storage& saved_xnorm,
                                                     const Storage& grad,
                                                     int,
                                                     int channels,
                                                     int,
                                                     int ndim,
                                                     const Shape& x_shape_param,
                                                     Dtype dt,
                                                     double eps,
                                                     GradMask3 grad_mask) {
    (void)x_shape_param;
    (void)eps;
    const auto& gm = std::get<GpuStorage>(saved_mean);
    const auto& gr = std::get<GpuStorage>(saved_rstd);
    const auto& ggrad = std::get<GpuStorage>(grad);
    return bn_backward_gpu(x, gamma, *gm.arr, *gr.arr, saved_xnorm, *ggrad.arr, channels, ndim, dt,
                           grad_mask);
}

std::vector<Storage> GpuBackend::group_norm_forward(const Storage& x,
                                                    const Storage& gamma,
                                                    const Storage& beta,
                                                    int batch,
                                                    int channels,
                                                    int,
                                                    int groups,
                                                    const std::vector<int>& spatial_dims,
                                                    double eps,
                                                    const Shape& x_shape,
                                                    Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(gamma);
    const auto& gb = std::get<GpuStorage>(beta);
    const int ndim = static_cast<int>(spatial_dims.size());
    const int channels_per_group = channels / groups;
    using SE = ::mlx::core::ShapeElem;
    const auto grouped = grouped_shape(batch, groups, channels_per_group, spatial_dims);
    auto x_g = ::mlx::core::reshape(*gx.arr, grouped);

    std::vector<int> reduce_axes;
    reduce_axes.reserve(static_cast<std::size_t>(ndim) + 1);
    reduce_axes.push_back(2);
    for (int i = 0; i < ndim; ++i)
        reduce_axes.push_back(3 + i);
    // Statistics in float32 for a half input, as in batch_norm_forward;
    // the affine pass runs in the grouped layout so it is one fused
    // kernel (gamma / beta are viewed per channel within each group).
    const auto x_dt = x_g.dtype();
    const auto acc_dt = accumulation_dtype(x_dt);
    auto mean_acc = ::mlx::core::mean(widen_half(x_g), reduce_axes, true);
    auto var_acc = ::mlx::core::mean(squared_deviation(x_g, mean_acc), reduce_axes, true);
    auto rstd_acc = ::mlx::core::rsqrt(::mlx::core::add(var_acc, gpu::mlx_scalar(eps, acc_dt)));
    ::mlx::core::Shape br_g(static_cast<std::size_t>(ndim) + 3, 1);
    br_g[1] = static_cast<SE>(groups);
    br_g[2] = static_cast<SE>(channels_per_group);
    auto g_view = ::mlx::core::reshape(*gg.arr, br_g);
    auto b_view = ::mlx::core::reshape(*gb.arr, br_g);
    auto y_g = mlx_fused(
        {x_g, mean_acc, rstd_acc, g_view, b_view}, [](const std::vector<::mlx::core::array>& in) {
            auto xn = ::mlx::core::multiply(::mlx::core::subtract(widen_half(in[0]), in[1]), in[2]);
            auto yv =
                ::mlx::core::add(::mlx::core::multiply(xn, widen_half(in[3])), widen_half(in[4]));
            return std::vector<::mlx::core::array>{narrow_to(yv, in[0].dtype())};
        })[0];
    auto y = ::mlx::core::reshape(y_g, gpu::to_mlx_shape(x_shape));
    ::mlx::core::Shape mr{static_cast<SE>(batch), static_cast<SE>(groups)};
    auto mean = narrow_to(::mlx::core::reshape(mean_acc, mr), x_dt);
    auto rstd = narrow_to(::mlx::core::reshape(rstd_acc, mr), x_dt);
    return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(y), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(mean), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(rstd), dt)}};
}

std::vector<Storage> GpuBackend::group_norm_backward(const Storage& x,
                                                     const Storage& gamma,
                                                     const Storage& saved_mean,
                                                     const Storage& saved_rstd,
                                                     const Storage& grad,
                                                     int batch,
                                                     int channels,
                                                     int,
                                                     int groups,
                                                     const std::vector<int>& spatial_dims,
                                                     const Shape& x_shape,
                                                     Dtype dt,
                                                     GradMask3 grad_mask) {
    if (!grad_mask.any())
        return GradMask3::none();
    const auto& ggrad = std::get<GpuStorage>(grad);
    const int ndim = static_cast<int>(spatial_dims.size());
    const int channels_per_group = channels / groups;
    using SE = ::mlx::core::ShapeElem;

    const auto ch_axes = non_channel_axes(ndim);
    const auto g_dt = ggrad.arr->dtype();
    const auto acc_dt = accumulation_dtype(g_dt);

    Storage dbeta_out = grad_mask.make(
        2, [&] { return grad_storage(sum_widened(*ggrad.arr, ch_axes, false), dt); });
    if (!grad_mask.want_any(0, 1))
        return {Storage{}, Storage{}, std::move(dbeta_out)};

    const auto& gx = std::get<GpuStorage>(x);
    const auto& gm = std::get<GpuStorage>(saved_mean);
    const auto& gr = std::get<GpuStorage>(saved_rstd);
    const auto grouped = grouped_shape(batch, groups, channels_per_group, spatial_dims);
    ::mlx::core::Shape mr_shape;
    mr_shape.reserve(static_cast<std::size_t>(ndim) + 3);
    mr_shape.push_back(static_cast<SE>(batch));
    mr_shape.push_back(static_cast<SE>(groups));
    for (int i = 0; i < ndim + 1; ++i)
        mr_shape.push_back(1);

    auto x_g = ::mlx::core::reshape(*gx.arr, grouped);
    auto mean_g = ::mlx::core::reshape(*gm.arr, mr_shape);
    auto rstd_g = ::mlx::core::reshape(*gr.arr, mr_shape);
    auto centered = ::mlx::core::subtract(x_g, mean_g);
    auto xnorm_g = ::mlx::core::multiply(centered, rstd_g);

    return {grad_mask.make(
                0,
                [&] {
                    const auto& gg = std::get<GpuStorage>(gamma);
                    ::mlx::core::Shape br_c(static_cast<std::size_t>(ndim) + 2, 1);
                    br_c[1] = static_cast<SE>(channels);
                    std::vector<int> red_axes;
                    red_axes.reserve(static_cast<std::size_t>(ndim) + 1);
                    red_axes.push_back(2);
                    for (int i = 0; i < ndim; ++i)
                        red_axes.push_back(3 + i);
                    auto gamma_view = ::mlx::core::reshape(*gg.arr, br_c);
                    auto gx_scaled4 = ::mlx::core::multiply(gamma_view, *ggrad.arr);
                    auto gx_scaled_g = ::mlx::core::reshape(gx_scaled4, grouped);

                    std::int64_t N_reduced = static_cast<std::int64_t>(channels_per_group);
                    for (int s : spatial_dims)
                        N_reduced *= static_cast<std::int64_t>(s);
                    const ::mlx::core::array inv_N(1.0 / static_cast<double>(N_reduced), acc_dt);
                    auto sum_gx_kept = ::mlx::core::sum(widen_half(gx_scaled_g), red_axes, true);
                    auto sum_gx_xn_kept =
                        ::mlx::core::sum(product_widened(gx_scaled_g, xnorm_g), red_axes, true);
                    auto mean1 = ::mlx::core::multiply(sum_gx_kept, inv_N);
                    auto mean2 = ::mlx::core::multiply(sum_gx_xn_kept, inv_N);
                    auto dx_g = normalized_input_grad(gx_scaled_g, xnorm_g, mean1, mean2, rstd_g);
                    return grad_storage(::mlx::core::reshape(dx_g, gpu::to_mlx_shape(x_shape)), dt);
                }),
            grad_mask.make(1,
                           [&] {
                               auto xnorm =
                                   ::mlx::core::reshape(xnorm_g, gpu::to_mlx_shape(x_shape));
                               return grad_storage(scale_grad(*ggrad.arr, xnorm, ch_axes), dt);
                           }),
            std::move(dbeta_out)};
}

std::vector<Storage> GpuBackend::batch_norm_eval_forward(const Storage& x,
                                                         const Storage& mean,
                                                         const Storage& var,
                                                         const Storage& gamma,
                                                         const Storage& beta,
                                                         const Shape& x_shape,
                                                         int C,
                                                         int,
                                                         double eps,
                                                         Dtype dt) {
    // Widen 16-bit floats to float32 at the door and round once on the way
    // out, matching the CPU eval kernel (LCD-487 / LCD-516).
    const auto xw = widen_gpu(x), vw = widen_gpu(var);
    auto rstd = ::mlx::core::reciprocal(
        ::mlx::core::sqrt(::mlx::core::add(vw, gpu::mlx_scalar(eps, xw.dtype()))));
    ::mlx::core::Shape b_shape(x_shape.size(), 1);
    b_shape[1] = C;
    auto m_b = ::mlx::core::reshape(widen_gpu(mean), b_shape);
    auto r_b = ::mlx::core::reshape(rstd, b_shape);
    auto g_b = ::mlx::core::reshape(widen_gpu(gamma), b_shape);
    auto bb_b = ::mlx::core::reshape(widen_gpu(beta), b_shape);
    auto y = ::mlx::core::add(
        ::mlx::core::multiply(g_b, ::mlx::core::multiply(::mlx::core::subtract(xw, m_b), r_b)),
        bb_b);
    return {wrap_narrow(y, dt), wrap_narrow(rstd, dt)};
}

std::vector<Storage> GpuBackend::batch_norm_eval_backward(const Storage& x,
                                                          const Storage& mean,
                                                          const Storage& gamma,
                                                          const Storage& rstd,
                                                          const Storage& grad_out,
                                                          const Shape& x_shape,
                                                          int C,
                                                          int,
                                                          Dtype dt,
                                                          GradMask5 grad_mask) {
    if (!grad_mask.any())
        return GradMask5::none();
    const auto rw = widen_gpu(rstd), gow = widen_gpu(grad_out);
    ::mlx::core::Shape b_shape(x_shape.size(), 1);
    b_shape[1] = C;
    auto r_b = ::mlx::core::reshape(rw, b_shape);
    const auto reduce_axes = non_channel_axes(static_cast<int>(x_shape.size()) - 2);

    // Slots: dx, dmean, dvar, dgamma, dbeta.
    std::optional<::mlx::core::array> gw;
    if (grad_mask.want_any(0, 1, 2))
        gw = widen_gpu(gamma);

    std::optional<::mlx::core::array> sum_g;
    if (grad_mask.want_any(1, 4))
        sum_g = ::mlx::core::sum(gow, reduce_axes, false);

    std::optional<::mlx::core::array> sum_xm_g;
    if (grad_mask.want_any(2, 3)) {
        const auto xw = widen_gpu(x), mw = widen_gpu(mean);
        auto m_b = ::mlx::core::reshape(mw, b_shape);
        auto x_minus_m = ::mlx::core::subtract(xw, m_b);
        sum_xm_g = ::mlx::core::sum(::mlx::core::multiply(x_minus_m, gow), reduce_axes, false);
    }

    return {
        grad_mask.make(0,
                       [&] {
                           auto g_b = ::mlx::core::reshape(*gw, b_shape);
                           auto dx = ::mlx::core::multiply(::mlx::core::multiply(g_b, r_b), gow);
                           return wrap_narrow(dx, dt);
                       }),
        grad_mask.make(1,
                       [&] {
                           auto dm = ::mlx::core::negative(
                               ::mlx::core::multiply(*gw, ::mlx::core::multiply(rw, *sum_g)));
                           return wrap_narrow(dm, dt);
                       }),
        grad_mask.make(2,
                       [&] {
                           auto half = gpu::mlx_scalar(-0.5, gow.dtype());
                           auto r3 = ::mlx::core::multiply(rw, ::mlx::core::multiply(rw, rw));
                           auto dv = ::mlx::core::multiply(
                               half,
                               ::mlx::core::multiply(*gw, ::mlx::core::multiply(r3, *sum_xm_g)));
                           return wrap_narrow(dv, dt);
                       }),
        grad_mask.make(3, [&] { return wrap_narrow(::mlx::core::multiply(*sum_xm_g, rw), dt); }),
        grad_mask.make(4, [&] { return wrap_narrow(*sum_g, dt); })};
}

Storage GpuBackend::p_norm_forward(const Storage& a,
                                   const Shape&,
                                   double p,
                                   const std::vector<int>& axes,
                                   bool keepdims,
                                   Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    const auto raw = pnorm::scaled_p_norm(widen_half(*ga.arr), p, axes, keepdims);
    return Storage{gpu::wrap_mlx_array(narrow_half(raw, ga.arr->dtype()), dt)};
}

std::vector<Storage> GpuBackend::lp_normalize_forward(
    const Storage& x, const Shape& x_shape, double ord, int axis, double eps, Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    // The p-norm is taken in float32 for a half input (see
    // ``widen_half``) — a float16 |x|^2 overflows past 256 and the norm
    // came back inf, normalising the row to zeros — and rounded back.
    // An order above 1 takes the range-safe kernel, so a row at 1e-20 keeps
    // its norm rather than flushing to the ``eps`` floor (``sum(|x|^p)`` of it
    // underflows) and one at 1e25 does not overflow.
    auto N = [&] {
        if (ord > 1.0)
            return narrow_half(
                pnorm::scaled_p_norm(widen_half(*gx.arr), ord, std::vector<int>{axis}, true),
                mlx_dt);
        auto abs_w = widen_half(::mlx::core::abs(*gx.arr));
        auto sum_p = ::mlx::core::sum(gpu_pow(abs_w, ord), std::vector<int>{axis}, true);
        return narrow_half(gpu_pow(sum_p, 1.0 / ord), mlx_dt);
    }();
    auto eps_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(eps)), mlx_dt);
    auto N_clip = ::mlx::core::maximum(N, eps_arr);
    auto saved_norm = N_clip;
    auto y = ::mlx::core::divide(*gx.arr, N_clip);
    return {Storage{gpu::wrap_mlx_array(std::move(y), dt)},
            Storage{gpu::wrap_mlx_array(std::move(saved_norm), dt)}};
}

Storage GpuBackend::lp_normalize_backward(const Storage& x,
                                          const Storage& saved_norm,
                                          const Storage& grad_out,
                                          const Shape&,
                                          double ord,
                                          int axis,
                                          Dtype dt) {
    namespace mx = ::mlx::core;
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gn = std::get<GpuStorage>(saved_norm);
    const auto& gg = std::get<GpuStorage>(grad_out);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    // dx = (g - sgn(u) |u|^(p-1) <g, u>) / d with u = x / d: the usual
    // g/d - sgn(x) |x|^(p-1) <g, x> / d^(p+1) after factoring out d.  u is the
    // forward output, at most 1 in magnitude, so no power of the norm is ever
    // formed: d^(p+1) left the float32 range for a norm outside ~1e-13 .. 1e13
    // (p = 2), and the gradient came back NaN where it is an ordinary float.
    // The arithmetic is float32 for a half input (see ``widen_half``).
    const auto xw = widen_half(*gx.arr);
    const auto gw = widen_half(*gg.arr);
    const auto d = widen_half(*gn.arr);
    auto u = mx::divide(xw, d);
    auto proj = mx::sum(mx::multiply(gw, u), std::vector<int>{axis}, true);
    // For p = 2, sgn(u) |u| is u.
    auto weight = ord == 2.0 ? u : mx::multiply(mx::sign(u), gpu_pow(mx::abs(u), ord - 1.0));
    auto dx = mx::divide(mx::subtract(gw, mx::multiply(weight, proj)), d);
    return Storage{gpu::wrap_mlx_array(narrow_half(dx, mlx_dt), dt)};
}

std::vector<Storage> GpuBackend::global_response_norm_forward(const Storage& x,
                                                              const Storage& gamma,
                                                              const Storage& beta,
                                                              const Shape& x_shape,
                                                              double eps,
                                                              Dtype dt) {
    const int C = static_cast<int>(x_shape[1]), B = static_cast<int>(x_shape[0]);
    const auto xw = widen_gpu(x);
    auto G =
        ::mlx::core::sqrt(::mlx::core::sum(::mlx::core::square(xw), std::vector<int>{2, 3}, true));
    auto m = ::mlx::core::mean(G, std::vector<int>{1}, true);
    auto denom = ::mlx::core::add(m, gpu::mlx_scalar(eps, G.dtype()));
    auto Nx = ::mlx::core::divide(G, denom);
    auto g_b = ::mlx::core::reshape(widen_gpu(gamma), {1, C, 1, 1});
    auto bb_b = ::mlx::core::reshape(widen_gpu(beta), {1, C, 1, 1});
    auto y = ::mlx::core::add(::mlx::core::multiply(g_b, ::mlx::core::multiply(xw, Nx)),
                              ::mlx::core::multiply(bb_b, xw));
    return {wrap_narrow(y, dt), wrap_narrow(::mlx::core::reshape(Nx, {B, C}), dt)};
}

std::vector<Storage> GpuBackend::global_response_norm_backward(const Storage& x,
                                                               const Storage& gamma,
                                                               const Storage& beta,
                                                               const Storage& saved_Nx,
                                                               const Storage& grad_out,
                                                               const Shape& x_shape,
                                                               double eps,
                                                               Dtype dt,
                                                               GradMask3 grad_mask) {
    const int B = static_cast<int>(x_shape[0]), C = static_cast<int>(x_shape[1]);
    const auto xw = widen_gpu(x), gw = widen_gpu(grad_out);
    const auto acc_dt = xw.dtype();
    auto gx_prod = ::mlx::core::multiply(gw, xw);
    std::optional<::mlx::core::array> Nx;
    if (grad_mask.want_any(0, 1))
        Nx = ::mlx::core::reshape(widen_gpu(saved_Nx), {B, C, 1, 1});
    return {grad_mask.make(
                0,
                [&] {
                    auto G = ::mlx::core::sqrt(
                        ::mlx::core::sum(::mlx::core::square(xw), std::vector<int>{2, 3}, true));
                    auto m_b = ::mlx::core::mean(G, std::vector<int>{1}, true);
                    auto denom = ::mlx::core::add(m_b, gpu::mlx_scalar(eps, acc_dt));
                    auto g_b = ::mlx::core::reshape(widen_gpu(gamma), {1, C, 1, 1});
                    auto sum_gx = ::mlx::core::sum(gx_prod, std::vector<int>{2, 3}, true);
                    auto A = ::mlx::core::multiply(g_b, sum_gx);
                    auto AG = ::mlx::core::multiply(A, G);
                    auto inner_sum = ::mlx::core::sum(AG, std::vector<int>{1}, true);
                    auto C_arr = gpu::mlx_scalar(static_cast<double>(C), acc_dt);
                    auto denom_sq = ::mlx::core::multiply(denom, denom);
                    auto second =
                        ::mlx::core::divide(inner_sum, ::mlx::core::multiply(denom_sq, C_arr));
                    auto dG = ::mlx::core::subtract(::mlx::core::divide(A, denom), second);
                    auto eps_g = gpu::mlx_scalar(1e-12, acc_dt);
                    auto G_safe = ::mlx::core::maximum(G, eps_g);
                    auto dG_term = ::mlx::core::divide(::mlx::core::multiply(dG, xw), G_safe);
                    auto bb_b = ::mlx::core::reshape(widen_gpu(beta), {1, C, 1, 1});
                    auto dx = ::mlx::core::add(
                        ::mlx::core::add(::mlx::core::multiply(::mlx::core::multiply(g_b, *Nx), gw),
                                         ::mlx::core::multiply(bb_b, gw)),
                        dG_term);
                    return wrap_narrow(dx, dt);
                }),
            grad_mask.make(1,
                           [&] {
                               auto gxN = ::mlx::core::multiply(gx_prod, *Nx);
                               return wrap_narrow(
                                   ::mlx::core::sum(gxN, std::vector<int>{0, 2, 3}, false), dt);
                           }),
            grad_mask.make(2, [&] {
                return wrap_narrow(::mlx::core::sum(gx_prod, std::vector<int>{0, 2, 3}, false), dt);
            })};
}

}  // namespace backend
}  // namespace lucid
