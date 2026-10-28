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
                                          Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(gamma);
    const auto& gr = std::get<GpuStorage>(saved_rstd);
    const auto& ggrad = std::get<GpuStorage>(grad);
    ::mlx::core::Shape flat_x{static_cast<int>(outer), static_cast<int>(normalized_size)};
    ::mlx::core::Shape flat_g{1, static_cast<int>(normalized_size)};
    auto x_2d = ::mlx::core::reshape(*gx.arr, flat_x);
    auto grad_2d = ::mlx::core::reshape(*ggrad.arr, flat_x);
    auto gamma_2d = ::mlx::core::reshape(*gg.arr, flat_g);
    auto xnorm = ::mlx::core::multiply(x_2d, *gr.arr);
    // Both sums accumulate in float32 for a half gradient (see
    // ``widen_half``); the dx chain runs fused in that dtype.
    const auto g_dt = grad_2d.dtype();
    const auto acc_dt = accumulation_dtype(g_dt);
    auto dgamma_2d = narrow_to(
        ::mlx::core::sum(product_widened(grad_2d, xnorm), std::vector<int>{0}, false), g_dt);
    auto gx_scaled = ::mlx::core::multiply(grad_2d, gamma_2d);
    // 3.4+ Phase A.7: mean → sum × 1/N (same as Phase A.2 for BN).
    // Helps Llama-style RMSNorm-using transformer workloads.
    const ::mlx::core::array inv_N(1.0 / static_cast<double>(normalized_size), acc_dt);
    auto sum_gx_xn_kept =
        ::mlx::core::sum(product_widened(gx_scaled, xnorm), std::vector<int>{1}, true);
    auto m = ::mlx::core::multiply(sum_gx_xn_kept, inv_N);
    // dx = rstd * (gx_scaled - xnorm * m): RMSNorm subtracts no mean.
    auto dx_2d =
        normalized_input_grad(gx_scaled, xnorm, ::mlx::core::zeros({}, acc_dt), m, *gr.arr);
    auto dx = ::mlx::core::reshape(dx_2d, gpu::to_mlx_shape(x_shape));
    auto dgamma = ::mlx::core::reshape(dgamma_2d, gpu::to_mlx_shape(gamma_shape));
    // 3.4 backward sweep: ``multiply`` / ``sum`` produce contiguous
    // tensors and ``reshape`` preserves contiguity, so the trailing
    // ``contiguous(...)`` wrap is a redundant materialisation that
    // breaks lazy fusion at the autograd boundary.
    return {Storage{gpu::wrap_mlx_array(std::move(dx), dt)},
            Storage{gpu::wrap_mlx_array(std::move(dgamma), dt)}};
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
                                                     Dtype dt) {
    (void)x_shape;
    (void)gamma_shape;
    (void)beta_shape;
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(gamma);
    const auto& gm = std::get<GpuStorage>(saved_mean);
    const auto& gr = std::get<GpuStorage>(saved_rstd);
    const auto& ggrad = std::get<GpuStorage>(grad);
    ::mlx::core::Shape flat_x{static_cast<int>(outer), static_cast<int>(normalized_size)};
    ::mlx::core::Shape flat_g{1, static_cast<int>(normalized_size)};
    auto x_2d = ::mlx::core::reshape(*gx.arr, flat_x);
    auto grad_2d = ::mlx::core::reshape(*ggrad.arr, flat_x);
    auto gamma_2d = ::mlx::core::reshape(*gg.arr, flat_g);
    auto centered = ::mlx::core::subtract(x_2d, *gm.arr);
    auto xnorm = ::mlx::core::multiply(centered, *gr.arr);
    // Every sum below accumulates in float32 for a half gradient (see
    // ``widen_half``); the dx chain runs fused in that dtype.
    const auto g_dt = grad_2d.dtype();
    const auto acc_dt = accumulation_dtype(g_dt);
    auto dbeta_2d = sum_widened(grad_2d, std::vector<int>{0}, false);
    auto dgamma_2d = narrow_to(
        ::mlx::core::sum(product_widened(grad_2d, xnorm), std::vector<int>{0}, false), g_dt);
    auto gx_scaled = ::mlx::core::multiply(grad_2d, gamma_2d);

    // 3.4+ Phase A.7: same restructure as Phase A.2 (BN backward) — replace
    // ``mean(...)`` with ``sum(..., keepdims=true) × 1/N`` so MLX dispatches
    // a sum kernel + scalar broadcast multiply instead of the dedicated
    // mean primitive.  Helps every transformer / NLP workload using
    // LayerNorm: the per-op cost-per-bwd-call drops a small amount, and
    // because the backward chain doesn't MLX-fuse (multi-output structural
    // blocker), the savings translate directly to wall time.
    const ::mlx::core::array inv_N(1.0 / static_cast<double>(normalized_size), acc_dt);
    auto sum_gx_kept = ::mlx::core::sum(widen_half(gx_scaled), std::vector<int>{1}, true);
    auto sum_gx_xn_kept =
        ::mlx::core::sum(product_widened(gx_scaled, xnorm), std::vector<int>{1}, true);
    auto mean1 = ::mlx::core::multiply(sum_gx_kept, inv_N);
    auto mean2 = ::mlx::core::multiply(sum_gx_xn_kept, inv_N);
    auto dx_2d = normalized_input_grad(gx_scaled, xnorm, mean1, mean2, *gr.arr);
    auto dx = ::mlx::core::reshape(dx_2d, gpu::to_mlx_shape(x_shape));
    auto dgamma = ::mlx::core::reshape(dgamma_2d, gpu::to_mlx_shape(gamma_shape));
    auto dbeta = ::mlx::core::reshape(dbeta_2d, gpu::to_mlx_shape(beta_shape));
    // 3.4 backward sweep: reduction outputs are contiguous and
    // ``reshape`` preserves contiguity.  See BatchNorm backward.
    return {Storage{gpu::wrap_mlx_array(std::move(dx), dt)},
            Storage{gpu::wrap_mlx_array(std::move(dgamma), dt)},
            Storage{gpu::wrap_mlx_array(std::move(dbeta), dt)}};
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
                                                     double eps) {
    (void)x_shape_param;
    (void)eps;
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(gamma);
    const auto& gm = std::get<GpuStorage>(saved_mean);
    const auto& gr = std::get<GpuStorage>(saved_rstd);
    const auto& ggrad = std::get<GpuStorage>(grad);
    return bn_backward_gpu(*gx.arr, *gg.arr, *gm.arr, *gr.arr, saved_xnorm, *ggrad.arr, channels,
                           ndim, dt);
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
    ::mlx::core::Shape grouped;
    grouped.reserve(static_cast<std::size_t>(ndim) + 3);
    grouped.push_back(static_cast<SE>(batch));
    grouped.push_back(static_cast<SE>(groups));
    grouped.push_back(static_cast<SE>(channels_per_group));
    for (int s : spatial_dims)
        grouped.push_back(static_cast<SE>(s));
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
                                                     Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(gamma);
    const auto& gm = std::get<GpuStorage>(saved_mean);
    const auto& gr = std::get<GpuStorage>(saved_rstd);
    const auto& ggrad = std::get<GpuStorage>(grad);
    const int ndim = static_cast<int>(spatial_dims.size());
    const int channels_per_group = channels / groups;
    using SE = ::mlx::core::ShapeElem;
    ::mlx::core::Shape grouped;
    grouped.reserve(static_cast<std::size_t>(ndim) + 3);
    grouped.push_back(static_cast<SE>(batch));
    grouped.push_back(static_cast<SE>(groups));
    grouped.push_back(static_cast<SE>(channels_per_group));
    for (int s : spatial_dims)
        grouped.push_back(static_cast<SE>(s));
    ::mlx::core::Shape mr_shape;
    mr_shape.reserve(static_cast<std::size_t>(ndim) + 3);
    mr_shape.push_back(static_cast<SE>(batch));
    mr_shape.push_back(static_cast<SE>(groups));
    for (int i = 0; i < ndim + 1; ++i)
        mr_shape.push_back(1);
    ::mlx::core::Shape br_c(static_cast<std::size_t>(ndim) + 2, 1);
    br_c[1] = static_cast<SE>(channels);

    auto x_g = ::mlx::core::reshape(*gx.arr, grouped);
    auto mean_g = ::mlx::core::reshape(*gm.arr, mr_shape);
    auto rstd_g = ::mlx::core::reshape(*gr.arr, mr_shape);
    auto centered = ::mlx::core::subtract(x_g, mean_g);
    auto xnorm_g = ::mlx::core::multiply(centered, rstd_g);
    auto xnorm = ::mlx::core::reshape(xnorm_g, gpu::to_mlx_shape(x_shape));

    std::vector<int> ch_axes;
    ch_axes.reserve(static_cast<std::size_t>(ndim) + 1);
    ch_axes.push_back(0);
    for (int i = 0; i < ndim; ++i)
        ch_axes.push_back(2 + i);
    // Every sum below accumulates in float32 for a half gradient (see
    // ``widen_half``); the dx chain runs fused in that dtype.
    const auto g_dt = ggrad.arr->dtype();
    const auto acc_dt = accumulation_dtype(g_dt);
    auto dgamma =
        narrow_to(::mlx::core::sum(product_widened(*ggrad.arr, xnorm), ch_axes, false), g_dt);
    auto dbeta = sum_widened(*ggrad.arr, ch_axes, false);

    std::vector<int> red_axes;
    red_axes.reserve(static_cast<std::size_t>(ndim) + 1);
    red_axes.push_back(2);
    for (int i = 0; i < ndim; ++i)
        red_axes.push_back(3 + i);
    auto gamma_view = ::mlx::core::reshape(*gg.arr, br_c);
    auto gx_scaled4 = ::mlx::core::multiply(gamma_view, *ggrad.arr);
    auto gx_scaled_g = ::mlx::core::reshape(gx_scaled4, grouped);

    // 3.4+ Phase A.7: mean → sum × 1/N (same as Phase A.2 for BN).
    // N = channels_per_group × product(spatial_dims).  Helps segmentation
    // models (U-Net) and ConvNeXt-family architectures using GroupNorm.
    std::int64_t N_reduced = static_cast<std::int64_t>(channels_per_group);
    for (int s : spatial_dims)
        N_reduced *= static_cast<std::int64_t>(s);
    const ::mlx::core::array inv_N(1.0 / static_cast<double>(N_reduced), acc_dt);
    auto sum_gx_kept = ::mlx::core::sum(widen_half(gx_scaled_g), red_axes, true);
    auto sum_gx_xn_kept = ::mlx::core::sum(product_widened(gx_scaled_g, xnorm_g), red_axes, true);
    auto mean1 = ::mlx::core::multiply(sum_gx_kept, inv_N);
    auto mean2 = ::mlx::core::multiply(sum_gx_xn_kept, inv_N);
    auto dx_g = normalized_input_grad(gx_scaled_g, xnorm_g, mean1, mean2, rstd_g);
    auto dx = ::mlx::core::reshape(dx_g, gpu::to_mlx_shape(x_shape));
    // 3.4 backward sweep: same as BatchNorm / LayerNorm.
    return {Storage{gpu::wrap_mlx_array(std::move(dx), dt)},
            Storage{gpu::wrap_mlx_array(std::move(dgamma), dt)},
            Storage{gpu::wrap_mlx_array(std::move(dbeta), dt)}};
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
                                                          Dtype dt) {
    const auto xw = widen_gpu(x), mw = widen_gpu(mean), gw = widen_gpu(gamma);
    const auto rw = widen_gpu(rstd), gow = widen_gpu(grad_out);
    ::mlx::core::Shape b_shape(x_shape.size(), 1);
    b_shape[1] = C;
    auto m_b = ::mlx::core::reshape(mw, b_shape);
    auto r_b = ::mlx::core::reshape(rw, b_shape);
    auto g_b = ::mlx::core::reshape(gw, b_shape);
    auto x_minus_m = ::mlx::core::subtract(xw, m_b);
    auto dx = ::mlx::core::multiply(::mlx::core::multiply(g_b, r_b), gow);
    std::vector<int> reduce_axes;
    for (std::size_t i = 0; i < x_shape.size(); ++i)
        if (i != 1)
            reduce_axes.push_back(static_cast<int>(i));
    auto sum_g = ::mlx::core::sum(gow, reduce_axes, false);
    auto sum_xm_g = ::mlx::core::sum(::mlx::core::multiply(x_minus_m, gow), reduce_axes, false);
    auto dg = ::mlx::core::multiply(sum_xm_g, rw);
    auto dm = ::mlx::core::negative(::mlx::core::multiply(gw, ::mlx::core::multiply(rw, sum_g)));
    auto half = gpu::mlx_scalar(-0.5, xw.dtype());
    auto r3 = ::mlx::core::multiply(rw, ::mlx::core::multiply(rw, rw));
    auto dv =
        ::mlx::core::multiply(half, ::mlx::core::multiply(gw, ::mlx::core::multiply(r3, sum_xm_g)));
    return {wrap_narrow(dx, dt), wrap_narrow(dm, dt), wrap_narrow(dv, dt), wrap_narrow(dg, dt),
            wrap_narrow(sum_g, dt)};
}

std::vector<Storage> GpuBackend::lp_normalize_forward(
    const Storage& x, const Shape& x_shape, double ord, int axis, double eps, Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    // The p-norm is taken in float32 for a half input (see
    // ``widen_half``) — a float16 |x|^2 overflows past 256 and the norm
    // came back inf, normalising the row to zeros — and rounded back.
    auto abs_w = widen_half(::mlx::core::abs(*gx.arr));
    auto sum_p = ::mlx::core::sum(gpu_pow(abs_w, ord), std::vector<int>{axis}, true);
    auto N = narrow_half(gpu_pow(sum_p, 1.0 / ord), mlx_dt);
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
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gn = std::get<GpuStorage>(saved_norm);
    const auto& gg = std::get<GpuStorage>(grad_out);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    auto proj = narrow_half(
        ::mlx::core::sum(product_widened(*gg.arr, *gx.arr), std::vector<int>{axis}, true), mlx_dt);
    auto first = ::mlx::core::divide(*gg.arr, *gn.arr);
    auto sign_x = ::mlx::core::sign(*gx.arr);
    auto abs_x = ::mlx::core::abs(*gx.arr);
    auto abs_pm1 = gpu_pow(abs_x, ord - 1.0);
    auto N_pp1 = gpu_pow(*gn.arr, ord + 1.0);
    auto second = ::mlx::core::divide(
        ::mlx::core::multiply(::mlx::core::multiply(sign_x, abs_pm1), proj), N_pp1);
    auto dx = ::mlx::core::subtract(first, second);
    return Storage{gpu::wrap_mlx_array(std::move(dx), dt)};
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
                                                               Dtype dt) {
    const int B = static_cast<int>(x_shape[0]), C = static_cast<int>(x_shape[1]);
    const auto xw = widen_gpu(x), gw = widen_gpu(grad_out);
    const auto acc_dt = xw.dtype();
    auto G =
        ::mlx::core::sqrt(::mlx::core::sum(::mlx::core::square(xw), std::vector<int>{2, 3}, true));
    auto m_b = ::mlx::core::mean(G, std::vector<int>{1}, true);
    auto denom = ::mlx::core::add(m_b, gpu::mlx_scalar(eps, acc_dt));
    auto Nx = ::mlx::core::reshape(widen_gpu(saved_Nx), {B, C, 1, 1});
    auto g_b = ::mlx::core::reshape(widen_gpu(gamma), {1, C, 1, 1});
    auto gx_prod = ::mlx::core::multiply(gw, xw);
    auto db = ::mlx::core::sum(gx_prod, std::vector<int>{0, 2, 3}, false);
    auto gxN = ::mlx::core::multiply(gx_prod, Nx);
    auto dg = ::mlx::core::sum(gxN, std::vector<int>{0, 2, 3}, false);
    auto sum_gx = ::mlx::core::sum(gx_prod, std::vector<int>{2, 3}, true);
    auto A = ::mlx::core::multiply(g_b, sum_gx);
    auto AG = ::mlx::core::multiply(A, G);
    auto inner_sum = ::mlx::core::sum(AG, std::vector<int>{1}, true);
    auto C_arr = gpu::mlx_scalar(static_cast<double>(C), acc_dt);
    auto denom_sq = ::mlx::core::multiply(denom, denom);
    auto second = ::mlx::core::divide(inner_sum, ::mlx::core::multiply(denom_sq, C_arr));
    auto dG = ::mlx::core::subtract(::mlx::core::divide(A, denom), second);
    auto eps_g = gpu::mlx_scalar(1e-12, acc_dt);
    auto G_safe = ::mlx::core::maximum(G, eps_g);
    auto dG_term = ::mlx::core::divide(::mlx::core::multiply(dG, xw), G_safe);
    auto bb_b = ::mlx::core::reshape(widen_gpu(beta), {1, C, 1, 1});
    auto dx =
        ::mlx::core::add(::mlx::core::add(::mlx::core::multiply(::mlx::core::multiply(g_b, Nx), gw),
                                          ::mlx::core::multiply(bb_b, gw)),
                         dG_term);
    return {wrap_narrow(dx, dt), wrap_narrow(dg, dt), wrap_narrow(db, dt)};
}

}  // namespace backend
}  // namespace lucid
