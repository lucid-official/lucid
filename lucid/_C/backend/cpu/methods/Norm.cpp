// lucid/_C/backend/cpu/methods/Norm.cpp
// Out-of-line CpuBackend method definitions for the Norm family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

StoragePair CpuBackend::rms_norm_forward(const Storage& x,
                                         const Storage& gamma,
                                         std::size_t outer,
                                         std::size_t normalized_size,
                                         double eps,
                                         const Shape& p5_,
                                         Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(rms_norm_forward(detail::as_f32(x), detail::as_f32(gamma), outer,
                                                    normalized_size, eps, p5_, Dtype::F32),
                                   dt);
    const std::size_t y_nbytes = outer * normalized_size * dtype_size(dt);
    const std::size_t rstd_nbytes = outer * dtype_size(dt);
    auto y_ptr = allocate_aligned_bytes(y_nbytes, Device::CPU);
    auto rstd_ptr = allocate_aligned_bytes(rstd_nbytes, Device::CPU);
    const auto& x_cpu = std::get<CpuStorage>(x);
    const auto& g_cpu = std::get<CpuStorage>(gamma);

    if (outer * normalized_size > 0) {
        if (dt == Dtype::F32) {
            cpu::rms_norm_forward_f32(reinterpret_cast<const float*>(x_cpu.ptr.get()),
                                      reinterpret_cast<const float*>(g_cpu.ptr.get()),
                                      reinterpret_cast<float*>(y_ptr.get()),
                                      reinterpret_cast<float*>(rstd_ptr.get()), outer,
                                      normalized_size, eps);
        } else if (dt == Dtype::F64) {
            cpu::rms_norm_forward_f64(reinterpret_cast<const double*>(x_cpu.ptr.get()),
                                      reinterpret_cast<const double*>(g_cpu.ptr.get()),
                                      reinterpret_cast<double*>(y_ptr.get()),
                                      reinterpret_cast<double*>(rstd_ptr.get()), outer,
                                      normalized_size, eps);
        } else {
            ErrorBuilder("cpu_backend::rms_norm_forward").not_implemented("dtype not supported");
        }
    }
    return {Storage{CpuStorage{y_ptr, y_nbytes, dt}},
            Storage{CpuStorage{rstd_ptr, rstd_nbytes, dt}}};
}

StoragePair CpuBackend::rms_norm_backward(const Storage& x,
                                          const Storage& gamma,
                                          const Storage& saved_rstd,
                                          const Storage& grad,
                                          std::size_t outer,
                                          std::size_t normalized_size,
                                          const Shape& p6_,
                                          const Shape& p7_,
                                          Dtype dt,
                                          GradMask2 grad_mask) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(rms_norm_backward(detail::as_f32(x), detail::as_f32(gamma),
                                                     detail::as_f32(saved_rstd),
                                                     detail::as_f32(grad), outer, normalized_size,
                                                     p6_, p7_, Dtype::F32, grad_mask),
                                   dt);
    cpu::GradBuffers<2> grads(grad_mask, dt, {outer * normalized_size, normalized_size});

    if (grad_mask.any() && outer * normalized_size > 0) {
        const auto& x_cpu = std::get<CpuStorage>(x);
        const auto& gamma_cpu = std::get<CpuStorage>(gamma);
        const auto& rstd_cpu = std::get<CpuStorage>(saved_rstd);
        const auto& grad_cpu = std::get<CpuStorage>(grad);
        if (dt == Dtype::F32) {
            cpu::rms_norm_backward_f32(reinterpret_cast<const float*>(x_cpu.ptr.get()),
                                       reinterpret_cast<const float*>(gamma_cpu.ptr.get()),
                                       reinterpret_cast<const float*>(rstd_cpu.ptr.get()),
                                       reinterpret_cast<const float*>(grad_cpu.ptr.get()),
                                       grads.data<float>(0), grads.data<float>(1), outer,
                                       normalized_size);
        } else if (dt == Dtype::F64) {
            cpu::rms_norm_backward_f64(reinterpret_cast<const double*>(x_cpu.ptr.get()),
                                       reinterpret_cast<const double*>(gamma_cpu.ptr.get()),
                                       reinterpret_cast<const double*>(rstd_cpu.ptr.get()),
                                       reinterpret_cast<const double*>(grad_cpu.ptr.get()),
                                       grads.data<double>(0), grads.data<double>(1), outer,
                                       normalized_size);
        } else {
            ErrorBuilder("cpu_backend::rms_norm_backward").not_implemented("dtype not supported");
        }
    } else {
        grads.zero();
    }
    return {grads.take(0), grads.take(1)};
}

std::vector<Storage> CpuBackend::layer_norm_forward(const Storage& x,
                                                    const Storage& gamma,
                                                    const Storage& beta,
                                                    std::size_t outer,
                                                    std::size_t normalized_size,
                                                    double eps,
                                                    const Shape& p6_,
                                                    Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(layer_norm_forward(detail::as_f32(x), detail::as_f32(gamma),
                                                      detail::as_f32(beta), outer, normalized_size,
                                                      eps, p6_, Dtype::F32),
                                   dt);
    const std::size_t y_nbytes = outer * normalized_size * dtype_size(dt);
    const std::size_t saved_nbytes = outer * dtype_size(dt);
    auto y_ptr = allocate_aligned_bytes(y_nbytes, Device::CPU);
    auto mean_ptr = allocate_aligned_bytes(saved_nbytes, Device::CPU);
    auto rstd_ptr = allocate_aligned_bytes(saved_nbytes, Device::CPU);
    const auto& x_cpu = std::get<CpuStorage>(x);
    const auto& g_cpu = std::get<CpuStorage>(gamma);
    const auto& b_cpu = std::get<CpuStorage>(beta);

    if (outer * normalized_size > 0) {
        if (dt == Dtype::F32) {
            cpu::layer_norm_forward_f32(
                reinterpret_cast<const float*>(x_cpu.ptr.get()),
                reinterpret_cast<const float*>(g_cpu.ptr.get()),
                reinterpret_cast<const float*>(b_cpu.ptr.get()),
                reinterpret_cast<float*>(y_ptr.get()), reinterpret_cast<float*>(mean_ptr.get()),
                reinterpret_cast<float*>(rstd_ptr.get()), outer, normalized_size, eps);
        } else if (dt == Dtype::F64) {
            cpu::layer_norm_forward_f64(
                reinterpret_cast<const double*>(x_cpu.ptr.get()),
                reinterpret_cast<const double*>(g_cpu.ptr.get()),
                reinterpret_cast<const double*>(b_cpu.ptr.get()),
                reinterpret_cast<double*>(y_ptr.get()), reinterpret_cast<double*>(mean_ptr.get()),
                reinterpret_cast<double*>(rstd_ptr.get()), outer, normalized_size, eps);
        } else {
            ErrorBuilder("cpu_backend::layer_norm_forward").not_implemented("dtype not supported");
        }
    }
    return {Storage{CpuStorage{y_ptr, y_nbytes, dt}},
            Storage{CpuStorage{mean_ptr, saved_nbytes, dt}},
            Storage{CpuStorage{rstd_ptr, saved_nbytes, dt}}};
}

std::vector<Storage> CpuBackend::layer_norm_backward(const Storage& x,
                                                     const Storage& gamma,
                                                     const Storage& saved_mean,
                                                     const Storage& saved_rstd,
                                                     const Storage& grad,
                                                     std::size_t outer,
                                                     std::size_t normalized_size,
                                                     const Shape& p7_,
                                                     const Shape& p8_,
                                                     const Shape& p9_,
                                                     Dtype dt,
                                                     GradMask3 grad_mask) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(layer_norm_backward(detail::as_f32(x), detail::as_f32(gamma),
                                                       detail::as_f32(saved_mean),
                                                       detail::as_f32(saved_rstd),
                                                       detail::as_f32(grad), outer, normalized_size,
                                                       p7_, p8_, p9_, Dtype::F32, grad_mask),
                                   dt);
    cpu::GradBuffers<3> grads(grad_mask, dt,
                              {outer * normalized_size, normalized_size, normalized_size});

    if (grad_mask.any() && outer * normalized_size > 0) {
        const auto& x_cpu = std::get<CpuStorage>(x);
        const auto& gamma_cpu = std::get<CpuStorage>(gamma);
        const auto& mean_cpu = std::get<CpuStorage>(saved_mean);
        const auto& rstd_cpu = std::get<CpuStorage>(saved_rstd);
        const auto& g_cpu = std::get<CpuStorage>(grad);
        if (dt == Dtype::F32) {
            cpu::layer_norm_backward_f32(reinterpret_cast<const float*>(x_cpu.ptr.get()),
                                         reinterpret_cast<const float*>(gamma_cpu.ptr.get()),
                                         reinterpret_cast<const float*>(mean_cpu.ptr.get()),
                                         reinterpret_cast<const float*>(rstd_cpu.ptr.get()),
                                         reinterpret_cast<const float*>(g_cpu.ptr.get()),
                                         grads.data<float>(0), grads.data<float>(1),
                                         grads.data<float>(2), outer, normalized_size);
        } else if (dt == Dtype::F64) {
            cpu::layer_norm_backward_f64(reinterpret_cast<const double*>(x_cpu.ptr.get()),
                                         reinterpret_cast<const double*>(gamma_cpu.ptr.get()),
                                         reinterpret_cast<const double*>(mean_cpu.ptr.get()),
                                         reinterpret_cast<const double*>(rstd_cpu.ptr.get()),
                                         reinterpret_cast<const double*>(g_cpu.ptr.get()),
                                         grads.data<double>(0), grads.data<double>(1),
                                         grads.data<double>(2), outer, normalized_size);
        } else {
            ErrorBuilder("cpu_backend::layer_norm_backward").not_implemented("dtype not supported");
        }
    } else {
        grads.zero();
    }
    return grads.take_all();
}

std::vector<Storage> CpuBackend::batch_norm_forward(const Storage& x,
                                                    const Storage& gamma,
                                                    const Storage& beta,
                                                    int batch,
                                                    int channels,
                                                    int spatial,
                                                    int p6_,
                                                    double eps,
                                                    const Shape& p8_,
                                                    Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(batch_norm_forward(detail::as_f32(x), detail::as_f32(gamma),
                                                      detail::as_f32(beta), batch, channels,
                                                      spatial, p6_, eps, p8_, Dtype::F32),
                                   dt);
    const std::size_t total = static_cast<std::size_t>(batch) * channels * spatial;
    auto y_ptr = allocate_aligned_bytes(total * dtype_size(dt), Device::CPU);
    auto mean_ptr =
        allocate_aligned_bytes(static_cast<std::size_t>(channels) * dtype_size(dt), Device::CPU);
    auto rstd_ptr =
        allocate_aligned_bytes(static_cast<std::size_t>(channels) * dtype_size(dt), Device::CPU);
    if (total > 0) {
        const auto& x_cpu = std::get<CpuStorage>(x);
        const auto& g_cpu = std::get<CpuStorage>(gamma);
        const auto& b_cpu = std::get<CpuStorage>(beta);
        if (dt == Dtype::F32) {
            batch_norm_forward_f32_fast(
                reinterpret_cast<const float*>(x_cpu.ptr.get()),
                reinterpret_cast<const float*>(g_cpu.ptr.get()),
                reinterpret_cast<const float*>(b_cpu.ptr.get()),
                reinterpret_cast<float*>(y_ptr.get()), reinterpret_cast<float*>(mean_ptr.get()),
                reinterpret_cast<float*>(rstd_ptr.get()), batch, channels, spatial, eps);
        } else if (dt == Dtype::F64) {
            batch_norm_forward_typed<double>(
                reinterpret_cast<const double*>(x_cpu.ptr.get()),
                reinterpret_cast<const double*>(g_cpu.ptr.get()),
                reinterpret_cast<const double*>(b_cpu.ptr.get()),
                reinterpret_cast<double*>(y_ptr.get()), reinterpret_cast<double*>(mean_ptr.get()),
                reinterpret_cast<double*>(rstd_ptr.get()), batch, channels, spatial, eps);
        } else {
            ErrorBuilder("cpu_backend::batch_norm_forward").not_implemented("dtype not supported");
        }
    }
    const std::size_t param_nbytes = static_cast<std::size_t>(channels) * dtype_size(dt);
    // 3.4+ Phase A.4: 4th slot is empty — CPU backward recomputes
    // xnorm inline (per-element loop cost is negligible).  Only the
    // MLX-path GPU backend benefits from saving the full xnorm.
    return {Storage{CpuStorage{y_ptr, total * dtype_size(dt), dt}},
            Storage{CpuStorage{mean_ptr, param_nbytes, dt}},
            Storage{CpuStorage{rstd_ptr, param_nbytes, dt}}, Storage{CpuStorage{}}};
}

std::vector<Storage> CpuBackend::batch_norm_backward(const Storage& x,
                                                     const Storage& gamma,
                                                     const Storage& saved_mean,
                                                     const Storage& saved_rstd,
                                                     const Storage& saved_xnorm,
                                                     const Storage& grad,
                                                     int batch,
                                                     int channels,
                                                     int spatial,
                                                     int p9_,
                                                     const Shape& p10_,
                                                     Dtype dt,
                                                     double eps_unused,
                                                     GradMask3 grad_mask) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            batch_norm_backward(detail::as_f32(x), detail::as_f32(gamma),
                                detail::as_f32(saved_mean), detail::as_f32(saved_rstd),
                                detail::as_f32(saved_xnorm), detail::as_f32(grad), batch, channels,
                                spatial, p9_, p10_, Dtype::F32, eps_unused, grad_mask),
            dt);
    // CPU backward uses saved_rstd directly via the chain-rule formula;
    // no need for eps reconstruction.  The signature carries eps for
    // GPU backend symmetry.
    // 3.4+ Phase A.4: CPU path ignores saved_xnorm — the typed C kernel
    // recomputes inline since the inner loop's per-element xnorm is free.
    (void)eps_unused;
    (void)saved_xnorm;
    const std::size_t total = static_cast<std::size_t>(batch) * channels * spatial;
    cpu::GradBuffers<3> grads(
        grad_mask, dt,
        {total, static_cast<std::size_t>(channels), static_cast<std::size_t>(channels)});
    if (grad_mask.any() && total > 0) {
        const auto& x_cpu = std::get<CpuStorage>(x);
        const auto& gamma_cpu = std::get<CpuStorage>(gamma);
        const auto& mean_cpu = std::get<CpuStorage>(saved_mean);
        const auto& rstd_cpu = std::get<CpuStorage>(saved_rstd);
        const auto& g_cpu = std::get<CpuStorage>(grad);
        const auto run = [&](auto tag) {
            using T = decltype(tag);
            batch_norm_backward_typed<T>(reinterpret_cast<const T*>(x_cpu.ptr.get()),
                                         reinterpret_cast<const T*>(gamma_cpu.ptr.get()),
                                         reinterpret_cast<const T*>(mean_cpu.ptr.get()),
                                         reinterpret_cast<const T*>(rstd_cpu.ptr.get()),
                                         reinterpret_cast<const T*>(g_cpu.ptr.get()),
                                         grads.data<T>(0), grads.data<T>(1), grads.data<T>(2),
                                         batch, channels, spatial);
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu_backend::batch_norm_backward").not_implemented("dtype not supported");
    } else {
        grads.zero();
    }
    return grads.take_all();
}

std::vector<Storage> CpuBackend::group_norm_forward(const Storage& x,
                                                    const Storage& gamma,
                                                    const Storage& beta,
                                                    int batch,
                                                    int channels,
                                                    int spatial,
                                                    int groups,
                                                    const std::vector<int>& p7_,
                                                    double eps,
                                                    const Shape& p9_,
                                                    Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(group_norm_forward(detail::as_f32(x), detail::as_f32(gamma),
                                                      detail::as_f32(beta), batch, channels,
                                                      spatial, groups, p7_, eps, p9_, Dtype::F32),
                                   dt);
    const std::size_t total = static_cast<std::size_t>(batch) * channels * spatial;
    const std::size_t saved_numel = static_cast<std::size_t>(batch) * groups;
    auto y_ptr = allocate_aligned_bytes(total * dtype_size(dt), Device::CPU);
    auto mean_ptr = allocate_aligned_bytes(saved_numel * dtype_size(dt), Device::CPU);
    auto rstd_ptr = allocate_aligned_bytes(saved_numel * dtype_size(dt), Device::CPU);
    if (total > 0) {
        const auto& x_cpu = std::get<CpuStorage>(x);
        const auto& g_cpu = std::get<CpuStorage>(gamma);
        const auto& b_cpu = std::get<CpuStorage>(beta);
        if (dt == Dtype::F32) {
            group_norm_forward_typed<float>(
                reinterpret_cast<const float*>(x_cpu.ptr.get()),
                reinterpret_cast<const float*>(g_cpu.ptr.get()),
                reinterpret_cast<const float*>(b_cpu.ptr.get()),
                reinterpret_cast<float*>(y_ptr.get()), reinterpret_cast<float*>(mean_ptr.get()),
                reinterpret_cast<float*>(rstd_ptr.get()), batch, channels, spatial, groups, eps);
        } else if (dt == Dtype::F64) {
            group_norm_forward_typed<double>(
                reinterpret_cast<const double*>(x_cpu.ptr.get()),
                reinterpret_cast<const double*>(g_cpu.ptr.get()),
                reinterpret_cast<const double*>(b_cpu.ptr.get()),
                reinterpret_cast<double*>(y_ptr.get()), reinterpret_cast<double*>(mean_ptr.get()),
                reinterpret_cast<double*>(rstd_ptr.get()), batch, channels, spatial, groups, eps);
        } else {
            ErrorBuilder("cpu_backend::group_norm_forward").not_implemented("dtype not supported");
        }
    }
    return {Storage{CpuStorage{y_ptr, total * dtype_size(dt), dt}},
            Storage{CpuStorage{mean_ptr, saved_numel * dtype_size(dt), dt}},
            Storage{CpuStorage{rstd_ptr, saved_numel * dtype_size(dt), dt}}};
}

std::vector<Storage> CpuBackend::group_norm_backward(const Storage& x,
                                                     const Storage& gamma,
                                                     const Storage& saved_mean,
                                                     const Storage& saved_rstd,
                                                     const Storage& grad,
                                                     int batch,
                                                     int channels,
                                                     int spatial,
                                                     int groups,
                                                     const std::vector<int>& p9_,
                                                     const Shape& p10_,
                                                     Dtype dt,
                                                     GradMask3 grad_mask) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            group_norm_backward(detail::as_f32(x), detail::as_f32(gamma),
                                detail::as_f32(saved_mean), detail::as_f32(saved_rstd),
                                detail::as_f32(grad), batch, channels, spatial, groups, p9_, p10_,
                                Dtype::F32, grad_mask),
            dt);
    const std::size_t total = static_cast<std::size_t>(batch) * channels * spatial;
    cpu::GradBuffers<3> grads(
        grad_mask, dt,
        {total, static_cast<std::size_t>(channels), static_cast<std::size_t>(channels)});
    if (grad_mask.any() && total > 0) {
        const auto& x_cpu = std::get<CpuStorage>(x);
        const auto& gamma_cpu = std::get<CpuStorage>(gamma);
        const auto& mean_cpu = std::get<CpuStorage>(saved_mean);
        const auto& rstd_cpu = std::get<CpuStorage>(saved_rstd);
        const auto& g_cpu = std::get<CpuStorage>(grad);
        const auto run = [&](auto tag) {
            using T = decltype(tag);
            group_norm_backward_typed<T>(reinterpret_cast<const T*>(x_cpu.ptr.get()),
                                         reinterpret_cast<const T*>(gamma_cpu.ptr.get()),
                                         reinterpret_cast<const T*>(mean_cpu.ptr.get()),
                                         reinterpret_cast<const T*>(rstd_cpu.ptr.get()),
                                         reinterpret_cast<const T*>(g_cpu.ptr.get()),
                                         grads.data<T>(0), grads.data<T>(1), grads.data<T>(2),
                                         batch, channels, spatial, groups);
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu_backend::group_norm_backward").not_implemented("dtype not supported");
    } else {
        grads.zero();
    }
    return grads.take_all();
}

std::vector<Storage> CpuBackend::batch_norm_eval_forward(const Storage& x,
                                                         const Storage& mean,
                                                         const Storage& var,
                                                         const Storage& gamma,
                                                         const Storage& beta,
                                                         const Shape& x_shape,
                                                         int C,
                                                         int spatial,
                                                         double eps,
                                                         Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            batch_norm_eval_forward(detail::as_f32(x), detail::as_f32(mean), detail::as_f32(var),
                                    detail::as_f32(gamma), detail::as_f32(beta), x_shape, C,
                                    spatial, eps, Dtype::F32),
            dt);
    const int B = static_cast<int>(x_shape[0]);
    auto rstd_cpu = alloc_cpu(static_cast<std::size_t>(C), dt);
    auto out_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * spatial, dt);
    auto run = [&](auto tag) {
        using T = decltype(tag);
        auto in = [](const Storage& s) {
            return reinterpret_cast<const T*>(std::get<CpuStorage>(s).ptr.get());
        };
        cpu::batch_norm_eval_forward_typed(
            in(x), in(mean), in(var), in(gamma), in(beta), reinterpret_cast<T*>(out_cpu.ptr.get()),
            reinterpret_cast<T*>(rstd_cpu.ptr.get()), B, C, spatial, eps);
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("batch_norm_eval_forward").not_implemented("dtype");
    return {Storage{std::move(out_cpu)}, Storage{std::move(rstd_cpu)}};
}

std::vector<Storage> CpuBackend::batch_norm_eval_backward(const Storage& x,
                                                          const Storage& mean,
                                                          const Storage& gamma,
                                                          const Storage& rstd,
                                                          const Storage& grad_out,
                                                          const Shape& x_shape,
                                                          int C,
                                                          int spatial,
                                                          Dtype dt,
                                                          GradMask5 grad_mask) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            batch_norm_eval_backward(detail::as_f32(x), detail::as_f32(mean), detail::as_f32(gamma),
                                     detail::as_f32(rstd), detail::as_f32(grad_out), x_shape, C,
                                     spatial, Dtype::F32, grad_mask),
            dt);
    const int B = static_cast<int>(x_shape[0]);
    const std::size_t channels = static_cast<std::size_t>(C);
    cpu::GradBuffers<5> grads(
        grad_mask, dt,
        {static_cast<std::size_t>(B) * C * spatial, channels, channels, channels, channels});
    auto run = [&](auto tag) {
        using T = decltype(tag);
        auto in = [](const Storage& s) {
            return reinterpret_cast<const T*>(std::get<CpuStorage>(s).ptr.get());
        };
        cpu::batch_norm_eval_backward_typed(in(x), in(mean), in(gamma), in(rstd), in(grad_out),
                                            grads.data<T>(0), grads.data<T>(1), grads.data<T>(2),
                                            grads.data<T>(3), grads.data<T>(4), B, C, spatial);
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("batch_norm_eval_backward").not_implemented("dtype");
    return grads.take_all();
}

std::vector<Storage> CpuBackend::lp_normalize_forward(
    const Storage& x, const Shape& x_shape, double ord, int axis, double eps, Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            lp_normalize_forward(detail::as_f32(x), x_shape, ord, axis, eps, Dtype::F32), dt);
    const auto& xs = std::get<CpuStorage>(x);
    const std::size_t numel = shape_numel(x_shape);
    int outer = 1, inner = 1;
    const int rank = static_cast<int>(x_shape.size());
    for (int i = 0; i < axis; ++i)
        outer *= static_cast<int>(x_shape[i]);
    for (int i = axis + 1; i < rank; ++i)
        inner *= static_cast<int>(x_shape[i]);
    const int axis_len = static_cast<int>(x_shape[axis]);
    auto y_cpu = alloc_cpu(numel, dt);
    auto norm_cpu = alloc_cpu(static_cast<std::size_t>(outer) * inner, dt);
    auto run = [&](auto tag) {
        using T = decltype(tag);
        cpu::lp_normalize_forward_typed(
            reinterpret_cast<const T*>(xs.ptr.get()), reinterpret_cast<T*>(y_cpu.ptr.get()),
            reinterpret_cast<T*>(norm_cpu.ptr.get()), outer, axis_len, inner, ord, eps);
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("lp_normalize_forward").not_implemented("dtype");
    return {Storage{std::move(y_cpu)}, Storage{std::move(norm_cpu)}};
}

Storage CpuBackend::p_norm_forward(const Storage& a,
                                   const Shape& shape,
                                   double p,
                                   const std::vector<int>& axes,
                                   bool keepdims,
                                   Dtype dt) {
    const auto& as = std::get<CpuStorage>(a);
    std::vector<bool> reduced(shape.size(), axes.empty());
    for (int axis : axes)
        reduced[static_cast<std::size_t>(axis < 0 ? axis + static_cast<int>(shape.size()) : axis)] =
            true;
    std::size_t kept = 1;
    for (std::size_t d = 0; d < shape.size(); ++d)
        if (!reduced[d])
            kept *= static_cast<std::size_t>(shape[d]);
    auto out = alloc_cpu(kept, dt);
    auto run = [&](auto tag) {
        using T = decltype(tag);
        cpu::p_norm_over_axes(reinterpret_cast<const T*>(as.ptr.get()),
                              reinterpret_cast<T*>(out.ptr.get()), shape, reduced, p);
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("cpu_backend::p_norm_forward").not_implemented("dtype not supported");
    return Storage{std::move(out)};
}

Storage CpuBackend::lp_normalize_backward(const Storage& x,
                                          const Storage& saved_norm,
                                          const Storage& grad_out,
                                          const Shape& x_shape,
                                          double ord,
                                          int axis,
                                          Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            lp_normalize_backward(detail::as_f32(x), detail::as_f32(saved_norm),
                                  detail::as_f32(grad_out), x_shape, ord, axis, Dtype::F32),
            dt);
    const auto& xs = std::get<CpuStorage>(x);
    const auto& ns = std::get<CpuStorage>(saved_norm);
    const auto& gs = std::get<CpuStorage>(grad_out);
    const std::size_t numel = shape_numel(x_shape);
    int outer = 1, inner = 1;
    const int rank = static_cast<int>(x_shape.size());
    for (int i = 0; i < axis; ++i)
        outer *= static_cast<int>(x_shape[i]);
    for (int i = axis + 1; i < rank; ++i)
        inner *= static_cast<int>(x_shape[i]);
    const int axis_len = static_cast<int>(x_shape[axis]);
    auto dx_cpu = alloc_cpu(numel, dt);
    auto run = [&](auto tag) {
        using T = decltype(tag);
        cpu::lp_normalize_backward_typed(
            reinterpret_cast<const T*>(xs.ptr.get()), reinterpret_cast<const T*>(ns.ptr.get()),
            reinterpret_cast<const T*>(gs.ptr.get()), reinterpret_cast<T*>(dx_cpu.ptr.get()), outer,
            axis_len, inner, ord);
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("lp_normalize_backward").not_implemented("dtype");
    return Storage{std::move(dx_cpu)};
}

std::vector<Storage> CpuBackend::global_response_norm_forward(const Storage& x,
                                                              const Storage& gamma,
                                                              const Storage& beta,
                                                              const Shape& x_shape,
                                                              double eps,
                                                              Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            global_response_norm_forward(detail::as_f32(x), detail::as_f32(gamma),
                                         detail::as_f32(beta), x_shape, eps, Dtype::F32),
            dt);
    const int B = static_cast<int>(x_shape[0]);
    const int C = static_cast<int>(x_shape[1]);
    const int spatial = static_cast<int>(x_shape[2] * x_shape[3]);
    auto y_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * spatial, dt);
    auto nx_cpu = alloc_cpu(static_cast<std::size_t>(B) * C, dt);
    auto run = [&](auto tag) {
        using T = decltype(tag);
        auto in = [](const Storage& s) {
            return reinterpret_cast<const T*>(std::get<CpuStorage>(s).ptr.get());
        };
        cpu::global_response_norm_forward_typed(
            in(x), in(gamma), in(beta), reinterpret_cast<T*>(y_cpu.ptr.get()),
            reinterpret_cast<T*>(nx_cpu.ptr.get()), B, C, spatial, eps);
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("grn_forward").not_implemented("dtype");
    return {Storage{std::move(y_cpu)}, Storage{std::move(nx_cpu)}};
}

std::vector<Storage> CpuBackend::global_response_norm_backward(const Storage& x,
                                                               const Storage& gamma,
                                                               const Storage& beta,
                                                               const Storage& saved_Nx,
                                                               const Storage& grad_out,
                                                               const Shape& x_shape,
                                                               double eps,
                                                               Dtype dt,
                                                               GradMask3 grad_mask) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            global_response_norm_backward(detail::as_f32(x), detail::as_f32(gamma),
                                          detail::as_f32(beta), detail::as_f32(saved_Nx),
                                          detail::as_f32(grad_out), x_shape, eps, Dtype::F32,
                                          grad_mask),
            dt);
    const int B = static_cast<int>(x_shape[0]);
    const int C = static_cast<int>(x_shape[1]);
    const int spatial = static_cast<int>(x_shape[2] * x_shape[3]);
    const std::size_t channels = static_cast<std::size_t>(C);
    cpu::GradBuffers<3> grads(grad_mask, dt,
                              {static_cast<std::size_t>(B) * C * spatial, channels, channels});
    auto run = [&](auto tag) {
        using T = decltype(tag);
        auto in = [](const Storage& s) {
            return reinterpret_cast<const T*>(std::get<CpuStorage>(s).ptr.get());
        };
        cpu::global_response_norm_backward_typed(in(x), in(gamma), in(beta), in(saved_Nx),
                                                 in(grad_out), grads.data<T>(0), grads.data<T>(1),
                                                 grads.data<T>(2), B, C, spatial, eps);
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("grn_backward").not_implemented("dtype");
    return grads.take_all();
}

void CpuBackend::batch_norm_forward_f32_fast(const float* x,
                                             const float* gamma,
                                             const float* beta,
                                             float* y,
                                             float* mean_per_c,
                                             float* rstd_per_c,
                                             int batch,
                                             int channels,
                                             int spatial,
                                             double eps) {
    // Centred before anything is squared or scaled.  The variance used
    // to be E[x²] − mean² and the output x·scale + (β − mean·scale):
    // both subtract two large, nearly equal numbers once the mean
    // outgrows the spread, and at a mean of 100× the spread the
    // input gradient was off by 1e-3 while Metal's held 1e-5.
    const std::size_t S = static_cast<std::size_t>(spatial);
    const float inv_M = 1.0f / static_cast<float>(batch * spatial);
    for (int c = 0; c < channels; ++c) {
        float sum = 0.f;
        for (int b = 0; b < batch; ++b)
            sum += cpu::vsum_f32(x + (static_cast<std::size_t>(b) * channels + c) * S, S);
        const float mean = sum * inv_M;
        mean_per_c[c] = mean;

        // y holds x − mean until the scale is known.
        float sumsq = 0.f;
        for (int b = 0; b < batch; ++b) {
            const std::size_t at = (static_cast<std::size_t>(b) * channels + c) * S;
            cpu::vsadd_f32(x + at, -mean, y + at, S);
            sumsq += cpu::vdotpr_f32(y + at, y + at, S);
        }
        const float var = sumsq * inv_M;
        const float rstd = 1.0f / std::sqrt(var + static_cast<float>(eps));
        rstd_per_c[c] = rstd;

        const float scale = gamma[c] * rstd;
        for (int b = 0; b < batch; ++b) {
            float* yb = y + (static_cast<std::size_t>(b) * channels + c) * S;
            cpu::vsmul_f32(yb, scale, yb, S);
            cpu::vsadd_f32(yb, beta[c], yb, S);
        }
    }
}

template <typename T>
void CpuBackend::batch_norm_forward_typed(const T* x,
                                          const T* gamma,
                                          const T* beta,
                                          T* y,
                                          T* mean_per_c,
                                          T* rstd_per_c,
                                          int batch,
                                          int channels,
                                          int spatial,
                                          double eps) {
    using Acc = NormAcc<T>;
    const std::size_t M = static_cast<std::size_t>(batch) * spatial;
    const Acc inv_M = Acc{1} / static_cast<Acc>(M);
    for (int c = 0; c < channels; ++c) {
        Acc mean_acc = Acc{};
        for (int b = 0; b < batch; ++b) {
            const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
            for (int i = 0; i < spatial; ++i)
                mean_acc += static_cast<Acc>(xb[i]);
        }
        mean_acc *= inv_M;
        const T mean = static_cast<T>(mean_acc);

        Acc var_acc = Acc{};
        for (int b = 0; b < batch; ++b) {
            const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
            for (int i = 0; i < spatial; ++i) {
                const Acc d = static_cast<Acc>(xb[i]) - mean_acc;
                var_acc += d * d;
            }
        }
        var_acc *= inv_M;
        const T rstd = static_cast<T>(Acc{1} / std::sqrt(var_acc + static_cast<Acc>(eps)));
        const T g = gamma[c];
        const T be = beta[c];
        for (int b = 0; b < batch; ++b) {
            const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
            T* yb = y + (static_cast<std::size_t>(b) * channels + c) * spatial;
            for (int i = 0; i < spatial; ++i)
                yb[i] = g * (xb[i] - mean) * rstd + be;
        }
        mean_per_c[c] = mean;
        rstd_per_c[c] = rstd;
    }
}

template <typename T>
void CpuBackend::batch_norm_backward_typed(const T* x,
                                           const T* gamma,
                                           const T* mean,
                                           const T* rstd,
                                           const T* g,
                                           T* dx,
                                           T* dgamma,
                                           T* dbeta,
                                           int batch,
                                           int channels,
                                           int spatial) {
    using Acc = NormAcc<T>;
    const std::size_t M = static_cast<std::size_t>(batch) * spatial;
    const Acc inv_M = Acc{1} / static_cast<Acc>(M);
    const bool need_x = any_set(dx, dgamma);
    for (int c = 0; c < channels; ++c) {
        const T m = at_or_zero(mean, c);
        const T r = at_or_zero(rstd, c);
        const T gc = at_or_zero(gamma, c);
        Acc sum_dxn = Acc{};
        Acc sum_dxn_xn = Acc{};
        Acc sum_g = Acc{};
        Acc sum_g_xn = Acc{};
        for (int b = 0; b < batch; ++b) {
            const T* xb = offset_or_null(x, (static_cast<std::size_t>(b) * channels + c) * spatial);
            const T* gb = g + (static_cast<std::size_t>(b) * channels + c) * spatial;
            for (int i = 0; i < spatial; ++i) {
                if (dbeta)
                    sum_g += static_cast<Acc>(gb[i]);
                if (need_x) {
                    const Acc xn_i = static_cast<Acc>((xb[i] - m) * r);
                    if (dgamma)
                        sum_g_xn += static_cast<Acc>(gb[i]) * xn_i;
                    if (dx) {
                        const Acc dxn_i = static_cast<Acc>(gc * gb[i]);
                        sum_dxn += dxn_i;
                        sum_dxn_xn += dxn_i * xn_i;
                    }
                }
            }
        }
        if (dx) {
            const T sum_dxn_t = static_cast<T>(sum_dxn);
            const T sum_dxn_xn_t = static_cast<T>(sum_dxn_xn);
            const T inv_M_t = static_cast<T>(inv_M);
            for (int b = 0; b < batch; ++b) {
                const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                const T* gb = g + (static_cast<std::size_t>(b) * channels + c) * spatial;
                T* dxb = dx + (static_cast<std::size_t>(b) * channels + c) * spatial;
                for (int i = 0; i < spatial; ++i) {
                    const T xn_i = (xb[i] - m) * r;
                    const T dxn_i = gc * gb[i];
                    dxb[i] =
                        inv_M_t * r * (static_cast<T>(M) * dxn_i - sum_dxn_t - xn_i * sum_dxn_xn_t);
                }
            }
        }
        if (dgamma)
            dgamma[c] = static_cast<T>(sum_g_xn);
        if (dbeta)
            dbeta[c] = static_cast<T>(sum_g);
    }
}

template <typename T>
void CpuBackend::group_norm_forward_typed(const T* x,
                                          const T* gamma,
                                          const T* beta,
                                          T* y,
                                          T* mean_bg,
                                          T* rstd_bg,
                                          int batch,
                                          int channels,
                                          int spatial,
                                          int groups,
                                          double eps) {
    using Acc = NormAcc<T>;
    const int Cg = channels / groups;
    const std::size_t per_group = static_cast<std::size_t>(Cg) * spatial;
    const Acc inv_pg = Acc{1} / static_cast<Acc>(per_group);
    for (int b = 0; b < batch; ++b) {
        for (int g = 0; g < groups; ++g) {
            Acc mean_acc = Acc{};
            for (int cc = 0; cc < Cg; ++cc) {
                const int c = g * Cg + cc;
                const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                for (int i = 0; i < spatial; ++i)
                    mean_acc += static_cast<Acc>(xb[i]);
            }
            mean_acc *= inv_pg;
            const T mean = static_cast<T>(mean_acc);
            Acc var_acc = Acc{};
            for (int cc = 0; cc < Cg; ++cc) {
                const int c = g * Cg + cc;
                const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                for (int i = 0; i < spatial; ++i) {
                    const Acc d = static_cast<Acc>(xb[i]) - mean_acc;
                    var_acc += d * d;
                }
            }
            var_acc *= inv_pg;
            const T rstd = static_cast<T>(Acc{1} / std::sqrt(var_acc + static_cast<Acc>(eps)));
            for (int cc = 0; cc < Cg; ++cc) {
                const int c = g * Cg + cc;
                const T gc = gamma[c];
                const T bc = beta[c];
                const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                T* yb = y + (static_cast<std::size_t>(b) * channels + c) * spatial;
                for (int i = 0; i < spatial; ++i)
                    yb[i] = gc * (xb[i] - mean) * rstd + bc;
            }
            mean_bg[b * groups + g] = mean;
            rstd_bg[b * groups + g] = rstd;
        }
    }
}

template <typename T>
void CpuBackend::group_norm_backward_typed(const T* x,
                                           const T* gamma,
                                           const T* mean_bg,
                                           const T* rstd_bg,
                                           const T* g,
                                           T* dx,
                                           T* dgamma,
                                           T* dbeta,
                                           int batch,
                                           int channels,
                                           int spatial,
                                           int groups) {
    using Acc = NormAcc<T>;
    const int Cg = channels / groups;
    const std::size_t per_group = static_cast<std::size_t>(Cg) * spatial;
    const Acc inv_pg = Acc{1} / static_cast<Acc>(per_group);
    const bool need_x = any_set(dx, dgamma);
    // dgamma / dbeta run over batch * spatial per channel, which is the
    // same length of sum the forward widened; they are accumulated apart
    // and rounded once on the way out.
    std::vector<Acc> dgamma_acc(size_if_set(dgamma, static_cast<std::size_t>(channels)), Acc{});
    std::vector<Acc> dbeta_acc(size_if_set(dbeta, static_cast<std::size_t>(channels)), Acc{});
    for (int b = 0; b < batch; ++b) {
        for (int gi = 0; gi < groups; ++gi) {
            const T m = at_or_zero(mean_bg, static_cast<std::size_t>(b * groups + gi));
            const T r = at_or_zero(rstd_bg, static_cast<std::size_t>(b * groups + gi));
            Acc sum_dxn = Acc{};
            Acc sum_dxn_xn = Acc{};
            for (int cc = 0; cc < Cg; ++cc) {
                const int c = gi * Cg + cc;
                const T gc = at_or_zero(gamma, static_cast<std::size_t>(c));
                const T* xb =
                    offset_or_null(x, (static_cast<std::size_t>(b) * channels + c) * spatial);
                const T* gb = g + (static_cast<std::size_t>(b) * channels + c) * spatial;
                for (int i = 0; i < spatial; ++i) {
                    if (dbeta)
                        dbeta_acc[c] += static_cast<Acc>(gb[i]);
                    if (need_x) {
                        const Acc xn_i = static_cast<Acc>((xb[i] - m) * r);
                        if (dgamma)
                            dgamma_acc[c] += static_cast<Acc>(gb[i]) * xn_i;
                        if (dx) {
                            const Acc dxn_i = static_cast<Acc>(gc * gb[i]);
                            sum_dxn += dxn_i;
                            sum_dxn_xn += dxn_i * xn_i;
                        }
                    }
                }
            }
            if (dx) {
                const T sum_dxn_t = static_cast<T>(sum_dxn);
                const T sum_dxn_xn_t = static_cast<T>(sum_dxn_xn);
                const T inv_pg_t = static_cast<T>(inv_pg);
                for (int cc = 0; cc < Cg; ++cc) {
                    const int c = gi * Cg + cc;
                    const T gc = gamma[c];
                    const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                    const T* gb = g + (static_cast<std::size_t>(b) * channels + c) * spatial;
                    T* dxb = dx + (static_cast<std::size_t>(b) * channels + c) * spatial;
                    for (int i = 0; i < spatial; ++i) {
                        const T xn_i = (xb[i] - m) * r;
                        const T dxn_i = gc * gb[i];
                        dxb[i] =
                            inv_pg_t * r *
                            (static_cast<T>(per_group) * dxn_i - sum_dxn_t - xn_i * sum_dxn_xn_t);
                    }
                }
            }
        }
    }
    store_narrowed(dgamma, dgamma_acc);
    store_narrowed(dbeta, dbeta_acc);
}

}  // namespace backend
}  // namespace lucid
