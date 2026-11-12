// lucid/_C/backend/gpu/methods/Loss.cpp
// Out-of-line GpuBackend method definitions for the Loss family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

Storage GpuBackend::ctc_loss_forward(const Storage& log_probs,
                                     const Storage& targets,
                                     const Storage& input_lengths,
                                     const Storage& target_lengths,
                                     const Shape& lp_shape,
                                     int blank,
                                     bool zero_infinity,
                                     Dtype dt) {
    // Download all tensors to CPU for the DP algorithm.
    auto dl = [](const Storage& s) -> CpuStorage {
        const auto& g = std::get<GpuStorage>(s);
        auto c = ::mlx::core::contiguous(*g.arr);
        c.eval();
        auto p = allocate_aligned_bytes(c.nbytes(), Device::CPU);
        MemoryTracker::track_host_sync(c.nbytes());
        std::memcpy(p.get(), c.data<void>(), c.nbytes());
        return CpuStorage{p, c.nbytes(), Dtype::F32};  // dtype set by caller
    };
    auto lp_cpu = dl(log_probs);
    auto tgt_cpu = dl(targets);
    auto il_cpu = dl(input_lengths);
    auto tl_cpu = dl(target_lengths);
    lp_cpu.dtype = dt;
    tgt_cpu.dtype = Dtype::I32;
    il_cpu.dtype = Dtype::I32;
    tl_cpu.dtype = Dtype::I32;
    auto result = Dispatcher::for_device(Device::CPU)
                      .ctc_loss_forward(Storage{lp_cpu}, Storage{tgt_cpu}, Storage{il_cpu},
                                        Storage{tl_cpu}, lp_shape, blank, zero_infinity, dt);
    const auto& res_cpu = std::get<CpuStorage>(result);
    const int N = static_cast<int>(lp_shape[1]);
    Shape out_shape{static_cast<std::int64_t>(N)};
    return Storage{gpu::upload_cpu_to_gpu(res_cpu, out_shape)};
}

Storage GpuBackend::ctc_loss_backward(const Storage& log_probs,
                                      const Storage& targets,
                                      const Storage& input_lengths,
                                      const Storage& target_lengths,
                                      const Storage& grad_out,
                                      const Shape& lp_shape,
                                      int blank,
                                      bool zero_infinity,
                                      Dtype dt) {
    auto dl = [](const Storage& s, Dtype as) -> CpuStorage {
        const auto& g = std::get<GpuStorage>(s);
        auto c = ::mlx::core::contiguous(*g.arr);
        c.eval();
        auto p = allocate_aligned_bytes(c.nbytes(), Device::CPU);
        MemoryTracker::track_host_sync(c.nbytes());
        std::memcpy(p.get(), c.data<void>(), c.nbytes());
        return CpuStorage{p, c.nbytes(), as};
    };
    auto grad =
        Dispatcher::for_device(Device::CPU)
            .ctc_loss_backward(Storage{dl(log_probs, dt)}, Storage{dl(targets, Dtype::I32)},
                               Storage{dl(input_lengths, Dtype::I32)},
                               Storage{dl(target_lengths, Dtype::I32)}, Storage{dl(grad_out, dt)},
                               lp_shape, blank, zero_infinity, dt);
    return Storage{gpu::upload_cpu_to_gpu(std::get<CpuStorage>(grad), lp_shape)};
}

Storage GpuBackend::mse_loss(
    const Storage& input, const Storage& target, const Shape&, Dtype dt, int reduction) {
    const auto& x = std::get<GpuStorage>(input);
    const auto& t = std::get<GpuStorage>(target);
    auto diff = ::mlx::core::subtract(*x.arr, *t.arr);
    auto sq = ::mlx::core::multiply(diff, diff);
    auto reduced = apply_loss_reduction(sq, reduction);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(reduced), dt)};
}

std::pair<Storage, Storage> GpuBackend::mse_loss_backward(const Storage& input,
                                                          const Storage& target,
                                                          const Storage& grad,
                                                          const Shape& shape,
                                                          Dtype dt,
                                                          int reduction) {
    const auto& x = std::get<GpuStorage>(input);
    const auto& t = std::get<GpuStorage>(target);
    const auto& g = std::get<GpuStorage>(grad);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    auto scaled = scale_loss_grad(*g.arr, reduction, shape_numel(shape), mlx_dt);
    if (reduction != 0)
        scaled = ::mlx::core::broadcast_to(scaled, gpu::to_mlx_shape(shape));
    auto diff = ::mlx::core::subtract(*x.arr, *t.arr);
    auto two = gpu::mlx_scalar(2.0, mlx_dt);
    auto dx = ::mlx::core::multiply(::mlx::core::multiply(two, diff), scaled);
    auto dtarget = ::mlx::core::negative(dx);
    return {
        Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(narrow_to(dx, mlx_dt)), dt)},
        Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(narrow_to(dtarget, mlx_dt)), dt)},
    };
}

Storage GpuBackend::huber_loss(const Storage& input,
                               const Storage& target,
                               const Shape&,
                               Dtype dt,
                               double delta,
                               int reduction) {
    const auto& x = std::get<GpuStorage>(input);
    const auto& t = std::get<GpuStorage>(target);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    auto d = gpu::mlx_scalar(delta, mlx_dt);
    auto half_d_sq = gpu::mlx_scalar(0.5 * delta * delta, mlx_dt);
    auto half = gpu::mlx_scalar(0.5, mlx_dt);
    auto r = ::mlx::core::subtract(*x.arr, *t.arr);
    auto ar = ::mlx::core::abs(r);
    auto sq_term = ::mlx::core::multiply(half, ::mlx::core::multiply(r, r));
    auto lin_term = ::mlx::core::subtract(::mlx::core::multiply(d, ar), half_d_sq);
    auto cond = ::mlx::core::less(ar, d);
    auto values = ::mlx::core::where(cond, sq_term, lin_term);
    auto reduced = apply_loss_reduction(values, reduction);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(reduced), dt)};
}

std::pair<Storage, Storage> GpuBackend::huber_loss_backward(const Storage& input,
                                                            const Storage& target,
                                                            const Storage& grad,
                                                            const Shape& shape,
                                                            Dtype dt,
                                                            double delta,
                                                            int reduction) {
    const auto& x = std::get<GpuStorage>(input);
    const auto& t = std::get<GpuStorage>(target);
    const auto& g = std::get<GpuStorage>(grad);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    auto d = gpu::mlx_scalar(delta, mlx_dt);
    auto neg_d = gpu::mlx_scalar(-delta, mlx_dt);
    auto r = ::mlx::core::subtract(*x.arr, *t.arr);
    auto ar = ::mlx::core::abs(r);
    auto cond = ::mlx::core::less(ar, d);
    auto sgn_d =
        ::mlx::core::where(::mlx::core::greater(r, gpu::mlx_scalar(0.0, mlx_dt)), d, neg_d);
    auto dr = ::mlx::core::where(cond, r, sgn_d);
    auto scaled = scale_loss_grad(*g.arr, reduction, shape_numel(shape), mlx_dt);
    if (reduction != 0)
        scaled = ::mlx::core::broadcast_to(scaled, gpu::to_mlx_shape(shape));
    auto dx = ::mlx::core::multiply(dr, scaled);
    auto dtarget = ::mlx::core::negative(dx);
    return {
        Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(narrow_to(dx, mlx_dt)), dt)},
        Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(narrow_to(dtarget, mlx_dt)), dt)},
    };
}

Storage GpuBackend::bce_loss(const Storage& input,
                             const Storage& target,
                             const Storage& weight,
                             const Shape&,
                             Dtype dt,
                             double eps,
                             int reduction) {
    const auto& x = std::get<GpuStorage>(input);
    const auto& t = std::get<GpuStorage>(target);
    const auto& w = std::get<GpuStorage>(weight);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    auto e_lo = gpu::mlx_scalar(eps, mlx_dt);
    auto one = gpu::mlx_scalar(1.0, mlx_dt);
    auto e_hi = ::mlx::core::subtract(one, e_lo);
    auto p = ::mlx::core::clip(*x.arr, std::optional<::mlx::core::array>(e_lo),
                               std::optional<::mlx::core::array>(e_hi));
    auto one_mt = ::mlx::core::subtract(one, *t.arr);
    auto one_mp = ::mlx::core::subtract(one, p);
    auto term1 = ::mlx::core::multiply(*t.arr, ::mlx::core::log(p));
    auto term2 = ::mlx::core::multiply(one_mt, ::mlx::core::log(one_mp));
    auto values =
        ::mlx::core::multiply(*w.arr, ::mlx::core::negative(::mlx::core::add(term1, term2)));
    auto reduced = apply_loss_reduction(values, reduction);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(reduced), dt)};
}

std::vector<Storage> GpuBackend::bce_loss_backward(const Storage& input,
                                                   const Storage& target,
                                                   const Storage& weight,
                                                   const Storage& grad,
                                                   const Shape& shape,
                                                   Dtype dt,
                                                   double eps,
                                                   int reduction) {
    const auto& x = std::get<GpuStorage>(input);
    const auto& t = std::get<GpuStorage>(target);
    const auto& w = std::get<GpuStorage>(weight);
    const auto& g = std::get<GpuStorage>(grad);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    auto e_lo = gpu::mlx_scalar(eps, mlx_dt);
    auto one = gpu::mlx_scalar(1.0, mlx_dt);
    auto e_hi = ::mlx::core::subtract(one, e_lo);
    auto p = ::mlx::core::clip(*x.arr, std::optional<::mlx::core::array>(e_lo),
                               std::optional<::mlx::core::array>(e_hi));
    auto one_mp = ::mlx::core::subtract(one, p);
    auto one_mt = ::mlx::core::subtract(one, *t.arr);
    auto log_p = ::mlx::core::log(p);
    auto log_1mp = ::mlx::core::log(one_mp);
    auto dlp = ::mlx::core::add(::mlx::core::negative(::mlx::core::divide(*t.arr, p)),
                                ::mlx::core::divide(one_mt, one_mp));
    auto dtarget_term = ::mlx::core::add(::mlx::core::negative(log_p), log_1mp);
    auto values = ::mlx::core::negative(::mlx::core::add(::mlx::core::multiply(*t.arr, log_p),
                                                         ::mlx::core::multiply(one_mt, log_1mp)));
    auto scaled = scale_loss_grad(*g.arr, reduction, shape_numel(shape), mlx_dt);
    if (reduction != 0)
        scaled = ::mlx::core::broadcast_to(scaled, gpu::to_mlx_shape(shape));
    auto dx = ::mlx::core::multiply(*w.arr, ::mlx::core::multiply(dlp, scaled));
    auto dtarget = ::mlx::core::multiply(*w.arr, ::mlx::core::multiply(dtarget_term, scaled));
    auto dweight = ::mlx::core::multiply(values, scaled);
    return {
        Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(narrow_to(dx, mlx_dt)), dt)},
        Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(narrow_to(dtarget, mlx_dt)), dt)},
        Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(narrow_to(dweight, mlx_dt)), dt)},
    };
}

Storage GpuBackend::bce_with_logits_loss(const Storage& input,
                                         const Storage& target,
                                         const Storage& weight,
                                         const Storage& pos_weight,
                                         const Shape&,
                                         const Shape&,
                                         const Shape&,
                                         Dtype dt,
                                         int reduction) {
    const auto& x = std::get<GpuStorage>(input);
    const auto& t = std::get<GpuStorage>(target);
    const auto& w = std::get<GpuStorage>(weight);
    const auto& pw = std::get<GpuStorage>(pos_weight);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    auto one = gpu::mlx_scalar(1.0, mlx_dt);
    auto zero = gpu::mlx_scalar(0.0, mlx_dt);
    auto pw_m1 = ::mlx::core::subtract(*pw.arr, one);
    auto log_weight = ::mlx::core::add(::mlx::core::multiply(pw_m1, *t.arr), one);
    auto log1pexp =
        ::mlx::core::log1p(::mlx::core::exp(::mlx::core::negative(::mlx::core::abs(*x.arr))));
    auto max0 = ::mlx::core::maximum(*x.arr, zero);
    auto loss = ::mlx::core::add(::mlx::core::subtract(max0, ::mlx::core::multiply(*x.arr, *t.arr)),
                                 ::mlx::core::multiply(log_weight, log1pexp));
    auto values = ::mlx::core::multiply(*w.arr, loss);
    auto reduced = apply_loss_reduction(values, reduction);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(reduced), dt)};
}

std::vector<Storage> GpuBackend::bce_with_logits_backward(const Storage& input,
                                                          const Storage& target,
                                                          const Storage& weight,
                                                          const Storage& pos_weight,
                                                          const Storage& grad,
                                                          const Shape& shape,
                                                          Dtype dt,
                                                          int reduction) {
    const auto& x = std::get<GpuStorage>(input);
    const auto& t = std::get<GpuStorage>(target);
    const auto& w = std::get<GpuStorage>(weight);
    const auto& pw = std::get<GpuStorage>(pos_weight);
    const auto& g = std::get<GpuStorage>(grad);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    auto one = gpu::mlx_scalar(1.0, mlx_dt);
    auto zero = gpu::mlx_scalar(0.0, mlx_dt);
    auto pw_m1 = ::mlx::core::subtract(*pw.arr, one);
    auto log_weight = ::mlx::core::add(::mlx::core::multiply(pw_m1, *t.arr), one);
    auto sigm = ::mlx::core::sigmoid(*x.arr);
    auto log1pexp =
        ::mlx::core::log1p(::mlx::core::exp(::mlx::core::negative(::mlx::core::abs(*x.arr))));
    auto dlx = ::mlx::core::subtract(::mlx::core::multiply(log_weight, sigm), *t.arr);
    auto dtarget_term =
        ::mlx::core::add(::mlx::core::negative(*x.arr), ::mlx::core::multiply(pw_m1, log1pexp));
    auto loss = ::mlx::core::add(::mlx::core::subtract(::mlx::core::maximum(*x.arr, zero),
                                                       ::mlx::core::multiply(*x.arr, *t.arr)),
                                 ::mlx::core::multiply(log_weight, log1pexp));
    auto dpos_weight = ::mlx::core::multiply(*w.arr, ::mlx::core::multiply(*t.arr, log1pexp));
    auto scaled = scale_loss_grad(*g.arr, reduction, shape_numel(shape), mlx_dt);
    if (reduction != 0)
        scaled = ::mlx::core::broadcast_to(scaled, gpu::to_mlx_shape(shape));
    auto dx = ::mlx::core::multiply(*w.arr, ::mlx::core::multiply(dlx, scaled));
    auto dtarget = ::mlx::core::multiply(*w.arr, ::mlx::core::multiply(dtarget_term, scaled));
    auto dweight = ::mlx::core::multiply(loss, scaled);
    auto dpos_weight_scaled = ::mlx::core::multiply(dpos_weight, scaled);
    return {
        Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(narrow_to(dx, mlx_dt)), dt)},
        Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(narrow_to(dtarget, mlx_dt)), dt)},
        Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(narrow_to(dweight, mlx_dt)), dt)},
        Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(narrow_to(dpos_weight_scaled, mlx_dt)),
                                    dt)},
    };
}

ClassLossForwardResult GpuBackend::cross_entropy_loss(const Storage& input,
                                                      const Storage& target,
                                                      const Storage* weight,
                                                      const Shape&,
                                                      const Shape& target_shape,
                                                      Dtype dt,
                                                      double eps,
                                                      int ignore_index,
                                                      int reduction) {
    const auto& x = std::get<GpuStorage>(input);
    const auto& t = std::get<GpuStorage>(target);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    auto softmax = ::mlx::core::softmax(*x.arr, std::vector<int>{1}, true);
    auto t_idx = class_target_indices(*t.arr, target_shape);
    auto ig_mask = class_ignore_mask(t_idx, ignore_index);
    auto safe_t = safe_class_indices(t_idx, ig_mask);

    auto pred = class_pick(softmax, safe_t);
    auto neg_log_pred = ::mlx::core::negative(
        ::mlx::core::log(::mlx::core::add(pred, gpu::mlx_scalar(eps, mlx_dt))));
    auto w_gather = class_weight_gather(weight, safe_t, neg_log_pred.shape(), mlx_dt);
    auto ig_mask_dt = ::mlx::core::astype(ig_mask, mlx_dt);
    auto loss = ::mlx::core::multiply(::mlx::core::multiply(w_gather, neg_log_pred), ig_mask_dt);
    auto loss_squeezed = ::mlx::core::squeeze(loss, std::vector<int>{1});
    auto valid_count = class_valid_count(ig_mask);
    auto output = reduce_class_loss(loss_squeezed, valid_count, dt, reduction);
    // PERF: softmax + valid_count are produced by fresh MLX kernels —
    // contiguous() wraps were redundant and forced a copy of the saved
    // softmax that the backward path then immediately re-read.
    return {std::move(output), Storage{gpu::wrap_mlx_array(std::move(softmax), dt)},
            Storage{gpu::wrap_mlx_array(narrow_to(valid_count, mlx_dt), dt)}};
}

Storage GpuBackend::cross_entropy_backward(const Storage& saved_softmax,
                                           const Storage& target,
                                           const Storage* weight,
                                           const Storage& valid_count,
                                           const Storage& grad,
                                           const Shape& input_shape,
                                           Dtype dt,
                                           int ignore_index,
                                           int reduction) {
    const auto& sm = std::get<GpuStorage>(saved_softmax);
    const auto& t = std::get<GpuStorage>(target);
    const auto& vc = std::get<GpuStorage>(valid_count);
    const auto& g = std::get<GpuStorage>(grad);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    auto full_shape = gpu::to_mlx_shape(input_shape);
    auto t_shape = full_shape;
    t_shape[1] = 1;
    auto t_idx = ::mlx::core::reshape(::mlx::core::astype(*t.arr, ::mlx::core::int64), t_shape);
    auto ig_mask = class_ignore_mask(t_idx, ignore_index);
    auto safe_t = safe_class_indices(t_idx, ig_mask);

    auto c_range = ::mlx::core::astype(::mlx::core::arange(0, static_cast<int>(input_shape[1]), 1),
                                       ::mlx::core::int64);
    ::mlx::core::Shape c_shape(full_shape.size(), 1);
    c_shape[1] = static_cast<int>(input_shape[1]);
    auto c_idx = ::mlx::core::reshape(c_range, c_shape);
    auto onehot = ::mlx::core::astype(::mlx::core::equal(c_idx, t_idx), mlx_dt);
    auto base = ::mlx::core::subtract(*sm.arr, onehot);

    auto w_gather = class_weight_gather(weight, safe_t, t_shape, mlx_dt);
    w_gather = ::mlx::core::multiply(w_gather, ::mlx::core::astype(ig_mask, mlx_dt));
    auto w_full = ::mlx::core::broadcast_to(w_gather, full_shape);
    // The saved count is in the loss dtype — inexact in float16 past
    // 2048 — so the float32 count is taken again from the targets.
    (void)vc;
    auto scaled = class_scaled_grad(*g.arr, class_valid_count(ig_mask), t_shape, reduction);
    auto scaled_full = ::mlx::core::broadcast_to(scaled, full_shape);
    auto dx = ::mlx::core::multiply(::mlx::core::multiply(base, w_full), scaled_full);
    // PERF: pure elementwise multiply chain — contiguous output already.
    return Storage{gpu::wrap_mlx_array(narrow_to(dx, mlx_dt), dt)};
}

ClassLossForwardResult GpuBackend::nll_loss(const Storage& input,
                                            const Storage& target,
                                            const Storage* weight,
                                            const Shape&,
                                            const Shape& target_shape,
                                            Dtype dt,
                                            int ignore_index,
                                            int reduction) {
    const auto& x = std::get<GpuStorage>(input);
    const auto& t = std::get<GpuStorage>(target);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    auto t_idx = class_target_indices(*t.arr, target_shape);
    auto ig_mask = class_ignore_mask(t_idx, ignore_index);
    auto safe_t = safe_class_indices(t_idx, ig_mask);
    auto pred = class_pick(*x.arr, safe_t);
    auto neg = ::mlx::core::negative(pred);
    auto w_gather = class_weight_gather(weight, safe_t, neg.shape(), mlx_dt);
    auto ig_mask_dt = ::mlx::core::astype(ig_mask, mlx_dt);
    auto loss = ::mlx::core::multiply(::mlx::core::multiply(w_gather, neg), ig_mask_dt);
    auto loss_squeezed = ::mlx::core::squeeze(loss, std::vector<int>{1});
    auto valid_count = class_valid_count(ig_mask);
    auto output = reduce_class_loss(loss_squeezed, valid_count, dt, reduction);
    return {std::move(output), Storage{},
            Storage{gpu::wrap_mlx_array(narrow_to(valid_count, mlx_dt), dt)}};
}

Storage GpuBackend::nll_loss_backward(const Storage& target,
                                      const Storage* weight,
                                      const Storage& valid_count,
                                      const Storage& grad,
                                      const Shape& input_shape,
                                      Dtype dt,
                                      int ignore_index,
                                      int reduction) {
    const auto& t = std::get<GpuStorage>(target);
    const auto& vc = std::get<GpuStorage>(valid_count);
    const auto& g = std::get<GpuStorage>(grad);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    auto full_shape = gpu::to_mlx_shape(input_shape);
    auto t_shape = full_shape;
    t_shape[1] = 1;
    auto t_idx = ::mlx::core::reshape(::mlx::core::astype(*t.arr, ::mlx::core::int64), t_shape);
    auto ig_mask = class_ignore_mask(t_idx, ignore_index);
    auto safe_t = safe_class_indices(t_idx, ig_mask);

    auto c_range = ::mlx::core::astype(::mlx::core::arange(0, static_cast<int>(input_shape[1]), 1),
                                       ::mlx::core::int64);
    ::mlx::core::Shape c_shape(full_shape.size(), 1);
    c_shape[1] = static_cast<int>(input_shape[1]);
    auto c_idx = ::mlx::core::reshape(c_range, c_shape);
    auto onehot = ::mlx::core::astype(::mlx::core::equal(c_idx, t_idx), mlx_dt);
    auto neg_onehot = ::mlx::core::negative(onehot);
    auto w_gather = class_weight_gather(weight, safe_t, t_shape, mlx_dt);
    w_gather = ::mlx::core::multiply(w_gather, ::mlx::core::astype(ig_mask, mlx_dt));
    auto w_full = ::mlx::core::broadcast_to(w_gather, full_shape);
    // See cross_entropy_backward: the count is retaken in float32.
    (void)vc;
    auto scaled = class_scaled_grad(*g.arr, class_valid_count(ig_mask), t_shape, reduction);
    auto scaled_full = ::mlx::core::broadcast_to(scaled, full_shape);
    auto dx = ::mlx::core::multiply(::mlx::core::multiply(neg_onehot, w_full), scaled_full);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(narrow_to(dx, mlx_dt)), dt)};
}

::mlx::core::array GpuBackend::apply_loss_reduction(const ::mlx::core::array& values,
                                                    int reduction) {
    if (reduction == 0)
        return values;
    auto sum = sum_all_f32(values);
    if (reduction == 1)
        sum = ::mlx::core::divide(sum,
                                  gpu::mlx_scalar(static_cast<double>(values.size()), sum.dtype()));
    return narrow_half(sum, values.dtype());
}

::mlx::core::array GpuBackend::scale_loss_grad(const ::mlx::core::array& grad,
                                               int reduction,
                                               std::size_t numel,
                                               ::mlx::core::Dtype dt) {
    (void)dt;
    if (reduction != 1)
        return grad;
    auto g = widen_half(grad);
    return ::mlx::core::divide(g, gpu::mlx_scalar(static_cast<double>(numel), g.dtype()));
}

::mlx::core::array GpuBackend::class_target_indices(const ::mlx::core::array& target,
                                                    const Shape& target_shape) {
    auto t_shape = gpu::to_mlx_shape(target_shape);
    t_shape.insert(t_shape.begin() + 1, 1);
    return ::mlx::core::reshape(::mlx::core::astype(target, ::mlx::core::int64), t_shape);
}

::mlx::core::array GpuBackend::class_ignore_mask(const ::mlx::core::array& target_idx,
                                                 int ignore_index) {
    auto ignore = ::mlx::core::astype(::mlx::core::array(ignore_index), ::mlx::core::int64);
    return ::mlx::core::not_equal(target_idx, ignore);
}

::mlx::core::array GpuBackend::safe_class_indices(const ::mlx::core::array& target_idx,
                                                  const ::mlx::core::array& keep_mask) {
    auto zero = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
    return ::mlx::core::where(keep_mask, target_idx, zero);
}

::mlx::core::array GpuBackend::class_pick(const ::mlx::core::array& x,
                                          const ::mlx::core::array& safe_t) {
    return gpu_take_along_axis(x, safe_t, 1, IndexSite::row("gpu_backend::class_pick"));
}

::mlx::core::array GpuBackend::class_weight_gather(const Storage* weight,
                                                   const ::mlx::core::array& safe_t,
                                                   const ::mlx::core::Shape& shape,
                                                   ::mlx::core::Dtype dt) {
    if (weight) {
        const auto& w = std::get<GpuStorage>(*weight);
        return gpu_take(*w.arr, safe_t, 0, IndexSite::row("gpu_backend::class_weight"));
    }
    return ::mlx::core::broadcast_to(gpu::mlx_scalar(1.0, dt), shape);
}

::mlx::core::array GpuBackend::class_valid_count(const ::mlx::core::array& keep_mask) {
    auto valid = ::mlx::core::sum(::mlx::core::astype(keep_mask, ::mlx::core::float32), false);
    return ::mlx::core::maximum(valid, ::mlx::core::array(1.0f));
}

Storage GpuBackend::reduce_class_loss(const ::mlx::core::array& values,
                                      const ::mlx::core::array& valid_count,
                                      Dtype dt,
                                      int reduction) {
    if (reduction == 0)
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(values), dt)};
    auto sum = sum_all_f32(values);
    if (reduction == 1)
        sum = ::mlx::core::divide(sum, ::mlx::core::astype(valid_count, sum.dtype()));
    return Storage{
        gpu::wrap_mlx_array(::mlx::core::contiguous(narrow_half(sum, values.dtype())), dt)};
}

::mlx::core::array GpuBackend::class_scaled_grad(const ::mlx::core::array& grad,
                                                 const ::mlx::core::array& valid_count,
                                                 const ::mlx::core::Shape& target_axis_shape,
                                                 int reduction) {
    ::mlx::core::array scaled = grad;
    if (reduction == 1)
        scaled = ::mlx::core::divide(widen_half(scaled), valid_count);
    if (reduction != 0)
        return ::mlx::core::broadcast_to(scaled, target_axis_shape);
    auto grad_shape = scaled.shape();
    grad_shape.insert(grad_shape.begin() + 1, 1);
    return ::mlx::core::reshape(scaled, grad_shape);
}

}  // namespace backend
}  // namespace lucid
