// lucid/_C/backend/gpu/methods/Reduction.cpp
// Out-of-line GpuBackend method definitions for the Reduction family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

Storage
GpuBackend::reduce_sum(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) {
    return mlx_reduce(a, in_shape, opts, dt, [](auto& x, auto& axes, bool keepdims) {
        return sum_widened(x, axes, keepdims);
    });
}

Storage
GpuBackend::reduce_mean(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) {
    return mlx_reduce(a, in_shape, opts, dt, [](auto& x, auto& axes, bool keepdims) {
        if (!is_half(x.dtype())) {
            auto s = ::mlx::core::sum(x, axes, keepdims);
            std::vector<int> norm = axes;
            std::size_t n = 1;
            const int nd = static_cast<int>(x.ndim());
            if (norm.empty()) {
                n = x.size();
            } else if (normalize_reduce_axes(nd, norm)) {
                for (int ax : norm)
                    n *= static_cast<std::size_t>(x.shape(ax));
            }
            return ::mlx::core::divide(s, ::mlx::core::array(static_cast<double>(n), s.dtype()));
        }
        return mean_widened(x, axes, keepdims);
    });
}

Storage GpuBackend::variance(const Storage& a, const Shape&, const ReduceOpts& opts, Dtype dt) {
    const auto& gs = std::get<GpuStorage>(a);
    auto result = var_widened(*gs.arr, opts.axes, opts.keepdims, 0);
    // PERF: var() produces contiguous output; drop defensive wrap.
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

Storage
GpuBackend::reduce_max(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) {
    return mlx_reduce(a, in_shape, opts, dt, [](auto& x, auto& axes, bool keepdims) {
        if (x.size() == 0)
            return empty_extreme(x, axes, keepdims, x.dtype());
        return ::mlx::core::max(x, axes, keepdims);
    });
}

Storage
GpuBackend::reduce_min(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) {
    return mlx_reduce(a, in_shape, opts, dt, [](auto& x, auto& axes, bool keepdims) {
        if (x.size() == 0)
            return empty_extreme(x, axes, keepdims, x.dtype());
        return ::mlx::core::min(x, axes, keepdims);
    });
}

Storage GpuBackend::cumsum(const Storage& a, const Shape&, int axis, Dtype dt) {
    const auto& gs = std::get<GpuStorage>(a);
    auto out = reduce_widened(
        *gs.arr, [axis](const ::mlx::core::array& w) { return ::mlx::core::cumsum(w, axis); });
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

Storage GpuBackend::cumprod(const Storage& a, const Shape&, int axis, Dtype dt) {
    const auto& gs = std::get<GpuStorage>(a);
    auto out = reduce_widened(
        *gs.arr, [axis](const ::mlx::core::array& w) { return ::mlx::core::cumprod(w, axis); });
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

::mlx::core::array GpuBackend::poison_after_nan(const ::mlx::core::array& input,
                                                const ::mlx::core::array& scanned,
                                                int axis) {
    if (!::mlx::core::issubdtype(input.dtype(), ::mlx::core::floating))
        return scanned;
    auto mask = ::mlx::core::astype(::mlx::core::isnan(input), ::mlx::core::int32);
    auto seen = ::mlx::core::cummax(mask, axis);
    auto nan_like =
        ::mlx::core::full(input.shape(), std::numeric_limits<float>::quiet_NaN(), scanned.dtype());
    return ::mlx::core::where(::mlx::core::greater(seen, ::mlx::core::zeros_like(seen)), nan_like,
                              scanned);
}

Storage GpuBackend::cummax(const Storage& a, const Shape&, int axis, Dtype dt) {
    const auto& gs = std::get<GpuStorage>(a);
    auto out = ::mlx::core::cummax(*gs.arr, axis);
    return Storage{gpu::wrap_mlx_array(poison_after_nan(*gs.arr, out, axis), dt)};
}

Storage GpuBackend::cummin(const Storage& a, const Shape&, int axis, Dtype dt) {
    const auto& gs = std::get<GpuStorage>(a);
    auto out = ::mlx::core::cummin(*gs.arr, axis);
    return Storage{gpu::wrap_mlx_array(poison_after_nan(*gs.arr, out, axis), dt)};
}

Storage GpuBackend::trace(const Storage& a, const Shape&, Dtype dt) {
    // A bool has no trace.  MLX summed the diagonal as bool, so a
    // matrix with two `true`s on it traced to `true` — saturating
    // rather than counting, and silently.  Both the reference and the
    // CPU refuse; refuse here so the two devices agree.
    if (dt == Dtype::Bool)
        ErrorBuilder("gpu_backend::trace")
            .not_implemented("trace is not defined for bool — cast to an integer dtype");
    const auto& gs = std::get<GpuStorage>(a);
    auto result = ::mlx::core::trace(*gs.arr, 0, 0, 1);
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

Storage GpuBackend::trace_backward(const Storage& grad_out, const Shape& input_shape, Dtype dt) {
    const auto& gg = std::get<GpuStorage>(grad_out);
    const std::int64_t M = input_shape[0];
    const std::int64_t N = input_shape[1];
    auto eye = eye_or_empty(static_cast<int>(M), static_cast<int>(N), 0, gpu::to_mlx_dtype(dt));
    auto result = ::mlx::core::multiply(eye, *gg.arr);
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

Storage GpuBackend::arg_reduce_index(
    const Storage& a, const Shape&, int axis, bool keepdims, Dtype dt, bool is_min) {
    // The reference refuses bool here and so does the CPU, on the
    // ground that "the largest of these booleans" is a question about
    // ordering that bool does not answer.  Refuse for the same reason
    // rather than let MLX pick one.
    if (dt == Dtype::Bool)
        ErrorBuilder("gpu_backend::arg_reduce_index")
            .not_implemented("argmax / argmin do not accept bool — cast to an integer dtype");
    const auto& ga = std::get<GpuStorage>(a);
    if (ga.arr->size() == 0)
        return Storage{gpu::wrap_mlx_array(
            empty_extreme(*ga.arr, {axis}, keepdims, ::mlx::core::int64), Dtype::I64)};
    auto out = is_min ? ::mlx::core::argmin(*ga.arr, axis, keepdims)
                      : ::mlx::core::argmax(*ga.arr, axis, keepdims);
    // MLX's argmax and argmin skip NaN, so they answer with the index
    // of some ordinary element — while ``max`` on the same axis answers
    // nan.  The pair then disagrees with itself, which is worse than
    // either convention: ``x[argmax(x)]`` was not ``max(x)``.
    //
    // The reference framework returns the index of the first NaN, and
    // MLX has no flag for it, so it is composed: ``argmax`` of the NaN
    // mask is the first NaN's position, and ``any`` says whether to use
    // it.  Only floats can carry one — bfloat16 among them, which this
    // list once left out, so its argmax skipped the NaN.
    if (dt == Dtype::F32 || dt == Dtype::F64 || dt == Dtype::F16 || dt == Dtype::BF16) {
        auto nan_mask = ::mlx::core::isnan(*ga.arr);
        auto has_nan = ::mlx::core::any(nan_mask, axis, keepdims);
        auto first_nan = ::mlx::core::argmax(nan_mask, axis, keepdims);
        out = ::mlx::core::where(has_nan, ::mlx::core::astype(first_nan, out.dtype()), out);
    }
    out = ::mlx::core::astype(out, ::mlx::core::int64);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), Dtype::I64)};
}

Storage GpuBackend::reduce_grad_to_shape(const Storage& grad,
                                         const Shape& grad_shape,
                                         const Shape& target_shape,
                                         Dtype dt) {
    const auto& src_gpu = std::get<GpuStorage>(grad);
    if (!src_gpu.arr)
        ErrorBuilder("gpu_backend::reduce_grad_to_shape").fail("null GPU array");
    if (grad_shape == target_shape) {
        auto cloned = ::mlx::core::contiguous(*src_gpu.arr);
        return Storage{gpu::wrap_mlx_array(std::move(cloned), dt)};
    }

    const std::size_t gn = grad_shape.size();
    const std::size_t tn = target_shape.size();
    const std::size_t lead = gn - tn;
    std::vector<int> axes_i;
    for (std::size_t i = 0; i < lead; ++i)
        axes_i.push_back(static_cast<int>(i));
    for (std::size_t i = 0; i < tn; ++i) {
        if (grad_shape[lead + i] != target_shape[i] && target_shape[i] == 1)
            axes_i.push_back(static_cast<int>(lead + i));
    }
    auto reduced = sum_widened(*src_gpu.arr, axes_i, false);
    auto reshaped = ::mlx::core::reshape(reduced, gpu::to_mlx_shape(target_shape));
    return Storage{gpu::wrap_mlx_array(std::move(reshaped), dt)};
}

Storage GpuBackend::broadcast_back_for_reduce(const Storage& grad,
                                              const Shape&,
                                              const Shape& input_shape,
                                              const std::vector<int>& axes,
                                              bool,
                                              Dtype dt) {
    const auto& g = std::get<GpuStorage>(grad);
    if (!g.arr)
        ErrorBuilder("gpu_backend::broadcast_back_for_reduce").fail("null GPU array");
    Shape kept_shape = input_shape;
    for (int a : axes)
        kept_shape[a] = 1;
    auto reshaped = ::mlx::core::reshape(*g.arr, gpu::to_mlx_shape(kept_shape));
    auto bcast = ::mlx::core::broadcast_to(reshaped, gpu::to_mlx_shape(input_shape));
    auto cloned = ::mlx::core::contiguous(bcast);
    return Storage{gpu::wrap_mlx_array(std::move(cloned), dt)};
}

Storage GpuBackend::reduce_broadcast(const Storage& grad,
                                     const Shape& input_shape,
                                     const Shape& output_shape,
                                     Dtype dt) {
    const auto& gg = std::get<GpuStorage>(grad);
    const std::size_t nout = output_shape.size();
    const std::size_t nin = input_shape.size();
    Shape padded(nout, 1);
    std::copy(input_shape.begin(), input_shape.end(), padded.begin() + (nout - nin));
    ::mlx::core::array out = *gg.arr;
    std::vector<int> sum_axes;
    for (std::size_t d = 0; d < nout; ++d) {
        if (padded[d] == 1 && output_shape[d] != 1)
            sum_axes.push_back(static_cast<int>(d));
    }
    if (!sum_axes.empty())
        out = sum_widened(out, sum_axes, true);
    if (nout != nin)
        out = ::mlx::core::reshape(out, gpu::to_mlx_shape(input_shape));
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

Storage GpuBackend::histogram_forward(
    const Storage&, const Shape&, Dtype, double, double, std::int64_t, bool) {
    ErrorBuilder("gpu::histogram_forward")
        .not_implemented("call to_cpu() first and dispatch to CPU");
    return {};
}

}  // namespace backend
}  // namespace lucid
