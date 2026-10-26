// lucid/_C/backend/gpu/methods/Indexing.cpp
// Out-of-line GpuBackend method definitions for the Indexing family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

Storage GpuBackend::gather(const Storage& a,
                           const Storage& indices,
                           const Shape&,
                           const Shape&,
                           int axis,
                           Dtype,
                           Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    const auto& gi = std::get<GpuStorage>(indices);
    // The index may be shorter than the input off ``axis`` (the reference's
    // rule, and the CPU's); MLX wants them to agree, and raised a broadcast
    // error for a valid gather.  Read from the corner the index covers.
    auto source = *ga.arr;
    ::mlx::core::Shape corner = gi.arr->shape();
    corner[static_cast<std::size_t>(axis)] = source.shape(axis);
    if (source.shape() != corner)
        source = ::mlx::core::slice(
            source, ::mlx::core::Shape(static_cast<std::size_t>(source.ndim()), 0), corner);
    auto result = gpu_take_along_axis(source, *gi.arr, axis);
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

Storage GpuBackend::gather_backward(const Storage& grad,
                                    const Storage& indices,
                                    const Shape& input_shape,
                                    const Shape&,
                                    int axis,
                                    Dtype,
                                    Dtype dt) {
    const auto& gg = std::get<GpuStorage>(grad);
    const auto& gi = std::get<GpuStorage>(indices);
    // The forward answered NaN where the index was out of range; nothing
    // was read there, so nothing flows back (policy B, gpu_axis_index).
    auto base = ::mlx::core::zeros(gpu::to_mlx_shape(input_shape), gpu::to_mlx_dtype(dt));
    auto result = gpu_scatter_reduce_axis(base, *gi.arr, *gg.arr, axis, ScatterReduce::Add,
                                          [axis](const auto& b, const auto& i, const auto& v) {
                                              return scatter_add_axis_widened(b, i, v, axis);
                                          });
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

std::pair<Storage, Storage> GpuBackend::sort_select(const Storage& a,
                                                    const Shape&,
                                                    const Shape& output_shape,
                                                    int axis,
                                                    Dtype dt,
                                                    bool descending) {
    const auto& ga = std::get<GpuStorage>(a);
    // The order lives in ``gpu/SortOrder.h``.
    const int k = static_cast<int>(output_shape[static_cast<std::size_t>(axis)]);
    auto idx = descending ? gpu_descending_order(*ga.arr, axis, k) : gpu_sort_order(*ga.arr, axis);
    auto values = ::mlx::core::take_along_axis(*ga.arr, idx, axis);
    return {
        Storage{gpu::wrap_mlx_array(std::move(values), dt)},
        Storage{gpu::wrap_mlx_array(::mlx::core::astype(idx, ::mlx::core::int64), Dtype::I64)},
    };
}

Storage GpuBackend::argsort(const Storage& a, const Shape&, int axis, Dtype) {
    const auto& ga = std::get<GpuStorage>(a);
    auto out = ::mlx::core::astype(gpu_sort_order(*ga.arr, axis), ::mlx::core::int64);
    return Storage{gpu::wrap_mlx_array(std::move(out), Dtype::I64)};
}

Storage GpuBackend::scatter_add_axis(const Storage& grad,
                                     const Storage& indices,
                                     const Shape& output_shape,
                                     const Shape&,
                                     int axis,
                                     Dtype dt) {
    const auto& g = std::get<GpuStorage>(grad);
    const auto& idx = std::get<GpuStorage>(indices);
    auto base = ::mlx::core::zeros(gpu::to_mlx_shape(output_shape), gpu::to_mlx_dtype(dt));
    auto out = gpu_scatter_reduce_axis(base, *idx.arr, *g.arr, axis, ScatterReduce::Add,
                                       [axis](const auto& b, const auto& i, const auto& v) {
                                           return scatter_add_axis_widened(b, i, v, axis);
                                       });
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

Storage GpuBackend::scatter_add(const Storage& base,
                                const Storage& indices,
                                const Storage& src,
                                const Shape& base_shape,
                                const Shape& idx_shape,
                                int dim,
                                Dtype dt) {
    const auto& gb = std::get<GpuStorage>(base);
    const auto& gi = std::get<GpuStorage>(indices);
    const auto& gs = std::get<GpuStorage>(src);
    const int ndim = static_cast<int>(base_shape.size());
    const int d = dim < 0 ? dim + ndim : dim;
    const std::int64_t extent = base_shape[static_cast<std::size_t>(d)];
    if (dt == Dtype::I64) {
        // The add runs on the CPU (see ``scatter_via_cpu``).  The index is
        // made safe on this side first: an out-of-range update becomes 0
        // aimed at an in-range position.  So the CPU kernel's IndexError
        // never fires, and Metal drops the update here as it does for
        // every other dtype.
        if (extent == 0)
            return base;
        auto ix = gpu_axis_index(*gi.arr, extent);
        auto values = ::mlx::core::where(ix.in_range, *gs.arr,
                                         scatter_identity(ScatterReduce::Add, gs.arr->dtype()));
        const Storage safe_idx{
            gpu::wrap_mlx_array(::mlx::core::astype(ix.safe, ::mlx::core::int64), Dtype::I64)};
        const Storage safe_src{gpu::wrap_mlx_array(std::move(values), dt)};
        return scatter_via_cpu(base, safe_idx, safe_src, base_shape, idx_shape,
                               [&](const Storage& b, const Storage& i, const Storage& v) {
                                   return backend::Dispatcher::for_device(Device::CPU)
                                       .scatter_add(b, i, v, base_shape, idx_shape, d, dt);
                               });
    }
    auto out = gpu_scatter_reduce_axis(*gb.arr, *gi.arr, *gs.arr, d, ScatterReduce::Add,
                                       [d](const auto& b, const auto& i, const auto& v) {
                                           return scatter_add_axis_widened(b, i, v, d);
                                       });
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

Storage GpuBackend::scatter_set(const Storage& base,
                                const Storage& indices,
                                const Storage& src,
                                const Shape& base_shape,
                                const Shape& idx_shape,
                                int dim,
                                Dtype dt) {
    const auto& gb = std::get<GpuStorage>(base);
    const auto& gi = std::get<GpuStorage>(indices);
    const auto& gs = std::get<GpuStorage>(src);
    const int ndim = static_cast<int>(base_shape.size());
    const int d = dim < 0 ? dim + ndim : dim;
    const std::int64_t extent = base_shape[static_cast<std::size_t>(d)];
    if (extent == 0)
        return base;
    auto sink = gpu_sink_index(*gi.arr, extent);
    auto padded = with_sink_slab(*gb.arr, d);
    if (dt == Dtype::I64 || dt == Dtype::C64) {
        Shape padded_shape = base_shape;
        padded_shape[static_cast<std::size_t>(d)] += 1;
        const Storage padded_base{gpu::wrap_mlx_array(std::move(padded), dt)};
        const Storage sink_idx{
            gpu::wrap_mlx_array(::mlx::core::astype(sink, ::mlx::core::int64), Dtype::I64)};
        const Storage out =
            scatter_via_cpu(padded_base, sink_idx, src, padded_shape, idx_shape,
                            [&](const Storage& b, const Storage& i, const Storage& v) {
                                return backend::Dispatcher::for_device(Device::CPU)
                                    .scatter_set(b, i, v, padded_shape, idx_shape, d, dt);
                            });
        return Storage{gpu::wrap_mlx_array(
            without_sink_slab(*std::get<GpuStorage>(out).arr, gb.arr->shape()), dt)};
    }
    auto out = scatter_on_index_corner(padded, sink, *gs.arr, d,
                                       [d](const auto& b, const auto& i, const auto& v) {
                                           if (Determinism::is_enabled())
                                               return put_along_axis_last_write_wins(b, i, v, d);
                                           return ::mlx::core::put_along_axis(b, i, v, d);
                                       });
    return Storage{gpu::wrap_mlx_array(without_sink_slab(out, gb.arr->shape()), dt)};
}

template <typename Fn>
Storage GpuBackend::scatter_via_cpu(const Storage& base,
                                    const Storage& indices,
                                    const Storage& src,
                                    const Shape& base_shape,
                                    const Shape& idx_shape,
                                    Fn&& run) {
    auto& cpu = backend::Dispatcher::for_device(Device::CPU);
    (void)cpu;
    Storage base_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(base), base_shape)};
    Storage idx_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(indices), idx_shape)};
    Storage src_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(src), idx_shape)};
    Storage out_cpu = run(base_cpu, idx_cpu, src_cpu);
    return Storage{gpu::upload_cpu_to_gpu(std::get<CpuStorage>(out_cpu), base_shape)};
}

template <typename ScatterFn>
Storage GpuBackend::axis_scatter_via_multiaxis(const Storage& base,
                                               const Storage& indices,
                                               const Storage& src,
                                               const Shape& base_shape,
                                               const Shape& idx_shape,
                                               int dim,
                                               Dtype dt,
                                               ScatterReduce op,
                                               ScatterFn scatter_fn) {
    const auto& gb = std::get<GpuStorage>(base);
    const auto& gi = std::get<GpuStorage>(indices);
    const auto& gs = std::get<GpuStorage>(src);
    const int ndim = static_cast<int>(base_shape.size());
    const int d = dim < 0 ? dim + ndim : dim;
    // The op (``axis_scatter_operands`` in Gfunc.cpp) has already made the
    // index no wider than the base off ``dim`` and cut src to its corner.
    const std::int64_t extent = base_shape[static_cast<std::size_t>(d)];
    if (extent == 0)
        return base;
    auto ix = gpu_axis_index(*gi.arr, extent);
    const auto coord_dt = ix.safe.dtype();

    std::vector<::mlx::core::array> index_list;
    std::vector<int> axes_v;
    index_list.reserve(static_cast<std::size_t>(ndim));
    axes_v.reserve(static_cast<std::size_t>(ndim));
    for (int a = 0; a < ndim; ++a) {
        axes_v.push_back(a);
        if (a == d) {
            index_list.push_back(ix.safe);
        } else {
            auto sz = static_cast<int>(idx_shape[static_cast<std::size_t>(a)]);
            auto coord = ::mlx::core::arange(0, sz, 1, coord_dt);
            std::vector<int> reshape_dims(ndim, 1);
            reshape_dims[a] = sz;
            coord = ::mlx::core::reshape(
                coord, ::mlx::core::Shape(reshape_dims.begin(), reshape_dims.end()));
            index_list.push_back(std::move(coord));
        }
    }
    // MLX multi-axis scatter with K=ndim: updates must include a leading
    // K-dim index-broadcast block followed by the trailing (ndim-K)
    // non-scatter dims.  With K=ndim, that's an extra K leading 1s
    // prepended to src to satisfy the rank check; the actual scatter
    // payload remains src broadcast across the leading singletons.
    std::vector<int> upd_shape;
    upd_shape.reserve(static_cast<std::size_t>(2 * ndim));
    for (int a = 0; a < ndim; ++a)
        upd_shape.push_back(static_cast<int>(idx_shape[static_cast<std::size_t>(a)]));
    for (int a = 0; a < ndim; ++a)
        upd_shape.push_back(1);
    auto values =
        ::mlx::core::where(ix.in_range, ::mlx::core::reshape(*gs.arr, ix.in_range.shape()),
                           scatter_identity(op, gs.arr->dtype()));
    auto updates =
        ::mlx::core::reshape(values, ::mlx::core::Shape(upd_shape.begin(), upd_shape.end()));

    auto out = scatter_fn(*gb.arr, index_list, updates, axes_v);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
}

Storage GpuBackend::scatter_amax(const Storage& base,
                                 const Storage& indices,
                                 const Storage& src,
                                 const Shape& base_shape,
                                 const Shape& idx_shape,
                                 int dim,
                                 Dtype dt) {
    return axis_scatter_via_multiaxis(
        base, indices, src, base_shape, idx_shape, dim, dt, ScatterReduce::Max,
        [](const ::mlx::core::array& a, const std::vector<::mlx::core::array>& idxs,
           const ::mlx::core::array& upd,
           const std::vector<int>& axes) { return gpu_scatter_max(a, idxs, upd, axes); });
}

Storage GpuBackend::scatter_amin(const Storage& base,
                                 const Storage& indices,
                                 const Storage& src,
                                 const Shape& base_shape,
                                 const Shape& idx_shape,
                                 int dim,
                                 Dtype dt) {
    return axis_scatter_via_multiaxis(
        base, indices, src, base_shape, idx_shape, dim, dt, ScatterReduce::Min,
        [](const ::mlx::core::array& a, const std::vector<::mlx::core::array>& idxs,
           const ::mlx::core::array& upd,
           const std::vector<int>& axes) { return gpu_scatter_min(a, idxs, upd, axes); });
}

Storage GpuBackend::scatter_prod(const Storage& base,
                                 const Storage& indices,
                                 const Storage& src,
                                 const Shape& base_shape,
                                 const Shape& idx_shape,
                                 int dim,
                                 Dtype dt) {
    return axis_scatter_via_multiaxis(
        base, indices, src, base_shape, idx_shape, dim, dt, ScatterReduce::Prod,
        [](const ::mlx::core::array& a, const std::vector<::mlx::core::array>& idxs,
           const ::mlx::core::array& upd,
           const std::vector<int>& axes) { return gpu_scatter_prod(a, idxs, upd, axes); });
}

Storage GpuBackend::masked_select_count(const Storage& mask, const Shape& shape, Dtype dt) {
    const auto& gm = std::get<GpuStorage>(mask);
    auto cm = ::mlx::core::contiguous(*gm.arr);
    cm.eval();
    auto cpu_m = allocate_aligned_bytes(cm.nbytes(), Device::CPU);
    MemoryTracker::track_host_sync(cm.nbytes());
    std::memcpy(cpu_m.get(), cm.data<void>(), cm.nbytes());
    CpuStorage m_cs{cpu_m, cm.nbytes(), dt};
    return Dispatcher::for_device(Device::CPU).masked_select_count(Storage{m_cs}, shape, dt);
}

Storage GpuBackend::masked_select(const Storage& a,
                                  const Storage& mask,
                                  const Shape& a_shape,
                                  const Shape& mask_shape,
                                  std::int64_t n_true,
                                  Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    const auto& gm = std::get<GpuStorage>(mask);
    auto ca = ::mlx::core::contiguous(*ga.arr);
    ca.eval();
    auto cm = ::mlx::core::contiguous(*gm.arr);
    cm.eval();
    auto cpu_a = allocate_aligned_bytes(ca.nbytes(), Device::CPU);
    auto cpu_m = allocate_aligned_bytes(cm.nbytes(), Device::CPU);
    MemoryTracker::track_host_sync(ca.nbytes());
    MemoryTracker::track_host_sync(cm.nbytes());
    std::memcpy(cpu_a.get(), ca.data<void>(), ca.nbytes());
    std::memcpy(cpu_m.get(), cm.data<void>(), cm.nbytes());
    CpuStorage a_cs{cpu_a, ca.nbytes(), dt};
    CpuStorage m_cs{cpu_m, cm.nbytes(), Dtype::Bool};
    auto result = Dispatcher::for_device(Device::CPU)
                      .masked_select(Storage{a_cs}, Storage{m_cs}, a_shape, mask_shape, n_true, dt);
    const auto& res_cpu = std::get<CpuStorage>(result);
    Shape out_shape{n_true};
    return Storage{gpu::upload_cpu_to_gpu(res_cpu, out_shape)};
}

Storage GpuBackend::one_hot_forward(const Storage& indices,
                                    const Shape& indices_shape,
                                    int num_classes,
                                    Dtype out_dtype) {
    const auto& gi = std::get<GpuStorage>(indices);
    auto cls = ::mlx::core::astype(::mlx::core::arange(0, num_classes, 1), ::mlx::core::int64);
    auto idx = ::mlx::core::astype(*gi.arr, ::mlx::core::int64);
    ::mlx::core::Shape idx_shape = idx.shape();
    idx_shape.push_back(1);
    auto idx_b = ::mlx::core::reshape(idx, idx_shape);
    ::mlx::core::Shape cls_shape(idx_shape.size(), 1);
    cls_shape.back() = num_classes;
    auto cls_b = ::mlx::core::reshape(cls, cls_shape);
    auto eq = ::mlx::core::equal(idx_b, cls_b);
    auto out = ::mlx::core::astype(eq, gpu::to_mlx_dtype(out_dtype));
    return Storage{gpu::wrap_mlx_array(std::move(out), out_dtype)};
}

CpuStorage
GpuBackend::nonzero_forward(const Storage&, const Shape&, Dtype, std::size_t& numel_out) {
    ErrorBuilder("gpu::nonzero_forward").not_implemented("call to_cpu() first");
    numel_out = 0;
    return CpuStorage{};
}

}  // namespace backend
}  // namespace lucid
