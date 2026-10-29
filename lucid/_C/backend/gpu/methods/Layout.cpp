// lucid/_C/backend/gpu/methods/Layout.cpp
// Out-of-line GpuBackend method definitions for the Layout family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

Storage GpuBackend::reverse_along_axis(const Storage& a, const Shape& shape, int axis, Dtype dt) {
    const auto& gs = std::get<GpuStorage>(a);
    std::vector<std::int32_t> idx(shape[static_cast<std::size_t>(axis)]);
    for (std::int64_t i = 0; i < shape[static_cast<std::size_t>(axis)]; ++i)
        idx[static_cast<std::size_t>(i)] =
            static_cast<std::int32_t>(shape[static_cast<std::size_t>(axis)] - 1 - i);
    ::mlx::core::Shape idx_shape(shape.size(), 1);
    idx_shape[static_cast<std::size_t>(axis)] = shape[static_cast<std::size_t>(axis)];
    ::mlx::core::array idx_arr(idx.data(), idx_shape, ::mlx::core::int32);
    idx_arr = ::mlx::core::broadcast_to(idx_arr, gpu::to_mlx_shape(shape));
    auto result = ::mlx::core::take_along_axis(*gs.arr, idx_arr, axis);
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

Storage
GpuBackend::diagonal(const Storage& a, const Shape&, int offset, int axis1, int axis2, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    auto result = ::mlx::core::contiguous(::mlx::core::diagonal(*ga.arr, offset, axis1, axis2));
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

Storage GpuBackend::diagonal_backward(const Storage& grad,
                                      const Shape& input_shape,
                                      const Shape& output_shape,
                                      int offset,
                                      int axis1,
                                      int axis2,
                                      Dtype dt) {
    const auto& gg = std::get<GpuStorage>(grad);
    const std::size_t ndim = input_shape.size();
    const int a1n = axis1 < 0 ? axis1 + static_cast<int>(ndim) : axis1;
    const int a2n = axis2 < 0 ? axis2 + static_cast<int>(ndim) : axis2;
    const std::int64_t r0 = (offset >= 0) ? 0 : -offset;
    const std::int64_t c0 = (offset >= 0) ? offset : 0;
    const std::int64_t L = output_shape.empty() ? 0 : output_shape.back();

    ::mlx::core::Shape mlx_in_shape = gpu::to_mlx_shape(input_shape);
    ::mlx::core::Shape mlx_out_shape = gpu::to_mlx_shape(output_shape);
    auto base = ::mlx::core::zeros(mlx_in_shape, gpu::to_mlx_dtype(dt));

    auto build_index = [&](int axis_in_input, std::int64_t start) {
        int out_axis;
        if (axis_in_input == a1n || axis_in_input == a2n) {
            out_axis = static_cast<int>(output_shape.size()) - 1;
        } else {
            int rel = 0;
            for (int d = 0; d < axis_in_input; ++d)
                if (d != a1n && d != a2n)
                    ++rel;
            out_axis = rel;
        }
        const std::int64_t span = (axis_in_input == a1n || axis_in_input == a2n)
                                      ? L
                                      : input_shape[static_cast<std::size_t>(axis_in_input)];
        auto arr = ::mlx::core::arange(static_cast<int>(start), static_cast<int>(start + span),
                                       ::mlx::core::int32);
        ::mlx::core::Shape bc(output_shape.size(), 1);
        bc[static_cast<std::size_t>(out_axis)] = static_cast<int>(span);
        arr = ::mlx::core::reshape(arr, bc);
        return ::mlx::core::broadcast_to(arr, mlx_out_shape);
    };

    std::vector<::mlx::core::array> indices;
    std::vector<int> axes_v;
    for (std::size_t d = 0; d < ndim; ++d) {
        std::int64_t start = 0;
        if (static_cast<int>(d) == a1n)
            start = r0;
        else if (static_cast<int>(d) == a2n)
            start = c0;
        indices.push_back(build_index(static_cast<int>(d), start));
        axes_v.push_back(static_cast<int>(d));
    }
    ::mlx::core::Shape upd_shape = mlx_out_shape;
    for (std::size_t d = 0; d < ndim; ++d)
        upd_shape.push_back(1);
    auto updates = ::mlx::core::reshape(*gg.arr, upd_shape);
    auto result = gpu_scatter_add(base, indices, updates, axes_v);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

Storage GpuBackend::roll(const Storage& a,
                         const Shape&,
                         Dtype dt,
                         const std::vector<std::int64_t>& shifts,
                         const std::vector<int>& axes) {
    const auto& ga = std::get<GpuStorage>(a);
    ::mlx::core::Shape mshifts(shifts.begin(), shifts.end());
    auto result = ::mlx::core::roll(*ga.arr, mshifts, axes);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

Storage GpuBackend::reshape(const Storage& a, const Shape&, const Shape& dst_shape, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    auto result = ::mlx::core::reshape(*ga.arr, gpu::to_mlx_shape(dst_shape));
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

Storage GpuBackend::slice_axis(const Storage& a,
                               const Shape& src_shape,
                               const Shape& slice_shape,
                               int axis,
                               std::int64_t offset,
                               Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    ::mlx::core::Shape lo(src_shape.size(), 0);
    ::mlx::core::Shape hi = gpu::to_mlx_shape(src_shape);
    lo[static_cast<std::size_t>(axis)] = static_cast<::mlx::core::ShapeElem>(offset);
    hi[static_cast<std::size_t>(axis)] =
        static_cast<::mlx::core::ShapeElem>(offset + slice_shape[static_cast<std::size_t>(axis)]);
    auto result = ::mlx::core::slice(*ga.arr, lo, hi);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

Storage GpuBackend::insert_axis_slice(const Storage& a,
                                      const Shape& src_shape,
                                      const Shape& dst_shape,
                                      int axis,
                                      std::int64_t offset,
                                      Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    std::vector<std::pair<int, int>> pad;
    pad.reserve(dst_shape.size());
    for (std::size_t d = 0; d < dst_shape.size(); ++d) {
        if (static_cast<int>(d) == axis) {
            const auto before = static_cast<int>(offset);
            const auto after = static_cast<int>(dst_shape[d] - offset - src_shape[d]);
            pad.emplace_back(before, after);
        } else {
            pad.emplace_back(0, 0);
        }
    }
    ::mlx::core::array zero(static_cast<float>(0.0), gpu::to_mlx_dtype(dt));
    auto result = ::mlx::core::pad(*ga.arr, pad, zero);
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

Storage GpuBackend::concatenate(const std::vector<Storage>& xs,
                                const std::vector<Shape>&,
                                int axis,
                                Dtype dt) {
    std::vector<::mlx::core::array> arrays;
    arrays.reserve(xs.size());
    for (const auto& x : xs)
        arrays.push_back(*std::get<GpuStorage>(x).arr);
    auto result = ::mlx::core::concatenate(std::move(arrays), axis);
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

Storage GpuBackend::stack(const std::vector<Storage>& xs, const Shape&, int axis, Dtype dt) {
    std::vector<::mlx::core::array> arrays;
    arrays.reserve(xs.size());
    for (const auto& x : xs)
        arrays.push_back(*std::get<GpuStorage>(x).arr);
    auto result = ::mlx::core::stack(arrays, axis);
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

std::vector<Storage> GpuBackend::split_equal(
    const Storage& a, const Shape&, int axis, std::int64_t num_splits, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    auto pieces = ::mlx::core::split(*ga.arr, static_cast<int>(num_splits), axis);
    std::vector<Storage> out;
    out.reserve(pieces.size());
    for (auto& p : pieces) {
        auto contiguous = ::mlx::core::contiguous(p);
        out.push_back(Storage{gpu::wrap_mlx_array(std::move(contiguous), dt)});
    }
    return out;
}

std::vector<Storage> GpuBackend::split_at(
    const Storage& a, const Shape&, int axis, const std::vector<std::int64_t>& indices, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    ::mlx::core::Shape mlx_idx(indices.begin(), indices.end());
    auto pieces = ::mlx::core::split(*ga.arr, mlx_idx, axis);
    std::vector<Storage> out;
    out.reserve(pieces.size());
    for (auto& p : pieces) {
        auto contiguous = ::mlx::core::contiguous(p);
        out.push_back(Storage{gpu::wrap_mlx_array(std::move(contiguous), dt)});
    }
    return out;
}

Storage GpuBackend::repeat_backward(const Storage& grad_out,
                                    const Shape& input_shape,
                                    const Shape& output_shape,
                                    int axis,
                                    std::int64_t repeats,
                                    Dtype dt) {
    const auto& g = std::get<GpuStorage>(grad_out);
    ::mlx::core::Shape reshape_shape;
    reshape_shape.reserve(output_shape.size() + 1);
    for (std::size_t d = 0; d < output_shape.size(); ++d) {
        if (static_cast<int>(d) == axis) {
            reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(input_shape[d]));
            reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(repeats));
        } else {
            reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(output_shape[d]));
        }
    }
    auto reshaped = ::mlx::core::reshape(*g.arr, reshape_shape);
    auto summed = sum_widened(reshaped, std::vector<int>{axis + 1}, false);
    auto result = ::mlx::core::reshape(summed, gpu::to_mlx_shape(input_shape));
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

Storage GpuBackend::tile_backward(const Storage& grad_out,
                                  const Shape& input_shape,
                                  const Shape& padded_shape,
                                  const Shape&,
                                  const std::vector<std::int64_t>& reps,
                                  Dtype dt) {
    const auto& g = std::get<GpuStorage>(grad_out);
    ::mlx::core::Shape reshape_shape;
    reshape_shape.reserve(reps.size() * 2);
    std::vector<int> sum_axes;
    sum_axes.reserve(reps.size());
    for (std::size_t d = 0; d < reps.size(); ++d) {
        sum_axes.push_back(static_cast<int>(reshape_shape.size()));
        reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(reps[d]));
        reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(padded_shape[d]));
    }
    auto reshaped = ::mlx::core::reshape(*g.arr, reshape_shape);
    auto summed = sum_axes.empty() ? reshaped : sum_widened(reshaped, sum_axes, false);
    auto result = ::mlx::core::reshape(summed, gpu::to_mlx_shape(input_shape));
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

Storage GpuBackend::unfold_dim(
    const Storage& a, const Shape& in_shape, int dim, int size, int step, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    const int ndim = static_cast<int>(in_shape.size());
    if (dim < 0)
        const_cast<int&>(dim) += ndim;
    const int dim_size = static_cast<int>(in_shape[static_cast<std::size_t>(dim)]);
    const int L = (dim_size - size) / step + 1;

    // ``take`` splices *all* of the index array's axes into the output at
    // ``dim``, so a broadcast-shaped (rank ndim+1) index would yield rank
    // 2*ndim — the extra singleton axes desync the MLX array from the Shape
    // Lucid reports, and for dim < ndim-1 the window axis also lands in the
    // wrong position.  Index with the plain (L, size) array instead and move
    // the window axis to the end, matching the CPU contract:
    //   out_shape = in_shape[:dim] + [L] + in_shape[dim+1:] + [size]
    auto starts = ::mlx::core::arange(0, L, 1, ::mlx::core::int32);  // (L,)
    starts = ::mlx::core::multiply(starts, ::mlx::core::array(step, ::mlx::core::int32));
    auto offsets = ::mlx::core::arange(0, size, 1, ::mlx::core::int32);  // (size,)
    // starts: (L,1) + offsets: (1,size) → (L, size)
    starts = ::mlx::core::reshape(starts, {L, 1});
    offsets = ::mlx::core::reshape(offsets, {1, size});
    auto idx = ::mlx::core::add(starts, offsets);  // (L, size)

    auto out = ::mlx::core::take(*ga.arr, idx, dim);  // (..., L, size, ...)

    // Move the window axis from dim+1 past the trailing input dims.
    // Identity when dim is already the last axis.
    const int out_rank = ndim + 1;
    std::vector<int> perm;
    perm.reserve(static_cast<std::size_t>(out_rank));
    for (int d = 0; d <= dim; ++d)
        perm.push_back(d);
    for (int d = dim + 2; d < out_rank; ++d)
        perm.push_back(d);
    perm.push_back(dim + 1);
    out = ::mlx::core::transpose(out, perm);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
}

Storage
GpuBackend::flip(const Storage& a, const Shape& shape, const std::vector<int>& dims, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    auto arr = ::mlx::core::contiguous(*ga.arr);
    for (int d : dims) {
        const int n = static_cast<int>(shape[d]);
        // Build reversed-index array [n-1, n-2, ..., 0].
        std::vector<int> idx_vec(n);
        for (int i = 0; i < n; ++i)
            idx_vec[i] = n - 1 - i;
        auto idx = ::mlx::core::array(idx_vec.data(), {n}, ::mlx::core::int32);
        arr = ::mlx::core::take(arr, idx, d);
    }
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(arr), dt)};
}

Storage
GpuBackend::broadcast(const Storage& a, const Shape& src_shape, const Shape& dst_shape, Dtype dt) {
    const auto& gs = std::get<GpuStorage>(a);
    auto result = ::mlx::core::broadcast_to(*gs.arr, gpu::to_mlx_shape(dst_shape));
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

// MLX takes a repeat count as an ``int``.  A larger one was truncated, and
// the result was shorter than the output shape the op records for it: the
// tensor claimed elements its buffer did not have.  The output extent is
// checked against MLX's ``int32`` limit instead, which bounds every count
// over a non-empty axis; over an empty one the count makes no elements.
Storage
GpuBackend::repeat(const Storage& a, const Shape& shape, Dtype dt, std::int64_t repeats, int axis) {
    const auto& gs = std::get<GpuStorage>(a);
    const auto ax = static_cast<std::size_t>(axis);
    Shape out_shape = shape;
    out_shape[ax] = mul_extent(shape[ax], repeats);
    static_cast<void>(gpu::to_mlx_shape(out_shape));
    const int count = shape[ax] == 0 ? 0 : static_cast<int>(repeats);
    auto result = ::mlx::core::repeat(*gs.arr, count, axis);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

Storage GpuBackend::tile(const Storage& a,
                         const Shape& shape,
                         Dtype dt,
                         const std::vector<std::int64_t>& reps) {
    const auto& gs = std::get<GpuStorage>(a);
    const std::size_t lead = reps.size() - shape.size();
    Shape out_shape(reps.size());
    std::vector<int> reps_int(reps.size());
    for (std::size_t d = 0; d < reps.size(); ++d) {
        const std::int64_t extent = d < lead ? 1 : shape[d - lead];
        out_shape[d] = mul_extent(extent, reps[d]);
        reps_int[d] = extent == 0 ? 0 : static_cast<int>(reps[d]);
    }
    static_cast<void>(gpu::to_mlx_shape(out_shape));
    auto result = ::mlx::core::tile(*gs.arr, std::move(reps_int));
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

Storage
GpuBackend::permute(const Storage& a, const Shape&, const std::vector<int>& perm, Dtype dt) {
    const auto& gs = std::get<GpuStorage>(a);
    auto result = ::mlx::core::transpose(*gs.arr, perm);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

Storage GpuBackend::pad(const Storage& a,
                        const Shape&,
                        Dtype dt,
                        const std::vector<std::pair<std::int64_t, std::int64_t>>& pad_width,
                        double constant) {
    const auto& gs = std::get<GpuStorage>(a);
    std::vector<std::pair<int, int>> mlx_pad;
    mlx_pad.reserve(pad_width.size());
    for (const auto& [lo, hi] : pad_width)
        mlx_pad.emplace_back(static_cast<int>(lo), static_cast<int>(hi));
    ::mlx::core::array pad_value(static_cast<float>(constant), gpu::to_mlx_dtype(dt));
    auto result = ::mlx::core::pad(*gs.arr, mlx_pad, pad_value);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

CpuStorage
GpuBackend::permute_cpu(const CpuStorage&, const Shape&, const std::vector<int>&, Dtype) {
    ErrorBuilder("gpu::permute_cpu")
        .not_implemented("GPU tensordot uses MLX einsum; this method is CPU-only");
    return CpuStorage{};
}

}  // namespace backend
}  // namespace lucid
