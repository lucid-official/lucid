// lucid/_C/backend/cpu/methods/Layout.cpp
// Out-of-line CpuBackend method definitions for the Layout family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

Storage CpuBackend::reverse_along_axis(const Storage& a, const Shape& shape, int axis, Dtype dt) {
    const auto& cs = std::get<CpuStorage>(a);
    auto ptr = allocate_aligned_bytes(cs.nbytes, Device::CPU);
    const std::size_t elem = dtype_size(dt);
    const int ndim = static_cast<int>(shape.size());
    std::size_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    for (int d = axis + 1; d < ndim; ++d)
        inner *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    const std::size_t L = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t k = 0; k < L; ++k) {
            std::memcpy(ptr.get() + ((o * L + k) * inner) * elem,
                        cs.ptr.get() + ((o * L + (L - 1 - k)) * inner) * elem, inner * elem);
        }
    }
    return Storage{CpuStorage{ptr, cs.nbytes, dt}};
}

Storage CpuBackend::diagonal(
    const Storage& a, const Shape& input_shape, int offset, int axis1, int axis2, Dtype dt) {
    const auto& as = std::get<CpuStorage>(a);
    const std::size_t ndim = input_shape.size();
    int a1 = axis1;
    int a2 = axis2;
    if (a1 > a2)
        std::swap(a1, a2);

    const std::int64_t M = input_shape[static_cast<std::size_t>(a1)];
    const std::int64_t N = input_shape[static_cast<std::size_t>(a2)];
    const std::int64_t r0 = (offset >= 0) ? 0 : -offset;
    const std::int64_t c0 = (offset >= 0) ? offset : 0;
    const std::int64_t L = std::max<std::int64_t>(0, std::min(M - r0, N - c0));

    Shape out_shape;
    for (std::size_t d = 0; d < ndim; ++d) {
        if (static_cast<int>(d) == a1 || static_cast<int>(d) == a2)
            continue;
        out_shape.push_back(input_shape[d]);
    }
    out_shape.push_back(L);
    std::size_t nb = shape_numel(out_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const std::size_t elem = dtype_size(dt);

    Stride a_stride(ndim);
    if (ndim > 0) {
        a_stride.back() = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d)
            a_stride[static_cast<std::size_t>(d)] = a_stride[static_cast<std::size_t>(d) + 1] *
                                                    input_shape[static_cast<std::size_t>(d) + 1];
    }

    std::vector<std::size_t> outer_dims;
    for (std::size_t d = 0; d < ndim; ++d)
        if (static_cast<int>(d) != a1 && static_cast<int>(d) != a2)
            outer_dims.push_back(d);

    std::size_t outer_numel = 1;
    for (auto d : outer_dims)
        outer_numel *= static_cast<std::size_t>(input_shape[d]);

    std::vector<std::int64_t> coord(ndim, 0);
    for (std::size_t o = 0; o < outer_numel; ++o) {
        std::size_t rem = o;
        for (auto d : outer_dims) {
            std::size_t prod = 1;
            for (std::size_t e : outer_dims)
                if (e > d)
                    prod *= static_cast<std::size_t>(input_shape[e]);
            coord[d] = static_cast<std::int64_t>(rem / prod);
            rem %= prod;
        }
        for (std::int64_t i = 0; i < L; ++i) {
            coord[static_cast<std::size_t>(a1)] = r0 + i;
            coord[static_cast<std::size_t>(a2)] = c0 + i;
            std::size_t a_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d)
                a_flat +=
                    static_cast<std::size_t>(coord[d]) * static_cast<std::size_t>(a_stride[d]);
            const std::size_t out_flat =
                o * static_cast<std::size_t>(L) + static_cast<std::size_t>(i);
            std::memcpy(ptr.get() + out_flat * elem, as.ptr.get() + a_flat * elem, elem);
        }
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::diagonal_backward(const Storage& grad,
                                      const Shape& input_shape,
                                      const Shape& output_shape,
                                      int offset,
                                      int axis1,
                                      int axis2,
                                      Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(diagonal_backward(detail::as_f32(grad), input_shape,
                                                     output_shape, offset, axis1, axis2,
                                                     Dtype::F32),
                                   dt);
    const auto& g = std::get<CpuStorage>(grad);
    auto out = zeros(input_shape, dt);
    const std::size_t ndim = input_shape.size();
    int a1 = axis1;
    int a2 = axis2;
    if (a1 > a2)
        std::swap(a1, a2);

    const std::int64_t M = input_shape[static_cast<std::size_t>(a1)];
    const std::int64_t N = input_shape[static_cast<std::size_t>(a2)];
    const std::int64_t r0 = (offset >= 0) ? 0 : -offset;
    const std::int64_t c0 = (offset >= 0) ? offset : 0;
    const std::int64_t L = std::max<std::int64_t>(0, std::min(M - r0, N - c0));

    Stride input_stride(ndim);
    if (ndim > 0) {
        input_stride.back() = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
            input_stride[static_cast<std::size_t>(d)] =
                input_stride[static_cast<std::size_t>(d) + 1] *
                input_shape[static_cast<std::size_t>(d) + 1];
        }
    }

    std::vector<std::size_t> outer_dims;
    for (std::size_t d = 0; d < ndim; ++d) {
        if (static_cast<int>(d) != a1 && static_cast<int>(d) != a2)
            outer_dims.push_back(d);
    }
    std::size_t outer_numel = 1;
    for (auto d : outer_dims)
        outer_numel *= static_cast<std::size_t>(input_shape[d]);

    std::vector<std::int64_t> coord(ndim, 0);
    if (dt == Dtype::F32) {
        const auto* gp = reinterpret_cast<const float*>(g.ptr.get());
        auto* dst = reinterpret_cast<float*>(std::get<CpuStorage>(out).ptr.get());
        for (std::size_t o = 0; o < outer_numel; ++o) {
            std::size_t rem = o;
            for (auto d : outer_dims) {
                std::size_t prod = 1;
                for (std::size_t e : outer_dims)
                    if (e > d)
                        prod *= static_cast<std::size_t>(input_shape[e]);
                coord[d] = static_cast<std::int64_t>(rem / prod);
                rem %= prod;
            }
            for (std::int64_t i = 0; i < L; ++i) {
                coord[static_cast<std::size_t>(a1)] = r0 + i;
                coord[static_cast<std::size_t>(a2)] = c0 + i;
                std::size_t input_flat = 0;
                for (std::size_t d = 0; d < ndim; ++d)
                    input_flat += static_cast<std::size_t>(coord[d]) *
                                  static_cast<std::size_t>(input_stride[d]);
                const std::size_t grad_flat =
                    o * static_cast<std::size_t>(L) + static_cast<std::size_t>(i);
                dst[input_flat] += gp[grad_flat];
            }
        }
    } else if (dt == Dtype::F64) {
        const auto* gp = reinterpret_cast<const double*>(g.ptr.get());
        auto* dst = reinterpret_cast<double*>(std::get<CpuStorage>(out).ptr.get());
        for (std::size_t o = 0; o < outer_numel; ++o) {
            std::size_t rem = o;
            for (auto d : outer_dims) {
                std::size_t prod = 1;
                for (std::size_t e : outer_dims)
                    if (e > d)
                        prod *= static_cast<std::size_t>(input_shape[e]);
                coord[d] = static_cast<std::int64_t>(rem / prod);
                rem %= prod;
            }
            for (std::int64_t i = 0; i < L; ++i) {
                coord[static_cast<std::size_t>(a1)] = r0 + i;
                coord[static_cast<std::size_t>(a2)] = c0 + i;
                std::size_t input_flat = 0;
                for (std::size_t d = 0; d < ndim; ++d)
                    input_flat += static_cast<std::size_t>(coord[d]) *
                                  static_cast<std::size_t>(input_stride[d]);
                const std::size_t grad_flat =
                    o * static_cast<std::size_t>(L) + static_cast<std::size_t>(i);
                dst[input_flat] += gp[grad_flat];
            }
        }
    } else {
        ErrorBuilder("cpu_backend::diagonal_backward").not_implemented("dtype not supported");
    }
    return out;
}

Storage CpuBackend::roll(const Storage& a,
                         const Shape& shape,
                         Dtype dt,
                         const std::vector<std::int64_t>& shifts,
                         const std::vector<int>& axes) {
    const auto& as = std::get<CpuStorage>(a);
    const std::size_t ndim = shape.size();
    const std::size_t elem = dtype_size(dt);
    const std::size_t total = shape_numel(shape);
    const std::size_t nb = total * elem;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (total == 0)
        return Storage{CpuStorage{ptr, nb, dt}};

    std::vector<std::int64_t> shift_per_dim(ndim, 0);
    for (std::size_t i = 0; i < axes.size(); ++i) {
        int ax = axes[i];
        if (ax < 0)
            ax += static_cast<int>(ndim);
        shift_per_dim[static_cast<std::size_t>(ax)] += shifts[i];
    }
    bool any_shift = false;
    for (std::size_t d = 0; d < ndim; ++d) {
        const std::int64_t L = shape[d];
        shift_per_dim[d] = L > 0 ? ((shift_per_dim[d] % L) + L) % L : 0;
        if (shift_per_dim[d] != 0)
            any_shift = true;
    }
    if (!any_shift) {
        std::memcpy(ptr.get(), as.ptr.get(), nb);
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Stride stride(ndim);
    stride.back() = 1;
    for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d)
        stride[static_cast<std::size_t>(d)] =
            stride[static_cast<std::size_t>(d) + 1] * shape[static_cast<std::size_t>(d) + 1];

    const std::size_t row_len = static_cast<std::size_t>(shape.back());
    const std::size_t s_last = static_cast<std::size_t>(shift_per_dim.back());
    const std::size_t tail_len = row_len - s_last;
    const std::size_t rows = total / row_len;

    std::vector<std::int64_t> coord(ndim > 1 ? ndim - 1 : 0, 0);
    for (std::size_t r = 0; r < rows; ++r) {
        std::size_t in_row_base = 0;
        for (std::size_t d = 0; d + 1 < ndim; ++d) {
            const std::int64_t L = shape[d];
            std::int64_t c = coord[d] - shift_per_dim[d];
            if (c < 0)
                c += L;
            in_row_base += static_cast<std::size_t>(c) * static_cast<std::size_t>(stride[d]);
        }
        auto* dst_row = ptr.get() + r * row_len * elem;
        const auto* src_row = as.ptr.get() + in_row_base * elem;
        if (s_last == 0) {
            std::memcpy(dst_row, src_row, row_len * elem);
        } else {
            std::memcpy(dst_row, src_row + tail_len * elem, s_last * elem);
            std::memcpy(dst_row + s_last * elem, src_row, tail_len * elem);
        }
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
            if (++coord[static_cast<std::size_t>(d)] < shape[static_cast<std::size_t>(d)])
                break;
            coord[static_cast<std::size_t>(d)] = 0;
        }
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::reshape(const Storage& a, const Shape&, const Shape&, Dtype dt) {
    const auto& as = std::get<CpuStorage>(a);
    auto ptr = allocate_aligned_bytes(as.nbytes, Device::CPU);
    if (as.nbytes > 0)
        std::memcpy(ptr.get(), as.ptr.get(), as.nbytes);
    return Storage{CpuStorage{ptr, as.nbytes, dt}};
}

Storage CpuBackend::slice_axis(const Storage& a,
                               const Shape& src_shape,
                               const Shape& slice_shape,
                               int axis,
                               std::int64_t offset,
                               Dtype dt) {
    const auto& as = std::get<CpuStorage>(a);
    std::size_t out_nbytes = shape_numel(slice_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
    const std::size_t elem = dtype_size(dt);
    std::size_t outer = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(src_shape[static_cast<std::size_t>(d)]);
    std::size_t inner_bytes = elem;
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < src_shape.size(); ++d)
        inner_bytes *= static_cast<std::size_t>(src_shape[d]);
    const std::size_t src_axis =
        static_cast<std::size_t>(src_shape[static_cast<std::size_t>(axis)]);
    const std::size_t slice_axis =
        static_cast<std::size_t>(slice_shape[static_cast<std::size_t>(axis)]);
    const std::size_t copy_bytes = slice_axis * inner_bytes;
    for (std::size_t o = 0; o < outer; ++o) {
        const auto* src_ptr =
            as.ptr.get() + (o * src_axis + static_cast<std::size_t>(offset)) * inner_bytes;
        auto* dst_ptr = ptr.get() + o * copy_bytes;
        if (copy_bytes > 0)
            std::memcpy(dst_ptr, src_ptr, copy_bytes);
    }
    return Storage{CpuStorage{ptr, out_nbytes, dt}};
}

Storage CpuBackend::insert_axis_slice(const Storage& a,
                                      const Shape& src_shape,
                                      const Shape& dst_shape,
                                      int axis,
                                      std::int64_t offset,
                                      Dtype dt) {
    const auto& as = std::get<CpuStorage>(a);
    std::size_t out_nbytes = shape_numel(dst_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
    std::memset(ptr.get(), 0, out_nbytes);
    const std::size_t elem = dtype_size(dt);
    std::size_t outer = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(dst_shape[static_cast<std::size_t>(d)]);
    std::size_t inner_bytes = elem;
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < dst_shape.size(); ++d)
        inner_bytes *= static_cast<std::size_t>(dst_shape[d]);
    const std::size_t dst_axis =
        static_cast<std::size_t>(dst_shape[static_cast<std::size_t>(axis)]);
    const std::size_t src_axis =
        static_cast<std::size_t>(src_shape[static_cast<std::size_t>(axis)]);
    const std::size_t copy_bytes = src_axis * inner_bytes;
    for (std::size_t o = 0; o < outer; ++o) {
        const auto* src_ptr = as.ptr.get() + o * copy_bytes;
        auto* dst_ptr = ptr.get() + (o * dst_axis + static_cast<std::size_t>(offset)) * inner_bytes;
        if (copy_bytes > 0)
            std::memcpy(dst_ptr, src_ptr, copy_bytes);
    }
    return Storage{CpuStorage{ptr, out_nbytes, dt}};
}

Storage CpuBackend::concatenate(const std::vector<Storage>& xs,
                                const std::vector<Shape>& shapes,
                                int axis,
                                Dtype dt) {
    Shape out_shape = shapes.front();
    std::int64_t cat_dim = 0;
    for (const auto& shape : shapes)
        cat_dim += shape[static_cast<std::size_t>(axis)];
    out_shape[static_cast<std::size_t>(axis)] = cat_dim;

    std::size_t out_nbytes = shape_numel(out_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
    const std::size_t elem = dtype_size(dt);
    std::size_t outer = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(out_shape[static_cast<std::size_t>(d)]);
    std::size_t inner_per_unit = elem;
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < out_shape.size(); ++d)
        inner_per_unit *= static_cast<std::size_t>(out_shape[d]);

    auto* dst = ptr.get();
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < xs.size(); ++i) {
            const auto& cs = std::get<CpuStorage>(xs[i]);
            const std::size_t L =
                static_cast<std::size_t>(shapes[i][static_cast<std::size_t>(axis)]);
            const std::size_t bytes = L * inner_per_unit;
            std::memcpy(dst, cs.ptr.get() + o * bytes, bytes);
            dst += bytes;
        }
    }
    return Storage{CpuStorage{ptr, out_nbytes, dt}};
}

Storage
CpuBackend::stack(const std::vector<Storage>& xs, const Shape& input_shape, int axis, Dtype dt) {
    Shape out_shape = input_shape;
    out_shape.insert(out_shape.begin() + axis, static_cast<std::int64_t>(xs.size()));
    std::size_t out_nbytes = shape_numel(out_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
    const std::size_t elem = dtype_size(dt);
    std::size_t outer = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(out_shape[static_cast<std::size_t>(d)]);
    std::size_t inner_bytes = elem;
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < out_shape.size(); ++d)
        inner_bytes *= static_cast<std::size_t>(out_shape[d]);
    const std::size_t block_bytes = static_cast<std::size_t>(xs.size()) * inner_bytes;
    for (std::size_t idx = 0; idx < xs.size(); ++idx) {
        const auto& cs = std::get<CpuStorage>(xs[idx]);
        for (std::size_t o = 0; o < outer; ++o) {
            std::memcpy(ptr.get() + o * block_bytes + idx * inner_bytes,
                        cs.ptr.get() + o * inner_bytes, inner_bytes);
        }
    }
    return Storage{CpuStorage{ptr, out_nbytes, dt}};
}

std::vector<Storage> CpuBackend::split_equal(
    const Storage& a, const Shape& shape, int axis, std::int64_t num_splits, Dtype dt) {
    std::vector<Storage> out;
    out.reserve(static_cast<std::size_t>(num_splits));
    Shape piece_shape = shape;
    const std::int64_t piece = shape[static_cast<std::size_t>(axis)] / num_splits;
    piece_shape[static_cast<std::size_t>(axis)] = piece;
    for (std::int64_t k = 0; k < num_splits; ++k) {
        out.push_back(slice_axis(a, shape, piece_shape, axis, k * piece, dt));
    }
    return out;
}

std::vector<Storage> CpuBackend::split_at(const Storage& a,
                                          const Shape& shape,
                                          int axis,
                                          const std::vector<std::int64_t>& indices,
                                          Dtype dt) {
    std::vector<Storage> out;
    out.reserve(indices.size() + 1);
    std::int64_t off = 0;
    for (auto idx : indices) {
        Shape piece_shape = shape;
        piece_shape[static_cast<std::size_t>(axis)] = idx - off;
        out.push_back(slice_axis(a, shape, piece_shape, axis, off, dt));
        off = idx;
    }
    Shape tail_shape = shape;
    tail_shape[static_cast<std::size_t>(axis)] = shape[static_cast<std::size_t>(axis)] - off;
    out.push_back(slice_axis(a, shape, tail_shape, axis, off, dt));
    return out;
}

Storage CpuBackend::repeat_backward(const Storage& grad_out,
                                    const Shape& input_shape,
                                    const Shape& output_shape,
                                    int axis,
                                    std::int64_t repeats,
                                    Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(repeat_backward(detail::as_f32(grad_out), input_shape,
                                                   output_shape, axis, repeats, Dtype::F32),
                                   dt);
    const auto& g = std::get<CpuStorage>(grad_out);
    std::size_t out_nbytes = shape_numel(input_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
    std::memset(ptr.get(), 0, out_nbytes);

    const std::size_t ndim = output_shape.size();
    Stride out_stride(ndim), in_stride(ndim);
    if (ndim > 0) {
        out_stride.back() = 1;
        in_stride.back() = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
            out_stride[static_cast<std::size_t>(d)] = out_stride[static_cast<std::size_t>(d) + 1] *
                                                      output_shape[static_cast<std::size_t>(d) + 1];
            in_stride[static_cast<std::size_t>(d)] = in_stride[static_cast<std::size_t>(d) + 1] *
                                                     input_shape[static_cast<std::size_t>(d) + 1];
        }
    }

    const std::size_t total = shape_numel(output_shape);
    if (dt == Dtype::F32) {
        auto* out = reinterpret_cast<float*>(ptr.get());
        const auto* gp = reinterpret_cast<const float*>(g.ptr.get());
        for (std::size_t flat = 0; flat < total; ++flat) {
            std::size_t rem = flat;
            std::size_t in_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                rem -= coord * static_cast<std::size_t>(out_stride[d]);
                if (static_cast<int>(d) == axis)
                    coord /= static_cast<std::size_t>(repeats);
                in_flat += coord * static_cast<std::size_t>(in_stride[d]);
            }
            out[in_flat] += gp[flat];
        }
    } else if (dt == Dtype::F64) {
        auto* out = reinterpret_cast<double*>(ptr.get());
        const auto* gp = reinterpret_cast<const double*>(g.ptr.get());
        for (std::size_t flat = 0; flat < total; ++flat) {
            std::size_t rem = flat;
            std::size_t in_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                rem -= coord * static_cast<std::size_t>(out_stride[d]);
                if (static_cast<int>(d) == axis)
                    coord /= static_cast<std::size_t>(repeats);
                in_flat += coord * static_cast<std::size_t>(in_stride[d]);
            }
            out[in_flat] += gp[flat];
        }
    } else if (dt == Dtype::I32) {
        auto* out = reinterpret_cast<std::int32_t*>(ptr.get());
        const auto* gp = reinterpret_cast<const std::int32_t*>(g.ptr.get());
        for (std::size_t flat = 0; flat < total; ++flat) {
            std::size_t rem = flat;
            std::size_t in_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                rem -= coord * static_cast<std::size_t>(out_stride[d]);
                if (static_cast<int>(d) == axis)
                    coord /= static_cast<std::size_t>(repeats);
                in_flat += coord * static_cast<std::size_t>(in_stride[d]);
            }
            out[in_flat] += gp[flat];
        }
    } else if (dt == Dtype::I64) {
        auto* out = reinterpret_cast<std::int64_t*>(ptr.get());
        const auto* gp = reinterpret_cast<const std::int64_t*>(g.ptr.get());
        for (std::size_t flat = 0; flat < total; ++flat) {
            std::size_t rem = flat;
            std::size_t in_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                rem -= coord * static_cast<std::size_t>(out_stride[d]);
                if (static_cast<int>(d) == axis)
                    coord /= static_cast<std::size_t>(repeats);
                in_flat += coord * static_cast<std::size_t>(in_stride[d]);
            }
            out[in_flat] += gp[flat];
        }
    } else {
        ErrorBuilder("cpu_backend::repeat_backward").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, out_nbytes, dt}};
}

Storage CpuBackend::tile_backward(const Storage& grad_out,
                                  const Shape& input_shape,
                                  const Shape& padded_shape,
                                  const Shape& output_shape,
                                  const std::vector<std::int64_t>& p4_,
                                  Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(tile_backward(detail::as_f32(grad_out), input_shape,
                                                 padded_shape, output_shape, p4_, Dtype::F32),
                                   dt);
    const auto& g = std::get<CpuStorage>(grad_out);
    std::size_t out_nbytes = shape_numel(input_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
    std::memset(ptr.get(), 0, out_nbytes);

    const std::size_t ndim = output_shape.size();
    Stride out_stride(ndim), padded_stride(ndim);
    if (ndim > 0) {
        out_stride.back() = 1;
        padded_stride.back() = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
            out_stride[static_cast<std::size_t>(d)] = out_stride[static_cast<std::size_t>(d) + 1] *
                                                      output_shape[static_cast<std::size_t>(d) + 1];
            padded_stride[static_cast<std::size_t>(d)] =
                padded_stride[static_cast<std::size_t>(d) + 1] *
                padded_shape[static_cast<std::size_t>(d) + 1];
        }
    }

    const std::size_t total = shape_numel(output_shape);
    if (dt == Dtype::F32) {
        auto* out = reinterpret_cast<float*>(ptr.get());
        const auto* gp = reinterpret_cast<const float*>(g.ptr.get());
        for (std::size_t flat = 0; flat < total; ++flat) {
            std::size_t rem = flat;
            std::size_t in_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                const std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                rem -= coord * static_cast<std::size_t>(out_stride[d]);
                const std::size_t in_coord = coord % static_cast<std::size_t>(padded_shape[d]);
                in_flat += in_coord * static_cast<std::size_t>(padded_stride[d]);
            }
            out[in_flat] += gp[flat];
        }
    } else if (dt == Dtype::F64) {
        auto* out = reinterpret_cast<double*>(ptr.get());
        const auto* gp = reinterpret_cast<const double*>(g.ptr.get());
        for (std::size_t flat = 0; flat < total; ++flat) {
            std::size_t rem = flat;
            std::size_t in_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                const std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                rem -= coord * static_cast<std::size_t>(out_stride[d]);
                const std::size_t in_coord = coord % static_cast<std::size_t>(padded_shape[d]);
                in_flat += in_coord * static_cast<std::size_t>(padded_stride[d]);
            }
            out[in_flat] += gp[flat];
        }
    } else if (dt == Dtype::I32) {
        auto* out = reinterpret_cast<std::int32_t*>(ptr.get());
        const auto* gp = reinterpret_cast<const std::int32_t*>(g.ptr.get());
        for (std::size_t flat = 0; flat < total; ++flat) {
            std::size_t rem = flat;
            std::size_t in_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                const std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                rem -= coord * static_cast<std::size_t>(out_stride[d]);
                const std::size_t in_coord = coord % static_cast<std::size_t>(padded_shape[d]);
                in_flat += in_coord * static_cast<std::size_t>(padded_stride[d]);
            }
            out[in_flat] += gp[flat];
        }
    } else if (dt == Dtype::I64) {
        auto* out = reinterpret_cast<std::int64_t*>(ptr.get());
        const auto* gp = reinterpret_cast<const std::int64_t*>(g.ptr.get());
        for (std::size_t flat = 0; flat < total; ++flat) {
            std::size_t rem = flat;
            std::size_t in_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                const std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                rem -= coord * static_cast<std::size_t>(out_stride[d]);
                const std::size_t in_coord = coord % static_cast<std::size_t>(padded_shape[d]);
                in_flat += in_coord * static_cast<std::size_t>(padded_stride[d]);
            }
            out[in_flat] += gp[flat];
        }
    } else {
        ErrorBuilder("cpu_backend::tile_backward").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, out_nbytes, dt}};
}

Storage CpuBackend::unfold_dim(
    const Storage& a, const Shape& in_shape, int dim, int size, int step, Dtype dt) {
    const auto& ca = std::get<CpuStorage>(a);
    const int ndim = static_cast<int>(in_shape.size());
    if (dim < 0)
        dim += ndim;

    const std::size_t dim_size = static_cast<std::size_t>(in_shape[static_cast<std::size_t>(dim)]);
    const std::size_t L =
        (dim_size - static_cast<std::size_t>(size)) / static_cast<std::size_t>(step) + 1;

    // out_shape = in_shape[:dim] + [L] + in_shape[dim+1:] + [size]
    Shape out_shape;
    for (int d = 0; d < dim; ++d)
        out_shape.push_back(in_shape[static_cast<std::size_t>(d)]);
    out_shape.push_back(static_cast<std::int64_t>(L));
    for (int d = dim + 1; d < ndim; ++d)
        out_shape.push_back(in_shape[static_cast<std::size_t>(d)]);
    out_shape.push_back(static_cast<std::int64_t>(size));

    const std::size_t out_n = shape_numel(out_shape);
    const std::size_t nbytes = out_n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);

    // outer = product of dims before dim
    std::size_t outer = 1;
    for (int d = 0; d < dim; ++d)
        outer *= static_cast<std::size_t>(in_shape[static_cast<std::size_t>(d)]);
    // inner = product of dims after dim (in the original tensor)
    std::size_t inner = 1;
    for (int d = dim + 1; d < ndim; ++d)
        inner *= static_cast<std::size_t>(in_shape[static_cast<std::size_t>(d)]);

    const std::size_t esz = dtype_size(dt);
    const std::uint8_t* src_ptr =
        static_cast<const std::uint8_t*>(static_cast<const void*>(ca.ptr.get()));
    std::uint8_t* dst_ptr = static_cast<std::uint8_t*>(static_cast<void*>(ptr.get()));

    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t l = 0; l < L; ++l) {
            for (std::size_t j = 0; j < inner; ++j) {
                for (std::size_t s = 0; s < static_cast<std::size_t>(size); ++s) {
                    // Source element: [o, l*step+s, j] in the input
                    const std::size_t src_dim_pos = l * static_cast<std::size_t>(step) + s;
                    const std::size_t src_flat = (o * dim_size + src_dim_pos) * inner + j;
                    // Destination: [o, l, j, s] in the output
                    const std::size_t dst_flat =
                        ((o * L + l) * inner + j) * static_cast<std::size_t>(size) + s;
                    std::memcpy(dst_ptr + dst_flat * esz, src_ptr + src_flat * esz, esz);
                }
            }
        }
    }
    return Storage{CpuStorage{ptr, nbytes, dt}};
}

Storage
CpuBackend::flip(const Storage& a, const Shape& shape, const std::vector<int>& dims, Dtype dt) {
    const auto& ca = std::get<CpuStorage>(a);
    const std::size_t n = shape_numel(shape);
    const std::size_t esz = dtype_size(dt);
    const std::size_t nb = n * esz;
    auto out_ptr = allocate_aligned_bytes(nb, Device::CPU);

    // Compute strides (row-major).
    const int ndim = static_cast<int>(shape.size());
    std::vector<std::size_t> strides(ndim, 1);
    for (int d = ndim - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * static_cast<std::size_t>(shape[d + 1]);

    // Build a set of dims to flip.
    std::vector<bool> flip_dim(ndim, false);
    for (int d : dims)
        flip_dim[d] = true;

    const std::uint8_t* src =
        static_cast<const std::uint8_t*>(static_cast<const void*>(ca.ptr.get()));
    std::uint8_t* dst = static_cast<std::uint8_t*>(static_cast<void*>(out_ptr.get()));

    // Iterate over all elements; for each, compute destination index after flipping.
    for (std::size_t flat = 0; flat < n; ++flat) {
        std::size_t rem = flat;
        std::size_t src_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            std::size_t coord = rem / strides[d];
            rem %= strides[d];
            std::size_t flipped =
                flip_dim[d] ? static_cast<std::size_t>(shape[d]) - 1 - coord : coord;
            src_idx += flipped * strides[d];
        }
        std::memcpy(dst + flat * esz, src + src_idx * esz, esz);
    }
    return Storage{CpuStorage{out_ptr, nb, dt}};
}

Storage
CpuBackend::broadcast(const Storage& a, const Shape& src_shape, const Shape& dst_shape, Dtype dt) {
    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t out_n = shape_numel(dst_shape);
    std::size_t nb = out_n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const BroadcastPlan plan = plan_broadcast(src_shape, dst_shape);

    auto run = [&](auto tag) {
        using T = decltype(tag);
        broadcast_runs(plan, reinterpret_cast<const T*>(cs.ptr.get()),
                       reinterpret_cast<T*>(ptr.get()));
    };

    switch (dt) {
    case Dtype::F32:
        run(float{});
        break;
    case Dtype::F64:
        run(double{});
        break;
    case Dtype::I32:
        run(std::int32_t{});
        break;
    case Dtype::I64:
        run(std::int64_t{});
        break;
    case Dtype::I16:
        run(std::int16_t{});
        break;
    case Dtype::I8:
    case Dtype::Bool:
        run(std::uint8_t{});
        break;
    case Dtype::F16:
    case Dtype::BF16:
        // Broadcasting replicates elements without reading them, so both
        // 16-bit floats ride the 16-bit path and complex64 the 64-bit one.
        run(std::uint16_t{});
        break;
    case Dtype::C64:
        run(std::uint64_t{});
        break;
    case Dtype::C128:
        run(std::complex<double>{});
        break;
    default:
        ErrorBuilder("cpu_backend::broadcast").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage
CpuBackend::repeat(const Storage& a, const Shape& shape, Dtype dt, std::int64_t repeats, int axis) {
    const auto& cs = std::get<CpuStorage>(a);
    Shape out_shape = shape;
    out_shape[static_cast<std::size_t>(axis)] *= repeats;
    std::size_t nb = shape_numel(out_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const std::size_t elem = dtype_size(dt);
    std::size_t outer = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    std::size_t inner = elem;
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d)
        inner *= static_cast<std::size_t>(shape[d]);
    const std::size_t L = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
    auto* dst = ptr.get();
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t k = 0; k < L; ++k) {
            const auto* src = cs.ptr.get() + (o * L + k) * inner;
            for (std::int64_t r = 0; r < repeats; ++r) {
                std::memcpy(dst, src, inner);
                dst += inner;
            }
        }
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::tile(const Storage& a,
                         const Shape& shape,
                         Dtype dt,
                         const std::vector<std::int64_t>& reps) {
    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t nout = reps.size();
    Shape padded(nout, 1);
    const std::size_t lead = nout - shape.size();
    for (std::size_t d = 0; d < shape.size(); ++d)
        padded[lead + d] = shape[d];
    Shape out_shape(nout);
    for (std::size_t d = 0; d < nout; ++d)
        out_shape[d] = padded[d] * reps[d];

    std::size_t nb = shape_numel(out_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const std::size_t elem = dtype_size(dt);
    Stride in_stride(nout);
    if (nout > 0) {
        in_stride.back() = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(nout) - 2; d >= 0; --d)
            in_stride[static_cast<std::size_t>(d)] = in_stride[static_cast<std::size_t>(d) + 1] *
                                                     padded[static_cast<std::size_t>(d) + 1];
    }
    const std::size_t total = shape_numel(out_shape);
    std::vector<std::int64_t> coord(nout, 0);
    for (std::size_t f = 0; f < total; ++f) {
        std::size_t in_flat = 0;
        for (std::size_t d = 0; d < nout; ++d)
            in_flat += static_cast<std::size_t>(coord[d] % padded[d]) *
                       static_cast<std::size_t>(in_stride[d]);
        std::memcpy(ptr.get() + f * elem, cs.ptr.get() + in_flat * elem, elem);
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(nout) - 1; d >= 0; --d) {
            if (++coord[static_cast<std::size_t>(d)] < out_shape[static_cast<std::size_t>(d)])
                break;
            coord[static_cast<std::size_t>(d)] = 0;
        }
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage
CpuBackend::permute(const Storage& a, const Shape& shape, const std::vector<int>& perm, Dtype dt) {
    const auto& cs = std::get<CpuStorage>(a);
    Shape out_shape;
    out_shape.reserve(perm.size());
    for (int p : perm)
        out_shape.push_back(shape[static_cast<std::size_t>(p)]);
    std::size_t nb = shape_numel(out_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (nb == 0)
        return Storage{CpuStorage{ptr, nb, dt}};
    cpu::permute_copy(cs.ptr.get(), ptr.get(), shape, perm, dtype_size(dt));
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::pad(const Storage& a,
                        const Shape& shape,
                        Dtype dt,
                        const std::vector<std::pair<std::int64_t, std::int64_t>>& pad_width,
                        double constant) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(pad(detail::as_f32(a), shape, Dtype::F32, pad_width, constant),
                                   dt);
    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t ndim = shape.size();
    Shape out_shape(ndim);
    for (std::size_t d = 0; d < ndim; ++d)
        out_shape[d] = shape[d] + pad_width[d].first + pad_width[d].second;

    std::size_t nb = shape_numel(out_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    detail::fill_pad_buffer(ptr.get(), nb, shape_numel(out_shape), dt, constant);

    const std::size_t elem = dtype_size(dt);
    Stride in_stride(ndim), out_stride(ndim);
    if (ndim > 0) {
        in_stride.back() = 1;
        out_stride.back() = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
            in_stride[static_cast<std::size_t>(d)] =
                in_stride[static_cast<std::size_t>(d) + 1] * shape[static_cast<std::size_t>(d) + 1];
            out_stride[static_cast<std::size_t>(d)] = out_stride[static_cast<std::size_t>(d) + 1] *
                                                      out_shape[static_cast<std::size_t>(d) + 1];
        }
    }
    const std::size_t row_in = static_cast<std::size_t>(shape.back());
    const std::size_t row_bytes = row_in * elem;
    const std::size_t in_numel = shape_numel(shape);
    const std::size_t rows = in_numel / row_in;

    std::vector<std::int64_t> coord(ndim - 1, 0);
    for (std::size_t r = 0; r < rows; ++r) {
        std::size_t out_off = static_cast<std::size_t>(pad_width.back().first);
        for (std::size_t d = 0; d + 1 < ndim; ++d)
            out_off += static_cast<std::size_t>(coord[d] + pad_width[d].first) *
                       static_cast<std::size_t>(out_stride[d]);
        std::size_t in_off = 0;
        for (std::size_t d = 0; d + 1 < ndim; ++d)
            in_off += static_cast<std::size_t>(coord[d]) * static_cast<std::size_t>(in_stride[d]);
        std::memcpy(ptr.get() + out_off * elem, cs.ptr.get() + in_off * elem, row_bytes);
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
            if (++coord[static_cast<std::size_t>(d)] < shape[static_cast<std::size_t>(d)])
                break;
            coord[static_cast<std::size_t>(d)] = 0;
        }
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

CpuStorage CpuBackend::permute_cpu(const CpuStorage& src,
                                   const Shape& src_shape,
                                   const std::vector<int>& perm,
                                   Dtype dt) {
    Shape dst_shape;
    for (int p : perm)
        dst_shape.push_back(src_shape[static_cast<std::size_t>(p)]);
    const std::size_t nb = shape_numel(dst_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    if (nb > 0)
        cpu::permute_copy(src.ptr.get(), ptr.get(), src_shape, perm, dtype_size(dt));
    return CpuStorage{ptr, nb, dt};
}

}  // namespace backend
}  // namespace lucid
