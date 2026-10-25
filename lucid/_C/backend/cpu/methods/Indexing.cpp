// lucid/_C/backend/cpu/methods/Indexing.cpp
// Out-of-line CpuBackend method definitions for the Indexing family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

Storage CpuBackend::gather(const Storage& a,
                           const Storage& indices,
                           const Shape& input_shape,
                           const Shape& output_shape,
                           int axis,
                           Dtype index_dtype,
                           Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(gather(detail::as_f32(a), detail::as_f32(indices), input_shape,
                                          output_shape, axis, index_dtype, Dtype::F32),
                                   dt);
    const auto& as = std::get<CpuStorage>(a);
    const auto& is = std::get<CpuStorage>(indices);
    std::size_t nb = shape_numel(output_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const std::size_t ndim = input_shape.size();
    const std::size_t elem = dtype_size(dt);

    Stride a_stride(ndim), out_stride(ndim);
    if (ndim > 0) {
        a_stride.back() = 1;
        out_stride.back() = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
            a_stride[static_cast<std::size_t>(d)] = a_stride[static_cast<std::size_t>(d) + 1] *
                                                    input_shape[static_cast<std::size_t>(d) + 1];
            out_stride[static_cast<std::size_t>(d)] = out_stride[static_cast<std::size_t>(d) + 1] *
                                                      output_shape[static_cast<std::size_t>(d) + 1];
        }
    }
    (void)index_dtype;
    const std::size_t total = shape_numel(output_shape);
    const auto load_idx = cpu::index_reader("gather", is);

    const std::int64_t axis_extent =
        ndim > 0 ? static_cast<std::int64_t>(input_shape[static_cast<std::size_t>(axis)]) : 0;
    // Checking the index alone is not enough.  Every *other* coordinate
    // walks the output extent while indexing the input's stride, so an
    // output wider than the input on any non-gathered axis reads past
    // the end however valid the indices are.  Gathering a (4, 5) index
    // along dim 1 of an *empty* (0, 5) operand has indices in [0, 5)
    // throughout and still walks four rows that do not exist.
    for (std::size_t d = 0; d < ndim; ++d) {
        if (static_cast<int>(d) == axis)
            continue;
        if (output_shape[d] > input_shape[d])
            ErrorBuilder("gather").fail("index is larger than the operand on a non-gathered axis");
    }

    // When every non-gathered axis of the output is as wide as the
    // input's — the usual case, and the only one ``index_select``
    // produces — output element (o, j, i) reads input (o, index, i).
    // That is two multiplies and an add on typed elements, instead of an
    // ndim-long coordinate walk and a memcpy call per element.  Through
    // the walk an index_select of a (3, 375, 500) image took 11 ms, and a
    // bicubic resize is eight of them (CHA-5).
    if (const auto rows = detail::GatherRows::of(input_shape, output_shape, axis)) {
        auto run = [&](auto* dst, const auto* src) {
            for (std::size_t o = 0; o < rows->outer; ++o) {
                for (std::size_t j = 0; j < rows->taken; ++j) {
                    const std::size_t row = (o * rows->taken + j) * rows->inner;
                    for (std::size_t i = 0; i < rows->inner; ++i) {
                        const std::int64_t k =
                            cpu::checked_axis("gather", load_idx(row + i), axis_extent,
                                              "index out of range for the gathered axis");
                        dst[row + i] =
                            src[(o * rows->extent + static_cast<std::size_t>(k)) * rows->inner + i];
                    }
                }
            }
        };
        bool copied = true;
        switch (elem) {
        case 1:
            run(reinterpret_cast<std::uint8_t*>(ptr.get()),
                reinterpret_cast<const std::uint8_t*>(as.ptr.get()));
            break;
        case 2:
            run(reinterpret_cast<std::uint16_t*>(ptr.get()),
                reinterpret_cast<const std::uint16_t*>(as.ptr.get()));
            break;
        case 4:
            run(reinterpret_cast<std::uint32_t*>(ptr.get()),
                reinterpret_cast<const std::uint32_t*>(as.ptr.get()));
            break;
        case 8:
            run(reinterpret_cast<std::uint64_t*>(ptr.get()),
                reinterpret_cast<const std::uint64_t*>(as.ptr.get()));
            break;
        default:
            copied = false;  // complex128: the walk below handles any width
        }
        if (copied)
            return Storage{CpuStorage{ptr, nb, dt}};
    }

    std::vector<std::int64_t> coord(ndim, 0);
    for (std::size_t out_flat = 0; out_flat < total; ++out_flat) {
        const std::int64_t k = cpu::checked_axis("gather", load_idx(out_flat), axis_extent,
                                                 "index out of range for the gathered axis");
        std::size_t a_flat = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            std::int64_t c = (static_cast<int>(d) == axis) ? k : coord[d];
            a_flat += static_cast<std::size_t>(c) * static_cast<std::size_t>(a_stride[d]);
        }
        std::memcpy(ptr.get() + out_flat * elem, as.ptr.get() + a_flat * elem, elem);
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d) {
            if (++coord[static_cast<std::size_t>(d)] < output_shape[static_cast<std::size_t>(d)]) {
                break;
            }
            coord[static_cast<std::size_t>(d)] = 0;
        }
    }
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::gather_backward(const Storage& grad,
                                    const Storage& indices,
                                    const Shape& input_shape,
                                    const Shape& output_shape,
                                    int axis,
                                    Dtype index_dtype,
                                    Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(gather_backward(detail::as_f32(grad), detail::as_f32(indices),
                                                   input_shape, output_shape, axis, index_dtype,
                                                   Dtype::F32),
                                   dt);
    const auto& g = std::get<CpuStorage>(grad);
    const auto& idx = std::get<CpuStorage>(indices);
    auto out = zeros(input_shape, dt);
    const std::size_t ndim = input_shape.size();
    Stride input_stride(ndim), output_stride(ndim);
    if (ndim > 0) {
        input_stride.back() = 1;
        output_stride.back() = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
            input_stride[static_cast<std::size_t>(d)] =
                input_stride[static_cast<std::size_t>(d) + 1] *
                input_shape[static_cast<std::size_t>(d) + 1];
            output_stride[static_cast<std::size_t>(d)] =
                output_stride[static_cast<std::size_t>(d) + 1] *
                output_shape[static_cast<std::size_t>(d) + 1];
        }
    }
    (void)index_dtype;
    const auto load_idx = cpu::index_reader("cpu_backend::gather_backward", idx);

    const std::size_t total = shape_numel(output_shape);
    // The forward pass reads at the index; this one *writes* at it, so
    // an out-of-range entry corrupts whatever follows the gradient
    // buffer rather than merely returning a wrong number.
    const std::int64_t axis_extent =
        ndim > 0 ? static_cast<std::int64_t>(input_shape[static_cast<std::size_t>(axis)]) : 0;
    // See the forward pass: the non-gathered coordinates walk the
    // output extent against the input's stride, so they need the same
    // shape check the index does.
    for (std::size_t d = 0; d < ndim; ++d) {
        if (static_cast<int>(d) == axis)
            continue;
        if (output_shape[d] > input_shape[d])
            ErrorBuilder("cpu_backend::gather_backward")
                .fail("index is larger than the operand on a non-gathered axis");
    }

    // The forward pass's fast case, accumulating instead of reading.
    if (const auto rows = detail::GatherRows::of(input_shape, output_shape, axis);
        rows && (dt == Dtype::F32 || dt == Dtype::F64)) {
        auto run = [&](auto* dst, const auto* gp) {
            for (std::size_t o = 0; o < rows->outer; ++o) {
                for (std::size_t j = 0; j < rows->taken; ++j) {
                    const std::size_t row = (o * rows->taken + j) * rows->inner;
                    for (std::size_t i = 0; i < rows->inner; ++i) {
                        const std::int64_t k = cpu::checked_axis(
                            "cpu_backend::gather_backward", load_idx(row + i), axis_extent,
                            "index out of range for the gathered axis");
                        dst[(o * rows->extent + static_cast<std::size_t>(k)) * rows->inner + i] +=
                            gp[row + i];
                    }
                }
            }
        };
        if (dt == Dtype::F32)
            run(reinterpret_cast<float*>(std::get<CpuStorage>(out).ptr.get()),
                reinterpret_cast<const float*>(g.ptr.get()));
        else
            run(reinterpret_cast<double*>(std::get<CpuStorage>(out).ptr.get()),
                reinterpret_cast<const double*>(g.ptr.get()));
        return out;
    }

    auto walk = [&](auto* dst, const auto* gp) {
        std::vector<std::int64_t> coord(ndim, 0);
        for (std::size_t out_flat = 0; out_flat < total; ++out_flat) {
            const std::int64_t k =
                cpu::checked_axis("cpu_backend::gather_backward", load_idx(out_flat), axis_extent,
                                  "index out of range for the gathered axis");
            std::size_t input_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                const std::int64_t c = (static_cast<int>(d) == axis) ? k : coord[d];
                input_flat +=
                    static_cast<std::size_t>(c) * static_cast<std::size_t>(input_stride[d]);
            }
            dst[input_flat] += gp[out_flat];
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d) {
                if (++coord[static_cast<std::size_t>(d)] <
                    output_shape[static_cast<std::size_t>(d)]) {
                    break;
                }
                coord[static_cast<std::size_t>(d)] = 0;
            }
        }
    };
    if (dt == Dtype::F32) {
        walk(reinterpret_cast<float*>(std::get<CpuStorage>(out).ptr.get()),
             reinterpret_cast<const float*>(g.ptr.get()));
    } else if (dt == Dtype::F64) {
        walk(reinterpret_cast<double*>(std::get<CpuStorage>(out).ptr.get()),
             reinterpret_cast<const double*>(g.ptr.get()));
    } else {
        ErrorBuilder("cpu_backend::gather_backward").not_implemented("dtype not supported");
    }
    return out;
}

std::pair<Storage, Storage> CpuBackend::sort_select_impl(const Storage& a,
                                                         const Shape& input_shape,
                                                         const Shape& output_shape,
                                                         int axis,
                                                         Dtype dt,
                                                         bool descending,
                                                         bool need_values) {
    // int8, int16, and bool widen to int64; float16 and bfloat16 widen to
    // float32 at the door and round once on the way out when values are requested.
    if (detail::is_narrow_int(dt)) {
        auto wide = sort_select_impl(detail::as_i64(a), input_shape, output_shape, axis, Dtype::I64,
                                     descending, need_values);
        if (!need_values)
            return wide;
        return {detail::narrow_int(wide.first, shape_numel(output_shape), dt), wide.second};
    }
    if (detail::is_half_like(dt)) {
        auto wide = sort_select_impl(detail::as_f32(a), input_shape, output_shape, axis, Dtype::F32,
                                     descending, need_values);
        return need_values ? detail::back_to_f16(std::move(wide), dt) : wide;
    }
    const auto& ca = std::get<CpuStorage>(a);
    auto run = [&](auto tag) {
        using T = decltype(tag);
        auto [v, idx] = cpu::sort_select_cpu<T>(ca, input_shape, output_shape, axis, dt, descending,
                                                need_values);
        return std::pair<Storage, Storage>{Storage{std::move(v)}, Storage{std::move(idx)}};
    };
    if (dt == Dtype::F32)
        return run(float{});
    if (dt == Dtype::F64)
        return run(double{});
    if (dt == Dtype::I32)
        return run(std::int32_t{});
    if (dt == Dtype::I64)
        return run(std::int64_t{});
    ErrorBuilder("cpu_backend::sort_select").not_implemented("dtype not supported");
}

std::pair<Storage, Storage> CpuBackend::sort_select(const Storage& a,
                                                    const Shape& input_shape,
                                                    const Shape& output_shape,
                                                    int axis,
                                                    Dtype dt,
                                                    bool descending) {
    return sort_select_impl(a, input_shape, output_shape, axis, dt, descending, true);
}

Storage CpuBackend::argsort(const Storage& a, const Shape& shape, int axis, Dtype dt) {
    return sort_select_impl(a, shape, shape, axis, dt, false, false).second;
}

Storage CpuBackend::scatter_add_axis(const Storage& grad,
                                     const Storage& indices,
                                     const Shape& output_shape,
                                     const Shape& grad_shape,
                                     int axis,
                                     Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(scatter_add_axis(detail::as_f32(grad), detail::as_f32(indices),
                                                    output_shape, grad_shape, axis, Dtype::F32),
                                   dt);
    const auto& g = std::get<CpuStorage>(grad);
    const auto& idx_storage = std::get<CpuStorage>(indices);
    const std::size_t nbytes = shape_numel(output_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
    std::memset(ptr.get(), 0, nbytes);
    auto run = [&](auto* dst, const auto* gp) {
        cpu::scatter_add_axis_any_index(dst, gp, idx_storage, output_shape, grad_shape, axis);
    };

    if (dt == Dtype::F32) {
        run(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(g.ptr.get()));
    } else if (dt == Dtype::F64) {
        run(reinterpret_cast<double*>(ptr.get()), reinterpret_cast<const double*>(g.ptr.get()));
    } else {
        ErrorBuilder("cpu_backend::scatter_add_axis").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nbytes, dt}};
}

Storage CpuBackend::scatter_add(const Storage& base,
                                const Storage& indices,
                                const Storage& src,
                                const Shape& base_shape,
                                const Shape& idx_shape,
                                int dim,
                                Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(scatter_add(detail::as_f32(base), detail::as_f32(indices),
                                               detail::as_f32(src), base_shape, idx_shape, dim,
                                               Dtype::F32),
                                   dt);
    // Integers take the same road, through int64.  Selection, movement
    // and addition are all exact there, so this is the answer a native
    // kernel would have produced rather than an approximation of it.
    if (detail::is_narrow_int(dt))
        return detail::back_to_int(scatter_add(detail::as_i64(base), indices, detail::as_i64(src),
                                               base_shape, idx_shape, dim, Dtype::I64),
                                   dt);
    const auto& cb = std::get<CpuStorage>(base);
    const auto& ci = std::get<CpuStorage>(indices);
    const auto& cs = std::get<CpuStorage>(src);

    const std::size_t base_n = shape_numel(base_shape);
    const std::size_t nbytes = base_n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
    // Start from a copy of base
    std::memcpy(ptr.get(), cb.ptr.get(), std::min(nbytes, cb.nbytes));

    const int ndim = static_cast<int>(base_shape.size());
    if (dim < 0)
        dim += ndim;

    // The walk is over the *index's* extents, which may be shorter than
    // base's on any axis but ``dim`` — see AxisScatterWalk.  It used
    // base's, and read past the end of an index and a src shorter than
    // base: out-of-range reads written into the answer.
    const AxisScatterWalk walk = axis_scatter_walk(base_shape, idx_shape, dim);
    const std::vector<std::size_t>& outer_off = walk.outer_off;
    const std::vector<std::size_t>& inner_off = walk.inner_off;
    const std::size_t outer = outer_off.size();
    const std::size_t inner = inner_off.size();
    const std::size_t base_dim = walk.base_dim;
    const std::size_t dim_stride = walk.dim_stride;
    const std::size_t idx_dim = walk.idx_dim;
    // The index buffer used to be read as ``int32`` whatever its actual
    // dtype.  An int16 index tensor therefore had two of its values
    // read as one, giving a silently wrong result; an int8 one had
    // four read as one, producing an index far outside the base and an
    // out-of-bounds write — a bus error reachable from Python.
    const auto load_index = cpu::index_reader("scatter_add", ci);

    auto run = [&](auto* dst, const auto* sp) {
        const auto len = static_cast<std::int64_t>(base_dim);
        for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t j = 0; j < inner; ++j) {
                for (std::size_t k = 0; k < idx_dim; ++k) {
                    const std::size_t src_flat = (o * idx_dim + k) * inner + j;
                    const std::int64_t tgt =
                        cpu::checked_axis("scatter_add", load_index(src_flat), len,
                                          "index out of range for the scattered axis");
                    const std::size_t dst_flat =
                        outer_off[o] + static_cast<std::size_t>(tgt) * dim_stride + inner_off[j];
                    dst[dst_flat] += sp[src_flat];
                }
            }
        }
    };

    if (dt == Dtype::F32)
        run(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(cs.ptr.get()));
    else if (dt == Dtype::F64)
        run(reinterpret_cast<double*>(ptr.get()), reinterpret_cast<const double*>(cs.ptr.get()));
    else if (dt == Dtype::I64)
        run(reinterpret_cast<std::int64_t*>(ptr.get()),
            reinterpret_cast<const std::int64_t*>(cs.ptr.get()));
    else if (dt == Dtype::C64)
        run(reinterpret_cast<std::complex<float>*>(ptr.get()),
            reinterpret_cast<const std::complex<float>*>(cs.ptr.get()));
    else if (dt == Dtype::C128)
        run(reinterpret_cast<std::complex<double>*>(ptr.get()),
            reinterpret_cast<const std::complex<double>*>(cs.ptr.get()));
    else
        ErrorBuilder("cpu_backend::scatter_add").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nbytes, dt}};
}

CpuBackend::AxisScatterWalk
CpuBackend::axis_scatter_walk(const Shape& base_shape, const Shape& idx_shape, int dim) {
    const int ndim = static_cast<int>(base_shape.size());
    const auto ext = [](const Shape& sh, int a) {
        return static_cast<std::size_t>(sh[static_cast<std::size_t>(a)]);
    };
    std::vector<std::size_t> base_stride(static_cast<std::size_t>(ndim), 1);
    for (int a = ndim - 2; a >= 0; --a)
        base_stride[static_cast<std::size_t>(a)] =
            base_stride[static_cast<std::size_t>(a + 1)] * ext(base_shape, a + 1);
    // Base offset of every coordinate over axes [lo, hi), by mixed-radix
    // decomposition over the index's extents.
    const auto offsets = [&](int lo, int hi) {
        std::size_t count = 1;
        for (int a = lo; a < hi; ++a)
            count *= ext(idx_shape, a);
        std::vector<std::size_t> off(count, 0);
        for (std::size_t f = 0; f < count; ++f) {
            std::size_t rem = f;
            for (int a = hi - 1; a >= lo; --a) {
                off[f] += (rem % ext(idx_shape, a)) * base_stride[static_cast<std::size_t>(a)];
                rem /= ext(idx_shape, a);
            }
        }
        return off;
    };
    AxisScatterWalk w;
    w.outer_off = offsets(0, dim);
    w.inner_off = offsets(dim + 1, ndim);
    w.idx_dim = ext(idx_shape, dim);
    w.base_dim = ext(base_shape, dim);
    w.dim_stride = base_stride[static_cast<std::size_t>(dim)];
    return w;
}

template <std::size_t W>
void CpuBackend::scatter_set_lines(std::byte* dst,
                                   const std::byte* src,
                                   cpu::IndexReader read_idx,
                                   const AxisScatterWalk& w) {
    const std::size_t outer = w.outer_off.size();
    const std::size_t inner = w.inner_off.size();
    const auto len = static_cast<std::int64_t>(w.base_dim);
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t k = 0; k < w.idx_dim; ++k) {
            const std::size_t row = (o * w.idx_dim + k) * inner;
            for (std::size_t j = 0; j < inner; ++j) {
                const std::int64_t tgt = cpu::checked_axis(
                    "scatter", read_idx(row + j), len, "index out of range for the scattered axis");
                const std::size_t at =
                    w.outer_off[o] + static_cast<std::size_t>(tgt) * w.dim_stride + w.inner_off[j];
                std::memcpy(dst + at * W, src + (row + j) * W, W);
            }
        }
    }
}

Storage CpuBackend::scatter_reduce_loop(const Storage& base,
                                        const Storage& indices,
                                        const Storage& src,
                                        const Shape& base_shape,
                                        const Shape& idx_shape,
                                        int dim,
                                        Dtype dt,
                                        const char* name,
                                        cpu::ScatterReduce reduce) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(scatter_reduce_loop(detail::as_f32(base), indices,
                                                       detail::as_f32(src), base_shape, idx_shape,
                                                       dim, Dtype::F32, name, reduce),
                                   dt);
    const auto& cb = std::get<CpuStorage>(base);
    const auto& ci = std::get<CpuStorage>(indices);
    const auto& cs = std::get<CpuStorage>(src);

    const int ndim = static_cast<int>(base_shape.size());
    if (dim < 0)
        dim += ndim;
    // The shape rule is the op's (``axis_scatter_operands`` in
    // Gfunc.cpp); this only refuses to read a src buffer shorter than
    // the index, as scatter_set does.
    const std::size_t count = shape_numel(idx_shape);
    if (cs.nbytes < count * dtype_size(dt))
        ErrorBuilder(name).fail("src must hold one element per index position");

    const std::size_t base_n = shape_numel(base_shape);
    const std::size_t nbytes = base_n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
    if (nbytes > 0)
        std::memcpy(ptr.get(), cb.ptr.get(), std::min(nbytes, cb.nbytes));
    if (count == 0)
        return Storage{CpuStorage{ptr, nbytes, dt}};

    const auto read_idx = cpu::index_reader(name, ci);
    const AxisScatterWalk w = axis_scatter_walk(base_shape, idx_shape, dim);
    const auto len = static_cast<std::int64_t>(w.base_dim);

    auto run = [&](auto* dst, const auto* sp) {
        const std::size_t inner = w.inner_off.size();
        for (std::size_t o = 0; o < w.outer_off.size(); ++o)
            for (std::size_t k = 0; k < w.idx_dim; ++k) {
                const std::size_t row = (o * w.idx_dim + k) * inner;
                for (std::size_t j = 0; j < inner; ++j) {
                    const std::int64_t tgt = cpu::checked_axis(
                        name, read_idx(row + j), len, "index out of range for the scattered axis");
                    const std::size_t at = w.outer_off[o] +
                                           static_cast<std::size_t>(tgt) * w.dim_stride +
                                           w.inner_off[j];
                    cpu::scatter_combine(reduce, dst[at], sp[row + j]);
                }
            }
    };

    cpu::visit_scatter_reduce_type(dt, reduce, name, [&](auto tag) {
        using T = decltype(tag);
        run(reinterpret_cast<T*>(ptr.get()), reinterpret_cast<const T*>(cs.ptr.get()));
    });
    return Storage{CpuStorage{ptr, nbytes, dt}};
}

Storage CpuBackend::scatter_amax(const Storage& base,
                                 const Storage& indices,
                                 const Storage& src,
                                 const Shape& base_shape,
                                 const Shape& idx_shape,
                                 int dim,
                                 Dtype dt) {
    return scatter_reduce_loop(base, indices, src, base_shape, idx_shape, dim, dt, "scatter_amax",
                               cpu::ScatterReduce::Amax);
}

Storage CpuBackend::scatter_amin(const Storage& base,
                                 const Storage& indices,
                                 const Storage& src,
                                 const Shape& base_shape,
                                 const Shape& idx_shape,
                                 int dim,
                                 Dtype dt) {
    return scatter_reduce_loop(base, indices, src, base_shape, idx_shape, dim, dt, "scatter_amin",
                               cpu::ScatterReduce::Amin);
}

Storage CpuBackend::scatter_prod(const Storage& base,
                                 const Storage& indices,
                                 const Storage& src,
                                 const Shape& base_shape,
                                 const Shape& idx_shape,
                                 int dim,
                                 Dtype dt) {
    return scatter_reduce_loop(base, indices, src, base_shape, idx_shape, dim, dt, "scatter_prod",
                               cpu::ScatterReduce::Prod);
}

Storage CpuBackend::scatter_set(const Storage& base,
                                const Storage& indices,
                                const Storage& src,
                                const Shape& base_shape,
                                const Shape& idx_shape,
                                int dim,
                                Dtype dt) {
    const auto& cb = std::get<CpuStorage>(base);
    const auto& ci = std::get<CpuStorage>(indices);
    const auto& cs = std::get<CpuStorage>(src);
    // The bytes are moved, not converted: a src of another dtype would be
    // reinterpreted, and one of another width read out of bounds.
    if (cs.dtype != dt || cb.dtype != dt)
        ErrorBuilder("scatter").fail("source and destination must share a dtype, got " +
                                     std::string(dtype_name(cs.dtype)) + " into " +
                                     std::string(dtype_name(cb.dtype)));

    const std::size_t esz = dtype_size(dt);
    const std::size_t nbytes = shape_numel(base_shape) * esz;
    auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
    if (nbytes > 0)
        std::memcpy(ptr.get(), cb.ptr.get(), std::min(nbytes, cb.nbytes));
    const std::size_t count = shape_numel(idx_shape);
    if (count == 0)
        return Storage{CpuStorage{ptr, nbytes, dt}};
    if (cs.nbytes < count * esz || ci.nbytes < count * dtype_size(ci.dtype))
        ErrorBuilder("scatter").fail("index and source must hold one "
                                     "element per index position");

    const int ndim = static_cast<int>(base_shape.size());
    if (dim < 0)
        dim += ndim;
    const AxisScatterWalk walk = axis_scatter_walk(base_shape, idx_shape, dim);
    const auto read_idx = cpu::index_reader("scatter", ci);
    std::byte* dst = ptr.get();
    const std::byte* sp = cs.ptr.get();
    switch (esz) {
    case 1:
        scatter_set_lines<1>(dst, sp, read_idx, walk);
        break;
    case 2:
        scatter_set_lines<2>(dst, sp, read_idx, walk);
        break;
    case 4:
        scatter_set_lines<4>(dst, sp, read_idx, walk);
        break;
    case 8:
        scatter_set_lines<8>(dst, sp, read_idx, walk);
        break;
    case 16:
        scatter_set_lines<16>(dst, sp, read_idx, walk);
        break;
    default:
        ErrorBuilder("cpu_backend::scatter_set").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nbytes, dt}};
}

Storage CpuBackend::masked_select_count(const Storage& mask, const Shape& shape, Dtype /*dt*/) {
    const auto& cm = std::get<CpuStorage>(mask);
    const std::size_t n = shape_numel(shape);
    const bool* mp = reinterpret_cast<const bool*>(cm.ptr.get());
    std::int64_t count = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (mp[i])
            ++count;
    auto out_ptr = allocate_aligned_bytes(sizeof(std::int64_t), Device::CPU);
    std::memcpy(out_ptr.get(), &count, sizeof(std::int64_t));
    return Storage{CpuStorage{out_ptr, sizeof(std::int64_t), Dtype::I64}};
}

Storage CpuBackend::masked_select(const Storage& a,
                                  const Storage& mask,
                                  const Shape& a_shape,
                                  const Shape& /*mask_shape*/,
                                  std::int64_t n_true,
                                  Dtype dt) {
    const auto& ca = std::get<CpuStorage>(a);
    const auto& cm = std::get<CpuStorage>(mask);
    const std::size_t n = shape_numel(a_shape);
    const std::size_t esz = dtype_size(dt);
    const std::size_t nb = static_cast<std::size_t>(n_true) * esz;
    auto out_ptr = allocate_aligned_bytes(nb, Device::CPU);
    const std::uint8_t* src =
        static_cast<const std::uint8_t*>(static_cast<const void*>(ca.ptr.get()));
    const bool* mp = reinterpret_cast<const bool*>(cm.ptr.get());
    std::uint8_t* dst = static_cast<std::uint8_t*>(static_cast<void*>(out_ptr.get()));
    std::size_t out_idx = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (mp[i]) {
            std::memcpy(dst + out_idx * esz, src + i * esz, esz);
            ++out_idx;
        }
    }
    return Storage{CpuStorage{out_ptr, nb, dt}};
}

Storage CpuBackend::one_hot_forward(const Storage& indices,
                                    const Shape& indices_shape,
                                    int num_classes,
                                    Dtype out_dtype) {
    // A half index is read the way Metal reads it: widened, then
    // truncated toward zero.
    if (detail::is_half_like(std::get<CpuStorage>(indices).dtype))
        return one_hot_forward(detail::as_f32(indices), indices_shape, num_classes, out_dtype);
    const std::size_t M = shape_numel(indices_shape);
    const auto& is_ = std::get<CpuStorage>(indices);
    auto out_cpu = alloc_cpu(M * static_cast<std::size_t>(num_classes), out_dtype);
    std::memset(out_cpu.ptr.get(), 0, out_cpu.nbytes);
    const auto read_idx = cpu::real_index_reader("one_hot", is_);
    for (std::size_t i = 0; i < M; ++i) {
        // A class outside [0, num_classes) is refused, as the reference
        // refuses it.  It used to leave its row zero, the same answer as
        // Metal, so a bad label vanished silently on both devices.  Metal
        // still answers a zero row (policy B, see GpuBackend's
        // ``gpu_axis_index``), because refusing there needs a host sync.
        const auto cls =
            static_cast<std::size_t>(cpu::checked_row("one_hot", read_idx(i), num_classes));
        const std::size_t pos = i * static_cast<std::size_t>(num_classes) + cls;
        switch (out_dtype) {
        case Dtype::F32:
            reinterpret_cast<float*>(out_cpu.ptr.get())[pos] = 1.f;
            break;
        case Dtype::F64:
            reinterpret_cast<double*>(out_cpu.ptr.get())[pos] = 1.0;
            break;
        case Dtype::I8:
        case Dtype::Bool:
            reinterpret_cast<std::uint8_t*>(out_cpu.ptr.get())[pos] = 1;
            break;
        case Dtype::I16:
            reinterpret_cast<std::int16_t*>(out_cpu.ptr.get())[pos] = 1;
            break;
        case Dtype::I32:
            reinterpret_cast<std::int32_t*>(out_cpu.ptr.get())[pos] = 1;
            break;
        case Dtype::I64:
            reinterpret_cast<std::int64_t*>(out_cpu.ptr.get())[pos] = 1;
            break;
        default:
            ErrorBuilder("one_hot").not_implemented("out dtype not supported");
        }
    }
    return Storage{std::move(out_cpu)};
}

CpuStorage CpuBackend::nonzero_forward(const Storage& input,
                                       const Shape& input_shape,
                                       Dtype input_dtype,
                                       std::size_t& numel_out) {
    if (detail::is_half_like(input_dtype))
        return nonzero_forward(detail::as_f32(input), input_shape, Dtype::F32, numel_out);
    if (detail::is_narrow_int(input_dtype))
        return nonzero_forward(detail::as_i64(input), input_shape, Dtype::I64, numel_out);
    const auto& cs = std::get<CpuStorage>(input);
    const std::size_t n = shape_numel(input_shape);
    const std::size_t ndim = input_shape.size();
    std::vector<std::size_t> hits;
    auto check_nonzero = [&](const auto* p) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(p)>>;
        for (std::size_t i = 0; i < n; ++i)
            if (p[i] != T{0})
                hits.push_back(i);
    };
    if (input_dtype == Dtype::F32)
        check_nonzero(reinterpret_cast<const float*>(cs.ptr.get()));
    else if (input_dtype == Dtype::F64)
        check_nonzero(reinterpret_cast<const double*>(cs.ptr.get()));
    else if (input_dtype == Dtype::I64)
        check_nonzero(reinterpret_cast<const std::int64_t*>(cs.ptr.get()));
    else if (input_dtype == Dtype::C64)
        check_nonzero(reinterpret_cast<const std::complex<float>*>(cs.ptr.get()));
    else if (input_dtype == Dtype::C128)
        check_nonzero(reinterpret_cast<const std::complex<double>*>(cs.ptr.get()));
    else
        ErrorBuilder("cpu::nonzero_forward").not_implemented("dtype not supported");
    numel_out = hits.size();
    CpuStorage out = alloc_cpu(numel_out * ndim, Dtype::I64);
    auto* dst = reinterpret_cast<std::int64_t*>(out.ptr.get());
    Stride stride(ndim, 1);
    for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d)
        stride[static_cast<std::size_t>(d)] =
            stride[static_cast<std::size_t>(d) + 1] * input_shape[static_cast<std::size_t>(d) + 1];
    for (std::size_t row = 0; row < numel_out; ++row) {
        std::size_t rem = hits[row];
        for (std::size_t d = 0; d < ndim; ++d) {
            dst[row * ndim + d] =
                static_cast<std::int64_t>(rem / static_cast<std::size_t>(stride[d]));
            rem %= static_cast<std::size_t>(stride[d]);
        }
    }
    return out;
}

}  // namespace backend
}  // namespace lucid
