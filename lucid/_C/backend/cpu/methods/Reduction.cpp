// lucid/_C/backend/cpu/methods/Reduction.cpp
// Out-of-line CpuBackend method definitions for the Reduction family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

Storage
CpuBackend::reduce_sum(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) {
    return reduce_axes(a, in_shape, opts, dt, ReduceOp::Sum);
}

Storage
CpuBackend::reduce_mean(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(reduce_mean(detail::as_f32(a), in_shape, opts, Dtype::F32), dt);
    return reduce_axes(a, in_shape, opts, dt, ReduceOp::Mean);
}

Storage
CpuBackend::variance(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(variance(detail::as_f32(a), in_shape, opts, Dtype::F32), dt);
    const auto& cs = std::get<CpuStorage>(a);
    auto is_reduced_axis = [&](std::size_t d) {
        return std::find(opts.axes.begin(), opts.axes.end(), static_cast<int>(d)) !=
               opts.axes.end();
    };
    Shape out_shape;
    if (opts.keepdims) {
        out_shape = in_shape;
        for (int ax : opts.axes)
            out_shape[static_cast<std::size_t>(ax)] = 1;
    } else {
        for (std::size_t d = 0; d < in_shape.size(); ++d) {
            if (!is_reduced_axis(d))
                out_shape.push_back(in_shape[d]);
        }
    }

    const std::size_t in_numel = shape_numel(in_shape);
    const std::size_t out_numel = shape_numel(out_shape);
    std::size_t reduced = 1;
    for (int ax : opts.axes)
        reduced *= static_cast<std::size_t>(in_shape[static_cast<std::size_t>(ax)]);
    if (reduced == 0)
        reduced = 1;

    const std::size_t out_nbytes = out_numel * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);

    auto compute = [&](auto* out_p, const auto* in_p) {
        using T = std::remove_pointer_t<decltype(out_p)>;
        std::vector<double> means(out_numel, 0.0);
        std::vector<std::int64_t> coords(in_shape.size(), 0);

        Shape kd_shape = in_shape;
        for (int ax : opts.axes)
            kd_shape[static_cast<std::size_t>(ax)] = 1;

        Stride kd_stride(kd_shape.size());
        if (!kd_shape.empty()) {
            kd_stride.back() = 1;
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(kd_shape.size()) - 2; i >= 0; --i) {
                kd_stride[static_cast<std::size_t>(i)] =
                    kd_stride[static_cast<std::size_t>(i) + 1] *
                    kd_shape[static_cast<std::size_t>(i) + 1];
            }
        }

        Stride in_stride(in_shape.size());
        if (!in_shape.empty()) {
            in_stride.back() = 1;
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(in_shape.size()) - 2; i >= 0; --i) {
                in_stride[static_cast<std::size_t>(i)] =
                    in_stride[static_cast<std::size_t>(i) + 1] *
                    in_shape[static_cast<std::size_t>(i) + 1];
            }
        }

        auto flat_to_coord = [&](std::size_t flat) {
            for (std::size_t d = 0; d < in_shape.size(); ++d) {
                coords[d] = flat / static_cast<std::size_t>(in_stride[d]);
                flat %= static_cast<std::size_t>(in_stride[d]);
            }
        };

        auto kd_flat = [&]() {
            std::size_t f = 0;
            for (std::size_t d = 0; d < in_shape.size(); ++d) {
                std::int64_t c = coords[d];
                if (kd_shape[d] == 1)
                    c = 0;
                f += c * static_cast<std::size_t>(kd_stride[d]);
            }
            return f;
        };

        for (std::size_t i = 0; i < in_numel; ++i) {
            flat_to_coord(i);
            means[kd_flat()] += static_cast<double>(in_p[i]);
        }
        for (auto& m : means)
            m /= static_cast<double>(reduced);

        std::vector<double> vars(out_numel, 0.0);
        for (std::size_t i = 0; i < in_numel; ++i) {
            flat_to_coord(i);
            const auto kf = kd_flat();
            const double d = static_cast<double>(in_p[i]) - means[kf];
            vars[kf] += d * d;
        }
        for (std::size_t k = 0; k < out_numel; ++k)
            out_p[k] = static_cast<T>(vars[k] / static_cast<double>(reduced));
    };

    if (dt == Dtype::F32)
        compute(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(cs.ptr.get()));
    else if (dt == Dtype::F64)
        compute(reinterpret_cast<double*>(ptr.get()),
                reinterpret_cast<const double*>(cs.ptr.get()));
    else
        ErrorBuilder("cpu_backend::variance").not_implemented("dtype not supported");

    return Storage{CpuStorage{ptr, out_nbytes, dt}};
}

Storage
CpuBackend::reduce_max(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) {
    return reduce_axes(a, in_shape, opts, dt, ReduceOp::Max);
}

Storage
CpuBackend::reduce_min(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) {
    return reduce_axes(a, in_shape, opts, dt, ReduceOp::Min);
}

Storage CpuBackend::cumsum(const Storage& a, const Shape& shape, int axis, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(cumsum(detail::as_f32(a), shape, axis, Dtype::F32), dt);
    return detail::scan_axis(a, shape, axis, dt, "cpu_backend::cumsum", std::plus<>{});
}

Storage CpuBackend::cumprod(const Storage& a, const Shape& shape, int axis, Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(cumprod(detail::as_f32(a), shape, axis, Dtype::F32), dt);
    return detail::scan_axis(a, shape, axis, dt, "cpu_backend::cumprod", std::multiplies<>{});
}

Storage CpuBackend::cummax(const Storage& a, const Shape& shape, int axis, Dtype dt) {
    // Exact through int64: a running extreme over widened values is the
    // same sequence, narrowed back.  bool travels with them — the
    // running maximum of a sequence of booleans is a running OR, and 0
    // and 1 survive the widening unchanged.  It was excluded here, so
    // Metal answered a bool scan and the CPU refused one.
    if (detail::is_narrow_int(dt)) {
        const std::size_t n = shape_numel(shape);
        return detail::narrow_int(cummax(detail::as_i64(a), shape, axis, Dtype::I64), n, dt);
    }
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(cummax(detail::as_f32(a), shape, axis, Dtype::F32), dt);
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t nb = cs.nbytes;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const int ndim = static_cast<int>(shape.size());
    std::size_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    for (int d = axis + 1; d < ndim; ++d)
        inner *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    const std::size_t L = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
    // An empty scan axis has nothing to seed the running value with; the
    // loop below read element 0 of every row anyway, past the end of an
    // empty buffer, and ``cumsum(zeros(0, 6), 0)`` died of SIGSEGV.
    if (L == 0)
        return Storage{CpuStorage{ptr, nb, dt}};

    auto run = [&](auto* dst, const auto* src) {
        using T = std::remove_pointer_t<decltype(dst)>;
        for (std::size_t o = 0; o < outer; ++o)
            for (std::size_t j = 0; j < inner; ++j) {
                T running = src[(o * L) * inner + j];
                dst[(o * L) * inner + j] = running;
                for (std::size_t k = 1; k < L; ++k) {
                    T v = src[(o * L + k) * inner + j];
                    // NaN poisons the rest of the scan, as it does in
                    // ``max``: once seen it is the running extreme and
                    // every later element inherits it.  ``v > running``
                    // is false against a NaN, so the scan skipped it and
                    // carried on with a plausible running maximum —
                    // Metal answered nan from that point and the CPU did
                    // not, and neither matched the reference until now.
                    if constexpr (std::is_floating_point_v<T>) {
                        if (std::isnan(running)) {
                            dst[(o * L + k) * inner + j] = running;
                            continue;
                        }
                        if (std::isnan(v))
                            running = v;
                    }
                    if (v > running)
                        running = v;
                    dst[(o * L + k) * inner + j] = running;
                }
            }
    };

    if (dt == Dtype::F32)
        run(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(cs.ptr.get()));
    else if (dt == Dtype::F64)
        run(reinterpret_cast<double*>(ptr.get()), reinterpret_cast<const double*>(cs.ptr.get()));
    else if (dt == Dtype::I32)
        run(reinterpret_cast<std::int32_t*>(ptr.get()),
            reinterpret_cast<const std::int32_t*>(cs.ptr.get()));
    else if (dt == Dtype::I64)
        run(reinterpret_cast<std::int64_t*>(ptr.get()),
            reinterpret_cast<const std::int64_t*>(cs.ptr.get()));
    else
        ErrorBuilder("cpu_backend::cummax").not_implemented("dtype not supported");

    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::cummin(const Storage& a, const Shape& shape, int axis, Dtype dt) {
    // Exact through int64: a running extreme over widened values is the
    // same sequence, narrowed back.  bool travels with them — the
    // running maximum of a sequence of booleans is a running OR, and 0
    // and 1 survive the widening unchanged.  It was excluded here, so
    // Metal answered a bool scan and the CPU refused one.
    if (detail::is_narrow_int(dt)) {
        const std::size_t n = shape_numel(shape);
        return detail::narrow_int(cummin(detail::as_i64(a), shape, axis, Dtype::I64), n, dt);
    }
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(cummin(detail::as_f32(a), shape, axis, Dtype::F32), dt);
    const auto& cs = std::get<CpuStorage>(a);
    std::size_t nb = cs.nbytes;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const int ndim = static_cast<int>(shape.size());
    std::size_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    for (int d = axis + 1; d < ndim; ++d)
        inner *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    const std::size_t L = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
    // An empty scan axis has nothing to seed the running value with; the
    // loop below read element 0 of every row anyway, past the end of an
    // empty buffer, and ``cumsum(zeros(0, 6), 0)`` died of SIGSEGV.
    if (L == 0)
        return Storage{CpuStorage{ptr, nb, dt}};

    auto run = [&](auto* dst, const auto* src) {
        using T = std::remove_pointer_t<decltype(dst)>;
        for (std::size_t o = 0; o < outer; ++o)
            for (std::size_t j = 0; j < inner; ++j) {
                T running = src[(o * L) * inner + j];
                dst[(o * L) * inner + j] = running;
                for (std::size_t k = 1; k < L; ++k) {
                    T v = src[(o * L + k) * inner + j];
                    // NaN poisons the rest of the scan, as it does in
                    // ``max``: once seen it is the running extreme and
                    // every later element inherits it.  ``v > running``
                    // is false against a NaN, so the scan skipped it and
                    // carried on with a plausible running maximum —
                    // Metal answered nan from that point and the CPU did
                    // not, and neither matched the reference until now.
                    if constexpr (std::is_floating_point_v<T>) {
                        if (std::isnan(running)) {
                            dst[(o * L + k) * inner + j] = running;
                            continue;
                        }
                        if (std::isnan(v))
                            running = v;
                    }
                    if (v < running)
                        running = v;
                    dst[(o * L + k) * inner + j] = running;
                }
            }
    };

    if (dt == Dtype::F32)
        run(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(cs.ptr.get()));
    else if (dt == Dtype::F64)
        run(reinterpret_cast<double*>(ptr.get()), reinterpret_cast<const double*>(cs.ptr.get()));
    else if (dt == Dtype::I32)
        run(reinterpret_cast<std::int32_t*>(ptr.get()),
            reinterpret_cast<const std::int32_t*>(cs.ptr.get()));
    else if (dt == Dtype::I64)
        run(reinterpret_cast<std::int64_t*>(ptr.get()),
            reinterpret_cast<const std::int64_t*>(cs.ptr.get()));
    else
        ErrorBuilder("cpu_backend::cummin").not_implemented("dtype not supported");

    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::trace(const Storage& a, const Shape& shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(trace(detail::as_f32(a), shape, Dtype::F32), dt);
    const auto& cs = std::get<CpuStorage>(a);
    const std::int64_t M = shape[0];
    const std::int64_t N = shape[1];
    const std::int64_t L = std::min(M, N);
    Shape out_shape(shape.begin() + 2, shape.end());
    const std::size_t out_numel = shape_numel(out_shape);
    const std::size_t out_nbytes = out_numel * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
    auto run = [&](auto* out_p, const auto* in_p) {
        using T = std::remove_pointer_t<decltype(out_p)>;
        for (std::size_t k = 0; k < out_numel; ++k) {
            T sum{};
            for (std::int64_t i = 0; i < L; ++i) {
                const std::size_t idx = (i * N + i) * out_numel + k;
                sum = static_cast<T>(static_cast<double>(sum) + static_cast<double>(in_p[idx]));
            }
            out_p[k] = sum;
        }
    };
    // Integers included: the reference framework sums the diagonal of
    // an integer matrix and keeps its dtype, and so does Metal.  Each
    // width accumulates in itself, which is what the reference does —
    // an overflowing trace overflows there too.
    switch (dt) {
    case Dtype::F32:
        run(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(cs.ptr.get()));
        break;
    case Dtype::F64:
        run(reinterpret_cast<double*>(ptr.get()), reinterpret_cast<const double*>(cs.ptr.get()));
        break;
    case Dtype::I64:
        run(reinterpret_cast<std::int64_t*>(ptr.get()),
            reinterpret_cast<const std::int64_t*>(cs.ptr.get()));
        break;
    case Dtype::I32:
        run(reinterpret_cast<std::int32_t*>(ptr.get()),
            reinterpret_cast<const std::int32_t*>(cs.ptr.get()));
        break;
    case Dtype::I16:
        run(reinterpret_cast<std::int16_t*>(ptr.get()),
            reinterpret_cast<const std::int16_t*>(cs.ptr.get()));
        break;
    case Dtype::I8:
        run(reinterpret_cast<std::int8_t*>(ptr.get()),
            reinterpret_cast<const std::int8_t*>(cs.ptr.get()));
        break;
    default:
        ErrorBuilder("cpu_backend::trace").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, out_nbytes, dt}};
}

Storage CpuBackend::trace_backward(const Storage& grad_out, const Shape& input_shape, Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            trace_backward(detail::as_f32(grad_out), input_shape, Dtype::F32), dt);
    const auto& cg = std::get<CpuStorage>(grad_out);
    const std::int64_t M = input_shape[0];
    const std::int64_t N = input_shape[1];
    const std::int64_t L = std::min(M, N);
    const std::size_t total = static_cast<std::size_t>(M * N);
    const std::size_t nbytes = total * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
    std::memset(ptr.get(), 0, nbytes);
    auto fill = [&](auto* dst, const auto* gp) {
        for (std::int64_t i = 0; i < L; ++i)
            dst[i * N + i] = *gp;
    };
    if (dt == Dtype::F32)
        fill(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(cg.ptr.get()));
    else if (dt == Dtype::F64)
        fill(reinterpret_cast<double*>(ptr.get()), reinterpret_cast<const double*>(cg.ptr.get()));
    else
        ErrorBuilder("cpu_backend::trace_backward").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nbytes, dt}};
}

Storage CpuBackend::arg_reduce_index(
    const Storage& a, const Shape& shape, int axis, bool keepdims, Dtype dt, bool is_min) {
    // The result is an index tensor — int64 by contract — so the
    // operand widens and nothing narrows back.
    if (detail::is_narrow_int(dt) && dt != Dtype::Bool)
        return arg_reduce_index(detail::as_i64(a), shape, axis, keepdims, Dtype::I64, is_min);
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            arg_reduce_index(detail::as_f32(a), shape, axis, keepdims, Dtype::F32, is_min), dt);
    Shape out_shape = shape;
    if (keepdims)
        out_shape[static_cast<std::size_t>(axis)] = 1;
    else
        out_shape.erase(out_shape.begin() + axis);
    const std::size_t nbytes = shape_numel(out_shape) * dtype_size(Dtype::I64);
    auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
    auto* dst = reinterpret_cast<std::int64_t*>(ptr.get());
    const auto& ca = std::get<CpuStorage>(a);

    std::size_t outer = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    std::size_t inner = 1;
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d)
        inner *= static_cast<std::size_t>(shape[d]);
    const std::size_t L = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);

    auto run = [&](const auto* src) {
        for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t j = 0; j < inner; ++j) {
                std::int64_t best = 0;
                auto best_v = src[o * L * inner + j];
                using V = std::remove_cv_t<std::remove_reference_t<decltype(best_v)>>;
                for (std::size_t k = 1; k < L; ++k) {
                    const auto v = src[(o * L + k) * inner + j];
                    // NaN wins, for both argmax and argmin.
                    //
                    // The comparison alone cannot say so — every test
                    // against a NaN is false, so the first NaN in the
                    // axis was skipped and the index of some ordinary
                    // element came back instead.  ``max`` reports nan
                    // and ``argmax`` reported the position of a number
                    // that is not the maximum, which is worse than
                    // either answer alone: the pair disagreed with each
                    // other.  The reference framework returns the NaN's
                    // index; once one is seen it cannot be displaced.
                    if constexpr (std::is_floating_point_v<V>) {
                        if (std::isnan(best_v))
                            continue;
                        if (std::isnan(v)) {
                            best_v = v;
                            best = static_cast<std::int64_t>(k);
                            continue;
                        }
                    }
                    if (is_min ? (v < best_v) : (v > best_v)) {
                        best_v = v;
                        best = static_cast<std::int64_t>(k);
                    }
                }
                dst[o * inner + j] = best;
            }
        }
    };

    // A contiguous axis: the extreme first, by the vector reduction (NaN
    // if any), then the first index holding it.  The same answer as the
    // scan above — the first of equal values, the first NaN — without a
    // compare-and-branch chained through every element (``argmax`` was
    // twenty times ``sum``).
    auto by_rows = [&](const auto* src, auto reduce) {
        using V = std::remove_cv_t<std::remove_pointer_t<decltype(src)>>;
        for (std::size_t o = 0; o < outer; ++o) {
            const V* row = src + o * L;
            V target;
            reduce(row, &target, 1, L, 1);
            std::size_t k = 0;
            if (std::isnan(target)) {
                while (!std::isnan(row[k]))
                    ++k;
            } else {
                while (row[k] != target)
                    ++k;
            }
            dst[o] = static_cast<std::int64_t>(k);
        }
    };
    const bool rows = inner == 1 && L > 0;

    if (dt == Dtype::F32 && rows) {
        by_rows(reinterpret_cast<const float*>(ca.ptr.get()),
                is_min ? &cpu::min_axis_f32 : &cpu::max_axis_f32);
    } else if (dt == Dtype::F64 && rows) {
        by_rows(reinterpret_cast<const double*>(ca.ptr.get()),
                is_min ? &cpu::min_axis_f64 : &cpu::max_axis_f64);
    } else if (dt == Dtype::F32) {
        run(reinterpret_cast<const float*>(ca.ptr.get()));
    } else if (dt == Dtype::F64) {
        run(reinterpret_cast<const double*>(ca.ptr.get()));
    } else if (dt == Dtype::I32) {
        run(reinterpret_cast<const std::int32_t*>(ca.ptr.get()));
    } else if (dt == Dtype::I64) {
        run(reinterpret_cast<const std::int64_t*>(ca.ptr.get()));
    } else {
        ErrorBuilder("cpu_backend::arg_reduce_index").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, nbytes, Dtype::I64}};
}

Storage CpuBackend::reduce_axes(
    const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt, ReduceOp op) {
    if (is_complex(dt)) {
        if (op != ReduceOp::Sum && op != ReduceOp::Mean)
            ErrorBuilder("cpu_backend::reduce").not_implemented("complex ordering is undefined");
        const auto lane = real_lane_of(dt);
        auto re = reduce_axes(complex_real(a, in_shape), in_shape, opts, lane, op);
        auto im = reduce_axes(complex_imag(a, in_shape), in_shape, opts, lane, op);
        const auto n = std::get<CpuStorage>(re).nbytes / dtype_size(lane);
        return complex_combine(re, im, Shape{static_cast<std::int64_t>(n)});
    }
    // Reductions accumulate, and there is no half accumulator on the
    // host.  Widening also removes the rounding-per-step error a true
    // half accumulation would carry.
    if (detail::is_half_like(dt) && !in_shape.empty()) {
        const std::size_t in_n = shape_numel(in_shape);
        const Storage wide = reduce_axes(Storage{detail::widen_half(std::get<CpuStorage>(a), in_n)},
                                         in_shape, opts, Dtype::F32, op);
        const std::size_t out_n = std::get<CpuStorage>(wide).nbytes / sizeof(float);
        return detail::narrow_half(wide, out_n, dt);
    }
    // The same shape for the narrow integers.  Only I64 was
    // implemented, so ``max`` and ``min`` refused int8/16/32 and bool
    // while Metal answered them — and taking the maximum of an integer
    // tensor is an ordinary thing to want.  Widening to I64 is exact
    // for every one of these (comparison and summation of a narrower
    // integer cannot overflow int64), so the answer is the same one
    // the wide path would give, narrowed back to the caller's dtype.
    if (!in_shape.empty() &&
        (dt == Dtype::I32 || dt == Dtype::I16 || dt == Dtype::I8 || dt == Dtype::Bool)) {
        const std::size_t in_n = shape_numel(in_shape);
        const Storage wide =
            reduce_axes(Storage{detail::widen_int(std::get<CpuStorage>(a), in_n, dt)}, in_shape,
                        opts, Dtype::I64, op);
        const std::size_t out_n = std::get<CpuStorage>(wide).nbytes / sizeof(std::int64_t);
        return detail::narrow_int(wide, out_n, dt);
    }
    // 0-d (scalar) input is already its own reduction. Returning early
    // avoids an infinite recursion in the empty-axes branch below, where
    // expanding `axes` to all dims would yield another empty list.
    if (in_shape.empty())
        return a;
    if (opts.axes.empty()) {
        std::vector<int> all_axes;
        for (int i = 0; i < static_cast<int>(in_shape.size()); ++i)
            all_axes.push_back(i);
        ReduceOpts all_opts{all_axes, opts.keepdims};
        return reduce_axes(a, in_shape, all_opts, dt, op);
    }

    auto axes = opts.axes;
    std::sort(axes.begin(), axes.end(), std::greater<int>());

    Storage cur = a;
    Shape cur_shape = in_shape;
    std::size_t total_rd = 1;

    for (int axis : axes) {
        const auto& cs = std::get<CpuStorage>(cur);
        const int ndim = static_cast<int>(cur_shape.size());
        const int ax = (axis < 0) ? axis + ndim : axis;
        const std::size_t outer = [&] {
            std::size_t o = 1;
            for (int i = 0; i < ax; ++i)
                o *= static_cast<std::size_t>(cur_shape[i]);
            return o;
        }();
        const std::size_t rd = static_cast<std::size_t>(cur_shape[ax]);
        total_rd *= rd;
        const std::size_t inner = [&] {
            std::size_t ii = 1;
            for (int i = ax + 1; i < ndim; ++i)
                ii *= static_cast<std::size_t>(cur_shape[i]);
            return ii;
        }();
        const std::size_t out_n = outer * inner;
        Shape out_shape;
        for (int i = 0; i < ndim; ++i)
            if (i != ax)
                out_shape.push_back(cur_shape[i]);
        if (out_shape.empty())
            out_shape.push_back(1);

        std::size_t nb = out_n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);

        if (dt == Dtype::F32) {
            const float* ip = reinterpret_cast<const float*>(cs.ptr.get());
            float* outp = reinterpret_cast<float*>(ptr.get());
            if (op == ReduceOp::Sum || op == ReduceOp::Mean)
                cpu::sum_axis_f32(ip, outp, outer, rd, inner);
            else if (op == ReduceOp::Max)
                cpu::max_axis_f32(ip, outp, outer, rd, inner);
            else
                cpu::min_axis_f32(ip, outp, outer, rd, inner);
        } else if (dt == Dtype::F64) {
            const double* ip = reinterpret_cast<const double*>(cs.ptr.get());
            double* outp = reinterpret_cast<double*>(ptr.get());
            if (op == ReduceOp::Sum || op == ReduceOp::Mean)
                cpu::sum_axis_f64(ip, outp, outer, rd, inner);
            else if (op == ReduceOp::Max)
                cpu::max_axis_f64(ip, outp, outer, rd, inner);
            else
                cpu::min_axis_f64(ip, outp, outer, rd, inner);
        } else if (dt == Dtype::I64) {
            // Native int64 reduce — no Accelerate primitive, but the
            // outer × inner × reduce_dim loop is straightforward.
            // ``sum_op`` / ``prod_op`` promote bool / I8 / I16 / I32 to
            // I64 via ``promote_int_for_reduce`` so all integral
            // reductions land here; this preserves the reference
            // framework's "int.sum() → int64" semantics without
            // round-tripping through F64 (which would lose
            // precision past 2^53).
            const std::int64_t* ip = reinterpret_cast<const std::int64_t*>(cs.ptr.get());
            std::int64_t* outp = reinterpret_cast<std::int64_t*>(ptr.get());
            for (std::size_t o = 0; o < outer; ++o) {
                for (std::size_t i = 0; i < inner; ++i) {
                    std::int64_t acc;
                    if (op == ReduceOp::Sum || op == ReduceOp::Mean) {
                        acc = 0;
                    } else if (op == ReduceOp::Max) {
                        acc = std::numeric_limits<std::int64_t>::min();
                    } else {  // Min
                        acc = std::numeric_limits<std::int64_t>::max();
                    }
                    for (std::size_t r = 0; r < rd; ++r) {
                        std::int64_t v = ip[(o * rd + r) * inner + i];
                        if (op == ReduceOp::Sum || op == ReduceOp::Mean) {
                            acc += v;
                        } else if (op == ReduceOp::Max) {
                            if (v > acc)
                                acc = v;
                        } else {
                            if (v < acc)
                                acc = v;
                        }
                    }
                    outp[o * inner + i] = acc;
                }
            }
        } else {
            ErrorBuilder("cpu_backend::reduce").not_implemented("dtype not supported");
        }
        cur = Storage{CpuStorage{ptr, nb, dt}};
        cur_shape = out_shape;
    }

    if (op == ReduceOp::Mean) {
        auto& cs = std::get<CpuStorage>(cur);
        const std::size_t out_n = cs.nbytes / dtype_size(dt);
        if (dt == Dtype::F32)
            for (std::size_t i = 0; i < out_n; ++i)
                reinterpret_cast<float*>(cs.ptr.get())[i] /= static_cast<float>(total_rd);
        else if (dt == Dtype::F64)
            for (std::size_t i = 0; i < out_n; ++i)
                reinterpret_cast<double*>(cs.ptr.get())[i] /= static_cast<double>(total_rd);
        else if (dt == Dtype::I64 && total_rd > 0)
            for (std::size_t i = 0; i < out_n; ++i)
                reinterpret_cast<std::int64_t*>(cs.ptr.get())[i] /=
                    static_cast<std::int64_t>(total_rd);
    }
    return cur;
}

Storage CpuBackend::reduce_grad_to_shape(const Storage& grad,
                                         const Shape& grad_shape,
                                         const Shape& target_shape,
                                         Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            reduce_grad_to_shape(detail::as_f32(grad), grad_shape, target_shape, Dtype::F32), dt);
    const std::size_t gn = grad_shape.size();
    const std::size_t tn = target_shape.size();
    if (grad_shape == target_shape) {
        const auto& src = std::get<CpuStorage>(grad);
        std::size_t nb = src.nbytes;
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (nb > 0)
            std::memcpy(ptr.get(), src.ptr.get(), nb);
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    (void)gn;
    (void)tn;
    const auto& src = std::get<CpuStorage>(grad);
    std::size_t tnumel = shape_numel(target_shape);
    std::size_t nb = tnumel * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    const BroadcastPlan plan = plan_broadcast(target_shape, grad_shape);

    // The same sum as reduce_broadcast, in runs: the per-element walk
    // divided every flat index back into coordinates, and this sits on
    // the backward of every broadcasting binary op — each bias add.
    auto do_reduce = [&](auto type_tag) {
        using T = decltype(type_tag);
        T* dst_p = reinterpret_cast<T*>(ptr.get());
        std::fill_n(dst_p, tnumel, T{});
        sum_broadcast_runs(plan, reinterpret_cast<const T*>(src.ptr.get()), dst_p);
    };
    if (dt == Dtype::F32)
        do_reduce(float{});
    else if (dt == Dtype::F64)
        do_reduce(double{});
    else if (dt == Dtype::C64)
        do_reduce(std::complex<float>{});
    else if (dt == Dtype::C128)
        do_reduce(std::complex<double>{});
    else
        ErrorBuilder("cpu_backend::reduce_grad_to_shape").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::broadcast_back_for_reduce(const Storage& grad,
                                              const Shape& grad_shape,
                                              const Shape& input_shape,
                                              const std::vector<int>& axes,
                                              bool keepdims,
                                              Dtype dt) {
    // Reached on the *backward* pass of any reduction, so a half model
    // could run forward and then fail at loss.backward().  Data
    // movement only, but it dispatches on dtype, so half needs saying.
    if (detail::is_half_like(dt)) {
        const Storage wide = broadcast_back_for_reduce(
            Storage{detail::widen_half(std::get<CpuStorage>(grad), shape_numel(grad_shape))},
            grad_shape, input_shape, axes, keepdims, Dtype::F32);
        return detail::narrow_half(wide, shape_numel(input_shape), dt);
    }
    Shape kept_shape = input_shape;
    for (int a : axes)
        kept_shape[a] = 1;

    Shape expected_grad;
    {
        std::vector<bool> rm(input_shape.size(), false);
        for (int a : axes)
            rm[a] = true;
        for (std::size_t i = 0; i < input_shape.size(); ++i) {
            if (rm[i]) {
                if (keepdims)
                    expected_grad.push_back(1);
            } else
                expected_grad.push_back(input_shape[i]);
        }
    }
    (void)grad_shape;
    (void)expected_grad;
    const auto& g_cpu = std::get<CpuStorage>(grad);
    std::size_t in_numel = shape_numel(input_shape);
    std::size_t nb = in_numel * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);

    // The reduced gradient, shaped as the input with the reduced axes
    // kept at 1, broadcast back out: a reduction's backward is exactly a
    // broadcast.  The walk this replaced recomputed every coordinate of
    // every element with a division and a modulo per axis — on the
    // backward of every sum and mean.
    const BroadcastPlan plan = plan_broadcast(kept_shape, input_shape);
    auto do_bcast = [&](auto type_tag) {
        using T = decltype(type_tag);
        broadcast_runs(plan, reinterpret_cast<const T*>(g_cpu.ptr.get()),
                       reinterpret_cast<T*>(ptr.get()));
    };
    if (dt == Dtype::F32)
        do_bcast(float{});
    else if (dt == Dtype::F64)
        do_bcast(double{});
    else if (dt == Dtype::C64)
        do_bcast(std::complex<float>{});
    else if (dt == Dtype::C128)
        do_bcast(std::complex<double>{});
    else
        ErrorBuilder("cpu_backend::broadcast_back_for_reduce")
            .not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::reduce_broadcast(const Storage& grad,
                                     const Shape& input_shape,
                                     const Shape& output_shape,
                                     Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            reduce_broadcast(detail::as_f32(grad), input_shape, output_shape, Dtype::F32), dt);
    const auto& gc = std::get<CpuStorage>(grad);
    std::size_t in_numel = shape_numel(input_shape);
    std::size_t nb = in_numel * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    std::memset(ptr.get(), 0, nb);
    const BroadcastPlan plan = plan_broadcast(input_shape, output_shape);

    // The broadcast's runs, read backwards; see sum_broadcast_runs for
    // why the sums are the same bits the element walk produced.
    auto run = [&]<typename T>() {
        sum_broadcast_runs(plan, reinterpret_cast<const T*>(gc.ptr.get()),
                           reinterpret_cast<T*>(ptr.get()));
    };
    if (dt == Dtype::F32)
        run.template operator()<float>();
    else if (dt == Dtype::F64)
        run.template operator()<double>();
    else if (dt == Dtype::I32)
        run.template operator()<std::int32_t>();
    else if (dt == Dtype::I64)
        run.template operator()<std::int64_t>();
    else if (dt == Dtype::C64)
        run.template operator()<std::complex<float>>();
    else if (dt == Dtype::C128)
        run.template operator()<std::complex<double>>();
    else
        ErrorBuilder("cpu_backend::reduce_broadcast").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::histogram_forward(const Storage& input,
                                      const Shape& input_shape,
                                      Dtype input_dtype,
                                      double lo,
                                      double hi,
                                      std::int64_t bins,
                                      bool density) {
    if (detail::is_half_like(input_dtype))
        return histogram_forward(detail::as_f32(input), input_shape, Dtype::F32, lo, hi, bins,
                                 density);
    if (detail::is_narrow_int(input_dtype))
        return histogram_forward(detail::as_i64(input), input_shape, Dtype::I64, lo, hi, bins,
                                 density);
    const auto& cs = std::get<CpuStorage>(input);
    const std::size_t n = shape_numel(input_shape);
    const double step = (hi - lo) / static_cast<double>(bins);
    CpuStorage counts = alloc_cpu(static_cast<std::size_t>(bins), Dtype::F64);
    auto* dst = reinterpret_cast<double*>(counts.ptr.get());
    std::memset(dst, 0, counts.nbytes);
    auto read_val = [&](std::size_t i) -> double {
        switch (input_dtype) {
        case Dtype::F32:
            return static_cast<double>(reinterpret_cast<const float*>(cs.ptr.get())[i]);
        case Dtype::F64:
            return reinterpret_cast<const double*>(cs.ptr.get())[i];
        case Dtype::I64:
            return static_cast<double>(reinterpret_cast<const std::int64_t*>(cs.ptr.get())[i]);
        default:
            ErrorBuilder("cpu::histogram_forward").not_implemented("dtype");
        }
    };
    for (std::size_t i = 0; i < n; ++i) {
        const double v = read_val(i);
        if (v < lo || v > hi)
            continue;
        std::int64_t bin = static_cast<std::int64_t>((v - lo) / step);
        if (bin >= bins)
            bin = bins - 1;
        dst[bin] += 1.0;
    }
    if (density) {
        for (std::int64_t i = 0; i < bins; ++i)
            dst[i] /= (static_cast<double>(n) * step);
    }
    return Storage{std::move(counts)};
}

}  // namespace backend
}  // namespace lucid
