// lucid/_C/ops/ufunc/Reductions.cpp
//
// CPU and GPU reduction kernels and gradient formulas for sum, mean, prod,
// max, min.  The file also contains two anonymous-namespace helpers shared by
// all CPU paths:
//   oir_for_axis       — decomposes a shape into (outer, reduce_dim, inner)
//                        factors for a single axis, used by the strided loop.
//   reduce_one_axis    — applies an Accelerate-backed 1-D reduction kernel over
//                        the inner/outer loop structure for one axis at a time.
//   multi_axis_reduce  — chains reduce_one_axis over a sorted axis list in
//                        descending order so indices remain valid after each
//                        dimension is removed.

#include "Reductions.h"

#include <algorithm>
#include <complex>
#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../../backend/cpu/Reduce.h"
#include "../../backend/gpu/HalfAccumulation.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Half.h"
#include "../../core/OpRegistry.h"
#include "../bfunc/Add.h"
#include "../bfunc/Compare.h"
#include "../bfunc/Div.h"
#include "../bfunc/Mul.h"
#include "../bfunc/Sub.h"
#include "../complex/Conj.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Layout.h"
#include "../utils/Select.h"
#include "Astype.h"
#include "Exponential.h"
#include "Var.h"

namespace lucid {

namespace {

// Outer / reduce_dim / inner decomposition of a shape for a given axis.
// outer   = product of all dims before axis
// reduce  = shape[axis]
// inner   = product of all dims after axis
struct OIR {
    std::size_t outer;
    std::size_t reduce_dim;
    std::size_t inner;
};

// Compute OIR factors so the strided reduction kernel knows how to traverse
// the input buffer without building an explicit index array.
OIR oir_for_axis(const Shape& shape, int axis) {
    OIR r{1, static_cast<std::size_t>(shape[axis]), 1};
    for (int d = 0; d < axis; ++d)
        r.outer *= static_cast<std::size_t>(shape[d]);
    for (std::size_t d = axis + 1; d < shape.size(); ++d)
        r.inner *= static_cast<std::size_t>(shape[d]);
    return r;
}

// Intermediate result of a single-axis reduction (data + new shape).
struct AxisResult {
    CpuStorage data;
    Shape shape;
};

// Product along one axis for integer and bool buffers.
// Accumulates in uint64_t so signed wraparound has defined two's-complement
// semantics and int64 values above 2^53 keep all 64 bits.
template <typename T>
void prod_axis_int(const void* src, void* dst, const OIR& oir, bool as_bool = false) {
    const auto* p = reinterpret_cast<const T*>(src);
    auto* out = reinterpret_cast<T*>(dst);
    for (std::size_t o = 0; o < oir.outer; ++o) {
        for (std::size_t i = 0; i < oir.inner; ++i) {
            std::uint64_t acc = 1;
            for (std::size_t r = 0; r < oir.reduce_dim; ++r) {
                acc *= static_cast<std::uint64_t>(p[(o * oir.reduce_dim + r) * oir.inner + i]);
            }
            out[o * oir.inner + i] = as_bool ? static_cast<T>(acc != 0) : static_cast<T>(acc);
        }
    }
}

template <typename T>
void prod_axis_complex(const void* src, void* dst, const OIR& oir) {
    const auto* p = reinterpret_cast<const std::complex<T>*>(src);
    auto* out = reinterpret_cast<std::complex<T>*>(dst);
    for (std::size_t o = 0; o < oir.outer; ++o) {
        for (std::size_t i = 0; i < oir.inner; ++i) {
            std::complex<T> acc{T(1), T(0)};
            for (std::size_t r = 0; r < oir.reduce_dim; ++r) {
                acc *= p[(o * oir.reduce_dim + r) * oir.inner + i];
            }
            out[o * oir.inner + i] = acc;
        }
    }
}

// Apply a typed Accelerate kernel (k32 for F32, k64 for F64) along one axis.
// Allocates the output buffer, then calls the appropriate typed overload.
template <class Kernel32, class Kernel64>
AxisResult reduce_one_axis(const CpuStorage& in,
                           const Shape& in_shape,
                           int axis,
                           Dtype dt,
                           Kernel32 k32,
                           Kernel64 k64,
                           const char* op_name) {
    const auto oir = oir_for_axis(in_shape, axis);
    Shape out_shape = in_shape;
    out_shape.erase(out_shape.begin() + axis);
    const std::size_t out_numel = (oir.outer * oir.inner);

    AxisResult r;
    r.shape = std::move(out_shape);
    r.data.dtype = dt;
    r.data.nbytes = out_numel * dtype_size(dt);
    r.data.ptr = allocate_aligned_bytes(r.data.nbytes);

    switch (dt) {
    case Dtype::F32:
        k32(reinterpret_cast<const float*>(in.ptr.get()),
            reinterpret_cast<float*>(r.data.ptr.get()), oir.outer, oir.reduce_dim, oir.inner);
        break;
    case Dtype::F64:
        k64(reinterpret_cast<const double*>(in.ptr.get()),
            reinterpret_cast<double*>(r.data.ptr.get()), oir.outer, oir.reduce_dim, oir.inner);
        break;
    case Dtype::C64:
        prod_axis_complex<float>(in.ptr.get(), r.data.ptr.get(), oir);
        break;
    case Dtype::C128:
        prod_axis_complex<double>(in.ptr.get(), r.data.ptr.get(), oir);
        break;
    case Dtype::F16:
    case Dtype::BF16: {
        // No half accumulator on the host: widen, reduce in float, round
        // once on the way back.  Metal reduced F16 / BF16 and the CPU raised,
        // so the same call worked on one device and not the other.
        const bool bf = dt == Dtype::BF16;
        const std::size_t in_numel = shape_numel(in_shape);
        std::vector<float> wide_in(in_numel), wide_out(out_numel);
        const auto* src = reinterpret_cast<const std::uint16_t*>(in.ptr.get());
        backend::detail::half_or_bfloat16_to_float_bulk(src, wide_in.data(), in_numel, bf);
        k32(wide_in.data(), wide_out.data(), oir.outer, oir.reduce_dim, oir.inner);
        auto* dst = reinterpret_cast<std::uint16_t*>(r.data.ptr.get());
        backend::detail::float_to_half_or_bfloat16_bulk(wide_out.data(), dst, out_numel, bf);
        break;
    }
    case Dtype::Bool:
        prod_axis_int<std::uint8_t>(in.ptr.get(), r.data.ptr.get(), oir, /*as_bool=*/true);
        break;
    case Dtype::I8:
        prod_axis_int<std::int8_t>(in.ptr.get(), r.data.ptr.get(), oir);
        break;
    case Dtype::I16:
        prod_axis_int<std::int16_t>(in.ptr.get(), r.data.ptr.get(), oir);
        break;
    case Dtype::I32:
        prod_axis_int<std::int32_t>(in.ptr.get(), r.data.ptr.get(), oir);
        break;
    case Dtype::I64:
        prod_axis_int<std::int64_t>(in.ptr.get(), r.data.ptr.get(), oir);
        break;
    default:
        ErrorBuilder(op_name).not_implemented("dtype not supported");
    }
    return r;
}

// Reduce over multiple axes by iterating them in descending order.
// Descending order ensures that lower axis indices remain valid as dimensions
// are successively removed from current_shape after each single-axis pass.
// keepdims support is tracked but the actual shape expansion happens at the
// ReduceKernel level; here we only care about the final data layout.
template <class Kernel32, class Kernel64>
CpuStorage multi_axis_reduce(const CpuStorage& a,
                             const Shape& in_shape,
                             const std::vector<int>& axes,
                             bool keepdims,
                             Dtype dt,
                             Kernel32 k32,
                             Kernel64 k64,
                             const char* op_name) {
    std::vector<int> ax_desc(axes.rbegin(), axes.rend());

    CpuStorage current = a;
    Shape current_shape = in_shape;
    for (int ax : ax_desc) {
        auto r = reduce_one_axis(current, current_shape, ax, dt, k32, k64, op_name);
        current = std::move(r.data);
        current_shape = std::move(r.shape);
    }

    if (keepdims) {
        // keepdims shape construction is handled by the ReduceKernel caller;
        // the kept variable below would hold it but is intentionally unused.
        Shape kept = in_shape;
        for (int ax : axes)
            kept[ax] = 1;

        (void)kept;
    }
    return current;
}

}  // namespace

// sum — broadcast-back is O(n) and requires no saved tensors.
//
// KeepInput, as for mean and prod below: inside an autocast scope a
// reduction runs in its input's dtype.  It was Promote, which cast a float32
// input down to float16 first, so ``full((64, 1024), 1.5).sum()`` under
// autocast answered float16 inf where the reference gives float32 98304 —
// and an int64 or float64 sum was demoted the same way.  ForceFP32 would
// fix the float32 case but turn every integer sum into float32: the
// SchemaGuard reads ForceFP32 as "the result is a real number" outside an
// autocast scope too.  The one remaining difference from the reference is
// a float16 input under a Metal scope, which the reference's GPU autocast
// widens to float32 (its CPU autocast does not); here it stays float16,
// accumulated in float32 by the kernel (backend/gpu/HalfAccumulation.h).
const OpSchema SumBackward::schema_v1{"sum", 1, AmpPolicy::KeepInput, true};

// dL/dx = broadcast(dL/dy) back to the original input shape along reduce_axes_.
Storage SumBackward::grad_formula(const Storage& grad_out) {
    return broadcast_back_for_reduce(grad_out, this->out_shape_, this->full_input_shape_,
                                     this->reduce_axes_, this->keepdims_, this->dtype_,
                                     this->device_);
}

// Reference-framework-compatible promotion for integer/bool reductions.  ``bool / int8 /
// int16 / int32`` all promote to ``int64`` before reducing — preserves
// overflow headroom for sum/prod and matches the reference framework's
// observable behaviour.  Floats / complex pass through unchanged.
//
// Without this, ``bool.sum()`` raised NotImplementedError on CPU and
// silently returned a 0-d bool (acting like ``any()``) on GPU — found
// during the M4 Max parity smoke when training accuracy reported 1.56%
// (= 1/batch) instead of the real count.
TensorImplPtr promote_int_for_reduce(const TensorImplPtr& a) {
    switch (a->dtype()) {
    case Dtype::Bool:
    case Dtype::I8:
    case Dtype::I16:
    case Dtype::I32:
        return astype_op(a, Dtype::I64);
    default:
        return a;
    }
}

TensorImplPtr sum_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims) {
    return SumBackward::forward(promote_int_for_reduce(a), axes, keepdims);
}
LUCID_REGISTER_OP(SumBackward)

// mean — KeepInput (see sum); divides broadcast gradient by the count of
// reduced elements.  The reference's autocast lists mean nowhere, so it
// runs in the input dtype there too.
const OpSchema MeanBackward::schema_v1{"mean", 1, AmpPolicy::KeepInput, true, "", true};

namespace {
// Count the number of elements collapsed by the given axes.
double reduced_count(const Shape& in_shape, const std::vector<int>& axes) {
    double n = 1.0;
    for (int ax : axes)
        n *= static_cast<double>(in_shape[ax]);
    return n;
}

// The share of a slice's gradient each position gets under max / min:
// ``mask / (ties + 1 - mask)``.  Where ``mask`` is 1 the denominator is
// the number of positions tied for the extremum, so the shares sum to one
// — a subgradient, as the reference framework's amax gives.  Where it is
// 0 the denominator is at least 1, so a slice nothing matches (a NaN
// extremum) gets zeros rather than 0 / 0.
Storage share_among_ties(const Storage& mask,
                         const Shape& in_shape,
                         const std::vector<int>& axes,
                         Dtype dt,
                         Device device) {
    const std::size_t n = shape_numel(in_shape);
    Shape kept = in_shape;
    for (int ax : axes)
        kept[static_cast<std::size_t>(ax)] = 1;
    Storage ties =
        backend::Dispatcher::for_device(device).reduce_sum(mask, in_shape, {axes, true}, dt);
    Storage ties_b = broadcast_back_for_reduce(ties, kept, in_shape, axes, true, dt, device);
    Storage denom =
        subtract_storages(add_scalar_storage(ties_b, 1.0, n, dt, device), mask, n, dt, device);
    return divide_storages(mask, denom, n, dt, device);
}

// Graph-mode counterpart of :func:`share_among_ties`, built from ops so a
// second derivative passes through it (the shares are piecewise constant
// in ``x``, so they contribute nothing of their own).
TensorImplPtr share_among_ties_graph(const TensorImplPtr& mask,
                                     const Shape& in_shape,
                                     const std::vector<int>& axes) {
    auto ties = broadcast_to_op(sum_op(mask, axes, /*keepdims=*/true), in_shape);
    auto denom = sub_op(add_op(ties, ones_like_op(mask)), mask);
    return div_op(mask, denom);
}
}  // namespace

// dL/dx = broadcast(dL/dy / N)  where N is the number of reduced elements.
Storage MeanBackward::grad_formula(const Storage& grad_out) {
    const double n = reduced_count(this->full_input_shape_, this->reduce_axes_);
    const std::size_t out_n = shape_numel(this->out_shape_);
    auto& be = backend::Dispatcher::for_device(this->device_);
    const bool half = is_half_float(this->dtype_);
    const Dtype div_dt = half ? Dtype::F32 : this->dtype_;
    Storage g = half ? be.cast(grad_out, this->out_shape_, this->dtype_, Dtype::F32) : grad_out;
    Storage denom = be.full(this->out_shape_, div_dt, n);
    Storage scaled = divide_storages(g, denom, out_n, div_dt, this->device_);
    if (half)
        scaled = be.cast(scaled, this->out_shape_, Dtype::F32, this->dtype_);
    return broadcast_back_for_reduce(scaled, this->out_shape_, this->full_input_shape_,
                                     this->reduce_axes_, this->keepdims_, this->dtype_,
                                     this->device_);
}

// Graph-mode mean backward: divide the broadcast-expanded gradient by n_reduced.
TensorImplPtr MeanBackward::scale_graph_grad(const TensorImplPtr& g) {
    const double n = reduced_count(this->full_input_shape_, this->reduce_axes_);
    if (is_half_float(g->dtype())) {
        auto g32 = astype_op(g, Dtype::F32);
        auto n_tensor = full_like_op(g32, n, /*requires_grad=*/false);
        return astype_op(div_op(g32, n_tensor), g->dtype());
    }
    // Create a constant tensor with value n (same shape as g) and divide.
    // The division is tracked by div_op so second-order gradients flow correctly.
    auto n_tensor = full_like_op(g, n, /*requires_grad=*/false);
    return div_op(g, n_tensor);
}

TensorImplPtr mean_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims) {
    return MeanBackward::forward(a, axes, keepdims);
}
LUCID_REGISTER_OP(MeanBackward)

// prod — CPU uses Accelerate via multi_axis_reduce; GPU uses mlx::core::prod.
// KeepInput for the reason sum gives.
const OpSchema ProdBackward::schema_v1{"prod", 1, AmpPolicy::KeepInput, true};

// Apply the multi_axis_reduce helper with Accelerate prod kernels.
CpuStorage ProdBackward::cpu_kernel(const CpuStorage& a,
                                    const Shape& in_shape,
                                    const std::vector<int>& axes,
                                    bool keepdims,
                                    Dtype dt) {
    return multi_axis_reduce(a, in_shape, axes, keepdims, dt, backend::cpu::prod_axis_f32,
                             backend::cpu::prod_axis_f64, "prod");
}

namespace {

// d prod(x) / d x_i over ``axes``, at x's shape: the product of every other
// element of x_i's slice.  That is prod / x_i only where x_i is not zero —
// at a zero, 0 / 0 gave NaN where the reference gives the others' product —
// so the zeros are split out instead of divided by.  With p the product of a
// slice's non-zero elements and Z that of its zeros (1 when it has none):
//   x_i != 0:  p / x_i * Z
//   x_i == 0:  p * (the slice's other zeros' product)
// The second factor is 1 for a lone zero and 0 for three or more.  For two
// it is the other zero, whose value is 0 but whose derivative is the one the
// product has; Z, likewise, is a product of x rather than a 0/1 constant.
// Both keep this formula's own derivative — the second derivative of prod —
// right at the zeros.  Only graph mode differentiates it, so only graph mode
// (``differentiable``) pays for the pair's term, which is 0 in value.
TensorImplPtr
prod_of_others(const TensorImplPtr& x, const std::vector<int>& axes, bool differentiable) {
    const Shape& shape = x->shape();
    auto slice_wide = [&](const TensorImplPtr& t) { return broadcast_to_op(t, shape); };
    auto one = ones_like_op(x);
    auto nil = zeros_like_op(x);
    auto zero = equal_op(x, nil);
    auto x_1 = where_op(zero, one, x);

    auto nonzero_prod = slice_wide(prod_op(x_1, axes, true));
    auto zero_count = slice_wide(sum_op(where_op(zero, one, nil), axes, true));
    auto count_is = [&](double n) { return equal_op(zero_count, full_like_op(zero_count, n)); };
    auto at_nonzero = div_op(nonzero_prod, x_1);
    if (!differentiable) {
        return where_op(zero, where_op(count_is(1.0), nonzero_prod, nil),
                        where_op(count_is(0.0), at_nonzero, nil));
    }
    auto zeros_prod = slice_wide(prod_op(where_op(zero, x, one), axes, true));
    at_nonzero = mul_op(at_nonzero, zeros_prod);
    auto other_zeros = where_op(count_is(1.0), one, nil);
    // (sum of the slice's zeros) - x_i: the other zero of a pair.
    auto zeros_sum = slice_wide(sum_op(where_op(zero, x, nil), axes, true));
    other_zeros = where_op(count_is(2.0), sub_op(zeros_sum, x), other_zeros);
    return where_op(zero, mul_op(nonzero_prod, other_zeros), at_nonzero);
}

template <typename T>
void cpu_prod_grad_row(const T* xo, T go, T* dxo, std::size_t n) {
    T nz = T(1);
    int zc = 0;
    for (std::size_t r = 0; r < n; ++r) {
        const bool is_z = xo[r] == T(0);
        zc += is_z ? 1 : 0;
        nz *= is_z ? T(1) : xo[r];
    }
    if (zc == 0) {
        for (std::size_t r = 0; r < n; ++r)
            dxo[r] = go * (nz / xo[r]);
        return;
    }
    std::memset(dxo, 0, n * sizeof(T));
    if (zc == 1) {
        for (std::size_t r = 0; r < n; ++r) {
            if (xo[r] == T(0)) {
                dxo[r] = go * nz;
                break;
            }
        }
    }
}

template <typename T>
Storage cpu_prod_grad_1d(const CpuStorage& x_in, const CpuStorage& g_in, const OIR& oir, Dtype dt) {
    const std::size_t in_n = oir.outer * oir.reduce_dim * oir.inner;
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = in_n * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    const auto* x = reinterpret_cast<const T*>(x_in.ptr.get());
    const auto* g = reinterpret_cast<const T*>(g_in.ptr.get());
    auto* dx = reinterpret_cast<T*>(out.ptr.get());
    if (oir.inner == 1) {
        for (std::size_t o = 0; o < oir.outer; ++o)
            cpu_prod_grad_row(x + o * oir.reduce_dim, g[o], dx + o * oir.reduce_dim,
                              oir.reduce_dim);
        return out;
    }
    std::vector<T> nz_prod(oir.inner);
    std::vector<int> zero_cnt(oir.inner);
    for (std::size_t o = 0; o < oir.outer; ++o) {
        std::fill(nz_prod.begin(), nz_prod.end(), T(1));
        std::fill(zero_cnt.begin(), zero_cnt.end(), 0);
        const T* xo = x + o * oir.reduce_dim * oir.inner;
        const T* go = g + o * oir.inner;
        T* dxo = dx + o * oir.reduce_dim * oir.inner;
        for (std::size_t r = 0; r < oir.reduce_dim; ++r) {
            const T* xr = xo + r * oir.inner;
            for (std::size_t i = 0; i < oir.inner; ++i) {
                const bool is_z = xr[i] == T(0);
                zero_cnt[i] += is_z ? 1 : 0;
                nz_prod[i] *= is_z ? T(1) : xr[i];
            }
        }
        for (std::size_t r = 0; r < oir.reduce_dim; ++r) {
            const T* xr = xo + r * oir.inner;
            T* dxr = dxo + r * oir.inner;
            for (std::size_t i = 0; i < oir.inner; ++i) {
                const int zc = zero_cnt[i];
                dxr[i] = zc == 0 ? go[i] * (nz_prod[i] / xr[i])
                                 : (zc == 1 && xr[i] == T(0) ? go[i] * nz_prod[i] : T(0));
            }
        }
    }
    return out;
}

}  // namespace

// dL/dx_i = dL/dy * (the product of x_i's slice without x_i), from the input
// saved by value — the graph-mode formula below, value for value, under
// no-grad.
Storage ProdBackward::grad_formula(const Storage& grad_out) {
    const Shape& shape = this->full_input_shape_;
    const bool one_or_all = this->reduce_axes_.size() == 1 ||
                            (!shape.empty() && this->reduce_axes_.size() == shape.size());
    if (this->device_ == Device::CPU && one_or_all && !PhantomMode::is_enabled() &&
        std::holds_alternative<CpuStorage>(this->saved_inputs_[0]) &&
        std::holds_alternative<CpuStorage>(grad_out)) {
        const OIR oir = this->reduce_axes_.size() == 1 ? oir_for_axis(shape, this->reduce_axes_[0])
                                                       : OIR{1, shape_numel(shape), 1};
        const auto& x_cpu = std::get<CpuStorage>(this->saved_inputs_[0]);
        const auto& g_cpu = std::get<CpuStorage>(grad_out);
        if (this->dtype_ == Dtype::F32)
            return cpu_prod_grad_1d<float>(x_cpu, g_cpu, oir, Dtype::F32);
        if (this->dtype_ == Dtype::F64)
            return cpu_prod_grad_1d<double>(x_cpu, g_cpu, oir, Dtype::F64);
    }
    NoGradGuard no_grad;
    Storage g_bcast =
        broadcast_back_for_reduce(grad_out, this->out_shape_, shape, this->reduce_axes_,
                                  this->keepdims_, this->dtype_, this->device_);
    auto g =
        std::make_shared<TensorImpl>(std::move(g_bcast), shape, this->dtype_, this->device_, false);
    auto x = std::make_shared<TensorImpl>(this->saved_inputs_[0], shape, this->dtype_,
                                          this->device_, false);
    auto others = prod_of_others(x, this->reduce_axes_, false);
    if (is_complex(this->dtype_))
        others = conj_op(others);
    return mul_op(g, others)->storage();
}

// Graph-mode prod backward.  The products are recomputed from the saved
// input impl rather than read from ``saved_output_``: the storage carries no
// graph, and a second derivative taken through it would be silently wrong
// even while the first stayed right.
TensorImplPtr ProdBackward::scale_graph_grad(const TensorImplPtr& g) {
    auto others = prod_of_others(this->saved_impl_inputs_[0], this->reduce_axes_, true);
    if (is_complex(this->dtype_))
        others = conj_op(others);
    return mul_op(g, others);
}

TensorImplPtr prod_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims) {
    return ProdBackward::forward(promote_int_for_reduce(a), axes, keepdims);
}
LUCID_REGISTER_OP(ProdBackward)

// max — saves the output (max values) to build the equality mask cheaply.
const OpSchema MaxBackward::schema_v1{"max", 1, AmpPolicy::KeepInput, true};

// dL/dx = dL/dy * (x == max_val), broadcast back to the input shape.
// The equality mask is constructed as ge(x, m) AND ge(m, x) to avoid a
// dedicated equal-mask primitive; this handles ties correctly.
Storage MaxBackward::grad_formula(const Storage& grad_out) {
    const std::size_t in_numel = shape_numel(this->full_input_shape_);
    Storage out_bcast =
        broadcast_back_for_reduce(this->saved_output_, this->out_shape_, this->full_input_shape_,
                                  this->reduce_axes_, this->keepdims_, this->dtype_, this->device_);

    Storage mask_eq;
    {
        // a == b iff a >= b AND b >= a, avoiding a separate equality kernel.
        Storage ge_a = ge_mask_storage(this->saved_inputs_[0], out_bcast, in_numel, this->dtype_,
                                       this->device_);
        Storage ge_b = ge_mask_storage(out_bcast, this->saved_inputs_[0], in_numel, this->dtype_,
                                       this->device_);
        mask_eq = multiply_storages(ge_a, ge_b, in_numel, this->dtype_, this->device_);
    }
    Storage share = share_among_ties(mask_eq, this->full_input_shape_, this->reduce_axes_,
                                     this->dtype_, this->device_);

    Storage g_bcast =
        broadcast_back_for_reduce(grad_out, this->out_shape_, this->full_input_shape_,
                                  this->reduce_axes_, this->keepdims_, this->dtype_, this->device_);
    return multiply_storages(g_bcast, share, in_numel, this->dtype_, this->device_);
}

// Graph-mode max backward: the same shares as ``grad_formula``, from ops.
TensorImplPtr MaxBackward::scale_graph_grad(const TensorImplPtr& g) {
    const auto& x = this->saved_impl_inputs_[0];
    auto out = max_op(x, this->reduce_axes_, /*keepdims=*/true);
    auto out_b = broadcast_to_op(out, this->full_input_shape_);
    auto mask = astype_op(equal_op(x, out_b), this->dtype_);
    return mul_op(g, share_among_ties_graph(mask, this->full_input_shape_, this->reduce_axes_));
}

TensorImplPtr max_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims) {
    return MaxBackward::forward(a, axes, keepdims);
}
LUCID_REGISTER_OP(MaxBackward)

// min — symmetric to max; the same equality-mask trick applies.
const OpSchema MinBackward::schema_v1{"min", 1, AmpPolicy::KeepInput, true};

// dL/dx = dL/dy * (x == min_val), broadcast back to the input shape.
Storage MinBackward::grad_formula(const Storage& grad_out) {
    const std::size_t in_numel = shape_numel(this->full_input_shape_);
    Storage out_bcast =
        broadcast_back_for_reduce(this->saved_output_, this->out_shape_, this->full_input_shape_,
                                  this->reduce_axes_, this->keepdims_, this->dtype_, this->device_);
    Storage mask_eq;
    {
        Storage ge_a = ge_mask_storage(this->saved_inputs_[0], out_bcast, in_numel, this->dtype_,
                                       this->device_);
        Storage ge_b = ge_mask_storage(out_bcast, this->saved_inputs_[0], in_numel, this->dtype_,
                                       this->device_);
        mask_eq = multiply_storages(ge_a, ge_b, in_numel, this->dtype_, this->device_);
    }
    Storage share = share_among_ties(mask_eq, this->full_input_shape_, this->reduce_axes_,
                                     this->dtype_, this->device_);
    Storage g_bcast =
        broadcast_back_for_reduce(grad_out, this->out_shape_, this->full_input_shape_,
                                  this->reduce_axes_, this->keepdims_, this->dtype_, this->device_);
    return multiply_storages(g_bcast, share, in_numel, this->dtype_, this->device_);
}

// Graph-mode min backward — the mirror of :meth:`MaxBackward::scale_graph_grad`.
TensorImplPtr MinBackward::scale_graph_grad(const TensorImplPtr& g) {
    const auto& x = this->saved_impl_inputs_[0];
    auto out = min_op(x, this->reduce_axes_, /*keepdims=*/true);
    auto out_b = broadcast_to_op(out, this->full_input_shape_);
    auto mask = astype_op(equal_op(x, out_b), this->dtype_);
    return mul_op(g, share_among_ties_graph(mask, this->full_input_shape_, this->reduce_axes_));
}

TensorImplPtr min_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims) {
    return MinBackward::forward(a, axes, keepdims);
}
LUCID_REGISTER_OP(MinBackward)

namespace {
// Thin helper: validate the GPU input and delegate to an MLX lambda, then wrap
// the resulting mlx::array back into a GpuStorage.
template <class F>
GpuStorage gpu_reduce_apply(const GpuStorage& a, Dtype dt, F&& f, const char* op) {
    if (!a.arr) {
        ErrorBuilder(op).fail("null GPU input");
    }
    auto out = f(*a.arr);
    return gpu::wrap_mlx_array(std::move(out), dt);
}
}  // namespace

// GPU prod: mlx::core::prod with the requested axes.  MLX multiplies in the
// input dtype, so a half input is multiplied in float32 and rounded back:
// a float16 product of 1000 x 1.01 came to 15672 where the reference
// gives 16624.
GpuStorage ProdBackward::gpu_kernel(
    const GpuStorage& a, const Shape&, const std::vector<int>& axes, bool keepdims, Dtype dt) {
    return gpu_reduce_apply(
        a, dt,
        [&axes, keepdims](const auto& x) {
            return backend::reduce_widened(x, [&](const ::mlx::core::array& w) {
                return ::mlx::core::prod(w, axes, keepdims);
            });
        },
        "prod");
}

// std_op composes sqrt and var entry points so that the standard autograd graph
// handles the gradient automatically via the chain rule:
//   std = sqrt(var(x, axes, keepdims))
// No new backward node is required; SqrtBackward and VarBackward cover it.
TensorImplPtr std_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims) {
    auto out = sqrt_op(var_op(a, axes, keepdims));
    return (a && is_half_float(a->dtype())) ? astype_op(out, a->dtype()) : out;
}

}  // namespace lucid
