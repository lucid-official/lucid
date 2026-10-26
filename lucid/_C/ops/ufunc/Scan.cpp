// lucid/_C/ops/ufunc/Scan.cpp
//
// Cumulative scan forward and backward pass implementations.
//
// Two anonymous-namespace helpers wrap the backend calls to keep apply()
// implementations readable:
//   reverse_along_axis_storage — reverses elements along one axis.
//   cumsum_storage_along       — inclusive prefix-sum along one axis.
//
// Backward for cumsum (reverse-cumsum trick):
//   Given dL/dy (upstream gradient of the output), the gradient for position i
//   is the sum of dL/dy_j for all j >= i.  This equals:
//     dx = reverse(cumsum(reverse(dy)))
//   which avoids an explicit O(n^2) summation.
//
// Backward for cumprod (``cumprod_grad``):
//   dx_i = sum_{j >= i} dy_j * prod_{k <= j, k != i} x_k, which is
//   reverse(cumsum(reverse(dy * y)))_i / x_i up to the first zero of x along
//   the axis; the first zero and what follows it are formed without that
//   division.

#include "Scan.h"

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/AutogradNode.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/MemoryStats.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/SchemaGuard.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/Compare.h"
#include "../bfunc/Div.h"
#include "../bfunc/Mul.h"
#include "../bfunc/_BinaryOp.h"
#include "../complex/Conj.h"
#include "../composite/Logical.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Layout.h"
#include "../utils/ScalarAxis.h"
#include "../utils/Select.h"
#include "Astype.h"
#include "Reductions.h"
#include "_Detail.h"

namespace lucid {

namespace {

using ufunc_detail::fresh;

// Promote ``Bool / I8 / I16 / I32`` to ``I64`` before an integer scan, exactly
// as sum/prod do (see ``promote_int_for_reduce`` in Reductions.cpp).  Without
// it ``int32.cumsum()`` accumulates in int32 and silently overflows, while
// ``int32.sum()`` promotes — an inconsistency the reference framework does not
// have.  Floats / I64 pass through unchanged.
TensorImplPtr promote_int_for_scan(const TensorImplPtr& a) {
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

// Thin wrapper: reverse the storage contents along the given axis.
Storage reverse_along_axis_storage(
    const Storage& s, const Shape& shape, int axis, Dtype dt, Device device) {
    return backend::Dispatcher::for_device(device).reverse_along_axis(s, shape, axis, dt);
}

// Thin wrapper: compute the inclusive prefix sum along the given axis.
Storage
cumsum_storage_along(const Storage& s, const Shape& shape, int axis, Dtype dt, Device device) {
    return backend::Dispatcher::for_device(device).cumsum(s, shape, axis, dt);
}

// Private backward node for cumsum.
//
// Saved state:
//   input_shape_ — shape of the forward input (= shape of the output).
//   axis_        — normalised (non-negative) reduction axis.
class CumsumBackward : public AutogradNode<CumsumBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;

    // dx = reverse(cumsum(reverse(dy))):
    // 1. Reverse dy along axis to turn suffix-sums into prefix-sums.
    // 2. Apply cumsum to accumulate the reversed gradients.
    // 3. Reverse again to restore the original axis order.
    std::vector<Storage> apply(Storage grad_out) override {
        Storage rev = reverse_along_axis_storage(grad_out, input_shape_, axis_, dtype_, device_);
        Storage cs = cumsum_storage_along(rev, input_shape_, axis_, dtype_, device_);
        Storage dx = reverse_along_axis_storage(cs, input_shape_, axis_, dtype_, device_);
        return {std::move(dx)};
    }

    // The same reverse-cumsum-reverse, recorded.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        return {flip_op(cumsum_op(flip_op(grad_out, {axis_}), axis_), {axis_})};
    }
};

// d cumprod(x) / dx contracted with g along ``axis``, given y = cumprod(x):
// dx_i = sum_{k >= i} g_k * prod_{j <= k, j != i} x_j.
//
// Dividing the suffix sum of g * y by x_i gives that only before the slice's
// first zero; at and after it, it was 0 / 0 — NaN where the reference has a
// value.  So the slice is cut at its first zero z (the reference's rule):
//   i < z:  suffix(g * y)_i / x_i                    (x_i is not zero)
//   i = z:  prod_{j<z} x_j * sum_{k>=z} g_k * r_k    r_k = prod_{z<j<=k} x_j
//   i > z:  0 — every product holds x_z.  Graph mode (``differentiable``)
//           carries the dependence on x_z across the non-zero positions
//           between z and the second zero z_2 (``at_between``) and at z_2
//           itself (``at_second``), so cumprod's second derivative is exact
//           for any number of zeros per slice.
// Every piece is an op, so the formula serves eager and graph-mode backward.
TensorImplPtr cumprod_grad(const TensorImplPtr& g,
                           const TensorImplPtr& x_in,
                           const TensorImplPtr& y_in,
                           int axis,
                           bool differentiable) {
    const bool cplx = is_complex(x_in->dtype());
    const TensorImplPtr x = cplx ? conj_op(x_in) : x_in;
    const TensorImplPtr y = cplx ? conj_op(y_in) : y_in;
    const Shape& shape = x->shape();
    auto slice_wide = [&](const TensorImplPtr& t) { return broadcast_to_op(t, shape); };
    auto suffix_sum = [axis](const TensorImplPtr& t) {
        return flip_op(cumsum_op(flip_op(t, {axis}), axis), {axis});
    };
    auto one = ones_like_op(x);
    auto nil = zeros_like_op(x);
    auto zero = equal_op(x, nil);
    auto zeros_so_far = cumsum_op(zero, axis);  // int64
    auto so_far_is = [&](double count) {
        return equal_op(zeros_so_far, full_like_op(zeros_so_far, count));
    };
    auto before = so_far_is(0.0);
    auto first = logical_and_op(zero, so_far_is(1.0));
    auto x_1 = where_op(zero, one, x);

    auto at_before = div_op(suffix_sum(mul_op(g, y)), x_1);
    auto lead = slice_wide(prod_op(where_op(before, x, one), {axis}, true));  // prod_{j<z} x_j
    auto restarted = cumprod_op(where_op(logical_or_op(before, first), one, x), axis);
    auto weighted = mul_op(where_op(before, nil, g), restarted);
    auto at_first = mul_op(lead, slice_wide(sum_op(weighted, {axis}, true)));
    auto after = nil;
    if (differentiable) {
        auto x_first = slice_wide(sum_op(where_op(first, x, nil), {axis}, true));
        auto between = logical_and_op(logical_not_op(zero), so_far_is(1.0));
        auto second = logical_and_op(zero, so_far_is(2.0));
        auto before_second = logical_or_op(before, so_far_is(1.0));
        auto lead_second =
            slice_wide(prod_op(where_op(logical_or_op(before, between), x, one), {axis}, true));
        auto restarted_second =
            cumprod_op(where_op(logical_or_op(before_second, second), one, x), axis);
        auto weighted_second = mul_op(where_op(before_second, nil, g), restarted_second);
        auto at_second =
            mul_op(mul_op(x_first, lead_second), slice_wide(sum_op(weighted_second, {axis}, true)));
        auto at_between = mul_op(mul_op(x_first, lead), div_op(suffix_sum(weighted), x_1));
        after = where_op(second, at_second, where_op(between, at_between, nil));
    }
    return where_op(before, at_before, where_op(first, at_first, after));
}

template <typename T>
Storage cpu_cumprod_grad(const CpuStorage& g_in,
                         const CpuStorage& x_in,
                         const CpuStorage& y_in,
                         const Shape& shape,
                         int axis,
                         Dtype dt) {
    std::size_t outer = 1, inner = 1;
    const auto len = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d)
        inner *= static_cast<std::size_t>(shape[d]);
    CpuStorage out = helpers::allocate_cpu(shape, dt);
    const auto* g = reinterpret_cast<const T*>(g_in.ptr.get());
    const auto* x = reinterpret_cast<const T*>(x_in.ptr.get());
    const auto* y = reinterpret_cast<const T*>(y_in.ptr.get());
    auto* dx = reinterpret_cast<T*>(out.ptr.get());
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t in = 0; in < inner; ++in) {
            const std::size_t base = o * len * inner + in;
            std::size_t z = len;
            T lead = T(1);
            for (std::size_t r = 0; r < len; ++r) {
                const T xv = x[base + r * inner];
                if (xv == T(0)) {
                    z = r;
                    break;
                }
                lead *= xv;
            }
            if (z < len) {
                T run = T(1), wsum = g[base + z * inner];
                for (std::size_t r = z + 1; r < len; ++r) {
                    dx[base + r * inner] = T(0);
                    run *= x[base + r * inner];
                    wsum += g[base + r * inner] * run;
                }
                dx[base + z * inner] = lead * wsum;
            }
            T suf = T(0);
            for (std::size_t r = len; r-- > z;)
                suf += g[base + r * inner] * y[base + r * inner];
            for (std::size_t r = z; r-- > 0;) {
                const std::size_t idx = base + r * inner;
                suf += g[idx] * y[idx];
                dx[idx] = suf / x[idx];
            }
        }
    }
    return out;
}

// Private backward node for cumprod.
//
// Saved state:
//   input_shape_  — shape of the forward input.
//   axis_         — normalised reduction axis.
//   saved_output_ — copy of the forward output (cumprod) tensor's storage.
class CumprodBackward : public AutogradNode<CumprodBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;

    // From the values saved at forward, under no-grad.
    std::vector<Storage> apply(Storage grad_out) override {
        if (device_ == Device::CPU && !PhantomMode::is_enabled() &&
            std::holds_alternative<CpuStorage>(grad_out) &&
            std::holds_alternative<CpuStorage>(saved_inputs_[0]) &&
            std::holds_alternative<CpuStorage>(saved_output_)) {
            const auto& g_cpu = std::get<CpuStorage>(grad_out);
            const auto& x_cpu = std::get<CpuStorage>(saved_inputs_[0]);
            const auto& y_cpu = std::get<CpuStorage>(saved_output_);
            if (dtype_ == Dtype::F32) {
                return {
                    cpu_cumprod_grad<float>(g_cpu, x_cpu, y_cpu, input_shape_, axis_, Dtype::F32)};
            }
            if (dtype_ == Dtype::F64) {
                return {
                    cpu_cumprod_grad<double>(g_cpu, x_cpu, y_cpu, input_shape_, axis_, Dtype::F64)};
            }
        }
        NoGradGuard no_grad;
        auto wrap = [this](Storage s) {
            return std::make_shared<TensorImpl>(std::move(s), input_shape_, dtype_, device_, false);
        };
        return {cumprod_grad(wrap(std::move(grad_out)), wrap(saved_inputs_[0]), wrap(saved_output_),
                             axis_, false)
                    ->storage()};
    }

    // The same formula, recorded, with the product recomputed from the input
    // so it carries its own dependence on x — the saved one is data only.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        const auto& x = saved_impl_inputs_[0];
        if (!x)
            ErrorBuilder("cumprod").fail("graph-mode backward is missing its saved input");
        return {cumprod_grad(grad_out, x, cumprod_op(x, axis_), axis_, true)};
    }
};

// KeepInput: scans are valid for integer types; no promotion needed.
const OpSchema CumsumBackward::schema_v1{"cumsum", 1, AmpPolicy::KeepInput, true};
const OpSchema CumprodBackward::schema_v1{"cumprod", 1, AmpPolicy::KeepInput, true};

// Shared forward dispatch for both scan ops.  Validates the input, normalises
// the axis, dispatches to the backend, and returns a fresh output tensor.
// Autograd wiring is left to the callers so that each can save the appropriate
// tensors (cumsum saves nothing extra; cumprod saves x and y).
TensorImplPtr scan_dispatch(const TensorImplPtr& a, int axis, bool is_prod, const char* name) {
    Validator::input(a, std::string(name) + ".a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    auto sh = a->shape();
    // The public ops route a 0-d input through on_scalar_axis first.
    const int ax = wrap_dim(axis, static_cast<int>(sh.size()), name);
    OpScopeFull scope{name, device, dt, sh};
    scope.set_attr("axis", static_cast<std::int64_t>(ax));

    Storage out_storage =
        is_prod ? backend::Dispatcher::for_device(device).cumprod(a->storage(), sh, ax, dt)
                : backend::Dispatcher::for_device(device).cumsum(a->storage(), sh, ax, dt);
    return fresh(std::move(out_storage), sh, dt, device);
}

}  // namespace

// Dispatch cumsum, then wire CumsumBackward.  The axis is re-normalised here
// (after scan_dispatch validated it) so that bwd->axis_ is always non-negative.
TensorImplPtr cumsum_op(const TensorImplPtr& a_in, int axis) {
    Validator::input(a_in, "cumsum.a").non_null();
    if (a_in->shape().empty())
        return on_scalar_axis(a_in, axis, "cumsum", cumsum_op);
    SchemaGuard sg{CumsumBackward::schema_v1, a_in->dtype(), a_in->device()};
    const auto a = promote_int_for_scan(astype_op(a_in, sg.effective_dtype()));
    auto out = scan_dispatch(a, axis, false, "cumsum");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CumsumBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    kernel::NaryKernel<CumsumBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

// Dispatch cumprod, then wire CumprodBackward with both input and output saved.
TensorImplPtr cumprod_op(const TensorImplPtr& a_in, int axis) {
    Validator::input(a_in, "cumprod.a").non_null();
    if (a_in->shape().empty())
        return on_scalar_axis(a_in, axis, "cumprod", cumprod_op);
    SchemaGuard sg{CumprodBackward::schema_v1, a_in->dtype(), a_in->device()};
    const auto a = promote_int_for_scan(astype_op(a_in, sg.effective_dtype()));
    auto out = scan_dispatch(a, axis, true, "cumprod");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CumprodBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    bwd->keep_output(out->storage());  // cumprod output, used as the weight
    // The input, needed for the final division in apply(), is saved by the
    // wiring into the slot its version guards.
    kernel::NaryKernel<CumprodBackward, 1>::wire_autograd(std::move(bwd), {a}, out);
    return out;
}

// ─── cummax / cummin backward ───────────────────────────────────────────────
//
// Backward for cummax: dy[k] flows to the position that first achieved
// the running maximum up to position k.  We process right-to-left along
// the scan axis, accumulating dy into a running sum that is "claimed" (reset)
// whenever we hit a new-max position (i.e., saved_y_[k] > saved_y_[k-1]).
//
// The is_new_max condition at k is:
//   k == 0  OR  saved_y_[k] > saved_y_[k-1]   (cummax is non-decreasing)
//
// Same logic applies for cummin with "saved_y_[k] < saved_y_[k-1]".

namespace {

// Dispatch cummax or cummin forward via the backend.
TensorImplPtr scan_ext_dispatch(const TensorImplPtr& a, int axis, bool is_max, const char* name) {
    Validator::input(a, std::string(name) + ".a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    auto sh = a->shape();
    // The public ops route a 0-d input through on_scalar_axis first.
    const int ax = wrap_dim(axis, static_cast<int>(sh.size()), name);
    OpScopeFull scope{name, device, dt, sh};
    scope.set_attr("axis", static_cast<std::int64_t>(ax));

    Storage out_storage =
        is_max ? backend::Dispatcher::for_device(device).cummax(a->storage(), sh, ax, dt)
               : backend::Dispatcher::for_device(device).cummin(a->storage(), sh, ax, dt);
    return fresh(std::move(out_storage), sh, dt, device);
}

// Generic segmented right-to-left accumulation.
// Template parameter IsMax selects the is_new_extreme condition.
template <bool IsMax, typename T>
void scan_ext_backward_loop(const T* dy, const T* y, T* dx, const Shape& shape, int axis) {
    const int ndim = static_cast<int>(shape.size());
    std::size_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    for (int d = axis + 1; d < ndim; ++d)
        inner *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    const std::size_t L = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);

    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t j = 0; j < inner; ++j) {
            T running_sum = T(0);
            // Right-to-left pass
            for (std::size_t k = L; k-- > 0;) {
                std::size_t idx = (o * L + k) * inner + j;
                running_sum += dy[idx];
                // is_new_extreme: first position (k==0) or y[k] changed from y[k-1]
                bool is_new = (k == 0);
                if (!is_new) {
                    std::size_t prev_idx = (o * L + k - 1) * inner + j;
                    if constexpr (IsMax)
                        is_new = (y[idx] > y[prev_idx]);
                    else
                        is_new = (y[idx] < y[prev_idx]);
                }
                if (is_new) {
                    dx[idx] = running_sum;
                    running_sum = T(0);
                } else {
                    dx[idx] = T(0);
                }
            }
        }
    }
}

Storage cummax_backward_cpu(
    const Storage& grad, const Storage& out, const Shape& shape, int axis, Dtype dt) {
    const auto& g_cs = std::get<CpuStorage>(grad);
    const auto& o_cs = std::get<CpuStorage>(out);
    std::size_t nb = g_cs.nbytes;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);

    if (dt == Dtype::F32)
        scan_ext_backward_loop<true>(reinterpret_cast<const float*>(g_cs.ptr.get()),
                                     reinterpret_cast<const float*>(o_cs.ptr.get()),
                                     reinterpret_cast<float*>(ptr.get()), shape, axis);
    else if (dt == Dtype::F64)
        scan_ext_backward_loop<true>(reinterpret_cast<const double*>(g_cs.ptr.get()),
                                     reinterpret_cast<const double*>(o_cs.ptr.get()),
                                     reinterpret_cast<double*>(ptr.get()), shape, axis);
    else
        ErrorBuilder("cummax_backward").not_implemented("dtype not supported");

    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage cummin_backward_cpu(
    const Storage& grad, const Storage& out, const Shape& shape, int axis, Dtype dt) {
    const auto& g_cs = std::get<CpuStorage>(grad);
    const auto& o_cs = std::get<CpuStorage>(out);
    std::size_t nb = g_cs.nbytes;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);

    if (dt == Dtype::F32)
        scan_ext_backward_loop<false>(reinterpret_cast<const float*>(g_cs.ptr.get()),
                                      reinterpret_cast<const float*>(o_cs.ptr.get()),
                                      reinterpret_cast<float*>(ptr.get()), shape, axis);
    else if (dt == Dtype::F64)
        scan_ext_backward_loop<false>(reinterpret_cast<const double*>(g_cs.ptr.get()),
                                      reinterpret_cast<const double*>(o_cs.ptr.get()),
                                      reinterpret_cast<double*>(ptr.get()), shape, axis);
    else
        ErrorBuilder("cummin_backward").not_implemented("dtype not supported");

    return Storage{CpuStorage{ptr, nb, dt}};
}

// Where each running extreme came from: ``src[k]`` is the last position
// j <= k at which the extreme changed, the element y[k] is a copy of and so
// the one its gradient belongs to.  The same strict comparison as the
// eager loop above, so a tie keeps crediting the earlier position.
template <bool IsMax, typename T>
void scan_ext_sources(const T* y, std::int32_t* src, const Shape& shape, int axis) {
    const int ndim = static_cast<int>(shape.size());
    std::size_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    for (int d = axis + 1; d < ndim; ++d)
        inner *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    const std::size_t L = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
    for (std::size_t o = 0; o < outer; ++o)
        for (std::size_t j = 0; j < inner; ++j) {
            std::int32_t last = 0;
            for (std::size_t k = 0; k < L; ++k) {
                const std::size_t idx = (o * L + k) * inner + j;
                if (k > 0) {
                    const std::size_t prev = (o * L + k - 1) * inner + j;
                    if (IsMax ? y[idx] > y[prev] : y[idx] < y[prev])
                        last = static_cast<std::int32_t>(k);
                }
                src[idx] = last;
            }
        }
}

// Graph-mode backward of cummax / cummin: a scatter-add of the gradient onto
// the positions the extremes came from.  The positions are data (read from
// the saved output, on the host, as the eager backward reads it); the
// scatter-add is an op, so the result is differentiable in the gradient —
// and, rightly, not in the input, on which it depends piecewise-constantly.
template <bool IsMax>
TensorImplPtr scan_ext_graph_backward(const TensorImplPtr& grad_out,
                                      const Storage& saved_y,
                                      const Shape& shape,
                                      int axis,
                                      Dtype dt,
                                      Device device,
                                      const char* name) {
    CpuStorage sources = helpers::allocate_cpu(shape, Dtype::I32);
    auto* src = reinterpret_cast<std::int32_t*>(sources.ptr.get());
    const auto fill = [&](const auto* y) { scan_ext_sources<IsMax>(y, src, shape, axis); };
    if (device == Device::GPU) {
        auto y = ::mlx::core::contiguous(*std::get<GpuStorage>(saved_y).arr);
        y.eval();
        MemoryTracker::track_host_sync(y.nbytes());
        if (dt == Dtype::F32)
            fill(y.data<float>());
        else if (dt == Dtype::F64)
            fill(y.data<double>());
        else
            ErrorBuilder(name).not_implemented("dtype");
    } else {
        const auto* y = std::get<CpuStorage>(saved_y).ptr.get();
        if (dt == Dtype::F32)
            fill(reinterpret_cast<const float*>(y));
        else if (dt == Dtype::F64)
            fill(reinterpret_cast<const double*>(y));
        else
            ErrorBuilder(name).not_implemented("dtype");
    }
    auto index = std::make_shared<TensorImpl>(
        backend::Dispatcher::for_device(device).from_cpu(std::move(sources), shape), shape,
        Dtype::I32, device, false);
    return scatter_add_op(zeros_op(shape, dt, device), index, grad_out, axis);
}

// Backward node for cummax.
class CummaxBackward : public AutogradNode<CummaxBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;

    std::vector<Storage> apply(Storage grad_out) override {
        // For GPU devices: evaluate MLX arrays, compute loop on CPU, rewrap.
        if (device_ == Device::GPU) {
            const auto& g_gs = std::get<GpuStorage>(grad_out);
            const auto& y_gs = std::get<GpuStorage>(saved_output_);
            g_gs.arr->eval();
            y_gs.arr->eval();
            auto g_cont = ::mlx::core::contiguous(*g_gs.arr);
            auto y_cont = ::mlx::core::contiguous(*y_gs.arr);
            g_cont.eval();
            y_cont.eval();

            std::size_t n = shape_numel(input_shape_);
            auto shape_mlx = gpu::to_mlx_shape(input_shape_);
            auto mlx_dt = gpu::to_mlx_dtype(dtype_);

            if (dtype_ == Dtype::F32) {
                std::vector<float> dx(n, 0.0f);
                MemoryTracker::track_host_sync(g_cont.nbytes());
                MemoryTracker::track_host_sync(y_cont.nbytes());
                scan_ext_backward_loop<true>(g_cont.data<float>(), y_cont.data<float>(), dx.data(),
                                             input_shape_, axis_);
                ::mlx::core::array dx_arr(dx.data(), shape_mlx, mlx_dt);
                return {Storage{gpu::wrap_mlx_array(std::move(dx_arr), dtype_)}};
            } else if (dtype_ == Dtype::F64) {
                std::vector<double> dx(n, 0.0);
                MemoryTracker::track_host_sync(g_cont.nbytes());
                MemoryTracker::track_host_sync(y_cont.nbytes());
                scan_ext_backward_loop<true>(g_cont.data<double>(), y_cont.data<double>(),
                                             dx.data(), input_shape_, axis_);
                ::mlx::core::array dx_arr(dx.data(), shape_mlx, mlx_dt);
                return {Storage{gpu::wrap_mlx_array(std::move(dx_arr), dtype_)}};
            } else {
                ErrorBuilder("cummax_backward").not_implemented("dtype");
                return {};
            }
        }
        // CPU path
        return {cummax_backward_cpu(grad_out, saved_output_, input_shape_, axis_, dtype_)};
    }

    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        return {scan_ext_graph_backward<true>(grad_out, saved_output_, input_shape_, axis_, dtype_,
                                              device_, "cummax_backward")};
    }
};

// Backward node for cummin.
class CumminBackward : public AutogradNode<CumminBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;

    std::vector<Storage> apply(Storage grad_out) override {
        if (device_ == Device::GPU) {
            const auto& g_gs = std::get<GpuStorage>(grad_out);
            const auto& y_gs = std::get<GpuStorage>(saved_output_);
            g_gs.arr->eval();
            y_gs.arr->eval();
            auto g_cont = ::mlx::core::contiguous(*g_gs.arr);
            auto y_cont = ::mlx::core::contiguous(*y_gs.arr);
            g_cont.eval();
            y_cont.eval();

            std::size_t n = shape_numel(input_shape_);
            auto shape_mlx = gpu::to_mlx_shape(input_shape_);
            auto mlx_dt = gpu::to_mlx_dtype(dtype_);

            if (dtype_ == Dtype::F32) {
                std::vector<float> dx(n, 0.0f);
                MemoryTracker::track_host_sync(g_cont.nbytes());
                MemoryTracker::track_host_sync(y_cont.nbytes());
                scan_ext_backward_loop<false>(g_cont.data<float>(), y_cont.data<float>(), dx.data(),
                                              input_shape_, axis_);
                ::mlx::core::array dx_arr(dx.data(), shape_mlx, mlx_dt);
                return {Storage{gpu::wrap_mlx_array(std::move(dx_arr), dtype_)}};
            } else if (dtype_ == Dtype::F64) {
                std::vector<double> dx(n, 0.0);
                MemoryTracker::track_host_sync(g_cont.nbytes());
                MemoryTracker::track_host_sync(y_cont.nbytes());
                scan_ext_backward_loop<false>(g_cont.data<double>(), y_cont.data<double>(),
                                              dx.data(), input_shape_, axis_);
                ::mlx::core::array dx_arr(dx.data(), shape_mlx, mlx_dt);
                return {Storage{gpu::wrap_mlx_array(std::move(dx_arr), dtype_)}};
            } else {
                ErrorBuilder("cummin_backward").not_implemented("dtype");
                return {};
            }
        }
        return {cummin_backward_cpu(grad_out, saved_output_, input_shape_, axis_, dtype_)};
    }

    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        return {scan_ext_graph_backward<false>(grad_out, saved_output_, input_shape_, axis_, dtype_,
                                               device_, "cummin_backward")};
    }
};

const OpSchema CummaxBackward::schema_v1{"cummax", 1, AmpPolicy::KeepInput, true};
const OpSchema CumminBackward::schema_v1{"cummin", 1, AmpPolicy::KeepInput, true};

}  // anonymous namespace

TensorImplPtr cummax_op(const TensorImplPtr& a, int axis) {
    if (a->shape().empty())
        return on_scalar_axis(a, axis, "cummax", cummax_op);
    auto out = scan_ext_dispatch(a, axis, true, "cummax");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CummaxBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    bwd->keep_output(out->storage());
    kernel::NaryKernel<CummaxBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

TensorImplPtr cummin_op(const TensorImplPtr& a, int axis) {
    if (a->shape().empty())
        return on_scalar_axis(a, axis, "cummin", cummin_op);
    auto out = scan_ext_dispatch(a, axis, false, "cummin");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CumminBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    bwd->keep_output(out->storage());
    kernel::NaryKernel<CumminBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

}  // namespace lucid
