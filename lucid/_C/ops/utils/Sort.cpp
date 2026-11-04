// lucid/_C/ops/utils/Sort.cpp
//
// Implements sort, argsort, argmax, argmin, nonzero, unique, and topk.
//
// sort_op and the topk values output are differentiable for float (half
// included) and complex inputs.  Their shared autograd node is
// IndexScatterBackward, which uses Dispatcher::scatter_add_axis to place each
// incoming gradient value back at the position in the input that produced the
// corresponding sorted output element.  This is the adjoint of the gather
// operation performed during sort.
//
// argsort, argmax, argmin return integer index tensors and have no gradient.
//
// nonzero and unique always execute on CPU because their output sizes are
// determined by data values and cannot be known statically; GPU tensors are
// transferred to CPU before computation.

#include "Sort.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include "../../autograd/FuncOp.h"
#include "../../backend/Dispatcher.h"
#include "../../compile/Tracer.h"
#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Half.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"
#include "../gfunc/Gfunc.h"
#include "ScalarAxis.h"
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::fresh;
using utils_detail::wrap_axis;

// Sorting moves values without changing them, so a gradient flows back
// through the permutation for every dtype that carries one: the real floats
// (half included) and the complex types.  Integer and bool results carry
// none.  This used to be the list ``F32 || F64``, which dropped the grad_fn
// of every float16 / bfloat16 / complex sort and topk without a word.
bool differentiable_dtype(Dtype dt) {
    return is_floating_point(dt) || is_complex(dt);
}

// Scatter-add gradient values from `grad` back to their original positions
// in a zero-initialised buffer of `input_shape`.  Element `grad[i]` at
// position i along `axis` is accumulated into position `indices[i]` in the
// output buffer.  Multiple source positions can map to the same destination
// (scatter-add semantics); this is handled by the backend.
Storage scatter_axis_add_storage(const Storage& grad,
                                 const Storage& indices,
                                 const Shape& input_shape,
                                 const Shape& grad_shape,
                                 int axis,
                                 Dtype dt,
                                 Device device) {
    return backend::Dispatcher::for_device(device).scatter_add_axis(grad, indices, input_shape,
                                                                    grad_shape, axis, dt);
}

// Backward node for sort_op and the values output of topk_op.
//
// Invariants:
//   indices_ — the sort permutation produced by the forward pass.  Element
//              indices_[i] along `axis_` holds the original position of the
//              element that ended up at position i in the sorted output.
//   grad_shape_ — shape of the sorted output (and of grad_out).
//   axis_       — the axis along which sorting was done.
//
// Backward formula: scatter-add grad_out to input positions using indices_,
// which exactly inverts the gather performed during the forward sort.
class IndexScatterBackward : public FuncOp<IndexScatterBackward, 1> {
public:
    static constexpr bool kKeepsOwnValues = true;
    static const OpSchema schema_v1;

    Storage indices_;
    Shape grad_shape_;
    int axis_ = 0;

    void release_own_saved() { clear_saved(indices_); }

    std::vector<Storage> apply(Storage grad_out) override {
        return {scatter_axis_add_storage(grad_out, indices_, input_shapes_[0], grad_shape_, axis_,
                                         dtype_, device_)};
    }

    // The same scatter-add, recorded: each selected element's gradient goes
    // back to the position it was taken from.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        auto index = std::make_shared<TensorImpl>(indices_, grad_shape_, storage_dtype(indices_),
                                                  device_, false);
        auto base = zeros_op(input_shapes_[0], dtype_, device_);
        return {scatter_add_op(base, index, grad_out, axis_)};
    }
};

const OpSchema IndexScatterBackward::schema_v1{
    "index_scatter", 1, AmpPolicy::KeepInput, true, "", -1, 1, {}, true};

// Wire IndexScatterBackward onto the sorted or topk output tensor.  Skips
// wiring entirely for non-differentiable dtypes (integer types) to avoid
// allocating a backward node that will never be called.  The `indices`
// storage is moved into the backward node so it is retained for the backward
// pass without an extra copy.
TensorImplPtr
attach_index_scatter_grad(const TensorImplPtr& a, TensorImplPtr out, Storage indices, int axis) {
    if (!differentiable_dtype(a->dtype())) {
        // ``wire_autograd`` is also what records a traced op's operands, so
        // skipping it has to record them by hand.  Without this an integer
        // ``sort`` reaches ``lucid.compile`` with no inputs and the whole
        // graph falls back to eager.  ``topk`` is worse: its follow-up
        // ``on_op_io`` for the indices becomes the *first* registration,
        // so the values tensor is never traced, turns into an external
        // feed, and every compiled replay returns the values of the call
        // that was traced.
        if (auto* trc = ::lucid::compile::current_tracer())
            trc->on_op_io({a}, out);
        return out;
    }
    auto bwd = std::make_shared<IndexScatterBackward>();
    bwd->grad_shape_ = out->shape();
    bwd->indices_ = std::move(indices);
    bwd->axis_ = axis;
    kernel::NaryKernel<IndexScatterBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

}  // namespace

// Sort `a` in ascending order along `axis`.  The backend's sort_select
// (topk=false) returns the sorted values and the permutation indices as a
// pair.  The indices are saved in IndexScatterBackward so the backward can
// scatter the incoming gradient back to the original positions.
TensorImplPtr sort_op(const TensorImplPtr& a, int axis) {
    Validator::input(a, "sort.a").non_null();
    if (a->shape().empty())
        return on_scalar_axis(a, axis, "sort", sort_op);
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"sort", device, dt, a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()), "sort");
    scope.set_attr("axis", static_cast<std::int64_t>(ax));
    Shape out_shape = a->shape();
    // sort_select(topk=false) returns all elements in sorted order.
    auto [values, indices] = backend::Dispatcher::for_device(device).sort_select(
        a->storage(), a->shape(), out_shape, ax, dt, false);
    auto out = fresh(std::move(values), std::move(out_shape), dt, device);
    return attach_index_scatter_grad(a, std::move(out), std::move(indices), ax);
}

// Compute the permutation indices that would sort `a` along `axis` in
// ascending order.  The output is int64, as every index-returning op's is,
// and carries no gradient.
TensorImplPtr argsort_op(const TensorImplPtr& a, int axis) {
    Validator::input(a, "argsort.a").non_null();
    if (a->shape().empty())
        return on_scalar_axis(a, axis, "argsort", argsort_op);
    const Device device = a->device();
    OpScopeFull scope{"argsort", device, a->dtype(), a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()), "argsort");
    scope.set_attr("axis", static_cast<std::int64_t>(ax));
    Storage out =
        backend::Dispatcher::for_device(device).argsort(a->storage(), a->shape(), ax, a->dtype());
    auto result = fresh(std::move(out), a->shape(), Dtype::I64, device);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, result);
    }
    return result;
}

namespace {
// Shared forward implementation for argmax (is_min=false) and argmin
// (is_min=true).  Returns an I64 index tensor whose shape is `a.shape` with
// dimension `axis` removed (or set to 1 when keepdims=true).
TensorImplPtr
argext_dispatch(const TensorImplPtr& a, int axis, bool keepdims, bool is_min, const char* name) {
    Validator::input(a, std::string(name) + ".a").non_null();
    if (a->shape().empty())
        return on_scalar_axis(a, axis, name, [&](const TensorImplPtr& v, int ax) {
            return argext_dispatch(v, ax, keepdims, is_min, name);
        });
    const Device device = a->device();
    OpScopeFull scope{name, device, a->dtype(), a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()), name);
    // An empty axis has no extreme to point at.  The CPU read index 0 of an
    // empty row and answered with it; Metal raised MLX's message.
    if (a->shape()[static_cast<std::size_t>(ax)] == 0)
        ErrorBuilder(name).index_error("Expected reduction dim " + std::to_string(ax) +
                                       " to have non-zero size.");
    scope.set_attr("axis", static_cast<std::int64_t>(ax));
    scope.set_attr("keepdim", keepdims);
    Shape out_shape = a->shape();
    if (keepdims)
        out_shape[ax] = 1;
    else
        out_shape.erase(out_shape.begin() + ax);
    Storage out = backend::Dispatcher::for_device(device).arg_reduce_index(
        a->storage(), a->shape(), ax, keepdims, a->dtype(), is_min);
    auto result = fresh(std::move(out), std::move(out_shape), Dtype::I64, device);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, result);
    }
    return result;
}
}  // namespace

// Return the index of the maximum element along `axis`; delegates to argext_dispatch.
TensorImplPtr argmax_op(const TensorImplPtr& a, int axis, bool keepdims) {
    return argext_dispatch(a, axis, keepdims, false, "argmax");
}
// Return the index of the minimum element along `axis`; delegates to argext_dispatch.
TensorImplPtr argmin_op(const TensorImplPtr& a, int axis, bool keepdims) {
    return argext_dispatch(a, axis, keepdims, true, "argmin");
}

// Find all non-zero element positions in `a`.  The data is always transferred
// to CPU before counting so that the dynamic output size (number of non-zero
// elements) can be determined before allocating the output buffer.
// Returns a 2-D tensor of shape (count, ndim) with dtype I64, where each row
// is the multi-dimensional index of one non-zero element in row-major order.
//
// The host round-trip is forced by the data-dependent output size, but the
// RESULT rides the input's device: returning CPU storage for a GPU input made
// every downstream op device-mismatch, so callers had to insert a manual
// ``.to()`` — which is the transfer the round-trip was already paying for.
TensorImplPtr nonzero_op(const TensorImplPtr& a) {
    Validator::input(a, "nonzero.a").non_null();
    const std::size_t ndim = a->shape().size();
    OpScopeFull scope{"nonzero", a->device(), a->dtype(), a->shape()};

    // Materialise on CPU so we can count non-zeros before allocating output.
    CpuStorage cpu = backend::Dispatcher::for_device(a->device()).to_cpu(a->storage(), a->shape());

    std::size_t count = 0;
    CpuStorage out = backend::Dispatcher::for_device(Device::CPU)
                         .nonzero_forward(Storage{cpu}, a->shape(), a->dtype(), count);

    Shape out_shape{static_cast<std::int64_t>(count), static_cast<std::int64_t>(ndim)};
    TensorImplPtr result;
    if (a->device() != Device::CPU) {
        Storage up =
            backend::Dispatcher::for_device(a->device()).from_cpu(std::move(out), out_shape);
        result = fresh(std::move(up), std::move(out_shape), Dtype::I64, a->device());
    } else {
        result = fresh(Storage{std::move(out)}, std::move(out_shape), Dtype::I64, Device::CPU);
    }
    // Record the trace I/O explicitly.  The output length is data-dependent
    // so this can never be emitted — the point is to keep the node in the
    // trace at all.  An unrecorded node is dropped as dead and its result
    // reaches the consumer as a fresh external feed, bound once at trace
    // time: the compiled model then returns the trace-time indices for
    // every later input, reporting success the whole way.
    if (auto* trc = ::lucid::compile::current_tracer())
        trc->on_op_io({a}, result);
    return result;
}

// Sort and deduplicate all elements of `a`, returning them as a 1-D tensor of
// the same dtype.  The computation runs on the host (std::sort / std::unique
// need host memory, and the output length is data-dependent), but the RESULT
// rides the input's device — see the note on ``nonzero_op``.
//
namespace {
template <typename T>
CpuStorage pack_unique_vec(Dtype dt, const std::vector<T>& vals) {
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = vals.size() * sizeof(T);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    if (out.nbytes > 0)
        std::memcpy(out.ptr.get(), vals.data(), out.nbytes);
    return out;
}

template <typename T>
CpuStorage run_unique_int(Dtype dt, const T* src, std::size_t n) {
    std::vector<T> vals(src, src + n);
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    return pack_unique_vec(dt, vals);
}

template <typename T>
CpuStorage run_unique_float(Dtype dt, const T* src, std::size_t n) {
    std::vector<T> vals;
    vals.reserve(n);
    std::vector<T> nans;
    bool seen_zero = false;
    T first_zero = T{0};
    for (std::size_t i = 0; i < n; ++i) {
        const T v = src[i];
        if (std::isnan(v)) {
            nans.push_back(v);
        } else {
            if (v == T{0} && !seen_zero) {
                seen_zero = true;
                first_zero = v;
            }
            vals.push_back(v);
        }
    }
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    if (seen_zero) {
        auto it = std::lower_bound(vals.begin(), vals.end(), T{0});
        if (it != vals.end() && *it == T{0})
            *it = first_zero;
    }
    vals.insert(vals.end(), nans.begin(), nans.end());
    return pack_unique_vec(dt, vals);
}

template <typename ToF32>
CpuStorage run_unique_half(Dtype dt, const void* raw, std::size_t n, ToF32 to_f32) {
    const auto* src = reinterpret_cast<const std::uint16_t*>(raw);
    std::vector<std::uint16_t> vals(src, src + n);
    std::stable_sort(vals.begin(), vals.end(), [&](std::uint16_t x, std::uint16_t y) {
        const float fx = to_f32(x), fy = to_f32(y);
        if (std::isnan(fx))
            return false;
        if (std::isnan(fy))
            return true;
        return fx < fy;
    });
    vals.erase(
        std::unique(vals.begin(), vals.end(),
                    [&](std::uint16_t x, std::uint16_t y) { return to_f32(x) == to_f32(y); }),
        vals.end());
    return pack_unique_vec(dt, vals);
}
}  // namespace

TensorImplPtr unique_op(const TensorImplPtr& a) {
    Validator::input(a, "unique.a").non_null();
    const Dtype dt = a->dtype();
    OpScopeFull scope{"unique", a->device(), dt, a->shape()};

    CpuStorage cpu = backend::Dispatcher::for_device(a->device()).to_cpu(a->storage(), a->shape());
    const std::size_t n = shape_numel(a->shape());
    const void* raw = cpu.ptr.get();

    CpuStorage out_cpu;
    if (dt == Dtype::F16)
        out_cpu = run_unique_half(dt, raw, n, backend::detail::half_bits_to_float);
    else if (dt == Dtype::BF16)
        out_cpu = run_unique_half(dt, raw, n, backend::detail::bfloat16_bits_to_float);
    else if (dt == Dtype::F32)
        out_cpu = run_unique_float(dt, reinterpret_cast<const float*>(raw), n);
    else if (dt == Dtype::F64)
        out_cpu = run_unique_float(dt, reinterpret_cast<const double*>(raw), n);
    else if (dt == Dtype::I8)
        out_cpu = run_unique_int(dt, reinterpret_cast<const std::int8_t*>(raw), n);
    else if (dt == Dtype::I16)
        out_cpu = run_unique_int(dt, reinterpret_cast<const std::int16_t*>(raw), n);
    else if (dt == Dtype::I32)
        out_cpu = run_unique_int(dt, reinterpret_cast<const std::int32_t*>(raw), n);
    else if (dt == Dtype::I64)
        out_cpu = run_unique_int(dt, reinterpret_cast<const std::int64_t*>(raw), n);
    else if (dt == Dtype::Bool)
        out_cpu = run_unique_int(dt, reinterpret_cast<const std::uint8_t*>(raw), n);
    else
        ErrorBuilder("unique").not_implemented("dtype not supported");
    Shape out_shape{static_cast<std::int64_t>(out_cpu.nbytes / dtype_size(dt))};
    TensorImplPtr result;
    if (a->device() != Device::CPU) {
        Storage up =
            backend::Dispatcher::for_device(a->device()).from_cpu(std::move(out_cpu), out_shape);
        result = fresh(std::move(up), std::move(out_shape), dt, a->device());
    } else {
        result = fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, Device::CPU);
    }
    // See ``nonzero_op``: recorded so the node survives the trace and the
    // stub emitter can decline, rather than the result being frozen into
    // the graph as a feed.
    if (auto* trc = ::lucid::compile::current_tracer())
        trc->on_op_io({a}, result);
    return result;
}

// Return the k largest values and their original positions along `axis`.
// Calls sort_select(topk=true), which returns exactly the k greatest elements
// and their sort-indices.  The values output receives IndexScatterBackward for
// differentiable dtypes; the indices output (int64) carries no gradient.
//
// The indices_out is constructed first so its storage can be shared by
// attach_index_scatter_grad without an extra allocation.
std::vector<TensorImplPtr> topk_op(const TensorImplPtr& a, std::int64_t k, int axis) {
    Validator::input(a, "topk.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    // The reference refuses bool here, as for argmax: which k booleans are
    // "largest" is not a question it answers.  Sorting a bool tensor is
    // allowed — ``sort`` / ``argsort`` accept it on both devices.
    if (dt == Dtype::Bool)
        ErrorBuilder("topk").not_implemented(
            "topk does not accept bool — cast to an integer dtype");
    // A 0-d input answers k = 0 with the scalar itself, as k = 1: the
    // reference's choice, since a 0-d result cannot hold zero elements.
    if (a->shape().empty()) {
        if (k < 0 || k > 1)
            ErrorBuilder("topk").invalid_argument(
                "selected index k out of range: k must be in [0, 1] for a 0-d input, got " +
                std::to_string(k));
        return on_scalar_axis(a, axis, "topk",
                              [](const TensorImplPtr& v, int ax) { return topk_op(v, 1, ax); });
    }
    OpScopeFull scope{"topk", device, dt, a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()), "topk");
    // k = 0 is an empty selection, not an error: the values still carry a
    // grad_fn, whose backward is a zero gradient.
    const std::int64_t size = a->shape()[static_cast<std::size_t>(ax)];
    if (k < 0 || k > size)
        ErrorBuilder("topk").invalid_argument("selected index k out of range: k must be in [0, " +
                                              std::to_string(size) + "], got " + std::to_string(k));
    // Carry axis + k into the trace for the compile-path TopKEmitter.
    scope.set_attr("axis", static_cast<std::int64_t>(ax));
    scope.set_attr("k", k);
    Shape out_shape = a->shape();
    out_shape[static_cast<std::size_t>(ax)] = k;
    auto [values, indices] = backend::Dispatcher::for_device(device).sort_select(
        a->storage(), a->shape(), out_shape, ax, dt, true);
    // Build indices_out before values_out so that its storage pointer remains
    // valid when it is borrowed by attach_index_scatter_grad below.
    auto indices_out = fresh(std::move(indices), out_shape, Dtype::I64, device);
    auto values_out = fresh(std::move(values), out_shape, dt, device);
    values_out = attach_index_scatter_grad(a, std::move(values_out), indices_out->storage(), ax);
    // Trace-side: register the indices_out as a second output of this
    // op.  ``attach_index_scatter_grad`` already wired ``values_out``
    // through ``wire_autograd``; ``indices_out`` is non-differentiable
    // (I64) so we hand-roll the trace hook so the compile-path
    // TopKEmitter sees a 2-output op.  The Tracer's first-vs-
    // subsequent-call detection (added 2026-05-23, see
    // [[engine-compile-multi-output-2026-05]]) handles the append.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, indices_out);
    }
    return {std::move(values_out), std::move(indices_out)};
}

LUCID_REGISTER_OP(IndexScatterBackward)

}  // namespace lucid
