// lucid/_C/ops/gfunc/Gfunc.cpp
//
// Implementations of tensor-creation ("generator") ops.
//
// Shared implementation pattern:
//   1. Validate arguments (shape non-negative, step non-zero, etc.).
//   2. Construct an OpScopeFull for profiling and debug instrumentation.
//   3. Allocate or fill storage via the backend or Allocator:
//        - Constant fills (zeros, ones, full, empty): use make_zero_storage /
//          make_ones_storage / IBackend::full.
//        - Structured fills (eye): delegate entirely to IBackend::eye.
//        - Sequential fills (arange, linspace): fill a CPU buffer inline and
//          upload with IBackend::from_cpu so both CPU and GPU output work
//          without separate backend implementations.
//   4. Wrap the resulting Storage and metadata into a TensorImpl via finalize().
//
// The finalize() helper creates a TensorImpl from Storage + Shape + Dtype +
// Device + requires_grad.  It is used by every op in this file; factoring it
// out avoids repeating the make_shared<TensorImpl>(...) boilerplate.
//
// arange and linspace are implemented directly in C++ because expressing
// sequential fills through the backend interface would require either:
//   (a) a special-purpose IBackend::arange() that every backend must implement,
//   or (b) an MLX arange call that would require bridging to MLX array on the
//       CPU path even when the device is CPU.
// The CPU-fill-then-from_cpu approach is simpler, portable, and fast enough
// for the typical sizes involved.
//
// Note on empty_op: despite the name "empty", the current implementation
// zero-fills the buffer to prevent undefined reads.  This is a deliberate
// conservative choice; callers must not rely on the values being zero.

#include "Gfunc.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/PhantomMode.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/BinaryKernel.h"
#include "../../ops/bfunc/Add.h"
#include "../../ops/bfunc/Compare.h"
#include "../../ops/bfunc/Div.h"
#include "../../ops/bfunc/Mul.h"
#include "../../ops/composite/Indexing.h"
#include "../../ops/composite/Logical.h"
#include "../../ops/ufunc/Transpose.h"
#include "../../ops/utils/Promote.h"
#include "../../ops/utils/ScalarAxis.h"
#include "../../ops/utils/Select.h"
#include "../../ops/utils/View.h"

namespace lucid {

namespace {

using helpers::allocate_cpu;

// Wrap storage and metadata into a TensorImpl.
//
// All generator ops in this file reduce to a single finalize() call at the
// end, which keeps the pattern uniform and makes it easy to spot-check that
// requires_grad, dtype, and device are forwarded correctly.
inline TensorImplPtr
finalize(Storage&& storage, Shape shape, Dtype dt, Device device, bool requires_grad) {
    return std::make_shared<TensorImpl>(std::move(storage), std::move(shape), dt, device,
                                        requires_grad);
}

// A factory's result, declared to an active compile trace as made from
// nothing (``compile::traced_factory``) — a constant to it.
inline TensorImplPtr made_from_nothing(TensorImplPtr out) {
    compile::traced_factory(out);
    return out;
}

}  // namespace

// Create a zero-filled tensor on the requested device.
//
// Delegates to make_zero_storage, which uses calloc-style zeroing on the CPU
// and MLX zeros on the GPU, both of which are more efficient than filling
// after allocation.
TensorImplPtr zeros_op(const Shape& shape, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"zeros", device, dt, shape};
    check_concrete_shape(shape);
    auto s = make_zero_storage(shape, dt, device);
    auto out = finalize(std::move(s), shape, dt, device, requires_grad);
    // Same rationale as full_op below — without this, downstream
    // consumers see a fresh impl pointer and mint a duplicate trace
    // id, breaking the MPSGraph DAG.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_factory(out);
        trc->on_op_io({}, out);
    }
    return out;
}

// Create a ones-filled tensor on the requested device.
//
// Delegates to make_ones_storage, which on the CPU allocates then memsets,
// and on the GPU uses MLX ones or from_cpu with a pre-filled buffer.
TensorImplPtr ones_op(const Shape& shape, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"ones", device, dt, shape};
    check_concrete_shape(shape);
    auto s = make_ones_storage(shape, dt, device);
    auto out = finalize(std::move(s), shape, dt, device, requires_grad);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_factory(out);
        trc->on_op_io({}, out);
    }
    return out;
}

// Create a constant-filled tensor by delegating to IBackend::full().
//
// IBackend::full() handles dtype conversion (the fill_value is a double;
// the backend casts to the target dtype) and device placement in one call,
// avoiding a CPU alloc + upload round-trip for the GPU path.
TensorImplPtr
full_op(const Shape& shape, double fill_value, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"full", device, dt, shape};
    check_concrete_shape(shape);
    // 3.5 Phase 1.3: report fill value for the compile-path Full emitter.
    scope.set_attr("fill_value", fill_value);
    auto s = backend::Dispatcher::for_device(device).full(shape, dt, fill_value);
    auto out = finalize(std::move(s), shape, dt, device, requires_grad);
    // Factory ops don't go through NaryKernel::wire_autograd, so the
    // trace's id↔TensorImpl map needs an explicit registration here —
    // otherwise the next op that consumes ``out`` sees an unrecognised
    // pointer and mints a *second* id for the same buffer, splitting
    // the MPSGraph DAG (which then disconnects every other param's
    // backward path through it).
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_factory(out);
        trc->on_op_io({}, out);
    }
    return out;
}

// Create a zero-filled tensor; semantically "uninitialised" but safe.
//
// Zero-fills rather than leaving the memory uninitialised to prevent
// undefined-behaviour reads if a caller forgets to fill before use.
// The "empty" name signals contract: callers must not depend on the values.
TensorImplPtr empty_op(const Shape& shape, Dtype dt, Device device, bool requires_grad) {
    OpScopeFull scope{"empty", device, dt, shape};
    check_concrete_shape(shape);
    auto s = make_zero_storage(shape, dt, device);
    return made_from_nothing(finalize(std::move(s), shape, dt, device, requires_grad));
}

// Create an N×M matrix with ones on diagonal k.
//
// k=0 is the main diagonal; k>0 places ones above; k<0 places ones below.
// Both N and M must be non-negative.  IBackend::eye handles the actual fill.
TensorImplPtr eye_op(
    std::int64_t N, std::int64_t M, std::int64_t k, Dtype dt, Device device, bool requires_grad) {
    Shape shape{N, M};
    OpScopeFull scope{"eye", device, dt, shape};
    check_concrete_shape(shape);
    auto s = backend::Dispatcher::for_device(device).eye(N, M, k, dt);
    return made_from_nothing(finalize(std::move(s), shape, dt, device, requires_grad));
}

namespace {

// The width a sequence is filled at.  The fill loop has four: a narrower
// float or integer — and, for ``linspace``, complex — is that sequence
// rounded once to the requested dtype, which is how the reference computes
// them too.  Bool has neither op there, and complex has no ``arange``.
Dtype sequence_fill_dtype(Dtype dt, const char* op, bool complex_ok) {
    switch (dt) {
    case Dtype::F32:
    case Dtype::F64:
    case Dtype::I32:
    case Dtype::I64:
        return dt;
    case Dtype::F16:
    case Dtype::BF16:
        return Dtype::F64;
    case Dtype::I8:
    case Dtype::I16:
        return Dtype::I64;
    case Dtype::C64:
    case Dtype::C128:
        if (complex_ok)
            return Dtype::F64;
        break;
    default:
        break;
    }
    ErrorBuilder(op).not_implemented("not implemented for " + std::string(dtype_name(dt)));
}

// Round a sequence filled at ``fill`` to ``dt`` on the CPU, where it was
// filled, and hand it to ``device``.
Storage
sequence_to_device(CpuStorage cpu, const Shape& shape, Dtype fill, Dtype dt, Device device) {
    if (fill != dt) {
        Storage narrow = backend::Dispatcher::for_device(Device::CPU)
                             .astype(Storage{std::move(cpu)}, shape, fill, dt);
        cpu = std::get<CpuStorage>(std::move(narrow));
    }
    return backend::Dispatcher::for_device(device).from_cpu(std::move(cpu), shape);
}

// ``ceil((stop - start) / step)`` elements, 0 when ``step`` points away from
// ``stop``.  The count is checked as a double: converting one past ``int64``
// is undefined, and an infinite ``stop`` came out as INT64_MAX elements.
std::int64_t arange_length(double start, double stop, double step) {
    if (step == 0.0)
        ErrorBuilder("arange").fail("step must be non-zero");
    const auto text = [](double v) {
        std::ostringstream os;
        os << v;
        return os.str();
    };
    if (!std::isfinite(start) || !std::isfinite(stop))
        ErrorBuilder("arange").invalid_argument("unsupported range: " + text(start) + " -> " +
                                                text(stop));
    const double diff = stop - start;
    const double count = (diff * step <= 0) ? 0.0 : std::ceil(diff / step);
    if (!(count <= static_cast<double>(kMaxNumel)))
        ErrorBuilder("arange").invalid_argument(
            "Storage size calculation overflowed: the range holds " + text(count) +
            " elements, more than the " + std::to_string(kMaxNumel) + " a tensor holds");
    return static_cast<std::int64_t>(count);
}

// A sequence fills a host buffer before it dispatches, which phantom mode
// does not reach: under it, the sequence's shape and no buffer at all.
TensorImplPtr phantom_sequence(const Shape& shape, Dtype dt, Device device, bool requires_grad) {
    return made_from_nothing(
        finalize(phantom_storage(shape, dt, device), shape, dt, device, requires_grad));
}

}  // namespace

// Create a 1-D tensor with arithmetic progression values.
//
// Computes the element count as ceil((stop - start) / step) and fills a CPU
// buffer with typed values, then routes through from_cpu so the result lands
// on the right device.  The element count formula matches numpy.arange exactly.
//
// When diff and step have opposite signs (e.g. start=5, stop=0, step=1), the
// range is empty and n=0, producing an empty 1-D tensor — not an error.
//
// The compute_cpu lambda is templated on pointer type via CTAD, avoiding
// a separate switch-inside-switch to handle the float/int cases.  Each case
// of the outer switch simply casts the raw buffer pointer to the correct type.
TensorImplPtr
arange_op(double start, double stop, double step, Dtype dt, Device device, bool requires_grad) {
    const std::int64_t n = arange_length(start, stop, step);
    Shape shape{n};
    OpScopeFull scope{"arange", device, dt, shape};
    // Record the generator parameters so the compile emitter can bake the
    // sequence as a constant (start / step are static; n is the output len).
    scope.set_attr("start", start);
    scope.set_attr("step", step);

    // Fill p[i] = start + i * step, cast to the target element type.
    auto compute_cpu = [&](auto* p) {
        using T = std::remove_pointer_t<decltype(p)>;
        for (std::int64_t i = 0; i < n; ++i) {
            p[i] = static_cast<T>(start + static_cast<double>(i) * step);
        }
    };

    const Dtype fill = sequence_fill_dtype(dt, "arange", /*complex_ok=*/false);
    if (PhantomMode::is_enabled())
        return phantom_sequence(shape, dt, device, requires_grad);
    auto cpu = allocate_cpu(shape, fill);
    switch (fill) {
    case Dtype::F32:
        compute_cpu(reinterpret_cast<float*>(cpu.ptr.get()));
        break;
    case Dtype::F64:
        compute_cpu(reinterpret_cast<double*>(cpu.ptr.get()));
        break;
    case Dtype::I32:
        compute_cpu(reinterpret_cast<std::int32_t*>(cpu.ptr.get()));
        break;
    default:
        compute_cpu(reinterpret_cast<std::int64_t*>(cpu.ptr.get()));
        break;
    }
    return made_from_nothing(finalize(sequence_to_device(std::move(cpu), shape, fill, dt, device),
                                      shape, dt, device, requires_grad));
}

// Create a 1-D tensor with num evenly spaced values between start and stop
// (inclusive on both ends).
//
// The last element is explicitly pinned to stop to prevent floating-point
// drift: computing start + (num-1) * step can accumulate error that causes
// the last element to differ slightly from stop.  Pinning matches the
// numpy.linspace and linspace behaviour.
//
// Special cases:
//   num == 0: returns an empty tensor (shape {0}).
//   num == 1: returns a tensor containing exactly [start]; stop is ignored
//             for the value computation and step is defined as 0.
TensorImplPtr linspace_op(
    double start, double stop, std::int64_t num, Dtype dt, Device device, bool requires_grad) {
    Shape shape{num};
    OpScopeFull scope{"linspace", device, dt, shape};
    check_concrete_shape(shape);
    // The compile emitter bakes the same sequence as a constant; the length
    // is the output dim.
    scope.set_attr("start", start);
    scope.set_attr("stop", stop);
    // step is only meaningful when num > 1; defined as 0 otherwise to avoid
    // division by zero when num == 1.
    const double step = (num > 1) ? (stop - start) / static_cast<double>(num - 1) : 0.0;

    auto compute_cpu = [&](auto* p) {
        using T = std::remove_pointer_t<decltype(p)>;
        for (std::int64_t i = 0; i < num; ++i) {
            // For num == 1, always emit start regardless of step.
            const double v = (num == 1) ? start : start + static_cast<double>(i) * step;
            p[i] = static_cast<T>(v);
        }
        // Pin the last element to stop to eliminate floating-point drift.
        if (num >= 2)
            p[num - 1] = static_cast<T>(stop);
    };

    const Dtype fill = sequence_fill_dtype(dt, "linspace", /*complex_ok=*/true);
    if (PhantomMode::is_enabled())
        return phantom_sequence(shape, dt, device, requires_grad);
    auto cpu = allocate_cpu(shape, fill);
    switch (fill) {
    case Dtype::F32:
        compute_cpu(reinterpret_cast<float*>(cpu.ptr.get()));
        break;
    case Dtype::F64:
        compute_cpu(reinterpret_cast<double*>(cpu.ptr.get()));
        break;
    case Dtype::I32:
        compute_cpu(reinterpret_cast<std::int32_t*>(cpu.ptr.get()));
        break;
    default:
        compute_cpu(reinterpret_cast<std::int64_t*>(cpu.ptr.get()));
        break;
    }
    return made_from_nothing(finalize(sequence_to_device(std::move(cpu), shape, fill, dt, device),
                                      shape, dt, device, requires_grad));
}

// Create a zero tensor with the same shape, dtype, and device as a.
//
// Delegates to zeros_op with the metadata extracted from a.
TensorImplPtr zeros_like_op(const TensorImplPtr& a, bool requires_grad) {
    Validator::input(a, "zeros_like.a").non_null();
    return zeros_op(a->shape(), a->dtype(), a->device(), requires_grad);
}

// Create a ones tensor with the same shape, dtype, and device as a.
TensorImplPtr ones_like_op(const TensorImplPtr& a, bool requires_grad) {
    Validator::input(a, "ones_like.a").non_null();
    return ones_op(a->shape(), a->dtype(), a->device(), requires_grad);
}

// Create an uninitialised (zero-filled in practice) tensor with the same
// shape, dtype, and device as a.
TensorImplPtr empty_like_op(const TensorImplPtr& a, bool requires_grad) {
    Validator::input(a, "empty_like.a").non_null();
    return empty_op(a->shape(), a->dtype(), a->device(), requires_grad);
}

// Create a constant-filled tensor with the same shape, dtype, and device as a.
TensorImplPtr full_like_op(const TensorImplPtr& a, double fill_value, bool requires_grad) {
    Validator::input(a, "full_like.a").non_null();
    return full_op(a->shape(), fill_value, a->dtype(), a->device(), requires_grad);
}

// ── logspace ──────────────────────────────────────────────────────────────────

TensorImplPtr logspace_op(double start,
                          double stop,
                          std::int64_t num,
                          double base,
                          Dtype dt,
                          Device device,
                          bool requires_grad) {
    Shape shape{num};
    OpScopeFull scope{"logspace", device, dt, shape};
    check_concrete_shape(shape);

    const double step = (num > 1) ? (stop - start) / static_cast<double>(num - 1) : 0.0;

    auto compute_cpu = [&](auto* p) {
        using T = std::remove_pointer_t<decltype(p)>;
        for (std::int64_t i = 0; i < num; ++i) {
            const double exp_v = (num == 1) ? start : start + static_cast<double>(i) * step;
            p[i] = static_cast<T>(std::pow(base, exp_v));
        }
        if (num >= 2)
            p[num - 1] = static_cast<T>(std::pow(base, stop));
    };

    const Dtype fill = sequence_fill_dtype(dt, "logspace", /*complex_ok=*/true);
    if (PhantomMode::is_enabled())
        return phantom_sequence(shape, dt, device, requires_grad);
    auto cpu = allocate_cpu(shape, fill);
    switch (fill) {
    case Dtype::F32:
        compute_cpu(reinterpret_cast<float*>(cpu.ptr.get()));
        break;
    case Dtype::F64:
        compute_cpu(reinterpret_cast<double*>(cpu.ptr.get()));
        break;
    case Dtype::I32:
        compute_cpu(reinterpret_cast<std::int32_t*>(cpu.ptr.get()));
        break;
    default:
        compute_cpu(reinterpret_cast<std::int64_t*>(cpu.ptr.get()));
        break;
    }
    return made_from_nothing(finalize(sequence_to_device(std::move(cpu), shape, fill, dt, device),
                                      shape, dt, device, requires_grad));
}

// ── axis-scatter operands ─────────────────────────────────────────────────────
// scatter_add, scatter_amax, scatter_amin and scatter_prod share one shape
// rule, the reference's.  It lived in scatter_add alone, so the other three
// handed the backends any shapes: an index wider than the base off ``dim``
// wrote past it, an index into an empty axis was refused on the CPU and
// silently empty on Metal, and a src wider than the index was misplaced on
// the CPU and a reshape error on Metal.

namespace {

// The kernels read src's buffer as base's dtype and the index's as integers,
// so an operand of any other dtype was reinterpreted byte for byte: a float64
// src added into float32 landed as 1.875 where 1.0 was meant, and a bool
// index scattered to the wrong slots.  The reference refuses a src whose
// dtype is not self's for every scatter and for ``index_copy``, so these ops
// refuse rather than cast; a caller that wants the cast makes it.
void require_scatter_dtypes(const char* op,
                            const TensorImplPtr& base,
                            const TensorImplPtr& indices,
                            const TensorImplPtr& src) {
    const Dtype idt = indices->dtype();
    if (idt != Dtype::I32 && idt != Dtype::I64)
        ErrorBuilder(op).index_error("index must be an int32 or int64 tensor, got " +
                                     std::string(dtype_name(idt)));
    if (src->dtype() != base->dtype())
        throw DtypeMismatch(std::string(dtype_name(base->dtype())),
                            std::string(dtype_name(src->dtype())),
                            std::string(op) + ": src must have self's dtype");
}

std::string format_scatter_shape(const Shape& sh) {
    std::string out = "[";
    for (std::size_t i = 0; i < sh.size(); ++i)
        out += (i ? ", " : "") + std::to_string(sh[i]);
    return out + "]";
}

AxisScatterOperands empty_axis_scatter_operands(int d,
                                                const Shape& bs,
                                                const TensorImplPtr& indices,
                                                const TensorImplPtr& src) {
    Shape empty_is = bs;
    empty_is[static_cast<std::size_t>(d)] = 0;
    TensorImplPtr idx = (indices->shape() == empty_is) ? indices : reshape_op(indices, empty_is);
    auto flat_src = reshape_op(src, Shape{static_cast<std::int64_t>(src->numel())});
    TensorImplPtr values =
        (src->shape() == empty_is) ? src : reshape_op(narrow_op(flat_src, 0, 0, 0), empty_is);
    return {d, std::move(idx), std::move(values)};
}

void check_axis_scatter_shapes(
    const char* op, int d, const Shape& bs, const Shape& is, const Shape& ss) {
    const int ndim = static_cast<int>(bs.size());
    if (is.size() != bs.size() || ss.size() != bs.size())
        ErrorBuilder(op).fail("index " + format_scatter_shape(is) + ", self " +
                              format_scatter_shape(bs) + " and src " + format_scatter_shape(ss) +
                              " must have the same number of dimensions");
    // The index walks src and lands in base, so it may be no larger than src
    // on any axis, nor than base on any but ``dim``.  Unchecked, a larger
    // index read src and wrote base past their ends.
    for (int i = 0; i < ndim; ++i) {
        const auto k = static_cast<std::size_t>(i);
        if (is[k] > ss[k] || (i != d && is[k] > bs[k]))
            ErrorBuilder(op).fail("Expected index " + format_scatter_shape(is) +
                                  " to be no larger than self " + format_scatter_shape(bs) +
                                  " apart from dimension " + std::to_string(d) +
                                  " and to be no larger than src " + format_scatter_shape(ss));
    }
    // Any index into an empty axis is out of range; saying so needs only the
    // shapes, so Metal, which reads no index back, refuses it too (as
    // ``scatter`` and ``gather`` do).
    if (bs[static_cast<std::size_t>(d)] == 0 && shape_numel(is) > 0)
        ErrorBuilder(op).index_error("index out of range: dimension " + std::to_string(d) +
                                     " has size 0");
}

}  // namespace

AxisScatterOperands axis_scatter_operands(const char* op,
                                          const TensorImplPtr& base,
                                          const TensorImplPtr& indices,
                                          const TensorImplPtr& src,
                                          int dim) {
    Validator::input(base, std::string(op) + ".base").non_null();
    Validator::input(indices, std::string(op) + ".indices").non_null();
    Validator::input(src, std::string(op) + ".src").non_null();
    Validator::pair(base, indices, op).same_device();
    Validator::pair(base, src, op).same_device();
    require_scatter_dtypes(op, base, indices, src);

    const Shape& bs = base->shape();
    const Shape& is = indices->shape();
    const Shape& ss = src->shape();
    const int ndim = static_cast<int>(bs.size());
    // A 0-d base never reaches here: on_scalar_scatter takes it first.
    const int d = wrap_dim(dim, ndim, op);
    // An empty index writes nothing and skips the rank and extent checks, as
    // the reference does.  Normalise the empty index and source slice to base's
    // rank so the backend kernels and backward nodes see an empty slice along d.
    if (indices->numel() == 0)
        return empty_axis_scatter_operands(d, bs, indices, src);
    check_axis_scatter_shapes(op, d, bs, is, ss);
    // Only the corner of src the index covers is scattered; every backend
    // reads src in the index's own layout.
    TensorImplPtr values = src;
    for (int i = 0; i < ndim; ++i) {
        const auto k = static_cast<std::size_t>(i);
        if (ss[k] > is[k])
            values = narrow_op(values, i, 0, is[k]);
    }
    return {d, indices, std::move(values)};
}

TensorImplPtr dispatch_axis_scatter(const char* scope_name,
                                    ScatterKernel kernel,
                                    const TensorImplPtr& base,
                                    const AxisScatterOperands& ops) {
    const Dtype dt = base->dtype();
    const Device dv = base->device();
    const Shape& bs = base->shape();
    const Shape& is = ops.indices->shape();
    OpScopeFull scope{scope_name, dv, dt, bs};
    scope.set_attr("dim", static_cast<std::int64_t>(ops.dim));

    auto& be = backend::Dispatcher::for_device(dv);
    Storage out_s = (be.*kernel)(base->storage(), ops.indices->storage(), ops.values->storage(), bs,
                                 is, ops.dim, dt);
    auto out = std::make_shared<TensorImpl>(std::move(out_s), bs, dt, dv, false);
    if (auto* trc = ::lucid::compile::current_tracer())
        trc->on_op_io({base, ops.indices, ops.values}, out);
    return out;
}

// ── scatter_add ───────────────────────────────────────────────────────────────
// Autograd: d/d(base)=grad_out; d/d(src)=gather(grad_out, dim, index).

TensorImplPtr scatter_add_op(const TensorImplPtr& base_in,
                             const TensorImplPtr& indices,
                             const TensorImplPtr& src_in,
                             int dim) {
    Validator::input(base_in, "scatter_add.base").non_null();
    Validator::input(src_in, "scatter_add.src").non_null();
    const AmpScope amp{"scatter_add", base_in->device()};
    const auto cast_ops = amp.cast_widest({base_in, src_in});
    const TensorImplPtr& base = cast_ops[0];
    const TensorImplPtr& src = cast_ops[1];
    if (auto out = on_scalar_scatter("scatter_add", base, indices, src, dim, scatter_add_op))
        return out;
    auto ops = axis_scatter_operands("scatter_add", base, indices, src, dim);
    auto out = dispatch_axis_scatter("scatter_add", &backend::IBackend::scatter_add, base, ops);
    const auto& [d, idx, values] = ops;
    const Dtype dt = base->dtype();
    const Device dv = base->device();
    const Shape& bs = base->shape();

    // Autograd wiring
    const bool needs_grad =
        GradMode::is_enabled() && (base->requires_grad() || values->requires_grad());
    if (!needs_grad)
        return out;

    // base_edge: gradient = grad_out  (identity)
    // src_edge:  gradient = gather(grad_out, indices, dim)
    auto base_edge = lucid::detail::ensure_grad_fn(base);
    auto src_edge = lucid::detail::ensure_grad_fn(values);

    struct ScatterAddNode : Node {
        int dim_;
        std::shared_ptr<TensorImpl> saved_indices_;
        Shape base_shape_;
        Dtype dtype_;
        Device device_;
        std::string node_name() const override { return "scatter_add"; }
        void release_saved() override { saved_indices_.reset(); }
        void
        for_each_owned_tensor(const std::function<void(const TensorImplPtr&)>& fn) const override {
            if (saved_indices_)
                fn(saved_indices_);
        }
        // The index is the one tensor read by value; base and src only route
        // the gradient.
        void validate_versions() override {
            check_saved_version(saved_indices_, 0, "scatter_add", 1);
        }

        std::vector<Storage> apply(Storage g) override {
            Storage grad_base = g;  // d/d(base) = identity
            auto g_impl = std::make_shared<TensorImpl>(g, base_shape_, dtype_, device_, false);
            // gather_op(a, indices, axis) — select along axis at index positions
            auto grad_src_impl = gather_op(g_impl, saved_indices_, dim_);
            Storage grad_src = grad_src_impl->storage();
            return {std::move(grad_base), std::move(grad_src)};
        }

        std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& g) override {
            // Both halves are already op-level: the base passes the gradient
            // through untouched, and the source takes back exactly the
            // elements it scattered.  The saved indices are integers with no
            // derivative of their own, so reusing them is safe here in a way
            // it is not for a saved statistic.
            return {g, gather_op(g, saved_indices_, dim_)};
        }
    };

    auto bwd = std::make_shared<ScatterAddNode>();
    bwd->dim_ = d;
    bwd->saved_indices_ = idx;
    bwd->base_shape_ = bs;
    bwd->dtype_ = dt;
    bwd->device_ = dv;
    bwd->set_next_edges(
        {Edge(base_edge, base->grad_output_nr()), Edge(src_edge, values->grad_output_nr())});
    bwd->set_saved_versions({idx->version()});

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

// ── scatter_set ───────────────────────────────────────────────────────────────
// ``index_copy``'s front door to the overwrite scatter: the reference's
// stricter shape rule for it, then ``scatter_op`` (composite/Indexing.cpp),
// which owns the kernel dispatch and the backward — d/d(base) = grad_out
// with the overwritten positions zeroed, d/d(src) = gather(grad_out, index).

TensorImplPtr scatter_set_op(const TensorImplPtr& base,
                             const TensorImplPtr& indices,
                             const TensorImplPtr& src,
                             int dim) {
    Validator::input(base, "scatter_set.base").non_null();
    Validator::input(indices, "scatter_set.indices").non_null();
    Validator::input(src, "scatter_set.src").non_null();
    // ``scatter_op`` casts src, as an assignment must; ``index_copy``, whose
    // door this is, refuses a src of another dtype in the reference.
    require_scatter_dtypes("scatter_set", base, indices, src);

    const Shape& bs = base->shape();
    const Shape& is = indices->shape();
    const Shape& ss = src->shape();
    const int ndim = static_cast<int>(bs.size());
    const int d = wrap_dim(dim, ndim, "scatter_set");

    // ``index_copy`` writes whole slices: index and src have one shape, equal
    // to base's on every axis but ``dim``.  ``scatter`` alone would take a
    // narrower source and write the corner it covers, where the reference
    // refuses — and before the kernels walked the index's own extents, a
    // source of the wrong width was read and written out of bounds.
    bool fits = is == ss && is.size() == bs.size() && d >= 0 && d < ndim;
    for (int i = 0; fits && i < ndim; ++i)
        fits = i == d || is[static_cast<std::size_t>(i)] == bs[static_cast<std::size_t>(i)];
    if (!fits)
        throw ShapeMismatch(bs, ss,
                            "scatter_set: source must match self on every axis but dim " +
                                std::to_string(d) + ", and the index must match the source");
    return scatter_op(base, d, indices, src);
}

// ── scatter_amax / scatter_amin / scatter_prod ────────────────────────────────
//
// Shared helper: dispatch the backend call, wire autograd using the mask-based
// rule described in Gfunc.h.

namespace {

// Build "winner" mask: 1.0 where `lhs == rhs`, 0.0 elsewhere (as float).
// Uses where_op to avoid bool * float dtype mismatch in the multiply path.
TensorImplPtr winner_mask(const TensorImplPtr& lhs, const TensorImplPtr& rhs) {
    auto eq = equal_op(lhs, rhs);  // bool
    return where_op(eq, ones_like_op(lhs), zeros_like_op(lhs));
}

// scatter_prod's gradients with respect to base and src, given grad g and the
// forward's out.  Each operand's is g times the product of the other operands
// at its position.  That is out / operand only where the operand is not zero;
// at a zero it was 0 / 0, NaN, where the reference gives the product of the
// others.  So the zeros are taken out of the division (the reference's rule):
//   base: the product with base's zeros read as 1, divided by base with its
//         zeros read as 1 — at a zero base, the others' product.
//   src:  where src[j] is the only zero src at its position, the product with
//         that zero read as 1; elsewhere out / src with src's zeros read as 1,
//         which is 0 at a zero that has company, as it must be.
std::pair<TensorImplPtr, TensorImplPtr> scatter_prod_grads(const TensorImplPtr& g,
                                                           const TensorImplPtr& base,
                                                           const TensorImplPtr& idx,
                                                           const TensorImplPtr& src,
                                                           int dim) {
    auto nil_b = zeros_like_op(base);
    auto one_b = ones_like_op(base);
    auto nil_s = zeros_like_op(src);
    auto one_s = ones_like_op(src);
    auto base_zero = equal_op(base, nil_b);
    auto src_zero = equal_op(src, nil_s);
    auto base_1 = where_op(base_zero, one_b, base);
    auto src_1 = where_op(src_zero, one_s, src);

    auto nz_prod = scatter_prod_op(base_1, idx, src_1, dim);
    auto src_zmarks = where_op(src_zero, one_s, nil_s);
    auto src_zcount = scatter_add_op(nil_b, idx, src_zmarks, dim);
    auto g_nz = mul_op(g, nz_prod);
    auto grad_base = where_op(equal_op(src_zcount, nil_b), div_op(g_nz, base_1), nil_b);

    auto base_zmarks = where_op(base_zero, one_b, nil_b);
    auto total_zcount = add_op(base_zmarks, src_zcount);
    auto no_other_zeros = equal_op(gather_op(total_zcount, idx, dim), src_zmarks);
    auto grad_src = where_op(no_other_zeros, div_op(gather_op(g_nz, idx, dim), src_1), nil_s);
    return {grad_base, grad_src};
}

bool same_off_axis_shape(const Shape& bs, const Shape& is, int dim) {
    if (bs.size() != is.size())
        return false;
    for (std::size_t a = 0; a < bs.size(); ++a) {
        if (static_cast<int>(a) != dim && bs[a] != is[a])
            return false;
    }
    return true;
}

template <typename IdxT, typename Fn>
void walk_scatter_pairs(const IdxT* idx,
                        std::size_t outer,
                        std::size_t base_d,
                        std::size_t idx_d,
                        std::size_t inner,
                        Fn&& fn) {
    const auto d_signed = static_cast<std::int64_t>(base_d);
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t r = 0; r < idx_d; ++r) {
            for (std::size_t in = 0; in < inner; ++in) {
                const std::size_t j = (o * idx_d + r) * inner + in;
                std::int64_t k = static_cast<std::int64_t>(idx[j]);
                if (k < 0)
                    k += d_signed;
                if (k >= 0 && k < d_signed)
                    fn(j, (o * base_d + static_cast<std::size_t>(k)) * inner + in);
            }
        }
    }
}

template <typename T, typename IdxT>
std::vector<Storage> cpu_scatter_prod_grads(const Storage& g_s,
                                            const TensorImplPtr& base,
                                            const TensorImplPtr& idx,
                                            const TensorImplPtr& src,
                                            int dim) {
    const Shape& bs = base->shape();
    const Shape& is = idx->shape();
    const Dtype dt = base->dtype();
    std::size_t outer = 1, inner = 1;
    for (int a = 0; a < dim; ++a)
        outer *= static_cast<std::size_t>(bs[static_cast<std::size_t>(a)]);
    for (std::size_t a = static_cast<std::size_t>(dim) + 1; a < bs.size(); ++a)
        inner *= static_cast<std::size_t>(bs[a]);
    const auto base_d = static_cast<std::size_t>(bs[static_cast<std::size_t>(dim)]);
    const auto idx_d = static_cast<std::size_t>(is[static_cast<std::size_t>(dim)]);
    const std::size_t base_n = outer * base_d * inner;

    CpuStorage gb_s = allocate_cpu(bs, dt);
    CpuStorage gs_s = allocate_cpu(is, dt);
    const auto* g = reinterpret_cast<const T*>(std::get<CpuStorage>(g_s).ptr.get());
    const auto* b = reinterpret_cast<const T*>(std::get<CpuStorage>(base->storage()).ptr.get());
    const auto* s = reinterpret_cast<const T*>(std::get<CpuStorage>(src->storage()).ptr.get());
    auto* gb = reinterpret_cast<T*>(gb_s.ptr.get());
    auto* gs = reinterpret_cast<T*>(gs_s.ptr.get());
    const auto* idx_p =
        reinterpret_cast<const IdxT*>(std::get<CpuStorage>(idx->storage()).ptr.get());

    std::vector<T> nz_prod(base_n);
    std::vector<int> zero_cnt(base_n, 0);
    for (std::size_t p = 0; p < base_n; ++p) {
        const bool is_z = b[p] == T(0);
        nz_prod[p] = is_z ? T(1) : b[p];
        zero_cnt[p] = is_z ? 1 : 0;
    }
    walk_scatter_pairs(idx_p, outer, base_d, idx_d, inner, [&](std::size_t j, std::size_t p) {
        const bool is_z = s[j] == T(0);
        zero_cnt[p] += is_z ? 1 : 0;
        nz_prod[p] *= is_z ? T(1) : s[j];
    });
    for (std::size_t p = 0; p < base_n; ++p) {
        const bool is_z = b[p] == T(0);
        const int src_zeros = zero_cnt[p] - (is_z ? 1 : 0);
        gb[p] = src_zeros == 0 ? g[p] * (nz_prod[p] / (is_z ? T(1) : b[p])) : T(0);
    }
    walk_scatter_pairs(idx_p, outer, base_d, idx_d, inner, [&](std::size_t j, std::size_t p) {
        const bool is_z = s[j] == T(0);
        const int other_zeros = zero_cnt[p] - (is_z ? 1 : 0);
        gs[j] = other_zeros == 0 ? g[p] * (nz_prod[p] / (is_z ? T(1) : s[j])) : T(0);
    });
    return {Storage{std::move(gb_s)}, Storage{std::move(gs_s)}};
}

// Snapshot ``t``'s forward-time values into a detached TensorImpl so
// ``allow_mutation_on_saved_tensors`` reads the original saved values while a
// CPU in-place write into ``t`` is not refused by ``storage_is_shared`` before
// ``validate_versions`` runs.
TensorImplPtr snapshot_scatter_operand(const TensorImplPtr& t) {
    Storage s = (!t->is_phantom() && t->device() == Device::CPU)
                    ? backend::Dispatcher::for_device(Device::CPU)
                          .clone(t->storage(), t->shape(), t->dtype())
                    : t->storage();
    return std::make_shared<TensorImpl>(std::move(s), t->shape(), t->dtype(), t->device(), false);
}

// What the scatter_amax / scatter_amin / scatter_prod backwards hold: both
// operands and the index saved by value in detached TensorImpls, strong
// handles to the live inputs so each one's version is guarded (the reference
// saves all four and refuses the same writes), and a weak handle to the result.
//
// The result is held weakly.  It owns this node through its grad_fn, so a
// strong handle closed a cycle only a backward broke: every forward that ran
// none kept the node, the result and, through the node's operands and edges,
// the graph behind them alive.  Nor is a copy of its buffer kept instead: on
// the CPU an in-place write into bytes a node also holds is refused at the
// write, where the reference refuses it at backward with the version error.
// Once nothing outside the graph holds the result, the backward that needs its
// values recomputes them (ScatterExtremumNode::result), and a write into it is
// detected through ``saved_out_version_sink_`` even after the output TensorImpl
// has been rebound or dropped (LCD-590).
struct ScatterReduceNode : Node {
    int dim_ = 0;
    std::array<TensorImplPtr, 3> saved_impl_inputs_;
    TensorImplPtr saved_base_;
    TensorImplPtr saved_indices_;
    TensorImplPtr saved_src_;
    std::weak_ptr<TensorImpl> saved_out_;
    std::shared_ptr<VersionCounter> saved_out_version_sink_;
    Shape base_shape_;
    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;

    // Hold the operands and the index by value, keep strong handles to the
    // live operands and a weak handle to the result at the versions they are
    // guarded at, and wire the edges to ``base`` and ``src``.
    void hold(const TensorImplPtr& base,
              const TensorImplPtr& indices,
              const TensorImplPtr& src,
              const TensorImplPtr& out,
              int dim) {
        dim_ = dim;
        saved_impl_inputs_ = {base, indices, src};
        saved_base_ = snapshot_scatter_operand(base);
        saved_indices_ = snapshot_scatter_operand(indices);
        saved_src_ = snapshot_scatter_operand(src);
        saved_out_ = out;
        saved_out_version_sink_ = out->version_sink();
        base_shape_ = base->shape();
        dtype_ = base->dtype();
        device_ = base->device();
        set_next_edges({Edge(lucid::detail::ensure_grad_fn(base), base->grad_output_nr()),
                        Edge(lucid::detail::ensure_grad_fn(src), src->grad_output_nr())});
        set_saved_versions({base->version(), indices->version(), src->version(), out->version()});
    }

    void validate_versions() override {
        const std::string op = node_name();
        check_saved_version(saved_impl_inputs_[0], 0, op, 0);
        check_saved_version(saved_impl_inputs_[1], 1, op, 1);
        check_saved_version(saved_impl_inputs_[2], 2, op, 2);
        if (saved_versions_.size() > 3)
            check_version_match(saved_out_version_sink_, saved_versions_[3], op + " output 0");
    }

    void advance_saved_output_version(std::int64_t v) noexcept override {
        if (saved_out_version_sink_)
            saved_out_version_sink_->store(static_cast<std::uint64_t>(v),
                                           std::memory_order_relaxed);
        if (saved_versions_.size() > 3)
            saved_versions_[3] = v;
    }

    void release_saved() override {
        for (auto& p : saved_impl_inputs_)
            p.reset();
        saved_base_.reset();
        saved_indices_.reset();
        saved_src_.reset();
        saved_out_.reset();
        saved_out_version_sink_.reset();
    }

    void for_each_owned_tensor(const std::function<void(const TensorImplPtr&)>& fn) const override {
        for (const auto& t : saved_impl_inputs_) {
            if (t)
                fn(t);
        }
    }
};

// The scatter_amax / scatter_amin backward.  The gradient at each output
// position is split equally among all tied winners there, base's element
// included (the reference framework's scatter_reduce convention).
struct ScatterExtremumNode : ScatterReduceNode {
    const char* name_;
    ScatterKernel kernel_;

    ScatterExtremumNode(const char* name, ScatterKernel kernel) : name_(name), kernel_(kernel) {}

    std::string node_name() const override { return name_; }

    // The result's values as forward computed them: the tensor itself while
    // it lives — :meth:`validate_versions` has just checked it — and once
    // nothing outside the graph holds it, the forward's kernel run again on
    // the saved operands.  A max or min selects its value, so outside
    // ``allow_mutation_on_saved_tensors`` (where a write into the live result
    // itself is not refused) the two agree bit for bit.
    TensorImplPtr result() const {
        if (auto out = saved_out_.lock();
            out && saved_versions_.size() > 3 && out->version() == saved_versions_[3])
            return out;
        auto& be = backend::Dispatcher::for_device(device_);
        Storage s =
            (be.*kernel_)(saved_base_->storage(), saved_indices_->storage(), saved_src_->storage(),
                          base_shape_, saved_indices_->shape(), dim_, dtype_);
        return std::make_shared<TensorImpl>(std::move(s), base_shape_, dtype_, device_, false);
    }

    std::vector<Storage> apply(Storage g) override {
        auto g_impl = std::make_shared<TensorImpl>(g, base_shape_, dtype_, device_, false);
        const TensorImplPtr out = result();

        // 1. Winner masks (float 0/1 per element)
        auto mask_base_f = winner_mask(saved_base_, out);
        auto gathered_out = gather_op(out, saved_indices_, dim_);
        auto mask_src_f = winner_mask(saved_src_, gathered_out);

        // 2. Count total tied winners per output position:
        //    winners_from_src[i] = Σ mask_src_f[j]  (over j where idx[j]==i)
        //    Since base[i] is always a participant, total_wins[i] == 0 occurs
        //    if and only if out[i] is NaN; dividing without clamping yields
        //    0 / 0 = NaN at every participant of a NaN bucket, matching the
        //    reference on both CPU and Metal.
        auto zeros_out = zeros_like_op(out);
        auto wins_from_src = scatter_add_op(zeros_out, saved_indices_, mask_src_f, dim_);
        auto total_wins = add_op(mask_base_f, wins_from_src);

        // 3. grad_base = g * mask_base_f / total_wins
        Storage grad_base = div_op(mul_op(g_impl, mask_base_f), total_wins)->storage();

        // 4. grad_src = gather(g) * mask_src_f / gather(total_wins)
        auto gathered_g = gather_op(g_impl, saved_indices_, dim_);
        auto gathered_wins = gather_op(total_wins, saved_indices_, dim_);
        Storage grad_src = div_op(mul_op(gathered_g, mask_src_f), gathered_wins)->storage();
        return {std::move(grad_base), std::move(grad_src)};
    }
};

// Attach ``node`` as ``out``'s grad_fn.
TensorImplPtr attach_scatter_reduce(std::shared_ptr<ScatterReduceNode> node,
                                    const TensorImplPtr& base,
                                    const TensorImplPtr& indices,
                                    const TensorImplPtr& src,
                                    TensorImplPtr out,
                                    int dim) {
    node->hold(base, indices, src, out, dim);
    out->set_grad_fn(std::move(node));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

struct ScatterProdNode : ScatterReduceNode {
    std::string node_name() const override { return "scatter_prod"; }

    bool can_use_cpu_fast_path(const Storage& g) const {
        if (device_ != Device::CPU || PhantomMode::is_enabled())
            return false;
        if (!std::holds_alternative<CpuStorage>(g) || saved_base_->is_phantom() ||
            saved_indices_->is_phantom() || saved_src_->is_phantom())
            return false;
        const Dtype idx_dt = saved_indices_->dtype();
        return (idx_dt == Dtype::I64 || idx_dt == Dtype::I32) &&
               same_off_axis_shape(saved_base_->shape(), saved_indices_->shape(), dim_);
    }

    std::vector<Storage> apply(Storage g) override {
        if (can_use_cpu_fast_path(g)) {
            const bool i64 = saved_indices_->dtype() == Dtype::I64;
            if (dtype_ == Dtype::F32)
                return i64 ? cpu_scatter_prod_grads<float, std::int64_t>(
                                 g, saved_base_, saved_indices_, saved_src_, dim_)
                           : cpu_scatter_prod_grads<float, std::int32_t>(
                                 g, saved_base_, saved_indices_, saved_src_, dim_);
            if (dtype_ == Dtype::F64)
                return i64 ? cpu_scatter_prod_grads<double, std::int64_t>(
                                 g, saved_base_, saved_indices_, saved_src_, dim_)
                           : cpu_scatter_prod_grads<double, std::int32_t>(
                                 g, saved_base_, saved_indices_, saved_src_, dim_);
        }
        NoGradGuard no_grad;
        auto g_impl = std::make_shared<TensorImpl>(g, base_shape_, dtype_, device_, false);
        auto [grad_base, grad_src] =
            scatter_prod_grads(g_impl, saved_base_, saved_indices_, saved_src_, dim_);
        return {grad_base->storage(), grad_src->storage()};
    }
};

TensorImplPtr run_scatter_reduce(const char* name,
                                 ScatterKernel kernel,
                                 const TensorImplPtr& base,
                                 const TensorImplPtr& indices,
                                 const TensorImplPtr& src_in,
                                 int dim,
                                 bool is_prod) {
    if (auto out = on_scalar_scatter(
            name, base, indices, src_in, dim,
            [&](const TensorImplPtr& b, const TensorImplPtr& i, const TensorImplPtr& s, int ax) {
                return run_scatter_reduce(name, kernel, b, i, s, ax, is_prod);
            }))
        return out;
    auto ops = axis_scatter_operands(name, base, indices, src_in, dim);
    auto out = dispatch_axis_scatter(name, kernel, base, ops);
    const auto& [d, idx, values] = ops;

    if (!GradMode::is_enabled() || !(base->requires_grad() || values->requires_grad()))
        return out;

    std::shared_ptr<ScatterReduceNode> node =
        is_prod ? std::static_pointer_cast<ScatterReduceNode>(std::make_shared<ScatterProdNode>())
                : std::static_pointer_cast<ScatterReduceNode>(
                      std::make_shared<ScatterExtremumNode>(name, kernel));
    return attach_scatter_reduce(std::move(node), base, idx, values, std::move(out), d);
}

}  // anonymous namespace

// ── scatter_amax ──────────────────────────────────────────────────────────────
TensorImplPtr scatter_amax_op(const TensorImplPtr& base,
                              const TensorImplPtr& indices,
                              const TensorImplPtr& src_in,
                              int dim) {
    return run_scatter_reduce("scatter_amax", &backend::IBackend::scatter_amax, base, indices,
                              src_in, dim, false);
}

// ── scatter_amin ──────────────────────────────────────────────────────────────
TensorImplPtr scatter_amin_op(const TensorImplPtr& base,
                              const TensorImplPtr& indices,
                              const TensorImplPtr& src_in,
                              int dim) {
    return run_scatter_reduce("scatter_amin", &backend::IBackend::scatter_amin, base, indices,
                              src_in, dim, false);
}

// ── scatter_prod ──────────────────────────────────────────────────────────────
TensorImplPtr scatter_prod_op(const TensorImplPtr& base,
                              const TensorImplPtr& indices,
                              const TensorImplPtr& src_in,
                              int dim) {
    return run_scatter_reduce("scatter_prod", &backend::IBackend::scatter_prod, base, indices,
                              src_in, dim, true);
}

// ── unfold_dim ─────────────────────────────────────────────────────────────────
// Autograd: d/d(a) scatter-adds each window slot back to the position it read.
// Overlapping windows (step < size) accumulate, which is why this is a
// scatter_add rather than a reshape.

namespace {

// Scatter index for the unfold_dim backward.  ``flat_shape`` is the incoming
// gradient with the (L, size) window axes already folded into one axis of
// length L*size at ``dim``; the value stored at position k along that axis is
// the input offset that slot read from, i.e. (k / size) * step + (k % size).
//
// Materialised in full (rather than broadcast from a 1-D pattern) because the
// CPU scatter_add reads the index buffer as densely contiguous over the whole
// index shape.  Dtype is I32 to match that reader.
TensorImplPtr
make_unfold_scatter_index(const Shape& flat_shape, int dim, int size, int step, Device device) {
    const int ndim = static_cast<int>(flat_shape.size());
    std::size_t outer = 1;
    for (int i = 0; i < dim; ++i)
        outer *= static_cast<std::size_t>(flat_shape[static_cast<std::size_t>(i)]);
    std::size_t inner = 1;
    for (int i = dim + 1; i < ndim; ++i)
        inner *= static_cast<std::size_t>(flat_shape[static_cast<std::size_t>(i)]);
    const auto mid = static_cast<std::size_t>(flat_shape[static_cast<std::size_t>(dim)]);
    const auto usize = static_cast<std::size_t>(size);
    const auto ustep = static_cast<std::size_t>(step);

    auto cpu = allocate_cpu(flat_shape, Dtype::I32);
    auto* p = reinterpret_cast<std::int32_t*>(cpu.ptr.get());
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t k = 0; k < mid; ++k) {
            const auto v = static_cast<std::int32_t>((k / usize) * ustep + (k % usize));
            for (std::size_t j = 0; j < inner; ++j)
                p[(o * mid + k) * inner + j] = v;
        }
    }
    return finalize(backend::Dispatcher::for_device(device).from_cpu(std::move(cpu), flat_shape),
                    flat_shape, Dtype::I32, device, false);
}

}  // namespace

TensorImplPtr unfold_dim_op(const TensorImplPtr& a, int dim, int size, int step) {
    Validator::input(a, "unfold_dim.a").non_null();
    if (size <= 0 || step <= 0)
        ErrorBuilder("unfold_dim").invalid_argument("size and step must be positive");

    const Shape& in_shape = a->shape();
    const int ndim = static_cast<int>(in_shape.size());
    // Bounds first.  ``in_shape[d]`` was read before ``d`` had been
    // checked, so an out-of-range dimension returned whatever sat past the
    // shape — and then *that* decided the call: garbage below ``size``
    // tripped the guard below and looked like a clean refusal, garbage
    // above it sailed through and became the output shape.  Same call,
    // two different answers on two runs.
    const int d = wrap_dim(dim, ndim, "unfold_dim");
    // A 0-d tensor is one element along its notional axis; the reference
    // gives its windows back as shape ``(size,)``, with no window axis.
    if (ndim == 0)
        return reshape_op(unfold_dim_op(reshape_op(a, {1}), 0, size, step), {size});
    const int dim_size = static_cast<int>(in_shape[static_cast<std::size_t>(d)]);
    if (size > dim_size)
        ErrorBuilder("unfold_dim").fail("size > dimension size");

    const int L = (dim_size - size) / step + 1;
    Shape out_shape;
    for (int i = 0; i < d; ++i)
        out_shape.push_back(in_shape[static_cast<std::size_t>(i)]);
    out_shape.push_back(static_cast<std::int64_t>(L));
    for (int i = d + 1; i < ndim; ++i)
        out_shape.push_back(in_shape[static_cast<std::size_t>(i)]);
    out_shape.push_back(static_cast<std::int64_t>(size));

    OpScopeFull scope{"unfold_dim", a->device(), a->dtype(), out_shape};
    scope.set_attr("dim", static_cast<std::int64_t>(d));
    scope.set_attr("size", static_cast<std::int64_t>(size));
    scope.set_attr("step", static_cast<std::int64_t>(step));

    TensorImplPtr result;
    if (a->views_by_stride()) {
        // On the CPU the windows are a view of the input: the window-count
        // axis steps ``step`` elements along ``d`` and the window axis one,
        // so a write through a window reaches the input — refused when the
        // windows overlap (step < size), since an element then repeats.
        const Stride& in_stride = a->stride();
        Stride stride;
        stride.reserve(out_shape.size());
        for (int i = 0; i < ndim; ++i) {
            const auto s = in_stride[static_cast<std::size_t>(i)];
            stride.push_back(i == d ? s * step : s);
        }
        stride.push_back(in_stride[static_cast<std::size_t>(d)]);
        result = TensorImpl::make_view(a, out_shape, std::move(stride));
    } else {
        auto& be = backend::Dispatcher::for_device(a->device());
        Storage out_s = be.unfold_dim(a->storage(), in_shape, d, size, step, a->dtype());
        result = std::make_shared<TensorImpl>(std::move(out_s), out_shape, a->dtype(), a->device(),
                                              false);
    }
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a}, result);
    }

    // Autograd wiring
    const bool needs_grad = GradMode::is_enabled() && a->requires_grad();
    if (!needs_grad)
        return result;

    auto a_edge = lucid::detail::ensure_grad_fn(a);

    struct UnfoldDimNode : Node {
        int dim_ = 0;
        int size_ = 0;
        int step_ = 0;
        Shape in_shape_;
        Shape out_shape_;
        Dtype dtype_ = Dtype::F32;
        Device device_ = Device::CPU;
        std::string node_name() const override { return "unfold_dim"; }

        std::vector<Storage> apply(Storage g) override {
            auto g_impl = std::make_shared<TensorImpl>(g, out_shape_, dtype_, device_, false);
            return {adjoint(g_impl)->storage()};
        }

        // Every step below is an op, so in graph mode the same adjoint
        // records itself and can be differentiated again.
        std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
            return {adjoint(grad_out)};
        }

        TensorImplPtr adjoint(const TensorImplPtr& g_impl) const {
            const int ndim = static_cast<int>(in_shape_.size());

            // Incoming grad axes are [pre..., L (at dim), post..., size (last)].
            // Move the window axis next to L: [pre..., L, size, post...].
            std::vector<int> perm;
            perm.reserve(static_cast<std::size_t>(ndim) + 1);
            for (int i = 0; i <= dim_; ++i)
                perm.push_back(i);
            perm.push_back(ndim);
            for (int i = dim_ + 1; i < ndim; ++i)
                perm.push_back(i);
            auto moved = permute_op(g_impl, perm);

            // Fold the now-adjacent (L, size) pair into one axis.
            Shape flat_shape;
            for (int i = 0; i < dim_; ++i)
                flat_shape.push_back(in_shape_[static_cast<std::size_t>(i)]);
            flat_shape.push_back(out_shape_[static_cast<std::size_t>(dim_)] *
                                 out_shape_[static_cast<std::size_t>(ndim)]);
            for (int i = dim_ + 1; i < ndim; ++i)
                flat_shape.push_back(in_shape_[static_cast<std::size_t>(i)]);
            auto src = reshape_op(moved, flat_shape);

            auto idx = make_unfold_scatter_index(flat_shape, dim_, size_, step_, device_);
            auto base = zeros_op(in_shape_, dtype_, device_);
            return scatter_add_op(base, idx, src, dim_);
        }
    };

    auto bwd = std::make_shared<UnfoldDimNode>();
    bwd->dim_ = d;
    bwd->size_ = size;
    bwd->step_ = step;
    bwd->in_shape_ = in_shape;
    bwd->out_shape_ = out_shape;
    bwd->dtype_ = a->dtype();
    bwd->device_ = a->device();
    // Nothing is held for backward — the adjoint reads shapes alone — so no
    // version is recorded, and a write into ``a`` after forward is allowed.
    bwd->set_next_edges({Edge(a_edge, a->grad_output_nr())});

    result->set_grad_fn(std::move(bwd));
    result->set_leaf(false);
    result->set_requires_grad(true);
    return result;
}

}  // namespace lucid
