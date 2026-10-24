// lucid/_C/autograd/Helpers.cpp
//
// Implements the utility functions declared in Helpers.h.  The file is split
// into three areas:
//   1. CPU in-place accumulation (typed loops + overloaded visitor).
//   2. Thin dispatcher wrappers for element-wise and reduction operations.
//   3. CPU-side random number generators (uniform, normal, Bernoulli, randint)
//      that produce CpuStorage buffers then transfer to the target device.

#include "Helpers.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "../backend/Dispatcher.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/Generator.h"
#include "../core/Half.h"
#include "../core/PhantomMode.h"
#include "../core/TensorImpl.h"
#include "AccumulateGrad.h"
#include "Node.h"

namespace lucid {

namespace {

// C++17 overloaded-lambda helper for std::visit.
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// Typed in-place addition loop: dst[i] += src[i] for numel elements of type T.
template <typename T>
void add_typed(std::byte* dst, const std::byte* src, std::size_t numel) {
    auto* td = reinterpret_cast<T*>(dst);
    const auto* ts = reinterpret_cast<const T*>(src);
    for (std::size_t i = 0; i < numel; ++i)
        td[i] = td[i] + ts[i];
}

// dst[i] += src[i] for a 16-bit float format, accumulated in float.
//
// Not add_typed<std::uint16_t> — that would add the *bit patterns*.  Widening
// each pair, adding, and rounding once is also what IEEE says a single half
// addition means, so the result matches the GPU rather than merely running.
// float16 and bfloat16 differ only in how the sixteen bits are read.
template <float (*Widen)(std::uint16_t), std::uint16_t (*Narrow)(float)>
void add_half_inplace(std::byte* dst, const std::byte* src, std::size_t numel) {
    auto* td = reinterpret_cast<std::uint16_t*>(dst);
    const auto* ts = reinterpret_cast<const std::uint16_t*>(src);
    for (std::size_t i = 0; i < numel; ++i)
        td[i] = Narrow(Widen(td[i]) + Widen(ts[i]));
}

// dst[i] += src[i] for n lanes of a native arithmetic dtype.
void add_lanes(Dtype lane, std::byte* dst, const std::byte* src, std::size_t n) {
    switch (lane) {
    case Dtype::F32:
        add_typed<float>(dst, src, n);
        break;
    case Dtype::F64:
        add_typed<double>(dst, src, n);
        break;
    case Dtype::I64:
        add_typed<std::int64_t>(dst, src, n);
        break;
    case Dtype::I32:
        add_typed<std::int32_t>(dst, src, n);
        break;
    case Dtype::I16:
        add_typed<std::int16_t>(dst, src, n);
        break;
    case Dtype::I8:
        add_typed<std::int8_t>(dst, src, n);
        break;
    default:
        ErrorBuilder("accumulate_into")
            .not_implemented("no accumulation for dtype " + std::string(dtype_name(lane)));
    }
}

// Perform dst += src for two CpuStorage buffers.
// Validates dtype and byte-count equality, then adds at the dtype's width.
void cpu_add_inplace(CpuStorage& dst, const CpuStorage& src) {
    if (dst.dtype != src.dtype) {
        throw DtypeMismatch(std::string(dtype_name(dst.dtype)), std::string(dtype_name(src.dtype)),
                            "accumulate_into");
    }
    if (dst.nbytes != src.nbytes) {
        ErrorBuilder("accumulate_into").fail("nbytes mismatch");
    }
    const std::size_t n = dst.nbytes / dtype_size(dst.dtype);
    std::byte* d = dst.ptr.get();
    const std::byte* s = src.ptr.get();
    switch (dst.dtype) {
    case Dtype::F16:
        add_half_inplace<backend::detail::half_bits_to_float, backend::detail::float_to_half_bits>(
            d, s, n);
        break;
    case Dtype::BF16:
        add_half_inplace<backend::detail::bfloat_bits_to_float,
                         backend::detail::float_to_bfloat_bits>(d, s, n);
        break;
    case Dtype::Bool: {
        // bool + bool is logical or; adding the bytes would leave a 2.
        auto* td = reinterpret_cast<std::uint8_t*>(d);
        const auto* ts = reinterpret_cast<const std::uint8_t*>(s);
        for (std::size_t i = 0; i < n; ++i)
            td[i] = (td[i] | ts[i]) != 0 ? 1 : 0;
        break;
    }
    case Dtype::C64:
    case Dtype::C128:
        // Complex addition *is* lane-wise addition, so an interleaved
        // buffer accumulates as twice as many reals.
        //
        // Reaching here at all is new: nothing produced a complex
        // gradient until ``real`` / ``imag`` / ``conj`` were given
        // backwards, and the first thing that did was ``abs(z)``, which
        // reads ``z`` through both projections and so needs its two
        // contributions summed.
        add_lanes(real_lane_of(dst.dtype), d, s, n * 2);
        break;
    default:
        add_lanes(dst.dtype, d, s, n);
    }
}

}  // namespace

Storage make_zero_storage(const Shape& shape, Dtype dtype, Device device) {
    return backend::Dispatcher::for_device(device).zeros(shape, dtype);
}

Storage make_ones_storage(const Shape& shape, Dtype dtype, Device device) {
    return backend::Dispatcher::for_device(device).ones(shape, dtype);
}

Storage reduce_grad_to_shape(const Storage& grad,
                             const Shape& grad_shape,
                             const Shape& target_shape,
                             Dtype dtype,
                             Device device) {
    if (grad_shape == target_shape)
        return grad;
    return backend::Dispatcher::for_device(device).reduce_grad_to_shape(grad, grad_shape,
                                                                        target_shape, dtype);
}

// Dispatch accumulation to the appropriate device handler using std::visit.
//
// The visitor handles five cases:
//   CPU+CPU          — direct typed loop via cpu_add_inplace.
//   GPU+GPU          — MLX add, rewrap result into the destination GpuStorage.
//   SharedStorage+SharedStorage — obtain cpu_view() of both, then cpu_add_inplace.
//   SharedStorage+CpuStorage   — view the shared buffer as CPU, then add.
//   CpuStorage+SharedStorage   — view the shared source as CPU, then add.
// Any other combination (CPU+GPU, GPU+CPU, etc.) throws DeviceMismatch.
void accumulate_into(Storage& dst, const Storage& src) {
    // The phantom write rule (core/PhantomMode.h): a phantom slot takes any
    // gradient and stays a phantom; a real one refuses a phantom.
    if (absorbs_phantom_write(dst, src, "accumulate_into"))
        return;
    // 3.3 AMP fix: when a non-leaf intermediate gradient is accumulated from
    // two paths whose effective dtypes differ (e.g. Promote + ForceFP32
    // siblings under autocast(F16)), coerce src to dst's dtype before
    // adding.  Without this, ``mlx::core::add`` would throw DtypeMismatch
    // and ``cpu_add_inplace`` would assert.  Casting src (rather than
    // promoting dst) keeps the destination dtype stable across additions
    // — once the first grad lands the slot's dtype is fixed, and all
    // subsequent grads conform.  Element-wise add is shape-agnostic so a
    // 1-D ``{numel}`` shape passed to ``be.astype`` is sufficient.
    const Dtype dst_dt = storage_dtype(dst);
    const Dtype src_dt = storage_dtype(src);
    Storage src_owned;
    const Storage* src_ptr = &src;
    if (src_dt != dst_dt) {
        const Device dev = storage_device(src);
        const std::size_t n = storage_nbytes(src) / dtype_size(src_dt);
        Shape one_d_shape = {static_cast<std::int64_t>(n)};
        src_owned = backend::Dispatcher::for_device(dev).astype(src, one_d_shape, src_dt, dst_dt);
        src_ptr = &src_owned;
    }
    std::visit(overloaded{
                   [&](CpuStorage& d, const CpuStorage& s) { cpu_add_inplace(d, s); },
                   [&](GpuStorage& d, const GpuStorage& s) {
                       if (!d.arr || !s.arr) {
                           ErrorBuilder("accumulate_into").fail("null GPU array");
                       }
                       if (d.dtype != s.dtype) {
                           throw DtypeMismatch(std::string(dtype_name(d.dtype)),
                                               std::string(dtype_name(s.dtype)), "accumulate_into");
                       }
                       // MLX add produces a new array; replace the destination's
                       // arr pointer in-place so callers holding a reference to
                       // d see the updated value.
                       auto next = ::mlx::core::add(*d.arr, *s.arr);
                       TensorImpl::sync_gpu_grad_slot(
                           d, gpu::wrap_mlx_array(std::move(next), d.dtype).arr);
                   },

                   [&](SharedStorage& d, const SharedStorage& s) {
                       auto dv = d.cpu_view();
                       auto sv = s.cpu_view();
                       cpu_add_inplace(dv, sv);
                   },
                   [&](SharedStorage& d, const CpuStorage& s) {
                       auto dv = d.cpu_view();
                       cpu_add_inplace(dv, s);
                   },
                   [&](CpuStorage& d, const SharedStorage& s) {
                       auto sv = s.cpu_view();
                       cpu_add_inplace(d, sv);
                   },
                   [&](auto&, auto&) {
                       throw DeviceMismatch("matching device", "mixed cpu/metal",
                                            "accumulate_into");
                   },
               },
               dst, *src_ptr);
}

// -------------------------------------------------------------------------
// Element-wise arithmetic helpers.
// Each function constructs a flat 1-D Shape from numel and delegates to the
// Dispatcher so that the correct backend (CPU/GPU) is used automatically.
// -------------------------------------------------------------------------

Storage negate_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).neg(s, flat, dt);
}

Storage
multiply_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).mul(a, b, flat, dt);
}

Storage
divide_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).div(a, b, flat, dt);
}

Storage
add_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).add(a, b, flat, dt);
}

Storage
subtract_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).sub(a, b, flat, dt);
}

Storage square_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).square(s, flat, dt);
}

Storage clone_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).clone(s, flat, dt);
}

Storage log_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).log(s, flat, dt);
}

Storage
pow_storage(const Storage& base, const Storage& expo, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).pow(base, expo, flat, dt);
}

Storage
ge_mask_storage(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).ge_mask(a, b, flat, dt);
}

Storage
lt_mask_storage(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).lt_mask(a, b, flat, dt);
}

Storage
add_scalar_storage(const Storage& s, double scalar, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).add_scalar(s, flat, dt, scalar);
}

Storage
mul_scalar_storage(const Storage& s, double scalar, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).mul_scalar(s, flat, dt, scalar);
}

// Macro-generated unary helpers.  Each expands to a free function that
// wraps numel into a flat Shape and calls the matching Dispatcher method.
#define LUCID_UNARY_HELPER(NAME, BACKEND_METHOD)                                                   \
    Storage NAME##_storage(const Storage& s, std::size_t n, Dtype dt, Device device) {             \
        const Shape flat{static_cast<std::int64_t>(n)};                                            \
        return backend::Dispatcher::for_device(device).BACKEND_METHOD(s, flat, dt);                \
    }

LUCID_UNARY_HELPER(exp, exp)
LUCID_UNARY_HELPER(sqrt, sqrt)
LUCID_UNARY_HELPER(abs, abs)
LUCID_UNARY_HELPER(reciprocal, reciprocal)
LUCID_UNARY_HELPER(sin, sin)
LUCID_UNARY_HELPER(cos, cos)
LUCID_UNARY_HELPER(asin, asin)
LUCID_UNARY_HELPER(acos, acos)
LUCID_UNARY_HELPER(atan, atan)
LUCID_UNARY_HELPER(sinh, sinh)
LUCID_UNARY_HELPER(cosh, cosh)
LUCID_UNARY_HELPER(tanh, tanh)

#undef LUCID_UNARY_HELPER

// tan and sign are defined outside the macro because they share the same
// dispatcher method name as other functions (avoiding collision) or because
// there is no 1-to-1 macro match.
Storage tan_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).tan(s, flat, dt);
}

Storage sign_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).sign(s, flat, dt);
}

Storage in_range_mask_storage(const Storage& s,
                              double lo,
                              double hi,
                              std::size_t numel,
                              Dtype dt,
                              Device device,
                              bool inclusive) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).in_range_mask(s, flat, dt, lo, hi, inclusive);
}

// -------------------------------------------------------------------------
// Version counter validation.
// -------------------------------------------------------------------------

// Process-wide flag controlled by
// ``lucid.autograd.graph.allow_mutation_on_saved_tensors``.  When set,
// ``check_version_match`` becomes a no-op — the user has explicitly opted
// into the unsafe contract that they will not mutate saved tensors in a
// way that would corrupt the gradient.  The flag
// is intentionally a plain ``bool`` (no atomic / thread-local): autograd is
// already a single-threaded contract in Lucid and the cost of synchronisation
// would dwarf the actual check.
static bool g_allow_mutation_on_saved = false;

bool is_mutation_on_saved_allowed() {
    return g_allow_mutation_on_saved;
}

void set_mutation_on_saved_allowed(bool v) {
    g_allow_mutation_on_saved = v;
}

static bool g_anomaly_enabled = false;

bool is_anomaly_enabled() {
    return g_anomaly_enabled;
}

void set_anomaly_enabled(bool v) {
    g_anomaly_enabled = v;
}

namespace {

template <typename T>
bool has_nan_typed(const std::byte* raw, std::size_t n) {
    const auto* p = reinterpret_cast<const T*>(raw);
    for (std::size_t i = 0; i < n; ++i)
        if (std::isnan(p[i]))
            return true;
    return false;
}

template <float (*Widen)(std::uint16_t)>
bool has_nan_half(const std::byte* raw, std::size_t n) {
    const auto* p = reinterpret_cast<const std::uint16_t*>(raw);
    for (std::size_t i = 0; i < n; ++i)
        if (std::isnan(Widen(p[i])))
            return true;
    return false;
}

bool cpu_storage_has_nan(const CpuStorage& c) {
    if (!c.ptr || c.nbytes == 0)
        return false;
    const std::size_t n = c.nbytes / dtype_size(c.dtype);
    const std::byte* raw = c.ptr.get();
    switch (c.dtype) {
    case Dtype::F32:
        return has_nan_typed<float>(raw, n);
    case Dtype::F64:
        return has_nan_typed<double>(raw, n);
    case Dtype::F16:
        return has_nan_half<backend::detail::half_bits_to_float>(raw, n);
    case Dtype::BF16:
        return has_nan_half<backend::detail::bfloat_bits_to_float>(raw, n);
    case Dtype::C64:
        return has_nan_typed<float>(raw, 2 * n);
    case Dtype::C128:
        return has_nan_typed<double>(raw, 2 * n);
    default:
        return false;
    }
}

}  // namespace

std::string backward_node_name(const Node& node) {
    const std::string cls = node.Node::node_name();
    if (cls.size() > 8 && cls.rfind("Backward") == cls.size() - 8)
        return cls;
    return node.node_name();
}

bool storage_has_nan(const Storage& s) {
    return std::visit(
        overloaded{
            [](const CpuStorage& c) { return cpu_storage_has_nan(c); },
            [](const GpuStorage& g) {
                if (!g.arr || g.nbytes == 0)
                    return false;
                if (!is_floating_point(g.dtype) && !is_complex(g.dtype))
                    return false;
                auto has_nan = ::mlx::core::any(::mlx::core::isnan(*g.arr));
                has_nan.eval();
                return has_nan.item<bool>();
            },
            [](const SharedStorage& sh) { return cpu_storage_has_nan(sh.cpu_view()); },
            // Anomaly detection looks for a NaN among the values; a phantom
            // has none to hold one.
            [](const PhantomStorage&) { return false; },
        },
        s);
}

void check_anomaly_grads(const Node& node, const std::vector<Storage>& grads) {
    if (!g_anomaly_enabled)
        return;
    for (std::size_t i = 0; i < grads.size(); ++i) {
        if (node.empty_grad_is_none() && storage_nbytes(grads[i]) == 0)
            continue;
        if (storage_has_nan(grads[i])) {
            ErrorBuilder("detect_anomaly")
                .fail("Function '" + backward_node_name(node) + "' returned nan values in its " +
                      std::to_string(i) + "th output.");
        }
    }
}

void check_anomaly_grads_for_graph(const Node& node,
                                   const std::vector<std::shared_ptr<TensorImpl>>& grads) {
    if (!g_anomaly_enabled)
        return;
    for (std::size_t i = 0; i < grads.size(); ++i) {
        if (!grads[i])
            continue;
        if (storage_has_nan(grads[i]->storage())) {
            ErrorBuilder("detect_anomaly")
                .fail("Function '" + backward_node_name(node) + "' returned nan values in its " +
                      std::to_string(i) + "th output.");
        }
    }
}

// Compare the version counter of the live TensorImpl against the saved value.
// A discrepancy means the tensor was mutated in-place between the forward pass
// and this backward call — which invalidates the saved activation data.
void check_version_match(const std::weak_ptr<TensorImpl>& live,
                         std::int64_t saved_version,
                         const std::string& what) {
    if (g_allow_mutation_on_saved) {
        return;  // user opted into the unsafe contract
    }
    auto t = live.lock();
    if (!t)
        return;
    if (t->version() != saved_version)
        throw VersionMismatch(saved_version, t->version(), what);
}

void check_version_match(const std::weak_ptr<TensorImpl>& live,
                         std::int64_t saved_version,
                         std::string_view op_name,
                         std::size_t input_idx) {
    if (g_allow_mutation_on_saved || live.expired())
        return;
    check_version_match(live, saved_version,
                        std::string(op_name) + " input " + std::to_string(input_idx));
}

std::shared_ptr<TensorImpl> forward_time_input(const std::shared_ptr<TensorImpl>& live,
                                               const Storage& values,
                                               const Shape& shape,
                                               const Edge& edge,
                                               std::string_view op_name,
                                               std::size_t input_idx) {
    // Only an in-place write that records a graph moves a non-leaf to another
    // grad_fn: it gives the tensor the node of the write.  A leaf never moves
    // this way — its write is refused while autograd records, and outside it
    // the version moves instead, which validate_versions has refused by now.
    // A tensor without a grad_fn was detached (``detach_``, or a write outside
    // the graph, whose version moved too) and is read as the constant it is.
    if (!live || live->is_leaf() || !live->grad_fn())
        return nullptr;
    if (live->grad_fn() == edge.node && live->grad_output_nr() == edge.input_nr)
        return nullptr;
    // Nothing saved by value: the formula reads no values of this input, and
    // the tensor it reads stands for its shape.
    if (!holds_saved_values(values))
        return nullptr;
    const std::size_t nbytes = storage_nbytes(values);
    const Dtype dtype = storage_dtype(values);
    if (nbytes != shape_numel(shape) * dtype_size(dtype)) {
        std::string msg = std::string(op_name) + " input " + std::to_string(input_idx);
        msg += " was modified by an in-place operation after the forward, and the values it "
               "held then cannot be rebuilt for create_graph=True — clone() it before the write";
        ErrorBuilder("backward").fail(msg);
    }
    auto t = std::make_shared<TensorImpl>(values, shape, dtype, storage_device(values),
                                          edge.node != nullptr);
    if (edge.node) {
        t->set_grad_fn(edge.node);
        t->set_grad_output_nr(edge.input_nr);
        t->set_leaf(dynamic_cast<const AccumulateGrad*>(edge.node.get()) != nullptr);
    }
    return t;
}

// -------------------------------------------------------------------------
// Shape / axis helpers for reduction backward passes.
// -------------------------------------------------------------------------

int wrap_dim(int dim, int ndim, const char* op) {
    const int extent = std::max(ndim, 1);
    const int wrapped = dim < 0 ? dim + extent : dim;
    if (wrapped < 0 || wrapped >= extent)
        ErrorBuilder(op).index_error("axis out of range: got " + std::to_string(dim) +
                                     ", expected [" + std::to_string(-extent) + ", " +
                                     std::to_string(extent - 1) + "] for a tensor of rank " +
                                     std::to_string(ndim));
    return wrapped;
}

// Resolve axis indices to a sorted list of non-negative values.  An empty
// input axes list is interpreted as "all axes".
std::vector<int> normalize_axes(const std::vector<int>& axes, int ndim, const char* op) {
    if (axes.empty()) {
        std::vector<int> out;
        out.reserve(ndim);
        for (int i = 0; i < ndim; ++i)
            out.push_back(i);
        return out;
    }
    std::vector<int> out = distinct_dims(axes, ndim, op);
    // A 0-d tensor's one axis is notional: reducing over it is the
    // identity, which is the empty axis list of rank 0.
    if (ndim == 0)
        return {};
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<int> distinct_dims(const std::vector<int>& dims, int ndim, const char* op) {
    std::vector<int> out;
    out.reserve(dims.size());
    std::vector<bool> seen(static_cast<std::size_t>(std::max(ndim, 1)), false);
    for (int d : dims) {
        const int wrapped = wrap_dim(d, ndim, op);
        if (seen[static_cast<std::size_t>(wrapped)])
            ErrorBuilder(op).fail("dim " + std::to_string(wrapped) +
                                  " appears multiple times in the list of dims");
        seen[static_cast<std::size_t>(wrapped)] = true;
        out.push_back(wrapped);
    }
    return out;
}

// Build the output shape of a reduction, applying keepdims semantics.
// Reduced dimensions either disappear (keepdims=false) or become 1.
Shape reduce_output_shape(const Shape& input_shape, const std::vector<int>& axes, bool keepdims) {
    std::vector<bool> reduce_mask(input_shape.size(), false);
    for (int a : axes)
        reduce_mask[a] = true;
    Shape out;
    out.reserve(input_shape.size());
    for (std::size_t i = 0; i < input_shape.size(); ++i) {
        if (reduce_mask[i]) {
            if (keepdims)
                out.push_back(1);
            // else: dimension is dropped
        } else {
            out.push_back(input_shape[i]);
        }
    }
    return out;
}

// Expand grad back from the reduced shape to input_shape so that it can be
// added to the input's gradient buffer.
//
// The expected shape of grad is validated against reduce_output_shape() to
// catch mismatches early.  The actual broadcasting is delegated to the
// Dispatcher's broadcast_back_for_reduce(), which handles both CPU and GPU.
Storage broadcast_back_for_reduce(const Storage& grad,
                                  const Shape& grad_shape,
                                  const Shape& input_shape,
                                  const std::vector<int>& axes,
                                  bool keepdims,
                                  Dtype dt,
                                  Device device) {
    Shape expected_grad = reduce_output_shape(input_shape, axes, keepdims);
    if (grad_shape != expected_grad) {
        throw ShapeMismatch(expected_grad, grad_shape,
                            "broadcast_back_for_reduce: grad shape mismatch");
    }
    return backend::Dispatcher::for_device(device).broadcast_back_for_reduce(
        grad, grad_shape, input_shape, axes, keepdims, dt);
}

Storage
leaky_mask_storage(const Storage& s, double slope, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).leaky_mask(s, flat, dt, slope);
}

Storage sigmoid_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).sigmoid(s, flat, dt);
}

// -------------------------------------------------------------------------
// Random tensor generation.  CPU device: produced on CPU then transferred.
// GPU device: filled on-device via the backend's mlx::core::random path,
// seeded by one draw from the Generator below (so results stay reproducible
// from the global seed) — avoids the per-element CPU loop + host->device upload.
// -------------------------------------------------------------------------

namespace {
// Pull a 64-bit PRNG key seed from the framework Generator (one counter
// advance, under its lock).  Used to seed on-device MLX RNG so GPU random stays
// reproducible from the global seed.  The exact values differ from the CPU
// per-element stream by design.
std::uint64_t gpu_random_key_seed(Generator& gen) {
    std::lock_guard<std::mutex> lock(gen.mutex());
    std::uint32_t k[4];
    gen.next_uint32x4(k);
    return (static_cast<std::uint64_t>(k[0]) << 32) | static_cast<std::uint64_t>(k[1]);
}

// Extract 53 random bits from two 32-bit Philox words and scale by 2^-53 to
// produce a full-mantissa IEEE-754 double in [0, 1).
inline double next_uniform_double(Generator& gen) {
    std::uint32_t buf[4];
    gen.next_uint32x4(buf);
    const std::uint64_t bits53 =
        ((static_cast<std::uint64_t>(buf[0]) << 32) | static_cast<std::uint64_t>(buf[1])) >> 11;
    return static_cast<double>(bits53) * 0x1.0p-53;
}

template <typename T>
inline T next_uniform(Generator& gen) {
    if constexpr (std::is_same_v<T, double>) {
        return next_uniform_double(gen);
    } else {
        return static_cast<T>(gen.next_uniform_float());
    }
}
}  // namespace

// Generate a Bernoulli mask with explicit shape.  Samples are drawn on CPU
// using the Generator's uniform float stream, then transferred to the target
// device via Dispatcher::from_cpu().
Storage bernoulli_mask_storage_shape(
    double keep_prob, const Shape& shape, Dtype dt, Device device, Generator& gen) {
    if (keep_prob < 0.0 || keep_prob > 1.0) {
        ErrorBuilder("bernoulli_mask").invalid_argument("keep_prob must be in [0, 1]");
    }
    // No draw under phantom mode, so the generator does not advance either.
    if (PhantomMode::is_enabled())
        return phantom_storage(shape, dt, device);
    // GPU: draw the whole mask on-device via MLX RNG instead of the per-element
    // CPU Philox loop + host->device upload below (which dominated Dropout's
    // cost — ~19 ms for a single (32,128,768) mask).  We advance the framework
    // Generator by one draw to seed the MLX key, so masks stay reproducible from
    // the global seed (the exact values differ from the CPU stream by design).
    if (device == Device::GPU) {
        return backend::Dispatcher::for_device(device).bernoulli_mask(keep_prob, shape, dt,
                                                                      gpu_random_key_seed(gen));
    }
    // Half draws wide and rounds, as the other CPU generators do — and a
    // Bernoulli mask loses nothing at all on the way, since 0 and 1 are
    // exact in every float format.  Without this, ``dropout`` and its
    // three spatial variants refused float16 on the CPU while Metal ran
    // them, which made ``model.half()`` a device-dependent proposition.
    if (is_half_float(dt)) {
        const Storage wide =
            bernoulli_mask_storage_shape(keep_prob, shape, Dtype::F32, device, gen);
        return backend::Dispatcher::for_device(device).astype(wide, shape, Dtype::F32, dt);
    }
    std::size_t numel = 1;
    for (auto d : shape)
        numel *= static_cast<std::size_t>(d);
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = numel * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    switch (dt) {
    case Dtype::F32: {
        auto* q = reinterpret_cast<float*>(out.ptr.get());
        const auto kp = static_cast<float>(keep_prob);
        for (std::size_t i = 0; i < numel; ++i) {
            q[i] = (gen.next_uniform_float() < kp) ? 1.f : 0.f;
        }
        break;
    }
    case Dtype::F64: {
        auto* q = reinterpret_cast<double*>(out.ptr.get());
        for (std::size_t i = 0; i < numel; ++i) {
            q[i] = (next_uniform_double(gen) < keep_prob) ? 1.0 : 0.0;
        }
        break;
    }
    default:
        ErrorBuilder("bernoulli_mask").not_implemented("dtype not supported (F32/F64)");
    }
    return backend::Dispatcher::for_device(device).from_cpu(std::move(out), shape);
}

// Flat-numel convenience wrapper around bernoulli_mask_storage_shape.
Storage bernoulli_mask_storage(
    double keep_prob, std::size_t numel, Dtype dt, Device device, Generator& gen) {
    Shape flat;
    flat.push_back(static_cast<std::int64_t>(numel));
    return bernoulli_mask_storage_shape(keep_prob, flat, dt, device, gen);
}

Storage positive_mask_storage(const Storage& s, std::size_t numel, Dtype dt, Device device) {
    const Shape flat{static_cast<std::int64_t>(numel)};
    return backend::Dispatcher::for_device(device).positive_mask(s, flat, dt);
}

namespace {

// Fill dst[0..numel) with uniform floats scaled to [lo, hi).
template <typename T>
void fill_uniform(T* dst, std::size_t numel, double lo, double hi, Generator& gen) {
    const T span = static_cast<T>(hi - lo);
    const T offset = static_cast<T>(lo);
    for (std::size_t i = 0; i < numel; ++i) {
        dst[i] = next_uniform<T>(gen) * span + offset;
    }
}

// Fill dst[0..numel) with samples from N(mean, std) using the Box-Muller
// transform.  Pairs of uniform samples produce two normal samples, so the
// loop processes two elements at a time with a tail case for odd numel.
// u1 is clamped away from zero to prevent log(0).
template <typename T>
void fill_normal(T* dst, std::size_t numel, double mean, double std, Generator& gen) {
    const T m = static_cast<T>(mean);
    const T s = static_cast<T>(std);
    constexpr T two_pi = static_cast<T>(6.28318530717958647692);
    constexpr T eps = std::is_same_v<T, double> ? static_cast<T>(1e-15) : static_cast<T>(1e-7);
    std::size_t i = 0;
    while (i + 1 < numel) {
        T u1 = next_uniform<T>(gen);
        if (u1 < eps)
            u1 = eps;
        T u2 = next_uniform<T>(gen);
        const T r = std::sqrt(static_cast<T>(-2) * std::log(u1));
        const T z0 = r * std::cos(two_pi * u2);
        const T z1 = r * std::sin(two_pi * u2);
        dst[i] = m + s * z0;
        dst[i + 1] = m + s * z1;
        i += 2;
    }
    if (i < numel) {
        // Odd element: use only z0 from the final Box-Muller pair.
        T u1 = next_uniform<T>(gen);
        if (u1 < eps)
            u1 = eps;
        T u2 = next_uniform<T>(gen);
        const T r = std::sqrt(static_cast<T>(-2) * std::log(u1));
        dst[i] = m + s * r * std::cos(two_pi * u2);
    }
}

// Fill dst[0..numel) with integers uniformly drawn from [low, high).
// Each element uses a 64-bit value formed from two 32-bit Philox words so
// large 31/32-bit and 64-bit ranges are free of 32-bit modulo bias, and
// span/addition run in uint64_t to avoid signed overflow on wide bounds.
template <typename Int>
void fill_randint(
    Int* dst, std::size_t numel, std::int64_t low, std::int64_t high, Generator& gen) {
    const std::uint64_t ulow = static_cast<std::uint64_t>(low);
    const std::uint64_t range = static_cast<std::uint64_t>(high) - ulow;
    if (range == 0) {
        for (std::size_t i = 0; i < numel; ++i)
            dst[i] = static_cast<Int>(low);
        return;
    }
    std::uint32_t buf[4];
    std::size_t i = 0;
    while (i < numel) {
        gen.next_uint32x4(buf);
        for (int k = 0; k < 2 && i < numel; ++k, ++i) {
            const std::uint64_t r = (static_cast<std::uint64_t>(buf[2 * k]) << 32) |
                                    static_cast<std::uint64_t>(buf[2 * k + 1]);
            dst[i] = static_cast<Int>(ulow + (r % range));
        }
    }
}

// Allocate an uninitialized CpuStorage sized for shape and dtype.
CpuStorage allocate_for_random(const Shape& shape, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = shape_numel(shape) * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

// The CPU draws at float32 and rounds, for the three float generators.
//
// There is nothing to widen here — a generator has no input, only an
// output dtype — so the widen-reuse-narrow pattern used everywhere else
// in the CPU backend reduces to its second half.  Rounding once at the
// end is also the only approximation: every float16 and bfloat16 is
// exactly representable in float32, so the draw itself is unaffected and
// the result is the one the caller asked for by naming a 16-bit dtype.
//
// Without this, ``rand_like``, ``randn_like``, ``dropout`` and every
// ``nn.init`` routine refused half on the CPU while Metal ran them —
// sixteen public symbols, all behind these three switches.
Storage random_half_via_f32(const Shape& shape,
                            Dtype dt,
                            Device device,
                            const std::function<Storage(Dtype)>& draw) {
    const Storage wide = draw(Dtype::F32);
    return backend::Dispatcher::for_device(device).astype(wide, shape, Dtype::F32, dt);
}

}  // namespace

namespace {

// Random draws are floating or complex.  Checked before the device split:
// on Metal an integer dtype reached MLX, whose ValueError came through raw.
void require_random_dtype(const char* op, Dtype dt) {
    if (!is_floating_point(dt) && !is_complex(dt))
        ErrorBuilder(op).not_implemented("not implemented for " + std::string(dtype_name(dt)) +
                                         "; draw a floating tensor and cast it");
}

}  // namespace

Storage random_uniform_storage(
    const Shape& shape, double lo, double hi, Dtype dt, Device device, Generator& gen) {
    require_random_dtype("random_uniform", dt);
    if (PhantomMode::is_enabled())
        return phantom_storage(shape, dt, device);
    if (device == Device::GPU) {
        return backend::Dispatcher::for_device(device).random_uniform(shape, lo, hi, dt,
                                                                      gpu_random_key_seed(gen));
    }
    if (is_half_float(dt))
        return random_half_via_f32(shape, dt, device, [&](Dtype inner) {
            return random_uniform_storage(shape, lo, hi, inner, device, gen);
        });
    auto cpu = allocate_for_random(shape, dt);
    const std::size_t n = shape_numel(shape);
    switch (dt) {
    case Dtype::F32:
        fill_uniform<float>(reinterpret_cast<float*>(cpu.ptr.get()), n, lo, hi, gen);
        break;
    case Dtype::F64:
        fill_uniform<double>(reinterpret_cast<double*>(cpu.ptr.get()), n, lo, hi, gen);
        break;
    case Dtype::C64:
        // Each part uniform on [lo, hi), as the reference and Metal draw it.
        fill_uniform<float>(reinterpret_cast<float*>(cpu.ptr.get()), 2 * n, lo, hi, gen);
        break;
    default:
        ErrorBuilder("random_uniform").not_implemented("dtype not supported (F32/F64)");
    }
    return backend::Dispatcher::for_device(device).from_cpu(std::move(cpu), shape);
}

Storage random_normal_storage(
    const Shape& shape, double mean, double std, Dtype dt, Device device, Generator& gen) {
    require_random_dtype("random_normal", dt);
    if (PhantomMode::is_enabled())
        return phantom_storage(shape, dt, device);
    if (device == Device::GPU) {
        return backend::Dispatcher::for_device(device).random_normal(shape, mean, std, dt,
                                                                     gpu_random_key_seed(gen));
    }
    if (is_half_float(dt))
        return random_half_via_f32(shape, dt, device, [&](Dtype inner) {
            return random_normal_storage(shape, mean, std, inner, device, gen);
        });
    auto cpu = allocate_for_random(shape, dt);
    const std::size_t n = shape_numel(shape);
    switch (dt) {
    case Dtype::F32:
        fill_normal<float>(reinterpret_cast<float*>(cpu.ptr.get()), n, mean, std, gen);
        break;
    case Dtype::F64:
        fill_normal<double>(reinterpret_cast<double*>(cpu.ptr.get()), n, mean, std, gen);
        break;
    case Dtype::C64: {
        // Each part N(0, std^2 / 2), so |z - mean|^2 averages std^2 — the
        // reference's convention, and MLX's on Metal.  ``mean`` is real.
        auto* parts = reinterpret_cast<float*>(cpu.ptr.get());
        fill_normal<float>(parts, 2 * n, 0.0, std / std::sqrt(2.0), gen);
        for (std::size_t i = 0; i < n; ++i)
            parts[2 * i] += static_cast<float>(mean);
        break;
    }
    default:
        ErrorBuilder("random_normal").not_implemented("dtype not supported (F32/F64)");
    }
    return backend::Dispatcher::for_device(device).from_cpu(std::move(cpu), shape);
}

Storage
random_bernoulli_storage(const Shape& shape, double p, Dtype dt, Device device, Generator& gen) {
    if (p < 0.0 || p > 1.0)
        ErrorBuilder("random_bernoulli").invalid_argument("p must be in [0, 1]");
    if (PhantomMode::is_enabled())
        return phantom_storage(shape, dt, device);
    if (device == Device::GPU) {
        return backend::Dispatcher::for_device(device).random_bernoulli(shape, p, dt,
                                                                        gpu_random_key_seed(gen));
    }
    if (is_half_float(dt))
        return random_half_via_f32(shape, dt, device, [&](Dtype inner) {
            return random_bernoulli_storage(shape, p, inner, device, gen);
        });
    auto cpu = allocate_for_random(shape, dt);
    const std::size_t n = shape_numel(shape);
    switch (dt) {
    case Dtype::F32: {
        auto* q = reinterpret_cast<float*>(cpu.ptr.get());
        const auto fp = static_cast<float>(p);
        for (std::size_t i = 0; i < n; ++i) {
            q[i] = (gen.next_uniform_float() < fp) ? 1.f : 0.f;
        }
        break;
    }
    case Dtype::F64: {
        auto* q = reinterpret_cast<double*>(cpu.ptr.get());
        for (std::size_t i = 0; i < n; ++i) {
            q[i] = (next_uniform_double(gen) < p) ? 1.0 : 0.0;
        }
        break;
    }
    default:
        ErrorBuilder("random_bernoulli").not_implemented("dtype not supported (F32/F64)");
    }
    return backend::Dispatcher::for_device(device).from_cpu(std::move(cpu), shape);
}

Storage random_randint_storage(const Shape& shape,
                               std::int64_t low,
                               std::int64_t high,
                               Dtype dt,
                               Device device,
                               Generator& gen) {
    if (high <= low)
        ErrorBuilder("random_randint").invalid_argument("high must be > low");
    if (PhantomMode::is_enabled())
        return phantom_storage(shape, dt, device);
    if (device == Device::GPU) {
        return backend::Dispatcher::for_device(device).random_randint(shape, low, high, dt,
                                                                      gpu_random_key_seed(gen));
    }
    auto cpu = allocate_for_random(shape, dt);
    const std::size_t n = shape_numel(shape);
    switch (dt) {
    case Dtype::I8:
        fill_randint<std::int8_t>(reinterpret_cast<std::int8_t*>(cpu.ptr.get()), n, low, high, gen);
        break;
    case Dtype::I16:
        fill_randint<std::int16_t>(reinterpret_cast<std::int16_t*>(cpu.ptr.get()), n, low, high,
                                   gen);
        break;
    case Dtype::I32:
        fill_randint<std::int32_t>(reinterpret_cast<std::int32_t*>(cpu.ptr.get()), n, low, high,
                                   gen);
        break;
    case Dtype::I64:
        fill_randint<std::int64_t>(reinterpret_cast<std::int64_t*>(cpu.ptr.get()), n, low, high,
                                   gen);
        break;
    default:
        ErrorBuilder("random_randint").not_implemented("dtype not supported (I8/I16/I32/I64)");
    }
    return backend::Dispatcher::for_device(device).from_cpu(std::move(cpu), shape);
}

}  // namespace lucid
