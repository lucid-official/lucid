// lucid/_C/bindings/bind_tensor.cpp
//
// Exposes the two fundamental value-type APIs of the Lucid engine to Python:
//
//   register_core()   — Device enum, Dtype enum, dtype helpers, NoGradGuard
//                       context manager, global grad-mode toggle, and
//                       MemoryStats / MemoryTracker.
//
//   register_tensor_impl() — TensorImpl (Python alias: Tensor), the central
//                            tensor type.  TensorImpl is held by shared_ptr
//                            on both sides; pybind11 reference-counts via the
//                            shared_ptr holder.  Also registers to_shared_storage()
//                            which re-wraps storage as MTLResourceStorageModeShared
//                            for zero-copy Metal kernel dispatch.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mlx/ops.h>
#include <mlx/transforms.h>  // mlx::core::eval(std::vector<array>)

#include "../autograd/Node.h"
#include "../autograd/TensorHooks.h"
#include "../backend/Dispatcher.h"
#include "../backend/gpu/MetalAllocator.h"
#include "../backend/gpu/MlxBridge.h"
#include "../compile/Tracer.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/MemoryStats.h"
#include "../core/PhantomMode.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"

namespace py = pybind11;

namespace {
// Every binding below that hands Python a tensor's values calls
// ``note_host_read``; every one that returns a new TensorImpl made from an
// existing one returns it through ``traced_derive``, and one that builds a
// fresh tensor from host data calls ``traced_factory`` (see Tracer.h).
using lucid::compile::note_host_read;
using lucid::compile::traced_derive;
using lucid::compile::traced_factory;

// Run a GPU tensor's pending work without holding the GIL.  The wait is the
// GPU's; holding the GIL through it stalled every other Python thread in the
// process — a server thread froze for each 1.6-3 s model pass.  Reading the
// values back afterwards (``item``, ``tolist``, ``numpy``) builds Python
// objects, so it runs with the GIL again and finds the array evaluated.
//
// The array is resolved while the GIL is still held.  A Metal view reads it
// out of its family's root and caches it in the tensor, and another Python
// thread may write the family meanwhile — replacing the root and dropping
// that cache — so only a handle owned here is safe to wait on unlocked.
//
// ``pending_array`` is that handle: null for a CPU tensor, a Metal tensor
// without an array, and a phantom, which has no graph to run.
std::shared_ptr<mlx::core::array> pending_array(const lucid::TensorImpl& t) {
    if (t.device() != lucid::Device::GPU || t.is_phantom())
        return nullptr;
    const auto* gpu_st = std::get_if<lucid::GpuStorage>(&t.storage());
    return gpu_st != nullptr ? gpu_st->arr : nullptr;
}

void evaluate_without_gil(const lucid::TensorImpl& t) {
    const std::shared_ptr<mlx::core::array> arr = pending_array(t);
    if (!arr)
        return;
    py::gil_scoped_release release;
    arr->eval();
}

// The arrays of ``tensors`` that have work to run, for one batched eval.
std::vector<mlx::core::array>
pending_arrays(const std::vector<std::shared_ptr<lucid::TensorImpl>>& tensors) {
    std::vector<mlx::core::array> arrays;
    arrays.reserve(tensors.size());
    for (const auto& t : tensors)
        if (const auto arr = t ? pending_array(*t) : nullptr)
            arrays.push_back(*arr);
    return arrays;
}

// The storage's base address plus the view's byte offset, so two tensors at
// the same place in one buffer compare equal.  A GPU array is evaluated
// first: its buffer may not exist yet.  A phantom has no buffer, so no
// address: the storage-sharing plans that compare these would otherwise take
// every phantom for one buffer.
std::uintptr_t data_address(const lucid::TensorImpl& t) {
    const std::size_t off = t.storage_offset();
    const lucid::Storage& s = t.raw_storage();
    if (lucid::storage_is_phantom(s))
        lucid::refuse_phantom_read("data_ptr");
    if (lucid::storage_is_cpu(s))
        return reinterpret_cast<std::uintptr_t>(lucid::storage_cpu(s).ptr.get()) + off;
    if (lucid::storage_is_gpu(s)) {
        const auto& g = lucid::storage_gpu(s);
        if (!g.arr)
            return 0;
        g.arr->eval();
        lucid::MemoryTracker::track_host_sync(g.nbytes);
        return reinterpret_cast<std::uintptr_t>(g.arr->data<std::uint8_t>()) + off;
    }
    return reinterpret_cast<std::uintptr_t>(lucid::storage_metal_shared(s).cpu_ptr) + off;
}

// A zero-filled CPU tensor in MTLResourceStorageModeShared memory — under
// phantom mode, a factory like any other: a phantom, and no Metal buffer.
std::shared_ptr<lucid::TensorImpl>
make_shared_tensor(const std::vector<std::int64_t>& shape, lucid::Dtype dtype, bool requires_grad) {
    if (lucid::PhantomMode::is_enabled())
        return std::make_shared<lucid::TensorImpl>(
            lucid::phantom_storage(lucid::Shape(shape), dtype, lucid::Device::CPU),
            lucid::Shape(shape), dtype, lucid::Device::CPU, requires_grad);
    std::size_t n = 1;
    for (auto d : shape)
        n *= static_cast<std::size_t>(d);
    const std::size_t nbytes = n * lucid::dtype_size(dtype);
    if (nbytes == 0) {
        // Empty tensor: fall back to ordinary CPU zeros.
        auto& be = lucid::backend::Dispatcher::for_device(lucid::Device::CPU);
        lucid::Storage s = be.zeros(lucid::Shape(shape), dtype);
        return std::make_shared<lucid::TensorImpl>(std::move(s), lucid::Shape(shape), dtype,
                                                   lucid::Device::CPU, requires_grad);
    }
    auto owned = lucid::gpu::make_metal_shared(nbytes);
    if (!owned.buf.cpu_ptr)
        throw std::runtime_error("make_shared_tensor: MTLBuffer allocation failed — "
                                 "Metal may not be available on this device");
    // macOS guarantees MTLResourceStorageModeShared buffers are
    // zero-initialized; no explicit memset needed.
    lucid::SharedStorage ss;
    ss.cpu_ptr = owned.buf.cpu_ptr;
    ss.mtl_handle = owned.buf.mtl_handle;
    ss.nbytes = nbytes;
    ss.dtype = dtype;
    ss.owner = std::move(owned.owner);
    return std::make_shared<lucid::TensorImpl>(lucid::Storage{std::move(ss)}, lucid::Shape(shape),
                                               dtype, lucid::Device::CPU, requires_grad);
}

// Puts g's buffer in self's gradient slot, replacing any gradient there;
// refused, with the slot as it was, unless g has self's dtype, device and
// shape (assignable_grad).  Backs ``Tensor.grad = g``.  A phantom's slot is
// the phantom assignable_grad gives, never g's graph.
void set_gradient(lucid::TensorImpl& self, const std::shared_ptr<lucid::TensorImpl>& g) {
    if (!g)
        lucid::ErrorBuilder("Tensor.grad")
            .invalid_argument("expected a tensor; zero_grad() clears a gradient");
    lucid::Storage slot = lucid::assignable_grad(self, *g);
    self.install_grad(g, std::move(slot));
}

std::shared_ptr<lucid::TensorImpl>
to_shared_storage_impl(const std::shared_ptr<lucid::TensorImpl>& t) {
    if (!t)
        throw std::invalid_argument("to_shared_storage: null tensor");
    if (t->metal_shared())
        return t;
    auto& be = lucid::backend::Dispatcher::for_device(lucid::Device::GPU);
    lucid::Storage shared = be.to_shared_storage(t->storage(), t->shape());
    return traced_derive("to_shared_storage", t,
                         std::make_shared<lucid::TensorImpl>(std::move(shared), t->shape(),
                                                             t->dtype(), t->device(),
                                                             t->requires_grad()));
}

std::shared_ptr<lucid::TensorImpl>
transfer_storage_impl(const std::shared_ptr<lucid::TensorImpl>& t, lucid::Device target_device) {
    if (!t)
        throw std::invalid_argument("transfer_storage: null tensor");
    const lucid::SharedStorage* sh = t->metal_shared();
    if (!sh)
        throw std::invalid_argument("transfer_storage: source tensor must be in SharedStorage "
                                    "(call to_shared_storage() or make_shared_tensor() first)");
    return traced_derive("transfer_storage", t,
                         std::make_shared<lucid::TensorImpl>(lucid::Storage{*sh}, t->shape(),
                                                             t->dtype(), target_device,
                                                             t->requires_grad()));
}

std::shared_ptr<lucid::TensorImpl>
make_view_from_elements(const std::shared_ptr<lucid::TensorImpl>& base,
                        const std::vector<std::int64_t>& shape,
                        const std::vector<std::int64_t>& stride,
                        std::int64_t offset) {
    // Testing hook: a view of ``base`` described in elements, for geometries
    // no public op makes.  The offset is from ``base``'s own first element,
    // as make_view's is.
    if (!base || shape.size() != stride.size() || offset < 0)
        throw std::invalid_argument(
            "_make_view: shape and stride need the same length and offset >= 0");
    const auto elem = static_cast<std::int64_t>(lucid::dtype_size(base->dtype()));
    lucid::Shape view_shape(shape.begin(), shape.end());
    lucid::Stride byte_stride;
    byte_stride.reserve(stride.size());
    // Bounds are the buffer's: ``base`` may itself start partway in.
    std::int64_t last = static_cast<std::int64_t>(base->storage_offset()) + offset * elem;
    bool empty = false;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0 || stride[i] < 0)
            throw std::invalid_argument("_make_view: negative shape or stride");
        empty = empty || shape[i] == 0;
        byte_stride.push_back(stride[i] * elem);
        last += (shape[i] - 1) * stride[i] * elem;
    }
    const auto size = static_cast<std::int64_t>(lucid::storage_nbytes(base->raw_storage()));
    if (!empty && last + elem > size)
        throw std::invalid_argument("_make_view: the view reaches past the end of the storage");
    return traced_derive("_make_view", base,
                         lucid::TensorImpl::make_view(base, std::move(view_shape),
                                                      std::move(byte_stride),
                                                      static_cast<std::size_t>(offset * elem)));
}
}  // namespace

namespace lucid::bindings {

// Registers Device, Dtype, dtype utilities, the no_grad context manager, and
// memory statistics types.  These must be present before TensorImpl and all
// op bindings that reference them.
void register_core(py::module_& m) {
    // Device and Dtype are exported as Python enums.  export_values() makes
    // the members available at module scope (e.g. engine.CPU) as well as on
    // the enum type (engine.Device.CPU).
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    py::enum_<Dtype>(m, "Dtype")
        .value("Bool", Dtype::Bool)
        .value("I8", Dtype::I8)
        .value("I16", Dtype::I16)
        .value("I32", Dtype::I32)
        .value("I64", Dtype::I64)
        .value("F16", Dtype::F16)
        .value("F32", Dtype::F32)
        .value("F64", Dtype::F64)
        .value("C64", Dtype::C64)
        .value("C128", Dtype::C128)
        .value("BF16", Dtype::BF16)
        .export_values();
    m.def("dtype_size", &dtype_size);

    // NoGradGuard implements the Python context-manager protocol.  The
    // __enter__ method returns a reference to self (return_value_policy::reference)
    // because Python's `with` binds the result but the guard object is already
    // owned by the surrounding scope.
    py::class_<NoGradGuard>(m, "NoGradGuard")
        .def(py::init<>())
        .def(
            "__enter__", [](NoGradGuard& g) -> NoGradGuard& { return g; },
            py::return_value_policy::reference)
        .def("__exit__",
             [](NoGradGuard&, py::object, py::object, py::object) { return py::none(); });

    m.def("grad_enabled", &GradMode::is_enabled);
    m.def("set_grad_enabled", &GradMode::set_enabled);

    // Phantom mode, per thread like grad mode: while it is on, every factory
    // and op on the calling thread returns a phantom tensor — shape, dtype,
    // device and autograd state, no storage, no kernel run.
    m.def("phantom_mode", &PhantomMode::is_enabled,
          "Whether factories and ops on the calling thread make phantom tensors.");
    m.def("set_phantom_mode", &PhantomMode::set_enabled, py::arg("enabled"),
          "Turn phantom mode on or off for the calling thread only.");

    // MemoryStats fields are read-only; the Python layer cannot mutate them.
    py::class_<MemoryStats>(m, "MemoryStats")
        .def_readonly("current_bytes", &MemoryStats::current_bytes)
        .def_readonly("peak_bytes", &MemoryStats::peak_bytes)
        .def_readonly("alloc_count", &MemoryStats::alloc_count)
        .def_readonly("free_count", &MemoryStats::free_count)
        .def("__repr__", [](const MemoryStats& s) {
            return "MemoryStats(current_bytes=" + std::to_string(s.current_bytes) +
                   ", peak_bytes=" + std::to_string(s.peak_bytes) +
                   ", alloc_count=" + std::to_string(s.alloc_count) +
                   ", free_count=" + std::to_string(s.free_count) + ")";
        });
    m.def("memory_stats", &MemoryTracker::get_stats, py::arg("device"));
    m.def("reset_peak_memory_stats", &MemoryTracker::reset_peak, py::arg("device"));
    // Host reads of evaluated engine arrays — each one a wait on the lazy
    // graph.  Monotonic; take differences.
    m.def("host_sync_count", &MemoryTracker::host_sync_count);
}

// Registers TensorImpl as the Python-visible `TensorImpl` type and the
// standalone to_shared_storage() utility.
void register_tensor_impl(py::module_& m) {
    // TensorImpl is the sole tensor representation exposed to Python.  It is
    // held as shared_ptr<TensorImpl> on both the C++ and Python sides.  The
    // constructor delegates to TensorImpl::from_numpy, which copies the numpy
    // array data into a managed Storage buffer.
    py::class_<TensorImpl, std::shared_ptr<TensorImpl>>(
        m, "TensorImpl", py::custom_type_setup(enable_tensor_autograd_gc))
        .def(py::init([](py::array arr, Device device, bool requires_grad) {
                 auto out = TensorImpl::from_numpy(std::move(arr), device, requires_grad);
                 traced_factory(out);
                 return out;
             }),
             py::arg("data"), py::arg("device") = Device::CPU, py::arg("requires_grad") = false)
        // Shape and stride are returned as Python lists via py::cast.
        .def_property_readonly("shape", [](const TensorImpl& t) { return py::cast(t.shape()); })
        .def_property_readonly("stride", [](const TensorImpl& t) { return py::cast(t.stride()); })
        .def_property_readonly("dtype", [](const TensorImpl& t) { return t.dtype(); })
        .def_property_readonly("device", [](const TensorImpl& t) { return t.device(); })
        .def_property_readonly("requires_grad",
                               [](const TensorImpl& t) { return t.requires_grad(); })
        .def_property_readonly(
            "grad_fn", [](const TensorImpl& t) -> std::shared_ptr<Node> { return t.grad_fn(); },
            "The backward function that produced this tensor, or None "
            "for leaf tensors.")
        .def_property_readonly("retains_grad", [](const TensorImpl& t) { return t.retains_grad(); })
        .def(
            "retain_grad_", [](TensorImpl& t) { lucid::retain_grad(t.shared_from_this()); },
            "Mark this tensor to retain its gradient during backward, even if "
            "it is not a leaf tensor.  What is kept is everything that reaches "
            "the tensor's producer slot, after the tensor's hooks.")
        .def_property_readonly("is_leaf", [](const TensorImpl& t) { return t.is_leaf(); })
        // version is a mutation counter; the autograd engine reads it to
        // detect in-place modifications to saved tensors.
        .def_property_readonly("version", [](const TensorImpl& t) { return t.version(); })
        .def_property_readonly(
            "is_metal_shared",
            [](const TensorImpl& t) -> bool { return t.metal_shared() != nullptr; },
            "True when the tensor's storage is a MTLResourceStorageModeShared "
            "allocation — simultaneously accessible from CPU and GPU without a "
            "memcpy.  Created by make_shared_tensor() or to_shared_storage().")
        .def_property_readonly(
            "is_phantom", [](const TensorImpl& t) { return t.is_phantom(); },
            "True for a phantom tensor: a shape, dtype, device and autograd state "
            "with no storage, made while phantom mode is on.  Reading its values raises.")
        .def("numel", &TensorImpl::numel)
        .def("nbytes", &TensorImpl::nbytes)
        .def("is_contiguous", &TensorImpl::is_contiguous)
        .def("is_dense", &TensorImpl::is_dense,
             "Contiguous, at the first byte of its storage, and covering all of it.")
        .def("storage_offset_bytes", &TensorImpl::allocation_offset,
             "Byte offset of the first element from the start of the storage.")
        .def("storage_nbytes", &TensorImpl::allocation_nbytes,
             "Total byte size of the underlying storage allocation.")
        .def("storage_data_ptr", &TensorImpl::allocation_base,
             "Address of the first byte of the underlying storage allocation.")
        .def("family_id", &TensorImpl::family_id,
             "Non-zero identity shared by members of the same view family, or 0.")
        .def("data_ptr", &data_address,
             "Address of the first element: the storage base plus the byte offset.")
        .def_static("_make_view", &make_view_from_elements, py::arg("base"), py::arg("shape"),
                    py::arg("stride"), py::arg("offset"),
                    "Testing hook: a view of base given in elements (shape, stride, offset).")
        // data_as_python / grad_as_python return the underlying buffer as a
        // numpy array view without copying; the TensorImpl must outlive the
        // returned array (keep-alive is handled by pybind11's default policy
        // for member functions on shared_ptr holders).
        .def("data_as_python",
             [](const std::shared_ptr<TensorImpl>& self) {
                 note_host_read(self);
                 evaluate_without_gil(*self);
                 return self->data_as_python();
             })
        .def("grad_as_python", &TensorImpl::grad_as_python)
        // NumPy-free interop — used by the Python serialization and repr
        // layers so ``import lucid`` works without numpy installed.
        .def(
            "to_bytes",
            [](const std::shared_ptr<TensorImpl>& self) {
                // pickle, deepcopy and lucid.save read the values through here.
                note_host_read(self);
                evaluate_without_gil(*self);
                return self->to_bytes();
            },
            "Return the tensor data as a contiguous bytes blob "
            "(row-major).  GPU tensors are downloaded to CPU first.")
        .def_static(
            "from_bytes",
            [](py::bytes data, const std::vector<std::int64_t>& shape, Dtype dtype, Device device,
               bool requires_grad) {
                auto out =
                    TensorImpl::from_bytes(std::move(data), shape, dtype, device, requires_grad);
                traced_factory(out);
                return out;
            },
            py::arg("data"), py::arg("shape"), py::arg("dtype"), py::arg("device") = Device::CPU,
            py::arg("requires_grad") = false,
            "Reconstruct a TensorImpl from a raw bytes blob + "
            "metadata.  ``len(data)`` must equal "
            "``prod(shape) * dtype_size(dtype)``.")
        .def(
            "transfer_to_device",
            [](const std::shared_ptr<TensorImpl>& self, Device target,
               bool requires_grad) -> std::shared_ptr<TensorImpl> {
                return traced_derive("to_device", self,
                                     self->transfer_to_device(target, requires_grad));
            },
            py::arg("target"), py::arg("requires_grad"),
            "Cross-device copy (CPU↔GPU) without going through numpy.  "
            "Used by ``Tensor.to(device=...)`` so the .to() call stays "
            "numpy-free for ``import lucid`` users who didn't install the "
            "``[numpy]`` extra.  SharedStorage tensors should use "
            "``transfer_storage`` instead — that path is zero-copy relabel.")
        .def(
            "tolist",
            [](const std::shared_ptr<TensorImpl>& self) {
                note_host_read(self);
                evaluate_without_gil(*self);
                return self->tolist();
            },
            "Convert the tensor to a nested Python list (or scalar for 0-d), "
            "covering every supported dtype including F16 and C64.  Mirrors "
            "``item()``'s dtype dispatch but walks the full shape.  Used by "
            "``Tensor.tolist()`` so the call stays numpy-free across all "
            "current Lucid dtypes — no struct.unpack-based fallback.")
        .def("to_string", &TensorImpl::to_string, py::arg("precision") = 4,
             py::arg("threshold") = 1000, py::arg("edgeitems") = 3,
             "Render the tensor's data as a human-readable string.  "
             "Used by ``__repr__`` to avoid going through numpy.")
        .def(
            "grad_to_tensor",
            [](const TensorImpl& t) -> std::shared_ptr<TensorImpl> { return t.grad_to_tensor(); },
            "Return the accumulated gradient as a TensorImpl reading the "
            "gradient's own buffer (no copy).  On the CPU the same TensorImpl "
            "comes back for as long as it lives, and an in-place op on it "
            "writes into the gradient.  The graph-mode gradient is returned "
            "as is when ``backward(create_graph=True)`` set one.  Returns "
            "None when no gradient has been accumulated.  Backs "
            "``Tensor.grad``.")
        .def(
            "item",
            [](const std::shared_ptr<TensorImpl>& self) {
                note_host_read(self);
                evaluate_without_gil(*self);
                return self->item();
            },
            "Extract a single-element tensor's value as a Python scalar "
            "(int / float / bool / complex).  Throws when numel() != 1.")
        .def(
            "grad_as_impl",
            [](const TensorImpl& t) -> std::shared_ptr<TensorImpl> { return t.grad_as_impl(); },
            "Return the gradient as a TensorImpl (set when backward was called with "
            "create_graph=True). Returns None if no graph-mode gradient is available.")
        .def("copy_from", &TensorImpl::copy_from)
        .def(
            "assign_from",
            [](TensorImpl& self, const TensorImpl& other, const char* name) {
                self.assign_from(other, name);
                // The write gives ``self`` the source's place in the graph.
                // A tensor that asked to keep its gradient keeps the one of
                // its new place, as after an engine in-place op
                // (inplace::adopt_graph_position) — left on the old slot,
                // ``.grad`` stayed None.  The core layer cannot reach the
                // hook slots, so it happens here.
                if (self.retains_grad() && !self.is_leaf())
                    lucid::retain_grad(self.shared_from_this());
            },
            py::arg("other"), py::arg("name"))
        .def("zero_grad", &TensorImpl::zero_grad)
        .def(
            "eval", [](TensorImpl& self) { evaluate_without_gil(self); },
            "Force evaluation of this tensor's lazy MLX graph.\n"
            "GPU tensors: calls mlx::core::eval() on the underlying array.\n"
            "CPU tensors: no-op.")
        .def(
            "clone_with_grad",
            [](const std::shared_ptr<TensorImpl>& self,
               bool requires_grad) -> std::shared_ptr<TensorImpl> {
                // A second TensorImpl over the same bytes, at the same place in
                // them, with its own autograd flags and no grad_fn — how a
                // factory or ``Parameter`` gets a tensor of its own without a
                // copy.  It joins self's view family, so an in-place write
                // through either reaches the other, which is what ``.data``
                // promises; and it keeps self's offset and strides instead of
                // resetting them.  To change an existing tensor's flag, use
                // ``_set_requires_grad``: the alias is a different tensor.
                auto alias = TensorImpl::make_view(self, self->shape(), self->stride(), 0);
                if (requires_grad)
                    alias->set_requires_grad(true);
                // Traced as ``detach``: the same values, a new leaf.
                return traced_derive("detach", self, std::move(alias));
            },
            py::arg("requires_grad"),
            "Return a new TensorImpl sharing the same storage but with a different "
            "requires_grad flag.  No data is copied.")
        .def(
            "_set_requires_grad",
            [](TensorImpl& self, bool requires_grad) {
                // The flag flips on this TensorImpl itself.  Swapping in a
                // ``clone_with_grad`` alias instead left every holder of the
                // old one behind: an engine optimizer kept stepping a tensor
                // that no longer received gradients, and the gradient already
                // accumulated stayed on the old one too.  Turning the flag
                // off keeps the autograd metadata, so ``.grad`` survives.
                // Private: ``Tensor.requires_grad`` / ``requires_grad_`` are
                // the entry points, and they hold the leaf rules.
                self.set_requires_grad(requires_grad);
            },
            py::arg("requires_grad"),
            "Set this tensor's requires_grad flag in place.  The TensorImpl, its "
            "storage and its accumulated gradient are kept.  Meant for leaf "
            "tensors: clearing the flag on a tensor with a grad_fn does not "
            "detach it, so ``Tensor.requires_grad_`` refuses that case before "
            "calling this.")
        .def(
            "data_alias",
            [](const std::shared_ptr<TensorImpl>& self) -> std::shared_ptr<TensorImpl> {
                // What ``.data`` hands out: a member of self's view family that
                // autograd does not follow, so a write through it is untracked.
                // Only ``.data`` is marked — clone_with_grad also makes the
                // tensor a factory returns, which must stay an ordinary member.
                auto alias = TensorImpl::make_view(self, self->shape(), self->stride(), 0);
                alias->set_detached_alias(true);
                // Traced as ``detach``: the value, cut from the gradient.
                // Untraced, the alias reached the next op as a fresh feed and
                // was pinned at its trace-time value — every compiled call
                // replayed the first call's detached tensor, so a straight-
                // through estimator (VQ-VAE, hard Gumbel) returned stale output.
                return traced_derive("detach", self, std::move(alias));
            },
            "Return the alias ``.data`` hands out: the same storage, no autograd "
            "state, and writes through it untracked.")
        .def("is_aliased", &TensorImpl::is_aliased,
             "Whether another live tensor reads this one's buffer as a view of it.")
        .def("set_grad", &set_gradient, py::arg("grad"),
             "Replace this tensor's gradient with grad's values.  grad must have "
             "this tensor's dtype, device and shape, and may not be this tensor "
             "itself; anything else raises and leaves the gradient as it was.  "
             "Backs ``Tensor.grad = grad``.");

    // to_shared_storage re-wraps the tensor's backing buffer as a Metal
    // MTLResourceStorageModeShared allocation.  The resulting tensor occupies
    // the same physical DRAM page as the GPU — use it with _run_metal_kernel
    // for zero-copy custom kernel dispatch.
    m.def("to_shared_storage", &to_shared_storage_impl, py::arg("tensor"),
          "Convert a tensor's storage to MTLResourceStorageModeShared.\n"
          "The returned tensor shares physical DRAM with the GPU — "
          "pass it to _run_metal_kernel for zero-copy custom Metal kernels.");

    // transfer_storage(tensor, device) — zero-copy device relabeling for
    // SharedStorage tensors.  Unlike to_shared_storage(), this never copies
    // data: it either wraps the Metal buffer as an MLX external array (GPU
    // target) or aliases cpu_ptr as a CpuStorage view (CPU target).
    // Raises if the source tensor is not already in SharedStorage.
    m.def("transfer_storage", &transfer_storage_impl, py::arg("tensor"), py::arg("device"),
          "Zero-copy device relabeling for SharedStorage tensors.\n\n"
          "GPU target: wraps the Metal buffer as an MLX external array — no memcpy.\n"
          "CPU target: aliases cpu_ptr as a CpuStorage view — no memcpy.\n"
          "Raises ValueError if the source is not in SharedStorage.");

    // make_shared_tensor(shape, dtype, requires_grad) — allocate a
    // zero-filled tensor directly in MTLResourceStorageModeShared memory.
    // The result starts with device=CPU so normal CPU ops work immediately;
    // call transfer_storage(t, Device::GPU) for a zero-copy GPU view.
    m.def("make_shared_tensor", &make_shared_tensor, py::arg("shape"),
          py::arg("dtype") = Dtype::F32, py::arg("requires_grad") = false,
          "Allocate a zero-filled tensor in MTLResourceStorageModeShared memory.\n\n"
          "The tensor is immediately readable on CPU (device=CPU) and can be\n"
          "transferred to GPU via transfer_storage() with no memcpy.\n"
          "Raises RuntimeError if Metal is not available.");

    // eval_tensors(list[TensorImpl]) — batch-evaluate multiple GPU tensors in
    // one mlx::core::eval() call.  CPU tensors in the list are silently ignored.
    // More efficient than calling .eval() individually because MLX schedules all
    // arrays in a single pass.
    m.def(
        "eval_tensors",
        [](const std::vector<std::shared_ptr<TensorImpl>>& tensors) {
            // Collect all GPU arrays, then evaluate in one batched MLX call.
            // mlx::core::eval(vector<array>) schedules them together —
            // significantly faster than calling arr->eval() individually.
            std::vector<mlx::core::array> arrays = pending_arrays(tensors);
            // The wait is the GPU's, not Python's: release the GIL for it so
            // other Python threads (a server, a data loader) keep running.
            if (!arrays.empty()) {
                py::gil_scoped_release release;
                mlx::core::eval(arrays);
            }
        },
        py::arg("tensors"),
        "Batch-evaluate GPU tensors in one mlx::core::eval() call.\n"
        "CPU tensors are silently ignored.  No-op when all tensors are on CPU.");

    // synchronize_gpu() — run every GPU array still waiting to run, then wait
    // for the device.  ``mlx::core::synchronize`` alone waits only for work
    // already submitted; with Lucid's lazy Metal path nothing is submitted
    // until a value is read, so a timing bracketed by it measured nothing.
    m.def(
        "synchronize_gpu",
        [] {
            py::gil_scoped_release release;
            const std::size_t ran = lucid::gpu::evaluate_pending();
            mlx::core::synchronize();
            return ran;
        },
        "Evaluate every pending (lazy) GPU array, then block until the GPU is idle.\n"
        "Returns how many arrays had pending work.");

    // eval_tensors_async(list[TensorImpl]) — schedule batched evaluation on
    // the GPU stream **without blocking** the CPU thread.  The lazy MLX
    // expression graph is still finalised (so the parent activation chain is
    // released and memory can be reclaimed), but the CPU does not wait for
    // the kernel to complete.  Used by the BN / InstanceNorm running-stats
    // update — 3.2.1 introduced ``eval_tensors`` there to break the lazy
    // chain that prevented `loss.item()` from triggering eval (since running
    // stats are not on the loss path), which caused a per-iter memory leak;
    // but with 16 BN modules in ResNet-18, the per-BN sync cost ~6 ms /
    // forward.  ``async_eval`` keeps the leak fix while dropping the sync,
    // because the parent reference release happens at *schedule* time, not
    // at *completion* time.
    m.def(
        "eval_tensors_async",
        [](const std::vector<std::shared_ptr<TensorImpl>>& tensors) {
            std::vector<mlx::core::array> arrays = pending_arrays(tensors);
            if (!arrays.empty())
                mlx::core::async_eval(arrays);
        },
        py::arg("tensors"),
        "Batch-schedule GPU evaluation without blocking the CPU thread.\n"
        "CPU tensors are silently ignored.  Finalises the MLX lazy graph\n"
        "(releases parent activation references) but returns immediately.");

    // eval_gpu(impl) — evaluate a single GPU TensorImpl in-place.
    // Faster than eval_tensors([impl]) for the single-tensor case because it
    // avoids Python list creation and the pybind11 vector conversion overhead
    // (~25 µs on hot paths).  Used by the benchmark's timing loop and by any
    // code that needs to force GPU computation for a single intermediate tensor.
    m.def(
        "eval_gpu",
        [](const std::shared_ptr<TensorImpl>& t) {
            if (t)
                evaluate_without_gil(*t);
        },
        py::arg("tensor"),
        "Force evaluation of a single GPU tensor (no-op for CPU tensors).\n"
        "Faster than eval_tensors([t]) for the single-tensor hot path.");
}

}  // namespace lucid::bindings
