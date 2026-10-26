// lucid/_C/core/TensorImpl.cpp
//
// Implementations of TensorImpl methods that are too large to inline.
// Key responsibilities handled here:
//
//   from_numpy  — converts a Python NumPy array to a TensorImpl, optionally
//                 uploading to GPU via the MLX bridge.
//   data/grad_as_python — materialises tensor data as a zero-copy (CPU) or
//                 synchronously-downloaded (GPU) NumPy array for Python.
//   copy_from   — typed in-place data copy with full variant dispatch.
//   make_view   — creates an aliasing TensorImpl with different geometry.

#include "TensorImpl.h"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <mlx/ops.h>

#include "../backend/gpu/MlxBridge.h"
#include "../backend/gpu/SharedBufferWrites.h"
#include "../compile/Tracer.h"
#include "Allocator.h"
#include "Error.h"
#include "ErrorBuilder.h"
#include "GpuStrided.h"
#include "GradMode.h"
#include "MemoryStats.h"
#include "PhantomMode.h"
#include "StridedCopy.h"

namespace lucid {

namespace {

// Overload-set helper for std::visit.  Deduces operator() from the provided
// lambdas using the standard "overloaded" pattern (C++17).  Each std::visit
// call below passes a set of lambdas, one per Storage variant.
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// Converts a NumPy dtype kind character and item size to the corresponding
// Lucid Dtype.  Unsigned integer kinds ('u') are rejected with a DtypeMismatch
// because the engine only supports signed integer arithmetic.
Dtype np_dtype_to_lucid(const py::dtype& d) {
    const char k = d.kind();
    const auto sz = static_cast<int>(d.itemsize());
    switch (k) {
    case 'b':
        return Dtype::Bool;
    case 'i':
        switch (sz) {
        case 1:
            return Dtype::I8;
        case 2:
            return Dtype::I16;
        case 4:
            return Dtype::I32;
        case 8:
            return Dtype::I64;
        }
        break;
    case 'u':
        throw DtypeMismatch("signed integer or float numpy dtype",
                            std::string("uint") + std::to_string(sz * 8),
                            "from_numpy: unsigned dtypes are not supported "
                            "(cast to int / float explicitly)");
    case 'f':
        switch (sz) {
        case 2:
            return Dtype::F16;
        case 4:
            return Dtype::F32;
        case 8:
            return Dtype::F64;
        }
        break;
    case 'c':
        if (sz == 8)
            return Dtype::C64;
        if (sz == 16)
            return Dtype::C128;
        break;
    default:
        break;
    }
    throw DtypeMismatch("supported numpy dtype",
                        std::string("kind='") + k + "',itemsize=" + std::to_string(sz),
                        "from_numpy");
}

// Maps a Lucid Dtype to the corresponding NumPy dtype descriptor.
py::dtype lucid_dtype_to_np(Dtype dt) {
    switch (dt) {
    case Dtype::Bool:
        return py::dtype("bool");
    case Dtype::I8:
        return py::dtype("int8");
    case Dtype::I16:
        return py::dtype("int16");
    case Dtype::I32:
        return py::dtype("int32");
    case Dtype::I64:
        return py::dtype("int64");
    case Dtype::F16:
        return py::dtype("float16");
    case Dtype::BF16:
        // NumPy has no bfloat16, so the bridge widens.  float32 rather
        // than float16 because the two 16-bit formats do not nest: a bf16
        // value can be larger than float16's maximum, and narrowing to it
        // would turn a perfectly good number into an infinity on the way
        // out of the framework.
        return py::dtype("float32");
    case Dtype::F32:
        return py::dtype("float32");
    case Dtype::F64:
        return py::dtype("float64");
    case Dtype::C64:
        return py::dtype("complex64");
    case Dtype::C128:
        return py::dtype("complex128");
    }
    ErrorBuilder("lucid_dtype_to_np").fail("unknown Dtype");
}

// Creates a zero-copy NumPy array view backed by a CpuStorage buffer.
// A py::capsule is used as the NumPy "base" object so that the underlying
// shared_ptr keeps the allocation alive for the entire lifetime of the
// returned array — even if the originating TensorImpl is destroyed first.
//
// py::array requires a raw void* for the data pointer but needs to know who
// owns the memory; passing a capsule as the "base" object transfers that
// responsibility to the ref-counted shared_ptr copy inside the capsule.
// Defined further down, in this file's second anonymous namespace.
std::vector<std::byte> contig_snapshot_cpu(const CpuStorage& s,
                                           const Shape& shape,
                                           const Stride& stride,
                                           std::size_t storage_offset);
void scatter_contig_to_strided(const std::byte* packed,
                               CpuStorage& dst,
                               const Shape& shape,
                               const Stride& stride,
                               std::size_t storage_offset);

py::object
make_numpy_view(const CpuStorage& s, const Shape& shape, const Stride& stride, std::size_t offset) {
    // Heap-allocate a copy of the shared_ptr so the capsule's destructor can
    // call delete on the correct type.  The capsule holds a void* internally,
    // so we need a stable heap address for the shared_ptr object itself.
    auto* keepalive = new std::shared_ptr<std::byte[]>(s.ptr);
    py::capsule owner(keepalive,
                      [](void* p) { delete static_cast<std::shared_ptr<std::byte[]>*>(p); });

    std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
    std::vector<py::ssize_t> py_stride(stride.begin(), stride.end());

    // bfloat16 cannot be a view.  Every other dtype hands NumPy the buffer
    // as it stands, but NumPy has no bfloat16, so ``lucid_dtype_to_np``
    // answers float32 — and a view would then describe a 2-byte buffer as
    // 4-byte elements and read straight across the boundaries.  The first
    // version of this did exactly that and returned values from the wrong
    // positions: ``1e30`` came back as ``4e-41``.  Convert instead — only the
    // tensor's own elements, in order, since the buffer may hold more.
    if (s.dtype == Dtype::BF16) {
        const auto packed = contig_snapshot_cpu(s, shape, stride, offset);
        const std::size_t n = packed.size() / sizeof(std::uint16_t);
        py::array_t<float> out(py_shape);
        const auto* bits = reinterpret_cast<const std::uint16_t*>(packed.data());
        auto* dst = static_cast<float*>(out.request().ptr);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t widened = static_cast<std::uint32_t>(bits[i]) << 16;
            std::memcpy(&dst[i], &widened, sizeof(float));
        }
        return out;
    }

    // The first element sits ``offset`` bytes into the buffer; handing NumPy
    // the buffer's first byte read a view at an offset from the wrong place.
    return py::array(lucid_dtype_to_np(s.dtype), py_shape, py_stride,
                     static_cast<const void*>(s.ptr.get() + offset), owner);
}

}  // namespace

// Stride is computed after all meta fields are set because contiguous_stride()
// needs both shape and elem_size; element size depends on dtype.
TensorImpl::TensorImpl(Storage storage, Shape shape, Dtype dtype, Device device, bool requires_grad)
    : storage_(std::move(storage)) {
    meta_.shape = std::move(shape);
    meta_.dtype = dtype;
    meta_.device = device;
    // Derive the C-contiguous stride from the final shape and element size.
    meta_.stride = contiguous_stride(meta_.shape, dtype_size(dtype));
    if (requires_grad) {
        // emplace() constructs AutogradMeta in-place inside the optional,
        // avoiding an extra heap allocation versus make_optional.
        autograd_.emplace();
        autograd_->requires_grad = true;
    }
    // Ops read CpuStorage or GpuStorage and nothing else — the CPU kernels
    // alone reach for ``std::get<CpuStorage>`` in over four hundred places,
    // and a shared buffer arriving as itself raised ``bad_variant_access``
    // from the first ``+``.  Adopt it as the view its device label calls for
    // and keep the descriptor aside, for relabelling and custom kernels.
    if (auto* sh = std::get_if<SharedStorage>(&storage_); sh && sh->cpu_ptr && sh->nbytes > 0) {
        SharedStorage desc = std::move(*sh);
        gpu::pin_shared_storage(desc, meta_.shape);
        if (meta_.device == Device::GPU)
            storage_ = Storage{gpu::shared_storage_to_gpu(desc, meta_.shape)};
        else
            storage_ = Storage{desc.cpu_view()};
        shared_ = std::move(desc);
    }
    // The phantom backend hands out storages without a size — it is told the
    // shape only as some methods' arguments — and the first tensor to adopt
    // one gives it its own.  A view adopts a sized one and keeps its size.
    if (auto* ph = std::get_if<PhantomStorage>(&storage_); ph && ph->nbytes == 0) {
        ph->nbytes = nbytes();
        ph->dtype = dtype;
        ph->device = device;
    }
}

const SharedStorage* TensorImpl::metal_shared() const {
    // Only a tensor that *is* the buffer — whole, contiguous, from its first
    // byte, in its dtype — can be relabelled without a copy.  Anything else
    // still carrying the descriptor (a view, or a tensor whose storage an op
    // replaced) reads through it at most, and answers no.
    if (!shared_ || offset_ != 0 || meta_.dtype != shared_->dtype || nbytes() != shared_->nbytes ||
        !is_contiguous())
        return nullptr;
    // A Metal family member is the buffer while the family's root still is.
    const Storage& own = raw_storage();
    if (const auto* cpu = std::get_if<CpuStorage>(&own))
        return cpu->ptr.get() == shared_->cpu_ptr ? &*shared_ : nullptr;
    if (const auto* gpu_st = std::get_if<GpuStorage>(&own)) {
        const auto& pin = shared_->gpu_alias;
        return gpu_st->arr && pin && gpu_st->arr->id() == pin->id() ? &*shared_ : nullptr;
    }
    return nullptr;
}

namespace {

// A CPU tensor carrying a shared buffer's descriptor writes into the buffer
// itself: the GPU work built over it before the write has to read first.
void settle_before_host_write(const std::optional<SharedStorage>& shared, const Storage& own) {
    if (shared && storage_is_cpu(own))
        gpu::settle_buffer_readers(*shared);
}

// Lays ``arr``'s values into the shared buffer ``sh``, which arrays aliasing
// it then read.
void copy_into_shared(const SharedStorage& sh, const ::mlx::core::array& arr) {
    // data() ignores strides, so lay the result out first — run in the same
    // pass as the GPU work that reads the bytes it replaces.
    ::mlx::core::array packed = ::mlx::core::contiguous(arr);
    gpu::settle_buffer_readers(sh, {packed});
    MemoryTracker::track_host_sync(packed.nbytes());
    auto* dst = static_cast<std::byte*>(sh.cpu_ptr);
    const auto* from = reinterpret_cast<const std::byte*>(packed.data<std::uint8_t>());
    if (from != dst)
        std::memcpy(dst, from, sh.nbytes);
}

}  // namespace

bool TensorImpl::write_into_shared(const TensorImpl& src) {
    const SharedStorage* sh = metal_shared();
    if (!sh || src.dtype() != meta_.dtype || src.shape() != meta_.shape ||
        src.device() != meta_.device)
        return false;
    // GPU work built before this write, run or not, reads the old bytes:
    // each branch settles it before laying the new ones in.
    auto* dst = static_cast<std::byte*>(sh->cpu_ptr);
    return std::visit(overloaded{
                          [&](const CpuStorage& c) {
                              if (!c.ptr || src.storage_offset() != 0 || !src.is_contiguous())
                                  return false;
                              gpu::settle_buffer_readers(*sh);
                              if (c.ptr.get() != dst)
                                  std::memmove(dst, c.ptr.get(), sh->nbytes);
                              return true;
                          },
                          [&](const GpuStorage& g) {
                              if (!g.arr)
                                  return false;
                              copy_into_shared(*sh, *g.arr);
                              return true;
                          },
                          [&](const SharedStorage& s) {
                              gpu::settle_buffer_readers(*sh);
                              if (s.cpu_ptr != sh->cpu_ptr)
                                  std::memcpy(dst, s.cpu_ptr, sh->nbytes);
                              return true;
                          },
                          // No values to lay in; the callers refuse a phantom
                          // source before a real target gets here.
                          [&](const PhantomStorage&) { return false; },
                      },
                      src.storage());
}

namespace {

// An in-place op computes its result out of place and splices it into the
// target; an active compile trace has to learn that the target now holds the
// result, or every later read of the target resolves to its old value.
void note_inplace_write(TensorImpl& dst, const TensorImpl& src) {
    auto* trc = compile::current_tracer();
    if (trc == nullptr)
        return;
    auto d = dst.weak_from_this().lock();
    auto from = std::const_pointer_cast<TensorImpl>(src.weak_from_this().lock());
    if (d && from)
        trc->on_inplace_write(d, from);
}

// Two of the target's elements are one byte: a value-dependent write has no
// single value to leave there.  The reference refuses only stride-0 internal
// overlap (expanded views) and lets overlapping unfold/as_strided views
// sequentially compound in storage order; Lucid computes in-place results out
// of place and scatters them back, so it deliberately refuses any view whose
// own elements overlap via strided::overlaps_itself (LCD-537).  ``fill_`` is
// the exception, as in the reference (``zero_`` is ``fill_(0)`` on this side):
// it leaves one value in every element, so the elements that share a byte
// agree.
void refuse_ambiguous_write(const TensorImpl& t, const char* name) {
    if (std::strcmp(name, "fill_") == 0 || t.is_contiguous() ||
        !strided::overlaps_itself(t.shape(), t.stride()))
        return;
    ErrorBuilder(name).fail(
        "an in-place write into a tensor whose elements overlap (an "
        "expanded view, or unfold windows that overlap) is ambiguous — clone() first");
}

// The tensor ``.grad`` reads, and why its pointers are not the slot's.
//
// A gradient lives in its tensor's AutogradMeta as a bare Storage — a slot,
// not a tensor — and grad_to_tensor wraps it in a TensorImpl over the same
// bytes.  That tensor used to hold the slot's own pointer, so the slot was
// one more holder of its buffer, which an in-place op cannot tell from the
// holders it protects (a node's saved input, a NumPy array):
// ``p.grad.mul_(s)`` was refused as storage shared with a view, while
// ``p.grad.neg_()`` took a new buffer of its own and left the gradient as it
// was.  Now each side holds the buffer through a pointer that owns a record
// of the other, which ``std::get_deleter`` finds from the pointer alone:
//
// * the slot's pointer owns a GradSlot — the buffer, and the tensor
//   ``.grad`` returned for it, handed out again while it lives, as the
//   reference's ``.grad`` is one tensor (a ``g = p.grad`` kept alive is not
//   then a second reader of ``p.grad.mul_(s)``'s buffer);
// * that tensor's pointer owns a GradSlotRef — the slot's pointer, and whose
//   slot it is — and so does every view of it, since a view copies or
//   aliases its base's pointer.
//
// The holders of the tensor's pointer are then the tensor's own, counted
// apart from the slot's.
struct GradSlot {
    std::shared_ptr<std::byte[]> buffer;
    std::weak_ptr<TensorImpl> handle;
    void operator()(void*) noexcept { buffer.reset(); }
};

struct GradSlotRef {
    std::shared_ptr<std::byte[]> slot;
    std::weak_ptr<const TensorImpl> owner;
    void operator()(void*) noexcept { slot.reset(); }
};

struct GpuGradSlot {
    std::shared_ptr<::mlx::core::array> arr;
    std::shared_ptr<TensorImpl> source;
    std::weak_ptr<TensorImpl> handle;
    std::shared_ptr<ViewFamily> family;
    void operator()(void*) noexcept {
        arr.reset();
        source.reset();
        family.reset();
    }
};

// Whether ``a`` and ``b`` share ownership — one control block, whatever
// address each points at.
bool same_owner(const std::shared_ptr<std::byte[]>& a, const std::shared_ptr<std::byte[]>& b) {
    return !a.owner_before(b) && !b.owner_before(a);
}

// Where an in-place write to a tensor holding ``storage`` belongs, when that
// storage reads a gradient slot.
enum class SlotWrite {
    kNone,     // not a gradient's buffer (any more): the ordinary rules apply
    kInto,     // into the slot's buffer, which the gradient reads
    kRefused,  // something besides the slot reads the buffer as well
};

SlotWrite gradient_slot_write(const Storage& storage, long members) {
    const auto* cpu = std::get_if<CpuStorage>(&storage);
    if (cpu == nullptr || !cpu->ptr)
        return SlotWrite::kNone;
    const auto* ref = std::get_deleter<GradSlotRef>(cpu->ptr);
    if (ref == nullptr || !ref->slot)
        return SlotWrite::kNone;
    // Read in place: a copy of the slot would hold the buffer while it is
    // being counted.
    const auto owner = ref->owner.lock();
    const CpuStorage* slot = nullptr;
    if (owner && owner->grad_storage())
        slot = std::get_if<CpuStorage>(&*owner->grad_storage());
    // The tensor has let go of that gradient — ``p.grad = None``, another one
    // assigned — so this is an ordinary tensor now.
    if (slot == nullptr || !same_owner(slot->ptr, ref->slot))
        return SlotWrite::kNone;
    // The slot and this pointer's GradSlotRef hold the slot's pointer, its
    // GradSlot alone holds the buffer, and the members of this tensor's view
    // family hold this tensor's pointer.  Anything more reads the same bytes
    // — a second tensor read from ``.grad``, a node that saved one for
    // backward, a NumPy array, a tensor the gradient was taken from — and the
    // write would change them behind its back.
    const auto* owned = std::get_deleter<GradSlot>(ref->slot);
    if (owned == nullptr || ref->slot.use_count() > 2 || cpu->ptr.use_count() > members)
        return SlotWrite::kRefused;
    return SlotWrite::kInto;
}

[[noreturn]] void refuse_shared_gradient_write(const char* name) {
    ErrorBuilder(name).fail(
        "an in-place write to a gradient is not supported while something else also reads "
        "its storage: another tensor read from .grad, another gradient, a NumPy array, or a "
        "tensor saved for backward — which the op saves itself when an operand requires grad "
        "(run it under no_grad, or clone() first)");
}

// A leaf's values are where its gradient accumulates.  A write through one
// of its views moves them as surely as a write to the leaf itself, which
// refuse_on_leaf refuses while autograd records (views_grad_leaf says which
// tensors are such views).  The caller checks the writer itself — a
// detached alias (``.data``) writes untracked — and the grad mode.
void refuse_write_to_leaf_view(const TensorImpl& t, const char* name) {
    if (t.views_grad_leaf())
        ErrorBuilder(name).fail("a view of a leaf tensor that requires grad cannot be "
                                "modified in place — wrap the call in no_grad, or use "
                                "the out-of-place form");
}

void refuse_custom_function_output_write(const TensorImpl& t, const char* name) {
    if (GradMode::is_enabled() && t.is_custom_function_output()) {
        ErrorBuilder(name).fail(
            "Output of a custom Function is a view and is being modified in place — custom "
            "Function output views cannot be modified in place; clone() first");
    }
}

void init_family_base(const std::shared_ptr<TensorImpl>& base,
                      std::shared_ptr<ViewFamily>& family,
                      std::optional<AutogradMeta>& autograd,
                      const Storage& storage) {
    if (family)
        return;
    family = std::make_shared<ViewFamily>();
    if (autograd && autograd->version > 0) {
        family->version.store(static_cast<std::uint64_t>(autograd->version),
                              std::memory_order_relaxed);
        autograd->version = 0;
    }
    if (autograd && autograd->version_sink)
        family->version_sink = std::move(autograd->version_sink);
    family->nbytes = base->allocation_nbytes();
    family->members.push_back(base);
    if (storage_is_gpu(storage))
        family->root = storage;
}

// Whether a Metal family's root still is the shared buffer ``shared`` its
// base was built over.
bool root_is_shared_buffer(const std::optional<SharedStorage>& shared,
                           const GpuStorage& root,
                           Dtype dtype) {
    return shared && shared->gpu_alias && root.arr && root.arr->id() == shared->gpu_alias->id() &&
           shared->nbytes == root.nbytes && shared->dtype == dtype;
}

// A Metal family member's own storage: empty, since its values are the
// family root's.  Typed, so the variant still says which device it is on.
Storage gpu_placeholder(Dtype dtype) {
    GpuStorage g;
    g.dtype = dtype;
    return Storage{std::move(g)};
}

// Whether a Metal member's cached ``packed`` storage no longer matches its
// family root — either because the root array was replaced or because a
// shared-buffer settle rebound the root's Data onto a snapshot.
bool stale_gpu_pack(const std::optional<Storage>& packed,
                    const std::weak_ptr<::mlx::core::array>& packed_root,
                    const GpuStorage& rg,
                    bool contig) {
    if (packed_root.lock() != rg.arr)
        return true;
    if (!contig)
        return false;
    const auto& pg = std::get<GpuStorage>(*packed);
    return pg.arr && pg.arr->data_shared_ptr() && rg.arr->data_shared_ptr() &&
           pg.arr->data_shared_ptr() != rg.arr->data_shared_ptr();
}

TensorImplPtr trace_gpu_op(compile::Tracer& trc,
                           std::string_view op,
                           const std::vector<TensorImplPtr>& inputs,
                           ::mlx::core::array arr,
                           const Shape& shape,
                           Dtype dt) {
    auto out = std::make_shared<TensorImpl>(Storage{gpu::wrap_mlx_array(std::move(arr), dt)}, shape,
                                            dt, Device::GPU, false);
    trc.on_op_enter(op, Device::GPU, dt, shape);
    if (op == "scatter_set")
        trc.on_op_attr("dim", static_cast<std::int64_t>(0));
    trc.on_op_io(inputs, out);
    return out;
}

void trace_gpu_member_scatter(compile::Tracer& trc,
                              const TensorImplPtr& m,
                              const TensorImplPtr& from,
                              const ::mlx::core::array& root_arr,
                              const ::mlx::core::array& val_arr,
                              const ::mlx::core::array& next,
                              const gpu_strided::Geometry& g,
                              std::int64_t root_numel,
                              Dtype dt) {
    const Shape flat_shape{root_numel};
    const Shape src_flat_shape{g.numel};
    TensorImplPtr flat_m =
        m->shape() == flat_shape
            ? m
            : trace_gpu_op(trc, "reshape", {m}, ::mlx::core::flatten(root_arr), flat_shape, dt);
    TensorImplPtr flat_src =
        from->is_contiguous() ? from
                              : trace_gpu_op(trc, "contiguous", {from},
                                             ::mlx::core::contiguous(val_arr), from->shape(), dt);
    if (flat_src->shape() != src_flat_shape)
        flat_src = trace_gpu_op(
            trc, "reshape", {flat_src},
            ::mlx::core::reshape(::mlx::core::contiguous(val_arr), {static_cast<int>(g.numel)}),
            src_flat_shape, dt);
    ::mlx::core::array pos_arr = gpu_strided::positions(g, root_numel);
    const Dtype idx_dt = pos_arr.dtype() == ::mlx::core::int64 ? Dtype::I64 : Dtype::I32;
    auto idx_impl =
        std::make_shared<TensorImpl>(Storage{gpu::wrap_mlx_array(std::move(pos_arr), idx_dt)},
                                     src_flat_shape, idx_dt, Device::GPU, false);
    TensorImplPtr flat_next =
        trace_gpu_op(trc, "scatter_set", {flat_m, idx_impl, flat_src},
                     ::mlx::core::reshape(next, {static_cast<int>(root_numel)}), flat_shape, dt);
    TensorImplPtr next_m =
        m->shape() == flat_shape
            ? flat_next
            : trace_gpu_op(trc, "reshape", {flat_next},
                           ::mlx::core::reshape(next, gpu_strided::to_mlx(m->shape())), m->shape(),
                           dt);
    trc.on_inplace_write(m, next_m);
}

void trace_gpu_member_write(const TensorImpl& dst,
                            const TensorImpl& src,
                            const ::mlx::core::array& root_arr,
                            const ::mlx::core::array& val_arr,
                            const ::mlx::core::array& next,
                            const gpu_strided::Geometry& g) {
    auto* trc = compile::current_tracer();
    if (!trc || root_arr.dtype() != val_arr.dtype())
        return;
    auto from = std::const_pointer_cast<TensorImpl>(src.weak_from_this().lock());
    if (!from)
        return;
    const auto root_numel = static_cast<std::int64_t>(root_arr.size());
    const Dtype dt = dst.dtype();
    for (const auto& m : dst.live_views()) {
        if (m->dtype() != dt || m->storage_offset() != 0 || !m->is_contiguous() ||
            m->nbytes() != root_arr.nbytes())
            continue;
        if (g.covers(root_numel)) {
            TensorImplPtr next_m =
                m->shape() == from->shape()
                    ? from
                    : trace_gpu_op(*trc, "reshape", {from},
                                   ::mlx::core::reshape(next, gpu_strided::to_mlx(m->shape())),
                                   m->shape(), dt);
            trc->on_inplace_write(m, next_m);
        } else if (g.numel > 0) {
            trace_gpu_member_scatter(*trc, m, from, root_arr, val_arr, next, g, root_numel, dt);
        }
    }
}

bool same_gpu_root_layout(const TensorImpl& h,
                          const ViewFamily& family,
                          const ::mlx::core::array& root_arr) {
    return h.storage_offset() == 0 && h.is_contiguous() && h.nbytes() == family.nbytes &&
           root_arr.dtype() == gpu::to_mlx_dtype(h.dtype()) &&
           root_arr.ndim() == h.shape().size() &&
           std::equal(h.shape().begin(), h.shape().end(), root_arr.shape().begin());
}

bool can_alias_gpu_grad(const TensorImpl& g) {
    return g.is_contiguous() || !strided::overlaps_itself(g.shape(), g.stride());
}

std::weak_ptr<TensorImpl> current_gpu_grad_handle(const std::optional<AutogradMeta>& autograd,
                                                  const std::shared_ptr<TensorImpl>& g) {
    if (!autograd || !autograd->grad.has_value())
        return {};
    const auto* cur = std::get_if<GpuStorage>(&*autograd->grad);
    if (!cur || !cur->arr)
        return {};
    const auto* owned = std::get_deleter<GpuGradSlot>(cur->arr);
    if (owned && owned->handle.lock() == g)
        return g;
    return {};
}

void update_gpu_grad_root(ViewFamily& family,
                          const TensorImpl* geom,
                          const std::shared_ptr<::mlx::core::array>& next_arr) {
    auto& root = std::get<GpuStorage>(*family.root);
    if (!geom || !root.arr || same_gpu_root_layout(*geom, family, *root.arr)) {
        root.arr = next_arr;
        return;
    }
    ::mlx::core::array updated = gpu_strided::write(*root.arr, *next_arr, geom->shape(),
                                                    geom->stride(), geom->storage_offset());
    root.arr = gpu::wrap_mlx_array(std::move(updated), root.dtype).arr;
}

void register_gpu_grad_owner(ViewFamily& family,
                             const TensorImpl* self_ptr,
                             std::weak_ptr<const TensorImpl> self_weak) {
    auto& owners = family.grad_owners;
    owners.erase(std::remove_if(owners.begin(), owners.end(),
                                [self_ptr](const std::weak_ptr<const TensorImpl>& w) {
                                    const auto p = w.lock();
                                    return !p || p.get() == self_ptr;
                                }),
                 owners.end());
    owners.push_back(std::move(self_weak));
}

void rebind_gpu_grad_slot(const std::shared_ptr<ViewFamily>& family,
                          const std::shared_ptr<::mlx::core::array>& root_arr) {
    if (!family || !root_arr)
        return;
    auto& owners = family->grad_owners;
    owners.erase(
        std::remove_if(owners.begin(), owners.end(),
                       [](const std::weak_ptr<const TensorImpl>& w) { return w.expired(); }),
        owners.end());
    for (const auto& w : owners) {
        auto owner = w.lock();
        if (!owner || !owner->grad_storage().has_value())
            continue;
        auto& slot = const_cast<std::optional<Storage>&>(owner->grad_storage());
        auto* slot_gpu = std::get_if<GpuStorage>(&*slot);
        if (!slot_gpu || !slot_gpu->arr)
            continue;
        auto* owned = std::get_deleter<GpuGradSlot>(slot_gpu->arr);
        if (!owned || owned->family != family)
            continue;
        const auto h = owned->handle.lock();
        const TensorImpl* geom = h ? h.get() : owned->source.get();
        std::shared_ptr<::mlx::core::array> slot_arr = root_arr;
        if (geom != nullptr && !same_gpu_root_layout(*geom, *family, *root_arr)) {
            ::mlx::core::array elems =
                gpu_strided::read(*root_arr, geom->shape(), geom->stride(), geom->storage_offset(),
                                  gpu::to_mlx_dtype(geom->dtype()));
            slot_arr = gpu::wrap_mlx_array(std::move(elems), geom->dtype()).arr;
        }
        const std::shared_ptr<void> record(
            nullptr, GpuGradSlot{slot_arr, owned->source, owned->handle, family});
        slot_gpu->arr = std::shared_ptr<::mlx::core::array>(record, slot_arr.get());
    }
}

void check_write_through_target(const TensorImpl& dst, const TensorImpl& src, const char* name) {
    if (src.shape() != dst.shape())
        throw ShapeMismatch(dst.shape(), src.shape(),
                            std::string(name) + " (in-place: shape changed)");
    if (src.dtype() != dst.dtype())
        ErrorBuilder(name).fail("an in-place write cannot change the dtype of a tensor that "
                                "shares storage with a live view — cast first, or use the "
                                "out-of-place form");
}

void check_cpu_write_through_target(const TensorImpl& dst,
                                    const Storage& dst_storage,
                                    const Storage& src_storage,
                                    long members,
                                    const char* name) {
    if (!storage_is_cpu(dst_storage) || !storage_is_cpu(src_storage))
        ErrorBuilder(name).not_implemented("writing through a view is only supported on the CPU");
    refuse_ambiguous_write(dst, name);
    if (gradient_slot_write(dst_storage, members) == SlotWrite::kRefused)
        refuse_shared_gradient_write(name);
    if (GradMode::is_enabled() && !dst.is_detached_alias())
        refuse_write_to_leaf_view(dst, name);
}

void write_cpu_buffer(const TensorImpl& dst_impl,
                      CpuStorage& dst,
                      const TensorImpl& src,
                      const CpuStorage& from,
                      const std::optional<SharedStorage>& shared) {
    // The op handed back this very buffer (a no-op for this dtype): the
    // values are already in place.
    if (from.ptr == dst.ptr && src.is_dense() && dst_impl.is_dense())
        return;
    // The source packed, then laid into this tensor's own elements — at its
    // offset and through its strides, which a dense tensor has trivially.
    const std::size_t n = dst_impl.nbytes();
    std::vector<std::byte> packed;
    const std::byte* bytes = from.ptr.get();
    if (!src.is_dense()) {
        packed = contig_snapshot_cpu(from, src.shape(), src.stride(), src.storage_offset());
        bytes = packed.data();
    }
    settle_before_host_write(shared, dst_impl.raw_storage());
    if (dst_impl.is_dense()) {
        if (n > 0)
            std::memcpy(dst.ptr.get(), bytes, n);
    } else {
        scatter_contig_to_strided(bytes, dst, dst_impl.shape(), dst_impl.stride(),
                                  dst_impl.storage_offset());
    }
}

}  // namespace

bool TensorImpl::write_lands_in_buffer() const {
    // The rule take_storage_from follows once a graph is recorded.
    if (!storage_is_cpu(storage_))
        return false;
    return is_aliased() || !is_dense() || allocation_offset() != 0 ||
           allocation_nbytes() != nbytes() ||
           gradient_slot_write(storage_, family_ ? family_.use_count() : 1) == SlotWrite::kInto;
}

void TensorImpl::take_storage_from(TensorImpl& out, const char* name) {
    refuse_custom_function_output_write(*this, name);
    // Checked here, the one place every in-place op's result lands, and not
    // only in write_through: an expanded view whose base is gone has no live
    // views, so it came down to the swap below and took a dense buffer in
    // silence — ``ones(1, 3).expand(2, 3).add_(1)`` raised nothing.
    refuse_ambiguous_write(*this, name);
    // The phantom write rule (core/PhantomMode.h): a phantom keeps its own
    // storage and the caller (inplace::write_result) moves the version; a
    // real tensor refuses a phantom result here, before any change.
    if (absorbs_phantom_write(storage_, out.raw_storage(), name))
        return;
    // The last Metal member left has no one to share the write with: when it
    // covers the whole root, it takes the result as its own storage below, as
    // any Metal tensor does (a dtype-changing op included).  A strided or
    // offset view whose base was collected keeps its family so its strides,
    // storage_offset, and allocation capacity survive the write.
    if (gpu_member() && !is_aliased() && covers_root())
        leave_family();
    // A tensor read from ``.grad`` writes into its gradient's buffer: a new
    // slot of its own would leave the gradient as it was.
    const SlotWrite slot = gradient_slot_write(storage_, family_ ? family_.use_count() : 1);
    if (slot == SlotWrite::kRefused)
        refuse_shared_gradient_write(name);
    note_inplace_write(*this, out);
    // Write through only when ``out`` recorded no graph: a node that saved
    // the pre-op input shares this buffer and would read the new values back
    // — the ``cos(sin(x))`` failure :file:`ops/utils/InplaceGraph.h` records.
    const bool graph = out.requires_grad() || out.grad_fn();
    if (!graph && write_into_shared(out))
        return;
    // A CPU tensor with live views or a strided layout takes the values into
    // its buffer, graph or no graph: a new slot would leave the views reading
    // the old ones or repack this tensor's strides.  So does a tensor read
    // from ``.grad``.  Either way its snapshot was a copy
    // (inplace::snapshot, by write_lands_in_buffer — the same rule), so no
    // node of this op holds the bytes being replaced; a node that had saved
    // the gradient tensor earlier would hold its pointer, and the write was
    // refused above.
    // A Metal member with live views writes into the family's root, where
    // they read.  Its snapshot shared the root's old value, which no write
    // changes: an MLX array is a value.
    if (gpu_member() || write_lands_in_buffer()) {
        write_through(out, name);
        return;
    }
    if (family_)
        leave_family();
    if (out.gpu_member()) {
        // ``out`` reads a family's root: take its elements, laid out densely.
        storage_ = out.storage();
        meta_.stride = contiguous_stride(out.shape(), dtype_size(out.dtype()));
        offset_ = 0;
        drop_shared();
        return;
    }
    storage_ = std::move(out.storage_);
    // The new buffer is laid out the way ``out`` lays it out — a dense run
    // from its first byte, at ``out``'s element size.  Keeping this
    // tensor's own stride and offset misread it whenever the two differ:
    // an in-place op that turned int64 into float32 (``exp_`` on integers)
    // kept 8-byte strides over 4-byte floats and read every other element.
    meta_.stride = out.meta_.stride;
    offset_ = out.offset_;
    drop_shared();
}

bool TensorImpl::covers_root() const {
    return offset_ == 0 && is_contiguous() && nbytes() == family_->nbytes &&
           storage_dtype(*family_->root) == meta_.dtype &&
           (numel() > 0 || meta_.stride == contiguous_stride(meta_.shape, dtype_size(meta_.dtype)));
}

const Storage& TensorImpl::storage() const {
    if (gpu_member()) {
        const Storage& root = *family_->root;
        const auto& rg = std::get<GpuStorage>(root);
        // All of the root in the root's own shape: the root is the value.
        if (!rg.arr ||
            (covers_root() && rg.arr->ndim() == meta_.shape.size() &&
             std::equal(meta_.shape.begin(), meta_.shape.end(), rg.arr->shape().begin())))
            return root;
        std::lock_guard<std::mutex> lock(packed_mu_);
        const std::int64_t v = version();
        if (!packed_ || packed_version_ != v ||
            stale_gpu_pack(packed_, packed_root_, rg, is_contiguous())) {
            const gpu_strided::Geometry g(meta_.shape, meta_.stride, offset_,
                                          dtype_size(meta_.dtype));
            ::mlx::core::array elems = gpu_strided::read(*rg.arr, meta_.shape, meta_.stride,
                                                         offset_, gpu::to_mlx_dtype(meta_.dtype));
            if (g.contiguous) {
                // A reshape or a run of the root reads the root's own buffer:
                // no allocation of its own for the memory statistics to count.
                GpuStorage view;
                view.dtype = meta_.dtype;
                view.nbytes = elems.nbytes();
                view.arr = std::make_shared<::mlx::core::array>(std::move(elems));
                packed_ = Storage{std::move(view)};
            } else {
                packed_ = Storage{gpu::wrap_mlx_array(std::move(elems), meta_.dtype)};
            }
            packed_version_ = v;
            packed_root_ = rg.arr;
        }
        return *packed_;
    }
    if (!storage_is_cpu(storage_)) {
        // Every op reads its inputs' values through here: outside phantom
        // mode a phantom has none to give (refuse_phantom_outside_mode).
        refuse_phantom_outside_mode(storage_, "an op read a phantom");
        return storage_;
    }
    if (is_dense())
        return storage_;
    // Packed once per version, not once per call: an op asks several times,
    // and a reference handed out earlier in the same op must still hold the
    // bytes it pointed at.
    std::lock_guard<std::mutex> lock(packed_mu_);
    const std::int64_t v = version();
    if (!packed_ || packed_version_ != v) {
        // Straight into the aligned allocation: this went through a zeroed
        // vector and a second memcpy.
        const auto& src = std::get<CpuStorage>(storage_);
        const std::size_t elem = dtype_size(meta_.dtype);
        const std::size_t nbytes = shape_numel(meta_.shape) * elem;
        CpuStorage packed;
        packed.ptr = allocate_aligned_bytes(nbytes);
        packed.nbytes = nbytes;
        packed.dtype = meta_.dtype;
        if (nbytes > 0)
            strided::pack(src.ptr.get() + offset_, packed.ptr.get(), meta_.shape, meta_.stride,
                          elem);
        packed_ = Storage{std::move(packed)};
        packed_version_ = v;
    }
    return *packed_;
}

std::vector<std::shared_ptr<TensorImpl>> TensorImpl::live_views() const {
    std::vector<std::shared_ptr<TensorImpl>> views;
    if (!family_)
        return views;
    std::lock_guard<std::mutex> lock(family_->mu);
    auto& members = family_->members;
    members.erase(std::remove_if(members.begin(), members.end(),
                                 [](const std::weak_ptr<TensorImpl>& w) { return w.expired(); }),
                  members.end());
    family_->sweep_threshold = std::max<std::size_t>(64, members.size() * 2);
    for (const auto& w : members)
        if (auto m = w.lock(); m && m.get() != this)
            views.push_back(std::move(m));
    return views;
}

void TensorImpl::write_through(const TensorImpl& src, const char* name) {
    check_write_through_target(*this, src, name);
    // Before the trace learns of the write: a refused one never happened.
    // The caller (inplace::write_result or assign_inplace_op) bumps the
    // version once in adopt_graph_position / detach_and_bump.
    if (absorbs_phantom_write(storage_, src.raw_storage(), name))
        return;
    note_inplace_write(*this, src);
    if (gpu_member()) {
        write_gpu_member(src, name, /*raw=*/false);
        return;
    }
    if (meta_.device == Device::GPU) {
        if (src.device() != Device::GPU)
            throw DeviceMismatch(std::string(device_name(meta_.device)),
                                 std::string(device_name(src.device())), name);
        const bool graph = src.requires_grad() || src.grad_fn();
        if (!graph && write_into_shared(src))
            return;
        const auto& value = std::get<GpuStorage>(src.readable());
        auto& dst = std::get<GpuStorage>(storage_);
        dst.arr = gpu::wrap_mlx_array(::mlx::core::copy(*value.arr), dst.dtype).arr;
        return;
    }
    const long members = family_ ? family_.use_count() : 1;
    check_cpu_write_through_target(*this, storage_, src.storage_, members, name);
    auto& dst = std::get<CpuStorage>(storage_);
    const auto& from = std::get<CpuStorage>(src.storage_);
    // Whatever else holds the buffer — a storage saved for backward, a NumPy
    // array — expects the values it has now, and nothing would tell it they
    // changed: a node's saved output carries no version.  A gradient's slot
    // is not among them: a tensor read from ``.grad`` holds the buffer
    // through a pointer of its own (GradSlotRef).
    if (dst.ptr.use_count() > members)
        ErrorBuilder(name).not_implemented(
            "an in-place write to a tensor that shares storage with a live view is not supported "
            "while something else also holds that storage (a tensor saved for backward, a NumPy "
            "array) — clone() first");
    write_cpu_buffer(*this, dst, src, from, shared_);
}

void TensorImpl::write_gpu_member(const TensorImpl& src, const char* name, bool raw) {
    if (src.device() != Device::GPU)
        throw DeviceMismatch(std::string(device_name(meta_.device)),
                             std::string(device_name(src.device())), name);
    refuse_ambiguous_write(*this, name);
    const bool graph = src.requires_grad() || src.grad_fn();
    if (!raw && GradMode::is_enabled() && !is_detached_alias()) {
        refuse_write_to_leaf_view(*this, name);
    }
    const auto& value = std::get<GpuStorage>(src.readable());
    auto& root = std::get<GpuStorage>(*family_->root);
    if (!value.arr || !root.arr)
        ErrorBuilder(name).fail("a Metal tensor without an array");
    const gpu_strided::Geometry g(meta_.shape, meta_.stride, offset_, dtype_size(meta_.dtype));
    ::mlx::core::array next =
        gpu_strided::write(*root.arr, *value.arr, meta_.shape, meta_.stride, offset_);
    trace_gpu_member_write(*this, src, *root.arr, *value.arr, next, g);
    // The family was built over a Metal shared buffer that the root still
    // reads.  With no graph to protect — a node's snapshot reads the same
    // bytes — the values go into that buffer, so its CPU alias sees them as
    // well (take_storage_from's rule), and the root stays the buffer.
    if ((raw || !graph) && root_is_shared_buffer(shared_, root, root.dtype)) {
        copy_into_shared(*shared_, next);
    } else if (root.dtype == meta_.dtype && g.covers(static_cast<std::int64_t>(root.arr->size())) &&
               value.arr->shape() == root.arr->shape()) {
        // The whole value in the root's shape: the root is the value itself,
        // as a slot swap would have made it.
        root.arr = value.arr;
    } else {
        root.arr = gpu::wrap_mlx_array(std::move(next), root.dtype).arr;
    }
    rebind_gpu_grad_slot(family_, root.arr);
    drop_family_caches();
    if (raw)
        bump_version();
}

void TensorImpl::drop_family_caches() {
    // The cached elements, and the value the base held before the family
    // formed: either would keep an old root's buffer alive.
    const auto drop = [](TensorImpl& t) {
        {
            std::lock_guard<std::mutex> lock(t.packed_mu_);
            t.packed_.reset();
            t.packed_root_.reset();
        }
        if (const auto* g = std::get_if<GpuStorage>(&t.storage_); g && g->arr)
            t.storage_ = gpu_placeholder(t.meta_.dtype);
    };
    drop(*this);
    for (const auto& m : live_views())
        drop(*m);
}

void TensorImpl::leave_family() {
    Storage own = storage();
    const auto carried =
        static_cast<std::int64_t>(family_->version.load(std::memory_order_relaxed));
    auto sink = family_->version_sink;
    {
        std::lock_guard<std::mutex> lock(packed_mu_);
        packed_.reset();
        packed_root_.reset();
    }
    family_.reset();
    view_lineage_.reset();
    // The family's counter folds into this tensor's own, so version() does
    // not move backwards (drop_shared does the same).
    if (carried > 0)
        ensure_autograd()->version += carried;
    if (sink)
        ensure_autograd()->version_sink = std::move(sink);
    storage_ = std::move(own);
    meta_.stride = contiguous_stride(meta_.shape, dtype_size(meta_.dtype));
    offset_ = 0;
}

void TensorImpl::check_wholesale_writable(const char* name) const {
    if (gpu_member() && !(offset_ == 0 && is_contiguous() && nbytes() == family_->nbytes))
        ErrorBuilder(name).not_implemented(
            "an engine in-place step (an optimizer, BatchNorm's running statistics, a compiled "
            "step) on a Metal view of part of a tensor is not supported — clone() first");
}

Storage& TensorImpl::mutable_storage() {
    if (!gpu_member()) {
        // A CPU writer (an optimizer step, BatchNorm's statistics) writes the
        // bytes it gets in place.
        settle_before_host_write(shared_, storage_);
        return storage_;
    }
    check_wholesale_writable("mutable_storage");
    auto& root = std::get<GpuStorage>(*family_->root);
    if (root.arr && root.dtype != meta_.dtype) {
        root.arr = gpu::wrap_mlx_array(::mlx::core::view(::mlx::core::flatten(*root.arr),
                                                         gpu::to_mlx_dtype(meta_.dtype)),
                                       meta_.dtype)
                       .arr;
        root.dtype = meta_.dtype;
    }
    if (root.arr &&
        !(root.arr->ndim() == meta_.shape.size() &&
          std::equal(meta_.shape.begin(), meta_.shape.end(), root.arr->shape().begin())))
        root.arr =
            gpu::wrap_mlx_array(::mlx::core::reshape(*root.arr, gpu_strided::to_mlx(meta_.shape)),
                                meta_.dtype)
                .arr;
    drop_family_caches();
    return *family_->root;
}

void TensorImpl::drop_shared() {
    if (!shared_)
        return;
    // Fold the buffer's counter into this tensor's own, so version() carries
    // on instead of dropping back: a saved snapshot that a later, smaller
    // count happened to reach again would hide a mutation.
    if (const auto carried = static_cast<std::int64_t>(shared_->get_version()); carried > 0)
        ensure_autograd()->version += carried;
    shared_.reset();
}

std::shared_ptr<TensorImpl>
TensorImpl::from_numpy(py::array arr, Device device, bool requires_grad) {
    if (!arr) {
        ErrorBuilder("from_numpy").fail("input is not a numpy array");
    }

    Dtype dtype = np_dtype_to_lucid(arr.dtype());
    Shape shape;
    shape.reserve(arr.ndim());
    for (py::ssize_t i = 0; i < arr.ndim(); ++i) {
        shape.push_back(static_cast<std::int64_t>(arr.shape(i)));
    }

    const std::size_t elem = dtype_size(dtype);
    const std::size_t total = shape_numel(shape) * elem;
    // A factory like any other: under phantom mode the array's values are
    // not copied.
    if (PhantomMode::is_enabled())
        return std::make_shared<TensorImpl>(phantom_storage(shape, dtype, device), std::move(shape),
                                            dtype, device, requires_grad);

    // Always land in a fresh aligned CPU buffer first, regardless of the
    // target device.  For GPU tensors the buffer is uploaded below via the
    // MLX bridge.  We never write into arr's buffer directly to avoid
    // inadvertently mutating the Python-side array.
    CpuStorage cpu;
    cpu.ptr = allocate_aligned_bytes(total);
    cpu.nbytes = total;
    cpu.dtype = dtype;

    // Preserve dtype while packing strides. An array_t<std::byte> forcecast
    // converts values rather than reinterpreting bytes: it allocated an unused
    // buffer and emitted lossy-cast warnings (or failed under warnings=error).
    py::array contig = py::array::ensure(arr, py::array::c_style | py::array::forcecast);
    if (!contig) {
        ErrorBuilder("from_numpy").fail("failed to obtain C-contiguous view");
    }
    if (total > 0) {
        std::memcpy(cpu.ptr.get(), contig.data(), total);
    }

    if (device == Device::GPU) {
        // Upload the CPU-side copy to the MLX lazy graph and discard the
        // temporary CPU buffer; the GpuStorage retains the only live reference.
        return std::make_shared<TensorImpl>(Storage{gpu::upload_cpu_to_gpu(cpu, shape)},
                                            std::move(shape), dtype, device, requires_grad);
    }
    return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, std::move(shape), dtype, device,
                                        requires_grad);
}

std::size_t TensorImpl::numel() const {
    return meta_.numel();
}

std::size_t TensorImpl::nbytes() const {
    return numel() * dtype_size(meta_.dtype);
}

bool TensorImpl::is_contiguous() const {
    for (const auto n : meta_.shape)
        if (n == 0)
            return true;
    std::int64_t expected = static_cast<std::int64_t>(dtype_size(meta_.dtype));
    for (std::size_t d = meta_.shape.size(); d-- > 0;) {
        if (meta_.shape[d] != 1) {
            if (meta_.stride[d] != expected)
                return false;
            expected *= meta_.shape[d];
        }
    }
    return true;
}

bool TensorImpl::is_dense() const {
    return offset_ == 0 && is_contiguous() && storage_nbytes(raw_storage()) == nbytes() &&
           (numel() > 0 || meta_.stride == contiguous_stride(meta_.shape, dtype_size(meta_.dtype)));
}

std::size_t TensorImpl::allocation_offset() const noexcept {
    if (const auto* cpu = std::get_if<CpuStorage>(&storage_))
        return cpu->offset_in_allocation + offset_;
    return offset_;
}

std::size_t TensorImpl::allocation_nbytes() const noexcept {
    if (const auto* cpu = std::get_if<CpuStorage>(&storage_))
        return cpu->allocation_nbytes != 0 ? cpu->allocation_nbytes : cpu->nbytes;
    return storage_nbytes(raw_storage());
}

std::uintptr_t TensorImpl::allocation_base() const {
    const Storage& s = raw_storage();
    if (storage_is_phantom(s))
        refuse_phantom_read("data_ptr");
    if (storage_is_cpu(s)) {
        const auto& cpu = storage_cpu(s);
        if (!cpu.ptr)
            return 0;
        return reinterpret_cast<std::uintptr_t>(cpu.ptr.get() - cpu.offset_in_allocation);
    }
    if (storage_is_gpu(s)) {
        const auto& g = storage_gpu(s);
        if (!g.arr)
            return 0;
        g.arr->eval();
        MemoryTracker::track_host_sync(g.nbytes);
        return reinterpret_cast<std::uintptr_t>(g.arr->data<std::uint8_t>());
    }
    return reinterpret_cast<std::uintptr_t>(storage_metal_shared(s).cpu_ptr);
}

// Dispatches on the active Storage variant to produce a NumPy array.
// GPU tensors are synchronously downloaded via the MLX bridge before wrapping;
// SharedStorage is exposed through its cpu_view() alias.
py::object TensorImpl::data_as_python() const {
    return std::visit(
        overloaded{
            [&](const CpuStorage& s) -> py::object {
                return make_numpy_view(s, meta_.shape, meta_.stride, offset_);
            },
            [&](const GpuStorage& g) -> py::object {
                // The download is a fresh, dense buffer, whatever
                // geometry a Metal view has over its family's root.
                CpuStorage cpu = gpu::download_gpu_to_cpu(g, meta_.shape);
                return make_numpy_view(cpu, meta_.shape,
                                       contiguous_stride(meta_.shape, dtype_size(meta_.dtype)), 0);
            },

            [&](const SharedStorage& sh) -> py::object {
                CpuStorage v = sh.cpu_view();
                return make_numpy_view(v, meta_.shape, meta_.stride, offset_);
            },
            [&](const PhantomStorage&) -> py::object { refuse_phantom_read("numpy"); },
        },
        readable());
}

// Same variant dispatch as data_as_python() but applied to the optional
// gradient storage.  Returns py::none() when no gradient has been accumulated.
py::object TensorImpl::grad_as_python() const {
    if (!autograd_) {
        return py::none();
    }
    if (autograd_->grad_impl) {
        return autograd_->grad_impl->data_as_python();
    }
    if (!autograd_->grad.has_value()) {
        return py::none();
    }
    // A gradient is always its own dense buffer, so it starts at byte 0.
    return std::visit(
        overloaded{
            [&](const CpuStorage& s) -> py::object {
                return make_numpy_view(s, meta_.shape, meta_.stride, 0);
            },
            [&](const GpuStorage& g) -> py::object {
                CpuStorage cpu = gpu::download_gpu_to_cpu(g, meta_.shape);
                return make_numpy_view(cpu, meta_.shape, meta_.stride, 0);
            },

            [&](const SharedStorage& sh) -> py::object {
                CpuStorage v = sh.cpu_view();
                return make_numpy_view(v, meta_.shape, meta_.stride, 0);
            },
            [&](const PhantomStorage&) -> py::object { refuse_phantom_read("grad numpy"); },
        },
        *autograd_->grad);
}

// ---------------------------------------------------------------------------
// NumPy-free Python interop: to_bytes / from_bytes / to_string
//
// These three implement everything ``import lucid`` needs for
// serialization and tensor printing without touching numpy.  Numpy is
// still used in the explicit bridge methods (data_as_python /
// from_numpy) but those are now strictly opt-in.
// ---------------------------------------------------------------------------

namespace {

// Materialises a contiguous byte snapshot of a CpuStorage view.
std::vector<std::byte> contig_snapshot_cpu(const CpuStorage& s,
                                           const Shape& shape,
                                           const Stride& stride,
                                           std::size_t storage_offset) {
    const std::size_t elem = dtype_size(s.dtype);
    const std::size_t total = shape_numel(shape) * elem;
    std::vector<std::byte> out(total);
    if (total == 0)
        return out;
    strided::pack(s.ptr.get() + storage_offset, out.data(), shape, stride, elem);
    return out;
}

// Lays ``packed`` — ``shape``'s elements, dense and row-major — into ``dst``
// through ``stride`` from ``storage_offset``: the inverse of
// :func:`contig_snapshot_cpu`, which is how a write reaches a view's own
// elements rather than the buffer's first bytes.  Runs move as a memcpy
// when the inner axis is contiguous and as typed stores otherwise.  A view
// written through has a repeated element only under ``fill_``, whose packed
// elements are all one value (refuse_ambiguous_write).
void scatter_contig_to_strided(const std::byte* packed,
                               CpuStorage& dst,
                               const Shape& shape,
                               const Stride& stride,
                               std::size_t storage_offset) {
    if (shape_numel(shape) == 0)
        return;
    strided::scatter(packed, dst.ptr.get() + storage_offset, shape, stride, dtype_size(dst.dtype));
}

// Format a single scalar element at ``ptr`` of dtype ``dt`` into ``os``.
// Mirrors NumPy's default formatting closely enough for human reading:
//   * floats use %.<precision>g, with explicit "nan" / "inf" / "-inf"
//   * bools render as True / False
//   * complex64 renders as "(re+imj)" with Python-side conventions
//   * integers render with %lld / %ld via std::to_string
void format_element(const std::byte* ptr, Dtype dt, int precision, std::ostringstream& os) {
    auto fmt_float = [&](double v) {
        if (std::isnan(v)) {
            os << "nan";
            return;
        }
        if (std::isinf(v)) {
            os << (v < 0 ? "-inf" : "inf");
            return;
        }
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
        std::string s(buf);
        // NumPy convention: a trailing dot is appended when the rendered
        // value would otherwise look like an integer (e.g. "1." instead
        // of "1") so the reader knows the dtype is floating-point.
        if (s.find_first_of(".eEnN") == std::string::npos) {
            s.push_back('.');
        }
        os << s;
    };

    switch (dt) {
    case Dtype::Bool: {
        const auto v = *reinterpret_cast<const std::uint8_t*>(ptr);
        os << (v ? "True" : "False");
        break;
    }
    case Dtype::I8:
        os << static_cast<int>(*reinterpret_cast<const std::int8_t*>(ptr));
        break;
    case Dtype::I16:
        os << *reinterpret_cast<const std::int16_t*>(ptr);
        break;
    case Dtype::I32:
        os << *reinterpret_cast<const std::int32_t*>(ptr);
        break;
    case Dtype::I64:
        os << *reinterpret_cast<const std::int64_t*>(ptr);
        break;
    case Dtype::BF16: {
        // bfloat16 is the top half of a float32: shift it back up.
        const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(ptr);
        const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &widened, sizeof(value));
        fmt_float(static_cast<double>(value));
        break;
    }
    case Dtype::F16: {
        // Half is stored as raw bits; cast through float for printing.  We
        // don't pull in __fp16 — manual IEEE-754 binary16 → float decode.
        const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(ptr);
        const std::uint32_t sign = (bits >> 15) & 0x1;
        const std::uint32_t exp = (bits >> 10) & 0x1f;
        const std::uint32_t mant = bits & 0x3ff;
        std::uint32_t f;
        if (exp == 0) {
            if (mant == 0) {
                f = sign << 31;
            } else {
                std::uint32_t e = 1;
                std::uint32_t m = mant;
                while ((m & 0x400) == 0) {
                    m <<= 1;
                    --e;
                }
                m &= 0x3ff;
                f = (sign << 31) | ((e + 112) << 23) | (m << 13);
            }
        } else if (exp == 31) {
            f = (sign << 31) | (0xff << 23) | (mant << 13);
        } else {
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
        float out;
        std::memcpy(&out, &f, sizeof(out));
        fmt_float(static_cast<double>(out));
        break;
    }
    case Dtype::F32:
        fmt_float(static_cast<double>(*reinterpret_cast<const float*>(ptr)));
        break;
    case Dtype::F64:
        fmt_float(*reinterpret_cast<const double*>(ptr));
        break;
    case Dtype::C64: {
        // Stored as two contiguous f32: real, imag.
        const float re = *reinterpret_cast<const float*>(ptr);
        const float im = *reinterpret_cast<const float*>(ptr + 4);
        os << "(";
        fmt_float(static_cast<double>(re));
        if (im >= 0 || std::isnan(im))
            os << "+";
        fmt_float(static_cast<double>(im));
        os << "j)";
        break;
    }
    case Dtype::C128: {
        // The same, in two contiguous f64.
        const double re = *reinterpret_cast<const double*>(ptr);
        const double im = *reinterpret_cast<const double*>(ptr + 8);
        os << "(";
        fmt_float(re);
        if (im >= 0 || std::isnan(im))
            os << "+";
        fmt_float(im);
        os << "j)";
        break;
    }
    }
}

// Renders a contiguous (already-snapshot) buffer of shape × dtype as a
// NumPy-flavoured nested-bracket string.  ``threshold`` applies to total
// element count; ``edgeitems`` clips long axes once that ceiling is
// exceeded.
void render_nested(const std::byte* base,
                   const Shape& shape,
                   std::size_t depth,
                   std::size_t off,
                   const std::vector<std::int64_t>& contig_stride_bytes,
                   Dtype dt,
                   int precision,
                   bool truncate,
                   std::size_t edgeitems,
                   std::ostringstream& os) {
    if (depth == shape.size()) {
        format_element(base + off, dt, precision, os);
        return;
    }
    os << "[";
    const std::int64_t n = shape[depth];
    const std::int64_t step = contig_stride_bytes[depth];

    auto render_one = [&](std::int64_t i) {
        render_nested(base, shape, depth + 1, off + static_cast<std::size_t>(i * step),
                      contig_stride_bytes, dt, precision, truncate, edgeitems, os);
    };

    if (truncate && n > static_cast<std::int64_t>(edgeitems) * 2) {
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(edgeitems); ++i) {
            if (i > 0)
                os << ", ";
            render_one(i);
        }
        os << ", ..., ";
        for (std::int64_t i = n - static_cast<std::int64_t>(edgeitems); i < n; ++i) {
            render_one(i);
            if (i + 1 < n)
                os << ", ";
        }
    } else {
        for (std::int64_t i = 0; i < n; ++i) {
            if (i > 0)
                os << ", ";
            render_one(i);
        }
    }
    os << "]";
}

}  // namespace

py::bytes TensorImpl::to_bytes() const {
    const std::size_t total = nbytes();
    // A phantom refuses even when empty: it has no values, not zero of them.
    if (total == 0 && !is_phantom())
        return py::bytes();

    return std::visit(
        overloaded{
            [&](const CpuStorage& s) -> py::bytes {
                auto buf = contig_snapshot_cpu(s, meta_.shape, meta_.stride, offset_);
                return py::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
            },
            [&](const GpuStorage& g) -> py::bytes {
                CpuStorage cpu = gpu::download_gpu_to_cpu(g, meta_.shape);
                return py::bytes(reinterpret_cast<const char*>(cpu.ptr.get()), total);
            },
            [&](const SharedStorage& sh) -> py::bytes {
                CpuStorage v = sh.cpu_view();
                auto buf = contig_snapshot_cpu(v, meta_.shape, meta_.stride, offset_);
                return py::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
            },
            [&](const PhantomStorage&) -> py::bytes { refuse_phantom_read("to_bytes"); },
        },
        readable());
}

std::shared_ptr<TensorImpl> TensorImpl::from_bytes(
    py::bytes data, Shape shape, Dtype dtype, Device device, bool requires_grad) {
    const std::size_t elem = dtype_size(dtype);
    const std::size_t expected = shape_numel(shape) * elem;

    char* raw_ptr = nullptr;
    py::ssize_t raw_len = 0;
    if (PyBytes_AsStringAndSize(data.ptr(), &raw_ptr, &raw_len) != 0) {
        ErrorBuilder("from_bytes").fail("input is not a bytes object");
    }
    if (static_cast<std::size_t>(raw_len) != expected) {
        ErrorBuilder("from_bytes")
            .fail("byte length mismatch (got " + std::to_string(raw_len) + ", expected " +
                  std::to_string(expected) + ")");
    }
    if (PhantomMode::is_enabled())
        return std::make_shared<TensorImpl>(phantom_storage(shape, dtype, device), std::move(shape),
                                            dtype, device, requires_grad);

    CpuStorage cpu;
    cpu.ptr = allocate_aligned_bytes(expected);
    cpu.nbytes = expected;
    cpu.dtype = dtype;
    if (expected > 0) {
        std::memcpy(cpu.ptr.get(), raw_ptr, expected);
    }

    if (device == Device::GPU) {
        return std::make_shared<TensorImpl>(Storage{gpu::upload_cpu_to_gpu(cpu, shape)},
                                            std::move(shape), dtype, device, requires_grad);
    }
    return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, std::move(shape), dtype, device,
                                        requires_grad);
}

// Numpy-free cross-device copy used by ``Tensor.to(device=...)``.
// Mirrors the structure of ``to_bytes``+``from_bytes`` but skips the
// intermediate ``py::bytes`` round-trip — the contiguous-snapshot vector
// is reused directly as the new CpuStorage backing, so the data moves
// exactly once between source and destination storage.
std::shared_ptr<TensorImpl> TensorImpl::transfer_to_device(Device target,
                                                           bool requires_grad) const {
    auto build_cpu_from_view = [&](const CpuStorage& src) {
        auto snap = contig_snapshot_cpu(src, meta_.shape, meta_.stride, offset_);
        const std::size_t total = snap.size();
        CpuStorage out;
        out.ptr = allocate_aligned_bytes(total);
        out.nbytes = total;
        out.dtype = meta_.dtype;
        if (total > 0) {
            std::memcpy(out.ptr.get(), snap.data(), total);
        }
        return out;
    };
    // Under phantom mode any tensor moves as a label: a new phantom of the
    // same shape and dtype on ``target``, the source untouched and no copy
    // run.  Outside it a phantom has no values to move.
    if (PhantomMode::is_enabled())
        return std::make_shared<TensorImpl>(phantom_storage(meta_.shape, meta_.dtype, target),
                                            meta_.shape, meta_.dtype, target, requires_grad);

    return std::visit(
        overloaded{
            [&](const CpuStorage& s) -> std::shared_ptr<TensorImpl> {
                CpuStorage cpu = build_cpu_from_view(s);
                if (target == Device::GPU) {
                    return std::make_shared<TensorImpl>(
                        Storage{gpu::upload_cpu_to_gpu(cpu, meta_.shape)}, meta_.shape, meta_.dtype,
                        target, requires_grad);
                }
                return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, meta_.shape,
                                                    meta_.dtype, target, requires_grad);
            },
            [&](const GpuStorage& g) -> std::shared_ptr<TensorImpl> {
                CpuStorage cpu = gpu::download_gpu_to_cpu(g, meta_.shape);
                if (target == Device::GPU) {
                    // GPU → GPU: rare path; round-trip through CPU rather than
                    // adding an MLX-level array clone helper just for this.
                    return std::make_shared<TensorImpl>(
                        Storage{gpu::upload_cpu_to_gpu(cpu, meta_.shape)}, meta_.shape, meta_.dtype,
                        target, requires_grad);
                }
                return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, meta_.shape,
                                                    meta_.dtype, target, requires_grad);
            },
            [&](const SharedStorage& sh) -> std::shared_ptr<TensorImpl> {
                // _to.py routes is_metal_shared tensors through transfer_storage()
                // — that path is zero-copy relabel.  We provide this fallthrough
                // for safety: contiguous CPU snapshot then route as CPU storage.
                CpuStorage cpu = build_cpu_from_view(sh.cpu_view());
                if (target == Device::GPU) {
                    return std::make_shared<TensorImpl>(
                        Storage{gpu::upload_cpu_to_gpu(cpu, meta_.shape)}, meta_.shape, meta_.dtype,
                        target, requires_grad);
                }
                return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, meta_.shape,
                                                    meta_.dtype, target, requires_grad);
            },
            [&](const PhantomStorage&) -> std::shared_ptr<TensorImpl> {
                refuse_phantom_read("to (a device copy) outside phantom mode");
            },
        },
        readable());
}

namespace {

// Decode a single IEEE-754 binary16 (half) bit pattern to a host float.
// Mirrors the inline decoder in ``TensorImpl::item()`` and is kept in
// this translation unit so the only divergence is the surrounding loop.
inline float decode_f16(std::uint16_t bits) {
    std::uint32_t sign = (bits >> 15) & 0x1u;
    std::uint32_t exp = (bits >> 10) & 0x1fu;
    std::uint32_t mant = bits & 0x3ffu;
    std::uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            std::uint32_t e = 1;
            std::uint32_t m = mant;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x3ffu;
            f = (sign << 31) | ((e + 112) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xffu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// Walk ``shape`` recursively over ``raw`` (a contiguous row-major byte
// buffer for the active dtype) and emit a nested py::list whose leaves
// are Python scalars produced by ``scalar``.  ``elem_size`` is the byte
// stride of one element in ``raw``.
template <typename ScalarFn>
py::object emit_nested(const char* raw,
                       const Shape& shape,
                       std::size_t depth,
                       std::size_t elem_size,
                       const ScalarFn& scalar) {
    if (depth == shape.size()) {
        return scalar(raw);
    }
    const std::int64_t dim = shape[depth];
    // Stride in bytes for one step along this axis = product of trailing
    // dims × elem_size.  Compute on the fly so we don't materialise a
    // stride vector for the (typically tiny) recursion depth.
    std::size_t stride = elem_size;
    for (std::size_t i = depth + 1; i < shape.size(); ++i) {
        stride *= static_cast<std::size_t>(shape[i]);
    }
    py::list out;
    for (std::int64_t i = 0; i < dim; ++i) {
        out.append(emit_nested(raw + static_cast<std::size_t>(i) * stride, shape, depth + 1,
                               elem_size, scalar));
    }
    return out;
}

}  // namespace

py::object TensorImpl::tolist() const {
    // Reuse the existing ``to_bytes()`` snapshot — it already handles the
    // CpuStorage / GpuStorage / SharedStorage variants + GPU download and
    // collapses non-contiguous views into a fresh row-major buffer.  The
    // returned ``py::bytes`` keeps the buffer alive for the duration of
    // this call; we read through ``raw`` below without owning it.
    py::bytes blob = to_bytes();
    char* raw = nullptr;
    py::ssize_t len = 0;
    PyBytes_AsStringAndSize(blob.ptr(), &raw, &len);
    (void)len;  // Trust shape × dtype_size; ``to_bytes`` already validated.

    const std::size_t elem = dtype_size(meta_.dtype);

    // Build a typed scalar emitter per dtype, then hand it to
    // ``emit_nested`` which walks the shape generically.  Each emitter
    // returns the appropriate Python type (``int`` / ``float`` / ``bool``
    // / ``complex``) so the resulting list matches what ``Tensor.item()``
    // would produce for a 0-d slice.
    auto walk = [&](auto scalar) -> py::object {
        if (meta_.shape.empty()) {
            // 0-d tensor → bare Python scalar.
            return scalar(raw);
        }
        return emit_nested(raw, meta_.shape, 0, elem, scalar);
    };

    switch (meta_.dtype) {
    case Dtype::F32:
        return walk([](const char* p) -> py::object {
            float v;
            std::memcpy(&v, p, sizeof(v));
            return py::float_(static_cast<double>(v));
        });
    case Dtype::F64:
        return walk([](const char* p) -> py::object {
            double v;
            std::memcpy(&v, p, sizeof(v));
            return py::float_(v);
        });
    case Dtype::BF16:
        return walk([](const char* p) -> py::object {
            std::uint16_t bits;
            std::memcpy(&bits, p, sizeof(bits));
            const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
            float value;
            std::memcpy(&value, &widened, sizeof(value));
            return py::float_(static_cast<double>(value));
        });
    case Dtype::F16:
        return walk([](const char* p) -> py::object {
            std::uint16_t bits;
            std::memcpy(&bits, p, sizeof(bits));
            return py::float_(static_cast<double>(decode_f16(bits)));
        });
    case Dtype::I64:
        return walk([](const char* p) -> py::object {
            std::int64_t v;
            std::memcpy(&v, p, sizeof(v));
            return py::int_(v);
        });
    case Dtype::I32:
        return walk([](const char* p) -> py::object {
            std::int32_t v;
            std::memcpy(&v, p, sizeof(v));
            return py::int_(v);
        });
    case Dtype::I16:
        return walk([](const char* p) -> py::object {
            std::int16_t v;
            std::memcpy(&v, p, sizeof(v));
            return py::int_(static_cast<int>(v));
        });
    case Dtype::I8:
        return walk([](const char* p) -> py::object {
            std::int8_t v;
            std::memcpy(&v, p, sizeof(v));
            return py::int_(static_cast<int>(v));
        });
    case Dtype::Bool:
        return walk([](const char* p) -> py::object {
            std::uint8_t v;
            std::memcpy(&v, p, sizeof(v));
            return py::bool_(v != 0);
        });
    case Dtype::C64:
        return walk([](const char* p) -> py::object {
            // complex64 = 2 contiguous f32, real then imag.
            float re, im;
            std::memcpy(&re, p, sizeof(re));
            std::memcpy(&im, p + sizeof(re), sizeof(im));
            return py::cast(std::complex<double>(static_cast<double>(re), static_cast<double>(im)));
        });
    case Dtype::C128:
        return walk([](const char* p) -> py::object {
            double re, im;
            std::memcpy(&re, p, sizeof(re));
            std::memcpy(&im, p + sizeof(re), sizeof(im));
            return py::cast(std::complex<double>(re, im));
        });
    }
    ErrorBuilder("tolist").fail("unsupported dtype");
    return py::none();
}

// Decode a single scalar element at ``raw`` to its Python representation,
// based on the engine dtype.  Shared between ``item()`` (fast path) and any
// other single-element decoders that need the same logic.  Branchless after
// the dtype switch; no allocations beyond the returned Python object itself.
static py::object decode_scalar(const char* raw, Dtype dt) {
    switch (dt) {
    case Dtype::F32: {
        float v;
        std::memcpy(&v, raw, sizeof(v));
        return py::float_(static_cast<double>(v));
    }
    case Dtype::F64: {
        double v;
        std::memcpy(&v, raw, sizeof(v));
        return py::float_(v);
    }
    case Dtype::BF16: {
        std::uint16_t bits;
        std::memcpy(&bits, raw, sizeof(bits));
        const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &widened, sizeof(value));
        return py::float_(static_cast<double>(value));
    }
    case Dtype::F16: {
        // IEEE-754 binary16 → float32 decode, kept inline so this stays a
        // self-contained engine helper (matches to_string()'s format_element).
        std::uint16_t bits;
        std::memcpy(&bits, raw, sizeof(bits));
        std::uint32_t sign = (bits >> 15) & 0x1u;
        std::uint32_t exp = (bits >> 10) & 0x1fu;
        std::uint32_t mant = bits & 0x3ffu;
        std::uint32_t f;
        if (exp == 0) {
            if (mant == 0) {
                f = sign << 31;
            } else {
                std::uint32_t e = 1;
                std::uint32_t m = mant;
                while ((m & 0x400u) == 0) {
                    m <<= 1;
                    --e;
                }
                m &= 0x3ffu;
                f = (sign << 31) | ((e + 112) << 23) | (m << 13);
            }
        } else if (exp == 31) {
            f = (sign << 31) | (0xffu << 23) | (mant << 13);
        } else {
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
        float out;
        std::memcpy(&out, &f, sizeof(out));
        return py::float_(static_cast<double>(out));
    }
    case Dtype::I64: {
        std::int64_t v;
        std::memcpy(&v, raw, sizeof(v));
        return py::int_(v);
    }
    case Dtype::I32: {
        std::int32_t v;
        std::memcpy(&v, raw, sizeof(v));
        return py::int_(v);
    }
    case Dtype::I16: {
        std::int16_t v;
        std::memcpy(&v, raw, sizeof(v));
        return py::int_(static_cast<int>(v));
    }
    case Dtype::I8: {
        std::int8_t v;
        std::memcpy(&v, raw, sizeof(v));
        return py::int_(static_cast<int>(v));
    }
    case Dtype::Bool: {
        std::uint8_t v;
        std::memcpy(&v, raw, sizeof(v));
        return py::bool_(v != 0);
    }
    case Dtype::C64: {
        float re, im;
        std::memcpy(&re, raw, sizeof(re));
        std::memcpy(&im, raw + sizeof(re), sizeof(im));
        return py::cast(std::complex<double>(static_cast<double>(re), static_cast<double>(im)));
    }
    case Dtype::C128: {
        double re, im;
        std::memcpy(&re, raw, sizeof(re));
        std::memcpy(&im, raw + sizeof(re), sizeof(im));
        return py::cast(std::complex<double>(re, im));
    }
    }
    ErrorBuilder("item").fail("unsupported dtype");
    return py::none();
}

py::object TensorImpl::item() const {
    if (numel() != 1) {
        ErrorBuilder("item").fail("item() can only be called on a tensor with one element");
    }

    // 3.2.0 fast path: read the single element directly from the storage
    // buffer.  The old implementation routed through to_bytes() →
    // download_gpu_to_cpu() → py::bytes() — three allocations + a memcpy
    // of the entire tensor buffer (here, just 1 element, but the
    // per-allocation overhead dominates for hot training loops).
    // cProfile of LeNet-5/MNIST training showed engine.item() as 48 % of
    // per-epoch CPU time (1.6 s of 3.3 s, 1874 calls/epoch).  This path
    // collapses the work to: eval (GPU only) → pointer-offset → scalar
    // decode → py::cast.  Measured on M4 Max: 870 µs/call → ~100 µs/call.
    // ``offset_`` is already in bytes (see TensorImpl.h); scaling it by the
    // element size again read the wrong element of any view at an offset.
    const std::size_t byte_offset = offset_;
    return std::visit(
        overloaded{
            [&](const CpuStorage& s) -> py::object {
                const auto* raw = reinterpret_cast<const char*>(s.ptr.get()) + byte_offset;
                return decode_scalar(raw, meta_.dtype);
            },
            [&](const GpuStorage& g) -> py::object {
                if (!g.arr) {
                    ErrorBuilder("item").fail("null GPU array");
                }
                g.arr->eval();
                MemoryTracker::track_host_sync(dtype_size(meta_.dtype));
                // A Metal tensor's value is its own elements: a family
                // member's offset counts in the root, which readable() has
                // already read through.
                const auto* raw = reinterpret_cast<const char*>(g.arr->data<std::uint8_t>());
                return decode_scalar(raw, meta_.dtype);
            },
            [&](const SharedStorage& sh) -> py::object {
                CpuStorage v = sh.cpu_view();
                const auto* raw = reinterpret_cast<const char*>(v.ptr.get()) + byte_offset;
                return decode_scalar(raw, meta_.dtype);
            },
            [&](const PhantomStorage&) -> py::object { refuse_phantom_read("item"); },
        },
        readable());
}

std::shared_ptr<TensorImpl> TensorImpl::grad_to_tensor() const {
    if (!autograd_)
        return nullptr;
    // Prefer the graph-mode gradient (set when backward(create_graph=True)).
    if (autograd_->grad_impl)
        return autograd_->grad_impl;
    if (!autograd_->grad.has_value())
        return nullptr;
    auto& slot = const_cast<std::optional<Storage>&>(autograd_->grad);
    if (auto* gpu = std::get_if<GpuStorage>(&*slot); gpu && gpu->arr)
        return gpu_grad_to_tensor(*gpu);
    auto* cpu = std::get_if<CpuStorage>(&*slot);
    if (cpu == nullptr || !cpu->ptr)
        return std::make_shared<TensorImpl>(*slot, meta_.shape, meta_.dtype, meta_.device, false);
    auto* owned = std::get_deleter<GradSlot>(cpu->ptr);
    if (owned == nullptr) {
        const std::shared_ptr<void> record(nullptr, GradSlot{cpu->ptr, {}});
        cpu->ptr = std::shared_ptr<std::byte[]>(record, cpu->ptr.get());
        owned = std::get_deleter<GradSlot>(cpu->ptr);
    }
    // The impl handed out last, if it lives and still reads the whole slot.
    if (auto h = owned->handle.lock()) {
        const auto* held = std::get_if<CpuStorage>(&h->raw_storage());
        const auto* ref = held != nullptr ? std::get_deleter<GradSlotRef>(held->ptr) : nullptr;
        if (ref != nullptr && same_owner(ref->slot, cpu->ptr) && ref->owner.lock().get() == this &&
            h->storage_offset() == 0 && h->is_contiguous() && h->shape() == meta_.shape &&
            h->dtype() == meta_.dtype)
            return h;
    }
    CpuStorage reads = *cpu;
    const std::shared_ptr<void> ref(nullptr, GradSlotRef{cpu->ptr, weak_from_this()});
    reads.ptr = std::shared_ptr<std::byte[]>(ref, cpu->ptr.get());
    auto h = std::make_shared<TensorImpl>(Storage{std::move(reads)}, meta_.shape, meta_.dtype,
                                          meta_.device, false);
    owned->handle = h;
    return h;
}

void TensorImpl::install_grad(const std::shared_ptr<TensorImpl>& g, Storage slot) {
    if (!storage_is_phantom(slot) && g && (g->requires_grad() || g->grad_fn())) {
        set_grad_impl(g);
        return;
    }
    if (meta_.device == Device::GPU && storage_is_gpu(slot) && g && can_alias_gpu_grad(*g)) {
        const auto handle = current_gpu_grad_handle(autograd_, g);
        if (!g->family_) {
            g->family_ = std::make_shared<ViewFamily>();
            if (g->autograd_ && g->autograd_->version > 0) {
                g->family_->version.store(static_cast<std::uint64_t>(g->autograd_->version),
                                          std::memory_order_relaxed);
                g->autograd_->version = 0;
            }
            g->family_->nbytes = g->allocation_nbytes();
            g->family_->members.push_back(g);
            g->family_->root = detach_gpu_grad_slot(g->storage_);
        }
        register_gpu_grad_owner(*g->family_, this, weak_from_this());
        auto& gpu = std::get<GpuStorage>(slot);
        const std::shared_ptr<::mlx::core::array> raw_arr = gpu.arr;
        const std::shared_ptr<void> record(nullptr, GpuGradSlot{raw_arr, g, handle, g->family_});
        gpu.arr = std::shared_ptr<::mlx::core::array>(record, raw_arr.get());
    }
    set_grad_storage(std::move(slot));
}

std::shared_ptr<TensorImpl> TensorImpl::gpu_grad_to_tensor(GpuStorage& gpu) const {
    auto* owned = std::get_deleter<GpuGradSlot>(gpu.arr);
    if (owned != nullptr && owned->family && owned->family->root.has_value()) {
        if (auto h = owned->handle.lock()) {
            if (h->family_ == owned->family && h->shape() == meta_.shape &&
                h->dtype() == meta_.dtype)
                return h;
        }
        auto h = std::make_shared<TensorImpl>(gpu_placeholder(meta_.dtype), meta_.shape,
                                              meta_.dtype, meta_.device, false);
        if (const auto& src = owned->source) {
            h->meta_.stride = src->stride();
            h->offset_ = src->storage_offset();
            h->shared_ = src->shared_;
        }
        h->family_ = owned->family;
        {
            std::lock_guard<std::mutex> lock(owned->family->mu);
            owned->family->members.push_back(h);
        }
        owned->handle = h;
        return h;
    }
    const std::shared_ptr<::mlx::core::array> raw_arr = owned != nullptr ? owned->arr : gpu.arr;
    GpuStorage root_gpu = gpu;
    root_gpu.arr = raw_arr;
    auto fam = std::make_shared<ViewFamily>();
    fam->nbytes = gpu.nbytes;
    fam->root = Storage{std::move(root_gpu)};
    fam->grad_owners.push_back(weak_from_this());
    auto h = std::make_shared<TensorImpl>(gpu_placeholder(meta_.dtype), meta_.shape, meta_.dtype,
                                          meta_.device, false);
    h->family_ = fam;
    fam->members.push_back(h);
    const std::shared_ptr<void> record(nullptr, GpuGradSlot{raw_arr, nullptr, h, fam});
    gpu.arr = std::shared_ptr<::mlx::core::array>(record, raw_arr.get());
    return h;
}

void TensorImpl::sync_gpu_grad_slot(GpuStorage& slot,
                                    std::shared_ptr<::mlx::core::array> next_arr) {
    auto* owned = slot.arr ? std::get_deleter<GpuGradSlot>(slot.arr) : nullptr;
    if (!owned) {
        slot.arr = std::move(next_arr);
        return;
    }
    const auto family = owned->family;
    const auto source = owned->source;
    const auto handle = owned->handle;
    if (family && family->root.has_value()) {
        const auto h = handle.lock();
        update_gpu_grad_root(*family, h ? h.get() : source.get(), next_arr);
        family->version.fetch_add(1, std::memory_order_relaxed);
        std::vector<std::shared_ptr<TensorImpl>> members;
        {
            std::lock_guard<std::mutex> lock(family->mu);
            for (const auto& w : family->members)
                if (auto m = w.lock())
                    members.push_back(std::move(m));
        }
        for (const auto& m : members) {
            {
                std::lock_guard<std::mutex> lock(m->packed_mu_);
                m->packed_.reset();
                m->packed_root_.reset();
            }
            if (const auto* g = std::get_if<GpuStorage>(&m->storage_); g && g->arr)
                m->storage_ = gpu_placeholder(m->meta_.dtype);
        }
        rebind_gpu_grad_slot(family, std::get<GpuStorage>(*family->root).arr);
        if (auto* refreshed = std::get_deleter<GpuGradSlot>(slot.arr);
            refreshed && refreshed->arr == next_arr)
            return;
    }
    const std::shared_ptr<void> record(nullptr, GpuGradSlot{next_arr, source, handle, family});
    slot.arr = std::shared_ptr<::mlx::core::array>(record, next_arr.get());
}

Storage TensorImpl::detach_gpu_grad_slot(Storage s) {
    if (auto* gpu = std::get_if<GpuStorage>(&s); gpu && gpu->arr) {
        if (auto* owned = std::get_deleter<GpuGradSlot>(gpu->arr))
            gpu->arr = owned->arr;
    } else if (auto* cpu = std::get_if<CpuStorage>(&s); cpu && cpu->ptr) {
        std::byte* addr = cpu->ptr.get();
        if (const auto* ref = std::get_deleter<GradSlotRef>(cpu->ptr); ref && ref->slot)
            cpu->ptr = std::shared_ptr<std::byte[]>(ref->slot, addr);
        if (const auto* owned = std::get_deleter<GradSlot>(cpu->ptr); owned && owned->buffer)
            cpu->ptr = std::shared_ptr<std::byte[]>(owned->buffer, addr);
    }
    return s;
}

std::string
TensorImpl::to_string(int precision, std::size_t threshold, std::size_t edgeitems) const {
    // Snapshot data on CPU regardless of device.  All formatting then
    // reads through a packed contiguous buffer with canonical stride.
    std::vector<std::byte> buf;
    std::visit(overloaded{
                   [&](const CpuStorage& s) {
                       buf = contig_snapshot_cpu(s, meta_.shape, meta_.stride, offset_);
                   },
                   [&](const GpuStorage& g) {
                       CpuStorage cpu = gpu::download_gpu_to_cpu(g, meta_.shape);
                       buf = contig_snapshot_cpu(
                           cpu, meta_.shape,
                           contiguous_stride(meta_.shape, dtype_size(meta_.dtype)), 0);
                   },
                   [&](const SharedStorage& sh) {
                       CpuStorage v = sh.cpu_view();
                       buf = contig_snapshot_cpu(v, meta_.shape, meta_.stride, offset_);
                   },
                   [&](const PhantomStorage&) { refuse_phantom_read("to_string"); },
               },
               readable());

    if (meta_.shape.empty()) {
        // 0-d scalar — render the single element with no brackets.
        std::ostringstream os;
        format_element(buf.data(), meta_.dtype, precision, os);
        return os.str();
    }

    // Compute byte-strides for the canonical contiguous layout we just
    // packed into ``buf``.
    const std::size_t elem = dtype_size(meta_.dtype);
    std::vector<std::int64_t> contig_stride_bytes(meta_.shape.size(), 0);
    if (!meta_.shape.empty()) {
        contig_stride_bytes.back() = static_cast<std::int64_t>(elem);
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(meta_.shape.size()) - 2; i >= 0; --i) {
            contig_stride_bytes[i] = contig_stride_bytes[i + 1] * meta_.shape[i + 1];
        }
    }

    const bool truncate = numel() > threshold;
    std::ostringstream os;
    render_nested(buf.data(), meta_.shape, 0, 0, contig_stride_bytes, meta_.dtype, precision,
                  truncate, edgeitems, os);
    return os.str();
}

namespace {

// Lays ``other``'s values, read as ``src_storage``, into ``dst_storage``, a
// whole buffer on ``dst_device``: copy_from's cases by storage kind, once
// the callers have handled a view, a Metal family and a phantom.
void copy_storage_values(Storage& dst_storage,
                         const Storage& src_storage,
                         const TensorImpl& other,
                         Device dst_device) {
    std::visit(overloaded{
                   [&](CpuStorage& dst, const CpuStorage& src) {
                       // A view source is read through its geometry, not from
                       // its buffer's first byte.
                       if (!other.is_dense()) {
                           const auto packed = contig_snapshot_cpu(
                               src, other.shape(), other.stride(), other.storage_offset());
                           if (packed.size() != dst.nbytes)
                               ErrorBuilder("copy_from").fail("nbytes mismatch");
                           if (!packed.empty())
                               std::memcpy(dst.ptr.get(), packed.data(), packed.size());
                           return;
                       }
                       if (dst.nbytes != src.nbytes) {
                           ErrorBuilder("copy_from").fail("nbytes mismatch");
                       }
                       if (dst.nbytes > 0) {
                           std::memcpy(dst.ptr.get(), src.ptr.get(), dst.nbytes);
                       }
                   },
                   [&](GpuStorage& dst, const GpuStorage& src) {
                       if (dst.nbytes != src.nbytes) {
                           ErrorBuilder("copy_from").fail("nbytes mismatch (GPU)");
                       }
                       if (!src.arr) {
                           ErrorBuilder("copy_from").fail("src GPU array is null");
                       }
                       // mlx::core::copy forces evaluation and produces a new
                       // independent array node; wrap_mlx_array converts it
                       // back into a GpuStorage with the correct dtype tag.
                       auto cloned = ::mlx::core::copy(*src.arr);
                       dst.arr = gpu::wrap_mlx_array(std::move(cloned), dst.dtype).arr;
                   },

                   [&](SharedStorage& dst, const SharedStorage& src) {
                       if (dst.nbytes != src.nbytes)
                           ErrorBuilder("copy_from").fail("nbytes mismatch (SharedStorage)");
                       if (dst.nbytes > 0)
                           std::memcpy(dst.cpu_ptr, src.cpu_ptr, dst.nbytes);
                   },
                   [&](SharedStorage& dst, const CpuStorage& src) {
                       if (dst.nbytes != src.nbytes)
                           ErrorBuilder("copy_from").fail("nbytes mismatch (Shared←CPU)");
                       if (dst.nbytes > 0)
                           std::memcpy(dst.cpu_ptr, src.ptr.get(), dst.nbytes);
                   },
                   [&](CpuStorage& dst, const SharedStorage& src) {
                       if (dst.nbytes != src.nbytes)
                           ErrorBuilder("copy_from").fail("nbytes mismatch (CPU←Shared)");
                       if (dst.nbytes > 0)
                           std::memcpy(dst.ptr.get(), src.cpu_ptr, dst.nbytes);
                   },
                   // Catch-all for cross-device variants that should never be
                   // reached given the device check above.
                   [&](auto&, auto&) {
                       throw DeviceMismatch(std::string(device_name(dst_device)),
                                            std::string(device_name(other.device())),
                                            "copy_from (storage variant)");
                   },
               },
               dst_storage, src_storage);
}

}  // namespace

// Copies element data from other into this tensor.
// Preconditions: same device, dtype, and shape (enforced via exceptions).
// Supported storage combinations: CPU←CPU, GPU←GPU, Shared←Shared,
// Shared←CPU, CPU←Shared.  Any other cross-variant combination throws a
// DeviceMismatch error.
void TensorImpl::copy_from(const TensorImpl& other) {
    copy_from_named(other, "copy_from");
}

void TensorImpl::copy_from_named(const TensorImpl& other, const char* name) {
    if (other.device() != device())
        throw DeviceMismatch(std::string(device_name(meta_.device)),
                             std::string(device_name(other.device())), name);
    if (other.dtype() != dtype())
        throw DtypeMismatch(std::string(dtype_name(meta_.dtype)),
                            std::string(dtype_name(other.dtype())), name);
    if (other.shape() != shape())
        throw ShapeMismatch(meta_.shape, other.shape(), name);
    // Before the trace learns of the write: a refused one never happened.
    if (absorbs_phantom_write(storage_, other.raw_storage(), name))
        return bump_version();
    note_inplace_write(*this, other);
    // A Metal family member writes into the root that every member reads.
    if (gpu_member())
        return write_gpu_member(other, name, /*raw=*/true);
    // A CPU view is written through its geometry: the source, packed, laid
    // into this tensor's own elements at its offset and strides.  The copy
    // below would land at the buffer's first byte.
    if (storage_is_cpu(storage_) && storage_is_cpu(other.storage_) && !is_dense()) {
        refuse_ambiguous_write(*this, name);
        const auto packed = contig_snapshot_cpu(std::get<CpuStorage>(other.storage_), other.shape(),
                                                other.stride(), other.storage_offset());
        settle_before_host_write(shared_, storage_);
        scatter_contig_to_strided(packed.data(), std::get<CpuStorage>(storage_), meta_.shape,
                                  meta_.stride, offset_);
        bump_version();
        return;
    }
    // A shared buffer is written in place so every alias sees the copy; the
    // GPU branch below would hand this tensor a fresh array instead.
    if (write_into_shared(other)) {
        bump_version();
        return;
    }
    // A CPU destination below is written in place, its shared buffer too.
    settle_before_host_write(shared_, storage_);
    copy_storage_values(storage_, other.readable(), other, meta_.device);

    bump_version();
}

void TensorImpl::assign_from(const TensorImpl& other, const char* name) {
    // is_leaf(), not !grad_fn(): a used leaf carries its accumulator as
    // grad_fn (see inplace::refuse_on_leaf).
    if (GradMode::is_enabled() && requires_grad() && is_leaf())
        ErrorBuilder(name).fail("a leaf tensor that requires grad cannot be assigned in place — "
                                "wrap the call in no_grad, or build a new tensor");
    refuse_custom_function_output_write(*this, name);
    // A write through a detached alias (``.data``) is untracked by design:
    // the values change and autograd is not told, as with the reference's
    // ``.data`` — refused only when the source carries a graph, which such
    // an alias cannot take on.  Anywhere else the copy moves the values of
    // every view, while the graph bookkeeping below moves only this tensor;
    // re-deriving the views belongs to the in-place ops
    // (inplace::rebase_views), which this layer cannot reach.  While
    // autograd records, refuse rather than leave them sending their gradient
    // to values the write replaced.
    if (GradMode::is_enabled() && is_aliased()) {
        const bool source_graph = other.grad_fn() || other.requires_grad();
        bool refuse = source_graph;
        if (!is_detached_alias()) {
            refuse_write_to_leaf_view(*this, name);
            bool involved = source_graph || requires_grad() || grad_fn();
            bool views = false;
            for (const auto& m : live_views()) {
                if (!m->rederived_by_family_writes())
                    continue;
                views = true;
                involved = involved || m->requires_grad() || m->grad_fn();
            }
            refuse = views && involved;
        }
        if (refuse)
            ErrorBuilder(name).not_implemented(
                "assigning into a tensor that shares storage with a live view is not supported "
                "while autograd records it — clone() first, or assign under no_grad()");
    }
    copy_from_named(other, name);
    if (!GradMode::is_enabled())
        return;
    if (other.grad_fn()) {
        set_grad_fn(other.grad_fn());
        set_grad_output_nr(other.grad_output_nr());
        set_requires_grad(true);
    } else if (grad_fn()) {
        set_grad_fn(nullptr);
        set_grad_output_nr(0);
        set_requires_grad(false);
    }
}

void TensorImpl::zero_grad() {
    if (autograd_) {
        autograd_->grad.reset();
        autograd_->grad_impl.reset();
    }
}

void TensorImpl::eval() const {
    if (meta_.device != Device::GPU || is_phantom())
        return;  // CPU: no-op, and a phantom has no graph to run
    const auto& gpu_st = std::get<GpuStorage>(readable());
    if (gpu_st.arr)
        gpu_st.arr->eval();  // mlx::core::array::eval()
}

bool TensorImpl::storage_is_shared() const noexcept {
    // Only CpuStorage can be aliased through views; GPU arrays are never
    // directly shared at this level.
    if (const auto* cpu = std::get_if<CpuStorage>(&storage_))
        return cpu->ptr.use_count() > 1;
    return false;
}

// Creates a view TensorImpl that shares storage with base but has an
// independent shape, stride, and byte offset.
//
// The view starts with no autograd state of its own — make_view is a
// metadata operation, and copying base's flags made a reshape under
// no_grad report requires_grad.  A caller that makes a differentiable view
// wires its grad_fn, as reshape does; one that makes a detached alias sets
// the flags it wants, as clone_with_grad does.
//
// The view joins base's view family (see ViewFamily) unless the caller asks
// it not to.  On Metal the family keeps the buffer's value as its root, and
// the view reads and writes through it.  A Metal view kept out of the family
// is a value of its own: base's elements under the view's geometry.  A view
// of a phantom joins no family: a family shares values and the version of
// writes to them, and a phantom has no values for a write to change.
std::shared_ptr<TensorImpl> TensorImpl::make_view(const std::shared_ptr<TensorImpl>& base,
                                                  Shape shape,
                                                  Stride stride,
                                                  std::size_t offset_bytes,
                                                  bool join_family) {
    return make_dtype_view(base, base->meta_.dtype, std::move(shape), std::move(stride),
                           offset_bytes, join_family);
}

std::shared_ptr<TensorImpl> TensorImpl::make_dtype_view(const std::shared_ptr<TensorImpl>& base,
                                                        Dtype dtype,
                                                        Shape shape,
                                                        Stride stride,
                                                        std::size_t offset_bytes,
                                                        bool join_family) {
    refuse_phantom_outside_mode(base->storage_, "a view of a phantom");
    // offset_ accumulates: a view of a view has the combined byte offset from
    // the original allocation's start.
    const std::size_t offset = base->offset_ + offset_bytes;
    if (base->gpu_member() && !join_family) {
        const auto& root = std::get<GpuStorage>(*base->family_->root);
        ::mlx::core::array elems =
            gpu_strided::read(*root.arr, shape, stride, offset, gpu::to_mlx_dtype(dtype));
        auto view =
            std::make_shared<TensorImpl>(Storage{gpu::wrap_mlx_array(std::move(elems), dtype)},
                                         std::move(shape), dtype, base->meta_.device, false);
        if (root_is_shared_buffer(base->shared_, root, root.dtype))
            view->shared_ = base->shared_;
        return view;
    }
    const bool gpu_family = join_family && storage_is_gpu(base->raw_storage());
    Storage view_storage = gpu_family ? gpu_placeholder(dtype) : base->storage_;
    if (auto* cpu = std::get_if<CpuStorage>(&view_storage)) {
        cpu->dtype = dtype;
    } else if (auto* ph = std::get_if<PhantomStorage>(&view_storage)) {
        ph->dtype = dtype;
        ph->nbytes = base->allocation_nbytes();
    }
    auto view = std::make_shared<TensorImpl>(std::move(view_storage), std::move(shape), dtype,
                                             base->meta_.device, false);
    view->meta_.stride = std::move(stride);
    view->offset_ = offset;
    // A view of a shared buffer reads the same bytes, so it shares the
    // buffer's version counter and keeps its MLX alias pinned.
    // metal_shared() still answers no unless the view is the whole buffer.
    view->shared_ = base->shared_;

    if (gpu_family || (join_family && storage_is_cpu(base->storage_)))
        add_to_family(base, view);
    return view;
}

void TensorImpl::add_to_family(const std::shared_ptr<TensorImpl>& base,
                               const std::shared_ptr<TensorImpl>& view) {
    init_family_base(base, base->family_, base->autograd_, base->storage_);
    view->family_ = base->family_;
    // Every leaf-eligible base on the way, not just this one: the views in
    // between are often temporaries (``w[0:2][0]``), and the leaf behind them
    // still takes the write.  Share the parent chain instead of copying a
    // vector so an n-view chain takes O(1) time and memory per view.
    if (!base->is_detached_alias()) {
        auto parent = base->view_lineage_;
        while (parent) {
            const auto m = parent->self.lock();
            if (m && !m->is_detached_alias() && m->is_leaf())
                break;
            parent = parent->parent;
        }
        if (base->is_leaf()) {
            view->view_lineage_ =
                std::make_shared<ViewLineage>(ViewLineage{base, std::move(parent)});
        } else {
            view->view_lineage_ = std::move(parent);
        }
    }
    std::lock_guard<std::mutex> lock(view->family_->mu);
    auto& members = view->family_->members;
    if (members.size() >= view->family_->sweep_threshold) {
        members.erase(
            std::remove_if(members.begin(), members.end(),
                           [](const std::weak_ptr<TensorImpl>& w) { return w.expired(); }),
            members.end());
        view->family_->sweep_threshold = std::max<std::size_t>(64, members.size() * 2);
    }
    members.push_back(view);
}

// is_leaf(), not a missing grad_fn: a leaf used in a graph holds an
// AccumulateGrad.
bool TensorImpl::views_grad_leaf() const {
    for (auto cur = view_lineage_; cur; cur = cur->parent) {
        const auto m = cur->self.lock();
        if (m && !m->is_detached_alias() && m->requires_grad() && m->is_leaf())
            return true;
    }
    return false;
}

std::shared_ptr<TensorImpl> TensorImpl::make_block_view(const std::shared_ptr<TensorImpl>& base,
                                                        Shape shape,
                                                        std::size_t byte_offset) {
    refuse_phantom_outside_mode(base->storage_, "a view of a phantom");
    const std::size_t elem = dtype_size(base->meta_.dtype);
    const std::size_t nbytes = shape_numel(shape) * elem;
    if (byte_offset + nbytes > storage_nbytes(base->storage_))
        ErrorBuilder("make_block_view").fail("the run reaches past the end of the buffer");
    // A run of a phantom is a phantom of the run's size.  It stays out of
    // the base's family, as every phantom view does (make_view): a family
    // shares values and their version, and a phantom has no values.
    if (base->is_phantom())
        return std::make_shared<TensorImpl>(
            phantom_storage(shape, base->meta_.dtype, base->meta_.device), std::move(shape),
            base->meta_.dtype, base->meta_.device, false);
    Stride stride = contiguous_stride(shape, elem);
    auto view = make_view(base, std::move(shape), std::move(stride), byte_offset);
    auto& cpu = std::get<CpuStorage>(view->storage_);
    const std::size_t alloc_nbytes =
        cpu.allocation_nbytes != 0 ? cpu.allocation_nbytes : cpu.nbytes;
    const std::size_t alloc_offset = cpu.offset_in_allocation + view->offset_;
    cpu = CpuStorage(std::shared_ptr<std::byte[]>(cpu.ptr, cpu.ptr.get() + view->offset_), nbytes,
                     cpu.dtype, cpu.version, alloc_offset, alloc_nbytes);
    view->offset_ = 0;
    return view;
}

std::shared_ptr<TensorImpl>
TensorImpl::make_allocation_view(const std::shared_ptr<TensorImpl>& base,
                                 Shape shape,
                                 Stride stride,
                                 std::size_t allocation_byte_offset) {
    auto view = make_view(base, std::move(shape), std::move(stride), 0);
    view->offset_ = allocation_byte_offset;
    if (auto* ph = std::get_if<PhantomStorage>(&view->storage_)) {
        ph->nbytes = base->allocation_nbytes();
    } else if (auto* cpu = std::get_if<CpuStorage>(&view->storage_)) {
        const std::size_t alloc_nbytes = base->allocation_nbytes();
        *cpu = CpuStorage(
            std::shared_ptr<std::byte[]>(cpu->ptr, cpu->ptr.get() - cpu->offset_in_allocation),
            alloc_nbytes, cpu->dtype, cpu->version, 0, alloc_nbytes);
    }
    return view;
}

// Accumulate a graph-mode gradient into this leaf's grad_impl slot.
// First arrival: stores the incoming TensorImpl directly.
// Subsequent arrivals: add via add_op so the result remains differentiable.
bool TensorImpl::absorbs_phantom_gradient(const Storage& grad) {
    if (!absorbs_phantom_write(storage_, grad, "a gradient"))
        return false;
    auto* ag = ensure_autograd();
    if (!ag->grad.has_value() && !ag->grad_impl)
        ag->grad = phantom_storage(meta_.shape, meta_.dtype, meta_.device);
    return true;
}

extern std::shared_ptr<TensorImpl> add_op(const std::shared_ptr<TensorImpl>&,
                                          const std::shared_ptr<TensorImpl>&);

namespace {

bool tracks_autograd(const std::shared_ptr<TensorImpl>& t) {
    return t && (t->requires_grad() || t->grad_fn());
}

std::shared_ptr<TensorImpl> zero_grad_like(const TensorImpl& g) {
    const std::size_t n = g.nbytes();
    if (g.device() == Device::GPU) {
        auto arr = ::mlx::core::zeros(gpu_strided::to_mlx(g.shape()), gpu::to_mlx_dtype(g.dtype()));
        return std::make_shared<TensorImpl>(Storage{gpu::wrap_mlx_array(std::move(arr), g.dtype())},
                                            g.shape(), g.dtype(), Device::GPU, false);
    }
    CpuStorage cpu;
    cpu.ptr = allocate_aligned_bytes(n);
    cpu.nbytes = n;
    cpu.dtype = g.dtype();
    if (n > 0)
        std::memset(cpu.ptr.get(), 0, n);
    return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, g.shape(), g.dtype(), Device::CPU,
                                        false);
}

std::shared_ptr<TensorImpl> clone_grad_impl(std::shared_ptr<TensorImpl> g) {
    if (!g || g->is_phantom())
        return g;
    if (!tracks_autograd(g)) {
        auto out = zero_grad_like(*g);
        out->copy_from(*g);
        return out;
    }
    const bool prev_mode = GradMode::is_enabled();
    GradMode::set_enabled(true);
    try {
        auto out = add_op(g, zero_grad_like(*g));
        GradMode::set_enabled(prev_mode);
        return out;
    } catch (...) {
        GradMode::set_enabled(prev_mode);
        throw;
    }
}

void merge_grad_impl(std::shared_ptr<TensorImpl>& cur, const std::shared_ptr<TensorImpl>& g) {
    const bool g_tracks = tracks_autograd(g);
    const bool prev_mode = GradMode::is_enabled();
    if (tracks_autograd(cur) || g_tracks)
        GradMode::set_enabled(true);
    std::shared_ptr<TensorImpl> sum;
    try {
        sum = add_op(cur, g);
        GradMode::set_enabled(prev_mode);
    } catch (...) {
        GradMode::set_enabled(prev_mode);
        throw;
    }
    const bool leaf_grad = cur->is_leaf() && cur->requires_grad();
    if (!g_tracks && !leaf_grad && (!cur->is_aliased() || !tracks_autograd(cur))) {
        cur->copy_from(*sum);
        cur->set_grad_fn(sum->grad_fn());
        cur->set_grad_output_nr(sum->grad_output_nr());
        cur->set_requires_grad(sum->requires_grad());
    } else {
        cur = std::move(sum);
    }
}

}  // namespace

std::shared_ptr<TensorImpl> TensorImpl::data_alias() {
    auto self = shared_from_this();
    auto alias = make_view(self, shape(), stride(), 0);
    alias->set_detached_alias(true);
    return compile::traced_derive("detach", self, std::move(alias));
}

std::shared_ptr<TensorImpl> TensorImpl::clone_with_grad(bool requires_grad) {
    auto self = shared_from_this();
    auto alias = make_view(self, shape(), stride(), 0);
    if (requires_grad)
        alias->set_requires_grad(true);
    return compile::traced_derive("detach", self, std::move(alias));
}

void TensorImpl::detach_inplace() {
    if (auto* trc = compile::current_tracer()) {
        auto self = shared_from_this();
        if (trc->sees(self.get())) {
            trc->on_op_enter("detach", meta_.device, meta_.dtype, meta_.shape);
            trc->on_op_io({self}, self);
        }
    }
    if (autograd_) {
        autograd_->grad_fn.reset();
        autograd_->grad_output_nr = 0;
        autograd_->requires_grad = false;
        autograd_->is_leaf = true;
    }
    detached_alias_ = true;
}

std::shared_ptr<TensorImpl> TensorImpl::custom_function_view(bool tracked) {
    auto self = shared_from_this();
    auto alias = make_view(self, shape(), stride(), 0);
    if (tracked) {
        alias->set_custom_function_output(true);
    } else {
        alias->set_detached_alias(true);
    }
    compile::traced_derive("reshape", self, alias);
    if (auto* trc = compile::current_tracer()) {
        if (trc->sees(alias.get()))
            trc->on_op_attr("shape", alias->shape());
    }
    return alias;
}

Storage TensorImpl::saved_output_storage() {
    if (!is_dense())
        return storage();
    auto* t_cpu = std::get_if<CpuStorage>(&storage_);
    if (!t_cpu || !t_cpu->ptr)
        return storage();
    const std::shared_ptr<std::byte[]> raw_buf = t_cpu->ptr;
    const std::shared_ptr<void> t_holder(nullptr,
                                         [buf = raw_buf](void*) mutable noexcept { buf.reset(); });
    const std::shared_ptr<void> saved_holder(
        nullptr, [buf = raw_buf](void*) mutable noexcept { buf.reset(); });
    t_cpu->ptr = std::shared_ptr<std::byte[]>(t_holder, raw_buf.get());
    for (const auto& m : live_views()) {
        if (auto* m_cpu = std::get_if<CpuStorage>(&m->storage_);
            m_cpu && same_owner(m_cpu->ptr, raw_buf)) {
            m_cpu->ptr = std::shared_ptr<std::byte[]>(t_holder, m_cpu->ptr.get());
        }
    }
    CpuStorage saved = *t_cpu;
    saved.ptr = std::shared_ptr<std::byte[]>(saved_holder, raw_buf.get());
    return Storage{std::move(saved)};
}

std::shared_ptr<VersionCounter> TensorImpl::version_sink() {
    if (family_) {
        std::lock_guard<std::mutex> lock(family_->mu);
        if (!family_->version_sink) {
            family_->version_sink =
                std::make_shared<VersionCounter>(static_cast<std::uint64_t>(version()));
        }
        return family_->version_sink;
    }
    auto* ag = ensure_autograd();
    if (!ag->version_sink) {
        ag->version_sink = std::make_shared<VersionCounter>(static_cast<std::uint64_t>(version()));
    }
    return ag->version_sink;
}

void TensorImpl::accumulate_grad_impl(std::shared_ptr<TensorImpl> g) {
    if (g && absorbs_phantom_gradient(g->raw_storage()))
        return;
    ensure_autograd();
    if (!autograd_->grad_impl && autograd_->grad.has_value()) {
        Storage prev = detach_gpu_grad_slot(std::move(*autograd_->grad));
        autograd_->grad.reset();
        autograd_->grad_impl = std::make_shared<TensorImpl>(std::move(prev), meta_.shape,
                                                            meta_.dtype, meta_.device, false);
    }
    if (!autograd_->grad_impl)
        autograd_->grad_impl = clone_grad_impl(g);
    else
        merge_grad_impl(autograd_->grad_impl, g);
}

}  // namespace lucid
