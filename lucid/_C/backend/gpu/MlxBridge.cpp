// lucid/_C/backend/gpu/MlxBridge.cpp
//
// Implements the Storage ↔ mlx::core::array bridge functions declared in
// MlxBridge.h.
//
// upload_cpu_to_gpu design:
//   A transient mlx::core::array is constructed that points at the CPU buffer
//   via a non-owning "external" view.  A shared_ptr keepalive token holds the
//   original CpuStorage ptr so it cannot be freed until after mlx::core::copy()
//   has finished reading it.  copy() returns a new array backed by GPU-private
//   memory that is safe to use from Metal compute kernels.
//
// download_gpu_to_cpu design:
//   Wraps the array in mlx::core::contiguous() before eval() so the underlying
//   buffer's byte order matches the logical shape — lazy transposes (e.g. the
//   NHWC↔NCHW perm at the conv bridge) leave stride metadata pointing at a
//   permuted view while the raw buffer is still in the source layout, and
//   array::data<T>() returns those raw bytes ignoring strides.  contiguous()
//   is the standard MLX layout-fixup primitive and is a cheap pass-through
//   when the input is already row-contiguous.  After eval() on the
//   materialised array, data<uint8_t>() yields shape-major bytes safe to
//   memcpy into a fresh aligned CPU buffer.  After eval() the data pointer
//   is guaranteed CPU-accessible on Apple Silicon unified-memory hardware
//   only when the array is not on a pure GPU stream.
//
// make_tracked:
//   Wraps a raw mlx::core::array* in a shared_ptr with a custom deleter that
//   notifies MemoryTracker on deallocation.  All GpuStorage objects are created
//   via this helper so memory accounting is consistent.

#include "MlxBridge.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <mlx/allocator.h>
#include <mlx/device.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/MemoryStats.h"
#include "MetalAllocator.h"
#include "SharedBufferWrites.h"

namespace lucid::gpu {

::mlx::core::Dtype to_mlx_dtype(Dtype dt) {
    switch (dt) {
    case Dtype::Bool:
        return ::mlx::core::bool_;
    case Dtype::I8:
        return ::mlx::core::int8;
    case Dtype::I16:
        return ::mlx::core::int16;
    case Dtype::I32:
        return ::mlx::core::int32;
    case Dtype::I64:
        return ::mlx::core::int64;
    case Dtype::F16:
        return ::mlx::core::float16;
    case Dtype::F32:
        return ::mlx::core::float32;
    case Dtype::F64:

        ErrorBuilder("to_mlx_dtype")
            .not_implemented("float64 is not supported on GPU (MLX-Metal limitation). "
                             "Cast to float32 first, or keep the tensor on CPU.");
    case Dtype::C128:
        ErrorBuilder("to_mlx_dtype")
            .not_implemented("complex128 is not supported on GPU (MLX has no complex128). "
                             "Cast to complex64 first, or keep the tensor on CPU. "
                             "linalg.eig on an f64 matrix produces one — it is the only "
                             "dtype wide enough to hold its eigenvalues.");
    case Dtype::C64:
        return ::mlx::core::complex64;
    case Dtype::BF16:
        // MLX carries bfloat16 natively, so the GPU side needs no
        // emulation — unlike the CPU, where Accelerate has no bf16 kernels
        // and every gate widens to float.
        return ::mlx::core::bfloat16;
    }
    ErrorBuilder("to_mlx_dtype").fail("unknown Dtype");
}

Dtype from_mlx_dtype(::mlx::core::Dtype dt) {
    using V = ::mlx::core::Dtype::Val;
    switch (dt.val()) {
    case V::bool_:
        return Dtype::Bool;
    case V::int8:
        return Dtype::I8;
    case V::int16:
        return Dtype::I16;
    case V::int32:
        return Dtype::I32;
    case V::int64:
        return Dtype::I64;
    case V::float16:
        return Dtype::F16;
    case V::bfloat16:
        return Dtype::BF16;
    case V::float32:
        return Dtype::F32;
    case V::float64:
        return Dtype::F64;
    case V::complex64:
        return Dtype::C64;
    default:
        ErrorBuilder("from_mlx_dtype")
            .not_implemented("unsupported MLX dtype (uint*/bfloat16 not yet wired into Lucid)");
    }
}

::mlx::core::Shape to_mlx_shape(const Shape& shape) {
    ::mlx::core::Shape out;
    out.reserve(shape.size());
    for (auto d : shape) {
        if (d > std::numeric_limits<::mlx::core::ShapeElem>::max()) {
            ErrorBuilder("to_mlx_shape")
                .fail("dimension exceeds int32 range (MLX requires shape elements fit in int32_t)");
        }
        out.push_back(static_cast<::mlx::core::ShapeElem>(d));
    }
    return out;
}

namespace {

// Creates a ref-counted mlx::core::array pointer that notifies MemoryTracker
// when the last reference is dropped.  bytes == 0 disables tracking (used for
// empty arrays returned from shape-only operations).
// ``MTLDevice.maxBufferLength``, read once: the largest buffer Metal hands
// out.  MLX refuses a larger one only when the array is evaluated — at a
// sync, perhaps many ops later — as a ``RuntimeError`` that names no op.
std::size_t max_buffer_bytes() {
    static const std::size_t bytes = [] {
        const auto& info = ::mlx::core::device_info(::mlx::core::Device(::mlx::core::Device::gpu));
        const auto it = info.find("max_buffer_length");
        const auto* v = it == info.end() ? nullptr : std::get_if<std::size_t>(&it->second);
        return v ? *v : std::numeric_limits<std::size_t>::max();
    }();
    return bytes;
}

// Refuses a GPU array no Metal buffer can hold, when the op that sized it
// asks rather than when the graph runs.
void refuse_past_max_buffer(std::size_t bytes) {
    if (bytes <= max_buffer_bytes()) [[likely]]
        return;
    const auto s = MemoryTracker::get_stats(Device::GPU);
    OutOfMemory err(bytes, s.current_bytes, s.peak_bytes, std::string(device_name(Device::GPU)));
    err.prefix_message("one Metal buffer cannot be larger than the device's " +
                       std::to_string(max_buffer_bytes()) + " bytes (maxBufferLength): ");
    throw err;
}

std::shared_ptr<::mlx::core::array> make_tracked(::mlx::core::array* raw, std::size_t bytes) {
    if (bytes > 0) {
        MemoryTracker::track_alloc(bytes, Device::GPU);
    }
    return std::shared_ptr<::mlx::core::array>(raw, [bytes](::mlx::core::array* p) {
        delete p;
        if (bytes > 0) {
            MemoryTracker::track_free(bytes, Device::GPU);
        }
    });
}
}  // namespace

GpuStorage upload_cpu_to_gpu(const CpuStorage& cpu, const Shape& shape) {
    refuse_past_max_buffer(cpu.nbytes);
    auto mlx_shape = to_mlx_shape(shape);
    auto mlx_dt = to_mlx_dtype(cpu.dtype);

    auto keepalive = std::make_shared<std::shared_ptr<std::byte[]>>(cpu.ptr);
    void* raw = static_cast<void*>(cpu.ptr.get());
    ::mlx::core::array external(raw, std::move(mlx_shape), mlx_dt,
                                [keepalive](void*) mutable { keepalive.reset(); });

    auto owned = ::mlx::core::copy(external);

    GpuStorage out;
    out.dtype = cpu.dtype;
    out.nbytes = cpu.nbytes;
    out.arr = make_tracked(new ::mlx::core::array(std::move(owned)), out.nbytes);
    return out;
}

CpuStorage download_gpu_to_cpu(const GpuStorage& gpu, const Shape& shape) {
    if (!gpu.arr) {
        ErrorBuilder("download_gpu_to_cpu").fail("null GPU array");
    }

    // Empty output (e.g. an empty-input/empty-result matmul): nothing to copy.
    // Short-circuit BEFORE touching MLX — realizing a 0-element array
    // (``contiguous()`` + ``eval()`` below) SIGSEGVs on Metal, an uncatchable
    // crash on ``.numpy()`` / repr of such a tensor.  The empty CpuStorage here
    // is identical to what the post-eval path produced.
    const std::size_t total = shape_numel(shape) * dtype_size(gpu.dtype);
    CpuStorage out;
    out.dtype = gpu.dtype;
    out.nbytes = total;
    if (total == 0) {
        return out;
    }

    // Force shape-major byte layout before reading.  array::data<T>() returns
    // bytes in the buffer's underlying layout and ignores stride metadata, so
    // a lazily-transposed array (e.g. the deferred NHWC→NCHW perm in
    // conv_nd_forward) would otherwise memcpy NHWC bytes out of a tensor
    // whose Lucid metadata advertises NCHW.  contiguous() is a no-op pass-
    // through on already row-contiguous inputs (cheap on the common path) and
    // materialises a properly-laid-out copy for strided views.
    ::mlx::core::array materialised = ::mlx::core::contiguous(*gpu.arr);
    materialised.eval();

    out.ptr = allocate_aligned_bytes(total, Device::CPU);
    MemoryTracker::track_host_sync(total);
    const auto* src = materialised.data<std::uint8_t>();
    std::memcpy(out.ptr.get(), src, total);
    return out;
}

namespace {

// Arrays that may still be unevaluated graphs, for evaluate_pending(), and
// arrays created while a Metal shared buffer is pinned, which an MLX op
// (copy, contiguous, reshape, slice, single-input cat/stack, unit-axis sum)
// may evaluate by aliasing the shared buffer's Data via copy_shared_buffer.
// Weak references, pruned whenever a list doubles, so tracking neither keeps
// an array alive nor grows without bound.
constexpr std::size_t kPruneFloor = 4096;
std::mutex g_pending_mu;
std::vector<std::weak_ptr<::mlx::core::array>> g_pending;
std::size_t g_prune_at = kPruneFloor;
std::vector<std::weak_ptr<::mlx::core::array>> g_shared_candidates;
std::size_t g_candidates_prune_at = kPruneFloor;
std::atomic<std::size_t> g_pinned_count{0};

bool is_pending(const ::mlx::core::array& arr) {
    return arr.status() == ::mlx::core::array::Status::unscheduled;
}

void note_pending(const std::shared_ptr<::mlx::core::array>& arr) {
    if (!arr)
        return;
    const bool pending = is_pending(*arr);
    const bool pinned = g_pinned_count.load(std::memory_order_relaxed) > 0;
    if (!pending && !pinned)
        return;
    std::lock_guard<std::mutex> lock(g_pending_mu);
    if (pending) {
        g_pending.push_back(arr);
        if (g_pending.size() >= g_prune_at) {
            std::erase_if(g_pending, [](const std::weak_ptr<::mlx::core::array>& weak) {
                const auto held = weak.lock();
                return !held || !is_pending(*held);
            });
            g_prune_at = std::max(kPruneFloor, 2 * g_pending.size());
        }
    }
    if (pinned) {
        g_shared_candidates.push_back(arr);
        if (g_shared_candidates.size() >= g_candidates_prune_at) {
            std::erase_if(g_shared_candidates, [](const std::weak_ptr<::mlx::core::array>& weak) {
                const auto held = weak.lock();
                return !held || (!is_pending(*held) && !held->data_shared_ptr());
            });
            g_candidates_prune_at = std::max(kPruneFloor, 2 * g_shared_candidates.size());
        }
    } else if (!g_shared_candidates.empty()) {
        g_shared_candidates.clear();
        g_candidates_prune_at = kPruneFloor;
    }
}

}  // namespace

GpuStorage wrap_mlx_array(::mlx::core::array&& arr, Dtype dtype) {
    // Reconcile, never relabel.
    //
    // MLX promotes on its own: ``mean`` of an int32 array comes back
    // float32, and this used to record the caller's dtype next to the
    // array's own byte count.  Nothing failed — the storage simply
    // claimed to be int32 while holding float32 bits, and every later
    // reader reinterpreted them.  ``avg_pool2d`` on int32 answered
    // 1083179008 for a window whose mean is 4.5, which is 0x40900000:
    // the float read as an integer.  A wrong answer that raises nothing
    // is worse than the refusal it replaced, and only a dtype sweep was
    // ever in a position to see it.
    //
    // The array's dtype is the fact; the parameter is the request.  When
    // they differ the values are converted, so the bytes match the label
    // that goes with them.
    const auto want = to_mlx_dtype(dtype);
    ::mlx::core::array held = arr.dtype() == want ? std::move(arr) : ::mlx::core::astype(arr, want);
    // An empty result is replaced by a fresh empty array, dropping the graph
    // that produced it.  This works around an MLX bug: ``contiguous()`` over
    // an empty ``matmul`` or ``fast::scaled_dot_product_attention`` result
    // segfaults in ``Contiguous::eval_gpu`` — and this backend wraps almost
    // every result as ``contiguous(op(...))``.  Minimal repro (MLX 0.32):
    //
    //     mx.eval(mx.contiguous(mx.matmul(mx.zeros((0, 4)), mx.zeros((4, 3)))))
    //
    // dies with SIGSEGV, while the same matmul without ``contiguous`` (and
    // ``contiguous`` of a plain ``zeros((0, 3))``) evaluates.  In Lucid it
    // killed the interpreter on a matmul gradient with an empty extent and
    // on an attention with no queries.  An array with no elements is fully
    // described by its shape and dtype, and nothing can be written into or
    // read out of it, so the substitute is indistinguishable.  Costs one
    // size comparison on every other result.
    if (held.size() == 0)
        held = ::mlx::core::zeros(held.shape(), held.dtype());
    else if (g_pinned_count.load(std::memory_order_relaxed) > 0 && !is_pending(held))
        // An MLX op that passed an evaluated input through unchanged
        // (``contiguous`` on a row-contiguous shared buffer) shares its
        // ArrayDesc with the input; ``copy`` gives the result a descriptor of
        // its own so detaching it before a host write leaves the input intact.
        held = ::mlx::core::copy(held);
    GpuStorage out;
    out.dtype = dtype;
    out.nbytes = held.nbytes();
    refuse_past_max_buffer(out.nbytes);
    out.arr = make_tracked(new ::mlx::core::array(std::move(held)), out.nbytes);
    note_pending(out.arr);
    return out;
}

namespace {

// ``first`` again, the first of ``failed`` arrays out of ``total`` whose
// evaluation threw, with the count in front of its message.  A Lucid error
// keeps its class; what MLX threw (an allocation past the device's buffer
// limit, a kernel that refused its arguments) becomes a ``LucidError`` —
// ``InvalidArgument`` for MLX's argument refusals, as the binding
// translator maps them everywhere else.
[[noreturn]] void
raise_eval_failure(const std::exception_ptr& first, std::size_t failed, std::size_t total) {
    const std::string what = "metal synchronize: " + std::to_string(failed) + " of " +
                             std::to_string(total) +
                             " pending GPU arrays failed to evaluate; the first raised: ";
    try {
        std::rethrow_exception(first);
    } catch (LucidError& e) {
        e.prefix_message(what);
        throw;
    } catch (const std::invalid_argument& e) {
        throw InvalidArgument(what + e.what());
    } catch (const std::exception& e) {
        throw LucidError(what + e.what());
    }
}

// Run each array on its own, so one graph that cannot run does not keep the
// others from completing, and raise the first failure once all have had
// their turn.  Rerunning an array the batch already evaluated is a no-op.
void evaluate_each(std::vector<::mlx::core::array>& arrays) {
    std::exception_ptr first;
    std::size_t failed = 0;
    for (auto& arr : arrays) {
        try {
            arr.eval();
        } catch (const std::exception&) {
            if (!first)
                first = std::current_exception();
            ++failed;
        }
    }
    if (first)
        raise_eval_failure(first, failed, arrays.size());
}

}  // namespace

std::size_t evaluate_pending() {
    std::vector<::mlx::core::array> arrays;
    {
        std::lock_guard<std::mutex> lock(g_pending_mu);
        for (const auto& weak : g_pending)
            if (auto arr = weak.lock(); arr && is_pending(*arr))
                arrays.push_back(*arr);
        g_pending.clear();
        g_prune_at = kPruneFloor;
    }
    if (arrays.empty())
        return 0;
    try {
        ::mlx::core::eval(arrays);
    } catch (const std::exception&) {
        // One graph that cannot run fails the whole batch and leaves the
        // rest unrun.  The batch's error is not raised here: the per-array
        // pass meets the same failure and raises it with the count, and a
        // batch too large as a whole that runs piece by piece is no error.
        evaluate_each(arrays);
    }
    return arrays.size();
}

namespace {

// The noted arrays that are still pending, dropping the rest from the list.
std::vector<std::shared_ptr<::mlx::core::array>> live_pending() {
    std::vector<std::shared_ptr<::mlx::core::array>> live;
    std::lock_guard<std::mutex> lock(g_pending_mu);
    std::erase_if(g_pending, [&live](const std::weak_ptr<::mlx::core::array>& weak) {
        auto held = weak.lock();
        if (!held || !is_pending(*held))
            return true;
        live.push_back(std::move(held));
        return false;
    });
    return live;
}

using ReadsMemo = std::unordered_map<std::uintptr_t, bool>;

// Graph nodes the settles may inspect between two points where nothing is
// pending, before one stops telling readers from the rest and runs every
// pending array instead.  Shared by the settles, not given to each: pending
// work that reads nothing stays pending and would be searched again by every
// write, so a run of writes over the same graph — or a graph that grows
// between them — spends this much in all and then runs it, which leaves the
// next write nothing to search.  About a millisecond of walking.
constexpr std::size_t kSettleWalkBudget = 4096;
std::atomic<std::size_t> g_walk_credit{kSettleWalkBudget};

// Whether ``root``'s graph reads the bytes ``buf_ptr`` points to: an array
// that has run reads them when its buffer is ``buf_ptr`` (the pinned alias,
// a DLPack import of the same Metal buffer, or anything MLX evaluated over
// either via copy_shared_buffer), a pending one when any input does.  Empty
// once the walk has spent ``budget``.  Iterative, because a lazy chain can be
// deeper than the C++ stack.  Frames hold handles, not pointers into a
// parent's inputs, which another thread's eval may clear.
std::optional<bool> graph_reads(const ::mlx::core::array& root,
                                const void* buf_ptr,
                                ReadsMemo& memo,
                                std::size_t& budget) {
    struct Frame {
        ::mlx::core::array arr;
        std::size_t next = 0;
    };
    std::vector<Frame> stack;
    stack.push_back({root});
    while (!stack.empty()) {
        Frame& top = stack.back();
        if (top.next == 0 && memo.contains(top.arr.id())) {
            stack.pop_back();
            continue;
        }
        if (top.next == 0 && budget-- == 0)
            return std::nullopt;
        if (top.arr.status() != ::mlx::core::array::Status::unscheduled) {
            const bool same_buf =
                top.arr.data_shared_ptr() != nullptr && top.arr.buffer().ptr() == buf_ptr;
            memo.emplace(top.arr.id(), same_buf);
            stack.pop_back();
            continue;
        }
        const auto& inputs = top.arr.inputs();
        while (top.next < inputs.size() && memo.contains(inputs[top.next].id()))
            ++top.next;
        if (top.next < inputs.size()) {
            ::mlx::core::array child = inputs[top.next++];
            stack.push_back({std::move(child)});
            continue;
        }
        const bool reads = std::any_of(inputs.begin(), inputs.end(),
                                       [&memo](const auto& in) { return memo.at(in.id()); });
        memo.emplace(top.arr.id(), reads);
        stack.pop_back();
    }
    return memo.at(root.id());
}

// The pending arrays that read ``sh``, or all of them once the walk credit
// (kSettleWalkBudget) runs out.
std::vector<::mlx::core::array> buffer_readers(const SharedStorage& sh) {
    std::vector<::mlx::core::array> readers;
    const auto pending = live_pending();
    if (pending.empty()) {
        g_walk_credit.store(kSettleWalkBudget, std::memory_order_relaxed);
        return readers;
    }
    // Without the pinned alias no tensor has handed the buffer to MLX.
    if (!sh.gpu_alias || !sh.gpu_alias->data_shared_ptr())
        return readers;
    const void* buf_ptr = sh.gpu_alias->buffer().ptr();
    // No reserve(): a power-of-two bucket count makes libc++ mask the
    // identity hash of aligned pointers into a few buckets.
    ReadsMemo memo;
    std::size_t budget = g_walk_credit.load(std::memory_order_relaxed);
    for (const auto& arr : pending) {
        const auto reads = graph_reads(*arr, buf_ptr, memo, budget);
        if (!reads) {
            // Everything runs now, so nothing is left to search.
            g_walk_credit.store(kSettleWalkBudget, std::memory_order_relaxed);
            readers.clear();
            for (const auto& each : pending)
                readers.push_back(*each);
            return readers;
        }
        if (*reads)
            readers.push_back(*arr);
    }
    g_walk_credit.store(budget, std::memory_order_relaxed);
    return readers;
}

// Rebind every evaluated op result that MLX aliased onto ``sh``'s buffer
// (via ``copy_shared_buffer``) to a fresh snapshot of ``sh``'s pre-write
// bytes, so the host write that follows does not mutate independent tensors.
void detach_buffer_aliases(const SharedStorage& sh) {
    if (!sh.gpu_alias || !sh.gpu_alias->data_shared_ptr() || !sh.cpu_ptr || sh.nbytes == 0)
        return;
    const void* buf_ptr = sh.gpu_alias->buffer().ptr();
    if (!buf_ptr)
        return;
    const auto pin_id = sh.gpu_alias->id();
    const bool sole_pinned = g_pinned_count.load(std::memory_order_relaxed) <= 1;
    std::vector<std::shared_ptr<::mlx::core::array>> aliases;
    {
        std::lock_guard<std::mutex> lock(g_pending_mu);
        std::erase_if(g_shared_candidates, [&](const std::weak_ptr<::mlx::core::array>& weak) {
            auto held = weak.lock();
            if (!held || held->id() == pin_id)
                return true;
            if (is_pending(*held))
                return false;
            if (held->data_shared_ptr() && held->buffer().ptr() == buf_ptr) {
                aliases.push_back(std::move(held));
                return true;
            }
            return sole_pinned || !held->data_shared_ptr();
        });
    }
    if (aliases.empty())
        return;
    ::mlx::core::allocator::Buffer fresh = ::mlx::core::allocator::malloc(sh.nbytes);
    std::memcpy(fresh.raw_ptr(), sh.cpu_ptr, sh.nbytes);
    ::mlx::core::array snap(fresh, sh.gpu_alias->shape(), sh.gpu_alias->dtype());
    for (const auto& arr : aliases) {
        const auto elem_off = arr->offset() / static_cast<std::int64_t>(arr->itemsize());
        arr->copy_shared_buffer(snap, arr->strides(), arr->flags(), arr->data_size(), elem_off);
    }
}

}  // namespace

void settle_buffer_readers(const SharedStorage& sh, std::vector<::mlx::core::array> with) {
    std::vector<::mlx::core::array> run = buffer_readers(sh);
    run.insert(run.end(), std::make_move_iterator(with.begin()),
               std::make_move_iterator(with.end()));
    if (!run.empty())
        ::mlx::core::eval(run);
    // Work already submitted may still be reading the buffer as well.
    ::mlx::core::synchronize();
    detach_buffer_aliases(sh);
}

namespace {

// An MLX array over the shared buffer's own bytes.  The deleter holds the
// Metal allocation, so the buffer outlives every array — and every graph
// node — that reads it.
::mlx::core::array external_over(const SharedStorage& sh, const Shape& shape) {
    auto owner_tok = sh.owner;
    return ::mlx::core::array(
        sh.cpu_ptr, to_mlx_shape(shape), to_mlx_dtype(sh.dtype),
        [owner_tok = std::move(owner_tok)](void*) mutable { owner_tok.reset(); });
}

}  // namespace

void pin_shared_storage(SharedStorage& sh, const Shape& shape) {
    // float64 and complex128 have no MLX dtype, so no GPU alias of them can
    // exist to be donated — and building one would refuse a tensor that only
    // ever lives on the CPU side.
    if (sh.gpu_alias || !sh.cpu_ptr || sh.nbytes == 0 || sh.dtype == Dtype::F64 ||
        sh.dtype == Dtype::C128)
        return;
    g_pinned_count.fetch_add(1, std::memory_order_relaxed);
    sh.gpu_alias = std::shared_ptr<::mlx::core::array>(
        new ::mlx::core::array(external_over(sh, shape)), [](::mlx::core::array* p) noexcept {
            g_pinned_count.fetch_sub(1, std::memory_order_relaxed);
            delete p;
        });
}

GpuStorage shared_storage_to_gpu(const SharedStorage& sh, const Shape& shape) {
    if (!sh.cpu_ptr || sh.nbytes == 0)
        ErrorBuilder("shared_storage_to_gpu").fail("SharedStorage is empty");

    // A copy of the pinned alias shares its array descriptor, so no alias is
    // ever the buffer's sole owner.  MLX donates an input whose descriptor
    // has one owner, and a donated shared buffer receives the op's output:
    // ``(s.to("metal") * 2)`` would have written the product into ``s``.
    // A descriptor that never went through TensorImpl is not pinned and
    // gets a fresh external array, as before.
    ::mlx::core::array alias = sh.gpu_alias ? *sh.gpu_alias : external_over(sh, shape);
    auto mlx_shape = to_mlx_shape(shape);
    if (alias.shape() != mlx_shape)
        alias = ::mlx::core::reshape(alias, std::move(mlx_shape));

    GpuStorage out;
    out.dtype = sh.dtype;
    out.nbytes = sh.nbytes;
    // The CPU view shares this counter too (``cpu_view``), so a write
    // through either alias is seen by autograd on both.
    out.version = sh.version;
    out.arr = make_tracked(new ::mlx::core::array(std::move(alias)), out.nbytes);
    return out;
}

}  // namespace lucid::gpu
