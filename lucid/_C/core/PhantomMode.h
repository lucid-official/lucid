// lucid/_C/core/PhantomMode.h
//
// Thread-local phantom mode: while it is on, every factory and op on the
// calling thread returns a phantom tensor — a real :class:`TensorImpl` whose
// storage is a :class:`PhantomStorage` — instead of computing anything.
//
// The one source of truth for "build this without allocating": the dispatcher
// routes every backend call to the phantom backend (:file:`PhantomBackend.h`)
// while the mode is on, the kernel bases skip their kernels, and the factories
// that fill a host buffer before they dispatch skip the fill.  Each output
// keeps the shape and dtype its op computes before it dispatches, so no shape
// rule is written twice.  Outside the mode a phantom is inert
// (refuse_phantom_outside_mode), and one rule governs every write that
// involves one (absorbs_phantom_write).
//
// The flag is per thread, as :class:`GradMode`'s is — and so is its contract:
// a thread starts with the mode off, and an asyncio task shares its thread's.
//
// Header-only: the flag is an ``inline thread_local``, so the engine holds one
// per thread whichever translation unit reads it.

#pragma once

#include <atomic>
#include <string>
#include <string_view>

#include "Error.h"
#include "ErrorBuilder.h"
#include "Shape.h"
#include "Storage.h"

namespace lucid {

namespace detail {
inline thread_local bool g_phantom_mode = false;
// How many threads have the mode on.  Every dispatch asks, and on Darwin a
// thread-local read is a call through its TLV descriptor; while no thread
// is in the mode — always, outside model construction — the answer is this
// one plain load.  A thread's own increment is visible to its own later
// reads, so the count never hides that thread's flag from itself.  A thread
// that ends with the mode on leaves the count raised: the others then read
// their own flags too, which costs the TLV call and never a wrong answer.
inline std::atomic<int> g_phantom_threads{0};
}  // namespace detail

// Thread-local switch for phantom tensors, modelled on :class:`GradMode`.
class PhantomMode {
public:
    // Whether factories and ops on the calling thread make phantom tensors.
    static bool is_enabled() noexcept {
        return detail::g_phantom_threads.load(std::memory_order_relaxed) != 0 &&
               detail::g_phantom_mode;
    }

    // Sets the flag for the calling thread only.  Prefer
    // :class:`PhantomModeGuard`, which restores the flag on unwind.
    static void set_enabled(bool value) noexcept {
        bool& mode = detail::g_phantom_mode;
        if (mode == value)
            return;
        mode = value;
        detail::g_phantom_threads.fetch_add(value ? 1 : -1, std::memory_order_relaxed);
    }
};

// RAII guard that sets phantom mode to ``enabled`` for its lifetime on the
// calling thread, and restores the flag it found.
class PhantomModeGuard {
public:
    explicit PhantomModeGuard(bool enabled) noexcept : prev_(PhantomMode::is_enabled()) {
        PhantomMode::set_enabled(enabled);
    }
    ~PhantomModeGuard() { PhantomMode::set_enabled(prev_); }

    PhantomModeGuard(const PhantomModeGuard&) = delete;
    PhantomModeGuard& operator=(const PhantomModeGuard&) = delete;

private:
    bool prev_;
};

// A phantom stands in for a tensor its device could hold, so what the device
// refuses for real it refuses here, with the same message.  Without this every
// shadow-built check is blind to float64 on Metal: the phantom reads as a float64
// Metal tensor, which no real run can make.  Every phantom is made through this,
// by phantom_storage and by PhantomBackend's unsized ones.
inline void refuse_what_device_cannot_hold(Dtype dt, Device device) {
    if (device == Device::GPU && !metal_holds(dt)) [[unlikely]]
        refuse_metal_dtype(dt);
}

// A phantom storage for ``shape`` elements of ``dt`` on ``device``, sized.
inline Storage phantom_storage(const Shape& shape, Dtype dt, Device device) {
    refuse_what_device_cannot_hold(dt, device);
    PhantomStorage p;
    p.nbytes = shape_numel(shape) * dtype_size(dt);
    p.dtype = dt;
    p.device = device;
    return Storage{p};
}

// The one refusal for a phantom tensor's values — a host read (``item``,
// ``tolist``, NumPy, bytes, printing), a data-dependent shape, a write of a
// phantom into a real tensor, and an op on a phantom outside the mode alike:
// there are no values to give.  The op stack the error context holds (every
// op's OpScopeFull pushes its name) says which op asked.
[[noreturn]] inline void refuse_phantom_read(std::string_view op) {
    std::string msg = "phantom tensor has no values: " + std::string(op);
    if (const std::string trace = ErrorContext::trace(); !trace.empty())
        msg += " [trace=" + trace + "]";
    throw LucidError(msg);
}

// The contract for a phantom outside phantom mode: it is inert.  Its shape,
// dtype, device and autograd flags read as any tensor's, and a write into it
// is absorbed (absorbs_phantom_write), but an op that would read its values
// or make a new phantom of it — every op does one or the other — refuses
// here.  A phantom is made only under the mode, so a block's result can never
// reach real code dressed as a real tensor.  Called where those two things
// happen, TensorImpl::storage() and TensorImpl::make_view.
inline void refuse_phantom_outside_mode(const Storage& s, std::string_view what) {
    if (storage_is_phantom(s) && !PhantomMode::is_enabled()) [[unlikely]]
        refuse_phantom_read(std::string(what) + " outside phantom mode");
}

// The one write rule for phantom storage, at every place a write lands —
// TensorImpl's copy_from / write_through / take_storage_from, write_elements
// (an optimizer's or BatchNorm's in-place update), accumulate_into and a leaf's
// first gradient, an assigned ``.grad``.
//
// Returns
// -------
// bool
//     ``true`` when ``dst`` is a phantom: it takes any write and keeps no
//     values, so the caller writes nothing (and moves the version where a
//     write would).  ``false`` for an ordinary write between real storages.
//
// Raises
// ------
// LucidError
//     ``dst`` is real and ``src`` a phantom.  The phantom has no values to
//     give, and taking it in place of the real values would turn a tensor
//     made outside the block into a phantom in silence.
inline bool absorbs_phantom_write(const Storage& dst, const Storage& src, std::string_view op) {
    if (storage_is_phantom(dst))
        return true;
    if (storage_is_phantom(src))
        refuse_phantom_read(op);
    return false;
}

}  // namespace lucid
