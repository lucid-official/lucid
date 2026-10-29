// lucid/_C/core/Allocator.h
//
// Aligned, pooled byte allocator for Lucid's CPU tensor storage.
//
// All host-side tensor data goes through :func:`allocate_aligned_bytes`
// rather than ``new`` / ``malloc`` for two reasons:
//
//   1. **Alignment** — Apple Accelerate BLAS and vDSP routines require
//      16- or 64-byte aligned buffers for maximum throughput.  Lucid
//      pins the alignment to :data:`kCpuAlignment` (64 B), which both
//      satisfies the Accelerate requirement and matches a typical ARM64
//      cache line.
//   2. **Pooling** — The implementation in ``Allocator.cpp`` maintains a
//      thread-local slab pool (23 power-of-two size classes, up to 4 MB,
//      ``kMaxDepth = 32`` blocks per class).  Returning a buffer to the
//      pool avoids the kernel round-trip of ``posix_memalign`` / ``free``,
//      which is a significant cost for the short-lived intermediate
//      tensors produced by ML workloads.
//
// MLX GPU allocations (``GpuStorage``) are managed by the MLX runtime
// and must not be served through this allocator — they live in
// GPU-private memory and would raise a ``SIGBUS`` if accessed via a CPU
// pointer.  The :class:`Device` parameter on :func:`allocate_aligned_bytes`
// exists primarily to (a) update the correct :class:`MemoryTracker`
// counter bank and (b) opt out of the pool path for non-CPU devices.
//
// See Also
// --------
// :class:`MemoryTracker` — receives ``track_alloc`` / ``track_free``
//     hooks installed by this allocator.
// :class:`Storage`       — owner of the ``shared_ptr<std::byte[]>`` block
//     returned here.

#pragma once

#include <cstddef>
#include <memory>

#include "Device.h"

namespace lucid {

// Required alignment, in bytes, for every CPU tensor buffer.
//
// Equal to one ARM64 / x86-64 cache line, and satisfies the minimum
// alignment expected by Accelerate vDSP and BLAS routines.  The value
// is also used as the smallest size-class boundary inside the pool —
// every pooled block is therefore both ``kCpuAlignment``-aligned and a
// multiple of ``kCpuAlignment`` in capacity.
//
// Notes
// -----
// Changing this constant requires re-validating the pool size-class
// table in ``Allocator.cpp`` (``kMinClass`` is hard-coded to equal
// ``kCpuAlignment``).
constexpr std::size_t kCpuAlignment = 64;

// Allocates an aligned, optionally-pooled byte buffer of ``nbytes``.
//
// The returned ``shared_ptr<std::byte[]>`` owns the block via a custom
// deleter that either (a) returns the block to the thread-local pool —
// for CPU allocations up to 4 MB whose class still has free slots — or
// (b) calls ``std::free`` directly, for large allocations, full pool
// slots, and non-CPU devices.  In both paths the deleter also fires the
// matching :func:`MemoryTracker::track_free` hook.
//
// On every successful allocation the matching
// :func:`MemoryTracker::track_alloc` hook is fired, so per-device
// counters stay in sync with the live byte total.
//
// Parameters
// ----------
// nbytes : std::size_t
//     Number of bytes to allocate.  A value of ``0`` returns an empty
//     ``shared_ptr`` (no allocation, no tracker update).
// device : Device, optional
//     Device tag controlling (a) which :class:`MemoryTracker` counter
//     bank is updated and (b) whether the pool path is taken — only
//     ``Device::CPU`` is poolable.  Defaults to :attr:`Device::CPU`.
//
// Returns
// -------
// std::shared_ptr<std::byte[]>
//     Owning handle to a ``kCpuAlignment``-aligned byte buffer at least
//     ``nbytes`` long.  Empty when ``nbytes == 0``.  The capacity may
//     exceed ``nbytes`` for pooled allocations (rounded up to the next
//     power of two ≥ ``kCpuAlignment``).
//
// Raises
// ------
// OutOfMemory
//     ``posix_memalign`` failed, or ``nbytes`` exceeds the machine's
//     physical memory.  The second is refused before anything is reserved:
//     macOS hands out address space for far more than it has (a 4 TiB
//     ``posix_memalign`` succeeds) and fails only when the pages are
//     touched, by killing the process.  The exception carries the requested
//     size, the current live bytes, and the historical peak for the
//     target device so the caller can format an informative message.
//
// Notes
// -----
// The pool is **thread-local**, so allocation and deallocation are
// lock-free on the fast path.  Cached blocks are released when the
// owning thread exits.  Blocks freed on a different thread than the
// one that allocated them still work correctly, but they will be
// returned to the freeing thread's pool — long-running thread-pool
// workloads with cross-thread tensor ownership may therefore observe
// slight imbalance in retained free-list size.
//
// Examples
// --------
// ::
//
//     auto block = allocate_aligned_bytes(1024);                // CPU, pooled
//     auto huge  = allocate_aligned_bytes(64 << 20);            // CPU, bypass pool
//     auto gpu   = allocate_aligned_bytes(N, Device::GPU);      // GPU, tracker only
std::shared_ptr<std::byte[]> allocate_aligned_bytes(std::size_t nbytes,
                                                    Device device = Device::CPU);

}  // namespace lucid
