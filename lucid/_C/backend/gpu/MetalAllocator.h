// lucid/_C/backend/gpu/MetalAllocator.h
//
// Thin C++ interface over MTLDevice buffer allocation.  All buffers produced
// here use MTLResourceStorageModeShared, which means the same physical pages
// are accessible from both the CPU (via cpu_ptr) and from GPU compute kernels
// (via the MTLBuffer handle) without any copy.  This is only possible on Apple
// Silicon due to the unified DRAM architecture.
//
// Ownership model:
//   MetalBuffer is a plain non-owning struct; the caller is responsible for
//   calling deallocate_shared().  OwnedMetalBuffer pairs a MetalBuffer with a
//   shared_ptr whose custom deleter calls deallocate_shared() on last release,
//   enabling ref-counted GPU-CPU shared memory regions.
//
// Note on wrap_existing: uses newBufferWithBytesNoCopy which requires the
// memory to be page-aligned and at least page-size bytes.  Use this only for
// MTLResourceStorageModeShared buffers you already own; MLX-managed arrays
// are NOT guaranteed to be page-aligned and must not be passed here.

#pragma once

#include <cstddef>
#include <memory>

namespace lucid::gpu {

// Plain descriptor for a Metal buffer that backs Lucid's `SharedStorage`.
//
// Holds the raw CPU pointer + opaque MTL handle + size for a Metal
// shared-memory buffer.  This struct does not own anything — lifetime is
// managed externally (see :struct:`OwnedMetalBuffer` for the RAII form).
// On Apple Silicon, ``cpu_ptr`` and ``mtl_handle`` reference the **same
// physical pages**, so CPU writes are immediately visible to GPU kernels
// (and vice versa) without an explicit transfer.
//
// Attributes
// ----------
// cpu_ptr : void*
//     CPU-accessible virtual address of the buffer.  ``nullptr`` if the
//     buffer is uninitialised or has been deallocated.
// mtl_handle : void*
//     Opaque ``CFRetain``'d ``id<MTLBuffer>`` cast to ``void*`` to keep
//     this header free of Objective-C types.  ``nullptr`` when invalid.
// nbytes : std::size_t
//     Allocation size in bytes.  Always ``0`` on an uninitialised value.
//
// Notes
// -----
// Use :func:`make_metal_shared` for everyday allocation — manual
// pairing of :func:`allocate_shared` and :func:`deallocate_shared` is
// only needed for low-level pool code.
//
// See Also
// --------
// :struct:`OwnedMetalBuffer` — RAII wrapper around this descriptor.
struct MetalBuffer {
    void* cpu_ptr;
    void* mtl_handle;
    std::size_t nbytes;
};

// Allocate a new ``MTLResourceStorageModeShared`` buffer of size ``nbytes``.
//
// The returned buffer is backed by unified-memory pages: the CPU pointer
// and the Metal buffer handle reference the same physical bytes, with no
// transfer required between domains.
//
// Parameters
// ----------
// nbytes : std::size_t
//     Allocation size in bytes.  Zero-sized allocations are legal and
//     return a buffer with ``cpu_ptr == nullptr``.
//
// Returns
// -------
// MetalBuffer
//     A populated descriptor on success; a zero-initialised value on
//     allocation failure (e.g. out-of-memory).  Callers should check
//     ``mtl_handle != nullptr`` before use.
//
// Notes
// -----
// The returned buffer must be released with :func:`deallocate_shared`
// to avoid leaking the underlying Metal allocation.
//
// See Also
// --------
// :func:`make_metal_shared` — ref-counted variant.
MetalBuffer allocate_shared(std::size_t nbytes);

// Release the Metal buffer held in ``buf`` and zero all fields.
//
// Drops the ``CFRetain`` on ``buf.mtl_handle`` and clears the
// descriptor in place so callers cannot reuse the dangling pointer
// by accident.  Calling this on a zero-initialised ``MetalBuffer``
// is a no-op (safe).
//
// Parameters
// ----------
// buf : MetalBuffer&
//     The descriptor to release.  Mutated to the zero state.
//
// Notes
// -----
// Marked ``noexcept`` — failures from the underlying Obj-C release
// are silently swallowed because there is no recoverable action.
void deallocate_shared(MetalBuffer& buf) noexcept;

// Wrap an externally-owned, page-aligned CPU pointer in a Metal buffer.
//
// Calls ``newBufferWithBytesNoCopy:`` on the shared MTLDevice, which
// requires the source memory to be page-aligned and at least
// page-size bytes.  No copy is performed — GPU kernels read/write the
// caller's buffer in place.
//
// Parameters
// ----------
// cpu_ptr : void*
//     CPU pointer to wrap.  Must be page-aligned (typically obtained
//     from :func:`posix_memalign` or another Metal allocation).
// nbytes : std::size_t
//     Byte length of the region.  Must be ≥ system page size.
//
// Returns
// -------
// MetalBuffer
//     Descriptor referring to ``cpu_ptr`` with ``mtl_handle`` set
//     to the freshly created ``id<MTLBuffer>``.
//
// Raises
// ------
// std::runtime_error
//     If Metal rejects the wrap (e.g. unaligned pointer, missing
//     MTLDevice).
//
// Warns
// -----
// Do **not** pass MLX-managed array data here — MLX buffers are not
// guaranteed to be page-aligned and ``newBufferWithBytesNoCopy:`` will
// silently produce undefined behaviour.
//
// See Also
// --------
// :func:`allocate_shared` — fresh allocation, no aliasing.
MetalBuffer wrap_existing(void* cpu_ptr, std::size_t nbytes);

// RAII-owned variant of :struct:`MetalBuffer`.
//
// Bundles a freshly allocated :struct:`MetalBuffer` with a
// ``std::shared_ptr`` whose custom deleter calls
// :func:`deallocate_shared` when the last reference drops.  Used by
// ``SharedStorage`` to express ref-counted shared GPU/CPU memory.
//
// Attributes
// ----------
// buf : MetalBuffer
//     The underlying descriptor (still queryable for ``cpu_ptr`` /
//     ``mtl_handle`` / ``nbytes``).
// owner : std::shared_ptr<void>
//     Ref-counted owner that releases ``buf`` on last drop.  Type-
//     erased to ``void`` because the deleter, not the type, carries
//     the cleanup logic.
//
// See Also
// --------
// :func:`make_metal_shared` — the factory that produces these.
struct OwnedMetalBuffer {
    MetalBuffer buf;
    std::shared_ptr<void> owner;
};

// Allocate a shared Metal buffer wrapped in an RAII owner.
//
// Combines :func:`allocate_shared` with a ``shared_ptr`` whose
// destructor releases the allocation.  Preferred entry point for
// ``Storage`` construction code paths that need automatic cleanup.
//
// Parameters
// ----------
// nbytes : std::size_t
//     Allocation size in bytes.
//
// Returns
// -------
// OwnedMetalBuffer
//     Bundle whose ``owner`` keeps ``buf`` alive for as long as any
//     copy of the ``shared_ptr`` survives.
//
// Examples
// --------
// Construct a 1 KiB shared buffer::
//
//     auto owned = lucid::gpu::make_metal_shared(1024);
//     // owned.buf.cpu_ptr ↔ owned.buf.mtl_handle alias same pages.
//
// See Also
// --------
// :func:`allocate_shared` — non-owning variant.
OwnedMetalBuffer make_metal_shared(std::size_t nbytes);

}  // namespace lucid::gpu
