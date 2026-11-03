// lucid/_C/backend/gpu/MlxBridge.h
//
// Conversion helpers between Lucid's Storage types and mlx::core::array.
//
// Critical constraint: mlx::core::allocator::malloc() allocates GPU-private
// pages on Apple Silicon Metal.  The raw data() pointer of an unevaluated
// or GPU-allocated mlx array must NEVER be accessed from the CPU — doing so
// causes SIGBUS on M-series hardware.  The upload_cpu_to_gpu function uses
// mlx::core::copy() (a lazy DRAM copy at ~100 GB/s) to ensure the resulting
// array owns GPU-accessible memory without aliasing the CPU buffer.
//
// GpuStorage holds a shared_ptr<mlx::core::array> so that multiple Storage
// objects can share the same underlying MLX array without copying.  The
// make_tracked helper in the .cpp registers each allocation with MemoryTracker
// so Python-side memory reporting remains accurate.

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>
#include <mlx/ops.h>
#include <mlx/version.h>

#include "../../api.h"
#include "../../core/Dtype.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"

namespace lucid::gpu {

// The ``UPLO`` flag that makes ``mlx::core::linalg::eigh`` read the lower
// triangle, which is what ``IBackend::linalg_eigh`` promises.  MLX before
// 0.32.1 read the triangle opposite the flag it was given
// (ml-explore/mlx#3834).  The engine only loads the libmlx it was built
// against, so the build-time version is the one that decides.
#if MLX_VERSION_NUMERIC >= 32001
inline constexpr const char* k_mlx_eigh_lower = "L";
inline constexpr const char* k_mlx_eigh_upper = "U";
#else
inline constexpr const char* k_mlx_eigh_lower = "U";
inline constexpr const char* k_mlx_eigh_upper = "L";
#endif

inline const char* mlx_eigh_uplo(bool upper) noexcept {
    return upper ? k_mlx_eigh_upper : k_mlx_eigh_lower;
}

// Map a Lucid :enum:`Dtype` to the matching ``mlx::core::Dtype``.
//
// Parameters
// ----------
// dt : Dtype
//     Lucid element type.
//
// Returns
// -------
// mlx::core::Dtype
//     The corresponding MLX dtype enumerator.
//
// Raises
// ------
// not_implemented
//     If ``dt == Dtype::F64`` — Apple's Metal stack has no native
//     64-bit floating-point path, so float64 tensors cannot live on
//     the GPU.  Callers must downcast to float32 first or keep the
//     tensor on the CPU.
//
// See Also
// --------
// :func:`from_mlx_dtype` — inverse direction.
::mlx::core::Dtype to_mlx_dtype(Dtype dt);

// Map an ``mlx::core::Dtype`` back to a Lucid :enum:`Dtype`.
//
// Parameters
// ----------
// dt : mlx::core::Dtype
//     MLX dtype enumerator.
//
// Returns
// -------
// Dtype
//     The corresponding Lucid dtype.
//
// Raises
// ------
// not_implemented
//     For unsigned integer types (``uint8/16/32/64``) and
//     ``bfloat16`` — Lucid does not currently expose Python-visible
//     equivalents.
//
// See Also
// --------
// :func:`to_mlx_dtype` — inverse direction.
Dtype from_mlx_dtype(::mlx::core::Dtype dt);

// Upload a CPU buffer to GPU-private memory via ``mlx::core::copy``.
//
// MLX arrays own GPU-private pages that **cannot alias** a CPU
// pointer (touching the raw ``data()`` from the CPU side raises
// SIGBUS on M-series hardware).  This routine allocates a new MLX
// array of the requested shape and dispatches a lazy
// ``mlx::core::copy`` from the CPU buffer.  The ``cpu`` storage is
// kept alive — via a custom deleter on the source MLX view — until
// the asynchronous copy completes.
//
// Parameters
// ----------
// cpu : const CpuStorage&
//     Source buffer in Accelerate-allocated CPU memory.
// shape : const Shape&
//     Logical shape of the array.  Must match ``cpu`` element count.
//
// Returns
// -------
// GpuStorage
//     A GPU-resident array participating in MemoryTracker accounting.
//
// Notes
// -----
// Copy bandwidth on M-series unified memory is ~100 GB/s, so the
// transfer is rarely the bottleneck — but the call still pays for a
// fresh GPU allocation.  Prefer batching uploads.
//
// See Also
// --------
// :func:`download_gpu_to_cpu` — inverse direction.
// :func:`shared_storage_to_gpu` — zero-copy variant for shared buffers.
LUCID_API GpuStorage upload_cpu_to_gpu(const CpuStorage& cpu, const Shape& shape);

// Adopt a unified-memory ``SharedStorage`` as a ``GpuStorage`` view.
//
// Wraps the underlying ``MTLResourceStorageModeShared`` buffer as a
// fresh MLX array without copying.  The result is readable + writable
// from both CPU and GPU (subject to Metal's memory-model rules).
//
// Parameters
// ----------
// sh : const SharedStorage&
//     Source storage backed by a shared Metal buffer.
// shape : const Shape&
//     Logical shape to view the buffer with.
//
// Returns
// -------
// GpuStorage
//     A non-copying view participating in MemoryTracker accounting.  It
//     shares ``sh.version`` with the CPU view.
//
// Notes
// -----
// When ``sh`` is pinned the result is a copy of ``sh.gpu_alias`` (reshaped
// if ``shape`` differs), so MLX never sees it as the buffer's sole owner and
// cannot donate the buffer to an op's output.  This is the canonical fast
// path for results produced by :func:`run_metal_kernel` that flow back into
// MLX-based ops.
LUCID_API GpuStorage shared_storage_to_gpu(const SharedStorage& sh, const Shape& shape);

// Give a shared buffer its one MLX external array (``sh.gpu_alias``).
//
// No-op when ``sh`` is already pinned or empty.  Every later copy of ``sh``
// shares the array, which is what keeps GPU aliases from being donated.
//
// Parameters
// ----------
// sh : SharedStorage&
//     Descriptor to pin.
// shape : const Shape&
//     Shape of the tensor adopting the buffer.
LUCID_API void pin_shared_storage(SharedStorage& sh, const Shape& shape);

// Download a GPU array to a freshly allocated CPU buffer.
//
// Calls ``arr.eval()`` to force materialisation of any pending lazy
// MLX kernels, then ``memcpy``s the contiguous data into a new
// :class:`CpuStorage` of the requested shape.
//
// Parameters
// ----------
// gpu : const GpuStorage&
//     Source MLX-backed storage on the GPU stream.
// shape : const Shape&
//     Logical shape of the result.
//
// Returns
// -------
// CpuStorage
//     A fresh CPU buffer.  Disjoint from the GPU source — mutations
//     do not propagate either way.
//
// Notes
// -----
// Always synchronous from the caller's perspective.  The cost is
// dominated by the GPU graph completion (the actual ``memcpy`` is
// at ~100 GB/s).
LUCID_API CpuStorage download_gpu_to_cpu(const GpuStorage& gpu, const Shape& shape);

// Wrap a caller-owned ``mlx::core::array`` into a tracked ``GpuStorage``.
//
// Takes ownership of ``arr`` (moves it) and packages it as a
// ``GpuStorage`` registered with the MemoryTracker.  Used by every
// op method in :class:`GpuBackend` to return MLX results back to
// Lucid.
//
// Parameters
// ----------
// arr : mlx::core::array&&
//     The MLX array to wrap.  Moved-from on return.
// dtype : Dtype
//     The Lucid dtype tag to associate with the result (must match
//     ``arr``'s MLX dtype via :func:`from_mlx_dtype`).
//
// Returns
// -------
// GpuStorage
//     The wrapped array tracked by Python-side memory accounting.
GpuStorage wrap_mlx_array(::mlx::core::array&& arr, Dtype dtype);

// Evaluate every GPU array still waiting to run, and return how many.
//
// Lucid's Metal path is lazy: an op records a graph node and returns, and
// nothing runs until a value is read.  ``lucid.metal.synchronize()``
// promises the queued work is done, but MLX keeps no list of unevaluated
// arrays, so :func:`wrap_mlx_array` notes each one it wraps (as a weak
// reference that never keeps it alive) and this runs whichever are still
// alive and unevaluated.
//
// Raises
// ------
// LucidError
//     When an array cannot be evaluated (an allocation past the device's
//     buffer limit, a kernel that refuses its arguments).  The others still
//     run first; then the first failure is raised, its message prefixed
//     with the count of failures.  A ``LucidError`` subclass keeps its
//     class, MLX's ``std::invalid_argument`` becomes ``InvalidArgument``,
//     and any other ``std::exception`` a ``LucidError``.
using EvalRunner = void (*)(std::vector<::mlx::core::array>);

std::size_t evaluate_pending(EvalRunner runner = nullptr);

// Convert a Lucid :class:`Shape` (int64) to an ``mlx::core::Shape`` (int32).
//
// Parameters
// ----------
// shape : const Shape&
//     Source shape vector.
//
// Returns
// -------
// mlx::core::Shape
//     ``std::vector<int32_t>`` view of the same dimensions.
//
// Raises
// ------
// std::overflow_error
//     If any dimension exceeds ``INT32_MAX`` — MLX is currently
//     int32-indexed and cannot represent larger extents.
::mlx::core::Shape to_mlx_shape(const Shape& shape);

// Convert an ``mlx::core::Shape`` to a Lucid :class:`Shape`.
//
// Widens the int32 dimensions to int64 for Lucid's canonical
// representation.
//
// Parameters
// ----------
// shape : const mlx::core::Shape&
//     Source MLX shape.
//
// Returns
// -------
// Shape
//     Lucid ``std::vector<int64_t>`` shape.
inline Shape mlx_shape_to_lucid(const ::mlx::core::Shape& shape) {
    Shape out;
    out.reserve(shape.size());
    for (auto dim : shape)
        out.push_back(static_cast<std::int64_t>(dim));
    return out;
}

// Build a 0-D MLX array holding ``v`` cast to ``dt``.
//
// Used as a broadcastable RHS in arithmetic helpers (e.g. ``x + 1``)
// to avoid materialising a same-shape constant tensor.  The literal
// flows into the kernel inline, paying zero allocation cost.
//
// Parameters
// ----------
// v : double
//     Scalar value to embed.  Cast to ``float`` before MLX construction.
// dt : mlx::core::Dtype
//     Final MLX dtype the scalar should be cast to.
//
// Returns
// -------
// mlx::core::array
//     A rank-0 MLX array of dtype ``dt`` holding ``v``.
//
// Notes
// -----
// The intermediate float cast loses precision for very large
// double-precision values; callers needing exact int64 broadcasts
// should construct the array directly.
inline ::mlx::core::array mlx_scalar(double v, ::mlx::core::Dtype dt) {
    return ::mlx::core::astype(::mlx::core::array(static_cast<float>(v)), dt);
}

}  // namespace lucid::gpu
