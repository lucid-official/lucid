// lucid/_C/core/Helpers.h
//
// Lightweight inline factory utilities for op kernel implementations.
//
// These helpers live in the :namespace:`lucid::helpers` sub-namespace
// (rather than directly under :namespace:`lucid`) so call sites are
// self-documenting at a glance — ``helpers::allocate_cpu(...)`` and
// ``helpers::fresh(...)`` clearly mark "this is boilerplate-reduction
// scaffolding, not part of the public API surface".
//
// Both functions exist to remove a recurring two-step pattern from op
// kernels: "allocate a zeroed CPU buffer of size ``numel * dtype_size``
// and wrap it in a :class:`TensorImpl`".  Inlining keeps the
// abstraction free at runtime — they compile to the same code as the
// hand-rolled equivalents.
//
// Notes
// -----
// Helpers are intentionally restricted to the CPU side and to plain
// :class:`Storage` wrapping.  GPU allocation goes through MLX-backed
// pathways elsewhere; lifecycle-sensitive constructs (autograd,
// views, gradient tracking) are out of scope and must be set up
// explicitly by the caller.
//
// See Also
// --------
// :class:`CpuStorage` — POD struct produced by :func:`allocate_cpu`.
// :class:`TensorImpl` — concrete tensor wrapper produced by
//     :func:`fresh`.

#pragma once

#include <cstring>
#include <memory>
#include <utility>

#include "../api.h"
#include "Allocator.h"
#include "ResultExtent.h"
#include "Shape.h"
#include "Storage.h"
#include "TensorImpl.h"
#include "fwd.h"

namespace lucid::helpers {

// Allocates a zero-initialised CPU buffer sized for ``shape`` elements
// of dtype ``dt`` and wraps it as a :class:`CpuStorage`.
//
// Parameters
// ----------
// shape : const Shape&
//     Logical shape of the tensor that will use the buffer.  Only the
//     element count (:func:`shape_numel`) influences allocation size.
// dt : Dtype
//     Element dtype.  Determines per-element size via
//     :func:`dtype_size`.
//
// Returns
// -------
// CpuStorage
//     Newly allocated storage with ``dtype = dt``,
//     ``nbytes = shape_numel(shape) * dtype_size(dt)``, and a 64-byte
//     aligned (``kCpuAlignment``) pointer to a zeroed buffer.
//     ``nbytes == 0`` (empty shape or any dim of 0) yields a storage
//     whose ``ptr`` is null — callers must check for this case.
//
// Notes
// -----
// The buffer is zeroed via :func:`std::memset`; allocation uses
// :func:`allocate_aligned_bytes` so the pointer satisfies SIMD-friendly
// alignment requirements expected by Apple Accelerate kernels.
//
// Zero-byte allocations short-circuit the memset and return a storage
// with a null ``ptr`` and ``nbytes == 0``.  This matches the
// convention used elsewhere in the engine for empty tensors.
//
// Examples
// --------
// Allocate a ``(3, 4)`` ``float32`` buffer::
//
//     auto storage = helpers::allocate_cpu({3, 4}, Dtype::Float32);
//     // storage.nbytes == 3 * 4 * 4 == 48; ptr is 64-byte aligned and
//     // memset-zeroed.
//
// See Also
// --------
// :func:`allocate_aligned_bytes` — the underlying aligned allocator.
// :func:`helpers::fresh` — wraps a Storage into a TensorImpl.
inline CpuStorage allocate_cpu(const Shape& shape, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = shape_numel(shape) * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    if (s.nbytes > 0)
        std::memset(s.ptr.get(), 0, s.nbytes);
    return s;
}

// Wraps an existing :class:`Storage` and tensor geometry into a new
// :class:`TensorImpl` with autograd disabled.
//
// Parameters
// ----------
// s : Storage&&
//     Storage holding the tensor data.  Moved into the new
//     :class:`TensorImpl` — no copy of the underlying buffer occurs.
//     May be CPU, GPU, or Shared storage.
// shape : Shape
//     Logical shape of the resulting tensor.
// dt : Dtype
//     Element dtype.  Must be consistent with the storage's element
//     layout.
// device : Device
//     Device on which the storage resides.  Must match the storage's
//     actual location.
// requires_grad : bool
//     ``true`` for the few forwards that hand back a result already marked
//     as a gradient leaf (an LSTM's outputs, a factory called with
//     ``requires_grad=True``).  Default ``false``.
//
// Returns
// -------
// TensorImplPtr
//     ``std::shared_ptr<TensorImpl>`` to a freshly constructed tensor.
//     The stride is initialised to the C-contiguous default by the
//     :class:`TensorImpl` constructor.
//
// Notes
// -----
// "Fresh" here means newly produced by an op kernel — neither a view
// onto an existing tensor nor an autograd leaf that needs gradient
// tracking.  Callers that want a gradient-tracking output must set
// the autograd metadata after construction.
//
// Storage is moved (rvalue reference), so the input is consumed.
// This matches the kernel pattern where storage and tensor are
// produced in adjacent statements and the storage handle is no
// longer needed after wrapping.
//
// A GPU ``s`` must hold exactly ``shape``'s element count of ``dt`` — a
// storage of another extent raises :class:`LucidError` naming the op
// (:func:`require_result_extent`).  The check lives here, not at each op,
// because this is the one function a forward builds its result through: a
// kernel family and an op with a forward of its own alike.  A view is not
// built here (it shares its parent's buffer on purpose), and a CPU or phantom
// storage is not compared.
//
// Examples
// --------
// Build a kernel output tensor in one statement::
//
//     auto out = helpers::fresh(
//         helpers::allocate_cpu(out_shape, dt),
//         out_shape, dt, Device::CPU);
//
// See Also
// --------
// :func:`helpers::allocate_cpu` — typical Storage producer for the
//     first argument.
// :class:`TensorImpl` — concrete wrapper type returned.
inline TensorImplPtr
fresh(Storage&& s, Shape shape, Dtype dt, Device device, bool requires_grad = false) {
    require_result_extent(s, shape, dt);
    return std::make_shared<TensorImpl>(std::move(s), std::move(shape), dt, device, requires_grad);
}

}  // namespace lucid::helpers
