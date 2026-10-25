// lucid/_C/ops/gfunc/Gfunc.h
//
// Generator ("gfunc") ops: tensor-creation primitives that produce new
// leaf tensors from scalar parameters or from the metadata of an
// existing tensor.  Includes constant fills, identity / diagonal,
// progressions (arange / linspace / logspace), shape-inheriting
// ``*_like`` variants, scatter-style assemblers, and ``unfold_dim``.
//
// Design invariants
// -----------------
// - None of these ops participate in autograd unless explicitly noted
//   (scatter family).  Pure generators emit leaf nodes whose
//   ``requires_grad`` is taken solely from the argument of that name;
//   they construct no ``FuncOp`` / ``NaryKernel``.
// - CPU storage is allocated via ``Allocator`` (``CpuStorage`` with a
//   managed ``unique_ptr``).  GPU storage uses ``IBackend::full`` /
//   ``IBackend::eye`` / ``IBackend::from_cpu``; the last route uploads
//   a CPU buffer to the GPU without a full backend rewrite.
// - ``arange`` and ``linspace`` materialise the buffer on CPU first
//   and then call ``from_cpu`` — the element computation is a simple
//   scalar loop faster to write in portable C++ than to route through
//   a backend-specific kernel.
// - Shape, dtype, and device arguments are forwarded verbatim to the
//   resulting ``TensorImpl``; no implicit promotion occurs.
//
// These functions are exposed to Python via the ``_bindings/`` layer,
// whose wrappers translate Python objects (list shapes, dtype strings)
// into the C++ types declared here.

#pragma once

#include <cstdint>

#include "../../api.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

namespace backend {
class IBackend;
}  // namespace backend

using ScatterKernel = Storage (backend::IBackend::*)(
    const Storage&, const Storage&, const Storage&, const Shape&, const Shape&, int, Dtype);

// Create a tensor of the given shape filled with zeros.
//
// Parameters
// ----------
// shape : Shape
//     Desired output shape.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
// requires_grad : bool
//     Attach a fresh autograd leaf when true.
//
// Returns
// -------
// TensorImplPtr
//     Leaf tensor of the specified shape, dtype, and device.
//
// See Also
// --------
// zeros_like_op : Shape-inheriting variant.
LUCID_API TensorImplPtr zeros_op(const Shape& shape,
                                 Dtype dt,
                                 Device device,
                                 bool requires_grad = false);

// Create a tensor of the given shape filled with ones.
//
// Parameters
// ----------
// shape : Shape
//     Desired output shape.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
// requires_grad : bool
//     Attach a fresh autograd leaf when true.
//
// Returns
// -------
// TensorImplPtr
//     Leaf tensor of the specified shape filled with $1$.
LUCID_API TensorImplPtr ones_op(const Shape& shape,
                                Dtype dt,
                                Device device,
                                bool requires_grad = false);

// Create a tensor of the given shape filled with a constant scalar.
//
// Parameters
// ----------
// shape : Shape
//     Desired output shape.
// fill_value : double
//     Scalar value cast to ``dt`` before filling.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
// requires_grad : bool
//     Attach a fresh autograd leaf when true.
//
// Returns
// -------
// TensorImplPtr
//     Leaf tensor whose every element equals ``fill_value`` (cast).
LUCID_API TensorImplPtr
full_op(const Shape& shape, double fill_value, Dtype dt, Device device, bool requires_grad = false);

// Create an uninitialised tensor of the given shape.
//
// Parameters
// ----------
// shape : Shape
//     Desired output shape.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
// requires_grad : bool
//     Attach a fresh autograd leaf when true.
//
// Returns
// -------
// TensorImplPtr
//     Leaf tensor with unspecified content.
//
// Notes
// -----
// The current implementation zero-fills (via ``make_zero_storage``) to
// avoid UB on uninitialised reads.  The ``empty`` name signals that
// callers must not rely on the values — future implementations may
// switch to a true uninitialised allocator.
LUCID_API TensorImplPtr empty_op(const Shape& shape,
                                 Dtype dt,
                                 Device device,
                                 bool requires_grad = false);

// Create an $N \times M$ identity-like matrix with ones on diagonal $k$.
//
// Math
// ----
// $$
//   \text{out}[i, j] = \begin{cases} 1 & j - i = k \\ 0 & \text{otherwise} \end{cases}
// $$
//
// Parameters
// ----------
// N : int64_t
//     Number of rows (must be $\ge 0$).
// M : int64_t
//     Number of columns (must be $\ge 0$).
// k : int64_t
//     Diagonal index.  ``k = 0`` is the main diagonal; positive ``k``
//     is above, negative ``k`` is below.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
// requires_grad : bool
//     Attach a fresh autograd leaf when true.
//
// Returns
// -------
// TensorImplPtr
//     2-D tensor of shape $(N, M)$.
LUCID_API TensorImplPtr eye_op(std::int64_t N,
                               std::int64_t M,
                               std::int64_t k,
                               Dtype dt,
                               Device device,
                               bool requires_grad = false);

// Create a 1-D tensor with values $[\text{start}, \text{start}+\text{step}, \dots)$
// up to but not including ``stop``.
//
// Math
// ----
// Output length is $\lceil (\text{stop} - \text{start}) / \text{step} \rceil$;
// element $i$ equals $\text{start} + i \cdot \text{step}$.
//
// Parameters
// ----------
// start : double
//     First value of the progression.
// stop : double
//     Exclusive upper bound.
// step : double
//     Constant stride.  Must be non-zero.
// dt : Dtype
//     Element dtype (the values are cast at the end).
// device : Device
//     Target device.
// requires_grad : bool
//     Attach a fresh autograd leaf when true.
//
// Returns
// -------
// TensorImplPtr
//     1-D tensor of length $\lceil (\text{stop} - \text{start}) / \text{step} \rceil$.
//
// Raises
// ------
// LucidError
//     If ``step == 0``.
LUCID_API TensorImplPtr arange_op(
    double start, double stop, double step, Dtype dt, Device device, bool requires_grad = false);

// Create a 1-D tensor with ``num`` linearly spaced values from
// ``start`` to ``stop`` inclusive.
//
// Math
// ----
// Element $i$ equals
// $\text{start} + i \cdot (\text{stop} - \text{start}) / (\text{num} - 1)$;
// the final element is pinned to ``stop`` exactly to avoid floating-point
// drift.
//
// Parameters
// ----------
// start : double
//     First value (inclusive).
// stop : double
//     Last value (inclusive).
// num : int64_t
//     Number of samples (must be $\ge 0$).
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
// requires_grad : bool
//     Attach a fresh autograd leaf when true.
//
// Returns
// -------
// TensorImplPtr
//     1-D tensor of length ``num``.
//
// Raises
// ------
// LucidError
//     If ``num < 0``.
LUCID_API TensorImplPtr linspace_op(double start,
                                    double stop,
                                    std::int64_t num,
                                    Dtype dt,
                                    Device device,
                                    bool requires_grad = false);

// Create a zero-filled tensor matching ``a`` in shape, dtype, and device.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor whose metadata is mirrored.
// requires_grad : bool
//     Attach a fresh autograd leaf when true.
//
// Returns
// -------
// TensorImplPtr
//     New leaf tensor with the same shape/dtype/device as ``a``.
LUCID_API TensorImplPtr zeros_like_op(const TensorImplPtr& a, bool requires_grad = false);

// Create a one-filled tensor matching ``a`` in shape, dtype, and device.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor whose metadata is mirrored.
// requires_grad : bool
//     Attach a fresh autograd leaf when true.
//
// Returns
// -------
// TensorImplPtr
//     New leaf tensor with the same shape/dtype/device as ``a``.
LUCID_API TensorImplPtr ones_like_op(const TensorImplPtr& a, bool requires_grad = false);

// Create an uninitialised tensor matching ``a`` in shape, dtype, and device.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor whose metadata is mirrored.
// requires_grad : bool
//     Attach a fresh autograd leaf when true.
//
// Returns
// -------
// TensorImplPtr
//     New leaf tensor; content must not be relied upon (see ``empty_op``).
LUCID_API TensorImplPtr empty_like_op(const TensorImplPtr& a, bool requires_grad = false);

// Create a constant-filled tensor matching ``a`` in shape, dtype, and device.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Tensor whose metadata is mirrored.
// fill_value : double
//     Scalar value cast to ``a.dtype()`` before filling.
// requires_grad : bool
//     Attach a fresh autograd leaf when true.
//
// Returns
// -------
// TensorImplPtr
//     New leaf tensor with constant content.
LUCID_API TensorImplPtr full_like_op(const TensorImplPtr& a,
                                     double fill_value,
                                     bool requires_grad = false);

// Create a 1-D tensor of ``num`` values logarithmically spaced.
//
// Math
// ----
// Element $i$ equals $\text{base}^{\,t_i}$ where $t_i$ are the values
// produced by ``linspace(start, stop, num)``.
//
// Parameters
// ----------
// start : double
//     Log-domain start.
// stop : double
//     Log-domain stop (inclusive).
// num : int64_t
//     Number of samples (must be $\ge 0$).
// base : double
//     Logarithm base.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
// requires_grad : bool
//     Attach a fresh autograd leaf when true.
//
// Returns
// -------
// TensorImplPtr
//     1-D tensor of length ``num``.
LUCID_API TensorImplPtr logspace_op(double start,
                                    double stop,
                                    std::int64_t num,
                                    double base,
                                    Dtype dt,
                                    Device device,
                                    bool requires_grad = false);

// User-facing scatter-add: ``out = base`` with ``src`` accumulated at
// positions given by ``indices`` along ``dim``.
//
// Math
// ----
// $$
//   \text{out}[i_0, \dots, i_{d-1}, \text{indices}[j], i_{d+1}, \dots]
//   \mathrel{+}= \text{src}[i_0, \dots, j, \dots]
// $$
//
// Parameters
// ----------
// base : TensorImplPtr
//     Tensor providing the initial values; same shape as the output.
// indices : TensorImplPtr
//     int32 or int64 tensor indexing into ``base`` along ``dim``.  Shape
//     matches ``src``.
// src : TensorImplPtr
//     Values to add, in ``base``'s dtype.
// dim : int
//     Axis along which scattering occurs.
//
// Returns
// -------
// TensorImplPtr
//     Same shape and dtype as ``base``.
//
// Raises
// ------
// DtypeMismatch
//     If ``src`` is not in ``base``'s dtype or ``indices`` is not int32 /
//     int64 — the rule of ``scatter_amax_op``, ``scatter_amin_op``,
//     ``scatter_prod_op`` and ``scatter_set_op`` too.  Nothing is cast.
//
// Notes
// -----
// Full autograd support.  Gradients: $\partial / \partial \text{base} = \text{grad\_out}$,
// $\partial / \partial \text{src} = \text{gather}(\text{grad\_out}, \text{dim}, \text{indices})$.
LUCID_API TensorImplPtr scatter_add_op(const TensorImplPtr& base,
                                       const TensorImplPtr& indices,
                                       const TensorImplPtr& src,
                                       int dim);

// Scatter with element-wise ``max`` reduction at duplicate indices.
//
// Math
// ----
// At each output slot $i$, the result is the maximum among ``base[i]``
// and all ``src[j]`` with ``indices[j] == i``.
//
// Parameters
// ----------
// base, indices, src, dim
//     Same semantics as ``scatter_add_op``.
//
// Returns
// -------
// TensorImplPtr
//     Reduced tensor, same shape and dtype as ``base``.
//
// Notes
// -----
// Backward flows only through "winning" elements:
//   $\partial / \partial \text{src}[j] = \text{grad}[\text{idx}[j]]$ when $\text{src}[j] = \text{out}[\text{idx}[j]]$, else $0$;
//   $\partial / \partial \text{base}[i] = \text{grad}[i]$ when $\text{base}[i] = \text{out}[i]$, else $0$.
// Ties distribute the gradient equally to all winners (matches the
// reference framework).
//
// See Also
// --------
// scatter_amin_op : Same pattern with ``min`` reduction.
LUCID_API TensorImplPtr scatter_amax_op(const TensorImplPtr& base,
                                        const TensorImplPtr& indices,
                                        const TensorImplPtr& src,
                                        int dim);

// Scatter with element-wise ``min`` reduction at duplicate indices.
//
// Math
// ----
// At each output slot $i$, the result is the minimum among ``base[i]``
// and all ``src[j]`` with ``indices[j] == i``.
//
// Parameters
// ----------
// base, indices, src, dim
//     Same semantics as ``scatter_add_op``.
//
// Returns
// -------
// TensorImplPtr
//     Reduced tensor, same shape and dtype as ``base``.
//
// Notes
// -----
// Backward identical to ``scatter_amax_op`` with the winner predicate
// flipped to "matches the output minimum".
//
// See Also
// --------
// scatter_amax_op : Companion ``max`` variant.
LUCID_API TensorImplPtr scatter_amin_op(const TensorImplPtr& base,
                                        const TensorImplPtr& indices,
                                        const TensorImplPtr& src,
                                        int dim);

// Scatter with multiplicative reduction.
//
// Math
// ----
// $$
//   \text{out}[i] = \text{base}[i] \cdot \prod_{j : \text{idx}[j] = i} \text{src}[j] .
// $$
//
// Parameters
// ----------
// base, indices, src, dim
//     Same semantics as ``scatter_add_op``.
//
// Returns
// -------
// TensorImplPtr
//     Same shape and dtype as ``base``.
//
// Notes
// -----
// Each operand's gradient is ``grad`` at its position times the product of
// the other operands there — ``out / operand`` where the operand is not
// zero, and computed without that division where it is, so a zero operand
// gets the others' product rather than NaN.
LUCID_API TensorImplPtr scatter_prod_op(const TensorImplPtr& base,
                                        const TensorImplPtr& indices,
                                        const TensorImplPtr& src,
                                        int dim);

// scatter_set: out[..., index[i], ...] = src[..., i, ...] along dim (overwrite).
// A copy of ``base`` with the indexed slices replaced by ``src`` — the single-op
// primitive that ``index_copy`` routes through.
//
// Parameters
// ----------
// base, indices, src, dim
//     Same semantics as ``scatter_add_op``.
//
// Returns
// -------
// TensorImplPtr
//     Same shape and dtype as ``base``.
//
// Notes
// -----
// Gradients:
//   $\partial / \partial \text{base}[i] = \text{grad}[i]$ except at overwritten
//   positions where it is 0; $\partial / \partial \text{src}[j] =
//   \text{grad}[\text{idx}[j]]$ (a plain gather).  Duplicate indices follow the
//   reference framework's last-writer-wins convention.
LUCID_API TensorImplPtr scatter_set_op(const TensorImplPtr& base,
                                       const TensorImplPtr& indices,
                                       const TensorImplPtr& src,
                                       int dim);

// Validated and normalised operands for an axis-scatter op.
struct AxisScatterOperands {
    int dim;                // normalised axis in [0, base->ndim)
    TensorImplPtr indices;  // reshaped to empty base slice when indices->numel() == 0
    TensorImplPtr values;   // src cut to the corner the index covers
};

// Validate device, dtypes, axis, and shapes for an axis-scatter op, returning
// the normalised axis, index tensor, and narrowed source slice.
AxisScatterOperands axis_scatter_operands(const char* op,
                                          const TensorImplPtr& base,
                                          const TensorImplPtr& indices,
                                          const TensorImplPtr& src,
                                          int dim);

// Run a backend axis-scatter kernel under ``OpScopeFull`` and record trace I/O.
TensorImplPtr dispatch_axis_scatter(const char* scope_name,
                                    ScatterKernel kernel,
                                    const TensorImplPtr& base,
                                    const AxisScatterOperands& ops);

// Sliding-window view of ``a`` along a single dimension.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// dim : int
//     Axis along which to unfold.
// size : int
//     Window width.
// step : int
//     Stride between successive windows.
//
// Returns
// -------
// TensorImplPtr
//     Tensor whose shape is
//     ``(*in_shape[:dim], L, *in_shape[dim+1:], size)``
//     where $L = \lfloor (\text{in\_shape}[\text{dim}] - \text{size}) / \text{step} \rfloor + 1$.
//
// Shape
// -----
// Adds a trailing ``size``-sized axis; the original ``dim`` is replaced
// by the number of windows ``L``.
//
// Notes
// -----
// Behaves as a view; backward passes through a ``gather`` that
// accumulates overlapping windows back into the input.
LUCID_API TensorImplPtr unfold_dim_op(const TensorImplPtr& a, int dim, int size, int step);

}  // namespace lucid
