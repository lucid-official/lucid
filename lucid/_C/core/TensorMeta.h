// lucid/_C/core/TensorMeta.h
//
// Plain-data metadata carriers attached to every :class:`TensorImpl`.
//
// :class:`TensorMeta` holds the geometric / type description (shape,
// stride, dtype, device) and :class:`AutogradMeta` holds the autograd
// bookkeeping (``requires_grad``, ``grad_fn``, ``grad``, version
// counter).  Separating these structs from :class:`TensorImpl` itself
// keeps the autograd layer and view-creation code readable: callers
// can manipulate a :class:`TensorMeta` or :class:`AutogradMeta` in
// isolation without touching the full tensor.
//
// Notes
// -----
// Both structs are intended to be **cheap to copy** — they hold only
// PODs, small standard containers, and ``shared_ptr`` handles.  Views
// of a tensor reuse the parent's :class:`Storage` but carry their own
// :class:`TensorMeta`, which is the entire mechanism behind reshape /
// transpose / slice without data copies.
//
// See Also
// --------
// :class:`TensorImpl` — owner of one :class:`TensorMeta` and one
//     optional :class:`AutogradMeta`.
// :class:`Storage`    — byte buffer pointed to by every tensor sharing
//     the same metadata family.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include "Device.h"
#include "Dtype.h"
#include "Shape.h"
#include "Storage.h"

namespace lucid {

class Node;
struct TensorHookSlot;

// Geometric and type description of a tensor.
//
// :class:`TensorMeta` is stored **by value** inside :class:`TensorImpl`
// (not on the heap) so that shape / dtype / device queries are a single
// pointer dereference.  Views share the same :class:`Storage` as their
// base tensor but carry an independent :class:`TensorMeta` with
// potentially different shape, stride, or implicit byte offset (the
// offset is encoded in :class:`Storage`).
//
// Attributes
// ----------
// shape : Shape
//     Per-dimension sizes.  Empty shape denotes a 0-dim scalar.
// stride : Stride
//     Per-dimension byte strides.  Must have ``stride.size() ==
//     shape.size()`` to be considered a valid contiguous layout by
//     :func:`is_contiguous_for`.
// dtype : Dtype, optional
//     Element type tag.  Defaults to :attr:`Dtype::F32`.
// device : Device, optional
//     Memory domain owning the underlying storage.  Defaults to
//     :attr:`Device::CPU`.
//
// Notes
// -----
// Equality of two :class:`TensorMeta` instances is structural —
// matching shape, stride, dtype, and device implies the two metas
// describe layout-equivalent tensors (modulo storage offset).
struct TensorMeta {
    Shape shape;
    Stride stride;
    Dtype dtype = Dtype::F32;
    Device device = Device::CPU;

    // Returns the total number of elements described by ``shape``.
    //
    // Computed as the product of all dimension sizes; a 0-dim scalar
    // (empty ``shape``) returns ``1`` by convention.
    //
    // Returns
    // -------
    // std::size_t
    //     ``shape[0] * shape[1] * ... * shape[N-1]``, or ``1`` for a
    //     scalar.
    //
    // Notes
    // -----
    // ``shape_numel`` is the one element-count rule; a second one here
    // cast a negative extent to ``2^64 - 1`` while allocation counted it
    // as empty.  A negative extent raises :class:`InvalidArgument`.
    //
    // Math
    // ----
    // $$ \mathrm{numel} = \prod_{i=0}^{N-1} \mathrm{shape}_i $$
    std::size_t numel() const { return shape_numel(shape); }

    // Returns whether the stored stride matches row-major (C-order)
    // contiguous layout for ``elem_size``-byte elements.
    //
    // A tensor is C-contiguous when ``stride[i] = elem_size *
    // prod(shape[i+1:])`` for every axis ``i``.  A 0-dim scalar is
    // contiguous by definition.
    //
    // Parameters
    // ----------
    // elem_size : std::size_t
    //     Size in bytes of one element (e.g. ``sizeof(float)`` for
    //     ``F32``).  Must match the ``dtype`` width for the result to
    //     be meaningful.
    //
    // Returns
    // -------
    // bool
    //     ``true`` if ``stride`` matches C-contiguous layout for the
    //     given element size, ``false`` otherwise.  Returns ``false``
    //     when ``stride.size()`` does not match ``shape.size()``.
    //
    // Notes
    // -----
    // Used by :func:`TensorImpl::is_contiguous` and by op kernels that
    // require contiguous input — non-contiguous tensors must be
    // materialised via a copy before such kernels run.
    bool is_contiguous_for(std::size_t elem_size) const noexcept {
        if (shape.empty())
            return true;
        if (stride.size() != shape.size())
            return false;
        std::int64_t expected = static_cast<std::int64_t>(elem_size);
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            if (stride[static_cast<std::size_t>(i)] != expected)
                return false;
            expected *= std::max<std::int64_t>(shape[static_cast<std::size_t>(i)], 1);
        }
        return true;
    }
};

// Autograd bookkeeping for a single tensor.
//
// Stored as ``std::optional<AutogradMeta>`` inside :class:`TensorImpl`
// and only heap-emplaced when a tensor actually participates in the
// autograd graph (``requires_grad == true`` or a ``grad_fn`` is
// attached).  Tensors that never need gradients therefore pay zero
// overhead beyond the empty-optional inline storage.
//
// Attributes
// ----------
// requires_grad : bool
//     If ``true``, gradients flowing into this tensor will be
//     accumulated into :attr:`grad` (leaves) or propagated through
//     :attr:`grad_fn` (non-leaves).
// is_leaf : bool
//     ``true`` for user-created tensors (no ``grad_fn``); ``false``
//     for outputs of differentiable ops.  Mirrors the reference
//     framework's ``.is_leaf`` flag.
// version : std::int64_t
//     Bumped on every in-place write via
//     :func:`TensorImpl::bump_version`.  Autograd saves the
//     pre-forward version for each input and raises
//     :class:`VersionMismatch` if the tensor is mutated between
//     forward and backward.
// grad_fn : std::shared_ptr<Node>
//     Edge into the autograd graph — the :class:`Node` that produced
//     this tensor.  ``nullptr`` for leaves.
// grad_output_nr : std::uint32_t
//     Index of this tensor among :attr:`grad_fn`'s outputs, used to
//     pick the right cotangent slot during backward.
// grad : std::optional<Storage>
//     Accumulated gradient storage.  Populated on leaves (and on
//     :attr:`retain_grad` non-leaves) after a normal
//     :func:`backward` call.  Cleared by :func:`zero_grad`.
// grad_impl : std::shared_ptr<TensorImpl>
//     Full :class:`TensorImpl` view of the gradient — populated
//     instead of (or in addition to) :attr:`grad` when
//     ``backward(create_graph=True)`` is used.  Allows the gradient
//     itself to participate in further autograd (second-order
//     derivatives, MAML, Hessian-vector products, …).
// retain_grad : bool
//     When ``true``, the engine accumulates incoming gradients into
//     :attr:`grad` even for non-leaf tensors — mirroring the
//     reference framework's ``Tensor.retain_grad()`` opt-in.  The engine
//     finds the tensor through its producer's hook slot
//     (``autograd/TensorHooks.h``); this flag is what says it still asks.
// leaf_hooks : std::shared_ptr<TensorHookSlot>
//     A leaf's gradient hooks, run before each accumulation into
//     :attr:`grad`.  Here rather than on the leaf's ``AccumulateGrad``,
//     which is made lazily and dropped by ``release_root`` / ``detach_``.
//     Null for a leaf without hooks — and for every non-leaf, whose hooks
//     live on its producer node.
//
// Notes
// -----
// Invariants enforced by the autograd layer:
//
//   * ``is_leaf == true``  ⇒ ``grad_fn == nullptr``.
//   * ``is_leaf == false`` ⇒ ``grad_fn != nullptr``.
//   * :attr:`version` is monotonically non-decreasing for the lifetime
//     of the owning :class:`TensorImpl`.
struct AutogradMeta {
    bool requires_grad = false;
    bool is_leaf = true;
    std::int64_t version = 0;
    std::shared_ptr<Node> grad_fn;
    std::uint32_t grad_output_nr = 0;
    // Accumulated gradient Storage; set on leaves after normal backward().
    std::optional<Storage> grad;
    // Gradient as a full TensorImpl when backward was run with create_graph=true.
    // Allows the gradient tensor itself to participate in further autograd ops
    // (second-order derivatives, MAML, Hessian-vector products, etc.).
    std::shared_ptr<class TensorImpl> grad_impl;
    // When true, Engine accumulates the incoming gradient into this tensor's
    // grad storage even if it is not a leaf (mirrors reference tensor.retain_grad()).
    bool retain_grad = false;
    // A leaf's gradient hooks; the type is complete only in the autograd layer.
    std::shared_ptr<TensorHookSlot> leaf_hooks;
    // Lazily allocated version counter shared with backward nodes that saved
    // this tensor as an output without keeping the TensorImpl alive.
    std::shared_ptr<VersionCounter> version_sink;
};

}  // namespace lucid
