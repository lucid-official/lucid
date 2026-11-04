// lucid/_C/autograd/AccumulateGrad.h
//
// Declares AccumulateGrad, the terminal Node placed at every leaf tensor that
// requires a gradient.  It is the last node visited by the Engine during any
// backward pass and is responsible for writing the accumulated gradient into
// the leaf's TensorImpl.

#pragma once

#include <memory>

#include "../core/TensorImpl.h"
#include "Node.h"

namespace lucid {

// Terminal backward node that accumulates an incoming gradient into a leaf
// tensor's ``.grad`` storage.
//
// An ``AccumulateGrad`` is attached to a leaf :class:`TensorImpl`'s
// ``grad_fn`` slot the first time the tensor is used in a computation that
// requires gradients (see ``_register_python_backward_node`` and the op
// builders under ``ops/``).  Leaves do not themselves produce a ``grad_fn``
// during their own "forward" step, so this sentinel terminates the backward
// walk: its ``next_edges()`` is always empty.
//
// Apply Semantics
// ---------------
// On each call to ``apply(grad_out)``:
//
// 1. If the leaf has been garbage-collected (``weak_ptr`` expired), the
//    gradient is silently discarded.
// 2. If the leaf no longer requires gradients (e.g. ``requires_grad_(False)``
//    was called between forward and backward), the gradient is discarded.
// 3. Incoming gradients are cast to the leaf's own dtype to handle AMP /
//    autocast paths where the same parameter may be reached via different
//    effective dtypes (e.g. F16 from a Conv and F32 from a ForceFP32 branch).
// 4. The leaf's own hooks (``Tensor.register_hook``) run on that gradient,
//    and what they return is what is accumulated.
// 5. If the leaf has no gradient yet, ``grad_out`` is moved in directly —
//    avoiding the cost of allocating a zero buffer and adding to it.
//    Otherwise ``accumulate_into()`` performs an in-place ``+=`` using the
//    appropriate backend (Accelerate on CPU, MLX on GPU).
//
// Steps 2–5 are :func:`accumulate_leaf`, which a backward pass rooted at a
// leaf with no ``AccumulateGrad`` of its own runs directly.
//
// Ownership
// ---------
// ``AccumulateGrad`` holds only a ``weak_ptr`` to its leaf so that it does
// not artificially extend the tensor's lifetime while the backward graph is
// alive.  Non-leaf intermediate tensors never carry an ``AccumulateGrad`` —
// their gradient flows through whatever produced them and is only stored
// transiently in the engine's ``pending`` map (or in ``.grad`` if the user
// explicitly called ``retain_grad`` on the intermediate).
//
// See Also
// --------
// :class:`Engine` : reverse-mode driver that dispatches into this node.
// :class:`Node` : abstract base class.
class AccumulateGrad : public Node {
public:
    // Construct an ``AccumulateGrad`` bound to a specific leaf tensor.
    //
    // Parameters
    // ----------
    // leaf : std::weak_ptr<TensorImpl>
    //     The leaf tensor whose ``.grad`` slot this node writes into.
    //     Must satisfy ``leaf->is_leaf() == true``; not asserted here but
    //     enforced at all call sites (op builders, hook-edge constructors).
    explicit AccumulateGrad(std::weak_ptr<TensorImpl> leaf);

    // Write ``grad_out`` into the leaf's gradient storage.
    //
    // The first call moves the gradient in directly; subsequent calls
    // accumulate ``+=`` in place via the backend-appropriate kernel.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Incoming gradient.  Will be cast to the leaf's dtype if it
    //     does not already match (handles AMP mixed-dtype paths).
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Always empty — ``AccumulateGrad`` has no outgoing edges and the
    //     engine has no further work to schedule for this branch.
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode variant invoked when ``Engine::backward`` runs with
    // ``create_graph=true``.  Stores ``grad_out`` (a :class:`TensorImpl`
    // that may carry its own ``grad_fn``) into the leaf's ``grad_impl``
    // slot so the gradient tensor itself remains differentiable for
    // higher-order gradient computations.
    //
    // Parameters
    // ----------
    // grad_out : const TensorImplPtr&
    //     Gradient tensor whose autograd metadata must be preserved.
    //
    // Returns
    // -------
    // std::vector<TensorImplPtr>
    //     Always empty for the same reason as :func:`apply`.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;

    // Human-readable node name surfaced in error messages and graph dumps.
    std::string node_name() const override { return "AccumulateGrad"; }

    // A leaf keeps its AccumulateGrad from one graph to the next, and it
    // holds nothing a pass frees.
    SavedKind saved_kind() const noexcept override { return SavedKind::kNothing; }

    Shape output_grad_shape(std::uint32_t /*slot*/) const override {
        if (auto t = leaf_.lock())
            return t->shape();
        return {};
    }

    // Accessor for the leaf ``weak_ptr``.
    //
    // Returns
    // -------
    // std::weak_ptr<TensorImpl>
    //     A weak reference to the bound leaf — may be expired.  Used by
    //     tests and graph-inspection utilities.
    std::weak_ptr<TensorImpl> leaf() const { return leaf_; }

private:
    std::weak_ptr<TensorImpl> leaf_;
};

// Prepare ``grad`` for ``leaf`` by casting to ``leaf.dtype()`` and running
// ``leaf``'s tensor hooks (``Tensor.register_hook``).
LUCID_API bool prepare_leaf_grad(TensorImpl& leaf, Storage& grad);
LUCID_API TensorImplPtr prepare_leaf_grad_for_graph(const TensorImplPtr& leaf,
                                                    const TensorImplPtr& grad);

// Commit a prepared gradient into ``leaf``'s ``.grad`` slot and run any
// registered post-accumulate-grad hooks.
LUCID_API void commit_leaf_grad(TensorImpl& leaf, Storage grad);
LUCID_API void commit_leaf_grad_for_graph(const TensorImplPtr& leaf, const TensorImplPtr& grad);

// Accumulate ``grad`` into ``leaf``'s ``.grad``: nothing unless the leaf
// requires grad, then the cast to its dtype, its hooks, and the sum.
//
// Parameters
// ----------
// leaf : TensorImpl&
//     The leaf the gradient is for.
// grad : Storage
//     The gradient, in any floating dtype.
LUCID_API void accumulate_leaf(TensorImpl& leaf, Storage grad);

// :func:`accumulate_leaf` for ``create_graph``: the gradient keeps its graph
// and is summed into ``grad_impl`` by ``add_op``.
LUCID_API void accumulate_leaf_for_graph(const TensorImplPtr& leaf, const TensorImplPtr& grad);

// A graph-mode gradient in the dtype of the tensor it is for.
//
// A complex gradient reaching a real tensor keeps its real part — ``fft`` of
// a real input answers complex, and ``d/dx`` of a real x is real — and any
// other difference is a cast.  :meth:`AccumulateGrad::apply` does this to a
// storage; the graph-mode paths (``backward(create_graph=True)`` and
// ``grad(create_graph=True)``) returned the gradient as it came, complex
// where the storage path was real.
//
// Parameters
// ----------
// grad : const TensorImplPtr&
//     The gradient, possibly null.
// like : const TensorImplPtr&
//     The tensor it is the gradient of.
//
// Returns
// -------
// TensorImplPtr
//     ``grad`` itself when the dtypes already agree, else a recorded
//     projection or cast of it.
TensorImplPtr gradient_in_dtype_of(const TensorImplPtr& grad, const TensorImplPtr& like);

}  // namespace lucid
