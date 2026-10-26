// lucid/_C/kernel/VariadicKernel.h
//
// CRTP base for ops whose input count is **only known at runtime** —
// canonical example :class:`Cat` (concatenate any number of tensors
// along an axis); also :class:`Stack`, :class:`Sum` of a tensor list,
// and certain multi-input pointwise fusions.
//
// Contrast with :class:`NaryKernel\<Derived, N\>` which fixes ``N`` at
// compile time: :class:`VariadicKernel` stores per-input metadata in
// :type:`std::vector` containers so the fan-out can be discovered at
// forward time and replayed during backward.  Because the slot count is
// dynamic, the base inherits :class:`Node` directly rather than
// :class:`AutogradNode\<Derived, N\>`, and re-declares the
// ``input_shapes_`` / ``input_tensors_`` / ``saved_inputs_`` /
// ``out_shape_`` members as vector-typed siblings suffixed ``_v_`` to
// avoid colliding with the :class:`AutogradNode` array members when a
// derived type happens to inherit both kernel bases (uncommon).
//
// Typical usage::
//
//     auto out = std::make_shared<TensorImpl>(...);
//     VariadicKernel<CatBackward>::wire_autograd(inputs, out);
//
// :meth:`validate_versions` is overridden to walk the runtime vector
// and check in-place mutations on every input saved by value;
// :meth:`apply` must be
// implemented by the concrete op (no shared backward formula is
// imposed because variadic backward shapes vary too widely).

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../compile/Tracer.h"  // 3.5 Phase 1.1: trace I/O wiring at wire_autograd
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/GradMode.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "BinaryKernel.h"
#include "IKernel.h"

namespace lucid {

class TensorImpl;

namespace kernel {

// CRTP base for runtime-variable-arity op kernels.
//
// Inherits :class:`Node` directly (not :class:`AutogradNode\<Derived, N\>`)
// so the saved-input and edge containers can grow at runtime, and
// :class:`IKernel` for polymorphic invocation.  Derived ops own their
// own ``forward()`` and ``apply()`` — only the autograd wiring helper
// is shared.
//
// Template Parameters
// -------------------
// Derived : class
//     The concrete CRTP self-type.
//
// Attributes
// ----------
// input_tensors_v_ : std::vector<std::weak_ptr<TensorImpl>>
//     Weak references to the original inputs, used for in-place
//     mutation version checks.
// input_shapes_v_ : std::vector<Shape>
//     Shapes of the original inputs, preserved for backward shape math.
// out_shape_ : Shape
//     Shape of the produced output tensor.
// dtype_ : Dtype
//     Effective dtype, taken from the first non-null input.
// device_ : Device
//     Effective device, taken from the first non-null input.
// saved_inputs_v_ : std::vector<Storage>
//     Snapshots of each input's :class:`Storage` at forward time, one
//     per input slot (empty for a null input), populated when
//     :meth:`wire_autograd` is called with ``save_ins=true``.
//
// Notes
// -----
// All inputs must share dtype and device — :meth:`wire_autograd` does
// **not** validate this; the concrete op's ``forward()`` is expected
// to have already enforced consistency before calling wire_autograd.
// :meth:`validate_versions` walks the runtime vector and checks each
// input whose values :attr:`saved_inputs_v_` holds — the rule
// :meth:`AutogradNode::validate_versions` follows.
//
// See Also
// --------
// :class:`NaryKernel` — fixed-arity sibling using :type:`std::array`.
// :class:`UnaryKernel`, :class:`BinaryKernel`.
// :class:`IKernel` — the abstract base above the CRTP layer.
template <class Derived>
class VariadicKernel : public Node, public IKernel {
public:
    // Return the canonical schema name of the concrete op.
    std::string_view name() const noexcept override { return Derived::schema_v1.name; }

    // Verify no input whose values :meth:`apply` reads was written in
    // place between forward and backward.
    //
    // Notes
    // -----
    // Checks each input whose values :attr:`saved_inputs_v_` holds and
    // raises a clear error (with the op name and slot index) if its version
    // drifted; an input saved by shape alone is not refused (see
    // :func:`holds_saved_values`).  Invoked by the autograd engine before
    // :meth:`apply`.
    void validate_versions() override {
        const std::size_t n = std::min(input_tensors_v_.size(), saved_inputs_v_.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (::lucid::holds_saved_values(saved_inputs_v_[i]))
                check_saved_version(input_tensors_v_[i], i, Derived::schema_v1.name, i);
        }
    }

    // Free the values saved for backward, as AutogradNode does.
    void release_saved() override {
        saved_inputs_v_.clear();
        input_tensors_v_.clear();
    }

    // :meth:`apply` reads values from the saved slots — filled for every
    // input when the op saves them, whatever their size — and from members of
    // the derived class's own where it declares ``kKeepsOwnValues``, as
    // AutogradNode reads them.  No tensor is kept for graph-mode backward, so
    // ``cat`` and ``stack``, which save none, may run again in either mode.
    SavedKind saved_kind() const noexcept override {
        if constexpr (requires { Derived::kKeepsOwnValues; }) {
            if (Derived::kKeepsOwnValues)
                return SavedKind::kEagerValues;
        }
        const bool holds = !saved_inputs_v_.empty();
        return holds ? SavedKind::kEagerValues : SavedKind::kNothing;
    }

    // Weak references to original input tensors; used for version checks.
    std::vector<std::weak_ptr<TensorImpl>> input_tensors_v_;
    // Shapes of the original inputs, preserved for backward shape math.
    std::vector<Shape> input_shapes_v_;
    // Shape of the produced output tensor.
    Shape out_shape_;
    // Dtype taken from the first non-null input (all inputs must match).
    Dtype dtype_ = Dtype::F32;
    // Device taken from the first non-null input (all inputs must match).
    Device device_ = Device::CPU;
    // Snapshots of each input's Storage at forward time, used by apply().
    std::vector<Storage> saved_inputs_v_;

    // Attach an existing backward node to the autograd graph.
    //
    // Parameters
    // ----------
    // bwd : std::shared_ptr<Derived>
    //     The backward node instance to install on ``out``.
    // inputs : const std::vector<TensorImplPtr>&
    //     The forward inputs in canonical order; slots may be ``nullptr``.
    // out : const TensorImplPtr&
    //     The freshly produced output tensor.
    // save_ins : bool, default ``true``
    //     Whether to snapshot each non-null input's storage into
    //     ``saved_inputs_v_`` for use during backward.
    //
    // Returns
    // -------
    // bool
    //     ``true`` when the autograd graph was wired; ``false`` when
    //     grad mode is off or no input requires a gradient (no-op).
    //
    // Notes
    // -----
    // Side effects on success: derives dtype/device from the first
    // non-null input, records ``out_shape_``, populates the per-input
    // vectors, records one :class:`Edge` per input via
    // :func:`detail::ensure_grad_fn`, captures versions, and marks
    // ``out`` as a non-leaf requires_grad tensor.
    static bool wire_autograd(std::shared_ptr<Derived> bwd,
                              const std::vector<TensorImplPtr>& inputs,
                              const TensorImplPtr& out,
                              bool save_ins = true) {
        // 3.5 Phase 1.1: trace wiring fires *before* GradMode short-circuit so
        // inference passes still record a complete TraceGraph (see
        // NaryKernel::wire_autograd for the same pattern).
        if (auto* trc = ::lucid::compile::current_tracer()) {
            trc->on_op_io(inputs, out);
        }

        if (!GradMode::is_enabled())
            return false;

        bool any_grad = false;
        for (const auto& inp : inputs)
            any_grad |= (inp && inp->requires_grad());
        if (!any_grad)
            return false;

        // Use the first non-null input to resolve dtype and device.
        for (const auto& inp : inputs) {
            if (inp) {
                bwd->dtype_ = inp->dtype();
                bwd->device_ = inp->device();
                break;
            }
        }
        bwd->out_shape_ = out->shape();

        const std::size_t n = inputs.size();
        bwd->input_shapes_v_.reserve(n);
        bwd->input_tensors_v_.reserve(n);
        if (save_ins)
            bwd->saved_inputs_v_.reserve(n);

        std::vector<Edge> edges;
        std::vector<std::int64_t> versions;
        edges.reserve(n);
        versions.reserve(n);

        for (const auto& inp : inputs) {
            bwd->input_shapes_v_.push_back(inp ? inp->shape() : Shape{});
            bwd->input_tensors_v_.push_back(inp);
            // One entry per slot, so entry ``i`` is input ``i``'s values.
            if (save_ins)
                bwd->saved_inputs_v_.push_back(inp ? inp->storage() : Storage{CpuStorage{}});
            edges.emplace_back(detail::ensure_grad_fn(inp), inp ? inp->grad_output_nr() : 0);
            versions.push_back(inp ? inp->version() : 0);
        }

        bwd->set_next_edges(std::move(edges));
        bwd->set_saved_versions(std::move(versions));

        out->set_grad_fn(std::move(bwd));
        out->set_leaf(false);
        out->set_requires_grad(true);
        return true;
    }

    // Convenience overload that constructs the backward node internally.
    //
    // Parameters
    // ----------
    // inputs : const std::vector<TensorImplPtr>&
    //     The forward inputs in canonical order.
    // out : const TensorImplPtr&
    //     The freshly produced output tensor.
    // save_ins : bool, default ``true``
    //     Forwarded to the primary overload.
    //
    // Returns
    // -------
    // bool
    //     See the primary :meth:`wire_autograd` overload.
    //
    // Notes
    // -----
    // Equivalent to
    // ``wire_autograd(std::make_shared<Derived>(), inputs, out, save_ins)``.
    static bool wire_autograd(const std::vector<TensorImplPtr>& inputs,
                              const TensorImplPtr& out,
                              bool save_ins = true) {
        return wire_autograd(std::make_shared<Derived>(), inputs, out, save_ins);
    }
};

}  // namespace kernel
}  // namespace lucid
