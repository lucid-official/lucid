// lucid/_C/kernel/NaryKernel.h
//
// CRTP base for fixed-arity ``N``-input, single-output op kernels with
// ``N >= 3`` — e.g. :class:`Linear` (input, weight, bias), :class:`Where`,
// fused activations with auxiliary tensors.
//
// Compared to :class:`BinaryKernel`, :class:`NaryKernel` stores per-input
// metadata in :type:`std::array\<..., N\>` (compile-time fixed) rather
// than a pair, and intentionally leaves ``forward()`` entirely to the
// concrete op — every N-ary op has bespoke validation and dispatch
// requirements that don't generalise into a single trampoline.
//
// What the base **does** provide is :meth:`wire_autograd`: the concrete
// op builds its output, then calls
// ``NaryKernel<LinearOp, 3>::wire_autograd(inputs, out)`` to install the
// backward node, snapshot inputs, capture versions, and record edges
// to every input's ``grad_fn`` with a single call.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/AutogradNode.h"
#include "../autograd/Node.h"
#include "../compile/Tracer.h"  // 3.5 Phase 1.1: trace I/O wiring at wire_autograd
#include "../core/GradMode.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "BinaryKernel.h"
#include "IKernel.h"

namespace lucid {
namespace kernel {

// CRTP base for fixed-arity ``N``-input op kernels.
//
// Inherits :class:`AutogradNode\<Derived, N\>` so input metadata
// (``input_shapes_``, ``saved_inputs_``, ``input_tensors_``,
// ``saved_impl_inputs_``) is stored in compile-time-sized
// :type:`std::array` containers.  Also inherits :class:`IKernel` for
// polymorphic invocation.
//
// Unlike :class:`UnaryKernel` and :class:`BinaryKernel`, this base does
// **not** provide a ``forward()`` trampoline — N-ary ops have too much
// op-specific validation to share a single template.  Instead, derived
// ops own their ``forward()`` and call :meth:`wire_autograd` once the
// output tensor exists to attach the backward node.
//
// Template Parameters
// -------------------
// Derived : class
//     The concrete CRTP self-type.
// N : std::size_t
//     The fixed input arity.  Typically 3 or more (use
//     :class:`BinaryKernel` for ``N == 2``).
//
// Notes
// -----
// Slot count is exactly ``N``.  Derived ops implement :meth:`apply`
// (and optionally :meth:`apply_for_graph`) themselves; no
// ``grad_formula`` contract is enforced because N-ary backward shapes
// vary too widely.
//
// See Also
// --------
// :class:`UnaryKernel`, :class:`BinaryKernel`, :class:`VariadicKernel`.
// :class:`IKernel` — the abstract base above the CRTP layer.
template <class Derived, std::size_t N>
class NaryKernel : public AutogradNode<Derived, N>, public IKernel {
public:
    // Return the canonical schema name of the concrete op.
    std::string_view name() const noexcept override { return Derived::schema_v1.name; }

    // Attach an existing backward node to the autograd graph.
    //
    // Parameters
    // ----------
    // bwd : std::shared_ptr<Derived>
    //     The backward node instance to install on ``out``.
    // inputs : const std::array<TensorImplPtr, N>&
    //     The forward inputs in canonical order.  Slots may be
    //     ``nullptr`` (e.g. an optional bias).
    // out : const TensorImplPtr&
    //     The freshly produced output tensor.
    // saves : const std::array<bool, N>&
    //     Which inputs the backward reads the values of: input ``i``'s
    //     storage is snapshotted into ``saved_inputs_[i]`` when
    //     ``saves[i]`` and the input is non-null.  Those are the inputs
    //     :meth:`AutogradNode::validate_versions` guards, so an input read
    //     by shape alone — a bias, a ``beta`` — is left out, and a write
    //     into it after forward is not refused.
    //
    // Returns
    // -------
    // bool
    //     ``true`` when the autograd graph was wired and ``out`` had
    //     its ``grad_fn`` installed; ``false`` when grad mode is off
    //     or no input requires a gradient (no-op).
    //
    // Notes
    // -----
    // Side effects on success: populates ``bwd``'s dtype/device/output
    // shape from the first non-null input, fills ``input_shapes_``,
    // ``input_tensors_``, ``saved_inputs_``, ``saved_impl_inputs_``,
    // records one :class:`Edge` per input via :func:`detail::ensure_grad_fn`,
    // captures per-input versions for in-place mutation detection, and
    // marks ``out`` as a non-leaf, requires_grad tensor.
    static bool wire_autograd(std::shared_ptr<Derived> bwd,
                              const std::array<TensorImplPtr, N>& inputs,
                              const TensorImplPtr& out,
                              const std::array<bool, N>& saves) {
        // 3.5 Phase 1.1: trace wiring fires *before* the GradMode short-circuit
        // so inference passes (no_grad / eval) still produce a complete
        // TraceGraph.  Outside any _tracing() scope this is one TLS load + null
        // check — no heap traffic, no input-array walk.
        if (auto* trc = ::lucid::compile::current_tracer()) {
            std::vector<TensorImplPtr> in_vec(inputs.begin(), inputs.end());
            trc->on_op_io(in_vec, out);
        }

        if (!GradMode::is_enabled())
            return false;

        bool any_grad = false;
        for (const auto& inp : inputs)
            any_grad |= (inp && inp->requires_grad());
        if (!any_grad)
            return false;

        // Derive dtype/device from the first non-null input.
        for (const auto& inp : inputs) {
            if (inp) {
                bwd->dtype_ = inp->dtype();
                bwd->device_ = inp->device();
                break;
            }
        }
        bwd->out_shape_ = out->shape();
        capture_inputs(*bwd, inputs, saves);

        out->set_grad_fn(std::move(bwd));
        out->set_leaf(false);
        out->set_requires_grad(true);
        return true;
    }

private:
    static void capture_inputs(Derived& bwd,
                               const std::array<TensorImplPtr, N>& inputs,
                               const std::array<bool, N>& saves) {
        std::vector<Edge> edges(N);
        std::vector<std::int64_t> versions(N, 0);
        for (std::size_t i = 0; i < N; ++i) {
            const auto& inp = inputs[i];
            bwd.input_shapes_[i] = inp ? inp->shape() : Shape{};
            bwd.input_tensors_[i] = inp;
            edges[i] = Edge{detail::ensure_grad_fn(inp), inp ? inp->grad_output_nr() : 0};
            versions[i] = inp ? inp->version() : 0;
        }
        bwd.set_next_edges(std::move(edges));
        bwd.set_saved_versions(std::move(versions));
        for (std::size_t i = 0; i < N; ++i)
            bwd.keep_needed_input(i, inputs[i], inputs[i], saves[i]);
    }

public:
    // The same, with every input's values saved (``save_ins``) or none.
    static bool wire_autograd(std::shared_ptr<Derived> bwd,
                              const std::array<TensorImplPtr, N>& inputs,
                              const TensorImplPtr& out,
                              bool save_ins = true) {
        std::array<bool, N> saves{};
        saves.fill(save_ins);
        return wire_autograd(std::move(bwd), inputs, out, saves);
    }

    // Convenience overload that constructs the backward node internally.
    //
    // Parameters
    // ----------
    // inputs : const std::array<TensorImplPtr, N>&
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
    // ``wire_autograd(std::make_shared<Derived>(), inputs, out, save_ins)``;
    // use this form when no custom state on the backward node is needed
    // before wiring.
    static bool wire_autograd(const std::array<TensorImplPtr, N>& inputs,
                              const TensorImplPtr& out,
                              bool save_ins = true) {
        return wire_autograd(std::make_shared<Derived>(), inputs, out, save_ins);
    }
};

}  // namespace kernel
}  // namespace lucid
