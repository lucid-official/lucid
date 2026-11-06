// lucid/_C/compile/VjpEmitters/VjpEmitter.h
//
// Manual VJP (vector-Jacobian product) emitter base + registry +
// reverse-mode walker.  This is the compile-path counterpart of
// :mod:`autograd` for cases where MPSGraph's
// ``gradientForPrimaryTensor:withTensors:`` cannot be used — embedding
// and split_at break it with ``Not a predecessor of primaryTensor``,
// and non-float feeds trigger ``Couldn't get gradient Tensor for
// tensor of op : feed_N``.  Activating the manual path lets Lucid emit
// the backward MPSGraph subgraph itself, op-by-op.
//
// One :class:`VjpEmitter` instance handles one op family's backward.
// Subclasses live next to their forward counterparts under
// ``VjpEmitters/{elementwise,linalg,reduce,shape,nn}/`` and self-
// register at process startup via :func:`register_vjp_emitter`.
//
// Activation knobs (read in :file:`MpsBuilder.mm`):
//
//   * ``LUCID_MANUAL_VJP=1``        — opt-in to manual VJP path.  When
//                                     a coverage gap is hit, the caller
//                                     transparently falls through to
//                                     MPSGraph autograd.
//   * ``LUCID_MANUAL_VJP_REQUIRE=1`` — hard fail on coverage gap
//                                     (intended for CI).
//
// Header is pure C++/Objective-C interop — Objective-C types are
// erased to ``void*`` at the boundary so plain ``.cpp`` files can
// hold the registry; concrete VJPs re-cast via ``__bridge`` in their
// ``.mm`` files.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../api.h"
#include "../OpEmitters/OpEmitter.h"
#include "../TraceIR.h"

namespace lucid::compile {

// Backward-side context container.
//
// Holds the active :class:`MPSGraph` (shared with the forward
// :class:`BuilderContext`), a back-reference to the forward context
// for saved-input lookups, and the trace-id → gradient-tensor map.
// Two id namespaces are kept disjoint: the forward context owns
// ``tid → MPSGraphTensor*`` (forward activations); this container
// owns ``tid → MPSGraphTensor*`` for backward gradients keyed on the
// *same* trace id.  VJPs read forward activations via :func:`forward`
// and write input grads via :func:`accumulate_grad`.
class LUCID_API BackwardContext {
public:
    BackwardContext(void* graph_void, BuilderContext& fwd);

    // Underlying graph (shared with the forward context).  Re-cast to
    // ``MPSGraph*`` via ``__bridge`` in ``.mm`` files.
    void* graph() const { return graph_; }

    // Forward activation by trace id.  Delegates to ``fwd_.resolve(tid)``.
    // Returns ``nullptr`` if the trace id has no forward binding (the
    // walker treats that as a coverage gap and bails out).
    void* forward(TensorId tid) const;

    // Forward context — exposed for VJPs that need additional fwd-side
    // bookkeeping (e.g. the device).  Most VJPs only need ``forward()``.
    BuilderContext& forward_ctx() { return fwd_; }

    // Look up the gradient tensor accumulated so far on ``tid``.
    // Returns ``nullptr`` if no demand has been registered yet.
    void* resolve_grad(TensorId tid) const;

    // Register a gradient contribution for ``tid``.  If a contribution
    // already exists, the two are summed (broadcast-compatible by
    // construction since both have the shape of the forward producer's
    // output).  If this is the first contribution, the tensor is stored
    // verbatim — no fresh add op emitted.
    void accumulate_grad(TensorId tid, void* grad);

    // Unreduce ``grad`` (currently sized as ``broadcast_shape``) back to
    // ``target_shape`` by summing along the axes that were broadcast.
    // Implemented in :file:`_VjpHelpers.h`; declared here so VJP ``.mm``
    // can call it through the context.
    void* unreduce(void* grad,
                   const std::vector<std::int64_t>& target_shape,
                   const std::vector<std::int64_t>& broadcast_shape);

private:
    void* graph_;
    BuilderContext& fwd_;
    std::unordered_map<TensorId, void*> grad_map_;
};

// Abstract VJP emitter — one per op family.
//
// Concrete subclasses live under :file:`VjpEmitters/<bucket>/` and
// register themselves at process startup.  The trace-IR conventions
// mirror :class:`OpEmitter` 1:1 (input slots from
// ``fwd_node.inputs``, output slots from ``fwd_node.outputs``, attrs
// from ``fwd_node.attrs``).
//
// Forward vs backward signature
// -----------------------------
// :class:`OpEmitter::emit` returns ``void*`` (the single primary
// output tensor; auto-bound by the builder).  This emitter returns
// ``bool`` because backward emission is fundamentally many-output:
// a binary op's VJP writes grads for two inputs, a 3-input
// ``linear``'s VJP writes three.  Rather than picking a "primary"
// grad to return, every VJP writes its outputs via
// :func:`BackwardContext::accumulate_grad` and returns ``true`` on
// success.  Returning ``false`` signals a per-call coverage gap —
// rare, e.g. an attr combination the VJP doesn't support — which
// the walker propagates back to the integration site for
// fall-through to ``gradientForPrimaryTensor:`` (or hard-fail under
// ``LUCID_MANUAL_VJP_REQUIRE=1``).
class LUCID_API VjpEmitter {
public:
    virtual ~VjpEmitter() = default;

    // Op name this VJP handles — matched against :attr:`OpNode::name`.
    virtual std::string_view op_name() const = 0;

    // Emit the backward MPSGraph subgraph for ``fwd_node``.
    //
    // Parameters
    // ----------
    // bctx : BackwardContext&
    //     Carries the in-progress graph + grad id map + fwd lookups.
    // fwd_node : const OpNode&
    //     The forward node being differentiated.  Read inputs / outputs
    //     / attrs the same way the forward emitter did.
    // grad_outs : const std::vector<void*>&
    //     One incoming gradient per forward output slot, in
    //     ``fwd_node.outputs`` order.  ``nullptr`` entries indicate
    //     dead output slots — the VJP must not produce contributions
    //     that depend on a dead-slot grad (most ops just check
    //     ``grad_outs[0] != nullptr`` and bail otherwise).
    //
    // Returns
    // -------
    // bool
    //     ``true`` on success.  ``false`` signals a per-call coverage
    //     gap (rare — typically the VJP doesn't support some attr
    //     combination); the walker aborts and the caller falls through
    //     to MPSGraph autograd (or hard-fails under
    //     ``LUCID_MANUAL_VJP_REQUIRE``).
    virtual bool
    emit(BackwardContext& bctx, const OpNode& fwd_node, const std::vector<void*>& grad_outs) = 0;
};

// Register ``vjp`` under its :func:`op_name`.  Process-global state;
// typically called from a static initialiser in the emitter's ``.mm``
// file.
LUCID_API void register_vjp_emitter(std::unique_ptr<VjpEmitter> vjp);

// Look up the VJP for ``op_name``; ``nullptr`` if not registered.
LUCID_API VjpEmitter* find_vjp_emitter(std::string_view op_name);

// Reverse-mode walker.  Owns the BackwardContext and drives the
// per-op VJP dispatch.
//
// Lifetime model: constructed inside the lazy ``derive_grads_now``
// lambda at the integration sites in :file:`MpsBuilder.mm`; runs once
// per compile.  No state survives the compile call.
class LUCID_API BackwardWalker {
public:
    BackwardWalker(void* graph_void, BuilderContext& fwd, const TraceGraph& trace);

    // Compute gradients of ``param_ids`` w.r.t. ``loss_id``.
    //
    // On success: ``out_grads`` is resized to ``param_ids.size()`` and
    // entry ``i`` holds the ``MPSGraphTensor*`` (as ``void*``) gradient
    // of the loss w.r.t. ``param_ids[i]``.  Returns ``true``.
    //
    // On failure: returns ``false`` and writes a diagnostic to
    // ``*error_msg`` (if non-null).  Common reasons:
    //   * VJP coverage gap — no emitter registered for an op that lies
    //     on the gradient path.
    //   * Forward binding missing for a saved-input id.
    //   * VJP emitter itself signalled per-call failure.
    //
    // ``absent``, when given, accepts a parameter the loss does not
    // depend on: its gradient comes back as zeros and its slot is marked
    // ``true`` — eager leaves such a gradient ``None``, and the caller
    // does the same.  Without it the parameter is an error.
    bool compute_grads(TensorId loss_id,
                       const std::vector<TensorId>& param_ids,
                       std::vector<void*>& out_grads,
                       std::string* error_msg,
                       std::vector<bool>* absent = nullptr);

private:
    void* graph_;
    const TraceGraph& trace_;
    BackwardContext bctx_;
};

// Env-var helpers used by :file:`MpsBuilder.mm` integration sites.
LUCID_API bool use_manual_vjp();
LUCID_API bool use_manual_vjp_require();

// ``LUCID_MANUAL_VJP_DEBUG=1`` — emit a structured stderr line every
// time the walker hits a coverage gap (op name, input/output shapes
// and dtypes, fallback verdict).  Off by default; turn on when
// diagnosing why a model falls back from manual VJP to MPSGraph
// autograd.
LUCID_API bool use_manual_vjp_debug();

// Three-state classification used by :func:`lucid.compile.diagnose`
// to bucket each op in a captured trace.  A future call's gradient
// path is fully described by these three states:
//
//   * ``Registered`` — has a :class:`VjpEmitter` (real gradient).
//   * ``GradSink``   — op is in :func:`no_grad_ops` (factory, integer
//                      cast, comparison, arg-reduce); the walker
//                      stops gradient flow here by design.
//   * ``Missing``    — no emitter, not a sink → soft-fallback to
//                      ``gradientForPrimaryTensor:`` (or hard-fail
//                      under ``LUCID_MANUAL_VJP_REQUIRE``).
enum class VjpRegistration : std::uint8_t {
    Registered = 0,
    GradSink = 1,
    Missing = 2,
};

// Look up an op's manual-VJP coverage status.  Stable across the
// process lifetime (the registry is static-init'd by each
// ``VjpEmitters/**/*.mm`` translation unit).
LUCID_API VjpRegistration vjp_registration_status(const std::string& op_name);

// Whether the backward stops at ``node``: its op is a :class:`VjpRegistration`
// ``GradSink`` or it was recorded with grad mode off (``kGradOffAttr``).  The one
// rule, for the walker here and for what ``lucid.compile`` works out in Python
// about where a gradient reaches.
LUCID_API bool stops_gradient(const OpNode& node);

// Tri-state return from :func:`try_manual_vjp_grads`.  Encodes which
// path the caller should take after the helper runs:
//
//   * ``Disabled``  — ``LUCID_MANUAL_VJP=0``, caller must use the
//                     existing ``gradientForPrimaryTensor:`` path
//                     (no grads were attempted).
//   * ``Success``   — Manual walker completed; ``out_grads`` is
//                     populated and the caller should consume it
//                     verbatim.
//   * ``FellBack``  — Manual walker hit a coverage gap; caller must
//                     fall through to ``gradientForPrimaryTensor:``.
//                     ``LUCID_MANUAL_VJP_REQUIRE`` was *not* set.
//   * ``HardFail``  — Manual walker hit a coverage gap and
//                     ``LUCID_MANUAL_VJP_REQUIRE=1`` is set; caller
//                     should propagate the error in ``*error_msg``.
enum class ManualVjpStatus { Disabled, Success, FellBack, HardFail };

// Shared dispatch for the 4 :file:`MpsBuilder.mm` integration sites.
// Reads the env gates, constructs a :class:`BackwardWalker` when
// enabled, and writes the resulting grads into ``out_grads`` on
// success.  Centralises the same 12-line block that previously lived
// inline at each of the 4 ``compile_*`` callers.
//
// Parameters mirror :func:`BackwardWalker::compute_grads` — see there
// for semantics.
LUCID_API ManualVjpStatus try_manual_vjp_grads(void* graph_void,
                                               BuilderContext& fwd,
                                               const TraceGraph& trace,
                                               TensorId loss_id,
                                               const std::vector<TensorId>& param_ids,
                                               std::vector<void*>& out_grads,
                                               std::string* error_msg,
                                               std::vector<bool>* absent = nullptr);

}  // namespace lucid::compile
