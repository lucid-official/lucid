// lucid/_C/compile/TraceIR.h
//
// Trace intermediate representation for :mod:`lucid.compile`.
//
// :class:`Tracer` records every op dispatched inside a ``_tracing()``
// scope as a :class:`OpNode`; the ordered list of those nodes is the
// :class:`TraceGraph`.  The IR is intentionally minimal in Phase 1.1 —
// each node captures only the data the MPSGraph builder needs to emit
// a corresponding subgraph in Phase 1.2: the op name, output tensor
// metadata (shape / dtype / device), and the integer ids of inputs and
// outputs that thread the DAG together.
//
// Notes
// -----
// IR ownership: the Tracer owns the :class:`TraceGraph`; each
// :class:`OpNode` is appended once and never moved.  Tensor metadata
// is keyed by a monotonic 64-bit id assigned by the Tracer when the op
// scope's outputs are wired in.  Ids are unique within one
// :class:`TraceGraph` only — they do not persist across traces.
//
// This header is declarations-only; the structs are aggregates with
// no out-of-line member functions, so no matching ``.cpp`` is needed.
//
// See Also
// --------
// :class:`lucid::compile::Tracer` — populator of these structs.
// :class:`lucid::compile::MpsBuilder` (Phase 1.2) — consumer.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "../api.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"

namespace lucid::compile {

// Stable named-int identifier minted by :class:`Tracer` for each
// distinct :class:`TensorImpl` observed during a trace.  Unique
// within one :class:`TraceGraph` only — never reused, never persisted.
//
// **Design.**  ``TraceId`` is a named ``std::int64_t`` rather than a
// raw alias so that:
//
//   * The trace-id namespace is greppable as a type (e.g. find every
//     function that takes a ``TraceId`` parameter).
//   * Hash specialization is bound to this type (no accidental
//     overload conflicts with other 64-bit identifiers).
//   * Future strict-typing upgrades (e.g. removing implicit
//     conversion to ``int64_t`` to prevent passing a raw int where
//     a TraceId is expected) only require flipping the ``operator
//     std::int64_t()`` conversion.  All call sites are already
//     compile-time-verified to use the named type.
//
// **Implicit conversion.**  We deliberately preserve implicit
// conversion to/from ``std::int64_t`` for the first pass — there are
// ~400 call sites across the compile path that mix raw ints (Python
// bindings, monotonic counter arithmetic, sentinel ``-1`` checks)
// with TraceId.  Switching to ``explicit`` would force a parallel
// migration we can do in a follow-up.  The named type still moves
// the codebase forward and unblocks the audit-flagged improvement.
//
// **Sentinel.**  ``TraceId::external_feed()`` (== -1) signals "not
// produced by any traced op" — i.e. an external feed (model
// parameter or user input).  Use ``id == TraceId::external_feed()``
// in new code; ``id.v < 0`` still works for backwards compat.
struct LUCID_API TraceId {
    std::int64_t v = -1;

    constexpr TraceId() noexcept = default;
    constexpr TraceId(std::int64_t x) noexcept : v(x) {}
    constexpr operator std::int64_t() const noexcept { return v; }

    TraceId& operator++() noexcept {
        ++v;
        return *this;
    }
    TraceId operator++(int) noexcept {
        auto t = *this;
        ++v;
        return t;
    }

    // No defaulted ``operator==`` / ``operator<=>`` here — they would
    // conflict with the implicit conversion to ``int64_t`` and cause
    // ambiguity at every ``tid < 0`` style site.  Equality between
    // TraceIds (and comparisons like ``tid < 0``) go through the
    // implicit conversion to ``int64_t`` and use the built-in ops.

    // Sentinel value for "this input is an external feed, not the
    // output of any traced op".  Used in :class:`OpNode::inputs`.
    static constexpr TraceId external_feed() noexcept { return TraceId{-1}; }
};

// Backwards-compatible alias.  Existing code that uses ``TensorId``
// (the original raw ``int64_t`` typedef) keeps working unchanged.
// New code should prefer ``TraceId`` directly.
using TensorId = TraceId;

// Metadata snapshot of a tensor at the moment it appears in the trace.
//
// Attributes
// ----------
// id : TensorId
//     Stable id assigned by the Tracer.  Used by :class:`OpNode` to
//     reference inputs and outputs without owning the underlying
//     :class:`TensorImpl`.
// shape : Shape
//     Output shape at trace time.  Phase 1.2 emits MPSGraph
//     placeholders with this shape; Phase 1.6 substitutes batch dim
//     with ``-1`` for ``dynamic=True``.
// dtype : Dtype
//     Element dtype.  Drives F16 vs F32 placeholder choice when the
//     builder consults :func:`current_amp_policy`.
// device : Device
//     Device of the original tensor.  The builder rejects traces
//     containing mixed-device ops; a single :class:`TraceGraph` runs
//     on one device.
//
// Notes
// -----
// Aggregate — copyable, comparable by member-wise equality.  No
// inheritance and no virtual members so the value can be stored in a
// :class:`std::vector` with no indirection.
struct LUCID_API TensorMeta {
    TensorId id = -1;
    Shape shape;
    Dtype dtype = Dtype::F32;
    Device device = Device::CPU;
};

// Per-op attribute value — covers the small set of plain-old-data
// payloads that op forwards report when they need to thread extra
// context (e.g. permutation, axis, stride, padding, keepdim, eps) to
// the MPSGraph emitter.
//
// Notes
// -----
// The variant intentionally stays small — adding a new alternative
// touches every emitter that visits the variant.  When a new op needs
// a payload that doesn't fit, encode it as a vector<int64_t> or a
// short string first; promote to a dedicated alternative only after
// at least two ops share the encoding.
using AttributeValue =
    std::variant<std::int64_t, std::vector<std::int64_t>, double, bool, std::string>;

// One recorded op in the trace.
//
// Attributes
// ----------
// name : std::string
//     Op name as passed to :class:`OpScopeFull` (e.g. ``"conv2d"``,
//     ``"layer_norm"``).  Matches the schema name in
//     :class:`OpRegistry`.
// inputs : std::vector<TensorId>
//     Ids of input tensors, in op-defined order.  Phase 1.1 records
//     these by appending to the vector after the backend dispatch
//     returns (the Tracer hook in :class:`OpScopeFull` runs at op
//     entry; input wiring happens later in :func:`wire_autograd`).
//     An :class:`OpNode` with a pending input slot indicates an op
//     that did not go through ``wire_autograd`` — the builder treats
//     that as an eager-only signature.
// outputs : std::vector<TensorMeta>
//     Per-output metadata snapshots.  Phase 1.1 has at most one
//     output per node; the vector is generic so multi-output ops
//     (e.g. ``topk``, ``lstm``) can be added without IR changes.
// attrs : std::unordered_map<std::string, AttributeValue>
//     Optional payload keyed by attribute name.  Populated by op
//     forwards via :func:`OpScopeFull::set_attr` when the emitter
//     needs context that can't be recovered from input + output
//     shapes alone (permutation, axes, stride, padding, eps, …).
//
// Notes
// -----
// Aggregate — no methods, no invariants enforced at construction.
// The Tracer is responsible for maintaining well-formedness (no
// dangling input ids, output ids in monotone increasing order).
struct LUCID_API OpNode {
    std::string name;
    std::vector<TensorId> inputs;
    std::vector<TensorMeta> outputs;
    std::unordered_map<std::string, AttributeValue> attrs;
};

// Custom backward subgraph captured for a :class:`lucid.autograd.Function`
// with ``_traced_backward = True``.
//
// Forward ops in ``[fwd_begin, fwd_end)`` compute the forward outputs
// ``output_metas`` from ``input_ids``.  During backward emission,
// :class:`BackwardWalker` skips per-op VJP emission for
// ``ops[fwd_begin .. fwd_end)`` and instead binds each output's resolved
// upstream cotangent (or a zero tensor when unused) to ``cotangent_ids[j]``,
// emits ``bwd_ops`` through the forward op emitters, and accumulates each
// non-negative ``input_grad_ids[k]`` onto ``input_ids[k]``.
struct LUCID_API VjpRegion {
    std::size_t fwd_begin = 0;
    std::size_t fwd_end = 0;
    std::vector<TensorId> input_ids;
    std::vector<TensorMeta> output_metas;
    std::vector<TensorId> cotangent_ids;
    std::vector<TensorId> input_grad_ids;
    std::vector<OpNode> bwd_ops;
};

// Ordered list of :class:`OpNode` records produced by one trace.
//
// Attributes
// ----------
// ops : std::vector<OpNode>
//     Nodes appended in dispatch order.  The list defines a DAG via
//     tensor ids — node ``j`` depends on node ``i`` iff one of
//     ``ops[j].inputs`` equals one of the ids in ``ops[i].outputs``.
// next_id : TensorId
//     Monotonic counter the Tracer uses to mint fresh ids when wiring
//     outputs.  Owned by :class:`TraceGraph` so concurrent traces on
//     different threads have independent id spaces.
//
// Notes
// -----
// Aggregate.  Phase 1.2 attaches an :class:`MpsBuilder` cache entry
// keyed by ``hash(ops)`` rather than mutating this struct.
struct LUCID_API TraceGraph {
    std::vector<OpNode> ops;
    TensorId next_id = 0;
    // Non-empty when something ran during the trace that the recording
    // cannot stand for — a backward pass inside the traced function
    // (``autograd.grad`` / ``backward``): its values are storage the
    // tracer never sees, so a compiled replay would reuse them.  Every
    // compile entry point refuses such a graph, and the call runs eager.
    std::string unsupported;
    std::vector<VjpRegion> vjp_regions;
};

}  // namespace lucid::compile

// Hash specialization for :class:`TraceId` so it can serve as a key
// in ``std::unordered_map`` / ``std::unordered_set`` without explicit
// hash arguments.  Delegates to ``std::hash<std::int64_t>`` on the
// underlying ``.v`` value.
namespace std {
template <>
struct hash<::lucid::compile::TraceId> {
    std::size_t operator()(const ::lucid::compile::TraceId& t) const noexcept {
        return std::hash<std::int64_t>{}(t.v);
    }
};
}  // namespace std
