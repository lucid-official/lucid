// lucid/_C/compile/Tracer.h
//
// Op-level tracer used by :mod:`lucid.compile`.
//
// :class:`Tracer` is a thread-local sink that records every op
// dispatched while it is installed via :func:`set_current_tracer`.
// The single hook site is the body of :class:`OpScopeFull`'s
// constructor — when an op enters its scope, ``OpScopeFull`` consults
// :func:`current_tracer` and, if non-null, forwards the op name and
// output meta into :func:`Tracer::on_op_enter`.  Ops outside any
// ``_tracing()`` scope take a single null check and incur no further
// cost (matches the :class:`OpScope` / :class:`Profiler` fast-path
// invariant in :file:`Profiler.h`).
//
// Phase 1.1 only records :class:`OpNode` headers (name + output
// metadata).  Input/output id wiring is added in Phase 1.2 inside the
// shared :func:`wire_autograd` boundary — the single site every op
// passes through after backend dispatch.  Until that wiring lands,
// ``inputs`` is left empty and the MpsBuilder treats such a node as
// a starting placeholder.
//
// Notes
// -----
// Thread safety:
//
//   - :func:`current_tracer` / :func:`set_current_tracer` operate on
//     a ``thread_local`` slot — each thread chooses its own tracer
//     independently.  Cross-thread tracing is not supported in 3.5.0
//     (training step is single-threaded by convention).
//   - :class:`Tracer` itself is **not** internally synchronised.  The
//     installing thread owns the object and must not share it with
//     other threads while it is the active tracer.  This matches the
//     trace lifecycle: install → run a single forward pass → detach.
//
// See Also
// --------
// :class:`lucid::compile::TraceGraph` — the recorded IR.
// :class:`lucid::OpScopeFull` — the single hook site.
// :func:`lucid::current_profiler` (:file:`Profiler.h:234`) — the TLS
//     pattern this header mirrors.

#pragma once

#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "../api.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/fwd.h"  // TensorImpl forward declaration
#include "TraceIR.h"

namespace lucid::compile {

// Thread-local op recorder consumed by :class:`OpScopeFull`.
//
// Constructing a Tracer is cheap — the underlying :class:`TraceGraph`
// starts empty.  Hooks fire only between :func:`set_current_tracer`
// (this) and :func:`set_current_tracer` (nullptr); outside that
// window the tracer accumulates nothing.
//
// Attributes
// ----------
// graph_ : TraceGraph
//     The recorded op DAG.  Appended to by :func:`on_op_enter` and
//     exposed read-only via :func:`graph`.
//
// Notes
// -----
// Non-copyable, non-movable — the active-tracer slot is a raw pointer
// owned by the caller; copying or moving the Tracer would either
// invalidate that pointer or duplicate the recording sink.
//
// Examples
// --------
// Native usage (Python binding mirrors this)::
//
//     Tracer t;
//     set_current_tracer(&t);
//     // ... run a forward pass — every OpScopeFull ctor records ...
//     set_current_tracer(nullptr);
//     const auto& g = t.graph();  // inspect recorded nodes
class LUCID_API Tracer {
public:
    // Construct an empty ``Tracer`` ready to be installed via
    // :func:`set_current_tracer`.
    //
    // Defaulted; ``graph_`` starts empty and ``impl_to_id_`` starts with
    // no entries.  No op hooks fire until the tracer is installed as the
    // active TLS sink — construction itself has zero side effects.
    Tracer() = default;

    Tracer(const Tracer&) = delete;
    Tracer& operator=(const Tracer&) = delete;
    Tracer(Tracer&&) = delete;
    Tracer& operator=(Tracer&&) = delete;

    // Records the entry of a new op into the active scope.
    //
    // Parameters
    // ----------
    // name : std::string_view
    //     Op name as passed to :class:`OpScopeFull`.  Copied into the
    //     resulting :class:`OpNode::name`.
    // device : Device
    //     Device on which the op is dispatching.  Stored in the
    //     single :class:`TensorMeta` appended to ``outputs``.
    // dtype : Dtype
    //     Output element dtype at op entry.
    // shape : Shape
    //     Output shape at op entry.  Moved into the appended
    //     :class:`TensorMeta`.
    //
    // Notes
    // -----
    // Phase 1.1 appends one :class:`OpNode` per call with an empty
    // ``inputs`` vector and a single-element ``outputs`` vector.
    // Phase 1.2 backfills ``inputs`` from the autograd wiring site
    // and may append additional outputs for multi-output ops.
    void on_op_enter(std::string_view name, Device device, Dtype dtype, Shape shape);

    // Wires the input/output TensorImpl identities of the most recently
    // recorded :class:`OpNode` into the trace IR.
    //
    // Parameters
    // ----------
    // inputs : const std::vector<TensorImplPtr>&
    //     Owning shared pointers to the input tensors in op-defined
    //     order.  ``nullptr`` slots (e.g. an optional bias) are passed
    //     through as ``-1`` in the resulting :class:`OpNode::inputs``.
    //     The Tracer retains a strong reference to every external
    //     feed (input not produced by an earlier traced op) so the
    //     :class:`MpsBuilder` can read its shape/dtype + the run path
    //     can bind its data buffer.
    // output : const TensorImplPtr&
    //     Output tensor.  The id minted earlier by :func:`on_op_enter`
    //     (stored in ``node.outputs[0].id``) is associated with this
    //     pointer in the internal map so subsequent ops that consume
    //     this tensor can resolve it back to the same id.  Only the
    //     raw pointer is kept (no strong ref) — the output's lifetime
    //     is the caller's responsibility.
    //
    // Notes
    // -----
    // Phase 1.1 wires only kernel families that go through
    // :class:`kernel::NaryKernel` or :class:`kernel::VariadicKernel`'s
    // ``wire_autograd`` boundary (Conv / Linear / BatchNorm / Norm /
    // Loss / …).  Ops dispatched through :class:`UnaryKernel` or
    // :class:`BinaryKernel`'s in-line ``forward()`` reach
    // :func:`on_op_enter` but skip this step — their ``inputs`` slot
    // stays empty, marking the node "pending" for the Phase 1.2
    // builder, which then routes that signature to the eager path.
    //
    // Calling this with an empty :class:`TraceGraph` (no preceding
    // :func:`on_op_enter`) is a no-op — defensive against ordering
    // mistakes at hook sites.
    void on_op_io(const std::vector<TensorImplPtr>& inputs, const TensorImplPtr& output);

    // Give later reads of a live buffer the SSA id of its updated value,
    // without mutating trace-time storage or losing the original feed.
    void on_buffer_update(const TensorImplPtr& buffer, const TensorImplPtr& value);

    TensorImplPtr buffer_value(const TensorImplPtr& buffer) const {
        const auto it = buffer_values_.find(buffer.get());
        return it == buffer_values_.end() ? buffer : it->second;
    }

    // Attach a single attribute to the most recently recorded
    // :class:`OpNode`.  Used by op forwards via
    // :func:`OpScopeFull::set_attr` to thread payloads (permutation,
    // axis, stride, padding, …) that the emitter would otherwise be
    // unable to recover from inputs + output shapes alone.
    //
    // Parameters
    // ----------
    // key : std::string_view
    //     Attribute name; chosen by the op forward and read by the
    //     matching emitter.  Convention: snake_case, matches the
    //     reference-framework keyword argument when one exists
    //     (``"permutation"``, ``"stride"``, ``"padding"``, ``"axis"``,
    //     ``"keepdim"``, ``"eps"``, …).
    // value : AttributeValue
    //     The payload, moved into the attribute map.
    //
    // Notes
    // -----
    // Calling this with an empty :class:`TraceGraph` (no preceding
    // :func:`on_op_enter`) is a no-op — defensive against ordering
    // mistakes at hook sites.  Overwrites any prior value under
    // ``key`` on the same node, so duplicate calls report the last
    // value (matches ``unordered_map`` semantics).
    void on_op_attr(std::string_view key, AttributeValue value);

    // Records ``output`` — just drawn by the RNG op ``name`` — as an
    // external feed instead of as that op's output, and drops the op's
    // node: the executable takes the draw as an input, and
    // :func:`redraw_rng_feeds` draws it again on every run.  See
    // :file:`RngFeeds.h`.
    void on_rng_feed(std::string_view name, const TensorImplPtr& output);

    // Marks the recording as one no executable can stand for (see
    // ``TraceGraph::unsupported``).  The first reason given is kept.
    void mark_unsupported(std::string why);

    // Whether a draw from the default generator becomes a feed drawn again
    // on every run (the compile path, :file:`RngFeeds.h`) — on by default —
    // or stays an op in the recording.  Core ML export turns it off: it
    // lifts each recorded draw into a model input, which needs the op.
    bool redraw_rng() const { return redraw_rng_; }
    void set_redraw_rng(bool on) { redraw_rng_ = on; }

    // Whether an op that reads nothing the trace sees is left out of the
    // recording: its output stays unseen, and an op that the trace does
    // record reads it as one constant feed, already on that op's device.
    // ``lucid.tensor([...]).to(x.device)`` then costs a feed, not a CPU op
    // in a Metal trace (which ran the whole call eagerly).  Off by default,
    // because "unseen" means "constant" only to a consumer that watches
    // every live tensor the call reads — its arguments and its model's
    // state.  The optimizer, decode and segmented tracers do not, and an
    // op on their state would be frozen at its trace-time value; Core ML
    // export folds on its own, knowing which shapes follow a flexible
    // input.  ``lucid.compile.compile`` turns it on.
    bool fold_constants() const { return fold_constants_; }
    void set_fold_constants(bool on) { fold_constants_ = on; }

    // ``out`` was made from nothing by the op now recording — a factory
    // (``zeros`` / ``arange`` / ``eye`` …), never a random draw.  With
    // folding on and ``out`` off Metal, the op is left out and ``out`` is a
    // constant; on Metal the compiled graph makes it itself, and otherwise
    // nothing changes.
    void on_factory(const TensorImplPtr& out);

    // Whether ``impl`` is a constant the trace made itself — a factory's
    // output, or a left-out op's — which no tensor outside the trace shares.
    bool made_here(const TensorImpl* impl) const {
        return made_.count(const_cast<TensorImpl*>(impl)) != 0;
    }

    // A value the trace knows — an op's output, an input, a parameter or a
    // random draw — read on the host (``item`` / ``tolist`` / ``numpy``,
    // and so ``float(t)`` or ``if t > 0``).  What the host does with it is
    // invisible to the recording, which would bake this call's value into
    // every later one: the trace is marked unsupported.  A tensor the trace
    // has not seen — a constant made on the spot — is left alone.
    void on_host_read(const TensorImpl* impl);

    // ``impl`` is an argument of the traced call.  Its values are the call's
    // data before any op has consumed it, so ``on_host_read`` treats a read
    // of it as a read of a traced value from the start of the trace.
    void watch(const TensorImplPtr& impl);

    // ``out`` was made from ``src`` outside the op dispatcher — a device
    // transfer, an alias with other autograd flags, a shared-buffer relabel
    // (the TensorImpl bindings, ``bind_tensor.cpp``).  When the trace knows
    // ``src``, ``out`` is recorded as op ``name`` reading it, so a later op
    // or host read of ``out`` reaches the traced value.  Unrecorded, ``out``
    // was a tensor the trace had never seen: a host read of it went
    // unnoticed, and an op reading it took it as a feed pinned at its
    // trace-time value — every compiled call answered with the first call's
    // data.  ``name`` is an emitter's when ``out`` holds ``src``'s values on
    // ``src``'s device (``"detach"``); any other name has no emitter and
    // sends the call to eager.  A ``src`` the trace has not seen is a
    // constant to it, and so is ``out``: nothing is recorded.  With several
    // sources (a fused kernel, a custom Metal kernel) the op reads all of
    // them, and is recorded when the trace sees any one.
    void on_derived(std::string_view name,
                    const std::vector<TensorImplPtr>& srcs,
                    const TensorImplPtr& out);

    // ``target`` now holds what ``value`` holds: an in-place op computed
    // ``value`` out of place and spliced it into ``target``
    // (``TensorImpl::take_storage_from`` / ``write_through``).  Later reads
    // of ``target`` resolve to ``value``'s id — without this, ``a.mul_(b)
    // .add_(c)`` recorded the add reading ``a`` from before the mul.  A
    // target from outside the trace keeps its feed id in ``external_feeds``
    // while its current id moves, which is how a write to a buffer is found
    // (``lucid/compile/_core/buffer_writes.py``); one the trace has not read
    // yet becomes a feed first.  A value the trace has not seen becomes a
    // feed of its own.
    void on_inplace_write(const TensorImplPtr& target, const TensorImplPtr& value);

    // Puts every tensor from outside the trace that the trace wrote in place
    // back to its value from before the trace, from the copy
    // :meth:`on_inplace_write` kept at the first write.  The trace-time call
    // ran those writes eagerly; whatever runs next — an executable or an
    // eager fallback — applies them itself, and would otherwise apply them
    // twice.  Call once the trace has finished.
    void restore_outside_writes();

    // ``(current id, tensor)`` for every tensor from outside the trace that
    // the trace wrote in place, in first-write order — what a compiled call
    // has to carry back.  Batch norm's running statistics are not among
    // them: its kernel leaves the buffers alone and hands the new values out
    // itself (:meth:`on_buffer_update`).
    std::vector<std::pair<TensorId, TensorImplPtr>> outside_writes() const;

    // Returns a read-only reference to the recorded graph.
    //
    // Returns
    // -------
    // const TraceGraph&
    //     The op DAG accumulated so far.  Caller must not access this
    //     reference after the Tracer is destroyed or a subsequent
    //     :func:`on_op_enter` invalidates iterators into ``ops``.
    const TraceGraph& graph() const { return graph_; }

    // Returns the id → external :type:`TensorImplPtr` map populated by
    // :func:`on_op_io` when an input wasn't produced by any earlier
    // traced op.  The builder uses this to materialise an MPSGraph
    // placeholder per external feed and to bind the corresponding
    // input data at execution time.
    //
    // The Tracer holds an owning :type:`TensorImplPtr` for every
    // external feed so the underlying buffers cannot be released
    // mid-trace.
    const std::unordered_map<TensorId, TensorImplPtr>& external_feeds() const {
        return external_feeds_;
    }

    // Every traced value that the tracer still holds, keyed by id.
    //
    // ``live_refs_`` already owns each :class:`TensorImpl` observed
    // during the trace, so this costs a walk rather than a copy.  It
    // exists for a consumer that wants a traced value's *contents* and
    // not only its shape — constant folding an export, where a
    // subgraph that depends on no input can be replaced by the value it
    // produced.
    //
    // A buffer written in place carries the id it had when the mapping
    // was last updated, which is the post-mutation state; a caller that
    // needs the earlier value must not use this.
    //
    // Returns
    // -------
    // std::unordered_map<TensorId, TensorImplPtr>
    //     Built fresh on each call.
    std::unordered_map<TensorId, TensorImplPtr> retained_values() const {
        std::unordered_map<TensorId, TensorImplPtr> out;
        out.reserve(live_refs_.size());
        for (const auto& impl : live_refs_) {
            if (!impl)
                continue;
            const auto it = impl_to_id_.find(impl.get());
            if (it != impl_to_id_.end())
                out.emplace(it->second, buffer_value(impl));
        }
        return out;
    }

    // Reverse lookup: find the :type:`TensorId` for ``impl`` if it was
    // observed during the trace (either as an external feed or as the
    // output of a traced op).  Returns ``-1`` when ``impl`` is not in
    // the trace — used by the Python compile harness to map the user's
    // return-value tensors back to graph ids so they can be passed to
    // :func:`compile_or_cached` as explicit graph outputs (which lets
    // the builder drop "unconsumed but Python-discarded" intermediates
    // like RNN's ``h_n`` tensor when the user only kept ``out``).
    TensorId lookup_id(const TensorImpl* impl) const {
        if (impl == nullptr)
            return TraceId::external_feed();
        const auto it = impl_to_id_.find(const_cast<TensorImpl*>(impl));
        return (it != impl_to_id_.end()) ? it->second : TraceId::external_feed();
    }

    // Whether ``impl`` is an op's output, a feed, or an argument of the traced call.
    bool sees(const TensorImpl* impl) const;

    // Number of recorded forward ops currently in ``graph_.ops``.
    std::size_t op_count() const { return graph_.ops.size(); }

    // Number of outside buffers mutated in-place so far in this trace.
    std::size_t outside_write_count() const { return outside_originals_.size(); }

    // Returns ``impl``'s existing :type:`TensorId` or mints an external-feed
    // id for it when ``impl`` has not been seen yet.
    TensorId ensure_id(const TensorImplPtr& impl);

    // Mints a symbolic cotangent :type:`TensorId` for ``impl`` without adding
    // ``impl`` to ``external_feeds_``.  Used while tracing a custom
    // :class:`Function`'s ``backward`` so the dummy cotangent tensor is bound
    // by :class:`BackwardWalker` from the upstream gradient tensor.
    TensorId register_cotangent(const TensorImplPtr& impl);

    // Drops any ops recorded at or after ``idx`` (used when tracing a custom
    // :class:`Function`'s ``backward`` fails and falls back to eager).
    void discard_ops_from(std::size_t idx);

    // Moves ``graph_.ops[bwd_begin .. end)`` into a :class:`VjpRegion`
    // covering forward ops ``[fwd_begin, fwd_end)`` and appends it to
    // ``graph_.vjp_regions``.
    void commit_vjp_region(std::size_t fwd_begin,
                           std::size_t fwd_end,
                           std::vector<TensorId> input_ids,
                           std::vector<TensorMeta> output_metas,
                           std::vector<TensorId> cotangent_ids,
                           std::vector<TensorId> input_grad_ids,
                           std::size_t bwd_begin);

private:
    // ``on_op_io`` for an op the recording keeps.
    void record_io(const std::vector<TensorImplPtr>& inputs, const TensorImplPtr& output);
    // With folding on, whether this ``on_op_io`` call is absorbed: every
    // input is a constant made inside this trace, so the op is held out.
    bool fold(const std::vector<TensorImplPtr>& inputs, const TensorImplPtr& output);
    // Notes what a held-out op made.
    void keep(const TensorImplPtr& output);
    // ``on_inplace_write`` for a target the trace made itself.
    void write_made(const TensorImplPtr& target, const TensorImplPtr& value);
    // ``impl``'s id, minting a feed for it when the trace has not seen it.
    TensorId feed_id(const TensorImplPtr& impl);
    // The id an in-place write's ``value`` goes by from now on.
    TensorId written_id(const TensorImplPtr& value);

    TraceGraph graph_;
    // Maps the raw :class:`TensorImpl` pointer of every traced tensor
    // (op output *or* external feed) back to the :type:`TensorId` it
    // carries inside the IR.  Used by :func:`on_op_io` to resolve
    // input ids — if the pointer isn't in this map yet, the op's
    // input is an external feed and a fresh id is minted on the spot.
    std::unordered_map<TensorImpl*, TensorId> impl_to_id_;
    // Trace-local values for state writes whose live storage stays untouched.
    std::unordered_map<TensorImpl*, TensorImplPtr> buffer_values_;
    // Owning subset of ``impl_to_id_``: every external feed (inputs
    // not produced by any earlier op) carries a strong reference so
    // its TensorImpl can't be released mid-trace.  The builder
    // consults this to build placeholders + record feed-order for
    // runtime data binding.
    std::unordered_map<TensorId, TensorImplPtr> external_feeds_;
    // Owning references to every TensorImpl observed during the trace
    // (both inputs and outputs).  Without this, short-lived tensors
    // (e.g. the zero-bias tensors materialised by
    // ``conv_bias_or_zero`` when ``Conv2d(bias=False)`` is used) can
    // be destroyed and their raw pointer recycled by the allocator
    // for the *next* op's output — at which point the recycled
    // pointer is already in ``impl_to_id_`` and gets resolved to the
    // wrong id.  Keeping every TensorImplPtr alive for the trace's
    // lifetime makes raw-pointer keys unambiguous.
    std::vector<TensorImplPtr> live_refs_;
    // (tensor from outside the trace, a copy of it from before its first
    // in-place write), in write order — see :meth:`restore_outside_writes`.
    std::vector<std::pair<TensorImplPtr, TensorImplPtr>> outside_originals_;
    // Owning: the call's arguments (``watch``), keyed by raw pointer.
    std::unordered_map<TensorImpl*, TensorImplPtr> watched_;
    bool redraw_rng_ = true;
    bool fold_constants_ = false;
    // The open op's node while it is held out of the recording; it goes
    // back in if a later ``on_op_io`` of the same op names a non-constant
    // input (an op that re-records its full input list), and is dropped at
    // the next op.
    std::optional<OpNode> held_;
    // Owning: every constant the trace made itself (see ``made_here``).
    std::unordered_map<TensorImpl*, TensorImplPtr> made_;
};

// Returns the :class:`Tracer` currently installed on the calling
// thread.
//
// Returns
// -------
// Tracer*
//     Non-owning pointer to the active tracer, or ``nullptr`` if no
//     tracer is installed.
//
// Notes
// -----
// The pointer lives in ``thread_local`` storage; each thread reads its
// own value.  Ownership of the pointed-to object remains with the
// caller of :func:`set_current_tracer`.
LUCID_API Tracer* current_tracer();

// Installs ``t`` as the active :class:`Tracer` for the calling thread.
//
// Parameters
// ----------
// t : Tracer*
//     Tracer to install, or ``nullptr`` to detach the active tracer
//     without destroying the object.
//
// Notes
// -----
// Ownership is not transferred — the caller keeps the Tracer alive
// for as long as any :class:`OpScopeFull` could observe the pointer.
// The Python binding holds the underlying ``shared_ptr`` for exactly
// this reason (matches :func:`lucid::set_current_profiler`).
//
// See Also
// --------
// :func:`current_tracer` — reader for the same TLS slot.
LUCID_API void set_current_tracer(Tracer* t);

// The two hooks every entry point outside the op dispatcher owes the
// active tracer, if one is installed.  A binding that hands Python a
// tensor's values (``item`` / ``tolist`` / ``numpy`` / bytes / a DLPack
// capsule) calls ``note_host_read`` first; one that makes a new tensor from
// an existing one returns it through ``traced_derive``.  The source scan in
// ``lucid/test/unit/compile/test_trace_entry_points.py`` holds every
// TensorImpl binding to this.
inline void note_host_read(const TensorImplPtr& impl) {
    if (auto* trc = current_tracer())
        trc->on_host_read(impl.get());
}

inline TensorImplPtr
traced_derive(std::string_view name, const std::vector<TensorImplPtr>& srcs, TensorImplPtr out) {
    if (auto* trc = current_tracer())
        trc->on_derived(name, srcs, out);
    return out;
}

inline TensorImplPtr
traced_derive(std::string_view name, const TensorImplPtr& src, TensorImplPtr out) {
    if (auto* trc = current_tracer())
        trc->on_derived(name, {src}, out);
    return out;
}

// The hook every factory owes the active tracer, if one is installed:
// ``out`` was made from nothing.  A random draw is not a factory — its value
// differs per call — and does not call this.
inline void traced_factory(const TensorImplPtr& out) {
    if (auto* trc = current_tracer())
        trc->on_factory(out);
}

}  // namespace lucid::compile
