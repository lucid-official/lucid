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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
    // ``graph_`` starts empty and ``impl_to_id_`` starts with no entries.  No
    // op hooks fire until the tracer is installed as the active TLS sink.
    Tracer();

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

    // Whether a backward will differentiate this recording, so grad mode decides
    // what it reaches.  Off by default: a forward-only tracer, Core ML export and
    // the optimizer's own recording run wholly under ``no_grad`` and answer no
    // question about gradients.  ``trace_context.recording`` turns it on when the
    // caller's grad mode is on, and ``fused_step`` turns it off again for the
    // update it records after the loss.
    bool differentiated() const { return differentiated_; }
    void set_differentiated(bool on) { differentiated_ = on; }

    // Whether what runs now is a constant to that backward: grad mode is off in a
    // recording to be differentiated.
    bool grad_stopped() const;

    // ``out`` was made from nothing by the op now open — a factory
    // (``zeros`` / ``arange`` / ``eye`` …), never a random draw.  With
    // folding on and ``out`` off Metal, the op is left out and ``out`` is a
    // constant; on Metal the compiled graph makes it itself where it has an
    // emitter (and otherwise reads it as a feed); and with folding off the
    // recording does not change.  Either way the call made ``out``
    // (:meth:`made_here`).
    void on_factory(const TensorImplPtr& out);

    // ``out`` was made from nothing by a binding that opens no op: it holds
    // host data (``lucid.tensor([...])``, an ndarray) or is a new buffer
    // (``shared_tensor``).  There is no op to leave out, and :meth:`on_factory`
    // would take whichever op ran before for it.
    void on_literal(const TensorImplPtr& out);

    // Whether the call itself made ``impl``, from nothing or from such tensors
    // alone, on any device and whether or not folding is on: a factory's
    // output, a literal, a left-out op's.  No tensor outside the trace shares
    // it and the next call makes another, so a write to it is not carried
    // back, and a feed reading it keeps its value from before the write.
    bool made_here(const TensorImpl* impl) const {
        return made_.count(const_cast<TensorImpl*>(impl)) != 0;
    }

    // Whether the value ``impl`` holds comes from the trace's constants alone
    // (:meth:`from_constants`): a table built by factories and pure ops, the same
    // on every call.  A tensor the trace never saw answers no — an argument
    // nothing read, a tensor from outside — as does anything that reads one of
    // those or draws.  For a caller that asks whether a value kept past the call
    // can differ from call to call.
    bool is_constant(const TensorImpl* impl) const;

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
    // constant when the call made it, and so is ``out``: nothing is recorded.
    // Made outside the call, it is an input: ``out`` that only relabels or
    // views its bytes (a detach, a shared-buffer relabel) needs no record, the
    // first op to read ``out`` feeds it and reads the bytes as they are; a
    // copy or a result computed from them (``to_shared``, a cross-device copy,
    // a fused or custom kernel) holds the values of the trace, so it is recorded
    // like a seen source and the source becomes a feed.  With several sources
    // (a fused kernel, a custom Metal kernel) the op reads all of them, and is
    // recorded when the trace sees any one, or any one is not the call's own.
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
    //
    // Where the write lands decides what happens to it, not what ``target``
    // is: a view the call took of a buffer from outside writes that buffer
    // (:meth:`lands_outside`), so the call owes the write like any other, and
    // carrying it back through ``target`` reaches every tensor reading those
    // bytes.
    void on_inplace_write(const TensorImplPtr& target, const TensorImplPtr& value);

    // ``view`` reads the buffer :meth:`on_inplace_write` was just told of,
    // and the recording does not carry the write to it: eager reads the new
    // values through the shared buffer, while the id the trace holds for
    // ``view`` names the ones from before.  Most such views are temporaries
    // nobody reads again (``self.buf[i] = self.buf[i] + m`` leaves two), so
    // nothing is recorded yet; a later read of ``view`` — an op's input, the
    // value an in-place write copies in, the source of a derived tensor, a
    // host read, a value the call returns (:meth:`read_id`) — marks the trace
    // unsupported and the call runs eager.  A view that is written itself
    // (``on_inplace_write`` on it, as for a Metal view that covers its whole
    // buffer) is current again.  Call before the write changes the buffer:
    // a constant of the trace that an earlier op read as a feed keeps a copy
    // of the values from before it.
    void on_stale_view(const TensorImplPtr& view);

    // :meth:`lookup_id` for a value the traced call hands back or keeps (a
    // result, a loss): reading a stale view marks the trace unsupported
    // first, instead of answering the id of its value from before the write
    // (which a caller takes for a value that flows out untouched).  Binding
    // an argument to its id is not a read of its value, and stays
    // ``lookup_id``.
    TensorId read_id(const TensorImpl* impl);

    // ``stand_in`` holds what ``of`` holds now, under a tensor of its own: an
    // in-place op recording a graph runs its forward against one
    // (``inplace::snapshot``), so that its node keeps the values ``of`` is
    // about to lose.  It goes by ``of``'s current id, for good: what the op
    // reads is the value ``of`` holds now, computed by whatever computed it.
    // Unnamed, it was a tensor the trace had never seen, so the op read a feed
    // pinned at the trace-time value, and a feed that requires grad besides:
    // the step backpropagated through the trace-time call's graph.
    void on_stand_in(const TensorImplPtr& stand_in, const TensorImplPtr& of);

    // Puts every tensor from outside the trace that the trace wrote in place
    // back to its value and place in the autograd graph from before the trace,
    // from the copy :meth:`on_inplace_write` kept at the first write.  The trace-time call
    // ran those writes eagerly; whatever runs next — an executable or an
    // eager fallback — applies them itself, and would otherwise apply them
    // twice.  Call once the trace has finished.
    void restore_outside_writes();

    // Marks the trace unsupported if a tensor the call made and wrote in place,
    // or a view of it, is still ``held`` by something outside the trace.  The
    // trace took it for the call's own, starting over every time; whoever holds
    // it (``self.h = lucid.tensor(...)``, once) expects each call to go on from
    // the last.  Call once the trace has finished and the call's result is
    // dropped: a result is handed back on every run and holds nothing over.
    //
    // A write whose value comes from the trace's constants alone
    // (:meth:`from_constants`) is left as it is: a cached or lazily built table
    // holds the same values whoever filled it.  What the walk cannot decide
    // counts as an input.  A table added to on every call by a constant
    // (``h.add_(1.0)``) stays frozen at its trace-time value, as before
    // (LCD-864).
    //
    // ``stands``: whether the caller can let the trace-time call stand as the
    // call, its result and writes the call's.  Then every other write counts.
    // A caller that cannot (a segmented step, a step with a cumulative batch
    // norm) has only an eager rerun from the tensor put back, and counts only
    // a write that reads the tensor's own earlier value
    // (:meth:`reads_its_own_value`): an overwrite lands the same values
    // however often it runs.  One that runs only on the first call (a buffer
    // filled from the first batch) is compiled into every call there, as
    // before (LCD-876).  ``compiled_step`` does not ask: it traces every call,
    // and the trace's writes are that call's.
    //
    // Answers ``(tensor, value before the trace's write, value after it)`` for
    // each such tensor and changes none of their values: the caller either lets
    // the trace stand as the call, or puts them back for an eager call to write
    // (``lucid/compile/_core/buffer_writes.py``).  Without ``stands`` they leave
    // the graph of the trace-time call here.
    std::vector<std::tuple<TensorImplPtr, TensorImplPtr, TensorImplPtr>>
    refuse_kept_writes(const std::function<bool(const TensorImpl*)>& held, bool stands);

    // Holds the call's result while ``refuse_kept_writes`` asks who else holds
    // what the call made: in C++, the result is not one of those holders.
    void keep_answer(std::vector<TensorImplPtr> answer) { answer_ = std::move(answer); }
    std::vector<TensorImplPtr> take_answer() { return std::exchange(answer_, {}); }

    // Writes into each buffer the value a kernel handed out for it instead of
    // writing it (:meth:`on_buffer_update`: batch norm's running statistics, a
    // functional one's too).  A trace that stands as the call does this, or the
    // statistics miss that call's update for good.
    void apply_buffer_updates() const;

    // Whether :meth:`restore_outside_writes` keeps the written values for
    // :meth:`retained_values`, for a caller that reads a traced value's contents
    // (Core ML, which folds them).  Off, the compiled paths pay for no copy.
    bool keep_written_values() const { return keep_written_values_; }
    void set_keep_written_values(bool on) { keep_written_values_ = on; }

    // ``(current id, tensor)`` for every write that landed outside the trace,
    // in first-write order — what a compiled call has to carry back.  The
    // tensor is the one written, which may be a view the call took of the
    // buffer; a view whose family also has a written tensor that is the whole
    // buffer is left out, as that tensor's value already holds its write.
    // Batch norm's running statistics are not among them: its kernel leaves
    // the buffers alone and hands the new values out itself
    // (:meth:`on_buffer_update`).
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
    // was last updated, which is the post-mutation state, also once
    // :meth:`restore_outside_writes` has put the tensor back; a caller that
    // needs the earlier value must not use this.
    //
    // Returns
    // -------
    // std::unordered_map<TensorId, TensorImplPtr>
    //     Built fresh on each call.
    std::unordered_map<TensorId, TensorImplPtr> retained_values() const;

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
    // Gives ``output``, first seen as something the trace produced, its id.
    void claim(const TensorImplPtr& output, TensorId id);
    // Notes what a held-out op made.
    void keep(const TensorImplPtr& output);
    // Notes that the call made ``out`` from nothing, and whether folding may
    // absorb what reads it.
    void note_made(const TensorImplPtr& out);
    // Marks the trace unsupported when ``feed``, about to be read from outside
    // the trace, was computed during it: its node is younger than the trace.
    // Whatever made it told the tracer nothing, and a compiled call would answer
    // with this call's value.
    void refuse_computed_feed(const TensorImpl& feed);
    // Marks the trace unsupported when ``target`` or a view of it holds a gradient
    // hook at the place in the graph the write about to land moves it from.
    void refuse_hooked_write(const TensorImplPtr& target);
    // Marks the trace unsupported when a write autograd does not track moves a value
    // the recording cannot keep apart from the gradient: eager leaves ``target``'s
    // place in the graph as it was and takes the new values as a constant, and a
    // recorded value has one id, either differentiated or not.
    void refuse_untracked_write(const TensorImplPtr& target, const TensorImplPtr& value);
    // Notes the place in the autograd graph of each tensor from outside the trace
    // in ``target``'s family, for :meth:`restore_outside_writes`.
    void keep_graph_positions(const TensorImplPtr& target);
    // Puts those tensors back in their places, and takes what the trace made of
    // a written family off the graph.
    void put_back_graph_positions();
    // ``on_inplace_write`` for a target the trace made itself.
    void write_made(const TensorImplPtr& target, const TensorImplPtr& value);
    // Whether ``impl`` reached the trace from outside it: neither made by the
    // call nor first seen as an op's output.  Asked of a tensor, not of its
    // current id, which a write moves.
    bool from_outside(const TensorImpl* impl) const;
    // Whether a write to ``target`` lands in bytes a tensor from outside the
    // trace reads: ``target`` is one, or a view of the same buffer is.  A view
    // made in the trace of an outside buffer holds no bytes of its own.
    bool lands_outside(const TensorImplPtr& target) const;
    // Whether ``target`` is a view of a tensor the call made.
    bool lands_in_made(const TensorImplPtr& target) const;
    // What :meth:`from_constants` has worked out so far, and the op that made
    // each id; one per :meth:`refuse_kept_writes`.
    struct ConstantWalk {
        std::unordered_map<TensorId, const OpNode*> producer;
        std::unordered_map<TensorId, bool> known;
    };
    // Whether the value ``id`` names comes from the trace's constants alone: every
    // op on the way is one :func:`pure_op` knows, and every feed reached is a
    // tensor the call made (``constant_feed``).  An id nothing recorded (a write
    // folded away) is a constant: folding only holds out an op whose every input
    // the call made.  Anything else — an argument, a buffer, a draw, an op it
    // does not know — answers no.
    bool from_constants(TensorId id, ConstantWalk& walk) const;
    // Fills ``walk.producer`` once.
    void index_producers(ConstantWalk& walk) const;
    // Records ``was`` as an earlier value of ``target`` (``prior_ids_``), for a
    // tensor the call made or a view of one, unless the write is one already
    // recorded (``now`` is ``was``).
    void note_prior(const TensorImplPtr& target, TensorId was, TensorId now, bool in_made);
    // Whether the value ``target`` was last written reads a value the tensor, or
    // a view of it, held before (``prior_ids_``): a read-modify-write, which
    // counts its runs.  A value that keeps an earlier one whole, through views
    // or as the base an overwrite at constant positions writes into (a
    // setitem's scatter), reads what that earlier value read.  The base of an
    // overwrite whose positions are not constants is read: the positions a call
    // leaves alone hold what earlier calls wrote there.
    bool reads_its_own_value(const TensorImplPtr& target, ConstantWalk& walk) const;
    // Whether operand ``i`` of ``node`` passes into its result whole, so an
    // earlier value reached through it is kept, not read (see
    // :meth:`reads_its_own_value`).
    bool keeps_operand(const OpNode& node, std::size_t i, ConstantWalk& walk) const;
    // One step of that walk: settles ``id`` when what it reads is known, and
    // otherwise pushes what is not and answers false.
    bool decide(TensorId id, ConstantWalk& walk, std::vector<TensorId>& stack) const;
    // Whether feed ``id`` holds a tensor the call made, or a copy of one, and no
    // other tensor shares its bytes.
    bool constant_feed(TensorId id) const;
    // Notes that feed ``id`` now holds a copy of ``of``, which the call made.
    void note_constant_feed(TensorId id, const TensorImpl& of);
    // ``on_inplace_write`` for a write that :meth:`lands_outside`: keeps the
    // value of what can put it back.
    void keep_outside(const TensorImplPtr& target);
    // The ``wanted`` member of ``target``'s Metal family (``target`` itself
    // first) that is the whole buffer and that a write through ``target``
    // re-derives; ``nullptr`` when there is none.
    TensorImplPtr whole_member(const TensorImplPtr& target,
                               const std::function<bool(const TensorImpl*)>& wanted) const;
    // Whether the compiled call owes ``target``'s write itself: no written
    // tensor that is ``target``'s whole buffer holds it already.
    bool carried(const TensorImplPtr& target) const;
    // Whether ``impl`` is a Metal shared buffer under a name outside the view
    // family the trace last wrote it through.  Two such tensors share nothing
    // but the bytes, so a write to one hands no stale view to the tracer, and
    // the bytes a run reads are from before the write.
    bool reads_a_shared_write(const TensorImpl* impl) const;
    // ``impl``'s id, minting a feed for it when the trace has not seen it.
    TensorId feed_id(const TensorImplPtr& impl);
    // The id an in-place write's ``value`` goes by from now on.
    TensorId written_id(const TensorImplPtr& value);
    // Marks the trace unsupported when ``impl`` is a stale view.
    void refuse_stale(const TensorImpl* impl);

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
    // A tensor's place in the autograd graph.
    struct GraphPosition {
        std::shared_ptr<Node> grad_fn;
        std::uint32_t output_nr = 0;
        bool leaf = true;
        bool requires_grad = false;
    };
    // Owning: each tensor from outside the trace in a family the trace wrote, and
    // where it sat before the first write (see :meth:`keep_graph_positions`).
    std::unordered_map<TensorImpl*, std::pair<TensorImplPtr, GraphPosition>> outside_positions_;
    // The same for what the call made — see :meth:`refuse_kept_writes`.
    std::vector<std::pair<TensorImplPtr, TensorImplPtr>> made_originals_;
    // Feeds that hold a copy of a tensor the call made (see ``constant_feed``).
    std::unordered_set<TensorId> constant_feeds_;
    // The call's result while the trace is judged (see ``keep_answer``).
    std::vector<TensorImplPtr> answer_;
    // For each tensor the call made (or a view of one) that it wrote, the ids
    // it went by before each write (see ``reads_its_own_value``).
    std::unordered_map<const TensorImpl*, std::vector<TensorId>> prior_ids_;
    // (buffer, value) for each :meth:`on_buffer_update`, in order.
    std::vector<std::pair<TensorImplPtr, TensorImplPtr>> buffer_updates_;
    bool keep_written_values_ = false;
    // Owning: the call's arguments (``watch``), keyed by raw pointer.
    std::unordered_map<TensorImpl*, TensorImplPtr> watched_;
    bool redraw_rng_ = true;
    bool fold_constants_ = false;
    bool differentiated_ = false;
    // The open op's node while it is held out of the recording; it goes
    // back in if a later ``on_op_io`` of the same op names a non-constant
    // input (an op that re-records its full input list), and is dropped at
    // the next op.
    std::optional<OpNode> held_;
    // Owning: every tensor the call made itself (see ``made_here``).
    std::unordered_map<TensorImpl*, TensorImplPtr> made_;
    // The part of ``made_`` folding may absorb an op reading (``fold_constants``).
    // ``made_`` owns them.
    std::unordered_set<TensorImpl*> foldable_;
    // Every tensor first seen as an op's output, or as a draw; ``live_refs_``
    // owns them.  The complement of this and ``made_`` is what came from outside.
    std::unordered_set<TensorImpl*> produced_;
    // For each Metal shared buffer the trace wrote, the tensor it last wrote it
    // through (see ``reads_a_shared_write``).
    std::unordered_map<const void*, TensorImplPtr> shared_writers_;
    // Owning: the views of a tensor written in place whose recorded value is
    // from before the write (see ``on_stale_view``).  Owning so a freed view's
    // address cannot be taken for a stale one.
    std::unordered_map<TensorImpl*, TensorImplPtr> stale_;
    // The lowest number an autograd node made during this trace can have.
    std::uint64_t first_node_;
    // Whether the in-place write being followed puts a constant of the trace
    // into its target: a view of a constant that no op has read yet then holds
    // that constant, and is not stale.  Only a value the trace folded counts;
    // one an op recorded (a scalar operand is a ``full`` op on Metal) does not,
    // and its views are marked: a fallback at worst, never a wrong answer.
    bool constant_write_ = false;
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

// Detaches the calling thread's tracer for a scope: what runs in it is no op of
// the recording.
class TracerDetach {
public:
    TracerDetach() : saved_(current_tracer()) { set_current_tracer(nullptr); }
    ~TracerDetach() { set_current_tracer(saved_); }
    TracerDetach(const TracerDetach&) = delete;
    TracerDetach& operator=(const TracerDetach&) = delete;

private:
    Tracer* saved_;
};

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
// ``out`` was made from nothing, by the op whose scope is open.  A random draw
// is not a factory — its value differs per call — and does not call this.
inline void traced_factory(const TensorImplPtr& out) {
    if (auto* trc = current_tracer())
        trc->on_factory(out);
}

// The same for a binding that makes a tensor from nothing and opens no op.
inline void traced_literal(const TensorImplPtr& out) {
    if (auto* trc = current_tracer())
        trc->on_literal(out);
}

}  // namespace lucid::compile
