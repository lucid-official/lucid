// lucid/_C/compile/Tracer.cpp
//
// Implementation of the thread-local op tracer for :mod:`lucid.compile`.
//
// The active Tracer for each thread lives in a single thread_local pointer
// (g_current).  :class:`OpScopeFull` reads this pointer at construction time
// and, if non-null, forwards the entering op into :func:`Tracer::on_op_enter`.
// Outside any ``_tracing()`` scope the pointer is nullptr and the hook
// short-circuits to a single TLS load — matching the un-profiled fast path
// described in :file:`Profiler.cpp`.
//
// Phase 1.1 records OpNode headers only — name, output dtype/shape/device,
// and a freshly-minted output TensorId.  ``inputs`` stays empty; Phase 1.2
// backfills it from the shared :func:`wire_autograd` site.

#include "Tracer.h"

#include <algorithm>
#include <set>

#include "../autograd/Node.h"
#include "../autograd/TensorHooks.h"
#include "../backend/Dispatcher.h"
#include "../core/GradMode.h"
#include "../core/StridedCopy.h"
#include "../core/TensorImpl.h"

namespace lucid::compile {

namespace {
// Per-thread active Tracer.  nullptr means "no tracing" for this thread —
// the OpScopeFull hook does nothing.
thread_local Tracer* g_current = nullptr;

// A copy of ``impl``'s current values, owned by no one else.
TensorImplPtr snapshot(const TensorImplPtr& impl) {
    Storage copy = backend::Dispatcher::for_device(impl->device())
                       .clone(impl->storage(), impl->shape(), impl->dtype());
    return std::make_shared<TensorImpl>(std::move(copy), impl->shape(), impl->dtype(),
                                        impl->device(), false);
}

// Whether ``t``'s elements are all of its buffer: what a write through any
// other view of the buffer changes, ``t``'s value includes.
bool whole_buffer(const TensorImpl& t) {
    return t.allocation_offset() == 0 && t.is_contiguous() && t.nbytes() == t.allocation_nbytes();
}

// Whether ``a`` and ``b`` read bytes of one buffer: members of one view family
// (a detach, a reshape, ``_make_view``) or two labels of one Metal shared
// buffer (a relabel between devices).  A new buffer made from another's
// values (a copy to shared storage, a cross-device copy, a fused kernel's
// result) does not.
bool shares_bytes(const TensorImpl& a, const TensorImpl& b) {
    if (a.family_id() != 0 && a.family_id() == b.family_id())
        return true;
    const SharedStorage* buffer = a.shared_buffer();
    return buffer != nullptr && b.shared_buffer() != nullptr &&
           buffer->cpu_ptr == b.shared_buffer()->cpu_ptr;
}

// Whether two of ``t``'s elements are one byte (``expand``, ``unfold``): only
// ``fill_`` writes through such a view, and ``copy_from`` refuses it.
bool repeats_elements(const TensorImpl& t) {
    return !t.is_contiguous() && strided::overlaps_itself(t.shape(), t.stride());
}

// Takes ``t`` off the autograd graph.  A write recording a graph gives a tensor
// a place in the trace-time call's graph: one put back has to leave it, or
// whatever runs next chains onto that call.
void leave_graph(TensorImpl& t) {
    if (!t.requires_grad() && !t.grad_fn())
        return;
    t.clear_grad_fn();
    t.set_grad_output_nr(0);
    t.set_leaf(true);
    t.set_requires_grad(false);
}

// Keeps ``target``'s current value in ``originals`` unless it is there already;
// answers the copy when it made one.
TensorImplPtr keep_original(std::vector<std::pair<TensorImplPtr, TensorImplPtr>>& originals,
                            const TensorImplPtr& target) {
    if (std::any_of(originals.begin(), originals.end(),
                    [&](const auto& kv) { return kv.first.get() == target.get(); }))
        return nullptr;
    TensorImplPtr before = snapshot(target);
    originals.emplace_back(target, before);
    return before;
}
}  // namespace

Tracer* current_tracer() {
    return g_current;
}

void set_current_tracer(Tracer* t) {
    g_current = t;
}

Tracer::Tracer() : first_node_(next_sequence_nr()) {}

void Tracer::on_op_enter(std::string_view name, Device device, Dtype dtype, Shape shape) {
    // The op before this one is over; if it was held out, it stays out, and
    // the id it drew goes back when nothing drew one since.  A trace then
    // numbers its values as if the folded ops had never run, so two calls
    // that differ only in how they build a constant draw the same ids.
    if (held_) {
        if (!held_->outputs.empty() && held_->outputs[0].id + 1 == graph_.next_id)
            graph_.next_id = held_->outputs[0].id;
        held_.reset();
    }
    // Mint a fresh TensorId for the (single, Phase-1.1) output.  Phase 1.2 will
    // append additional TensorMeta entries here for multi-output ops; the
    // append-after-mint pattern keeps ids monotone within one TraceGraph.
    const TensorId out_id = graph_.next_id++;

    OpNode node;
    node.name = std::string(name);
    // inputs filled by on_op_io if the op goes through wire_autograd.
    node.outputs.push_back(TensorMeta{out_id, std::move(shape), dtype, device});
    if (grad_stopped())
        node.attrs.emplace(kGradOffAttr, true);
    graph_.ops.push_back(std::move(node));
}

bool Tracer::grad_stopped() const {
    return differentiated_ && !GradMode::is_enabled();
}

void Tracer::on_op_io(const std::vector<TensorImplPtr>& inputs, const TensorImplPtr& output) {
    // Before folding: a held-out op reads a stale view as the constant it
    // holds now, which the next call would not recompute.
    for (const auto& input : inputs)
        refuse_stale(input.get());
    // Then, since a held-out op has no node in ``graph_.ops``.
    if (fold_constants_ && fold(inputs, output))
        return;
    if (output)
        stale_.erase(output.get());  // an op defined it again: no longer from before a write
    record_io(inputs, output);
}

void Tracer::record_io(const std::vector<TensorImplPtr>& inputs, const TensorImplPtr& output) {
    // Defensive: if on_op_enter never ran for this op (e.g. an unexpected
    // hook ordering), there is no node to attach I/O to.  Drop silently
    // rather than corrupting the IR.
    if (graph_.ops.empty())
        return;

    // Hold owning references to every TensorImpl we see for the rest
    // of the trace's lifetime.  Short-lived tensors (e.g. the
    // zero-bias materialised by ``conv_bias_or_zero``) would otherwise
    // be destroyed before the trace ends, freeing their raw pointer
    // back to the allocator pool — the next allocation in the same
    // trace can then reuse that pointer, and the stale
    // ``impl_to_id_`` entry would resolve to the wrong id.
    for (const auto& inp : inputs) {
        if (inp)
            live_refs_.push_back(inp);
    }
    if (output)
        live_refs_.push_back(output);

    OpNode& node = graph_.ops.back();

    // Resolve each input id.
    //   - nullptr slot (e.g. an optional bias) → -1 sentinel.
    //   - Pointer already known (output of an earlier traced op) →
    //     look up its id in ``impl_to_id_``.
    //   - Pointer not yet known → external feed.  Mint a fresh id,
    //     remember it both in ``impl_to_id_`` (so a later op that
    //     consumes the same buffer resolves to the same id) and in
    //     ``external_feeds_`` (with an owning :type:`TensorImplPtr`
    //     so the builder can materialise an MPSGraph placeholder and
    //     bind the input data at run time without races against
    //     buffer release).
    // Last-write-wins semantics: callers (e.g. ops with non-diff
    // inputs like ``embedding``'s indices) may invoke ``on_op_io``
    // a second time after the autograd kernel's internal call to
    // fill in the full graph-level input list.  Clear here so the
    // most recent call is authoritative; append would otherwise
    // duplicate ids and confuse the emitter's resolve step.
    //
    // BUT only when this call actually carries inputs.  A multi-output op
    // (``lstm`` returns out / h_n / c_n) registers its extra outputs with
    // *follow-up* ``on_op_io({}, piece)`` calls that pass an empty input
    // list — those are output-only registrations, not a re-record of the
    // inputs.  Clearing on them would wipe the inputs recorded by the first
    // call and leave the emitter with a zero-input node (eager fallback).
    // Genuine zero-input ops (factories) record their single empty call and
    // their inputs stay empty regardless.
    if (!inputs.empty()) {
        node.inputs.clear();
        node.inputs.reserve(inputs.size());
        for (const auto& inp : inputs) {
            TensorImpl* raw = inp.get();
            if (raw == nullptr) {
                node.inputs.push_back(-1);
                continue;
            }
            node.inputs.push_back(feed_id(inp));
        }
    }

    // Register the output's identity so the next op that consumes it can
    // resolve its input id.  In-place ops (output pointer aliases an
    // input) intentionally overwrite the prior mapping — the new id now
    // represents the post-mutation state of the buffer, which is what
    // any later consumer should see.
    //
    // Multi-output handling: a single op (split / split_at / unbind /
    // chunk / lstm / topk …) may call ``on_op_io`` multiple times within
    // one ``OpScopeFull``, once per piece.  ``on_op_enter`` always seeds
    // ``node.outputs`` with a single placeholder TensorMeta.  We detect
    // "is this the first ``on_op_io`` call for this op?" by checking
    // whether the placeholder's id has been claimed by any impl in
    // ``impl_to_id_``.  First call → update ``outputs[0]`` (existing
    // single-output semantics).  Subsequent calls → append a fresh
    // ``TensorMeta`` so each piece gets its own id; the emit-side
    // emitter binds piece-N's MPSGraphTensor to ``outputs[N].id`` via
    // :func:`BuilderContext::bind`.
    if (output && !node.outputs.empty()) {
        // In-place / alias detection: if this output impl is already
        // mapped, just overwrite the mapping to the latest "post-
        // mutation" id (preserves the original single-output in-place
        // semantics).
        const auto existing = impl_to_id_.find(output.get());
        if (existing != impl_to_id_.end()) {
            existing->second = node.outputs[0].id;
            // Refresh the metadata too — same rationale as below.
            node.outputs[0].shape = output->shape();
            node.outputs[0].dtype = output->dtype();
            node.outputs[0].device = output->device();
        } else {
            // Has any prior impl claimed outputs[0].id?  Walk
            // ``impl_to_id_`` (small map — order of dozens) once.
            const TensorId slot0_id = node.outputs[0].id;
            bool first_call = true;
            for (const auto& kv : impl_to_id_) {
                if (kv.second == slot0_id) {
                    first_call = false;
                    break;
                }
            }
            if (first_call) {
                // First on_op_io call for this op — update outputs[0].
                // OpScopeFull ctor records the shape that was known at
                // scope entry — for some ops (tile / repeat / pad …)
                // that's stale by the time the actual output is
                // materialised.  Refresh it from the real output so
                // downstream emitters get the right shape / dtype /
                // device meta.
                claim(output, slot0_id);
                node.outputs[0].shape = output->shape();
                node.outputs[0].dtype = output->dtype();
                node.outputs[0].device = output->device();
            } else {
                // Subsequent call — multi-output op.  Mint a fresh id
                // and append a new TensorMeta so the emitter can bind
                // this piece independently.
                const TensorId new_id = graph_.next_id++;
                node.outputs.push_back(
                    TensorMeta{new_id, output->shape(), output->dtype(), output->device()});
                claim(output, new_id);
            }
        }
    }
}

void Tracer::claim(const TensorImplPtr& output, TensorId id) {
    impl_to_id_[output.get()] = id;
    produced_.insert(output.get());
}

void Tracer::on_buffer_update(const TensorImplPtr& buffer, const TensorImplPtr& value) {
    if (!buffer || !value)
        return;
    const auto it = impl_to_id_.find(value.get());
    if (it != impl_to_id_.end()) {
        impl_to_id_[buffer.get()] = it->second;
        buffer_values_[buffer.get()] = value;
        buffer_updates_.emplace_back(buffer, value);
        // The kernel leaves the buffer's bytes alone and hands the new values
        // out, so no hook sees this write; eager has updated every view of it.
        // The values come from the input: never the constant write an earlier
        // in-place op may have left ``constant_write_`` saying.
        constant_write_ = false;
        for (const auto& view : buffer->live_views())
            on_stale_view(view);
    }
}

void Tracer::on_rng_feed(std::string_view name, const TensorImplPtr& output) {
    if (!output)
        return;
    if (!graph_.ops.empty() && graph_.ops.back().name == name && graph_.ops.back().inputs.empty())
        graph_.ops.pop_back();
    live_refs_.push_back(output);
    const TensorId fresh = graph_.next_id++;
    claim(output, fresh);
    external_feeds_[fresh] = output;
}

void Tracer::on_inplace_write(const TensorImplPtr& target, const TensorImplPtr& value) {
    if (!target || !value || target.get() == value.get())
        return;
    refuse_hooked_write(target);
    if (differentiated_ && (!GradMode::is_enabled() || target->is_detached_alias()))
        refuse_untracked_write(target, value);
    // Copying ``value`` in reads it.
    refuse_stale(value.get());
    constant_write_ = made_here(value.get()) && !sees(value.get());
    // ``target`` holds ``value`` now, whatever it held before.
    stale_.erase(target.get());
    if (const SharedStorage* buffer = target->shared_buffer())
        shared_writers_[buffer->cpu_ptr] = target;
    if (made_here(target.get())) {
        write_made(target, value);
        return;
    }
    // Landing in bytes a tensor from outside reads: keep what puts them back.
    const auto made = [&](const TensorImpl* impl) { return made_here(impl); };
    const bool in_made = !lands_outside(target) && lands_in_made(target);
    if (!in_made && lands_outside(target)) {
        keep_graph_positions(target);
        keep_outside(target);
    } else if (in_made && !whole_member(target, made)) {
        keep_original(made_originals_, target);
    }
    const TensorId was = feed_id(target);
    const TensorId now = written_id(value);
    note_prior(target, was, now, in_made);
    impl_to_id_[target.get()] = now;
}

void Tracer::refuse_hooked_write(const TensorImplPtr& target) {
    // The hook stays on the slot ``target`` is about to leave, where the scan
    // at the end of the recording, which asks each tensor about the slot it
    // holds then, no longer finds it.  So does a view the write rebases.
    const auto members = target->live_views();
    if (has_tensor_hooks(*target) ||
        std::any_of(members.begin(), members.end(),
                    [](const TensorImplPtr& member) { return has_tensor_hooks(*member); }))
        mark_unsupported("a gradient hook sits on a value an in-place write replaces, and a "
                         "compiled backward runs no hook");
}

void Tracer::refuse_untracked_write(const TensorImplPtr& target, const TensorImplPtr& value) {
    const auto members = target->live_views();
    if (value->requires_grad() || target->requires_grad() ||
        std::any_of(members.begin(), members.end(),
                    [](const TensorImplPtr& member) { return member->requires_grad(); }))
        mark_unsupported("a write autograd does not track (under no_grad, or through .data or "
                         ".detach()) reached a tensor a gradient flows through, or copied one "
                         "in, and a compiled backward has one value there");
}

void Tracer::note_prior(const TensorImplPtr& target, TensorId was, TensorId now, bool in_made) {
    // A write is noted twice when the result is spliced and then written
    // through (``take_storage_from``): the second time the tensor already goes
    // by the value's id, which is no earlier value of it.
    if (in_made && was != now)
        prior_ids_[target.get()].push_back(was);
}

TensorImplPtr Tracer::whole_member(const TensorImplPtr& target,
                                   const std::function<bool(const TensorImpl*)>& wanted) const {
    if (target->family_key() == nullptr)
        return nullptr;
    if (wanted(target.get()) && whole_buffer(*target))
        return target;
    for (const auto& member : target->live_views()) {
        if (wanted(member.get()) && whole_buffer(*member) && member->dtype() == target->dtype())
            return member;
    }
    return nullptr;
}

void Tracer::keep_outside(const TensorImplPtr& target) {
    const auto outside = [&](const TensorImpl* impl) { return from_outside(impl); };
    // A write through a Metal view re-derives the member that is the whole
    // buffer (``trace_gpu_member_write``): its value holds the write and its
    // copy puts it back, however many temporaries reach it.  Every ``buf[i] =
    // v`` makes two, and each would be a copy of its own and an output on
    // every run.
    if (const TensorImplPtr carrier = whole_member(target, outside)) {
        keep_original(outside_originals_, carrier);
        return;
    }
    if (!repeats_elements(*target)) {
        keep_original(outside_originals_, target);
        return;
    }
    // The compiled call has no way to copy a value into such a view, and neither
    // does the restore: the tensors it was taken from put its write back.
    mark_unsupported("an in-place write through a view whose elements overlap (expand, unfold) "
                     "has no one value to carry back to the next call");
    for (const auto& member : target->live_views()) {
        if (from_outside(member.get()) && !repeats_elements(*member))
            keep_original(outside_originals_, member);
    }
}

void Tracer::keep_graph_positions(const TensorImplPtr& target) {
    // A tensor from outside was there before the trace: it is in the family at
    // the family's first write, with the place it had then, and the first entry
    // stands.  The write about to land moves every member (``rebase_views``).
    const auto keep = [&](const TensorImplPtr& t) {
        if (from_outside(t.get()))
            outside_positions_.try_emplace(
                t.get(), t,
                GraphPosition{t->grad_fn(), t->grad_output_nr(), t->is_leaf(), t->requires_grad()});
    };
    keep(target);
    for (const auto& member : target->live_views())
        keep(member);
}

void Tracer::put_back_graph_positions() {
    // What the trace took of a written family has no place from before it.
    const auto leave = [&](const TensorImplPtr& t) {
        if (!from_outside(t.get()))
            leave_graph(*t);
    };
    for (const auto& [target, before] : outside_originals_) {
        leave(target);
        for (const auto& member : target->live_views())
            leave(member);
    }
    for (const auto& [raw, kept] : outside_positions_) {
        const auto& [t, position] = kept;
        if (!position.requires_grad && !position.grad_fn && !t->requires_grad() && !t->grad_fn())
            continue;
        t->set_grad_fn(position.grad_fn);
        t->set_grad_output_nr(position.output_nr);
        t->set_leaf(position.leaf);
        t->set_requires_grad(position.requires_grad);
    }
}

bool Tracer::lands_in_made(const TensorImplPtr& target) const {
    const auto members = target->live_views();
    return std::any_of(members.begin(), members.end(),
                       [&](const TensorImplPtr& member) { return made_here(member.get()); });
}

std::vector<std::tuple<TensorImplPtr, TensorImplPtr, TensorImplPtr>>
Tracer::refuse_kept_writes(const std::function<bool(const TensorImpl*)>& held, bool stands) {
    std::vector<std::tuple<TensorImplPtr, TensorImplPtr, TensorImplPtr>> kept;
    ConstantWalk walk;
    for (auto it = made_originals_.rbegin(); it != made_originals_.rend(); ++it) {
        const auto members = it->first->live_views();
        if (!held(it->first.get()) &&
            std::none_of(members.begin(), members.end(),
                         [&](const TensorImplPtr& member) { return held(member.get()); }))
            continue;
        // A write that reads the tensor's own earlier value (``add_``,
        // ``h.copy_(h * 2)``) is not idempotent: each call has to start from
        // what the last one left, and a compiled call rebuilds the tensor from
        // its factory every time, so the accumulate never carries.  The other
        // operands being constants changes nothing (LCD-864).
        const bool reads = reads_its_own_value(it->first, walk);
        // Written from the trace's constants alone, without reading its own
        // value, the tensor holds the same values whoever wrote it and however
        // often: a cached or lazily built table.  The holder keeps them;
        // putting them back for an eager call that never writes them again
        // would empty the table for good.
        if (!reads && from_constants(lookup_id(it->first.get()), walk))
            continue;
        // Standing, the trace is the call whatever the write: a buffer filled
        // from the first batch keeps that batch, as eager does.  Without that, a
        // write that only overwrites (``copy_``, ``fill_``, a setitem) lands the
        // same values however often it runs, and only one that reads the tensor
        // has to run eagerly; one that runs only on the first call is compiled
        // into every call (LCD-876).
        if (!stands && !reads)
            continue;
        mark_unsupported("a tensor the call made is still held after the call and was written "
                         "in place, so a compiled call would start it over every time");
        // ``fill_`` is the only write through repeated elements, and running it
        // again overwrites them: nothing to put back.
        if (!repeats_elements(*it->first))
            kept.emplace_back(it->first, it->second, snapshot(it->first));
        // The eager call after it adds to the tensor put back, and must not
        // chain onto the trace-time call's graph.  A tensor the call made sat in
        // no graph before the write.
        if (!stands) {
            leave_graph(*it->first);
            for (const auto& member : members)
                leave_graph(*member);
        }
    }
    return kept;
}

namespace {
// Ops that answer from their inputs alone, by the same rule every time: what
// they compute from constants is a constant.  Anything not listed counts as
// reading something else (a draw, the host, state of its own).
bool pure_op(std::string_view name, bool no_inputs) {
    static const std::unordered_set<std::string_view> made_from_nothing{
        "zeros", "ones", "full", "empty", "eye", "arange", "linspace", "logspace"};
    static const std::unordered_set<std::string_view> pure{
        "reshape",       "view",        "squeeze",      "unsqueeze",    "flatten",
        "permute",       "transpose",   "broadcast_to", "expand",       "contiguous",
        "split_at",      "split",       "narrow",       "slice",        "select",
        "diagonal",      "triu",        "tril",         "concatenate",  "cat",
        "stack",         "flip",        "detach",       "to_device",    "astype",
        "clone",         "add",         "sub",          "mul",          "div",
        "neg",           "abs",         "pow",          "sqrt",         "exp",
        "log",           "maximum",     "minimum",      "clip",         "floor",
        "ceil",          "round",       "equal",        "not_equal",    "greater",
        "greater_equal", "less",        "less_equal",   "logical_and",  "logical_or",
        "logical_not",   "bitwise_and", "bitwise_or",   "bitwise_not",  "where",
        "masked_fill",   "gather",      "scatter_set",  "index_select", "nonzero",
        "matmul",        "sum"};
    return no_inputs ? made_from_nothing.count(name) != 0 : pure.count(name) != 0;
}
}  // namespace

void Tracer::index_producers(ConstantWalk& walk) const {
    if (!walk.producer.empty())
        return;
    for (const auto& op : graph_.ops)
        for (const auto& out : op.outputs)
            walk.producer.emplace(out.id, &op);
}

bool Tracer::keeps_operand(const OpNode& node, std::size_t i, ConstantWalk& walk) const {
    static const std::unordered_set<std::string_view> views{"reshape", "view",      "contiguous",
                                                            "squeeze", "unsqueeze", "flatten"};
    // Each takes (base, positions, values): the base passes into the result
    // where the positions leave it, and those are the same every call only
    // when the positions are constants.
    static const std::unordered_set<std::string_view> overwrites{"scatter_set", "masked_fill"};
    if (views.count(node.name) != 0)
        return true;
    return i == 0 && overwrites.count(node.name) != 0 && node.inputs.size() > 1 &&
           from_constants(node.inputs[1], walk);
}

bool Tracer::reads_its_own_value(const TensorImplPtr& target, ConstantWalk& walk) const {
    std::unordered_set<TensorId> prior;
    const auto add = [&](const TensorImpl* impl) {
        if (const auto p = prior_ids_.find(impl); p != prior_ids_.end())
            prior.insert(p->second.begin(), p->second.end());
    };
    add(target.get());
    for (const auto& member : target->live_views())
        add(member.get());
    index_producers(walk);
    // (id, reached only through kept operands): an earlier value reached that
    // way is the one an overwrite keeps, and what it read is read; reached any
    // other way, it is read.
    std::vector<std::pair<TensorId, bool>> stack{{lookup_id(target.get()), true}};
    std::set<std::pair<TensorId, bool>> visited;
    while (!stack.empty()) {
        const auto [id, kept] = stack.back();
        stack.pop_back();
        if (id < 0 || !visited.insert({id, kept}).second)
            continue;
        if (prior.count(id) != 0 && !kept)
            return true;
        const auto p = walk.producer.find(id);
        if (p == walk.producer.end())
            continue;
        const OpNode& node = *p->second;
        for (std::size_t i = 0; i < node.inputs.size(); ++i)
            stack.emplace_back(node.inputs[i], kept && keeps_operand(node, i, walk));
    }
    return false;
}

void Tracer::apply_buffer_updates() const {
    for (const auto& [buffer, value] : buffer_updates_)
        buffer->copy_from(*value);
}

bool Tracer::is_constant(const TensorImpl* impl) const {
    const TensorId id = lookup_id(impl);
    if (id < 0)
        return false;
    ConstantWalk walk;
    return from_constants(id, walk);
}

bool Tracer::from_constants(TensorId root, ConstantWalk& walk) const {
    index_producers(walk);
    // Depth first, without recursion: a chain of a few hundred writes is a
    // chain of a few hundred ops.
    std::vector<TensorId> stack{root};
    while (!stack.empty()) {
        const TensorId id = stack.back();
        if (id >= 0 && walk.known.count(id) == 0 && !decide(id, walk, stack))
            continue;
        stack.pop_back();
    }
    return root < 0 || walk.known[root];
}

bool Tracer::decide(TensorId id, ConstantWalk& walk, std::vector<TensorId>& stack) const {
    const auto p = walk.producer.find(id);
    if (p == walk.producer.end()) {
        walk.known[id] = constant_feed(id);
        return true;
    }
    const OpNode& node = *p->second;
    if (!pure_op(node.name, node.inputs.empty())) {
        for (const auto& out : node.outputs)
            walk.known[out.id] = false;
        return true;
    }
    bool constant = true;
    const std::size_t pushed = stack.size();
    for (const TensorId in : node.inputs) {
        const auto k = walk.known.find(in);
        if (in < 0)
            continue;
        if (k == walk.known.end())
            stack.push_back(in);
        else
            constant = constant && k->second;
    }
    if (stack.size() != pushed)
        return false;
    for (const auto& out : node.outputs)
        walk.known[out.id] = constant;
    return true;
}

bool Tracer::constant_feed(TensorId id) const {
    if (constant_feeds_.count(id) != 0)
        return true;
    // A feed of a tensor the call made, unless other tensors share its bytes.
    const auto feed = external_feeds_.find(id);
    return feed != external_feeds_.end() && made_here(feed->second.get()) &&
           feed->second->shared_buffer() == nullptr;
}

bool Tracer::from_outside(const TensorImpl* impl) const {
    auto* key = const_cast<TensorImpl*>(impl);
    return made_.count(key) == 0 && produced_.count(key) == 0;
}

bool Tracer::lands_outside(const TensorImplPtr& target) const {
    if (from_outside(target.get()))
        return true;
    const auto members = target->live_views();
    return std::any_of(members.begin(), members.end(),
                       [&](const TensorImplPtr& member) { return from_outside(member.get()); });
}

bool Tracer::carried(const TensorImplPtr& target) const {
    const auto family = target->family_id();
    return family == 0 || whole_buffer(*target) ||
           std::none_of(outside_originals_.begin(), outside_originals_.end(), [&](const auto& kv) {
               return kv.first->family_id() == family && whole_buffer(*kv.first);
           });
}

TensorId Tracer::written_id(const TensorImplPtr& value) {
    // A constant the trace made is about to lose its buffer to the target
    // (``take_storage_from``); the feed keeps a copy of what it held.
    if (made_here(value.get()) && !sees(value.get())) {
        const TensorImplPtr copy = snapshot(value);
        const TensorId fresh = graph_.next_id++;
        impl_to_id_[value.get()] = fresh;
        external_feeds_[fresh] = copy;
        note_constant_feed(fresh, *value);
        live_refs_.push_back(copy);
        return fresh;
    }
    return feed_id(value);
}

TensorId Tracer::feed_id(const TensorImplPtr& impl) {
    const auto it = impl_to_id_.find(impl.get());
    if (it != impl_to_id_.end())
        return it->second;
    refuse_computed_feed(*impl);
    const TensorId fresh = graph_.next_id++;
    impl_to_id_[impl.get()] = fresh;
    external_feeds_[fresh] = impl;
    live_refs_.push_back(impl);
    return fresh;
}

void Tracer::refuse_computed_feed(const TensorImpl& feed) {
    // A leaf's node (``AccumulateGrad``) is made when something first asks for it.
    // The tracer is per thread, so a value another thread computes during the
    // trace is refused too: compiled, it would be this call's value on every call.
    if (feed.is_leaf() || !feed.grad_fn() || feed.grad_fn()->sequence_nr() < first_node_)
        return;
    mark_unsupported("a value computed during the trace reached an op under a tensor the trace "
                     "was never told of, so each call would reuse the first call's value");
}

void Tracer::on_stand_in(const TensorImplPtr& stand_in, const TensorImplPtr& of) {
    if (!stand_in || !of)
        return;
    refuse_stale(of.get());
    live_refs_.push_back(stand_in);
    impl_to_id_[stand_in.get()] = feed_id(of);
}

void Tracer::write_made(const TensorImplPtr& target, const TensorImplPtr& value) {
    // Nothing outside the trace shares ``target``, so the write is not one a
    // compiled call carries back: it is the next value of a constant when
    // ``value`` was also made here.  Unless something outside turns out to hold
    // it after all (``refuse_kept_writes``), which is why the value goes on file.
    const TensorImplPtr before = keep_original(made_originals_, target);
    const auto it = impl_to_id_.find(target.get());
    if (it == impl_to_id_.end() && made_here(value.get()) && !sees(value.get()))
        return;
    if (it != impl_to_id_.end()) {
        // Already read as a feed, by ops recorded before this write: they
        // keep the value from before it, which the buffer is about to lose.
        const auto feed = external_feeds_.find(it->second);
        if (feed != external_feeds_.end() && feed->second.get() == target.get()) {
            feed->second = before ? before : snapshot(target);
            note_constant_feed(it->second, *target);
        }
    }
    const TensorId now = written_id(value);
    if (it != impl_to_id_.end())
        note_prior(target, it->second, now, true);
    impl_to_id_[target.get()] = now;
}

void Tracer::on_stale_view(const TensorImplPtr& view) {
    if (!view)
        return;
    const auto seen = impl_to_id_.find(view.get());
    if (seen == impl_to_id_.end()) {
        // A constant no op has read yet holds, whenever it is read, what a
        // constant write left in it.
        if (constant_write_ && made_here(view.get()))
            return;
    } else if (made_here(view.get())) {
        // Read as a feed by ops recorded before this write: they keep the
        // value from before it, which the buffer is about to lose (see
        // ``write_made``, which does the same for the target).
        const auto feed = external_feeds_.find(seen->second);
        if (feed != external_feeds_.end() && feed->second.get() == view.get()) {
            feed->second = snapshot(view);
            note_constant_feed(seen->second, *view);
        }
    }
    stale_.emplace(view.get(), view);
}

void Tracer::note_constant_feed(TensorId id, const TensorImpl& of) {
    // A copy of a shared buffer stands for bytes another tensor may write.
    if (of.shared_buffer() == nullptr)
        constant_feeds_.insert(id);
}

bool Tracer::reads_a_shared_write(const TensorImpl* impl) const {
    if (shared_writers_.empty())
        return false;
    const SharedStorage* buffer = impl->shared_buffer();
    if (buffer == nullptr)
        return false;
    const auto writer = shared_writers_.find(buffer->cpu_ptr);
    if (writer == shared_writers_.end() || writer->second.get() == impl)
        return false;
    // Within one family the stale views say which members a write left behind.
    const auto family = impl->family_id();
    return family == 0 || family != writer->second->family_id();
}

void Tracer::refuse_stale(const TensorImpl* impl) {
    if (impl == nullptr ||
        (stale_.count(const_cast<TensorImpl*>(impl)) == 0 && !reads_a_shared_write(impl)))
        return;
    mark_unsupported("a view of a tensor was read after an in-place write to that tensor, "
                     "which the recording gives the value from before the write");
}

TensorId Tracer::read_id(const TensorImpl* impl) {
    refuse_stale(impl);
    return lookup_id(impl);
}

void Tracer::restore_outside_writes() {
    // Before the copies below: ``copy_from`` is a write like any other, and
    // clears the tensor it writes from the stale views.  A tensor written
    // twice, the second time through an overlapping view, carries back the
    // value of the first write only.
    for (const auto& [target, before] : outside_originals_) {
        if (carried(target))
            refuse_stale(target.get());
    }
    // What the trace wrote stays readable under the id it wrote, for the caller
    // that takes a traced value's contents (retained_values): the tensor is
    // about to stop holding it.
    if (keep_written_values_) {
        for (const auto& [target, before] : outside_originals_) {
            if (buffer_values_.count(target.get()) == 0)
                buffer_values_[target.get()] = snapshot(target);
        }
    }
    // Latest first: each tensor has one entry, its value from before the
    // trace's first write to it, and a tensor that overlaps an earlier one
    // kept that one's write in its copy.  The earlier copy goes last, so
    // every byte ends as it was before the trace.
    for (auto it = outside_originals_.rbegin(); it != outside_originals_.rend(); ++it)
        it->first->copy_from(*it->second);
    put_back_graph_positions();
}

std::vector<std::pair<TensorId, TensorImplPtr>> Tracer::outside_writes() const {
    std::vector<std::pair<TensorId, TensorImplPtr>> out;
    out.reserve(outside_originals_.size());
    for (const auto& [target, before] : outside_originals_) {
        if (carried(target))
            out.emplace_back(lookup_id(target.get()), target);
    }
    return out;
}

std::unordered_map<TensorId, TensorImplPtr> Tracer::retained_values() const {
    std::unordered_map<TensorId, TensorImplPtr> out;
    out.reserve(live_refs_.size());
    for (const auto& impl : live_refs_) {
        // A stale view holds the values after the write under the id of
        // the ones before it.
        if (!impl || stale_.count(impl.get()) != 0)
            continue;
        const auto it = impl_to_id_.find(impl.get());
        if (it != impl_to_id_.end())
            out.emplace(it->second, buffer_value(impl));
    }
    // A tensor's kept value beats whatever else goes by its id: the op that
    // computed it may have handed its buffer over (``take_storage_from``).
    for (const auto& [impl, value] : buffer_values_) {
        const auto it = impl_to_id_.find(impl);
        if (it != impl_to_id_.end() && stale_.count(impl) == 0)
            out.insert_or_assign(it->second, value);
    }
    return out;
}

bool Tracer::sees(const TensorImpl* impl) const {
    auto* key = const_cast<TensorImpl*>(impl);
    return impl != nullptr && (impl_to_id_.count(key) != 0 || watched_.count(key) != 0);
}

void Tracer::on_host_read(const TensorImpl* impl) {
    if (sees(impl))
        mark_unsupported("a traced value was read on the host (item / tolist / numpy / "
                         "bytes / pickle / dlpack), so each call would reuse the first "
                         "call's value");
    else if (refuse_outside_reads_ && impl != nullptr && !made_here(impl))
        mark_unsupported("a tensor from outside the call was read on the host (item / tolist / "
                         "numpy / bytes / pickle / dlpack), so each call would reuse the "
                         "first call's value");
    else
        refuse_stale(impl);
}

void Tracer::watch(const TensorImplPtr& impl) {
    if (impl)
        watched_.emplace(impl.get(), impl);
}

void Tracer::on_derived(std::string_view name,
                        const std::vector<TensorImplPtr>& srcs,
                        const TensorImplPtr& out) {
    if (!out)
        return;
    for (const auto& src : srcs)
        refuse_stale(src.get());
    const auto is = [&](const TensorImplPtr& src) { return src.get() == out.get(); };
    const auto seen = [&](const TensorImplPtr& src) { return sees(src.get()); };
    if (std::any_of(srcs.begin(), srcs.end(), is))
        return;
    if (std::none_of(srcs.begin(), srcs.end(), seen)) {
        // Made from the trace's own constants only, ``out`` is one too.
        const auto made = [&](const TensorImplPtr& src) { return !src || made_here(src.get()); };
        const auto foldable = [&](const TensorImplPtr& src) {
            return !src || foldable_.count(src.get()) != 0;
        };
        if (std::all_of(srcs.begin(), srcs.end(), made)) {
            made_.emplace(out.get(), out);
            if (fold_constants_ && std::all_of(srcs.begin(), srcs.end(), foldable))
                foldable_.insert(out.get());
            return;
        }
        // From a tensor the call did not make: ``out`` reads that tensor's bytes,
        // and stays live with them, only when it is a label or view of them.  A
        // copy holds the values of the moment, and a feed of it would hand every
        // later call those.  Recorded, the source is a feed and the op is the
        // call's, as when the trace already knew the source.
        const auto labels = [&](const TensorImplPtr& src) {
            return src && shares_bytes(*src, *out);
        };
        if (std::any_of(srcs.begin(), srcs.end(), labels))
            return;
    }
    on_op_enter(name, out->device(), out->dtype(), out->shape());
    on_op_io(srcs, out);
}

void Tracer::mark_unsupported(std::string why) {
    if (graph_.unsupported.empty())
        graph_.unsupported = std::move(why);
}

void Tracer::on_op_attr(std::string_view key, AttributeValue value) {
    OpNode* node = held_ ? &*held_ : (graph_.ops.empty() ? nullptr : &graph_.ops.back());
    if (node != nullptr)
        node->attrs.insert_or_assign(std::string(key), std::move(value));
}

void Tracer::on_factory(const TensorImplPtr& out) {
    if (!out)
        return;
    note_made(out);
    // The open op is the factory: every caller opened it for ``out``.
    if (foldable_.count(out.get()) != 0 && !held_ && !graph_.ops.empty()) {
        held_ = std::move(graph_.ops.back());
        graph_.ops.pop_back();
    }
}

void Tracer::on_literal(const TensorImplPtr& out) {
    if (out)
        note_made(out);
}

void Tracer::note_made(const TensorImplPtr& out) {
    made_.emplace(out.get(), out);
    // On Metal the compiled graph makes a factory itself, as a constant
    // MPSGraph can fold into its neighbours; held out, it would be one more
    // feed bound on every run.  Anywhere else the graph can only receive it.
    if (fold_constants_ && out->device() != Device::GPU)
        foldable_.insert(out.get());
}

bool Tracer::fold(const std::vector<TensorImplPtr>& inputs, const TensorImplPtr& output) {
    const auto all_made = std::all_of(inputs.begin(), inputs.end(), [&](const TensorImplPtr& in) {
        return !in || (foldable_.count(in.get()) != 0 && !sees(in.get()));
    });
    if (held_) {
        if (all_made) {
            keep(output);
            return true;
        }
        // A later call names an input from outside or already in the graph
        // (an op re-recording its full input list): the op is recorded after all.
        graph_.ops.push_back(std::move(*held_));
        held_.reset();
        if (output) {
            made_.erase(output.get());
            foldable_.erase(output.get());
        }
        return false;
    }
    // Only an op's first wiring decides; a node that already has inputs was
    // recorded.  An op with no tensor input at all is a random draw kept as
    // an op, or the extra output of a recorded one — never a constant here
    // (factories say so through ``on_factory``).
    if (!all_made || graph_.ops.empty() || !graph_.ops.back().inputs.empty())
        return false;
    const auto tensor = [](const TensorImplPtr& in) { return in != nullptr; };
    const auto aliases = [&](const TensorImplPtr& in) { return in.get() == output.get(); };
    if (std::none_of(inputs.begin(), inputs.end(), tensor) ||
        (output && std::any_of(inputs.begin(), inputs.end(), aliases)))
        return false;
    held_ = std::move(graph_.ops.back());
    graph_.ops.pop_back();
    keep(output);
    return true;
}

void Tracer::keep(const TensorImplPtr& output) {
    if (output) {
        made_.emplace(output.get(), output);
        foldable_.insert(output.get());
    }
}

TensorId Tracer::ensure_id(const TensorImplPtr& impl) {
    if (!impl)
        return TraceId::external_feed();
    refuse_stale(impl.get());
    return feed_id(impl);
}

TensorId Tracer::register_cotangent(const TensorImplPtr& impl) {
    if (!impl)
        return TraceId::external_feed();
    if (held_) {
        if (!held_->outputs.empty() && held_->outputs[0].id + 1 == graph_.next_id)
            graph_.next_id = held_->outputs[0].id;
        held_.reset();
    }
    const TensorId fresh = graph_.next_id++;
    claim(impl, fresh);
    live_refs_.push_back(impl);
    watched_.emplace(impl.get(), impl);
    return fresh;
}

void Tracer::discard_ops_from(std::size_t idx) {
    if (held_) {
        if (!held_->outputs.empty() && held_->outputs[0].id + 1 == graph_.next_id)
            graph_.next_id = held_->outputs[0].id;
        held_.reset();
    }
    if (idx < graph_.ops.size()) {
        graph_.ops.erase(graph_.ops.begin() + static_cast<std::ptrdiff_t>(idx), graph_.ops.end());
    }
}

void Tracer::commit_vjp_region(std::size_t fwd_begin,
                               std::size_t fwd_end,
                               std::vector<TensorId> input_ids,
                               std::vector<TensorMeta> output_metas,
                               std::vector<TensorId> cotangent_ids,
                               std::vector<TensorId> input_grad_ids,
                               std::size_t bwd_begin) {
    if (held_) {
        if (!held_->outputs.empty() && held_->outputs[0].id + 1 == graph_.next_id)
            graph_.next_id = held_->outputs[0].id;
        held_.reset();
    }
    VjpRegion region;
    region.fwd_begin = fwd_begin;
    region.fwd_end = fwd_end;
    region.input_ids = std::move(input_ids);
    region.output_metas = std::move(output_metas);
    region.cotangent_ids = std::move(cotangent_ids);
    region.input_grad_ids = std::move(input_grad_ids);
    if (bwd_begin < graph_.ops.size()) {
        region.bwd_ops.assign(
            std::make_move_iterator(graph_.ops.begin() + static_cast<std::ptrdiff_t>(bwd_begin)),
            std::make_move_iterator(graph_.ops.end()));
        graph_.ops.erase(graph_.ops.begin() + static_cast<std::ptrdiff_t>(bwd_begin),
                         graph_.ops.end());
    }
    graph_.vjp_regions.push_back(std::move(region));
}

}  // namespace lucid::compile
