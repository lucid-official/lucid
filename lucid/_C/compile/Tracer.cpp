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

#include "../backend/Dispatcher.h"
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
}  // namespace

Tracer* current_tracer() {
    return g_current;
}

void set_current_tracer(Tracer* t) {
    g_current = t;
}

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
    graph_.ops.push_back(std::move(node));
}

void Tracer::on_op_io(const std::vector<TensorImplPtr>& inputs, const TensorImplPtr& output) {
    // First, since a held-out op has no node in ``graph_.ops``.
    if (!fold_constants_ || !fold(inputs, output))
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
            const auto it = impl_to_id_.find(raw);
            if (it != impl_to_id_.end()) {
                node.inputs.push_back(it->second);
            } else {
                const TensorId fresh = graph_.next_id++;
                impl_to_id_[raw] = fresh;
                external_feeds_[fresh] = inp;  // owning ref
                node.inputs.push_back(fresh);
            }
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
                impl_to_id_[output.get()] = slot0_id;
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
                impl_to_id_[output.get()] = new_id;
            }
        }
    }
}

void Tracer::on_buffer_update(const TensorImplPtr& buffer, const TensorImplPtr& value) {
    if (!buffer || !value)
        return;
    const auto it = impl_to_id_.find(value.get());
    if (it != impl_to_id_.end()) {
        impl_to_id_[buffer.get()] = it->second;
        buffer_values_[buffer.get()] = value;
    }
}

void Tracer::on_rng_feed(std::string_view name, const TensorImplPtr& output) {
    if (!output)
        return;
    if (!graph_.ops.empty() && graph_.ops.back().name == name && graph_.ops.back().inputs.empty())
        graph_.ops.pop_back();
    live_refs_.push_back(output);
    const TensorId fresh = graph_.next_id++;
    impl_to_id_[output.get()] = fresh;
    external_feeds_[fresh] = output;
}

void Tracer::on_inplace_write(const TensorImplPtr& target, const TensorImplPtr& value) {
    if (!target || !value || target.get() == value.get())
        return;
    if (made_here(target.get())) {
        write_made(target, value);
        return;
    }
    // From outside the trace — never seen, or seen only as a feed — and not
    // written yet in this trace: keep its current value to put back later.
    const auto seen = impl_to_id_.find(target.get());
    const bool outside = seen == impl_to_id_.end() || external_feeds_.count(seen->second) != 0;
    const bool first = std::none_of(outside_originals_.begin(), outside_originals_.end(),
                                    [&](const auto& kv) { return kv.first.get() == target.get(); });
    if (outside && first)
        outside_originals_.emplace_back(target, snapshot(target));
    feed_id(target);
    impl_to_id_[target.get()] = written_id(value);
}

TensorId Tracer::written_id(const TensorImplPtr& value) {
    // A constant the trace made is about to lose its buffer to the target
    // (``take_storage_from``); the feed keeps a copy of what it held.
    if (made_here(value.get()) && !sees(value.get())) {
        const TensorImplPtr copy = snapshot(value);
        const TensorId fresh = graph_.next_id++;
        impl_to_id_[value.get()] = fresh;
        external_feeds_[fresh] = copy;
        live_refs_.push_back(copy);
        return fresh;
    }
    return feed_id(value);
}

TensorId Tracer::feed_id(const TensorImplPtr& impl) {
    const auto it = impl_to_id_.find(impl.get());
    if (it != impl_to_id_.end())
        return it->second;
    const TensorId fresh = graph_.next_id++;
    impl_to_id_[impl.get()] = fresh;
    external_feeds_[fresh] = impl;
    live_refs_.push_back(impl);
    return fresh;
}

void Tracer::write_made(const TensorImplPtr& target, const TensorImplPtr& value) {
    // Nothing outside the trace shares ``target``, so the write is not one a
    // compiled call carries back: it is the next value of a constant when
    // ``value`` was also made here.
    const auto it = impl_to_id_.find(target.get());
    if (it == impl_to_id_.end() && made_here(value.get()) && !sees(value.get()))
        return;
    if (it != impl_to_id_.end()) {
        // Already read as a feed, by ops recorded before this write: they
        // keep the value from before it, which the buffer is about to lose.
        const auto feed = external_feeds_.find(it->second);
        if (feed != external_feeds_.end() && feed->second.get() == target.get())
            feed->second = snapshot(target);
    }
    impl_to_id_[target.get()] = written_id(value);
}

void Tracer::restore_outside_writes() {
    // Latest first is not needed: each tensor has one entry, its value from
    // before the trace's first write to it.
    for (const auto& [target, before] : outside_originals_)
        target->copy_from(*before);
}

std::vector<std::pair<TensorId, TensorImplPtr>> Tracer::outside_writes() const {
    std::vector<std::pair<TensorId, TensorImplPtr>> out;
    out.reserve(outside_originals_.size());
    for (const auto& [target, before] : outside_originals_)
        out.emplace_back(lookup_id(target.get()), target);
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
    const auto is = [&](const TensorImplPtr& src) { return src.get() == out.get(); };
    const auto seen = [&](const TensorImplPtr& src) { return sees(src.get()); };
    if (std::any_of(srcs.begin(), srcs.end(), is))
        return;
    if (std::none_of(srcs.begin(), srcs.end(), seen)) {
        // Made from the trace's own constants only, ``out`` is one too.
        const auto made = [&](const TensorImplPtr& src) { return !src || made_here(src.get()); };
        if (fold_constants_ && std::all_of(srcs.begin(), srcs.end(), made))
            made_.emplace(out.get(), out);
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
    // On Metal the compiled graph makes the factory itself, as a constant
    // MPSGraph can fold into its neighbours; held out, it would be one more
    // feed bound on every run.  Anywhere else the graph can only receive it.
    if (!fold_constants_ || !out || out->device() == Device::GPU)
        return;
    if (!held_ && !graph_.ops.empty()) {
        held_ = std::move(graph_.ops.back());
        graph_.ops.pop_back();
    }
    made_.emplace(out.get(), out);
}

bool Tracer::fold(const std::vector<TensorImplPtr>& inputs, const TensorImplPtr& output) {
    const auto all_made = std::all_of(inputs.begin(), inputs.end(), [&](const TensorImplPtr& in) {
        return !in || (made_here(in.get()) && !sees(in.get()));
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
        if (output)
            made_.erase(output.get());
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
    if (output)
        made_.emplace(output.get(), output);
}

TensorId Tracer::ensure_id(const TensorImplPtr& impl) {
    if (!impl)
        return TraceId::external_feed();
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
    impl_to_id_[impl.get()] = fresh;
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
