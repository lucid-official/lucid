// lucid/_C/core/Scope.h
//
// Combined error-context and profiling RAII scope for op entry points.
//
// :class:`OpScopeFull` bundles an :class:`ErrorContextGuard` — which
// pushes the op name onto the thread-local error call-stack so any
// raised :class:`LucidError` carries the surrounding op as context —
// and an :class:`OpScope` — which starts wall-clock timing and memory
// tracking — into a single object.  Op implementations that want both
// behaviours can declare one :class:`OpScopeFull` at function entry
// instead of managing two separate guards.
//
// Destruction order is significant: members are listed with ``ctx_``
// before ``op_`` so the context guard is constructed first and
// destroyed last.  The op name therefore remains on the error stack
// for the entire duration of the OpScope event — including the
// :func:`Profiler::record` call inside :class:`OpScope`'s destructor —
// which means any allocator or profiler-side exception still inherits
// the right error context.
//
// Notes
// -----
// Typical usage at op entry::
//
//     OpScopeFull scope("conv2d", x->device(), x->dtype(), out_shape);
//     scope.set_flops(2LL * N * C * H * W * K);
//
// See Also
// --------
// :class:`OpScope` — profiler-only RAII.
// :class:`ErrorContextGuard` — error-context-only RAII.

#pragma once

#include <string>
#include <string_view>
#include <utility>

#include "../api.h"
#include "../compile/Tracer.h"
#include "Device.h"
#include "Dtype.h"
#include "ErrorBuilder.h"
#include "Profiler.h"
#include "Shape.h"

namespace lucid {

// Composite RAII guard activating both error-context annotation and
// profiler timing for the duration of an op's body.
//
// Holds an :class:`ErrorContextGuard` (constructed first, destroyed
// last) and an :class:`OpScope` (constructed second, destroyed first)
// as members.  Construction pushes the op name onto the error stack
// and starts the profiler timer; destruction first stops the profiler
// (recording the event) and then pops the error frame.  This ordering
// ensures the error frame is still active if the profiler's
// :func:`record` call itself throws.
//
// Attributes
// ----------
// ctx_ : ErrorContextGuard
//     Pushes ``name`` onto the thread-local error call-stack at
//     construction; pops it on destruction.  Declared first so it
//     outlives ``op_``.
// op_ : OpScope
//     Wall-clock and memory-tracking RAII recorder.  Declared second
//     so it is destroyed first, while ``ctx_`` is still on the stack.
//
// Notes
// -----
// Non-copyable and non-movable — both member guards hold thread-local
// state whose correct LIFO nesting depends on stack-discipline
// construction and destruction.  Copying or moving the composite
// would either duplicate the error frame or move it out from under
// the inner scope.
//
// Examples
// --------
// Declare once at op entry::
//
//     auto matmul(TensorPtr a, TensorPtr b) {
//         OpScopeFull scope("matmul", a->device(), a->dtype(),
//                           {a->shape()[0], b->shape()[1]});
//         scope.set_flops(2LL * M * N * K);
//         // ... kernel body — any error here carries "matmul" in its
//         // context; the OpEvent records the full body's time.
//     }
//
// See Also
// --------
// :class:`OpScope` — timing-only RAII (no error context).
// :class:`ErrorContextGuard` — error-context-only RAII (no timing).
class LUCID_API OpScopeFull {
public:
    // Pushes ``name`` onto the error-context stack and starts an
    // :class:`OpScope` for ``(name, device, dtype, shape)``.
    //
    // Parameters
    // ----------
    // name : std::string_view
    //     Op name; used both as the error-context frame label and the
    //     :class:`OpEvent::name` field.
    // device : Device
    //     Device on which the op is executing.
    // dtype : Dtype
    //     Output element dtype.
    // shape : Shape
    //     Output shape; moved into the underlying :class:`OpScope`.
    //
    // Notes
    // -----
    // Both guards are torn down when the composite object goes out of
    // scope.  The profiler-side teardown happens first; the
    // error-context teardown second.
    OpScopeFull(std::string_view name, Device device, Dtype dtype, Shape shape)
        : ctx_(std::string(name)), op_(name, device, dtype, shape) {
        // 3.5 Phase 1.1: lucid.compile() tracer hook.  Outside any _tracing()
        // scope this is a single TLS load + null check (zero allocations,
        // no shape copy beyond the one OpScope already needs).  Inside one,
        // the op's entry feeds the active TraceGraph, and unwinding on an
        // exception before on_op_io pops the pending node.  See compile/Tracer.h.
        if (auto* t = lucid::compile::current_tracer()) {
            lucid::compile::detail::on_scope_enter(&op_, t, name, device, dtype, std::move(shape));
        }
    }

    OpScopeFull(const OpScopeFull&) = delete;
    OpScopeFull& operator=(const OpScopeFull&) = delete;

    // Forwards the FLOPs estimate to the underlying :class:`OpScope`.
    //
    // Parameters
    // ----------
    // f : int64_t
    //     Estimated floating-point operations performed by the op
    //     body.  Stored on the pending :class:`OpEvent` for later
    //     aggregation.
    //
    // See Also
    // --------
    // :func:`OpScope::set_flops` — receiver of the forwarded value.
    void set_flops(std::int64_t f) { op_.set_flops(f); }

    // Attach a single attribute to the trace's most recently recorded
    // :class:`OpNode` when a Tracer is installed; a no-op otherwise.
    //
    // Used by op forwards to thread emitter context (permutation,
    // axis, stride, padding, eps, …) that the MPSGraph builder would
    // be unable to recover from input + output shapes alone.
    //
    // Parameters
    // ----------
    // key : std::string_view
    //     Attribute name.  Convention: snake_case, matches the
    //     reference-framework keyword argument when one exists.
    // value : compile::AttributeValue
    //     Payload (variant over int64/vector<int64>/double/bool/string).
    //     Moved into the attribute map.
    //
    // Notes
    // -----
    // Outside any ``_tracing()`` scope this is a single TLS load +
    // null check; no allocations.  Cheap enough to put unconditionally
    // at every op forward that has shape-invisible parameters.
    void set_attr(std::string_view key, compile::AttributeValue value) {
        if (auto* t = compile::current_tracer())
            t->on_op_attr(key, std::move(value));
    }

private:
    // Declared before ``op_`` so it is constructed first and destroyed
    // last, keeping the op name on the error stack throughout the
    // profiler's :func:`record` call.
    ErrorContextGuard ctx_;
    OpScope op_;
};

}  // namespace lucid
