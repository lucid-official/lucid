// lucid/_C/core/Profiler.h
//
// Lightweight op-level profiler and RAII scope recorder.
//
// :class:`Profiler` is an event sink that collects :class:`OpEvent`
// records while active.  It is intentionally not a global singleton —
// instead, a per-thread pointer (:func:`current_profiler`) designates
// the active profiler for the calling thread.  This lets Python test
// code and benchmark harnesses swap profilers without disturbing other
// threads, and it lets ops dispatched from DataLoader workers either
// share the main profiler (when registered into the worker's TLS) or
// silently skip recording.
//
// :class:`OpScope` is the RAII mechanism that ops use to register
// themselves with the active profiler.  Constructing an OpScope
// captures a wall-clock start time and a memory baseline; the
// destructor computes the elapsed time and memory delta, then calls
// :func:`Profiler::record`.  When :func:`current_profiler` is
// ``nullptr`` or the profiler is not active, the OpScope's ``sink_``
// is set to ``nullptr`` at construction and the destructor becomes a
// single null-check branch — near-zero overhead on the hot path.
//
// Notes
// -----
// Thread safety:
//
//   - :func:`Profiler::record`, :func:`Profiler::start`,
//     :func:`Profiler::stop`, :func:`Profiler::clear`, and
//     :func:`Profiler::events` are protected by an internal mutex, so
//     events from multiple worker threads can be funnelled into a
//     single :class:`Profiler` safely.
//   - The :func:`current_profiler` pointer itself is ``thread_local``,
//     so each thread independently chooses which profiler (if any) to
//     report into.
//
// See Also
// --------
// :class:`lucid.profiler.Profiler` — Python-side wrapper.
// :func:`lucid.profiler.profile` — Python context-manager convenience.

#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "../api.h"
#include "Device.h"
#include "Dtype.h"
#include "Shape.h"

namespace lucid {

// Record of a single op execution captured by :class:`OpScope`.
//
// Each event is a snapshot of one dispatched op: its name, the device
// and dtype it ran on, the output shape, the wall-clock duration, the
// net memory delta, and an optional FLOP estimate.  Profilers store
// vectors of these records and expose them to the Python layer for
// trace export (Chrome / Perfetto) and aggregated summaries.
//
// Attributes
// ----------
// name : std::string
//     Op name as recorded at OpScope construction (typically the
//     schema name, e.g. ``"matmul"``, ``"conv2d"``).
// device : Device
//     Device on which the op executed.
// dtype : Dtype
//     Element dtype of the output tensor at OpScope construction time.
// shape : Shape
//     Output tensor shape at OpScope construction time.
// time_ns : int64_t
//     Wall-clock duration from OpScope construction to destruction,
//     in nanoseconds.
// memory_delta_bytes : int64_t
//     Change in :func:`MemoryTracker::current_bytes` for ``device``
//     across the scope.  May be negative when the op frees more memory
//     than it allocates (e.g. in-place ops releasing temporaries).
// flops : int64_t
//     Estimated floating-point operation count, populated by
//     :func:`OpScope::set_flops`.  Defaults to ``0`` when the op
//     does not report a count.
// backward : bool
//     Whether the event is backward work — a whole backward pass
//     (``"backward"``) or one node of it, named after the node
//     (``"MulBackward"``) — rather than a dispatched op.
//
// Notes
// -----
// All fields are owned by value; the struct is freely copyable and
// suitable for return from :func:`Profiler::events`.
struct LUCID_API OpEvent {
    std::string name;
    Device device;
    Dtype dtype;
    Shape shape;
    std::int64_t time_ns = 0;
    std::int64_t memory_delta_bytes = 0;
    std::int64_t flops = 0;
    bool backward = false;
};

// Thread-safe event collector for op profiling.
//
// Holds a vector of :class:`OpEvent` records and a single mutex
// protecting both the active flag and the event list.  Multiple
// :class:`OpScope` instances on different threads may concurrently
// call :func:`record` — the mutex serialises insertions.
//
// Attributes
// ----------
// active_ : bool
//     Whether the profiler is currently recording.  Toggled by
//     :func:`start` and :func:`stop`.
// events_ : std::vector<OpEvent>
//     Storage for recorded events; appended under ``mu_``.
// mu_ : mutable std::mutex
//     Synchronises every public method that touches ``active_`` or
//     ``events_``.
//
// Notes
// -----
// Typical lifecycle::
//
//     Profiler prof;
//     set_current_profiler(&prof);
//     prof.start();
//     // ... execute ops ...
//     prof.stop();
//     auto events = prof.events();
//     set_current_profiler(nullptr);
//
// The Python wrapper :class:`lucid.profiler.Profiler` runs this exact
// sequence inside ``__enter__`` / ``__exit__``.
//
// See Also
// --------
// :class:`OpScope` — RAII recorder that feeds events into this sink.
// :func:`set_current_profiler` — thread-local registration.
class LUCID_API Profiler {
public:
    // Enables event collection.
    //
    // Notes
    // -----
    // After this call, every :class:`OpScope` destructor whose
    // ``sink_`` points at this profiler will append its event to
    // ``events_``.  Calling :func:`start` on an already-active
    // profiler is idempotent.
    void start();

    // Disables event collection without clearing recorded events.
    //
    // Notes
    // -----
    // Events already in ``events_`` are preserved — call
    // :func:`clear` to drop them.  In-flight :class:`OpScope`
    // destructors may still record their event if they raced with the
    // ``stop()`` write; :func:`record` re-checks ``active_`` under the
    // mutex to discard such latecomers.
    void stop();

    // Returns whether the profiler is currently recording.
    //
    // Returns
    // -------
    // bool
    //     ``true`` between :func:`start` and :func:`stop`, ``false``
    //     otherwise.
    //
    // Notes
    // -----
    // Read without acquiring the mutex — the result is a snapshot and
    // may be stale by the time the caller acts on it.  The internal
    // ``record()`` call re-checks under the lock to avoid races.
    bool is_active() const { return active_; }

    // Removes all stored events without changing the active state.
    //
    // Notes
    // -----
    // Resets ``events_.size()`` to zero; the active/stopped state of
    // the profiler is left untouched, so a running profiler keeps
    // collecting new events afterwards.
    void clear();

    // Returns a copy of the recorded event list.
    //
    // Returns
    // -------
    // std::vector<OpEvent>
    //     Snapshot of all events recorded so far, copied under
    //     ``mu_``.  Caller can iterate without holding the lock.
    //
    // Notes
    // -----
    // Returns by value to decouple iteration from concurrent writers;
    // the cost is one ``OpEvent`` vector copy per call.
    std::vector<OpEvent> events() const;

    // Appends ``event`` to the recorded list if the profiler is active.
    //
    // Parameters
    // ----------
    // event : OpEvent
    //     The event to store.  Moved into ``events_``.
    //
    // Notes
    // -----
    // Thread-safe.  Called by :class:`OpScope` on destruction.
    // Re-checks :func:`is_active` under ``mu_`` so events that race
    // with :func:`stop` are discarded rather than appended.
    void record(OpEvent event);

private:
    bool active_ = false;
    std::vector<OpEvent> events_;
    mutable std::mutex mu_;
};

// Returns the :class:`Profiler` currently registered for the calling
// thread.
//
// Returns
// -------
// Profiler*
//     Non-owning pointer to the active profiler, or ``nullptr`` if no
//     profiler is registered on this thread.
//
// Notes
// -----
// The pointer is stored in ``thread_local`` storage, so each thread
// sees its own value independently.  Ownership of the pointed-to
// :class:`Profiler` remains with the caller.
LUCID_API Profiler* current_profiler();

// Registers ``p`` as the active :class:`Profiler` for the calling
// thread.
//
// Parameters
// ----------
// p : Profiler*
//     Profiler to register, or ``nullptr`` to deactivate profiling on
//     this thread without destroying the existing object.
//
// Notes
// -----
// Does not transfer ownership — the caller must keep ``p`` alive for
// as long as any :class:`OpScope` could observe it as the current
// profiler.  The Python wrapper holds the binding object's
// ``shared_ptr`` for exactly this reason.
//
// See Also
// --------
// :func:`current_profiler` — reader for the same TLS slot.
LUCID_API void set_current_profiler(Profiler* p);

// RAII scope that records one :class:`OpEvent` into the thread's
// active :class:`Profiler` on destruction.
//
// Constructing the scope captures the wall-clock start time and the
// memory baseline for ``device``; the destructor computes ``time_ns``
// and ``memory_delta_bytes`` and calls :func:`Profiler::record`.  When
// no profiler is active, ``sink_`` is set to ``nullptr`` at
// construction and the destructor short-circuits — the hot path costs
// one TLS load and one null check.
//
// Attributes
// ----------
// sink_ : Profiler*
//     Non-owning pointer to the profiler captured at construction
//     time.  ``nullptr`` indicates "profiling off for this scope";
//     once set to ``nullptr`` the destructor performs no work.
// event_ : OpEvent
//     Partially-filled event accumulated through the scope's lifetime
//     and forwarded to :func:`Profiler::record` on destruction.
// start_time_ : std::chrono::steady_clock::time_point
//     Wall-clock instant captured at construction.
// start_memory_bytes_ : std::size_t
//     :func:`MemoryTracker::current_bytes` snapshot taken at
//     construction; subtracted on destruction to compute
//     ``memory_delta_bytes``.
//
// Notes
// -----
// Ownership: OpScope never owns the Profiler.  Callers of
// :func:`set_current_profiler` are responsible for keeping the
// pointed-to object alive for the entire scope of any OpScope that
// could observe it.
//
// The scope is non-copyable — copying would double-record the same
// event into the sink.
//
// Examples
// --------
// Wrap a kernel implementation::
//
//     OpScope scope("matmul", a->device(), a->dtype(), out_shape);
//     scope.set_flops(2LL * M * N * K);
//     // ... kernel work ...
//
// See Also
// --------
// :class:`OpScopeFull` — composite that also pushes an error-context
//     frame.
class LUCID_API OpScope {
public:
    // Captures the start wall-clock time and memory baseline.
    //
    // Parameters
    // ----------
    // name : std::string_view
    //     Op name to store in the resulting :class:`OpEvent`.
    // device : Device
    //     Device on which the op is executing.
    // dtype : Dtype
    //     Element dtype of the op's output.
    // shape : Shape
    //     Output shape; moved into ``event_``.
    //
    // Notes
    // -----
    // If no :class:`Profiler` is active at this moment, ``sink_`` is
    // set to ``nullptr`` immediately and the destructor will skip the
    // :func:`MemoryTracker` query — keeping un-profiled ops cheap.
    OpScope(std::string_view name, Device device, Dtype dtype, Shape shape);

    // Finalises the event with elapsed time and memory delta, then
    // forwards it to :func:`Profiler::record`.
    //
    // Notes
    // -----
    // Short-circuits when ``sink_ == nullptr`` (the no-profiler fast
    // path).  Otherwise performs a steady-clock subtraction, a
    // memory-tracker query, and one mutex-guarded vector append on the
    // sink.
    ~OpScope();

    OpScope(const OpScope&) = delete;
    OpScope& operator=(const OpScope&) = delete;

    // Records the estimated floating-point operation count for this
    // scope.
    //
    // Parameters
    // ----------
    // f : int64_t
    //     FLOPs estimate (e.g. ``2 * M * N * K`` for an ``M x N``
    //     matmul with ``K`` inner dim).
    //
    // Notes
    // -----
    // Must be called before the scope exits; otherwise the recorded
    // event reports ``flops == 0``.  Ops free to omit this if they
    // do not have a meaningful FLOPs count.
    void set_flops(std::int64_t f);

    // Mark the event as backward work (:attr:`OpEvent::backward`).
    void set_backward() noexcept { event_.backward = true; }

private:
    // Non-owning pointer to the active Profiler; ``nullptr`` if no
    // profiling is in effect for this scope.
    Profiler* sink_;
    OpEvent event_;
    std::chrono::steady_clock::time_point start_time_;
    // :func:`MemoryTracker::current_bytes` at construction time; used
    // to compute the memory delta on destruction.
    std::size_t start_memory_bytes_;
};

namespace detail {

// Internal per-thread exit hook invoked at the start of
// :func:`OpScope::~OpScope` before profiler recording.  Installed by
// :class:`compile::Tracer` while an :class:`OpScopeFull` has a pending trace
// node so an exception unwind pops it without a ``core/ -> compile/`` include.
using OpScopeExitHook = void (*)(const OpScope*) noexcept;
void set_op_scope_exit_hook(OpScopeExitHook fn) noexcept;

}  // namespace detail

}  // namespace lucid
