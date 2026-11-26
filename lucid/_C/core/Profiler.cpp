// lucid/_C/core/Profiler.cpp
//
// Implementation of the Profiler event sink and OpScope RAII recorder.
//
// The active Profiler for each thread is stored in a thread_local pointer
// (g_current).  OpScope reads this pointer at construction time and caches it
// in sink_; subsequent accesses during the scope's lifetime go directly to the
// cached pointer without a thread_local load, which avoids TLS overhead on the
// destructor hot path.
//
// All Profiler methods that touch the events_ vector hold mu_ to allow safe
// concurrent use from multiple threads (e.g. a DataLoader worker and the main
// training thread both writing into the same Profiler).

#include "Profiler.h"

#include "MemoryStats.h"

namespace lucid {

namespace {
// Per-thread active Profiler.  nullptr means "no profiling" for this thread.
thread_local Profiler* g_current = nullptr;
// Per-thread exit hook for active traced OpScopeFull frames.
thread_local detail::OpScopeExitHook g_exit_hook = nullptr;
}  // namespace

namespace detail {

void set_op_scope_exit_hook(OpScopeExitHook fn) noexcept {
    g_exit_hook = fn;
}

}  // namespace detail

Profiler* current_profiler() {
    return g_current;
}

void set_current_profiler(Profiler* p) {
    g_current = p;
}

void Profiler::start() {
    std::lock_guard<std::mutex> lock(mu_);
    active_ = true;
}

void Profiler::stop() {
    std::lock_guard<std::mutex> lock(mu_);
    active_ = false;
}

void Profiler::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    events_.clear();
}

// Returns a snapshot copy so callers can iterate without holding the mutex.
std::vector<OpEvent> Profiler::events() const {
    std::lock_guard<std::mutex> lock(mu_);
    return events_;
}

void Profiler::record(OpEvent event) {
    std::lock_guard<std::mutex> lock(mu_);
    // Re-check active_ under the lock: stop() may have raced with this call.
    if (active_)
        events_.push_back(std::move(event));
}

// Captures the start state.  sink_ is set to nullptr immediately if the
// profiler is not active, making the destructor a single branch with no
// memory reads.
OpScope::OpScope(std::string_view name, Device device, Dtype dtype, Shape shape)
    : sink_(g_current),
      event_{std::string(name), device, dtype, std::move(shape), 0, 0, 0},
      start_time_(std::chrono::steady_clock::now()),
      start_memory_bytes_(0) {
    if (sink_ && sink_->is_active()) {
        start_memory_bytes_ = MemoryTracker::get_stats(device).current_bytes;
    } else {
        // No active profiler — disable all recording for this scope to
        // avoid the MemoryTracker query on destruction.
        sink_ = nullptr;
    }
}

// Finalises timing and memory fields and forwards the event to the Profiler.
OpScope::~OpScope() {
    if (g_exit_hook)
        g_exit_hook(this);
    if (!sink_)
        return;
    const auto now = std::chrono::steady_clock::now();
    event_.time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time_).count();
    // memory_delta_bytes can be negative if the op freed more than it allocated
    // (e.g. a reshape that collapses a temporary buffer).
    const auto cur = MemoryTracker::get_stats(event_.device).current_bytes;
    event_.memory_delta_bytes =
        static_cast<std::int64_t>(cur) - static_cast<std::int64_t>(start_memory_bytes_);
    sink_->record(std::move(event_));
}

void OpScope::set_flops(std::int64_t f) {
    event_.flops = f;
}

}  // namespace lucid
